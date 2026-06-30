//===- BCIRConvertToLLVM.cpp - the -convert-bcir-to-llvm lowering -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRPassSupport.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <string>

using namespace mlir;

namespace bcir {

// Translate a GCC extended-asm operand-syntax template to LLVM-IR `$`-syntax. The cfront rail
// emits GCC templates (the C->clang path is correct for them), but the MLIR->LLVM path feeds the
// template straight to `llvm.inline_asm`, which uses LLVM-IR operand syntax. `llc` REJECTS the GCC
// forms (`%w1` reads as a literal register: "invalid register name"); the `$`-forms assemble. The
// bounded, cfront-scoped rule set (all confirmed against clang/llc):
//   `%%`              -> `%`              (literal percent)
//   `%<letter><N>`    -> `${N:<letter>}`  (size-modified operand, e.g. `%w1` -> `${1:w}`)
//   `%<N>`            -> `$N`             (bare operand, INCLUDING multi-digit: `%10` -> `$10`)
//   a literal `$`     -> `$$`             (LLVM reserves `$` for operands, so escape input `$`)
// EXOTIC GCC forms cfront never emits -- `%[name]` (named operands), `%P`/`%c`/`%a`/... (non-size
// operand modifiers), `%=` -- are REJECTED via `emitErr` rather than mistranslated. `emitErr` takes
// the offending fragment and returns failure (the caller wires it to `op.emitError`); a std::nullopt
// return means a diagnostic was already emitted.
//
// Standalone (no MLIR Op dependency beyond the error callback) so it is unit-testable: the result is
// a pure function of the input string + the rejection of exotic forms.
inline std::optional<std::string>
translateGccAsmTemplate(StringRef tmpl,
                        const std::function<LogicalResult(const Twine &)> &emitErr) {
  std::string out;
  out.reserve(tmpl.size());
  for (size_t i = 0, n = tmpl.size(); i < n;) {
    char c = tmpl[i];
    if (c == '$') {
      // Escape a literal `$` so LLVM does not read it as an operand reference.
      out += "$$";
      ++i;
      continue;
    }
    if (c != '%') {
      out += c;
      ++i;
      continue;
    }
    // c == '%': decode the GCC operand form.
    if (i + 1 >= n) {
      (void)emitErr("bcir.asm: unsupported GCC asm operand syntax (trailing '%')"
                    " (translate or use $-form)");
      return std::nullopt;
    }
    char d = tmpl[i + 1];
    if (d == '%') {
      // `%%` -> `%`.
      out += '%';
      i += 2;
      continue;
    }
    auto isDigit = [](char x) { return x >= '0' && x <= '9'; };
    auto isLetter = [](char x) {
      return (x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z');
    };
    if (isDigit(d)) {
      // `%<digits>` -> `$<digits>` (bare operand, multi-digit allowed).
      size_t j = i + 1;
      while (j < n && isDigit(tmpl[j]))
        ++j;
      out += '$';
      out += tmpl.substr(i + 1, j - (i + 1));
      i = j;
      continue;
    }
    if (isLetter(d)) {
      // A size modifier (`%w1`, `%b0`, `%k0`) MUST be `<single letter><digits>` -> `${digits:letter}`.
      // A letter not followed by a digit is an operand MODIFIER we cannot translate (`%P`, `%c`,
      // `%a`, ...) -- REJECT it rather than mistranslate.
      size_t j = i + 2;
      if (j >= n || !isDigit(tmpl[j])) {
        (void)emitErr(Twine("bcir.asm: unsupported GCC asm operand modifier '%") + Twine(d) +
                      "' (translate or use $-form)");
        return std::nullopt;
      }
      size_t k = j;
      while (k < n && isDigit(tmpl[k]))
        ++k;
      out += "${";
      out += tmpl.substr(j, k - j); // the operand index
      out += ':';
      out += d; // the size letter
      out += '}';
      i = k;
      continue;
    }
    // `%[name]` (named operand), `%=` (unique id), or any other exotic form -- REJECT.
    (void)emitErr(Twine("bcir.asm: unsupported GCC asm operand syntax '%") + Twine(d) +
                  "' (translate or use $-form)");
    return std::nullopt;
  }
  return out;
}

namespace {
struct ComputeOpLowering : public OpConversionPattern<ComputeOp> {
  using OpConversionPattern<ComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ComputeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "only binary compute is lowered");
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    StringRef kind = op.getKind();
    Value res;
    if (kind == "fadd")
      res = rewriter.create<LLVM::FAddOp>(op.getLoc(), lhs, rhs);
    else if (kind == "fsub")
      res = rewriter.create<LLVM::FSubOp>(op.getLoc(), lhs, rhs);
    else if (kind == "fmul")
      res = rewriter.create<LLVM::FMulOp>(op.getLoc(), lhs, rhs);
    else
      return rewriter.notifyMatchFailure(op, "unsupported compute kind");
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct BarrierOpLowering : public OpConversionPattern<BarrierOp> {
  using OpConversionPattern<BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Phase-8 memory model: the BCIR barrier ordering maps to the LLVM fence
    // ordering (default seq_cst). Fences require >= acquire, so unordered/
    // monotonic fall back to seq_cst.
    LLVM::AtomicOrdering ord = LLVM::AtomicOrdering::seq_cst;
    if (auto o = op.getOrdering()) {
      switch (*o) {
      case MemOrdering::Acquire: ord = LLVM::AtomicOrdering::acquire; break;
      case MemOrdering::Release: ord = LLVM::AtomicOrdering::release; break;
      case MemOrdering::AcqRel:  ord = LLVM::AtomicOrdering::acq_rel; break;
      default:                   ord = LLVM::AtomicOrdering::seq_cst; break;
      }
    }
    rewriter.replaceOpWithNewOp<LLVM::FenceOp>(op, ord, StringRef());
    return success();
  }
};

struct AsmOpLowering : public OpConversionPattern<AsmOp> {
  using OpConversionPattern<AsmOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AsmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // ASM1 -> llvm.inline_asm (LLVM's `call asm`). The single LLVM constraint string is the
    // out constraints (each already "=..."), then the in constraints, then each clobber rendered
    // as a "~{<clobber>}" entry -- ALL comma-joined (LLVM has no separate clobber field; clobbers
    // ARE "~{...}" constraint entries). E.g. outs ["=r"], ins ["r"], clobbers ["memory"] ->
    // "=r,r,~{memory}".
    SmallVector<std::string, 8> entries;
    for (Attribute a : op.getOutConstraints())
      entries.push_back(cast<StringAttr>(a).getValue().str());
    for (Attribute a : op.getInConstraints())
      entries.push_back(cast<StringAttr>(a).getValue().str());
    for (Attribute a : op.getClobbers())
      entries.push_back("~{" + cast<StringAttr>(a).getValue().str() + "}");
    std::string constraints;
    for (size_t i = 0; i < entries.size(); ++i) {
      if (i)
        constraints += ",";
      constraints += entries[i];
    }

    // Result type: SCOPE THE FIRST SLICE to 0 or 1 result. A >1-output asm makes LLVM return a
    // struct needing extractvalue unpacking -- a clean follow-on (SEG8.x), rejected here rather
    // than shipping a wrong lowering.
    unsigned numOut = op.getOutConstraints().size();
    if (numOut > 1)
      return op.emitError(
          "bcir.asm: multi-output inline asm lowering is a follow-on (SEG8.x)");

    // A read-write "+" output is NOT a write-only result: it ties an input via a matching
    // constraint. Lowering it as-is yields `llvm.inline_asm ... "+r,r" %x : (i32) -> i32`, which
    // opt/llc REJECT ("inline asm without outputs must return void", since "+" is read-write, not a
    // "=" output). The full "+" handling (passing the tied input + a matching constraint) is the same
    // follow-on as multi-output (SEG8.x); reject it cleanly here rather than ship invalid LLVM.
    for (Attribute a : op.getOutConstraints())
      if (cast<StringAttr>(a).getValue().starts_with("+"))
        return op.emitError(
            "bcir.asm: read-write '+' output lowering is a follow-on (SEG8.x)");

    Type resTy; // null -> void (no result)
    if (numOut == 1) {
      resTy = getTypeConverter()->convertType(op.getResult(0).getType());
      if (!resTy)
        return rewriter.notifyMatchFailure(op, "result type not convertible");
    }

    // The LLVM `call asm` operand list is the INPUT operands ONLY. A write-only "=" output is
    // NOT an asm-call argument -- it is the RESULT (the `$0` placeholder binds to the return
    // value). bcir.asm carries the output-lvalue SSA value as a leading operand (outputs-then-
    // inputs, mirroring the cfront read/write set), so the LLVM operands are args[numOut:] (the
    // inputs). (Read-write "+" outputs, which DO tie an input operand via a matching constraint,
    // are a follow-on alongside the multi-output struct unpack; the scoped first slice is the
    // write-only "=" case.)
    ValueRange inputs = adaptor.getArgs().drop_front(numOut);

    // asm is conservatively side-effecting (a volatile / "memory"-clobber form is an ordering
    // fence); never align-stack; default (AT&T) asm dialect; no per-operand attrs. Build via the
    // GENERIC attribute-list builder (TypeRange, ValueRange, ArrayRef<NamedAttribute>) rather than a
    // positional InlineAsmOp::build: the positional signature is NOT stable across LLVM versions
    // (LLVM >= 21 inserts a `tail_call_kind` parameter the LLVM-20 form lacks, so a fixed positional
    // call matches no overload on one of the two toolchains), whereas the generic builder is
    // identical on both. Only the attributes we set are present; is_align_stack / asm_dialect /
    // tail_call_kind stay absent -- Operation::create fills any default-valued attr -- so the result
    // is byte-identical to the old explicit form on LLVM-20 and well-formed on LLVM >= 21.
    SmallVector<Type, 1> resultTypes;
    if (resTy)
      resultTypes.push_back(resTy);

    // Translate the user's GCC operand syntax (`%0`, `%w1`, `%%`) to LLVM-IR `$`-syntax before
    // storing it as `asm_string` -- the MLIR->LLVM path feeds the template to `llvm.inline_asm` (LLVM
    // operand syntax), and `llc` rejects the GCC forms. Constraints are left ALONE: the existing
    // bcir.asm tests use plain `=r`/`r`, which allocate fine; no cfront c.asm path emits `=a`-style
    // register-class constraints (portio's qualified constraints are handled in PortioOpLowering).
    std::optional<std::string> translated = translateGccAsmTemplate(
        op.getAsmTemplate(),
        [&](const Twine &msg) { return op.emitError(msg); });
    if (!translated)
      return failure();
    SmallVector<NamedAttribute, 3> asmAttrs = {
        rewriter.getNamedAttr("asm_string",
                              rewriter.getStringAttr(*translated)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr(constraints)),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    auto newOp = rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), TypeRange(resultTypes), inputs, asmAttrs);

    if (numOut == 0)
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, newOp.getRes());
    return success();
  }
};

struct PortioOpLowering : public OpConversionPattern<PortioOp> {
  using OpConversionPattern<PortioOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PortioOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // ASM2 -> llvm.inline_asm (the x86 `in`/`out` instruction behind a volatile `call asm`). The
    // template + constraint strings are the LLVM-IR-correct forms (VERIFIED to assemble through
    // `llc`): the accumulator is `${0:b}`/`${0:w}`/`${0:k}` (al/ax/eax) and the port is `${1:w}` (the
    // 16-bit dx). The cfront rail's `_PORTIO_IN_ASM`/`_PORTIO_OUT_ASM` carry the GCC spellings
    // (`%w1`/`%b0`), which are correct on the C->clang rail but which `llc` rejects on this MLIR->LLVM
    // path ("invalid register name" / "couldn't allocate output register"); clang's frontend does the
    // `%b0`->`${0:b}` / `=a`->`={ax}` translation, and the MLIR rail (no frontend) emits the qualified
    // forms directly. A READ writes the accumulator (constraint "={ax},N{dx}", one result); a WRITE
    // reads it (constraint "{ax},N{dx}", no result). The single LLVM constraint string is the
    // output(s) then the input(s), comma-joined -- exactly as AsmOpLowering builds it.
    int64_t width = static_cast<int64_t>(op.getWidth());
    bool isIn = op.getDirection() == PortDir::In;

    // The 6 templates, keyed (direction, width), in LLVM-IR operand syntax: the size modifier is
    // byte/word/long (b/w/k) on the accumulator operand `${0:...}`, `${1:w}` forces the port to 16-bit
    // dx. These assemble to `inb %dx, %al` / `outb %al, %dx` etc. (confirmed via llc -filetype=asm).
    StringRef asmTemplate;
    if (isIn) {
      asmTemplate = (width == 8)    ? "inb ${1:w}, ${0:b}"
                    : (width == 16) ? "inw ${1:w}, ${0:w}"
                                    : "inl ${1:w}, ${0:k}";
    } else {
      asmTemplate = (width == 8)    ? "outb ${0:b}, ${1:w}"
                    : (width == 16) ? "outw ${0:w}, ${1:w}"
                                    : "outl ${0:k}, ${1:w}";
    }
    // Constraint string: in -> "={ax},N{dx}" (output "={ax}", then input "N{dx}"); out ->
    // "{ax},N{dx}" (two inputs, no output). The fully-qualified register names are what LLVM's x86
    // backend needs: the GCC `=a`/`Nd` short forms fail register allocation ("couldn't allocate
    // output register for constraint 'a'"); clang's frontend does this `=a`->`={ax}`, `Nd`->`N{dx}`
    // translation, which the frontend-less MLIR rail must emit directly.
    StringRef constraints = isIn ? "={ax},N{dx}" : "{ax},N{dx}";

    // Result type: an `in` returns the type-converted i{width}; an `out` is void (no result). The LLVM
    // `call asm` operand list is the type-converted operands in order (in: just the port; out: value
    // then port) -- a write-only "=a" output is the RESULT, never an asm-call argument, so the in form
    // passes ONLY the port (args[0]).
    SmallVector<Type, 1> resultTypes;
    ValueRange callOperands;
    if (isIn) {
      Type resTy = getTypeConverter()->convertType(op.getResult(0).getType());
      if (!resTy)
        return rewriter.notifyMatchFailure(op, "result type not convertible");
      resultTypes.push_back(resTy);
      callOperands = adaptor.getArgs().take_front(1); // the port only
    } else {
      callOperands = adaptor.getArgs(); // value, then port
    }

    // Build via the GENERIC attribute-list builder (TypeRange, ValueRange, ArrayRef<NamedAttribute>),
    // NOT a positional InlineAsmOp::build -- the positional signature is NOT stable across LLVM versions
    // (LLVM >= 21 inserts a `tail_call_kind` parameter the LLVM-20 form lacks), whereas the generic
    // builder is identical on both. has_side_effects is ALWAYS set (port I/O is volatile / ordered, like
    // the cfront `__volatile__`); is_align_stack / asm_dialect / tail_call_kind stay absent (default).
    // This is copied verbatim from AsmOpLowering.
    SmallVector<NamedAttribute, 3> asmAttrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr(asmTemplate)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr(constraints)),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    auto newOp = rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), TypeRange(resultTypes), callOperands, asmAttrs);

    if (isIn)
      rewriter.replaceOp(op, newOp.getRes());
    else
      rewriter.eraseOp(op);
    return success();
  }
};

struct VolatileLoadOpLowering : public OpConversionPattern<VolatileLoadOp> {
  using OpConversionPattern<VolatileLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VolatileLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // First-class MMIO read -> llvm.inttoptr(addr) + a VOLATILE llvm.load. Mirrors the cfront emit
    // `*(volatile T *)(intaddr)`. `volatile` is set via the generated setter (not a positional builder
    // arg) so it is stable across LLVM versions (the LoadOp builder gained params over 20->22).
    Type resTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!resTy)
      return rewriter.notifyMatchFailure(op, "result type not convertible");
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value ptr =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), ptrTy, adaptor.getAddr());
    auto load = rewriter.create<LLVM::LoadOp>(op.getLoc(), resTy, ptr);
    load.setVolatile_(true);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct VolatileStoreOpLowering : public OpConversionPattern<VolatileStoreOp> {
  using OpConversionPattern<VolatileStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VolatileStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // First-class MMIO write -> llvm.inttoptr(addr) + a VOLATILE llvm.store. The void counterpart of
    // VolatileLoadOpLowering (`*(volatile T *)(intaddr) = value`); `volatile` set via the setter.
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value ptr =
        rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), ptrTy, adaptor.getAddr());
    auto store =
        rewriter.create<LLVM::StoreOp>(op.getLoc(), adaptor.getValue(), ptr);
    store.setVolatile_(true);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CRegReadOpLowering : public OpConversionPattern<CRegReadOp> {
  using OpConversionPattern<CRegReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CRegReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Control-register read -> llvm.inline_asm "mov %<reg>, $0", "=r,~{memory}". LLVM-IR inline-asm syntax
    // ($0 operand, single-% register) -- the exact form clang emits; built via the same generic
    // attribute-list InlineAsmOp builder as AsmOpLowering (version-stable). has_side_effects always.
    // The `~{memory}` clobber is intentional even for a READ: a control register reflects the live memory
    // system (CR2 = the faulting address, CR3 = the live page-table base, CR0/CR4 = paging/protection), so
    // the read is an ORDERING POINT -- without the clobber LLVM could sink an unrelated load below a CR2
    // read in a fault handler. (`has_side_effects` blocks elision/reordering vs other side-effecting asm,
    // but NOT vs ordinary memory ops; the `~{memory}` clobber adds that.) Mirrors creg_write's clobber.
    Type resTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!resTy)
      return rewriter.notifyMatchFailure(op, "result type not convertible");
    std::string tmpl =
        (Twine("mov %") + stringifyCtrlReg(op.getReg()) + ", $0").str();
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr(tmpl)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("=r,~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    auto newOp = rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), TypeRange{resTy}, ValueRange{}, attrs);
    rewriter.replaceOp(op, newOp.getRes());
    return success();
  }
};

struct CRegWriteOpLowering : public OpConversionPattern<CRegWriteOp> {
  using OpConversionPattern<CRegWriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CRegWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Control-register write -> llvm.inline_asm "mov $0, %<reg>", "r,~{memory}" (a CR write reloads page
    // tables / toggles paging+protection -- a system-wide memory effect, hence the ~{memory} clobber);
    // no result. has_side_effects always.
    std::string tmpl =
        (Twine("mov $0, %") + stringifyCtrlReg(op.getReg())).str();
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr(tmpl)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("r,~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    rewriter.create<LLVM::InlineAsmOp>(op.getLoc(), TypeRange{},
                                       ValueRange{adaptor.getValue()}, attrs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MsrReadOpLowering : public OpConversionPattern<MsrReadOp> {
  using OpConversionPattern<MsrReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MsrReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // MSR read -> `rdmsr` (no operand placeholders -- the registers are fixed by the constraints): the
    // index is bound to ECX (`{cx}`) and the two 32-bit result halves come back in EAX/EDX
    // (`={ax},={dx}`), so the inline_asm yields an `!llvm.struct<(i32, i32)>`. has_side_effects always;
    // the `~{memory}` clobber makes the read an ordering point (an MSR is live CPU state -- see the op
    // doc). The i64 surface value is REASSEMBLED here as `(zext hi) << 32 | zext lo`, so callers see a
    // clean 64-bit register (the EDX:EAX split is an ISA detail of `rdmsr`, not part of the op contract).
    Location loc = op.getLoc();
    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    Type pairTy = LLVM::LLVMStructType::getLiteral(getContext(), {i32Ty, i32Ty});
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr("rdmsr")),
        rewriter.getNamedAttr("constraints",
                              rewriter.getStringAttr("={ax},={dx},{cx},~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        loc, TypeRange{pairTy}, ValueRange{adaptor.getIndex()}, attrs);
    Value lo = rewriter.create<LLVM::ExtractValueOp>(loc, asmOp.getRes(), 0);
    Value hi = rewriter.create<LLVM::ExtractValueOp>(loc, asmOp.getRes(), 1);
    Value lo64 = rewriter.create<LLVM::ZExtOp>(loc, i64Ty, lo);
    Value hi64 = rewriter.create<LLVM::ZExtOp>(loc, i64Ty, hi);
    Value c32 = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(32));
    Value hiShifted = rewriter.create<LLVM::ShlOp>(loc, hi64, c32);
    Value full = rewriter.create<LLVM::OrOp>(loc, hiShifted, lo64);
    rewriter.replaceOp(op, full);
    return success();
  }
};

struct MsrWriteOpLowering : public OpConversionPattern<MsrWriteOp> {
  using OpConversionPattern<MsrWriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MsrWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // MSR write -> `wrmsr` (index in ECX, the 64-bit value split EDX:EAX): the i64 is split into its low
    // (`trunc`) and high (`trunc (value >> 32)`) 32-bit halves, then `llvm.inline_asm "wrmsr",
    // "{cx},{ax},{dx},~{memory}"` consumes (index, low, high); no result. The `~{memory}` clobber is
    // mandatory -- a wrmsr can toggle long mode (IA32_EFER) or relocate the APIC (IA32_APIC_BASE), a
    // system-wide memory effect. has_side_effects always.
    Location loc = op.getLoc();
    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    Value val = adaptor.getValue();
    Value lo = rewriter.create<LLVM::TruncOp>(loc, i32Ty, val);
    Value c32 = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(32));
    Value hiShifted = rewriter.create<LLVM::LShrOp>(loc, val, c32);
    Value hi = rewriter.create<LLVM::TruncOp>(loc, i32Ty, hiShifted);
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr("wrmsr")),
        rewriter.getNamedAttr("constraints",
                              rewriter.getStringAttr("{cx},{ax},{dx},~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    rewriter.create<LLVM::InlineAsmOp>(
        loc, TypeRange{}, ValueRange{adaptor.getIndex(), lo, hi}, attrs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertToLLVMPass
    : public PassWrapper<ConvertToLLVMPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToLLVMPass)

  StringRef getArgument() const final { return "convert-bcir-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower BCIR compute/barrier to the LLVM dialect (partial).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<ComputeOp, BarrierOp, AsmOp, PortioOp, VolatileLoadOp,
                        VolatileStoreOp, CRegReadOp, CRegWriteOp, MsrReadOp,
                        MsrWriteOp>();

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<ComputeOpLowering, BarrierOpLowering, AsmOpLowering,
                 PortioOpLowering, VolatileLoadOpLowering,
                 VolatileStoreOpLowering, CRegReadOpLowering,
                 CRegWriteOpLowering, MsrReadOpLowering, MsrWriteOpLowering>(
        typeConverter, &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}

}  // namespace bcir
