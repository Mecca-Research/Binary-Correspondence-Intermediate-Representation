//===- BCIRConvertToLLVM.cpp - the -convert-bcir-to-llvm lowering -*- C++ -*-===//
//
// Part of the modular BCIR MLIR pass library (split out of the former monolithic
// BCIRPasses.cpp). Shared helpers live in BCIRPassSupport.h; registration in
// BCIRPasses.cpp. C++23.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRPasses.h"
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
#include "llvm/Support/raw_ostream.h"

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
    auto isLetter = [](char x) { return (x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z'); };
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

// Append one independently-authored module-assembly fragment without replacing fragments installed by
// another lowering. The LLVM dialect deliberately represents `llvm.module_asm` as an array of strings;
// preserving that structure keeps independently lowered entry/trampoline symbols deterministic without
// merging their text. A non-array pre-existing attribute is malformed and is rejected before mutating it.
static LogicalResult appendModuleAsm(Operation *source, StringRef assembly,
                                     PatternRewriter &rewriter) {
  auto module = source->getParentOfType<::mlir::ModuleOp>();
  if (!module)
    return source->emitError("x86 top-level assembly edge must be nested in a builtin module");
  StringRef attrName = LLVM::LLVMDialect::getModuleLevelAsmAttrName();
  SmallVector<Attribute, 4> fragments;
  if (Attribute current = module->getAttr(attrName)) {
    auto array = dyn_cast<ArrayAttr>(current);
    if (!array)
      return source->emitError("existing llvm.module_asm attribute is not an array");
    for (Attribute fragment : array)
      fragments.push_back(fragment);
  }
  fragments.push_back(rewriter.getStringAttr(assembly));
  rewriter.modifyOpInPlace(module,
                           [&] { module->setAttr(attrName, rewriter.getArrayAttr(fragments)); });
  return success();
}

// Intel-defined exception vectors for which the processor pushes an error code before entering the
// handler. Keeping this architectural fact in the lowering removes a caller-controlled boolean that
// could shift the entire saved frame and turn iretq into a corrupted return. Reserved vectors not in
// this list use the ordinary no-error-code shape.
static bool x86VectorHasErrorCode(int64_t vector) {
  switch (vector) {
  case 8:  // #DF
  case 10: // #TS
  case 11: // #NP
  case 12: // #SS
  case 13: // #GP
  case 14: // #PF
  case 17: // #AC
  case 21: // #CP
  case 29: // #VC
  case 30: // #SX
    return true;
  default:
    return false;
  }
}

struct EntryOpLowering : public OpRewritePattern<EntryOp> {
  using OpRewritePattern<EntryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(EntryOp op, PatternRewriter &rewriter) const override {
    std::string assembly;
    llvm::raw_string_ostream out(assembly);
    StringRef entry = op.getSymName();
    out << ".text\n"
        << ".p2align 4\n"
        << ".globl " << entry << "\n"
        << ".type " << entry << ",@function\n"
        << entry
        << ":\n"
        // Do not permit a maskable interrupt to observe the old/partially switched
        // stack. Reset starts with IF clear, but this long-mode handoff is also usable
        // after firmware or an earlier stage, so establish the invariant explicitly.
        << "  cli\n"
        << "  leaq " << op.getStackTop() << "(%rip), %rsp\n"
        << "  andq $-16, %rsp\n"
        // A normal SysV x86-64 callee observes RSP == 8 (mod 16) after CALL has pushed a return
        // address. This edge tail-jumps, so push a zero sentinel for the missing slot explicitly. The
        // target is required to be noreturn; an accidental return deterministically faults at zero.
        << "  pushq $0\n"
        << "  xorq %rbp, %rbp\n"
        << "  cld\n"
        << "  jmp " << op.getCEntry() << "\n"
        << ".size " << entry << ", .-" << entry << "\n";
    out.flush();
    if (failed(appendModuleAsm(op, assembly, rewriter)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct DescriptorLoadOpLowering : public OpConversionPattern<DescriptorLoadOp> {
  using OpConversionPattern<DescriptorLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(DescriptorLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef instruction = op.getTable() == DescriptorTable::GDT ? "lgdt ($0)" : "lidt ($0)";
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr(instruction)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("r,~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    LLVM::InlineAsmOp::create(rewriter, op.getLoc(), TypeRange{}, ValueRange{adaptor.getAddr()},
                              attrs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct TaskRegisterLoadOpLowering : public OpConversionPattern<TaskRegisterLoadOp> {
  using OpConversionPattern<TaskRegisterLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TaskRegisterLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr("ltr ${0:w}")),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("r,~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    LLVM::InlineAsmOp::create(rewriter, op.getLoc(), TypeRange{}, ValueRange{adaptor.getSelector()},
                              attrs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct SegmentReloadOpLowering : public OpConversionPattern<SegmentReloadOp> {
  using OpConversionPattern<SegmentReloadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SegmentReloadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // A far return is the long-mode mechanism for reloading CS. The `${:uid}` local-label suffix is
    // LLVM's inline-asm uniqueness escape, so cloning/inlining this op cannot create duplicate labels.
    Value code64 = LLVM::ZExtOp::create(rewriter, op.getLoc(), rewriter.getI64Type(),
                                        adaptor.getCodeSelector());
    StringRef assembly = "movw ${0:w}, %ds; movw ${0:w}, %es; movw ${0:w}, %ss; "
                         "pushq ${1:q}; leaq .Lbcir_seg_${:uid}(%rip), %rax; pushq %rax; "
                         "lretq; .Lbcir_seg_${:uid}:";
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr(assembly)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("r,r,~{rax},~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    LLVM::InlineAsmOp::create(rewriter, op.getLoc(), TypeRange{},
                              ValueRange{adaptor.getDataSelector(), code64}, attrs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct InterruptTrampolineOpLowering : public OpRewritePattern<InterruptTrampolineOp> {
  using OpRewritePattern<InterruptTrampolineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InterruptTrampolineOp op,
                                PatternRewriter &rewriter) const override {
    int64_t vector = static_cast<int32_t>(op.getVector());
    StringRef symbol = op.getSymName();
    std::string assembly;
    llvm::raw_string_ostream out(assembly);
    out << ".text\n"
        << ".p2align 4\n"
        << ".globl " << symbol << "\n"
        << ".type " << symbol << ",@function\n"
        << symbol
        << ":\n"
        // The hardware frame retains the interrupted IF value for iretq. Clear
        // the live IF here as well so a trap-gate descriptor cannot admit a
        // maskable interrupt while the normalized/private frame is incomplete.
        << "  cli\n";

    // Normalize the CPU frame. Hardware-error vectors already enter with `error,rip,cs,rflags`; all
    // other vectors receive a synthetic zero error. Pushing the vector then gives one stable shape.
    if (!x86VectorHasErrorCode(vector))
      out << "  pushq $0\n";
    out << "  pushq $" << vector << "\n";

    // Save every SysV integer GPR. The resulting fixed AMD64 long-mode frame begins
    // r15,r14,...,rax,vector,error,rip,cs,rflags,rsp,ss, and its address is the sole C argument in
    // RDI. R13 is callee-saved by SysV;
    // after saving its interrupted value, use it as immutable trampoline-private state recording the
    // original privilege class. Only ring 0 and ring 3 are supported. The exit checks that the handler
    // did not change this class: even though IRETQ always consumes SS:RSP in long mode, changing class
    // would violate the selector validation, return-protection, and SWAPGS policy established on entry.
    out << "  pushq %rax\n"
        << "  pushq %rbx\n"
        << "  pushq %rcx\n"
        << "  pushq %rdx\n"
        << "  pushq %rbp\n"
        << "  pushq %rsi\n"
        << "  pushq %rdi\n"
        << "  pushq %r8\n"
        << "  pushq %r9\n"
        << "  pushq %r10\n"
        << "  pushq %r11\n"
        << "  pushq %r12\n"
        << "  pushq %r13\n"
        << "  pushq %r14\n"
        << "  pushq %r15\n"
        << "  movq 144(%rsp), %r13\n"
        << "  andq $3, %r13\n"
        << "  testq %r13, %r13\n"
        << "  jz 1f\n"
        << "  cmpq $3, %r13\n"
        << "  jne 9f\n";
    if (op.getSwapgsOnUser())
      out << "  swapgs\n";
    out << "1:\n";
    if (op.getSwapgsOnUser())
      out
          // Fence both sides of the conditional decision before a C handler may
          // dereference GS-relative per-CPU state (Spectre-v1 swapgs boundary).
          << "  lfence\n";
    out << "  movq %rsp, %rdi\n"
        << "  movq %rsp, %r12\n"
        << "  andq $-16, %rsp\n"
        << "  cld\n"
        << "  call " << op.getHandler() << "\n"
        << "  movq %r12, %rsp\n"
        // The C body may edit a selector while preserving its RPL, but it must
        // not change the ring-0/ring-3 frame class owned by the trampoline.
        << "  movq 144(%rsp), %rax\n"
        << "  andq $3, %rax\n"
        << "  cmpq %r13, %rax\n"
        << "  jne 9f\n";
    if (op.getSwapgsOnUser())
      out << "  testq %r13, %r13\n"
          << "  jz 2f\n"
          << "  swapgs\n"
          << "2:\n";
    out << "  popq %r15\n"
        << "  popq %r14\n"
        << "  popq %r13\n"
        << "  popq %r12\n"
        << "  popq %r11\n"
        << "  popq %r10\n"
        << "  popq %r9\n"
        << "  popq %r8\n"
        << "  popq %rdi\n"
        << "  popq %rsi\n"
        << "  popq %rbp\n"
        << "  popq %rdx\n"
        << "  popq %rcx\n"
        << "  popq %rbx\n"
        << "  popq %rax\n";
    out << "  addq $16, %rsp\n"
        << "  iretq\n"
        << "9:\n"
        // Fail closed before reinterpreting a malformed privilege-transition
        // frame. The resident kernel's #UD/paranoid policy owns diagnosis.
        << "  ud2\n"
        << ".size " << symbol << ", .-" << symbol << "\n";
    out.flush();
    if (failed(appendModuleAsm(op, assembly, rewriter)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct ComputeOpLowering : public OpConversionPattern<ComputeOp> {
  using OpConversionPattern<ComputeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ComputeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "only binary compute is lowered");
    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    StringRef kind = op.getKind();
    Value res;
    if (kind == "fadd")
      res = LLVM::FAddOp::create(rewriter, op.getLoc(), lhs, rhs);
    else if (kind == "fsub")
      res = LLVM::FSubOp::create(rewriter, op.getLoc(), lhs, rhs);
    else if (kind == "fmul")
      res = LLVM::FMulOp::create(rewriter, op.getLoc(), lhs, rhs);
    else
      return rewriter.notifyMatchFailure(op, "unsupported compute kind");
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct BarrierOpLowering : public OpConversionPattern<BarrierOp> {
  using OpConversionPattern<BarrierOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Phase-8 memory model: the BCIR barrier ordering maps to the LLVM fence
    // ordering (default seq_cst). Fences require >= acquire, so unordered/
    // monotonic fall back to seq_cst.
    LLVM::AtomicOrdering ord = LLVM::AtomicOrdering::seq_cst;
    if (auto o = op.getOrdering()) {
      switch (*o) {
      case MemOrdering::Acquire:
        ord = LLVM::AtomicOrdering::acquire;
        break;
      case MemOrdering::Release:
        ord = LLVM::AtomicOrdering::release;
        break;
      case MemOrdering::AcqRel:
        ord = LLVM::AtomicOrdering::acq_rel;
        break;
      default:
        ord = LLVM::AtomicOrdering::seq_cst;
        break;
      }
    }
    rewriter.replaceOpWithNewOp<LLVM::FenceOp>(op, ord, StringRef());
    return success();
  }
};

struct AsmOpLowering : public OpConversionPattern<AsmOp> {
  using OpConversionPattern<AsmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AsmOp op, OpAdaptor adaptor,
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
      return op.emitError("bcir.asm: multi-output inline asm lowering is a follow-on (SEG8.x)");

    // A read-write "+" output is NOT a write-only result: it ties an input via a matching
    // constraint. Lowering it as-is yields `llvm.inline_asm ... "+r,r" %x : (i32) -> i32`, which
    // opt/llc REJECT ("inline asm without outputs must return void", since "+" is read-write, not a
    // "=" output). The full "+" handling (passing the tied input + a matching constraint) is the same
    // follow-on as multi-output (SEG8.x); reject it cleanly here rather than ship invalid LLVM.
    for (Attribute a : op.getOutConstraints())
      if (cast<StringAttr>(a).getValue().starts_with("+"))
        return op.emitError("bcir.asm: read-write '+' output lowering is a follow-on (SEG8.x)");

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
        op.getAsmTemplate(), [&](const Twine &msg) { return op.emitError(msg); });
    if (!translated)
      return failure();
    SmallVector<NamedAttribute, 3> asmAttrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr(*translated)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr(constraints)),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    auto newOp =
        LLVM::InlineAsmOp::create(rewriter, op.getLoc(), TypeRange(resultTypes), inputs, asmAttrs);

    if (numOut == 0)
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, newOp.getRes());
    return success();
  }
};

struct PortioOpLowering : public OpConversionPattern<PortioOp> {
  using OpConversionPattern<PortioOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PortioOp op, OpAdaptor adaptor,
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
    auto newOp = LLVM::InlineAsmOp::create(rewriter, op.getLoc(), TypeRange(resultTypes),
                                           callOperands, asmAttrs);

    if (isIn)
      rewriter.replaceOp(op, newOp.getRes());
    else
      rewriter.eraseOp(op);
    return success();
  }
};

struct VolatileLoadOpLowering : public OpConversionPattern<VolatileLoadOp> {
  using OpConversionPattern<VolatileLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(VolatileLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // First-class MMIO read -> llvm.inttoptr(addr) + a VOLATILE llvm.load. Mirrors the cfront emit
    // `*(volatile T *)(intaddr)`. `volatile` is set via the generated setter (not a positional builder
    // arg) so it is stable across LLVM versions (the LoadOp builder gained params over 20->22).
    Type resTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!resTy)
      return rewriter.notifyMatchFailure(op, "result type not convertible");
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value ptr = LLVM::IntToPtrOp::create(rewriter, op.getLoc(), ptrTy, adaptor.getAddr());
    auto load = LLVM::LoadOp::create(rewriter, op.getLoc(), resTy, ptr);
    load.setVolatile_(true);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct VolatileStoreOpLowering : public OpConversionPattern<VolatileStoreOp> {
  using OpConversionPattern<VolatileStoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(VolatileStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // First-class MMIO write -> llvm.inttoptr(addr) + a VOLATILE llvm.store. The void counterpart of
    // VolatileLoadOpLowering (`*(volatile T *)(intaddr) = value`); `volatile` set via the setter.
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value ptr = LLVM::IntToPtrOp::create(rewriter, op.getLoc(), ptrTy, adaptor.getAddr());
    auto store = LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(), ptr);
    store.setVolatile_(true);
    rewriter.eraseOp(op);
    return success();
  }
};

// Maps the optional #bcir.mem_ordering to the LLVM atomic ordering for an RMW/CAS access:
// ABSENT = seq_cst (the C11/`__sync` default, the bcir.barrier precedent); unordered/monotonic
// clamp to monotonic (LLVM requires an atomic access ordering >= monotonic).
static LLVM::AtomicOrdering atomicOrdering(std::optional<MemOrdering> o) {
  if (!o)
    return LLVM::AtomicOrdering::seq_cst;
  switch (*o) {
  case MemOrdering::Unordered:
  case MemOrdering::Monotonic:
    return LLVM::AtomicOrdering::monotonic;
  case MemOrdering::Acquire:
    return LLVM::AtomicOrdering::acquire;
  case MemOrdering::Release:
    return LLVM::AtomicOrdering::release;
  case MemOrdering::AcqRel:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    return LLVM::AtomicOrdering::seq_cst;
  }
}

// The LLVM strongest-failure-ordering rule for cmpxchg: the failure ordering is never a release
// and never stronger than success (release -> monotonic, acq_rel -> acquire, else success).
static LLVM::AtomicOrdering casFailureOrdering(LLVM::AtomicOrdering success) {
  switch (success) {
  case LLVM::AtomicOrdering::release:
    return LLVM::AtomicOrdering::monotonic;
  case LLVM::AtomicOrdering::acq_rel:
    return LLVM::AtomicOrdering::acquire;
  default:
    return success;
  }
}

struct AtomicRMWOpLowering : public OpConversionPattern<AtomicRMWOp> {
  using OpConversionPattern<AtomicRMWOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AtomicRMWOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // First-class atomic RMW -> llvm.inttoptr(addr) + llvm.atomicrmw <kind>, the structural twin
    // of the cfront `c.atomic.*` / `c.c11atom.fetch_*` lowering (`__atomic_fetch_*` semantics).
    LLVM::AtomicBinOp bin;
    StringRef k = op.getKind();
    if (k == "add")
      bin = LLVM::AtomicBinOp::add;
    else if (k == "sub")
      bin = LLVM::AtomicBinOp::sub;
    else if (k == "xor")
      bin = LLVM::AtomicBinOp::_xor;
    else if (k == "exchange")
      bin = LLVM::AtomicBinOp::xchg;
    else
      return rewriter.notifyMatchFailure(op, "unknown atomic_rmw kind");
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value ptr = LLVM::IntToPtrOp::create(rewriter, op.getLoc(), ptrTy, adaptor.getAddr());
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(op, bin, ptr, adaptor.getValue(),
                                                   atomicOrdering(op.getOrdering()));
    return success();
  }
};

struct AtomicCASOpLowering : public OpConversionPattern<AtomicCASOp> {
  using OpConversionPattern<AtomicCASOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AtomicCASOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // First-class CAS -> llvm.inttoptr(addr) + llvm.cmpxchg (+ weak) + extractvalue[0] (the OLD
    // value, `c.cmpxchg.val` semantics). `weak` set via the setter (version-stable, like the
    // volatile setters in the MMIO lowerings).
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value ptr = LLVM::IntToPtrOp::create(rewriter, op.getLoc(), ptrTy, adaptor.getAddr());
    LLVM::AtomicOrdering succ = atomicOrdering(op.getOrdering());
    auto cas = LLVM::AtomicCmpXchgOp::create(rewriter, op.getLoc(), ptr, adaptor.getExpected(),
                                             adaptor.getDesired(), succ, casFailureOrdering(succ));
    if (op.getWeak())
      cas.setWeak(true);
    Value old =
        LLVM::ExtractValueOp::create(rewriter, op.getLoc(), cas.getRes(), ArrayRef<int64_t>{0});
    rewriter.replaceOp(op, old);
    return success();
  }
};

struct CRegReadOpLowering : public OpConversionPattern<CRegReadOp> {
  using OpConversionPattern<CRegReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(CRegReadOp op, OpAdaptor adaptor,
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
    std::string tmpl = (Twine("mov %") + stringifyCtrlReg(op.getReg()) + ", $0").str();
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr(tmpl)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("=r,~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    auto newOp =
        LLVM::InlineAsmOp::create(rewriter, op.getLoc(), TypeRange{resTy}, ValueRange{}, attrs);
    rewriter.replaceOp(op, newOp.getRes());
    return success();
  }
};

struct CRegWriteOpLowering : public OpConversionPattern<CRegWriteOp> {
  using OpConversionPattern<CRegWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(CRegWriteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Control-register write -> llvm.inline_asm "mov $0, %<reg>", "r,~{memory}" (a CR write reloads page
    // tables / toggles paging+protection -- a system-wide memory effect, hence the ~{memory} clobber);
    // no result. has_side_effects always.
    std::string tmpl = (Twine("mov $0, %") + stringifyCtrlReg(op.getReg())).str();
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr(tmpl)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("r,~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    LLVM::InlineAsmOp::create(rewriter, op.getLoc(), TypeRange{}, ValueRange{adaptor.getValue()},
                              attrs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MsrReadOpLowering : public OpConversionPattern<MsrReadOp> {
  using OpConversionPattern<MsrReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MsrReadOp op, OpAdaptor adaptor,
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
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("={ax},={dx},{cx},~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    auto asmOp = LLVM::InlineAsmOp::create(rewriter, loc, TypeRange{pairTy},
                                           ValueRange{adaptor.getIndex()}, attrs);
    Value lo = LLVM::ExtractValueOp::create(rewriter, loc, asmOp.getRes(), 0);
    Value hi = LLVM::ExtractValueOp::create(rewriter, loc, asmOp.getRes(), 1);
    Value lo64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, lo);
    Value hi64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, hi);
    Value c32 = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(32));
    Value hiShifted = LLVM::ShlOp::create(rewriter, loc, hi64, c32);
    Value full = LLVM::OrOp::create(rewriter, loc, hiShifted, lo64);
    rewriter.replaceOp(op, full);
    return success();
  }
};

struct MsrWriteOpLowering : public OpConversionPattern<MsrWriteOp> {
  using OpConversionPattern<MsrWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MsrWriteOp op, OpAdaptor adaptor,
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
    Value lo = LLVM::TruncOp::create(rewriter, loc, i32Ty, val);
    Value c32 = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(32));
    Value hiShifted = LLVM::LShrOp::create(rewriter, loc, val, c32);
    Value hi = LLVM::TruncOp::create(rewriter, loc, i32Ty, hiShifted);
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr("wrmsr")),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("{cx},{ax},{dx},~{memory}")),
        rewriter.getNamedAttr("has_side_effects", rewriter.getUnitAttr()),
    };
    LLVM::InlineAsmOp::create(rewriter, loc, TypeRange{}, ValueRange{adaptor.getIndex(), lo, hi},
                              attrs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertToLLVMPass : public PassWrapper<ConvertToLLVMPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToLLVMPass)

  StringRef getArgument() const final { return "convert-bcir-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower the supported BCIR compute, memory, and x86 asm-edge subset to LLVM.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // Module-level assembly mutates the containing builtin module while erasing the source symbol.
    // Perform that ancestor mutation in the ordinary rewrite driver before dialect conversion; the
    // conversion rewriter intentionally scopes transactional edits to the matched operation tree.
    RewritePatternSet moduleAsmPatterns(&getContext());
    moduleAsmPatterns.add<EntryOpLowering, InterruptTrampolineOpLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(moduleAsmPatterns)))) {
      signalPassFailure();
      return;
    }

    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<EntryOp, DescriptorLoadOp, TaskRegisterLoadOp, SegmentReloadOp,
                        InterruptTrampolineOp, ComputeOp, BarrierOp, AsmOp, PortioOp,
                        VolatileLoadOp, VolatileStoreOp, AtomicRMWOp, AtomicCASOp, CRegReadOp,
                        CRegWriteOp, MsrReadOp, MsrWriteOp>();

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<DescriptorLoadOpLowering, TaskRegisterLoadOpLowering, SegmentReloadOpLowering,
                 ComputeOpLowering, BarrierOpLowering, AsmOpLowering, PortioOpLowering,
                 VolatileLoadOpLowering, VolatileStoreOpLowering, AtomicRMWOpLowering,
                 AtomicCASOpLowering, CRegReadOpLowering, CRegWriteOpLowering, MsrReadOpLowering,
                 MsrWriteOpLowering>(typeConverter, &getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}

} // namespace bcir
