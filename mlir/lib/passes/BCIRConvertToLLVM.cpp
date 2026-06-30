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
    SmallVector<NamedAttribute, 3> asmAttrs = {
        rewriter.getNamedAttr("asm_string",
                              rewriter.getStringAttr(op.getAsmTemplate())),
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
    // template strings + constraint strings are BYTE-IDENTICAL to the cfront rail's _PORTIO_IN_ASM /
    // _PORTIO_OUT_ASM (bcir/frontends/cfront/emit.py): the accumulator is `%b0/%w0/%k0` (al/ax/eax)
    // and the port is `%w1` (the 16-bit dx, the `Nd` immediate-or-dx constraint). A READ writes the
    // accumulator (constraint "=a,Nd", one result); a WRITE reads it (constraint "a,Nd", no result).
    // The single LLVM constraint string is the output(s) then the input(s), comma-joined -- exactly
    // as AsmOpLowering builds it.
    int64_t width = static_cast<int64_t>(op.getWidth());
    bool isIn = op.getDirection() == PortDir::In;

    // The 6 templates, keyed (direction, width). KEPT BYTE-IDENTICAL to cfront's _PORTIO_IN_ASM /
    // _PORTIO_OUT_ASM (the values inside the Python string literals): the size modifier is byte/word/
    // long (b/w/l) on the accumulator, `%w1` forces the port to 16-bit dx.
    StringRef asmTemplate;
    if (isIn) {
      asmTemplate = (width == 8)    ? "inb %w1, %b0"
                    : (width == 16) ? "inw %w1, %w0"
                                    : "inl %w1, %k0";
    } else {
      asmTemplate = (width == 8)    ? "outb %b0, %w1"
                    : (width == 16) ? "outw %w0, %w1"
                                    : "outl %k0, %w1";
    }
    // Constraint string: in -> "=a,Nd" (output "=a", then input "Nd"); out -> "a,Nd" (two inputs, no
    // output). Byte-identical to the cfront `"=a" (result) : "Nd" (port)` / `"a" (value), "Nd" (port)`.
    StringRef constraints = isIn ? "=a,Nd" : "a,Nd";

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
    // Control-register read -> llvm.inline_asm "mov %<reg>, $0", "=r". LLVM-IR inline-asm syntax ($0
    // operand, single-% register) -- the exact form clang emits; built via the same generic
    // attribute-list InlineAsmOp builder as AsmOpLowering (version-stable). has_side_effects always.
    Type resTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!resTy)
      return rewriter.notifyMatchFailure(op, "result type not convertible");
    std::string tmpl =
        (Twine("mov %") + stringifyCtrlReg(op.getReg()) + ", $0").str();
    SmallVector<NamedAttribute, 3> attrs = {
        rewriter.getNamedAttr("asm_string", rewriter.getStringAttr(tmpl)),
        rewriter.getNamedAttr("constraints", rewriter.getStringAttr("=r")),
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
                        VolatileStoreOp, CRegReadOp, CRegWriteOp>();

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<ComputeOpLowering, BarrierOpLowering, AsmOpLowering,
                 PortioOpLowering, VolatileLoadOpLowering,
                 VolatileStoreOpLowering, CRegReadOpLowering,
                 CRegWriteOpLowering>(typeConverter, &getContext());

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
