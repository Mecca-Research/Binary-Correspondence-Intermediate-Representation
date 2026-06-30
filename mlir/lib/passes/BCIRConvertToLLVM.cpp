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
    target.addIllegalOp<ComputeOp, BarrierOp, AsmOp>();

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<ComputeOpLowering, BarrierOpLowering, AsmOpLowering>(
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
