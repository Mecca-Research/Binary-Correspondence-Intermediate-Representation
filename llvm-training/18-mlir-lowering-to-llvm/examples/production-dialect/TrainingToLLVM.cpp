//===- TrainingToLLVM.cpp --------------------------------------*- C++ -*-===//
//
// A partial dialect conversion from `training` to the LLVM dialect. Partial on
// purpose: what it cannot lower stays in the output and is reported, rather
// than being silently dropped or forced through a full conversion that fails
// the whole module.
//
// Production reference: ../../../../mlir/lib/passes/BCIRConvertToLLVM.cpp
//
//===----------------------------------------------------------------------===//
#include "TrainingDialect.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::training;

namespace {

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// training.scale %x { factor = k }  ->  llvm.mul %x, k
struct ScaleOpLowering : public OpConversionPattern<ScaleOp> {
  using OpConversionPattern<ScaleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ScaleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // adaptor.getInput(), never op.getInput(): during conversion the operand
    // may already have been replaced by a value of the converted type, and the
    // original may be about to be erased.
    Value input = adaptor.getInput();
    auto type = dyn_cast<IntegerType>(input.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op, "operand is not an integer");

    // MLIR 22/23 spelling. The older `rewriter.create<OpTy>(...)` form is gone
    // in 23; both spellings work in 22, which is what makes the migration
    // possible before the version pin moves.
    Value factor = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), type,
        rewriter.getIntegerAttr(type, op.getFactor()));
    Value scaled = LLVM::MulOp::create(rewriter, op.getLoc(), input, factor);

    rewriter.replaceOp(op, scaled);
    return success();
  }
};

/// training.lane_cast is deliberately NOT lowered here: a lane is an opaque
/// runtime concept in this skeleton. Leaving it illegal-but-unpatterned would
/// fail the conversion, so the target below marks it legal instead -- the
/// difference between "not lowered yet" and "cannot be lowered" has to be
/// stated somewhere, and the target is where.

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TrainingToLLVMPass
    : public PassWrapper<TrainingToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TrainingToLLVMPass)

  StringRef getArgument() const final { return "training-to-llvm"; }
  StringRef getDescription() const final {
    return "Partially convert the training dialect to the LLVM dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // A conversion pass must declare every dialect it can EMIT, not only the
    // ones it reads. Omitting this fails when the first LLVM op is created,
    // which is nowhere near the cause.
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LLVMTypeConverter converter(ctx);

    // !training.lane<width = N> becomes an opaque pointer. A type the converter
    // does not handle stays unconverted, which makes every op using it illegal
    // -- and the resulting error names the op, not the type.
    converter.addConversion([](LaneType type) -> std::optional<Type> {
      return LLVM::LLVMPointerType::get(type.getContext());
    });

    LLVMConversionTarget target(*ctx);
    target.addLegalOp<ModuleOp>();
    // Declared legal because no pattern lowers them: legality and patterns are
    // one decision, made in one place.
    target.addLegalOp<LaneCastOp, EmitOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<ScaleOpLowering>(converter, ctx);

    // Partial, so residual `training` ops survive and are visible in the
    // output. A full conversion here would turn "this part is not lowered yet"
    // into "the module failed".
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::training::registerTrainingToLLVMPass() {
  PassRegistration<TrainingToLLVMPass>();
}
