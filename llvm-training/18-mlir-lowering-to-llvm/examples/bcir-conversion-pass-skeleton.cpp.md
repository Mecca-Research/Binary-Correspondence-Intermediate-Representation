# BCIR conversion pass skeleton

This is a teaching skeleton, not a drop-in file. It shows where
`ConversionTarget`, `RewritePatternSet`, `OpConversionPattern`, `TypeConverter`,
and materialization hooks fit in a BCIR-to-LLVM lowering pass.

```c++
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

class BCIRTypeConverter : public LLVMTypeConverter {
public:
  explicit BCIRTypeConverter(MLIRContext *ctx, const DataLayoutAnalysis &layout)
      : LLVMTypeConverter(ctx, layout) {
    addConversion([](Type type) { return type; });

    // Example policy decisions. Real code would use real BCIR C++ type classes.
    addConversion([&](/* bcir::VertexType */ Type type) -> std::optional<Type> {
      if (!isBCIRVertexType(type))
        return std::nullopt;
      return IntegerType::get(type.getContext(), 64);
    });

    addTargetMaterialization(materializeToTarget);
    addSourceMaterialization(materializeToSource);
    addArgumentMaterialization(materializeArgument);
  }

private:
  static bool isBCIRVertexType(Type type) {
    // Replace this placeholder with isa<bcir::VertexType>(type).
    return false;
  }

  static Value materializeToTarget(OpBuilder &builder, Type type,
                                   ValueRange inputs, Location loc) {
    // Insert a short-lived cast/bridge op only if the pass pipeline legalizes it.
    return nullptr;
  }

  static Value materializeToSource(OpBuilder &builder, Type type,
                                   ValueRange inputs, Location loc) {
    return nullptr;
  }

  static Value materializeArgument(OpBuilder &builder, Type type,
                                   ValueRange inputs, Location loc) {
    return nullptr;
  }
};

struct LowerRegisterPrelockPattern
    : public OpConversionPattern</* bcir::RegisterPrelockOp */ Operation> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // 1. Read claim/register attributes before erasing the source op.
    Attribute claimId = op->getAttr("claim_id");
    Attribute logicalRegister = op->getAttr("logical");

    // 2. Emit explicit table lookup / pointer / ABI data.
    // Value table = ...;
    // Value slot = rewriter.create<LLVM::GEPOp>(...);
    // Value resource = rewriter.create<LLVM::LoadOp>(...);

    // 3. Attach translated metadata or diagnostic attributes to replacements.
    // attachBCIRMetadata(resource.getDefiningOp(), claimId, logicalRegister);

    // 4. Replace source results with converted values.
    // rewriter.replaceOp(op, resource);
    return failure(); // Skeleton placeholder.
  }
};

struct LowerBCIRToLLVMPass
    : public PassWrapper<LowerBCIRToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBCIRToLLVMPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    // registry.insert<bcir::BCIRDialect, func::FuncDialect, arith::ArithDialect>();
  }

  StringRef getArgument() const final { return "lower-bcir-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower BCIR graph/resource/HAM operations to LLVM dialect";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    const DataLayoutAnalysis &layout = getAnalysis<DataLayoutAnalysis>();
    BCIRTypeConverter converter(ctx, layout);

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<BuiltinDialect>();
    // target.addIllegalDialect<bcir::BCIRDialect>();
    // target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    //   return converter.isSignatureLegal(op.getFunctionType());
    // });

    RewritePatternSet patterns(ctx);
    patterns.add<LowerRegisterPrelockPattern>(converter, ctx);
    // populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    // populateFuncToLLVMConversionPatterns(converter, patterns);
    // populateVectorToLLVMConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
```

## Fill-in points for real code

- Replace placeholder operation and type names with the generated BCIR dialect
  classes.
- Implement materializations only for bridge values that a later conversion phase
  will remove.
- Use partial conversion for staged graph/affine/vector lowering and full
  conversion for the final LLVM-dialect boundary.
- Add tests that fail when claim IDs, HAM hints, register bindings, or graph IDs
  disappear before translation.
