//===- BCIRPromotePass.cpp - the -bcir-promote-lanes opt-law -*- C++ -*-===//
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

using namespace mlir;

namespace bcir {
namespace {
// A claim declared GGG/Random carrying a `bcir.bucketable` proof attribute is
// promoted to UX/Cacheline (LangRef Sec. 11; mirrors bcir.opt.rewrite_rule
// @ggg_to_ux). Legal only with the proof; never invents locality.
struct PromoteGGGtoUX : public OpRewritePattern<ClaimOp> {
  using OpRewritePattern<ClaimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ClaimOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLane() != Lane::GGG || op.getStrideClass() != StrideClass::Random)
      return failure();
    auto proof = op->getAttrOfType<BoolAttr>("bcir.bucketable");
    if (!proof || !proof.getValue())
      return failure();
    rewriter.modifyOpInPlace(op, [&] {
      op.setLaneAttr(LaneAttr::get(op.getContext(), Lane::UX));
      op.setStrideClassAttr(
          StrideClassAttr::get(op.getContext(), StrideClass::Cacheline));
    });
    return success();
  }
};

struct PromoteLanesPass
    : public PassWrapper<PromoteLanesPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteLanesPass)

  StringRef getArgument() const final { return "bcir-promote-lanes"; }
  StringRef getDescription() const final {
    return "Apply the BCIR lane-promotion opt-law (GGG -> UX with proof).";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PromoteGGGtoUX>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createPromoteLanesPass() {
  return std::make_unique<PromoteLanesPass>();
}

}  // namespace bcir
