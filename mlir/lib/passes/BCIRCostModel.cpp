//===- BCIRCostModel.cpp - the K_BCIR cost algebra on the MLIR rail -*- C++ -*-===//
//
// -bcir-cost-model: the keystone of the optimizer-core port. It recomputes each
// claim's candidate set + 12-d cost vectors FROM FIRST PRINCIPLES -- the claim graph
// + the target capability -- instead of trusting the emitter-baked bcir.kbcir.path
// costs. The cost algebra (cost.py::_cost / realize.candidates_for / _stride_penalty)
// and the intra-phase fusion credits (realize.fused_candidates: producer->consumer
// deforestation + CSE) live in BCIRCostModel.h, shared with the -bcir-plan pass.
//
// Annotates each claim with kbcir.cm_candidates, kbcir.cm_min_cost (the argmin
// candidate's fused base cost), kbcir.cm_min_score (its scalarized score under the
// first policy), and kbcir.cm_fusion (the credit earned, if any). Reproduces the
// oracle bit-for-bit: vec16 @ 7808, gather @ 528384, tile @ 126976; deforest 5888,
// CSE 5100 (see mlir/test/passes/cost_model{,_fusion}.mlir).
//
//===----------------------------------------------------------------------===//

#include "BCIRCostModel.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace bcir {
namespace {

struct CostModelPass : public PassWrapper<CostModelPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CostModelPass)

  StringRef getArgument() const final { return "bcir-cost-model"; }
  StringRef getDescription() const final {
    return "Recompute each claim's candidate costs from the claim + target "
           "capability (the K_BCIR cost algebra + intra-phase fusion, C++23 port of "
           "cost.py); annotates kbcir.cm_candidates / cm_min_cost / cm_min_score / "
           "cm_fusion.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    // Scope per bcir.module: each carries its own capability / policy / registry, so
    // a multi-module file (e.g. the widened corpus) is costed module-by-module.
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    // The kbcir.cm_* annotations are not plan inputs -> keep the shared plan analysis.
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    if (!pa.hasCap)
      return; // no target declared -> nothing to compute against
    ArrayRef<int64_t> weights = pa.weights;

    for (const cm::Column &col : pa.cols) {
      if (col.cands.empty())
        continue;
      ClaimOp c = col.claim;
      c->setAttr("kbcir.cm_candidates",
                 b.getI64IntegerAttr(static_cast<int64_t>(col.cands.size())));
      if (!col.fusion.empty())
        c->setAttr("kbcir.cm_fusion", b.getStringAttr(col.fusion));

      size_t best = 0;
      if (!weights.empty()) {
        int64_t bestScore = cm::scalarize(col.cands[0].cost, weights);
        for (size_t i = 1; i < col.cands.size(); ++i) {
          int64_t s = cm::scalarize(col.cands[i].cost, weights);
          if (s < bestScore) {
            bestScore = s;
            best = i;
          }
        }
        c->setAttr("kbcir.cm_min_score", b.getI64IntegerAttr(bestScore));
      }
      const cm::Cost &bc = col.cands[best].cost;
      c->setAttr("kbcir.cm_min_cost",
                 CostVectorAttr::get(&getContext(), bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], bc[6],
                                     bc[7], bc[8], bc[9], bc[10], bc[11]));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCostModelPass() {
  return std::make_unique<CostModelPass>();
}

} // namespace bcir
