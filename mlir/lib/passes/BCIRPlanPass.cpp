//===- BCIRPlanPass.cpp - the K_BCIR layered min-plus shortest path -*- C++ -*-===//
//
// -bcir-plan: optimizer-core step 3. The full bcir/kbcir/realize.optimize() in C++.
// Over the fused candidate columns (BCIRCostModel.h's fusedColumns), cm::planChosen
// runs the min-plus (tropical) shortest path of semiring.dag_shortest_path, each edge
// coupling the path-based _context_factor (shared-input fusion: a wide candidate whose
// wide predecessor shares a read operand reuses the loaded lines -> x0.75 memory).
// Unlike the per-claim argmin of -bcir-select-realization, this is the genuine coupled
// shortest path, so the C++ plan matches the oracle's optimize() for *every* module.
//
// Annotates each claim with kbcir.plan_width + kbcir.plan_cost (its edge on the chosen
// path) and the enclosing bcir.module with kbcir.plan_score (the total). Reproduces
// 7808 on vector_add and the oracle's coupled scores on multi-claim modules.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace bcir {
namespace {

struct PlanPass : public PassWrapper<PlanPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanPass)

  StringRef getArgument() const final { return "bcir-plan"; }
  StringRef getDescription() const final {
    return "K_BCIR min-plus shortest path over the fused candidate DAG (the full "
           "realize.optimize in C++): selects the coupled plan and annotates "
           "kbcir.plan_width / plan_cost / plan_score.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    // The kbcir.plan_* annotations added here are not inputs to the plan, so the shared
    // PlanAnalysis stays valid for the next pass in the pipeline.
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    if (!pa.valid)
      return;
    const std::vector<cm::Column> &cols = pa.cols;
    const SmallVector<int> &chosen = pa.chosen;
    const int64_t theta = pa.thetaThermal;

    for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
      // The chosen edge cost = this candidate coupled by the context of the chosen
      // predecessor (source = no fusion, thermal coupling still applies), scalarized.
      cm::Cost e = cols[i].cands[chosen[i]].cost;
      cm::Factor f = (i > 0)
          ? cm::contextFactor(theta, cols[i - 1].reads,
                              cols[i - 1].cands[chosen[i - 1]].width, cols[i].reads,
                              cols[i].cands[chosen[i]].width)
          : cm::contextFactor(theta, {}, 0, cols[i].reads, cols[i].cands[chosen[i]].width);
      cm::applyFactor(e, f);
      int64_t edge = cm::scalarize(e, pa.weights);
      cols[i].claim->setAttr("kbcir.plan_width",
                             b.getI64IntegerAttr(cols[i].cands[chosen[i]].width));
      cols[i].claim->setAttr("kbcir.plan_cost", b.getI64IntegerAttr(edge));
    }
    root->setAttr("kbcir.plan_score", b.getI64IntegerAttr(pa.total));
  }
};

}  // namespace

std::unique_ptr<Pass> createPlanPass() { return std::make_unique<PlanPass>(); }

}  // namespace bcir
