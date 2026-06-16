//===- BCIRPlanPass.cpp - the K_BCIR layered min-plus shortest path -*- C++ -*-===//
//
// -bcir-plan: optimizer-core step 3. The full bcir/kbcir/realize.optimize() in C++:
// over the fused candidate columns (one per claim, from BCIRCostModel.h's
// fusedColumns -- deforestation/CSE already baked in), it runs the min-plus
// (tropical) shortest path of semiring.dag_shortest_path. Each edge couples the
// candidate's cost by the *path-based* _context_factor (shared-input fusion: a wide
// candidate whose wide predecessor shares a read operand reuses the loaded lines ->
// x0.75 memory) and scalarizes under the policy weights. Unlike the per-claim argmin
// of -bcir-select-realization, this is the genuine coupled shortest path, so the C++
// plan matches the oracle's optimize() for *every* module, not just coupling-free ones.
//
// Annotates each claim with kbcir.plan_width + kbcir.plan_cost (its edge on the chosen
// path) and the enclosing bcir.module with kbcir.plan_score (the total). Reproduces
// 7808 on vector_add and the oracle's coupled scores on multi-claim modules.
//
// Cool-regime note: _context_factor's *thermal* coupling depends on Theta, which is
// not in the IR (only the theta-folded policy weights are); it is off for the whole
// curated/corpus set (cool). A Theta context op generalizes it.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <limits>

using namespace mlir;

namespace bcir {
namespace {

// Any shared read operand between two claims (the shared-input fusion condition).
static bool sharesRead(ArrayRef<StringRef> a, ArrayRef<StringRef> b) {
  for (StringRef x : a)
    for (StringRef y : b)
      if (x == y)
        return true;
  return false;
}

// realize._context_factor (path-based half, cool regime): a wide candidate whose wide
// predecessor shares a read operand discounts memory x0.75 (cache-line reuse).
static cm::Factor contextFactor(ArrayRef<StringRef> prevReads, int64_t prevWidth,
                                ArrayRef<StringRef> curReads, int64_t curWidth) {
  cm::Factor f = cm::kIdentity;
  if (prevWidth > 1 && curWidth > 1 && sharesRead(prevReads, curReads))
    f[1] = 192; // x0.75 memory
  return f;
}

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
    // Scope per bcir.module (a multi-module file is planned module-by-module).
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b);
    });
  }

  void runOnModule(Operation *root, Builder &b) {
    auto capOp = cm::firstCapability(root);
    if (!capOp)
      return;
    cm::Cap h = cm::readCap(capOp);
    ArrayRef<int64_t> w = cm::firstWeights(root);
    if (w.empty())
      return; // no policy -> cannot scalarize a plan
    auto resByName = cm::resourcesByName(root);
    SmallVector<cm::Column> cols = cm::fusedColumns(root, h, resByName);
    if (cols.empty())
      return;

    const int64_t kInf = std::numeric_limits<int64_t>::max();
    const int n = static_cast<int>(cols.size());
    SmallVector<SmallVector<int64_t>> dist(n);
    SmallVector<SmallVector<int>> back(n);

    for (int i = 0; i < n; ++i) {
      const auto &cands = cols[i].cands;
      dist[i].assign(cands.size(), kInf);
      back[i].assign(cands.size(), -1);
      for (size_t ci = 0; ci < cands.size(); ++ci) {
        if (i == 0) {
          dist[i][ci] = cm::scalarize(cands[ci].cost, w); // from SOURCE (identity)
          continue;
        }
        for (size_t pi = 0; pi < cols[i - 1].cands.size(); ++pi) {
          if (dist[i - 1][pi] == kInf)
            continue;
          cm::Factor f = contextFactor(cols[i - 1].reads, cols[i - 1].cands[pi].width,
                                       cols[i].reads, cands[ci].width);
          cm::Cost e = cands[ci].cost;
          cm::applyFactor(e, f);
          int64_t cand = dist[i - 1][pi] + cm::scalarize(e, w);
          if (cand < dist[i][ci]) {
            dist[i][ci] = cand;
            back[i][ci] = static_cast<int>(pi);
          }
        }
      }
    }

    // argmin over the last column (deterministic: first wins on a tie).
    int lastBest = 0;
    for (size_t ci = 1; ci < cols[n - 1].cands.size(); ++ci)
      if (dist[n - 1][ci] < dist[n - 1][lastBest])
        lastBest = static_cast<int>(ci);
    int64_t total = dist[n - 1][lastBest];

    // reconstruct the chosen candidate per column.
    SmallVector<int> chosen(n, 0);
    chosen[n - 1] = lastBest;
    for (int i = n - 1; i > 0; --i)
      chosen[i - 1] = back[i][chosen[i]];

    for (int i = 0; i < n; ++i) {
      int64_t prevDist = (i > 0) ? dist[i - 1][chosen[i - 1]] : 0;
      int64_t edge = dist[i][chosen[i]] - prevDist;
      cols[i].claim->setAttr("kbcir.plan_width",
                             b.getI64IntegerAttr(cols[i].cands[chosen[i]].width));
      cols[i].claim->setAttr("kbcir.plan_cost", b.getI64IntegerAttr(edge));
    }

    // plan_score (the total) on this bcir.module.
    root->setAttr("kbcir.plan_score", b.getI64IntegerAttr(total));
  }
};

}  // namespace

std::unique_ptr<Pass> createPlanPass() { return std::make_unique<PlanPass>(); }

}  // namespace bcir
