//===- BCIRRcspPlanPass.cpp - plan-level constrained selection (RCSP) -*- C++ -*-===//
//
// -bcir-rcsp-plan: optimizer-core step 5 (the last piece). The C++ port of
// bcir/kbcir/rcsp.py::optimize_constrained -- the resource-constrained shortest path
// over the *whole* plan. Where -bcir-rcsp caps each claim's candidate independently,
// this runs the accumulated-budget label DP across the fused candidate columns: a
// label carries (score, per-tracked-dim resource totals) along the coupled path,
// dominated labels are pruned and infeasible extensions cut, so a budget bounds the
// PLAN's accumulated thermal/power -- a decision no per-claim cap can make.
//
// Reads the first bcir.kbcir.budget (dims/caps) + the target capability + registry,
// and annotates the bcir.module with kbcir.rcsp_plan_score (the constrained optimum)
// and each claim with kbcir.rcsp_width. Reproduces optimize_constrained bit-for-bit:
// e.g. two vec16 claims (thermal 2176) under thermal<=2000 -> {16,8} @ 17280 (one
// claim narrows to fit the accumulated cap), under <=1500 -> {8,8} @ 18944.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <vector>

using namespace mlir;

namespace bcir {
namespace {

// Cost-vector dimension index by name (DIMS order; cost.py).
static int dimIndex(StringRef name) {
  return llvm::StringSwitch<int>(name)
      .Case("compute", 0).Case("memory", 1).Case("fabric", 2).Case("sync", 3)
      .Case("compile", 4).Case("thermal", 5).Case("power", 6).Case("reliability", 7)
      .Case("security", 8).Case("accuracy", 9).Case("contention", 10)
      .Case("verification", 11).Default(-1);
}

// A non-dominated partial plan (rcsp._Label): scalar score + tracked-dim resource
// totals, plus the trail to reconstruct the chosen widths.
struct Label {
  int64_t score;
  SmallVector<int64_t> res;       // accumulated, per tracked dim
  int64_t lastWidth;              // the candidate realized here (for the next context)
  ArrayRef<StringRef> lastReads;  // ... and its claim's reads
  int parent;                     // index into the previous column's labels (-1 = source)
  int candIdx;                    // candidate chosen at this column
};

static bool dominates(const Label &a, const Label &b) {
  if (a.score > b.score)
    return false;
  for (size_t i = 0; i < a.res.size(); ++i)
    if (a.res[i] > b.res[i])
      return false;
  return true;
}

// rcsp._insert: keep `labels` a non-dominated set (stable; ties keep the first).
static void insertLabel(std::vector<Label> &labels, Label nw) {
  for (const Label &ex : labels)
    if (dominates(ex, nw))
      return;
  labels.erase(std::remove_if(labels.begin(), labels.end(),
                              [&](const Label &ex) { return dominates(nw, ex); }),
               labels.end());
  labels.push_back(std::move(nw));
}

struct RcspPlanPass : public PassWrapper<RcspPlanPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RcspPlanPass)

  StringRef getArgument() const final { return "bcir-rcsp-plan"; }
  StringRef getDescription() const final {
    return "Plan-level constrained selection (RCSP): the accumulated-budget label DP "
           "over the fused candidate DAG (port of rcsp.optimize_constrained); "
           "annotates kbcir.rcsp_plan_score / rcsp_width.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    // Only kbcir.* annotations are added -> the shared plan analysis stays valid.
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    if (!pa.hasCap || pa.weights.empty())
      return;
    ArrayRef<int64_t> w = pa.weights;
    // The first budget op supplies the caps (tracked dims).
    KBCIRBudgetOp budget;
    root->walk([&](KBCIRBudgetOp bd) {
      if (!budget)
        budget = bd;
    });
    if (!budget)
      return; // no budget -> nothing to constrain (use -bcir-plan)

    SmallVector<int> trackedDim;
    SmallVector<int64_t> caps;
    ArrayAttr dims = budget.getDims();
    ArrayRef<int64_t> capArr = budget.getCaps();
    for (size_t i = 0; i < dims.size() && i < capArr.size(); ++i)
      if (auto nm = dyn_cast<StringAttr>(dims[i])) {
        int d = dimIndex(nm.getValue());
        if (d >= 0) {
          trackedDim.push_back(d);
          caps.push_back(capArr[i]);
        }
      }
    if (trackedDim.empty())
      return;

    const std::vector<cm::Column> &cols = pa.cols;
    int64_t theta = pa.thetaThermal;

    const int nt = static_cast<int>(trackedDim.size());
    // The label DP (rcsp._expand): one non-dominated label set per column.
    std::vector<std::vector<Label>> layer(cols.size());
    std::vector<Label> source = {
        Label{0, SmallVector<int64_t>(nt, 0), 0, {}, -1, -1}};

    for (size_t i = 0; i < cols.size(); ++i) {
      std::vector<Label> &prev = (i == 0) ? source : layer[i - 1];
      for (size_t ci = 0; ci < cols[i].cands.size(); ++ci) {
        const cm::Cand &cand = cols[i].cands[ci];
        for (size_t pi = 0; pi < prev.size(); ++pi) {
          const Label &p = prev[pi];
          cm::Cost e = cand.cost;
          cm::applyFactor(e, cm::contextFactor(theta, p.lastReads, p.lastWidth,
                                               cols[i].reads, cand.width));
          SmallVector<int64_t> res(nt);
          bool feasible = true;
          for (int t = 0; t < nt; ++t) {
            res[t] = saturatingAddNonnegative(p.res[t], e[trackedDim[t]]);
            if (res[t] > caps[t]) {
              feasible = false;
              break;
            }
          }
          if (!feasible)
            continue;
          insertLabel(layer[i], Label{saturatingAddNonnegative(
                                          p.score, cm::scalarize(e, w)),
                                      std::move(res),
                                      cand.width, cols[i].reads,
                                      static_cast<int>(pi), static_cast<int>(ci)});
        }
      }
      if (layer[i].empty()) {
        root->emitError("bcir-rcsp-plan: no plan satisfies the budget @")
            << budget.getSymName();
        signalPassFailure();
        return;
      }
    }

    // The constrained optimum: min (score, res) over the sink labels.
    std::vector<Label> &sink = layer.back();
    auto less = [](const Label &a, const Label &z) {
      if (a.score != z.score)
        return a.score < z.score;
      for (size_t i = 0; i < a.res.size(); ++i)
        if (a.res[i] != z.res[i])
          return a.res[i] < z.res[i];
      return false;
    };
    int best = 0;
    for (size_t i = 1; i < sink.size(); ++i)
      if (less(sink[i], sink[best]))
        best = static_cast<int>(i);
    int64_t optScore = sink[best].score;

    // Reconstruct the chosen width per column via the parent trail.
    SmallVector<int> chosen(cols.size(), 0);
    int li = best;
    for (int i = static_cast<int>(cols.size()) - 1; i >= 0; --i) {
      const Label &lab = layer[i][li];
      chosen[i] = lab.candIdx;
      li = lab.parent;
    }
    for (size_t i = 0; i < cols.size(); ++i)
      cols[i].claim->setAttr("kbcir.rcsp_width",
                             b.getI64IntegerAttr(cols[i].cands[chosen[i]].width));
    root->setAttr("kbcir.rcsp_plan_score", b.getI64IntegerAttr(optScore));
  }
};

}  // namespace

std::unique_ptr<Pass> createRcspPlanPass() {
  return std::make_unique<RcspPlanPass>();
}

}  // namespace bcir
