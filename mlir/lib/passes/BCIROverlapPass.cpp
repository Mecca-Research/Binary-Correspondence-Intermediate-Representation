//===- BCIROverlapPass.cpp - the scheduled price M(pi,Theta) over the canonical artifact -*- C++ -*-===//
//
// -bcir-overlap: optimizer-core step 4. The C++ port of bcir/gem/overlap.py's
// price_scheduled / _makespan. Over the coupled plan (cm::planChosen) it READS the one
// canonical schedule artifact (BCIRSchedule.h, the twin of gem.schedule.schedule_plan --
// G1 / S1-A): the plan's coupled step costs are the durations, the hazard DAG (data hazards
// and ordering fences) is built over every claim of a phase before the stream split, and
// the hazard-honoring LPT/EFT placement -bcir-schedule-eft annotates is what is priced.
// M(pi, Theta) is that placement's makespan; the serial score is the sum of its durations.
//
// Before this slice the pass priced fixed greedy conflict waves with round-robin affinity
// bins, re-coupled in-bin chains and ran the GGG tail as a hazard-free chain alongside --
// a pricer the executor never ran (the 2026-08-12 report's P0.1: 51,200 against 25,700 on
// four independent claims). The oracle keeps that pricer as `price_waves_legacy`, the
// divergence witness; this rail prices only the artifact.
//
// Annotates the bcir.module with kbcir.overlap_{makespan,serial,gain}. The R9 law
// (differential.check_overlap / VerifyPass) holds by construction: makespan + gain ==
// serial == the plan score, and 0 <= makespan <= serial. For a single claim
// makespan == serial (the degenerate case); for independent same-phase claims the
// makespan drops below the serial sum.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRPasses.h"
#include "BCIRCostModel.h"
#include "BCIRSchedule.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"

#include <vector>

using namespace mlir;

namespace bcir {
namespace {

// The serial (min,+) plan score of an assignment: the chosen coupled edge per column,
// scalarized and summed (overlap.py::_serial_result -- R9-consistent: score == Sigma steps).
static int64_t serialScore(const std::vector<cm::Column> &cols, ArrayRef<int> assign, int64_t theta,
                           ArrayRef<int64_t> w) {
  int64_t total = 0;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
    cm::Cost e = cols[i].cands[assign[i]].cost;
    cm::Factor f =
        (i > 0)
            ? cm::contextFactor(theta, cols[i - 1].reads, cols[i - 1].cands[assign[i - 1]].width,
                                cols[i].reads, cols[i].cands[assign[i]].width)
            : cm::contextFactor(theta, {}, 0, cols[i].reads, cols[i].cands[assign[i]].width);
    cm::applyFactor(e, f);
    total = saturatingAddNonnegative(total, cm::scalarize(e, w));
  }
  return total;
}

struct OverlapPass : public PassWrapper<OverlapPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OverlapPass)

  StringRef getArgument() const final { return "bcir-overlap"; }
  StringRef getDescription() const final {
    return "Price the coupled plan under the CT2 wave schedule -- M(pi,Theta), the "
           "(max,+) scheduled price (port of gem/overlap.py); annotates "
           "kbcir.overlap_makespan / overlap_serial / overlap_gain.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    // The kbcir.overlap_* annotations are not plan inputs -> keep the shared plan.
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    if (!pa.valid)
      return;
    SmallVector<int32_t> phaseIds = canonicalPhaseIds(root);
    int64_t makespan = sched::makespanOf(pa.cols, pa.chosen, pa.thetaThermal, pa.weights, phaseIds,
                                         pa.affinityDomains, sched::kneeOf(pa));
    root->setAttr("kbcir.overlap_serial", b.getI64IntegerAttr(pa.total));
    root->setAttr("kbcir.overlap_makespan", b.getI64IntegerAttr(makespan));
    root->setAttr("kbcir.overlap_gain",
                  b.getI64IntegerAttr(makespan <= pa.total ? pa.total - makespan : 0));
  }
};

// -bcir-overlap-optimize: the makespan-driven re-selection sweep (overlap.py
// ::optimize_scheduled). Starting from the serial optimum, sweep each claim once in column
// (flatten) order and adopt the legal alternative that *strictly* lowers the scheduled
// makespan most (deterministic first-best tie-break, carrying earlier adoptions forward).
// The makespan is a placement of the plan's step costs, so only an alternative that SHORTENS
// a step it touches -- its own, or its textual successor's through the context coupling -- is
// placed and compared (a step that only lengthens cannot lower the makespan except through a
// list-scheduling anomaly, which is not a property of the plan); each trial re-prices those
// two steps and re-places. The serial optimum is usually already makespan-optimal (the sweep
// is then a no-op -- it must not churn a plan the coupled shortest path got right), but where
// a narrower lane packs the waves better it adopts it. Re-prices serially so R9 still holds
// (makespan <= serial). Annotates kbcir.overlap_opt_{makespan,serial,gain} + per-claim
// opt_width.
struct OverlapOptimizePass : public PassWrapper<OverlapOptimizePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OverlapOptimizePass)

  StringRef getArgument() const final { return "bcir-overlap-optimize"; }
  StringRef getDescription() const final {
    return "Makespan-driven re-selection sweep (port of gem/overlap.py::optimize_scheduled): "
           "adopt the per-claim alternative that strictly lowers the scheduled makespan; "
           "annotates kbcir.overlap_opt_makespan / overlap_opt_serial / overlap_opt_gain.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    // The sweep re-selects realizations -> it does NOT preserve the base plan analysis.
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    if (!pa.valid)
      return;
    const std::vector<cm::Column> &cols = pa.cols;
    ArrayRef<int64_t> w = pa.weights;
    const int64_t theta = pa.thetaThermal, domains = pa.affinityDomains;
    const int64_t knee = sched::kneeOf(pa);
    const int n = static_cast<int>(cols.size());
    SmallVector<int32_t> phaseIds = canonicalPhaseIds(root);

    SmallVector<int> assign(pa.chosen.begin(), pa.chosen.end()); // start: the serial optimum
    // The artifact's per-claim durations (the plan's step costs); each trial patches the two
    // steps an alternative touches and re-places the same claims over the same hazard DAG.
    SmallVector<sched::Info> infos = sched::buildInfos(cols, assign, theta, w);
    auto place = [&]() {
      llvm::DenseMap<int64_t, sched::Slot> slots;
      return sched::placeBarriered(phaseIds, infos, domains, knee, slots);
    };
    int64_t bestM = place();
    for (int i = 0; i < n; ++i) {
      const int cur = assign[i];
      int bestCand = cur;
      int64_t bestTrial = bestM, bestDur = infos[i].dur, bestNext = -1;
      for (int ci = 0; ci < static_cast<int>(cols[i].cands.size()); ++ci) {
        if (ci == cur)
          continue;
        assign[i] = ci;
        int64_t di = sched::stepCost(cols, assign, i, theta, w);
        int64_t dn = (i + 1 < n) ? sched::stepCost(cols, assign, i + 1, theta, w) : -1;
        assign[i] = cur;
        bool shorter = di < infos[i].dur || (i + 1 < n && dn < infos[i + 1].dur);
        if (!shorter)
          continue; // a step that only lengthens cannot lower a placement of step costs
        int64_t savedI = infos[i].dur, savedN = (i + 1 < n) ? infos[i + 1].dur : -1;
        infos[i].dur = di;
        if (i + 1 < n)
          infos[i + 1].dur = dn;
        int64_t m = place();
        infos[i].dur = savedI;
        if (i + 1 < n)
          infos[i + 1].dur = savedN;
        if (m < bestTrial) { // strict: first alternative reaching the new minimum wins
          bestCand = ci;
          bestTrial = m;
          bestDur = di;
          bestNext = dn;
        }
      }
      assign[i] = bestCand; // commit (carry the adoption into later claims' sweeps)
      if (bestCand != cur) {
        bestM = bestTrial;
        infos[i].dur = bestDur;
        if (i + 1 < n)
          infos[i + 1].dur = bestNext;
      }
    }

    int64_t serial = serialScore(cols, assign, theta, w); // R9-consistent re-price
    for (int i = 0; i < static_cast<int>(cols.size()); ++i)
      cols[i].claim->setAttr("kbcir.overlap_opt_width",
                             b.getI64IntegerAttr(cols[i].cands[assign[i]].width));
    root->setAttr("kbcir.overlap_opt_serial", b.getI64IntegerAttr(serial));
    root->setAttr("kbcir.overlap_opt_makespan", b.getI64IntegerAttr(bestM));
    root->setAttr("kbcir.overlap_opt_gain", b.getI64IntegerAttr(serial - bestM));
  }
};

} // namespace

std::unique_ptr<Pass> createOverlapPass() {
  return std::make_unique<OverlapPass>();
}

std::unique_ptr<Pass> createOverlapOptimizePass() {
  return std::make_unique<OverlapOptimizePass>();
}

} // namespace bcir
