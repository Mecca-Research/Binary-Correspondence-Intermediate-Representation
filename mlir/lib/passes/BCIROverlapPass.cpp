//===- BCIROverlapPass.cpp - the (max,+) scheduled price M(pi,Theta) -*- C++ -*-===//
//
// -bcir-overlap: optimizer-core step 4. The C++ port of bcir/gem/overlap.py's
// price_scheduled / _makespan. Over the coupled plan (cm::planChosen), it prices the
// CT2 wave schedule M(pi, Theta): same-phase claims fan out across the target's
// affinity domains (combine with max), claims time-sharing a bin serialize with the
// context factor recomputed against the *in-bin* predecessor (a fusion discount only
// for claims that really run back-to-back), and the decoupled GGG/random tail runs
// alongside (phase cost = max(waves, tail)); phases compose in series.
//
// Annotates the bcir.module with kbcir.overlap_{makespan,serial,gain}. The R9 law
// (differential.check_overlap / VerifyPass) holds by construction: makespan + gain ==
// serial == the plan score, and 0 <= makespan <= serial. For a single claim
// makespan == serial (the degenerate case); for independent same-phase claims the
// makespan drops below the serial sum.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <vector>

using namespace mlir;

namespace bcir {
namespace {

// A claim's chosen realization for scheduling: the picked candidate + its I/O. `rmask`/`wmask`
// are the read/write resource symbols as bitmasks over a per-module resource->id map (filled
// when the module has <= 64 distinct resource symbols, the overwhelming common case), so the
// O(C^2) hazard test in the wave loop becomes O(1) bit-ops instead of O(R^2) string compares.
struct Scheduled {
  ClaimOp claim;
  int32_t phase;
  int32_t id;
  bool sparse;            // GGG / random tail (decoupled from the wave order)
  cm::Cand cand;          // chosen width + fused cost
  ArrayRef<StringRef> reads;
  ArrayRef<StringRef> writes;
  uint64_t rmask = 0, wmask = 0;   // resource-symbol bitmasks (valid iff `maskable`)
};

// RAW / WAR / WAW between two claims' read/write sets (mirrors verify._conflict). When the
// module is small enough to bitmask, this is a 3-op intersection; otherwise the string
// fallback is bit-identical (set membership over distinct symbols).
static bool conflict(const Scheduled &a, const Scheduled &b, bool maskable) {
  if (maskable)
    return (a.wmask & (b.rmask | b.wmask)) | (b.wmask & a.rmask);
  for (StringRef w : a.writes)
    for (StringRef x : b.reads)
      if (w == x)
        return true;
  for (StringRef w : a.writes)
    for (StringRef x : b.writes)
      if (w == x)
        return true;
  for (StringRef w : b.writes)
    for (StringRef x : a.reads)
      if (w == x)
        return true;
  return false;
}

// Serial (min,+) cost of a real in-bin execution chain, re-coupling each step against
// its actual in-bin predecessor (overlap.py::_chain_cost). The thermal coupling
// applies per step (even the first) when hot.
static int64_t chainCost(ArrayRef<const Scheduled *> chain, ArrayRef<int64_t> w,
                         int64_t theta) {
  int64_t total = 0;
  const Scheduled *prev = nullptr;
  for (const Scheduled *s : chain) {
    cm::Cost e = s->cand.cost;
    cm::applyFactor(e, prev ? cm::contextFactor(theta, prev->reads, prev->cand.width,
                                                s->reads, s->cand.width)
                            : cm::contextFactor(theta, {}, 0, s->reads, s->cand.width));
    total = saturatingAddNonnegative(total, cm::scalarize(e, w));
    prev = s;
  }
  return total;
}

// The (max,+) scheduled makespan of a candidate-index assignment: per phase, greedy
// conflict waves over the affinity-domain bins (overlap.py::_makespan). A pure function of
// the assignment, so the re-selection sweep can re-price trial assignments.
static int64_t computeMakespan(const std::vector<cm::Column> &cols, ArrayRef<int> assign,
                               int64_t theta, ArrayRef<int64_t> w, int64_t domains) {
  // One resource-symbol -> id map for the whole module (reads + writes). When it fits in 64
  // ids, every claim's I/O becomes a bitmask and conflict() is O(1); otherwise the string
  // fallback runs (bit-identical). The id space is tiny in practice (a few resources/module).
  llvm::DenseMap<StringRef, unsigned> resId;
  for (const cm::Column &col : cols) {
    for (StringRef r : col.reads)
      resId.insert({r, static_cast<unsigned>(resId.size())});
    for (StringRef r : col.writes)
      resId.insert({r, static_cast<unsigned>(resId.size())});
  }
  const bool maskable = resId.size() <= 64;
  auto maskOf = [&](ArrayRef<StringRef> syms) -> uint64_t {
    uint64_t m = 0;
    for (StringRef s : syms)
      m |= (uint64_t{1} << resId.lookup(s));
    return m;
  };

  std::vector<Scheduled> sched;
  SmallVector<int32_t> phaseOrder;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
    ClaimOp claim = cols[i].claim;
    bool sparse =
        claim.getLane() == Lane::GGG || claim.getStrideClass() == StrideClass::Random;
    Scheduled s{claim, cols[i].phase, static_cast<int32_t>(claim.getClaimId()), sparse,
                cols[i].cands[assign[i]], cols[i].reads, cols[i].writes, 0, 0};
    if (maskable) {
      s.rmask = maskOf(cols[i].reads);
      s.wmask = maskOf(cols[i].writes);
    }
    sched.push_back(s);
    if (std::find(phaseOrder.begin(), phaseOrder.end(), cols[i].phase) == phaseOrder.end())
      phaseOrder.push_back(cols[i].phase);
  }
  llvm::sort(phaseOrder);

  int64_t makespan = 0;
  for (int32_t pid : phaseOrder) {
    SmallVector<const Scheduled *> phaseClaims;
    for (const Scheduled &s : sched)
      if (s.phase == pid)
        phaseClaims.push_back(&s);
    llvm::sort(phaseClaims,
               [](const Scheduled *a, const Scheduled *c) { return a->id < c->id; });

    SmallVector<const Scheduled *> main, tail;
    for (const Scheduled *s : phaseClaims)
      (s->sparse ? tail : main).push_back(s);

    llvm::DenseMap<int32_t, int> waveOf;
    int nwaves = 0;
    for (size_t i = 0; i < main.size(); ++i) {
      int wv = 0;
      for (size_t j = 0; j < i; ++j)
        if (conflict(*main[j], *main[i], maskable))
          wv = std::max(wv, waveOf[main[j]->id] + 1);
      waveOf[main[i]->id] = wv;
      nwaves = std::max(nwaves, wv + 1);
    }

    int64_t mainTotal = 0;
    for (int wv = 0; wv < nwaves; ++wv) {
      SmallVector<const Scheduled *> members;
      for (const Scheduled *s : main)
        if (waveOf[s->id] == wv)
          members.push_back(s);
      std::vector<SmallVector<const Scheduled *>> bins(domains);
      for (size_t slot = 0; slot < members.size(); ++slot)
        bins[slot % domains].push_back(members[slot]);
      int64_t waveMax = 0;
      for (auto &bin : bins)
        if (!bin.empty())
          waveMax = std::max(waveMax, chainCost(bin, w, theta));
      mainTotal = saturatingAddNonnegative(mainTotal, waveMax);
    }

    int64_t tailTotal = chainCost(tail, w, theta);
    makespan = saturatingAddNonnegative(makespan,
                                        std::max(mainTotal, tailTotal));
  }
  return makespan;
}

// The serial (min,+) plan score of an assignment: the chosen coupled edge per column,
// scalarized and summed (overlap.py::_serial_result -- R9-consistent: score == Sigma steps).
static int64_t serialScore(const std::vector<cm::Column> &cols, ArrayRef<int> assign,
                           int64_t theta, ArrayRef<int64_t> w) {
  int64_t total = 0;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
    cm::Cost e = cols[i].cands[assign[i]].cost;
    cm::Factor f = (i > 0)
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
    int64_t makespan =
        computeMakespan(pa.cols, pa.chosen, pa.thetaThermal, pa.weights, pa.affinityDomains);
    root->setAttr("kbcir.overlap_serial", b.getI64IntegerAttr(pa.total));
    root->setAttr("kbcir.overlap_makespan", b.getI64IntegerAttr(makespan));
    root->setAttr("kbcir.overlap_gain",
                  b.getI64IntegerAttr(makespan <= pa.total
                                          ? pa.total - makespan
                                          : 0));
  }
};

// -bcir-overlap-optimize: the makespan-driven re-selection sweep (overlap.py
// ::optimize_scheduled). Starting from the serial optimum, sweep each claim once in column
// (flatten) order and adopt the legal alternative that *strictly* lowers the scheduled
// makespan most (deterministic first-best tie-break, carrying earlier adoptions forward).
// The serial optimum is usually already makespan-optimal (the sweep is then a no-op -- it
// must not churn a plan the coupled shortest path got right), but where a narrower lane
// packs the waves better it adopts it. Re-prices serially so R9 still holds (makespan <=
// serial). Annotates kbcir.overlap_opt_{makespan,serial,gain} + per-claim opt_width.
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

    SmallVector<int> assign(pa.chosen.begin(), pa.chosen.end());   // start: the serial optimum
    int64_t bestM = computeMakespan(cols, assign, theta, w, domains);
    for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
      const int cur = assign[i];
      int bestCand = cur;
      int64_t bestTrial = bestM;
      for (int ci = 0; ci < static_cast<int>(cols[i].cands.size()); ++ci) {
        if (ci == cur)
          continue;
        assign[i] = ci;
        int64_t m = computeMakespan(cols, assign, theta, w, domains);
        if (m < bestTrial) {       // strict: first alternative reaching the new minimum wins
          bestCand = ci;
          bestTrial = m;
        }
      }
      assign[i] = bestCand;        // commit (carry the adoption into later claims' sweeps)
      if (bestCand != cur)
        bestM = bestTrial;
    }

    int64_t serial = serialScore(cols, assign, theta, w);   // R9-consistent re-price
    for (int i = 0; i < static_cast<int>(cols.size()); ++i)
      cols[i].claim->setAttr("kbcir.overlap_opt_width",
                             b.getI64IntegerAttr(cols[i].cands[assign[i]].width));
    root->setAttr("kbcir.overlap_opt_serial", b.getI64IntegerAttr(serial));
    root->setAttr("kbcir.overlap_opt_makespan", b.getI64IntegerAttr(bestM));
    root->setAttr("kbcir.overlap_opt_gain", b.getI64IntegerAttr(serial - bestM));
  }
};

}  // namespace

std::unique_ptr<Pass> createOverlapPass() {
  return std::make_unique<OverlapPass>();
}

std::unique_ptr<Pass> createOverlapOptimizePass() {
  return std::make_unique<OverlapOptimizePass>();
}

}  // namespace bcir
