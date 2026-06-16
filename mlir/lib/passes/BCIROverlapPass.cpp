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

// A claim's chosen realization for scheduling: the picked candidate + its I/O.
struct Scheduled {
  ClaimOp claim;
  int32_t phase;
  int32_t id;
  bool sparse;            // GGG / random tail (decoupled from the wave order)
  cm::Cand cand;          // chosen width + fused cost
  ArrayRef<StringRef> reads;
  ArrayRef<StringRef> writes;
};

static bool conflict(const Scheduled &a, const Scheduled &b) {
  // RAW / WAR / WAW: aw & (br|bw) || bw & ar (mirrors verify._conflict).
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
    total += cm::scalarize(e, w);
    prev = s;
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
      return;
    auto resByName = cm::resourcesByName(root);
    std::vector<cm::Column> cols = cm::fusedColumns(root, h, resByName);
    if (cols.empty())
      return;

    int64_t theta = cm::firstThetaThermal(root);
    int64_t serial = 0;
    SmallVector<int> chosen = cm::planChosen(cols, w, theta, serial);
    if (chosen.empty())
      return;

    // The scheduled view of each chosen claim.
    std::vector<Scheduled> sched;
    SmallVector<int32_t> phaseOrder;
    for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
      cm::Column &col = cols[i];
      bool sparse = col.claim.getLane() == Lane::GGG ||
                    col.claim.getStrideClass() == StrideClass::Random;
      sched.push_back({col.claim, col.phase, static_cast<int32_t>(col.claim.getClaimId()),
                       sparse, col.cands[chosen[i]], col.reads, col.writes});
      if (std::find(phaseOrder.begin(), phaseOrder.end(), col.phase) == phaseOrder.end())
        phaseOrder.push_back(col.phase);
    }
    llvm::sort(phaseOrder);

    int64_t domains = std::max<int64_t>(1, capOp.getAffinityDomains());
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

      // Greedy wave assignment by conflict (concurrency.schedule_concurrent).
      llvm::DenseMap<int32_t, int> waveOf;
      int nwaves = 0;
      for (size_t i = 0; i < main.size(); ++i) {
        int wv = 0;
        for (size_t j = 0; j < i; ++j)
          if (conflict(*main[j], *main[i]))
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
        // Round-robin affinity bins; one bin's claims run back-to-back.
        std::vector<SmallVector<const Scheduled *>> bins(domains);
        for (size_t slot = 0; slot < members.size(); ++slot)
          bins[slot % domains].push_back(members[slot]);
        int64_t waveMax = 0;
        for (auto &bin : bins)
          if (!bin.empty())
            waveMax = std::max(waveMax, chainCost(bin, w, theta));
        mainTotal += waveMax;
      }

      int64_t tailTotal = chainCost(tail, w, theta);
      makespan += std::max(mainTotal, tailTotal);
    }

    root->setAttr("kbcir.overlap_serial", b.getI64IntegerAttr(serial));
    root->setAttr("kbcir.overlap_makespan", b.getI64IntegerAttr(makespan));
    root->setAttr("kbcir.overlap_gain", b.getI64IntegerAttr(serial - makespan));
  }
};

}  // namespace

std::unique_ptr<Pass> createOverlapPass() {
  return std::make_unique<OverlapPass>();
}

}  // namespace bcir
