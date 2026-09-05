//===- BCIRSchedule.h - the ONE canonical schedule artifact (gem.schedule) -*- C++ -*-===//
//
// The C++ twin of bcir/gem/schedule.py + concurrency.hazard_predecessors (G1 / S1-A),
// shared header-only by the passes that price or place a plan:
//   -bcir-overlap / -bcir-overlap-optimize (BCIROverlapPass.cpp): M(pi,Theta) is the makespan
//                                          of this placement over the plan's step costs;
//   -bcir-schedule-eft / -bcir-async / -bcir-power-rail (BCIRScheduleEftPass.cpp): the
//                                          placement itself, annotated per claim.
//
// One placement: the plan's coupled step costs are the durations (their sum is the serial
// score, so makespan <= serial by construction); the hazard DAG is built over EVERY claim
// of a phase -- data hazards (RAW/WAR/WAW) AND ordering fences (a `barriered` or volatile
// claim) -- BEFORE the stream split; an event-driven LPT list scheduler places each ready
// claim at its earliest finish over the affinity domains (locality tie-break, the bandwidth
// knee for bandwidth-class claims) with the sparse GGG/random tail on its own stream inside
// the same loop, so a gather that reads what a wave claim writes waits for it. Before this
// slice -bcir-overlap priced fixed conflict waves with round-robin bins and the executor
// passes ran EFT placement with the tail as a hazard-free chain: two prices for one plan
// (the 2026-08-12 report's P0.1). NOT a public API.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_LIB_PASSES_BCIRSCHEDULE_H
#define BCIR_LIB_PASSES_BCIRSCHEDULE_H

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"
#include "BCIRPassSupport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace bcir {
namespace sched {

constexpr int64_t kTailStream = -1; // the decoupled GGG/random tail's own stream (TAIL_STREAM)

// gem.concurrency._is_sparse: the decoupled GGG/random tail.
inline bool isSparse(ClaimOp c) {
  return c.getLane() == Lane::GGG || c.getStrideClass() == StrideClass::Random;
}

// gem.concurrency.is_fence: an ordering fence (ASM3b `barriered`, §5.14 volatile) -- never
// reordered, fused, bundled or overlapped with any other claim, on either side.
inline bool isFence(ClaimOp c) {
  return c.getHazard() == HazardMode::Barriered || c.getIsVolatile();
}

// gem.concurrency._conflict: a RAW/WAR/WAW between two claims' read/write symbol sets.
inline bool dataConflict(ArrayRef<StringRef> ar, ArrayRef<StringRef> aw, ArrayRef<StringRef> br,
                         ArrayRef<StringRef> bw) {
  auto hits = [](ArrayRef<StringRef> x, ArrayRef<StringRef> y) {
    for (StringRef a : x)
      for (StringRef b : y)
        if (a == b)
          return true;
    return false;
  };
  return hits(aw, br) || hits(aw, bw) || hits(bw, ar);
}

// One claim of the plan as the scheduler sees it.
struct Info {
  ClaimOp claim;
  int32_t phase = 0;
  int64_t id = 0;
  int64_t dur = 0;                         // the plan's coupled step cost (durations_from)
  int64_t baseCompute = 0, baseMemory = 0; // the chosen candidate's base cost (power rail)
  bool bandwidth = true;                   // cost_class == bandwidth (the default) -> knee-clamped
  bool sparse = false;                     // GGG / random tail -> the tail stream
  bool fence = false;                      // barriered / volatile -> conflicts with everything
  SmallVector<StringRef> reads, writes;
  llvm::DenseSet<StringRef> rids; // _rids = set(rd) | set(wr) -- deduped, for locality
};

// A placed slot (gem.schedule.Slot): the stream (an affinity domain or kTailStream) and the
// [start, finish) interval.
struct Slot {
  int64_t domain = 0, start = 0, finish = 0;
};

// gem.concurrency.hazard_conflict: the ONE scheduling-hazard predicate.
inline bool hazardConflict(const Info &a, const Info &b) {
  return a.fence || b.fence || dataConflict(a.reads, a.writes, b.reads, b.writes);
}

// gem.schedule.bandwidth_knee: min(affinity domains, memory channels), at least 1.
inline int64_t kneeOf(const cm::PlanAnalysis &pa) {
  TargetCapabilityOp cap = pa.capOp; // value-handle copy for the non-const accessor
  return std::max<int64_t>(1, std::min<int64_t>(pa.affinityDomains, cap.getMemChannels()));
}

// The coupled step cost of column `i` under an assignment: the chosen candidate's cost coupled
// against its textual predecessor (realize.edge_cost -- the planner's DAG edge weight, the
// number R9 re-derives, and the plan's own step cost).
inline int64_t stepCost(const std::vector<cm::Column> &cols, ArrayRef<int> assign, int i,
                        int64_t theta, ArrayRef<int64_t> w) {
  cm::Cost e = cols[i].cands[assign[i]].cost;
  cm::Factor f =
      (i > 0) ? cm::contextFactor(theta, cols[i - 1].reads, cols[i - 1].cands[assign[i - 1]].width,
                                  cols[i].reads, cols[i].cands[assign[i]].width)
              : cm::contextFactor(theta, {}, 0, cols[i].reads, cols[i].cands[assign[i]].width);
  cm::applyFactor(e, f);
  return std::max<int64_t>(0, cm::scalarize(e, w));
}

// Per-claim durations from a candidate-index assignment over the plan's columns: the chosen
// coupled edge cost per column (gem.schedule.durations_from over overlap._serial_result --
// exactly the step costs the plan sums, so their sum is the serial score; a zero-cost step
// is a zero-length slot, not a unit one).
inline SmallVector<Info> buildInfos(const std::vector<cm::Column> &cols, ArrayRef<int> assign,
                                    int64_t theta, ArrayRef<int64_t> w) {
  SmallVector<Info> infos;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
    Info in;
    in.claim = cols[i].claim;
    in.phase = cols[i].phase;
    in.id = static_cast<int64_t>(in.claim.getClaimId());
    in.dur = stepCost(cols, assign, i, theta, w);
    // The base (uncoupled) compute/memory of the chosen candidate -- gem.dvfs.classify keys off
    // this, not the contextual duration (COMPUTE=0, MEMORY=1).
    in.baseCompute = cols[i].cands[assign[i]].cost[0];
    in.baseMemory = cols[i].cands[assign[i]].cost[1];
    in.bandwidth = !in.claim.getCostClass() || *in.claim.getCostClass() == CostClass::Bandwidth;
    in.sparse = isSparse(in.claim);
    in.fence = isFence(in.claim);
    in.reads.assign(cols[i].reads.begin(), cols[i].reads.end());
    in.writes.assign(cols[i].writes.begin(), cols[i].writes.end());
    for (StringRef r : in.reads)
      in.rids.insert(r);
    for (StringRef r : in.writes)
      in.rids.insert(r);
    infos.push_back(std::move(in));
  }
  return infos;
}

// gem.concurrency.hazard_predecessors over `claims` in position order: every earlier claim
// with a data hazard, plus the fence edges -- a fence waits for every claim since the previous
// fence and for that fence, every later claim waits for the most recent fence (transitively
// "a fence conflicts with everything", O(claims) fence edges). Byte-identical to the data-only
// DAG on a fence-free module. Predecessors are listed in position order.
inline llvm::DenseMap<int64_t, SmallVector<int64_t>> hazardPredecessors(ArrayRef<Info *> claims) {
  llvm::DenseMap<int64_t, SmallVector<int64_t>> preds;
  int64_t lastFence = -1;
  bool haveFence = false;
  SmallVector<unsigned> segment; // positions of the non-fence claims since the last fence
  for (unsigned i = 0; i < claims.size(); ++i) {
    SmallVector<int64_t> &out = preds[claims[i]->id];
    llvm::DenseSet<int64_t> have;
    SmallVector<unsigned> positions;
    for (unsigned j = 0; j < i; ++j)
      if (dataConflict(claims[j]->reads, claims[j]->writes, claims[i]->reads, claims[i]->writes))
        positions.push_back(j);
    if (claims[i]->fence) {
      if (haveFence)
        positions.push_back(static_cast<unsigned>(lastFence));
      positions.append(segment.begin(), segment.end());
      lastFence = static_cast<int64_t>(i);
      haveFence = true;
      segment.clear();
    } else {
      if (haveFence)
        positions.push_back(static_cast<unsigned>(lastFence));
      segment.push_back(i);
    }
    llvm::sort(positions);
    for (unsigned j : positions)
      if (have.insert(claims[j]->id).second)
        out.push_back(claims[j]->id);
  }
  return preds;
}

// gem.schedule._dispatch: event-driven LPT list scheduling honoring `preds`, placing each claim
// at the earliest finish over its eligible streams -- the bandwidth knee (bandwidth-class) or
// every affinity domain (compute-class), or the tail stream (index `domains`) for a sparse
// claim -- with the locality tie-break. `domainFree` / `resident` carry domains + 1 entries.
inline void eftDispatch(ArrayRef<Info *> claims,
                        const llvm::DenseMap<int64_t, SmallVector<int64_t>> &preds, int64_t t0,
                        int64_t domains, int64_t knee, SmallVector<int64_t> &domainFree,
                        SmallVector<llvm::DenseSet<StringRef>> &resident,
                        llvm::DenseMap<int64_t, int64_t> &finishOf,
                        llvm::DenseMap<int64_t, Slot> &slots) {
  SmallVector<Info *> pending(claims.begin(), claims.end());
  while (!pending.empty()) {
    // Ready = every predecessor has finished. Pick the longest duration first (ties by id).
    Info *pick = nullptr;
    unsigned pickIdx = 0;
    for (unsigned k = 0; k < pending.size(); ++k) {
      Info *c = pending[k];
      bool ready = true;
      for (int64_t p : preds.lookup(c->id))
        if (!finishOf.count(p)) {
          ready = false;
          break;
        }
      if (!ready)
        continue;
      if (!pick || c->dur > pick->dur || (c->dur == pick->dur && c->id < pick->id)) {
        pick = c;
        pickIdx = k;
      }
    }
    if (!pick)
      break; // a cycle (cannot happen on a legal DAG)
    pending.erase(pending.begin() + pickIdx);

    int64_t readyT = t0;
    for (int64_t p : preds.lookup(pick->id))
      readyT = std::max(readyT, finishOf.lookup(p));
    int64_t lo = 0, hi = pick->bandwidth ? knee : domains; // eligible streams [lo, hi)
    if (pick->sparse) {
      lo = domains;
      hi = domains + 1;
    }
    int64_t bestD = lo;
    int64_t bestFinish = std::numeric_limits<int64_t>::max(), bestScore = -1;
    for (int64_t d = lo; d < hi; ++d) {
      int64_t start = std::max(domainFree[d], readyT);
      int64_t finish = saturatingAddNonnegative(start, pick->dur);
      int64_t score = 0;
      for (StringRef r : pick->rids)
        if (resident[d].count(r))
          ++score;
      // key = (finish, -score, d) ascending.
      if (finish < bestFinish || (finish == bestFinish && score > bestScore)) {
        bestFinish = finish;
        bestScore = score;
        bestD = d;
      }
    }
    int64_t start = std::max(domainFree[bestD], readyT);
    int64_t finish = saturatingAddNonnegative(start, pick->dur);
    domainFree[bestD] = finish;
    for (StringRef r : pick->rids)
      resident[bestD].insert(r);
    finishOf[pick->id] = finish;
    slots[pick->id] = Slot{bestD == domains ? kTailStream : bestD, start, finish};
  }
}

// gem.schedule.schedule_eft: phase-barriered placement over the canonical phase order. Each
// phase builds the hazard DAG over ALL its claims (tail included) and dispatches them in one
// loop; phases compose at the barrier. Returns the makespan; `slots` receives every claim.
inline int64_t placeBarriered(ArrayRef<int32_t> phaseIds, SmallVector<Info> &infos, int64_t domains,
                              int64_t knee, llvm::DenseMap<int64_t, Slot> &slots) {
  SmallVector<llvm::DenseSet<StringRef>> resident(domains + 1); // locality memory, persists
  int64_t t0 = 0;
  for (int32_t pid : phaseIds) {
    SmallVector<Info *> phaseClaims;
    for (Info &in : infos)
      if (in.phase == pid)
        phaseClaims.push_back(&in);
    std::sort(phaseClaims.begin(), phaseClaims.end(),
              [](Info *a, Info *z) { return a->id < z->id; });
    llvm::DenseMap<int64_t, SmallVector<int64_t>> preds = hazardPredecessors(phaseClaims);
    SmallVector<int64_t> domainFree(domains + 1, t0);
    llvm::DenseMap<int64_t, int64_t> finishOf;
    eftDispatch(phaseClaims, preds, t0, domains, knee, domainFree, resident, finishOf, slots);
    for (auto &kv : finishOf)
      t0 = std::max(t0, kv.second);
  }
  return t0;
}

// gem.async_tokens.async_plan + gem.schedule.execute_tokens: the fork order is the canonical
// phase order then claim id; the awaits are the hazard DAG over the whole module (no phase
// barriers); one dispatch places everything. Returns the makespan; `awaits` receives each
// claim's awaited ids and `slots` every claim.
inline int64_t placeTokens(ArrayRef<int32_t> phaseIds, SmallVector<Info> &infos, int64_t domains,
                           int64_t knee, llvm::DenseMap<int64_t, SmallVector<int64_t>> &awaits,
                           llvm::DenseMap<int64_t, Slot> &slots) {
  SmallVector<Info *> flat;
  for (int32_t pid : phaseIds) {
    SmallVector<Info *> ph;
    for (Info &in : infos)
      if (in.phase == pid)
        ph.push_back(&in);
    std::sort(ph.begin(), ph.end(), [](Info *a, Info *z) { return a->id < z->id; });
    flat.append(ph.begin(), ph.end());
  }
  awaits = hazardPredecessors(flat);
  SmallVector<int64_t> domainFree(domains + 1, 0);
  SmallVector<llvm::DenseSet<StringRef>> resident(domains + 1);
  llvm::DenseMap<int64_t, int64_t> finishOf;
  eftDispatch(flat, awaits, 0, domains, knee, domainFree, resident, finishOf, slots);
  int64_t makespan = 0;
  for (auto &kv : finishOf)
    makespan = std::max(makespan, kv.second);
  return makespan;
}

// M(pi,Theta) of a candidate-index assignment (overlap._makespan): the phase-barriered
// placement's makespan over the assignment's coupled step costs. A pure function of the
// assignment, so the re-selection sweep can re-price trial assignments.
inline int64_t makespanOf(const std::vector<cm::Column> &cols, ArrayRef<int> assign, int64_t theta,
                          ArrayRef<int64_t> w, ArrayRef<int32_t> phaseIds, int64_t domains,
                          int64_t knee) {
  SmallVector<Info> infos = buildInfos(cols, assign, theta, w);
  llvm::DenseMap<int64_t, Slot> slots;
  return placeBarriered(phaseIds, infos, domains, knee, slots);
}

} // namespace sched
} // namespace bcir

#endif // BCIR_LIB_PASSES_BCIRSCHEDULE_H
