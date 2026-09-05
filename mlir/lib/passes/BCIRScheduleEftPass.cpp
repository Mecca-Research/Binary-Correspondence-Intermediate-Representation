//===- BCIRScheduleEftPass.cpp - duration-aware EFT + pipelined token scheduling -*- C++ -*-===//
//
// Two duration-aware schedulers sharing the same earliest-finish-time placement (the law-rail
// port of gem.schedule), built on per-claim durations from the K_BCIR plan:
//
//   -bcir-schedule-eft : gem.schedule.schedule_eft -- phase-barriered HEFT-lite waves. Per phase
//                        (topo order): LPT priority, EFT domain placement, locality tie-breaks,
//                        the bandwidth-knee clamp; the GGG/random tail runs decoupled; phases
//                        compose serially. Annotates kbcir.sched_domain/start/finish + makespan/knee.
//
//   -bcir-async        : gem.async_tokens.async_plan + schedule.execute_tokens -- the !bcir.token
//                        fork/await DAG drives a SINGLE cross-phase dispatch (no phase barriers),
//                        so an independent claim of a later phase overlaps an earlier one --
//                        software pipelining falls out of the dependency structure. Annotates
//                        kbcir.async_awaits (the awaited claim ids) + async_domain/start/finish +
//                        async_makespan. The phase-barriered schedule is its degenerate case.
//
//   -bcir-power-rail   : gem.schedule.schedule_power_rail -- a per-slot DVFS overlay on the EFT
//                        placed timeline (the join of -bcir-schedule-eft and -bcir-dvfs). Each
//                        scheduled slot is classified by its base compute:memory mix and gets a
//                        per-slot Q8 clock for its [start,finish) interval -- memory-bound slots
//                        downclock (power saved at no throughput cost), keying off the real slot
//                        intervals rather than -bcir-dvfs's per-phase totals. Annotates
//                        kbcir.rail_class / rail_clock (per slot) + rail_energy_saved (per module).
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRPasses.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

static constexpr int kTailStream = -1; // the decoupled GGG/random tail's own stream

// gem.dvfs Q8 clocks + intensity thresholds + Theta safety caps (shared with -bcir-dvfs).
static constexpr int64_t kNominal = 256, kOverclock = 320, kDownclock = 192;
static constexpr int64_t kHiIntensity = 1000, kLoIntensity = 250;
static constexpr int64_t kThermalCap = 70, kPowerCap = 70;

// gem.dvfs.classify + clock_for: the Q8 clock for a slot's (compute, memory) base mix under Theta.
// Downclock a memory-bound slot (bandwidth-bound -> power saved, throughput unaffected); overclock
// a compute-bound one unless Theta is thermal/power capped; nominal otherwise.
static int64_t railClock(int64_t compute, int64_t memory, int64_t thermal, int64_t power,
                         StringRef &klass) {
  int64_t intensity = saturatingMulNonnegative(compute, 1000) / std::max<int64_t>(1, memory);
  if (intensity >= kHiIntensity) {
    klass = "compute";
    return (thermal >= kThermalCap || power >= kPowerCap) ? kNominal : kOverclock;
  }
  if (intensity <= kLoIntensity) {
    klass = "memory";
    return kDownclock;
  }
  klass = "balanced";
  return kNominal;
}

static bool isSparse(ClaimOp c) {
  return c.getLane() == Lane::GGG || c.getStrideClass() == StrideClass::Random;
}

// gem.concurrency._conflict: a RAW/WAR/WAW between two claims' read/write symbol sets.
static bool conflict(ArrayRef<StringRef> ar, ArrayRef<StringRef> aw, ArrayRef<StringRef> br,
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

struct Info {
  ClaimOp claim;
  int32_t phase;
  int64_t id;
  int64_t dur;
  int64_t baseCompute, baseMemory; // the chosen candidate's base cost (for the power-rail classify)
  bool bandwidth;                  // cost_class == bandwidth (the default) -> knee-clamped
  SmallVector<StringRef> reads, writes;
  llvm::DenseSet<StringRef> rids; // _rids = set(rd) | set(wr) -- deduped, for locality
};

// Per-claim durations from the K_BCIR plan: the chosen coupled edge cost (durations_from).
static SmallVector<Info> buildInfos(const std::vector<cm::Column> &cols, ArrayRef<int> chosen,
                                    int64_t theta, ArrayRef<int64_t> w) {
  SmallVector<Info> infos;
  for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
    cm::Cost e = cols[i].cands[chosen[i]].cost;
    cm::Factor f =
        (i > 0)
            ? cm::contextFactor(theta, cols[i - 1].reads, cols[i - 1].cands[chosen[i - 1]].width,
                                cols[i].reads, cols[i].cands[chosen[i]].width)
            : cm::contextFactor(theta, {}, 0, cols[i].reads, cols[i].cands[chosen[i]].width);
    cm::applyFactor(e, f);
    Info in;
    in.claim = cols[i].claim;
    in.phase = cols[i].phase;
    in.id = static_cast<int64_t>(in.claim.getClaimId());
    in.dur = std::max<int64_t>(1, cm::scalarize(e, w));
    // The base (uncoupled) compute/memory of the chosen candidate -- gem.dvfs.classify keys off
    // this, not the contextual duration (COMPUTE=0, MEMORY=1).
    in.baseCompute = cols[i].cands[chosen[i]].cost[0];
    in.baseMemory = cols[i].cands[chosen[i]].cost[1];
    in.bandwidth = !in.claim.getCostClass() || *in.claim.getCostClass() == CostClass::Bandwidth;
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

// gem.schedule._dispatch: event-driven LPT list scheduling onto the affinity domains, placing
// each claim at the earliest finish time (locality + knee), annotating kbcir.<prefix>_*.
static void eftDispatch(ArrayRef<Info *> claims,
                        const llvm::DenseMap<int64_t, SmallVector<int64_t>> &preds, int64_t t0,
                        int64_t domains, int64_t knee, SmallVector<int64_t> &domainFree,
                        SmallVector<llvm::DenseSet<StringRef>> &resident,
                        llvm::DenseMap<int64_t, int64_t> &finishOf, Builder &b, StringRef prefix) {
  std::string p = prefix.str();
  std::string an = "kbcir." + p + "_domain";
  std::string sn = "kbcir." + p + "_start";
  std::string fn = "kbcir." + p + "_finish";
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
      break; // a cycle (shouldn't happen on a legal DAG)
    pending.erase(pending.begin() + pickIdx);

    int64_t readyT = t0;
    for (int64_t p : preds.lookup(pick->id))
      readyT = std::max(readyT, finishOf.lookup(p));
    int64_t width = pick->bandwidth ? knee : domains;

    int bestD = 0;
    int64_t bestFinish = std::numeric_limits<int64_t>::max(), bestScore = -1;
    for (int64_t d = 0; d < width; ++d) {
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
        bestD = static_cast<int>(d);
      }
    }
    int64_t start = std::max(domainFree[bestD], readyT);
    int64_t finish = saturatingAddNonnegative(start, pick->dur);
    domainFree[bestD] = finish;
    for (StringRef r : pick->rids)
      resident[bestD].insert(r);
    finishOf[pick->id] = finish;
    pick->claim->setAttr(an, b.getI64IntegerAttr(bestD));
    pick->claim->setAttr(sn, b.getI64IntegerAttr(start));
    pick->claim->setAttr(fn, b.getI64IntegerAttr(finish));
  }
}

// gem.concurrency._topo_phase_ids: phases in dependency order (deps first) -- the ONE
// canonical order of BCIRPassSupport.h (S0-6: shared with -bcir-schedule, -bcir-overlap and
// the cost-model columns; this pass carried its own recursive port before).
static SmallVector<int32_t> topoPhases(Operation *root) {
  return canonicalPhaseIds(root);
}

// Shared front end: plan the module and build the per-claim Info list. Returns false if the
// module is not plannable (no capability / weights / claims).
static bool planInfos(const cm::PlanAnalysis &pa, SmallVector<Info> &infos, int64_t &domains,
                      int64_t &knee, int64_t &theta) {
  if (!pa.valid)
    return false;
  theta = pa.thetaThermal;
  domains = pa.affinityDomains;
  TargetCapabilityOp cap = pa.capOp; // value-handle copy for the non-const accessor
  knee = std::max<int64_t>(1, std::min<int64_t>(domains, cap.getMemChannels()));
  infos = buildInfos(pa.cols, pa.chosen, theta, pa.weights);
  return true;
}

// gem.schedule.schedule_eft: phase-barriered EFT waves over the topo-ordered phases. Each phase
// dispatches its main claims (LPT + EFT placement + locality, knee-clamped) and runs its sparse
// GGG/random tail serially on its own stream; phases compose at the barrier. Annotates each claim
// kbcir.<prefix>_domain / _start / _finish, records [start,finish) per claim id in `intervals`
// when non-null, and returns the makespan.
static int64_t placeBarriered(Operation *root, SmallVector<Info> &infos, int64_t domains,
                              int64_t knee, Builder &b, StringRef prefix,
                              llvm::DenseMap<int64_t, std::pair<int64_t, int64_t>> *intervals) {
  std::string p = prefix.str();
  std::string dn = "kbcir." + p + "_domain";
  std::string sn = "kbcir." + p + "_start";
  std::string fn = "kbcir." + p + "_finish";
  SmallVector<llvm::DenseSet<StringRef>> resident(domains); // locality memory, persists
  int64_t t0 = 0, makespan = 0;
  for (int32_t pid : topoPhases(root)) {
    SmallVector<Info *> phaseClaims;
    for (Info &in : infos)
      if (in.phase == pid)
        phaseClaims.push_back(&in);
    std::sort(phaseClaims.begin(), phaseClaims.end(),
              [](Info *a, Info *z) { return a->id < z->id; });
    SmallVector<Info *> main, tail;
    for (Info *in : phaseClaims)
      (isSparse(in->claim) ? tail : main).push_back(in);

    // Intra-phase hazard DAG (lower claim id is the producer of a conflicting pair).
    llvm::DenseMap<int64_t, SmallVector<int64_t>> preds;
    for (unsigned i = 0; i < main.size(); ++i)
      for (unsigned j = 0; j < i; ++j)
        if (conflict(main[j]->reads, main[j]->writes, main[i]->reads, main[i]->writes))
          preds[main[i]->id].push_back(main[j]->id);

    SmallVector<int64_t> domainFree(domains, t0);
    llvm::DenseMap<int64_t, int64_t> finishOf;
    eftDispatch(main, preds, t0, domains, knee, domainFree, resident, finishOf, b, prefix);
    if (intervals)
      for (Info *in : main) {
        int64_t f = finishOf.lookup(in->id);
        (*intervals)[in->id] = {f - in->dur, f}; // start == finish - dur (eftDispatch invariant)
      }

    int64_t tt = t0;
    for (Info *in : tail) { // the decoupled tail: a serial chain on its own stream
      in->claim->setAttr(dn, b.getI64IntegerAttr(kTailStream));
      in->claim->setAttr(sn, b.getI64IntegerAttr(tt));
      int64_t finish = saturatingAddNonnegative(tt, in->dur);
      in->claim->setAttr(fn, b.getI64IntegerAttr(finish));
      if (intervals)
        (*intervals)[in->id] = {tt, finish};
      tt = finish;
    }
    int64_t next = std::max(t0, tt);
    for (auto &kv : finishOf)
      next = std::max(next, kv.second);
    t0 = next;
    makespan = next;
  }
  return makespan;
}

struct ScheduleEftPass : public PassWrapper<ScheduleEftPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleEftPass)
  StringRef getArgument() const final { return "bcir-schedule-eft"; }
  StringRef getDescription() const final {
    return "Duration-aware EFT wave scheduling (gem.schedule.schedule_eft): LPT priority + "
           "earliest-finish placement + locality + the bandwidth-knee clamp; annotates "
           "kbcir.sched_domain / sched_start / sched_finish / sched_makespan / sched_knee.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    SmallVector<Info> infos;
    int64_t domains = 1, knee = 1, theta = 0;
    if (!planInfos(pa, infos, domains, knee, theta))
      return;
    int64_t makespan = placeBarriered(root, infos, domains, knee, b, "sched", nullptr);
    root->setAttr("kbcir.sched_makespan", b.getI64IntegerAttr(makespan));
    root->setAttr("kbcir.sched_knee", b.getI64IntegerAttr(knee));
  }
};

struct AsyncPass : public PassWrapper<AsyncPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AsyncPass)
  StringRef getArgument() const final { return "bcir-async"; }
  StringRef getDescription() const final {
    return "Async token plan + pipelined schedule (gem.async_tokens.async_plan + "
           "schedule.execute_tokens): the fork/await DAG drives a single cross-phase EFT "
           "dispatch; annotates kbcir.async_awaits / async_domain / async_start / async_finish "
           "/ async_makespan (later-phase claims overlap earlier ones -- software pipelining).";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    SmallVector<Info> infos;
    int64_t domains = 1, knee = 1, theta = 0;
    if (!planInfos(pa, infos, domains, knee, theta))
      return;

    // The fork order: claims in topo-phase order, then by id (gem.async_tokens.async_plan).
    SmallVector<Info *> flat;
    for (int32_t pid : topoPhases(root)) {
      SmallVector<Info *> ph;
      for (Info &in : infos)
        if (in.phase == pid)
          ph.push_back(&in);
      std::sort(ph.begin(), ph.end(), [](Info *a, Info *z) { return a->id < z->id; });
      flat.append(ph.begin(), ph.end());
    }

    // await DAG: each claim awaits the EARLIER (in fork order) claims it conflicts with --
    // the full cross-phase RAW/WAR/WAW dependency (no phase barriers).
    llvm::DenseMap<int64_t, SmallVector<int64_t>> awaits;
    for (unsigned i = 0; i < flat.size(); ++i) {
      SmallVector<int64_t> &aw = awaits[flat[i]->id];
      for (unsigned j = 0; j < i; ++j)
        if (conflict(flat[j]->reads, flat[j]->writes, flat[i]->reads, flat[i]->writes))
          aw.push_back(flat[j]->id);
      flat[i]->claim->setAttr("kbcir.async_awaits", b.getI64ArrayAttr(aw));
    }

    // A single dispatch over every claim, preds = the awaits -> independent later-phase claims
    // start at 0 and overlap earlier phases (schedule.execute_tokens).
    SmallVector<int64_t> domainFree(domains, 0);
    SmallVector<llvm::DenseSet<StringRef>> resident(domains);
    llvm::DenseMap<int64_t, int64_t> finishOf;
    eftDispatch(flat, awaits, 0, domains, knee, domainFree, resident, finishOf, b, "async");
    int64_t makespan = 0;
    for (auto &kv : finishOf)
      makespan = std::max(makespan, kv.second);
    root->setAttr("kbcir.async_makespan", b.getI64IntegerAttr(makespan));
  }
};

struct PowerRailPass : public PassWrapper<PowerRailPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PowerRailPass)
  StringRef getArgument() const final { return "bcir-power-rail"; }
  StringRef getDescription() const final {
    return "Per-slot DVFS over the EFT placed timeline (gem.schedule.schedule_power_rail): "
           "classify each scheduled slot by its base compute:memory mix and set a per-slot Q8 "
           "clock for its [start,finish) interval (downclock memory-bound slots, overclock "
           "compute-bound ones honoring Theta); annotates kbcir.rail_domain / rail_start / "
           "rail_finish / rail_class / rail_clock + rail_makespan / rail_knee / rail_energy_saved.";
  }

  void runOnOperation() override {
    Builder b(&getContext());
    getOperation()->walk([&](Operation *mod) {
      if (mod->getName().getStringRef() == "bcir.module")
        runOnModule(mod, b, getChildAnalysis<cm::PlanAnalysis>(mod));
    });
    markAnalysesPreserved<cm::PlanAnalysis>();
  }

  void runOnModule(Operation *root, Builder &b, const cm::PlanAnalysis &pa) {
    SmallVector<Info> infos;
    int64_t domains = 1, knee = 1, thermal = 0;
    if (!planInfos(pa, infos, domains, knee, thermal))
      return;
    // Theta power for the overclock safety gate (planInfos returns thermal only).
    KBCIRThetaOp tOp;
    root->walk([&](KBCIRThetaOp t) {
      if (!tOp)
        tOp = t;
    });
    int64_t power = tOp ? static_cast<int64_t>(tOp.getPower()) : 0;

    // The placed timeline is exactly schedule_eft's; the rail keys its per-slot clock off each
    // slot's real [start,finish) interval (vs -bcir-dvfs's per-phase totals).
    llvm::DenseMap<int64_t, std::pair<int64_t, int64_t>> intervals;
    int64_t makespan = placeBarriered(root, infos, domains, knee, b, "rail", &intervals);

    // Per slot: classify the claim's base mix, set its clock, and accumulate the modeled energy
    // saved by downclocking (sum of (nominal - clock) x interval, in milli of a nominal cycle).
    int64_t energySaved = 0;
    for (Info &in : infos) {
      StringRef klass;
      int64_t clock = railClock(in.baseCompute, in.baseMemory, thermal, power, klass);
      in.claim->setAttr("kbcir.rail_class", b.getStringAttr(klass));
      in.claim->setAttr("kbcir.rail_clock", b.getI64IntegerAttr(clock));
      if (clock < kNominal) {
        auto iv = intervals.lookup(in.id);
        int64_t dur = std::max<int64_t>(0, iv.second - iv.first);
        int64_t saved = saturatingMulNonnegative(kNominal - clock, dur);
        saved = saturatingMulNonnegative(saved, 1000) / kNominal;
        energySaved = saturatingAddNonnegative(energySaved, saved);
      }
    }
    root->setAttr("kbcir.rail_makespan", b.getI64IntegerAttr(makespan));
    root->setAttr("kbcir.rail_knee", b.getI64IntegerAttr(knee));
    root->setAttr("kbcir.rail_energy_saved", b.getI64IntegerAttr(energySaved));
  }
};

} // namespace

std::unique_ptr<Pass> createScheduleEftPass() {
  return std::make_unique<ScheduleEftPass>();
}
std::unique_ptr<Pass> createAsyncPass() {
  return std::make_unique<AsyncPass>();
}
std::unique_ptr<Pass> createPowerRailPass() {
  return std::make_unique<PowerRailPass>();
}

} // namespace bcir
