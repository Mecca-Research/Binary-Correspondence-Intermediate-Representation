//===- BCIRScheduleEftPass.cpp - duration-aware EFT + pipelined token scheduling -*- C++ -*-===//
//
// Three passes over the ONE canonical schedule artifact (BCIRSchedule.h, the law-rail twin of
// gem.schedule.schedule_plan -- G1 / S1-A), built on per-claim durations from the K_BCIR plan
// (the coupled step costs) and the hazard DAG over every claim of a phase (data hazards AND
// ordering fences, built before the stream split):
//
//   -bcir-schedule-eft : gem.schedule.schedule_eft -- phase-barriered HEFT-lite waves. Per phase
//                        (canonical order): LPT priority, EFT stream placement, locality
//                        tie-breaks, the bandwidth-knee clamp, the GGG/random tail on its own
//                        stream inside the same dispatch (a dependent tail claim waits for its
//                        producer; an independent one overlaps); phases compose serially.
//                        Annotates kbcir.sched_domain/start/finish + makespan/knee.
//
//   -bcir-async        : gem.async_tokens.async_plan + schedule.execute_tokens -- the !bcir.token
//                        fork/await DAG (hazards and fences over the whole module) drives a
//                        SINGLE cross-phase dispatch (no phase barriers), so an independent claim
//                        of a later phase overlaps an earlier one -- software pipelining falls
//                        out of the dependency structure. Annotates kbcir.async_awaits (the
//                        awaited claim ids) + async_domain/start/finish + async_makespan. The
//                        phase-barriered schedule is its degenerate case.
//
//   -bcir-power-rail   : gem.schedule.schedule_power_rail -- a per-slot DVFS overlay on the EFT
//                        placed timeline (the join of -bcir-schedule-eft and -bcir-dvfs). Each
//                        scheduled slot is classified by its base compute:memory mix and gets a
//                        per-slot Q8 clock for its [start,finish) interval -- memory-bound slots
//                        downclock (power saved at no throughput cost), keying off the real slot
//                        intervals rather than -bcir-dvfs's per-phase totals. Annotates
//                        kbcir.rail_class / rail_clock (per slot) + rail_energy_saved (per module).
//
// -bcir-overlap prices the same placement (its makespan is M(pi,Theta)), so the objective and
// the executor annotations never describe two schedules.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRPasses.h"
#include "BCIRCostModel.h"
#include "BCIRSchedule.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <string>

using namespace mlir;

namespace bcir {
namespace {

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

// Shared front end: plan the module and build the per-claim Info list. Returns false if the
// module is not plannable (no capability / weights / claims).
static bool planInfos(const cm::PlanAnalysis &pa, SmallVector<sched::Info> &infos, int64_t &domains,
                      int64_t &knee, int64_t &theta) {
  if (!pa.valid)
    return false;
  theta = pa.thetaThermal;
  domains = pa.affinityDomains;
  knee = sched::kneeOf(pa);
  infos = sched::buildInfos(pa.cols, pa.chosen, theta, pa.weights);
  return true;
}

// Annotate every placed claim with kbcir.<prefix>_domain / _start / _finish from the artifact.
static void annotateSlots(const SmallVector<sched::Info> &infos,
                          const llvm::DenseMap<int64_t, sched::Slot> &slots, Builder &b,
                          StringRef prefix) {
  std::string p = prefix.str();
  std::string dn = "kbcir." + p + "_domain";
  std::string sn = "kbcir." + p + "_start";
  std::string fn = "kbcir." + p + "_finish";
  for (const sched::Info &in : infos) {
    auto it = slots.find(in.id);
    if (it == slots.end())
      continue;
    in.claim->setAttr(dn, b.getI64IntegerAttr(it->second.domain));
    in.claim->setAttr(sn, b.getI64IntegerAttr(it->second.start));
    in.claim->setAttr(fn, b.getI64IntegerAttr(it->second.finish));
  }
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
    SmallVector<sched::Info> infos;
    int64_t domains = 1, knee = 1, theta = 0;
    if (!planInfos(pa, infos, domains, knee, theta))
      return;
    llvm::DenseMap<int64_t, sched::Slot> slots;
    int64_t makespan = sched::placeBarriered(canonicalPhaseIds(root), infos, domains, knee, slots);
    annotateSlots(infos, slots, b, "sched");
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
    SmallVector<sched::Info> infos;
    int64_t domains = 1, knee = 1, theta = 0;
    if (!planInfos(pa, infos, domains, knee, theta))
      return;
    // The fork order is the canonical phase order then claim id; each claim awaits the EARLIER
    // claims it has a hazard with -- data conflicts and the ordering fences, across phases (no
    // phase barriers) -- and one dispatch places everything: independent later-phase claims
    // start at 0 and overlap earlier phases (schedule.execute_tokens).
    llvm::DenseMap<int64_t, SmallVector<int64_t>> awaits;
    llvm::DenseMap<int64_t, sched::Slot> slots;
    int64_t makespan =
        sched::placeTokens(canonicalPhaseIds(root), infos, domains, knee, awaits, slots);
    for (sched::Info &in : infos)
      in.claim->setAttr("kbcir.async_awaits", b.getI64ArrayAttr(awaits.lookup(in.id)));
    annotateSlots(infos, slots, b, "async");
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
    SmallVector<sched::Info> infos;
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

    // The placed timeline is exactly schedule_eft's (the canonical artifact); the rail keys its
    // per-slot clock off each slot's real [start,finish) interval (vs -bcir-dvfs's totals).
    llvm::DenseMap<int64_t, sched::Slot> slots;
    int64_t makespan = sched::placeBarriered(canonicalPhaseIds(root), infos, domains, knee, slots);
    annotateSlots(infos, slots, b, "rail");

    // Per slot: classify the claim's base mix, set its clock, and accumulate the modeled energy
    // saved by downclocking (sum of (nominal - clock) x interval, in milli of a nominal cycle).
    int64_t energySaved = 0;
    for (sched::Info &in : infos) {
      StringRef klass;
      int64_t clock = railClock(in.baseCompute, in.baseMemory, thermal, power, klass);
      in.claim->setAttr("kbcir.rail_class", b.getStringAttr(klass));
      in.claim->setAttr("kbcir.rail_clock", b.getI64IntegerAttr(clock));
      if (clock < kNominal) {
        sched::Slot iv = slots.lookup(in.id);
        int64_t dur = std::max<int64_t>(0, iv.finish - iv.start);
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
