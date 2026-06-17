//===- BCIRCimDvfsPass.cpp - recompute the CIM/PIM dispatch + DVFS clock decisions -*- C++ -*-===//
//
// -bcir-cim / -bcir-dvfs: the law-rail RECOMPUTE of the two GEM scheduling decisions, the way
// -bcir-cost-model recomputes cost. Where R14/R15 only *verify* a declared dispatch / clock,
// these passes compute the decision from first principles (the claim + the capability + Theta),
// so the law no longer trusts the emitter's choice -- it derives it.
//
//   -bcir-cim : ports gem.cim.cim_decision. For a reduction over `count` elements, model
//               core_cost (stream every operand byte + reduce) vs pim_cost (surcharged
//               in-memory compute + a fixed dispatch setup + a 1-element result), and annotate
//               kbcir.cim_offload / cim_core_cost / cim_pim_cost. Offload iff pim strictly wins
//               (large reductions); non-reductions are never annotated (PIM does element-local
//               reduce work, not general SIMD).
//
//   -bcir-dvfs: ports gem.dvfs (classify + clock_for). It plans the module (the shared cost
//               model), sums each phase's (compute, memory) over the chosen realizations, and
//               sets a per-phase Q8 clock: downclock a memory-bound phase (power saved, the
//               bandwidth-bound throughput unchanged), overclock a compute-bound one (unless
//               Theta is thermal/power capped), nominal otherwise. Annotates the bcir.phase
//               with kbcir.dvfs_class / dvfs_clock.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIROps.h"
#include "BCIRCostModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <utility>

using namespace mlir;

namespace bcir {
namespace {

// gem.cim: PIM does element-local compute slower than the core (Q8 384 == x1.5) but moves no
// operand bytes; a fixed dispatch setup amortizes only over large reductions.
static constexpr int64_t kPimComputeQ8 = 384;
static constexpr int64_t kPimDispatchOverhead = 4096;

// gem.dvfs Q8 clocks + intensity thresholds + the Theta safety caps.
static constexpr int64_t kNominal = 256, kOverclock = 320, kDownclock = 192;
static constexpr int64_t kHiIntensity = 1000, kLoIntensity = 250;
static constexpr int64_t kThermalCap = 70, kPowerCap = 70;

static bool isReduction(StringRef op) {
  return op.starts_with("reduce.") || op.starts_with("reduce_");
}

struct CimPass : public PassWrapper<CimPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CimPass)
  StringRef getArgument() const final { return "bcir-cim"; }
  StringRef getDescription() const final {
    return "Recompute the CIM/PIM dispatch decision (gem.cim.cim_decision) -- core vs PIM cost "
           "for a reduction; annotates kbcir.cim_offload / cim_core_cost / cim_pim_cost.";
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
    if (!pa.capOp)
      return; // no target declared
    const cm::Cap &h = pa.h;
    root->walk([&](ClaimOp c) {
      if (!isReduction(c.getOp()))
        return; // non-reductions are never offloaded
      int64_t count = std::max<int64_t>(1, static_cast<int64_t>(c.getCount()));
      int64_t eb = h.elemBytes;            // operand element width (the capability's)
      int64_t traffic = count * eb;        // operand bytes streamed to the core
      int64_t compute = count * h.memUnit; // per-element reduce work
      int64_t coreCost = traffic + compute;
      int64_t pimCost = ((compute * kPimComputeQ8) >> 8) + kPimDispatchOverhead + eb;
      c->setAttr("kbcir.cim_offload", b.getBoolAttr(pimCost < coreCost));
      c->setAttr("kbcir.cim_core_cost", b.getI64IntegerAttr(coreCost));
      c->setAttr("kbcir.cim_pim_cost", b.getI64IntegerAttr(pimCost));
    });
  }
};

struct DvfsPass : public PassWrapper<DvfsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DvfsPass)
  StringRef getArgument() const final { return "bcir-dvfs"; }
  StringRef getDescription() const final {
    return "Recompute the phase-aware DVFS clock (gem.dvfs) -- classify each phase by its "
           "compute:memory mix and set a Q8 clock (down/over/nominal) honoring Theta; "
           "annotates kbcir.dvfs_class / dvfs_clock.";
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
    if (!pa.valid)
      return;
    ArrayRef<int64_t> w = pa.weights;
    const std::vector<cm::Column> &cols = pa.cols;
    KBCIRThetaOp theta;
    root->walk([&](KBCIRThetaOp t) {
      if (!theta)
        theta = t;
    });
    // pa.thetaThermal is firstThetaThermal(root) == this first theta op's thermal.
    int64_t thermal = pa.thetaThermal;
    int64_t power = theta ? static_cast<int64_t>(theta.getPower()) : 0;
    const SmallVector<int> &chosen = pa.chosen;
    int64_t total = pa.total;

    // Per-phase (compute, memory) over the chosen realizations' base costs (COMPUTE=0,
    // MEMORY=1; gem.dvfs.phase_totals).
    llvm::DenseMap<int32_t, std::pair<int64_t, int64_t>> phaseCM;
    for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
      const cm::Cost &cost = cols[i].cands[chosen[i]].cost;
      std::pair<int64_t, int64_t> &pc = phaseCM[cols[i].phase];
      pc.first += cost[0];   // COMPUTE
      pc.second += cost[1];  // MEMORY
    }
    root->walk([&](PhaseOp p) {
      auto it = phaseCM.find(p.getId());
      int64_t compute = it != phaseCM.end() ? it->second.first : 0;
      int64_t memory = it != phaseCM.end() ? it->second.second : 0;
      int64_t intensity = (compute * 1000) / std::max<int64_t>(1, memory);
      StringRef klass;
      int64_t clock;
      if (intensity >= kHiIntensity) {
        klass = "compute";
        clock = (thermal >= kThermalCap || power >= kPowerCap) ? kNominal : kOverclock;
      } else if (intensity <= kLoIntensity) {
        klass = "memory";
        clock = kDownclock;
      } else {
        klass = "balanced";
        clock = kNominal;
      }
      p->setAttr("kbcir.dvfs_class", b.getStringAttr(klass));
      p->setAttr("kbcir.dvfs_clock", b.getI64IntegerAttr(clock));
    });
  }
};

} // namespace

std::unique_ptr<Pass> createCimPass() { return std::make_unique<CimPass>(); }
std::unique_ptr<Pass> createDvfsPass() { return std::make_unique<DvfsPass>(); }

} // namespace bcir
