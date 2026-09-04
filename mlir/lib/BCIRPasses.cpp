//===- BCIRPasses.cpp - BCIR pass registration ----------------------------===//
//
// Central registration for the modular BCIR MLIR pass library. The passes
// themselves live in lib/passes/*.cpp (one translation unit per pass group),
// sharing lib/passes/BCIRPassSupport.h; this file only wires their factories into
// the global pass registry so `bcir-opt` exposes the CLI flags. Registering via the
// factory callback keeps each pass struct private to its own TU.
//
//   -bcir-verify            R1-R23 semantic laws            (BCIRVerifyPass.cpp)
//   -bcir-promote-lanes     GGG->UX opt-law rewrite         (BCIRPromotePass.cpp)
//   -convert-bcir-to-llvm   compute/barrier -> LLVM dialect (BCIRConvertToLLVM.cpp)
//   -bcir-classify-lanes /  the GEM pipeline                (BCIRGEMPasses.cpp)
//   -bcir-batch / -bcir-schedule / -bcir-lower-to-llvm
//   -bcir-select-realization  min-plus selection            (BCIRSelectPass.cpp)
//   -bcir-cost-model        K_BCIR cost algebra (cost.py)    (BCIRCostModel.cpp)
//   -bcir-plan              min-plus shortest path (optimize) (BCIRPlanPass.cpp)
//   -bcir-overlap           (max,+) scheduled price M(pi,T)  (BCIROverlapPass.cpp)
//   -bcir-rcsp              RCSP / Pareto (per-claim)        (BCIRRcspPass.cpp)
//   -bcir-rcsp-plan         plan-level accumulated-budget DP (BCIRRcspPlanPass.cpp)
//
// registerBCIRPipelines() wires these into declared input/output-level pipelines
// with verifier checkpoints (so callers stop hand-ordering passes):
//   bcir-audit       verify -> classify -> cost-model -> plan -> overlap (read-only)
//   bcir-optimize    classify -> cost-model -> plan        (claims+H -> coupled plan)
//   bcir-hydrate     classify -> select -> batch -> schedule -> lower  (plan -> StreamPack)
//   bcir-lower-llvm  verify -> convert-bcir-to-llvm        (-> LLVM dialect)
//   bcir-aot         partial AOT preparation; mixed BCIR/GEM/LLVM dialect IR
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace bcir {

void registerBCIRPasses() {
  ::mlir::registerPass([] { return createVerifyPass(); });
  ::mlir::registerPass([] { return createPromoteLanesPass(); });
  ::mlir::registerPass([] { return createConvertToLLVMPass(); });
  ::mlir::registerPass([] { return createClassifyLanesPass(); });
  ::mlir::registerPass([] { return createSelectRealizationPass(); });
  ::mlir::registerPass([] { return createCostModelPass(); });
  ::mlir::registerPass([] { return createPlanPass(); });
  ::mlir::registerPass([] { return createOverlapPass(); });
  ::mlir::registerPass([] { return createOverlapOptimizePass(); });
  ::mlir::registerPass([] { return createSensePass(); });
  ::mlir::registerPass([] { return createRcspPass(); });
  ::mlir::registerPass([] { return createRcspPlanPass(); });
  ::mlir::registerPass([] { return createBundlePass(); });
  ::mlir::registerPass([] { return createExplainPass(); });
  ::mlir::registerPass([] { return createReplayPass(); });
  ::mlir::registerPass([] { return createComposePass(); });
  ::mlir::registerPass([] { return createCimPass(); });
  ::mlir::registerPass([] { return createDvfsPass(); });
  ::mlir::registerPass([] { return createScheduleEftPass(); });
  ::mlir::registerPass([] { return createAsyncPass(); });
  ::mlir::registerPass([] { return createPowerRailPass(); });
  ::mlir::registerPass([] { return createAllocPoolPass(); });
  ::mlir::registerPass([] { return createBatchPass(); });
  ::mlir::registerPass([] { return createSchedulePass(); });
  ::mlir::registerPass([] { return createLowerToLLVMPass(); });
  ::mlir::registerPass([] { return createGemMatmulCostPass(); });
  ::mlir::registerPass([] { return createGemActivationCostPass(); });
  ::mlir::registerPass([] { return createGemConvCostPass(); });
  ::mlir::registerPass([] { return createGemAttentionCostPass(); });
  ::mlir::registerPass([] { return createLowerGemMatmulPass(); });
  ::mlir::registerPass([] { return createLowerGemMatmulBufferPass(); });
  ::mlir::registerPass([] { return createLowerGemActivationPass(); });
  ::mlir::registerPass([] { return createLowerGemConvPass(); });
  ::mlir::registerPass([] { return createLowerGemAttentionPass(); });
  ::mlir::registerPass([] { return createFuseMatmulActivationPass(); });
  ::mlir::registerPass([] { return createCacheContentionPass(); });
  ::mlir::registerPass([] { return createLayoutPivotPass(); });
}

void registerBCIRPipelines() {
  using ::mlir::OpPassManager;
  using ::mlir::PassPipelineRegistration;

  // The GEM hydrate chain (plan -> StreamPack), reused by aot.
  auto hydrate = [](OpPassManager &pm) {
    pm.addPass(createClassifyLanesPass());
    pm.addPass(createSelectRealizationPass());
    pm.addPass(createBatchPass());
    pm.addPass(createSchedulePass());
    pm.addPass(createLowerToLLVMPass());
  };

  // Read-only analysis + the R1-R23 verifier checkpoint up front: recompute the cost
  // model, the coupled plan, and the scheduled price (no IR lowering).
  static PassPipelineRegistration<> audit(
      "bcir-audit", "Verify (R1-R23) then recompute cost/plan/overlap annotations (read-only).",
      [](OpPassManager &pm) {
        pm.addPass(createVerifyPass());
        pm.addPass(createClassifyLanesPass());
        pm.addPass(createCostModelPass());
        pm.addPass(createPlanPass());
        pm.addPass(createOverlapPass());
      });

  // claims + capability -> the coupled K_BCIR plan (the optimize pipeline; named
  // bcir-optimize because -bcir-plan is the single-pass flag). Verifier-checkpointed on
  // entry AND on the plan it emits (S0-3): the pipelines advertised as checkpointed were
  // the only two that ran no verifier at all, so an illegal module planned and hydrated.
  static PassPipelineRegistration<> optimize(
      "bcir-optimize",
      "verify -> classify -> cost-model -> plan -> verify: the coupled K_BCIR plan from "
      "first principles, checkpointed on entry and on the plan it emits.",
      [](OpPassManager &pm) {
        pm.addPass(createVerifyPass());
        pm.addPass(createClassifyLanesPass());
        pm.addPass(createCostModelPass());
        pm.addPass(createPlanPass());
        pm.addPass(createVerifyPass());
      });

  // verify checkpoint -> declared plan -> hydrated, lane-lowered GEM StreamPack
  // (R12/R14-R16 checked in the lowering).
  static PassPipelineRegistration<> hydratePipe(
      "bcir-hydrate",
      "verify -> classify -> select -> batch -> schedule -> lower: plan -> GEM StreamPack, "
      "checkpointed on entry.",
      [hydrate](OpPassManager &pm) {
        pm.addPass(createVerifyPass());
        hydrate(pm);
      });

  // verify checkpoint -> BCIR compute/barrier lowered to the LLVM dialect.
  static PassPipelineRegistration<> lowerLLVM(
      "bcir-lower-llvm", "Verify then lower BCIR compute/barrier to the LLVM dialect.",
      [](OpPassManager &pm) {
        pm.addPass(createVerifyPass());
        pm.addPass(createConvertToLLVMPass());
      });

  // Partial AOT preparation: hydration and conversion intentionally leave
  // unsupported BCIR/GEM operations in the mixed-dialect result.
  static PassPipelineRegistration<> aot(
      "bcir-aot", "Partial AOT preparation producing mixed BCIR/GEM/LLVM dialect IR.",
      [hydrate](OpPassManager &pm) {
        pm.addPass(createVerifyPass());
        hydrate(pm);
        pm.addPass(createConvertToLLVMPass());
      });
}

} // namespace bcir
