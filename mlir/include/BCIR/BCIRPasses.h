//===- BCIRPasses.h - BCIR compiler passes -----------------------*- C++ -*-===//
//
// Phase 6: the MLIR rail as a real compiler.
//   -bcir-verify           semantic laws R1-R16 as a module pass
//   -bcir-promote-lanes    the opt-law (GGG -> UX promotion) as a rewrite
//   -convert-bcir-to-llvm  BCIR compute/barrier -> LLVM dialect (TypeConverter + patterns)
//   -bcir-classify-lanes / -select-realization / -rcsp / -batch / -schedule /
//   -lower-to-llvm         the GEM pipeline + RCSP/Pareto (the optimizer core, C++23)
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_BCIRPASSES_H
#define BCIR_BCIRPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace bcir {

std::unique_ptr<mlir::Pass> createVerifyPass();
std::unique_ptr<mlir::Pass> createPromoteLanesPass();
std::unique_ptr<mlir::Pass> createConvertToLLVMPass();

// The GEM pipeline (LangRef Milestone 4..7): classify -> select -> batch ->
// schedule -> lower. MLIR-native implementations of the bcir/ oracle stages,
// cross-checked against its pinned constants (docs/PARITY.md).
std::unique_ptr<mlir::Pass> createClassifyLanesPass();
std::unique_ptr<mlir::Pass> createSelectRealizationPass();
// -bcir-cost-model: the K_BCIR cost algebra (cost.py) on the MLIR rail -- recompute
// candidate costs from claim + target.capability instead of trusting declared paths.
std::unique_ptr<mlir::Pass> createCostModelPass();
// -bcir-plan: the layered min-plus shortest path (the full realize.optimize in C++).
std::unique_ptr<mlir::Pass> createPlanPass();
// -bcir-overlap: the (max,+) scheduled price M(pi,Theta) (gem/overlap.py).
std::unique_ptr<mlir::Pass> createOverlapPass();
// -bcir-rcsp: constrained selection (budget label-DP) + the Pareto front, the
// deterministic optimizer core ported from bcir/kbcir/rcsp.py.
std::unique_ptr<mlir::Pass> createRcspPass();
// -bcir-rcsp-plan: plan-level constrained selection (accumulated-budget label-DP).
std::unique_ptr<mlir::Pass> createRcspPlanPass();
// -bcir-bundle: detect + joint-reorder multi-claim input-sharing bundles (kbcir.bundle).
std::unique_ptr<mlir::Pass> createBundlePass();
// -bcir-explain: the proof-carrying decision record (proof.explain) as IR annotations --
// per claim the candidates weighed + chosen width/score; per module the total plan score.
std::unique_ptr<mlir::Pass> createExplainPass();
// -bcir-compose: compositional cost over the kbcir.func/call/cond region tree
// (compose.plan_composite) -- annotates kbcir.compose_worst / compose_expected per func.
std::unique_ptr<mlir::Pass> createComposePass();
// -bcir-cim / -bcir-dvfs: recompute the CIM/PIM dispatch + DVFS clock decisions (gem.cim /
// gem.dvfs) from the IR, instead of R14/R15 only verifying a declared attr.
std::unique_ptr<mlir::Pass> createCimPass();
std::unique_ptr<mlir::Pass> createDvfsPass();
std::unique_ptr<mlir::Pass> createBatchPass();
std::unique_ptr<mlir::Pass> createSchedulePass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

/// Register all BCIR passes with the global pass registry (for bcir-opt).
void registerBCIRPasses();

/// Register the named pass pipelines (bcir-audit / -optimize / -hydrate /
/// -lower-llvm / -aot) with verifier checkpoints.
void registerBCIRPipelines();

}  // namespace bcir

#endif  // BCIR_BCIRPASSES_H
