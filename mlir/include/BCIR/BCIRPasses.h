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
// -bcir-rcsp: constrained selection (budget label-DP) + the Pareto front, the
// deterministic optimizer core ported from bcir/kbcir/rcsp.py.
std::unique_ptr<mlir::Pass> createRcspPass();
std::unique_ptr<mlir::Pass> createBatchPass();
std::unique_ptr<mlir::Pass> createSchedulePass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

/// Register all BCIR passes with the global pass registry (for bcir-opt).
void registerBCIRPasses();

}  // namespace bcir

#endif  // BCIR_BCIRPASSES_H
