//===- BCIRPasses.cpp - BCIR pass registration ----------------------------===//
//
// Central registration for the modular BCIR MLIR pass library. The passes
// themselves live in lib/passes/*.cpp (one translation unit per pass group),
// sharing lib/passes/BCIRPassSupport.h; this file only wires their factories into
// the global pass registry so `bcir-opt` exposes the CLI flags. Registering via the
// factory callback keeps each pass struct private to its own TU.
//
//   -bcir-verify            R1-R16 semantic laws            (BCIRVerifyPass.cpp)
//   -bcir-promote-lanes     GGG->UX opt-law rewrite         (BCIRPromotePass.cpp)
//   -convert-bcir-to-llvm   compute/barrier -> LLVM dialect (BCIRConvertToLLVM.cpp)
//   -bcir-classify-lanes /  the GEM pipeline                (BCIRGEMPasses.cpp)
//   -bcir-batch / -bcir-schedule / -bcir-lower-to-llvm
//   -bcir-select-realization  min-plus selection            (BCIRSelectPass.cpp)
//   -bcir-cost-model        K_BCIR cost algebra (cost.py)    (BCIRCostModel.cpp)
//   -bcir-plan              min-plus shortest path (optimize) (BCIRPlanPass.cpp)
//   -bcir-overlap           (max,+) scheduled price M(pi,T)  (BCIROverlapPass.cpp)
//   -bcir-rcsp              RCSP / Pareto (optimizer core)   (BCIRRcspPass.cpp)
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRPasses.h"

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
  ::mlir::registerPass([] { return createRcspPass(); });
  ::mlir::registerPass([] { return createBatchPass(); });
  ::mlir::registerPass([] { return createSchedulePass(); });
  ::mlir::registerPass([] { return createLowerToLLVMPass(); });
}

}  // namespace bcir
