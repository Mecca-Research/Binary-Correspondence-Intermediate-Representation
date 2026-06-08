//===- BCIRPasses.h - BCIR compiler passes -----------------------*- C++ -*-===//
//
// Phase 6: the MLIR rail as a real compiler.
//   -bcir-verify           semantic laws R1/R2/R4/R6 as a module pass
//   -bcir-promote-lanes    the opt-law (GGG -> UX promotion) as a rewrite
//   -convert-bcir-to-llvm  BCIR compute/barrier -> LLVM dialect (TypeConverter + patterns)
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

/// Register all BCIR passes with the global pass registry (for bcir-opt).
void registerBCIRPasses();

}  // namespace bcir

#endif  // BCIR_BCIRPASSES_H
