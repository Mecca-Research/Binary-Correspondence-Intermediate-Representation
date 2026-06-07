//===- GEMPasses.td - GEM / K_BCIR pass declarations --------*- tablegen -*-===//
//
// Pass declarations for the GEM pipeline (LangRef Milestone 4..7). Implemented in
// C++ in a later toolchain phase; the bcir/ oracle is the executable reference
// today (classify -> realize -> batch -> schedule -> prefetch -> StreamPack).
//
//===----------------------------------------------------------------------===//
#ifndef GEM_PASSES
#define GEM_PASSES

include "mlir/Pass/PassBase.td"

def BCIRClassifyLanes : Pass<"bcir-classify-lanes", "::mlir::ModuleOp"> {
  let summary = "Classify each claim's access pattern into a lane/stride class.";
  let constructor = "::bcir::createClassifyLanesPass()";
}

def BCIRSelectRealization : Pass<"bcir-select-realization", "::mlir::ModuleOp"> {
  let summary = "K_BCIR min-plus realization selection over the candidate DAG.";
  let constructor = "::bcir::createSelectRealizationPass()";
}

def BCIRBatch : Pass<"bcir-batch", "::mlir::ModuleOp"> {
  let summary = "Form GEM batches (fusion vs register pressure) from selected paths.";
  let constructor = "::bcir::createBatchPass()";
}

def BCIRSchedule : Pass<"bcir-schedule", "::mlir::ModuleOp"> {
  let summary = "Phase-sliced deterministic scheduling + prefetch attachment.";
  let constructor = "::bcir::createSchedulePass()";
}

def BCIRLowerToLLVM : Pass<"bcir-lower-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower a hydrated StreamPack to the LLVM dialect (R12-preserving).";
  let constructor = "::bcir::createLowerToLLVMPass()";
}

#endif // GEM_PASSES
