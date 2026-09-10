//===- TrainingDialect.h ---------------------------------------*- C++ -*-===//
//
// Hand-written header that pulls in the TableGen-generated declarations. The
// .inc files are produced into the BUILD tree, so the build directory must be
// on the include path (see CMakeLists.txt and chapter 08).
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRAINING_TRAININGDIALECT_H
#define LLVM_TRAINING_TRAININGDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TrainingEnums.h.inc"
#include "TrainingOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "TrainingOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "TrainingOps.h.inc"

namespace mlir::training {
/// Registers the training -> LLVM dialect conversion pass by name.
void registerTrainingToLLVMPass();
} // namespace mlir::training

#endif // LLVM_TRAINING_TRAININGDIALECT_H
