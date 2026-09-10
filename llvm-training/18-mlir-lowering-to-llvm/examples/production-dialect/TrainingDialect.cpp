//===- TrainingDialect.cpp -------------------------------------*- C++ -*-===//
//
// Dialect initialization, generated definitions, and the one verifier that a
// trait could not express.
//
//===----------------------------------------------------------------------===//
#include "TrainingDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::training;

#include "TrainingEnums.cpp.inc"
#include "TrainingOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TrainingOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TrainingOps.cpp.inc"

void TrainingDialect::initialize() {
  // Three separate registrations, and all three are required. Registering the
  // dialect without its operations gives a dialect whose ops do not exist.
  addTypes<
#define GET_TYPEDEF_LIST
#include "TrainingOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "TrainingOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Hand-written verification
//===----------------------------------------------------------------------===//

LogicalResult ScaleOp::verify() {
  // A domain rule no trait expresses. Everything a trait *does* express --
  // purity, operand/result type agreement -- is declared in the ODS instead of
  // repeated here.
  if (getFactor() <= 0)
    return emitOpError("factor must be positive, got ") << getFactor();

  // Refusals name the value they refuse. A diagnostic that says only "invalid"
  // is a diagnostic the reader has to reproduce to understand.
  if (getRounding() && getInput().getType().getIntOrFloatBitWidth() < 8)
    return emitOpError(
        "rounding is only meaningful on inputs of at least 8 bits");

  return success();
}
