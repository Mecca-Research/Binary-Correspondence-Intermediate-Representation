//===- BCIRDialect.cpp - BCIR dialect registration ------------------------===//
//
// Registers the BCIR dialect family (types, attributes, enums, ops) generated
// from the .td law. Built by the out-of-tree CMake (see ../CMakeLists.txt) into
// the `bcir-opt` tool, which parses/verifies the pretty ODS corpus.
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"

#include "BCIR/BCIRAttrs.h"
#include "BCIR/BCIROps.h"
#include "BCIR/BCIRTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace bcir;

#include "BCIRDialect.cpp.inc"

#include "BCIREnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "BCIRAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "BCIRTypes.cpp.inc"

#define GET_OP_CLASSES
#include "BCIROps.cpp.inc"

void BCIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "BCIROps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "BCIRAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "BCIRTypes.cpp.inc"
      >();
}
