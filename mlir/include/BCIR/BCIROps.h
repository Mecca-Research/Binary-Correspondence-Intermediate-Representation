//===- BCIROps.h - BCIR operation declarations -------------------*- C++ -*-===//
#ifndef BCIR_BCIROPS_H
#define BCIR_BCIROPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "BCIR/BCIRAttrs.h"
#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIRTypes.h"

#define GET_OP_CLASSES
#include "BCIROps.h.inc" // generated: -gen-op-decls

#endif // BCIR_BCIROPS_H
