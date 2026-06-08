//===- BCIRTypes.h - BCIR type declarations ----------------------*- C++ -*-===//
#ifndef BCIR_BCIRTYPES_H
#define BCIR_BCIRTYPES_H

#include "mlir/IR/Types.h"

#include "BCIR/BCIRDialect.h"

#define GET_TYPEDEF_CLASSES
#include "BCIRTypes.h.inc"  // generated: -gen-typedef-decls

#endif // BCIR_BCIRTYPES_H
