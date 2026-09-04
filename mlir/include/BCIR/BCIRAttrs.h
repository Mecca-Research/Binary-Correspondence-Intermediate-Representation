//===- BCIRAttrs.h - BCIR attribute declarations -----------------*- C++ -*-===//
#ifndef BCIR_BCIRATTRS_H
#define BCIR_BCIRATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIREnums.h"

#define GET_ATTRDEF_CLASSES
#include "BCIRAttrs.h.inc" // generated: -gen-attrdef-decls

#endif // BCIR_BCIRATTRS_H
