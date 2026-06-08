//===- bcir-opt.cpp - BCIR optimizer driver -------------------------------===//
//
// A minimal mlir-opt that registers the BCIR dialect so the pretty ODS corpus
// (mlir/examples/*.mlir) can be parsed, verified, and round-tripped:
//
//   bcir-opt mlir/examples/full_vec_add_ct1.mlir
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<bcir::BCIRDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "BCIR optimizer driver\n", registry));
}
