//===- bcir-opt.cpp - BCIR optimizer driver -------------------------------===//
//
// An mlir-opt that registers the BCIR dialect + the Phase-6 passes, so the
// pretty ODS corpus can be parsed, verified (-bcir-verify), rewritten
// (-bcir-promote-lanes), and lowered (-convert-bcir-to-llvm):
//
//   bcir-opt -bcir-verify mlir/examples/full_vec_add_ct1.mlir
//   bcir-opt -convert-bcir-to-llvm input.mlir
//
//===----------------------------------------------------------------------===//

#include "BCIR/BCIRDialect.h"
#include "BCIR/BCIRPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<bcir::BCIRDialect, mlir::LLVM::LLVMDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
  bcir::registerBCIRPasses();
  bcir::registerBCIRPipelines();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "BCIR optimizer driver\n", registry));
}
