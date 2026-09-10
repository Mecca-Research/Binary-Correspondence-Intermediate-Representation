//===- training-opt.cpp - the dialect's own opt driver ---------*- C++ -*-===//
//
// Mirrors ../../../../mlir/tools/bcir-opt.cpp: build a registry, insert every
// dialect the tool must understand, register the passes, and hand off to
// MlirOptMain.
//
//===----------------------------------------------------------------------===//
#include "TrainingDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::training::TrainingDialect, mlir::LLVM::LLVMDialect,
                  mlir::func::FuncDialect>();

  mlir::training::registerTrainingToLLVMPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "training dialect driver\n", registry));
}
