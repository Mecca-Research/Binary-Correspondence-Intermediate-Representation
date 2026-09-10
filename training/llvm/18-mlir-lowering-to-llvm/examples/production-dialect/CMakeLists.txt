# Standalone out-of-tree MLIR dialect.
#
# Deliberately NOT part of the training corpus's own CMake project: building it
# needs an MLIR development install (mlir-tblgen, the MLIR CMake package, and
# headers), which the corpus gates do not assume. The production reference that
# this repository's CI does build is ../../../../mlir/.
#
#   cmake -S training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect \
#         -B build/training-dialect \
#         -DMLIR_DIR="$(llvm-config --prefix)/lib/cmake/mlir" \
#         -DLLVM_DIR="$(llvm-config --cmakedir)"
#   cmake --build build/training-dialect -j2

cmake_minimum_required(VERSION 3.20)
project(TrainingDialect LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Building the training dialect against MLIR ${MLIR_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# TableGen writes *.inc into the BUILD tree; without this every generated
# include fails to resolve, and the error names the .inc file rather than the
# missing include path.
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# --- ODS -> C++ -------------------------------------------------------------
# set(LLVM_TARGET_DEFINITIONS ...) is positional state: it applies to the
# mlir_tablegen calls that follow it. Reordering these blocks silently
# generates from the wrong .td file.
set(LLVM_TARGET_DEFINITIONS TrainingOps.td)
mlir_tablegen(TrainingOpsDialect.h.inc   -gen-dialect-decls -dialect=training)
mlir_tablegen(TrainingOpsDialect.cpp.inc -gen-dialect-defs  -dialect=training)
mlir_tablegen(TrainingOps.h.inc          -gen-op-decls)
mlir_tablegen(TrainingOps.cpp.inc        -gen-op-defs)
mlir_tablegen(TrainingOpsTypes.h.inc     -gen-typedef-decls)
mlir_tablegen(TrainingOpsTypes.cpp.inc   -gen-typedef-defs)
mlir_tablegen(TrainingEnums.h.inc        -gen-enum-decls)
mlir_tablegen(TrainingEnums.cpp.inc      -gen-enum-defs)
add_public_tablegen_target(TrainingOpsIncGen)

# --- dialect library --------------------------------------------------------
add_mlir_dialect_library(MLIRTraining
  TrainingDialect.cpp
  TrainingToLLVM.cpp

  DEPENDS
  TrainingOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion)

# --- tool -------------------------------------------------------------------
add_llvm_executable(training-opt training-opt.cpp)
llvm_update_compile_flags(training-opt)
target_link_libraries(training-opt PRIVATE MLIRTraining MLIROptLib)
mlir_check_all_link_libraries(training-opt)
