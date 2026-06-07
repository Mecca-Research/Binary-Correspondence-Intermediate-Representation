# Out-of-tree build for the BCIR MLIR dialect family + GEM passes.
#
# NOT built in this milestone: this host has clang only (no mlir-tblgen, MLIR/LLVM
# dev libs, or a configured MLIR_DIR/LLVM_DIR). These .td files are authored v0.x
# specs validated against MLIR 20 in WSL. To build:
#
#   source tools/wsl/env.sh          # sets MLIR_DIR / LLVM_DIR
#   tools/wsl/tblgen_check.sh        # ODS -> *.inc generation only
#   tools/wsl/build_mlir.sh          # full dialect lib + bcir-opt
#
cmake_minimum_required(VERSION 3.20)
project(bcir-mlir LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})  # TableGen-generated *.inc

# --- enums + attributes ---
set(LLVM_TARGET_DEFINITIONS include/BCIR/BCIRAttrs.td)
mlir_tablegen(BCIREnums.h.inc -gen-enum-decls)
mlir_tablegen(BCIREnums.cpp.inc -gen-enum-defs)
mlir_tablegen(BCIRAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(BCIRAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(BCIRAttrsIncGen)

# --- types ---
set(LLVM_TARGET_DEFINITIONS include/BCIR/BCIRTypes.td)
mlir_tablegen(BCIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(BCIRTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(BCIRTypesIncGen)

# --- op interfaces ---
set(LLVM_TARGET_DEFINITIONS include/BCIR/BCIRInterfaces.td)
mlir_tablegen(BCIRInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(BCIRInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(BCIRInterfacesIncGen)

# --- dialect + ops (umbrella pulls the whole family) ---
set(LLVM_TARGET_DEFINITIONS include/BCIR/BCIROps.td)
mlir_tablegen(BCIRDialect.h.inc -gen-dialect-decls -dialect=bcir)
mlir_tablegen(BCIRDialect.cpp.inc -gen-dialect-defs -dialect=bcir)
mlir_tablegen(BCIROps.h.inc -gen-op-decls)
mlir_tablegen(BCIROps.cpp.inc -gen-op-defs)
add_public_tablegen_target(BCIROpsIncGen)

# --- GEM passes ---
set(LLVM_TARGET_DEFINITIONS passes/GEMPasses.td)
mlir_tablegen(GEMPasses.h.inc -gen-pass-decls -name GEM)
add_public_tablegen_target(GEMPassIncGen)

# ODS -> C++ generation in one shot (this milestone's build gate).
add_custom_target(bcir_tablegen DEPENDS
  BCIRAttrsIncGen BCIRTypesIncGen BCIRInterfacesIncGen BCIROpsIncGen GEMPassIncGen)

# The dialect C++ library + bcir-opt tool are added once lib/*.cpp lands
# (LangRef Milestone 3+). Kept commented so this file documents the target shape
# without requiring source files that do not exist yet.
#
# add_mlir_dialect_library(MLIRBCIR lib/BCIRDialect.cpp
#   DEPENDS BCIRAttrsIncGen BCIRTypesIncGen BCIRInterfacesIncGen BCIROpsIncGen
#   LINK_LIBS PUBLIC MLIRIR MLIRSupport)
# add_mlir_tool(bcir-opt tools/bcir-opt.cpp)
# target_link_libraries(bcir-opt PRIVATE MLIRBCIR MLIROptLib MLIRParser MLIRFuncDialect)
