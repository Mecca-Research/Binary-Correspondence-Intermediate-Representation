# Out-of-tree build for the BCIR MLIR dialect family + GEM/RCSP passes (C++23).
#
# Build (needs cmake + ninja + libmlir-NN-dev + llvm-NN-dev; CI installs LLVM 18+19):
#
#   sudo apt-get install -y mlir-18-tools libmlir-18-dev llvm-18-dev cmake ninja-build
#   tools/wsl/build_mlir.sh          # full dialect lib + bcir-opt
#   tools/wsl/check_passes.sh        # validate every pass (incl. -bcir-rcsp + corpus)
#
# MLIR_DIR / LLVM_DIR auto-resolve from /usr/lib/llvm-*/lib/cmake; override to pin.
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

# BCIR lowering targets C++23 (HandleLLVMOptions defaults to LLVM's C++17). The
# deterministic optimizer-core passes use C++20/23 idioms (std::ranges, structured
# bindings, designated inits, <=> where useful). C++26 (-std=c++2c) is still
# experimental in clang-18/gcc-13 and the stock MLIR 18 headers do not yet compile
# clean under it, so 23 is the gating standard; the BCIR code itself is c++2c-ready.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# ODS -> C++ generation in one shot.
add_custom_target(bcir_tablegen DEPENDS
  BCIRAttrsIncGen BCIRTypesIncGen BCIRInterfacesIncGen BCIROpsIncGen GEMPassIncGen)

# --- dialect C++ library + bcir-opt (LangRef Milestone 3) ---
add_mlir_dialect_library(MLIRBCIR
  lib/BCIRDialect.cpp
  lib/BCIRPasses.cpp                 # registration only
  lib/passes/BCIRVerifyPass.cpp      # R1-R16 semantic laws
  lib/passes/BCIRPromotePass.cpp     # GGG->UX opt-law
  lib/passes/BCIRConvertToLLVM.cpp   # compute/barrier -> LLVM dialect
  lib/passes/BCIRGEMPasses.cpp       # classify / batch / schedule / lower
  lib/passes/BCIRSelectPass.cpp      # min-plus selection
  lib/passes/BCIRCostModel.cpp       # K_BCIR cost algebra (cost.py port)
  lib/passes/BCIRPlanPass.cpp        # min-plus shortest path (realize.optimize)
  lib/passes/BCIROverlapPass.cpp     # (max,+) scheduled price (gem/overlap.py)
  lib/passes/BCIRRcspPass.cpp        # RCSP / Pareto (per-claim)
  lib/passes/BCIRRcspPlanPass.cpp    # plan-level accumulated-budget label-DP
  lib/passes/BCIRBundlePass.cpp      # multi-claim (joint) bundle detection + reorder re-price
  lib/passes/BCIRExplainPass.cpp     # proof-carrying decision record (proof.explain) as IR
  lib/passes/BCIRComposePass.cpp     # compositional cost over the func/if region tree
  lib/passes/BCIRCimDvfsPass.cpp     # recompute CIM/PIM dispatch + DVFS clock (gem.cim/dvfs)
  lib/passes/BCIRScheduleEftPass.cpp # duration-aware EFT wave scheduling (gem.schedule)
  DEPENDS
    BCIRAttrsIncGen
    BCIRTypesIncGen
    BCIROpsIncGen
  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTransformUtils
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
)

add_mlir_tool(bcir-opt tools/bcir-opt.cpp)
llvm_update_compile_flags(bcir-opt)
target_link_libraries(bcir-opt
  PRIVATE
    MLIRBCIR
    MLIROptLib
    MLIRParser
    MLIRIR
    MLIRSupport
)
mlir_check_all_link_libraries(bcir-opt)
