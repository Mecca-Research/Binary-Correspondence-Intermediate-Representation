# Out-of-tree build for the BCIR MLIR dialect family + GEM/RCSP passes (C++23).
#
# Build (needs cmake + ninja + libmlir-NN-dev + llvm-NN-dev; CI tracks the latest LLVM, 22,
# installed from apt.llvm.org since Ubuntu's default repos top out near 18/19):
#
#   sudo apt-get install -y mlir-22-tools libmlir-22-dev llvm-22-dev cmake ninja-build
#   tools/wsl/build_mlir.sh          # full dialect lib + bcir-opt
#   tools/wsl/check_passes.sh        # validate every pass (incl. -bcir-rcsp + corpus)
#
# MLIR_DIR / LLVM_DIR auto-resolve to the highest installed /usr/lib/llvm-*/lib/cmake; override to pin.
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

# BCIR lowering targets C++26 (-std=c++2c). HandleLLVMOptions defaults to LLVM's C++17; the
# deterministic optimizer-core passes use C++20/23 idioms (std::ranges, structured bindings,
# designated inits, <=> where useful) and are c++2c-clean. CMake's CXX_STANDARD=26 GNU flag
# mapping only landed in CMake ~3.30 (ubuntu-latest + conda ship 3.28), so to stay portable
# the base standard stays 23 (mapped everywhere) and -std=c++2c is appended when the compiler
# accepts it -- the later -std wins, so the actual compile is C++26 on clang-22 / gcc-15,
# while older toolchains gracefully stay at 23.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++2c" BCIR_HAS_CXX2C)
if(BCIR_HAS_CXX2C)
  add_compile_options(-std=c++2c)
endif()

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

# ODS -> C++ generation in one shot.
add_custom_target(bcir_tablegen DEPENDS
  BCIRAttrsIncGen BCIRTypesIncGen BCIRInterfacesIncGen BCIROpsIncGen)

# --- dialect C++ library + bcir-opt (LangRef Milestone 3) ---
add_mlir_dialect_library(MLIRBCIR
  lib/BCIRDialect.cpp
  lib/BCIRPasses.cpp                 # registration only
  lib/passes/BCIRVerifyPass.cpp      # R1-R16 semantic laws
  lib/passes/BCIRPromotePass.cpp     # GGG->UX opt-law
  lib/passes/BCIRConvertToLLVM.cpp   # compute/barrier -> LLVM dialect
  lib/passes/BCIRGEMPasses.cpp       # classify / batch / schedule / lower
  lib/passes/BCIRGemMatmulCostPass.cpp # recompute the B1 matmul roofline (matmul.py::cost_of) parity + annotate (informs-only)
  lib/passes/BCIRGemActivationCostPass.cpp # recompute the G1 activation roofline (activation.py::cost_of) parity + annotate (informs-only)
  lib/passes/BCIRLowerGemMatmulPass.cpp # lower gem.matmul plan -> tiled gem.block sequence
  lib/passes/BCIRLowerGemMatmulBufferPass.cpp # lower gem.matmul_buffer -> tiled scf.for nest (C += A*B)
  lib/passes/BCIRLowerGemActivationPass.cpp # lower gem.activation plan -> lane-width gem.block stripes (G1)
  lib/passes/BCIRLowerGemConvPass.cpp # lower gem.conv plan -> im2col-gemm tiled gem.block sequence (G7)
  lib/passes/BCIRLowerGemAttentionPass.cpp # lower gem.attention plan -> two gem.matmul tile seqs + softmax (G7)
  lib/passes/BCIRFuseMatmulActivationPass.cpp # fuse sole-consumer gem.matmul -> gem.activation epilogue (G2)
  lib/passes/BCIRCacheContentionPass.cpp # recompute the G4 cache-line/bank-conflict CONTENTION cost signal (informs-only)
  lib/passes/BCIRLayoutPivotPass.cpp  # recompute the G3 SoA<->AoS layout-pivot pricing (informs-only)
  lib/passes/BCIRSelectPass.cpp      # min-plus selection
  lib/passes/BCIRCostModel.cpp       # K_BCIR cost algebra (cost.py port)
  lib/passes/BCIRPlanPass.cpp        # min-plus shortest path (realize.optimize)
  lib/passes/BCIROverlapPass.cpp     # (max,+) scheduled price (gem/overlap.py)
  lib/passes/BCIRRcspPass.cpp        # RCSP / Pareto (per-claim)
  lib/passes/BCIRRcspPlanPass.cpp    # plan-level accumulated-budget label-DP
  lib/passes/BCIRBundlePass.cpp      # multi-claim (joint) bundle detection + reorder re-price
  lib/passes/BCIRExplainPass.cpp     # proof.explain record + proof.replay recheck as IR
  lib/passes/BCIRComposePass.cpp     # compositional cost over the func/if region tree
  lib/passes/BCIRCimDvfsPass.cpp     # recompute CIM/PIM dispatch + DVFS clock (gem.cim/dvfs)
  lib/passes/BCIRScheduleEftPass.cpp # duration-aware EFT wave scheduling (gem.schedule)
  lib/passes/BCIRAllocPoolPass.cpp   # liveness-based memory pool planning (allocator.pool_plan)
  lib/passes/BCIRSensePass.cpp       # regret-driven telemetry resolution gate (kbcir/sensing.py)
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
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRArithDialect
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
