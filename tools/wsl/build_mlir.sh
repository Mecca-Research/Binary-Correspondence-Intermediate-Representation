#!/usr/bin/env bash
# Configure + build the compiled BCIR MLIR dialect and the `bcir-opt` tool
# (LangRef Milestone 3). Resolves the MLIR/LLVM cmake dirs automatically; override
# with MLIR_DIR / LLVM_DIR. Requires: cmake, a C++ compiler, libmlir-NN-dev +
# llvm-NN-dev (for MLIRConfig.cmake + LLVMConfig.cmake).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${ROOT}/build/mlir-build"

MLIR_DIR="${MLIR_DIR:-$(ls -d /usr/lib/llvm-*/lib/cmake/mlir 2>/dev/null | sort -V | tail -1)}"
LLVM_DIR="${LLVM_DIR:-$(ls -d /usr/lib/llvm-*/lib/cmake/llvm 2>/dev/null | sort -V | tail -1)}"
: "${MLIR_DIR:?MLIRConfig.cmake not found (install libmlir-NN-dev), or set MLIR_DIR}"
: "${LLVM_DIR:?LLVMConfig.cmake not found (install llvm-NN-dev), or set LLVM_DIR}"

GEN="Unix Makefiles"
command -v ninja >/dev/null 2>&1 && GEN="Ninja"

echo "[build_mlir] generator=${GEN} MLIR_DIR=${MLIR_DIR}"
cmake -G "${GEN}" -S "${ROOT}/mlir" -B "${BUILD}" \
  -DMLIR_DIR="${MLIR_DIR}" -DLLVM_DIR="${LLVM_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" --target bcir-opt
echo "[build_mlir] bcir-opt: $(find "${BUILD}" -name bcir-opt -type f | head -1)"
