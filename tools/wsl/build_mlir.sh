#!/usr/bin/env bash
# Configure + build the compiled BCIR MLIR dialect and the `bcir-opt` tool
# (LangRef Milestone 3). Resolves the MLIR/LLVM cmake dirs automatically; override
# with MLIR_DIR / LLVM_DIR. Requires: cmake, a C++ compiler, libmlir-NN-dev +
# llvm-NN-dev (for MLIRConfig.cmake + LLVMConfig.cmake).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${ROOT}/build/mlir-build"

MLIR_DIR="${MLIR_DIR:-$(ls -d /usr/lib/llvm-*/lib/cmake/mlir 2>/dev/null | sort -V | tail -1 || true)}"
LLVM_DIR="${LLVM_DIR:-$(ls -d /usr/lib/llvm-*/lib/cmake/llvm 2>/dev/null | sort -V | tail -1 || true)}"
: "${MLIR_DIR:?MLIRConfig.cmake not found (install libmlir-NN-dev), or set MLIR_DIR}"
: "${LLVM_DIR:?LLVMConfig.cmake not found (install llvm-NN-dev), or set LLVM_DIR}"

GEN="Unix Makefiles"
command -v ninja >/dev/null 2>&1 && GEN="Ninja"
JOBS="${BCIR_BUILD_JOBS:-2}"
if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || [ "${JOBS}" -gt 64 ]; then
  echo "BCIR_BUILD_JOBS must be an integer in [1, 64]" >&2
  exit 2
fi

# Use ccache as the compiler launcher when present (CI caches CCACHE_DIR across runs);
# the bcir-opt compile against the MLIR headers dominates the rail's wall time, and a
# warm cache turns it from a full rebuild into a near-instant relink. No-op locally if
# ccache isn't installed.
LAUNCH=()
if command -v ccache >/dev/null 2>&1; then
  LAUNCH=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache)
  echo "[build_mlir] ccache enabled ($(ccache --version | head -1))"
fi

echo "[build_mlir] generator=${GEN} MLIR_DIR=${MLIR_DIR}"
cmake -G "${GEN}" -S "${ROOT}/mlir" -B "${BUILD}" \
  -DMLIR_DIR="${MLIR_DIR}" -DLLVM_DIR="${LLVM_DIR}" -DCMAKE_BUILD_TYPE=Release "${LAUNCH[@]}"
cmake --build "${BUILD}" --target bcir-opt --parallel "${JOBS}"
echo "[build_mlir] bcir-opt: $(find "${BUILD}" -name bcir-opt -type f | head -1)"
