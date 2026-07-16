#!/usr/bin/env bash
# Build bcir-opt against the local conda-forge MLIR 22 toolchain (ABI-matched compiler)
# and run the WHOLE rail against TRUE MLIR 22 -- including the 22-only checks an 18 build
# cannot do (the IRDL named-operand corpus). Run tools/local/setup_mlir22.sh first.
# CI (apt.llvm.org) stays the authoritative gate; this is the fast local mirror of it.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/bcir-rail22.XXXXXXXX")" || exit 1
trap 'rm -rf -- "${TMP_ROOT}"' EXIT
BUILD_LOG="${TMP_ROOT}/build.log"
STEP_LOG="${TMP_ROOT}/step.log"
# shellcheck source=/dev/null
source "${ROOT}/tools/local/env_mlir22.sh"
ENV="${MLIR22_ENV}"
if [ ! -x "${ENV}/bin/mlir-opt" ]; then
  echo "MLIR 22 toolchain missing; run tools/local/setup_mlir22.sh first." >&2
  exit 1
fi

BUILD="${ROOT}/build/mlir22"
JOBS="${BCIR_BUILD_JOBS:-2}"
if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || [ "${JOBS}" -gt 64 ]; then
  echo "BCIR_BUILD_JOBS must be an integer in [1, 64]" >&2
  exit 2
fi
echo "[rail22] building bcir-opt against $(${ENV}/bin/mlir-opt --version | grep -io 'LLVM version [0-9.]*')"
cmake -G Ninja -S "${ROOT}/mlir" -B "${BUILD}" \
  -DMLIR_DIR="${ENV}/lib/cmake/mlir" -DLLVM_DIR="${ENV}/lib/cmake/llvm" \
  -DCMAKE_C_COMPILER="${MLIR22_CC}" -DCMAKE_CXX_COMPILER="${MLIR22_CXX}" \
  -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 || { echo "[rail22] cmake configure failed"; exit 1; }
cmake --build "${BUILD}" --target bcir-opt --parallel "${JOBS}" \
  >"${BUILD_LOG}" 2>&1 || { echo "[rail22] build failed"; tail -20 "${BUILD_LOG}"; exit 1; }
BO="${BUILD}/bcir-opt"

fail=0
step() { printf '  %-44s' "$1"; shift; if "$@" >"${STEP_LOG}" 2>&1; then echo "OK"; else echo "FAIL"; tail -8 "${STEP_LOG}"; fail=1; fi; }
echo "[rail22] running the full rail on true MLIR 22:"
step "tblgen (ODS family)"                 env MLIR_TBLGEN="${MLIR_TBLGEN}" MLIR_INCLUDE="${MLIR_INCLUDE}" bash "${ROOT}/tools/wsl/tblgen_check.sh"
step "passes (R1-R23 / GEM / optimizer)"   env BCIR_OPT="${BO}" bash "${ROOT}/tools/wsl/check_passes.sh"
step "ODS pretty examples"                 env BCIR_OPT="${BO}" bash "${ROOT}/tools/wsl/check_ods_examples.sh"
step "bytecode round-trip"                 env BCIR_OPT="${BO}" bash "${ROOT}/tools/wsl/check_bytecode.sh"
step "IRDL corpus (22-only named syntax)"  env MLIR_OPT="${MLIR_OPT}" bash "${ROOT}/tools/irdl/check_corpus.sh"

[ "${fail}" -eq 0 ] && echo "[rail22] FULL MLIR 22 RAIL GREEN LOCALLY" || echo "[rail22] FAILURES"
exit "${fail}"
