#!/usr/bin/env bash
# Load the BCIR IRDL projection into stock mlir-opt and round-trip a generic BCIR
# program. Proves the WASM-like property: BCIR artifacts are pure data loaded by a
# standard prebuilt engine, with no BCIR-authored C++ compiled first.
#
#   tools/irdl/run_stock_mlir_opt.sh [program.mlir]
#
# Resolves mlir-opt (or mlir-opt-NN). The IRDL load flag is --irdl-file=<file>
# (LLVM >= 17); override via BCIR_IRDL_FLAG if your build differs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IRDL="${ROOT}/mlir/irdl/bcir.irdl.mlir"
PROG="${1:-${ROOT}/mlir/test/irdl/00_smoke_generic.mlir}"
FLAG="${BCIR_IRDL_FLAG:---irdl-file=${IRDL}}"

MLIR_OPT="${MLIR_OPT:-$(ls /usr/lib/llvm-*/bin/mlir-opt 2>/dev/null | sort -V | tail -1)}"
MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt-23 || command -v mlir-opt-22 || command -v mlir-opt || command -v mlir-opt-18 || true)}"
if [ -z "${MLIR_OPT}" ]; then
  echo "mlir-opt not found; skipping (pure-IR projection is optional on this host)." >&2
  exit 0
fi

echo "[irdl] ${MLIR_OPT} ${FLAG} ${PROG}"
"${MLIR_OPT}" "${FLAG}" "${PROG}"
