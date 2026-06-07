#!/usr/bin/env bash
# Load the BCIR IRDL projection into stock mlir-opt and round-trip a generic-syntax
# BCIR program. Proves the WASM-like property: BCIR artifacts are pure data loaded
# by a standard prebuilt engine, with no BCIR-authored C++ compiled first.
#
#   tools/irdl/run_stock_mlir_opt.sh [program.mlir]
#
# The IRDL load flag is whatever `probe_irdl_flags.sh` reports for your mlir-opt
# (recent LLVM: --irdl-file=<file>). Override via BCIR_IRDL_FLAG if needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IRDL="${ROOT}/mlir/irdl/bcir.irdl.mlir"
PROG="${1:-${ROOT}/mlir/test/irdl/00_smoke_generic.mlir}"
FLAG="${BCIR_IRDL_FLAG:---irdl-file=${IRDL}}"

if ! command -v mlir-opt >/dev/null 2>&1; then
  echo "mlir-opt not found; skipping (pure-IR projection is optional on this host)." >&2
  exit 0
fi

echo "[irdl] mlir-opt ${FLAG} ${PROG}"
mlir-opt "${FLAG}" "${PROG}"
