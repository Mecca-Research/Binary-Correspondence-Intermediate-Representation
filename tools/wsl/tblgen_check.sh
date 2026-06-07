#!/usr/bin/env bash
# Validate the BCIR ODS dialect family by running every TableGen generator over
# the .td files (no full build, no C++ sources needed). Requires mlir-tblgen and
# the MLIR include dir. Set MLIR_INCLUDE (e.g. $MLIR_DIR/../../include) first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INC="${ROOT}/mlir/include"
OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

if ! command -v mlir-tblgen >/dev/null 2>&1; then
  echo "mlir-tblgen not found; cannot validate ODS on this host." >&2
  exit 0
fi
: "${MLIR_INCLUDE:?set MLIR_INCLUDE to your MLIR headers dir (contains mlir/IR/OpBase.td)}"

I=(-I "${INC}" -I "${MLIR_INCLUDE}")
run() { echo "[tblgen] $*"; mlir-tblgen "$@" "${I[@]}" -o /dev/null; }

run -gen-enum-decls    "${INC}/BCIR/BCIRAttrs.td"
run -gen-attrdef-decls "${INC}/BCIR/BCIRAttrs.td"
run -gen-typedef-decls "${INC}/BCIR/BCIRTypes.td"
run -gen-op-interface-decls "${INC}/BCIR/BCIRInterfaces.td"
run -gen-dialect-decls -dialect=bcir "${INC}/BCIR/BCIROps.td"
run -gen-op-decls      "${INC}/BCIR/BCIROps.td"
run -gen-pass-decls -name GEM "${ROOT}/mlir/passes/GEMPasses.td"
echo "[tblgen] all BCIR ODS generators passed"
