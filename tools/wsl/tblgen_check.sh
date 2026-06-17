#!/usr/bin/env bash
# Validate the BCIR ODS dialect family by running every TableGen generator over
# the .td files (no full build, no C++ sources needed). Resolves mlir-tblgen (or
# mlir-tblgen-NN) and the MLIR include dir automatically; override with
# MLIR_TBLGEN / MLIR_INCLUDE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INC="${ROOT}/mlir/include"

# Resolve mlir-tblgen version-agnostically: the highest /usr/lib/llvm-*/bin/mlir-tblgen (the
# installed major -- 22 now), then versioned / unversioned names on PATH.
MLIR_TBLGEN="${MLIR_TBLGEN:-$(ls /usr/lib/llvm-*/bin/mlir-tblgen 2>/dev/null | sort -V | tail -1)}"
MLIR_TBLGEN="${MLIR_TBLGEN:-$(command -v mlir-tblgen-22 || command -v mlir-tblgen || command -v mlir-tblgen-18 || true)}"
if [ -z "${MLIR_TBLGEN}" ]; then
  echo "mlir-tblgen not found; cannot validate ODS on this host (install mlir-NN-tools)." >&2
  exit 0
fi

# Auto-detect the MLIR/LLVM .td include dir (contains mlir/IR/OpBase.td).
if [ -z "${MLIR_INCLUDE:-}" ]; then
  for d in /usr/lib/llvm-*/include /usr/include; do
    [ -f "${d}/mlir/IR/OpBase.td" ] && MLIR_INCLUDE="${d}" && break
  done
fi
: "${MLIR_INCLUDE:?could not find MLIR headers; set MLIR_INCLUDE to the dir holding mlir/IR/OpBase.td}"

echo "[tblgen] $(${MLIR_TBLGEN} --version | head -1)"
echo "[tblgen] include: ${MLIR_INCLUDE}"
I=(-I "${INC}" -I "${MLIR_INCLUDE}")
run() { echo "[tblgen] $1"; "${MLIR_TBLGEN}" "$@" "${I[@]}" -o /dev/null; }

# Decls + defs (defs catch assemblyFormat/builder errors that decls miss).
run -gen-enum-decls         "${INC}/BCIR/BCIRAttrs.td"
run -gen-enum-defs          "${INC}/BCIR/BCIRAttrs.td"
run -gen-attrdef-decls      "${INC}/BCIR/BCIRAttrs.td"
run -gen-attrdef-defs       "${INC}/BCIR/BCIRAttrs.td"
run -gen-typedef-decls      "${INC}/BCIR/BCIRTypes.td"
run -gen-typedef-defs       "${INC}/BCIR/BCIRTypes.td"
run -gen-op-interface-decls "${INC}/BCIR/BCIRInterfaces.td"
run -gen-op-interface-defs  "${INC}/BCIR/BCIRInterfaces.td"
run -gen-dialect-decls -dialect=bcir "${INC}/BCIR/BCIROps.td"
run -gen-dialect-defs  -dialect=bcir "${INC}/BCIR/BCIROps.td"
run -gen-op-decls           "${INC}/BCIR/BCIROps.td"
run -gen-op-defs            "${INC}/BCIR/BCIROps.td"
echo "[tblgen] all BCIR ODS generators passed"
