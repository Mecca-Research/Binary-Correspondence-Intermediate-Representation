#!/usr/bin/env bash
# Validate the BCIR dialect's MLIR bytecode round-trip (an MLIR-22 feature the rail
# did not previously exercise): every valid example + positive pass corpus must survive
# text -> bytecode -> text *identically*. This proves the dialect's stable, versioned
# bytecode encoding works -- including the inherent Properties storage every BCIR op uses
# (usePropertiesForAttributes defaults on), which serializes through bytecode separately
# from discardable attributes. Run tools/wsl/build_mlir.sh first (or set BCIR_OPT).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BO="${BCIR_OPT:-$(find "${ROOT}/build/mlir-build" -name bcir-opt -type f 2>/dev/null | head -1)}"
if [ -z "${BO}" ]; then
  echo "bcir-opt not built; run tools/wsl/build_mlir.sh first (skipping)." >&2
  exit 0
fi

# Valid, positive modules only (the verify_*.mlir negative corpus holds intentionally
# malformed modules guarded by -verify-diagnostics, so they do not parse standalone).
MODULES=("${ROOT}"/mlir/examples/*.mlir)
for p in plan gem_corpus gem_passes target_matrix theta_hot cost_model overlap rcsp \
         compose_ops async_tokens; do
  [ -f "${ROOT}/mlir/test/passes/${p}.mlir" ] && MODULES+=("${ROOT}/mlir/test/passes/${p}.mlir")
done

fail=0
for f in "${MODULES[@]}"; do
  base="$(basename "${f}")"
  txt="$("${BO}" "${f}" 2>/tmp/bce)" || { echo "  FAIL parse ${base}"; cat /tmp/bce; fail=1; continue; }
  "${BO}" "${f}" --emit-bytecode >/tmp/bc.mlirbc 2>/tmp/bce \
    || { echo "  FAIL emit-bytecode ${base}"; cat /tmp/bce; fail=1; continue; }
  bc="$("${BO}" /tmp/bc.mlirbc 2>/tmp/bce)" \
    || { echo "  FAIL parse-bytecode ${base}"; cat /tmp/bce; fail=1; continue; }
  if [ "${txt}" = "${bc}" ]; then
    echo "  PASS ${base} ($(wc -c </tmp/bc.mlirbc | tr -d ' ') B)"
  else
    echo "  FAIL bytecode round-trip differs ${base}"; diff <(echo "${txt}") <(echo "${bc}") | head; fail=1
  fi
done

[ "${fail}" -eq 0 ] && echo "[bytecode] all modules round-trip through MLIR bytecode" || echo "[bytecode] FAILURES"
exit "${fail}"
