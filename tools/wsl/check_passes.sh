#!/usr/bin/env bash
# Validate the Phase-6 bcir-opt passes: -bcir-verify (R1/R2/R4/R6),
# -bcir-promote-lanes (GGG->UX opt-law), -convert-bcir-to-llvm (LLVM lowering).
# Run tools/wsl/build_mlir.sh first (or set BCIR_OPT).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BO="${BCIR_OPT:-$(find "${ROOT}/build/mlir-build" -name bcir-opt -type f 2>/dev/null | head -1)}"
if [ -z "${BO}" ]; then
  echo "bcir-opt not built; run tools/wsl/build_mlir.sh first (skipping)." >&2
  exit 0
fi
FC="$(command -v FileCheck-18 || command -v FileCheck || true)"
T="${ROOT}/mlir/test/passes"
fail=0

echo "[passes] -bcir-verify negative cases (-verify-diagnostics)"
"${BO}" -bcir-verify -verify-diagnostics -split-input-file "${T}/verify_laws.mlir" \
  && echo "  PASS verify_laws (R1/R2/R4/R6)" || { echo "  FAIL verify_laws"; fail=1; }

echo "[passes] -bcir-verify on the pretty corpus (must be clean)"
for f in "${ROOT}"/mlir/examples/*.mlir; do
  "${BO}" -bcir-verify "${f}" >/dev/null 2>/tmp/pe \
    && echo "  PASS verify $(basename "${f}")" || { echo "  FAIL verify $(basename "${f}")"; cat /tmp/pe; fail=1; }
done

run_fc() { # pass-flag, test-file
  if [ -n "${FC}" ]; then
    "${BO}" "$1" "$2" 2>/tmp/pe | "${FC}" "$2" \
      && echo "  PASS $(basename "$2")" || { echo "  FAIL $(basename "$2")"; cat /tmp/pe; fail=1; }
  else
    "${BO}" "$1" "$2" >/dev/null 2>/tmp/pe \
      && echo "  RUN-ONLY $(basename "$2")" || { echo "  FAIL $(basename "$2")"; cat /tmp/pe; fail=1; }
  fi
}
echo "[passes] -bcir-promote-lanes"
run_fc -bcir-promote-lanes "${T}/promote_lanes.mlir"
echo "[passes] -convert-bcir-to-llvm"
run_fc -convert-bcir-to-llvm "${T}/convert_to_llvm.mlir"

[ "${fail}" -eq 0 ] && echo "[passes] all Phase-6 passes validate" || echo "[passes] FAILURES"
exit "${fail}"
