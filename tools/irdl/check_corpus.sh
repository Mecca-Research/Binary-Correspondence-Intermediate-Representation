#!/usr/bin/env bash
# Validate the BCIR IRDL projection by round-tripping every generic-syntax corpus
# file in mlir/test/irdl/ through stock mlir-opt with --irdl-file, and checking
# the embedded `// CHECK:` patterns appear in the output (a FileCheck-free check,
# so it runs anywhere mlir-opt is present).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IRDL="${ROOT}/mlir/irdl/bcir.irdl.mlir"
CORPUS="${ROOT}/mlir/test/irdl"

# Resolve mlir-opt version-agnostically (highest /usr/lib/llvm-*/bin/mlir-opt, then PATH names).
MLIR_OPT="${MLIR_OPT:-$(ls /usr/lib/llvm-*/bin/mlir-opt 2>/dev/null | sort -V | tail -1)}"
MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt-22 || command -v mlir-opt || command -v mlir-opt-18 || true)}"
if [ -z "${MLIR_OPT}" ]; then
  echo "mlir-opt not found; skipping IRDL corpus validation (pure-IR rail is optional)." >&2
  exit 0
fi

echo "[irdl] mlir-opt: ${MLIR_OPT} ($(${MLIR_OPT} --version | head -1))"
echo "[irdl] projection: ${IRDL}"
"${MLIR_OPT}" --irdl-file="${IRDL}" /dev/null >/dev/null || { echo "[irdl] projection failed to load"; exit 1; }

fail=0
for f in "${CORPUS}"/*.mlir; do
  out="$("${MLIR_OPT}" --irdl-file="${IRDL}" "$f" 2>/tmp/irdl_err)" \
    || { echo "FAIL (load) $(basename "$f")"; cat /tmp/irdl_err; fail=1; continue; }
  n=0; miss=0
  while IFS= read -r pat; do
    n=$((n+1))
    printf '%s' "$out" | grep -qF "$pat" || { echo "  MISS [$(basename "$f")] $pat"; miss=1; fail=1; }
  done < <(grep -oP '(?<=// CHECK: ).*' "$f")
  [ "$miss" -eq 0 ] && echo "PASS $(basename "$f") (${n} checks)"
done

if [ "$fail" -eq 0 ]; then echo "[irdl] all corpus files round-trip"; else echo "[irdl] FAILURES"; fi
exit "$fail"
