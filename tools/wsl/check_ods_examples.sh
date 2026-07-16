#!/usr/bin/env bash
# Validate the *pretty* ODS corpus (mlir/examples/*.mlir) by parsing/verifying it
# with the compiled bcir-opt and checking the embedded `// CHECK:` lines with
# FileCheck. Run tools/wsl/build_mlir.sh first (or set BCIR_OPT).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/bcir-ods.XXXXXXXX")" || exit 1
trap 'rm -rf -- "${TMP_ROOT}"' EXIT
ERR="${TMP_ROOT}/ods.err"
BO="${BCIR_OPT:-$(find "${ROOT}/build/mlir-build" -name bcir-opt -type f 2>/dev/null | head -1)}"
if [ -z "${BO}" ]; then
  echo "bcir-opt not built; run tools/wsl/build_mlir.sh first (skipping)." >&2
  exit 0
fi
# Resolve FileCheck version-agnostically (highest /usr/lib/llvm-*/bin/FileCheck, then PATH names).
FC="$(ls /usr/lib/llvm-*/bin/FileCheck 2>/dev/null | sort -V | tail -1)"
[ -n "${FC}" ] || FC="$(command -v FileCheck-22 || command -v FileCheck-19 || command -v FileCheck-18 || command -v FileCheck || true)"
echo "[ods] bcir-opt: ${BO}"
echo "[ods] FileCheck: ${FC:-<none; parse-only>}"

fail=0
for f in "${ROOT}"/mlir/examples/*.mlir; do
  if [ -n "${FC}" ]; then
    if "${BO}" "${f}" 2>"${ERR}" | "${FC}" "${f}"; then
      echo "PASS $(basename "${f}")"
    else
      echo "FAIL $(basename "${f}")"; cat "${ERR}"; fail=1
    fi
  else
    if "${BO}" "${f}" >/dev/null 2>"${ERR}"; then
      echo "PARSE+VERIFY OK $(basename "${f}")"
    else
      echo "FAIL $(basename "${f}")"; cat "${ERR}"; fail=1
    fi
  fi
done
[ "${fail}" -eq 0 ] && echo "[ods] all pretty ODS examples validate" || echo "[ods] FAILURES"
exit "${fail}"
