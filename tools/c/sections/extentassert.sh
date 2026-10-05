#!/usr/bin/env bash
# extentassert: rid->extent tamper-evidence: BCIR_EXTENT_ASSERT (freestanding lightweight check) (#extentassert)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-extentassert` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binary is the
# test_extent_assert harness, a correct extent that compiled; the script compiles the same unit
# with a tampered extent under CC, which the caller names: the gate passes its own, the CTest entry
# the configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> extentassert.sh <test_extent_assert>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> extentassert.sh <test_extent_assert>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <test_extent_assert>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# rid->extent tamper-evidence (#extentassert, §5.12): the guard trusts the inline `n`, and the freestanding
# unit has NO registry to resolve `rid`->extent (a DOCUMENTED limit; see bcir_quarantine.h). The feasible
# lightweight check, for a KNOWN-EXTENT array, ties `n` to the array's true storage via a COMPILE-TIME
# BCIR_EXTENT_ASSERT(arr, n) == _Static_assert(n == sizeof(arr)/sizeof(arr[0])): a correct extent compiles,
# a TAMPERED `n` fails to compile -- so the extent is tamper-evident with no runtime cost and no registry.
"${HARNESS}" || { echo "  FAIL: BCIR_EXTENT_ASSERT rejected a CORRECT extent"; exit 1; }
# The tampered unit must fail on the ASSERTION, in a standard CC compiles the correct unit in. The gate
# used to compile it with -std=c23 alone, which GCC 13 does not know: that compile failed on the option,
# never reaching the assertion, and the check passed having judged nothing (laws.md L2).
ext_std=""
for std in c23 c2x c11; do
  if "${CC}" -std=${std} -fsyntax-only -I "${C}" "${C}/test_extent_assert.c" 2>/dev/null; then
    ext_std="${std}"; break
  fi
done
[ -n "${ext_std}" ] || { echo "  FAIL: ${CC} compiles the correct extent under none of C23, C2x or C11"; exit 1; }
if "${CC}" -std=${ext_std} -fsyntax-only -DBCIR_TEST_EXTENT=99u -I "${C}" "${C}/test_extent_assert.c" \
     2> "${tmp}/ext_bad.err"; then
  echo "  FAIL: BCIR_EXTENT_ASSERT did NOT catch a tampered extent (n=99 vs storage 8)"; exit 1
fi
grep -q "guard extent disagrees with array storage" "${tmp}/ext_bad.err" \
  || { echo "  FAIL: the tampered extent failed to compile, but not on BCIR_EXTENT_ASSERT:"; sed 's/^/    /' "${tmp}/ext_bad.err"; exit 1; }
echo "  PASS extentassert: correct extent compiles; tampered n=99 fails to compile (tamper-evident) (#extentassert)"
