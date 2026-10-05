#!/usr/bin/env bash
# linkflags_lapack: LAPACK link-flag rule (bcir_cfront_link_flags twin): LAPACKE_*/sgesv_ -> -llapack (#linkflags-lapack)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-linkflags_lapack` CTest entry. The body is the gate's
# section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binary is the
# test_link_flag_rules harness, built once by the gate for the five link-flag sections, and this
# section runs its `lapack` table.
#
#   usage: linkflags_lapack.sh <test_link_flag_rules>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: linkflags_lapack.sh <test_link_flag_rules>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# B-breadth (#61) LAPACK link-flag rule (dual-rail): the LAPACK edge (LAPACKE_sgesv) is minted by the kernel
# EMITTER (emit_lapack_solve_c), not reachable from a cfront source. So this probe drives the C twin's
# bcir_cfront_link_flags over FABRICATED units carrying a `c.call.libm:LAPACKE_sgesv` edge and asserts it
# derives `-llapack` (the LAPACK rule), with a Fortran-ABI `sgesv_` edge -> -llapack too, and no regression
# on fftwf_* -> -lfftw3, cblas_* -> -lcblas, libm -> -lm, and unknown -> no flag. The oracle
# (linkflags.library_for_callee) is pinned in test_c_cfront.py + test_lapack.py.
"${HARNESS}" lapack | grep -q "^OK linkflags-lapack" \
  && echo "  PASS linkflags-lapack: C twin derives LAPACKE_*/sgels/sgesv_ -> -llapack (fftw/cblas/-lm/unknown unchanged)" \
  || { echo "  FAIL: LAPACK link-flag rule diverged on the C twin"; "${HARNESS}" lapack; exit 1; }
