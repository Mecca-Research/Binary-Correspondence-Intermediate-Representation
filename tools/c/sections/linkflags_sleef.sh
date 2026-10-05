#!/usr/bin/env bash
# linkflags_sleef: SLEEF link-flag rule (bcir_cfront_link_flags twin): Sleef_* -> -lsleef (#linkflags-sleef)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-linkflags_sleef` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. The binary is the test_link_flag_rules harness, built once by
# BCIR Make for the five link-flag sections, and this section runs its `sleef` table.
#
#   usage: linkflags_sleef.sh <test_link_flag_rules>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: linkflags_sleef.sh <test_link_flag_rules>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Area-B breadth (#63) SLEEF link-flag rule (dual-rail): the SLEEF edge (Sleef_expf1_u10) is minted by the
# kernel EMITTER (emit_sleef_exp_c), not reachable from a cfront source. So this probe drives the C twin's
# bcir_cfront_link_flags over FABRICATED units carrying a `c.call.libm:Sleef_expf1_u10` edge and asserts it
# derives `-lsleef` (the SLEEF rule), with any Sleef_* -> -lsleef too, and no regression on gsl_* -> -lgsl,
# LAPACKE_* -> -llapack, fftwf_* -> -lfftw3, cblas_* -> -lcblas, libm -> -lm, and unknown -> no flag. The
# oracle (linkflags.library_for_callee) is pinned in test_c_cfront.py + test_sleef.py.
"${HARNESS}" sleef | grep -q "^OK linkflags-sleef" \
  && echo "  PASS linkflags-sleef: C twin derives Sleef_* -> -lsleef (gsl/lapack/fftw/cblas/-lm/unknown unchanged)" \
  || { echo "  FAIL: SLEEF link-flag rule diverged on the C twin"; "${HARNESS}" sleef; exit 1; }
