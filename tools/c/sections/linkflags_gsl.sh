#!/usr/bin/env bash
# linkflags_gsl: GSL link-flag rule (bcir_cfront_link_flags twin): gsl_* -> -lgsl (#linkflags-gsl)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-linkflags_gsl` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. The binary is the test_link_flag_rules harness, built once by
# BCIR Make for the five link-flag sections, and this section runs its `gsl` table.
#
#   usage: linkflags_gsl.sh <test_link_flag_rules>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: linkflags_gsl.sh <test_link_flag_rules>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Area-B breadth (#62) GSL link-flag rule (dual-rail): the GSL edge (gsl_stats_mean) is minted by the kernel
# EMITTER (emit_gsl_stats_c), not reachable from a cfront source. So this probe drives the C twin's
# bcir_cfront_link_flags over FABRICATED units carrying a `c.call.libm:gsl_stats_mean` edge and asserts it
# derives `-lgsl` (the GSL rule), with any gsl_* -> -lgsl too, and no regression on LAPACKE_* -> -llapack,
# fftwf_* -> -lfftw3f, cblas_* -> -lcblas, libm -> -lm, and unknown -> no flag. The oracle
# (linkflags.library_for_callee) is pinned in test_c_cfront.py + test_gsl.py.
"${HARNESS}" gsl | grep -q "^OK linkflags-gsl" \
  && echo "  PASS linkflags-gsl: C twin derives gsl_* -> -lgsl (lapack/fftw/cblas/-lm/unknown unchanged)" \
  || { echo "  FAIL: GSL link-flag rule diverged on the C twin"; "${HARNESS}" gsl; exit 1; }
