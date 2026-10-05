#!/usr/bin/env bash
# linkflags_fftw: B2 FFTW link-flag rule (bcir_cfront_link_flags twin): fftwf_* -> -lfftw3 (#linkflags-fftw)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-linkflags_fftw` CTest entry. The body is the gate's
# section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binary is the
# test_link_flag_rules harness, built once by the gate for the five link-flag sections, and this
# section runs its `fftw` table.
#
#   usage: linkflags_fftw.sh <test_link_flag_rules>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: linkflags_fftw.sh <test_link_flag_rules>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# B2 FFTW link-flag rule (dual-rail): the trusted-library edges (cblas_*/fftwf_*) are NOT reachable from a
# cfront SOURCE (an unknown callee lowers to an in-unit `c.call:` edge, not `c.call.libm:`) -- they are
# minted by the kernel EMITTERS (emit_blas_gemm_c / emit_fftw_fft_c). So this probe drives the C twin's
# bcir_cfront_link_flags over a FABRICATED unit carrying a `c.call.libm:fftwf_execute` edge and asserts it
# derives `-lfftw3` (the B2 rule), with a cblas edge -> -lcblas (no B5 regression), a libm edge -> -lm,
# and an unknown edge -> no flag. The oracle (linkflags.library_for_callee) is pinned in test_c_cfront.py.
"${HARNESS}" fftw | grep -q "^OK linkflags-fftw" \
  && echo "  PASS linkflags-fftw: C twin derives fftwf_*/fftw_* -> -lfftw3 (cblas/-lm/unknown unchanged)" \
  || { echo "  FAIL: FFTW link-flag rule diverged on the C twin"; "${HARNESS}" fftw; exit 1; }
