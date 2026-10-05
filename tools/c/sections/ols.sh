#!/usr/bin/env bash
# ols: E1 OLS portable fallback (emit_lapack_ols_c): normal-equations recovers a known x (#ols)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-ols` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binary is the
# manifest kernel kernel_ols: C the Python oracle emits (bcir.lower.c_kernel) with its driver
# appended by tools/build/emit_kernel.py, the writer the gate and the CMake build share.
#
#   usage: ols.sh <kernel_ols>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: ols.sh <kernel_ols>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
KERNEL="$(abs "$1")"; [ -x "${KERNEL}" ] || { echo "  FAIL: KERNEL: ${KERNEL} is not an executable kernel"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# E1 (ML-breadth) OLS portable fallback (#ols): the OVERDETERMINED least-squares wrap (emit_lapack_ols_c)
# generalizes the square sgesv solve to linear regression (minimize ||A x - b||_2). The linked path is the
# QR-based LAPACKE_sgels (~cond(A)); the portable fallback forms the NORMAL EQUATIONS G = A^T A (~cond(A)^2,
# the textbook OLS twin of kbcir.ols.ols_reference). This probe compiles + runs the FALLBACK (no LAPACK
# needed -- CI is LAPACK-free, exactly the Area-B norm) over a CONSISTENT overdetermined system b = A x_true
# and checks it RECOVERS x_true on a well-conditioned A. The LAPACKE_sgels -> -llapack dual-rail is confirmed
# by the #linkflags-lapack probe above (LAPACKE_sgels rides the SAME LAPACKE_* rule -- no linkflags change).
ols_out="$("${KERNEL}")"; ols_rc=$?    # rc=0 IS the recovery check: |c0-1|<1e-3 && |c1-2|<1e-3 (in the driver)
{ [ "${ols_rc}" = "0" ] && printf '%s' "${ols_out}" | grep -q "^c0=.* c1=.*"; } \
  && echo "  PASS ols: normal-equations fallback recovers y=2x+1 (${ols_out}; LAPACKE_sgels -> -llapack via #linkflags-lapack)" \
  || { echo "  FAIL: OLS fallback did not recover [1,2] (rc=${ols_rc}: ${ols_out})"; exit 1; }
