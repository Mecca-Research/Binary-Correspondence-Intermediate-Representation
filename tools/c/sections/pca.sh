#!/usr/bin/env bash
# pca: E2 PCA portable fallback (emit_lapack_eigh_c): Jacobi recovers a known spectrum (#pca)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-pca` CTest
# entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so
# the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. The binary is the manifest kernel kernel_pca: C the Python oracle emits
# (bcir.lower.c_kernel) with its driver appended by tools/build/emit_kernel.py, the writer BCIR Make
# and the CMake build share.
#
#   usage: pca.sh <kernel_pca>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: pca.sh <kernel_pca>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
KERNEL="$(abs "$1")"; [ -x "${KERNEL}" ] || { echo "  FAIL: KERNEL: ${KERNEL} is not an executable kernel"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# E2 (ML-breadth) PCA portable fallback (#pca): the SYMMETRIC EIGENDECOMPOSITION wrap (emit_lapack_eigh_c) is
# the PCA sibling of E1's OLS solve -- where OLS forms a symmetric Gram matrix and SOLVES it, PCA forms a
# symmetric covariance and EIGENDECOMPOSES it. The linked path is LAPACKE_ssyev (Householder + implicit-QR);
# the portable fallback is the classic JACOBI rotation sweep (the C twin of kbcir.pca._jacobi_eigh). This probe
# compiles + runs the FALLBACK (no LAPACK needed -- CI is LAPACK-free, exactly the Area-B norm) on a hand-built
# DIAGONAL symmetric matrix diag(5,3,1) with DISTINCT (well-separated) eigenvalues, and checks it recovers the
# eigenvalues DESCENDING [5,3,1] and the standard-basis eigenvectors (sign convention: largest-magnitude entry
# positive). The LAPACKE_ssyev -> -llapack dual-rail is confirmed by the #linkflags-lapack probe above
# (LAPACKE_ssyev rides the SAME LAPACKE_* rule -- no linkflags change).
eigh_out="$("${KERNEL}")"; eigh_rc=$?    # rc=0 IS the recovery check: eigenvalues [5,3,1] + standard-basis vecs
{ [ "${eigh_rc}" = "0" ] && printf '%s' "${eigh_out}" | grep -q "^l0=.* l1=.* l2=.*"; } \
  && echo "  PASS pca: Jacobi fallback recovers diag(5,3,1) (${eigh_out}; LAPACKE_ssyev -> -llapack via #linkflags-lapack)" \
  || { echo "  FAIL: PCA fallback did not recover [5,3,1] (rc=${eigh_rc}: ${eigh_out})"; exit 1; }
