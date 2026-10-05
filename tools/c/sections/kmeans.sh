#!/usr/bin/env bash
# kmeans: E6 unsupervised K-means assign (emit_kmeans_assign_c): nearest-centroid argmin, NO libm (#kmeans)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-kmeans` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binary is the
# manifest kernel kernel_kmeans: C the Python oracle emits (bcir.lower.c_kernel) with its driver
# appended by tools/build/emit_kernel.py, the writer the gate and the CMake build share.
#
#   usage: kmeans.sh <kernel_kmeans>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: kmeans.sh <kernel_kmeans>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
KERNEL="$(abs "$1")"; [ -x "${KERNEL}" ] || { echo "  FAIL: KERNEL: ${KERNEL} is not an executable kernel"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# E6 (ML-breadth) UNSUPERVISED K-means nearest-centroid ASSIGN (#kmeans): the unsupervised analog of the E5 EXACT
# tree kernel. K-means FIT is a bounded iterative optimization (Lloyd's, library/Python-shaped); ASSIGN over
# BAKED centroids is the fixed-shape PREDICT kernel (G5 baked-weights pattern). It is the EXACT half of the
# Area-B pattern -- argmin_c ||x-centroid[c]||^2 by squared distance: pure subtract/multiply/add + comparisons,
# NO transcendental, NO libm (needs no -lm). It returns an int cluster id, so the C-vs-oracle check is
# INTEGER-EXACT (the same argmin). C twin of kbcir.unsupervised.kmeans_assign.
kmeans_out="$("${KERNEL}")"; kmeans_rc=$?    # rc=0 IS the check: the three argmins match exactly (driver)
{ [ "${kmeans_rc}" = "0" ] && printf '%s' "${kmeans_out}" | grep -q "^a=.* b=.* c=.*"; } \
  && echo "  PASS kmeans: exact nearest-centroid argmin returns the known clusters (${kmeans_out}; NO libm -- the EXACT half of the Area-B pattern, integer-exact)" \
  || { echo "  FAIL: kmeans assign did not return the known clusters (rc=${kmeans_rc}: ${kmeans_out})"; exit 1; }
