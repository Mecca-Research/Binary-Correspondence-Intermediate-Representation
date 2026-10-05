#!/usr/bin/env bash
# svm: E5 classical-ML RBF-SVM predict (emit_svm_rbf_predict_c): decision function matches reference (#classical #svm)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-svm` CTest
# entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so
# the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. The binary is the manifest kernel kernel_svm: C the Python oracle emits
# (bcir.lower.c_kernel) with its driver appended by tools/build/emit_kernel.py, the writer BCIR Make
# and the CMake build share.
#
#   usage: svm.sh <kernel_svm>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: svm.sh <kernel_svm>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
KERNEL="$(abs "$1")"; [ -x "${KERNEL}" ] || { echo "  FAIL: KERNEL: ${KERNEL} is not an executable kernel"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# E5 (ML-breadth) CLASSICAL-ML PREDICT path: the baked-model fixed-shape predict kernels (#classical). The
# honest framing E7 cites: classical-ML TRAINING (tree induction, the SVM QP solve, NB fitting) is iterative/
# combinatorial -- a POOR fit for BCIR's fixed-shape claim model (library/Python). PREDICT over a BAKED model is
# the opposite: a deterministic, fixed-shape kernel = the G5 baked-weights pattern. Two probes show the Area-B
# pattern covers BOTH halves: the RBF-SVM (transcendental -- its only external is expf on the c.call.libm: edge,
# -lm, ALREADY mapped -- no linkflags change) and the decision tree (EXACT -- pure comparisons + a leaf return,
# NO transcendental, NO libm). C twins of kbcir.classical.svm_decision_rbf / tree_predict.
svm_out="$("${KERNEL}")"; svm_rc=$?    # rc=0 IS the check: f matches the hand-computed value (driver)
{ [ "${svm_rc}" = "0" ] && printf '%s' "${svm_out}" | grep -q "^f=.*"; } \
  && echo "  PASS svm: RBF decision matches reference (${svm_out}; expf -> -lm via the libm rule, no linkflags change)" \
  || { echo "  FAIL: svm RBF decision did not match the reference (rc=${svm_rc}: ${svm_out})"; exit 1; }
