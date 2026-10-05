#!/usr/bin/env bash
# tree: E5 classical-ML decision-tree predict (emit_tree_predict_c): exact threshold traversal, NO libm (#classical #tree)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-tree`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. The binary is the manifest kernel kernel_tree: C the Python oracle emits
# (bcir.lower.c_kernel) with its driver appended by tools/build/emit_kernel.py, the writer BCIR Make
# and the CMake build share.
#
#   usage: tree.sh <kernel_tree>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: tree.sh <kernel_tree>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
KERNEL="$(abs "$1")"; [ -x "${KERNEL}" ] || { echo "  FAIL: KERNEL: ${KERNEL} is not an executable kernel"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT


tree_out="$("${KERNEL}")"; tree_rc=$?    # rc=0 IS the check: the three leaves match exactly (driver)
{ [ "${tree_rc}" = "0" ] && printf '%s' "${tree_out}" | grep -q "^a=.* b=.* c=.*"; } \
  && echo "  PASS tree: exact threshold traversal returns the known leaves (${tree_out}; NO libm -- the EXACT half of the Area-B pattern)" \
  || { echo "  FAIL: tree predict did not return the known leaves (rc=${tree_rc}: ${tree_out})"; exit 1; }
