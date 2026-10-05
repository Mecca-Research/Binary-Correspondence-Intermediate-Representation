#!/usr/bin/env bash
# lstm: E4 recurrent LSTM cell (emit_lstm_cell_c): 1x1 cell matches hand-computed forward (#lstm)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-lstm` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binary is the
# manifest kernel kernel_lstm: C the Python oracle emits (bcir.lower.c_kernel) with its driver
# appended by tools/build/emit_kernel.py, the writer the gate and the CMake build share.
#
#   usage: lstm.sh <kernel_lstm>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: lstm.sh <kernel_lstm>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
KERNEL="$(abs "$1")"; [ -x "${KERNEL}" ] || { echo "  FAIL: KERNEL: ${KERNEL} is not an executable kernel"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# E4 (ML-breadth) recurrent cells: the LSTM CELL kernel (emit_lstm_cell_c) -- the recurrent-cell executable seam
# (#lstm). E4 is a TWO-TIER design: Tier A (the closed-set relu-RNN) is already lowerable through the EXISTING
# autodiff/closed-set machinery (relu = select, exact, no libm), so the net-new C seam is the TRANSCENDENTAL
# tier, of which the LSTM cell is the canonical gate-rich representative: per unit f=sigmoid(Wf x+Uf h+bf),
# i,o likewise, g=tanhf(Wg x+Ug h+bg), c=f*c_prev+i*g, h=o*tanhf(c). Its only transcendentals are tanhf + the
# expf inside the (numerically-guarded) sigmoid, BOTH riding the c.call.libm: edge (-lm, ALREADY mapped -- no
# linkflags change; sqrtf/tanhf/expf all ride the libm rule, confirmed by the #linkflags-* probes). This probe
# compiles + runs the kernel (only libm needed) on a 1x1 cell with KNOWN weights (W_*=1, U_*=0, b_*=0, x=0.5,
# h_prev=0, c_prev=0) and asserts it matches the hand-computed output to float round-off -- the C twin of
# kbcir.recurrent.lstm_cell_reference.
lstm_out="$("${KERNEL}")"; lstm_rc=$?    # rc=0 IS the check: H/C match the hand-computed reference (driver)
{ [ "${lstm_rc}" = "0" ] && printf '%s' "${lstm_out}" | grep -q "^h=.* c=.*"; } \
  && echo "  PASS lstm: 1x1 cell matches hand-computed forward (${lstm_out}; tanhf/expf -> -lm via the libm rule, no linkflags change)" \
  || { echo "  FAIL: lstm cell did not match the reference (rc=${lstm_rc}: ${lstm_out})"; exit 1; }
