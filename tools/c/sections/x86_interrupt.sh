#!/usr/bin/env bash
# x86_interrupt: x86 interrupt-frame ABI header (C11 + C23)
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them (the harness once per C standard the header promises, from the real source) and runs
# this script as a task, whose verdict the gate shows; the CMake project builds the same binaries
# from the same manifest (the harness and its C11 `variant`) and runs this script as the
# `c-section-x86_interrupt` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical.
#
#   usage: x86_interrupt.sh <test_x86_interrupt_c11> <test_x86_interrupt>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_x86_interrupt_c11> <test_x86_interrupt>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
C11="$(abs "$1")"; [ -x "${C11}" ] || { echo "  FAIL: C11: ${C11} is not an executable harness"; exit 2; }
C23="$(abs "$2")"; [ -x "${C23}" ] || { echo "  FAIL: C23: ${C23} is not an executable harness"; exit 2; }
declare -A BY_STD=([c11]="${C11}" [c23]="${C23}")

# bcir_x86_interrupt.h fixes the long-mode interrupt frame BCIR's drivers see (a 176-byte frame);
# the harness asserts its layout and exercises its helpers, once per standard (C11 and C23).
for std in c11 c23; do
  "${BY_STD[${std}]}" \
    || { echo "  FAIL: x86 interrupt-frame layout/helper under -std=${std}"; exit 1; }
done
echo "  PASS x86 interrupt-frame ABI (fixed 176-byte long-mode frame)"
