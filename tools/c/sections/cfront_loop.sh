#!/usr/bin/env bash
# cfront_loop: full C compile->execute loop (cfront -> plan -> hydrate -> exec, no Python)
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them from runtime/manifest.json and runs this script as a task, whose verdict the gate
# shows; the CMake project builds them from the same manifest and runs this script as the
# `c-section-cfront_loop` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. The freestanding compiles of bcir_plan.c and bcir_hydrate.c
# are the freestanding section's.
#
#   usage: cfront_loop.sh <test_cfront_loop>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_cfront_loop>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }

for fx in cfront_regmap.c cfront_array.c cfront_array2d.c cfront_widerow.c cfront_deref.c cfront_callgraph.c cfront_typedef.c cfront_enum.c cfront_ternary.c cfront_sizeof.c cfront_cast.c cfront_alignof.c cfront_charlit.c cfront_strtab.c cfront_strconcat.c cfront_widelit.c cfront_static.c cfront_global.c cfront_compound.c cfront_logic.c cfront_atomic.c cfront_cmpxchg.c cfront_atomic11.c cfront_atomic_xchg.c cfront_driver.c cfront_driver_uart.c; do
  out="$("${HARNESS}" "${C}/${fx}")" || { echo "  FAIL: loop ${fx}: ${out}"; exit 1; }
  case "${out}" in
    loop:*executed=*) echo "  PASS loop ${fx} (${out#loop: })" ;;
    *) echo "  FAIL: loop ${fx}: ${out}"; exit 1 ;;
  esac
done
