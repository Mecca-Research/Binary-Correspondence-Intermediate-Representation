#!/usr/bin/env bash
# bcir_cc: bcir-cc compiler driver: compile a driver (sibling header) + emit artifacts
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-bcir_cc`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical.
#
#   usage: bcir_cc.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
[ $# -eq 1 ] || { echo "usage: bcir_cc.sh <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

ccsum="$("${BCIR_CC}" "${C}/cfront_driver_uart.c")" || { echo "  FAIL: bcir-cc compile"; echo "${ccsum}"; exit 1; }
# Phase 3 breadth: the CMSIS-style GPIO fixture (__IO macro, RESERVED pads, write-only BSRR,
# RCC gate-first) must ALSO compile clean through the C rail -- real header shapes, not just
# the synthetic UART block.
gpsum="$("${BCIR_CC}" "${C}/cfront_driver_gpio.c")" || { echo "  FAIL: bcir-cc gpio compile"; echo "${gpsum}"; exit 1; }
case "${gpsum}" in
  *ok=1*) echo "  PASS bcir-cc CMSIS gpio fixture (${gpsum##*: })" ;;
  *) echo "  FAIL: bcir-cc gpio: ${gpsum}"; exit 1 ;;
esac
case "${ccsum}" in
  *ok=1*) echo "  PASS bcir-cc compile (${ccsum##*: })" ;;
  *) echo "  FAIL: bcir-cc compile: ${ccsum}"; exit 1 ;;
esac

"${BCIR_CC}" --emit-pack -o "${tmp}/uart.pack" "${C}/cfront_driver_uart.c" || { echo "  FAIL: bcir-cc --emit-pack"; exit 1; }
[ "$(head -c4 "${tmp}/uart.pack")" = "BSPK" ] \
  && echo "  PASS bcir-cc --emit-pack (valid StreamPack)" \
  || { echo "  FAIL: bcir-cc --emit-pack: bad magic"; exit 1; }
