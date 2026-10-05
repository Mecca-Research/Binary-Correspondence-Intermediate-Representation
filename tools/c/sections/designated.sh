#!/usr/bin/env bash
# designated: designated initializers (bcir-cc): dispatch-table emit == Clang (#designated)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-designated` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> designated.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> designated.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Designated initializers for a file-scope table (#designated): `static const T NAME[N] = {[i]=v,...}`
# (the driver opcode-dispatch / jump-table pattern, with a gap that zero-fills). Both rails now parse
# the designated initializer; the table is referenced by name (defined in the source), so the twin's
# --emit-c is Clang-behaviour-equivalent. Compile the emitted bcir_* beside the source + a driver that
# sweeps every opcode (incl. the zero-filled gap).
"${BCIR_CC}" --emit-c "${C}/cfront_dispatch_table.c" > "${tmp}/dt_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; cat "${C}/cfront_dispatch_table.c" "${tmp}/dt_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned op=0; op<10u; op++)
    if(weigh(op)!=bcir_weigh(op)){printf("MISMATCH op=%u: %u vs %u\n",op,weigh(op),bcir_weigh(op));return 1;}
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/dt_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/dt_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/dt_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/dt_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/dt_h" \
  || { echo "  FAIL: designated harness build"; exit 1; }
dtr="$("${tmp}/dt_h")"
[ "${dtr}" = "MATCH" ] \
  && echo "  PASS designated: enum-indexed dispatch table (+ zero-fill gap) == Clang" \
  || { echo "  FAIL: designated behaviour (${dtr})"; exit 1; }
