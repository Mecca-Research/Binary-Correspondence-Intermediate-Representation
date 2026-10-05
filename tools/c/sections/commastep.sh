#!/usr/bin/env bash
# commastep: comma-operator for-step (bcir-cc): emit == Clang (#commastep)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-commastep` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> commastep.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> commastep.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Comma operator in the for-step (#commastep): `for(...; ...; i++, j--)` -- two-pointer / reversal loops
# and parallel-counter updates. Each comma-separated step element runs in order every iteration, so the
# twin's --emit-c (which loops the recorded step tokens on commas) is Clang-behaviour-equivalent.
"${BCIR_CC}" --emit-c "${C}/cfront_commastep.c" > "${tmp}/cs_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bspan_xor\b/span_xor_src/' -e 's/\btwin_acc\b/twin_acc_src/' "${C}/cfront_commastep.c"
  cat "${tmp}/cs_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned n=0;n<6000u;n++){
    if(span_xor_src(n)!=bcir_span_xor(n)){printf("span_xor MISMATCH n=%u\n",n);return 1;}
    if(twin_acc_src(n)!=bcir_twin_acc(n)){printf("twin_acc MISMATCH n=%u\n",n);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/cs_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/cs_harness.c" -o "${tmp}/cs_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/cs_harness.c" -o "${tmp}/cs_h" \
  || { echo "  FAIL: commastep harness build"; exit 1; }
csr="$("${tmp}/cs_h")"
[ "${csr}" = "MATCH" ] \
  && echo "  PASS commastep: i++,j-- + parallel compound-assign steps == Clang" \
  || { echo "  FAIL: commastep behaviour (${csr})"; exit 1; }
