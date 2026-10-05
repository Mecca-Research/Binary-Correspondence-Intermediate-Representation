#!/usr/bin/env bash
# aggregate: local aggregate init (bcir-cc): struct/union {.field=v} emit == Clang (#aggregate)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-aggregate` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> aggregate.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> aggregate.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Local aggregate initializers for a struct/union (#aggregate): `struct cfg c = {.field=v, ...}` lowers
# to a `= {}` zero baseline + a c.store per initialized member (uninitialized members zero-fill). The
# twin's --emit-c is Clang-behaviour-equivalent. Compile the emitted bcir_* beside the source + a driver.
"${BCIR_CC}" --emit-c "${C}/cfront_agginit.c" > "${tmp}/ag_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; cat "${C}/cfront_agginit.c" "${tmp}/ag_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned x=0; x<5000u; x++)
    if(config(x)!=bcir_config(x)||positional(x)!=bcir_positional(x)||overlap(x)!=bcir_overlap(x)){
      printf("MISMATCH x=%u\n",x);return 1;}
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ag_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ag_harness.c" -o "${tmp}/ag_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ag_harness.c" -o "${tmp}/ag_h" \
  || { echo "  FAIL: aggregate harness build"; exit 1; }
agr="$("${tmp}/ag_h")"
[ "${agr}" = "MATCH" ] \
  && echo "  PASS aggregate: struct/union designated + positional init (+ zero-fill) == Clang" \
  || { echo "  FAIL: aggregate behaviour (${agr})"; exit 1; }
grep -qF "= {};" "${tmp}/ag_emit.c" \
  && echo "  PASS aggregate: emit carries the = {} zero baseline" \
  || { echo "  FAIL: aggregate emit missing zero baseline"; exit 1; }
