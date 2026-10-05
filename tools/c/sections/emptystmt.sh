#!/usr/bin/env bash
# emptystmt: empty statements (bcir-cc): emit == Clang (#emptystmt)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-emptystmt` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> emptystmt.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> emptystmt.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Empty statements (#emptystmt): a bare `;` -- the body of `for(...);` (work in the header) / `if(c);`
# and stray `;;`. Both rails consume it and emit no claim, so the twin's --emit-c is Clang-equivalent.
"${BCIR_CC}" --emit-c "${C}/cfront_emptystmt.c" > "${tmp}/es_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bcount_below\b/count_below_src/' -e 's/\bmasked\b/masked_src/' "${C}/cfront_emptystmt.c"
  cat "${tmp}/es_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<8000u;a++){
    if(count_below_src(a)!=bcir_count_below(a)){printf("count MISMATCH a=%u\n",a);return 1;}
    if(masked_src(a)!=bcir_masked(a)){printf("masked MISMATCH a=%u\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/es_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/es_harness.c" -o "${tmp}/es_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/es_harness.c" -o "${tmp}/es_h" \
  || { echo "  FAIL: emptystmt harness build"; exit 1; }
esr="$("${tmp}/es_h")"
[ "${esr}" = "MATCH" ] \
  && echo "  PASS emptystmt: for(...); / if(c); / stray ;; == Clang" \
  || { echo "  FAIL: emptystmt behaviour (${esr})"; exit 1; }
