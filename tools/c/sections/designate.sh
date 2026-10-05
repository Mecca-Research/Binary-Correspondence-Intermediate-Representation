#!/usr/bin/env bash
# designate: nested designated initializers -- .a.b / .v[i] / .m[i][j] (#designate)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-designate` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> designate.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> designate.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Nested / chained designated initializers (#designate): a designator list `.a.b` / `.v[i]` / `.m[i][j]`
# resolving to a cumulative byte offset. The differential drives nested struct chains, 1-D/2-D member
# arrays, out-of-order mixes and a 3-level chain -- a wrong offset would store to the wrong slot.
"${BCIR_CC}" --emit-c "${C}/cfront_designate.c" > "${tmp}/de_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bdesig_chain\b/desig_chain_s/g' -e 's/\bdesig_memarr\b/desig_memarr_s/g' \
      -e 's/\bdesig_md\b/desig_md_s/g' -e 's/\bdesig_mix\b/desig_mix_s/g' -e 's/\bdesig_deep\b/desig_deep_s/g' \
      "${C}/cfront_designate.c"
  cat "${tmp}/de_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-300;x<300;x++){
    if(desig_chain_s(x)!=bcir_desig_chain(x)){printf("chain@%d\n",x);return 1;}
    if(desig_memarr_s(x)!=bcir_desig_memarr(x)){printf("memarr@%d\n",x);return 1;}
    if(desig_md_s(x)!=bcir_desig_md(x)){printf("md@%d\n",x);return 1;}
    if(desig_mix_s(x)!=bcir_desig_mix(x)){printf("mix@%d\n",x);return 1;}
    if(desig_deep_s(x)!=bcir_desig_deep(x)){printf("deep@%d\n",x);return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/de_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/de_harness.c" -o "${tmp}/de_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/de_harness.c" -o "${tmp}/de_h" \
  || { echo "  FAIL: designate harness build"; exit 1; }
der="$("${tmp}/de_h")"
[ "${der}" = "MATCH" ] \
  && echo "  PASS designate: .a.b / .v[i] / .m[i][j] / 3-level chain == Clang" \
  || { echo "  FAIL: designate behaviour (${der})"; exit 1; }
