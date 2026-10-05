#!/usr/bin/env bash
# nestoffset: nested member access at a non-first offset (#nestoffset)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-nestoffset` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> nestoffset.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> nestoffset.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Nested member access at a non-first offset (#nestoffset): `t.q.a` where `q` is not the first member --
# the oracle dropped q's byte offset (read+write), the twin over-aligned a nested struct member to its
# size. The differential drives read/write, a member array, a 3-level chain, and a non-first designated
# init; a dropped offset / wrong alignment would alias members and diverge.
"${BCIR_CC}" --emit-c "${C}/cfront_nestoffset.c" > "${tmp}/no_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bno_rw\b/no_rw_s/g' -e 's/\bno_memarr\b/no_memarr_s/g' -e 's/\bno_deep\b/no_deep_s/g' \
      -e 's/\bno_desig\b/no_desig_s/g' \
      "${C}/cfront_nestoffset.c"
  cat "${tmp}/no_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-300;x<300;x++){
    if(no_rw_s(x)!=bcir_no_rw(x)){printf("rw@%d\n",x);return 1;}
    if(no_memarr_s(x)!=bcir_no_memarr(x)){printf("memarr@%d\n",x);return 1;}
    if(no_deep_s(x)!=bcir_no_deep(x)){printf("deep@%d\n",x);return 1;}
    if(no_desig_s(x)!=bcir_no_desig(x)){printf("desig@%d\n",x);return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/no_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/no_harness.c" -o "${tmp}/no_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/no_harness.c" -o "${tmp}/no_h" \
  || { echo "  FAIL: nestoffset harness build"; exit 1; }
nor="$("${tmp}/no_h")"
[ "${nor}" = "MATCH" ] \
  && echo "  PASS nestoffset: non-first nested member read/write/array/chain/designated == Clang" \
  || { echo "  FAIL: nestoffset behaviour (${nor})"; exit 1; }
