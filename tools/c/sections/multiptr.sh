#!/usr/bin/env bash
# multiptr: per-declarator pointer/array in a multi-declarator decl -- int *p, q; (#multiptr)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-multiptr` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> multiptr.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> multiptr.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Per-declarator pointer/array shape in a multi-declarator declaration (#multiptr): `int *p, q;` types
# p as `int*` and q as `int` (the `*` binds to the declarator); `int *p, *q;` types both as pointers
# (was rejected by the twin). The twin now parses the specifier once and applies each declarator's own
# `*`/`[]` on a fresh copy, for locals + struct members. A differential uses each trailing declarator
# AS a scalar (a wide store would clobber); also pins a `long m` member store moving 8 bytes, not 4.
"${BCIR_CC}" --emit-c "${C}/cfront_multiptr.c" > "${tmp}/mpt_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bmd_local_mixed\b/md_local_mixed_s/' -e 's/\bmd_local_two_ptr\b/md_local_two_ptr_s/' \
      -e 's/\bmd_local_ptr_arr\b/md_local_ptr_arr_s/' -e 's/\bmd_struct\b/md_struct_s/' \
      "${C}/cfront_multiptr.c"
  cat "${tmp}/mpt_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-300;i<300;i++){
    int a=i*3-1,b=7-i;
    if(md_local_mixed_s(i)!=bcir_md_local_mixed(i)){printf("mixed@%d\n",i);return 1;}
    if(md_local_two_ptr_s(a,b)!=bcir_md_local_two_ptr(a,b)){printf("twoptr@%d\n",i);return 1;}
    if(md_local_ptr_arr_s(i)!=bcir_md_local_ptr_arr(i)){printf("ptrarr@%d\n",i);return 1;}
    struct Mix m1,m2;
    if(md_struct_s(&m1,i)!=bcir_md_struct(&m2,i)){printf("struct@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/mpt_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/mpt_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/mpt_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/mpt_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/mpt_h" \
  || { echo "  FAIL: multiptr harness build"; exit 1; }
mptr="$("${tmp}/mpt_h")"
[ "${mptr}" = "MATCH" ] \
  && echo "  PASS multiptr: int *p, q; / int *p, *q; / struct + wide member store == Clang" \
  || { echo "  FAIL: multiptr behaviour (${mptr})"; exit 1; }
