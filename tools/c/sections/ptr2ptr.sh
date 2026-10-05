#!/usr/bin/env bash
# ptr2ptr: pointer-to-pointer -- int **pp, **pp (bcir-cc): emit == Clang (#ptr2ptr)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-ptr2ptr`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> ptr2ptr.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> ptr2ptr.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Pointer-to-pointer (#ptr2ptr): `int **pp`, `**pp`, `*pp = q` (out-param), `**pp = v`, `int **pp = &p`.
# Both rails modeled `int **` as a single `int *` (no indirection depth), so `*pp` read the base width,
# `**pp` fell back, and a store truncated. Now the type carries a pointer DEPTH. A bespoke harness (the
# generic one would fill a pointee with random bytes, invalid to deref for a double pointer) builds real
# x / &x / &&x chains.
"${BCIR_CC}" --emit-c "${C}/cfront_ptr2ptr.c" > "${tmp}/pp_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bp2_read\b/p2_read_s/' -e 's/\bp2_get\b/p2_get_s/' -e 's/\bp2_set\b/p2_set_s/' \
      -e 's/\bp2_store_through\b/p2_st_s/' -e 's/\bp2_rmw\b/p2_rmw_s/' -e 's/\bp2_local\b/p2_local_s/' \
      "${C}/cfront_ptr2ptr.c"
  sed -e 's/\bbcir_p2_store_through\b/bcir_p2_st/' "${tmp}/pp_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-50;i<3000;i+=7){
    int x=i*3-7; int *px=&x; int **ppx=&px;
    if(p2_read_s(ppx)!=bcir_p2_read(ppx)){printf("read@%d\n",i);return 1;}
    if(p2_get_s(ppx)!=bcir_p2_get(ppx)){printf("get@%d\n",i);return 1;}
    int y=i+9; int *a1=px,*a2=px; int **b1=&a1,**b2=&a2;
    p2_set_s(b1,&y); bcir_p2_set(b2,&y);
    if(*b1!=*b2){printf("set@%d\n",i);return 1;}
    int s1=x,s2=x; int *p1=&s1,*p2=&s2; int **q1=&p1,**q2=&p2;
    if(p2_st_s(q1,i)!=bcir_p2_st(q2,i)||s1!=s2){printf("store@%d\n",i);return 1;}
    int u1=x,u2=x; int *r1=&u1,*r2=&u2; int **w1=&r1,**w2=&r2;
    if(p2_rmw_s(w1,i)!=bcir_p2_rmw(w2,i)||u1!=u2){printf("rmw@%d\n",i);return 1;}
    if(p2_local_s(i)!=bcir_p2_local(i)){printf("local@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/pp_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/pp_harness.c" -o "${tmp}/pp_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/pp_harness.c" -o "${tmp}/pp_h" \
  || { echo "  FAIL: ptr2ptr harness build"; exit 1; }
ppr="$("${tmp}/pp_h")"
[ "${ppr}" = "MATCH" ] \
  && echo "  PASS ptr2ptr: int **pp / **pp / *pp=q / **pp=v / &p == Clang" \
  || { echo "  FAIL: ptr2ptr behaviour (${ppr})"; exit 1; }
