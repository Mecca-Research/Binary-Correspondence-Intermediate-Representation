#!/usr/bin/env bash
# fieldderef: deref-through a loaded pointer field -- *(s->p) / s->mid->leaf->x / s->p[i] (#fieldderef)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-fieldderef` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It compiles what bcir-cc emits with CC, which the caller
# names: BCIR Make passes the gate's, the CTest entry the configured C compiler, and the
# section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> fieldderef.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> fieldderef.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Deref-through a loaded pointer field (#fieldderef): `*(s->p)`, the chain `s->mid->k` / two-hop
# `s->mid->leaf->x`, and the subscript `s->p[i]` -- reads, writes, RMW. A member used as a base was
# resolved to the struct's address + the field type (so a deref read the struct's own bytes); now a
# pointer-valued field used as a base is loaded and the loaded pointer becomes the new base. A bespoke
# harness builds real Box->Mid->Leaf chains (the generic one would fill a pointee with random bytes).
"${BCIR_CC}" --emit-c "${C}/cfront_fieldderef.c" > "${tmp}/fd_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bfd_read\b/fd_read_s/' -e 's/\bfd_write\b/fd_write_s/' -e 's/\bfd_qread\b/fd_qread_s/' \
      -e 's/\bfd_index\b/fd_index_s/' -e 's/\bfd_index_set\b/fd_index_set_s/' -e 's/\bfd_rmw\b/fd_rmw_s/' \
      -e 's/\bfd_chain1\b/fd_chain1_s/' -e 's/\bfd_chain1_set\b/fd_chain1_set_s/' -e 's/\bfd_chain1_rmw\b/fd_chain1_rmw_s/' \
      -e 's/\bfd_chain2\b/fd_chain2_s/' -e 's/\bfd_chain2_long\b/fd_chain2_long_s/' -e 's/\bfd_chain2_set\b/fd_chain2_set_s/' \
      "${C}/cfront_fieldderef.c"
  cat "${tmp}/fd_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-40;i<2000;i+=7){
    int buf[8]; for(int k=0;k<8;k++) buf[k]=i*k-3;
    long lq=(long)i*1000003L-7;
    struct Box b={0,&buf[0],i,&lq};
    if(fd_read_s(&b)!=bcir_fd_read(&b)){printf("read@%d\n",i);return 1;}
    if(fd_qread_s(&b)!=bcir_fd_qread(&b)){printf("qread@%d\n",i);return 1;}
    if(fd_index_s(&b,5)!=bcir_fd_index(&b,5)){printf("index@%d\n",i);return 1;}
    int w1[4]={0},w2[4]={0};
    struct Box c1={0,&w1[0],0,&lq},c2={0,&w2[0],0,&lq};
    fd_write_s(&c1,i); bcir_fd_write(&c2,i);
    if(w1[0]!=w2[0]){printf("write@%d\n",i);return 1;}
    fd_index_set_s(&c1,3,i); bcir_fd_index_set(&c2,3,i);
    if(w1[3]!=w2[3]){printf("iset@%d\n",i);return 1;}
    if(fd_rmw_s(&c1,i)!=bcir_fd_rmw(&c2,i)||w1[0]!=w2[0]){printf("rmw@%d\n",i);return 1;}
    struct Leaf lf1={i+1,(long)i*7+2},lf2={i+1,(long)i*7+2};
    struct Mid m1={&lf1,i+5},m2={&lf2,i+5};
    struct Box d1={&m1,&buf[0],0,&lq},d2={&m2,&buf[0],0,&lq};
    if(fd_chain1_s(&d1)!=bcir_fd_chain1(&d2)){printf("chain1@%d\n",i);return 1;}
    if(fd_chain2_s(&d1)!=bcir_fd_chain2(&d2)){printf("chain2@%d\n",i);return 1;}
    if(fd_chain2_long_s(&d1)!=bcir_fd_chain2_long(&d2)){printf("chain2l@%d\n",i);return 1;}
    fd_chain1_set_s(&d1,i*3); bcir_fd_chain1_set(&d2,i*3);
    if(m1.k!=m2.k){printf("c1set@%d\n",i);return 1;}
    if(fd_chain1_rmw_s(&d1,i)!=bcir_fd_chain1_rmw(&d2,i)||m1.k!=m2.k){printf("c1rmw@%d\n",i);return 1;}
    fd_chain2_set_s(&d1,i*2); bcir_fd_chain2_set(&d2,i*2);
    if(lf1.x!=lf2.x){printf("c2set@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/fd_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/fd_harness.c" -o "${tmp}/fd_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/fd_harness.c" -o "${tmp}/fd_h" \
  || { echo "  FAIL: fieldderef harness build"; exit 1; }
fdr="$("${tmp}/fd_h")"
[ "${fdr}" = "MATCH" ] \
  && echo "  PASS fieldderef: *(s->p) / s->mid->leaf->x / s->p[i] (read+write+rmw) == Clang" \
  || { echo "  FAIL: fieldderef behaviour (${fdr})"; exit 1; }
