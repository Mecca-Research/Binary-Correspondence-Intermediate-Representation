#!/usr/bin/env bash
# ptrstore: store through a pointer (bcir-cc): emit == Clang (#ptrstore)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-ptrstore` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> ptrstore.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> ptrstore.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Store through a pointer (#ptrstore): `*p = v` / `*p OP= v` / `*(p + i) = v` -- the write counterpart
# of the deref load. The twin parsed `*p` only as a read; now it lowers a deref store (offset-0 imm for
# `*p`, the indexed `p[i]` shape for `*(p + i)`). Verified on INDEPENDENT buffers (these mutate through p).
"${BCIR_CC}" --emit-c "${C}/cfront_ptrstore.c" > "${tmp}/ps_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bstore_scaled\b/store_scaled_src/' -e 's/\baccum_at\b/accum_at_src/' \
      -e 's/\bset_then_bump\b/set_then_bump_src/' "${C}/cfront_ptrstore.c"
  cat "${tmp}/ps_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned v=0;v<4000u;v++){
    unsigned a=v,b=v; store_scaled_src(&a,v); bcir_store_scaled(&b,v);
    if(a!=b){printf("store MISMATCH v=%u\n",v);return 1;}
    unsigned b1[8],b2[8]; for(int k=0;k<8;k++){b1[k]=b2[k]=v+(unsigned)k;}
    unsigned i=v%8u; accum_at_src(b1,i,v); bcir_accum_at(b2,i,v);
    for(int k=0;k<8;k++) if(b1[k]!=b2[k]){printf("accum MISMATCH v=%u\n",v);return 1;}
    unsigned c=0,d=0; set_then_bump_src(&c,v); bcir_set_then_bump(&d,v);
    if(c!=d){printf("setbump MISMATCH v=%u\n",v);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ps_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ps_harness.c" -o "${tmp}/ps_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ps_harness.c" -o "${tmp}/ps_h" \
  || { echo "  FAIL: ptrstore harness build"; exit 1; }
psr="$("${tmp}/ps_h")"
[ "${psr}" = "MATCH" ] \
  && echo "  PASS ptrstore: *p= / *p OP= / *(p+i)= mutate-through-pointer == Clang" \
  || { echo "  FAIL: ptrstore behaviour (${psr})"; exit 1; }
