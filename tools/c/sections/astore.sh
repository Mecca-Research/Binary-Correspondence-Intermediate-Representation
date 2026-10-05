#!/usr/bin/env bash
# astore: array element stores (bcir-cc): a[i]=v emit == Clang (#astore)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-astore` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> astore.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> astore.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Array element stores (#astore): a[i] = v / a[i] OP= v -- the driver buffer-fill / scatter idiom,
# lowered to a 3-read c.store and emitted as a[i] = v. A proper differential drives the source fn over
# buffer A and the twin's bcir_* over buffer B, then compares the buffers (these are void writers).
"${BCIR_CC}" --emit-c "${C}/cfront_arraystore.c" > "${tmp}/st_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; cat "${C}/cfront_arraystore.c" "${tmp}/st_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned t=0;t<3000u;t++){
    unsigned A[32]={0}, B[32]={0}; unsigned i=t%30u, j=(t*3u)%30u, v=t*7u+1u, base=t+5u;
    fill(A,i,base);   bcir_fill(B,i,base);
    scatter(A,i,j,v); bcir_scatter(B,i,j,v);
    accum(A,i,v);     bcir_accum(B,i,v);
    for(int k=0;k<32;k++) if(A[k]!=B[k]){printf("MISMATCH t=%u k=%d\n",t,k);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/st_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/st_harness.c" -o "${tmp}/st_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/st_harness.c" -o "${tmp}/st_h" \
  || { echo "  FAIL: astore harness build"; exit 1; }
str="$("${tmp}/st_h")"
[ "${str}" = "MATCH" ] \
  && echo "  PASS astore: a[i]=v / a[i] OP= v == Clang (source buffer == twin buffer)" \
  || { echo "  FAIL: astore behaviour (${str})"; exit 1; }
