#!/usr/bin/env bash
# ptrfield: pointer struct fields -- s->p = q (bcir-cc): emit == Clang (#ptrfield)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-ptrfield`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> ptrfield.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> ptrfield.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Pointer values stored into / loaded from a struct field (#ptrfield): the twin modeled a pointer member
# as 4 bytes, so the struct LAYOUT was wrong (an adjacent field overlapped the high half of the pointer)
# and a store truncated the 8-byte pointer. Now the member occupies pointer_size and the store/load moves
# the full pointer with its real `T *` type. Each function stores then reads back; the scalar between two
# pointers (written first) survives only under the 8-byte layout, and the returned pointers round-trip.
"${BCIR_CC}" --emit-c "${C}/cfront_ptrfield.c" > "${tmp}/pf_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bpf_store_return\b/pf_sr_src/' -e 's/\bpf_layout\b/pf_ly_src/' \
      -e 's/\bpf_long\b/pf_lg_src/' -e 's/\bpf_both\b/pf_bo_src/' "${C}/cfront_ptrfield.c"
  cat "${tmp}/pf_emit.c"
  cat <<'DRV'
int main(void){
  static int iv[300]; static long lv[300]; struct Node a, b;
  for(int i=0;i<290;i++){
    int k = i*7 - 11;
    if(pf_sr_src(&a,&iv[i])           != bcir_pf_store_return(&b,&iv[i]))     {printf("sr MISMATCH i=%d\n",i);return 1;}
    if(pf_ly_src(&a,&iv[i],k)         != bcir_pf_layout(&b,&iv[i],k))         {printf("ly MISMATCH i=%d\n",i);return 1;}
    if(pf_lg_src(&a,&lv[i])           != bcir_pf_long(&b,&lv[i]))             {printf("lg MISMATCH i=%d\n",i);return 1;}
    if(pf_bo_src(&a,&iv[i],&lv[i],k)  != bcir_pf_both(&b,&iv[i],&lv[i],k))    {printf("bo MISMATCH i=%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/pf_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/pf_harness.c" -o "${tmp}/pf_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/pf_harness.c" -o "${tmp}/pf_h" \
  || { echo "  FAIL: ptrfield harness build"; exit 1; }
pfr="$("${tmp}/pf_h")"
[ "${pfr}" = "MATCH" ] \
  && echo "  PASS ptrfield: 8-byte member layout + untruncated store/load == Clang" \
  || { echo "  FAIL: ptrfield behaviour (${pfr})"; exit 1; }
