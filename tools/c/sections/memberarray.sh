#!/usr/bin/env bash
# memberarray: struct member arrays (bcir-cc): layout + access == Clang (#memberarray)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-memberarray` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> memberarray.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> memberarray.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Native struct member arrays (#memberarray): `s.arr[i]` -- a 1-D array member. The element lands at
# `&base + member_off + i*elem_size`; the twin emits this as a memcpy at that offset, so a wrong stride
# or layout would diverge. Differential checks sizeof + read/write/compound + a narrowing uint8 element.
"${BCIR_CC}" --emit-c "${C}/cfront_memberarray.c" > "${tmp}/mar_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bpkt_sum\b/pkt_src/' -e 's/\bbuf_pack\b/buf_src/' -e 's/\bgrid_pick\b/grid_src/' "${C}/cfront_memberarray.c"
  cat "${tmp}/mar_emit.c"
  cat <<'DRV'
int main(void){
  if(sizeof(struct Packet)!=28u||sizeof(struct Buf)!=12u||sizeof(struct Grid)!=52u){printf("LAYOUT bad\n");return 2;}
  for(unsigned i=0;i<6000u;i++)for(unsigned a=0;a<40u;a++){
    if(pkt_src(i,a)!=bcir_pkt_sum(i,a)){printf("pkt MISMATCH i=%u a=%u\n",i,a);return 1;}
    if(buf_src(i,a)!=bcir_buf_pack(i,a)){printf("buf MISMATCH i=%u a=%u\n",i,a);return 1;}
    if(grid_src(i,a)!=bcir_grid_pick(i,a)){printf("grid MISMATCH i=%u a=%u\n",i,a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/mar_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/mar_harness.c" -o "${tmp}/mar_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/mar_harness.c" -o "${tmp}/mar_h" \
  || { echo "  FAIL: memberarray harness build"; exit 1; }
marr="$("${tmp}/mar_h")"
[ "${marr}" = "MATCH" ] \
  && echo "  PASS memberarray: s.arr[i] read/write/compound + uint8 element == Clang" \
  || { echo "  FAIL: memberarray behaviour (${marr})"; exit 1; }
