#!/usr/bin/env bash
# structmulti: multi-declarator struct members (bcir-cc): layout + access == Clang (#structmulti)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-structmulti` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It compiles what bcir-cc emits with CC, which the caller
# names: BCIR Make passes the gate's, the CTest entry the configured C compiler, and the
# section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> structmulti.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> structmulti.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Multi-declarator struct/union members (#structmulti): `unsigned x, y, z;` -- several members off one
# specifier (incl. multi-declarator bitfields). Each lays out as if on its own line, so offsets / size
# match Clang; the twin's --emit-c accesses members by byte offset into the source-defined struct, so a
# wrong offset would diverge. Differential checks sizeof + per-member round-trip.
"${BCIR_CC}" --emit-c "${C}/cfront_structmulti.c" > "${tmp}/sm_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bpt_sum\b/pt_sum_src/' -e 's/\bfl_pack\b/fl_pack_src/' "${C}/cfront_structmulti.c"
  cat "${tmp}/sm_emit.c"
  cat <<'DRV'
int main(void){
  if(sizeof(struct Pt)!=12u||sizeof(struct Flags)!=4u){printf("LAYOUT Pt=%zu Flags=%zu\n",sizeof(struct Pt),sizeof(struct Flags));return 2;}
  for(unsigned a=0;a<6000u;a++){
    if(pt_sum_src(a)!=bcir_pt_sum(a)){printf("pt MISMATCH a=%u\n",a);return 1;}
    if(fl_pack_src(a)!=bcir_fl_pack(a)){printf("fl MISMATCH a=%u\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sm_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sm_harness.c" -o "${tmp}/sm_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sm_harness.c" -o "${tmp}/sm_h" \
  || { echo "  FAIL: structmulti harness build"; exit 1; }
smr="$("${tmp}/sm_h")"
[ "${smr}" = "MATCH" ] \
  && echo "  PASS structmulti: unsigned x,y,z + multi-bitfield layout/access == Clang" \
  || { echo "  FAIL: structmulti behaviour (${smr})"; exit 1; }
