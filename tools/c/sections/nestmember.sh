#!/usr/bin/env bash
# nestmember: nested struct member access (bcir-cc): layout + access == Clang (#nestmember)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-nestmember` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> nestmember.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> nestmember.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Nested struct member access (#nestmember): `o.pos.lo` / `dev->ctrl.flags` -- a struct-in-struct. Both
# rails flatten the chain to one offset access; the twin emits member access by byte offset, so a wrong
# accumulated offset (or layout) would diverge. Differential checks sizeof + nested read/write/bitfield.
"${BCIR_CC}" --emit-c "${C}/cfront_nestmember.c" > "${tmp}/nm_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bouter_sum\b/outer_sum_src/' -e 's/\bdev_pack\b/dev_pack_src/' "${C}/cfront_nestmember.c"
  cat "${tmp}/nm_emit.c"
  cat <<'DRV'
int main(void){
  if(sizeof(struct Outer)!=12u||sizeof(struct Dev)!=8u){printf("LAYOUT O=%zu D=%zu\n",sizeof(struct Outer),sizeof(struct Dev));return 2;}
  for(unsigned a=0;a<8000u;a++){
    if(outer_sum_src(a)!=bcir_outer_sum(a)){printf("outer MISMATCH a=%u\n",a);return 1;}
    if(dev_pack_src(a)!=bcir_dev_pack(a)){printf("dev MISMATCH a=%u\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/nm_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/nm_harness.c" -o "${tmp}/nm_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/nm_harness.c" -o "${tmp}/nm_h" \
  || { echo "  FAIL: nestmember harness build"; exit 1; }
nmr="$("${tmp}/nm_h")"
[ "${nmr}" = "MATCH" ] \
  && echo "  PASS nestmember: o.pos.lo / dev->ctrl.bf layout + access == Clang" \
  || { echo "  FAIL: nestmember behaviour (${nmr})"; exit 1; }
