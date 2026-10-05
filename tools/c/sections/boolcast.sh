#!/usr/bin/env bash
# boolcast: cast to _Bool normalizes the full value (bcir-cc): emit == Clang (#boolcast)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-boolcast` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> boolcast.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> boolcast.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Cast to _Bool / bool (#boolcast): `(_Bool)x` is `x != 0` on the FULL value (§6.3.1.2), so `(_Bool)256`
# is 1 and `(_Bool)0.5` is 1. Both rails rendered it as `(uint8_t)x`, truncating to 8 bits before the bool
# test (`(_Bool)256` -> 0), and the twin did not normalize at all. Now both rails emit a real `_Bool` cast.
"${BCIR_CC}" --emit-c "${C}/cfront_boolcast.c" > "${tmp}/bx_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdbool.h>'; echo '#include <stdio.h>'
  sed -e 's/\bbcast_low\b/bx_lo_src/' -e 's/\bbcast_high\b/bx_hi_src/' -e 's/\bbcast_mask\b/bx_mk_src/' \
      -e 's/\bbcast_float\b/bx_fl_src/' -e 's/\bbcast_arith\b/bx_ar_src/' "${C}/cfront_boolcast.c"
  cat "${tmp}/bx_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<70000u;a+=7u)for(unsigned b=0;b<300u;b++){
    if(bx_lo_src(a,b)!=bcir_bcast_low(a,b)){printf("lo MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bx_hi_src(a,b)!=bcir_bcast_high(a,b)){printf("hi MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bx_mk_src(a,b)!=bcir_bcast_mask(a,b)){printf("mk MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bx_fl_src(a,b)!=bcir_bcast_float(a,b)){printf("fl MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bx_ar_src(a,b)!=bcir_bcast_arith(a,b)){printf("ar MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/bx_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/bx_harness.c" -o "${tmp}/bx_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/bx_harness.c" -o "${tmp}/bx_h" \
  || { echo "  FAIL: boolcast harness build"; exit 1; }
bxr="$("${tmp}/bx_h")"
[ "${bxr}" = "MATCH" ] \
  && echo "  PASS boolcast: (_Bool)x normalizes the full value == Clang" \
  || { echo "  FAIL: boolcast behaviour (${bxr})"; exit 1; }
