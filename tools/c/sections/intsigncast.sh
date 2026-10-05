#!/usr/bin/env bash
# intsigncast: integer cast to a signed type (bcir-cc): emit == Clang (#intsigncast)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-intsigncast` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> intsigncast.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> intsigncast.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Pure-integer narrowing cast to a SIGNED integer used directly (#intsigncast): the twin typed every
# integer cast temp as unsigned, so a signed sub-int target lost its sign when the cast value was used
# without landing in a signed named local (`(signed char)(-5)` -> 251), and `(int)u` read back unsigned
# (a logical `>>`). Now the twin types the cast temp with the target's signedness, matching the oracle.
"${BCIR_CC}" --emit-c "${C}/cfront_intsigncast.c" > "${tmp}/is_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bi2schar\b/i2c_src/' -e 's/\bi2short\b/i2s_src/' -e 's/\bi2int_shr\b/i2i_src/' \
      -e 's/\bi2schar_arith\b/i2a_src/' -e 's/\bi2long_div\b/i2l_src/' -e 's/\bi2uchar\b/i2u_src/' \
      "${C}/cfront_intsigncast.c"
  cat "${tmp}/is_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<5000u;a++)for(unsigned b=0;b<80u;b++){
    if(i2c_src(a,b)!=bcir_i2schar(a,b)){printf("i2c MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2s_src(a,b)!=bcir_i2short(a,b)){printf("i2s MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2i_src(a,b)!=bcir_i2int_shr(a,b)){printf("i2i MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2a_src(a,b)!=bcir_i2schar_arith(a,b)){printf("i2a MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2l_src(a,b)!=bcir_i2long_div(a,b)){printf("i2l MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(i2u_src(a,b)!=bcir_i2uchar(a,b)){printf("i2u MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/is_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/is_harness.c" -o "${tmp}/is_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/is_harness.c" -o "${tmp}/is_h" \
  || { echo "  FAIL: intsigncast harness build"; exit 1; }
isr="$("${tmp}/is_h")"
[ "${isr}" = "MATCH" ] \
  && echo "  PASS intsigncast: integer cast to a signed type == Clang" \
  || { echo "  FAIL: intsigncast behaviour (${isr})"; exit 1; }
