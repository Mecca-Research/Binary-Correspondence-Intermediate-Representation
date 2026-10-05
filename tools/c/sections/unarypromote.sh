#!/usr/bin/env bash
# unarypromote: integer-promotion + float unary -/~ (bcir-cc): emit == Clang (#unarypromote)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-unarypromote` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> unarypromote.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> unarypromote.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Integer promotion + float on unary `-` / `~` (#unarypromote): a sub-int unsigned operand promotes to
# *signed* int, so `~(unsigned char)0` is -1 (the twin kept it unsigned -> wrong sign test); and `-x` on
# a float stays float (both rails forced a uint32 temp, truncating -2.5 to a huge integer). Now both rails
# take the promoted operand type. Differential == Clang.
"${BCIR_CC}" --emit-c "${C}/cfront_unarypromote.c" > "${tmp}/up_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bucomplement\b/uc_src/' -e 's/\bunegsign\b/un_src/' -e 's/\bushortshift\b/us_src/' \
      -e 's/\bfnegate\b/fn_src/' -e 's/\bdnegate\b/dn_src/' "${C}/cfront_unarypromote.c"
  cat "${tmp}/up_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<4000u;a++)for(unsigned b=0;b<70u;b++){
    if(uc_src(a,b)!=bcir_ucomplement(a,b)){printf("uc MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(un_src(a,b)!=bcir_unegsign(a,b)){printf("un MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(us_src(a,b)!=bcir_ushortshift(a,b)){printf("us MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(fn_src(a,b)!=bcir_fnegate(a,b)){printf("fn MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(dn_src(a,b)!=bcir_dnegate(a,b)){printf("dn MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/up_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/up_harness.c" -o "${tmp}/up_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/up_harness.c" -o "${tmp}/up_h" \
  || { echo "  FAIL: unarypromote harness build"; exit 1; }
upr="$("${tmp}/up_h")"
[ "${upr}" = "MATCH" ] \
  && echo "  PASS unarypromote: sub-int -> signed int + float -/~ == Clang" \
  || { echo "  FAIL: unarypromote behaviour (${upr})"; exit 1; }
