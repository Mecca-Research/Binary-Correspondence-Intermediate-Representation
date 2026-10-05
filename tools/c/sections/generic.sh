#!/usr/bin/env bash
# generic: _Generic -- C11 type-generic selection (#generic)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-generic` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> generic.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> generic.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# _Generic (#generic): C11 generic selection on the controlling expression's static type. The differential
# drives int / long / unsigned / double / float / char / pointer controls -- a wrong type-match would pick
# the wrong arm and diverge. Both rails read the controlling type the same way (the twin off a
# speculatively-lowered value) so they select the same association.
"${BCIR_CC}" --emit-c "${C}/cfront_generic.c" > "${tmp}/gn_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bg_int\b/g_int_s/g' -e 's/\bg_long\b/g_long_s/g' -e 's/\bg_uint\b/g_uint_s/g' \
      -e 's/\bg_double\b/g_double_s/g' -e 's/\bg_float\b/g_float_s/g' -e 's/\bg_char\b/g_char_s/g' \
      -e 's/\bg_ptr\b/g_ptr_s/g' -e 's/\bg_exprtype\b/g_exprtype_s/g' -e 's/\bg_default\b/g_default_s/g' \
      -e 's/\bg_compute\b/g_compute_s/g' \
      "${C}/cfront_generic.c"
  cat "${tmp}/gn_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-200;i<200;i++){ long b=(long)i*7777; int x=i;
    if(g_int_s(i)!=bcir_g_int(i)){puts("g_int");return 1;}
    if(g_long_s(b)!=bcir_g_long(b)){puts("g_long");return 1;}
    if(g_uint_s((unsigned)i)!=bcir_g_uint((unsigned)i)){puts("g_uint");return 1;}
    if(g_double_s((double)i)!=bcir_g_double((double)i)){puts("g_double");return 1;}
    if(g_float_s((float)i)!=bcir_g_float((float)i)){puts("g_float");return 1;}
    if(g_char_s((char)i)!=bcir_g_char((char)i)){puts("g_char");return 1;}
    if(g_ptr_s(&x)!=bcir_g_ptr(&x)){puts("g_ptr");return 1;}
    if(g_exprtype_s(i,b)!=bcir_g_exprtype(i,b)){puts("g_exprtype");return 1;}
    if(g_default_s((double)i)!=bcir_g_default((double)i)){puts("g_default");return 1;}
    if(g_compute_s(b)!=bcir_g_compute(b)){puts("g_compute");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/gn_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/gn_harness.c" -o "${tmp}/gn_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/gn_harness.c" -o "${tmp}/gn_h" \
  || { echo "  FAIL: generic harness build"; exit 1; }
gnr="$("${tmp}/gn_h")"
[ "${gnr}" = "MATCH" ] \
  && echo "  PASS generic: _Generic selection (int/long/uint/double/float/char/ptr/default) == Clang" \
  || { echo "  FAIL: generic behaviour (${gnr})"; exit 1; }
