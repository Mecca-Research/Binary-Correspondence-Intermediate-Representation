#!/usr/bin/env bash
# boolnorm: store into a _Bool normalizes to 0/1 (bcir-cc): emit == Clang (#boolnorm)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-boolnorm` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> boolnorm.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> boolnorm.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Storing a value into a `_Bool` / `bool` object (#boolnorm): C normalizes any nonzero to 1 on the
# conversion to bool (§6.3.1.2). The twin emitted a bool local as a plain `uint8_t`, so the store kept
# the raw value (`_Bool x = 2` left x == 2): a silent miscompile. Now the twin emits `_Bool`, so the
# store normalizes -- on a decl init, compound assignment, array element, parameter, and return.
"${BCIR_CC}" --emit-c "${C}/cfront_boolnorm.c" > "${tmp}/bn_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bbool_norm\b/bn_src/' -e 's/\bbool_mask\b/bm_src/' -e 's/\bbool_mod\b/bd_src/' \
      -e 's/\bbool_compound\b/bc_src/' -e 's/\bbool_array\b/ba_src/' "${C}/cfront_boolnorm.c"
  cat "${tmp}/bn_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<300u;a++)for(unsigned b=0;b<260u;b++){
    if(bn_src(a,b)!=bcir_bool_norm(a,b)){printf("bn MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bm_src(a,b)!=bcir_bool_mask(a,b)){printf("bm MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bd_src(a,b)!=bcir_bool_mod(a,b)){printf("bd MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(bc_src(a,b)!=bcir_bool_compound(a,b)){printf("bc MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(ba_src(a,b)!=bcir_bool_array(a,b)){printf("ba MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/bn_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/bn_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/bn_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/bn_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/bn_h" \
  || { echo "  FAIL: boolnorm harness build"; exit 1; }
bnr="$("${tmp}/bn_h")"
[ "${bnr}" = "MATCH" ] \
  && echo "  PASS boolnorm: store into _Bool normalizes to 0/1 == Clang" \
  || { echo "  FAIL: boolnorm behaviour (${bnr})"; exit 1; }
