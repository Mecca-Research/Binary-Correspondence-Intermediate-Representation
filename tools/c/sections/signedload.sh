#!/usr/bin/env bash
# signedload: signed sub-int storage read sign-extends (bcir-cc): emit == Clang (#signedload)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-signedload` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> signedload.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> signedload.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Signed sub-int storage read sign-extension (#signedload): reading a `signed char`/`short` from a struct
# member, a member array, or a local array must sign-extend to int. The twin loaded a member via a
# zero-extending memcpy into a uint32 temp, and typed the array-element temp unsigned -- so a signed read
# came back as a large positive and its `< 0` test was always false. Now the twin types each load temp with
# the element's (width, signedness) and memcpy's the exact width. Unsigned elements unchanged.
"${BCIR_CC}" --emit-c "${C}/cfront_signedload.c" > "${tmp}/sl_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bm_signtest\b/sl_ms_src/' -e 's/\bm_value\b/sl_mv_src/' -e 's/\bm_arr_signtest\b/sl_ma_src/' \
      -e 's/\bloc_signtest\b/sl_ls_src/' -e 's/\bloc_value\b/sl_lv_src/' -e 's/\buc_control\b/sl_uc_src/' \
      "${C}/cfront_signedload.c"
  cat "${tmp}/sl_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<70000u;a+=3u)for(unsigned b=0;b<300u;b++){
    if(sl_ms_src(a,b)!=bcir_m_signtest(a,b)){printf("ms MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_mv_src(a,b)!=bcir_m_value(a,b)){printf("mv MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_ma_src(a,b)!=bcir_m_arr_signtest(a,b)){printf("ma MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_ls_src(a,b)!=bcir_loc_signtest(a,b)){printf("ls MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_lv_src(a,b)!=bcir_loc_value(a,b)){printf("lv MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sl_uc_src(a,b)!=bcir_uc_control(a,b)){printf("uc MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sl_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/sl_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/sl_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/sl_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/sl_h" \
  || { echo "  FAIL: signedload harness build"; exit 1; }
slr="$("${tmp}/sl_h")"
[ "${slr}" = "MATCH" ] \
  && echo "  PASS signedload: signed sub-int member/array read sign-extends == Clang" \
  || { echo "  FAIL: signedload behaviour (${slr})"; exit 1; }
