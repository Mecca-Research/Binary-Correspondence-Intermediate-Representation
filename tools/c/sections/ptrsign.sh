#!/usr/bin/env bash
# ptrsign: pointer-element signedness -- signed vs unsigned pointee load/store (#ptrsign)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-ptrsign`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> ptrsign.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> ptrsign.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Pointer-element signedness (#ptrsign): a load / store / subscript through a pointer carries the
# pointee's SIGNEDNESS, not just its width -- a signed sub-int pointee sign-extends, an unsigned one
# zero-extends, and the loaded value drives signed-vs-unsigned divide / shift / comparison / UAC. A
# bespoke harness sweeps negative + boundary pointee values (a width-only model would diverge here).
"${BCIR_CC}" --emit-c "${C}/cfront_ptrsign.c" > "${tmp}/psn_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bps_s8\b/ps_s8_s/' -e 's/\bps_u8\b/ps_u8_s/' -e 's/\bps_s16\b/ps_s16_s/' \
      -e 's/\bps_u16\b/ps_u16_s/' -e 's/\bps_s8_divrem\b/ps_s8_divrem_s/' -e 's/\bps_u8_div\b/ps_u8_div_s/' \
      -e 's/\bps_s8_shr\b/ps_s8_shr_s/' -e 's/\bps_u8_shr\b/ps_u8_shr_s/' -e 's/\bps_s8_cmp\b/ps_s8_cmp_s/' \
      -e 's/\bps_s64_div\b/ps_s64_div_s/' -e 's/\bps_u64_div\b/ps_u64_div_s/' -e 's/\bps_arith\b/ps_arith_s/' \
      -e 's/\bps_uac\b/ps_uac_s/' -e 's/\bps_field\b/ps_field_s/' -e 's/\bps_w8\b/ps_w8_s/' \
      "${C}/cfront_ptrsign.c"
  cat "${tmp}/psn_emit.c"
  cat <<'DRV'
int main(void){
  for(long i=-400;i<400;i++){
    int8_t sb=(int8_t)(i*7-3); uint8_t ub=(uint8_t)(i*5+1);
    int16_t ha[4]={(int16_t)(i*3),(int16_t)(-i),(int16_t)(i+9),(int16_t)(i*7)};
    uint16_t ua[4]={(uint16_t)(i*3),(uint16_t)(i),(uint16_t)(i+9),(uint16_t)(i*7)};
    long lv=i*1000000007L-7; unsigned long ul=(unsigned long)(i*2654435761UL+9);
    int8_t a[4]={(int8_t)i,(int8_t)(i-1),(int8_t)(i+2),(int8_t)(-i)};
    if(ps_s8_s(&sb)!=bcir_ps_s8(&sb)){printf("s8@%ld\n",i);return 1;}
    if(ps_u8_s(&ub)!=bcir_ps_u8(&ub)){printf("u8@%ld\n",i);return 1;}
    if(ps_s16_s(ha,2)!=bcir_ps_s16(ha,2)){printf("s16@%ld\n",i);return 1;}
    if(ps_u16_s(ua,2)!=bcir_ps_u16(ua,2)){printf("u16@%ld\n",i);return 1;}
    if(ps_s8_divrem_s(&sb)!=bcir_ps_s8_divrem(&sb)){printf("divrem@%ld\n",i);return 1;}
    if(ps_u8_div_s(&ub)!=bcir_ps_u8_div(&ub)){printf("udiv@%ld\n",i);return 1;}
    if(ps_s8_shr_s(&sb)!=bcir_ps_s8_shr(&sb)){printf("sshr@%ld\n",i);return 1;}
    if(ps_u8_shr_s(&ub)!=bcir_ps_u8_shr(&ub)){printf("ushr@%ld\n",i);return 1;}
    if(ps_s8_cmp_s(&sb)!=bcir_ps_s8_cmp(&sb)){printf("cmp@%ld\n",i);return 1;}
    if(ps_s64_div_s(&lv)!=bcir_ps_s64_div(&lv)){printf("s64@%ld\n",i);return 1;}
    if(ps_u64_div_s(&ul)!=bcir_ps_u64_div(&ul)){printf("u64@%ld\n",i);return 1;}
    if(ps_arith_s(a,2)!=bcir_ps_arith(a,2)){printf("arith@%ld\n",i);return 1;}
    if(ps_uac_s(&ub,(int)i)!=bcir_ps_uac(&ub,(int)i)){printf("uac@%ld\n",i);return 1;}
    struct Buf bb={&sb,&ub}; if(ps_field_s(&bb)!=bcir_ps_field(&bb)){printf("field@%ld\n",i);return 1;}
    signed char w1=0,w2=0; ps_w8_s(&w1,(int)i); bcir_ps_w8(&w2,(int)i);
    if(w1!=w2){printf("w8@%ld\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/psn_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/psn_harness.c" -o "${tmp}/psn_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/psn_harness.c" -o "${tmp}/psn_h" \
  || { echo "  FAIL: ptrsign harness build"; exit 1; }
psnr="$("${tmp}/psn_h")"
[ "${psnr}" = "MATCH" ] \
  && echo "  PASS ptrsign: signed/unsigned pointee load/store/divide/shift/cmp == Clang" \
  || { echo "  FAIL: ptrsign behaviour (${psnr})"; exit 1; }
