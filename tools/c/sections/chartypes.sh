#!/usr/bin/env bash
# chartypes: faithful char types -- char / signed char / unsigned char (#chartypes)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-chartypes`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> chartypes.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> chartypes.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Faithful char types (#chartypes): the three distinct one-byte char types emit faithfully -- plain
# `char` -> `char` (impl-defined sign), `signed char` -> always signed, `unsigned char` -> always
# unsigned. Built under BOTH -fsigned-char AND -funsigned-char so plain char's platform sign is
# exercised both ways (the old emit collapsed `signed char` -> `char` / plain `char` -> int8_t, wrong
# on one of the two).
"${BCIR_CC}" --emit-c "${C}/cfront_chartypes.c" > "${tmp}/cht_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bct_plain_deref\b/ct_plain_deref_s/' -e 's/\bct_signed_deref\b/ct_signed_deref_s/' \
      -e 's/\bct_unsigned_deref\b/ct_unsigned_deref_s/' -e 's/\bct_plain_cmp\b/ct_plain_cmp_s/' \
      -e 's/\bct_signed_cmp\b/ct_signed_cmp_s/' -e 's/\bct_plain_div\b/ct_plain_div_s/' \
      -e 's/\bct_signed_div\b/ct_signed_div_s/' -e 's/\bct_unsigned_div\b/ct_unsigned_div_s/' \
      -e 's/\bct_roundtrip\b/ct_roundtrip_s/' -e 's/\bct_plain_widen\b/ct_plain_widen_s/' \
      "${C}/cfront_chartypes.c"
  cat "${tmp}/cht_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-200;i<200;i++){
    char pc=(char)(i*7-3); signed char sc=(signed char)(i*5+1); unsigned char uc=(unsigned char)(i*3+2);
    if(ct_plain_deref_s(&pc)!=bcir_ct_plain_deref(&pc)){printf("pd@%d\n",i);return 1;}
    if(ct_signed_deref_s(&sc)!=bcir_ct_signed_deref(&sc)){printf("sd@%d\n",i);return 1;}
    if(ct_unsigned_deref_s(&uc)!=bcir_ct_unsigned_deref(&uc)){printf("ud@%d\n",i);return 1;}
    if(ct_plain_cmp_s(pc)!=bcir_ct_plain_cmp(pc)){printf("pc@%d\n",i);return 1;}
    if(ct_signed_cmp_s(sc)!=bcir_ct_signed_cmp(sc)){printf("sc@%d\n",i);return 1;}
    if(ct_plain_div_s(&pc)!=bcir_ct_plain_div(&pc)){printf("pdv@%d\n",i);return 1;}
    if(ct_signed_div_s(&sc)!=bcir_ct_signed_div(&sc)){printf("sdv@%d\n",i);return 1;}
    if(ct_unsigned_div_s(&uc)!=bcir_ct_unsigned_div(&uc)){printf("udv@%d\n",i);return 1;}
    if(ct_roundtrip_s(&pc)!=bcir_ct_roundtrip(&pc)){printf("rt@%d\n",i);return 1;}
    if(ct_plain_widen_s(&pc)!=bcir_ct_plain_widen(&pc)){printf("pw@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/cht_harness.c"
for cm in -fsigned-char -funsigned-char; do
  "${CC}" -std=c23 -O2 "${cm}" "${tmp}/cht_harness.c" -o "${tmp}/cht_h" 2>/dev/null \
    || "${CC}" -std=c2x -O2 "${cm}" "${tmp}/cht_harness.c" -o "${tmp}/cht_h" \
    || { echo "  FAIL: chartypes harness build (${cm})"; exit 1; }
  chtr="$("${tmp}/cht_h")"
  [ "${chtr}" = "MATCH" ] \
    && echo "  PASS chartypes (${cm}): char / signed char / unsigned char == Clang" \
    || { echo "  FAIL: chartypes behaviour (${cm}: ${chtr})"; exit 1; }
done
