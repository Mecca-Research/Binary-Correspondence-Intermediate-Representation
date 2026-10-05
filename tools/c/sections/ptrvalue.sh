#!/usr/bin/env bash
# ptrvalue: pointer values -- return p + i (bcir-cc): emit == Clang (#ptrvalue)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-ptrvalue`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> ptrvalue.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> ptrvalue.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Pointer values returned by value (#ptrvalue): pointer arithmetic `p + i` as an rvalue yields a pointer
# that is returned. The twin typed that temp as uint32_t -- an invalid pointer-from-integer return and an
# 8-byte-pointer truncation on a 64-bit target. Now both rails carry the pointee type (`T *t = p + i`).
# Each function returns a pointer into a shared buffer; compare the returned offsets against Clang.
"${BCIR_CC}" --emit-c "${C}/cfront_ptrvalue.c" > "${tmp}/pv_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bpv_advance\b/pv_adv_src/' -e 's/\bpv_commute\b/pv_com_src/' -e 's/\bpv_back\b/pv_bk_src/' \
      -e 's/\bpv_uadvance\b/pv_uadv_src/' -e 's/\bpv_sadvance\b/pv_sadv_src/' -e 's/\bpv_chain\b/pv_ch_src/' \
      "${C}/cfront_ptrvalue.c"
  cat "${tmp}/pv_emit.c"
  cat <<'DRV'
int main(void){
  static int ib[512]; static unsigned ub[512]; static short sb[512];
  for(unsigned n=0;n<190u;n++){
    if(pv_adv_src(ib,n)  !=bcir_pv_advance(ib,n))  {printf("adv MISMATCH n=%u\n",n);return 1;}
    if(pv_com_src(ib,n)  !=bcir_pv_commute(ib,n))  {printf("com MISMATCH n=%u\n",n);return 1;}
    if(pv_bk_src(ib,n)   !=bcir_pv_back(ib,n))     {printf("bk MISMATCH n=%u\n",n);return 1;}
    if(pv_uadv_src(ub,n) !=bcir_pv_uadvance(ub,n)) {printf("uadv MISMATCH n=%u\n",n);return 1;}
    if(pv_sadv_src(sb,n) !=bcir_pv_sadvance(sb,n)) {printf("sadv MISMATCH n=%u\n",n);return 1;}
    if(pv_ch_src(ib,n)   !=bcir_pv_chain(ib,n))    {printf("ch MISMATCH n=%u\n",n);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/pv_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/pv_harness.c" -o "${tmp}/pv_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/pv_harness.c" -o "${tmp}/pv_h" \
  || { echo "  FAIL: ptrvalue harness build"; exit 1; }
pvr="$("${tmp}/pv_h")"
[ "${pvr}" = "MATCH" ] \
  && echo "  PASS ptrvalue: returned p + i pointer == Clang (int/unsigned/short pointees)" \
  || { echo "  FAIL: ptrvalue behaviour (${pvr})"; exit 1; }
