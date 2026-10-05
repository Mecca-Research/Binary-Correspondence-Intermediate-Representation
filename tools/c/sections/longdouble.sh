#!/usr/bin/env bash
# longdouble: long double -- 80/128-bit extended float, +l libm variants (#longdouble)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-longdouble` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> longdouble.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> longdouble.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# long double (#longdouble): the extended floating type -- the twin emits real `long double` C (closing a
# parse-gap vs the oracle). Differential over arithmetic, an `L` constant, double/int conversions, a
# `long double *`, the +l libm variants (sqrtl/fabsl), and += accumulation; a wrong width / a dropped
# `long double` would diverge from Clang's 80-bit result.
"${BCIR_CC}" --emit-c "${C}/cfront_longdouble.c" > "${tmp}/ld_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <math.h>'
  sed -e 's/\bld_arith\b/ld_arith_s/g' -e 's/\bld_promote\b/ld_promote_s/g' -e 's/\bld_to_int\b/ld_to_int_s/g' \
      -e 's/\bld_narrow\b/ld_narrow_s/g' -e 's/\bld_libm\b/ld_libm_s/g' -e 's/\bld_ptr\b/ld_ptr_s/g' \
      -e 's/\bld_acc\b/ld_acc_s/g' \
      "${C}/cfront_longdouble.c"
  cat "${tmp}/ld_emit.c"
  cat <<'DRV'
int main(void){
  for(int i=-300;i<300;i++){ long double a=(long double)i*0.3L, b=(long double)(i+5)*0.13L;
    if(ld_arith_s(a,b)!=bcir_ld_arith(a,b)){printf("arith@%d\n",i);return 1;}
    if(ld_promote_s((double)i*0.5,i)!=bcir_ld_promote((double)i*0.5,i)){printf("promote@%d\n",i);return 1;}
    if(ld_to_int_s(a,b)!=bcir_ld_to_int(a,b)){printf("toint@%d\n",i);return 1;}
    if(ld_narrow_s(a)!=bcir_ld_narrow(a)){printf("narrow@%d\n",i);return 1;}
    if(i>=0 && ld_libm_s(a)!=bcir_ld_libm(a)){printf("libm@%d\n",i);return 1;}
    if(ld_ptr_s(a)!=bcir_ld_ptr(a)){printf("ptr@%d\n",i);return 1;}
    if(ld_acc_s(i%17,a)!=bcir_ld_acc(i%17,a)){printf("acc@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ld_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ld_harness.c" -o "${tmp}/ld_h" -lm 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ld_harness.c" -o "${tmp}/ld_h" -lm \
  || { echo "  FAIL: longdouble harness build"; exit 1; }
ldr="$("${tmp}/ld_h")"
[ "${ldr}" = "MATCH" ] \
  && echo "  PASS longdouble: arith / L-const / conversions / long double * / sqrtl / += == Clang" \
  || { echo "  FAIL: longdouble behaviour (${ldr})"; exit 1; }
