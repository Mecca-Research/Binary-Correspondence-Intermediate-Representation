#!/usr/bin/env bash
# variadic: variadic functions -- va_list / va_start / va_arg / va_end / va_copy (#variadic)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-variadic`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> variadic.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> variadic.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Variadic functions (#variadic): `f(T last, ...)` with <stdarg.h> -- a va_list cursor (va_start/va_arg/
# va_end), va_copy, a va_list parameter (vprintf-style forwarding), and a same-unit variadic call passing
# args past the fixed params (default promotions ride the real call). va_start/va_arg/va_end/va_copy lower
# as opaque builtins emitted verbatim; va_arg(ap, T) carries type T. Differential drives the int / float /
# va_copy / forwarding paths; a wrong emit (a truncated va_arg load, a dropped `...`) would diverge.
"${BCIR_CC}" --emit-c "${C}/cfront_variadic.c" > "${tmp}/va_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <stdarg.h>'
  sed -e 's/\bisum\b/isum_s/g' -e 's/\btwice\b/twice_s/g' -e 's/\bvsumv\b/vsumv_s/g' \
      -e 's/\bforward\b/forward_s/g' -e 's/\bnth\b/nth_s/g' -e 's/\bcaller\b/caller_s/g' \
      "${C}/cfront_variadic.c"
  cat "${tmp}/va_emit.c"
  cat <<'DRV'
int main(void){
  for(int a=-40;a<40;a++) for(int b=-40;b<40;b++){ int c=a-2*b;
    if(caller_s(a,b,c)!=bcir_caller(a,b,c)){printf("caller@%d,%d\n",a,b);return 1;}
    if(isum_s(4,a,b,c,a+b)!=bcir_isum(4,a,b,c,a+b)){printf("isum@%d,%d\n",a,b);return 1;}
    if(twice_s(3,a,b,c)!=bcir_twice(3,a,b,c)){printf("twice@%d,%d\n",a,b);return 1;}
    if(forward_s(3,a,b,c)!=bcir_forward(3,a,b,c)){printf("forward@%d,%d\n",a,b);return 1;}
    if(nth_s(2,(double)a,(double)b,(double)c)!=bcir_nth(2,(double)a,(double)b,(double)c)){printf("nth@%d,%d\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/va_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/va_harness.c" -o "${tmp}/va_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/va_harness.c" -o "${tmp}/va_h" \
  || { echo "  FAIL: variadic harness build"; exit 1; }
var="$("${tmp}/va_h")"
[ "${var}" = "MATCH" ] \
  && echo "  PASS variadic: va_list/va_start/va_arg/va_end/va_copy + va_list param + same-unit call == Clang" \
  || { echo "  FAIL: variadic behaviour (${var})"; exit 1; }
