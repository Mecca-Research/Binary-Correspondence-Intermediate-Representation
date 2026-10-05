#!/usr/bin/env bash
# atomiclocal: _Atomic local objects -- _Atomic int / _Atomic(int) (#atomiclocal)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-atomiclocal` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It compiles what bcir-cc emits with CC, which the caller
# names: BCIR Make passes the gate's, the CTest entry the configured C compiler, and the
# section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> atomiclocal.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> atomiclocal.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# _Atomic local objects (#atomiclocal): `_Atomic int a;` / `_Atomic(int) a;` / const _Atomic / a
# pointer-to-atomic, as function locals (previously only the global form parsed). An unshared atomic local
# equals its plain type single-threaded; the differential drives compound-assign / shift / deref.
"${BCIR_CC}" --emit-c "${C}/cfront_atomiclocal.c" > "${tmp}/at_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <stdatomic.h>'
  sed -e 's/\ba_qual\b/a_qual_s/g' -e 's/\ba_paren\b/a_paren_s/g' -e 's/\ba_long\b/a_long_s/g' \
      -e 's/\ba_const\b/a_const_s/g' -e 's/\ba_ptr\b/a_ptr_s/g' \
      "${C}/cfront_atomiclocal.c"
  cat "${tmp}/at_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-300;x<300;x++){ long b=(long)x*100000;
    if(a_qual_s(x)!=bcir_a_qual(x)){puts("qual");return 1;}
    if(a_paren_s(x)!=bcir_a_paren(x)){puts("paren");return 1;}
    if(a_long_s(b)!=bcir_a_long(b)){puts("long");return 1;}
    if(a_const_s(x)!=bcir_a_const(x)){puts("const");return 1;}
    if(a_ptr_s(x)!=bcir_a_ptr(x)){puts("ptr");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/at_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/at_harness.c" -o "${tmp}/at_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/at_harness.c" -o "${tmp}/at_h" \
  || { echo "  FAIL: atomiclocal harness build"; exit 1; }
atr="$("${tmp}/at_h")"
[ "${atr}" = "MATCH" ] \
  && echo "  PASS atomiclocal: _Atomic int / _Atomic(int) / const _Atomic / _Atomic(int)* == Clang" \
  || { echo "  FAIL: atomiclocal behaviour (${atr})"; exit 1; }
