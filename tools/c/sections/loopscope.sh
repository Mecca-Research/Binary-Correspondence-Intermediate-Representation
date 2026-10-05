#!/usr/bin/env bash
# loopscope: for-loop variable scope (bcir-cc): emit == Clang (#loopscope)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-loopscope`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> loopscope.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> loopscope.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# For-loop variable scope (#loopscope): a `for(unsigned i = ...)` scopes `i` to the loop, so a post-loop
# read of `i` resolves to a same-named param / outer -- the for-init must not leak into the enclosing
# block. Both rails save/restore the name env around the loop; the differential pins the post-loop value.
"${BCIR_CC}" --emit-c "${C}/cfront_loopscope.c" > "${tmp}/lsc_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bparam_shadow\b/param_shadow_src/' -e 's/\bouter_shadow\b/outer_shadow_src/' "${C}/cfront_loopscope.c"
  cat "${tmp}/lsc_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned x=0;x<20000u;x++){
    if(param_shadow_src(x)!=bcir_param_shadow(x)){printf("param MISMATCH x=%u\n",x);return 1;}
    if(outer_shadow_src(x)!=bcir_outer_shadow(x)){printf("outer MISMATCH x=%u\n",x);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/lsc_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/lsc_harness.c" -o "${tmp}/lsc_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/lsc_harness.c" -o "${tmp}/lsc_h" \
  || { echo "  FAIL: loopscope harness build"; exit 1; }
lscr="$("${tmp}/lsc_h")"
[ "${lscr}" = "MATCH" ] \
  && echo "  PASS loopscope: post-loop read resolves to the shadowed param/outer == Clang" \
  || { echo "  FAIL: loopscope behaviour (${lscr})"; exit 1; }
