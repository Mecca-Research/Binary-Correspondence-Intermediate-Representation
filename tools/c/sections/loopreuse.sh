#!/usr/bin/env bash
# loopreuse: reused loop-counter names (bcir-cc): unique emit == Clang (#loopreuse)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-loopreuse`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> loopreuse.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> loopreuse.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Reused local names across scopes (#loopreuse): two/three separate `for` loops each declaring the same
# counter name (`i`, `k`) are distinct flattened locals; the emit must give each a unique C identifier
# (`i`, `i_2`, ...) -- declaring both at function scope was a C redefinition (the unit was is_clean yet
# its emit did not compile, on BOTH rails). Differential confirms the disjoint-scope reuse == Clang.
"${BCIR_CC}" --emit-c "${C}/cfront_loopreuse.c" > "${tmp}/lr_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\btwo_pass\b/two_pass_src/' -e 's/\btriple_pass\b/triple_pass_src/' "${C}/cfront_loopreuse.c"
  cat "${tmp}/lr_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned n=0;n<20000u;n++){
    if(two_pass_src(n)!=bcir_two_pass(n)){printf("two MISMATCH n=%u\n",n);return 1;}
    if(triple_pass_src(n)!=bcir_triple_pass(n)){printf("triple MISMATCH n=%u\n",n);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/lr_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/lr_harness.c" -o "${tmp}/lr_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/lr_harness.c" -o "${tmp}/lr_h" \
  || { echo "  FAIL: loopreuse harness build (emit did not compile)"; exit 1; }
lrr="$("${tmp}/lr_h")"
[ "${lrr}" = "MATCH" ] \
  && echo "  PASS loopreuse: two/three loops reusing a counter name == Clang" \
  || { echo "  FAIL: loopreuse behaviour (${lrr})"; exit 1; }
