#!/usr/bin/env bash
# multidecl: multi-declarator locals (bcir-cc): emit == Clang (#multidecl)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-multidecl`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> multidecl.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> multidecl.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Multi-declarator local declarations (#multidecl): `T a = x, b, c = z;` -- the canonical two-variable
# loop init + grouped temporaries. Each declarator lowers to its own storage + copy (identical to
# separate decls), so the twin's --emit-c is Clang-behaviour-equivalent over both fixture functions.
"${BCIR_CC}" --emit-c "${C}/cfront_multidecl.c" > "${tmp}/md_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bblend\b/blend_src/' -e 's/\bwindowed\b/windowed_src/' "${C}/cfront_multidecl.c"
  cat "${tmp}/md_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned n=0;n<6000u;n++){
    if(blend_src(n)!=bcir_blend(n)){printf("blend MISMATCH n=%u\n",n);return 1;}
    if(windowed_src(n)!=bcir_windowed(n)){printf("windowed MISMATCH n=%u\n",n);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/md_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/md_harness.c" -o "${tmp}/md_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/md_harness.c" -o "${tmp}/md_h" \
  || { echo "  FAIL: multidecl harness build"; exit 1; }
mdr="$("${tmp}/md_h")"
[ "${mdr}" = "MATCH" ] \
  && echo "  PASS multidecl: T a=x, b, c=z + two-variable for-init == Clang" \
  || { echo "  FAIL: multidecl behaviour (${mdr})"; exit 1; }
