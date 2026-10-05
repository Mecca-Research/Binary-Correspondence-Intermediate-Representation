#!/usr/bin/env bash
# localmd: multi-dimensional local arrays (bcir-cc): emit == Clang (#localmd)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-localmd` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> localmd.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> localmd.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Multi-dimensional local arrays (#localmd): `T m[A][B]` up to 3 dims -- a flat resource of A*B elements
# with the per-dim flatten shape, so `m[i][j]` -> `m[i*B + j]` (declared `m[A*B]`). The twin emits the
# flattened `m[lin]`; a wrong stride or size would diverge. Differential checks 2-D + 3-D fill/read.
"${BCIR_CC}" --emit-c "${C}/cfront_localmd.c" > "${tmp}/lmd_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bgrid_diag\b/grid_src/' -e 's/\bcube_sum\b/cube_src/' "${C}/cfront_localmd.c"
  cat "${tmp}/lmd_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned i=0;i<8000u;i++)for(unsigned a=0;a<30u;a++){
    if(grid_src(i,a)!=bcir_grid_diag(i,a)){printf("grid MISMATCH i=%u a=%u\n",i,a);return 1;}
    if(cube_src(i,a)!=bcir_cube_sum(i,a)){printf("cube MISMATCH i=%u a=%u\n",i,a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/lmd_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/lmd_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/lmd_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/lmd_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/lmd_h" \
  || { echo "  FAIL: localmd harness build"; exit 1; }
lmdr="$("${tmp}/lmd_h")"
[ "${lmdr}" = "MATCH" ] \
  && echo "  PASS localmd: 2-D + 3-D local array fill/read == Clang" \
  || { echo "  FAIL: localmd behaviour (${lmdr})"; exit 1; }
