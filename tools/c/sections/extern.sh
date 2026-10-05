#!/usr/bin/env bash
# extern: extern linkage (bcir-cc): extern global emit == Clang (#extern)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-extern` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> extern.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> extern.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# `extern` linkage (#extern): a symbol declared here but defined in another TU. `extern T g;` is
# referenced by name (no storage emitted), so the twin's --emit-c is Clang-behaviour-equivalent once
# linked against the definition (supplied by the driver below).
"${BCIR_CC}" --emit-c "${C}/cfront_extern.c" > "${tmp}/ex_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo 'unsigned cfg_base = 0u;'   # the DEFINITION (other TU)
  cat "${C}/cfront_extern.c" "${tmp}/ex_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned t=0;t<3000u;t++){
    unsigned x=t*7u+1u;
    cfg_base = t & 255u;
    if(scaled(x)!=bcir_scaled(x) || offs(x)!=bcir_offs(x)){printf("read MISMATCH t=%u\n",t);return 1;}
    cfg_base=t; bump(3u);      unsigned src=cfg_base;     /* source writes the extern */
    cfg_base=t; bcir_bump(3u); unsigned twn=cfg_base;     /* twin writes the extern */
    if(src!=twn){printf("write MISMATCH t=%u\n",t);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ex_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ex_harness.c" -o "${tmp}/ex_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ex_harness.c" -o "${tmp}/ex_h" \
  || { echo "  FAIL: extern harness build"; exit 1; }
exr="$("${tmp}/ex_h")"
[ "${exr}" = "MATCH" ] \
  && echo "  PASS extern: read + write a cross-TU global == Clang (linked to the definition)" \
  || { echo "  FAIL: extern behaviour (${exr})"; exit 1; }
