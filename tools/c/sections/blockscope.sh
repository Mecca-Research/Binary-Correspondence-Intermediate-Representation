#!/usr/bin/env bash
# blockscope: bare-block variable scope (bcir-cc): emit == Clang (#blockscope)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-blockscope` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It compiles what bcir-cc emits with CC, which the caller
# names: BCIR Make passes the gate's, the CTest entry the configured C compiler, and the
# section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> blockscope.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> blockscope.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Bare-block variable scope (#blockscope): a `{ unsigned x = ...; }` block scopes `x`, so a post-block
# read resolves to a same-named outer -- the block must not leak. Both rails save/restore the name env
# around every `{ ... }` (the general case of #loopscope; the oracle also lowers a bare block inline now,
# via a Block node, so its claim count matches the twin -- no spurious if(1) wrapper).
"${BCIR_CC}" --emit-c "${C}/cfront_blockscope.c" > "${tmp}/bs_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bnested_shadow\b/ns_src/' -e 's/\bblock_then_use\b/bt_src/' "${C}/cfront_blockscope.c"
  cat "${tmp}/bs_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<5000u;a++){
    if(ns_src(a)!=bcir_nested_shadow(a)){printf("ns MISMATCH a=%u\n",a);return 1;}
    for(unsigned b=0;b<20u;b++) if(bt_src(a%500u,b)!=bcir_block_then_use(a%500u,b)){printf("bt MISMATCH\n");return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/bs_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/bs_harness.c" -o "${tmp}/bs_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/bs_harness.c" -o "${tmp}/bs_h" \
  || { echo "  FAIL: blockscope harness build"; exit 1; }
bsr="$("${tmp}/bs_h")"
[ "${bsr}" = "MATCH" ] \
  && echo "  PASS blockscope: post-block read resolves to the shadowed outer == Clang" \
  || { echo "  FAIL: blockscope behaviour (${bsr})"; exit 1; }
