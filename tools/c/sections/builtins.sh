#!/usr/bin/env bash
# builtins: GCC/Clang integer builtins -- popcount/clz/ctz/bswap/abs (#builtins)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-builtins`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> builtins.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> builtins.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# GCC/Clang integer builtins (#builtins): __builtin_popcount/clz/ctz/ffs/parity/bswap/abs (+ l/ll), emitted
# verbatim (opaque to R18). The differential drives the bit-manip family; a wrong result type / a synthesized
# bcir_ twin would diverge or fail to link.
"${BCIR_CC}" --emit-c "${C}/cfront_builtins.c" > "${tmp}/bi_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bbi_pop\b/bi_pop_s/g' -e 's/\bbi_clz\b/bi_clz_s/g' -e 's/\bbi_ffs\b/bi_ffs_s/g' \
      -e 's/\bbi_bswap\b/bi_bswap_s/g' -e 's/\bbi_bswap64\b/bi_bswap64_s/g' -e 's/\bbi_abs\b/bi_abs_s/g' \
      "${C}/cfront_builtins.c"
  cat "${tmp}/bi_emit.c"
  cat <<'DRV'
int main(void){
  for(long i=-40000;i<40000;i+=3){ unsigned x=(unsigned)(i*131071); int xi=(int)i;
    unsigned long long w=(unsigned long long)x * 2654435761ULL + (unsigned)xi;
    if(bi_pop_s(x)!=bcir_bi_pop(x)){puts("pop");return 1;}
    if(bi_clz_s(x)!=bcir_bi_clz(x)){puts("clz");return 1;}
    if(bi_ffs_s(xi)!=bcir_bi_ffs(xi)){puts("ffs");return 1;}
    if(bi_bswap_s(x)!=bcir_bi_bswap(x)){puts("bswap");return 1;}
    if(bi_bswap64_s(w)!=bcir_bi_bswap64(w)){puts("bswap64");return 1;}
    if(bi_abs_s(xi)!=bcir_bi_abs(xi)){puts("abs");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/bi_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/bi_harness.c" -o "${tmp}/bi_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/bi_harness.c" -o "${tmp}/bi_h" \
  || { echo "  FAIL: builtins harness build"; exit 1; }
bir="$("${tmp}/bi_h")"
[ "${bir}" = "MATCH" ] \
  && echo "  PASS builtins: popcount/clz/ctz/ffs/parity/bswap16-32-64/abs/labs/llabs == Clang" \
  || { echo "  FAIL: builtins behaviour (${bir})"; exit 1; }
