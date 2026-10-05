#!/usr/bin/env bash
# longunary: unary on a wide operand (bcir-cc): emit == Clang (#longunary)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-longunary`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> longunary.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> longunary.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Unary `-` / `~` on a wide (long / long long) operand (#longunary): the result was forced to a 4-byte
# uint32, so negating a `long` truncated the sign (a -1 long became 4294967295), breaking `x < 0` / abs.
# Now both rails keep the promoted operand type. Differential == Clang.
"${BCIR_CC}" --emit-c "${C}/cfront_longunary.c" > "${tmp}/lu_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\blong_abs\b/la_src/' -e 's/\blong_complement\b/lc_src/' "${C}/cfront_longunary.c"
  cat "${tmp}/lu_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<8000u;a++)for(unsigned b=0;b<40u;b++){
    if(la_src(a,b)!=bcir_long_abs(a,b)){printf("la MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(lc_src(a,b)!=bcir_long_complement(a,b)){printf("lc MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/lu_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/lu_harness.c" -o "${tmp}/lu_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/lu_harness.c" -o "${tmp}/lu_h" \
  || { echo "  FAIL: longunary harness build"; exit 1; }
lur="$("${tmp}/lu_h")"
[ "${lur}" = "MATCH" ] \
  && echo "  PASS longunary: -/~ on long/long long == Clang" \
  || { echo "  FAIL: longunary behaviour (${lur})"; exit 1; }
