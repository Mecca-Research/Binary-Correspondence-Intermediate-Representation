#!/usr/bin/env bash
# signedty: the 'signed' type specifier (bcir-cc): emit == Clang (#signedty)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-signedty`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> signedty.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> signedty.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# The `signed` type specifier (#signedty): `signed char` / `signed int` / `signed long` decl + cast. The
# twin recognized `unsigned` but not `signed` in its decl/cast type-start detection, so it rejected these
# (-> fallback) while the oracle accepted -- a rail disagreement, now fixed. Differential == Clang.
"${BCIR_CC}" --emit-c "${C}/cfront_signed.c" > "${tmp}/sg_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bsigned_roundtrip\b/sr_src/' -e 's/\bsigned_scale\b/ss_src/' "${C}/cfront_signed.c"
  cat "${tmp}/sg_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<8000u;a++)for(unsigned b=0;b<24u;b++){
    if(sr_src(a)!=bcir_signed_roundtrip(a)){printf("sr MISMATCH a=%u\n",a);return 1;}
    if(ss_src(a,b)!=bcir_signed_scale(a,b)){printf("ss MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sg_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sg_harness.c" -o "${tmp}/sg_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sg_harness.c" -o "${tmp}/sg_h" \
  || { echo "  FAIL: signedty harness build"; exit 1; }
sgr="$("${tmp}/sg_h")"
[ "${sgr}" = "MATCH" ] \
  && echo "  PASS signedty: signed char/int/long decl + cast == Clang" \
  || { echo "  FAIL: signedty behaviour (${sgr})"; exit 1; }
