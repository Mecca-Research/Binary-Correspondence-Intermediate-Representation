#!/usr/bin/env bash
# signedcmp: signed comparison vs literal (bcir-cc): emit == Clang (#signedcmp)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-signedcmp` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> signedcmp.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> signedcmp.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Signed comparison against an integer literal (#signedcmp): `x < 0` / `x >= 0` for a signed `x` (the abs
# / clamp / signum idiom). The twin hardcoded integer constants as uint32_t, so the comparison promoted
# to unsigned (a silent miscompile); now each constant carries its own type. Differential == Clang.
"${BCIR_CC}" --emit-c "${C}/cfront_signedcmp.c" > "${tmp}/sc_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\biabs\b/iabs_src/' -e 's/\bclamp_lo\b/clamp_src/' -e 's/\bsignum\b/signum_src/' "${C}/cfront_signedcmp.c"
  cat "${tmp}/sc_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<8000u;a++)for(unsigned b=0;b<40u;b++){
    if(iabs_src(a,b)!=bcir_iabs(a,b)){printf("iabs MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(clamp_src(a,b)!=bcir_clamp_lo(a,b)){printf("clamp MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(signum_src(a,b)!=bcir_signum(a,b)){printf("signum MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sc_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sc_harness.c" -o "${tmp}/sc_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sc_harness.c" -o "${tmp}/sc_h" \
  || { echo "  FAIL: signedcmp harness build"; exit 1; }
scr="$("${tmp}/sc_h")"
[ "${scr}" = "MATCH" ] \
  && echo "  PASS signedcmp: signed x < 0 / x >= 0 == Clang" \
  || { echo "  FAIL: signedcmp behaviour (${scr})"; exit 1; }
