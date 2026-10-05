#!/usr/bin/env bash
# enumtype: enum used as a type (bcir-cc): emit == Clang (#enumtype)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-enumtype` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> enumtype.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> enumtype.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# `enum` as a type (#enumtype): `enum Tag` is a valid int-sized type specifier (decl / cast / parameter).
# The twin accepted it; the oracle's type parser recomputed the base from an empty keyword run after the
# enum branch already set it, raising "expected a type" -- rejecting every enum-as-type (a rail
# disagreement, oracle stricter than the twin and Clang). Fixed in the oracle; both rails lower enum -> int.
"${BCIR_CC}" --emit-c "${C}/cfront_enumtype.c" > "${tmp}/et_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\be_cast\b/et_c_src/' -e 's/\be_select\b/et_s_src/' -e 's/\be_param\b/et_p_src/' \
      -e 's/\bsign_mag\b/et_sm_src/' "${C}/cfront_enumtype.c"
  cat "${tmp}/et_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<3000u;a++)for(unsigned b=0;b<120u;b++){
    if(et_c_src(a,b)!=bcir_e_cast(a,b)){printf("c MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(et_s_src(a,b)!=bcir_e_select(a,b)){printf("s MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(et_p_src(a,b)!=bcir_e_param(a,b)){printf("p MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/et_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/et_harness.c" -o "${tmp}/et_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/et_harness.c" -o "${tmp}/et_h" \
  || { echo "  FAIL: enumtype harness build"; exit 1; }
etr="$("${tmp}/et_h")"
[ "${etr}" = "MATCH" ] \
  && echo "  PASS enumtype: enum decl/cast/param == Clang" \
  || { echo "  FAIL: enumtype behaviour (${etr})"; exit 1; }
