#!/usr/bin/env bash
# typeof: typeof -- typeof(type-name) / typeof(variable) / typeof(expr) (#typeof)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-typeof` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> typeof.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> typeof.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# typeof (#typeof): `typeof(type-name)` / `typeof(variable)` / `typeof(expression)` resolves to the
# operand's type -- each case is built so the WRONG type (int vs long, signed vs unsigned, a short
# truncation) would diverge. The expression operand is type-inferred (oracle) / speculatively lowered
# then rolled back (twin), as it is unevaluated.
"${BCIR_CC}" --emit-c "${C}/cfront_typeof.c" > "${tmp}/to_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bto_width\b/to_width_s/' -e 's/\bto_sign\b/to_sign_s/' -e 's/\bto_typename\b/to_typename_s/' \
      -e 's/\bto_ptr\b/to_ptr_s/' -e 's/\bto_struct\b/to_struct_s/' -e 's/\bto_unqual\b/to_unqual_s/' \
      -e 's/\bto_ebinop\b/to_ebinop_s/' -e 's/\bto_ebinsign\b/to_ebinsign_s/' -e 's/\bto_ecast\b/to_ecast_s/' \
      -e 's/\bto_ederef\b/to_ederef_s/' -e 's/\bto_emember\b/to_emember_s/' -e 's/\bto_eindex\b/to_eindex_s/' \
      "${C}/cfront_typeof.c"
  cat "${tmp}/to_emit.c"
  cat <<'DRV'
int main(void){
  for(long a=-50;a<50;a++){
    if(to_width_s(a)!=bcir_to_width(a)){printf("width@%ld\n",a);return 1;}
    if(to_sign_s((unsigned)a)!=bcir_to_sign((unsigned)a)){printf("sign@%ld\n",a);return 1;}
    if(to_typename_s((int)a)!=bcir_to_typename((int)a)){printf("tn@%ld\n",a);return 1;}
    if(to_ptr_s((int)a)!=bcir_to_ptr((int)a)){printf("ptr@%ld\n",a);return 1;}
    if(to_struct_s((int)a)!=bcir_to_struct((int)a)){printf("struct@%ld\n",a);return 1;}
    if(to_unqual_s(a)!=bcir_to_unqual(a)){printf("unq@%ld\n",a);return 1;}
    if(to_ebinop_s(a)!=bcir_to_ebinop(a)){printf("ebinop@%ld\n",a);return 1;}
    if(to_ebinsign_s((unsigned)a)!=bcir_to_ebinsign((unsigned)a)){printf("ebinsign@%ld\n",a);return 1;}
    if(to_ecast_s((int)a)!=bcir_to_ecast((int)a)){printf("ecast@%ld\n",a);return 1;}
    if(to_ederef_s(a)!=bcir_to_ederef(a)){printf("ederef@%ld\n",a);return 1;}
    if(to_emember_s((int)a)!=bcir_to_emember((int)a)){printf("emember@%ld\n",a);return 1;}
    if(to_eindex_s(a)!=bcir_to_eindex(a)){printf("eindex@%ld\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/to_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/to_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/to_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/to_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/to_h" \
  || { echo "  FAIL: typeof harness build"; exit 1; }
tor="$("${tmp}/to_h")"
[ "${tor}" = "MATCH" ] \
  && echo "  PASS typeof: typeof type-name / variable / expr (a+b, (short)x, *p, s.f, arr[i]) == Clang" \
  || { echo "  FAIL: typeof behaviour (${tor})"; exit 1; }
