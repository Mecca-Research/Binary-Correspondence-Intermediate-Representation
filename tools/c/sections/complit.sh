#!/usr/bin/env bash
# complit: compound literals -- (type){init} by value / scalar / &(literal) / .field (#complit)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-complit` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> complit.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> complit.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Compound literals (#complit): `(type){init}` materialized as a nameless local, in rvalue position
# (by-value struct arg / scalar value / member init), under `&` (pointer to the temporary), and with
# direct postfix on the literal (`(struct P){...}.field`).
"${BCIR_CC}" --emit-c "${C}/cfront_complit.c" > "${tmp}/cl_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bcl_byval\b/cl_byval_s/' -e 's/\bcl_designated\b/cl_designated_s/' \
      -e 's/\bcl_partial\b/cl_partial_s/' -e 's/\bcl_scalar\b/cl_scalar_s/' \
      -e 's/\bcl_addr_scalar\b/cl_addr_scalar_s/' -e 's/\bcl_addr_struct\b/cl_addr_struct_s/' \
      -e 's/\bcl_nested\b/cl_nested_s/' \
      -e 's/\bcl_dot\b/cl_dot_s/' -e 's/\bcl_dot_desig\b/cl_dot_desig_s/' \
      -e 's/\bcl_dot_part\b/cl_dot_part_s/' -e 's/\bcl_dot_wide\b/cl_dot_wide_s/' \
      "${C}/cfront_complit.c"
  cat "${tmp}/cl_emit.c"
  cat <<'DRV'
int main(void){
  for(int a=-40;a<40;a++) for(int b=-7;b<7;b++){
    if(cl_byval_s(a,b)!=bcir_cl_byval(a,b)){printf("byval@%d,%d\n",a,b);return 1;}
    if(cl_designated_s(a,b)!=bcir_cl_designated(a,b)){printf("desig@%d,%d\n",a,b);return 1;}
    if(cl_partial_s(a)!=bcir_cl_partial(a)){printf("partial@%d\n",a);return 1;}
    if(cl_scalar_s(a)!=bcir_cl_scalar(a)){printf("scalar@%d\n",a);return 1;}
    if(cl_addr_scalar_s(a)!=bcir_cl_addr_scalar(a)){printf("as@%d\n",a);return 1;}
    if(cl_addr_struct_s(a,b)!=bcir_cl_addr_struct(a,b)){printf("ast@%d,%d\n",a,b);return 1;}
    if(cl_nested_s(a)!=bcir_cl_nested(a)){printf("nested@%d\n",a);return 1;}
    if(cl_dot_s(a,b)!=bcir_cl_dot(a,b)){printf("dot@%d,%d\n",a,b);return 1;}
    if(cl_dot_desig_s(a,b)!=bcir_cl_dot_desig(a,b)){printf("dotdes@%d,%d\n",a,b);return 1;}
    if(cl_dot_part_s(a)!=bcir_cl_dot_part(a)){printf("dotpart@%d\n",a);return 1;}
    if(cl_dot_wide_s(a)!=bcir_cl_dot_wide(a)){printf("dotwide@%d\n",a);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/cl_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/cl_harness.c" -o "${tmp}/cl_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/cl_harness.c" -o "${tmp}/cl_h" \
  || { echo "  FAIL: complit harness build"; exit 1; }
clr="$("${tmp}/cl_h")"
[ "${clr}" = "MATCH" ] \
  && echo "  PASS complit: by value / designators / &(int){v} / &(struct){...} / (struct){...}.field == Clang" \
  || { echo "  FAIL: complit behaviour (${clr})"; exit 1; }
