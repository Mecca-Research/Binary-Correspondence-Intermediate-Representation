#!/usr/bin/env bash
# compoundwide: wide/float compound assignment -- OP= / ++ / -- keeps width (#compoundwide)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-compoundwide` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> compoundwide.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> compoundwide.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Wide / floating compound assignment (#compoundwide): `OP=` / ++ / -- on a long/double lvalue keeps the
# operand width/float-ness (was truncated to a 4-byte uint32 at the compound-assign sites) -- across a
# local, a struct member, an array element and a pointer deref -- plus the float/wide-int variadic
# accumulation it unblocks. The driver spans values that overflow 32 bits, so a truncating result diverges.
"${BCIR_CC}" --emit-c "${C}/cfront_compoundwide.c" > "${tmp}/cw_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <stdarg.h>'
  sed -e 's/\bl_local\b/l_local_s/g' -e 's/\bd_local\b/d_local_s/g' -e 's/\bl_inc\b/l_inc_s/g' \
      -e 's/\bl_member\b/l_member_s/g' -e 's/\bl_array\b/l_array_s/g' -e 's/\bl_ptr\b/l_ptr_s/g' \
      -e 's/\bd_vararg\b/d_vararg_s/g' -e 's/\bl_vararg\b/l_vararg_s/g' -e 's/\bdriver\b/driver_s/g' \
      "${C}/cfront_compoundwide.c"
  cat "${tmp}/cw_emit.c"
  cat <<'DRV'
int main(void){
  long V[]={0,1,-1,1000000000L,-1000000000L,5000000000L,-5000000000L,99999999999L};
  int n=(int)(sizeof V/sizeof V[0]);
  for(int i=0;i<n;i++) for(int j=0;j<n;j++){ long x=V[i],y=V[j];
    if(driver_s(x,y)!=bcir_driver(x,y)){printf("driver@%ld,%ld\n",x,y);return 1;}
    if(l_member_s(x,y)!=bcir_l_member(x,y)){printf("l_member@%ld,%ld\n",x,y);return 1;}
    if(l_ptr_s(x,y)!=bcir_l_ptr(x,y)){printf("l_ptr@%ld,%ld\n",x,y);return 1;}
    if(d_vararg_s(3,(double)x,(double)y,1.5)!=bcir_d_vararg(3,(double)x,(double)y,1.5)){printf("d_vararg@%ld,%ld\n",x,y);return 1;}
    if(l_vararg_s(3,x,y,x+y)!=bcir_l_vararg(3,x,y,x+y)){printf("l_vararg@%ld,%ld\n",x,y);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/cw_harness.c"
"${CC}" -std=c23 -O2 -I "${C}" "${tmp}/cw_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/cw_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 -I "${C}" "${tmp}/cw_harness.c" "${C}/bcir_quarantine.c" -o "${tmp}/cw_h" \
  || { echo "  FAIL: compoundwide harness build"; exit 1; }
cwr="$("${tmp}/cw_h")"
[ "${cwr}" = "MATCH" ] \
  && echo "  PASS compoundwide: long/double OP= / ++ / -- (local/member/array/ptr) + vararg accum == Clang" \
  || { echo "  FAIL: compoundwide behaviour (${cwr})"; exit 1; }
