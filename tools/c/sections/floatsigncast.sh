#!/usr/bin/env bash
# floatsigncast: float -> signed-int cast (bcir-cc): emit == Clang (#floatsigncast)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-floatsigncast` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> floatsigncast.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> floatsigncast.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# float / double -> SIGNED integer cast (#floatsigncast): a floating value converted to a signed integer
# must use a signed cast operator. Both rails canonicalized integer casts to the unsigned spelling, making
# it a float -> unsigned conversion -- UB for a negative value, target-divergent (the #395 aarch64 32-bit
# failure), and even on x86 a sub-int signed target lost the sign. Now emit a signed temp + signed operator.
"${BCIR_CC}" --emit-c "${C}/cfront_floatsigncast.c" > "${tmp}/fs_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'
  sed -e 's/\bf2int\b/f2i_src/' -e 's/\bf2short\b/f2s_src/' -e 's/\bf2schar\b/f2c_src/' \
      -e 's/\bd2long\b/d2l_src/' -e 's/\bf2uint\b/f2u_src/' "${C}/cfront_floatsigncast.c"
  cat "${tmp}/fs_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<4000u;a++)for(unsigned b=0;b<90u;b++){
    if(f2i_src(a,b)!=bcir_f2int(a,b)){printf("f2i MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(f2s_src(a,b)!=bcir_f2short(a,b)){printf("f2s MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(f2c_src(a,b)!=bcir_f2schar(a,b)){printf("f2c MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(d2l_src(a,b)!=bcir_d2long(a,b)){printf("d2l MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(f2u_src(a,b)!=bcir_f2uint(a,b)){printf("f2u MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/fs_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/fs_harness.c" -o "${tmp}/fs_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/fs_harness.c" -o "${tmp}/fs_h" \
  || { echo "  FAIL: floatsigncast harness build"; exit 1; }
fsr="$("${tmp}/fs_h")"
[ "${fsr}" = "MATCH" ] \
  && echo "  PASS floatsigncast: float/double -> signed int == Clang" \
  || { echo "  FAIL: floatsigncast behaviour (${fsr})"; exit 1; }
