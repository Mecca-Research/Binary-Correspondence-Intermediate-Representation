#!/usr/bin/env bash
# extvariadic: external variadic calls -- snprintf / vsnprintf passthrough (#extvariadic)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-extvariadic` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> extvariadic.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> extvariadic.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# External variadic calls (#extvariadic): the printf/scanf-family <stdio.h> variadics (snprintf/vsnprintf)
# emit verbatim, stay opaque to R18 (no bcir_ twin), return int; the format string passes through; a
# vsnprintf-forwarding wrapper hands its own va_list to the external. The differential compares the
# formatted BUFFER and the returned count against the real libc.
"${BCIR_CC}" --emit-c "${C}/cfront_extvariadic.c" > "${tmp}/ev_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'; echo '#include <stdarg.h>'
  sed -e 's/\bev_int\b/ev_int_s/g' -e 's/\bev_mix\b/ev_mix_s/g' -e 's/\bev_width\b/ev_width_s/g' \
      -e 's/\bev_fwd\b/ev_fwd_s/g' -e 's/\bev_call\b/ev_call_s/g' \
      "${C}/cfront_extvariadic.c"
  cat "${tmp}/ev_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-3000;x<3000;x++){ long y=(long)x*1234567L; char b1[64],b2[64]; int r1,r2;
    r1=ev_int_s(b1,x);   r2=bcir_ev_int(b2,x);   if(r1!=r2||strcmp(b1,b2)){printf("int@%d\n",x);return 1;}
    r1=ev_mix_s(b1,x,y); r2=bcir_ev_mix(b2,x,y); if(r1!=r2||strcmp(b1,b2)){printf("mix@%d\n",x);return 1;}
    r1=ev_width_s(b1,x); r2=bcir_ev_width(b2,x); if(r1!=r2||strcmp(b1,b2)){printf("width@%d\n",x);return 1;}
    r1=ev_call_s(b1,x,(int)(y&0xff)); r2=bcir_ev_call(b2,x,(int)(y&0xff));
    if(r1!=r2||strcmp(b1,b2)){printf("call@%d\n",x);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ev_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ev_harness.c" -o "${tmp}/ev_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ev_harness.c" -o "${tmp}/ev_h" \
  || { echo "  FAIL: extvariadic harness build"; exit 1; }
evr="$("${tmp}/ev_h")"
[ "${evr}" = "MATCH" ] \
  && echo "  PASS extvariadic: snprintf/vsnprintf passthrough (buffer + count) == Clang" \
  || { echo "  FAIL: extvariadic behaviour (${evr})"; exit 1; }
