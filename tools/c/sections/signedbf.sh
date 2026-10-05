#!/usr/bin/env bash
# signedbf: signed bitfield read sign-extends (bcir-cc): emit == Clang (#signedbf)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-signedbf`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> signedbf.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> signedbf.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Signed bitfield read sign-extension (#signedbf): a `signed`/`int` bitfield of width N holds an N-bit
# two's-complement value, so reading it sign-extends from bit N-1 (`int x:4` of 1111 reads -1, not 15).
# Both rails extracted `(unit >> off) & mask` and stopped, zero-extending every read. Now both carry the
# field's signedness on c.bf.get and sign-extend a signed field. Unsigned fields (zero-extend) unchanged.
"${BCIR_CC}" --emit-c "${C}/cfront_signedbf.c" > "${tmp}/sb_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bbf_read\b/sb_rd_src/' -e 's/\bbf_signtest\b/sb_st_src/' -e 's/\bbf_arith\b/sb_ar_src/' \
      -e 's/\bbf_onebit\b/sb_ob_src/' -e 's/\bbf_unsigned\b/sb_un_src/' "${C}/cfront_signedbf.c"
  cat "${tmp}/sb_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<70000u;a+=3u)for(unsigned b=0;b<300u;b++){
    if(sb_rd_src(a,b)!=bcir_bf_read(a,b)){printf("rd MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sb_st_src(a,b)!=bcir_bf_signtest(a,b)){printf("st MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sb_ar_src(a,b)!=bcir_bf_arith(a,b)){printf("ar MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sb_ob_src(a,b)!=bcir_bf_onebit(a,b)){printf("ob MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(sb_un_src(a,b)!=bcir_bf_unsigned(a,b)){printf("un MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/sb_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sb_harness.c" -o "${tmp}/sb_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sb_harness.c" -o "${tmp}/sb_h" \
  || { echo "  FAIL: signedbf harness build"; exit 1; }
sbr="$("${tmp}/sb_h")"
[ "${sbr}" = "MATCH" ] \
  && echo "  PASS signedbf: signed bitfield read sign-extends == Clang" \
  || { echo "  FAIL: signedbf behaviour (${sbr})"; exit 1; }
