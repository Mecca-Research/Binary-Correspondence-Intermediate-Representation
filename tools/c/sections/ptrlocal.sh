#!/usr/bin/env bash
# ptrlocal: pointer locals -- T *p = &x (bcir-cc): emit == Clang (#ptrlocal)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-ptrlocal` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> ptrlocal.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> ptrlocal.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Pointer locals (#ptrlocal): `T *p = &x;` + read/write `*p`. The twin emitted a pointer local as a plain
# uint32_t -- an invalid pointer-to-integer assignment under C23 and an 8-byte-pointer truncation on a
# 64-bit target. Now it carries the pointee type on the resource and emits a real `T *p`, with the deref
# reading exactly the pointee width (a signed sub-int pointee sign-extends). The oracle was already correct.
"${BCIR_CC}" --emit-c "${C}/cfront_ptrlocal.c" > "${tmp}/pl_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bpl_store\b/pl_st_src/' -e 's/\bpl_read\b/pl_rd_src/' -e 's/\bpl_schar\b/pl_sc_src/' \
      -e 's/\bpl_compound\b/pl_cp_src/' -e 's/\bpl_struct\b/pl_su_src/' -e 's/\bpl_reassign\b/pl_re_src/' \
      "${C}/cfront_ptrlocal.c"
  cat "${tmp}/pl_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned a=0;a<70000u;a+=3u)for(unsigned b=0;b<300u;b++){
    if(pl_st_src(a,b)!=bcir_pl_store(a,b)){printf("st MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_rd_src(a,b)!=bcir_pl_read(a,b)){printf("rd MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_sc_src(a,b)!=bcir_pl_schar(a,b)){printf("sc MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_cp_src(a,b)!=bcir_pl_compound(a,b)){printf("cp MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_su_src(a,b)!=bcir_pl_struct(a,b)){printf("su MISMATCH a=%u b=%u\n",a,b);return 1;}
    if(pl_re_src(a,b)!=bcir_pl_reassign(a,b)){printf("re MISMATCH a=%u b=%u\n",a,b);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/pl_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/pl_harness.c" -o "${tmp}/pl_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/pl_harness.c" -o "${tmp}/pl_h" \
  || { echo "  FAIL: ptrlocal harness build"; exit 1; }
plr="$("${tmp}/pl_h")"
[ "${plr}" = "MATCH" ] \
  && echo "  PASS ptrlocal: T *p = &x read/write/compound/struct == Clang" \
  || { echo "  FAIL: ptrlocal behaviour (${plr})"; exit 1; }
