#!/usr/bin/env bash
# addrmember: address-of a (nested) struct member -- &s.f / &t.q.a / &t.q (#addrmember)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-addrmember` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It compiles what bcir-cc emits with CC, which the caller
# names: BCIR Make passes the gate's, the CTest entry the configured C compiler, and the
# section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> addrmember.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> addrmember.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Address-of a (nested) struct member (#addrmember): `&s.field` / `&t.q.a` / `&t.q` -> a typed
# `(T *)((char *)&base + off)`, used through a pointer and passed to a helper. A wrong offset/type would
# write the wrong slot or pass a bad address.
"${BCIR_CC}" --emit-c "${C}/cfront_addrmember.c" > "${tmp}/am_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\baddone\b/addone_s/g' -e 's/\bam_first\b/am_first_s/g' -e 's/\bam_nested\b/am_nested_s/g' \
      -e 's/\bam_struct\b/am_struct_s/g' -e 's/\bam_arg\b/am_arg_s/g' \
      "${C}/cfront_addrmember.c"
  cat "${tmp}/am_emit.c"
  cat <<'DRV'
int main(void){
  for(int x=-200;x<200;x++){
    if(am_first_s(x)!=bcir_am_first(x)){puts("first");return 1;}
    if(am_nested_s(x)!=bcir_am_nested(x)){puts("nested");return 1;}
    if(am_struct_s(x)!=bcir_am_struct(x)){puts("struct");return 1;}
    if(am_arg_s(x)!=bcir_am_arg(x)){puts("arg");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/am_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/am_harness.c" -o "${tmp}/am_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/am_harness.c" -o "${tmp}/am_h" \
  || { echo "  FAIL: addrmember harness build"; exit 1; }
amr="$("${tmp}/am_h")"
[ "${amr}" = "MATCH" ] \
  && echo "  PASS addrmember: &s.f / &t.q.a / &t.q (read/write/struct-ptr/arg) == Clang" \
  || { echo "  FAIL: addrmember behaviour (${amr})"; exit 1; }
