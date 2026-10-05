#!/usr/bin/env bash
# threadlocal: thread-local storage (bcir-cc): _Thread_local emit == Clang (#threadlocal)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-threadlocal` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It compiles what bcir-cc emits with CC, which the caller
# names: BCIR Make passes the gate's, the CTest entry the configured C compiler, and the
# section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> threadlocal.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> threadlocal.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# `_Thread_local` / `thread_local` (#threadlocal): a per-thread storage class. Recognized + consumed;
# the global is referenced by name (the source defines it). In the deterministic single-thread harness a
# thread-local behaves as a global, so the twin's --emit-c is Clang-behaviour-equivalent.
"${BCIR_CC}" --emit-c "${C}/cfront_threadlocal.c" > "${tmp}/tl_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; cat "${C}/cfront_threadlocal.c" "${tmp}/tl_emit.c"
  cat <<'DRV'
int main(void){
  for(unsigned t=0;t<3000u;t++){
    unsigned x=t*7u+1u;
    tls_ctr = t & 255u;
    if(scaled(x)!=bcir_scaled(x) || offs(x)!=bcir_offs(x)){printf("read MISMATCH t=%u\n",t);return 1;}
    tls_ctr=t; bump(3u);      unsigned src=tls_ctr;
    tls_ctr=t; bcir_bump(3u); unsigned twn=tls_ctr;
    if(src!=twn){printf("write MISMATCH t=%u\n",t);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/tl_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/tl_harness.c" -o "${tmp}/tl_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/tl_harness.c" -o "${tmp}/tl_h" \
  || { echo "  FAIL: threadlocal harness build"; exit 1; }
tlr="$("${tmp}/tl_h")"
[ "${tlr}" = "MATCH" ] \
  && echo "  PASS threadlocal: read + write a _Thread_local global == Clang" \
  || { echo "  FAIL: threadlocal behaviour (${tlr})"; exit 1; }
