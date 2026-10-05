#!/usr/bin/env bash
# intpromote: integer promotions + UAC (bcir-cc): signed/mixed-width emit == Clang full-range (#intpromote)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-intpromote` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It compiles what bcir-cc emits with CC, which the caller
# names: BCIR Make passes the gate's, the CTest entry the configured C compiler, and the
# section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> intpromote.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> intpromote.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Integer promotions + usual arithmetic conversions (#intpromote): the C twin types each temp by its
# true (width, signedness), so a signed int divide / remainder / right-shift / comparison emits signed
# C (not the old flat uint32_t) and a mixed-width op widens per §6.3.1.8. The twin's --emit-c is
# behaviour-equivalent to Clang over the FULL signed range (negatives) -- exactly what the old model
# got wrong. Compile the emitted bcir_* beside the source + a full-range differential driver.
"${BCIR_CC}" --emit-c "${C}/cfront_intpromote.c" > "${tmp}/ip_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; cat "${C}/cfront_intpromote.c" "${tmp}/ip_emit.c"
  cat <<'DRV'
static uint64_t S=0x9E3779B97F4A7C15u;
static uint64_t nx(void){S=S*6364136223846793005u+1442695040888963407u;return S>>32;}
int main(void){
  for(int i=0;i<300000;i++){
    int a=(int)nx(), b=(int)nx(); long lb=(long)nx()<<3 | (long)nx();
    if(sdiv(a,b)!=bcir_sdiv(a,b)||smod(a,b)!=bcir_smod(a,b)||sshr(a)!=bcir_sshr(a)
     ||scmp(a,b)!=bcir_scmp(a,b)||udiv((unsigned)a,(unsigned)b)!=bcir_udiv((unsigned)a,(unsigned)b)
     ||wide(a,lb)!=bcir_wide(a,lb)||umix((unsigned)a,lb)!=bcir_umix((unsigned)a,lb)){
       printf("MISMATCH@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/ip_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/ip_harness.c" -o "${tmp}/ip_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/ip_harness.c" -o "${tmp}/ip_h" \
  || { echo "  FAIL: intpromote harness build"; exit 1; }
ipr="$("${tmp}/ip_h")"
[ "${ipr}" = "MATCH" ] \
  && echo "  PASS intpromote: signed div/mod/shr/compare + mixed-width == Clang (full signed range)" \
  || { echo "  FAIL: intpromote behaviour (${ipr})"; exit 1; }
# teeth: the emit carries the real signed + wide integer types (the old model emitted only uint32_t).
{ grep -q "int32_t" "${tmp}/ip_emit.c" && grep -q "int64_t" "${tmp}/ip_emit.c"; } \
  && echo "  PASS intpromote: emit carries true signed (int32_t) + wide (int64_t) types" \
  || { echo "  FAIL: intpromote emit missing signed/wide types"; exit 1; }
