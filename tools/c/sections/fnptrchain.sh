#!/usr/bin/env bash
# fnptrchain: funcptr dispatch through a loaded pointer -- d->ops->fn(args) (#fnptrchain)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the
# `c-section-fnptrchain` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It compiles what bcir-cc emits with CC, which the caller
# names: BCIR Make passes the gate's, the CTest entry the configured C compiler, and the
# section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> fnptrchain.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> fnptrchain.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Funcptr dispatch through a loaded pointer (#fnptrchain): a function-pointer struct member reached
# THROUGH a loaded pointer-to-struct field -- `d->ops->fn(args)`, the two-hop `s->dev->ops->fn(args)`.
# The postfix pointer chain recognizes a `(` after a member as a fused indirect call on the loaded
# pointer base (`ptr->fn(args)`). A bespoke harness wires real operation tables (R18-opaque dispatch).
"${BCIR_CC}" --emit-c "${C}/cfront_fnptrchain.c" > "${tmp}/fcc_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bfc_add\b/fc_add_s/' -e 's/\bfc_combo\b/fc_combo_s/' -e 's/\bfc_twohop\b/fc_twohop_s/' \
      "${C}/cfront_fnptrchain.c"
  cat "${tmp}/fcc_emit.c"
  cat <<'DRV'
static int real_add(int a,int b){return a+b;}
static int real_sub(int a,int b){return a-b;}
static int real_mul(int a,int b){return a*b;}
int main(void){
  struct Ops ops={real_add,real_sub,real_mul};
  struct Dev dev={&ops,42};
  struct Sys sys={&dev,7};
  for(int i=-200;i<200;i++){
    int a=i*3-1,b=7-i;
    if(fc_add_s(&dev,a,b)!=bcir_fc_add(&dev,a,b)){printf("add@%d\n",i);return 1;}
    if(fc_combo_s(&dev,a,b)!=bcir_fc_combo(&dev,a,b)){printf("combo@%d\n",i);return 1;}
    if(fc_twohop_s(&sys,a,b)!=bcir_fc_twohop(&sys,a,b)){printf("twohop@%d\n",i);return 1;}
  }
  printf("MATCH\n");return 0;}
DRV
} > "${tmp}/fcc_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/fcc_harness.c" -o "${tmp}/fcc_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/fcc_harness.c" -o "${tmp}/fcc_h" \
  || { echo "  FAIL: fnptrchain harness build"; exit 1; }
fccr="$("${tmp}/fcc_h")"
[ "${fccr}" = "MATCH" ] \
  && echo "  PASS fnptrchain: d->ops->fn(args) / s->dev->ops->fn(args) == Clang" \
  || { echo "  FAIL: fnptrchain behaviour (${fccr})"; exit 1; }
