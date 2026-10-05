#!/usr/bin/env bash
# stmtexpr: GCC statement expressions -- ({ ...; e; }) (#stmtexpr)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-stmtexpr` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> stmtexpr.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> stmtexpr.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# GCC statement expressions (#stmtexpr): `({ ...; e; })` -- a scoped compound statement whose value is the
# last expression. The differential drives the temporary idiom, a loop inside, nesting, and scope
# shadowing; a wrong value / leaked scope would diverge.
"${BCIR_CC}" --emit-c "${C}/cfront_stmtexpr.c" > "${tmp}/sx_emit.c" || { echo "  FAIL: --emit-c"; exit 1; }
{ echo '#include <stdint.h>'; echo '#include <stdio.h>'; echo '#include <string.h>'
  sed -e 's/\bse_simple\b/se_simple_s/g' -e 's/\bse_max\b/se_max_s/g' -e 's/\bse_embed\b/se_embed_s/g' \
      -e 's/\bse_loop\b/se_loop_s/g' -e 's/\bse_nest\b/se_nest_s/g' -e 's/\bse_scope\b/se_scope_s/g' \
      -e 's/\bse_void\b/se_void_s/g' \
      "${C}/cfront_stmtexpr.c"
  cat "${tmp}/sx_emit.c"
  cat <<'DRV'
int main(void){
  for(int a=-60;a<60;a++) for(int b=-25;b<25;b++){
    if(se_simple_s(a)!=bcir_se_simple(a)){puts("simple");return 1;}
    if(se_max_s(a,b)!=bcir_se_max(a,b)){puts("max");return 1;}
    if(se_embed_s(a)!=bcir_se_embed(a)){puts("embed");return 1;}
    if(se_loop_s(b)!=bcir_se_loop(b)){puts("loop");return 1;}
    if(se_nest_s(a)!=bcir_se_nest(a)){puts("nest");return 1;}
    if(se_scope_s(a)!=bcir_se_scope(a)){puts("scope");return 1;}
    if(se_void_s(a)!=bcir_se_void(a)){puts("void");return 1;}
  }
  puts("MATCH");return 0;}
DRV
} > "${tmp}/sx_harness.c"
"${CC}" -std=c23 -O2 "${tmp}/sx_harness.c" -o "${tmp}/sx_h" 2>/dev/null \
  || "${CC}" -std=c2x -O2 "${tmp}/sx_harness.c" -o "${tmp}/sx_h" \
  || { echo "  FAIL: stmtexpr harness build"; exit 1; }
sxr="$("${tmp}/sx_h")"
[ "${sxr}" = "MATCH" ] \
  && echo "  PASS stmtexpr: temporary / max / embedded / loop / nested / scope / void == Clang" \
  || { echo "  FAIL: stmtexpr behaviour (${sxr})"; exit 1; }
