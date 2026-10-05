#!/usr/bin/env bash
# scale: scalable IR (bcir-cc, no fixed BCIR_MAX_*): cap-busting unit compiles + matches oracle (#scale)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-scale` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical.
#
#   usage: scale.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: scale.sh <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Scalable IR (no fixed BCIR_MAX_*): a unit that busts every OLD ceiling -- 43 functions (> the old
# BCIR_MAX_FUNCS 16), many12 with 12 params (> 8), agg with 40 calls (> 32), big with 7500 claims
# (> the old 4096 per-function cap). The IR grows geometrically, so the twin compiles it clean and
# matches the oracle's structural counts; the old fixed arrays would have rejected it.
"${PYTHON}" - "${tmp}/cfront_scale.c" <<'PY'
import sys
fns=[f"unsigned g{k}(void){{ return {k}u; }}" for k in range(40)]
ps=[chr(ord('a')+i) for i in range(12)]
fns.append("unsigned many12("+",".join(f"unsigned {p}" for p in ps)+"){ return "+"+".join(ps)+"; }")
fns.append("unsigned agg(void){ return "+"+".join(f"g{k}()" for k in range(40))+"; }")
body="\n".join("  acc = acc + 1u;" for _ in range(2500))
fns.append("unsigned big(unsigned acc){\n"+body+"\n  return acc;\n}")
open(sys.argv[1],"w").write("\n".join(fns)+"\n")
PY
c_scale="$("${BCIR_CC}" --emit-claimgraph "${tmp}/cfront_scale.c" 2>&1 | grep -oE 'funcs=[0-9]+ claims=[0-9]+.*ok=[0-9]')"
case "${c_scale}" in
  "funcs=43 claims=7500"*"ok=1") echo "  PASS scale: 43 funcs / 7500-claim fn compile clean (old caps 16 / 4096 exceeded)";;
  *) echo "  FAIL: scale unit (twin: ${c_scale})"; exit 1;;
esac
py_scale="$("${PYTHON}" -c "
from bcir.frontends.cfront import compile_unit
r=compile_unit(open('${tmp}/cfront_scale.c').read(), check_clang=False)
print('funcs=%d big=%d' % (len(r.lowered.functions), len(r.lowered.functions['big'].claims)))")"
[ "${py_scale}" = "funcs=43 big=7500" ] \
  && echo "  PASS scale: oracle agrees (${py_scale})" \
  || { echo "  FAIL: scale oracle mismatch (${py_scale})"; exit 1; }
