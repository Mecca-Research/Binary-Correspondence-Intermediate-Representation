#!/usr/bin/env bash
# pscale: scalable parser state (bcir-cc, no fixed gv/s/td/ec/env caps): cap-busting unit == oracle (#pscale)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-pscale`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical.
#
#   usage: pscale.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: pscale.sh <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Scalable parser state (#pscale): segment-1 made the IR arrays grow; this removes the twin's fixed
# *parser-state* caps too -- struct defs (was s[16]), file-scope globals (was gv[16]), typedefs (was
# td[64]), enum constants (was ec[256]) and locals (was env[256]) all grow geometrically (reused across
# compiles via a save/restore around the static CC). A unit that busts every old cap compiles clean on
# the twin and matches the oracle's structure -- real headers (many globals / structs / typedefs) lower.
"${PYTHON}" - "${tmp}/pstress.c" <<'PY'
import sys
L=[f"struct S{k} {{ unsigned m0; unsigned m1; }};" for k in range(20)]      # 20 struct defs (> old s[16])
L+=[f"typedef unsigned U{k};" for k in range(20)]                            # 20 typedefs
L+=[f"static const unsigned G{k}[2] = {{ {k}u, {k+1}u }};" for k in range(25)]  # 25 globals (> old gv[16])
L.append("unsigned big(void){\n"+"\n".join(f"  unsigned v{i} = {i}u;" for i in range(300))+   # 300 locals (>256)
         "\n  return "+"+".join(f"v{i}" for i in range(300))+"; }")
L.append("unsigned useg(unsigned i){ return G0[i%2u] + G19[i%2u] + G24[i%2u]; }")
open(sys.argv[1],"w").write("\n".join(L)+"\n")
PY
c_ps="$("${BCIR_CC}" --emit-claimgraph "${tmp}/pstress.c" 2>&1 | grep -oE 'funcs=[0-9]+ claims=[0-9]+.*ok=[0-9] digest=[0-9a-f]+' | tail -1)"
py_ps="$("${PYTHON}" -c "
from bcir.frontends.cfront import compile_unit
from bcir.model import Domain
from bcir.verify import cfront_structural_digest          # the cross-rail per-claim STRUCTURAL digest
r=compile_unit(open('${tmp}/pstress.c').read(), check_clang=False)
fns=r.lowered.functions; lf=fns[next(reversed(fns))]
kn=sum(1 for c in lf.claims if c.op=='c.const'); bo=sum(1 for c in lf.claims if c.op.startswith('c.bin.'))
repro=sum(1 for f in fns.values() if getattr(f,'reproducible',False))  # A1.3: matches the C twin's repro=N
dg=cfront_structural_digest(r.lowered)                    # byte-identical to the C twin's bcir_cfront_digest
print(f'funcs={len(fns)} claims={len(lf.claims)} mmio=0 bf=0 const={kn} binop={bo} call=0 repro={repro} ok={1 if r.is_clean else 0} digest={dg:016x}')")"
[ -n "${c_ps}" ] && [ "${c_ps}" = "${py_ps}" ] \
  && echo "  PASS pscale: 20 structs / 25 globals / 300 locals compile clean == oracle (${c_ps})" \
  || { echo "  FAIL: pscale (C='${c_ps}' PY='${py_ps}')"; exit 1; }
