#!/usr/bin/env bash
# effects: effect footprint + escape analysis (bcir-cc --emit-effects/--emit-escape) == oracle over the corpus, generated units and forms (#effects, G10)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-effects` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical.
#
#   usage: effects.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: effects.sh <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# The effect footprint, escape verdicts and indirect-call narrowing of a unit (#effects, G10: the C
# twin of bcir/frontends/cfront/escape.py). bcir-cc --emit-effects and --emit-escape must print the
# oracle's reports byte for byte over every corpus unit, the generated units and the forms of
# bcir/tests/escape_fixtures.py (every declaration kind against every access form, in every storage
# place); a unit the twin refuses is a failure, except the pinned preprocessor limits. The same rows,
# with the dynamic commute witness, gate in tools/perf/check_escape.py.
BCIR_CC="${BCIR_CC}" FXDIR="${tmp}/fx" "${PYTHON}" -c "
import os, sys
from bcir.frontends.cfront import compile_unit
from bcir.tests import escape_fixtures as ef
units = ef.corpus()
pairs = [(p, r) for _n, p, _s, r in units]
os.makedirs(os.environ['FXDIR'], exist_ok=True)
extra = [(f'gen_{s}.c', ef.generate(s)[0]) for s in ef.SEEDS]
extra += [(f'forms_{p}.c', src) for p, src in ef.form_units()]
for name, src in extra:
    path = os.path.join(os.environ['FXDIR'], name)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(src)
    pairs.append((path, compile_unit(src, check_clang=False)))
eff, esc, n = ef.rail_parity(os.environ['BCIR_CC'], pairs, ef.TWIN_PREPROCESSOR_LIMITS)
want = len(pairs) - len(ef.TWIN_PREPROCESSOR_LIMITS)
print(f'  compared {n} of {len(pairs)} units; effects mismatches {eff}; escape mismatches {esc}')
sys.exit(0 if (eff, esc, n) == (0, 0, want) and len(units) > 100 else 1)
" && echo "  PASS effects + escape reports byte-identical over every unit (oracle == C)" \
  || { echo "  FAIL: effects/escape parity (#effects)"; exit 1; }
# the gate must span a commuting pair (1) and a conflict (0), else it has no teeth; the report is
# captured, not piped into grep -q (an early exit under pipefail would fail a found pattern)
fx_out="$("${BCIR_CC}" --emit-effects "${C}/cfront_effects.c")" \
  || { echo "  FAIL: bcir-cc --emit-effects cfront_effects.c"; exit 1; }
grep -q "commute read_a read_b = 1" <<<"${fx_out}" && grep -q "= 0$" <<<"${fx_out}" \
  && echo "  PASS effects analysis distinguishes commute (1) from conflict (0)" \
  || { echo "  FAIL: effects gate did not span commute + conflict"; exit 1; }
