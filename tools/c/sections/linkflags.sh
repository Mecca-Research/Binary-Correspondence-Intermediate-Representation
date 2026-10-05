#!/usr/bin/env bash
# linkflags: automatic link-flag emission (bcir-cc --emit-link-flags): derived flags == oracle (#linkflags)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-linkflags` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical.
#
#   usage: linkflags.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: linkflags.sh <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Automatic link-flag emission (#linkflags, B1): the compiler DERIVES the linker flags a translation
# unit needs from its external-call edges (c.call.libm:/.void:/extern:), instead of every harness
# hard-coding -lm. The callee->library mapping (linkflags.py / bcir_cfront.c bcir_lib_for_callee) is the
# dual-rail source of truth; bcir-cc --emit-link-flags must match the oracle (bcir-cfront
# --emit-link-flags) BYTE-FOR-BYTE. Cover a pure-integer unit (no flags), a math.h unit (-lm), and a
# free()/malloc unit (no flag -- libc is implicit), spanning the empty + non-empty cases.
mkdir -p "${tmp}/lf"
printf 'unsigned f(unsigned a){ return a*3u + 1u; }\n'                                              > "${tmp}/lf/pureint.c"
printf '#include <math.h>\ndouble f(double x){ return sqrt(x) + floor(x); }\n'                       > "${tmp}/lf/mathh.c"
printf '#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*4u); unsigned r=p[0]; free(p); return r; }\n' > "${tmp}/lf/free.c"
lf_seen=""
for lf in pureint mathh free; do
  c_lf="$("${BCIR_CC}" --emit-link-flags "${tmp}/lf/${lf}.c")" || { echo "  FAIL: bcir-cc --emit-link-flags ${lf}"; exit 1; }
  py_lf="$("${PYTHON}" -m bcir.frontends.cfront --emit-link-flags "${tmp}/lf/${lf}.c")" || { echo "  FAIL: oracle link-flags ${lf}"; exit 1; }
  [ "${c_lf}" = "${py_lf}" ] \
    && echo "  PASS linkflags ${lf} (oracle == C: [${c_lf}])" \
    || { echo "  FAIL: linkflags ${lf} (C='[${c_lf}]' PY='[${py_lf}]')"; exit 1; }
  lf_seen="${lf_seen}[${c_lf}]"
done
# expected: pure-int empty, math.h -lm, free empty -- the gate must span the empty + the -lm case.
[ "${lf_seen}" = "[][-lm][]" ] \
  && echo "  PASS linkflags spans pure-int (no flags) / math.h (-lm) / free (libc-implicit, no flag)" \
  || { echo "  FAIL: linkflags gate did not span the expected cases: ${lf_seen}"; exit 1; }
# --emit-c is self-describing: the C.2 attestation header carries the derived link_flags line.
"${BCIR_CC}" --emit-c "${tmp}/lf/mathh.c" | grep -q "link_flags  -lm" \
  && echo "  PASS linkflags: --emit-c attestation header carries the derived flags (link_flags -lm)" \
  || { echo "  FAIL: --emit-c header missing the derived link_flags line"; exit 1; }
