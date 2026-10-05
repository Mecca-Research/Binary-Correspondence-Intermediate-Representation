#!/usr/bin/env bash
# cexpr: constant-expression evaluator: enum/case/static folds == reference (#cexpr)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-cexpr`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. It compiles what bcir-cc emits with CC, which the caller names: BCIR Make passes
# the gate's, the CTest entry the configured C compiler, and the section-parity gate the same one to
# both runs.
#
#   usage: CC=<compiler> cexpr.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> cexpr.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# The §5.9 constant-expression evaluator (#cexpr): enum initializers, case labels and
# static-local initializers fold comparisons / logical ops / the ternary on BOTH rails
# (ce_expr == the oracle's _const_eval vocabulary); the emitted program behaves like the
# reference (41 + 7 + 53 == 101). The emit builds as C11 with every pedantic diagnostic an
# error: each case label ends in a null statement, so the temporary declared after it is
# no C23 label-then-declaration (CF-CASELABEL).
cat > "${tmp}/cexpr.c" <<'CEXPR'
enum { A = 1 << 4, B = A + 2, C = (A > 10) ? 7 : 9, D = (1 < 2) && (3 != 4) };
unsigned pick(unsigned x) {
    static unsigned seed = (5 > 3) ? 40u : 2u;
    switch (x) {
    case A: return seed + 1u;
    case B: return (unsigned)C;
    default: return (unsigned)D + 52u;
    }
}
int main(void) { return (int)(pick(16u) + pick(18u) + pick(0u)); }
CEXPR
"${CC}" -std=c11 "${tmp}/cexpr.c" -o "${tmp}/cexpr_ref" || { echo "  FAIL: cexpr reference build"; exit 1; }
"${tmp}/cexpr_ref"; cexpr_ref=$?
{ echo '#include <stdint.h>'; "${BCIR_CC}" --emit-c "${tmp}/cexpr.c"; \
  echo 'int main(void){ return (int)bcir_main(); }'; } > "${tmp}/cexpr_emit.c" \
  || { echo "  FAIL: bcir-cc --emit-c (cexpr)"; exit 1; }
grep -q "static uint32_t seed = 40u;" "${tmp}/cexpr_emit.c" \
  || { echo "  FAIL: the ternary static initializer did not fold to 40"; exit 1; }
"${CC}" -std=c11 -pedantic-errors "${tmp}/cexpr_emit.c" -o "${tmp}/cexpr_prog" || { echo "  FAIL: cexpr emitted build"; exit 1; }
"${tmp}/cexpr_prog"; cexpr_rc=$?
[ "${cexpr_rc}" = "${cexpr_ref}" ] && [ "${cexpr_rc}" = "101" ] \
  && echo "  PASS constant-expression folds (rc=${cexpr_rc} == ref; the emit builds as pedantic C11)" \
  || { echo "  FAIL: cexpr rc=${cexpr_rc} != ref=${cexpr_ref}"; exit 1; }
