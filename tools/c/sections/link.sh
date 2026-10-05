#!/usr/bin/env bash
# link: Phase 3 linking (bcir-cc prototypes): emitted caller + host linker == reference (#link)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-link` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. It compiles what
# bcir-cc emits with CC, which the caller names: the gate passes its own, the CTest entry the
# configured C compiler, and the section-parity gate the same one to both runs.
#
#   usage: CC=<compiler> link.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: CC=<compiler> link.sh <bcir-cc>" >&2; exit 2; }
[ -n "${CC:-}" ] || { echo "usage: CC=<compiler> $(basename "$0") <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Phase 3 LINKING (#link): a file-scope PROTOTYPE makes a cross-TU call a typed external edge
# (c.call.tu:, R18-opaque, extern-declared in the emitted prelude, NO -l derived), and the bcir-cc
# EMITTED caller object host-links against the callee TU with the DERIVED --emit-link-flags and
# behaves exactly like the all-original reference build. A same-unit prototype stays a forward
# declaration (definition wins: the call is a real R18 edge). Both rails accept the caller (rc 0).
mkdir -p "${tmp}/link"
printf 'double scale(double x);\nint main(void) { double v = scale(16.0); return (int)v; }\n' > "${tmp}/link/main.c"
printf '#include <math.h>\ndouble scale(double x) { return sqrt(x) + 1.0; }\n' > "${tmp}/link/lib.c"
"${CC}" -std=c11 "${tmp}/link/main.c" "${tmp}/link/lib.c" -o "${tmp}/link/ref" -lm \
  || { echo "  FAIL: reference build"; exit 1; }
"${tmp}/link/ref"; ref_rc=$?
lflags="$("${BCIR_CC}" --emit-link-flags "${tmp}/link/lib.c")" || { echo "  FAIL: link flags"; exit 1; }
{ echo '#include <stdint.h>'; "${BCIR_CC}" --emit-c "${tmp}/link/main.c"; \
  echo 'int main(void){ return (int)bcir_main(); }'; } > "${tmp}/link/main_emit.c" \
  || { echo "  FAIL: bcir-cc --emit-c (prototyped caller)"; exit 1; }
grep -q "extern double scale(double);" "${tmp}/link/main_emit.c" \
  || { echo "  FAIL: emitted prelude lacks the extern declaration"; exit 1; }
# shellcheck disable=SC2086
"${CC}" -std=c11 "${tmp}/link/main_emit.c" "${tmp}/link/lib.c" -o "${tmp}/link/prog" ${lflags} \
  || { echo "  FAIL: emitted caller did not link"; exit 1; }
"${tmp}/link/prog"; bcir_rc=$?
"${PYTHON}" -m bcir.frontends.cfront -o /dev/null "${tmp}/link/main.c" >/dev/null 2>&1; py_rc=$?
[ "${ref_rc}" = "${bcir_rc}" ] && [ "${ref_rc}" = "5" ] && [ "${py_rc}" = "0" ] \
  && echo "  PASS linking (ref=${ref_rc} == bcir=${bcir_rc}; oracle accepts the caller)" \
  || { echo "  FAIL: linking (ref=${ref_rc} bcir=${bcir_rc} oracle=${py_rc})"; exit 1; }
printf 'unsigned g(unsigned x);\nunsigned f(unsigned y) { return g(y) + 1u; }\nunsigned g(unsigned x) { return x * 2u; }\n' > "${tmp}/link/fwd.c"
"${BCIR_CC}" --emit-claimgraph "${tmp}/link/fwd.c" | grep -q "c.call:g" \
  && echo "  PASS forward declaration stays a real R18 edge (definition wins)" \
  || { echo "  FAIL: forward declaration did not rewrite to c.call:g"; exit 1; }
# ... and the LINKABLE artifact (the C twin of the oracle's emit_linkable): BOTH TUs re-rendered
# by bcir-cc --linkable (external linkage, real names, derived includes) link TO EACH OTHER --
# no original source in the image -- and behave exactly like the reference build.
"${BCIR_CC}" --linkable "${tmp}/link/main.c" > "${tmp}/link/main_lk.c" || { echo "  FAIL: --linkable main"; exit 1; }
"${BCIR_CC}" --linkable "${tmp}/link/lib.c" > "${tmp}/link/lib_lk.c" || { echo "  FAIL: --linkable lib"; exit 1; }
"${CC}" -std=c11 "${tmp}/link/main_lk.c" "${tmp}/link/lib_lk.c" -o "${tmp}/link/prog_lk" -lm \
  || { echo "  FAIL: two --linkable TUs did not link"; exit 1; }
"${tmp}/link/prog_lk"; lk_rc=$?
[ "${lk_rc}" = "${ref_rc}" ] \
  && echo "  PASS linkable artifact (two emitted TUs link to each other; rc=${lk_rc} == ref)" \
  || { echo "  FAIL: linkable artifact rc=${lk_rc} != ref=${ref_rc}"; exit 1; }
# ... and SOURCE-STATIC HONORING on the twin: two TUs each carrying a SAME-NAMED static helper
# keep `static` in the --linkable rendering (internal linkage), so the pair still links into
# one binary -- a stripped-static (exported) rendering would be a duplicate-symbol link error.
printf 'static unsigned mix(unsigned x) { return x * 3u; }\nunsigned fa(unsigned x) { return mix(x) + 5u; }\n' > "${tmp}/link/sa.c"
printf 'static unsigned mix(unsigned x) { return x + 100u; }\nunsigned fb(unsigned x) { return mix(x); }\n' > "${tmp}/link/sb.c"
printf 'extern unsigned fa(unsigned);\nextern unsigned fb(unsigned);\nint main(void){ return (int)(fa(2u) + fb(1u)); }\n' > "${tmp}/link/smain.c"
"${BCIR_CC}" --linkable "${tmp}/link/sa.c" > "${tmp}/link/sa_lk.c" || { echo "  FAIL: --linkable sa"; exit 1; }
"${BCIR_CC}" --linkable "${tmp}/link/sb.c" > "${tmp}/link/sb_lk.c" || { echo "  FAIL: --linkable sb"; exit 1; }
grep -q "static uint32_t mix(uint32_t x)" "${tmp}/link/sa_lk.c" \
  || { echo "  FAIL: source-static helper lost its static in the linkable rendering"; exit 1; }
grep -q "^uint32_t fa(uint32_t x)" "${tmp}/link/sa_lk.c" \
  || { echo "  FAIL: the exported function should be non-static under its real name"; exit 1; }
"${CC}" -std=c11 -Wall -Werror "${tmp}/link/smain.c" "${tmp}/link/sa_lk.c" "${tmp}/link/sb_lk.c" \
  -o "${tmp}/link/prog_st" || { echo "  FAIL: same-named statics did not link"; exit 1; }
"${tmp}/link/prog_st"; st_rc=$?
[ "${st_rc}" = "112" ] \
  && echo "  PASS source-static honoring (same-named statics per TU; rc=${st_rc})" \
  || { echo "  FAIL: source-static honoring rc=${st_rc} != 112"; exit 1; }
# ... review-hardened edges: (1) the static-forward-declaration idiom -- the tudefs prelude
# re-keys `extern` -> `static` for a kept-static function, so the artifact compiles (C11
# 6.2.2p7); (2) a `_BitInt(N)` return spelling carries parens BEFORE the name -- the static
# keeper must still find the function name (grep-only: host cc under -std=c11 has no _BitInt).
printf 'static int r(int v);\nint s(int x) { return r(x); }\nstatic int r(int x) { return x + 1; }\n' > "${tmp}/link/sp.c"
"${BCIR_CC}" --linkable "${tmp}/link/sp.c" > "${tmp}/link/sp_lk.c" || { echo "  FAIL: --linkable sp"; exit 1; }
grep -q "^static int32_t r(int32_t);" "${tmp}/link/sp_lk.c" \
  || { echo "  FAIL: static prototype should re-key its extern prelude to static"; exit 1; }
"${CC}" -std=c11 -Wall -Werror -c "${tmp}/link/sp_lk.c" -o "${tmp}/link/sp_lk.o" \
  && echo "  PASS static forward declaration (extern prelude re-keyed; artifact compiles)" \
  || { echo "  FAIL: static-forward-declaration artifact did not compile"; exit 1; }
printf 'static _BitInt(13) bh(int x) { return (_BitInt(13))(x + 1); }\nint bu(int y) { return (int)bh(y); }\n' > "${tmp}/link/sb2.c"
"${BCIR_CC}" --linkable "${tmp}/link/sb2.c" > "${tmp}/link/sb2_lk.c" || { echo "  FAIL: --linkable sb2"; exit 1; }
grep -q "^static _BitInt(13) bh(" "${tmp}/link/sb2_lk.c" \
  && echo "  PASS _BitInt-return static keeps its static (multi-paren name scan)" \
  || { echo "  FAIL: a _BitInt return spelling defeated the static keeper"; exit 1; }
