#!/usr/bin/env bash
# r21policy: R21 lifetime policy (bcir-cc --r21): verdict + exit-code parity vs the oracle (#r21policy)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-r21policy` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical.
#
#   usage: r21policy.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: r21policy.sh <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# R21 lifetime policy (bcir-cc --r21, §5.12): a detected use-after-free / double-free becomes a
# VERDICT under a non-advisory policy, and the C twin (bcir-cc) and the Python rail (bcir-cfront) must
# draw the SAME exit code. advisory (default) never gates (0); fallback routes to LLVM (2); reject is a
# hard error (1); a clean (no-UAF) unit stays 0 under every policy. Detection is the same freed-set walk.
mkdir -p "${tmp}/r21"
printf '#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); return p[0]; }\n'           > "${tmp}/r21/uaf.c"
printf '#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); free(p); free(p); return n; }\n'      > "${tmp}/r21/dfree.c"
printf '#include <stdlib.h>\nunsigned f(unsigned n){ unsigned *p=malloc(n*sizeof(unsigned)); unsigned r=p[0]; free(p); return r; }\n' > "${tmp}/r21/clean.c"
r21_seen=""
for pol in advisory fallback reject; do
  for fx in uaf dfree clean; do
    "${BCIR_CC}" --r21=${pol} "${tmp}/r21/${fx}.c" >/dev/null 2>&1; trc=$?
    "${PYTHON}" -m bcir.frontends.cfront --r21=${pol} -o /dev/null "${tmp}/r21/${fx}.c" >/dev/null 2>&1; prc=$?
    [ "${trc}" = "${prc}" ] \
      && echo "  PASS r21 ${pol}/${fx} (rc oracle == C: ${trc})" \
      || { echo "  FAIL: r21 ${pol}/${fx} (C rc=${trc} PY rc=${prc})"; exit 1; }
    r21_seen="${r21_seen}${trc}"
  done
done
# the policy gate must reach all three verdicts (clean 0 / reject 1 / fallback 2), else it has no teeth.
case "${r21_seen}" in
  *0*) case "${r21_seen}" in *1*) case "${r21_seen}" in *2*)
    echo "  PASS r21 policy spans advisory / fallback / reject verdicts" ;;
    *) echo "  FAIL: r21 gate never reached a fallback (2): ${r21_seen}"; exit 1 ;; esac ;;
    *) echo "  FAIL: r21 gate never reached a reject (1): ${r21_seen}"; exit 1 ;; esac ;;
  *) echo "  FAIL: r21 gate never reached a clean (0): ${r21_seen}"; exit 1 ;;
esac
