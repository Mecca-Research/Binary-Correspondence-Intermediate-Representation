#!/usr/bin/env bash
# project: project mode (bcir-cc multi-file): verdict line + exit code == oracle (#project)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-project`
# CTest entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md),
# so the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical.
#
#   usage: project.sh <bcir-cc>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: project.sh <bcir-cc>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BCIR_CC="$(abs "$1")"; [ -x "${BCIR_CC}" ] || { echo "  FAIL: BCIR_CC: ${BCIR_CC} is not an executable bcir-cc"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Project mode (#project, Phase 3): over a multi-file invocation both drivers print ONE per-project
# verdict line -- CLEAN (every unit clean) / PARTIAL-FALLBACK (no failure, >=1 unit routed to LLVM)
# / DIRTY (>=1 unit failed) -- and draw the same exit code (a hard error 1 DOMINATES a fallback 2).
# The verdict line must match the oracle BYTE-FOR-BYTE and the gate must span all three verdicts.
mkdir -p "${tmp}/proj"
printf 'unsigned f(unsigned x){ return x + 1u; }\n'                    > "${tmp}/proj/a.c"
printf 'unsigned g(unsigned x){ return x * 2u; }\n'                    > "${tmp}/proj/b.c"
printf 'unsigned h(unsigned n){ unsigned a[n][n][n][n]; return a[0][0][0][0]; }\n' > "${tmp}/proj/fb.c"   # >3-D VLA: fallback
printf 'unsigned r(unsigned n){ return r(n-1u); }\n'                   > "${tmp}/proj/bad.c"  # R18 recursion: dirty
proj_seen=""
proj_case() {  # $1 = scenario name; the rest is the shared argv both drivers get verbatim
  pname="$1"; shift
  c_all="$("${BCIR_CC}" "$@" 2>/dev/null)"; c_rc=$?
  p_all="$("${PYTHON}" -m bcir.frontends.cfront "$@" 2>/dev/null)"; p_rc=$?
  c_out="$(printf '%s\n' "${c_all}" | tail -1)"
  p_out="$(printf '%s\n' "${p_all}" | tail -1)"
  [ "${c_rc}" = "${p_rc}" ] || { echo "  FAIL: project ${pname} (C rc=${c_rc} PY rc=${p_rc})"; exit 1; }
  [ "${c_out}" = "${p_out}" ] || { echo "  FAIL: project ${pname} (C '${c_out}' != PY '${p_out}')"; exit 1; }
  case "${c_out}" in "project: "*) : ;; *) echo "  FAIL: project ${pname}: no verdict line ('${c_out}')"; exit 1 ;; esac
  echo "  PASS project ${pname} (rc=${c_rc}: ${c_out})"
  proj_seen="${proj_seen}${c_out};"
}
proj_case clean               "${tmp}/proj/a.c" "${tmp}/proj/b.c"
proj_case single --project    "${tmp}/proj/a.c"
proj_case fellback --fallback "${tmp}/proj/a.c" "${tmp}/proj/fb.c"
# dirty AFTER fallback in rc terms: bad.c sets rc=1 first, fb.c must NOT overwrite it with 2.
proj_case dirty  --fallback   "${tmp}/proj/a.c" "${tmp}/proj/bad.c" "${tmp}/proj/fb.c"
# the gate must span CLEAN / PARTIAL-FALLBACK / DIRTY, and 1 must have dominated 2 in the dirty mix.
case "${proj_seen}" in *"CLEAN"*) case "${proj_seen}" in *"PARTIAL-FALLBACK"*) case "${proj_seen}" in *"DIRTY"*)
  echo "  PASS project verdicts span CLEAN / PARTIAL-FALLBACK / DIRTY" ;;
  *) echo "  FAIL: project gate never reached DIRTY"; exit 1 ;; esac ;;
  *) echo "  FAIL: project gate never reached PARTIAL-FALLBACK"; exit 1 ;; esac ;;
  *) echo "  FAIL: project gate never reached CLEAN"; exit 1 ;;
esac
