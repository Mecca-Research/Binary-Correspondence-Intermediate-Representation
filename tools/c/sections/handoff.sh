#!/usr/bin/env bash
# handoff: data-plane hand-off (G16): freestanding pack table + shard manifest + per-step freeze + oracle/C traces
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: the gate
# compiles them (the harness, its optimisation variants and the fault-injected mutant, from the
# real source) and calls this script; the CMake project builds the same binaries from
# runtime/manifest.json (`harnesses` and `variants`) and runs this script as the
# `c-section-handoff` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds
# the two builds' outputs byte-identical.
#
#   usage: handoff.sh <test_handoff> <test_handoff_O0> <test_handoff_O3> <test_handoff_mutant>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 4 ] || { echo "usage: $(basename "$0") <test_handoff> <test_handoff_O0> <test_handoff_O3> <test_handoff_mutant>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
O0="$(abs "$2")"; [ -x "${O0}" ] || { echo "  FAIL: O0: ${O0} is not an executable harness"; exit 2; }
O3="$(abs "$3")"; [ -x "${O3}" ] || { echo "  FAIL: O3: ${O3} is not an executable harness"; exit 2; }
MUTANT="$(abs "$4")"; [ -x "${MUTANT}" ] || { echo "  FAIL: MUTANT: ${MUTANT} is not an executable harness"; exit 2; }
declare -A VARIANT=([o0]="${O0}" [o3]="${O3}")
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# bcir_handoff.h is the C twin of bcir/gem/handoff.py (a pack table whose handles carry an epoch:
# a view that outlives its owner is refused; admission is the live control plane's predicate and
# dispatch runs only what was admitted at the resident generation, as a phase of the plane),
# bcir_shard_manifest.h of bcir/abi/shard_manifest.py (BSHM version zero: runnable shards and a
# frame named by digest, reassembling to the whole pack's bytes) and bcir_hydrate_generations of
# handoff.freeze_claims (the dynamic-graph builder's per-step freeze, bound to the live registry).
# The same harness runs the Stage 3 exit flow (test_stage3.h): one generation through the control
# ring, the plane, the pack table, the telemetry ring and the intake, and the old generation offered
# at every boundary after the switch -- which is why the rings and the envelope codec link here.
# The grading is tools/c/check_handoff.py -> bcir/tests/handoff_fixtures.py::measure, the function
# the tests, the G16 harness rows, the C++ gate (tools/cpp/check_handoff.sh) and the fault table
# (tools/testing/faults/handoff.json) use; this section grades the oracle and the C twin.
handoff_api="$("${HARNESS}" --api)" \
  || { echo "  FAIL: hand-off API laws"; echo "${handoff_api}"; exit 1; }
case "${handoff_api}" in
  OK\ *) ;;
  *) echo "  FAIL: unexpected hand-off API output"; echo "${handoff_api}"; exit 1 ;;
esac
handoff_measure() {  # <C harness> -> prints the row summary; exits as tools/c/check_handoff.py does
  "${PYTHON}" "${ROOT}/tools/c/check_handoff.py" --exe "$1" --no-cpp --tmp "${tmp}" \
    > "${tmp}/handoff_rows.txt" 2>&1
  local status=$?
  tail -n 1 "${tmp}/handoff_rows.txt"
  return "${status}"
}
handoff_rows="$(handoff_measure "${HARNESS}")" \
  || { echo "  FAIL: a G16 row is not zero: ${handoff_rows}"; cat "${tmp}/handoff_rows.txt"; exit 1; }
echo "  PASS data-plane hand-off (freestanding C11 + C23; API fail-closed laws, ${handoff_api#OK } checks; ${handoff_rows#rows })"
# -O0 == -O3 == the oracle: the twin's traces, freeze bytes, split bytes and refusals do not depend
# on the optimizer (a fast path that reads an uninitialized byte would diverge here first).
for opt in o0 o3; do
  opt_rows="$(handoff_measure "${VARIANT[${opt}]}")" \
    || { echo "  FAIL: the -${opt^^} hand-off harness diverges from the oracle: ${opt_rows}"; exit 1; }
done
echo "  PASS data-plane hand-off optimisation parity (-O0 == -O3 == the oracle)"
# The gate must be able to fail (L2): a table that dispatches a pack admitted in a generation the
# plane has left -- the admission's generation no longer re-checked at dispatch -- must turn a row
# red. Built from the real source; if the law's line changes, the injection fails loudly.
mutant_rows="$(handoff_measure "${MUTANT}")"; mutant_status=$?
if [ "${mutant_status}" -ne 1 ]; then
  echo "  FAIL: a table that dispatches across a generation switch passed the G16 gate (exit ${mutant_status}): ${mutant_rows}"; exit 1
fi
echo "  PASS hand-off gate fires on an injected fault (dispatch-generation law removed: ${mutant_rows#rows })"
