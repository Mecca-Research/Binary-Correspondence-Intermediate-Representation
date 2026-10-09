#!/usr/bin/env bash
# kplan: native K_BCIR planner (G17): freestanding compile (C11 + C23) + byte parity with the oracle
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them (the harness, its optimisation variants and the fault-injected mutant, from the real
# source) and runs this script as a task, whose verdict the gate shows; the CMake project builds the
# same binaries from the same manifest (`harnesses` and `variants`) and runs this script as the
# `c-section-kplan` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical.
#
#   usage: kplan.sh <test_kplan> <test_kplan_O0> <test_kplan_O3> <test_kplan_mutant> <test_kplan_hydrate_mutant>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 5 ] || { echo "usage: $(basename "$0") <test_kplan> <test_kplan_O0> <test_kplan_O3> <test_kplan_mutant> <test_kplan_hydrate_mutant>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
O0="$(abs "$2")"; [ -x "${O0}" ] || { echo "  FAIL: O0: ${O0} is not an executable harness"; exit 2; }
O3="$(abs "$3")"; [ -x "${O3}" ] || { echo "  FAIL: O3: ${O3} is not an executable harness"; exit 2; }
MUTANT="$(abs "$4")"; [ -x "${MUTANT}" ] || { echo "  FAIL: MUTANT: ${MUTANT} is not an executable harness"; exit 2; }
HMUTANT="$(abs "$5")"; [ -x "${HMUTANT}" ] || { echo "  FAIL: HMUTANT: ${HMUTANT} is not an executable harness"; exit 2; }
declare -A VARIANT=([O0]="${O0}" [O3]="${O3}")
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# bcir_kplan.c is the C twin of bcir/kbcir/realize.py's planner (the compact offer and the min-plus
# path) and of bcir/abi/planner_abi.py (the BKPI input and BKPR realization records, version zero).
# The grading is tools/c/check_planner.py -> bcir/tests/planner_fixtures.py::measure, the function the
# tests, the G17 harness rows and the fault table (tools/testing/faults/planner.json) use: the compact
# planner held to the pre-G17 reference (bcir/kbcir/realize_reference.py) and this planner held to
# both, byte for byte, over the fixed corpus under every target, Theta and policy and a seeded
# generated corpus; one malformed record per wire and planning law refused with its declared status on
# both rails.
"${PYTHON}" - "${ROOT}" "${tmp}/kplan_seed.bkpi" <<'PY' || { echo "  FAIL: the planner's seed record"; exit 1; }
import sys
sys.path.insert(0, sys.argv[1])
from bcir.tests.planner_fixtures import _seed_input
with open(sys.argv[2], "wb") as f:
    f.write(_seed_input()[0])
PY
kplan_api="$("${HARNESS}" --api "${tmp}/kplan_seed.bkpi")" \
  || { echo "  FAIL: planner API laws"; echo "${kplan_api}"; exit 1; }
case "${kplan_api}" in
  API\ OK\ *) ;;
  *) echo "  FAIL: unexpected planner API output"; echo "${kplan_api}"; exit 1 ;;
esac
kplan_measure() {  # <C harness> -> prints the row summary; exits as tools/c/check_planner.py does
  "${PYTHON}" "${ROOT}/tools/c/check_planner.py" --exe "$1" --tmp "${tmp}" > "${tmp}/kplan_rows.txt" 2>&1
  local status=$?
  tail -n 1 "${tmp}/kplan_rows.txt"
  return "${status}"
}
kplan_rows="$(kplan_measure "${HARNESS}")" \
  || { echo "  FAIL: a G17 row is not zero: ${kplan_rows}"; cat "${tmp}/kplan_rows.txt"; exit 1; }
echo "  PASS native planner (freestanding C11 + C23; API fail-closed laws, ${kplan_api#API OK } checks; ${kplan_rows#rows })"
# -O0 == -O3 == the oracle: the plans and refusals do not depend on the optimizer.
for opt in O0 O3; do
  opt_rows="$(kplan_measure "${VARIANT[${opt}]}")" \
    || { echo "  FAIL: the -${opt} planner diverges from the oracle: ${opt_rows}"; exit 1; }
done
echo "  PASS native planner optimisation parity (-O0 == -O3 == the oracle)"
# The gate must be able to fail (L2): a planner whose 128-bit addition drops its carry wraps a path
# weight past 2**64 (bcir/tests/planner_fixtures.py::wide_path_case) and must turn a row red. Built
# from the real source; if the line changes, the injection fails loudly.
mutant_rows="$(kplan_measure "${MUTANT}")"; mutant_status=$?
if [ "${mutant_status}" -ne 1 ]; then
  echo "  FAIL: a planner that wraps a path weight passed the G17 gate (exit ${mutant_status}): ${mutant_rows}"; exit 1
fi
echo "  PASS planner gate fires on an injected fault (the 128-bit carry dropped: ${mutant_rows#rows })"

# The native hydrate (CXX4): BKPI + BKPR + BKPB -> the oracle's StreamPack, byte for byte. The grading
# is tools/c/check_hydrate.py -> bcir/tests/hydrate_fixtures.py::measure, the function the tests, the
# CXX4 harness rows and the fault table (tools/testing/faults/hydrate.json) use: every corpus case's
# native pack against encode(hydrate(...)), and one malformed record per law refused alike on both rails.
hydrate_api="$("${HARNESS}" --api-hydrate "${tmp}/kplan_seed.bkpi")" \
  || { echo "  FAIL: hydrate API laws"; echo "${hydrate_api}"; exit 1; }
case "${hydrate_api}" in
  API\ OK\ *) ;;
  *) echo "  FAIL: unexpected hydrate API output"; echo "${hydrate_api}"; exit 1 ;;
esac
hydrate_measure() {  # <C harness> -> prints the row summary; exits as tools/c/check_hydrate.py does
  "${PYTHON}" "${ROOT}/tools/c/check_hydrate.py" --exe "$1" --tmp "${tmp}" > "${tmp}/khydrate_rows.txt" 2>&1
  local status=$?
  tail -n 1 "${tmp}/khydrate_rows.txt"
  return "${status}"
}
hydrate_rows="$(hydrate_measure "${HARNESS}")" \
  || { echo "  FAIL: a CXX4 row is not zero: ${hydrate_rows}"; cat "${tmp}/khydrate_rows.txt"; exit 1; }
echo "  PASS native hydrate (API fail-closed laws, ${hydrate_api#API OK } checks; ${hydrate_rows#rows })"
for opt in O0 O3; do
  opt_rows="$(hydrate_measure "${VARIANT[${opt}]}")" \
    || { echo "  FAIL: the -${opt} hydrate diverges from the oracle: ${opt_rows}"; exit 1; }
done
echo "  PASS native hydrate optimisation parity (-O0 == -O3 == the oracle)"
# The gate must be able to fail (L2): a hydrate that writes every pack as v4 -- a header pad and a
# generation count where a pack without a vector has neither -- must turn the parity row red.
hmutant_rows="$(hydrate_measure "${HMUTANT}")"; hmutant_status=$?
if [ "${hmutant_status}" -ne 1 ]; then
  echo "  FAIL: a hydrate that writes v1 packs as v4 passed the CXX4 gate (exit ${hmutant_status}): ${hmutant_rows}"; exit 1
fi
echo "  PASS hydrate gate fires on an injected fault (a v1 pack written as v4: ${hmutant_rows#rows })"
