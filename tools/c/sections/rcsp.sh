#!/usr/bin/env bash
# rcsp: budgeted planning (N14, bcir_kp_plan_budgeted in bcir_kplan.c): the budgeted planner's API laws, and its plans
# under budgets == the oracle's (bcir/kbcir/rcsp.py's optimize_constrained), with no cap == bcir_kp_plan's
#
# One section of tools/c/check_runtime.sh, run against a binary built elsewhere: BCIR Make builds it from
# runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake project builds it
# from the same manifest and runs this script as the `c-section-rcsp` CTest entry, and tools/build/section_parity.py
# holds the two builds' outputs byte-identical. The comparison with the oracle is the Python test's own
# (bcir/tests/test_c_rcsp.py), run here against the built harness.
#
#   usage: rcsp.sh <test_kplan>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_kplan>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

"${PYTHON}" - "${tmp}/seed.bkpi" <<'PY' || { echo "  FAIL: the planner's seed record"; exit 1; }
import sys
from bcir.tests.planner_fixtures import _seed_input
with open(sys.argv[1], "wb") as f:
    f.write(_seed_input()[0])
PY
api="$("${HARNESS}" --api-budget "${tmp}/seed.bkpi")" || { echo "  FAIL: budgeted planner API laws"; echo "${api}"; exit 1; }
case "${api}" in
  API\ OK\ *) ;;
  *) echo "  FAIL: unexpected budgeted planner API output"; echo "${api}"; exit 1 ;;
esac
echo "  PASS budgeted planner API laws (${api#API OK } checks): no cap is the unconstrained plan, every refusal zeroes the output"
out="$("${PYTHON}" - "${HARNESS}" <<'PY' 2>&1
import sys
from bcir.tests.test_c_rcsp import check_driver
seen = check_driver(sys.argv[1])
print(f"{seen['runs']} budgeted plans: {seen['planned']} planned ({seen['binding']} where the budget binds), "
      f"{seen['infeasible']} infeasible at the oracle's claim, {seen['unbounded']} unbounded == bcir_kp_plan, "
      f"{seen['labels']} labels kept as the oracle keeps them")
PY
)" || { echo "  FAIL: the budgeted planner != the oracle"; echo "${out}" | tail -20; exit 1; }
echo "  PASS budgeted planner == the oracle: ${out}"
