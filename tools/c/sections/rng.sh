#!/usr/bin/env bash
# rng: the counter-based RNG (Philox4x32-10): test_rng's known answers and contracts, and its text driver == the
# oracle's (bcir/frontends/models/rng.py)
#
# One section of tools/c/check_runtime.sh, run against a binary built elsewhere: BCIR Make builds it from
# runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake project builds it
# from the same manifest and runs this script as the `c-section-rng` CTest entry, and tools/build/section_parity.py
# holds the two builds' outputs byte-identical. The harness holds the published known answers and the streams'
# contracts; the comparison with the oracle is the Python test's own (bcir/tests/test_c_rng.py), run here against
# the built harness.
#
#   usage: rng.sh <test_rng>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_rng>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }

out="$("${HARNESS}" 2>&1)" || { echo "  FAIL: counter-based RNG contracts"; echo "${out}"; exit 1; }
[ "${out}" = "test_rng: OK" ] || { echo "  FAIL: unexpected RNG harness output"; echo "${out}"; exit 1; }
echo "  PASS counter-based RNG: the published Philox4x32-10 known answers, fills from every lane, bounded draws, permutations"
out="$("${PYTHON}" - "${HARNESS}" <<'PY' 2>&1
import sys
from bcir.tests.test_c_rng import check_driver
print(f"{check_driver(sys.argv[1])} requests")
PY
)" || { echo "  FAIL: test_rng != the oracle"; echo "${out}" | tail -20; exit 1; }
echo "  PASS test_rng == the oracle: ${out}"
