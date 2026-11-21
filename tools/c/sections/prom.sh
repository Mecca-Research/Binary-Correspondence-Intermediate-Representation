#!/usr/bin/env bash
# prom: the telemetry export's Prometheus exposition (bcir_prom.c, N13): test_prom's contracts, and its text ==
# the oracle's octet for octet (bcir/telemetry_export.py's to_prometheus)
#
# One section of tools/c/check_runtime.sh, run against a binary built elsewhere: BCIR Make builds it from
# runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake project builds it
# from the same manifest and runs this script as the `c-section-prom` CTest entry, and tools/build/section_parity.py
# holds the two builds' outputs byte-identical. The comparison with the oracle is the Python test's own
# (bcir/tests/test_c_prom.py), run here against the built harness.
#
#   usage: prom.sh <test_prom>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_prom>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }

out="$("${HARNESS}" 2>&1)" || { echo "  FAIL: Prometheus exposition contracts"; echo "${out}"; exit 1; }
[ "${out}" = "test_prom: OK" ] || { echo "  FAIL: unexpected Prometheus harness output"; echo "${out}"; exit 1; }
echo "  PASS Prometheus exposition: a short output, names out of order or not UTF-8, a reading without its definition, every NULL, a known text"
out="$("${PYTHON}" - "${HARNESS}" <<'PY' 2>&1
import sys
from bcir.tests.test_c_prom import check_driver
seen = check_driver(sys.argv[1])
print(f"{seen['snapshots']} snapshots, {seen['series']} series ({seen['absent']} not reading), {seen['witnesses']} witnesses, "
      f"{seen['octets']} octets")
PY
)" || { echo "  FAIL: test_prom != the oracle"; echo "${out}" | tail -20; exit 1; }
echo "  PASS test_prom == the oracle: ${out}"
