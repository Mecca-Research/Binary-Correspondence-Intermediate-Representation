#!/usr/bin/env bash
# control_plane: ControlRecordV1 (G14): freestanding plane + Python->C->Python round trip + declared refusal statuses + identical two-rail plane traces
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them (the harness, its optimisation variants and the fault-injected mutant, from the real
# source) and runs this script as a task, whose verdict the gate shows; the CMake project builds the
# same binaries from the same manifest (`harnesses` and `variants`) and runs this script as the
# `c-section-control_plane` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical.
#
#   usage: control_plane.sh <test_control_plane> <test_control_plane_mutant>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_control_plane> <test_control_plane_mutant>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
MUTANT="$(abs "$2")"; [ -x "${MUTANT}" ] || { echo "  FAIL: MUTANT: ${MUTANT} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# bcir_control_plane.h is the C twin of bcir/abi/control_abi.py (the wire laws, in the same order)
# and bcir/gem/control.py (the resident plane): lease, generation, quiesce, activate, rollback and
# cancel, decided by their bytes. It links the shared SHA-256 / HMAC-SHA256 (bcir_sha256.c) and the
# StreamPack / plan verifiers (bcir_runtime.c). The grading is bcir/tests/control_fixtures.py::
# measure -- the function the tests and the G14 harness rows use: every corpus record round-trips
# byte for byte with its MAC accepted, every malformed variant is refused with its declared status
# on both rails, every scenario decides as specified on both rails, and the two rails' traces
# (verdicts, refusals, statuses and the resident state digest after every operation) are identical.
ctl_api="$("${HARNESS}" --api)" || { echo "  FAIL: control API laws"; echo "${ctl_api}"; exit 1; }
[ "${ctl_api}" = "OK" ] || { echo "  FAIL: unexpected control API output"; echo "${ctl_api}"; exit 1; }
ctl_measure() {  # <harness> -> prints the six rows; exit 0 only when every row is zero
  "${PYTHON}" - "$1" "${tmp}" <<'PY'
import sys
from bcir.tests.control_fixtures import ROWS, measure
rows = measure(sys.argv[1], sys.argv[2])
if set(rows) != set(ROWS):
    print(f"rows {sorted(rows)} != {sorted(ROWS)}")
    sys.exit(2)
print(" ".join(f"{key.removeprefix('control.')}={int(value)}" for key, value in rows.items()))
sys.exit(1 if any(rows.values()) else 0)
PY
}
ctl_rows="$(ctl_measure "${HARNESS}")" \
  || { echo "  FAIL: a G14 row is not zero: ${ctl_rows}"; exit 1; }
echo "  PASS ControlRecordV1 (freestanding C11 + C23; API fail-closed laws; ${ctl_rows})"
# The gate must be able to fail (L2): a plane that applies a switch mid-phase -- the deferral law
# removed -- must turn a row red. The mutant is built from the real source; if the law's line
# ever changes, the injection fails loudly instead of grading an unmutated copy.
if mutant_rows="$(ctl_measure "${MUTANT}")"; then
  echo "  FAIL: a plane that applies a switch mid-phase passed the G14 gate: ${mutant_rows}"; exit 1
fi
echo "  PASS ControlRecordV1 gate fires on an injected fault (deferral law removed: ${mutant_rows})"
