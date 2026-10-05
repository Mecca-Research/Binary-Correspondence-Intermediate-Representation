#!/usr/bin/env bash
# ring: live SPSC ring + TelemetryEnvelopeV0 (G15): freestanding ring/envelope + generated signal table + two-rail scenario traces + concurrency under ThreadSanitizer
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: the gate
# compiles them (the harness, its optimisation variants and the fault-injected mutant, from the
# real source) and calls this script; the CMake project builds the same binaries from
# runtime/manifest.json (`harnesses` and `variants`) and runs this script as the
# `c-section-ring` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds
# the two builds' outputs byte-identical.
#
#   usage: ring.sh <test_ring> <test_ring_O0> <test_ring_O3> <test_ring_torn>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 4 ] || { echo "usage: $(basename "$0") <test_ring> <test_ring_O0> <test_ring_O3> <test_ring_torn>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
O0="$(abs "$2")"; [ -x "${O0}" ] || { echo "  FAIL: O0: ${O0} is not an executable harness"; exit 2; }
O3="$(abs "$3")"; [ -x "${O3}" ] || { echo "  FAIL: O3: ${O3} is not an executable harness"; exit 2; }
MUTANT="$(abs "$4")"; [ -x "${MUTANT}" ] || { echo "  FAIL: MUTANT: ${MUTANT} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# bcir_ring.h is the C twin of bcir/gem/ring.py (the version-zero live ring: acquire/release
# publication, a per-slot seqlock, backpressure or overwrite, exact loss accounting, epochs and
# takeover) and bcir_telemetry_envelope.h of bcir/abi/telemetry_envelope.py + bcir/telemetry_intake.py
# (the identity-carrying record and the host intake). The grading is
# bcir/tests/ring_fixtures.py::measure -- the function the tests and the G15 harness rows use:
# every scripted scenario decides as specified on both rails with identical traces (the region's
# CRC after every operation included), the accounting is exact, nothing torn is delivered, the
# continuity report is the declared one, every malformed or stale fixture is refused as declared,
# the 52 G14 control scenarios decide identically through a live control ring, and the concurrent
# runs (threads; processes with a peer SIGKILLed and taken over) end with nothing torn or
# unaccounted.
"${PYTHON}" -m bcir.signal_table --check >/dev/null \
  || { echo "  FAIL: runtime/c/bcir_signal_table.h differs from its generator (python -m bcir.signal_table --emit)"; exit 1; }
ring_api="$("${HARNESS}" --api)" || { echo "  FAIL: ring API laws"; echo "${ring_api}"; exit 1; }
[ "${ring_api}" = "OK" ] || { echo "  FAIL: unexpected ring API output"; echo "${ring_api}"; exit 1; }
ring_measure() {  # <harness> -> prints the row summary; exits as tools/c/check_ring.py does
  # check_ring.py is the one grading entry point (tools/testing/faults/ring.json runs it too):
  # 0 every row zero, 1 a row fired, 2 UNAVAILABLE. Its findings stay in ${tmp}/ring_rows.txt.
  "${PYTHON}" "${ROOT}/tools/c/check_ring.py" --exe "$1" --tmp "${tmp}" > "${tmp}/ring_rows.txt" 2>&1
  local status=$?
  tail -1 "${tmp}/ring_rows.txt" | sed 's/^rows: //'
  return "${status}"
}
ring_rows="$(ring_measure "${HARNESS}")" \
  || { echo "  FAIL: a G15 row is not zero: ${ring_rows}"; grep '^  - ' "${tmp}/ring_rows.txt"; exit 1; }
echo "  PASS live ring (freestanding C11 + C23; generated signal table current; API fail-closed laws; ${ring_rows})"
# The twin's answers must not depend on the optimiser, the discipline every C twin here gets: the
# reader runs u64 position arithmetic over offsets a hostile region chose, and undefined behaviour
# there would entitle -O3 to decide differently from -O0 exactly where a malformed region lives.
ring_opt="$("${PYTHON}" - "${O0}" "${O3}" "${tmp}" <<'PY'
import sys
from bcir.tests import ring_fixtures as rf
o0, o3, tmp = sys.argv[1:]
scenarios = rf.all_scenarios()
blobs = [data for _, data in rf.envelope_corpus()] + [data for _, data, _ in rf.malformed_envelopes()]
answers = [(rf.run_c(exe, tmp, scenarios), rf.c_envelopes(exe, tmp, blobs), rf.c_signals(exe))
           for exe in (o0, o3)]
oracle = [rf.run_python(scenario, index) for index, scenario in enumerate(scenarios)]
if answers[0] != answers[1] or answers[0][0] != oracle:
    sys.exit(1)
print(f"{len(scenarios)} scenarios, {len(blobs)} envelopes")
PY
)" || { echo "  FAIL: the ring twin's answers depend on the optimisation level (or leave the oracle's)"; exit 1; }
echo "  PASS live ring optimisation parity (-O0 == -O3 == the oracle over ${ring_opt})"
# The rows must be able to fail (L2): a consumer that trusts a slot it copied while the producer
# rewrote it (the seqlock re-check removed) must turn rows red. The mutant is built from the real
# source; if the anchor ever changes, the injection fails loudly instead of grading a copy.
torn_rows="$(ring_measure "${MUTANT}")"; torn_status=$?
if [ "${torn_status}" -eq 0 ]; then
  echo "  FAIL: a ring that delivers what a writer tore passed the G15 gate: ${torn_rows}"; exit 1
elif [ "${torn_status}" -ne 1 ]; then  # UNAVAILABLE is not a catch (L1)
  echo "  FAIL: the mutant could not be graded (exit ${torn_status}): ${torn_rows}"; exit 1
fi
echo "  PASS live-ring gate fires on an injected fault (seqlock re-check removed: ${torn_rows})"
