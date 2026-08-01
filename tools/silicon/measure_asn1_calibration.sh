#!/usr/bin/env bash
# measure_asn1_calibration.sh -- J6's target calibration, measured on ONE declared target.
#
# `certified.py` will only let the planner use a timing from a frozen, generation-tagged
# table measured on a real target. `native_bench.py` can produce such a table, and
# `measured_table(target=...)` takes the target on TRUST -- a string. This script is how a
# target earns one honestly: it measures, it declares, and it records the machine's own
# accounting of whether the declaration held. Admissibility is decided separately, by
# `bcir/asn1/calibration.py`, against the same rules `simd_hosts.py` applies to SIMD records.
#
# It prints a JSON object to stdout. Paste it into the "targets" list of
# docs/measurements/asn1_calibration.json and `python3 -m bcir.asn1.calibration` (or the
# test) decides whether it is admissible. The script makes no admissibility claim itself --
# a machine describing itself is exactly the evidence that needs checking elsewhere.
#
# CORRECTNESS FIRST, ALWAYS. A timing from a build whose encoders disagree with the Python
# oracle is worse than no timing: it is a number attached to the wrong octets, and it would
# then be frozen and trusted. The parity gates run first and a failure aborts before a single
# round is measured.
#
# RUNNING IT ON THE PHONE, which is the dedicated aarch64 target this phase needs:
#
#   pkg install clang python git       # Termux
#   git clone <repo> && cd <repo>
#   tools/silicon/measure_asn1_calibration.sh \
#       --target "Samsung S24+ (SM-S926B), Snapdragon 8 Gen 3" --pin 7
#
# --pin matters more here than on any other host. A Snapdragon 8 Gen 3 is big.LITTLE: one
# Cortex-X4, four A720s, three A520s. A calibration averaged across a prime core and an
# efficiency core describes no core that exists -- and unlike a SIMD record, which supports
# one sentence in a document, a cost table steers production selection. The reader REFUSES a
# record whose rounds span more than one CPU, so an unpinned run cannot silently become the
# table the planner trusts. On this SoC the big core is usually cpu7; `--pin 7` asks and the
# recorded `cpus` field says whether the kernel agreed.
#
# THE CORPUS IS NOT A FLAG. Every target measures `calibration_corpus()`, and the record
# carries `corpus_digest()`. Two targets that measured different schemas produce two tables
# that look identical, compare cleanly and mean nothing -- a silent failure, so the corpus
# lives in the repository and the digest is checked rather than trusted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TARGET=""
PIN=""
ROUNDS=41
ITERATIONS=64
TENANCY="dedicated"
CAL_GEN=1
NOTES=""
SKIP_GATE=0

usage() {
  cat <<'USAGE'
usage: measure_asn1_calibration.sh --target NAME [options]

  --target NAME    a name a reader can look up ("Samsung S24+ (SM-S926B), SD 8 Gen 3").
                   Required: "linux" is not a target, and the record is a claim about a
                   specific machine that a certificate will later name.
  --pin N          run every round on CPU N (sched_setaffinity). Strongly recommended on
                   big.LITTLE, where a mixed run describes no core that exists.
  --rounds N       measured rounds per candidate (default 41).
  --iterations N   iterations per round; the round's figure is their median (default 64).
  --tenancy WHICH  dedicated | shared (default dedicated). Say `shared` for a cloud runner
                   or a laptop doing other work: the reader refuses those, which is the
                   point rather than an inconvenience.
  --cal-gen N      calibration generation (default 1). Bump it when the target changes in a
                   way a frozen table should not silently survive.
  --notes TEXT     thermal state, power source, anything a reader would want.
  --skip-gate      skip the parity gates. For debugging only; a record measured this way is
                   a timing whose correctness nobody checked.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --pin) PIN="${2:-}"; shift 2 ;;
    --rounds) ROUNDS="${2:-}"; shift 2 ;;
    --iterations) ITERATIONS="${2:-}"; shift 2 ;;
    --tenancy) TENANCY="${2:-}"; shift 2 ;;
    --cal-gen) CAL_GEN="${2:-}"; shift 2 ;;
    --notes) NOTES="${2:-}"; shift 2 ;;
    --skip-gate) SKIP_GATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${TARGET}" ]; then
  echo "measure_asn1_calibration.sh: --target is required (see --help)" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "measure_asn1_calibration.sh: python3 is required" >&2
  exit 2
fi
if ! command -v clang >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 \
   && ! command -v cc >/dev/null 2>&1; then
  echo "measure_asn1_calibration.sh: no C compiler; the native harness cannot be built" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# Echo a file to stderr with a prefix. A shell loop rather than `sed`, which Termux does not
# carry in its base install.
indent() {
  local prefix="$1" line
  while IFS= read -r line; do printf '%s%s\n' "${prefix}" "${line}" >&2; done <"$2"
}

if [ "${SKIP_GATE}" -eq 0 ]; then
  echo "[calibrate] correctness first: the C encoders must agree with the Python oracle" >&2
  # These three are the gates that bear on what is about to be timed: the plan-driven C
  # emitter against the oracle's encoders, the emit rail's own conformance, and the harness
  # that will produce the numbers. The full suite is not run -- it is long on a phone, and
  # most of it is not about these octets.
  if ! ( cd "${ROOT}" && python3 - <<'PYGATE'
import sys
from bcir.tests import run_all

modules = ("bcir.tests.test_c_emit", "bcir.tests.test_asn1_emit",
           "bcir.tests.test_asn1_native_bench")
ok = True
for name in modules:
    module = __import__(name, fromlist=["x"])
    for attr in sorted(d for d in dir(module) if d.startswith("test_")):
        try:
            getattr(module, attr)()
        except Exception as error:                       # noqa: BLE001 - report and fail
            ok = False
            print(f"FAIL {name}.{attr}: {type(error).__name__}: {error}")
sys.exit(0 if ok else 1)
PYGATE
  ) >"${tmp}/gate.txt" 2>&1; then
    echo "[calibrate] REFUSED: a parity gate failed on this target. A timing from a build" >&2
    echo "            whose encoders disagree with the oracle is a number attached to the" >&2
    echo "            wrong octets -- and it would then be frozen and trusted." >&2
    indent "            " "${tmp}/gate.txt"
    exit 1
  fi
  echo "[calibrate] parity gates pass" >&2
else
  echo "[calibrate] WARNING: --skip-gate; this record's correctness is unchecked" >&2
fi

# Pinning goes through python's `os.sched_setaffinity` rather than `taskset`: python3 is
# already required, while taskset lives in util-linux, a separate package on Termux -- and
# the phone is the target this most needs to work on.
PIN_ARGS=""
PIN_STATE="not requested"
OBSERVED_CPUS="-1"
if [ -n "${PIN}" ]; then
  if python3 -c 'import os; os.sched_setaffinity(0, os.sched_getaffinity(0))' 2>/dev/null; then
    PIN_ARGS="${PIN}"
    PIN_STATE="sched_setaffinity cpu ${PIN}"
    OBSERVED_CPUS="${PIN}"
  else
    # Recorded rather than fatal. The reader decides on the OBSERVED CPUs, so a kernel that
    # refuses affinity still produces an honest record -- it simply will not be admissible,
    # which is the correct outcome rather than a lost run.
    PIN_STATE="requested cpu ${PIN}, but this platform has no sched_setaffinity"
    echo "[calibrate] warning: ${PIN_STATE}" >&2
  fi
fi

# Contention accounting, sampled ACROSS the measured rounds. `--tenancy dedicated` is a claim
# about the machine; these two counters are the machine's own record of whether it held.
read_contention() {
  local steal=0 throttled=0
  if [ -r /proc/stat ]; then
    while read -r name _u _n _s _i _w _q _sq st _rest; do
      [ "${name}" = "cpu" ] && { steal="${st:-0}"; break; }
    done </proc/stat
  fi
  for f in /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/cpu/cpu.stat; do
    if [ -r "${f}" ]; then
      while read -r key value; do
        case "${key}" in throttled_usec|throttled_time) throttled="${value}" ;; esac
      done <"${f}"
      break
    fi
  done
  printf '%s %s\n' "${steal}" "${throttled}"
}
read -r STEAL_BEFORE THROTTLED_BEFORE <<<"$(read_contention)"

echo "[calibrate] ${ROUNDS} rounds x ${ITERATIONS} iterations per candidate, both axes" >&2
MEASURE_ARGS=(--measure --target "${TARGET}" --tenancy "${TENANCY}" --cal-gen "${CAL_GEN}"
              --rounds "${ROUNDS}" --iterations "${ITERATIONS}" --cpus "${OBSERVED_CPUS}"
              --notes "pin: ${PIN_STATE}; ${NOTES}")

if [ -n "${PIN_ARGS}" ]; then
  cat >"${tmp}/pin.py" <<'PYPIN'
import os
import runpy
import sys

# The helper lives in a temporary directory, so `sys.path[0]` is that directory rather than
# the repository. The caller has already chdir'd to the root; put it on the path explicitly
# so `bcir` resolves the same way it would for `python3 -m`.
sys.path.insert(0, os.getcwd())
os.sched_setaffinity(0, {int(sys.argv[1])})
sys.argv = ["bcir.asn1.calibration", *sys.argv[2:]]
runpy.run_module("bcir.asn1.calibration", run_name="__main__")
PYPIN
  ( cd "${ROOT}" && python3 "${tmp}/pin.py" "${PIN_ARGS}" "${MEASURE_ARGS[@]}" ) \
    >"${tmp}/record.json"
else
  ( cd "${ROOT}" && python3 -m bcir.asn1.calibration "${MEASURE_ARGS[@]}" ) \
    >"${tmp}/record.json"
fi

read -r STEAL_AFTER THROTTLED_AFTER <<<"$(read_contention)"
STEAL_DELTA=$((STEAL_AFTER - STEAL_BEFORE))
THROTTLED_DELTA=$((THROTTLED_AFTER - THROTTLED_BEFORE))
echo "[calibrate] contention during the rounds: steal ${STEAL_DELTA} ticks, throttled ${THROTTLED_DELTA} us" >&2

# The counters are stamped into the record AFTER the rounds, because a delta is only knowable
# once they are over. Rewriting the two fields here keeps the measurement and the accounting
# in one artifact rather than asking whoever pastes it to remember two numbers.
python3 - "${tmp}/record.json" "${STEAL_DELTA}" "${THROTTLED_DELTA}" <<'PYSTAMP'
import json
import sys

path, steal, throttled = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8") as handle:
    body = "".join(line for line in handle if not line.startswith("#"))
record = json.loads(body)
record["steal_ticks"] = steal
record["throttled_usec"] = throttled
print(json.dumps(record, indent=2, sort_keys=True))
PYSTAMP

echo "[calibrate] paste the object above into the \"targets\" list of" >&2
echo "            docs/measurements/asn1_calibration.json, then run" >&2
echo "            python3 -m bcir.asn1.calibration   to see whether it is admissible" >&2
