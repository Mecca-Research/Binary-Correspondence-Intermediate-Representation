#!/usr/bin/env bash
# measure_jer_simd.sh -- J5's advantage clause, measured on ONE declared host.
#
# J5's gate asks for a "statistically significant measured advantage on at least two hosts".
# check_jer_simd.sh settles the two clauses decidable on one machine (identical status and
# offset at every tier; no unsupported-CPU fault) and deliberately does NOT check this one:
# 8 refuses timing thresholds on shared runners, so the clause needs a DECLARED target and a
# controlled run. This script is how one such host produces its record.
#
# It prints a JSON object to stdout. Paste it into docs/measurements/jer_simd_hosts.json and
# `python3 -m bcir.asn1.simd_hosts` (or the test) decides whether it is admissible. The
# script itself makes no claim -- it measures and declares, and admissibility is a separate
# judgement made against 8's rules rather than by the machine describing itself.
#
# CORRECTNESS FIRST, ALWAYS. A timing from a build that disagrees with the scalar rail is
# worse than no timing: it is a number attached to the wrong answer. The gate runs first and
# a failure aborts before a single round is measured.
#
# RUNNING IT ON A PHONE, which is the first dedicated aarch64 host this project has:
#
#   pkg install clang python git      # Termux
#   git clone <repo> && cd <repo>
#   tools/silicon/measure_jer_simd.sh --pin 7 --host "Samsung S24+ (SM-S926B), SD 8 Gen 3"
#
# --pin matters more on a phone than anywhere else. A Snapdragon 8 Gen 3 is big.LITTLE: one
# Cortex-X4, four A720s, three A520s. The same code on the largest and the smallest core
# differs by more than the SIMD advantage being measured, so a run the scheduler moves
# between clusters is two machines averaged. The driver reports which CPU each round ran on
# and the reader REFUSES a record whose rounds span more than one -- so an unpinned run does
# not silently become evidence. On this SoC the big core is usually cpu7; `--pin 7` asks for
# it and the recorded `cpus` field says whether the kernel agreed.
#
# Thermal throttling needs no flag. The rounds are INTERLEAVED -- scalar, vector, scalar,
# vector -- so a downward frequency ramp is spread evenly across both candidates and becomes
# noise in each rather than a bias in whichever ran last. That is the same argument
# bcir_asn1_bench.c makes for its own round-robin, and it is why this script does not simply
# run all the scalar rounds and then all the vector ones.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CPP="${ROOT}/runtime/cpp"

HOST=""
PIN=""
ROUNDS=41
ITERATIONS=64
TENANCY="dedicated"
NOTES=""

usage() {
  cat <<'USAGE'
usage: measure_jer_simd.sh [options]

  --host NAME      a name a reader can look up ("Samsung S24+ (SM-S926B), SD 8 Gen 3").
                   Required: "linux" is not a host, and the record is a claim about a
                   specific machine.
  --pin N          run every round on CPU N (sched_setaffinity). Strongly recommended
                   on big.LITTLE, where the largest and smallest core differ by more
                   than the advantage being measured.
  --rounds N       measured rounds per candidate (default 41; the reader needs at least the
                   minimum an order-statistic interval covers a median with).
  --iterations N   iterations per round; the round's figure is their MEDIAN (default 64).
  --tenancy WHICH  dedicated | shared (default dedicated). Say `shared` for a cloud runner
                   or a laptop doing other work: the reader refuses those, which is the
                   point rather than an inconvenience.
  --notes TEXT     thermal state, power source, anything a reader would want.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --pin) PIN="${2:-}"; shift 2 ;;
    --rounds) ROUNDS="${2:-}"; shift 2 ;;
    --iterations) ITERATIONS="${2:-}"; shift 2 ;;
    --tenancy) TENANCY="${2:-}"; shift 2 ;;
    --notes) NOTES="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${HOST}" ]; then
  echo "measure_jer_simd.sh: --host is required (see --help)" >&2
  exit 2
fi

CXX="${CXX:-}"
if [ -z "${CXX}" ]; then
  for candidate in clang++ g++ c++; do
    if command -v "${candidate}" >/dev/null 2>&1; then CXX="${candidate}"; break; fi
  done
fi
CC="${CC:-}"
if [ -z "${CC}" ]; then
  for candidate in clang gcc cc; do
    if command -v "${candidate}" >/dev/null 2>&1; then CC="${candidate}"; break; fi
  done
fi
if [ -z "${CXX}" ] || [ -z "${CC}" ]; then
  echo "measure_jer_simd.sh: no C++/C compiler pair found" >&2
  exit 2
fi

# Echo a file to stderr with a prefix. A shell loop rather than `sed`, because this is
# cosmetic and `set -e` would otherwise abort the whole measurement over a missing package.
# Nothing here should fail for a reason unrelated to what is being measured.
indent() {
  local prefix="$1" line
  while IFS= read -r line; do printf '%s%s\n' "${prefix}" "${line}" >&2; done <"$2"
}

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

echo "[measure] correctness first: the SIMD gate must pass before anything is timed" >&2
if ! CXX="${CXX}" CC="${CC}" bash "${ROOT}/tools/cpp/check_jer_simd.sh" >"${tmp}/gate.txt" 2>&1; then
  echo "[measure] REFUSED: the SIMD gate failed on this host. A timing from a build that" >&2
  echo "          disagrees with the scalar rail is a number attached to the wrong answer." >&2
  indent "          " "${tmp}/gate.txt"
  exit 1
fi
indent "  " "${tmp}/gate.txt"

echo "[measure] building the bench driver" >&2
for source in "${CPP}/bcir_jer_simd.cpp" "${CPP}/test_jer_simd.cpp"; do
  "${CXX}" -std=c++17 -O2 -I "${C}" -I "${CPP}" -c "${source}" \
    -o "${tmp}/$(basename "${source}").o"
done
for source in "${C}/bcir_jer.c" "${C}/bcir_runtime.c"; do
  "${CC}" -std=c11 -O2 -I "${C}" -c "${source}" -o "${tmp}/$(basename "${source}").o"
done
"${CXX}" "${tmp}"/*.o -o "${tmp}/bench"

# Pinning goes through python's `os.sched_setaffinity` rather than `taskset`. python3 is
# already required here, while taskset lives in util-linux -- a separate package on Termux,
# and the phone is the host this most needs to work on. `os.execv` replaces the interpreter
# so nothing sits between the affinity call and the measurement.
PIN_HELPER="${tmp}/pin.py"
cat >"${PIN_HELPER}" <<'PYPIN'
import os
import sys

cpu, program = int(sys.argv[1]), sys.argv[2]
os.sched_setaffinity(0, {cpu})
os.execv(program, [program])
PYPIN

RUN=("${tmp}/bench")
PIN_STATE="not requested"
if [ -n "${PIN}" ]; then
  if python3 -c 'import os; os.sched_setaffinity(0, os.sched_getaffinity(0))' 2>/dev/null; then
    RUN=(python3 "${PIN_HELPER}" "${PIN}" "${tmp}/bench")
    PIN_STATE="sched_setaffinity cpu ${PIN}"
  else
    # Recorded rather than fatal. The reader decides on the OBSERVED CPUs, so a kernel that
    # refuses affinity still produces an honest record -- it simply will not be admissible
    # if the rounds then wandered, which is the correct outcome rather than a lost run.
    PIN_STATE="requested cpu ${PIN}, but this platform has no sched_setaffinity"
    echo "[measure] warning: ${PIN_STATE}" >&2
  fi
fi

TIERS="$(printf 'tiers\n' | "${RUN[@]}")"
# `tiers <available> <name> <compiled>`; read splits it without needing awk, which Termux
# does not carry in its base install.
read -r _kw _available TIER_NAME _compiled <<<"${TIERS}"
echo "[measure] resolved tier: ${TIER_NAME} (${TIERS})" >&2

# The document is the one 7.3 measures: an all-ASCII claim graph, which is the shape a JER
# control plane actually carries and the shape the vector rail targets. Generated here so
# the two rails see identical octets -- a benchmark that regenerates its input times the
# generator.
DOC_HEX="$(python3 - "$ROOT" <<'PYDOC'
import sys
nodes = 400
body = b",".join(b'{"kind":"claim","label":"%d","attributes":['
                 b'{"name":"op","value":"add"}]}' % i for i in range(nodes))
print((b'{"version":1,"nodes":[' + body + b'],"roots":[0]}').hex())
PYDOC
)"

echo "[measure] ${ROUNDS} interleaved rounds x ${ITERATIONS} iterations per candidate" >&2
{
  round=0
  while [ "${round}" -lt "${ROUNDS}" ]; do
    printf 'bench scalar 1 %s %s\n' "${ITERATIONS}" "${DOC_HEX}"
    printf 'bench auto 1 %s %s\n' "${ITERATIONS}" "${DOC_HEX}"
    round=$((round + 1))
  done
} | "${RUN[@]}" > "${tmp}/samples.txt"

python3 - "${tmp}/samples.txt" "${HOST}" "${TIER_NAME}" "${TENANCY}" "${PIN_STATE}" "${NOTES}" <<'PYREPORT'
import json
import platform
import sys

path, host, tier, tenancy, pin_state, notes = sys.argv[1:7]
scalar, vector, cpus = [], [], set()
with open(path, encoding="utf-8") as handle:
    for line in handle:
        parts = line.split()
        if len(parts) < 4 or parts[0] != "sample":
            continue
        # `sample <tier> <round> <ns> <cpu>`; the cpu field is absent on a driver older than
        # the one that added it, and absent must read as UNKNOWN rather than as clean.
        value = int(parts[3])
        cpus.add(int(parts[4]) if len(parts) > 4 else -1)
        (scalar if parts[1] == "scalar" else vector).append(value)

record = {
    "host": host,
    "arch": platform.machine(),
    "tenancy": tenancy,
    "tier": tier,
    "scalar_ns": scalar,
    "vector_ns": vector,
    "cpus": sorted(cpus),
    "notes": "; ".join(part for part in (f"pin: {pin_state}", platform.platform(), notes)
                       if part),
}
print(json.dumps(record, indent=2))
PYREPORT
