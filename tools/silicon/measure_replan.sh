#!/usr/bin/env bash
# Rig runbook: run the CT4 measured replan end to end and report the win.
#
# On a bare-metal rig with the capabilities docs/kernel/HARDWARE_VALIDATION.md names -- a hardware
# PMU (perf_event_paranoid <= 1), a cpufreq userspace governor, and RAPL energy exposed --
# this drives the real loop: read real PMU + RAPL + on-die thermal -> fold into Theta ->
# train + freeze a LinearCalibrator -> replan -> certify the win, and prints the MEASURED
# replan win (provenance=real). In a sandbox without those capabilities it degrades
# HONESTLY: it reports exactly which signals were unavailable, runs the same path on
# synthetic telemetry (provenance=synthetic, win 0), and exits 0 -- it never fabricates a
# measured number. The point: the rig path is push-button and CI-exercised in degrade mode,
# so it can never silently rot; the measured win lights up the instant the rig is present.
#
#   bash tools/silicon/measure_replan.sh                 # auto: real on a rig, synthetic in a sandbox
#   bash tools/silicon/measure_replan.sh --require-real  # FAIL if no real signals (use on the rig)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PROGRAM="${PROGRAM:-vector_add}"
PYTHON="${BCIR_PYTHON:-python3}"
REQUIRE_REAL=0
[ "${1:-}" = "--require-real" ] && REQUIRE_REAL=1

echo "[silicon] capability probe (the three real signals the measured replan is gated on):"
# The rig contract (docs/kernel/HARDWARE_VALIDATION.md): the measured win fires the moment a bare-metal
# host exposes ALL THREE -- a hardware PMU (perf_event_open), RAPL energy, and a cpufreq
# userspace governor. The probe enumerates each, names what is missing, and prints whether
# the host is rig-ready; the measured number itself still comes only from the real loop below.
"${PYTHON}" - <<'PY'
import bcir.silicon as s
pmu = bool(s.perf_counters_available())
rapl = bool(s.rapl_available())
gov = bool(getattr(s.cpufreq_info(), "actuatable", False))
caps = {
    "PMU (perf_event_open)": pmu,
    "RAPL energy (running average power limit)": rapl,
    "DVFS governor (cpufreq userspace)": gov,
}
for name, ok in caps.items():
    print(f"  {name}: {'REAL' if ok else 'unavailable'}")
missing = [n for n, ok in caps.items() if not ok]
if missing:
    print("  rig-ready: NO -- missing: " + "; ".join(missing))
else:
    print("  rig-ready: YES (PMU + RAPL + userspace governor) -- measured win is live")
PY

echo "[silicon] measured replan (bcir.run --silicon):"
out="$("${PYTHON}" -m bcir.run "${PROGRAM}" --silicon 2>&1)" || { echo "  FAIL: --silicon errored"; echo "${out}"; exit 1; }
echo "${out}" | sed -n 's/^/  /p' | grep -E "silicon|win=" || true

# Parse the provenance + win the loop certified.
prov_line="$(echo "${out}" | grep -oE "provenance=\([^)]*\)" | head -1)"
win="$(echo "${out}" | grep -oE "win=-?[0-9]+" | head -1 | cut -d= -f2)"
is_real=0
echo "${prov_line}" | grep -qi "real" && is_real=1

echo "[silicon] verdict:"
if [ "${is_real}" -eq 1 ]; then
  echo "  MEASURED replan win = ${win:-?} (provenance=real) -- the CT4 evidence."
  exit 0
fi
echo "  degraded (synthetic): no MEASURED win. The software path ran end to end and"
echo "  certified win=${win:-0} on synthetic telemetry; a real win needs the rig"
echo "  (docs/kernel/HARDWARE_VALIDATION.md: PMU + cpufreq userspace governor + RAPL)."
if [ "${REQUIRE_REAL}" -eq 1 ]; then
  echo "  FAIL: --require-real set but no real signals on this host."
  exit 1
fi
exit 0
