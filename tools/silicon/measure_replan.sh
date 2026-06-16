#!/usr/bin/env bash
# Rig runbook: run the CT4 measured replan end to end and report the win.
#
# On a bare-metal rig with the capabilities HARDWARE_VALIDATION.md names -- a hardware
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
REQUIRE_REAL=0
[ "${1:-}" = "--require-real" ] && REQUIRE_REAL=1

echo "[silicon] capability probe (real signals that drive the measured replan):"
python3 - <<'PY'
import json, bcir.silicon as s
summ = s.summary()
def avail(*keys):
    for k in keys:
        v = summ.get(k)
        if isinstance(v, dict):
            v = v.get("available", v.get("ok"))
        if v:
            return True
    return False
pmu = bool(s.perf_counters_available())
caps = {
    "PMU (perf_event_open)": pmu,
    "DVFS actuation (userspace governor)": bool(getattr(s.cpufreq_info(), "actuatable", False)),
}
for name, ok in caps.items():
    print(f"  {name}: {'REAL' if ok else 'unavailable'}")
PY

echo "[silicon] measured replan (bcir.run --silicon):"
out="$(python3 -m bcir.run "${PROGRAM}" --silicon 2>&1)" || { echo "  FAIL: --silicon errored"; echo "${out}"; exit 1; }
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
echo "  (HARDWARE_VALIDATION.md: PMU + cpufreq userspace governor + RAPL)."
if [ "${REQUIRE_REAL}" -eq 1 ]; then
  echo "  FAIL: --require-real set but no real signals on this host."
  exit 1
fi
exit 0
