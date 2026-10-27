#!/bin/bash
# NEXT-13's survey, run from <REPO> on the committed tree (HEAD) one step at a time with nothing else running, two
# workers at most; <SCRATCH> a scratch directory. Where the GEO step's attention jobs lose their workers: NEXT-12's probe
# read the attention's 16 tasks (4 batches x 4 KV heads) leaving their workers up to 16% of a job apart.
#  1 vcpu     -- vcpu_probe.c: a fixed latency chain, 16 AVX-512 FMA chains and a 64 MiB read, each pinned to one CPU at a
#                time (alone) and on every CPU at once (together), 41 rounds: is one of this guest's CPUs slower?
#  2 probe    -- ../../2026-10-24/gp-lanes/probe_jobs.py's copies of the unit and the trainer (every pool job's launch,
#                tasks, wall cycles and, a worker at a time, entry, busy cycles, finish and tasks) in a worktree of HEAD;
#                ../../2026-10-24/gp-lanes/probe_step.py at the GEO dimensions, kblock+exp+reduce, four threads, T1024
#                and T2048, 8 logged steps each, twice, in alternating order
# Writes host.json, vcpu.json and probe.json beside itself.
set -euo pipefail
repo="<REPO>"; s="<SCRATCH>/next13-survey"
here="${repo}/docs/research/measurements/2026-10-27/next13"
gp="${repo}/docs/research/measurements/2026-10-24/gp-lanes"
py="<SCRATCH>/r814/mlenv/bin/python"  # the model labs' interpreter (NumPy); probe_step.py runs under it with -I
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
rm -rf "${s:?}"; mkdir -p "${s}/cwd"
head=$(git -C "${repo}" rev-parse HEAD)
lines() { python3 -c 'import json, sys; json.dump([json.loads(l) for l in sys.stdin], open(sys.argv[1], "w"), indent=1)' "$1"; }

# host.json: the host as /proc and /sys present it, the three clocks a record here can and cannot read
python3 - "${here}/host.json" <<'PY'
import json, os, sys
def first(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None
info = {}
for line in (first("/proc/cpuinfo") or "").splitlines():
    if ":" in line:
        k, v = (x.strip() for x in line.split(":", 1))
        info.setdefault(k, v)
flags = set(info.get("flags", "").split())
want = ("hypervisor", "constant_tsc", "nonstop_tsc", "tsc_known_freq", "tsc_deadline_timer", "x2apic", "arat",
        "aperfmperf", "arch_perfmon", "avx2", "avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512_vnni", "amx_tile")
sources = sorted(os.listdir("/sys/bus/event_source/devices")) if os.path.isdir("/sys/bus/event_source/devices") else []
msr_events = sorted(os.listdir("/sys/bus/event_source/devices/msr/events")) if "msr" in sources else []
record = {
    "brand": info.get("model name"), "vendor": info.get("vendor_id"),
    "signature": "%02X_%02XH" % (int(info.get("cpu family", "0")), int(info.get("model", "0"))),
    "family": int(info.get("cpu family", "0")), "model": int(info.get("model", "0")),
    "stepping": int(info.get("stepping", "0")),
    "microcode": {"presented": info.get("microcode"),
                  "trust": "hypervisor-presented" if "hypervisor" in flags else "native"},
    "flags": {f: f in flags for f in want},
    "cpus": os.cpu_count(),
    "clocksource": first("/sys/devices/system/clocksource/clocksource0/current_clocksource"),
    "perf_event_paranoid": first("/proc/sys/kernel/perf_event_paranoid"),
    "event_sources": sources, "msr_events": msr_events,
    "powercap": os.path.isdir("/sys/class/powercap"),
    "clocks": {
        "elapsed": "CLOCK_MONOTONIC: every time in these records",
        "reference_ticks": "the TSC (invariant: constant_tsc and nonstop_tsc); its rate as the kernel detected it at "
                           "boot, not read by these tools; no tick count is derived from a time here",
        "core_cycles": None,
        "core_cycles_reason": "no hardware PMU event source (a KVM guest): core cycles, instructions retired and the "
                              "decoded-µop-cache events cannot be counted here, and none is inferred from the TSC",
        "effective_frequency": None,
        "effective_frequency_reason": "neither APERF/MPERF (the aperfmperf flag absent; the msr event source offers "
                                      "only tsc and smi) nor a core-cycle counter",
        "energy": None, "energy_reason": "no RAPL (no powercap; the power event source lists no events)"},
    "level": "A (timing-qualified): correctness witnesses and elapsed time against an A/A control; no "
             "microarchitectural counter (Level B) is available on this host",
}
json.dump(record, open(sys.argv[1], "w"), indent=1)
PY

# vcpu.json: the probe under GCC with AVX-512 (its workloads are the CPU's, not the compiler's)
gcc -std=c11 -O2 -mavx512f -Wall -Wextra -Werror -pthread "${here}/vcpu_probe.c" -o "${s}/vcpu_probe" -lm
"${s}/vcpu_probe" 41 | lines "${here}/vcpu.json"

# probe.json: the step's pool jobs a worker at a time, on HEAD's unit and trainer with the probe's clock reads
git -C "${repo}" worktree add --detach "${s}/wt" "${head}" > /dev/null
python3 -I "${gp}/probe_jobs.py" "${s}/wt/runtime/c" "${s}/probe-src" > /dev/null
cp "${s}/probe-src/bcir_tensor.c" "${s}/probe-src/bcir_decoder_train.c" "${s}/wt/runtime/c/"
for run in 1 2; do
  if [ "${run}" = 1 ]; then order="1024 2048"; else order="2048 1024"; fi
  for t in ${order}; do
    (cd "${s}/cwd" && "${py}" -I "${gp}/probe_step.py" "${s}/wt" "${t}" kblock+exp+reduce 8 "${s}/probe-T${t}-${run}.json" 4) \
      > "${s}/probe-T${t}-${run}.txt" 2>&1
  done
done
python3 - "${s}" "${here}/probe.json" "${head}" <<'PY'
import json, sys
from pathlib import Path
s, out, head = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
record = {"tree": head, "runs": []}
for run in (1, 2):
    for t in (1024, 2048):
        record["runs"].append({"run": run, "T": t, "summary": (s / f"probe-T{t}-{run}.txt").read_text().splitlines(),
                               "jobs": json.loads((s / f"probe-T{t}-{run}.json").read_text())})
json.dump(record, open(out, "w"), indent=1)
PY
git -C "${repo}" worktree remove --force "${s}/wt"
git -C "${repo}" worktree prune
echo "survey written"
