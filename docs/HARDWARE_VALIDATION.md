# Hardware Validation: limits & the rig the full tests need

The adaptive layer (`bcir.silicon`, `bcir.gem.dvfs`, `kbcir.allocator`,
`telemetry.TelemetryRing`) is wired to **real machine signals** and attempts the
privileged operations (hardware PMU counters, DVFS actuation), **degrading
gracefully** and reporting the boundary rather than faking a result. This note
states exactly how far validation goes in the CI sandbox, what is blocked and why,
and the test rig required to run the full hardware validation.

`python3 -c "import json,bcir.silicon as s; print(json.dumps(s.summary(),indent=2))"`
prints the live real/unavailable split for any host.

## What IS validated here (real, measured — `bcir/tests/test_silicon.py`)

| Capability | Real signal used | Measured result on the CI Xeon |
|---|---|---|
| Allocator tier map | `/sys/.../cache` topology | real L1 48K / L2 2MB / L3 260MB; placement fits the real cache |
| Cache-tier latency | compiled pointer-chase | **L1 ≈ 1 ns vs DRAM ≈ 166 ns** — the real basis for hot→SRAM |
| Zero-copy telemetry ring | `struct.pack_into` vs `json.dumps` | **≈ 31× faster** than serialization |
| Ring feed | `getrusage` + CPU/wall timers | real CPU-ns / page-fault counts reach the ring |
| DVFS plan anchor | `/proc/cpuinfo` nominal (2100 MHz) | clock plan quantized to the real nominal frequency |

These are genuine and reproducible; they do not need privilege.

## What is BLOCKED in this sandbox (and why)

| Operation | Blocked by | Symptom |
|---|---|---|
| Hardware PMU counters (cycles / instructions / **cache-misses**, IPC) | no virtualized PMU exposed by the hypervisor | `perf_event_open` → **ENOENT**; `read_hw_counters()` → `None` (falls back to OS counters) |
| DVFS **actuation** (writing the CPU clock) | no cpufreq driver / governor in the guest | `scaling_governor` absent → `cpufreq_info().actuatable == False`; `actuate()` returns a **dry-run** with the reason |
| **Power-savings** measurement (the DVFS payoff) | no RAPL / power-cap interface, and no frequency control | `/sys/class/powercap/intel-rapl*` absent; cannot measure energy delta |
| Root-only signals (kernel-mode events, turbo control) | unprivileged container, no `CAP_*` | n/a |

The code paths for all of these exist, are exercised by tests on the
**attempt-then-degrade** path, and **light up automatically** on a host that grants
the capability — no code change required.

## The rig required for FULL hardware validation

A **bare-metal Linux host** (or a "metal" cloud instance / a KVM guest started with
`-cpu host,pmu=on`), configured as follows:

1. **Hardware PMU** — for real cycles / instructions / **cache-miss** attribution
   and IPC:
   - a real or virtualized PMU (`dmesg | grep -i PMU`; `perf stat true` works);
   - `sysctl kernel.perf_event_paranoid=1` (user-space HW events; we set
     `exclude_kernel`), or `0` / `CAP_PERFMON` for kernel events too.
   - Then `read_hw_counters()` returns an `HwSample` and the ring carries real
     cache-miss counts instead of page faults.

2. **DVFS actuation** — for writing the per-phase clock:
   - a cpufreq driver with a settable governor: boot `intel_pstate=passive` (or use
     `acpi-cpufreq`), then `cpupower frequency-set -g userspace` (exposes
     `scaling_setspeed` + `scaling_available_frequencies`);
   - run as **root** / `CAP_SYS_ADMIN` so `scaling_setspeed` is writable.
   - Then `cpufreq_info().actuatable == True` and `actuate()` sets the clock and
     reads `scaling_cur_freq` back to confirm.

3. **Power measurement** — to *prove* the DVFS power-savings claim (not just model
   it):
   - Intel **RAPL** at `/sys/class/powercap/intel-rapl:0/energy_uj` (read access;
     may need root after the recent side-channel mitigation), or AMD `amd_energy`,
     or an external wall meter.
   - Protocol: run a **memory-bound** kernel at nominal vs at `DOWNCLOCK` (0.75×),
     measure (a) throughput — expected **unchanged** (bandwidth-bound) — and (b)
     energy via RAPL — expected **lower**. That is the measured payoff; until then
     the power-savings number stays *modeled*, not claimed.

4. **Measurement hygiene** (so the numbers are trustworthy):
   - pin work with `taskset` / `cpuset`; isolate cores (`isolcpus=`);
   - fix turbo for determinism (`echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo`);
   - a quiet machine (no co-tenant noise — the opposite of a shared CI runner).

## Honest status line

- **Measured & real now:** the cache tier map, cache-latency tiers, the zero-copy
  ring throughput, the ring fed by real OS counters.
- **Coded, attempted, gated (needs the rig above):** hardware cache-miss counters,
  DVFS clock actuation, and the RAPL-measured power-savings claim.
- We do **not** claim a DVFS power win until it is measured on a host with RAPL +
  frequency control. The model says it should hold (memory-bound throughput is
  clock-insensitive); the rig above is what turns "should" into "did".
