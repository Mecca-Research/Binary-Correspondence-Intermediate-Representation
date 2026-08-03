# Target access: what BCIR's remaining phases need, and what this environment cannot give them

BCIR's open phases are blocked on **access**, not on code. J6's hardware counters need a PMU.
J7's driver experiment needs to build a kernel module, load it, and bind a device. Every
timing claim wants a core whose frequency does not move underneath it.

This document exists so those limits are **stated before** they are hit, rather than
discovered one phase at a time. Everything in it is probed rather than assumed:
[`tools/silicon/probe_capabilities.sh`](../tools/silicon/probe_capabilities.sh) prints a JSON
record for any candidate host, and the tables below were produced by running it.

## 1. Privilege is not capability

The single most misleading thing about this environment is that it looks powerful. The
session container runs as **uid 0** with nearly the full capability set — including
`cap_sys_module`, `cap_sys_rawio`, `cap_perfmon`, `cap_bpf` and `cap_sys_admin` — and it
still cannot count one CPU cycle, load one kernel module, or observe its own clock frequency.

The reason is that those capabilities are *permissions to use hardware surfaces the
hypervisor never exposed*:

- `perf_event_paranoid` is **2**, which permits user-space counting. There is still no PMU:
  the event sources are `breakpoint msr power software tracepoint uprobe`, with **no `cpu`
  entry**, so `perf_event_open` returns `ENOENT` and no policy change can alter that.
- `cap_sys_module` is held, but `/lib/modules/$(uname -r)` does not exist and there are no
  kernel headers, so there is nothing to build a module against and nothing to insert.
- `/dev/mem` and `/dev/cpu/0/msr` exist, but there is no IOMMU, no VFIO and no UIO, so no
  device can be bound.

**Granting more privilege would not help.** These need a different machine, not a different
policy. That distinction is why the probe reports `perf_event_paranoid` and `hardware_pmu`
as separate fields: one is granted, the other is provisioned.

## 2. What the two available hosts actually provide

| Capability | Session container (x86-64, Docker/KVM) | Samsung S24+ (aarch64, Termux) |
|---|---|---|
| Hardware PMU (`cpu` event source) | **no** — hypervisor exposes none | **no** — `perf_event_paranoid=3` denies it |
| `perf_event_open` | `ENOENT` | `Permission denied` |
| cpufreq / governor control | **not exposed** | not exposed to Termux |
| Turbo / boost control | **not exposed** | n/a |
| Core pinning (`sched_setaffinity`) | yes | yes |
| `SCHED_FIFO` | yes | not attempted |
| `isolcpus` / `nohz_full` | **empty** | none |
| Kernel headers + `/lib/modules` | **no** | no (unrooted Android) |
| VFIO / UIO / IOMMU | **no** | no |
| `/dev/mem`, MSR | yes | no |
| `mlockall`, hugepages | yes / 0 | n/a |
| `/proc/stat` (steal accounting) | readable | **restricted** — records `null` |
| Clocksource | `tsc` (`constant_tsc nonstop_tsc rdtscp`) | 19.2 MHz ARM architectural timer |
| Effective timer granularity | ~1 ns | **52.083 ns** |
| Tenancy | **shared** — refused for timing | dedicated, pinnable per cluster |

Two consequences already shape the repository. The container is a shared runner, so
`calibration.py` refuses it for any frozen cost table — correctly, and permanently. The
phone's 52 ns timer is what forced grouped timing (`TIMING_METHOD = 2`), and even after that
fix its residual granularity is 52.083/8 ≈ 6.5 ns, which is still coarse enough that BER and
DER decode are **indistinguishable** on the fast core.

## 3. What each open phase needs

| Open phase | Required access | Available today |
|---|---|---|
| **J6 — target hardware counters** | A `cpu`/`armv8_pmuv3` event source, plus `perf_event_paranoid ≤ 2` or `CAP_PERFMON` | **Neither host.** Code is written and compiles; it has executed nowhere |
| **J6 — frequency-invariant timing** | cpufreq with `performance` governor; turbo disabled | **Neither host.** Frequency is not even observable |
| **J6 — low-noise timing** | `isolcpus`, `nohz_full`, `rcu_nocbs`, IRQ affinity off the measured core | **Neither host** |
| **J7 — D0/D1 driver ingest and views** | Kernel headers matching the running kernel; `/lib/modules`; module load permitted | **Neither host** |
| **J7 — D2/D3 device binding and trace parity** | A real device or VFIO passthrough; IOMMU; MMIO; IRQ delivery; the Linux reference driver | **Neither host** |
| **Hosted tools against bionic** | An Android NDK sysroot, or an Android CI runner | **No** — the network policy denies `dl.google.com` (403 on CONNECT) |
| Freestanding core across ABIs | A cross-capable clang | **yes** — `#targetabi` covers Android, armv7a, i686, riscv64, wasm32 |
| Correctness, differential and fuzz rails | Nothing special | **yes** — unaffected by any of the above |

The last row matters as much as the others: **none of this blocks correctness work.** The
Python oracle, the MLIR law rail, the C twins, the differential and fuzz gates, and the ECN
work all run here unimpeded. What is blocked is exclusively *measurement* and *device*
evidence.

## 4. The bare-metal targets that would unblock it

### x86-64

A physical machine, or a VM whose host explicitly passes through a virtual PMU. A stock cloud
VM is usually **not** sufficient — most hypervisors hide the PMU, which is exactly this
container's failure.

- root, and a kernel whose headers are installed and whose modules can be loaded
  (Secure Boot off, or a signing key enrolled);
- `perf_event_paranoid=0` in `/etc/sysctl.d`, or `CAP_PERFMON` granted to the harness;
- `intel_pstate` (or `acpi-cpufreq`) with the `performance` governor and
  `intel_pstate/no_turbo=1`, so a nanosecond means the same thing twice;
- kernel command line `isolcpus=N nohz_full=N rcu_nocbs=N` reserving at least one core, with
  IRQs steered away from it;
- `intel_iommu=on` (or `amd_iommu=on`) plus `vfio-pci` if J7 binds a real device;
- optionally Intel RDT/CAT to stop a noisy neighbour evicting the measured working set.

### aarch64

A single-board computer or ARM server — Raspberry Pi 5, RK3588 board, Ampere, or a bare-metal
Graviton instance. **Android is not sufficient at any privilege short of root**: the PMU is
denied by policy and `/proc/stat` is restricted, so a phone can produce wall-clock
calibration records and nothing more.

- root, kernel headers, module loading;
- the ARM PMU driver bound (`armv8_pmuv3` under `/sys/bus/event_source/devices`) and
  `perf_event_paranoid=0`;
- cpufreq governor control — on big.LITTLE, per cluster;
- `isolcpus`/`nohz_full` as above;
- for big.LITTLE, the ability to pin per cluster, which the phone already provides and which
  is why two admitted targets exist despite everything above.

### What a single machine of each would close

An x86-64 and an aarch64 box meeting the above would close: J6's counters on both
architectures, frequency-stable recalibration of both existing targets, J7's D0–D3 gates, and
the hosted-tool bionic gap if the aarch64 box can host an NDK. That is every measurement-
and device-blocked item in this document.

## 5. How to record a new host

```
tools/silicon/probe_capabilities.sh > host.json
```

The probe only reads — it configures nothing and does not touch `/dev/mem`, because a probe
that changed the machine would be describing a machine that no longer exists. It is portable
to Termux deliberately (no `awk`, `seq`, `sed` or `taskset`), so the constrained hosts can be
recorded on the same terms as the capable ones.

Compare `hardware_pmu`, `cpufreq_exposed`, `isolated_cpus`, `kernel_headers` and
`observed_clock_tick_ns` against §3 before promising a phase can be finished on it.
