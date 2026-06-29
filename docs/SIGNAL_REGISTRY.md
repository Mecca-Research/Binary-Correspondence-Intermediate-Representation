# Telemetry signal-provider registry (T1)

> The **PAPI-component model** for BCIR: a vendor-neutral registry of typed telemetry
> signal providers, mapped to the 12-D cost vector, with honest `None`/unavailable when a
> source is absent. The T1 row of the [`TELEMETRY_PIPELINE_RESEARCH.md`](TELEMETRY_PIPELINE_RESEARCH.md)
> §6 build order. Code: [`bcir/signal_registry.py`](../bcir/signal_registry.py); tests:
> [`bcir/tests/test_signal_registry.py`](../bcir/tests/test_signal_registry.py).
>
> **The governing invariant first.** Telemetry is a *graded L2/L3 signal*. The registry READS
> signals and may inform plan cost (the `theta` runtime state / cost-weight calibration) but it
> MUST NEVER be — or alter — an R-law legality verdict, emit a Diagnostic, or sit on the legality
> path. There is deliberately **no `verify` / R-law surface** here: a provider returns a `Reading`
> or `None`, never a legality object. The two-truth quarantine, applied to measurement.

## What it is

`bcir/silicon.py` already reads real host signals (RAPL energy, PMU counters, on-die thermal,
cpufreq, cache topology, OS counters), but as a flat bag of functions: no uniform typed interface,
no provenance/units on a *definition*, no registry/plugin seam to add vendor sources without editing
the core. T1 formalizes that into a registry of typed providers. It **builds on** `silicon.py`'s
readers (wrapping them) — it does not replace or modify them.

### Design DNA (research §3)

| Copied abstraction | Realized as |
|---|---|
| **PAPI component model** — one symbolic name behind a stable typed interface; components plug in | `SignalProvider` (ABC) + `SignalRegistry.register()` (the plugin seam, mirroring `bcir/channel_plugin.py`) |
| **hwmon typed schema** — fixed units (m°C/µJ/kHz/…), absent file → honest `None` | `Unit` (PERCENT / MILLICELSIUS / MICROJOULE / KHZ / BYTES / COUNT / RATIO_MILLI / NONE); `read() → None` when absent |
| **Redfish 4-resource split** — provenance + units on the *definition*, not the sample | `MetricDefinition` (the metadata) vs `Reading` (one sample) |
| **`sampling_model`** (research §4 gap) | `SamplingModel` ∈ {POLLED, STREAMED, EVENT_DRIVEN} on the definition |
| **Provenance vocabulary** (reuses the channel `real`/`modeled`/`simulated`) | `Provenance` ∈ {MEASURED, MODELED, SIMULATED} (`real` ↔ `MEASURED`) |

## Core types

- **`MetricDefinition`** (frozen) — `name` (stable dotted id, e.g. `"thermal.pressure"`), `unit`,
  `cost_dim` (one of `DIMS` or `None`), `provenance`, `sampling_model`, `description`. `validate()`
  returns schema errors. Provenance/units live HERE, once — not on every reading.
- **`Reading`** (frozen) — one sample: `definition`, `value` (in the definition's unit), and the
  *actual* `provenance` of this read. `read() → None` means the source is genuinely absent.
- **`SignalProvider`** (ABC) — `definition` + `available() → bool` + `read() → Reading | None`.
  Honest by construction: `available()` and `read()` always agree (a `Reading` iff available,
  `None` iff not — never a fabricated value).
- **`SignalRegistry`** — providers keyed by `definition.name`. `register()` (rejects a duplicate
  name + an invalid definition), `get()`, `providers_for_dim(dim)`, `snapshot() → {name: Reading|None}`
  (the `silicon.summary()` analog, with provenance), `availability() → {name: bool}` (the honest split).

## Providers

### Wired (wrapping `silicon.py`, mapped to a cost dim)

| Provider | Source | Unit | cost_dim |
|---|---|---|---|
| `ThermalPressureProvider` | `thermal_pressure()` | PERCENT | `thermal` |
| `DieTempProvider` | `read_thermal_millideg()` | MILLICELSIUS | `thermal` |
| `RaplEnergyProvider` | `read_rapl_uj()` | MICROJOULE | `power` |
| `CpuFreqProvider` | `cpufreq_info().nominal_khz` | KHZ | `compute` |
| `CacheCapacityProvider(L1/L2/L3)` | `tier_capacities()` | BYTES | `memory` |
| `PmuAvailabilityProvider` | `perf_counters_available()` | COUNT | `compute` |

Two honesty notes baked into the definitions: RAPL `energy_uj` is a **monotonic counter** — *watts is
a T3-derived metric* (energy Δ / wall Δ); here we expose the raw counter + availability, not a
fabricated power. The PMU **counters are work-scoped** (`silicon.read_hw_counters(work)` /
`silicon_dna`) and stay there; the registry exposes only the *capability + definition*, since a
point-sample of a raw counter outside a work window is meaningless.

### Honest-unavailable (vocabulary-completing gap providers, research §4)

Each has a real `MetricDefinition` so the namespace is complete, but reports `available() → False` /
`read() → None` on a host without the backend/hardware — exactly like `channels.py`'s *modeled*
future channels. A future NVML / amd-smi / Redfish / PCM / DCGM backend fills these in; they never
fabricate a value.

| Provider | Backend needed | cost_dim |
|---|---|---|
| `GpuPowerProvider` | NVML / amd-smi | `power` |
| `BmcPowerProvider` | Redfish (out-of-band) | `power` |
| `MemBandwidthProvider` | PCM IMC / DCGM DRAM-active | `memory` |
| `FabricBytesProvider` | NVLink / PCIe / UPI counters | `fabric` |
| `ThrottleStateProvider` | NVML ClocksEventReasons / amd-smi / RAPL | `contention` |
| `ReliabilityProvider` | ECC / XID / RAS / margin-drift / RUL | `reliability` |
| `HwmonPowerProvider` | hwmon `power*` / INA226 rail | `power` |

## Builders + the channel↔provider mapping

- **`default_registry()`** — every provider above, with RAPL as the default in-band power source.
- **`registry_for_channel(channel)`** — the default registry but the POWER provider matches the
  channel's `runtime.energy_source`: `rapl` → `RaplEnergyProvider`, `nvml` → `GpuPowerProvider`,
  `hwmon` → `HwmonPowerProvider`, `none` → no dedicated power provider. (Unmapped → RAPL fallback.)
  This demonstrates the channel↔provider mapping the research describes.
- **`theta_pressures(registry)`** — a read-only convenience that reads the registry's 0..100 PERCENT
  pressure signals (thermal, …) keyed by cost dim, suitable to FEED `theta`. It does NOT modify the
  legality path, the cost-vector DIMS, or the existing calibrate path — it only surfaces the graded
  signal on the *cost/optimization* side. Absent signals are omitted (honest), never defaulted.

## Honest real/unavailable split (typical sandbox)

OS-/cache-derived signals are usually present (CPU nominal frequency, L1/L2/L3 cache capacity);
RAPL / PMU / thermal / GPU / BMC / fabric / reliability are typically absent → `read() → None`. The
tests assert the *agreement* (`read() is None` iff `not available()`) and in-range/right-unit when
present — they self-adapt to the host rather than assume a specific signal exists.

## Next (T2–T4)

T2 — the UART telemetry frame (StreamPack/SyS-T-style, CRC-sealed); T3 — derived/aggregate metrics +
sensitivity (watts, IPC, bandwidth from the raw counters here); T4 — export adapters (OTLP/Prometheus
+ out-of-band Redfish). See [`TELEMETRY_PIPELINE_RESEARCH.md`](TELEMETRY_PIPELINE_RESEARCH.md) §6.
