# Telemetry signal-provider registry (T1)

> The **PAPI-component model** for BCIR: a vendor-neutral registry of typed telemetry
> signal providers, mapped to the 12-D cost vector, with honest `None`/unavailable when a
> source is absent. The T1 row of the [`TELEMETRY_PIPELINE_RESEARCH.md`](TELEMETRY_PIPELINE_RESEARCH.md)
> §6 build order. Code: [`bcir/signal_registry.py`](../../bcir/signal_registry.py); tests:
> [`bcir/tests/test_signal_registry.py`](../../bcir/tests/test_signal_registry.py).
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
| **PAPI/DCGM component model** — one name and stable numeric ID behind a typed interface; components plug in | `SignalProvider` + `SignalRegistry.register()`, `get()` and `get_by_id()` |
| **hwmon typed schema** — fixed units (m°C/µW/mW/µJ/kHz/byte/s/…), absent file → honest `None` | `Unit`; `read() → None` when absent |
| **Redfish 4-resource split** — provenance + units on the *definition*, not the sample | `MetricDefinition` (the metadata) vs `Reading` (one sample) |
| **Explicit metric semantics** | `MetricKind` ∈ {GAUGE, COUNTER}, `Temporality` ∈ {UNSPECIFIED, DELTA, CUMULATIVE}, `monotonic`, and `min_interval_ns` on the definition |
| **`sampling_model`** | `SamplingModel` ∈ {POLLED, STREAMED, EVENT_DRIVEN} on the definition |
| **Provenance vocabulary** (reuses the channel `real`/`modeled`/`simulated`) | `Provenance` ∈ {MEASURED, MODELED, SIMULATED} (`real` ↔ `MEASURED`) |

## Core types

- **`MetricDefinition`** (frozen Python value type) — `name`, stable `signal_id:u32`,
  `unit`, `cost_dim`, `provenance`, `sampling_model`, `metric_kind`, `temporality`,
  `monotonic`, `min_interval_ns`, and `description`. ID `0` is local/unassigned; built-ins
  use unique nonzero IDs. `validate()` rejects inconsistent gauge/counter semantics.
- **`Reading`** (frozen) — one sample: `definition`, `value` (in the definition's unit), and the
  *actual* `provenance` of this read. `read() → None` means the source is genuinely absent.
- **`SignalProvider`** (ABC) — `definition` + `available() → bool` + `read() → Reading | None`.
  Both calls are honest point-in-time probes, but hardware can disappear between calls;
  consumers must not treat two separate calls as an atomic transaction.
- **`SignalRegistry`** — providers keyed by `definition.name`. `register()` (rejects a duplicate
  name, duplicate nonzero ID, or invalid definition), `get()`, `get_by_id()`,
  `providers_for_dim(dim)`, and `snapshot() → {name: Reading|None}`. Derive availability
  from that same snapshot with `availability(snapshot)` when coherence matters. The
  snapshot boundary refuses a non-`Reading`, a mismatched definition, invalid provenance,
  bool/non-numeric/non-finite values, negative counters, and percentage values that are
  not exact integers in `0..100`.

## Providers

### Wired (wrapping `silicon.py`, mapped to a cost dim)

| ID | Provider | Source | Unit / semantics | cost_dim |
|---:|---|---|---|---|
| 1 | `ThermalPressureProvider` | `thermal_pressure()` | PERCENT gauge | `thermal` |
| 2 | `DieTempProvider` | `read_thermal_millideg()` | MILLICELSIUS gauge | `thermal` |
| 3 | `RaplEnergyProvider` | `read_rapl_uj()` | MICROJOULE cumulative monotonic counter | `power` |
| 4 | `CpuFreqProvider` | `cpufreq_info().nominal_khz` | KHZ gauge | `compute` |
| 5–7 | `CacheCapacityProvider(L1/L2/L3)` | `tier_capacities()` | BYTES gauge | `memory` |
| 8 | `PmuAvailabilityProvider` | `perf_counters_available()` | COUNT gauge | `compute` |

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

This honesty is now mechanically consumed by the hardware-policy rail: unavailable readings map
to a cleared `TelemetryToken.availability_mask` bit, not to a learned “zero pressure” sample. The
bounded simulated training gate does not change any provider's availability and is not physical
hardware evidence.

| ID | Provider | Backend needed | Unit / semantics | cost_dim |
|---:|---|---|---|---|
| 9 | `GpuPowerProvider` | NVML / amd-smi | MICROJOULE cumulative monotonic counter | `power` |
| 10 | `BmcPowerProvider` | Redfish | MILLIWATT gauge | `power` |
| 11 | `MemBandwidthProvider` | PCM IMC / DCGM | BYTES_PER_SECOND gauge | `memory` |
| 12 | `FabricBytesProvider` | NVLink / PCIe / UPI | BYTES cumulative monotonic counter | `fabric` |
| 13 | `ThrottleStateProvider` | NVML / amd-smi / RAPL | BITMASK gauge | `contention` |
| 14 | `ReliabilityProvider` | ECC / XID / RAS / RUL | COUNT gauge | `reliability` |
| 15 | `HwmonPowerProvider` | hwmon `power*` / INA226 | MICROWATT gauge | `power` |

## Builders + the channel↔provider mapping

- **`default_registry()`** — every provider above, with RAPL as the default in-band power source.
- **`registry_for_channel(channel)`** — the default registry but the POWER provider matches the
  channel's `runtime.energy_source`: `rapl` → `RaplEnergyProvider`, `nvml` → `GpuPowerProvider`,
  `hwmon` → `HwmonPowerProvider`, `none` → no dedicated power provider. (Unmapped → RAPL fallback.)
  This demonstrates the channel↔provider mapping the research describes.
- **`theta_pressures(registry)`** — a read-only convenience that reads the registry's 0..100 PERCENT
  pressure signals (thermal, …) keyed by cost dim, suitable to FEED `theta`. It does NOT modify the
  legality path, the cost-vector DIMS, or the existing calibrate path — it only surfaces the graded
  signal on the *cost/optimization* side. It consumes one validated snapshot and never truncates a
  fractional provider value. Absent signals are omitted (honest), never defaulted.

## Honest real/unavailable split (typical sandbox)

OS-/cache-derived signals are usually present (CPU nominal frequency, L1/L2/L3 cache capacity);
RAPL / PMU / thermal / GPU / BMC / fabric / reliability are typically absent → `read() → None`. The
tests use one snapshot for coherent availability and assert in-range/right-unit readings when
present—they self-adapt to the host rather than assume a specific signal exists.

## Status and pre-driver boundary

T1–T4 data contracts are implemented. Exporters consume the definition's declared
counter/temporality fields; they no longer infer behavior from a metric name or unit.
`metric_definitions()` includes the BCIR signal ID and semantics in its Redfish OEM block.

This Python registry is the normative taxonomy oracle, not yet a driver UAPI. Before D2,
generate a fixed-width C definition table from the same source, reserve ID ranges for
BCIR/vendor/device-local signals, define unknown-required-signal refusal, and carry the ID
in the new driver telemetry envelope. Live providers and UART/HTTP/OTLP/Redfish transports
remain unimplemented. See [`TELEMETRY_PIPELINE_RESEARCH.md`](TELEMETRY_PIPELINE_RESEARCH.md)
and [`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md).
