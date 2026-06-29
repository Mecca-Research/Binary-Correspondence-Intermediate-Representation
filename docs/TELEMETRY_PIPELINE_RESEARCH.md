# BCIR telemetry / monitoring pipeline — research & scoping

> Research input for the BCIR telemetry/monitoring pipeline. Surveys industry telemetry
> sources and standards across seven layers, maps each to BCIR's existing surfaces, and
> scopes what to include. Related: [`HETEROGENEOUS_CHANNELS.md`](HETEROGENEOUS_CHANNELS.md)
> (the channel signal-provider seam), [`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md)
> (the airlock shape this mirrors), [`SYCL_INTEROP.md`](SYCL_INTEROP.md).
>
> **The governing invariant first:** telemetry is a *graded L2/L3 signal*. It MAY inform plan
> cost (the `theta` runtime state, calibration of cost-vector weights) but it may NEVER be or
> alter an R-law legality verdict. The decision/legality path stays deterministic/integer/Q8;
> telemetry feeds the *cost/optimization* side only — the two-truth quarantine, applied to
> measurement.

## 0. What BCIR already has (the substrate this builds on)
- **`bcir/telemetry.py`** — `TelemetryRing` (fixed-width RAM ring; fixed-stride `DataDNA`
  records `<7q>` = claim_id, cycles, bytes, misses, thermal, voltage, utilization; monotonic
  head/tail; overwrite-on-full with a `dropped` count), `TelemetryIntegrity` + `sanitize_events`
  (RT3 ingest validation: 0..100 band / finite / non-negative; `blind` suppression detection;
  `monotonic` replay/reorder detection), `parse_shared_ring` (hardened header decode, OOB-read
  defense). This IS the RT3 telemetry-stream-integrity layer.
- **`runtime/c/bcir_streampack.h`** — `"BSPK"` magic + version gate + length-prefixed records +
  trailing CRC-32, little-endian, frozen ABI. A framed, integrity-checked binary wire format.
- **`RuntimeChannel`** (per hardware channel) — `energy_source ∈ {rapl, hwmon, nvml, none}`,
  `perf_syscall_nr` (the per-ABI `perf_event_open` number: x86_64=298, aarch64=241, …),
  `thermal_zone_types` (`/sys/class/thermal/*/type`).
- **`calibration` + `provenance`** (per channel) — `provenance ∈ {real, modeled, simulated}` +
  a `CalibrationArtifact` (ref + content digest + cal_gen). The measured-vs-modeled fidelity model.
- **The 12-D cost vector** — `compute, memory, fabric, sync, compile, thermal, power, reliability,
  security, accuracy, contention, verification`. Telemetry calibrates the hardware-sensitive dims.

## 1. Layered model of telemetry sources (where each tool sits)

| Layer | Representative tools | Carries | BCIR surface |
|---|---|---|---|
| **L7 In-silicon sensors** | Synopsys SLM (PVT + Path-Margin Monitors), proteanTecs Agents | timing-margin, drift, aging → **RUL** | `reliability` dim; `provenance=real` + calibration digest |
| **L6 Physical bus / monitor ICs** | INA219/INA226 (I²C power/current), LM75/TMP102 (temp); Saleae decodes the bus | volts/amps/°C/W | sources of `power`/`thermal` |
| **L5 OS kernel interfaces** | Linux **hwmon** sysfs, **perf_event_open**, **RAPL** (`/sys/class/powercap`), MSRs; Windows kernel-driver/WMI | temp/power/PMU counters | `energy_source=hwmon/rapl`, `perf_syscall_nr`, `thermal_zone_types` |
| **L4 Vendor accelerator libs** | NVIDIA **NVML/DCGM** (nvidia-smi), AMD **amd-smi/ROCm-SMI**, Intel **PCM/VTune** | GPU/CPU power, clocks, util, VRAM/DRAM BW, NVLink/PCIe BW, ECC, throttle reasons | `energy_source=nvml`; **gaps**: amd_smi, mem-BW, fabric, throttle, reliability providers |
| **L3 Embedded real-time trace** | SEGGER **RTT**/J-Scope, **STM32CubeMonitor**, ARM **CoreSight ITM/SWO/DWT** | live MCU variables, cycle counters, `printf` | the **UART driver** tap; `TelemetryRing`; DWT→`cycles`/`misses` |
| **L2 DC / BMC management** | DMTF **Redfish** (REST/JSON), **IPMI**, **MCTP/PLDM** (SMBus/PCIe), SNMP | node/chassis power/thermal/fan, component sensors | out-of-band export + internal component transport |
| **L1 Software pipeline / export** | **OpenTelemetry** (OTLP), **Prometheus/OpenMetrics** | metric model + transport + time-series | the external export boundary |

## 2. Metric taxonomy → BCIR cost dimensions
| Telemetry class | Source(s) | → cost dim |
|---|---|---|
| Power / energy (W, µJ) | RAPL, NVML/DCGM, hwmon `power*`, Redfish, INA226 | **power** |
| Thermal (°C, thermal-throttle) | hwmon `temp*`, NVML, coretemp/x86_pkg_temp, LM75 | **thermal** |
| Clock / freq (SM/mem/core MHz, APERF/MPERF) | NVML clocks, turbostat | **thermal/power** (DVFS evidence) |
| Utilization (SM%, busy%) | NVML/DCGM, /proc, perf | **compute**, **contention** |
| PMU counters (instr, IPC, cache, branch) | perf_event_open, PAPI, PCM, VTune, DWT | **compute**, **memory** |
| Memory bandwidth (IMC / VRAM BW) | PCM uncore IMC, DCGM DRAM-active | **memory** |
| Interconnect (NVLink / PCIe / UPI bytes) | DCGM, PCM | **fabric**, **sync** |
| Throttle / violation bits (HW slowdown, power-cap, thermal) | NVML ClocksEventReasons, amd-smi, RAPL | **thermal/power/contention** |
| Reliability (ECC, XID, RAS, margin/drift, RUL) | NVML/DCGM ECC, amd-smi RAS, SLM PMM, proteanTecs | **reliability** |

Thin direct coverage: **sync** (inferred from fabric + throttle) and **contention** (inferred from
util saturation + counter-multiplex scaling).

## 3. Abstractions worth copying (the design DNA)
1. **PAPI's vendor-neutral counter namespace + component model** — one symbolic name resolves to
   native events across Intel/AMD/ARM/NVIDIA/ROCm; *components* (perf, rapl, nvml, rocm_smi,
   lm-sensors) plug behind a stable API. **This is exactly BCIR's signal-provider pattern.**
2. **hwmon typed schema** (`<type><n>_<item>`, fixed units m°C/µW/mV/RPM) — one reader, every
   conforming driver; absent files → honest `None` (BCIR already does this).
3. **DCGM stable numeric field-IDs + exporter** — decouple "what is measured" from "how it's
   transported"; version-stable IDs.
4. **Redfish 4-resource split** — `MetricDefinition` (taxonomy + units + **provenance/calibration ref**)
   / `MetricReportDefinition` (cadence + cohort) / `MetricReport` (readings) / `Triggers`
   (thresholds). Provenance lives on the *definition*, not every sample.
5. **OpenTelemetry point types + explicit temporality** — `Gauge` / monotonic `Sum` / `Histogram` /
   `ExponentialHistogram`; declare **delta vs cumulative** and **monotonicity** per channel (these
   break silently if implicit). Compact binary export (OTLP/protobuf) + a text exposition for debug.
6. **MIPI SyS-T** — compact framed trace records + *optional* integrity checksum, explicitly over
   **UART/USB/TCP**. The closest standard to BCIR's StreamPack-over-UART goal; use as the wire-frame
   interop reference. **STP** adds multi-stream channel-tagging + timestamps if sources are multiplexed.
7. **SEGGER RTT pattern** — small RAM ring + out-of-band egress; target cost = one bounded `memcpy`;
   loss handled by overwrite (the host must *detect* loss). BCIR's `TelemetryRing` already is this;
   the UART driver is the "drain" half.
8. **SPICE/PSpice monitoring metaphor** (from the uploaded guide; OCR'd from scanned page images —
   late-1980s edition, predates `.MEASURE`/`.STEP`):
   - **`.PROBE [vars]`** → tap layer + cardinality: bare = capture-all firehose; named = selective
     taps. Typed addressable signal names `V(node)`, `V(a,b)` (differential), `I(device)`, mag/phase/dB.
   - **`.TRAN <print interval> <final> [<no-print> [<step ceiling>]]`** → **separate cadences**:
     record-rate (`print interval`) vs solver/collection-rate (`step ceiling`) vs warm-up gate
     (`no-print interval`). Three independent telemetry knobs.
   - **`.MEASURE`-concept** (realized in this edition via `.TF`/`.FOUR`/`.NOISE`/`.MC YMAX`) →
     derived/aggregate metrics (max/min/avg/RMS/trigger→target) computed at the edge from raw waveforms.
   - **`.DC`/`.AC`/`.MC`** → parametric / what-if sweeps (linear vs log spacing; Monte Carlo over
     DEV/LOT tolerances) for worst-case telemetry exploration.
   - **`.SENS`** → signal prioritization: rank which signals most affect the output/cost (the book
     literally names "drift") → drive sampling budget + alerting toward high-sensitivity signals.
   - **`.OPTIONS`** → governance knobs: `LIMPTS` (retention/cardinality cap), `RELTOL/VNTOL`
     (precision/quantization tolerance), `TNOM=27°C` (**calibration reference condition** — reinforces
     BCIR's calibration+provenance: the condition under which a measurement is valid).

## 4. Gaps to add to BCIR's existing surfaces
- **`energy_source`**: add `amd_smi`/`rocm_smi` (AMD parity with `nvml`) and `redfish` (out-of-band
  BMC; the only OS-independent tier, valuable when in-band RAPL is restricted post-CVE).
- **New runtime signal-providers** for under-covered dims: a **memory-bandwidth** provider (PCM IMC /
  DCGM DRAM-active → `memory`), a **fabric/interconnect** provider (NVLink/PCIe/UPI bytes →
  `fabric`/`sync`), a **throttle/clock-event-reason** field (→ `thermal`/`power`/`contention`), a
  **reliability** provider (ECC/XID/RAS/margin/drift/RUL → `reliability`).
- **`sampling_model ∈ {polled, streamed, event_driven}`** + min-interval hint (PMU/VTune are
  event-driven; DCGM streams at 100 ms–1 s; current fields imply polled-only).
- **Correction**: drop **Intel ISS** from the design — it is a motion/ambient sensor-*hub* driver,
  not CPU power/thermal/PMU telemetry. Intel's real telemetry is RAPL + PMU (PCM/perf/VTune).

## 5. Recommended architecture (vendor-neutral, two-truth-safe)
```
  SOURCES (L7..L3) ──► PROVIDERS (PAPI-style components, one behind each energy_source/dim)
        │                     │  typed records, fixed units (hwmon schema), provenance on definition
        ▼                     ▼
  INGEST + VALIDATE  ──► RT3 (sanitize_events + TelemetryIntegrity + parse_shared_ring)
        │  reject-and-count out-of-band / suppression / replay; provenance witness
        ▼
  GRADED SIGNAL ──► theta / cost-weight calibration  ── (NEVER an R-law verdict; two-truth airlock)
        │
        ▼
  EXPORT BOUNDARY ──► framed wire: StreamPack/SyS-T-style (magic+version+len+payload+CRC, resync-able)
                      over UART (embedded) or OTLP/Redfish (data-center); pull + push.
```
Internal component transport ≈ MCTP/PLDM tier (below the OS); external export ≈ Redfish/OTLP tier —
mirroring BCIR's internal-channel vs external-boundary distinction (and the G8 airlock shape in
[`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md)).

## 6. Suggested build order (each a gated segment)
- **T1** — a vendor-neutral **signal-provider registry** (PAPI-component model): one provider per
  `energy_source`/dim behind a stable typed interface; honest `None` when absent; provenance on the
  definition (Redfish 4-split). Pure Python oracle + a metric-definition schema. Fully testable.
  **BUILT** — `bcir/signal_registry.py` (+ `bcir/tests/test_signal_registry.py`). See
  [`SIGNAL_REGISTRY.md`](SIGNAL_REGISTRY.md). `MetricDefinition` (units/dim/provenance/`sampling_model`)
  / `Reading` (one sample) / `SignalProvider` (ABC) / `SignalRegistry` (the `register()` plugin seam,
  `snapshot()`/`availability()`). **Wired** (wrapping `bcir/silicon.py`, mapped to a cost dim):
  thermal pressure + die-temp (`thermal`), RAPL energy counter (`power`; watts is T3-derived), CPU
  nominal freq (`compute`), L1/L2/L3 cache capacity (`memory`), PMU capability (`compute`; the actual
  counters stay work-scoped in `silicon.py`). **Honest-unavailable** vocabulary-completing gap
  providers (a real definition, `available()→False`/`read()→None` until a backend lands): GPU power
  (NVML/amd-smi) + BMC power (Redfish) → `power`, memory bandwidth (PCM/DCGM) → `memory`, fabric bytes
  (NVLink/PCIe/UPI) → `fabric`, throttle state → `contention`, reliability (ECC/XID/RUL) →
  `reliability`. `registry_for_channel(ch)` picks the power provider matching `ch.runtime.energy_source`
  (`rapl`→RAPL, `nvml`→GPU, `hwmon`→hwmon-power, `none`→none). Off the legality path: a provider returns
  only a `Reading`/`None`, never a verdict/Diagnostic; `theta_pressures()` surfaces the graded 0..100
  signal to feed `theta` without touching `bcir/verify` or the cost-vector DIMS.
- **T2** — the **UART telemetry frame**: a StreamPack/SyS-T-style framed, CRC-sealed record stream; a
  C producer drains `TelemetryRing`; the host decoder reuses RT3 (`sanitize_events`/`TelemetryIntegrity`).
  Resync-on-magic; per-frame CRC; sequence/timestamp for drop/reorder. (Feeds the planned UART driver.)
- **T3** — **derived/aggregate metrics + sensitivity** (the `.MEASURE`/`.SENS` analogy): edge-computed
  figures-of-merit + a sensitivity rank that steers the sampling budget toward high-impact signals.
- **T4** — **export adapters**: OTLP/Prometheus exposition (data-center) + the out-of-band Redfish read.

Everything above is the *cost/optimization* side. The legality verdict (R1–R21) never reads telemetry.

## Sources
NVML/DCGM <https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html>;
ROCm/amd-smi <https://rocm.blogs.amd.com/software-tools-optimization/amd-smi-overview/README.html>;
RAPL <https://web.eece.maine.edu/~vweaver/projects/rapl/>;
perf_event_open <https://www.man7.org/linux/man-pages/man2/perf_event_open.2.html>;
PAPI <https://icl.utk.edu/papi/>; hwmon <https://docs.kernel.org/hwmon/sysfs-interface.html>;
lm-sensors <https://github.com/lm-sensors/lm-sensors>;
LibreHardwareMonitor <https://github.com/LibreHardwareMonitor/LibreHardwareMonitor>;
Synopsys SLM <https://www.synopsys.com/solutions/silicon-lifecycle-management.html>;
proteanTecs <https://www.proteantecs.com/technology>;
Redfish Telemetry <https://www.dmtf.org/sites/default/files/standards/documents/DSP2051_1.0.1.pdf>;
PLDM/MCTP <https://www.dmtf.org/standards/pmci>;
OpenTelemetry metrics <https://opentelemetry.io/docs/specs/otel/metrics/data-model/>;
SEGGER RTT <https://www.segger.com/products/debug-probes/j-link/technology/about-real-time-transfer/>;
STM32CubeMonitor <https://wiki.st.com/stm32mcu/wiki/STM32CubeMonitor:How_to_extract_address_from_ELF_files>;
CoreSight ITM/SWO <https://pyocd.io/docs/swo_swv.html>;
MIPI SyS-T <https://www.mipi.org/specifications/sys-t>; CBOR (RFC 8949) <https://www.rfc-editor.org/info/rfc8949/>.
