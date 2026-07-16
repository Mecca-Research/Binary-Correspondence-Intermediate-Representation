"""Vendor-neutral telemetry signal-provider registry — the PAPI-component model for BCIR.

The repo already READS real host signals in :mod:`bcir.silicon` (RAPL energy, PMU
counters, on-die thermal, cpufreq, cache topology, OS counters), but as a flat bag of
functions: no uniform typed interface, no provenance/units metadata carried on a
*definition*, and no registry/plugin seam to add vendor sources (NVML / amd-smi /
Redfish) without editing the core. This module formalizes that into a **registry of
typed signal providers**, mapped to the 12-D cost vector, with **honest ``None`` /
unavailable** when a source is genuinely absent. It is the T1 foundation the rest of the
telemetry pipeline (T2 UART frame, T3 derived metrics, T4 export) builds on. See
``docs/kernel/TELEMETRY_PIPELINE_RESEARCH.md`` §3/§4/§6 and ``docs/kernel/SIGNAL_REGISTRY.md``.

Design DNA copied from the survey (research §3):

  * **PAPI's component model** — one symbolic name (``"thermal.pressure"``) behind a
    stable typed interface; *components* (perf / rapl / nvml / rocm_smi / lm-sensors)
    plug in without the core knowing them in advance. This module is exactly that
    pattern, mirroring :mod:`bcir.channel_plugin`'s ``register_*`` seam.
  * **hwmon typed schema** — fixed units (m°C / µW / µJ / kHz …). A reading carries a
    :class:`Unit`, not a free-float number; absent source → honest ``None`` (already the
    way :mod:`bcir.silicon` behaves; we preserve it).
  * **Redfish 4-resource split** — provenance + units live on the *definition*
    (:class:`MetricDefinition`), NOT on every sample (:class:`Reading`). The definition is
    the taxonomy; the reading is one observation of it.
  * **``sampling_model``** (the research §4 gap) — ``polled`` / ``streamed`` /
    ``event_driven`` declared per definition (PMU is event-driven; DCGM streams; sysfs
    polls). Defaults to ``polled``.

**The governing invariant (the two-truth quarantine, applied to measurement).**
Telemetry is a *graded L2/L3 signal*. This registry READS signals and may inform plan
cost (the ``theta`` runtime state / cost-weight calibration) but it MUST NEVER be — or
alter — an R-law legality verdict, emit a Diagnostic, or sit on the legality path. There
is deliberately no ``verify`` / R-law surface here: a provider returns a
:class:`Reading` or ``None``, never a legality object. The decision/legality path
(``bcir/verify``) stays deterministic/integer/Q8 and never reads telemetry.

**Honest depth.** Every provider's ``available()`` / ``read()`` reflect the *real* host.
A source absent on this machine returns ``available() → False`` and ``read() → None`` —
NEVER a fabricated value. The vocabulary-completing gap providers (GPU power, BMC power,
memory bandwidth, fabric bytes, throttle state, reliability) carry a real
:class:`MetricDefinition` so the namespace is complete, but report unavailable on a host
without that backend — exactly like :mod:`bcir.channels`'s *modeled* future channels make
room for hardware that has no resident driver yet.

This module BUILDS ON :mod:`bcir.silicon`'s readers; it does not replace or modify them.
"""

from __future__ import annotations

import abc
import math
import threading
from dataclasses import dataclass, field

from . import silicon
from .kbcir.cost import DIMS

# --- Unit vocabulary (hwmon-schema-inspired fixed units) -----------------------------
# Plain string constants: a reading carries one of these so a consumer never has to guess
# the scale (m°C vs °C, µJ vs J). Matches the fixed-unit discipline of hwmon / Redfish.


class Unit:
    """The fixed, typed telemetry units. A :class:`Reading`'s value is in its
    definition's ``unit`` — no ambiguous bare floats."""

    PERCENT = "percent"            # 0..100 normalized pressure (Theta's scale)
    MILLICELSIUS = "millicelsius"  # m°C (hwmon temp* convention)
    MICROJOULE = "microjoule"      # µJ (RAPL energy_uj; a monotonic counter)
    MILLIWATT = "milliwatt"        # mW (Redfish/BMC power level)
    MICROWATT = "microwatt"        # µW (hwmon power* convention)
    KHZ = "khz"                    # kilohertz (cpufreq)
    BYTES = "bytes"                # bytes (cache capacities, transfer sizes)
    BYTES_PER_SECOND = "bytes_per_second"  # byte/s bandwidth/rate
    BITMASK = "bitmask"            # fixed-width state/reason bits
    COUNT = "count"                # a dimensionless count / capability flag (0 or 1)
    RATIO_MILLI = "ratio_milli"    # a ratio in milli-units (e.g. IPC*1000), Q-free
    NONE = "none"                  # no meaningful unit

    ALL = frozenset({PERCENT, MILLICELSIUS, MICROJOULE, MILLIWATT, MICROWATT,
                     KHZ, BYTES, BYTES_PER_SECOND, BITMASK, COUNT, RATIO_MILLI, NONE})


# --- Provenance + sampling model (carried on the definition, Redfish-style) ----------
# Provenance reuses the channel vocabulary (real | modeled | simulated): `real` is
# `MEASURED` here. This is HOW a reading was obtained, not WHETHER it is present.


class Provenance:
    """How a reading is obtained. Reuses the channel/calibration vocabulary
    (``real`` ↔ ``MEASURED``)."""

    MEASURED = "measured"     # read from real silicon / OS (channel `real`)
    MODELED = "modeled"       # derived from a model, not directly sensed
    SIMULATED = "simulated"   # produced by a simulator

    ALL = frozenset({MEASURED, MODELED, SIMULATED})


class SamplingModel:
    """How a source delivers values (research §4 gap). Most sysfs/OS sources are
    ``POLLED``; PMU is ``EVENT_DRIVEN``; DCGM/NVML telemetry ``STREAMED``."""

    POLLED = "polled"
    STREAMED = "streamed"
    EVENT_DRIVEN = "event_driven"

    ALL = frozenset({POLLED, STREAMED, EVENT_DRIVEN})


class MetricKind:
    """Whether a definition is an instantaneous level or an accumulating sum."""

    GAUGE = "gauge"
    COUNTER = "counter"
    ALL = frozenset({GAUGE, COUNTER})


class Temporality:
    """Aggregation interval semantics carried by a counter definition."""

    UNSPECIFIED = "unspecified"
    DELTA = "delta"
    CUMULATIVE = "cumulative"
    ALL = frozenset({UNSPECIFIED, DELTA, CUMULATIVE})


# --- the Redfish-style definition / sample split -------------------------------------


@dataclass(frozen=True)
class MetricDefinition:
    """The *definition* of a metric (Redfish ``MetricDefinition``): its stable taxonomy,
    units, cost-dim mapping, nominal provenance, and sampling model — the metadata, NOT a
    sample. Provenance/units live HERE, once, not on every reading.

    ``name`` is a stable dotted id (e.g. ``"thermal.pressure"``). ``cost_dim`` is the
    12-D cost dimension this metric calibrates (one of ``DIMS``), or ``None`` for a metric
    that informs no dim directly (e.g. a pure capability flag). ``provenance`` is the
    source's *nominal* provenance; the actual provenance of one read is carried on the
    :class:`Reading` (they agree for the measured providers here)."""

    name: str
    unit: str
    cost_dim: str | None
    provenance: str = Provenance.MEASURED
    sampling_model: str = SamplingModel.POLLED
    description: str = ""
    signal_id: int = 0
    metric_kind: str = MetricKind.GAUGE
    temporality: str = Temporality.UNSPECIFIED
    monotonic: bool = False
    min_interval_ns: int = 0

    def validate(self) -> list[str]:
        """Schema errors ([] == valid): unit in the vocab, cost_dim in DIMS-or-None,
        provenance + sampling_model in their vocabularies, name present."""
        errs: list[str] = []
        if not self.name:
            errs.append("name is required")
        if self.unit not in Unit.ALL:
            errs.append(f"unit {self.unit!r} not in {sorted(Unit.ALL)}")
        if self.cost_dim is not None and self.cost_dim not in DIMS:
            errs.append(f"cost_dim {self.cost_dim!r} not in DIMS or None")
        if self.provenance not in Provenance.ALL:
            errs.append(f"provenance {self.provenance!r} not in {sorted(Provenance.ALL)}")
        if self.sampling_model not in SamplingModel.ALL:
            errs.append(f"sampling_model {self.sampling_model!r} not in "
                        f"{sorted(SamplingModel.ALL)}")
        if (not isinstance(self.signal_id, int) or isinstance(self.signal_id, bool)
                or not 0 <= self.signal_id <= 0xFFFFFFFF):
            errs.append("signal_id must be an unsigned 32-bit integer")
        if self.metric_kind not in MetricKind.ALL:
            errs.append(f"metric_kind {self.metric_kind!r} not in {sorted(MetricKind.ALL)}")
        if self.temporality not in Temporality.ALL:
            errs.append(f"temporality {self.temporality!r} not in {sorted(Temporality.ALL)}")
        if not isinstance(self.monotonic, bool):
            errs.append("monotonic must be bool")
        if (not isinstance(self.min_interval_ns, int) or isinstance(self.min_interval_ns, bool)
                or not 0 <= self.min_interval_ns <= 0xFFFFFFFFFFFFFFFF):
            errs.append("min_interval_ns must be an unsigned 64-bit integer")
        if self.metric_kind == MetricKind.GAUGE:
            if self.temporality != Temporality.UNSPECIFIED:
                errs.append("gauge temporality must be unspecified")
            if self.monotonic:
                errs.append("a gauge cannot be monotonic")
        elif self.metric_kind == MetricKind.COUNTER:
            if self.temporality == Temporality.UNSPECIFIED:
                errs.append("counter temporality must be delta or cumulative")
            if not self.monotonic:
                errs.append("counter must be monotonic")
        return errs


@dataclass(frozen=True)
class Reading:
    """One sample (Redfish ``MetricReport`` entry): the metric definition, the observed
    ``value`` (in the definition's unit), and the *actual* provenance of THIS read. A
    provider returns a ``Reading`` only when the source is genuinely present;
    ``read() → None`` means the source is absent on this host (never a fabricated value)."""

    definition: MetricDefinition
    value: int | float
    provenance: str = Provenance.MEASURED

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def unit(self) -> str:
        return self.definition.unit

    @property
    def cost_dim(self) -> str | None:
        return self.definition.cost_dim


# --- the provider interface (PAPI component) -----------------------------------------


class SignalProvider(abc.ABC):
    """One telemetry source behind a stable typed interface (a PAPI *component*). Honest:
    ``available()`` and ``read()`` reflect the real host. ``read()`` returns a
    :class:`Reading` or ``None``. Availability can change between two calls, so consumers
    that need a coherent transaction use one ``read()``/registry snapshot rather than
    comparing separately sampled calls. A provider NEVER returns a legality object."""

    @property
    @abc.abstractmethod
    def definition(self) -> MetricDefinition:
        """The metric this provider supplies (units/dim/provenance/sampling on it)."""

    @abc.abstractmethod
    def available(self) -> bool:
        """True iff this source is genuinely present + readable on this host."""

    @abc.abstractmethod
    def read(self) -> Reading | None:
        """One sample, or ``None`` when the source is absent (never fabricated)."""

    @property
    def name(self) -> str:
        return self.definition.name


class _DefinedProvider(SignalProvider):
    """A provider with a fixed :class:`MetricDefinition` instance (the common case)."""

    _DEF: MetricDefinition

    @property
    def definition(self) -> MetricDefinition:
        return self._DEF

    def _reading(self, value: int | float) -> Reading:
        return Reading(self._DEF, value, self._DEF.provenance)


# --- concrete providers WRAPPING the point-readable silicon.py signals ----------------
# Each maps to a cost_dim. They WRAP silicon.py (do not duplicate or modify it) and
# preserve its honesty: absent source -> available()=False, read()=None.


class ThermalPressureProvider(_DefinedProvider):
    """The real die temperature mapped to Theta's 0..100 thermal pressure
    (:func:`silicon.thermal_pressure`). PERCENT, dim ``thermal``."""

    _DEF = MetricDefinition(
        "thermal.pressure", Unit.PERCENT, "thermal", Provenance.MEASURED,
        SamplingModel.POLLED,
        "On-die temperature mapped to Theta's 0..100 thermal pressure (silicon.thermal_pressure).",
        signal_id=1)

    def available(self) -> bool:
        return silicon.thermal_pressure() is not None

    def read(self) -> Reading | None:
        v = silicon.thermal_pressure()
        return None if v is None else self._reading(v)


class DieTempProvider(_DefinedProvider):
    """The CPU/package die temperature in milli-°C (:func:`silicon.read_thermal_millideg`).
    MILLICELSIUS, dim ``thermal``."""

    _DEF = MetricDefinition(
        "thermal.die_millicelsius", Unit.MILLICELSIUS, "thermal", Provenance.MEASURED,
        SamplingModel.POLLED,
        "CPU/package die temperature in milli-degrees Celsius (silicon.read_thermal_millideg).",
        signal_id=2)

    def available(self) -> bool:
        return silicon.read_thermal_millideg() is not None

    def read(self) -> Reading | None:
        v = silicon.read_thermal_millideg()
        return None if v is None else self._reading(v)


class RaplEnergyProvider(_DefinedProvider):
    """The RAPL package energy counter in microjoules (:func:`silicon.read_rapl_uj`).
    MICROJOULE, dim ``power``.

    This is a *monotonic counter* (an OpenTelemetry monotonic Sum), wrapping at
    ``max_energy_range_uj``; **watts is a DERIVED metric for T3** (energy delta / wall
    delta). Here we honestly expose the raw counter + its availability, not a fabricated
    power figure — the work-scoped delta lives in :class:`silicon.RaplSampler`."""

    _DEF = MetricDefinition(
        "power.rapl_energy_uj", Unit.MICROJOULE, "power", Provenance.MEASURED,
        SamplingModel.POLLED,
        "RAPL package energy counter, microjoules (monotonic; watts is a T3-derived metric).",
        signal_id=3, metric_kind=MetricKind.COUNTER,
        temporality=Temporality.CUMULATIVE, monotonic=True)

    def available(self) -> bool:
        return silicon.rapl_available()

    def read(self) -> Reading | None:
        v = silicon.read_rapl_uj()
        return None if v is None else self._reading(v)


class CpuFreqProvider(_DefinedProvider):
    """The nominal CPU frequency in kHz (:func:`silicon.cpufreq_info`). KHZ, dim
    ``compute``."""

    _DEF = MetricDefinition(
        "compute.cpu_nominal_khz", Unit.KHZ, "compute", Provenance.MEASURED,
        SamplingModel.POLLED,
        "Nominal CPU frequency in kHz from cpufreq / /proc/cpuinfo (silicon.cpufreq_info).",
        signal_id=4)

    def available(self) -> bool:
        return silicon.cpufreq_info().nominal_khz is not None

    def read(self) -> Reading | None:
        khz = silicon.cpufreq_info().nominal_khz
        return None if khz is None else self._reading(khz)


# Cache capacity: one provider per L1/L2/L3 tier (static topology). They share the
# tier_capacities() probe; each exposes its own tier's byte capacity.
from .kbcir.cost import MemTier  # noqa: E402 -- kept local to the cache providers


class CacheCapacityProvider(_DefinedProvider):
    """The byte capacity of one fast cache tier (L1/L2/L3) from the real cache topology
    (:func:`silicon.tier_capacities`). BYTES, dim ``memory``. Static topology (polled
    once); absent when ``/sys`` cache is unreadable."""

    _DEFS = {
        MemTier.L1: MetricDefinition(
            "memory.cache_l1_bytes", Unit.BYTES, "memory", Provenance.MEASURED,
            SamplingModel.POLLED, "L1 data-cache capacity in bytes (silicon.tier_capacities).",
            signal_id=5),
        MemTier.L2: MetricDefinition(
            "memory.cache_l2_bytes", Unit.BYTES, "memory", Provenance.MEASURED,
            SamplingModel.POLLED, "L2 cache capacity in bytes (silicon.tier_capacities).",
            signal_id=6),
        MemTier.L3: MetricDefinition(
            "memory.cache_l3_bytes", Unit.BYTES, "memory", Provenance.MEASURED,
            SamplingModel.POLLED, "L3 cache capacity in bytes (silicon.tier_capacities).",
            signal_id=7),
    }

    def __init__(self, tier: MemTier):
        if tier not in self._DEFS:
            raise ValueError(f"CacheCapacityProvider supports L1/L2/L3, not {tier!r}")
        self._tier = tier
        self._DEF = self._DEFS[tier]

    @property
    def definition(self) -> MetricDefinition:
        return self._DEF

    def _caps(self):
        return silicon.tier_capacities() or {}

    def available(self) -> bool:
        return self._tier in self._caps()

    def read(self) -> Reading | None:
        caps = self._caps()
        return self._reading(caps[self._tier]) if self._tier in caps else None


class PmuAvailabilityProvider(_DefinedProvider):
    """The hardware PMU *capability* (:func:`silicon.perf_counters_available`). COUNT
    (1 = a PMU is exposed, else absent), dim ``compute``.

    The actual PMU counters (cycles / instructions / cache-misses) are **WORK-SCOPED**:
    they only mean something measured *around a unit of work* via
    :func:`silicon.read_hw_counters` / :func:`silicon.silicon_dna`. A point-sample of a
    raw counter outside a work window is meaningless, so the registry exposes the
    *capability + definition* here and the work-scoped path stays in :mod:`bcir.silicon`
    (this module deliberately does NOT move it). ``read()`` returns ``1`` (COUNT) when a
    PMU is present, else ``None``."""

    _DEF = MetricDefinition(
        "compute.pmu_available", Unit.COUNT, "compute", Provenance.MEASURED,
        SamplingModel.EVENT_DRIVEN,
        "Hardware PMU capability flag (silicon.perf_counters_available); actual counters "
        "are work-scoped via silicon.read_hw_counters / silicon_dna and stay there.",
        signal_id=8)

    def available(self) -> bool:
        return silicon.perf_counters_available()

    def read(self) -> Reading | None:
        return self._reading(1) if silicon.perf_counters_available() else None


# --- vocabulary-completing UNAVAILABLE gap providers (research §4) -------------------
# Each carries a REAL MetricDefinition so the registry's namespace is complete, but
# reports unavailable on a host without that backend/hardware -- exactly like channels.py's
# *modeled* future channels. A future NVML / amd-smi / Redfish / PCM / DCGM backend fills
# these in. They NEVER fabricate a value: available()->False, read()->None.


class _UnavailableProvider(_DefinedProvider):
    """A gap provider: a complete definition, but no backend on this host. Honest by
    construction — ``available()`` is False and ``read()`` is None until a backend lands."""

    def available(self) -> bool:
        return False

    def read(self) -> Reading | None:
        return None


class GpuPowerProvider(_UnavailableProvider):
    """GPU board power (NVML / amd-smi). MICROJOULE-or-watts class energy → dim ``power``.
    Needs an NVML / amd-smi backend (no GPU telemetry binding here yet)."""

    _DEF = MetricDefinition(
        "power.gpu_energy_uj", Unit.MICROJOULE, "power", Provenance.MEASURED,
        SamplingModel.STREAMED,
        "GPU board energy via NVML/amd-smi (gap: needs a vendor GPU telemetry backend).",
        signal_id=9, metric_kind=MetricKind.COUNTER,
        temporality=Temporality.CUMULATIVE, monotonic=True)


class BmcPowerProvider(_UnavailableProvider):
    """Out-of-band node/chassis power via a BMC (Redfish). dim ``power``. The only
    OS-independent tier; needs a Redfish/IPMI backend."""

    _DEF = MetricDefinition(
        "power.bmc_node_milliwatt", Unit.MILLIWATT, "power", Provenance.MEASURED,
        SamplingModel.POLLED,
        "Out-of-band node/chassis power via Redfish/BMC (gap: needs an out-of-band backend).",
        signal_id=10)


class MemBandwidthProvider(_UnavailableProvider):
    """Memory (IMC / VRAM) bandwidth via Intel PCM uncore IMC or DCGM DRAM-active →
    dim ``memory``. Needs a PCM/DCGM backend (under-covered dim, research §4)."""

    _DEF = MetricDefinition(
        "memory.bandwidth_bytes_per_s", Unit.BYTES_PER_SECOND, "memory", Provenance.MEASURED,
        SamplingModel.STREAMED,
        "Memory/VRAM bandwidth via PCM IMC / DCGM DRAM-active (gap: needs a PCM/DCGM backend).",
        signal_id=11)


class FabricBytesProvider(_UnavailableProvider):
    """Interconnect bytes moved (NVLink / PCIe / UPI) → dim ``fabric``. Needs a
    DCGM / PCM backend (under-covered dim, research §4)."""

    _DEF = MetricDefinition(
        "fabric.interconnect_bytes", Unit.BYTES, "fabric", Provenance.MEASURED,
        SamplingModel.STREAMED,
        "Interconnect bytes (NVLink/PCIe/UPI) via DCGM/PCM (gap: needs a fabric-counter backend).",
        signal_id=12, metric_kind=MetricKind.COUNTER,
        temporality=Temporality.CUMULATIVE, monotonic=True)


class ThrottleStateProvider(_UnavailableProvider):
    """Throttle / clock-event-reason bits (HW slowdown, power-cap, thermal) via NVML
    ClocksEventReasons / amd-smi / RAPL → dim ``contention``. Needs a vendor backend."""

    _DEF = MetricDefinition(
        "contention.throttle_state", Unit.BITMASK, "contention", Provenance.MEASURED,
        SamplingModel.EVENT_DRIVEN,
        "Throttle / clock-event-reason bitfield (gap: needs NVML/amd-smi/RAPL throttle backend).",
        signal_id=13)


class ReliabilityProvider(_UnavailableProvider):
    """Reliability signals (ECC / XID / RAS / margin-drift / RUL) via NVML-DCGM ECC /
    amd-smi RAS / Synopsys SLM / proteanTecs → dim ``reliability``. Needs a backend."""

    _DEF = MetricDefinition(
        "reliability.ecc_rul", Unit.COUNT, "reliability", Provenance.MEASURED,
        SamplingModel.EVENT_DRIVEN,
        "Reliability (ECC/XID/RAS/margin-drift/RUL) via DCGM/amd-smi/SLM (gap: needs a backend).",
        signal_id=14)


class HwmonPowerProvider(_UnavailableProvider):
    """Shunt-monitor (hwmon ``power*`` / INA226-class) board power → dim ``power``. The
    ``energy_source="hwmon"`` channel's power source; needs an hwmon power-rail backend
    (silicon.py exposes RAPL + thermal sysfs, not a generic hwmon ``power*`` rail), so it
    reports unavailable here rather than pretend."""

    _DEF = MetricDefinition(
        "power.hwmon_microwatt", Unit.MICROWATT, "power", Provenance.MEASURED,
        SamplingModel.POLLED,
        "hwmon power* / INA226 shunt-monitor board power (gap: needs an hwmon power-rail backend).",
        signal_id=15)


# --- the registry --------------------------------------------------------------------


class SignalRegistry:
    """A registry of typed signal providers, keyed by ``definition.name`` (the PAPI
    namespace). The plugin seam mirrors :func:`bcir.channels.register_channel` /
    :mod:`bcir.channel_plugin`: a new vendor provider joins via :meth:`register` without
    editing the core. It is off the legality path: it only ever returns Readings/None +
    availability flags, never a verdict or a Diagnostic."""

    def __init__(self) -> None:
        self._providers: dict[str, SignalProvider] = {}
        self._providers_by_id: dict[int, SignalProvider] = {}
        self._lock = threading.RLock()

    def register(self, provider: SignalProvider) -> SignalProvider:
        """Add a provider (the plugin seam). Rejects a duplicate name and an invalid
        definition, so a malformed provider never silently joins the registry."""
        d = provider.definition
        errs = d.validate()
        if errs:
            raise ValueError(f"invalid provider definition {d.name!r}: {'; '.join(errs)}")
        with self._lock:
            if d.name in self._providers:
                raise ValueError(f"duplicate provider name {d.name!r}")
            if d.signal_id != 0 and d.signal_id in self._providers_by_id:
                other = self._providers_by_id[d.signal_id].definition.name
                raise ValueError(f"duplicate signal_id {d.signal_id} for {d.name!r} and {other!r}")
            # Name and stable-ID indices become visible as one transaction.
            self._providers[d.name] = provider
            if d.signal_id != 0:
                self._providers_by_id[d.signal_id] = provider
        return provider

    def get(self, name: str) -> SignalProvider | None:
        with self._lock:
            return self._providers.get(name)

    def get_by_id(self, signal_id: int) -> SignalProvider | None:
        """Resolve a stable nonzero built-in/vendor signal ID. ID 0 is intentionally
        local/unassigned and is therefore never indexed."""
        if (not isinstance(signal_id, int) or isinstance(signal_id, bool)
                or not 0 < signal_id <= 0xFFFFFFFF):
            return None
        with self._lock:
            return self._providers_by_id.get(signal_id)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._providers)

    def providers(self) -> list[SignalProvider]:
        with self._lock:
            return [self._providers[n] for n in sorted(self._providers)]

    def providers_for_dim(self, dim: str) -> list[SignalProvider]:
        """Every registered provider whose definition maps to cost dimension ``dim``."""
        with self._lock:
            return [self._providers[n] for n in sorted(self._providers)
                    if self._providers[n].definition.cost_dim == dim]

    def snapshot(self) -> dict[str, Reading | None]:
        """One entry per provider: the current :class:`Reading`, or ``None`` when the
        source is absent (the :func:`silicon.summary` analog, with provenance on each
        Reading's definition). A present reading is validated at this provider boundary:
        its definition must match, provenance and numeric representation must be valid,
        counters cannot be negative, and percent pressures must be exact integers in
        ``0..100``. Honest — never a fabricated or silently coerced value."""
        with self._lock:
            providers = tuple((name, self._providers[name])
                              for name in sorted(self._providers))
        out: dict[str, Reading | None] = {}
        for name, provider in providers:
            reading = provider.read()
            if reading is not None:
                self._validate_reading(provider, reading)
            out[name] = reading
        return out

    @staticmethod
    def _validate_reading(provider: SignalProvider, reading: Reading) -> None:
        """Refuse malformed plugin output before it reaches exporters or ``theta``."""
        name = provider.definition.name
        if not isinstance(reading, Reading):
            raise ValueError(f"provider {name!r} returned {type(reading).__name__}, not Reading")
        if reading.definition != provider.definition:
            raise ValueError(f"provider {name!r} returned a reading for a different definition")
        if reading.provenance not in Provenance.ALL:
            raise ValueError(f"provider {name!r} returned invalid provenance "
                             f"{reading.provenance!r}")
        value = reading.value
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or (isinstance(value, float) and not math.isfinite(value))):
            raise ValueError(f"provider {name!r} returned a non-finite or non-numeric value")
        if reading.definition.metric_kind == MetricKind.COUNTER and value < 0:
            raise ValueError(f"provider {name!r} returned a negative monotonic counter")
        if reading.definition.unit == Unit.PERCENT:
            if not isinstance(value, int) or not 0 <= value <= 100:
                raise ValueError(
                    f"provider {name!r} percent pressure must be an integer in 0..100")

    def availability(self, snapshot: dict[str, Reading | None] | None = None) -> dict[str, bool]:
        """The coherent real/unavailable split derived from one canonical read snapshot.

        Pass an existing :meth:`snapshot` to guarantee the two views describe the same
        sample. With no argument this method takes one snapshot itself; it deliberately
        does not race a separate ``available()`` call against ``read()``."""
        snap = self.snapshot() if snapshot is None else snapshot
        return {name: reading is not None for name, reading in snap.items()}


# --- builders ------------------------------------------------------------------------

# The power providers selectable by a channel's energy_source (the channel↔provider map).
_POWER_PROVIDER_BY_ENERGY_SOURCE = {
    "rapl": RaplEnergyProvider,
    "nvml": GpuPowerProvider,
    "hwmon": HwmonPowerProvider,
    "none": None,
}


def _base_providers() -> list[SignalProvider]:
    """The full provider set (point-readable silicon wrappers + vocabulary-completing gap
    providers), excluding the energy_source-selected power provider, which the builders
    add. ``RaplEnergyProvider`` is the default power source (added in ``default_registry``)."""
    return [
        ThermalPressureProvider(),
        DieTempProvider(),
        CpuFreqProvider(),
        CacheCapacityProvider(MemTier.L1),
        CacheCapacityProvider(MemTier.L2),
        CacheCapacityProvider(MemTier.L3),
        PmuAvailabilityProvider(),
        # gap providers (vocabulary-completing; honest-unavailable here)
        GpuPowerProvider(),
        BmcPowerProvider(),
        MemBandwidthProvider(),
        FabricBytesProvider(),
        ThrottleStateProvider(),
        ReliabilityProvider(),
    ]


def default_registry() -> SignalRegistry:
    """Build the registry with every provider above registered, using RAPL as the default
    in-band power source (``RaplEnergyProvider``). On a host without RAPL the provider is
    still registered but reports unavailable (honest)."""
    reg = SignalRegistry()
    reg.register(RaplEnergyProvider())
    for p in _base_providers():
        reg.register(p)
    return reg


def registry_for_channel(channel) -> SignalRegistry:
    """Build the default registry but select the POWER provider matching the channel's
    ``runtime.energy_source`` — the channel↔provider mapping the research describes:

      * ``rapl``  → :class:`RaplEnergyProvider` (Intel in-band package energy)
      * ``nvml``  → :class:`GpuPowerProvider`   (GPU board power; gap-unavailable here)
      * ``hwmon`` → :class:`HwmonPowerProvider` (shunt-monitor rail; gap-unavailable here)
      * ``none``  → no dedicated power provider registered (honest: the channel reads no Joules)

    Every other provider is the same as :func:`default_registry`. Simple + honest: an
    unmapped energy_source falls back to RAPL (the in-band default)."""
    src = getattr(getattr(channel, "runtime", None), "energy_source", "rapl")
    reg = SignalRegistry()
    cls = _POWER_PROVIDER_BY_ENERGY_SOURCE.get(src, RaplEnergyProvider)
    if cls is not None:
        reg.register(cls())
    for p in _base_providers():
        # _base_providers includes the gap GpuPowerProvider; skip whichever power provider
        # the channel already selected so we don't register its name twice.
        if cls is not None and isinstance(p, cls):
            continue
        reg.register(p)
    return reg


def power_provider_for_energy_source(energy_source: str) -> str | None:
    """The metric name of the power provider a channel's ``energy_source`` selects, or
    ``None`` for ``"none"``. A read-only convenience for inspecting the mapping."""
    cls = _POWER_PROVIDER_BY_ENERGY_SOURCE.get(energy_source, RaplEnergyProvider)
    return None if cls is None else cls.__dict__["_DEF"].name


# --- the graded-signal convenience (two-truth boundary) ------------------------------


def theta_pressures(registry: SignalRegistry) -> dict[str, int]:
    """Read the registry's 0..100 PERCENT pressure signals (thermal, …) and return them
    keyed by their cost dim — the graded signal suitable to FEED ``theta`` (the runtime
    cost state). Read-only: this does NOT modify the legality path, the cost-vector DIMS,
    or the existing calibrate path. It only SURFACES the graded L2/L3 signal so a caller
    on the *cost/optimization* side can use it.

    **Two-truth boundary.** The returned pressures may inform plan cost / theta
    calibration; they may NEVER become or alter an R-law legality verdict. Absent signals
    are simply omitted (honest) — never defaulted to a fabricated pressure."""
    out: dict[str, int] = {}
    snapshot = registry.snapshot()
    for name in registry.names():
        d = registry.get(name).definition
        if d.unit != Unit.PERCENT or d.cost_dim is None:
            continue
        r = snapshot[name]
        if r is not None:
            # snapshot() has already proved this is an exact 0..100 integer. Do not
            # silently truncate a plugin float at the optimization boundary.
            out[d.cost_dim] = r.value
    return out
