"""Real-signal probes: wire the adaptive capabilities to the actual machine.

The "smart" layer (allocator / DVFS / telemetry ring) was modeled against the cost
model. This module reads the *real* substrate so those decisions and measurements
use silicon, not assumptions -- and it is scrupulously honest about what a given
host (or sandbox) exposes:

  * **cache topology** (`/sys/devices/system/cpu/cpu0/cache`): real L1/L2/L3 sizes,
    so the allocator's fast-tier capacities are the machine's, not a guess. (Read.)
  * **cpufreq** (`/sys/.../cpufreq`, `/proc/cpuinfo`): the nominal frequency and, if
    present, the discrete scaling steps + governor. SETTING frequency needs a
    `userspace` governor + privilege; we report whether actuation is possible rather
    than pretend. (Read; actuation gated.)
  * **OS counters** (`resource.getrusage`, `time.*_ns`): real page faults, context
    switches, and CPU-/wall-nanoseconds to feed the telemetry ring. Hardware PMU
    counters (`perf_event_open`) are used when available and degrade gracefully when
    the guest exposes no PMU (common in cloud VMs).

Every probe is read-only and returns ``None`` / falls back cleanly when a signal is
absent, so importing or calling this never fails on a minimal host. `summary()`
reports the real/unavailable split for honest provenance.
"""

from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass

from .kbcir.cost import MemTier

_SYS_CPU = "/sys/devices/system/cpu/cpu0"


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _parse_size(text: str | None) -> int | None:
    """'48K' / '2048K' / '512' -> bytes."""
    if not text:
        return None
    text = text.strip()
    mult = {"K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}.get(text[-1].upper())
    try:
        return int(text[:-1]) * mult if mult else int(text)
    except ValueError:
        return None


# --- cache topology --------------------------------------------------------------

@dataclass(frozen=True)
class CacheLevel:
    level: int
    kind: str          # "Data" | "Instruction" | "Unified"
    size_bytes: int


def cache_topology() -> tuple[CacheLevel, ...]:
    """The real CPU cache levels from /sys, or () if unreadable."""
    out: list[CacheLevel] = []
    base = f"{_SYS_CPU}/cache"
    try:
        entries = sorted(d for d in os.listdir(base) if d.startswith("index"))
    except OSError:
        return ()
    for d in entries:
        level = _read(f"{base}/{d}/level")
        kind = _read(f"{base}/{d}/type")
        size = _parse_size(_read(f"{base}/{d}/size"))
        if level and kind and size:
            out.append(CacheLevel(level=int(level), kind=kind, size_bytes=size))
    return tuple(out)


def tier_capacities() -> dict[MemTier, int] | None:
    """Real fast-tier byte capacities {L1, L2, L3} from the cache topology, for the
    allocator. L1 uses the data cache; each level takes its largest reported cache.
    Returns None when /sys cache is unreadable (caller falls back to its defaults)."""
    topo = cache_topology()
    if not topo:
        return None
    caps: dict[MemTier, int] = {}
    for lvl, tier in ((1, MemTier.L1), (2, MemTier.L2), (3, MemTier.L3)):
        at_level = [c for c in topo if c.level == lvl]
        if lvl == 1:
            data = [c for c in at_level if c.kind == "Data"] or at_level
            at_level = data
        if at_level:
            caps[tier] = max(c.size_bytes for c in at_level)
    return caps or None


# --- cpufreq (read; actuation gated) ---------------------------------------------

@dataclass(frozen=True)
class CpuFreq:
    nominal_khz: int | None
    steps_khz: tuple[int, ...]          # discrete scaling steps ((),) if none exposed
    governor: str | None
    actuatable: bool                    # can we set the frequency here?

    @property
    def available(self) -> bool:
        return self.nominal_khz is not None


def cpufreq_info() -> CpuFreq:
    """Read the real cpufreq table + governor; report whether DVFS can be actuated.
    Actuation needs the `userspace` governor and a writable `scaling_setspeed`
    (root) -- absent in most sandboxes, so we never pretend to have set a clock."""
    fq = f"{_SYS_CPU}/cpufreq"
    steps_raw = _read(f"{fq}/scaling_available_frequencies")
    steps = tuple(int(x) for x in steps_raw.split()) if steps_raw else ()
    governor = _read(f"{fq}/scaling_governor")
    nominal = _read(f"{fq}/cpuinfo_max_freq")
    nominal_khz = int(nominal) if nominal else _cpuinfo_khz()
    setspeed = f"{fq}/scaling_setspeed"
    actuatable = governor == "userspace" and os.access(setspeed, os.W_OK)
    return CpuFreq(nominal_khz=nominal_khz, steps_khz=steps,
                   governor=governor, actuatable=actuatable)


def _cpuinfo_khz() -> int | None:
    """Fallback nominal frequency from /proc/cpuinfo 'cpu MHz' (read-only)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    return int(float(line.split(":")[1]) * 1000)
    except OSError:
        pass
    return None


# --- OS / hardware counters (the telemetry-ring feed) ----------------------------

@dataclass(frozen=True)
class Counters:
    """A real measurement snapshot delta (all genuine OS/timer counters)."""

    wall_ns: int
    cpu_ns: int
    minor_faults: int
    major_faults: int
    vol_ctx: int
    invol_ctx: int


class CounterSampler:
    """Sample real OS counters (getrusage + monotonic/CPU clocks) around a unit of
    work -- the actual perf signal that feeds the telemetry ring."""

    def __init__(self):
        self._w = time.perf_counter_ns()
        self._c = time.process_time_ns()
        self._r = resource.getrusage(resource.RUSAGE_SELF)

    def lap(self) -> Counters:
        w, c = time.perf_counter_ns(), time.process_time_ns()
        r = resource.getrusage(resource.RUSAGE_SELF)
        out = Counters(
            wall_ns=w - self._w, cpu_ns=c - self._c,
            minor_faults=r.ru_minflt - self._r.ru_minflt,
            major_faults=r.ru_majflt - self._r.ru_majflt,
            vol_ctx=r.ru_nvcsw - self._r.ru_nvcsw,
            invol_ctx=r.ru_nivcsw - self._r.ru_nivcsw)
        self._w, self._c, self._r = w, c, r
        return out


def perf_counters_available() -> bool:
    """True iff the guest exposes a hardware PMU via perf_event_open (else we use the
    OS counters above). Best-effort; never raises."""
    return _perf_open(0) is not None


# --- hardware PMU counters (perf_event_open; degrades to None with no PMU) --------

_PERF_HW = {"cycles": 0, "instructions": 1, "cache_refs": 2, "cache_misses": 3,
            "branches": 4, "branch_misses": 5}
_IOC_ENABLE, _IOC_DISABLE, _IOC_RESET = 0x2400, 0x2401, 0x2403
_SYS_perf_event_open = 298  # x86_64


def _perf_open(config: int):
    """Open one user-space hardware counter; return an fd (int) or None if no PMU /
    not permitted. Caller closes the fd."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)

        class _attr(ctypes.Structure):
            _fields_ = [("type", ctypes.c_uint32), ("size", ctypes.c_uint32),
                        ("config", ctypes.c_uint64), ("sample_period", ctypes.c_uint64),
                        ("sample_type", ctypes.c_uint64), ("read_format", ctypes.c_uint64),
                        ("flags", ctypes.c_uint64), ("wakeup", ctypes.c_uint32),
                        ("bp_type", ctypes.c_uint32), ("bp_addr", ctypes.c_uint64),
                        ("bp_len", ctypes.c_uint64)]
        a = _attr(); a.type = 0; a.size = ctypes.sizeof(a); a.config = config
        a.flags = (1 << 0) | (1 << 20)            # disabled + exclude_kernel
        fd = libc.syscall(_SYS_perf_event_open, ctypes.byref(a), 0, -1, -1, 0)
        return int(fd) if fd >= 0 else None
    except Exception:
        return None


@dataclass(frozen=True)
class HwSample:
    cycles: int
    instructions: int
    cache_misses: int

    @property
    def ipc_milli(self) -> int:
        return (self.instructions * 1000) // max(1, self.cycles)


def read_hw_counters(work) -> HwSample | None:
    """Run `work()` under real hardware PMU counters (cycles / instructions /
    cache-misses, user-space). Returns an `HwSample`, or **None when the host exposes
    no PMU** (e.g. most cloud guests: perf_event_open -> ENOENT) -- and in that case
    `work()` is NOT run, so the caller can run it once under the OS-counter fallback.
    Closes every fd it opens."""
    import ctypes
    fds = {name: _perf_open(cfg) for name, cfg in
           (("cycles", 0), ("instructions", 1), ("cache_misses", 3))}
    if any(fd is None for fd in fds.values()):     # no PMU: do NOT run work here
        for fd in fds.values():
            if fd is not None:
                os.close(fd)
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        for fd in fds.values():
            libc.ioctl(fd, _IOC_RESET, 0)
            libc.ioctl(fd, _IOC_ENABLE, 0)
        work()                                     # committed: work runs exactly once
        vals = {}
        for name, fd in fds.items():
            libc.ioctl(fd, _IOC_DISABLE, 0)
            try:
                vals[name] = int.from_bytes(os.read(fd, 8), "little", signed=False)
            except OSError:
                vals[name] = 0
        return HwSample(cycles=vals["cycles"], instructions=vals["instructions"],
                        cache_misses=vals["cache_misses"])
    finally:
        for fd in fds.values():
            os.close(fd)


# --- RAPL energy + on-die thermal (the real power/heat signals) ------------------
#
# These close the calibration loop's *physical* half: RAPL gives the package energy
# (Joules) consumed by a unit of work, and the thermal zones give the die
# temperature -- the real drivers of the thermal/power pressure Theta that flips
# vec16 -> vec8. Both are read-only and degrade to None on a host that does not
# expose them (most cloud guests hide RAPL; many hide coretemp), so a measured
# calibration on real silicon lights up exactly the same code path that reports
# "unavailable" here.

_RAPL_BASE = "/sys/class/powercap"
_THERMAL_BASE = "/sys/class/thermal"


def rapl_available() -> bool:
    """True iff a readable RAPL package energy counter is exposed."""
    return _rapl_energy_path() is not None


def _rapl_energy_path() -> str | None:
    """The first readable intel-rapl package `energy_uj` (or None)."""
    try:
        for d in sorted(os.listdir(_RAPL_BASE)):
            if d.startswith("intel-rapl:"):
                p = f"{_RAPL_BASE}/{d}/energy_uj"
                if _read(p) is not None:
                    return p
    except OSError:
        pass
    return None


def read_rapl_uj() -> int | None:
    """Current RAPL package energy counter in microjoules, or None if unreadable.
    The counter wraps at `max_energy_range_uj`; use `RaplSampler` for deltas."""
    p = _rapl_energy_path()
    if p is None:
        return None
    v = _read(p)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


class RaplSampler:
    """Measure real package energy consumed across a unit of work (microjoules),
    handling the counter wraparound at `max_energy_range_uj`. Returns None deltas on
    a host with no RAPL -- never a fabricated figure."""

    def __init__(self):
        self._path = _rapl_energy_path()
        self._max = self._read_max()
        self._start = self._now()

    def _read_max(self) -> int | None:
        if self._path is None:
            return None
        v = _read(self._path.replace("energy_uj", "max_energy_range_uj"))
        try:
            return int(v) if v else None
        except ValueError:
            return None

    def _now(self) -> int | None:
        v = _read(self._path) if self._path else None
        try:
            return int(v) if v is not None else None
        except ValueError:
            return None

    @property
    def available(self) -> bool:
        return self._start is not None

    def lap_uj(self) -> int | None:
        """Microjoules since the last lap (or construction); None if no RAPL."""
        now = self._now()
        if now is None or self._start is None:
            return None
        delta = now - self._start
        if delta < 0 and self._max:                  # the counter wrapped
            delta += self._max
        self._start = now
        return max(0, delta)


_CPU_ZONE_TYPES = ("x86_pkg_temp", "coretemp", "cpu", "soc")


def read_thermal_millideg() -> int | None:
    """The CPU/package die temperature in milli-degrees Celsius, or None. Prefers a
    zone typed as the CPU package (x86_pkg_temp / coretemp / cpu / soc) so a hot but
    irrelevant zone (battery, wifi, ambient) cannot drive Theta; falls back to the
    max across all zones only when no CPU zone is typed."""
    try:
        zones = [d for d in os.listdir(_THERMAL_BASE) if d.startswith("thermal_zone")]
    except OSError:
        return None

    def _temp(z: str) -> int | None:
        t = _read(f"{_THERMAL_BASE}/{z}/temp")
        try:
            return int(t) if t is not None else None
        except ValueError:
            return None

    cpu_temps, all_temps = [], []
    for z in zones:
        v = _temp(z)
        if v is None:
            continue
        all_temps.append(v)
        ztype = (_read(f"{_THERMAL_BASE}/{z}/type") or "").lower()
        if any(tag in ztype for tag in _CPU_ZONE_TYPES):
            cpu_temps.append(v)
    pool = cpu_temps or all_temps                 # prefer CPU zones; else any zone
    return max(pool) if pool else None


# Thermal pressure mapping: 40 C (idle) -> 0, 95 C (throttle) -> 100, clamped. The
# same 0..100 scale Theta.thermal uses, so a measured temperature drives the replan.
_THERMAL_FLOOR_C = 40
_THERMAL_CEIL_C = 95


def thermal_pressure() -> int | None:
    """The real die temperature mapped to Theta's 0..100 thermal pressure, or None
    when no thermal zone is exposed (sandbox)."""
    milli = read_thermal_millideg()
    if milli is None:
        return None
    c = milli / 1000.0
    span = _THERMAL_CEIL_C - _THERMAL_FLOOR_C
    return max(0, min(100, int(round((c - _THERMAL_FLOOR_C) * 100 / span))))


def silicon_dna(work, claim_id: int = 0):
    """Run `work()` once under every real signal the host exposes (PMU cache-misses,
    RAPL energy, on-die temperature, OS utilization) and assemble a `DataDNA` -- the
    *measured* telemetry that feeds the calibration loop. Returns (DataDNA,
    provenance) where `provenance` is the set of signals that were genuinely real
    (e.g. {"pmu", "rapl", "thermal", "os"}); a signal absent on this host falls back
    to a derived/zero value and is omitted from the set (honest, never faked)."""
    from .telemetry import DataDNA

    provenance: set[str] = set()
    rapl = RaplSampler()
    sampler = CounterSampler()
    hw = read_hw_counters(work)                       # runs work() iff a PMU is present
    if hw is None:
        work()                                       # no PMU: run once under OS counters
    else:
        provenance.add("pmu")
    c = sampler.lap()
    energy_uj = rapl.lap_uj()
    if energy_uj is not None:
        provenance.add("rapl")
    if "os" not in provenance:
        provenance.add("os")
    therm = thermal_pressure()
    thermal = therm if therm is not None else 0
    if therm is not None:
        provenance.add("thermal")
    misses = hw.cache_misses if hw else (c.minor_faults + c.major_faults)
    util = min(100, (c.cpu_ns * 100) // max(1, c.wall_ns))
    # Voltage axis (reserved): a coarse power proxy from RAPL (uJ over wall-ns ->
    # milliwatts), clamped to 0..100; 0 when no RAPL.
    voltage = 0
    if energy_uj is not None and c.wall_ns > 0:
        voltage = min(100, (energy_uj * 1000) // max(1, c.wall_ns))
    dna = DataDNA(segment_id="silicon", claim_id=claim_id, cycles=(hw.cycles if hw else c.cpu_ns),
                  bytes=0, misses=misses, thermal=thermal, voltage=int(voltage), utilization=util)
    return dna, provenance


# --- the telemetry-ring feed -----------------------------------------------------

def sample_into_ring(ring, claim_id: int, work) -> Counters:
    """Run `work()` once, measure it, and write one record into the zero-copy
    telemetry ring. When a hardware PMU is present, the real cycle + cache-miss
    counts are used (`misses` carries L*-miss counts); otherwise `cycles` = measured
    CPU nanoseconds and `misses` = page faults. The kernel writes, the consumer
    `drain()`s the same buffer -- a real measured signal."""
    from .telemetry import DataDNA
    sampler = CounterSampler()
    hw = read_hw_counters(work)              # runs work() iff a PMU is present
    if hw is None:
        work()                              # no PMU: run once under OS counters
    c = sampler.lap()
    cycles = hw.cycles if hw else c.cpu_ns
    misses = hw.cache_misses if hw else (c.minor_faults + c.major_faults)
    ring.write(DataDNA(segment_id="", claim_id=claim_id, cycles=cycles, bytes=0,
                       misses=misses, thermal=0, voltage=0,
                       utilization=min(100, (c.cpu_ns * 100) // max(1, c.wall_ns))))
    return c


# --- honest provenance -----------------------------------------------------------

def summary() -> dict:
    """What the current host actually exposes -- the real/unavailable split."""
    topo = cache_topology()
    fq = cpufreq_info()
    return {
        "cache_levels": [(c.level, c.kind, c.size_bytes) for c in topo],
        "tier_capacities": {t.name: b for t, b in (tier_capacities() or {}).items()},
        "cpu_nominal_khz": fq.nominal_khz,
        "cpufreq_steps": list(fq.steps_khz),
        "cpufreq_governor": fq.governor,
        "dvfs_actuatable": fq.actuatable,
        "hw_pmu": perf_counters_available(),
        "os_counters": True,
        "rapl_energy": rapl_available(),
        "thermal_zone": read_thermal_millideg() is not None,
    }
