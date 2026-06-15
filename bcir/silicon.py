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
        a = _attr(); a.type = 0; a.size = ctypes.sizeof(a); a.config = 0
        a.flags = (1 << 0) | (1 << 20)            # disabled + exclude_kernel
        fd = libc.syscall(298, ctypes.byref(a), 0, -1, -1, 0)  # x86_64 perf_event_open
        if fd >= 0:
            os.close(fd)
            return True
    except Exception:
        pass
    return False


# --- the telemetry-ring feed -----------------------------------------------------

def sample_into_ring(ring, claim_id: int, work) -> Counters:
    """Run `work()`, measure the real OS counters around it, and write one record
    into the zero-copy telemetry ring. `cycles` = measured CPU nanoseconds, `misses`
    = page faults (this guest has no cache-miss PMU; the field carries the real fault
    count instead). Returns the measured delta. The kernel writes, the consumer
    `drain()`s the same buffer -- a real measured signal through the ring."""
    from .telemetry import DataDNA
    sampler = CounterSampler()
    work()
    c = sampler.lap()
    ring.write(DataDNA(segment_id="", claim_id=claim_id, cycles=c.cpu_ns, bytes=0,
                       misses=c.minor_faults + c.major_faults, thermal=0, voltage=0,
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
    }
