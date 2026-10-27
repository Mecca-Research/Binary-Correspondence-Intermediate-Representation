"""Real-signal probes: wire the adaptive capabilities to the actual machine.

The "smart" layer (allocator / DVFS / telemetry ring) was modeled against the cost
model. This module reads the *real* substrate so those decisions and measurements
use silicon, not assumptions -- and it is scrupulously honest about what a given
host (or sandbox) exposes:

  * **cache topology** (`/sys/devices/system/cpu/cpu0/cache`): real L1/L2/L3 sizes,
    so the allocator's fast-tier capacities are the machine's, not a guess. (Read.)
  * **cpufreq** (`/sys/.../cpufreq`): the driver's `cpuinfo_max_freq` and, if present, the
    discrete scaling steps + governor. A host without a cpufreq driver (most cloud guests) has no
    frequency here: `/proc/cpuinfo`'s "cpu MHz" is the current or hypervisor-reported clock (on a
    KVM guest, the TSC's rate), never taken for a nominal one. SETTING frequency needs a
    `userspace` governor + privilege; we report whether actuation is possible rather than
    pretend. (Read; actuation gated.)
  * **OS counters** (`resource.getrusage` when available, plus `time.*_ns` everywhere):
    real page faults, context switches, and CPU-/wall-nanoseconds to feed the telemetry ring. Hardware PMU
    counters (`perf_event_open`) are used when available and degrade gracefully when
    the guest exposes no PMU (common in cloud VMs).

Every probe is read-only and returns ``None`` / falls back cleanly when a signal is
absent, so importing or calling this never fails on a minimal host. `summary()`
reports the real/unavailable split for honest provenance.

A value this host cannot read is ``None`` -- never 0 and never a nearby number: a counter the
kernel refused, never ran or ran for part of its window, a read that failed or came back short, an
energy counter that wrapped past a range nobody could read, a clock no cpufreq driver states. The
kernel's text is read in its own grammar (`_decimal`): ASCII digits, a '-' only where a value may
be negative, never through `int()` or `float()`, which take '+5', '1_000', 'inf' and non-ASCII
digits.
"""

from __future__ import annotations

import os
import re
import struct
import sys
import time
from dataclasses import dataclass

try:  # POSIX-only; timers remain available on Windows and other minimal hosts.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised on native Windows
    _resource = None

from .kbcir.cost import MemTier

_SYS_CPU = "/sys/devices/system/cpu/cpu0"


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


_UNSIGNED = re.compile(r"[0-9]+", re.ASCII)
_SIGNED = re.compile(r"-?[0-9]+", re.ASCII)
_SIZE = re.compile(r"([0-9]+)([KMGkmg]?)", re.ASCII)
_SIZE_UNIT = {"": 1, "K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}


def _decimal(text: str | None, *, signed: bool = False) -> int | None:
    """An integer the kernel wrote, in its own grammar -- ASCII digits, with a leading '-' only
    where the file may hold a negative value (`signed`) -- or None. Never `int()`: it takes '+5',
    '1_000', surrounding blanks and non-ASCII digits, a larger language than sysfs writes."""
    if text is None or not (_SIGNED if signed else _UNSIGNED).fullmatch(text):
        return None
    return int(text)


def _parse_size(text: str | None) -> int | None:
    """'48K' / '2048K' / '512' -> bytes; None for any other spelling."""
    match = _SIZE.fullmatch(text.strip()) if text else None
    if match is None:
        return None
    return int(match.group(1)) * _SIZE_UNIT[match.group(2).upper()]


# --- cache topology --------------------------------------------------------------


@dataclass(frozen=True)
class CacheLevel:
    level: int
    kind: str  # "Data" | "Instruction" | "Unified"
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
        level = _decimal(_read(f"{base}/{d}/level"))
        kind = _read(f"{base}/{d}/type")
        size = _parse_size(_read(f"{base}/{d}/size"))
        if level and kind and size:
            out.append(CacheLevel(level=level, kind=kind, size_bytes=size))
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
    nominal_khz: int | None  # cpufreq's cpuinfo_max_freq; None without a cpufreq driver
    steps_khz: tuple[int, ...]  # discrete scaling steps (() if none exposed)
    governor: str | None
    actuatable: bool  # can we set the frequency here?

    @property
    def available(self) -> bool:
        return self.nominal_khz is not None


def cpufreq_info() -> CpuFreq:
    """Read the real cpufreq table + governor; report whether DVFS can be actuated.
    Actuation needs the `userspace` governor and a writable `scaling_setspeed`
    (root) -- absent in most sandboxes, so we never pretend to have set a clock.

    The frequency is the cpufreq driver's `cpuinfo_max_freq` (its maximum: on intel_pstate the
    turbo ceiling) or None. There is no fallback: `/proc/cpuinfo`'s "cpu MHz" is the core's current
    clock on bare metal and the hypervisor's reported one in a guest (on a KVM guest, the TSC's
    rate), and neither is a nominal frequency. A scaling table with one malformed entry is no
    table: every step or none."""
    fq = f"{_SYS_CPU}/cpufreq"
    steps_raw = _read(f"{fq}/scaling_available_frequencies")
    steps = tuple(_decimal(x) for x in steps_raw.split()) if steps_raw else ()
    if None in steps or 0 in steps:
        steps = ()
    governor = _read(f"{fq}/scaling_governor")
    nominal_khz = _decimal(_read(f"{fq}/cpuinfo_max_freq")) or None  # 0 kHz is no frequency
    setspeed = f"{fq}/scaling_setspeed"
    actuatable = governor == "userspace" and os.access(setspeed, os.W_OK)
    return CpuFreq(
        nominal_khz=nominal_khz, steps_khz=steps, governor=governor, actuatable=actuatable
    )


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
        self._r = _resource.getrusage(_resource.RUSAGE_SELF) if _resource is not None else None

    def lap(self) -> Counters:
        w, c = time.perf_counter_ns(), time.process_time_ns()
        r = _resource.getrusage(_resource.RUSAGE_SELF) if _resource is not None else None
        out = Counters(
            wall_ns=w - self._w,
            cpu_ns=c - self._c,
            minor_faults=(r.ru_minflt - self._r.ru_minflt)
            if r is not None and self._r is not None
            else 0,
            major_faults=(r.ru_majflt - self._r.ru_majflt)
            if r is not None and self._r is not None
            else 0,
            vol_ctx=(r.ru_nvcsw - self._r.ru_nvcsw) if r is not None and self._r is not None else 0,
            invol_ctx=(r.ru_nivcsw - self._r.ru_nivcsw)
            if r is not None and self._r is not None
            else 0,
        )
        self._w, self._c, self._r = w, c, r
        return out


def perf_counters_available() -> bool:
    """True iff the guest exposes a hardware PMU via perf_event_open (else we use the
    OS counters above). Best-effort; never raises."""
    return _perf_open(0) is not None


# --- hardware PMU counters (perf_event_open; degrades to None with no PMU) --------

_PERF_HW = {
    "cycles": 0,
    "instructions": 1,
    "cache_refs": 2,
    "cache_misses": 3,
    "branches": 4,
    "branch_misses": 5,
}
_IOC_ENABLE, _IOC_DISABLE, _IOC_RESET = 0x2400, 0x2401, 0x2403

# perf_event_attr's flag word, the u64 after read_format (include/uapi/linux/perf_event.h).
PERF_ATTR_DISABLED = 1 << 0
PERF_ATTR_EXCLUDE_KERNEL = 1 << 5
PERF_ATTR_EXCLUDE_HV = 1 << 6
PERF_ATTR_EXCLUDE_GUEST = 1 << 20  # not ours: named so the bit it was mistaken for has a name
#: A user-space counter, created disabled: what `perf stat -e cycles:u` asks for, and the only kind
#: an unprivileged process may open at the default perf_event_paranoid=2.
_PERF_USER_FLAGS = PERF_ATTR_DISABLED | PERF_ATTR_EXCLUDE_KERNEL | PERF_ATTR_EXCLUDE_HV
# read_format: a read returns the count, then the time the event was enabled and the time it ran.
PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
_PERF_READ_FORMAT = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING
#: perf_event_attr through bp_len (PERF_ATTR_SIZE_VER1, 72 bytes): type, size, config,
#: sample_period, sample_type, read_format, the flag word, wakeup, bp_type, bp_addr, bp_len.
_PERF_ATTR = struct.Struct("=IIQQQQQIIQQ")
_PERF_READING = struct.Struct("=QQQ")  # value, time_enabled, time_running
#: The kernel allocates the flag word's bitfields from bit 0 on a little-endian host and from the
#: top on a big-endian one; the probe opens nothing on the latter rather than ask for other bits.
_BYTE_ORDER = sys.byteorder


# The perf_event_open syscall number is NOT portable (298 x86_64, 241 aarch64/riscv64, 364 armv7).
# It is owned by the host's HardwareChannel runtime -- a single source of truth, so x86's and ARM's
# ABI rules never collide here (the regression #255 fixed, now structural). See bcir/channels.py.
def _perf_syscall_nr() -> int:
    from .channels import host_perf_syscall_nr

    return host_perf_syscall_nr()


_SYS_perf_event_open = _perf_syscall_nr()  # from the host channel (the ABI differs per machine)


def _perf_attr(config: int) -> bytes:
    """One user-space hardware counter's perf_event_attr, as bytes: PERF_TYPE_HARDWARE `config`,
    created disabled, excluding the kernel and the hypervisor, reading its enabled and running
    times beside its count."""
    return _PERF_ATTR.pack(
        0, _PERF_ATTR.size, config, 0, 0, _PERF_READ_FORMAT, _PERF_USER_FLAGS, 0, 0, 0, 0
    )


def _perf_syscall(attr: bytes) -> int:
    """perf_event_open(attr, this thread, any CPU, no group, no flags): an fd, or -1."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    return int(
        libc.syscall(
            _SYS_perf_event_open, ctypes.create_string_buffer(attr, len(attr)), 0, -1, -1, 0
        )
    )


def _perf_ioctl(fd: int, request: int) -> None:
    import ctypes

    ctypes.CDLL(None, use_errno=True).ioctl(fd, request, 0)


def _perf_read(fd: int) -> bytes:
    return os.read(fd, _PERF_READING.size)


def _perf_close(fd: int) -> None:
    os.close(fd)


def _perf_open(config: int) -> int | None:
    """Open one user-space hardware counter; return an fd (int) or None if no PMU /
    not permitted. Caller closes the fd."""
    if _BYTE_ORDER != "little":
        return None
    try:
        fd = _perf_syscall(_perf_attr(config))
    except Exception:
        return None
    return fd if fd >= 0 else None


def _perf_count(fd: int) -> int | None:
    """One counter's count, or None: a read that failed or came back short, a counter the kernel
    never ran (its value 0 though nothing was counted), or one it ran for only part of the time it
    was enabled (multiplexed: the value is a sample of the work, not its count)."""
    try:
        raw = _perf_read(fd)
    except OSError:
        return None
    if len(raw) != _PERF_READING.size:
        return None
    value, enabled, running = _PERF_READING.unpack(raw)
    if running == 0 or running != enabled:
        return None
    return value


@dataclass(frozen=True)
class HwSample:
    """One run's user-space PMU counts. A count this host could not read is None, never 0."""

    cycles: int | None
    instructions: int | None
    cache_misses: int | None

    @property
    def ipc_milli(self) -> int | None:
        """Instructions per thousand cycles; None unless both counts were read and cycles moved."""
        if self.cycles is None or self.instructions is None or self.cycles <= 0:
            return None
        return (self.instructions * 1000) // self.cycles


def pmu_counts(sample: HwSample | None) -> tuple[int, int, int] | None:
    """(cycles, instructions, cache misses) when `sample` read all three, else None: the one test
    every record makes before it carries PMU counts, so none carries two counts and a gap."""
    if sample is None or None in (sample.cycles, sample.instructions, sample.cache_misses):
        return None
    return sample.cycles, sample.instructions, sample.cache_misses


def read_hw_counters(work) -> HwSample | None:
    """Run `work()` under real hardware PMU counters (cycles / instructions /
    cache-misses, user-space). Returns an `HwSample`, or **None when the host exposes
    no PMU** (e.g. most cloud guests: perf_event_open -> ENOENT) -- and in that case
    `work()` is NOT run, so the caller can run it once under the OS-counter fallback.
    Once the counters are open, `work()` runs exactly once and a sample comes back, each count in
    it None where the counter could not be read (`_perf_count`). Closes every fd it opens."""
    fds = {
        name: _perf_open(cfg)
        for name, cfg in (("cycles", 0), ("instructions", 1), ("cache_misses", 3))
    }
    if any(fd is None for fd in fds.values()):  # no PMU: do NOT run work here
        for fd in fds.values():
            if fd is not None:
                _perf_close(fd)
        return None
    try:
        for fd in fds.values():
            _perf_ioctl(fd, _IOC_RESET)
            _perf_ioctl(fd, _IOC_ENABLE)
        work()  # committed: work runs exactly once
        counts = {}
        for name, fd in fds.items():
            _perf_ioctl(fd, _IOC_DISABLE)
            counts[name] = _perf_count(fd)
        return HwSample(**counts)
    finally:
        for fd in fds.values():
            _perf_close(fd)


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
    return None if p is None else _decimal(_read(p))


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
        return _decimal(_read(self._path.replace("energy_uj", "max_energy_range_uj"))) or None

    def _now(self) -> int | None:
        return _decimal(_read(self._path)) if self._path else None

    @property
    def available(self) -> bool:
        """Whether the next lap has a start: RAPL is readable and its last read succeeded."""
        return self._start is not None

    def lap_uj(self) -> int | None:
        """Microjoules since the last lap (or construction), or None: no RAPL; a read that failed,
        and the lap after it, which has no start (the energy of a failed lap is in no lap); or a
        wrap whose range is unreadable, or a reading past the range it names. A counter that
        wrapped once gains the range back -- the powercap convention, exact to one energy unit; a
        lap longer than a whole wrap cannot be told from a shorter one, so lap within it."""
        now = self._now()
        start, self._start = self._start, now
        if now is None or start is None:
            return None
        delta = now - start
        if delta >= 0:
            return delta
        if self._max is None or now > self._max or start > self._max:
            return None
        return delta + self._max


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
        return _decimal(_read(f"{_THERMAL_BASE}/{z}/temp"), signed=True)  # below 0 C reads negative

    cpu_temps, all_temps = [], []
    for z in zones:
        v = _temp(z)
        if v is None:
            continue
        all_temps.append(v)
        ztype = (_read(f"{_THERMAL_BASE}/{z}/type") or "").lower()
        if any(tag in ztype for tag in _CPU_ZONE_TYPES):
            cpu_temps.append(v)
    pool = cpu_temps or all_temps  # prefer CPU zones; else any zone
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
    to a derived/zero value and is omitted from the set (honest, never faked). Cycles and
    misses are the PMU's only when it counted both (`pmu_counts`), else both the OS's."""
    from .telemetry import DataDNA

    provenance: set[str] = set()
    rapl = RaplSampler()
    sampler = CounterSampler()
    hw = read_hw_counters(work)  # runs work() iff a PMU is present
    if hw is None:
        work()  # no PMU: run once under OS counters
    counts = pmu_counts(hw)
    if counts is not None:
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
    if counts is not None:
        cycles, misses = counts[0], counts[2]
    else:
        cycles, misses = c.cpu_ns, c.minor_faults + c.major_faults
    util = min(100, (c.cpu_ns * 100) // max(1, c.wall_ns))
    # Voltage axis (reserved): a coarse power proxy from RAPL (uJ over wall-ns ->
    # milliwatts), clamped to 0..100; 0 when no RAPL.
    voltage = 0
    if energy_uj is not None and c.wall_ns > 0:
        voltage = min(100, (energy_uj * 1000) // max(1, c.wall_ns))
    dna = DataDNA(
        segment_id="silicon",
        claim_id=claim_id,
        cycles=cycles,
        bytes=0,
        misses=misses,
        thermal=thermal,
        voltage=int(voltage),
        utilization=util,
    )
    return dna, provenance


# --- the telemetry-ring feed -----------------------------------------------------


def sample_into_ring(ring, claim_id: int, work) -> Counters:
    """Run `work()` once, measure it, and write one record into the zero-copy
    telemetry ring. When a hardware PMU is present, the real cycle + cache-miss
    counts are used (`misses` carries L*-miss counts); otherwise `cycles` = measured
    CPU nanoseconds, falling back to measured wall nanoseconds when the host CPU clock
    is too coarse to move during short work (notably Windows), and `misses` = page
    faults; a PMU that counted one of the two and not the other gives the OS pair
    (`pmu_counts`). The raw ``Counters`` delta remains unchanged. The kernel writes, the
    consumer `drain()`s the same buffer -- a real measured signal."""
    from .telemetry import DataDNA

    sampler = CounterSampler()
    hw = read_hw_counters(work)  # runs work() iff a PMU is present
    if hw is None:
        work()  # no PMU: run once under OS counters
    c = sampler.lap()
    counts = pmu_counts(hw)
    if counts is not None:
        cycles, misses = counts[0], counts[2]
    else:
        cycles = c.cpu_ns if c.cpu_ns > 0 else c.wall_ns
        misses = c.minor_faults + c.major_faults
    ring.write(
        DataDNA(
            segment_id="",
            claim_id=claim_id,
            cycles=cycles,
            bytes=0,
            misses=misses,
            thermal=0,
            voltage=0,
            utilization=min(100, (c.cpu_ns * 100) // max(1, c.wall_ns)),
        )
    )
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
        "rusage_counters": _resource is not None,
        "rapl_energy": rapl_available(),
        "thermal_zone": read_thermal_millideg() is not None,
    }
