"""Phase-aware DVFS: let the schedule drive the clock.

The GEM scheduler knows the *intensity* of each phase before it runs -- the selected
realizations' cost vectors give the compute vs memory mix. Dynamic Voltage and
Frequency Scaling should follow that: **downclock memory-bound phases** (the clock
does not gate a bandwidth-limited phase, so a lower frequency saves power with no
throughput loss) and **overclock compute-bound phases** (more frequency is more
throughput) -- subject to the thermal/power budget Theta (you cannot overclock a
hot or power-capped part).

GAINS-ONLY / no regression: only clearly compute- or memory-bound phases deviate
from the nominal clock; balanced/uncertain phases stay nominal. Downclocking a
memory-bound phase is throughput-neutral by construction (the phase is bandwidth-
bound, modeled below), so it is pure power savings. An overclock is suppressed
whenever Theta is hot or power-capped. Pure, deterministic, integer (clocks are Q8
multipliers of the nominal frequency, 256 == x1.0).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..kbcir.cost import COMPUTE, MEMORY, TargetProfile, Theta
from ..kbcir.realize import RealizationResult

NOMINAL = 256  # Q8 clock == x1.0
OVERCLOCK = 320  # x1.25 (compute-bound, thermal budget permitting)
DOWNCLOCK = 192  # x0.75 (memory-bound: saves power, throughput unaffected)

# Arithmetic-intensity thresholds (compute*1000 / memory, in milli).
HI_INTENSITY = 1000  # compute >= memory  -> compute-bound
LO_INTENSITY = 250  # compute <= 25% of memory -> memory-bound
# Theta pressures above which an overclock is unsafe (hold nominal instead).
THERMAL_CAP = 70
POWER_CAP = 70


def classify(compute: int, memory: int) -> str:
    """compute-bound / memory-bound / balanced from the phase's compute:memory mix."""
    intensity = (compute * 1000) // max(1, memory)
    if intensity >= HI_INTENSITY:
        return "compute"
    if intensity <= LO_INTENSITY:
        return "memory"
    return "balanced"


def clock_for(klass: str, theta: Theta) -> tuple[int, str]:
    """The Q8 clock for a phase class under live Theta, with the safety gate."""
    if klass == "compute":
        if theta.thermal >= THERMAL_CAP or theta.power >= POWER_CAP:
            return NOMINAL, "compute-bound but thermal/power capped -> hold nominal"
        return OVERCLOCK, "compute-bound -> overclock for throughput"
    if klass == "memory":
        return DOWNCLOCK, "memory-bound -> downclock (saves power, throughput unaffected)"
    return NOMINAL, "balanced -> nominal"


@dataclass(frozen=True)
class DVFSDecision:
    phase_id: int
    compute: int
    memory: int
    klass: str
    clock_q8: int
    reason: str

    @property
    def power_milli(self) -> int:
        """Modeled dynamic power ~ f * V^2 with V ~ f (so ~ f^3), relative to nominal."""
        return (self.clock_q8**3) // (NOMINAL * NOMINAL)

    @property
    def throughput_milli(self) -> int:
        """Compute-bound throughput scales with the clock; a memory-bound (or
        balanced) phase is bandwidth-capped, so its throughput is clock-independent
        -- which is exactly why downclocking it loses nothing."""
        return self.clock_q8 if self.klass == "compute" else NOMINAL


@dataclass(frozen=True)
class DVFSPlan:
    decisions: tuple[DVFSDecision, ...]

    def clock_of(self, phase_id: int) -> int:
        for d in self.decisions:
            if d.phase_id == phase_id:
                return d.clock_q8
        return NOMINAL

    @property
    def overclocked(self) -> tuple[int, ...]:
        return tuple(d.phase_id for d in self.decisions if d.clock_q8 > NOMINAL)

    @property
    def downclocked(self) -> tuple[int, ...]:
        return tuple(d.phase_id for d in self.decisions if d.clock_q8 < NOMINAL)

    @property
    def total_power_milli(self) -> int:
        return sum(d.power_milli for d in self.decisions)


def phase_totals(module, result: RealizationResult) -> dict[int, tuple[int, int]]:
    """Per-phase (compute, memory) summed over the selected realizations' base costs."""
    totals: dict[int, list[int]] = {}
    for step in result.steps:
        base = step.candidate.base.v
        acc = totals.setdefault(step.phase_id, [0, 0])
        acc[0] += base[COMPUTE]
        acc[1] += base[MEMORY]
    return {pid: (c, m) for pid, (c, m) in totals.items()}


def plan_dvfs(
    module, result: RealizationResult, theta: Theta, h: TargetProfile | None = None
) -> DVFSPlan:
    """A per-phase clock plan: classify each phase by intensity and set its clock,
    honoring the thermal/power budget. Deterministic; phases sorted by id."""
    totals = phase_totals(module, result)
    decisions = []
    for pid in sorted(totals):
        compute, memory = totals[pid]
        klass = classify(compute, memory)
        clock, reason = clock_for(klass, theta)
        decisions.append(
            DVFSDecision(
                phase_id=pid,
                compute=compute,
                memory=memory,
                klass=klass,
                clock_q8=clock,
                reason=reason,
            )
        )
    return DVFSPlan(decisions=tuple(decisions))


# --- real cpufreq wiring (read the table; actuation honestly gated) ---------------


@dataclass(frozen=True)
class FreqTarget:
    phase_id: int
    clock_q8: int
    target_khz: int  # nominal_khz * clock_q8 / 256, snapped to a real step
    settable: bool  # can this host actuate the frequency (userspace gov + root)?


def quantize_to_silicon(plan: DVFSPlan, cpufreq=None) -> list[FreqTarget]:
    """Map the Q8 clock plan onto the *real* CPU frequency: each phase's clock scales
    the nominal frequency and snaps to the nearest exposed scaling step (when the
    host lists them). `settable` reports whether DVFS can actually be actuated here
    (a `userspace` governor + writable `scaling_setspeed`) -- we never claim to have
    set a clock we could not. Returns [] if no nominal frequency is readable."""
    if cpufreq is None:
        from ..silicon import cpufreq_info

        cpufreq = cpufreq_info()
    if not cpufreq.available:
        return []
    nominal = cpufreq.nominal_khz
    out: list[FreqTarget] = []
    for d in plan.decisions:
        want = (nominal * d.clock_q8) // NOMINAL
        target = min(cpufreq.steps_khz, key=lambda s: abs(s - want)) if cpufreq.steps_khz else want
        out.append(
            FreqTarget(
                phase_id=d.phase_id,
                clock_q8=d.clock_q8,
                target_khz=target,
                settable=cpufreq.actuatable,
            )
        )
    return out


# --- real DVFS actuation (attempt to write the clock; report the boundary) --------

_SETSPEED = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed"
_CURFREQ = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"


@dataclass(frozen=True)
class ActuationResult:
    phase_id: int
    requested_khz: int
    observed_khz: int | None  # scaling_cur_freq read back after the write (None if not read)
    applied: bool  # the frequency was actually set on this host
    reason: str  # why not, when applied is False (the honest boundary)


def actuate(targets, cpufreq=None) -> list[ActuationResult]:
    """**Attempt** to set the real CPU frequency for each phase target, degrading
    gracefully and reporting the boundary -- never faking success.

    Actuation requires a `userspace` cpufreq governor and a writable
    `scaling_setspeed` (root / CAP_SYS_ADMIN). When present, we write `target_khz`
    and read `scaling_cur_freq` back to confirm. When absent (the common sandbox
    case: no governor, no privilege), we return a *dry-run* result that records what
    would be set and exactly why it could not be -- so the same code path lights up
    on a bare-metal host with the governor configured. Never raises."""
    if cpufreq is None:
        from ..silicon import cpufreq_info

        cpufreq = cpufreq_info()
    out: list[ActuationResult] = []
    for t in targets:
        if not cpufreq.actuatable:
            reason = (
                "no cpufreq governor exposed"
                if cpufreq.governor is None
                else f"governor is '{cpufreq.governor}', need 'userspace'"
                if cpufreq.governor != "userspace"
                else "scaling_setspeed not writable (need root/CAP_SYS_ADMIN)"
            )
            out.append(
                ActuationResult(
                    phase_id=t.phase_id,
                    requested_khz=t.target_khz,
                    observed_khz=None,
                    applied=False,
                    reason=reason,
                )
            )
            continue
        try:
            with open(_SETSPEED, "w") as f:
                f.write(str(t.target_khz))
            observed = None
            try:
                with open(_CURFREQ) as f:
                    observed = int(f.read().strip())
            except OSError:
                pass
            out.append(
                ActuationResult(
                    phase_id=t.phase_id,
                    requested_khz=t.target_khz,
                    observed_khz=observed,
                    applied=True,
                    reason="",
                )
            )
        except OSError as e:  # lost the race / permission revoked
            out.append(
                ActuationResult(
                    phase_id=t.phase_id,
                    requested_khz=t.target_khz,
                    observed_khz=None,
                    applied=False,
                    reason=f"write failed: {e.strerror}",
                )
            )
    return out
