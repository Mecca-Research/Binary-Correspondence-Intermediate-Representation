"""CT4: ML-guided cost calibration, adaptive policy, and the rehydrate loop.

Observed data-DNA telemetry is folded back into the runtime state Theta (an EWMA
calibrator stands in for a pluggable ML model), an adaptive policy is chosen from
Theta, and a rehydrate decision (keep/patch/repack/replan) is made from generation
drift and thermal drift. This is the AI-guided half of `K_BCIR`: the optimizer's
inputs adapt to observed reality.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Module
from ..telemetry import DataDNA
from .cost import HProfile, Theta
from .realize import RealizationResult, optimize
from .weights import ENERGY, PERF, THROUGHPUT, Policy


def _avg(events: list[DataDNA], attr: str) -> int:
    if not events:
        return 0
    return sum(getattr(e, attr) for e in events) // len(events)


@dataclass(frozen=True)
class EwmaCalibrator:
    """Exponentially-weighted calibrator: new = alpha*observed + (1-alpha)*old.

    `alpha` is in Q8 (256 == 1.0). Stands in for a trained ML cost model.
    """

    alpha: int = 128  # 0.5 in Q8

    def _ewma(self, old: int, observed: int) -> int:
        return (self.alpha * observed + (256 - self.alpha) * old) >> 8

    def update(self, theta: Theta, events: list[DataDNA]) -> Theta:
        if not events:
            return theta
        return Theta(
            thermal=self._ewma(theta.thermal, _avg(events, "thermal")),
            power=theta.power,
            mem_pressure=self._ewma(theta.mem_pressure, _avg(events, "misses")),
            contention=theta.contention,
            noise=theta.noise,
            wear=theta.wear,
            utilization=self._ewma(theta.utilization, _avg(events, "utilization")),
            voltage=self._ewma(theta.voltage, _avg(events, "voltage")),
        )


def adaptive_policy(theta: Theta) -> Policy:
    """Pick a scalarization policy from the live runtime state."""
    if theta.thermal >= 60 or theta.power >= 60 or theta.voltage >= 60:
        return ENERGY          # hot/constrained -> minimize heat/power
    if theta.mem_pressure >= 60 or theta.contention >= 60:
        return THROUGHPUT      # bandwidth-starved -> weight memory/contention
    return PERF                # nominal -> latency


def rehydrate_decide(
    pack_map_gen: int,
    pack_data_gen: int,
    runtime_map_gen: int,
    runtime_data_gen: int,
    theta: Theta,
    baseline_thermal: int = 0,
) -> str:
    """Decide keep / patch / repack / replan for a StreamPack vs runtime state."""
    if pack_data_gen != runtime_data_gen:
        return "replan"        # data changed -> the plan may be invalid
    if pack_map_gen != runtime_map_gen:
        return "repack"        # mapping changed -> re-hydrate the same plan
    drift = abs(theta.thermal - baseline_thermal)
    if drift >= 40:
        return "replan"        # large thermal drift -> a cheaper lane may win
    if drift >= 20:
        return "patch"         # minor drift -> tweak in place
    return "keep"


def calibrate_and_replan(
    module: Module,
    h: HProfile,
    events: list[DataDNA],
    base_theta: Theta = Theta.cool(),
    calibrator: EwmaCalibrator = EwmaCalibrator(),
) -> tuple[Theta, Policy, RealizationResult]:
    """Calibrate Theta from telemetry, pick an adaptive policy, and re-optimize."""
    theta = calibrator.update(base_theta, events)
    policy = adaptive_policy(theta)
    result = optimize(module, h, theta, policy)
    return theta, policy, result
