"""bcir.kbcir.calibloop -- closing the calibration loop: measure -> freeze ->
apply -> replan, certified and generation-tagged.

The calibration *halves* already exist: `kbcir.microbench` measures the host and
freezes a Q8 cost table (the physics-anchored constants), and `kbcir.calibrate`
folds data-DNA telemetry into the runtime state Theta. This module *closes the
loop* and makes its value a checkable artifact:

    measure (real host)  ->  freeze Q8 table (cal_gen)  ->  apply to H  ->
    fold telemetry into Theta  ->  replan  ->  certify the win.

The **win** is the measured cost of *not* recalibrating: the plan you would ship
under stale (default) assumptions, rescored on the machine the telemetry actually
reports, minus the optimum the recalibrated planner picks there. It is >= 0 by
construction (the recalibrated plan is the optimum under the measured model), and
a `CalibrationCertificate` records it alongside the frozen table's generation and
provenance. A ratio-1 reference table on a nominal machine yields win == 0 (no
value, correctly); a hot machine flips vec16 -> vec8 and the win is the heat/
current the stale wide plan would have wasted.

Determinism: measurement is offline/non-deterministic (L2/L3, `measure_and_close`
on real silicon), but the frozen table and every downstream decision are integer
and reproducible -- the loop's certificate is the deterministic artifact (the hot
path never measures; it runs the compiled-out decision). Witnessed by R13
(`verify.verify_calibration`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from ..model import Module
from ..telemetry import DataDNA
from .calibrate import EwmaCalibrator
from .cost import HProfile, Theta
from .microbench import CalibratedProfile, calibrate_profile, reference_table
from .realize import (
    RealizationResult,
    _context_factor,
    _flatten,
    candidates_for,
    optimize,
)
from .weights import PERF, Policy, weights


def rescore_plan(module: Module, result: RealizationResult, h: HProfile,
                 theta: Theta, policy: Policy = PERF) -> int | None:
    """Score a *fixed* plan (the realizations `result` already chose) under a new
    target/Theta -- reusing the optimizer's own primitives (weights, candidate
    costs, context coupling), not a reimplementation. Returns None if a chosen
    realization is not available under `h` (it cannot be faithfully rescored).

    This is how the loop prices a stale plan on the machine the telemetry reports:
    `rescore_plan(m, optimize(m,h,th,pol), h, th, pol) == optimize(...).score`."""
    chosen = {s.claim_id: s.candidate.name for s in result.steps}
    total = 0
    prev = None
    for phase_id, claim in _flatten(module):
        w = weights(h, theta, phase_id, policy)
        rid = claim.rd[0] if claim.rd else claim.primary_rid
        resource = module.resource(rid) if rid is not None else None
        cand = next((c for c in candidates_for(claim, h, resource)
                     if c.name == chosen.get(claim.id)), None)
        if cand is None:
            return None
        total += cand.base.couple(_context_factor(prev, cand, theta)).dot(w)
        prev = cand
    return total


@dataclass(frozen=True)
class CalibrationCertificate:
    """The closed-loop witness: a frozen table's identity + the measured value of
    recalibrating (the replan win). `win >= 0` and `cal_gen >= 1` make it
    admissible (R13)."""

    target: str
    cal_gen: int
    provenance: str
    ratios: tuple              # (stream_q8, strided_q8, random_q8, compute_q8)
    measured_thermal: int      # the telemetry-calibrated Theta thermal pressure
    seeded_widths: tuple       # ((claim_id, width), ...) -- the stale (default) plan
    calibrated_widths: tuple   # ((claim_id, width), ...) -- the recalibrated plan
    stale_cost: int            # the seeded plan rescored on the measured machine
    calibrated_cost: int       # the recalibrated optimum on the measured machine

    @property
    def replanned(self) -> bool:
        return self.seeded_widths != self.calibrated_widths

    @property
    def win(self) -> int:
        """The measured cost of *not* recalibrating (>= 0)."""
        return self.stale_cost - self.calibrated_cost

    @property
    def admissible(self) -> bool:
        return self.cal_gen >= 1 and self.win >= 0

    def to_json(self) -> str:
        d = asdict(self)
        d["ratios"] = list(self.ratios)
        d["seeded_widths"] = [list(x) for x in self.seeded_widths]
        d["calibrated_widths"] = [list(x) for x in self.calibrated_widths]
        return json.dumps(d, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "CalibrationCertificate":
        d = json.loads(text)
        return CalibrationCertificate(
            target=d["target"], cal_gen=d["cal_gen"], provenance=d["provenance"],
            ratios=tuple(d["ratios"]), measured_thermal=d["measured_thermal"],
            seeded_widths=tuple(tuple(x) for x in d["seeded_widths"]),
            calibrated_widths=tuple(tuple(x) for x in d["calibrated_widths"]),
            stale_cost=d["stale_cost"], calibrated_cost=d["calibrated_cost"])


def _widths(result: RealizationResult) -> tuple:
    return tuple(sorted((cid, c.width) for cid, c in result.by_claim().items()))


def close_loop(module: Module, h: HProfile, *, table: CalibratedProfile = None,
               events=(), default_theta: Theta = None,
               calibrator: EwmaCalibrator = None,
               policy: Policy = PERF) -> CalibrationCertificate:
    """Close the loop with a (frozen) table + telemetry: compare the stale plan
    (shipped under `default_theta`) against the recalibrated optimum on the
    machine the telemetry reports, and certify the win. `policy` is held fixed for
    both sides so the win isolates the measured table/Theta, not a policy swap."""
    table = table if table is not None else reference_table()
    default_theta = default_theta if default_theta is not None else Theta.cool()
    calibrator = calibrator if calibrator is not None else EwmaCalibrator()

    measured_theta = calibrator.update(default_theta, list(events))
    hc = table.apply(h)

    seeded = optimize(module, h, default_theta, policy)          # the stale (default) plan
    calibrated = optimize(module, hc, measured_theta, policy)    # the recalibrated optimum
    stale = rescore_plan(module, seeded, hc, measured_theta, policy)

    return CalibrationCertificate(
        target=h.name, cal_gen=table.cal_gen, provenance=table.provenance,
        ratios=(table.stream_q8, table.strided_q8, table.random_q8, table.compute_q8),
        measured_thermal=measured_theta.thermal,
        seeded_widths=_widths(seeded), calibrated_widths=_widths(calibrated),
        stale_cost=int(stale if stale is not None else calibrated.score),
        calibrated_cost=int(calibrated.score))


def measure_and_close(module: Module, h: HProfile, *, events=(), n: int = 1 << 16,
                      repeats: int = 5, cal_gen: int = 1, **kw):
    """Run the loop on REAL hardware: microbench this host, freeze the Q8 table,
    then close the loop. Returns (certificate, table). Offline (L2/L3) -- never on
    the hot path. The table is the deterministic artifact the planner consumes."""
    table = calibrate_profile(h, n=n, repeats=repeats, cal_gen=cal_gen)
    return close_loop(module, h, table=table, events=events, **kw), table
