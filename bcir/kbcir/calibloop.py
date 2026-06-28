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
        """Parse + VALIDATE a closed-loop certificate. The certificate is the R13
        witness a downstream consumer trusts (its `cal_gen`/`win` make a replan
        admissible), so a forged or malformed one must be REJECTED, not seated:
        `cal_gen` and `measured_thermal` are finite ints (`cal_gen >= 0`,
        `measured_thermal` in 0..100), the costs are finite ints, `ratios` is exactly
        four finite ints, and the width tuples are well-formed `(claim_id, width)`
        pairs. Any violation raises `ValueError`."""
        d = json.loads(text)
        if not isinstance(d, dict):
            raise ValueError("CalibrationCertificate JSON must be an object")

        def _int(key: str, *, lo: int | None = None, hi: int | None = None) -> int:
            if key not in d:
                raise ValueError(f"CalibrationCertificate missing field {key!r}")
            v = d[key]
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"CalibrationCertificate {key} must be an int, got {v!r}")
            if lo is not None and v < lo:
                raise ValueError(f"CalibrationCertificate {key}={v} below {lo}")
            if hi is not None and v > hi:
                raise ValueError(f"CalibrationCertificate {key}={v} above {hi}")
            return v

        cal_gen = _int("cal_gen", lo=0)
        measured_thermal = _int("measured_thermal", lo=0, hi=100)
        stale_cost = _int("stale_cost")
        calibrated_cost = _int("calibrated_cost")
        ratios = d.get("ratios")
        if not isinstance(ratios, (list, tuple)) or len(ratios) != 4:
            raise ValueError("CalibrationCertificate ratios must be exactly 4 values")
        for i, r in enumerate(ratios):
            if isinstance(r, bool) or not isinstance(r, int):
                raise ValueError(f"CalibrationCertificate ratios[{i}] must be an int, got {r!r}")

        def _widths(key: str) -> tuple:
            raw = d.get(key)
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"CalibrationCertificate {key} must be an array")
            pairs = []
            for x in raw:
                if not isinstance(x, (list, tuple)) or len(x) != 2:
                    raise ValueError(f"CalibrationCertificate {key} entries must be (claim_id, width)")
                cid, width = x
                if any(isinstance(v, bool) or not isinstance(v, int) for v in (cid, width)):
                    raise ValueError(f"CalibrationCertificate {key} entries must be ints")
                pairs.append((cid, width))
            return tuple(pairs)

        if not isinstance(d.get("target"), str) or not isinstance(d.get("provenance"), str):
            raise ValueError("CalibrationCertificate target/provenance must be strings")
        return CalibrationCertificate(
            target=d["target"], cal_gen=cal_gen, provenance=d["provenance"],
            ratios=tuple(ratios), measured_thermal=measured_thermal,
            seeded_widths=_widths("seeded_widths"),
            calibrated_widths=_widths("calibrated_widths"),
            stale_cost=stale_cost, calibrated_cost=calibrated_cost)


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


@dataclass(frozen=True)
class MeasuredReplanCertificate:
    """A `CalibrationCertificate` plus the *provenance of the telemetry* it closed
    on: which physical signals (PMU / RAPL / thermal / OS) were genuinely measured,
    or `("synthetic",)` when the host exposes none (a sandbox). `measured` is True
    only when at least one real silicon signal drove the replan -- so a "measured
    replan win" is never claimed from fabricated telemetry."""

    cert: CalibrationCertificate
    provenance: tuple                  # sorted signal tags, or ("synthetic",)
    samples: int

    @property
    def measured(self) -> bool:
        return self.provenance not in ((), ("synthetic",))

    @property
    def win(self) -> int:
        return self.cert.win

    @property
    def replanned(self) -> bool:
        return self.cert.replanned

    def to_json(self) -> str:
        import json
        return json.dumps({"provenance": list(self.provenance), "samples": self.samples,
                           "measured": self.measured, "certificate": json.loads(self.cert.to_json())},
                          indent=2, sort_keys=True)


def measured_replan(module: Module, h: HProfile, *, work=None, samples: int = 8,
                    default_theta: Theta = None, policy: Policy = PERF,
                    cal_gen: int = 1):
    """Close the calibration loop on **real silicon signals**: sample the host's PMU
    / RAPL / thermal counters around `work` `samples` times (a memory-streaming probe
    by default), train + freeze a `LinearCalibrator` on that measured telemetry, then
    compare the shipped (cool) plan against the recalibrated optimum and certify the
    replan win. Returns a `MeasuredReplanCertificate` whose `provenance` records the
    real signals used; on a host that exposes none it degrades to a single nominal
    sample tagged `("synthetic",)` -- honest, never a faked measurement. The frozen
    table is microbenched once (the deterministic Q8 artifact); only the Theta-side
    telemetry comes from the live signals."""
    from ..silicon import silicon_dna
    from .calibrate import train_calibrator
    from .microbench import calibrate_profile

    default_theta = default_theta if default_theta is not None else Theta.cool()
    n = max(1, module.phases[0].claims[0].count) if module.phases and module.phases[0].claims else 1 << 16

    def _probe():
        # a memory-streaming unit of work sized to the kernel -- the real bandwidth
        # signal the cost model prices against (sum over a freshly-touched buffer).
        buf = bytearray(min(1 << 20, max(4096, n)))
        s = 0
        for i in range(0, len(buf), 64):
            s += buf[i]
        return s

    probe = work if work is not None else _probe
    events = []
    provenance: set[str] = set()
    for _ in range(max(1, samples)):
        dna, prov = silicon_dna(probe, claim_id=0)
        events.append(dna)
        provenance |= prov

    real = provenance - {"os"}                          # OS counters alone are not "silicon"
    tags = tuple(sorted(provenance)) if real else ("synthetic",)
    table = calibrate_profile(h, n=min(n, 1 << 16), repeats=3, cal_gen=cal_gen)
    cert = close_loop(module, h, table=table, events=events, default_theta=default_theta,
                      calibrator=(train_calibrator(events, gen=cal_gen) if real else None),
                      policy=policy)
    return MeasuredReplanCertificate(cert=cert, provenance=tags, samples=len(events))


def measure_and_close(module: Module, h: HProfile, *, events=(), n: int = 1 << 16,
                      repeats: int = 5, cal_gen: int = 1, native: bool = False, **kw):
    """Run the loop on REAL hardware: microbench this host, freeze the Q8 table,
    then close the loop. Returns (certificate, table). Offline (L2/L3) -- never on
    the hot path. The table is the deterministic artifact the planner consumes.

    `native=True` uses the **bare-metal C microbench** (real cache latency) instead
    of the conservative interpreter harness."""
    if native:
        from .microbench import calibrate_native
        table = calibrate_native(h, n=max(n, 1 << 22), repeats=repeats, cal_gen=cal_gen)
    else:
        table = calibrate_profile(h, n=n, repeats=repeats, cal_gen=cal_gen)
    return close_loop(module, h, table=table, events=events, **kw), table
