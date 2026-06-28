"""Active "regret" sensing: spend telemetry where the model is *unsure*.

Logging everything is too slow; logging nothing is blind. `kbcir.regret` measures
hindsight regret retrospectively; this module is the *active* upgrade -- it
estimates each code path's performance **uncertainty** (variance of the realized
cost) and turns on high-resolution telemetry only for the paths where uncertainty
is high. Confident (low-variance) paths drop to low-rate or off, so the system
minimizes overhead while maximizing what it learns.

The uncertainty signal here is the empirical variance of the per-path samples
(deterministic integer Welford-style sums), which is exactly the label a GNN
uncertainty head would be trained to predict; the gate logic is identical either
way. Deterministic and integer (population variance via integer arithmetic, stdev
via `math.isqrt`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _valid_cost(cost) -> bool:
    """A cost sample is admissible iff it is a finite, non-negative real number.

    INTEGRITY (CT4-sec): `RegretSensor` trusts the `cost` it is fed; a NaN/inf or a
    negative cost is a clearly-bogus injection that would corrupt the variance/CV the
    gate keys on (e.g. a NaN poisons every downstream sum). Such values are REJECTED at
    `observe`. This bounds the blast radius but does NOT defend the harder attack -- a
    *plausible, in-range, low-variance fabricated* cost for a genuinely-bad path (it
    looks confident, so telemetry is forced "off"). That residual needs source
    authentication (a MAC on the producer), which is out of scope; the witness here
    surfaces *how many* samples were rejected and flags a path whose "off" verdict
    rests on too few samples (`forced_off`), so the suppression is at least observable.
    """
    if isinstance(cost, bool):
        return False
    if isinstance(cost, int):
        return cost >= 0
    if isinstance(cost, float):
        return math.isfinite(cost) and cost >= 0
    return False

# A path is "uncertain" once its coefficient of variation (stdev/mean) exceeds this
# many milli-units (200 == 0.20 == 20% relative spread). Below it the model is
# confident enough that high-resolution telemetry is not worth the overhead.
DEFAULT_CV_THRESHOLD_MILLI = 200
# At most this many paths run high-resolution telemetry at once (the overhead cap).
DEFAULT_HIRES_BUDGET = 4
# Relative cost of each telemetry resolution (overhead model; "off" is free).
_RES_COST = {"off": 0, "low": 1, "high": 16}


@dataclass
class PathSamples:
    """Streaming sufficient statistics for one path (integer, deterministic)."""

    path: str
    n: int = 0
    total: int = 0
    total_sq: int = 0

    def observe(self, cost: int) -> None:
        self.n += 1
        self.total += int(cost)
        self.total_sq += int(cost) * int(cost)

    @property
    def mean(self) -> int:
        return self.total // self.n if self.n else 0

    @property
    def variance(self) -> int:
        """Population variance, integer: (n*Σx² - (Σx)²) / n²."""
        if self.n == 0:
            return 0
        return (self.n * self.total_sq - self.total * self.total) // (self.n * self.n)

    @property
    def stdev(self) -> int:
        return math.isqrt(self.variance)

    @property
    def cv_milli(self) -> int:
        """Coefficient of variation in milli (relative uncertainty); 0 if no mean."""
        m = self.mean
        return (1000 * self.stdev) // m if m else 0


@dataclass(frozen=True)
class TelemetryDecision:
    path: str
    resolution: str        # "high" | "low" | "off"
    cv_milli: int
    samples: int
    forced_off: bool = False   # "off" rests on < min_samples -> the verdict is unwitnessed

    @property
    def overhead(self) -> int:
        return _RES_COST[self.resolution]


@dataclass
class RegretSensor:
    """Accumulates per-path samples and decides where to spend telemetry."""

    paths: dict[str, PathSamples] = field(default_factory=dict)
    rejected: int = 0          # cost samples dropped as non-finite/negative (injection)

    def observe(self, path: str, cost: int) -> None:
        """Record one cost sample for `path`. A non-finite or negative `cost` is a bogus
        injection and is REJECTED-and-counted (it never enters the variance/CV the gate
        keys on). See `_valid_cost` for the honest defends/does-NOT note: a *plausible*
        fabricated cost still passes -- that residual is surfaced via `forced_off`."""
        if not _valid_cost(cost):
            self.rejected += 1
            return
        self.paths.setdefault(path, PathSamples(path=path)).observe(cost)

    def uncertainty(self, path: str) -> int:
        s = self.paths.get(path)
        return s.cv_milli if s else 0

    def sense(self, *, cv_threshold_milli: int = DEFAULT_CV_THRESHOLD_MILLI,
              budget: int = DEFAULT_HIRES_BUDGET, min_samples: int = 2) -> list[TelemetryDecision]:
        """Per-path telemetry resolution. Uncertain paths (cv over threshold, with
        enough samples to trust the estimate) get high-resolution -- most uncertain
        first, capped at `budget`. Everything else is low (a trickle) or off (a path
        not yet sampled enough to judge). Deterministic: sorted by (-cv, path)."""
        ranked = sorted(self.paths.values(), key=lambda s: (-s.cv_milli, s.path))
        out: list[TelemetryDecision] = []
        hires = 0
        for s in ranked:
            uncertain = s.n >= min_samples and s.cv_milli >= cv_threshold_milli
            forced_off = False
            if uncertain and hires < budget:
                res = "high"
                hires += 1
            elif s.n >= min_samples:
                res = "low"
            else:
                res = "off"
                forced_off = True            # "off" with too few samples to judge: unwitnessed
            out.append(TelemetryDecision(path=s.path, resolution=res,
                                         cv_milli=s.cv_milli, samples=s.n,
                                         forced_off=forced_off))
        return sorted(out, key=lambda d: d.path)


DEFAULT_MARGIN_THRESHOLD = 64    # Q8 ranker margin below which a column is "uncertain"


def sense_by_ranker(ranker, blocks, *, margin_threshold: int = DEFAULT_MARGIN_THRESHOLD,
                    budget: int = DEFAULT_HIRES_BUDGET) -> list[TelemetryDecision]:
    """A-priori telemetry gating by the LearnedRanker's predictive confidence (the
    complement of `RegretSensor.sense`, which gates a posteriori on observed
    variance). For each block `(path, cands, scores)` the ranker reports its decision
    margin; a wide margin (>= threshold) means the planner already knows the optimal
    realization, so telemetry stays **off** (the noop -- you do not pay to confirm
    what the model is sure of). A narrow margin means the candidates look alike, so
    the column is instrumented at high resolution -- narrowest margin first, capped at
    `budget`. Deterministic; sorted by path. (`cv_milli` carries the margin here.)"""
    scored = [(path, ranker.confidence(cands, scores)) for path, cands, scores in blocks]
    ranked = sorted(scored, key=lambda t: (t[1], t[0]))     # least-confident first
    out: list[TelemetryDecision] = []
    hires = 0
    for path, margin in ranked:
        if margin < margin_threshold and hires < budget:
            res = "high"
            hires += 1
        else:
            res = "off"                                      # confident: no telemetry
        out.append(TelemetryDecision(path=path, resolution=res, cv_milli=margin,
                                     samples=0))
    return sorted(out, key=lambda d: d.path)


def total_overhead(decisions: list[TelemetryDecision]) -> int:
    return sum(d.overhead for d in decisions)


def full_resolution_overhead(decisions: list[TelemetryDecision]) -> int:
    """What 'log everything at high resolution' would cost -- the baseline the
    active gate beats by only sampling uncertain paths."""
    return _RES_COST["high"] * len(decisions)
