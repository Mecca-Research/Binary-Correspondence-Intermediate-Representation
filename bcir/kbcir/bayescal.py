"""Bayesian cost-model calibration with conformal error bars (the learned organ).

The point microbench (`kbcir.microbench`) gives a single Q8 ratio per access
regime. This module upgrades that to a *posterior* with a *certified* error bar:

  * **Bayesian posterior** over each ratio -- a conjugate Gaussian (Normal-Normal)
    update of the seeded prior with repeated microbench observations. For a
    Gaussian likelihood the conjugate posterior *is* the ELBO-optimal variational
    Gaussian, so this is exact VI: a (mean, variance) per ratio, no autograd, no
    third-party deps.
  * **Conformal error bars** -- split conformal prediction (Vovk): from the
    calibration residuals, the (1-alpha) quantile is a DISTRIBUTION-FREE +/- delta
    with finite-sample coverage. This is the certified interval a verifiable
    learned cost model needs -- a guarantee, not an assumption.
  * **ABC** -- Approximate Bayesian Computation using the GEM/`optimize` forward
    model as the simulator: propose ratio params, simulate the plan score, accept
    if it lands within epsilon of the observed telemetry score. The accepted set
    is a likelihood-free posterior grounded in the actual scheduler.

Frozen to Q8 with the certified +/- delta and generation-tagged: a
`BayesianCalibratedProfile` applies to a `TargetProfile` exactly like the point
table (`CalibratedProfile`), and additionally carries the conformal delta so a
later selection can be made *robust* over the credible interval. The point table
is the degenerate case (one observation, zero delta).

LEARNING-PLACEMENT LAW (LangRef Sec. 13): floats, L2/L3 offline only, never the
hot path. Inference happens here; the frozen artifact is Q8 integers + an integer
delta, deployed on the certified rail and witnessed by R8/R13.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from .._artifact_json import strict_json_loads
from .cost import TargetProfile
from .microbench import Q8, CalibratedProfile, MicrobenchRaw, calibrate_from_raw
from .weights import PERF, Policy


# --- conjugate-Gaussian (VI-exact) posterior over a scalar ratio -----------------

@dataclass(frozen=True)
class Posterior:
    """A Gaussian posterior over one ratio (Q8 units). For a Gaussian likelihood
    this is the exact conjugate posterior == the ELBO-optimal variational Gaussian."""

    mean: float
    var: float

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.var))


def gaussian_update(obs: list[float], prior_mean: float, prior_var: float = 1e9,
                    obs_var: float | None = None) -> Posterior:
    """Normal-Normal conjugate update over a batch (the exact Gaussian VI).

    A weak prior (`prior_var` large) makes the posterior mean the sample mean.
    `obs_var` defaults to the sample variance (floored), so noisier measurements
    widen the posterior -- the Bayesian shrinkage the point estimate cannot express.
    """
    if not obs:
        return Posterior(prior_mean, prior_var)
    if obs_var is None:
        m = sum(obs) / len(obs)
        obs_var = max(1.0, sum((o - m) ** 2 for o in obs) / len(obs))
    prec = 1.0 / prior_var
    mean_num = prior_mean / prior_var
    for o in obs:
        prec += 1.0 / obs_var
        mean_num += o / obs_var
    var = 1.0 / prec
    return Posterior(mean_num * var, var)


# --- split conformal prediction (distribution-free +/- delta) --------------------

def conformal_delta(residuals: list[float], coverage: float = 0.9) -> float:
    """The split-conformal +/- half-width with finite-sample coverage >= `coverage`
    (Vovk): the ceil((n+1)(1-alpha))-th smallest absolute residual. Distribution
    free -- the certified error bar. When n is too small for the requested level
    the bound widens to the largest residual (the best finite witness available)."""
    n = len(residuals)
    if n == 0:
        return 0.0
    alpha = 1.0 - coverage
    rank = math.ceil((n + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), n)
    return sorted(abs(r) for r in residuals)[rank - 1]


# --- the frozen artifact: point table + certified conformal interval -------------

@dataclass(frozen=True)
class BayesianCalibratedProfile:
    """A frozen point table (posterior means, Q8) plus a certified conformal +/-
    delta on the random ratio (the constant whose uncertainty drives gather cost).
    Duck-types `CalibratedProfile` for R13 (cal_gen / gather_penalty / ...)."""

    point: CalibratedProfile
    coverage_milli: int        # conformal coverage x 1000 (e.g. 900 = 90%)
    random_delta_q8: int       # +/- half-width on random_q8 (Q8 units), >= 0

    # -- delegate the point-table contract (so verify_provenance works unchanged) --
    @property
    def name(self) -> str:
        return self.point.name

    @property
    def cal_gen(self) -> int:
        return self.point.cal_gen

    @property
    def gather_penalty(self) -> int:
        return self.point.gather_penalty

    @property
    def base_overhead(self) -> int:
        return self.point.base_overhead

    @property
    def mem_unit(self) -> int:
        return self.point.mem_unit

    def apply(self, h: TargetProfile) -> TargetProfile:
        return self.point.apply(h)

    def gather_penalty_interval(self) -> tuple[int, int]:
        """The certified [low, high] on gather_penalty at the stated coverage."""
        lo = max(1, (self.point.random_q8 - self.random_delta_q8) // Q8)
        hi = max(1, (self.point.random_q8 + self.random_delta_q8) // Q8)
        return (lo, hi)

    def to_json(self) -> str:
        return json.dumps({"point": asdict(self.point),
                           "coverage_milli": self.coverage_milli,
                           "random_delta_q8": self.random_delta_q8},
                          indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "BayesianCalibratedProfile":
        d = strict_json_loads(text, "Bayesian calibrated profile")
        if not isinstance(d, dict) or set(d) != {"point", "coverage_milli", "random_delta_q8"}:
            raise ValueError("Bayesian calibrated profile has unknown or missing fields")
        coverage = d["coverage_milli"]
        delta = d["random_delta_q8"]
        if (isinstance(coverage, bool) or not isinstance(coverage, int) or
                not 1 <= coverage <= 1000):
            raise ValueError("coverage_milli must be an integer in [1, 1000]")
        if (isinstance(delta, bool) or not isinstance(delta, int) or
                not 0 <= delta <= (1 << 63) - 1):
            raise ValueError("random_delta_q8 must be a non-negative i63")
        if not isinstance(d["point"], dict):
            raise ValueError("Bayesian calibrated profile point must be an object")
        return BayesianCalibratedProfile(
            point=CalibratedProfile.from_json(json.dumps(d["point"])),
            coverage_milli=coverage, random_delta_q8=delta)


def _freeze_point(name: str, cal_gen: int, samples: int, provenance: str,
                  stream_q8: int, strided_q8: int, random_q8: int,
                  compute_q8: int) -> CalibratedProfile:
    return CalibratedProfile(name=name, cal_gen=max(1, cal_gen), samples=samples,
                             provenance=provenance, stream_q8=stream_q8,
                             strided_q8=strided_q8, random_q8=random_q8,
                             compute_q8=compute_q8)


def bayes_calibrate(h: TargetProfile, raws: list[MicrobenchRaw],
                    coverage: float = 0.9, cal_gen: int = 1) -> BayesianCalibratedProfile:
    """Conjugate-Gaussian posterior over each ratio + a split-conformal delta on
    the random ratio, frozen to a Q8 table. The Bayesian twin of
    `microbench.calibrate_from_raw` over *repeated* measurements."""
    if not raws:
        raise ValueError("bayes_calibrate needs >= 1 microbench raw")
    points = [calibrate_from_raw(r, h) for r in raws]
    stream = [p.stream_q8 for p in points]
    strided = [p.strided_q8 for p in points]
    random = [p.random_q8 for p in points]
    compute = [p.compute_q8 for p in points]

    post_random = gaussian_update(random, prior_mean=32 * Q8)
    mean_random = max(Q8, round(post_random.mean))
    residuals = [r - mean_random for r in random]
    delta = round(conformal_delta(residuals, coverage))

    point = _freeze_point(
        name=h.name, cal_gen=cal_gen, samples=len(raws),
        provenance=f"bayes (conjugate VI + split conformal cov={coverage}) over {len(raws)} raws",
        stream_q8=Q8, strided_q8=max(Q8, round(gaussian_update(strided, Q8).mean)),
        random_q8=mean_random, compute_q8=max(Q8, round(gaussian_update(compute, Q8).mean)))
    return BayesianCalibratedProfile(point=point,
                                     coverage_milli=round(coverage * 1000),
                                     random_delta_q8=delta)


# --- ABC: the GEM/optimize forward model as the simulator ------------------------

@dataclass(frozen=True)
class ABCPosterior:
    """The likelihood-free posterior: the calibration profiles whose *simulated*
    plan score landed within epsilon of the observed telemetry score."""

    accepted: tuple
    proposed: int
    epsilon: int
    observed: int

    def gather_penalties(self) -> list[int]:
        return sorted(p.gather_penalty for p in self.accepted)

    def to_bayesian(self, h: TargetProfile, coverage: float = 0.9,
                    cal_gen: int = 1) -> BayesianCalibratedProfile:
        """Fold the accepted ABC sample into a frozen Bayesian table: posterior
        mean = accepted-sample mean random ratio, delta = split-conformal over it."""
        if not self.accepted:
            raise ValueError("empty ABC posterior: widen epsilon or the proposal set")
        random = [p.random_q8 for p in self.accepted]
        post = gaussian_update(random, prior_mean=32 * Q8)
        mean_random = max(Q8, round(post.mean))
        delta = round(conformal_delta([r - mean_random for r in random], coverage))
        a0 = self.accepted[0]
        point = _freeze_point(h.name, cal_gen, len(self.accepted),
                              f"abc (GEM-simulated, eps={self.epsilon}, "
                              f"{len(self.accepted)}/{self.proposed} accepted)",
                              Q8, a0.strided_q8, mean_random, a0.compute_q8)
        return BayesianCalibratedProfile(point, round(coverage * 1000), delta)


def abc_calibrate(module, h: TargetProfile, theta, observed_score: int,
                  proposals: list[CalibratedProfile], epsilon: int,
                  judge: Policy = PERF) -> ABCPosterior:
    """ABC with the optimizer as the simulator: accept a proposed cost table iff
    `optimize(module, table.apply(h), theta, judge).score` is within epsilon of
    the observed score. Likelihood-free inference grounded in the real scheduler."""
    from .realize import optimize

    accepted = tuple(
        p for p in proposals
        if abs(optimize(module, p.apply(h), theta, judge).score - observed_score) <= epsilon)
    return ABCPosterior(accepted=accepted, proposed=len(proposals),
                        epsilon=epsilon, observed=observed_score)
