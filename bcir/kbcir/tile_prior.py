"""D3 (ML/AI roadmap §8.2): LEARNED COST-MODEL PRIORS AT L1 -- tile/loop-order proposals.

The accel-ranker precedent (`kbcir.accel`) generalized one level down, to the L1 tile
search: `plan_matmul` enumerates every (loop order x tile_m x tile_n x tile_k) candidate
and costs each one. This module learns WHICH candidates tend to be optimal -- a logistic
prior over CHEAP shape/tile features (no `cost_of` call needed to rank), trained OFFLINE on
the exact search's own choices under the calibrated cost model, frozen to a Q8 integer
table -- and uses it only to ORDER the search:

    guided_plan_matmul = evaluate candidates in the prior's order, keep the best, and
    STOP at the first PROVABLY-unbeatable one (admissible early exit: the compute term is
    tile-independent, so a cache-fitting candidate whose mem term <= its compute term
    already sits on the bottleneck floor -- nothing later can score lower).

So the prior can only REDUCE SEARCH (fewer `cost_of` evaluations), never change the
optimum: the early exit fires only on a proof, and with a useless prior the guided search
degenerates to the exhaustive one. `TilePriorCertificate` is the safety witness (the
AccelCertificate pattern): over a case family, the guided optimum's (fits_cache,
bottleneck) must EQUAL the exhaustive optimum's on every shape -- mismatches MUST be 0 --
and the recorded node counts quantify the reduction. Cost-equal tie-breaks remain a free
choice (the same freedom the optimizer already has); both searches stay fully
deterministic.

The calibloop wiring is BY CONSTRUCTION, not by hook: the prior trains against whatever
`TargetProfile` the caller passes -- hand it the microbench-calibrated profile the
resident calibloop froze and the proposals follow the measured cost model; the exact
search underneath (same profile) still verifies every choice. `plan_matmul` itself is
untouched -- the prior is opt-in (non-disturbance, vacuous by default).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .._artifact_json import strict_json_loads
from .cost import TargetProfile
from .matmul import _LOOP_ORDERS, TilePlan, _cache_budget_elems, _divisor_tiles, cost_of

Q8 = 256


def shape_class(M: int, N: int, K: int) -> tuple[int, int, int]:
    """The op-shape-class key: log2 buckets per dimension (the D3 table key)."""
    _validate_shape(M, N, K)
    return (M.bit_length(), N.bit_length(), K.bit_length())


def _validate_shape(M: int, N: int, K: int) -> None:
    for name, value in (("M", M), ("N", N), ("K", K)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")


def candidates_for(M: int, N: int, K: int) -> list[tuple[int, int, int, str]]:
    """The exact search space of plan_matmul, as (tm, tn, tk, loop_order) tuples, in the
    exhaustive enumeration order."""
    _validate_shape(M, N, K)
    return [(tm, tn, tk, lo)
            for lo in _LOOP_ORDERS
            for tm in _divisor_tiles(M)
            for tn in _divisor_tiles(N)
            for tk in _divisor_tiles(K)]


def tile_features(M: int, N: int, K: int, tm: int, tn: int, tk: int, lo: str,
                  target: TargetProfile) -> list[float]:
    """CHEAP candidate features -- pure arithmetic on the shape/tile/budget, NO cost_of
    call (ordering must be cheaper than what it saves)."""
    _validate_shape(M, N, K)
    for name, value in (("tm", tm), ("tn", tn), ("tk", tk)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
    if lo not in _LOOP_ORDERS:
        raise ValueError(f"unsupported loop order {lo!r}")
    if not isinstance(target, TargetProfile):
        raise ValueError("tile-prior target must be a TargetProfile")
    budget = _cache_budget_elems(target)
    ws = tm * tk + tk * tn + tm * tn
    return [1.0 if ws <= budget else 0.0,
            min(2.0, ws / max(1, budget)),
            math.log2(max(1, tm * tn * tk)) / 30.0,
            tm / M, tn / N, tk / K,
            _LOOP_ORDERS.index(lo) / 2.0,
            1.0]


@dataclass
class LearnedTilePrior:
    """A logistic model over tile features predicting P(candidate is the exact choice).
    The float training half; freeze() is what ships."""

    w: list = field(default_factory=lambda: [0.0] * 8)

    def _p(self, feats):
        z = sum(wi * xi for wi, xi in zip(self.w, feats))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def freeze(self) -> "FrozenTilePrior":
        return FrozenTilePrior(wq=tuple(round(v * Q8) for v in self.w))


@dataclass(frozen=True)
class FrozenTilePrior:
    """The Q8-frozen prior: a deterministic INTEGER ordering of the tile-search space,
    keyed by nothing but the candidate's cheap features (so one table serves every
    shape class it was trained over)."""

    wq: tuple

    def __post_init__(self):
        if not isinstance(self.wq, tuple) or len(self.wq) != 8:
            raise ValueError("FrozenTilePrior requires exactly 8 Q8 weights")
        for index, weight in enumerate(self.wq):
            if (isinstance(weight, bool) or not isinstance(weight, int) or
                    not -(1 << 31) <= weight <= (1 << 31) - 1):
                raise ValueError(f"FrozenTilePrior weight[{index}] must be a signed i32")

    def order(self, M: int, N: int, K: int, target: TargetProfile) -> list:
        _validate_shape(M, N, K)
        cands = candidates_for(M, N, K)
        scored = []
        for i, (tm, tn, tk, lo) in enumerate(cands):
            feats = tile_features(M, N, K, tm, tn, tk, lo, target)
            z = sum(self.wq[k] * round(feats[k] * Q8) for k in range(len(self.wq))) >> 8
            scored.append((-z, i))                   # higher z = try first; index tie-break
        return [cands[i] for _, i in sorted(scored)]


def prior_samples(shapes: list, target: TargetProfile) -> list:
    """Offline training data from the EXACT search under the (calibrated) cost model:
    per shape, the exhaustive optimum's (fits, bottleneck) marks every score-equal
    candidate positive, the rest negative."""
    samples = []
    for (M, N, K) in shapes:
        best = _exhaustive(M, N, K, target)[0]
        for (tm, tn, tk, lo) in candidates_for(M, N, K):
            compute, mem, fits = cost_of(M, N, K, tm, tn, tk, target)
            label = 1.0 if (fits == best.fits_cache and max(compute, mem) == best.bottleneck) \
                else 0.0
            samples.append((tile_features(M, N, K, tm, tn, tk, lo, target), label))
    return samples


def train_tile_prior(samples: list, epochs: int = 200, lr: float = 0.5) -> LearnedTilePrior:
    """Deterministic logistic-regression SGD (the train_ranker recipe)."""
    p = LearnedTilePrior()
    for _ in range(epochs):
        for feats, label in samples:
            err = p._p(feats) - label
            p.w = [wi - lr * err * xi for wi, xi in zip(p.w, feats)]
    return p


def _plan_of(M, N, K, tm, tn, tk, lo, target) -> TilePlan:
    compute, mem, fits = cost_of(M, N, K, tm, tn, tk, target)
    return TilePlan(tm, tn, tk, lo, compute, mem, max(compute, mem), fits)


def _exhaustive(M: int, N: int, K: int, target: TargetProfile) -> tuple[TilePlan, int]:
    """plan_matmul's search, instrumented: (the optimum, nodes costed)."""
    best = None
    n = 0
    for (tm, tn, tk, lo) in candidates_for(M, N, K):
        cand = _plan_of(M, N, K, tm, tn, tk, lo, target)
        n += 1
        key = (not cand.fits_cache, cand.bottleneck, -(tm * tn * tk), lo)
        if best is None or key < (not best.fits_cache, best.bottleneck,
                                  -(best.tile_m * best.tile_n * best.tile_k), best.loop_order):
            best = cand
    return best, n


def guided_plan_matmul(M: int, N: int, K: int, target: TargetProfile,
                       prior: FrozenTilePrior) -> tuple[TilePlan, int]:
    """The prior-ORDERED search with the admissible early exit. Returns (plan, nodes
    costed). The exit is a PROOF, not a heuristic: compute is tile-independent, so a
    cache-fitting candidate with mem <= compute sits on the bottleneck floor -- no
    remaining candidate can beat (fits_cache, bottleneck). Worst case (prior useless):
    every node is costed and the result is the exhaustive optimum's score anyway."""
    _validate_shape(M, N, K)
    if not isinstance(target, TargetProfile):
        raise ValueError("tile-prior target must be a TargetProfile")
    if not isinstance(prior, FrozenTilePrior):
        raise ValueError("guided matmul requires a FrozenTilePrior")
    best = None
    n = 0
    for (tm, tn, tk, lo) in prior.order(M, N, K, target):
        cand = _plan_of(M, N, K, tm, tn, tk, lo, target)
        n += 1
        key = (not cand.fits_cache, cand.bottleneck, -(tm * tn * tk), lo)
        if best is None or key < (not best.fits_cache, best.bottleneck,
                                  -(best.tile_m * best.tile_n * best.tile_k), best.loop_order):
            best = cand
        if best.fits_cache and best.mem_cost <= best.compute_cost:
            break                                    # provably unbeatable: bottleneck floor
    return best, n


@dataclass(frozen=True)
class TilePriorCertificate:
    """The safety witness (the AccelCertificate pattern): over the case family the guided
    optimum's (fits_cache, bottleneck) equals the exhaustive one's on EVERY shape --
    mismatches MUST be 0 -- and the node counts quantify the search reduction."""

    checked: int
    mismatches: int
    nodes_exhaustive: int
    nodes_guided: int

    @property
    def admitted(self) -> bool:
        return self.checked >= 1 and self.mismatches == 0

    @property
    def reduction(self) -> float:
        return 1.0 - self.nodes_guided / max(1, self.nodes_exhaustive)


def tile_prior_certificate(shapes: list, prior: FrozenTilePrior,
                           target: TargetProfile) -> TilePriorCertificate:
    checked = mism = ne = ng = 0
    for (M, N, K) in shapes:
        exact, n_ex = _exhaustive(M, N, K, target)
        guided, n_gd = guided_plan_matmul(M, N, K, target, prior)
        checked += 1
        ne += n_ex
        ng += n_gd
        if (guided.fits_cache, guided.bottleneck) != (exact.fits_cache, exact.bottleneck):
            mism += 1
    return TilePriorCertificate(checked=checked, mismatches=mism,
                                nodes_exhaustive=ne, nodes_guided=ng)


# --- D3 slice 2: the PERSISTED prior table, generation-tied to the calibloop ------------------

PRIOR_KIND = "bcir.tile_prior"
PRIOR_SCHEMA = 1


def save_tile_prior(prior: FrozenTilePrior, *, target_name: str, cal_gen: int) -> str:
    """Serialize a frozen prior as a self-describing envelope (the 0.4b decision-record
    pattern), GENERATION-TIED to the calibration table it was trained under: `cal_gen` is
    the microbench table's generation (`CalibratedProfile.cal_gen`), so a persisted prior
    names exactly which measured cost model produced its training labels."""
    import json  # noqa: PLC0415
    if not isinstance(prior, FrozenTilePrior):
        raise ValueError("tile prior must be a FrozenTilePrior")
    if (not isinstance(target_name, str) or not target_name or len(target_name) > 4096 or
            any(ord(ch) < 0x20 for ch in target_name)):
        raise ValueError("tile-prior target must be a bounded string")
    if (isinstance(cal_gen, bool) or not isinstance(cal_gen, int) or
            not 0 <= cal_gen <= (1 << 63) - 1):
        raise ValueError("tile-prior cal_gen must be a non-negative i63")
    return json.dumps({"kind": PRIOR_KIND, "schema": PRIOR_SCHEMA,
                       "target": target_name, "cal_gen": cal_gen,
                       "wq": list(prior.wq)}, indent=2, sort_keys=True)


def load_tile_prior(text: str, *, expect_target: str, expect_cal_gen: int) -> FrozenTilePrior:
    """Decode a persisted prior and REFUSE a stale one: a prior trained under another
    calibration generation (or target) proposes orderings from a cost model that no longer
    holds -- it must retrain, never silently apply (the calibloop's generation discipline;
    exact search would still keep the OPTIMUM safe, but a stale prior's search reduction is
    an unearned claim)."""
    d = strict_json_loads(text, "tile prior")
    fields = {"kind", "schema", "target", "cal_gen", "wq"}
    if not isinstance(d, dict) or set(d) != fields:
        raise ValueError(f"tile-prior fields must be exactly {sorted(fields)}")
    if d["kind"] != PRIOR_KIND:
        raise ValueError(f"not a {PRIOR_KIND} document (kind={d.get('kind')!r})")
    schema = d["schema"]
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise ValueError("tile-prior schema must be an integer")
    if schema > PRIOR_SCHEMA:
        raise ValueError(f"tile-prior schema v{schema} is newer than this build's "
                         f"v{PRIOR_SCHEMA}; upgrade BCIR to load this prior")
    if schema != PRIOR_SCHEMA:
        raise ValueError(f"unsupported tile-prior schema v{schema}")
    target = d["target"]
    if (not isinstance(target, str) or not target or len(target) > 4096 or
            any(ord(ch) < 0x20 for ch in target)):
        raise ValueError("tile-prior target must be a bounded string")
    if target != expect_target:
        raise ValueError(f"tile prior was trained for target {target!r}, "
                         f"not {expect_target!r} -- retrain")
    cal_gen = d["cal_gen"]
    if (isinstance(cal_gen, bool) or not isinstance(cal_gen, int) or
            not 0 <= cal_gen <= (1 << 63) - 1):
        raise ValueError("tile-prior cal_gen must be a non-negative i63")
    if (isinstance(expect_cal_gen, bool) or not isinstance(expect_cal_gen, int) or
            not 0 <= expect_cal_gen <= (1 << 63) - 1):
        raise ValueError("expected tile-prior cal_gen must be a non-negative i63")
    if cal_gen != expect_cal_gen:
        raise ValueError(f"tile prior is STALE: trained under cal_gen {cal_gen}, "
                         f"the live table is cal_gen {expect_cal_gen} -- retrain")
    weights = d["wq"]
    if not isinstance(weights, list):
        raise ValueError("tile-prior wq must be an array")
    return FrozenTilePrior(wq=tuple(weights))
