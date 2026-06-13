"""The propose-verify search accelerator: a learned ordering over an exact search.

The K_BCIR plan is found by an exact search over the legal candidate DAG
(`realize.optimize` / `rcsp.optimize_constrained`). This module makes that search
*faster* with a learned candidate ordering -- and **provably returns the same
optimum**, because the optimum value is invariant to the order in which candidates
are visited. That invariance is the safety property: a learned heuristic guides an
exact, bound-pruned branch-and-bound search; the search (and the R-law verifier)
are ground truth. The network proposes an order; the verifier disposes -- this is
the only place learning touches the (hot-ish) plan-time search, and it cannot
corrupt the result, only the work.

  * **Exact branch-and-bound** (`optimize_ordered`): DFS over the columns of the
    candidate DAG with an *admissible* lower bound (each remaining column's
    cheapest optimistic coupled cost) and budget feasibility pruning. With any
    candidate order it returns the same optimal score as `optimize_constrained`;
    a *good* order tightens the incumbent earlier, so the optimum is found sooner
    and more branches are pruned (fewer expansions).
  * **Learned ranker** (`LearnedRanker`): a tiny logistic model over candidate
    features (static score, width, vectorness, lane), trained on the exact
    optimizer's own choices (which candidate is on the optimal path) to order
    most-likely-optimal first. Frozen to Q8 integers for deterministic ordering.
  * **Equivalence certificate** (`accelerator_certificate`): checks that the
    ordered search reproduces the exact optimum across a test set (mismatches
    must be 0), witnessed by R9/R13 (`bcir.kbcir.search_accel`).

LEARNING-PLACEMENT LAW (LangRef Sec. 13): the ranker trains offline (floats) and
freezes to Q8. Crucially, *even a bad or stale ranker is safe* -- ordering never
changes the optimum, only the search effort -- so this is the lowest-risk way for
learning to reach the search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..model import Claim, Lane, Module
from .cost import MEMORY, N, Theta
from .rcsp import Budget
from .realize import (
    Candidate,
    ChosenStep,
    RealizationResult,
    _context_factor,
    _flatten,
    candidates_for,
)
from .weights import PERF, Policy, weights

Q8 = 256


@dataclass
class SearchStats:
    """The work an ordered search did (the acceleration signal)."""

    expansions: int = 0
    first_complete_cost: int | None = None   # cost of the FIRST complete plan found


# --- the exact branch-and-bound search (optimum invariant to candidate order) ----

def _columns(module: Module, h, theta: Theta, policy: Policy):
    """Per claim: (phase_id, claim, w_phase, candidates, static_scores)."""
    cols = []
    for phase_id, claim in _flatten(module):
        w = weights(h, theta, phase_id, policy)
        cost_rid = claim.rd[0] if claim.rd else claim.primary_rid
        resource = module.resource(cost_rid) if cost_rid is not None else None
        cands = candidates_for(claim, h, resource)
        scores = [c.base.dot(w) for c in cands]                 # static (no coupling)
        cols.append((phase_id, claim, w, cands, scores))
    return cols


def _optimistic(cand: Candidate, w) -> int:
    """A lower bound on a candidate's coupled cost: apply the most-favourable
    coupling (max fusion discount on memory, no thermal/power penalty). Always
    <= the true coupled cost (admissible)."""
    factor = [Q8] * N
    factor[MEMORY] = 192                                        # the only discount
    return cand.base.couple(tuple(factor)).dot(w)


def identity_order(cands, scores):
    return list(range(len(cands)))


def greedy_order(cands, scores):
    """Cheapest static cost first -- a strong hand heuristic baseline."""
    return sorted(range(len(cands)), key=lambda i: (scores[i], i))


def worst_order(cands, scores):
    return sorted(range(len(cands)), key=lambda i: (-scores[i], i))


def optimize_ordered(module: Module, h, theta: Theta, policy: Policy = PERF,
                     budget: Budget = Budget.unbounded(), order=greedy_order
                     ) -> tuple[RealizationResult, SearchStats]:
    """Exact branch-and-bound; `order(cands, scores)->index order` guides the DFS.
    Returns the optimal plan (== optimize_constrained's score) and the search
    stats. The optimum is invariant to `order`; only the stats change."""
    cols = _columns(module, h, theta, policy)
    if not cols:
        return RealizationResult([], 0), SearchStats(0, 0)
    ncols = len(cols)
    dims = budget.dims
    caps = dict(budget.caps)
    # Admissible suffix bound: remaining[col] = sum_{c>=col} min optimistic cost.
    opt_min = [min(_optimistic(c, w) for c in cands)
               for (_pid, _cl, w, cands, _s) in cols]
    remaining = [0] * (ncols + 1)
    for col in range(ncols - 1, -1, -1):
        remaining[col] = remaining[col + 1] + opt_min[col]

    stats = SearchStats()
    best = {"score": None, "chain": None}

    def dfs(col, prev_cand, acc_cost, acc_res, chain):
        if best["score"] is not None and acc_cost + remaining[col] >= best["score"]:
            return                                              # admissible-bound prune
        if col == ncols:
            if stats.first_complete_cost is None:
                stats.first_complete_cost = acc_cost
            if best["score"] is None or acc_cost < best["score"]:
                best["score"], best["chain"] = acc_cost, list(chain)
            return
        _pid, claim, w, cands, scores = cols[col]
        for i in order(cands, scores):
            cand = cands[i]
            coupled = cand.base.couple(_context_factor(prev_cand, cand, theta))
            step = coupled.dot(w)
            res = tuple(acc_res[k] + coupled.v[d] for k, d in enumerate(dims))
            if any(r > caps[d] for r, d in zip(res, dims) if d in caps):
                continue                                        # budget infeasible
            stats.expansions += 1
            chain.append((col, cand, step))
            dfs(col + 1, cand, acc_cost + step, res, chain)
            chain.pop()

    dfs(0, None, 0, (0,) * len(dims), [])
    steps = []
    if best["chain"] is not None:
        for (col, cand, step) in best["chain"]:
            pid, claim, _w, _c, _s = cols[col]
            steps.append(ChosenStep(claim.id, pid, cand, step))
    return RealizationResult(steps, best["score"] or 0), stats


# --- the learned ranker (orders candidates most-likely-optimal first) ------------

def candidate_features(cand: Candidate, static_score: int, score_scale: int) -> list[float]:
    return [static_score / max(1, score_scale),
            cand.width / 32.0,
            1.0 if cand.width > 1 else 0.0,
            int(cand.lane) / 5.0,
            1.0]


@dataclass
class LearnedRanker:
    """A logistic model over candidate features predicting P(on the optimal path).
    Orders candidates by descending probability -- most-likely-optimal first."""

    w: list = field(default_factory=lambda: [0.0] * 5)

    def _p(self, feats):
        z = sum(wi * xi for wi, xi in zip(self.w, feats))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def order(self, cands, scores):
        scale = max(scores) if scores else 1
        scored = [(self._p(candidate_features(c, scores[i], scale)), i)
                  for i, c in enumerate(cands)]
        return [i for _, i in sorted(scored, key=lambda t: (-t[0], t[1]))]

    def freeze(self) -> "FrozenRanker":
        return FrozenRanker(wq=[round(v * Q8) for v in self.w])


@dataclass(frozen=True)
class FrozenRanker:
    """Q8-frozen ranker: deterministic integer ordering for the certified rail."""

    wq: list

    def order(self, cands, scores):
        scale = max(scores) if scores else 1
        out = []
        for i, c in enumerate(cands):
            feats = candidate_features(c, scores[i], scale)
            z = sum(self.wq[k] * round(feats[k] * Q8) for k in range(len(self.wq))) >> 8
            out.append((z, i))                                  # higher z = try first
        return [i for _, i in sorted(out, key=lambda t: (-t[0], t[1]))]


def ranker_samples(module: Module, h, thetas: list, policy: Policy = PERF):
    """Training data from the EXACT optimizer: per column, the chosen candidate is
    a positive (on the optimal path), the rest negatives."""
    from .realize import optimize

    samples = []
    for theta in thetas:
        chosen = {s.claim_id: s.candidate.name for s in optimize(module, h, theta, policy).steps}
        for _pid, claim, w, cands, scores in _columns(module, h, theta, policy):
            scale = max(scores) if scores else 1
            for i, c in enumerate(cands):
                label = 1.0 if c.name == chosen.get(claim.id) else 0.0
                samples.append((candidate_features(c, scores[i], scale), label))
    return samples


def train_ranker(samples, epochs: int = 400, lr: float = 0.3) -> LearnedRanker:
    """Logistic-regression SGD on (features, on-optimal-path) -- deterministic."""
    r = LearnedRanker(w=[0.0] * 5)
    for _ in range(epochs):
        for feats, label in samples:
            err = r._p(feats) - label
            r.w = [wi - lr * err * xi for wi, xi in zip(r.w, feats)]
    return r


# --- the equivalence certificate (the accelerator must not change the optimum) ---

@dataclass(frozen=True)
class AccelCertificate:
    """The safety witness: an ordered search reproduced the exact optimum."""

    order_name: str
    checked: int
    mismatches: int

    @property
    def admitted(self) -> bool:
        return self.checked >= 1 and self.mismatches == 0


def accelerator_certificate(cases: list, order, order_name: str = "learned"
                            ) -> AccelCertificate:
    """Check the ordered B&B optimum == the exact `optimize_constrained` optimum
    over `cases` = [(module, h, theta, policy, budget)]. mismatches MUST be 0."""
    from .rcsp import optimize_constrained

    mism = 0
    for module, h, theta, policy, budget in cases:
        exact = optimize_constrained(module, h, theta, policy, budget).score
        fast, _ = optimize_ordered(module, h, theta, policy, budget, order)
        if fast.score != exact:
            mism += 1
    return AccelCertificate(order_name=order_name, checked=len(cases), mismatches=mism)
