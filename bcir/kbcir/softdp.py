"""kbcir.softdp -- the temperature dial: a soft, differentiable twin of optimize.

`realize.optimize` solves the K_BCIR plan as a tropical (min,+) shortest path:
the *zero-temperature* limit of a log-sum-exp dynamic program. This module is the
finite-temperature generalization -- the same layered candidate DAG, scored with

    F_T(G|H,Theta) = -T * log  sum_{pi in Legal}  exp( -score(pi) / T )

the Gibbs **free energy** over realization plans. The identity

    softmin_T(costs) = -T log sum_i exp(-cost_i/T)  -->  min_i cost_i   as T -> 0

makes this the exact soft counterpart of the Viterbi/min-plus optimizer (the same
relation that links the forward algorithm to Viterbi in HMMs, and soft-Bellman to
hard-Bellman in max-entropy RL). Three things fall out:

  * **T -> 0** recovers the deterministic optimizer *exactly* -- at T == 0 this
    module delegates to the integer `optimize`, so every pinned score is
    bit-identical. F_0 == the hard K_BCIR score; the marginals are one-hot on the
    chosen plan.
  * **T > 0** yields a *posterior over plans* (Solomonoff's universal-prior soft
    selection / a Bayesian plan posterior / a softmax layer) via a soft
    forward-backward pass: per-claim candidate **marginals** and the **expected
    cost vector**.
  * **Differentiability.** F_T is smooth in the edge weights, so the compiler
    becomes differentiable. The gradient of the free energy w.r.t. a uniformly
    applied policy weight w_d is the expected sufficient statistic
    `dF/dw_d = E_pi[ C_d ]` -- the d-th component of `expected_cost` -- computed
    here exactly from edge marginals, no autograd needed.

LEARNING-PLACEMENT LAW (LangRef Sec. 13): this is an **L2/L3 offline** organ. It
uses floats and is NOT deterministic across architectures at T > 0; it must never
run on the L0/L1 hot path. Learning happens here at T > 0; you anneal and *freeze*
to a T = 0 integer table for deployment (the distillation step). The T == 0 path
is the only one that touches the certified rail, and it is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..model import Module
from .cost import N, Theta
from .realize import (
    Candidate,
    RealizationResult,
    _context_factor,
    _flatten,
    candidates_for,
    fused_candidates,
    optimize,
)
from .weights import PERF, Policy, weights


@dataclass(frozen=True)
class SoftResult:
    """The finite-temperature plan distribution (the soft twin of RealizationResult)."""

    temperature: float
    free_energy: float  # F_T = softmin path cost; <= hard_score
    hard_score: int  # the T=0 tropical score (the gap reference)
    marginals: dict[int, dict[str, float]]  # claim id -> {candidate name: P(selected)}
    expected_cost: tuple[float, ...]  # E_pi[ total coupled 12-d cost vector ]
    map_by_claim: dict[int, str]  # argmax-marginal candidate per claim

    @property
    def gap(self) -> float:
        """hard_score - F_T >= 0: the free energy never exceeds the hard minimum."""
        return self.hard_score - self.free_energy

    def expected_score(self, h, theta: Theta, policy: Policy = PERF, phase_id: int = 0) -> float:
        """E_pi[score] = E[C] . w >= F_T (Jensen/Gibbs bound; equality at T=0)."""
        w = weights(h, theta, phase_id, policy)
        return float(sum(c * wi for c, wi in zip(self.expected_cost, w)))


def _softmin(vals: list[float], T: float) -> float:
    """-T log sum exp(-v/T), computed stably (factor out the min; result <= min)."""
    m = min(vals)
    s = sum(math.exp(-(v - m) / T) for v in vals)
    return m - T * math.log(s)


def _build_dag(module: Module, h, theta: Theta, policy: Policy):
    """Replicate realize.optimize's layered DAG, also recording each edge's coupled
    cost vector. Returns (n_nodes, columns, out_edges, in_edges) where an edge is
    (other_node, scalar_weight, coupled_cost_vector)."""
    flat = _flatten(module)
    cand_map = fused_candidates(module, h)  # producer->consumer deforestation (shared)
    n0 = 1
    node_meta: list[tuple | None] = [None]  # 0 = SOURCE
    out_edges: list[list[tuple[int, float, tuple]]] = [[]]
    in_edges: list[list[tuple[int, float, tuple]]] = [[]]
    columns: list[tuple[int, list[int], list[Candidate]]] = []  # (claim_id, nodes, cands)
    prev_nodes: list[int] = [0]
    prev_cands: list[Candidate | None] = [None]
    zero = (0,) * N

    for phase_id, claim in flat:
        w_phase = weights(h, theta, phase_id, policy)
        col_nodes: list[int] = []
        col_cands: list[Candidate] = []
        for cand in cand_map[claim.id]:
            nid = len(node_meta)
            node_meta.append((phase_id, claim, cand))
            out_edges.append([])
            in_edges.append([])
            for pu, pc in zip(prev_nodes, prev_cands):
                factor = _context_factor(pc, cand, theta)
                cvec = cand.base.couple(factor).v
                w = float(sum(c * wi for c, wi in zip(cvec, w_phase)))
                out_edges[pu].append((nid, w, cvec))
                in_edges[nid].append((pu, w, cvec))
            col_nodes.append(nid)
            col_cands.append(cand)
        columns.append((claim.id, col_nodes, col_cands))
        prev_nodes, prev_cands = col_nodes, col_cands

    sink = len(node_meta)
    node_meta.append(None)
    out_edges.append([])
    in_edges.append([])
    for pu in prev_nodes:
        out_edges[pu].append((sink, 0.0, zero))
        in_edges[sink].append((pu, 0.0, zero))
    return sink + 1, sink, columns, out_edges, in_edges


def _expected_cost_at_zero(
    module: Module, h, theta: Theta, policy: Policy, result: RealizationResult
) -> tuple[float, ...]:
    """The coupled cost vector summed along the hard-chosen path (the T=0 E[C])."""
    acc = [0.0] * N
    prev: Candidate | None = None
    for step in result.steps:
        cand = step.candidate
        factor = _context_factor(prev, cand, theta)
        for d, c in enumerate(cand.base.couple(factor).v):
            acc[d] += c
        prev = cand
    return tuple(acc)


def softselect(
    module: Module, h, theta: Theta, policy: Policy = PERF, temperature: float = 0.0
) -> SoftResult:
    """The finite-temperature plan distribution. At temperature 0 this is the
    exact integer optimizer; at T > 0 it runs the soft forward-backward DP."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    hard = optimize(module, h, theta, policy)

    if temperature == 0:
        marg = {s.claim_id: {s.candidate.name: 1.0} for s in hard.steps}
        return SoftResult(
            temperature=0.0,
            free_energy=float(hard.score),
            hard_score=hard.score,
            marginals=marg,
            map_by_claim={s.claim_id: s.candidate.name for s in hard.steps},
            expected_cost=_expected_cost_at_zero(module, h, theta, policy, hard),
        )

    n, sink, columns, out_edges, in_edges = _build_dag(module, h, theta, policy)
    if not columns:
        return SoftResult(temperature, 0.0, 0, {}, (0.0,) * N, {})

    T = float(temperature)
    alpha = [0.0] * n  # soft cost SOURCE -> node
    for node in range(1, n):
        ins = in_edges[node]
        if ins:
            alpha[node] = _softmin([alpha[u] + w for u, w, _ in ins], T)
    beta = [0.0] * n  # soft cost node -> SINK
    for node in range(n - 2, -1, -1):
        outs = out_edges[node]
        if outs:
            beta[node] = _softmin([w + beta[v] for v, w, _ in outs], T)
    F = alpha[sink]

    # Per-claim candidate marginals: P(path through node) = exp(-(alpha+beta-F)/T).
    marginals: dict[int, dict[str, float]] = {}
    map_by_claim: dict[int, str] = {}
    for claim_id, nodes, cands in columns:
        probs = [math.exp(-(alpha[nd] + beta[nd] - F) / T) for nd in nodes]
        total = sum(probs) or 1.0
        dist = {cand.name: p / total for cand, p in zip(cands, probs)}
        marginals[claim_id] = dist
        map_by_claim[claim_id] = max(dist, key=dist.get)

    # Expected cost vector via edge marginals: E[C] = sum_edges P(edge) * cvec.
    acc = [0.0] * N
    for u in range(n):
        for v, w, cvec in out_edges[u]:
            if any(cvec):
                p = math.exp(-(alpha[u] + w + beta[v] - F) / T)
                for d, c in enumerate(cvec):
                    acc[d] += p * c
    return SoftResult(
        temperature=T,
        free_energy=F,
        hard_score=hard.score,
        marginals=marginals,
        expected_cost=tuple(acc),
        map_by_claim=map_by_claim,
    )


def free_energy(
    module: Module, h, theta: Theta, policy: Policy = PERF, temperature: float = 0.0
) -> float:
    """F_T(G|H,Theta) -- the Gibbs free energy over realization plans."""
    return softselect(module, h, theta, policy, temperature).free_energy


def grad_free_energy_wrt_weights(soft: SoftResult) -> tuple[float, ...]:
    """dF_T/dw = E_pi[C], the expected sufficient statistic (exact for a uniformly
    applied policy weight vector). The analytic gradient that makes the compiler
    differentiable -- backprop into the policy weights without autograd."""
    return soft.expected_cost


def temperature_sweep(
    module: Module,
    h,
    theta: Theta,
    policy: Policy = PERF,
    temperatures=(0.0, 1.0, 100.0, 1000.0, 10000.0),
) -> list[tuple[float, float]]:
    """(T, F_T) across an annealing schedule. F_T rises toward the hard minimum as
    T -> 0 (always F_T <= hard_score); the dashboard for the anneal-and-freeze step."""
    return [(float(t), free_energy(module, h, theta, policy, t)) for t in temperatures]
