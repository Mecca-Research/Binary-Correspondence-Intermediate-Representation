"""K_BCIR <-> GEM coupling: price plans under wave overlap -- M(pi, Theta).

Serial Sigma pricing (`realize.optimize`'s score) composes claims as one textual
chain; the CT2 executor actually runs them as concurrent waves with a decoupled
GGG tail. This module prices the *schedule* (the M(pi, Theta) of the LangRef
central equation):

  - claims co-executing in a wave on distinct affinity domains combine with max;
  - claims time-sharing one domain (a bin) serialize with (min,+) (+) -- and the
    context coupling f_i is recomputed against the claim's *actual* in-bin
    predecessor, not its textual one (a fusion discount only applies to claims
    that really run back-to-back);
  - the GGG tail runs decoupled alongside the waves: phase cost = max(waves, tail);
  - phases compose serially in topological order.

The serial score is the degenerate case (one domain, textual chaining): for a
single-claim module makespan == score exactly. `optimize_scheduled` is the
minimal select -> schedule -> re-price iteration: one deterministic sweep that
adopts a per-claim alternative only if it strictly lowers the scheduled makespan.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Claim, Module
from ..kbcir.realize import (
    Candidate,
    ChosenStep,
    RealizationResult,
    _context_factor,
    _flatten,
    candidates_for,
)
from ..kbcir.weights import PERF, Policy, weights
from .concurrency import _is_sparse, _topo_phase_ids, _wave_indices


@dataclass(frozen=True)
class ScheduledPrice:
    """M(pi, Theta) alongside the serial Sigma bound it generalizes."""

    makespan: int   # (max,+) over the wave schedule
    serial: int     # Sigma over the textual chain (the degenerate price)

    @property
    def overlap_gain(self) -> int:
        return self.serial - self.makespan


def _resource_for(module: Module, claim: Claim):
    cost_rid = claim.rd[0] if claim.rd else claim.primary_rid
    return module.resource(cost_rid) if cost_rid is not None else None


def _chain_cost(chain: list[tuple[Claim, Candidate]], w_phase, theta) -> int:
    """Serial (min,+) composition of a real execution chain with true coupling."""
    total = 0
    prev: Candidate | None = None
    for _claim, cand in chain:
        factor = _context_factor(prev, cand, theta)
        total += cand.base.couple(factor).dot(w_phase)
        prev = cand
    return total


def _makespan(module: Module, assignment: dict[int, Candidate], h, theta,
              policy: Policy) -> int:
    """M(pi, Theta): max over parallel bins/tail, sum over series, per phase."""
    domains = max(1, getattr(h, "affinity_domains", 1))
    pmap = module.phase_map()
    total = 0

    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        claims = [c for c in claims if c.id in assignment]
        main = [c for c in claims if not _is_sparse(c)]
        tail = [c for c in claims if _is_sparse(c)]
        w_phase = weights(h, theta, pid, policy)

        # Greedy wave assignment (identical to concurrency.schedule_concurrent).
        wave_of, wave_members = _wave_indices(main)

        main_total = 0
        nwaves = (max(wave_of.values()) + 1) if wave_of else 0
        for w in range(nwaves):
            members = wave_members.get(w, ())
            # Round-robin affinity bins; claims in one bin execute back-to-back.
            bins: dict[int, list[tuple[Claim, Candidate]]] = {}
            for slot, c in enumerate(members):
                bins.setdefault(slot % domains, []).append((c, assignment[c.id]))
            main_total += max(
                (_chain_cost(chain, w_phase, theta) for chain in bins.values()),
                default=0,
            )

        tail_total = _chain_cost([(c, assignment[c.id]) for c in tail], w_phase, theta)
        total += max(main_total, tail_total)
    return total


def _serial_result(module: Module, assignment: dict[int, Candidate], h, theta,
                   policy: Policy) -> RealizationResult:
    """Re-price a fixed assignment under the serial chain semantics (so the
    returned plan's score == Sigma of its step costs: R9-consistent)."""
    steps: list[ChosenStep] = []
    total = 0
    prev: Candidate | None = None
    for phase_id, claim in _flatten(module):
        cand = assignment[claim.id]
        w_phase = weights(h, theta, phase_id, policy)
        cost = cand.base.couple(_context_factor(prev, cand, theta)).dot(w_phase)
        steps.append(ChosenStep(claim.id, phase_id, cand, cost))
        total += cost
        prev = cand
    return RealizationResult(steps, total)


def price_scheduled(module: Module, result: RealizationResult, h, theta,
                    policy: Policy = PERF) -> ScheduledPrice:
    """Price a selected plan under the CT2 wave schedule (the M of the equation)."""
    assignment = result.by_claim()
    return ScheduledPrice(
        makespan=_makespan(module, assignment, h, theta, policy),
        serial=result.score,
    )


def optimize_scheduled(module: Module, h, theta, policy: Policy = PERF
                       ) -> tuple[RealizationResult, ScheduledPrice]:
    """Select -> schedule -> re-price, iterated once.

    Starts from the serial optimum, then sweeps each claim once (in flatten
    order), adopting the legal alternative that strictly lowers the scheduled
    makespan most (deterministic first-best tie-break). Returns the plan
    (serial-repriced, so verify_plan R9 holds) and its scheduled price.
    """
    from ..kbcir.realize import fused_candidates, optimize

    result = optimize(module, h, theta, policy)
    assignment = dict(result.by_claim())
    if not assignment:
        return result, ScheduledPrice(0, 0)

    cand_map = fused_candidates(module, h)    # fusion-aware alternatives (consistent costs)
    best_m = _makespan(module, assignment, h, theta, policy)
    changed = False
    for _phase_id, claim in _flatten(module):
        current = assignment[claim.id]
        best_cand, best_trial = current, best_m
        for alt in cand_map[claim.id]:
            if alt == current:
                continue
            trial = dict(assignment)
            trial[claim.id] = alt
            m = _makespan(module, trial, h, theta, policy)
            if m < best_trial:
                best_cand, best_trial = alt, m
        if best_cand is not current:
            assignment[claim.id] = best_cand
            best_m = best_trial
            changed = True

    if changed:
        result = _serial_result(module, assignment, h, theta, policy)
    return result, ScheduledPrice(makespan=best_m, serial=result.score)
