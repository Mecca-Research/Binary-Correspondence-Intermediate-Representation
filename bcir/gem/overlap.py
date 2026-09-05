"""K_BCIR <-> GEM coupling: price plans under the canonical schedule -- M(pi, Theta).

Serial Sigma pricing (`realize.optimize`'s score) composes claims as one textual
chain; the GEM executor places them as concurrent work over the target's affinity
domains with a decoupled GGG tail. This module prices the *schedule* (the
M(pi, Theta) of the LangRef central equation) by READING the one canonical
schedule artifact (`gem.schedule.schedule_plan`, G1 / S1-A):

  - a plan's step costs are its durations (the same coupled edge costs the planner
    summed and R9 re-derives -- one predicate, `realize.edge_cost`);
  - they are placed by the hazard-honoring LPT/EFT dispatch the executors run
    (`schedule_eft` phase-barriered, `execute_tokens` token-pipelined), so the
    objective and the executor read the same slots;
  - the serial score is the sum of the durations, so `makespan + overlap_gain ==
    serial` and `0 <= makespan <= serial` hold by construction of the artifact.

Before this slice the price was a separate pricer -- fixed greedy waves, round-robin
affinity bins, in-bin re-coupling, the tail as a free parallel chain -- that the
executor never ran: the 2026-08-12 report's P0.1 measured it at 51,200 against the
executor's 25,700 on four independent claims (`pricing.eft.divergence` = 1.9922).
That pricer is kept as `price_waves_legacy`, the witness the harness fixture
still exhibits, and is read by nothing else.

The serial score is the degenerate case (one domain, no overlap): for a single-
claim module makespan == score exactly. `optimize_scheduled` is the minimal
select -> schedule -> re-price iteration: one deterministic sweep that adopts a
per-claim alternative only if it strictly lowers the canonical makespan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Claim, Module
from ..kbcir.realize import (
    Candidate,
    ChosenStep,
    RealizationResult,
    _flatten,
    edge_cost,
)
from ..kbcir.weights import PERF, Policy, weights
from .concurrency import _is_sparse, _topo_phase_ids, _wave_indices
from .schedule import GemSchedule, phase_hazards, schedule_eft, schedule_plan


@dataclass(frozen=True)
class ScheduledPrice:
    """M(pi, Theta) alongside the serial Sigma bound it generalizes."""

    makespan: int  # the canonical schedule artifact's makespan
    serial: int  # Sigma over the textual chain (the degenerate price)
    # The artifact the makespan was read from (None for the legacy witness). Derived
    # state: excluded from equality/repr, like RealizationResult.cand_map.
    schedule: GemSchedule | None = field(default=None, compare=False, repr=False)

    @property
    def overlap_gain(self) -> int:
        return self.serial - self.makespan


def _serial_result(
    module: Module, assignment: dict[int, Candidate], h, theta, policy: Policy
) -> RealizationResult:
    """Re-price a fixed assignment under the serial chain semantics through the planner's
    own edge predicate (`realize.edge_cost`, the number R9 re-derives per step), so the
    returned plan's score == Sigma of its step costs. Under the scope the plan was
    selected in, this reproduces the plan's own steps and score exactly."""
    steps: list[ChosenStep] = []
    total = 0
    prev: Candidate | None = None
    for phase_id, claim in _flatten(module):
        cand = assignment.get(claim.id)
        if cand is None:
            raise ValueError(f"the plan does not cover claim {claim.id} (R9 coverage)")
        cost = edge_cost(prev, cand, theta, weights(h, theta, phase_id, policy))
        steps.append(ChosenStep(claim.id, phase_id, cand, cost))
        total += cost
        prev = cand
    return RealizationResult(steps, total)


def _makespan(
    module: Module,
    assignment: dict[int, Candidate],
    h,
    theta,
    policy: Policy,
    mode: str = "eft",
) -> int:
    """M(pi, Theta) of an assignment: the canonical artifact's makespan over the
    assignment's serially re-priced step costs."""
    return schedule_plan(
        module, _serial_result(module, assignment, h, theta, policy), h, mode
    ).makespan


def price_scheduled(
    module: Module,
    result: RealizationResult,
    h,
    theta,
    policy: Policy = PERF,
    mode: str = "eft",
) -> ScheduledPrice:
    """Price a selected plan under the canonical schedule artifact (the M of the equation).

    The plan's assignment is re-priced serially under (h, theta, policy) -- under the
    plan's own scope that is the plan's own steps -- and placed by `schedule_plan`
    (`mode` "eft" phase-barriered, the default, or "tokens" pipelined). The returned
    `schedule` IS the artifact: its slots are the executor's, its makespan the price.
    """
    priced = _serial_result(module, result.by_claim(), h, theta, policy)
    sched = schedule_plan(module, priced, h, mode)
    return ScheduledPrice(makespan=sched.makespan, serial=priced.score, schedule=sched)


# --- the retired wave pricer, kept as the divergence witness ------------------------


def _chain_cost(chain: list[tuple[Claim, Candidate]], w_phase, theta) -> int:
    """Serial (min,+) composition of an in-bin chain re-coupled against its bin predecessor
    (the legacy pricer's discipline; the canonical artifact keeps the plan's own costs)."""
    total = 0
    prev: Candidate | None = None
    for _claim, cand in chain:
        total += edge_cost(prev, cand, theta, w_phase)
        prev = cand
    return total


def _legacy_wave_makespan(
    module: Module, assignment: dict[int, Candidate], h, theta, policy: Policy
) -> int:
    """The pre-G1 price: fixed greedy waves, round-robin bins, tail as a free chain."""
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


def price_waves_legacy(
    module: Module, result: RealizationResult, h, theta, policy: Policy = PERF
) -> ScheduledPrice:
    """The retired wave pricer (P0.1 of the 2026-08-12 report), kept ONLY as the
    divergence witness: it prices fixed greedy conflict waves with round-robin affinity
    bins, re-couples each claim against its in-bin predecessor, runs the GGG tail as a
    chain alongside the waves with no hazard edge across the streams, and ignores the
    bandwidth knee and every ordering fence. Nothing reads it but the harness fixture
    (`pricing.eft.divergence`, whose baseline it reproduces) and its tests; the price of
    a plan is `price_scheduled`."""
    assignment = result.by_claim()
    return ScheduledPrice(
        makespan=_legacy_wave_makespan(module, assignment, h, theta, policy),
        serial=result.score,
    )


def optimize_scheduled(
    module: Module, h, theta, policy: Policy = PERF
) -> tuple[RealizationResult, ScheduledPrice]:
    """Select -> schedule -> re-price, iterated once.

    Starts from the serial optimum, then sweeps each claim once (in flatten order),
    adopting the legal alternative that strictly lowers the canonical makespan most
    (deterministic first-best tie-break, earlier adoptions carried forward). The
    makespan is a placement of the plan's step costs, so only an alternative that
    SHORTENS a step it touches -- its own, or its textual successor's through the
    context coupling -- is placed and compared: a step that only lengthens cannot
    lower the makespan except through a list-scheduling anomaly, which is not a
    property of the plan. The hazard DAG is built once; each trial re-prices two
    steps and re-places. Returns the plan (serial-repriced, so verify_plan R9 holds)
    and its scheduled price -- the same artifact `price_scheduled` reads.
    """
    from ..kbcir.realize import fused_candidates, optimize

    result = optimize(module, h, theta, policy)
    if not result.steps:
        return result, ScheduledPrice(0, 0)
    flat = _flatten(module)
    if len(flat) != len(result.steps):
        raise ValueError("the serial optimum does not cover the module (R9 coverage)")

    cand_map = fused_candidates(module, h)  # fusion-aware alternatives (consistent costs)
    w_of = {pid: weights(h, theta, pid, policy) for pid, _claim in flat}
    assignment = dict(result.by_claim())
    cands = [assignment[claim.id] for _pid, claim in flat]
    ids = [claim.id for _pid, claim in flat]
    n = len(flat)

    def step_cost(index: int, cand: Candidate, prev: Candidate | None) -> int:
        return edge_cost(prev, cand, theta, w_of[flat[index][0]])

    costs = [step_cost(i, cands[i], cands[i - 1] if i else None) for i in range(n)]
    durations = {ids[i]: max(0, costs[i]) for i in range(n)}
    hazards = phase_hazards(module)
    best_m = schedule_eft(module, durations, h, hazards=hazards).makespan
    changed = False
    for i in range(n):
        current = cands[i]
        best_cand, best_trial, best_costs = current, best_m, (costs[i], None)
        prev = cands[i - 1] if i else None
        for alt in cand_map[ids[i]]:
            if alt == current:
                continue
            cost_i = step_cost(i, alt, prev)
            cost_next = step_cost(i + 1, cands[i + 1], alt) if i + 1 < n else None
            shorter = cost_i < costs[i] or (cost_next is not None and cost_next < costs[i + 1])
            if not shorter:
                continue
            saved = (durations[ids[i]], durations[ids[i + 1]] if cost_next is not None else None)
            durations[ids[i]] = max(0, cost_i)
            if cost_next is not None:
                durations[ids[i + 1]] = max(0, cost_next)
            m = schedule_eft(module, durations, h, hazards=hazards).makespan
            durations[ids[i]] = saved[0]
            if cost_next is not None:
                durations[ids[i + 1]] = saved[1]
            if m < best_trial:  # strict: the first alternative reaching the new minimum wins
                best_cand, best_trial, best_costs = alt, m, (cost_i, cost_next)
        if best_cand is not current:  # commit (carry the adoption into later claims' sweeps)
            cands[i] = best_cand
            assignment[ids[i]] = best_cand
            costs[i] = best_costs[0]
            durations[ids[i]] = max(0, costs[i])
            if best_costs[1] is not None:
                costs[i + 1] = best_costs[1]
                durations[ids[i + 1]] = max(0, costs[i + 1])
            best_m = best_trial
            changed = True

    if changed:
        result = _serial_result(module, assignment, h, theta, policy)
    return result, price_scheduled(module, result, h, theta, policy)
