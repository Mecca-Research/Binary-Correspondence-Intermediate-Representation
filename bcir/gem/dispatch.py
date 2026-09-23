"""The dispatch law (G12 / S2-C): a deterministic choice of solver the certificate can name.

"A solver portfolio" (the 2026-08-12 report's section 11.3) is only an architecture once the
choice of solver is a function -- of the region kind, the instance size, the certificate
class the caller asks for and the work budget it grants -- that a certificate records with
the budget, the stop reason and the bound source. `dispatch` is that function: a table, not a
heuristic, so two callers with equal requests get the same rail and the same solver, and a
reader of the certificate can rerun exactly what was run.

Three laws hold across every rail:

  * **work-unit budgets.** A budget is counted in the solver's own units -- node expansions
    (`exact_schedule`), candidate placements (`exact_layout`), assignments (`exact_selection`)
    -- never seconds, so a certificate class means the same thing on every host and two runs
    with equal budgets and inputs produce identical plans and stop reasons.
  * **an incumbent first.** Every proof-rail solver starts from its fast rail's answer, so a
    legal plan exists at every interruption point, budget zero included.
  * **resumable state.** Every proof-rail solver returns a content-addressed search state,
    and continuing from it with a fresh budget reproduces the uninterrupted run.

The learned ranker (`kbcir.moegate`, the L2 gate over the policy portfolio) may ORDER what a
rail enumerates and may never remove a member: `ranked` holds any ranker to a permutation of
the census, so the set of candidates a certificate ranges over is the same with the ranker and
without it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Module

REGION_KINDS = ("path", "schedule", "memory", "selection")
CERTIFICATE_CLASSES = ("TMSAO-1", "TMSAO-2", "TMSAO-3", "TMSAO-4")
RAILS = ("fast", "proof", "measured")

#: (region kind, rail) -> (solver, its work unit). The fast rail's unit is what it places once.
#: The measured rail (G13) ranks whole plans by the corpus's evidence, so it exists for the
#: two region kinds whose result IS a whole plan; a schedule or memory region has no measured
#: solver and a measured request for one falls to the fast rail with the reason.
SOLVERS: dict[tuple[str, str], tuple[str, str]] = {
    # additive layered candidates: the min-plus path is exact, so the fast rail IS the proof
    # rail (the report's section 11.3, first row)
    ("path", "fast"): ("optimize", "relaxations"),
    ("path", "proof"): ("optimize", "relaxations"),
    ("path", "measured"): ("measured_best", "samples"),
    ("schedule", "fast"): ("schedule_eft", "placements"),
    ("schedule", "proof"): ("exact_schedule", "expansions"),
    ("memory", "fast"): ("first_fit_layout", "placements"),
    ("memory", "proof"): ("exact_layout", "candidate placements"),
    ("selection", "fast"): ("optimize_scheduled", "trials"),
    ("selection", "proof"): ("exact_selection", "assignments"),
    ("selection", "measured"): ("measured_best", "samples"),
}

#: The instance size (claims, items, claims) up to which the proof rail is expected to close
#: within a default budget. Above it the proof rail is still dispatched -- the budget bounds
#: the work -- but the decision says a budget stop (TMSAO-2) is what to expect.
BOUNDED_SIZE = {
    "path": 1 << 62,
    "schedule": 64,
    "memory": 512,
    "selection": 12,
}  # a DP closes at any size

STOP_REASONS = ("optimal", "budget", "heuristic", "measured")


@dataclass(frozen=True)
class DispatchRequest:
    """What a caller asks for: a region kind, its size, the class wanted and the budget granted."""

    kind: str
    size: int
    requested: str = "TMSAO-2"
    budget: int = 200_000
    #: The declared workload's digest (`kbcir.workload.Workload.digest()`, G13), or None when
    #: no workload was declared -- a measured-best claim ranges over W, so the law needs it.
    workload: str | None = None
    #: How many measured plans the corpus holds for (program, target, workload): the law
    #: dispatches the measured rail only over evidence that exists.
    evidence: int = 0

    def __post_init__(self) -> None:
        if self.kind not in REGION_KINDS:
            raise ValueError(f"region kind must be one of {REGION_KINDS}")
        if self.requested not in CERTIFICATE_CLASSES:
            raise ValueError(f"requested class must be one of {CERTIFICATE_CLASSES}")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("instance size must be a non-negative integer")
        if not isinstance(self.budget, int) or isinstance(self.budget, bool) or self.budget < 0:
            raise ValueError("work budget must be a non-negative integer (work units)")
        if self.workload is not None and (not isinstance(self.workload, str) or not self.workload):
            raise ValueError("workload must be the declared workload's digest, or None")
        if (
            not isinstance(self.evidence, int)
            or isinstance(self.evidence, bool)
            or self.evidence < 0
        ):
            raise ValueError("evidence must be a non-negative count of measured plans")


@dataclass(frozen=True)
class DispatchDecision:
    """The law's answer: the rail and solver, the units its budget is counted in, the class
    the rail can grant at best on this instance, and why."""

    kind: str
    size: int
    requested: str
    budget: int
    rail: str
    solver: str
    units: str
    expected: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "size": self.size,
            "requested": self.requested,
            "budget": self.budget,
            "rail": self.rail,
            "solver": self.solver,
            "units": self.units,
            "expected": self.expected,
            "reason": self.reason,
        }


def dispatch(request: DispatchRequest) -> DispatchDecision:
    """The dispatch law. A pure table over the request:

    * TMSAO-4 requested, or a zero budget: the fast rail (a heuristic incumbent, legality
      and reproducibility only);
    * TMSAO-3 requested: the measured rail (G13) when the request declares a workload, the
      region kind has a measured solver (a whole plan: `path` or `selection`) and the corpus
      holds evidence for the scope; otherwise the fast rail, with the reason naming which of
      the three is missing. The rail ranks the census by evidence; the class it can grant is
      still the ladder's (`kbcir.scope.certificate_class_allowed`), which holds a measured
      claim to the two-target rule;
    * TMSAO-1 or TMSAO-2 requested with a budget: the proof rail, expected to close
      (TMSAO-1) up to the bounded size and to stop on its budget with a stated gap
      (TMSAO-2) above it.
    """
    fast_solver, fast_units = SOLVERS[(request.kind, "fast")]
    proof_solver, proof_units = SOLVERS[(request.kind, "proof")]
    if request.requested == "TMSAO-4":
        return DispatchDecision(
            request.kind,
            request.size,
            request.requested,
            request.budget,
            "fast",
            fast_solver,
            fast_units,
            "TMSAO-4",
            "a heuristic incumbent was asked for: the fast rail places once",
        )
    if request.budget == 0:
        return DispatchDecision(
            request.kind,
            request.size,
            request.requested,
            0,
            "fast",
            fast_solver,
            fast_units,
            "TMSAO-4",
            "no work budget was granted: the fast rail places once and no bound is searched",
        )
    if request.requested == "TMSAO-3":
        measured = SOLVERS.get((request.kind, "measured"))
        if request.workload is None:
            why = (
                "a measured-best claim ranges over the workload component W (G13) and none "
                "was declared: the fast rail places once"
            )
        elif measured is None:
            why = (
                f"the measured rail ranks whole plans and a {request.kind} region is not one "
                "(no measured solver): the fast rail places once"
            )
        elif request.evidence == 0:
            why = (
                "the corpus holds no measured plan for this (program, target, workload): "
                "the fast rail places once"
            )
        else:
            return DispatchDecision(
                request.kind,
                request.size,
                request.requested,
                request.budget,
                "measured",
                measured[0],
                measured[1],
                "TMSAO-3",
                f"the measured rail ranks the census by the corpus's {request.evidence} "
                "measured plans for this scope; TMSAO-3 is granted only under the "
                "two-target rule (attested silicon on two materially different targets "
                "with counters), TMSAO-4 with the reason otherwise",
            )
        return DispatchDecision(
            request.kind,
            request.size,
            request.requested,
            request.budget,
            "fast",
            fast_solver,
            fast_units,
            "TMSAO-4",
            why,
        )
    if request.size <= BOUNDED_SIZE[request.kind]:
        expected, why = (
            "TMSAO-1",
            f"the proof rail is expected to close at {request.size} <= "
            f"{BOUNDED_SIZE[request.kind]} {proof_units.split()[-1]} within the budget",
        )
    else:
        expected, why = (
            "TMSAO-2",
            f"the proof rail is bounded by its budget above {BOUNDED_SIZE[request.kind]}: "
            "expect a budget stop with an explicit gap",
        )
    return DispatchDecision(
        request.kind,
        request.size,
        request.requested,
        request.budget,
        "proof",
        proof_solver,
        proof_units,
        expected,
        why,
    )


@dataclass(frozen=True)
class DispatchRecord:
    """What a certificate records about the run: the decision, how it stopped, the units it
    spent, where its bound came from and the class it was granted."""

    decision: DispatchDecision
    stop_reason: str
    spent: int
    bound_source: str
    granted: str

    def __post_init__(self) -> None:
        if self.stop_reason not in STOP_REASONS:
            raise ValueError(f"stop reason must be one of {STOP_REASONS}")
        if self.granted not in CERTIFICATE_CLASSES:
            raise ValueError(f"granted class must be one of {CERTIFICATE_CLASSES}")

    def to_dict(self) -> dict:
        return {
            **self.decision.to_dict(),
            "stop_reason": self.stop_reason,
            "spent": self.spent,
            "bound_source": self.bound_source,
            "granted": self.granted,
        }


def _strongest(bounds) -> str:
    best = None
    for bound in bounds:
        if best is None or bound.value > best.value:
            best = bound
    return best.name if best is not None else "none"


# --- the runners: one per region kind --------------------------------------------------------


def solve_path(request: DispatchRequest, module: Module, h, theta, policy):
    """Dispatch a path region (additive layered candidates): the min-plus dynamic program is
    exact over the candidate census whatever the rail asked for, so the record grants
    TMSAO-1 when a proof was asked and TMSAO-4 when only a heuristic incumbent was."""
    from ..kbcir.realize import optimize

    decision = dispatch(request)
    result = optimize(module, h, theta, policy)
    relaxations = (
        sum(len(result.cand_map.get(step.claim_id, ())) for step in result.steps)
        if result.cand_map
        else len(result.steps)
    )
    granted = "TMSAO-4" if decision.rail == "fast" else "TMSAO-1"
    record = DispatchRecord(
        decision,
        "heuristic" if decision.rail == "fast" else "optimal",
        relaxations,
        "min-plus path",
        granted,
    )
    return result, record


def solve_schedule(
    request: DispatchRequest, module: Module, durations: dict[int, int], target=None, *, resume=None
):
    """Dispatch a schedule region: the fast rail returns the artifact (`schedule_eft`), the
    proof rail an `ExactSchedule` (with its state). Returns (result, record)."""
    from .exact import exact_schedule
    from .schedule import schedule_eft

    decision = dispatch(request)
    if decision.rail == "fast":
        sched = schedule_eft(module, durations, target)
        record = DispatchRecord(decision, "heuristic", len(sched.slots), "none", "TMSAO-4")
        return sched, record
    exact = exact_schedule(module, durations, target, budget=decision.budget, resume=resume)
    granted = "TMSAO-1" if exact.optimal else "TMSAO-2"
    record = DispatchRecord(
        decision, exact.stop_reason, exact.expansions, _strongest(exact.bounds), granted
    )
    return exact, record


def solve_memory(request: DispatchRequest, items, *, resume=None):
    """Dispatch a memory region: first-fit or the bounded exact layout over `LayoutItem`s."""
    from ..kbcir.static_memory import exact_layout, first_fit_layout

    decision = dispatch(request)
    rows = list(items)
    if decision.rail == "fast":
        layout = first_fit_layout(rows)
        return layout, DispatchRecord(
            decision, "heuristic", len(rows), "concurrent-live", "TMSAO-4"
        )
    layout = exact_layout(rows, max(1, decision.budget), resume=resume)
    granted = "TMSAO-1" if layout.stop_reason == "optimal" else "TMSAO-2"
    return layout, DispatchRecord(
        decision, layout.stop_reason, layout.expansions, "concurrent-live", granted
    )


def solve_selection(request: DispatchRequest, module: Module, h, theta, policy, *, resume=None):
    """Dispatch a selection region: the one-sweep re-selection or the exhaustive enumeration."""
    from .exact import exact_selection
    from .overlap import optimize_scheduled

    decision = dispatch(request)
    if decision.rail == "fast":
        stats: dict = {}
        result, price = optimize_scheduled(module, h, theta, policy, stats=stats)
        return (result, price), DispatchRecord(
            decision, "heuristic", stats.get("trials", 0), "none", "TMSAO-4"
        )
    selection = exact_selection(module, h, theta, policy, limit=decision.budget, resume=resume)
    granted = "TMSAO-1" if selection.stop_reason == "optimal" else "TMSAO-2"
    return selection, DispatchRecord(
        decision, selection.stop_reason, selection.assignments, "enumeration", granted
    )


def solve_measured(
    request: DispatchRequest, corpus, module: Module, h, theta, census, *, workload, incumbent=None
):
    """Dispatch a measured request over whole plans (G13): the measured rail ranks the census
    (policies) by the corpus's pooled median wall time for (program, target, workload, Theta)
    and returns the measured-best policy's plan RE-DERIVED from the planner -- evidence whose
    plan digest the planner no longer reproduces is stale and is not used, and the corpus
    never decides legality (the plan is `optimize`'s, verified like any other). Falls to the
    fast rail (the incumbent policy's plan, placed once) when the law does or when every
    evidence row is stale. Returns `((policy, result), record)`; the record's `granted` is
    the rail's own claim under the two-target rule (attested silicon on two targets), the
    certificate re-judges it against the declared scope."""
    from ..kbcir.measured import assignment_digest, pooled_median, scope_key, theta_key
    from ..kbcir.realize import optimize
    from ..kbcir.weights import PERF

    decision = dispatch(request)
    members = list(census)
    fallback = incumbent if incumbent is not None else (members[0] if members else PERF)

    def once():
        result = optimize(module, h, theta, fallback)
        record = DispatchRecord(decision, "heuristic", len(result.steps), "none", "TMSAO-4")
        return (fallback, result), record

    if decision.rail != "measured":
        return once()
    program, target, workload_digest = scope_key(module, h, workload)
    episode = theta_key(theta)
    best = None
    spent = 0
    for policy in members:
        rows = corpus.lookup(program, target, workload_digest, theta=episode, policy=policy.name)
        if not rows:
            continue
        result = optimize(module, h, theta, policy)
        digest = assignment_digest(result)
        live = [row for row in rows if row.plan == digest]  # stale evidence is not used
        if not live:
            continue
        spent += sum(len(row.samples) for row in live)
        key = (pooled_median(live), policy.name)
        if best is None or key < best[0]:
            best = (key, policy, result, live)
    if best is None:
        return once()
    _, policy, result, live = best
    silicon = all(row.silicon for row in live) and corpus.physical_targets() >= 2
    record = DispatchRecord(
        decision, "measured", spent, "corpus median", "TMSAO-3" if silicon else "TMSAO-4"
    )
    return (policy, result), record


# --- the census and the ranker ----------------------------------------------------------------


class CensusError(ValueError):
    """A ranker changed the census: a member removed, added or duplicated."""


def ranked(census, ranker=None) -> list:
    """Order `census` by `ranker` (a callable from a list to a list) and hold the ranker to a
    permutation: the learned ranker may prefer, never remove. Without a ranker the census is
    returned in its own order."""
    members = list(census)
    if ranker is None:
        return members
    order = list(ranker(list(members)))
    if len(order) != len(members) or sorted(map(repr, order)) != sorted(map(repr, members)):
        missing = [m for m in members if m not in order]
        extra = [m for m in order if m not in members]
        raise CensusError(f"the ranker changed the census: removed {missing!r}, added {extra!r}")
    return order


def policy_ranking(gate, module: Module, theta, portfolio=None) -> list:
    """The learned gate as a ranker over the policy portfolio: the portfolio's entries ordered
    by the gate's weight for this (module, Theta), every entry kept."""
    from ..kbcir.portfolio import PolicyPortfolio

    if portfolio is None:
        portfolio = PolicyPortfolio.default()
    census = [entry.policy for entry in portfolio.entries.values()]
    weights = gate.distribution(module, theta)
    by_name = {policy.name: index for index, policy in enumerate(getattr(gate, "experts", []))}

    def ranker(members: list) -> list:
        def key(policy):
            index = by_name.get(policy.name)
            return (
                -(weights[index] if index is not None and index < len(weights) else 0),
                policy.name,
            )

        return sorted(members, key=key)

    return ranked(census, ranker)


__all__ = [
    "BOUNDED_SIZE",
    "CERTIFICATE_CLASSES",
    "RAILS",
    "REGION_KINDS",
    "SOLVERS",
    "STOP_REASONS",
    "CensusError",
    "DispatchDecision",
    "DispatchRecord",
    "DispatchRequest",
    "dispatch",
    "policy_ranking",
    "ranked",
    "solve_measured",
    "solve_memory",
    "solve_path",
    "solve_schedule",
    "solve_selection",
]
