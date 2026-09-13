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

REGION_KINDS = ("schedule", "memory", "selection")
CERTIFICATE_CLASSES = ("TMSAO-1", "TMSAO-2", "TMSAO-3", "TMSAO-4")
RAILS = ("fast", "proof")

#: (region kind, rail) -> (solver, its work unit). The fast rail's unit is what it places once.
SOLVERS: dict[tuple[str, str], tuple[str, str]] = {
    ("schedule", "fast"): ("schedule_eft", "placements"),
    ("schedule", "proof"): ("exact_schedule", "expansions"),
    ("memory", "fast"): ("first_fit_layout", "placements"),
    ("memory", "proof"): ("exact_layout", "candidate placements"),
    ("selection", "fast"): ("optimize_scheduled", "trials"),
    ("selection", "proof"): ("exact_selection", "assignments"),
}

#: The instance size (claims, items, claims) up to which the proof rail is expected to close
#: within a default budget. Above it the proof rail is still dispatched -- the budget bounds
#: the work -- but the decision says a budget stop (TMSAO-2) is what to expect.
BOUNDED_SIZE = {"schedule": 64, "memory": 512, "selection": 12}

STOP_REASONS = ("optimal", "budget", "heuristic")


@dataclass(frozen=True)
class DispatchRequest:
    """What a caller asks for: a region kind, its size, the class wanted and the budget granted."""

    kind: str
    size: int
    requested: str = "TMSAO-2"
    budget: int = 200_000

    def __post_init__(self) -> None:
        if self.kind not in REGION_KINDS:
            raise ValueError(f"region kind must be one of {REGION_KINDS}")
        if self.requested not in CERTIFICATE_CLASSES:
            raise ValueError(f"requested class must be one of {CERTIFICATE_CLASSES}")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("instance size must be a non-negative integer")
        if not isinstance(self.budget, int) or isinstance(self.budget, bool) or self.budget < 0:
            raise ValueError("work budget must be a non-negative integer (work units)")


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
    * TMSAO-3 requested: the fast rail, because a measured-best claim needs the workload
      component `W` and the measured-candidate corpus (G13), which no rail here has;
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
        return DispatchDecision(
            request.kind,
            request.size,
            request.requested,
            request.budget,
            "fast",
            fast_solver,
            fast_units,
            "TMSAO-4",
            "a measured-best claim needs the workload component W and the measured-candidate "
            "corpus (G13); neither rail here can grant it, so the fast rail places once",
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
    "solve_memory",
    "solve_schedule",
    "solve_selection",
]
