"""Deterministic phase-sliced execution over a BCIR phase DAG (GEM runtime).

This ports the unique capability of the legacy C++ `ir/runtime/` (`gem-runtime`):
a deterministic, phase-ordered executor with per-phase telemetry. Claims run in
topological phase order; within a phase, deterministic mode dispatches by ascending
claim id (matching the C++ engine's `deterministicOrdering`). Optional per-claim
kernels are invoked so the same scheduler can drive real work (e.g. the lowered
kernels from `bcir.lower`).

Part of the staged `ir/` -> `bcir/`+`mlir/` fold (docs/BCIR_Repo_Structure.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from operator import attrgetter

from ..model import Module, topological_phase_ids

_claim_id = attrgetter("id")  # the deterministic dispatch order: ascending claim id


@dataclass
class PhaseStat:
    phase_id: int
    scheduled: int = 0
    executed: int = 0


@dataclass
class ExecResult:
    order: list[int] = field(default_factory=list)  # claim ids, in dispatch order
    phase_order: list[int] = field(default_factory=list)  # phase ids, in execution order
    phases: list[PhaseStat] = field(default_factory=list)
    executed: int = 0


def _topo_phases(module: Module) -> list[int]:
    """Topologically order phase ids honoring `Phase.deps` (R4 assumed: acyclic)."""
    return topological_phase_ids(module)


def execute(
    module: Module,
    kernels: Optional[dict[int, Callable[[], None]]] = None,
    deterministic: bool = True,
) -> ExecResult:
    """Run a module's claims in deterministic phase order, collecting telemetry.

    `kernels` optionally maps claim id -> a zero-arg callable invoked when the
    claim is dispatched. With `deterministic=True`, claims within a phase run by
    ascending id, so the dispatch order is reproducible run to run.
    """
    pmap = module.phase_map()
    result = ExecResult()
    order, phase_order, phases = result.order, result.phase_order, result.phases
    executed = 0
    for pid in _topo_phases(module):
        claims = pmap[pid].claims
        if deterministic:
            claims = sorted(claims, key=_claim_id)
        ids = [claim.id for claim in claims]
        count = len(ids)
        stat = PhaseStat(pid, count, count)  # every dispatched claim runs (a kernel that
        phase_order.append(pid)  # raises takes the result with it)
        if kernels:
            for claim_id in ids:
                order.append(claim_id)
                if claim_id in kernels:
                    kernels[claim_id]()
        else:
            order += ids
        executed += count
        phases.append(stat)
    result.executed = executed
    return result
