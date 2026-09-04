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

from ..model import Module, topological_phase_ids


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
    for pid in _topo_phases(module):
        phase = pmap[pid]
        claims = list(phase.claims)
        if deterministic:
            claims.sort(key=lambda c: c.id)
        stat = PhaseStat(phase_id=pid, scheduled=len(claims))
        result.phase_order.append(pid)
        for claim in claims:
            result.order.append(claim.id)
            if kernels and claim.id in kernels:
                kernels[claim.id]()
            stat.executed += 1
            result.executed += 1
        result.phases.append(stat)
    return result
