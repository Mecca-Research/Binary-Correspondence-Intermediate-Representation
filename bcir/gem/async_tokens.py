"""CT2 / Phase 8: the explicit !bcir.token async-dependency model.

The phase DAG gives coarse ordering; async tokens make the *fine-grained*
dependencies explicit and verifiable. Each claim `fork`s (launches asynchronously,
producing a completion token); a claim `await`s the tokens of the earlier claims it
conflicts with (RAW/WAR/WAW) and the ordering fences (`barriered` / `volatile`
claims, which await everything before them and are awaited by everything after --
`concurrency.hazard_predecessors`, the one hazard DAG of the GEM rails). Independent
claims await nothing -- they run fully concurrently. This is the SSA-token form of the
CT2 concurrent waves and the oracle counterpart of `bcir.async.fork` /
`bcir.async.await`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Module
from .concurrency import _topo_phase_ids, hazard_predecessors


@dataclass
class AsyncPlan:
    forks: list[int] = field(default_factory=list)  # claim ids launched async
    awaits: dict[int, list[int]] = field(default_factory=dict)  # claim id -> awaited claim ids

    def is_independent(self, claim_id: int) -> bool:
        return not self.awaits.get(claim_id)


def async_plan(module: Module) -> AsyncPlan:
    """Build the async fork/await plan: each claim forks; it awaits its earlier hazards
    (data conflicts and fences, `concurrency.hazard_predecessors`)."""
    pmap = module.phase_map()
    flat = []
    for pid in _topo_phase_ids(module):
        for c in sorted(pmap[pid].claims, key=lambda c: c.id):
            flat.append(c)

    plan = AsyncPlan(forks=[c.id for c in flat])
    plan.awaits = hazard_predecessors(flat)
    return plan
