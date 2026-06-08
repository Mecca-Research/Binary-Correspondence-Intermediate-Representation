"""CT2 / Phase 8: the explicit !bcir.token async-dependency model.

The phase DAG gives coarse ordering; async tokens make the *fine-grained*
dependencies explicit and verifiable. Each claim `fork`s (launches asynchronously,
producing a completion token); a claim `await`s the tokens of the earlier claims it
conflicts with (RAW/WAR/WAW). Independent claims await nothing -- they run fully
concurrently. This is the SSA-token form of the CT2 concurrent waves and the oracle
counterpart of `bcir.async.fork` / `bcir.async.await`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Module
from .concurrency import _conflict, _topo_phase_ids


@dataclass
class AsyncPlan:
    forks: list[int] = field(default_factory=list)          # claim ids launched async
    awaits: dict[int, list[int]] = field(default_factory=dict)  # claim id -> awaited claim ids

    def is_independent(self, claim_id: int) -> bool:
        return not self.awaits.get(claim_id)


def async_plan(module: Module) -> AsyncPlan:
    """Build the async fork/await plan: each claim forks; it awaits its earlier conflicts."""
    pmap = module.phase_map()
    flat = []
    for pid in _topo_phase_ids(module):
        for c in sorted(pmap[pid].claims, key=lambda c: c.id):
            flat.append(c)

    plan = AsyncPlan(forks=[c.id for c in flat])
    for i, ci in enumerate(flat):
        plan.awaits[ci.id] = [cj.id for cj in flat[:i] if _conflict(cj, ci)]
    return plan
