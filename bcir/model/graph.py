"""BCIR-0..2 data model: registry-first resources, claims, phases, modules.

This is the semantic claim graph G. Memory is a registry-indexed resource system
(no raw pointers): every claim references resources by RID. Ordering is a phase
DAG plus explicit dependencies, never accidental textual order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .lanes import Domain, Lane, StrideClass
from .opcodes import Opcode


@dataclass(frozen=True)
class Resource:
    """A registry-governed resource (LangRef Sec. 4) — addressed by RID, not pointer."""

    rid: int
    domain: Domain = Domain.RAM
    elem_bytes: int = 4
    shape: tuple[int, ...] = ()
    layout: str = "soa"
    align: int = 64
    access: str = "flat"      # "flat" (O(1)) or "ham" (O(log n) hierarchical access)
    priority: int = 0         # CXL semantic hotness (promotion/eviction)
    map_gen: int = 0
    data_gen: int = 0
    name: str = ""

    @property
    def count(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class Claim:
    """The primitive object of BCIR (LangRef Sec. 5): op + resources + contract."""

    id: int
    opcode: Opcode
    lane: Lane = Lane.U
    stride_class: StrideClass = StrideClass.UNIT
    count: int = 1
    stride_k: int = 1
    rd: tuple[int, ...] = ()         # read RIDs
    wr: tuple[int, ...] = ()         # write RIDs
    imm: tuple[int, ...] = ()
    hazard: str = "unique"           # unique | atomic | barriered
    domain: Domain = Domain.RAM
    verify: str = "bounds"           # none | bounds | exact | hash
    bounds: str = "strict"           # strict | masked | assumed_safe
    op: str = ""                     # semantic op string, e.g. "vector.add"
    offset: int = 0
    cost_class: str = "bandwidth"
    primary_rid: Optional[int] = None
    precision: str = ""              # "" (naive) | "compensated" (residual-carry MAC)
    tolerance_ulp: int = 0           # accuracy contract: 0 = none; >0 = max Q8-ULP error (R17)
    dynamic: bool = False            # True: `count` is a static UPPER BOUND (dynamic shape);
                                     # the plan is valid + worst-case-priced for any actual <= count

    def io_rids(self) -> tuple[int, ...]:
        return tuple(self.rd) + tuple(self.wr)


@dataclass
class Phase:
    """A node in the phase DAG (LangRef Sec. 6)."""

    phase_id: int
    deps: tuple[int, ...] = ()
    claims: list[Claim] = field(default_factory=list)


@dataclass
class Module:
    """A registry-governed execution universe (LangRef Sec. 3)."""

    name: str = "module"
    cacheline: int = 64
    align: int = 64
    target: str = "registry-first"
    resources: dict[int, Resource] = field(default_factory=dict)
    phases: list[Phase] = field(default_factory=list)

    def add_resource(self, resource: Resource) -> Resource:
        if resource.rid in self.resources:
            raise ValueError(f"duplicate RID {resource.rid} (LangRef R1)")
        self.resources[resource.rid] = resource
        return resource

    def add_phase(self, phase: Phase) -> Phase:
        self.phases.append(phase)
        return phase

    def resource(self, rid: Optional[int]) -> Optional[Resource]:
        if rid is None:
            return None
        return self.resources.get(rid)

    def phase_map(self) -> dict[int, Phase]:
        return {p.phase_id: p for p in self.phases}
