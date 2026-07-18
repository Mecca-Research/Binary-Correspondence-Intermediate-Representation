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


@dataclass(frozen=True)
class Lifetime:
    """OPTIONAL pointer-lifetime annotation on a claim (§5.12, the naked-pointer safety track). A claim may
    mark itself as ALLOCATING the resources it writes (`event='alloc'`) or FREEING the resources it reads
    (`event='free'`, the `free(p)` shape); any other claim is an implicit `use`. `epoch` distinguishes a
    re-allocation from the original. Every field defaults to the inert value, so a claim with no lifetime
    annotation (`Claim.lifetime is None` -- the entire scalar / C-frontend subset today) places no R21
    constraint: the use-after-free / double-free law is a complete no-op until a frontend opts in by
    annotating its malloc/free calls."""

    event: str = "use"                   # "use" | "alloc" | "free"
    epoch: int = 0                       # the allocation generation (a re-alloc after free is a new epoch)


@dataclass(frozen=True)
class Timing:
    """OPTIONAL RTL / synchronous-timing metadata on a claim (§5.11). Every field defaults to the
    'unspecified' value, so a claim with no timing concern carries none (`Claim.timing is None`) and the
    timing laws (R19/R20) are vacuous over it -- the whole scalar / C-frontend subset is unaffected (the
    non-disturbance invariant). The fields are architecture-AGNOSTIC; the channel / lowering layer
    interprets them per target (CPU microarchitectural cost vs FPGA/ASIC register-transfer)."""

    clock_domain: str = ""               # the clock net / frequency domain ("" = the implicit default)
    latency_cycles: int = 0              # expected stage latency in cycles (0 = unspecified)
    min_throughput_q16: int = 0          # items per cycle, Q16 fixed point (0 = unspecified)
    critical_path: bool = False          # is this claim on the longest timing path?
    power_domain: str = ""               # the power / voltage domain ("" = the implicit default)
    sync_type: str = ""                  # "" | "synchronous" | "asynchronous" | "mixed"
    clock_frequency_mhz: int = 0         # target clock (0 = unspecified)
    setup_hold_margin: int = 0           # extra setup/hold safety margin in cycles


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
    quantized_bits: int = 0          # 0 = dense (no-op); >0 = inputs are `_BitInt(N)` per-group quantized
                                     # lanes (A1), so R17 adds the Q8<->f32<->Q8 bridge step to the bound
    dynamic: bool = False            # True: `count` is a static UPPER BOUND (dynamic shape);
                                     # the plan is valid + worst-case-priced for any actual <= count
    timing: Optional["Timing"] = None  # OPTIONAL RTL/synchronous-timing metadata (§5.11); None = the
                                     # vacuous default (no R19/R20 constraint -- the whole scalar subset)
    lifetime: Optional["Lifetime"] = None  # OPTIONAL pointer-lifetime annotation (§5.12); None = the vacuous
                                     # default (no R21 use-after-free / double-free constraint)
    callee_sig: str = ""             # OPTIONAL declared indirect-callee signature (§5.14 Phase 2):
                                     # "ret(param, ...)" on a c.call.indirect / c.call.imember dispatch
                                     # claim, so R18 + the effect/commutation analysis see the callee
                                     # TYPE instead of a fully opaque edge. "" = unset (vacuous;
                                     # digest-excluded from R13).
    bounds_provenance: str = ""      # OPTIONAL extent-provenance signal (§5.14 Phase 2): WHY the bounds
                                     # contract is what it is -- "declared_extent" (local/static array of
                                     # known shape) | "recovered_count" (malloc/calloc stable count) |
                                     # "snapshot_extent" (pure count expression snapshotted at the alloc) |
                                     # "mmio_register" | "string_literal" | "unknown_extent". "" = unset
                                     # (the vacuous default; digest-excluded from R13).
    volatile: bool = False           # OPTIONAL volatile qualifier (§5.14 Phase 2): a volatile-qualified
                                     # access (MMIO) the ordering law sees STRUCTURALLY -- R5 requires an
                                     # ordered hazard, and the optimizer fences it like `barriered` (never
                                     # reordered / fused / bundled / elided). Digest-excluded (R13's
                                     # hash_module folds a fixed field list), so the False default is
                                     # non-disturbing over the whole existing corpus.

    def io_rids(self) -> tuple[int, ...]:
        return tuple(self.rd) + tuple(self.wr)


@dataclass
class Phase:
    """A node in the phase DAG (LangRef Sec. 6)."""

    phase_id: int
    deps: tuple[int, ...] = ()
    claims: list[Claim] = field(default_factory=list)
    event: str = ""                  # OPTIONAL event source (driver roadmap Part VII A1): "" = an
                                     # ordinary program-ordered phase (the entire existing corpus);
                                     # non-"" names the interrupt/event source that TRIGGERS this
                                     # phase -- asynchronous entry, checked by
                                     # `kbcir.events.check_event_phases` (EV1-EV3). Digest-excluded
                                     # (R13's hash_module folds a fixed field list), so the default
                                     # is non-disturbing over every existing module.


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


def topological_phase_ids(module: Module) -> list[int]:
    """Return phase ids in the repository's canonical dependency-first order.

    This is the iterative equivalent of the historical depth-first traversal used by
    the verifier, K_BCIR, and GEM.  Keeping explicit stack frames preserves dependency
    and module insertion order exactly while avoiding Python's recursion limit on a
    valid deep phase DAG.  Missing dependencies are ignored here, as before; R4 reports
    them before an artifact is eligible for execution.
    """
    pmap = module.phase_map()
    color: dict[int, int] = {}
    order: list[int] = []
    for phase in module.phases:
        root = phase.phase_id
        if color.get(root, 0) != 0:
            continue
        color[root] = 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            pid, next_dep = stack[-1]
            deps = pmap[pid].deps
            while next_dep < len(deps):
                dep = deps[next_dep]
                next_dep += 1
                stack[-1] = (pid, next_dep)
                if dep in pmap and color.get(dep, 0) == 0:
                    color[dep] = 1
                    stack.append((dep, 0))
                    break
            else:
                stack.pop()
                if color.get(pid) != 2:
                    color[pid] = 2
                    order.append(pid)
    return order


def phase_graph_has_cycle(module: Module) -> bool:
    """Detect a phase-DAG cycle without recursive Python calls."""
    pmap = module.phase_map()
    color: dict[int, int] = {}
    for phase in module.phases:
        root = phase.phase_id
        if color.get(root, 0) != 0:
            continue
        color[root] = 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            pid, next_dep = stack[-1]
            deps = pmap[pid].deps
            while next_dep < len(deps):
                dep = deps[next_dep]
                next_dep += 1
                stack[-1] = (pid, next_dep)
                if dep not in pmap:
                    continue
                state = color.get(dep, 0)
                if state == 1:
                    return True
                if state == 0:
                    color[dep] = 1
                    stack.append((dep, 0))
                    break
            else:
                stack.pop()
                color[pid] = 2
    return False
