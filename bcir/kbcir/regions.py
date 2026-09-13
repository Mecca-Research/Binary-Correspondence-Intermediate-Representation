"""Typed regions over the claim DAG (G6 / S2-D): a strong local model with a conservative
expansion back to claims.

The 2026-08-12 report (section 8): the answer to what a flat claim DAG cannot express -- affine
iteration spaces and dependence distances, fixed-rate streams, timed events, index maps -- is
not to abandon the DAG but to give its nodes typed regions with stronger local mathematics,
and an opaque claim DAG as the universal conservative fallback. The roadmap's G6 asks every
region to supply a verifier, a conservative claim expansion, a cost/lower-bound interface and
refusal conditions, and gates the layer on one exact property: every region expands
conservatively to claims -- differential, per region.

This module lands the carrier and the first two kinds:

  * **affine** -- a maximal run of consecutive claims of one phase whose accesses are 1-D
    affine maps with static trip counts: element `i` of a unit or strided claim touches
    `offset + i * stride_k` of each resource it names, `count` is static, the hazard contract
    is `unique`, the claim is not volatile, and every map stays inside its resource. The
    local model is the family of access maps and the dependence distances between the
    region's claims (the offset difference on a shared resource). Refused -- and left to the
    opaque region -- are dynamic trip counts, `volatile`, ordering fences, atomics, gathers
    and random strides, cacheline-indexed and tile claims (those are their own region kinds,
    not landed here), and any map that leaves its resource.
  * **opaque** -- any claims at all, the universal fallback: no local model, the claims
    themselves as the expansion.

Every region's `expand()` returns exactly the claims it was recognized from, in their declared
order -- the expansion is the identity on the carrier, which is what makes it conservative:
`expand(region_graph(module))` is the module claim for claim, and the plan `optimize` selects
over the expansion is the plan it selects over the module. A region is never declared by hand;
it is recognized from the phases and re-verified, so a region graph cannot claim structure the
claims do not have.

The cost interface is a LOWER BOUND on what the region's claims can cost under the planner's
own edge predicate (`realize.edge_cost`): for every claim, the cheapest of its (deforested)
candidates under the most favourable context the coupling can give it -- the memory discount
only where a vector predecessor sharing a read operand is possible (the affine region knows
its access maps and its textual predecessor, so a claim that shares no read with its
predecessor is bound without the discount), the thermal penalty where the candidate's width
and Theta impose it. Summed over the region it is a floor no realization of those claims can
go below, and summed over the graph a floor on the selection's serial score -- the structural
floor a selection certificate reports beside the exact optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Claim, Lane, Module, StrideClass
from .cost import IDENTITY_FACTOR, MEMORY, POWER, THERMAL, Theta
from .weights import PERF, Policy, weights

REGION_KINDS = ("affine", "opaque")

#: Why a run of claims is not an affine region (the refusal conditions, named).
REFUSALS = (
    "dynamic-trip-count",
    "volatile",
    "fence",
    "atomic",
    "sparse",
    "cacheline-indexed",
    "tile",
    "lane",
    "stride",
    "extent",
    "count",
)


@dataclass(frozen=True)
class AccessMap:
    """One claim's 1-D affine access to one resource: element i -> offset + i * stride."""

    claim_id: int
    rid: int
    mode: str  # "read" | "write"
    offset: int
    stride: int
    count: int

    @property
    def last(self) -> int:
        return self.offset + (self.count - 1) * self.stride


@dataclass(frozen=True)
class Dependence:
    """A dependence between two claims of one region on a shared resource, with the offset
    distance in elements (consumer minus producer)."""

    producer: int
    consumer: int
    rid: int
    kind: str  # "raw" | "war" | "waw"
    distance: int


@dataclass(frozen=True)
class Region:
    """A typed region: its kind, its phase, the claims it expands to (in order), and the
    local model for the kinds that have one."""

    kind: str
    phase_id: int
    claim_ids: tuple[int, ...]
    maps: tuple[AccessMap, ...] = ()
    dependences: tuple[Dependence, ...] = ()
    refusal: str = ""  # for an opaque region: why the affine model refused its claims

    def __post_init__(self) -> None:
        if self.kind not in REGION_KINDS:
            raise ValueError(f"region kind must be one of {REGION_KINDS}")
        if not self.claim_ids:
            raise ValueError("a region holds at least one claim")

    @property
    def size(self) -> int:
        return len(self.claim_ids)


@dataclass
class RegionGraph:
    """The regions of a module, phase by phase in topological order, claims in declared
    order -- a partition of the module's claims."""

    regions: list[Region] = field(default_factory=list)

    def by_claim(self) -> dict[int, Region]:
        return {cid: region for region in self.regions for cid in region.claim_ids}

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for region in self.regions:
            out[region.kind] = out.get(region.kind, 0) + 1
        return out


# --- recognition and the verifier ------------------------------------------------------------


def affine_refusal(claim: Claim, module: Module) -> str:
    """Why `claim` cannot belong to an affine region; '' when it can."""
    if claim.dynamic:
        return "dynamic-trip-count"
    if claim.volatile:
        return "volatile"
    if claim.hazard == "barriered":
        return "fence"
    if claim.hazard == "atomic" or claim.lane == Lane.A:
        return "atomic"
    if claim.lane == Lane.GGG or claim.stride_class == StrideClass.RANDOM:
        return "sparse"
    if claim.stride_class == StrideClass.CACHELINE or claim.lane == Lane.UX:
        return "cacheline-indexed"
    if claim.stride_class == StrideClass.TILE or claim.lane == Lane.T:
        return "tile"
    if claim.lane != Lane.U:
        return "lane"
    if claim.stride_class not in (StrideClass.SCALAR, StrideClass.UNIT, StrideClass.STRIDED):
        return "stride"
    if claim.count < 1 or claim.stride_k < 1 or claim.offset < 0:
        return "count"
    for rid in (*claim.rd, *claim.wr):
        resource = module.resource(rid)
        if resource is None:
            return "extent"
        last = claim.offset + (claim.count - 1) * claim.stride_k
        if resource.count and last >= resource.count:
            return "extent"
    return ""


def _maps_of(claim: Claim) -> tuple[AccessMap, ...]:
    stride = 1 if claim.stride_class in (StrideClass.SCALAR, StrideClass.UNIT) else claim.stride_k
    return tuple(
        AccessMap(claim.id, rid, mode, claim.offset, stride, claim.count)
        for mode, rids in (("read", claim.rd), ("write", claim.wr))
        for rid in rids
    )


def _dependences(claims: list[Claim]) -> tuple[Dependence, ...]:
    out: list[Dependence] = []
    for i, later in enumerate(claims):
        for earlier in claims[:i]:
            for rid in earlier.wr:
                if rid in later.rd:
                    out.append(
                        Dependence(earlier.id, later.id, rid, "raw", later.offset - earlier.offset)
                    )
                if rid in later.wr:
                    out.append(
                        Dependence(earlier.id, later.id, rid, "waw", later.offset - earlier.offset)
                    )
            for rid in earlier.rd:
                if rid in later.wr and rid not in earlier.wr:
                    out.append(
                        Dependence(earlier.id, later.id, rid, "war", later.offset - earlier.offset)
                    )
    return tuple(out)


def _admit(
    graph: RegionGraph, module: Module, pid: int, run: list[Claim], kind: str, refusal: str
) -> None:
    """Close one maximal run as a region, verify it, and append it to the graph."""
    if not run:
        return
    if kind == "affine":
        region = Region(
            "affine",
            pid,
            tuple(c.id for c in run),
            tuple(m for c in run for m in _maps_of(c)),
            _dependences(list(run)),
        )
    else:
        region = Region("opaque", pid, tuple(c.id for c in run), refusal=refusal)
    problems = verify_region(region, module)
    if problems:  # pragma: no cover - recognition and verification share one rule
        raise ValueError(f"recognized region failed its verifier: {problems}")
    graph.regions.append(region)
    run.clear()


def region_graph(module: Module) -> RegionGraph:
    """Partition every phase (topological order) into maximal affine runs and opaque runs,
    claims in declared order. Recognized, then verified: a region the verifier refuses is
    never returned."""
    from ..gem.concurrency import _topo_phase_ids

    graph = RegionGraph()
    pmap = module.phase_map()
    for pid in _topo_phase_ids(module):
        run: list[Claim] = []
        run_kind = "affine"
        run_refusal = ""
        for claim in pmap[pid].claims:
            refusal = affine_refusal(claim, module)
            kind = "opaque" if refusal else "affine"
            if run and kind != run_kind:
                _admit(graph, module, pid, run, run_kind, run_refusal)
            if not run:
                run_kind, run_refusal = kind, refusal
            run.append(claim)
        _admit(graph, module, pid, run, run_kind, run_refusal)
    return graph


def verify_region(region: Region, module: Module) -> list[str]:
    """The region's laws against the module it claims to describe: the claims exist in its
    phase in declared order, an affine region's claims all pass the affine conditions and its
    maps and dependences are exactly the claims', an opaque region names its refusal."""
    problems: list[str] = []
    phase = module.phase_map().get(region.phase_id)
    if phase is None:
        return [f"phase {region.phase_id} is not in the module"]
    order = [c.id for c in phase.claims]
    positions = [order.index(cid) if cid in order else -1 for cid in region.claim_ids]
    if -1 in positions:
        return ["a region claim is not in its phase"]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        problems.append("region claims are not consecutive in declared order")
    claims = [phase.claims[p] for p in positions if p >= 0]
    if region.kind == "affine":
        for claim in claims:
            why = affine_refusal(claim, module)
            if why:
                problems.append(f"claim {claim.id} is not affine: {why}")
        if region.maps != tuple(m for c in claims for m in _maps_of(c)):
            problems.append("access maps are not the claims' maps")
        if region.dependences != _dependences(claims):
            problems.append("dependences are not the claims' dependences")
        if region.refusal:
            problems.append("an affine region carries a refusal")
    else:
        if region.maps or region.dependences:
            problems.append("an opaque region carries a local model")
        if region.refusal and region.refusal not in REFUSALS:
            problems.append(f"unknown refusal {region.refusal!r}")
    return problems


def expand(graph: RegionGraph, module: Module) -> list[tuple[int, Claim]]:
    """The conservative expansion: (phase id, claim) for every region in graph order -- the
    module's own claims, verified and in declared order."""
    pmap = module.phase_map()
    out: list[tuple[int, Claim]] = []
    for region in graph.regions:
        problems = verify_region(region, module)
        if problems:
            raise ValueError(f"region does not describe the module: {problems}")
        by_id = {c.id: c for c in pmap[region.phase_id].claims}
        out.extend((region.phase_id, by_id[cid]) for cid in region.claim_ids)
    return out


# --- the cost interface: a floor per region ----------------------------------------------------


def _best_factor(
    prev: Claim | None, cand, theta: Theta, prev_vector_possible: bool
) -> tuple[int, ...]:
    """The most favourable context factor `realize._context_factor` can give `cand`: the
    memory discount only if a vector predecessor sharing a read is possible, the thermal
    penalty exactly as the model imposes it (it depends on Theta and the candidate alone)."""
    factor = list(IDENTITY_FACTOR)
    if (
        prev is not None
        and prev_vector_possible
        and cand.width > 1
        and set(prev.rd) & set(cand.reads)
    ):
        factor[MEMORY] = 192
    if theta.thermal >= 60 and cand.width >= 16:
        factor[THERMAL] = 320
        factor[POWER] = 320
    return tuple(factor)


def claim_floor(claim: Claim, prev: Claim | None, cand_map: dict, h, theta: Theta, w_phase) -> int:
    """A lower bound on `claim`'s step cost under any predecessor candidate."""
    prev_vector_possible = prev is not None and any(c.width > 1 for c in cand_map.get(prev.id, ()))
    return min(
        cand.base.couple(_best_factor(prev, cand, theta, prev_vector_possible)).dot(w_phase)
        for cand in cand_map[claim.id]
    )


def region_floor(
    region: Region, module: Module, h, theta: Theta, policy: Policy = PERF, cand_map=None
) -> int:
    """The region's floor: the sum of its claims' floors, each under its textual predecessor
    (the previous claim in flatten order, inside or before the region)."""
    from .realize import _flatten, fused_candidates

    if cand_map is None:
        cand_map = fused_candidates(module, h)
    flat = _flatten(module)
    index = {claim.id: i for i, (_pid, claim) in enumerate(flat)}
    w_phase = weights(h, theta, region.phase_id, policy)
    total = 0
    for cid in region.claim_ids:
        i = index[cid]
        prev = flat[i - 1][1] if i else None
        total += claim_floor(flat[i][1], prev, cand_map, h, theta, w_phase)
    return total


def module_floor(module: Module, h, theta: Theta, policy: Policy = PERF) -> int:
    """The graph's floor: no plan `optimize` can select scores below it."""
    from .realize import fused_candidates

    cand_map = fused_candidates(module, h)
    return sum(
        region_floor(region, module, h, theta, policy, cand_map)
        for region in region_graph(module).regions
    )


__all__ = [
    "REFUSALS",
    "REGION_KINDS",
    "AccessMap",
    "Dependence",
    "Region",
    "RegionGraph",
    "affine_refusal",
    "claim_floor",
    "expand",
    "module_floor",
    "region_floor",
    "region_graph",
    "verify_region",
]
