"""The K_BCIR realization optimizer.

For each claim we enumerate *legal* candidate lowerings (scalar / vector-W /
strided / gather / ux-bucket / tile). We then build a layered realization DAG
(one column of candidate nodes per claim) and run a min-plus shortest path. The
chosen path is

    pi* = argmin_pi  sum_i  scalarize( T_i (X) f_i(pi) ; w(H,Theta,phase) )

where T_i is the candidate's base cost vector, f_i(pi) is the path-dependent
context factor (fusion when a neighbor shares an operand, thermal coupling for
wide SIMD when hot), and the scalarization weights depend on H, Theta, and the
claim's phase. The path structure matters: f_i depends on the *previous* chosen
candidate, so this is a genuine shortest path, not a per-claim argmin.

Correctness rule: RANDOM gathers are never silently "bucketed" into a cheaper
cacheline-local lane -- only claims that *declare* CACHELINE locality get the UX
candidate. The optimizer reduces cost only among provably-legal realizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Claim, Lane, Module, Opcode, StrideClass
from .cost import (
    MEMORY,
    POWER,
    THERMAL,
    CostVector,
    HProfile,
    IDENTITY_FACTOR,
    Theta,
)
from .semiring import dag_shortest_path
from .weights import PERF, Policy, weights


@dataclass(frozen=True)
class Candidate:
    lane: Lane
    width: int          # vector element lanes (1 = scalar)
    name: str           # scalar / vec8 / vec16 / strided / gather / ux_bucket / tile / noop / barrier
    base: CostVector
    reads: tuple[int, ...] = ()


@dataclass
class ChosenStep:
    claim_id: int
    phase_id: int
    candidate: Candidate
    cost: int


@dataclass
class RealizationResult:
    steps: list[ChosenStep] = field(default_factory=list)
    score: int = 0

    def by_claim(self) -> dict[int, Candidate]:
        return {s.claim_id: s.candidate for s in self.steps}


# --- cost derivation ------------------------------------------------------------

def _opclass(op: Opcode) -> int:
    if op in (Opcode.ADD, Opcode.SUB, Opcode.MUL):
        return 1
    if op == Opcode.T_MACC:
        return 2
    return 0


def _streams(claim: Claim) -> int:
    return len(claim.rd) + len(claim.wr)


def _stride_penalty(claim: Claim, h: HProfile) -> int:
    sc = claim.stride_class
    if sc in (StrideClass.UNIT, StrideClass.SCALAR, StrideClass.TILE):
        return 1
    if sc == StrideClass.STRIDED:
        return min(max(claim.stride_k, 1), h.cacheline // h.elem_bytes)
    if sc == StrideClass.CACHELINE:
        return 2
    if sc == StrideClass.RANDOM:
        return h.gather_penalty
    return 1


def _cost(claim: Claim, h: HProfile, width: int, stride_penalty: int, extra_compile: int = 0,
          tier=None) -> CostVector:
    n = max(1, claim.count)
    streams = _streams(claim)
    ceil = (n + width - 1) // width
    compute = ceil * _opclass(claim.opcode)
    # Memory hierarchy: scale traffic by the tier bandwidth factor and latency by the
    # tier latency factor. DRAM (or tier=None) uses Q8 256/256 == x1.0, so RAM
    # resources cost exactly as before (back-compat).
    bw_f = tier.bw_factor if tier is not None else 256
    lat_f = tier.lat_factor if tier is not None else 256
    mem_shared = (n * streams * h.mem_unit * bw_f) >> 8
    access_ops = ceil * streams
    overhead = h.base_overhead * stride_penalty
    memory = mem_shared + ((access_ops * overhead * lat_f) >> 8)
    thermal = width * h.thermal_density + ceil * h.per_op_heat
    power = width * h.power_density + ceil * h.per_op_heat
    return CostVector.of(
        compute=compute, memory=memory, thermal=thermal, power=power, compile=extra_compile
    )


def candidates_for(claim: Claim, h: HProfile, resource=None) -> list[Candidate]:
    op = claim.opcode
    sc = claim.stride_class
    if op in (Opcode.NOP, Opcode.PHASE_ENTER, Opcode.PHASE_LEAVE, Opcode.PROV_NOTE):
        return [Candidate(Lane.H, 1, "noop", CostVector.zero())]
    if op == Opcode.BARRIER:
        return [Candidate(Lane.H, 1, "barrier", CostVector.of(sync=16))]

    # Memory tier + addressing model come from the (primary) resource, if known.
    tier = h.mem.tier_for(resource.domain) if resource is not None else None
    access = resource.access if resource is not None else "flat"

    # Hierarchical Access Memory turns random access from O(gather_penalty) into O(log n).
    gp = h.gather_penalty
    if access == "ham":
        n = max(1, claim.count)
        gp = max(1, (n - 1).bit_length())  # ceil(log2 n)

    cands: list[Candidate] = []
    if sc in (StrideClass.UNIT, StrideClass.SCALAR):
        for w in h.widths():
            name = "scalar" if w == 1 else f"vec{w}"
            lane = claim.lane if w == 1 else Lane.U
            cands.append(Candidate(lane, w, name, _cost(claim, h, w, 1, tier=tier), claim.rd))
    elif sc == StrideClass.STRIDED:
        cands.append(Candidate(Lane.U, 1, "strided",
                               _cost(claim, h, 1, _stride_penalty(claim, h), tier=tier), claim.rd))
        cands.append(Candidate(Lane.GGG, 1, "gather", _cost(claim, h, 1, gp, tier=tier), claim.rd))
    elif sc == StrideClass.CACHELINE:
        uxw = min(8, max(h.widths()))
        cands.append(Candidate(Lane.UX, uxw, "ux_bucket",
                               _cost(claim, h, uxw, 2, extra_compile=claim.count // 4, tier=tier), claim.rd))
        cands.append(Candidate(Lane.GGG, 1, "gather", _cost(claim, h, 1, gp, tier=tier), claim.rd))
    elif sc == StrideClass.RANDOM:
        # Correctness: do not assume locality. Only the declared GGG realization is legal (HAM-aware).
        cands.append(Candidate(Lane.GGG, 1, "gather", _cost(claim, h, 1, gp, tier=tier), claim.rd))
    elif sc == StrideClass.TILE:
        cands.append(Candidate(Lane.T, 16, "tile", _cost(claim, h, 16, 1, tier=tier), claim.rd))

    if not cands:  # defensive fallback
        cands.append(Candidate(claim.lane, 1, "scalar",
                               _cost(claim, h, 1, _stride_penalty(claim, h), tier=tier), claim.rd))
    return cands


# --- context coupling f_i(pi) ---------------------------------------------------

def _context_factor(prev: "Candidate | None", cand: Candidate, theta: Theta) -> tuple[int, ...]:
    f = list(IDENTITY_FACTOR)
    # Fusion / locality: a vector candidate that shares a read operand with its vector
    # predecessor reuses loaded cache lines -> discount memory traffic.
    if prev is not None and cand.width > 1 and prev.width > 1 and set(prev.reads) & set(cand.reads):
        f[MEMORY] = 192  # x0.75
    # Thermal coupling: wide SIMD on a hot machine pays extra heat/current (AVX-512 downclock).
    if theta.thermal >= 60 and cand.width >= 16:
        f[THERMAL] = 320  # x1.25
        f[POWER] = 320
    return tuple(f)


# --- phase ordering -------------------------------------------------------------

def _topo_phases(module: Module):
    pmap = module.phase_map()
    color: dict[int, int] = {}
    order: list[int] = []

    def visit(pid: int) -> None:
        color[pid] = 1
        for d in pmap[pid].deps:
            if d in pmap and color.get(d, 0) == 0:
                visit(d)
        color[pid] = 2
        order.append(pid)

    for p in module.phases:
        if color.get(p.phase_id, 0) == 0:
            visit(p.phase_id)
    return [pmap[pid] for pid in order]


def _flatten(module: Module) -> list[tuple[int, Claim]]:
    flat: list[tuple[int, Claim]] = []
    for ph in _topo_phases(module):
        for cl in ph.claims:
            flat.append((ph.phase_id, cl))
    return flat


# --- the optimizer --------------------------------------------------------------

def optimize(module: Module, h: HProfile, theta: Theta, policy: Policy = PERF) -> RealizationResult:
    flat = _flatten(module)
    if not flat:
        return RealizationResult([], 0)

    # Build a layered DAG. Node 0 = SOURCE; one column per claim; final node = SINK.
    adj: list[list[tuple[int, int]]] = [[]]  # adj[0] = SOURCE
    node_meta: list[tuple[int, Claim, Candidate] | None] = [None]  # (phase_id, claim, candidate) per node
    prev_nodes: list[int] = [0]
    prev_cands: list[Candidate | None] = [None]

    for phase_id, claim in flat:
        w_phase = weights(h, theta, phase_id, policy)
        col_nodes: list[int] = []
        col_cands: list[Candidate] = []

        # Memory traffic is dominated by the streamed/gathered read source; fall back to the write target.
        cost_rid = claim.rd[0] if claim.rd else claim.primary_rid
        resource = module.resource(cost_rid) if cost_rid is not None else None

        for cand in candidates_for(claim, h, resource):
            nid = len(node_meta)
            node_meta.append((phase_id, claim, cand))
            adj.append([])
            for pu, pc in zip(prev_nodes, prev_cands):
                factor = _context_factor(pc, cand, theta)
                cost = cand.base.couple(factor).dot(w_phase)
                adj[pu].append((nid, cost))
            col_nodes.append(nid)
            col_cands.append(cand)
        prev_nodes, prev_cands = col_nodes, col_cands

    sink = len(node_meta)
    node_meta.append(None)
    adj.append([])
    for pu in prev_nodes:
        adj[pu].append((sink, 0))

    dist, pred = dag_shortest_path(len(node_meta), adj, source=0)

    # Reconstruct the chosen path SINK -> SOURCE.
    steps: list[ChosenStep] = []
    node = sink
    while pred[node] != -1:
        p = pred[node]
        meta = node_meta[node]
        if meta is not None:
            phase_id, claim, cand = meta
            steps.append(ChosenStep(claim.id, phase_id, cand, int(dist[node] - dist[p])))
        node = p
    steps.reverse()
    return RealizationResult(steps, int(dist[sink]))
