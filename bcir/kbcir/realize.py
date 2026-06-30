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

from dataclasses import dataclass, field, replace

from ..model import Claim, Lane, Module, Opcode, StrideClass
from .cost import (
    COMPUTE,
    MEMORY,
    N,
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
    writes: tuple[int, ...] = ()    # the claim's write RIDs (for producer->consumer fusion)


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
    # The deforested candidate map optimize() already built (claim id -> candidates).
    # Carried so downstream consumers (plan_view / to_mlir) reuse it instead of
    # recomputing fused_candidates(). Excluded from equality/repr -- it is derived
    # state, not part of the result's identity.
    cand_map: dict[int, list[Candidate]] | None = field(
        default=None, compare=False, repr=False)

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


def _verify_cost(claim: Claim) -> int:
    """The cost of discharging the claim's verify contract -- the producer for the 12th
    cost axis (VERIFICATION). `none`/`bounds` are modeled as free: a bounds check is
    fused into / hoisted above the access the memory axis already prices. The expensive
    contracts carry a real, size-proportional cost the optimizer can trade against the
    rest of the vector (e.g. an RCSP cap on verification, or a policy that weights it):
    `exact` recomputes + compares every element, `hash` digests every output element --
    both O(n). Width-independent (the contract is a property of the claim, not the lane),
    so it shifts every realization equally and never perturbs the per-claim selection."""
    n = max(1, claim.count)
    return n if claim.verify in ("exact", "hash") else 0


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
        compute=compute, memory=memory, thermal=thermal, power=power, compile=extra_compile,
        verification=_verify_cost(claim),
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

    # Reducible-permutation gather (`reduce.gather`): + is commutative and the
    # index is a permutation, so the random gather (O(gather_penalty)) has a
    # semantically identical *blocked* sequential realization (O(1) overhead). The
    # cost model offers both; the blocked one wins, avoiding gather_penalty.
    if claim.op == "reduce.gather":
        return [
            Candidate(Lane.U, 1, "blocked", _cost(claim, h, 1, 1, tier=tier), claim.rd, claim.wr),
            Candidate(Lane.GGG, 1, "gather", _cost(claim, h, 1, gp, tier=tier), claim.rd, claim.wr),
        ]


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
        # The tile lane is the widest the hardware can issue, capped at 16 (AVX-512 f32):
        # 16 on AVX-512/SVE/RVV/PTX, but 4 on NEON, 8 on AVX2 -- never an unrealizable width.
        tw = min(16, h.lane_widths[-1])
        cands.append(Candidate(Lane.T, tw, "tile", _cost(claim, h, tw, 1, tier=tier), claim.rd))

    if not cands:  # defensive fallback
        cands.append(Candidate(claim.lane, 1, "scalar",
                               _cost(claim, h, 1, _stride_penalty(claim, h), tier=tier), claim.rd))
    wr = tuple(claim.wr)
    return [replace(c, writes=wr) for c in cands]


# --- context coupling f_i(pi) ---------------------------------------------------

# Deforestation discount (x0.75 memory): a fused producer->consumer pass elides the
# intermediate operand's round-trip. Baked into the consumer's base cost in `optimize`
# (dependency-based) so it prices identically in the plan score and the GEM makespan.
_DEFOREST_FACTOR = tuple(192 if i == MEMORY else 256 for i in range(len(IDENTITY_FACTOR)))


def _context_factor(prev: "Candidate | None", cand: Candidate, theta: Theta) -> tuple[int, ...]:
    f = list(IDENTITY_FACTOR)
    # Fusion / locality: a vector candidate that shares a read operand with its vector
    # predecessor reuses loaded cache lines -> discount memory traffic. (Producer->
    # consumer "deforestation" fusion is dependency-based, not path-based, so it is
    # baked into the consumer's base cost in `optimize`, not here.)
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


# --- fusion-aware candidate generation (shared by the tropical + RCSP rails) -----

def _cse_factor(claim: Claim) -> tuple[int, ...]:
    """The copy-cost factor for a claim that is a common subexpression of an earlier
    one: the value is already computed, so there is no recompute (compute zeroed) and
    only the result is copied to this claim's output instead of re-reading every
    operand (memory scaled from `len(rd)+len(wr)` streams down to the `1 + len(wr)` a
    copy needs). Conservative on thermal/power (left as-is)."""
    full = len(claim.rd) + len(claim.wr)
    copy = 1 + len(claim.wr)
    mem_q8 = (copy * 256) // max(1, full)
    return tuple(0 if i == COMPUTE else (mem_q8 if i == MEMORY else 256) for i in range(N))


def fused_candidates(module: Module, h: HProfile) -> dict[int, list[Candidate]]:
    """Per-claim candidate lists with the **redundancy discounts** baked in, computed
    from intra-phase data flow (not path adjacency) so all five rails -- tropical,
    RCSP, soft, accel, scheduled overlap -- price them identically and the plan score,
    makespan, and serial bound stay consistent (makespan <= serial). Two discounts,
    applied at most one per claim (CSE wins, being the larger):

      * **CSE / duplicate elimination**: a claim whose (op, operand value-numbers)
        matches an earlier same-phase claim recomputes a value already in hand -- it
        becomes a copy (no recompute, no operand reload). Value numbering (a write
        bumps an operand's version) makes the match sound: a rewrite between the two
        invalidates it. The egraph proves the same liked-pair CSE; this prices it.
      * **producer->consumer deforestation**: a claim reading an operand a prior
        same-phase claim produced fuses with it, so the intermediate never round-trips
        to memory (a memory discount).

    A barrier between phases materializes intermediates, so both credits are
    intra-phase only; single-claim programs (e.g. vector_add) get neither (a no-op,
    so the pinned scores are preserved)."""
    out: dict[int, list[Candidate]] = {}
    produced: dict[int, set[int]] = {}            # phase -> rids written so far (deforestation)
    version: dict[int, dict[int, int]] = {}       # phase -> {rid: write count} (value numbering)
    seen: dict[int, dict[tuple, int]] = {}        # phase -> {compute signature: first claim}
    barr_prod: dict[int, set[int]] = {}           # phase -> rids written by a barriered producer (ASM3b)
    for phase_id, claim in _flatten(module):
        cost_rid = claim.rd[0] if claim.rd else claim.primary_rid
        resource = module.resource(cost_rid) if cost_rid is not None else None
        pset = produced.setdefault(phase_id, set())
        ver = version.setdefault(phase_id, {})
        seenmap = seen.setdefault(phase_id, {})
        bset = barr_prod.setdefault(phase_id, set())
        # value-numbered compute signature: same op + same operands AT THE SAME VERSIONS.
        sig = (claim.op or claim.opcode, tuple((r, ver.get(r, 0)) for r in claim.rd))

        cands = candidates_for(claim, h, resource)
        shared = pset & set(claim.rd)
        if claim.rd and sig in seenmap:                  # CSE: identical value already computed
            cands = [replace(c, base=c.base.couple(_cse_factor(claim))) for c in cands]
        elif shared:                                     # producer->consumer deforestation
            # ASM3b: a barriered claim is a first-class ordering edge -- no fusion across it. Skip the
            # deforestation discount when the consumer is barriered OR a shared operand was produced by
            # a barriered producer (the fence forces the intermediate to materialize, no round-trip elision).
            if claim.hazard != "barriered" and not (shared & bset):
                cands = [replace(c, base=c.base.couple(_DEFOREST_FACTOR)) for c in cands]

        if claim.rd:
            seenmap.setdefault(sig, claim.id)            # first occurrence pays full
        for r in claim.wr:                               # a write creates a new operand version
            ver[r] = ver.get(r, 0) + 1
        pset |= set(claim.wr)
        if claim.hazard == "barriered":                  # ASM3b: a barriered producer fences its writes
            bset |= set(claim.wr)
        out[claim.id] = cands
    return out


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

    cand_map = fused_candidates(module, h)    # candidates with deforestation baked in
    for phase_id, claim in flat:
        w_phase = weights(h, theta, phase_id, policy)
        col_nodes: list[int] = []
        col_cands: list[Candidate] = []

        for cand in cand_map[claim.id]:
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
    return RealizationResult(steps, int(dist[sink]), cand_map=cand_map)
