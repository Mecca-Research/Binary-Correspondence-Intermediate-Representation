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

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from ..model import (
    ATOMIC_OPCODES,
    ISOLATED_DOMAINS,
    Claim,
    Lane,
    Module,
    Opcode,
    StrideClass,
    topological_phase_ids,
)
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
    width: int  # vector element lanes (1 = scalar)
    name: str  # scalar / vec8 / vec16 / strided / gather / ux_bucket / tile / noop / barrier
    base: CostVector
    reads: tuple[int, ...] = ()
    writes: tuple[int, ...] = ()  # the claim's write RIDs (for producer->consumer fusion)


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
    # recomputing fused_candidates(). Since G17 it is a read-only view of the planner's
    # compact offer that builds a claim's `Candidate` list on first access (`OfferMap`).
    # Excluded from equality/repr -- it is derived state, not part of the result's identity.
    cand_map: Mapping[int, list[Candidate]] | None = field(default=None, compare=False, repr=False)

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


_ALU_OPCODES = frozenset({Opcode.ADD, Opcode.SUB, Opcode.MUL})
_T_MACC = Opcode.T_MACC
_PRICED_CONTRACTS = frozenset({"exact", "hash"})


def _base_cost(
    claim: Claim, h: HProfile, width: int, stride_penalty: int, extra_compile: int, bw_f, lat_f
) -> tuple[int, ...]:
    """The one spelling of a realization's base cost: the 12 axes in `cost.DIMS` order as
    plain integers. `bw_f`/`lat_f` are the Q8 bandwidth and latency factors of the tier the
    claim's primary resource lives in (256/256, DRAM, when there is none).

    `_cost` wraps it in a `CostVector` for the object API; the compact planner reads the
    tuple directly. It is `_opclass`, `_streams` and `_verify_cost` inlined -- the planner's
    innermost arithmetic, run once per candidate per claim on the planner and on R9."""
    count = claim.count
    n = count if count > 1 else 1
    streams = len(claim.rd) + len(claim.wr)
    ceil = (n + width - 1) // width
    op = claim.opcode
    compute = ceil if op in _ALU_OPCODES else (ceil * 2 if op == _T_MACC else 0)
    # Memory hierarchy: scale traffic by the tier bandwidth factor and latency by the
    # tier latency factor. DRAM (or no resource) uses Q8 256/256 == x1.0, so RAM
    # resources cost exactly as before (back-compat).
    mem_shared = (n * streams * h.mem_unit * bw_f) >> 8
    access_ops = ceil * streams
    overhead = h.base_overhead * stride_penalty
    memory = mem_shared + ((access_ops * overhead * lat_f) >> 8)
    heat = ceil * h.per_op_heat
    thermal = width * h.thermal_density + heat
    power = width * h.power_density + heat
    verification = n if claim.verify in _PRICED_CONTRACTS else 0
    # Positional in DIMS order (compute, memory, fabric, sync, compile, thermal, power,
    # reliability, security, accuracy, contention, verification).
    return (compute, memory, 0, 0, extra_compile, thermal, power, 0, 0, 0, 0, verification)


def _cost(
    claim: Claim, h: HProfile, width: int, stride_penalty: int, extra_compile: int = 0, tier=None
) -> CostVector:
    bw_f = tier.bw_factor if tier is not None else 256
    lat_f = tier.lat_factor if tier is not None else 256
    return CostVector(_base_cost(claim, h, width, stride_penalty, extra_compile, bw_f, lat_f))


# --- the candidate enumeration ----------------------------------------------------

_CONTROL_NOOPS = frozenset({Opcode.NOP, Opcode.PHASE_ENTER, Opcode.PHASE_LEAVE, Opcode.PROV_NOTE})
_BARRIER = Opcode.BARRIER
_ZERO_BASE = (0,) * N
_BARRIER_BASE = CostVector.of(sync=16).v
_UNIT_OR_SCALAR = frozenset({StrideClass.UNIT, StrideClass.SCALAR})
_LANE_U, _LANE_UX, _LANE_T, _LANE_GGG, _LANE_A, _LANE_H = (
    Lane.U,
    Lane.UX,
    Lane.T,
    Lane.GGG,
    Lane.A,
    Lane.H,
)

# How a realization carries its claim's operands (`Candidate.reads` / `.writes`), exactly as
# the object enumeration always built them: a control op carries none, the two realizations of
# a reducible-permutation gather carry the claim's `wr` as declared, every other realization a
# tuple of it.
_RW_NONE, _RW_TUPLE, _RW_RAW = 0, 1, 2


def _geometry(h: HProfile) -> tuple:
    """The target's lane geometry as the enumeration reads it, derived once per plan:
    (the issuable widths, ascending; the cacheline-bucket width; the tile width)."""
    widths = h.widths()
    # The tile lane is the widest the hardware can issue, capped at 16 (AVX-512 f32): 16 on
    # AVX-512/SVE/RVV/PTX, but 4 on NEON, 8 on AVX2 -- never an unrealizable width.
    return widths, min(8, max(widths)), min(16, h.lane_widths[-1])


def _offer_rows(
    claim: Claim,
    h: HProfile,
    geometry: tuple,
    bw_f,
    lat_f,
    access: str,
    only=None,
    mode: int = 0,
) -> list[tuple]:
    """The one enumeration of a claim's legal realizations, as rows
    `(lane, width, name, base, rw)` -- `base` the 12-int cost tuple, `rw` how the candidate
    carries the claim's operands. No `Candidate` is built: `candidates_for` materializes rows
    for the object API, and the compact planner (`fused_offer`) reads them as they are.

    `geometry` is `_geometry(h)`, and (`bw_f`, `lat_f`, `access`) describe the claim's
    primary resource (its tier's Q8 factors and its addressing model; 256, 256, "flat"
    without one). `only`, a set of `(lane, width, name)`, prices just the realizations it
    names -- R9's single-candidate re-derivation -- and returns an empty list when the claim
    admits none of them. `mode` is the intra-phase discount `fused_offer` found (`_CSE`,
    `_DEFOREST`), applied to every row's base."""
    op = claim.opcode
    if op in _CONTROL_NOOPS:
        rows = [(_LANE_H, 1, "noop", _ZERO_BASE, _RW_NONE)]
        rows = rows if only is None or (_LANE_H, 1, "noop") in only else []
        return _discounted(claim, rows, mode) if mode else rows
    if op == _BARRIER:
        rows = [(_LANE_H, 1, "barrier", _BARRIER_BASE, _RW_NONE)]
        rows = rows if only is None or (_LANE_H, 1, "barrier") in only else []
        return _discounted(claim, rows, mode) if mode else rows

    # An ATOMIC read-modify-write has exactly one realization: itself, one element at a
    # time, on the atomic lane. Candidate generation used to dispatch on `stride_class`
    # alone, so an ATOMIC_ADD inherited whatever the geometry admitted -- a 16-wide vector
    # for SCALAR, a gather for RANDOM -- and BOTH verifiers passed the resulting plan
    # clean, because R9 also only ever compared lane against stride class. A vectorized
    # read-modify-write is not a faster atomic; it is not an atomic at all, and the lost
    # ordering and synchronization cost were not priced either.
    if op in ATOMIC_OPCODES:
        specs = [(_LANE_A, 1, "atomic", 1, 0, _RW_TUPLE)]
    else:
        # Hierarchical Access Memory turns random access from O(gather_penalty) into
        # O(log n).
        gp = h.gather_penalty
        if access == "ham":
            n = max(1, claim.count)
            gp = max(1, (n - 1).bit_length())  # ceil(log2 n)
        sc = claim.stride_class
        # Reducible-permutation gather (`reduce.gather`): + is commutative and the index is
        # a permutation, so the random gather (O(gather_penalty)) has a semantically
        # identical *blocked* sequential realization (O(1) overhead). The cost model offers
        # both; the blocked one wins, avoiding gather_penalty.
        if claim.op == "reduce.gather":
            specs = [
                (_LANE_U, 1, "blocked", 1, 0, _RW_RAW),
                (_LANE_GGG, 1, "gather", gp, 0, _RW_RAW),
            ]
        elif sc in _UNIT_OR_SCALAR:
            widths = geometry[0]
            specs = [
                (
                    claim.lane if w == 1 else _LANE_U,
                    w,
                    "scalar" if w == 1 else f"vec{w}",
                    1,
                    0,
                    _RW_TUPLE,
                )
                for w in widths
            ]
        elif sc == StrideClass.STRIDED:
            specs = [
                (_LANE_U, 1, "strided", _stride_penalty(claim, h), 0, _RW_TUPLE),
                (_LANE_GGG, 1, "gather", gp, 0, _RW_TUPLE),
            ]
        elif sc == StrideClass.CACHELINE:
            specs = [
                (_LANE_UX, geometry[1], "ux_bucket", 2, claim.count // 4, _RW_TUPLE),
                (_LANE_GGG, 1, "gather", gp, 0, _RW_TUPLE),
            ]
        elif sc == StrideClass.RANDOM:
            # Correctness: do not assume locality. Only the declared GGG realization is
            # legal (HAM-aware).
            specs = [(_LANE_GGG, 1, "gather", gp, 0, _RW_TUPLE)]
        elif sc == StrideClass.TILE:
            specs = [(_LANE_T, geometry[2], "tile", 1, 0, _RW_TUPLE)]
        else:  # defensive fallback
            specs = [(claim.lane, 1, "scalar", _stride_penalty(claim, h), 0, _RW_TUPLE)]
    if mode == _CSE:
        # The copy-cost couple (`_cse_memory_q8`): no recompute, the result copied once.
        q = _cse_memory_q8(claim)
        return [
            (lane, w, name, (0, (b[1] * q) >> 8) + b[2:], rw)
            for lane, w, name, sp, extra, rw in specs
            if only is None or (lane, w, name) in only
            for b in (_base_cost(claim, h, w, sp, extra, bw_f, lat_f),)
        ]
    if mode == _DEFOREST:
        # The deforestation couple (`_DEFOREST_FACTOR`): x0.75 on the memory axis.
        return [
            (lane, w, name, (b[0], (b[1] * 192) >> 8) + b[2:], rw)
            for lane, w, name, sp, extra, rw in specs
            if only is None or (lane, w, name) in only
            for b in (_base_cost(claim, h, w, sp, extra, bw_f, lat_f),)
        ]
    return [
        (lane, w, name, _base_cost(claim, h, w, sp, extra, bw_f, lat_f), rw)
        for lane, w, name, sp, extra, rw in specs
        if only is None or (lane, w, name) in only
    ]


def _materialize(claim: Claim, row: tuple) -> Candidate:
    """A `Candidate` from an offer row, carrying the claim's operands the way `rw` says."""
    lane, width, name, base, rw = row
    if rw == _RW_TUPLE:
        return Candidate(lane, width, name, CostVector(base), claim.rd, tuple(claim.wr))
    if rw == _RW_RAW:
        return Candidate(lane, width, name, CostVector(base), claim.rd, claim.wr)
    return Candidate(lane, width, name, CostVector(base))


def _resource_factors(h: HProfile, resource) -> tuple[int, int, str]:
    """(bw_factor, lat_factor, access) of a claim's primary resource on `h`."""
    if resource is None:
        return 256, 256, "flat"
    tier = h.mem.tier_for(resource.domain)
    return tier.bw_factor, tier.lat_factor, resource.access


def candidates_for(claim: Claim, h: HProfile, resource=None) -> list[Candidate]:
    bw_f, lat_f, access = _resource_factors(h, resource)
    return [
        _materialize(claim, row) for row in _offer_rows(claim, h, _geometry(h), bw_f, lat_f, access)
    ]


# --- context coupling f_i(pi) ---------------------------------------------------

# Deforestation discount (x0.75 memory): a fused producer->consumer pass elides the
# intermediate operand's round-trip. Baked into the consumer's base cost in `optimize`
# (dependency-based) so it prices identically in the plan score and the GEM makespan.
_DEFOREST_FACTOR = tuple(192 if i == MEMORY else 256 for i in range(len(IDENTITY_FACTOR)))
# The same x0.75 on the memory axis, path-based: a vector candidate that shares a read operand
# with its vector predecessor reuses loaded cache lines.
_FUSED_MEMORY_Q8 = 192
# Thermal coupling: wide SIMD on a hot machine pays extra heat/current (AVX-512 downclock).
_HOT_WIDE_Q8 = 320


def _shares_reads(prev: "Candidate | None", cand: Candidate) -> bool:
    """Path fusion: `cand` and its predecessor are both vector realizations over a common
    read operand."""
    return (
        prev is not None
        and cand.width > 1
        and prev.width > 1
        and bool(set(prev.reads) & set(cand.reads))
    )


def _hot_wide(theta: Theta, width: int) -> bool:
    return theta.thermal >= 60 and width >= 16


def _context_factor(prev: "Candidate | None", cand: Candidate, theta: Theta) -> tuple[int, ...]:
    f = list(IDENTITY_FACTOR)
    # Fusion / locality: a vector candidate that shares a read operand with its vector
    # predecessor reuses loaded cache lines -> discount memory traffic. (Producer->
    # consumer "deforestation" fusion is dependency-based, not path-based, so it is
    # baked into the consumer's base cost in `optimize`, not here.)
    if _shares_reads(prev, cand):
        f[MEMORY] = _FUSED_MEMORY_Q8
    if _hot_wide(theta, cand.width):
        f[THERMAL] = _HOT_WIDE_Q8
        f[POWER] = _HOT_WIDE_Q8
    return tuple(f)


def _edge_cost_pair(base, hot: bool, w) -> tuple[int, int]:
    """The one spelling of a DAG edge's weight: a realization's base cost coupled and
    scalarized under the phase weights `w`, for both path contexts at once -- (after a
    predecessor it does not fuse with, after one it does). `hot` is `_hot_wide`.

    Equal to `CostVector(base).couple(_context_factor(prev, cand, theta)).dot(w)` in every
    context: the Q8 identity (x256 >> 8) is exact on integers, so only the memory axis
    (path fusion) and the thermal and power axes (hot wide SIMD) are ever rescaled."""
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11 = base
    if hot:
        b5 = (b5 * _HOT_WIDE_Q8) >> 8
        b6 = (b6 * _HOT_WIDE_Q8) >> 8
    rest = (
        b0 * w[0]
        + b2 * w[2]
        + b3 * w[3]
        + b4 * w[4]
        + b5 * w[5]
        + b6 * w[6]
        + b7 * w[7]
        + b8 * w[8]
        + b9 * w[9]
        + b10 * w[10]
        + b11 * w[11]
    )
    return rest + b1 * w[1], rest + ((b1 * _FUSED_MEMORY_Q8) >> 8) * w[1]


def edge_cost(prev: "Candidate | None", cand: Candidate, theta: Theta, w_phase) -> int:
    """The realized cost of `cand` following `prev` under the phase weights `w_phase`: the
    planner's DAG edge weight, and the number R9 re-derives per step. One predicate for
    both, so the plan score and the verdict cannot disagree about what a step costs."""
    plain, fused = _edge_cost_pair(cand.base.v, _hot_wide(theta, cand.width), w_phase)
    return fused if _shares_reads(prev, cand) else plain


def step_cost(
    prev: "Candidate | None",
    cand: Candidate,
    h: HProfile,
    theta: Theta,
    phase_id: int,
    policy: Policy = PERF,
) -> int:
    """`edge_cost` with the phase weights derived from the scope (h, theta, policy)."""
    return edge_cost(prev, cand, theta, weights(h, theta, phase_id, policy))


# --- phase ordering -------------------------------------------------------------


def _topo_phases(module: Module):
    pmap = module.phase_map()
    return [pmap[pid] for pid in topological_phase_ids(module)]


def _flatten(module: Module) -> list[tuple[int, Claim]]:
    flat: list[tuple[int, Claim]] = []
    for ph in _topo_phases(module):
        for cl in ph.claims:
            flat.append((ph.phase_id, cl))
    return flat


# --- fusion-aware candidate generation (shared by the tropical + RCSP rails) -----


#: Opcodes that are EFFECT edges, never a value in hand: a second occurrence is a second
#: effect, not a copy of the first (G1 / S1-A CSE exclusions; `cm::cseEligible` mirrors it).
EFFECTFUL_OPCODES = frozenset(
    {Opcode.BARRIER, Opcode.PHASE_ENTER, Opcode.PHASE_LEAVE, Opcode.GEM_DISPATCH, Opcode.PROV_NOTE}
)


def cse_eligible(claim: Claim, module: Module) -> bool:
    """Whether a claim may take part in CSE at all -- as the duplicate OR as the seed an
    earlier occurrence provides. Categorical, before any signature is compared: an atomic
    read-modify-write, a barriered or volatile claim, an effect opcode (barrier, phase
    enter/leave, dispatch, provenance note), an indirect call, a claim with timing or
    lifetime metadata, a sparse GGG/random access (its elements depend on index data the
    claim does not carry), a claim of or touching an isolated (MMIO) domain, a claim with
    a non-default numeric contract (`precision`, `tolerance_ulp`, `quantized_bits`) and a
    claim carrying a field the law rail cannot see (`imm`) are never a common
    subexpression. The 2026-09-04 assessment (row 7) found the credit
    granted to atomic and volatile duplicates with the barrier guard checked afterwards;
    `BCIRCostModel.h::cseEligible` is the byte-for-byte mirror (R13 parity)."""
    if not claim.rd:
        return False
    if claim.hazard != "unique" or claim.volatile:
        return False
    if claim.opcode in ATOMIC_OPCODES or claim.opcode in EFFECTFUL_OPCODES:
        return False
    if claim.callee_sig or claim.timing is not None or claim.lifetime is not None:
        return False
    if claim.lane == Lane.GGG or claim.stride_class == StrideClass.RANDOM:
        return False
    if claim.imm or claim.tolerance_ulp or claim.quantized_bits or claim.precision:
        return False
    if claim.domain in ISOLATED_DOMAINS:
        return False
    # Every operand, reads then writes (`claim.io_rids()`), resolved as `module.resource`
    # resolves it -- inline, because this runs once per claim on the planner and on R9.
    resources = module.resources
    for rids in (claim.rd, claim.wr):
        for rid in rids:
            if rid is not None and rid in resources and resources[rid].domain in ISOLATED_DOMAINS:
                return False
    return True


def cse_identity(claim: Claim, version: dict[int, int]) -> tuple:
    """The complete semantic identity of a claim's VALUE: the op (string and opcode), the
    access pattern that decides which elements it touches (lane, stride class, stride
    multiplier, offset, count), the domain, the dynamic-bound flag, and its read operands
    AT THEIR CURRENT VERSIONS (a write bumps a version, so
    a rewrite between two occurrences invalidates the match). Two claims with the same
    identity compute the same value; the second is a copy of the first. The old signature
    was the op and the read versions alone, so a duplicate over a different count, offset
    or stride took the copy credit."""
    return (
        claim.op,
        int(claim.opcode),
        int(claim.lane),
        int(claim.stride_class),
        claim.count,
        claim.stride_k,
        claim.offset,
        int(claim.domain),
        bool(claim.dynamic),
        tuple([(r, version[r] if r in version else 0) for r in claim.rd]),
    )


def _cse_memory_q8(claim: Claim) -> int:
    """The copy-cost factor on the memory axis for a claim that is a common subexpression of
    an earlier one: the value is already computed, so there is no recompute (compute zeroed)
    and only the result is copied to this claim's output instead of re-reading every
    operand (memory scaled from `len(rd)+len(wr)` streams down to the `1 + len(wr)` a copy
    needs). Conservative on thermal/power (left as-is)."""
    return ((1 + len(claim.wr)) * 256) // max(1, len(claim.rd) + len(claim.wr))


# How the intra-phase data flow discounted a claim's realizations (`Offer.mode`).
_PLAIN, _CSE, _DEFOREST = 0, 1, 2

# The discount rule, as ONE table both evaluations of the offer read (GEM+ G18): the sequential
# walk of `fused_offer` and the indexed re-derivation of `kbcir.delta` gather the three facts
# their own way and look the mode up here, so the rule itself is spelled once.
#   _DISCOUNT[duplicate][consumes][fenced]
#   duplicate  -- the claim is CSE-eligible and an earlier eligible claim of its phase has its
#                 value-numbered identity: CSE wins, being the larger credit
#   consumes   -- a read operand was written earlier in the phase (producer -> consumer)
#   fenced     -- the claim is barriered, or one of its reads was written by a barriered claim
#                 earlier in the phase (ASM3b: the fence materializes the intermediate)
_DISCOUNT = (
    ((_PLAIN, _PLAIN), (_DEFOREST, _PLAIN)),
    ((_CSE, _CSE), (_CSE, _CSE)),
)


class Offer:
    """The planner's offer as compact indexed arrays (GEM+ G17): one column per claim in
    planning order, one row tuple `(lane, width, name, base, rw)` per realization, and no
    `Candidate` or `CostVector` until one is asked for.

    `optimize`, `fused_candidates` and R9 read the same arrays -- the one derivation of what
    the planner may choose from. `rows[j]` is the offer of the j-th flat entry;
    `source[j]` is the entry whose rows column j plans with, which is `j` unless a claim id
    occurs twice (then every occurrence plans with the last one's rows, as the id-keyed
    candidate map always made it)."""

    __slots__ = ("claims", "phase_ids", "rows", "source", "modes", "factors", "geometry", "h")

    def __init__(self, claims, phase_ids, rows, source, modes, factors, geometry, h):
        self.claims = claims
        self.phase_ids = phase_ids
        self.rows = rows
        self.source = source
        self.modes = modes
        self.factors = factors
        self.geometry = geometry
        self.h = h

    def __len__(self) -> int:
        return len(self.claims)

    def candidates(self, j: int) -> list[Candidate]:
        """Column j's realizations as `Candidate` objects."""
        s = self.source[j]
        claim = self.claims[s]
        return [_materialize(claim, row) for row in self.rows[s]]

    def candidate_map(self) -> dict[int, list[Candidate]]:
        """`fused_candidates`' dict: claim id -> candidates, first-occurrence order."""
        out: dict[int, list[Candidate]] = {}
        for j, claim in enumerate(self.claims):
            out[claim.id] = self.candidates(j)
        return out

    def full_rows(self, j: int) -> list[tuple]:
        """Every realization of entry j with its discount applied, whatever `only` priced --
        the diagnostic path of R9's single-candidate re-derivation."""
        bw_f, lat_f, access = self.factors[j]
        return _offer_rows(
            self.claims[j], self.h, self.geometry, bw_f, lat_f, access, None, self.modes[j]
        )


def _discounted(claim: Claim, rows: list[tuple], mode: int) -> list[tuple]:
    """Apply an intra-phase discount to rows already priced (the Q8 couple, inline)."""
    if mode == _CSE:
        q = _cse_memory_q8(claim)
        return [(lane, w, name, (0, (b[1] * q) >> 8) + b[2:], rw) for lane, w, name, b, rw in rows]
    if mode == _DEFOREST:
        return [
            (lane, w, name, (b[0], (b[1] * 192) >> 8) + b[2:], rw) for lane, w, name, b, rw in rows
        ]
    return rows


_NO_REALIZATIONS: frozenset = frozenset()


def fused_offer(module: Module, h: HProfile, only=None) -> Offer:
    """Per-claim offers with the **redundancy discounts** baked in, computed from
    intra-phase data flow (not path adjacency) so all five rails -- tropical, RCSP, soft,
    accel, scheduled overlap -- price them identically and the plan score, makespan, and
    serial bound stay consistent (makespan <= serial). Two discounts, applied at most one
    per claim (CSE wins, being the larger):

      * **CSE / duplicate elimination**: a claim whose complete semantic identity
        (`cse_identity`: op, access pattern, domain, numeric contract, operand
        value-numbers) matches an earlier same-phase claim recomputes a value already
        in hand -- it becomes a copy (no recompute, no operand reload). Value numbering
        (a write bumps an operand's version) makes the match sound: a rewrite between
        the two invalidates it. Only `cse_eligible` claims seed or take the credit --
        an atomic, barriered, volatile, effectful, sparse or isolated-domain claim is
        never a common subexpression. The egraph proves the same liked-pair CSE; this
        prices it.
      * **producer->consumer deforestation**: a claim reading an operand a prior
        same-phase claim produced fuses with it, so the intermediate never round-trips
        to memory (a memory discount).

    A barrier between phases materializes intermediates, so both credits are
    intra-phase only; single-claim programs (e.g. vector_add) get neither (a no-op,
    so the pinned scores are preserved).

    `only` (claim id -> set of `(lane, width, name)`) prices only the named realizations of
    the named claims -- R9 re-derives exactly what the plan chose -- while the data-flow
    state still walks every claim, because a discount depends on everything before it."""
    flat = _flatten(module)
    count = len(flat)
    geometry = _geometry(h)
    resources = module.resources
    tiers: dict = {}  # domain -> (bw_factor, lat_factor): `tier_for` once per domain
    claims: list = [None] * count
    phase_ids: list = [None] * count
    rows: list = [None] * count
    modes: list = [_PLAIN] * count
    factors: list = [None] * count
    last: dict[int, int] = {}  # claim id -> its last flat entry
    phase = None
    pset: set[int] = set()  # rids written so far in the phase (deforestation)
    ver: dict[int, int] = {}  # {rid: write count} in the phase (value numbering)
    seenmap: dict[tuple, int] = {}  # compute signature -> first claim in the phase
    bset: set[int] = set()  # rids written by a barriered producer in the phase (ASM3b)
    for j in range(count):
        phase_id, claim = flat[j]
        if phase_id != phase:  # a phase's claims are contiguous in planning order
            phase = phase_id
            pset, ver, seenmap, bset = set(), {}, {}, set()
        rd = claim.rd
        cost_rid = rd[0] if rd else claim.primary_rid
        if cost_rid is not None and cost_rid in resources:
            resource = resources[cost_rid]
            domain = resource.domain
            if domain in tiers:
                bw_f, lat_f = tiers[domain]
            else:
                tier = h.mem.tier_for(domain)
                bw_f, lat_f = tiers[domain] = (tier.bw_factor, tier.lat_factor)
            access = resource.access
        else:
            bw_f, lat_f, access = 256, 256, "flat"
        # The value-numbered semantic identity (G1): the same value at the same versions.
        eligible = cse_eligible(claim, module)
        sig = cse_identity(claim, ver) if eligible else None
        # The discount (`_DISCOUNT`): CSE when an identical value is already computed, else
        # producer->consumer deforestation unless a fence stands between them. ASM3b: a
        # barriered claim is a first-class ordering edge, so neither a barriered consumer nor a
        # read a barriered producer wrote fuses (the fence materializes the intermediate). Every
        # barriered write is also in `pset`, so the reads a fence covers are `bset & rd`. A
        # duplicate's row of the table does not depend on the other two facts, so they are
        # gathered only for a claim that is not one.
        if eligible and sig in seenmap:
            mode = _DISCOUNT[True][False][False]
        else:
            mode = _DISCOUNT[False][not pset.isdisjoint(rd)][
                claim.hazard == "barriered" or (not bset.isdisjoint(rd) if bset else False)
            ]
        want = None if only is None else only.get(claim.id, _NO_REALIZATIONS)
        rows[j] = _offer_rows(claim, h, geometry, bw_f, lat_f, access, want, mode)

        if eligible and sig not in seenmap:
            seenmap[sig] = claim.id  # first occurrence pays full
        wr = claim.wr
        for r in wr:  # a write creates a new operand version
            ver[r] = ver[r] + 1 if r in ver else 1
        pset.update(wr)
        if claim.hazard == "barriered":  # ASM3b: a barriered producer fences its writes
            bset.update(wr)
        last[claim.id] = j
        claims[j] = claim
        phase_ids[j] = phase_id
        modes[j] = mode
        factors[j] = (bw_f, lat_f, access)
    source = range(count) if len(last) == count else [last[claim.id] for claim in claims]
    return Offer(claims, phase_ids, rows, source, modes, factors, geometry, h)


def fused_candidates(module: Module, h: HProfile) -> dict[int, list[Candidate]]:
    """Per-claim candidate lists with the redundancy discounts baked in (`fused_offer`),
    as `Candidate` objects: claim id -> candidates."""
    return fused_offer(module, h).candidate_map()


class OfferMap(Mapping):
    """`RealizationResult.cand_map` over a compact `Offer`: claim id -> candidates, each
    list built on first access and then kept. Read-only, like the dict it stands in for."""

    __slots__ = ("_offer", "_column", "_cache")

    def __init__(self, offer: Offer, column: dict[int, int] | None = None):
        """`column` (claim id -> its first entry) may be handed in by a caller that already
        holds it for the same claim order -- the incremental planner, whose deltas never move
        a claim -- and is otherwise derived from the offer."""
        self._offer = offer
        if column is None:
            column = {}
            for j, claim in enumerate(offer.claims):
                if claim.id not in column:
                    column[claim.id] = j
        self._column = column
        self._cache: dict[int, list[Candidate]] = {}

    def __getitem__(self, claim_id: int) -> list[Candidate]:
        cached = self._cache.get(claim_id)
        if cached is None:
            cached = self._cache[claim_id] = self._offer.candidates(self._column[claim_id])
        return cached

    def __iter__(self) -> Iterator[int]:
        return iter(self._column)

    def __len__(self) -> int:
        return len(self._column)


# --- the optimizer --------------------------------------------------------------


def _relax_column(crow, k, w, hot, shares, pn, dn, pw, dw, dist, pred, first):
    """Relax one column of the realization DAG -- the one relaxation `optimize` and the
    incremental planner (`kbcir.delta`, GEM+ G18) both run.

    Slot `first + i` (i < k, the column's `len(crow)`, which every caller already holds)
    receives row i's least path weight (`dist`) and the predecessor slot it came
    from (`pred`), from the previous column's cheapest narrow slot `pn` (weight `dn`) and
    cheapest wide slot `pw` (weight `dw`), each -1 when absent: both -1 is the first column,
    entered from SOURCE (pred -1). An edge into row i costs its fused value when the predecessor
    is wide, i is wide and the two claims share a read (`shares`), its plain value otherwise; a
    tie keeps the predecessor that comes first -- the order of the slot ids the caller passes,
    whatever their origin (`optimize` passes flat slots, the incremental planner column-relative
    ones), exactly as `semiring.dag_shortest_path` relaxes with a strict `<`. Only differences
    of weights decide anything here, so a caller may pass weights shifted by any constant and
    receive them shifted by the same constant. Returns this column's cheapest narrow and wide
    slots, the first on a tie, -1 when the column has none."""
    for i in range(k):
        width = crow[i][1]
        plain, fused = _edge_cost_pair(crow[i][3], hot and width >= 16, w)
        g = first + i
        if pn >= 0:
            best, bp = dn + plain, pn
            if pw >= 0:
                d = dw + (fused if shares and width > 1 else plain)
                if d < best or (d == best and pw < bp):
                    best, bp = d, pw
        elif pw >= 0:
            best, bp = dw + (fused if shares and width > 1 else plain), pw
        else:  # SOURCE -> c
            best, bp = plain, -1
        dist[g] = best
        pred[g] = bp
    narrow = wide = -1
    for i in range(k):
        g = first + i
        if crow[i][1] > 1:
            if wide < 0 or dist[g] < dist[wide]:
                wide = g
        elif narrow < 0 or dist[g] < dist[narrow]:
            narrow = g
    return narrow, wide


def optimize(module: Module, h: HProfile, theta: Theta, policy: Policy = PERF) -> RealizationResult:
    """The min-plus shortest path over the layered realization DAG (one column per claim,
    one node per realization, SOURCE before the first column and SINK after the last), over
    the compact offer.

    An edge into realization c costs `_edge_cost_pair(c)` -- the fused value when c and its
    predecessor are both vector realizations over a shared read, the plain value otherwise --
    so for each column only two predecessors can win: the cheapest narrow one and the
    cheapest wide one. The relaxation keeps the first predecessor in node order on a tie,
    exactly as `semiring.dag_shortest_path` relaxes with a strict `<`; the plan is the
    pre-G17 planner's plan, byte for byte (`planner.parity`, `realize_reference`)."""
    offer = fused_offer(module, h)
    claims = offer.claims
    n = len(claims)
    if not n:
        return RealizationResult([], 0)
    rows, source, phase_ids = offer.rows, offer.source, offer.phase_ids
    hot = theta.thermal >= 60  # `_hot_wide` once per plan: hot and width >= 16

    # One slot per node, column by column: `start[j]` is column j's first slot, `dist` the
    # least path weight into a slot and `pred` the slot it came from (-1: SOURCE). Two flat
    # lists rather than two per column, so the collector has two objects to walk, not 2n; sized
    # by the most realizations a column can hold (every width, or two), so no pass counts them.
    cap = n * max(2, len(offer.geometry[0]))
    start = [0] * n
    dist = [0] * cap
    pred = [-1] * cap
    total = 0
    prev_rd = None
    narrow = wide = -1  # the previous column's cheapest narrow / wide slot (first on a tie)
    wpid = w = None
    for j in range(n):
        s = source[j]
        pid = phase_ids[j]
        if w is None or pid != wpid:  # a phase's columns are one run: weigh it once per run
            w, wpid = weights(h, theta, pid, policy), pid
        crow = rows[s]
        k = len(crow)
        first = start[j] = total
        total += k
        rd = claims[s].rd
        shares = prev_rd is not None and not set(prev_rd).isdisjoint(rd)
        narrow, wide = _relax_column(
            crow,
            k,
            w,
            hot,
            shares,
            narrow,
            dist[narrow] if narrow >= 0 else 0,
            wide,
            dist[wide] if wide >= 0 else 0,
            dist,
            pred,
            first,
        )
        prev_rd = rd

    # SINK: the first last-column slot with the least distance.
    g = start[n - 1]
    for k in range(g + 1, total):
        if dist[k] < dist[g]:
            g = k
    score = dist[g]

    # Reconstruct the chosen path SINK -> SOURCE.
    steps: list = [None] * n
    for j in range(n - 1, -1, -1):
        s = source[j]
        p = pred[g]
        claim = claims[s]
        lane, width, name, base, rw = rows[s][g - start[j]]
        if rw == _RW_TUPLE:
            cand = Candidate(lane, width, name, CostVector(base), claim.rd, tuple(claim.wr))
        elif rw == _RW_RAW:
            cand = Candidate(lane, width, name, CostVector(base), claim.rd, claim.wr)
        else:
            cand = Candidate(lane, width, name, CostVector(base))
        steps[j] = ChosenStep(
            claims[j].id, phase_ids[j], cand, dist[g] - (dist[p] if p >= 0 else 0)
        )
        g = p
    return RealizationResult(steps, score, cand_map=OfferMap(offer))
