"""ASM3b: a `barriered`-hazard claim is a FIRST-CLASS ORDERING EDGE.

ASM3 made memory fences emit real per-ISA asm; ASM3b adds the other half the docs promise: a
barriered claim FENCES OTHER claims -- no claim may be reordered across it, and a producer->consumer
pair spanning it must NOT get the fusion (deforestation) cost discount. The claim kinds that carry
`hazard="barriered"` are memory fences, MMIO loads/stores, port-I/O, and volatile/"memory"-clobber
inline asm; ASM3b wires the fence into the existing bundle/cost machinery:

  * Guard 1 (`bundle._conflict`): a barriered claim conflicts with everything, so `find_bundles` /
    `_legal_reorder` never bundle or reorder any claim across it (a hard reorder fence).
  * Guard 2 (`realize.fused_candidates`): the x0.75 deforestation discount is SKIPPED when the
    consumer is barriered OR a shared operand was produced by a barriered producer (the fence forces
    the intermediate to materialize -- the round-trip is not elided across an ordering edge).
  * The MLIR cost model (`BCIRCostModel.h::fusedColumns`) mirrors Guard 2 byte-for-byte (R13 parity;
    the FileCheck twin is mlir/test/passes/cost_model_barrier.mlir).
  * A STRUCTURAL invariant (`verify.verify_barrier_ordering`, NOT a verdict R-law) verifies a realized
    plan respects the fence -- the property Guard 1 guarantees by construction.

A SAFE NO-OP on the existing corpus: a module with no barriered claim gets neither guard (vacuous),
so the pinned scores are preserved.
"""

from dataclasses import replace

from bcir.examples import scan_chain
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.bundle import _conflict, _legal_reorder, find_bundles, optimize_bundled
from bcir.kbcir.cost import Theta
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify, verify_barrier_ordering, verify_plan

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def _interleaved(barriered_member=False):
    """c1 reads A,B; c2 reads D,E (no share); c3 reads A,G (shares A with c1). Declared order
    separates the A-sharers, so the pairwise model misses their fusion -- the bundle reorderer
    normally clusters {c1, c3}. With `barriered_member`, c3 is a barrier and may not be bundled."""
    m = Module(name="interleaved")
    for rid in range(1, 9):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(1024,)))
    cs = [
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=1024,
              rd=(1, 2), wr=(3,), op="vector.add", domain=Domain.RAM),
        Claim(id=2, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=1024,
              rd=(4, 5), wr=(6,), op="vector.add", domain=Domain.RAM),
        Claim(id=3, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=1024,
              rd=(1, 7), wr=(8,), op="vector.add", domain=Domain.RAM,
              hazard="barriered" if barriered_member else "unique"),
    ]
    m.add_phase(Phase(phase_id=0, deps=(), claims=cs))
    return m


# --- Guard 1: a barrier is a hard reorder fence -----------------------------------

def test_conflict_treats_a_barrier_as_conflicting_with_everything():
    """`_conflict` is the predicate find_bundles / _legal_reorder use; a barriered claim conflicts
    with every other claim even when there is NO data hazard between them."""
    a = Claim(id=1, opcode=Opcode.ADD, rd=(10,), wr=(11,), hazard="barriered")
    b = Claim(id=2, opcode=Opcode.ADD, rd=(20,), wr=(21,))           # disjoint operands, no RAW/WAR/WAW
    assert _conflict(a, b) and _conflict(b, a)                       # the fence makes them conflict
    # control: two independent unique claims do NOT conflict (the fence is the only reason here).
    c = replace(a, hazard="unique")
    assert not _conflict(c, b)


def test_a_barriered_claim_is_never_bundled():
    """find_bundles clusters mutually-independent same-phase sharers; a barriered member can never
    join a bundle (it conflicts with every candidate), so the discount-bearing reorder is forbidden."""
    assert find_bundles(_interleaved()) and find_bundles(_interleaved())[0].claim_ids == (1, 3)
    assert find_bundles(_interleaved(barriered_member=True)) == []   # the barrier dissolves the bundle


def test_legal_reorder_never_moves_a_claim_across_a_barrier():
    """A claim before a barrier stays before it (and after stays after): a barrier is a hard fence
    in the linear-extension legality check, independent of any data hazard."""
    c1 = Claim(id=1, opcode=Opcode.ADD, rd=(1,))
    bar = Claim(id=2, opcode=Opcode.ADD, rd=(4,), hazard="barriered")
    c3 = Claim(id=3, opcode=Opcode.ADD, rd=(1,))
    orig = [c1, bar, c3]
    assert _legal_reorder(orig, [c1, bar, c3])                       # identity is legal
    assert not _legal_reorder(orig, [c1, c3, bar])                   # c3 hopped before the barrier
    assert not _legal_reorder(orig, [bar, c1, c3])                   # c1 hopped after the barrier


def test_bundled_optimizer_is_a_noop_when_the_sharer_is_a_barrier():
    """optimize_bundled cannot reorder across a barrier, so the would-be {c1,c3} fusion win is
    forfeited: the barriered module's bundled plan equals its greedy plan (no certificate)."""
    res, certs = optimize_bundled(_interleaved(barriered_member=True), AVX, COOL)
    assert certs == [] and res.score == optimize(_interleaved(barriered_member=True), AVX, COOL).score


# --- Guard 2: no deforestation discount across an ordering edge --------------------

def test_deforestation_skipped_when_consumer_is_barriered():
    """scan_chain's c2 (8002) reads c1's write (rid 83) -> producer->consumer deforestation
    (x0.75 memory). Making the CONSUMER barriered fences the fusion: the intermediate must
    materialize, so the discount is skipped and the plan is STRICTLY MORE EXPENSIVE."""
    base = optimize(scan_chain(1024), AVX, COOL).score
    m = scan_chain(1024)
    m.phases[0].claims[1] = replace(m.phases[0].claims[1], hazard="barriered")
    fenced = optimize(m, AVX, COOL).score
    assert base == 13696 and fenced == 15616                         # pinned: 1920 = the forfeited x0.75
    assert fenced > base


def test_deforestation_skipped_when_producer_is_barriered():
    """Symmetric: making the PRODUCER (c1) barriered fences the SAME fusion -- c2's shared read (83)
    was produced by a barriered producer, so c2 loses the discount. Same forfeited score."""
    base = optimize(scan_chain(1024), AVX, COOL).score
    m = scan_chain(1024)
    m.phases[0].claims[0] = replace(m.phases[0].claims[0], hazard="barriered")
    fenced = optimize(m, AVX, COOL).score
    assert fenced == 15616 and fenced > base                         # the producer-side fence, same gap


def _mmio_chain(barriered=True):
    """An MMIO producer->consumer chain (lane H, Domain.MMIO -- the port-mapped device register case
    ASM3b's broad scope must fence): c1 loads a device register into T; c2 reads T. An MMIO write
    needs an ordered hazard (R3), and an MMIO access carries `barriered` -- so c2's read of c1's
    output must NOT fuse across the device-ordering edge."""
    m = Module(name="mmio_chain")
    hz = "barriered" if barriered else "atomic"
    for rid, nm in ((1, "REG"), (2, "T"), (3, "C"), (4, "O")):
        dom = Domain.MMIO if rid == 1 else Domain.RAM
        m.add_resource(Resource(rid=rid, domain=dom, shape=(1024,), name=nm))
    c1 = Claim(id=1, opcode=Opcode.LOAD, lane=Lane.H, stride_class=StrideClass.SCALAR, count=1024,
               rd=(1,), wr=(2,), op="vector.add", domain=Domain.MMIO, hazard=hz)
    c2 = Claim(id=2, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=1024,
               rd=(2, 3), wr=(4,), op="vector.add", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c1, c2]))
    return m


def test_mmio_barrier_prevents_the_deforestation_discount():
    """A barriered MMIO producer fences the downstream fusion: the consumer's plan cost is strictly
    higher than the same chain whose producer carries a non-fencing (atomic) hazard. This pins
    Guard 2's REAL effect on the MMIO/port-I/O kind ASM3b's broad scope targets."""
    barriered = optimize(_mmio_chain(barriered=True), AVX, COOL).score
    unfenced = optimize(_mmio_chain(barriered=False), AVX, COOL).score
    assert barriered > unfenced                                     # the device-ordering edge blocks fusion


# --- the structural invariant (advisory, never a verdict) -------------------------

def test_structural_invariant_passes_for_a_legal_plan():
    """A realized plan over a barriered module respects the fence by construction -- the structural
    check is clean (and vacuous for a no-barrier module)."""
    m = scan_chain(1024)
    m.phases[0].claims[1] = replace(m.phases[0].claims[1], hazard="barriered")
    assert verify_barrier_ordering(m, optimize(m, AVX, COOL)) == []
    plain = scan_chain(1024)
    assert verify_barrier_ordering(plain, optimize(plain, AVX, COOL)) == []   # vacuous: no barrier


def test_structural_invariant_flags_a_hand_built_crossing():
    """A hand-built plan that reorders a claim across the barrier is flagged (an ASM3b structural
    breach, NOT a verdict R-law -- it never appears in verify()/verify_plan())."""
    import copy
    m = scan_chain(1024)
    m.phases[0].claims[1] = replace(m.phases[0].claims[1], hazard="barriered")
    good = optimize(m, AVX, COOL)
    bad = copy.deepcopy(good)
    bad.steps = list(reversed(bad.steps))                           # barriered c2 now precedes c1
    diags = verify_barrier_ordering(m, bad)
    assert diags and all(d.law == "ASM3b" for d in diags)
    # the breach is a STRUCTURAL advisory, never a legality verdict (R1-R12 stay clean).
    assert verify(m) == [] and not any(d.law == "ASM3b" for d in verify_plan(m, bad))


def test_corpus_is_a_safe_noop_without_a_barrier():
    """The non-disturbance invariant: a module carrying no barriered claim gets neither guard, so
    its plan score is unchanged and the structural check is vacuous (the measured no-op)."""
    m = scan_chain(1024)
    assert optimize(m, AVX, COOL).score == 13696                    # the deforested score, intact
    assert verify_barrier_ordering(m, optimize(m, AVX, COOL)) == []
