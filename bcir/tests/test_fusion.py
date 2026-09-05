"""Multi-claim fusion: where the (max,+) overlap pricing and batching earn their
keep. Two independent claims sharing a read operand run as concurrent waves
(makespan < serial), and the StreamPack/batch groups them."""

from dataclasses import replace

from bcir.examples import fused_chain, histogram_gather, saxpy_strided, scan_chain, vector_add
from bcir.gem import hydrate, price_scheduled
from bcir.gem.overlap import optimize_scheduled
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def test_multi_claim_overlap_beats_the_serial_sum():
    # fused_chain: two independent vec16 claims -> they co-execute on distinct
    # affinity domains, so the (max,+) makespan is below the serial Sigma.
    m = fused_chain(1024)
    r = optimize(m, AVX, COOL)
    sp = price_scheduled(m, r, AVX, COOL)
    assert len(r.steps) == 2
    assert sp.makespan < sp.serial and sp.overlap_gain > 0


def test_single_claim_has_no_overlap():
    m = vector_add(1024)
    sp = price_scheduled(m, optimize(m, AVX, COOL), AVX, COOL)
    assert sp.overlap_gain == 0  # one claim: makespan == serial (the degenerate case)


def test_fused_chain_hydrates_two_segments():
    m = fused_chain(1024)
    pack = hydrate(m, optimize(m, AVX, COOL))
    assert len(pack.segments) == 2 and pack.provenance_ok()


def test_optimize_scheduled_is_r9_consistent():
    # the scheduled optimizer returns a serial-repriced plan (score == sum of step
    # costs) whose makespan does not exceed the serial bound.
    m = fused_chain(1024)
    r, sp = optimize_scheduled(m, AVX, COOL)
    assert sp.makespan <= sp.serial
    assert r.score == sum(s.cost for s in r.steps)


# --- scan: a dependency chain serializes (the counterpart to overlap) ------------


def test_dependent_chain_does_not_overlap():
    m = scan_chain(1024)
    sp = price_scheduled(m, optimize(m, AVX, COOL), AVX, COOL)
    assert len(m.phases[0].claims) == 2
    assert sp.overlap_gain == 0  # RAW dependency: no overlap, no fusion


# --- the fusion discount (distinct from overlap): shared operand, one bin --------


def _no_share(n=1024):
    m = Module(name="no_share")
    for rid, nm in ((60, "A"), (61, "B"), (62, "C"), (63, "E"), (64, "D"), (65, "F")):
        m.add_resource(Resource(rid=rid, shape=(n,), name=nm))
    c1 = Claim(
        id=6001,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=n,
        rd=(60, 61),
        wr=(62,),
        op="vector.add",
    )
    c2 = Claim(
        id=6002,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=n,
        rd=(63, 65),
        wr=(64,),
        op="vector.add",
    )
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c1, c2]))
    return m


def test_fusion_discount_rewards_a_shared_operand_in_one_bin():
    # On a single affinity domain both claims share one bin (back-to-back). When
    # they share a read operand (A), the second reuses A's loaded lines -> the
    # memory fusion discount; an otherwise-identical no-share chain pays full price.
    h1 = replace(AVX, affinity_domains=1)
    shared = price_scheduled(fused_chain(1024), optimize(fused_chain(1024), h1, COOL), h1, COOL)
    noshare = price_scheduled(_no_share(1024), optimize(_no_share(1024), h1, COOL), h1, COOL)
    assert shared.makespan < noshare.makespan  # the fusion discount is real


# --- per-target parity beyond vector_add -----------------------------------------


def test_per_target_parity_saxpy_and_histogram():
    # pin the selected scores across targets (the worked-example matrix widened).
    saxpy = {
        t: optimize(saxpy_strided(1024), TARGETS[t], COOL).score
        for t in ("x86_avx512", "x86_avx2", "arm64_neon", "nvidia_ptx")
    }
    assert saxpy == {
        "x86_avx512": 71680,
        "x86_avx2": 71680,
        "arm64_neon": 71680,
        "nvidia_ptx": 71680,
    }
    histo = {
        t: optimize(histogram_gather(1024), TARGETS[t], COOL).score
        for t in ("x86_avx512", "nvidia_ptx")
    }
    assert histo == {"x86_avx512": 528384, "nvidia_ptx": 266240}


# --- producer->consumer deforestation (the second fusion kind) --------------------


def _pc(c2_reads, n=1 << 16):
    """A two-claim phase; c1 writes 62; c2 reads `c2_reads` (use 62 to consume it)."""
    m = Module(name="pc")
    for rid in (60, 61, 62, 63, 64, 65):
        m.add_resource(Resource(rid=rid, shape=(n,)))
    c1 = Claim(
        id=1,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=n,
        rd=(60, 61),
        wr=(62,),
        op="vector.add",
    )
    c2 = Claim(
        id=2,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=n,
        rd=c2_reads,
        wr=(64,),
        op="vector.add",
    )
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c1, c2]))
    return m


def test_producer_consumer_deforestation_lowers_the_plan_score():
    # c2 consuming c1's output (62) fuses: the intermediate never round-trips to
    # memory (deforestation), so the plan is strictly cheaper than an identical chain
    # whose 2nd claim reads an UNRELATED operand -- with the widths unchanged.
    fused = optimize(_pc((62, 63)), AVX, COOL)  # producer -> consumer
    indep = optimize(_pc((65, 63)), AVX, COOL)  # no dependency
    assert fused.score < indep.score  # the deforestation gain (~12%)
    assert {c: x.width for c, x in fused.by_claim().items()} == {
        c: x.width for c, x in indep.by_claim().items()
    }  # no width churn


def test_deforestation_preserves_makespan_le_serial():
    # the discount is dependency-based (uniform in plan + schedule pricing), so a
    # serialized producer->consumer chain still has makespan == serial, gain 0.
    m = _pc((62, 63))
    sp = price_scheduled(m, optimize(m, AVX, COOL), AVX, COOL)
    assert sp.makespan == sp.serial and sp.overlap_gain == 0


def test_single_claim_is_unaffected_by_deforestation():
    assert optimize(vector_add(1024), AVX, COOL).score == 7808  # pinned: no producer


# --- CSE / duplicate-claim elimination (the egraph's liked-pair, finally priced) ---


def _cse(c2_reads, rewrite_a=False, n=1 << 16):
    """c1 = A(10)+B(11)->D(12); optionally a claim rewriting A(10); then c2 over
    `c2_reads`->E(13). With c2_reads=(10,11) and no rewrite, c2 duplicates c1 (CSE)."""
    m = Module(name="cse")
    for rid in (10, 11, 12, 13, 14, 15):
        m.add_resource(Resource(rid=rid, shape=(n,)))
    claims = [
        Claim(
            id=1,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=n,
            rd=(10, 11),
            wr=(12,),
            op="vector.add",
        )
    ]
    if rewrite_a:  # bumps A(10)'s value-number
        claims.append(
            Claim(
                id=9,
                opcode=Opcode.ADD,
                lane=Lane.U,
                stride_class=StrideClass.UNIT,
                count=n,
                rd=(14, 15),
                wr=(10,),
                op="vector.add",
            )
        )
    claims.append(
        Claim(
            id=2,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=n,
            rd=c2_reads,
            wr=(13,),
            op="vector.add",
        )
    )
    m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
    return m


def _c2(result):
    return next(s.cost for s in result.steps if s.claim_id == 2)


def test_cse_prices_a_duplicate_claim_as_a_copy():
    # c2 = A+B recomputes c1's value -> CSE: priced as a copy (no recompute, no
    # operand reload), strictly cheaper than an identical-shape distinct claim.
    dup = optimize(_cse((10, 11)), AVX, COOL)  # duplicate of c1 -> CSE
    distinct = optimize(_cse((14, 11)), AVX, COOL)  # different operand -> full
    assert _c2(dup) < _c2(distinct)  # the CSE copy is cheaper (~15%)
    assert {c: x.width for c, x in dup.by_claim().items()} == {
        c: x.width for c, x in distinct.by_claim().items()
    }  # no width churn


def test_cse_value_numbering_invalidates_when_an_operand_is_rewritten():
    # rewriting A(10) between the two A+B claims bumps its version, so c2 is NOT a CSE
    # of c1 (the values differ) -- it is no longer priced as the cheap copy.
    valid = optimize(_cse((10, 11)), AVX, COOL)  # CSE holds
    invalidated = optimize(_cse((10, 11), rewrite_a=True), AVX, COOL)  # A rewritten between
    assert _c2(invalidated) > _c2(valid)  # CSE correctly withdrawn


def test_cse_is_consistent_across_the_tropical_and_rcsp_rails():
    from bcir.kbcir.rcsp import optimize_constrained

    m = _cse((10, 11))
    assert optimize_constrained(m, AVX, COOL).score == optimize(m, AVX, COOL).score


def test_cse_leaves_the_single_claim_pin_intact():
    assert optimize(vector_add(1024), AVX, COOL).score == 7808  # no duplicate -> no-op


# --- G1 / S1-A (assessment row 7): the CSE identity is complete and the exclusions categorical --


def _dup_module(second: dict, first: dict | None = None, resource_domain=None):
    """c1 = A(10)+B(11)->D(12); c2 = A+B->E(13) with `second`'s field overrides (a would-be
    duplicate of c1); `first` overrides c1 the same way, `resource_domain` re-homes A and B."""
    from bcir.model import Domain

    m = Module(name="cse_neg")
    for rid in (10, 11, 12, 13):
        m.add_resource(
            Resource(
                rid=rid,
                shape=(1 << 16,),
                domain=resource_domain if (resource_domain and rid in (10, 11)) else Domain.RAM,
            )
        )
    base = dict(
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=1 << 16,
        op="vector.add",
    )
    c1 = Claim(id=1, rd=(10, 11), wr=(12,), **{**base, **(first or {})})
    c2 = Claim(id=2, rd=(10, 11), wr=(13,), **{**base, **second})
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c1, c2]))
    return m


def _credited(m) -> bool:
    """Whether claim 2 took the CSE copy credit: its fused candidates are cheaper than its own
    offer (`candidates_for`, the price it pays with no earlier occurrence)."""
    from bcir.kbcir.realize import candidates_for, fused_candidates

    c2 = m.phases[0].claims[1]
    own = candidates_for(c2, AVX, m.resource(c2.rd[0]))
    return fused_candidates(m, AVX)[2] != own


def test_cse_requires_the_complete_semantic_identity():
    """The old key was the op string and the read versions: a duplicate over a different
    count, offset, stride, lane, opcode or dynamic bound took the copy credit. The value is
    the same only when the whole access pattern and contract are."""
    assert _credited(_dup_module({}))  # the positive control: an exact duplicate is a copy
    for label, diff in {
        "count": {"count": 4096},
        "offset": {"offset": 8},
        "stride": {"stride_class": StrideClass.STRIDED, "stride_k": 2},
        "lane": {"lane": Lane.UX, "stride_class": StrideClass.CACHELINE},
        "opcode": {"opcode": Opcode.SUB},  # the op STRING agrees; the opcode does not
        "dynamic": {"dynamic": True},
    }.items():
        assert not _credited(_dup_module(diff)), label


def test_cse_never_merges_atomic_barriered_volatile_or_effectful_claims():
    """Both claims carry the trait, so their identities agree and only the categorical
    exclusion withholds the credit: an atomic read-modify-write, a barriered or volatile
    access, an effect opcode, an indirect call, a claim with lifetime metadata, a sparse
    gather, a non-default numeric contract, and the oracle-only fields the law rail cannot see."""
    from bcir.model import Lifetime

    for label, trait in {
        "atomic hazard": {"hazard": "atomic"},
        "barriered": {"hazard": "barriered"},
        "volatile": {"hazard": "barriered", "volatile": True},
        "atomic opcode": {"opcode": Opcode.ATOMIC_ADD, "op": "atomic.add", "hazard": "atomic"},
        "barrier opcode": {"opcode": Opcode.BARRIER, "op": "c.fence", "hazard": "barriered"},
        "dispatch": {
            "opcode": Opcode.GEM_DISPATCH,
            "op": "c.call.indirect",
            "callee_sig": "int(int)",
        },
        "lifetime": {"lifetime": Lifetime("free")},
        "gather": {
            "lane": Lane.GGG,
            "stride_class": StrideClass.RANDOM,
            "opcode": Opcode.GGG_LOAD,
            "op": "histogram.gather",
        },
        "immediate": {"imm": (7,)},
        "tolerance": {"tolerance_ulp": 2},
        "quantized": {"quantized_bits": 8},
        "compensated": {"precision": "compensated"},
    }.items():
        assert not _credited(_dup_module(trait, first=trait)), label


def test_cse_never_touches_an_isolated_domain():
    """An MMIO register read is an effect: a claim in, or touching, an isolated domain is never
    a common subexpression, whichever side declares it."""
    from bcir.model import Domain

    assert not _credited(_dup_module({}, resource_domain=Domain.MMIO))
    mmio = {"domain": Domain.MMIO}
    assert not _credited(_dup_module(mmio, first=mmio))


def test_an_ineligible_claim_never_seeds_a_duplicate():
    """The seed must be eligible too: a volatile read of A+B followed by an ordinary A+B is not
    a value already in hand (the volatile read is not elidable and its value is not stable)."""
    assert not _credited(_dup_module({}, first={"hazard": "barriered", "volatile": True}))
    assert _credited(_dup_module({}))  # the same pair with an eligible seed IS a copy


def test_cse_eligibility_and_identity_are_the_explicit_predicates():
    """The two predicates the law rail mirrors (BCIRCostModel.h cseEligible / cseIdentity):
    the identity carries the whole access pattern and the read versions; a rewrite of an
    operand changes it, a bounds or verify contract does not."""
    from bcir.kbcir.realize import cse_eligible, cse_identity

    m = _dup_module({})
    c1, c2 = m.phases[0].claims
    assert cse_eligible(c1, m) and cse_eligible(c2, m)
    assert cse_identity(c1, {}) == cse_identity(c2, {})
    assert cse_identity(c1, {10: 1}) != cse_identity(c2, {})  # A rewritten: a new version
    relaxed = replace(c2, verify="none", bounds="assumed_safe", cost_class="compute")
    assert cse_identity(relaxed, {}) == cse_identity(c1, {})  # contracts are not the value
    assert not cse_eligible(replace(c2, rd=()), m)  # nothing read: nothing to duplicate
