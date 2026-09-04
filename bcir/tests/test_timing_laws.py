"""R19 (synchronous-timing legality) + R20 (clock-domain-crossing) as bcir.verify functions over the
OPTIONAL `claim.timing` metadata -- the RTL / synchronous-timing track (§5.11), oracle prototype (step 1).

The governing invariant is NON-DISTURBANCE: a claim with no timing (the default) is entirely unconstrained,
so the existing scalar / C-frontend / optimizer corpus verifies IDENTICALLY with R19/R20 present -- the
additive seam, exactly like R14/R15/R16 are vacuous for the scalar subset and R17 is vacuous without a
declared tolerance. The negative cases mirror the per-law discipline of test_smart_laws.py.
"""

import dataclasses

from bcir.examples import gather_reduce, vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.model import Claim, Domain, Module, Opcode, Phase, Resource, Timing
from bcir.verify import verify_smart_lowering, verify_timing

COOL = Theta.cool()


def _mod(claims):
    r0 = Resource(rid=0, domain=Domain.RAM, elem_bytes=4, shape=(16,))
    r1 = Resource(rid=1, domain=Domain.RAM, elem_bytes=4, shape=(16,))
    return Module(resources={0: r0, 1: r1}, phases=[Phase(phase_id=0, claims=claims)])


# --- NON-DISTURBANCE: the whole existing subset carries no timing -> R19/R20 are a NO-OP -----------


def test_timing_is_vacuous_without_metadata():
    """A claim with `timing is None` (the default for the entire scalar / C-frontend subset) places no
    R19/R20 constraint -- the law is a complete no-op."""
    assert verify_timing(_mod([Claim(id=0, opcode=Opcode.LOAD, rd=(0,), wr=(1,))])) == []


def test_existing_corpus_verifies_identically_with_timing_laws_present():
    """The non-disturbance proof: real optimizer modules (which carry no timing) verify with EXACTLY the
    same diagnostics whether or not the timing laws run -- so wiring R19/R20 into verify_smart_lowering
    cannot perturb any existing plan, score, or verdict."""
    for build in (vector_add, gather_reduce):
        m = build(1 << 16)
        optimize(m, TARGETS["x86_avx512"], COOL)  # a real plan exists
        assert verify_timing(m) == []  # no timing -> no R19/R20
        assert [d.law for d in verify_smart_lowering(m) if d.law in ("R19", "R20")] == []


# --- R19: synchronous-timing legality --------------------------------------------------------------


def test_r19_a_well_formed_timing_block_is_legal():
    tm = Timing(
        clock_domain="core0",
        sync_type="synchronous",
        clock_frequency_mhz=3200,
        latency_cycles=12,
        setup_hold_margin=2,
        critical_path=True,
    )
    assert verify_timing(_mod([Claim(id=0, opcode=Opcode.LOAD, rd=(0,), wr=(1,), timing=tm)])) == []


def test_r19_a_synchronous_claim_needs_a_clock():
    tm = Timing(sync_type="synchronous", clock_frequency_mhz=0)
    d = verify_timing(_mod([Claim(id=0, opcode=Opcode.LOAD, rd=(0,), wr=(1,), timing=tm)]))
    assert [x.law for x in d] == ["R19"] and "clock" in d[0].message


def test_r19_setup_hold_margin_cannot_exceed_the_stage_latency():
    tm = Timing(
        sync_type="synchronous", clock_frequency_mhz=1000, latency_cycles=4, setup_hold_margin=9
    )
    d = verify_timing(_mod([Claim(id=0, opcode=Opcode.LOAD, rd=(0,), wr=(1,), timing=tm)]))
    assert any(x.law == "R19" and "setup_hold_margin" in x.message for x in d)


def test_r19_rejects_an_unknown_sync_type_and_negative_cycles():
    tm = Timing(sync_type="bogus", latency_cycles=-1)
    laws = [
        x.law
        for x in verify_timing(_mod([Claim(id=0, opcode=Opcode.LOAD, rd=(0,), wr=(1,), timing=tm)]))
    ]
    assert laws.count("R19") == 2  # unknown sync_type + negative latency


# --- R20: clock-domain-crossing --------------------------------------------------------------------


def _cdc(consumer_timing):
    """A producer in clock domain core0 writes RID 1; the consumer reads RID 1 in core1."""
    producer = Claim(
        id=0, opcode=Opcode.STORE, rd=(0,), wr=(1,), timing=Timing(clock_domain="core0")
    )
    consumer = Claim(id=1, opcode=Opcode.LOAD, rd=(1,), wr=(0,), timing=consumer_timing)
    return verify_timing(_mod([producer, consumer]))


def test_r20_an_unguarded_clock_domain_crossing_is_rejected():
    d = _cdc(Timing(clock_domain="core1", sync_type="synchronous", clock_frequency_mhz=2000))
    assert [x.law for x in d] == ["R20"] and "clock-domain crossing" in d[0].message


def test_r20_is_cleared_by_a_mixed_sync_type():
    assert _cdc(Timing(clock_domain="core1", sync_type="mixed")) == []


def test_r20_is_cleared_by_a_barriered_synchronizer():
    producer = Claim(
        id=0, opcode=Opcode.STORE, rd=(0,), wr=(1,), timing=Timing(clock_domain="core0")
    )
    consumer = Claim(
        id=1,
        opcode=Opcode.LOAD,
        rd=(1,),
        wr=(0,),
        hazard="barriered",
        timing=Timing(clock_domain="core1", sync_type="synchronous", clock_frequency_mhz=2000),
    )
    assert verify_timing(_mod([producer, consumer])) == []


def test_r20_same_clock_domain_is_not_a_crossing():
    producer = Claim(
        id=0, opcode=Opcode.STORE, rd=(0,), wr=(1,), timing=Timing(clock_domain="core0")
    )
    consumer = Claim(
        id=1,
        opcode=Opcode.LOAD,
        rd=(1,),
        wr=(0,),
        timing=Timing(clock_domain="core0", sync_type="synchronous", clock_frequency_mhz=2000),
    )
    assert verify_timing(_mod([producer, consumer])) == []
