"""K_BCIR <-> GEM coupling tests: the (max,+) wave-overlap price M(pi, Theta)."""

from dataclasses import replace

from bcir.examples import PROGRAMS, vector_add
from bcir.gem import optimize_scheduled, price_scheduled
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_plan


def _claim(
    cid, rd, wr, n=1024, lane=Lane.U, sc=StrideClass.UNIT, op="vector.add", opcode=Opcode.ADD
):
    return Claim(id=cid, opcode=opcode, lane=lane, stride_class=sc, count=n, rd=rd, wr=wr, op=op)


def _module(claims, rids):
    m = Module(name="overlap")
    for rid in rids:
        m.add_resource(Resource(rid=rid, shape=(1024,)))
    m.add_phase(Phase(phase_id=0, claims=list(claims)))
    return m


def test_single_claim_is_the_degenerate_case():
    # One claim: no overlap exists, so M(pi,Theta) == the serial Sigma score.
    m = vector_add(1024)
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool())
    p = price_scheduled(m, r, h, Theta.cool())
    assert p.makespan == r.score == 7808
    assert p.overlap_gain == 0


def test_independent_claims_overlap_to_the_max():
    # Two independent claims co-execute in one wave on distinct domains:
    # makespan == max(step costs), gain == min(step costs).
    m = _module([_claim(1, (10, 11), (12,)), _claim(2, (20, 21), (22,))], (10, 11, 12, 20, 21, 22))
    h = TARGETS["x86_avx512"]  # 8 affinity domains
    r = optimize(m, h, Theta.cool())
    p = price_scheduled(m, r, h, Theta.cool())
    costs = sorted(s.cost for s in r.steps)
    assert p.serial == sum(costs)
    assert p.makespan == max(costs)
    assert p.overlap_gain == min(costs)


def test_conflicting_claims_serialize_across_waves():
    # B reads what A writes (RAW): successive waves -- series composition, no gain.
    m = _module([_claim(1, (10, 11), (12,)), _claim(2, (12, 20), (21,))], (10, 11, 12, 20, 21))
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool())
    p = price_scheduled(m, r, h, Theta.cool())
    assert p.makespan == p.serial
    assert p.overlap_gain == 0


def test_ggg_tail_overlaps_the_main_stream():
    # A unit-stride claim plus a decoupled gather: phase cost = max(main, tail).
    gather = _claim(
        2,
        (20,),
        (21,),
        lane=Lane.GGG,
        sc=StrideClass.RANDOM,
        op="histogram.scatter",
        opcode=Opcode.GGG_LOAD,
    )
    m = _module([_claim(1, (10, 11), (12,)), gather], (10, 11, 12, 20, 21))
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool())
    p = price_scheduled(m, r, h, Theta.cool())
    by = {s.claim_id: s.cost for s in r.steps}
    assert p.makespan == max(by[1], by[2])
    assert p.overlap_gain == min(by[1], by[2])


def test_one_domain_serializes_bins_back_to_the_serial_price():
    # With a single affinity domain the wave degenerates to one bin chained in
    # textual order -- the schedule price collapses to the serial price exactly.
    m = _module([_claim(1, (10, 11), (12,)), _claim(2, (10, 20), (21,))], (10, 11, 12, 20, 21))
    h = replace(TargetProfile.x86_avx512(), affinity_domains=1)
    r = optimize(m, h, Theta.cool())
    p = price_scheduled(m, r, h, Theta.cool())
    assert p.makespan == p.serial


def test_parallel_bins_do_not_inherit_textual_fusion():
    # The two claims share a read, so the *serial* chain discounts claim 2's
    # memory (fusion x0.75). On separate domains they run in parallel: the bin
    # re-price must drop the phantom discount, so the makespan exceeds claim 2's
    # serially-discounted step cost.
    m = _module([_claim(1, (10, 11), (12,)), _claim(2, (10, 20), (21,))], (10, 11, 12, 20, 21))
    h = TARGETS["x86_avx512"]  # 8 domains: claims land in different bins
    r = optimize(m, h, Theta.cool())
    p = price_scheduled(m, r, h, Theta.cool())
    by = {s.claim_id: s.cost for s in r.steps}
    assert by[2] < by[1]  # serial pricing gave claim 2 the discount
    assert p.makespan == by[1]  # in parallel, both cost the undiscounted max
    assert p.makespan < p.serial  # ...and overlap still beats the serial chain


def test_optimize_scheduled_is_stable_on_the_canonical_corpus():
    # The one-sweep re-selection must not churn plans the serial optimum already
    # got right; the result stays R8/R9-verifiable with makespan <= serial.
    for name, build in PROGRAMS.items():
        m = build()
        h = TARGETS["x86_avx512"]
        base = optimize(m, h, Theta.cool())
        r, p = optimize_scheduled(m, h, Theta.cool())
        assert {c: x.width for c, x in r.by_claim().items()} == {
            c: x.width for c, x in base.by_claim().items()
        }, name
        assert verify_plan(m, r) == []
        assert p.makespan <= p.serial
