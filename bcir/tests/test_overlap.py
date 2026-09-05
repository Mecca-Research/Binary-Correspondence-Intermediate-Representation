"""K_BCIR <-> GEM coupling tests: the scheduled price M(pi, Theta) over the canonical artifact."""

from dataclasses import replace

from bcir.examples import PROGRAMS, vector_add
from bcir.gem import optimize_scheduled, price_scheduled, price_waves_legacy
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_plan


def _claim(
    cid,
    rd,
    wr,
    n=1024,
    lane=Lane.U,
    sc=StrideClass.UNIT,
    op="vector.add",
    opcode=Opcode.ADD,
    **contract,
):
    return Claim(
        id=cid, opcode=opcode, lane=lane, stride_class=sc, count=n, rd=rd, wr=wr, op=op, **contract
    )


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


def test_parallel_claims_price_at_the_longest_step():
    # The two claims share a read, so the plan's chain discounts claim 2's memory
    # (fusion x0.75). On separate domains they run in parallel: the artifact places
    # the plan's own step costs (the executor's durations), so the makespan is the
    # longest step -- claim 1's undiscounted cost -- and overlap beats the chain.
    m = _module([_claim(1, (10, 11), (12,)), _claim(2, (10, 20), (21,))], (10, 11, 12, 20, 21))
    h = TARGETS["x86_avx512"]  # 8 domains: claims land on different domains
    r = optimize(m, h, Theta.cool())
    p = price_scheduled(m, r, h, Theta.cool())
    by = {s.claim_id: s.cost for s in r.steps}
    assert by[2] < by[1]  # serial pricing gave claim 2 the discount
    assert p.makespan == by[1]  # in parallel, the longest step bounds the phase
    assert p.makespan < p.serial  # ...and overlap still beats the serial chain
    assert p.schedule.slot_of(1).domain != p.schedule.slot_of(2).domain


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


# --- G1 / S1-A: the price reads the canonical artifact; the wave pricer is the witness ------


def _tail_raw():
    """A wave claim writes @12; a GGG gather reads it -- RAW across the streams (row 6)."""
    gather = _claim(
        2,
        (12,),
        (21,),
        lane=Lane.GGG,
        sc=StrideClass.RANDOM,
        op="histogram.gather",
        opcode=Opcode.GGG_LOAD,
        hazard="atomic",
    )
    return _module([_claim(1, (10, 11), (12,), hazard="atomic"), gather], (10, 11, 12, 21))


def test_a_dependent_tail_is_priced_serialized_and_the_legacy_pricer_is_the_witness():
    # The canonical price places the gather after its producer (gain 0). The retired wave
    # pricer ran the tail as a free chain alongside the waves and priced the overlap of a
    # value with its own producer -- kept as `price_waves_legacy`, read by nothing else.
    m = _tail_raw()
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool())
    p = price_scheduled(m, r, h, Theta.cool())
    assert p.makespan == p.serial == r.score and p.overlap_gain == 0
    assert p.schedule.slot_of(2).start == p.schedule.slot_of(1).finish
    legacy = price_waves_legacy(m, r, h, Theta.cool())
    assert legacy.serial == r.score and legacy.makespan < legacy.serial  # the defect, witnessed


def test_a_fence_is_priced_serialized():
    # A barriered claim is overlapped by nothing, data hazard or not; the legacy pricer
    # ignored every fence (it binned the two claims round-robin and took the max).
    m = _module(
        [_claim(1, (10, 11), (12,), hazard="barriered"), _claim(2, (20, 21), (22,))],
        (10, 11, 12, 20, 21, 22),
    )
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool())
    p = price_scheduled(m, r, h, Theta.cool())
    assert p.makespan == p.serial and p.overlap_gain == 0
    assert price_waves_legacy(m, r, h, Theta.cool()).makespan < p.serial


def test_the_price_carries_the_artifact_in_both_modes():
    # `schedule` IS the artifact the makespan was read from: phase-barriered by default,
    # token-pipelined on request -- the second phase's independent claim overlaps the first.
    m = Module(name="modes")
    for rid in (10, 11, 20, 21):
        m.add_resource(Resource(rid=rid, shape=(1024,)))
    m.add_phase(Phase(phase_id=0, claims=[_claim(1, (10,), (11,))]))
    m.add_phase(Phase(phase_id=1, deps=(0,), claims=[_claim(2, (20,), (21,))]))
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool())
    barriered = price_scheduled(m, r, h, Theta.cool())
    pipelined = price_scheduled(m, r, h, Theta.cool(), mode="tokens")
    assert barriered.schedule.mode == "eft" and pipelined.schedule.mode == "tokens"
    assert barriered.makespan == barriered.serial  # phases compose serially
    assert pipelined.makespan < pipelined.serial == barriered.serial  # tokens overlap them
    assert pipelined.schedule.slot_of(2).start == 0


def test_optimize_scheduled_returns_the_canonical_price():
    # The sweep's reported price is the same artifact `price_scheduled` reads on its plan.
    h = TARGETS["x86_avx512"]
    for name, build in PROGRAMS.items():
        r, p = optimize_scheduled(build(), h, Theta.cool())
        q = price_scheduled(build(), r, h, Theta.cool())
        assert (p.makespan, p.serial) == (q.makespan, q.serial), name
        assert p.schedule.slots == q.schedule.slots, name


def test_a_judge_policy_reprices_the_plan_under_its_own_weights():
    # Pricing a PERF plan under the ENERGY judge re-derives every step through the planner's
    # own edge predicate under the judge's weights: the serial bound is the judge's chain,
    # the makespan its placement, and the R9 identity holds under the judge too.
    from bcir.gem.overlap import _serial_result
    from bcir.kbcir.weights import ENERGY, PERF

    m = PROGRAMS["fused_chain"]()
    h = TARGETS["x86_avx512"]
    r = optimize(m, h, Theta.cool(), PERF)
    judged = price_scheduled(m, r, h, Theta.cool(), ENERGY)
    assert judged.serial == _serial_result(m, r.by_claim(), h, Theta.cool(), ENERGY).score
    assert judged.serial != r.score  # the judge's weights, not the plan's
    assert judged.makespan + judged.overlap_gain == judged.serial
    assert 0 <= judged.makespan <= judged.serial
    own = price_scheduled(m, r, h, Theta.cool(), PERF)
    assert own.serial == r.score  # under the plan's own scope: the plan's own steps
