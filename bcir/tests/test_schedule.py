"""Duration-aware GEM scheduling tests: EFT/LPT waves, tokens, locality, the knee."""

from dataclasses import replace

from bcir.examples import vector_add
from bcir.gem import (
    TAIL_STREAM,
    bandwidth_knee,
    durations_from,
    execute_tokens,
    hydrate_pipelined,
    schedule_eft,
)
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_pack

AVX = TargetProfile.x86_avx512()


def _claim(cid, rd, wr, lane=Lane.U, sc=StrideClass.UNIT, cost_class="bandwidth",
           op="vector.add", opcode=Opcode.ADD):
    return Claim(id=cid, opcode=opcode, lane=lane, stride_class=sc, count=1024,
                 rd=rd, wr=wr, op=op, cost_class=cost_class)


def _module(phases, rids):
    m = Module(name="sched")
    for rid in rids:
        m.add_resource(Resource(rid=rid, shape=(1024,)))
    for pid, deps, claims in phases:
        m.add_phase(Phase(phase_id=pid, deps=deps, claims=list(claims)))
    return m


def test_lpt_beats_id_order_dispatch():
    # Three independent claims, durations 5/6/10, two domains. Longest-first
    # packs to makespan 11; naive id-order dispatch (5->d0, 6->d1, 10->d0)
    # would finish at 15. Duration-awareness is the whole point.
    m = _module([(0, (), [_claim(1, (10,), (11,)), _claim(2, (20,), (21,)),
                          _claim(3, (30,), (31,))])],
                (10, 11, 20, 21, 30, 31))
    h = replace(AVX, affinity_domains=2, mem_channels=2)
    s = schedule_eft(m, {1: 5, 2: 6, 3: 10}, h)
    assert s.makespan == 11
    assert s.slot_of(3).start == 0          # longest claim dispatches first


def test_conflicts_serialize_by_producer_id():
    # RAW: claim 2 reads what claim 1 writes -- it must start at 1's finish.
    m = _module([(0, (), [_claim(1, (10,), (11,)), _claim(2, (11,), (12,))])],
                (10, 11, 12))
    s = schedule_eft(m, {1: 7, 2: 3}, AVX)
    assert s.slot_of(2).start == s.slot_of(1).finish == 7
    assert s.makespan == 10


def test_bandwidth_knee_clamps_streaming_parallelism():
    # Four bandwidth-bound claims, 8 domains but a 2-channel memory system:
    # only 2 stream concurrently (makespan 8, not 4). Compute-bound claims
    # are not clamped (makespan 4).
    claims = [_claim(i, (10 * i,), (10 * i + 1,)) for i in (1, 2, 3, 4)]
    rids = [r for c in claims for r in (*c.rd, *c.wr)]
    m = _module([(0, (), claims)], rids)
    h = replace(AVX, affinity_domains=8, mem_channels=2)
    assert bandwidth_knee(h) == 2
    s = schedule_eft(m, {c.id: 4 for c in claims}, h)
    assert s.makespan == 8

    compute = [replace_cost(c) for c in claims]
    m2 = _module([(0, (), compute)], rids)
    s2 = schedule_eft(m2, {c.id: 4 for c in compute}, h)
    assert s2.makespan == 4


def replace_cost(c: Claim) -> Claim:
    return Claim(id=c.id, opcode=c.opcode, lane=c.lane, stride_class=c.stride_class,
                 count=c.count, rd=c.rd, wr=c.wr, op=c.op, cost_class="compute")


def test_locality_ties_prefer_the_warm_domain():
    # Claims 1 and 2 fill domains 0 and 1; claim 3 ties on finish time but
    # shares its operands with claim 2 -- locality places it on the warm domain
    # (cache reuse), where blind tie-breaking would pick domain 0.
    m = _module([(0, (), [_claim(1, (20, 21), (22,)), _claim(2, (10, 11), (12,)),
                          _claim(3, (10, 11), (13,))])],
                (10, 11, 12, 13, 20, 21, 22))
    h = replace(AVX, affinity_domains=2, mem_channels=2)
    s = schedule_eft(m, {1: 4, 2: 4, 3: 2}, h)
    assert s.affinity[3] == s.affinity[2] == 1
    blind = schedule_eft(m, {1: 4, 2: 4, 3: 2}, h, locality=False)
    assert blind.affinity[3] == 0  # without locality, the tie falls to domain 0


def test_ggg_tail_runs_on_its_own_stream():
    gather = _claim(2, (20,), (21,), lane=Lane.GGG, sc=StrideClass.RANDOM,
                    op="histogram.scatter", opcode=Opcode.GGG_LOAD)
    m = _module([(0, (), [_claim(1, (10,), (11,)), gather])], (10, 11, 20, 21))
    s = schedule_eft(m, {1: 6, 2: 9}, AVX)
    assert s.affinity[2] == TAIL_STREAM
    assert s.makespan == 9  # tail overlaps the wave stream: max(6, 9)


def test_tokens_pipeline_independent_phases():
    # Phase 1 depends on phase 0 in the DAG, but its claim conflicts with
    # nothing: with phase barriers (EFT) the makespan is serial; under the
    # token DAG it overlaps -- pipelined phases fall out of the awaits.
    m = _module([(0, (), [_claim(1, (10,), (11,))]),
                 (1, (0,), [_claim(2, (20,), (21,))])],
                (10, 11, 20, 21))
    durations = {1: 8, 2: 5}
    barrier = schedule_eft(m, durations, AVX)
    tokens = execute_tokens(m, durations, AVX)
    assert barrier.makespan == 13          # phases serialize at the barrier
    assert tokens.makespan == 8            # independent claims overlap: max(8, 5)


def test_tokens_respect_await_chains():
    # A cross-phase RAW conflict forks an await edge: tokens serialize exactly
    # like the barrier schedule (the degenerate case).
    m = _module([(0, (), [_claim(1, (10,), (11,))]),
                 (1, (0,), [_claim(2, (11,), (12,))])],
                (10, 11, 12))
    durations = {1: 8, 2: 5}
    assert execute_tokens(m, durations, AVX).makespan == 13
    assert schedule_eft(m, durations, AVX).makespan == 13


def test_durations_come_from_the_kbcir_plan():
    m = vector_add(1024)
    r = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    d = durations_from(r)
    assert d == {1000: 7808}
    s = schedule_eft(m, d, TARGETS["x86_avx512"])
    assert s.makespan == 7808  # single claim: the degenerate case again


def test_pipelined_hydration_emits_double_buffer_contracts():
    # Two phases: the v2 pack carries pipeline_depth=2 and a double-buffer
    # prefetch for the next phase's reads; the pack stays verifier-clean.
    m = _module([(0, (), [_claim(1, (10,), (11,))]),
                 (1, (0,), [_claim(2, (20,), (21,))])],
                (10, 11, 20, 21))
    r = optimize(m, AVX, Theta.cool())
    pack = hydrate_pipelined(m, r, depth=2)
    assert pack.pipeline_depth == 2
    db = [pf for pf in pack.prefetches if pf.pattern == "double_buffer"]
    assert len(db) == 1 and db[0].buffers == 2 and db[0].targets == (20,)
    assert verify_pack(m, pack) == []


def test_invalid_pipeline_contracts_are_R10():
    m = vector_add(1024)
    r = optimize(m, AVX, Theta.cool())
    pack = hydrate_pipelined(m, r, depth=2)
    pack.pipeline_depth = 0
    laws = {d.law for d in verify_pack(m, pack)}
    assert "R10" in laws


def test_mlir_schedule_eft_constants_are_in_sync():
    """Pins the constants -bcir-schedule-eft annotates in mlir/test/passes/schedule_eft.mlir."""
    from bcir.gem.schedule import schedule_eft, durations_from, bandwidth_knee
    from bcir.kbcir import TARGETS, optimize
    from bcir.kbcir.cost import Theta
    from bcir.model import Module, Claim, Domain, Lane, Opcode, Resource, StrideClass, Phase
    AVX = TARGETS["x86_avx512"]
    m = Module(name="s")
    for r in (10, 11, 12, 13, 14):
        m.add_resource(Resource(rid=r, domain=Domain.RAM, shape=(1024,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=1024,
              rd=(10, 11), wr=(12,), op="vector.add", domain=Domain.RAM, cost_class="compute"),
        Claim(id=2, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=1024,
              rd=(13, 11), wr=(14,), op="vector.add", domain=Domain.RAM, cost_class="compute")]))
    sch = schedule_eft(m, durations_from(optimize(m, AVX, Theta.cool())), AVX)
    assert bandwidth_knee(AVX) == 4 and sch.makespan == 7808
    assert sch.slot_of(1).domain == 0 and sch.slot_of(1).finish == 7808
    assert sch.slot_of(2).domain == 1 and sch.slot_of(2).finish == 5888
