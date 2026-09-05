"""Duration-aware GEM scheduling tests: EFT/LPT waves, tokens, locality, the knee."""

from dataclasses import replace

from bcir.examples import PROGRAMS, vector_add
from bcir.gem import (
    TAIL_STREAM,
    bandwidth_knee,
    durations_from,
    execute_tokens,
    hydrate_pipelined,
    price_scheduled,
    schedule_eft,
    schedule_plan,
)
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_pack

AVX = TargetProfile.x86_avx512()


def _claim(
    cid,
    rd,
    wr,
    lane=Lane.U,
    sc=StrideClass.UNIT,
    cost_class="bandwidth",
    op="vector.add",
    opcode=Opcode.ADD,
    **contract,
):
    return Claim(
        id=cid,
        opcode=opcode,
        lane=lane,
        stride_class=sc,
        count=1024,
        rd=rd,
        wr=wr,
        op=op,
        cost_class=cost_class,
        **contract,
    )


def _gather(cid, rd, wr, **contract):
    return _claim(
        cid,
        rd,
        wr,
        lane=Lane.GGG,
        sc=StrideClass.RANDOM,
        op="histogram.gather",
        opcode=Opcode.GGG_LOAD,
        **contract,
    )


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
    m = _module(
        [(0, (), [_claim(1, (10,), (11,)), _claim(2, (20,), (21,)), _claim(3, (30,), (31,))])],
        (10, 11, 20, 21, 30, 31),
    )
    h = replace(AVX, affinity_domains=2, mem_channels=2)
    s = schedule_eft(m, {1: 5, 2: 6, 3: 10}, h)
    assert s.makespan == 11
    assert s.slot_of(3).start == 0  # longest claim dispatches first


def test_conflicts_serialize_by_producer_id():
    # RAW: claim 2 reads what claim 1 writes -- it must start at 1's finish.
    m = _module([(0, (), [_claim(1, (10,), (11,)), _claim(2, (11,), (12,))])], (10, 11, 12))
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
    return Claim(
        id=c.id,
        opcode=c.opcode,
        lane=c.lane,
        stride_class=c.stride_class,
        count=c.count,
        rd=c.rd,
        wr=c.wr,
        op=c.op,
        cost_class="compute",
    )


def test_locality_ties_prefer_the_warm_domain():
    # Claims 1 and 2 fill domains 0 and 1; claim 3 ties on finish time but
    # shares its operands with claim 2 -- locality places it on the warm domain
    # (cache reuse), where blind tie-breaking would pick domain 0.
    m = _module(
        [
            (
                0,
                (),
                [
                    _claim(1, (20, 21), (22,)),
                    _claim(2, (10, 11), (12,)),
                    _claim(3, (10, 11), (13,)),
                ],
            )
        ],
        (10, 11, 12, 13, 20, 21, 22),
    )
    h = replace(AVX, affinity_domains=2, mem_channels=2)
    s = schedule_eft(m, {1: 4, 2: 4, 3: 2}, h)
    assert s.affinity[3] == s.affinity[2] == 1
    blind = schedule_eft(m, {1: 4, 2: 4, 3: 2}, h, locality=False)
    assert blind.affinity[3] == 0  # without locality, the tie falls to domain 0


def test_ggg_tail_runs_on_its_own_stream():
    gather = _claim(
        2,
        (20,),
        (21,),
        lane=Lane.GGG,
        sc=StrideClass.RANDOM,
        op="histogram.scatter",
        opcode=Opcode.GGG_LOAD,
    )
    m = _module([(0, (), [_claim(1, (10,), (11,)), gather])], (10, 11, 20, 21))
    s = schedule_eft(m, {1: 6, 2: 9}, AVX)
    assert s.affinity[2] == TAIL_STREAM
    assert s.makespan == 9  # tail overlaps the wave stream: max(6, 9)


# --- G1 / S1-A: the hazard DAG is built over every claim BEFORE the stream split -------------


def test_a_dependent_tail_claim_waits_for_its_producer():
    """RAW wave -> tail: the gather reads @11, which claim 1 writes. The parent build split the
    streams first and ran the tail as a chain from the phase start, so the gather started at 0
    with its input unwritten (assessment row 6, the negative witness). Both modes place the
    same slots here -- one artifact, two edge sets that coincide on one phase."""
    m = _module([(0, (), [_claim(1, (10,), (11,)), _gather(2, (11,), (12,))])], (10, 11, 12))
    s = schedule_eft(m, {1: 6, 2: 9}, AVX)
    assert s.slot_of(2).domain == TAIL_STREAM  # still the decoupled stream ...
    assert s.slot_of(2).start == s.slot_of(1).finish == 6  # ... but after its producer
    assert s.makespan == 15
    assert execute_tokens(m, {1: 6, 2: 9}, AVX).slots == s.slots


def test_a_wave_claim_waits_for_a_tail_producer_and_a_tail_writer_for_its_reader():
    """The edges hold in both directions: RAW tail -> wave, and WAR wave -> tail."""
    raw = _module([(0, (), [_gather(1, (10,), (11,)), _claim(2, (11,), (12,))])], (10, 11, 12))
    s = schedule_eft(raw, {1: 9, 2: 6}, AVX)
    assert s.slot_of(2).start == s.slot_of(1).finish == 9 and s.makespan == 15
    war = _module([(0, (), [_claim(1, (11,), (12,)), _gather(2, (10,), (11,))])], (10, 11, 12))
    w = schedule_eft(war, {1: 6, 2: 9}, AVX)
    assert w.slot_of(2).start == w.slot_of(1).finish == 6 and w.makespan == 15


def test_a_fence_is_overlapped_by_nothing():
    """Claim 2 is barriered (then volatile) and shares no resource with 1 or 3: everything before
    it finishes first and everything after it starts after it. The parent build placed all
    three at 0 on separate domains."""
    for fence in ({"hazard": "barriered"}, {"hazard": "barriered", "volatile": True}):
        m = _module(
            [
                (
                    0,
                    (),
                    [
                        _claim(1, (10,), (11,)),
                        _claim(2, (20,), (21,), **fence),
                        _claim(3, (30,), (31,)),
                    ],
                )
            ],
            (10, 11, 20, 21, 30, 31),
        )
        s = schedule_eft(m, {1: 4, 2: 5, 3: 6}, AVX)
        assert s.slot_of(2).start == s.slot_of(1).finish == 4, fence
        assert s.slot_of(3).start == s.slot_of(2).finish == 9, fence
        assert s.makespan == 15, fence


def test_tokens_honor_a_fence_across_phases():
    """The phase-1 claim is data-independent of the phase-0 fence: under tokens it used to
    start at 0 (the pipelining of test_tokens_pipeline_independent_phases); a fence forbids
    exactly that overlap, and the await list says why."""
    from bcir.gem import async_plan

    m = _module(
        [
            (0, (), [_claim(1, (10,), (11,), hazard="barriered")]),
            (1, (0,), [_claim(2, (20,), (21,))]),
        ],
        (10, 11, 20, 21),
    )
    assert async_plan(m).awaits == {1: [], 2: [1]}
    assert execute_tokens(m, {1: 8, 2: 5}, AVX).makespan == 13
    assert schedule_eft(m, {1: 8, 2: 5}, AVX).makespan == 13


def test_a_zero_cost_step_is_a_point_on_the_timeline():
    """Durations are exactly the plan's step costs: a zero-cost step is a zero-length slot,
    so a serialized chain never prices above its own serial sum (the R9 bound)."""
    from types import SimpleNamespace

    fake = SimpleNamespace(
        steps=[SimpleNamespace(claim_id=1, cost=0), SimpleNamespace(claim_id=2, cost=5)]
    )
    assert durations_from(fake) == {1: 0, 2: 5}
    m = _module([(0, (), [_claim(1, (10,), (11,)), _claim(2, (11,), (12,))])], (10, 11, 12))
    s = schedule_eft(m, durations_from(fake), AVX)
    assert (s.slot_of(1).start, s.slot_of(1).finish) == (0, 0)
    assert (s.slot_of(2).start, s.slot_of(2).finish) == (0, 5) and s.makespan == 5


def test_schedule_plan_is_the_one_artifact_the_price_and_the_executors_read():
    """G1's gate: token execution and the priced schedule agree -- identical slot assignment.
    `schedule_plan` is the artifact; `price_scheduled(...).schedule` IS it, and each executor
    returns the same slots over the plan's own step costs, on every corpus program."""
    theta = Theta.cool()
    for name, build in PROGRAMS.items():
        m = build()
        r = optimize(m, AVX, theta)
        for mode, executor in (("eft", schedule_eft), ("tokens", execute_tokens)):
            artifact = schedule_plan(m, r, AVX, mode=mode)
            assert (
                artifact.mode == mode
                and artifact.slots == executor(m, durations_from(r), AVX).slots
            ), (name, mode)
            priced = price_scheduled(m, r, AVX, theta, mode=mode)
            assert (
                priced.schedule.slots == artifact.slots and priced.makespan == artifact.makespan
            ), (name, mode)
            assert priced.serial == r.score == sum(durations_from(r).values()), name
            assert 0 <= priced.makespan <= priced.serial, name


def test_schedule_plan_refuses_an_unknown_mode():
    m = vector_add(1024)
    r = optimize(m, AVX, Theta.cool())
    try:
        schedule_plan(m, r, AVX, mode="waves")
    except ValueError as exc:
        assert "waves" in str(exc)
    else:
        raise AssertionError("an unknown schedule mode must be refused, not defaulted")


def test_mlir_schedule_hazards_are_in_sync():
    """Pins the constants -bcir-schedule-eft / -bcir-async / -bcir-overlap annotate in
    mlir/test/passes/schedule_hazards.mlir (the module is emitted from this very build): the
    gather waits for its producer on the tail stream, the fence waits for everything before it
    and holds everything after it, and the price is the placement's makespan (gain 0)."""
    from bcir.gem import async_plan
    from bcir.gem.overlap import price_waves_legacy
    from bcir.kbcir.weights import PERF
    from bcir.model import Domain

    def C(i, rd, wr, **kw):
        kw.setdefault("lane", Lane.U)
        kw.setdefault("sc", StrideClass.UNIT)
        kw.setdefault("opcode", Opcode.ADD)
        kw.setdefault("op", "vector.add")
        return _claim(i, rd, wr, cost_class="compute", domain=Domain.RAM, **kw)

    m = Module(name="hazards")
    for r in range(10, 17):
        m.add_resource(Resource(rid=r, domain=Domain.RAM, shape=(1024,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                C(1, (10,), (11,), hazard="atomic"),
                C(
                    2,
                    (11,),
                    (12,),
                    lane=Lane.GGG,
                    sc=StrideClass.RANDOM,
                    opcode=Opcode.GGG_LOAD,
                    op="histogram.gather",
                    hazard="atomic",
                ),
                C(3, (13,), (14,), hazard="barriered"),
                C(4, (15,), (16,)),
            ],
        )
    )
    theta = Theta.cool()
    r = optimize(m, AVX, theta, PERF)
    assert durations_from(r) == {1: 5248, 2: 396288, 3: 5248, 4: 5248}
    assert async_plan(m).awaits == {1: [], 2: [1], 3: [1, 2], 4: [3]}
    expected = [
        (1, 0, 0, 5248),
        (2, TAIL_STREAM, 5248, 401536),
        (3, 0, 401536, 406784),
        (4, 0, 406784, 412032),
    ]
    for sched in (
        schedule_eft(m, durations_from(r), AVX),
        execute_tokens(m, durations_from(r), AVX),
    ):
        assert [(x.claim_id, x.domain, x.start, x.finish) for x in sched.slots] == expected
        assert sched.makespan == 412032 and sched.knee == 4
    priced = price_scheduled(m, r, AVX, theta, PERF)
    assert (priced.makespan, priced.serial, priced.overlap_gain) == (412032, 412032, 0)
    legacy = price_waves_legacy(m, r, AVX, theta, PERF)  # the witness: the parent's price
    assert (legacy.makespan, legacy.overlap_gain) == (396288, 15744)


def test_tokens_pipeline_independent_phases():
    # Phase 1 depends on phase 0 in the DAG, but its claim conflicts with
    # nothing: with phase barriers (EFT) the makespan is serial; under the
    # token DAG it overlaps -- pipelined phases fall out of the awaits.
    m = _module(
        [(0, (), [_claim(1, (10,), (11,))]), (1, (0,), [_claim(2, (20,), (21,))])], (10, 11, 20, 21)
    )
    durations = {1: 8, 2: 5}
    barrier = schedule_eft(m, durations, AVX)
    tokens = execute_tokens(m, durations, AVX)
    assert barrier.makespan == 13  # phases serialize at the barrier
    assert tokens.makespan == 8  # independent claims overlap: max(8, 5)


def test_tokens_respect_await_chains():
    # A cross-phase RAW conflict forks an await edge: tokens serialize exactly
    # like the barrier schedule (the degenerate case).
    m = _module(
        [(0, (), [_claim(1, (10,), (11,))]), (1, (0,), [_claim(2, (11,), (12,))])], (10, 11, 12)
    )
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
    m = _module(
        [(0, (), [_claim(1, (10,), (11,))]), (1, (0,), [_claim(2, (20,), (21,))])], (10, 11, 20, 21)
    )
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
    m.add_phase(
        Phase(
            phase_id=0,
            deps=(),
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=1024,
                    rd=(10, 11),
                    wr=(12,),
                    op="vector.add",
                    domain=Domain.RAM,
                    cost_class="compute",
                ),
                Claim(
                    id=2,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=1024,
                    rd=(13, 11),
                    wr=(14,),
                    op="vector.add",
                    domain=Domain.RAM,
                    cost_class="compute",
                ),
            ],
        )
    )
    sch = schedule_eft(m, durations_from(optimize(m, AVX, Theta.cool())), AVX)
    assert bandwidth_knee(AVX) == 4 and sch.makespan == 7808
    assert sch.slot_of(1).domain == 0 and sch.slot_of(1).finish == 7808
    assert sch.slot_of(2).domain == 1 and sch.slot_of(2).finish == 5888


def test_mlir_async_pipelined_schedule_is_in_sync():
    """Pins the constants -bcir-async annotates in mlir/test/passes/async.mlir: a phase-1
    independent claim overlaps phase 0 (start 0), a dependent one awaits -> starts later."""
    from bcir.gem.schedule import execute_tokens, durations_from
    from bcir.gem.async_tokens import async_plan
    from bcir.kbcir import TARGETS, optimize
    from bcir.kbcir.cost import Theta
    from bcir.model import Module, Claim, Domain, Lane, Opcode, Resource, StrideClass, Phase

    AVX = TARGETS["x86_avx512"]

    def C(i, a, b, c):
        return Claim(
            id=i,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1024,
            rd=(a, b),
            wr=(c,),
            op="vector.add",
            domain=Domain.RAM,
            cost_class="compute",
        )

    m = Module(name="async")
    for r in range(10, 18):
        m.add_resource(Resource(rid=r, domain=Domain.RAM, shape=(1024,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[C(1, 10, 11, 12)]))
    m.add_phase(Phase(phase_id=1, deps=(0,), claims=[C(2, 13, 14, 15), C(3, 12, 16, 17)]))
    assert async_plan(m).awaits == {1: [], 2: [], 3: [1]}
    sch = execute_tokens(m, durations_from(optimize(m, AVX, Theta.cool())), AVX)
    assert sch.makespan == 15616
    assert sch.slot_of(2).start == 0 and sch.slot_of(2).finish == 7808  # overlaps phase 0
    assert sch.slot_of(3).start == 7808  # awaits c1


def test_mlir_power_rail_per_slot_clock_is_in_sync():
    """Pins the constants -bcir-power-rail annotates in mlir/test/passes/power_rail.mlir: a
    per-slot DVFS overlay on the EFT timeline -- both slots are memory-bound, so each downclocks
    over its own interval and the modeled energy saved sums to 3424000."""
    from bcir.gem.schedule import schedule_eft, durations_from, schedule_power_rail
    from bcir.kbcir import TARGETS, optimize
    from bcir.kbcir.cost import Theta
    from bcir.model import Module, Claim, Domain, Lane, Opcode, Resource, StrideClass, Phase

    AVX = TARGETS["x86_avx512"]
    theta = Theta.cool()
    m = Module(name="rail")
    for r in (10, 11, 12, 13, 14):
        m.add_resource(Resource(rid=r, domain=Domain.RAM, shape=(1024,)))
    m.add_phase(
        Phase(
            phase_id=0,
            deps=(),
            claims=[
                Claim(
                    id=1,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=1024,
                    rd=(10, 11),
                    wr=(12,),
                    op="vector.add",
                    domain=Domain.RAM,
                    cost_class="compute",
                ),
                Claim(
                    id=2,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=1024,
                    rd=(13, 11),
                    wr=(14,),
                    op="vector.add",
                    domain=Domain.RAM,
                    cost_class="compute",
                ),
            ],
        )
    )
    r = optimize(m, AVX, theta)
    sch = schedule_eft(m, durations_from(r), AVX)
    rail = schedule_power_rail(sch, r, theta, AVX)
    by_id = {d.claim_id: d for d in rail.decisions}
    assert by_id[1].klass == "memory" and by_id[1].clock_q8 == 192 and by_id[1].finish == 7808
    assert by_id[2].klass == "memory" and by_id[2].clock_q8 == 192 and by_id[2].finish == 5888
    assert rail.downclocked == (1, 2) and rail.energy_saved_milli == 3424000
