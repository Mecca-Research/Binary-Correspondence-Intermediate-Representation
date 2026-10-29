"""ExecutionPlanV1 -- the plan as bytes (GEM+ roadmap G11, staged plan S1-C).

Before this slice the canonical plan -- G1's schedule artifact over a K_BCIR realization --
was Python objects: nothing could round-trip it, the C twin and a resident executor could not
read it, and a stale or malformed plan had no bytes to be refused by. These tests pin the four
G11 gates and the laws around them:

    plan.abi.roundtrip      Python encode -> C decode -> Python re-encode, byte-identical
    plan.readers.agree      pricing, both executors, static memory and the StreamPack lowering
                            read the plan and reproduce their in-memory traces
    stale vector refused    a plan minted under an older vector, and a pack older than its
                            plan, are refused on both rails
    malformed plan refused  truncated, duplicated, out-of-order and unknown-claim records (and
                            the rest of the wire laws) are refused before publication

The C-rail tests build `runtime/c/test_execution_plan.c` and skip without a compiler (the
quick tier hides one on purpose; the c-runtime tier exposes it).
"""

from __future__ import annotations

from dataclasses import replace
import os
import random
import struct
import subprocess
import tempfile
import time

from bcir.abi import AbiError, decode, decode_plan, encode, encode_plan
from bcir.abi.execution_plan_abi import PLAN_HEADER_SIZE, PLAN_MAGIC, TAIL_STREAM_WIRE
from bcir.examples import PROGRAMS
from bcir.gem.execution_plan import (
    TAIL_STREAM,
    ExecutionPlan,
    Lifetime,
    MovementEdge,
    PlanStep,
    plan_from_realization,
    realization_of,
    schedule_of,
)
from bcir.gem.overlap import price_scheduled
from bcir.gem.schedule import durations_from, execute_tokens, schedule_plan, stream_geometry
from bcir.gem.streampack import Generation, generation_vector, hydrate
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.provenance import hash_module, hash_target
from bcir.kbcir.realize import optimize
from bcir.kbcir.weights import PERF
from bcir.model import Lane
from bcir.tests.plan_fixtures import (
    audit_fixture,
    build_harness,
    c_refuses,
    c_roundtrip,
    compiler,
    malformed_plan_bytes,
    malformed_variants,
    out_of_order_variant,
    parse_c_dump,
    python_refuses,
    raw_encode,
    reseal,
    run_harness,
    stale_cases,
    stale_fixtures,
    static_memory_lifetimes_agree,
    v3_variants,
    wire_refuses,
)
from bcir.verify import verify_execution_plan, verify_pack

AVX, COOL = TargetProfile.x86_avx512(), Theta.cool()

#: The host's C compiler, or None: the C-rail witnesses return early without one.
_CC = compiler()


def _corpus():
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        yield name, module, AVX, COOL, optimize(module, AVX, COOL)
    module, target, theta, result = audit_fixture()
    yield "audit.kbcir-streampack.1", module, target, theta, result


def _slots(sched) -> dict:
    return {s.claim_id: (s.domain, s.start, s.finish) for s in sched.slots}


# --- the bytes ---------------------------------------------------------------------------------


def test_the_plan_round_trips_byte_identically_over_the_corpus():
    count = 0
    for name, module, target, _theta, result in _corpus():
        for mode in ("eft", "tokens"):
            plan = plan_from_realization(module, result, target, mode, plan="plan0")
            blob = encode_plan(plan)
            assert blob[:4] == PLAN_MAGIC and len(blob) > PLAN_HEADER_SIZE + 4, name
            again = decode_plan(blob)
            assert again == plan, (name, mode)
            assert encode_plan(again) == blob, (name, mode)
            count += 1
    assert count >= 26, f"corpus degenerated to {count} plans"


def test_the_producer_binds_the_module_the_target_and_the_registry():
    for name, module, target, _theta, result in _corpus():
        plan = plan_from_realization(module, result, target, "eft", plan="p")
        assert plan.module_hash == hash_module(module), name
        assert plan.target_hash == hash_target(target), name
        assert plan.generations == generation_vector(module), name
        assert (plan.streams, plan.knee) == stream_geometry(target), name
        assert plan.serial == result.score == sum(s.cost for s in result.steps), name
        assert [s.claim_id for s in plan.steps] == [s.claim_id for s in result.steps], name
        assert plan.moves == [] and plan.lifetimes == [], name
        assert plan.makespan == schedule_plan(module, result, target, "eft").makespan, name


def test_the_tail_stream_and_a_signed_cost_survive_the_wire():
    tails = 0
    for name, module, target, _theta, result in _corpus():
        plan = plan_from_realization(module, result, target, "eft")
        blob = encode_plan(plan)
        for step in plan.steps:
            if step.stream == TAIL_STREAM:
                tails += 1
                assert struct.pack("<I", TAIL_STREAM_WIRE) in blob, name
        assert decode_plan(blob).slot_map() == plan.slot_map(), name
    assert tails, "no corpus plan places a claim on the decoupled tail; the sentinel is untested"
    plan = ExecutionPlan(
        source_plan="neg",
        mode="tokens",
        streams=2,
        knee=1,
        makespan=0,
        steps=[PlanStep(7, 0, "barrier", Lane.H, 1, -5, TAIL_STREAM, 0, 0)],
        generations=[Generation(1, 0, 0)],
    )
    again = decode_plan(encode_plan(plan))
    assert again == plan and again.steps[0].cost == -5 and again.steps[0].stream == TAIL_STREAM


def test_lifetimes_and_movement_edges_round_trip_and_obey_their_laws():
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft")
    plan.lifetimes = [Lifetime(1, "ram", 0, 4096, 64, 0, 2), Lifetime(2, "ram", 4096, 64, 64, 1, 1)]
    plan.moves = [
        MovementEdge(1, "ram", "hbm", 0, 4096, "pcie", "staged", "flush", 1, 2, 5, 9),
        MovementEdge(2, "ram", "hbm", 0, 64),
    ]
    assert decode_plan(encode_plan(plan)) == plan
    for name, bad in {
        "lifetime.unsorted": replace(plan, lifetimes=list(reversed(plan.lifetimes))),
        "lifetime.reversed": replace(plan, lifetimes=[Lifetime(1, "ram", 0, 64, 64, 3, 1)]),
        "lifetime.misaligned": replace(plan, lifetimes=[Lifetime(1, "ram", 8, 64, 64, 0, 0)]),
        "lifetime.zero": replace(plan, lifetimes=[Lifetime(1, "ram", 0, 0, 64, 0, 0)]),
        "lifetime.nobank": replace(plan, lifetimes=[Lifetime(1, "", 0, 64, 64, 0, 0)]),
        "move.zero": replace(plan, moves=[MovementEdge(1, "ram", "hbm", 0, 0)]),
        "move.kind": replace(plan, moves=[MovementEdge(1, "ram", "hbm", 0, 8, kind="teleport")]),
        "move.coherence": replace(plan, moves=[MovementEdge(1, "ram", "hbm", 0, 8, coherence="x")]),
        "knee": replace(plan, knee=plan.streams + 1),
        "mode": replace(plan, mode="waves"),
    }.items():
        try:
            encode_plan(bad)
        except AbiError:
            continue
        raise AssertionError(f"the encoder published a malformed plan: {name}")


# --- the readers ------------------------------------------------------------------------------


def test_every_reader_reads_the_plan_and_reproduces_its_in_memory_trace():
    """plan.readers.agree: one artifact, four readers, identical traces."""
    checked = 0
    for name, module, target, theta, result in _corpus():
        for mode in ("eft", "tokens"):
            plan = decode_plan(encode_plan(plan_from_realization(module, result, target, mode)))
            # the pricer: its makespan is the plan's and its schedule IS the plan's slots
            priced = price_scheduled(module, result, target, theta, PERF, mode)
            assert priced.makespan == plan.makespan, (name, mode)
            assert _slots(priced.schedule) == plan.slot_map(), (name, mode)
            # the executors: the placement they run is the placement the plan carries, and
            # re-placing the plan's own step costs reproduces it
            placed = schedule_plan(module, result, target, mode)
            assert _slots(placed) == plan.slot_map(), (name, mode)
            if mode == "tokens":
                assert (
                    _slots(execute_tokens(module, durations_from(result), target))
                    == plan.slot_map()
                )
            again = schedule_plan(module, realization_of(plan), target, mode)
            assert _slots(again) == plan.slot_map() and again.makespan == plan.makespan, (
                name,
                mode,
            )
            read = schedule_of(plan)
            assert _slots(read) == plan.slot_map() and read.makespan == placed.makespan
            assert read.knee == placed.knee and read.mode == mode
            for step in plan.steps:
                slot = read.slot_of(step.claim_id)
                assert (slot.start, slot.finish) == (step.start, step.start + step.duration)
            # the StreamPack lowering: the pack hydrated from the plan's bytes is the pack
            pack_from_plan = hydrate(module, realization_of(plan), plan.source_plan)
            assert encode(pack_from_plan) == encode(hydrate(module, result, plan.source_plan))
            assert verify_pack(module, pack_from_plan, realization_of(plan)) == []
            assert (
                verify_execution_plan(
                    module, plan, target=target, result=result, pack=pack_from_plan
                )
                == []
            )
            checked += 1
    assert checked >= 26


def test_the_static_memory_planner_s_lifetimes_are_carried_exactly():
    assert static_memory_lifetimes_agree()


# --- the verifier ---------------------------------------------------------------------------


def test_the_verifier_names_every_binding_it_checks():
    module, target, theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    pack = hydrate(module, result, "plan0")
    assert verify_execution_plan(module, plan, target=target, result=result, pack=pack) == []
    # another target: R13, and the stream geometry it implies
    other = TargetProfile.x86_avx512()
    laws = {d.law for d in verify_execution_plan(module, plan, target=other)}
    assert "R13" in laws
    # another module's plan: R13 (and R9 for the claims it does not declare)
    foreign = next(build() for name, build in sorted(PROGRAMS.items()))
    laws = {d.law for d in verify_execution_plan(foreign, plan)}
    assert {"R9", "R13"} <= laws
    # a pack of another plan name / another realization: R10
    stranger = hydrate(module, result, "other")
    messages = [
        d.message for d in verify_execution_plan(module, plan, pack=stranger) if d.law == "R10"
    ]
    assert any("source_plan" in m for m in messages)
    narrowed = replace(
        result,
        steps=[
            replace(result.steps[0], candidate=replace(result.steps[0].candidate, width=1)),
            *result.steps[1:],
        ],
    )
    messages = [
        d.message for d in verify_execution_plan(module, plan, result=narrowed) if d.law == "R9"
    ]
    assert any("is not the realization's step" in m for m in messages)
    # a placement that is not the canonical dispatch's own: R9
    moved = replace(
        plan, steps=[replace(plan.steps[0], start=plan.steps[0].start + 1), *plan.steps[1:]]
    )
    moved = replace(moved, makespan=moved.makespan + 1)
    messages = [
        d.message for d in verify_execution_plan(module, moved, target=target) if d.law == "R9"
    ]
    assert any("canonical dispatch" in m for m in messages)


def test_malformed_plans_are_refused_before_publication_on_the_python_rail():
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    variants = malformed_plan_bytes(module, plan)
    assert len(variants) >= 18
    for name, blob, _rails in variants:
        assert python_refuses(module, blob), name
    name, owner, blob, _rails = out_of_order_variant()
    assert python_refuses(owner, blob), name
    # and the well-formed plan is not refused by any of it
    assert not python_refuses(module, encode_plan(plan))


def test_a_stale_generation_vector_is_refused_on_the_python_rail():
    seen = set()
    for name, module, plan, pack, _live in stale_cases():
        diags = [d for d in verify_execution_plan(module, plan, pack=pack) if d.law == "R11"]
        assert diags, name
        seen.add(name)
        text = " ".join(d.message for d in diags)
        if name == "registry-moved":
            assert "repack" in text
        if name == "declared-after":
            assert "declared after" in text
        if name == "pack-older-than-plan":
            assert "pack's generation vector" in text
    assert seen == {"registry-moved", "declared-after", "pack-older-than-plan"}


# --- the C twin ------------------------------------------------------------------------------


def test_the_c_twin_decodes_and_the_python_re_encode_is_byte_identical():
    """plan.abi.roundtrip on the C rail, over every corpus plan in both modes."""
    if compiler() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        count = 0
        for name, module, target, _theta, result in _corpus():
            for mode in ("eft", "tokens"):
                blob = encode_plan(
                    plan_from_realization(module, result, target, mode, plan="plan0")
                )
                again = parse_c_dump(c_roundtrip(exe, tmp, blob))
                assert encode_plan(again) == blob, (name, mode)
                count += 1
        assert count >= 26
        # a plan carrying every record family, including the signed cost and the tail sentinel
        module, target, _theta, result = audit_fixture()
        plan = plan_from_realization(module, result, target, "tokens", plan="full")
        plan.lifetimes = [Lifetime(1, "ram", 0, 4096, 64, 0, 2)]
        plan.moves = [MovementEdge(1, "ram", "hbm", 0, 4096, "pcie", "staged", "flush", 1, 2, 5, 9)]
        plan.steps = [
            replace(plan.steps[0], cost=-3, duration=0, stream=TAIL_STREAM),
            *plan.steps[1:],
        ]
        blob = encode_plan(plan)
        dump = c_roundtrip(exe, tmp, blob)
        assert "cost=-3" in dump and f"stream={TAIL_STREAM_WIRE}" in dump
        assert encode_plan(parse_c_dump(dump)) == blob


def test_the_c_twin_refuses_every_malformed_and_stale_variant():
    if compiler() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        refused, count = malformed_variants(exe, tmp)
        assert count >= 36 and refused == count, (refused, count)
        refused, count = stale_fixtures(exe, tmp)
        assert count == 6 and refused == count, (refused, count)
        # the well-formed plan passes every verdict the harness can render
        module, target, _theta, result = audit_fixture()
        plan = plan_from_realization(module, result, target, "eft", plan="plan0")
        code, out = run_harness(
            exe,
            tmp,
            encode_plan(plan),
            pack_bytes=encode(hydrate(module, result, "plan0")),
            live=generation_vector(module),
        )
        assert code == 0 and "pack=BCIR_OK" in out and "vector=BCIR_OK" in out, out


def test_the_c_twin_holds_one_step_a_claim_in_every_order():
    """The step walk's claim check rescans the steps before a claim only when the claim is at or
    below the greatest so far (QUAL-LIN's rule for a pack's segments, on the plan): the C twin's
    verdict is the codec's on every order of the audit plan's steps -- as minted, reversed,
    shuffled -- and on each order with one claim stepped twice: the greatest claim again (a tie
    with the greatest so far), the least again (below it), the greatest again right after
    itself. Every duplicate is refused (BCIR_ERR_PROVENANCE) and every order without one
    accepted, on both rails."""
    if compiler() is None:
        return
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    steps = list(plan.steps)
    assert len(steps) >= 4
    rng = random.Random(0x5EED_E9)
    orders = [steps, steps[::-1]] + [rng.sample(steps, len(steps)) for _ in range(3)]
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        for order in orders:
            top = max(range(len(order)), key=lambda k: order[k].claim_id)
            low = min(order, key=lambda st: st.claim_id)
            cases = {
                "once": order,
                "greatest.again": order + [order[top]],
                "least.again": order + [low],
                "greatest.twice": order[: top + 1] + [order[top]] + order[top + 1 :],
            }
            for name, case in cases.items():
                blob = raw_encode(replace(plan, steps=case))
                twice = len({st.claim_id for st in case}) != len(case)
                assert wire_refuses(blob) == twice, name
                code, out = run_harness(exe, tmp, blob)
                if twice:
                    assert code != 0 and "verify=BCIR_ERR_PROVENANCE" in out, (name, out)
                else:
                    assert code == 0 and "\nOK\n" in out + "\n", (name, out)
                checked += 1
    assert checked == 4 * len(orders)


def test_the_c_twin_walks_a_plans_steps_in_time_linear_in_them():
    """The complexity witness of the step walk's claim check: the fastest of several
    bcir_ep_verify runs over one plan shape -- the claims ascending, as GEM steps them, one
    lifetime -- at two sizes. Linear work grows with the size ratio (16x here); a walk that
    rescans the steps before each one grows with its square (256x; a ten-step training plan's
    2,231 steps cost 17 ms of rescans before PX-2b). The bound is a wide band, four times the
    linear ratio, so a shared runner's noise cannot reach it; neither can a quadratic walk get
    under it."""
    if compiler() is None:
        return
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    step = plan.steps[0]

    def shape(n: int) -> bytes:
        steps = [replace(step, claim_id=k, cost=0, start=0, duration=0) for k in range(n)]
        return raw_encode(replace(plan, steps=steps, lifetimes=plan.lifetimes[:1], moves=[]))

    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)

        def fastest(n: int, reps: int) -> float:
            path = os.path.join(tmp, f"steps{n}.bin")
            with open(path, "wb") as f:
                f.write(shape(n))
            r = subprocess.run(
                [exe, "--time", path, str(reps)], capture_output=True, text=True, timeout=300
            )
            assert r.returncode == 0, r.stdout + r.stderr
            fields = r.stdout.split()
            assert fields[:2] == ["verify", "ns"], r.stdout
            return max(float(fields[2]), 1.0)

        ratio = fastest(8_000, 3) / fastest(500, 7)
        assert ratio < 64.0, f"500 -> 8000 steps cost {ratio:.0f}x (bound 64x)"


def test_the_c_twin_binds_a_pack_to_its_plan():
    if compiler() is None:
        return
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    blob = encode_plan(plan)
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        # another plan's pack (by name)
        code, out = run_harness(exe, tmp, blob, pack_bytes=encode(hydrate(module, result, "other")))
        assert code != 0 and "pack=BCIR_ERR_PROVENANCE" in out, out
        # a pack whose realization is not the plan's (one segment narrowed)
        pack = hydrate(module, result, "plan0")
        pack.segments[0] = replace(pack.segments[0], width=1)
        code, out = run_harness(exe, tmp, blob, pack_bytes=encode(pack))
        assert code != 0 and "pack=BCIR_ERR_PROVENANCE" in out, out
        # a pack carrying a different vector than its plan
        pack = hydrate(module, result, "plan0")
        pack.generations[0] = replace(
            pack.generations[0], data_gen=pack.generations[0].data_gen + 1
        )
        pack.data_gen = max(g.data_gen for g in pack.generations)
        code, out = run_harness(exe, tmp, blob, pack_bytes=encode(pack))
        assert code != 0 and "pack=BCIR_ERR_STALE" in out, out
        # bytes the plan decoder refuses are refused before any binding is attempted
        assert c_refuses(exe, tmp, reseal(blob[:-16]))
        assert not c_refuses(exe, tmp, blob)


def test_a_decoded_plan_is_the_same_value_the_c_dump_describes():
    """The dump parser is exact: the plan it rebuilds equals the decoded plan, field for field."""
    if compiler() is None:
        return
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "tokens", plan="plan0")
    blob = encode_plan(plan)
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        assert parse_c_dump(c_roundtrip(exe, tmp, blob)) == decode_plan(blob)


# --- v2 (G5): the liveness domain and the lifetime ticks -----------------------------------


def _schedule_liveness_plan():
    from bcir.gem.schedule import schedule_plan as _schedule
    from bcir.kbcir.static_memory import plan_static_memory
    from bcir.performance_audit import _AuditHardware, static_memory_module

    module = static_memory_module(1)
    target = TargetProfile.x86_avx2()
    result = optimize(module, target, COOL)
    placement = _schedule(module, result, target, "tokens")
    static = plan_static_memory(
        module, {rid: "ram" for rid in module.resources}, _AuditHardware(), schedule=placement
    )
    return (
        module,
        target,
        result,
        plan_from_realization(module, result, target, "tokens", static_plan=static),
    )


def test_a_v1_plan_stays_v1_and_a_schedule_liveness_plan_is_v2():
    from bcir.abi.execution_plan_abi import plan_version

    module, target, result, plan = _schedule_liveness_plan()
    assert plan.liveness == "schedule" and plan_version(plan) == 2
    blob = encode_plan(plan)
    assert blob[4] == 2 and blob[9] == 1
    again = decode_plan(blob)
    assert again == plan and encode_plan(again) == blob
    assert all(lt.last_tick > lt.first_tick for lt in again.lifetimes)
    assert any(not lt.phase_default for lt in again.lifetimes)
    # the same realization without lifetimes, or with phase-default lifetimes, is v1 bytes
    bare = plan_from_realization(module, result, target, "tokens")
    assert plan_version(bare) == 1 and encode_plan(bare)[4] == 1 and encode_plan(bare)[9] == 0
    phase = replace(
        plan,
        liveness="phase",
        lifetimes=[
            replace(lt, first_tick=lt.first_phase, last_tick=lt.last_phase + 1)
            for lt in plan.lifetimes
        ],
    )
    assert plan_version(phase) == 1
    # a lifetime default-constructs its ticks from its phases
    assert Lifetime(1, "ram", 0, 64, 64, 2, 5).phase_default
    assert (
        Lifetime(1, "ram", 0, 64, 64, 2, 5).first_tick,
        Lifetime(1, "ram", 0, 64, 64, 2, 5).last_tick,
    ) == (2, 6)
    # a v2 plan is refused by the version gate of a v1-only reader: the header says so
    assert struct.unpack_from("<H", blob, 4)[0] == 2


def test_the_verifier_refuses_lifetimes_that_do_not_cover_the_schedule():
    module, target, result, plan = _schedule_liveness_plan()
    assert verify_execution_plan(module, plan, target=target, result=result) == []
    lt0 = plan.lifetimes[0]
    shrunk = replace(
        plan, lifetimes=[replace(lt0, last_tick=lt0.last_tick - 1), *plan.lifetimes[1:]]
    )
    messages = [d.message for d in verify_execution_plan(module, shrunk) if d.law == "R9"]
    assert any("does not cover the schedule" in m for m in messages), messages
    # two lifetimes of one bank live at once at one address: refused by the codec and the verifier
    a, b = plan.lifetimes[0], plan.lifetimes[1]
    overlapping = replace(
        plan,
        lifetimes=[
            a,
            replace(b, offset=a.offset, first_tick=a.first_tick, last_tick=a.last_tick),
            *plan.lifetimes[2:],
        ],
    )
    try:
        encode_plan(overlapping)
        raise AssertionError("aliasing lifetimes were published")
    except AbiError as exc:
        assert "alias" in str(exc)
    messages = [d.message for d in verify_execution_plan(module, overlapping) if d.law == "R9"]
    assert any("alias" in m for m in messages), messages
    # a lifetime for a resource nothing touches
    stray = replace(plan, lifetimes=[*plan.lifetimes, replace(lt0, rid=999_999, offset=1 << 40)])
    messages = [d.message for d in verify_execution_plan(module, stray) if d.law == "R9"]
    assert any("no claim touches" in m for m in messages), messages


def test_the_c_twin_reads_a_v2_plan_and_refuses_an_aliasing_one():
    if compiler() is None:
        return
    module, target, result, plan = _schedule_liveness_plan()
    blob = encode_plan(plan)
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        dump = c_roundtrip(exe, tmp, blob)
        assert "header version=2 mode=1 liveness=1" in dump
        assert encode_plan(parse_c_dump(dump)) == blob
        assert parse_c_dump(dump) == plan
        # a v1 plan dumps the phase default for its lifetimes' ticks
        bare = plan_from_realization(module, result, target, "eft")
        assert "liveness=0" in c_roundtrip(exe, tmp, encode_plan(bare))
        # the alias law and the empty-interval law on the C rail (raw bytes the codec refuses)
        from bcir.tests.plan_fixtures import raw_encode

        a, b = plan.lifetimes[0], plan.lifetimes[1]
        aliasing = replace(
            plan,
            lifetimes=[
                a,
                replace(b, offset=a.offset, first_tick=a.first_tick, last_tick=a.last_tick),
                *plan.lifetimes[2:],
            ],
        )
        assert c_refuses(exe, tmp, raw_encode(aliasing))
        late = next(lt for lt in plan.lifetimes if lt.first_tick > 0)  # (0, 0) is the phase default
        empty = replace(
            plan,
            lifetimes=[
                replace(lt, last_tick=lt.first_tick) if lt is late else lt for lt in plan.lifetimes
            ],
        )
        assert not empty.lifetimes[plan.lifetimes.index(late)].phase_default
        assert c_refuses(exe, tmp, raw_encode(empty))
        assert python_refuses(module, raw_encode(empty))
        # a v1 plan with a nonzero liveness byte is refused as reserved
        v1 = bytearray(encode_plan(bare))
        v1[9] = 1
        assert c_refuses(exe, tmp, reseal(bytes(v1)))
        assert python_refuses(module, reseal(bytes(v1)))


def test_the_c_twin_reads_every_planned_v3_plan_and_refuses_every_malformed_edge():
    """G8 (S5-C) on the C rail: every plan the movement planner mints decodes on the twin and
    re-encodes byte-identically -- the v3 move tail and the binding trailer included -- and
    every malformed move and binding is refused by both rails (one verdict per variant)."""
    if compiler() is None:
        return
    from bcir.abi.execution_plan_abi import plan_version
    from bcir.kbcir.movement import execution_plan_of
    from bcir.tests import movement_fixtures as mf

    h, _theta = mf.target_and_theta()
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        moved, edges = set(), set()
        for name, (_module, _spec, mp) in mf.planned().items():
            plan = execution_plan_of(mp.best, h)
            blob = encode_plan(plan)
            dump = c_roundtrip(exe, tmp, blob)
            assert parse_c_dump(dump) == plan, name
            assert encode_plan(parse_c_dump(dump)) == blob, name
            # the plan binds its pack and the live registry of the module it realizes (M'): the
            # vector is located before the v3 binding trailer, not at the end of the body
            mod = mp.best.transform.module
            code, out = run_harness(
                exe,
                tmp,
                blob,
                pack_bytes=encode(hydrate(mod, mp.best.result, "plan0")),
                live=generation_vector(mod),
            )
            assert code == 0 and "pack=BCIR_OK" in out and "vector=BCIR_OK" in out, (name, out)
            if plan_version(plan) == 3:
                assert f"binding source_hash={plan.source_hash} spec_hash={plan.spec_hash}" in dump
                moved.add(name)
                edges |= {(mv.kind, mv.coherence) for mv in plan.moves}
        # ... and a pack or a registry that moved after the v3 plan was minted is stale
        _module, _spec, mp = mf.planned("writeback")["writeback"]
        plan = execution_plan_of(mp.best, h)
        mod = mp.best.transform.module
        rid = min(mod.resources)
        moved_mod = replace(
            mod,
            resources={
                **mod.resources,
                rid: replace(mod.resources[rid], data_gen=mod.resources[rid].data_gen + 1),
            },
        )
        code, out = run_harness(
            exe,
            tmp,
            encode_plan(plan),
            pack_bytes=encode(hydrate(moved_mod, mp.best.result, "plan0")),
            live=generation_vector(moved_mod),
        )
        assert code != 0 and "pack=BCIR_ERR_STALE" in out and "vector=BCIR_ERR_STALE" in out, out
        # the plans that move data, and every edge shape they carry, crossed the C rail
        assert moved == set(mf.planned()) - mf.IDENTITY, moved
        assert edges == {
            ("direct", "none"),
            ("direct", "writeback"),
            ("staged", "none"),
            ("compressed", "none"),
            ("rematerialized", "none"),
        }, edges
        variants = v3_variants()
        assert len(variants) >= 30
        for name, blob, _plan in variants:
            assert wire_refuses(blob), name
            assert c_refuses(exe, tmp, blob), name


def test_the_c_twin_reads_an_explicit_placement_and_refuses_undefined_flags():
    """G19 on the C rail: a plan that states its placement (header flag bit 0, the exact
    solver's optimum) decodes, verifies and re-encodes byte for byte; an undefined flag is
    refused as reserved and the flag on a token-pipelined plan as a plan law -- the statuses
    the Python codec's refusals correspond to."""
    from bcir.gem.execution_plan import exact_plan
    from bcir.tests.exact_fixtures import six_job_module, six_job_realization, six_job_target

    if compiler() is None:
        return
    module, target = six_job_module(), six_job_target(2)
    plan, _certificate = exact_plan(module, six_job_realization((1, 3, 3, 3, 4, 4)), target)
    assert plan.explicit_placement
    blob = encode_plan(plan)
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        dump = c_roundtrip(exe, tmp, blob)
        assert " flags=1 " in dump, dump[:300]
        assert encode_plan(parse_c_dump(dump)) == blob
        for flags, mode, status in ((2, 0, "BCIR_ERR_RESERVED"), (1, 1, "BCIR_ERR_PLAN")):
            wire = bytearray(blob)
            struct.pack_into("<H", wire, 6, flags)
            wire[8] = mode
            code, out = run_harness(exe, tmp, reseal(bytes(wire)))
            assert code != 0 and f"status={status}" in out, (flags, mode, out)


# --- ALIAS-SWEEP: the alias law by a sweep, every verdict the pairwise law's -------------------


def _pairwise_first_alias(rows) -> int:
    """The alias law as both rails judged it before ALIAS-SWEEP: each lifetime against every
    earlier one, in order -- the first that aliases one, `len(rows)` when none does."""
    from bcir.abi.execution_plan_abi import lifetimes_alias

    for i, a in enumerate(rows):
        if any(lifetimes_alias(b, a) for b in rows[:i]):
            return i
    return len(rows)


def _crowded(rng: random.Random, n: int) -> list:
    """`n` lifetimes in three banks over ten ticks and 128 bytes: crowded, so most draws alias and
    many alias only late; equal ticks, touching spans and nested intervals among them."""
    rows = []
    for rid in range(1, n + 1):
        first = rng.randrange(10)
        alignment = rng.choice((1, 4, 16))
        rows.append(Lifetime(rid=rid, bank=rng.choice(("ram", "scratch", "l2")),
                             offset=rng.randrange(128 // alignment) * alignment,
                             size_bytes=1 + rng.randrange(24), alignment=alignment,
                             first_phase=first, last_phase=first, first_tick=first,
                             last_tick=first + 1 + rng.randrange(4)))  # fmt: skip
    return rows


def _apart(rows: list) -> list:
    """`rows` moved apart in one bank so that none aliases another, the ticks kept."""
    out, at = [], 0
    for lt in rows:
        out.append(replace(lt, bank="ram", offset=at, alignment=1))
        at += lt.size_bytes
    return out


def _reused(n: int) -> list:
    """GEM's shape: `n` values reusing two spans of one bank as their ticks part -- value i live on
    [i, i + 2) at the span i % 2 -- so none aliases another while every tick holds two."""
    return [Lifetime(rid=i + 1, bank="ram", offset=64 * (i % 2), size_bytes=64, alignment=64,
                     first_phase=i, last_phase=i + 1, first_tick=i, last_tick=i + 2)
            for i in range(n)]  # fmt: skip


def test_the_alias_laws_sweep_names_the_pairwise_laws_first_alias_on_every_draw():
    """ALIAS-SWEEP on the Python rail: `first_alias` -- a sweep over the ticks deciding whether a
    prefix holds an aliasing pair, and a binary search of sweeps for the first lifetime aliasing an
    earlier one -- names the pairwise law's lifetime on 3,000 crowded draws of 0 to 40 lifetimes
    and on each draw moved apart (none aliases); `validate_plan` refuses with the pairwise law's
    message, naming the same pair. The sweep's boundaries are pinned: an interval ending at the
    tick another starts at, and two spans that only touch, do not alias; one byte or one tick of
    overlap does, and another bank never does."""
    from bcir.abi.execution_plan_abi import first_alias, lifetimes_alias, validate_plan

    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    rng = random.Random(0xA11A5)
    aliased = clean = 0
    for draw in range(3_000):
        rows = _crowded(rng, rng.randrange(41))
        for case in (rows, _apart(rows)):
            want = _pairwise_first_alias(case)
            assert first_alias(case) == want, (draw, want)
            aliased += want < len(case)
            clean += want == len(case)
        if draw % 10 == 0 and _pairwise_first_alias(rows) < len(rows):
            first = _pairwise_first_alias(rows)
            prior = next(r for r in rows[:first] if lifetimes_alias(r, rows[first]))
            try:
                validate_plan(replace(plan, lifetimes=rows, moves=[]))
            except AbiError as exc:
                assert str(exc) == (f"lifetimes of RIDs {prior.rid} and {rows[first].rid} alias "
                                    f"in bank {rows[first].bank!r}"), (draw, str(exc))  # fmt: skip
            else:
                raise AssertionError(f"draw {draw}: an aliasing plan validated")
    assert aliased > 1_000 and clean > 1_000, (aliased, clean)
    a = Lifetime(1, "ram", 0, 16, 1, 0, 1, 0, 2)
    assert first_alias([a, Lifetime(2, "ram", 0, 16, 1, 2, 3, 2, 4)]) == 2  # [0, 2) then [2, 4)
    assert first_alias([a, Lifetime(2, "ram", 16, 16, 1, 0, 1, 0, 2)]) == 2  # touching spans
    assert first_alias([a, Lifetime(2, "ram", 15, 16, 1, 0, 1, 0, 2)]) == 1  # one byte
    assert first_alias([a, Lifetime(2, "ram", 0, 16, 1, 1, 1, 1, 3)]) == 1  # one tick
    assert first_alias([a, Lifetime(2, "l2", 0, 16, 1, 0, 1, 0, 2)]) == 2  # another bank
    assert first_alias(_reused(5_000)) == 5_000


def test_the_alias_laws_sweep_grows_with_the_lifetimes_not_their_square():
    """The Python rail's complexity witness: `first_alias` over GEM's shape (`_reused`, none
    aliasing: one sweep) at 2,000 and 32,000 lifetimes -- 16x the lifetimes in under 64x the time
    (n log n gives about 21x; the pairwise law gave 256x, 2.9 s already at 8,000 lifetimes on the
    PX-3b host), the fastest of three runs each."""
    from bcir.abi.execution_plan_abi import first_alias

    def fastest(n: int) -> float:
        rows = _reused(n)
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            assert first_alias(rows) == n
            elapsed = time.perf_counter() - t0
            best = elapsed if best is None else min(best, elapsed)
        return max(best, 1e-6)

    ratio = fastest(32_000) / fastest(2_000)
    assert ratio < 64.0, f"2,000 -> 32,000 lifetimes cost {ratio:.0f}x (bound 64x)"


def _sweep_verdicts(exe: str, tmp: str, blob: bytes) -> dict:
    """The harness's `--sweep` lines over `blob`, as {key: value}."""
    path = os.path.join(tmp, "sweep.bin")
    with open(path, "wb") as f:
        f.write(blob)
    r = subprocess.run([exe, "--sweep", path], capture_output=True, text=True, timeout=300)
    out = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    assert r.returncode == 0 and out.get("canary") == "ok", r.stdout + r.stderr
    return out


def _own_law_breaker(rows) -> int:
    """The first lifetime breaking a law of its own (the walk's order of them), `len(rows)` when
    none does."""
    prev = None
    for i, lt in enumerate(rows):
        if prev is not None and lt.rid <= prev:
            return i
        if (
            lt.size_bytes == 0
            or not lt.bank
            or lt.alignment & (lt.alignment - 1)
            or not lt.alignment
        ):
            return i
        if (
            lt.offset % lt.alignment
            or lt.last_phase < lt.first_phase
            or lt.last_tick <= lt.first_tick
        ):
            return i
        prev = lt.rid
    return len(rows)


def test_the_c_twin_sweeps_the_alias_law_to_the_pairwise_laws_verdict():
    """ALIAS-SWEEP on the C rail: `bcir_ep_verify_lent` -- the alias law swept over scratch the
    caller lends, of exactly `bcir_ep_sweep_bytes`' length at an odd address -- returns
    `bcir_ep_verify`'s status on 400 crowded plans, a third with a lifetime breaking a law of its
    own at a random place (zero bytes, an empty interval, a misaligned offset, a RID out of order);
    one byte of scratch less judges the law pairwise, the same status; the plan walk hands over the
    same lifetimes with scratch and without -- exactly those before the first record the walk
    refuses, an aliasing one or one breaking its own law -- and no byte past the lent scratch is
    written. The verdict is the codec's on every plan."""
    if compiler() is None:
        return
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    rng = random.Random(0x5EEB)
    refused = accepted = broken = 0
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        for draw in range(400):
            rows = _crowded(rng, rng.randrange(1, 40))
            if rng.random() < 0.5:
                rows = _apart(rows)
            if draw % 3 == 0:
                at = rng.randrange(len(rows))
                lt = rows[at]
                faults = [replace(lt, size_bytes=0), replace(lt, last_tick=lt.first_tick),
                          replace(lt, offset=2, alignment=4)]  # fmt: skip
                if at:
                    faults.append(replace(lt, rid=rows[at - 1].rid))  # a RID out of order
                rows[at] = rng.choice(faults)
                broken += 1
            blob = raw_encode(replace(plan, lifetimes=rows, moves=[]))
            v = _sweep_verdicts(exe, tmp, blob)
            assert v["verify"] == v["lent"] == v["short"], (draw, v)
            assert v["walk"] == v["walk_lent"], (draw, v)
            first = min(
                _pairwise_first_alias(rows[: _own_law_breaker(rows)]), _own_law_breaker(rows)
            )
            assert int(v["walk"].split()[1]) == first, (draw, v, first)
            assert (v["verify"] != "BCIR_OK") == wire_refuses(blob), (draw, v)
            refused += v["verify"] != "BCIR_OK"
            accepted += v["verify"] == "BCIR_OK"
    assert refused > 100 and accepted > 100 and broken > 100, (refused, accepted, broken)


def test_the_c_twin_sweeps_the_alias_law_in_time_n_log_n():
    """The C rail's complexity witness: `bcir_ep_verify_lent` over GEM's shape (`_reused`, none
    aliasing) at 2,000 and 32,000 lifetimes -- 16x the lifetimes in under 64x the time (n log n:
    about 21x; the pairwise law 256x) -- and the lent verdict under a quarter of the pairwise one's
    time at 32,000, the fastest of several runs each."""
    if compiler() is None:
        return
    module, target, _theta, result = audit_fixture()
    plan = plan_from_realization(module, result, target, "eft", plan="plan0")
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)

        def fastest(n: int, reps: int, *mode: str) -> float:
            path = os.path.join(tmp, f"life{n}.bin")
            with open(path, "wb") as f:
                f.write(raw_encode(replace(plan, lifetimes=_reused(n), moves=[])))
            r = subprocess.run([exe, "--time", path, str(reps), *mode], capture_output=True,
                               text=True, timeout=600)  # fmt: skip
            assert r.returncode == 0, r.stdout + r.stderr
            fields = r.stdout.split()
            assert fields[:2] == ["verify", "ns"], r.stdout
            return max(float(fields[2]), 1.0)

        lent_big = fastest(32_000, 3, "lent")
        ratio = lent_big / fastest(2_000, 7, "lent")
        assert ratio < 64.0, f"2,000 -> 32,000 lifetimes cost {ratio:.0f}x (bound 64x)"
        assert lent_big * 4 < fastest(32_000, 1), (
            "the lent verdict is not under a quarter of the pairwise one's time"
        )
