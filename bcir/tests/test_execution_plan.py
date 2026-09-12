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
import struct
import tempfile

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
    reseal,
    run_harness,
    stale_cases,
    stale_fixtures,
    static_memory_lifetimes_agree,
)
from bcir.verify import verify_execution_plan, verify_pack

AVX, COOL = TargetProfile.x86_avx512(), Theta.cool()


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
