"""G19 EXEC-EXACT: the proved optimum is the plan that runs.

The 2026-10-06 audit (item 1, Partial) found the exact rail certifying the heuristic's
placement without replacing it. `verify_execution_plan` re-derived the placement with the
canonical dispatch and refused any other, so the solver's shorter -- and equally legal --
placement could be proved but never carried: 190 of the section 6.1 corpus's 1,716 instances
on two domains (18 on three) ran suboptimal, with a certificate saying so.

A plan may now STATE its placement (header flag bit 0, `PLAN_FLAG_EXPLICIT_PLACEMENT`). The
verifier holds such a plan to the legality of a phase-barriered placement
(`gem.schedule.placement_violations`) instead of to equality with the dispatch; the native
codec, the ASN.1 projection and the C twin carry the flag; and `exact_plan` mints it where the
solver beats the dispatch -- and the canonical plan, byte for byte, everywhere else.

Every test here was RED on the parent (the minting and the flag did not exist, and a plan
carrying the optimum was refused by R9) and states the property, not the repair. The C twin's
half lives in `test_execution_plan.py`, with the harness it builds.
"""

from __future__ import annotations

import functools
import struct
from dataclasses import replace

from bcir.abi import AbiError, decode_plan, encode_plan
from bcir.asn1 import Asn1Error
from bcir.asn1.execution_plan import (
    EXECUTION_PLAN,
    MODULE,
    _plan_value,
    decode_plan_der,
    decode_plan_jer,
    decode_plan_oer,
    encode_plan_der,
    encode_plan_jer,
    encode_plan_oer,
    plan_to_value,
)
from bcir.examples import PROGRAMS
from bcir.gem.execution_plan import (
    PLAN_FLAG_EXPLICIT_PLACEMENT,
    TAIL_STREAM,
    exact_plan,
    plan_from_realization,
    schedule_of,
)
from bcir.gem.schedule import (
    Slot,
    durations_from,
    phase_hazards,
    placement_violations,
    schedule_plan,
    stream_geometry,
)
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.realize import optimize
from bcir.tests.exact_fixtures import (
    partition_optimum,
    six_job_corpus,
    six_job_module,
    six_job_realization,
    six_job_target,
)
from bcir.tests.plan_fixtures import reseal
from bcir.verify import verify_execution_plan

AVX, COOL = TargetProfile.x86_avx512(), Theta.cool()


@functools.cache
def _suboptimal(domains: int, count: int = 8) -> tuple[tuple[int, ...], ...]:
    """The first `count` six-job instances (corpus order) the canonical dispatch places above
    the partition oracle's optimum on `domains` domains."""
    module, target, found = six_job_module(), six_job_target(domains), []
    for durs in six_job_corpus():
        placed = schedule_plan(module, six_job_realization(durs), target, "eft")
        if placed.makespan > partition_optimum(durs, domains):
            found.append(durs)
            if len(found) == count:
                break
    return tuple(found)


def _explicit():
    """(module, target, realization, plan, certificate) for the first instance the dispatch
    gets wrong on two domains: durations (1, 3, 3, 3, 4, 4), the dispatch's 10 ticks against
    the optimum's 9, placed as stream 0 = claims 1, 5, 6 and stream 1 = claims 2, 3, 4."""
    module, target = six_job_module(), six_job_target(2)
    result = six_job_realization(_suboptimal(2)[0])
    plan, certificate = exact_plan(module, result, target)
    return module, target, result, plan, certificate


def _r9(module, plan, target) -> list[str]:
    return [d.message for d in verify_execution_plan(module, plan, target=target) if d.law == "R9"]


def _moved(plan, claim_id: int, **fields):
    """`plan` with one step's fields replaced (the flag and everything else kept)."""
    return replace(
        plan,
        steps=[replace(s, **fields) if s.claim_id == claim_id else s for s in plan.steps],
    )


def _refused(error, needle: str, call, *args, **kwargs) -> None:
    """`call(*args, **kwargs)` raises `error` with a message naming `needle`."""
    try:
        call(*args, **kwargs)
    except error as exc:
        assert needle in str(exc), (needle, str(exc))
    else:  # pragma: no cover - the defect
        raise AssertionError(f"accepted; expected a refusal naming {needle!r}")


def test_the_solvers_optimum_is_the_plan_that_runs():
    """On every sampled instance the dispatch gets wrong, the plan `exact_plan` mints carries
    the proved optimum, survives its bytes, is admitted by the verifier with the target and the
    realization, and is the placement its readers take from the bytes (`schedule_of`)."""
    checked = 0
    for domains in (2, 3):
        module, target = six_job_module(), six_job_target(domains)
        for durs in _suboptimal(domains):
            result = six_job_realization(durs)
            optimum = partition_optimum(durs, domains)
            canonical = plan_from_realization(module, result, target, "eft")
            plan, certificate = exact_plan(module, result, target)
            assert canonical.makespan == certificate.heuristic > optimum, durs
            assert certificate.stop_reason == "optimal" and certificate.incumbent == optimum
            assert plan.explicit_placement and plan.makespan == optimum, durs
            read = decode_plan(encode_plan(plan))
            assert read == plan, durs
            assert verify_execution_plan(module, read, target=target, result=result) == [], durs
            assert schedule_of(read).makespan == optimum, durs
            checked += 1
    assert checked == 16


def test_where_the_dispatch_is_optimal_the_plan_is_the_canonical_one_byte_for_byte():
    """Non-disturbance: wherever the solver cannot beat the dispatch, `exact_plan` mints the
    canonical plan -- no flag, the same bytes -- over every corpus program and every six-job
    instance the dispatch already places optimally."""
    pairs = []
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        pairs.append((name, module, optimize(module, AVX, COOL), AVX))
    module, target = six_job_module(), six_job_target(2)
    wrong = set(_suboptimal(2, 190))
    for durs in six_job_corpus()[:24]:
        if durs not in wrong:
            pairs.append((str(durs), module, six_job_realization(durs), target))
    assert len(pairs) >= len(PROGRAMS) + 12
    for name, module, result, target in pairs:
        plan, certificate = exact_plan(module, result, target)
        canonical = plan_from_realization(module, result, target, "eft")
        assert certificate.incumbent == certificate.heuristic, name
        assert not plan.explicit_placement and encode_plan(plan) == encode_plan(canonical), name


def test_a_forged_explicit_placement_is_refused_with_its_reason():
    """The flag admits a legal placement, never an illegal one: each forgery breaks exactly one
    rule of a phase-barriered placement and draws exactly that refusal."""
    module, target, _result, plan, _certificate = _explicit()
    assert _r9(module, plan, target) == []
    first = {s.claim_id: (s.stream, s.start) for s in plan.steps}
    assert first[1] == (0, 0) and first[2] == (1, 0)  # the anchors the forgeries are written on
    cases = {
        # claim 1 joins claim 2 on stream 1 at tick 0
        "overlap": (
            _moved(plan, 1, stream=1),
            "explicit placement: claims 1 and 2 overlap on stream 1",
        ),
        # a slot shorter than its cost: the wire laws refuse it before the placement is read
        "duration": (
            _moved(plan, 4, duration=2),
            "malformed execution plan: step[3] duration 2 is not the slot its cost 3 places",
        ),
        "tail": (
            _moved(plan, 1, stream=TAIL_STREAM),
            "explicit placement: claim 1 is not eligible for the tail",
        ),
        # a stream the target does not have: a wire law too
        "stream": (
            _moved(plan, 1, stream=2),
            "malformed execution plan: step[0] stream 2 is outside 2 streams",
        ),
        "makespan": (
            replace(plan, makespan=plan.makespan + 1),
            "explicit placement: the makespan is 10, the last finish 9",
        ),
        "tokens": (
            replace(plan, mode="tokens"),
            "malformed execution plan: an explicit placement is a phase-barriered (eft) placement",
        ),
    }
    for name, (forgery, why) in cases.items():
        assert _r9(module, forgery, target) == [why], (name, _r9(module, forgery, target))


def test_a_forged_explicit_placement_cannot_cross_a_phase_barrier_or_a_hazard():
    """The two rules the independent six jobs cannot exercise, on corpus programs: a claim of a
    later phase may not start before every earlier phase has finished, and a claim may not
    start before an intra-phase hazard predecessor finishes."""
    checked = set()
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        result = optimize(module, AVX, COOL)
        plan = replace(
            plan_from_realization(module, result, AVX, "eft"),
            flags=PLAN_FLAG_EXPLICIT_PLACEMENT,
        )
        assert _r9(module, plan, AVX) == [], name  # the canonical placement is legal
        step = {s.claim_id: s for s in plan.steps}
        if "barrier" not in checked and len(module.phases) >= 2:
            later = max(plan.steps, key=lambda s: s.start)
            if later.start > 0:
                why = _r9(module, _moved(plan, later.claim_id, start=0), AVX)
                assert any(
                    f"claim {later.claim_id} starts at 0, before its phase" in w for w in why
                ), (name, why)
                checked.add("barrier")
        for edges in phase_hazards(module).values():
            for claim_id, preds in edges.items():
                for pred in preds:
                    if "hazard" in checked or step[pred].duration == 0:
                        continue
                    why = _r9(module, _moved(plan, claim_id, start=step[pred].start), AVX)
                    assert any(
                        f"before its hazard predecessor {pred} finishes" in w for w in why
                    ), (name, why)
                    checked.add("hazard")
    assert checked == {"barrier", "hazard"}, checked


def test_a_plan_without_the_flag_is_held_to_the_canonical_placement_as_before():
    """The flag is what admits a stated placement: the same optimum without it is refused as
    the parent refused it, and a canonical plan verifies exactly as before."""
    module, target, result, plan, _certificate = _explicit()
    unflagged = replace(plan, flags=0)
    assert _r9(module, unflagged, target) == [
        "the plan's placement is not what the canonical dispatch produces from its own step costs"
    ]
    canonical = plan_from_realization(module, result, target, "eft")
    assert canonical.flags == 0 and _r9(module, canonical, target) == []


def test_the_flag_survives_the_native_bytes_and_every_asn1_transfer_syntax():
    module, target, result, plan, _certificate = _explicit()
    blob = encode_plan(plan)
    assert struct.unpack_from("<H", blob, 6)[0] == PLAN_FLAG_EXPLICIT_PLACEMENT
    assert decode_plan(blob).explicit_placement
    assert plan_to_value(plan)["flags"] == PLAN_FLAG_EXPLICIT_PLACEMENT
    for encode, decode in (
        (encode_plan_der, decode_plan_der),
        (encode_plan_oer, decode_plan_oer),
        (encode_plan_jer, decode_plan_jer),
    ):
        back = decode(encode(plan))
        assert back == plan and back.explicit_placement, encode.__name__
        assert encode_plan(back) == blob, encode.__name__
    # the canonical plan's header and projection carry no flag
    canonical = plan_from_realization(module, result, target, "eft")
    assert struct.unpack_from("<H", encode_plan(canonical), 6)[0] == 0
    assert plan_to_value(canonical)["flags"] == 0


def test_an_undefined_flag_and_an_explicit_token_plan_are_refused_everywhere():
    """Bit 0 is the only defined flag, and only an eft plan carries it: the native encoder and
    decoder and every ASN.1 transfer syntax refuse the rest, as the C twin does."""
    module, target, result, plan, certificate = _explicit()
    blob = encode_plan(plan)
    for flags, mode, needle in (
        (2, "eft", "undefined ExecutionPlan flags 0x0002"),
        (0x8000, "eft", "undefined ExecutionPlan flags 0x8000"),
        (PLAN_FLAG_EXPLICIT_PLACEMENT, "tokens", "a phase-barriered (eft) placement"),
    ):
        bad = replace(plan, flags=flags, mode=mode)
        _refused(AbiError, needle, encode_plan, bad)
        wire = bytearray(blob)
        struct.pack_into("<H", wire, 6, flags)
        wire[8] = 1 if mode == "tokens" else 0
        _refused(AbiError, needle, decode_plan, reseal(bytes(wire)))
        document = _plan_value(bad)
        for decode, data in (
            (decode_plan_der, MODULE.encode("ExecutionPlan", document)),
            (decode_plan_oer, _oer(document)),
            (decode_plan_jer, _jer(document)),
        ):
            _refused(Asn1Error, needle, decode, data)
    # the minting side: a stated placement is an eft placement, and a legal one
    _refused(
        ValueError,
        "a phase-barriered (eft) placement",
        plan_from_realization,
        module,
        result,
        target,
        "tokens",
        placement=certificate.schedule,
    )
    crowded = replace(
        certificate.schedule,
        slots=[replace(s, domain=0) for s in certificate.schedule.slots],
    )
    _refused(
        ValueError,
        "the explicit placement is not a legal placement: claims",
        plan_from_realization,
        module,
        result,
        target,
        "eft",
        placement=crowded,
    )


def _oer(document) -> bytes:
    from bcir.asn1.oer import OerRules, encode_oer

    return encode_oer(EXECUTION_PLAN, document, rules=OerRules.CANONICAL)


def _jer(document) -> bytes:
    from bcir.asn1.jer import JerRules, encode_jer

    return encode_jer(EXECUTION_PLAN, document, rules=JerRules.CANONICAL)


def test_placement_violations_names_every_slot_that_is_not_one_claim_once():
    """The legality predicate is total over the slot list it is handed -- the minting side
    (`plan_from_realization(placement=)`) reads it before any wire law could: a claim placed
    twice, a stream past the target's (which the dispatch's index for the tail must not
    alias), a slot that is not its claim's duration, a claim with no slot and a slot for a
    claim the module does not declare are each named, in that order."""
    module, target = six_job_module(), six_job_target(2)
    durations = dict(enumerate((1, 3, 3, 3, 4, 4), start=1))
    slots = [
        Slot(1, 0, 0, 1),
        Slot(1, 1, 0, 1),
        Slot(99, 1, 1, 2),
        Slot(2, 0, 1, 4),
        Slot(3, 0, 5, 8),
        Slot(4, 2, 9, 12),
        Slot(5, 0, 13, 16),
    ]
    assert placement_violations(module, durations, target, slots, 16) == [
        "claim 1 is placed twice",
        "claim 4 is placed on stream 2, outside the target's 2 streams",
        "claim 5 runs 3 ticks, its duration is 4",
        "claim 6 has no slot",
        "claim 99 is not a claim of the module",
    ]


def test_on_hazard_bearing_modules_the_solvers_placement_is_minted_and_admitted():
    """A generated differential: over seeded random modules with intra-phase hazards and
    seeded random step costs, every plan `exact_plan` mints -- explicit wherever the solver
    beats the dispatch -- passes the legality the mint reads and the verifier holds, so the
    predicate and the solver agree on what a legal placement is where hazards bind. At least
    fifteen explicit placements, every one on a module with hazard edges, or the corpus did
    not test what it is here for."""
    import random

    from bcir.kbcir import TARGETS
    from bcir.kbcir.realize import RealizationResult
    from bcir.tests.sweep_fixtures import random_module

    rng = random.Random(19)
    minted = explicit = 0
    for seed in range(400):
        module = random_module(seed)
        if any(len(phase.claims) > 6 for phase in module.phases):
            continue
        edges = sum(
            len(preds) for phase in phase_hazards(module).values() for preds in phase.values()
        )
        for name in ("x86_avx2", "nvidia_ptx"):
            target = replace(TARGETS[name], affinity_domains=rng.choice([2, 3]))
            chosen = optimize(module, target, COOL).steps
            for _trial in range(4):
                steps = [replace(s, cost=rng.randint(0, 9)) for s in chosen]
                result = RealizationResult(steps=steps, score=sum(s.cost for s in steps))
                plan, certificate = exact_plan(module, result, target)
                read = decode_plan(encode_plan(plan))
                assert verify_execution_plan(module, read, target=target, result=result) == [], (
                    seed,
                    name,
                )
                assert plan.makespan == certificate.incumbent <= certificate.heuristic
                assert plan.explicit_placement == (certificate.incumbent < certificate.heuristic)
                minted += 1
                if plan.explicit_placement:
                    assert edges > 0, seed
                    explicit += 1
    assert minted >= 500 and explicit >= 15, (minted, explicit)


def test_a_tail_claim_past_the_last_stream_is_not_read_as_the_tail():
    """The dispatch indexes the tail one past the last domain. A slot on that stream NUMBER is
    a stream the target does not have, not the tail -- even for a claim the tail would take,
    which the first version of the predicate accepted there."""
    module = PROGRAMS["histogram_gather"]()
    result = optimize(module, AVX, COOL)
    placed = schedule_plan(module, result, AVX, "eft")
    domains, _knee = stream_geometry(AVX)
    durations = durations_from(result)
    tail = next(s for s in placed.slots if s.domain == TAIL_STREAM)
    assert placement_violations(module, durations, AVX, placed.slots, placed.makespan) == []
    moved = [replace(s, domain=domains) if s is tail else s for s in placed.slots]
    assert placement_violations(module, durations, AVX, moved, placed.makespan) == [
        f"claim {tail.claim_id} is placed on stream {domains}, outside the target's "
        f"{domains} streams"
    ]


def test_a_schedule_liveness_memory_plan_must_be_planned_against_the_carried_placement():
    """A schedule-liveness memory plan is bound to the placement whose ticks it was planned
    against: one planned against the dispatch's placement is refused for a plan that carries
    the solver's, and one planned against the solver's is carried and verifies."""
    from bcir.kbcir.static_memory import plan_static_memory
    from bcir.performance_audit import _AuditHardware

    module, target, result, _plan, certificate = _explicit()
    banks = {rid: "ram" for rid in module.resources}
    stale = plan_static_memory(
        module, banks, _AuditHardware(), schedule=schedule_plan(module, result, target, "eft")
    )
    _refused(
        ValueError,
        "planned against another placement than the solver's",
        exact_plan,
        module,
        result,
        target,
        static_plan=stale,
    )
    bound = plan_static_memory(module, banks, _AuditHardware(), schedule=certificate.schedule)
    plan, _again = exact_plan(module, result, target, static_plan=bound)
    assert plan.explicit_placement and plan.liveness == "schedule" and plan.lifetimes
    assert verify_execution_plan(module, decode_plan(encode_plan(plan)), target=target) == []
