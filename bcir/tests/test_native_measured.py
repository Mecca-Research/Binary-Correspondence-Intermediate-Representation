"""PX-3b-2: the trainer's measured dispatch carried into K_BCIR's own candidates, and back out.

`native_choose.measured_prices` turns the dispatch `choose_dispatch` takes from a choice table into
a `cost.MeasuredPrices` table keyed on the program's claims: one entry per (op, count) the claims
of a kind's jobs hold, the kind's pick at its measured price, a key two kinds hold carried only
when both take the same pick (at the higher price), and a kind any of whose keys cannot carry its
pick left out and named. A target carrying the table (`TargetProfile.with_measured`) offers each
such claim its realizations run the pick (`<name>@<alternative>`); K_BCIR takes them by its own
min-plus; the ExecutionPlanV1 carries the plan's candidate strings; and the C lowering of the pack
(bcir_decoder_pack.c) reads each step's alternative -- in the one walk that verifies the plan
(`bcir_ep_check_pack_walk`) -- into the job's dispatch words 41 and 42, refusing an alternative
outside the grammar (ALTERNATIVE). `native_pack.plan_dispatch` is the lowering's reading in
Python.

The witnesses hold, without a compiler: the table's entries, prices and left-out kinds, and its
refusals; the planner taking every carried pick for every claim of the kind's jobs and nothing
else, R9 holding it; the plan's dispatch, read as the lowering reads it, being the program the
carried picks state word for word; the reading's grammar. With one: the check walk handing over
exactly the steps and lifetimes the plan's own walks do, with `bcir_ep_check_pack`'s status on
every pair the lowering refuses; and `train.plan.measured.mismatch` at zero -- every seeded table's
picks, priced by K_BCIR, planned and lowered by the C, are the dispatched program's words and
train the C step's bits.
"""

import copy
import ctypes as ct
import dataclasses
import tempfile
from pathlib import Path

from bcir.hosted.models import native as nat
from bcir.hosted.models.native_choose import (
    ChoiceEntry,
    ChoiceTable,
    choose_dispatch,
    measured_prices,
)
from bcir.hosted.models.native_launch import (
    RECORD_WORDS,
    WORD_INSTANCE,
    WORD_WORKERS,
    program_words,
    step_program,
)
from bcir.hosted.models.native_pack import (
    pack_artifacts,
    placed_words,
    plan_dispatch,
    training_pack,
)
from bcir.kbcir.cost import MeasuredEntry, TargetProfile, Theta
from bcir.tests import native_decoder_fixtures as nf

#: The host's C compiler, or None: the C witnesses' one capability.
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()

#: An AMX host's masks: the bf16 GEMM's family (portable, AVX-512, AMX) and the fused GEMM's.
_MASKS = {"gemm": 0b1101, "attention": 0b0111}
#: The plan corpus's untied decoder, one step of one micro-batch, under bf16 at three threads.
_SPEC, _BATCH, _TIME = nf.PLAN_CORPUS[1]
_DECISIONS = dict(contract=nat._CONTRACTS["bf16"], workspace_bytes=1 << 24, threads=3,
                  instances=_MASKS)  # fmt: skip


def _expect_refused(label, fn, reason=None):
    try:
        fn()
    except ValueError as exc:
        assert reason is None or reason in str(exc), (label, str(exc))
        return
    raise AssertionError(f"{label} admitted")


def _shape(**extra):
    return {"vocab": _SPEC.vocab_size, "d_model": _SPEC.d_model, "n_heads": _SPEC.n_heads,
            "n_layers": _SPEC.n_layers, "d_ff": _SPEC.d_ff, "kv_heads": _SPEC.kv_heads,
            "batch": _BATCH, "time": _TIME, "threads": 3, "steps": 1, "recompute": 0,
            "workspace_bytes": 0, "contract": "bf16", "instances": dict(_MASKS)} | extra  # fmt: skip


#: Picks over every way a key is held: a kernel alone (rms, attention), two GEMM roles sharing a
#: key at one pick (down.fwd, head.fwd: `train.gemm.alone.nt` 768), a role sharing its keys with
#: a kind that took no pick (o.vjp with qkv.vjp), a role whose keys are its own (qkv.fwd), an
#: unresolved alternative (gateup.fwd: not a pick), and a kind with no job (provider.silu).
_ENTRIES = (
    ChoiceEntry("dispatch:rms", "t1", 200, 180, 240, 5),
    ChoiceEntry("dispatch:attention", "i2", 240, 230, 250, 5),
    ChoiceEntry("dispatch:down.fwd", "t2", 210, 200, 240, 5),
    ChoiceEntry("dispatch:head.fwd", "t2", 230, 200, 250, 5),
    ChoiceEntry("dispatch:o.vjp", "t1", 230, 200, 250, 5),
    ChoiceEntry("dispatch:qkv.fwd", "i4", 220, 200, 240, 5),
    ChoiceEntry("dispatch:gateup.fwd", "t1", 200, 150, 260, 5),
    ChoiceEntry("dispatch:provider.silu", "t1", 100, 90, 110, 5),
)
_CARRIED = {"attention": "i2", "down.fwd": "t2", "head.fwd": "t2", "qkv.fwd": "i4", "rms": "t1"}


def _keys(**decisions):
    """Each dispatch kind's claim keys in the default program, by its dispatch labels."""
    from bcir.hosted.models.native_pack import op_name, work

    labels: list = []
    program = step_program(_SPEC, _BATCH, _TIME, **decisions, dispatch_labels=labels)
    keys: dict = {}
    for launch, kind in zip(program, labels):
        for r in launch.records:
            keys.setdefault(kind, set()).add((op_name(r), work(r)))
    return keys


def test_measured_prices_carry_each_pick_on_its_kinds_claims():
    """One entry per (op, count) the claims of a carried kind's jobs hold, its pick at its price --
    the highest of the kinds holding the key -- in the table's one order and the choice table's
    generation; a kind sharing a key with a kind of another pick, or none, left out with no entry,
    and a kind with no job left out too; nothing else priced. No table, or no pick, is no prices;
    a table of another shape, decisions that dispatch or state other threads or instances, are
    refused."""
    table = ChoiceTable(7, _shape(), {}, _ENTRIES)
    picks = choose_dispatch(table, _SPEC, _shape())
    assert set(picks) == set(_CARRIED) | {"o.vjp", "provider.silu"}  # gateup.fwd: not resolved
    prices, left = measured_prices(table, _SPEC, _shape(), **_DECISIONS)
    assert left == ("o.vjp", "provider.silu"), left
    assert prices.generation == 7
    keys = _keys(**_DECISIONS)
    want = {}
    for kind, pick in _CARRIED.items():
        for key in keys[kind]:
            want.setdefault(key, set()).add(kind)
    q8 = {e.kind.split(":", 1)[1]: e.q8 for e in _ENTRIES}
    assert list(prices.entries) == sorted(
        (MeasuredEntry(op, count, _CARRIED[next(iter(kinds))], max(q8[k] for k in kinds))
         for (op, count), kinds in want.items()),
        key=lambda e: (e.op, e.count),
    )  # fmt: skip
    shared = prices.lookup("train.gemm.alone.nt", 768)
    assert [(e.alternative, e.q8) for e in shared] == [("t2", 230)]  # down.fwd's and head.fwd's
    for key in keys["o.vjp"] | keys["qkv.vjp"] | keys["gateup.fwd"]:
        assert not prices.lookup(*key), key  # a kind left out, or one with no pick: no entry
    # no prices
    assert measured_prices(None, _SPEC, _shape(), **_DECISIONS) == (None, ())
    unresolved = ChoiceTable(1, _shape(), {}, _ENTRIES[6:7])
    assert measured_prices(unresolved, _SPEC, _shape(), **_DECISIONS) == (None, ())
    lonely = ChoiceTable(1, _shape(), {}, (ChoiceEntry("dispatch:o.vjp", "t1", 200, 190, 210, 5),))
    assert measured_prices(lonely, _SPEC, _shape(), **_DECISIONS) == (None, ("o.vjp",))
    for label, fn, reason in (
        ("another shape", lambda: measured_prices(table, _SPEC, _shape(threads=4), **_DECISIONS),
         "the table was measured at"),
        ("a dispatched program", lambda: measured_prices(
            table, _SPEC, _shape(), **_DECISIONS, dispatch={"rms": "t1"}), "no dispatch"),
        ("other threads", lambda: measured_prices(
            table, _SPEC, _shape(), **(_DECISIONS | {"threads": 2})), "threads"),
        ("other instances", lambda: measured_prices(
            table, _SPEC, _shape(), **(_DECISIONS | {"instances": {"gemm": 1, "attention": 1}})),
         "instances"),
    ):  # fmt: skip
        _expect_refused(label, fn, reason)


def test_a_pick_is_carried_only_where_its_keys_are_held_by_its_pick_alone():
    """`carried_kinds` is the greatest set of picked kinds whose keys only kinds of the same pick
    hold: a kind sharing a key with another pick, or with a kind that took none, is dropped, and so
    is a kind that shares a key only with a kind dropped so -- the chain a single pass would leave
    carried: here `a` shares `k1` with `b` (same pick), `b` shares `k2` with `c` (no pick), and
    both `a` and `b` are left out, while `d`, holding `k3` alone, and `e` and `f`, sharing `k4` at
    one pick, are carried; a kind with no claim is not."""
    from bcir.hosted.models.native_choose import carried_kinds

    keys = {"a": {"k1"}, "b": {"k1", "k2"}, "c": {"k2"}, "d": {"k3"}, "e": {"k4"}, "f": {"k4"},
            "g": {"k5"}, "h": {"k5"}}  # fmt: skip
    holders: dict = {}
    for kind, ks in keys.items():
        for k in ks:
            holders.setdefault(k, set()).add(kind)
    picks = {"a": "t1", "b": "t1", "d": "i4", "e": "t2", "f": "t2", "g": "t1", "h": "t2",
             "z": "t1"}  # fmt: skip
    assert carried_kinds(picks, keys, holders) == {"d", "e", "f"}
    # one pass would have kept `a`: its one key is held by `b`, of its pick, until `b` is dropped
    one_pass = {kind for kind in picks if keys.get(kind)
                and all(h in picks and picks[h] == picks[kind]
                        for key in keys[kind] for h in holders[key])}  # fmt: skip
    assert "a" in one_pass and "a" not in carried_kinds(picks, keys, holders)
    assert carried_kinds({}, keys, holders) == set()


def _planned(table):
    """The default program's pack planned on the AVX-512 profile carrying `table`'s prices: (the
    training pack, the plan's ExecutionPlanV1 value, the realization, the carried dispatch)."""
    from bcir.kbcir import optimize

    prices, left = measured_prices(table, _SPEC, _shape(), **_DECISIONS)
    carried = {k: a for k, a in choose_dispatch(table, _SPEC, _shape()).items() if k not in left}
    tp = training_pack(_SPEC, _BATCH, _TIME, **_DECISIONS)
    target = TargetProfile.x86_avx512().with_measured(prices)
    _, _, _, plan = pack_artifacts(tp, target=target)
    return tp, plan, optimize(tp.module, target, Theta.cool()), carried, target


def test_the_planner_takes_every_carried_pick_and_the_plan_states_its_dispatch():
    """On a target carrying the prices, K_BCIR plans every claim of a carried kind's jobs at its
    own realization run the pick (`<name>@<pick>`) and every other claim at its own realization;
    R9 holds the plan under the target; and the plan's dispatch, read as the C lowering reads it
    (`plan_dispatch`), puts into the default program's words exactly the program the carried picks
    state (`step_program(dispatch=...)`), word for word."""
    from bcir.verify import verify_plan

    tp, plan, result, carried, target = _planned(ChoiceTable(7, _shape(), {}, _ENTRIES))
    labels: list = []
    step_program(_SPEC, _BATCH, _TIME, **_DECISIONS, dispatch_labels=labels)
    kind_of_claim = {c.id: labels[ph.phase_id] for ph in tp.module.phases for c in ph.claims}
    for s in result.steps:
        pick = carried.get(kind_of_claim[s.claim_id])
        name = s.candidate.name
        assert (name.rpartition("@")[2] if "@" in name else None) == pick, (s.claim_id, name)
    assert not [d for d in verify_plan(tp.module, result, target, theta=Theta.cool())
                if d.law in ("R8", "R9")]  # fmt: skip
    dispatch = plan_dispatch(plan)
    assert placed_words(tp, tp.placements, dispatch) == tuple(
        program_words(step_program(_SPEC, _BATCH, _TIME, **_DECISIONS, dispatch=carried))
    )
    assert {v for v in dispatch.values() if v != (0, 0)} == {(2, 0), (4, 0), (0, 1), (0, 2)}


def test_the_plan_dispatch_reads_the_lowerings_grammar():
    """`plan_dispatch` reads a step's `<name>@i<bit>` as that instance and `<name>@t<k>` as k
    workers, a step without one as the default, and refuses every other alternative -- the C
    lowering's ALTERNATIVE set: two bits, no workers, a leading zero, too many workers, a
    non-ASCII digit, another letter, an empty one, a second `@`."""
    from types import SimpleNamespace

    def read(*names):
        steps = [SimpleNamespace(claim_id=i, candidate=n) for i, n in enumerate(names)]
        return plan_dispatch(SimpleNamespace(steps=steps))

    assert read("tile", "tile@i8", "vec16@t32767", "scalar@t1", "tile@i1") == {
        0: (0, 0), 1: (8, 0), 2: (0, 32767), 3: (0, 1), 4: (1, 0)}  # fmt: skip
    for alternative in ("i3", "t0", "t01", "t32768", "t١", "w1", "", "i4@t2", "i"):
        _expect_refused(alternative, lambda a=alternative: read("tile@" + a), "ALTERNATIVE")


# --- the C witnesses --------------------------------------------------------------------------------


def test_the_check_walk_hands_over_every_step_and_lifetime():
    """`bcir_ep_check_pack_walk` -- the lowering's one walk of the plan's laws -- returns
    `bcir_ep_check_pack`'s status on the intact pair and on every pack and plan the lowering's
    refusals name, and on an admitted pair hands over exactly the steps `bcir_ep_for_each_step`
    and the lifetimes `bcir_ep_for_each_lifetime` walk, in their order, steps first; and with its
    alias law swept over scratch the caller lends (ALIAS-SWEEP, `bcir_ep_sweep_bytes` of it) the
    same status and the same steps and lifetimes, on every pair."""
    if _CC is None:
        return
    from bcir.tests.test_native_run_pack import _refusals

    case = nf.trace_corpus()[0]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return
        trainer, decisions = nf._case_trainer(path, case)
        try:
            tp = training_pack(case.spec, case.batch, case.time, **decisions)
            pack_bytes, plan_bytes, pack, plan = pack_artifacts(tp)
            lib = trainer.lib
            hand = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p)

            def collector(into: list, tag: str):
                def on(view, _ctx):  # a step view opens with its claim id (u64), a lifetime's RID
                    kind = ct.c_uint64 if tag == "step" else ct.c_uint32
                    into.append((tag, ct.cast(view, ct.POINTER(kind))[0]))
                    return 0

                return hand(on)

            check = lib.bcir_ep_check_pack
            check.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t]
            check.restype = ct.c_int
            walk = lib.bcir_ep_check_pack_walk
            walk.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t, hand, hand,
                             ct.c_void_p, ct.c_void_p, ct.c_size_t]  # fmt: skip
            walk.restype = ct.c_int
            sweep_bytes = lib.bcir_ep_sweep_bytes
            sweep_bytes.argtypes = [ct.c_uint32]
            sweep_bytes.restype = ct.c_size_t
            steps = lib.bcir_ep_for_each_step
            steps.argtypes = [ct.c_char_p, ct.c_size_t, hand, ct.c_void_p]
            steps.restype = ct.c_int
            lives = lib.bcir_ep_for_each_lifetime
            lives.argtypes = [ct.c_char_p, ct.c_size_t, hand, ct.c_void_p]
            lives.restype = ct.c_int
            pairs = [("intact", pack_bytes, plan_bytes, 0)] + [
                (name, pk, pl, reason if reason < 100 else 0)
                for name, pk, pl, reason in _refusals(tp, pack, plan)
            ]
            admitted = refused = 0
            for name, pk, pl, verdict in pairs:
                handed: list = []
                status = walk(pl, len(pl), pk, len(pk), collector(handed, "step"),
                              collector(handed, "life"), None, None, 0)  # fmt: skip
                assert status == check(pl, len(pl), pk, len(pk)) == verdict, (name, status)
                # the same walk, its alias law swept over lent scratch: the same verdict and hand-over
                declared = int.from_bytes(pl[24:28], "little") if len(pl) >= 28 else 0
                need = sweep_bytes(declared)
                room = (ct.c_uint8 * max(1, need))() if need <= 1 << 26 else None
                lent: list = []
                status_lent = walk(pl, len(pl), pk, len(pk), collector(lent, "step"),
                                   collector(lent, "life"), None, room, need if room else 0)  # fmt: skip
                assert (status_lent, lent) == (status, handed), name
                if status:
                    refused += 1
                    continue
                walked: list = []
                assert steps(pl, len(pl), collector(walked, "step"), None) == 0, name
                assert lives(pl, len(pl), collector(walked, "life"), None) == 0, name
                assert handed == walked and handed, name
                if name == "intact":
                    assert [v for t, v in handed if t == "step"] == [s.claim_id for s in plan.steps]
                admitted += 1
            assert admitted >= 2 and refused >= 5, (admitted, refused)
        finally:
            trainer.close()


def test_every_measured_run_lowers_its_dispatch_and_trains_the_c_steps_bits():
    """`train.plan.measured.mismatch`: each state case of the trace corpus at two threads at
    least, under each of its seeded choice tables, its picks carried into K_BCIR by
    `measured_prices`, planned, hydrated, placed and lowered by the C, is the program its carried
    dispatch states and trains the C step's state and events bit for bit -- 0 of
    `measured_run_count()` runs, the same count on every host; and the runs dispatch: every one
    carries a pick."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        packed = nf.pack_library(Path(tmp) / "pack")
        if packed is None:
            return
        bad, runs, notes, picks = nf.measured_mismatch(packed)
        assert (bad, runs) == (0, nf.measured_run_count()), notes[:3]
        assert picks >= runs, (picks, runs)  # the rms kernel's t1 at least, in every table


def test_a_measured_plan_the_host_cannot_run_lowers_and_is_refused_by_the_run():
    """The lowering writes the plan's alternative as it reads it and leaves its admission to the
    run's own check (bcir_decoder_run_program): a plan naming an instance past the host's family,
    workers past the pool, or one job's claims at two alternatives lowers, and the run refuses it
    with nothing trained -- the dispatch laws are the run's, held once."""
    if _CC is None:
        return
    from bcir.abi.execution_plan_abi import encode_plan
    from bcir.abi.streampack_abi import encode
    from bcir.hosted.models.native import NativeRunStopped

    spec, batch, time_ = nf.PLAN_CORPUS[0]
    case = nf.TraceCase(spec, batch, time_, "kblock+exp+reduce", threads=2)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return
        trainer, decisions = nf._case_trainer(path, case)
        try:
            tp = training_pack(spec, batch, time_, **decisions)
            _, _, pack, plan = pack_artifacts(tp)
            group = next(ph for ph in tp.module.phases if len(ph.claims) > 1
                         and ph.claims[0].op.startswith("train.gemm"))  # fmt: skip
            first, second = group.claims[0].id, group.claims[1].id
            assert not decisions["instances"]["gemm"] & 8  # the kblock family has no AMX entry
            edits = (("workers past the pool", {first: "@t3", second: "@t3"}),
                     ("two alternatives in one job", {first: "@t1"}),
                     ("an instance the host lacks", {first: "@i8", second: "@i8"}))  # fmt: skip
            tokens, targets, optimizer = nf._case_inputs(case)
            for label, names in edits:
                q = copy.deepcopy(plan)
                q.steps = [dataclasses.replace(s, candidate=s.candidate + names[s.claim_id])
                           if s.claim_id in names else s for s in q.steps]  # fmt: skip
                words = trainer.lower_pack(encode(pack), encode_plan(q))
                for claim, suffix in names.items():
                    word = WORD_INSTANCE if suffix[1] == "i" else WORD_WORKERS
                    assert words[claim * RECORD_WORDS + word] == int(suffix[2:]), (label, claim)
                before = nf.state_digest(trainer)
                try:
                    trainer.run_program(words, tokens, targets, steps=1, optimizer=optimizer)
                except NativeRunStopped:
                    pass
                else:
                    raise AssertionError(f"{label}: the run admitted it")
                assert nf.state_digest(trainer) == before, label
        finally:
            trainer.close()
