"""PX-2 (PACK): the native training step run from the program its planner states.

`bcir_decoder_run_program` runs a launch program -- the records PX-1's probe traces, with the
host's checks and updates in place -- through the trainer's own kernels, once the whole program has
passed its check (bcir_decoder_train.h). `native_launch.step_program` states the program for a
trainer's state and the planner's own choices. The witness holds every program run of the trace
corpus to the C step on the same case, state and events bit for bit, and its trace to the program;
the other tests hold the check (every class of malformed program refused before anything is
written), the verdicts of a run that fails, and the record format to the header's.
"""

import ctypes as ct
import math
import re
import struct
import tempfile
from array import array
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from bcir.hosted.models import native as nat
from bcir.hosted.models.native import NativeAdamW, NativeDecoder, NativeRunStopped
from bcir.hosted.models.native_launch import (
    CHECK,
    CHECK_FORWARD,
    CHECK_LOSS,
    CHECK_RAN,
    GEMM,
    LOSS,
    PROGRAM_KIND_NAMES,
    RECORD_WORDS,
    ROPE,
    ROPE_TABLE,
    SCAN,
    UPDATE,
    _layout,
    program_words,
    step_program,
)
from bcir.hosted.models.native_program import training_step_program
from bcir.tests import native_decoder_fixtures as nf

#: The host's C compiler, or None: the witnesses' one capability (a fault sweep `--require`s it,
#: so a host without one is never read as a sweep the witnesses passed).
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()
_C = Path(__file__).resolve().parents[2] / "runtime" / "c"
_W = RECORD_WORDS
_ONE = struct.unpack("<I", struct.pack("<f", 1.0))[0]


def _slot(q: int) -> int:
    """The first word of operand slot q (its space; then its offset and its bytes)."""
    return 11 + 3 * q


def test_a_program_trains_the_c_steps_bits():
    """Every program run of the trace corpus -- every case from the program that takes the C
    step's own decisions, its state cases under each of the planner's other choices -- trains the
    C step's state and events on the same case bit for bit, runs to the end, and traces exactly the
    launches its program states (`train.plan.program.mismatch` 0). Each choice changes some
    program (no variant is the default again, L2) and states what it names (`_takes`), and the
    programs reach every record kind."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.trace_library(Path(tmp) / "trace")
        if path is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        problems, changed, kinds, defaults, runs = [], Counter(), Counter(), {}, 0
        for case, choices, step, run in nf.program_runs(path):
            runs += 1
            found = nf.program_problems(step, run)
            if found:
                problems.append((case, choices, found[:2]))
            words = program_words(run.program)
            if not choices:
                defaults[case] = words
            else:
                changed[next(iter(choices))] += words != defaults[case]
                if not _takes(run.program, choices, case):
                    problems.append((case, choices, "the program is not the choice it names"))
            for launch in run.program:
                kinds[launch.records[0].kind] += 1
        assert not problems, problems[:3]
        assert runs == nf.program_run_count(), (runs, nf.program_run_count())
        assert set(changed) == {"group", "pair", "panels", "facts", "recompute"}, changed
        assert all(changed.values()), changed
        assert set(kinds) == set(PROGRAM_KIND_NAMES) - {17, 18, 19}, sorted(kinds)


def _takes(program, choices, case) -> bool:
    """Whether a program states the choice it was built under: calls alone (every GEMM job one
    call or a linear's two VJPs, a chain call then another), no such pair, no panel anywhere, the
    whole weights scanned in every micro-batch; a recompute set changes the program itself."""
    roles = [tuple(r.flags >> 4 & 15 for r in launch.records) for launch in program
             if launch.records[0].kind == GEMM]  # fmt: skip
    if choices.get("group") is False:
        return all(len(job) == 1 or job == (1, 2) for job in roles)
    if choices.get("pair") is False:
        return all(job != (1, 2) for job in roles)
    if choices.get("panels") is False:
        return not any(r.flags >> 24 & 3 or r.kind in (11, 13) for launch in program
                       for r in launch.records)  # fmt: skip
    if choices.get("facts") is False:
        params = sum(4 * b.elements for b in training_step_program(case.spec, case.batch,
                     case.time).buffers if b.arena == "weights")  # fmt: skip
        scans = sum(1 for launch in program for r in launch.records
                    if r.kind == SCAN and r.refs[0] == (1, 0, params))  # fmt: skip
        return scans == case.steps * case.accumulation
    return True


def _trainer(lib, case):
    trainer, decisions = nf._case_trainer(lib, case)
    return trainer, decisions


def _snapshot(trainer):
    """Every byte a run could write: the arenas, the workspace, the step and the bias powers."""
    data = tuple(ct.string_at(buf, ct.sizeof(buf)) for buf in trainer.buffers.values())
    return data, (trainer.state.step, trainer.state.beta1_power, trainer.state.beta2_power)


def _raw(trainer, words, case, *, records=None, program=None):
    """bcir_decoder_run_program's status on `words` for `case`'s tokens, and the count it left
    (lent as 777, so a refusal that touched it shows)."""
    fn = trainer.lib.bcir_decoder_run_program
    fn.argtypes = [
        ct.POINTER(nat._State), ct.POINTER(nat._Optimizer), ct.c_void_p, ct.c_size_t,
        ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint32), ct.c_size_t, ct.c_void_p, ct.c_size_t,
        ct.c_size_t, ct.POINTER(nat._Event), ct.c_size_t, ct.POINTER(ct.c_size_t),
    ]  # fmt: skip
    fn.restype = ct.c_int
    tokens, targets, optimizer = nf._case_inputs(case)
    x, y = (ct.c_uint32 * len(tokens))(*tokens), (ct.c_uint32 * len(targets))(*targets)
    code = (ct.c_uint64 * max(len(words), 1))(*words)
    opt = nat._Optimizer(**asdict(optimizer))
    events, done = (nat._Event * case.steps)(), ct.c_size_t(777)
    rc = fn(ct.byref(trainer.state), ct.byref(opt), program if program is not None else code,
            len(words) // _W if records is None else records, x, y, len(tokens), None,
            case.steps, case.accumulation, events, case.steps, ct.byref(done))  # fmt: skip
    return rc, done.value


def _records(words):
    return [list(words[i : i + _W]) for i in range(0, len(words), _W)]


def _words(records):
    return [word for r in records for word in r]


_renumber = nf.renumber_jobs


def _at(records, kind, nth=0, entry=None):
    """The index of the nth record of `kind` (and GEMM `entry`) in the program."""
    hits = [i for i, r in enumerate(records)
            if r[0] == kind and (entry is None or r[2] & 15 == entry)]  # fmt: skip
    return hits[nth]


def _job(records, i):
    """The indices of the GEMM job record i starts."""
    out = [i]
    while (
        out[-1] + 1 < len(records)
        and records[out[-1] + 1][0] == GEMM
        and (records[out[-1] + 1][1] == records[i][1])
    ):
        out.append(out[-1] + 1)
    return out


def _v1_refusals(records, n, activations, tmp_offset):
    """(name, edit) for each class of malformed program the check refuses, as edits of the v1
    base: every rule of bcir_decoder_train.h's check, the bound checks at their edges."""
    rms, add, silu = _at(records, 4), _at(records, 9), _at(records, 10)
    one = _at(records, GEMM, entry=0)  # a call alone (the attention's output projection)
    pair = _at(records, GEMM, entry=4)
    chain = next(i for i in range(len(records))
                 if records[i][0] == GEMM and records[i][2] & 15 == 5 and (records[i][2] >> 4) & 15 == 1)  # fmt: skip
    loss, update = _at(records, LOSS), _at(records, UPDATE)
    rope, table = _at(records, ROPE), _at(records, ROPE_TABLE)
    checks = [i for i, r in enumerate(records) if r[0] == CHECK and r[2] == CHECK_LOSS]

    def put(i, word, value):
        def edit(rs):
            rs[i][word] = value

        return edit

    def bump(i, word, by):
        def edit(rs):
            rs[i][word] += by

        return edit

    def ops(*edits):
        def edit(rs):
            for e in edits:
                e(rs)

        return edit

    def drop(*indices):
        def edit(rs):
            for i in sorted(indices, reverse=True):
                del rs[i]
            _renumber(rs)

        return edit

    def twice(i):
        def edit(rs):
            rs.insert(i + 1, list(rs[i]))
            _renumber(rs)

        return edit

    def split(i):
        def edit(rs):
            rs[i][1] = -1  # a launch of its own, then every job numbered again
            _renumber(rs)

        return edit

    y = _slot(2)
    past = activations - records[rms][y + 2] + 4
    return [
        ("a kind no record has", put(rms, 0, ROPE_TABLE + 1)),
        ("a kind a trace alone holds", put(rms, 0, 17)),
        ("a spare word set", put(rms, _W - 1, 1)),
        ("a dimension the kind leaves unused", put(rms, 3 + 5, 1)),
        ("a zero dimension", put(rms, 3, 0)),
        ("an operand past its space by one float", put(rms, y + 1, past)),
        ("an operand of another size", bump(rms, _slot(0) + 2, 4)),
        ("an operand off its float alignment", bump(rms, y + 1, 2)),
        ("a kernel writing the weights", ops(put(rms, y, 1), put(rms, y + 1, 0))),
        ("a kernel reading a moment", put(rms, _slot(1), 3)),
        ("an operand in the workspace", put(rms, _slot(0), 7)),
        ("an operand in a space there is not", put(rms, _slot(0), 9)),
        ("a slot named but absent", put(rms, _slot(5) + 2, 4)),
        (
            "a written operand that is also read",
            ops(*(put(add, _slot(1) + k, records[add][_slot(0) + k]) for k in range(3))),
        ),
        ("another contract's GEMM entry", put(one, 2, (records[one][2] & ~0xFF000F) | 1 | 1 << 16)),
        (
            "a provider's call without the provider",
            put(one, 2, (records[one][2] & ~0xFF000F) | 7 | 15 << 16),
        ),
        ("a GEMM's alpha 2", put(one, 3 + 5, struct.unpack("<I", struct.pack("<f", 2.0))[0])),
        ("a GEMM's beta one half", put(one, 3 + 6, struct.unpack("<I", struct.pack("<f", 0.5))[0])),
        ("a GEMM transposed twice over", put(one, 3, 2)),
        (
            "a pair's calls in each other's roles",
            ops(bump(pair, 2, 1 << 4), bump(pair + 1, 2, -(1 << 4))),
        ),
        ("a pair split into two jobs", split(pair + 1)),
        ("a chain into two C", put(chain + 1, _slot(2) + 1, tmp_offset)),
        ("a panel outside the bf16 group", bump(one, 2, 1 << 24)),
        ("a job sharing the launch before it", put(add + 1, 1, records[add][1])),
        ("BCIR's SiLU under the v2 contract's name", put(silu, 2, 1)),
        (
            "the provider's SiLU without the provider",
            ops(
                put(silu, 0, 21),
                *(put(silu, _slot(1) + k, records[silu][_slot(2) + k]) for k in range(3)),
                *(put(silu, _slot(2) + k, 0) for k in range(3)),
            ),
        ),
        ("an RMSNorm of another eps", bump(rms, 5, 1)),
        # ROPE-TABLE: a rotation reads its table, time x head-dim floats; the fill writes it
        ("a rotation without its table", ops(*(put(rope, _slot(1) + k, 0) for k in range(3)))),
        ("a rotation's table of another time", bump(rope, _slot(1) + 2, -4 * records[rope][6])),
        (
            "a rotation's table inside the x it writes",  # the table's size, at x's place
            ops(
                put(rope, _slot(1), records[rope][_slot(0)]),
                put(rope, _slot(1) + 1, records[rope][_slot(0) + 1]),
            ),
        ),
        (
            "a table fill of an odd width",  # its size the dims', inside the arena
            ops(
                bump(table, 4, 1),
                bump(table, _slot(0) + 2, 4 * records[table][3]),
                bump(table, _slot(0) + 1, -4 * records[table][3]),
            ),
        ),
        (
            "a table fill into the weights",
            ops(put(table, _slot(0), 1), put(table, _slot(0) + 1, 0)),
        ),
        ("a table fill with a dimension it leaves unused", put(table, 3 + 2, 1)),
        ("the tokens at an offset", put(_at(records, 3), _slot(0) + 1, 4)),
        ("the weights read before they are known", drop(_at(records, SCAN))),
        ("a micro-batch without its loss", drop(loss)),
        ("two losses in one micro-batch", twice(loss)),
        ("a check of no test", put(checks[0], 2, CHECK_FORWARD | CHECK_RAN)),
        ("a check with an operand", put(checks[0], _slot(0), 5)),
        ("an update after one micro-batch of two", drop(*range(checks[0] + 1, checks[1] + 1))),
        ("a step short", drop(*range(update + 1, len(records)))),
        ("an update of another chunk", bump(update, 4, 1024)),
        ("an update of the moments in another order", put(update, _slot(2), 4)),
    ]


def test_a_refused_program_writes_nothing():
    """Every class of malformed program -- a record outside its grammar, an operand outside its
    space, size or alignment or meeting a written one, a GEMM job or flags the state's kernels do
    not take, the weights read before they are known, a micro-batch or step of another shape, a
    program for another state or lying inside this one -- is refused (status 1) before anything
    runs: no byte of the state written, not even the completed count. The base program itself
    runs (so each refusal is its edit's), under v1 and under the bf16 contract with its panels."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib = nf.checkout_library(Path(tmp) / "plain")
        if lib is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        spec, batch, time_ = nf.PLAN_CORPUS[0]
        case = nf.TraceCase(spec, batch, time_, "v1", steps=2, accumulation=2)
        trainer, decisions = _trainer(lib, case)
        try:
            words = program_words(step_program(spec, batch, time_, **decisions))
            records = _records(words)
            lay = _layout(training_step_program(spec, batch, time_, trainer.recompute))
            refusals = _v1_refusals(records, batch * time_, 4 * trainer.activation_count,
                                    lay["tmp"][1])  # fmt: skip
            assert len(refusals) >= 37
            before, wrong = _snapshot(trainer), []
            for name, edit in refusals:
                edited = [list(r) for r in records]
                edit(edited)
                rc, done = _raw(trainer, _words(edited), case)
                if rc != 1 or done != 777 or _snapshot(trainer) != before:
                    wrong.append((name, rc, done))
            # a program short of its last record
            rc, done = _raw(trainer, words, case, records=len(records) - 1)
            if rc != 1 or done != 777 or _snapshot(trainer) != before:
                wrong.append(("a program short of its last record", rc, done))
            assert not wrong, wrong
            assert _raw(trainer, words, case) == (0, case.steps)  # the base runs
        finally:
            trainer.close()
        # a valid program lying inside the state it would train (its second moment, which the
        # update commits): refused, as the run would overwrite the program it reads
        bench = nf.PANELS_BENCH
        case = nf.TraceCase(*bench, "v1")
        trainer, decisions = _trainer(lib, case)
        try:
            words = program_words(step_program(*bench, **decisions))
            inside = trainer.buffers["moment2"]
            assert ct.sizeof(inside) >= 8 * len(words) and ct.addressof(inside) % 8 == 0
            ct.memmove(inside, struct.pack(f"<{len(words)}Q", *words), 8 * len(words))
            before = _snapshot(trainer)
            rc, done = _raw(trainer, words, case, program=ct.addressof(inside))
            assert (rc, done, _snapshot(trainer) == before) == (1, 777, True), (rc, done)
            assert _raw(trainer, words, case) == (0, case.steps)  # the same words, outside
        finally:
            trainer.close()
        # a program for another state: BCIR's kernels where a provider replaces the GEMM and the
        # SiLU, BCIR's SiLU where a provider replaces it, or a cache the state's arena cannot hold
        for other, choices, name in (
            (nf.TraceCase(spec, batch, time_, "v1", steps=2, accumulation=2, provider="mm+silu"),
             {"provider_mm": False, "provider_silu": False}, "another state's kernels"),
            (nf.TraceCase(spec, batch, time_, "v1", steps=2, accumulation=2, provider="mm+silu"),
             {"provider_silu": False}, "BCIR's SiLU under the provider's"),
            (nf.TraceCase(spec, batch, time_, "v1", recompute=3, steps=2, accumulation=2),
             {"recompute": 0}, "a cache the arena cannot hold"),
        ):  # fmt: skip
            trainer, decisions = _trainer(lib, other)
            try:
                plain = decisions | choices
                before = _snapshot(trainer)
                rc, done = _raw(trainer, program_words(step_program(spec, batch, time_, **plain)),
                                other)  # fmt: skip
                assert (rc, done, _snapshot(trainer) == before) == (1, 777, True), name
            finally:
                trainer.close()
        # the bf16 contract's own rules, on the panels' bench with its panels taken
        spec, batch, time_ = nf.PANELS_BENCH
        case = nf.TraceCase(spec, batch, time_, "bf16", workspace="panels")
        trainer, decisions = _trainer(lib, case)
        try:
            words = program_words(step_program(spec, batch, time_, **decisions))
            records = _records(words)
            vjp = next(i for i, r in enumerate(records)
                       if r[0] == GEMM and r[2] & 15 == 6 and len(_job(records, i)) == 4)  # fmt: skip
            one = next(i for i, r in enumerate(records) if r[0] == GEMM and r[2] & 15 == 6)
            assert records[vjp][2] >> 24 & 1 and records[vjp + 1][2] >> 24 & 1  # panels taken
            ws = _slot(3)
            bf16 = [
                ("a panel two calls share",
                 [(vjp + 1, _slot(0) + k, records[vjp][_slot(0) + k]) for k in range(3)]),
                ("the workspace short of the state's", [(one, ws + 2, records[one][ws + 2] - 64)]),
                ("a workspace slot off the workspace's start", [(one, ws + 1, 64)]),
                ("a GEMM operand in the workspace", [(one, _slot(0), 7)]),
                ("a bf16 call alone with no workspace", [(one, ws, 0), (one, ws + 1, 0), (one, ws + 2, 0)]),
            ]  # fmt: skip
            before, wrong = _snapshot(trainer), []
            for name, writes in bf16:
                edited = [list(r) for r in records]
                for i, word, value in writes:
                    edited[i][word] = value
                rc, done = _raw(trainer, _words(edited), case)
                if rc != 1 or done != 777 or _snapshot(trainer) != before:
                    wrong.append((name, rc, done))
            assert not wrong, wrong
            assert _raw(trainer, words, case) == (0, case.steps)
        finally:
            trainer.close()
        # a panel on a call of partial tiles (rows 4: no 32-row panel)
        spec, batch, time_ = nf.PLAN_CORPUS[0]
        case = nf.TraceCase(spec, batch, time_, "bf16")
        trainer, decisions = _trainer(lib, case)
        try:
            records = _records(program_words(step_program(spec, batch, time_, **decisions)))
            one = next(i for i, r in enumerate(records) if r[0] == GEMM and r[2] & 15 == 6)
            m, k = records[one][5], records[one][7]
            records[one][2] |= 1 << 24
            records[one][_slot(0) + 2] = m * k * 2
            before = _snapshot(trainer)
            rc, done = _raw(trainer, _words(records), case)
            assert (rc, done, _snapshot(trainer) == before) == (1, 777, True), (rc, done)
        finally:
            trainer.close()


def test_a_blocked_job_states_its_workspace_or_is_refused():
    """KSPLIT: under the blocked GEMM contract, in a state that lends a workspace (a long context's:
    the attention's copies), a GEMM job with a weight adjoint states the state's whole workspace on
    every record, and nothing else does -- a weight adjoint's job without it, one of its records
    without it, its workspace a float short or off its start, and a job without a weight adjoint
    lent it are each refused before anything runs (no byte written); the programs themselves run,
    the pairs and groups of the C step's decisions and the weight adjoints alone (no pair). A short
    context's blocked state short of four K blocks (KSPLIT-ROOM) lends none, and its program,
    stating none, runs."""
    from bcir.frontends.models.decode import DecoderSpec

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib = nf.checkout_library(Path(tmp) / "plain")
        if lib is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        spec = DecoderSpec(16, 8, 2, 1, 16, activation="silu_gate", n_kv_heads=1)
        case = nf.TraceCase(spec, 1, 1024, "kblock")
        trainer, decisions = _trainer(lib, case)
        try:
            lent = decisions["workspace_bytes"]
            assert lent > 0, decisions  # the attention's copies: the workspace the GEMMs are lent
            ws, wrong, edits = _slot(3), [], 0
            for choices in ({}, {"pair": False}):
                words = program_words(step_program(spec, 1, 1024, **decisions, **choices))
                records = _records(words)
                jobs = [_job(records, i) for i, r in enumerate(records)
                        if r[0] == GEMM and (not i or records[i - 1][:2] != r[:2])]  # fmt: skip
                weight = next(j for j in jobs if any(records[i][3] == 1 for i in j)
                              and len(j) == (1 if choices else 2))  # fmt: skip
                other = next(j for j in jobs if all(records[i][3] == 0 for i in j))
                assert all(records[i][ws : ws + 3] == [7, 0, lent] for i in weight), weight
                assert all(records[i][ws : ws + 3] == [0, 0, 0] for i in other), other
                cases = [
                    ("a weight adjoint's job without the workspace",
                     [(i, ws + k, 0) for i in weight for k in range(3)]),
                    ("its workspace a float short", [(i, ws + 2, lent - 4) for i in weight]),
                    ("its workspace off its start", [(i, ws + 1, 64) for i in weight]),
                    ("a job without a weight adjoint lent the workspace",
                     [(i, ws + k, (7, 0, lent)[k]) for i in other for k in range(3)]),
                ]  # fmt: skip
                if len(weight) > 1:
                    cases.append(("one record of a weight adjoint's job without the workspace",
                                  [(weight[0], ws + k, 0) for k in range(3)]))  # fmt: skip
                before = _snapshot(trainer)
                for name, writes in cases:
                    edited = [list(r) for r in records]
                    for i, word, value in writes:
                        edited[i][word] = value
                    rc, done = _raw(trainer, _words(edited), case)
                    edits += 1
                    if rc != 1 or done != 777 or _snapshot(trainer) != before:
                        wrong.append((choices, name, rc, done))
                assert _raw(trainer, words, case) == (0, case.steps), choices  # the base runs
            assert not wrong, wrong
            assert edits == 9, edits
        finally:
            trainer.close()
        # a short context's blocked state four rows deep lends no workspace (no attention copies,
        # no K block to split): its program states none, and runs
        short = nf.TraceCase(*nf.PLAN_CORPUS[0], "kblock")
        trainer, decisions = _trainer(lib, short)
        try:
            assert not decisions["workspace_bytes"], decisions
            words = program_words(step_program(*nf.PLAN_CORPUS[0], **decisions))
            records = _records(words)
            assert all(r[_slot(3) : _slot(3) + 3] == [0, 0, 0] for r in records if r[0] == GEMM)
            assert any(r[0] == GEMM and r[3] == 1 for r in records)  # it has weight adjoints
            assert _raw(trainer, words, short) == (0, short.steps)
        finally:
            trainer.close()


def _stop(trainer, words, tokens, targets, case, rates=None):
    """`case`'s run on `trainer` -- from `words` where given, else the C step's own -- as its
    (status, completed) and every byte it left (`_snapshot`)."""
    common = dict(steps=case.steps, gradient_accumulation=case.accumulation,
                  learning_rates=rates, optimizer=NativeAdamW())  # fmt: skip
    try:
        if words is None:
            trainer.run(tokens, targets, **common)
        else:
            trainer.run_program(words, tokens, targets, **common)
        stopped = (0, case.steps)
    except NativeRunStopped as error:
        stopped = (error.status, error.completed)
    return stopped, _snapshot(trainer)


def test_a_failing_run_stops_where_the_step_stops():
    """A run that fails stops where the C step stops, every byte of the state as the step leaves
    it: weights not finite before the first step -- NUMERIC from the opening scan, nothing written;
    a forward that overflows on finite weights (RMSNorm gains of 1e30) -- NUMERIC at the forward's
    check, before the backward writes a gradient; an update the preflight refuses -- NUMERIC after
    the steps before it. And a product that refuses its workspace -- a program taking the panels
    on a state whose workspace holds only the floor -- fails the run with CAPACITY before any
    commit, reported before the gradient's verdict when both fail, as the step orders them."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib = nf.checkout_library(Path(tmp) / "plain")
        if lib is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        spec, batch, time_ = nf.PLAN_CORPUS[1]
        case = nf.TraceCase(spec, batch, time_, "kblock+exp", steps=2, accumulation=2)
        tokens, targets, _ = nf._case_inputs(case)

        def poison(trainer):
            trainer.buffers["weights"][3] = math.nan

        def gains(trainer):
            for name, _shape, offset, count in trainer.layout:
                if "layernorm" in name:
                    for i in range(count):
                        trainer.buffers["weights"][offset + i] = 1e30

        for label, rates, edit, stops in (
            ("weights not finite", None, poison, (3, 0)),
            ("a forward that overflows", None, gains, (3, 0)),
            ("an update the preflight refuses", [1e-3, 1e39], None, (3, 1)),
        ):  # fmt: skip
            outcomes, befores = [], []
            for from_program in (False, True):
                trainer, decisions = _trainer(lib, case)
                try:
                    if edit:
                        edit(trainer)
                    befores.append(_snapshot(trainer))
                    words = None
                    if from_program:
                        words = program_words(step_program(spec, batch, time_, **decisions))
                    outcomes.append(_stop(trainer, words, tokens, targets, case, rates))
                finally:
                    trainer.close()
            assert befores[0] == befores[1], label  # one seeded state, one edit
            assert outcomes[0][0] == outcomes[1][0] == stops, (
                label,
                outcomes[0][0],
                outcomes[1][0],
            )
            assert outcomes[0][1] == outcomes[1][1], f"{label}: the program left other bytes"
            if edit is poison:
                assert outcomes[1][1] == befores[1], "the poisoned run wrote a byte"
        spec, batch, time_ = nf.PANELS_BENCH
        for accumulation, nan_gradients in ((1, False), (2, True)):
            case = nf.TraceCase(spec, batch, time_, "bf16", workspace="floor",
                                accumulation=accumulation)  # fmt: skip
            tokens, targets, _ = nf._case_inputs(case)
            trainer, decisions = _trainer(lib, case)
            try:
                program = step_program(spec, batch, time_, **(decisions | {"panels_need": 0}))
                assert any(r.flags >> 24 & 1 for launch in program for r in launch.records)
                if nan_gradients:  # the refused products leave them: the scan would fail too
                    for i in range(trainer.state.plan.parameters):
                        trainer.buffers["gradients"][i] = math.nan
                committed = nf.state_digest(trainer)
                stopped, _ = _stop(trainer, program_words(program), tokens, targets, case)
                assert stopped == (2, 0), (accumulation, stopped)
                assert nf.state_digest(trainer) == committed, accumulation
            finally:
                trainer.close()


def test_the_record_format_is_the_headers():
    """The record kinds, the check's tests and the record's words the launch model spells are
    the header's (bcir_decoder_train.h), read from its text: two rails out of their own sources --
    PX-3b's two dispatch words among them."""
    header = _C / "bcir_decoder_train.h"
    if not header.is_file():
        return  # an installed package ships no C units: the one absence that is expected
    text = header.read_text(encoding="utf-8")
    words = re.search(r"#define BCIR_DT_RECORD_WORDS (\d+)", text)
    assert words and int(words.group(1)) == RECORD_WORDS
    from bcir.hosted.models.native_launch import WORD_INSTANCE, WORD_WORKERS

    dispatch = dict(re.findall(r"#define BCIR_DT_WORD_(INSTANCE|WORKERS) (\d+)", text))
    assert {k: int(v) for k, v in dispatch.items()} == {
        "INSTANCE": WORD_INSTANCE, "WORKERS": WORD_WORKERS}  # fmt: skip
    enum = re.search(r"enum \{ (BCIR_DT_REC_SCAN=1,[^}]*)\}", text, re.S)
    assert enum, "the header names no record kinds"
    names = [name.split("=")[0].strip() for name in enum.group(1).split(",") if name.strip()]
    from bcir.hosted.models import native_launch as nl

    header_kinds = {name[len("BCIR_DT_REC_") :]: i + 1 for i, name in enumerate(names)}
    ours = {name: getattr(nl, name) for name in header_kinds}
    assert ours == header_kinds, (ours, header_kinds)
    assert set(header_kinds.values()) == set(PROGRAM_KIND_NAMES)
    checks = dict(re.findall(r"BCIR_DT_CHECK_(\w+)=(\d+)u", text))
    assert {k: int(v) for k, v in checks.items()} == {
        "FORWARD": CHECK_FORWARD, "RAN": CHECK_RAN, "LOSS": CHECK_LOSS,
    }  # fmt: skip


def test_the_program_model_refuses_what_it_cannot_state():
    """A choice that is no bool, and words that are no program -- not whole records, a word
    outside 64 bits, a bool or a float for a word, an array of another type -- are refused
    before any call; the launch model's own refusals stand. An `array('Q')` of a program's
    words is lent as it is and runs to the list's state."""
    spec, batch, time_ = nf.PLAN_CORPUS[0]
    for bad in (
        lambda: step_program(spec, batch, time_, contract=0, group=1),
        lambda: step_program(spec, batch, time_, contract=0, facts="no"),
        lambda: step_program(spec, batch, time_, contract=0, steps=0),
        lambda: step_program(spec, batch, time_, contract=0, accumulation=True),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("the program model stated a program for a refused call")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib = nf.checkout_library(Path(tmp) / "plain")
        if lib is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        trainer = NativeDecoder(lib, spec, batch_size=batch, context_length=time_)
        try:
            n = batch * time_
            for words in ([0] * (_W - 1), [-1] + [0] * (_W - 1), [1 << 64] + [0] * (_W - 1),
                          [1.0] + [0] * (_W - 1), [True] + [0] * (_W - 1),
                          array("d", [0.0] * _W), array("q", [-1] + [0] * (_W - 1))):  # fmt: skip
                try:
                    trainer.run_program(words, [0] * n, [0] * n, steps=1)
                except NativeRunStopped as error:
                    raise AssertionError(f"words reached the C: {error}") from error
                except ValueError:
                    continue
                raise AssertionError("run_program took words that are no program")
        finally:
            trainer.close()
        # the same program as an array('Q'): lent as it is, the list's state bit for bit
        case = nf.TraceCase(spec, batch, time_, "v1")
        tokens, targets, _ = nf._case_inputs(case)
        digests = []
        for lend in (list, lambda words: array("Q", words)):
            trainer, decisions = _trainer(lib, case)
            try:
                words = program_words(step_program(spec, batch, time_, **decisions))
                trainer.run_program(lend(words), tokens, targets, steps=case.steps)
                digests.append(nf.state_digest(trainer))
            finally:
                trainer.close()
        assert digests[0] == digests[1], digests


def test_the_fuzz_seeds_start_inside_the_grammar():
    """The fuzz harness's seeds (runtime/c/fuzz_decoder_train.c, seeded by
    tools/c/fuzz_streampack.sh) are what they claim on the harness's own state: every planner
    program -- every configuration the harness names, two of them at two steps of two, four of them
    placed by lifetime on the arenas they reach (PX-4) -- admitted and run to its end, every near
    miss refused, the placed program a float short among them. A canonical seed refused would be a
    corpus that never reaches the run (L2), and the campaign would hold only the check."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib = nf.checkout_library(Path(tmp) / "plain")
        if lib is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        names = {bits: name for name, bits in nat._CONTRACTS.items()}
        spec, batch, time_ = nf.FUZZ_STATE
        wrong, configs = [], set()
        for name, data, admitted in nf.fuzz_seeds():
            config, shape = data[0], data[1]
            steps, accumulation = 1 + (shape & 1), 1 + (shape >> 1 & 1)
            contract = nf.FUZZ_CONTRACTS[config & 7]
            words = list(struct.unpack(f"<{(len(data) - 2) // 8}Q", data[2:]))
            common = dict(batch_size=batch, context_length=time_, contract=names[contract],
                          recompute=3 if config & 8 else 0,
                          workspace_bytes=nf.FUZZ_WORKSPACE if contract & 16 else None)  # fmt: skip
            trainer = NativeDecoder(lib, spec, **common)
            lent = nf.fuzz_lent(words, config, trainer.state.plan)
            if lent is not None:  # PX-4: the arenas the program reaches, as the harness lends them
                trainer.close()
                trainer = NativeDecoder(lib, spec, transient=lent, **common)
            try:
                n = batch * time_ * steps * accumulation
                tokens = [(7 * i + 3) % spec.vocab_size for i in range(n)]
                targets = [(5 * i + 1) % spec.vocab_size for i in range(n)]
                try:
                    trainer.run_program(words, tokens, targets, steps=steps,
                                        gradient_accumulation=accumulation)  # fmt: skip
                    status = 0
                except NativeRunStopped as error:
                    status = error.status
            finally:
                trainer.close()
            if status != (0 if admitted else 1):
                wrong.append((name, status))
            if admitted:
                configs.add(config)
        assert not wrong, wrong
        placed = {config | nf.FUZZ_PLACED for config in nf.FUZZ_PLACED_CONFIGS}
        assert configs == set(range(16)) | placed, sorted(configs)
