"""PX-3b (DISPATCH): the kernel instance and the pool's thread split as choices of the step program.

`native_launch.step_program(dispatch=...)` takes, per dispatch kind -- a GEMM's role, an attention,
or any other kernel -- the instance its jobs run (`i<bit>`: one of bcir_tensor.h's bits, 1
portable, 2 AVX2, 4 AVX-512, 8 AMX, that the job's family admits) or the workers they run on
(`t<k>`, k up to the pool's threads), into words 41 and 42 of every record of each job of the kind.
The C holds them to the host (bcir_decoder_run_program's check), dispatches each job so (the
`bcir_tensor_*_variant_pool` entries and `bcir_tensor_pool_cap`), and its trace records the
instance a job ran -- the entry's own return -- and the workers it ran on. Every instance of a
family gives the same bits (the contract) and a pool's partition is a function of the shapes alone,
so every dispatch trains the same bits as the step: what differs is the time, which PX-3's
measurement prices (`native_choose`, the `dispatch:<kind>` kinds).

The witnesses hold, without a compiler: the alternatives a host admits; a dispatch takes its
alternatives and nothing else, into its kind's jobs and no other; the default dispatch is the
program before it, word for word; the model resolves the defaults as the trace records them; a
table that measured the dispatch names the instances it measured on, has one spelling, and is
chosen from by `choose`'s rule. With one: every alternative of every kind, and every kind at once,
trains the C step's bits and is traced as its program states, under contracts of every family at
one thread and three; every program the host cannot dispatch is refused with nothing written; a
run leaves the pool's cap as it found it; a measurement of the dispatch is a table at the host's
instances.
"""

import ctypes as ct
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from bcir.hosted.models import native as nat
from bcir.hosted.models.native_choose import (
    DISPATCH_CHOICE_KINDS,
    ChoiceEntry,
    ChoiceTable,
    choose,
    choose_dispatch,
    dispatch_kind,
    measure,
    trainer_shape,
)
from bcir.hosted.models.native_launch import (
    ATTN,
    ATTN_VJP,
    CHECK,
    DISPATCH_KINDS,
    E_PROVIDER,
    GEMM,
    GEMM_DISPATCH_KINDS,
    INSTANCE_KINDS,
    KERNEL_DISPATCH_KINDS,
    RECORD_WORDS,
    RMS,
    UPDATE,
    WORD_INSTANCE,
    WORD_WORKERS,
    dispatch_alternatives,
    parse_dispatch,
    program_launches,
    program_words,
    read_trace,
    step_program,
)
from bcir.tests import native_decoder_fixtures as nf

#: The host's C compiler, or None: the C witnesses' one capability.
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()

#: An AMX host's masks: the bf16 GEMM's family (portable, AVX-512, AMX) and the fused GEMM's.
_MASKS = {"gemm": 0b1101, "attention": 0b0111}
#: The plan corpus's first decoder, a call of two steps of two micro-batches, under bf16 (whose
#: workspace holds its groups) -- every dispatch kind but a provider's has jobs in its program.
_SPEC, _BATCH, _TIME = nf.PLAN_CORPUS[1]
_BF16 = dict(contract=nat._CONTRACTS["bf16"], workspace_bytes=1 << 24, steps=2, accumulation=2)


def _expect_refused(label, fn, reason=None):
    """`fn` refused (ValueError) -- by the rule whose message holds `reason`, where one is named:
    two rules refusing the same input must not stand in for each other."""
    try:
        fn()
    except ValueError as exc:
        assert reason is None or reason in str(exc), (label, str(exc))
        return
    raise AssertionError(f"{label} admitted")


def _labelled(**decisions):
    """The program for `decisions` and each launch's dispatch label."""
    labels: list = []
    program = step_program(_SPEC, _BATCH, _TIME, **decisions, dispatch_labels=labels)
    return program, labels


def test_the_host_admits_its_alternatives_and_no_other():
    """`dispatch_alternatives` names, for every dispatch kind, each instance its family admits but
    the widest (the default) and each worker count below the pool's threads; a GEMM's family is the
    contract's, an attention's its products'; a provider's GEMMs name no instance; one thread and a
    portable-only family name nothing; a thread count that is not a positive integer is refused."""
    alts = dispatch_alternatives(threads=3, instances=_MASKS)
    assert tuple(alts) == DISPATCH_KINDS and len(set(DISPATCH_KINDS)) == len(DISPATCH_KINDS)
    assert set(INSTANCE_KINDS) == set(GEMM_DISPATCH_KINDS) | {"attention", "attention.vjp"}
    for kind in DISPATCH_KINDS:
        if kind in GEMM_DISPATCH_KINDS:
            want = ("i1", "i4", "t1", "t2")  # AMX the widest: the default's
        elif kind in ("attention", "attention.vjp"):
            want = ("i1", "i2", "t1", "t2")
        else:
            want = ("t1", "t2")
        assert alts[kind] == want, (kind, alts[kind])
    provider = dispatch_alternatives(threads=2, instances=_MASKS, provider_mm=True)
    assert all(provider[k] == ("t1",) for k in GEMM_DISPATCH_KINDS)
    assert provider["attention"] == ("i1", "i2", "t1")
    lone = dispatch_alternatives(threads=1, instances={"gemm": 1, "attention": 1})
    assert all(a == () for a in lone.values())
    # the decisions a trainer states pass whole: the rest are ignored
    assert dispatch_alternatives(threads=3, instances=_MASKS, contract=16, steps=2) == alts
    for bad in (0, -1, True, 1.0, "2"):
        _expect_refused(f"threads {bad!r}",
                        lambda b=bad: dispatch_alternatives(threads=b, instances=_MASKS),
                        "threads must be a positive integer")  # fmt: skip


def test_a_dispatch_takes_its_alternatives_and_nothing_else():
    """Each alternative of each kind writes its value into words 41 (an instance) or 42 (workers)
    of every record of every job its dispatch label names, and into no other word of any record;
    every launch is labelled with its dispatch kind (a CHECK with none); and a dispatch outside the
    grammar -- not a mapping, a kind or alternative misspelled, an instance on a kernel without
    one or on a provider's GEMM, an instance the host's family lacks, more workers than the pool's
    threads, labels already filled -- is refused by its own rule."""
    decisions = dict(_BF16, threads=3, instances=_MASKS)
    base, labels = _labelled(**decisions)
    assert len(labels) == len(base)
    for launch, label in zip(base, labels):
        kind = launch.records[0].kind
        assert (label is None) == (kind == CHECK), (label, kind)
        assert label is None or label in DISPATCH_KINDS
        if kind == GEMM:
            assert label in GEMM_DISPATCH_KINDS
    held = {label for label in labels if label is not None}
    # the panels' kernels under the panels' bench; a provider's SiLU and its product under one
    others = set()
    for args, kw in (
        (nf.PANELS_BENCH, dict(contract=16, workspace_bytes=1 << 26)),
        (nf.PLAN_CORPUS[1], dict(contract=0, provider_mm=True, provider_silu=True)),
    ):
        more: list = []
        step_program(*args, **kw, dispatch_labels=more)
        others |= {label for label in more if label is not None}
    rest = {"silu_mul.panels", "swiglu.vjp.panels", "provider.silu", "mul"}
    assert held == set(DISPATCH_KINDS) - rest and rest <= others, sorted(set(DISPATCH_KINDS) - held)
    words = program_words(base)
    assert all(w == 0 for i in range(0, len(words), RECORD_WORDS)
               for w in words[i + WORD_INSTANCE : i + RECORD_WORDS])  # fmt: skip
    for kind, alternatives in dispatch_alternatives(**decisions).items():
        for alternative in alternatives:
            what, value = parse_dispatch(kind, alternative)
            program, got = _labelled(**decisions, dispatch={kind: alternative})
            assert got == labels, (kind, alternative)
            for launch, was, label in zip(program, base, labels):
                want = (tuple(replace(r, **{what: value}) for r in was.records)
                        if label == kind else was.records)  # fmt: skip
                assert launch.records == want and launch.optional == was.optional, (kind, label)
    every = {kind: alts[0] for kind, alts in dispatch_alternatives(**decisions).items() if alts}
    program, _ = _labelled(**decisions, dispatch=every)
    # the words: the instance at 41, the workers at 42 (the header's), and a trace's reading of them
    words = program_words(program)
    flat = [r for launch in program for r in launch.records]
    assert len(words) == RECORD_WORDS * len(flat) and (WORD_INSTANCE, WORD_WORKERS) == (41, 42)
    for i, r in enumerate(flat):
        assert words[i * RECORD_WORDS + 41 : i * RECORD_WORDS + 43] == [r.instance, r.workers]
    assert [r for _, r in read_trace(words, len(flat))] == flat
    for launch, label in zip(program, labels):
        for r in launch.records:
            what, value = parse_dispatch(label, every[label]) if label else ("instance", 0)
            assert getattr(r, what) == value and getattr(
                r, "workers" if what == "instance" else "instance") == 0  # fmt: skip
    gemm = GEMM_DISPATCH_KINDS[0]
    for label, dispatch, reason in (
        ("a list", [("rms", "t1")], "dispatch maps a dispatch kind"),
        ("no kind", {"nope": "t1"}, "no dispatch kind"),
        ("a kind not a string", {1: "t1"}, "no dispatch kind"),
        ("the check", {"check": "t1"}, "no dispatch kind"),
        ("an update's pass", {"update.pass": "t1"}, "no dispatch kind"),
        ("not a string", {"rms": 1}, "is not i<bit> or t<workers>"),
        ("empty", {"rms": ""}, "is not i<bit> or t<workers>"),
        ("a letter alone", {"rms": "t"}, "is not i<bit> or t<workers>"),
        ("another letter", {"rms": "w1"}, "is not i<bit> or t<workers>"),
        ("no workers", {"rms": "t0"}, "is not i<bit> or t<workers>"),
        ("a leading zero", {"rms": "t01"}, "is not i<bit> or t<workers>"),
        ("a sign", {"rms": "t+1"}, "is not i<bit> or t<workers>"),
        ("a space", {"rms": "t1 "}, "is not i<bit> or t<workers>"),
        ("non-ASCII digits", {"rms": "t١"}, "is not i<bit> or t<workers>"),
        ("two bits", {gemm: "i3"}, "names no instance (1, 2, 4 or 8)"),
        ("a bit past AMX", {gemm: "i16"}, "names no instance (1, 2, 4 or 8)"),
        ("an instance on a kernel", {"rms": "i1"}, "names no instance (only a GEMM"),
        ("an instance on an update", {"update": "i1"}, "names no instance (only a GEMM"),
        ("AVX2 the bf16 family lacks", {gemm: "i2"}, "admits no instance 2"),
        ("AMX the attention lacks", {"attention": "i8"}, "admits no instance 8"),
        ("past the pool", {"rms": "t4"}, "past the pool's 3 threads"),
    ):
        _expect_refused(label, lambda d=dispatch: step_program(
            _SPEC, _BATCH, _TIME, **decisions, dispatch=d), reason)  # fmt: skip
    _expect_refused("a provider's GEMM", lambda: step_program(
        _SPEC, _BATCH, _TIME, **dict(decisions, contract=0, workspace_bytes=0, provider_mm=True),
        dispatch={gemm: "i1"}), "a provider's GEMM names no instance")  # fmt: skip
    for bad in ([None], (), "x"):
        _expect_refused(f"labels {bad!r}", lambda b=bad: step_program(
            _SPEC, _BATCH, _TIME, **decisions, dispatch_labels=b), "dispatch_labels is an empty")  # fmt: skip


def test_the_default_dispatch_is_the_program_before_it():
    """No dispatch, an empty one, and every kind's widest instance named nowhere state the same
    program, record for record, under every contract the corpus traces: a program of the defaults
    has zero in words 41 and 42 of every record (non-disturbance: the words a program had before
    PX-3b), and the five spare words stay zero under any dispatch."""
    for contract in nf.TRACE_CONTRACTS:
        bits = nat._CONTRACTS[contract]
        decisions = dict(contract=bits, workspace_bytes=1 << 24 if bits & 16 else 0, steps=2,
                         accumulation=2, threads=2, instances=_MASKS)  # fmt: skip
        base = step_program(_SPEC, _BATCH, _TIME, **decisions)
        assert step_program(_SPEC, _BATCH, _TIME, **decisions, dispatch=None) == base
        assert step_program(_SPEC, _BATCH, _TIME, **decisions, dispatch={}) == base
        assert all(r.instance == 0 and r.workers == 0 for launch in base for r in launch.records)
        for pick in (0, -1):
            every = {kind: alts[pick] for kind, alts in dispatch_alternatives(**decisions).items()
                     if alts}  # fmt: skip
            words = program_words(step_program(_SPEC, _BATCH, _TIME, **decisions, dispatch=every))
            assert len(words) == len(program_words(base))
            for i in range(0, len(words), RECORD_WORDS):
                assert not any(words[i + WORD_WORKERS + 1 : i + RECORD_WORDS]), contract


def test_the_model_resolves_the_defaults_as_the_trace_records_them():
    """`program_launches` states each record as the probe traces it: a GEMM of BCIR's entries
    with no instance named as its family's widest, an attention as its products' widest, a
    provider's GEMM and every other kernel with none; a job with no workers named on all the pool's
    threads; a named instance or worker count as it stands. Without the host's instances a default
    instance stays unresolved; the workers resolve to the threads given."""
    decisions = dict(_BF16, threads=3, instances=_MASKS)
    program = step_program(_SPEC, _BATCH, _TIME, **decisions,
                           dispatch={"qkv.fwd": "i1", "rms": "t2", "attention": "i2"})  # fmt: skip
    flat = [r for launch in program_launches(program, threads=3, instances=_MASKS)
            for r in launch.records]  # fmt: skip
    assert all(r.kind != CHECK for r in flat)
    assert {r.instance for r in flat if r.kind == GEMM} == {1, 8}  # AMX the bf16 family's widest
    assert {r.instance for r in flat if r.kind == ATTN} == {2}
    assert {r.instance for r in flat if r.kind == ATTN_VJP} == {4}  # the products' widest
    assert {r.instance for r in flat if r.kind not in (GEMM, ATTN, ATTN_VJP)} == {0}
    assert {r.workers for r in flat if r.kind == RMS} == {2}
    assert {r.workers for r in flat if r.kind != RMS} == {3}
    unresolved = [r for launch in program_launches(program, threads=3) for r in launch.records]
    assert {r.instance for r in unresolved if r.kind == GEMM} == {0, 1}
    assert {r.workers for r in unresolved} == {2, 3}
    provider = step_program(_SPEC, _BATCH, _TIME, contract=0, provider_mm=True, threads=2,
                            instances=_MASKS)  # fmt: skip
    gemms = [r for launch in program_launches(provider, threads=2, instances=_MASKS)
             for r in launch.records if r.kind == GEMM]  # fmt: skip
    assert gemms and all(r.flags & 15 == E_PROVIDER and r.instance == 0 for r in gemms)


def _shape(**extra):
    return {"vocab": _SPEC.vocab_size, "d_model": _SPEC.d_model, "n_heads": _SPEC.n_heads,
            "n_layers": _SPEC.n_layers, "d_ff": _SPEC.d_ff, "kv_heads": _SPEC.kv_heads,
            "batch": _BATCH, "time": _TIME, "threads": 3, "steps": 1, "recompute": 0,
            "workspace_bytes": 0, "contract": "bf16"} | extra  # fmt: skip


def test_a_dispatch_table_names_its_instances_and_has_one_spelling():
    """A table that measured the dispatch names the instances it measured on in its shape, and
    holds only alternatives that host admits; it round-trips through its one JSON spelling. A
    dispatch entry in a table without instances, instances in a table without a dispatch entry, a
    family missing or added, a mask without the portable bit, past AMX or not an integer, an
    alternative the shape's host lacks (the widest, a bit outside its family, workers at the
    threads), a dispatch kind that does not exist, and a second spelling are refused, each by its
    own rule."""
    shape = _shape(instances=dict(_MASKS))
    entries = (ChoiceEntry("dispatch:rms", "t1", 240, 230, 250, 5),
               ChoiceEntry("dispatch:qkv.fwd", "i4", 250, 240, 260, 5),
               ChoiceEntry("qkv.vjp", "pair", 250, 240, 270, 5))  # fmt: skip
    table = ChoiceTable(1, shape, {"probe": "test"}, entries)
    assert ChoiceTable.from_json(table.to_json()) == table
    assert [e.kind for e in table.entries] == ["dispatch:qkv.fwd", "dispatch:rms", "qkv.vjp"]
    assert DISPATCH_CHOICE_KINDS == tuple("dispatch:" + k for k in DISPATCH_KINDS)
    assert dispatch_kind("dispatch:rms") == "rms" and dispatch_kind("rms") is None
    assert dispatch_kind("dispatch:nope") is None and dispatch_kind(None) is None
    plain = ChoiceTable(1, _shape(), {}, (entries[2],))  # a PX-3 table, as it was spelled
    assert "instances" not in json.loads(plain.to_json())["shape"]
    for label, shape_, entries_, reason in (
        ("a dispatch entry without instances", _shape(), entries[:1], "names the instances it"),
        ("instances without a dispatch entry", shape, entries[2:], "names instances only when"),
        ("a family missing", _shape(instances={"gemm": 13}), entries[:1], "each family's mask"),
        ("a family added", _shape(instances=dict(_MASKS, cpu=1)), entries[:1], "each family's"),
        ("no portable bit", _shape(instances=dict(_MASKS, gemm=12)), entries[:1], "each family"),
        ("past AMX", _shape(instances=dict(_MASKS, gemm=17)), entries[:1], "each family's mask"),
        ("a mask a bool", _shape(instances=dict(_MASKS, gemm=True)), entries[:1], "each family"),
        ("not a mapping", _shape(instances=[13, 7]), entries[:1], "each family's mask"),
        (
            "the widest",
            shape,
            (ChoiceEntry("dispatch:o.vjp", "i8", 250, 240, 260, 5),),
            "no kind 'dispatch:o.vjp' with the alternative 'i8'",
        ),
        (
            "a bit the family lacks",
            shape,
            (ChoiceEntry("dispatch:o.vjp", "i2", 250, 240, 260, 5),),
            "no kind 'dispatch:o.vjp' with the alternative 'i2'",
        ),
        (
            "workers at the threads",
            shape,
            (ChoiceEntry("dispatch:rms", "t3", 250, 240, 260, 5),),
            "no kind 'dispatch:rms' with the alternative 't3'",
        ),
        (
            "an instance on a kernel",
            shape,
            (ChoiceEntry("dispatch:rms", "i1", 250, 240, 260, 5),),
            "no kind 'dispatch:rms' with the alternative 'i1'",
        ),
        (
            "no such kind",
            shape,
            (ChoiceEntry("dispatch:nope", "t1", 250, 240, 260, 5),),
            "no kind 'dispatch:nope'",
        ),
    ):
        _expect_refused(label, lambda s=shape_, e=entries_: ChoiceTable(1, s, {}, e), reason)
    text = table.to_json()
    doc = json.loads(text)
    doc["shape"]["instances"] = {"gemm": 13, "attention": 7}  # the families in another order
    for label, other in (
        ("another order", json.dumps(doc, separators=(",", ":"))),
        ("spaces", text.replace(",", ", ")),
    ):
        _expect_refused(label, lambda o=other: ChoiceTable.from_json(o), "one spelling")


def test_choose_dispatch_takes_only_alternatives_resolved_faster():
    """`choose_dispatch` takes, per dispatch kind, the alternative whose whole interval lies below
    256 with the lowest mean (the narrower interval on a tie, then the table's order) and leaves
    every other kind to its default; it reads no site entry, and `choose` reads no dispatch entry.
    No table, or one that measured no dispatch, is no dispatch; a table of another shape -- other
    instances among it -- or another decoder, or not a table, is refused."""
    shape = _shape(instances=dict(_MASKS))
    entries = (
        ChoiceEntry("dispatch:rms", "t1", 200, 180, 250, 5),  # taken
        ChoiceEntry("dispatch:rms", "t2", 190, 120, 260, 5),  # lower mean, crosses 256
        ChoiceEntry("dispatch:qkv.fwd", "i1", 300, 280, 320, 5),  # slower
        ChoiceEntry("dispatch:qkv.fwd", "i4", 220, 200, 240, 5),  # taken
        ChoiceEntry("dispatch:rope", "t1", 200, 190, 240, 5),
        ChoiceEntry("dispatch:rope", "t2", 200, 195, 230, 5),  # a tie: the narrower interval
        ChoiceEntry("dispatch:embed", "t1", 200, 190, 230, 5),  # a tie in both: the table's order
        ChoiceEntry("dispatch:embed", "t2", 200, 190, 230, 5),
        ChoiceEntry("dispatch:attention", "i2", 256, 250, 262, 5),  # no faster
        ChoiceEntry("qkv.vjp", "calls", 100, 90, 110, 5),  # a site's
    )
    table = ChoiceTable(1, shape, {}, entries)
    assert choose_dispatch(table, _SPEC, shape) == {
        "embed": "t1", "qkv.fwd": "i4", "rms": "t1", "rope": "t2"}  # fmt: skip
    sites = choose(table, _SPEC, shape)
    assert {s: a for s, a in sites.items() if s.endswith("qkv.vjp")} == {
        f"L{i}.qkv.vjp": "calls" for i in range(_SPEC.n_layers)}  # fmt: skip
    assert choose_dispatch(None, _SPEC, shape) == {}
    sites_only = ChoiceTable(1, _shape(), {}, entries[-1:])
    assert choose_dispatch(sites_only, _SPEC, _shape()) == {}
    step_program(_SPEC, _BATCH, _TIME, contract=16, workspace_bytes=1 << 24, threads=3,
                 instances=_MASKS, dispatch=choose_dispatch(table, _SPEC, shape))  # fmt: skip
    from bcir.frontends.models.decode import DecoderSpec

    other = DecoderSpec(16, 8, 2, 3, 16, activation="silu_gate")
    for label, fn, reason in (
        ("other instances", lambda: choose_dispatch(
            table, _SPEC, _shape(instances=dict(_MASKS, gemm=5))), "the table was measured at"),
        ("instances left out", lambda: choose_dispatch(table, _SPEC, _shape()),
         "the table was measured at"),
        ("another decoder", lambda: choose_dispatch(table, other, shape), "another decoder"),
        ("not a table", lambda: choose_dispatch({}, _SPEC, shape), "takes a ChoiceTable"),
    ):  # fmt: skip
        _expect_refused(label, fn, reason)


# --- the C witnesses --------------------------------------------------------------------------------


def _cases(threads):
    """One case of the plan corpus's untied decoder per traced contract, two steps of two."""
    spec, batch, time_ = nf.PLAN_CORPUS[1]
    return [nf.TraceCase(spec, batch, time_, contract, threads=threads, steps=2, accumulation=2)
            for contract in nf.TRACE_CONTRACTS]  # fmt: skip


def test_every_dispatch_trains_the_c_steps_bits_and_is_traced_as_stated():
    """Under every contract the corpus traces (each GEMM and attention family), at one thread and
    at three: every alternative this host admits for every dispatch kind, every kind at its first
    alternative and every kind at its last, run from its program, leaves the C step's state and
    events on the same case, and its trace is the launches its program states as the host resolves
    them -- the instance each GEMM and attention job ran, the workers each job ran on."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.trace_library(Path(tmp) / "trace")
        if path is None:
            return
        wrong, runs, named = [], 0, set()
        for threads in (1, 3):
            for case in _cases(threads):
                step = nf.trace_case(path, case)
                trainer, decisions = nf._case_trainer(path, case)
                trainer.close()
                alternatives = dispatch_alternatives(**decisions)
                dispatches = [{kind: a} for kind, alts in alternatives.items() for a in alts]
                for pick in (0, -1):
                    every = {kind: alts[pick] for kind, alts in alternatives.items() if alts}
                    if every:
                        dispatches.append(every)
                for kind in sorted(INSTANCE_KINDS):  # the default's own instance, named
                    if not (case.provider and kind in GEMM_DISPATCH_KINDS):
                        family = "gemm" if kind in GEMM_DISPATCH_KINDS else "attention"
                        mask = decisions["instances"][family]
                        dispatches.append({kind: f"i{max(b for b in (1, 2, 4, 8) if mask & b)}"})
                for dispatch in dispatches:
                    run = nf.program_case(path, case, {"dispatch": dispatch})
                    problems = nf.program_problems(step, run)
                    runs += 1
                    named.update((parse_dispatch(k, a)[0], a) for k, a in dispatch.items())
                    if problems:
                        wrong.append((case.contract, threads, dispatch, problems[:2]))
        assert not wrong, wrong[:4]
        assert {what for what, _ in named} == {"instance", "workers"}, named
        assert runs >= 2 * len(nf.TRACE_CONTRACTS) * 2


def test_every_dispatch_of_the_corpus_trains_the_c_steps_bits():
    """`train.plan.dispatch.mismatch`: each state case of the trace corpus, under every dispatch
    kind switched alone to the portable instance and to its family's widest named (a GEMM of BCIR's
    entries, an attention) and to each worker count below its threads, and three seeded draws of
    one a kind, run from its program on the trace build, traces exactly the dispatched program as
    the host resolves it and trains the C step's state and events bit for bit -- 0 of
    `dispatch_run_count()` runs, the same count on every host; and the runs are dispatches: most
    of them run another program than the C step's own."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        traced = nf.trace_library(Path(tmp) / "trace")
        if traced is None:
            return
        bad, runs, notes, other = nf.dispatch_mismatch(traced)
        assert (bad, runs) == (0, nf.dispatch_run_count()), notes[:3]
        assert 2 * other > runs, (other, runs)  # the dispatches ran: not the C step's program again


def _raw(trainer, words, case):
    """bcir_decoder_run_program's status on `words` for `case`'s tokens, and the count it left
    (lent as 777, so a refusal that touched it shows)."""
    from bcir.tests.test_native_run_program import _raw as raw

    return raw(trainer, words, case)


def _snapshot(trainer):
    from bcir.tests.test_native_run_program import _snapshot as snapshot

    return snapshot(trainer)


def _refusals(records, decisions):
    """The near misses of the dispatch rules on a dispatched program's records: each an edited
    copy, by the rule it breaks."""
    job = next(r[1] for i, r in enumerate(records[1:], 1)
               if r[0] == GEMM and records[i - 1][:2] == r[:2])  # fmt: skip
    gemms = [i for i, r in enumerate(records) if r[0] == GEMM and r[1] == job]
    rms = next(i for i, r in enumerate(records) if r[0] == RMS)
    update = next(i for i, r in enumerate(records) if r[0] == UPDATE)
    check = next(i for i, r in enumerate(records) if r[0] == CHECK)
    assert len(gemms) > 1

    def edited(at, word, value):
        return [r if i not in at else r[:word] + [value] + r[word + 1 :]
                for i, r in enumerate(records)]  # fmt: skip

    out = {
        "an instance the family never admits": edited(gemms, WORD_INSTANCE, 8),
        "two instances": edited(gemms, WORD_INSTANCE, 3),
        "an instance past AMX": edited(gemms, WORD_INSTANCE, 16),
        "an instance on a kernel": edited([rms], WORD_INSTANCE, 1),
        "an instance on the update": edited([update], WORD_INSTANCE, 1),
        "workers past the pool": edited([rms], WORD_WORKERS, decisions["threads"] + 1),
        "workers past 2^32": edited([rms], WORD_WORKERS, (1 << 32) + 1),
        "workers on a CHECK": edited([check], WORD_WORKERS, 1),
        "a job split on its instance": edited(gemms[1:], WORD_INSTANCE, 0),
        "a job split on its workers": edited(gemms[1:], WORD_WORKERS, 1),
        "the first spare word set": edited([rms], WORD_WORKERS + 1, 1),
    }
    return out


def _named(records, kind, value):
    """The records with every record of `kind` naming instance `value`."""
    return [r if r[0] != kind else r[:WORD_INSTANCE] + [value] + r[WORD_INSTANCE + 1 :]
            for r in records]  # fmt: skip


def test_a_program_the_host_cannot_dispatch_is_refused_with_nothing_written():
    """On a v1 trainer of two threads (v1's GEMMs: no AMX on any host), a program whose dispatch
    names an instance the family never admits, two instances, an instance past AMX, an instance on
    a kernel or on the update, more workers than the pool's threads -- by one, or by 2^32 + 1, which
    a 32-bit cap would read as one -- workers on a CHECK, a GEMM job whose records disagree on
    either word, or the spare word after them set; on a provider's trainer, an instance on the
    provider's GEMMs; and under the bf16 attention, AVX2 (its products' family has none) -- is
    refused (status 1) before anything runs: no byte of the state written, not even the completed
    count. Each program it edits runs (so each refusal is its edit's)."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib = nf.checkout_library(Path(tmp) / "plain")
        if lib is None:
            return
        spec, batch, time_ = nf.PLAN_CORPUS[0]
        untied = nf.PLAN_CORPUS[1]
        cases = (
            nf.TraceCase(spec, batch, time_, "v1", threads=2, steps=2, accumulation=2),
            nf.TraceCase(*untied, "v1", threads=2, provider="mm"),
            nf.TraceCase(spec, batch, time_, "bf16+exp+attnbf16", threads=1),
        )
        wrong, seen = [], 0
        for case in cases:
            trainer, decisions = nf._case_trainer(lib, case)
            try:
                assert not decisions["instances"]["gemm"] & 8 or case.contract != "v1"
                dispatch = {
                    kind: "i1" if kind in INSTANCE_KINDS else "t1" for kind in DISPATCH_KINDS
                    if kind != "provider.silu" and not (case.provider and kind in GEMM_DISPATCH_KINDS)
                }  # fmt: skip
                words = program_words(
                    step_program(case.spec, case.batch, case.time, **decisions, dispatch=dispatch)
                )
                w = RECORD_WORDS
                records = [list(words[i : i + w]) for i in range(0, len(words), w)]
                if case.provider:  # the provider's GEMMs are not BCIR's to name
                    refusals = {"an instance on a provider's GEMM": _named(records, GEMM, 1)}
                elif "attnbf16" in case.contract:  # its products' family: no AVX2
                    assert not decisions["instances"]["attention"] & 2
                    refusals = {"AVX2 the bf16 attention lacks": _named(records, ATTN, 2)}
                else:
                    refusals = _refusals(records, decisions)
                before = _snapshot(trainer)
                for name, edit in refusals.items():
                    rc, done = _raw(trainer, [x for r in edit for x in r], case)
                    seen += 1
                    if rc != 1 or done != 777 or _snapshot(trainer) != before:
                        wrong.append((case.contract, name, rc, done))
                assert _raw(trainer, words, case) == (0, case.steps), case  # the program runs
            finally:
                trainer.close()
        assert not wrong, wrong
        assert seen == 13, seen


def test_a_run_leaves_the_pool_as_it_found_it():
    """A run of a program caps each job's workers as the job names and leaves the pool's cap as it
    found it, however the run ends: on a trainer of three threads whose pool the caller capped at
    two, a program of the defaults runs every job on all three threads (0 is the pool's threads,
    not the caller's cap), one capping every job at one runs each on one, and after either the cap
    reads two again; the fixed step after them runs on the caller's two."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.trace_library(Path(tmp) / "trace")
        if path is None:
            return
        spec, batch, time_ = nf.PLAN_CORPUS[0]
        case = nf.TraceCase(spec, batch, time_, "kblock+exp+reduce", threads=3)
        trainer, decisions = nf._case_trainer(path, case)
        try:
            lib = trainer.lib
            cap = lib.bcir_tensor_pool_cap
            cap.argtypes, cap.restype = [ct.c_void_p, ct.c_uint], ct.c_uint
            pool = trainer.state.pool
            assert pool and cap(pool, 2) == 0
            begin, end = lib.bcir_decoder_trace_probe_begin, lib.bcir_decoder_trace_probe_end
            begin.argtypes, begin.restype = [ct.c_void_p, ct.c_void_p, ct.c_size_t], None
            end.argtypes, end.restype = [ct.POINTER(ct.c_size_t)], ct.c_size_t
            tokens, targets, optimizer = nf._case_inputs(case)

            def traced(run):
                capacity = 4096
                words = (ct.c_uint64 * (capacity * RECORD_WORDS))()
                begin(ct.addressof(trainer.state), ct.addressof(words), capacity)
                try:
                    run()
                finally:
                    dropped = ct.c_size_t()
                    count = end(ct.byref(dropped))
                assert dropped.value == 0
                return {r.workers for _, r in read_trace(words, count)}

            one = {kind: "t1" for kind in DISPATCH_KINDS}
            for dispatch, want in (({}, {3}), (one, {1})):
                program = step_program(spec, batch, time_, **decisions, dispatch=dispatch)
                seen = traced(lambda p=program: trainer.run_program(
                    program_words(p), tokens, targets, steps=1, optimizer=optimizer))  # fmt: skip
                assert seen == want, (dispatch and "t1", seen)
                assert cap(pool, 2) == 2  # the caller's cap, as it left it
            assert traced(lambda: trainer.run(tokens, targets, steps=1, optimizer=optimizer)) == {2}
            assert cap(pool, 0) == 2
        finally:
            trainer.close()


def test_a_dispatch_measurement_is_a_table_at_the_hosts_instances():
    """`measure` with dispatch kinds on a clocked trainer of two threads returns a table whose
    shape names the host's instances, with one entry per alternative the host admits for each
    kind asked (each a worker count, and each instance but the widest); its readings name every
    arm, each reading the kind's own jobs; it round-trips, `choose_dispatch` on it is a dispatch
    the program takes, and a dispatch kind that does not exist is refused."""
    if _CC is None:
        return
    spec, batch, time_ = nf.PLAN_CORPUS[1]
    case = nf.TraceCase(spec, batch, time_, "bf16+exp+attnbf16", threads=2)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        timed = nf.time_library(Path(tmp) / "time")
        if timed is None:
            return
        trainer, decisions = nf._case_trainer(timed, case)
        try:
            kinds = ("dispatch:qkv.fwd", "dispatch:attention", "dispatch:rms", "head.vjp")
            raw: list = []
            table = measure(trainer, rounds=5, kinds=kinds, host={"probe": "test"}, raw=raw)
            shape = trainer_shape(trainer, instances=True)
            assert table.shape == shape and shape["instances"] == trainer.instance_families()
            alts = dispatch_alternatives(**decisions)
            want = {(k, a) for k in kinds if dispatch_kind(k) for a in alts[dispatch_kind(k)]}
            want.add(("head.vjp", "calls"))
            assert {(e.kind, e.alternative) for e in table.entries} == want
            assert ("dispatch:rms", "t1") in want  # two threads: every host has an arm
            arms = {"default"} | {f"{k}={a}" for k, a in want}
            assert len(raw) == 5 and all(set(r["order"]) == arms for r in raw)
            for r in raw:
                for name, ns in r["site_ns"].items():
                    assert 0 < ns < r["total_ns"][name.split("@")[0]], (name, r)
            assert ChoiceTable.from_json(table.to_json()) == table
            dispatch = choose_dispatch(table, spec, shape)
            step_program(spec, batch, time_, **trainer.program_decisions(), dispatch=dispatch)
            _expect_refused("no such dispatch kind",
                            lambda: measure(trainer, rounds=5, kinds=("dispatch:nope",)),
                            "kinds names kinds")  # fmt: skip
        finally:
            trainer.close()


assert KERNEL_DISPATCH_KINDS  # the kernels' kinds the alternatives above range over
