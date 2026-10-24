"""PX-3 (CHOOSE): the step program's per-site choices taken from measured candidates.

`native_launch.step_program(sites=...)` takes each choice site's alternative -- each layer's q/k/v
and gate/up projections, its SwiGLU block's panels, the VJPs of its down, o, gate/up and q/k/v
projections, and the head's VJP -- and labels every launch with the site it belongs to;
`native_choose` measures the alternatives in place on a library built with the job clock
(BCIR_DT_TIME_PROBE), keeps the result as a frozen, generation-tagged integer table, and chooses
from it. The witnesses hold, without a compiler: a site takes exactly its alternatives; switching a
kind changes only the launches its scope labels, and each label covers only its own kernels -- so
a measurement reads every launch a choice changes and nothing else; the estimator's numbers; the
table's one spelling and its refusals; the choice taking only alternatives resolved faster. With
one: every alternative, and random mixes of them, trains the C step's bits from its program and
from its pack (planned by K_BCIR, lowered by the C); the job clock records each job once, at its
first record, writes nothing else, and leaves the state the library's; a measurement is a table at
the trainer's shape.
"""

import ctypes as ct
import itertools
import json
import random
import tempfile
from array import array
from pathlib import Path

from bcir.frontends.models.decode import DecoderSpec
from bcir.hosted.models import native as nat
from bcir.hosted.models.native_choose import (
    CHOICE_KINDS,
    Q8,
    ChoiceEntry,
    ChoiceTable,
    choose,
    estimate,
    kind_of,
    measure,
    scope,
    timed_run,
    trainer_shape,
)
from bcir.hosted.models.native_launch import (
    HEAD_SITE,
    KIND_NAMES,
    RECORD_WORDS,
    SITE_KINDS,
    choice_sites,
    default_sites,
    program_words,
    step_program,
)
from bcir.hosted.models.native_pack import pack_artifacts, training_pack
from bcir.tests import native_decoder_fixtures as nf

#: The host's C compiler, or None: the C witnesses' one capability.
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()

#: The panels' bench (n, d and f multiples of 32) under the bf16 contract with the panels' need
#: lent, where every kind's alternatives state programs of their own, and under kblock.
_BF16 = dict(contract=nat._CONTRACTS["bf16"], workspace_bytes=1 << 30, panels_need=0)
_KBLOCK = dict(contract=nat._CONTRACTS["kblock+exp+reduce"])


def _expect_refused(label, fn, reason=None):
    """`fn` refused (ValueError) -- by the rule whose message holds `reason`, where one is named:
    two rules refusing the same input must not stand in for each other."""
    try:
        fn()
    except ValueError as exc:
        assert reason is None or reason in str(exc), (label, str(exc))
        return
    raise AssertionError(f"{label} admitted")


def _switched(spec, kind, alternative):
    return {site: alternative for site in choice_sites(spec) if kind_of(site) == kind}


def _labelled(spec, batch, time_, **decisions):
    labels = []
    return step_program(spec, batch, time_, **decisions, labels=labels), labels


def test_a_site_takes_its_alternatives_and_nothing_else():
    """Every site of the program, the layers' in order then the head's, each with its kind's
    alternatives; each combination of the booleans takes the alternatives they name at every site,
    and its program is the one those default sites state (no sites, an empty map and the full
    default map alike); and a site the program lacks,
    an alternative its kind lacks, a map that is not one, or labels that are not an empty list are
    refused."""
    spec, batch, time_ = nf.PANELS_BENCH
    sites = choice_sites(spec)
    assert list(sites) == [f"L{l}.{k}" for l in range(spec.n_layers) for k in SITE_KINDS] + [
        HEAD_SITE
    ]
    assert all(sites[s] == CHOICE_KINDS[kind_of(s)] for s in sites)
    for decisions in (_BF16, _KBLOCK):
        for group, pair, panels in itertools.product((True, False), repeat=3):
            flags = dict(group=group, pair=pair, panels=panels)
            base = step_program(spec, batch, time_, **decisions, **flags)
            defaults = default_sites(spec, **flags)
            grouped = "group" if group else "pair" if pair else "calls"
            want = {"qkv.fwd": "group" if group else "calls", "gateup.fwd": "group" if group
                    else "calls", "panels": "panels" if panels else "plain",
                    "down.vjp": "pair" if pair else "calls", "gateup.vjp": grouped,
                    "o.vjp": "pair" if pair else "calls", "qkv.vjp": grouped,
                    HEAD_SITE: "pair" if pair else "calls"}  # fmt: skip
            assert all(defaults[site] == want[kind_of(site)] for site in sites), (flags, defaults)
            for given in (None, {}, defaults):
                assert step_program(spec, batch, time_, **decisions, **flags, sites=given) == base
            assert step_program(spec, batch, time_, **decisions, sites=defaults) == base
    for label, sites_ in (
        ("a layer the decoder lacks", {f"L{spec.n_layers}.qkv.vjp": "pair"}),
        ("an alternative its kind lacks", {"L0.qkv.vjp": "panels"}),
        ("the head grouped", {HEAD_SITE: "group"}),
        ("a site of no kind", {"L0.attention": "calls"}),
        ("a site named by a number", {0: "calls"}),
        ("a list of pairs", [("L0.qkv.vjp", "pair")]),
    ):
        _expect_refused(label, lambda s=sites_: step_program(spec, batch, time_, **_BF16, sites=s))
    for label, labels in (("labels already filled", [None]), ("labels a tuple", ())):
        _expect_refused(label, lambda x=labels: step_program(spec, batch, time_, labels=x, **_BF16))
    for site in ("L01.qkv.vjp", "L0", "l0.qkv.vjp", "L١.qkv.vjp", "L0.qkv"):
        _expect_refused(f"the site {site!r}", lambda s=site: kind_of(s))
    _expect_refused("a kind of no site", lambda: scope("attention"))


def test_a_kind_switched_changes_only_the_launches_its_scope_labels():
    """For every kind and alternative, at every site of the kind (and at one layer's alone), the
    switched program and the default one differ only in launches the kind's scope labels -- the
    panels' scope their block, the down VJP and the gate/up VJP -- and every choice under bf16 on
    the panels' bench states a program of its own: what a measurement reads in place is every
    launch the choice changes."""
    spec, batch, time_ = nf.PANELS_BENCH
    for decisions in (_BF16, _KBLOCK):
        base, base_labels = _labelled(spec, batch, time_, **decisions)
        assert len(base_labels) == len(base)
        defaults = default_sites(spec)
        for kind, alternatives in CHOICE_KINDS.items():
            names = scope(kind)

            def outside(program, labels, names=names):
                return [launch for launch, label in zip(program, labels)
                        if label is None or kind_of(label) not in names]  # fmt: skip

            for alternative in alternatives:
                one = next(site for site in choice_sites(spec) if kind_of(site) == kind)
                for sites in (_switched(spec, kind, alternative), {one: alternative}):
                    labels = []
                    program = step_program(spec, batch, time_, **decisions, sites=sites,
                                           labels=labels)  # fmt: skip
                    assert outside(program, labels) == outside(base, base_labels), (kind, sites)
                    if decisions is _BF16 and alternative != defaults[one]:
                        assert program != base, (kind, alternative)


def test_each_label_names_its_own_kernels():
    """A projection's site labels its GEMM jobs and nothing else, the panels' site the SwiGLU
    block's kernels (and the GEMM reading the panel): no launch of another kernel is ever read as a
    site's; and every site of the program labels launches of its own."""
    spec, batch, time_ = nf.PANELS_BENCH
    panels = {"gemm", "silu_mul", "silu_mul.panels", "swiglu.vjp", "swiglu.vjp.panels"}
    for decisions in (_BF16, _KBLOCK):
        for sites in (None, *(_switched(spec, k, a) for k in CHOICE_KINDS
                              for a in CHOICE_KINDS[k])):  # fmt: skip
            labels = []
            program = step_program(spec, batch, time_, **decisions, sites=sites, labels=labels)
            seen = set()
            for launch, label in zip(program, labels):
                if label is None:
                    continue
                seen.add(label)
                kinds = {KIND_NAMES[r.kind] for r in launch.records}
                allowed = panels if kind_of(label) == "panels" else {"gemm"}
                assert kinds <= allowed, (label, kinds)
            wanted = set(choice_sites(spec))
            if decisions is _KBLOCK:  # no panels outside the bf16 contract: the block plain
                wanted = {s for s in wanted if kind_of(s) != "panels"} | {
                    s for s in seen if kind_of(s) == "panels"
                }
            assert wanted <= seen, sorted(wanted - seen)


def test_the_estimate_is_the_paired_geometric_mean_and_its_interval():
    """`estimate` is the geometric mean of the paired ratios and its 95% Student-t interval, in Q8
    rounded half up -- at the numbers worked by hand, a quantile between the table's rows taken at
    the row below (the wider interval) -- and it refuses unpaired, short or non-positive rounds and
    times that are not integers."""
    assert estimate([100] * 5, [90] * 5) == (230, 230, 230)
    assert estimate([100] * 5, [90, 110, 100, 95, 105]) == (255, 231, 282)
    assert estimate([1000] * 16, [900 if i % 2 else 950 for i in range(16)]) == (237, 233, 240)
    assert estimate((7, 7, 7, 7, 7), (7, 7, 7, 7, 7)) == (Q8, Q8, Q8)
    # 33 degrees of freedom, between the table's 31 and 35: the 31's quantile (the wider interval)
    assert estimate([1000] * 34, [10000 if i % 2 else 100 for i in range(34)]) == (256, 113, 580)
    _expect_refused("four rounds", lambda: estimate([100] * 4, [90] * 4), "five paired rounds")
    for label, d, a, reason in (
        ("unpaired rounds", [100] * 5, [90] * 6, "paired rounds"),
        ("a zero time", [100] * 5, [90, 90, 0, 90, 90], "positive integer"),
        ("a negative time", [100, -1, 100, 100, 100], [90] * 5, "positive integer"),
        ("a float time", [100.0] * 5, [90] * 5, "positive integer"),
        ("a bool time", [True] * 5, [90] * 5, "positive integer"),
    ):
        _expect_refused(label, lambda d=d, a=a: estimate(d, a), reason)


def _shape(**over):
    spec, batch, time_ = nf.PANELS_BENCH
    shape = dict(vocab=spec.vocab_size, d_model=spec.d_model, n_heads=spec.n_heads,
                 n_layers=spec.n_layers, d_ff=spec.d_ff, kv_heads=spec.kv_heads, batch=batch,
                 time=time_, threads=4, steps=1, recompute=0, workspace_bytes=0,
                 contract="bf16")  # fmt: skip
    shape.update(over)
    return shape


def _table(entries, **over):
    return ChoiceTable(1, _shape(**over), {"cpu": "test", "cpus": 4}, tuple(entries))


def test_a_choice_table_has_one_spelling_and_refuses_the_rest():
    """A table round-trips through its JSON, in one order whatever order it was given, and its
    decoder refuses every other spelling and every value its rules exclude: a generation below one,
    a shape naming other fields or values outside them, a host value that is neither text nor an
    integer, an entry of another arity, kind or alternative, measured twice, an interval out of
    order or of fewer than five rounds, and anything not integers where integers are due."""
    a = ChoiceEntry("qkv.vjp", "pair", 240, 230, 250, 16)
    b = ChoiceEntry("o.vjp", "calls", 250, 240, 260, 16)
    table = _table([a, b])
    text = table.to_json()
    assert ChoiceTable.from_json(text) == table == _table([b, a])
    assert _table([b, a]).to_json() == text and table.entries == (b, a)
    doc = json.loads(text)
    respelled = (
        json.dumps(doc, indent=1, sort_keys=True),
        json.dumps(doc, sort_keys=False, separators=(",", ":")),
        text.replace('"cpus":4', '"cpus":4.0'),
        text + "\n",
        " " + text,
    )
    for spelling in respelled:
        if spelling != text:
            _expect_refused("another spelling", lambda s=spelling: ChoiceTable.from_json(s))

    def variant(edit):
        d = json.loads(text)
        edit(d)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    for label, edit in (
        ("generation 0", lambda d: d.update(generation=0)),
        ("generation true", lambda d: d.update(generation=True)),
        ("a shape key missing", lambda d: d["shape"].pop("threads")),
        ("a shape key extra", lambda d: d["shape"].update(seed=1)),
        ("a shape of no threads", lambda d: d["shape"].update(threads=0)),
        ("a negative recompute set", lambda d: d["shape"].update(recompute=-1)),
        ("a contract by number", lambda d: d["shape"].update(contract=16)),
        ("a host value a list", lambda d: d["host"].update(cpus=[4])),
        ("a host value a bool", lambda d: d["host"].update(cpus=True)),
        ("an entry of five fields", lambda d: d["entries"][0].pop()),
        ("an entry of another kind", lambda d: d["entries"][0].__setitem__(0, "attention")),
        ("an alternative its kind lacks", lambda d: d["entries"][0].__setitem__(1, "group")),
        # beside itself, where the one spelling puts it: refused as measured twice, not as respelled
        ("an entry measured twice", lambda d: d["entries"].insert(0, list(d["entries"][0]))),
        ("a low end above the mean", lambda d: d["entries"][0].__setitem__(3, 251)),
        ("a mean above the high end", lambda d: d["entries"][0].__setitem__(2, 261)),
        ("a low end of zero", lambda d: d["entries"][0].__setitem__(3, 0)),
        ("four rounds", lambda d: d["entries"][0].__setitem__(5, 4)),
        ("a float mean", lambda d: d["entries"][0].__setitem__(2, 250.0)),
        ("a key extra", lambda d: d.update(note="x")),
        ("entries an object", lambda d: d.update(entries={})),
    ):
        _expect_refused(label, lambda s=variant(edit): ChoiceTable.from_json(s))
    for label, text_ in (("not JSON", "{"), ("a list", "[]"), ("bytes", text.encode())):
        _expect_refused(label, lambda s=text_: ChoiceTable.from_json(s))
    _expect_refused("an entry twice", lambda: _table([a, b, a]), "measured twice")
    _expect_refused("entries a generator", lambda: ChoiceTable(1, _shape(), {}, (e for e in [a])))
    _expect_refused("an entry a tuple", lambda: ChoiceTable(1, _shape(), {}, (tuple(a.__dict__),)))


def test_choose_takes_only_alternatives_resolved_faster():
    """For each kind the alternative whose whole interval lies below 256 and whose mean is the
    lowest -- the narrower interval on a tie -- at every site of the kind, every other site its
    default; no table is the defaults; a table of another shape, a shape of another decoder and
    anything but a table are refused; and the choice is a program `step_program` states."""
    spec, batch, time_ = nf.PANELS_BENCH
    defaults = default_sites(spec)
    assert choose(None, spec, _shape()) == defaults
    table = _table([
        ChoiceEntry("qkv.vjp", "pair", 240, 230, 250, 16),
        ChoiceEntry("qkv.vjp", "calls", 200, 150, 260, 16),  # lower, but not resolved
        ChoiceEntry("o.vjp", "calls", 250, 240, 260, 16),  # not resolved
        ChoiceEntry("gateup.vjp", "pair", 245, 240, 250, 16),
        ChoiceEntry("gateup.vjp", "calls", 245, 235, 255, 16),  # a tie, the wider interval
        ChoiceEntry("panels", "plain", 280, 270, 290, 16),  # slower
    ])  # fmt: skip
    chosen = choose(table, spec, _shape())
    for site, alternative in chosen.items():
        kind = kind_of(site)
        expected = {"qkv.vjp": "pair", "gateup.vjp": "pair"}.get(kind, defaults[site])
        assert alternative == expected, (site, alternative)
    assert chosen != defaults
    program = step_program(spec, batch, time_, **_BF16, sites=chosen)
    assert program != step_program(spec, batch, time_, **_BF16)
    missing = {k: v for k, v in _shape().items() if k != "steps"}
    for label, args, reason in (
        ("another thread count", (table, spec, _shape(threads=2)), "measured at"),
        ("another contract", (table, spec, _shape(contract="kblock")), "measured at"),
        ("a shape of another decoder", (table, nf.BENCH[0], _shape()), "another decoder"),
        ("a shape missing a field", (table, spec, missing), "shape names"),
        ("a table's JSON", (table.to_json(), spec, _shape()), "ChoiceTable or None"),
    ):
        _expect_refused(label, lambda a=args: choose(*a), reason)


# --- the C ------------------------------------------------------------------------------------


def _corpus_case(contract, n_tokens):
    """The first trace-corpus case of `contract` over `n_tokens` rows, one step of one batch."""
    return next(c for c in nf.trace_corpus()
                if c.contract == contract and c.batch * c.time == n_tokens and c.steps == 1
                and c.accumulation == 1 and c.recompute == 0 and c.workspace != "floor"
                and not c.provider)  # fmt: skip


def _mixes(spec, seed, count):
    rng = random.Random(seed)
    sites = choice_sites(spec)
    return [{site: rng.choice(alternatives) for site, alternatives in sites.items()}
            for _ in range(count)]  # fmt: skip


def test_every_alternative_trains_the_c_steps_bits():
    """On a bf16 case whose shapes take the panels and on a kblock case, every kind switched to
    every alternative, and seeded per-site mixes of them, train the C step's state and events bit
    for bit -- run as the program states them, and run from the pack that states them (planned by
    K_BCIR, hydrated by GEM, lowered by the C to exactly the program's words): every choice is a
    choice of time, never of a value."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return  # no C units (an installed package): a documented skip
        for case in (_corpus_case("bf16", 64), _corpus_case("kblock+exp+reduce", 64)):
            step = nf.trace_case(path, case, trace=False)
            tokens, targets, optimizer = nf._case_inputs(case)
            probe, decisions = nf._case_trainer(path, case)
            probe.close()
            mixes = [_switched(case.spec, k, a) for k in CHOICE_KINDS for a in CHOICE_KINDS[k]]
            mixes += _mixes(case.spec, 1729, 3)
            for sites in mixes:
                program = step_program(case.spec, case.batch, case.time, **decisions,
                                       sites=sites)  # fmt: skip
                words = array("Q", program_words(program))
                tp = training_pack(case.spec, case.batch, case.time, **decisions, sites=sites)
                pack, plan, _, _ = pack_artifacts(tp)
                for source in ("program", "pack"):
                    trainer, _ = nf._case_trainer(path, case)
                    try:
                        run = words if source == "program" else trainer.lower_pack(pack, plan)
                        assert list(run) == list(words), (case.contract, sites, source)
                        events = trainer.run_program(run, tokens, targets, steps=case.steps,
                                                     optimizer=optimizer,
                                                     gradient_accumulation=case.accumulation)  # fmt: skip
                        assert nf.state_digest(trainer) == step.state, (case.contract, sites)
                        assert tuple(events) == tuple(step.events), (case.contract, sites)
                    finally:
                        trainer.close()


def test_every_choice_of_the_corpus_trains_the_c_steps_bits():
    """`train.plan.choose.mismatch`: each state case of the trace corpus, under every kind of
    choice site switched alone to each alternative the C step does not take and three seeded draws
    of an alternative a site, run from its program on the trace build, traces exactly the chosen
    program and trains the C step's state and events bit for bit -- 0 of `choose_run_count()`
    runs; and the runs are the choices: every kind's alternatives change the program on the
    panels' bench under bf16, and most runs differ from the C step's own program."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        traced = nf.trace_library(Path(tmp) / "trace")
        if traced is None:
            return
        bad, runs, notes, other = nf.choose_mismatch(traced)
        assert (bad, runs) == (0, nf.choose_run_count()), notes[:3]
        assert 2 * other > runs, (other, runs)  # the choices ran: not the C step's program again
        corpus = nf.trace_corpus()
        bench = [corpus[i] for i in nf.state_cases(corpus) if corpus[i].spec == nf.PANELS_BENCH[0]
                 and corpus[i].contract == "bf16" and corpus[i].workspace == "panels"]  # fmt: skip
        assert bench, "the panels' bench is a state case"
        bad, runs, _, other = nf.choose_mismatch(traced, bench)
        assert bad == 0 and other == runs == nf.choose_run_count(bench)  # every choice its own


def test_the_job_clock_reads_each_job_once_and_writes_nothing_else():
    """A build with the job clock (`NativeDecoder.build(probe="time")`, a library of its own beside
    the plain one; a probe of another kind refused) records each job's nanoseconds at its first
    record's index and nothing at any other; trains the library's bits;
    writes and counts nothing when no words are lent; counts every job past the words lent and
    writes nothing past them; and a library built without the clock is refused by `timed_run`."""
    if _CC is None:
        return
    case = _corpus_case("v1", 64)
    tokens, targets, optimizer = nf._case_inputs(case)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        timed = nf.time_library(Path(tmp) / "lib")  # one directory: a name each
        plain = nf.checkout_library(Path(tmp) / "lib")
        if timed is None or plain is None:
            return
        assert timed != plain and timed.is_file() and plain.is_file()
        _expect_refused(
            "a probe of no kind", lambda: nat.NativeDecoder.build(Path(tmp), probe="trace")
        )
        step = nf.trace_case(plain, case, trace=False)
        run = dict(optimizer=optimizer, gradient_accumulation=case.accumulation)
        trainer, decisions = nf._case_trainer(timed, case)
        try:
            program = step_program(case.spec, case.batch, case.time, **decisions)
            words = array("Q", program_words(program))
            firsts, at = set(), 0
            for launch in program:
                firsts.add(at)
                at += len(launch.records)
            events, ns = timed_run(trainer, words, tokens, targets, steps=case.steps, **run)
            assert len(ns) == len(words) // RECORD_WORDS == at
            assert all(v == 0 for i, v in enumerate(ns) if i not in firsts), "a word not a job's"
            assert sum(ns) > 0 and tuple(events) == tuple(step.events)
            assert nf.state_digest(trainer) == step.state
        finally:
            trainer.close()
        trainer, _ = nf._case_trainer(timed, case)
        try:  # not begun: nothing written, nothing counted, the same bits
            trainer.run_program(words, tokens, targets, steps=case.steps, **run)
            end = trainer.lib.bcir_decoder_time_probe_end
            end.argtypes, end.restype = [], ct.c_size_t
            assert end() == 0 and nf.state_digest(trainer) == step.state
        finally:
            trainer.close()
        trainer, _ = nf._case_trainer(timed, case)
        try:  # lent fewer words than the program's records: the jobs past them counted
            ordered = sorted(firsts)
            capacity = ordered[len(ordered) // 2]
            lent = (ct.c_uint64 * (capacity + 8))(*([0] * capacity + [77] * 8))
            begin = trainer.lib.bcir_decoder_time_probe_begin
            begin.argtypes, begin.restype = [ct.POINTER(ct.c_uint64), ct.c_size_t], None
            end = trainer.lib.bcir_decoder_time_probe_end
            end.argtypes, end.restype = [], ct.c_size_t
            begin(lent, capacity)
            try:
                trainer.run_program(words, tokens, targets, steps=case.steps, **run)
            finally:
                dropped = end()
            assert dropped == sum(1 for i in ordered if i >= capacity)
            assert list(lent[capacity:]) == [77] * 8 and sum(lent[:capacity]) > 0
        finally:
            trainer.close()
        trainer, _ = nf._case_trainer(plain, case)
        try:
            _expect_refused("a library without the clock",
                            lambda: timed_run(trainer, words, tokens, targets,
                                              steps=case.steps, **run), "no job clock")  # fmt: skip
        finally:
            trainer.close()


def test_a_measurement_is_a_table_at_the_trainers_shape():
    """`measure` on a trainer of a clocked library returns a table at the trainer's shape with one
    entry per measured alternative that states a program of its own, each of the rounds asked and
    an interval in order; its readings, round by round, name every arm in a rotated order, each
    reading a kind's own launches (more than none, less than the whole run); the table
    round-trips and its choice is a program; and a measurement of fewer than five rounds, of no
    kind, of a kind twice, into a list already filled, or on a library without the clock is
    refused."""
    if _CC is None:
        return
    case = _corpus_case("v1", 64)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        timed = nf.time_library(Path(tmp) / "time")
        if timed is None:
            return
        trainer, _ = nf._case_trainer(timed, case)
        try:
            raw = []
            kinds = ("qkv.vjp", "panels", HEAD_SITE)  # no panels under v1: nothing to measure
            table = measure(trainer, rounds=5, kinds=kinds, host={"probe": "test"}, raw=raw)
            assert table.shape == trainer_shape(trainer) and table.host == {"probe": "test"}
            measured = {(e.kind, e.alternative) for e in table.entries}
            # the defaults group the q/k/v VJPs and pair the head's: the rest are measured
            assert measured == {("qkv.vjp", "pair"), ("qkv.vjp", "calls"), (HEAD_SITE, "calls")}
            assert all(e.rounds == 5 and 0 < e.low_q8 <= e.q8 <= e.high_q8 for e in table.entries)
            arms = {"default", "qkv.vjp=pair", "qkv.vjp=calls", "head.vjp=calls"}
            assert len(raw) == 5 and all(set(r["order"]) == arms for r in raw)
            assert len({tuple(r["order"]) for r in raw}) > 1  # rotated
            readings = (arms - {"default"}) | {"default@qkv.vjp", f"default@{HEAD_SITE}"}
            for r in raw:  # a kind's own launches: a part of the run, never none, never all
                assert set(r["total_ns"]) == arms and set(r["site_ns"]) == readings
                for name, ns in r["site_ns"].items():
                    assert 0 < ns < r["total_ns"][name.split("@")[0]], (name, r)
                assert r["site_ns"]["default@qkv.vjp"] != r["site_ns"][f"default@{HEAD_SITE}"]
            assert ChoiceTable.from_json(table.to_json()) == table
            sites = choose(table, case.spec, trainer_shape(trainer))
            step_program(case.spec, case.batch, case.time, **trainer.program_decisions(),
                         sites=sites)  # fmt: skip
            for label, kw, reason in (
                ("four rounds", dict(rounds=4), "a measurement takes at least five rounds"),
                ("no kind", dict(kinds=()), "kinds names kinds"),
                ("a kind twice", dict(kinds=("o.vjp", "o.vjp")), "kinds names kinds"),
                ("a kind of no site", dict(kinds=("attention",)), "kinds names kinds"),
                ("raw already filled", dict(raw=[{}]), "raw is an empty list"),
            ):
                _expect_refused(label, lambda kw=kw: measure(trainer, **kw), reason)
        finally:
            trainer.close()
        plain = nf.checkout_library(Path(tmp) / "plain")
        trainer, _ = nf._case_trainer(plain, case)
        try:
            _expect_refused("a library without the clock",
                            lambda: measure(trainer, rounds=5, kinds=(HEAD_SITE,)),
                            "no job clock")  # fmt: skip
        finally:
            trainer.close()


def test_the_trainer_states_its_own_decisions():
    """`NativeDecoder.program_decisions` reads the decisions a program for the trainer is stated
    under off the trainer itself -- the ones the corpus's witnesses spell out by hand from their
    cases -- and `trainer_shape` the shape a table measured on it names, on cases where every
    decision but the arenas' alignment takes more than one value
    (contracts, recompute sets, the workspace's tiers and the panels' need, a provider's GEMM and
    its SiLU, steps and accumulation, the pool's threads and the instances each job family admits
    -- these last wherever the host's families differ: on a host whose families all admit the same
    instances, an aarch64 host's the portable one alone, they take that one value, which the cases
    spell from each contract's name rather than read off the trainer), and refuses steps or an
    accumulation that is not a positive integer."""
    if _CC is None:
        return
    corpus = nf.trace_corpus()
    cases = [
        _corpus_case("bf16", 64),
        next(
            c for c in corpus if c.contract == "kblock+exp+reduce" and c.recompute and c.steps == 2
        ),
        *(
            c
            for c in corpus
            if c.spec == nf.PANELS_BENCH[0] and c.contract == "bf16" and c.threads == 1
        ),
        *(c for c in corpus if c.provider),
        next(c for c in corpus if c.threads > 1),
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.checkout_library(Path(tmp) / "plain")
        if path is None:
            return
        seen: dict[str, set] = {}
        spelled: set = set()  # the host's instances, spelled from each case's contract
        for i, case in enumerate(cases):
            trainer, decisions = nf._case_trainer(path, case)
            spelled.add(repr(decisions["instances"]))
            try:
                got = trainer.program_decisions(
                    steps=case.steps, gradient_accumulation=case.accumulation
                )
                assert got == decisions, (case, got, decisions)
                spec = case.spec
                assert trainer_shape(trainer, steps=case.steps) == {
                    "vocab": spec.vocab_size, "d_model": spec.d_model, "n_heads": spec.n_heads,
                    "n_layers": spec.n_layers, "d_ff": spec.d_ff, "kv_heads": spec.kv_heads,
                    "batch": case.batch, "time": case.time, "threads": case.threads,
                    "steps": case.steps, "recompute": case.recompute,
                    "workspace_bytes": decisions["workspace_bytes"], "contract": case.contract,
                }  # fmt: skip
                for field, value in got.items():
                    seen.setdefault(field, set()).add(repr(value))
                for bad in (0, -1, 1.0, True) if i == 0 else ():
                    _expect_refused(
                        f"steps {bad!r}", lambda b=bad, t=trainer: t.program_decisions(steps=b)
                    )
                    _expect_refused(
                        f"accumulation {bad!r}",
                        lambda b=bad, t=trainer: t.program_decisions(gradient_accumulation=b),
                    )
            finally:
                trainer.close()
        varied = {field for field, values in seen.items() if len(values) > 1}
        # the instances vary with the contract only where the host's families differ; where they
        # all admit the same instances (an aarch64 host: the portable one alone) the one value is
        # held above to the masks spelled from each contract's name
        fixed = {"alignment"} | ({"instances"} if len(spelled) == 1 else set())
        assert set(seen) - fixed <= varied, sorted(set(seen) - fixed - varied)


assert DecoderSpec  # the spec type the fixtures' shapes are made of
