"""Every GEM+ baseline row carries a lower bound -- the thirteen that had none measure theirs.

`tools/perf/gemplus_baseline.py` grades each row against the frozen baseline and states its
`headroom`: how far the incumbent sits above a floor no implementation can go below. Thirteen
rows had no floor, so no optimality statement was available for them (TMSAO-4). Each now carries
one measured in the SAME run as its value (`Metric.floor_key`) -- the same process, host, fixture
and clock, which is the condition under which a trivial solution is a floor at all (laws.md L23):

  * the digest row: the FNV-1a chain alone over the pre-rendered canonical stream (it returns the
    same digest, so it is the digest's own arithmetic with the walk and rendering removed);
  * the call-count rows: the emission floor -- the call plus one constructor call per record the
    output must carry fresh, a distinct value once, a value the previous link holds not at all
    (an update costs at least its change);
  * the byte-producing ratios and the native planner: writing the output's bytes once;
  * the audit rows: the fixture each case builds inside its own timed interval (plus, for the
    static-lifetime planner, the digest its plan carries);
  * the verifier ratio: re-deriving each chosen step's cost once through R9's own predicate.

A floor is work every implementation does, so the incumbent can meet it and never beat it: a
value past its floor means the floor measured something else, and the harness refuses the table
(`floor_violated`). These tests hold each floor to the work it stands for (the digest floor IS the
digest, the verifier floor re-derives the score, the emission floor counts exactly the fresh
values) and to its value on small fixtures, where the rows' own scale is too slow for a unit test.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, make_dataclass

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from tools.perf import gemplus_baseline as gb  # noqa: E402
from tools.perf.gemplus_baseline import (  # noqa: E402
    METRICS,
    Metric,
    _value_key,
    audit_floor,
    chain_floor,
    compare,
    delta_floor,
    emission_floor,
    encode_floor,
    plan_floor,
    render,
    render_list,
)

#: The rows that had no lower bound before their floors were measured (the 2026-09-30 table).
MEASURED_FLOORS = (
    "static_memory.digest.2048",
    "planner.calls",
    "planner.native.scale4",
    "kbcir-streampack.delta",
    "kbcir-streampack.delta.calls",
    "streampack.encode.calls",
    "streampack.encode",
    "kbcir-streampack.full.calls",
    "audit.kbcir-streampack.scale4",
    "audit.static-lifetime-planner.scale4",
    "audit.mixed-wave-token-eft.scale4",
    "audit.iterative-phase-dag.scale4",
    "verify.plan.scope.overhead",
)


def test_every_row_has_a_frozen_bound_or_a_floor_measured_beside_it() -> None:
    """No row is left without a floor, and the ones measured in-run are exactly these thirteen:
    `--list` has no "no lower bound yet" section to print."""
    keys = {m.key for m in METRICS}
    measured = {m.key for m in METRICS if m.floor_key}
    assert measured == set(MEASURED_FLOORS)
    for metric in METRICS:
        assert metric.bound is not None or metric.floor_key, metric.key
        if metric.floor_key:
            assert metric.bound is None, f"{metric.key}: a measured floor and a frozen one"
            assert metric.floor_key == metric.key + ".floor", metric.key
            assert metric.floor_key not in keys, metric.key
            assert metric.bound_source, f"{metric.key} has a floor with no justification"
    listing = render_list()
    assert "no lower bound yet" not in listing and "no bound computed yet" not in listing
    assert listing.count("bound measured in the same run") == len(MEASURED_FLOORS)


def _row(rows, key):
    (row,) = [r for r in rows if r["key"] == key]
    return row


def test_compare_grades_headroom_against_the_floor_of_the_same_run() -> None:
    """The measured floor is the row's bound; without it (or without the value) there is no
    headroom to state -- least of all the baseline's, which was measured on another host."""
    metric = next(m for m in METRICS if m.key == "planner.calls")
    rows = compare({metric.key: 400.0, metric.floor_key: 100.0}, same_host=False)
    row = _row(rows, metric.key)
    assert row["bound"] == 100.0 and row["bound_measured"] is True
    assert row["headroom"] == 0.75 and row["floor_violated"] is False
    row = _row(compare({metric.key: 400.0}, same_host=False), metric.key)
    assert row["bound"] is None and row["headroom"] is None
    row = _row(compare({metric.floor_key: 100.0}, same_host=False), metric.key)
    assert row["headroom"] is None and row["floor_violated"] is False
    # A frozen-bound row still reads its frozen bound, measured or not.
    frozen = next(m for m in METRICS if m.bound is not None and not m.floor_key)
    row = _row(compare({}, same_host=False), frozen.key)
    assert row["bound"] == frozen.bound and row["bound_measured"] is False


def test_a_floor_past_its_value_blocks_the_table() -> None:
    """An exact floor is held exactly; a timed one past the kind's noise band. Either way the
    render names it and the tool exits nonzero -- a false floor makes every headroom false."""
    exact = Metric("e", "g", "w", 10.0, "calls", "exact", floor_key="e.floor", slice_owner="G0")
    assert not exact.floor_violated(10.0, 10.0)
    assert exact.floor_violated(9.9, 10.0)  # an exact floor has no noise band to hide in
    wall = Metric("w", "g", "w", 10.0, "ms", "wall", floor_key="w.floor", slice_owner="G0")
    assert not wall.floor_violated(9.0, 10.0)  # inside the 15% band: both sides carry jitter
    assert wall.floor_violated(8.0, 10.0)
    assert not wall.floor_violated(None, 10.0) and not wall.floor_violated(5.0, None)
    higher = Metric(
        "h", "g", "w", 1.0, "x", "ratio", lower_is_better=False, floor_key="h.f", slice_owner="G0"
    )
    assert higher.floor_violated(2.0, 1.0) and not higher.floor_violated(1.1, 1.0)

    rows = compare({"planner.calls": 10.0, "planner.calls.floor": 11.0}, same_host=False)
    assert _row(rows, "planner.calls")["floor_violated"] is True
    text = render(rows)
    assert "FLOORS PAST THEIR VALUE" in text and "planner.calls" in text.split("FLOORS")[1]

    saved = gb.measure
    gb.measure = lambda *_: {"planner.calls": 10.0, "planner.calls.floor": 11.0}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assert gb.main(["--compare", "--group", "kplan"]) == 1
            gb.measure = lambda *_: {"planner.calls": 12.0, "planner.calls.floor": 11.0}
            assert gb.main(["--compare", "--group", "kplan"]) == 0
    finally:
        gb.measure = saved


@dataclass(frozen=True)
class _Rec:
    a: int
    b: tuple = ()


@dataclass
class _Holder:
    items: list
    cache: object = field(default=None, compare=False)


def test_the_emission_floor_counts_each_fresh_record_value_once() -> None:
    """The call, plus one constructor per distinct record value; a derived cache is not value,
    a value `prior` holds can be shared, bytes are no record, a foreign dataclass is not ours."""
    out = _Holder([_Rec(1), _Rec(1), _Rec(2, (3,))], cache=_Rec(9))
    assert emission_floor(out) == 1 + 1 + 2  # the call, the holder, two distinct values
    assert emission_floor(out, prior=[_Rec(1)]) == 1 + 1 + 1
    assert emission_floor(out, prior=_Holder([_Rec(1), _Rec(1), _Rec(2, (3,))])) == 1
    assert emission_floor(b"\x00" * 64) == 1
    assert emission_floor({"k": [_Rec(4)], "j": (_Rec(4),)}) == 1 + 1
    foreign = make_dataclass("Foreign", [("x", int)])
    foreign.__module__ = "elsewhere"  # a dataclass, but not a BCIR record
    assert emission_floor([foreign(1), foreign(2)]) == 1
    records: list = []
    _value_key([_Rec(5), _Rec(5)], {}, records)
    assert len(records) == 2 and len(set(records)) == 1  # two objects, one value


def test_the_plan_floor_is_a_step_per_claim_and_its_distinct_values() -> None:
    """The planner's floor, recounted by hand on the audit fixture at scale 1: a ChosenStep per
    claim, the result, and each distinct Candidate and CostVector -- below the counted calls."""
    from bcir.kbcir import realize
    from bcir.tests.planner_fixtures import call_count

    counted: list = []
    calls = call_count(realize.optimize, 1, out=counted)
    result, module = counted
    claims = sum(len(ph.claims) for ph in module.phases)
    assert len(result.steps) == claims > 0
    candidates = {s.candidate for s in result.steps}
    vectors = {s.candidate.base for s in result.steps}
    floor = plan_floor(result, module)
    assert floor == 1 + claims + 1 + len(candidates) + len(vectors)
    assert 0 < floor < calls


def test_the_delta_floor_is_its_change_and_the_previous_link_survives() -> None:
    """A one-claim delta's floor is its change, at every scale: the link, the plan's result and the
    changed step (its candidate and cost vector), the pack and the changed block, the declared
    module and its changed phase, and the three verdicts the edit adds (its count overruns three
    resources, R7). The previous link survives the apply, so sharing its records is sound."""
    from bcir.gem.delta_chain import DeltaChain
    from bcir.kbcir.delta import Delta
    from bcir.kbcir.weights import PERF
    from bcir.tests import delta_fixtures as df
    from bcir.tests.planner_fixtures import audit_fixture

    expected = Counter(
        Link=1,
        RealizationResult=1,
        ChosenStep=1,
        Candidate=1,
        CostVector=1,
        StreamPack=1,
        Block=1,
        Module=1,
        Phase=1,
        Diagnostic=3,
    )
    for scale in (1, 2):
        links: list = []
        step, full = df.delta_calls(scale, links=links)
        before, after, delta = links
        fresh: list = []
        held: list = []
        _value_key(after, {}, fresh)
        _value_key((before, delta), {}, held)
        assert Counter(key[2] for key in set(fresh) - set(held)) == expected, scale
        floor = delta_floor(before, after, delta)
        assert floor == 1 + sum(expected.values()) < step < full
        assert emission_floor(after, prior=before) == floor + 1  # the delta's own claim is input

    module, h, theta = audit_fixture(1)
    chain = DeltaChain.build(module, h, theta, PERF, df.PLAN, 2)
    old = chain.link

    def snapshot(link):
        steps = [(s.claim_id, s.phase_id, s.candidate, s.cost) for s in link.result.steps]
        return steps, list(link.pack.segments), list(link.pack.blocks), link.data

    kept = snapshot(old)
    chain.apply(Delta(*df.audit_delta(chain.module, 0)))
    assert snapshot(old) == kept and chain.link is not old


def test_the_chain_floor_counts_the_plan_and_the_pack() -> None:
    """From scratch nothing can be shared: the floor is every distinct record of the plan and the
    pack plus the link, and the chain's calls sit far above it."""
    from bcir.tests import delta_fixtures as df

    chained: list = []
    calls = df.full_calls(1, links=chained)
    reference, module = chained
    pack = reference.pack
    records = (
        len(pack.segments)
        + len(pack.prefetches)
        + len(pack.blocks)
        + len(pack.trace_notes)
        + len(pack.generations)
    )
    floor = chain_floor(reference, module)
    assert floor >= 1 + 1 + len(reference.result.steps) + 1 + records
    assert floor < calls


def test_the_encode_floor_is_its_call() -> None:
    from bcir.tests.encode_fixtures import encode_calls

    counted: list = []
    calls, reference = encode_calls(1, out=counted)
    data, pack = counted
    assert isinstance(data, bytes) and data
    assert encode_floor(data, pack) == 1 < calls < reference


def test_the_digest_floor_is_the_digest() -> None:
    """`fnv_chain` over `rendered_stream` returns the module digest on every fixture, and a
    one-field edit moves both: the floor chains the content, not a stand-in for it."""
    from dataclasses import replace

    from bcir.examples import PROGRAMS
    from bcir.kbcir.provenance import fnv_chain, hash_module, rendered_stream
    from bcir.performance_audit import static_memory_module
    from bcir.tests.sweep_fixtures import random_module

    modules = [build() for _, build in sorted(PROGRAMS.items())]
    modules += [random_module(seed) for seed in range(12)] + [static_memory_module(1)]
    for module in modules:
        rendered = rendered_stream(module)
        assert rendered and all(chunk.endswith(b"\xff") for chunk in rendered)
        assert fnv_chain(rendered) == hash_module(module), module.name
    module = static_memory_module(1)
    before = fnv_chain(rendered_stream(module))
    phase = next(ph for ph in module.phases if ph.claims)
    phase.claims[0] = replace(phase.claims[0], count=phase.claims[0].count + 1)
    module.touch()
    assert fnv_chain(rendered_stream(module)) == hash_module(module) != before


def test_the_ratio_floors_are_timed_beside_their_values() -> None:
    """The delta and encode ratios on small fixtures: the floor writes the output's own bytes once
    -- the new link's whole pack, the whole wire image -- in the same rounds, below the ratio."""
    from bcir.abi.streampack_abi import encode
    from bcir.gem.delta_chain import DeltaChain
    from bcir.kbcir.delta import Delta
    from bcir.kbcir.weights import PERF
    from bcir.tests import delta_fixtures as df
    from bcir.tests.encode_fixtures import audit_pack, encode_ratio
    from bcir.tests.planner_fixtures import audit_fixture

    floor: dict = {}
    value = df.delta_ratio(1, 3, floor=floor)
    assert 0 < floor["ratio"] < value
    module, h, theta = audit_fixture(1)
    chain = DeltaChain.build(module, h, theta, PERF, df.PLAN, 2)
    for k in range(3):
        link = chain.apply(Delta(*df.audit_delta(chain.module, k)))
    assert floor["bytes"] == len(link.data) > 0

    floor = {}
    value = encode_ratio(1, 3, floor=floor)
    assert 0 < floor["ratio"] < value
    assert floor["bytes"] == len(encode(audit_pack(1))) > 0


def test_the_verifier_floor_re_derives_every_step() -> None:
    from tools.perf.gemplus_baseline import measure_verifier

    out = measure_verifier()  # raises if the re-derived costs do not sum to the plan's score
    value, floor = out["verify.plan.scope.overhead"], out["verify.plan.scope.overhead.floor"]
    assert 0 < floor < value


def test_the_audit_floors_are_the_fixtures_the_cases_build() -> None:
    """Each floor fixture is the one its case builds -- the case calls it, once, inside its timed
    interval -- a row per case; the static-lifetime floor adds the chain of the digest its plan
    carries; and the floor sits below the case that builds it and then does the GEM work."""
    import statistics
    import time

    from bcir import performance_audit as pa

    cases = {name: fn for _, name, _, fn in pa._case_definitions(1)}
    assert set(pa.AUDIT_FIXTURES) <= set(cases)
    rows = {m.key for m in METRICS if m.floor_key and m.group == "audit"}
    assert rows == {f"audit.{name}.scale4" for name in pa.AUDIT_FIXTURES}

    for name, build in pa.AUDIT_FIXTURES.items():
        calls = []

        def counting(scale, build=build, calls=calls):
            calls.append(scale)
            return build(scale)

        setattr(pa, build.__name__, counting)
        try:
            cases[name](1)
        finally:
            setattr(pa, build.__name__, build)
        assert calls == [1], (name, build.__name__, calls)

    def median_ms(fn, repeats=5):
        fn()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1e3)
        return statistics.median(samples)

    for name in pa.AUDIT_FIXTURES:
        parts = audit_floor(name, 1, 5)
        expected = {"fixture", "digest"} if name == "static-lifetime-planner" else {"fixture"}
        assert set(parts) == expected and all(v > 0 for v in parts.values()), (name, parts)
        metric = next(m for m in METRICS if m.key == f"audit.{name}.scale4")
        value = median_ms(lambda name=name: cases[name](1))
        floor = sum(parts.values())
        assert not metric.floor_violated(value, floor), (name, value, floor)


def test_the_native_floor_writes_the_whole_realization() -> None:
    """`test_kplan --bench-floor` stores the fixed-size BKPR once (32 + 120 x claims + 4 bytes) and
    is timed beside `--bench` on the same record. Needs a C compiler: NOT-MEASURED without one."""
    import shutil
    import tempfile

    from bcir.tests.planner_fixtures import audit_fixture, build_harness, native_floor, native_ms

    tmp = tempfile.mkdtemp(prefix="bcir-kplan-floor-")
    try:
        exe = build_harness(tmp)
        if exe is None:
            return
        module, _, _ = audit_fixture(1)
        claims = sum(len(ph.claims) for ph in module.phases)
        floor_ms, size = native_floor(exe, tmp, scale=1)
        assert size == 32 + 120 * claims + 4
        assert 0 < floor_ms < native_ms(exe, tmp, scale=1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_roadmap_lists_the_harness_floors() -> None:
    """The GEM+ roadmap's §0.3 names the rows whose floor is measured in-run, and the harness
    measures exactly those; no row is left unbounded. §0.3 once listed five unbounded rows while
    the harness had thirteen -- a mirror list drifts (laws.md L15), so this reads both."""
    import re

    path = os.path.join(_ROOT, "docs", "research", "BCIR_GEMPLUS_ROADMAP.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    section = text.split("### 0.3 ", 1)[1].split("\n## ", 1)[0]
    block = re.search(r"```\n(.*?)```", section, re.DOTALL)
    assert block, "§0.3 lost its fenced list of measured-floor rows"
    listed = block.group(1).split()
    assert len(listed) == len(set(listed)), "a row is listed twice"
    assert set(listed) == {m.key for m in METRICS if m.floor_key}
    assert not [m.key for m in METRICS if m.bound is None and not m.floor_key]
