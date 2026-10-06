"""The before/after audit (`tools/perf/ab_audit.py`) must fail on the changes it exists to catch.

Every slice of the GEM+ completion ladder lands with this audit's table, so the tool is only worth
having if it cannot pass a change that moved something it grades. These tests feed it exactly
those changes: an `exact` row that got one unit worse, a metric that disappeared, an output digest
that moved without a declaration, a regression nobody explained, a row whose output is not even
stable within one tree. Each must fail; the honest version of each must pass. Two tests then run
the real measuring child and the real two-tree parent over a fixture-free row, so the paths a
slice actually takes are exercised end to end -- and an unreachable ref must come back as a
structured UNAVAILABLE report, never a traceback (laws.md L1).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TOOL = os.path.join(_ROOT, "tools", "perf", "ab_audit.py")

sys.path.insert(0, _ROOT)
from tools.perf import ab_audit  # noqa: E402
from tools.perf.ab_audit import (  # noqa: E402
    FAIL,
    PASS,
    UNAVAILABLE,
    grade_audit,
    grade_harness,
    grade_rows,
    merge_rounds,
    verdict_of,
)


def _row(key, kind, measured, improvement, verdict="GAIN"):
    return {
        "key": key,
        "kind": kind,
        "measured": measured,
        "improvement": improvement,
        "verdict": verdict,
    }


def _verdicts(findings):
    return {f.key: f.verdict for f in findings}


def test_an_exact_row_is_graded_with_zero_tolerance():
    """One interpreter, one host: an exact row is bit-identical unless the change moved it, so
    one unit worse is a REGRESSION -- the harness's 2% band is for a baseline frozen elsewhere."""
    before = [
        _row("same", "exact", 10, 0.5),
        _row("better", "exact", 10, 0.5),
        _row("worse", "exact", 1000, 0.5),
    ]
    after = [
        _row("same", "exact", 10, 0.5),
        _row("better", "exact", 9, 0.55),
        _row("worse", "exact", 1001, 0.4995),
    ]
    got = _verdicts(grade_harness(before, after))
    assert got == {"same": "SAME", "better": "GAIN", "worse": "REGRESSION"}, got
    assert verdict_of(grade_harness(before, after), {}, []) == FAIL


def test_a_ratio_row_moves_only_past_its_band():
    before = [
        _row("inside", "ratio", 2.0, 0.90),
        _row("slower", "ratio", 2.0, 0.90),
        _row("faster", "ratio", 2.0, 0.90),
    ]
    after = [
        _row("inside", "ratio", 2.4, 0.88),
        _row("slower", "ratio", 3.0, 0.85),
        _row("faster", "ratio", 1.0, 0.95),
    ]
    got = _verdicts(grade_harness(before, after))
    assert got == {"inside": "NO-CHANGE", "slower": "REGRESSION", "faster": "GAIN"}, got


def test_a_wall_row_is_never_graded():
    """A millisecond is evidence for a human, not a gate: ten times slower is still INDICATIVE."""
    findings = grade_harness([_row("w", "wall", 10.0, 0.9)], [_row("w", "wall", 100.0, 0.1)])
    assert [f.verdict for f in findings] == ["INDICATIVE"] and "slower" in findings[0].note
    assert verdict_of(findings, {}, []) == PASS


def test_a_removed_metric_fails_and_a_new_one_is_reported():
    findings = grade_harness(
        [_row("kept", "exact", 1, 0.0), _row("gone", "exact", 1, 0.0)],
        [_row("kept", "exact", 1, 0.0), _row("fresh", "exact", 0, 1.0)],
    )
    assert _verdicts(findings) == {"kept": "SAME", "gone": "REMOVED", "fresh": "NEW"}
    assert verdict_of(findings, {}, []) == FAIL, "a metric that disappeared is lost coverage"
    assert verdict_of([f for f in findings if f.key != "gone"], {}, []) == PASS


def test_an_output_change_fails_unless_declared():
    before = {
        "correctness_sha256": "aa",
        "samples": [{"name": "c", "result_sha256": "11", "median_ns": 10}],
    }
    after = {
        "correctness_sha256": "bb",
        "samples": [{"name": "c", "result_sha256": "22", "median_ns": 5}],
    }
    findings = grade_audit(before, after)
    assert _verdicts(findings) == {"correctness_sha256": "CHANGED", "c": "CHANGED"}
    assert verdict_of(findings, {}, []) == FAIL
    declared = {
        "correctness_sha256": "the slice changes the plan",
        "c": "the slice changes the plan",
    }
    assert verdict_of(findings, declared, []) == PASS
    same = grade_audit(before, dict(before))
    assert verdict_of(same, {}, []) == PASS and _verdicts(same)["correctness_sha256"] == "SAME"


def test_a_hot_path_row_grades_its_digest_and_calls_exactly():
    b = [{"row": "r", "status": "measured", "times_ns": [10, 12], "calls": 100, "digest": "d1"}]
    worse = [{"row": "r", "status": "measured", "times_ns": [5, 6], "calls": 101, "digest": "d1"}]
    moved = [{"row": "r", "status": "measured", "times_ns": [5, 6], "calls": 90, "digest": "d2"}]
    calls = [f for f in grade_rows(b, worse) if f.kind == "calls"]
    assert [f.verdict for f in calls] == ["REGRESSION"], (
        "one more call is a regression: the count is exact"
    )
    graded = {f.kind: f.verdict for f in grade_rows(b, moved)}
    assert graded == {"digest": "CHANGED", "calls": "GAIN", "time": "INDICATIVE"}, graded
    assert verdict_of(grade_rows(b, moved), {}, []) == FAIL


def test_a_regression_lands_only_with_its_explanation():
    findings = grade_harness([_row("x", "exact", 5, 0.5)], [_row("x", "exact", 6, 0.4)])
    assert verdict_of(findings, {}, []) == FAIL
    assert verdict_of(findings, {"x": "exactness bought with one more expansion"}, []) == PASS
    assert ab_audit.main(["--before", "HEAD", "--explain", "x"]) == UNAVAILABLE, (
        "an explanation needs a reason"
    )


def test_a_nondeterministic_row_is_named_not_averaged():
    rounds = [
        [{"row": "r", "status": "measured", "times_ns": [1], "calls": 10, "digest": "d"}],
        [{"row": "r", "status": "measured", "times_ns": [1], "calls": 11, "digest": "d"}],
    ]
    merged = merge_rounds(rounds)
    assert merged[0]["status"] == "nondeterministic"
    stable = merge_rounds([rounds[0], rounds[0]])
    findings = grade_rows(stable, merged)
    assert [f.verdict for f in findings] == ["CHANGED"] and "nondeterministic" in findings[0].note
    assert verdict_of(findings, {}, []) == FAIL


def test_a_bad_invocation_is_a_verdict_not_a_traceback():
    assert ab_audit.main(["--before", "HEAD", "--rounds", "0"]) == UNAVAILABLE
    assert ab_audit.main(["--before", "HEAD", "--group", "no-such-group"]) == UNAVAILABLE
    assert ab_audit.main(["--before", "HEAD", "--rows", "no-such-row"]) == UNAVAILABLE
    assert ab_audit.main([]) == UNAVAILABLE
    assert ab_audit.main(["--child", "--rows", "selftest"]) == UNAVAILABLE


def test_the_child_measures_a_row_in_the_tree_it_names():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "rows.json")
        empty = os.path.join(tmp, "tree")
        os.mkdir(empty)
        proc = subprocess.run(
            [
                sys.executable,
                _TOOL,
                "--child",
                "--tree",
                _ROOT,
                "--rows",
                "selftest,no-such",
                "--samples",
                "2",
                "--json",
                out,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-800:]
        rows = {r["row"]: r for r in json.load(open(out, encoding="utf-8"))["rows"]}
        got = rows["selftest"]
        assert got["status"] == "measured" and len(got["times_ns"]) == 2 and got["calls"] > 0
        assert len(got["digest"]) == 16 and rows["no-such"]["status"] == "unknown-row"
        # a tree without `bcir` cannot build a row that imports it: unavailable, not a crash
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                _TOOL,
                "--child",
                "--tree",
                empty,
                "--rows",
                "plan@4",
                "--samples",
                "1",
                "--json",
                out,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-800:]
        (row,) = json.load(open(out, encoding="utf-8"))["rows"]
        assert row["status"] == "unavailable" and "ModuleNotFoundError" in row["why"], row


def test_two_trees_give_one_report_on_every_path():
    with tempfile.TemporaryDirectory() as tmp:
        js, md = os.path.join(tmp, "ab.json"), os.path.join(tmp, "ab.md")
        proc = subprocess.run(
            [
                sys.executable,
                _TOOL,
                "--before",
                _ROOT,
                "--after",
                _ROOT,
                "--no-harness",
                "--no-audit",
                "--rows",
                "selftest",
                "--rounds",
                "2",
                "--samples",
                "1",
                "--json",
                js,
                "--markdown",
                md,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == PASS, proc.stdout[-800:] + proc.stderr[-800:]
        report = json.load(open(js, encoding="utf-8"))
        assert report["verdict"] == "PASS" and report["schema"] == "bcir.ab_audit.v1"
        kinds = {f["kind"]: f["verdict"] for f in report["findings"]}
        assert kinds == {"digest": "SAME", "calls": "SAME", "time": "INDICATIVE"}, kinds
        assert "Hot paths (interleaved A/B)" in open(md, encoding="utf-8").read()
        # an unreachable ref is a structured UNAVAILABLE, with the report still written
        os.remove(js)
        proc = subprocess.run(
            [
                sys.executable,
                _TOOL,
                "--before",
                "refs/heads/no-such-branch-for-ab-audit",
                "--no-harness",
                "--no-audit",
                "--rows",
                "selftest",
                "--rounds",
                "1",
                "--samples",
                "1",
                "--json",
                js,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == UNAVAILABLE, proc.stdout[-800:] + proc.stderr[-800:]
        report = json.load(open(js, encoding="utf-8"))
        assert report["verdict"] == "UNAVAILABLE" and "cannot check out" in report["unavailable"][0]


def test_a_nondeterministic_row_names_what_moved():
    """A row unstable between rounds says WHICH exact quantity moved, so the finding can be
    followed up rather than re-measured blind."""
    rounds = [
        [{"row": "r", "status": "measured", "times_ns": [1], "calls": 10, "digest": "d"}],
        [{"row": "r", "status": "measured", "times_ns": [1], "calls": 11, "digest": "d"}],
    ]
    merged = merge_rounds(rounds)
    assert "call count 10 vs 11" in merged[0]["why"], merged[0]
    (finding,) = grade_rows(merge_rounds([rounds[0]]), merged)
    assert "call count 10 vs 11" in finding.note and finding.verdict == "CHANGED"


def test_a_ratio_regression_is_confirmed_before_it_is_graded():
    """One timed ratio per tree can leave its band on a busy host with no change in the code.
    It earns confirming measurements, and is graded on their median: noise becomes NO-CHANGE,
    a real slowdown stays a REGRESSION."""
    import types

    graded = grade_harness([_row("r", "ratio", 1.0, 0.5)], [_row("r", "ratio", 1.5, 0.25)])
    assert [f.verdict for f in graded] == ["REGRESSION"]
    rows = [dict(_row("r", "ratio", 1.0, 0.5), group="g")]
    args = types.SimpleNamespace(ratio_confirm=2, timeout=10)
    real = ab_audit.run_harness
    try:
        for after_values, want in (([1.02, 1.05], "NO-CHANGE"), ([1.6, 1.55], "REGRESSION")):
            calls = []

            def fake(tree, scratch, label, timeout, group="", _after=after_values, _calls=calls):
                _calls.append((tree, group))
                side = 1 if tree == "AFTER" else 0
                value = (
                    _after[min(len([c for c in _calls if c[0] == tree]) - 1, 1)] if side else 1.0
                )
                return [dict(_row("r", "ratio", value, 0.5), group="g")]

            ab_audit.run_harness = fake
            (confirmed,) = ab_audit.confirm_ratios(
                graded, rows, rows, "BEFORE", "AFTER", "/tmp", args
            )
            assert confirmed.verdict == want, (after_values, confirmed)
            assert {g for _t, g in calls} == {"g"} and len(calls) == 4, calls
    finally:
        ab_audit.run_harness = real


def test_measuring_processes_share_one_hash_seed():
    proc = ab_audit._run(
        [sys.executable, "-c", "import os; print(os.environ['PYTHONHASHSEED'])"], _ROOT, 60
    )
    assert proc.stdout.strip() == "0", proc.stdout


def test_a_rows_call_count_counts_two_functions_that_share_a_label():
    """The count is per code object. `pstats` merged two functions with one (file, line, name)
    label -- every dataclass `__init__` -- and kept one count, which one depending on where the
    code objects were allocated: the audit's own run read `sched_eft@4` as 34,368 calls in one
    process and 36,415 in the next."""
    from bcir.tests.call_counts import profiled
    from bcir.tests.test_call_counts import _work

    for shared in (False, True):
        ab_audit.ROWS["twins"] = lambda shared=shared: _work(shared_label=shared)
        try:
            measured = ab_audit.measure_row("twins", 1)["calls"]
        finally:
            del ab_audit.ROWS["twins"]
        assert measured == profiled(_work(shared_label=False))[0], (shared, measured)


def test_the_child_counts_with_this_checkouts_counter_whatever_tree_it_measures():
    """One instrument for both arms: the counter is loaded by path from the tool's checkout, so
    importing it cannot bind `bcir` to this checkout before the measured tree is imported."""
    counter = ab_audit._call_counter()
    assert counter.__file__ == os.path.join(_ROOT, "bcir", "tests", "call_counts.py")
    assert counter.__name__ == "_ab_audit_call_counts"


def test_no_perf_tool_reads_the_merged_pstats_total():
    from bcir.tests.test_call_counts import _reads_pstats

    tools = os.path.join(_ROOT, "tools")
    offenders = []
    for folder, _dirs, files in os.walk(tools):
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as fh:
                    if _reads_pstats(fh.read()):
                        offenders.append(os.path.relpath(path, _ROOT))
    assert offenders == [], offenders
