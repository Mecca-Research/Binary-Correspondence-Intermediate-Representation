"""Every call-count gate counts through one counter, and the counter counts every function.

`pstats.Stats(profile).total_calls` merged two functions that share a (file, line, name) label
-- every dataclass-generated `__init__` -- and kept one of their counts, so the GEM+ call-count
rows lost the constructor calls their emission floors are made of, and a count moved between
processes with nothing changed (`bcir/tests/call_counts.py`). Each test here states a property
the merged total violates.
"""

from __future__ import annotations

import ast
import cProfile
import gc
import pathlib
import pstats

import bcir
from bcir.tests import delta_fixtures, planner_fixtures
from bcir.tests.call_counts import profiled, total_calls

_TWIN_CALLS = 2048


def _work(shared_label: bool):
    """A run calling one function 2048 times and another once. With `shared_label` the two share
    a label, as every dataclass `__init__` does; otherwise they differ only in name."""
    twins = []
    for which in ("a", "b"):
        name = "twin" if shared_label else f"twin_{which}"
        namespace: dict = {}
        exec(compile(f"def {name}():\n    return 1\n", "<twin>", "exec"), namespace)
        twins.append(namespace[name])
    many, once = twins

    def run():
        for _ in range(_TWIN_CALLS):
            many()
        once()

    return run


def test_two_functions_that_share_a_label_are_both_counted():
    apart = profiled(_work(shared_label=False))[0]
    assert profiled(_work(shared_label=True))[0] == apart
    assert apart >= _TWIN_CALLS + 1
    # The instrument this replaces, on the same profile: one of the two entries is gone.
    run = _work(shared_label=True)
    profile = cProfile.Profile()
    profile.enable()
    run()
    profile.disable()
    assert pstats.Stats(profile).total_calls < total_calls(profile) == apart


def test_a_finalizer_the_collector_runs_is_not_a_call_the_code_made():
    """Two runs differing only in the garbage they leave count the same: the collector is paused
    across the counted call, even with a threshold that would collect at every allocation."""

    class Litter:
        def __init__(self):
            self.me = self  # a cycle: only the collector frees it, and runs __del__ when it does

        def __del__(self):
            pass

    class Plain:
        def __init__(self):
            self.me = None

    def make(kind):
        def run():
            for _ in range(100):
                kind()

        return run

    threshold = gc.get_threshold()
    gc.set_threshold(1)
    try:
        littered = profiled(make(Litter))[0]
        plain = profiled(make(Plain))[0]
    finally:
        gc.set_threshold(*threshold)
        gc.collect()
    assert littered == plain, (littered, plain)


def test_the_counter_restores_the_collector_it_found():
    seen = []
    profiled(lambda: seen.append(gc.isenabled()))
    assert seen == [False] and gc.isenabled()
    gc.disable()
    try:
        profiled(lambda: None)
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_a_call_that_raises_still_stops_the_profiler_and_restores_the_collector():
    def boom():
        raise ValueError("boom")

    try:
        profiled(boom)
    except ValueError:
        pass
    else:  # pragma: no cover - the call raises
        raise AssertionError("the call's exception was swallowed")
    assert gc.isenabled()
    assert profiled(lambda: None)[0] >= 1, "a profiler left running would count this frame twice"


def test_every_call_count_fixture_counts_through_the_one_counter():
    run = _work(shared_label=True)
    exact = profiled(run)[0]
    assert delta_fixtures._profiled(run)[0] == exact
    assert delta_fixtures._profiled_calls(run) == exact

    def planner(_module, _h, _theta):
        run()

    assert planner_fixtures.call_count(planner, scale=1) == profiled(planner, None, None, None)[0]


def _reads_pstats(source: str) -> bool:
    """Whether code (not prose: a docstring may name the defect) imports `pstats` or reads a
    `total_calls`."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(
            a.name.split(".")[0] == "pstats" for a in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "pstats":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "total_calls":
            return True
    return False


def test_no_call_count_in_the_package_reads_the_merged_pstats_total():
    """The repeated defect has one predicate (laws.md L14): a count read off `pstats` anywhere
    else in the package is the defect returning. Only this module may use it, to prove it."""
    assert _reads_pstats("import pstats\n") and _reads_pstats("n = stats.total_calls\n")
    assert not _reads_pstats('"""pstats.Stats(profile).total_calls merged them."""\n')
    root = pathlib.Path(bcir.__file__).resolve().parent
    me = pathlib.Path(__file__).resolve()
    offenders = [
        str(path.relative_to(root))
        for path in sorted(root.rglob("*.py"))
        if path.resolve() != me and _reads_pstats(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], offenders
