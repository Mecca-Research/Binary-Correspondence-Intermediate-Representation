"""The GEM+ baseline harness must reproduce the report it claims to measure against.

`tools/perf/gemplus_baseline.py` freezes the numbers from the 2026-08-12 architecture and
performance audit so that every GEM+ slice can be graded rather than asserted. A harness like
that is only worth having if it is itself checked, and the two things worth checking are the
two ways it can lie:

  * **it can manufacture a verdict.** The first run of the divergence row used a fixture that
    did not match the report — four claims sharing one resource on a one-domain target — and
    reported a 1.99 → 1.11 GAIN against a slice that had not been written. The fixture now
    reproduces the report's own ratio to every digit the report prints, and this test pins
    that, because a metric that does not measure what it names is worse than no metric.

  * **it can grade what it cannot compare.** An absolute millisecond measured on a different
    machine than the report's is not evidence in either direction. The same first run called
    `optimize_scheduled.512` a 15.9% REGRESSION while the *ratio* row from the identical run
    landed within 0.4% of the baseline. Wall rows off the baseline host are therefore
    reported and never graded, and this test pins that too.

The bounds matter as much as the values: `headroom` is what turns a measurement into a TMSAO
statement, so a row claiming a bound it cannot justify is a row that will eventually be used
to claim an optimality it has not earned.
"""

from __future__ import annotations

import os
import subprocess
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TOOL = os.path.join("tools", "perf", "gemplus_baseline.py")

sys.path.insert(0, _ROOT)
from tools.perf.gemplus_baseline import (  # noqa: E402
    METRICS, SOURCE, Metric, compare, measure_exact,
)


def test_the_report_the_baseline_quotes_is_in_the_tree() -> None:
    """A frozen number whose source has moved is a number nobody can check."""
    assert os.path.exists(os.path.join(_ROOT, SOURCE)), SOURCE


def test_every_metric_is_well_formed_and_points_the_right_way() -> None:
    """The invariants a wrong row would violate silently.

    `lower_is_better` is the one that would do real damage: it decides the SIGN of every
    verdict and every headroom, so a throughput row marked the wrong way would report a
    regression as a gain forever.
    """
    seen = set()
    for metric in METRICS:
        assert metric.key not in seen, f"duplicate metric key {metric.key}"
        seen.add(metric.key)
        assert metric.kind in ("wall", "ratio", "exact"), (metric.key, metric.kind)
        assert metric.group and metric.what and metric.unit, metric.key
        assert metric.baseline > 0, metric.key
        assert metric.slice_owner, f"{metric.key} names no owning slice"
        if metric.bound is not None:
            assert metric.bound_source, f"{metric.key} has a bound with no justification"
            # A bound must actually be a bound: reachable, and on the improving side.
            if metric.lower_is_better:
                assert metric.bound <= metric.baseline, (
                    f"{metric.key}: bound {metric.bound} is worse than the baseline")
            else:
                assert metric.bound >= metric.baseline, (
                    f"{metric.key}: bound {metric.bound} is worse than the baseline")


def test_headroom_is_zero_at_the_bound_and_positive_above_it() -> None:
    """The column that turns a measurement into an optimality statement."""
    lower = Metric("t", "g", "w", 100.0, "ms", "wall", bound=25.0, slice_owner="G0")
    assert lower.headroom(100.0) == 0.75
    assert lower.headroom(50.0) == 0.5
    assert lower.headroom(25.0) == 0.0
    assert lower.headroom(10.0) == 0.0, "past the bound is finished, never negative"

    higher = Metric("t", "g", "w", 5.0, "x", "ratio", bound=10.0,
                    lower_is_better=False, slice_owner="G0")
    assert higher.headroom(5.0) == 0.5
    assert higher.headroom(10.0) == 0.0
    assert higher.headroom(12.0) == 0.0

    # No bound means no optimality claim is available, which must read as unknown rather
    # than as zero -- "finished" and "never measured" are opposite statements.
    assert Metric("t", "g", "w", 1.0, "x", "ratio", slice_owner="G0").headroom(1.0) is None


def test_a_wall_row_is_never_graded_off_the_baseline_host() -> None:
    """The manufactured-verdict failure, in the direction that blocks good work."""
    wall = Metric("w", "g", "w", 100.0, "ms", "wall", slice_owner="G0")
    assert wall.verdict(140.0, same_host=False) == "INDICATIVE"
    assert wall.verdict(40.0, same_host=False) == "INDICATIVE"
    assert wall.verdict(140.0, same_host=True) == "REGRESSION"
    assert wall.verdict(40.0, same_host=True) == "GAIN"
    assert wall.verdict(None) == "NOT-MEASURED"

    # Exact rows are deterministic, so they grade on any host -- that is the whole reason
    # the roadmap gates on them.
    exact = Metric("e", "g", "w", 2.0, "x", "exact", slice_owner="G0")
    assert exact.verdict(1.0, same_host=False) == "GAIN"
    assert exact.verdict(2.0, same_host=False) == "NO-CHANGE"
    assert exact.verdict(3.0, same_host=False) == "REGRESSION"


def test_the_noise_band_widens_with_how_noisy_the_kind_is() -> None:
    """A timing-derived ratio inherits both timings' variance; a solved one has none.

    Measured 68.95 and then 76.29 minutes apart on one host, which is where the `ratio`
    band comes from. Pinned so nobody tightens it back to the `exact` band and starts
    grading noise.
    """
    wall = Metric("a", "g", "w", 1.0, "ms", "wall", slice_owner="G0")
    ratio = Metric("b", "g", "w", 1.0, "x", "ratio", slice_owner="G0")
    exact = Metric("c", "g", "w", 1.0, "x", "exact", slice_owner="G0")
    assert exact.noise < wall.noise < ratio.noise


def test_the_divergence_row_reproduces_the_reports_own_number() -> None:
    """§6.2's fixture, and the reason this test is the important one in the file.

    The report measured `price_scheduled` at 51,200 against `schedule_eft` at 25,700 on the
    same plan. This fixture's absolute costs are 4x the report's, and the RATIO matches to
    every digit the report prints. If it ever stops matching, either the fixture drifted or
    one of the two schedulers changed — and both of those need a human before any slice
    claims credit for the difference.
    """
    measured = measure_exact()
    value = measured.get("pricing.eft.divergence")
    assert value is not None, "the §6.2 fixture no longer builds; the row measures nothing"
    assert abs(value - 51200 / 25700) < 1e-6, (
        f"the divergence fixture measures {value}, the report measures "
        f"{51200 / 25700}; the fixture and the report have come apart")

    # And the meaning: any value but 1.0 is two prices for one plan.
    assert value > 1.5, "the fixture must still EXHIBIT the divergence it exists to track"


def test_compare_reports_every_metric_exactly_once() -> None:
    rows = compare({}, same_host=False)
    assert len(rows) == len(METRICS)
    assert {r["key"] for r in rows} == {m.key for m in METRICS}
    assert all(r["verdict"] == "NOT-MEASURED" for r in rows)
    assert all(r["slice"] for r in rows), "every row names the slice that must move it"


def test_the_tool_runs_and_lists_its_baseline() -> None:
    """`--list` is how a reader finds the target before writing a slice."""
    done = subprocess.run([sys.executable, _TOOL, "--list"], cwd=_ROOT,
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    assert "frozen metrics" in done.stdout
    for key in ("optimize_scheduled.512", "pricing.eft.divergence", "memory.worst.ratio"):
        assert key in done.stdout, key
