#!/usr/bin/env python3
"""Self-test for analyze-benchmark-samples.py.

Two layers:

  1. Statistical kernels checked against values that can be worked out by hand
     or looked up -- ranks with ties, Mann-Whitney U, Cliff's delta, the
     Hodges-Lehmann shift, percentiles, drift correlation.
  2. Every checked-in fixture graded, with its expected verdict pinned.

The fixtures are the part that matters. Each one is a *shape of wrong answer*
this analysis exists to refuse: an effect below the rig's resolution, a
significant-but-trivial difference, a drifting baseline, a wall-clock row used
as a gate. A tool that only passes its clean-improvement case would keep
passing after every refusal was removed from it.

Deterministic: the bootstrap seed is fixed, so this is a regression test on the
intervals as well as on the verdicts.

    python3 training/llvm/tools/verify-benchmark-analysis.py
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = TRAINING_ROOT.parent.parent
TOOL = TRAINING_ROOT / "tools" / "analyze-benchmark-samples.py"
FIXTURES = TRAINING_ROOT / "21-performance-methodology" / "examples"

# fixture -> (expected verdict, must-appear substring of the reason)
EXPECTED = {
    "clean-improvement.json": ("improvement", "faster"),
    "noise-only.json": ("no-detectable-difference", "resolution"),
    "below-resolution.json": ("no-detectable-difference", "resolution"),
    "thermal-drift.json": ("inconclusive", "drifts monotonically"),
    "significant-but-trivial.json": ("detectable-but-immaterial", "threshold"),
    "wall-clock-indicative.json": ("improvement", "faster"),
}


def load_tool():
    spec = importlib.util.spec_from_file_location("analyze_benchmark_samples", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before executing: the tool uses `from __future__ import
    # annotations`, so @dataclass resolves its field types through
    # sys.modules[cls.__module__], which does not exist yet otherwise.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, description: str, actual, expected) -> None:
        self.checks += 1
        ok = (
            math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
            if isinstance(actual, float) and isinstance(expected, float)
            else actual == expected
        )
        if not ok:
            self.failures.append(f"{description}: expected {expected!r}, got {actual!r}")

    def check_true(self, description: str, condition: bool) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(description)


def test_kernels(m, c: Checker) -> None:
    # Ranks with a tie: [10, 20, 20, 30] -> 1, 2.5, 2.5, 4
    c.check("average_ranks with ties", m.average_ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    # No ties -> correction 0; three tied values -> 3^3 - 3 = 24
    c.check("tie_correction, no ties", m.tie_correction([1, 2, 3]), 0.0)
    c.check("tie_correction, one triple", m.tie_correction([1, 1, 1, 2]), 24.0)

    # Complete separation: every a exceeds every b, so U_a = n1*n2 and
    # Cliff's delta = +1.
    a = [10.0, 11.0, 12.0]
    b = [1.0, 2.0, 3.0]
    u_a, p = m.mann_whitney(a, b)
    c.check("U with complete separation", u_a, 9.0)
    c.check("p withheld below the small-sample floor", p, None)
    c.check("Cliff's delta with complete separation", m.cliffs_delta(a, b, u_a), 1.0)
    c.check("Cliff's delta is antisymmetric", m.cliffs_delta(b, a, m.mann_whitney(b, a)[0]), -1.0)

    # Identical samples: U sits exactly on its mean, delta is 0.
    same = [5.0] * 10
    u_same, p_same = m.mann_whitney(same, same)
    c.check("U for identical groups", u_same, 50.0)
    c.check("p for identical groups", p_same, 1.0)
    c.check("delta for identical groups", m.cliffs_delta(same, same, u_same), 0.0)

    # Hodges-Lehmann shift: b is a exactly +5 everywhere.
    c.check(
        "HL shift of a pure translation",
        m.hodges_lehmann_shift([1.0, 2.0, 3.0], [6.0, 7.0, 8.0]),
        5.0,
    )

    # Percentiles on a known ladder.
    ladder = [1.0, 2.0, 3.0, 4.0, 5.0]
    c.check("median percentile", m.percentile(ladder, 0.5), 3.0)
    c.check("min percentile", m.percentile(ladder, 0.0), 1.0)
    c.check("max percentile", m.percentile(ladder, 1.0), 5.0)
    c.check("interpolated percentile", m.percentile(ladder, 0.25), 2.0)

    # Drift: a monotonically increasing series is rho = +1, decreasing is -1,
    # and a series with no order effect is near 0.
    c.check("drift of a rising series", m.spearman_rho([1.0, 2.0, 3.0, 4.0]), 1.0)
    c.check("drift of a falling series", m.spearman_rho([4.0, 3.0, 2.0, 1.0]), -1.0)
    c.check_true(
        "drift of a flat series is zero", abs(m.spearman_rho([2.0, 2.0, 2.0, 2.0])) < 1e-12
    )

    # The bootstrap is seeded: same inputs, same interval, every time.
    first = m.bootstrap_ratio_ci(ladder, ladder, iterations=200, alpha=0.05, seed=7)
    second = m.bootstrap_ratio_ci(ladder, ladder, iterations=200, alpha=0.05, seed=7)
    c.check("bootstrap is deterministic", first, second)

    # A series compared against itself has a resolution, and it is finite.
    resolution = m.noise_floor(
        [100.0 + (i % 5) for i in range(40)], iterations=200, alpha=0.05, seed=3
    )
    c.check_true(
        "noise floor is finite and non-negative", not math.isnan(resolution) and resolution >= 0.0
    )


def test_fixtures(m, c: Checker) -> None:
    seen = set()
    for name, (verdict, needle) in EXPECTED.items():
        path = FIXTURES / name
        c.check_true(f"fixture exists: {name}", path.is_file())
        if not path.is_file():
            continue
        seen.add(name)
        document = m.load(path)
        result = m.analyze(
            document,
            alpha=0.05,
            iterations=2000,
            seed=20260101,
            min_samples=m.DEFAULT_MIN_SAMPLES,
            min_effect=m.DEFAULT_MIN_EFFECT,
        )
        c.check(f"{name} verdict", result.verdict, verdict)
        c.check_true(
            f"{name} reason mentions {needle!r} (got: {result.reason[:70]}...)",
            needle in result.reason,
        )

    # Any fixture nobody grades is a fixture nobody maintains.
    on_disk = {p.name for p in FIXTURES.glob("*.json")}
    unexpected = on_disk - seen
    c.check_true(
        f"every fixture is graded (ungraded: {sorted(unexpected)})",
        not unexpected,
    )


def test_gate_behaviour(c: Checker) -> None:
    """--gate must refuse a wall-clock row rather than blessing it."""
    wall = FIXTURES / "wall-clock-indicative.json"
    completed = subprocess.run(
        [sys.executable, str(TOOL), str(wall), "--gate"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    c.check("--gate refuses a wall-class row", completed.returncode, 3)
    c.check_true(
        "--gate says why it refused",
        "REFUSED to gate" in completed.stderr,
    )

    ratio = FIXTURES / "clean-improvement.json"
    completed = subprocess.run(
        [sys.executable, str(TOOL), str(ratio), "--gate"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    c.check("--gate passes a clean improvement on a ratio row", completed.returncode, 0)

    # A malformed document is an error, never a silent pass.
    completed = subprocess.run(
        [sys.executable, str(TOOL), str(FIXTURES / "does-not-exist.json")],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    c.check("a missing input file is an error", completed.returncode, 2)


def main() -> int:
    if not TOOL.is_file():
        print(f"benchmark analysis self-test: FAILED (missing {TOOL})", file=sys.stderr)
        return 1

    module = load_tool()
    checker = Checker()
    test_kernels(module, checker)
    test_fixtures(module, checker)
    test_gate_behaviour(checker)

    if checker.failures:
        print("benchmark analysis self-test: FAILED", file=sys.stderr)
        for failure in checker.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"benchmark analysis self-test: PASSED ({checker.checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
