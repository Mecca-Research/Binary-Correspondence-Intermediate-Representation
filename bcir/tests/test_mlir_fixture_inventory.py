"""Every MLIR pass fixture on disk is executed by a runner, and every runner reference names
a fixture on disk (S0-3; laws.md L15: what a gate scans is reconciled against what the
repository tracks).

The 2026-07/08 assessment found `verify_timing_lifetime.mlir` and `cost_model_barrier.mlir`
carrying RUN lines and expected-error markers that nothing ran: fixtures only
`tools/docs/gen_status.py` counted. A fixture nothing executes is a claim of coverage with
no witness -- its expected-error lines assert nothing -- so this gate reads the fixture
directory and the runner scripts (`tools/wsl/*.sh`) themselves and refuses a fixture
without a runner and a runner reference without a fixture. The IRDL corpus and the
examples are run by directory loops, which the gate reads too.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path("mlir/test/passes")
RUNNERS = Path("tools/wsl")
# `"${T}/name.mlir"` in check_passes.sh (T = mlir/test/passes) and the spelled-out path in
# the other runners.
_REFERENCE = re.compile(r"(?:\$\{T\}|mlir/test/passes)/([A-Za-z0-9_]+\.mlir)")


def reconcile(fixture_dir: Path, runner_dir: Path) -> dict:
    fixtures = {path.name for path in fixture_dir.glob("*.mlir")}
    referenced: set[str] = set()
    for script in sorted(runner_dir.glob("*.sh")):
        referenced |= set(_REFERENCE.findall(script.read_text(encoding="utf-8")))
    return {
        "executed": sorted(fixtures & referenced),
        "unexecuted": sorted(fixtures - referenced),
        "dangling": sorted(referenced - fixtures),
    }


def test_every_pass_fixture_is_executed_and_every_reference_resolves() -> None:
    report = reconcile(_ROOT / FIXTURES, _ROOT / RUNNERS)
    assert len(report["executed"]) > 50, "the fixture directory or the runners moved"
    assert report["unexecuted"] == [], f"fixtures nothing runs: {report['unexecuted']}"
    assert report["dangling"] == [], f"runner references with no fixture: {report['dangling']}"


def test_the_directory_loops_over_examples_and_the_irdl_corpus_are_present() -> None:
    """Those two trees are run by loops, not by name; the loops must still exist."""
    passes = (_ROOT / RUNNERS / "check_passes.sh").read_text(encoding="utf-8")
    assert 'for f in "${ROOT}"/mlir/examples/*.mlir; do' in passes
    corpus = (_ROOT / "tools/irdl/check_corpus.sh").read_text(encoding="utf-8")
    assert 'for f in "${CORPUS}"/*.mlir; do' in corpus


def test_an_unexecuted_fixture_is_a_finding() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixtures, runners = Path(tmp) / "passes", Path(tmp) / "wsl"
        fixtures.mkdir()
        runners.mkdir()
        (fixtures / "run.mlir").write_text("// RUN: bcir-opt %s\n", encoding="utf-8")
        (fixtures / "inert.mlir").write_text("// RUN: bcir-opt %s\n", encoding="utf-8")
        (runners / "check.sh").write_text('"${BO}" "${T}/run.mlir"\n', encoding="utf-8")
        report = reconcile(fixtures, runners)
        assert report["executed"] == ["run.mlir"]
        assert report["unexecuted"] == ["inert.mlir"]
        assert report["dangling"] == []


def test_a_dangling_reference_is_a_finding() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixtures, runners = Path(tmp) / "passes", Path(tmp) / "wsl"
        fixtures.mkdir()
        runners.mkdir()
        (fixtures / "run.mlir").write_text("// RUN: bcir-opt %s\n", encoding="utf-8")
        (runners / "check.sh").write_text(
            'run_fc -bcir-verify "${T}/run.mlir"\nrun_fc -bcir-verify "${T}/ghost.mlir"\n'
            '"${BO}" mlir/test/passes/gone.mlir\n',
            encoding="utf-8",
        )
        report = reconcile(fixtures, runners)
        assert report["dangling"] == ["ghost.mlir", "gone.mlir"]
        assert report["unexecuted"] == []
