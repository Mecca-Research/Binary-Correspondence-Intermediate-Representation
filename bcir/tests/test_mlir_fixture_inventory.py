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


def active_shell_text(text: str) -> str:
    """The script with its comments removed, by bash's rule: a `#` that begins a word outside
    quotes starts a comment to the end of the line; a `#` inside '...' or "..." or in `${#v}` is
    data. A fixture named only in a comment -- a disabled invocation, or shell documentation --
    is not executed, and the gate must say so: it stayed green on the text of the comment, the
    exact scenario it exists to catch (laws.md L2)."""
    out: list[str] = []
    quote = ""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote:
            out.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:  # an escape inside "..."
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c == "\\" and i + 1 < n:  # an escaped character outside quotes
            out.append(c)
            out.append(text[i + 1])
            i += 2
            continue
        if c in "'\"":
            quote = c
            out.append(c)
            i += 1
            continue
        if c == "#" and (i == 0 or text[i - 1] in " \t\n;|&()<>"):  # a word-initial `#`
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def reconcile(fixture_dir: Path, runner_dir: Path) -> dict:
    fixtures = {path.name for path in fixture_dir.glob("*.mlir")}
    referenced: set[str] = set()
    for script in sorted(runner_dir.glob("*.sh")):
        referenced |= set(_REFERENCE.findall(active_shell_text(script.read_text(encoding="utf-8"))))
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


def test_a_commented_out_reference_is_not_an_execution() -> None:
    """L2: disabling a fixture's invocation with `#`, or naming the fixture in a comment, leaves
    it unexecuted -- the gate fires on both. A `#` inside a quoted string is data (the runners
    echo `#PF` and `#DF` in their labels), so a reference on such a line still counts."""
    with tempfile.TemporaryDirectory() as tmp:
        fixtures, runners = Path(tmp) / "passes", Path(tmp) / "wsl"
        fixtures.mkdir()
        runners.mkdir()
        for name in ("run", "disabled", "documented", "quoted"):
            (fixtures / f"{name}.mlir").write_text("// RUN: bcir-opt %s\n", encoding="utf-8")
        (runners / "check.sh").write_text(
            '"${BO}" "${T}/run.mlir" # see mlir/test/passes/documented.mlir for the shape\n'
            '# "${BO}" "${T}/disabled.mlir"\n'
            "  #  run_fc -bcir-verify mlir/test/passes/disabled.mlir\n"
            'echo "ok   #PF hardware error ${T}/quoted.mlir"; "${BO}" "${T}/quoted.mlir"\n',
            encoding="utf-8",
        )
        report = reconcile(fixtures, runners)
        assert report["executed"] == ["quoted.mlir", "run.mlir"]
        assert report["unexecuted"] == ["disabled.mlir", "documented.mlir"]
        assert report["dangling"] == []
    assert active_shell_text("a='#x' b=\"y #z\" ${#v} # c\n#d\n e \\# f") == (
        "a='#x' b=\"y #z\" ${#v} \n\n e \\# f"
    )
