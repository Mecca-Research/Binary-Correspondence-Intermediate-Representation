"""The BCIR Make runner (BUILD-7): what runs, what is reused, and what a task may touch.

A synthetic project with one tool, a shell script whose mode decides what it does, holds the
runner to the roadmap's gate -- a no-change second run executes nothing, a one-file change executes
exactly the targets that read it and their readers -- and to the rest of the contract: the artifact
cache restores a generation without running it, a task that writes a file no target claims or
leaves a claimed one unwritten fails (the observed footprint, MK3), a failure stops its readers,
at most `workers` tasks run at once and never two that write into one directory, and every task
that ran or was restored left one telemetry record, bound to its generation, that came back
through the ring.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from bcir.make import lower, parse
from bcir.make.__main__ import main
from bcir.make.run import execute


def _posix() -> bool:
    """BCIR Make names a tool by its POSIX path -- the grammar has no other spelling of an absolute
    path -- and runs POSIX programs, so on another host (a Windows runner) there is nothing here to
    judge, and each test that builds a BCIRfile over a temporary tree says so by returning."""
    return os.name == "posix"


TOOL = r"""#!/bin/sh
# tool MODE SRC DST: copy SRC to DST, as MODE says
set -e
mode=$1; src=$2; dst=$3
here=$(dirname "$0")
echo "$dst" >> "$here/../counter/runs"
case "$mode" in
  copy) cat "$src" > "$dst" ;;
  extra) cat "$src" > "$dst"; echo sneaky > "$(dirname "$dst")/sneaky.txt" ;;
  nothing) : ;;
  fail) exit 3 ;;
  staged)
    # as Clang writes an object: a temporary beside the output, renamed into it at the end
    cat "$src" > "$dst.tmp-$$"; sleep 0.4; mv "$dst.tmp-$$" "$dst" ;;
  slow)
    mkdir -p "$here/../counter/active"
    touch "$here/../counter/active/$$"
    n=$(ls "$here/../counter/active" | wc -l)
    echo "$n" >> "$here/../counter/widths"
    sleep 0.3
    rm -f "$here/../counter/active/$$"
    cat "$src" > "$dst" ;;
esac
"""


def _project(tmp: str, modes: dict[str, str] | None = None) -> tuple[Path, str]:
    root = Path(tmp) / "tree"
    for d in ("bin", "src", "counter"):
        (root / d).mkdir(parents=True, exist_ok=True)
    tool = root / "bin" / "tool"
    tool.write_text(TOOL, encoding="utf-8")
    tool.chmod(0o755)
    for name in ("a", "b", "c"):
        (root / "src" / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
    identity = "sha256:" + hashlib.sha256(tool.read_bytes()).hexdigest()
    m = {"a": "copy", "b": "copy", "c": "copy", "ab": "copy", **(modes or {})}
    text = f"""bcirfile 1
tool tool {tool} {identity}
target a
  reads src/a.txt
  writes out/a.txt
  run tool {m["a"]} src/a.txt out/a.txt
target b
  reads src/b.txt
  writes out/b.txt
  run tool {m["b"]} src/b.txt out/b.txt
target c
  reads src/c.txt
  writes out/c.txt
  run tool {m["c"]} src/c.txt out/c.txt
target ab
  reads out/a.txt out/b.txt
  writes out/ab/ab.txt
  run tool {m["ab"]} out/a.txt out/ab/ab.txt
"""
    return root, text


def _run(root: Path, text: str, **kw):
    bf = parse(text.encode("ascii"))
    meta = root.parent / "meta"
    return execute(
        bf,
        lower(bf),
        root,
        state_path=meta / "state.json",
        cache_dir=meta / "cache",
        log_dir=meta / "logs",
        **kw,
    )


def _runs(root: Path) -> list[str]:
    path = root / "counter" / "runs"
    return path.read_text(encoding="utf-8").split() if path.exists() else []


def test_a_second_run_executes_nothing_and_a_change_executes_its_readers():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp)
        first = _run(root, text)
        assert first.ok and sorted(first.executed()) == ["a", "ab", "b", "c"], first.results
        assert (root / "out/ab/ab.txt").read_text() == "a\n"
        second = _run(root, text)
        assert second.ok and second.executed() == [], second.results
        assert second.count("up-to-date") == 4
        (root / "src/b.txt").write_text("b changed\n")
        third = _run(root, text)
        assert sorted(third.executed()) == ["ab", "b"], third.results  # b, and ab which reads it
        (root / "src/c.txt").write_text("c changed\n")
        assert _run(root, text).executed() == ["c"]


def test_the_cache_restores_a_generation_without_running_it():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp)
        assert _run(root, text).ok
        runs = len(_runs(root))
        shutil.rmtree(root / "out")
        (root.parent / "meta" / "state.json").unlink()
        again = _run(root, text)
        assert again.ok and again.count("cache") == 4 and again.executed() == [], again.results
        assert len(_runs(root)) == runs, "a restored target ran its command"
        assert (root / "out/ab/ab.txt").read_text() == "a\n"
        # a cached file that is not its recorded bytes is no hit
        entry = next((root.parent / "meta" / "cache").glob("*/*/files/out/c.txt"))
        entry.write_text("tampered\n")
        shutil.rmtree(root / "out")
        (root.parent / "meta" / "state.json").unlink()
        third = _run(root, text)
        assert third.results["c"].outcome == "ran", third.results["c"]
        assert (root / "out/c.txt").read_text() == "c\n"


def test_the_observed_footprint_is_held_to_the_claims():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp, {"c": "extra"})
        report = _run(root, text, keep_going=True)
        assert report.results["c"].outcome == "failed"
        assert "wrote out/sneaky.txt, which no target claims" in report.results["c"].detail
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp, {"b": "nothing"})
        report = _run(root, text, keep_going=True)
        assert (
            report.results["b"].outcome == "failed"
            and "did not write out/b.txt" in report.results["b"].detail
        )
        assert (
            report.results["ab"].outcome == "skipped" and report.results["ab"].detail == "needs b"
        )
        assert report.results["a"].outcome == "ran" and report.results["c"].outcome == "ran"


def test_a_failure_stops_the_run_unless_told_to_keep_going():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp, {"a": "fail"})
        report = _run(root, text, workers=1)
        assert not report.ok and report.results["a"].outcome == "failed"
        assert all(report.results[n].outcome == "skipped" for n in ("b", "c", "ab")), report.results
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp, {"a": "fail"})
        report = _run(root, text, workers=1, keep_going=True)
        assert [report.results[n].outcome for n in ("a", "b", "c", "ab")] == [
            "failed",
            "ran",
            "ran",
            "skipped",
        ]


def _own_directories(text: str) -> str:
    """The project with a, b and c each writing into a directory of its own."""
    for name in ("a", "b", "c"):
        text = text.replace(f"out/{name}.txt", f"out/{name}/{name}.txt")
    return text


def test_at_most_the_workers_run_at_once_and_they_do():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp, {"a": "slow", "b": "slow", "c": "slow"})
        assert _run(root, _own_directories(text), workers=2).ok
        widths = [int(w) for w in (root / "counter" / "widths").read_text().split()]
        assert max(widths) == 2, widths
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp, {"a": "slow", "b": "slow", "c": "slow"})
        assert _run(root, text, workers=1).ok
        assert max(int(w) for w in (root / "counter" / "widths").read_text().split()) == 1


def test_two_targets_writing_into_one_directory_never_run_at_once():
    """What appears in a directory while a target runs is that target's only if no other target
    writing there runs beside it. Found by the first BCIR Make build of the rails with Clang, which
    stages an object as `<name>.o.tmp` beside it: a neighbour compiling into the same directory saw
    the temporary and failed as if it had written a file nothing claims. So the runner never starts
    a target whose output directory a running target writes into; targets with directories of
    their own still run two at once (the test above)."""
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp, {"a": "slow", "b": "slow", "c": "slow"})
        assert _run(root, text, workers=2).ok  # a, b and c all write into out/
        assert max(int(w) for w in (root / "counter" / "widths").read_text().split()) == 1
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp, {"a": "staged", "b": "staged", "c": "staged"})
        report = _run(root, text, workers=2)
        assert report.ok, {n: (r.outcome, r.detail) for n, r in report.results.items()}
        assert sorted(p.name for p in (root / "out").iterdir() if p.is_file()) == [
            "a.txt",
            "b.txt",
            "c.txt",
        ]


def test_every_task_leaves_one_record_through_the_ring():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp)
        bf = parse(text.encode())
        lowered = lower(bf)
        report = _run(root, text)
        records = report.telemetry
        assert len(records) == 4 and [r["seq"] for r in records] == [1, 2, 3, 4]
        by_claim = {r["claim_id"]: r for r in records}
        assert set(by_claim) == set(lowered.phase.values())
        for name, pid in lowered.phase.items():
            tag = report.results[name].tag
            assert by_claim[pid]["generation"] == (int(tag[:8], 16) or 1)
            assert by_claim[pid]["bytes"] == sum(
                (root / w).stat().st_size for w in bf.targets[pid - 1].writes
            )
        assert _run(root, text).telemetry == [], "an up-to-date target ran no task"


def test_the_cli_runs_and_says_so():
    if not _posix():
        return
    with tempfile.TemporaryDirectory() as tmp:
        root, text = _project(tmp)
        bcirfile = Path(tmp) / "BCIRfile"
        bcirfile.write_text(text, encoding="ascii")
        log = Path(tmp) / "telemetry.jsonl"
        argv = ["-f", str(bcirfile), "--root", str(root), "--telemetry", str(log)]
        assert main(argv) == 0
        assert len(log.read_text().splitlines()) == 4
        assert main(argv) == 0
        assert len(log.read_text().splitlines()) == 4, "a no-change run wrote telemetry"
        assert json.loads((root / "build/bcir-make/state.json").read_text()).keys() == {
            "a",
            "b",
            "c",
            "ab",
        }
        assert os.path.isdir(root / "build/bcir-make/cache")
