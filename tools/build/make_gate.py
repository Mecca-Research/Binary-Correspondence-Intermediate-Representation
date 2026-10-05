#!/usr/bin/env python3
"""The BCIR Make gate (BUILD-7, docs/BCIR_BUILD_ROADMAP.md §9): the runner builds the C rails and
runs the runtime gate's sections, and holds to what the roadmap promised of it.

    make_gate.py --cc gcc --cxx g++ [--out-dir build/bcir-make-gcc] [--scratch DIR]

1. tools/build/bcirfile.py writes the C rails' build with every section as a task, and bcir-make
   runs it: every target must run, and pass.
2. **A no-change second run executes zero tasks.**
3. **The sections' verdicts are their shell runs':** each section's script, run directly over the
   same binaries with the same CC and PYTHON, prints byte-identical stdout and stderr and exits with
   the same status as its task recorded -- after the section's declared `varies` masks, as the
   section-parity gate compares.
4. **A one-unit change executes exactly its dependents:** in a scratch copy of the tree carrying the
   first run's outputs, state and cache, one runtime unit is edited and bcir-make re-run. What it
   executes must be exactly the targets whose generation tag the edit changed, and those must be
   exactly the edited file's readers and their readers in the lowered graph, computed apart.

Exit 0 when every step holds, 1 with a finding, 2 when the gate cannot run: every exit is a verdict
(docs/security/laws.md L1).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bcir.make import check, generation_tags, lower, parse  # noqa: E402
from bcir.make.grammar import is_repo_path  # noqa: E402
from bcir.make.run import execute  # noqa: E402

# The unit the change step edits: a library unit with readers on every side of it (the library,
# the tools and harnesses that link it, their sections), appended a comment so it builds the same.
CHANGED_UNIT = "runtime/c/bcir_diag.c"
COPIED = ("bcir", "channels", "runtime", "tools", "pyproject.toml")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"bcir_build_{name}", ROOT / "tools" / "build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def readers_of(bf, path: str) -> set[str]:
    """The targets that read ``path`` -- by name, or under a tree they claim -- and every target
    that reads what one of them writes, transitively."""
    direct = {
        t.name
        for t in bf.targets
        if path in t.reads or any(path.startswith(d.rstrip("/") + "/") for d in t.reads_tree)
    }
    by_output = {w: t.name for t in bf.targets for w in t.writes}
    readers: dict[str, set[str]] = {}
    for t in bf.targets:
        for r in t.reads:
            if r in by_output:
                readers.setdefault(by_output[r], set()).add(t.name)
    found, todo = set(direct), list(direct)
    while todo:
        for nxt in readers.get(todo.pop(), ()):
            if nxt not in found:
                found.add(nxt)
                todo.append(nxt)
    return found


def _run(bf, root: Path, meta: Path, workers: int):
    return execute(
        bf,
        lower(bf),
        root,
        state_path=meta / "state.json",
        cache_dir=meta / "cache",
        log_dir=meta / "logs",
        workers=workers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cxx")
    parser.add_argument("--out-dir", default="build/bcir-make-gate")
    parser.add_argument("--scratch", type=Path, help="keep the change step's copy here")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if not is_repo_path(args.out_dir):
        print(f"make-gate: UNUSABLE: --out-dir {args.out_dir!r} is not a repo-relative path")
        return 2
    generator = _load("bcirfile")
    section_parity = _load("section_parity")
    sanitizer = _load("sanitizer")
    tools = {}
    for key, value in (
        ("cc", args.cc),
        ("cxx", args.cxx),
        ("ar", "ar"),
        ("python", sys.executable),
        ("bash", "bash"),
    ):
        try:
            tools[key] = None if value is None else generator.resolve_tool(value, key)
        except LookupError as exc:
            print(f"make-gate: UNUSABLE: {exc}")
            return 2
    tsan, _why = sanitizer.probe(tools["cc"], "thread")
    text, planned = generator.generate(
        tools["cc"],
        tools["cxx"],
        tools["ar"],
        tools["python"],
        tsan,
        out=args.out_dir,
        bash=tools["bash"],
        path_compilers=generator.path_compilers(),
    )
    bf = parse(text.encode("ascii"))
    findings = check(bf, ROOT, check_tools=True)
    if findings:
        print("make-gate: FAIL: the rails' BCIRfile breaks a law")
        for f in findings:
            print(f"  {f}")
        return 1
    meta = ROOT / args.out_dir / ".bcir-make"
    shutil.rmtree(ROOT / args.out_dir, ignore_errors=True)
    failures: list[str] = []
    t0 = time.monotonic()
    first = _run(bf, ROOT, meta, args.workers)
    print(
        f"make-gate: first run: {first.count('ran')} ran, {first.count('failed')} failed, "
        f"{first.count('skipped')} skipped of {len(bf.targets)} targets in {time.monotonic() - t0:.0f} s"
    )
    if not first.ok or first.count("ran") != len(bf.targets):
        for name, r in first.results.items():
            if r.outcome in ("failed", "skipped"):
                failures.append(f"{name}: {r.outcome} -- {r.detail}")
        print("\n".join(f"  FAIL {f}" for f in failures[:40]))
        print("make-gate: FAIL (the first run did not build and pass everything)")
        return 1
    second = _run(bf, ROOT, meta, args.workers)
    if second.executed():
        failures.append(
            f"a no-change second run executed {len(second.executed())}: {second.executed()[:8]}"
        )
    else:
        print(f"make-gate: second run: 0 executed, {second.count('up-to-date')} up to date")
    # the verdicts are the shell runs'
    shell = section_parity.posix_shell()
    manifest = json.loads((ROOT / "runtime" / "manifest.json").read_text(encoding="utf-8"))
    compared = 0
    for name in planned["sections"]:
        section = manifest["sections"][name]
        verdict = ROOT / args.out_dir / "verdicts" / name
        recorded = (
            int((verdict / "status").read_text().strip()),
            (verdict / "stdout").read_bytes(),
            (verdict / "stderr").read_bytes(),
        )
        # the task's own argv, relative to the root both run in: a script that names its binary
        # or itself in a diagnostic prints the same text either way
        binaries = [Path(args.out_dir) / "bin" / b for b in section["harnesses"]]
        direct = section_parity.run_section(
            shell, Path(section["script"]), binaries, tools["python"], 900.0, tools["cc"]
        )
        varies = section.get("varies", [])
        if (direct[0], section_parity.mask(direct[1], varies)[0], direct[2]) != (
            recorded[0],
            section_parity.mask(recorded[1], varies)[0],
            recorded[2],
        ):
            failures.append(f"section {name}: its task's verdict is not its shell run's")
        compared += 1
    print(f"make-gate: {compared} section verdicts identical to their shell runs")
    # a one-unit change executes exactly its dependents
    with tempfile.TemporaryDirectory(prefix="bcir-make-gate-") as tmp:
        scratch = args.scratch.resolve() if args.scratch else Path(tmp) / "tree"
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", *COPIED], cwd=ROOT, capture_output=True, check=False
        )
        if tracked.returncode != 0:
            print("make-gate: UNUSABLE: git ls-files did not run")
            return 2
        for rel in filter(None, tracked.stdout.decode("utf-8").split("\0")):
            src = ROOT / rel
            if src.is_file():
                (scratch / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, scratch / rel)
        shutil.copytree(ROOT / args.out_dir, scratch / args.out_dir, symlinks=True)
        before = generation_tags(bf, lower(bf), scratch)
        with open(scratch / CHANGED_UNIT, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("/* make-gate: an edit that changes no code */\n")
        after = generation_tags(bf, lower(bf), scratch)
        changed = {n for n in before if before[n] != after[n]}
        expected = readers_of(bf, CHANGED_UNIT)
        third = _run(bf, scratch, scratch / args.out_dir / ".bcir-make", args.workers)
        executed = set(third.executed())
        if not third.ok:
            failures.append("the change step's run failed")
        if changed != expected:
            failures.append(
                f"the edit changed {len(changed)} tags, the graph has {len(expected)} readers"
            )
        if executed != expected:
            extra, missed = sorted(executed - expected), sorted(expected - executed)
            failures.append(
                f"the change executed {len(executed)}, not its {len(expected)} dependents (extra {extra[:5]}, missed {missed[:5]})"
            )
        else:
            print(
                f"make-gate: editing {CHANGED_UNIT} executed exactly its {len(expected)} dependents "
                f"of {len(bf.targets)} targets"
            )
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        print(f"make-gate: FAIL ({len(failures)} finding(s))")
        return 1
    print("make-gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
