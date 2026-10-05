"""`bcir-make` (python -m bcir.make): judge, plan and run a BCIRfile.

    bcir-make [--dry-run] [-f BCIRfile] [--root DIR] [--state FILE] [--workers N] [--check-tools]
              [--cache DIR] [--logs DIR] [--telemetry FILE] [--keep-going]

Prints the laws' verdict (MK0-MK5). With ``--dry-run`` it then prints the plan -- every target's
generation tag, whether it runs or is reused against the recorded ``--state``, and the waves it runs
in. Without it, the runner (bcir.make.run, BUILD-7) executes the plan: each target up to date,
restored from the artifact cache, or run, at most ``--workers`` at once, its observed footprint
held to its claims, its generation recorded, and one telemetry record per task appended to
``--telemetry``. Exit 0 with a plan or a clean run, 1 when the BCIRfile breaks a law (a grammar
error is MK0) or a target fails, 2 when nothing can be judged (an unreadable file or state, a bad
option). Every exit is a verdict (docs/security/laws.md L1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .grammar import GrammarError, parse
from .laws import check, lower
from .plan import decide, generation_tags, load_state, schedule
from .run import execute


def dry_run(
    data: bytes,
    root: Path,
    *,
    source: str = "BCIRfile",
    state: dict[str, str] | None = None,
    workers: int = 2,
    check_tools: bool = False,
) -> tuple[int, str]:
    """The verdict and the plan, as the exit status and the text `--dry-run` prints."""
    out: list[str] = []
    try:
        bf = parse(data)
    except GrammarError as exc:
        out.append(f"bcir-make: {source}: not a BCIRfile")
        out.append("laws: FAIL (1 finding)")
        out.append(f"  MK0: {exc}")
        return 1, "\n".join(out) + "\n"
    out.append(f"bcir-make: {source}: {len(bf.tools)} tools, {len(bf.targets)} targets (dry run)")
    findings = check(bf, root, check_tools=check_tools)
    if findings:
        out.append(f"laws: FAIL ({len(findings)} finding(s))")
        out += [f"  {f}" for f in findings]
        return 1, "\n".join(out) + "\n"
    out.append("laws: PASS (MK0-MK5)")
    lowered = lower(bf)
    tags = generation_tags(bf, lowered, root)
    decisions = decide(bf, tags, root, state)
    waves = schedule(bf, lowered, workers)
    runs = sum(1 for d in decisions.values() if d.action == "run")
    out.append(
        f"plan: {runs} to run, {len(bf.targets) - runs} to reuse, {workers} worker(s), {len(waves)} wave(s)"
    )
    for i, wave in enumerate(waves, 1):
        out.append(f"wave {i}: {' '.join(wave)}")
    for t in bf.targets:
        d = decisions[t.name]
        out.append(f"{d.action} {t.name} {d.tag[:16]} ({d.why})")
    return 0, "\n".join(out) + "\n"


def run(
    data: bytes,
    root: Path,
    *,
    source: str,
    state_path: Path,
    cache_dir: Path,
    log_dir: Path,
    telemetry: Path | None,
    workers: int,
    check_tools: bool,
    keep_going: bool,
) -> tuple[int, str]:
    """Judge, then run: the runner only ever sees a lawful BCIRfile. The exit status and the text."""
    out: list[str] = []
    try:
        bf = parse(data)
    except GrammarError as exc:
        out += [f"bcir-make: {source}: not a BCIRfile", "laws: FAIL (1 finding)", f"  MK0: {exc}"]
        return 1, "\n".join(out) + "\n"
    out.append(f"bcir-make: {source}: {len(bf.tools)} tools, {len(bf.targets)} targets")
    findings = check(bf, root, check_tools=check_tools)
    if findings:
        out.append(f"laws: FAIL ({len(findings)} finding(s))")
        out += [f"  {f}" for f in findings]
        return 1, "\n".join(out) + "\n"
    out.append("laws: PASS (MK0-MK5)")
    t0 = time.monotonic()
    report = execute(
        bf,
        lower(bf),
        root,
        state_path=state_path,
        cache_dir=cache_dir,
        log_dir=log_dir,
        workers=workers,
        keep_going=keep_going,
    )
    out.append(
        f"run: {report.count('ran')} ran, {report.count('cache')} from the cache, "
        f"{report.count('up-to-date')} up to date, {report.count('failed')} failed, "
        f"{report.count('skipped')} skipped; {workers} worker(s), {time.monotonic() - t0:.1f} s"
    )
    for name, result in report.results.items():
        if result.outcome != "up-to-date":
            detail = f" -- {result.detail}" if result.detail else ""
            out.append(f"{result.outcome} {name} {result.tag[:16]}{detail}")
    if telemetry is not None:
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        with open(telemetry, "a", encoding="utf-8", newline="\n") as handle:
            for record in report.telemetry:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return (0 if report.ok else 1), "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bcir-make", description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="judge and plan; run nothing")
    parser.add_argument("-f", "--file", type=Path, default=Path("BCIRfile"), help="the BCIRfile")
    parser.add_argument("--root", type=Path, default=Path("."), help="the tree its paths are under")
    parser.add_argument("--state", type=Path, help="recorded generation tags (JSON target -> tag)")
    parser.add_argument("--workers", type=int, default=2, help="the cap on concurrent targets (2)")
    parser.add_argument(
        "--check-tools", action="store_true", help="hold each tool's bytes to its identity"
    )
    parser.add_argument(
        "--cache", type=Path, help="the artifact cache (default: <state dir>/cache)"
    )
    parser.add_argument(
        "--logs", type=Path, help="per-target command logs (default: <state dir>/logs)"
    )
    parser.add_argument(
        "--telemetry", type=Path, help="append the task telemetry here (JSON lines)"
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="after a failure, run what does not need it"
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    if args.workers < 1:
        print("bcir-make: UNUSABLE: --workers must be at least 1", file=sys.stderr)
        return 2
    try:
        data = args.file.read_bytes()
    except OSError as exc:
        print(f"bcir-make: UNUSABLE: {args.file}: {exc.strerror}", file=sys.stderr)
        return 2
    root = args.root.resolve()
    if not root.is_dir():
        print(f"bcir-make: UNUSABLE: {root} is no directory", file=sys.stderr)
        return 2
    if args.dry_run:
        state = None
        if args.state is not None:
            try:
                state = load_state(args.state)
            except (OSError, ValueError) as exc:
                print(f"bcir-make: UNUSABLE: {exc}", file=sys.stderr)
                return 2
        rc, text = dry_run(
            data,
            root,
            source=str(args.file),
            state=state,
            workers=args.workers,
            check_tools=args.check_tools,
        )
        sys.stdout.write(text)
        return rc
    state_path = args.state or root / "build" / "bcir-make" / "state.json"
    if state_path.exists():
        try:
            load_state(state_path)
        except (OSError, ValueError) as exc:
            print(f"bcir-make: UNUSABLE: {exc}", file=sys.stderr)
            return 2
    rc, text = run(
        data,
        root,
        source=str(args.file),
        state_path=state_path,
        cache_dir=args.cache or state_path.parent / "cache",
        log_dir=args.logs or state_path.parent / "logs",
        telemetry=args.telemetry,
        workers=args.workers,
        check_tools=args.check_tools,
        keep_going=args.keep_going,
    )
    sys.stdout.write(text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
