#!/usr/bin/env python3
"""One harness for proving a gate can fail, and for proving the proof happened.

A RED sweep is how this repository establishes that a check works: inject the
defect the check exists to catch, watch the check fire, restore, and only then
believe the green run (`docs/security/laws.md` L2 -- a gate must be able to fire).
The method is written down in `.claude/skills/bcir-cicd`; the mechanics were not
written down anywhere, so each sweep grew its own copy of inject/run/restore.

Four copies were written in one session. One of them was wrong, and it was wrong
in the direction that matters: it reported a defect as **not caught** when the
defect had never reached the interpreter. Restoring a fault whose replacement was
the same length can satisfy CPython's `(mtime, size)` staleness test, so the gate
subprocess imported a cached `.pyc` compiled from the *faulted* source -- or, in
the restore direction, kept running faulted bytecode after the file was clean. A
sweep that cannot tell "the check did not fire" from "the fault never arrived"
proves nothing in either direction, and reads as evidence in both.

So this module is the single implementation (L14 -- one predicate per repeated
defect), and it refuses to produce a verdict it cannot stand behind:

* **Bytecode is discarded** before the gate runs and again after the restore, so
  no process under test can load a module compiled from different source.
* **The injection is confirmed to have landed** -- the file is re-read and its
  digest compared -- so an anchor that silently matched nothing is a harness
  error, not a passing fault.
* **The restore is confirmed exact**, by digest, before the next fault is
  injected. A sweep that corrupts the tree it is auditing is worse than no sweep.
* **A vacuous sweep is refused**: an empty fault table, or a table whose faults
  all fail to anchor, is a failure rather than a clean run (L2).
* **The clean tree must be GREEN first.** "The gate went red with the fault in"
  means nothing if it was already red; that is the control, and it is mandatory.
* **Every exit is a verdict** (L1). A gate that times out, dies on a signal, or
  cannot be launched is reported as such, never as a caught or uncaught fault.

Use it as a library from a sweep script, or run a fault table straight from JSON:

    python3 tools/testing/red_sweep.py --faults training/tools/red/database.json
    python3 tools/testing/red_sweep.py --faults ... --only S18 --json-out r.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

#: How a gate in this tree prints a finding: two spaces, a dash, the check's name,
#: then a colon. The name is what a fault declares it expects to fire.
FINDING = re.compile(r"^\s*-\s+([A-Za-z0-9_. -]+?):")

#: A gate may take a while; a sweep runs it once per fault plus once for the
#: control. Per-gate rather than per-sweep so the bound scales with the campaign
#: (`docs/security/laws.md` L8).
DEFAULT_TIMEOUT_SECONDS = 5400


class SweepError(RuntimeError):
    """The sweep cannot produce a trustworthy verdict. Never a quiet degradation."""


@dataclass(frozen=True)
class Fault:
    """One defect to inject, and the check that must notice it.

    `old` and `new` are exact text. `old` must occur **exactly once** in the file:
    an anchor that matches twice would inject two defects and attribute the result
    to one, and an anchor that matches zero times would inject nothing while the
    sweep reported a passing gate as a caught fault.
    """

    label: str
    expects: str
    path: Path
    old: str
    new: str

    def anchor_count(self, text: str) -> int:
        return text.count(self.old)


@dataclass
class Result:
    label: str
    expects: str
    fired: tuple[str, ...] = ()
    exit_code: int | None = None
    verdict: str = "UNKNOWN"
    detail: str = ""

    @property
    def caught(self) -> bool:
        return self.verdict == "RED"


@dataclass
class Sweep:
    """A fault table and the gate it is run against."""

    command: list[str]
    faults: list[Fault]
    cwd: Path = REPO_ROOT
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    results: list[Result] = field(default_factory=list)

    # -- the two operations that make a verdict trustworthy --------------------

    def discard_bytecode(self) -> int:
        """Remove every cached module under the tree. Returns how many were removed.

        Called before the gate runs and after every restore. The cost is a
        recompile; the alternative is a verdict about source that was never
        executed.
        """
        removed = 0
        for cache in self.cwd.rglob("__pycache__"):
            for stale in cache.glob("*.pyc"):
                stale.unlink(missing_ok=True)
                removed += 1
        return removed

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # -- running ---------------------------------------------------------------

    def run_gate(self) -> tuple[int | None, set[str], str]:
        """Run the gate once. A launch failure is a verdict, not an exception (L1)."""
        self.discard_bytecode()
        try:
            done = subprocess.run(
                self.command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as expired:
            tail = (expired.stdout or b"")[-2000:]
            return None, set(), f"timed out after {self.timeout}s; tail: {tail!r}"
        except OSError as exc:
            return None, set(), f"could not launch {self.command[0]!r}: {exc}"
        output = done.stdout + done.stderr
        fired = {
            match.group(1).strip() for line in output.splitlines() if (match := FINDING.match(line))
        }
        return done.returncode, fired, output

    def control(self) -> None:
        """The gate must be GREEN before anything is injected.

        Without this the sweep cannot distinguish "the fault made it red" from "it
        was red already", and every row below would be meaningless.
        """
        code, fired, output = self.run_gate()
        if code is None:
            raise SweepError(f"control run did not complete: {output}")
        if code != 0:
            tail = "\n    ".join(output.strip().splitlines()[-8:])
            raise SweepError(
                "control run is already FAILING, so a red run with a fault injected "
                f"would prove nothing (exit {code}, checks fired: {sorted(fired) or 'none'})"
                f"\n    {tail}"
            )

    def inject(self, fault: Fault) -> str:
        """Apply one fault, and prove it landed. Returns the original bytes' digest."""
        original = fault.path.read_bytes()
        before = hashlib.sha256(original).hexdigest()
        text = original.decode("utf-8")
        occurrences = fault.anchor_count(text)
        if occurrences != 1:
            raise SweepError(
                f"anchor for {fault.label!r} occurs {occurrences} time(s) in "
                f"{fault.path.name}; a fault table entry must name exactly one site"
            )
        fault.path.write_text(text.replace(fault.old, fault.new), encoding="utf-8", newline="")
        if self.digest(fault.path) == before:
            raise SweepError(
                f"injecting {fault.label!r} did not change {fault.path.name}; the "
                "replacement is identical to the anchor, so nothing was injected"
            )
        return before

    def restore(self, fault: Fault, original: bytes, expected_digest: str) -> None:
        """Put the file back, prove it is byte-identical, and drop stale bytecode."""
        fault.path.write_bytes(original)
        actual = self.digest(fault.path)
        if actual != expected_digest:
            raise SweepError(
                f"restoring {fault.path} left different bytes "
                f"({actual[:12]} != {expected_digest[:12]}); the tree is now dirty"
            )
        self.discard_bytecode()

    def run(self, only: str | None = None) -> int:
        """Run the whole table. Returns the number of faults NOT caught."""
        selected = [f for f in self.faults if only is None or only in f.label or only in f.expects]
        if not selected:
            raise SweepError(
                "no faults selected: a sweep that injects nothing cannot show that "
                "anything works (L2)"
            )

        self.control()

        missed = 0
        for fault in selected:
            original = fault.path.read_bytes()
            expected = hashlib.sha256(original).hexdigest()
            result = Result(label=fault.label, expects=fault.expects)
            try:
                self.inject(fault)
                code, fired, output = self.run_gate()
                result.fired = tuple(sorted(fired))
                result.exit_code = code
                if code is None:
                    result.verdict = "UNAVAILABLE"
                    result.detail = output
                elif code == 0:
                    result.verdict = "NOT CAUGHT"
                    result.detail = "the gate passed with the defect in place"
                elif not any(name.startswith(fault.expects) for name in fired):
                    result.verdict = "WRONG CHECK"
                    result.detail = (
                        f"the gate failed but {fault.expects!r} did not fire; "
                        f"fired: {result.fired or '(none named)'}"
                    )
                else:
                    result.verdict = "RED"
            finally:
                self.restore(fault, original, expected)
            self.results.append(result)
            missed += not result.caught
        return missed

    # -- reporting -------------------------------------------------------------

    def report(self) -> str:
        width = max((len(r.label) for r in self.results), default=10)
        lines = [
            "=" * (width + 46),
            f"{'injected defect'.ljust(width)}  {'check fired':22} verdict",
        ]
        lines.append("=" * (width + 46))
        for result in self.results:
            fired = ", ".join(result.fired) or "(none)"
            lines.append(f"{result.label.ljust(width)}  {fired[:22]:22} {result.verdict}")
            if not result.caught:
                lines.append(f"    expected {result.expects!r}; {result.detail.strip()[:400]}")
        caught = sum(1 for r in self.results if r.caught)
        lines.append("=" * (width + 46))
        lines.append(f"{len(self.results)} defect(s) injected, {caught} caught by their own check")
        return "\n".join(lines)


def load_table(path: Path) -> tuple[list[str], list[Fault]]:
    """Read a fault table. A malformed table is a refusal, never a partial sweep."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SweepError(f"{path} is not a readable fault table: {exc}") from exc
    if not isinstance(data, dict) or "command" not in data or "faults" not in data:
        raise SweepError(f"{path} must be an object with 'command' and 'faults'")
    command = data["command"]
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise SweepError(f"{path}: 'command' must be a list of strings")
    command = [sys.executable if part == "@python" else part for part in command]

    faults = []
    for index, entry in enumerate(data["faults"]):
        missing = {"label", "expects", "path", "old", "new"} - set(entry)
        if missing:
            raise SweepError(f"{path}: fault {index} is missing {sorted(missing)}")
        target = REPO_ROOT / entry["path"]
        if not target.is_file():
            raise SweepError(f"{path}: fault {index} names {entry['path']}, which does not exist")
        faults.append(
            Fault(
                label=entry["label"],
                expects=entry["expects"],
                path=target,
                old=entry["old"],
                new=entry["new"],
            )
        )
    if not faults:
        raise SweepError(f"{path} declares no faults; an empty sweep proves nothing (L2)")
    return command, faults


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--faults", type=Path, required=True, help="a fault table, as JSON")
    parser.add_argument("--only", help="run just the faults whose label or check contains this")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    try:
        command, faults = load_table(args.faults)
        sweep = Sweep(command=command, faults=faults, timeout=args.timeout)
        missed = sweep.run(only=args.only)
    except SweepError as exc:
        print(f"red_sweep: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("red_sweep: interrupted; the tree has been restored", file=sys.stderr)
        return EXIT_FAILED

    print(sweep.report())
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "table": str(args.faults),
                    "command": command,
                    "results": [
                        {
                            "label": r.label,
                            "expects": r.expects,
                            "fired": list(r.fired),
                            "exit_code": r.exit_code,
                            "verdict": r.verdict,
                        }
                        for r in sweep.results
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return EXIT_FAILED if missed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
