#!/usr/bin/env python3
"""Fail-closed independent-review harness.

Missing command, failed start, timeout, non-zero exit, undecodable or
unparseable JSON is FAIL — never a silent skip and never a traceback in
place of the structured report. `--self-check` proves the contract with
synthetic reviewer programs, driving each failure mode.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_KEYS = ("passed", "security_concerns", "logic_errors", "summary")
REVIEW_TIMEOUT = 120


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"duplicate keys in review JSON: {duplicates}")
    return dict(pairs)


def parse_review(text: str) -> dict[str, Any]:
    payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("review JSON must be an object")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"review JSON missing keys: {missing}")
    if not isinstance(payload["passed"], bool):
        raise ValueError("passed must be a boolean")
    if not isinstance(payload["security_concerns"], list):
        raise ValueError("security_concerns must be a list")
    if not isinstance(payload["logic_errors"], list):
        raise ValueError("logic_errors must be a list")
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise ValueError("summary must be a nonempty string")
    if payload["security_concerns"] or payload["logic_errors"]:
        payload["passed"] = False
    return payload


def run_reviewer(command: list[str], cwd: Path, timeout: int = REVIEW_TIMEOUT) -> dict[str, Any]:
    if not command:
        return {
            "state": "FAIL",
            "reason": "no reviewer command configured",
            "fail_closed": True,
        }
    try:
        # Bytes, not text=True: a reviewer emitting non-UTF-8 must become a
        # structured FAIL below, not a UnicodeDecodeError traceback here.
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "state": "FAIL",
            "reason": f"reviewer timed out after {timeout}s",
            "fail_closed": True,
        }
    except OSError as exc:
        return {
            "state": "FAIL",
            "reason": f"reviewer failed to start: {type(exc).__name__}: {exc}",
            "fail_closed": True,
        }
    if result.returncode != 0:
        return {
            "state": "FAIL",
            "reason": f"reviewer exited {result.returncode}",
            "fail_closed": True,
            "stderr_tail": result.stderr.decode("utf-8", "replace")[-400:],
        }
    try:
        stdout = result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "state": "FAIL",
            "reason": "reviewer output is not UTF-8",
            "fail_closed": True,
        }
    try:
        payload = parse_review(stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "state": "FAIL",
            "reason": f"unparseable reviewer output: {exc}",
            "fail_closed": True,
        }
    return {
        "state": "PASS" if payload["passed"] else "FAIL",
        "fail_closed": True,
        "review": payload,
    }


def env_command(raw: str, posix: bool | None = None) -> list[str] | dict[str, Any]:
    """Parse BCIR_INDEPENDENT_REVIEW_CMD with shell quoting; a malformed value
    is a structured FAIL report, never a ValueError traceback.

    On Windows the non-POSIX rules apply so backslashed executable paths
    survive; surrounding quotes are stripped per token there."""
    if not raw:
        return []
    if posix is None:
        posix = os.name != "nt"
    try:
        tokens = shlex.split(raw, posix=posix)
    except ValueError as exc:
        return {
            "state": "FAIL",
            "reason": f"malformed BCIR_INDEPENDENT_REVIEW_CMD: {exc}",
            "fail_closed": True,
        }
    if not posix:
        tokens = [
            token[1:-1]
            if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'"
            else token
            for token in tokens
        ]
    return tokens


def self_check() -> dict[str, Any]:
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.py"
        bad = Path(tmp) / "bad.py"
        empty = Path(tmp) / "empty.py"
        dupkeys = Path(tmp) / "dupkeys.py"
        nullsum = Path(tmp) / "nullsum.py"
        rawbytes = Path(tmp) / "rawbytes.py"
        sleeper = Path(tmp) / "sleeper.py"
        good.write_text(
            "print('{\"passed\": true, \"security_concerns\": [], "
            "\"logic_errors\": [], \"summary\": \"clean\"}')\n",
            encoding="utf-8",
        )
        bad.write_text("print('not-json')\n", encoding="utf-8")
        empty.write_text("print('')\n", encoding="utf-8")
        dupkeys.write_text(
            "print('{\"passed\": false, \"passed\": true, \"security_concerns\": [], "
            "\"logic_errors\": [], \"summary\": \"clean\"}')\n",
            encoding="utf-8",
        )
        nullsum.write_text(
            "print('{\"passed\": true, \"security_concerns\": [], "
            "\"logic_errors\": [], \"summary\": null}')\n",
            encoding="utf-8",
        )
        rawbytes.write_text(
            "import sys\nsys.stdout.buffer.write(b'\\xff\\xfe not utf-8')\n",
            encoding="utf-8",
        )
        sleeper.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        python = sys.executable
        cases.append(("missing-command", run_reviewer([], Path(tmp))))
        cases.append(("missing-executable",
                      run_reviewer([str(Path(tmp) / "absent-reviewer")], Path(tmp))))
        cases.append(("valid-json", run_reviewer([python, str(good)], Path(tmp))))
        cases.append(("unparseable", run_reviewer([python, str(bad)], Path(tmp))))
        cases.append(("empty-output", run_reviewer([python, str(empty)], Path(tmp))))
        cases.append(("duplicate-keys", run_reviewer([python, str(dupkeys)], Path(tmp))))
        cases.append(("null-summary", run_reviewer([python, str(nullsum)], Path(tmp))))
        cases.append(("non-utf8", run_reviewer([python, str(rawbytes)], Path(tmp))))
        cases.append(("timeout", run_reviewer([python, str(sleeper)], Path(tmp), timeout=1)))
    env_bad = env_command("python -c 'unterminated")
    cases.append(("malformed-env-command",
                  env_bad if isinstance(env_bad, dict) else {"state": "PASS"}))
    expected = {
        "missing-command": "FAIL",
        "missing-executable": "FAIL",
        "valid-json": "PASS",
        "unparseable": "FAIL",
        "empty-output": "FAIL",
        "duplicate-keys": "FAIL",
        "null-summary": "FAIL",
        "non-utf8": "FAIL",
        "timeout": "FAIL",
        "malformed-env-command": "FAIL",
    }
    mismatches = [
        name for name, report in cases if report["state"] != expected[name]
    ]
    return {
        "state": "PASS" if not mismatches else "FAIL",
        "cases": {name: report["state"] for name, report in cases},
        "mismatches": mismatches,
        "fail_closed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="independent_review")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--json-out", type=Path)
    # argparse.REMAINDER: everything AFTER --command on the command line is
    # the reviewer's argv, its own flags included. Argv position decides, not
    # declaration order — put --json-out and friends BEFORE --command.
    parser.add_argument(
        "--command", nargs=argparse.REMAINDER,
        help="reviewer argv; must be the final option — everything after it "
             "belongs to the reviewer",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        report = self_check()
    else:
        command: list[str] | dict[str, Any]
        if args.command:
            command = args.command
        else:
            command = env_command(os.environ.get("BCIR_INDEPENDENT_REVIEW_CMD", ""))
        if isinstance(command, dict):
            report = command
        else:
            report = run_reviewer(command, args.root)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"independent_review: {report['state']} fail_closed={report.get('fail_closed')}")
    if report.get("reason"):
        print(f"  {report['reason']}")
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
