#!/usr/bin/env python3
"""Fail-closed independent-review harness.

Missing command, non-zero exit, or unparseable JSON is FAIL — never a silent skip.
`--self-check` proves the contract with synthetic reviewer programs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_KEYS = ("passed", "security_concerns", "logic_errors", "summary")


def parse_review(text: str) -> dict[str, Any]:
    payload = json.loads(text)
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
    if payload["security_concerns"] or payload["logic_errors"]:
        payload["passed"] = False
    return payload


def run_reviewer(command: list[str], cwd: Path) -> dict[str, Any]:
    if not command:
        return {
            "state": "FAIL",
            "reason": "no reviewer command configured",
            "fail_closed": True,
        }
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False, timeout=120,
        )
    except FileNotFoundError as exc:
        return {
            "state": "FAIL",
            "reason": f"reviewer could not start: {exc}",
            "fail_closed": True,
        }
    except subprocess.TimeoutExpired:
        return {
            "state": "FAIL",
            "reason": "reviewer timed out",
            "fail_closed": True,
        }
    if result.returncode != 0:
        return {
            "state": "FAIL",
            "reason": f"reviewer exited {result.returncode}",
            "fail_closed": True,
            "stderr_tail": result.stderr[-400:],
        }
    try:
        payload = parse_review(result.stdout)
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


def self_check() -> dict[str, Any]:
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.py"
        bad = Path(tmp) / "bad.py"
        empty = Path(tmp) / "empty.py"
        good.write_text(
            "print('{\"passed\": true, \"security_concerns\": [], "
            "\"logic_errors\": [], \"summary\": \"clean\"}')\n",
            encoding="utf-8",
        )
        bad.write_text("print('not-json')\n", encoding="utf-8")
        empty.write_text("print('')\n", encoding="utf-8")
        python = sys.executable
        cases.append(("missing-command", run_reviewer([], Path(tmp))))
        cases.append(("missing-executable", run_reviewer(
            [str(Path(tmp) / "no-such-reviewer")], Path(tmp),
        )))
        cases.append(("valid-json", run_reviewer([python, str(good)], Path(tmp))))
        cases.append(("unparseable", run_reviewer([python, str(bad)], Path(tmp))))
        cases.append(("empty-output", run_reviewer([python, str(empty)], Path(tmp))))
    expected = {
        "missing-command": "FAIL",
        "missing-executable": "FAIL",
        "valid-json": "PASS",
        "unparseable": "FAIL",
        "empty-output": "FAIL",
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
    parser.add_argument(
        "--command", nargs=argparse.REMAINDER,
        help="reviewer argv; must be last so reviewer flags such as --format are kept",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        report = self_check()
    else:
        command = args.command or (
            os.environ.get("BCIR_INDEPENDENT_REVIEW_CMD", "").split() or None
        )
        report = run_reviewer(command or [], args.root)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"independent_review: {report['state']} fail_closed={report.get('fail_closed')}")
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
