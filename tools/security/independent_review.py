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
import threading
from pathlib import Path
from typing import Any

try:
    from tools.security.proc_bounds import put_down_group
    from tools.security.report_hygiene import mapped, reject_duplicate_keys
    from tools.security.scan_secrets import redacted_text
except ModuleNotFoundError:  # script execution: sys.path[0] is tools/security
    from proc_bounds import put_down_group
    from report_hygiene import mapped, reject_duplicate_keys
    from scan_secrets import redacted_text

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_KEYS = ("passed", "security_concerns", "logic_errors", "summary")
REVIEW_TIMEOUT = 120
REVIEW_OUTPUT_CAP = 1 << 20  # 1 MiB per stream; a review JSON is small
REVIEW_PIPE_GRACE = 5.0  # seconds to wait for drains after the reviewer exits


def _put_down(proc: subprocess.Popen) -> None:
    """Kill the reviewer's whole process tree. One predicate for every rail:
    a descendant that inherited the output pipes must not outlive the bound
    by holding them open, on POSIX (session kill) or Windows (tree kill)."""
    put_down_group(proc)


# The duplicate-key refusal this rail has always had now lives in
# report_hygiene, because the dependency audit needed the same one and
# a second copy is how the campaign's repeat defects start (L14).
_reject_duplicate_keys = reject_duplicate_keys


def _reject_constant(name: str) -> Any:
    # NaN/Infinity are a permissive json.loads extension, not JSON; the
    # fail-closed parse contract rejects them.
    raise ValueError(f"non-standard JSON constant: {name}")


def parse_review(text: str) -> dict[str, Any]:
    payload = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
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


def _drain(stream: Any, chunks: list[bytes], cap: int, state: dict[str, Any]) -> None:
    """Read one reviewer pipe under a byte budget. On overflow the reviewer
    is put down and the pipe kept draining unretained — a full pipe would
    block the child forever while the job waits to fail it."""
    received = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        if state["overflow"]:
            continue
        received += len(chunk)
        if received > cap:
            state["overflow"] = True
            _put_down(state["proc"])
            continue
        chunks.append(chunk)


def run_reviewer(command: list[str], cwd: Path, timeout: int = REVIEW_TIMEOUT) -> dict[str, Any]:
    if not command:
        return {
            "state": "FAIL",
            "reason": "no reviewer command configured",
            "fail_closed": True,
        }
    cap = REVIEW_OUTPUT_CAP
    try:
        # Popen with bounded drains, not run(): capture_output accumulates
        # without limit, so a flooding reviewer would OOM the job before the
        # timeout could ever produce the structured report. Bytes, not text:
        # non-UTF-8 output must become a structured FAIL below.
        proc = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    except OSError as exc:
        return {
            "state": "FAIL",
            "reason": f"reviewer failed to start: {type(exc).__name__}: {exc}",
            "fail_closed": True,
        }
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    state: dict[str, Any] = {"overflow": False, "proc": proc}
    readers = (
        threading.Thread(target=_drain, args=(proc.stdout, out_chunks, cap, state), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err_chunks, cap, state), daemon=True),
    )
    for reader in readers:
        reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _put_down(proc)
        proc.wait()
        for reader in readers:
            reader.join(REVIEW_PIPE_GRACE)
        return {
            "state": "FAIL",
            "reason": f"reviewer timed out after {timeout}s",
            "fail_closed": True,
        }
    # Bounded joins: a descendant that inherited the pipes and outlived the
    # reviewer would otherwise hold the drains (and this harness) open
    # indefinitely; after the grace the session is put down and re-joined.
    for reader in readers:
        reader.join(REVIEW_PIPE_GRACE)
    if any(reader.is_alive() for reader in readers):
        _put_down(proc)
        for reader in readers:
            reader.join(REVIEW_PIPE_GRACE)
        if any(reader.is_alive() for reader in readers):
            return {
                "state": "FAIL",
                "reason": "reviewer descendants held its output pipes open",
                "fail_closed": True,
            }
    if state["overflow"]:
        return {
            "state": "FAIL",
            "reason": f"reviewer output exceeded {cap} bytes",
            "fail_closed": True,
        }
    if proc.returncode != 0:
        return {
            "state": "FAIL",
            "reason": f"reviewer exited {proc.returncode}",
            "fail_closed": True,
            "stderr_tail": b"".join(err_chunks).decode("utf-8", "replace")[-400:],
        }
    try:
        stdout = b"".join(out_chunks).decode("utf-8")
    except UnicodeDecodeError:
        return {
            "state": "FAIL",
            "reason": "reviewer output is not UTF-8",
            "fail_closed": True,
        }
    try:
        payload = parse_review(stdout)
    except (ValueError, json.JSONDecodeError, RecursionError) as exc:
        # RecursionError: json.loads raises it on a depth-bomb payload; that
        # is unparseable reviewer output under the same contract.
        return {
            "state": "FAIL",
            "reason": f"unparseable reviewer output: {exc}",
            "fail_closed": True,
        }
    return {
        "state": "PASS" if payload["passed"] else "FAIL",
        "fail_closed": True,
        # A reviewer quotes the code it is reviewing. Its findings are
        # therefore the one report field GUARANTEED to carry whatever
        # secret it just discovered, and this rail copied them verbatim
        # into --json-out. Redacted through the scan's own predicate, so
        # this cannot remove less than the scanner would report; the
        # reviewer's prose around the value survives, and `passed` stays
        # a boolean because `mapped` only touches strings (L7).
        "review": mapped(payload, redacted_text),
    }


def env_command(raw: str, posix: bool | None = None) -> list[str] | dict[str, Any]:
    """Parse BCIR_INDEPENDENT_REVIEW_CMD with shell quoting; a malformed value
    is a structured FAIL report, never a ValueError traceback.

    On Windows the non-POSIX rules apply so backslashed executable paths
    survive; surrounding quotes are stripped per token there. The supported
    Windows subset quotes WHOLE tokens ("--note=two words" or --note "two
    words") — mid-token quotes such as --note="two words" are not re-joined,
    and a mis-split command fails closed as a structured reviewer error."""
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
        deep = Path(tmp) / "deep.py"
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
        deep.write_text("print('[' * 200000)\n", encoding="utf-8")
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
        cases.append(("depth-bomb", run_reviewer([python, str(deep)], Path(tmp))))
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
        "depth-bomb": "FAIL",
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
