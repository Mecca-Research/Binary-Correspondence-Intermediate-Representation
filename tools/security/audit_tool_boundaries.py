#!/usr/bin/env python3
"""Static heuristic audit for obvious process-boundary literals.

This is not a sound Python interpreter. It flags ``os.system`` /
``os.popen``, the ``subprocess.getoutput``/``getstatusoutput`` shell
helpers, and literal ``shell=True`` / string-command subprocess calls.
Aliases, control flow, ``**kwargs``, and nested scopes are out of
scope — those belong to a dedicated linter, not a BCIR rail.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# ".claude" carries tracked developer scripts (digest/skill tooling); a
# developer-tool boundary policy that skips them audits less than it claims.
TREES = ("bcir", "tools", ".claude")
SKIP_PARTS = {".git", "build", "__pycache__", "dataset"}
# subprocess helpers that ARE shell string-command execution, like os.system.
SHELL_HELPERS = {"getoutput", "getstatusoutput"}


def _tracked(root: Path) -> set[str] | None:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return {item.decode("utf-8") for item in result.stdout.split(b"\0") if item}


def _iter_python(root: Path) -> list[Path]:
    # The policy is about TRACKED developer code; an untracked scratch file
    # under .claude/ must not fail a local run that CI would pass. Outside a
    # git checkout (unit-test fixtures) the plain walk stands in.
    tracked = _tracked(root)
    files = []
    for tree in TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            # Relative parts only: a checkout that itself lives under a
            # directory named "build" must not have every file skipped
            # (and the audit reported vacuous).
            if any(part in SKIP_PARTS for part in path.relative_to(root).parts):
                continue
            if tracked is not None:
                rel = str(path.relative_to(root)).replace("\\", "/")
                if rel not in tracked:
                    continue
            files.append(path)
    return files


def _is_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def audit_boundaries(root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    symlinks: list[str] = []
    scanned = 0
    for path in _iter_python(root):
        rel = str(path.relative_to(root)).replace("\\", "/")
        if path.is_symlink():
            # The tracked blob is the target string, not Python source;
            # following it would audit arbitrary host content (or crash on
            # an unreadable procfs target). Record it, never dereference.
            symlinks.append(rel)
            continue
        scanned += 1
        try:
            source = path.read_bytes()
        except OSError:
            # A tracked file the auditor cannot read was not audited — a
            # failing finding, never an escaping traceback.
            findings.append({"path": rel, "line": 0, "rule": "file-unreadable"})
            continue
        try:
            # Bytes, not text: ast.parse honors a PEP 263 encoding cookie, so
            # a valid latin-1 source is audited rather than falsely rejected.
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, ValueError) as exc:
            # An unparseable file is uninspectable; fail closed with a finding
            # rather than escaping as a traceback.
            findings.append({
                "path": rel,
                "line": int(getattr(exc, "lineno", 0) or 0),
                "rule": "python-parse-error",
            })
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in {"system", "popen"} and isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name) and func.value.id == "os":
                        findings.append({
                            "path": rel, "line": node.lineno, "rule": "os.system-or-popen",
                        })
                if name in SHELL_HELPERS and isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                        findings.append({
                            "path": rel, "line": node.lineno,
                            "rule": "subprocess-shell-helper",
                        })
                if name in {"run", "Popen", "check_output", "check_call", "call"}:
                    if not (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "subprocess"
                    ):
                        continue
                    keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                    if "shell" in keywords and _is_true(keywords["shell"]):
                        findings.append({
                            "path": rel, "line": node.lineno, "rule": "subprocess-shell-true",
                        })
                    # The command may be the first positional OR the literal
                    # `args=` keyword — subprocess's public parameter name.
                    command = node.args[0] if node.args else keywords.get("args")
                    if isinstance(command, ast.Constant) and isinstance(command.value, str):
                        if "/tests/" not in f"/{rel}" and not rel.startswith("bcir/tests/"):
                            findings.append({
                                "path": rel, "line": node.lineno, "rule": "subprocess-string-command",
                            })
    report = {
        "state": "FAIL" if findings else "PASS",
        "scanned_files": scanned,
        "findings": findings,
        "symlinks": symlinks,
    }
    if scanned == 0:
        report["state"] = "INVALID/VACUOUS"
        report["error"] = "no python files scanned"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit_tool_boundaries")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = audit_boundaries(args.root)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"audit_tool_boundaries: {report['state']} scanned={report['scanned_files']} "
        f"findings={len(report['findings'])}"
    )
    for finding in report["findings"][:20]:
        print(f"  {finding['path']}:{finding['line']} {finding['rule']}")
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
