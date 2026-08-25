#!/usr/bin/env python3
"""Static heuristic audit for obvious process-boundary literals.

This is not a sound Python interpreter. It flags ``os.system`` /
``os.popen`` and literal ``shell=True`` / string-command subprocess
calls. Aliases, control flow, ``**kwargs``, and nested scopes are out
of scope — those belong to a dedicated linter, not a BCIR rail.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TREES = ("bcir", "tools")
SKIP_PARTS = {".git", "build", "__pycache__", "dataset"}
PRODUCTION_STRING_SUBPROCESS = True


def _iter_python(root: Path) -> list[Path]:
    files = []
    for tree in TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            files.append(path)
    return files


def _is_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def audit_boundaries(root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in _iter_python(root):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = str(path.relative_to(root)).replace("\\", "/")
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
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        if "/tests/" not in f"/{rel}" and not rel.startswith("bcir/tests/"):
                            findings.append({
                                "path": rel, "line": node.lineno, "rule": "subprocess-string-command",
                            })
    report = {
        "state": "FAIL" if findings else "PASS",
        "scanned_files": scanned,
        "findings": findings,
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
