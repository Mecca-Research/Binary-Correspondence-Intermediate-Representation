#!/usr/bin/env python3
"""Static policy audit for lowering and developer-tool process/path/temp boundaries."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TREES = ("bcir", "tools")
SKIP_PARTS = {".git", "build", "__pycache__", "dataset"}
OS_FUNCS = frozenset({"system", "popen"})
SUBPROCESS_FUNCS = frozenset({"run", "Popen", "check_output", "check_call", "call"})


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


def _shell_not_provably_false(node: ast.AST, constants: dict[str, Any]) -> bool:
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.Name) and node.id in constants:
        return bool(constants[node.id])
    return True


def _resolve_bound_value(value: ast.AST, names: dict[str, str]) -> str | None:
    if isinstance(value, ast.Name):
        return names.get(value.id)
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        owner = names.get(value.value.id)
        if owner == "os" and value.attr in OS_FUNCS:
            return f"os.{value.attr}"
        if owner == "subprocess" and value.attr in SUBPROCESS_FUNCS:
            return f"subprocess.{value.attr}"
    return None


def _bindings(tree: ast.AST) -> tuple[dict[str, str], dict[str, Any]]:
    names: dict[str, str] = {"os": "os", "subprocess": "subprocess"}
    constants: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name == "os":
                    names[local] = "os"
                elif alias.name == "subprocess":
                    names[local] = "subprocess"
        elif isinstance(node, ast.ImportFrom):
            if node.module == "os":
                for alias in node.names:
                    if alias.name in OS_FUNCS:
                        names[alias.asname or alias.name] = f"os.{alias.name}"
            elif node.module == "subprocess":
                for alias in node.names:
                    if alias.name in SUBPROCESS_FUNCS:
                        names[alias.asname or alias.name] = f"subprocess.{alias.name}"
    assigns = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
    ]
    assigns.sort(key=lambda node: (node.lineno, node.col_offset))
    for node in assigns:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if isinstance(node.value, ast.Constant):
            constants[target] = node.value.value
        mapped = _resolve_bound_value(node.value, names)
        if mapped:
            names[target] = mapped
    return names, constants


def _call_kind(func: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(func, ast.Name):
        return bindings.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        owner = bindings.get(func.value.id)
        if owner == "os" and func.attr in OS_FUNCS:
            return f"os.{func.attr}"
        if owner == "subprocess" and func.attr in SUBPROCESS_FUNCS:
            return f"subprocess.{func.attr}"
    return None


def audit_boundaries(root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in _iter_python(root):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = str(path.relative_to(root)).replace("\\", "/")
        bindings, constants = _bindings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            kind = _call_kind(node.func, bindings)
            if kind is None:
                continue
            if kind.startswith("os."):
                findings.append({
                    "path": rel, "line": node.lineno, "rule": "os.system-or-popen",
                })
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            if "shell" in keywords and _shell_not_provably_false(keywords["shell"], constants):
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
