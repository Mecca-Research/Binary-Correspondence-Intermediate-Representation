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
SUBPROCESS_FUNCS = frozenset({
    "run", "Popen", "check_output", "check_call", "call",
    "getoutput", "getstatusoutput",
})
SHELL_ALWAYS = frozenset({"subprocess.getoutput", "subprocess.getstatusoutput"})


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


class _BoundaryVisitor(ast.NodeVisitor):
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.findings: list[dict[str, Any]] = []
        self.scopes: list[tuple[dict[str, str], dict[str, Any]]] = [
            ({"os": "os", "subprocess": "subprocess"}, {}),
        ]

    def _names(self) -> dict[str, str]:
        return self.scopes[-1][0]

    def _constants(self) -> dict[str, Any]:
        return self.scopes[-1][1]

    def _push_scope(self) -> None:
        self.scopes.append((dict(self._names()), {}))

    def _apply_import(self, node: ast.AST) -> None:
        names = self._names()
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

    def _apply_assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        target = node.targets[0].id
        constants = self._constants()
        if isinstance(node.value, ast.Constant):
            constants[target] = node.value.value
        else:
            constants.pop(target, None)
        mapped = _resolve_bound_value(node.value, self._names())
        if mapped:
            self._names()[target] = mapped
        else:
            self._names().pop(target, None)

    def _shadow_params(self, node: ast.AST) -> None:
        args = getattr(node, "args", None)
        if args is None:
            return
        names = self._names()
        constants = self._constants()
        params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        for param in params:
            names.pop(param.arg, None)
            constants.pop(param.arg, None)
        if args.vararg is not None:
            names.pop(args.vararg.arg, None)
            constants.pop(args.vararg.arg, None)
        if args.kwarg is not None:
            names.pop(args.kwarg.arg, None)
            constants.pop(args.kwarg.arg, None)

    def _is_string_command(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return True
        if isinstance(node, ast.JoinedStr):
            return True
        if isinstance(node, ast.Name):
            return isinstance(self._constants().get(node.id), str)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._is_string_command(node.left) or self._is_string_command(node.right)
        return False

    def _record_call(self, node: ast.Call) -> None:
        kind = _call_kind(node.func, self._names())
        if kind is None:
            return
        if kind.startswith("os."):
            self.findings.append({
                "path": self.rel, "line": node.lineno, "rule": "os.system-or-popen",
            })
            return
        if kind in SHELL_ALWAYS:
            self.findings.append({
                "path": self.rel, "line": node.lineno, "rule": "subprocess-shell-true",
            })
            return
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        in_tests = "/tests/" in f"/{self.rel}" or self.rel.startswith("bcir/tests/")
        if any(kw.arg is None for kw in node.keywords):
            if not in_tests:
                self.findings.append({
                    "path": self.rel, "line": node.lineno, "rule": "subprocess-shell-true",
                })
        elif "shell" in keywords and _shell_not_provably_false(keywords["shell"], self._constants()):
            self.findings.append({
                "path": self.rel, "line": node.lineno, "rule": "subprocess-shell-true",
            })
        if node.args and self._is_string_command(node.args[0]):
            if not in_tests:
                self.findings.append({
                    "path": self.rel, "line": node.lineno, "rule": "subprocess-string-command",
                })

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push_scope()
        self._shadow_params(node)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push_scope()
        self._shadow_params(node)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Import(self, node: ast.Import) -> None:
        self._apply_import(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._apply_import(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        self._apply_assign(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.value is not None:
            fake = ast.Assign(targets=[node.target], value=node.value)
            self._apply_assign(fake)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        self._record_call(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self.scopes.pop()


def audit_boundaries(root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in _iter_python(root):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = str(path.relative_to(root)).replace("\\", "/")
        visitor = _BoundaryVisitor(rel)
        visitor.visit(tree)
        findings.extend(visitor.findings)
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
