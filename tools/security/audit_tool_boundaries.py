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
# Any-component skips: these names mean the same thing at every depth.
SKIP_PARTS = {".git", "__pycache__"}
# Generated-tree skips, scoped to the roots the build actually writes. As a
# component-wide predicate these excluded a tracked developer script under
# ANY nested directory so named (tools/build/release.py), and the
# missing-file reconciliation excluded it too, so the audit could report
# PASS around a script carrying os.system() or shell=True.
SKIP_ROOTS = (("build",), ("dataset",), ("bcir", "dataset"))
SOURCE_SIZE_CAP = 1 << 23  # 8 MiB: ast.parse multiplies source size in memory
# subprocess helpers that ARE shell string-command execution, like os.system.
SHELL_HELPERS = {"getoutput", "getstatusoutput"}


# Sentinel: discovery FAILED inside a git checkout — distinct from the
# intentional fixture walk (no .git at all), which returns None.
_DISCOVERY_FAILED = object()


def _tracked(root: Path) -> Any:
    if not (root / ".git").exists():
        # A fixture tree without git metadata: the plain walk IS the
        # contract there, not a degraded mode.
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return _DISCOVERY_FAILED
    if result.returncode != 0:
        # ls-files failing INSIDE a checkout (corrupt metadata, dubious
        # ownership, git absent) means the audit no longer knows what is
        # tracked; that must fail the audit, never downgrade it.
        return _DISCOVERY_FAILED
    # surrogateescape: a non-UTF-8 tracked filename must not kill discovery.
    return {
        item.decode("utf-8", "surrogateescape")
        for item in result.stdout.split(b"\0") if item
    }


def _iter_python(root: Path) -> tuple[list[Path], list[str], bool]:
    # The policy is about TRACKED developer code; an untracked scratch file
    # under .claude/ must not fail a local run that CI would pass. Outside a
    # git checkout (unit-test fixtures) the plain walk stands in.
    tracked = _tracked(root)
    if tracked is _DISCOVERY_FAILED:
        return [], [], True
    files: list[Path] = []
    seen: set[str] = set()
    for tree in TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            # Suffix matched case-insensitively: rglob("*.py") is literal on
            # Linux, so a tracked check.PY would dodge the audit there while
            # the case-folding hosts inspect it.
            if path.suffix.lower() != ".py":
                continue
            # Relative parts only: a checkout that itself lives under a
            # directory named "build" must not have every file skipped
            # (and the audit reported vacuous).
            rel_parts = path.relative_to(root).parts
            if any(part in SKIP_PARTS for part in rel_parts) or _is_generated(rel_parts):
                continue
            rel = str(path.relative_to(root)).replace("\\", "/")
            if tracked is not None and rel not in tracked:
                continue
            seen.add(rel)
            files.append(path)
    missing: list[str] = []
    if tracked is not None:
        # A tracked developer script the walk never yielded is absent from
        # the worktree (sparse checkout, unstaged deletion); reporting PASS
        # around it would claim more than was inspected.
        for rel in sorted(tracked):
            if not rel.lower().endswith(".py") or rel in seen:
                continue
            parts = rel.split("/")
            if (
                parts[0] not in TREES
                or any(part in SKIP_PARTS for part in parts)
                or _is_generated(tuple(parts))
            ):
                continue
            missing.append(rel)
    return files, missing, False


def _is_generated(parts: tuple[str, ...]) -> bool:
    """A generated tree is a PREFIX of the path, never a name anywhere in it."""
    return any(parts[:len(root)] == root for root in SKIP_ROOTS)


def _is_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def audit_boundaries(root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    symlinks: list[str] = []
    scanned = 0
    paths, missing, discovery_failed = _iter_python(root)
    if discovery_failed:
        return {
            "state": "FAIL",
            "scanned_files": 0,
            "findings": [
                {"path": ".", "line": 0, "rule": "tracked-discovery-failed"},
            ],
            "symlinks": [],
        }
    for rel in missing:
        printable = rel.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        findings.append({"path": printable, "line": 0, "rule": "file-missing"})
    for path in paths:
        rel = str(path.relative_to(root)).replace("\\", "/")
        if path.is_symlink():
            # The tracked blob is the target string, not Python source;
            # following it would audit arbitrary host content (or crash on
            # an unreadable procfs target). Record it, never dereference.
            symlinks.append(rel)
            continue
        scanned += 1
        try:
            if path.stat().st_size > SOURCE_SIZE_CAP:
                # Bounds at ingress: ast.parse on an accidentally tracked
                # blob must be a finding, never an OOM of the audit.
                findings.append({"path": rel, "line": 0, "rule": "file-oversized"})
                continue
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
                    # An f-string is a string command too: ast.JoinedStr,
                    # not ast.Constant, but a string at the call site.
                    if isinstance(command, ast.JoinedStr) or (
                        isinstance(command, ast.Constant)
                        and isinstance(command.value, str)
                    ):
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
