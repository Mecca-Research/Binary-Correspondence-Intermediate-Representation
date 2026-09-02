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

try:
    from tools.security.git_index import (
        STAGED_OVERSIZED, staged_blob, staged_divergent, staged_mode,
    )
except ModuleNotFoundError:  # script execution: sys.path[0] is tools/security
    from git_index import STAGED_OVERSIZED, staged_blob, staged_divergent, staged_mode

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


def _boundary_findings(tree: ast.AST, shown: str, rel: str) -> list[dict[str, Any]]:
    """Every prohibited process-boundary literal in one parsed source.

    One predicate, two callers: the worktree file and the stage-0 blob
    are the same audit question asked of different bytes.
    """
    findings: list[dict[str, Any]] = []
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
                        "path": shown, "line": node.lineno, "rule": "os.system-or-popen",
                    })
            if name in SHELL_HELPERS and isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                    findings.append({
                        "path": shown, "line": node.lineno,
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
                        "path": shown, "line": node.lineno, "rule": "subprocess-shell-true",
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
                            "path": shown, "line": node.lineno, "rule": "subprocess-string-command",
                        })
    return findings


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
        # rel keeps its surrogates for logic (tracked-set membership needs
        # the raw form); shown is what every finding and the symlink list
        # carry, because a surrogate printed under a strict stdout
        # (PYTHONIOENCODING=utf-8:strict) tracebacks the audit after its
        # summary — the same printable form the missing-file branch uses.
        shown = rel.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        if path.is_symlink():
            # The tracked blob is the target string, not Python source;
            # following it would audit arbitrary host content (or crash on
            # an unreadable procfs target). Record it, never dereference.
            symlinks.append(shown)
            continue
        scanned += 1
        try:
            if path.stat().st_size > SOURCE_SIZE_CAP:
                # Bounds at ingress: ast.parse on an accidentally tracked
                # blob must be a finding, never an OOM of the audit.
                findings.append({"path": shown, "line": 0, "rule": "file-oversized"})
                continue
            with path.open("rb") as handle:
                # The stat above is not the read bound: a file that grows
                # between the two (an editor or generator writing while the
                # local audit runs) would otherwise be materialized whole,
                # and ast.parse would multiply it in memory. Read one byte
                # past the cap and refuse the remainder — the same shape the
                # staged blob already uses.
                source = handle.read(SOURCE_SIZE_CAP + 1)
            if len(source) > SOURCE_SIZE_CAP:
                findings.append({"path": shown, "line": 0, "rule": "file-oversized"})
                continue
        except OSError:
            # A tracked file the auditor cannot read was not audited — a
            # failing finding, never an escaping traceback.
            findings.append({"path": shown, "line": 0, "rule": "file-unreadable"})
            continue
        try:
            # Bytes, not text: ast.parse honors a PEP 263 encoding cookie, so
            # a valid latin-1 source is audited rather than falsely rejected.
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, ValueError) as exc:
            # An unparseable file is uninspectable; fail closed with a finding
            # rather than escaping as a traceback.
            findings.append({
                "path": shown,
                "line": int(getattr(exc, "lineno", 0) or 0),
                "rule": "python-parse-error",
            })
            continue
        findings.extend(_boundary_findings(tree, shown, rel))
    # The walk above audited the WORKTREE; the next commit records the
    # INDEX. A tracked script staged with os.system() and then overwritten
    # with benign worktree bytes would otherwise pass this rail exactly as
    # it once passed the secret scan — same defect, same shared predicate.
    staged = staged_divergent(root) if (root / ".git").exists() else []
    if staged is None:
        findings.append({"path": ".", "line": 0, "rule": "staged-discovery-failed"})
    for rel in staged or ():
        parts = rel.split("/")
        if (
            not rel.lower().endswith(".py")
            or parts[0] not in TREES
            or any(part in SKIP_PARTS for part in parts)
            or _is_generated(tuple(parts))
        ):
            continue
        shown = rel.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        shown = f"{shown} (staged)"
        if staged_mode(root, rel) == "120000":
            # The index entry is a SYMLINK: its blob is the target string,
            # not Python. A clean checkout of the same index takes the
            # worktree symlink branch and records it without parsing, so
            # parsing it here made the local audit FAIL on a commit CI
            # passes. Record it the same way, never dereference it.
            symlinks.append(shown)
            continue
        blob = staged_blob(root, rel, cap=SOURCE_SIZE_CAP)
        if blob is None:
            # Divergent per git, but its index object cannot be read (or the
            # path is index-deleted): not audited, so not clean.
            findings.append({"path": shown, "line": 0, "rule": "staged-unreadable"})
            continue
        if blob is STAGED_OVERSIZED:
            findings.append({"path": shown, "line": 0, "rule": "file-oversized"})
            continue
        scanned += 1
        try:
            staged_tree = ast.parse(blob, filename=shown)
        except (SyntaxError, ValueError) as exc:
            findings.append({
                "path": shown,
                "line": int(getattr(exc, "lineno", 0) or 0),
                "rule": "python-parse-error",
            })
            continue
        findings.extend(_boundary_findings(staged_tree, shown, rel))
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
