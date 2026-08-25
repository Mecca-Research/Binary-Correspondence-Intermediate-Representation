#!/usr/bin/env python3
"""Inventory-first dependency audit.

An empty-dependency report is INVALID unless the expected runtime inventory is
also empty and that emptiness was asserted. Advisory scanners run only after the
declared inventory matches the committed expected file.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = Path(__file__).resolve().parent / "expected_inventory.json"


def parse_pyproject(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(text)
        project = data.get("project") or {}
        optional = project.get("optional-dependencies") or {}
        build = data.get("build-system") or {}
        return {
            "runtime": list(project.get("dependencies") or []),
            "build_system": list(build.get("requires") or []),
            "optional": {name: list(items) for name, items in optional.items()},
            "dynamic": list(project.get("dynamic") or []),
        }
    return _parse_pyproject_legacy(text)


def _extract_bracket_list(text: str, start: int) -> str:
    if start >= len(text) or text[start] != "[":
        raise ValueError("expected '['")
    depth = 0
    for index, char in enumerate(text[start:], start):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("unterminated list")


def _parse_pyproject_legacy(text: str) -> dict[str, Any]:
    """Subset parser for Python 3.10, where tomllib is absent."""
    runtime_at = re.search(r"(?m)^dependencies\s*=\s*", text)
    build_at = re.search(r"(?m)^requires\s*=\s*", text)
    if runtime_at is None or build_at is None:
        raise ValueError("could not parse pyproject.toml inventories without tomllib")
    runtime = ast.literal_eval(_extract_bracket_list(text, runtime_at.end()))
    build = ast.literal_eval(_extract_bracket_list(text, build_at.end()))
    optional: dict[str, list[str]] = {}
    extras = re.search(
        r"(?ms)^\[project\.optional-dependencies\]\s*\n(.*?)(?=^\[|\Z)", text,
    )
    if extras:
        block = extras.group(1)
        for match in re.finditer(r"(?m)^([A-Za-z0-9_-]+)\s*=\s*", block):
            optional[match.group(1)] = list(
                ast.literal_eval(_extract_bracket_list(block, match.end()))
            )
    dynamic: list[str] = []
    dyn = re.search(r"(?m)^dynamic\s*=\s*", text)
    if dyn:
        dynamic = list(ast.literal_eval(_extract_bracket_list(text, dyn.end())))
    return {
        "runtime": list(runtime),
        "build_system": list(build),
        "optional": optional,
        "dynamic": dynamic,
    }


def audit(root: Path, expected_path: Path = EXPECTED) -> dict[str, Any]:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    declared = parse_pyproject(root / "pyproject.toml")
    mismatches = []
    for field in ("runtime", "build_system", "optional"):
        if declared[field] != expected[field]:
            mismatches.append({
                "field": field,
                "declared": declared[field],
                "expected": expected[field],
            })
    expected_count = (
        len(expected["runtime"])
        + len(expected["build_system"])
        + sum(len(items) for items in expected["optional"].values())
    )
    report: dict[str, Any] = {
        "state": "PASS",
        "inventory_asserted": True,
        "expected_packages": expected_count,
        "declared": declared,
        "mismatches": mismatches,
        "advisory": {"state": "UNAVAILABLE/SKIPPED", "engine": None},
    }
    if mismatches:
        report["state"] = "FAIL"
        return report
    unresolved = [
        name for name in declared.get("dynamic") or []
        if name in {"dependencies", "optional-dependencies"}
    ]
    if unresolved:
        report["state"] = "FAIL"
        report["error"] = f"unresolved dynamic dependency metadata: {unresolved}"
        return report
    engine = shutil.which("pip-audit")
    if not engine:
        return report
    listed = list(expected["build_system"])
    for items in expected["optional"].values():
        listed.extend(items)
    listed.extend(expected["runtime"])
    if not listed:
        report["advisory"] = {
            "state": "PASS",
            "engine": "pip-audit",
            "note": "no install set remains after asserting the empty runtime inventory",
        }
        return report
    result = subprocess.run(
        [engine, "--requirement", "-"],
        input="\n".join(listed) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    report["advisory"] = {
        "state": "PASS" if result.returncode == 0 else "FAIL",
        "engine": "pip-audit",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-500:],
    }
    if result.returncode != 0:
        report["state"] = "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit_dependencies")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expected", type=Path, default=EXPECTED)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.root, args.expected)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"audit_dependencies: {report['state']} asserted={report['inventory_asserted']} "
        f"expected={report['expected_packages']} mismatches={len(report['mismatches'])} "
        f"advisory={report['advisory']['state']}"
    )
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
