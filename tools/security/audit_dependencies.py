#!/usr/bin/env python3
"""Inventory-first dependency audit.

An empty-dependency report is INVALID unless the expected runtime inventory is
also empty and that emptiness was asserted. Advisory scanners run only after the
declared inventory matches the committed expected file.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = Path(__file__).resolve().parent / "expected_inventory.json"


def parse_pyproject(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project") or {}
    optional = project.get("optional-dependencies") or {}
    build = data.get("build-system") or {}
    return {
        "runtime": list(project.get("dependencies") or []),
        "build_system": list(build.get("requires") or []),
        "optional": {name: list(items) for name, items in optional.items()},
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
