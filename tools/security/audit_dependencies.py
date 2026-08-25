#!/usr/bin/env python3
"""Inventory-first dependency audit.

An empty-dependency report is INVALID unless the expected runtime inventory is
also empty and that emptiness was asserted. Advisory scanners run only after the
declared inventory matches the committed expected file.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: requires-python is >=3.10
    tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = Path(__file__).resolve().parent / "expected_inventory.json"

_SECTION = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
_ARRAY_KEY = re.compile(r"^(?P<key>[A-Za-z0-9_.\-\"']+)\s*=\s*(?P<rest>\[.*)$")
_STRING_ITEM = re.compile(r"\"([^\"]*)\"|'([^']*)'")
_SENSITIVE = re.compile(r"(?:^|\.)(?:optional-)?(?:dependencies|dynamic)$")


def _strip_toml_comment(line: str) -> str:
    """Cut an unquoted # comment; quote- and escape-aware so a # inside a
    string, or a quote inside a comment, cannot desynchronize the parser."""
    out: list[str] = []
    quote = ""
    escaped = False
    for char in line:
        if quote:
            out.append(char)
            if escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
            out.append(char)
        elif char == "#":
            break
        else:
            out.append(char)
    return "".join(out)


def _fallback_parse(text: str) -> dict[str, Any]:
    """Minimal TOML-subset reader for the audited fields on Python 3.10.

    It understands `key = [ "string", ... ]` arrays (single- or multi-line,
    plain or dotted keys) under bracketed sections, with comments stripped.
    Any dependency-shaped key it sees but cannot attribute sets
    ``_unasserted`` and the audit fails closed on this host rather than
    passing a file it may have misread; a parity test pins the reader against
    tomllib where tomllib exists.
    """
    arrays: dict[str, list[str]] = {}
    section = ""
    pending: str | None = None
    buffer = ""
    unasserted = False
    known = {
        "project.dependencies", "build-system.requires", "project.dynamic",
    }
    for raw_line in text.splitlines():
        stripped = _strip_toml_comment(raw_line).strip()
        if pending is None:
            match = _SECTION.match(stripped)
            if match:
                section = match.group("name").strip().strip("\"'")
                continue
            match = _ARRAY_KEY.match(stripped)
            if not match:
                if "=" in stripped and _SENSITIVE.search(
                    stripped.split("=", 1)[0].strip().strip("\"'")
                ):
                    unasserted = True
                continue
            key = match.group("key").strip().strip("\"'")
            full = f"{section}.{key}" if section else key
            if full not in known and not full.startswith("project.optional-dependencies."):
                if _SENSITIVE.search(full):
                    unasserted = True
                continue
            buffer = match.group("rest")
            pending = full
        else:
            buffer += " " + stripped
        if pending is not None and buffer.count("[") == buffer.count("]"):
            items = [a or b for a, b in _STRING_ITEM.findall(buffer)]
            arrays[pending] = items
            pending = None
            buffer = ""
    if pending is not None:
        unasserted = True  # an array never closed; the tail was not read
    optional = {
        full.rsplit(".", 1)[1]: items
        for full, items in arrays.items()
        if full.startswith("project.optional-dependencies.")
    }
    return {
        "runtime": arrays.get("project.dependencies", []),
        "build_system": arrays.get("build-system.requires", []),
        "optional": optional,
        "dynamic": arrays.get("project.dynamic", []),
        "_unasserted": unasserted,
    }


def parse_pyproject(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if tomllib is None:
        return _fallback_parse(text)
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


def audit(root: Path, expected_path: Path = EXPECTED) -> dict[str, Any]:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    declared = parse_pyproject(root / "pyproject.toml")
    if declared.pop("_unasserted", False):
        # The 3.10 fallback reader saw a dependency-shaped key it could not
        # attribute; passing a possibly-misread file would assert nothing.
        return {
            "state": "FAIL",
            "inventory_asserted": False,
            "expected_packages": 0,
            "declared": declared,
            "mismatches": [],
            "advisory": {"state": "UNAVAILABLE/SKIPPED", "engine": None},
            "error": "fallback parser could not fully read dependency metadata",
        }
    dynamic = [
        item for item in declared.get("dynamic", [])
        if item in ("dependencies", "optional-dependencies")
    ]
    if dynamic:
        # Dynamic metadata resolves at build time from sources this audit does
        # not read; treating it as an empty declared set would assert nothing.
        return {
            "state": "FAIL",
            "inventory_asserted": False,
            "expected_packages": 0,
            "declared": declared,
            "mismatches": [{"field": "dynamic", "declared": dynamic, "expected": []}],
            "advisory": {"state": "UNAVAILABLE/SKIPPED", "engine": None},
            "error": f"dynamic dependency metadata cannot be asserted: {dynamic}",
        }
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
