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
_SENSITIVE = re.compile(r"(?:^|\.)(?:optional-)?(?:dependencies|dynamic)$")
_ESCAPES = {"\\": "\\", '"': '"', "b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r"}


def _string_items(buffer: str) -> tuple[list[str], bool]:
    """Escape-aware TOML string extraction; returns (items, ok).

    Basic strings decode the short escapes (an environment marker such as
    'foo; python_version < \\"3.12\\"' round-trips); literal strings take no
    escapes. Anything outside that subset (\\uXXXX, an unterminated string)
    reports ok=False so the caller fails closed instead of misreading."""
    items: list[str] = []
    index = 0
    length = len(buffer)
    while index < length:
        char = buffer[index]
        if char == "'":
            end = buffer.find("'", index + 1)
            if end < 0:
                return items, False
            items.append(buffer[index + 1:end])
            index = end + 1
        elif char == '"':
            out: list[str] = []
            index += 1
            closed = False
            while index < length:
                current = buffer[index]
                if current == "\\":
                    if index + 1 >= length or buffer[index + 1] not in _ESCAPES:
                        return items, False
                    out.append(_ESCAPES[buffer[index + 1]])
                    index += 2
                elif current == '"':
                    closed = True
                    index += 1
                    break
                else:
                    out.append(current)
                    index += 1
            if not closed:
                return items, False
            items.append("".join(out))
        else:
            index += 1
    return items, True


def _bracket_depth(buffer: str) -> int:
    """Net [ ] depth OUTSIDE quoted strings (quote- and escape-aware), so a
    bracket inside a dependency string cannot hold an array open and fail a
    file tomllib accepts."""
    depth = 0
    quote = ""
    escaped = False
    for char in buffer:
        if quote:
            if escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
    return depth


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
    arrays: dict[tuple[str, ...], list[str]] = {}
    section: tuple[str, ...] = ()
    pending: tuple[str, ...] | None = None
    buffer = ""
    unasserted = False
    known = {
        ("project", "dependencies"),
        ("build-system", "requires"),
        ("project", "dynamic"),
    }
    optional_prefix = ("project", "optional-dependencies")
    for raw_line in text.splitlines():
        stripped = _strip_toml_comment(raw_line).strip()
        if pending is None:
            match = _SECTION.match(stripped)
            if match:
                section = tuple(
                    part.strip().strip("\"'")
                    for part in match.group("name").strip().split(".")
                )
                continue
            match = _ARRAY_KEY.match(stripped)
            if not match:
                if "=" in stripped:
                    lhs, rhs = stripped.split("=", 1)
                    if _SENSITIVE.search(lhs.strip().strip("\"'")):
                        unasserted = True
                    elif "{" in rhs and re.search(
                        r"(?:optional-)?dependencies|dynamic", rhs
                    ):
                        # An inline table can carry dependency metadata the
                        # subset reader does not parse; refuse rather than
                        # report an asserted empty inventory.
                        unasserted = True
                continue
            raw_key = match.group("key").strip()
            if raw_key[:1] in "\"'":
                # A quoted key is one literal segment; its dots are not paths.
                segments: tuple[str, ...] = (raw_key.strip("\"'"),)
            else:
                segments = tuple(part.strip() for part in raw_key.split("."))
            full = section + segments
            is_optional = len(full) == 3 and full[:2] == optional_prefix
            if full not in known and not is_optional:
                if _SENSITIVE.search(".".join(full)) or full[:2] == optional_prefix:
                    unasserted = True
                continue
            buffer = match.group("rest")
            pending = full
        else:
            buffer += " " + stripped
        if pending is not None and _bracket_depth(buffer) == 0:
            items, parsed_ok = _string_items(buffer)
            if not parsed_ok:
                unasserted = True
            arrays[pending] = items
            pending = None
            buffer = ""
    if pending is not None:
        unasserted = True  # an array never closed; the tail was not read
    optional = {
        full[2]: items
        for full, items in arrays.items()
        if len(full) == 3 and full[:2] == optional_prefix
    }
    return {
        "runtime": arrays.get(("project", "dependencies"), []),
        "build_system": arrays.get(("build-system", "requires"), []),
        "optional": optional,
        "dynamic": arrays.get(("project", "dynamic"), []),
        "_unasserted": unasserted,
    }


def parse_pyproject(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if tomllib is None:
        return _fallback_parse(text)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        # The 3.10 fallback reports a malformed file as a structured FAIL via
        # _unasserted; the tomllib path must not diverge into a traceback.
        return {
            "runtime": [], "build_system": [], "optional": {}, "dynamic": [],
            "_unasserted": True,
        }
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
        # Either parser path saw metadata it could not fully read (a
        # dependency-shaped key the 3.10 fallback cannot attribute, or a file
        # tomllib rejects); passing a possibly-misread file asserts nothing.
        return {
            "state": "FAIL",
            "inventory_asserted": False,
            "expected_packages": 0,
            "declared": declared,
            "mismatches": [],
            "advisory": {"state": "UNAVAILABLE/SKIPPED", "engine": None},
            "error": "dependency metadata could not be fully read",
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
