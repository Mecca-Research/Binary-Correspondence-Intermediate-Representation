#!/usr/bin/env python3
"""Inventory-first dependency audit (Python 3.11+, tomllib only).

An empty-dependency report is INVALID unless the expected runtime inventory is
also empty and that emptiness was asserted. Advisory scanners run only after the
declared inventory matches the committed expected file. The metadata parser is
the standard library's tomllib, unconditionally: the repository's floor is
3.11, and no hand-rolled TOML subset stands in anywhere — a subset reader has
an unbounded surface of valid spellings and can only lose to them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import stat
from pathlib import Path
from typing import Any

try:
    from tools.security.proc_bounds import run_bounded
except ModuleNotFoundError:  # script execution: sys.path[0] is tools/security
    from proc_bounds import run_bounded

import tomllib

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = Path(__file__).resolve().parent / "expected_inventory.json"
PYPROJECT_SIZE_CAP = 1 << 20  # 1 MiB: parsing multiplies metadata in memory
ADVISORY_TIMEOUT = 300.0  # pip-audit resolves against a network index; stalls expire
ADVISORY_OUTPUT_CAP = 1 << 20  # per stream; the engine reports advisories, not payload

def _table(value: Any) -> dict[str, Any] | None:
    """A metadata table is a table or it is unreadable. None means refuse:
    an absent table is an empty one, but a scalar or list in its place must
    not be coerced away by ``or {}`` (project = [] would read as no
    dependencies) nor dereferenced (project = "x" makes .get raise)."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        return None
    return value


def _string_list(value: Any) -> list[str] | None:
    """A dependency field is a list of strings or it is unreadable. None
    means refuse: bool is excluded because it is an int, and a bare string
    is refused rather than shredded into characters by list()."""
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return list(value)


_UNASSERTED: dict[str, Any] = {
    "runtime": [], "build_system": [], "optional": {}, "dynamic": [],
    "_unasserted": True,
}


def parse_pyproject(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError:
        # An absent or unreadable metadata file asserts nothing; it must not
        # escape as a traceback from the required audit either.
        return dict(_UNASSERTED)
    if not stat.S_ISREG(info.st_mode):
        # lstat, not stat: a SYMLINK to /dev/zero reports size 0 through a
        # following stat and then reads without end. The secret scan records
        # links without following them; this rail refuses them outright,
        # along with every other non-regular file (FIFO, device), because a
        # stat size means nothing for those.
        return dict(_UNASSERTED)
    if info.st_size > PYPROJECT_SIZE_CAP:
        # Bounds at ingress: tomllib and the fallback both build structures
        # a multiple of the file's size, so an oversized metadata file is
        # unasserted (a fail-closed verdict), never an OOM of the audit.
        return dict(_UNASSERTED)
    try:
        with path.open("rb") as handle:
            # The declared size is not the read bound: read one byte past
            # the cap and refuse if anything remains, so a file that grows
            # (or lies) between stat and read cannot slip through.
            raw = handle.read(PYPROJECT_SIZE_CAP + 1)
        if len(raw) > PYPROJECT_SIZE_CAP:
            return dict(_UNASSERTED)
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return dict(_UNASSERTED)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        # The 3.10 fallback reports a malformed file as a structured FAIL via
        # _unasserted; the tomllib path must not diverge into a traceback.
        return dict(_UNASSERTED)
    project = _table(data.get("project"))
    build = _table(data.get("build-system"))
    groups = _table(data.get("dependency-groups"))
    if project is None or build is None or groups is None:
        # The enclosing tables are validated BEFORE their fields: a scalar
        # in place of a table raises on .get, and an empty list would be
        # coerced to an empty table and pass as "no dependencies".
        return {
            "runtime": [], "build_system": [], "optional": {}, "dynamic": [],
            "_unasserted": True,
        }
    optional = _table(project.get("optional-dependencies"))
    runtime = _string_list(project.get("dependencies"))
    requires = _string_list(build.get("requires"))
    dynamic = _string_list(project.get("dynamic"))
    extras: dict[str, list[str]] | None = None
    if optional is not None:
        extras = {}
        for name, items in optional.items():
            values = _string_list(items)
            if values is None:
                extras = None
                break
            extras[name] = values
    if runtime is None or requires is None or dynamic is None or extras is None:
        # Syntactically valid TOML can still carry a nonsense metadata shape
        # (dependencies = 42, or a bare string that list() would silently
        # shred into characters). The 3.10 fallback already refuses these;
        # the tomllib path must fail closed the same way, never traceback.
        return {
            "runtime": [], "build_system": [], "optional": {}, "dynamic": [],
            "_unasserted": True,
        }
    return {
        "runtime": runtime,
        "build_system": requires,
        "optional": extras,
        "dynamic": dynamic,
        "dependency_groups": sorted(groups),
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
    groups = declared.pop("dependency_groups", [])
    if groups:
        # PEP 735 dependency groups are real declared dependencies that the
        # expected-inventory schema cannot express; asserting an inventory
        # around them would let a new group ride in under PASS.
        return {
            "state": "FAIL",
            "inventory_asserted": False,
            "expected_packages": 0,
            "declared": declared,
            "mismatches": [
                {"field": "dependency-groups", "declared": groups, "expected": []},
            ],
            "advisory": {"state": "UNAVAILABLE/SKIPPED", "engine": None},
            "error": (
                f"dependency-groups tables are not part of the asserted inventory: {groups}"
            ),
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
    # The shared bounded runner: its own session, a wall bound, per-stream
    # byte budgets, and a process-group put-down — the resolver does network
    # work, spawns pip children, and can flood; every failure shape becomes
    # the advisory's fail-closed state, never a hang, OOM, or traceback.
    outcome = run_bounded(
        [engine, "--requirement", "-"],
        timeout=ADVISORY_TIMEOUT,
        cap=ADVISORY_OUTPUT_CAP,
        stdin_data=("\n".join(listed) + "\n").encode("utf-8"),
    )
    failure = ""
    if not outcome["launched"]:
        failure = outcome["error"]
    elif outcome["timed_out"]:
        failure = f"pip-audit timed out after {ADVISORY_TIMEOUT:g}s"
    elif outcome["overflow"]:
        failure = f"pip-audit output exceeded {ADVISORY_OUTPUT_CAP} bytes per stream"
    elif outcome["pipes_held"]:
        failure = "descendant processes still hold pip-audit's pipes"
    if failure:
        report["advisory"] = {
            "state": "FAIL",
            "engine": "pip-audit",
            "error": failure,
        }
        report["state"] = "FAIL"
        return report
    report["advisory"] = {
        "state": "PASS" if outcome["returncode"] == 0 else "FAIL",
        "engine": "pip-audit",
        "returncode": outcome["returncode"],
        "stdout_tail": outcome["stdout"].decode("utf-8", "replace")[-500:],
    }
    if outcome["returncode"] != 0:
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
