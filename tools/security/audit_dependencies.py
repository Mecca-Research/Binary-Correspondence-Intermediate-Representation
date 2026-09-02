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
import re
import shutil
import stat
from pathlib import Path
from typing import Any

try:
    from tools.security.git_index import STAGED_OVERSIZED, staged_blob, staged_divergent
    from tools.security.proc_bounds import run_bounded
except ModuleNotFoundError:  # script execution: sys.path[0] is tools/security
    from git_index import STAGED_OVERSIZED, staged_blob, staged_divergent
    from proc_bounds import run_bounded

import tomllib

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = Path(__file__).resolve().parent / "expected_inventory.json"
PYPROJECT_SIZE_CAP = 1 << 20  # 1 MiB: parsing multiplies metadata in memory
INVENTORY_SIZE_CAP = 1 << 20  # 1 MiB: the gate's own reference data is input too
ADVISORY_TIMEOUT = 300.0  # pip-audit resolves against a network index; stalls expire
ADVISORY_OUTPUT_CAP = 1 << 20  # per stream; the engine reports advisories, not payload

# A dependency specifier can CARRY a credential: PEP 508 direct references
# admit a full URL, and `pkg @ https://user:secret@host/x.whl` puts the
# secret in metadata this gate then copies into every report field. The
# scanner's rules do not recognize URL userinfo, so nothing else catches
# it — L7 is this rail's job here, not the scan's.
_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^/\s:@]+:)[^/\s@]+@")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:token|key|api[_-]?key|password|passwd|secret|access[_-]?token)=)"
    r"[^&\s\"']+"
)


def _redacted_requirement(text: str) -> str:
    """A dependency string with any embedded credential replaced.

    The declaration still names its package, host and path — everything a
    reader needs to act on the mismatch — with only the secret removed.
    """
    redacted = _URL_USERINFO.sub(r"\1<redacted>@", text)
    return _QUERY_SECRET.sub(r"\1<redacted>", redacted)


def _redacted(value: Any) -> Any:
    """``_redacted_requirement`` applied through the report's shapes."""
    if isinstance(value, str):
        return _redacted_requirement(value)
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    if isinstance(value, dict):
        return {key: _redacted(item) for key, item in value.items()}
    return value


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
    return parse_metadata(text)


def parse_metadata(text: str) -> dict[str, Any]:
    """The metadata contract, independent of where the bytes came from.

    The worktree file and the stage-0 blob are the same question asked of
    different bytes, so they share this validation rather than growing a
    second, subtly different copy.
    """
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


def _unreadable_inventory(reason: str) -> dict[str, Any]:
    return {
        "state": "FAIL",
        "inventory_asserted": False,
        "expected_packages": 0,
        "declared": {},
        "mismatches": [],
        "advisory": {"state": "UNAVAILABLE/SKIPPED", "engine": None},
        "error": f"expected inventory could not be read: {reason}",
    }


def _expected_inventory(expected_path: Path) -> dict[str, Any] | None:
    """The committed expected inventory, or None when it cannot be trusted.

    The inventory is INPUT to this gate exactly as pyproject.toml is: an
    absent, unreadable, non-UTF-8, malformed, or wrong-shaped file left the
    required audit raising before it built a verdict — no exit-code
    contract, no --json-out artifact. Shape is checked here too, because a
    JSON document that parses can still be the wrong document.
    """
    try:
        info = expected_path.lstat()
        if not stat.S_ISREG(info.st_mode):
            # Same refusal pyproject.toml gets: a stat size means nothing
            # for a symlink, FIFO or device, and following one to /dev/zero
            # reads without end.
            return None
        with expected_path.open("rb") as handle:
            # Bounds at ingress, as this rail already does for the metadata
            # it audits: `json.loads` allocates a multiple of its input, so
            # a padding-heavy inventory could exhaust the job before any
            # shape check ran. The declared size is not the read bound —
            # read one byte past the cap and refuse the remainder.
            blob = handle.read(INVENTORY_SIZE_CAP + 1)
        if len(blob) > INVENTORY_SIZE_CAP:
            return None
        raw = blob.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    # PRESENCE before shape: `_string_list(None)` and `_table(None)` both
    # normalize an ABSENT field to an empty one, which is the right reading
    # for optional metadata in pyproject.toml but the wrong one here — an
    # inventory that never mentions `runtime` has not asserted an empty
    # runtime, it has failed to say anything, and `audit()` then raised
    # KeyError reaching for it.
    missing = [
        field for field in ("runtime", "build_system", "optional")
        if field not in data
    ]
    if missing:
        return None
    if _string_list(data.get("runtime")) is None:
        return None
    if _string_list(data.get("build_system")) is None:
        return None
    optional = _table(data.get("optional"))
    if optional is None or any(
        _string_list(items) is None for items in optional.values()
    ):
        return None
    return data


def _staged_mismatches(root: Path, expected: dict[str, Any]) -> list[dict[str, Any]]:
    """Mismatches carried by the STAGED pyproject.toml, if it diverges.

    In a clean checkout — every CI run — nothing diverges and this costs two
    git invocations. Anything the index says that the inventory does not is
    a mismatch reported against the staged field, because that is what the
    next commit publishes.
    """
    if not (root / ".git").exists():
        return []
    divergent = staged_divergent(root)
    if divergent is None:
        return [{
            "field": "staged-discovery",
            "declared": "unavailable",
            "expected": "an answerable index/worktree comparison",
        }]
    if "pyproject.toml" not in divergent:
        return []
    blob = staged_blob(root, "pyproject.toml", cap=PYPROJECT_SIZE_CAP)
    if blob is None or blob is STAGED_OVERSIZED:
        return [{
            "field": "staged-pyproject",
            "declared": "unreadable" if blob is None else "oversized",
            "expected": "readable staged metadata within the ingress cap",
        }]
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return [{
            "field": "staged-pyproject",
            "declared": "not-utf-8",
            "expected": "decodable staged metadata",
        }]
    staged = parse_metadata(text)
    if staged.pop("_unasserted", False):
        return [{
            "field": "staged-pyproject",
            "declared": "unasserted",
            "expected": "fully attributable staged metadata",
        }]
    found: list[dict[str, Any]] = []
    if staged.pop("dependency_groups", []):
        found.append({
            "field": "staged:dependency-groups",
            "declared": staged.get("dependency_groups", []),
            "expected": [],
        })
    for field in ("runtime", "build_system", "optional"):
        if staged.get(field) != expected[field]:
            found.append({
                "field": f"staged:{field}",
                "declared": _redacted(staged.get(field)),
                "expected": _redacted(expected[field]),
            })
    dynamic = [
        item for item in staged.get("dynamic", [])
        if item in ("dependencies", "optional-dependencies")
    ]
    if dynamic:
        found.append({"field": "staged:dynamic", "declared": _redacted(dynamic), "expected": []})
    return found


def audit(root: Path, expected_path: Path = EXPECTED) -> dict[str, Any]:
    expected = _expected_inventory(expected_path)
    if expected is None:
        return _unreadable_inventory(str(expected_path))
    declared = parse_pyproject(root / "pyproject.toml")
    if declared.pop("_unasserted", False):
        # Either parser path saw metadata it could not fully read (a
        # dependency-shaped key the 3.10 fallback cannot attribute, or a file
        # tomllib rejects); passing a possibly-misread file asserts nothing.
        return {
            "state": "FAIL",
            "inventory_asserted": False,
            "expected_packages": 0,
            "declared": _redacted(declared),
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
            "declared": _redacted(declared),
            "mismatches": [
                {"field": "dependency-groups", "declared": _redacted(groups), "expected": []},
            ],
            "advisory": {"state": "UNAVAILABLE/SKIPPED", "engine": None},
            "error": (
                f"dependency-groups tables are not part of the asserted inventory: {_redacted(groups)}"
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
            "declared": _redacted(declared),
            "mismatches": [{"field": "dynamic", "declared": _redacted(dynamic), "expected": []}],
            "advisory": {"state": "UNAVAILABLE/SKIPPED", "engine": None},
            "error": f"dynamic dependency metadata cannot be asserted: {_redacted(dynamic)}",
        }
    mismatches = []
    for field in ("runtime", "build_system", "optional"):
        if declared[field] != expected[field]:
            mismatches.append({
                "field": field,
                "declared": _redacted(declared[field]),
                "expected": _redacted(expected[field]),
            })
    # The worktree file is not what the next commit records. A dependency
    # staged and then restored to benign worktree content shipped under a
    # PASS on this rail while the secret scan and the boundary audit had
    # already learned to reconcile against the index; all three now use the
    # one shared predicate.
    mismatches.extend(_staged_mismatches(root, expected))
    expected_count = (
        len(expected["runtime"])
        + len(expected["build_system"])
        + sum(len(items) for items in expected["optional"].values())
    )
    report: dict[str, Any] = {
        "state": "PASS",
        "inventory_asserted": True,
        "expected_packages": expected_count,
        "declared": _redacted(declared),
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
