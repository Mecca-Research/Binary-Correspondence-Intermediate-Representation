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
import stat
from pathlib import Path
from typing import Any

try:
    from tools.security.proc_bounds import run_bounded
except ModuleNotFoundError:  # script execution: sys.path[0] is tools/security
    from proc_bounds import run_bounded

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: requires-python is >=3.10
    tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = Path(__file__).resolve().parent / "expected_inventory.json"
PYPROJECT_SIZE_CAP = 1 << 20  # 1 MiB: parsing multiplies metadata in memory
ADVISORY_TIMEOUT = 300.0  # pip-audit resolves against a network index; stalls expire
ADVISORY_OUTPUT_CAP = 1 << 20  # per stream; the engine reports advisories, not payload

_METADATA_TABLES = frozenset({"project", "build-system", "dependency-groups"})
_SECTION = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
# [[array-of-tables]] is a valid header the subset does not attribute; it
# opens an unattributable section rather than reading as a garbage line.
_ARRAY_TABLE = re.compile(r"^\[\[(?P<name>[^\]]+)\]\]\s*$")
# Sentinel: an array whose key this reader does not track. Its BODY must
# still be consumed to the closing bracket, or every continuation line
# reads as unrecognized garbage.
_SKIP_ARRAY = ("<skip>",)
# The key charset admits spaces/tabs (TOML permits whitespace around dotted
# separators; _split_key strips each segment) and backslashes — an escaped
# quoted key must REACH _split_key, which refuses escapes into _unasserted,
# instead of slipping past both this regex and the sensitivity net.
_ARRAY_KEY = re.compile(r"^(?P<key>[A-Za-z0-9_.\-\"'\\ \t]+)\s*=\s*(?P<rest>\[.*)$")
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
        if buffer[index:index + 3] in ('"""', "'''"):
            # TOML multiline strings are outside the subset; refuse rather
            # than misreading the delimiters as empty strings.
            return items, False
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
        elif char in "[], \t":
            index += 1
        else:
            # Non-string array content (an integer, a bool, a nested table)
            # is outside the subset; refuse rather than silently dropping it
            # from the declared inventory tomllib would have retained.
            return items, False
    return items, True


def _split_key(raw_key: str) -> tuple[str, ...] | None:
    """Quote-aware dotted-key split: a quoted segment keeps its dots and
    drops its quotes ('project."dependencies"' -> (project, dependencies)),
    matching tomllib's resolution. An unterminated quote or a backslash
    escape inside a key is outside the subset; None tells the caller to
    fail closed via ``_unasserted``."""
    segments: list[str] = []
    buffer: list[str] = []
    quote = ""
    for char in raw_key:
        if quote:
            if char == "\\":
                return None
            if char == quote:
                quote = ""
            else:
                buffer.append(char)
        elif char in "\"'":
            quote = char
        elif char == ".":
            segments.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
    if quote:
        return None
    segments.append("".join(buffer).strip())
    return tuple(segments)


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
    groups: set[str] = set()
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
            array_table = _ARRAY_TABLE.match(stripped)
            if array_table:
                split = _split_key(array_table.group("name").strip())
                if split is None or split[0] in _METADATA_TABLES:
                    # [[dependency-groups]] declares the metadata table as an
                    # ARRAY of tables; tomllib then exposes it as a non-table
                    # shape and fails closed. Treating it as merely
                    # unattributable let the group inside it disappear.
                    unasserted = True
                # Otherwise unattributable, not invalid: a dependency-shaped
                # key inside it still fails closed through the nets below.
                section = ("<unattributed>",)
                continue
            match = _SECTION.match(stripped)
            if match:
                split = _split_key(match.group("name").strip())
                if split is None:
                    unasserted = True
                    section = ("<unattributed>",)
                else:
                    if split in known:
                        # [project.dependencies] spells an ARRAY field as a
                        # table. tomllib projects it as a dict and fails the
                        # list-shape validation; the bracket spelling must
                        # not slip past that check here.
                        unasserted = True
                    section = split
                continue
            match = _ARRAY_KEY.match(stripped)
            if not match:
                if "=" in stripped:
                    if section[:1] == ("dependency-groups",):
                        # A group shape the subset reader cannot attribute
                        # (multi-line or non-array) fails closed.
                        unasserted = True
                    lhs, rhs = stripped.split("=", 1)
                    key = lhs.strip().strip("\"'")
                    if key.split(".", 1)[0].strip().strip("\"'") in _METADATA_TABLES:
                        # `project = "x"` / `build-system = 42` assigns a
                        # whole metadata table to a non-table. tomllib
                        # refuses it below; the subset reader must not read
                        # past it into an asserted empty inventory.
                        unasserted = True
                    if key == "dependency-groups" or key.startswith("dependency-groups."):
                        # dependency-groups = { qa = [...] } is the bracket
                        # table in inline spelling; the subset reader does
                        # not parse inline tables, so it refuses instead of
                        # asserting an inventory around the declaration.
                        unasserted = True
                    if section[:2] == optional_prefix:
                        # Every key under [project.optional-dependencies] IS
                        # an optional group. A non-array shape there (bad =
                        # {}) is one tomllib retains and this reader cannot,
                        # so the local-key net never saw it.
                        unasserted = True
                    qualified = ".".join(section + tuple(key.split(".")))
                    if _SENSITIVE.search(key) or _SENSITIVE.search(qualified):
                        # Section-qualified: a bare `dependencies = {}` under
                        # [project] is dependency-shaped only once its
                        # section is composed in.
                        unasserted = True
                    elif "{" in rhs and (
                        re.search(r"(?:optional-)?dependencies|dynamic", rhs)
                        # An escaped key inside the table never reaches
                        # _split_key, so its spelling cannot be compared:
                        # "dependencies" reads as dependencies to
                        # tomllib and as nothing here. Refuse every escaped
                        # inline table rather than scan the raw spelling.
                        or "\\" in rhs
                    ):
                        # An inline table can carry dependency metadata the
                        # subset reader does not parse; refuse rather than
                        # report an asserted empty inventory.
                        unasserted = True
                elif stripped:
                    # Neither a section, an array key, nor any assignment —
                    # and not blank or comment-only, both of which strip to
                    # empty. tomllib rejects such a line outright, so the
                    # fallback must not read past it and report an asserted
                    # inventory on a file newer hosts refuse.
                    unasserted = True
                continue
            raw_key = match.group("key").strip()
            split_key = _split_key(raw_key)
            if split_key is None:
                # A key shape the splitter cannot attribute must fail closed,
                # not vanish into an empty (matching) declared set.
                unasserted = True
                continue
            segments = split_key
            full = section + segments
            if len(full) == 1 and full[0] in _METADATA_TABLES:
                # An ARRAY assigned to a whole metadata table (project = [])
                # is not a table either, and this branch — not the scalar
                # one — is where that spelling lands. tomllib refuses it.
                unasserted = True
                continue
            if full[:1] == ("dependency-groups",):
                # PEP 735 groups are real declared dependencies; record them
                # so the audit refuses to assert an inventory around them.
                groups.add(full[1] if len(full) > 1 else "<unnamed>")
                continue
            is_optional = len(full) == 3 and full[:2] == optional_prefix
            if full not in known and not is_optional:
                if _SENSITIVE.search(".".join(full)) or full[:2] == optional_prefix:
                    unasserted = True
                if _bracket_depth(match.group("rest")) != 0:
                    # A multi-line array this reader does not track: consume
                    # to its closing bracket so its element lines are not
                    # mistaken for unrecognized garbage.
                    buffer = match.group("rest")
                    pending = _SKIP_ARRAY
                continue
            if full in arrays:
                # tomllib rejects a second assignment to the same key; the
                # unconditional overwrite here let a later `dependencies =
                # []` erase a real declaration and match the empty inventory.
                unasserted = True
            buffer = match.group("rest")
            pending = full
        else:
            buffer += " " + stripped
        if pending is _SKIP_ARRAY and _bracket_depth(buffer) == 0:
            # Untracked body consumed; nothing to record or validate.
            pending = None
            buffer = ""
            continue
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
        "dependency_groups": sorted(groups),
        "_unasserted": unasserted,
    }


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
    if tomllib is None:
        return _fallback_parse(text)
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
