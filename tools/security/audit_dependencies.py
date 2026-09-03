#!/usr/bin/env python3
"""Inventory-first dependency audit (Python 3.11+, tomllib only).

An empty-dependency report is INVALID unless the expected runtime inventory is
also empty and that emptiness was asserted. The advisory rail runs only after the
declared inventory matches the committed expected file: pip-audit resolves the
asserted install set (pip's resolver, then PyPI's advisory database) and its
JSON report is parsed strictly into vulnerability IDs, aliases and fix versions
-- never its prose. The rail is opportunistic by default (no engine on the host:
``UNAVAILABLE/SKIPPED``, the inventory verdict stands) and REQUIRED under
``--require-advisory``, which the one CI job that installs the engine passes:
there an absent engine is a FAIL. In either mode an engine that ran gates: a
finding, a dependency it could not collect, a report it did not write in its own
shape, or an audit that collected nothing from a non-empty install set is a FAIL,
never a quieter kind of pass. The metadata parser is the standard library's
tomllib, unconditionally: the repository's floor is 3.11, and no hand-rolled TOML
subset stands in anywhere — a subset reader has an unbounded surface of valid
spellings and can only lose to them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

try:
    from tools.security.git_index import (
        STAGED_OVERSIZED, staged_blob, staged_divergent, staged_mode,
    )
    from tools.security.report_hygiene import DuplicateKeys, mapped, strict_loads
    from tools.security.proc_bounds import run_bounded
except ModuleNotFoundError:  # script execution: sys.path[0] is tools/security
    from git_index import (
        STAGED_OVERSIZED, staged_blob, staged_divergent, staged_mode,
    )
    from report_hygiene import DuplicateKeys, mapped, strict_loads
    from proc_bounds import run_bounded

import tomllib

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = Path(__file__).resolve().parent / "expected_inventory.json"
PYPROJECT_SIZE_CAP = 1 << 20  # 1 MiB: parsing multiplies metadata in memory
INVENTORY_SIZE_CAP = 1 << 20  # 1 MiB: the gate's own reference data is input too
ADVISORY_TIMEOUT = 300.0  # pip-audit resolves against a network index; stalls expire
ADVISORY_OUTPUT_CAP = 1 << 20  # per stream; the engine reports advisories, not payload
ADVISORY_ENGINE = "pip-audit"
# The engine is configured the way CLANG and BCIR_OPT configure theirs: a path
# or a command NAME, resolved through the same lookup the default takes (L13).
ADVISORY_ENGINE_ENV = "PIP_AUDIT"
ADVISORY_TAIL = 500  # bytes of engine stderr/stdout retained in the report, redacted

# A dependency specifier can CARRY a credential: PEP 508 direct references
# admit a full URL, and `pkg @ https://user:secret@host/x.whl` puts the
# secret in metadata this gate then copies into every report field. The
# scanner's rules do not recognize URL userinfo, so nothing else catches
# it — L7 is this rail's job here, not the scan's.
# ALL of userinfo, not just the password half. `user:secret@` was the
# shape this caught first, but a token is just as often the whole
# username (`ghp_...@host`, `x-access-token:...@host` inverted), and a
# username-only URL matched nothing while a username beside a password
# was preserved verbatim. Userinfo is credential material by position,
# so position is what this redacts; host and path still name the
# declaration for the reader.
_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^/?#\s@]+@")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:token|key|api[_-]?key|password|passwd|secret|access[_-]?token)=)"
    r"[^&\s\"']+"
)


def _redacted_requirement(text: str) -> str:
    """A dependency string with any embedded credential replaced.

    The declaration still names its package, host and path — everything a
    reader needs to act on the mismatch — with the credential removed.
    """
    redacted = _URL_USERINFO.sub(r"\1<redacted>@", text)
    return _QUERY_SECRET.sub(r"\1<redacted>", redacted)


def _redacted(value: Any) -> Any:
    """``_redacted_requirement`` applied through the report's shapes."""
    return mapped(value, _redacted_requirement)


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
    return _inventory_from_text(raw)


def _inventory_from_text(raw: str) -> dict[str, Any] | None:
    try:
        # STRICT: json.loads keeps the LAST value for a repeated key, so an
        # inventory declaring `"runtime": ["hidden-package==1"]` and later
        # `"runtime": []` parses clean and audits clean while saying two
        # different things. A gate cannot attribute a contradictory
        # declaration, so it refuses it (L4) -- the review parser has
        # refused exactly this since R23, and now they share the predicate.
        data = strict_loads(raw)
    except (json.JSONDecodeError, DuplicateKeys, RecursionError):
        # RecursionError, not just malformed JSON: every token can be valid
        # and the payload still bottom out the parser -- 20k nested arrays
        # is ~40 KiB, well inside the ingress cap, so a size bound is no
        # defence. It escaped before any report existed, taking --json-out
        # with it. Unreadable is a VERDICT this gate already knows how to
        # report (L1), and the review parser has caught this since R30 --
        # one predicate, both parsers (L14).
        return None
    return _inventory_shape(data)


def _inventory_shape(data: Any) -> dict[str, Any] | None:
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


def _staged_inventory_mismatches(
    root: Path, expected_path: Path, declared: dict[str, Any]
) -> list[dict[str, Any]]:
    """Mismatches carried by the STAGED expected inventory, if it diverges.

    The audit asserts the declared metadata against the inventory, so BOTH
    sides of that comparison are index-sensitive. Reconciling only
    pyproject.toml left the mirror-image hole: stage an inventory change,
    restore the benign worktree copy, and the audit passes while the commit
    it produces fails the same audit in CI.
    """
    try:
        rel = expected_path.resolve().relative_to(root.resolve())
    except ValueError:
        # An inventory outside the checkout is not something the index can
        # carry, so there is nothing to reconcile.
        return []
    divergent = staged_divergent(root)
    if divergent is None:
        return [{
            "field": "staged-discovery",
            "declared": "unavailable",
            "expected": "an answerable index/worktree comparison",
        }]
    key = str(rel).replace("\\", "/")
    if key not in divergent:
        return []
    if staged_mode(root, key) == "120000":
        # The index entry is a SYMLINK: its blob is a TARGET PATH, not
        # inventory JSON. Parsing it audits whatever that path names on
        # this host -- and a target whose text happens to be matching JSON
        # would pass here while a clean checkout of the same commit takes
        # the non-regular-file branch and fails. Never dereference an
        # index symlink (L12); the scan and boundary rails already refuse.
        return [{
            "field": "staged-inventory",
            "declared": "symlink",
            "expected": "a regular-file staged inventory",
        }]
    blob = staged_blob(root, key, cap=INVENTORY_SIZE_CAP)
    if blob is None or blob is STAGED_OVERSIZED:
        return [{
            "field": "staged-inventory",
            "declared": "unreadable" if blob is None else "oversized",
            "expected": "readable staged inventory within the ingress cap",
        }]
    try:
        # STRICT, exactly as the worktree read and the staged pyproject
        # read already are. A replacement decode turns an invalid byte into
        # U+FFFD, so a staged inventory that CI refuses to decode could
        # match a worktree copy carrying the replacement character and pass
        # locally -- the gate disagreeing with itself across two paths (L12).
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return [{
            "field": "staged-inventory",
            "declared": "not-utf-8",
            "expected": "decodable staged inventory",
        }]
    staged = _inventory_from_text(text)
    if staged is None:
        return [{
            "field": "staged-inventory",
            "declared": "unusable",
            "expected": "a well-shaped staged inventory",
        }]
    found: list[dict[str, Any]] = []
    for field in ("runtime", "build_system", "optional"):
        if declared.get(field) != staged[field]:
            found.append({
                "field": f"staged-inventory:{field}",
                "declared": _redacted(declared.get(field)),
                "expected": _redacted(staged[field]),
            })
    return found


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
    if staged_mode(root, "pyproject.toml") == "120000":
        # Same refusal as the staged inventory above: a symlink's blob is
        # its target path, and parse_pyproject() rejects a non-regular file
        # in the worktree, so parsing the target here made the local audit
        # PASS on a commit whose clean checkout FAILS (L12).
        return [{
            "field": "staged-pyproject",
            "declared": "symlink",
            "expected": "regular-file staged metadata",
        }]
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


def audit(
    root: Path, expected_path: Path = EXPECTED, *, require_advisory: bool = False
) -> dict[str, Any]:
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
    if (root / ".git").exists():
        mismatches.extend(
            _staged_inventory_mismatches(root, expected_path, declared)
        )
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
    engine = advisory_engine()
    if not engine:
        if require_advisory:
            # The job that installed the engine passes --require-advisory and
            # owns its absence (L10); a skip nobody owns is where a shipping
            # defect hides (L2). Everywhere else the inventory verdict stands
            # and the skip is recorded, not silent.
            report["advisory"] = {
                "state": "FAIL",
                "engine": ADVISORY_ENGINE,
                "error": _absent_engine_error(),
            }
            report["state"] = "FAIL"
        return report
    listed = list(expected["build_system"])
    for items in expected["optional"].values():
        listed.extend(items)
    listed.extend(expected["runtime"])
    if not listed:
        report["advisory"] = {
            "state": "PASS",
            "engine": ADVISORY_ENGINE,
            "note": "no install set remains after asserting the empty runtime inventory",
        }
        return report
    report["advisory"] = _run_advisory(engine, listed)
    if report["advisory"]["state"] != "PASS":
        report["state"] = "FAIL"
    return report


def advisory_engine() -> str | None:
    """The advisory engine's executable, or None when the host has none.

    ``PIP_AUDIT`` may name a path or a command; either goes through the same
    lookup the default ``pip-audit`` takes, so a configured engine that does
    not resolve is reported as absent rather than silently replaced by
    whatever PATH holds (L13).
    """
    configured = os.environ.get(ADVISORY_ENGINE_ENV, "").strip()
    return shutil.which(configured or ADVISORY_ENGINE)


def _absent_engine_error() -> str:
    configured = os.environ.get(ADVISORY_ENGINE_ENV, "").strip()
    if configured:
        return (
            f"advisory rail required and {ADVISORY_ENGINE_ENV}="
            f"{_redacted_requirement(configured)} does not resolve to an executable"
        )
    return (
        f"advisory rail required and no engine found: {ADVISORY_ENGINE} on PATH, "
        f"or {ADVISORY_ENGINE_ENV} naming one"
    )


# The floor grammar, declared here exactly (L4, L18): a name, optional extras,
# ONE of `==` / `>=`, and a plain version. No markers, URLs, wildcards, compound
# or arbitrary-equality specifiers -- the inventory's own shapes are the two
# above, and a declaration outside the grammar is refused and reported, never
# approximated. Growing the grammar is a deliberate edit here plus a witness.
_FLOOR_GRAMMAR = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?P<extras>\s*\[[A-Za-z0-9._,\s-]*\])?"
    r"\s*(?P<op>==|>=)\s*(?P<version>[0-9][0-9A-Za-z.!+-]*)\s*$"
)


def canonical_name(name: str) -> str:
    """PEP 503 normalization; the engine reports names in this form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def floor_pins(listed: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(exact pins at each declaration's lowest admitted version, their
    canonical names, the declarations the grammar refuses).

    A declaration is a promise over a SET of versions. The resolved run audits
    the set's practical maximum (what pip installs today); this pins its
    minimum, which is the version a stale cache or an old lock still
    satisfies the declaration with, and the one most likely to be vulnerable.
    """
    pins: list[str] = []
    names: list[str] = []
    refused: list[str] = []
    for item in listed:
        match = _FLOOR_GRAMMAR.match(item)
        if not match:
            refused.append(item)
            continue
        extras = (match.group("extras") or "").strip()
        pins.append(f"{match.group('name')}{extras}=={match.group('version')}")
        names.append(canonical_name(match.group("name")))
    return pins, names, refused


def _advisory_command(engine: str, requirements: Path, *, floor: bool) -> list[str]:
    cmd = [
        engine,
        "--requirement", str(requirements),
        # The engine's own report, parsed strictly below; the columns
        # rendering was a tail of prose the gate could only grep.
        "--format", "json",
        # A dependency the engine cannot collect fails the audit outright
        # rather than becoming a skip inside a green run (L15).
        "--strict",
        # IDs, aliases and fix versions are the finding; the advisory prose
        # is not copied into a report that is itself an egress surface (L7).
        "--desc", "off",
        "--progress-spinner", "off",
    ]
    if floor:
        # Exact pins: no resolver, no scratch venv, only the advisory
        # database. This is also the run that covers what the resolved run
        # structurally omits -- pip-audit drops its scratch venv's own
        # pip/setuptools/wheel from the resolver's report, so a requirements
        # file holding only `setuptools>=83.0.0` audits NOTHING there and
        # exits 0. The floor run audits that floor by name.
        cmd += ["--no-deps", "--disable-pip"]
    return cmd


def _run_advisory(engine: str, listed: list[str]) -> dict[str, Any]:
    """The engine over the asserted install set, twice, reconciled against
    the declaration: the resolved run (what pip installs today, transitive
    closure included) and the floor run (each declaration at its lowest
    admitted version). A declared name neither run audited is uncovered,
    and uncovered is a FAIL (L15) -- the report says which."""
    pins, names, refused = floor_pins(listed)
    runs: dict[str, dict[str, Any]] = {"resolved": _engine_run(engine, listed, floor=False)}
    if pins:
        runs["floor"] = _engine_run(engine, pins, floor=True)
    else:
        runs["floor"] = {
            "state": "SKIPPED", "audited": 0,
            "note": "no declaration inside the floor grammar",
        }
    return _combined_verdict(listed, names, refused, runs)


def _engine_run(engine: str, requirements: list[str], *, floor: bool) -> dict[str, Any]:
    """One engine invocation, as a report field.

    The engine reads a requirements FILE (2.10.1 refuses ``-``: "invalid
    requirements input"; the rail that passed it had never run). The set is
    written to a private temporary directory that is removed on every exit
    path, including the engine never launching: a declaration can carry a
    credential, and the file must not outlive the audit (L7).
    """
    try:
        tmp = tempfile.mkdtemp(prefix="bcir-advisory-")
    except OSError as exc:
        # A full or unwritable temporary directory is a verdict of this run,
        # never a traceback out of the required audit (L1).
        return {
            "state": "FAIL", "audited": 0,
            "error": _redacted_requirement(f"could not create the requirements directory: {exc}"),
        }
    try:
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=tmp, suffix=".txt", delete=False
            ) as handle:
                handle.write("\n".join(requirements) + "\n")
                path = Path(handle.name)
        except OSError as exc:
            return {
                "state": "FAIL", "audited": 0,
                "error": _redacted_requirement(f"could not write the requirements file: {exc}"),
            }
        # The shared bounded runner: its own session, a wall bound, per-stream
        # byte budgets, and a process-group put-down — the resolver does
        # network work, spawns pip children, and can flood; every failure
        # shape becomes the run's fail-closed state, never a hang, OOM, or
        # traceback (L8).
        outcome = run_bounded(
            _advisory_command(engine, path, floor=floor),
            timeout=ADVISORY_TIMEOUT,
            cap=ADVISORY_OUTPUT_CAP,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _run_verdict(outcome)


def _tail(stream: bytes) -> str:
    return _redacted_requirement(stream.decode("utf-8", "replace")[-ADVISORY_TAIL:])


def _run_verdict(outcome: dict[str, Any]) -> dict[str, Any]:
    """One run's outcome as a report field. The ``_names`` / ``_skipped`` /
    ``_vulnerable`` entries are for the combining step, which pops them."""
    run: dict[str, Any] = {"state": "FAIL", "audited": 0}
    failure = ""
    if not outcome["launched"]:
        failure = outcome["error"]
    elif outcome["timed_out"]:
        failure = f"{ADVISORY_ENGINE} timed out after {ADVISORY_TIMEOUT:g}s"
    elif outcome["overflow"]:
        failure = f"{ADVISORY_ENGINE} output exceeded {ADVISORY_OUTPUT_CAP} bytes per stream"
    elif outcome["pipes_held"]:
        failure = f"descendant processes still hold {ADVISORY_ENGINE}'s pipes"
    if failure:
        run["error"] = _redacted_requirement(failure)
        return run
    run["returncode"] = outcome["returncode"]
    # The engine says what it found on stderr ("No known vulnerabilities
    # found", "Found N known vulnerabilities in M packages", or the
    # resolver's complaint) and echoes the requirements it was handed in
    # every one of them; a report field is an egress surface, so the tail
    # is retained redacted (L7).
    run["stderr_tail"] = _tail(outcome["stderr"])
    parsed = _parse_advisory_report(outcome["stdout"])
    if parsed is None:
        run["error"] = "unusable engine output (not the engine's own JSON report)"
        run["stdout_tail"] = _tail(outcome["stdout"])
        return run
    names, skipped, vulnerable = parsed
    run["audited"] = len(names)
    run["_names"] = names
    run["_skipped"] = skipped
    run["_vulnerable"] = vulnerable
    if skipped:
        run["error"] = f"{len(skipped)} dependency(ies) the engine could not audit"
    elif vulnerable:
        count = sum(len(item["vulns"]) for item in vulnerable)
        run["error"] = f"{count} known vulnerability(ies) in {len(vulnerable)} package(s)"
    elif outcome["returncode"] != 0:
        run["error"] = f"{ADVISORY_ENGINE} exited {outcome['returncode']} without a finding"
    else:
        run["state"] = "PASS"
    return run


def _combined_verdict(
    listed: list[str],
    names: list[str],
    refused: list[str],
    runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    advisory: dict[str, Any] = {
        "state": "FAIL", "engine": ADVISORY_ENGINE, "declared": len(listed), "runs": runs,
    }
    reasons: list[str] = []
    if refused:
        advisory["unattributable"] = _redacted(refused)
        reasons.append(
            f"{len(refused)} declaration(s) outside the floor grammar "
            "(name[extras]==version or name[extras]>=version)"
        )
    audited = 0
    covered: set[str] = set()
    skipped: list[dict[str, Any]] = []
    vulnerable: list[dict[str, Any]] = []
    for label, run in runs.items():
        covered.update(run.pop("_names", []))
        audited += run.get("audited", 0)
        skipped.extend({"run": label, **item} for item in run.pop("_skipped", []))
        vulnerable.extend({"run": label, **item} for item in run.pop("_vulnerable", []))
        if run["state"] == "FAIL":
            reasons.append(f"{label} run: {run['error']}")
    declared = set(names)
    uncovered = sorted(declared - covered)
    advisory["audited"] = audited
    advisory["covered"] = sorted(declared & covered)
    advisory["uncovered"] = uncovered
    if skipped:
        advisory["skipped"] = _redacted(skipped)
    if uncovered:
        if audited == 0:
            reasons.append(
                f"INVALID/VACUOUS: the engine audited none of the {len(listed)} declared "
                "dependencies"
            )
        else:
            reasons.append(
                f"{len(uncovered)} declared dependency(ies) never audited: "
                + ", ".join(uncovered)
            )
    if vulnerable:
        advisory["vulnerable"] = _redacted(vulnerable)
    if reasons:
        advisory["error"] = "; ".join(reasons)
        return advisory
    advisory["state"] = "PASS"
    return advisory


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _parse_advisory_report(
    raw: bytes,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """pip-audit's ``--format json`` report reduced to (audited canonical
    names, skipped, vulnerable), or None when the bytes are not that report.

    The report is INPUT to this gate exactly as the metadata is: strict
    decode, strict JSON (a duplicate key says two things; the parser has
    refused that since R23), and the shape checked field by field, because
    a document that parses can still be the wrong document (L4). A depth
    that bottoms out the parser under the byte cap is the same refusal (L1).
    Every string that survives is a report field; the caller redacts.
    """
    try:
        data = strict_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateKeys, RecursionError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("dependencies"), list):
        return None
    names: list[str] = []
    skipped: list[dict[str, Any]] = []
    vulnerable: list[dict[str, Any]] = []
    for entry in data["dependencies"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            return None
        if "skip_reason" in entry or not isinstance(entry.get("version"), str):
            skipped.append({
                "name": entry["name"],
                "reason": str(entry.get("skip_reason", "no version collected")),
            })
            continue
        vulns = entry.get("vulns")
        if not isinstance(vulns, list):
            return None
        findings: list[dict[str, Any]] = []
        for vuln in vulns:
            if not isinstance(vuln, dict) or not isinstance(vuln.get("id"), str):
                return None
            findings.append({
                "id": vuln["id"],
                "aliases": _string_items(vuln.get("aliases")),
                "fix_versions": _string_items(vuln.get("fix_versions")),
            })
        names.append(canonical_name(entry["name"]))
        if findings:
            vulnerable.append({
                "name": entry["name"], "version": entry["version"], "vulns": findings,
            })
    return names, skipped, vulnerable

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit_dependencies")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expected", type=Path, default=EXPECTED)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--require-advisory", action="store_true",
        help="the advisory rail must RUN: no engine is a FAIL. The CI job that installs "
             "pip-audit passes this and owns the rail's absence (L10); every other caller "
             "asserts the inventory and records a missing engine as skipped.",
    )
    args = parser.parse_args(argv)
    report = audit(args.root, args.expected, require_advisory=args.require_advisory)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    advisory = report["advisory"]
    detail = ""
    if "audited" in advisory:
        detail = (
            f" audited={advisory['audited']} "
            f"covered={len(advisory['covered'])}/{len(advisory['covered']) + len(advisory['uncovered'])}"
        )
    if "error" in advisory:
        detail += f" error={advisory['error']!r}"
    print(
        f"audit_dependencies: {report['state']} asserted={report['inventory_asserted']} "
        f"expected={report['expected_packages']} mismatches={len(report['mismatches'])} "
        f"advisory={advisory['state']}{detail}"
    )
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
