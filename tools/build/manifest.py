#!/usr/bin/env python3
"""The build manifest checker: runtime/manifest.json is held to the tree and the tree to it.

The manifest is the one list of what the C and C++ rails are made of (CMake reads it directly;
docs/BCIR_BUILD_ROADMAP.md). This checker is the reconciliation the roadmap's BUILD-1 slice
promised: every rule below is a question a gate, a harness or the memory-class inventory used
to answer on its own, and `bcir/tests/test_build_manifest.py` injects a violation of each one
and asserts that it fires (docs/security/laws.md L2, L11).

    python tools/build/manifest.py --check            # exit 0 clean, 1 with findings, 2 unusable
    python tools/build/manifest.py --closure bcir-cc  # the sources, libraries and options a unit links
    python tools/build/manifest.py --deps-index build/cmake/bcir-deps.json   # the configure's index

The rules, each with the code its findings carry:

  M1  schema       the schema tag, the six unit kinds and a non-empty freestanding_checks list
  M2  shape        sources/libraries/link/options are lists of strings; class only on libraries,
                   max_len only on fuzzers and a positive integer; no unknown keys
  M3  files        every source exists under runtime/c (seam kinds: runtime/cpp) with its extension
  M4  roles        every bcir_*.c is in exactly one library or is one tool's main; every test_*.c
                   one harness; every fuzz_*.c one fuzzer; every runtime/cpp/*.cpp one seam unit;
                   fixtures (cfront_*.c) are never units
  M5  references   `libraries` name known libraries, never the unit itself, with no cycle
  M6  link/options link names are m or pthread; options are flags, optionally gcc:/clang: scoped
  M7  classes      a library's class is one of runtime/c/MEMORY_CLASSIFICATION.txt's and every unit
                   of it carries that class there; every tool main is a hosted_tool
  M8  freestanding every freestanding check is a freestanding_core library unit, listed once, and
                   the set covers every unit the shell gates compile -ffreestanding
  M9  gates        every runtime source a shell gate names is a manifest unit; wherever a gate's
                   compile line or array names a main, the units beside it lie in that unit's
                   closure; fuzz targets match tools/c/fuzz_streampack.sh's sources and -max_len
  M10 harnesses    the same for every Python module under bcir/ and tools/ that names a source,
                   per list, tuple or set literal that names it
  M11 seam         the C++ seam's units equal handoff_fixtures/check_handoff.sh's (the #719 pair)
  M12 presets      CMakePresets.json builds and tests with two workers (AGENTS.md's cap)
  M13 sections     every `sections` entry names a script under tools/c/sections/ that exists and
                   manifest harnesses in argument order; every script in that directory is one
                   section; tools/c/check_runtime.sh calls each script (the gate and the CTest entry
                   run one text)
  D1  deps index   bcir-deps.json (written at configure) is JSON of schema bcir-deps.v1 with the
                   compilers named and one {name, found: true|false, detail} row per dependency,
                   names unique, PYTHON3 / THREADS / MLIR among them

Scope: the checker reads the gates and the Python modules as text (string literals and compile
lines); a source list assembled at run time is outside it, and a gate that links a unit the
manifest lacks is a finding, never a skip. Nothing here runs a compiler. The tree is read once
(`Tree`) so a test can hold many manifests to the same facts.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "runtime" / "manifest.json"
PRESETS = ROOT / "CMakePresets.json"
SCHEMA = "bcir-build-manifest.v1"
C_KINDS = ("libraries", "tools", "harnesses", "fuzzers")
CPP_KINDS = ("seam_libraries", "seam_tests")
KINDS = C_KINDS + CPP_KINDS
LIBRARY_KINDS = ("libraries", "seam_libraries")
UNIT_KEYS = ("sources", "libraries", "link", "options", "class", "max_len")
TOP_KEYS = ("schema", "comment", "freestanding_checks", "sections", *KINDS)
SECTIONS_DIR = "tools/c/sections"
RUNTIME_GATE = "tools/c/check_runtime.sh"
LINK_NAMES = ("m", "pthread")
CLASSES = ("freestanding_core", "hosted_tool", "driver_adapter")
WORKERS = 2
GATES = (
    "tools/c/check_runtime.sh",
    "tools/c/fuzz_streampack.sh",
    "tools/c/check_memory_discipline.sh",
    "tools/c/check_streampack_semantic.sh",
    "tools/c/sanitize_cfront.sh",
    "tools/c/check_target_abi.sh",
    "tools/cpp/check_handoff.sh",
    "tools/cpp/check_jer_index.sh",
    "tools/cpp/check_jer_simd.sh",
    "tools/cpp/check_sycl.sh",
)
FUZZ_GATE = "tools/c/fuzz_streampack.sh"
SEAM_PAIRS = (
    ("tools/cpp/check_handoff.sh", r"seam_cpp=\(([^)]*)\)", r"seam_c=\(([^)]*)\)"),
    (
        "bcir/tests/handoff_fixtures.py",
        r"CPP_UNITS\s*=\s*\(([^)]*)\)",
        r"C_UNITS\s*=\s*\(([^)]*)\)",
    ),
)
PYTHON_TREES = ("bcir", "tools")
SOURCE_TOKEN = re.compile(r"\b((?:bcir|test|fuzz)_\w+\.(?:cpp|c))\b")
SOURCE_LITERAL = re.compile(r"(?:bcir|test|fuzz)_\w+\.(?:cpp|c)")
OPTION = re.compile(r"^(?:(?:gcc|clang):)?-\S+$")


class ManifestError(Exception):
    """The manifest cannot be read at all (exit 2: unusable, distinct from findings)."""


def load(path: Path = MANIFEST) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ManifestError(f"{path}: not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: the manifest is not a JSON object")
    return data


# --- the manifest's shape helpers ---------------------------------------------------------------


def _units(manifest: dict, kind: str) -> dict:
    units = manifest.get(kind)
    return units if isinstance(units, dict) else {}


def _sources(unit: object) -> list[str]:
    if isinstance(unit, dict) and isinstance(unit.get("sources"), list):
        return [s for s in unit["sources"] if isinstance(s, str)]
    return []


def _list(unit: object, key: str) -> list[str]:
    if isinstance(unit, dict) and isinstance(unit.get(key), list):
        return [s for s in unit[key] if isinstance(s, str)]
    return []


def _library(manifest: dict, name: str) -> dict | None:
    for kind in LIBRARY_KINDS:
        if name in _units(manifest, kind):
            unit = _units(manifest, kind)[name]
            return unit if isinstance(unit, dict) else None
    return None


def closure(manifest: dict, kind: str, name: str) -> dict[str, list[str]]:
    """Everything a unit links: its sources, then each library's (dependents first), each once."""
    unit = _units(manifest, kind).get(name)
    if unit is None:
        raise KeyError(f"{kind}/{name}")
    sources = list(_sources(unit))
    link = list(_list(unit, "link"))
    options = list(_list(unit, "options"))
    libraries: list[str] = []
    todo = list(_list(unit, "libraries"))
    seen: set[str] = set()
    while todo:
        lib = todo.pop(0)
        if lib in seen:
            continue
        seen.add(lib)
        found = _library(manifest, lib)
        if found is None:
            continue
        libraries.append(lib)
        sources.extend(s for s in _sources(found) if s not in sources)
        link.extend(s for s in _list(found, "link") if s not in link)
        options.extend(s for s in _list(found, "options") if s not in options)
        todo.extend(_list(found, "libraries"))
    return {"sources": sources, "libraries": libraries, "link": link, "options": options}


def _find_unit(manifest: dict, source: str) -> tuple[str, str] | None:
    """The (kind, name) whose own sources carry `source`, or None."""
    for kind in KINDS:
        for name, unit in _units(manifest, kind).items():
            if source in _sources(unit):
                return kind, name
    return None


# --- the tree's facts, read once -----------------------------------------------------------------


def read_classification(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return entries
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        parts = line.split()
        if len(parts) == 2 and parts[0] in CLASSES:
            entries[parts[1]] = parts[0]
    return entries


def gate_groups(text: str) -> list[tuple[str, str]]:
    """A gate's link groups: every `name=( ... )` array and every line that names a runtime source
    by its directory variable, continuation lines joined, comments dropped."""
    text = re.sub(r"\\\n\s*", " ", text)
    groups: list[tuple[str, str]] = []
    for match in re.finditer(r"(\w+)=\(([^()]*)\)", text):
        groups.append((f"array {match.group(1)}", match.group(2)))
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if re.search(r"\$\{?C(?:PP)?\}?/", line):
            groups.append((f"line {number}", line))
    return groups


def fuzz_targets(text: str) -> list[tuple[str, str | None, list[str]]]:
    """tools/c/fuzz_streampack.sh's add_target rows: (key, -max_len or None, sources)."""
    text = re.sub(r"\\\n\s*", " ", text)
    rows: list[tuple[str, str | None, list[str]]] = []
    for match in re.finditer(r'^add_target\s+(\w+)\s+"[^"]*"\s+"([^"]*)"\s+(.*)$', text, re.M):
        key, flags, rest = match.groups()
        length = re.search(r"-max_len=(\d+)", flags)
        rows.append((key, length.group(1) if length else None, SOURCE_TOKEN.findall(rest)))
    return rows


def python_groups(text: str) -> list[list[str]]:
    """The runtime sources a Python module names, grouped by the list, tuple or set literal that
    names them (a harness's argv or source tuple); a literal outside any such container is a
    group of its own. A module that does not parse yields its literals as one group."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return [[m.group(0) for m in SOURCE_LITERAL.finditer(text)]]
    groups: list[list[str]] = []

    def visit(node: ast.AST, group: list[str] | None) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and SOURCE_LITERAL.fullmatch(node.value):
                if group is None:
                    groups.append([node.value])
                else:
                    group.append(node.value)
            return
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            own: list[str] = []
            for child in ast.iter_child_nodes(node):
                visit(child, own)
            if own:
                groups.append(own)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, group)

    visit(tree, None)
    return groups


class Tree:
    """What the checker reads from the repository: the source directories, the memory classes,
    the gates' link groups, the Python modules' source literals, the seam's two lists and the
    presets. Read once; `check()` holds any manifest to it."""

    def __init__(self, root: Path = ROOT, presets: Path | None = PRESETS) -> None:
        self.root = root
        self.c_dir = root / "runtime" / "c"
        self.cpp_dir = root / "runtime" / "cpp"
        self.c_files = sorted(p.name for p in self.c_dir.glob("*.c"))
        self.cpp_files = sorted(p.name for p in self.cpp_dir.glob("*.cpp"))
        self.classification = read_classification(self.c_dir / "MEMORY_CLASSIFICATION.txt")
        self.gates: dict[str, list[tuple[str, list[str]]]] = {}
        self.gate_tokens: dict[str, list[str]] = {}
        self.missing_gates: list[str] = []
        self.freestanding: set[str] = set()
        for rel in GATES:
            text = self._read(root / rel)
            if text is None:
                self.missing_gates.append(rel)
                continue
            self.gates[rel] = [
                (tag, self.existing(SOURCE_TOKEN.findall(body))) for tag, body in gate_groups(text)
            ]
            self.gate_tokens[rel] = self.existing(SOURCE_TOKEN.findall(text))
            for line in re.sub(r"\\\n\s*", " ", text).splitlines():
                if "-ffreestanding" in line and not line.lstrip().startswith("#"):
                    self.freestanding.update(
                        t
                        for t in SOURCE_TOKEN.findall(line)
                        if t.startswith("bcir_") and t.endswith(".c")
                    )
        fuzz_text = self._read(root / FUZZ_GATE)
        self.fuzz: list[tuple[str, str | None, list[str]]] | None = (
            None
            if fuzz_text is None
            else [(k, n, self.existing(s)) for k, n, s in fuzz_targets(fuzz_text)]
        )
        self.python: list[tuple[str, list[str]]] = []
        for tree in PYTHON_TREES:
            for path in sorted((root / tree).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                text = self._read(path)
                if text is None:
                    continue
                rel_path = path.relative_to(root).as_posix()
                for number, group in enumerate(python_groups(text), 1):
                    tokens = self.existing(group)
                    if tokens:
                        self.python.append((f"{rel_path} (literal group {number})", tokens))
        self.seam: dict[str, tuple[set[str], set[str]] | None] = {}
        for rel, cpp_pattern, c_pattern in SEAM_PAIRS:
            text = self._read(root / rel)
            cpp_match = re.search(cpp_pattern, text, re.S) if text else None
            c_match = re.search(c_pattern, text, re.S) if text else None
            if cpp_match is None or c_match is None:
                self.seam[rel] = None
            else:
                self.seam[rel] = (
                    set(re.findall(r"(\w+\.cpp)", cpp_match.group(1))),
                    set(re.findall(r"(\w+\.c)\b", c_match.group(1))),
                )
        self.section_scripts = sorted(p.name for p in (root / SECTIONS_DIR).glob("*.sh"))
        self.runtime_gate_text = self._read(root / RUNTIME_GATE) or ""
        self.presets: dict | str | None = None
        if presets is not None:
            try:
                self.presets = json.loads(presets.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.presets = f"{presets.name}: {exc}"

    @staticmethod
    def _read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def exists(self, source: str) -> bool:
        return source in (self.cpp_files if source.endswith(".cpp") else self.c_files)

    def existing(self, tokens: list[str]) -> list[str]:
        seen: list[str] = []
        for token in tokens:
            if token not in seen and self.exists(token):
                seen.append(token)
        return seen


# --- the rules ------------------------------------------------------------------------------------


def check_schema(manifest: dict, errors: list[str]) -> None:
    if manifest.get("schema") != SCHEMA:
        errors.append(f"M1: schema is {manifest.get('schema')!r}, not {SCHEMA!r}")
    for key in manifest:
        if key not in TOP_KEYS:
            errors.append(f"M2: unknown top-level key {key!r}")
    for kind in KINDS:
        if not isinstance(manifest.get(kind), dict):
            errors.append(f"M1: no {kind!r} object")
    free = manifest.get("freestanding_checks")
    if not isinstance(free, list) or not free or any(not isinstance(f, str) for f in free):
        errors.append(
            "M1: freestanding_checks is missing or empty (the freestanding core is unproven)"
        )
    for kind in KINDS:
        for name, unit in _units(manifest, kind).items():
            where = f"{kind}/{name}"
            if not isinstance(unit, dict):
                errors.append(f"M2: {where} is not an object")
                continue
            for key in unit:
                if key not in UNIT_KEYS:
                    errors.append(f"M2: {where} has unknown key {key!r}")
            if not _sources(unit) or len(_sources(unit)) != len(unit.get("sources", [])):
                errors.append(f"M2: {where} needs a non-empty list of source strings")
            for key in ("libraries", "link", "options"):
                value = unit.get(key)
                if value is not None and (
                    not isinstance(value, list) or any(not isinstance(v, str) for v in value)
                ):
                    errors.append(f"M2: {where}.{key} is not a list of strings")
            if kind == "libraries":
                if unit.get("class") not in CLASSES:
                    errors.append(f"M2: {where} needs a class from {CLASSES}")
            elif "class" in unit:
                errors.append(f"M2: {where} carries a class; only libraries have one")
            if "max_len" in unit:
                value = unit["max_len"]
                if kind != "fuzzers":
                    errors.append(f"M2: {where} carries max_len; only fuzzers have one")
                elif not (isinstance(value, str) and value.isdigit() and int(value) > 0):
                    errors.append(f"M2: {where}.max_len must be a positive integer string")


def check_files_and_roles(manifest: dict, tree: Tree, errors: list[str]) -> None:
    role_of: dict[str, list[str]] = {}
    for kind in KINDS:
        ext = ".cpp" if kind in CPP_KINDS else ".c"
        for name, unit in _units(manifest, kind).items():
            for source in _sources(unit):
                if not source.endswith(ext) or "/" in source:
                    errors.append(f"M3: {kind}/{name} source {source!r} is not a {ext} basename")
                elif not tree.exists(source):
                    errors.append(
                        f"M3: {kind}/{name} source {source!r} is not in runtime/{'cpp' if ext == '.cpp' else 'c'}"
                    )
                role_of.setdefault(source, []).append(f"{kind}/{name}")
    for source, roles in sorted(role_of.items()):
        if len(roles) > 1:
            errors.append(f"M4: {source} is listed by {', '.join(roles)}; a source has one unit")
        if source.startswith("cfront_"):
            errors.append(f"M4: {source} is a fixture, never a unit")
    expected = {"bcir_": ("libraries", "tools"), "test_": ("harnesses",), "fuzz_": ("fuzzers",)}
    for source in tree.c_files:
        for prefix, kinds in expected.items():
            if source.startswith(prefix):
                roles = role_of.get(source, [])
                if not roles:
                    errors.append(f"M4: {source} is in runtime/c but no manifest unit lists it")
                elif roles[0].split("/")[0] not in kinds:
                    errors.append(f"M4: {source} belongs to {'/'.join(kinds)}, not {roles[0]}")
    for source in tree.cpp_files:
        roles = role_of.get(source, [])
        if not roles:
            errors.append(f"M4: {source} is in runtime/cpp but no manifest unit lists it")
        elif roles[0].split("/")[0] not in CPP_KINDS:
            errors.append(f"M4: {source} belongs to a seam unit, not {roles[0]}")


def check_references(manifest: dict, errors: list[str]) -> None:
    known = {kind: set(_units(manifest, kind)) for kind in LIBRARY_KINDS}
    for kind in KINDS:
        visible = known["libraries"] | (known["seam_libraries"] if kind in CPP_KINDS else set())
        for name, unit in _units(manifest, kind).items():
            for lib in _list(unit, "libraries"):
                if lib == name and kind in LIBRARY_KINDS:
                    errors.append(f"M5: {kind}/{name} links itself")
                elif lib not in visible:
                    errors.append(f"M5: {kind}/{name} links unknown library {lib!r}")
            for sys_lib in _list(unit, "link"):
                if sys_lib not in LINK_NAMES:
                    errors.append(f"M6: {kind}/{name} link {sys_lib!r} is not one of {LINK_NAMES}")
            for option in _list(unit, "options"):
                if not OPTION.match(option):
                    errors.append(f"M6: {kind}/{name} option {option!r} is not a flag")
    for kind in LIBRARY_KINDS:
        for name, unit in _units(manifest, kind).items():
            todo = list(_list(unit, "libraries"))
            seen: set[str] = set()
            while todo:
                lib = todo.pop()
                if lib == name:
                    errors.append(f"M5: {kind}/{name} is in a dependency cycle")
                    break
                if lib in seen:
                    continue
                seen.add(lib)
                found = _library(manifest, lib)
                if found is not None:
                    todo.extend(_list(found, "libraries"))


def check_classes(manifest: dict, tree: Tree, errors: list[str]) -> None:
    if not tree.classification:
        errors.append("M7: runtime/c/MEMORY_CLASSIFICATION.txt is missing or empty")
        return
    for name, unit in _units(manifest, "libraries").items():
        declared = unit.get("class") if isinstance(unit, dict) else None
        for source in _sources(unit):
            actual = tree.classification.get(source)
            if actual is None:
                errors.append(f"M7: libraries/{name} unit {source} is not classified")
            elif actual != declared:
                errors.append(f"M7: libraries/{name} is {declared} but {source} is {actual}")
    for name, unit in _units(manifest, "tools").items():
        for source in _sources(unit):
            if tree.classification.get(source) != "hosted_tool":
                errors.append(f"M7: tools/{name} main {source} is not a hosted_tool")


def check_freestanding(manifest: dict, tree: Tree, errors: list[str]) -> None:
    free = [f for f in manifest.get("freestanding_checks", []) if isinstance(f, str)]
    for source in free:
        if free.count(source) > 1:
            errors.append(f"M8: freestanding check {source} is listed twice")
        unit = _find_unit(manifest, source)
        if unit is None or unit[0] != "libraries":
            errors.append(f"M8: freestanding check {source} is not a library unit")
        elif tree.classification.get(source) != "freestanding_core":
            errors.append(
                f"M8: freestanding check {source} is classified {tree.classification.get(source)}"
            )
    for source in sorted(tree.freestanding - set(free)):
        errors.append(
            f"M8: the gates compile {source} -ffreestanding but freestanding_checks omits it"
        )


def _check_group(
    manifest: dict, errors: list[str], code: str, where: str, tokens: list[str]
) -> None:
    """The rule M9 and M10 share: named units exist in the manifest; the units co-named with a
    main lie in the closure of one of the mains named beside them."""
    mains: list[tuple[str, str, str]] = []
    others: list[str] = []
    for token in tokens:
        if token.startswith("cfront_"):
            continue
        unit = _find_unit(manifest, token)
        if unit is None:
            errors.append(f"{code}: {where} names {token}, which no manifest unit lists")
        elif unit[0] in LIBRARY_KINDS:
            others.append(token)
        else:
            mains.append((token, *unit))
    if not mains or not others:
        return
    covered: set[str] = set()
    for _token, kind, name in mains:
        covered.update(closure(manifest, kind, name)["sources"])
    for token in others:
        if token not in covered:
            names = ", ".join(f"{k}/{n}" for _t, k, n in mains)
            errors.append(
                f"{code}: {where} links {token} with {names}, outside the manifest closure"
            )


def check_gates(manifest: dict, tree: Tree, errors: list[str]) -> None:
    for rel in tree.missing_gates:
        errors.append(f"M9: {rel} is missing")
    for rel, groups in tree.gates.items():
        for tag, tokens in groups:
            _check_group(manifest, errors, "M9", f"{rel} {tag}", tokens)
        for token in tree.gate_tokens[rel]:
            if not token.startswith("cfront_") and _find_unit(manifest, token) is None:
                errors.append(f"M9: {rel} names {token}, which no manifest unit lists")
    if tree.fuzz is None:
        return
    fuzzers = _units(manifest, "fuzzers")
    seen_mains: set[str] = set()
    for key, gate_len, tokens in tree.fuzz:
        mains = [t for t in tokens if t.startswith("fuzz_")]
        if len(mains) != 1:
            errors.append(f"M9: fuzz target {key} names {len(mains)} fuzz_*.c mains")
            continue
        seen_mains.add(mains[0])
        unit = next((n for n, u in fuzzers.items() if mains[0] in _sources(u)), None)
        if unit is None:
            errors.append(f"M9: fuzz target {key} ({mains[0]}) has no manifest fuzzer")
            continue
        wanted = set(closure(manifest, "fuzzers", unit)["sources"])
        for token in tokens:
            if token not in wanted:
                errors.append(
                    f"M9: fuzz target {key} links {token}, outside fuzzers/{unit}'s closure"
                )
        ours = fuzzers[unit].get("max_len") if isinstance(fuzzers[unit], dict) else None
        if gate_len != ours:
            errors.append(
                f"M9: fuzz target {key} has -max_len {gate_len}, fuzzers/{unit} says {ours}"
            )
    for name, unit in fuzzers.items():
        for source in _sources(unit):
            if source not in seen_mains:
                errors.append(f"M9: fuzzers/{name} ({source}) is not a {FUZZ_GATE} target")


def check_python_harnesses(manifest: dict, tree: Tree, errors: list[str]) -> None:
    for where, tokens in tree.python:
        _check_group(manifest, errors, "M10", where, tokens)


def check_seam(manifest: dict, tree: Tree, errors: list[str]) -> None:
    seam = _units(manifest, "seam_libraries").get("bcir_seam")
    if not isinstance(seam, dict):
        errors.append("M11: seam_libraries/bcir_seam is missing")
        return
    cpp_units = set(_sources(seam))
    c_closure = set(closure(manifest, "seam_libraries", "bcir_seam")["sources"])
    for rel, lists in tree.seam.items():
        if lists is None:
            errors.append(f"M11: {rel} no longer spells the seam's unit lists")
            continue
        theirs_cpp, theirs_c = lists
        if theirs_cpp != cpp_units:
            errors.append(
                f"M11: {rel} C++ seam units {sorted(theirs_cpp)} != manifest {sorted(cpp_units)}"
            )
        for unit in sorted(theirs_c - c_closure):
            errors.append(f"M11: {rel} links {unit} into the seam, outside bcir_seam's closure")


def check_sections(manifest: dict, tree: Tree, errors: list[str]) -> None:
    sections = manifest.get("sections")
    if sections is None:
        sections = {}
    if not isinstance(sections, dict):
        errors.append("M13: sections is not an object")
        return
    harnesses = set(_units(manifest, "harnesses"))
    registered: list[str] = []
    for name, section in sections.items():
        where = f"sections/{name}"
        if not isinstance(section, dict) or set(section) != {"script", "harnesses"}:
            errors.append(f"M13: {where} is not {{script, harnesses}}")
            continue
        script = section["script"]
        if (
            not isinstance(script, str)
            or not script.startswith(SECTIONS_DIR + "/")
            or not script.endswith(".sh")
        ):
            errors.append(f"M13: {where} script {script!r} is not a tools/c/sections/*.sh path")
        else:
            basename = script[len(SECTIONS_DIR) + 1 :]
            if basename not in tree.section_scripts:
                errors.append(f"M13: {where} script {script} does not exist")
            registered.append(basename)
            if script not in tree.runtime_gate_text:
                errors.append(
                    f"M13: {RUNTIME_GATE} does not call {script}; the gate and the CTest entry would judge different text"
                )
        names = section["harnesses"]
        if not isinstance(names, list) or not names or any(not isinstance(h, str) for h in names):
            errors.append(f"M13: {where} needs a non-empty list of harness names")
            continue
        for harness in names:
            if harness not in harnesses:
                errors.append(
                    f"M13: {where} names harness {harness!r}, which the manifest does not build"
                )
        if len(set(names)) != len(names):
            errors.append(f"M13: {where} names a harness twice")
    for basename in tree.section_scripts:
        if basename not in registered:
            errors.append(f"M13: {SECTIONS_DIR}/{basename} is not a manifest section")
    for basename in sorted(set(registered)):
        if registered.count(basename) > 1:
            errors.append(f"M13: {SECTIONS_DIR}/{basename} is registered by two sections")


def check_presets(presets: dict | str | None, errors: list[str], workers: int = WORKERS) -> None:
    if presets is None:
        return
    if not isinstance(presets, dict):
        errors.append(f"M12: {presets}")
        return
    configure = {p.get("name") for p in presets.get("configurePresets", [])}
    if not configure:
        errors.append("M12: no configure presets")
    for preset in presets.get("buildPresets", []):
        if preset.get("jobs") != workers:
            errors.append(
                f"M12: build preset {preset.get('name')!r} runs {preset.get('jobs')} jobs, not {workers}"
            )
        if preset.get("configurePreset") not in configure:
            errors.append(
                f"M12: build preset {preset.get('name')!r} names an unknown configure preset"
            )
    tests = presets.get("testPresets", [])
    if not tests:
        errors.append("M12: no test presets")
    for preset in tests:
        jobs = (
            preset.get("execution", {}).get("jobs")
            if isinstance(preset.get("execution"), dict)
            else None
        )
        if jobs != workers:
            errors.append(
                f"M12: test preset {preset.get('name')!r} runs {jobs} jobs, not {workers}"
            )
        if preset.get("configurePreset") not in configure:
            errors.append(
                f"M12: test preset {preset.get('name')!r} names an unknown configure preset"
            )


def check(manifest: dict, tree: Tree | None = None) -> list[str]:
    """Every finding over `manifest` against the tree (empty when clean)."""
    if tree is None:
        tree = Tree()
    errors: list[str] = []
    check_schema(manifest, errors)
    if errors:
        return errors  # the shape is wrong; the rules below would only echo it
    check_files_and_roles(manifest, tree, errors)
    check_references(manifest, errors)
    check_classes(manifest, tree, errors)
    check_freestanding(manifest, tree, errors)
    check_gates(manifest, tree, errors)
    check_python_harnesses(manifest, tree, errors)
    check_seam(manifest, tree, errors)
    check_sections(manifest, tree, errors)
    check_presets(tree.presets, errors)
    return errors


# --- D1: the dependency index -------------------------------------------------------------------

DEPS_SCHEMA = "bcir-deps.v1"
DEPS_REQUIRED = ("PYTHON3", "THREADS", "MLIR")


def check_deps_index(path: Path) -> list[str]:
    """Findings over the configure's dependency index (empty when it is well-formed)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"D1: {path}: {exc}"]
    try:
        data = json.loads(text)
    except ValueError as exc:
        return [f"D1: {path} is not JSON ({exc})"]
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"D1: {path} is not a JSON object"]
    if data.get("schema") != DEPS_SCHEMA:
        errors.append(f"D1: schema is {data.get('schema')!r}, not {DEPS_SCHEMA!r}")
    for key in ("c_compiler", "cxx_compiler", "system"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            errors.append(f"D1: {key} is missing or empty")
    rows = data.get("dependencies")
    if not isinstance(rows, list) or not rows:
        return errors + [
            "D1: dependencies is missing or empty (an index that records nothing indexes nothing)"
        ]
    names: list[str] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "found", "detail"}:
            errors.append(f"D1: dependency row {i} is not {{name, found, detail}}")
            continue
        if not isinstance(row["name"], str) or not row["name"]:
            errors.append(f"D1: dependency row {i} has no name")
        if not isinstance(row["found"], bool):
            errors.append(
                f"D1: dependency {row.get('name')!r} has found={row['found']!r}, not a JSON boolean"
            )
        if not isinstance(row["detail"], str):
            errors.append(f"D1: dependency {row.get('name')!r} detail is not a string")
        names.append(str(row["name"]))
    for name in sorted(set(names)):
        if names.count(name) > 1:
            errors.append(f"D1: dependency {name} is recorded twice")
    for name in DEPS_REQUIRED:
        if name not in names:
            errors.append(f"D1: dependency {name} is not recorded")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--check", action="store_true", help="reconcile the manifest with the tree")
    parser.add_argument(
        "--closure", metavar="UNIT", help="print what a unit links (kind/name or name)"
    )
    parser.add_argument(
        "--deps-index", type=Path, metavar="JSON", help="validate a configure's bcir-deps.json"
    )
    args = parser.parse_args(argv)
    if args.deps_index is not None:
        errors = check_deps_index(args.deps_index)
        for error in errors:
            print(f"FAIL: {error}")
        if errors:
            print(f"deps-index: {len(errors)} finding(s) in {args.deps_index}")
            return 1
        print(f"deps-index: ok ({args.deps_index})")
        return 0
    try:
        manifest = load(args.manifest)
    except ManifestError as exc:
        print(f"manifest: unusable: {exc}")
        return 2
    if args.closure:
        kind, _, name = args.closure.rpartition("/")
        for k in [kind] if kind else list(KINDS):
            if name in _units(manifest, k):
                data = closure(manifest, k, name)
                print(f"{k}/{name}")
                for key in ("sources", "libraries", "link", "options"):
                    print(f"  {key}: {' '.join(data[key])}")
                return 0
        print(f"manifest: no unit named {args.closure!r}")
        return 2
    errors = check(manifest)
    counts = {kind: len(_units(manifest, kind)) for kind in KINDS}
    for error in errors:
        print(f"FAIL: {error}")
    summary = ", ".join(f"{n} {kind}" for kind, n in counts.items())
    sections = manifest.get("sections")
    summary += f", {len(sections) if isinstance(sections, dict) else 0} sections"
    free = (
        len(manifest.get("freestanding_checks", []))
        if isinstance(manifest.get("freestanding_checks"), list)
        else 0
    )
    if errors:
        print(f"manifest: {len(errors)} finding(s) over {summary}, {free} freestanding checks")
        return 1
    print(
        f"manifest: ok ({summary}, {free} freestanding checks; gates, harnesses, classes, presets reconciled)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
