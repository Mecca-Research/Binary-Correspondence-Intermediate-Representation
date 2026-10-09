#!/usr/bin/env python3
"""The oracle inventory (CXX0): every module of the Python oracle, its native twin or none, the
gates that hold each twin to it, and the hot paths the before/after audit times.

The 2026-10-06 audit (item 17) found the whole-oracle inventory the migration proposal asked for
(its CXX0) never done: which oracle code has a C twin, what holds the two together, and what is
still Python-only was known only by reading the tree. This tool reads it out of the tree's own
sources -- never a mirrored list (a mirror list drifts):

* the oracle: every `bcir/**/*.py` outside `bcir/tests`;
* the twins: each C unit of `runtime/c` (a library or tool source in `runtime/manifest.json`)
  names the oracle modules it mirrors in its leading comment -- `bcir/abi/planner_abi.py` or
  `bcir.gem.handoff.freeze_claims` -- and a unit whose header names them shares them;
* the gates: the manifest's harnesses, sections and fuzzers that build each unit;
* the hot paths: `tools/perf/ab_audit.py`'s row functions, each importing the oracle entry points
  it times.

It writes `docs/BCIR_ORACLE_INVENTORY.md`; `--check` fails (exit 1) when the committed file is not
what the sources give. Deterministic: no timing, no interpreter-dependent count -- the measured
numbers live in the GEM+ harness and the A/B audit, which the inventory names.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join("docs", "BCIR_ORACLE_INVENTORY.md")
_C_DIR = os.path.join("runtime", "c")
_LEAD = 60  # the leading comment of a C unit: where it names what it mirrors
_PATH_REF = re.compile(r"\bbcir/([A-Za-z0-9_/]+?)(?:\.py\b|/?(?=[\s`'\",;:).]|$))")
_DOTTED_REF = re.compile(r"\bbcir\.([A-Za-z0-9_.]+)")


def _rel(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def oracle_modules() -> list[str]:
    """Every oracle module as a dotted name (`bcir.gem.streampack`), tests excluded."""
    out = []
    for base, dirs, files in os.walk(_rel("bcir")):
        dirs[:] = sorted(d for d in dirs if d not in ("tests", "__pycache__"))
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.relpath(os.path.join(base, name), ROOT)
                dotted = path[:-3].replace(os.sep, ".")
                out.append(dotted.removesuffix(".__init__"))
    return sorted(out)


def _source(module: str) -> str:
    base = _rel(*module.split("."))
    return os.path.join(base, "__init__.py") if os.path.isdir(base) else base + ".py"


def _tree(module: str) -> ast.Module:
    with open(_source(module), encoding="utf-8") as f:
        return ast.parse(f.read())


def _own_code(module: str) -> bool:
    """A module defines code of its own: a function or a class at its top level (a package whose
    `__init__` only re-exports mirrors nothing itself)."""
    return any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in _tree(module).body)


def _defining(module: str, name: str, modules: set[str]) -> str:
    """The module that defines `name` as `module` exposes it: a package's relative re-export
    (`from .sub import name`) is followed to `sub`; anything else is `module` itself."""
    for node in _tree(module).body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            if any((a.asname or a.name) == name for a in node.names):
                package = (
                    module if _source(module).endswith("__init__.py") else module.rpartition(".")[0]
                )
                target = f"{package}.{node.module}"
                if target in modules:
                    return target
    return module


def _resolve_path(ref: str, modules: set[str]) -> str | None:
    """`bcir/<ref>` names a module (or a package with code of its own) exactly, or nothing."""
    name = "bcir." + ref.strip("/").replace("/", ".")
    return name if name in modules and _own_code(name) else None


def _resolve_dotted(ref: str, modules: set[str]) -> str | None:
    """`bcir.<ref>` is a module, or a module and an attribute: the module that defines it."""
    parts = ref.strip(".").split(".")
    for cut in range(len(parts), 0, -1):
        name = "bcir." + ".".join(parts[:cut])
        if name in modules:
            if cut < len(parts):
                name = _defining(name, parts[cut], modules)
            return name if _own_code(name) else None
    return None


def _lead(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readline() for _ in range(_LEAD))


def native_units(manifest: dict) -> list[str]:
    """The C units of the production rail: every library and tool source in the manifest."""
    units = set()
    for section in ("libraries", "tools"):
        for entry in manifest.get(section, {}).values():
            units.update(entry.get("sources", []))
    return sorted(units)


_ANY_PY = re.compile(r"\b((?:[A-Za-z0-9_]+/)+[A-Za-z0-9_]+\.py)\b")
_INCLUDE = re.compile(r'^\s*#\s*include\s+"(bcir_[A-Za-z0-9_]+\.h)"', re.M)


def _unit_texts(unit: str) -> list[str]:
    """The leading comments a unit speaks through: its own, its header's, and those of the
    header-only interfaces it implements -- a header with no unit of its own that names this
    one in its leading comment (the ExecutionPlan view names `bcir_runtime.c`, which decodes
    it); an includer the header does not name only uses the interface."""
    texts, seen = [], set()
    names = [unit, unit[:-2] + ".h"]
    path = _rel(_C_DIR, unit)
    if os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for header in _INCLUDE.findall(f.read()):
                own = _rel(_C_DIR, header[:-2] + ".c")
                if not os.path.isfile(own) and unit in _lead(_rel(_C_DIR, header)):
                    names.append(header)
    for name in names:
        path = _rel(_C_DIR, name)
        if name not in seen and os.path.isfile(path):
            seen.add(name)
            texts.append(_lead(path))
    return texts


def _references(text: str, modules: set[str], packages: set[str]) -> tuple[set[str], set[str]]:
    """(modules, packages) a comment names: `bcir/x/y.py`, `bcir.x.y.attr`, a path below `bcir/`
    without its prefix when exactly one module ends with it (`cfront/diagnostics.py`), and a
    package by its directory (`bcir/asn1/`)."""
    named, whole = set(), set()
    for ref in _PATH_REF.findall(text):
        module = _resolve_path(ref, modules)
        if module is not None:
            named.add(module)
        elif "bcir." + ref.strip("/").replace("/", ".") in packages:
            whole.add("bcir." + ref.strip("/").replace("/", "."))
    for ref in _DOTTED_REF.findall(text):
        module = _resolve_dotted(ref, modules)
        if module is not None:
            named.add(module)
        elif "bcir." + ref.strip(".") in packages:
            whole.add("bcir." + ref.strip("."))
    for ref in _ANY_PY.findall(text):
        if ref.startswith("bcir/"):
            continue
        suffix = "." + ref[:-3].replace("/", ".")
        hits = [m for m in modules if m.endswith(suffix) and not m.startswith("bcir.tests")]
        if len(hits) == 1 and _own_code(hits[0]):
            named.add(hits[0])
    return named, whole


def twins(manifest: dict, modules: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    """C unit -> (the oracle modules it names, the packages it names as a whole -- only where it
    names no module inside them)."""
    known = set(modules)
    packages = {m for m in modules if _source(m).endswith("__init__.py")}
    out = {}
    for unit in native_units(manifest):
        named, whole = set(), set()
        for text in _unit_texts(unit):
            n, w = _references(text, known, packages)
            named |= n
            whole |= w
        whole = {p for p in whole if not any(m.startswith(p + ".") or m == p for m in named)}
        if named or whole:
            out[unit] = (sorted(named), sorted(whole))
    return out


def gates(manifest: dict) -> dict[str, dict[str, list[str]]]:
    """C unit -> {'sections': [...], 'fuzzers': [...]}: the gates that build it."""
    libraries = {name: set(e.get("sources", [])) for name, e in manifest["libraries"].items()}

    def units_of(entry) -> set[str]:
        units = set(entry.get("sources", []))
        for lib in entry.get("libraries", []):
            units |= libraries.get(lib, set())
        return units

    harness_units = {name: units_of(e) for name, e in manifest["harnesses"].items()}
    for name, variant in manifest.get("variants", {}).items():
        if variant.get("of") in harness_units:
            harness_units[name] = harness_units[variant["of"]]
    out: dict[str, dict[str, list[str]]] = {}
    for section, entry in manifest["sections"].items():
        covered = set()
        for harness in entry.get("harnesses", []):
            covered |= harness_units.get(harness, set())
        for unit in covered:
            out.setdefault(unit, {"sections": [], "fuzzers": []})["sections"].append(section)
    for fuzzer, entry in manifest["fuzzers"].items():
        for unit in units_of(entry):
            out.setdefault(unit, {"sections": [], "fuzzers": []})["fuzzers"].append(fuzzer)
    for entry in out.values():
        entry["sections"].sort()
        entry["fuzzers"].sort()
    return out


#: The modules that build the A/B audit's fixtures: what a row times is the call it makes on them.
_FIXTURES = ("bcir.performance_audit", "bcir.examples")


def _is_function(module: str, name: str) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in _tree(module).body)


def hot_paths(modules: list[str]) -> list[tuple[str, list[str]]]:
    """(A/B row function, the oracle entry points it imports), from tools/perf/ab_audit.py."""
    known = set(modules)
    with open(_rel("tools", "perf", "ab_audit.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = []
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("row_")):
            continue
        entries = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom) and (sub.module or "").startswith("bcir"):
                if sub.module not in known or sub.module in _FIXTURES:
                    continue
                for alias in sub.names:
                    defined = _defining(sub.module, alias.name, known)
                    if f"{sub.module}.{alias.name}" in known or not _is_function(
                        defined, alias.name
                    ):
                        continue  # a module, a class or a constant: not an entry point
                    entries.add(f"{defined}.{alias.name}")
        if entries:
            out.append((node.name.removeprefix("row_"), sorted(entries)))
    return out


def _package(module: str) -> str:
    """`bcir.<package>` for a module inside one (or the package itself); `bcir` for the rest."""
    parts = module.split(".")
    if len(parts) > 2:
        return ".".join(parts[:2])
    if len(parts) == 2 and _source(module).endswith("__init__.py"):
        return module
    return "bcir"


def render() -> str:
    with open(_rel("runtime", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    modules = oracle_modules()
    twin_of = twins(manifest, modules)
    held = gates(manifest)
    mirrored: dict[str, list[str]] = {}
    for unit, (named, _whole) in twin_of.items():
        for module in named:
            mirrored.setdefault(module, []).append(unit)
    paths = hot_paths(modules)
    hot_modules = {entry.rsplit(".", 1)[0] for _row, entries in paths for entry in entries}

    packages: dict[str, list[str]] = {}
    for module in modules:
        packages.setdefault(_package(module), []).append(module)

    lines = [
        "# The oracle inventory (CXX0)",
        "",
        "<!-- Generated by tools/perf/oracle_inventory.py from the tree's own sources; do not edit.",
        "     `python3 tools/perf/oracle_inventory.py --check` fails when this file is stale. -->",
        "",
        "Every module of the Python oracle (`bcir/`, tests excluded), whether a C unit of the",
        "production rail mirrors it, the gates that hold each twin to it, and the hot paths the",
        "before/after audit (`tools/perf/ab_audit.py`) times. A twin is a unit that names the module",
        "in its leading comment; its gates are the `runtime/manifest.json` sections and fuzzers",
        "that build it. The measured numbers are not here: they live in the GEM+ harness",
        "(`tools/perf/gemplus_baseline.py`) and the A/B audit, by the row names below.",
        "",
        "## Summary",
        "",
        "| package | modules | with a native twin | on a timed hot path |",
        "|---|---:|---:|---:|",
    ]
    for package in sorted(packages):
        members = packages[package]
        lines.append(
            f"| `{package}` | {len(members)} | {sum(m in mirrored for m in members)} | "
            f"{sum(m in hot_modules for m in members)} |"
        )
    lines += [
        f"| **all** | **{len(modules)}** | **{len(mirrored)}** | **{len(hot_modules)}** |",
        "",
        "## Hot paths",
        "",
        "Each row of the A/B audit, the oracle entry points it times, and whether their module has",
        "a native twin.",
        "",
        "| A/B row | oracle entry points | native twin of the module |",
        "|---|---|---|",
    ]
    for row, entries in paths:
        units = sorted({u for e in entries for u in mirrored.get(e.rsplit(".", 1)[0], [])})
        twin = ", ".join(f"`{u}`" for u in units) or "Python-only"
        lines.append(f"| `{row}` | {', '.join(f'`{e}`' for e in entries)} | {twin} |")
    lines += [
        "",
        "## Native twins",
        "",
        "| C unit | oracle modules it mirrors | sections | fuzzers |",
        "|---|---|---|---|",
    ]
    for unit in sorted(twin_of):
        g = held.get(unit, {"sections": [], "fuzzers": []})
        named, whole = twin_of[unit]
        mirrors = [f"`{m}`" for m in named] + [f"`{p}` (the package)" for p in whole]
        lines.append(
            f"| `{unit}` | {', '.join(mirrors)} | "
            f"{', '.join(g['sections']) or '—'} | {', '.join(g['fuzzers']) or '—'} |"
        )
    silent = [u for u in native_units(manifest) if u not in twin_of]
    lines += [
        "",
        "A unit that names a package mirrors it as a whole, and is not counted against any one",
        "of its modules above. The units that name no oracle module are native-only (the model",
        "loaders and kernels, the quarantine, the driver hooks, the shared digests) or speak",
        "through another unit's interface: " + ", ".join(f"`{u}`" for u in silent) + ".",
    ]
    lines += [
        "",
        "## Python-only, by package",
        "",
        "The oracle modules no C unit mirrors: the semantics, the research and the generators the",
        "migration proposal keeps in Python (its §8.2), and the candidates for the next twin.",
        "",
    ]
    for package in sorted(packages):
        rest = [m for m in packages[package] if m not in mirrored]
        if rest:
            lines.append(f"- `{package}` ({len(rest)}): " + ", ".join(f"`{m}`" for m in rest))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail when the committed file is stale"
    )
    args = parser.parse_args(argv)
    text = render()
    path = _rel(OUT)
    if args.check:
        try:
            with open(path, encoding="utf-8") as f:
                current = f.read()
        except OSError:
            current = None
        if current != text:
            print(f"{OUT} is STALE -- regenerate: python3 tools/perf/oracle_inventory.py")
            return 1
        print(f"{OUT} is current.")
        return 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
