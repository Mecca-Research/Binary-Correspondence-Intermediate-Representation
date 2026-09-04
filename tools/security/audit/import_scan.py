#!/usr/bin/env python3
"""Every third-party module the tree imports, reconciled against the declared extras.

    python tools/security/audit/import_scan.py [--root .] [--out report.json]

Report-only audit tooling (not a gate). The declared inventory says what the package
PROMISES to need; this says what the code actually reaches for. A module imported
somewhere but declared nowhere is an optional dependency no advisory scan can audit,
because it has no declared version (kafka-python was the 2026-09-04 instance).
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import sys
import tomllib

SKIP = {".git", "build", "dist", "__pycache__", "node_modules", ".mypy_cache"}
# Distribution names whose import name differs, for the declared-extras reconciliation.
IMPORT_NAMES = {"pre-commit": "pre_commit", "kafka-python": "kafka", "safetensors": "safetensors"}


def declared_imports(root: str) -> dict[str, str]:
    """import name -> the declaration (runtime or extra) that provides it."""
    out: dict[str, str] = {}
    try:
        with open(os.path.join(root, "pyproject.toml"), "rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return out
    groups = {"runtime": project.get("dependencies", [])}
    groups.update(project.get("optional-dependencies", {}))
    for group, items in groups.items():
        for item in items:
            name = item.split("[")[0].split(">")[0].split("=")[0].split("<")[0].split("~")[0]
            name = name.strip().lower()
            out[IMPORT_NAMES.get(name, name.replace("-", "_"))] = f"{group}: {item}"
    return out


def local_names(root: str) -> set[str]:
    names: set[str] = set()
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP]
        if "__init__.py" in fn:
            names.add(os.path.basename(dp))
        if dp == root:
            names.update(f[:-3] for f in fn if f.endswith(".py"))
            names.update(d.replace("-", "_") for d in dn)
            names.update(dn)
        # scripts add their own directory to sys.path and import siblings by bare name
        names.update(f[:-3] for f in fn if f.endswith(".py"))
    return names


def scan(root: str) -> dict:
    stdlib = set(sys.stdlib_module_names)
    local = local_names(root)
    declared = declared_imports(root)
    uses: dict[str, dict] = collections.defaultdict(
        lambda: {"files": set(), "guarded": 0, "unguarded": 0}
    )
    scanned = 0
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP]
        for f in fn:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dp, f)
            try:
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=path)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            scanned += 1
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in stdlib or name in local or name.startswith("_"):
                        continue
                    chain = []
                    cur = node
                    while cur in parents:
                        cur = parents[cur]
                        chain.append(cur)
                    guarded = any(
                        isinstance(p, (ast.Try, ast.If, ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef, ast.With)) for p in chain
                    )
                    rec = uses[name]
                    rec["files"].add(os.path.relpath(path, root))
                    rec["guarded" if guarded else "unguarded"] += 1
    modules = []
    for name, rec in sorted(uses.items(), key=lambda kv: (-len(kv[1]["files"]), kv[0])):
        modules.append({
            "module": name, "declared_by": declared.get(name), "files": sorted(rec["files"]),
            "guarded_imports": rec["guarded"], "unguarded_imports": rec["unguarded"],
        })
    return {"scanned_python_files": scanned, "declared": declared, "modules": modules,
            "undeclared": [m["module"] for m in modules if not m["declared_by"]]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args(argv)
    report = scan(os.path.abspath(args.root))
    print(f"import_scan: {report['scanned_python_files']} files, "
          f"{len(report['modules'])} third-party modules, {len(report['undeclared'])} undeclared")
    for m in report["modules"]:
        print(f"  {m['module']:20s} files={len(m['files']):3d} unguarded={m['unguarded_imports']:3d} "
              f"guarded={m['guarded_imports']:3d}  {m['declared_by'] or 'UNDECLARED'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
