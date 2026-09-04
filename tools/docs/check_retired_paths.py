#!/usr/bin/env python3
"""Fail CI when an *active* doc points readers at a retired tree, so agents/readers never learn
removed paths. Retired trees were superseded (e.g. `runtime/llvm/`, the early LLVM-IR-schema
runtime -> now the MLIR dialect `mlir/include/BCIR/` + the C runtime `runtime/c/`).

Policy (the requested allowlist):
  * ALLOWED  -- a *historical* doc that explains the removal: it carries the marker
                `<!-- allow-retired-paths -->` near the top (a deprecation banner).
  * DISALLOWED -- a current quickref / active guide that references a retired path with no marker.

    python tools/docs/check_retired_paths.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SKIP_DIRS = (".git", "__pycache__", "node_modules", ".mypy_cache")

# Retired path prefixes (removed trees). Add to this list as trees are retired.
RETIRED = ("runtime/llvm/", "runtime/llvm`", "runtime/llvm)")
ALLOW_MARKER = "allow-retired-paths"


def _md_files() -> list[str]:
    out: list[str] = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in _SKIP_DIRS]
        out += [os.path.join(dp, f) for f in fn if f.endswith(".md")]
    return sorted(out)


def offenders() -> list[tuple[str, int, str]]:
    bad: list[tuple[str, int, str]] = []
    for md in _md_files():
        try:
            text = open(md, encoding="utf-8").read()
        except OSError:
            continue
        if ALLOW_MARKER in text:
            continue  # historical doc: explicitly allowed
        rel = os.path.relpath(md, ROOT)
        for i, line in enumerate(text.splitlines(), 1):
            if any(r in line for r in RETIRED):
                bad.append((rel, i, line.strip()[:100]))
    return bad


def main() -> int:
    bad = offenders()
    if bad:
        sys.stderr.write(
            f"[check_retired_paths] {len(bad)} active-doc reference(s) to a retired "
            f"tree (mark the doc historical with <!-- {ALLOW_MARKER} --> or remove "
            f"the reference):\n"
        )
        for f, ln, txt in bad:
            sys.stderr.write(f"  {f}:{ln}  {txt}\n")
        return 1
    sys.stderr.write("[check_retired_paths] no active-doc references to retired trees.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
