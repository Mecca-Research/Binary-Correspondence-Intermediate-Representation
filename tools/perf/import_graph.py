#!/usr/bin/env python3
"""Hot/cold import quarantine — the research organs must stay OFF the simple path.

The lowest-level performance finding (the audit's performance-audit section): the dominant
cost of the simplest BCIR use — plan a program, emit a kernel — was never the planning math
or the emitted loop, but the *fixed import overhead* of eagerly loading the Phase-13..26
learned/categorical research stack. The packages import those lazily (PEP 562); this tool keeps
them quarantined.

It measures the real *eager* import closure: each HOT entry point is run in a fresh interpreter
and we read which ``bcir.*`` modules ended up in ``sys.modules`` (lazy / function-level imports
never appear, which is exactly the property we want to hold). The COLD set is the opt-in
research organs + heavy capabilities; none may be reachable from a hot entry.

    python tools/perf/import_graph.py            # print the per-entry eager import graph
    python tools/perf/import_graph.py --check     # CI gate: fail if a cold organ is eager-loaded
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- the COLD set: opt-in research organs + heavy capabilities (never on the simple path) ---
_COLD_KBCIR = ("calibloop", "microbench", "accel", "portfolio", "moegate", "regret", "softdp",
               "egraph", "operad", "twotruth", "memory", "throttle", "mapping", "bayescal",
               "allocator", "sensing", "precision")
_COLD_LOWER = ("jit", "wasm", "specialist")
_COLD_GEM = ("concurrency", "overlap", "schedule", "async_tokens", "cim", "dvfs")
_COLD_TOP = ("silicon",)

COLD: frozenset[str] = frozenset(
    [f"bcir.kbcir.{m}" for m in _COLD_KBCIR]
    + [f"bcir.lower.{m}" for m in _COLD_LOWER]
    + [f"bcir.gem.{m}" for m in _COLD_GEM]
    + [f"bcir.{m}" for m in _COLD_TOP])

# --- the HOT entry points that must stay lean ---
HOT_ENTRIES: dict[str, str] = {
    "import bcir": "import bcir",
    "import bcir.kbcir": "import bcir.kbcir",
    "build_artifact('vector_add')": "from bcir.api import build_artifact; build_artifact('vector_add')",
    "python -m bcir.run vector_add":
        "import sys; sys.argv=['bcir.run','vector_add']; from bcir.run import main; main()",
}


def eager_bcir_modules(code: str) -> set[str]:
    """Run `code` in a fresh interpreter; return the ``bcir.*`` modules it eagerly imported."""
    src = code + "\nimport sys\nprint('\\n'.join(m for m in sys.modules if m.startswith('bcir')))"
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError(f"probe failed for {code!r}:\n{out.stderr}")
    return set(out.stdout.split())


def graph() -> dict[str, set[str]]:
    return {label: eager_bcir_modules(code) for label, code in HOT_ENTRIES.items()}


def leaks() -> dict[str, list[str]]:
    """HOT entry -> the COLD organs it eagerly (and wrongly) pulled in."""
    return {label: sorted(COLD & loaded) for label, loaded in graph().items()
            if COLD & loaded}


def main() -> int:
    check = "--check" in sys.argv
    bad = leaks()
    if not check:
        for label, loaded in graph().items():
            cold = sorted(COLD & loaded)
            print(f"\n{label}  ({len(loaded)} bcir modules, {len(cold)} cold):")
            for m in sorted(loaded):
                print(f"  {'COLD ' if m in COLD else '     '}{m}")
    if bad:
        sys.stderr.write("[import_graph] cold research organs leaked onto the simple path:\n")
        for label, cold in bad.items():
            sys.stderr.write(f"  {label} -> {cold}\n")
        return 1
    sys.stderr.write(f"[import_graph] hot/cold quarantine holds across "
                     f"{len(HOT_ENTRIES)} entry points ({len(COLD)} cold organs, none eager).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
