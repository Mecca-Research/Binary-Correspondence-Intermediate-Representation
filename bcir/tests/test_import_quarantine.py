"""Hot/cold import quarantine as a dependency-graph test (drives `tools/perf/import_graph.py`).

Complements `test_perf.py` (which pins specific simple-path entry points) and `test_hot_cold.py`
(which pins specific hot *files*) with the package-root cold guard and the full hot→cold
reachability graph: importing the core must never eagerly drag a Phase-13..26 research organ in.
"""

import os
import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.perf.import_graph import COLD, HOT_ENTRIES, eager_bcir_modules, graph, leaks


def test_import_bcir_root_is_lean_and_cold_free():
    loaded = eager_bcir_modules("import bcir")
    assert not (COLD & loaded), f"import bcir eagerly pulled cold organs: {sorted(COLD & loaded)}"
    # the package root is the model layer only — a tiny, stable closure.
    assert len(loaded) <= 8, f"import bcir closure grew unexpectedly: {sorted(loaded)}"


def test_quarantine_holds_across_every_hot_entry():
    bad = leaks()
    assert not bad, f"cold research organs leaked onto the simple path: {bad}"


def test_cold_manifest_is_nontrivial_and_resolves_to_real_modules():
    # The manifest can't silently empty out (which would make the gate vacuous)...
    assert len(COLD) >= 20
    # ...and every declared cold organ must name a real module. A cold organ may be a
    # PACKAGE as well as a single module (bcir.asn1 is the whole X.690 codec), so
    # resolve both spellings: `<name>.py` and `<name>/__init__.py`. Resolving only the
    # first would force a package to be quarantined submodule-by-submodule and would
    # silently reject the package's own name.
    for mod in COLD:
        base = os.path.join(_ROOT, *mod.split("."))
        candidates = (base + ".py", os.path.join(base, "__init__.py"))
        assert any(os.path.exists(path) for path in candidates), (
            f"cold manifest names a missing module: {mod} "
            f"(tried {', '.join(candidates)})")


def test_every_hot_entry_actually_imports_some_bcir():
    # guards against a probe silently importing nothing (e.g. a renamed entry point).
    g = graph()
    assert set(g) == set(HOT_ENTRIES)
    for label, loaded in g.items():
        assert loaded, f"hot entry {label!r} imported no bcir modules (probe broken?)"
