"""CXX0: the oracle inventory -- every oracle module, its native twin or none, the gates that hold
each twin, and the hot paths the A/B audit times -- read out of the tree's own sources.

The 2026-10-06 audit (item 17) found the whole-oracle inventory never done. These witnesses hold
the committed inventory to the sources (`--check` fails on a stale file), pin the twins the tree
is known to have, show a C unit's comment is what makes it a twin (and that a truncated path, a
re-export-only package or a header a unit merely includes does not), and that the hot paths are
the audit's own rows.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "tools", "perf"))

import oracle_inventory as oi  # noqa: E402


def _manifest() -> dict:
    with open(os.path.join(_ROOT, "runtime", "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def test_the_committed_inventory_is_current():
    if not os.path.isfile(os.path.join(_ROOT, oi.OUT)):
        return  # an installed package ships no docs tree
    assert oi.main(["--check"]) == 0


def test_a_stale_inventory_fails_the_check():
    tmp = tempfile.mkdtemp(prefix="bcir-inventory-")
    original = oi.OUT
    try:
        # absolute: `_rel` joins it onto ROOT, which keeps an absolute path (and its drive) as it
        # is -- a path relative to ROOT does not exist when the temp directory is on another
        # drive, as on GitHub's Windows runners (C: against the D: workspace)
        oi.OUT = os.path.join(tmp, "inventory.md")
        assert oi.main(["--check"]) == 1  # missing
        assert oi.main([]) == 0 and oi.main(["--check"]) == 0
        with open(os.path.join(oi.ROOT, oi.OUT), "a", encoding="utf-8") as f:
            f.write("an edit by hand\n")
        assert oi.main(["--check"]) == 1
    finally:
        oi.OUT = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_known_twins():
    """The twins the tree is known to have, each read from its unit's own comment."""
    modules = oi.oracle_modules()
    twins = {unit: named for unit, (named, _whole) in oi.twins(_manifest(), modules).items()}
    assert {"bcir.kbcir.realize", "bcir.abi.planner_abi", "bcir.gem.streampack"} <= set(
        twins["bcir_kplan.c"]
    )
    assert "bcir.gem.handoff" in twins["bcir_handoff.c"]
    assert "bcir.gem.ring" in twins["bcir_ring.c"]
    assert "bcir.abi.streampack_abi" in twins["bcir_runtime.c"]
    # through the header-only interface the runtime implements (it names bcir_runtime.c) -- and
    # not through an includer the header does not name
    assert "bcir.abi.execution_plan_abi" in twins["bcir_runtime.c"]
    assert "bcir.abi.execution_plan_abi" not in twins.get("bcir_control_plane.c", [])
    # a path below bcir/ named without its prefix (cfront/diagnostics.py)
    assert twins["bcir_diag.c"] == ["bcir.frontends.cfront.diagnostics"]
    assert "bcir.verify" in twins["bcir_verify.c"]
    assert all(not m.startswith("bcir.tests") for named in twins.values() for m in named)


def test_a_reference_is_exact_or_it_is_nothing():
    """A comment makes a twin only by naming a module exactly: a truncated path names nothing
    (`bcir/gem/hydrate` is no module), a re-export-only package names nothing, an attribute is
    followed to the module that defines it, and a package named as a whole is reported as one."""
    modules = set(oi.oracle_modules())
    packages = {m for m in modules if oi._source(m).endswith("__init__.py")}
    named, whole = oi._references("The C twin of bcir/gem/hydrate.", modules, packages)
    assert named == set()
    named, _ = oi._references("see bcir.abi.encode", modules, packages)
    assert named == {"bcir.abi.streampack_abi"}  # bcir.abi re-exports streampack_abi.encode
    named, whole = oi._references("The C twin of bcir/frontends/cfront/.", modules, packages)
    assert named == set() and whole == {"bcir.frontends.cfront"}
    named, _ = oi._references(
        "bcir/kbcir/realize.py and bcir.gem.handoff.freeze_claims", modules, packages
    )
    assert named == {"bcir.kbcir.realize", "bcir.gem.handoff"}


def test_the_hot_paths_are_the_audits_rows():
    paths = dict(oi.hot_paths(oi.oracle_modules()))
    assert paths["plan"] == ["bcir.kbcir.realize.optimize"]
    assert paths["hydrate"] == ["bcir.gem.streampack.hydrate_pipelined"]
    assert "bcir.abi.streampack_abi.decode" in paths["decode"]
    # fixtures, classes and constants are not entry points
    assert all(not e.startswith("bcir.performance_audit") for es in paths.values() for e in es)
    assert all(e.rsplit(".", 1)[1][:1].islower() for es in paths.values() for e in es)


def test_every_twin_is_held_by_a_gate_or_said_not_to_be():
    """Each twin unit is built by a section or a fuzzer -- or the inventory says so with a dash,
    never silently: the rendered table carries the gates of every unit."""
    manifest = _manifest()
    held = oi.gates(manifest)
    assert "kplan" in held["bcir_kplan.c"]["sections"]
    assert "fuzz_kplan" in held["bcir_kplan.c"]["fuzzers"]
    text = oi.render()
    for unit in oi.twins(manifest, oi.oracle_modules()):
        assert f"| `{unit}` |" in text, unit
