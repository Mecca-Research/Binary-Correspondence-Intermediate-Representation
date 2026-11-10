"""CXX0: the oracle inventory -- every oracle module, its native twins or none, the gates that hold
each twin, and the hot paths the A/B audit times -- read out of the tree's own sources. PORT-0
(2026-11-10) reads the C++ seam and the MLIR law rail beside the C rail.

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


def test_the_cpp_and_mlir_twins():
    """PORT-0: a C++ seam unit and an MLIR source mirror an oracle module the way a C unit does,
    by naming it in their leading comment; the inventory reads all three rails."""
    modules = oi.oracle_modules()
    cpp = {unit: named for unit, (named, _whole) in oi.cpp_twins(_manifest(), modules).items()}
    assert "bcir.gem.handoff" in cpp["bcir_handoff.cpp"]
    if not oi.mlir_units():
        return  # an installed package carries no mlir/ tree
    mlir = {unit: named for unit, (named, _whole) in oi.mlir_twins(modules).items()}
    assert mlir["passes/BCIROverlapPass.cpp"] == ["bcir.gem.overlap"]
    assert "bcir.kbcir.cost" in mlir["passes/BCIRCostModel.h"]
    assert "bcir.gem.schedule" in mlir["passes/BCIRSchedule.h"]
    assert "bcir.gem.async_tokens" in mlir["passes/BCIRScheduleEftPass.cpp"]
    assert {"bcir.gem.cim", "bcir.gem.dvfs"} <= set(mlir["passes/BCIRCimDvfsPass.cpp"])
    assert "bcir.kbcir.allocator" in mlir["passes/BCIRAllocPoolPass.cpp"]
    assert "bcir.verify" in mlir["passes/BCIRVerifyPass.cpp"]
    # a dialect operation's spelling makes no twin: no source names the packages by path
    assert not any({"bcir.kbcir", "bcir.gem"} & set(named) for named in mlir.values())


def test_a_dialect_operation_is_no_module_on_the_law_rail():
    """The MLIR rail spells dialect operations that look like dotted module names: read with the
    dotted rule, `bcir.kbcir.select` would make every pass that mentions it a twin of the
    `bcir.kbcir` package. The law rail counts paths only."""
    modules = set(oi.oracle_modules())
    packages = {m for m in modules if oi._source(m).endswith("__init__.py")}
    text = "annotates bcir.kbcir.select and bcir.gem.lane_segment"
    named, _ = oi._references(text, modules, packages)
    assert named  # the hazard, under the C rail's rule
    assert oi._references(text, modules, packages, dotted=False) == (set(), set())
    named, _ = oi._references("the C++ port of bcir/gem/overlap.py", modules, packages, False)
    assert named == {"bcir.gem.overlap"}


def test_the_twins_that_did_not_name_their_module():
    """Two C units mirrored an oracle module without naming it, so the inventory listed the
    module as Python-only (the native port audit, 2026-11-09): BCAB's reader and selection, and
    the BCIRQ4T v1 kernel."""
    twins = {u: named for u, (named, _w) in oi.twins(_manifest(), oi.oracle_modules()).items()}
    assert "bcir.abi.artifact_bundle" in twins["bcir_artifact_bundle.c"]
    assert "bcir.kbcir.lowbit" in twins["bcir_q4_kernel.c"]


def test_a_hot_path_shows_its_twins_on_every_rail():
    """The A/B audit's scheduler rows read Python-only while only `runtime/c` was read; their
    modules are held to MLIR passes."""
    if not oi.mlir_units():
        return
    text = oi.render()
    row = next(line for line in text.splitlines() if line.startswith("| `sched_eft` |"))
    assert "`passes/BCIRSchedule.h`" in row
    row = next(line for line in text.splitlines() if line.startswith("| `plan` |"))
    assert "`bcir_kplan.c`" in row and "`passes/BCIRPlanPass.cpp`" in row
