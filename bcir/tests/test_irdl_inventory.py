"""The ODS->IRDL inventory gate (S0-9) reconciles three sources and fires on every way they
can drift.

`tools/irdl/check_inventory.py` reads the dialect (ODS), the projection (IRDL) and the
manifest of deliberately unprojected operations, and refuses an undeclared gap, a stale
declaration, a ghost, an orphan and a naming collision. These tests prove the gate can fail
(laws.md L2) on a copy of the real tree with ONE fault injected per test, and that over the
real tree it is a PASS over a non-trivial inventory, never a vacuous one. No toolchain is
involved: the inputs are text, so the gate runs on every host and in the quick tier.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from tools.irdl.check_inventory import (  # noqa: E402
    INCLUDE_DIR,
    IRDL_FILE,
    MANIFEST,
    active_text,
    audit,
    main,
    projected_name,
)


def _copy_tree(tmp: Path) -> Path:
    """The three inputs, in their repository layout, so `audit(tmp)` reads a real tree."""
    (tmp / INCLUDE_DIR).mkdir(parents=True)
    for td in (_ROOT / INCLUDE_DIR).glob("*.td"):
        shutil.copy(td, tmp / INCLUDE_DIR / td.name)
    (tmp / IRDL_FILE).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(_ROOT / IRDL_FILE, tmp / IRDL_FILE)
    shutil.copy(_ROOT / MANIFEST, tmp / MANIFEST)
    return tmp


def _add_irdl_op(tmp: Path, name: str) -> None:
    path = tmp / IRDL_FILE
    text = path.read_text(encoding="utf-8")
    closing = text.rstrip().rfind("}")  # the dialect's closing brace
    path.write_text(
        text[:closing] + f"  irdl.operation @{name} {{\n  }}\n" + text[closing:], encoding="utf-8"
    )


def _manifest(tmp: Path) -> dict:
    return json.loads((tmp / MANIFEST).read_text(encoding="utf-8"))


def _write_manifest(tmp: Path, manifest: dict) -> None:
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _kinds(report: dict) -> set[tuple[str, str]]:
    return {(finding["kind"], finding["operation"]) for finding in report["findings"]}


def test_the_real_tree_reconciles_over_a_non_trivial_inventory() -> None:
    """PASS, and a PASS that examined something (L2): the dialect is large, most of it is
    projected under the naming rule, and the manifest accounts for exactly the rest."""
    report = audit(_ROOT)
    assert report["state"] == "PASS", report["findings"][:8]
    counts = report["counts"]
    assert counts["ods"] >= 100 and counts["irdl"] >= 50, counts
    assert counts["projected_exact"] >= 5 and counts["projected_renamed"] >= 50, counts
    assert (
        counts["unprojected_declared"]
        == counts["ods"] - counts["projected_exact"] - counts["projected_renamed"]
    )
    manifest = _manifest(_ROOT)
    assert manifest["naming_rule"] == report["naming_rule"] == "dots-to-underscores"
    assert all(reason.strip() for reason in manifest["unprojected"].values())


def test_the_naming_rule_is_the_projections_only_deviation() -> None:
    assert projected_name("gem.stream_pack") == "gem_stream_pack"
    assert projected_name("module") == "module"
    assert projected_name("atomic_rmw") == "atomic_rmw"


def test_an_undeclared_gap_is_a_finding() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        manifest = _manifest(tree)
        dropped = sorted(manifest["unprojected"])[0]
        del manifest["unprojected"][dropped]
        _write_manifest(tree, manifest)
        report = audit(tree)
        assert report["state"] == "FAIL"
        assert ("undeclared", dropped) in _kinds(report), report["findings"]


def test_a_stale_declaration_is_a_finding() -> None:
    """An operation the manifest calls unprojected while the projection defines it."""
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        stale = sorted(_manifest(tree)["unprojected"])[0]
        _add_irdl_op(tree, projected_name(stale))
        report = audit(tree)
        assert report["state"] == "FAIL"
        assert ("stale-unprojected", stale) in _kinds(report), report["findings"]


def test_an_orphan_projection_is_a_finding() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        _add_irdl_op(tree, "no_such_operation")
        report = audit(tree)
        assert report["state"] == "FAIL"
        assert ("orphan", "no_such_operation") in _kinds(report), report["findings"]


def test_a_ghost_manifest_entry_is_a_finding() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        manifest = _manifest(tree)
        manifest["unprojected"]["ghost.op"] = "never defined by ODS"
        _write_manifest(tree, manifest)
        report = audit(tree)
        assert report["state"] == "FAIL"
        assert ("ghost", "ghost.op") in _kinds(report), report["findings"]


def test_a_naming_collision_is_a_finding() -> None:
    """Two ODS operations that spell one IRDL name: the rule must stay injective."""
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        td = tree / INCLUDE_DIR / "BCIRGEMOps.td"
        td.write_text(
            td.read_text(encoding="utf-8")
            + '\ndef BCIR_CollideOp : BCIR_Op<"gem_stream_pack", []> {}\n',
            encoding="utf-8",
        )
        report = audit(tree)
        assert report["state"] == "FAIL"
        assert ("collision", "gem.stream_pack, gem_stream_pack") in _kinds(report), report[
            "findings"
        ]


def test_a_malformed_manifest_is_a_finding_not_a_traceback() -> None:
    """L1: every exit is a verdict. A manifest that does not parse, or whose entries carry
    no reason, is reported, never raised."""
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        (tree / MANIFEST).write_text("{not json", encoding="utf-8")
        report = audit(tree)
        assert report["state"] == "FAIL"
        assert any(f["kind"] == "manifest-unreadable" for f in report["findings"])
        manifest = _manifest(_ROOT)
        manifest["unprojected"] = {name: "" for name in manifest["unprojected"]}
        _write_manifest(tree, manifest)
        report = audit(tree)
        assert report["state"] == "FAIL"
        assert any(f["kind"] == "manifest-shape" for f in report["findings"])


def test_an_empty_inventory_is_vacuous_not_green() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        for td in (tree / INCLUDE_DIR).glob("*.td"):
            td.unlink()
        report = audit(tree)
        assert report["state"] == "INVALID/VACUOUS"
        assert any(f["kind"] == "empty-inventory" for f in report["findings"])
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["--root", str(tree)]) == 2


def test_the_cli_exit_code_is_the_verdict() -> None:
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert main(["--root", str(_ROOT)]) == 0
    assert out.getvalue().startswith("check_inventory: PASS")
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        _add_irdl_op(tree, "no_such_operation")
        report_path = tree / "report.json"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            assert main(["--root", str(tree), "--json-out", str(report_path)]) == 1
        assert "orphan" in out.getvalue()
        assert json.loads(report_path.read_text(encoding="utf-8"))["state"] == "FAIL"


def test_a_commented_out_declaration_is_not_a_declaration() -> None:
    """The gate reads the sources the way their compilers do: an `irdl.operation` disabled with
    `//` no longer projects its ODS operation (undeclared), and an ODS `def` disabled with a
    block comment leaves its projection an orphan. Both used to count, and the gate stayed PASS
    over a projection that defined one operation less (L2)."""
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        irdl = tree / IRDL_FILE
        text = irdl.read_text(encoding="utf-8")
        assert text.count("irdl.operation @gem_stream_pack") == 1
        irdl.write_text(
            text.replace("irdl.operation @gem_stream_pack", "// irdl.operation @gem_stream_pack"),
            encoding="utf-8",
        )
        report = audit(tree)
        assert report["state"] == "FAIL"
        assert ("undeclared", "gem.stream_pack") in _kinds(report), report["findings"]
    with tempfile.TemporaryDirectory() as tmp:
        tree = _copy_tree(Path(tmp))
        td = tree / INCLUDE_DIR / "BCIRGEMOps.td"
        text = td.read_text(encoding="utf-8")
        needle = 'BCIR_Op<"gem.stream_pack"'
        assert text.count(needle) == 1
        start = text.rfind("\ndef ", 0, text.index(needle)) + 1
        end = text.index("\n", text.index(needle))
        td.write_text(text[:start] + "/* " + text[start:end] + " */" + text[end:], encoding="utf-8")
        report = audit(tree)
        assert report["state"] == "FAIL"
        assert ("orphan", "gem_stream_pack") in _kinds(report), report["findings"]
    # the stripper itself: comments go, string literals and newlines stay, block comments nest.
    assert (
        active_text('a // x\nb /* c /* d */ e */ f "g // h /* i */" j')
        == 'a \nb  f "g // h /* i */" j'
    )


def test_a_manifest_whose_root_is_not_an_object_is_a_finding_not_a_traceback() -> None:
    """L1: `[]` and `null` are valid JSON; dereferencing them raised AttributeError in place of
    the verdict (no --json-out report, no exit code contract)."""
    for text in ("[]", "null", '"unprojected"', "3"):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _copy_tree(Path(tmp))
            (tree / MANIFEST).write_text(text + "\n", encoding="utf-8")
            report = audit(tree)
            assert report["state"] == "FAIL", text
            assert any(f["kind"] == "manifest-shape" for f in report["findings"]), text
            with contextlib.redirect_stdout(io.StringIO()):
                assert main(["--root", str(tree)]) == 1


#: Programs the projection must REFUSE: each breaks one constraint G32 projected (the address
#: is not an integer, a register value is not i64, an MSR index is not i32, a selector is not
#: i16, the CAS operands and result disagree, a matmul buffer is not a memref, port I/O has
#: three operands). (name, generic-syntax body)
_REFUSED = (
    ("volatile_load float address", '%r = "bcir.volatile_load"(%f) : (f32) -> i32'),
    ("creg_write i32", '"bcir.creg_write"(%i32) : (i32) -> ()'),
    ("msr_read i64 index", '%m = "bcir.msr_read"(%i64) : (i64) -> i64'),
    ("segment_reload i32", '"bcir.segment_reload"(%i32, %i32) : (i32, i32) -> ()'),
    ("atomic_cas mixed", '%c = "bcir.atomic_cas"(%i64, %i32, %i64) : (i64, i32, i64) -> i32'),
    ("matmul_buffer scalar", '"bcir.gem_matmul_buffer"(%f, %f, %f) : (f32, f32, f32) -> ()'),
    ("portio three operands", '"bcir.portio"(%i16, %i16, %i16) : (i16, i16, i16) -> ()'),
)


def test_the_projection_refuses_what_ods_refuses() -> None:
    """G32: the newly projected operations are constrained, not merely named -- stock mlir-opt
    with the projection refuses each mistyped program above and accepts its well-typed twin
    (the corpus). A skip without mlir-opt, never a pass."""
    import subprocess

    from bcir.toolchain import resolve_llvm_tools

    tools = resolve_llvm_tools("mlir-opt", pipeline="IRDL projection")
    if not tools.ok:
        return
    irdl = _ROOT / "mlir/irdl/bcir.irdl.mlir"
    with tempfile.TemporaryDirectory(prefix="bcir-irdl-neg-") as work:
        for name, body in _REFUSED:
            path = Path(work) / "neg.mlir"
            path.write_text(
                "func.func @f(%f: f32, %i16: i16, %i32: i32, %i64: i64) {\n"
                f"  {body}\n  return\n}}\n",
                encoding="utf-8",
            )
            run = subprocess.run(
                [tools.paths["mlir-opt"], f"--irdl-file={irdl}", str(path)],
                capture_output=True,
                text=True,
            )
            assert run.returncode != 0, f"the projection accepted {name}"
            assert "error:" in run.stderr, (name, run.stderr)
        for corpus in ("driver_core_generic", "gem_model_generic", "ecn_generic"):
            run = subprocess.run(
                [
                    tools.paths["mlir-opt"],
                    f"--irdl-file={irdl}",
                    str(_ROOT / f"mlir/test/irdl/{corpus}.mlir"),
                ],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, (corpus, run.stderr)
