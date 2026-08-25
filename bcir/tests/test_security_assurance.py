"""Security-assurance rails: secrets, inventory, campaigns, boundaries, fail-closed review."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.security.audit_dependencies import audit as audit_deps
from tools.security.audit_tool_boundaries import audit_boundaries
from tools.security.independent_review import self_check as review_self_check
from tools.security.run_decoder_campaign import REQUIRED_PYTHON, run_python_campaign
from tools.security.run_malformed_differential import run_differential
from tools.security.scan_secrets import scan_tree


def test_secret_scan_fails_closed_on_a_planted_text_secret() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        planted = "ghp_" + ("ab" * 18)
        aws = "AKIA" + ("A" * 16)
        (root / "leak.py").write_text('token = "' + planted + '"\n', encoding="utf-8")
        (root / "blob.bin").write_bytes(b"\x00\x01\x02\xff" + aws.encode("ascii"))
        subprocess.run(["git", "add", "leak.py", "blob.bin"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert report["text_files"] >= 1
        assert report["binary_files"] >= 1
        rules = {item["rule"] for item in report["findings"]}
        assert "github-token" in rules
        assert all("ghp_" not in json.dumps(item) for item in report["findings"])


def test_secret_scan_records_archive_path_traversal_without_extracting() -> None:
    import io
    import zipfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("../escape.txt", "nope")
        (root / "payload.zip").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "payload.zip"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["archive_files"] == 1
        assert report["archives"][0]["extracted"] is False
        assert any(item["rule"] == "archive-path-traversal" for item in report["findings"])


def test_secret_scan_of_the_current_tree_is_non_vacuous() -> None:
    report = scan_tree(_ROOT)
    assert report["tracked_files"] > 100
    assert report["text_files"] > 100
    assert report["state"] in {"PASS", "FAIL"}
    assert report["state"] != "INVALID/VACUOUS"
    assert report["state"] == "PASS", report["findings"][:8]


def test_dependency_inventory_must_be_asserted_before_advisories() -> None:
    report = audit_deps(_ROOT)
    assert report["inventory_asserted"] is True
    assert report["expected_packages"] >= 1
    assert report["declared"]["runtime"] == []
    assert report["mismatches"] == []
    assert report["state"] == "PASS"


def test_dependency_inventory_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        expected = Path(tmp) / "expected.json"
        expected.write_text(json.dumps({
            "schema": "bcir-dependency-inventory.v1",
            "runtime": ["definitely-not-a-bcir-dep==1.0"],
            "build_system": ["setuptools>=83.0.0"],
            "optional": {},
        }), encoding="utf-8")
        report = audit_deps(_ROOT, expected)
        assert report["state"] == "FAIL"
        assert report["mismatches"]


def test_decoder_campaign_hits_required_python_surfaces() -> None:
    rows = run_python_campaign(mutations=8, seed=7)
    names = {row["surface"] for row in rows}
    assert names == set(REQUIRED_PYTHON)
    assert all(row["mutations"] == 8 for row in rows)
    assert all(row["state"] == "PASS" for row in rows), rows


def test_malformed_differential_runs_required_rails() -> None:
    from unittest.mock import patch
    with patch("tools.security.run_malformed_differential.find_bcir_opt", return_value=None):
        report = run_differential(_ROOT)
    assert report["state"] == "PASS", report.get("error") or report["disagreements"]
    names = {case["name"] for case in report["cases"]}
    assert "clean-vector-add" in names
    assert "oracle-r1.1-duplicate-claim" in names
    assert "compiled-official-r1-duplicate-rid" in names
    assert "paired-official-r5" in names
    r11 = next(case for case in report["cases"] if case["name"] == "oracle-r1.1-duplicate-claim")
    assert r11["python"]["rejected"] is True
    assert "R1.1" in r11["python"]["laws"]
    assert r11["mlir_compiled"]["state"] == "UNAVAILABLE/SKIPPED"
    r5 = next(case for case in report["cases"] if case["name"] == "paired-official-r5")
    assert r5["python"]["rejected"] is True
    assert "R5" in r5["python"]["laws"]
    assert "R6" not in r5["python"]["laws"]


def test_find_bcir_opt_never_returns_stock_mlir_opt() -> None:
    from unittest.mock import patch
    from tools.security.run_malformed_differential import find_bcir_opt
    with patch.dict(os.environ, {"BCIR_OPT": str(_ROOT / "mlir-opt")}, clear=False):
        with patch("tools.security.run_malformed_differential.shutil.which", return_value="mlir-opt"):
            assert find_bcir_opt(_ROOT) is None


def test_r5_witness_matches_official_compiled_fixture() -> None:
    from tools.security.run_malformed_differential import _r5_module
    from bcir.verify import verify
    laws = {diag.law for diag in verify(_r5_module())}
    assert "R5" in laws
    assert "R6" not in laws


def test_tool_boundaries_scan_is_non_vacuous() -> None:
    report = audit_boundaries(_ROOT)
    assert report["scanned_files"] > 50
    assert report["state"] == "PASS", report["findings"][:12]


def test_independent_review_is_fail_closed() -> None:
    report = review_self_check()
    assert report["fail_closed"] is True
    assert report["cases"]["missing-command"] == "FAIL"
    assert report["cases"]["unparseable"] == "FAIL"
    assert report["cases"]["empty-output"] == "FAIL"
    assert report["cases"]["valid-json"] == "PASS"
    assert report["state"] == "PASS"


def test_placeholder_on_the_line_does_not_hide_a_real_token() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        planted = "ghp_" + ("ab" * 18)
        (root / "leak.py").write_text(
            'token = "' + planted + '"  # example configuration\n', encoding="utf-8",
        )
        subprocess.run(["git", "add", "leak.py"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "github-token" for item in report["findings"])


def test_unreadable_archive_fails_the_scan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (root / "bad.zip").write_bytes(b"not-a-zip")
        subprocess.run(["git", "add", "ok.py", "bad.zip"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


def test_archive_scan_normalizes_windows_separators() -> None:
    import io
    import zipfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("..\\escape.txt", "nope")
        (root / "win.zip").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "win.zip"], cwd=root, check=True)
        report = scan_tree(root)
        assert any(item["rule"] == "archive-path-traversal" for item in report["findings"])


def test_archive_member_cap_is_enforced() -> None:
    import io
    import zipfile
    from tools.security import scan_secrets as secrets
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("a.txt", "1")
            archive.writestr("b.txt", "2")
        (root / "many.zip").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "many.zip"], cwd=root, check=True)
        old = secrets.ARCHIVE_MEMBER_CAP
        secrets.ARCHIVE_MEMBER_CAP = 1
        try:
            report = scan_tree(root)
        finally:
            secrets.ARCHIVE_MEMBER_CAP = old
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


def test_boundary_audit_ignores_unrelated_run_methods() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "bcir").mkdir()
        (root / "tools").mkdir()
        (root / "bcir" / "job.py").write_text(
            "class Job:\n"
            "    def run(self, cmd):\n"
            "        return cmd\n"
            "Job().run('safe')\n",
            encoding="utf-8",
        )
        report = audit_boundaries(root)
        assert report["findings"] == []


def test_decoder_seed_rejection_is_a_finding() -> None:
    from tools.security.run_decoder_campaign import _probe
    import random

    def reject(_data: bytes) -> None:
        raise ValueError("seed")

    row = _probe("streampack", reject, b"seed", random.Random(0), mutations=2)
    assert row["state"] == "FAIL"
    assert any(item["kind"] == "seed-rejected" for item in row["findings"])


def test_r5_text_check_is_per_claim() -> None:
    from tools.security.run_malformed_differential import parse_mlir_text
    legal = (
        "bcir.module @ok {\n"
        "  bcir.claim @a attributes { lane = #bcir.lane<a>, "
        "hazard = #bcir.hazard<atomic> } { }\n"
        "  bcir.claim @b attributes { lane = #bcir.lane<u>, "
        "hazard = #bcir.hazard<unique> } { }\n"
        "}\n"
    )
    assert parse_mlir_text(legal)["rejected"] is False
    isolated = (
        "bcir.module @r5 {\n"
        "  bcir.claim @c attributes { lane = #bcir.lane<a>, "
        "hazard = #bcir.hazard<unique> } { }\n"
        "}\n"
    )
    assert parse_mlir_text(isolated)["reason"] == "r5-atomic-unique"
