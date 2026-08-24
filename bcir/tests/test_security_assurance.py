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
    planted = "ghp_" + ("ab" * 18)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "leak.py").write_text(f'token = "{planted}"\n', encoding="utf-8")
        (root / "blob.bin").write_bytes(b"\x00\x01\x02\xff" + b"AKIA" + b"IOSFODNN7EXAMPLE")
        subprocess.run(["git", "add", "leak.py", "blob.bin"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert report["text_files"] >= 1
        assert report["binary_files"] >= 1
        rules = {item["rule"] for item in report["findings"]}
        assert "github-token" in rules
        dumped = json.dumps(report["findings"])
        assert planted not in dumped
        assert "ghp_" not in dumped


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


def test_decoder_campaign_fails_when_a_canonical_seed_is_rejected() -> None:
    from tools.security.run_decoder_campaign import _probe
    import random
    row = _probe("streampack", lambda _data: (_ for _ in ()).throw(ValueError("nope")),
                 b"seed", random.Random(0), mutations=2)
    assert row["state"] == "FAIL"
    assert any(item["kind"] == "seed-rejected" for item in row["findings"])


def test_decoder_campaign_hits_required_python_surfaces() -> None:
    rows = run_python_campaign(mutations=8, seed=7)
    names = {row["surface"] for row in rows}
    assert names == set(REQUIRED_PYTHON)
    assert all(row["mutations"] == 8 for row in rows)
    assert all(row["state"] == "PASS" for row in rows), rows


def test_malformed_differential_runs_required_rails() -> None:
    report = run_differential(_ROOT)
    assert report["state"] == "PASS", report["disagreements"]
    names = {case["name"] for case in report["cases"]}
    assert "clean-vector-add" in names
    assert "python-duplicate-claim-id" in names
    assert "python-illegal-module" in names
    assert "truncated-mlir" in names
    assert report["malformed_rejected"] >= 3
    duplicate = next(case for case in report["cases"] if case["name"] == "python-duplicate-claim-id")
    assert duplicate["python"]["rejected"] is True
    truncated = next(case for case in report["cases"] if case["name"] == "truncated-mlir")
    assert truncated["mlir_text"]["rejected"] is True


def test_secret_scan_does_not_let_a_comment_hide_a_match() -> None:
    planted = "ghp_" + ("cd" * 18)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "leak.py").write_text(
            f'token = "{planted}"  # example fixture\n', encoding="utf-8",
        )
        subprocess.run(["git", "add", "leak.py"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "github-token" for item in report["findings"])


def test_secret_scan_fails_closed_on_an_unreadable_archive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "readme.md").write_text("ok\n", encoding="utf-8")
        (root / "broken.zip").write_bytes(b"PK\x03\x04not-a-zip")
        subprocess.run(["git", "add", "readme.md", "broken.zip"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


def test_dependency_parser_works_without_tomllib() -> None:
    from tools.security.audit_dependencies import _parse_pyproject_legacy
    parsed = _parse_pyproject_legacy((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert parsed["runtime"] == []
    assert parsed["build_system"] == ["setuptools>=83.0.0"]
    assert "dev" in parsed["optional"]


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
