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
    """One self_check() run covers every contract case — the harness spawns
    ~9 child processes and a deliberate 1s timeout, so it runs exactly once."""
    report = review_self_check()
    assert report["fail_closed"] is True
    assert report["cases"]["valid-json"] == "PASS"
    for case in ("missing-command", "unparseable", "empty-output",
                 "missing-executable", "timeout", "duplicate-keys",
                 "null-summary", "non-utf8", "malformed-env-command"):
        assert report["cases"][case] == "FAIL", case
    assert report["state"] == "PASS", report["mismatches"]


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
            archive.writestr("C:\\drive-escape.txt", "nope")
        (root / "win.zip").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "win.zip"], cwd=root, check=True)
        report = scan_tree(root)
        traversals = [i for i in report["findings"] if i["rule"] == "archive-path-traversal"]
        assert len(traversals) == 2


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


def test_single_file_compression_is_binary_not_archive() -> None:
    """A legitimate tracked .gz or .7z has no inspection path; it must follow
    the binary policy instead of failing the scan as an unreadable archive."""
    import gzip
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (root / "notes.txt.gz").write_bytes(gzip.compress(b"plain payload"))
        (root / "blob.7z").write_bytes(b"7z\xbc\xaf\x27\x1c junk")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "PASS", report["findings"]
        assert report["archive_files"] == 0
        assert report["binary_files"] == 2
        # A real tar.gz archive is still inspected as an archive.
        import io
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as archive:
            info = tarfile.TarInfo("inner.txt")
            info.size = 2
            archive.addfile(info, io.BytesIO(b"ok"))
        (root / "bundle.tar.gz").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "bundle.tar.gz"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["archive_files"] == 1
        assert report["state"] == "PASS", report["findings"]
        # A gz-wrapped tar WITHOUT a .tar.gz name is still an archive an
        # extractor unpacks — content probing keeps its members inspected.
        evil = io.BytesIO()
        with tarfile.open(fileobj=evil, mode="w:gz") as archive:
            link = tarfile.TarInfo("safe-name")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../escape"
            archive.addfile(link)
        (root / "backup.gz").write_bytes(evil.getvalue())
        subprocess.run(["git", "add", "backup.gz"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-path-traversal" for item in report["findings"])


def test_tar_link_targets_are_checked_for_traversal() -> None:
    import io
    import tarfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as archive:
            link = tarfile.TarInfo("safe-name")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../escape"
            archive.addfile(link)
        (root / "links.tar").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "links.tar"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-path-traversal" for item in report["findings"])


def test_zip_symlink_targets_are_checked_for_traversal() -> None:
    import io
    import zipfile as zf
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as archive:
            info = zf.ZipInfo("safe-name")
            info.create_system = 3
            info.external_attr = (0o120777 << 16)
            archive.writestr(info, "../../escape")
            # Symlink mode bits without the Unix create_system byte are still
            # honored by permissive extractors — the target must be checked.
            other = zf.ZipInfo("other-name")
            other.create_system = 0
            other.external_attr = (0o120777 << 16)
            archive.writestr(other, "../../also-escapes")
        (root / "links.zip").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "links.zip"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        traversals = [i for i in report["findings"] if i["rule"] == "archive-path-traversal"]
        assert len(traversals) == 2


def test_gitleaks_nonzero_fails_the_scan() -> None:
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "ok.py"], cwd=root, check=True)
        fake = {"engine": "gitleaks", "returncode": 2, "stdout_bytes": 10,
                "stderr_tail": "", "available": True}
        with patch.object(secrets, "_try_gitleaks", return_value=fake):
            code = secrets.main(["--root", str(root), "--allow-gitleaks"])
        assert code == 1


def test_reviewer_command_keeps_its_own_flags() -> None:
    import io
    from contextlib import redirect_stdout
    from tools.security.independent_review import main as review_main
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        script = (
            "import argparse, json; p = argparse.ArgumentParser();"
            "p.add_argument('--format'); a = p.parse_args();"
            "print(json.dumps({'passed': True, 'security_concerns': [],"
            " 'logic_errors': [], 'summary': 'flags kept: ' + a.format}))"
        )
        with redirect_stdout(io.StringIO()):
            code = review_main([
                "--json-out", str(out),
                "--command", sys.executable, "-c", script, "--format", "json",
            ])
        assert code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["review"]["summary"] == "flags kept: json"


def test_reviewer_env_command_preserves_quoting() -> None:
    from tools.security.independent_review import env_command
    parsed = env_command('reviewer --note "two words"')
    assert parsed == ["reviewer", "--note", "two words"]
    failed = env_command('reviewer "unterminated')
    assert isinstance(failed, dict)
    assert failed["state"] == "FAIL"
    # Windows rules: backslashed executable paths survive, quotes are shed.
    windows = env_command('C:\\tools\\reviewer.exe --note "two words"', posix=False)
    assert windows == ["C:\\tools\\reviewer.exe", "--note", "two words"]


def test_decoder_campaign_rejects_vacuous_mutation_counts() -> None:
    from tools.security.run_decoder_campaign import run_campaign
    report = run_campaign(_ROOT, mutations=0, seed=1, fuzz_runs=1, fuzz_seconds=1)
    assert report["state"] == "INVALID/VACUOUS"
    report = run_campaign(_ROOT, mutations=-4, seed=1, fuzz_runs=1, fuzz_seconds=1)
    assert report["state"] == "INVALID/VACUOUS"
    # The C rail's iteration budget is bounded by the same rationale.
    report = run_campaign(_ROOT, mutations=1, seed=1, fuzz_runs=0, fuzz_seconds=1)
    assert report["state"] == "INVALID/VACUOUS"
    report = run_campaign(_ROOT, mutations=1, seed=1, fuzz_runs=1, fuzz_seconds=0)
    assert report["state"] == "INVALID/VACUOUS"


def test_decoder_campaign_require_c_fails_when_unavailable() -> None:
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign
    rows = [
        {"surface": name, "state": "PASS", "accepted": 1, "rejected": 1,
         "mutations": 1, "findings": []}
        for name in campaign.REQUIRED_PYTHON
    ]
    unavailable = {"state": "UNAVAILABLE/SKIPPED", "reason": "clang missing"}
    with patch.object(campaign, "run_python_campaign", return_value=rows):
        with patch.object(campaign, "run_c_campaign", return_value=unavailable):
            relaxed = campaign.run_campaign(_ROOT, 1, 1, 1, 1)
            required = campaign.run_campaign(_ROOT, 1, 1, 1, 1, require_c=True)
    assert relaxed["state"] == "PASS"
    assert required["state"] == "FAIL"


def test_boundary_audit_covers_tracked_claude_scripts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".claude").mkdir()
        (root / ".claude" / "helper.py").write_text(
            "import subprocess\nsubprocess.run('echo hi', shell=True)\n",
            encoding="utf-8",
        )
        report = audit_boundaries(root)
        rules = {item["rule"] for item in report["findings"]}
        assert "subprocess-shell-true" in rules


def test_boundary_audit_honors_encoding_declarations() -> None:
    """A valid PEP 263 latin-1 source must be audited, not falsely rejected —
    and an actually unparseable file still fails closed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tools").mkdir()
        (root / "tools" / "legacy.py").write_bytes(
            b"# -*- coding: latin-1 -*-\n# caf\xe9\nx = 1\n"
        )
        report = audit_boundaries(root)
        assert report["findings"] == []
        (root / "tools" / "junk.py").write_bytes(b"\x00\x01 not python")
        report = audit_boundaries(root)
        assert any(item["rule"] == "python-parse-error" for item in report["findings"])


def test_mlir_text_duplicate_checks_are_per_module() -> None:
    """IDs are unique per Module in the oracle; two independent modules
    reusing an ID must not be rejected by the text rail."""
    from tools.security.run_malformed_differential import parse_mlir_text
    two_modules = (
        "bcir.module @a { bcir.claim @c attributes { claim_id = 1 : i32 } { } }\n"
        "bcir.module @b { bcir.claim @c attributes { claim_id = 1 : i32 } { } }\n"
    )
    assert parse_mlir_text(two_modules)["rejected"] is False
    one_module = (
        "bcir.module @a {\n"
        "  bcir.claim @c attributes { claim_id = 1 : i32 } { }\n"
        "  bcir.claim @d attributes { claim_id = 1 : i32 } { }\n"
        "}\n"
    )
    assert parse_mlir_text(one_module)["reason"] == "duplicate-claim-id"


def test_dependency_audit_rejects_dynamic_dependencies() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["setuptools>=83.0.0"]\n'
            '[project]\nname = "x"\nversion = "0.0.1"\ndynamic = ["dependencies"]\n',
            encoding="utf-8",
        )
        report = audit_deps(root)
        assert report["state"] == "FAIL"
        assert report["inventory_asserted"] is False


def test_dependency_fallback_parser_matches_tomllib() -> None:
    import tools.security.audit_dependencies as deps
    if deps.tomllib is None:
        return  # the fallback IS the parser on this host
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    full = deps.parse_pyproject(_ROOT / "pyproject.toml")
    fallback = deps._fallback_parse(text)
    assert fallback["runtime"] == full["runtime"]
    assert fallback["build_system"] == full["build_system"]
    assert fallback["optional"] == full["optional"]
    assert fallback["dynamic"] == full["dynamic"]
    assert fallback["_unasserted"] is False


def test_dependency_fallback_parser_survives_hostile_toml_shapes() -> None:
    """Fixed-expectation fixtures runnable without tomllib — these exact
    shapes made an earlier fallback silently misread dependency metadata."""
    from tools.security.audit_dependencies import _fallback_parse
    commented = (
        '[project]  # metadata\n'
        'dependencies = ["a>=1"]  # uses "b" internally\n'
        'dynamic = ["dependencies"]\n'
    )
    parsed = _fallback_parse(commented)
    assert parsed["runtime"] == ["a>=1"]
    assert parsed["dynamic"] == ["dependencies"]
    assert parsed["_unasserted"] is False
    dotted = 'project.dependencies = ["requests"]\n'
    assert _fallback_parse(dotted)["runtime"] == ["requests"]
    apostrophe = (
        '[project]\n'
        'dependencies = [\n'
        '  "a>=1",  # don\'t pin\n'
        '  "b==2",\n'
        ']\n'
    )
    assert _fallback_parse(apostrophe)["runtime"] == ["a>=1", "b==2"]
    # A quoted key is one literal segment; its dots are not path separators.
    quoted_group = (
        '[project.optional-dependencies]\n'
        '"docs.build" = ["sphinx"]\n'
    )
    assert _fallback_parse(quoted_group)["optional"] == {"docs.build": ["sphinx"]}
    # An unquoted dotted key under optional-dependencies nests tables PEP 621
    # forbids for groups — the reader refuses rather than misattributes.
    nested_group = '[project.optional-dependencies]\ndocs.build = ["sphinx"]\n'
    assert _fallback_parse(nested_group)["_unasserted"] is True
    # A dependency-shaped key the reader cannot attribute must fail closed,
    # not vanish into an empty (matching) declared set.
    poetry = '[tool.poetry]\ndependencies = ["c"]\n'
    assert _fallback_parse(poetry)["_unasserted"] is True
    unclosed = '[project]\ndependencies = [\n  "a>=1",\n'
    assert _fallback_parse(unclosed)["_unasserted"] is True


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
