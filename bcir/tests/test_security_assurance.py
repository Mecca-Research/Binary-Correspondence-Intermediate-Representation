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
        # The raw token must be absent from the ENTIRE serialized report, not
        # just the findings list — every field is a potential echo path.
        assert "ghp_" not in json.dumps(report)


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
    from unittest.mock import patch
    from tools.security import audit_dependencies as deps
    # The advisory engine is opportunistic; a host that happens to have
    # pip-audit installed must not turn this unit test into a live network
    # scan whose result decides the verdict.
    with patch.object(deps.shutil, "which", return_value=None):
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
    # Version-suffixed stock binaries are stock too; only a bcir-opt passes.
    with tempfile.TemporaryDirectory() as tmp:
        stock = Path(tmp) / "mlir-opt-22"
        stock.write_text("", encoding="utf-8")
        real = Path(tmp) / "bcir-opt"
        real.write_text("", encoding="utf-8")
        with patch("tools.security.run_malformed_differential.shutil.which", return_value=None):
            with patch.dict(os.environ, {"BCIR_OPT": str(stock)}, clear=False):
                assert find_bcir_opt(Path(tmp)) is None
            with patch.dict(os.environ, {"BCIR_OPT": str(real)}, clear=False):
                assert find_bcir_opt(Path(tmp)) == str(real)


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
    ~10 child processes and a deliberate 1s timeout, so it runs exactly once."""
    report = review_self_check()
    assert report["fail_closed"] is True
    assert report["cases"]["valid-json"] == "PASS"
    for case in ("missing-command", "unparseable", "empty-output",
                 "missing-executable", "timeout", "duplicate-keys",
                 "null-summary", "non-utf8", "depth-bomb",
                 "malformed-env-command"):
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


def test_decoder_that_accepts_everything_is_a_finding() -> None:
    """Removing all validation must make the campaign redder, not greener."""
    from tools.security.run_decoder_campaign import _probe
    import random

    def accept_all(_data: bytes) -> str:
        return "decoded"

    row = _probe("streampack", accept_all, b"seed", random.Random(0), mutations=2)
    assert row["state"] == "FAIL"
    assert any(item["kind"] == "invalid-accepted" for item in row["findings"])


def test_unseeded_c_fuzzing_is_recorded_as_unavailable() -> None:
    """The fuzz script exits 0 after 'SKIP BCIRQ8 seed corpus'; that run did
    not exercise the target and must not satisfy --require-c. The POSIX rail
    is simulated so the check exercises the same branch on native Windows,
    where run_c_campaign otherwise short-circuits before the mock."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign
    fake = SimpleNamespace(
        returncode=0,
        stdout=b"  SKIP BCIRQ8 seed corpus (could not build a seed artifact); fuzzing unseeded\n",
        stderr=b"",
    )
    posix_os = SimpleNamespace(name="posix", environ=dict(os.environ))
    with patch.object(campaign, "os", posix_os):
        with patch.object(campaign.shutil, "which", return_value="/usr/bin/tool"):
            with patch.object(campaign.subprocess, "run", return_value=fake):
                report = campaign.run_c_campaign(_ROOT, runs=1, seconds=1)
    assert report["state"] == "UNAVAILABLE/SKIPPED"
    assert "unseeded" in report["reason"]


def test_compiled_verifier_timeout_is_a_structured_failure() -> None:
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with patch.object(differential, "find_bcir_opt", return_value="/fake/bcir-opt"):
        with patch.object(
            differential.subprocess, "run",
            side_effect=differential.subprocess.TimeoutExpired(cmd="bcir-opt", timeout=20),
        ):
            result = differential._compiled_mlir("bcir.module @m { }", _ROOT)
    assert result["state"] == "FAIL"
    assert "timed out" in result["reason"]


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


def test_tracked_symlinks_are_recorded_not_followed() -> None:
    """A tracked symlink must never be dereferenced (its target may be any
    host file, or an unreadable procfs path that would crash the scan)."""
    if os.name == "nt":
        return  # creating symlinks needs privilege on Windows; POSIX covers it
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        secret_target = root / "outside.txt"
        secret_target.write_text('token = "ghp_' + ("ab" * 18) + '"\n', encoding="utf-8")
        (root / "link.py").symlink_to(secret_target)
        subprocess.run(["git", "add", "ok.py", "link.py"], cwd=root, check=True)
        report = scan_tree(root)
        assert "link.py" in report.get("symlinks", [])
        # The linked-to secret was NOT scanned through the link, and the scan
        # did not crash; only real file content decides the verdict.
        assert all(item["path"] != "link.py" for item in report["findings"])


def test_compressed_tar_aliases_are_inspected() -> None:
    import io
    import tarfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:bz2") as archive:
            link = tarfile.TarInfo("safe-name")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../escape"
            archive.addfile(link)
        (root / "bundle.tbz2").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "bundle.tbz2"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-path-traversal" for item in report["findings"])


def test_python_verifier_crash_is_a_structured_disagreement() -> None:
    """A crashing verify() is the exact failure this campaign diagnoses; it
    must become a recorded disagreement, never an escaping traceback."""
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with patch.object(differential, "find_bcir_opt", return_value=None):
        with patch("bcir.verify.verify", side_effect=RuntimeError("boom")):
            report = differential.run_differential(_ROOT)
    assert report["state"] == "FAIL"
    assert any("crashed" in item for item in report["disagreements"])


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


def test_boundary_audit_flags_args_keyword_string_commands() -> None:
    """`args=` is subprocess's public command parameter; a literal string
    there is the same policy violation as the first positional."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tools").mkdir()
        (root / "tools" / "kw.py").write_text(
            "import subprocess\nsubprocess.run(args='tool --flag')\n",
            encoding="utf-8",
        )
        report = audit_boundaries(root)
        rules = {item["rule"] for item in report["findings"]}
        assert "subprocess-string-command" in rules


def test_non_utf8_text_files_are_still_scanned() -> None:
    """A NUL-free Latin-1 source is text, not binary — an ASCII credential in
    it must not hide behind the binary policy."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        planted = "ghp_" + ("ab" * 18)
        (root / "legacy.py").write_bytes(
            b"# -*- coding: latin-1 -*-\n# caf\xe9\n"
            + f'token = "{planted}"\n'.encode("latin-1")
        )
        subprocess.run(["git", "add", "legacy.py"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "github-token" for item in report["findings"])


def test_tar_logical_size_cap_is_enforced() -> None:
    import io
    import tarfile
    from tools.security import scan_secrets as secrets
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as archive:
            info = tarfile.TarInfo("big.bin")
            payload = b"\x00" * 4096
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        (root / "bomb.tgz").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "bomb.tgz"], cwd=root, check=True)
        old = secrets.ARCHIVE_LOGICAL_CAP
        secrets.ARCHIVE_LOGICAL_CAP = 1024
        try:
            report = scan_tree(root)
        finally:
            secrets.ARCHIVE_LOGICAL_CAP = old
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


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
    # Basic-string escapes decode: a real-world environment marker with an
    # embedded quoted version must round-trip, not split at the escape.
    marker = '[project]\ndependencies = ["foo; python_version < \\"3.12\\""]\n'
    parsed = _fallback_parse(marker)
    assert parsed["runtime"] == ['foo; python_version < "3.12"']
    assert parsed["_unasserted"] is False
    # An escape outside the subset fails closed rather than misreads.
    exotic = '[project]\ndependencies = ["\\u0041"]\n'
    assert _fallback_parse(exotic)["_unasserted"] is True
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
    # An inline table can carry dependency metadata the subset never parses.
    inline = 'project = { dependencies = ["requests==2"] }\n'
    assert _fallback_parse(inline)["_unasserted"] is True
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


def test_corrupt_zip_symlink_payload_is_unreadable_not_a_crash() -> None:
    """A crafted symlink member whose deflate stream is garbage raises
    zlib.error out of the bounded target read; that is the archive-unreadable
    finding, never an escaping traceback."""
    import io
    import struct
    import zipfile as zf
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as archive:
            info = zf.ZipInfo("lnk")
            info.external_attr = 0o120777 << 16
            info.compress_type = zf.ZIP_DEFLATED
            archive.writestr(info, "target/path")
        raw = bytearray(buf.getvalue())
        header = raw.find(b"PK\x03\x04")
        name_len, extra_len = struct.unpack("<HH", raw[header + 26:header + 30])
        payload = header + 30 + name_len + extra_len
        raw[payload:payload + 4] = b"\xff\xff\xff\xff"
        (root / "links.zip").write_bytes(bytes(raw))
        subprocess.run(["git", "add", "links.zip"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


def test_corrupt_tar_xz_payload_is_unreadable_not_a_crash() -> None:
    """tarfile wraps gzip/bz2 corruption into ReadError but lets LZMAError
    escape; iterating a corrupt .tar.xz must fail closed as unreadable."""
    import io
    import tarfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:xz") as archive:
            payload = b"A" * 200000
            first = tarfile.TarInfo("a.txt")
            first.size = len(payload)
            archive.addfile(first, io.BytesIO(payload))
            second = tarfile.TarInfo("b.txt")
            second.size = 10
            archive.addfile(second, io.BytesIO(b"0123456789"))
        raw = bytearray(buf.getvalue())
        middle = len(raw) // 2
        raw[middle:middle + 8] = b"\xff" * 8
        (root / "bundle.tar.xz").write_bytes(bytes(raw))
        subprocess.run(["git", "add", "bundle.tar.xz"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


def test_missing_tracked_file_is_a_finding_not_a_silent_skip() -> None:
    """A tracked path absent from the worktree was not inspected; its indexed
    blob still ships in every clone, so the scan must not PASS around it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        gone = root / "gone.py"
        gone.write_text("y = 2\n", encoding="utf-8")
        subprocess.run(["git", "add", "ok.py", "gone.py"], cwd=root, check=True)
        gone.unlink()
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "file-missing" for item in report["findings"])


def test_suffix_binaries_are_recorded_without_reading() -> None:
    """Bounds are enforced at ingress: a suffix-classified binary (a model or
    dataset blob) is recorded from its name alone, never materialized."""
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"\x00" * 64)
        subprocess.run(["git", "add", "ok.py", "model.safetensors"], cwd=root, check=True)
        real_read = Path.read_bytes
        seen: list[str] = []

        def counting(self: Path) -> bytes:
            seen.append(self.name)
            return real_read(self)

        with patch.object(Path, "read_bytes", counting):
            report = scan_tree(root)
        assert report["state"] == "PASS", report["findings"]
        assert "model.safetensors" in report["binaries"]
        assert "model.safetensors" not in seen


def test_oversized_archive_is_a_finding_not_an_oom() -> None:
    """An archive larger than the inspection budget is refused at ingress —
    a failing finding — instead of being read into memory whole."""
    import io
    import zipfile as zf
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as archive:
            archive.writestr("member.txt", "0" * 2048)
        (root / "big.zip").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "ok.py", "big.zip"], cwd=root, check=True)
        with patch.object(secrets, "ARCHIVE_LOGICAL_CAP", 128):
            report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-oversized" for item in report["findings"])
        assert any(meta.get("status") == "oversized" for meta in report["archives"])


def test_gitleaks_is_invoked_redacted() -> None:
    """The stderr tail the report keeps must never carry raw secret values;
    redaction is part of the opt-in engine's contract, not a preference."""
    if os.name == "nt":
        return  # the gitleaks rail is POSIX-gated
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets
    seen: dict[str, list[str]] = {}

    def capture(cmd, **kwargs):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with patch.object(secrets.subprocess, "run", side_effect=capture):
        with patch("shutil.which", return_value="/usr/bin/gitleaks"):
            result = secrets._try_gitleaks(Path("."))
    assert result is not None and result["available"] is True
    assert "--redact" in seen["cmd"]


def test_depth_bomb_reviewer_output_is_a_structured_failure() -> None:
    """json.loads raises RecursionError on pathologically nested output; the
    reviewer contract turns it into the unparseable FAIL, never a traceback."""
    from tools.security.independent_review import run_reviewer
    with tempfile.TemporaryDirectory() as tmp:
        bomb = Path(tmp) / "bomb.py"
        bomb.write_text("print('[' * 200000)\n", encoding="utf-8")
        report = run_reviewer([sys.executable, str(bomb)], Path(tmp))
    assert report["state"] == "FAIL"
    assert "unparseable" in report["reason"]


def test_c_campaign_survives_non_utf8_fuzzer_output() -> None:
    """Sanitizer and fuzzer binaries write raw bytes; the campaign records
    them replace-decoded instead of dying on a strict decode."""
    if os.name == "nt":
        return  # the fixture fuzz script needs a real POSIX shell
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script_dir = root / "tools" / "c"
        script_dir.mkdir(parents=True)
        script = script_dir / "fuzz_streampack.sh"
        script.write_text("#!/bin/sh\nprintf 'raw \\377\\376 bytes'\nexit 1\n", encoding="utf-8")
        script.chmod(0o755)
        real_which = campaign.shutil.which

        def fake_which(name):
            return real_which(name) if name == "bash" else "/fake/clang"

        with patch.object(campaign.shutil, "which", side_effect=fake_which):
            report = campaign.run_c_campaign(root, runs=1, seconds=1)
    assert report["state"] == "FAIL"
    assert "raw" in report["stdout_tail"]


def test_compiled_verifier_output_is_never_strict_decoded() -> None:
    """bcir-opt's captured output is diagnostic bytes the differential never
    parses; a crashing build spraying non-UTF-8 must not raise mid-campaign."""
    if os.name == "nt":
        return  # the fixture verifier is a POSIX shell script
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake-bcir-opt"
        fake.write_text("#!/bin/sh\nprintf '\\377\\376 diag'\nexit 1\n", encoding="utf-8")
        fake.chmod(0o755)
        with patch.object(differential, "find_bcir_opt", return_value=str(fake)):
            result = differential._compiled_mlir("bcir.module @m { }", _ROOT)
    assert result["state"] == "PASS"
    assert result["rejected"] is True


def test_boundary_audit_skip_dirs_are_relative_to_the_checkout() -> None:
    """A checkout that lives under a directory named build (or dataset, or
    __pycache__) must not have every file skipped into a vacuous report."""
    with tempfile.TemporaryDirectory() as tmp:
        nest = Path(tmp) / "build" / "checkout"
        nest.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=nest, check=True)
        tools = nest / "tools"
        tools.mkdir()
        (tools / "x.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "tools/x.py"], cwd=nest, check=True)
        report = audit_boundaries(nest)
        assert report["scanned_files"] == 1
        assert report["state"] == "PASS", report.get("error")


def test_boundary_audit_records_symlinks_without_following() -> None:
    """A tracked symlink's blob is its target string, not Python source;
    following it would audit arbitrary host content or crash on procfs."""
    if os.name == "nt":
        return  # creating symlinks needs privilege on Windows; POSIX covers it
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tools = root / "tools"
        tools.mkdir()
        (tools / "real.py").write_text("x = 1\n", encoding="utf-8")
        target = root / "outside.py"
        target.write_text("import os\nos.system('boom')\n", encoding="utf-8")
        (tools / "alias.py").symlink_to(target)
        subprocess.run(["git", "add", "tools/real.py", "tools/alias.py"], cwd=root, check=True)
        report = audit_boundaries(root)
        assert "tools/alias.py" in report.get("symlinks", [])
        assert report["state"] == "PASS", report["findings"]


def test_boundary_audit_unreadable_file_is_a_finding() -> None:
    """A tracked file the auditor cannot read was not audited — that is a
    failing finding, never an escaping OSError."""
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tools = root / "tools"
        tools.mkdir()
        (tools / "x.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "tools/x.py"], cwd=root, check=True)
        with patch.object(Path, "read_bytes", side_effect=OSError("denied")):
            report = audit_boundaries(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "file-unreadable" for item in report["findings"])


def test_truncated_compressed_tar_is_unreadable_not_a_crash() -> None:
    """A truncated .tgz still passes is_tarfile's header sniff, then raises
    EOFError mid-iteration; that must be the unreadable finding."""
    import io
    import tarfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as archive:
            payload = b"A" * 200000
            info = tarfile.TarInfo("a.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        whole = buf.getvalue()
        (root / "cut.tgz").write_bytes(whole[: len(whole) // 2])
        subprocess.run(["git", "add", "cut.tgz"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


def test_prefixed_assignment_secrets_are_matched() -> None:
    """A word boundary before the keyword let every prefixed key
    (DB_PASSWORD, AWS_SECRET_ACCESS_KEY, SECRET_KEY) escape the assignment
    rule — the dominant real-world shape must be visible, while identifier
    substrings (tokenizer, passwords_file) must not fire."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        # Concatenation keeps the assembled key = "value" shapes out of this
        # tracked source, exactly like the other planted fixtures above.
        (root / "settings.py").write_text(
            'DB_PASSWORD = "' + "correct-horse-battery" + '"\n'
            'AWS_SECRET_ACCESS_KEY = "' + "d8f7a6s5kjh324kjh9x" + '"\n'
            'SECRET_KEY = "' + "django-insecure-9f8s7d6f5" + '"\n'
            'tokenizer = "hf-internal/llama-tokenizer"\n'
            'passwords_file = "/etc/passwd-list-x"\n',
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "settings.py"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        hits = [i for i in report["findings"] if i["rule"] == "assignment-secret"]
        assert len(hits) == 3, report["findings"]


def test_boundary_audit_flags_shell_helper_calls() -> None:
    """subprocess.getoutput/getstatusoutput ARE shell string-command
    execution; the declared scope covers them like os.system."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tools = root / "tools"
        tools.mkdir()
        (tools / "x.py").write_text(
            "import subprocess\nsubprocess.getoutput('ls -la')\n", encoding="utf-8",
        )
        subprocess.run(["git", "add", "tools/x.py"], cwd=root, check=True)
        report = audit_boundaries(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "subprocess-shell-helper" for item in report["findings"])


def test_malformed_pyproject_fails_structured_on_both_parser_paths() -> None:
    """tomllib's TOMLDecodeError must produce the same structured FAIL the
    3.10 fallback gives — a divergent contract between hosts is a hole."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text(
            "[project\ndependencies = [\n", encoding="utf-8",
        )
        report = audit_deps(root)
    assert report["state"] == "FAIL"
    assert report["inventory_asserted"] is False


def test_unquoted_assignment_secrets_are_matched() -> None:
    """.env-style credentials are unquoted; a bounded unquoted value (16+
    chars carrying a digit) is a finding, while an identifier RHS
    (AUTH_TOKEN = DEFAULT_AUTH_TOKEN) stays out of the rule."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        # Concatenation keeps the assembled shapes out of this tracked source.
        (root / "prod.env").write_text(
            "DB_PASSWORD=" + "hunter2hunter2hunter2" + "\n"
            "aws_secret_access_key = " + "abcdef1234567890abcd" + "\n"
            "AUTH_TOKEN = DEFAULT_AUTH_TOKEN\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "prod.env"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        hits = [i for i in report["findings"] if i["rule"] == "assignment-secret"]
        assert len(hits) == 2, report["findings"]


def test_fallback_parser_ignores_brackets_inside_strings() -> None:
    """A bracket inside a dependency string must not hold the array open —
    the 3.10 fallback would otherwise fail a file tomllib accepts."""
    from tools.security.audit_dependencies import _fallback_parse
    text = '[project]\ndependencies = ["pkg; implementation_name == \'[\'"]\n'
    parsed = _fallback_parse(text)
    assert parsed["_unasserted"] is False
    assert parsed["runtime"] == ["pkg; implementation_name == '['"]


def test_non_utf8_tracked_filename_is_a_finding_not_a_crash() -> None:
    """git ls-files -z hands back raw filename bytes; a non-UTF-8 name must
    not kill the scan with a UnicodeDecodeError before any verdict."""
    if os.name == "nt":
        return  # byte filenames need a POSIX filesystem
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        with open(os.path.join(os.fsencode(tmp), b"caf\xe9.txt"), "wb") as handle:
            handle.write(b"data\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "filename-not-utf8" for item in report["findings"])


def test_boundary_audit_missing_tracked_python_is_a_finding() -> None:
    """A tracked developer script absent from the worktree was not audited;
    rglob-based discovery must not silently omit it under PASS."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tools = root / "tools"
        tools.mkdir()
        (tools / "x.py").write_text("x = 1\n", encoding="utf-8")
        gone = tools / "gone.py"
        gone.write_text("y = 2\n", encoding="utf-8")
        subprocess.run(["git", "add", "tools/x.py", "tools/gone.py"], cwd=root, check=True)
        gone.unlink()
        report = audit_boundaries(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "file-missing" for item in report["findings"])


def test_boundary_audit_survives_non_utf8_tracked_names() -> None:
    """Tracked-file discovery must decode raw git bytes without raising even
    when some tracked filename is not UTF-8."""
    if os.name == "nt":
        return  # byte filenames need a POSIX filesystem
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tools = root / "tools"
        tools.mkdir()
        (tools / "x.py").write_text("x = 1\n", encoding="utf-8")
        with open(os.path.join(os.fsencode(tmp), b"caf\xe9.txt"), "wb") as handle:
            handle.write(b"data\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = audit_boundaries(root)
        assert report["state"] == "PASS", report["findings"]
        assert report["scanned_files"] == 1


def test_hanging_decoder_is_a_structured_finding() -> None:
    """A decoder that stops terminating is the exact regression the campaign
    hunts; it must become a finding, not a hung required job."""
    if os.name == "nt":
        return  # the SIGALRM watchdog is a POSIX rail
    import random
    import time
    from tools.security.run_decoder_campaign import _probe
    report = _probe(
        "sleepy", lambda blob: time.sleep(5), b"seed",
        random.Random(0), mutations=1, timeout=0.5,
    )
    assert report["state"] == "FAIL"
    assert any(f["kind"] == "decoder-hang" for f in report["findings"])


def test_campaign_io_failure_is_not_a_graceful_rejection() -> None:
    """A disk-full or permission error in the campaign's own temp-file plumbing
    is environmental, not a decode verdict — it must surface as a finding
    (RuntimeError, outside the graceful set), never count as rejection."""
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign
    with patch.object(
        campaign.tempfile, "TemporaryDirectory", side_effect=OSError("disk full"),
    ):
        try:
            campaign._decode_q8_bytes(b"\x00")
        except RuntimeError as exc:
            assert "campaign I/O failed" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


def test_placeholder_suppression_is_value_shaped_not_substring() -> None:
    """A real credential merely CONTAINING a placeholder-ish substring stays
    a finding; only values that ARE placeholders (a leading placeholder word,
    or long filler runs) are suppressed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "conf.py").write_text(
            'DB_PASSWORD = "' + "this_is_a_fake_but_real_password_123" + '"\n'
            'API_TOKEN = "' + "changeme" + "-please-now" + '"\n'
            'GH = "ghp_' + "abQ" * 8 + "0000" + 'Z"\n',
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "conf.py"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        rules = [i["rule"] for i in report["findings"]]
        assert rules.count("assignment-secret") == 1, report["findings"]
        assert "github-token" in rules


def test_zip_member_cap_fires_before_the_central_directory_is_read() -> None:
    """The declared EOCD size caps members BEFORE ZipFile materializes every
    ZipInfo — a many-entry ZIP is refused without the allocation."""
    import io
    import zipfile as zf
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as archive:
            for index in range(3):
                archive.writestr(f"m{index}.txt", "data")
        (root / "many.zip").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "ok.py", "many.zip"], cwd=root, check=True)

        def bomb(*args, **kwargs):
            raise AssertionError("ZipFile was constructed before the cap check")

        with patch.object(secrets, "ARCHIVE_MEMBER_CAP", 2):
            with patch.object(secrets.zipfile, "ZipFile", side_effect=bomb):
                report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


def test_differential_requires_the_compiled_rail_when_told() -> None:
    """--require-compiled makes a missing bcir-opt fatal (mirroring
    --require-c): the CI job that builds the binary must not pass without it."""
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with patch.object(differential, "find_bcir_opt", return_value=None):
        report = differential.run_differential(_ROOT, require_compiled=True)
    assert report["state"] == "FAIL"
    assert "bcir-opt" in report["error"]


def test_r1_python_construction_guard_is_paired_in_the_differential() -> None:
    """R1's Python enforcement is the add_resource construction guard; if it
    regresses to silently overwriting a duplicate RID, the differential must
    record a python-rail disagreement, not leave the case python-less."""
    from unittest.mock import patch
    from bcir.model.graph import Module
    from tools.security import run_malformed_differential as differential

    def overwrite(self, resource):
        self.resources[resource.rid] = resource
        return resource

    with patch.object(differential, "find_bcir_opt", return_value=None):
        with patch.object(Module, "add_resource", overwrite):
            report = differential.run_differential(_ROOT)
    assert report["state"] == "FAIL"
    assert any(
        "compiled-official-r1-duplicate-rid/python" in item
        for item in report["disagreements"]
    ), report["disagreements"]


def test_quoted_dotted_toml_keys_are_attributed() -> None:
    """tomllib resolves project."dependencies" to project.dependencies; the
    3.10 fallback must attribute the same key instead of silently asserting
    an empty inventory, and an unterminated key quote fails closed."""
    from tools.security.audit_dependencies import _fallback_parse
    parsed = _fallback_parse('project."dependencies" = ["requests==2"]\n')
    assert parsed["_unasserted"] is False
    assert parsed["runtime"] == ["requests==2"]
    broken = _fallback_parse('project."dependencies = ["requests==2"]\n')
    assert broken["_unasserted"] is True


def test_verbose_reviewer_output_is_bounded() -> None:
    """capture without a byte budget lets a flooding reviewer OOM the job
    before the timeout fires; the budget produces the structured FAIL."""
    from unittest.mock import patch
    from tools.security import independent_review as review
    with tempfile.TemporaryDirectory() as tmp:
        chatty = Path(tmp) / "chatty.py"
        chatty.write_text(
            "import sys\n"
            "block = b'x' * 65536\n"
            "for _ in range(64):\n"
            "    sys.stdout.buffer.write(block)\n",
            encoding="utf-8",
        )
        with patch.object(review, "REVIEW_OUTPUT_CAP", 1 << 16):
            report = review.run_reviewer([sys.executable, str(chatty)], Path(tmp))
    assert report["state"] == "FAIL"
    assert "exceeded" in report["reason"]


def test_whitespace_around_dotted_toml_keys_is_attributed() -> None:
    """TOML permits whitespace around dotted-key separators; the fallback
    must resolve `project . dependencies` as tomllib does, not skip it."""
    from tools.security.audit_dependencies import _fallback_parse
    parsed = _fallback_parse('project . dependencies = ["requests==2"]\n')
    assert parsed["_unasserted"] is False
    assert parsed["runtime"] == ["requests==2"]


def test_unquoted_alphabetic_secrets_are_matched() -> None:
    """An all-letter passphrase value (20+ lowercase, no separators) is not
    an identifier reference; it must match unquoted, while UPPER_SNAKE and
    snake_case RHS identifiers stay out."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "prod.env").write_text(
            "DB_PASSWORD=" + "correcthorsebatterystaple" + "\n"
            "AUTH_TOKEN = DEFAULT_AUTH_TOKEN\n"
            "SESSION_TOKEN = default_session_token\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "prod.env"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        hits = [i for i in report["findings"] if i["rule"] == "assignment-secret"]
        assert len(hits) == 1, report["findings"]


def test_tar_metadata_bomb_is_bounded_before_iteration() -> None:
    """PAX and GNU long-name headers materialize while the iterator advances,
    BEFORE any TarInfo is yielded — the per-member size loop cannot bound
    them. The whole decompressed stream is bounded up front instead."""
    import io
    import tarfile
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            for index in range(64):
                info = tarfile.TarInfo("d/" + "n" * 512 + f"/{index}")
                info.size = 0  # all bulk lives in metadata headers
                archive.addfile(info)
        (root / "meta.tar.gz").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "meta.tar.gz"], cwd=root, check=True)
        with patch.object(secrets, "ARCHIVE_LOGICAL_CAP", 4096):
            report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


def test_reviewer_descendants_cannot_hold_the_harness_open() -> None:
    """A reviewer child that inherits the output pipes and outlives its
    parent must not stall the bounded harness; the process session is put
    down and the drains released."""
    if os.name == "nt":
        return  # process-group teardown is a POSIX rail
    import time
    from unittest.mock import patch
    from tools.security import independent_review as review
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp) / "parent.py"
        parent.write_text(
            "import json, subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'])\n"
            "print(json.dumps({'passed': True, 'security_concerns': [], "
            "'logic_errors': [], 'summary': 'ok'}))\n",
            encoding="utf-8",
        )
        started = time.monotonic()
        with patch.object(review, "REVIEW_PIPE_GRACE", 1.0):
            report = review.run_reviewer([sys.executable, str(parent)], Path(tmp))
        elapsed = time.monotonic() - started
    assert report["state"] == "PASS", report
    assert elapsed < 4.0, elapsed


def test_duplicate_key_detection_is_linear() -> None:
    """One duplicate among tens of thousands of keys sits under the output
    cap; detection must be single-pass, not a per-key list scan that burns
    minutes of CPU after the process timeout already finished."""
    import time
    from tools.security.independent_review import parse_review
    body = ", ".join(f'"k{index}": 1' for index in range(60000))
    text = (
        '{"passed": true, "security_concerns": [], "logic_errors": [], '
        '"summary": "s", ' + body + ', "k0": 2}'
    )
    started = time.monotonic()
    try:
        parse_review(text)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected duplicate-key rejection")
    assert time.monotonic() - started < 5.0


def test_zip_magic_under_a_foreign_suffix_is_inspected() -> None:
    """An archive is what an extractor says it is, not what its suffix says:
    a ZIP named payload.dat still gets its members traversal-checked instead
    of hiding behind the binary policy via its NUL bytes."""
    import io
    import zipfile as zf
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as archive:
            archive.writestr("../../escape", "nope")
        (root / "payload.dat").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "payload.dat"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-path-traversal" for item in report["findings"])


def test_boundary_audit_fails_when_git_discovery_fails() -> None:
    """ls-files failing INSIDE a git checkout (corrupt metadata, missing git)
    is a discovery failure the audit must surface — never a silent downgrade
    to the fixture walk that no longer knows what is tracked."""
    from unittest.mock import patch
    from tools.security import audit_tool_boundaries as atb
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".git").mkdir()
        tools = root / "tools"
        tools.mkdir()
        (tools / "x.py").write_text("x = 1\n", encoding="utf-8")
        with patch.object(atb.subprocess, "run", side_effect=OSError("no git")):
            report = audit_boundaries(root)
    assert report["state"] == "FAIL"
    assert any(item["rule"] == "tracked-discovery-failed" for item in report["findings"])


def test_escaped_quoted_toml_keys_fail_closed() -> None:
    """A basic-string escape in a quoted key resolves for tomllib but sits
    outside the fallback subset; it must land in _unasserted, never vanish
    into an asserted empty inventory."""
    from tools.security.audit_dependencies import _fallback_parse
    parsed = _fallback_parse('project."depend\\u0065ncies" = ["requests==2"]\n')
    assert parsed["_unasserted"] is True


def test_hanging_python_verifier_is_a_structured_disagreement() -> None:
    """A verify() that stops terminating on a malformed witness must become
    a recorded timeout disagreement, not a hung required job — the compiled
    rail already carries its own 20s bound."""
    if os.name == "nt":
        return  # the SIGALRM watchdog is a POSIX rail
    import time
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with patch.object(differential, "find_bcir_opt", return_value=None):
        with patch.object(differential, "PYTHON_VERIFY_TIMEOUT", 0.5):
            with patch("bcir.verify.verify", side_effect=lambda module: time.sleep(5)):
                report = differential.run_differential(_ROOT)
    assert report["state"] == "FAIL"
    assert any("timed out" in item for item in report["disagreements"])


def test_nonstandard_json_constants_are_rejected() -> None:
    """NaN and Infinity are not JSON; the permissive json.loads default must
    not let them ride under the fail-closed parse contract."""
    from tools.security.independent_review import parse_review
    text = (
        '{"passed": true, "security_concerns": [], "logic_errors": [], '
        '"summary": "ok", "metric": NaN}'
    )
    try:
        parse_review(text)
    except ValueError:
        pass
    else:
        raise AssertionError("expected rejection of NaN")


def test_zip_magic_under_a_binary_suffix_is_inspected() -> None:
    """The no-read ingress shortcut probes a bounded 262 bytes of signature
    first: a ZIP named payload.pdf still has its members traversal-checked
    while true model blobs stay unmaterialized."""
    import io
    import zipfile as zf
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as archive:
            archive.writestr("../../escape", "nope")
        (root / "payload.pdf").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "payload.pdf"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "archive-path-traversal" for item in report["findings"])


def test_boundary_audit_flags_fstring_commands() -> None:
    """An f-string command is a string command; the JoinedStr expression
    shape must not slip past the literal-string check."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tools = root / "tools"
        tools.mkdir()
        (tools / "x.py").write_text(
            'import subprocess\nflag = "-l"\nsubprocess.run(f"ls {flag}")\n',
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "tools/x.py"], cwd=root, check=True)
        report = audit_boundaries(root)
        assert report["state"] == "FAIL"
        assert any(
            item["rule"] == "subprocess-string-command" for item in report["findings"]
        )


def test_non_string_toml_array_elements_fail_closed() -> None:
    """dependencies = [42] is valid TOML the tomllib path would flag as a
    mismatch; the fallback must not silently drop the element into an
    asserted empty inventory."""
    from tools.security.audit_dependencies import _fallback_parse
    parsed = _fallback_parse('[project]\ndependencies = [42]\n')
    assert parsed["_unasserted"] is True


def test_compressed_tar_magic_under_foreign_suffixes_is_inspected() -> None:
    """A gzip'd tar renamed to payload.dat (or hidden under a binary suffix
    like payload.pdf) must still have its members traversal-checked; the
    magic table covers the compressed-tar formats _archive_entries parses."""
    import io
    import tarfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as archive:
            link = tarfile.TarInfo("safe-name")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../escape"
            archive.addfile(link)
        (root / "payload.dat").write_bytes(buf.getvalue())
        (root / "payload.pdf").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "payload.dat", "payload.pdf"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        traversals = [
            item for item in report["findings"]
            if item["rule"] == "archive-path-traversal"
        ]
        assert len(traversals) == 2, report["findings"]


def test_toml_multiline_strings_fail_closed() -> None:
    """Triple-quoted TOML strings are outside the fallback subset; they must
    refuse into _unasserted, never misparse into empty-string delimiters."""
    from tools.security.audit_dependencies import _fallback_parse
    parsed = _fallback_parse('[project]\ndependencies = ["""requests==2"""]\n')
    assert parsed["_unasserted"] is True


def test_witness_rejected_for_the_wrong_law_is_a_disagreement() -> None:
    """A witness drifting into a different rejection (syntax rot, another
    law) must not keep the differential green; a rail's rejection has to
    name the law the case exists to test."""
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    real = differential.parse_mlir_text

    def wrong_reason(text):
        out = real(text)
        if out.get("reason") == "duplicate-rid":
            return dict(out, reason="unbalanced")
        return out

    with patch.object(differential, "find_bcir_opt", return_value=None):
        with patch.object(differential, "parse_mlir_text", side_effect=wrong_reason):
            report = differential.run_differential(_ROOT)
    assert report["state"] == "FAIL"
    assert any("reason" in item for item in report["disagreements"])


def test_prefixed_zip_under_foreign_suffixes_is_inspected() -> None:
    """zipfile accepts a ZIP with arbitrary bytes before its first local
    header (the self-extracting shape); the probes must be EOCD-aware, not
    offset-zero magic only — under foreign and binary suffixes alike."""
    import io
    import zipfile as zf
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as archive:
            archive.writestr("../../escape", "nope")
        blob = b"SFX\x00STUB" + buf.getvalue()
        (root / "payload.dat").write_bytes(blob)
        (root / "payload.bin").write_bytes(blob)
        subprocess.run(["git", "add", "payload.dat", "payload.bin"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        traversals = [
            item for item in report["findings"]
            if item["rule"] == "archive-path-traversal"
        ]
        assert len(traversals) == 2, report["findings"]


def test_oversized_text_candidate_is_refused_at_ingress() -> None:
    """Every materialized read is bounded: a pathological tracked file with
    no recognized suffix cannot OOM the scan out of its verdict."""
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (root / "huge.log").write_text("A" * 4096, encoding="utf-8")
        subprocess.run(["git", "add", "ok.py", "huge.log"], cwd=root, check=True)
        with patch.object(secrets, "ARCHIVE_LOGICAL_CAP", 1024):
            report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "file-oversized" for item in report["findings"])


def test_find_bcir_opt_searches_the_build_tree_layout() -> None:
    """cmake decides where bcir-opt lands (bin/, tools/...); discovery must
    mirror check_passes.sh's find over the build tree, not a fixed path."""
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nested = root / "build" / "mlir-build" / "tools" / "bcir-opt-target"
        nested.mkdir(parents=True)
        binary = nested / "bcir-opt"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
        with patch.dict(differential.os.environ, {"BCIR_OPT": ""}):
            with patch.object(differential.shutil, "which", return_value=None):
                found = differential.find_bcir_opt(root)
    assert found == str(binary)


def test_c_campaign_timeout_keeps_the_captured_tails() -> None:
    """On TimeoutExpired the captured output is the only evidence of which
    target hung; the structured FAIL must carry it."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign
    exc = campaign.subprocess.TimeoutExpired(
        cmd="fuzz", timeout=5, output=b"target seven \xff hung", stderr=b"asan \xfe tail",
    )
    posix_os = SimpleNamespace(name="posix", environ=dict(os.environ))
    with patch.object(campaign, "os", posix_os):
        with patch.object(campaign.shutil, "which", return_value="/usr/bin/tool"):
            with patch.object(campaign.subprocess, "run", side_effect=exc):
                report = campaign.run_c_campaign(_ROOT, runs=1, seconds=1)
    assert report["state"] == "FAIL"
    assert "hung" in report["stdout_tail"]
    assert "asan" in report["stderr_tail"]
