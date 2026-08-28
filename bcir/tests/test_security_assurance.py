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
    from bcir.abi.streampack_abi import AbiError
    import random

    def reject(_data: bytes) -> None:
        # The surface's DECLARED rejection type: rejecting the valid seed is
        # a finding whichever way the decoder signals it, but only the
        # declared type is a rejection at all (see _DECLARED_REJECTIONS).
        raise AbiError("seed")

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
    import io

    class FakeProc:
        pid = 4242
        returncode = 0
        stdout = io.BytesIO(
            b"  SKIP BCIRQ8 seed corpus (could not build a seed artifact); "
            b"fuzzing unseeded\n"
        )
        stderr = io.BytesIO(b"")

        def wait(self, timeout=None):
            return 0

    posix_os = SimpleNamespace(name="posix", environ=dict(os.environ))
    with patch.object(campaign, "os", posix_os):
        with patch.object(campaign.shutil, "which", return_value="/usr/bin/tool"):
            with patch.object(campaign.subprocess, "Popen", side_effect=lambda *a, **k: FakeProc()):
                report = campaign.run_c_campaign(_ROOT, runs=1, seconds=1)
    assert report["state"] == "UNAVAILABLE/SKIPPED"
    assert "unseeded" in report["reason"]


def test_compiled_verifier_timeout_is_a_structured_failure() -> None:
    if os.name == "nt":
        return  # the fixture verifier is a POSIX shell script
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with tempfile.TemporaryDirectory() as tmp:
        hang = Path(tmp) / "fake-bcir-opt"
        hang.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
        hang.chmod(0o755)
        with patch.object(differential, "find_bcir_opt", return_value=str(hang)):
            with patch.object(
                differential.subprocess.Popen, "wait", autospec=True,
                side_effect=[
                    differential.subprocess.TimeoutExpired(cmd="bcir-opt", timeout=20),
                    0,
                ],
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
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets
    seen: dict[str, list[str]] = {}

    def capture(cmd, **kwargs):
        seen["cmd"] = cmd
        return {
            "launched": True, "timed_out": False, "overflow": False,
            "pipes_held": False, "returncode": 0, "stdout": b"",
            "stderr": b"", "error": "",
        }

    with patch.object(secrets, "run_bounded", side_effect=capture):
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


def test_compiled_diagnostic_marker_survives_long_notes() -> None:
    """MLIR prints the error line FIRST, then a source quote and a 'see
    current operation' note dump that easily overflows a tail window; the
    law marker must be checked against a head-biased diagnostic capture."""
    if os.name == "nt":
        return  # the fixture verifier is a POSIX shell script
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake-bcir-opt"
        fake.write_text(
            "#!/bin/sh\n"
            "printf 'case.mlir:4:5: error: R1: duplicate RID 10\\n' >&2\n"
            "printf 'case.mlir:4:5: note: see current operation: %s\\n' "
            "\"$(printf 'x%.0s' $(seq 1 600))\" >&2\n"
            "exit 1\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        with patch.object(differential, "find_bcir_opt", return_value=str(fake)):
            result = differential._compiled_mlir("bcir.module @m { }", _ROOT)
    assert result["rejected"] is True
    assert "R1:" in result["diagnostic"]
    assert result["diagnostic"].startswith("case.mlir")


def test_tar_probe_never_parses_compressed_bytes() -> None:
    """is_tarfile advances into the first member and can materialize a PAX
    payload BEFORE any bound; tarfile must only ever see plain bytes that
    already passed the bounded decompression."""
    import io
    import tarfile as tf
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets
    real = tf.is_tarfile

    def guarded(target):
        if hasattr(target, "read"):
            blob = target.read()
            target.seek(0)
        else:
            with open(target, "rb") as handle:
                blob = handle.read()
        assert not blob.startswith(b"\x1f\x8b"), "is_tarfile saw compressed bytes"
        return real(io.BytesIO(blob))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        buf = io.BytesIO()
        with tf.open(fileobj=buf, mode="w:gz") as archive:
            info = tf.TarInfo("member.txt")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"data"))
        (root / "bundle.tar.gz").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "ok.py", "bundle.tar.gz"], cwd=root, check=True)
        with patch.object(secrets.tarfile, "is_tarfile", side_effect=guarded):
            report = scan_tree(root)
        assert report["state"] == "PASS", report["findings"]


def test_q8_read_io_failure_is_not_graceful() -> None:
    """The decoder's open/fstat/read OSError is environmental (its content
    rejections are all ValueError); it must become a campaign failure, not a
    graceful malformed-input rejection."""
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign
    with patch(
        "bcir.frontends.models.weights_io.read_q8_decoder",
        side_effect=OSError("EIO"),
    ):
        try:
            campaign._decode_q8_bytes(b"\x00")
        except RuntimeError as exc:
            assert "campaign I/O" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


def test_bom_marked_unicode_text_is_scanned() -> None:
    """UTF-16/32 text carries NUL bytes that the binary heuristic would eat;
    a BOM names the real encoding, and the secrets inside must be seen."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        payload = 'DB_PASSWORD="' + "correct-horse-battery-123" + '"\n'
        (root / "secrets.ps1").write_bytes(payload.encode("utf-16"))
        subprocess.run(["git", "add", "secrets.ps1"], cwd=root, check=True)
        report = scan_tree(root)
        assert report["state"] == "FAIL"
        assert any(item["rule"] == "assignment-secret" for item in report["findings"])


def test_c_campaign_runs_in_its_own_session() -> None:
    """The fuzz wrapper backgrounds per-target subshells; only a process-
    group kill can enforce the wall bound on the whole tree, so the wrapper
    must start in its own session."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign
    captured: dict[str, object] = {}

    import io

    class FakeProc:
        pid = 4242
        returncode = 0
        stdout = io.BytesIO(b"ok")
        stderr = io.BytesIO(b"")

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return FakeProc()

    posix_os = SimpleNamespace(name="posix", environ=dict(os.environ))
    with patch.object(campaign, "os", posix_os):
        with patch.object(campaign.shutil, "which", return_value="/usr/bin/tool"):
            with patch.object(campaign.subprocess, "Popen", side_effect=fake_popen):
                report = campaign.run_c_campaign(_ROOT, runs=1, seconds=1)
    assert report["state"] == "PASS"
    assert captured.get("start_new_session") is True


def test_verbose_compiled_verifier_is_bounded() -> None:
    """A regressed bcir-opt spraying diagnostics must hit a byte budget and
    become a structured failure, not accumulate unbounded for 20 seconds."""
    if os.name == "nt":
        return  # the fixture verifier is a POSIX shell script
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake-bcir-opt"
        fake.write_text(
            "#!/bin/sh\n"
            "i=0\n"
            "while [ $i -lt 64 ]; do printf 'x%.0s' $(seq 1 65536); i=$((i+1)); done\n"
            "exit 1\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        with patch.object(differential, "VERIFY_OUTPUT_CAP", 1 << 16):
            with patch.object(differential, "find_bcir_opt", return_value=str(fake)):
                result = differential._compiled_mlir("bcir.module @m { }", _ROOT)
    assert result["state"] == "FAIL"
    assert "exceeded" in result["reason"]


def test_oversized_python_source_is_a_finding() -> None:
    """ast.parse builds a tree far larger than the source; a tracked blob
    must be stat-gated into a finding, never an OOM of the audit."""
    from unittest.mock import patch
    from tools.security import audit_tool_boundaries as atb
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tools = root / "tools"
        tools.mkdir()
        (tools / "x.py").write_text("x = 1\n", encoding="utf-8")
        (tools / "big.py").write_text("y = 2\n" + "# pad\n" * 800, encoding="utf-8")
        subprocess.run(["git", "add", "tools/x.py", "tools/big.py"], cwd=root, check=True)
        with patch.object(atb, "SOURCE_SIZE_CAP", 1024):
            report = audit_boundaries(root)
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

    import io

    class FakeProc:
        pid = 4242
        returncode = -9
        calls = 0
        stdout = io.BytesIO(b"target seven \xff hung")
        stderr = io.BytesIO(b"asan \xfe tail")

        def wait(self, timeout=None):
            FakeProc.calls += 1
            if FakeProc.calls == 1:
                raise campaign.subprocess.TimeoutExpired(cmd="fuzz", timeout=5)
            return -9

        def kill(self):
            pass

    # The put-down lives in the shared predicate, so simulating POSIX means
    # patching ITS os too: otherwise a Windows host takes the tree-kill
    # branch and shells out mid-test, which is not what this pins.
    from tools.security import proc_bounds
    posix_os = SimpleNamespace(name="posix", environ=dict(os.environ))
    with patch.object(campaign, "os", posix_os):
        with patch.object(proc_bounds, "os", posix_os):
            with patch.object(campaign.shutil, "which", return_value="/usr/bin/tool"):
                with patch.object(campaign.subprocess, "Popen", side_effect=lambda *a, **k: FakeProc()):
                    report = campaign.run_c_campaign(_ROOT, runs=1, seconds=1)
    assert report["state"] == "FAIL"
    assert "hung" in report["stdout_tail"]
    assert "asan" in report["stderr_tail"]


def test_xz_dictionary_memory_is_bounded() -> None:
    """A kilobyte .tar.xz can declare a giant LZMA2 dictionary that the
    decoder allocates BEFORE the first output byte, so the emitted-bytes cap
    alone cannot bound peak memory; the decode must honor an explicit memory
    limit and fail closed exactly like an over-cap stream."""
    import io
    import tarfile as tf
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets
    buf = io.BytesIO()
    with tf.open(fileobj=buf, mode="w:xz") as archive:
        info = tf.TarInfo("member.txt")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"data"))
    blob = buf.getvalue()
    assert secrets._decompress_bounded(blob)  # a sane limit decodes fine
    with patch.object(secrets, "XZ_MEMORY_LIMIT", 1 << 15):
        assert secrets._tar_probe(blob) is True  # fail-closed, not "binary"
        try:
            secrets._decompress_bounded(blob)
        except ValueError as exc:
            assert "archive-logical-cap" in str(exc)
        else:
            raise AssertionError("expected fail-closed ValueError")


def test_flooding_c_campaign_is_bounded() -> None:
    """A concurrent fuzzer spraying output must hit a per-stream byte budget
    that puts the whole process group down, not accumulate in communicate()
    until the runner OOMs or the wall bound fires."""
    if os.name == "nt":
        return  # the fixture wrapper is a POSIX shell script
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = root / "tools" / "c" / "fuzz_streampack.sh"
        script.parent.mkdir(parents=True)
        script.write_text(
            "#!/bin/bash\n"
            "i=0\n"
            "while [ $i -lt 16 ]; do printf 'x%.0s' $(seq 1 65536); i=$((i+1)); done\n"
            "sleep 30\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        with patch.object(campaign, "CAMPAIGN_OUTPUT_CAP", 1 << 16):
            with patch.object(
                campaign.shutil, "which",
                side_effect=lambda name: "/usr/bin/" + name,
            ):
                report = campaign.run_c_campaign(root, runs=1, seconds=1)
    assert report["state"] == "FAIL"
    assert "exceeded" in report["reason"]


def test_stalled_pip_audit_is_bounded() -> None:
    """The advisory engine resolves against a network index and spawns pip
    children; a stall must expire into the advisory's fail-closed state,
    never hang the inventory gate."""
    from unittest.mock import patch
    from tools.security import audit_dependencies as deps

    def stalled(cmd, **kwargs):
        return {
            "launched": True, "timed_out": True, "overflow": False,
            "pipes_held": False, "returncode": -9, "stdout": b"",
            "stderr": b"", "error": "",
        }

    with patch.object(deps.shutil, "which", return_value="/usr/bin/pip-audit"):
        with patch.object(deps, "run_bounded", side_effect=stalled):
            report = audit_deps(_ROOT)
    assert report["state"] == "FAIL"
    assert report["advisory"]["state"] == "FAIL"
    assert "timed out" in report["advisory"]["error"]


def test_oversized_pyproject_is_unasserted() -> None:
    """tomllib and the fallback multiply metadata size in memory; an
    oversized pyproject must stat-gate into the fail-closed inventory
    verdict, never an OOM of the required audit."""
    from unittest.mock import patch
    from tools.security import audit_dependencies as deps
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pyproject.toml"
        path.write_text(
            '[project]\nname = "x"\n' + "#" * 4096 + "\n", encoding="utf-8",
        )
        parsed = deps.parse_pyproject(path)
        assert parsed.get("_unasserted", False) is False  # below cap: normal
        with patch.object(deps, "PYPROJECT_SIZE_CAP", 1024):
            gated = deps.parse_pyproject(path)
    assert gated["_unasserted"] is True

def test_temporary_aws_keys_are_findings() -> None:
    """ASIA-prefixed STS keys are real credentials in the same 20-character
    shape as AKIA; an AKIA-only rule passes a standard leak."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        text = "key = " + "ASIA" + "J9K2M4P6Q8R1S3T5" + "\n"
        (root / "creds.txt").write_text(text, encoding="utf-8")
        subprocess.run(["git", "add", "creds.txt"], cwd=root, check=True)
        report = scan_tree(root)
    assert report["state"] == "FAIL"
    assert any(item["rule"] == "aws-access-key-id" for item in report["findings"])


def test_colon_delimited_mappings_are_scanned() -> None:
    """YAML and JSON store credentials behind a colon, not an equals sign;
    template references (${{ ... }}) stay suppressed as pointers."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        payload = (
            "password" + ': "' + "correct-horse-battery-123" + '"\n'
            + "token" + ': "${{ secrets.GITHUB_TOKEN }}"\n'
            + '"api_key"' + ': "' + "correct-horse-battery-456" + '"\n'
        )
        (root / "config.yaml").write_text(payload, encoding="utf-8")
        subprocess.run(["git", "add", "config.yaml"], cwd=root, check=True)
        report = scan_tree(root)
    hits = [item for item in report["findings"] if item["rule"] == "assignment-secret"]
    assert report["state"] == "FAIL"
    assert sorted(item["line"] for item in hits) == [1, 3]


def test_symlink_target_text_is_scanned() -> None:
    """The committed blob of a tracked symlink IS its target string; a token
    embedded there must be a finding without the target being followed."""
    if os.name == "nt":
        return  # symlink creation is privilege-gated on Windows
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        os.symlink("creds/" + "ghp_" + "k9J2mQ4pR6sT1vX3zB5dF7hL0nCwEyGa", root / "link")
        subprocess.run(["git", "add", "link"], cwd=root, check=True)
        report = scan_tree(root)
    assert "link" in report["symlinks"]
    assert report["state"] == "FAIL"
    assert any(
        item["rule"] == "github-token" and item["path"] == "link"
        for item in report["findings"]
    )


def _v7_tar(name: bytes, payload: bytes) -> bytes:
    """A checksum-valid legacy V7 tar: no ustar magic at offset 257."""
    header = bytearray(512)
    header[0:len(name)] = name
    header[100:108] = b"0000644\x00"
    header[108:116] = b"0000000\x00"
    header[116:124] = b"0000000\x00"
    header[124:136] = ("%011o" % len(payload)).encode("ascii") + b"\x00"
    header[136:148] = b"00000000000\x00"
    header[148:156] = b"        "
    header[156] = 0x30
    header[148:156] = ("%06o" % sum(header)).encode("ascii") + b"\x00 "
    padded = payload + b"\x00" * ((512 - len(payload) % 512) % 512)
    return bytes(header) + padded + b"\x00" * 1024


def test_legacy_v7_tar_members_are_inspected() -> None:
    """A V7 tar under an unrecognized suffix has no ustar marker; only a
    header-checksum probe keeps its member names from hiding behind the NUL
    heuristic's binary policy (the archive contract is names and links)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (root / "payload.dat").write_bytes(_v7_tar(b"../escape", b"owned\n"))
        subprocess.run(["git", "add", "ok.py", "payload.dat"], cwd=root, check=True)
        report = scan_tree(root)
    assert report["archive_files"] >= 1
    assert report["state"] == "FAIL"
    assert any(
        item["rule"] == "archive-path-traversal" for item in report["findings"]
    )


def test_bounded_runner_expires_and_caps() -> None:
    """The shared runner's two put-down triggers: a stalled group expires at
    the wall bound; a flooding group dies at the byte budget."""
    if os.name == "nt":
        return  # the fixture commands are POSIX shell
    from tools.security.proc_bounds import run_bounded
    bash = "/bin/bash"
    stalled = run_bounded(
        [bash, "-c", "sleep 30"], timeout=0.5, cap=1 << 16,
    )
    assert stalled["timed_out"] is True
    flooded = run_bounded(
        [bash, "-c",
         "i=0; while [ $i -lt 16 ]; do printf 'x%.0s' $(seq 1 65536); i=$((i+1)); done; sleep 30"],
        timeout=30.0, cap=1 << 16,
    )
    assert flooded["overflow"] is True
    assert flooded["timed_out"] is False
    assert len(flooded["stdout"]) <= (1 << 16)


def test_stalled_gitleaks_is_bounded() -> None:
    """The opted-in engine must come back as a failing structured outcome
    when it stalls or floods, never hang the scan or exit clean."""
    if os.name == "nt":
        return  # the gitleaks rail is POSIX-gated
    from unittest.mock import patch
    from tools.security import scan_secrets as secrets

    def stalled(cmd, **kwargs):
        return {
            "launched": True, "timed_out": True, "overflow": False,
            "pipes_held": False, "returncode": 0, "stdout": b"",
            "stderr": b"", "error": "",
        }

    with patch.object(secrets, "run_bounded", side_effect=stalled):
        with patch("shutil.which", return_value="/usr/bin/gitleaks"):
            result = secrets._try_gitleaks(Path("."))
    assert result is not None
    assert "timed out" in result["error"]
    assert result["returncode"] != 0


def test_dependency_groups_fail_closed() -> None:
    """PEP 735 [dependency-groups] tables declare real dependencies the
    expected-inventory schema cannot express; the parser must surface them
    and the audit must refuse to assert around them."""
    from tools.security import audit_dependencies as deps
    text = (
        '[project]\nname = "x"\ndependencies = []\n'
        '[build-system]\nrequires = ["setuptools>=83.0.0"]\n'
        '[dependency-groups]\nqa = ["requests==2.32.5"]\n'
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text(text, encoding="utf-8")
        parsed = deps.parse_pyproject(root / "pyproject.toml")
        assert parsed["dependency_groups"] == ["qa"]
        expected = root / "expected.json"
        expected.write_text(json.dumps({
            "schema": "bcir-dependency-inventory.v1",
            "runtime": [], "build_system": ["setuptools>=83.0.0"], "optional": {},
        }), encoding="utf-8")
        report = audit_deps(root, expected)
    assert report["state"] == "FAIL"
    assert "dependency-groups" in report["error"]

def test_unquoted_passphrases_are_findings() -> None:
    """Separator-delimited lowercase passphrases are a standard credential
    shape; requiring a digit or one uninterrupted run passed them clean."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        payload = (
            "password" + ": " + "correct-horse-battery-staple" + "\n"
            + "password" + ": " + "your-password-here" + "\n"
        )
        (root / "config.yaml").write_text(payload, encoding="utf-8")
        subprocess.run(["git", "add", "config.yaml"], cwd=root, check=True)
        report = scan_tree(root)
    hits = [item for item in report["findings"] if item["rule"] == "assignment-secret"]
    assert report["state"] == "FAIL"
    assert [item["line"] for item in hits] == [1]  # the placeholder stays out


def test_uppercase_python_suffix_is_audited() -> None:
    """rglob('*.py') is case-literal on Linux; a tracked check.PY carrying a
    boundary violation must be discovered and flagged, not silently skipped
    while the same tree passes on case-folding hosts."""
    from tools.security.audit_tool_boundaries import audit_boundaries
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tools = root / "tools"
        tools.mkdir()
        (tools / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (tools / "check.PY").write_text(
            "import os\nos.system(" + '"ls"' + ")\n", encoding="utf-8",
        )
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = audit_boundaries(root)
    assert report["state"] == "FAIL"
    assert any(
        item["rule"] == "os.system-or-popen" and item["path"].endswith("check.PY")
        for item in report["findings"]
    )

def test_scalar_dependency_fields_are_unasserted() -> None:
    """Valid TOML can still carry a nonsense metadata shape; the parser
    must fail closed, never traceback (list(42)) and never silently shred
    a bare string into characters."""
    from tools.security import audit_dependencies as deps
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "pyproject.toml"
        for body in ('dependencies = 42\n', 'dependencies = "requests"\n'):
            path.write_text('[project]\nname = "x"\n' + body, encoding="utf-8")
            parsed = deps.parse_pyproject(path)
            assert parsed["_unasserted"] is True, body
            assert parsed["runtime"] == [], body


def test_reviewer_put_down_kills_the_tree_on_windows() -> None:
    """Windows has no session to kill, so the put-down must terminate the
    process TREE; a descendant holding the pipes would otherwise outlive
    every advertised bound."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.security import proc_bounds
    killed: dict[str, object] = {}

    class FakeProc:
        pid = 4242

        def kill(self):
            killed["direct"] = True

    def capture(cmd, **kwargs):
        killed["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with patch.object(proc_bounds, "os", SimpleNamespace(name="nt")):
        with patch.object(proc_bounds.subprocess, "run", side_effect=capture):
            proc_bounds.put_down_group(FakeProc())
    assert killed["cmd"][:3] == ["taskkill", "/F", "/T"]
    assert killed["cmd"][-1] == "4242"
    assert killed.get("direct") is True


def test_credential_shaped_filenames_are_findings() -> None:
    """Git commits tree-entry names like it commits blobs; a token spelled
    into a filename ships in every clone, and the finding must not echo the
    credential back into the report."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        leaked = "ghp_" + "k9J2mQ4pR6sT1vX3zB5dF7hL0nCwEyGa" + ".txt"
        (root / "keys").mkdir()
        (root / "keys" / leaked).write_text("benign\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
    hits = [item for item in report["findings"] if item["rule"] == "filename-secret"]
    assert report["state"] == "FAIL"
    assert len(hits) == 1
    assert hits[0]["path"] == "keys/<redacted>"
    assert "ghp_" not in json.dumps(report)

def test_put_down_never_raises_from_the_tree_terminator() -> None:
    """The put-down runs inside timeout and overflow paths that must still
    return a structured verdict; no failure of the terminator may escape and
    turn a bounded FAIL into a traceback."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.security import proc_bounds
    killed: dict[str, object] = {}

    class FakeProc:
        pid = 4242

        def kill(self):
            killed["direct"] = True

    def exploding(cmd, **kwargs):
        raise TypeError("a wrapped subprocess implementation")

    with patch.object(proc_bounds, "os", SimpleNamespace(name="nt")):
        with patch.object(proc_bounds.subprocess, "run", side_effect=exploding):
            proc_bounds.put_down_group(FakeProc())  # must not raise
    assert killed.get("direct") is True


def test_wrong_shaped_metadata_tables_are_unasserted() -> None:
    """A scalar or list where a metadata TABLE belongs raises on .get (or is
    coerced away by `or {}`); the parser must fail closed instead."""
    from tools.security import audit_dependencies as deps
    bodies = ('project = "x"\n', 'project = []\n', 'build-system = 42\n')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pyproject.toml"
        for body in bodies:
            path.write_text(body, encoding="utf-8")
            parsed = deps.parse_pyproject(path)
            assert parsed["_unasserted"] is True, body
            assert parsed["runtime"] == [], body


def test_credential_shaped_names_are_redacted_in_every_report_field() -> None:
    """A finding that redacts while the binaries/archives/symlinks lists keep
    the raw name still republishes the credential through --json-out."""
    if os.name == "nt":
        return  # symlink creation is privilege-gated on Windows
    token = "ghp_" + "k9J2mQ4pR6sT1vX3zB5dF7hL0nCwEyGa"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / (token + ".bin")).write_bytes(b"\x00\x01binary\x00")
        (root / (token + ".zip")).write_bytes(
            b"PK\x03\x04" + b"\x00" * 26 + b"PK\x05\x06" + b"\x00" * 18
        )
        os.symlink("elsewhere", root / (token + ".link"))
        (root / (token + ".txt")).write_text("benign\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
    assert report["state"] == "FAIL"
    assert len([f for f in report["findings"] if f["rule"] == "filename-secret"]) == 4
    # The whole serialized report, not just the findings, must be clean.
    assert token not in json.dumps(report)
    assert report["binaries"] == ["<redacted>"]
    assert report["symlinks"] == ["<redacted>"]

def test_secret_bearing_directories_are_redacted() -> None:
    """A credential can name a DIRECTORY; copying the parent through
    verbatim republishes it just as surely as the basename would."""
    token = "ghp_" + "k9J2mQ4pR6sT1vX3zB5dF7hL0nCwEyGa"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / token).mkdir()
        (root / token / "safe.txt").write_text("benign\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
    hits = [f for f in report["findings"] if f["rule"] == "filename-secret"]
    assert report["state"] == "FAIL" and len(hits) == 1
    assert hits[0]["path"] == "<redacted>/safe.txt"  # the safe half survives
    assert token not in json.dumps(report)


def test_schema_prose_is_not_a_secret() -> None:
    """The quoted branch is the only value shape with no structural
    requirement, so prose lands in it. A short whitespace-bearing value is
    a description; a passphrase-length one is still a credential."""
    from tools.security.scan_secrets import _scan_text
    prose = _scan_text("f", '"token_counts": "object or null",')
    assert prose == []
    for line in (
        "password" + ': "' + "correct-horse-battery-123" + '"',
        "password" + ' = "' + "correct horse battery staple" + '"',
    ):
        assert [h["rule"] for h in _scan_text("f", line)] == ["assignment-secret"], line


def test_compiled_verifier_descendants_are_a_structured_failure() -> None:
    """BCIR_OPT may name a wrapper; a helper holding the pipes after the
    direct process exits must be reported, not silently outlive the rail."""
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential

    def held(cmd, **kwargs):
        return {
            "launched": True, "timed_out": False, "overflow": False,
            "pipes_held": True, "returncode": 0, "stdout": b"",
            "stderr": b"", "error": "",
        }

    with patch.object(differential, "find_bcir_opt", return_value="/usr/bin/fake"):
        with patch.object(differential, "run_bounded", side_effect=held):
            result = differential._compiled_mlir("bcir.module @m { }", _ROOT)
    assert result["state"] == "FAIL"
    assert "descendants" in result["reason"]

def test_python_witness_paired_to_its_intended_law() -> None:
    """A witness that keeps rejecting under a DIFFERENT law leaves the rail
    green while the law it exists to test has regressed; the Python rail
    needs the same pairing the text and compiled rails carry."""
    from unittest.mock import patch
    from types import SimpleNamespace
    from tools.security import run_malformed_differential as differential
    real = differential._bounded_verify

    def wrong_law(probe):
        diags = real(probe)
        # Simulate the regression: still rejected, but by another law.
        return [SimpleNamespace(law="R99") for _ in diags] if diags else diags

    with patch.object(differential, "_bounded_verify", side_effect=wrong_law):
        report = differential.run_differential(_ROOT)
    assert report["state"] == "FAIL"
    assert any(
        "/python: rejected under laws" in item for item in report["disagreements"]
    ), report["disagreements"]


def test_zip_symlink_under_lzma_is_uninspectable() -> None:
    """LZMA builds its decompressor from member properties BEFORE any byte
    emerges, so a declared dictionary allocates ahead of the read cap; a
    symlink member under that method must fail closed, never be opened."""
    import io
    import zipfile as zf
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as archive:
            info = zf.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (0o120777 << 16)  # S_IFLNK
            info.compress_type = 14  # ZIP_LZMA
            archive.writestr(info, "target")
        (root / "bundle.zip").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
    assert report["state"] == "FAIL"
    assert any(item["rule"] == "archive-unreadable" for item in report["findings"])


def test_symlinked_pyproject_is_unasserted() -> None:
    """stat() follows links, so a symlink to /dev/zero reports size 0 and
    then reads without end; the metadata ingress must refuse every
    non-regular file rather than dereference it."""
    if os.name == "nt":
        return  # symlink creation is privilege-gated on Windows
    from tools.security import audit_dependencies as deps
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        link = root / "pyproject.toml"
        os.symlink("/dev/zero", link)
        parsed = deps.parse_pyproject(link)
        assert parsed["_unasserted"] is True
        missing = deps.parse_pyproject(root / "absent.toml")
        assert missing["_unasserted"] is True
        real = root / "real.toml"
        real.write_text('[project]\ndependencies = ["a>=1"]\n', encoding="utf-8")
        assert deps.parse_pyproject(real)["runtime"] == ["a>=1"]

def test_credential_in_non_utf8_filename_is_redacted() -> None:
    """Replacement decoding rewrites only the invalid byte, so an ASCII
    credential survives it intact; the non-UTF-8 branch must redact like
    the UTF-8 one instead of printing the token."""
    if os.name == "nt":
        return  # the raw byte cannot be written into an NT filename
    token = "ghp_" + "k9J2mQ4pR6sT1vX3zB5dF7hL0nCwEyGa"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        raw = (token + "\udcff.txt").encode("utf-8", "surrogateescape")
        (root / os.fsdecode(raw)).write_bytes(b"benign\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
    rules = {item["rule"] for item in report["findings"]}
    assert report["state"] == "FAIL"
    assert "filename-not-utf8" in rules and "filename-secret" in rules
    assert token not in json.dumps(report)


def test_nested_build_directories_are_still_audited() -> None:
    """A generated tree is a path PREFIX; as a component-wide name it
    excluded tracked developer scripts under any nested directory so
    called, and the missing-file reconciliation excluded them too."""
    from tools.security.audit_tool_boundaries import audit_boundaries
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        nested = root / "tools" / "build"
        nested.mkdir(parents=True)
        (root / "tools" / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (nested / "release.py").write_text(
            "import os\nos.system(" + '"ls"' + ")\n", encoding="utf-8",
        )
        generated = root / "build"
        generated.mkdir()
        (generated / "gen.py").write_text(
            "import os\nos.system(" + '"ls"' + ")\n", encoding="utf-8",
        )
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = audit_boundaries(root)
    flagged = {item["path"] for item in report["findings"]}
    assert report["state"] == "FAIL"
    assert "tools/build/release.py" in flagged   # nested: audited
    assert not any(p.startswith("build/") for p in flagged)  # root: generated


def test_debug_preset_build_is_discovered() -> None:
    """mlir/CMakePresets sends the debug preset to build/mlir-build-debug;
    discovery that searched only the release tree recorded the compiled rail
    unavailable beside a valid official build."""
    from unittest.mock import patch
    from tools.security import run_malformed_differential as differential
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        binary = root / "build" / "mlir-build-debug" / "bin" / "bcir-opt"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
        with patch.dict(differential.os.environ, {"BCIR_OPT": ""}):
            with patch.object(differential.shutil, "which", return_value=None):
                found = differential.find_bcir_opt(root)
    assert found == str(binary)


def test_configured_clang_is_honored() -> None:
    """The wrapper supports CLANG, so a host with only a versioned clang
    must run the campaign rather than skip it and fail --require-c."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign
    seen: dict[str, object] = {}

    def which(name):
        seen.setdefault("asked", []).append(name)  # type: ignore[union-attr]
        return "/usr/bin/" + Path(name).name if "clang" in name or name == "bash" else None

    import io

    class FakeProc:
        pid = 4242
        returncode = 0
        stdout = io.BytesIO(b"ok")
        stderr = io.BytesIO(b"")

        def wait(self, timeout=None):
            return 0

    # Popen is faked: this pins the PREFLIGHT's toolchain resolution, and a
    # real spawn of the resolved path would depend on the host (it does not
    # exist on Windows, where the launch guard would then report FAIL).
    posix_os = SimpleNamespace(
        name="posix", environ={**os.environ, "CLANG": "/opt/llvm/bin/clang-19"},
    )
    with patch.object(campaign, "os", posix_os):
        with patch.object(campaign.shutil, "which", side_effect=which):
            with patch.object(
                campaign.subprocess, "Popen", side_effect=lambda *a, **k: FakeProc()
            ):
                report = campaign.run_c_campaign(_ROOT, runs=1, seconds=1)
    assert "/opt/llvm/bin/clang-19" in seen["asked"]  # type: ignore[operator]
    assert report["state"] == "PASS", report


def test_campaign_launch_failure_is_structured() -> None:
    """`which` can succeed and the exec still fail — a dangling symlink, a
    non-executable wrapper, a path that moved. Every path out of this rail
    is a structured state, never a traceback in place of the verdict."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from tools.security import run_decoder_campaign as campaign

    def exploding(*args, **kwargs):
        raise FileNotFoundError(2, "The system cannot find the file specified")

    posix_os = SimpleNamespace(name="posix", environ=dict(os.environ))
    with patch.object(campaign, "os", posix_os):
        with patch.object(campaign.shutil, "which", return_value="/usr/bin/tool"):
            with patch.object(campaign.subprocess, "Popen", side_effect=exploding):
                report = campaign.run_c_campaign(_ROOT, runs=1, seconds=1)
    assert report["state"] == "FAIL"
    assert "failed to start" in report["reason"]


def test_implementation_errors_are_never_graceful() -> None:
    """A blanket exception tuple counted unchecked indexing as a graceful
    rejection, so a decoder could regress to raising IndexError on every
    mutation and still report PASS. Only the declared type counts."""
    import random
    from tools.security.run_decoder_campaign import _probe

    def index_error(_data: bytes) -> str:
        raise IndexError("unchecked indexing")

    row = _probe("streampack", index_error, b"seed", random.Random(0), mutations=3)
    assert row["state"] == "FAIL"
    kinds = {item["kind"] for item in row["findings"]}
    assert "ungraceful-seed" in kinds and "ungraceful" in kinds
    assert all(item.get("type") == "IndexError" for item in row["findings"])

def test_decoder_watchdog_cannot_be_swallowed() -> None:
    """Decoders wrap broad `except Exception` around inner decodes; an
    Exception-derived watchdog is caught there and re-raised as the
    surface's declared rejection, so the stalled mutation reads as
    graceful and the campaign reports PASS."""
    import random
    from tools.security.run_decoder_campaign import _DecodeHang, _probe
    from bcir.abi.streampack_abi import AbiError
    assert not issubclass(_DecodeHang, Exception)

    def swallowing(_data: bytes):
        # Exactly the shape of artifact_bundle's inner validation.
        try:
            raise _DecodeHang("decode exceeded 10s")
        except Exception as exc:  # noqa: BLE001
            raise AbiError("wrapped") from exc

    row = _probe("streampack", swallowing, b"seed", random.Random(0), mutations=2)
    assert row["state"] == "FAIL"
    assert any(item["kind"] == "decoder-hang" for item in row["findings"])


def test_bomless_utf16_text_is_scanned() -> None:
    """BOM-less UTF-16 carries interleaved NULs; the generic heuristic filed
    it as binary and never looked inside."""
    for encoding in ("utf-16-le", "utf-16-be"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            payload = "password" + ": " + "correct-horse-battery-staple" + "\n"
            (root / "config.yaml").write_bytes(payload.encode(encoding))
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
            report = scan_tree(root)
        assert report["state"] == "FAIL", encoding
        assert report["text_files"] == 1, encoding
        assert any(
            item["rule"] == "assignment-secret" for item in report["findings"]
        ), encoding


def test_archive_member_names_are_scanned_for_secrets() -> None:
    """A member name or link target is committed metadata like a tree entry;
    the traversal predicate read those strings but the secret rules did not."""
    import io
    import zipfile as zf
    token = "ghp_" + "k9J2mQ4pR6sT1vX3zB5dF7hL0nCwEyGa"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        buf = io.BytesIO()
        with zf.ZipFile(buf, "w") as archive:
            archive.writestr(token + ".txt", "benign\n")
        (root / "bundle.zip").write_bytes(buf.getvalue())
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
    assert report["state"] == "FAIL"
    assert any(
        item["rule"] == "archive-metadata-secret" for item in report["findings"]
    )
    assert token not in json.dumps(report)  # the member name is never echoed


def test_yaml_block_scalar_secrets_are_findings() -> None:
    """`password: |-` puts the key and its value on separate lines; neither
    is credential-shaped alone."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        body = (
            "password" + ": |-\n  " + "correct-horse-battery-staple" + "\n"
            + "note" + ": |-\n  " + "your-password-here" + "\n"
        )
        (root / "config.yaml").write_text(body, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
    hits = [i for i in report["findings"] if i["rule"] == "assignment-secret"]
    assert report["state"] == "FAIL"
    assert [i["line"] for i in hits] == [1]  # the placeholder stays suppressed

def test_non_utf8_archive_member_names_are_findings_not_crashes() -> None:
    """tarfile exposes a member's invalid byte as a surrogate; a strict
    fingerprint encode then tracebacks the required scan in place of its
    archive-path-traversal finding."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        (root / "bundle.tar").write_bytes(_v7_tar(b"../bad\xff", b"owned\n"))
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = scan_tree(root)
    assert report["state"] == "FAIL"
    assert any(
        item["rule"] == "archive-path-traversal" for item in report["findings"]
    )


def test_boundary_findings_survive_strict_stdout() -> None:
    """A finding on a non-UTF-8 path must be printable under a strict
    stdout, exactly like the missing-file branch's printable form — the
    audit's report is its deliverable, not a traceback trigger."""
    if os.name == "nt":
        return  # the raw byte cannot be written into an NT filename
    from tools.security.audit_tool_boundaries import audit_boundaries
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        tools = root / "tools"
        tools.mkdir()
        raw = ("bad\udcff.py").encode("utf-8", "surrogateescape")
        (tools / os.fsdecode(raw)).write_bytes(b"import os\nos.system('ls')\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        report = audit_boundaries(root)
    assert report["state"] == "FAIL"
    flagged = [f for f in report["findings"] if f["rule"] == "os.system-or-popen"]
    assert flagged
    for finding in flagged:
        finding["path"].encode("utf-8")  # strict: printable everywhere


def test_json_escaped_credential_keys_are_findings() -> None:
    """JSON resolves ASCII \\uXXXX escapes at runtime, so an escaped key
    names the same credential; the raw-line rule never saw it."""
    from tools.security.scan_secrets import _scan_text
    line = (
        '{"passw' + "\\u006f" + 'rd": "' + "correct-horse-battery-staple" + '"}'
    )
    hits = _scan_text("f", line)
    assert [h["rule"] for h in hits] == ["assignment-secret"]
    benign = '{"c\\u006flor": "blue-green-teal-cyan"}'
    assert _scan_text("f", benign) == []


def test_toml_multiline_string_secrets_are_findings() -> None:
    """A TOML triple-quoted opener puts the key and its value on separate
    lines, exactly like a YAML block scalar; neither line is
    credential-shaped alone."""
    from tools.security.scan_secrets import _scan_text
    basic = '"' * 3
    literal = "'" * 3
    body = (
        "password" + " = " + basic + "\n"
        + "correct-horse-battery-staple" + "\n" + basic + "\n"
        + "token" + " = " + literal + "\n"
        + "example" + "\n" + literal + "\n"
    )
    hits = _scan_text("config.toml", body)
    assert [h["line"] for h in hits] == [1]  # the placeholder stays suppressed


def test_bomless_utf16_with_cjk_preamble_is_scanned() -> None:
    """A CJK preamble carries no NUL in either byte of its code units, so
    the sample's parity vote is silent — but the ASCII credential further
    down still interleaves NULs the whole-file vote can see."""
    from tools.security.scan_secrets import _decode_text, _scan_text, _utf16_bomless
    text = (
        "汉" * 5000 + "\n"
        + "password" + ": " + "correct-horse-battery-staple" + "\n"
    )
    blob = text.encode("utf-16-le")
    assert _utf16_bomless(blob) == "utf-16-le"
    hits = _scan_text("notes.txt", _decode_text(blob))
    assert [h["rule"] for h in hits] == ["assignment-secret"]
    # An even-length plain-ASCII file must NOT start decoding as UTF-16
    # garbage: valid UTF-8 with no NUL signal is single-byte text.
    assert _utf16_bomless(b"ab" * 4096) is None


def test_staged_secrets_are_scanned() -> None:
    """The index is what the next commit records: a secret staged and then
    overwritten with a benign worktree copy must still be a finding."""
    token = "ghp_" + ("cd" * 18)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        target = root / "config.py"
        target.write_text('token = "' + token + '"\n', encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        target.write_text("token = None\n", encoding="utf-8")
        report = scan_tree(root)
    assert report["state"] == "FAIL"
    staged = [i for i in report["findings"] if i["path"].endswith("(staged)")]
    assert "github-token" in {i["rule"] for i in staged}
    assert token not in json.dumps(report)  # fingerprints only, ever


def test_seed_construction_failure_is_a_structured_campaign_verdict() -> None:
    """An encoder or I/O failure while BUILDING a seed is the campaign
    failing to run — a structured surface FAIL, never a traceback that
    also silences the C rail behind it."""
    from tools.security import run_decoder_campaign as campaign

    def broken() -> bytes:
        raise OSError("disk full")

    original = campaign._q8_seed
    campaign._q8_seed = broken
    try:
        results = campaign.run_python_campaign(2, 20260828)
    finally:
        campaign._q8_seed = original
    by_name = {item["surface"]: item for item in results}
    assert set(by_name) == {"streampack", "bcab", "bcirq8"}
    assert by_name["bcirq8"]["state"] == "FAIL"
    assert [f["kind"] for f in by_name["bcirq8"]["findings"]] == [
        "seed-construction-failed"
    ]
    assert by_name["streampack"]["state"] == "PASS"


def test_assignment_matcher_is_linear_time() -> None:
    """A keyword-free separator run made the greedy prefix group retry from
    every word boundary — tens of kilobytes stalled the required scan
    quadratically. The wall bound is generous: the linear rule finishes
    ~1000x inside it, the quadratic one was half a minute outside."""
    import time
    from tools.security.scan_secrets import RULES
    rule = dict(RULES)["assignment-secret"]
    lines = ["a-" * 20000 + "b", ("password" + "_") * 8000 + "x"]
    start = time.perf_counter()
    for line in lines:
        assert rule.search(line) is None
    assert time.perf_counter() - start < 2.0
    # The rewrite keeps the rule's exact reach: separator-joined key
    # shapes still match, substring identifiers still do not.
    assert rule.search("DB_" + "PASSWORD" + ' = "' + "hunter2hunter2007" + '"')
    assert rule.search("client_" + "secret" + ": " + "abc123def456ghi789")
    assert not rule.search("secretariat" + ' = "' + "abcdefghijklmno" + '"')
    assert not rule.search("passwords_file" + ' = "' + "abcdefghijklmno" + '"')
