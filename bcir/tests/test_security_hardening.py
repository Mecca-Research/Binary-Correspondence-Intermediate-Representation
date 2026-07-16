"""Deterministic regressions for local/CI privilege and staging boundaries."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from bcir.channel_plugin import load_manifest


_ROOT = Path(__file__).resolve().parents[2]


def test_mlir_bootstrap_never_executes_an_unverified_existing_file() -> None:
    if os.name == "nt" or shutil.which("bash") is None:
        return
    # Keep the probe on the native POSIX filesystem.  WSL may inherit a Windows
    # TEMP path whose translated permission bits are intentionally non-private.
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        private = Path(td) / "private"
        private.mkdir(mode=0o700)
        marker = private / "executed"
        micromamba = private / "micromamba"
        micromamba.write_text(
            "#!/bin/sh\nprintf reached > \"$BCIR_BOOTSTRAP_MARKER\"\n",
            encoding="utf-8",
        )
        micromamba.chmod(0o700)
        env = dict(os.environ)
        env.update({
            "MICROMAMBA": str(micromamba),
            "MAMBA_ROOT_PREFIX": str(private / "mamba"),
            "BCIR_BOOTSTRAP_MARKER": str(marker),
        })
        result = subprocess.run(
            ["bash", str(_ROOT / "tools/local/setup_mlir22.sh")],
            cwd=_ROOT, env=env, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
        assert "unexpected SHA-256" in result.stderr
        assert not marker.exists(), "an unverified prepositioned bootstrap was executed"


def test_validation_scripts_use_operation_private_tempdirs() -> None:
    scripts = (
        "tools/wsl/check_passes.sh",
        "tools/wsl/check_bytecode.sh",
        "tools/wsl/check_ods_examples.sh",
        "tools/irdl/check_corpus.sh",
        "tools/local/check_rail22.sh",
    )
    fixed_names = re.compile(r"/tmp/(?:pe|bce|bc\.mlirbc|ods_err|irdl_err|rail22_)")
    for relative in scripts:
        text = (_ROOT / relative).read_text(encoding="utf-8")
        assert "mktemp -d" in text, relative
        assert not fixed_names.search(text), relative


def test_workflow_dependencies_are_sha_pinned_and_tokens_are_read_only() -> None:
    sha = re.compile(r"^[0-9a-f]{40}$")
    for workflow in sorted((_ROOT / ".github/workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        assert "pull_request_target:" not in text
        assert re.search(r"(?m)^permissions:\n  contents: read$", text), workflow.name
        dependencies = re.findall(r"(?m)^\s*uses:\s*([^@\s]+)@([^\s#]+)", text)
        assert dependencies, workflow.name
        for name, revision in dependencies:
            assert sha.fullmatch(revision), f"{workflow.name}: {name}@{revision} is mutable"


def test_deeply_nested_plugin_json_fails_as_a_boundary_value_error() -> None:
    """Host recursion limits must not escape an artifact parser as an internal error."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "deep.channel.json"
        path.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
        try:
            load_manifest(str(path))
            raise AssertionError("deeply nested plugin JSON was accepted")
        except ValueError:
            pass
