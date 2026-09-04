"""Bounded, deterministic subprocess execution for the training grader."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping, Sequence

MAX_CAPTURE_BYTES = 64 * 1024
DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"


@dataclasses.dataclass(frozen=True)
class ProcessResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_truncated: bool = False


def controlled_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the small environment inherited by all grader subprocesses."""
    environment = {
        "PATH": os.environ.get("PATH", DEFAULT_PATH),
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "HOME": "",
        "TMPDIR": "",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "http_proxy": "",
        "https_proxy": "",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "all_proxy": "",
    }
    if extra:
        environment.update(extra)
    return environment


def _bounded_text(path: Path, limit: int) -> tuple[str, bool]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        data = stream.read(limit)
    text = data.decode("utf-8", errors="replace")
    if size > limit:
        text += f"\n[output truncated after {limit} bytes]"
    return text, size > limit


def run_bounded(
    command: Sequence[str],
    *,
    timeout: float,
    stdin: str | None = None,
    max_output_bytes: int = MAX_CAPTURE_BYTES,
    environment: Mapping[str, str] | None = None,
) -> ProcessResult:
    """Run an argument-array command in a fresh directory with bounded capture."""
    if not command or any(not isinstance(item, str) for item in command):
        raise ValueError("external commands must be non-empty argument arrays of strings")
    with tempfile.TemporaryDirectory(prefix="llvm-training-tool-") as temp_name:
        root = Path(temp_name)
        stdout_path = root / "stdout"
        stderr_path = root / "stderr"
        env = controlled_environment(environment)
        env["HOME"] = temp_name
        env["TMPDIR"] = temp_name
        try:
            with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
                completed = subprocess.run(
                    list(command),
                    cwd=root,
                    input=stdin.encode("utf-8") if stdin is not None else None,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
            returncode = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = 124
            timed_out = True
            with stderr_path.open("ab") as stream:
                stream.write(f"\ntimed out after {timeout:g}s\n".encode())
        stdout, stdout_truncated = _bounded_text(stdout_path, max_output_bytes)
        stderr, stderr_truncated = _bounded_text(stderr_path, max_output_bytes)
        return ProcessResult(
            list(command),
            returncode,
            stdout,
            stderr,
            timed_out,
            stdout_truncated or stderr_truncated,
        )
