#!/usr/bin/env python3
"""Scan tracked files for secret-shaped text with an explicit binary/archive policy.

A zero-file scan is INVALID, not clean. Matches are reported by path, rule, line,
and a non-reversible fingerprint — never the matched value.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

BINARY_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".bin", ".o", ".a", ".so", ".dll", ".dylib", ".exe", ".wasm",
    ".bc", ".bcab", ".bcirq8", ".safetensors", ".pt", ".pth", ".onnx",
    ".pdf", ".mp3", ".mp4", ".wav",
})
# Only formats _archive_entries can actually open are archives. Bare single-file
# compression and 7z have no inspection path here, so they follow the binary
# policy (recorded, never parsed) — calling them archives made the scan FAIL on
# legitimate tracked files it could never read.
ARCHIVE_SUFFIXES = frozenset({
    ".zip", ".whl", ".egg", ".jar",
    ".tar", ".tgz", ".tbz", ".tbz2", ".txz",
    ".tar.gz", ".tar.bz2", ".tar.xz",
})
SINGLE_FILE_COMPRESSION = frozenset({".gz", ".bz2", ".xz", ".7z"})
TEXT_SAMPLE = 8192
ARCHIVE_MEMBER_CAP = 1 << 20
ARCHIVE_LOGICAL_CAP = 1 << 28  # 256 MiB declared bytes per tracked archive
ZIP_SYMLINK_MAX = 4096

RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key-header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws-access-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github-fine-grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("assignment-secret", re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|token|passwd)\s*=\s*['\"][^'\"]{12,}['\"]"
    )),
)

PLACEHOLDER = re.compile(
    r"(?i)(example|changeme|change_me|placeholder|redacted|dummy|fake|xxxx|0000|1111|2222)"
)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _git_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _kind_for(path: str, data: bytes) -> str:
    lower = path.lower()
    if any(lower.endswith(suffix) for suffix in ARCHIVE_SUFFIXES):
        return "archive"
    if any(lower.endswith(suffix) for suffix in SINGLE_FILE_COMPRESSION):
        # A compressed tar without a .tar.* name (backup.gz) is still an
        # archive extractors will unpack — probe the content so genuine
        # single-file streams stay binary without losing tar inspection.
        try:
            if tarfile.is_tarfile(io.BytesIO(data)):
                return "archive"
        except (OSError, EOFError, tarfile.TarError):
            pass
        return "binary"
    if any(lower.endswith(suffix) for suffix in BINARY_SUFFIXES):
        return "binary"
    sample = data[:TEXT_SAMPLE]
    if b"\0" in sample:
        return "binary"
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        # NUL-free but not UTF-8: a Latin-1/PEP 263 source, not a binary.
        # It still gets scanned — ASCII-shaped secrets survive the
        # errors="replace" decode — instead of hiding in the binary policy.
        return "text"
    return "text"


def _scan_text(path: str, text: str) -> list[dict[str, Any]]:
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in RULES:
            for match in pattern.finditer(line):
                value = match.group(0)
                if PLACEHOLDER.search(value):
                    continue
                findings.append({
                    "path": path,
                    "line": number,
                    "rule": rule,
                    "fingerprint": _fingerprint(value),
                })
    return findings


def _archive_path_unsafe(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//"):
        return True
    if re.match(r"^[A-Za-z]:", normalized):
        return True  # Windows drive-qualified absolute (C:\escape.txt)
    return ".." in Path(normalized).parts


def _archive_entries(path: Path, data: bytes) -> tuple[list[str], int]:
    """Member names plus link targets — every string an extractor could turn
    into a filesystem path. Raises when the archive cannot be fully inspected,
    which the caller treats as a failing finding, never as a clean scan."""
    checks: list[str] = []
    members = 0
    if zipfile.is_zipfile(path) or data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > ARCHIVE_MEMBER_CAP:
                raise ValueError("archive-member-cap")
            for info in infos:
                members += 1
                checks.append(info.filename)
                mode = info.external_attr >> 16
                # Symlink mode bits alone decide: gating on create_system == 3
                # would leave a mode-bit symlink from another creator unchecked
                # while permissive extractors still honor it.
                if stat.S_ISLNK(mode):
                    # A stored symlink target traverses like a member name. The
                    # payload is read through the ZipInfo (a later same-named
                    # entry cannot alias it), bounded, and an encrypted or
                    # oversized target fails closed instead of passing unread.
                    if info.flag_bits & 0x1 or info.file_size > ZIP_SYMLINK_MAX:
                        raise ValueError("zip-symlink-uninspectable")
                    with archive.open(info) as handle:
                        target = handle.read(ZIP_SYMLINK_MAX + 1)
                    checks.append(target.decode("utf-8", "replace"))
    elif tarfile.is_tarfile(path):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            # Incremental: never materialize getmembers() for a hostile tar.
            logical = 0
            for member in archive:
                members += 1
                if members > ARCHIVE_MEMBER_CAP:
                    raise ValueError("archive-member-cap")
                # Advancing the iterator decompresses through each payload;
                # bound the declared logical bytes so a tiny compression bomb
                # cannot burn the job before the member cap ever triggers.
                logical += max(0, member.size)
                if logical > ARCHIVE_LOGICAL_CAP:
                    raise ValueError("archive-logical-cap")
                checks.append(member.name)
                if member.issym() or member.islnk():
                    checks.append(member.linkname)
    else:
        raise zipfile.BadZipFile("unrecognized archive")
    return checks, members


def _scan_archive(rel: str, path: Path, data: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        checks, members = _archive_entries(path, data)
    except (zipfile.BadZipFile, tarfile.TarError, OSError, ValueError, RuntimeError) as exc:
        findings.append({
            "path": rel, "line": 0, "rule": "archive-unreadable",
            "fingerprint": _fingerprint(type(exc).__name__),
        })
        return findings, {"path": rel, "status": "unreadable", "error": type(exc).__name__}
    unsafe = [name for name in checks if _archive_path_unsafe(name)]
    for name in unsafe:
        findings.append({
            "path": rel,
            "line": 0,
            "rule": "archive-path-traversal",
            "fingerprint": _fingerprint(name),
        })
    return findings, {
        "path": rel,
        "status": "inspected",
        "members": members,
        "unsafe_members": len(unsafe),
        "extracted": False,
    }


def scan_tree(root: Path) -> dict[str, Any]:
    files = _git_files(root)
    report: dict[str, Any] = {
        "state": "PASS",
        "root": str(root),
        "tracked_files": len(files),
        "text_files": 0,
        "binary_files": 0,
        "archive_files": 0,
        "skipped_missing": 0,
        "findings": [],
        "binaries": [],
        "archives": [],
        "engine": "bcir-scan_secrets",
    }
    if not files:
        report["state"] = "INVALID/VACUOUS"
        report["error"] = "git ls-files returned no paths"
        return report
    report["symlinks"] = []
    for rel in files:
        path = root / rel
        if path.is_symlink():
            # A tracked symlink's blob is its target string. Dereferencing
            # would scan arbitrary host bytes (or crash on a procfs target);
            # record it and move on without following.
            report["symlinks"].append(rel)
            continue
        if not path.is_file():
            report["skipped_missing"] += 1
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            # A tracked file the scanner cannot read was not inspected — that
            # is a failing finding, never a silent skip.
            report["findings"].append({
                "path": rel, "line": 0, "rule": "file-unreadable",
                "fingerprint": _fingerprint(type(exc).__name__),
            })
            continue
        kind = _kind_for(rel, data)
        if kind == "binary":
            report["binary_files"] += 1
            report["binaries"].append(rel)
            continue
        if kind == "archive":
            report["archive_files"] += 1
            extra, meta = _scan_archive(rel, path, data)
            report["archives"].append(meta)
            report["findings"].extend(extra)
            continue
        report["text_files"] += 1
        report["findings"].extend(_scan_text(rel, data.decode("utf-8", "replace")))
    if report["findings"]:
        report["state"] = "FAIL"
    elif report["text_files"] == 0:
        report["state"] = "INVALID/VACUOUS"
        report["error"] = "no text files were scanned"
    return report


def _try_gitleaks(root: Path) -> dict[str, Any] | None:
    if os.name == "nt":
        return None
    from shutil import which
    exe = which("gitleaks")
    if not exe:
        return None
    result = subprocess.run(
        [exe, "detect", "--no-git", "--source", str(root), "--config",
         str(root / ".gitleaks.toml"), "--report-format", "json", "--exit-code", "2"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    return {
        "engine": "gitleaks",
        "returncode": result.returncode,
        "stdout_bytes": len(result.stdout),
        "stderr_tail": result.stderr[-400:],
        "available": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scan_secrets")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-gitleaks", action="store_true")
    args = parser.parse_args(argv)
    report = scan_tree(args.root)
    if args.allow_gitleaks:
        extra = _try_gitleaks(args.root)
        report["gitleaks"] = extra or {"available": False}
        if extra and extra["returncode"] != 0:
            # An opted-in engine that reports findings or dies must fail the
            # scan, not ride along as metadata under a green verdict.
            report["state"] = "FAIL"
            report["findings"].append({
                "path": ".", "line": 0, "rule": "gitleaks-nonzero",
                "fingerprint": _fingerprint(f"gitleaks:{extra['returncode']}"),
            })
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"scan_secrets: {report['state']} tracked={report['tracked_files']} "
          f"text={report['text_files']} binary={report['binary_files']} "
          f"archive={report['archive_files']} findings={len(report['findings'])}")
    for finding in report["findings"][:20]:
        print(f"  {finding['path']}:{finding['line']} {finding['rule']} {finding['fingerprint']}")
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
