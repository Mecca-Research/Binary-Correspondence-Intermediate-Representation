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
ARCHIVE_SUFFIXES = frozenset({
    ".zip", ".whl", ".egg", ".jar", ".7z",
    ".tar", ".tgz", ".gz", ".bz2", ".xz",
})
TEXT_SAMPLE = 8192
ARCHIVE_MEMBER_CAP = 1 << 20

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
    if any(lower.endswith(suffix) for suffix in BINARY_SUFFIXES):
        return "binary"
    sample = data[:TEXT_SAMPLE]
    if b"\0" in sample:
        return "binary"
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return "binary"
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


def _archive_members(path: Path, data: bytes) -> list[str]:
    names: list[str] = []
    if zipfile.is_zipfile(path) or data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
    elif tarfile.is_tarfile(path):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            names = [member.name for member in archive.getmembers()]
    return names


def _scan_archive(rel: str, path: Path, data: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        names = _archive_members(path, data)
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        findings.append({
            "path": rel,
            "line": 0,
            "rule": "archive-unreadable",
            "fingerprint": _fingerprint(type(exc).__name__),
        })
        return findings, {
            "path": rel,
            "status": "unreadable",
            "error": type(exc).__name__,
            "extracted": False,
        }
    inspectable = (
        zipfile.is_zipfile(path) or data[:4] == b"PK\x03\x04" or tarfile.is_tarfile(path)
    )
    if not inspectable:
        findings.append({
            "path": rel,
            "line": 0,
            "rule": "archive-unreadable",
            "fingerprint": _fingerprint("uninspectable"),
        })
        return findings, {
            "path": rel,
            "status": "unreadable",
            "error": "no-supported-archive-parser",
            "extracted": False,
        }
    unsafe = [
        name for name in names
        if name.startswith("/") or name.startswith("\\") or ".." in Path(name).parts
    ]
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
        "members": len(names),
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
    for rel in files:
        path = root / rel
        if not path.is_file():
            report["skipped_missing"] += 1
            continue
        data = path.read_bytes()
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
    if report["text_files"] == 0:
        report["state"] = "INVALID/VACUOUS"
        report["error"] = "no text files were scanned"
    elif report["findings"]:
        report["state"] = "FAIL"
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
        if extra and extra.get("returncode") not in (0, None):
            report["state"] = "FAIL"
            report["findings"].append({
                "path": "<gitleaks>",
                "line": 0,
                "rule": "gitleaks-nonzero",
                "fingerprint": _fingerprint(str(extra.get("returncode"))),
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
