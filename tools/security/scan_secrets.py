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
    ".gz", ".bz2", ".xz", ".7z",
})
ARCHIVE_SUFFIXES = frozenset({
    ".zip", ".whl", ".egg", ".jar",
    ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz",
})
TEXT_SAMPLE = 8192
ARCHIVE_MEMBER_CAP = 4096
ZIP_SYMLINK_MAX = 4096

RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key-header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws-access-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github-fine-grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("assignment-secret", re.compile(
        r"(?i)['\"]?(?:[A-Za-z0-9]+_)*(?:api[_-]?key|secret|password|token|passwd|aws_secret_access_key)(?:_[A-Za-z0-9]+)*['\"]?"
        r"\s*[:=]\s*(?:'[^'\s]{12,}'|\"[^\"\s]{12,}\"|[A-Za-z0-9_/+-]{16,})"
    )),
)

EXACT_PLACEHOLDERS = frozenset({
    "changeme", "change_me", "placeholder", "redacted", "dummy", "fake",
    "example", "xxxx", "password", "secret", "your_token_here",
    "not_a_secret", "todo",
})


def _assigned_value(match_text: str) -> str:
    extracted = re.search(r"""[:=]\s*['\"]?([^'\"]+?)['\"]?\s*$""", match_text.strip())
    return (extracted.group(1) if extracted else match_text).strip()


def _is_placeholder(value: str) -> bool:
    return _assigned_value(value).lower() in EXACT_PLACEHOLDERS


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _archive_path_unsafe(name: str) -> bool:
    norm = name.replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", norm):
        return True
    if norm.startswith("/") or norm.startswith("//"):
        return True
    return any(part == ".." for part in norm.split("/") if part)


def _read_index_blob(root: Path, rel: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f":{rel}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


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
    lower = path.lower().replace("\\", "/")
    if any(lower.endswith(suffix) for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz")):
        return "archive"
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
                if _is_placeholder(value):
                    continue
                findings.append({
                    "path": path,
                    "line": number,
                    "rule": rule,
                    "fingerprint": _fingerprint(value),
                })
    return findings


def _zip_symlink_target(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str | None:
    unix_mode = info.external_attr >> 16
    if not (info.create_system == 3 and (unix_mode & 0o170000) == 0o120000):
        return None
    if info.file_size > ZIP_SYMLINK_MAX:
        return "../oversized-symlink"
    try:
        return archive.read(info.filename)[:ZIP_SYMLINK_MAX].decode("utf-8", "replace")
    except (RuntimeError, UnicodeError, OSError):
        return None


def _is_zip_bytes(data: bytes) -> bool:
    return data[:2] == b"PK" and data[2:4] in {b"\x03\x04", b"\x05\x06", b"\x07\x08"}


def _archive_members(data: bytes) -> tuple[list[str], bool]:
    names: list[str] = []
    capped = False
    if _is_zip_bytes(data) or zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if len(names) >= ARCHIVE_MEMBER_CAP:
                    capped = True
                    break
                names.append(info.filename)
                target = _zip_symlink_target(archive, info)
                if target:
                    if len(names) >= ARCHIVE_MEMBER_CAP:
                        capped = True
                        break
                    names.append(target)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            for member in archive:
                if len(names) >= ARCHIVE_MEMBER_CAP:
                    capped = True
                    break
                names.append(member.name)
                if getattr(member, "linkname", ""):
                    if len(names) >= ARCHIVE_MEMBER_CAP:
                        capped = True
                        break
                    names.append(member.linkname)
    return names, capped


def _scan_archive(rel: str, data: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        names, capped = _archive_members(data)
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
    if capped:
        findings.append({
            "path": rel,
            "line": 0,
            "rule": "archive-member-cap",
            "fingerprint": _fingerprint(str(ARCHIVE_MEMBER_CAP)),
        })
    unsafe = [name for name in names if _archive_path_unsafe(name)]
    for name in unsafe:
        findings.append({
            "path": rel,
            "line": 0,
            "rule": "archive-path-traversal",
            "fingerprint": _fingerprint(name),
        })
    return findings, {
        "path": rel,
        "status": "capped" if capped else "inspected",
        "members": len(names),
        "unsafe_members": len(unsafe),
        "extracted": False,
        "capped": capped,
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
        "source": "index",
    }
    if not files:
        report["state"] = "INVALID/VACUOUS"
        report["error"] = "git ls-files returned no paths"
        return report
    for rel in files:
        data = _read_index_blob(root, rel)
        if data is None:
            report["findings"].append({
                "path": rel,
                "line": 0,
                "rule": "index-unreadable",
                "fingerprint": _fingerprint(rel),
            })
            continue
        kind = _kind_for(rel, data)
        if kind == "binary":
            report["binary_files"] += 1
            report["binaries"].append(rel)
            continue
        if kind == "archive":
            report["archive_files"] += 1
            extra, meta = _scan_archive(rel, data)
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
