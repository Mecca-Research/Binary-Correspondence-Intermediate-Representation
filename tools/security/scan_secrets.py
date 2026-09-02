#!/usr/bin/env python3
"""Scan tracked files for secret-shaped text with an explicit binary/archive policy.

A zero-file scan is INVALID, not clean. Matches are reported by path, rule, line,
and a non-reversible fingerprint — never the matched value.
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import io
import json
import lzma
import os
import re
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any

try:
    from tools.security.git_index import STAGED_OVERSIZED, staged_blob, staged_divergent
    from tools.security.proc_bounds import run_bounded
except ModuleNotFoundError:  # script execution: sys.path[0] is tools/security
    from git_index import STAGED_OVERSIZED, staged_blob, staged_divergent
    from proc_bounds import run_bounded

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
XZ_MEMORY_LIMIT = 1 << 27  # 128 MiB: the declared LZMA2 dictionary allocates up front
XZ_FEED_CHUNK = 1 << 20  # bounded input slice per decompressor step
XZ_STREAM_CAP = 4096  # concatenated streams; a count is a resource too
GITLEAKS_TIMEOUT = 300.0  # the opted-in engine walks the whole tree; stalls expire
GITLEAKS_OUTPUT_CAP = 1 << 20  # per stream; findings are JSON, not payload
ZIP_SYMLINK_MAX = 4096
# Only the two methods a real symlink member ever uses. LZMA (14) and BZIP2
# (12) build a decompressor from member properties BEFORE a byte emerges —
# an attacker-declared LZMA dictionary allocates ahead of every read cap —
# and no legitimate stored link needs them, so they fail closed instead.
ZIP_SYMLINK_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})

RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key-header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # AKIA = long-lived, ASIA = STS temporary: both are real credentials.
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github-fine-grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    # Separator-joined prefixes and suffixes are the dominant real key shape
    # (DB_PASSWORD, AWS_SECRET_ACCESS_KEY, SECRET_KEY); a bare \b before the
    # keyword made every one of them invisible. Substring identifiers without
    # a separator (tokenizer, passwords_file) stay out of the rule.
    # LINEAR by construction: the keyword is matched directly — \b covers
    # start-of-identifier and hyphen-joined prefixes, the lookbehind covers
    # underscore-joined ones — never through a greedy prefix group, which
    # re-split a long a-a-a… run from every word boundary at every start
    # position and stalled the required scan quadratically on keyword-free
    # lines. The suffix run is possessive (*+): no shorter split of it can
    # rescue a failed delimiter check, because the delimiter characters are
    # not suffix characters, so it never backtracks — and it is BOUNDED
    # (16 segments of 64 chars), so a line that repeats the keyword cannot
    # make every attempt re-walk the rest of the line either. No real key
    # identifier approaches a kilobyte; longer ones are out of scope by
    # declaration, not by accident.
    # Values: quoted (12+ chars) or the unquoted shapes — 16+ value
    # characters carrying at least one digit, OR a 20+ all-lowercase run,
    # OR a hyphen-delimited lowercase passphrase (three-plus words, 16+
    # chars: correct-horse-battery-staple). Hyphens only: snake_case is the
    # identifier shape (default_session_token is a REFERENCE), so
    # underscore-joined values stay out along with UPPER_SNAKE and
    # camelCase references and short config words.
    # Delimiters: assignment (=) AND mapping (:) syntax — YAML `password:`
    # and JSON `"password":` store the same credential; the optional quote
    # before the delimiter is the JSON key's closing quote.
    ("assignment-secret", re.compile(
        r"(?i)(?:\b|(?<=[0-9A-Za-z]_))(?:api[_-]?key|secret|password|token|passwd)"
        r"(?:[_-][A-Za-z0-9]{1,64}){0,16}+['\"]?\s*[:=]\s*"
        r"(?:['\"](?P<value>[^'\"]{12,})['\"]"
        r"|(?P<uvalue>(?=[A-Za-z0-9+/_.\-]*\d)[A-Za-z0-9+/_.\-]{16,}={0,2}"
        r"|(?-i:[a-z]{20,})(?![A-Za-z0-9_])"
        r"|(?-i:(?=[a-z\-]{16,}(?![A-Za-z0-9_.\-]))[a-z]+(?:-[a-z]+){2,})"
        r"(?![A-Za-z0-9_.\-])))"
    )),
)

# Suppression recognizes placeholder VALUES, never substrings: a real
# credential that merely CONTAINS "fake" or "0000" stays a finding. Long
# filler runs suppress anywhere (the doc idiom ghp_xxxxxxxx / AKIA000000);
# word placeholders suppress only when they lead or end the value.
FILLER = re.compile(r"(?i)x{6,}|\*{4,}|0{6,}|1{6,}|2{6,}|9{6,}")
WORD_PLACEHOLDER = re.compile(
    r"(?i)(?:^[\W_]*(?:example|sample|changeme|change[_-]me|placeholder|"
    r"redacted|dummy|fake|your|insert|replace)(?:\b|_)"
    r"|(?:example|sample|placeholder|redacted|changeme)[\W_]*$)"
)
# A template or variable REFERENCE points at a secret, it is not one:
# ${VAR}, $(cmd), {{ mustache }} / ${{ workflow }} openers.
REFERENCE = re.compile(r"^\s*(?:\$\{|\$\(|\{\{)")
# The quoted branch is the only value shape with no structural requirement
# (12+ characters of anything), so prose lands in it: a schema description
# like "object or null" under a key named token_counts. Credentials are
# opaque; when they carry whitespace at all they are passphrases, and
# passphrases are long. Below that length, whitespace means prose.
PROSE_MIN = 20


def _is_prose(value: str) -> bool:
    return bool(re.search(r"\s", value)) and len(value) < PROSE_MIN


def _is_placeholder(value: str) -> bool:
    return bool(
        FILLER.search(value)
        or WORD_PLACEHOLDER.search(value)
        or REFERENCE.match(value)
    )


def _redacted_path(rel: str) -> str:
    """Redact every secret-bearing component, not just the basename: a
    credential can name a DIRECTORY (ghp_.../safe.txt), and copying the
    parent through verbatim republishes it. Components that carry no match
    survive so the finding still says where to look; a match that spans a
    separator redacts the whole path."""
    parts = rel.split("/")
    redacted = [
        "<redacted>" if _scan_text(part, part) else part for part in parts
    ]
    if redacted == parts:
        # The match crossed a component boundary, so no single component
        # reproduces it — nothing here can be shown safely.
        return "<redacted-path>"
    return "/".join(redacted)


def _fingerprint(value: str) -> str:
    # Replacement encoding: a tar member name with a non-UTF-8 byte reaches
    # here as a surrogate, and a strict encode would traceback the required
    # scan in place of its finding. The fingerprint is a stable opaque id,
    # not a reversible value, so replacement costs nothing.
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]


def _git_files(root: Path) -> list[str] | None:
    """Tracked paths, or None when discovery itself failed.

    None is a fail-closed VERDICT for the caller, not an exception: a root
    that is not a checkout, corrupt metadata, dubious-ownership refusal, or
    a git that will not launch all left the required scan emitting a
    traceback in place of its report and its --json-out artifact. The
    boundary audit already answered this the same way.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    # surrogateescape: git hands back raw filename bytes; a non-UTF-8 name
    # must reach the scan loop (which records it) instead of killing
    # discovery with a UnicodeDecodeError before any verdict.
    return [
        item.decode("utf-8", "surrogateescape")
        for item in result.stdout.split(b"\0") if item
    ]


def _suffix_matches(lower: str, suffixes: frozenset[str]) -> bool:
    return any(lower.endswith(suffix) for suffix in suffixes)


# UTF-32 before UTF-16: BOM_UTF32_LE starts with BOM_UTF16_LE's bytes.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"),
    (codecs.BOM_UTF8, "utf-8-sig"),
)


def _utf16_bomless(data: bytes) -> str | None:
    """The BOM-less UTF-16 encoding of ``data``, or None.

    Without a BOM the interleaved NULs read as binary to the generic
    heuristic, so a UTF-16LE config file hid every credential in it. The
    parity of the NUL bytes names the endianness; a STRICT decode of a
    bounded sample plus a printability floor then confirms it, which
    arbitrary binary payloads do not satisfy. Only the sample is decoded —
    this answers a classification question, and the caller that needs the
    whole text decodes it once."""
    if len(data) % 2:
        return None
    sample = data[:TEXT_SAMPLE - (TEXT_SAMPLE % 4)]
    sample = sample[:len(sample) - (len(sample) % 4)]
    if len(sample) < 4:
        return None
    even = sample[0::2].count(0)
    odd = sample[1::2].count(0)
    half = len(sample) // 2
    if odd > half * 0.3 and odd > even * 4:
        encoding = "utf-16-le"
    elif even > half * 0.3 and even > odd * 4:
        encoding = "utf-16-be"
    else:
        # No NUL signal in the sample is not proof of single-byte text: a
        # CJK preamble's code units carry no NUL in either byte (U+4E00–
        # U+7FFF is even ASCII-clean as raw bytes), while an ASCII
        # credential later in the file still does. The parity vote is
        # re-taken over the WHOLE file under an absolute floor instead of
        # a density one: a plain single-byte text file has no NULs at all,
        # an ASCII-bearing UTF-16 file always has them somewhere, and a
        # UTF-16 file with no ASCII anywhere has no ASCII-shaped secrets
        # to miss. Chunked counting: the stride starts stay even, and no
        # 128 MiB slice copy is materialized for a classification
        # question.
        even = 0
        odd = 0
        for start in range(0, len(data), 1 << 16):
            chunk = data[start:start + (1 << 16)]
            even += chunk[0::2].count(0)
            odd += chunk[1::2].count(0)
        if odd >= 8 and odd > even * 4:
            encoding = "utf-16-le"
        elif even >= 8 and even > odd * 4:
            encoding = "utf-16-be"
        else:
            return None
    # A four-byte multiple is NOT code-point alignment: an odd number of BMP
    # characters ahead of a supplementary one puts its surrogate pair astride
    # the sample boundary, leaving a high surrogate with no low half. The
    # strict decode below would then reject a file that is in fact valid
    # UTF-16 — and the caller would file it as binary and never scan it. Drop
    # a trailing unpaired high surrogate before confirming; the sample exists
    # to answer a classification question, not to be complete.
    high = 0
    if encoding == "utf-16-le":
        high = int.from_bytes(sample[-2:], "little")
    else:
        high = int.from_bytes(sample[-2:], "big")
    if 0xD800 <= high <= 0xDBFF:
        sample = sample[:-2]
    if len(sample) < 2:
        return None
    try:
        text = sample.decode(encoding)
    except (UnicodeDecodeError, ValueError):
        return None
    if not text:
        return None
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
    return encoding if printable >= len(text) * 0.9 else None


def _decode_text(data: bytes) -> str:
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return data.decode(encoding, "replace")
    bomless = _utf16_bomless(data)
    if bomless is not None:
        # Replacement decode of the WHOLE file here: the classifier proved
        # the encoding on a sample, and a trailing malformed unit must not
        # cost the scan the rest of the text.
        return data.decode(bomless, "replace")
    return data.decode("utf-8", "replace")


def _tar_header_ok(block: bytes) -> bool:
    """Checksum-valid first tar block — the only signature a legacy V7
    archive carries (no ustar magic at offset 257)."""
    if len(block) < 512:
        return False
    try:
        tarfile.TarInfo.frombuf(block[:512], "utf-8", "surrogateescape")
    except (EOFError, ValueError, tarfile.TarError):
        return False
    return True


def _tar_probe(data: bytes) -> bool:
    """Bounded tar sniff: decompress under the cap FIRST — is_tarfile
    advances into the first member and would otherwise materialize a PAX
    metadata bomb before any bound applies. An over-budget stream probes
    True so the archive path fails it closed; a corrupt stream keeps the
    single-file binary policy."""
    try:
        plain = _decompress_bounded(data)
    except ValueError:
        return True
    except (OSError, EOFError, zlib.error, lzma.LZMAError):
        return False
    try:
        return tarfile.is_tarfile(io.BytesIO(plain))
    except (OSError, EOFError, tarfile.TarError):
        return False


def _kind_for(path: str, data: bytes) -> str:
    lower = path.lower()
    if _suffix_matches(lower, ARCHIVE_SUFFIXES):
        return "archive"
    if _suffix_matches(lower, SINGLE_FILE_COMPRESSION) or any(
        data.startswith(magic) for magic, _ in _COMPRESSED_TAR_MAGIC
    ):
        # A compressed tar without a .tar.* name (backup.gz) is still an
        # archive extractors will unpack — probe the content so genuine
        # single-file streams stay binary without losing tar inspection.
        return "archive" if _tar_probe(data) else "binary"
    if (
        data[:4] == b"PK\x03\x04"
        or data[257:262] == b"ustar"
        or _tar_header_ok(data[:512])
        or zipfile.is_zipfile(io.BytesIO(data))
    ):
        # An archive is what an extractor says it is, not what its suffix
        # says: a ZIP or tar named payload.dat must still have its members
        # inspected instead of hiding behind the binary policy's NUL check.
        # is_zipfile is EOCD-aware, so a prefixed (self-extracting) ZIP
        # cannot dodge the offset-zero magic; the header-checksum probe
        # catches legacy V7 tars that never carry the ustar marker.
        return "archive"
    if _suffix_matches(lower, BINARY_SUFFIXES):
        return "binary"
    for bom, _ in _BOMS:
        if data.startswith(bom):
            # A BOM names the real text encoding; UTF-16/32 NUL bytes must
            # not shunt the file into the binary policy unscanned.
            return "text"
    if _utf16_bomless(data) is not None:
        # BOM-less UTF-16 carries interleaved NULs; the generic heuristic
        # below would file it as binary and never scan it.
        return "text"
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


# A YAML block scalar puts the secret-named key and its value on separate
# lines: `password: |-` then an indented run. Line-by-line, the key has no
# value and the continuation is not token-shaped, so both slipped through.
BLOCK_SCALAR = re.compile(
    r"(?i)^(?P<indent>\s*)(?:-\s+)?['\"]?(?:[A-Za-z0-9]+[_-])*"
    r"(?:api[_-]?key|secret|password|token|passwd)(?:[_-][A-Za-z0-9]+)*"
    r"['\"]?\s*:\s*[|>][-+0-9]*\s*$"
)


def _block_scalar_findings(path: str, lines: list[str]) -> list[dict[str, Any]]:
    """Findings for credentials held in YAML block scalars under a
    secret-named key. The value is the dedented continuation run; it is
    judged by the same placeholder, reference and prose rules as an inline
    value, so a documented example stays suppressed."""
    findings: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = BLOCK_SCALAR.match(line)
        if not match:
            continue
        base = len(match.group("indent"))
        collected: list[str] = []
        for follower in lines[index + 1:]:
            if not follower.strip():
                collected.append("")
                continue
            if len(follower) - len(follower.lstrip()) <= base:
                break
            collected.append(follower.strip())
        value = "\n".join(collected).strip()
        if len(value) < 12 or _is_placeholder(value) or _is_prose(value):
            continue
        findings.append({
            "path": path,
            "line": index + 1,
            "rule": "assignment-secret",
            "fingerprint": _fingerprint(value),
        })
    return findings


# A TOML multiline string does the same key/value split with quote
# delimiters: `password = """` (basic) or `'''` (literal) opens a value
# whose lines follow until the closing delimiter. Line by line, the key
# line ends in an opener the inline rule cannot pair and the continuation
# is bare text. Anchored like BLOCK_SCALAR, so the prefix group cannot
# retry across a long line from more than one start position.
TOML_MULTILINE = re.compile(
    r"(?i)^\s*['\"]?(?:[A-Za-z0-9]+[_-])*"
    r"(?:api[_-]?key|secret|password|token|passwd)(?:[_-][A-Za-z0-9]+)*"
    r"['\"]?\s*=\s*(?P<delim>\"\"\"|''')(?P<rest>.*)$"
)


def _multiline_close(text: str, delim: str) -> int:
    """Index of ``delim`` closing a TOML multiline value in ``text``, or -1.

    A raw ``find`` is not a closing-delimiter test. Inside a BASIC
    (three-double-quote) string a backslash escapes the next character, so
    an escaped quote followed by two literal ones is part of the value,
    not its end — ``tomllib`` reads the credential after it while a raw
    find stopped short and inspected only the prefix. Literal
    (three-single-quote) strings have no escape character at all, so their
    scan stays literal. Walking left to right past escaped characters is
    exact for this construct rather than another subset reader: the escape
    is the only ambiguity the grammar admits here.
    """
    if delim[0] == "'":
        return text.find(delim)
    index = 0
    limit = len(text)
    while index < limit:
        if text[index] == "\\":
            index += 2          # the escaped character closes nothing
            continue
        if text.startswith(delim, index):
            return index
        index += 1
    return -1


def _toml_multiline_findings(path: str, lines: list[str]) -> list[dict[str, Any]]:
    """Findings for credentials held in TOML multiline strings under a
    secret-named key. The value runs to the closing delimiter and is
    judged by the same placeholder, reference and prose rules as an
    inline value, so a documented example stays suppressed."""
    findings: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = TOML_MULTILINE.match(line)
        if not match:
            continue
        delim = match.group("delim")
        rest = match.group("rest")
        closing = _multiline_close(rest, delim)
        if closing >= 0:
            # Opened AND closed on one line. This is NOT the inline rule's
            # territory: its quoted branch reads one quote, then demands 12+
            # non-quote characters, and the delimiter's second quote ends the
            # match immediately — so `password = """secret"""` matched
            # nothing anywhere. The value is the text between the delimiters,
            # judged by the same rules as any other.
            value = rest[:closing].strip()
            if len(value) >= 12 and not _is_placeholder(value) and not _is_prose(value):
                findings.append({
                    "path": path,
                    "line": index + 1,
                    "rule": "assignment-secret",
                    "fingerprint": _fingerprint(value),
                })
            continue
        collected = [rest] if rest else []
        for follower in lines[index + 1:]:
            closing = _multiline_close(follower, delim)
            if closing >= 0:
                collected.append(follower[:closing])
                break
            collected.append(follower)
        value = "\n".join(collected).strip()
        if len(value) < 12 or _is_placeholder(value) or _is_prose(value):
            continue
        findings.append({
            "path": path,
            "line": index + 1,
            "rule": "assignment-secret",
            "fingerprint": _fingerprint(value),
        })
    return findings


# Escaped spellings a loader resolves at runtime: JSON's \uXXXX, and YAML's
# \xNN and \UNNNNNNNN. Each names the same key as its plain spelling, so a
# raw-line rule that only sees the escaped form finds nothing.
_TEXT_ESCAPE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|x([0-9a-fA-F]{2})|U([0-9a-fA-F]{8}))")


def _decode_text_escapes(line: str) -> str:
    r"""ASCII-range \uXXXX / \xNN / \UNNNNNNNN escapes decoded in place.

    ``"passw\x6frd"`` is an ordinary ``password`` key to every YAML loader,
    and ``"password"`` is one to every JSON parser; the assignment rule
    only ever saw the raw spelling. Decoding is restricted to the ASCII
    range on purpose: a non-ASCII escape cannot complete a keyword the
    rules match, and leaving it as written keeps the shadow line the same
    length as the raw one so reported columns stay meaningful.
    """
    def _one(match: re.Match[str]) -> str:
        digits = match.group(1) or match.group(2) or match.group(3)
        point = int(digits, 16)
        return chr(point) if 0x20 <= point < 0x7F else match.group(0)

    return _TEXT_ESCAPE.sub(_one, line)

def _scan_text(path: str, text: str) -> list[dict[str, Any]]:
    findings = []
    lines = text.splitlines()
    findings.extend(_block_scalar_findings(path, lines))
    findings.extend(_toml_multiline_findings(path, lines))
    for number, line in enumerate(text.splitlines(), 1):
        if "\\" in line:
            # Cheap gate: only a line carrying a backslash can carry an
            # escape at all. The decoder decides which spellings count.
            decoded = _decode_text_escapes(line)
            if decoded != line:
                # The decoded shadow is scanned alongside the raw line; a
                # hit in either is a finding on this line. Duplicates from
                # a value matching in both dedupe by fingerprint below.
                seen = {item["fingerprint"] for item in findings}
                for rule, pattern in RULES:
                    for match in pattern.finditer(decoded):
                        value = match.group(0)
                        groups = match.groupdict()
                        if _is_placeholder(
                            groups.get("value") or groups.get("uvalue") or value
                        ):
                            continue
                        if groups.get("value") is not None and _is_prose(groups["value"]):
                            continue
                        fingerprint = _fingerprint(value)
                        if fingerprint in seen:
                            continue
                        findings.append({
                            "path": path, "line": number,
                            "rule": rule, "fingerprint": fingerprint,
                        })
        for rule, pattern in RULES:
            for match in pattern.finditer(line):
                value = match.group(0)
                groups = match.groupdict()
                # Placeholder checks run on the VALUE (assignment RHS or the
                # token itself), not on the key or surrounding text.
                if _is_placeholder(groups.get("value") or groups.get("uvalue") or value):
                    continue
                # Prose suppression applies ONLY to the quoted branch: the
                # unquoted branches and every token rule already carry a
                # shape, so this cannot weaken them.
                if groups.get("value") is not None and _is_prose(groups["value"]):
                    continue
                findings.append({
                    "path": path,
                    "line": number,
                    "rule": rule,
                    "fingerprint": _fingerprint(value),
                })
    return findings


_COMPRESSED_TAR_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gzip"), (b"BZh", "bz2"), (b"\xfd7zXZ\x00", "xz"),
)


def _decompress_bounded(data: bytes) -> bytes:
    """Pre-decompress a compressed tar under ARCHIVE_LOGICAL_CAP. PAX and
    GNU long-name headers materialize while the iterator advances, BEFORE
    any TarInfo is yielded, so the per-member size loop cannot bound them —
    bounding the whole decompressed stream up front does, with one
    predicate. Unrecognized magic returns the bytes unchanged."""
    kind = next(
        (name for magic, name in _COMPRESSED_TAR_MAGIC if data.startswith(magic)),
        None,
    )
    if kind is None:
        return data
    if kind == "xz":
        return _xz_bounded(data)
    if kind == "gzip":
        import gzip
        stream: Any = gzip.GzipFile(fileobj=io.BytesIO(data))
    else:
        import bz2
        stream = bz2.BZ2File(io.BytesIO(data))
    chunks: list[bytes] = []
    total = 0
    with stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > ARCHIVE_LOGICAL_CAP:
                raise ValueError("archive-logical-cap")
            chunks.append(chunk)
    return b"".join(chunks)


def _xz_bounded(data: bytes) -> bytes:
    """LZMAFile enforces no memory limit: a kilobyte stream declaring a
    multi-GiB LZMA2 dictionary allocates it before the first read returns,
    so only the incremental decompressor's memlimit bounds peak memory.
    Exceeding it is the same hostile shape as an over-cap stream and fails
    closed the same way; gzip/bz2 need none (their windows are fixed)."""
    chunks: list[bytes] = []
    total = 0
    view = memoryview(data)
    offset = 0
    streams = 0
    try:
        while offset < len(view):
            # Inter-stream null padding is not a stream; skip it in place.
            while offset < len(view) and view[offset] == 0:
                offset += 1
            if offset >= len(view):
                break
            streams += 1
            if streams > XZ_STREAM_CAP:
                # Concatenated streams are valid xz, but an unbounded COUNT
                # of them is its own resource: each one costs a decompressor
                # and a pass over the remainder, and empty streams advance
                # no output cap at all, so a 2.5 MiB file of 32-byte streams
                # already burned 8 seconds and the ingress cap allowed ~8M
                # of them. The count is bounded like every other budget.
                raise ValueError("archive-logical-cap")
            decomp = lzma.LZMADecompressor(memlimit=XZ_MEMORY_LIMIT)
            while not decomp.eof:
                if decomp.needs_input:
                    if offset >= len(view):
                        raise EOFError("truncated xz stream")
                    # Feed a BOUNDED slice, never the whole remainder: the
                    # decompressor hands back what it did not consume, so
                    # feeding the suffix made every stream re-copy the tail
                    # after it — quadratic in the stream count.
                    block = bytes(view[offset:offset + XZ_FEED_CHUNK])
                    offset += len(block)
                else:
                    block = b""
                chunk = decomp.decompress(block, max_length=1 << 20)
                total += len(chunk)
                if total > ARCHIVE_LOGICAL_CAP:
                    raise ValueError("archive-logical-cap")
                if chunk:
                    chunks.append(chunk)
            # Only the tail of the LAST fed block can be unconsumed, so this
            # rewind is bounded by XZ_FEED_CHUNK rather than by the file.
            offset -= len(decomp.unused_data)
    except lzma.LZMAError as exc:
        if "limit" in str(exc).lower():
            raise ValueError("archive-logical-cap") from exc
        raise
    return b"".join(chunks)


def _zip_declared_members(data: bytes) -> int | None:
    """Bounded EOCD tail read: the declared entry count and central-directory
    byte size cap the member estimate BEFORE ZipFile materializes a ZipInfo
    per entry. A ZIP64 sentinel exceeds the inspection budget outright; an
    absent record returns None and ZipFile decides (and fails closed)."""
    tail = data[-((1 << 16) + 22):]
    at = tail.rfind(b"PK\x05\x06")
    if at < 0 or at + 16 > len(tail):
        return None
    entries, size_cd = struct.unpack("<HI", tail[at + 10:at + 16])
    if entries == 0xFFFF or size_cd == 0xFFFFFFFF:
        return ARCHIVE_MEMBER_CAP + 1
    # 46 bytes is the minimal central-directory record; the declared size
    # therefore bounds how many records ZipFile could parse.
    return max(entries, size_cd // 46)


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
        declared = _zip_declared_members(data)
        if declared is not None and declared > ARCHIVE_MEMBER_CAP:
            raise ValueError("archive-member-cap")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            # The EOCD preflight above bounds the central directory BEFORE
            # ZipFile materializes it; this length check is the backstop for
            # archives whose EOCD understates the truth.
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
                    if (
                        info.flag_bits & 0x1
                        or info.file_size > ZIP_SYMLINK_MAX
                        or info.compress_type not in ZIP_SYMLINK_METHODS
                    ):
                        # Encrypted, oversized, or built on a decompressor
                        # whose allocation precedes the read cap: all three
                        # are uninspectable, never quietly skipped.
                        raise ValueError("zip-symlink-uninspectable")
                    with archive.open(info) as handle:
                        target = handle.read(ZIP_SYMLINK_MAX + 1)
                    checks.append(target.decode("utf-8", "replace"))
    elif (
        # Compressed magic decides FIRST: is_tarfile would advance into the
        # first member and materialize a PAX bomb before any bound; a plain
        # tar's probe is bounded by the bytes already in memory.
        any(data.startswith(magic) for magic, _ in _COMPRESSED_TAR_MAGIC)
        or data[257:262] == b"ustar"
        or tarfile.is_tarfile(io.BytesIO(data))
    ):
        # The stream is decompressed under the logical cap FIRST (a PAX
        # metadata bomb never reaches iteration), then parsed uncompressed.
        plain = _decompress_bounded(data)
        with tarfile.open(fileobj=io.BytesIO(plain), mode="r:") as archive:
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
    except (
        zipfile.BadZipFile, tarfile.TarError, OSError, ValueError, RuntimeError,
        # The decompressor error classes escape the stdlib wrappers: EOFError
        # from a truncated gzip stream mid-iteration, zlib.error from a
        # corrupt deflate payload on the ZIP symlink read, LZMAError from a
        # corrupt xz stream. Each is the same uninspectable-archive failure.
        EOFError, zlib.error, lzma.LZMAError,
    ) as exc:
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
    for name in checks:
        # A member name or link target is committed metadata exactly like a
        # tree entry: a credential spelled into one ships in every clone.
        # These strings were already collected for the traversal predicate;
        # the secret rules run over them too, and the finding carries only
        # the archive's path plus the fingerprint — never the member name.
        for hit in _scan_text(rel, name):
            findings.append({
                "path": rel, "line": 0, "rule": "archive-metadata-secret",
                "fingerprint": hit["fingerprint"],
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
    if files is None:
        # Discovery failed: the scan does not know what it was meant to
        # inspect, so it cannot be clean. A structured FAIL, never a
        # traceback — the exit code and the JSON artifact are the contract.
        return {
            "state": "FAIL",
            "root": str(root),
            "tracked_files": 0,
            "text_files": 0,
            "binary_files": 0,
            "archive_files": 0,
            "skipped_missing": 0,
            "findings": [{
                "path": ".", "line": 0, "rule": "tracked-discovery-failed",
                "fingerprint": _fingerprint("tracked-discovery-failed"),
            }],
            "binaries": [],
            "archives": [],
            "symlinks": [],
            "engine": "bcir-scan_secrets",
            "error": "git ls-files could not enumerate tracked files",
        }
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
        try:
            rel.encode("utf-8")
            printable = rel
            encodable = True
        except UnicodeEncodeError:
            # The tracked filename itself is not UTF-8; everything reported
            # about it uses the printable (replace-decoded) form, which the
            # name scan below sees exactly as the report will print it.
            printable = rel.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
            encodable = False
        # Git commits the tree-entry NAME as surely as it commits blob
        # bytes, so a token spelled into a filename ships in every clone.
        # Once the name itself matches, EVERY report field that would carry
        # it switches to a redacted display path — a finding that redacts
        # while the binaries/archives/symlinks lists keep the raw name
        # still republishes the credential through --json-out and CI logs.
        # Replacement decoding only rewrites the invalid BYTES, so an ASCII
        # credential survives it intact: this scan must run on both paths,
        # not only the UTF-8 one.
        name_hits = _scan_text(printable, printable)
        display = printable
        if name_hits:
            display = _redacted_path(printable)
            for hit in name_hits:
                report["findings"].append({
                    "path": display, "line": 0,
                    "rule": "filename-secret", "fingerprint": hit["fingerprint"],
                })
        if not encodable:
            # Record it fail-closed rather than scanning through — or
            # printing — an unencodable path.
            report["findings"].append({
                "path": display, "line": 0, "rule": "filename-not-utf8",
                "fingerprint": _fingerprint(printable),
            })
            continue
        path = root / rel
        if path.is_symlink():
            # A tracked symlink's blob is its target string. Dereferencing
            # would scan arbitrary host bytes (or crash on a procfs target);
            # scan the committed target TEXT itself — a token can hide in a
            # path — and never follow it.
            report["symlinks"].append(display)
            try:
                target = os.readlink(path)
            except OSError as exc:
                report["findings"].append({
                    "path": display, "line": 0, "rule": "file-unreadable",
                    "fingerprint": _fingerprint(type(exc).__name__),
                })
                continue
            report["findings"].extend(_scan_text(display, target))
            continue
        if not path.is_file():
            if path.is_dir():
                # A tracked directory entry is a submodule gitlink; its
                # content belongs to another repository's scan.
                report["skipped_missing"] += 1
                continue
            # A tracked path absent from the worktree was not inspected; its
            # indexed blob still ships in every clone, so a silent skip could
            # hide a removed-but-tracked credential under PASS.
            report["findings"].append({
                "path": display, "line": 0, "rule": "file-missing",
                "fingerprint": _fingerprint("missing-worktree-file"),
            })
            continue
        lower = rel.lower()
        archive_shaped = _suffix_matches(lower, ARCHIVE_SUFFIXES)
        compressed_shaped = _suffix_matches(lower, SINGLE_FILE_COMPRESSION)
        if (
            not archive_shaped and not compressed_shaped
            and _suffix_matches(lower, BINARY_SUFFIXES)
        ):
            # Suffix-classified binaries are recorded, never materialized —
            # but bounded head and tail signature probes run first, so an
            # archive hiding under payload.pdf cannot dodge inspection. The
            # tail probe matters because ZIP's EOCD lives at the END and
            # zipfile accepts prefixed (self-extracting) archives.
            try:
                probe_size = path.stat().st_size
                with path.open("rb") as handle:
                    head = handle.read(512)
                    tail_len = min(probe_size, (1 << 16) + 22)
                    handle.seek(probe_size - tail_len)
                    tail = handle.read(tail_len)
            except OSError as exc:
                report["findings"].append({
                    "path": display, "line": 0, "rule": "file-unreadable",
                    "fingerprint": _fingerprint(type(exc).__name__),
                })
                continue
            if (
                head[:4] == b"PK\x03\x04"
                or head[257:262] == b"ustar"
                or _tar_header_ok(head)
                or any(head.startswith(magic) for magic, _ in _COMPRESSED_TAR_MAGIC)
                or (tail.rfind(b"PK\x05\x06") >= 0 and zipfile.is_zipfile(path))
            ):
                archive_shaped = True
            else:
                report["binary_files"] += 1
                report["binaries"].append(display)
                continue
        if archive_shaped or compressed_shaped:
            try:
                size = path.stat().st_size
            except OSError as exc:
                report["findings"].append({
                    "path": display, "line": 0, "rule": "file-unreadable",
                    "fingerprint": _fingerprint(type(exc).__name__),
                })
                continue
            if size > ARCHIVE_LOGICAL_CAP:
                if archive_shaped:
                    # Bounds are enforced at ingress: an archive larger than
                    # the inspection budget is a finding, not an OOM.
                    report["archive_files"] += 1
                    report["archives"].append(
                        {"path": display, "status": "oversized", "size": size}
                    )
                    report["findings"].append({
                        "path": display, "line": 0, "rule": "archive-oversized",
                        "fingerprint": _fingerprint("archive-oversized"),
                    })
                else:
                    # A compressed stream too large to probe follows the
                    # binary policy: recorded, never parsed.
                    report["binary_files"] += 1
                    report["binaries"].append(display)
                continue
        else:
            # Every materialized read is bounded, not just archive-shaped
            # ones: a pathological unknown-suffix file must be a finding,
            # never an OOM that kills the scan before its verdict.
            try:
                size = path.stat().st_size
            except OSError as exc:
                report["findings"].append({
                    "path": display, "line": 0, "rule": "file-unreadable",
                    "fingerprint": _fingerprint(type(exc).__name__),
                })
                continue
            if size > ARCHIVE_LOGICAL_CAP:
                report["findings"].append({
                    "path": display, "line": 0, "rule": "file-oversized",
                    "fingerprint": _fingerprint("file-oversized"),
                })
                continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            # A tracked file the scanner cannot read was not inspected — that
            # is a failing finding, never a silent skip.
            report["findings"].append({
                "path": display, "line": 0, "rule": "file-unreadable",
                "fingerprint": _fingerprint(type(exc).__name__),
            })
            continue
        kind = _kind_for(rel, data)
        if kind == "binary":
            report["binary_files"] += 1
            report["binaries"].append(display)
            continue
        if kind == "archive":
            report["archive_files"] += 1
            extra, meta = _scan_archive(display, path, data)
            report["archives"].append(meta)
            report["findings"].extend(extra)
            continue
        report["text_files"] += 1
        report["findings"].extend(_scan_text(display, _decode_text(data)))
    # The walk above scanned the WORKTREE; the next commit records the
    # INDEX. Every path where the two diverge gets its staged blob scanned
    # through the same classification, under the same caps — otherwise a
    # staged secret hides behind a benign unstaged overwrite while the
    # mandated before-commit validation reports PASS.
    staged = staged_divergent(root)
    if staged is None:
        report["findings"].append({
            "path": ".", "line": 0, "rule": "staged-discovery-failed",
            "fingerprint": _fingerprint("staged-discovery-failed"),
        })
    for rel in staged or ():
        try:
            rel.encode("utf-8")
        except UnicodeEncodeError:
            # The tracked-file walk already recorded this name fail-closed
            # as filename-not-utf8; its staged blob stays uninspected for
            # the same reason its worktree copy did.
            continue
        shown = _redacted_path(rel) if _scan_text(rel, rel) else rel
        display = f"{shown} (staged)"
        data = staged_blob(root, rel, cap=ARCHIVE_LOGICAL_CAP)
        if data is None:
            # git reported the path divergent but its index blob cannot be
            # read (or the path is index-deleted): not inspected, so not
            # clean.
            report["findings"].append({
                "path": display, "line": 0, "rule": "staged-unreadable",
                "fingerprint": _fingerprint("staged-unreadable"),
            })
            continue
        if data is STAGED_OVERSIZED:
            # Refused at ingress by `cat-file -s`, before any bytes were
            # materialized — a finding, never an OOM of the required scan.
            report["findings"].append({
                "path": display, "line": 0, "rule": "file-oversized",
                "fingerprint": _fingerprint("file-oversized"),
            })
            continue
        kind = _kind_for(rel, data)
        if kind == "binary":
            report["binary_files"] += 1
            report["binaries"].append(display)
            continue
        if kind == "archive":
            # _archive_entries probes ZIP through a real path before
            # parsing the bytes; a staged blob has no path, so spool it
            # (already bounded by the cap above) for the probe's benefit.
            report["archive_files"] += 1
            with tempfile.NamedTemporaryFile(delete=False) as spool:
                spool.write(data)
                spool_path = Path(spool.name)
            try:
                extra, meta = _scan_archive(display, spool_path, data)
            finally:
                spool_path.unlink(missing_ok=True)
            report["archives"].append(meta)
            report["findings"].extend(extra)
            continue
        report["text_files"] += 1
        report["findings"].extend(_scan_text(display, _decode_text(data)))
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
    # --redact: the stderr tail this report keeps must never depend on an
    # external tool's logging default to avoid echoing raw secret values.
    # The shared bounded runner keeps the opted-in engine inside a wall
    # bound and per-stream byte budgets: a stalled or flooding gitleaks is
    # a structured (and failing) outcome, never a hang or an OOM.
    outcome = run_bounded(
        [exe, "detect", "--redact", "--no-git", "--source", str(root), "--config",
         str(root / ".gitleaks.toml"), "--report-format", "json", "--exit-code", "2"],
        timeout=GITLEAKS_TIMEOUT,
        cap=GITLEAKS_OUTPUT_CAP,
        cwd=root,
    )
    failure = ""
    if not outcome["launched"]:
        failure = outcome["error"]
    elif outcome["timed_out"]:
        failure = f"gitleaks timed out after {GITLEAKS_TIMEOUT:g}s"
    elif outcome["overflow"]:
        failure = f"gitleaks output exceeded {GITLEAKS_OUTPUT_CAP} bytes per stream"
    elif outcome["pipes_held"]:
        failure = "descendant processes still hold gitleaks's pipes"
    result = {
        "engine": "gitleaks",
        "returncode": outcome["returncode"],
        "stdout_bytes": len(outcome["stdout"]),
        "stderr_tail": outcome["stderr"].decode("utf-8", "replace")[-400:],
        "available": True,
    }
    if failure:
        result["error"] = failure
        if result["returncode"] == 0 or result["returncode"] is None:
            # A bounded put-down is never a clean engine exit; the opted-in
            # gate must fail on it.
            result["returncode"] = -1
    return result


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
