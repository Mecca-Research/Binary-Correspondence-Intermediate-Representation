#!/usr/bin/env python3
"""Build Tier-2 retrieval chunks from the training corpus.

Tier 2 is the layer between prose a human reads and records a model trains on:
embedding-ready units, each one traceable to the exact lines it came from.

Three properties make these usable as a corpus rather than as a dump:

  * **Deterministic.** The same tree always produces the same chunks in the same
    order with the same identifiers. Identifiers are content-addressed, so
    editing one section does not renumber its neighbours and reordering a file
    does not invalidate the rest of it.
  * **Traceable.** Every chunk names its source path, line span, heading trail,
    and the gates that check that source. A retrieved chunk can always be opened
    at the line it came from.
  * **Honest about what it is not.** The chunker never writes an embedding
    vector. It has no model, and a fabricated vector is indistinguishable
    downstream from a real one -- the same defect class as a benchmark harness
    recording an unreadable counter as zero. `embedding` stays null until a
    named model fills it, and the verifier enforces that.

Chunking rules, in order:

  1. Split on markdown headings; each chunk keeps its heading ancestry.
  2. Never split a fenced code block. A code fence longer than the budget stays
     whole -- a truncated program teaches the wrong thing.
  3. Oversized prose sections split at blank lines, never mid-sentence.
  4. Context comes from the heading trail, not from token overlap: deterministic,
     and it does not duplicate neighbouring text into the index.

    python3 training/tools/build_chunks.py --out build/training/chunks
    python3 training/tools/build_chunks.py --subject llvm --stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

SCHEMA = "bcir-training/chunk/v1"
LICENSE = "LicenseRef-BCIR-NC-1.0"
BUILDER = "training/tools/build_chunks.py"

# Budget in characters. Chars, not tokens: the chunker must not depend on a
# tokenizer it cannot pin, and `token_estimate` is declared an estimate.
DEFAULT_MAX_CHARS = 4000
DEFAULT_MIN_CHARS = 120
CHARS_PER_TOKEN = 4  # coarse, declared, and used only for budgeting

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

# Files that are inputs to the corpus's own machinery rather than teaching text.
EXCLUDED_DIR_NAMES = {"autograder", "dataset", "tests", "eval"}


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


@dataclass
class Section:
    """A heading and the lines beneath it, before size splitting."""

    heading_trail: list[str]
    title: str
    start_line: int
    lines: list[str] = field(default_factory=list)


def split_sections(text: str, document_title: str) -> list[Section]:
    """Split markdown into heading-rooted sections, respecting code fences."""
    sections: list[Section] = []
    trail: list[str] = []
    current = Section(heading_trail=[], title=document_title, start_line=1)
    fence: str | None = None

    for offset, line in enumerate(text.splitlines(), start=1):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(2)
            if fence is None:
                fence = marker
            elif line.strip().startswith(fence):
                fence = None
            current.lines.append(line)
            continue

        # A '#' inside a fence is a comment, not a heading.
        heading = None if fence is not None else HEADING_RE.match(line)
        if heading is None:
            current.lines.append(line)
            continue

        if any(entry.strip() for entry in current.lines):
            sections.append(current)

        depth = len(heading.group(1))
        title = heading.group(2).strip()
        trail = trail[: depth - 1]
        while len(trail) < depth - 1:
            trail.append("")
        trail.append(title)
        current = Section(
            heading_trail=[entry for entry in trail if entry],
            title=title,
            start_line=offset,
        )

    if any(entry.strip() for entry in current.lines):
        sections.append(current)
    return sections


def fenced_blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return (start, end, language) index ranges of fenced blocks."""
    blocks: list[tuple[int, int, str]] = []
    fence: str | None = None
    start = 0
    language = ""
    for index, line in enumerate(lines):
        match = FENCE_RE.match(line)
        if not match:
            continue
        marker = match.group(2)
        if fence is None:
            fence = marker
            start = index
            language = match.group(3).strip()
        elif line.strip().startswith(fence):
            blocks.append((start, index, language))
            fence = None
    if fence is not None:  # unterminated fence: treat the tail as one block
        blocks.append((start, len(lines) - 1, language))
    return blocks


def classify(lines: list[str], blocks: list[tuple[int, int, str]]) -> tuple[str, str | None]:
    """Decide chunk kind and fence language."""
    covered = sum(end - start + 1 for start, end, _ in blocks)
    body = [line for line in lines if line.strip()]
    if not body:
        return "prose", None
    language = next((lang for _, _, lang in blocks if lang), None) or None
    if blocks and covered >= len(lines) - 2:
        return "code", language
    if blocks:
        return "mixed", language
    table_rows = sum(1 for line in body if TABLE_ROW_RE.match(line))
    if table_rows >= 2 and table_rows >= len(body) // 2:
        return "table", None
    return "prose", None


def split_oversized(
    lines: list[str], start_line: int, max_chars: int
) -> list[tuple[list[str], int]]:
    """Split a section at blank lines, never inside a fenced block."""
    blocks = fenced_blocks(lines)
    protected: set[int] = set()
    for start, end, _ in blocks:
        protected.update(range(start, end + 1))

    pieces: list[tuple[list[str], int]] = []
    buffer: list[str] = []
    buffer_start = start_line
    size = 0

    for index, line in enumerate(lines):
        candidate = size + len(line) + 1
        breakable = (
            index not in protected and not line.strip() and buffer and size >= max_chars // 2
        )
        if candidate > max_chars and breakable:
            pieces.append((buffer, buffer_start))
            buffer, size = [], 0
            buffer_start = start_line + index
            continue
        buffer.append(line)
        size = candidate

    if any(entry.strip() for entry in buffer):
        pieces.append((buffer, buffer_start))
    return pieces


def build_chunk(
    *,
    subject: str,
    source: Path,
    source_digest: str,
    heading_trail: list[str],
    title: str,
    lines: list[str],
    start_line: int,
    gates: list[str],
) -> dict | None:
    body = "\n".join(lines).strip("\n")
    if not body.strip():
        return None

    # Heading trail as retrieval context: a chunk found on its own still says
    # where in the document it lives.
    prefix = " > ".join(heading_trail) if heading_trail else title
    text = f"[{subject}] {prefix}\n\n{body}"

    blocks = fenced_blocks(lines)
    kind, language = classify(lines, blocks)

    identity = canonical_json(
        {"subject": subject, "source": relpath(source), "trail": heading_trail, "text": text}
    )
    return {
        "schema": SCHEMA,
        "chunk_id": f"sha256:{sha256_text(identity)}",
        "corpus": "training",
        "subject": subject,
        "source_path": relpath(source),
        "source_sha256": source_digest,
        "span": {"start_line": start_line, "end_line": start_line + len(lines) - 1},
        "heading_trail": heading_trail,
        "title": title,
        "kind": kind,
        "language": language,
        "text": text,
        "char_count": len(text),
        "token_estimate": max(1, len(text) // CHARS_PER_TOKEN),
        "embedding": None,
        "embedding_spec": {
            "model": None,
            "revision": None,
            "dim": None,
            "normalize": "l2",
            "semantics": None,
        },
        "provenance": {"license": LICENSE, "builder": BUILDER},
        "verified_by": gates,
    }


def gates_for(source: Path, subject_root: Path) -> list[str]:
    """Which checked-in gates cover this source file.

    Conservative on purpose: a chunk claiming verification it does not have is
    worse than one honestly marked unverified, because a consumer weights the
    first higher.
    """
    tools = subject_root / "tools"
    rel = source.relative_to(subject_root).as_posix()
    gates: list[str] = []

    def add(name: str) -> None:
        candidate = tools / name
        if candidate.is_file():
            gates.append(relpath(candidate))

    # Prose is checked for link integrity repo-wide.
    gates.append("tools/docs/check_links.py")
    if "/examples/" in rel or rel.startswith("examples/"):
        add("verify-manifest.sh")
        add("verify-examples.sh")
    if rel.startswith("exercises/"):
        add("verify-exercises.sh")
    if rel.startswith("20-clang-frontend/"):
        add("verify-frontend-lowering.py")
    if rel.startswith("18-mlir-lowering-to-llvm/"):
        add("verify-mlir-rail-references.py")
    if rel.startswith("21-performance-methodology/"):
        add("verify-benchmark-analysis.py")
    return sorted(set(gates))


def iter_sources(subject_root: Path) -> list[Path]:
    sources = [
        path
        for path in subject_root.rglob("*.md")
        if not any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(subject_root).parts)
    ]
    return sorted(sources, key=lambda p: p.relative_to(subject_root).as_posix())


def build_subject(subject_root: Path, *, max_chars: int, min_chars: int) -> list[dict]:
    subject = subject_root.name
    chunks: list[dict] = []

    for source in iter_sources(subject_root):
        text = source.read_text(encoding="utf-8")
        digest = sha256_text(text)
        gates = gates_for(source, subject_root)
        document_title = source.stem

        for section in split_sections(text, document_title):
            for lines, start_line in split_oversized(section.lines, section.start_line, max_chars):
                chunk = build_chunk(
                    subject=subject,
                    source=source,
                    source_digest=digest,
                    heading_trail=section.heading_trail,
                    title=section.title,
                    lines=lines,
                    start_line=start_line,
                    gates=gates,
                )
                if chunk is None:
                    continue
                # Drop fragments too small to retrieve usefully, unless they are
                # code: a short example is often the whole point.
                if chunk["char_count"] < min_chars and chunk["kind"] != "code":
                    continue
                chunks.append(chunk)

    chunks.sort(key=lambda c: (c["source_path"], c["span"]["start_line"], c["chunk_id"]))
    return chunks


def subject_roots(selected: str | None) -> list[Path]:
    roots = [
        path
        for path in sorted(CORPUS_ROOT.iterdir())
        if path.is_dir() and not path.name.startswith(".") and path.name not in {"tools", "schema"}
    ]
    if selected:
        roots = [path for path in roots if path.name == selected]
        if not roots:
            raise SystemExit(f"no such subject: {selected}")
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", help="build one subject folder instead of all")
    parser.add_argument("--out", type=Path, help="directory for <subject>.chunks.jsonl")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--stats", action="store_true", help="print a per-subject summary")
    args = parser.parse_args(argv)

    if args.max_chars < 500:
        print("build_chunks: --max-chars below 500 fragments code blocks", file=sys.stderr)
        return 2

    total = 0
    for root in subject_roots(args.subject):
        chunks = build_subject(root, max_chars=args.max_chars, min_chars=args.min_chars)
        total += len(chunks)

        # A subject folder that is still just a scope statement produces
        # nothing; write no file rather than an empty one for the gate to
        # special-case.
        if args.out and chunks:
            args.out.mkdir(parents=True, exist_ok=True)
            destination = args.out / f"{root.name}.chunks.jsonl"
            with destination.open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(canonical_json(chunk) + "\n")
            print(f"[write] {destination} ({len(chunks)} chunk(s))")

        if args.stats or not args.out:
            kinds: dict[str, int] = {}
            verified = 0
            for chunk in chunks:
                kinds[chunk["kind"]] = kinds.get(chunk["kind"], 0) + 1
                if len(chunk["verified_by"]) > 1:
                    verified += 1
            shape = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
            print(
                f"{root.name}: {len(chunks)} chunk(s) [{shape}]; "
                f"{verified} covered by a subject gate beyond link-checking"
            )

    print(f"build_chunks: {total} chunk(s) total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
