#!/usr/bin/env python3
"""An index-backed concept memory, and retrieval you can read (Phase 0.6).

Direct vector retrieval answers a question with a list of chunks and a cosine.
It cannot say *why* those chunks. The similarity that produced them lives in 512
coordinates that mean nothing to a reader, so the only way to check the answer
is to read the chunks and judge for yourself.

This adds a second path with a named intermediate:

    query ──▶ concept ──▶ documents
              ^^^^^^^
              written by a person, in a table, in the repository

`training/llvm/indexes/` already holds that table. Each row says "this concept
is covered in these chapters", in an author's own words. Embedding the
*concepts* rather than the prose turns the index into a queryable memory whose
every answer arrives with a legible reason:

    concept  cos=+0.41  "Compiling and running code at run time"
                        indexes/backend-and-jit.md:17
       -> 12-backend-jit/03-orc-jit.md

**What this makes transparent, precisely.** The memory, not the model. Nothing
here opens up a learned model's weights or explains what a network computes --
that claim would be false and worth refusing. What it does is replace an opaque
nearest-neighbour hop with two hops through a human-readable, auditable,
version-controlled intermediate: you can see which concept fired, disagree with
it, and fix it by editing a table. The retrieval becomes reviewable in the way
the rest of this repository is reviewable.

**The circularity this creates, stated once and enforced elsewhere.** The
evaluation's `index` query family is derived from these same rows. Scoring this
memory on that family would be testing it against its own keys. The evaluation
refuses to do that; see `evaluate_retrieval.py` and `verify_retrieval.py`.

    python3 training/tools/build_index_memory.py --out build/training/memory
    python3 training/tools/build_index_memory.py --recall "run generated code at runtime"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from array import array
from operator import mul
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

SCHEMA = "bcir-training/index-memory/v1"
LICENSE = "LicenseRef-BCIR-NC-1.0"
BUILDER = "training/tools/build_index_memory.py"
DEFAULT_OUT = Path("build/training/memory")

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+?)(?:#[^)]*)?\)")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*?)\s*#*$")
TABLE_ROW = re.compile(r"^\s*\|")
HEADER_WORDS = {
    "concept",
    "topic",
    "keyword",
    "instruction",
    "pass",
    "symbol",
    "term",
    "intrinsic",
    "type",
    "pattern",
    "name",
    "skill",
    "task",
    "what",
    "where",
}


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_cell(cell: str) -> str:
    text = LINK_RE.sub(r"\1", cell).replace("`", "").replace("**", "").replace("*", "")
    return " ".join(text.split()).strip()


def read_entries() -> list[dict]:
    """Every index row that names a concept and points somewhere."""
    entries: list[dict] = []
    index_dir = CORPUS_ROOT / "llvm" / "indexes"
    for index_file in sorted(index_dir.glob("*.md")):
        text = index_file.read_text(encoding="utf-8")
        if "Retired / historical material" in text:
            continue  # its targets were deliberately removed
        relative = index_file.relative_to(REPO_ROOT).as_posix()
        section = ""
        for number, line in enumerate(text.splitlines(), start=1):
            heading = HEADING_RE.match(line)
            if heading:
                section = heading.group(2).strip()
                continue
            if not TABLE_ROW.match(line):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 2 or set(cells[0]) <= set("-: "):
                continue
            concept = clean_cell(cells[0])
            if not concept or concept.lower() in HEADER_WORDS or len(concept) < 6:
                continue
            targets = set()
            for _label, href in LINK_RE.findall(line):
                if not href.endswith(".md"):
                    continue
                try:
                    resolved = (index_file.parent / href).resolve().relative_to(REPO_ROOT)
                except (ValueError, OSError):
                    continue
                targets.add(resolved.as_posix())
            if not targets:
                continue
            identity = canonical_json({"concept": concept, "targets": sorted(targets)})
            entries.append(
                {
                    "schema": SCHEMA,
                    "entry_id": f"sha256:{sha256_text(identity)}",
                    "concept": concept,
                    "section": section,
                    "targets": sorted(targets),
                    "origin": f"{relative}:{number}",
                    "index": relative,
                }
            )
    unique: dict[str, dict] = {}
    for entry in entries:
        unique.setdefault(entry["entry_id"], entry)
    return sorted(unique.values(), key=lambda e: e["entry_id"])


class IndexMemory:
    """Concepts, their vectors, and the documents they point at."""

    def __init__(self, entries: list[dict], codes: array, dim: int, manifest: dict) -> None:
        self.entries = entries
        self.dim = dim
        self.manifest = manifest
        self.views = [codes[i * dim : (i + 1) * dim] for i in range(len(entries))]
        self.squares = [sum(map(mul, view, view)) for view in self.views]

    @classmethod
    def load(cls, root: Path) -> "IndexMemory":
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(
                f"build_index_memory: no memory at {root}\n"
                f"  build one first: python3 {BUILDER} --out {root}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = [
            json.loads(line)
            for line in (root / "memory.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        codes = array("h")
        codes.frombytes((root / "vectors.q15").read_bytes())
        if sys.byteorder == "big":
            codes.byteswap()
        return cls(entries, codes, manifest["dim"], manifest)

    def recall(self, query_codes: array, *, concepts: int = 5) -> list[dict]:
        """Rank concepts, then the documents they name. Every hit keeps its reason."""
        query_square = sum(map(mul, query_codes, query_codes))
        scored = []
        for row, view in enumerate(self.views):
            dot = sum(map(mul, query_codes, view))
            scored.append((query_square + self.squares[row] - 2 * dot, row))
        scored.sort()
        scale = float(self.manifest["scale"])
        return [
            {
                **self.entries[row],
                "cosine": 1.0 - distance / (2.0 * scale * scale),
            }
            for distance, row in scored[:concepts]
        ]


def documents_from(hits: list[dict], limit: int) -> list[str]:
    """Documents in concept order, each attributed to the concept that found it."""
    ordered: list[str] = []
    for hit in hits:
        for target in hit["targets"]:
            if target not in ordered:
                ordered.append(target)
                if len(ordered) >= limit:
                    return ordered
    return ordered


def build(out: Path, *, dim: int) -> dict:
    embed = load_tool("embed_chunks")
    entries = read_entries()
    if not entries:
        raise SystemExit(
            "build_index_memory: no index row yielded a concept with a target; a "
            "memory of nothing recalls nothing and would pass every check"
        )
    provider = embed.build_provider(embed.LexicalHashProvider.name, dim=dim, revision=None)
    vectors, _norms = embed.embed(
        [
            {"text": e["concept"], "chunk_id": e["entry_id"], "source_path": e["origin"]}
            for e in entries
        ],
        provider,
    )
    payload = embed.pack_q15(embed.quantize_q15(vectors))

    out.mkdir(parents=True, exist_ok=True)
    (out / "vectors.q15").write_bytes(payload)
    (out / "memory.jsonl").write_text(
        "\n".join(canonical_json(e) for e in entries) + "\n", encoding="utf-8"
    )
    documents = sorted({t for e in entries for t in e["targets"]})
    manifest = {
        "schema": SCHEMA,
        "corpus": "training",
        "builder": BUILDER,
        "model": provider.name,
        "revision": provider.revision,
        "semantics": provider.semantics,
        "dim": provider.dim,
        "scale": embed.Q15_SCALE,
        "entries": len(entries),
        "documents": len(documents),
        "indexes": sorted({e["index"] for e in entries}),
        "transparency": (
            "Every recall names the concept that produced it and the index row that "
            "defines the concept. This makes the MEMORY auditable; it says nothing "
            "about a learned model's internals."
        ),
        "license": LICENSE,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--recall", help="query the memory and show the reasoning")
    parser.add_argument("--concepts", type=int, default=4)
    args = parser.parse_args(argv)

    if not args.recall:
        manifest = build(args.out, dim=args.dim)
        print(
            f"[memory]  {manifest['entries']} concept(s) over "
            f"{manifest['documents']} document(s) from {len(manifest['indexes'])} index file(s)"
        )
        print(f"[write]   {args.out}")
        return 0

    if not (args.out / "manifest.json").is_file():
        build(args.out, dim=args.dim)
    memory = IndexMemory.load(args.out)
    embed = load_tool("embed_chunks")
    provider = embed.build_provider(memory.manifest["model"], dim=memory.dim, revision=None)
    vectors, _ = embed.embed(
        [{"text": args.recall, "chunk_id": "<query>", "source_path": "<query>"}], provider
    )
    hits = memory.recall(embed.quantize_q15(vectors), concepts=args.concepts)

    print(f'\nQ: "{args.recall}"\n')
    for hit in hits:
        print(f'  concept  cos={hit["cosine"]:+.3f}  "{hit["concept"]}"')
        print(f"           {hit['origin']}")
        for target in hit["targets"]:
            print(f"              -> {target.replace('training/llvm/', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
