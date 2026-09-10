#!/usr/bin/env python3
"""Derive a judged retrieval query set from the corpus itself (Phase 0.6).

The embedding gate proves the vectors are attributed, aligned, non-degenerate,
and searched correctly. None of that is evidence they return the *right* chunk
for a real question. Measuring that needs judgments -- (question, correct
documents) pairs -- and where those come from decides whether the measurement
means anything.

**Nothing here is written for the evaluation.** A query set hand-authored by
whoever also tuned the retriever measures the author's memory of the corpus, not
the retriever. Every query below is instead *derived from a binding the corpus
already made for a different purpose*:

  index     an index row already maps a concept to the chapters covering it.
            Written so a human could navigate, long before this existed.
  crossref  a prose link's text already describes the document it points at.
  distill   a Tier-3 record already names the sources its answer came from,
            and that binding is gate-backed.
  heading   a section's own heading trail. TRIVIAL BY CONSTRUCTION -- see below.

**Every family declares its bias**, because a mixed number hides which part of
the corpus was easy. `index` and `crossref` concepts are written by the same
hand that wrote the chapters, so they share vocabulary; that is a real
limitation and it is recorded in the set rather than discovered later.

`heading` is deliberately trivial: `build_chunks.py` prefixes each chunk with
its own heading trail, so the query is the target's own heading wording. It is
not a quality measure and must never be read as one -- it is a *positive control
for the harness*. If it does not score near-perfectly, retrieval is broken and
every other number in the run is noise.

"Verbatim" is true of the WORD SEQUENCE, not of the string. This asks with
`" ".join(trail)` while `build_chunks.py` writes `"[subject] " + " > ".join(trail)`,
so only 4 of the 60 control queries are literal substrings of their target -- but
`>` is not a word to `LexicalHashProvider`, so all 60 are contiguous runs of words
inside the target's heading line once that tokenizer has seen them. That is the
claim `verify_retrieval.py` checks, in the provider's own tokenizer rather than a
copy of it, because a control whose premise nobody checks is a sentence, not a
control.

    python3 training/tools/build_eval_queries.py --out build/training/eval
    python3 training/tools/build_eval_queries.py --stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

SCHEMA = "bcir-training/eval-query/v1"
LICENSE = "LicenseRef-BCIR-NC-1.0"
BUILDER = "training/tools/build_eval_queries.py"

DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_DISTILL = Path("build/training/distill")
DEFAULT_OUT = Path("build/training/eval")

MIN_QUERY_CHARS = 6
MAX_QUERY_CHARS = 300

# Every family this builder knows about is declared here, and the manifest
# reports all of them -- including the ones that produce nothing. A family that
# silently yields zero is indistinguishable from a family that does not exist,
# which is how the distillation family was lost the first time this ran: a
# length cap deleted all of it and the stats simply did not mention it.
DECLARED_FAMILIES = ("index", "index-authored", "crossref", "distill", "heading")

# Index files written in the same change as this evaluator. Their rows are NOT
# the independent, pre-existing bindings the `index` family claims to be: they
# were authored after retrieval gaps and scores had been measured, by the same
# hand, so they can encode knowledge of the retriever.
#
# They are still useful judgments and still useful navigation, so they are kept
# and separated rather than deleted -- reporting them apart is what lets a
# reader see whether the authored rows score differently from the inherited
# ones. Declared by name because a provenance cutoff read from git would break
# the moment the corpus is exported, vendored, or squashed.
EVALUATION_AUTHORED_INDEXES = frozenset(
    {
        "training/llvm/indexes/backend-and-jit.md",
        "training/llvm/indexes/concurrency-and-atomics.md",
        "training/llvm/indexes/exercises.md",
        "training/llvm/indexes/frontend-and-production-lowering.md",
        "training/llvm/indexes/performance-and-evidence.md",
    }
)
# The heading control is a fixed, evenly spaced sample: enough to detect a
# broken harness, not so many that a trivial family dominates the totals.
HEADING_CONTROL_COUNT = 60

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+?)(?:#[^)]*)?\)")
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
    "what",
    "where",
}

# A family's bias is part of its contract: a consumer that cannot see why a
# family is easy cannot weight its score.
FAMILY_BIAS = {
    "index": (
        "Concepts are written by the same author as the chapters they point at, "
        "so they share vocabulary. The phrasing is descriptive rather than "
        "quoted, so a match is not a substring match, but this is not an "
        "independent judge. These rows predate the evaluation."
    ),
    "index-authored": (
        "WRITTEN ALONGSIDE THIS EVALUATION, after retrieval gaps and scores had "
        "been measured, by the same hand. These rows can encode knowledge of the "
        "retriever and are therefore NOT independent judgments. Reported "
        "separately so the difference from the inherited `index` rows is "
        "visible instead of averaged away."
    ),
    "crossref": (
        "Link text describes the target in the linking author's words. Closest "
        "to a natural question of the four, and the smallest family."
    ),
    "distill": (
        "The question was derived from the same source file it is judged "
        "against, so surface overlap is expected and the score reads high."
    ),
    "heading": (
        "TRIVIAL BY CONSTRUCTION: build_chunks.py prefixes every chunk with its "
        "heading trail, so the query is the target's own heading wording, verbatim "
        "as a word sequence. This is a positive control for the harness, never a "
        "quality measure."
    ),
}


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_cell(cell: str) -> str:
    """Strip markdown decoration from a table cell, keeping the prose."""
    text = LINK_RE.sub(r"\1", cell)
    text = text.replace("`", "").replace("**", "").replace("*", "")
    return " ".join(text.split()).strip()


def resolve(base: Path, href: str) -> str | None:
    if not href.endswith(".md") or href.startswith(("http://", "https://", "mailto:")):
        return None
    try:
        target = (base.parent / href).resolve().relative_to(REPO_ROOT)
    except (ValueError, OSError):
        return None
    return target.as_posix()


def load_chunk_sources(chunk_dir: Path) -> set[str]:
    sources: set[str] = set()
    for path in sorted(chunk_dir.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sources.add(json.loads(line)["source_path"])
    return sources


def corpus_digest(chunk_dir: Path) -> str:
    """`embed_chunks.corpus_digest`, called rather than reimplemented.

    A query set is only meaningful against the corpus it was derived from. Names
    and counts are not enough: a chapter can be rewritten in place, leaving both
    unchanged while every judgment in it becomes stale. The comparison is only
    sound if both sides compute the same function, so there is one function --
    a second implementation "kept in step" is a mismatch waiting for its first
    edit, and it would surface as a spurious staleness failure.
    """
    import importlib.util

    module = sys.modules.get("embed_chunks")
    if module is None:
        spec = importlib.util.spec_from_file_location("embed_chunks", TOOLS_DIR / "embed_chunks.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["embed_chunks"] = module
        spec.loader.exec_module(module)

    chunks = []
    for path in sorted(chunk_dir.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunks.append(json.loads(line))
    return module.corpus_digest(chunks)


def make_query(
    *, family: str, query: str, targets: set[str], origin: str, note: str = ""
) -> dict | None:
    text = " ".join(query.split())
    if not (MIN_QUERY_CHARS <= len(text) <= MAX_QUERY_CHARS) or not targets:
        return None
    identity = canonical_json({"family": family, "query": text, "targets": sorted(targets)})
    return {
        "schema": SCHEMA,
        "query_id": f"sha256:{sha256_text(identity)}",
        "corpus": "training",
        "family": family,
        "query": text,
        "targets": sorted(targets),
        "origin": origin,
        "note": note,
        "bias": FAMILY_BIAS[family],
        "provenance": {"license": LICENSE, "builder": BUILDER},
    }


def from_indexes(sources: set[str]) -> list[dict]:
    """An index row already says: this concept lives in these chapters."""
    queries: list[dict] = []
    for index_file in sorted((CORPUS_ROOT / "llvm" / "indexes").glob("*.md")):
        text = index_file.read_text(encoding="utf-8")
        # A retired index points at deliberately removed paths; its rows would
        # be judgments whose answers do not exist.
        if "Retired / historical material" in text:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if not TABLE_ROW.match(line):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 2 or set(cells[0]) <= set("-: "):
                continue
            concept = clean_cell(cells[0])
            if not concept or concept.lower() in HEADER_WORDS:
                continue
            targets = {
                resolved
                for _text, href in LINK_RE.findall(line)
                if (resolved := resolve(index_file, href)) and resolved in sources
            }
            relative = index_file.relative_to(REPO_ROOT).as_posix()
            entry = make_query(
                family=("index-authored" if relative in EVALUATION_AUTHORED_INDEXES else "index"),
                query=concept,
                targets=targets,
                origin=f"{relative}:{number}",
            )
            if entry:
                queries.append(entry)
    return queries


def from_crossrefs(sources: set[str]) -> list[dict]:
    """A prose link whose text describes the target in the author's own words."""
    queries: list[dict] = []
    for page in sorted((CORPUS_ROOT / "llvm").rglob("*.md")):
        if "indexes/" in page.relative_to(CORPUS_ROOT).as_posix():
            continue
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), start=1):
            if TABLE_ROW.match(line):
                continue  # table rows belong to the index family
            for text, href in LINK_RE.findall(line):
                label = clean_cell(text)
                # A link whose text is its own path describes nothing.
                if "/" in label or label.endswith(".md"):
                    continue
                resolved = resolve(page, href)
                if not resolved or resolved not in sources:
                    continue
                if resolved == page.relative_to(REPO_ROOT).as_posix():
                    continue
                entry = make_query(
                    family="crossref",
                    query=label,
                    targets={resolved},
                    origin=f"{page.relative_to(REPO_ROOT).as_posix()}:{number}",
                )
                if entry:
                    queries.append(entry)
    return queries


def measure_distillation(distill_dir: Path, sources: set[str]) -> dict:
    """Measure the Tier-3 prompts, and record why they are not judged queries.

    A distillation record does name the sources its answer came from, which
    looks like a ready-made judgment. It is not one. Every record whose sources
    are actually chunked carries the *whole task prompt* as its question --
    hundreds to thousands of characters, much of it IR listings.

    That fails as a retrieval query twice over. It is not a question anyone
    asks, and its length is precisely what would flatter a lexical retriever
    against the very file it was derived from: more text, more n-gram overlap,
    a higher score that measures nothing.

    So the family is excluded, and the numbers behind that decision are
    measured here rather than asserted, so the reason stays true if the records
    change.
    """
    lengths: list[int] = []
    if distill_dir.is_dir():
        for path in sorted(distill_dir.glob("*.distill.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if not any(source in sources for source in record["source_paths"]):
                    continue
                lengths.append(len(" ".join(record["messages"][1]["content"].split())))
    if not lengths:
        return {
            "candidates": 0,
            "reason": "no distillation record names a chunked source file",
        }
    lengths.sort()
    return {
        "candidates": len(lengths),
        "min_chars": lengths[0],
        "median_chars": lengths[len(lengths) // 2],
        "max_chars": lengths[-1],
        "reason": (
            f"{len(lengths)} record(s) name a chunked source, but their prompts run "
            f"{lengths[0]}-{lengths[-1]} characters (median {lengths[len(lengths) // 2]}). "
            "Those are task descriptions, not retrieval queries, and their length "
            "would inflate a lexical score against the file they were derived from."
        ),
    }


def from_headings(chunk_dir: Path) -> list[dict]:
    """The harness control: a chunk's own heading trail must find that chunk."""
    chunks: list[dict] = []
    for path in sorted(chunk_dir.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = json.loads(line)
                if chunk["heading_trail"]:
                    chunks.append(chunk)
    if not chunks:
        return []
    step = max(1, len(chunks) // HEADING_CONTROL_COUNT)
    queries: list[dict] = []
    for chunk in chunks[::step][:HEADING_CONTROL_COUNT]:
        entry = make_query(
            family="heading",
            query=" ".join(chunk["heading_trail"]),
            targets={chunk["source_path"]},
            origin=f"{chunk['source_path']}:{chunk['span']['start_line']}",
            note="harness control",
        )
        if entry:
            queries.append(entry)
    return queries


def build(chunk_dir: Path, distill_dir: Path) -> tuple[list[dict], dict]:
    if not chunk_dir.is_dir():
        raise SystemExit(
            f"build_eval_queries: no chunk build at {chunk_dir}\n"
            "  build one first: python3 training/tools/build_chunks.py "
            f"--out {chunk_dir}"
        )
    sources = load_chunk_sources(chunk_dir)
    if not sources:
        raise SystemExit(
            f"build_eval_queries: {chunk_dir} holds no chunks; a query set judged "
            "against an empty corpus would score perfectly by measuring nothing"
        )

    queries = from_indexes(sources) + from_crossrefs(sources) + from_headings(chunk_dir)
    # Content-addressed identity makes duplicates exact: the same concept
    # pointing at the same targets from two index files is one judgment.
    unique: dict[str, dict] = {}
    for entry in queries:
        unique.setdefault(entry["query_id"], entry)
    ordered = sorted(unique.values(), key=lambda q: (q["family"], q["query_id"]))

    counts = {family: 0 for family in DECLARED_FAMILIES}
    for entry in ordered:
        counts[entry["family"]] += 1

    families: dict[str, dict] = {}
    for family in DECLARED_FAMILIES:
        block = {
            "queries": counts[family],
            "bias": FAMILY_BIAS[family],
            "control": family == "heading",
            "excluded": None,
        }
        if family == "distill":
            block["excluded"] = measure_distillation(distill_dir, sources)["reason"]
        families[family] = block

    manifest = {
        "schema": "bcir-training/eval-set/v1",
        "corpus": "training",
        "builder": BUILDER,
        "judged_from": (
            "bindings the corpus already made for other purposes; no query was "
            "written for this evaluation"
        ),
        "queries": len(ordered),
        "documents": len(sources),
        "corpus_sha256": corpus_digest(chunk_dir),
        "families": families,
        "license": LICENSE,
    }
    return ordered, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--distill", type=Path, default=DEFAULT_DISTILL)
    parser.add_argument("--out", type=Path, help="directory for queries.jsonl")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args(argv)

    queries, manifest = build(args.chunks, args.distill)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        destination = args.out / "queries.jsonl"
        destination.write_text(
            "\n".join(canonical_json(q) for q in queries) + "\n", encoding="utf-8"
        )
        (args.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[write]   {destination} ({len(queries)} query/queries)")

    if args.stats or not args.out:
        by_family: dict[str, list[dict]] = {}
        for entry in queries:
            by_family.setdefault(entry["family"], []).append(entry)
        for family in DECLARED_FAMILIES:
            group = by_family.get(family, [])
            block = manifest["families"][family]
            if block["excluded"]:
                print(f"  {family:9s}    - excluded: {block['excluded']}")
                continue
            spread = sum(len(q["targets"]) for q in group) / len(group) if group else 0.0
            tag = "  (harness control)" if block["control"] else ""
            print(f"  {family:9s} {len(group):4d} query/queries, {spread:.1f} target(s) each{tag}")
    print(f"build_eval_queries: {len(queries)} judged query/queries total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
