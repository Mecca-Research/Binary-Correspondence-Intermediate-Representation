#!/usr/bin/env python3
"""Gate the Tier-2 embedding sets (Phase 0.5).

The chunk gate proves the corpus is chunked honestly. This one proves the
vectors built from it are worth computing against. What it enforces, in order of
how badly each failure would poison a consumer:

  1. **Anti-vacuity.** A set covering nothing FAILS. A vector gate that passes
     over an empty set is green for not running.
  2. **No unattributable vector.** Every set names its model, its pinned
     revision, its dimension, and -- the field this phase adds -- whether its
     similarity is `lexical` or `learned`. Naming a model is necessary and not
     sufficient: a hashed-n-gram vector and a trained encoder are both named and
     are not interchangeable.
  3. **No degenerate vector.** Finite, correct length, non-zero, and unit-length
     where the set claims L2. A zero vector has no direction and cosine against
     it is undefined; storing one is the vector form of recording an unreadable
     counter as 0.
  4. **Row alignment.** Embedding a chunk's own text must retrieve that chunk at
     rank 1 with distance exactly 0. This is what catches an off-by-one between
     index.jsonl and the vector rows -- a defect every structural check passes
     over, and one that silently returns the neighbouring chunk forever.
  5. **Discrimination.** A model mapping every chunk to nearly one direction
     passes every check above while being useless. A set whose vectors do not
     separate is a vacuous set.
  6. **Binding and staleness.** Every row names a chunk that exists, and the
     digest of the text actually embedded still matches that chunk today.
  7. **Coverage is declared.** Partial coverage is legal; silent partial
     coverage is not.
  8. **Determinism, for the sets that claim it.** A set declaring
     `deterministic: true` is gated on byte-identity across two builds. A
     learned model that honestly declares `false` is gated on its invariants
     instead -- contracting for bit-identity across arbitrary hardware would be
     a promise the rail cannot keep, and gating on a promise nobody can keep
     produces flaky red rather than evidence.
  9. **The native differential.** The pure-Python ranking and BCIR's own
     `bcir_ai_q15_topk` must agree exactly -- same rows, same integer distances.
     Two independent implementations of one contract, checked on the real
     corpus. Absent a C compiler this is an honest skip, and never a pass.

    python3 training/tools/verify_embeddings.py
    python3 training/tools/verify_embeddings.py --require-native
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from array import array
from operator import mul
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

SET_SCHEMA = "bcir-training/embedding-set/v1"
SEMANTICS = {"lexical", "learned"}
NORMALIZE = {"l2", "none"}

MANIFEST_REQUIRED = {
    "schema",
    "corpus",
    "model",
    "revision",
    "dim",
    "normalize",
    "semantics",
    "deterministic",
    "builder",
    "built_from",
    "coverage",
    "vectors",
    "quantized",
    "index",
    "license",
}

# Unit-length within a tolerance that admits float32 storage of a float64
# normalization, and nothing looser.
NORM_TOLERANCE = 2e-6
# One Q15 code of slack: the stored f32 is a rounded copy of the f64 the
# quantizer saw, so the two can disagree by a single least-significant code.
Q15_TOLERANCE = 1
# A set whose vectors are nearly parallel cannot rank anything. A lexical model
# on one technical corpus sits far below this; a collapsed model sits at ~1.0.
MAX_MEAN_ABS_COSINE = 0.60
# Legitimate duplicates exist (the same passage under the same trail in two
# files). A model that has collapsed produces them in bulk.
MIN_DISTINCT_FRACTION = 0.95
PROBE_COUNT = 12


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.skips: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)

    def skip(self, message: str) -> None:
        self.skips.append(message)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_f32(path: Path) -> array:
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder == "big":
        values.byteswap()
    return values


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_manifest(manifest: dict, report: Report) -> None:
    missing = MANIFEST_REQUIRED - set(manifest)
    if not report.require(not missing, f"manifest: missing key(s) {sorted(missing)}"):
        return
    extra = set(manifest) - MANIFEST_REQUIRED
    report.require(not extra, f"manifest: unexpected key(s) {sorted(extra)}")

    report.require(manifest["schema"] == SET_SCHEMA, "manifest: wrong schema tag")
    report.require(
        isinstance(manifest["model"], str) and manifest["model"].strip(),
        "manifest: a set with no model named is a set of unattributable vectors",
    )
    report.require(
        isinstance(manifest["revision"], str) and manifest["revision"].strip(),
        "manifest: revision is unpinned; the same manifest would describe different vectors later",
    )
    report.require(isinstance(manifest["dim"], int) and manifest["dim"] >= 1, "manifest: bad dim")
    report.require(manifest["normalize"] in NORMALIZE, "manifest: bad normalize")
    report.require(
        manifest["semantics"] in SEMANTICS,
        f"manifest: semantics must be one of {sorted(SEMANTICS)} -- a consumer "
        "cannot weight a set that does not say what kind of similarity it has",
    )
    report.require(
        isinstance(manifest["deterministic"], bool), "manifest: deterministic must be a bool"
    )
    report.require(
        (REPO_ROOT / manifest["builder"]).is_file(),
        f"manifest: builder does not exist: {manifest['builder']}",
    )


def check_coverage(manifest: dict, rows: list[dict], report: Report) -> None:
    coverage = manifest["coverage"]
    total = manifest["built_from"]["chunks_total"]
    report.require(coverage["embedded"] >= 1, "coverage: a set that embedded nothing")
    report.require(
        coverage["embedded"] == len(rows),
        f"coverage: declares {coverage['embedded']} embedded, index holds {len(rows)}",
    )
    report.require(
        coverage["embedded"] + coverage["skipped"] == total,
        f"coverage: {coverage['embedded']} + {coverage['skipped']} != {total} chunks",
    )
    if coverage["skipped"]:
        report.require(
            isinstance(coverage["reason"], str) and coverage["reason"].strip(),
            f"coverage: {coverage['skipped']} chunk(s) skipped with no reason given -- "
            "partial coverage is legal, silent partial coverage is not",
        )


def check_digests(root: Path, manifest: dict, report: Report) -> None:
    for section in ("vectors", "index", "quantized"):
        block = manifest[section]
        if block is None:
            continue
        path = root / block["path"]
        if not report.require(path.is_file(), f"{section}: {block['path']} is missing"):
            continue
        report.require(
            sha256_bytes(path.read_bytes()) == block["sha256"],
            f"{section}: {block['path']} does not match its recorded digest",
        )


def check_vectors(root: Path, manifest: dict, rows: list[dict], report: Report) -> None:
    dim = manifest["dim"]
    values = read_f32(root / manifest["vectors"]["path"])
    expected = len(rows) * dim
    if not report.require(
        len(values) == expected,
        f"vectors: {len(values)} floats for {len(rows)} rows x dim {dim} = {expected}",
    ):
        return
    report.require(
        manifest["vectors"]["rows"] == len(rows),
        "vectors: declared row count disagrees with the index",
    )

    quantized = manifest["quantized"]
    codes: array | None = None
    if quantized is not None:
        codes = array("h")
        codes.frombytes((root / quantized["path"]).read_bytes())
        if sys.byteorder == "big":
            codes.byteswap()
        report.require(
            len(codes) == expected,
            f"quantized: {len(codes)} codes, expected {expected}",
        )
    else:
        report.require(
            manifest["normalize"] == "none",
            "quantized: only an unnormalized set may decline a Q15 view",
        )

    non_finite = zero = unnormalized = asymmetric = drifted = 0
    for index, entry in enumerate(rows):
        vector = values[index * dim : (index + 1) * dim]
        if not all(math.isfinite(value) for value in vector):
            non_finite += 1
            continue
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm == 0.0:
            zero += 1
            continue
        if manifest["normalize"] == "l2" and abs(norm - 1.0) > NORM_TOLERANCE:
            unnormalized += 1
        if entry["raw_norm"] <= 0.0:
            zero += 1

        if codes is not None:
            scale = quantized["scale"]
            for offset, value in enumerate(vector):
                code = codes[index * dim + offset]
                if code == -scale - 1:
                    asymmetric += 1
                    break
                if abs(round(value * scale) - code) > Q15_TOLERANCE:
                    drifted += 1
                    break

    report.require(not non_finite, f"vectors: {non_finite} vector(s) contain a non-finite value")
    report.require(
        not zero,
        f"vectors: {zero} zero-length vector(s); a vector with no direction cannot "
        "be compared and must never be stored in place of a missing one",
    )
    report.require(
        not unnormalized,
        f"vectors: {unnormalized} vector(s) are not unit length though the set declares l2",
    )
    report.require(
        not asymmetric,
        f"quantized: {asymmetric} vector(s) use the forbidden asymmetric code -32768",
    )
    report.require(
        not drifted,
        f"quantized: {drifted} vector(s) disagree with their f32 source by more than "
        f"{Q15_TOLERANCE} code -- the two views are not the same vectors",
    )


def check_binding(
    manifest: dict, rows: list[dict], chunks: dict[str, dict], report: Report
) -> None:
    seen: set[str] = set()
    for index, entry in enumerate(rows):
        report.require(entry["row"] == index, f"index[{index}]: row field disagrees with position")
        report.require(entry["chunk_id"] not in seen, f"index[{index}]: duplicate chunk_id")
        seen.add(entry["chunk_id"])

        chunk = chunks.get(entry["chunk_id"])
        if not report.require(chunk is not None, f"index[{index}]: names a chunk not in the build"):
            continue
        assert chunk is not None
        report.require(
            sha256_text(chunk["text"]) == entry["text_sha256"],
            f"index[{index}]: the text embedded is not the chunk's text today -- "
            "this vector is stale",
        )
        report.require(
            chunk["source_path"] == entry["source_path"],
            f"index[{index}]: source_path disagrees with the chunk",
        )

    identity = sorted(
        (
            {
                "chunk_id": entry["chunk_id"],
                "source_sha256": chunks[entry["chunk_id"]]["source_sha256"],
            }
            for entry in rows
            if entry["chunk_id"] in chunks
        ),
        key=lambda row: row["chunk_id"],
    )
    recomputed = sha256_text(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    report.require(
        recomputed == manifest["built_from"]["corpus_digest"],
        "built_from: corpus_digest does not match the chunks this set covers",
    )


def check_discrimination(embedding_set, report: Report) -> None:
    """A set whose vectors do not separate cannot rank anything."""
    rows = len(embedding_set.rows)
    distinct = len({view.tobytes() for view in embedding_set.row_views})
    report.require(
        distinct >= rows * MIN_DISTINCT_FRACTION,
        f"discrimination: only {distinct} distinct vectors among {rows} chunks "
        f"(< {MIN_DISTINCT_FRACTION:.0%}); the model has collapsed",
    )

    # Deterministic, evenly spaced sample -- no RNG, so the gate checks the same
    # pairs on every host.
    step = max(1, rows // 48)
    sample = list(range(0, rows, step))[:48]
    scale = float(embedding_set.scale)
    total = 0.0
    pairs = 0
    for position, left in enumerate(sample):
        for right in sample[position + 1 :]:
            a, b = embedding_set.row_views[left], embedding_set.row_views[right]
            norm_a = math.sqrt(embedding_set.row_squares[left]) or 1.0
            norm_b = math.sqrt(embedding_set.row_squares[right]) or 1.0
            total += abs(sum(map(mul, a, b)) / (norm_a * norm_b))
            pairs += 1
    mean = total / pairs if pairs else 1.0
    report.require(pairs > 0, "discrimination: sampled no pairs -- the check examined nothing")
    report.require(
        mean <= MAX_MEAN_ABS_COSINE,
        f"discrimination: mean |cosine| over {pairs} sampled pairs is {mean:.3f} "
        f"(> {MAX_MEAN_ABS_COSINE}); these vectors are nearly parallel and cannot rank",
    )
    print(f"[discrim] {distinct}/{rows} distinct; mean |cos| {mean:.3f} over {pairs} pairs")
    return mean


def check_row_alignment(embedding_set, chunks: dict[str, dict], search, report: Report) -> list:
    """A chunk's own text must retrieve that chunk, at distance exactly zero.

    This is the check that catches a row misalignment between index.jsonl and
    the vector block. Every structural check passes over that defect, and it
    silently returns the neighbouring chunk for the life of the index.
    """
    rows = len(embedding_set.rows)
    step = max(1, rows // PROBE_COUNT)
    probes = list(range(0, rows, step))[:PROBE_COUNT]
    queries: list[tuple[int, array]] = []

    for row in probes:
        entry = embedding_set.rows[row]
        chunk = chunks.get(entry["chunk_id"])
        if chunk is None:
            continue
        query = search.embed_query(chunk["text"], embedding_set)
        queries.append((row, query))
        ranked = search.topk_reference(query, embedding_set, 1)
        report.require(
            bool(ranked) and ranked[0][0] == row,
            f"alignment: row {row} ({entry['source_path']}) does not retrieve itself; "
            "index rows and vector rows are misaligned",
        )
        report.require(
            bool(ranked) and ranked[0][1] == 0,
            f"alignment: row {row} retrieves itself at distance {ranked[0][1] if ranked else '?'}, "
            "not 0 -- the stored vector is not the vector its text produces",
        )
    report.require(
        len(queries) == len(probes),
        f"alignment: only {len(queries)} of {len(probes)} probes ran",
    )
    print(f"[align]   {len(queries)} probe(s) retrieved themselves at distance 0")
    return queries


def check_native_differential(
    embedding_set, queries, search, report: Report, *, require: bool
) -> bool:
    """Reference ranking versus BCIR's own kernel, on the real corpus."""
    if not queries:
        report.require(False, "differential: no queries to compare")
        return False
    try:
        for row, query in queries:
            native = search.topk_native(query, embedding_set, 3)
            reference = search.topk_reference(query, embedding_set, 3)
            report.require(
                native == reference,
                f"differential: row {row} ranks differently under bcir_ai_q15_topk "
                f"({native}) than under the reference ({reference})",
            )
    except search.BackendUnavailable as exc:
        if require:
            report.require(
                False,
                f"differential: native rail required but unavailable: {exc}",
            )
            return False
        report.skip(f"native differential: {exc}")
        return False
    print(f"[diff]    bcir_ai_q15_topk == reference on {len(queries)} query/3-result set(s)")
    return True


def build_once(directory: Path, chunks_module, embed_module) -> tuple[Path, Path]:
    chunk_dir = directory / "chunks"
    embed_dir = directory / "embeddings"
    chunks_module.main(["--out", str(chunk_dir)])
    status = embed_module.main(["--chunks", str(chunk_dir), "--out", str(embed_dir)])
    if status != 0:
        raise SystemExit(f"verify_embeddings: embedding build failed with status {status}")
    return chunk_dir, embed_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="fail instead of skipping when BCIR's native kernels cannot be built",
    )
    parser.add_argument("--keep", type=Path, help="also write the verified build here")
    args = parser.parse_args(argv)

    chunks_module = load_module(TOOLS_DIR / "build_chunks.py", "build_chunks")
    embed_module = load_module(TOOLS_DIR / "embed_chunks.py", "embed_chunks")
    search = load_module(TOOLS_DIR / "search_chunks.py", "search_chunks")

    report = Report()
    workspace = Path(tempfile.mkdtemp(prefix="bcir-embeddings-"))
    try:
        chunk_dir_a, embed_dir_a = build_once(workspace / "a", chunks_module, embed_module)
        chunk_dir_b, embed_dir_b = build_once(workspace / "b", chunks_module, embed_module)

        chunks: dict[str, dict] = search.load_chunk_texts(chunk_dir_a)
        report.require(bool(chunks), "no chunks to embed -- the corpus build is empty")

        set_roots = sorted(path for path in embed_dir_a.iterdir() if path.is_dir())
        report.require(bool(set_roots), "no embedding set was produced")

        for root in set_roots:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            print(f"[set]     {root.name}: {len(rows)} vector(s), dim {manifest['dim']}")

            check_manifest(manifest, report)
            check_coverage(manifest, rows, report)
            check_digests(root, manifest, report)
            check_vectors(root, manifest, rows, report)
            check_binding(manifest, rows, chunks, report)

            embedding_set = search.EmbeddingSet(root)
            check_discrimination(embedding_set, report)
            queries = check_row_alignment(embedding_set, chunks, search, report)
            check_native_differential(
                embedding_set, queries, search, report, require=args.require_native
            )

            twin = embed_dir_b / root.name
            if manifest["deterministic"]:
                for name in ("vectors.f32", "index.jsonl", "vectors.q15"):
                    left, right = root / name, twin / name
                    if not left.is_file():
                        continue
                    report.require(
                        right.is_file() and left.read_bytes() == right.read_bytes(),
                        f"determinism: {root.name}/{name} differs between two builds "
                        "though the set declares deterministic: true",
                    )
                print(f"[det]     {root.name}: byte-identical across two builds")
            else:
                report.skip(
                    f"{root.name} declares deterministic: false; gated on invariants, "
                    "not byte-identity"
                )

        if args.keep:
            if args.keep.exists():
                shutil.rmtree(args.keep)
            shutil.copytree(workspace / "a", args.keep)
            print(f"[keep]    {args.keep}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    for message in report.skips:
        print(f"[skip]    {message}")

    if report.failures:
        print("embedding gate: FAILED", file=sys.stderr)
        for failure in report.failures[:40]:
            print(f"  - {failure}", file=sys.stderr)
        if len(report.failures) > 40:
            print(f"  ... and {len(report.failures) - 40} more", file=sys.stderr)
        return 1

    print(f"embedding gate: PASSED ({report.checks} checks, {len(report.skips)} honest skip(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
