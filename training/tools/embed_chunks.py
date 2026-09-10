#!/usr/bin/env python3
"""Fill Tier-2 chunk vectors with a named embedding model (Phase 0.5).

The chunker deliberately leaves `embedding: null`. This is the separate,
explicitly-named step that fills it -- separate because a chunk is a
deterministic function of the corpus alone, while a vector is a function of the
corpus *and* a model. Binding them into one file would make every chunk record
change whenever a model did, and would make it impossible to hold two models'
vectors for one corpus.

So vectors live in a sidecar embedding set:

    build/training/embeddings/<model-slug>/
      manifest.json    the model, its revision, what it covers, and the digests
      index.jsonl      one row per vector: chunk_id, row, text digest, raw norm
      vectors.f32      raw little-endian float32, row-major, dim floats per row

`--inline` additionally joins the vectors back into chunk records for consumers
that want the single-file shape the chunk schema describes.

**The rule this tool exists to keep.** A vector is written only by a `Provider`,
and the manifest's model/revision/dim/semantics fields are read off that same
provider object. There is deliberately no code path that produces a vector
without attributing it, because a vector nobody can attribute is
indistinguishable downstream from one a real model produced, and it silently
poisons every similarity computed against it.

**The declaration this tool adds.** Naming the model is necessary but not
sufficient: a hashed-n-gram vector and a trained sentence encoder are both
"named", and they are not interchangeable. Every set therefore declares
`semantics` -- `lexical` (a specified function of the surface text) or `learned`
(trained weights) -- so a consumer can weight them differently instead of
discovering the difference in its retrieval quality.

    # the hermetic baseline: no dependencies, bit-identical anywhere
    python3 training/tools/embed_chunks.py --chunks build/training/chunks \
                                           --out build/training/embeddings

    # a trained model, when the environment has one
    python3 training/tools/embed_chunks.py --model sentence-transformers/all-MiniLM-L6-v2 \
                                           --require-provider
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

SCHEMA = "bcir-training/embedding-set/v1"
CHUNK_SCHEMA = "bcir-training/chunk/v1"
LICENSE = "LicenseRef-BCIR-NC-1.0"
BUILDER = "training/tools/embed_chunks.py"
CHUNK_BUILDER = "training/tools/build_chunks.py"

DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_OUT = Path("build/training/embeddings")

# BCIR's own Q15 convention, reused rather than reinvented: signed int16, and
# -32768 is forbidden so the code space stays symmetric about zero -- the same
# discipline runtime/c applies to Q8 (-128 forbidden) and Q4 (-8 forbidden).
# Reusing it is what lets the corpus's vectors be searched by the very kernel
# BCIR uses for its own optimization memory, `bcir_ai_q15_topk`, instead of by a
# second convention this corpus would have to defend on its own.
Q15_SCALE = 32767
Q15_KERNEL = "bcir_ai_q15_topk"

# Exit codes are part of this tool's contract: CI distinguishes "the optional
# rail is absent here" from "the rail this job installed is broken".
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_PROVIDER_UNAVAILABLE = 3


class ProviderUnavailable(RuntimeError):
    """The named model cannot be loaded here. Honest skip, never a fake vector."""


class Provider(Protocol):
    """What every embedding model must declare before it may write a vector."""

    name: str
    revision: str
    dim: int
    normalize: str
    semantics: str
    deterministic: bool

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one raw (un-normalized) vector per text, in order."""


# --------------------------------------------------------------------------
# The hermetic baseline
# --------------------------------------------------------------------------

# Explicitly ASCII, not \w: `\w` without re.ASCII matches Unicode digits and
# letters, which is the same class of defect the wire-format decoders in this
# repository are gated against. A tokenizer that silently accepts a wider
# language than it documents produces vectors nobody can reproduce.
_WORD_RE = re.compile(r"[a-z0-9_]+", re.ASCII)

_CHAR_NGRAM = 4


class LexicalHashProvider:
    """Signed-hash n-gram embedding: specified, dependency-free, reproducible.

    The algorithm, completely, so any consumer can reimplement it and get the
    same vectors:

      1. Normalize: NFKC, casefold, collapse whitespace runs to one space.
      2. Extract features: word unigrams and bigrams over `[a-z0-9_]+`, plus
         character 4-grams over the normalized text. Character n-grams are what
         make this usable on a compiler corpus -- they match `getelementptr`
         against `getelementptr inbounds` where word tokens alone would not.
      3. Weight each feature `1 + ln(count)`: sublinear, so a term repeated
         thirty times in one chunk does not dominate its direction.
      4. Hash each feature with BLAKE2b (personalized `bcir-emb`, 9 bytes):
         the first 8 bytes choose the dimension, the 9th byte's low bit chooses
         the sign. Signed hashing makes collisions cancel in expectation rather
         than always accumulate.
      5. Accumulate in sorted feature order, then L2-normalize.

    **What it is not.** This is a lexical model: it scores near-duplicate text
    highly and a paraphrase not at all. It carries no trained semantics, and it
    declares `semantics = "lexical"` so no consumer has to find that out by
    experiment.

    **A deliberate omission: no IDF.** Inverse document frequency would improve
    retrieval measurably, and it would also make every vector a function of the
    whole corpus -- so adding one chapter would invalidate every previously
    computed vector, and no chunk's vector could be recomputed in isolation. The
    per-text property is worth more here than the ranking gain, and the learned
    providers are the answer for retrieval quality.
    """

    name = "lexical-hash-v1"
    revision = "1"
    normalize = "l2"
    semantics = "lexical"
    deterministic = True

    def __init__(self, dim: int = 512) -> None:
        if dim < 32:
            raise ValueError("lexical-hash-v1 needs dim >= 32; collisions swamp it below that")
        self.dim = dim

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    @classmethod
    def _features(cls, text: str) -> Counter[str]:
        norm = cls._normalize_text(text)
        counts: Counter[str] = Counter()

        tokens = _WORD_RE.findall(norm)
        for token in tokens:
            counts[f"w:{token}"] += 1
        for left, right in zip(tokens, tokens[1:]):
            counts[f"b:{left} {right}"] += 1
        for start in range(max(0, len(norm) - _CHAR_NGRAM + 1)):
            counts[f"c:{norm[start : start + _CHAR_NGRAM]}"] += 1
        return counts

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(
            feature.encode("utf-8"), digest_size=9, person=b"bcir-emb"
        ).digest()
        index = int.from_bytes(digest[:8], "big") % self.dim
        sign = 1.0 if digest[8] & 1 == 0 else -1.0
        return index, sign

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            # Sorted, so the float accumulation order is fixed and the result is
            # bit-identical across runs and hosts.
            for feature, count in sorted(self._features(text).items()):
                index, sign = self._bucket(feature)
                vector[index] += sign * (1.0 + math.log(count))
            vectors.append(vector)
        return vectors


# --------------------------------------------------------------------------
# Learned models
# --------------------------------------------------------------------------


class SentenceTransformerProvider:
    """A trained sentence encoder, when the environment actually has one.

    Absent weights or package are a `ProviderUnavailable` skip, never a
    substituted vector: falling back to a different model while keeping the
    requested model's name in the manifest would be the exact lie this rail
    exists to prevent.

    `deterministic` is False and that is not a defect. A learned model's output
    depends on kernel selection, thread count, and hardware; contracting for
    bit-identity would be a promise the rail cannot keep. Such a set is gated on
    its invariants -- finite, correct dimension, unit norm, non-degenerate --
    rather than on byte-identity.
    """

    normalize = "l2"
    semantics = "learned"
    deterministic = False

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised by absence
            raise ProviderUnavailable(
                f"sentence-transformers is not installed ({exc}); "
                "install it, or use the hermetic lexical-hash-v1 provider"
            ) from exc

        try:
            self._model = SentenceTransformer(model_id, revision=revision)
        except Exception as exc:  # noqa: BLE001 - any load failure is a skip
            raise ProviderUnavailable(f"could not load {model_id!r}: {exc}") from exc

        self.name = model_id
        self.revision = revision or self._resolve_revision(model_id)
        dim = self._model.get_sentence_embedding_dimension()
        if not dim:
            raise ProviderUnavailable(f"{model_id!r} did not report an embedding dimension")
        self.dim = int(dim)

    @staticmethod
    def _resolve_revision(model_id: str) -> str:
        """Pin to the exact weights, or refuse.

        An unpinned revision is a set nobody can reproduce: the same manifest
        would describe different vectors a month later. If the concrete commit
        cannot be resolved, that is a usage error to be fixed with --revision,
        not a detail to paper over with "main".
        """
        try:
            from huggingface_hub import model_info

            sha = model_info(model_id).sha
            if sha:
                return str(sha)
        except Exception:  # noqa: BLE001 - fall through to the explicit refusal
            pass
        raise ProviderUnavailable(
            f"could not resolve a pinned revision for {model_id!r}; "
            "pass --revision <commit-sha> so the set names exactly which weights produced it"
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False
        )
        return [[float(value) for value in row] for row in vectors]


def build_provider(model: str, *, dim: int, revision: str | None) -> Provider:
    if model == LexicalHashProvider.name:
        return LexicalHashProvider(dim=dim)
    if model.startswith("sentence-transformers/") or "/" in model:
        return SentenceTransformerProvider(model, revision=revision)
    raise SystemExit(
        f"embed_chunks: unknown model {model!r}; "
        f"use {LexicalHashProvider.name!r} or a sentence-transformers model id"
    )


# --------------------------------------------------------------------------
# Set construction
# --------------------------------------------------------------------------


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def read_chunks(chunk_dir: Path, subject: str | None) -> list[dict]:
    if not chunk_dir.is_dir():
        raise SystemExit(
            f"embed_chunks: no chunk build at {chunk_dir}\n"
            f"  build one first: python3 {CHUNK_BUILDER} --out {chunk_dir}"
        )
    pattern = f"{subject}.chunks.jsonl" if subject else "*.chunks.jsonl"
    chunks: list[dict] = []
    for path in sorted(chunk_dir.glob(pattern)):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"embed_chunks: {path.name}:{number}: {exc}") from exc
    if not chunks:
        raise SystemExit(
            f"embed_chunks: {chunk_dir} contains no chunks"
            + (f" for subject {subject!r}" if subject else "")
            + " -- embedding nothing would produce a set that passes every check "
            "by covering no corpus"
        )
    # Stable row order, independent of filesystem enumeration.
    chunks.sort(
        key=lambda c: (c["subject"], c["source_path"], c["span"]["start_line"], c["chunk_id"])
    )
    return chunks


def corpus_digest(chunks: Iterable[dict]) -> str:
    """A digest of what is covered, so a stale set is detectable in one compare."""
    identity = [
        {"chunk_id": chunk["chunk_id"], "source_sha256": chunk["source_sha256"]} for chunk in chunks
    ]
    identity.sort(key=lambda row: row["chunk_id"])
    return sha256_text(canonical_json(identity))


def l2_norm(vector: list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def pack_float32(vectors: list[list[float]]) -> bytes:
    flat = array("f", (value for vector in vectors for value in vector))
    if sys.byteorder == "big":
        flat.byteswap()  # the manifest promises little-endian on every host
    return flat.tobytes()


def quantize_q15(vectors: list[list[float]]) -> array:
    """Project unit vectors onto BCIR's symmetric Q15 code space.

    Valid only for L2-normalized vectors, whose coordinates lie in [-1, 1]. For
    two unit vectors, squared L2 distance and cosine similarity are related by
    ||a - b||^2 = 2 - 2*cos(a, b) -- a strictly decreasing function -- so an
    exact squared-L2 ranking *is* a cosine ranking. That is what lets corpus
    retrieval run on `bcir_ai_q15_topk`, which computes squared distances in
    exact integer arithmetic and therefore has no floating-point tie-break to
    disagree about across hosts.
    """
    codes = array("h")
    for vector in vectors:
        for value in vector:
            code = int(round(value * Q15_SCALE))
            # Symmetric clamp. -32768 has no positive counterpart, so admitting
            # it would make negation lossy for exactly one code.
            codes.append(max(-Q15_SCALE, min(Q15_SCALE, code)))
    return codes


def pack_q15(codes: array) -> bytes:
    packed = array("h", codes)
    if sys.byteorder == "big":
        packed.byteswap()
    return packed.tobytes()


def embed(chunks: list[dict], provider: Provider) -> tuple[list[list[float]], list[float]]:
    """Encode, validate, and normalize. Never emits a vector it cannot defend."""
    raw = provider.encode([chunk["text"] for chunk in chunks])
    if len(raw) != len(chunks):
        raise SystemExit(
            f"embed_chunks: {provider.name} returned {len(raw)} vectors for {len(chunks)} chunks"
        )

    vectors: list[list[float]] = []
    raw_norms: list[float] = []
    for chunk, vector in zip(chunks, raw):
        if len(vector) != provider.dim:
            raise SystemExit(
                f"embed_chunks: {provider.name} returned dim {len(vector)} for "
                f"{chunk['chunk_id']}, but declares {provider.dim}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise SystemExit(
                f"embed_chunks: {provider.name} produced a non-finite value for "
                f"{chunk['chunk_id']}; a NaN poisons every comparison it enters"
            )

        norm = l2_norm(vector)
        if norm == 0.0:
            # Normalizing would divide by zero; storing zeros would hide it. A
            # direction-less vector is the vector equivalent of recording an
            # unreadable counter as 0.
            raise SystemExit(
                f"embed_chunks: {provider.name} extracted no signal from "
                f"{chunk['chunk_id']} ({chunk['source_path']}); a zero vector has no "
                "direction and cosine similarity against it is undefined"
            )

        raw_norms.append(norm)
        if provider.normalize == "l2":
            vectors.append([value / norm for value in vector])
        else:
            vectors.append(list(vector))
    return vectors, raw_norms


def write_set(
    destination: Path,
    *,
    chunks: list[dict],
    vectors: list[list[float]],
    raw_norms: list[float],
    provider: Provider,
    subjects: list[str],
    chunks_total: int,
    skipped_reason: str | None,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)

    index_lines = [
        canonical_json(
            {
                "chunk_id": chunk["chunk_id"],
                "subject": chunk["subject"],
                "source_path": chunk["source_path"],
                "row": row,
                "text_sha256": sha256_text(chunk["text"]),
                "raw_norm": norm,
            }
        )
        for row, (chunk, norm) in enumerate(zip(chunks, raw_norms))
    ]
    index_payload = ("\n".join(index_lines) + "\n").encode("utf-8")
    vector_payload = pack_float32(vectors)

    (destination / "index.jsonl").write_bytes(index_payload)
    (destination / "vectors.f32").write_bytes(vector_payload)

    # The BCIR-native view. Only meaningful for unit vectors, so a set that
    # declines to normalize declares no quantized view rather than emitting one
    # whose coordinates would saturate.
    quantized: dict | None = None
    if provider.normalize == "l2":
        q15_payload = pack_q15(quantize_q15(vectors))
        (destination / "vectors.q15").write_bytes(q15_payload)
        quantized = {
            "path": "vectors.q15",
            "dtype": "int16",
            "byte_order": "little",
            "scale": Q15_SCALE,
            "symmetric": True,
            "kernel": Q15_KERNEL,
            "sha256": sha256_bytes(q15_payload),
        }

    skipped = chunks_total - len(chunks)
    manifest = {
        "schema": SCHEMA,
        "corpus": "training",
        "model": provider.name,
        "revision": provider.revision,
        "dim": provider.dim,
        "normalize": provider.normalize,
        "semantics": provider.semantics,
        "deterministic": provider.deterministic,
        "builder": BUILDER,
        "built_from": {
            "chunk_builder": CHUNK_BUILDER,
            "chunk_schema": CHUNK_SCHEMA,
            "corpus_digest": corpus_digest(chunks),
            "chunks_total": chunks_total,
            "subjects": subjects,
        },
        "coverage": {
            "embedded": len(chunks),
            "skipped": skipped,
            "reason": skipped_reason if skipped else None,
        },
        "vectors": {
            "path": "vectors.f32",
            "dtype": "float32",
            "byte_order": "little",
            "rows": len(vectors),
            "sha256": sha256_bytes(vector_payload),
        },
        "quantized": quantized,
        "index": {
            "path": "index.jsonl",
            "sha256": sha256_bytes(index_payload),
            "row_schema": "training/schema/embedding-set-v1.json#/$defs/embeddingRow",
        },
        "license": LICENSE,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def write_inline(
    destination: Path, chunks: list[dict], vectors: list[list[float]], provider: Provider
) -> None:
    """Join vectors back into chunk records, the shape chunk-v1 describes inline."""
    destination.mkdir(parents=True, exist_ok=True)
    by_subject: dict[str, list[str]] = {}
    for chunk, vector in zip(chunks, vectors):
        joined = dict(chunk)
        joined["embedding"] = vector
        joined["embedding_spec"] = {
            "model": provider.name,
            "revision": provider.revision,
            "dim": provider.dim,
            "normalize": provider.normalize,
            "semantics": provider.semantics,
        }
        by_subject.setdefault(chunk["subject"], []).append(canonical_json(joined))

    for subject, lines in sorted(by_subject.items()):
        path = destination / f"{subject}.chunks.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[inline] {path} ({len(lines)} chunk(s) with vectors)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS, help="chunk build directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="embedding set root")
    parser.add_argument("--model", default=LexicalHashProvider.name)
    parser.add_argument("--revision", help="pin a learned model to an exact weights revision")
    parser.add_argument("--dim", type=int, default=512, help="lexical-hash-v1 dimension")
    parser.add_argument("--subject", help="embed one subject instead of all")
    parser.add_argument(
        "--inline", type=Path, help="also write chunk records with vectors joined in"
    )
    parser.add_argument(
        "--require-provider",
        action="store_true",
        help="fail instead of skipping when the model cannot be loaded; pass this "
        "from the CI job that installed it, where absence is a defect rather "
        "than an expected local condition",
    )
    args = parser.parse_args(argv)

    try:
        provider = build_provider(args.model, dim=args.dim, revision=args.revision)
    except ProviderUnavailable as exc:
        if args.require_provider:
            print(f"embed_chunks: provider required but unavailable: {exc}", file=sys.stderr)
            return EXIT_PROVIDER_UNAVAILABLE
        print(f"[skip]    {args.model}: {exc}")
        print("[skip]    no set written -- a skip is never a pass")
        return EXIT_OK
    except ValueError as exc:
        print(f"embed_chunks: {exc}", file=sys.stderr)
        return EXIT_USAGE

    chunks = read_chunks(args.chunks, args.subject)
    subjects = sorted({chunk["subject"] for chunk in chunks})

    vectors, raw_norms = embed(chunks, provider)
    # A single-subject set gets its own destination. Both manifests are honest
    # about what they cover, but writing a partial set over a whole-corpus one
    # would leave two different sets sharing a path, distinguishable only by
    # opening them.
    slug = model_slug(provider.name) + (f".{args.subject}" if args.subject else "")
    destination = args.out / slug
    manifest = write_set(
        destination,
        chunks=chunks,
        vectors=vectors,
        raw_norms=raw_norms,
        provider=provider,
        subjects=subjects,
        chunks_total=len(chunks),
        skipped_reason=None,
    )

    if args.inline:
        write_inline(args.inline, chunks, vectors, provider)

    print(
        f"[embed]   {provider.name} rev {provider.revision} "
        f"dim={provider.dim} semantics={provider.semantics} "
        f"deterministic={str(provider.deterministic).lower()}"
    )
    print(f"[write]   {destination} ({manifest['coverage']['embedded']} vector(s))")
    print(f"[cover]   {', '.join(subjects)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
