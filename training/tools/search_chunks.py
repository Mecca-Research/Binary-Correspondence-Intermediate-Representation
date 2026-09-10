#!/usr/bin/env python3
"""Retrieve corpus chunks from an embedding set (Phase 0.5).

This is what the vectors are *for*, and it is deliberately a real tool rather
than an example: a rail that builds vectors nothing ever queries has not been
shown to work.

Three backends, and the difference between them matters:

  * ``reference`` -- exact Q15 squared-L2 in pure Python. Always available, no
    compiler, no third-party package. This is the definition.
  * ``native``    -- the same ranking through BCIR's own ``bcir_ai_q15_topk``
    from ``runtime/c``, the exact-integer top-k kernel the oracle uses for its
    own optimization memory.
  * ``q8``        -- a *different arithmetic*: the corpus matrix quantized by
    ``bcir_ai_quantize_q8_f64`` into BCIRQ8 -- per-group power-of-two exponents
    and signed 8-bit codes, the format BCIR ships model weights in -- and scored
    by ``bcir_ai_q8_rows_dot_f64``, the kernel whose header names embedding
    projections as its purpose. Half the bytes per coordinate, and lossy.

``--backend both`` runs ``reference`` and ``native`` and requires them to agree
exactly. That is the differential this repository prefers to an assertion: two
independent implementations of one contract, checked against each other on real
data rather than on a fixture. Because both work in exact integer arithmetic,
"agree" means identical indices *and* identical squared distances -- there is no
tolerance to tune and no floating-point tie-break to excuse a mismatch.

``q8`` is deliberately NOT in that equality: requiring a lossier view to rank
identically would be requiring quantization not to quantize. The gate instead
requires the one thing loss must not break -- a query that is exactly some
chunk's vector still ranks that chunk first -- and *measures* the rest.

**On the direction of the dependency.** `training/` is never a build dependency
of BCIR, and that is unchanged here: this tool imports BCIR lazily, only when
the native backend is asked for, and the reference backend is always sufficient
on its own. The corpus may call the implementation it teaches; the
implementation still knows nothing about the corpus.

    python3 training/tools/search_chunks.py --query "how do opaque pointers change getelementptr"
    python3 training/tools/search_chunks.py --query "musttail" --backend both --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from operator import mul
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

DEFAULT_SET = Path("build/training/embeddings/lexical-hash-v1")
DEFAULT_CHUNKS = Path("build/training/chunks")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_BACKEND_UNAVAILABLE = 3


def _native():
    """Load the corpus's single door to BCIR's kernels, on demand."""
    global _NATIVE
    if _NATIVE is None:
        _NATIVE = _load_tool("bcir_native")
    return _NATIVE


def _load_tool(name: str):
    import importlib.util

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_NATIVE = None
_EMBED = None


class BackendUnavailable(RuntimeError):
    """The native rail is not built here. An honest skip, never a silent switch.

    Kept as this module's name for the condition; `bcir_native` raises its own,
    and `_native_unavailable` is the predicate both are checked through so a
    caller never has to know which door raised.
    """


def _native_unavailable() -> tuple:
    """Every exception type that means "the native rail is not reachable"."""
    return (BackendUnavailable, _native().BackendUnavailable)


# --------------------------------------------------------------------------
# Loading a set
# --------------------------------------------------------------------------


class EmbeddingSet:
    def __init__(self, root: Path) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(
                f"search_chunks: no embedding set at {root}\n"
                "  build one first: python3 training/tools/embed_chunks.py"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.dim = int(self.manifest["dim"])
        self.rows = [
            json.loads(line)
            for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        quantized = self.manifest.get("quantized")
        if not quantized:
            raise SystemExit(
                f"search_chunks: {root} has no quantized view; retrieval here is "
                "defined over BCIR's Q15 code space"
            )
        codes = array("h")
        codes.frombytes((root / quantized["path"]).read_bytes())
        if sys.byteorder == "big":
            codes.byteswap()  # the file is little-endian by contract
        expected = len(self.rows) * self.dim
        if len(codes) != expected:
            raise SystemExit(
                f"search_chunks: {quantized['path']} holds {len(codes)} codes, "
                f"expected {expected} ({len(self.rows)} rows x dim {self.dim})"
            )
        self.codes = codes
        self.scale = int(quantized["scale"])
        # Materialize the rows once. `array` slicing copies, and an iterator
        # like islice cannot seek -- it would re-walk the prefix on every row,
        # turning a linear scan into a quadratic one.
        self.row_views = [
            codes[row * self.dim : (row + 1) * self.dim] for row in range(len(self.rows))
        ]
        # Squared length of each row, once. See topk_reference for why.
        self.row_squares = [sum(map(mul, view, view)) for view in self.row_views]

        self._q8 = None

    def row_codes(self, row: int) -> array:
        return self.row_views[row]

    def float_vectors(self) -> array:
        """The unit vectors themselves, as stored before quantization."""
        values = array("f")
        values.frombytes((self.root / self.manifest["vectors"]["path"]).read_bytes())
        if sys.byteorder == "big":
            values.byteswap()
        return values

    def q8_tensor(self):
        """Quantize the corpus matrix with BCIR's bridge, once per set.

        Built from the f32 vectors rather than re-quantizing the Q15 codes: Q8
        of a Q15 approximation would compound two roundings, and BCIR's bridge
        is specified over the real values.
        """
        if self._q8 is None:
            native = _native()
            self._q8 = native.quantize_q8(list(self.float_vectors()))
        return self._q8


def load_chunk_texts(chunk_dir: Path) -> dict[str, dict]:
    texts: dict[str, dict] = {}
    if not chunk_dir.is_dir():
        return texts
    for path in sorted(chunk_dir.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = json.loads(line)
                texts[chunk["chunk_id"]] = chunk
    return texts


# --------------------------------------------------------------------------
# Query projection
# --------------------------------------------------------------------------


def embed_query_float(text: str, embedding_set: EmbeddingSet) -> list[float]:
    """Project a query with the set's own model, as a unit vector."""
    global _EMBED
    if _EMBED is None:
        _EMBED = _load_tool("embed_chunks")

    manifest = embedding_set.manifest
    provider = _EMBED.build_provider(
        manifest["model"], dim=embedding_set.dim, revision=manifest["revision"]
    )
    if provider.dim != embedding_set.dim:
        raise SystemExit(
            f"search_chunks: {manifest['model']} reports dim {provider.dim}, "
            f"but the set was built at dim {embedding_set.dim}"
        )
    vectors, _ = _EMBED.embed(
        [{"text": text, "chunk_id": "<query>", "source_path": "<query>"}], provider
    )
    return vectors[0]


def embed_query(text: str, embedding_set: EmbeddingSet) -> array:
    """The same projection, in the Q15 space the top-k kernels work over."""
    if _EMBED is None:
        embed_query_float(text, embedding_set)
    return _EMBED.quantize_q15([embed_query_float(text, embedding_set)])


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


def topk_reference(query: array, embedding_set: EmbeddingSet, top_k: int) -> list[tuple[int, int]]:
    """Exact Q15 squared-L2 top-k. Ties break by row, matching the C contract.

    Expanded rather than summed term by term:

        ||q - p||^2  =  ||q||^2 + ||p||^2 - 2 (q . p)

    Every term is an integer, so this is not an approximation of the direct
    form -- it is the same value by integer algebra, which is what lets the
    result be compared to the C kernel's for exact equality rather than within
    a tolerance. The payoff is that the only per-row work is one dot product,
    and ||p||^2 is computed once per set instead of once per query.
    """
    row_squares = embedding_set.row_squares
    query_square = sum(map(mul, query, query))

    scored: list[tuple[int, int]] = []
    for row, view in enumerate(embedding_set.row_views):
        dot = sum(map(mul, query, view))
        scored.append((query_square + row_squares[row] - 2 * dot, row))
    scored.sort()
    return [(row, distance) for distance, row in scored[:top_k]]


def topk_native(query: array, embedding_set: EmbeddingSet, top_k: int) -> list[tuple[int, int]]:
    """The same ranking through BCIR's own `bcir_ai_q15_topk`."""
    return _native().q15_topk(
        query,
        embedding_set.codes,
        rows=len(embedding_set.rows),
        dim=embedding_set.dim,
        top_k=top_k,
    )


def topk_q8(query: list[float], embedding_set: EmbeddingSet, top_k: int) -> list[tuple[int, float]]:
    """Rank by cosine through BCIRQ8, the format BCIR ships model weights in.

    A different arithmetic from the other two backends, not a third spelling of
    it: the corpus matrix is quantized by `bcir_ai_quantize_q8_f64` into
    per-group power-of-two exponents and 8-bit codes, then scored by
    `bcir_ai_q8_rows_dot_f64` -- the kernel whose header names embedding
    projections as its purpose. Eight bits per coordinate instead of sixteen.

    It is **lossier by construction**, so its ranking is not required to equal
    the exact Q15 one and no gate asserts that it does. What is worth knowing is
    how often it agrees, and that is measured and reported rather than assumed.
    """
    native = _native()
    tensor = embedding_set.q8_tensor()
    scores = native.q8_rows_dot(list(query), tensor, rows=len(embedding_set.rows))
    ranked = sorted(((-score, row) for row, score in enumerate(scores)))
    return [(row, -negated) for negated, row in ranked[:top_k]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET, dest="embedding_set")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=("reference", "native", "q8", "both"),
        default="reference",
        help="reference = exact Q15 in pure Python (the definition); native = the "
        "same ranking through bcir_ai_q15_topk; q8 = BCIRQ8 cosine through "
        "bcir_ai_q8_rows_dot_f64, lossier by construction; both = reference "
        "and native, required to agree exactly",
    )
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="fail instead of skipping when the native rail cannot be built; pass "
        "this from the CI job that provides a C compiler",
    )
    args = parser.parse_args(argv)

    if args.top_k < 1:
        print("search_chunks: --top-k must be at least 1", file=sys.stderr)
        return EXIT_USAGE

    embedding_set = EmbeddingSet(args.embedding_set)
    top_k = min(args.top_k, len(embedding_set.rows))
    unit_query = embed_query_float(args.query, embedding_set)
    query = embed_query(args.query, embedding_set)

    reference = native = None
    if args.backend in ("reference", "both"):
        reference = topk_reference(query, embedding_set, top_k)
    if args.backend in ("native", "q8", "both"):
        try:
            if args.backend == "q8":
                # Scores here are cosines, not squared distances: rank by them
                # descending. Reported through the same (row, score) shape.
                native = [(row, score) for row, score in topk_q8(unit_query, embedding_set, top_k)]
            else:
                native = topk_native(query, embedding_set, top_k)
        except _native_unavailable() as exc:
            if args.require_native:
                print(f"search_chunks: native backend required: {exc}", file=sys.stderr)
                return EXIT_BACKEND_UNAVAILABLE
            print(f"[skip]   native backend unavailable: {exc}")
            if reference is None:
                return EXIT_OK

    if reference is not None and native is not None and args.backend == "both":
        if reference != native:
            print("search_chunks: reference and native rankings DISAGREE", file=sys.stderr)
            print(f"  reference: {reference}", file=sys.stderr)
            print(f"  native:    {native}", file=sys.stderr)
            return EXIT_FAILED
        print(f"[differential] reference == native on {len(reference)} result(s), exactly")

    results = reference if reference is not None else native
    assert results is not None
    texts = load_chunk_texts(args.chunks)

    print(
        f"\nQ: {args.query!r}   [{embedding_set.manifest['model']}, "
        f"semantics={embedding_set.manifest['semantics']}]"
    )
    for rank, (row, distance) in enumerate(results, start=1):
        entry = embedding_set.rows[row]
        chunk = texts.get(entry["chunk_id"])
        # Squared Q15 distance is exact but unit-free; cosine is what a reader
        # can compare across queries. The identity is exact for unit vectors.
        cosine = (
            distance
            if args.backend == "q8"
            else 1.0 - distance / (2.0 * embedding_set.scale * embedding_set.scale)
        )
        trail = " > ".join(chunk["heading_trail"]) if chunk and chunk["heading_trail"] else ""
        location = (
            f"{entry['source_path']}:{chunk['span']['start_line']}"
            if chunk
            else entry["source_path"]
        )
        print(f"  {rank}. cos={cosine:+.3f}  {location}")
        if trail:
            print(f"     {trail}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
