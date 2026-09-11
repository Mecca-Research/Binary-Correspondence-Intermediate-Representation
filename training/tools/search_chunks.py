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
from dataclasses import dataclass
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from db import analytics, engine, relational  # noqa: E402  (path set above)

CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

DEFAULT_SET = Path("build/training/embeddings/lexical-hash-v1")
DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_CATALOG = Path("build/training/catalog")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_BACKEND_UNAVAILABLE = 3


def _native():
    """Load the corpus's single door to BCIR's kernels, on demand."""
    global _NATIVE
    if _NATIVE is None:
        _NATIVE = engine.load("bcir_native")
    return _NATIVE


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

        # Derived columns are declared here and computed by the first reader that
        # actually wants them -- see `row_views` for which backends never do.
        self._row_views: list[array] | None = None
        self._row_squares: list[int] | None = None
        self._q8 = None

    # ----------------------------------------------------------------------
    # Derived columns
    #
    # Both are a pure function of `codes`, and both used to be built in the
    # constructor for every caller. Exactly one of the three backends reads them:
    # `topk_native` hands `codes` to the C kernel whole and `topk_q8` works from
    # the f32 vectors, so for those two the work was the largest single cost in
    # the command and bought nothing. Computing a derived column on demand is the
    # cheapest form of the thing a column store does deliberately.
    # ----------------------------------------------------------------------

    @property
    def row_views(self) -> list[array]:
        """Every row as its own array, materialized together on first use.

        Together, because `array` slicing copies and an iterator like islice cannot
        seek -- it would re-walk the prefix on every row, turning a linear scan into
        a quadratic one. A caller that wants *one* row wants `row_codes` instead.
        """
        if self._row_views is None:
            dim = self.dim
            codes = self.codes
            self._row_views = [codes[row * dim : (row + 1) * dim] for row in range(len(self.rows))]
        return self._row_views

    @property
    def row_squares(self) -> list[int]:
        """``||p||^2`` per row. See `topk_reference` for why the value is needed."""
        if self._row_squares is None:
            self._row_squares = [sum(map(mul, view, view)) for view in self.row_views]
        return self._row_squares

    def row_codes(self, row: int) -> array:
        """One row's codes, without materializing the column it lives in.

        A point lookup slices `codes` directly; going through `row_views` would build
        every row to return one. The bounds check is not decoration: `array` slicing
        is total, so an out-of-range row would otherwise return a short or empty
        array and rank it, which is the silent-wrong-answer shape this rail refuses
        everywhere else.
        """
        if not 0 <= row < len(self.rows):
            raise IndexError(f"row {row} is outside this set's {len(self.rows)} rows")
        if self._row_views is not None:
            return self._row_views[row]
        start = row * self.dim
        return self.codes[start : start + self.dim]

    def derived_columns_built(self) -> tuple[str, ...]:
        """Which derived columns this set has actually computed, for gates to assert."""
        built = []
        if self._row_views is not None:
            built.append("row_views")
        if self._row_squares is not None:
            built.append("row_squares")
        if self._q8 is not None:
            built.append("q8")
        return tuple(built)

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
        _EMBED = engine.load("embed_chunks")

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


def topk_reference(
    query: array, embedding_set: EmbeddingSet, top_k: int, rows=None
) -> list[tuple[int, int]]:
    """Exact Q15 squared-L2 top-k. Ties break by row, matching the C contract.

    Expanded rather than summed term by term:

        ||q - p||^2  =  ||q||^2 + ||p||^2 - 2 (q . p)

    Every term is an integer, so this is not an approximation of the direct
    form -- it is the same value by integer algebra, which is what lets the
    result be compared to the C kernel's for exact equality rather than within
    a tolerance. The payoff is that the only per-row work is one dot product,
    and ||p||^2 is computed once per set instead of once per query.
    """
    query_square = sum(map(mul, query, query))
    scored: list[tuple[int, int]] = []

    if rows is None:
        # Every row: build the derived columns once, because each is read once per
        # row and `array` slicing in a loop would otherwise re-walk the prefix.
        row_squares = embedding_set.row_squares
        for row, view in enumerate(embedding_set.row_views):
            dot = sum(map(mul, query, view))
            scored.append((query_square + row_squares[row] - 2 * dot, row))
    else:
        # A selection: slice only the rows the predicate admitted and recompute
        # ``||p||^2`` for each. Building the whole column to read a fraction of it is
        # the materialize-before-you-select mistake -- and the arithmetic is the same
        # integers either way, so the two paths agree bit for bit.
        for row in rows:
            view = embedding_set.row_codes(row)
            dot = sum(map(mul, query, view))
            scored.append((query_square + sum(map(mul, view, view)) - 2 * dot, row))

    scored.sort()
    return [(row, distance) for distance, row in scored[:top_k]]


def topk_native(
    query: array, embedding_set: EmbeddingSet, top_k: int, rows=None
) -> list[tuple[int, int]]:
    """The same ranking through BCIR's own `bcir_ai_q15_topk`.

    A selection becomes the kernel's `eligible` mask, so a rejected row never has its
    dot product computed. The kernel returns fewer than `top_k` matches when the mask
    admits fewer -- that shorter result is the honest answer, not a truncation.
    """
    return _native().q15_topk(
        query,
        embedding_set.codes,
        rows=len(embedding_set.rows),
        dim=embedding_set.dim,
        top_k=top_k,
        eligible=rows,
    )


def topk_q8(
    query: list[float], embedding_set: EmbeddingSet, top_k: int, rows=None
) -> list[tuple[int, float]]:
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
    # `bcir_ai_q8_rows_dot_f64` takes no eligibility mask -- it is a rows-dot, not a
    # top-k -- so a selection here filters *after* the scan rather than inside it.
    # The planner prices that honestly: a q8 plan scans every row whatever the
    # predicate says, which is one of the reasons a narrow predicate makes the
    # native backend win.
    candidates = enumerate(scores) if rows is None else ((row, scores[row]) for row in rows)
    ranked = sorted((-score, row) for row, score in candidates)
    return [(row, -negated) for negated, row in ranked[:top_k]]


def _catalog(args) -> tuple[object, str]:
    """The catalog, or a stated reason there is none. Never a silent degradation.

    A missing or stale catalog does not make a query wrong -- it makes it read the
    whole chunk table to render `top_k` rows, and lose `--where` entirely. Both are
    losses a reader should be told about, so the reason is returned and printed
    rather than swallowed.
    """
    module = engine.catalog()
    try:
        return module.Catalog.load(args.catalog, args.chunks), ""
    except module.CatalogError as exc:
        return None, str(exc).splitlines()[0]


def _print_grouped_aggregate(plan_module, catalog, args, selection, where: str) -> int:
    """GROUP BY with per-group aggregates, and HAVING over them.

    The groups partition the admitted rows, so the printed TOTAL is the aggregate of
    exactly the groups above it -- recomputed over their union rather than carried
    down from the unfiltered table, which would disagree with the rows on screen the
    moment a HAVING term removed one.
    """
    terms = [analytics.parse_having(text) for text in args.having]
    groups = analytics.group_rows(catalog, args.group_by, selection)
    null_key = engine.catalog().NULL_KEY

    kept: list[tuple[str, dict]] = []
    surviving: list[int] = []
    for value, rows in groups:
        summary = analytics.group_summary(catalog, args.stats, rows)
        if all(analytics.having_holds(term, summary) for term in terms):
            kept.append((value, summary))
            surviving.extend(rows)

    fields = ("rows",) if args.stats is None else ("rows", "count", "nulls", "min", "max", "sum")
    names = [("(null)" if value == null_key else value) for value, _ in kept] or [""]
    width = max(len(name) for name in names + ["TOTAL"])
    having = " AND ".join(str(term) for term in terms) or "(none)"
    subject = f"AGGREGATE {args.stats}" if args.stats else "COUNT(*)"
    print(
        f"{subject} GROUP BY {args.group_by} WHERE {where} HAVING {having}"
        f"   [{len(groups)} group(s), {len(kept)} kept]"
    )
    header = "  " + "GROUP".ljust(width) + "".join(f"{name.upper():>10s}" for name in fields)
    print(header + (f"{'AVG':>12s}" if args.stats else ""))
    for value, summary in kept:
        shown = "(null)" if value == null_key else value
        print("  " + shown.ljust(width) + _aggregate_cells(summary, fields, args.stats))
    total = analytics.group_summary(catalog, args.stats, sorted(surviving))
    print("  " + "TOTAL".ljust(width) + _aggregate_cells(total, fields, args.stats))
    return EXIT_OK


def _aggregate_cells(summary: dict, fields: tuple[str, ...], stats_column: str | None) -> str:
    cells = "".join(f"{'-' if summary[name] is None else summary[name]:>10}" for name in fields)
    if stats_column is None:
        return cells
    average = summary["avg"]
    return cells + (f"{average:>12.2f}" if average is not None else f"{'-':>12}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query")
    parser.add_argument("--set", type=Path, default=DEFAULT_SET, dest="embedding_set")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=("reference", "native", "q8", "both", "auto"),
        default="reference",
        help="reference = exact Q15 in pure Python (the definition); native = the "
        "same ranking through bcir_ai_q15_topk; q8 = BCIRQ8 cosine through "
        "bcir_ai_q8_rows_dot_f64, lossier by construction; both = reference "
        "and native, required to agree exactly; auto = let the planner choose "
        "the cheapest legal plan for --objective",
    )
    parser.add_argument(
        "--objective",
        choices=tuple(name.lower() for name in ("EXACTNESS", "LATENCY", "FOOTPRINT", "STARTUP")),
        default="latency",
        help="which cost axis --backend auto minimizes; the names are the axes of "
        "BCIR's own 12-dimensional cost vector",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="TERM",
        help="narrow the scan: column=value, column!=value, column^=prefix, or "
        "column=one,two for a set. Repeatable, and repeated terms are ANDed. "
        "A predicate removes rows; it never reorders the rows it keeps.",
    )
    parser.add_argument(
        "--select",
        metavar="COLUMNS",
        help="comma-separated columns to print per result instead of the default "
        "location and heading trail",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="report how many rows the predicate admits and stop; no ranking, no text",
    )
    parser.add_argument(
        "--group-by",
        metavar="COLUMN",
        help="with --count, report the row count per distinct value of an indexed "
        "column, answered from the catalog without scanning",
    )
    parser.add_argument(
        "--order-by",
        metavar="COLUMN[:asc|desc]",
        help="order by a column instead of by distance. Only valid WITHOUT --query: "
        "with a query the order is the ranking, and a tool that quietly reordered "
        "a ranked result would be answering a different question than it was asked.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip this many results before --top-k; pagination over a stable order",
    )
    parser.add_argument(
        "--distinct",
        metavar="COLUMN",
        help="with --count, list the distinct values of an indexed column",
    )
    parser.add_argument(
        "--stats",
        metavar="COLUMN",
        help="with --count, report MIN/MAX/SUM/AVG over a numeric column. Unfiltered "
        "this is answered from the catalog's statistics without reading a row; "
        "with --where it gathers exactly the admitted cells. Combined with "
        "--group-by it reports them per group.",
    )
    parser.add_argument(
        "--having",
        action="append",
        default=[],
        metavar="TERM",
        help="with --group-by, keep only the groups whose aggregate satisfies a term: "
        f"{', '.join(analytics.AGGREGATE_FIELDS)} compared against one integer with =, !=, "
        ">=, <=, > or <. Repeatable and ANDed, like --where. 'rows' is COUNT(*); "
        "the rest need --stats and describe that column.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print every plan considered, its cost on the twelve axes, and why the chosen one won",
    )
    parser.add_argument(
        "--materialize",
        choices=("auto", "seek", "full"),
        default="auto",
        help="how result text is read: seek = the catalog's recorded byte spans for "
        "the k ranked rows; full = parse every chunk record; auto = seek when a "
        "fresh catalog is there, full otherwise with a printed reason",
    )
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="fail instead of skipping when the native rail cannot be built; pass "
        "this from the CI job that provides a C compiler",
    )
    parser.add_argument(
        "--require-catalog",
        action="store_true",
        help="fail instead of degrading when no fresh catalog is available; pass "
        "this from the job that builds one",
    )
    args = parser.parse_args(argv)

    if args.top_k < 1:
        print("search_chunks: --top-k must be at least 1", file=sys.stderr)
        return EXIT_USAGE
    if args.query is None and not (args.count or args.order_by):
        print(
            "search_chunks: --query is required unless --count or --order-by is given",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.order_by and args.query is not None:
        print(
            "search_chunks: --order-by and --query ask for two different orders. With "
            "a query the order is the ranking; drop --query for a relational scan, or "
            "drop --order-by to rank.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.offset < 0:
        print("search_chunks: --offset cannot be negative", file=sys.stderr)
        return EXIT_USAGE
    if args.group_by and not args.count:
        print("search_chunks: --group-by is only defined with --count", file=sys.stderr)
        return EXIT_USAGE
    if args.stats and not args.count:
        print("search_chunks: --stats is only defined with --count", file=sys.stderr)
        return EXIT_USAGE
    if args.distinct and not args.count:
        print("search_chunks: --distinct is only defined with --count", file=sys.stderr)
        return EXIT_USAGE
    if args.having and not args.group_by:
        print(
            "search_chunks: --having filters groups, so it needs --group-by; to filter "
            "rows, pass --where",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # --require-native is read only inside the native branch below, and --backend
    # defaults to `reference`, so the default invocation passed the flag and never
    # entered a path that could honour it: no import of BCIR, no kernel built, no native
    # dot product, exit 0. A flag that cannot fail on the configuration it is most likely
    # to be typed with is not a requirement, so say so rather than accepting it.
    if args.require_native and args.backend == "reference":
        print(
            "search_chunks: --require-native with --backend reference asks for a "
            "guarantee about a backend this run will not touch; pass --backend native, "
            "q8, both or auto",
            file=sys.stderr,
        )
        return EXIT_USAGE

    plan_module = engine.planner()
    catalog, catalog_reason = _catalog(args)

    # A predicate, a projection, a count and a seek all read the catalog. Asking for
    # any of them without one is a refusal rather than an answer over every row,
    # because silently ignoring --where would return rows the caller excluded.
    needs_catalog = bool(args.where) or args.count or args.select or args.materialize == "seek"
    if catalog is None:
        if args.require_catalog or needs_catalog:
            print(f"search_chunks: {catalog_reason}", file=sys.stderr)
            return EXIT_BACKEND_UNAVAILABLE
        print(f"[catalog] unavailable, reading every chunk record: {catalog_reason}")

    if args.count:
        try:
            selection = relational.resolve_selection(catalog, args.where)
        except plan_module.PlanError as exc:
            print(f"search_chunks: {exc}", file=sys.stderr)
            return EXIT_USAGE
        where = " AND ".join(str(p) for p in selection.predicates) or "(none)"

        if args.group_by and (args.stats or args.having):
            try:
                return _print_grouped_aggregate(plan_module, catalog, args, selection, where)
            except (KeyError, IndexError) as exc:
                print(f"search_chunks: {exc}", file=sys.stderr)
                return EXIT_USAGE
            except plan_module.PlanError as exc:
                print(f"search_chunks: {exc}", file=sys.stderr)
                return EXIT_USAGE

        if args.group_by:
            try:
                groups, scanned = analytics.group_counts(catalog, args.group_by, selection)
            except KeyError as exc:
                print(f"search_chunks: {exc}", file=sys.stderr)
                return EXIT_USAGE
            null_key = engine.catalog().NULL_KEY
            names = [("(null)" if name == null_key else name) for name, _ in groups] or [""]
            width = max(len(name) for name in names + ["TOTAL"])
            source = "postings intersection" if scanned else "statistics, no scan"
            print(f"COUNT(*) GROUP BY {args.group_by} WHERE {where}   [{source}]")
            for name, count in groups:
                shown = "(null)" if name == null_key else name
                print(f"  {shown.ljust(width)}  {count:6d}")
            print(f"  {'TOTAL'.ljust(width)}  {selection.admitted:6d}")
            return EXIT_OK

        if args.distinct:
            try:
                groups, scanned = analytics.group_counts(catalog, args.distinct, selection)
            except KeyError as exc:
                print(f"search_chunks: {exc}", file=sys.stderr)
                return EXIT_USAGE
            null_key = engine.catalog().NULL_KEY
            source = "postings intersection" if scanned else "statistics, no scan"
            print(f"DISTINCT {args.distinct} WHERE {where}   [{source}]")
            for name, _ in groups:
                print(f"  {'(null)' if name == null_key else name}")
            print(f"  -- {len(groups)} distinct value(s)")
            return EXIT_OK

        if args.stats:
            try:
                summary = catalog.aggregate(args.stats, selection.rows)
            except KeyError as exc:
                print(f"search_chunks: {exc}", file=sys.stderr)
                return EXIT_USAGE
            source = (
                f"gathered {summary['scanned']} cell(s)"
                if selection.rows is not None
                else "statistics, no scan"
            )
            print(f"AGGREGATE {args.stats} WHERE {where}   [{source}]")
            for field in ("count", "nulls", "min", "max", "sum"):
                print(f"  {field.upper().ljust(6)}  {summary[field]}")
            average = summary["avg"]
            print(
                f"  {'AVG'.ljust(6)}  {average:.2f}" if average is not None else "  AVG     (none)"
            )
            return EXIT_OK

        print(f"COUNT(*) WHERE {where} = {selection.admitted}")
        if selection.predicates:
            exactness = "exact" if selection.estimate_exact else "bound"
            print(f"  catalog estimate before evaluation: {selection.estimated} [{exactness}]")
        return EXIT_OK

    if args.order_by:
        # A pure relational scan: no query, no vectors, no backend. The rows come
        # from the predicate and the order from a column, which is the one shape of
        # question this tool could not previously be asked at all.
        try:
            selection = relational.resolve_selection(catalog, args.where)
            column, descending = relational.parse_order(args.order_by)
            ordered = relational.ordered_rows(catalog, selection, column, descending)
        except (plan_module.PlanError, KeyError, ValueError) as exc:
            print(f"search_chunks: {exc}", file=sys.stderr)
            return EXIT_USAGE
        window = ordered[args.offset : args.offset + args.top_k]
        records = catalog.fetch(window)
        where = " AND ".join(str(p) for p in selection.predicates) or "(none)"
        print(
            f"SELECT * WHERE {where} ORDER BY {column} "
            f"{'DESC' if descending else 'ASC'} LIMIT {args.top_k} OFFSET {args.offset}"
            f"   -- {len(ordered)} row(s) matched"
        )
        projection = (
            tuple(part.strip() for part in args.select.split(","))
            if args.select
            else (
                "source_path",
                column,
            )
        )
        for rank, row in enumerate(window, start=args.offset + 1):
            record = records.get(row, {})
            values = []
            for name in projection:
                value = record.get(name)
                if isinstance(value, list):
                    value = " > ".join(str(part) for part in value)
                values.append(f"{name}={value}")
            print(f"  {rank}. " + "  ".join(values))
        if not window:
            print("  (no rows)")
        return EXIT_OK

    embedding_set = EmbeddingSet(args.embedding_set)
    if args.require_native and not embedding_set.rows:
        # top_k = min(top_k, len(rows)) becomes 0, every ranking is empty, and the
        # printing loop iterates zero times while the flag reports success.
        print(
            f"search_chunks: --require-native over an embedding set with no rows "
            f"({args.embedding_set}); ranking nothing exercises no kernel",
            file=sys.stderr,
        )
        return EXIT_USAGE

    selection = plan_module.Selection(None, len(embedding_set.rows), True, ())
    if args.where:
        try:
            selection = relational.resolve_selection(catalog, args.where)
        except plan_module.PlanError as exc:
            print(f"search_chunks: {exc}", file=sys.stderr)
            return EXIT_USAGE
        if catalog.rows_total != len(embedding_set.rows):
            print(
                f"search_chunks: the catalog holds {catalog.rows_total} rows and the "
                f"embedding set holds {len(embedding_set.rows)}; a predicate resolved "
                "against one cannot select rows of the other",
                file=sys.stderr,
            )
            return EXIT_FAILED

    top_k = min(
        args.top_k, selection.admitted if selection.rows is not None else len(embedding_set.rows)
    )
    if top_k < 1:
        where = " AND ".join(str(p) for p in selection.predicates)
        print(f"\nQ: {args.query!r}   WHERE {where}\n  (no rows admitted)")
        return EXIT_OK

    materialize = args.materialize
    if materialize == "auto":
        materialize = "seek" if catalog is not None else "full"

    backend = args.backend
    plans = chosen = None
    if backend == "auto" or args.explain:
        available = {"reference"}
        try:
            _native().load_kernels()
            available |= {"native", "q8", "both"}
        except Exception:  # noqa: BLE001 -- any failure here means "not reachable"
            pass
        objective = plan_module.Objective[args.objective.upper()]
        plans = plan_module.candidates(
            catalog if catalog is not None else _RowsOnly(len(embedding_set.rows)),
            selection,
            top_k=top_k,
            dim=embedding_set.dim,
            want_text=True,
            require_exact=objective is plan_module.Objective.EXACTNESS,
            available_backends=frozenset(available),
            kernel_cached=True,
            files_touched=1,
            model=plan_module.DEFAULT_MODEL,
        )
        if backend == "auto":
            try:
                chosen = plan_module.choose(plans, objective)
            except plan_module.PlanError as exc:
                print(f"search_chunks: {exc}", file=sys.stderr)
                return EXIT_BACKEND_UNAVAILABLE
            backend = chosen.backend
            materialize = chosen.materialize if args.materialize == "auto" else materialize
        else:
            chosen = next(
                (p for p in plans if p.backend == backend and p.materialize == materialize), None
            )
        if args.explain:
            print(plan_module.explain(plans, chosen, objective, selection=selection))

    rows = selection.rows
    reference = native = None
    if backend in ("reference", "both"):
        reference = topk_reference(
            query := embed_query(args.query, embedding_set), embedding_set, top_k, rows=rows
        )
    else:
        query = embed_query(args.query, embedding_set)
    if backend in ("native", "q8", "both"):
        try:
            if backend == "q8":
                # Scores here are cosines, not squared distances: rank by them
                # descending. Reported through the same (row, score) shape.
                native = [
                    (row, score)
                    for row, score in topk_q8(
                        embed_query_float(args.query, embedding_set),
                        embedding_set,
                        top_k,
                        rows=rows,
                    )
                ]
            else:
                native = topk_native(query, embedding_set, top_k, rows=rows)
        except _native_unavailable() as exc:
            if args.require_native:
                print(f"search_chunks: native backend required: {exc}", file=sys.stderr)
                return EXIT_BACKEND_UNAVAILABLE
            print(f"[skip]   native backend unavailable: {exc}")
            if reference is None:
                return EXIT_OK

    if reference is not None and native is not None and backend == "both":
        if reference != native:
            print("search_chunks: reference and native rankings DISAGREE", file=sys.stderr)
            print(f"  reference: {reference}", file=sys.stderr)
            print(f"  native:    {native}", file=sys.stderr)
            return EXIT_FAILED
        print(f"[differential] reference == native on {len(reference)} result(s), exactly")

    results = reference if reference is not None else native
    assert results is not None
    if args.offset:
        results = results[args.offset :]

    if materialize == "seek" and catalog is not None:
        chunks = catalog.fetch(row for row, _ in results)
    else:
        by_id = load_chunk_texts(args.chunks)
        chunks = {row: by_id.get(embedding_set.rows[row]["chunk_id"]) for row, _ in results}

    where = " AND ".join(str(p) for p in selection.predicates)
    heading = f"\nQ: {args.query!r}   [{embedding_set.manifest['model']}, semantics={embedding_set.manifest['semantics']}]"
    if where:
        heading += f"\n   WHERE {where}  ->  {selection.admitted} of {len(embedding_set.rows)} row(s) scanned"
    print(heading)

    projection = tuple(part.strip() for part in args.select.split(",")) if args.select else ()
    for rank, (row, distance) in enumerate(results, start=1):
        entry = embedding_set.rows[row]
        chunk = chunks.get(row)
        # Squared Q15 distance is exact but unit-free; cosine is what a reader
        # can compare across queries. The identity is exact for unit vectors.
        cosine = (
            distance
            if backend == "q8"
            else 1.0 - distance / (2.0 * embedding_set.scale * embedding_set.scale)
        )
        if projection:
            values = []
            for column in projection:
                if chunk is not None and column in chunk:
                    value = chunk[column]
                elif column in entry:
                    value = entry[column]
                else:
                    print(
                        f"search_chunks: no column {column!r} on a chunk or index row",
                        file=sys.stderr,
                    )
                    return EXIT_USAGE
                if isinstance(value, list):
                    value = " > ".join(str(part) for part in value)
                values.append(f"{column}={value}")
            print(f"  {rank}. cos={cosine:+.3f}  " + "  ".join(values))
            continue
        trail = " > ".join(chunk["heading_trail"]) if chunk and chunk.get("heading_trail") else ""
        location = (
            f"{entry['source_path']}:{chunk['span']['start_line']}"
            if chunk
            else entry["source_path"]
        )
        print(f"  {rank}. cos={cosine:+.3f}  {location}")
        if trail:
            print(f"     {trail}")
    return EXIT_OK


@dataclass
class _RowsOnly:
    """The one fact the planner needs when there is no catalog: how many rows exist."""

    rows_total: int


if __name__ == "__main__":
    sys.exit(main())
