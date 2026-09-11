"""The corpus catalog: where a row's bytes are, and which rows a predicate admits.

Retrieval here used to be one fixed plan over one undifferentiated heap of rows.
Every query read every column of every row, because nothing recorded which rows a
question wanted or where their bytes began. The catalog is what a *plan* needs in
order to be a choice rather than a constant:

  row locator    row -> (file, byte offset, length), so `k` rows can be read
                 instead of all of them -- late materialization
  postings       indexed column value -> the rows holding it, and the same for
                 every directory prefix of `source_path`; this is `CREATE INDEX`,
                 and it makes a predicate cost the rows it admits rather than the
                 rows it rejects
  statistics     exact row counts per value, so selectivity is known *before* a
                 plan is chosen
  parts          contiguous row ranges with a content hash each, so a rebuild can
                 re-embed what changed instead of everything
  zone maps      per part, the MIN/MAX/COUNT/NULLS of each measurement column, so a
                 range predicate can rule a whole part out on four integers before
                 reading one of its cells -- a skip index, at the level of the
                 hierarchy the parts already established

The artifacts are split by how often they are wanted, and each is read on first
use, never on load:

  catalog.json   manifest, parts, statistics -- small, always read
  postings.json  the inverted index -- read when a predicate is evaluated
  locator.bin    14 bytes per row -- read when text is fetched
  ids.txt        row -> chunk_id -- read only for lookups by primary key
  numeric.bin    8 bytes per row per measurement column, column-major -- read when a
                 range predicate straddles a part, or an aggregate is gathered over
                 a selection
  order.bin      4 bytes per row per measurement column: that column's rows in
                 ascending order -- read when a comparison is evaluated, which it
                 then answers by two binary searches instead of a pass

That split is the same rule the embedding set follows for its derived columns: a
structure nobody asked for should not be built. Loading a catalog costs one small
JSON parse and a staleness check; a query that needs no predicate never pays for
the inverted index, and one that needs no text never pays for the locator.

Three properties are deliberate, and each is a refusal.

**Statistics are exact, not sampled.** At this size a distinct-value map over a
column costs a single pass and answers `count(subject = 'llvm')` with the number.
A planner that must guess selectivity needs histograms, sketches, and a model of
their error; one that knows needs none of that. Where a predicate cannot be
answered exactly -- a path prefix that is not itself a counted directory -- the
catalog returns a bound and says that it is a bound. Returning the bound as
though it were exact is how a planner comes to be confidently wrong.

**Staleness is loud, and checked one way.** The manifest records every chunk
file's name, size and content digest, and a load digests every file. Any file that
is missing, added, or whose digest changed refuses the load -- answering from stale
statistics produces a wrong plan silently, which is worse than not answering.
There is no declared scope to this, because there is nothing it does not see.

This used to be two levels: a stat comparison against a recorded size *and*
nanosecond mtime, then a digest for whatever the stat flagged. That fast path is
gone, and both halves of why are worth keeping written down.

The mtime had to go because a catalog must be a function of the chunk bytes alone.
Recording when a checkout happened made `catalog.json` differ between two builds of
byte-identical input, which quietly denied a generation's `tree_digest` any meaning
across machines. Keeping the mtime out but the fast path in -- pre-filtering on size
alone -- was measured and rejected: it widens the scope note above from "size and
nanosecond mtime deliberately preserved" to *any* same-size edit, which on JSONL is
every single-character correction anyone will ever make. The fast path's own scope
note was the thing it could no longer honour (`docs/security/laws.md` L21: a skip is
where a shipping defect hides).

So one level. The measured price is the whole of it: 0.23 ms -> 3.19 ms per load on
this corpus (8 files, 3.50 MiB), against a 67 ms cold relational command. Reopening
condition, so this is a decision and not a habit: if the corpus grows until digesting
it dominates a command, the answer is a design that stays one path -- not a second
one. `[wall, indicative]`.

**The direction of dependency is unchanged.** `training/` is never a build
dependency of BCIR, and nothing here reverses that.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import shutil
import struct
import sys
from array import array
from contextlib import contextmanager
import tempfile
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import schema as schema_module  # noqa: E402  (path set above)

#: v2 added the per-part numeric zone map. v3 split parts into bounded blocks and made
#: a part's content digest cover its own rows rather than its whole file. An older
#: catalog is refused rather than read with the difference papered over: pruning and
#: rebuild grain change only the *cost* of an answer, so a stale layout produces the
#: same rows and the same green run, and only one of them is honest
#: (`docs/security/laws.md` L2).
SCHEMA = "bcir-training/catalog/v4"
BUILDER = "training/tools/catalog.py"
LICENSE = "LicenseRef-BCIR-NC-1.0"

DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_CATALOG = Path("build/training/catalog")

#: Where a published set of artifacts lives, and the one mutable thing in a catalog
#: root. A publication is a directory under `SETS_DIR` named by the digest of the
#: artifacts it holds; `CURRENT_FILE` holds the name of the one to read. See `write`.
SETS_DIR = "sets"
CURRENT_FILE = "CURRENT"

CATALOG_FILE = "catalog.json"
POSTINGS_FILE = "postings.json"
LOCATOR_FILE = "locator.bin"
IDS_FILE = "ids.txt"
NUMERIC_FILE = "numeric.bin"
ORDER_FILE = "order.bin"

#: The most rows one part may cover. A part is the unit of two different things -- the
#: grain an incremental rebuild re-embeds at, and the span one zone map summarises --
#: and a part per chunk file served neither, because this corpus keeps 2,160 of its
#: 2,215 rows in a single file. Measured over that corpus, splitting each file into
#: blocks of at most this many rows:
#:
#:     block rows | parts | catalog.json | one-row edit | estimate for >= 5000
#:     per file   |       |              | invalidates  | (truth: 16 rows)
#:     -----------+-------+--------------+--------------+---------------------
#:     whole file |     8 |     24.7 KiB |    2160 rows | 2160
#:            256 |    16 |     27.1 KiB |     256 rows |  880
#:            128 |    24 |     29.5 KiB |     128 rows |  624
#:             64 |    41 |     34.4 KiB |      64 rows |  432
#:             32 |    75 |     44.2 KiB |      32 rows |  240
#:
#: 128 is the knee. `catalog.json` is the one artifact read on every load, so its
#: growth is paid by every caller, while the rebuild grain and the estimate are paid
#: only by the callers that rebuild or plan. Below 128 the manifest grows faster than
#: either improves. A block never spans two files, so a part still belongs to exactly
#: one source and a file's rows are still a contiguous run of blocks.
MAX_BLOCK_ROWS = 128

#: Which columns play which role, read out of the declared table rather than named
#: again here. These three tuples used to be the only statement anywhere that
#: `subject` is indexed and `char_count` is a measurement, and they said nothing about
#: what either holds -- so an index could be declared for a column the corpus does not
#: have, and the build would produce a postings list of one entry, all rows, null
#: (`docs/security/laws.md` L15). `schema.py` declares the table; this reads it.
#:
#: Indexed columns are low cardinality by construction, so a distinct-value map over
#: any of them is a handful of entries rather than a histogram. Numeric columns are
#: stored as a packed integer column instead: a distinct-value map over a measurement
#: is as large as the table, and what an aggregate wants is the values themselves,
#: contiguous, so MIN/MAX/SUM over a selection is a gather and not a scan.
INDEXED_COLUMNS = schema_module.INDEXED_COLUMNS
PATH_COLUMN = schema_module.PATH_COLUMN
NUMERIC_COLUMNS = schema_module.NUMERIC_COLUMNS

#: How a missing value is spelled in an index key. JSON object keys are strings,
#: so `null` needs a spelling that no real value can collide with.
NULL_KEY = "\0null"

#: One locator entry: file index, byte offset, byte length. Little-endian and
#: unpadded, so the file is byte-identical on every host -- the same rule the
#: embedding set's `vectors.q15` follows.
_LOCATOR_STRUCT = struct.Struct("<HQI")
LOCATOR_ENTRY_BYTES = _LOCATOR_STRUCT.size

#: One numeric cell: a signed 64-bit value, little-endian. `NUMERIC_NULL` is the
#: spelling of a missing measurement -- distinct from zero, which is a real length.
_NUMERIC_STRUCT = struct.Struct("<q")
NUMERIC_CELL_BYTES = _NUMERIC_STRUCT.size
NUMERIC_NULL = -(2**63)

#: One entry of a sorted index: a row number, little-endian and unsigned. Each
#: measurement column contributes `rows_total` of them -- the measured rows first, in
#: ascending (value, row) order, then the unmeasured rows in row order. The split
#: point is the column's own `count`, so the index needs no length of its own to be
#: read correctly, and a comparison becomes two binary searches over a slice.
_ORDER_STRUCT = struct.Struct("<I")
ORDER_ENTRY_BYTES = _ORDER_STRUCT.size
MAX_INDEXABLE_ROWS = 2**32 - 1


#: The share of the table at or above which scanning the packed column beats seeking
#: the sorted index and sorting the window it returns. Below it the seek wins by up to
#: 90x; over the whole table the scan wins by about 6x, because a scan produces rows
#: in order for free while a seek has to sort what it found.
#:
#: Measured over this corpus, both measurement columns, best of 15 (ms):
#:
#:     share of table |  seek |  scan
#:     ---------------+-------+------
#:               50%  | 0.063 | 0.108
#:               70%  | 0.106 | 0.109   <- char_count crosses here
#:               75%  | 0.120 | 0.110
#:               80%  | 0.119 | 0.110   <- token_estimate crosses here
#:               90%  | 0.156 | 0.111
#:
#: 0.70 is the earlier of the two crossings, so neither column is pushed past its own.
#: Metric class `wall`: this orders two plans, it does not predict a duration, and a
#: wrong choice costs time and never rows -- which is why the gate checks that the
#: choice *is made*, and never checks a clock.
SORTED_SCAN_SHARE = 0.70


class CatalogError(RuntimeError):
    """The catalog is absent, stale, or malformed. Never a silent fallback."""


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


def chunk_files(chunk_dir: Path) -> list[Path]:
    """Every chunk file, in one order, so two callers cannot disagree about the set."""
    return sorted(chunk_dir.glob("*.chunks.jsonl"))


def row_sort_key(record: dict) -> tuple:
    """**The** row order of the corpus. One definition, imported by both writers.

    Row `i` of an embedding set must be row `i` of this catalog, because that is what
    lets a ranked row number be turned into bytes on disk with no lookup. Before this
    function existed the two arrived at the same order by two independent rules --
    `embed_chunks` sorted records, `catalog` walked files -- and they agreed only
    because the files happen to be named by subject and written in this order.

    Two rails computing the same thing separately is the defect shape this repository
    has paid for most often (L12, L14). So the order is defined here and imported,
    rather than restated; `verify_database.py` still asserts the result, because a
    shared definition that only one caller actually uses is the same bug wearing a
    better name.
    """
    span = record.get("span") or {}
    return (
        str(record.get("subject") or ""),
        str(record.get("source_path") or ""),
        int(span.get("start_line") or 0),
        str(record.get("chunk_id") or ""),
    )


def _file_stat(path: Path) -> dict:
    """What the manifest records about one chunk file: name, size, content digest.

    **No modification time, and no directory.** Everything in a catalog is a function
    of the chunk bytes alone, so two checkouts of the same corpus produce the same
    catalog -- which is what lets `tree_digest` over a generation mean anything across
    machines. An `mtime_ns` here made `catalog.json` differ between two builds of
    byte-identical input, and the determinism gate could not see it because it built
    twice from one directory, where the mtimes are equal by construction.

    The mtime was a cache key for `_check_freshness`'s fast path: a file whose size
    *and* mtime match the record skips its digest. Size alone still pre-filters, and
    on this corpus the cost of the mtimes it no longer skips is 0.23 ms -> 2.87 ms per
    load. That is the whole price of the property, and it is paid here rather than
    bought back with a rule that excludes one file from a digest -- an exclusion would
    make the digest silent about the one file most likely to be wrong
    (`docs/security/laws.md` L21).
    """
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def fingerprint(chunk_dir: Path) -> str:
    """Content address of the whole chunk table, names included.

    Names as well as contents: a file that appears or disappears changes the answer
    to "how many rows are there", and a digest over contents alone would miss it.
    """
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode("utf-8"))
    digest.update(b"\0")
    for path in chunk_files(chunk_dir):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def path_prefixes(source_path: str) -> tuple[str, ...]:
    """Every directory prefix of a path, shallowest first, excluding the file itself.

    ``a/b/c.md`` -> ``("a", "a/b")``. These are the prefixes a reader would type,
    and indexing all of them is what makes a prefix predicate exact rather than an
    estimate.
    """
    parts = source_path.split("/")[:-1]
    return tuple("/".join(parts[: i + 1]) for i in range(len(parts)))


def index_key(value) -> str:
    """The index spelling of a column value."""
    return NULL_KEY if value is None else str(value)


def pair_key(left: str, right: str) -> str:
    """How a column pair is named in the statistics, in one canonical order.

    Sorted, so `(kind, language)` and `(language, kind)` name the same entry and the
    build cannot store one while a reader looks for the other. The separator is a NUL,
    which no column name can contain -- column names are declared identifiers in
    `schema.py`, but a separator chosen from the printable characters is a rule about
    names rather than a property of them, and the two drift differently.
    """
    first, second = sorted((left, right))
    return f"{first}\0{second}"


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def _scan_chunk_file(path: Path, file_index: int, table) -> list[dict]:
    """One pass over a chunk file, recording each row's columns and its byte span.

    The offset is into the file as bytes, not as decoded text, because that is what
    `seek` takes. Lines are split on b"\\n" and the span covers the line without its
    terminator, so a fetch returns exactly one JSON document.
    """
    raw = path.read_bytes()
    rows: list[dict] = []
    offset = 0
    total = len(raw)
    while offset < total:
        end = raw.find(b"\n", offset)
        if end == -1:
            end = total
        line = raw[offset:end]
        if line.strip():
            record = json.loads(line.decode("utf-8"))
            # Every declared column of the row as it came off disk, before the
            # projection below drops the ones the catalog does not index. Checking the
            # projection instead would check the six columns copied out and say
            # nothing about the twelve left behind, which is where a renamed field
            # hides (`docs/security/laws.md` L15).
            with _as_catalog_error():
                table.check_row(record)
            rows.append(
                {
                    "chunk_id": record.get("chunk_id"),
                    "subject": record.get("subject"),
                    "source_path": record.get("source_path"),
                    "kind": record.get("kind"),
                    "language": record.get("language"),
                    "span": record.get("span"),
                    **{name: record.get(name) for name in table.numeric},
                    "file": file_index,
                    "offset": offset,
                    "length": end - offset,
                    # The row's own bytes, digested while they are already in hand. A
                    # part's content is derived from these rather than from its file,
                    # so an edit invalidates the block holding it and not the 2,160
                    # rows that happen to share a file with it.
                    "digest": hashlib.sha256(line).hexdigest(),
                }
            )
        offset = end + 1
    return rows


def blocks(start: int, end: int, size: int | None = None) -> list[tuple[int, int]]:
    """Split a row range into contiguous blocks of at most `size` rows.

    One rule, used at build time to cut the parts and by the gate to predict where
    they fall, so "how a file is divided" is stated once rather than twice
    (`docs/security/laws.md` L14). A range shorter than one block yields one block,
    so a small file is not split and an empty one yields nothing to describe.
    """
    # Read at call time, not bound as a default argument: a default is evaluated once
    # when the module is imported, so `MAX_BLOCK_ROWS` could be changed and have no
    # effect -- which is exactly what happened to the harness that chose its value.
    size = MAX_BLOCK_ROWS if size is None else size
    if size < 1:
        raise CatalogError(f"catalog: a block must hold at least one row, not {size}")
    return [(lo, min(end, lo + size)) for lo in range(start, end, size)]


def block_digest(rows: list[dict]) -> str:
    """A part's content: the digest of its own rows, in row order.

    Built from the per-row digests rather than from the file, which is the whole point
    of splitting: two blocks of one file have different contents, so an edit moves one
    of them. Digesting the row digests rather than the rows again is the same
    construction `generations.tree_digest` uses -- each input is already a fixed-width
    canonical string, so no separator can be forged by a row's own bytes.
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["digest"].encode("ascii"))
    return digest.hexdigest()


@contextmanager
def _as_catalog_error():
    """Turn the schema's refusals into this module's, so a caller catches one type.

    A `SchemaError` escaping a build reaches the CLI as a traceback rather than as a
    verdict, which is the same exit either way to a human and a different one to a
    script (`docs/security/laws.md` L1).
    """
    try:
        yield
    except schema_module.SchemaError as exc:
        raise CatalogError(f"catalog: {exc}") from exc


def numeric_cell(row: dict, column: str, table=None) -> int | None:
    """One numeric cell, or None where the row carries no measurement. Or a refusal.

    Three cases, and they must stay three. A row carrying no value at all has no
    measurement: that is `None`, and the packed column records it as absent. An
    integer inside its declared domain is the measurement. **Anything else is a schema
    error and is refused**, naming the row and the column.

    Reading a wrong type as absent -- which this used to do -- is the shape
    `docs/security/laws.md` L1 is about: `char_count` holding `"not-an-int"` built a
    clean catalog, exited 0, and appeared downstream as one more null among the
    genuine ones, indistinguishable from a chunk that really has no measurement. The
    gap and the error must not share a spelling; a build over a malformed row is a
    failed build, not a build with one more unknown in it.

    The decision itself lives in `schema.cell`, which is also what a projection and a
    presence test ask, so "absent" cannot come to mean one thing to the packed column
    and another to a query (`docs/security/laws.md` L14).
    """
    with _as_catalog_error():
        return (table or schema_module.TABLE).cell(row, column)


def numeric_summary(rows: list[dict], column: str, table=None) -> dict:
    """MIN/MAX/SUM/COUNT/NULLS over one numeric column of a row range.

    The whole-corpus statistics and every part's zone map are produced by this one
    function, so a part cannot summarise its rows by a different rule than the catalog
    summarises all of them (`docs/security/laws.md` L14). That matters more here than
    in most shared predicates: a zone map that disagreed with the statistics by even
    one would prune a part that holds a matching row, and the query would return a
    wrong answer quickly rather than a right answer slowly.

    `min` and `max` are over present values only. The null sentinel is a legal int64
    and would otherwise become the minimum of every column that has a gap.
    """
    present = [
        value
        for value in (numeric_cell(row, column, table) for row in rows)
        if value is not None
    ]
    return {
        "count": len(present),
        "nulls": len(rows) - len(present),
        "min": min(present) if present else None,
        "max": max(present) if present else None,
        "sum": sum(present),
    }


def build(
    chunk_dir: Path = DEFAULT_CHUNKS, table=None
) -> tuple[dict, dict, bytes, str, bytes, bytes]:
    """Read the chunk table once and derive every fact a plan can use.

    One pass. The locator, the postings, the statistics and the parts all come from
    the rows that scan already produced -- a catalog needing its own second pass
    over the corpus would cost more than the scans it exists to avoid.

    Returns the manifest, the postings, the packed locator, the id list, the packed
    numeric columns and the sorted index over them.

    `table` is the declared column table this corpus is built against, defaulting to
    the shipped one. It is a parameter because nullability is a property of a table
    and not of the format: the chunk contract requires every measurement, so the
    absence a packed column must still be able to spell is only reachable from a
    corpus whose table admits it (`schema.Table`).
    """
    table = table or schema_module.TABLE
    files = chunk_files(chunk_dir)
    if not files:
        raise CatalogError(
            f"catalog: no chunk files under {chunk_dir}\n"
            "  build them first: python3 training/tools/build_chunks.py "
            f"--out {chunk_dir}"
        )

    rows: list[dict] = []
    parts: list[dict] = []
    stats_files: list[dict] = []
    for index, path in enumerate(files):
        start = len(rows)
        scanned = _scan_chunk_file(path, index, table)
        # Sorted by the canonical key rather than taken in file order, so the catalog
        # and the embedding set share one definition of row `i` instead of two that
        # happen to agree. A part stays one contiguous run because `subject` leads the
        # key and a chunk file holds one subject.
        scanned.sort(key=row_sort_key)
        rows.extend(scanned)
        stat = _file_stat(path)
        stats_files.append(stat)
        source = path.name[: -len(".chunks.jsonl")]
        for number, (low, high) in enumerate(blocks(start, len(rows))):
            parts.append(
                {
                    "part_id": f"{source}#{number:04d}",
                    "source": source,
                    "file": path.name,
                    "rows": [low, high],
                    "content": block_digest(rows[low:high]),
                }
            )

    seen: dict[str, int] = {}
    for position, row in enumerate(rows):
        # Presence, type and emptiness were settled by `schema.check_row` as the row
        # was read. What is left is the one property no single row can have: being
        # the only row with this key.
        chunk_id = row["chunk_id"]
        if chunk_id in seen:
            raise CatalogError(
                f"catalog: chunk_id {chunk_id} appears at rows {seen[chunk_id]} and "
                f"{position}; a primary key that is not unique cannot locate a row"
            )
        seen[chunk_id] = position

    postings_columns: dict[str, dict[str, list[int]]] = {}
    for name in table.indexed:
        column: dict[str, list[int]] = {}
        for position, row in enumerate(rows):
            column.setdefault(index_key(row.get(name)), []).append(position)
        postings_columns[name] = dict(sorted(column.items()))

    prefix_postings: dict[str, list[int]] = {}
    path_postings: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        source_path = row.get(table.path)
        if not isinstance(source_path, str):
            continue
        path_postings.setdefault(source_path, []).append(position)
        for prefix in path_prefixes(source_path):
            prefix_postings.setdefault(prefix, []).append(position)

    # Exact co-occurrence counts for every pair of indexed columns. The catalog held
    # only marginals, so a conjunction was priced at the tightest of them -- a bound,
    # and one that was reported as a count: on this corpus 83% of indexed value pairs
    # were labelled `[exact]` over a number that was wrong, and `kind=code AND
    # language=asm` claimed a row where there are none.
    #
    # The reason the marginals were all the catalog held is that joint statistics are
    # expensive in the general case, which is why every planner that estimates them
    # does so with sketches and independence assumptions. That reasoning does not
    # transfer here. These columns are low cardinality *by role* -- an indexed column
    # is one whose distinct-value map is a handful of entries -- so the full joint
    # distribution over any two of them is bounded by the product of two handfuls, and
    # the corpus is static, so it is computed once at build. Exactly the case where
    # the textbook answer is the wrong one.
    pair_counts: dict[str, dict[str, dict[str, int]]] = {}
    for first in range(len(table.indexed)):
        for second in range(first + 1, len(table.indexed)):
            # Sorted, because `pair_key` sorts: the table's outer key must be the
            # column whose name `pair_key` put first, or the build stores the
            # transpose of what every reader looks up. Taking the pair in declaration
            # order stored two of these three tables transposed, and every lookup into
            # them returned zero -- an under-estimate, which is the one direction a
            # bound must never go (`docs/security/laws.md` L12: one condition, one
            # answer, on both paths).
            left, right = sorted((table.indexed[first], table.indexed[second]))
            counted: dict[str, dict[str, int]] = {}
            for row in rows:
                inner = counted.setdefault(index_key(row.get(left)), {})
                key = index_key(row.get(right))
                inner[key] = inner.get(key, 0) + 1
            pair_counts[pair_key(left, right)] = {
                value: dict(sorted(inner.items())) for value, inner in sorted(counted.items())
            }

    postings = {
        "schema": SCHEMA,
        "columns": postings_columns,
        "path_prefixes": dict(sorted(prefix_postings.items())),
        "paths": dict(sorted(path_postings.items())),
    }

    locator = bytearray()
    for row in rows:
        locator += _LOCATOR_STRUCT.pack(int(row["file"]), int(row["offset"]), int(row["length"]))
    ids = "".join(f"{row['chunk_id']}\n" for row in rows)

    numeric = bytearray()
    numeric_stats: dict[str, dict] = {}
    for name in table.numeric:
        for row in rows:
            value = numeric_cell(row, name, table)
            numeric += _NUMERIC_STRUCT.pack(NUMERIC_NULL if value is None else value)
        numeric_stats[name] = numeric_summary(rows, name, table)

    # The zone map: the range of values each part could possibly hold. A range
    # predicate reads this before it reads a row, so a part whose whole span lies
    # outside the range is skipped without touching the packed column -- the pruning a
    # skip index does, at the level of the hierarchy this catalog already had. It costs
    # one more pass over rows already in memory, not another pass over the corpus.
    for part in parts:
        start, end = part["rows"]
        span = rows[start:end]
        part["numeric"] = {name: numeric_summary(span, name, table) for name in table.numeric}

    # The sorted index: each measurement column's rows in ascending order. Two binary
    # searches over it answer a comparison exactly, where the zone map above can only
    # rule parts out -- and on this corpus one part holds most of the rows, so ruling
    # parts out is worth far less than being able to seek.
    #
    # Row numbers are stored unsigned 32-bit, which is a bound on what this format can
    # index. It is checked here, where the rows are counted, rather than discovered as
    # a wrapped index at query time (`docs/security/laws.md` L3).
    if len(rows) > MAX_INDEXABLE_ROWS:
        raise CatalogError(
            f"catalog: {len(rows)} rows exceeds the {MAX_INDEXABLE_ROWS} a 32-bit "
            "sorted-index entry can name"
        )
    order = bytearray()
    for name in table.numeric:
        measured = []
        absent = []
        for position, row in enumerate(rows):
            value = numeric_cell(row, name, table)
            (absent if value is None else measured).append(
                position if value is None else (value, position)
            )
        # Ties by row, ascending, so the index and a stable sort of the same rows
        # agree -- and so a page boundary inside a run of equal values falls in the
        # same place every time.
        measured.sort()
        for _, position in measured:
            order += _ORDER_STRUCT.pack(position)
        for position in absent:
            order += _ORDER_STRUCT.pack(position)

    manifest = {
        "schema": SCHEMA,
        "builder": BUILDER,
        "license": LICENSE,
        "fingerprint": fingerprint(chunk_dir),
        "rows_total": len(rows),
        "files": stats_files,
        "parts": parts,
        "statistics": {
            "columns": {
                name: {value: len(positions) for value, positions in column.items()}
                for name, column in postings_columns.items()
            },
            "pairs": pair_counts,
            "path_prefixes": {
                prefix: len(positions) for prefix, positions in postings["path_prefixes"].items()
            },
            "paths": {path: len(positions) for path, positions in postings["paths"].items()},
            "numeric": numeric_stats,
        },
        "artifacts": {
            "postings": {
                "path": POSTINGS_FILE,
                "sha256": hashlib.sha256(
                    json.dumps(postings, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
            "locator": {
                "path": LOCATOR_FILE,
                "sha256": hashlib.sha256(bytes(locator)).hexdigest(),
                "entry_bytes": LOCATOR_ENTRY_BYTES,
            },
            "ids": {
                "path": IDS_FILE,
                "sha256": hashlib.sha256(ids.encode("utf-8")).hexdigest(),
            },
            "numeric": {
                "path": NUMERIC_FILE,
                "sha256": hashlib.sha256(bytes(numeric)).hexdigest(),
                "columns": list(table.numeric),
                "cell_bytes": NUMERIC_CELL_BYTES,
            },
            "order": {
                "path": ORDER_FILE,
                "sha256": hashlib.sha256(bytes(order)).hexdigest(),
                "columns": list(table.numeric),
                "entry_bytes": ORDER_ENTRY_BYTES,
            },
        },
    }
    return manifest, postings, bytes(locator), ids, bytes(numeric), bytes(order)


def _publish(root: Path, name: str, payload: bytes) -> Path:
    """Write one artifact, durably. No reader ever sees half of one.

    The `fsync` is what makes the later pointer swap mean something: a rename that
    reaches the directory before the file's own bytes reach the disk publishes a name
    for content that a crash can still lose. Ordering the two is the whole of the D in
    a publish that has no log to replay.
    """
    target = root / name
    descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.tmp-", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def _sync_directory(path: Path) -> None:
    """Flush a directory's own entries, where the host has a way to say that.

    POSIX needs this: a file's contents being durable does not make the *name* that
    reaches it durable. Windows has no directory handle to sync and its rename is
    already ordered, so there is nothing to call -- declared here as a per-host
    property rather than swallowed, because a silent `except` around an I/O call
    cannot tell "this host does not do that" from "this disk is failing"
    (`docs/security/laws.md` L1).
    """
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publication_id(payloads: dict[str, bytes]) -> str:
    """The name of a published set: the digest of everything in it.

    Content-addressed for the same reason a generation is. Publishing the same
    catalog twice names the same directory, finds it already complete, and moves the
    pointer -- so a rebuild that changes nothing writes nothing, and two hosts that
    build the same corpus agree on the name as well as on the bytes.
    """
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode("utf-8"))
    for name in sorted(payloads):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payloads[name]).digest())
    return digest.hexdigest()[:32]


def current_set(root: Path) -> Path:
    """The directory the pointer names, or a refusal. Where every reader starts."""
    pointer = root / CURRENT_FILE
    try:
        name = pointer.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise CatalogError(
            f"catalog: none at {root}\n  build one first: python3 {BUILDER} --out {root}"
        ) from exc
    except OSError as exc:
        raise CatalogError(f"catalog: {pointer} is unreadable: {exc}") from exc
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise CatalogError(f"catalog: {pointer} does not name a published set ({name!r})")
    directory = root / SETS_DIR / name
    if not directory.is_dir():
        raise CatalogError(
            f"catalog: {pointer} names {name}, which is not published under "
            f"{root / SETS_DIR}\n  rebuild it: python3 {BUILDER} --out {root}"
        )
    return directory


def write(built: tuple[dict, dict, bytes, str, bytes, bytes], root: Path = DEFAULT_CATALOG) -> Path:
    """Publish a whole set of artifacts, or publish none of it.

    **One pointer is the whole transaction.** The six artifacts go into a directory
    named by their own digest, which nothing has ever read from because nothing knew
    the name; they are made durable there; and then a single `os.replace` of `CURRENT`
    makes them the catalog. A reader is holding the old name or the new one, and both
    name a complete set. There is no instant at which it holds half of a rebuild.

    This used to be six `os.replace` calls into one directory, manifest last. Each was
    atomic on its own, which is a different and much weaker claim: between the second
    and the third, the directory held four old artifacts and two new ones, and a
    reader that opened the old manifest and then read a new sidecar found a digest
    that did not match. That failed *loudly* -- the manifest vouches for every
    artifact, so the mismatch was a refusal and never a wrong answer -- but a rebuild
    that makes concurrent readers fail is still a rebuild that cannot be done while
    anything is reading. Atomicity is what turns "loud" into "invisible".

    Idempotent, because the name is the content: republishing an unchanged catalog
    finds the set already there, verifies it, and moves the pointer. Immutable, for
    the same reason -- nothing is ever rewritten inside a published set, so the
    previous one stays readable for a reader that is still in it.
    """
    manifest, postings, locator, ids, numeric, order = built
    payloads = {
        POSTINGS_FILE: json.dumps(postings, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        LOCATOR_FILE: bytes(locator),
        IDS_FILE: ids.encode("utf-8"),
        NUMERIC_FILE: bytes(numeric),
        ORDER_FILE: bytes(order),
        CATALOG_FILE: json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    }
    name = publication_id(payloads)
    sets = root / SETS_DIR
    sets.mkdir(parents=True, exist_ok=True)
    destination = sets / name

    if not _set_is_complete(destination, payloads):
        staging = sets / f".staging-{name}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            for artifact, payload in sorted(payloads.items()):
                _publish(staging, artifact, payload)
            _sync_directory(staging)
            try:
                os.replace(staging, destination)
            except OSError:
                # Another writer published this same content-addressed set between the
                # check above and this rename. POSIX and Windows disagree about
                # renaming onto a directory that now exists, so the outcome is decided
                # here rather than by the host (`docs/security/laws.md` L12). The name
                # is the content, so if what is in place is complete this publish
                # already happened; if it is not, the refusal stands.
                if not _set_is_complete(destination, payloads):
                    raise
                shutil.rmtree(staging, ignore_errors=True)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        _sync_directory(sets)

    previous = None
    try:
        previous = current_set(root).name
    except CatalogError:
        pass
    _point_at(root, name)
    _prune_sets(sets, keep={name} | ({previous} if previous else set()))
    return destination / CATALOG_FILE


def _set_is_complete(directory: Path, payloads: dict[str, bytes]) -> bool:
    """Whether a published set holds exactly these artifacts, byte for byte.

    Compared against the bytes about to be written rather than against a recorded
    digest, so a set left behind by an interrupted publish -- or one a filesystem
    truncated -- is republished instead of pointed at.
    """
    if not directory.is_dir():
        return False
    try:
        present = {path.name for path in directory.iterdir() if path.is_file()}
        if present != set(payloads):
            return False
        return all((directory / name).read_bytes() == payload for name, payload in payloads.items())
    except OSError:
        return False


def _point_at(root: Path, name: str) -> None:
    """Move `CURRENT` atomically. The pointer is the only mutable thing in a catalog."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{CURRENT_FILE}.tmp-", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(name + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / CURRENT_FILE)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    _sync_directory(root)


def _prune_sets(sets: Path, keep: set[str]) -> None:
    """Drop published sets older than the one the pointer just replaced.

    The previous set is kept because a reader that resolved `CURRENT` a moment before
    the swap is still reading from it, and nothing here can ask whether it has
    finished -- there is no server, no lock and no lease, and inventing one to delete
    a directory sooner would be a much larger mechanism than the directory is worth.
    Anything older than that has no reader that a single swap could have left behind.
    """
    for path in sets.iterdir():
        if path.name in keep or not path.is_dir():
            continue
        shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


class Catalog:
    """A loaded catalog, bound to the chunk directory it describes.

    Only the manifest is read here. Every other artifact is loaded by the first
    call that needs it and verified against the digest the manifest recorded.
    """

    def __init__(self, manifest: dict, root: Path, chunk_dir: Path, set_dir: Path) -> None:
        self.manifest = manifest
        self.root = root
        #: The published set this catalog reads its artifacts from. Fixed at load, so
        #: a rebuild that lands mid-read moves `CURRENT` without moving this object:
        #: every artifact it goes on to read comes from the set its manifest vouches
        #: for, not from whichever set is newest by the time it asks.
        self.set_dir = set_dir
        self.chunk_dir = chunk_dir
        self.rows_total = int(manifest["rows_total"])
        self.files = [chunk_dir / entry["name"] for entry in manifest["files"]]
        self.parts = manifest["parts"]
        self._statistics = manifest["statistics"]
        self._artifacts = manifest["artifacts"]
        self._postings: dict | None = None
        self._locator: bytes | None = None
        self._ids: list[str] | None = None
        self._numeric: dict | None = None
        self._order: dict | None = None
        self._part_starts_cache: list[int] | None = None
        self._by_chunk_id: dict[str, int] | None = None

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        root: Path = DEFAULT_CATALOG,
        chunk_dir: Path = DEFAULT_CHUNKS,
    ) -> "Catalog":
        """Load the catalog, or refuse. A stale catalog is never answered from.

        Every failure is the same verdict -- a `CatalogError` naming the rebuild
        command -- because a caller that cannot tell "absent" from "stale" from
        "malformed" will eventually treat one of them as "fine".
        """
        set_dir = current_set(root)
        target = set_dir / CATALOG_FILE
        try:
            manifest = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CatalogError(
                f"catalog: {root / CURRENT_FILE} names a set with no {CATALOG_FILE}\n"
                f"  rebuild it: python3 {BUILDER} --out {root}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise CatalogError(f"catalog: {target} is unreadable: {exc}") from exc
        if not isinstance(manifest, dict):
            raise CatalogError(f"catalog: {target} is {type(manifest).__name__}, not an object")
        if manifest.get("schema") != SCHEMA:
            raise CatalogError(
                f"catalog: {target} is not {SCHEMA} (found {manifest.get('schema')!r})"
            )
        for required in ("rows_total", "files", "parts", "statistics", "artifacts", "fingerprint"):
            if required not in manifest:
                raise CatalogError(f"catalog: {target} has no {required!r}")
        cls._check_freshness(manifest, root, chunk_dir)
        return cls(manifest, root, chunk_dir, set_dir)

    @staticmethod
    def _check_freshness(manifest: dict, root: Path, chunk_dir: Path) -> None:
        """The chunk table this catalog was built from, still byte for byte.

        One level, digesting every file. The size in the manifest is recorded for the
        refusal message and for anyone reading it, not consulted as a pre-filter -- a
        pre-filter is a second path that can disagree with the first, and this one
        would have disagreed for exactly the same-size edits that are the common case
        (`docs/security/laws.md` L12). The module docstring carries the measurement
        that says the pre-filter was not worth its scope.
        """
        recorded = {entry["name"]: entry for entry in manifest["files"]}
        actual = {path.name: path for path in chunk_files(chunk_dir)}
        missing = sorted(set(recorded) - set(actual))
        added = sorted(set(actual) - set(recorded))
        if missing or added:
            raise CatalogError(
                f"catalog: {root / CATALOG_FILE} describes a different chunk table\n"
                + (f"  missing now: {', '.join(missing)}\n" if missing else "")
                + (f"  new since:   {', '.join(added)}\n" if added else "")
                + f"  rebuild it: python3 {BUILDER} --out {root}"
            )
        moved = []
        for name, entry in sorted(recorded.items()):
            if hashlib.sha256(actual[name].read_bytes()).hexdigest() != entry["sha256"]:
                moved.append(name)
        if moved:
            raise CatalogError(
                f"catalog: {root / CATALOG_FILE} was built from different bytes\n"
                f"  changed: {', '.join(moved)}\n"
                f"  rebuild it: python3 {BUILDER} --out {root}"
            )

    @classmethod
    def load_or_build(
        cls, root: Path = DEFAULT_CATALOG, chunk_dir: Path = DEFAULT_CHUNKS
    ) -> "Catalog":
        """Load, or build once and load.

        A convenience for tools that may run before any catalog exists, not a
        fallback: it builds the same catalog the gate checks, and a chunk table that
        cannot be read still raises.
        """
        try:
            return cls.load(root, chunk_dir)
        except CatalogError:
            write(build(chunk_dir), root)
            return cls.load(root, chunk_dir)

    # -- lazily loaded artifacts -----------------------------------------

    def _artifact_bytes(self, name: str) -> bytes:
        entry = self._artifacts.get(name)
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            raise CatalogError(f"catalog: the manifest does not describe the {name!r} artifact")
        path = self.set_dir / entry["path"]
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CatalogError(
                f"catalog: {path} is unreadable: {exc}\n"
                f"  rebuild it: python3 {BUILDER} --out {self.root}"
            ) from exc
        actual = hashlib.sha256(raw).hexdigest()
        if actual != entry["sha256"]:
            raise CatalogError(
                f"catalog: {path} is not the artifact the manifest vouches for\n"
                f"  recorded: {entry['sha256']}\n  actual:   {actual}\n"
                f"  rebuild it: python3 {BUILDER} --out {self.root}"
            )
        return raw

    @property
    def postings(self) -> dict:
        """The inverted index, read and verified on first use."""
        if self._postings is None:
            payload = json.loads(self._artifact_bytes("postings").decode("utf-8"))
            if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
                raise CatalogError(f"catalog: {POSTINGS_FILE} is not {SCHEMA}")
            self._postings = payload
        return self._postings

    @property
    def locator(self) -> bytes:
        """The packed row locator, read and verified on first use."""
        if self._locator is None:
            raw = self._artifact_bytes("locator")
            if len(raw) != self.rows_total * LOCATOR_ENTRY_BYTES:
                raise CatalogError(
                    f"catalog: {LOCATOR_FILE} holds {len(raw)} bytes, expected "
                    f"{self.rows_total * LOCATOR_ENTRY_BYTES} "
                    f"({self.rows_total} rows x {LOCATOR_ENTRY_BYTES})"
                )
            self._locator = raw
        return self._locator

    @property
    def ids(self) -> list[str]:
        """Row -> chunk_id, read and verified on first use."""
        if self._ids is None:
            text = self._artifact_bytes("ids").decode("utf-8")
            ids = text.split("\n")
            if ids and ids[-1] == "":
                ids.pop()
            if len(ids) != self.rows_total:
                raise CatalogError(
                    f"catalog: {IDS_FILE} holds {len(ids)} ids, expected {self.rows_total}"
                )
            self._ids = ids
        return self._ids

    def row_of(self, chunk_id: str) -> int:
        """The row holding a chunk_id. Builds the reverse map once, on first ask."""
        if self._by_chunk_id is None:
            self._by_chunk_id = {value: row for row, value in enumerate(self.ids)}
        row = self._by_chunk_id.get(chunk_id)
        if row is None:
            raise KeyError(f"catalog: no chunk_id {chunk_id}")
        return row

    # -- the row locator (late materialization) --------------------------

    def span(self, row: int) -> tuple[int, int, int]:
        """(file index, byte offset, byte length) for one row."""
        if not 0 <= row < self.rows_total:
            raise IndexError(f"catalog: row {row} outside this catalog's {self.rows_total} rows")
        start = row * LOCATOR_ENTRY_BYTES
        return _LOCATOR_STRUCT.unpack_from(self.locator, start)

    def fetch(self, rows) -> dict[int, dict]:
        """Read exactly the named rows by seeking to each one's recorded offset.

        Reads are grouped by file and issued in ascending offset order: one open per
        file that is touched at all, and none for a file no requested row lives in.
        What this replaces parsed every row of every file to return five of them.

        A span that does not decode, or that decodes to a row whose `chunk_id` is not
        the one recorded for it, is a stale catalog and raises -- rather than
        returning the wrong text under the right row number.
        """
        wanted: dict[int, list[tuple[int, int, int]]] = {}
        for row in rows:
            file_index, offset, length = self.span(int(row))
            wanted.setdefault(file_index, []).append((offset, length, int(row)))
        found: dict[int, dict] = {}
        ids = self.ids
        for file_index in sorted(wanted):
            path = self.files[file_index]
            with path.open("rb") as handle:
                for offset, length, row in sorted(wanted[file_index]):
                    handle.seek(offset)
                    raw = handle.read(length)
                    try:
                        record = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise CatalogError(
                            f"catalog: {path.name}@{offset}+{length} is not one JSON row "
                            f"({exc}); the catalog is stale -- rebuild it"
                        ) from exc
                    if record.get("chunk_id") != ids[row]:
                        raise CatalogError(
                            f"catalog: {path.name}@{offset} holds "
                            f"{record.get('chunk_id')!r}, not row {row}'s {ids[row]!r}; "
                            "the catalog is stale -- rebuild it"
                        )
                    found[row] = record
        return found

    # -- numeric columns (aggregates without a scan) ---------------------

    @property
    def numeric(self) -> dict[str, array]:
        """Each numeric column as a packed integer array, read on first use.

        Stored contiguously and per column, which is what makes an aggregate over a
        selection a gather rather than a walk of the corpus. A missing measurement is
        `NUMERIC_NULL` rather than zero: zero is a real length, and a format in which
        the two are the same cannot answer MIN honestly.
        """
        if self._numeric is None:
            raw = self._artifact_bytes("numeric")
            names = tuple(self._artifacts["numeric"].get("columns") or ())
            expected = len(names) * self.rows_total * NUMERIC_CELL_BYTES
            if len(raw) != expected:
                raise CatalogError(
                    f"catalog: {NUMERIC_FILE} holds {len(raw)} bytes, expected {expected} "
                    f"({len(names)} column(s) x {self.rows_total} rows x {NUMERIC_CELL_BYTES})"
                )
            values = array("q")
            values.frombytes(raw)
            if sys.byteorder == "big":
                values.byteswap()  # the file is little-endian by contract
            self._numeric = {
                name: values[index * self.rows_total : (index + 1) * self.rows_total]
                for index, name in enumerate(names)
            }
        return self._numeric

    @property
    def order(self) -> dict[str, array]:
        """Each measurement column's rows in ascending order, read on first use.

        Measured rows first -- ascending by value, ties by row -- then the unmeasured
        rows in row order. The split point is the column's own `count`, so the index
        carries no length of its own to disagree with the statistics.
        """
        if self._order is None:
            raw = self._artifact_bytes("order")
            names = tuple(self._artifacts["order"].get("columns") or ())
            expected = len(names) * self.rows_total * ORDER_ENTRY_BYTES
            if len(raw) != expected:
                raise CatalogError(
                    f"catalog: {ORDER_FILE} holds {len(raw)} bytes, expected {expected} "
                    f"({len(names)} column(s) x {self.rows_total} rows x {ORDER_ENTRY_BYTES})"
                )
            rows = array("I")
            rows.frombytes(raw)
            if sys.byteorder == "big":
                rows.byteswap()  # the file is little-endian by contract
            self._order = {
                name: rows[index * self.rows_total : (index + 1) * self.rows_total]
                for index, name in enumerate(names)
            }
        return self._order

    def measured(self, column: str) -> int:
        """How many rows carry a measurement -- where the sorted index splits."""
        self.require_numeric(column)
        return int(self._statistics["numeric"][column]["count"])

    def seek_bounds(self, column: str, low: int | None, high: int | None) -> tuple[int, int]:
        """The half-open slice of the sorted index that `[low, high]` names.

        Two binary searches, and the one place a comparison is turned into a window:
        `count_in_range` and `seek_range` both ask here, so the number of rows a plan
        is priced on and the rows it returns cannot come from different arithmetic
        (`docs/security/laws.md` L14).

        Unmeasured rows live past the split and are never inside the window, so a row
        carrying no value satisfies no comparison without anything testing for the
        sentinel: the SQL rule is in the layout rather than in a branch that could be
        forgotten on one path.
        """
        self.require_numeric(column)
        rows = self.order[column]
        split = self.measured(column)
        cells = self.numeric[column]
        start = (
            0
            if low is None
            else bisect.bisect_left(rows, low, 0, split, key=lambda row: cells[row])
        )
        stop = (
            split
            if high is None
            else bisect.bisect_right(rows, high, 0, split, key=lambda row: cells[row])
        )
        return start, max(start, stop)

    def seek_range(self, column: str, low: int | None = None, high: int | None = None):
        """The rows in `[low, high]`, in *value* order, plus what the seek cost.

        Value order rather than row order because the caller that needs rows sorted
        sorts once, and the caller that builds a set from them -- which is how a
        conjunction is evaluated -- would otherwise pay for an ordering nobody reads.
        """
        start, stop = self.seek_bounds(column, low, high)
        window = list(self.order[column][start:stop])
        return window, {"probes": 2, "window": len(window), "measured": self.measured(column)}

    def count_in_range(self, column: str, low: int | None = None, high: int | None = None) -> int:
        """Exactly how many rows are in `[low, high]`, without naming one of them."""
        start, stop = self.seek_bounds(column, low, high)
        return stop - start

    def numeric_columns(self) -> tuple[str, ...]:
        return tuple(self._statistics.get("numeric", {}).keys())

    def aggregate(self, column: str, rows=None) -> dict:
        """MIN, MAX, SUM, COUNT and AVG over a numeric column.

        With no selection this is answered from the statistics the build already
        computed -- no artifact is read and no row is touched. With a selection it
        gathers exactly the admitted cells from the packed column. `scanned` reports
        which happened, because "answered from statistics" and "answered by reading
        every admitted row" are different claims and only one of them is free.
        """
        self.require_numeric(column)
        stats = self._statistics["numeric"][column]
        if rows is None:
            return {
                "column": column,
                "count": int(stats["count"]),
                "nulls": int(stats["nulls"]),
                "min": stats["min"],
                "max": stats["max"],
                "sum": int(stats["sum"]),
                "avg": (stats["sum"] / stats["count"]) if stats["count"] else None,
                "scanned": 0,
            }
        values = self.numeric[column]
        admitted = [int(row) for row in rows]
        for row in admitted:
            if not 0 <= row < self.rows_total:
                raise IndexError(
                    f"catalog: row {row} outside this catalog's {self.rows_total} rows"
                )
        present = [values[row] for row in admitted if values[row] != NUMERIC_NULL]
        return {
            "column": column,
            "count": len(present),
            "nulls": len(admitted) - len(present),
            "min": min(present) if present else None,
            "max": max(present) if present else None,
            "sum": sum(present),
            "avg": (sum(present) / len(present)) if present else None,
            "scanned": len(present),
        }

    def require_numeric(self, column: str) -> None:
        """Refuse a column that has no order to compare against.

        The aggregate and every range term ask this one question through this one
        predicate, so `--stats` and `--where` can never disagree about which columns
        are measurements (`docs/security/laws.md` L14). Comparing `kind` against 500
        would otherwise succeed somewhere and fail somewhere else.
        """
        if column not in self.numeric_columns():
            known = ", ".join(self.numeric_columns()) or "(none)"
            raise KeyError(f"catalog: column {column!r} is not a numeric column; known: {known}")

    # -- ranges (the zone map first, then only the parts it could not settle) --

    def zone(self, part: dict, column: str) -> dict:
        """One part's summary of one numeric column, or a refusal.

        A part with no zone map is a catalog that cannot prune, and it has to say so:
        pruning changes only the cost of an answer, never the answer, so a silent
        fallback to reading every row produces an identical result and an identical
        green run. The absence has to be a verdict or it is invisible
        (`docs/security/laws.md` L1, L2).
        """
        zones = part.get("numeric")
        if not isinstance(zones, dict) or column not in zones:
            raise CatalogError(
                f"catalog: part {part.get('part_id')!r} carries no zone map for {column!r}\n"
                f"  rebuild it: python3 {BUILDER} --out {self.root}"
            )
        return zones[column]

    @staticmethod
    def _zone_verdict(zone: dict, low: int | None, high: int | None) -> str:
        """How one part stands to the closed interval `[low, high]`, from four integers.

        `none`    -- no row in this part can match; skip it unread.
        `all`     -- every row matches; take the whole span unread.
        `present` -- every *measured* row matches, and the part also holds gaps; the
                     count is known exactly but naming the rows needs the cells.
        `some`    -- the part straddles a bound; the cells decide.

        Nulls are excluded before anything else because the sentinel is a legal int64
        that compares below every real measurement: an interval left open at the
        bottom would otherwise sweep up every gap in the column.
        """
        if zone["count"] == 0:
            return "none"
        if low is not None and zone["max"] < low:
            return "none"
        if high is not None and zone["min"] > high:
            return "none"
        if (low is None or zone["min"] >= low) and (high is None or zone["max"] <= high):
            return "all" if zone["nulls"] == 0 else "present"
        return "some"

    def range_scan(self, column: str, low: int | None = None, high: int | None = None):
        """The rows in `[low, high]` by scanning the column, zone map first.

        The second, slower answer to the question `seek_range` answers -- kept, and
        kept correct, because a lookup that changes the result has nothing to be
        checked against. It is this rail's oracle/production-twin pattern applied
        inside one module: `verify_database.py` runs both over every interval and
        requires them to agree, which is a claim the sorted index alone could not
        make about itself.

        The returned counters are the audit trail -- parts ruled out on four integers,
        parts taken whole, cells actually read. Pruning changes only the cost of an
        answer, so a zone map that stopped pruning would keep every answer right; the
        counters are the only thing that notices.
        """
        self.require_numeric(column)
        values = None
        matched: list[int] = []
        read = {"parts": len(self.parts), "skipped": 0, "whole": 0, "scanned": 0, "cells": 0}
        for part in self.parts:
            start, end = part["rows"]
            verdict = self._zone_verdict(self.zone(part, column), low, high)
            if verdict == "none":
                read["skipped"] += 1
                continue
            if verdict == "all":
                read["whole"] += 1
                matched.extend(range(start, end))
                continue
            read["scanned"] += 1
            if values is None:
                values = self.numeric[column]
            read["cells"] += end - start
            for row in range(start, end):
                value = values[row]
                if value == NUMERIC_NULL:
                    continue
                if low is not None and value < low:
                    continue
                if high is not None and value > high:
                    continue
                matched.append(row)
        return matched, read

    def range_strategy(self, column: str, low: int | None, high: int | None) -> str:
        """`"seek"` or `"scan"`: how an ascending answer to this range will be produced.

        Decided from the *exact* count rather than from the zone map's bound. The bound
        was tried first and measured unusable for this: over this corpus it calls a
        25%-selective predicate 98% selective at every block size from 32 rows to a
        whole file, because an open interval keeps any block holding one large value,
        however few of that block's rows are in range. Deciding on it sends predicates
        the seek wins by 20x down the scan.

        The exact count costs the two binary searches the seek would make anyway, and
        the index they read is cached for the rest of the process -- so the price of
        choosing well is paid once, and only by a caller that asked about a range.

        Returned as a value rather than branched on inside `rows_in_range`, because
        both paths return identical rows: a gate comparing their output passes
        whichever one ran, so the decision would otherwise be the one thing here that
        nothing could observe (`docs/security/laws.md` L2).
        """
        self.require_numeric(column)
        if not self.rows_total:
            return "seek"
        share = self.count_in_range(column, low, high) / self.rows_total
        return "scan" if share >= SORTED_SCAN_SHARE else "seek"

    def rows_in_range(self, column: str, low: int | None = None, high: int | None = None):
        """The rows in `[low, high]`, ascending, by whichever path is cheaper here."""
        if self.range_strategy(column, low, high) == "scan":
            return self.range_scan(column, low, high)[0]
        return sorted(self.seek_range(column, low, high)[0])

    # -- presence (IS NULL / IS NOT NULL) --------------------------------

    def rows_absent(self, column: str) -> list[int]:
        """Rows carrying no value for `column`. SQL's IS NULL, over any indexed column."""
        if column in self.numeric_columns():
            return self._numeric_presence(column, want_present=False)
        if column == PATH_COLUMN:
            seen = self._path_rows()
            return [row for row in range(self.rows_total) if row not in seen]
        postings = self.postings["columns"].get(column)
        if postings is None:
            raise KeyError(f"catalog: column {column!r} is not indexed")
        return sorted(postings.get(NULL_KEY, ()))

    def rows_present(self, column: str) -> list[int]:
        """Rows carrying a value for `column`. SQL's IS NOT NULL.

        Gathered, not derived as the complement of `rows_absent`. Defining either as
        the negation of the other would make "these two partition the table" true by
        construction, and a property that cannot fail is not a property a gate can
        check (`docs/security/laws.md` L2). Held apart, the partition is a differential
        between two walks, and `check_presence` can watch it break.
        """
        if column in self.numeric_columns():
            return self._numeric_presence(column, want_present=True)
        if column == PATH_COLUMN:
            return sorted(self._path_rows())
        postings = self.postings["columns"].get(column)
        if postings is None:
            raise KeyError(f"catalog: column {column!r} is not indexed")
        gathered: set[int] = set()
        for key, rows in postings.items():
            if key != NULL_KEY:
                gathered.update(rows)
        return sorted(gathered)

    def _path_rows(self) -> set[int]:
        gathered: set[int] = set()
        for rows in self.postings["paths"].values():
            gathered.update(rows)
        return gathered

    def _numeric_presence(self, column: str, *, want_present: bool) -> list[int]:
        """Measured or unmeasured rows of a packed column, skipping parts the zone settles."""
        values = None
        found: list[int] = []
        for part in self.parts:
            start, end = part["rows"]
            zone = self.zone(part, column)
            if zone["nulls"] == 0:
                if want_present:
                    found.extend(range(start, end))
                continue
            if zone["count"] == 0:
                if not want_present:
                    found.extend(range(start, end))
                continue
            if values is None:
                values = self.numeric[column]
            for row in range(start, end):
                if (values[row] != NUMERIC_NULL) == want_present:
                    found.append(row)
        return found

    # -- statistics (selectivity, exactly) -------------------------------

    def distinct(self, column: str) -> dict[str, int]:
        """Row count per value for an indexed column."""
        counts = self._statistics["columns"].get(column)
        if counts is None:
            raise KeyError(f"catalog: column {column!r} is not indexed")
        return counts

    def indexed_columns(self) -> tuple[str, ...]:
        return tuple(self._statistics["columns"].keys())

    def count_equals(self, column: str, value) -> int | None:
        """Exact row count for ``column = value``, or None when it is not indexed."""
        if column == PATH_COLUMN:
            return int(self._statistics["paths"].get(str(value), 0))
        counts = self._statistics["columns"].get(column)
        if counts is None:
            return None
        return int(counts.get(index_key(value), 0))

    def has_pairs(self, left: str, right: str) -> bool:
        """Whether exact co-occurrence counts were stored for these two columns."""
        return pair_key(left, right) in self._statistics.get("pairs", {})

    def count_pair(self, left: str, left_value, right: str, right_value) -> int:
        """Exact rows carrying both values, from the stored joint distribution.

        A pair the build never saw is absent from the table and counts zero -- which
        is the common answer and the one the marginals could never give: most value
        pairs in a corpus do not co-occur at all, and calling that "at most the
        smaller marginal" is the estimate that was 100% over on every empty pair.
        """
        table = self._statistics.get("pairs", {}).get(pair_key(left, right))
        if table is None:
            raise CatalogError(
                f"catalog: no joint statistic for {left!r} and {right!r}; the catalog "
                f"pairs {', '.join(sorted(self.indexed_columns()))}"
            )
        # `pair_key` sorted the names, so the table's outer key is the lower name.
        if left > right:
            left_value, right_value = right_value, left_value
        return int(table.get(index_key(left_value), {}).get(index_key(right_value), 0))

    def count_prefix(self, prefix: str) -> tuple[int, bool]:
        """Rows whose `source_path` starts with `prefix`, and whether that is exact.

        Exact when the prefix names a counted directory or a whole path. Otherwise
        the nearest counted ancestor's count is returned as an upper bound, with
        False -- so the caller prices a bound as a bound.
        """
        prefixes = self._statistics["path_prefixes"]
        cleaned = prefix.rstrip("/")
        if cleaned in prefixes:
            return int(prefixes[cleaned]), True
        whole = self._statistics["paths"].get(prefix)
        if whole is not None:
            return int(whole), True
        parts = cleaned.split("/")
        for depth in range(len(parts) - 1, 0, -1):
            ancestor = "/".join(parts[:depth])
            if ancestor in prefixes:
                return int(prefixes[ancestor]), False
        return self.rows_total, False

    # -- postings (which rows, not how many) -----------------------------

    def rows_equal(self, column: str, value) -> list[int] | None:
        """The rows where ``column = value``, or None when the column is not indexed."""
        if column == PATH_COLUMN:
            return list(self.postings["paths"].get(str(value), ()))
        column_postings = self.postings["columns"].get(column)
        if column_postings is None:
            return None
        return list(column_postings.get(index_key(value), ()))

    def rows_with_prefix(self, prefix: str) -> list[int]:
        """The rows whose `source_path` starts with `prefix`, exactly.

        A counted directory answers from the index. Anything else -- a partial
        segment, a prefix no row shares -- falls back to scanning the path postings,
        which is still only over distinct paths rather than over rows.
        """
        cleaned = prefix.rstrip("/")
        by_prefix = self.postings["path_prefixes"]
        if cleaned in by_prefix:
            return list(by_prefix[cleaned])
        matched: list[int] = []
        for path, positions in self.postings["paths"].items():
            if path.startswith(prefix):
                matched.extend(positions)
        return sorted(matched)

    # -- parts (incremental rebuild) -------------------------------------

    def part_of(self, row: int) -> dict:
        """The part holding a row, by bisection rather than by walking every part.

        Parts partition the rows in ascending order, which the gate checks, so the
        first part starting after `row` is one past the answer. Splitting files into
        blocks multiplied the part count; a linear walk would have quietly turned this
        into work proportional to the corpus.
        """
        starts = self._part_starts()
        index = bisect.bisect_right(starts, row) - 1
        if 0 <= index < len(self.parts):
            start, end = self.parts[index]["rows"]
            if start <= row < end:
                return self.parts[index]
        raise IndexError(f"catalog: row {row} belongs to no part")

    def _part_starts(self) -> list[int]:
        if self._part_starts_cache is None:
            self._part_starts_cache = [int(part["rows"][0]) for part in self.parts]
        return self._part_starts_cache

    def changed_parts(self, other: "Catalog") -> tuple[str, ...]:
        """Part ids whose content differs from `other`'s, or that only one side has."""
        mine = {part["part_id"]: part["content"] for part in self.parts}
        theirs = {part["part_id"]: part["content"] for part in other.parts}
        names = mine.keys() | theirs.keys()
        return tuple(sorted(name for name in names if mine.get(name) != theirs.get(name)))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--stats", action="store_true", help="print the column index")
    args = parser.parse_args(argv)

    try:
        built = build(args.chunks)
    except CatalogError as exc:
        print(exc)
        return 1
    manifest = built[0]
    target = write(built, args.out)
    print(f"[catalog] {manifest['rows_total']} row(s) in {len(manifest['parts'])} part(s)")
    print(f"[write]   {target}")
    if args.stats:
        loaded = Catalog.load(args.out, args.chunks)
        for column in loaded.indexed_columns():
            counts = loaded.distinct(column)
            shown = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            print(f"  {column:12s} {len(counts):3d} distinct  {shown[:110]}")
        prefixes = manifest["statistics"]["path_prefixes"]
        print(f"  {'path prefix':12s} {len(prefixes):3d} counted directories")
        for name, entry in sorted(manifest["artifacts"].items()):
            size = (args.out / entry["path"]).stat().st_size
            print(f"  {name:12s} {entry['path']:14s} {size / 1024:8.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
