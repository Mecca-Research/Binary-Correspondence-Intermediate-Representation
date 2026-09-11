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

The artifacts are split by how often they are wanted, and each is read on first
use, never on load:

  catalog.json   manifest, parts, statistics -- small, always read
  postings.json  the inverted index -- read when a predicate is evaluated
  locator.bin    14 bytes per row -- read when text is fetched
  ids.txt        row -> chunk_id -- read only for lookups by primary key

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

**Staleness is loud, and checked in two levels.** The manifest records every
chunk file's size, modification time and content digest. A load compares size and
mtime first and re-hashes only the files whose stat moved; a file whose stat moved
but whose bytes did not is still fresh. Any file that is missing, added, or whose
digest changed refuses the load -- answering from stale statistics produces a
wrong plan silently, which is worse than not answering. `verify_content=True`
skips the stat level and hashes everything; the gate uses it, because a fast path
nobody ever checks is a fast path that can be wrong for a year.

Declared scope of the stat level: a chunk file rewritten with its size and
nanosecond mtime deliberately preserved is not distinguished until something asks
for the content check. Nothing else is out of scope -- files appearing,
disappearing, or changing length or content are all caught.

**The direction of dependency is unchanged.** `training/` is never a build
dependency of BCIR, and nothing here reverses that.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

SCHEMA = "bcir-training/catalog/v1"
BUILDER = "training/tools/catalog.py"
LICENSE = "LicenseRef-BCIR-NC-1.0"

DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_CATALOG = Path("build/training/catalog")

CATALOG_FILE = "catalog.json"
POSTINGS_FILE = "postings.json"
LOCATOR_FILE = "locator.bin"
IDS_FILE = "ids.txt"

#: Columns indexed by whole value. Low cardinality by construction, so a
#: distinct-value map over any of them is a handful of entries, not a histogram.
INDEXED_COLUMNS = ("subject", "kind", "language")

#: The column indexed by every directory prefix rather than by whole value.
PATH_COLUMN = "source_path"

#: How a missing value is spelled in an index key. JSON object keys are strings,
#: so `null` needs a spelling that no real value can collide with.
NULL_KEY = "\0null"

#: One locator entry: file index, byte offset, byte length. Little-endian and
#: unpadded, so the file is byte-identical on every host -- the same rule the
#: embedding set's `vectors.q15` follows.
_LOCATOR_STRUCT = struct.Struct("<HQI")
LOCATOR_ENTRY_BYTES = _LOCATOR_STRUCT.size


class CatalogError(RuntimeError):
    """The catalog is absent, stale, or malformed. Never a silent fallback."""


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


def chunk_files(chunk_dir: Path) -> list[Path]:
    """Every chunk file, in one order, so two callers cannot disagree about the set.

    This order is load-bearing beyond this module: `embed_chunks.py` walks the same
    sorted glob, so position `i` here is row `i` of the embedding set. That
    correspondence is what lets a ranked row be turned into bytes on disk without a
    lookup, and `verify_database.py` asserts it rather than trusting it.
    """
    return sorted(chunk_dir.glob("*.chunks.jsonl"))


def _file_stat(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
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


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def _scan_chunk_file(path: Path, file_index: int) -> list[dict]:
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
            rows.append(
                {
                    "chunk_id": record.get("chunk_id"),
                    "subject": record.get("subject"),
                    "source_path": record.get("source_path"),
                    "kind": record.get("kind"),
                    "language": record.get("language"),
                    "file": file_index,
                    "offset": offset,
                    "length": end - offset,
                }
            )
        offset = end + 1
    return rows


def build(chunk_dir: Path = DEFAULT_CHUNKS) -> tuple[dict, dict, bytes, str]:
    """Read the chunk table once and derive every fact a plan can use.

    One pass. The locator, the postings, the statistics and the parts all come from
    the rows that scan already produced -- a catalog needing its own second pass
    over the corpus would cost more than the scans it exists to avoid.

    Returns the manifest, the postings, the packed locator, and the id list.
    """
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
        rows.extend(_scan_chunk_file(path, index))
        stat = _file_stat(path)
        stats_files.append(stat)
        parts.append(
            {
                "part_id": path.name[: -len(".chunks.jsonl")],
                "file": path.name,
                "rows": [start, len(rows)],
                "content": stat["sha256"],
            }
        )

    seen: dict[str, int] = {}
    for position, row in enumerate(rows):
        chunk_id = row["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id:
            raise CatalogError(f"catalog: row {position} has no chunk_id")
        if chunk_id in seen:
            raise CatalogError(
                f"catalog: chunk_id {chunk_id} appears at rows {seen[chunk_id]} and "
                f"{position}; a primary key that is not unique cannot locate a row"
            )
        seen[chunk_id] = position

    postings_columns: dict[str, dict[str, list[int]]] = {}
    for name in INDEXED_COLUMNS:
        column: dict[str, list[int]] = {}
        for position, row in enumerate(rows):
            column.setdefault(index_key(row.get(name)), []).append(position)
        postings_columns[name] = dict(sorted(column.items()))

    prefix_postings: dict[str, list[int]] = {}
    path_postings: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        source_path = row.get(PATH_COLUMN)
        if not isinstance(source_path, str):
            continue
        path_postings.setdefault(source_path, []).append(position)
        for prefix in path_prefixes(source_path):
            prefix_postings.setdefault(prefix, []).append(position)

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

    manifest = {
        "schema": SCHEMA,
        "builder": BUILDER,
        "license": LICENSE,
        "chunk_dir": str(chunk_dir).replace(os.sep, "/"),
        "fingerprint": fingerprint(chunk_dir),
        "rows_total": len(rows),
        "files": stats_files,
        "parts": parts,
        "statistics": {
            "columns": {
                name: {value: len(positions) for value, positions in column.items()}
                for name, column in postings_columns.items()
            },
            "path_prefixes": {
                prefix: len(positions) for prefix, positions in postings["path_prefixes"].items()
            },
            "paths": {path: len(positions) for path, positions in postings["paths"].items()},
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
        },
    }
    return manifest, postings, bytes(locator), ids


def _publish(root: Path, name: str, payload: bytes) -> Path:
    """Write one artifact atomically. No reader ever sees half of one."""
    target = root / name
    descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.tmp-", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def write(built: tuple[dict, dict, bytes, str], root: Path = DEFAULT_CATALOG) -> Path:
    """Publish every artifact, the manifest last.

    Last because the manifest is what vouches for the others' digests: a reader that
    finds the manifest can rely on the sidecars beside it already being complete.
    """
    manifest, postings, locator, ids = built
    root.mkdir(parents=True, exist_ok=True)
    _publish(
        root,
        POSTINGS_FILE,
        json.dumps(postings, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    _publish(root, LOCATOR_FILE, locator)
    _publish(root, IDS_FILE, ids.encode("utf-8"))
    return _publish(
        root,
        CATALOG_FILE,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


class Catalog:
    """A loaded catalog, bound to the chunk directory it describes.

    Only the manifest is read here. Every other artifact is loaded by the first
    call that needs it and verified against the digest the manifest recorded.
    """

    def __init__(self, manifest: dict, root: Path, chunk_dir: Path) -> None:
        self.manifest = manifest
        self.root = root
        self.chunk_dir = chunk_dir
        self.rows_total = int(manifest["rows_total"])
        self.files = [chunk_dir / entry["name"] for entry in manifest["files"]]
        self.parts = manifest["parts"]
        self._statistics = manifest["statistics"]
        self._artifacts = manifest["artifacts"]
        self._postings: dict | None = None
        self._locator: bytes | None = None
        self._ids: list[str] | None = None
        self._by_chunk_id: dict[str, int] | None = None

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        root: Path = DEFAULT_CATALOG,
        chunk_dir: Path = DEFAULT_CHUNKS,
        *,
        verify_content: bool = False,
    ) -> "Catalog":
        """Load the catalog, or refuse. A stale catalog is never answered from.

        Every failure is the same verdict -- a `CatalogError` naming the rebuild
        command -- because a caller that cannot tell "absent" from "stale" from
        "malformed" will eventually treat one of them as "fine".
        """
        target = root / CATALOG_FILE
        try:
            manifest = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CatalogError(
                f"catalog: none at {target}\n  build one first: python3 {BUILDER} --out {root}"
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
        cls._check_freshness(manifest, root, chunk_dir, verify_content=verify_content)
        return cls(manifest, root, chunk_dir)

    @staticmethod
    def _check_freshness(
        manifest: dict, root: Path, chunk_dir: Path, *, verify_content: bool
    ) -> None:
        """Two levels: stat first, content for whatever the stat says may have moved.

        `verify_content` skips the first level entirely. Both levels end in the same
        comparison against the recorded digest, so the fast path can only ever skip
        work that would have agreed with it.
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
        suspect = []
        for name, entry in sorted(recorded.items()):
            path = actual[name]
            if verify_content:
                suspect.append((name, path, entry))
                continue
            stat = path.stat()
            if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
                suspect.append((name, path, entry))
        moved = []
        for name, path, entry in suspect:
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
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
        path = self.root / entry["path"]
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
        for part in self.parts:
            start, end = part["rows"]
            if start <= row < end:
                return part
        raise IndexError(f"catalog: row {row} belongs to no part")

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
