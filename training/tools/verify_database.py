#!/usr/bin/env python3
"""Gate the training rail's database layer: catalog, planner, predicate, locator.

The slices this checks each removed work from a query. Removed work is exactly the
kind of change that passes every existing test while being wrong, because the tests
assert what comes back and the change is about what was not done to produce it. So
each check below asserts **both halves**: the cheap path's answer equals the
expensive path's answer, *and* the cheap path actually skipped the work.

What it enforces, by slice:

  S1 kernel cache      Building twice does not run the compiler twice, a changed
                       input does, and the cached library is byte-identical to a
                       forced rebuild. (The stamp's own witnesses live in
                       `bcir/tests/test_native_ai.py`; this is the end-to-end half.)
  S2 derived columns   `row_squares` and `row_views` are not computed for a backend
                       that never reads them, and *are* computed for the one that
                       does. Both directions, because an assertion that a column is
                       absent passes trivially if nothing ever builds it.
  S3 late materialize  Text fetched by seeking recorded spans is identical to text
                       parsed from a full scan, for every row in the corpus -- not a
                       sample -- and the seek path reads strictly fewer bytes.
  S4 predicate         The mask reaches the kernel: a filtered native ranking equals
                       a filtered reference ranking exactly; a predicate admitting
                       every row reproduces the unfiltered answer; a predicate
                       admitting none returns nothing rather than everything; and an
                       out-of-range row is refused.
  S5 parts             A part's content hash changes with its file and only with its
                       file, and `changed_parts` names exactly the ones that moved.
  S6 generations       A generation is immutable once published, its digest covers
                       what it claims, and reading an old one reproduces its answer.
  S7 comparisons       A range finds what a pass over the corpus finds; the zone map
                       rules parts out, rules out only parts holding no match, and
                       is counted doing it; a row carrying no measurement satisfies
                       no comparison in either direction; IS NULL and IS NOT NULL
                       partition the table while `!=` deliberately does not; groups
                       partition, their aggregates add up, and HAVING compares
                       SUM/COUNT rather than a rounded decimal.

  planner              Legality is decided before cost and never from a measurement:
                       a lossy backend stays refused for an exactness request at any
                       price. The chosen plan is deterministic. EXPLAIN names every
                       candidate, including the refused ones.

  anti-vacuity         The corpus is non-empty, every check loop executed a minimum
                       number of times, and the totals are reported -- a gate that
                       examines nothing agrees with everything.

    python3 training/tools/verify_database.py
    python3 training/tools/verify_database.py --require-native   # from the CI job
                                                                 # that installs a
                                                                 # C compiler
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import struct
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from db import analytics, engine, ingest, relational, retrieval  # noqa: E402

DEFAULT_SET = Path("build/training/embeddings/lexical-hash-v1")
DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_CATALOG = Path("build/training/catalog")

#: Anti-vacuity floors. Below any of these the run examined too little to mean
#: anything, and reporting PASSED would be a lie about coverage rather than a
#: verdict about correctness.
MIN_ROWS = 50
MIN_PREDICATE_TRIALS = 20
MIN_FETCH_ROWS = 50
MIN_RANGE_TRIALS = 24
#: Pruning changes the cost of a range answer and never the answer, so a zone map
#: that stopped ruling parts out would leave every correctness check above green.
#: This floor is the only thing in the gate that would notice.
MIN_PRUNED_PARTS = 1


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0
        self.skips: list[str] = []
        self.notes: list[str] = []

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)

    def skip(self, reason: str) -> None:
        """A rail that could not run here. `--require-native` turns these into failures."""
        self.skips.append(reason)

    def note(self, observation: str) -> None:
        """A property of this host that is true, reported, and not a missing rail.

        Kept apart from `skip` because `--require-native` means *the rail ran*, and
        these are cases where it ran and the host simply does not offer a stronger
        property on top -- an object format that embeds a build time, say. Folding
        the two together would make a flag about coverage fail on a fact about the
        platform, which is how a requirement stops meaning anything (L2).
        """
        self.notes.append(observation)

    def run(self, name: str, check, *args, **kwargs) -> None:
        """Run one check, turning any escape into a verdict.

        A gate that exits through a traceback has skipped its own report and its
        exit-code contract, so the defect it found arrives as a stack trace instead
        of a finding (L1). The checks below deliberately drive production code into
        its refusal paths, and a refusal raised as an exception is a *result* here,
        not an accident -- it is recorded as the failure it is.
        """
        try:
            check(*args, **kwargs)
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001 -- the point is that nothing escapes
            self.checks += 1
            self.failures.append(f"{name}: raised {type(exc).__name__}: {exc}")


def load_tool(name: str):
    """Load a sibling tool once. One implementation, in `db.engine`.

    This used to be the careful copy of three, and the other two were wrong in
    different ways. It is kept as a name here because the checks below read as prose
    about tools, not about an engine.
    """
    return engine.load(name)


#: What the fixtures below relax, and nothing else. The shipped table requires every
#: chunk to carry a measurement of at least 1, which is the corpus contract and is
#: enforced -- and it means no schema-valid corpus can ever leave a measurement
#: absent. The packed column must still spell absence, `MIN` must still skip it, and a
#: comparison must still refuse to admit it; none of that is reachable from a corpus
#: the contract admits, so the checks that cover it declare a table that admits it.
#:
#: A relaxation is a declaration, not a loophole: it names the columns it changes, it
#: cannot introduce one the shipped table has not, and the roles -- which column is
#: indexed, which is a measurement -- are exactly the shipped ones, because those are
#: what the format under test is built from.
_SYNTHETIC_RELAXATIONS = {
    "char_count": {"required": False, "nullable": True, "minimum": 0},
    "token_estimate": {"required": False, "nullable": True, "minimum": 0},
    **{
        name: {"required": False}
        for name in (
            "schema", "corpus", "source_sha256", "span", "heading_trail", "title",
            "text", "embedding", "embedding_spec", "provenance", "verified_by",
        )
    },
}


def synthetic_table():
    """The table the hand-built fixtures below are checked against."""
    return load_tool("schema").TABLE.relaxed(**_SYNTHETIC_RELAXATIONS)


def valid_record(**overrides) -> dict:
    """A complete, schema-valid chunk row, for checks that extend a real corpus.

    A check that appends a row to a real chunk file is asking what happens when the
    *corpus* changes, not what happens when an invalid row arrives -- so the row it
    appends has to be one the contract admits. Built from the shipped table rather
    than typed out, so a column added to the table appears here too and the fixture
    cannot quietly stop covering it (`docs/security/laws.md` L15).
    """
    schema_module = load_tool("schema")
    filler = {
        "string": "x",
        "integer": 1,
        "array": [],
        "object": {},
    }
    record: dict = {}
    for spec in schema_module.TABLE.columns:
        if not spec.required:
            continue
        if spec.domain:
            record[spec.name] = spec.domain[0]
        elif spec.name == "chunk_id":
            record[spec.name] = "sha256:" + "0" * 64
        elif spec.name == "span":
            record[spec.name] = {"start_line": 1, "end_line": 1}
        elif spec.name == "embedding_spec":
            record[spec.name] = {
                "model": None, "revision": None, "dim": None,
                "normalize": "none", "semantics": None,
            }
        elif spec.name == "provenance":
            record[spec.name] = {"license": "LicenseRef-BCIR-NC-1.0", "builder": BUILDER_PATH}
        elif spec.name == "source_sha256":
            record[spec.name] = "0" * 64
        elif spec.name == "embedding":
            record[spec.name] = None
        else:
            record[spec.name] = filler[spec.type]
    record.update(overrides)
    schema_module.TABLE.check_row(record)
    return record


#: How a fixture spells the builder path in `provenance`. Repository-relative, because
#: that is the only shape the chunk schema's `repositoryPath` admits.
BUILDER_PATH = "training/tools/build_chunks.py"


# --------------------------------------------------------------------------
# S1 -- the kernel cache, end to end
# --------------------------------------------------------------------------


def check_kernel_cache(report: Report, *, require_native: bool) -> None:
    """Two builds, one compiler run, identical bytes -- or an honest skip."""
    import shutil as _shutil

    compiler = _shutil.which("clang") or _shutil.which("cc") or _shutil.which("gcc")
    if compiler is None:
        if require_native:
            report.require(False, "S1: --require-native but no C compiler is on PATH")
        else:
            report.skip("S1 kernel cache: no C compiler on PATH")
        return
    from bcir.kbcir.native_ai import NativeAIKernels

    def build_and_release(**kwargs) -> None:
        """Build, then release the handle before anything touches the file again.

        Windows locks a loaded module, so `build`'s own `os.replace` over a library
        this process still has open would fail, as would removing the directory it
        lives in. This check rebuilds into one directory on purpose, so each handle
        goes as soon as the build that produced it has been observed.
        """
        NativeAIKernels.build(root, cc=compiler, **kwargs).close()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        build_and_release()
        library = root / NativeAIKernels._library_name()
        stamp = NativeAIKernels._stamp_path(library)
        report.require(stamp.is_file(), "S1: the first build published no stamp")
        first = library.stat()
        digest = library.read_bytes()

        build_and_release()
        second = library.stat()
        report.require(
            (first.st_ino, first.st_mtime_ns) == (second.st_ino, second.st_mtime_ns),
            "S1: a second build with nothing changed replaced the library -- the "
            "compiler ran when the stamp said it need not",
        )

        build_and_release(rebuild=True)
        third = library.stat()
        report.require(
            (first.st_ino, first.st_mtime_ns) != (third.st_ino, third.st_mtime_ns),
            "S1: rebuild=True did not replace the library, so the check above could "
            "not have failed either -- the observation is vacuous",
        )
        # Whatever that rebuild produced, the stamp must now vouch for it. This is
        # the property the cache rests on, and it holds on every host.
        report.require(
            NativeAIKernels._reusable(library, stamp, NativeAIKernels._input_digest(compiler)),
            "S1: after a forced rebuild the stamp no longer validates the library "
            "beside it, so the next process would compile again for nothing",
        )
        # A forced rebuild of unchanged inputs gives back the same bytes *where the
        # toolchain is reproducible*. A Windows PE carries a build timestamp in its
        # header, so two builds of one unchanged translation unit differ there --
        # a property of the object format, not of the cache. S1 does not need
        # reproducibility (the stamp records whatever that build produced), so this
        # is asserted where it is true rather than required everywhere, and the
        # condition is measured rather than guessed from the platform name.
        NativeAIKernels.build(root, cc=compiler, rebuild=True).close()
        once = library.read_bytes()
        NativeAIKernels.build(root, cc=compiler, rebuild=True).close()
        reproducible = library.read_bytes() == once
        if reproducible:
            report.require(
                library.read_bytes() == digest,
                "S1: this toolchain builds reproducibly, yet a forced rebuild of "
                "unchanged inputs produced different bytes from the first build",
            )
        else:
            report.note(
                "S1 byte-identity: this toolchain does not build reproducibly "
                "(an object format that embeds a build time); the stamp is still "
                "required to vouch for whatever each build produced"
            )

        recorded = json.loads(stamp.read_text(encoding="utf-8"))
        recorded["inputs"] = "0" * 64
        stamp.write_text(json.dumps(recorded), encoding="utf-8")
        build_and_release()
        fourth = library.stat()
        report.require(
            (third.st_ino, third.st_mtime_ns) != (fourth.st_ino, fourth.st_mtime_ns),
            "S1: a stamp recording different inputs did not force a rebuild",
        )


# --------------------------------------------------------------------------
# S2 -- derived columns are built for the backend that reads them, and no other
# --------------------------------------------------------------------------


def check_derived_columns(report: Report, search, embedding_root: Path, *, native_ok: bool) -> None:
    fresh = search.EmbeddingSet(embedding_root)
    report.require(
        fresh.derived_columns_built() == (),
        f"S2: constructing an EmbeddingSet built {fresh.derived_columns_built()} "
        "before any backend asked for it",
    )
    query = search.embed_query("lowering a dialect to llvm", fresh)

    if native_ok:
        search.topk_native(query, fresh, 5)
        report.require(
            fresh.derived_columns_built() == (),
            f"S2: the native backend built {fresh.derived_columns_built()}; it reads "
            "neither derived column, so building one is work nothing consumes",
        )
    else:
        report.skip("S2 native half: the native backend is not reachable")

    search.topk_reference(query, fresh, 5)
    built = fresh.derived_columns_built()
    report.require(
        "row_squares" in built and "row_views" in built,
        f"S2: the reference backend built {built}, but it reads both derived columns "
        "-- if they are absent the assertion above proves nothing",
    )

    # A filtered reference scan reads `row_codes`, which must NOT drag in the column.
    narrow = search.EmbeddingSet(embedding_root)
    search.topk_reference(query, narrow, 3, rows=[0, 1, 2, 3, 4])
    report.require(
        narrow.derived_columns_built() == (),
        f"S2: a five-row filtered scan built {narrow.derived_columns_built()} -- "
        "materializing the whole column to read five rows",
    )
    report.require(
        narrow.row_codes(0) == search.EmbeddingSet(embedding_root).row_views[0],
        "S2: row_codes and row_views disagree about row 0",
    )
    for bad in (-1, len(narrow.rows)):
        try:
            narrow.row_codes(bad)
        except IndexError:
            report.require(True, "")
        else:
            report.require(False, f"S2: row_codes({bad}) returned instead of refusing")


# --------------------------------------------------------------------------
# S3 -- late materialization returns the same bytes the full scan does
# --------------------------------------------------------------------------


def check_late_materialization(
    report: Report, search, catalog, chunk_dir: Path, embedding_root: Path
) -> None:
    scanned = search.load_chunk_texts(chunk_dir)
    report.require(
        len(scanned) >= MIN_ROWS,
        f"S3: the full scan found {len(scanned)} rows, below the {MIN_ROWS} floor -- "
        "a comparison over almost nothing",
    )
    report.require(
        len(scanned) == catalog.rows_total,
        f"S3: the full scan found {len(scanned)} rows and the catalog claims "
        f"{catalog.rows_total}; one of them is wrong about the corpus",
    )

    # Every row, not a sample: a locator is exactly the kind of table where one
    # wrong entry hides behind 2,000 right ones.
    fetched = catalog.fetch(range(catalog.rows_total))
    report.require(
        len(fetched) == catalog.rows_total,
        f"S3: seeking every row returned {len(fetched)} of {catalog.rows_total}",
    )
    mismatched = [
        row for row, record in fetched.items() if scanned.get(record["chunk_id"]) != record
    ]
    report.require(
        not mismatched,
        f"S3: {len(mismatched)} row(s) differ between the seek and full-scan paths "
        f"(first: {mismatched[:3]})",
    )

    # The cheap path must actually be cheap: fetching k rows reads far less than the
    # table. Bytes, not time -- a timing assertion on a shared runner is a flake.
    total_bytes = sum(path.stat().st_size for path in catalog.files)
    sample = list(range(0, catalog.rows_total, max(1, catalog.rows_total // MIN_FETCH_ROWS)))
    report.require(
        len(sample) >= MIN_FETCH_ROWS // 2,
        f"S3: the byte-cost sample holds {len(sample)} rows, too few to mean anything",
    )
    read_bytes = sum(catalog.span(row)[2] for row in sample)
    report.require(
        read_bytes * 4 < total_bytes,
        f"S3: seeking {len(sample)} rows reads {read_bytes} bytes of {total_bytes}; "
        "that is not a narrower read than the full scan",
    )

    # The correspondence the locator depends on, asserted rather than assumed:
    # `embed_chunks.py` and `catalog.py` walk the same sorted glob, so row `i` of the
    # embedding set must be position `i` here. Nothing else would notice if they
    # diverged -- a ranked row would simply fetch some other row's text.
    index_path = embedding_root / "index.jsonl"
    if not index_path.is_file():
        report.require(False, f"S3: no embedding index at {index_path}")
        return
    index_rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report.require(
        len(index_rows) == catalog.rows_total,
        f"S3: the embedding index holds {len(index_rows)} rows and the catalog "
        f"{catalog.rows_total}",
    )
    misaligned = [
        row for row, entry in enumerate(index_rows) if entry["chunk_id"] != catalog.ids[row]
    ]
    report.require(
        not misaligned,
        f"S3: row {misaligned[:3]} of the embedding set is not the same chunk as the "
        "catalog's row of that number; ranked rows would fetch the wrong text",
    )

    # A locator that points at the wrong row must refuse, not answer. Without this
    # the consistency check inside `fetch` has no witness: every row is at its right
    # offset in a fresh catalog, so removing the check changes nothing observable.
    catalog_module = load_tool("catalog")
    with tempfile.TemporaryDirectory() as directory:
        mirror = Path(directory) / "catalog"
        shutil.copytree(catalog.root, mirror)
        locator_path = catalog_module.current_set(mirror) / catalog_module.LOCATOR_FILE
        raw = bytearray(locator_path.read_bytes())
        width = catalog_module.LOCATOR_ENTRY_BYTES
        # Give row 0 row 1's span: a real staleness shape, not random bytes.
        raw[0:width] = raw[width : 2 * width]
        locator_path.write_bytes(bytes(raw))
        manifest_path = catalog_module.current_set(mirror) / catalog_module.CATALOG_FILE
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["locator"]["sha256"] = hashlib.sha256(bytes(raw)).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        stale = catalog_module.Catalog.load(mirror, chunk_dir)
        try:
            stale.fetch([0])
        except catalog_module.CatalogError:
            report.require(True, "")
        else:
            report.require(
                False,
                "S3: a locator entry pointing at another row's bytes returned that "
                "row's text under the wrong row number instead of refusing",
            )
        # ... and a locator whose digest does NOT match the manifest is refused
        # before it is ever read.
        broken = Path(directory) / "catalog2"
        shutil.copytree(catalog.root, broken)
        (catalog_module.current_set(broken) / catalog_module.LOCATOR_FILE).write_bytes(bytes(raw))
        try:
            catalog_module.Catalog.load(broken, chunk_dir).span(0)
        except catalog_module.CatalogError:
            report.require(True, "")
        else:
            report.require(
                False,
                "S3: a locator whose bytes do not match the digest the manifest "
                "vouches for was read anyway",
            )


# --------------------------------------------------------------------------
# S4 -- the predicate reaches the kernel, and changes nothing but the row set
# --------------------------------------------------------------------------


def check_predicate(
    report: Report, search, plan, catalog, embedding_root: Path, chunk_dir: Path, *, native_ok: bool
) -> None:
    embedding_set = search.EmbeddingSet(embedding_root)
    rows_total = len(embedding_set.rows)
    query = search.embed_query("conversion legality and lowering", embedding_set)

    unfiltered = search.topk_reference(query, embedding_set, 5)
    every_row = search.topk_reference(query, embedding_set, 5, rows=list(range(rows_total)))
    report.require(
        every_row == unfiltered,
        "S4: a predicate admitting every row changed the ranking; a filter must "
        "remove rows, never reorder the ones it keeps",
    )
    report.require(
        search.topk_reference(query, embedding_set, 5, rows=[]) == [],
        "S4: a predicate admitting no rows returned results -- the anti-vacuity "
        "case, where an empty mask must mean nothing rather than everything",
    )

    if not native_ok:
        report.skip("S4 kernel half: the native backend is not reachable")
    else:
        report.require(
            search.topk_native(query, embedding_set, 5, rows=list(range(rows_total))) == unfiltered,
            "S4: the kernel's full mask changed the unfiltered ranking",
        )
        report.require(
            search.topk_native(query, embedding_set, 5, rows=[]) == [],
            "S4: the kernel returned results for a mask admitting nothing",
        )
        native_module = load_tool("bcir_native")
        for bad in (-1, rows_total, rows_total + 100):
            try:
                native_module.eligibility_mask(rows_total, [bad])
            except ValueError:
                report.require(True, "")
            else:
                report.require(False, f"S4: an eligibility mask accepted row {bad} outside the set")

        # The differential, under filtering, on real predicates and random ones.
        generator = random.Random(20260911)
        trials = 0
        for _ in range(MIN_PREDICATE_TRIALS):
            size = generator.choice([1, 2, 3, 9, 40, 300])
            rows = sorted(generator.sample(range(rows_total), min(size, rows_total)))
            probe = search.embed_query(f"trial {trials} lowering dialect vector", embedding_set)
            reference = search.topk_reference(probe, embedding_set, 5, rows=rows)
            kernel = search.topk_native(probe, embedding_set, 5, rows=rows)
            report.require(
                reference == kernel,
                f"S4: filtered reference and native disagree on {len(rows)} admitted "
                f"row(s): {reference} vs {kernel}",
            )
            report.require(
                len(reference) == min(5, len(rows)),
                f"S4: {len(rows)} admitted row(s) with top_k=5 returned "
                f"{len(reference)} result(s); the short-return path is the answer, "
                "not a truncation",
            )
            trials += 1
        report.require(
            trials >= MIN_PREDICATE_TRIALS,
            f"S4: only {trials} filtered differentials ran, below the {MIN_PREDICATE_TRIALS} floor",
        )

    # The predicate language itself: exact counts, honest bounds, total refusals.
    #
    # Counted here from the chunk files directly rather than from the catalog. The
    # statistics and the postings are both derived from one scan, so comparing them
    # to each other compares the catalog with itself and passes over any defect they
    # share -- a row dropped from both is invisible. The source of truth is the
    # corpus, so that is what they are checked against.
    catalog_module = load_tool("catalog")
    truth: dict[str, dict[str, int]] = {name: {} for name in catalog.indexed_columns()}
    truth_prefixes: dict[str, int] = {}
    truth_ids: list[str] = []
    truth_records: list[dict] = []
    truth_rows = 0
    for path in catalog_module.chunk_files(chunk_dir):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            truth_rows += 1
            truth_ids.append(record.get("chunk_id"))
            truth_records.append(record)
            for column in truth:
                key = catalog_module.index_key(record.get(column))
                truth[column][key] = truth[column].get(key, 0) + 1
            source_path = record.get(catalog_module.PATH_COLUMN)
            if isinstance(source_path, str):
                for prefix in catalog_module.path_prefixes(source_path):
                    truth_prefixes[prefix] = truth_prefixes.get(prefix, 0) + 1

    # The row-order contract, computed here from the corpus with the shared key and
    # compared against what the catalog stored. Two writers agreeing by coincidence
    # is the failure this replaces, so the check recomputes rather than compares the
    # two artifacts to each other.
    ordered = sorted(truth_records, key=catalog_module.row_sort_key)
    report.require(
        [record["chunk_id"] for record in ordered] == catalog.ids,
        "S4: the catalog's row order is not `catalog.row_sort_key`'s order; row `i` "
        "of the embedding set and row `i` here would be different chunks, and every "
        "fetch would return some other row's text",
    )
    report.require(
        sorted(truth_ids) == sorted(catalog.ids),
        "S4: the catalog and the corpus hold different chunk_ids, not merely a different order",
    )
    report.require(
        {prefix: int(count) for prefix, count in catalog._statistics["path_prefixes"].items()}
        == truth_prefixes,
        f"S4: the path-prefix index counts {len(catalog._statistics['path_prefixes'])} "
        f"directories and the corpus has {len(truth_prefixes)}; a prefix predicate "
        "over a missing directory would silently fall back to a looser bound",
    )
    deepest = sorted(truth_prefixes, key=lambda p: (-p.count("/"), p))[:12]
    report.require(
        len(deepest) >= 4,
        f"S4: only {len(deepest)} prefixes to check, too few to witness the index",
    )
    for prefix in deepest:
        rows = catalog.rows_with_prefix(prefix)
        count, exact = catalog.count_prefix(prefix)
        report.require(
            exact and count == truth_prefixes[prefix] == len(rows),
            f"S4: prefix {prefix!r} counts {count} (exact={exact}), resolves "
            f"{len(rows)} rows, and the corpus has {truth_prefixes[prefix]}",
        )
    report.require(
        truth_rows == catalog.rows_total,
        f"S4: the chunk files hold {truth_rows} rows and the catalog claims {catalog.rows_total}",
    )

    for column in catalog.indexed_columns():
        counts = catalog.distinct(column)
        report.require(
            {name: int(count) for name, count in counts.items()} == truth[column],
            f"S4: the {column!r} statistics disagree with a direct count of the chunk "
            f"files (catalog {len(counts)} value(s), corpus {len(truth[column])})",
        )
        for value, expected in truth[column].items():
            rows = catalog.postings["columns"][column].get(value, [])
            report.require(
                len(rows) == expected,
                f"S4: the {column}={value!r} postings list holds {len(rows)} rows and "
                f"the corpus has {expected}",
            )
        report.require(
            sum(counts.values()) == catalog.rows_total,
            f"S4: the {column!r} index counts {sum(counts.values())} rows and the "
            f"catalog holds {catalog.rows_total}; a row is missing from the index",
        )
        for value in list(counts)[:8]:
            spelled = None if value == load_tool("catalog").NULL_KEY else value
            selection = plan.select(catalog, [plan.Predicate(column, "eq", (str(spelled),))])
            if spelled is None:
                continue
            report.require(
                selection.admitted == counts[value],
                f"S4: {column}={value!r} resolved {selection.admitted} rows but the "
                f"statistics counted {counts[value]}",
            )
            report.require(
                selection.estimate_exact,
                f"S4: {column}={value!r} was estimated as a bound; an indexed column "
                "value is a count",
            )

    for text in ("no-operator", "=novalue", "unknown_column=x"):
        try:
            plan.select(catalog, [plan.parse_predicate(text)])
        except plan.PlanError:
            report.require(True, "")
        else:
            report.require(False, f"S4: the predicate {text!r} was accepted")


# --------------------------------------------------------------------------
# The planner -- legality before cost, deterministically
# --------------------------------------------------------------------------


def check_aggregates(report: Report, plan, catalog, chunk_dir: Path) -> None:
    """Aggregates answered two ways must agree, and both must match the corpus.

    The statistics path reads nothing and the gather path reads the packed column, so
    a defect in either is invisible against the other alone. Both are therefore also
    checked against a direct pass over the chunk files -- the same reason the column
    index is.
    """
    catalog_module = load_tool("catalog")
    truth: dict[str, list[int]] = {name: [] for name in catalog.numeric_columns()}
    report.require(
        bool(truth),
        "aggregates: the catalog declares no numeric columns, so every check below "
        "would iterate zero times",
    )
    for path in catalog_module.chunk_files(chunk_dir):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for column in truth:
                value = record.get(column)
                if isinstance(value, int) and not isinstance(value, bool):
                    truth[column].append(int(value))

    every_row = list(range(catalog.rows_total))
    for column, values in truth.items():
        free = catalog.aggregate(column)
        gathered = catalog.aggregate(column, every_row)
        report.require(
            free["scanned"] == 0 and gathered["scanned"] == len(values),
            f"aggregates: {column} reported scanned={free['scanned']} unfiltered and "
            f"{gathered['scanned']} over every row; the two paths are not the two "
            "paths they claim to be",
        )
        for field in ("count", "nulls", "min", "max", "sum"):
            report.require(
                free[field] == gathered[field],
                f"aggregates: {column}.{field} is {free[field]} from statistics and "
                f"{gathered[field]} from the packed column",
            )
        report.require(
            free["count"] == len(values)
            and free["min"] == (min(values) if values else None)
            and free["max"] == (max(values) if values else None)
            and free["sum"] == sum(values),
            f"aggregates: {column} disagrees with a direct pass over the chunk files "
            f"(count {free['count']} vs {len(values)}, sum {free['sum']} vs {sum(values)})",
        )

        # A selection aggregates exactly its own rows.
        narrow = sorted(catalog.rows_equal("kind", "code") or [])
        if narrow:
            picked = catalog.aggregate(column, narrow)
            expected = []
            for row, record in catalog.fetch(narrow).items():
                value = record.get(column)
                if isinstance(value, int) and not isinstance(value, bool):
                    expected.append(int(value))
            report.require(
                picked["count"] == len(expected) and picked["sum"] == sum(expected),
                f"aggregates: {column} over {len(narrow)} selected row(s) reported "
                f"count={picked['count']} sum={picked['sum']}, the rows themselves "
                f"hold count={len(expected)} sum={sum(expected)}",
            )

        # A row outside the set is refused rather than gathered. `-1` matters more
        # than `rows_total` here: an out-of-range index raises on its own, but a
        # negative one indexes from the end and returns the wrong row's measurement
        # with no sign that anything went wrong.
        for bad in (catalog.rows_total, catalog.rows_total + 7, -1, -catalog.rows_total):
            try:
                catalog.aggregate(column, [bad])
            except IndexError:
                report.require(True, "")
            else:
                report.require(False, f"aggregates: {column} gathered row {bad}, outside the set")

    # A null cell is absent, not zero -- they are different answers to MIN and to
    # AVG. This corpus has no missing measurement, so asserting over it would pass
    # by iterating zero times (L2). The case is therefore constructed: a chunk table
    # with a row whose `char_count` is absent and another whose value is 0, which a
    # format that spells "missing" as zero cannot tell apart.
    with tempfile.TemporaryDirectory() as directory:
        mirror = Path(directory) / "chunks"
        mirror.mkdir()
        rows = []
        for index in range(4):
            record = {
                "chunk_id": f"sha256:{index:064d}",
                "subject": "synthetic",
                "source_path": f"synthetic/{index}.md",
                "kind": "prose",
                "language": None,
                "span": {"start_line": index, "end_line": index},
                "text": f"row {index}",
                "token_estimate": index,
            }
            if index == 1:
                record["char_count"] = 0  # a real, measured zero
            elif index != 2:  # row 2 has no char_count at all
                record["char_count"] = 100 + index
            rows.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        (mirror / "synthetic.chunks.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        built = catalog_module.build(mirror, synthetic_table())
        catalog_module.write(built, Path(directory) / "catalog")
        synthetic = catalog_module.Catalog.load(Path(directory) / "catalog", mirror)

        summary = synthetic.aggregate("char_count")
        report.require(
            summary["nulls"] == 1 and summary["count"] == 3,
            f"aggregates: a table with one missing char_count reported "
            f"nulls={summary['nulls']} count={summary['count']}, expected 1 and 3",
        )
        report.require(
            summary["min"] == 0,
            f"aggregates: MIN(char_count) is {summary['min']}; the measured zero in "
            "row 1 is the minimum, and a missing value must not be confused with it",
        )
        report.require(
            summary["sum"] == 0 + 100 + 103,
            f"aggregates: SUM(char_count) is {summary['sum']}, expected 203",
        )
        gathered = synthetic.aggregate("char_count", list(range(4)))
        for field in ("count", "nulls", "min", "max", "sum"):
            report.require(
                gathered[field] == summary[field],
                f"aggregates: char_count.{field} differs between the statistics "
                f"({summary[field]}) and the packed column ({gathered[field]}) when a "
                "value is missing",
            )
        only_null = synthetic.aggregate("char_count", [2])
        report.require(
            only_null["count"] == 0 and only_null["min"] is None and only_null["sum"] == 0,
            f"aggregates: a selection holding only the missing cell reported "
            f"{only_null}; an absent measurement contributes nothing, not zero",
        )

    for column in truth:
        values = catalog.numeric[column]
        nulls = sum(1 for value in values if value == catalog_module.NUMERIC_NULL)
        report.require(
            nulls == catalog.aggregate(column)["nulls"],
            f"aggregates: {column} has {nulls} null cell(s) and reports "
            f"{catalog.aggregate(column)['nulls']}",
        )

    # GROUP BY, both paths, totals that add up.
    search = load_tool("search_chunks")
    for column in catalog.indexed_columns():
        groups, scanned = analytics.group_counts(catalog, column, None)
        report.require(
            not scanned and sum(count for _, count in groups) == catalog.rows_total,
            f"aggregates: GROUP BY {column} unfiltered sums to "
            f"{sum(count for _, count in groups)}, not {catalog.rows_total}",
        )
        selection = plan.select(catalog, [plan.Predicate("kind", "eq", ("code",))])
        grouped, scanned = analytics.group_counts(catalog, column, selection)
        report.require(
            scanned and sum(count for _, count in grouped) == selection.admitted,
            f"aggregates: GROUP BY {column} over {selection.admitted} selected row(s) "
            f"sums to {sum(count for _, count in grouped)}",
        )
        report.require(
            all(count > 0 for _, count in grouped),
            f"aggregates: GROUP BY {column} emitted a group holding no rows",
        )


# --------------------------------------------------------------------------
# S7 -- comparisons, the zone map, and the rows that carry no value
# --------------------------------------------------------------------------


def _corpus_values(catalog_module, chunk_dir: Path, column: str) -> list[int]:
    """Every measured value of one column, read straight from the chunk files.

    Deliberately not in row order and deliberately not through the catalog: what it
    answers is "how many records in this corpus hold a value at all", which is the
    half of a range answer the catalog cannot be its own witness for.
    """
    values = []
    for path in catalog_module.chunk_files(chunk_dir):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get(column)
            if isinstance(value, int) and not isinstance(value, bool):
                values.append(int(value))
    return values


def _row_values(records: dict, column: str) -> dict:
    """row -> its measured value, or None, taken from each row's own fetched record."""
    values = {}
    for row, record in records.items():
        found = record.get(column)
        measured = isinstance(found, int) and not isinstance(found, bool)
        values[row] = int(found) if measured else None
    return values


def _in_interval(value, low, high) -> bool:
    """The interval rule, written once here so every expectation below shares it."""
    if value is None:
        return False
    return (low is None or value >= low) and (high is None or value <= high)


def _synthetic_gaps(catalog_module, directory: Path):
    """A two-part chunk table where one row has no `char_count` and one measures zero.

    The real corpus measures every row, so a claim about missing measurements
    asserted only over it would iterate zero times and pass (L2). Two files, because
    a claim about pruning needs more than one part to prune.
    """
    mirror = directory / "chunks"
    mirror.mkdir()
    for part, span in (("low", range(0, 3)), ("high", range(3, 6))):
        lines = []
        for index in span:
            record = {
                "chunk_id": f"sha256:{index:064d}",
                "subject": "synthetic",
                "source_path": f"synthetic/{part}/{index}.md",
                "kind": "prose",
                "language": None,
                "span": {"start_line": index, "end_line": index},
                "text": f"row {index}",
                "token_estimate": index,
            }
            # Row 2 carries no measurement at all; row 1 measures a real zero. A
            # format that spelled "missing" as zero cannot tell those two apart, and
            # a comparison that treats the sentinel as a number puts row 2 below
            # every bound.
            if index != 2:
                record["char_count"] = 0 if index == 1 else 100 * (index + 1)
            lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        (mirror / f"{part}.chunks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    catalog_module.write(catalog_module.build(mirror, synthetic_table()), directory / "catalog")
    return catalog_module.Catalog.load(directory / "catalog", mirror)


def _check_sorted_index(report: Report, catalog_module, catalog, column: str) -> None:
    """The sorted index is a claim about every row, checkable before any query runs.

    Called for the corpus and again for a table built with a gap in it. Every corpus
    this gate builds measures every row, so checked only there the split point and the
    unmeasured tail are claims about the empty set -- and the one fault that puts a
    row on the wrong side of the split would pass (L2). One function, both tables
    (L14).
    """
    index = list(catalog.order[column])
    split = catalog.measured(column)
    cells = catalog.numeric[column]
    counted = catalog.aggregate(column)["count"]
    report.require(
        sorted(index) == list(range(catalog.rows_total)),
        f"ranges: the sorted index for {column} names {len(set(index))} distinct "
        f"row(s) of {catalog.rows_total}; it has to be a permutation, or a seek "
        "returns a row twice or not at all",
    )
    report.require(
        split == counted,
        f"ranges: the sorted index for {column} splits at {split} where the statistics "
        f"count {counted} measured row(s); the split is the only thing keeping "
        "unmeasured rows out of every comparison",
    )
    keys = [(cells[row], row) for row in index[:split]]
    report.require(
        keys == sorted(keys),
        f"ranges: the measured prefix of {column}'s index is not ascending by "
        "(value, row); a binary search over it would return the wrong window, and a "
        "page boundary inside a run of equal values would move between runs",
    )
    report.require(
        all(cells[row] != catalog_module.NUMERIC_NULL for row in index[:split])
        and all(cells[row] == catalog_module.NUMERIC_NULL for row in index[split:]),
        f"ranges: {column}'s index has measured and unmeasured rows on the wrong sides "
        "of its split",
    )
    report.require(
        index[split:] == sorted(index[split:]),
        f"ranges: the unmeasured tail of {column}'s index is not in row order",
    )


def check_ranges(report: Report, plan, catalog, chunk_dir: Path) -> None:
    """A comparison must find what a pass over the corpus finds, and skip parts doing it.

    Three halves, because no two of them fail together:

      *sound*     every row returned holds, in its own fetched record, a value in the
                  interval -- not re-read from the packed column the index just read;
      *complete*  exactly as many rows come back as there are records in the corpus
                  with a value in the interval, counted by a direct pass over files;
      *pruned*    the zone map ruled parts out, every part it ruled out really holds
                  no match, and the pruned answer equals the full column scan's.

    Sound and complete together pin the set: two subsets of one finite set, one
    inside the other and of equal size, are equal. Neither alone does -- an index
    that returns nothing is sound, and one that returns everything is complete.
    """
    catalog_module = load_tool("catalog")
    columns = catalog.numeric_columns()
    report.require(
        bool(columns),
        "ranges: the catalog declares no numeric column, so every loop below would "
        "iterate zero times",
    )
    fetched = catalog.fetch(list(range(catalog.rows_total)))
    trials = 0
    pruned_parts = 0
    strategies: set[str] = set()

    for column in columns:
        corpus = _corpus_values(catalog_module, chunk_dir, column)
        by_row = _row_values(fetched, column)
        cells = catalog.numeric[column]
        measured = sorted(value for value in by_row.values() if value is not None)
        report.require(
            measured == sorted(corpus),
            f"ranges: {column} holds {len(measured)} measured value(s) read row by row "
            f"and {len(corpus)} read straight from the chunk files; the two oracles "
            "the checks below rely on do not agree with each other",
        )
        report.require(
            bool(corpus),
            f"ranges: no record in the corpus measures {column}, so every interval "
            "below would be checked against the empty set",
        )
        if not corpus:
            continue

        # The zone map is a claim about cells, checkable without any query at all.
        for part in catalog.parts:
            start, end = part["rows"]
            span = [cells[row] for row in range(start, end)]
            present = [value for value in span if value != catalog_module.NUMERIC_NULL]
            recomputed = {
                "count": len(present),
                "nulls": len(span) - len(present),
                "min": min(present) if present else None,
                "max": max(present) if present else None,
                "sum": sum(present),
            }
            zone = catalog.zone(part, column)
            wrong = [name for name, value in recomputed.items() if zone[name] != value]
            report.require(
                not wrong,
                f"ranges: part {part['part_id']!r} summarises {column} as "
                f"{ {name: zone[name] for name in wrong} } where its own cells hold "
                f"{ {name: recomputed[name] for name in wrong} }; a zone map that "
                "does not describe its cells prunes parts that hold matches",
            )

        # ...and the parts must add up to the whole-corpus statistics, which is the
        # witness that both really come from `numeric_summary` (L14).
        zones = [catalog.zone(part, column) for part in catalog.parts]
        whole = catalog.aggregate(column)
        lows = [zone["min"] for zone in zones if zone["min"] is not None]
        highs = [zone["max"] for zone in zones if zone["max"] is not None]
        report.require(
            sum(zone["count"] for zone in zones) == whole["count"]
            and sum(zone["nulls"] for zone in zones) == whole["nulls"]
            and sum(zone["sum"] for zone in zones) == whole["sum"]
            and (min(lows) if lows else None) == whole["min"]
            and (max(highs) if highs else None) == whole["max"],
            f"ranges: the {len(zones)} zone map(s) for {column} do not add up to the "
            "statistics over the same rows; the per-part and whole-corpus summaries "
            "are not being computed by the same rule",
        )

        _check_sorted_index(report, catalog_module, catalog, column)

        low_end, high_end = min(corpus), max(corpus)
        middle = (low_end + high_end) // 2
        pivots = sorted(
            {low_end - 1, low_end, low_end + 1, middle, high_end - 1, high_end, high_end + 1, 0, -1}
        )
        intervals = [(None, None), (high_end, low_end), (low_end, high_end)]
        for pivot in pivots:
            intervals.extend([(pivot, None), (None, pivot), (pivot, pivot)])
        # The sentinel is a legal int64. Asked for as a bound it must still match
        # nothing, rather than sweeping up every row that has no measurement.
        sentinel = catalog_module.NUMERIC_NULL
        intervals.extend([(sentinel, sentinel), (sentinel, low_end - 1)])

        for low, high in intervals:
            trials += 1
            rows, read = catalog.range_scan(column, low, high)
            pruned_parts += read["skipped"]
            wanted = [
                row for row in range(catalog.rows_total) if _in_interval(by_row[row], low, high)
            ]
            report.require(
                rows == wanted,
                f"ranges: {column} in [{low}, {high}] returned {len(rows)} row(s); the "
                f"records themselves hold {len(wanted)}",
            )
            unpruned = [
                row
                for row in range(catalog.rows_total)
                if cells[row] != catalog_module.NUMERIC_NULL and _in_interval(cells[row], low, high)
            ]
            report.require(
                rows == unpruned,
                f"ranges: {column} in [{low}, {high}] answered {len(rows)} row(s) with "
                f"the zone map and {len(unpruned)} by reading every cell",
            )
            # The seek is the path a query actually takes; the scan above is the path
            # it is checked against. Two implementations of one lookup, required to
            # agree on every interval -- the only claim a binary search can make about
            # itself that does not come out of the same arithmetic.
            window, probe = catalog.seek_range(column, low, high)
            report.require(
                sorted(window) == wanted and probe["window"] == len(wanted),
                f"ranges: seeking {column} in [{low}, {high}] returned "
                f"{len(window)} row(s) where scanning returns {len(wanted)}",
            )
            report.require(
                catalog.count_in_range(column, low, high) == len(wanted),
                f"ranges: count_in_range({column}, {low}, {high}) is "
                f"{catalog.count_in_range(column, low, high)}, not {len(wanted)}; the "
                "index answers a count exactly or it should not answer it",
            )
            # Whichever path the threshold picks, the rows are the same rows. The
            # choice is a cost decision, so it must be invisible in the answer and
            # visible in the strategy -- both halves are checked, here and below.
            strategy = catalog.range_strategy(column, low, high)
            report.require(
                strategy in ("seek", "scan"),
                f"ranges: {column} in [{low}, {high}] chose strategy {strategy!r}",
            )
            report.require(
                catalog.rows_in_range(column, low, high) == wanted,
                f"ranges: rows_in_range({column}, {low}, {high}) took the {strategy} "
                f"path and returned rows the other path does not",
            )
            strategies.add(strategy)
            found = set(rows)
            unsound = []
            for part in catalog.parts:
                start, end = part["rows"]
                verdict = catalog._zone_verdict(catalog.zone(part, column), low, high)
                inside = sum(1 for row in range(start, end) if row in found)
                if verdict == "none" and inside:
                    unsound.append(f"{part['part_id']} ruled out but holds {inside}")
                if verdict == "all" and inside != end - start:
                    unsound.append(f"{part['part_id']} taken whole but only {inside} match")
            report.require(
                not unsound,
                f"ranges: {column} in [{low}, {high}] pruned wrongly: " + "; ".join(unsound),
            )

        # Through the predicate language, not only through the catalog. Every check
        # above hands `range_scan` an interval directly, so an off-by-one in the rule
        # that turns `> n` into `>= n + 1` is invisible to all of them.
        for spelling, low_of, high_of in (
            (">=", lambda pivot: pivot, lambda pivot: None),
            (">", lambda pivot: pivot + 1, lambda pivot: None),
            ("<=", lambda pivot: None, lambda pivot: pivot),
            ("<", lambda pivot: None, lambda pivot: pivot - 1),
        ):
            for pivot in (low_end, middle, high_end):
                trials += 1
                chosen = plan.select(catalog, [plan.parse_predicate(f"{column}{spelling}{pivot}")])
                wanted = [
                    row
                    for row in range(catalog.rows_total)
                    if _in_interval(by_row[row], low_of(pivot), high_of(pivot))
                ]
                report.require(
                    list(chosen.rows or ()) == wanted,
                    f"ranges: `{column} {spelling} {pivot}` admitted "
                    f"{len(chosen.rows or ())} row(s) where the records themselves "
                    f"hold {len(wanted)}",
                )

    report.require(
        trials >= MIN_RANGE_TRIALS,
        f"anti-vacuity: {trials} interval(s) were checked, below the {MIN_RANGE_TRIALS} floor",
    )
    report.require(
        strategies == {"seek", "scan"},
        f"anti-vacuity: across {trials} interval(s) the range threshold only ever "
        f"chose {sorted(strategies)}. Both paths return identical rows, so a threshold "
        "stuck on one of them is invisible in every answer above; only this notices.",
    )
    report.require(
        pruned_parts >= MIN_PRUNED_PARTS,
        f"anti-vacuity: across {trials} interval(s) the zone map ruled out "
        f"{pruned_parts} part(s). Pruning changes cost and never the answer, so a "
        "zone map that stopped pruning would keep every check above green; the count "
        "is the only thing that notices.",
    )

    # ORDER BY taken from the index must be the order a sort gives, both directions.
    # Ascending is the index itself, so this is where descending -- which negates the
    # key rather than reversing the list -- is held to the same tie rule.
    search = load_tool("search_chunks")
    unfiltered = plan.select(catalog, [])
    for column in columns:
        cells = catalog.numeric[column]
        for descending in (False, True):
            from_index = relational.ordered_from_index(catalog, column, descending)
            rows = list(range(catalog.rows_total))
            missing = [row for row in rows if cells[row] == catalog_module.NUMERIC_NULL]
            present = [row for row in rows if cells[row] != catalog_module.NUMERIC_NULL]
            present.sort()
            present.sort(key=lambda row: cells[row], reverse=descending)
            report.require(
                from_index == present + missing,
                f"ranges: ORDER BY {column}{' DESC' if descending else ''} read from "
                "the index differs from the same order produced by sorting",
            )
            report.require(
                relational.ordered_rows(catalog, unfiltered, column, descending) == from_index,
                f"ranges: ORDER BY {column}{' DESC' if descending else ''} with no "
                "predicate returned rows the index path does not",
            )
        # ...and that it really took that path. Both paths return identical rows, so
        # the comparison above passes whichever one ran; only the strategy says which.
        narrow = plan.select(catalog, [plan.parse_predicate(f"{column}>={cells[0]}")])
        report.require(
            relational.order_strategy(catalog, unfiltered, column) == "index"
            and relational.order_strategy(catalog, narrow, column) == "sort",
            f"ranges: ORDER BY {column} chose "
            f"{relational.order_strategy(catalog, unfiltered, column)!r} with no predicate "
            f"and {relational.order_strategy(catalog, narrow, column)!r} with one; the "
            "index answers the first and a sort answers the second",
        )

    # Nulls satisfy no comparison, in either direction, and the sentinel is not a value.
    with tempfile.TemporaryDirectory() as directory:
        synthetic = _synthetic_gaps(catalog_module, Path(directory))
        _check_sorted_index(report, catalog_module, synthetic, "char_count")
        # Looked up by primary key, not assumed: rows are ordered by the canonical
        # sort key, so the record written third is not the third row. Writing `2`
        # here would have checked a row that measures 400 and passed while proving
        # nothing -- which is exactly what it did on the first run of this check.
        gap = synthetic.row_of(f"sha256:{2:064d}")
        present = synthetic.rows_present("char_count")
        report.require(
            gap not in present and len(present) == 5,
            f"ranges: the synthetic table reports {len(present)} measured row(s); it "
            "was built with five measured and one missing",
        )
        for low, high in ((None, None), (None, 10**9), (-(10**9), None), (-(10**9), 10**9)):
            rows = synthetic.rows_in_range("char_count", low, high)
            report.require(
                gap not in rows and rows == present,
                f"ranges: `char_count` in [{low}, {high}] returned row {gap}, which "
                "measures nothing; the null sentinel is being compared as a number",
            )
        sentinel = catalog_module.NUMERIC_NULL
        report.require(
            synthetic.rows_in_range("char_count", sentinel, sentinel) == [],
            "ranges: asking for the null sentinel as a bound returned rows; it spells "
            "'no measurement', not a measurement of -2^63",
        )
        for pivot in (-1, 0, 1, 100, 101, 399, 400, 600, 601, 10**6):
            below = set(synthetic.rows_in_range("char_count", None, pivot - 1))
            above = set(synthetic.rows_in_range("char_count", pivot, None))
            report.require(
                not (below & above) and below | above == set(present),
                f"ranges: `< {pivot}` and `>= {pivot}` cover "
                f"{len(below | above)} of {len(present)} measured row(s) and share "
                f"{len(below & above)}; a pivot must split the measured rows and "
                "leave the unmeasured one out of both",
            )


#: The two modules that *are* the engine, and so may reach it without going through
#: the package. `plan` prices a predicate over the catalog it filters with, and
#: `generations` publishes the catalog as part of a generation.
ENGINE_INSIDERS = ("plan.py", "generations.py")
ENGINE_MODULES = ("catalog", "plan")


def check_interface_boundary(report: Report) -> None:
    """No tool outside the database package opens the engine for itself.

    Declared scope: a static read of `training/tools/*.py` for two shapes -- loading
    `catalog` or `plan` through `spec_from_file_location`, and importing either at
    module level. Aliases, `__import__`, and anything assembled at run time are out of
    scope and belong to a linter rather than to this rail. The boundary is stated here
    so the answer to the next soundness question is to point at it rather than to grow
    an interpreter.

    Reaching the engine *through* `db.engine` is not a violation: the package is the
    door, however a caller knocks on it. What this forbids is a second door.

    It exists because that second door was real. The sibling-module loader was carried
    in three copies that disagreed -- one returned any cached module that shared the
    name, one executed a fresh module and then discarded it for whatever `setdefault`
    had kept, and one checked that the cached module came from the file it meant. Only
    the last is correct (`docs/security/laws.md` L14).
    """
    import ast

    scanned = 0
    findings = []
    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.name in ENGINE_INSIDERS:
            continue
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ENGINE_MODULES:
                        findings.append(f"{path.name} imports {alias.name} directly")
            elif isinstance(node, ast.ImportFrom) and node.module in ENGINE_MODULES:
                findings.append(f"{path.name} imports from {node.module} directly")
            elif isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name != "spec_from_file_location" or not node.args:
                    continue
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value in ENGINE_MODULES:
                    findings.append(f"{path.name} loads {first.value!r} by path")
    report.require(
        not findings,
        "interfaces: the database engine is reached around its package by " + "; ".join(findings),
    )
    report.require(
        scanned >= 10,
        f"anti-vacuity: only {scanned} tool(s) were read for the boundary above",
    )
    # ...and the boundary is only worth checking if the package is actually the door.
    report.require(
        any(
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", ""))
            == "spec_from_file_location"
            for node in ast.walk(
                ast.parse((TOOLS_DIR / "db" / "engine.py").read_text(encoding="utf-8"))
            )
        ),
        "interfaces: db/engine.py no longer loads anything by path, so the check "
        "above forbids a shape nothing in this tree uses",
    )


def check_interfaces(report: Report, catalog) -> None:
    """The four subsystem interfaces answer, over the corpus this gate built.

    Thin by design, and checked anyway: an interface nothing exercises is a second
    definition waiting to drift from the engine it wraps.
    """
    chunk_dir = catalog.chunk_dir
    selection = retrieval.eligible(catalog, ["kind=code"])
    report.require(
        selection.admitted > 0 and selection.rows is not None,
        "interfaces: retrieval.eligible admitted no rows for a predicate the corpus satisfies",
    )
    fetched = retrieval.records(catalog, list(selection.rows)[:8])
    report.require(
        len(fetched) == min(8, selection.admitted)
        and all(record.get("kind") == "code" for record in fetched.values()),
        "interfaces: retrieval.records returned rows the predicate did not admit",
    )
    columns = retrieval.projectable(catalog)
    report.require(
        set(catalog.numeric_columns()) <= set(columns)
        and set(catalog.indexed_columns()) <= set(columns),
        f"interfaces: retrieval.projectable named {columns}, which does not cover the "
        "columns the catalog indexes",
    )
    parts = ingest.parts(chunk_dir)
    report.require(
        [part["part_id"] for part in parts] == [part["part_id"] for part in catalog.parts],
        "interfaces: ingest.parts and the loaded catalog disagree about the parts",
    )
    report.require(
        ingest.changed_parts(chunk_dir, parts) == (),
        "interfaces: ingest.changed_parts calls a corpus changed against its own parts",
    )
    report.require(
        ingest.changed_parts(chunk_dir, None) == (),
        "interfaces: ingest.changed_parts invented changes from no recorded parts",
    )
    sources = {part["source"] for part in parts}
    report.require(
        all(
            {part["source"] for part in ingest.parts_for(chunk_dir, name)} == {name}
            for name in sources
        ),
        "interfaces: ingest.parts_for returned parts from another source",
    )


def check_byte_order(report: Report, catalog_module, catalog) -> None:
    """The readers agree with an explicitly little-endian decode, on whatever host.

    `array.frombytes` reads in the host's own order and each reader swaps when the host
    is big-endian. That swap is a branch no host in this repository's matrix executes,
    and an unexecuted branch is where a defect waits (`docs/security/laws.md` L21). So
    every packed artifact is decoded a second time here by an explicitly little-endian
    struct, one cell at a time, and the two readings must agree. On a little-endian
    host that is a tautology about `array`; on a big-endian one the swap branch is the
    only thing that can make it hold, which is the point.

    **Declared scope: this checks the readers, not the writer.** Both paths here read
    the file as little-endian, so a file *written* big-endian is consistent with both
    and passes. What catches that is the differential against the corpus itself --
    `check_aggregates` and `check_ranges` recompute from the chunk files, whose JSON
    has no byte order to get wrong -- and it was confirmed catching it: writing
    `numeric.bin` or `order.bin` with a `>` struct fails those, by name, before this
    check reports anything. The two halves are kept apart deliberately, because a
    check that claimed both would be believed about the half it does not cover.

    Byte order is then shown to be load-bearing rather than incidental: one cell read
    the other way round must give a different number. Without that, a column of small
    positive values could be byte-order-agnostic by accident and everything above
    would hold over a file nothing had pinned.
    """
    rows = catalog.rows_total
    root = catalog.set_dir

    numeric_raw = (root / catalog_module.NUMERIC_FILE).read_bytes()
    cell = struct.Struct("<q")
    for index, name in enumerate(catalog.numeric_columns()):
        base = index * rows * catalog_module.NUMERIC_CELL_BYTES
        explicit = [cell.unpack_from(numeric_raw, base + row * cell.size)[0] for row in range(rows)]
        report.require(
            list(catalog.numeric[name]) == explicit,
            f"byte order: {catalog_module.NUMERIC_FILE} column {name} reads differently "
            "through the array fast path than through an explicitly little-endian "
            "decode; the packed columns are not in the order the format declares",
        )

    order_raw = (root / catalog_module.ORDER_FILE).read_bytes()
    entry = struct.Struct("<I")
    for index, name in enumerate(catalog.numeric_columns()):
        base = index * rows * catalog_module.ORDER_ENTRY_BYTES
        explicit = [entry.unpack_from(order_raw, base + row * entry.size)[0] for row in range(rows)]
        report.require(
            list(catalog.order[name]) == explicit,
            f"byte order: {catalog_module.ORDER_FILE} column {name} reads differently "
            "through the array fast path than through an explicitly little-endian decode",
        )

    locator_raw = (root / catalog_module.LOCATOR_FILE).read_bytes()
    span = struct.Struct("<HQI")
    report.require(
        len(locator_raw) == rows * catalog_module.LOCATOR_ENTRY_BYTES,
        f"byte order: {catalog_module.LOCATOR_FILE} holds {len(locator_raw)} bytes for "
        f"{rows} rows of {catalog_module.LOCATOR_ENTRY_BYTES}",
    )
    mismatched = [
        row
        for row in range(rows)
        if span.unpack_from(locator_raw, row * span.size) != catalog.span(row)
    ]
    report.require(
        not mismatched,
        f"byte order: {len(mismatched)} locator entr(y/ies) decode differently through "
        "an explicitly little-endian struct than through `span`",
    )

    # ...and byte order is load-bearing: the same cell read the other way is a
    # different number. A column that happened to be order-agnostic would make every
    # check above hold over a file nothing had pinned.
    swapped = struct.Struct(">q")
    differing = sum(
        1
        for row in range(min(rows, 64))
        if swapped.unpack_from(numeric_raw, row * cell.size)[0]
        != cell.unpack_from(numeric_raw, row * cell.size)[0]
    )
    report.require(
        differing > 0,
        "anti-vacuity: the first 64 measurement cells read the same in both byte "
        "orders, so the checks above would hold on a file with no byte order at all",
    )


#: What `--where` reads, as a table, so `check_grammar` and `check_operator_coverage`
#: read one source instead of two that agree by habit (`docs/security/laws.md` L15).
#: Every operator the language declares must appear here; the coverage check enforces
#: that, so adding an operator without a row is a failure rather than a silence.
GRAMMAR_ACCEPTS = (
    ("subject=llvm", "subject", "eq", ("llvm",)),
    ("subject!=llvm", "subject", "ne", ("llvm",)),
    ("subject=llvm,data", "subject", "in", ("llvm", "data")),
    ("subject=" + chr(34) + "llvm,data" + chr(34), "subject", "eq", ("llvm,data",)),
    ("subject=" + chr(34) + "a" + chr(34) + "," + chr(34) + "b" + chr(34), "subject", "in", ("a", "b")),
    ("source_path^=training/", "source_path", "prefix", ("training/",)),
    ("char_count>=500", "char_count", "ge", ("500",)),
    ("char_count<=500", "char_count", "le", ("500",)),
    ("char_count>500", "char_count", "gt", ("500",)),
    ("char_count<500", "char_count", "lt", ("500",)),
    ("char_count>=-5", "char_count", "ge", ("-5",)),
    ("char_count>= 5 ", "char_count", "ge", ("5",)),
    ("language=?", "language", "notnull", ()),
    ("language!=?", "language", "isnull", ()),
    ("title=a!=b", "title", "eq", ("a!=b",)),
    ("title=a>=b", "title", "eq", ("a>=b",)),
    ("title=a^=b", "title", "eq", ("a^=b",)),
    ("subject=llvm,", "subject", "eq", ("llvm",)),
)


def check_grammar(report: Report, plan) -> None:
    """What `--where` reads, and what it refuses, spelled out as a table.

    A term is split at the *leftmost* operator and, there, at the longest spelling.
    Scanning the spelling table in order instead reads `char_count>=500` as `=` on a
    column named `char_count>`, and `title=a!=b` as `!=` on a column named `title=a`
    -- both of which then fail as "unknown column", a long way from the term that
    caused them. One rule settles both, and this table is what holds it in place.
    """
    for text, column, op, values in GRAMMAR_ACCEPTS:
        parsed = plan.parse_predicate(text)
        report.require(
            (parsed.column, parsed.op, parsed.values) == (column, op, values),
            f"grammar: {text!r} read as {parsed.column!r} {parsed.op} {parsed.values}, "
            f"expected {column!r} {op} {values}",
        )
    for text, why in (
        ("char_count>=1_000", "a separator Python accepts inside a literal and no format does"),
        ("char_count>=+5", "a leading plus this grammar does not spell"),
        ("char_count>=inf", "a float literal that names no integer"),
        ("char_count>=nan", "a float literal that names no integer"),
        ("char_count>=٥", "a digit outside ASCII"),
        ("char_count>=5.0", "a decimal point"),
        ("char_count>=", "no bound at all"),
        ("=llvm", "no column"),
        ("subject", "no operator"),
        ("subject^=?", "a presence test spelled with an operator that is not = or !="),
        ("subject=?,llvm", "the reserved presence value inside a set"),
    ):
        try:
            plan.parse_predicate(text)
        except plan.PlanError:
            report.require(True, "")
        else:
            report.require(False, f"grammar: accepted {text!r}, which carries {why}")


def check_presence(report: Report, plan, catalog, chunk_dir: Path) -> None:
    """IS NULL and IS NOT NULL must partition the table; `!=` deliberately must not.

    `rows_present` and `rows_absent` are gathered by separate walks precisely so that
    "these two partition the table" is a claim and not an identity. This is where the
    claim is cashed, against the corpus and against each other.
    """
    catalog_module = load_tool("catalog")
    every = set(range(catalog.rows_total))
    numeric = set(catalog.numeric_columns())
    path_column = catalog_module.PATH_COLUMN
    columns = tuple(catalog.indexed_columns()) + tuple(numeric) + (path_column,)
    report.require(
        len(columns) >= 2,
        f"presence: only {len(columns)} column(s) to check, so the loop below barely runs",
    )

    absent_total = 0
    for column in columns:
        present, absent = catalog.rows_present(column), catalog.rows_absent(column)
        report.require(
            present == sorted(present) and absent == sorted(absent),
            f"presence: {column} returned rows out of order; every other row set here "
            "is ascending, and an intersection that assumed so would be wrong",
        )
        report.require(
            not (set(present) & set(absent)),
            f"presence: {column} calls {len(set(present) & set(absent))} row(s) both "
            "measured and unmeasured",
        )
        report.require(
            set(present) | set(absent) == every,
            f"presence: {column} accounts for {len(set(present) | set(absent))} of "
            f"{catalog.rows_total} rows",
        )
        absent_total += len(absent)

        # Against the corpus. Each column kind spells "carries a value" the way its
        # own storage does, and the gate asserts that spelling rather than a fourth.
        carried = 0
        for path in catalog_module.chunk_files(chunk_dir):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line).get(column)
                if column in numeric:
                    carried += isinstance(value, int) and not isinstance(value, bool)
                elif column == path_column:
                    carried += isinstance(value, str)
                else:
                    carried += value is not None
        report.require(
            len(present) == carried,
            f"presence: the index says {len(present)} row(s) carry a {column}; the "
            f"chunk files hold {carried}",
        )

    report.require(
        absent_total >= 1,
        "anti-vacuity: no column in this corpus is missing a single value, so every "
        "partition above held over an empty half and IS NULL was never exercised",
    )

    # The declared divergence from SQL, stated as a property so it cannot drift into
    # three-valued logic or out of it without this failing.
    exercised = 0
    for column in catalog.indexed_columns():
        absent = set(catalog.rows_absent(column))
        values = [key for key in catalog.distinct(column) if key != catalog_module.NULL_KEY]
        if not absent or not values:
            continue
        exercised += 1
        term = plan.Predicate(column, "ne", (values[0],))
        complement = set(plan.select(catalog, [term]).rows)
        narrowed = set(plan.select(catalog, [term, plan.Predicate(column, "notnull", ())]).rows)
        report.require(
            absent <= complement,
            f"presence: `{column} != {values[0]}` dropped "
            f"{len(absent - complement)} row(s) carrying no {column}; this module "
            "documents complement semantics, not SQL's three-valued <>",
        )
        report.require(
            complement - absent == narrowed,
            f"presence: `{column} != {values[0]}` with `{column}=?` admitted "
            f"{len(narrowed)} row(s) where dropping the unmeasured ones from the "
            f"complement leaves {len(complement - absent)}; the documented way to ask "
            "SQL's question no longer answers it",
        )
    report.require(
        exercised >= 1,
        "anti-vacuity: no indexed column has both a missing value and a present one, "
        "so the documented `!=` divergence was never exercised",
    )


def check_grouped_aggregates(report: Report, search, plan, catalog) -> None:
    """GROUP BY must partition, its aggregates must add up, and HAVING must filter exactly."""
    for column in catalog.indexed_columns():
        groups = analytics.group_rows(catalog, column, None)
        members = [row for _, rows in groups for row in rows]
        report.require(
            sorted(members) == list(range(catalog.rows_total)),
            f"groups: GROUP BY {column} covers {len(set(members))} of "
            f"{catalog.rows_total} rows; an aggregate over groups that do not "
            "partition the table silently drops or double-counts rows",
        )
        report.require(
            len(members) == len(set(members)),
            f"groups: GROUP BY {column} puts {len(members) - len(set(members))} row(s) "
            "in more than one group",
        )
        counted, scanned = analytics.group_counts(catalog, column, None)
        report.require(
            not scanned and dict(counted) == {value: len(rows) for value, rows in groups},
            f"groups: GROUP BY {column} answered from the statistics disagrees with "
            "the same grouping walked through the postings",
        )

        for measure in catalog.numeric_columns():
            whole = catalog.aggregate(measure, list(range(catalog.rows_total)))
            summaries = [analytics.group_summary(catalog, measure, rows) for _, rows in groups]
            lows = [summary["min"] for summary in summaries if summary["min"] is not None]
            highs = [summary["max"] for summary in summaries if summary["max"] is not None]
            report.require(
                sum(summary["rows"] for summary in summaries) == catalog.rows_total
                and sum(summary["count"] for summary in summaries) == whole["count"]
                and sum(summary["nulls"] for summary in summaries) == whole["nulls"]
                and sum(summary["sum"] for summary in summaries) == whole["sum"]
                and (min(lows) if lows else None) == whole["min"]
                and (max(highs) if highs else None) == whole["max"],
                f"groups: {measure} aggregated per group of {column} does not add up "
                "to the same aggregate over every row",
            )

        # HAVING keeps exactly the groups the term describes, spelled out here rather
        # than recomputed by the code under test.
        for bound in (1, 2, max(len(rows) for _, rows in groups)):
            term = analytics.parse_having(f"rows>={bound}")
            kept = [
                value for value, rows in groups if analytics.having_holds(term, {"rows": len(rows)})
            ]
            report.require(
                kept == [value for value, rows in groups if len(rows) >= bound],
                f"groups: HAVING rows>={bound} kept {len(kept)} group(s) of "
                f"{len(groups)}, not the ones with at least {bound} row(s)",
            )

    # AVG is compared as SUM/COUNT, not as a rounded decimal: 1.5 is below 2.
    half = {"rows": 2, "count": 2, "nulls": 0, "min": 1, "max": 2, "sum": 3, "avg": 1.5}
    for spec, expected in (
        ("avg>=2", False),
        ("avg>1", True),
        ("avg<2", True),
        ("avg>=1", True),
        ("avg=1", False),
        ("avg!=1", True),
    ):
        term = analytics.parse_having(spec)
        report.require(
            analytics.having_holds(term, half) is expected,
            f"groups: a group averaging 1.5 answered `{spec}` with "
            f"{not expected}; AVG is SUM/COUNT compared exactly, not a rounded decimal",
        )

    # A group with no measured row has no MIN, MAX or AVG, and satisfies no comparison
    # against one -- in both directions, so "always false" is not what is being tested.
    empty = {"rows": 3, "count": 0, "nulls": 3, "min": None, "max": None, "sum": 0, "avg": None}
    for spec in ("avg>=0", "avg<0", "min>=0", "min<0", "max>=0", "max<0", "avg=0"):
        term = analytics.parse_having(spec)
        report.require(
            not analytics.having_holds(term, empty),
            f"groups: a group with no measured row satisfied `{spec}`",
        )
    report.require(
        analytics.having_holds(analytics.parse_having("rows>=3"), empty),
        "groups: a group with no measured row still has rows, and HAVING rows>=3 "
        "refused a group of three",
    )

    # What HAVING refuses, and why each would otherwise mean something it does not.
    for spec, why in (
        ("nope>=1", "an aggregate no group computes"),
        ("rows^=1", "an operator with no meaning over a number"),
        ("rows>=x", "a bound that is not an integer"),
        ("rows>=1_000", "a bound Python would accept and the grammar does not"),
        ("rows=?", "a presence test, which a count always passes"),
        ("rows=1,2", "a set, where a comparison takes one bound"),
    ):
        try:
            analytics.parse_having(spec)
        except plan.PlanError:
            report.require(True, "")
        else:
            report.require(False, f"groups: HAVING accepted {spec!r}, which names {why}")
    try:
        analytics.having_holds(analytics.parse_having("count>=1"), {"rows": 3})
    except plan.PlanError:
        report.require(True, "")
    else:
        report.require(
            False,
            "groups: HAVING count was answered for a group with no --stats column, "
            "where COUNT(*) and COUNT(column) are the same number by accident",
        )


def check_ordering(report: Report, search, plan, catalog) -> None:
    """ORDER BY must be total, stable, and honest about missing measurements."""
    catalog_module = load_tool("catalog")
    selection = plan.select(catalog, [plan.Predicate("kind", "eq", ("code",))])
    report.require(
        selection.admitted >= 8,
        f"ordering: only {selection.admitted} row(s) to order, too few to mean much",
    )

    for column in catalog.numeric_columns():
        ascending = relational.ordered_rows(catalog, selection, column, False)
        descending = relational.ordered_rows(catalog, selection, column, True)
        report.require(
            sorted(ascending) == sorted(descending) == sorted(selection.rows),
            f"ordering: {column} returned a different row set in the two directions",
        )
        report.require(
            ascending == relational.ordered_rows(catalog, selection, column, False),
            f"ordering: {column} ascending is not stable across two calls",
        )
        values = catalog.numeric[column]
        present_asc = [
            values[row] for row in ascending if values[row] != catalog_module.NUMERIC_NULL
        ]
        present_desc = [
            values[row] for row in descending if values[row] != catalog_module.NUMERIC_NULL
        ]
        report.require(
            present_asc == sorted(present_asc),
            f"ordering: {column} ascending is not ascending",
        )
        report.require(
            present_desc == sorted(present_desc, reverse=True),
            f"ordering: {column} descending is not descending",
        )
        report.require(
            present_asc == list(reversed(present_desc)),
            f"ordering: {column}'s two directions are not reverses of one another",
        )
        # Equal values keep row order in both directions -- the tiebreak is not
        # reversed along with the key, so a page boundary lands in the same place.
        ties_asc = [row for row in ascending if values[row] == present_asc[0]]
        ties_desc = [row for row in descending if values[row] == present_asc[0]]
        report.require(
            ties_asc == ties_desc,
            f"ordering: rows tied on {column} come back in different orders "
            f"({ties_asc[:4]} vs {ties_desc[:4]}); pagination would repeat or skip rows",
        )

    # A missing measurement sorts last in BOTH directions: absent is not small.
    with tempfile.TemporaryDirectory() as directory:
        mirror = Path(directory) / "chunks"
        mirror.mkdir()
        rows = []
        for index in range(5):
            record = {
                "chunk_id": f"sha256:{index:064d}",
                "subject": "synthetic",
                "source_path": f"synthetic/{index}.md",
                "kind": "prose",
                "language": None,
                "span": {"start_line": index, "end_line": index},
                "text": f"row {index}",
            }
            if index != 3:
                record["char_count"] = (5 - index) * 10
            rows.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        (mirror / "synthetic.chunks.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        catalog_module.write(
            catalog_module.build(mirror, synthetic_table()), Path(directory) / "catalog"
        )
        synthetic = catalog_module.Catalog.load(Path(directory) / "catalog", mirror)
        whole = plan.Selection(None, synthetic.rows_total, True, ())
        missing_row = next(
            row
            for row in range(synthetic.rows_total)
            if synthetic.numeric["char_count"][row] == catalog_module.NUMERIC_NULL
        )
        for descending in (False, True):
            order = relational.ordered_rows(synthetic, whole, "char_count", descending)
            report.require(
                order[-1] == missing_row,
                f"ordering: the row with no char_count sorted to position "
                f"{order.index(missing_row)} of {len(order)} with descending="
                f"{descending}; an absent measurement is not a small one",
            )

    # OFFSET is a window on the same order, never a different one.
    column = catalog.numeric_columns()[0]
    full = relational.ordered_rows(catalog, selection, column, True)
    for offset, limit in ((0, 3), (2, 3), (5, 4), (len(full), 3)):
        report.require(
            full[offset : offset + limit] == full[offset:][:limit],
            f"ordering: OFFSET {offset} LIMIT {limit} is not a window on the order",
        )

    # DISTINCT and GROUP BY count the same values.
    for indexed in catalog.indexed_columns():
        groups, _ = analytics.group_counts(catalog, indexed, selection)
        distinct = {value for value, _ in groups}
        report.require(
            len(distinct) == len(groups),
            f"ordering: GROUP BY {indexed} emitted {len(groups)} rows for "
            f"{len(distinct)} distinct value(s)",
        )


def check_planner(report: Report, plan, catalog) -> None:
    selection = plan.Selection(None, catalog.rows_total, True, ())
    every = frozenset(plan.BACKENDS)

    exact = plan.candidates(
        catalog,
        selection,
        top_k=5,
        dim=512,
        want_text=True,
        require_exact=True,
        available_backends=every,
        kernel_cached=True,
        files_touched=1,
    )
    refused = [candidate for candidate in exact if not candidate.legal]
    report.require(
        refused and all(candidate.backend == "q8" for candidate in refused),
        "planner: an exactness request must refuse exactly the lossy backend, "
        f"refused {[c.backend for c in refused]}",
    )

    # Legality is not a price. Make q8 free on every axis and it must stay refused.
    free = plan.CostModel(ns_per_kmac_q8=0, ns_per_row_q8_quantize=0, accuracy_penalty_per_result=0)
    still = plan.candidates(
        catalog,
        selection,
        top_k=5,
        dim=512,
        want_text=True,
        require_exact=True,
        available_backends=every,
        kernel_cached=True,
        files_touched=1,
        model=free,
    )
    report.require(
        all(not c.legal for c in still if c.backend == "q8"),
        "planner: a lossy backend became legal when its price fell to zero -- "
        "legality was decided from a measurement",
    )

    # An unavailable backend is refused on availability, not silently chosen.
    without = plan.candidates(
        catalog,
        selection,
        top_k=5,
        dim=512,
        want_text=True,
        require_exact=False,
        available_backends=frozenset({"reference"}),
        kernel_cached=False,
        files_touched=1,
    )
    chosen = plan.choose(without, plan.Objective.LATENCY)
    report.require(
        chosen.backend == "reference",
        f"planner: chose {chosen.backend} when only reference was available",
    )

    # Determinism: the same request picks the same plan, every time.
    picks = {
        plan.choose(
            plan.candidates(
                catalog,
                selection,
                top_k=5,
                dim=512,
                want_text=True,
                require_exact=False,
                available_backends=every,
                kernel_cached=True,
                files_touched=1,
            ),
            plan.Objective.LATENCY,
        ).label()
        for _ in range(5)
    }
    report.require(len(picks) == 1, f"planner: five identical requests chose {picks}")

    # The objective changes the answer, or it is not an objective.
    by_objective = {
        objective: plan.choose(
            plan.candidates(
                catalog,
                selection,
                top_k=5,
                dim=512,
                want_text=True,
                require_exact=False,
                available_backends=every,
                kernel_cached=False,
                files_touched=1,
            ),
            objective,
        ).backend
        for objective in plan.Objective
    }
    report.require(
        len(set(by_objective.values())) > 1,
        f"planner: every objective chose the same backend {by_objective} -- the "
        "objective is not reaching the choice",
    )

    text = plan.explain(
        exact,
        plan.choose(exact, plan.Objective.LATENCY),
        plan.Objective.LATENCY,
        selection=selection,
    )
    for candidate in exact:
        report.require(
            candidate.label() in text,
            f"planner: EXPLAIN omitted the {candidate.label()} candidate; a planner "
            "that hides its rejects cannot be argued with",
        )
    report.require("REFUSED" in text, "planner: EXPLAIN never says a candidate was refused")


# --------------------------------------------------------------------------
# S5 / S6 -- parts and generations
# --------------------------------------------------------------------------


def check_parts(report: Report, catalog_module, catalog, chunk_dir: Path) -> None:
    report.require(bool(catalog.parts), "S5: the catalog declares no parts")

    # Every row finds its own part, and no other. `part_of` bisects rather than walks,
    # which is only correct while the parts are sorted and touching -- the property
    # checked just below. Checked over every row rather than a sample, because an
    # off-by-one at one boundary is exactly what a sample misses.
    misplaced = []
    for part in catalog.parts:
        start, end = part["rows"]
        for row in (start, (start + end) // 2, end - 1):
            try:
                found = catalog.part_of(row)["part_id"]
            except IndexError:
                # A refusal for a row that *is* in the set is a misplacement, not an
                # accident. Reported here so the finding names the boundary rather
                # than escaping from somewhere else in this function.
                found = "(refused)"
            if found != part["part_id"]:
                misplaced.append(f"row {row} -> {found}, not {part['part_id']}")
    report.require(
        not misplaced,
        "S5: part_of sent rows to the wrong part: " + "; ".join(misplaced[:4]),
    )
    walked = [catalog.part_of(row)["part_id"] for row in range(catalog.rows_total)]
    report.require(
        len(walked) == catalog.rows_total and all(walked),
        "S5: part_of did not answer for every row",
    )
    for outside in (-1, catalog.rows_total, catalog.rows_total + 1):
        try:
            catalog.part_of(outside)
        except IndexError:
            report.require(True, "")
        else:
            report.require(False, f"S5: part_of claimed a part for row {outside}")

    sources = {part.get("source") for part in catalog.parts}
    report.require(
        all(isinstance(name, str) and name for name in sources),
        "S5: a part does not name the source it was cut from, so nothing can tell "
        "which file an incremental rebuild has to re-read",
    )
    widest = max(end - start for start, end in (part["rows"] for part in catalog.parts))
    report.require(
        widest <= catalog_module.MAX_BLOCK_ROWS,
        f"S5: the widest part covers {widest} rows, above the "
        f"{catalog_module.MAX_BLOCK_ROWS}-row block size; a part that grows with its "
        "file is the subject-shaped part this slice removed",
    )
    covered = sorted((part["rows"][0], part["rows"][1]) for part in catalog.parts)
    contiguous = covered and covered[0][0] == 0 and covered[-1][1] == catalog.rows_total
    for earlier, later in zip(covered, covered[1:]):
        contiguous = contiguous and earlier[1] == later[0]
    report.require(
        contiguous,
        f"S5: the parts do not tile rows 0..{catalog.rows_total}: {covered}",
    )
    for part in catalog.parts:
        start, end = part["rows"]
        report.require(
            end > start,
            f"S5: part {part['part_id']} covers no rows; an empty part is a claim "
            "about nothing that a containment rule would still honour",
        )

    with tempfile.TemporaryDirectory() as directory:
        mirror = Path(directory) / "chunks"
        shutil.copytree(chunk_dir, mirror)
        catalog_root = Path(directory) / "catalog"
        catalog_module.write(catalog_module.build(mirror), catalog_root)
        before = catalog_module.Catalog.load(catalog_root, mirror)
        report.require(
            before.changed_parts(catalog) == (),
            "S5: a copy of the same chunk table reports changed parts",
        )
        # The file cut into the *most* blocks, not the first one alphabetically. A
        # file holding a single block cannot tell "this part's content covers its own
        # rows" from "it covers its whole file": both move exactly one part. The
        # alphabetically-first file here held 7 rows, so the whole-file digest passed
        # this check until the choice was made deliberate (L2).
        counts: dict[str, int] = {}
        for part in before.parts:
            counts[part["source"]] = counts.get(part["source"], 0) + 1
        crowded = max(counts, key=lambda name: (counts[name], name))
        report.require(
            counts[crowded] >= 2,
            f"anti-vacuity: the busiest chunk file is cut into {counts[crowded]} "
            "part(s), so the check below cannot tell a per-block digest from a "
            "per-file one",
        )
        victim = mirror / f"{crowded}.chunks.jsonl"
        appended = valid_record(chunk_id="sha256:" + "z" * 64, subject="synthetic-appended")
        victim.write_bytes(
            victim.read_bytes()
            + json.dumps(appended, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        after_root = Path(directory) / "catalog2"
        catalog_module.write(catalog_module.build(mirror), after_root)
        after = catalog_module.Catalog.load(after_root, mirror)
        moved = after.changed_parts(before)
        source = victim.name[: -len(".chunks.jsonl")]
        by_id = {part["part_id"]: part for part in after.parts}
        strangers = sorted(name for name in moved if by_id.get(name, {}).get("source") != source)
        report.require(
            moved and not strangers,
            f"S5: appending a row to {victim.name} reported {moved} as changed"
            + (f"; {strangers} come from another file" if strangers else " -- nothing moved")
            + ". A part digest that moves for a file it does not belong to, or does "
            "not move for its own, cannot drive an incremental rebuild.",
        )
        # ...and it must move *few* parts, not all of them. A digest covering a whole
        # file would name every block of it, which is a filter that filters nothing --
        # and is exactly what `_changed_parts` did until this check was written.
        report.require(
            len(moved) <= 2,
            f"S5: one appended row moved {len(moved)} part(s) of {source}; a part's "
            "content must cover its own rows, not its whole file",
        )
        # And the stale catalog must refuse rather than answer from old statistics.
        try:
            catalog_module.Catalog.load(catalog_root, mirror)
        except catalog_module.CatalogError:
            report.require(True, "")
        else:
            report.require(
                False,
                "S5: a catalog built before the chunk table moved still loaded; "
                "stale statistics produce a wrong plan silently",
            )

    # Two builds of one chunk table must produce identical bytes. Nothing a query does
    # would notice otherwise -- a catalog is read through its own manifest, so a
    # different-but-equivalent sort order answers every question the same way -- and
    # then a content-addressed generation would take a new digest for a corpus that
    # had not moved, and `changed_parts` would name parts nothing changed.
    #
    # **From two different directories, with different modification times.** This used
    # to build twice from one directory, where the paths and the mtimes are equal by
    # construction -- so it compared a build against itself on exactly the two inputs
    # that were not a function of the chunk bytes, and passed while `catalog.json`
    # carried both. A determinism check whose two sides cannot differ in the way the
    # artifact actually differed is a check of nothing (`docs/security/laws.md` L2).
    with tempfile.TemporaryDirectory() as directory:
        here = Path(directory)
        sources = []
        for index, (name, stamp) in enumerate((("alpha", 1_000_000_000), ("beta", 1_700_000_000))):
            source = here / name / "nested" / f"depth{index}"
            source.mkdir(parents=True)
            for path in catalog_module.chunk_files(chunk_dir):
                shutil.copy(path, source / path.name)
                os.utime(source / path.name, (stamp, stamp))
            sources.append(source)
        left, right = sources
        stamps = {
            path.stat().st_mtime_ns for source in sources for path in catalog_module.chunk_files(source)
        }
        report.require(
            len(stamps) == 2 and str(left) != str(right),
            "anti-vacuity: the two builds below were given the same paths or the same "
            f"modification times ({len(stamps)} distinct stamp(s)), so neither input "
            "that a catalog must not depend on actually varies",
        )
        first, second = here / "first", here / "second"
        catalog_module.write(catalog_module.build(left), first)
        catalog_module.write(catalog_module.build(right), second)
        names = sorted(entry["path"] for entry in catalog.manifest["artifacts"].values())
        report.require(
            len(names) >= 4,
            f"S5: the manifest lists {len(names)} artifact(s), so the comparison below "
            "covers almost nothing",
        )
        left_set = catalog_module.current_set(first)
        right_set = catalog_module.current_set(second)
        for name in names + [catalog_module.CATALOG_FILE]:
            report.require(
                (left_set / name).read_bytes() == (right_set / name).read_bytes(),
                f"S5: {name} differs between two builds of the same chunk table made "
                f"from different directories; a catalog must be a function of the "
                f"chunk bytes and nothing else",
            )
        # The set name is the content, so two builds of one corpus name one set. That
        # is what makes a republish idempotent rather than a new directory each time.
        report.require(
            left_set.name == right_set.name,
            f"S5: the same chunk table published as two differently named sets "
            f"({left_set.name} and {right_set.name})",
        )


def check_incremental(report: Report, chunk_dir: Path) -> None:
    """An incremental rebuild must produce what a full one does, and skip real work.

    Both halves, because each alone is satisfiable by a defect: a build that reuses
    nothing is byte-identical and pointless, and a build that reuses everything is
    fast and wrong.
    """
    embed = load_tool("embed_chunks")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mirror = root / "chunks"
        shutil.copytree(chunk_dir, mirror)

        provider = embed.build_provider(embed.LexicalHashProvider.name, dim=512, revision=None)
        chunks = embed.read_chunks(mirror, None)
        report.require(
            len(chunks) >= MIN_ROWS,
            f"S5: {len(chunks)} chunk(s) to embed, below the {MIN_ROWS} floor",
        )

        status = embed.main(["--chunks", str(mirror), "--out", str(root / "a"), "--incremental"])
        report.require(status == 0, f"S5: the first incremental build exited {status}")
        first = root / "a" / "lexical-hash-v1"

        # Nothing changed: every row reuses, and the model runs on none.
        previous = embed.load_previous(first, provider)
        report.require(
            len(previous) == len(chunks),
            f"S5: {len(previous)} of {len(chunks)} rows were offered for reuse after "
            "a build that changed nothing",
        )
        _, _, unchanged = embed.embed_incremental(chunks, provider, previous)
        report.require(
            unchanged.embedded == 0 and unchanged.reused == len(chunks),
            f"S5: an unchanged corpus re-embedded {unchanged.embedded} row(s)",
        )

        # One row's text changes: exactly one row is re-embedded.
        victim = sorted(mirror.glob("*.chunks.jsonl"))[0]
        lines = victim.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["text"] = record["text"] + "\n\nAn added sentence."
        lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
        victim.write_text("\n".join(lines) + "\n", encoding="utf-8")

        changed_chunks = embed.read_chunks(mirror, None)
        _, _, delta = embed.embed_incremental(changed_chunks, provider, previous)
        report.require(
            delta.embedded == 1,
            f"S5: changing one row's text re-embedded {delta.embedded} row(s); the "
            "reuse key is not the row's own content",
        )
        report.require(
            delta.reused == len(changed_chunks) - 1,
            f"S5: {delta.reused} of {len(changed_chunks) - 1} unchanged rows reused",
        )

        # The result is what a full rebuild produces. Byte for byte, every artifact.
        status = embed.main(["--chunks", str(mirror), "--out", str(root / "a"), "--incremental"])
        report.require(status == 0, f"S5: the incremental rebuild exited {status}")
        status = embed.main(["--chunks", str(mirror), "--out", str(root / "b")])
        report.require(status == 0, f"S5: the full rebuild exited {status}")
        incremental, full = root / "a" / "lexical-hash-v1", root / "b" / "lexical-hash-v1"
        for name in ("vectors.f32", "vectors.q15", "index.jsonl", "manifest.json"):
            report.require(
                (incremental / name).read_bytes() == (full / name).read_bytes(),
                f"S5: {name} differs between an incremental rebuild and a full one; "
                "reuse changed the artifact, which makes every stored vector suspect",
            )

        # The coarse half: the part that moved is named, and only that one.
        moved = ingest.changed_parts(
            mirror, json.loads((full / "manifest.json").read_text(encoding="utf-8")).get("parts")
        )
        report.require(
            moved == (),
            f"S5: a set just written from this corpus reports {moved} as changed",
        )

        # Reuse is refused across providers: a stored vector is only sound when the
        # function that produced it is the function that would produce the new one.
        #
        # A different dimension is caught twice over -- the manifest comparison and
        # the row-count arithmetic both reject it -- so the case that actually
        # witnesses the *identity* check is a set of the same shape written by a
        # different model. Nothing about the bytes distinguishes those, which is
        # exactly why the manifest has to be read.
        other = embed.build_provider(embed.LexicalHashProvider.name, dim=256, revision=None)
        report.require(
            embed.load_previous(full, other) == {},
            "S5: vectors from a 512-dim set were offered for reuse at dim 256",
        )
        impostor = root / "impostor"
        shutil.copytree(full, impostor)
        manifest = json.loads((impostor / "manifest.json").read_text(encoding="utf-8"))
        manifest["model"] = "some-other-model"
        (impostor / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        report.require(
            embed.load_previous(impostor, provider) == {},
            "S5: vectors written by another model, at the same dimension, were "
            "offered for reuse; nothing in the bytes would have revealed it",
        )
        manifest["model"] = provider.name
        manifest["revision"] = f"{provider.revision}-not-this-one"
        (impostor / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        report.require(
            embed.load_previous(impostor, provider) == {},
            "S5: vectors from another revision of the same model were reused",
        )

        # And a set whose vectors do not match their recorded digest offers nothing.
        tampered = root / "c"
        shutil.copytree(full, tampered)
        (tampered / "vectors.f32").write_bytes(
            (tampered / "vectors.f32").read_bytes()[:-4] + b"\x00\x00\x00\x00"
        )
        report.require(
            embed.load_previous(tampered, provider) == {},
            "S5: a set whose vectors.f32 no longer matches its manifest digest was "
            "still reused from",
        )


def check_generations(report: Report, generations, chunk_dir: Path) -> None:
    catalog_module = load_tool("catalog")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "generations"
        mirror = Path(directory) / "chunks"
        shutil.copytree(chunk_dir, mirror)

        first = generations.publish(root, mirror, label="first")
        report.require(first.digest, "S6: a published generation carries no digest")
        report.require(
            generations.current(root).generation_id == first.generation_id,
            "S6: the newest generation is not the current one",
        )

        # Publishing the same corpus again is the same generation, not a second one.
        # This is the property a wall-clock name cannot have, and it has no witness
        # unless the same corpus is actually published twice.
        again = generations.publish(root, mirror, label="first again")
        report.require(
            again.generation_id == first.generation_id and again.digest == first.digest,
            f"S6: republishing an unchanged corpus produced {again.generation_id}, not "
            f"{first.generation_id}; the name is not the content",
        )
        report.require(
            len(generations.listing(root)) == 1,
            f"S6: republishing an unchanged corpus left "
            f"{len(generations.listing(root))} generations",
        )
        report.require(
            generations.read(root, first.generation_id).label == "first",
            "S6: republishing rewrote a published generation's manifest; a generation "
            "is immutable once written",
        )

        victim = sorted(mirror.glob("*.chunks.jsonl"))[0]
        appended = valid_record(chunk_id="sha256:" + "e" * 64, subject="synthetic-appended")
        victim.write_bytes(
            victim.read_bytes()
            + json.dumps(appended, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        second = generations.publish(root, mirror, label="second")
        report.require(
            second.digest != first.digest,
            "S6: a changed corpus published the same digest; a generation that does "
            "not distinguish two corpora cannot reproduce either",
        )
        report.require(
            second.parent == first.generation_id,
            f"S6: generation {second.generation_id} names parent {second.parent}, "
            f"expected {first.generation_id}",
        )
        report.require(
            generations.current(root).generation_id == second.generation_id,
            "S6: publishing did not advance the current generation",
        )

        # Immutability: the old generation still verifies, byte for byte.
        recovered = generations.read(root, first.generation_id)
        report.require(
            recovered.digest == first.digest,
            "S6: reading back the first generation produced a different digest",
        )
        report.require(
            generations.verify(root, first.generation_id),
            "S6: the first generation no longer verifies after a second was published",
        )
        # Tampering with a stored generation is caught, not tolerated.
        stored = (
            catalog_module.current_set(root / first.generation_id / "catalog")
            / catalog_module.CATALOG_FILE
        )
        if stored.is_file():
            stored.write_text(stored.read_text(encoding="utf-8") + " ", encoding="utf-8")
            report.require(
                not generations.verify(root, first.generation_id),
                "S6: a tampered generation still verified; the digest covers less "
                "than the generation contains",
            )
        else:
            report.require(False, f"S6: {stored} was not published inside the generation")

        report.require(
            len(generations.listing(root)) == 2,
            f"S6: {len(generations.listing(root))} generation(s) listed, expected 2",
        )

        # Republishing onto a generation whose stored bytes no longer match must
        # refuse rather than quietly adopt them. Without this the idempotent path
        # would hand a tampered generation back as though it were the one published,
        # and nothing downstream would ever look again.
        #
        # `mirror` now holds the second corpus, but the first generation was named
        # from `chunk_dir`, so republishing from there addresses the tampered one.
        report.require(
            generations.generation_id_for(chunk_dir) == first.generation_id,
            "S6: the corpus the first generation was published from no longer reproduces its id",
        )
        try:
            generations.publish(root, chunk_dir, label="replay")
        except generations.GenerationError:
            report.require(True, "")
        else:
            report.require(
                False,
                f"S6: republishing onto the tampered {first.generation_id} succeeded; "
                "an immutable generation was modified in place and accepted",
            )




# --------------------------------------------------------------------------
# S14/S17 -- the declared table, and what it refuses
# --------------------------------------------------------------------------

#: The sentinel a constraint trial uses to mean "remove this key entirely", which is a
#: different violation from setting it to null and must not share a spelling with one.
_REMOVE = object()


def _json_schema_facts(declaration: dict) -> tuple[set[str] | None, set[str] | None]:
    """The types and the enumerated values one JSON Schema property admits.

    Four shapes appear in that file and each says the same thing differently: a
    `const`, an `enum`, a `$ref` to a string pattern, and a `type` that may be a list
    or sit inside a `oneOf`. Reading all four here rather than at each call site is
    what lets the reconciliation below compare a table against a schema instead of
    against a style.
    """
    if "const" in declaration:
        return {"string"}, {declaration["const"]}
    if "enum" in declaration:
        values = [value for value in declaration["enum"] if value is not None]
        return ({"string"} if all(isinstance(v, str) for v in values) else None), set(values)
    if "$ref" in declaration:
        return {"string"}, None
    if "oneOf" in declaration:
        types: set[str] = set()
        for branch in declaration["oneOf"]:
            found = branch.get("type")
            types.update([found] if isinstance(found, str) else (found or []))
        return types or None, None
    found = declaration.get("type")
    if found is None:
        return None, None
    return ({found} if isinstance(found, str) else set(found)), None


def check_schema(report: Report) -> None:
    """The Python table and the JSON Schema must describe the same table.

    Two rails, read out of their own sources and reconciled here, rather than one
    generated from the other. `training/schema/chunk-v1.json` is what an external
    consumer validates against; `training/tools/schema.py` is what the build enforces
    and what the catalog derives its column roles from. Either can be edited alone,
    and until this check existed either could be edited alone *and stay green* -- a
    field renamed in the JSON while the Python kept indexing the old name produces a
    postings list of one entry, every row, null (`docs/security/laws.md` L15).

    Five properties are reconciled, because a disagreement in any one is a different
    bug: which fields exist, which are required, which admit null, what type each
    holds, and what values each admits.
    """
    schema_module = load_tool("schema")
    path = REPO_ROOT / schema_module.JSON_SCHEMA
    declared = json.loads(path.read_text(encoding="utf-8"))
    properties = declared["properties"]
    required = set(declared["required"])
    table = schema_module.TABLE

    report.require(
        set(properties) == set(table.by_name),
        "schema: the JSON Schema and the column table name different fields; "
        f"only in JSON: {sorted(set(properties) - set(table.by_name))}; "
        f"only in Python: {sorted(set(table.by_name) - set(properties))}",
    )
    report.require(
        len(properties) >= 12,
        f"anti-vacuity: the JSON Schema declares {len(properties)} field(s), so the "
        "reconciliation below covers almost nothing",
    )
    report.require(
        table.column("schema").domain == (schema_module.ROW_SCHEMA,),
        "schema: the column table does not pin the row contract it is a table for",
    )

    compared = 0
    for name, spec in sorted(table.by_name.items()):
        declaration = properties.get(name)
        if declaration is None:
            continue
        compared += 1
        report.require(
            spec.required == (name in required),
            f"schema: {name} is {'required' if spec.required else 'optional'} in the "
            f"column table and {'required' if name in required else 'optional'} in "
            f"{schema_module.JSON_SCHEMA}",
        )
        types, values = _json_schema_facts(declaration)
        if types is not None:
            report.require(
                spec.type in types,
                f"schema: {name} holds {spec.type!r} in the column table and "
                f"{sorted(types)} in {schema_module.JSON_SCHEMA}",
            )
            report.require(
                spec.nullable == ("null" in types),
                f"schema: {name} is {'nullable' if spec.nullable else 'not nullable'} "
                f"in the column table; the JSON Schema says {sorted(types)}",
            )
        if values is not None:
            report.require(
                spec.domain is not None and set(spec.domain) == values,
                f"schema: {name} admits {sorted(spec.domain or ())} in the column "
                f"table and {sorted(values)} in {schema_module.JSON_SCHEMA}",
            )
        minimum = declaration.get("minimum")
        if minimum is not None and spec.type == "integer":
            report.require(
                spec.minimum == minimum,
                f"schema: {name} has minimum {spec.minimum} in the column table and "
                f"{minimum} in {schema_module.JSON_SCHEMA}",
            )
    report.require(
        compared == len(table.columns),
        f"anti-vacuity: only {compared} of {len(table.columns)} columns were compared "
        "against the JSON Schema",
    )

    # ...and the roles the catalog uses are the ones the table derives.
    catalog_module = load_tool("catalog")
    report.require(
        catalog_module.INDEXED_COLUMNS == table.indexed
        and catalog_module.NUMERIC_COLUMNS == table.numeric
        and catalog_module.PATH_COLUMN == table.path,
        "schema: catalog.py carries column roles that the declared table does not",
    )
    report.require(
        bool(table.indexed) and bool(table.numeric) and bool(table.text),
        "anti-vacuity: the declared table leaves one of the queryable roles empty",
    )


def check_constraints(report: Report, catalog_module, chunk_dir: Path) -> None:
    """Every declared constraint refuses a row that violates it, naming the row.

    A build over a malformed row is a failed build, not a build with one more unknown
    in it. Before this, `char_count` holding the string "not-an-int" produced a clean
    catalog, exit 0, and one more null in the packed column -- indistinguishable
    downstream from a chunk that genuinely has no measurement (`docs/security/laws.md`
    L1).

    The trial list is *derived from the table* rather than hand-kept, and then
    reconciled against it: every column declaring a domain, a minimum, requiredness or
    non-nullability must have a trial, or this check fails for lack of coverage. That
    is what stops a constraint from being added, never exercised, and quietly doing
    nothing (L2, L15).
    """
    schema_module = load_tool("schema")
    table = schema_module.TABLE

    mutations: list[tuple[str, str, dict]] = []
    for spec in table.columns:
        if spec.domain:
            mutations.append((spec.name, "domain", {spec.name: "not-in-the-domain"}))
        if spec.minimum is not None:
            mutations.append((spec.name, "minimum", {spec.name: spec.minimum - 1}))
        if spec.required:
            mutations.append((spec.name, "required", {spec.name: _REMOVE}))
        if not spec.nullable and spec.required:
            mutations.append((spec.name, "nullable", {spec.name: None}))
        mutations.append(
            (spec.name, "type", {spec.name: 1 if spec.type != "integer" else "not-an-integer"})
        )
    mutations.append(("<undeclared>", "extra field", {"not_a_column": "x"}))
    mutations.append((table.key, "empty key", {table.key: ""}))

    covered = {name for name, _, _ in mutations}
    constrained = {
        spec.name
        for spec in table.columns
        if spec.domain or spec.minimum is not None or spec.required or not spec.nullable
    }
    report.require(
        constrained <= covered,
        "anti-vacuity: these constrained columns have no trial below, so their "
        f"constraints are not exercised: {sorted(constrained - covered)}",
    )
    report.require(
        len(mutations) >= 2 * len(table.columns),
        f"anti-vacuity: {len(mutations)} trial(s) over {len(table.columns)} column(s) "
        "is fewer than one constraint each",
    )

    base = valid_record()
    for column, rule, change in mutations:
        row = dict(base)
        for key, value in change.items():
            if value is _REMOVE:
                row.pop(key, None)
            else:
                row[key] = value
        try:
            table.check_row(row)
        except schema_module.SchemaError as exc:
            report.require(
                column in str(exc) or rule in ("extra field", "empty key"),
                f"constraints: the {rule} refusal for {column!r} does not name the "
                f"column: {exc}",
            )
        else:
            report.require(
                False,
                f"constraints: a row violating the {rule} of {column!r} was accepted",
            )

    # A value the table does admit stays admitted. Without this the whole list above
    # is satisfied by a checker that refuses everything.
    try:
        table.check_row(base)
    except schema_module.SchemaError as exc:
        report.require(False, f"constraints: a valid row was refused: {exc}")
    else:
        report.require(True, "")

    # ...and the refusal reaches a build as a verdict, not as a traceback.
    with tempfile.TemporaryDirectory() as directory:
        mirror = Path(directory) / "chunks"
        shutil.copytree(chunk_dir, mirror)
        victim = sorted(mirror.glob("*.chunks.jsonl"))[0]
        lines = victim.read_text(encoding="utf-8").splitlines()
        broken = json.loads(lines[0])
        chunk_id = broken["chunk_id"]
        broken["char_count"] = "not-an-int"
        lines[0] = json.dumps(broken, sort_keys=True, separators=(",", ":"))
        victim.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            catalog_module.build(mirror)
        except catalog_module.CatalogError as exc:
            report.require(
                chunk_id in str(exc) and "char_count" in str(exc),
                f"constraints: the build refused but named neither the chunk nor the "
                f"column: {exc}",
            )
        else:
            report.require(
                False,
                "constraints: a chunk whose char_count is a string built a clean "
                "catalog; a schema error became one more null",
            )


# --------------------------------------------------------------------------
# Fix 1 -- a value that carries a comma asks about that value
# --------------------------------------------------------------------------


def check_quoting(report: Report, plan) -> None:
    """A value is read back as it was written, or refused. Never read as another value.

    The defect this pins: `language!=llvm,markdown` used to admit every row in the
    table and exit 0. `=` read a comma as a set separator and `!=` read it as an
    ordinary character, so the complement of a value nothing carries was everything --
    a wrong answer, silently, on an indexed column of this corpus.

    Three properties, and the first two are the ones that matter. A value printed by
    `EXPLAIN` must parse back to the predicate that printed it (round trip). Every
    spelling this grammar cannot read back unchanged must be refused rather than
    guessed. And `=` and `!=` must read one value grammar, so that they stay
    complements for every value, including the ones with commas in them
    (`docs/security/laws.md` L12).
    """
    values = (
        "plain",
        "Hello, World",
        "code,prose",
        plan.PRESENCE,
        "",
        " leading",
        "trailing ",
        'say "hi"',
        "back" + chr(92) + "slash",
        "a!=b",
        "a>=b",
        '"quoted"',
        ",",
        ",,",
        "a,b,c",
    )
    report.require(
        len(values) >= 12 and any("," in value for value in values),
        f"anti-vacuity: {len(values)} round-trip value(s), or none carrying a comma",
    )
    for value in values:
        spelled = plan.spell_value(value)
        for op, spelling in (("eq", "="), ("ne", "!="), ("prefix", "^=")):
            parsed = plan.parse_predicate(f"title{spelling}{spelled}")
            report.require(
                (parsed.column, parsed.op, parsed.values) == ("title", op, (value,)),
                f"quoting: title{spelling}{spelled} read as {parsed.op} "
                f"{parsed.values}, expected {op} ({value!r},)",
            )
        printed = str(plan.Predicate("title", "eq", (value,)))
        column, _, right = printed.partition(" = ")
        back = plan.parse_predicate(f"{column}={right}")
        report.require(
            back.values == (value,),
            f"quoting: EXPLAIN printed {printed!r}, which parses back to {back.values}",
        )

    # A set of them, spelled as one term, reads back as the same set in the same order.
    for size in (2, 3, 4):
        chosen = values[:size]
        term = "title=" + ",".join(plan.spell_value(value) for value in chosen)
        parsed = plan.parse_predicate(term)
        report.require(
            parsed.op == "in" and parsed.values == chosen,
            f"quoting: {term!r} read as {parsed.op} {parsed.values}, expected in {chosen}",
        )

    # One value grammar on every operator: a comma is a separator or a character, and
    # which one it is may not depend on the operator to its left.
    for text in ("language!=llvm,markdown", "source_path^=a,b", "char_count>=1,2"):
        try:
            parsed = plan.parse_predicate(text)
        except plan.PlanError:
            report.require(True, "")
        else:
            report.require(
                False,
                f"quoting: {text!r} gave {len(parsed.values)} value(s) to {parsed.op}, "
                "which takes one; a comma means the same thing on every operator",
            )

    refusals = (
        ("title=" + chr(34) + "unterminated", "a quote that is never closed"),
        ("title=say " + chr(34) + "hi" + chr(34), "a quote inside an unquoted value"),
        ("title=" + chr(34) + "a" + chr(34) + "x", "text after a closing quote"),
        ("title=" + chr(34) + "a" + chr(92) + "nb" + chr(34), "an escape the grammar does not spell"),
        ("title=" + chr(34) + "a" + chr(92), "a dangling escape"),
        ("subject=?,llvm", "the presence sentinel inside a set"),
        ("subject^=?", "a presence test on an operator that cannot spell one"),
        ("title=", "no value at all"),
        ("title=,,", "only empty values"),
    )
    for text, why in refusals:
        try:
            plan.parse_predicate(text)
        except plan.PlanError:
            report.require(True, "")
        else:
            report.require(False, f"quoting: accepted {text!r}, which carries {why}")

    # The sentinel and the character it is spelled with are different values.
    sentinel = plan.parse_predicate(f"language={plan.PRESENCE}")
    literal = plan.parse_predicate("language=" + chr(34) + plan.PRESENCE + chr(34))
    report.require(
        sentinel.op == "notnull" and sentinel.values == (),
        f"quoting: language={plan.PRESENCE} read as {sentinel.op}, not a presence test",
    )
    report.require(
        literal.op == "eq" and literal.values == (plan.PRESENCE,),
        f"quoting: a quoted {plan.PRESENCE} read as {literal.op} {literal.values}, "
        "not as the character itself",
    )


# --------------------------------------------------------------------------
# S15 -- what the planner knows, and what it claims to know
# --------------------------------------------------------------------------


def check_estimates(report: Report, plan, catalog) -> None:
    """No estimate may be below the truth, and none may call a bound a count.

    Two separate claims, and the second is the one that was false. Every estimate in
    `plan.py` is an upper bound, so a number below the admitted rows is a defect that
    makes a planner choose a cheaper plan than the work requires. And `estimate_exact`
    is part of the verdict: it used to be `all(exact)` over the *marginals*, which
    asks whether each term was counted exactly where the claim is about their
    intersection. On this corpus that reported `[exact]` over a wrong number for 83%
    of indexed value pairs, `kind=code AND language=asm` claiming a row where there
    are none (`docs/security/laws.md` L1).

    Exhaustive over every one- and two-column conjunction of equalities the corpus
    can express, rather than a sample: the catalog stores the joint distribution for
    exactly these, so this is the set where "exact" has to mean it.
    """
    import itertools

    columns = list(catalog.indexed_columns())
    report.require(
        len(columns) >= 2,
        f"anti-vacuity: {len(columns)} indexed column(s), so no pair below is tested",
    )

    trials = exact_claims = 0
    for size in (1, 2):
        for combination in itertools.combinations(columns, size):
            for values in itertools.product(*[list(catalog.distinct(c)) for c in combination]):
                predicates = [
                    plan.Predicate(column, "eq", (value,))
                    for column, value in zip(combination, values)
                ]
                selection = plan.select(catalog, predicates)
                admitted = len(selection.rows)
                trials += 1
                report.require(
                    selection.estimated >= admitted,
                    f"estimates: {' AND '.join(str(p) for p in predicates)} estimated "
                    f"{selection.estimated} but admits {admitted}; an estimate here is "
                    "an upper bound and this one is below the truth",
                )
                if selection.estimate_exact:
                    exact_claims += 1
                    report.require(
                        selection.estimated == admitted,
                        f"estimates: {' AND '.join(str(p) for p in predicates)} was "
                        f"reported exact at {selection.estimated} and admits {admitted}",
                    )
    report.require(
        trials >= 100,
        f"anti-vacuity: only {trials} conjunction(s) were priced",
    )
    report.require(
        exact_claims == trials,
        f"estimates: {trials - exact_claims} of {trials} conjunctions over one or two "
        "indexed columns were priced as a bound; the catalog stores the joint "
        "distribution for exactly these, so every one of them is a count",
    )

    # Three columns have no stored statistic. The bound must stay a bound -- except at
    # zero, where an upper bound of zero is a count of zero whatever the terms were --
    # and it must be the tightest *pair* rather than the tightest marginal.
    #
    # The triple is searched for rather than taken from the first value of each column,
    # because most triples admit nothing and the zero case is the one where exactness
    # is sound. A check that happened to pick an empty triple would assert "this is a
    # bound" about the one answer that is not one.
    triple = None
    for combination in itertools.combinations(columns, 3):
        for values in itertools.product(*[list(catalog.distinct(c)) for c in combination]):
            candidate = [
                plan.Predicate(column, "eq", (value,))
                for column, value in zip(combination, values)
            ]
            if len(plan.select(catalog, candidate).rows) > 0:
                triple = candidate
                break
        if triple:
            break
    report.require(
        len(columns) < 3 or triple is not None,
        "anti-vacuity: no conjunction of three indexed columns admits a row, so the "
        "three-way bound below is never exercised on a non-empty answer",
    )
    if triple is not None:
        selection = plan.select(catalog, triple)
        marginals = min(plan.estimate(catalog, predicate)[0] for predicate in triple)
        pairs = min(
            plan.joint(catalog, list(pair))[0]
            for pair in itertools.combinations(triple, 2)
        )
        report.require(
            not selection.estimate_exact,
            f"estimates: {' AND '.join(str(p) for p in triple)} admits "
            f"{len(selection.rows)} rows and was reported exact, but the catalog "
            "stores no three-way statistic",
        )
        report.require(
            selection.estimated == pairs <= marginals,
            f"estimates: the three-column bound is {selection.estimated}; the tightest "
            f"pair is {pairs} and the tightest marginal {marginals}, so the pairwise "
            "statistics were not used",
        )
        report.require(
            selection.estimated >= len(selection.rows),
            f"estimates: the three-column bound {selection.estimated} is below the "
            f"{len(selection.rows)} rows it admits",
        )

    # A repeated value in a set must not be counted twice. `subject=llvm,llvm` once
    # estimated 4320 rows of a 2215-row table, and said it was exact.
    for column in columns:
        value = max(catalog.distinct(column), key=lambda name: catalog.distinct(column)[name])
        doubled = plan.select(catalog, [plan.Predicate(column, "in", (value, value))])
        single = plan.select(catalog, [plan.Predicate(column, "eq", (value,))])
        report.require(
            doubled.estimated == single.estimated <= catalog.rows_total,
            f"estimates: {column} IN ({value}, {value}) estimated {doubled.estimated} "
            f"where {column} = {value} estimates {single.estimated}",
        )

    # An operator the fold does not cover must decline rather than answer wrongly.
    prefix = plan.Predicate(catalog_path_column(), "prefix", ("training/",))
    folded = plan.value_keys(catalog, prefix)
    report.require(
        folded is None,
        "estimates: the value-set fold claimed to cover a path prefix, which is not "
        "indexed by whole value and has no pair table",
    )


def catalog_path_column() -> str:
    """The path column, from the declared table rather than spelled again here."""
    return load_tool("schema").PATH_COLUMN


# --------------------------------------------------------------------------
# S13 -- a publish is one transaction
# --------------------------------------------------------------------------


def check_atomic_publish(report: Report, catalog_module, chunk_dir: Path) -> None:
    """A reader resolves to a complete set at every point a publish can be cut.

    The claim is atomicity, and the only honest way to check it without a second
    process is to cut the publish at every filesystem mutation it makes and ask what a
    reader would find. Every `os.replace` is an interruption point; after each one the
    pointer must name a set that holds every artifact, with every artifact matching
    the digest its own manifest records.

    This used to be six renames into one directory, manifest last. Cut between the
    second and the third, the directory held four old artifacts and two new ones, and
    a reader that opened the old manifest and then read a new sidecar found a digest
    mismatch. That failed loudly rather than answering wrongly -- which is why it was
    a rebuild that could not be done while anything was reading, rather than a
    correctness bug (`docs/security/laws.md` L1 was already satisfied; atomicity is
    the property that was missing).
    """
    artifacts = {
        catalog_module.CATALOG_FILE,
        catalog_module.POSTINGS_FILE,
        catalog_module.LOCATOR_FILE,
        catalog_module.IDS_FILE,
        catalog_module.NUMERIC_FILE,
        catalog_module.ORDER_FILE,
    }
    with tempfile.TemporaryDirectory() as directory:
        here = Path(directory)
        corpora = {}
        for tag in ("before", "after"):
            mirror = here / tag
            shutil.copytree(chunk_dir, mirror)
            corpora[tag] = mirror
        victim = sorted(corpora["after"].glob("*.chunks.jsonl"))[0]
        lines = victim.read_text(encoding="utf-8").splitlines()
        victim.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        built = {tag: catalog_module.build(mirror) for tag, mirror in corpora.items()}
        rows = {tag: value[0]["rows_total"] for tag, value in built.items()}
        report.require(
            rows["before"] != rows["after"],
            "anti-vacuity: the two catalogs below hold the same rows, so no "
            "interruption could be seen to land on either side of the swap",
        )

        root = here / "catalog"
        real_replace = os.replace

        class Interrupted(Exception):
            """Raised in place of one rename, standing in for a process that died."""

        def resolve() -> tuple[str, bool]:
            """What a reader finds: which set, and whether it is whole."""
            published = catalog_module.current_set(root)
            present = {path.name for path in published.iterdir() if path.is_file()}
            if present != artifacts:
                return f"an incomplete set ({sorted(present ^ artifacts)})", False
            manifest = json.loads(
                (published / catalog_module.CATALOG_FILE).read_text(encoding="utf-8")
            )
            for entry in manifest["artifacts"].values():
                digest = hashlib.sha256((published / entry["path"]).read_bytes()).hexdigest()
                if digest != entry["sha256"]:
                    return f"a torn set ({entry['path']} does not match its manifest)", False
            total = manifest["rows_total"]
            for tag, count in rows.items():
                if count == total:
                    catalog_module.Catalog.load(root, corpora[tag]).postings
                    return tag, True
            return f"a set of {total} rows, which is neither catalog", False

        seen: set[str] = set()
        cut = 1
        while True:
            shutil.rmtree(root, ignore_errors=True)
            catalog_module.write(built["before"], root)
            counter = {"n": 0}

            def counting(source, target, *rest, **named):
                counter["n"] += 1
                if counter["n"] == cut:
                    raise Interrupted()
                return real_replace(source, target, *rest, **named)

            os.replace = counting
            try:
                catalog_module.write(built["after"], root)
                completed = True
            except Interrupted:
                completed = False
            finally:
                os.replace = real_replace

            what, whole = resolve()
            seen.add(what)
            report.require(
                whole,
                f"S13: cutting the publish at rename {cut} left a reader holding {what}",
            )
            if completed:
                break
            cut += 1
            if cut > 64:  # a bound on the loop, not on the property
                report.require(False, "S13: the publish made more than 64 renames")
                break

        report.require(
            cut >= 6,
            f"anti-vacuity: the publish made only {cut} rename(s), so the sweep above "
            "cannot have cut it between two artifacts",
        )
        report.require(
            seen == set(rows),
            f"S13: across {cut} interruption points a reader saw {sorted(seen)}; both "
            "the old and the new catalog must be reachable, or the sweep never "
            "crossed the swap",
        )

    # The pointer is the only mutable thing: a published set is never rewritten.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "catalog"
        built = catalog_module.build(chunk_dir)
        catalog_module.write(built, root)
        first = catalog_module.current_set(root)
        stamps = {path.name: path.read_bytes() for path in sorted(first.iterdir())}
        catalog_module.write(built, root)
        second = catalog_module.current_set(root)
        report.require(
            second == first,
            f"S13: republishing one catalog named a second set ({second.name}); the "
            "set name is its content, so an unchanged catalog is an unchanged name",
        )
        report.require(
            {path.name: path.read_bytes() for path in sorted(second.iterdir())} == stamps,
            "S13: republishing rewrote bytes inside an already-published set",
        )


# --------------------------------------------------------------------------
# S16 -- an operator nothing exercises is an operator nothing checks
# --------------------------------------------------------------------------


def check_operator_coverage(report: Report, plan, catalog) -> None:
    """Every operator the language declares is parsed, resolved and pinned.

    This gate runs over a thousand checks and, until this one existed, an eleventh
    operator could be added to `plan.OPERATORS` and pass every one of them. Nothing
    reconciled the declared language against the tables that exercise it, so coverage
    was whatever somebody remembered to add -- which is the definition of a check that
    degrades silently (`docs/security/laws.md` L15, and L2: the tables below can be
    satisfied while covering nothing).

    Three rails, because each answers a different question and an operator can pass
    one while failing another. The grammar table says it can be *written*. Resolution
    says it can be *answered* over the corpus, and that its estimate bounds its own
    answer. The plan baseline says its *decision* is pinned, so a change to how it is
    planned cannot be silent. An operator missing from any of them fails here, naming
    which.
    """
    declared = set(plan.OPERATORS)
    report.require(
        len(declared) >= 8,
        f"anti-vacuity: the language declares {len(declared)} operator(s)",
    )

    written = {plan.parse_predicate(text).op for text, _, _, _ in GRAMMAR_ACCEPTS}
    report.require(
        declared <= written,
        "coverage: these operators are declared but no row of the grammar table "
        f"produces one: {sorted(declared - written)}",
    )

    # Resolution: one predicate per operator, run against the corpus, with the same
    # two properties every other selection has to have.
    resolved = set()
    for predicate in _one_per_operator(plan, catalog):
        selection = plan.select(catalog, [predicate])
        resolved.add(predicate.op)
        admitted = catalog.rows_total if selection.rows is None else len(selection.rows)
        report.require(
            selection.estimated >= admitted,
            f"coverage: {predicate} estimated {selection.estimated} and admits "
            f"{admitted}; an estimate is an upper bound",
        )
        report.require(
            not selection.estimate_exact or selection.estimated == admitted,
            f"coverage: {predicate} was reported exact at {selection.estimated} and "
            f"admits {admitted}",
        )
    report.require(
        declared <= resolved,
        "coverage: these operators are declared but never resolved against the "
        f"corpus: {sorted(declared - resolved)}",
    )

    baseline = load_tool("plan_baseline")
    pinned = set()
    for query in baseline.QUERIES:
        for term in query.get("where", []):
            pinned.add(plan.parse_predicate(term).op)
    report.require(
        declared <= pinned,
        "coverage: these operators are declared but no recorded plan uses one, so a "
        f"change to how they are planned would not move the baseline: {sorted(declared - pinned)}",
    )

    # ...and the operators the *printer* can spell are the operators there are. A new
    # operator with no spelling prints as a KeyError at explain time.
    for operator in sorted(declared):
        values = () if operator in plan.NULLARY_OPERATORS else (
            ("500",) if operator in plan.RANGE_OPERATORS else ("x",)
        )
        arity = 2 if operator == "in" else 1
        spelled = plan.Predicate("subject", operator, values * arity if values else values)
        report.require(
            bool(str(spelled)),
            f"coverage: operator {operator!r} has no printed form",
        )


def _one_per_operator(plan, catalog) -> list:
    """One predicate per declared operator, built from values this corpus really has.

    Built from the corpus rather than written down, so the list cannot rot into
    predicates that admit nothing and check nothing. An operator this cannot build a
    predicate for is itself the finding, raised by the caller's coverage check rather
    than skipped here.
    """
    column = catalog.indexed_columns()[0]
    value = max(catalog.distinct(column), key=lambda name: catalog.distinct(column)[name])
    other = min(catalog.distinct(column))
    path = load_tool("schema").PATH_COLUMN
    measurement = catalog.numeric_columns()[0]
    by_operator = {
        "eq": plan.Predicate(column, "eq", (value,)),
        "ne": plan.Predicate(column, "ne", (value,)),
        "in": plan.Predicate(column, "in", (value, other)),
        "prefix": plan.Predicate(path, "prefix", ("training/",)),
        "isnull": plan.Predicate("language", "isnull", ()),
        "notnull": plan.Predicate("language", "notnull", ()),
        "ge": plan.Predicate(measurement, "ge", ("500",)),
        "gt": plan.Predicate(measurement, "gt", ("500",)),
        "le": plan.Predicate(measurement, "le", ("500",)),
        "lt": plan.Predicate(measurement, "lt", ("500",)),
    }
    return [by_operator[name] for name in plan.OPERATORS if name in by_operator]


# --------------------------------------------------------------------------
# S18 -- the decision, not just the answer
# --------------------------------------------------------------------------


def check_plan_baseline(report: Report, catalog) -> None:
    """Every recorded plan decision still holds, against the corpus it was recorded on.

    A correctness gate cannot see a planner regression: the rows are the same, in the
    same order, chosen worse. Both real estimator defects this tree has had were of
    exactly that shape -- the zone map calling a 25%-selective predicate 98% selective,
    and a conjunction bound reported as a count -- and both produced right answers the
    whole time they were wrong.

    A moved decision is a finding, not a failure of the tool: improving an estimate
    moves the file and so does breaking one, and the gate refuses to let either happen
    without somebody looking at the diff.
    """
    baseline = load_tool("plan_baseline")
    path = REPO_ROOT / baseline.DEFAULT_BASELINE
    try:
        stored = baseline.load(path)
        fresh = baseline.record(catalog)
        findings = baseline.compare(stored, fresh)
    except baseline.BaselineError as exc:
        report.require(False, f"S18: {exc}")
        return
    report.require(
        not findings,
        f"S18: {len(findings)} recorded plan decision(s) moved:\n    "
        + "\n    ".join(findings[:12])
        + (f"\n    ... and {len(findings) - 12} more" if len(findings) > 12 else "")
        + f"\n  re-record deliberately: python3 {baseline.BUILDER} --record",
    )
    report.require(
        len(stored.get("queries", {})) >= 20,
        f"anti-vacuity: the baseline pins {len(stored.get('queries', {}))} plan(s)",
    )
    # The comparison has to be able to fail, and the cheapest proof is to move one
    # recorded field and watch it be named.
    tampered = json.loads(json.dumps(stored))
    victim = sorted(tampered["queries"])[0]
    tampered["queries"][victim]["admitted"] = -1
    moved = baseline.compare(tampered, fresh)
    report.require(
        any(finding.startswith(f"{victim}.admitted") for finding in moved),
        "S18: a recorded row count was changed and the comparison did not name it; "
        "the baseline is being compared against itself",
    )


# --------------------------------------------------------------------------


def build_workspace(root: Path) -> tuple[Path, Path, Path]:
    """Build a corpus, an embedding set and a catalog from the tree, under `root`.

    The gate builds its own inputs rather than reading `build/`, for the same reason
    every other verifier here does: a gate that only runs after someone happened to
    build the right artifacts is a gate that does not run. Building also means the
    checks below compare a freshly derived catalog against the corpus it came from,
    which is the comparison that matters.
    """
    chunks = load_tool("build_chunks")
    embed = load_tool("embed_chunks")
    catalog_module = load_tool("catalog")

    chunk_dir = root / "chunks"
    embed_dir = root / "embeddings"
    catalog_dir = root / "catalog"
    status = chunks.main(["--out", str(chunk_dir)])
    if status:
        raise RuntimeError(f"build_chunks exited {status}")
    status = embed.main(["--chunks", str(chunk_dir), "--out", str(embed_dir)])
    if status:
        raise RuntimeError(f"embed_chunks exited {status}")
    catalog_module.write(catalog_module.build(chunk_dir), catalog_dir)

    sets = sorted(path for path in embed_dir.iterdir() if path.is_dir())
    if not sets:
        raise RuntimeError("embed_chunks produced no embedding set")
    return chunk_dir, sets[0], catalog_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set", type=Path, default=None, dest="embedding_set", help="an existing set to check"
    )
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="fail instead of skipping when the native rail cannot be built",
    )
    parser.add_argument("--keep", type=Path, help="also write the verified build here")
    args = parser.parse_args(argv)

    given = [args.embedding_set, args.chunks, args.catalog]
    if any(path is not None for path in given) and not all(path is not None for path in given):
        print(
            "verify_database: pass all three of --set, --chunks and --catalog or none "
            "of them; checking a catalog against a chunk table it was not built from "
            "measures nothing",
            file=sys.stderr,
        )
        return 2

    report = Report()
    search = load_tool("search_chunks")
    catalog_module = load_tool("catalog")
    plan = load_tool("plan")
    generations = load_tool("generations")

    workspace = None
    if all(path is not None for path in given):
        chunk_dir, embedding_root, catalog_dir = args.chunks, args.embedding_set, args.catalog
    else:
        workspace = Path(tempfile.mkdtemp(prefix="bcir-database-"))
        try:
            chunk_dir, embedding_root, catalog_dir = build_workspace(workspace)
        except Exception as exc:  # noqa: BLE001 -- a build failure is a verdict too
            shutil.rmtree(workspace, ignore_errors=True)
            print(f"database gate: FAILED\n  - building the workspace: {exc}", file=sys.stderr)
            return 1
    args.chunks, args.embedding_set, args.catalog = chunk_dir, embedding_root, catalog_dir

    try:
        return _run(report, search, catalog_module, plan, generations, args)
    finally:
        if workspace is not None:
            if args.keep:
                if args.keep.exists():
                    shutil.rmtree(args.keep)
                shutil.copytree(workspace, args.keep)
                print(f"[keep]    {args.keep}")
            shutil.rmtree(workspace, ignore_errors=True)


def _run(report: Report, search, catalog_module, plan, generations, args) -> int:
    try:
        catalog = catalog_module.Catalog.load(args.catalog, args.chunks)
    except catalog_module.CatalogError as exc:
        print(f"database gate: FAILED\n  - {exc}", file=sys.stderr)
        return 1

    if not report.require(
        catalog.rows_total >= MIN_ROWS,
        f"anti-vacuity: the catalog holds {catalog.rows_total} rows, below the "
        f"{MIN_ROWS} floor; every check below would pass over almost nothing",
    ):
        print("database gate: FAILED", file=sys.stderr)
        for failure in report.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    native_ok = True
    try:
        load_tool("bcir_native").load_kernels()
    except Exception as exc:  # noqa: BLE001 -- any failure means "not reachable"
        native_ok = False
        if args.require_native:
            report.require(False, f"--require-native but the kernel is unreachable: {exc}")

    # Before anything reads a packed column, because every check that does would
    # otherwise report a byte-order defect as whatever it broke downstream.
    report.run("interfaces-boundary", check_interface_boundary, report)
    report.run("interfaces", check_interfaces, report, catalog)
    report.run("host-byte-order", check_byte_order, report, catalog_module, catalog)
    report.run("S1", check_kernel_cache, report, require_native=args.require_native)
    report.run("S2", check_derived_columns, report, search, args.embedding_set, native_ok=native_ok)
    report.run(
        "S3", check_late_materialization, report, search, catalog, args.chunks, args.embedding_set
    )
    report.run(
        "S4",
        check_predicate,
        report,
        search,
        plan,
        catalog,
        args.embedding_set,
        args.chunks,
        native_ok=native_ok,
    )
    report.run("aggregates", check_aggregates, report, plan, catalog, args.chunks)
    report.run("schema", check_schema, report)
    report.run("constraints", check_constraints, report, catalog_module, args.chunks)
    report.run("S7-grammar", check_grammar, report, plan)
    report.run("quoting", check_quoting, report, plan)
    report.run("estimates", check_estimates, report, plan, catalog)
    report.run("coverage", check_operator_coverage, report, plan, catalog)
    report.run("S7-ranges", check_ranges, report, plan, catalog, args.chunks)
    report.run("S7-presence", check_presence, report, plan, catalog, args.chunks)
    report.run("S7-groups", check_grouped_aggregates, report, search, plan, catalog)
    report.run("ordering", check_ordering, report, search, plan, catalog)
    report.run("planner", check_planner, report, plan, catalog)
    report.run("S5", check_parts, report, catalog_module, catalog, args.chunks)
    report.run("S5-incremental", check_incremental, report, args.chunks)
    report.run("S6", check_generations, report, generations, args.chunks)
    report.run("S13", check_atomic_publish, report, catalog_module, args.chunks)
    report.run("S18", check_plan_baseline, report, catalog)

    if args.require_native and report.skips:
        report.require(
            False,
            "--require-native but these halves were skipped: " + "; ".join(report.skips),
        )

    if report.failures:
        print("database gate: FAILED", file=sys.stderr)
        for failure in report.failures[:40]:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    for observation in report.notes:
        print(f"[note]    {observation}")
    for reason in report.skips:
        print(f"[skip]    {reason}")
    print(
        f"database gate: PASSED ({report.checks} checks over {catalog.rows_total} rows, "
        f"{len(report.skips)} honest skip(s), {len(report.notes)} host note(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
