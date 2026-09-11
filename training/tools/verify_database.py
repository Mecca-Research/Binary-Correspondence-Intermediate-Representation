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
import random
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_SET = Path("build/training/embeddings/lexical-hash-v1")
DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_CATALOG = Path("build/training/catalog")

#: Anti-vacuity floors. Below any of these the run examined too little to mean
#: anything, and reporting PASSED would be a lie about coverage rather than a
#: verdict about correctness.
MIN_ROWS = 50
MIN_PREDICATE_TRIALS = 20
MIN_FETCH_ROWS = 50


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0
        self.skips: list[str] = []

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)

    def skip(self, reason: str) -> None:
        self.skips.append(reason)

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
    """Load a sibling tool once, reusing the module object.

    Executing the file again would return a NEW module with NEW class objects, so an
    `except module.SomeError` would compare an instance of one module's class against
    another's, match nothing, and let the exception escape.
    """
    cached = sys.modules.get(name)
    path = TOOLS_DIR / f"{name}.py"
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        NativeAIKernels.build(root, cc=compiler)
        library = root / NativeAIKernels._library_name()
        stamp = NativeAIKernels._stamp_path(library)
        report.require(stamp.is_file(), "S1: the first build published no stamp")
        first = library.stat()
        digest = library.read_bytes()

        NativeAIKernels.build(root, cc=compiler)
        second = library.stat()
        report.require(
            (first.st_ino, first.st_mtime_ns) == (second.st_ino, second.st_mtime_ns),
            "S1: a second build with nothing changed replaced the library -- the "
            "compiler ran when the stamp said it need not",
        )

        NativeAIKernels.build(root, cc=compiler, rebuild=True)
        third = library.stat()
        report.require(
            (first.st_ino, first.st_mtime_ns) != (third.st_ino, third.st_mtime_ns),
            "S1: rebuild=True did not replace the library, so the check above could "
            "not have failed either -- the observation is vacuous",
        )
        report.require(
            library.read_bytes() == digest,
            "S1: a forced rebuild of unchanged inputs produced different bytes; the "
            "cached library and the compiled one are not the same artifact",
        )

        recorded = json.loads(stamp.read_text(encoding="utf-8"))
        recorded["inputs"] = "0" * 64
        stamp.write_text(json.dumps(recorded), encoding="utf-8")
        NativeAIKernels.build(root, cc=compiler)
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
        locator_path = mirror / catalog_module.LOCATOR_FILE
        raw = bytearray(locator_path.read_bytes())
        width = catalog_module.LOCATOR_ENTRY_BYTES
        # Give row 0 row 1's span: a real staleness shape, not random bytes.
        raw[0:width] = raw[width : 2 * width]
        locator_path.write_bytes(bytes(raw))
        manifest_path = mirror / catalog_module.CATALOG_FILE
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
        (broken / catalog_module.LOCATOR_FILE).write_bytes(bytes(raw))
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
    truth_rows = 0
    for path in catalog_module.chunk_files(chunk_dir):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            truth_rows += 1
            truth_ids.append(record.get("chunk_id"))
            for column in truth:
                key = catalog_module.index_key(record.get(column))
                truth[column][key] = truth[column].get(key, 0) + 1
            source_path = record.get(catalog_module.PATH_COLUMN)
            if isinstance(source_path, str):
                for prefix in catalog_module.path_prefixes(source_path):
                    truth_prefixes[prefix] = truth_prefixes.get(prefix, 0) + 1

    report.require(
        catalog.ids == truth_ids,
        "S4: the catalog's row -> chunk_id list is not the corpus's row order; every "
        "fetch would return some other row's text",
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
        victim = sorted(mirror.glob("*.chunks.jsonl"))[0]
        victim.write_bytes(victim.read_bytes() + b'{"chunk_id":"sha256:zz","subject":"x"}\n')
        after_root = Path(directory) / "catalog2"
        catalog_module.write(catalog_module.build(mirror), after_root)
        after = catalog_module.Catalog.load(after_root, mirror)
        moved = after.changed_parts(before)
        expected = victim.name[: -len(".chunks.jsonl")]
        report.require(
            moved == (expected,),
            f"S5: changing {victim.name} reported {moved}, expected exactly "
            f"({expected!r},) -- a part digest that moves for unrelated files, or "
            "does not move for its own, cannot drive an incremental rebuild",
        )
        # And the stale catalog must refuse rather than answer from old statistics.
        try:
            catalog_module.Catalog.load(catalog_root, mirror, verify_content=True)
        except catalog_module.CatalogError:
            report.require(True, "")
        else:
            report.require(
                False,
                "S5: a catalog built before the chunk table moved still loaded; "
                "stale statistics produce a wrong plan silently",
            )


def check_generations(report: Report, generations, chunk_dir: Path) -> None:
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

        victim = sorted(mirror.glob("*.chunks.jsonl"))[0]
        victim.write_bytes(victim.read_bytes() + b'{"chunk_id":"sha256:new","subject":"x"}\n')
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
        stored = root / first.generation_id / "catalog" / "catalog.json"
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
        catalog = catalog_module.Catalog.load(args.catalog, args.chunks, verify_content=True)
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
    report.run("planner", check_planner, report, plan, catalog)
    report.run("S5", check_parts, report, catalog_module, catalog, args.chunks)
    report.run("S6", check_generations, report, generations, args.chunks)

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

    for reason in report.skips:
        print(f"[skip]    {reason}")
    print(
        f"database gate: PASSED ({report.checks} checks over {catalog.rows_total} rows, "
        f"{len(report.skips)} honest skip(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
