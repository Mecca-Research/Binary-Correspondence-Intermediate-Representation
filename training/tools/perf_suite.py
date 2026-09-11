#!/usr/bin/env python3
"""What the training database costs, measured, and what part of it is a fact.

Every number here carries a **class**, and the class decides what it may be used
for (`docs/BCIR_MASTER_ROADMAP.md`, the measurement discipline):

    exact   a deterministic count -- modules imported, bytes read, rows scanned,
            parts pruned. The same on every host and every run. These MAY gate.
    ratio   one duration over another, measured in the same process in the same
            run. Cancels most host variation but not all; a 25% band.
    wall    an absolute duration. INDICATIVE ONLY. Never gates, on any host.

This host has no PMU (`perf_event_open` is unavailable in the container, see
`docs/BCIR_TARGET_ACCESS.md`), so there are no cycle or cache-miss rows and none
are faked. A `wall` row here is a shared-runner wall clock and is reported with
its spread so a reader can see how much of it is noise.

**Why the exact rows matter more than the times.** A latency number on a 4-core
shared container tells you about the container. What does not move between hosts
is *how much work the design commits to*: how many modules are imported before a
count is answered, how many bytes are read to answer it, how many rows are
touched. Those are properties of the code, they are reproducible, and every
optimization worth making moves one of them. The counts are not repeated in this
docstring, because a number written in prose beside the code that measures it is
a number that will disagree with it -- run the suite.

The wall times are here to say which of those counts is worth attacking. They
earned that job on the first run: the exact rows said a ranked query imports 123
modules and scans 2,215 rows, which is true of every backend, and only the clock
distinguished the backend the planner would pick from the one that was 6x slower
because its setup cost was mispriced by a factor of 37.

    python3 training/tools/perf_suite.py                  # the suite
    python3 training/tools/perf_suite.py --json-out p.json
    python3 training/tools/perf_suite.py --repeats 25     # tighter, slower
    python3 training/tools/perf_suite.py --only resolve   # one family

`--compare BASELINE` re-measures and prints what moved, exact rows first: those
are the ones where a difference is a fact rather than a fluctuation.

**A recorded baseline is host-local and is not tracked.** `--json-out` writes
under `build/`, which is generated rather than committed, because a stored `wall`
row is a statement about the machine that produced it: comparing this container's
milliseconds against a CI runner's would report a finding per row and not one of
them would be about the code. The comparison this file is for is *before and
after one change on one host*, which is the only comparison its numbers support.
The decisions that must hold across hosts are pinned somewhere else, by
`plan_baseline.py`, in exact terms that do not involve a clock.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from db import analytics, engine, relational, retrieval  # noqa: E402

SCHEMA = "bcir-training/perf-suite/v1"
BUILDER = "training/tools/perf_suite.py"

DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_CATALOG = Path("build/training/catalog")
DEFAULT_EMBEDDINGS = Path("build/training/embeddings")

#: Repeats for an in-process timing. The median of this many is reported with the
#: interquartile range beside it, because a median alone on a shared runner invites
#: a reader to believe a difference that is inside the noise.
DEFAULT_REPEATS = 15

#: Repeats for a cold-process timing. Fewer, because each pays a full interpreter
#: start; more than enough to see a 10 ms effect against a ~15 ms floor.
DEFAULT_COLD_REPEATS = 9


class Row:
    """One measured fact, with the class that says what it may be used for."""

    def __init__(self, family, name, value, unit, metric_class, note=""):
        self.family = family
        self.name = name
        self.value = value
        self.unit = unit
        self.metric_class = metric_class
        self.note = note

    def as_dict(self) -> dict:
        return {
            "family": self.family,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "class": self.metric_class,
            "note": self.note,
        }


def timed(function, repeats: int) -> tuple[float, float]:
    """Median and interquartile range of `repeats` calls, in milliseconds.

    The IQR travels with the median everywhere a duration is reported. A single
    number invites the reader to compare two runs that differ by less than the
    spread, which is the most common way a performance claim becomes false.
    """
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    samples.sort()
    if len(samples) >= 4:
        half = len(samples) // 2
        low = statistics.median(samples[:half])
        high = statistics.median(samples[-half:])
    else:
        low, high = samples[0], samples[-1]
    return statistics.median(samples), high - low


def cold(args: list[str], repeats: int) -> tuple[float, float]:
    """A whole process, start to exit. The number a caller actually waits for."""

    def run():
        result = subprocess.run([sys.executable, *args], capture_output=True, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise RuntimeError(
                f"{' '.join(args)} exited {result.returncode}: "
                f"{result.stderr.decode('utf-8', 'replace')[-400:]}"
            )

    return timed(run, repeats)


def imported_modules(args: list[str]) -> tuple[int, float, dict]:
    """How many modules a command imports, and what the costly ones are.

    The count is `exact`: it is a property of the import graph, identical on every
    host. The times beside it are the interpreter's own accounting under
    `-X importtime`, which inflates absolute values and is used here only to rank.
    """
    result = subprocess.run(
        [sys.executable, "-X", "importtime", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    self_us: dict[str, int] = {}
    for line in result.stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        parts = line[len("import time:") :].split("|")
        if len(parts) != 3 or "self" in parts[0]:
            continue
        try:
            self_us[parts[2].strip()] = self_us.get(parts[2].strip(), 0) + int(parts[0].strip())
        except ValueError:
            continue
    return len(self_us), sum(self_us.values()) / 1000.0, self_us


def artifact_bytes(catalog) -> dict[str, int]:
    """How many bytes each artifact holds. Exact, and the ceiling on what a read costs."""
    published = engine.catalog().current_set(catalog.root)
    return {path.name: path.stat().st_size for path in sorted(published.iterdir())}


# --------------------------------------------------------------------------
# The families
# --------------------------------------------------------------------------


def family_startup(rows, options):
    """The floor: what a command costs before it has done anything at all."""
    floor, spread = cold(["-c", "pass"], options.cold_repeats)
    rows.append(Row("startup", "bare interpreter", floor, "ms", "wall", f"IQR {spread:.1f}"))

    for label, args in (
        (
            "relational --count",
            ["training/tools/search_chunks.py", "--count", "--where", "kind=code"],
        ),
        (
            "analytic --stats",
            ["training/tools/search_chunks.py", "--count", "--stats", "char_count"],
        ),
        (
            "grouped --group-by",
            [
                "training/tools/search_chunks.py",
                "--count",
                "--group-by",
                "subject",
                "--stats",
                "char_count",
            ],
        ),
        ("ranked --query", ["training/tools/search_chunks.py", "--query", "jit", "--top-k", "5"]),
    ):
        total, spread = cold(args, options.cold_repeats)
        rows.append(Row("startup", f"cold {label}", total, "ms", "wall", f"IQR {spread:.1f}"))
        rows.append(
            Row(
                "startup",
                f"cold {label} over floor",
                total - floor,
                "ms",
                "wall",
                f"{100 * floor / total:.0f}% of it is interpreter start",
            )
        )

    count, import_ms, self_us = imported_modules(
        ["training/tools/search_chunks.py", "--count", "--where", "kind=code"]
    )
    rows.append(Row("startup", "modules imported for a count", count, "modules", "exact"))
    rows.append(
        Row(
            "startup",
            "import self-time (importtime, inflated)",
            import_ms,
            "ms",
            "wall",
            "ranks modules; absolute value not comparable to the cold rows",
        )
    )
    for name, micros in sorted(self_us.items(), key=lambda kv: -kv[1])[:5]:
        rows.append(Row("startup", f"  import {name}", micros / 1000.0, "ms", "wall", "self time"))


def family_catalog(rows, options, catalog, chunk_dir):
    """Opening the database: the freshness digest and the manifest parse."""
    catalog_module = engine.catalog()

    load, spread = timed(
        lambda: catalog_module.Catalog.load(catalog.root, chunk_dir), options.repeats
    )
    rows.append(Row("catalog", "Catalog.load", load, "ms", "wall", f"IQR {spread:.2f}"))

    files = catalog_module.chunk_files(chunk_dir)
    corpus = sum(path.stat().st_size for path in files)
    rows.append(Row("catalog", "chunk files digested on every load", len(files), "files", "exact"))
    rows.append(Row("catalog", "corpus bytes digested on every load", corpus, "bytes", "exact"))

    published = engine.catalog().current_set(catalog.root)
    manifest = published / catalog_module.CATALOG_FILE
    parse, spread = timed(lambda: json.loads(manifest.read_text(encoding="utf-8")), options.repeats)
    rows.append(Row("catalog", "manifest parse alone", parse, "ms", "wall", f"IQR {spread:.2f}"))
    rows.append(
        Row(
            "catalog",
            "digest share of load",
            load / max(load, 1e-9) if False else (load - parse) / load,
            "fraction",
            "ratio",
            "the rest is the manifest parse",
        )
    )

    for name, size in artifact_bytes(catalog).items():
        rows.append(Row("catalog", f"artifact {name}", size, "bytes", "exact"))

    # Lazily loaded artifacts: what each costs the first time something asks.
    for label, pull in (
        ("postings", lambda c: c.postings),
        ("ids", lambda c: c.ids),
        ("locator", lambda c: c.locator),
        ("numeric", lambda c: c.numeric),
        ("order", lambda c: c.order),
    ):

        def once(pull=pull):
            fresh = catalog_module.Catalog.load(catalog.root, chunk_dir)
            pull(fresh)

        warm = catalog_module.Catalog.load(catalog.root, chunk_dir)
        pull(warm)  # prove it resolves before timing it
        first, spread = timed(once, max(3, options.repeats // 3))
        rows.append(
            Row("catalog", f"load + first {label}", first, "ms", "wall", f"IQR {spread:.2f}")
        )


def family_resolve(rows, options, catalog):
    """Turning a predicate into rows: the planner's own work, artifacts warm."""
    plan = engine.planner()
    cases = (
        ("eq indexed", ["subject=llvm"]),
        ("eq rare", ["subject=data"]),
        ("eq absent value", ["subject=nothing-has-this"]),
        ("ne indexed", ["language!=llvm"]),
        ("in set", ["subject=llvm,data"]),
        ("pair covered by joint stats", ["subject=llvm", "kind=code"]),
        ("pair empty", ["subject=data", "kind=code"]),
        ("triple indexed", ["subject=llvm", "kind=code", "language=llvm"]),
        ("prefix counted", ["source_path^=training/llvm"]),
        ("prefix uncounted", ["source_path^=training/llvm/12-backend"]),
        ("range open", ["char_count>=500"]),
        ("range interval", ["char_count>=500", "char_count<=2000"]),
        ("range narrow", ["char_count>=9000"]),
        ("presence", ["language=?"]),
        ("absence", ["language!=?"]),
    )
    for label, terms in cases:
        selection = relational.resolve_selection(catalog, terms)
        admitted = catalog.rows_total if selection.rows is None else len(selection.rows)
        elapsed, spread = timed(
            lambda terms=terms: relational.resolve_selection(catalog, terms), options.repeats
        )
        rows.append(Row("resolve", f"{label}", elapsed, "ms", "wall", f"IQR {spread:.3f}"))
        rows.append(Row("resolve", f"{label}: rows admitted", admitted, "rows", "exact"))
        rows.append(
            Row(
                "resolve",
                f"{label}: estimate exact",
                int(selection.estimate_exact),
                "bool",
                "exact",
            )
        )
        if admitted:
            rows.append(
                Row(
                    "resolve",
                    f"{label}: us per admitted row",
                    1000 * elapsed / admitted,
                    "us/row",
                    "wall",
                )
            )

    # Range pruning is the zone map's whole claim; count what it skipped.
    for label, low, high in (("range open", 500, None), ("range narrow", 9000, None)):
        pruned = sum(
            1 for part in catalog.parts if _pruned(catalog.zone(part, "char_count"), low, high)
        )
        rows.append(Row("resolve", f"{label}: parts pruned", pruned, "parts", "exact"))
        rows.append(Row("resolve", f"{label}: parts total", len(catalog.parts), "parts", "exact"))


def _pruned(zone, low, high) -> bool:
    if zone.get("count", 0) == 0:
        return True
    if low is not None and zone["max"] < low:
        return True
    if high is not None and zone["min"] > high:
        return True
    return False


def family_aggregate(rows, options, catalog):
    """Aggregates and grouping: answered from the index, or by gathering the column."""
    unfiltered = relational.resolve_selection(catalog, [])
    narrow = relational.resolve_selection(catalog, ["subject=data"])
    wide = relational.resolve_selection(catalog, ["subject=llvm"])

    for label, selection in (("whole table", unfiltered), ("wide", wide), ("narrow", narrow)):
        admitted = catalog.rows_total if selection.rows is None else len(selection.rows)
        elapsed, spread = timed(
            lambda s=selection: catalog.aggregate("char_count", s.rows), options.repeats
        )
        rows.append(
            Row("aggregate", f"MIN/MAX/SUM {label}", elapsed, "ms", "wall", f"IQR {spread:.3f}")
        )
        rows.append(Row("aggregate", f"MIN/MAX/SUM {label}: rows", admitted, "rows", "exact"))

    for label, selection in (("unfiltered", unfiltered), ("filtered", wide)):
        for column in ("subject", "kind", "language"):
            elapsed, spread = timed(
                lambda c=column, s=selection: analytics.group_counts(catalog, c, s), options.repeats
            )
            groups, scanned = analytics.group_counts(catalog, column, selection)
            rows.append(
                Row(
                    "aggregate",
                    f"GROUP BY {column} {label}",
                    elapsed,
                    "ms",
                    "wall",
                    "scanned postings" if scanned else "from statistics",
                )
            )
            rows.append(
                Row(
                    "aggregate",
                    f"GROUP BY {column} {label}: groups",
                    len(groups),
                    "groups",
                    "exact",
                )
            )
            rows.append(
                Row(
                    "aggregate",
                    f"GROUP BY {column} {label}: read postings",
                    int(scanned),
                    "bool",
                    "exact",
                )
            )


def family_order(rows, options, catalog):
    """ORDER BY: a slice of the sorted index, or a sort of what a predicate admitted."""
    for label, terms in (
        ("unfiltered", []),
        ("wide", ["subject=llvm"]),
        ("narrow", ["subject=data"]),
    ):
        selection = relational.resolve_selection(catalog, terms)
        strategy = relational.order_strategy(catalog, selection, "char_count")
        admitted = catalog.rows_total if selection.rows is None else len(selection.rows)
        elapsed, spread = timed(
            lambda s=selection: relational.ordered_rows(catalog, s, "char_count", True),
            options.repeats,
        )
        rows.append(
            Row(
                "order",
                f"ORDER BY char_count {label}",
                elapsed,
                "ms",
                "wall",
                f"strategy={strategy}, IQR {spread:.3f}",
            )
        )
        rows.append(Row("order", f"ORDER BY char_count {label}: rows", admitted, "rows", "exact"))
        rows.append(
            Row("order", f"ORDER BY char_count {label}: strategy", strategy, "name", "exact")
        )


def family_fetch(rows, options, catalog):
    """Late materialization: reading the text of k rows by seeking their spans."""
    every = relational.resolve_selection(catalog, [])
    candidates = list(range(catalog.rows_total))
    for k in (1, 5, 25, 100, 500):
        picked = candidates[:: max(1, catalog.rows_total // k)][:k]
        elapsed, spread = timed(lambda p=picked: retrieval.records(catalog, p), options.repeats)
        rows.append(Row("fetch", f"fetch {k} rows", elapsed, "ms", "wall", f"IQR {spread:.3f}"))
        rows.append(Row("fetch", f"fetch {k} rows: rows", len(picked), "rows", "exact"))
        if picked:
            rows.append(
                Row(
                    "fetch",
                    f"fetch {k} rows: us per row",
                    1000 * elapsed / len(picked),
                    "us/row",
                    "wall",
                )
            )
    del every

    # The ceiling a seek is measured against: what it costs to get the same records
    # without the locator. Both halves are reported, because only one of them is the
    # comparison. Reading the bytes is not the alternative to a seek -- a caller wants
    # *records*, and the whole-corpus path has to parse every line to produce one. An
    # earlier revision of this suite timed `read_bytes` alone, which made the seek path
    # look like it lost to a 0.58 ms scan; the honest comparison is against parsing.
    catalog_module = engine.catalog()
    files = catalog_module.chunk_files(catalog.chunk_dir)
    repeats = max(3, options.repeats // 3)
    raw, raw_spread = timed(lambda: sum(len(path.read_bytes()) for path in files), repeats)
    rows.append(
        Row(
            "fetch",
            "whole corpus: bytes only",
            raw,
            "ms",
            "wall",
            f"IQR {raw_spread:.2f} -- NOT the seek alternative, no record is produced",
        )
    )
    parsed, parsed_spread = timed(
        lambda: sum(
            1
            for path in files
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)
        ),
        repeats,
    )
    rows.append(
        Row(
            "fetch",
            "whole corpus: records parsed",
            parsed,
            "ms",
            "wall",
            f"IQR {parsed_spread:.2f} -- the ceiling a seek is measured against",
        )
    )


def family_build(rows, options, chunk_dir):
    """Rebuilding the catalog: the price of a corpus change."""
    catalog_module = engine.catalog()
    built = catalog_module.build(chunk_dir)
    elapsed, spread = timed(lambda: catalog_module.build(chunk_dir), max(3, options.repeats // 3))
    rows.append(Row("build", "catalog.build", elapsed, "ms", "wall", f"IQR {spread:.2f}"))
    rows.append(Row("build", "rows built", built[0]["rows_total"], "rows", "exact"))
    rows.append(Row("build", "parts built", len(built[0]["parts"]), "parts", "exact"))
    rows.append(
        Row("build", "us per row", 1000 * elapsed / built[0]["rows_total"], "us/row", "wall")
    )

    manifest = json.dumps(built[0], sort_keys=True, separators=(",", ":")).encode("utf-8")
    rows.append(Row("build", "manifest bytes", len(manifest), "bytes", "exact"))
    pairs = built[0]["statistics"].get("pairs", {})
    cells = sum(len(inner) for table in pairs.values() for inner in table.values())
    rows.append(Row("build", "joint statistic cells", cells, "cells", "exact"))


def family_memory(rows, options, catalog, chunk_dir):
    """What the database holds in memory once every artifact has been touched.

    `tracemalloc`'s own accounting of bytes this code asked for, rather than a
    resident-set reading that would include whatever the allocator happened to keep.

    **These two rows are `ratio`, not `exact`, and the distinction was earned.** They
    read like counts -- the interpreter is counting, and it counts the same objects
    every time -- so they were written as `exact`, which `compare` holds to equality.
    They are not equal. Measured back to back in one process they move by about 2.5 kB
    run to run, and the same measurement taken after the other seven families have run
    lands about 19 kB lower than it does under `--only memory`, because by then the
    allocations this region would have made have already been made and freed elsewhere.
    The number is a true statement about one process at one moment; it is not a
    property of the artifacts, and holding it to equality would have produced a finding
    on every run until somebody stopped reading the findings.

    What *is* exact about the artifacts is how many bytes they occupy on disk, so that
    is reported here too -- deterministic, and the denominator the ratio is worth
    reading against.
    """
    catalog_module = engine.catalog()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    fresh = catalog_module.Catalog.load(catalog.root, chunk_dir)
    # Every artifact is lazy, so naming them is what loads them. The references are
    # held in one tuple and inspected *after* the measurement stops: a dict of their
    # sizes built inside the traced region would be charging the report for the
    # report's own bookkeeping, which is a small lie of exactly the kind an `exact`
    # row is not allowed to tell.
    loaded = (fresh.postings, fresh.ids, fresh.locator, fresh.numeric, fresh.order)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # ...and they are inspected rather than discarded, because a property that
    # silently returned nothing would make this family report a small number and
    # present it as good news.
    names = ("postings", "ids", "locator", "numeric", "order")
    sizes = (len(loaded[0]["columns"]), *(len(part) for part in loaded[1:]))
    empty = sorted(name for name, size in zip(names, sizes) if not size)
    if empty:
        raise SystemExit(
            f"perf_suite: {', '.join(empty)} loaded empty; the memory "
            "figures below would be about a catalog that holds nothing"
        )
    rows.append(
        Row(
            "memory",
            "every artifact resident",
            current - before,
            "bytes",
            "ratio",
            "one process at one moment, +/- a few kB; not a property of the artifacts",
        )
    )
    rows.append(Row("memory", "peak while loading", peak - before, "bytes", "ratio"))
    # The same accessor the catalog family reports per artifact, summed -- rather than
    # a directory walk, which would have to know that `root` holds a pointer and the
    # artifacts live in the set it points at.
    rows.append(
        Row(
            "memory",
            "every artifact on disk",
            sum(artifact_bytes(catalog).values()),
            "bytes",
            "exact",
        )
    )

    corpus = sum(path.stat().st_size for path in catalog_module.chunk_files(chunk_dir))
    rows.append(
        Row(
            "memory",
            "resident over corpus",
            (current - before) / corpus,
            "fraction",
            "ratio",
            "the index is this fraction of the text it indexes",
        )
    )


# --------------------------------------------------------------------------


FAMILIES = {
    "startup": None,
    "catalog": None,
    "resolve": None,
    "aggregate": None,
    "order": None,
    "fetch": None,
    "build": None,
    "memory": None,
}


def measure(options) -> dict:
    catalog = engine.open_catalog(options.catalog, options.chunks)
    rows: list[Row] = []
    wanted = options.only or list(FAMILIES)

    if "startup" in wanted:
        family_startup(rows, options)
    if "catalog" in wanted:
        family_catalog(rows, options, catalog, options.chunks)
    if "resolve" in wanted:
        family_resolve(rows, options, catalog)
    if "aggregate" in wanted:
        family_aggregate(rows, options, catalog)
    if "order" in wanted:
        family_order(rows, options, catalog)
    if "fetch" in wanted:
        family_fetch(rows, options, catalog)
    if "build" in wanted:
        family_build(rows, options, options.chunks)
    if "memory" in wanted:
        family_memory(rows, options, catalog, options.chunks)

    return {
        "schema": SCHEMA,
        "builder": BUILDER,
        "corpus": catalog.manifest["fingerprint"],
        "rows_total": catalog.rows_total,
        "host": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cpus": os.cpu_count(),
            "pmu": False,
            "note": "no PMU in this container; every duration is a shared-runner wall clock",
        },
        "repeats": {"in_process": options.repeats, "cold": options.cold_repeats},
        "rows": [row.as_dict() for row in rows],
    }


def render(report: dict) -> str:
    out = []
    host = report["host"]
    out.append(
        f"perf suite: {report['rows_total']} rows, Python {host['python']}, "
        f"{host['cpus']} CPU(s), no PMU"
    )
    out.append(
        f"  in-process repeats {report['repeats']['in_process']}, "
        f"cold repeats {report['repeats']['cold']}; medians with IQR"
    )
    out.append("")
    family = None
    for row in report["rows"]:
        if row["family"] != family:
            family = row["family"]
            out.append(f"  [{family}]")
        value = row["value"]
        if isinstance(value, float):
            shown = f"{value:12.3f}" if abs(value) < 1000 else f"{value:12.1f}"
        else:
            shown = f"{value:>12}"
        note = f"   {row['note']}" if row["note"] else ""
        out.append(f"    {row['name']:<48} {shown} {row['unit']:<9} [{row['class']}]{note}")
    return "\n".join(out)


def compare(stored: dict, fresh: dict) -> list[str]:
    """What moved. Exact rows first: a difference there is a fact, not a fluctuation."""
    if stored.get("schema") != SCHEMA:
        return [f"baseline is not {SCHEMA}"]
    was = {(r["family"], r["name"]): r for r in stored["rows"]}
    now = {(r["family"], r["name"]): r for r in fresh["rows"]}
    exact_moves, timing_moves = [], []
    for key in sorted(set(was) | set(now)):
        left, right = was.get(key), now.get(key)
        label = f"{key[0]}/{key[1]}"
        if left is None:
            exact_moves.append(f"  + {label}: new row")
            continue
        if right is None:
            exact_moves.append(f"  - {label}: gone")
            continue
        if left["class"] == "exact":
            if left["value"] != right["value"]:
                exact_moves.append(f"  ! {label}: {left['value']} -> {right['value']} [exact]")
            continue
        a, b = left["value"], right["value"]
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not a:
            continue
        change = (b - a) / a
        if abs(change) >= 0.25:
            timing_moves.append(
                f"    {label}: {a:.3f} -> {b:.3f} {left['unit']} ({change:+.0%}) [{left['class']}]"
            )
    return (
        exact_moves
        + (["  indicative rows outside the 25% band:"] if timing_moves else [])
        + timing_moves
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cold-repeats", type=int, default=DEFAULT_COLD_REPEATS)
    parser.add_argument("--only", nargs="*", choices=sorted(FAMILIES), default=None)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--compare", type=Path, help="a stored report to diff against")
    options = parser.parse_args(argv)

    if options.repeats < 3 or options.cold_repeats < 3:
        print("perf_suite: a median of fewer than 3 samples is not a median", file=sys.stderr)
        return 1

    try:
        report = measure(options)
    except (engine.catalog().CatalogError, engine.planner().PlanError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(render(report))
    if options.json_out:
        options.json_out.parent.mkdir(parents=True, exist_ok=True)
        options.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"\n[write] {options.json_out}")
    if options.compare:
        stored = json.loads(options.compare.read_text(encoding="utf-8"))
        moves = compare(stored, report)
        print("\n" + ("\n".join(moves) if moves else "  nothing moved outside its band"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
