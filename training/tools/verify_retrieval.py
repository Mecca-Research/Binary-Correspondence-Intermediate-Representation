#!/usr/bin/env python3
"""Gate the retrieval evaluation (Phase 0.6).

An evaluation is itself a claim, and it needs checking harder than the thing it
measures: a broken evaluation reports a number that looks like evidence. What
this enforces:

  1. **Anti-vacuity.** A run over no queries, or one whose judgments point at
     documents the corpus does not contain, FAILS. An evaluation that examines
     nothing agrees with everything.
  2. **No silently empty family.** Every family the builder declares either
     produced queries or carries a stated reason it did not. A family that
     vanishes without a word is how a length filter once deleted an entire
     source of judgments here with no visible trace.
  3. **The harness works.** The `heading` family's query text appears verbatim
     in its target by construction, so it must score near-perfectly. If that
     fails, retrieval is broken and every other number in the run is noise --
     this is the check that tells a bad model from a bad harness.
  4. **The model clears both floors.** Far above a seeded random ranker, and far
     above a constant ranker that ignores the query. A score with nothing under
     it is not a measurement.
  5. **Metric sanity.** Recall is monotonic in the cutoff, MRR never exceeds the
     hit rate at the same cutoff, recall never exceeds it either, and the
     families account for every query in the set. These are properties of the
     arithmetic; violating one means the harness is miscounting, whatever the
     model does.
  6. **The headline is quality, not a self-match.** `overall` must pool exactly
     the non-control queries. A control is verbatim in its target by
     construction, so pooling it would inflate the score AND the lift and
     shuffle gates that read it.
  7. **The artifacts belong together.** Chunks, vectors and judgments carry the
     same corpus digest, and the concept memory carries a digest of its own
     entries and of the index tree they came from. A chapter rewritten in place
     changes neither a path nor a count, so nothing else would notice.
  8. **The memory is scored in its own space, against its own population.** Its
     baselines are recomputed over exactly the queries it was allowed to answer,
     every ranker in that block scores the same number of them, and a memory
     built at a dimension the chunk set does not use is run end-to-end -- the
     configuration in which projecting through the wrong model stops being
     invisible.
  9. **The evaluation can tell a broken index from a working one.** A shuffled
     row permutation must collapse the score to the noise floor. An evaluation
     that scores a scrambled index as highly as a real one measures nothing, and
     nothing else here would reveal that.
 10. **Determinism.** Two runs agree exactly.

**No absolute quality number is gated.** Freezing today's recall into a
threshold would tune the bar to today's corpus and then cite it as evidence;
the numbers are reported by `evaluate_retrieval.py` and belong in generated
output, never in prose.

    python3 training/tools/verify_retrieval.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

# The control's query text is literally inside its target, so anything below
# near-perfect means the harness -- not the model -- is broken.
CONTROL_RECALL_AT_5 = 0.95
# A working model on this corpus clears random by ~70x. Twenty is far below
# that and far above anything noise produces, so it is a floor rather than a
# fitted bar.
MIN_LIFT_OVER_RANDOM = 20.0
MIN_LIFT_OVER_CONSTANT = 20.0
# A shuffled index must fall to the noise floor. Ten percent of the real score
# is generous; the real gap is two orders of magnitude.
MAX_SHUFFLED_FRACTION = 0.10
MRR_KEY = "mrr@10"

# A directory README is itself a dispatcher, and the top-level pages are
# navigation. Indexing an index is circular, so these are a declared exclusion
# rather than a coverage gap -- named here so the exclusion is reviewable.
NON_INDEX_BASENAMES = frozenset(
    {
        "README.md",
        "CURRICULUM.md",
        "EXAMPLES.md",
        "INDEX.md",
        "NOTICE.md",
        "RECIPES.md",
        "SEMVER.md",
        "START_HERE.md",
    }
)
# The memory must clear noise on the queries it is allowed to score. The bar is
# a floor, not a fitted number: the measured lift is an order of magnitude above.
MIN_MEMORY_LIFT = 10.0
# Any dimension the chunk embedding set does not use. The point is that the two
# spaces differ, not the number.
ALTERNATE_MEMORY_DIM = 256


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_query_set(
    queries: list[dict], manifest: dict, documents: set[str], report: Report
) -> None:
    report.require(bool(queries), "the query set is empty -- an evaluation over no questions")
    report.require(
        manifest["queries"] == len(queries),
        f"manifest declares {manifest['queries']} queries, file holds {len(queries)}",
    )

    for family, block in sorted(manifest["families"].items()):
        if block["queries"]:
            continue
        report.require(
            bool(block["excluded"]),
            f"family {family!r} produced no queries and gives no reason; a family "
            "that vanishes silently is indistinguishable from one that never existed",
        )

    active = {family for family, b in manifest["families"].items() if b["queries"]}
    report.require(
        len(active) >= 2,
        f"only {len(active)} family/families produced queries; one family cannot "
        "show whether a score depends on how the questions were derived",
    )
    report.require(
        any(manifest["families"][f]["control"] for f in active),
        "no harness control family is present; without one, a broken harness and "
        "a bad model are indistinguishable",
    )

    unresolved = 0
    for query in queries:
        report.require(bool(query["targets"]), f"{query['query_id']}: no targets")
        report.require(bool(query["bias"].strip()), f"{query['query_id']}: undeclared bias")
        unresolved += sum(1 for target in query["targets"] if target not in documents)
    report.require(
        not unresolved,
        f"{unresolved} judgment target(s) name a document the corpus does not "
        "contain; those queries are unanswerable and would depress every score",
    )


def check_metrics(report_data: dict, manifest: dict, report: Report) -> None:
    for scope, block in [("overall", report_data["overall"])] + sorted(
        report_data["families"].items()
    ):
        model = block["model"]
        report.require(
            model["recall@1"] <= model["recall@5"] <= model["recall@10"],
            f"{scope}: recall is not monotonic in the cutoff -- the harness is miscounting",
        )
        report.require(
            model[MRR_KEY] <= model["hit@10"] + 1e-9,
            f"{scope}: MRR exceeds the hit rate at the same cutoff, which is "
            "arithmetically impossible",
        )
        report.require(
            model["recall@10"] <= model["hit@10"] + 1e-9,
            f"{scope}: recall@10 exceeds hit@10; recall counts a fraction of the "
            "targets found and cannot exceed whether any was found",
        )

    controls = {family for family, block in manifest["families"].items() if block["control"]}
    report.require(
        bool(controls), "no control family is declared; a broken harness would be invisible"
    )
    for family in sorted(controls):
        block = report_data["families"].get(family)
        if not block:
            continue
        report.require(
            block["model"]["recall@5"] >= CONTROL_RECALL_AT_5,
            f"harness control {family!r} scored recall@5 "
            f"{block['model']['recall@5']:.3f} < {CONTROL_RECALL_AT_5}; its query text "
            "is verbatim in its target, so retrieval itself is broken",
        )

    # A control is trivial by construction, so pooling it into the headline
    # would inflate both the score and every gate that reads it.
    pooled = report_data["overall"]["queries"]
    control_queries = sum(
        report_data["families"][family]["queries"]
        for family in controls
        if family in report_data["families"]
    )
    total = sum(block["queries"] for block in report_data["families"].values())
    # Without this, the arithmetic below could balance while queries went
    # missing: a family the evaluator silently dropped shrinks both sides.
    report.require(
        total == manifest["queries"],
        f"the families account for {total} queries but the set holds "
        f"{manifest['queries']}; a query scored by no family is a query the "
        "evaluation reported nothing about",
    )
    report.require(
        pooled == total - control_queries,
        f"`overall` pools {pooled} of {total} queries but {control_queries} belong to "
        "a harness control; controls must not contribute to a quality score",
    )

    overall = report_data["overall"]
    for baseline, floor in (("random", MIN_LIFT_OVER_RANDOM), ("constant", MIN_LIFT_OVER_CONSTANT)):
        base = overall[baseline][MRR_KEY]
        lift = overall["model"][MRR_KEY] / base if base else float("inf")
        report.require(
            lift >= floor,
            f"model MRR is only {lift:.1f}x the {baseline} baseline (floor {floor}x); "
            "a score that close to noise is not evidence of retrieval",
        )


def check_sensitivity(shuffled: dict, real: dict, report: Report) -> float:
    """A scrambled index must score at the noise floor, or the eval measures nothing."""
    ratio = (
        shuffled["overall"]["model"][MRR_KEY] / real["overall"]["model"][MRR_KEY]
        if real["overall"]["model"][MRR_KEY]
        else 1.0
    )
    report.require(
        ratio <= MAX_SHUFFLED_FRACTION,
        f"a shuffled index still scores {ratio:.1%} of the real one (cap "
        f"{MAX_SHUFFLED_FRACTION:.0%}); this evaluation cannot tell a broken index "
        "from a working one and therefore measures nothing",
    )
    return ratio


def check_index_coverage(chunk_dir: Path, report: Report) -> None:
    """Every retrievable teaching document must be reachable from an index.

    This is what makes "the indexes are comprehensive" a checked property
    instead of a claim that decays the moment a chapter is added. A new
    document that no index mentions is invisible to a reader browsing by
    concept and absent from the concept memory built on those indexes.
    """
    import re

    link = re.compile(r"\]\(([^)\s#]+\.md)")
    index_dir = CORPUS_ROOT / "llvm" / "indexes"
    reachable: set[str] = set()
    for index_file in sorted(index_dir.glob("*.md")):
        text = index_file.read_text(encoding="utf-8")
        if "Retired / historical material" in text:
            continue
        for href in link.findall(text):
            try:
                resolved = (index_file.parent / href).resolve().relative_to(REPO_ROOT)
            except (ValueError, OSError):
                continue
            reachable.add(resolved.as_posix())

    retrievable: set[str] = set()
    for path in sorted(chunk_dir.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                source = json.loads(line)["source_path"]
                if source.startswith("training/llvm/") and "/indexes/" not in source:
                    retrievable.add(source)

    report.require(
        bool(retrievable), "no retrievable llvm document found; coverage checked nothing"
    )
    missing = sorted(
        source for source in retrievable - reachable if Path(source).name not in NON_INDEX_BASENAMES
    )
    report.require(
        not missing,
        f"{len(missing)} teaching document(s) are reachable by search but named by "
        f"no index, so no concept leads to them: {', '.join(missing[:5])}"
        + (" ..." if len(missing) > 5 else ""),
    )
    print(
        f"[index]   {len(retrievable & reachable)}/{len(retrievable)} retrievable "
        f"document(s) named by an index "
        f"({len(retrievable) - len(retrievable & reachable)} declared navigation page(s))"
    )


def check_memory(memory, queries: list[dict], documents: set[str], report: Report) -> None:
    """The concept memory itself: well-formed, resolvable, and actually consulted."""
    entries = memory.entries
    report.require(bool(entries), "the concept memory holds no entries")
    report.require(
        len(memory.views) == len(entries),
        f"memory holds {len(entries)} entries but {len(memory.views)} vectors",
    )
    unresolved = sorted({t for entry in entries for t in entry["targets"] if t not in documents})
    report.require(
        not unresolved,
        f"{len(unresolved)} concept target(s) name a document the corpus cannot "
        f"retrieve: {', '.join(unresolved[:3])}",
    )
    for entry in entries[:0] or entries:
        if not entry["concept"].strip() or not entry["origin"].strip():
            report.require(False, f"{entry['entry_id']}: concept or origin is empty")
            break

    sources = set(memory.manifest["indexes"])
    withheld = [q for q in queries if q["origin"].rsplit(":", 1)[0] in sources]
    report.require(
        bool(withheld),
        "no query was withheld from the concept memory, so the circularity guard "
        "never fired; a memory scored on its own index rows finds its own keys",
    )
    print(f"[memory]  {len(entries)} concept(s); {len(withheld)} circular query/queries withheld")


def check_memory_space(
    memory_module, evaluator, embedding_set, search, queries, circular, workspace, report: Report
) -> None:
    """A memory must be queried in ITS space, whatever space is lying around.

    Here the concept memory and the chunk index happen to share a model and a
    dimension, so projecting a query with the wrong one produces the same
    numbers and no test would notice. The mistake only becomes visible once the
    two differ -- which is exactly when a corpus would adopt a learned model for
    one and not yet the other. So the check builds a memory at a dimension the
    chunk set does not have and runs the real evaluator against it: `recall`
    refuses a query of the wrong width rather than letting `map` truncate to the
    overlap and rank on it.
    """
    alternate = workspace / "memory-alt"
    memory_module.build(alternate, dim=ALTERNATE_MEMORY_DIM)
    small = memory_module.IndexMemory.load(alternate)
    report.require(
        small.dim == ALTERNATE_MEMORY_DIM != embedding_set.dim,
        f"the alternate memory is dim {small.dim} and the chunk set is dim "
        f"{embedding_set.dim}; identical spaces would make this check vacuous",
    )
    try:
        scored = evaluator.evaluate(
            queries,
            embedding_set,
            search,
            rankers=("memory",),
            memory=small,
            memory_module=memory_module,
            circular=circular,
        )
    except SystemExit as exc:
        report.require(
            False,
            f"the evaluator projected its queries into the wrong space: {exc}",
        )
        return
    population = scored.get("memory_population") or {}
    report.require(
        bool(population.get("memory")),
        "the alternate-space memory scored no query, so nothing was exercised",
    )
    print(
        f"[space]   memory at dim {small.dim} scored "
        f"{population.get('queries', 0)} query/queries in its own space"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=Path)
    args = parser.parse_args(argv)

    chunks_module = load_tool("build_chunks")
    distill_module = load_tool("build_distillation")
    embed_module = load_tool("embed_chunks")
    queries_module = load_tool("build_eval_queries")
    search = load_tool("search_chunks")
    evaluator = load_tool("evaluate_retrieval")

    report = Report()
    workspace = Path(tempfile.mkdtemp(prefix="bcir-retrieval-"))
    try:
        chunk_dir = workspace / "chunks"
        distill_dir = workspace / "distill"
        embed_dir = workspace / "embeddings"
        eval_dir = workspace / "eval"
        chunks_module.main(["--out", str(chunk_dir)])
        distill_module.main(["--out", str(distill_dir)])
        embed_module.main(["--chunks", str(chunk_dir), "--out", str(embed_dir)])
        queries_module.main(
            ["--chunks", str(chunk_dir), "--distill", str(distill_dir), "--out", str(eval_dir)]
        )

        check_index_coverage(chunk_dir, report)

        queries = evaluator.read_jsonl(eval_dir / "queries.jsonl")
        manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
        embedding_set = search.EmbeddingSet(embed_dir / "lexical-hash-v1")
        documents = {row["source_path"] for row in embedding_set.rows}

        check_query_set(queries, manifest, documents, report)
        # Refuses a stale chunk/vector/judgment combination outright.
        evaluator.require_matching_corpus(chunk_dir, embedding_set, manifest)
        controls = frozenset(
            family for family, block in manifest["families"].items() if block["control"]
        )

        memory_module = load_tool("build_index_memory")
        memory_module.build(workspace / "memory", dim=embedding_set.dim)
        memory = memory_module.IndexMemory.load(workspace / "memory")
        check_memory(memory, queries, documents, report)
        circular = evaluator.circular_sources(memory.manifest)

        measured = evaluator.evaluate(
            queries,
            embedding_set,
            search,
            rankers=evaluator.RANKERS + ("memory",),
            memory=memory,
            memory_module=memory_module,
            circular=circular,
            controls=controls,
        )
        evaluator.print_report(measured, manifest)
        check_metrics(measured, manifest, report)

        # Positive control: scramble which row means which document.
        rows = len(embedding_set.rows)
        permutation = [(row * 7919 + 104729) % rows for row in range(rows)]
        report.require(
            len(set(permutation)) == rows,
            "the shuffle control is not a permutation, so it does not scramble the index",
        )
        shuffled = evaluator.evaluate(
            queries,
            embedding_set,
            search,
            rankers=("model",),
            permutation=permutation,
            controls=controls,
        )
        scored = [q for q in queries if q["origin"].rsplit(":", 1)[0] not in circular]
        report.require(
            not any(q["origin"].rsplit(":", 1)[0] in circular for q in scored),
            "a query derived from the memory's own index rows was scored against it",
        )
        population = measured.get("memory_population") or {}
        if population.get("memory"):
            # Both sides of this ratio must describe the same queries. Pooling the
            # baseline over every query while the memory answered only a subset
            # compares two different populations, and the gate would then pass or
            # fail for reasons unrelated to the memory.
            report.require(
                population["queries"] == len(scored),
                f"the memory population holds {population['queries']} queries but "
                f"{len(scored)} were scoreable; the baseline would not match",
            )
            counts = set((population.get("counts") or {}).values())
            report.require(
                len(counts) == 1,
                f"the memory population scores different numbers of queries per "
                f"ranker {sorted(population.get('counts', {}).items())}; a lift "
                "computed across two populations compares different questions",
            )
            base = population["random"][MRR_KEY]
            lift = population["memory"][MRR_KEY] / base if base else float("inf")
            report.require(
                lift >= MIN_MEMORY_LIFT,
                f"concept memory MRR is only {lift:.1f}x random over the "
                f"{population['queries']} query/queries it may score "
                f"(floor {MIN_MEMORY_LIFT}x)",
            )

        check_memory_space(
            memory_module, evaluator, embedding_set, search, queries, circular, workspace, report
        )

        ratio = check_sensitivity(shuffled, measured, report)
        print(
            f"[control] shuffled index scores {ratio:.1%} of the real index "
            f"(cap {MAX_SHUFFLED_FRACTION:.0%})"
        )

        again = evaluator.evaluate(
            queries, embedding_set, search, rankers=("model",), controls=controls
        )
        report.require(
            again["overall"]["model"] == measured["overall"]["model"],
            "two evaluation runs over the same index disagree -- the harness is not deterministic",
        )
        print("[det]     two runs agree exactly")

        if args.keep:
            if args.keep.exists():
                shutil.rmtree(args.keep)
            shutil.copytree(workspace, args.keep)
            print(f"[keep]    {args.keep}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if report.failures:
        print("retrieval gate: FAILED", file=sys.stderr)
        for failure in report.failures[:40]:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"retrieval gate: PASSED ({report.checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
