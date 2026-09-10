#!/usr/bin/env python3
"""Measure whether the vectors return the right chunk for a real question (0.6).

Phase 0.5 proved the vectors are attributed, aligned, non-degenerate, and
searched correctly. That is a statement about the index, not about retrieval:
an index can be perfectly well-formed and still rank the wrong document first.
This measures the difference, against the judged set from
`build_eval_queries.py`.

**Judgments are document-level, so ranking is too.** A query's answer is "these
chapters cover it", not "this chunk". The chunk ranking is therefore collapsed
to its documents in order -- first appearance wins -- and the metrics run over
that. Scoring at chunk level instead would reward a model for filling the top
ten with ten pieces of one correct chapter.

**Two baselines, because a number with nothing under it says nothing.**
`random` is the floor: a seeded shuffle per query, which is what "no signal at
all" scores on this corpus. `constant` returns the same documents for every
query, which is what a broken index that ignores the query scores. A model that
does not clear both by a wide margin has not been shown to retrieve.

**Absolute scores are reported, never gated.** Freezing "recall@10 >= 0.71" into
a check would be tuning the bar to today's corpus and calling the result
evidence; the gate in `verify_retrieval.py` asserts properties instead.

    python3 training/tools/evaluate_retrieval.py
    python3 training/tools/evaluate_retrieval.py --json-out build/training/eval/report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

DEFAULT_SET = Path("build/training/embeddings/lexical-hash-v1")
DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_EVAL = Path("build/training/eval")

CUTOFFS = (1, 5, 10)
MRR_AT = 10
# Enough chunks to yield ten distinct documents even when one chapter dominates
# the head of the ranking.
CHUNK_DEPTH = 64

RANKERS = ("model", "random", "constant")


def circular_sources(memory_manifest: dict) -> frozenset[str]:
    """The files a concept memory is built from.

    A query derived from one of these cannot score that memory: it would be
    asking the memory to find its own keys, and it would succeed while
    demonstrating nothing.

    The exclusion is **per query, not per family**. Judging a whole family
    circular because some of its queries came from an index file throws away
    every query in it that did not -- which is what a first cut of this did to
    the `heading` control, disqualifying the whole family over a handful of
    queries. The overlap is read from the memory's own declared sources, so a
    new index file cannot quietly reintroduce the circle.
    """
    return frozenset(memory_manifest["indexes"])


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def require_matching_corpus(chunk_dir: Path, embedding_set, query_manifest: dict) -> str:
    """Refuse to score a stale combination of artifacts.

    Chunks, vectors and judgments are three files built at three moments. If the
    corpus changed after any of them, the run still produces a plausible number
    and attributes it to the corpus as it stands today. Source paths and counts
    do not catch that -- a chapter rewritten in place leaves both untouched --
    so all three are bound to the same content digest.
    """
    builder = load_tool("build_eval_queries")
    if not chunk_dir.is_dir():
        raise SystemExit(
            f"evaluate_retrieval: no chunk build at {chunk_dir}; the digest that "
            "binds these artifacts together cannot be recomputed"
        )
    actual = builder.corpus_digest(chunk_dir)
    embedded = embedding_set.manifest["built_from"]["corpus_digest"]
    judged = query_manifest.get("corpus_sha256")
    if embedded != actual:
        raise SystemExit(
            "evaluate_retrieval: the embedding set was built from a different "
            f"corpus than {chunk_dir}\n  vectors: {embedded}\n  chunks:  {actual}\n"
            "  rebuild the set: python3 training/tools/embed_chunks.py"
        )
    if judged != actual:
        raise SystemExit(
            "evaluate_retrieval: the query set was judged against a different "
            f"corpus than {chunk_dir}\n  queries: {judged}\n  chunks:  {actual}\n"
            "  rebuild the set: python3 training/tools/build_eval_queries.py"
        )
    return actual


def documents_of(chunk_ranking, rows, limit: int) -> list[str]:
    """Collapse a chunk ranking to a document ranking, first appearance wins."""
    ordered: list[str] = []
    for row, _score in chunk_ranking:
        document = rows[row]["source_path"]
        if document not in ordered:
            ordered.append(document)
            if len(ordered) >= limit:
                break
    return ordered


def score(ranked: list[str], targets: set[str]) -> dict:
    """Recall at each cutoff, plus the reciprocal rank of the first hit.

    Recall is the fraction of the judgment's targets that appear in the top k,
    NOT whether any one of them did. 119 judgments here name more than one
    document and some name fifteen; scoring those as a hit the moment a single
    target appears would report coverage the retrieval never achieved. For a
    single-target judgment the two coincide, which is exactly why the difference
    is easy to miss and worth stating.
    """
    result = {f"recall@{k}": len(targets & set(ranked[:k])) / len(targets) for k in CUTOFFS}
    # Kept alongside recall because the two answer different questions: "did it
    # find anything relevant" versus "how much of what was relevant".
    result["hit@10"] = float(bool(targets & set(ranked[:MRR_AT])))
    reciprocal = 0.0
    for position, document in enumerate(ranked[:MRR_AT], start=1):
        if document in targets:
            reciprocal = 1.0 / position
            break
    result[f"mrr@{MRR_AT}"] = reciprocal
    return result


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {}
    keys = scores[0]
    return {key: sum(s[key] for s in scores) / len(scores) for key in keys}


def evaluate(
    queries: list[dict],
    embedding_set,
    search,
    *,
    rankers: tuple[str, ...] = RANKERS,
    permutation: list[int] | None = None,
    memory=None,
    memory_module=None,
    circular: frozenset[str] = frozenset(),
    controls: frozenset[str] = frozenset(),
) -> dict:
    """`circular` holds source files whose derived queries may not score `memory`."""
    """Run every ranker over every query.

    `permutation` remaps chunk rows before they are read as documents. The gate
    uses it as a positive control: a shuffled index must score materially worse,
    because an evaluation that cannot tell a broken index from a working one is
    not measuring retrieval.
    """
    rows = embedding_set.rows
    documents = sorted({row["source_path"] for row in rows})
    per_family: dict[str, dict[str, list[dict]]] = {}
    # Scores for exactly the queries the memory was allowed to answer, so its
    # lift is measured against the same population it competed on.
    memory_population: dict[str, list[dict]] = {name: [] for name in rankers}

    for query in queries:
        targets = set(query["targets"])
        family = per_family.setdefault(query["family"], {name: [] for name in rankers})

        if "model" in rankers:
            projected = search.embed_query(query["query"], embedding_set)
            ranking = search.topk_reference(projected, embedding_set, CHUNK_DEPTH)
            if permutation is not None:
                ranking = [(permutation[row], dist) for row, dist in ranking]
            family["model"].append(score(documents_of(ranking, rows, MRR_AT), targets))

        if "random" in rankers:
            # Seeded from the query's own id: no run-to-run drift, and no shared
            # stream that would correlate one query's draw with the next.
            rng = random.Random(int(query["query_id"][7:23], 16))
            shuffled = list(documents)
            rng.shuffle(shuffled)
            family["random"].append(score(shuffled[:MRR_AT], targets))

        if "constant" in rankers:
            family["constant"].append(score(documents[:MRR_AT], targets))

        if "memory" in rankers and query["origin"].rsplit(":", 1)[0] not in circular:
            projected = memory_module.project_query(query["query"], memory)
            hits = memory.recall(projected, concepts=MRR_AT)
            ranked = memory_module.documents_from(hits, MRR_AT)
            memory_score = score(ranked, targets)
            family["memory"].append(memory_score)
            memory_population["memory"].append(memory_score)
            for name in ("model", "random", "constant"):
                if name in rankers and family[name]:
                    memory_population[name].append(family[name][-1])

    excluded = sum(1 for q in queries if q["origin"].rsplit(":", 1)[0] in circular)
    report: dict = {
        "families": {},
        "overall": {},
        "controls": {},
        "memory_population": {},
        "circular_sources": sorted(circular),
        "circular_queries": excluded,
    }
    for family, by_ranker in sorted(per_family.items()):
        block = {
            "queries": len(by_ranker["model"]) if by_ranker.get("model") else 0,
            "control": family in controls,
            **{name: aggregate(scores) for name, scores in by_ranker.items() if scores},
        }
        report["families"][family] = block

    # `overall` is quality, so it excludes the harness controls. Their queries
    # ask with their targets' own heading wording by construction; pooling them inflates the
    # headline numbers AND the lift and shuffle gates that read `overall`, so
    # those gates would partly be measuring a self-match. Controls are reported
    # in their own block, where they cannot be mistaken for quality.
    for name in rankers:
        pooled = [
            s
            for family, by_ranker in per_family.items()
            if family not in controls
            for s in by_ranker[name]
        ]
        if pooled:
            report["overall"][name] = aggregate(pooled)
        control_scores = [
            s
            for family, by_ranker in per_family.items()
            if family in controls
            for s in by_ranker[name]
        ]
        if control_scores:
            report["controls"][name] = aggregate(control_scores)
        if memory_population.get(name):
            report["memory_population"][name] = aggregate(memory_population[name])

    report["overall"]["queries"] = sum(
        block["queries"] for family, block in report["families"].items() if family not in controls
    )
    report["memory_population"]["queries"] = len(memory_population.get("memory", []))
    # Every ranker in this block must describe the SAME queries. Recording the
    # per-ranker counts is what lets a gate say so; a lift computed from two
    # populations of different sizes is a comparison between different questions.
    report["memory_population"]["counts"] = {
        name: len(scores) for name, scores in memory_population.items() if scores
    }
    return report


def print_report(report: dict, manifest: dict) -> None:
    show_memory = any(block.get("memory") for block in report["families"].values())
    header = (
        f"{'family':16s} {'n':>4s}  "
        + "  ".join(f"{'R@' + str(k):>6s}" for k in CUTOFFS)
        + f"  {'MRR':>6s}   {'vs random':>9s}"
    )
    if show_memory:
        header += f"   {'mem R@10':>9s} {'mem MRR':>8s}"
    print(header)
    print("-" * len(header))

    def row(label: str, block: dict, suffix: str = "") -> None:
        model = block["model"]
        rnd = block.get("random", {})
        lift = (
            model[f"mrr@{MRR_AT}"] / rnd[f"mrr@{MRR_AT}"]
            if rnd.get(f"mrr@{MRR_AT}")
            else float("inf")
        )
        cells = "  ".join(f"{model[f'recall@{k}']:6.3f}" for k in CUTOFFS)
        line = (
            f"{label:16s} {block['queries']:4d}  {cells}  "
            f"{model[f'mrr@{MRR_AT}']:6.3f}   {lift:8.0f}x"
        )
        if show_memory:
            if block.get("memory"):
                line += (
                    f"   {block['memory']['recall@10']:9.3f} "
                    f"{block['memory'][f'mrr@{MRR_AT}']:8.3f}"
                )
            else:
                line += f"   {'circular':>9s} {'-':>8s}"
        print(line + suffix)

    for family, block in sorted(report["families"].items()):
        if block["control"]:
            continue
        row(family, block)
    print("-" * len(header))
    row("overall", report["overall"])
    for name in ("random", "constant"):
        if name in report["overall"]:
            base = report["overall"][name]
            cells = "  ".join(f"{base[f'recall@{k}']:6.3f}" for k in CUTOFFS)
            print(
                f"{name:16s} {report['overall']['queries']:4d}  {cells}  "
                f"{base[f'mrr@{MRR_AT}']:6.3f}   baseline"
            )

    controls = {f: b for f, b in report["families"].items() if b["control"]}
    if controls:
        print()
        print("harness controls -- NOT retrieval quality; excluded from `overall`.")
        print("They ask with their targets' own heading wording by construction; a low")
        print("score here means the harness is broken, not that the model is bad.")
        for family, block in sorted(controls.items()):
            row(family, block)

    population = report.get("memory_population") or {}
    if population.get("memory"):
        print()
        print(
            f"concept memory, over the {population['queries']} query/queries it may "
            "answer (baselines recomputed on that same subset):"
        )
        for name in ("memory", "model", "random", "constant"):
            if name in population:
                block = population[name]
                cells = "  ".join(f"{block[f'recall@{k}']:6.3f}" for k in CUTOFFS)
                print(
                    f"  {name:14s} {population['queries']:4d}  {cells}  "
                    f"{block[f'mrr@{MRR_AT}']:6.3f}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET, dest="embedding_set")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument(
        "--memory",
        type=Path,
        default=Path("build/training/memory"),
        help="index-backed concept memory to compare against direct retrieval",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    search = load_tool("search_chunks")
    queries_path = args.eval / "queries.jsonl"
    if not queries_path.is_file():
        raise SystemExit(
            f"evaluate_retrieval: no query set at {queries_path}\n"
            "  build one first: python3 training/tools/build_eval_queries.py "
            f"--out {args.eval}"
        )
    queries = read_jsonl(queries_path)
    manifest = json.loads((args.eval / "manifest.json").read_text(encoding="utf-8"))
    if not queries:
        raise SystemExit(
            "evaluate_retrieval: the query set is empty; an evaluation over no "
            "questions reports perfect agreement with nothing"
        )

    embedding_set = search.EmbeddingSet(args.embedding_set)
    corpus_sha = require_matching_corpus(args.chunks, embedding_set, manifest)

    rankers, memory, memory_module, circular = RANKERS, None, None, frozenset()
    if (args.memory / "manifest.json").is_file():
        memory_module = load_tool("build_index_memory")
        memory = memory_module.IndexMemory.load(args.memory)
        circular = circular_sources(memory.manifest)
        rankers = RANKERS + ("memory",)

    controls = frozenset(
        family for family, block in manifest["families"].items() if block["control"]
    )
    report = evaluate(
        queries,
        embedding_set,
        search,
        rankers=rankers,
        memory=memory,
        memory_module=memory_module,
        circular=circular,
        controls=controls,
    )
    report["corpus_sha256"] = corpus_sha
    report["model"] = {
        "name": embedding_set.manifest["model"],
        "revision": embedding_set.manifest["revision"],
        "semantics": embedding_set.manifest["semantics"],
    }
    report["documents"] = manifest["documents"]

    print(
        f"[eval]    {report['model']['name']} rev {report['model']['revision']} "
        f"semantics={report['model']['semantics']} over {manifest['documents']} document(s)"
    )
    print_report(report, manifest)
    if report["circular_queries"]:
        print(
            f"\n[circular] {report['circular_queries']} of {len(queries)} "
            "query/queries are withheld from the concept memory: each was derived "
            "from an index row the memory is built from, so scoring it there would "
            "be finding its own keys. Every other query scores both strategies."
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[write]   {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
