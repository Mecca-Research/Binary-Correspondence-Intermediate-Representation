#!/usr/bin/env python3
"""What the planner decided, recorded, so a change to it cannot be silent.

Every other gate in this tree checks that an answer is *right*. None of them checks
that it was reached the same way. That gap is not hypothetical here: the zone map
once priced a 25%-selective predicate at 98% selective, the conjunction estimate
reported a bound as a count for every indexed pair in the corpus, and both produced
correct rows the whole time. A wrong plan behind a right answer is invisible to a
correctness gate by construction -- it is the same rows, in the same order, more
slowly or on a worse-informed decision -- and it stays invisible until somebody
measures a query that got slow and works backwards.

So this records the decision itself. For a fixed set of requests it stores what the
planner knew (the estimate and whether that estimate was a count or a bound), what it
resolved (the admitted rows), and what it chose (the plan, per objective, and the
strategy for ordering and for ranges). `--compare` rebuilds all of it and prints
every field that moved.

**A difference is a finding, not a failure of the tool.** Improving an estimate moves
this file, and so does breaking one; the gate cannot tell those apart and does not
try. What it does is refuse to let either happen without somebody looking. Re-record
deliberately, and the diff in review says exactly which decisions changed.

    python3 training/tools/plan_baseline.py --record
    python3 training/tools/plan_baseline.py --compare

**Bound to the corpus it was recorded against.** The admitted counts are facts about
this chunk table, so the file carries the catalog's fingerprint and a comparison
against a different corpus is refused rather than reported as 40 regressions. That
makes adding corpus content a re-record, which is the point: content changes move
plans, and the diff is where anybody finds out (`docs/security/laws.md` L2 -- a check
that cannot fail on the input it is given is not checking that input).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from db import analytics, engine, relational  # noqa: E402

SCHEMA = "bcir-training/plan-baseline/v1"
BUILDER = "training/tools/plan_baseline.py"
LICENSE = "LicenseRef-BCIR-NC-1.0"

DEFAULT_BASELINE = Path("training/plans/baseline-v1.json")
DEFAULT_CHUNKS = Path("build/training/chunks")
DEFAULT_CATALOG = Path("build/training/catalog")

#: The requests whose plans are pinned. Chosen to reach every decision the planner
#: makes rather than to look like a workload: each operator at least once, one and
#: several terms, a conjunction the joint statistics cover and one they do not, a
#: range that prunes parts and one that cannot, an order the sorted index serves and
#: one it does not. A request nothing here exercises is a decision nothing here pins,
#: which is why `check_plan_baseline` reconciles this list against the operator table
#: rather than trusting it to stay complete (`docs/security/laws.md` L15).
QUERIES: tuple[dict, ...] = (
    {"name": "unfiltered", "where": []},
    {"name": "eq-indexed", "where": ["subject=llvm"]},
    {"name": "eq-rare", "where": ["subject=data"]},
    {"name": "eq-absent-value", "where": ["subject=nothing-has-this"]},
    {"name": "ne-indexed", "where": ["language!=llvm"]},
    {"name": "in-set", "where": ["subject=llvm,data"]},
    {"name": "in-quoted", "where": ['subject="llvm","data"']},
    {"name": "prefix-counted", "where": ["source_path^=training/llvm"]},
    {"name": "prefix-uncounted", "where": ["source_path^=training/llvm/12-backend"]},
    {"name": "isnull", "where": ["language=?"]},
    {"name": "notnull", "where": ["language!=?"]},
    {"name": "ge", "where": ["char_count>=500"]},
    {"name": "gt", "where": ["char_count>5000"]},
    {"name": "le", "where": ["char_count<=500"]},
    {"name": "lt", "where": ["char_count<80"]},
    {"name": "pair-covered", "where": ["subject=llvm", "kind=code"]},
    {"name": "pair-empty", "where": ["subject=data", "kind=code"]},
    {"name": "pair-negated", "where": ["subject!=llvm", "kind=prose"]},
    {"name": "triple-indexed", "where": ["subject=llvm", "kind=code", "language=llvm"]},
    {"name": "indexed-and-range", "where": ["kind=code", "char_count>=500"]},
    {"name": "indexed-and-prefix", "where": ["subject=llvm", "source_path^=training/llvm"]},
    {"name": "range-interval", "where": ["char_count>=500", "char_count<=2000"]},
    {"name": "order-unfiltered", "where": [], "order_by": "char_count:desc"},
    {"name": "order-narrow", "where": ["subject=data"], "order_by": "char_count:asc"},
    {"name": "group-subject", "where": [], "group_by": "subject", "stats": "char_count"},
    {
        "name": "group-having",
        "where": [],
        "group_by": "kind",
        "stats": "char_count",
        "having": ["rows>=100"],
    },
)

#: What a recorded plan says, per objective. Objectives are named rather than taken
#: from the enum's iteration order so that adding one is a visible change to this file.
OBJECTIVES = ("EXACTNESS", "LATENCY", "FOOTPRINT", "STARTUP")


class BaselineError(RuntimeError):
    """A baseline that cannot be compared. Never a comparison that quietly passes."""


def _facts(catalog, query: dict) -> dict:
    """Everything pinned about one request. Deterministic, and host-independent.

    No duration and no cost total. A wall time cannot gate -- it is a property of the
    runner, not of the plan (`docs/security/laws.md`, the metric classes) -- and the
    scalarized cost moves whenever the model's calibration constants are retuned,
    which says nothing about whether the planner's *decision* changed. What is pinned
    is the decision and the knowledge behind it: both are exact integers and both are
    the same on every host.
    """
    plan = engine.planner()
    selection = relational.resolve_selection(catalog, query.get("where", []))
    recorded: dict = {
        "where": list(query.get("where", [])),
        "predicates": [str(p) for p in selection.predicates],
        "admitted": catalog.rows_total if selection.rows is None else len(selection.rows),
        "estimated": selection.estimated,
        "estimate_exact": selection.estimate_exact,
    }

    chosen: dict[str, str] = {}
    illegal: dict[str, int] = {}
    for objective in OBJECTIVES:
        plans = plan.candidates(
            catalog,
            selection,
            top_k=5,
            dim=512,
            want_text=True,
            require_exact=objective == "EXACTNESS",
            available_backends=frozenset(plan.BACKENDS),
            kernel_cached=True,
            files_touched=len(catalog.files),
        )
        chosen[objective] = plan.choose(plans, plan.Objective[objective]).label()
        illegal[objective] = sum(1 for candidate in plans if not candidate.legal)
    recorded["chosen"] = chosen
    recorded["illegal_candidates"] = illegal

    # How a range is answered, and how much of the table it never touches. The zone
    # map's whole claim is that it skips parts; a claim with no recorded number is one
    # nobody can see stop being true.
    ranges = [p for p in selection.predicates if p.op in plan.RANGE_OPERATORS]
    if ranges:
        column = ranges[0].column
        low, high = None, None
        for predicate in ranges:
            if predicate.column != column:
                continue
            first, second = plan.interval(predicate)
            low = first if low is None else max(low, first) if first is not None else low
            high = second if high is None else min(high, second) if second is not None else high
        recorded["range"] = {
            "column": column,
            "strategy": catalog.range_strategy(column, low, high),
            "parts_total": len(catalog.parts),
            "parts_pruned": sum(
                1 for part in catalog.parts if _pruned(catalog.zone(part, column), low, high)
            ),
        }

    if query.get("order_by"):
        column, descending = relational.parse_order(query["order_by"])
        recorded["order"] = {
            "column": column,
            "descending": descending,
            "strategy": relational.order_strategy(catalog, selection, column),
        }

    if query.get("group_by"):
        _, scanned = analytics.group_counts(catalog, query["group_by"], selection)
        kept = []
        for value, rows in analytics.group_rows(catalog, query["group_by"], selection):
            summary = analytics.group_summary(catalog, query.get("stats"), rows)
            if all(
                analytics.having_holds(analytics.parse_having(term), summary)
                for term in query.get("having", [])
            ):
                kept.append([value, len(rows)])
        recorded["group"] = {"column": query["group_by"], "scanned": scanned, "kept": kept}

    return recorded


def _pruned(zone: dict, low: int | None, high: int | None) -> bool:
    """Whether a part's zone map rules it out for `[low, high]`. One rule, here."""
    if zone.get("count", 0) == 0:
        return True
    if low is not None and zone["max"] < low:
        return True
    if high is not None and zone["min"] > high:
        return True
    return False


def record(catalog) -> dict:
    """The whole baseline, as it will be written."""
    return {
        "schema": SCHEMA,
        "builder": BUILDER,
        "license": LICENSE,
        "corpus": catalog.manifest["fingerprint"],
        "rows_total": catalog.rows_total,
        "objectives": list(OBJECTIVES),
        "queries": {query["name"]: _facts(catalog, query) for query in QUERIES},
    }


def _flatten(value, prefix: str = "") -> dict[str, str]:
    """One recorded plan as leaf paths, so a diff names the field that moved."""
    if isinstance(value, dict):
        out: dict[str, str] = {}
        for key in sorted(value):
            out.update(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{index}]"))
        return out
    return {prefix: json.dumps(value, sort_keys=True)}


def compare(stored: dict, fresh: dict) -> list[str]:
    """Every field that moved, named. Empty means every recorded decision held.

    The corpus check comes first and refuses rather than reporting, because comparing
    two different corpora produces a finding per query and not one of them is about
    the planner.
    """
    if stored.get("schema") != SCHEMA:
        raise BaselineError(f"plan baseline: not {SCHEMA} (found {stored.get('schema')!r})")
    if stored.get("corpus") != fresh["corpus"]:
        raise BaselineError(
            "plan baseline: recorded against a different corpus\n"
            f"  recorded: {stored.get('corpus')}\n"
            f"  now:      {fresh['corpus']}\n"
            f"  re-record it: python3 {BUILDER} --record"
        )
    findings = []
    missing = sorted(set(fresh["queries"]) - set(stored.get("queries", {})))
    removed = sorted(set(stored.get("queries", {})) - set(fresh["queries"]))
    for name in missing:
        findings.append(f"{name}: no recorded plan; the query set grew")
    for name in removed:
        findings.append(f"{name}: recorded but no longer in the query set")
    for name in sorted(set(fresh["queries"]) & set(stored.get("queries", {}))):
        left = _flatten(stored["queries"][name])
        right = _flatten(fresh["queries"][name])
        for field in sorted(set(left) | set(right)):
            was, now = left.get(field, "<absent>"), right.get(field, "<absent>")
            if was != now:
                findings.append(f"{name}.{field}: {was} -> {now}")
    return findings


def load(path: Path) -> dict:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BaselineError(
            f"plan baseline: none at {path}\n  record one: python3 {BUILDER} --record"
        ) from exc
    except (OSError, ValueError) as exc:
        raise BaselineError(f"plan baseline: {path} is unreadable: {exc}") from exc
    if not isinstance(stored, dict):
        raise BaselineError(f"plan baseline: {path} is not an object")
    return stored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", action="store_true", help="write the baseline")
    group.add_argument("--compare", action="store_true", help="refuse any plan that moved")
    args = parser.parse_args(argv)

    try:
        catalog = engine.open_catalog(args.catalog, args.chunks)
        fresh = record(catalog)
        if args.record:
            args.baseline.parent.mkdir(parents=True, exist_ok=True)
            args.baseline.write_text(
                json.dumps(fresh, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"[baseline] {len(fresh['queries'])} plan(s) over {fresh['rows_total']} rows")
            print(f"[write]    {args.baseline}")
            return 0
        findings = compare(load(args.baseline), fresh)
    except (BaselineError, engine.catalog().CatalogError, engine.planner().PlanError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if findings:
        print(f"plan baseline: {len(findings)} decision(s) moved", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        print(f"  re-record deliberately: python3 {BUILDER} --record", file=sys.stderr)
        return 1
    print(f"plan baseline: {len(fresh['queries'])} plan(s) unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
