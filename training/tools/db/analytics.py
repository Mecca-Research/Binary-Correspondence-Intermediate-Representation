"""Analytics over the catalog: how many, grouped by what, aggregated how.

What `search_chunks --count` needs, and nothing else. A caller here is asking a
question *about* the rows rather than for the rows themselves, which is why grouping
and aggregation live apart from `relational`: the answers are per group, not per row,
and the fast paths are different -- an unfiltered count comes from statistics the
build already wrote, while a filtered one intersects postings.
"""

from __future__ import annotations

from . import engine

#: What a HAVING term may compare. `rows` is SQL's COUNT(*) -- every row of the group;
#: `count` is COUNT(<the measurement column>) -- only the rows carrying a measurement.
#: SQL spells that distinction with an argument; here they are two names, so a term
#: cannot mean one and be read as the other.
AGGREGATE_FIELDS = ("rows", "count", "nulls", "min", "max", "sum", "avg")


def group_counts(catalog, column: str, selection=None) -> tuple[list[tuple[str, int]], bool]:
    """GROUP BY over an indexed column, and whether it cost a scan.

    Unfiltered it is answered from the statistics the build already computed -- no
    artifact read, no row touched. Filtered it intersects each value's postings list
    with the admitted rows, which is work proportional to the *index*, not to the
    corpus. Both are exact; the flag says which happened, because "counted from
    statistics" and "counted by intersecting postings" are different claims.
    """
    if selection is None or selection.rows is None:
        counts = catalog.distinct(column)
        return sorted(counts.items(), key=lambda pair: (-int(pair[1]), pair[0])), False
    return [(value, len(rows)) for value, rows in group_rows(catalog, column, selection)], True


def group_rows(catalog, column: str, selection=None) -> list[tuple[str, list[int]]]:
    """The admitted rows of each distinct value of an indexed column, largest group first.

    One membership rule, used by the count and by every per-group aggregate, so
    `COUNT(*) GROUP BY kind` and `AVG(char_count) GROUP BY kind` cannot disagree about
    which rows are in a group (`docs/security/laws.md` L14). Every row carries a key
    for every indexed column -- `NULL_KEY` where it has no value -- so the groups
    partition the admitted rows rather than covering some of them.
    """
    postings = catalog.postings["columns"].get(column)
    if postings is None:
        raise KeyError(f"catalog: column {column!r} is not indexed")
    admitted = None if selection is None or selection.rows is None else set(selection.rows)
    groups = []
    for value, rows in postings.items():
        kept = list(rows) if admitted is None else [row for row in rows if row in admitted]
        if kept:
            groups.append((value, kept))
    groups.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    return groups


def parse_having(text: str):
    """One HAVING term, in the same grammar `--where` uses.

    Reusing `parse_predicate` is not a shortcut: it means maximal munch, the
    ASCII-integer rule and the refusal wording are defined once and apply to both
    clauses (`docs/security/laws.md` L14). What differs is only what a term may name,
    so only that is checked here.
    """
    plan = engine.planner()
    predicate = plan.parse_predicate(text)
    if predicate.column not in AGGREGATE_FIELDS:
        raise plan.PlanError(
            f"HAVING {predicate.column!r} is not an aggregate; the aggregates are "
            f"{', '.join(AGGREGATE_FIELDS)}"
        )
    if predicate.op not in ("eq", "ne", *plan.RANGE_OPERATORS):
        raise plan.PlanError(
            f"HAVING {text!r}: an aggregate is compared against one integer, so the "
            "operators are =, !=, >=, <=, > and <"
        )
    # `parse_predicate` reads the bound for the comparisons only, so `rows=x` would
    # otherwise reach the group loop and fail there, once per group.
    plan.require_integer(predicate.column, predicate.op, predicate.values[0])
    return predicate


def having_holds(predicate, summary: dict) -> bool:
    """Whether one group's aggregate satisfies one HAVING term.

    `avg` is compared as the exact rational SUM/COUNT rather than as the printed
    float: `AVG >= 500` asks a question about the measurements themselves, and
    rounding them to a decimal first would put a group on the wrong side of its own
    bound. A group with no measured row has no MIN, MAX or AVG and satisfies no
    comparison against one -- the rule a row with no value already follows.
    """
    plan = engine.planner()
    field = predicate.column
    if field not in summary:
        raise plan.PlanError(
            f"HAVING {field} needs --stats <column>; without one a group knows only "
            "its row count, spelled 'rows'"
        )
    bound = predicate.bound
    if field == "avg":
        if not summary["count"]:
            return False
        left, right = summary["sum"], bound * summary["count"]
    else:
        left = summary[field]
        if left is None:
            return False
        right = bound
    if predicate.op == "eq":
        return left == right
    if predicate.op == "ne":
        return left != right
    if predicate.op == "ge":
        return left >= right
    if predicate.op == "gt":
        return left > right
    if predicate.op == "le":
        return left <= right
    if predicate.op == "lt":
        return left < right
    raise plan.PlanError(f"HAVING has no rule for operator {predicate.op!r}")


def group_summary(catalog, stats_column: str | None, rows: list[int]) -> dict:
    """One group's aggregates. `rows` is always present; the rest need a --stats column."""
    if stats_column is None:
        return {"rows": len(rows)}
    summary = dict(catalog.aggregate(stats_column, rows))
    summary["rows"] = len(rows)
    return summary
