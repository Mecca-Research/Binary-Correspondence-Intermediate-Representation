"""Rows, in an order, without vectors: the relational half of the query surface.

What a caller needs when it wants rows the way a table returns them -- a predicate
conjunction resolved to a row set, an ORDER BY that is total and stable, and a page
of it. Ranked search is a different subsystem with a different definition of "in an
order"; it lives in `retrieval`, and the two deliberately do not share an entry
point, because a tool that quietly reordered a ranked result would be answering a
different question than it was asked.
"""

from __future__ import annotations

from . import engine


def resolve_selection(catalog, terms):
    """Parse and resolve `--where`, or refuse."""
    plan = engine.planner()
    predicates = [plan.parse_predicate(term) for term in terms]
    return plan.select(catalog, predicates)


def parse_order(spec: str) -> tuple[str, bool]:
    column, _, direction = spec.partition(":")
    direction = direction.strip().lower() or "asc"
    if direction not in ("asc", "desc"):
        raise ValueError(f"--order-by direction must be asc or desc, not {direction!r}")
    return column.strip(), direction == "desc"


def order_strategy(catalog, selection, column: str) -> str:
    """How ORDER BY will produce its rows: by reading the index, or by sorting.

    With no predicate the sorted index already holds exactly the order asked for, so
    the answer is a slice of it rather than a sort of every row. With a predicate the
    rows still have to be tested one at a time, and walking 2,215 index entries to
    find the 16 a narrow predicate admits costs more than sorting those 16.

    The choice is returned as a value rather than made inside a branch because both
    paths produce identical rows -- so a gate comparing their output passes whichever
    one ran, and the decision would be the one thing about this slice that nothing
    could observe (`docs/security/laws.md` L2).
    """
    if selection.rows is None and column in catalog.numeric_columns():
        return "index"
    return "sort"


def ordered_from_index(catalog, column: str, descending: bool) -> list[int]:
    """Every row ordered by a measurement, taken from the sorted index.

    Ascending is the index itself, read and returned. Descending is one stable sort
    of the measured prefix on the *negated* value, which reverses the values while
    leaving the rows inside a run of equal ones ascending -- `reverse=True` would
    reverse the ties along with the values, and a page boundary falling inside such a
    run has to land in the same place whichever direction was asked for. Walking the
    prefix to flip it run by run gives the same answer and measured slower here, on a
    corpus where half the adjacent index entries are ties.

    Rows carrying no measurement live past the index's split point and stay last in
    both directions -- absent is not small, and it is not large either.
    """
    rows = list(catalog.order[column])
    split = catalog.measured(column)
    present, missing = rows[:split], rows[split:]
    if not descending:
        return present + missing
    values = catalog.numeric[column]
    return sorted(present, key=lambda row: -values[row]) + missing


def ordered_rows(catalog, selection, column: str, descending: bool) -> list[int]:
    """The admitted rows, ordered by a column rather than by distance.

    A numeric column orders by its packed values, a missing measurement sorting last
    in either direction -- absent is not small, and sorting it as though it were is
    how a "shortest chunks" query comes back full of rows that were never measured.
    An indexed column orders by value, then row, so the result is total and stable.
    """
    catalog_module = engine.catalog()
    if order_strategy(catalog, selection, column) == "index":
        return ordered_from_index(catalog, column, descending)
    rows = list(selection.rows) if selection.rows is not None else list(range(catalog.rows_total))
    if column in catalog.numeric_columns():
        values = catalog.numeric[column]
        missing = [row for row in rows if values[row] == catalog_module.NUMERIC_NULL]
        present = [row for row in rows if values[row] != catalog_module.NUMERIC_NULL]
        # Two passes, because Python's sort is stable: rows ascending first, then by
        # value. A single `(values[row], row)` key with `reverse=True` would reverse
        # the tiebreak along with the key, so a run of equal values would come back
        # in a different order than ascending gives -- and a page boundary inside
        # such a run would then repeat or skip rows.
        present.sort()
        present.sort(key=lambda row: values[row], reverse=descending)
        return present + missing
    by_row: dict[int, str] = {}
    for value, positions in catalog.postings["columns"].get(column, {}).items():
        for row in positions:
            by_row[row] = value
    if not by_row:
        raise KeyError(f"catalog: column {column!r} is neither numeric nor indexed")
    rows.sort()
    rows.sort(key=lambda row: by_row.get(row, ""), reverse=descending)
    return rows
