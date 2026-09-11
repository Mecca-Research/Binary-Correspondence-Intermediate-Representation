"""What a ranked search needs from the database, and nothing about ranking.

The line is deliberate. Which rows a search may consider, and what those rows say,
are database questions. How a vector is scored, which kernel does it and whether the
answer is exact are `search_chunks`' own, and putting them here would make this module
depend on an embedding set to answer a question about rows.

So: a predicate narrows the candidates, the kernel ranks what survives, and the text
of the few that come back is fetched by seeking recorded spans rather than by reading
the corpus -- which is the whole of late materialization from the caller's side.
"""

from __future__ import annotations

from . import engine, relational


def eligible(catalog, terms):
    """The rows a ranked search may consider, from a conjunction of `--where` terms.

    The same resolution the relational side uses, named again here because the
    guarantee a ranked caller needs is different: a predicate removes rows, it never
    reorders the ones it keeps, so narrowing a search must not change the order of
    what survives.
    """
    return relational.resolve_selection(catalog, terms)


def records(catalog, rows):
    """Row -> its chunk record, read by seeking the recorded spans.

    For `k` results this reads `k` spans rather than the corpus. The catalog verifies
    each fetched row against the chunk_id it recorded, so a stale locator refuses
    instead of returning a neighbouring row's text.
    """
    return catalog.fetch(list(rows))


def projectable(catalog) -> tuple[str, ...]:
    """The columns a caller may ask to be projected, by name.

    Taken from the catalog rather than listed here: a third hand-written copy of a
    column list is how the first two came to disagree (`docs/security/laws.md` L15).
    """
    module = engine.catalog()
    return tuple(catalog.indexed_columns()) + catalog.numeric_columns() + (module.PATH_COLUMN,)
