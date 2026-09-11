"""The training corpus's database layer, addressed one subsystem at a time.

`catalog.py` and `plan.py` are the engine: row locator, inverted index, zone maps,
sorted index, statistics, and a cost-governed planner over them. This package is what
the rest of the tree is supposed to talk to instead.

The split is by *consumer need*, not by engine structure, because those are different
shapes. A tool that ranks vectors and a tool that rebuilds an embedding set both use
the catalog, and almost nothing they need from it overlaps:

    relational   which rows a predicate admits, in what order, and which page of them
                 -- `search_chunks --where/--order-by/--offset`, and any caller that
                 wants rows without wanting vectors
    analytics    how many, grouped by what, aggregated how, filtered by HAVING
                 -- `search_chunks --count`
    retrieval    the rows a ranked search is allowed to consider, and the text and
                 columns of the ones it returns
    ingest       the canonical row order, what a part is, and which parts moved
                 -- `embed_chunks`, and anything else that rebuilds a derived set
    history      generations: publish, read, verify (`generations.py`, unchanged --
                 it was already exactly this shape)

Why this exists rather than everyone importing `catalog`: before it, three tools each
carried their own copy of the sibling-module loader, and the three disagreed. One
returned any cached module that happened to share the name, one executed a fresh
module and then threw it away in favour of whatever `setdefault` kept, and one -- the
gate's -- checked that the cached module came from the file it meant. The last is
correct and is now the only one (`docs/security/laws.md` L14).

**The rail boundary is unchanged.** `training/` is never a build dependency of BCIR,
and nothing here reverses that. These modules import `bcir.kbcir.cost` the same way
`plan.py` already did -- to price a plan on BCIR's own twelve axes -- and BCIR imports
nothing from here.
"""
