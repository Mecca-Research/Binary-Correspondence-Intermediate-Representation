"""What a tool rebuilding a derived set needs: row order, parts, and what moved.

`embed_chunks` is the caller this exists for. It does not query anything -- it walks
the corpus in the catalog's order, writes a vector per row, and wants to re-embed as
little as possible. Three facts serve all of that, and all three have to come from
the catalog rather than be re-derived, because a derived set whose idea of row `i`
differs from the catalog's is wrong in a way nothing downstream can detect.
"""

from __future__ import annotations

from pathlib import Path

from . import engine


def row_sort_key(record: dict):
    """The canonical order of rows, as the catalog defines it.

    Re-exported rather than reimplemented. An embedding set and a catalog that sorted
    by two rules agreeing today would disagree the first time a subject or a path
    changed, and every vector after the divergence would describe the wrong chunk
    (`docs/security/laws.md` L12, L14).
    """
    return engine.catalog().row_sort_key(record)


def parts(chunk_dir: Path) -> list[dict]:
    """The catalog's parts for this corpus, or none if it cannot be built.

    The single answer to "what is a part". Both the parts a set records and the parts
    it later compares against come through here, so a set cannot be written against
    one definition and read against another.
    """
    catalog = engine.catalog()
    try:
        return catalog.build(chunk_dir)[0].get("parts") or []
    except catalog.CatalogError:
        return []


def parts_for(chunk_dir: Path, source: str | None) -> list[dict]:
    """The parts cut from one chunk file, or all of them when `source` is None.

    Matched on `source`, the file a part came from, rather than on `part_id`: a part
    is a bounded block, so its id carries a block number and a subject names all the
    parts cut from that subject's file rather than one part.
    """
    found = parts(chunk_dir)
    if source is None:
        return found
    return [part for part in found if part.get("source") == source]


def changed_parts(chunk_dir: Path, recorded: list[dict] | None) -> tuple[str, ...]:
    """Part ids whose content differs from what was recorded, or that only one side has.

    The coarse half of the incremental rule: a part whose rows are unchanged cannot
    hold a changed row, so nothing in it needs looking at. The fine half is the
    per-row text digest, which is what actually decides reuse -- this answers a
    different question, and a part that moved while every one of its rows stayed
    identical is worth seeing.
    """
    before = {part.get("part_id"): part.get("content") for part in (recorded or [])}
    if not before:
        return ()
    after = {part.get("part_id"): part.get("content") for part in parts(chunk_dir)}
    names = before.keys() | after.keys()
    return tuple(sorted(name for name in names if before.get(name) != after.get(name)))
