#!/usr/bin/env python3
"""Immutable, content-addressed generations of the corpus.

The training rail publishes one set and overwrites it. A retrieval result is
therefore reproducible only for as long as nobody rebuilds: the corpus a score was
measured against is gone the moment it changes, and there is no way to ask what the
answer used to be. This is the artifact discipline BCIR already states for its own
outputs -- immutable within a generation, promoted at quiescent boundaries, never
mutated in place -- applied to the corpus rather than reinvented for it.

A generation's name is its content. `generation_id` is derived from the digest of
the chunk files it holds and nothing else: no wall clock, no random label, no
sequence counter. Publishing the same corpus twice produces the same id and writes
nothing new, and two hosts that build the same corpus name it identically.

That is deliberate and it is the thing to keep. TileDB reaches for the same
capability -- a fragment whose name carries its validity interval -- and reaches it
with `generate_timestamped_name()`, wall-clock plus a random label, which means the
same array built twice is two different arrays. Byte identity is the property this
repository gates on everywhere else; a versioning scheme that destroys it would cost
more than the time travel it buys.

`parent` records which generation was current when this one was published. It is
history, not identity: a corpus that changes and changes back returns to its first
id, with a new parent link recording the path. So the id answers "which corpus is
this", and the parent chain answers "how did we get here" -- two questions that a
timestamped name conflates and answers badly.

    python3 training/tools/generations.py --publish --label "before the rewrite"
    python3 training/tools/generations.py --list
    python3 training/tools/generations.py --verify <generation_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import catalog as catalog_module  # noqa: E402

SCHEMA = "bcir-training/generation/v1"
BUILDER = "training/tools/generations.py"
LICENSE = "LicenseRef-BCIR-NC-1.0"

DEFAULT_ROOT = Path("build/training/generations")
DEFAULT_CHUNKS = Path("build/training/chunks")

CURRENT_FILE = "CURRENT"
MANIFEST_FILE = "manifest.json"

#: How many hex digits of the corpus digest name a generation. Sixteen is far past
#: the collision horizon for a corpus that is rebuilt by hand, and short enough to
#: read in a path.
ID_DIGITS = 16


class GenerationError(RuntimeError):
    """A generation is absent, malformed, or does not match its own digest."""


@dataclass(frozen=True)
class Generation:
    generation_id: str
    parent: str | None
    label: str
    corpus: str
    digest: str
    rows_total: int
    files: tuple[str, ...]


# --------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------


def corpus_digest(chunk_dir: Path) -> str:
    """The content address of a chunk table: its file names and their bytes.

    Identical to what `catalog.fingerprint` computes, and deliberately so -- two
    definitions of "the same corpus" is one definition too many.
    """
    return catalog_module.fingerprint(chunk_dir)


def tree_digest(root: Path) -> str:
    """A digest over every file under `root`, by relative path and content.

    Paths are included, and in sorted order, so a file that moves or disappears
    changes the answer. A digest over contents alone would let a generation lose an
    artifact and still verify.
    """
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode("utf-8"))
    digest.update(b"\0")
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def generation_id_for(chunk_dir: Path) -> str:
    return f"g-{corpus_digest(chunk_dir)[:ID_DIGITS]}"


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def _manifest_path(root: Path, generation_id: str) -> Path:
    return root / generation_id / MANIFEST_FILE


def read(root: Path, generation_id: str) -> Generation:
    """One generation's manifest, or a refusal naming what is wrong."""
    path = _manifest_path(root, generation_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GenerationError(f"generations: no generation {generation_id} under {root}") from exc
    except (OSError, ValueError) as exc:
        raise GenerationError(f"generations: {path} is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise GenerationError(f"generations: {path} is not {SCHEMA}")
    for required in ("generation_id", "corpus", "digest", "rows_total", "files"):
        if required not in payload:
            raise GenerationError(f"generations: {path} has no {required!r}")
    return Generation(
        generation_id=payload["generation_id"],
        parent=payload.get("parent"),
        label=payload.get("label", ""),
        corpus=payload["corpus"],
        digest=payload["digest"],
        rows_total=int(payload["rows_total"]),
        files=tuple(payload["files"]),
    )


def listing(root: Path) -> list[Generation]:
    """Every published generation, oldest first by parent chain where one exists."""
    if not root.is_dir():
        return []
    found = []
    for entry in sorted(root.iterdir()):
        # A dot-prefixed directory is staging, not a generation. `publish` writes the
        # manifest into `.staging-<id>/` and only then renames, so a process killed in
        # that window -- SIGKILL, OOM, power loss, none of which run the `except`
        # clause below -- leaves a complete manifest under a name that was never
        # published. This listed it under the real id, which `read` then could not
        # open, so the catalogue of generations advertised one that does not exist.
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        if not (entry / MANIFEST_FILE).is_file():
            continue
        generation = read(root, entry.name)
        # And the name must be the id. A generation is content-addressed, so a
        # directory whose manifest names something else is not a generation under a
        # different name -- it is a manifest somebody moved (L1: say so rather than
        # list it as though the two agreed).
        if generation.generation_id != entry.name:
            raise GenerationError(
                f"generations: {entry} holds a manifest naming "
                f"{generation.generation_id!r}; a generation's directory is its id"
            )
        found.append(generation)
    by_id = {generation.generation_id: generation for generation in found}
    ordered: list[Generation] = []
    remaining = dict(by_id)
    # Walk parent links from the roots outward; anything left over keeps its sorted
    # position, so an orphan is listed rather than silently dropped.
    roots = [g for g in found if g.parent is None or g.parent not in by_id]
    for generation in roots:
        cursor: Generation | None = generation
        while cursor is not None:
            if cursor.generation_id not in remaining:
                break
            ordered.append(remaining.pop(cursor.generation_id))
            cursor = next(
                (g for g in remaining.values() if g.parent == ordered[-1].generation_id), None
            )
    ordered.extend(remaining.values())
    return ordered


def current(root: Path) -> Generation:
    """The generation `CURRENT` points at."""
    pointer = root / CURRENT_FILE
    try:
        generation_id = pointer.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise GenerationError(f"generations: no {pointer}; nothing has been published") from exc
    if not generation_id:
        raise GenerationError(f"generations: {pointer} is empty")
    return read(root, generation_id)


def verify(root: Path, generation_id: str) -> bool:
    """True when the stored generation still hashes to the digest it published.

    The digest covers every file in the generation directory except the manifest
    that records it, so a tampered catalog, a truncated chunk file, or an added
    stray artifact all fail. A generation that cannot vouch for itself is not a
    generation to reproduce an answer from.
    """
    try:
        generation = read(root, generation_id)
    except GenerationError:
        return False
    directory = root / generation_id
    with tempfile.TemporaryDirectory() as scratch:
        mirror = Path(scratch) / "g"
        shutil.copytree(directory, mirror)
        (mirror / MANIFEST_FILE).unlink(missing_ok=True)
        return tree_digest(mirror) == generation.digest


# --------------------------------------------------------------------------
# Publishing
# --------------------------------------------------------------------------


def publish(
    root: Path,
    chunk_dir: Path = DEFAULT_CHUNKS,
    *,
    label: str = "",
    extra: dict | None = None,
) -> Generation:
    """Publish the current corpus as a generation, and make it current.

    Idempotent by construction: the id is the corpus digest, so republishing an
    unchanged corpus finds the directory already there, verifies it, and only moves
    the `CURRENT` pointer. Nothing inside a published generation is ever rewritten --
    that is what "immutable within a generation" means, and it is why an old answer
    stays reproducible.

    The whole generation is staged in a sibling directory and moved into place with
    one `os.replace`, so a reader never sees a partially written generation and an
    interrupted publish leaves nothing half-committed under a name that claims to be
    complete.
    """
    root.mkdir(parents=True, exist_ok=True)
    generation_id = generation_id_for(chunk_dir)
    destination = root / generation_id

    try:
        parent: str | None = current(root).generation_id
    except GenerationError:
        parent = None
    if parent == generation_id:
        parent = read(root, generation_id).parent

    if destination.is_dir():
        if not verify(root, generation_id):
            raise GenerationError(
                f"generations: {destination} exists but does not match its digest; "
                "an immutable generation was modified in place"
            )
        _point_at(root, generation_id)
        return read(root, generation_id)

    staging = root / f".staging-{generation_id}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        (staging / "chunks").mkdir(parents=True)
        for path in catalog_module.chunk_files(chunk_dir):
            shutil.copy2(path, staging / "chunks" / path.name)
        built = catalog_module.build(staging / "chunks")
        catalog_module.write(built, staging / "catalog")
        manifest_rows = built[0]["rows_total"]
        files = tuple(
            sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
        )
        digest = tree_digest(staging)
        manifest = {
            "schema": SCHEMA,
            "builder": BUILDER,
            "license": LICENSE,
            "generation_id": generation_id,
            "parent": parent,
            "label": label,
            "corpus": corpus_digest(chunk_dir),
            "digest": digest,
            "rows_total": manifest_rows,
            "files": list(files),
        }
        if extra:
            manifest["extra"] = extra
        (staging / MANIFEST_FILE).write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )
        # Durability, through `catalog`'s helpers rather than a second copy of them.
        # The catalog inside this generation was published with fsync at every step;
        # the chunks beside it and the manifest above them were not, so a crash could
        # leave a generation whose directory entry exists and whose contents are
        # whatever the page cache had got to -- the half of the ACID publish that
        # landed on one rail and not on the one wrapping it (L14).
        _sync_tree(staging)
        try:
            os.replace(staging, destination)
        except OSError:
            # Another writer published this same content-addressed generation between
            # the existence check above and this rename. POSIX and Windows disagree
            # about renaming onto a directory that now exists -- one may succeed, the
            # other refuses -- so the outcome is decided here rather than by the host
            # (`docs/security/laws.md` L12). The id is the corpus digest, so if what
            # is in place verifies, this publish already happened and the staged copy
            # is redundant; if it does not, the refusal stands.
            if not (destination.is_dir() and verify(root, generation_id)):
                raise
            shutil.rmtree(staging, ignore_errors=True)
        catalog_module._sync_directory(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _point_at(root, generation_id)
    return read(root, generation_id)


def _sync_tree(directory: Path) -> None:
    """fsync every file under `directory`, then the directories holding them.

    `catalog._publish` does this per artifact and `catalog._sync_directory` does the
    directory half; this is the same discipline over a staged generation, whose
    chunk copies and manifest were written with `shutil.copy2` and `write_text` and
    therefore reached the page cache and no further.
    """
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_dir():
            catalog_module._sync_directory(path)
    catalog_module._sync_directory(directory)


def _point_at(root: Path, generation_id: str) -> None:
    """Move `CURRENT` atomically. The pointer is the only mutable thing here.

    Flushed and fsynced like `catalog._point_at`, which is the same function: a
    pointer that survives the crash its atomic rename exists to survive has to
    reach the disk, not the page cache.
    """
    descriptor, temporary = tempfile.mkstemp(prefix=f".{CURRENT_FILE}.tmp-", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(generation_id + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / CURRENT_FILE)
        catalog_module._sync_directory(root)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def open_catalog(root: Path, generation_id: str):
    """A `Catalog` over a stored generation -- how an old answer is reproduced."""
    directory = root / generation_id
    if not (directory / MANIFEST_FILE).is_file():
        raise GenerationError(f"generations: no generation {generation_id} under {root}")
    return catalog_module.Catalog.load(directory / "catalog", directory / "chunks")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--publish", action="store_true", help="publish the current corpus")
    parser.add_argument("--label", default="", help="a human note stored with the generation")
    parser.add_argument("--list", action="store_true", dest="listing")
    parser.add_argument("--verify", metavar="GENERATION", help="re-hash a stored generation")
    args = parser.parse_args(argv)

    if not (args.publish or args.listing or args.verify):
        parser.error("choose one of --publish, --list, --verify")

    try:
        if args.publish:
            generation = publish(args.root, args.chunks, label=args.label)
            print(
                f"[generation] {generation.generation_id}  {generation.rows_total} row(s)"
                + (f"  parent={generation.parent}" if generation.parent else "  (first)")
            )
            print(f"[digest]     {generation.digest}")
            if generation.label:
                print(f"[label]      {generation.label}")
        if args.listing:
            found = listing(args.root)
            if not found:
                print(f"[generations] none under {args.root}")
            else:
                try:
                    head = current(args.root).generation_id
                except GenerationError:
                    head = ""
                for generation in found:
                    marker = "*" if generation.generation_id == head else " "
                    print(
                        f" {marker} {generation.generation_id}  {generation.rows_total:6d} row(s)"
                        f"  {generation.label or ''}"
                    )
        if args.verify:
            ok = verify(args.root, args.verify)
            print(f"[verify] {args.verify}: {'MATCHES' if ok else 'DOES NOT MATCH'}")
            return 0 if ok else 1
    except GenerationError as exc:
        print(f"generations: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
