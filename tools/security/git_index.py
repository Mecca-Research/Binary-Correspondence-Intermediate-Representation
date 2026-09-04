#!/usr/bin/env python3
"""Index-vs-worktree reconciliation, shared by the assurance rails.

Every rail that inspects tracked files reads the WORKTREE, but the next
commit records the INDEX. A secret (or an ``os.system`` call) staged and
then overwritten with a benign unstaged copy passes a worktree-only gate
while shipping unchanged. Both the secret scan and the tool-boundary
audit need the same reconciliation, so it lives here once: the same
defect fixed twice in two places is how there come to be two defects.

In a clean checkout — every CI run — the divergent set is empty and the
whole pass costs two `git` invocations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

# Sentinel: the index object is real but larger than the caller's
# inspection budget — a finding, distinct from None (unreadable).
STAGED_OVERSIZED = object()


def staged_divergent(root: Path) -> list[str] | None:
    """Tracked paths whose stage-0 blob differs from the worktree copy.

    None means the divergence question itself could not be answered — a
    fail-closed finding for the caller, never a silent downgrade.

    ``git diff`` alone is not the whole answer: ``--assume-unchanged`` and
    ``--skip-worktree`` tell git to stop comparing those entries, so a
    staged secret under either flag reports no divergence while shipping
    exactly the same. Entries git has been told not to check are precisely
    the ones a gate must check itself, so they are unioned in.
    """
    paths: list[str] = []
    seen: set[str] = set()
    try:
        diffed = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", "-z"],
            capture_output=True,
            check=False,
        )
        # -v tags every entry: lowercase means assume-unchanged, S/s means
        # skip-worktree. Both suppress the comparison above.
        flagged = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-v", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if diffed.returncode != 0 or flagged.returncode != 0:
        return None
    for item in diffed.stdout.split(b"\0"):
        if not item:
            continue
        rel = item.decode("utf-8", "surrogateescape")
        if rel not in seen:
            seen.add(rel)
            paths.append(rel)
    for item in flagged.stdout.split(b"\0"):
        # `<tag><space><path>`: a one-character tag, then the path verbatim.
        if len(item) < 3 or item[1:2] != b" ":
            continue
        tag = item[0:1]
        if not (tag.islower() or tag in (b"S", b"s")):
            continue
        rel = item[2:].decode("utf-8", "surrogateescape")
        if rel not in seen:
            seen.add(rel)
            paths.append(rel)
    return paths


def staged_blob(root: Path, rel: str, *, cap: int) -> Any:
    """The stage-0 index blob for ``rel``, bounded at ingress.

    The cap is asked of the OBJECT before a byte is materialized: a
    `git show` into `capture_output` expands the whole blob into this
    process first, so a highly compressible staged object could exhaust
    the runner well before a post-hoc length check ever ran. Bounds live
    where the resource commits — here that is `cat-file -s`, then a read
    of one byte past the cap so an object that under-reports its size
    between the two calls cannot slip through either.

    Returns the bytes, ``None`` when the object cannot be read, or
    ``STAGED_OVERSIZED`` when it exceeds ``cap``.
    """
    try:
        sized = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-s", f":0:{rel}"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if sized.returncode != 0:
        return None
    try:
        size = int(sized.stdout.strip())
    except ValueError:
        return None
    if size > cap:
        return STAGED_OVERSIZED
    try:
        with subprocess.Popen(
            ["git", "-C", str(root), "show", f":0:{rel}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as proc:
            assert proc.stdout is not None
            data = proc.stdout.read(cap + 1)
            proc.stdout.close()
            if proc.wait() != 0:
                return None
    except OSError:
        return None
    if len(data) > cap:
        return STAGED_OVERSIZED
    return data


def staged_mode(root: Path, rel: str) -> str | None:
    """The stage-0 file mode for ``rel`` (``"120000"`` for a symlink), or None.

    A gate that inspects staged CONTENT has to ask what kind of entry it is
    first: the blob behind a symlink entry is its target string, and reading
    it as file content is a category error the worktree path never makes
    because `is_symlink()` answers there.
    """
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-s", "-z", "--", rel],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if listed.returncode != 0 or not listed.stdout:
        return None
    # `<mode> <object> <stage>\t<path>\0`
    first = listed.stdout.split(b"\0")[0]
    head = first.split(b"\t", 1)[0].split()
    return head[0].decode("ascii", "replace") if head else None
