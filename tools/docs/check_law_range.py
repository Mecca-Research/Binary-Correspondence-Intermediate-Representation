#!/usr/bin/env python3
"""Fail CI when an active document states a verifier-law range that has drifted from the tree.

    python tools/docs/check_law_range.py            # verify every Markdown file
    python tools/docs/check_law_range.py --list     # show the derived last law and the scan scope

WHY THIS EXISTS. The verifier rail grew R19-R21, then R22/R23, then R24, then R25, and each
time a "last law" moved, prose written against the previous one stayed behind: on 2026-09-03
about twenty active documents still said "R1-R23" (or R1-R24, R14-R23) while the generated
`docs/STATUS.md` said R1-R25. Summaries are written once and reread by nobody; the same
shape of drift that made STATUS.md a generated file (the 580/615/631 counts). This gate
derives the last law from the tree the same way `gen_status.py` does and refuses a range
that names a former last law as its upper bound.

WHAT IS FLAGGED. Exactly the two drift shapes that occurred, and nothing else:

  * a whole-rail claim ``R1-RN`` whose N is above the C-front's fixed ``R1-R18`` scope and
    below the current last law (so ``R1-R18`` for the frontend twin, ``R1-R12`` for the
    legality core and ``R14-R16`` for the smart-lowering laws are fixed subsystem scopes
    and never flagged; ``R1-R23`` is a stale whole-rail claim and is);
  * the non-disturbance form ``R14-RN`` whose N is below the current last law (new optional
    laws ship vacuous-by-default over the whole R14.. tail, so that range must end at the
    last law).

A range ending at the current last law is correct by construction and needs no marker; when
the next law lands, every such range fails at once, which is the point.

WHEN A RANGE IS DELIBERATE. A line that quotes a historical state carries a line marker, a
document that is a dated snapshot carries a file marker:

    ... R1-R23 ... <!-- law-scope: as of the 2026-07-15 audit -->
    <!-- allow-law-ranges -->        (anywhere in the file; dated audits and the history)

Any hyphen, en dash, em dash, figure dash or minus sign between the two law names is
accepted, because the documents use all of them.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SKIP_DIRS = (".git", "__pycache__", "node_modules", ".mypy_cache", "build", "dist")
LINE_MARKER = "law-scope:"
FILE_MARKER = "allow-law-ranges"
# The frontend twin's documented scope is fixed at R1-R18; ranges ending at or below it are
# subsystem scopes, never a claim about the whole rail.
_FIXED_SCOPE_CEILING = 18
_RANGE = re.compile(r"\bR(\d{1,2})\s*[‐‑‒–—―−-]\s*R(\d{1,2})\b")


def last_law() -> int:
    """The tree's last verifier law, derived exactly as generated STATUS.md derives it."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_status import verifier_laws  # noqa: PLC0415

    laws = verifier_laws()
    return int(laws[-1][0][1:])


def md_files(root: str) -> list[str]:
    out: list[str] = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in _SKIP_DIRS]
        out += [os.path.join(dp, f) for f in fn if f.endswith(".md")]
    return sorted(out)


def is_stale(lo: int, hi: int, last: int) -> bool:
    """True for the two drift shapes: a whole-rail ``R1-RN`` or a non-disturbance ``R14-RN``
    whose N lies strictly between the fixed subsystem ceiling (R18) and the current last law.
    ``R14-R16`` (smart lowering) and ``R14-R18`` are fixed groups and pass; ``R14-R23`` is a
    former last law and fails."""
    if hi >= last or hi <= _FIXED_SCOPE_CEILING:
        return False
    return lo in (1, 14)


def offenders(root: str, last: int) -> list[tuple[str, int, str, str]]:
    bad: list[tuple[str, int, str, str]] = []
    for md in md_files(root):
        try:
            text = open(md, encoding="utf-8").read()
        except OSError:
            continue
        if FILE_MARKER in text:
            continue
        rel = os.path.relpath(md, root)
        for i, line in enumerate(text.splitlines(), 1):
            if LINE_MARKER in line:
                continue
            for m in _RANGE.finditer(line):
                lo, hi = int(m.group(1)), int(m.group(2))
                if is_stale(lo, hi, last):
                    bad.append((rel, i, m.group(0), line.strip()[:100]))
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT, help="tree to scan (tests point this at a fixture)")
    ap.add_argument("--last-law", type=int, default=None,
                    help="override the derived last law (tests only)")
    ap.add_argument("--list", action="store_true", help="print the derived last law and scope")
    args = ap.parse_args(argv)
    last = args.last_law if args.last_law is not None else last_law()
    files = md_files(args.root)
    if args.list:
        print(f"[check_law_range] last law R{last}; scanning {len(files)} Markdown files under "
              f"{os.path.relpath(args.root, os.getcwd()) or '.'}")
        print(f"[check_law_range] line marker '<!-- {LINE_MARKER} ... -->', "
              f"file marker '<!-- {FILE_MARKER} -->'")
    bad = offenders(args.root, last)
    if bad:
        sys.stderr.write(f"[check_law_range] {len(bad)} stale law range(s); the tree's last law "
                         f"is R{last}. Fix the range, or mark a deliberate historical quote with "
                         f"'<!-- {LINE_MARKER} <reason> -->' on the line (or '<!-- {FILE_MARKER} "
                         f"-->' in a dated snapshot):\n")
        for rel, i, rng, line in bad:
            sys.stderr.write(f"  {rel}:{i}: {rng}    {line}\n")
        return 1
    sys.stderr.write(f"[check_law_range] no stale verifier-law ranges (last law R{last}, "
                     f"{len(files)} files).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
