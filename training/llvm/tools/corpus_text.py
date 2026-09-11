"""What counts as corpus prose, and what counts as code inside it.

Both surface gates -- `verify-langref-delta.py` for LLVM's LangRef and
`verify-mlir-coverage.py` for MLIR's dialects -- need the same two predicates, and both
got them wrong in the same two ways first. They live here so there is one answer.

**A citation must be code, not prose.** Many of the names these gates track are ordinary
English words: the `range` attribute, the `shape`, `index`, `math`, `complex`, `vector`
and `func` dialects. A substring search for them matches sentences that have nothing to
do with the construct, so a chapter can be credited with teaching `range` because it used
the word. Only text the author wrote as code counts.

Extracting that is where the second mistake lives. A regex like ``r"`[^`]*\\bindex\\."``
looks like it matches an inline code span, but it has no idea which backticks open and
which close -- so in ``a constant `i32` field index.`` it happily starts at the *closing*
backtick of ``i32`` and matches " field index.". Inline spans have to be paired, and
fenced blocks have to be removed first so a backtick inside one cannot open a span.

**Tooling is not teaching material.** The corpus's own `.py` files are excluded from the
text these gates measure. Python's standard library collides with MLIR's dialect
namespaces head-on: `math.sqrt`, `math.floor` and `math.ceil` are real `math` dialect
operations *and* real Python calls, and a benchmark script importing `math` made the
`math` dialect look a sixth covered when the corpus does not mention it at all.
"""

from __future__ import annotations

import re
from pathlib import Path

# Fenced blocks first, then inline spans from what is left.
_FENCED = re.compile(r"```.*?```", re.DOTALL)
_INLINE = re.compile(r"`[^`\n]+`")

# Documents and examples a reader learns from. `.py` is deliberately absent: see above.
TEACHING_SUFFIXES = frozenset({".md", ".mlir", ".ll", ".json", ".td", ".cpp", ".h", ".c"})

# Files that enumerate a surface by construction. Measuring coverage over them reports
# every name as covered -- the first run of the LangRef measurement returned 100% on all
# three surfaces for exactly this reason.
#
# This is a RULE rather than a list, and it is a rule because the list failed twice. The
# names were hand-maintained, so adding `langref-attribute-dispositions.json` -- a file
# whose entire purpose is to name the attributes the corpus does not name -- silently
# dropped the measured gap from 32 attributes to 1. A mirror list drifts; a predicate
# over the naming convention does not, and every surface table here already follows one.
_SELF_LISTING_SUFFIXES = ("-dispositions.json", "-surface-23.json", "-surface-18.json")


def is_self_listing(path: Path) -> bool:
    """True for a reference table that enumerates the surface it is measured against."""
    return path.parent.name == "reference" and path.name.endswith(_SELF_LISTING_SUFFIXES)


def code_spans(text: str) -> str:
    """Everything in a markdown document written as code, concatenated.

    Fenced blocks are taken whole and removed before inline spans are matched, so a
    backtick inside a fence cannot open an inline span that swallows prose after it.
    """
    fenced = _FENCED.findall(text)
    prose = _FENCED.sub("\n", text)
    return "\n".join(fenced) + "\n" + "\n".join(_INLINE.findall(prose))


def teaching_files(root: Path, extra_exclusions: frozenset[str] = frozenset()) -> list[Path]:
    """Every file under `root` a reader learns from, in a stable order."""
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in TEACHING_SUFFIXES
        and not is_self_listing(path)
        and path.name not in extra_exclusions
    ]


def teaching_text(root: Path, extra_exclusions: frozenset[str] = frozenset()) -> str:
    return "\n".join(
        path.read_text(errors="replace") for path in teaching_files(root, extra_exclusions)
    )
