#!/usr/bin/env python3
"""Fail CI when a documentation SUMMARY claim stops matching the thing it summarises.

    python tools/docs/check_claims.py            # verify every annotated claim
    python tools/docs/check_claims.py --list     # show the registry and what each asserts

WHY THIS EXISTS. Four summary claims were found stale in a single review pass, all the same
shape: a sentence written before the work landed and never revisited when it did.

  * the buildout roadmap's X.692 row said "One refusal remains" after that refusal was built;
  * the LangRef said the oracle implements "ECN's built-in model" after all three parts landed;
  * it also said JER "has no C/MLIR/direct-claims rail yet" after J3, J4 and J5 built all three;
  * the JSON roadmap's ECN row still ended "Still refused: ... §21.3.6's `container`".

The detailed sections stayed accurate, because they were edited alongside the work. It is the
SUMMARIES that rot -- they are written once, read often, and touched by nobody when the thing
they summarise changes. A human reread finds them eventually; this finds them the same day.

HOW IT WORKS. A summary sentence whose truth depends on the code carries a marker:

    <!-- claim: ecn-refusal-list-empty -->

and this file holds a predicate of the same name that asks the CODE whether the claim holds.
Three failures are reported, and the third is the one that catches the drift:

  1. a marker naming a claim that is not in the registry (a typo, or a deleted predicate);
  2. a registered claim that NO document references -- either the predicate is dead weight, or
     a doc was edited and quietly lost its marker, which is exactly how a claim goes unchecked;
  3. a claim whose predicate is now false -- the summary and the code have diverged.

WHAT BELONGS HERE. Only claims a machine can settle: a file exists, a table is empty, a symbol
is defined. A predicate that needs judgement does not belong -- an approximate check that fails
on correct prose would train everyone to ignore it, which is worse than no check at all.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SKIP_DIRS = (".git", "__pycache__", "node_modules", ".mypy_cache", "build")
_MARKER = re.compile(r"<!--\s*claim:\s*([a-z0-9][a-z0-9-]*)\s*-->")


# --- the things a predicate needs to ask ---------------------------------------------------


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


def _exists(*parts: str) -> bool:
    return os.path.exists(os.path.join(ROOT, *parts))


# --- the claims ----------------------------------------------------------------------------
#
# Each returns (ok, detail). `detail` is printed on failure and must say what the code
# ACTUALLY looks like now, not merely that the claim failed -- the point is to make the
# correction obvious without a second investigation.


def _claim_ecn_refusal_list_empty() -> tuple[bool, str]:
    """X.692's unbuilt-group refusal list is empty: every notation group is implemented."""
    sys.path.insert(0, ROOT)
    from bcir.asn1.ecn_syntax import SYNTAX_VERSION, _UNSUPPORTED_KEYWORDS

    rows = sorted(_UNSUPPORTED_KEYWORDS)
    return (
        not rows,
        f"_UNSUPPORTED_KEYWORDS holds {len(rows)} row(s) at SYNTAX_VERSION "
        f"{SYNTAX_VERSION}: {rows}",
    )


def _claim_ecn_three_parts_built() -> tuple[bool, str]:
    """All three parts of ECN are built: the model, the user-defined half, the syntax."""
    parts = {
        "part 1 (model, EDM/ELM, built-in sets)": ("bcir", "asn1", "ecn.py"),
        "part 2 (user-defined encodings)": ("bcir", "asn1", "ecn_user.py"),
        "part 3 (clause 20 defined syntax)": ("bcir", "asn1", "ecn_syntax.py"),
    }
    missing = [name for name, path in parts.items() if not _exists(*path)]
    return not missing, f"missing: {missing}" if missing else "all three modules present"


def _claim_jer_has_all_three_rails() -> tuple[bool, str]:
    """JER has a C twin, an MLIR rule family and a direct-claims plan -- J3, J4 and J5.

    Pinned because the LangRef said the opposite long after these landed, and "has no rail
    yet" is the kind of sentence a reader takes at face value.
    """
    rails = {
        "C twin": _exists("runtime", "c", "bcir_jer.c"),
        "MLIR asn1_rules family": '"jer"' in _read("mlir", "include", "BCIR", "BCIRAttrs.td"),
        "direct-claims plan": _exists("bcir", "asn1", "jer_plan.py"),
    }
    absent = [name for name, present in rails.items() if not present]
    return not absent, f"absent: {absent}" if absent else "all three rails present"


def _claim_r25_covers_parameterization() -> tuple[bool, str]:
    """R25's ECN dialect carries an operation for Annex C's dummy parameters."""
    td = _read("mlir", "include", "BCIR", "BCIREcnOps.td")
    ok = "BCIR_EcnParameterizedOp" in td and '"ecn.parameterized"' in td
    return ok, (
        "BCIREcnOps.td defines BCIR_EcnParameterizedOp"
        if ok
        else "BCIREcnOps.td has no BCIR_EcnParameterizedOp / ecn.parameterized op"
    )


def _claim_asn1_c_twins_exist() -> tuple[bool, str]:
    """Every C twin the LangRef's transfer-syntax table names is a file that exists.

    This one cross-references a TABLE rather than a sentence: the table's last column lists
    twins by filename, and a reader will believe it. Reading the names out of the document
    rather than restating them here is deliberate -- a hard-coded list would drift the same
    way the prose did, and a row added to the table with no file behind it is precisely the
    error worth catching.
    """
    langref = _read("docs", "BCIR_LANGREF.md")
    section = langref.split("### 17.2", 1)
    if len(section) != 2:
        return False, "LangRef §17.2 not found; the twin table moved and this claim needs re-aiming"
    table = section[1].split("###", 1)[0]
    named = sorted(set(re.findall(r"`(bcir_[a-z0-9_]+\.c)`", table)))
    if not named:
        return False, "LangRef §17.2 names no `bcir_*.c` twin; the table shape changed"
    missing = [name for name in named if not _exists("runtime", "c", name)]
    return not missing, (
        f"named but absent: {missing}"
        if missing
        else f"all {len(named)} named twins exist: {named}"
    )


_CLAIMS = {
    "ecn-refusal-list-empty": _claim_ecn_refusal_list_empty,
    "ecn-three-parts-built": _claim_ecn_three_parts_built,
    "jer-has-all-three-rails": _claim_jer_has_all_three_rails,
    "r25-covers-parameterization": _claim_r25_covers_parameterization,
    "asn1-c-twins-exist": _claim_asn1_c_twins_exist,
}


# --- the scan ------------------------------------------------------------------------------


def _md_files() -> list[str]:
    out: list[str] = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        out.extend(os.path.join(base, f) for f in files if f.endswith(".md"))
    return sorted(out)


def _markers() -> dict[str, list[str]]:
    """claim name -> the documents that assert it."""
    found: dict[str, list[str]] = {}
    for path in _md_files():
        with open(path, encoding="utf-8") as handle:
            for name in _MARKER.findall(handle.read()):
                found.setdefault(name, []).append(os.path.relpath(path, ROOT))
    return found


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for name, fn in sorted(_CLAIMS.items()):
            summary = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"{name}\n    {summary}")
        return 0

    asserted = _markers()
    failures: list[str] = []

    for name in sorted(set(asserted) - set(_CLAIMS)):
        failures.append(
            f"{', '.join(asserted[name])}: claim {name!r} has no predicate in "
            f"tools/docs/check_claims.py -- add one, or fix the marker's spelling"
        )

    for name in sorted(set(_CLAIMS) - set(asserted)):
        failures.append(
            f"claim {name!r} is registered but NO document asserts it; a doc that dropped its "
            f"marker stops being checked, which is how these go stale -- restore the marker or "
            f"retire the predicate"
        )

    for name in sorted(set(_CLAIMS) & set(asserted)):
        ok, detail = _CLAIMS[name]()
        if not ok:
            failures.append(
                f"{', '.join(asserted[name])}: claim {name!r} is NO LONGER TRUE -- {detail}"
            )

    if failures:
        sys.stderr.write("[check_claims] summary claims that no longer match the code:\n")
        for line in failures:
            sys.stderr.write(f"  {line}\n")
        return 1

    print(f"[check_claims] {len(_CLAIMS)} summary claim(s) still match the code they summarise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
