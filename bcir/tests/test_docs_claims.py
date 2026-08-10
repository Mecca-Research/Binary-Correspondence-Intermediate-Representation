"""The summary-claim checker itself, and the claims it currently guards.

`tools/docs/check_claims.py` exists because four documentation summaries were found stale in
one review pass — a roadmap row saying a refusal remained after it was built, the LangRef
saying ECN was only its built-in model and that JER had no C/MLIR/direct-claims rail, and the
JSON roadmap still listing two refusals as outstanding. The detailed prose stayed accurate in
every case; it was the summaries that rotted.

A checker that only ever passes is indistinguishable from one that does nothing, so this
drives its three failure modes as well as its success, on a COPY of the tree — the point is
that it fires, and a test that only asserted the happy path would have let a broken predicate
through exactly as the docs let a broken claim through.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CHECKER = os.path.join("tools", "docs", "check_claims.py")


def _run(cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, _CHECKER], cwd=cwd, capture_output=True,
                          text=True, timeout=180)


def _worktree(tmp: str) -> str:
    """A copy holding only what the checker reads: the docs, the sources it asks about."""
    root = os.path.join(tmp, "tree")
    for part in ("docs", "tools", "bcir/asn1", "mlir/include/BCIR", "runtime/c"):
        src = os.path.join(_ROOT, part)
        dst = os.path.join(root, part)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build"))
    # `bcir.asn1` is imported by a predicate, so the package roots have to exist.
    for pkg in ("bcir", os.path.join("bcir", "asn1")):
        init = os.path.join(root, pkg, "__init__.py")
        if not os.path.exists(init):
            open(init, "a").close()
    return root


def _edit(root: str, relpath: str, old: str, new: str) -> None:
    path = os.path.join(root, relpath)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    assert old in text, f"{relpath}: anchor not found, the test needs re-aiming"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.replace(old, new, 1))


def test_every_claim_the_docs_assert_is_true_today() -> None:
    """The live tree passes. This is the assertion the other tests give meaning to."""
    done = _run(_ROOT)
    assert done.returncode == 0, (
        f"a documentation summary no longer matches the code:\n{done.stderr}\n{done.stdout}")
    assert "still match the code" in done.stdout


def test_a_claim_that_stops_being_true_is_caught() -> None:
    """The mode that matters: the code moves and the summary does not.

    Simulated the way it would really happen — an ECN notation group returning to the
    refusal table — rather than by deleting the predicate. An earlier draft of this test
    inserted the regression BEFORE the real table was built, so the module overwrote it and
    the checker passed; the check was fine and the simulation was not.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = _worktree(tmp)
        with open(os.path.join(root, "bcir", "asn1", "ecn_syntax.py"), "a",
                  encoding="utf-8") as handle:
            handle.write('\n_UNSUPPORTED_KEYWORDS["CONTENTS-ENCODING"] = "simulated"\n')
        done = _run(root)
        assert done.returncode == 1, "a returning refusal must fail the check"
        assert "ecn-refusal-list-empty" in done.stderr
        assert "NO LONGER TRUE" in done.stderr
        # The message has to say what the code looks like NOW, or the reader investigates twice.
        assert "CONTENTS-ENCODING" in done.stderr


def test_a_document_that_drops_its_marker_is_caught() -> None:
    """A claim nobody asserts stops being checked, which is how one goes stale unnoticed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _worktree(tmp)
        _edit(root, os.path.join("docs", "BCIR_LANGREF.md"),
              "<!-- claim: jer-has-all-three-rails -->", "")
        done = _run(root)
        assert done.returncode == 1, "an unasserted claim must fail the check"
        assert "jer-has-all-three-rails" in done.stderr
        assert "NO document asserts it" in done.stderr


def test_a_marker_naming_no_predicate_is_caught() -> None:
    """A typo in a marker would otherwise disable the claim silently."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _worktree(tmp)
        _edit(root, os.path.join("docs", "BCIR_LANGREF.md"),
              "<!-- claim: ecn-three-parts-built -->", "<!-- claim: ecn-three-parts-buit -->")
        done = _run(root)
        assert done.returncode == 1, "an unknown claim name must fail the check"
        assert "has no predicate" in done.stderr


def test_the_twin_table_claim_reads_the_document_rather_than_a_hardcoded_list() -> None:
    """§17.2's table names its C twins, and the predicate must check what the TABLE says.

    A hard-coded list inside the checker would drift the same way the prose did, so the
    predicate parses the filenames out of the document. Adding a row for a twin that does not
    exist is the error worth catching, and this is what proves the reading is live.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = _worktree(tmp)
        _edit(root, os.path.join("docs", "BCIR_LANGREF.md"),
              "| X.690 | BER, CER, DER | DER | `bcir_asn1.c` |",
              "| X.690 | BER, CER, DER | DER | `bcir_asn1.c` |\n"
              "| X.999 | INVENTED | none | `bcir_invented.c` |")
        done = _run(root)
        assert done.returncode == 1, "a table row with no file behind it must fail the check"
        assert "asn1-c-twins-exist" in done.stderr
        assert "bcir_invented.c" in done.stderr


def test_the_registry_is_listable_without_running_the_predicates() -> None:
    """`--list` documents what is guarded, so adding a claim is discoverable."""
    done = subprocess.run([sys.executable, _CHECKER, "--list"], cwd=_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0
    for name in ("ecn-refusal-list-empty", "jer-has-all-three-rails",
                 "r25-covers-parameterization", "asn1-c-twins-exist",
                 "ecn-three-parts-built"):
        assert name in done.stdout, name
