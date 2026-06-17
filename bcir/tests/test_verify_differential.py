"""The verifier law-for-law differential -- both rails, systematically.

Roadmap (BCIR_MASTER_ROADMAP.md §5.1 / §6): the C++ `-bcir-verify` law-for-law
differential on the toolchain rail. Two halves:

  * **oracle rail** -- `differential.run_verifier_campaign` fault-injects a law into a
    clean module and confirms the Python `verify` flags exactly that law (already shipped);
    here we assert it is clean and covers its injected set.
  * **toolchain rail** -- every law R1..R17 must have a negative `-verify-diagnostics`
    case in the committed `verify_laws*.mlir` / `verify_accuracy.mlir`, so `bcir-opt
    -bcir-verify` is differentially tested law-for-law (the cases run under bcir-opt in CI,
    `tools/wsl/check_passes.sh`). This test is the *coverage gate*: it guarantees no law
    silently loses its toolchain-rail negative case -- locally checkable by parsing the
    committed `.mlir` (no bcir-opt needed).
"""

import os
import re

from bcir.kbcir.differential import run_verifier_campaign

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PASSES = os.path.join(_ROOT, "mlir", "test", "passes")
_FILES = ("verify_laws.mlir", "verify_laws_deep.mlir", "verify_accuracy.mlir",
          "verify_callgraph.mlir")
_ALL_LAWS = tuple(f"R{i}" for i in range(1, 19))   # R1 .. R18 (R18 = call-graph integrity)


def _laws_with_negative_cases() -> set:
    """The set of laws that have a negative `expected-error {{Rxx: ...}}` case in the
    committed verifier-law .mlir files (the toolchain-rail differential corpus)."""
    found: set[str] = set()
    pat = re.compile(r"expected-error[^\n]*\{\{(R\d+):")
    for name in _FILES:
        with open(os.path.join(_PASSES, name), encoding="utf-8") as f:
            found |= set(pat.findall(f.read()))
    return found


def test_every_law_has_a_toolchain_negative_case():
    """R1..R18 are each negatively tested by `bcir-opt -bcir-verify -verify-diagnostics`."""
    have = _laws_with_negative_cases()
    missing = [law for law in _ALL_LAWS if law not in have]
    assert not missing, f"laws lacking a toolchain-rail negative case: {missing}"


def test_r17_negative_case_is_committed():
    """The new accuracy law specifically -- a regression guard on the R17 .mlir."""
    with open(os.path.join(_PASSES, "verify_accuracy.mlir"), encoding="utf-8") as f:
        text = f.read()
    assert "RUN: bcir-opt -bcir-verify -verify-diagnostics" in text
    assert "R17: accuracy bound 1000 ULP exceeds tolerance 1 ULP" in text
    assert "exact = true" in text and "exact = false" in text   # both poles present


def test_oracle_verifier_differential_is_clean():
    """The Python half: fault-inject each law into a clean base and confirm the oracle
    verifier flags it (zero misses)."""
    misses = run_verifier_campaign(n=200, seed=3)
    assert misses == [], f"oracle verifier missed: {[m.detail for m in misses]}"


def test_run_line_uses_split_input_file():
    """Each verifier-law file drives bcir-opt with -split-input-file so every module is an
    independent negative case (the differential granularity)."""
    for name in _FILES:
        with open(os.path.join(_PASSES, name), encoding="utf-8") as f:
            first = f.readline()
        assert "-bcir-verify -verify-diagnostics -split-input-file" in first, name
