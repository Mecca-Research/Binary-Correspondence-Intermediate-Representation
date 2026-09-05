"""The shared structural-law corpus (S0-6) -- the oracle runner, the drift gate on the law-rail
projection, and the cross-rail comparison when `bcir-opt` is built (laws.md L2/L11/L15).

`bcir/verify/structural_corpus.py` is the ONE corpus for the structural laws rows 9, 12, 13, 17,
19, 20 and 21 of the 2026-07/08 assessment named: every case is a rail-neutral spec with the
verdict each rail must reach. This module (the quick tier, every host) runs the oracle rail
over every case, refuses drift between the corpus and its committed `-verify-diagnostics`
projection (`mlir/test/passes/structural_corpus.mlir`, which `tools/wsl/check_passes.sh`
executes on every MLIR job), and -- when the toolchain is built -- drives `bcir-opt` over every
law-rail case and asserts zero findings. The comparison itself is witnessed: an injected
disagreement of every kind is a finding, never a pass.
"""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from bcir.verify import structural_corpus as sc  # noqa: E402

_FAMILIES = {"", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R12", "R13", "R21", "R22"}


def test_the_corpus_is_non_trivial_and_declares_every_rail() -> None:
    """L2: a corpus that examined nothing would pass everything. Every kind, both rails, every
    law family the slice closes, and both a legal and an illegal case per family."""
    assert len(sc.CASES) >= 80
    kinds = {c.kind for c in sc.CASES}
    assert kinds == {
        "module",
        "map",
        "rop",
        "target",
        "mmio",
        "manifest",
        "portfolio",
        "calibration",
        "binary",
        "stream",
        "fsm",
        "grammar",
        "conv",
    }, kinds
    assert {c.law for c in sc.CASES} == _FAMILIES
    assert sum(1 for c in sc.CASES if sc.ORACLE in c.rails) >= 75
    assert sum(1 for c in sc.CASES if sc.MLIR in c.rails) >= 75
    assert len({c.name for c in sc.CASES}) == len(sc.CASES), "case names are unique"
    for c in sc.CASES:
        assert c.rails and set(c.rails) <= set(sc.RAILS), c.name
        if c.law:
            if sc.ORACLE in c.rails:
                assert c.oracle, f"{c.name}: an illegal oracle case names its diagnostic"
            if sc.MLIR in c.rails:
                assert c.mlir, f"{c.name}: an illegal law-rail case names its diagnostic"
        else:
            assert not (c.oracle or c.mlir), f"{c.name}: a legal case carries no diagnostic"
        if set(c.rails) != set(sc.RAILS):
            assert c.note, f"{c.name}: a single-rail case says why"


def test_the_oracle_rail_reaches_every_corpus_verdict() -> None:
    """The Python runner over every oracle-rail case: no finding of any kind."""
    found = []
    for case in sc.CASES:
        if sc.ORACLE not in case.rails:
            continue
        found += sc.findings(replace(case, rails=(sc.ORACLE,)), {sc.ORACLE: sc.run_oracle(case)})
    assert found == [], "\n".join(f"{f.rail} {f.kind} {f.case}: {f.detail}" for f in found)


def test_the_committed_fixture_is_the_emitted_corpus() -> None:
    """L15: the law-rail projection check_passes.sh executes is exactly what the corpus emits."""
    assert sc.check_fixture() == []
    text = (_ROOT / sc.FIXTURE).read_text(encoding="utf-8")
    assert text.startswith("// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s")
    law_rail = [c for c in sc.CASES if sc.MLIR in c.rails]
    assert text.count("// -----") == len(law_rail)  # one per case, after the header
    assert text.count("expected-error @+") == sum(len(c.mlir_expected) for c in law_rail)


def test_every_law_rail_case_renders_its_diagnostic_line_once() -> None:
    """An illegal law-rail case attaches its expected-error to exactly one line; a legal one to none."""
    for case in sc.CASES:
        if sc.MLIR not in case.rails:
            continue
        rendered = sc.render(case, expectations=True)
        assert rendered.count("expected-error") == len(case.mlir_expected), case.name
        assert bool(case.mlir_expected) == bool(case.law), case.name
        assert "\x00" not in rendered, case.name


def test_the_law_rail_agrees_with_the_corpus_when_built() -> None:
    """Drive bcir-opt over every law-rail case (self-skips without the toolchain)."""
    bo = sc.find_bcir_opt()
    if not bo:
        return
    found, counts = sc.run(bo)
    assert counts["mlir"] >= 75
    assert found == [], "\n".join(f"{f.rail} {f.kind} {f.case}: {f.detail}" for f in found)


def test_findings_name_every_kind_of_disagreement() -> None:
    """L2: the comparison can fail. Each way a rail can disagree with the corpus is a finding."""
    legal = next(c for c in sc.CASES if c.legal and c.rails == sc.RAILS)
    illegal = next(c for c in sc.CASES if c.law == "R7" and c.rails == sc.RAILS)
    refused = sc.Verdict(sc.ORACLE, True, ("R7: something",), ("R7",))
    admitted = sc.Verdict(sc.ORACLE, False)
    wrong_message = sc.Verdict(sc.ORACLE, True, ("R7: another reason",), ("R7",))
    wrong_law = sc.Verdict(sc.ORACLE, True, (f"R2: {illegal.oracle}",), ("R2",))
    kinds = lambda case, v: {f.kind for f in sc.findings(case, {sc.ORACLE: v, sc.MLIR: v})}  # noqa: E731
    assert kinds(legal, refused) == {"refused-legal"}
    assert kinds(illegal, admitted) == {"admitted-illegal"}
    assert kinds(illegal, wrong_message) == {"message"}
    assert kinds(illegal, wrong_law) == {"law"}
    assert kinds(illegal, sc.Verdict(sc.ORACLE, True, (f"R7: {illegal.oracle}",), ("R7",))) == set()
    assert {f.kind for f in sc.findings(illegal, {})} == {"no-verdict"}


def test_the_pointer_width_table_is_the_one_the_corpus_exercises() -> None:
    """The address-width cases are generated from the oracle's triple -> width table; the table
    knows every profile triple the repository ships and refuses to guess an unknown one."""
    from bcir.kbcir import TARGETS
    from bcir.kbcir.cost import pointer_width

    for name, profile in TARGETS.items():
        assert pointer_width(profile.triple) in (32, 64), (name, profile.triple)
    assert pointer_width("x") is None and pointer_width("") is None
    triples = {c.spec["triple"] for c in sc.CASES if c.kind == "mmio" and c.spec["triple"]}
    assert {pointer_width(t) for t in triples} == {64, 32, None}


def test_the_cli_reports_the_verdict() -> None:
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert sc.main(["--check"]) == 0
    assert "fixture current" in out.getvalue()
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = sc.main(["--run"])  # the built bcir-opt when present, else the oracle rail alone
    text = out.getvalue()
    assert text.startswith("structural_corpus: ") and "findings 0" in text, text
    assert rc == 0


def test_a_declared_additional_law_is_required_not_merely_allowed() -> None:
    """`also_laws` names the further families BOTH rails must report. A rail that drops one --
    the R5 volatile-hazard diagnostic beside the R3 domain refusal -- passed the subset check;
    it is a `law` finding, and the complete verdict is not (#762 review)."""
    case = next(c for c in sc.CASES if c.also_laws)
    oracle_only = replace(case, rails=(sc.ORACLE,))
    partial = sc.Verdict(sc.ORACLE, True, (f"R3: {case.oracle}",), ("R3",))
    found = sc.findings(oracle_only, {sc.ORACLE: partial})
    assert [f.kind for f in found] == ["law"], found
    assert "requires ['R3', 'R5']" in found[0].detail
    full = sc.Verdict(sc.ORACLE, True, (f"R3: {case.oracle}", "R5: ordered"), ("R3", "R5"))
    assert sc.findings(oracle_only, {sc.ORACLE: full}) == []
    extra = sc.Verdict(sc.ORACLE, True, (f"R3: {case.oracle}",), ("R3", "R5", "R7"))
    assert [f.kind for f in sc.findings(oracle_only, {sc.ORACLE: extra})] == ["law"]


_SUPPORT_H = Path("mlir/lib/passes/BCIRPassSupport.h")
_TABLE_ENTRY = re.compile(r'\{"([a-z0-9_]+)",\s*(\d+)\}')


def _law_rail_table(name: str) -> list[tuple[str, int]]:
    """The `{"arch", bits}` pairs of one initializer of pointerWidthOfTriple, read out of the
    C++ source itself (L14: never a third copy)."""
    text = (_ROOT / _SUPPORT_H).read_text(encoding="utf-8")
    start = text.index(f"{name}[] = {{")
    end = text.index("};", start)
    return [(arch, int(bits)) for arch, bits in _TABLE_ENTRY.findall(text[start:end])]


def test_the_pointer_width_tables_are_one_table_on_both_rails() -> None:
    """`pointer_width` and `pointerWidthOfTriple` are read out of their own sources and must
    agree entry for entry, in order (L14: a mirror list WILL drift; the corpus's cases exercise
    a handful of triples, this exercises every row). And no row is below the address floor:
    a 16-bit contract could be met by no operand the op floor admits, so R12 refused every
    address on `avr`/`msp430` -- a row no input can satisfy is a defect, not strictness (L22);
    the watchOS ILP32 ABI (`arm64_32`) is 32-bit, which the `arm64` family prefix used to
    claim as 64 (#762 review)."""
    from bcir.kbcir import cost

    assert _law_rail_table("kExact") == list(cost._POINTER_BITS.items())
    assert _law_rail_table("kPrefixes") == list(cost._POINTER_BITS_PREFIX)
    assert len(cost._POINTER_BITS) >= 50 and len(cost._POINTER_BITS_PREFIX) >= 5
    floor = cost.ADDRESS_FLOOR_BITS
    assert floor == 32
    assert f"kAddressFloorBits = {floor};" in (_ROOT / _SUPPORT_H).read_text(encoding="utf-8")
    assert all(bits >= floor for bits in cost._POINTER_BITS.values())
    assert all(bits >= floor for _, bits in cost._POINTER_BITS_PREFIX)
    assert cost.pointer_width("arm64_32-apple-watchos") == 32
    assert cost.pointer_width("aarch64_32-apple-watchos") == 32
    assert cost.pointer_width("arm64e-apple-ios") == 64
    assert cost.pointer_width("avr-unknown-unknown") is None  # below the floor: no contract
    assert cost.pointer_width("msp430-none-elf") is None
    assert cost.pointer_width("spirv") is None  # logical addressing: no pointer width


def test_the_address_floor_is_a_law_on_the_oracle_too() -> None:
    """The law rail's op verifiers refuse an address narrower than 32 bits whatever the target;
    the oracle used to accept i16 under an unknown triple (vacuous), a rail disagreement the
    corpus never exercised. Now `addr.i16_below_the_floor` runs on both rails."""
    from bcir.verify import verify_address_width

    assert [d.law for d in verify_address_width("x", 16)] == ["R12"]
    assert verify_address_width("x", 32) == []
    assert [d.law for d in verify_address_width("arm64_32-apple-watchos", 64)] == ["R12"]
    assert verify_address_width("arm64_32-apple-watchos", 32) == []
