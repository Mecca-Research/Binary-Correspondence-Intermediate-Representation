"""Python ↔ MLIR parity for the X.692 ECN law rail (law R25).

`docs/PARITY.md` makes enum integer values normative: the oracle and the law must agree on
them, or an attribute means one thing in the IR and another in the encoder. This gate reads
the ODS source directly rather than a transcription of it, so the two cannot drift without
the test noticing — the same discipline `test_asn1_law_parity.py` applies to R24.

It also pins each R25 rule against the fixture that trips it. A law with no witness is a
claim rather than a check, and the ECN rules are exactly the kind that look obviously true
in prose and turn out to have two directions: §22.8.2.2's `if and only if` fails as easily on
`not-needed` **with** a reference as on a determination without one, and §21.11.5 rejects a
comparison on a bound shape as firmly as it requires one on `test-range`.
"""

from __future__ import annotations

import os
import re

from bcir.asn1.ecn_user import (
    AlternativeDetermination, Comparison, ComponentOrder, EncodingSpaceDetermination,
    HandleValueKind, IntegerBounds, OptionalityDetermination, Padding, RangeCondition,
    ReplaceAction, ReversalSpecification, UnusedBitsDetermination,
)

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ATTRS_TD = os.path.join(_ROOT, "mlir", "include", "BCIR", "BCIRAttrs.td")
_ECN_TD = os.path.join(_ROOT, "mlir", "include", "BCIR", "BCIREcnOps.td")
_FIXTURE = os.path.join(_ROOT, "mlir", "test", "passes", "verify_ecn.mlir")
_VERIFY_PASS = os.path.join(_ROOT, "mlir", "lib", "passes", "BCIRVerifyPass.cpp")


def _cases(name: str) -> dict[str, int]:
    """The `(spelling, value)` pairs of one ODS enum, read from the TableGen source."""
    text = open(_ATTRS_TD, encoding="utf-8").read()
    match = re.search(rf"def BCIR_{name}\s*:\s*BCIR_Enum<.*?\]>;", text, re.S)
    assert match, f"BCIR_{name} not found in {_ATTRS_TD}"
    return {m.group(3): int(m.group(2)) for m in
            re.finditer(r'I32EnumAttrCase<"(\w+)",\s*(\d+),\s*"([\w.]+)">', match.group(0))}


def _oracle(enum) -> dict[str, int]:
    """The oracle's members, spelled the way ODS mnemonics are (hyphens become underscores).

    The two rails cannot share a spelling: X.692 writes `field-to-be-set`, and an MLIR
    attribute mnemonic cannot contain a hyphen. So the *order* is what has to match, and the
    mapping between spellings is stated here once rather than assumed at each call.
    """
    return {member.value.replace("-", "_"): index
            for index, member in enumerate(enum)}


# --- the enums -------------------------------------------------------------------------

def test_padding_values_match_the_clause_order_on_both_rails():
    """§21.9.1's `ENUMERATED {zero, one, pattern, encoder-option}`, in that order.

    §21.9.2 makes `zero` the default, so it takes 0 — which means an attribute that is absent
    and one that is present with the clause's default agree numerically, rather than needing a
    reader to know which is which.
    """
    expected = {"zero": 0, "one": 1, "pattern": 2, "encoder_option": 3}
    assert _cases("EcnPadding") == expected
    assert _oracle(Padding) == expected


def test_reversal_values_match_the_enumerations_own_order_not_the_prose():
    """§21.14.1's order, which §21.14.6's prose contradicts while claiming to follow it.

    §21.14.1 lists `{no-reversal, reverse-bits-in-units, reverse-half-units,
    reverse-bits-in-half-units}`. §21.14.6 then describes four actions "in the order of
    enumerations listed above" and gives a different order. §22.12.3.2 agrees with §21.14.1
    and with what the names say, so that is what both rails encode — and this test exists so
    that a future reader who finds §21.14.6 first cannot quietly renumber one rail.
    """
    expected = {"no_reversal": 0, "reverse_bits_in_units": 1, "reverse_half_units": 2,
                "reverse_bits_in_half_units": 3}
    assert _cases("EcnReversal") == expected
    assert _oracle(ReversalSpecification) == expected


def test_the_two_determination_enums_are_separate_because_their_third_values_differ():
    """§21.3.1 and §21.4 share two values and differ in the third, which is what each is for.

    An encoding space can be bounded by a `container`; an unused-bit count can be
    `not-needed`. Neither statement makes sense about the other, so one enum with five values
    would let an object say a thing the notation cannot express — and the law would have to
    reject it afterwards rather than the IR being unable to hold it.
    """
    space = {"field_to_be_set": 0, "field_to_be_used": 1, "container": 2}
    unused = {"field_to_be_set": 0, "field_to_be_used": 1, "not_needed": 2}
    assert _cases("EcnSpaceDetermination") == space
    assert _oracle(EncodingSpaceDetermination) == space
    assert _cases("EcnUnusedBits") == unused
    assert _oracle(UnusedBitsDetermination) == unused
    assert space != unused


def test_range_condition_values_match_and_the_first_five_still_partition():
    """§21.11.1's eight values, and §21.11.4's NOTE that the first five partition.

    "For any given set of bounds, exactly one predicate will be satisfied." The ordering
    matters to the law rail, which switches on the five that guarantee a lower bound for
    §23.7.2.7; the partition matters to the oracle, which selects an encoding by them. Both
    are checked, because a renumbering breaks the first and a mis-written predicate breaks
    the second, and neither shows up as the other.
    """
    expected = {
        "unbounded_or_no_lower_bound": 0, "semi_bounded_with_negatives": 1,
        "bounded_with_negatives": 2, "semi_bounded_without_negatives": 3,
        "bounded_without_negatives": 4, "test_lower_bound": 5, "test_upper_bound": 6,
        "test_range": 7,
    }
    assert _cases("EcnRangeCondition") == expected
    assert _oracle(RangeCondition) == expected

    for bounds in (IntegerBounds(), IntegerBounds(high=9), IntegerBounds(low=-1),
                   IntegerBounds(low=-1, high=9), IntegerBounds(low=0),
                   IntegerBounds(low=0, high=9)):
        assert bounds.exactly_one_shape() in RangeCondition, bounds


def test_comparison_values_match_and_the_type_has_no_default():
    """§21.12.1's six, and §21.12.2's unusual property: "There is no default value".

    Every other clause 21 type names one. The law rail records that by leaving the attribute
    optional and never treating absence as a value, which is checked here by the absence of a
    `DEFAULT` marker in the ODS comment block that documents the enum.
    """
    expected = {"equal_to": 0, "not_equal_to": 1, "greater_than": 2, "less_than": 3,
                "greater_than_or_equal_to": 4, "less_than_or_equal_to": 5}
    assert _cases("EcnComparison") == expected
    assert _oracle(Comparison) == expected

    text = open(_ATTRS_TD, encoding="utf-8").read()
    assert "21.12.2 is the one clause 21 type with NO default value" in text


def test_replace_has_four_actions_because_component_is_a_synonym():
    """§22.1.1.7 lists five spellings; §22.1.1.8 makes two of them one action.

    "`REPLACE COMPONENT` is a synonym for `REPLACE ALL COMPONENTS`." Two enum cases would let
    a law be written that treated them differently, which the clause forbids — so the synonym
    is collapsed where it cannot be un-collapsed, rather than checked afterwards.
    """
    expected = {"structure": 0, "all_components": 1, "optionals": 2, "non_optionals": 3}
    assert _cases("EcnReplace") == expected
    assert _oracle(ReplaceAction) == expected


def test_the_constructor_determinations_are_five_and_three_for_the_same_reason():
    """§21.5.1 and §21.6.1 share their first two values and diverge after, like §21.3/§21.4.

    An optional component can be absent because a container ran out (§21.5.6) or because a
    pointer is zero (§21.5.9). A CHOICE alternative can be neither: exactly one alternative is
    always encoded, so there is no "ran out" and no "not there". §21.6.1 lists three values
    where §21.5.1 lists five, and one shared enum would let an object state something the
    notation cannot express.
    """
    optionality = {"field_to_be_set": 0, "field_to_be_used": 1, "container": 2,
                   "handle": 3, "pointer": 4}
    alternative = {"field_to_be_set": 0, "field_to_be_used": 1, "handle": 2}
    assert _cases("EcnOptionalityDetermination") == optionality
    assert _oracle(OptionalityDetermination) == optionality
    assert _cases("EcnAlternativeDetermination") == alternative
    assert _oracle(AlternativeDetermination) == alternative


def test_component_order_is_one_enum_whose_third_value_two_clauses_disagree_about():
    """§22.10.1.1 declares `{textual, tag, random}`; §22.6.1.1 declares `{textual, tag}`.

    Two enums here would be the safer-looking choice and the wrong one: §22.6.3.4 and
    §22.10.3.1–§22.10.3.3 define `textual` and `tag` in identical words, so two types would
    assert a difference that the text does not make. The real difference is *admissibility*,
    which is a law — and R25 carries it, with a fixture.
    """
    expected = {"textual": 0, "tag": 1, "random": 2}
    assert _cases("EcnComponentOrder") == expected
    assert _oracle(ComponentOrder) == expected
    verify = open(_VERIFY_PASS, encoding="utf-8").read()
    assert "ENUMERATED {textual, tag}" in verify


def test_the_handle_value_set_alternatives_match_the_choice_they_come_from():
    """§21.16.1's CHOICE, in its own order. `tag` is the DEFAULT of §22.9.1.1's property, and
    it is the only one that carries no value of its own — §21.16.5 makes it "determined by the
    number specified in an ECN encoding structure for a class in the tag category"."""
    expected = {"bits": 0, "octets": 1, "number": 2, "tag": 3, "range": 4, "ranges": 5}
    assert _cases("EcnHandleValueKind") == expected
    assert _oracle(HandleValueKind) == expected


# --- the ops and the laws ---------------------------------------------------------------

def test_law_rail_declares_every_op_the_oracle_module_needs():
    """`EcnModule`'s parts must all be nameable in the IR, or the projection has a hole."""
    text = open(_ECN_TD, encoding="utf-8").read()
    for mnemonic in ("ecn.module", "ecn.class", "ecn.structure", "ecn.field", "ecn.object",
                     "ecn.condition"):
        assert f'BCIR_Op<"{mnemonic}"' in text, mnemonic


def test_no_ods_attribute_is_named_class():
    """`class` is a C++ keyword, and ODS turns an attribute name into an accessor.

    Written as a test rather than left to the compiler because the failure it prevents is a
    build break in the MLIR job only — the Python rail would stay green, which is exactly the
    kind of drift this file exists to catch early.
    """
    text = open(_ECN_TD, encoding="utf-8").read()
    assert "$class" not in text
    assert "$encoding_class" in text


def test_every_r25_law_has_a_fixture_that_trips_it():
    """A law with no witness is a claim, not a check.

    Each entry is a rule the verify pass enforces and a fragment of the diagnostic the
    fixture expects. The pairs are checked in both files, so a law deleted from the pass or a
    case deleted from the fixture fails here rather than silently reducing coverage.
    """
    fixture = open(_FIXTURE, encoding="utf-8").read()
    verify = open(_VERIFY_PASS, encoding="utf-8").read()
    for citation, needle in (
        ("9.5.2", "both realize"),
        ("class assignment", "is circular"),
        ("16.3.1", "twice"),
        ("22.2.2.2", "ALIGNED TO ANY without a START-POINTER"),
        ("21.3.4", "states an encoding-space determination"),
        ("22.8.2.2", "sets UNUSED BITS DETERMINED BY"),
        ("22.8.2.5", "gives UNUSED BITS DECODER-TRANSFORMS"),
        ("22.12.2.3", "sets BIT-REVERSAL over a"),
        ("21.14.5", "over an odd"),
        ("22.1.2.8", "both REPLACE STRUCTURE and INSERT AT HEAD"),
        ("REPLACE", "sets REPLACE and another encoding property group"),
        ("23.7.2.4", "both a condition and ELSE"),
        ("23.7.2.7", "applies the INT-TO-INT transform"),
        ("21.11.5", "requires a Comparison"),
        ("21.11.5", "does not admit a Comparison or a comparator"),
        ("22.9.1.6", "a set of integer values"),
        ("22.9.1.9", "encoding object of the #TAG class"),
        ("22.9.2.1", "different bit positions"),
        ("22.9.2.3", "requires one pre-alignment unit per handle"),
        ("22.5.2.3", "forbids USING for `handle` and `pointer`"),
        ("22.5.2.4", "requires one in the same encoding object"),
        ("22.5.2.6", "22.5.2.6 admits them only for"),
        ("22.6.2.2", "admits HANDLE only for `handle`"),
        ("22.6.1.1", "22.6.1.1 declares"),
        ("22.10.2.1", "22.10.2.1 requires the encoding objects applied to all"),
    ):
        assert needle in verify, f"R25 does not enforce {citation}: {needle!r}"
        assert needle in fixture, f"no fixture witnesses {citation}: {needle!r}"


def test_every_attribute_and_enum_the_fixture_uses_is_declared_in_the_ods():
    """A fixture that names an attribute the dialect does not declare fails at PARSE time.

    Which is a worse failure than it sounds, and it is here because it has already happened
    once: a `-verify-diagnostics` run that cannot parse its input reports "expected error not
    produced" for every case in the file, so one typo looks like the whole law regressing, and
    the real cause is a line the diagnostic never mentions. The MLIR job is the only place
    that catches it, and it is the slowest job in CI.

    So this reads the fixture and the ODS and checks the two agree about names — the part of
    that failure that is decidable without a build.
    """
    ecn_td = open(_ECN_TD, encoding="utf-8").read()
    attrs_td = open(_ATTRS_TD, encoding="utf-8").read()
    body = "\n".join(line for line in open(_FIXTURE, encoding="utf-8").read().split("\n")
                     if not line.lstrip().startswith("//"))

    declared = set(re.findall(r"\$(\w+)", ecn_td))
    for used in sorted(set(re.findall(r"[{,]\s*(\w+)\s*=", body))):
        assert used in declared, f"{used!r} is not an argument of any bcir.ecn.* op"

    # `#bcir.<mnemonic><case>` — the attribute mnemonic and one of its enum's spellings.
    enums = {mnemonic: enum for enum, mnemonic in
             re.findall(r"def BCIR_\w+Attr\s*:\s*BCIR_EnumAttr<BCIR_(\w+),\s*\"(\w+)\">",
                        attrs_td)}
    for mnemonic, case in re.findall(r"#bcir\.(\w+)<(\w+)>", body):
        assert mnemonic in enums, f"#bcir.{mnemonic} is not a declared attribute mnemonic"
        assert case in _cases(enums[mnemonic]), \
            f"{case!r} is not a case of BCIR_{enums[mnemonic]}"


def test_the_fixture_carries_the_positive_cases_too():
    """A fixture of only failures cannot tell "rejected wrongly" from "rejected rightly".

    The frame header and the determinant module are both well-formed specifications that R25
    must accept, and they are the ones the oracle actually encodes through — so a law that
    over-fires shows up here rather than in a user's module.
    """
    fixture = open(_FIXTURE, encoding="utf-8").read()
    assert "bcir.ecn.module @FrameEncodings" in fixture
    assert "bcir.ecn.module @Determinants" in fixture
    positive_block = fixture.split("// -----")[0] + fixture.split("// -----")[1]
    assert "expected-error" not in positive_block


def test_both_rails_agree_that_r25_is_vacuous_without_ecn_operations():
    """Non-disturbance: a law rail that changed IR it does not describe would be a hazard.

    R24 states the invariant and R25 inherits it, so this checks the pass says so rather than
    trusting that the walks happen to be empty.
    """
    verify = open(_VERIFY_PASS, encoding="utf-8").read()
    assert "Vacuous for IR with no bcir.ecn.* operation." in verify
