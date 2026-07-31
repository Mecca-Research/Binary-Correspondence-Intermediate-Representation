"""E1 — the write-side plan, the neutral value stream, and one emitter per candidate.

#682 established that a **schema-free** encode column is not the decode column's mirror:
`encode_der` takes a value and no type, every other encoder takes the type first, and so a
schema-free harness would produce a two-row table with JER missing. The fix is to make every
emitter schema-directed, which is what makes their costs comparable at all.

**The tests that matter are the parity tests.** An emitter that is merely self-consistent
proves nothing; each one must reproduce the oracle's octets exactly, for the same value,
through a descriptor. Everything else here is about the two places a plan-driven encoder can
go quietly wrong: a construct it does not understand, and a stream that does not match the
plan it is read against.
"""

from __future__ import annotations

from bcir.asn1.codec import NULL, Oid, encode_tlv
from bcir.asn1.emit import EmitRules, emit, flatten
from bcir.asn1.encode_plan import EncodePlan, compile_encode_plan
from bcir.asn1.jer import encode_jer
from bcir.asn1.oer import encode_oer
from bcir.asn1.schema import Choice, Component, Primitive, Sequence, SequenceOf, Set
from bcir.asn1.tags import Asn1Error, Universal

_I = Primitive(Universal.INTEGER)
_S = Primitive(Universal.UTF8_STRING)
_B = Primitive(Universal.BOOLEAN)
_N = Primitive(Universal.NULL)
_O = Primitive(Universal.OCTET_STRING)
_OID = Primitive(Universal.OBJECT_IDENTIFIER)


def _seq(*components, name="X") -> Sequence:
    return Sequence(tuple(components), name=name)


_CHOICE = Choice((Component("num", _I, tag=0), Component("txt", _S, tag=1)), name="C")

#: Every case is `(label, type, value)`, and the value is the DER/OER spelling. A JER
#: spelling is given only where the oracle's value mapping differs — see the NULL test.
_CORPUS = (
    ("a negative integer", _seq(Component("v", _I)), {"v": -1}),
    ("an integer past 64 bits", _seq(Component("v", _I)), {"v": 2 ** 64 + 7}),
    ("a large negative integer", _seq(Component("v", _I)), {"v": -(2 ** 70)}),
    ("zero", _seq(Component("v", _I)), {"v": 0}),
    ("the 128 length boundary", _seq(Component("v", _I)), {"v": 128}),
    ("both booleans", _seq(Component("a", _B), Component("b", _B)), {"a": True, "b": False}),
    ("an octet string", _seq(Component("v", _O)), {"v": b"\x00\xff\x10"}),
    ("an empty octet string", _seq(Component("v", _O)), {"v": b""}),
    ("an empty string", _seq(Component("v", _S)), {"v": ""}),
    ("a non-ASCII string", _seq(Component("v", _S)), {"v": "café \U0001f600"}),
    ("a string past the short length form", _seq(Component("v", _S)), {"v": "x" * 300}),
    ("an object identifier", _seq(Component("v", _OID)),
     {"v": Oid((1, 3, 6, 1, 4, 1, 62596, 1))}),
    ("an optional present", _seq(Component("a", _I), Component("b", _I, optional=True)),
     {"a": 1, "b": 2}),
    ("an optional absent", _seq(Component("a", _I), Component("b", _I, optional=True)),
     {"a": 1}),
    ("a default absent", _seq(Component("a", _I), Component("b", _B, default=False)),
     {"a": 1}),
    ("a default present", _seq(Component("a", _I), Component("b", _B, default=False)),
     {"a": 1, "b": True}),
    # Twelve optionals is more than one OER preamble octet, which is where a bitmap that
    # forgot to carry into a second octet stops agreeing with the oracle.
    ("twelve optionals, alternate present",
     _seq(*[Component(f"c{i}", _I, optional=True) for i in range(12)]),
     {f"c{i}": i for i in range(0, 12, 2)}),
    ("an empty SEQUENCE OF", _seq(Component("v", SequenceOf(_I, "SEQUENCE OF INTEGER"))),
     {"v": []}),
    ("a short SEQUENCE OF", _seq(Component("v", SequenceOf(_I, "SEQUENCE OF INTEGER"))),
     {"v": [1, 2, 3]}),
    ("a SEQUENCE OF past 255 elements",
     _seq(Component("v", SequenceOf(_I, "SEQUENCE OF INTEGER"))), {"v": list(range(300))}),
    ("a nested SEQUENCE", _seq(Component("in", _seq(Component("a", _I), name="Inner"))),
     {"in": {"a": 5}}),
    ("implicit context tags", _seq(Component("a", _I, tag=0), Component("b", _S, tag=1)),
     {"a": 1, "b": "x"}),
    ("an explicit context tag", _seq(Component("a", _I, tag=0, explicit=True)), {"a": 1}),
    ("a tag past the low-tag form", _seq(Component("a", _I, tag=100, explicit=True)),
     {"a": 1}),
    ("a CHOICE on its integer arm", _seq(Component("v", _CHOICE, tag=5, explicit=True)),
     {"v": ("num", 7)}),
    ("a CHOICE on its string arm", _seq(Component("v", _CHOICE, tag=5, explicit=True)),
     {"v": ("txt", "hi")}),
)


def _plan(kind, label: str) -> EncodePlan:
    return compile_encode_plan(kind, module="Test", type_name=label)


# --- parity: the only reason to trust any of this ------------------------------------------


def test_the_plan_driven_der_matches_the_oracle_octet_for_octet():
    for label, kind, value in _CORPUS:
        plan = _plan(kind, label)
        assert emit(plan, flatten(plan, value), rules=EmitRules.DER) == \
            encode_tlv(kind.encode(value)), label


def test_the_plan_driven_jer_matches_the_oracle_octet_for_octet():
    for label, kind, value in _CORPUS:
        plan = _plan(kind, label)
        assert emit(plan, flatten(plan, value), rules=EmitRules.JER) == \
            encode_jer(kind, value), label


def test_the_plan_driven_oer_matches_the_oracle_octet_for_octet():
    """OER is the payoff: a row the *decode* table can never hold, measurable on the write side."""
    for label, kind, value in _CORPUS:
        plan = _plan(kind, label)
        assert emit(plan, flatten(plan, value), rules=EmitRules.COER) == \
            encode_oer(kind, value), label


def test_ber_differs_from_der_exactly_where_the_standard_says_it_may():
    """§8.1.3.6 lets BER leave a constructed length open and close with an EOC; §10.1 does not.

    If BER and DER came out identical this candidate would be an alias, and the cost table
    would carry two rows for one measurement.
    """
    kind = _seq(Component("a", _I), Component("b", _S))
    plan, value = _plan(kind, "x"), {"a": 1, "b": "hello"}
    stream = flatten(plan, value)
    der = emit(plan, stream, rules=EmitRules.DER)
    ber = emit(plan, stream, rules=EmitRules.BER)
    assert ber != der
    assert ber.endswith(b"\x00\x00") and ber[1] == 0x80      # indefinite, closed by an EOC
    assert der[1] == len(der) - 2                            # definite and minimal
    # The primitives inside are untouched: only the CONSTRUCTED length form differs.
    assert der[2:] == ber[2:-2]


# --- the finding the neutral stream exposed ---------------------------------------------------


def test_the_oracle_encoders_disagree_about_how_python_spells_null():
    """Pinned, not fixed: there is no single Python value all three encoders accept.

    `codec` wants its `NULL` sentinel and refuses `None`; `encode_jer` wants `None` and
    refuses `NULL`; `encode_oer` takes either. The ambiguity is in the value mapping rather
    than in any encoding, and it stays invisible until something drives every encoder from
    one input — which is precisely what a matched cost comparison has to do.

    This test exists so that unifying the spelling is a deliberate act with a visible
    dependent, rather than a change that quietly makes a passing harness pass differently.
    """
    kind = _seq(Component("v", _N))
    assert encode_tlv(kind.encode({"v": NULL}))
    assert encode_oer(kind, {"v": NULL}) is not None
    assert encode_jer(kind, {"v": None})
    for spelling, encoder in ((NULL, lambda v: encode_jer(kind, v)),
                              (None, lambda v: encode_tlv(kind.encode(v)))):
        try:
            encoder({"v": spelling})
        except Asn1Error:
            pass
        else:
            raise AssertionError(
                f"the NULL spelling {spelling!r} is now accepted where it was not; if that "
                f"was deliberate, the harness below can stop special-casing it")


def test_the_neutral_stream_has_no_null_ambiguity_to_disagree_about():
    """A NULL contributes zero octets, so every emitter agrees from the same input."""
    kind = _seq(Component("a", _I), Component("v", _N), Component("b", _I))
    plan = _plan(kind, "null")
    der_value = {"a": 1, "v": NULL, "b": 2}
    jer_value = {"a": 1, "v": None, "b": 2}
    stream = flatten(plan, der_value)
    assert stream == flatten(plan, jer_value), "the stream must not carry the spelling"
    assert emit(plan, stream, rules=EmitRules.DER) == encode_tlv(kind.encode(der_value))
    assert emit(plan, stream, rules=EmitRules.JER) == encode_jer(kind, jer_value)
    assert emit(plan, stream, rules=EmitRules.COER) == encode_oer(kind, der_value)


# --- the plan refuses what it cannot emit ------------------------------------------------------


def test_a_construct_without_a_rule_is_refused_at_compile_time_naming_its_clause():
    """A plan that silently skipped a construct would surface as an unexplained byte diff."""
    for kind, fragment in (
            (Set((Component("a", _I),), name="S"), "SET"),
            (_seq(Component("v", Primitive(Universal.REAL))), "no leaf rule"),
    ):
        try:
            compile_encode_plan(kind, module="Test", type_name="bad")
        except Asn1Error as error:
            assert fragment in str(error), f"{kind}: {error}"
        else:
            raise AssertionError(f"{kind} should not compile")


def test_an_extension_addition_is_refused_because_x690_and_x691_disagree_about_it():
    kind = Sequence((Component("a", _I), Component("b", _I, optional=True, extension=True)),
                    name="X")
    try:
        compile_encode_plan(kind, module="Test", type_name="ext")
    except Asn1Error as error:
        assert "extension addition" in str(error)
    else:
        raise AssertionError("an extension addition needs PER before it means anything here")


# --- the plan is a descriptor, with a descriptor's properties -------------------------------


def test_the_plan_serializes_byte_identically_and_carries_no_process_pointers():
    """§5.1: repeated compilation is byte-identical, and a descriptor holds no callables."""
    kind = _seq(Component("a", _I), Component("b", _S, optional=True))
    first = compile_encode_plan(kind, module="Test", type_name="R", source=b"source")
    again = compile_encode_plan(kind, module="Test", type_name="R", source=b"source")
    assert first.serialize() == again.serialize()
    assert first.sha256() == again.sha256()
    text = first.serialize().decode("utf-8")
    assert "<" not in text and "object at 0x" not in text
    # The live type is reachable for tests and excluded from the bytes.
    assert first._kind is kind
    assert "Sequence" not in text


def test_a_different_schema_gives_a_different_plan_digest():
    """The digest names the descriptor, so two schemas must not share one."""
    a = compile_encode_plan(_seq(Component("a", _I)), module="T", type_name="R")
    b = compile_encode_plan(_seq(Component("b", _I)), module="T", type_name="R")
    c = compile_encode_plan(_seq(Component("a", _S)), module="T", type_name="R")
    assert len({a.sha256(), b.sha256(), c.sha256()}) == 3


# --- the stream and the plan must agree --------------------------------------------------------


def test_a_truncated_stream_is_refused_rather_than_emitted_as_a_prefix():
    kind = _seq(Component("a", _I), Component("b", _S))
    plan = _plan(kind, "x")
    full = flatten(plan, {"a": 1, "b": "hello"})
    for rules in EmitRules:
        try:
            emit(plan, full[:-3], rules=rules)
        except Asn1Error as error:
            assert "truncated" in str(error), f"{rules}: {error}"
        else:
            raise AssertionError(f"{rules} emitted from a short stream")


def test_a_stream_with_a_leftover_suffix_is_refused():
    """A prefix would emit a VALID document of the wrong value — the worst failure to allow."""
    kind = _seq(Component("a", _I))
    plan = _plan(kind, "x")
    stream = flatten(plan, {"a": 1})
    for rules in EmitRules:
        try:
            emit(plan, stream + b"\x00\x00", rules=rules)
        except Asn1Error as error:
            assert "left over" in str(error), f"{rules}: {error}"
        else:
            raise AssertionError(f"{rules} ignored a stream suffix")


def test_a_required_component_that_is_absent_is_refused_when_flattened():
    kind = _seq(Component("a", _I), Component("b", _I))
    try:
        flatten(_plan(kind, "x"), {"a": 1})
    except Asn1Error as error:
        assert "required and absent" in str(error)
    else:
        raise AssertionError("a required component cannot be silently omitted")


def test_a_choice_value_must_name_exactly_one_alternative():
    kind = _seq(Component("v", _CHOICE, tag=5, explicit=True))
    plan = _plan(kind, "x")
    for value, fragment in ((({"num": 7}), "(alternative, value) pair"),
                            (("nope", 1), "not an alternative")):
        try:
            flatten(plan, {"v": value})
        except Asn1Error as error:
            assert fragment in str(error), error
        else:
            raise AssertionError(f"{value!r} should be refused")


def test_every_emitter_consumes_the_identical_stream():
    """The property the whole comparison rests on: one input, four emitters, no adapters."""
    for label, kind, value in _CORPUS:
        plan = _plan(kind, label)
        stream = flatten(plan, value)
        outputs = {rules: emit(plan, stream, rules=rules) for rules in EmitRules}
        assert len(outputs) == 4
        # Distinct octets from one input is the point; DER and BER may coincide only when
        # there is no constructed length to leave open, which this corpus never hits.
        assert outputs[EmitRules.DER] != outputs[EmitRules.JER]
        assert outputs[EmitRules.DER] != outputs[EmitRules.COER]


# --- constraints: the bug this found, and what still stands between the plan and PER ---------


def test_a_constrained_type_reaches_the_oer_form_its_constraint_selects():
    """The defect this investigation found in already-landed code, now fixed and pinned.

    DER, BER and JER encode a value identically whether or not a subtype constraint exists,
    and X.697 §7.2.2 l)/h) hide integer and string constraints from JER outright. **OER does
    not** — X.696 §10.3 gives a constrained INTEGER a fixed-width form with no length
    determinant, so `INTEGER (0..255)` holding 42 is `2A` where the unconstrained type is
    `01 2A`.

    Plan version 2 dropped constraints, so the OER emitter it drove wrote the unconstrained
    spelling for **every** type: a well-formed document of a different value. Every parity
    test passed because the corpus contained no constrained type — corpus blindness rather
    than a subtle bug, since the emitter was simply never asked.

    Each case below is one clause of §10.2's dispatch, and every one of them differs from
    the length-prefixed form the old emitter produced.
    """
    from bcir.asn1.constraints import Size, ValueRange

    cases = (
        # (label, type, value, the octets X.696 requires)
        ("§10.3 a) one unsigned octet", Primitive(
            Universal.INTEGER, "I", constraint=ValueRange(0, 255)), 42, b"\x2a"),
        ("§10.3 b) two unsigned octets", Primitive(
            Universal.INTEGER, "I", constraint=ValueRange(0, 65535)), 42, b"\x00\x2a"),
        ("§10.3 d) eight unsigned octets", Primitive(
            Universal.INTEGER, "I", constraint=ValueRange(0, 2 ** 64 - 1)), 2 ** 63,
         b"\x80" + b"\x00" * 7),
        ("§10.4 a) one signed octet", Primitive(
            Universal.INTEGER, "I", constraint=ValueRange(-128, 127)), -5, b"\xfb"),
        ("§10.3 e) a lower bound alone stays length-prefixed", Primitive(
            Universal.INTEGER, "I", constraint=ValueRange(0, None)), 300, b"\x02\x01\x2c"),
        ("§14.1 a fixed size drops the determinant", Primitive(
            Universal.OCTET_STRING, "O", constraint=Size(ValueRange(3, 3))), b"abc",
         b"abc"),
        ("§14.2 a size RANGE does not", Primitive(
            Universal.OCTET_STRING, "O", constraint=Size(ValueRange(1, 3))), b"abc",
         b"\x03abc"),
        ("§27.2 a known-multiplier string, fixed", Primitive(
            Universal.IA5_STRING, "A", constraint=Size(ValueRange(3, 3))), "abc", b"abc"),
        # §27.1 keeps UTF8String out of §27.2: a character costs 1..4 octets there, so a
        # character count never implies an octet count and the determinant must stay.
        ("§27.1 UTF8String is never known-multiplier", Primitive(
            Universal.UTF8_STRING, "U", constraint=Size(ValueRange(3, 3))), "abc",
         b"\x03abc"),
    )
    for label, kind, value, want in cases:
        outer = _seq(Component("v", kind))
        plan = compile_encode_plan(outer, module="Test", type_name="S")
        got = emit(plan, flatten(plan, {"v": value}), rules=EmitRules.COER)
        assert got == want, f"{label}: {got.hex()} != {want.hex()}"
        # The oracle is the arbiter; `want` is spelled out so a reader can check the clause
        # without running the oracle, and the two agreeing is what makes both trustworthy.
        assert got == encode_oer(outer, {"v": value}), label


def test_an_enumerated_is_not_an_integer_under_oer():
    """The second defect the same investigation found, which no constraint was needed to hit.

    X.696 §11 gives ENUMERATED its own form: §11.3's short form is the bare octet below 128,
    and §11.4's long form sets the top bit of a count octet over a SIGNED body. §10's
    length-prefixed integer is a different encoding, and the plan-driven emitter shared a
    branch with it — so every enumerated it ever encoded was wrong, unconstrained ones
    included. `_CORPUS` had no ENUMERATED, so nothing asked.

    A construct absent from the corpus is untested however many tests run over the corpus,
    which is why this one names its own values rather than adding a row and hoping.
    """
    wanted = ((0, b"\x00"), (5, b"\x05"), (127, b"\x7f"), (128, b"\x82\x00\x80"),
              (200, b"\x82\x00\xc8"), (-1, b"\x81\xff"), (-129, b"\x82\xff\x7f"))
    kind = Primitive(Universal.ENUMERATED, "E",
                     enumeration=tuple((f"i{index}", value)
                                       for index, (value, _want) in enumerate(wanted)))
    outer = _seq(Component("v", kind))
    plan = compile_encode_plan(outer, module="Test", type_name="S")
    for value, want in wanted:
        got = emit(plan, flatten(plan, {"v": value}), rules=EmitRules.COER)
        assert got == want, f"{value}: {got.hex()} != {want.hex()}"
        assert got == encode_oer(outer, {"v": value}), value


def test_a_jer_enumerated_is_its_identifier_and_never_its_number():
    """The third defect of the same family: JER shared a branch with INTEGER too.

    X.697 §22.2 spells an enumerated value as "the identifier of the chosen enumeration
    item", and §22.1 gives it **no numeric spelling at all**. X.690 §8.4 encodes the number,
    so the two rules disagree by design — and the plan-driven JER emitter wrote the number,
    producing a document no JER decoder can map back to an enumeration item.

    This is also the reason the plan carries the enumeration: the identifier is not derivable
    from the number, so no amount of care in the emitter could have covered for a plan that
    dropped it. Version 4 records it, and a bare ENUMERATED is refused at compile time rather
    than three emitters downstream.
    """
    kind = Primitive(Universal.ENUMERATED, "E",
                     enumeration=(("red", 4), ("green", 9), ("deep-blue", -3)))
    outer = _seq(Component("v", kind))
    plan = compile_encode_plan(outer, module="Test", type_name="S")
    for value, want in ((4, b'{"v":"red"}'), (9, b'{"v":"green"}'),
                        (-3, b'{"v":"deep-blue"}')):
        got = emit(plan, flatten(plan, {"v": value}), rules=EmitRules.JER)
        assert got == want, f"{value}: {got!r} != {want!r}"
        assert got == encode_jer(outer, {"v": value}), value
    # A number outside the enumeration has no JER document, so it is refused rather than
    # falling back to the numeric spelling §22.1 does not define.
    try:
        emit(plan, flatten(plan, {"v": 7}), rules=EmitRules.JER)
    except Asn1Error as error:
        assert "§22.1" in str(error), error
    else:
        raise AssertionError("a value outside the enumeration produced a JER document")


def test_a_bare_enumerated_is_refused_because_two_rules_need_the_identifiers():
    """One plan drives four emitters, so it must refuse what any one of them cannot encode.

    A bare ENUMERATED looks encodable: X.690 §8.4 and X.696 §11 read the number the value
    already carries. X.697 §22.2 needs the identifier and X.691 §14.1 needs the whole root to
    compute an index, and neither is derivable from a number. The oracle's `encode_jer`
    refuses such a type for exactly this reason; version 3 compiled it happily and the JER
    emitter wrote the bare number.

    Refusing at compile time keeps the failure where the missing information is.
    """
    outer = _seq(Component("v", Primitive(Universal.ENUMERATED, "E")))
    try:
        compile_encode_plan(outer, module="Test", type_name="S")
    except Asn1Error as error:
        assert "§22.2" in str(error) and "§14.1" in str(error), error
    else:
        raise AssertionError("a bare ENUMERATED compiled")


def test_the_three_candidates_that_ignore_constraints_still_ignore_them():
    """The other half: DER, BER and JER are unaffected, which is why version 2 was ever enough.

    Without this the fix above reads as "the plan was broken". It was not — it was complete
    for the candidates whose encodings do not read constraints, and the boundary is a
    property of the encodings rather than of the compiler. Recording a constraint must not
    move a byte for these three.
    """
    from bcir.asn1.constraints import ValueRange

    unconstrained = _seq(Component("v", Primitive(Universal.INTEGER, "INTEGER")))
    bounded = _seq(Component("v", Primitive(Universal.INTEGER, "INTEGER",
                                            constraint=ValueRange(0, 255))))
    value = {"v": 42}
    free_plan = compile_encode_plan(unconstrained, module="Test", type_name="S")
    bound_plan = compile_encode_plan(bounded, module="Test", type_name="S")
    for rules in (EmitRules.DER, EmitRules.BER, EmitRules.JER):
        assert (emit(free_plan, flatten(free_plan, value), rules=rules)
                == emit(bound_plan, flatten(bound_plan, value), rules=rules)), rules
    assert encode_tlv(unconstrained.encode(value)) == encode_tlv(bounded.encode(value))
    assert encode_jer(unconstrained, value) == encode_jer(bounded, value)
    # OER is the one that differs, which is what made the fix necessary.
    assert encode_oer(unconstrained, value) != encode_oer(bounded, value)


def test_a_constraint_the_plan_cannot_write_down_is_refused_not_truncated():
    """The format states its arithmetic range, and says so rather than wrapping.

    A descriptor is read by a freestanding C twin with no bignum, so `BOUND_MAX` and
    `ALPHABET_MAX` are §5.1's "state your scratch bound" applied to a text format. Both are
    chosen from the standard rather than from convenience: X.696 §10.3 d)'s widest fixed
    word is eight octets, and X.691 §30.5's alphabet decides bits-per-character.

    Truncating either produces a *different type* — a narrower OER field, or a different
    character width — so both refuse.
    """
    from bcir.asn1.constraints import PermittedAlphabet, Union, ValueRange
    from bcir.asn1.encode_plan import ALPHABET_MAX, BOUND_MAX

    too_wide = _seq(Component("v", Primitive(
        Universal.INTEGER, "I", constraint=ValueRange(0, BOUND_MAX + 1))))
    try:
        compile_encode_plan(too_wide, module="Test", type_name="S")
    except Asn1Error as error:
        assert "value_high" in str(error) and "§10.3" in str(error), error
    else:
        raise AssertionError("a bound past the format's range compiled")

    # A permitted alphabet one character past the buffer. Built as a union of single
    # characters so the count is exact rather than an artefact of a range's endpoints.
    from bcir.asn1.constraints import SingleValue
    wide = Union(tuple(SingleValue(chr(0x21 + i)) for i in range(ALPHABET_MAX + 1)))
    too_many = _seq(Component("v", Primitive(
        Universal.IA5_STRING, "A", constraint=PermittedAlphabet(wide))))
    try:
        compile_encode_plan(too_many, module="Test", type_name="S")
    except Asn1Error as error:
        assert "alphabet" in str(error) and "§30.5" in str(error), error
    else:
        raise AssertionError("an alphabet past the format's buffer compiled")


def test_what_still_stands_between_this_plan_and_a_per_emitter():
    """The remaining PER blocker, as a checked fact rather than a plan.

    Version 3 recorded constraints completely — every field X.691 reads off one, including
    the extension-root bounds and the permitted alphabet OER never looks at. Version 4 adds
    the two *schema* facts PER also needs, and both arrived as bug fixes rather than as
    features: the enumeration, because X.697 §22.2 encodes the identifier and the JER emitter
    was writing a number; and the extension marker, because X.691 §19.1, §23.5 and §14.3
    each emit a leading bit for it.

    **One blocker is left: extension additions.** `_compile_members` refuses a component
    marked `extension`, because X.691 §19.7 splits the root from the additions and X.690
    does not — one plan cannot describe both until the emitter that needs the split exists.
    Everything else PER reads is now in the descriptor.
    """
    from bcir.asn1.constraints import ValueRange
    from bcir.asn1.encode_plan import PLAN_VERSION
    from bcir.asn1.per import PerRules, PerVariant, encode_per

    assert PLAN_VERSION == 4, "plan version moved; re-derive what this test asserts"

    # The extension marker: two SEQUENCEs differing only there encode differently under PER,
    # and the plan now tells them apart. This is the fix, asserted as one.
    plain = Sequence((Component("a", _B),), name="S")
    extensible = Sequence((Component("a", _B),), name="S", extensible=True)
    for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
        assert (encode_per(plain, {"a": True}, variant=variant, rules=PerRules.CANONICAL)
                != encode_per(extensible, {"a": True}, variant=variant,
                              rules=PerRules.CANONICAL)), variant
    assert (compile_encode_plan(plain, module="T", type_name="S").sha256()
            != compile_encode_plan(extensible, module="T", type_name="S").sha256())

    # The enumeration carries NUMBERS, which is what §14.1 indexes — names alone could not
    # produce an index, and the identifier alone could not satisfy X.697.
    enumerated = _seq(Component("v", Primitive(
        Universal.ENUMERATED, "E", enumeration=(("red", 4), ("green", 9)))))
    node = compile_encode_plan(enumerated, module="T", type_name="S").root.members[0].node
    assert node.enumeration == (("red", 4), ("green", 9))

    # The constraint half stays complete.
    bounded = _seq(Component("v", Primitive(
        Universal.INTEGER, "I", constraint=ValueRange(0, 255))))
    recorded = compile_encode_plan(bounded, module="T", type_name="S").root.members[0].node
    assert recorded.constraint is not None
    assert recorded.constraint.root_value_low == 0
    assert recorded.constraint.root_value_high == 255

    # And what is left is exactly one thing, which refuses by name.
    additions = Sequence((Component("a", _B), Component("b", _I, optional=True,
                                                        extension=True)),
                         name="S", extensible=True)
    try:
        compile_encode_plan(additions, module="T", type_name="S")
    except Asn1Error as error:
        assert "19.7" in str(error), error
    else:
        raise AssertionError(
            "the plan now compiles extension additions; that was the last PER prerequisite, "
            "so build the emitter rather than deleting this test")
