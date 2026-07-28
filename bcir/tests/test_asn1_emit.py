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
