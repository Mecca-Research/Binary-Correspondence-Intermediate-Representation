"""ASN.1 NULL has one abstract value, `codec.NULL`, on every encoding rule.

The 2026-10-06 audit (§4.5, found by ASN1-R's review) found it with three:

  * BER/DER took `codec.NULL` and refused `None`;
  * PER and OER took any value at all -- `5` encoded as a NULL without complaint -- and decoded
    `codec.NULL`;
  * XER and JER took `None`, refused `codec.NULL`, and decoded `None`.

So a value one rule decoded was refused by another rule's encoder, and `None`, which the value
model reserves for an ABSENT component, was a NULL on two rules. One predicate,
`codec.require_null`, now answers for every encoder, and every decoder returns `codec.NULL`.

Every test here was RED on the parent and states the property, not the repair.
"""

from __future__ import annotations

from bcir.asn1 import jer, oer, per, xer
from bcir.asn1.codec import NULL
from bcir.asn1.tags import Asn1Error
from bcir.frontends.asn1 import compile_module

_MODULE = compile_module(
    """M DEFINITIONS AUTOMATIC TAGS ::= BEGIN
  S ::= SEQUENCE { a INTEGER, n NULL, o NULL OPTIONAL, l SEQUENCE OF NULL, c C }
  C ::= CHOICE { none NULL, num INTEGER }
END
""",
    "<null>",
).module
_S = _MODULE.types["S"]


def _rules():
    """(name, encode(value), decode(data)) for every encoding rule, over S."""
    return (
        ("DER", lambda v: _MODULE.encode("S", v), lambda d: _MODULE.decode("S", d)),
        ("PER", lambda v: per.encode_per(_S, v), lambda d: per.decode_per(d, _S)),
        ("OER", lambda v: oer.encode_oer(_S, v), lambda d: oer.decode_oer(_S, d)),
        ("XER", lambda v: xer.encode_xer(_S, v), lambda d: xer.decode_xer(d, _S)),
        ("JER", lambda v: jer.encode_jer(_S, v), lambda d: jer.decode_jer(d, _S)),
    )


_VALUE = {"a": 1, "n": NULL, "o": NULL, "l": [NULL, NULL], "c": ("none", NULL)}


def test_a_value_any_rule_decodes_encodes_on_every_rule():
    """The decoded value is the same Python value on every rule, so it can be handed to any
    other rule's encoder -- the property a cross-rule transcode rests on."""
    decoded = {}
    for name, encode, decode in _rules():
        value = decode(encode(_VALUE))
        assert value == _VALUE, (name, value)
        assert value["n"] is NULL and value["l"][0] is NULL and value["c"][1] is NULL, name
        decoded[name] = value
    for source, value in decoded.items():
        for name, encode, _ in _rules():
            assert encode(value) == encode(_VALUE), (source, name)


def test_none_is_refused_as_a_null_on_every_rule():
    """`None` means an absent component; as the value of a present NULL it is refused, with a
    verdict that says so, on every rule."""
    for name, encode, _ in _rules():
        try:
            encode({**_VALUE, "n": None})
        except Asn1Error as exc:
            assert "None means an absent component" in str(exc), (name, exc)
        else:  # pragma: no cover - the defect
            raise AssertionError(f"{name}: None encoded as a NULL")


def test_a_value_of_another_type_is_refused_as_a_null():
    """PER and OER put nothing on the wire for a NULL, and encoded whatever value they were
    given as one: `5` went out as a NULL on both."""
    for name, encode, _ in _rules():
        try:
            encode({**_VALUE, "n": 5})
        except Asn1Error as exc:
            assert "the value of a NULL is codec.NULL" in str(exc), (name, exc)
        else:  # pragma: no cover - the defect
            raise AssertionError(f"{name}: 5 encoded as a NULL")


def test_an_absent_optional_null_and_a_present_one_differ_on_every_rule():
    """Absent is a missing key; present is `NULL`. Each rule tells them apart in its octets and
    gives each back as it was."""
    absent = {key: value for key, value in _VALUE.items() if key != "o"}
    for name, encode, decode in _rules():
        assert encode(absent) != encode(_VALUE), name
        assert "o" not in decode(encode(absent)), name
        assert decode(encode(_VALUE))["o"] is NULL, name
