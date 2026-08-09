"""PER's abstract values must be the projection's abstract values.

Review findings from PR #662, still live on main. These are not encoding bugs: the octets
were right for the value PER thought it had. They are places where PER's idea of a Python
value diverged from the rest of the projection, so a value could not cross between the rails,
or a value the ASN.1 type does not contain was accepted and encoded as one that it does.

  * §23 -- a CHOICE is an `(alternative, value)` pair everywhere in this model (`schema`,
    DER, OER, JER). PER demanded a single-entry mapping and refused the pair, so a CHOICE
    decoded on one rail could not be re-encoded on this one.
  * §18 -- ASN.1 NULL is `codec.NULL`; Python `None` means ABSENT. PER decoded NULL to
    `None`, collapsing the two.
  * §14 -- `int(value)` accepted `"1"`, `1.9` and `True` as enumeration values and encoded
    each as an enumerator, silently changing the abstract value on the wire.
  * §14.3 -- an ENUMERATED item after the `...` is an extension addition with its own
    encoding. The front-end put both halves of the marker in one root list, so such an item
    was encoded with a zero extension bit and a ROOT index: octets meaning a different
    enumerator.
  * §30.5.4 a) -- the natural-code character path checked only that a code fit the bit-field,
    never that it was in the type's repertoire, so an unconstrained VisibleString carried
    control characters and DEL.
  * The dispatch -- PER selects on the schema CLASS, so a front-end forward reference (how a
    recursive type is represented until its module finishes building) fell off the end of the
    chain and a recursive value could not be encoded at all.
"""

from __future__ import annotations

from bcir.asn1.codec import NULL, Strictness
from bcir.asn1.constraints import ValueRange
from bcir.asn1.per import (
    BitWriter,
    PerVariant,
    _encode_length_and_payload,
    decode_per,
    encode_per,
)
from bcir.asn1.schema import Choice, Component, Primitive, Sequence
from bcir.asn1.tags import Universal
from bcir.asn1.tlv import encode_tlv
from bcir.asn1.values import Asn1Error
from bcir.frontends.asn1 import compile_module

_VARIANTS = (PerVariant.UNALIGNED, PerVariant.ALIGNED)
_BYTE = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
_TEXT = Primitive(Universal.UTF8_STRING, "UTF8String")


def _refused(fn, what: str) -> str:
    try:
        result = fn()
    except Asn1Error as exc:
        return str(exc)
    raise AssertionError(f"{what}: accepted, returning {result!r}")


def test_a_choice_crosses_between_the_rails_in_both_directions() -> None:
    """§23: the `(alternative, value)` pair is the model's CHOICE value, PER included.

    The point is not that PER accepts a second spelling -- it is that a value produced by one
    rail can be consumed by another, which is what a schema-level abstract value is for.
    """
    kind = Choice((Component("a", _BYTE), Component("b", _TEXT)), name="C")
    for variant in _VARIANTS:
        pair = ("b", "hi")
        raw = encode_per(kind, pair, variant=variant)
        assert decode_per(raw, kind, variant=variant) == pair

        # The mapping spelling still works and must encode identically -- otherwise
        # "accepted" would be a quieter version of the same disagreement.
        assert encode_per(kind, {"b": "hi"}, variant=variant) == raw

        # Out of the tag-first rail and into PER...
        from_der = kind.decode(kind.encode(pair), strictness=Strictness.DER)
        assert from_der == pair
        assert encode_per(kind, from_der, variant=variant) == raw
        # ... and back the other way, with no adapter in between.
        assert encode_tlv(kind.encode(decode_per(raw, kind, variant=variant)))

    assert "alternative" in _refused(
        lambda: encode_per(kind, {"a": 1, "b": "x"}, variant=PerVariant.UNALIGNED),
        "a two-entry mapping as a CHOICE value")


def test_null_decodes_to_the_singleton_not_to_absence() -> None:
    """§18: `codec.NULL` is ASN.1 NULL; Python `None` is reserved for ABSENT.

    Returning `None` made a present NULL component indistinguishable from a missing one, and
    made a PER-decoded NULL unusable as input to the DER and OER encoders, which both want
    the singleton.
    """
    kind = Primitive(Universal.NULL, "NULL")
    for variant in _VARIANTS:
        decoded = decode_per(encode_per(kind, NULL, variant=variant), kind, variant=variant)
        assert decoded is NULL, f"{variant.value}: NULL decoded to {decoded!r}"
        # The value the other rails want, straight out of PER.
        assert encode_tlv(kind.encode(decoded))

    # Inside a SEQUENCE, a present NULL is now distinguishable from an absent OPTIONAL one.
    seq = Sequence((Component("v", kind, optional=True), Component("n", _BYTE)), name="S")
    for variant in _VARIANTS:
        present = decode_per(encode_per(seq, {"v": NULL, "n": 1}, variant=variant), seq,
                             variant=variant)
        absent = decode_per(encode_per(seq, {"n": 1}, variant=variant), seq, variant=variant)
        assert present["v"] is NULL
        assert "v" not in absent


def test_an_enumerated_value_must_actually_be_an_integer() -> None:
    """§14: `int(value)` coerced its way onto an enumerator and encoded it as one.

    The guard is the one X.696's ENUMERATED encoder and this module's own INTEGER path
    already use, so all three now refuse the same inputs.
    """
    kind = Primitive(Universal.ENUMERATED, "ENUMERATED", enumeration=(("a", 0), ("b", 1)))
    for variant in _VARIANTS:
        for bad in ("1", 1.9, True, None):
            assert "expected int" in _refused(
                lambda b=bad, v=variant: encode_per(kind, b, variant=v),
                f"{bad!r} as an ENUMERATED value")
        for value in (0, 1):
            assert decode_per(encode_per(kind, value, variant=variant), kind,
                              variant=variant) == value


def test_an_enumeration_extension_item_is_not_a_root_value() -> None:
    """§14.3: an item after the `...` has its own encoding, not a §14.2 root index.

    The front-end flattened both halves of the marker into one list, so `b` in
    `ENUMERATED { a(0), ..., b(1) }` was encoded with a zero extension bit and root index 1 --
    octets whose abstract meaning is a different enumerator. The two lists are separate now
    and §14.3's form is not built, so this is a refusal BY NAME rather than a wrong encoding;
    the decoder already refused the mirror case, so neither direction invents an answer.
    """
    module = compile_module("M DEFINITIONS ::= BEGIN E ::= ENUMERATED { a(0), ..., b(1) } END")
    kind = module.types["E"]
    assert kind.enumeration == (("a", 0),), "the root holds only what precedes the marker"
    assert kind.enum_extension == (("b", 1),)
    assert kind.enum_extensible is True

    for variant in _VARIANTS:
        assert "extension addition" in _refused(
            lambda v=variant: encode_per(kind, 1, variant=v),
            "an extension-addition enumerator encoded as a root value")
        assert decode_per(encode_per(kind, 0, variant=variant), kind, variant=variant) == 0

    # A non-extensible enumeration keeps every item in the root, and still round-trips.
    plain = compile_module(
        "M DEFINITIONS ::= BEGIN F ::= ENUMERATED { a(0), b(1), c(2) } END").types["F"]
    assert plain.enumeration == (("a", 0), ("b", 1), ("c", 2))
    assert plain.enum_extension is None
    for variant in _VARIANTS:
        for value in (0, 1, 2):
            assert decode_per(encode_per(plain, value, variant=variant), plain,
                              variant=variant) == value

    # §20.1 numbers an un-numbered item from the next unused value, counting ACROSS the
    # marker -- the split must not restart it.
    implicit = compile_module(
        "M DEFINITIONS ::= BEGIN G ::= ENUMERATED { a, b, ..., c } END").types["G"]
    assert implicit.enumeration == (("a", 0), ("b", 1))
    assert implicit.enum_extension == (("c", 2),)


def test_a_natural_code_character_is_checked_against_its_repertoire() -> None:
    """§30.5.4 a): fitting the bit-field is not the same as being in the type.

    VisibleString's repertoire is 32..126, but the natural-code path only asked whether the
    code fit `bits`, so DEL (127) and the control characters below space both encoded and
    decoded. The permitted-alphabet path never had the gap, because renumbering can only
    spell the characters it listed.
    """
    kind = Primitive(Universal.VISIBLE_STRING, "VisibleString")
    for variant in _VARIANTS:
        for bad in ("\x7f", "\x00", "a\x1fb"):
            assert "repertoire" in _refused(
                lambda b=bad, v=variant: encode_per(kind, b, variant=v),
                f"{bad!r} in a VisibleString")
        for good in ("", " ", "Hello, world!", "~"):
            assert decode_per(encode_per(kind, good, variant=variant), kind,
                              variant=variant) == good

    # The decoding half of the same gap. Built with the module's own length helper and the
    # alignment its emit uses, so the octets are what a peer sending code 127 would send --
    # `~` (126) is the neighbour that must still decode, which is what makes the pair a test
    # of the boundary rather than of the writer.
    for code, wanted in ((126, "~"), (127, None)):
        writer = BitWriter(PerVariant.UNALIGNED)
        _encode_length_and_payload(
            writer, 1, 0, None,
            lambda start, stop, c=code, w=writer: (w.align(), w.put_bits(c, 7)))
        raw = writer.to_bytes()
        if wanted is None:
            assert "repertoire" in _refused(
                lambda r=raw: decode_per(r, kind, variant=PerVariant.UNALIGNED),
                f"character code {code} decoded into a VisibleString")
        else:
            assert decode_per(raw, kind, variant=PerVariant.UNALIGNED) == wanted


def test_a_recursive_type_encodes_at_depth() -> None:
    """The dispatch selects on the schema CLASS, so a forward reference fell off its end.

    `compile_module` represents a recursive definition with a lazy placeholder until the
    module finishes building. The tag-first rails go through its `encode`/`decode` and never
    notice; PER raised "no encoding for schema type _LazyType" the moment a nested value was
    supplied, so `Node` encoded only while `next` was absent.
    """
    module = compile_module(
        "M DEFINITIONS ::= BEGIN "
        "Node ::= SEQUENCE { value INTEGER, next Node OPTIONAL } END")
    kind = module.types["Node"]
    values = [
        {"value": 1},
        {"value": 1, "next": {"value": 2}},
        {"value": 1, "next": {"value": 2, "next": {"value": 3}}},
        {"value": 1, "next": {"value": 2, "next": {"value": 3, "next": {"value": 4}}}},
    ]
    for variant in _VARIANTS:
        for value in values:
            raw = encode_per(kind, value, variant=variant)
            assert decode_per(raw, kind, variant=variant) == value, (variant.value, value)
