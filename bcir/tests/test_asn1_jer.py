"""X.697 JER conformance.

Annex A.3 is deliberately titled "**A possible** representation of this record value", not
"the" — JER has real encoder's options (§27.3.3 member order, §7.6.3 escapes, white-space),
so unlike X.691 Annex A there is no octet string to compare against. What A.3 CAN pin is
semantics: the annex's own text, indentation and all, must decode to the A.2 value, and the
members and values this rail emits must be exactly the ones A.3 shows.

So the load-bearing checks here are of two other kinds. First, the §7.2 constraint
visibility rules — the place JER differs most sharply from PER, and the only place a
constraint reaches a JER encoder at all. Second, the refusals: `json` accepts several things
that are not JSON and several more that are not conforming JER, and each is pinned.

A.1's PersonnelRecord is the same type X.690, X.691 and X.693 Annex A all use, so this file
completes the set: one abstract value now has a checked encoding under five rule families.
"""

from __future__ import annotations

from bcir.asn1.codec import Asn1Error
from bcir.asn1.constraints import Extensible, Size, ValueRange
from bcir.asn1.jer import (
    JER_OID,
    JER_OID_DESCRIPTOR,
    Array,
    Base64,
    JerInstructions,
    JerRules,
    Name,
    NameKeyword,
    Not,
    ObjectAs,
    Text,
    Unwrapped,
    apply_name_keyword,
    decode_jer,
    encode_jer,
)
from bcir.asn1.schema import (Choice, Component, ObjectSetTable, OpenType, Primitive,
                              Sequence, SequenceOf, Set, SetOf)
from bcir.asn1.tags import Universal
from bcir.asn1.tlv import encode_tlv
from bcir.asn1.values import BitString
from bcir.frontends.asn1.lower import compile_module

_ANNEX_A_MODULE = """
AnnexA DEFINITIONS ::= BEGIN
  PersonnelRecord ::= [APPLICATION 0] IMPLICIT SET {
      name Name, title [0] VisibleString, number EmployeeNumber,
      dateOfHire [1] Date, nameOfSpouse [2] Name,
      children [3] IMPLICIT SEQUENCE OF ChildInformation DEFAULT {} }
  ChildInformation ::= SET { name Name, dateOfBirth [0] Date }
  Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {
      givenName VisibleString, initial VisibleString, familyName VisibleString }
  EmployeeNumber ::= [APPLICATION 2] IMPLICIT INTEGER
  Date ::= [APPLICATION 3] IMPLICIT VisibleString
END
"""

_PERSONNEL_RECORD = {
    "name": {"givenName": "John", "initial": "P", "familyName": "Smith"},
    "title": "Director",
    "number": 51,
    "dateOfHire": "19710917",
    "nameOfSpouse": {"givenName": "Mary", "initial": "T", "familyName": "Smith"},
    "children": [
        {"name": {"givenName": "Ralph", "initial": "T", "familyName": "Smith"},
         "dateOfBirth": "19571111"},
        {"name": {"givenName": "Susan", "initial": "B", "familyName": "Jones"},
         "dateOfBirth": "19590717"},
    ],
}

#: A.3 as printed, with its white-space. The annex's second child spells `name` without
#: quotation marks, which §27.3.3's NOTE forbids ("The use of quotation marks around each
#: component identifier is required") and ECMA-404 does not admit; that is a defect in the
#: printed annex, so the quotation marks are restored here and the unquoted form is pinned
#: as a refusal in `test_an_unquoted_member_name_is_refused`.
_ANNEX_A3 = """{
    "name" : {
        "givenName" : "John",
        "initial" : "P",
        "familyName" : "Smith"
    },
    "title" : "Director",
    "number" : 51,
    "dateOfHire" : "19710917",
    "nameOfSpouse" : {
        "givenName" : "Mary",
        "initial" : "T",
        "familyName" : "Smith"
    },
    "children" : [
        {
            "name" : {
                "givenName" : "Ralph",
                "initial" : "T",
                "familyName" : "Smith"
            },
            "dateOfBirth": "19571111"
        },
        {
            "name" : {
                "givenName" : "Susan",
                "initial" : "B",
                "familyName" : "Jones"
            },
            "dateOfBirth" : "19590717"
        }
    ]
}"""


def _annex_a():
    return compile_module(_ANNEX_A_MODULE, "<annex>").module.types["PersonnelRecord"]


def _refuses(action, needle: str) -> None:
    try:
        action()
    except Asn1Error as error:
        assert needle in str(error), f"expected {needle!r} in {error}"
        return
    raise AssertionError(f"expected a refusal mentioning {needle!r}")


# --- Annex A -----------------------------------------------------------------------------


def test_annex_a3_as_printed_decodes_to_the_annex_a2_value():
    """A.3 is "a possible representation", so what it pins is meaning, not octets."""
    assert decode_jer(_ANNEX_A3, _annex_a(), rules=JerRules.BASIC) == _PERSONNEL_RECORD


def test_the_canonical_encoding_carries_exactly_the_members_annex_a3_shows():
    """Same members, same values, one spelling — A.3's own content with the options pinned."""
    got = encode_jer(_annex_a(), _PERSONNEL_RECORD).decode("utf-8")
    assert got == (
        '{"name":{"givenName":"John","initial":"P","familyName":"Smith"},'
        '"title":"Director","number":51,"dateOfHire":"19710917",'
        '"nameOfSpouse":{"givenName":"Mary","initial":"T","familyName":"Smith"},'
        '"children":[{"name":{"givenName":"Ralph","initial":"T","familyName":"Smith"},'
        '"dateOfBirth":"19571111"},'
        '{"name":{"givenName":"Susan","initial":"B","familyName":"Jones"},'
        '"dateOfBirth":"19590717"}]}'), got


def test_the_annex_a_record_round_trips_under_both_profiles():
    kind = _annex_a()
    for rules in (JerRules.BASIC, JerRules.CANONICAL):
        data = encode_jer(kind, _PERSONNEL_RECORD, rules=rules)
        assert decode_jer(data, kind, rules=rules) == _PERSONNEL_RECORD


def test_a_canonical_encoding_is_also_a_basic_one():
    """§6.3 — a JER decoder "shall support all JER encoding alternatives", and the
    canonical profile emits a subset of them, so the strict encoder's output must satisfy
    the lenient decoder."""
    kind = _annex_a()
    canonical = encode_jer(kind, _PERSONNEL_RECORD)
    assert decode_jer(canonical, kind, rules=JerRules.BASIC) == _PERSONNEL_RECORD


# --- clause 42: what X.697 actually registers --------------------------------------------


def test_x697_registers_one_object_identifier_and_no_canonical_variant():
    """§42.2 — the finding that shapes this whole module.

    There is exactly ONE object identifier and no canonical clause anywhere in X.697. The
    repo's rule that no encoding rule ships without a canonical variant is satisfied by a
    *BCIR* profile, which must therefore not claim an arc under `jer-encoding(7)`. This test
    exists so that inventing one later fails loudly rather than quietly shipping a
    registration that does not exist.
    """
    assert JER_OID == (2, 1, 7)
    assert JER_OID_DESCRIPTOR == "JER encoding of a single ASN.1 type"
    import bcir.asn1.jer as jer

    registered = [name for name in dir(jer) if name.endswith("_OID")]
    assert registered == ["JER_OID"], (
        f"X.697 42.2 assigns one object identifier; {registered} claims more")


# --- clause 7.2: what JER can and cannot see ---------------------------------------------


def test_an_integer_constraint_is_not_jer_visible():
    """§7.2.2 l) — "value and value range constraints on integer types" are NOT visible.

    The sharpest contrast with PER in the whole suite: the constraint that shrinks a PER
    field from 32 bits to 8 changes nothing at all here.
    """
    plain = Primitive(Universal.INTEGER, "INTEGER")
    bounded = Primitive(Universal.INTEGER, "INTEGER", ValueRange(0, 255))
    assert encode_jer(plain, 200) == encode_jer(bounded, 200) == b"200"


def test_a_size_constraint_on_an_octetstring_is_not_jer_visible():
    """§7.2.2 h) — "size constraints applied to a character string or octet string type"."""
    plain = Primitive(Universal.OCTET_STRING, "OCTET STRING")
    sized = Primitive(Universal.OCTET_STRING, "OCTET STRING", Size(ValueRange(2, 2)))
    assert encode_jer(plain, b"\xde\xad") == encode_jer(sized, b"\xde\xad") == b'"DEAD"'


def test_a_bitstring_size_is_the_one_constraint_that_chooses_a_form():
    """§7.2.1 a) with §24.1 — and Annex A.4's `MyBitString1` versus `MyBitString2`.

    `BIT STRING (SIZE (10))` is a fixed-size bitstring and encodes as a bare hexadecimal
    JSON string. `BIT STRING (SIZE (10), ...)` looks identical at the root, but §7.2.2 g)
    makes an *extensible* subtype constraint invisible, so it is a variable-size bitstring
    and takes §24.3's object form. One `...` changes the shape of the JSON.
    """
    value = BitString(b"\xa0\x40", 6)                       # ten bits
    fixed = Primitive(Universal.BIT_STRING, "BIT STRING", Size(ValueRange(10, 10)))
    extensible = Primitive(Universal.BIT_STRING, "BIT STRING",
                           Extensible(Size(ValueRange(10, 10))))
    unconstrained = Primitive(Universal.BIT_STRING, "BIT STRING")
    assert encode_jer(fixed, value) == b'"A040"'                        # §24.2
    assert encode_jer(extensible, value) == b'{"value":"A040","length":10}'   # §24.3
    assert encode_jer(unconstrained, value) == b'{"value":"A040","length":10}'
    for kind in (fixed, extensible, unconstrained):
        assert decode_jer(encode_jer(kind, value), kind) == value
    # A bare string where the type is variable-size is a form §24.1 c) does not offer.
    _refuses(lambda: decode_jer('"A040"', extensible), "24.1 c)")


def test_a_contents_constraint_unlocks_the_containing_form():
    """§7.2.1 e) with §25.4 — CONTAINING but without ENCODED BY."""
    inner = Sequence((Component("n", Primitive(Universal.INTEGER, "INTEGER")),), "Inner")
    octets = encode_tlv(inner.encode({"n": 7}))
    contained = Primitive(Universal.OCTET_STRING, "OCTET STRING", contains=inner)
    assert encode_jer(contained, octets) == b'{"containing":{"n":7}}'
    assert decode_jer(b'{"containing":{"n":7}}', contained) == octets
    # §7.2.1 e) says "without ENCODED BY": naming other rules is a statement about the
    # octets, not licence to read them as this rail's own type.
    encoded_by = Primitive(Universal.OCTET_STRING, "OCTET STRING", contains=inner,
                           encoded_by=(2, 1, 2, 1))
    assert encode_jer(encoded_by, octets) == b'"' + octets.hex().upper().encode() + b'"'


# --- clauses 20-41: the per-type encodings -----------------------------------------------


def test_the_scalar_encodings():
    assert encode_jer(Primitive(Universal.BOOLEAN, "BOOLEAN"), True) == b"true"    # §20
    assert encode_jer(Primitive(Universal.BOOLEAN, "BOOLEAN"), False) == b"false"
    assert encode_jer(Primitive(Universal.INTEGER, "INTEGER"), -42) == b"-42"      # §21
    assert encode_jer(Primitive(Universal.INTEGER, "INTEGER"), 0) == b"0"
    assert encode_jer(Primitive(Universal.NULL, "NULL"), None) == b"null"          # §26
    oid = Primitive(Universal.OBJECT_IDENTIFIER, "OBJECT IDENTIFIER")
    assert encode_jer(oid, (2, 1, 7)) == b'"2.1.7"'                                # §32
    assert decode_jer(b'"2.1.7"', oid) == (2, 1, 7)


def test_an_enumerated_value_is_a_json_string_and_has_no_numeric_spelling():
    """§22.1/§22.2 — the identifier, never the number.

    Same shape as XER §8.3.7 and for the same reason: BER encodes the enumeration *value*
    (X.690 §8.4), so the identifier is not recoverable from the octets alone, and a type
    without its enumeration cannot be encoded here at all.
    """
    kind = Primitive(Universal.ENUMERATED, "ENUMERATED",
                     enumeration=(("red", 0), ("yellow", 1), ("green", 2)))
    assert encode_jer(kind, 1) == b'"yellow"'
    assert decode_jer(b'"green"', kind) == 2
    _refuses(lambda: decode_jer(b"1", kind), "expected a JSON string")
    _refuses(lambda: encode_jer(kind, 7), "no numeric spelling")
    bare = Primitive(Universal.ENUMERATED, "ENUMERATED")
    _refuses(lambda: encode_jer(bare, 1), "no enumeration")


def test_the_special_real_values_are_json_strings():
    """§23.1.1 with Table 2 — and the round trip through each."""
    kind = Primitive(Universal.REAL, "REAL")
    assert encode_jer(kind, float("inf")) == b'"INF"'
    assert encode_jer(kind, float("-inf")) == b'"-INF"'
    assert encode_jer(kind, float("nan")) == b'"NaN"'
    assert encode_jer(kind, -0.0) == b'"-0"'
    assert encode_jer(kind, 0.0) == b"0"                    # §23.1.2: the value 0
    assert encode_jer(kind, 3.5) == b"3.5"                  # §23.3: a base-2 value
    assert decode_jer(b'"INF"', kind) == float("inf")
    assert decode_jer(b"3.5", kind) == 3.5
    assert decode_jer(b'{"base10Value":1.5}', kind) == 1.5   # §23.4
    nan = decode_jer(b'"NaN"', kind)
    assert nan != nan
    _refuses(lambda: decode_jer(b'"Inf"', kind), "Table 2")


def test_a_set_is_encoded_as_if_it_were_a_sequence():
    """§29 — "A value of a set type shall be encoded as if the type had been declared a
    sequence type", so there is no tag-order rule the way CXER §9.6.1 has one."""
    components = (Component("b", Primitive(Universal.INTEGER, "INTEGER"), tag=1),
                  Component("a", Primitive(Universal.INTEGER, "INTEGER"), tag=0))
    value = {"b": 2, "a": 1}
    assert encode_jer(Set(components), value) == encode_jer(Sequence(components), value)
    assert encode_jer(Set(components), value) == b'{"b":2,"a":1}'


def test_sequence_of_keeps_order_and_set_of_is_sorted_only_by_the_canonical_profile():
    """§28 preserves order; §30.2 leaves set-of order free, so the profile pins it."""
    element = Primitive(Universal.INTEGER, "INTEGER")
    assert encode_jer(SequenceOf(element), [3, 1, 2]) == b"[3,1,2]"
    assert encode_jer(SetOf(element), [3, 1, 2]) == b"[1,2,3]"
    assert encode_jer(SetOf(element), [3, 1, 2], rules=JerRules.BASIC) == b"[3,1,2]"


def test_a_choice_is_an_object_with_exactly_one_member():
    """§31.3.1/§31.3.2 — the wrapped encoding, which is the default without UNWRAPPED."""
    kind = Choice((Component("a", Primitive(Universal.INTEGER, "INTEGER"), tag=0),
                   Component("b", Primitive(Universal.UTF8_STRING, "UTF8String"), tag=1)))
    assert encode_jer(kind, ("b", "hi")) == b'{"b":"hi"}'
    assert decode_jer(b'{"a":1}', kind) == ("a", 1)
    _refuses(lambda: decode_jer(b'{"a":1,"b":"x"}', kind), "exactly one member")
    _refuses(lambda: decode_jer(b'{"z":1}', kind), "matches no alternative")


def test_the_two_families_of_restricted_character_string():
    """§38.1 gives a JSON string; §38.2 sends the other five through X.690 §8.23.5 as an
    octetstring, i.e. hexadecimal."""
    text = Primitive(Universal.UTF8_STRING, "UTF8String")
    assert encode_jer(text, "héllo") == '"héllo"'.encode()
    assert decode_jer('"héllo"'.encode(), text) == "héllo"
    octets = Primitive(Universal.GRAPHIC_STRING, "GraphicString")
    assert encode_jer(octets, b"\x41\x42") == b'"4142"'


def test_an_open_type_has_no_hexadecimal_fallback():
    """§41 — "the encoding of the value of the contained type", and nothing else.

    XER's §8.5 offers an `xmlhstring` alternative for octets whose type is unknown; JER
    offers none, so an unresolvable open type is not encodable rather than encodable badly.
    """
    inner = Sequence((Component("n", Primitive(Universal.INTEGER, "INTEGER")),), "Inner")
    octets = encode_tlv(inner.encode({"n": 7}))
    table = ObjectSetTable("C", ({"&id": 1, "&Type": inner},))
    opened = OpenType("OPEN", table=table, field="&Type",
                      governing=(("id",),), governing_fields=("&id",))
    kind = Sequence((Component("id", Primitive(Universal.INTEGER, "INTEGER")),
                     Component("body", opened)), "Outer")
    assert encode_jer(kind, {"id": 1, "body": octets}) == b'{"id":1,"body":{"n":7}}'
    decoded = decode_jer(b'{"id":1,"body":{"n":7}}', kind)
    assert decoded["body"] == octets
    assert decoded["body.resolved"] == {"n": 7}
    _refuses(lambda: encode_jer(kind, {"id": 9, "body": octets}),
             "no hexadecimal alternative")


# --- the canonical profile, and what it refuses ------------------------------------------


def test_a_default_valued_component_is_omitted_by_the_canonical_profile():
    """X.697 states no rule, so the profile picks one — and says which tradition it follows.

    DER §11.5 and COER omit; CXER §9.5 requires the value present. JER is in the candidate
    set on size, so it omits.
    """
    kind = Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),
                     Component("y", Primitive(Universal.BOOLEAN, "BOOLEAN"), tag=0,
                               default=False)))
    assert encode_jer(kind, {"x": 1, "y": False}) == b'{"x":1}'
    assert encode_jer(kind, {"x": 1, "y": False}, rules=JerRules.BASIC) \
        == b'{"x":1,"y":false}'
    assert encode_jer(kind, {"x": 1, "y": True}) == b'{"x":1,"y":true}'
    # Either spelling decodes to the same abstract value (X.680 §25.12).
    for text in (b'{"x":1}', b'{"x":1,"y":false}'):
        assert decode_jer(text, kind, rules=JerRules.BASIC) == {"x": 1, "y": False}


def test_member_order_is_free_for_basic_and_fixed_for_the_canonical_profile():
    """§27.3.3 — "may be added to the encoding in any order"."""
    kind = Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),
                     Component("y", Primitive(Universal.BOOLEAN, "BOOLEAN"), tag=0)))
    assert decode_jer(b'{"y":true,"x":1}', kind, rules=JerRules.BASIC) \
        == {"x": 1, "y": True}
    _refuses(lambda: decode_jer(b'{"y":true,"x":1}', kind), "canonical profile")


def test_an_extensible_type_skips_a_member_it_does_not_know():
    """A newer peer's addition is a member name this version has no component for.

    JSON is self-delimiting, so skipping is structural — PER needs §19.9's open-type wrapper
    to achieve the same thing.
    """
    extensible = Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),),
                          extensible=True)
    assert decode_jer(b'{"x":1,"future":[1,2]}', extensible,
                      rules=JerRules.BASIC) == {"x": 1}
    closed = Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),))
    _refuses(lambda: decode_jer(b'{"x":1,"future":2}', closed, rules=JerRules.BASIC),
             "no extension marker")


def test_the_things_json_accepts_that_jer_does_not():
    """Each of these is a real gap between "the `json` module parsed it" and "it is a
    conforming JER encoding", which is why the decoder installs hooks rather than trusting
    the defaults."""
    integer = Primitive(Universal.INTEGER, "INTEGER")
    real = Primitive(Universal.REAL, "REAL")
    # §21: no fractional part and no exponent.
    _refuses(lambda: decode_jer(b"1.0", integer), "no fractional part")
    _refuses(lambda: decode_jer(b"1e2", integer), "no exponent")
    # ECMA-404 clause 8 has no NaN/Infinity literals; `json` accepts them by default.
    _refuses(lambda: decode_jer(b"NaN", real), "not a JSON token")
    _refuses(lambda: decode_jer(b"Infinity", real), "not a JSON token")
    _refuses(lambda: decode_jer(b"-Infinity", real), "not a JSON token")
    # A duplicate member: `json` silently keeps the last, which would let one value hide
    # behind another in an encoding that digests differently but decodes identically.
    kind = Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),))
    _refuses(lambda: decode_jer(b'{"x":1,"x":2}', kind), "more than once")
    # §7.6.2: the encoding is UTF-8.
    _refuses(lambda: decode_jer(b'"\xff"', Primitive(Universal.UTF8_STRING, "UTF8String")),
             "7.6.2")


def test_an_unquoted_member_name_is_refused():
    """§27.3.3's NOTE — "The use of quotation marks around each component identifier is
    required", which the printed Annex A.3 violates in its second child element."""
    kind = Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),))
    _refuses(lambda: decode_jer(b"{x:1}", kind), "not a JSON text")


def test_a_string_carries_only_the_escapes_ecma_404_requires():
    """§7.6.3 permits any escape; emitting the minimum is what makes the output stable."""
    kind = Primitive(Universal.UTF8_STRING, "UTF8String")
    assert encode_jer(kind, 'a"b\\c\nd\te') == b'"a\\"b\\\\c\\nd\\te"'
    assert encode_jer(kind, "\x01") == b'"\\u0001"'
    assert encode_jer(kind, "é") == '"é"'.encode()
    # A conforming peer may escape anything, and the decoder must accept it (§6.3).
    assert decode_jer(b'"\\u00e9"', kind) == "é"
    _refuses(lambda: encode_jer(kind, "a\ud800"), "unpaired surrogate")


# --- clauses 14-19: the JER encoding instructions -----------------------------------------


def test_the_five_name_keyword_case_operations():
    """§16.1.5.1-§16.1.5.5 — each is narrower than its name suggests.

    `UPPERCASED` changes "all characters of the identifier that are lower-case letters …
    Other characters are unchanged", so hyphens survive; `UPPERCAMELCASED` removes them.
    `CAPITALIZED` changes exactly one character. `LOWERCASED` on an all-lower-case
    identifier is a no-op, which is the case a looser implementation gets wrong by also
    stripping hyphens.
    """
    assert apply_name_keyword("some-name-here", NameKeyword.CAPITALIZED) \
        == "Some-name-here"
    assert apply_name_keyword("some-name-here", NameKeyword.UPPERCASED) \
        == "SOME-NAME-HERE"
    assert apply_name_keyword("some-name-here", NameKeyword.LOWERCASED) \
        == "some-name-here"
    assert apply_name_keyword("some-name-here", NameKeyword.UPPERCAMELCASED) \
        == "SomeNameHere"
    assert apply_name_keyword("some-name-here", NameKeyword.LOWERCAMELCASED) \
        == "someNameHere"
    assert apply_name_keyword("A-b", NameKeyword.LOWERCASED) == "a-b"


def _abc_sequence() -> Sequence:
    return Sequence((Component("a", Primitive(Universal.INTEGER, "INTEGER")),
                     Component("b", Primitive(Universal.BOOLEAN, "BOOLEAN"), tag=0),
                     Component("c", Primitive(Universal.INTEGER, "INTEGER"), tag=1,
                               optional=True)))


def test_array_encodes_a_sequence_positionally():
    """§14.1.2 with §27.2 — a JSON array instead of a JSON object."""
    kind = _abc_sequence()
    instructions = JerInstructions().assign(kind, Array())
    assert encode_jer(kind, {"a": 1, "b": True, "c": 3},
                      instructions=instructions) == b"[1,true,3]"
    assert encode_jer(kind, {"a": 1, "b": True}) == b'{"a":1,"b":true}'
    assert decode_jer(b"[1,true,3]", kind, instructions=instructions) \
        == {"a": 1, "b": True, "c": 3}
    assert decode_jer(b"[1,true,null]", kind, instructions=instructions) \
        == {"a": 1, "b": True}
    _refuses(lambda: decode_jer(b"[1,true,3,4]", kind, instructions=instructions),
             "27.2.1")


def test_a_trailing_null_may_be_omitted_and_the_canonical_profile_omits_it():
    """§27.2.2 — "Any number of instances of the JSON token null may be omitted from the
    end of the JSON array, as a sender's option"."""
    kind = _abc_sequence()
    instructions = JerInstructions().assign(kind, Array())
    assert encode_jer(kind, {"a": 1, "b": True}, instructions=instructions) == b"[1,true]"
    assert encode_jer(kind, {"a": 1, "b": True}, rules=JerRules.BASIC,
                      instructions=instructions) == b"[1,true,null]"
    # Both spellings denote the same abstract value.
    for text in (b"[1,true]", b"[1,true,null]"):
        assert decode_jer(text, kind, rules=JerRules.BASIC,
                          instructions=instructions) == {"a": 1, "b": True}


def test_array_forbids_an_optional_component_that_could_itself_be_null():
    """§14.2 — because the array has no names, an absent component and a present NULL
    would be the same three characters."""
    kind = Sequence((Component("a", Primitive(Universal.INTEGER, "INTEGER")),
                     Component("n", Primitive(Universal.NULL, "NULL"), tag=0,
                               optional=True)))
    _refuses(lambda: JerInstructions().assign(kind, Array()), "14.2")
    _refuses(lambda: JerInstructions().assign(
        Set((Component("a", Primitive(Universal.INTEGER, "INTEGER")),)), Array()),
        "14.2 restricts ARRAY to a sequence type")


def test_base64_replaces_the_hexadecimal_octetstring_encoding():
    """§15.1.2 with §25.2 — RFC 2045 §6.8, "except that the 76-character limit does not
    apply", which is why the encoder never folds lines."""
    kind = Primitive(Universal.OCTET_STRING, "OCTET STRING")
    instructions = JerInstructions().assign(kind, Base64())
    assert encode_jer(kind, b"hello world", instructions=instructions) \
        == b'"aGVsbG8gd29ybGQ="'
    assert encode_jer(kind, b"hello world") == b'"68656C6C6F20776F726C64"'
    assert decode_jer(b'"aGVsbG8gd29ybGQ="', kind, instructions=instructions) \
        == b"hello world"
    assert encode_jer(kind, bytes(range(70)), instructions=instructions).count(b"\n") == 0
    _refuses(lambda: decode_jer(b'"not base64!"', kind, instructions=instructions),
             "RFC 2045")
    _refuses(lambda: JerInstructions().assign(
        Primitive(Universal.INTEGER, "INTEGER"), Base64()), "15.2")


def test_name_changes_a_member_name_and_is_keyed_on_the_component():
    """§16.1.4, and §9.9's exception — NAME is the one instruction a typereference does
    NOT inherit, which is why it is assigned against the component rather than its type.

    The two components below share one `Primitive` object, exactly as two components
    referencing one assigned type would. Keying NAME on the type would rename both.
    """
    shared = Primitive(Universal.INTEGER, "INTEGER")
    kind = Sequence((Component("first", shared),
                     Component("second", shared, tag=0)))
    instructions = JerInstructions().assign(kind.components[0], Name("alpha"))
    assert encode_jer(kind, {"first": 1, "second": 2}, instructions=instructions) \
        == b'{"alpha":1,"second":2}'
    assert decode_jer(b'{"alpha":1,"second":2}', kind, instructions=instructions) \
        == {"first": 1, "second": 2}
    keyworded = JerInstructions().assign(kind.components[1],
                                         Name(NameKeyword.UPPERCASED))
    assert encode_jer(kind, {"first": 1, "second": 2}, instructions=keyworded) \
        == b'{"first":1,"SECOND":2}'


def test_name_applies_to_a_choice_alternative_too():
    """§31.3.2 a) — the wrapped choice's single member name is subject to NAME."""
    kind = Choice((Component("alpha", Primitive(Universal.INTEGER, "INTEGER"), tag=0),))
    instructions = JerInstructions().assign(kind.alternatives[0], Name("A"))
    assert encode_jer(kind, ("alpha", 1), instructions=instructions) == b'{"A":1}'
    assert decode_jer(b'{"A":1}', kind, instructions=instructions) == ("alpha", 1)


def _map_setof() -> SetOf:
    return SetOf(Sequence((
        Component("key", Primitive(Universal.UTF8_STRING, "UTF8String")),
        Component("val", Primitive(Universal.INTEGER, "INTEGER"), tag=0))))


def test_object_turns_a_set_of_pairs_into_a_json_map():
    """§17.1.2 with §30.3 — "A typical use … is to produce a JSON object that represents an
    unordered set of associations … Such a set is often called a map"."""
    kind = _map_setof()
    instructions = JerInstructions().assign(kind, ObjectAs())
    items = [{"key": "b", "val": 2}, {"key": "a", "val": 1}]
    assert encode_jer(kind, items, instructions=instructions) == b'{"a":1,"b":2}'
    assert encode_jer(kind, items) \
        == b'[{"key":"a","val":1},{"key":"b","val":2}]'
    assert decode_jer(b'{"a":1,"b":2}', kind, instructions=instructions) \
        == [{"key": "a", "val": 1}, {"key": "b", "val": 2}]


def test_object_restrictions_are_all_of_clause_17_2():
    """§17.2 — every clause of it, because a map is only unambiguous if all of them hold."""
    integer = Primitive(Universal.INTEGER, "INTEGER")
    pair = Sequence((Component("key", Primitive(Universal.UTF8_STRING, "UTF8String")),
                     Component("val", integer, tag=0)))
    _refuses(lambda: JerInstructions().assign(SequenceOf(pair), ObjectAs()),
             "17.2 restricts OBJECT to a set-of type")
    _refuses(lambda: JerInstructions().assign(SetOf(integer), ObjectAs()),
             "to be a sequence type")
    _refuses(lambda: JerInstructions().assign(
        SetOf(Sequence((Component("a", integer), Component("b", integer, tag=0),
                        Component("c", integer, tag=1)))), ObjectAs()),
        "exactly two components")
    # The key becomes a JSON member name, which ECMA-404 clause 6 requires to be a string.
    _refuses(lambda: JerInstructions().assign(
        SetOf(Sequence((Component("k", integer), Component("v", integer, tag=0)))),
        ObjectAs()), "17.2 restricts the first component")
    _refuses(lambda: JerInstructions().assign(
        SetOf(Sequence((Component("k", Primitive(Universal.UTF8_STRING, "UTF8String")),
                        Component("v", integer, tag=0, optional=True)))), ObjectAs()),
        "OPTIONAL or DEFAULT")
    _refuses(lambda: JerInstructions().assign(
        SetOf(Sequence((Component("k", Primitive(Universal.UTF8_STRING, "UTF8String")),
                        Component("v", integer, tag=0)), extensible=True)), ObjectAs()),
        "without an extension marker")


def _colour() -> Primitive:
    return Primitive(Universal.ENUMERATED, "ENUMERATED",
                     enumeration=(("red", 0), ("light-blue", 1), ("green", 2)))


def test_text_rewrites_the_enumeration_strings_and_all_covers_the_rest():
    """§18.1.4 with §18.1.5 — a named identifier wins, and ALL "applies to all the
    enumeration items whose identifiers do not appear in this TEXT encoding instruction"."""
    kind = _colour()
    instructions = JerInstructions().assign(
        kind, Text((("red", "ROT"), ("ALL", NameKeyword.UPPERCAMELCASED))))
    assert encode_jer(kind, 0, instructions=instructions) == b'"ROT"'
    assert encode_jer(kind, 1, instructions=instructions) == b'"LightBlue"'
    assert encode_jer(kind, 2, instructions=instructions) == b'"Green"'
    for number in (0, 1, 2):
        text = encode_jer(kind, number, instructions=instructions)
        assert decode_jer(text, kind, instructions=instructions) == number
    # Without the instruction the identifier is used unchanged (§22.2).
    assert encode_jer(kind, 1) == b'"light-blue"'


def test_text_restrictions():
    """§18.2.1-§18.2.3."""
    kind = _colour()
    _refuses(lambda: JerInstructions().assign(
        Primitive(Universal.INTEGER, "INTEGER"), Text((("a", "b"),))), "18.2.1")
    _refuses(lambda: JerInstructions().assign(kind, Text((("purple", "X"),))),
             "not an enumeration identifier")
    _refuses(lambda: JerInstructions().assign(kind, Text((("red", "A"), ("red", "B")))),
             "18.2.2")
    _refuses(lambda: JerInstructions().assign(kind, Text((("ALL", "X"),))),
             "to be a Keyword when the IdentifierOrAll is ALL")
    # §18.2.3: the final set of strings shall not contain two identical strings.
    _refuses(lambda: JerInstructions().assign(kind, Text((("red", "green"),))), "18.2.3")


def test_unwrapped_drops_the_wrapping_object():
    """§19.1.2 with §31.2 — "the encoding of the chosen alternative", and nothing else."""
    kind = Choice((
        Component("n", Primitive(Universal.INTEGER, "INTEGER"), tag=0),
        Component("s", Primitive(Universal.UTF8_STRING, "UTF8String"), tag=1),
        Component("l", SequenceOf(Primitive(Universal.INTEGER, "INTEGER")), tag=2),
        Component("z", Primitive(Universal.NULL, "NULL"), tag=3),
    ))
    instructions = JerInstructions().assign(kind, Unwrapped())
    for value, text in ((("n", 5), b"5"), (("s", "hi"), b'"hi"'),
                        (("l", [1, 2]), b"[1,2]"), (("z", None), b"null")):
        assert encode_jer(kind, value, instructions=instructions) == text
        assert decode_jer(text, kind, instructions=instructions) == value
    assert encode_jer(kind, ("n", 5)) == b'{"n":5}'         # §31.3 without the instruction


def test_unwrapped_discriminates_two_object_alternatives_by_member_name():
    """§19.2.3 — the rule that lets two sequence alternatives coexist unwrapped."""
    left = Sequence((Component("p", Primitive(Universal.INTEGER, "INTEGER")),))
    right = Sequence((Component("q", Primitive(Universal.INTEGER, "INTEGER")),))
    kind = Choice((Component("x", left, tag=0), Component("y", right, tag=1)))
    instructions = JerInstructions().assign(kind, Unwrapped())
    assert decode_jer(b'{"p":1}', kind, instructions=instructions) == ("x", {"p": 1})
    assert decode_jer(b'{"q":1}', kind, instructions=instructions) == ("y", {"q": 1})


def test_a_19_2_violation_surfaces_as_an_ambiguity_rather_than_a_guess():
    """§19.2.2, enforced where its absence would actually hurt.

    §6.6's NOTE says "It is the final encoding instructions that determine conformity", so
    a check at assignment time could be wrong in either direction once a later assignment
    lands. Decoding is the point where the ambiguity becomes real, so that is where it is
    reported — against the clause, rather than by picking a winner.
    """
    kind = Choice((Component("i", Primitive(Universal.INTEGER, "INTEGER"), tag=0),
                   Component("j", Primitive(Universal.INTEGER, "INTEGER"), tag=1)))
    instructions = JerInstructions().assign(kind, Unwrapped())
    # Encoding is unambiguous -- the caller named the alternative.
    assert encode_jer(kind, ("i", 1), instructions=instructions) == b"1"
    _refuses(lambda: decode_jer(b"1", kind, instructions=instructions), "19.2.2")
    _refuses(lambda: JerInstructions().assign(_abc_sequence(), Unwrapped()), "19.2.1")


def test_no_alternative_of_the_right_shape_is_refused():
    kind = Choice((Component("n", Primitive(Universal.INTEGER, "INTEGER"), tag=0),))
    instructions = JerInstructions().assign(kind, Unwrapped())
    _refuses(lambda: decode_jer(b'"text"', kind, instructions=instructions), "31.2")


def test_clause_13_precedence_negating_and_replacement():
    """§13.2 and §13.3 — the two ways an assignment changes the associated set.

    §9.8 is the invariant these maintain: "An ASN.1 type can never have more than one
    associated JER encoding instruction of a given category, no matter how they are
    assigned."
    """
    octets = Primitive(Universal.OCTET_STRING, "OCTET STRING")
    # §13.2: a negating instruction removes the one of that category, and never joins the
    # set itself.
    negated = JerInstructions().assign(octets, Base64()).assign(octets, Not(Base64))
    assert encode_jer(octets, b"\xde\xad", instructions=negated) == b'"DEAD"'
    # §13.3.2: a second positive instruction of the same category REPLACES the first.
    kind = _abc_sequence()
    replaced = JerInstructions().assign(kind.components[0], Name("one"), Name("two"))
    assert encode_jer(kind, {"a": 1, "b": True}, instructions=replaced) \
        == b'{"two":1,"b":true}'
    # Negating a category that was never assigned is harmless (§13.2's NOTE 1).
    JerInstructions().assign(octets, Not(Base64))


def test_instructions_compose_across_a_whole_value():
    """All six at once, because the interesting failures are in the interactions."""
    colour = _colour()
    pairs = _map_setof()
    inner = Sequence((Component("blob", Primitive(Universal.OCTET_STRING,
                                                  "OCTET STRING")),
                      Component("shade", colour, tag=0)))
    kind = Sequence((Component("head", inner),
                     Component("tags", pairs, tag=0)))
    instructions = (JerInstructions()
                    .assign(inner, Array())
                    .assign(inner.components[0].type, Base64())
                    .assign(colour, Text((("ALL", NameKeyword.UPPERCASED),)))
                    .assign(pairs, ObjectAs())
                    .assign(kind.components[1], Name("labels")))
    value = {"head": {"blob": b"\x01\x02", "shade": 1},
             "tags": [{"key": "b", "val": 2}, {"key": "a", "val": 1}]}
    assert encode_jer(kind, value, instructions=instructions) \
        == b'{"head":["AQI=","LIGHT-BLUE"],"labels":{"a":1,"b":2}}'
    # The set-of comes back in canonical order rather than the caller's, which is what
    # §30.3.3 leaves free and X.680 §28.3's NOTE 2 says outright: "Encoding rules are not
    # required to preserve the order of these values."
    sorted_value = dict(value, tags=[{"key": "a", "val": 1}, {"key": "b", "val": 2}])
    assert decode_jer(encode_jer(kind, value, instructions=instructions), kind,
                      instructions=instructions) == sorted_value
    # Without the instructions the same value is a completely different document.
    assert encode_jer(kind, value) == (
        b'{"head":{"blob":"0102","shade":"light-blue"},'
        b'"tags":[{"key":"a","val":1},{"key":"b","val":2}]}')
