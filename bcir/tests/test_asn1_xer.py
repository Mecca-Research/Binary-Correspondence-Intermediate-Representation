"""X.693 XER conformance.

The load-bearing test here is `test_annex_a4_matches_the_specification_characters`. XER is
a text format, so it is tempting to check it by "does it look like XML" — which a round
trip also cannot answer, because an encoder and a decoder that share a misreading agree
with each other. Annex A.4 prints the standard's own CXER character string for a worked
`PersonnelRecord` value, and A.3 states the length of the same encoding as **653 octets
ignoring white-space**. Both are checked below, and the second is an independent arithmetic
check on the first.

A.1's `PersonnelRecord` is deliberately the same type X.690 Annex A and X.691 Annex A use,
so the fixture here is the one `test_asn1_per.py` already carries. That is the point: one
abstract value, five rule sets, five encodings, and the only thing that varies is the rule
set.
"""

from __future__ import annotations

from bcir.asn1.codec import Asn1Error
from bcir.asn1.schema import (
    Choice,
    Component,
    ObjectSetTable,
    OpenType,
    Primitive,
    Sequence,
    SequenceOf,
    Set,
    SetOf,
)
from bcir.asn1.tags import TagClass, Universal
from bcir.asn1.tlv import encode_tlv
from bcir.asn1.values import BitString
from bcir.asn1.xer import (
    ASN1_NAMESPACE,
    ASN1_NAMESPACE_OID,
    BASIC_XER_OID,
    CANONICAL_XER_OID,
    EXTENDED_XER_OID,
    XML_PROLOG,
    XerRules,
    XerTypeNames,
    canonical_realnumber,
    decode_xer,
    encode_xer,
    escape_xmlcstring,
    rules_oid,
    xml_type_name,
)
from bcir.frontends.asn1.lower import compile_module

# --- X.693 Annex A -----------------------------------------------------------------------

_ANNEX_A_MODULE = """
AnnexA DEFINITIONS ::= BEGIN
  PersonnelRecord ::= [APPLICATION 0] IMPLICIT SET {
      name Name,
      title [0] VisibleString,
      number EmployeeNumber,
      dateOfHire [1] Date,
      nameOfSpouse [2] Name,
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
        {
            "name": {"givenName": "Ralph", "initial": "T", "familyName": "Smith"},
            "dateOfBirth": "19571111",
        },
        {
            "name": {"givenName": "Susan", "initial": "B", "familyName": "Jones"},
            "dateOfBirth": "19590717",
        },
    ],
}

#: A.4, verbatim.
_ANNEX_A4 = (
    "<PersonnelRecord>"
    "<name><givenName>John</givenName><initial>P</initial>"
    "<familyName>Smith</familyName></name>"
    "<number>51</number>"
    "<title>Director</title>"
    "<dateOfHire>19710917</dateOfHire>"
    "<nameOfSpouse><givenName>Mary</givenName><initial>T</initial>"
    "<familyName>Smith</familyName></nameOfSpouse>"
    "<children>"
    "<ChildInformation><name><givenName>Ralph</givenName><initial>T</initial>"
    "<familyName>Smith</familyName></name><dateOfBirth>19571111</dateOfBirth>"
    "</ChildInformation>"
    "<ChildInformation><name><givenName>Susan</givenName><initial>B</initial>"
    "<familyName>Jones</familyName></name><dateOfBirth>19590717</dateOfBirth>"
    "</ChildInformation>"
    "</children>"
    "</PersonnelRecord>"
)

#: A.3, verbatim, indentation and all — including the trailing SPACEs after each
#: `<initial>` element and the SPACE inside the two `<name>` start tags, which are exactly
#: the "white-space" §8.1.4 permits between lexical items.
_ANNEX_A3 = """\
<PersonnelRecord>
\t<name>
\t\t<givenName>John</givenName>
\t\t<initial>P</initial>      \n\
\t\t<familyName>Smith</familyName>
\t</name>
\t<title>Director</title>
\t<number>51</number>
\t<dateOfHire>19710917</dateOfHire>
\t<nameOfSpouse>
\t\t<givenName>Mary</givenName>
\t\t<initial>T</initial>      \n\
\t\t<familyName>Smith</familyName>
\t</nameOfSpouse>
\t<children>
\t\t<ChildInformation>
\t\t\t<name> \n\
\t\t\t\t<givenName>Ralph</givenName>
\t\t\t\t<initial>T</initial>      \n\
\t\t\t\t<familyName>Smith</familyName>
\t\t\t</name>
\t\t\t<dateOfBirth>19571111</dateOfBirth>
\t\t</ChildInformation>
\t\t<ChildInformation>
\t\t\t<name> \n\
\t\t\t\t<givenName>Susan</givenName>
\t\t\t\t<initial>B</initial>      \n\
\t\t\t\t<familyName>Jones</familyName>
\t\t\t</name>
\t\t\t<dateOfBirth>19590717</dateOfBirth>
\t\t</ChildInformation>
\t</children>
</PersonnelRecord>"""


def _annex_a():
    lowered = compile_module(_ANNEX_A_MODULE, "<annex>")
    return (lowered.module.types["PersonnelRecord"], XerTypeNames(lowered.module))


def test_annex_a4_matches_the_specification_characters():
    """A.4: the standard's own CXER encoding of the record, character for character."""
    kind, names = _annex_a()
    got = encode_xer(kind, _PERSONNEL_RECORD, names=names).decode("utf-8")
    assert got == _ANNEX_A4, f"\n got {got}\n exp {_ANNEX_A4}"


def test_annex_a3_states_the_length_and_the_encoding_is_that_length():
    """A.3: "The length of this encoding in BASIC-XER is 653 octets ignoring white-space".

    Checked against the BASIC encoding rather than the canonical one, because that is what
    the clause measures — and the two differ only in the order of a SET's components, which
    moves no characters. An encoding that agreed with A.4 but not with this number would
    have to be right by accident.
    """
    kind, names = _annex_a()
    basic = encode_xer(kind, _PERSONNEL_RECORD, names=names, rules=XerRules.BASIC)
    assert len(basic) == 653, f"A.3 is 653 octets, got {len(basic)}"
    assert len(encode_xer(kind, _PERSONNEL_RECORD, names=names)) == 653


def test_annex_a3_as_printed_decodes_to_the_annex_a2_value():
    """§8.1.4/§8.3.4: the indentation A.3 prints is white-space an encoder may insert."""
    kind, names = _annex_a()
    assert decode_xer(_ANNEX_A3, kind, names=names, rules=XerRules.BASIC) == _PERSONNEL_RECORD


def test_annex_a4_decodes_and_round_trips_under_both_rule_sets():
    kind, names = _annex_a()
    assert decode_xer(_ANNEX_A4, kind, names=names) == _PERSONNEL_RECORD
    for rules in (XerRules.BASIC, XerRules.CANONICAL):
        data = encode_xer(kind, _PERSONNEL_RECORD, names=names, rules=rules)
        assert decode_xer(data, kind, names=names, rules=rules) == _PERSONNEL_RECORD


def test_a_cxer_encoding_is_also_a_basic_xer_encoding():
    """§5.3: CXER "is defined as a restriction of ... the BASIC-XER encoding".

    That is the whole basis for emitting canonically and accepting both, so it is pinned
    rather than assumed: the strict encoder's output must satisfy the lenient decoder.
    """
    kind, names = _annex_a()
    canonical = encode_xer(kind, _PERSONNEL_RECORD, names=names)
    assert decode_xer(canonical, kind, names=names, rules=XerRules.BASIC) == _PERSONNEL_RECORD


def test_the_set_components_move_into_canonical_tag_order():
    """§9.6.1 with X.680 §8.6 — the single most visible difference in Annex A.

    `PersonnelRecord` is written `name, title, number, ...` and A.4 prints
    `name, number, title, ...`, because `number` is [APPLICATION 2] and `title` is [0],
    and §8.6 orders APPLICATION ahead of context-specific regardless of number.
    """
    kind, names = _annex_a()
    canonical = encode_xer(kind, _PERSONNEL_RECORD, names=names).decode()
    basic = encode_xer(kind, _PERSONNEL_RECORD, names=names, rules=XerRules.BASIC).decode()
    assert canonical.index("<number>") < canonical.index("<title>")
    assert basic.index("<title>") < basic.index("<number>")
    # Reordering is not rewriting: the two encodings hold the same characters.
    assert sorted(canonical) == sorted(basic)


def test_a_set_of_untagged_choices_is_ordered_by_its_smallest_alternative_tag():
    """§9.6.1's extra rule, which exists nowhere else in the suite.

    A component that is an untagged CHOICE has no single tag to sort on, so §9.6.1 gives it
    "a tag equal to that of the smallest tag in the RootAlternativeTypeList of that choice
    type or any such choice types nested within it".
    """
    inner = Choice((Component("deep", Primitive(Universal.INTEGER, "INTEGER"), tag=9),))
    kind = Set(
        (
            Component("plain", Primitive(Universal.INTEGER, "INTEGER"), tag=5),
            Component(
                "picked",
                Choice(
                    (
                        Component("small", Primitive(Universal.INTEGER, "INTEGER"), tag=1),
                        Component("nested", inner, tag=None, explicit=True),
                    )
                ),
            ),
        )
    )
    text = encode_xer(kind, {"plain": 1, "picked": ("small", 2)}, name="T").decode()
    assert text.index("<picked>") < text.index("<plain>"), text


# --- clause 9: what makes CXER canonical -------------------------------------------------


def test_a_default_valued_component_is_present_under_cxer_and_absent_under_basic():
    """§9.5/§9.6.3 against X.690 §11.5 — two canonical rule sets, opposite answers."""
    kind = Sequence(
        (
            Component("x", Primitive(Universal.INTEGER, "INTEGER")),
            Component("flag", Primitive(Universal.BOOLEAN, "BOOLEAN"), tag=0, default=False),
        )
    )
    assert encode_xer(kind, {"x": 1}, name="T").decode() == "<T><x>1</x><flag><false/></flag></T>"
    assert (
        encode_xer(kind, {"x": 1, "flag": False}, name="T").decode()
        == "<T><x>1</x><flag><false/></flag></T>"
    )
    assert (
        encode_xer(kind, {"x": 1, "flag": False}, name="T", rules=XerRules.BASIC).decode()
        == "<T><x>1</x></T>"
    )
    # Either spelling decodes to the same abstract value (X.680 §25.12).
    for text in ("<T><x>1</x><flag><false/></flag></T>", "<T><x>1</x></T>"):
        assert decode_xer(text, kind, name="T", rules=XerRules.BASIC) == {"x": 1, "flag": False}


def test_set_of_is_sorted_by_the_code_points_of_its_element_encodings():
    """§9.7.2/§9.7.3 — a conceptual pad character "precedes all other characters".

    `"a"` therefore sorts before `"ab"`, which is exactly how Python ranks a prefix, so the
    rule needs no padding code. The elements are given out of order to prove the encoder
    does the sorting rather than the caller.
    """
    kind = SetOf(Primitive(Universal.UTF8_STRING, "UTF8String"))
    text = encode_xer(kind, ["b", "ab", "a"], name="T").decode()
    assert text == (
        "<T><UTF8String>a</UTF8String><UTF8String>ab</UTF8String><UTF8String>b</UTF8String></T>"
    )
    # BASIC has no such rule, so the caller's order survives (X.680 §28.3 NOTE 2).
    assert encode_xer(kind, ["b", "ab", "a"], name="T", rules=XerRules.BASIC).decode().index(
        "<UTF8String>b<"
    ) < encode_xer(kind, ["b", "ab", "a"], name="T", rules=XerRules.BASIC).decode().index(
        "<UTF8String>a<"
    )


def test_sequence_of_keeps_its_order_under_both_rule_sets():
    """X.680 §26.3 NOTE: order is significant in a SEQUENCE OF, so §9.7 does not apply."""
    kind = SequenceOf(Primitive(Universal.UTF8_STRING, "UTF8String"))
    for rules in (XerRules.BASIC, XerRules.CANONICAL):
        text = encode_xer(kind, ["b", "a"], name="T", rules=rules).decode()
        assert text.index("<UTF8String>b<") < text.index("<UTF8String>a<")


def test_canonical_real_normalization():
    """§9.2.1-§9.2.5: one non-zero integer digit, a trimmed fraction, `E`, no `+`."""
    assert canonical_realnumber(0.0) == "0"  # §9.2.1
    assert canonical_realnumber(1.0) == "1.0E0"  # §9.2.3: a fraction digit
    assert canonical_realnumber(-1.0) == "-1.0E0"  # §9.2.5: no "+" for "-1"
    assert canonical_realnumber(0.5) == "5.0E-1"
    assert canonical_realnumber(314.159) == "3.14159E2"
    assert canonical_realnumber(1234.0) == "1.234E3"  # trailing zeros trimmed
    assert canonical_realnumber(1e300) == "1.0E300"  # §9.2.5: no "+" exponent
    assert canonical_realnumber(1e-5) == "1.0E-5"
    for text in ("+", "e"):
        assert text not in canonical_realnumber(1e300)


def test_special_real_values_use_the_empty_element_form():
    """§8.3.8 — `XMLSpecialRealValue` shall only be `EmptyElementReal`."""
    kind = Primitive(Universal.REAL, "REAL")
    assert encode_xer(kind, float("inf"), name="T").decode() == "<T><PLUS-INFINITY/></T>"
    assert encode_xer(kind, float("-inf"), name="T").decode() == "<T><MINUS-INFINITY/></T>"
    assert encode_xer(kind, float("nan"), name="T").decode() == "<T><NOT-A-NUMBER/></T>"
    assert decode_xer("<T><PLUS-INFINITY/></T>", kind, name="T") == float("inf")
    nan = decode_xer("<T><NOT-A-NUMBER/></T>", kind, name="T")
    assert nan != nan
    _refuses(lambda: decode_xer("<T><INF/></T>", kind, name="T"), "XMLSpecialRealValue")


def test_octetstring_hex_is_upper_case_with_no_white_space():
    """§9.4, and X.680 §12.13's `xmlhstring` on the way back."""
    kind = Primitive(Universal.OCTET_STRING, "OCTET STRING")
    assert encode_xer(kind, b"\xde\xad\xbe\xef", name="T").decode() == "<T>DEADBEEF</T>"
    assert decode_xer("<T>de ad\tbe\nef</T>", kind, name="T") == b"\xde\xad\xbe\xef"
    _refuses(lambda: decode_xer("<T>DEA</T>", kind, name="T"), "even number")


def test_bitstring_is_an_xmlbstring_and_never_an_identifier_list():
    """§8.3.9 removes the `XMLIdentifierList` alternative of X.680 §22.9."""
    kind = Primitive(Universal.BIT_STRING, "BIT STRING")
    assert encode_xer(kind, BitString(b"\xa0", 5), name="T").decode() == "<T>101</T>"
    assert decode_xer("<T>101</T>", kind, name="T") == BitString(b"\xa0", 5)
    assert decode_xer("<T/>", kind, name="T") == BitString(b"", 0)
    _refuses(lambda: decode_xer("<T>1<flagName/>1</T>", kind, name="T"), "an element interrupts")


def test_canonical_time_trims_the_fraction_and_demands_zulu_and_seconds():
    """§9.10.1-§9.10.4 and §9.11.1-§9.11.2, with no time arithmetic anywhere."""
    generalized = Primitive(Universal.GENERALIZED_TIME, "GeneralizedTime")
    assert (
        encode_xer(generalized, "19710917123456.500Z", name="T").decode()
        == "<T>19710917123456.5Z</T>"
    )
    assert (
        encode_xer(generalized, "19710917123456.000Z", name="T").decode()
        == "<T>19710917123456Z</T>"
    )  # §9.10.3
    assert (
        encode_xer(generalized, "19710917123456,5Z", name="T").decode()
        == "<T>19710917123456.5Z</T>"
    )  # §9.10.4
    _refuses(lambda: encode_xer(generalized, "19710917123456+0100", name="T"), "terminate with")
    utc = Primitive(Universal.UTC_TIME, "UTCTime")
    _refuses(lambda: encode_xer(utc, "9205211234Z", name="T"), "seconds")
    # BASIC-XER states no such rule, so the value notation passes through unchanged.
    assert (
        encode_xer(utc, "9205211234Z", name="T", rules=XerRules.BASIC).decode()
        == "<T>9205211234Z</T>"
    )


def test_the_time_type_is_refused_by_cxer_rather_than_approximated():
    """§9.13 b)-d) operate on a parsed duration or interval, which this rail has not got."""
    for universal in (Universal.TIME, Universal.DURATION, Universal.DATE_TIME):
        kind = Primitive(universal, "TIME")
        _refuses(lambda k=kind: encode_xer(k, "P1Y2M", name="T"), "9.13")
        assert encode_xer(kind, "P1Y2M", name="T", rules=XerRules.BASIC).decode() == "<T>P1Y2M</T>"


# --- clause 8: the BASIC-XER restrictions on X.680's notation ----------------------------


def test_boolean_is_the_empty_element_form_only():
    """§8.3.5 — the text `true`/`false` spelling is an EXTENDED-XER addition."""
    kind = Primitive(Universal.BOOLEAN, "BOOLEAN")
    assert encode_xer(kind, True, name="T").decode() == "<T><true/></T>"
    assert encode_xer(kind, False, name="T").decode() == "<T><false/></T>"
    assert decode_xer("<T><false/></T>", kind, name="T") is False
    _refuses(lambda: decode_xer("<T>true</T>", kind, name="T"), "8.3.5")


def test_integer_is_an_xmlsignednumber_only():
    """§8.3.6 removes both `identifier` forms of X.680 §19.9."""
    kind = Primitive(Universal.INTEGER, "INTEGER")
    assert encode_xer(kind, -42, name="T").decode() == "<T>-42</T>"
    assert encode_xer(kind, 0, name="T").decode() == "<T>0</T>"  # X.680 §19.13
    assert decode_xer("<T> -42 </T>", kind, name="T") == -42
    _refuses(lambda: decode_xer("<T><one/></T>", kind, name="T"), "8.3.6")
    _refuses(lambda: decode_xer("<T>01</T>", kind, name="T"), "leading zero")
    _refuses(lambda: decode_xer("<T>-0</T>", kind, name="T"), "19.13")


def test_enumerated_has_no_numeric_spelling_at_all():
    """§8.3.7 with X.680 §20.8 — unlike INTEGER, the identifier is the ONLY form.

    This is why `Primitive.enumeration` is as load-bearing for XER as it is for PER's
    §14.1 index, and for the opposite reason: PER cannot find the index without it, XER
    cannot find a name.
    """
    kind = Primitive(
        Universal.ENUMERATED, "ENUMERATED", enumeration=(("red", 0), ("green", 1), ("blue", 2))
    )
    assert encode_xer(kind, 1, name="T").decode() == "<T><green/></T>"
    assert encode_xer(kind, "blue", name="T").decode() == "<T><blue/></T>"
    assert decode_xer("<T><blue/></T>", kind, name="T") == 2
    _refuses(lambda: decode_xer("<T>1</T>", kind, name="T"), "8.3.7")
    _refuses(lambda: encode_xer(kind, 7, name="T"), "no numeric spelling")
    bare = Primitive(Universal.ENUMERATED, "ENUMERATED")
    _refuses(lambda: encode_xer(bare, 1, name="T"), "no enumeration")


def test_null_and_every_empty_value_use_the_empty_element_tag():
    """X.680 §17.8 as an option, §9.1.4 as a requirement."""
    assert encode_xer(Primitive(Universal.NULL, "NULL"), None, name="T").decode() == "<T/>"
    assert (
        encode_xer(Primitive(Universal.UTF8_STRING, "UTF8String"), "", name="T").decode() == "<T/>"
    )
    assert (
        encode_xer(SequenceOf(Primitive(Universal.INTEGER, "INTEGER")), [], name="T").decode()
        == "<T/>"
    )
    # X.680 §17.8: `<T></T>` denotes the same value and is what the option replaces.
    assert decode_xer("<T></T>", Primitive(Universal.NULL, "NULL"), name="T") is None
    assert (
        decode_xer("<T></T>", SequenceOf(Primitive(Universal.INTEGER, "INTEGER")), name="T") == []
    )


def test_sequence_of_null_spells_each_element_as_an_empty_element():
    """X.680 §26.4 and its NOTE: "This occurs only for SEQUENCE OF NULL"."""
    kind = SequenceOf(Primitive(Universal.NULL, "NULL"))
    assert encode_xer(kind, [None, None], name="T").decode() == "<T><NULL/><NULL/></T>"
    assert decode_xer("<T><NULL/><NULL/></T>", kind, name="T") == [None, None]


def test_the_list_notation_follows_table_5():
    """X.680 Table 5 with §26.6/§26.7 — which element types are delimited and which are not.

    BOOLEAN and ENUMERATED are conditional in Table 5, and X.693 §8.3.5/§8.3.7 settle the
    condition by making the empty-element form the only one available.
    """
    boolean = SequenceOf(Primitive(Universal.BOOLEAN, "BOOLEAN"))
    assert encode_xer(boolean, [True, False], name="T").decode() == "<T><true/><false/></T>"
    enumerated = SequenceOf(
        Primitive(Universal.ENUMERATED, "ENUMERATED", enumeration=(("red", 0), ("blue", 1)))
    )
    assert encode_xer(enumerated, [0, 1], name="T").decode() == "<T><red/><blue/></T>"
    choice = SequenceOf(
        Choice(
            (
                Component("a", Primitive(Universal.INTEGER, "INTEGER"), tag=0),
                Component("b", Primitive(Universal.INTEGER, "INTEGER"), tag=1),
            )
        )
    )
    assert encode_xer(choice, [("a", 1), ("b", 2)], name="T").decode() == "<T><a>1</a><b>2</b></T>"
    integer = SequenceOf(Primitive(Universal.INTEGER, "INTEGER"))
    assert (
        encode_xer(integer, [1, 2], name="T").decode()
        == "<T><INTEGER>1</INTEGER><INTEGER>2</INTEGER></T>"
    )


def test_a_delimited_item_is_named_by_the_typereference_when_there_is_one():
    """X.680 §26.10 — Annex A's `<ChildInformation>` rather than Table 4's `<SET>`."""
    lowered = compile_module(_ANNEX_A_MODULE, "<annex>")
    names = XerTypeNames(lowered.module)
    children = lowered.module.types["PersonnelRecord"]
    element = next(c for c in children.components if c.name == "children").type.element
    assert xml_type_name(element, names) == "ChildInformation"
    assert xml_type_name(element) == "SET"  # X.680 Table 4, no names
    assert xml_type_name(Primitive(Universal.IA5_STRING, "IA5String")) == "IA5String"
    assert xml_type_name(SequenceOf(Primitive(Universal.NULL, "NULL"))) == "SEQUENCE_OF"


def test_a_type_name_beginning_with_xml_gets_a_low_line():
    """X.680 §14.2 — XML 1.0 reserves every name starting with "xml"."""
    kind = Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),))
    names = XerTypeNames({"XMLThing": kind})
    assert xml_type_name(kind, names) == "_XMLThing"
    assert encode_xer(kind, {"x": 1}, names=names).decode() == "<_XMLThing><x>1</x></_XMLThing>"


def test_object_identifier_uses_the_number_form():
    """§9.8/§9.9 — `XMLObjIdComponent` shall be `XMLNumberForm`, so no `name(number)`."""
    kind = Primitive(Universal.OBJECT_IDENTIFIER, "OBJECT IDENTIFIER")
    assert encode_xer(kind, (2, 1, 5, 1), name="T").decode() == "<T>2.1.5.1</T>"
    assert decode_xer("<T>2.1.5.1</T>", kind, name="T") == (2, 1, 5, 1)
    _refuses(lambda: decode_xer("<T>joint-iso-itu-t.1</T>", kind, name="T"), "XMLNumberForm")


# --- X.680 §12.15: the xmlcstring lexical item -------------------------------------------


def test_the_markup_characters_are_escaped_and_only_ever_the_named_way():
    """X.680 §12.15.2/§12.15.4 with §9.1.3 — the numeric escapes are unavailable to CXER."""
    assert escape_xmlcstring("a<b>&c") == "a&lt;b&gt;&amp;c"
    kind = Primitive(Universal.UTF8_STRING, "UTF8String")
    assert decode_xer("<T>a&lt;b&gt;&amp;c</T>", kind, name="T") == "a<b>&c"
    _refuses(lambda: decode_xer("<T>a>b</T>", kind, name="T"), "12.15.2")
    _refuses(lambda: decode_xer("<T>a&nbsp;b</T>", kind, name="T"), "12.15.4")


def test_control_characters_become_the_table_3_empty_elements():
    """X.680 §12.15.5 — and note that 9, 10 and 13 are NOT in Table 3."""
    assert escape_xmlcstring("\x00\x1f") == "<nul/><is1/>"
    assert escape_xmlcstring("\t\n\r") == "\t\n\r"
    kind = Primitive(Universal.UTF8_STRING, "UTF8String")
    text = encode_xer(kind, "a\x01b\x1fc", name="T").decode()
    assert text == "<T>a<soh/>b<is1/>c</T>"
    assert decode_xer(text, kind, name="T") == "a\x01b\x1fc"
    _refuses(lambda: decode_xer("<T>a<nope/>b</T>", kind, name="T"), "Table 3")


def test_a_numeric_escape_is_basic_only():
    """§9.1.3: "The escape sequences specified in ... 12.15.8 shall not be used"."""
    kind = Primitive(Universal.UTF8_STRING, "UTF8String")
    assert decode_xer("<T>a&#233;b&#xEE;c</T>", kind, name="T", rules=XerRules.BASIC) == "aébîc"
    _refuses(lambda: decode_xer("<T>a&#233;</T>", kind, name="T"), "9.1.3")


def test_a_character_outside_the_xmlcstring_repertoire_is_refused():
    """X.680 §12.15.1 with §41.10's NOTE — there is no spelling to fall back to."""
    kind = Primitive(Universal.UTF8_STRING, "UTF8String")
    for character in ("\ud800", "￾", "￿"):
        _refuses(lambda c=character: encode_xer(kind, "a" + c, name="T"), "12.15.1")


def test_white_space_is_stripped_around_a_number_and_kept_inside_a_string():
    """X.680 §16.2 permits it around an XMLValue; §41.9 carves out the string case."""
    assert decode_xer("<T>\n  42 </T>", Primitive(Universal.INTEGER, "INTEGER"), name="T") == 42
    assert (
        decode_xer("<T> a b </T>", Primitive(Universal.UTF8_STRING, "UTF8String"), name="T")
        == " a b "
    )


# --- clause 8.6: decoding types with extension markers -----------------------------------


def test_an_unknown_extension_element_is_skipped_whole():
    """§8.6.1/§8.6.2 — a decoder "shall accept" an encoding carrying unknown extensions."""
    kind = Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),), extensible=True)
    assert decode_xer("<T><x>1</x><future>9</future></T>", kind, name="T") == {"x": 1}
    # The skip is structural, so a whole subtree goes with it.
    assert decode_xer("<T><x>1</x><future><a>1</a><b/></future></T>", kind, name="T") == {"x": 1}
    closed = Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),))
    _refuses(
        lambda: decode_xer("<T><x>1</x><future>9</future></T>", closed, name="T"),
        "no extension marker",
    )


def test_an_unknown_choice_alternative_keeps_its_raw_xml():
    """§8.6.3 — one unexpected element in place of a known alternative.

    There is no type to decode it with, so the characters are kept exactly as they arrived,
    the same posture an unresolvable open type takes with its octets. `encode_xer` refuses
    the pair rather than emitting text it did not build.
    """
    kind = Choice(
        (Component("a", Primitive(Universal.INTEGER, "INTEGER"), tag=0),), extensible=True
    )
    assert decode_xer("<T><later>9</later></T>", kind, name="T") == ("later", "<later>9</later>")
    _refuses(lambda: encode_xer(kind, ("later", "<later>9</later>"), name="T"), "not re-encodable")
    closed = Choice((Component("a", Primitive(Universal.INTEGER, "INTEGER"), tag=0),))
    _refuses(lambda: decode_xer("<T><later>9</later></T>", closed, name="T"), "no extension marker")


def test_a_set_accepts_its_components_in_any_order():
    """X.680 §27.9 NOTE — and a SEQUENCE does not (§25.20)."""
    components = (
        Component("a", Primitive(Universal.INTEGER, "INTEGER"), tag=0),
        Component("b", Primitive(Universal.INTEGER, "INTEGER"), tag=1),
    )
    assert decode_xer("<T><b>2</b><a>1</a></T>", Set(components), name="T") == {"a": 1, "b": 2}
    _refuses(
        lambda: decode_xer("<T><b>2</b><a>1</a></T>", Sequence(components), name="T"),
        "missing or out of order",
    )


# --- clause 8.2/8.1.2: the document, and what is not part of one -------------------------


def test_the_prolog_is_optional_for_basic_and_forbidden_for_cxer():
    """§8.2.1 gives two alternatives; §9.1.1 deletes the second."""
    kind = Primitive(Universal.INTEGER, "INTEGER")
    assert (
        encode_xer(kind, 1, name="T", prolog=True, rules=XerRules.BASIC).decode()
        == XML_PROLOG + "<T>1</T>"
    )
    assert decode_xer(XML_PROLOG + " <T>1</T>", kind, name="T", rules=XerRules.BASIC) == 1
    _refuses(lambda: encode_xer(kind, 1, name="T", prolog=True), "9.1.1")
    _refuses(lambda: decode_xer(XML_PROLOG + "<T>1</T>", kind, name="T"), "9.1.1")
    _refuses(
        lambda: decode_xer('<?xml version="1.0"?><T>1</T>', kind, name="T", rules=XerRules.BASIC),
        "8.2.1",
    )


def test_the_constructs_a_conforming_encoder_never_produces_are_refused_by_name():
    """§8.1.2's NOTE, and the EXTENDED-XER surface that is deliberately not implemented.

    A general XML parser accepts every one of these and hands back something that looks
    decoded. Naming each one against the clause that excludes it is the reason the reader
    in `xer.py` is hand-written.
    """
    kind = Primitive(Universal.INTEGER, "INTEGER")
    for text, expected in (
        ("<T><!-- c -->1</T>", "comments"),
        ("<T><?go?>1</T>", "processing instruction"),
        ("<!DOCTYPE T><T>1</T>", "document type declaration"),
        ("<T><![CDATA[1]]></T>", "CDATA"),
        ('<T a="1">1</T>', "ATTRIBUTE instruction"),
        ("<a:T>1</a:T>", "NAMESPACE instruction"),
    ):
        _refuses(lambda t=text: decode_xer(t, kind, name="T"), expected)


def test_one_document_element_and_no_more():
    """§8.1.1 — an XER encoding is a prolog and ONE document element."""
    kind = Primitive(Universal.INTEGER, "INTEGER")
    assert decode_xer("<T>1</T>\n ", kind, name="T") == 1  # §8.2.1 b) white-space
    _refuses(lambda: decode_xer("<T>1</T><T>2</T>", kind, name="T"), "8.1.1")


def test_the_encoding_is_utf_8_octets():
    """§8.1.3 — "shall be encoded using UTF-8 to produce a string of octets"."""
    kind = Primitive(Universal.UTF8_STRING, "UTF8String")
    assert encode_xer(kind, "é", name="T") == b"<T>\xc3\xa9</T>"
    assert decode_xer(b"<T>\xc3\xa9</T>", kind, name="T") == "é"
    assert (
        decode_xer(
            b"\xef\xbb\xbf<T>1</T>",  # a byte order mark
            Primitive(Universal.INTEGER, "INTEGER"),
            name="T",
        )
        == 1
    )
    _refuses(lambda: decode_xer(b"<T>\xff</T>", kind, name="T"), "8.1.3")


# --- open types and contents constraints -------------------------------------------------


def _inner_and_octets():
    inner = Sequence((Component("n", Primitive(Universal.INTEGER, "INTEGER")),), "Inner")
    return inner, encode_tlv(inner.encode({"n": 7}))


def test_a_containing_constraint_chooses_the_encoding_under_cxer():
    """§9.4 with X.680 §23.4 — "if ... can be used, then it shall be used"."""
    inner, octets = _inner_and_octets()
    names = XerTypeNames({"Inner": inner})
    kind = Primitive(Universal.OCTET_STRING, "OCTET STRING", contains=inner)
    assert (
        encode_xer(kind, octets, name="T", names=names).decode() == "<T><Inner><n>7</n></Inner></T>"
    )
    assert (
        encode_xer(kind, octets, name="T", names=names, rules=XerRules.BASIC).decode()
        == f"<T>{octets.hex().upper()}</T>"
    )
    assert decode_xer("<T><Inner><n>7</n></Inner></T>", kind, name="T", names=names) == octets


def test_an_encoded_by_constraint_keeps_the_string_form():
    """X.680 §22.11/§23.4 both say "and does not include an ENCODED BY".

    Naming other rules is a statement about the octets, not a licence to read them as this
    rail's own type — so the alternative stays unavailable and the hex form is emitted.
    """
    inner, octets = _inner_and_octets()
    kind = Primitive(
        Universal.OCTET_STRING, "OCTET STRING", contains=inner, encoded_by=(2, 1, 2, 1)
    )
    assert encode_xer(kind, octets, name="T").decode() == f"<T>{octets.hex().upper()}</T>"


def test_an_open_type_is_hex_under_basic_and_its_selected_type_under_cxer():
    """§8.5 with X.681 §14.6, and §9.12 removing the `xmlhstring` alternative."""
    inner, octets = _inner_and_octets()
    names = XerTypeNames({"Inner": inner})
    table = ObjectSetTable("C", ({"&id": 1, "&Type": inner},))
    opened = OpenType(
        "OPEN", table=table, field="&Type", governing=(("id",),), governing_fields=("&id",)
    )
    kind = Sequence(
        (Component("id", Primitive(Universal.INTEGER, "INTEGER")), Component("body", opened)),
        "Outer",
    )
    value = {"id": 1, "body": octets}
    assert (
        encode_xer(kind, value, name="T", names=names).decode()
        == "<T><id>1</id><body><Inner><n>7</n></Inner></body></T>"
    )
    assert (
        encode_xer(kind, value, name="T", names=names, rules=XerRules.BASIC).decode()
        == f"<T><id>1</id><body>{octets.hex().upper()}</body></T>"
    )
    decoded = decode_xer(
        "<T><id>1</id><body><Inner><n>7</n></Inner></body></T>", kind, name="T", names=names
    )
    assert decoded["body"] == octets
    # The X.682 §10.19 enrichment lands here exactly as it does on the DER rail.
    assert decoded["body.resolved"] == {"n": 7}


def test_an_unresolvable_open_type_has_no_canonical_encoding():
    """§9.12 leaves only the typed form, and X.681 §12.9 permits an unknown object."""
    inner, octets = _inner_and_octets()
    table = ObjectSetTable("C", ({"&id": 1, "&Type": inner},), extensible=True)
    opened = OpenType(
        "OPEN", table=table, field="&Type", governing=(("id",),), governing_fields=("&id",)
    )
    kind = Sequence(
        (Component("id", Primitive(Universal.INTEGER, "INTEGER")), Component("body", opened)),
        "Outer",
    )
    unknown = {"id": 9, "body": octets}
    _refuses(lambda: encode_xer(kind, unknown, name="T"), "9.12")
    assert (
        encode_xer(kind, unknown, name="T", rules=XerRules.BASIC).decode()
        == f"<T><id>9</id><body>{octets.hex().upper()}</body></T>"
    )


# --- clause 40 ---------------------------------------------------------------------------


def test_the_rule_object_identifiers():
    """§40.2/§40.3 — {joint-iso-itu-t asn1(1) xml-encoding(5) ...}."""
    assert BASIC_XER_OID == (2, 1, 5, 0)
    assert CANONICAL_XER_OID == (2, 1, 5, 1)
    assert EXTENDED_XER_OID == (2, 1, 5, 2)
    assert ASN1_NAMESPACE_OID == (2, 1, 5, 2, 0, 1)
    assert ASN1_NAMESPACE == "urn:oid:2.1.5.2.0.1"  # §16.9
    assert rules_oid(XerRules.BASIC) == BASIC_XER_OID
    assert rules_oid(XerRules.CANONICAL) == CANONICAL_XER_OID


def _refuses(action, needle: str) -> None:
    try:
        action()
    except Asn1Error as error:
        assert needle in str(error), f"expected {needle!r} in {error}"
        return
    raise AssertionError(f"expected a refusal mentioning {needle!r}")
