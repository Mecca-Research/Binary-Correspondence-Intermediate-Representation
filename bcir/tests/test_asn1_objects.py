"""X.681 information objects and X.682 constraints.

The centrepiece is X.682 clause 10's own worked example (ERROR-CLASS / ErrorSet /
ErrorReturn), reproduced verbatim from the standard. It exercises the whole chain in one
module: a class with WITH SYNTAX, an object set written positionally, the associated table
of X.681 §13, a simple table constraint on a value field, and a component relation
constraint that selects a row from two siblings and thereby names the type of an open type.

The law that matters most here is a NEGATIVE one. X.691 §10.3.4 and §10.3.5 make table and
component relation constraints **not PER-visible**, and X.696 agrees; so however much this
machinery narrows a value set, it must not move a single bit of any encoding. That is why
a table-constrained value field's column lands on `Primitive.table_values` and never on
`Primitive.constraint` -- the latter is what OER and PER read to choose a field's width.
"""

from __future__ import annotations

from bcir.asn1.oer import OerRules, encode_oer
from bcir.asn1.per import PerVariant, encode_per
from bcir.asn1.schema import OpenType, Primitive
from bcir.asn1.tags import Universal
from bcir.asn1.tlv import encode_tlv
from bcir.frontends.asn1 import Asn1SemanticError, compile_module

# X.682 clause 10, verbatim apart from whitespace.
_ERROR_MODULE = """
ErrorModule DEFINITIONS ::= BEGIN
  ERROR-CLASS ::= CLASS
  {
      &category  PrintableString (SIZE(1)),
      &code      INTEGER,
      &Type
  }
  WITH SYNTAX {&category &code &Type}

  ErrorSet ERROR-CLASS ::=
  {
      {"A" 1 INTEGER} |
      {"A" 2 BOOLEAN} |
      {"B" 1 PrintableString} |
      {"B" 2 OCTET STRING}
  }

  ErrorReturn ::= SEQUENCE
  {
      errorCategory ERROR-CLASS.&category ({ErrorSet}),
      errorCode     ERROR-CLASS.&code ({ErrorSet}{@errorCategory}),
      errorInfo     ERROR-CLASS.&Type ({ErrorSet}{@errorCategory,@errorCode})
  }
END
"""


def _error_module():
    return compile_module(_ERROR_MODULE, "<x682-10>").module


def _wrap(kind: Primitive, value) -> bytes:
    """One component's complete encoding, which is what an open type carries."""
    return encode_tlv(kind.encode(value))


def test_the_associated_table_has_one_row_per_object_and_one_column_per_field():
    """X.681 §13.1/§13.4: rows are objects, columns are the class's fields."""
    module = _error_module()
    open_type = module.types["ErrorReturn"].components[-1].type
    table = open_type.table
    assert table is not None, "the table constraint named an object set with no table"
    assert table.object_class == "ERROR-CLASS"
    assert len(table.rows) == 4, f"ErrorSet has four objects, table has {len(table.rows)}"
    assert table.column("&category") == ("A", "A", "B", "B")
    assert table.column("&code") == (1, 2, 1, 2)
    # §13.1: a TYPE field's column is a column of TYPES, which is what makes an open type
    # resolvable at all.
    types = table.column("&Type")
    assert [t.universal for t in types] == [
        Universal.INTEGER,
        Universal.BOOLEAN,
        Universal.PRINTABLE_STRING,
        Universal.OCTET_STRING,
    ]


def test_with_syntax_lets_an_object_be_written_positionally():
    """X.681 §11.4: a class WITH SYNTAX spells its objects in DefinedSyntax.

    `{"A" 1 INTEGER}` carries no field names, so the syntax list's field order is the only
    thing that can turn it back into field settings. Dropping the WITH SYNTAX clause -- as
    the front-end used to -- leaves every such object unreadable and every table empty.
    """
    module = _error_module()
    rows = module.types["ErrorReturn"].components[-1].type.table.rows
    assert rows[0]["&category"] == "A" and rows[0]["&code"] == 1
    assert rows[3]["&category"] == "B" and rows[3]["&code"] == 2


def test_a_simple_table_constraint_restricts_a_value_field_to_its_column():
    """X.682 §10.6 b): a value field is constrained to the values in its column."""
    module = _error_module()
    category = module.types["ErrorReturn"].components[0].type
    assert category.table_values == ("A", "A", "B", "B")


def test_a_table_constraint_never_moves_a_bit_of_any_encoding():
    """X.691 §10.3.4/§10.3.5 and X.696: table constraints are NOT visible to the encoder.

    This is the one that would be silently catastrophic to get wrong. A table constraint
    narrows a value set, and OER and PER choose a field's WIDTH from a constraint -- so if
    the column were attached as an ordinary subtype constraint, adding a table constraint to
    a module would change its wire format while looking like a documentation change.
    """
    shared = """
      C ::= CLASS { &code INTEGER (0..255), &Type } WITH SYNTAX {&code &Type}
      S C ::= { {1 INTEGER} | {2 BOOLEAN} }
    """
    with_table = compile_module(
        f"M DEFINITIONS ::= BEGIN {shared}\n"
        "  T ::= SEQUENCE {{ code C.&code ({{S}}) }}\n".replace("{{", "{").replace("}}", "}")
        + "END\n",
        "<with>",
    ).module.types["T"]
    without = compile_module(
        f"M DEFINITIONS ::= BEGIN {shared}\n  T ::= SEQUENCE {{ code C.&code }}\n".replace(
            "{{", "{"
        ).replace("}}", "}")
        + "END\n",
        "<without>",
    ).module.types["T"]

    assert with_table.components[0].type.table_values == (1, 2)
    assert with_table.components[0].type.constraint == without.components[0].type.constraint

    value = {"code": 2}
    for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
        assert encode_per(with_table, value, variant=variant) == encode_per(
            without, value, variant=variant
        ), variant
    assert encode_oer(with_table, value, rules=OerRules.CANONICAL) == encode_oer(
        without, value, rules=OerRules.CANONICAL
    )


def test_a_component_relation_constraint_resolves_the_open_type():
    """X.682 §10.19/§10.20 through a real DER round trip.

    This is what the whole chapter is for: `errorInfo`'s type is not in the schema, it is in
    the TABLE, selected by the values of two sibling components. The decoder therefore has
    to resolve it while it walks -- which it can, because §10.15 puts the referenced
    components in the same enclosing type and a SEQUENCE decodes in definition order.
    """
    module = _error_module()
    cases = [
        ("A", 1, Primitive(Universal.INTEGER, "INTEGER"), 42),
        ("A", 2, Primitive(Universal.BOOLEAN, "BOOLEAN"), True),
        ("B", 1, Primitive(Universal.PRINTABLE_STRING, "PrintableString"), "hi"),
        ("B", 2, Primitive(Universal.OCTET_STRING, "OCTET STRING"), b"\xde\xad"),
    ]
    for category, code, kind, value in cases:
        octets = _wrap(kind, value)
        der = module.encode(
            "ErrorReturn", {"errorCategory": category, "errorCode": code, "errorInfo": octets}
        )
        back = module.decode("ErrorReturn", der)
        assert back["errorInfo"] == octets, "the octets must survive unchanged"
        assert back["errorInfo.resolved"] == value, (
            f"row {category}/{code} should resolve to {value!r}, "
            f"got {back.get('errorInfo.resolved')!r}"
        )


def test_an_unknown_row_keeps_the_octets_instead_of_guessing():
    """X.681 §12.9: a peer may legitimately use an object outside the set.

    So an unresolvable open type is ordinary traffic, not a fault. The octets stay, and no
    `.resolved` key appears -- the absence is the honest signal.
    """
    module = _error_module()
    octets = _wrap(Primitive(Universal.INTEGER, "INTEGER"), 7)
    der = module.encode("ErrorReturn", {"errorCategory": "B", "errorCode": 9, "errorInfo": octets})
    back = module.decode("ErrorReturn", der)
    assert back["errorInfo"] == octets
    assert "errorInfo.resolved" not in back, (
        "an unmatched row must not be resolved to some other row's type"
    )


def test_an_object_set_unions_references_and_inherits_extensibility():
    """X.681 §12.3 (union, `...`) and §12.5 (a referenced set's marker is inherited)."""
    module = compile_module(
        """
      M DEFINITIONS ::= BEGIN
        C ::= CLASS { &code INTEGER, &Type } WITH SYNTAX {&code &Type}
        base C ::= {3 BOOLEAN}
        Inner C ::= { {1 INTEGER}, ... }
        Outer C ::= { Inner | {2 PrintableString} | base }
        T ::= SEQUENCE { code C.&code ({Outer}) }
      END
    """,
        "<sets>",
    ).module
    values = module.types["T"].components[0].type.table_values
    assert values == (1, 2, 3), f"the union should hold all three objects, got {values}"


def test_a_contents_constraint_resolves_the_contained_value():
    """X.682 §11.4: the octet string's abstract value IS an encoding of the named type."""
    module = compile_module(
        """
      M DEFINITIONS ::= BEGIN
        T ::= SEQUENCE { blob OCTET STRING (CONTAINING INTEGER ENCODED BY {2 1 1}) }
      END
    """,
        "<contents>",
    ).module
    blob = module.types["T"].components[0].type
    assert blob.contains is not None and blob.contains.universal == Universal.INTEGER
    assert blob.encoded_by == (2, 1, 1), "§11.2's object identifier must be recorded"
    octets = _wrap(Primitive(Universal.INTEGER, "INTEGER"), 1234)
    back = module.decode("T", module.encode("T", {"blob": octets}))
    assert back["blob"] == octets
    assert back["blob.resolved"] == 1234


def test_a_contents_constraint_is_refused_where_it_cannot_apply():
    """§11.3 allows it only on OCTET STRING and BIT STRING."""
    try:
        compile_module(
            "M DEFINITIONS ::= BEGIN\n  T ::= INTEGER (CONTAINING BOOLEAN)\nEND\n", "<bad>"
        )
        raise AssertionError("a contents constraint on an INTEGER must be refused")
    except Asn1SemanticError as exc:
        assert "11.3" in str(exc), exc


def test_a_user_defined_constraint_is_recorded_and_changes_no_encoding():
    """X.682 §9 NOTE 1 calls it "a special form of ASN.1 comment", and X.691 §10.3.3 makes
    it not PER-visible -- so it must parse, and it must not reach the encoder."""
    constrained = compile_module(
        "M DEFINITIONS ::= BEGIN\n"
        "  T ::= SEQUENCE { v INTEGER (0..255) (CONSTRAINED BY {-- prime --}) }\nEND\n",
        "<udc>",
    ).module.types["T"]
    plain = compile_module(
        "M DEFINITIONS ::= BEGIN\n  T ::= SEQUENCE { v INTEGER (0..255) }\nEND\n", "<plain>"
    ).module.types["T"]
    for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
        assert encode_per(constrained, {"v": 7}, variant=variant) == encode_per(
            plain, {"v": 7}, variant=variant
        )


def test_an_x509_shaped_attribute_resolves_by_its_sibling_oid():
    """The pattern the whole chapter exists for, in the shape X.509 actually uses.

    `AttributeTypeAndValue` is `{ type ATTRIBUTE.&id, value ATTRIBUTE.&Type }` where the OID
    in `type` selects the type of `value`. Before this, `value` decoded to opaque octets and
    a caller had to know the mapping out of band; now the module carries it.
    """
    module = compile_module(
        """
      Pkix DEFINITIONS ::= BEGIN
        ATTRIBUTE ::= CLASS { &id OBJECT IDENTIFIER UNIQUE, &Type }
          WITH SYNTAX {&Type IDENTIFIED BY &id}
        SupportedAttributes ATTRIBUTE ::= {
            {PrintableString IDENTIFIED BY {2 5 4 6}} |
            {UTF8String IDENTIFIED BY {2 5 4 3}} }
        AttributeTypeAndValue ::= SEQUENCE {
            type  ATTRIBUTE.&id ({SupportedAttributes}),
            value ATTRIBUTE.&Type ({SupportedAttributes}{@type}) }
      END
    """,
        "<pkix>",
    ).module
    from bcir.asn1.codec import Oid

    for arcs, kind, value in (
        ((2, 5, 4, 6), Primitive(Universal.PRINTABLE_STRING, "PrintableString"), "GB"),
        ((2, 5, 4, 3), Primitive(Universal.UTF8_STRING, "UTF8String"), "Example CA"),
    ):
        octets = _wrap(kind, value)
        der = module.encode("AttributeTypeAndValue", {"type": Oid(arcs), "value": octets})
        back = module.decode("AttributeTypeAndValue", der)
        assert back["value"] == octets
        assert back["value.resolved"] == value, (
            f"{arcs} should select {kind.name} and decode to {value!r}, "
            f"got {back.get('value.resolved')!r}"
        )


def test_an_open_type_with_no_table_constraint_stays_opaque():
    """A bare `CLASS.&Type` names no object set, so there is nothing to resolve against.

    It must keep behaving exactly as it did -- opaque octets -- rather than acquiring a
    resolution from some unrelated set in the module.
    """
    module = compile_module(
        """
      M DEFINITIONS ::= BEGIN
        C ::= CLASS { &code INTEGER, &Type } WITH SYNTAX {&code &Type}
        S C ::= { {1 INTEGER} }
        T ::= SEQUENCE { code C.&code, body C.&Type }
      END
    """,
        "<opaque>",
    ).module
    body = module.types["T"].components[1].type
    assert isinstance(body, OpenType) and body.table is None
    octets = _wrap(Primitive(Universal.INTEGER, "INTEGER"), 5)
    back = module.decode("T", module.encode("T", {"code": 1, "body": octets}))
    assert back["body"] == octets and "body.resolved" not in back


# --- X.683 parameterization --------------------------------------------------------------


def test_a_parameterized_type_instantiates_per_actual_parameter():
    """X.683 §9.7: the actual parameter takes the place of the dummy reference."""
    module = compile_module(
        """
      M DEFINITIONS ::= BEGIN
        Pair {X} ::= SEQUENCE { a X, b INTEGER }
        BoolPair ::= Pair {BOOLEAN}
        StrPair  ::= Pair {PrintableString}
      END
    """,
        "<param>",
    ).module
    assert module.types["BoolPair"].components[0].type.universal == Universal.BOOLEAN
    assert module.types["StrPair"].components[0].type.universal == Universal.PRINTABLE_STRING
    # Both instantiations share the un-parameterized component untouched.
    for name in ("BoolPair", "StrPair"):
        assert module.types[name].components[1].type.universal == Universal.INTEGER


def test_the_rfc5280_shape_resolves_through_a_parameterized_object_set():
    """The pattern real PKIX modules are written in, end to end.

    `AttributeTypeAndValue` is parameterized on the object set, and its table constraints
    name the DUMMY (`{Supported}`). Instantiating with a real set has to rewrite those
    constraints, or the open type has nothing to resolve against -- which is exactly the
    state the front-end was in before X.683: X.681/682 were built and could not fire on an
    unmodified module.
    """
    from bcir.asn1.codec import Oid

    module = compile_module(
        """
      Pkix DEFINITIONS ::= BEGIN
        ATTRIBUTE ::= CLASS { &id OBJECT IDENTIFIER UNIQUE, &Type }
          WITH SYNTAX {&Type IDENTIFIED BY &id}
        SupportedAttributes ATTRIBUTE ::= {
            {PrintableString IDENTIFIED BY {2 5 4 6}} |
            {UTF8String IDENTIFIED BY {2 5 4 3}} }
        AttributeTypeAndValue {ATTRIBUTE:Supported} ::= SEQUENCE {
            type  ATTRIBUTE.&id ({Supported}),
            value ATTRIBUTE.&Type ({Supported}{@type}) }
        Attr ::= AttributeTypeAndValue {SupportedAttributes}
      END
    """,
        "<rfc5280>",
    ).module

    open_type = module.types["Attr"].components[1].type
    assert open_type.table is not None and len(open_type.table.rows) == 2, (
        "the dummy object set must be rewritten to the actual before the table is built"
    )

    for arcs, kind, value in (
        ((2, 5, 4, 6), Primitive(Universal.PRINTABLE_STRING, "PrintableString"), "GB"),
        ((2, 5, 4, 3), Primitive(Universal.UTF8_STRING, "UTF8String"), "Example CA"),
    ):
        octets = _wrap(kind, value)
        back = module.decode("Attr", module.encode("Attr", {"type": Oid(arcs), "value": octets}))
        assert back["value.resolved"] == value, (arcs, back.get("value.resolved"))


def test_two_instantiations_of_one_parameterized_type_stay_independent():
    """Memoising instantiations must key on the ACTUALS, not just the name."""
    module = compile_module(
        """
      M DEFINITIONS ::= BEGIN
        C ::= CLASS { &id INTEGER UNIQUE, &Type } WITH SYNTAX {&Type IDENTIFIED BY &id}
        SetA C ::= { {INTEGER IDENTIFIED BY 1} }
        SetB C ::= { {BOOLEAN IDENTIFIED BY 1} | {PrintableString IDENTIFIED BY 2} }
        Holder {C:S} ::= SEQUENCE { id C.&id ({S}), body C.&Type ({S}{@id}) }
        UsesA ::= Holder {SetA}
        UsesB ::= Holder {SetB}
      END
    """,
        "<two>",
    ).module
    a = module.types["UsesA"].components[1].type
    b = module.types["UsesB"].components[1].type
    assert len(a.table.rows) == 1 and len(b.table.rows) == 2, (
        f"instantiations must not share a table: {len(a.table.rows)} vs {len(b.table.rows)}"
    )
    assert a.resolve({("id",): 1}).universal == Universal.INTEGER
    assert b.resolve({("id",): 1}).universal == Universal.BOOLEAN


def test_a_nested_parameterized_reference_instantiates():
    """A parameterized type may itself be an actual parameter (§9.5's Type alternative)."""
    module = compile_module(
        """
      M DEFINITIONS ::= BEGIN
        Box {X} ::= SEQUENCE { item X }
        Pair {Y} ::= SEQUENCE { left Y, right Y }
        T ::= Pair {Box {INTEGER}}
      END
    """,
        "<nested>",
    ).module
    left = module.types["T"].components[0].type
    assert left.components[0].type.universal == Universal.INTEGER
