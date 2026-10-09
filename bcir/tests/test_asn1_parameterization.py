"""X.683 parameterization is hygienic, and every name a constraint mentions is resolved or refused.

The 2026-10-06 audit (docs/research/BCIR_GEMPLUS_TMSAO_AUDIT_2026-10-06.md §4.1) found the
front end lowering a parameterized type with DYNAMIC scope: an object-set actual was installed in
the module's own table under the dummy's name for the duration of the instantiation, so a nested
template resolved its own references through the enclosing instance's bindings, and the name-keyed
caches kept what it found -- a module's meaning depended on the order of its assignments. Building
the fix found five more of the same family, each a name the front end dropped or mis-resolved:

  * a constraint bound written as a value reference (`INTEGER (0..ub)`) was DROPPED, which under
    PER and OER changed the octets (`c8` became `0200c8`);
  * a value actual (`Bounded {5}`) did not parse, so value parameters did not exist;
  * a braced actual was looked up as a name made of its own text;
  * a table constraint naming more than one object set, or a parameterized one, was silently
    untabled, leaving the open type with nothing to resolve against;
  * a contained subtype (`INTEGER (Small ^ 0..3)`) and a value-set assignment were dropped or
    refused.

Every test here was RED on `main` at 5cd8e03 and states the property, not the repair.
"""

from __future__ import annotations

import itertools

from bcir.asn1.codec import Oid
from bcir.asn1.oer import encode_oer
from bcir.asn1.per import encode_per
from bcir.frontends.asn1 import Asn1SemanticError, compile_module, parse_module, print_module

_CLASS = """
    ATTRIBUTE ::= CLASS { &id OBJECT IDENTIFIER UNIQUE, &Type }
      WITH SYNTAX {&Type IDENTIFIED BY &id}
"""


def _module(body: str, name: str = "M", tags: str = "") -> str:
    return f"{name} DEFINITIONS {tags + ' TAGS ' if tags else ''}::= BEGIN\n{body}\nEND\n"


def _rows(open_type) -> int:
    assert open_type.table is not None, "the open type carries no table: the constraint was dropped"
    return len(open_type.table.rows)


def _refused(text: str, *needles: str, imports=None) -> str:
    try:
        compile_module(text, "<t>", imports)
    except Asn1SemanticError as exc:
        message = str(exc)
        for needle in needles:
            assert needle in message, (needle, message)
        return message
    raise AssertionError("the module compiled; it should have been refused")


# --- hygiene ---------------------------------------------------------------------------


def test_a_template_resolves_its_names_where_the_template_is_written():
    """Lexical scope: a name a template's body leaves free means what it means at module level,
    however the template is reached. `Inner2` names an object set the module never defines, so
    it is refused used directly AND used inside a template whose dummy happens to share the
    name -- where the old lowering resolved it through that dummy's binding."""
    body = (
        _CLASS
        + """
    Other ATTRIBUTE ::= { {UTF8String IDENTIFIED BY {2 5 4 3}} | {BOOLEAN IDENTIFIED BY {2 5 4 99}} }
    Inner2 {ATTRIBUTE:P} ::= SEQUENCE {
        type  ATTRIBUTE.&id ({Ghost}),
        value ATTRIBUTE.&Type ({Ghost}{@type}) }
"""
    )
    _refused(_module(body + "    UseInner2 ::= Inner2 {Other}\n"), "Ghost", "does not define")
    _refused(
        _module(
            body
            + "    Outer2 {ATTRIBUTE:Ghost} ::= SEQUENCE { a Inner2 {Ghost} }\n"
            + "    UseOuter2 ::= Outer2 {Other}\n"
        ),
        "Ghost",
        "does not define",
    )


def test_a_dummy_never_rebinds_the_module_for_a_nested_template_in_any_order():
    """`Inner` names the MODULE's `Supported` (one row). Reached through `Outer`, whose dummy is
    also spelled `Supported` and bound to `Other` (two rows), it must still see one row -- and
    the answer must not depend on which assignment the module lowers first, which the old
    name-keyed memo and table cache made it do."""
    head = (
        _CLASS
        + """
    Supported ATTRIBUTE ::= { {PrintableString IDENTIFIED BY {2 5 4 6}} }
    Other ATTRIBUTE ::= { {UTF8String IDENTIFIED BY {2 5 4 3}} | {BOOLEAN IDENTIFIED BY {2 5 4 99}} }
    Inner {ATTRIBUTE:P} ::= SEQUENCE {
        type  ATTRIBUTE.&id ({Supported}),
        value ATTRIBUTE.&Type ({Supported}{@type}) }
    Outer {ATTRIBUTE:Supported} ::= SEQUENCE { a Inner {Supported} }
"""
    )
    uses = (
        "    UseOuter ::= Outer {Other}\n",
        "    UseInner ::= Inner {Other}\n",
        "    UseAgain ::= Inner {Supported}\n",
    )
    for order in itertools.permutations(uses):
        module = compile_module(_module(head + "".join(order)), "<t>").module
        assert _rows(module.types["UseOuter"].components[0].type.components[1].type) == 1, order
        assert _rows(module.types["UseInner"].components[1].type) == 1, order
        assert _rows(module.types["UseAgain"].components[1].type) == 1, order


def test_an_instance_is_memoised_by_what_it_means():
    """One spelling can mean two things and two spellings one: `Pair {T}` written inside `Wrap`
    is a different instance in every instance of `Wrap`, while `Bounded {5}` and
    `Bounded {five}` are the same instance. A memo keyed by the spelling gets both wrong."""
    module = compile_module(
        _module(
            """
    Pair {T} ::= SEQUENCE { a T, b T }
    P1 ::= Pair {INTEGER}
    P2 ::= Pair {INTEGER}
    P3 ::= Pair {BOOLEAN}
    Wrap {T} ::= SEQUENCE { p Pair {T} }
    W1 ::= Wrap {INTEGER}
    W2 ::= Wrap {BOOLEAN}
    Bounded {INTEGER:ub} ::= INTEGER (0..ub)
    B1 ::= Bounded {5}
    B2 ::= Bounded {five}
    B3 ::= Bounded {6}
    five INTEGER ::= 5
"""
        ),
        "<t>",
    ).module
    types = module.types
    assert types["P1"] is types["P2"], "equal actuals denote one instance"
    assert types["P1"] is not types["P3"]
    assert types["P3"].components[0].type.name == "BOOLEAN"
    inner = [types[w].components[0].type for w in ("W1", "W2")]
    assert inner[0] is not inner[1], "one spelling, two meanings: two instances"
    assert [p.components[0].type.name for p in inner] == ["INTEGER", "BOOLEAN"]
    assert inner[0] is types["P1"], "the nested instance is the one its meaning names"
    assert types["B1"] is types["B2"], "two spellings of one value: one instance"
    assert types["B1"] is not types["B3"]
    assert str(types["B3"].constraint) == "(0..6)"


# --- named values and types in constraints ---------------------------------------------------


def _same_octets(a, b, values):
    for value in values:
        assert encode_per(a, value) == encode_per(b, value), value
        assert encode_oer(a, value) == encode_oer(b, value), value


def test_a_value_reference_bounds_a_constraint_like_the_literal_it_names():
    """X.680 §51: an endpoint may be a DefinedValue. It used to drop the constraint, and under
    PER and OER `INTEGER (0..ub)` with ub = 255 encoded 200 as `0200c8`, not `c8`. Forward
    references (the value assigned after its use) resolve too."""
    literal = compile_module(
        _module("  T ::= INTEGER (0..255)\n  S ::= IA5String (SIZE (1..64))\n"), "<t>"
    ).module
    named = compile_module(
        _module(
            "  T ::= INTEGER (0..ub)\n  S ::= IA5String (SIZE (1..ub-name))\n"
            "  ub INTEGER ::= 255\n  ub-name INTEGER ::= 64\n"
        ),
        "<t>",
    ).module
    assert named.types["T"].constraint == literal.types["T"].constraint
    assert named.types["S"].constraint == literal.types["S"].constraint
    assert encode_per(named.types["T"], 200) == bytes.fromhex("c8")
    _same_octets(literal.types["T"], named.types["T"], (0, 1, 200, 255))
    _same_octets(literal.types["S"], named.types["S"], ("a", "abc", "x" * 64))


def test_a_value_reference_resolves_through_an_imported_module():
    bounds = compile_module(_module("  ub-name INTEGER ::= 64\n", "Bounds"), "<bounds>")
    user = compile_module(
        _module("  S ::= IA5String (SIZE (1..ub-name))\n", "User"), "<user>", {"Bounds": bounds}
    ).module
    assert str(user.types["S"].constraint) == "(SIZE (1..64))"


def test_a_value_is_read_against_the_type_it_was_assigned():
    """X.680 §16.2: `x C ::= red` is the value `red` of C wherever `x` is named, so a bound
    naming `x` reads `red` against C, not against the type being constrained. The parser used
    to read every `lower-name Type ::= ...` as an information object, so a value whose type is
    a defined one was no value at all -- harmless while the bound was dropped, and a refusal
    once it is resolved."""
    module = compile_module(
        _module(
            "  Count ::= INTEGER\n  v Count ::= 255\n  C ::= INTEGER {red(1), green(2)}\n"
            "  x C ::= red\n  E ::= ENUMERATED {a, b, c}\n  e1 E ::= b\n"
            "  T ::= INTEGER (0..v)\n  U ::= INTEGER (0..x)\n  F ::= E (e1)\n"
        ),
        "<t>",
    ).module
    literal = compile_module(_module("  T ::= INTEGER (0..255)\n"), "<t>").module
    assert str(module.types["T"].constraint) == "(0..255)"
    _same_octets(literal.types["T"], module.types["T"], (0, 200, 255))
    assert str(module.types["U"].constraint) == "(0..1)"
    assert str(module.types["F"].constraint) == "(1)"
    parsed = parse_module(
        _module(_CLASS + "  v Count ::= 7\n  attr ATTRIBUTE ::= {INTEGER IDENTIFIED BY {1 2}}\n")
    )
    kinds = {a.name: type(a).__name__ for a in parsed.assignments}
    assert kinds["v"] == "ValueAssignment" and kinds["attr"] == "ObjectAssignment", kinds
    _refused(_module("  T ::= INTEGER (0..a)\n  a INTEGER ::= b\n  b INTEGER ::= a\n"), "itself")


def test_a_name_a_constraint_cannot_resolve_is_refused_not_dropped():
    _refused(_module("  T ::= INTEGER (0..nowhere)\n"), "nowhere", "neither")
    _refused(_module("  T ::= INTEGER (0..ub)\n  ub BOOLEAN ::= TRUE\n"), "ub", "not an INTEGER")
    _refused(
        _module("  S ::= OCTET STRING (SIZE (0..neg))\n  neg INTEGER ::= -1\n"),
        "neg",
        "non-negative",
    )
    _refused(_module("  T ::= INTEGER (Missing)\n"), "Missing", "never")


def test_a_contained_subtype_and_a_value_set_constrain_like_their_definitions():
    """§51.3 a ContainedSubtype, and §15.6 a ValueSetTypeAssignment: both were dropped."""
    module = compile_module(
        _module(
            """
    Small ::= INTEGER (0..7)
    T ::= INTEGER (Small ^ 0..3)
    U ::= INTEGER (INCLUDES Small)
    Valid INTEGER ::= {1 | 2 | 3}
    V ::= Valid
"""
        ),
        "<t>",
    ).module
    literal = compile_module(
        _module("  T ::= INTEGER (0..3)\n  U ::= INTEGER (0..7)\n  V ::= INTEGER (1..3)\n"), "<t>"
    ).module
    for name in ("T", "U", "V"):
        assert module.types[name].constraint is not None, name
        _same_octets(literal.types[name], module.types[name], (1, 2, 3))
    assert (
        print_module(parse_module(_module("  Valid INTEGER ::= {1 | 2 | 3}\n"))).count("Valid ::=")
        == 1
    )


# --- parameters of every kind -----------------------------------------------------------


def test_value_parameters_bind_at_the_point_of_application():
    """§9.5: an actual may be a value, and the bound it carries is the application point's."""
    module = compile_module(
        _module(
            """
    Bounded {INTEGER:ub} ::= SEQUENCE (SIZE (1..ub)) OF INTEGER
    Ranged {INTEGER:hi} ::= INTEGER (0..hi)
    Opt {INTEGER:d} ::= SEQUENCE { x INTEGER DEFAULT d }
    OneOf {INTEGER:Set} ::= INTEGER (Set)
    B5 ::= Bounded {5}
    B9 ::= Bounded {9}
    R ::= Ranged {top}
    O ::= Opt {7}
    X ::= OneOf {{1 | 2 | 3}}
    top INTEGER ::= 255
"""
        ),
        "<t>",
    ).module
    literal = compile_module(
        _module(
            "  B5 ::= SEQUENCE (SIZE (1..5)) OF INTEGER\n  B9 ::= SEQUENCE (SIZE (1..9)) OF INTEGER\n"
            "  R ::= INTEGER (0..255)\n  X ::= INTEGER (1..3)\n"
        ),
        "<t>",
    ).module
    assert module.types["B5"].constraint == literal.types["B5"].constraint
    assert module.types["B9"].constraint == literal.types["B9"].constraint
    _same_octets(literal.types["B5"], module.types["B5"], ([1], [1, 2, 3, 4, 5]))
    _same_octets(literal.types["R"], module.types["R"], (0, 200, 255))
    _same_octets(literal.types["X"], module.types["X"], (1, 3))
    assert module.types["O"].components[0].default == 7


def test_braced_and_parameterized_object_sets_build_the_tables_their_names_would():
    """An object set written in place -- a braced actual, a union in a table constraint, a
    parameterized object set, an inline object -- builds the same associated table as the
    named set it spells. Each used to be refused or silently untabled."""
    body = (
        _CLASS
        + """
    cn ATTRIBUTE ::= {UTF8String IDENTIFIED BY {2 5 4 3}}
    c ATTRIBUTE ::= {PrintableString IDENTIFIED BY {2 5 4 6}}
    Named ATTRIBUTE ::= { cn | c }
    A ATTRIBUTE ::= { cn }
    B ATTRIBUTE ::= { c }
    Pick {ATTRIBUTE:o} ATTRIBUTE ::= { o }
    AV {ATTRIBUTE:S} ::= SEQUENCE { type ATTRIBUTE.&id ({S}), value ATTRIBUTE.&Type ({S}{@type}) }
    ByName ::= AV {Named}
    Braced ::= AV {{ cn | c }}
    Union ::= SEQUENCE { type ATTRIBUTE.&id ({A | B}), value ATTRIBUTE.&Type ({A | B}{@type}) }
    Picked ::= SEQUENCE { type ATTRIBUTE.&id ({Pick {cn} | B}), value ATTRIBUTE.&Type ({Pick {cn} | B}{@type}) }
    Inline ::= SEQUENCE { type ATTRIBUTE.&id ({ {UTF8String IDENTIFIED BY {2 5 4 3}} | c }) }
"""
    )
    module = compile_module(_module(body), "<t>").module
    expect = _rows(module.types["ByName"].components[1].type)
    assert expect == 2
    for name in ("Braced", "Union", "Picked"):
        assert _rows(module.types[name].components[1].type) == expect, name
    assert tuple(module.types["Inline"].components[0].type.table_values) == (
        Oid((2, 5, 4, 3)),
        Oid((2, 5, 4, 6)),
    )


def test_a_braced_actual_naming_a_dummy_is_substituted_token_by_token():
    module = compile_module(
        _module(
            _CLASS
            + """
    cn ATTRIBUTE ::= {UTF8String IDENTIFIED BY {2 5 4 3}}
    c ATTRIBUTE ::= {PrintableString IDENTIFIED BY {2 5 4 6}}
    AV {ATTRIBUTE:S} ::= SEQUENCE { type ATTRIBUTE.&id ({S}), value ATTRIBUTE.&Type ({S}{@type}) }
    Outer {ATTRIBUTE:o} ::= AV {{ o | c }}
    Two ::= Outer {cn}
"""
        ),
        "<t>",
    ).module
    assert _rows(module.types["Two"].components[1].type) == 2


def test_an_actual_must_fit_the_dummy_it_binds():
    """§9.6: the actual's form must fit the dummy's kind; a mismatch is named, not guessed."""
    _refused(
        _module(
            "  Bounded {INTEGER:ub} ::= SEQUENCE (SIZE (1..ub)) OF INTEGER\n  B ::= Bounded {BOOLEAN}\n"
        ),
        "must be a value",
    )
    _refused(_module("  Pair {T} ::= SEQUENCE { a T }\n  P ::= Pair {5}\n"), "must be a type")
    _refused(
        _module(
            _CLASS
            + "  AV {ATTRIBUTE:S} ::= SEQUENCE { t ATTRIBUTE.&id ({S}) }\n  X ::= AV {Nope}\n"
        ),
        "must be an object set",
    )


def test_a_recursive_template_terminates_and_an_infinite_family_is_refused():
    module = compile_module(
        _module(
            "  Tree {T} ::= SEQUENCE { v T, kids SEQUENCE OF Tree {T} }\n  IntTree ::= Tree {INTEGER}\n"
        ),
        "<t>",
    ).module
    tree = {"v": 1, "kids": [{"v": 2, "kids": []}]}
    assert module.decode("IntTree", module.encode("IntTree", tree)) == tree
    _refused(
        _module(
            "  Grow {T} ::= SEQUENCE { a T, b Grow {SEQUENCE OF T} OPTIONAL }\n  G ::= Grow {INTEGER}\n"
        ),
        "infinite family",
    )


# --- X.683 §9.8: a template of another module ---------------------------------------------


def test_an_imported_template_lowers_in_its_module_and_its_actual_in_this_one():
    """§9.8's NOTE: the actual parameter's tagging environment applies, not the dummy's. `Wrap`
    lives in an EXPLICIT TAGS module; the actual `[1] INTEGER` is written in an IMPLICIT TAGS
    one, so its tag is IMPLICIT -- lowered in the template's environment it would have been
    EXPLICIT, `30 05 a1 03 02 01 05` instead of `30 03 81 01 05`. The template's own rules stay
    its module's: `Auto` lives in an AUTOMATIC TAGS module, so an untagged actual is tagged
    automatically there, and a tagged one switches automatic tagging off for that SEQUENCE
    (X.680 §25.3, decided after substitution)."""
    lib = compile_module(_module("  Wrap {T} ::= SEQUENCE { a T }\n", "Lib", "EXPLICIT"), "<lib>")
    auto_lib = compile_module(
        _module("  Auto {T} ::= SEQUENCE { a T, b BOOLEAN }\n", "AutoLib", "AUTOMATIC"), "<al>"
    )
    user = compile_module(
        _module(
            "  W ::= Wrap {[1] INTEGER}\n  Plain ::= Auto {INTEGER}\n  Tagged ::= Auto {[5] INTEGER}\n",
            "User",
            "IMPLICIT",
        ),
        "<user>",
        {"Lib": lib, "AutoLib": auto_lib},
    ).module
    assert user.encode("W", {"a": 5}).hex() == "3003810105"
    value = {"a": 5, "b": True}
    automatic = compile_module(
        _module("  P ::= SEQUENCE { a INTEGER, b BOOLEAN }\n", "R", "AUTOMATIC"), "<r>"
    ).module
    assert user.encode("Plain", value) == automatic.encode("P", value)
    switched_off = compile_module(
        _module("  T ::= SEQUENCE { a [5] IMPLICIT INTEGER, b BOOLEAN }\n", "R2", "EXPLICIT"),
        "<r2>",
    ).module
    assert user.encode("Tagged", value) == switched_off.encode("T", value)
    assert user.encode("Tagged", value) != automatic.encode("P", value)


def test_an_imported_template_takes_object_sets_and_values_from_the_instantiating_module():
    lib = compile_module(
        _module(
            _CLASS
            + "  AV {ATTRIBUTE:S} ::= SEQUENCE { type ATTRIBUTE.&id ({S}), value ATTRIBUTE.&Type ({S}{@type}) }\n"
            + "  Bounded {INTEGER:ub} ::= SEQUENCE (SIZE (1..ub)) OF INTEGER\n",
            "Lib",
        ),
        "<lib>",
    )
    user = compile_module(
        _module(
            "  Mine ATTRIBUTE ::= { {&Type UTF8String, &id {2 5 4 3}} | {&Type BOOLEAN, &id {2 5 4 99}} }\n"
            "  X ::= AV {Mine}\n  B ::= Bounded {limit}\n  limit INTEGER ::= 3\n",
            "User",
        ),
        "<user>",
        {"Lib": lib},
    ).module
    assert _rows(user.types["X"].components[1].type) == 2
    assert str(user.types["B"].constraint) == "(SIZE (1..3))"


# --- the round-trip law -------------------------------------------------------------------


def test_the_round_trip_law_holds_for_parameterized_modules():
    """parse(print(parse(t))) == parse(t), over every construct this file adds: parameterized
    assignments of each kind with governors, value / braced / parameterized actuals, table
    constraints with unions and at-notations, value references and contained subtypes in
    constraints, value sets, and a class WITH SYNTAX (which the printer used to drop)."""
    text = _module(
        _CLASS
        + """
    cn ATTRIBUTE ::= {UTF8String IDENTIFIED BY {2 5 4 3}}
    A ATTRIBUTE ::= { cn }
    Pick {ATTRIBUTE:o} ATTRIBUTE ::= { o }
    AV {ATTRIBUTE:S} ::= SEQUENCE { type ATTRIBUTE.&id ({S}), value ATTRIBUTE.&Type ({S}{@type}) }
    Bounded {INTEGER:ub} ::= SEQUENCE (SIZE (1..ub)) OF INTEGER
    Small ::= INTEGER (0..lim)
    T ::= INTEGER (Small ^ 0..3)
    Valid INTEGER ::= {1 | 2 | 3}
    U ::= SEQUENCE { type ATTRIBUTE.&id ({A | Pick {cn}}), value ATTRIBUTE.&Type ({A | Pick {cn}}{@type}) }
    X ::= AV {{ cn }}
    B ::= Bounded {4}
    lim INTEGER ::= 7
"""
    )
    node = parse_module(text, "<t>")
    again = parse_module(print_module(node), "<printed>")
    assert again == node
