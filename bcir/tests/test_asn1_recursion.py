"""Recursive types on every encoding rule, held to one bound, from one containment graph.

The 2026-10-06 audit (§4.2) found a recursive type round-tripping on BER/DER only: OER, XER and
JER had no case for the front end's forward reference, the JER plan refused it as an open type,
and the write plan as a SET. Building the fix found three more:

  * a recursive CHOICE (`Expr ::= CHOICE { lit INTEGER, neg Expr }`) did not lower at all -- the
    front end took "the referenced type is not built yet" for "tag it IMPLICITly", and an
    implicit tag over a CHOICE needs the tag a CHOICE does not have;
  * nothing bounded a recursion: a 1,000-deep value raised `RecursionError` from DER and PER
    encoders, and a crafted input of a few kilobytes did the same to every decoder but BER's;
  * `decode_jer` let a deeply nested JSON text raise `RecursionError` from the parser.

Reviewing the fix found four more, in the front end: a recursive X.683 instance whose template
is a CHOICE did not lower (the tag over it was decided as for a type that is not a CHOICE); a
type defined only as a reference to itself (`A ::= B`, `B ::= A`) lowered with no type behind
it; a recursive reference to a type with an assignment-level tag lost the tag; and a contained
subtype naming an alias of a recursive type was refused as "a contained subtype of itself".

Every test here was RED on the parent and states the property, not the repair.
"""

from __future__ import annotations

from unittest import mock

from bcir.asn1 import encode_plan, jer, jer_plan, oer, per, xer
from bcir.asn1.schema import MAX_RECURSION, SequenceOf, Primitive
from bcir.asn1.tags import Asn1Error, Universal
from bcir.frontends.asn1 import Asn1SemanticError, compile_module
from bcir.frontends.asn1.lower import Lowerer, containment_graph

_SOURCE = """M DEFINITIONS AUTOMATIC TAGS ::= BEGIN
  Node ::= SEQUENCE { label INTEGER (0..255), children SEQUENCE (SIZE (0..4)) OF Node }
  List ::= SEQUENCE { value INTEGER (0..1), next List OPTIONAL }
  Expr ::= CHOICE { lit INTEGER, neg Expr, sum SEQUENCE { a Expr, b Expr } }
  A ::= SEQUENCE { b B OPTIONAL }
  B ::= SEQUENCE { a A OPTIONAL, n BOOLEAN }
  Leaf ::= SEQUENCE { x INTEGER, n Node }
  Small ::= INTEGER (0..9)
  Uses ::= INTEGER (Small)
  Tree {T} ::= SEQUENCE { v T, kids SEQUENCE OF Tree {T} }
  IntTree ::= Tree {INTEGER}
END
"""

_VALUES = {
    "Node": {
        "label": 1,
        "children": [
            {"label": 2, "children": []},
            {"label": 3, "children": [{"label": 4, "children": []}]},
        ],
    },
    "List": {"value": 1, "next": {"value": 0, "next": {"value": 1}}},
    "Expr": ("sum", {"a": ("lit", 1), "b": ("neg", ("neg", ("lit", -2)))}),
    "A": {"b": {"a": {"b": {"n": True}}, "n": False}},
    "IntTree": {"v": 1, "kids": [{"v": 2, "kids": []}, {"v": 3, "kids": [{"v": 4, "kids": []}]}]},
}


def _lowered():
    return compile_module(_SOURCE, "<recursion>")


def _rails(module):
    """(name, encode(kind, value), decode(kind, data)) for every encoding rule."""
    return (
        ("DER", lambda k, v: module.encode(k.name, v), lambda k, d: module.decode(k.name, d)),
        ("PER", per.encode_per, lambda k, d: per.decode_per(d, k)),
        (
            "APER",
            lambda k, v: per.encode_per(k, v, variant=per.PerVariant.ALIGNED),
            lambda k, d: per.decode_per(d, k, variant=per.PerVariant.ALIGNED),
        ),
        ("OER", oer.encode_oer, oer.decode_oer),
        ("XER", xer.encode_xer, lambda k, d: xer.decode_xer(d, k)),
        ("JER", jer.encode_jer, lambda k, d: jer.decode_jer(d, k)),
    )


def test_every_rail_round_trips_recursive_types():
    module = _lowered().module
    for name, value in _VALUES.items():
        kind = module.types[name]
        for rail, encode, decode in _rails(module):
            if rail == "DER" and name == "IntTree":
                continue  # an instance type: the module's own name for it is IntTree, below
            data = encode(kind, value)
            assert decode(kind, data) == value, (rail, name)


def test_a_recursive_choice_lowers_with_its_reference_tagged_explicitly():
    """X.680 §31.2.7: a tag over a CHOICE is EXPLICIT -- for a CHOICE that is still being built
    too. Taking "not built yet" for "not a CHOICE" made `neg [1] Expr` IMPLICIT, and the module
    did not lower at all."""
    expr = _lowered().module.types["Expr"]
    neg = next(alt for alt in expr.alternatives if alt.name == "neg")
    assert neg.explicit is True
    assert _lowered().module.encode("Expr", ("neg", ("lit", 5))).hex() == "a103800105"


def test_an_untagged_choice_containing_itself_is_refused_by_name():
    """§29.3: the alternatives' tags must be distinct, and an untagged alternative that IS the
    choice has no defined set of tags at all -- a refusal that says so, not a stack overflow
    and not "unresolved forward reference"."""
    try:
        compile_module(
            "M DEFINITIONS EXPLICIT TAGS ::= BEGIN\n  E ::= CHOICE { lit INTEGER, e E }\nEND\n",
            "<t>",
        )
    except (Asn1SemanticError, Asn1Error) as exc:
        assert "contains itself" in str(exc), exc
    else:  # pragma: no cover - the defect
        raise AssertionError("an untagged CHOICE that contains itself was accepted")


def _list(levels: int) -> dict:
    value = {"value": 0}
    for _ in range(levels - 1):
        value = {"value": 1, "next": value}
    return value


def test_recursion_is_bounded_on_every_encoder():
    """One bound for every rail: a value nesting past MAX_RECURSION references is refused with
    a verdict naming recursion, never a `RecursionError`, and the refusal leaves nothing
    behind -- a shallow value encodes after it."""
    module = _lowered().module
    kind = module.types["List"]
    for rail, encode, decode in _rails(module):
        shallow = _list(MAX_RECURSION // 2)
        assert decode(kind, encode(kind, shallow)) == shallow, rail
        try:
            encode(kind, _list(4 * MAX_RECURSION))
        except Asn1Error as exc:
            assert "recurses deeper" in str(exc), (rail, exc)
        else:  # pragma: no cover - the defect
            raise AssertionError(f"{rail}: a value {4 * MAX_RECURSION} deep encoded")
        assert decode(kind, encode(kind, shallow)) == shallow, rail


def _bits(levels: int) -> bytes:
    """Unaligned PER of `_list(levels)` with `value` 1 throughout but the last: per level one
    presence bit for `next` and one bit for `value (0..1)`."""
    bits = "11" * (levels - 1) + "00"
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def test_a_crafted_deep_input_is_refused_by_every_decoder():
    """The decoders follow the input's nesting, so the input chooses the depth: a few kilobytes
    used to raise `RecursionError` from PER, OER, XER and JER. Each is a verdict now."""
    kind = _lowered().module.types["List"]
    levels = 5000
    deep = {
        "PER": lambda: per.decode_per(_bits(levels), kind),
        "OER": lambda: oer.decode_oer(kind, b"\x80\x01" * (levels - 1) + b"\x00\x00"),
        "XER": lambda: xer.decode_xer(
            "<SEQUENCE>"
            + "<value>1</value><next>" * (levels - 1)
            + "<value>0</value>"
            + "</next>" * (levels - 1)
            + "</SEQUENCE>",
            kind,
        ),
        "JER": lambda: jer.decode_jer('{"value":1,"next":' * 200 + '{"value":0}' + "}" * 200, kind),
    }
    for rail, decode in deep.items():
        try:
            decode()
        except Asn1Error as exc:
            assert "recurses deeper" in str(exc), (rail, exc)
        else:  # pragma: no cover - the defect
            raise AssertionError(f"{rail}: a {levels}-deep input decoded")
    assert per.decode_per(_bits(MAX_RECURSION // 2), kind) == _list(MAX_RECURSION // 2)


def test_a_json_text_nested_past_the_parser_is_a_verdict():
    """Not a recursive schema at all: `json.loads` recursed on the text's own nesting, so
    200,000 brackets reached `decode_jer` as a `RecursionError`."""
    flat = SequenceOf(Primitive(Universal.INTEGER, "INTEGER"))
    try:
        jer.decode_jer("[" * 200_000 + "]" * 200_000, flat)
    except Asn1Error as exc:
        assert "nests deeper" in str(exc), exc
    else:  # pragma: no cover - the defect
        raise AssertionError("a JSON text 200,000 deep decoded")


def test_a_recursive_type_compiles_to_a_jer_plan_with_a_back_edge():
    """The plan stays finite: the reference compiles to a `ref` node naming the ancestor's
    path, the plan says it needs a reader that knows one (version 2), and a decode through
    the plan traces the value's whole depth. A plan with no recursion is version 1 and has no
    `ref` -- the bytes of every existing descriptor are unchanged."""
    module = _lowered().module
    plan = jer_plan.compile_plan(module.types["Node"], module="M", type_name="Node")
    text = plan.serialize().decode()
    assert plan.plan_version == jer_plan.PLAN_VERSION_RECURSIVE
    assert "node ./children[] kind=ref" in text and text.rstrip().endswith("ref=.")
    assert " depth=? " in text
    value, trace = jer_plan.decode_with_plan(
        plan, jer.encode_jer(module.types["Node"], _VALUES["Node"])
    )
    assert value == _VALUES["Node"]
    # One `sequence` per Node value, and the walk reaches the deepest one through the back-edge.
    assert sum(e.startswith("enter ") and e.endswith(" sequence") for e in trace) == 4
    assert "enter ./children[]/children[]/label integer" in trace
    mutual = jer_plan.compile_plan(module.types["B"], module="M", type_name="B")
    assert "ref=./a" in mutual.serialize().decode()
    flat = jer_plan.compile_plan(module.types["Leaf"].components[0].type, module="M", type_name="X")
    assert flat.plan_version == jer_plan.PLAN_VERSION and "ref=" not in flat.serialize().decode()


def test_the_write_plan_refuses_a_recursive_type_by_naming_its_cycle():
    module = _lowered().module
    for name, cycle in (("Node", "Node -> Node"), ("A", "A -> B -> A"), ("Expr", "Expr -> Expr")):
        try:
            encode_plan.compile_encode_plan(module.types[name], module="M", type_name=name)
        except Asn1Error as exc:
            assert f"is recursive ({cycle})" in str(exc), (name, exc)
        else:  # pragma: no cover - the defect
            raise AssertionError(f"{name}: the write plan compiled a recursive type")


def test_the_containment_graph_names_exactly_the_recursive_types():
    """Pin the sets, not membership: `Leaf` contains a recursive type without being one, a
    name in a constraint is not containment, and an instance's template is."""
    lowered = _lowered()
    assert lowered.recursive == {
        "A": ("A", "B"),
        "B": ("A", "B"),
        "Expr": ("Expr",),
        "List": ("List",),
        "Node": ("Node",),
        "Tree": ("Tree",),
    }
    graph = containment_graph(lowered.node)
    assert graph["Leaf"] == ("Node",) and graph["Uses"] == () and graph["IntTree"] == ("Tree",)


def test_recursion_round_trips_in_every_tagging_environment():
    """The shapes the AUTOMATIC module above does not reach: an untagged recursive component
    under EXPLICIT and IMPLICIT TAGS, a recursive SET, an EXPLICIT and an IMPLICIT tag over the
    reference, and a CHOICE recursive directly and through a SEQUENCE. An IMPLICIT tag over a
    recursive CHOICE is refused, as it is over any CHOICE (X.680 31.2.7)."""
    cases = (
        (
            "EXPLICIT",
            "List ::= SEQUENCE { value INTEGER, next List OPTIONAL }",
            "List",
            {"value": 1, "next": {"value": 2, "next": {"value": 3}}},
        ),
        (
            "IMPLICIT",
            "List ::= SEQUENCE { value INTEGER, next List OPTIONAL }",
            "List",
            {"value": 1, "next": {"value": 2}},
        ),
        (
            "EXPLICIT",
            "S ::= SET { a INTEGER, s S OPTIONAL }",
            "S",
            {"a": 1, "s": {"a": 2, "s": {"a": 3}}},
        ),
        (
            "EXPLICIT",
            "T ::= SEQUENCE { a INTEGER, b [0] T OPTIONAL, c [1] IMPLICIT T OPTIONAL }",
            "T",
            {"a": 1, "b": {"a": 2}, "c": {"a": 3, "b": {"a": 4}}},
        ),
        ("EXPLICIT", "C ::= CHOICE { a INTEGER, b [0] C }", "C", ("b", ("b", ("a", 7)))),
        (
            "EXPLICIT",
            "C ::= CHOICE { a INTEGER, b SEQUENCE { c C } }",
            "C",
            ("b", {"c": ("b", {"c": ("a", 5)})}),
        ),
    )
    for tags, body, name, value in cases:
        module = compile_module(
            f"M DEFINITIONS {tags} TAGS ::= BEGIN\n  {body}\nEND\n", "<t>"
        ).module
        kind = module.types[name]
        for rail, encode, decode in _rails(module):
            assert decode(kind, encode(kind, value)) == value, (tags, body, rail)
    try:
        compile_module(
            "M DEFINITIONS IMPLICIT TAGS ::= BEGIN\n  C ::= CHOICE { a INTEGER, b [0] IMPLICIT C }\nEND\n",
            "<t>",
        )
    except Asn1SemanticError as exc:
        assert "IMPLICIT cannot tag a CHOICE" in str(exc), exc
    else:  # pragma: no cover - the defect
        raise AssertionError("an IMPLICIT tag over a recursive CHOICE was accepted")


def _module(body: str, name: str = "M", tags: str = "AUTOMATIC") -> str:
    return f"{name} DEFINITIONS {tags} TAGS ::= BEGIN\n{body}\nEND\n"


def _refused(text: str, needle: str, imports=None) -> None:
    try:
        compile_module(text, "<t>", imports)
    except Asn1SemanticError as exc:
        assert needle in str(exc), (needle, str(exc))
    else:  # pragma: no cover - the defect
        raise AssertionError(f"the module compiled; it should have been refused ({needle})")


def test_a_recursive_instance_of_a_choice_template_lowers_on_every_rail():
    """X.683 instances recurse as assigned types do. In `E {T} ::= CHOICE { lit T, neg E {T} }`
    the inner `E {T}` is the instance being built, a CHOICE, so the tag over it is EXPLICIT
    (X.680 31.2.7) -- decided from the instance's substituted body, since the instance does not
    exist yet. The same holds for a template of another module (X.683 9.8), and for an imported
    template whose tag over a dummy (IMPLICIT by its module's default) stands over this
    module's recursive CHOICE: the tag is EXPLICIT because of what the actual is."""
    value = ("neg", ("neg", ("lit", 5)))
    local = compile_module(
        _module("  E {T} ::= CHOICE { lit T, neg E {T} }\n  IntE ::= E {INTEGER}"), "<t>"
    ).module
    assert local.encode("IntE", ("neg", ("lit", 5))).hex() == "a103800105"
    lib = compile_module(_module("  E {T} ::= CHOICE { lit T, neg E {T} }", "Lib"), "<lib>")
    imported = compile_module(
        _module("  IntE ::= E {INTEGER}", "User", "EXPLICIT"), "<u>", {"Lib": lib}
    ).module
    wrap = compile_module(
        _module("  Wrap {T} ::= SEQUENCE { a [0] T }", "WrapLib", "IMPLICIT"), "<w>"
    )
    wrapped = compile_module(
        _module("  E ::= CHOICE { lit INTEGER, w Wrap {E} }", "User", "EXPLICIT"),
        "<u>",
        {"WrapLib": wrap},
    ).module
    assert wrapped.encode("E", ("w", {"a": ("lit", 7)})).hex() == "3005a003020107"
    cases = (
        (local, "IntE", value),
        (imported, "IntE", value),
        (wrapped, "E", ("w", {"a": ("w", {"a": ("lit", 7)})})),
    )
    for module, name, val in cases:
        kind = module.types[name]
        assert module.decode(name, module.encode(name, val)) == val, (name, "DER")
        for rail, encode, decode in _rails(module)[1:]:  # DER by the module's name, above
            assert decode(kind, encode(kind, val)) == val, (name, rail)


def test_a_type_defined_only_as_a_reference_to_itself_is_refused_by_name():
    """A recursion that passes through no SEQUENCE, SET, CHOICE or OF names no type: there is
    no value of `A ::= B` with `B ::= A` to encode. It lowered, and failed only when a value was
    encoded, as "defined as itself"; a module that defines no type is refused when it is
    lowered, naming the cycle."""
    for body, cycle in (
        ("  A ::= B\n  B ::= A", "(A -> B -> A)"),
        ("  A ::= A", "(A -> A)"),
        ("  A ::= [0] A", "(A -> A)"),
        ("  A ::= B (SIZE (1..2))\n  B ::= A", "(A -> B -> A)"),
        ("  W {T} ::= W {T}\n  A ::= W {INTEGER}", "(W{...} -> W{...})"),
    ):
        _refused(_module(body), f"is defined only as a reference to itself {cycle}")


def test_a_tag_decided_before_its_type_exists_is_held_to_the_type_built():
    """The tag over a reference to a type still being built is decided from the syntax the
    type is being built from. Once the module is complete the decision is checked against the
    type built, so a misreading is refused rather than encoded -- proven by injecting one: a
    reader that takes every definition for "not a CHOICE"."""
    text = _module("  Expr ::= CHOICE { lit INTEGER, neg Expr }")
    with mock.patch.object(Lowerer, "_reads_as_choice_or_open", lambda *_: False):
        _refused(text, "a tag over Expr was decided IMPLICIT before Expr was built")
    assert compile_module(text, "<t>").module.encode("Expr", ("neg", ("lit", 5))).hex() == (
        "a103800105"
    )


def test_a_recursive_reference_carries_its_types_assigned_tag():
    """X.680 31: an assignment's own tag is part of the type, so every component naming the
    type carries it -- the one inside the type's own definition too. That one was looked up
    before the tag was recorded, so `next T` went out under SEQUENCE's universal tag."""
    module = compile_module(
        _module(
            "  T ::= [APPLICATION 5] IMPLICIT SEQUENCE { v INTEGER, next T OPTIONAL }\n"
            "  H ::= SEQUENCE { t T }\n"
            "  U ::= [APPLICATION 6] EXPLICIT CHOICE { a INTEGER, u U }\n"
            "  HU ::= SEQUENCE { u U }",
            tags="IMPLICIT",
        ),
        "<t>",
    ).module
    value = {"t": {"v": 1, "next": {"v": 2}}}
    assert module.encode("H", value).hex() == "300a65080201016503020102"
    assert module.decode("H", module.encode("H", value)) == value
    choice = {"u": ("u", ("a", 3))}
    assert module.encode("HU", choice).hex() == "300766056603020103"
    assert module.decode("HU", module.encode("HU", choice)) == choice


def test_a_contained_subtype_naming_an_alias_of_a_recursive_type_is_that_type():
    """`B ::= A` inside the recursive `A` is a reference to A; naming B as a contained subtype
    elsewhere names A, not the type being constrained."""
    module = compile_module(
        _module("  A ::= SEQUENCE { v INTEGER, next B OPTIONAL }\n  B ::= A\n  C ::= A (B)"),
        "<t>",
    ).module
    value = {"v": 1, "next": {"v": 2}}
    assert module.decode("C", module.encode("C", value)) == value
