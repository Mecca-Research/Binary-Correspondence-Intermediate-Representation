"""A tagged type is a type: its tags go wherever it goes, on every rule that shows tags.

The 2026-10-06 audit (§4.4, found by ASN1-R's review) found the type model carrying tags on
components only. A tagged assignment's tag was copied onto each untagged component that named
it, so every other place a tagged type can stand dropped a tag:

  * a component's own tag over a tagged type replaced it. RFC 4120's `ticket [3] Ticket`, with
    `Ticket ::= [APPLICATION 1] SEQUENCE {...}`, went out without the `[APPLICATION 1]`;
  * an element of SEQUENCE OF lost it: `SEQUENCE OF [0] INTEGER` encoded as SEQUENCE OF INTEGER,
    and `SEQUENCE OF Ticket` without Ticket's tag;
  * a chain of tagged aliases (`A ::= [1] B`, `B ::= [2] INTEGER`) kept the outermost tag at
    best, and none at all encoded directly;
  * a direct encode of a tagged assignment carried no tag;
  * `[1] IMPLICIT T` with `T ::= [5] CHOICE {...}` was refused, as if T were an untagged CHOICE.

Reviewing the fix found the write plan's OER emitter writing an untagged CHOICE alternative's
INDEX as its tag, on both of its rails (`emit.py` and the C twin, `bcir_emit.c`).

Every test here was RED on the parent and states the property, not the repair -- but one:
`test_sets_and_choices_are_ordered_and_told_apart_by_the_outermost_tag` pins what copying the
tag onto components already got right, so moving the tag onto the type had to keep it.
"""

from __future__ import annotations

from bcir.asn1 import jer, oer, per, xer
from bcir.asn1.emit import EmitRules, emit, flatten
from bcir.asn1.encode_plan import PLAN_VERSION, PLAN_VERSION_TAGGED, compile_encode_plan
from bcir.asn1.tags import Asn1Error
from bcir.frontends.asn1 import Asn1SemanticError, compile_module

_TICKET = "  Ticket ::= [APPLICATION 1] SEQUENCE { v [0] INTEGER }\n"


def _module(body: str, tags: str):
    return compile_module(f"M DEFINITIONS {tags} TAGS ::= BEGIN\n{body}\nEND\n", "<t>").module


def _round_trips(module, name: str, value) -> bytes:
    data = module.encode(name, value)
    assert module.decode(name, data) == value, (name, data.hex())
    return data


def test_a_components_own_tag_keeps_the_tag_of_the_type_it_names():
    """X.680 §31: `[3] Ticket` tags the type Ticket, which is itself tagged, so X.690 nests
    both. IMPLICIT replaces the outermost of Ticket's own tags and keeps that layer's form
    (§8.14.4); a component with no tag of its own shows Ticket's."""
    explicit = _module(
        _TICKET
        + "  AP ::= SEQUENCE { ticket [3] Ticket }\n"
        + "  Over ::= SEQUENCE { ticket [3] IMPLICIT Ticket }\n"
        + "  Holder ::= SEQUENCE { ticket Ticket }",
        "EXPLICIT",
    )
    value = {"ticket": {"v": 5}}
    assert _round_trips(explicit, "AP", value).hex() == "300ba30961073005a003020105"
    assert _round_trips(explicit, "Over", value).hex() == "3009a3073005a003020105"
    assert _round_trips(explicit, "Holder", value).hex() == "300961073005a003020105"
    implicit = _module(_TICKET + "  AP ::= SEQUENCE { ticket [3] Ticket }", "IMPLICIT")
    assert _round_trips(implicit, "AP", value).hex() == "3005a303800105"


def test_a_tagged_assignment_encodes_with_its_tag():
    """A direct encode of `Ticket` is the encoding of the tagged type, and a decode is told the
    tag to expect: the untagged spelling is refused (X.690 §8.1.2)."""
    module = _module(_TICKET, "EXPLICIT")
    assert _round_trips(module, "Ticket", {"v": 5}).hex() == "61073005a003020105"
    assert _module(_TICKET, "IMPLICIT").encode("Ticket", {"v": 5}).hex() == "6103800105"
    try:
        module.decode("Ticket", bytes.fromhex("3005a003020105"))
    except Asn1Error as exc:
        assert "expected [APPLICATION 1]" in str(exc), exc
    else:  # pragma: no cover - the defect
        raise AssertionError("an encoding without Ticket's tag decoded as a Ticket")


def test_an_element_keeps_its_tag():
    """An element of SEQUENCE OF is a type like any other: written tagged, or naming a tagged
    assignment, it carries the tag. The rules on which tags are invisible are unchanged."""
    module = _module(
        _TICKET
        + "  Tagged ::= SEQUENCE OF [0] INTEGER\n"
        + "  Plain ::= SEQUENCE OF INTEGER\n"
        + "  Tickets ::= SEQUENCE OF Ticket",
        "EXPLICIT",
    )
    assert _round_trips(module, "Tagged", [1, 2]).hex() == "300aa003020101a003020102"
    assert _round_trips(module, "Tickets", [{"v": 5}]).hex() == "300961073005a003020105"
    tagged, plain = module.types["Tagged"], module.types["Plain"]
    for encode in (per.encode_per, oer.encode_oer, jer.encode_jer, xer.encode_xer):
        assert encode(tagged, [1, 2]) == encode(plain, [1, 2]), encode


def test_a_chain_of_tagged_aliases_keeps_every_tag():
    body = "  A ::= [1] B\n  B ::= [2] INTEGER\n  S ::= SEQUENCE { a A }"
    explicit = _module(body, "EXPLICIT")
    assert _round_trips(explicit, "A", 5).hex() == "a105a203020105"
    assert _round_trips(explicit, "S", {"a": 5}).hex() == "3007a105a203020105"
    assert _round_trips(_module(body, "IMPLICIT"), "S", {"a": 5}).hex() == "3003810105"


def test_an_implicit_tag_over_a_tagged_choice_is_legal():
    """X.680 §31.2.7 forces EXPLICIT over an UNTAGGED choice only. A tagged CHOICE has a tag
    an IMPLICIT one can replace, so `[1] IMPLICIT T` lowers, and `[1] T` under EXPLICIT TAGS
    nests both tags."""
    choice = "  T ::= [5] CHOICE { a INTEGER, b BOOLEAN }\n"
    module = _module(
        choice + "  Over ::= SEQUENCE { t [1] IMPLICIT T }\n" + "  Nested ::= SEQUENCE { t [1] T }",
        "EXPLICIT",
    )
    value = {"t": ("a", 3)}
    assert _round_trips(module, "Over", value).hex() == "3005a103020103"
    assert _round_trips(module, "Nested", value).hex() == "3007a105a503020103"
    try:
        _module(
            "  U ::= CHOICE { a INTEGER, b BOOLEAN }\n  S ::= SEQUENCE { u [1] IMPLICIT U }",
            "EXPLICIT",
        )
    except Asn1SemanticError as exc:
        assert "IMPLICIT cannot tag a CHOICE" in str(exc), exc
    else:  # pragma: no cover - a regression
        raise AssertionError("an IMPLICIT tag over an untagged CHOICE was accepted")


def test_sets_and_choices_are_ordered_and_told_apart_by_the_outermost_tag():
    """X.680 §8.6 orders a SET by the outermost tag, and OER puts a CHOICE alternative's
    outermost tag on the wire (X.696 §20.1): Ticket's own `[APPLICATION 1]`."""
    module = _module(
        _TICKET
        + "  S ::= SET { b Ticket, a INTEGER }\n"
        + "  C ::= CHOICE { t Ticket, n INTEGER }",
        "EXPLICIT",
    )
    value = {"a": 1, "b": {"v": 5}}
    assert _round_trips(module, "S", value).hex() == "310c02010161073005a003020105"
    assert oer.encode_oer(module.types["S"], value).hex() == "01010105"
    assert oer.encode_oer(module.types["C"], ("t", {"v": 5})).hex() == "410105"
    assert _round_trips(module, "C", ("t", {"v": 5})).hex() == "61073005a003020105"


def _plan_parity(module, name: str, value):
    kind = module.types[name]
    plan = compile_encode_plan(kind, module="M", type_name=name)
    stream = flatten(plan, value)
    assert emit(plan, stream, rules=EmitRules.DER) == module.encode(name, value), name
    assert emit(plan, stream, rules=EmitRules.COER) == oer.encode_oer(kind, value), name
    assert emit(plan, stream, rules=EmitRules.CANONICAL_PER_UNALIGNED) == per.encode_per(
        kind, value
    ), name
    return plan


def test_the_write_plan_carries_every_tag_and_is_unchanged_without_them():
    """The plan-driven emitters reach the oracle's octets for every shape above. A tag a
    member shows on the wire stays where version 5 put it -- on the member -- so a plan for a
    component naming a tagged assignment is version 5 with no `tags` line; a layer no member
    shows (an element's, the root's, one under a member's EXPLICIT tag) makes it version 6,
    which a version-5 reader refuses."""
    module = _module(
        _TICKET
        + "  AP ::= SEQUENCE { ticket [3] Ticket }\n"
        + "  Holder ::= SEQUENCE { ticket Ticket }\n"
        + "  Tagged ::= SEQUENCE OF [0] INTEGER",
        "EXPLICIT",
    )
    holder = _plan_parity(module, "Holder", {"ticket": {"v": 5}})
    text = holder.serialize().decode()
    assert holder.plan_version == PLAN_VERSION and "\ntags " not in text
    assert "tag=1 class=application exp=1" in text
    for name, value in (("AP", {"ticket": {"v": 5}}), ("Tagged", [1, 2]), ("Ticket", {"v": 5})):
        plan = _plan_parity(module, name, value)
        assert plan.plan_version == PLAN_VERSION_TAGGED and "\ntags " in plan.serialize().decode()


def test_the_write_plan_shows_an_untagged_alternatives_universal_tag_under_oer():
    """X.696 §20.1: an alternative is identified by its outermost tag. The OER emitter wrote an
    untagged alternative's index instead, so `CHOICE { a INTEGER, ... }` under EXPLICIT TAGS
    went out as `80 01 03` where the oracle writes `02 01 03`."""
    module = _module(
        "  C ::= CHOICE { a INTEGER, b BOOLEAN }\n  W ::= SEQUENCE { c C }", "EXPLICIT"
    )
    _plan_parity(module, "C", ("a", 3))
    _plan_parity(module, "C", ("b", True))
    _plan_parity(module, "W", {"c": ("b", True)})
