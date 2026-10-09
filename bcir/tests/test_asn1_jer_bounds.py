"""ASN1-B: semantic bounds on canonical JER, derived at the point of application.

The 2026-10-06 audit (items 19 and 20) found no schema-derived JER memory bound: the J2 plan
bounds only BOOLEAN, NULL, ENUMERATED and a fixed BIT STRING, because a value constraint does
not reach a JER encoder. It does restrict the values, and these witnesses hold the bound that
follows to the canonical encoder: every bound reached by a valid value, none exceeded by
sampled valid values, at the point a parameterized type is applied, with the types that stay
unbounded named -- and the differential shown to fire on wrong copies of the derivation.
"""

from __future__ import annotations

import sys
import types

from bcir.asn1 import jer_bounds
from bcir.asn1.jer import encode_jer
from bcir.asn1.jer_bounds import max_octets, witness
from bcir.frontends.asn1 import compile_module
from bcir.tests import jer_bounds_fixtures as fixtures


def _types(body: str) -> dict:
    text = f"M DEFINITIONS AUTOMATIC TAGS ::= BEGIN\n{body}\nEND\n"
    return compile_module(text, "<t>").module.types


def test_hand_derived_bounds():
    t = _types(
        """
        Byte ::= INTEGER (0..255)
        Signed ::= INTEGER (-128..127)
        Four ::= OCTET STRING (SIZE (4))
        Ia5 ::= IA5String (SIZE (3))
        Printable ::= PrintableString (SIZE (3))
        Visible ::= VisibleString (SIZE (2))
        Hex ::= IA5String (FROM ("a".."c")) (SIZE (1..3))
        One ::= UTF8String (SIZE (1))
        Fixed ::= BIT STRING (SIZE (12))
        Ranged ::= BIT STRING (SIZE (0..12))
        Flags ::= SEQUENCE (SIZE (0..2)) OF BOOLEAN
        Colour ::= ENUMERATED {red, ultraviolet}
        Pick ::= CHOICE { n NULL, b BOOLEAN }
        Pair ::= SEQUENCE { a Byte, b Pick OPTIONAL }
        """
    )
    expected = {
        "Byte": 3,  # 255
        "Signed": 4,  # -128
        "Four": 10,  # "FFFFFFFF"
        "Ia5": 2 + 3 * 6,  # three \u0000
        "Printable": 5,
        "Visible": 2 + 2 * 2,  # two \"
        "Hex": 5,
        "One": 2 + 6,  # \u0000
        "Fixed": 2 + 2 * 2,
        "Ranged": len('{"value":"FFFF","length":12}'),
        "Flags": len("[false,false]"),
        "Colour": len('"ultraviolet"'),
        "Pick": len('{"b":false}'),
        "Pair": len('{"a":255,"b":{"b":false}}'),
    }
    for name, octets in expected.items():
        assert max_octets(t[name]) == octets, name
        value = witness(t[name])
        assert fixtures.admits(t[name], value), name
        assert len(encode_jer(t[name], value)) == octets, name


def test_the_bound_is_taken_at_the_point_of_application():
    """A parameterized type is bounded as applied: the same template, two bounds."""
    t = _types(
        """
        Bounded {INTEGER: ub} ::= INTEGER (0..ub)
        Sized {INTEGER: ub} ::= OCTET STRING (SIZE (0..ub))
        Listed {Elem, INTEGER: ub} ::= SEQUENCE (SIZE (1..ub)) OF Elem
        Small ::= Bounded {9}
        Big ::= Bounded {99999}
        Short ::= Sized {2}
        Long ::= Sized {300}
        Nested ::= SEQUENCE { k Bounded {42}, v Listed {Bounded {7}, 3} }
        """
    )
    assert [max_octets(t[n]) for n in ("Small", "Big", "Short", "Long")] == [1, 5, 6, 602]
    assert max_octets(t["Nested"]) == len('{"k":42,"v":[7,7,7]}')
    assert len(encode_jer(t["Nested"], witness(t["Nested"]))) == max_octets(t["Nested"])


def test_what_stays_unbounded():
    t = _types(
        """
        Any ::= INTEGER
        Open ::= INTEGER (0..9, ...)
        Text ::= UTF8String
        Big ::= OCTET STRING (SIZE (1..MAX))
        Real ::= REAL
        Id ::= OBJECT IDENTIFIER
        When ::= GeneralizedTime
        Tree ::= SEQUENCE { v BOOLEAN, kids SEQUENCE (SIZE (0..2)) OF Tree }
        Holder ::= SEQUENCE { a BOOLEAN, b Text }
        """
    )
    for name in t:
        assert max_octets(t[name]) is None, name
        try:
            witness(t[name])
        except ValueError:
            continue
        raise AssertionError(f"{name} produced a witness")


def test_every_bound_is_reached_and_never_exceeded():
    """The generated corpus (120 modules: constrained leaves, structures, templates applied at
    drawn actuals) and the repository's own modules: every stated bound reached by a valid
    witness, none exceeded by thousands of sampled valid values, and the parent's rail -- the
    J2 plan, which bounds four leaf kinds and counts a comma per member -- leaving most of the
    bounded types unstated."""
    stats: dict = {}
    assert fixtures.measure(stats=stats) == {row: 0.0 for row in fixtures.ROWS}
    assert stats["bounded"] >= 600 and stats["samples"] >= 10 * stats["bounded"], stats
    parent = fixtures.measure(fixtures.plan_rail)
    assert parent["asn1.jer.bound.unstated"] > 500 and parent["asn1.jer.bound.unsound"] == 0


def test_the_differential_fires_on_wrong_derivations():
    """Wrong copies of the derivation, loaded from the module source (never by editing the
    tree): an integer's spelling without its sign, VisibleString's quote unescaped, a comma for
    every member, and the narrowest alternative of a choice. Each is caught."""
    source = open(jer_bounds.__file__, encoding="utf-8").read()
    for name, old, new, row in (
        ("sign", "best = max((low, high), key=lambda v: (len(str(v)), v == low))\n"
         "        return len(str(best)), best",
         "best = max((low, high), key=lambda v: (len(str(abs(v))), v == low))\n"
         "        return len(str(abs(best))), best", "asn1.jer.bound.loose"),
        ("visible", "Universal.VISIBLE_STRING: (0x20, 0x7E)", "Universal.VISIBLE_STRING: (0x23, 0x5B)",
         "asn1.jer.bound.unsound"),
        ("commas", "return 2 + sum(members) + max(len(members) - 1, 0), value",
         "return 2 + sum(members) + len(members), value", "asn1.jer.bound.loose"),
        ("choice", "if best is None or total > best[0]:", "if best is None or total < best[0]:",
         "asn1.jer.bound.unsound"),
    ):  # fmt: skip
        assert source.count(old) == 1, name
        mutant = types.ModuleType(f"bcir.asn1._jer_bounds_{name}")
        mutant.__package__ = "bcir.asn1"
        sys.modules[mutant.__name__] = mutant
        try:
            exec(compile(source.replace(old, new), mutant.__name__, "exec"), mutant.__dict__)
            saved = jer_bounds.max_octets, jer_bounds.witness
            jer_bounds.max_octets, jer_bounds.witness = mutant.max_octets, mutant.witness
            try:
                rows = fixtures.measure()
            finally:
                jer_bounds.max_octets, jer_bounds.witness = saved
            assert rows[row] > 0, f"the differential cannot see {name}: {rows}"
        finally:
            del sys.modules[mutant.__name__]
