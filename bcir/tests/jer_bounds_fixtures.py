"""ASN1-B fixtures, shared by `test_asn1_jer_bounds` and the GEM+ harness so both grade the same
way: generated constrained types (written as ASN.1 and compiled, parameterized templates applied
with drawn actuals among them), the repository's own modules, a sampler of valid values that
shares no code with the bound, and the grading.

    asn1.jer.bound.unsound   sampled valid values whose canonical JER exceeds the bound stated
                             for their type
    asn1.jer.bound.unstated  types with a finite canonical JER maximum -- shown by a valid
                             witness that attains it -- for which the rail states no bound
    asn1.jer.bound.loose     types whose stated bound no valid value attains (a witness that is
                             invalid, or shorter than the bound)
"""

from __future__ import annotations

import os
import random

from bcir.asn1.codec import NULL, BitString
from bcir.asn1.jer import encode_jer
from bcir.asn1.schema import Choice, Primitive, Reference, Sequence, SequenceOf, Set, SetOf
from bcir.asn1.tags import Universal

ROWS = ("asn1.jer.bound.unsound", "asn1.jer.bound.unstated", "asn1.jer.bound.loose")
GENERATED = 120  # generated modules
SAMPLES = 24  # valid values sampled per bounded type

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


# --- the corpus --------------------------------------------------------------------------------

_STRINGS = ("IA5String", "VisibleString", "PrintableString", "NumericString", "UTF8String",
            "BMPString", "UniversalString")  # fmt: skip


def _leaf(r: random.Random) -> str:
    kind = r.randrange(11)
    lo = r.randint(-300, 300)
    if kind == 0:
        return f"INTEGER ({lo}..{lo + r.randint(0, 100000)})"
    if kind == 1:
        return "INTEGER"  # unbounded
    if kind == 2:
        return f"INTEGER ({lo}..{lo + 9}, ...)"  # extensible: unbounded
    if kind == 3:
        return f"OCTET STRING (SIZE ({r.randint(0, 3)}..{r.randint(3, 9)}))"
    if kind == 4:
        size = r.randint(1, 20)
        return (
            f"BIT STRING (SIZE ({size}))" if r.random() < 0.5 else f"BIT STRING (SIZE (0..{size}))"
        )
    if kind == 5:
        string = r.choice(_STRINGS)
        limit = f"(SIZE ({r.randint(0, 2)}..{r.randint(2, 6)}))"
        if string in ("IA5String", "VisibleString", "UTF8String") and r.random() < 0.4:
            limit = (
                f'(FROM ("a".."f" | """")) {limit}'
                if r.random() < 0.5
                else f'(FROM ("0".."9")) {limit}'
            )
        return f"{string} {limit}"
    if kind == 6:
        return "BOOLEAN"
    if kind == 7:
        return "NULL"
    if kind == 8:
        names = ["red", "green", "ultraviolet", "x"][: r.randint(1, 4)]
        return "ENUMERATED {" + ", ".join(names) + "}"
    if kind == 9:
        return f"OCTET STRING (SIZE ({r.randint(1, 5)}))"
    return "UTF8String"  # unbounded


def _type(r: random.Random, depth: int) -> str:
    if depth >= 2 or r.random() < 0.45:
        return _leaf(r)
    shape = r.randrange(4)
    if shape == 0:
        comps = [f"c{i} {_type(r, depth + 1)}{' OPTIONAL' if r.random() < 0.3 else ''}"
                 for i in range(r.randint(1, 3))]  # fmt: skip
        return "SEQUENCE { " + ", ".join(comps) + " }"
    if shape == 1:
        alts = [f"a{i} {_type(r, depth + 1)}" for i in range(r.randint(1, 3))]
        return "CHOICE { " + ", ".join(alts) + " }"
    if shape == 2:
        return f"SEQUENCE (SIZE (0..{r.randint(0, 4)})) OF {_type(r, depth + 1)}"
    return f"SET (SIZE ({r.randint(1, 2)}..3)) OF {_type(r, depth + 1)}"


def generated_module(seed: int) -> str:
    """A module of drawn constrained types, two templates applied at drawn actuals, and a
    type that applies a template inside a structure -- the point of application."""
    r = random.Random(seed)
    lines = [f"Gen{seed} DEFINITIONS AUTOMATIC TAGS ::= BEGIN"]
    lines.append("Ranged {INTEGER: lo, INTEGER: hi} ::= INTEGER (lo..hi)")
    lines.append("Sized {INTEGER: ub} ::= OCTET STRING (SIZE (0..ub))")
    lines.append("Listed {Elem, INTEGER: ub} ::= SEQUENCE (SIZE (1..ub)) OF Elem")
    for i in range(r.randint(2, 5)):
        lines.append(f"T{i} ::= {_type(r, 0)}")
    lo = r.randint(-5000, 5000)
    lines.append(f"P0 ::= Ranged {{{lo}, {lo + r.randint(0, 10 ** r.randint(0, 9))}}}")
    lines.append(f"P1 ::= Sized {{{r.randint(0, 40)}}}")
    lines.append(f"P2 ::= SEQUENCE {{ head P0, body Listed {{T0, {r.randint(1, 3)}}} }}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def corpus() -> list[tuple[str, object]]:
    """(label, type): every assigned type of the generated modules and of the repository's own
    ASN.1 modules (the four BCIR projections and the PKIX test data)."""
    from bcir.frontends.asn1 import compile_module

    out = []
    for seed in range(GENERATED):
        module = compile_module(generated_module(seed), f"<gen{seed}>").module
        out += [(f"gen{seed}.{name}", kind) for name, kind in module.types.items()]
    for path in (
        "bcir/asn1/BCIR-ArtifactBundle.asn1",
        "bcir/asn1/BCIR-ControlPlane.asn1",
        "bcir/asn1/BCIR-ExecutionPlan.asn1",
        "bcir/asn1/BCIR-StreamPack.asn1",
    ):
        with open(os.path.join(_ROOT, path), encoding="utf-8") as f:
            module = compile_module(f.read(), path).module
        out += [(f"{os.path.basename(path)}.{n}", k) for n, k in module.types.items()]
    return out


# --- the judge and the sampler (no code shared with bcir.asn1.jer_bounds) -------------------

_VISIBLE = (0x20, 0x7E)


def _repertoire_ok(universal: int, text: str) -> bool:
    from bcir.asn1.values import _NUMERIC, _PRINTABLE

    if universal == Universal.PRINTABLE_STRING:
        return set(text) <= _PRINTABLE
    if universal == Universal.NUMERIC_STRING:
        return set(text) <= _NUMERIC
    if universal == Universal.IA5_STRING:
        return all(ord(c) < 0x80 for c in text)
    if universal == Universal.VISIBLE_STRING:
        return all(_VISIBLE[0] <= ord(c) <= _VISIBLE[1] for c in text)
    if universal == Universal.BMP_STRING:
        return all(ord(c) <= 0xFFFF for c in text)
    return all(not 0xD800 <= ord(c) <= 0xDFFF for c in text)


def admits(kind, value) -> bool:
    """X.680's value set of `kind`: each primitive's constraint (`Constraint.permits`, the model's
    own statement of the value set) and repertoire, each component and element."""
    if isinstance(kind, Reference):
        return admits(kind.resolved(), value)
    if isinstance(kind, Primitive):
        if kind.universal == Universal.BIT_STRING:
            size = value.bit_length if isinstance(value, BitString) else None
            return size is not None and (kind.constraint is None or kind.constraint.permits(
                b"\0" * size))  # fmt: skip
        if isinstance(value, str) and not _repertoire_ok(kind.universal, value):
            return False
        if kind.universal == Universal.ENUMERATED:
            names = {n for n, _ in (kind.enumeration or ())} | {
                n for n, _ in (kind.enum_extension or ())
            }
            return value in names
        return kind.constraint is None or kind.constraint.permits(value)
    if isinstance(kind, (Sequence, Set)):
        from bcir.asn1.jer import _flatten

        return all(
            admits(c.type, value[c.name]) if c.name in value else (c.optional or c.has_default)
            for c in _flatten(kind.components)
        )
    if isinstance(kind, (SequenceOf, SetOf)):
        ok = kind.constraint is None or kind.constraint.permits(value)
        return ok and all(admits(kind.element, v) for v in value)
    if isinstance(kind, Choice):
        from bcir.asn1.jer import _flatten

        name, inner = value
        return any(a.name == name and admits(a.type, inner) for a in _flatten(kind.alternatives))
    return False


_POOL = (
    "a", "f", "0", "9", " ", "Z", "'", '"', "\\", "\n", "\x00", "\x1f", "~", "\x7f", "\x80",
    "é", "ÿ", "€", "￿", "日", "\U0001f600", "\U0010ffff",
)  # fmt: skip


def sample(kind, r: random.Random, depth: int = 0):
    """A valid value of `kind` drawn at random -- extremes as often as the middle -- or None
    when none was drawn (the caller keeps only values `admits` accepts)."""
    if isinstance(kind, Reference):
        return None if depth > 6 else sample(kind.resolved(), r, depth + 1)
    if isinstance(kind, Primitive):
        return _sample_primitive(kind, r)
    if isinstance(kind, (Sequence, Set)):
        from bcir.asn1.jer import _flatten

        out = {}
        for comp in _flatten(kind.components):
            if comp.optional and r.random() < 0.3:
                continue
            item = sample(comp.type, r, depth + 1)
            if item is None:
                if comp.optional or comp.has_default:
                    continue
                return None
            out[comp.name] = item
        return out
    if isinstance(kind, (SequenceOf, SetOf)):
        n = _draw_size(kind, r)
        items = [sample(kind.element, r, depth + 1) for _ in range(n)]
        return None if any(i is None for i in items) else items
    if isinstance(kind, Choice):
        from bcir.asn1.jer import _flatten

        alt = r.choice(_flatten(kind.alternatives))
        inner = sample(alt.type, r, depth + 1)
        return None if inner is None else (alt.name, inner)
    return None


def _sample_primitive(kind: Primitive, r: random.Random):
    u = kind.universal
    if u == Universal.BOOLEAN:
        return r.random() < 0.5
    if u == Universal.NULL:
        return NULL
    if u == Universal.ENUMERATED:
        names = [n for n, _ in (kind.enumeration or ())]
        return r.choice(names) if names else None
    if u == Universal.INTEGER:
        bounds = kind.constraint.value_bounds() if kind.constraint is not None else (None, None)
        lo, hi = bounds if bounds[0] is not None and bounds[1] is not None else (-(10**6), 10**6)
        return r.choice((lo, hi, r.randint(lo, hi), (lo + hi) // 2))
    n = _draw_size(kind, r)
    if u == Universal.OCTET_STRING:
        return bytes(r.randrange(256) for _ in range(n))
    if u == Universal.BIT_STRING:
        bits = n
        octets = (bits + 7) // 8
        return BitString(bytes(r.randrange(256) for _ in range(octets)), 8 * octets - bits)
    if u in (Universal.IA5_STRING, Universal.VISIBLE_STRING, Universal.PRINTABLE_STRING,
             Universal.NUMERIC_STRING, Universal.UTF8_STRING, Universal.BMP_STRING,
             Universal.UNIVERSAL_STRING):  # fmt: skip
        allowed = kind.constraint.alphabet() if kind.constraint is not None else None
        pool = [c for c in _POOL if allowed is None or c in allowed] or sorted(allowed or "a")
        pool = [c for c in pool if _repertoire_ok(u, c)] or ["0"]
        return "".join(r.choice(pool) for _ in range(n))
    return None


def _draw_size(kind, r: random.Random) -> int:
    """A length the type's SIZE admits, the ends as often as the middle (`Constraint.size_bounds`,
    the model's own statement; a drawn length outside an extensible root is still judged)."""
    lo, hi = (None, None)
    if getattr(kind, "constraint", None) is not None:
        lo, hi = kind.constraint.size_bounds()
    lo = 0 if lo is None else lo
    hi = lo + 40 if hi is None else hi
    return r.choice((lo, hi, r.randint(lo, hi), r.randint(lo, hi)))


# --- the grading ---------------------------------------------------------------------------------


def _stated(kind, rail) -> int | None:
    return rail(kind)


def measure(rail=None, stats: dict | None = None) -> dict[str, float]:
    """The ASN1-B rows. `rail(kind)` is the bound the rail states (`jer_bounds.max_octets` by
    default; the parent's rail is the J2 plan's `bounded_octets`). `stats` receives the number
    of types, of bounded types and of admitted samples -- the row is only as strong as these."""
    from bcir.asn1 import jer_bounds

    rail = rail or jer_bounds.max_octets
    out = {row: 0.0 for row in ROWS}
    counts = {"types": 0, "bounded": 0, "samples": 0}
    r = random.Random(2026)
    for _label, kind in corpus():
        counts["types"] += 1
        stated = rail(kind)
        counts["bounded"] += stated is not None
        try:
            truth = jer_bounds.witness(kind)
        except ValueError:
            truth = None
        if truth is not None and admits(kind, truth):
            longest = len(encode_jer(kind, truth))
            if stated is None:
                out["asn1.jer.bound.unstated"] += 1
            elif stated != longest:
                out["asn1.jer.bound.loose"] += 1
        elif stated is not None:
            out["asn1.jer.bound.loose"] += 1  # a bound no valid value was shown to reach
        if stated is None:
            continue
        for _ in range(SAMPLES):
            value = sample(kind, r)
            if value is None or not admits(kind, value):
                continue
            counts["samples"] += 1
            if len(encode_jer(kind, value)) > stated:
                out["asn1.jer.bound.unsound"] += 1
    if stats is not None:
        stats.update(counts)
    return out


def plan_rail(kind) -> int | None:
    """The parent's statement: the J2 plan compiler's `bounded_octets` for the type."""
    from bcir.asn1.jer import JerRules, _Opts
    from bcir.asn1.jer_plan import _compile_node

    try:
        return _compile_node(kind, _Opts(JerRules.CANONICAL, None), "$", 0).bounded_octets
    except Exception:  # noqa: BLE001 -- a type the plan compiler refuses states no bound
        return None
