"""J4 part 2 — the commuting projection between the `bcir.asn1.*` dialect and JER.

§7.1 of the JSON roadmap sets this phase's gate, and it names two *different* laws:

    MLIR -> JER -> MLIR   is the identity on the dialect
    JER -> MLIR -> JER    is byte-identical under the canonical profile

**The asymmetry is deliberate and is the thing most likely to be got wrong.** Canonical
JER defines exactly one octet string per abstract value, so a byte claim is meaningful
there and a failure points at an octet. MLIR textual assembly defines no such thing —
`mlir-opt` may reprint attributes in another order and stay correct, and the fixture
corpus is hand-aligned into columns — so a byte claim about MLIR text would be a
statement about a *formatter*, and would fail for reasons unrelated to whether the
projection is right. What must survive that direction is the dialect: the operations,
their symbols, their attributes and their nesting.

The corpus is `mlir/test/passes/verify_asn1.mlir` — the real law fixtures, including the
NEGATIVE ones. That is on purpose: a projection is not a filter, and a module that R24
rejects must round-trip exactly as faithfully as one it accepts. Testing only the legal
modules would hide precisely the case where a lossy projection is dangerous, because a
module that loses an attribute on the way through could come back *legal*.
"""

from __future__ import annotations

import json
import os

from bcir.asn1.dialect import (
    MODULE_TYPE,
    PROJECTION_VERSION,
    DialectModule,
    DialectOperation,
    emit_mlir,
    jer_to_module,
    module_to_jer,
    module_to_value,
    parse_mlir,
    value_to_module,
)
from bcir.asn1.jer import JerRules, decode_jer, encode_jer
from bcir.asn1.jer_bounded import JerBoundedError, decode_bounded
from bcir.asn1.tags import Asn1Error

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FIXTURE = os.path.join(_ROOT, "mlir", "test", "passes", "verify_asn1.mlir")


def _corpus() -> tuple[DialectModule, ...]:
    return parse_mlir(open(_FIXTURE, encoding="utf-8").read())


def test_the_corpus_parses_into_a_nontrivial_set_of_modules():
    """A parser that silently matched nothing would make every law below vacuous.

    This is the guard that keeps the rest of the file honest: every other test iterates
    the corpus, so a regex that stopped matching would turn them all green at once.
    """
    modules = _corpus()
    assert len(modules) >= 20, f"the fixture corpus collapsed to {len(modules)} modules"
    names = {m.name for m in modules}
    assert "BCIR_StreamPack" in names, "the positive StreamPack fixture vanished"
    assert "transcodes" in names, "the transcode fixtures vanished"
    # Every operation kind the dialect has must actually appear, or a whole arm of the
    # CHOICE would go untested by everything below.
    kinds = {o.op for m in modules for o in m.operations}
    assert kinds == {"encode", "decode", "transcode", "projection"}, sorted(kinds)
    # And the type side must carry components, constraints and element references.
    assert any(t.components for m in modules for t in m.types)
    assert any(t.element for m in modules for t in m.types)
    assert any(
        t.constraint_low is not None or t.size_low is not None for m in modules for t in m.types
    )


def test_mlir_to_jer_to_mlir_is_the_identity_on_the_dialect():
    """§7.1's first law, over every fixture module — legal and illegal alike."""
    for module in _corpus():
        assert jer_to_module(module_to_jer(module)) == module, module.name


def test_jer_to_mlir_to_jer_is_byte_identical():
    """§7.1's second law. Bytes, because canonical JER is where bytes are defined."""
    for module in _corpus():
        octets = module_to_jer(module)
        reparsed = parse_mlir(emit_mlir(jer_to_module(octets)))
        assert len(reparsed) == 1, module.name
        assert module_to_jer(reparsed[0]) == octets, module.name


def test_emitting_and_reparsing_mlir_is_the_identity_on_the_dialect():
    """The text rail alone, so a failure above can be localized to a side.

    Without this, a bug in the emitter and a compensating bug in the parser would keep
    both round trips green while the projected JER was wrong.
    """
    for module in _corpus():
        again = parse_mlir(emit_mlir(module))
        assert len(again) == 1 and again[0] == module, module.name


def test_the_projection_is_a_projection_and_not_a_filter():
    """A module R24 REJECTS must round-trip as faithfully as one it accepts.

    This is the case that matters most and is easiest to get wrong. `cer_module` declares
    non-canonical rules and `transcode_identity` transcodes a syntax to itself; both are
    R24 failures. If the projection quietly dropped or normalized what makes them
    illegal, they would come back *legal* — a projection that launders its input is worse
    than one that refuses it, because nothing downstream can tell.
    """
    by_name = {m.name: m for m in _corpus()}
    cer = by_name["cer_module"]
    assert cer.rules == "cer", "the illegal rules value was normalized away"
    assert jer_to_module(module_to_jer(cer)).rules == "cer"

    identity = by_name["transcode_identity"]
    op = identity.operations[0]
    assert op.from_rules == op.to_rules == "der", "the illegal transcode was repaired"
    assert jer_to_module(module_to_jer(identity)).operations[0] == op


def test_every_operation_field_survives_the_round_trip():
    """Field by field, not just record equality, so a dropped attribute is named.

    Record equality already covers this, but it reports "these differ" rather than which
    attribute vanished — and an attribute that silently stops projecting is exactly the
    failure this phase exists to prevent.
    """
    for module in _corpus():
        back = jer_to_module(module_to_jer(module))
        for before, after in zip(module.operations, back.operations, strict=True):
            for slot in (
                "op",
                "name",
                "type",
                "rules",
                "strict_der",
                "strict_canonical",
                "source",
                "from_rules",
                "to_rules",
                "preserve_value",
                "native",
                "additive",
            ):
                assert getattr(before, slot) == getattr(after, slot), (
                    f"{module.name}.{before.name}: {slot} was "
                    f"{getattr(before, slot)!r}, came back {getattr(after, slot)!r}"
                )


def test_an_operation_cannot_carry_another_arms_fields():
    """X.680 §29's CHOICE, doing the job the flat record could not.

    A `SEQUENCE` of optional `native`/`from`/`to`/`rules` fields would let an encode
    travel with a `native` attribute — expressible and meaningless. The CHOICE makes the
    contradiction unrepresentable rather than merely refused, which is the same argument
    J4 part 1 used to keep `BCIR_Asn1Rules` one enum instead of a (family, profile) pair.
    """
    smuggler = DialectOperation(
        op="encode",
        name="e",
        type="T",
        rules="der",
        native="streampack",
        additive=True,
        preserve_value=True,
    )
    module = DialectModule(
        name="M",
        oid=(1, 3, 6, 1, 4, 1, 62596, 99),
        rules="der",
        default_tagging="implicit",
        operations=(smuggler,),
    )
    back = jer_to_module(module_to_jer(module))
    carried = back.operations[0]
    assert carried.op == "encode" and carried.rules == "der"
    # The other arm's fields did not travel: they are not part of what an encode IS.
    assert carried.native is None and not carried.additive and not carried.preserve_value
    # And the JSON itself shows one alternative, keyed by name (X.697 §23).
    document = json.loads(module_to_jer(module))
    assert list(document["operations"][0]) == ["encode"], document["operations"][0]


def test_the_canonical_projection_omits_a_flag_that_is_false():
    """X.690 §11.5 via X.697 §21.2: a component equal to its DEFAULT is not encoded.

    The flags are DEFAULT FALSE rather than OPTIONAL for exactly this reason — absent and
    `false` are the same fact, and admitting both spellings would give one module two
    canonical forms.
    """
    plain = DialectOperation(op="decode", name="d", type="T", rules="ber")
    strict = DialectOperation(op="decode", name="d", type="T", rules="ber", strict_canonical=True)
    base = dict(
        name="M", oid=(1, 3, 6, 1, 4, 1, 62596, 98), rules="der", default_tagging="implicit"
    )
    quiet = json.loads(module_to_jer(DialectModule(**base, operations=(plain,))))
    loud = json.loads(module_to_jer(DialectModule(**base, operations=(strict,))))
    assert "strictCanonical" not in quiet["operations"][0]["decode"]
    assert loud["operations"][0]["decode"]["strictCanonical"] is True


def test_the_projection_carries_its_version_and_refuses_another():
    """A reader must not infer the shape from which members happen to be present.

    Inferring a version from a shape is how a format acquires two incompatible readings
    of one document; the version is carried so a reader can refuse instead of guess.
    """
    module = _corpus()[0]
    value = module_to_value(module)
    assert value["version"] == PROJECTION_VERSION
    value["version"] = PROJECTION_VERSION + 1
    try:
        value_to_module(value)
    except Asn1Error as error:
        assert "refuses rather than inferring" in str(error)
    else:
        raise AssertionError("a future projection version was silently accepted")


def test_the_projected_document_goes_through_the_bounded_reader():
    """The projection is a trust boundary like any other, so J1's limits apply.

    A dialect module that arrived over a wire is attacker-chosen input. `decode_bounded`
    enforces every §4.3 limit before a value graph exists; this pins that the projected
    schema is readable through it, and that a document exceeding a limit is refused with
    a structured diagnostic rather than parsed.
    """
    from bcir.asn1.jer_bounded import STRICT_LIMITS

    module = _corpus()[0]
    octets = module_to_jer(module)
    assert value_to_module(decode_bounded(octets, MODULE_TYPE)) == module
    # And a ceiling the document exceeds is a refusal, not a parse.
    tight = STRICT_LIMITS.tightened(input_bytes=len(octets) - 1)
    try:
        decode_bounded(octets, MODULE_TYPE, limits=tight)
    except JerBoundedError as error:
        assert error.diagnostic.code.value == "input-too-large"
    else:
        raise AssertionError("the bounded reader ignored its input ceiling")


def test_the_parser_refuses_an_attribute_form_it_cannot_represent():
    """Fail closed. A silently ignored attribute is an attribute that vanishes.

    The parser covers the value forms this dialect uses and nothing wider. That bound is
    what makes it safe to hand-write, and it is only safe if the boundary is a refusal
    rather than a skip — otherwise a module carrying an unsupported attribute would
    project to a *smaller* module than it was given, and round-trip cleanly while doing it.
    """
    text = """
    bcir.asn1.module @Odd attributes {
      oid = array<i64: 1, 3, 6>,
      rules = #bcir.asn1_rules<der>,
      default_tagging = #bcir.asn1_tagging<implicit>
    } {
      bcir.asn1.type @T attributes { kind = "primitive", universal = 2 : i64,
                                     weird = dense<[1, 2]> : tensor<2xi64> } { }
    }
    """
    try:
        parse_mlir(text)
    except Asn1Error as error:
        assert "unsupported attribute value" in str(error), str(error)
    else:
        raise AssertionError("an unrepresentable attribute was silently dropped")


def test_a_brace_inside_a_string_does_not_end_the_attribute_dictionary():
    """The scanner is quote-aware, and this is why.

    `default_value = "}"` is legal — DEFAULT values are rendered as text — and a naive
    brace counter would end the dictionary there and parse the rest of the module as
    something else entirely.
    """
    text = """
    bcir.asn1.module @Braces attributes {
      oid = array<i64: 1, 3, 6>,
      rules = #bcir.asn1_rules<der>,
      default_tagging = #bcir.asn1_tagging<implicit>
    } {
      bcir.asn1.type @S attributes { kind = "sequence" } {
        bcir.asn1.component { name = "a", type = @S, has_default, default_value = "}" }
      }
      bcir.asn1.encode @e { type = @S, rules = #bcir.asn1_rules<der> }
    }
    """
    modules = parse_mlir(text)
    assert len(modules) == 1
    assert modules[0].types[0].components[0].default_value == "}"
    assert [o.name for o in modules[0].operations] == ["e"], (
        "the operation after the brace-carrying string was lost"
    )
    assert jer_to_module(module_to_jer(modules[0])) == modules[0]


def test_the_projection_is_the_same_type_under_another_transfer_syntax():
    """The schema is not where the realization choice lives (roadmap §0).

    One type model, several syntaxes: the same `MODULE_TYPE` that produced the JER above
    encodes under BASIC JER too, and the two decode to the same value. This is what makes
    the projection a *transfer syntax* question rather than a schema question, and it is
    the property J4 part 1's `bcir.asn1.transcode` names on the law rail.
    """
    module = _corpus()[0]
    value = module_to_value(module)
    canonical = encode_jer(MODULE_TYPE, value, rules=JerRules.CANONICAL)
    basic = encode_jer(MODULE_TYPE, value, rules=JerRules.BASIC)
    # Compared as octets and as modules, never as raw ASN.1 values. A decoded SEQUENCE OF
    # is a list where `module_to_value` built a tuple, and an omitted DEFAULT comes back
    # present — both are representation differences in the intermediate form, not
    # differences in the value it denotes. Asserting equality there would be testing
    # Python container identity, and would have to be "fixed" by making the pivot produce
    # whatever the decoder happens to return, which is the tail wagging the dog.
    assert (
        encode_jer(
            MODULE_TYPE,
            decode_jer(canonical, MODULE_TYPE, rules=JerRules.CANONICAL),
            rules=JerRules.CANONICAL,
        )
        == canonical
    )
    assert value_to_module(decode_jer(canonical, MODULE_TYPE, rules=JerRules.CANONICAL)) == module
    assert value_to_module(decode_jer(basic, MODULE_TYPE, rules=JerRules.BASIC)) == module


# --- §6.3: StreamPack over JER ---------------------------------------------------------------


def _packs():
    from bcir.examples import PROGRAMS
    from bcir.gem import hydrate
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta

    h, theta = TargetProfile.x86_avx512(), Theta.cool()
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        yield name, hydrate(module, optimize(module, h, theta))


def test_a_streampack_survives_the_jer_projection():
    """The same round-trip law the DER and OER projections already hold to."""
    from bcir.asn1.streampack import decode_pack_jer, encode_pack_jer

    count = 0
    for name, pack in _packs():
        assert decode_pack_jer(encode_pack_jer(pack)) == pack, name
        count += 1
    assert count >= 10, f"corpus degenerated to {count} program(s)"


def test_the_native_streampack_bytes_survive_the_jer_round_trip():
    """§6.3, as a test rather than a promise: **additive** means the frozen wire format is
    unchanged by a trip through JSON.

    This is the claim that keeps JER from becoming a second definition of a StreamPack. It
    compares the NATIVE octets, not the JSON — a projection that round-tripped its own
    text while perturbing the artifact would satisfy a weaker test and be useless.
    """
    from bcir.abi.streampack_abi import decode as abi_decode
    from bcir.abi.streampack_abi import encode as abi_encode
    from bcir.asn1.streampack import decode_pack_jer, encode_pack_jer

    for name, pack in _packs():
        native = abi_encode(pack)
        rebuilt = abi_encode(decode_pack_jer(encode_pack_jer(abi_decode(native))))
        assert rebuilt == native, name


def test_the_jer_projection_is_canonical_and_idempotent():
    """One StreamPack, one canonical JER spelling — the property a digest rests on."""
    from bcir.asn1.streampack import decode_pack_jer, encode_pack_jer

    for name, pack in _packs():
        raw = encode_pack_jer(pack)
        assert encode_pack_jer(decode_pack_jer(raw)) == raw, name


def test_jer_is_larger_than_the_binary_projections_and_that_is_recorded():
    """§1's boundary, kept honest by measurement rather than by assertion.

    JER is a build-, control- and load-plane format. The roadmap says so; this checks the
    reason is real, so nobody later proposes it for a hot path on the assumption that the
    cost is small. If this ever fails because JER got *smaller* than DER, that is a
    finding worth having rather than a test to relax.
    """
    from bcir.asn1.streampack import encode_pack, encode_pack_jer, encode_pack_oer

    for name, pack in _packs():
        der, oer, jer = (
            len(encode_pack(pack)),
            len(encode_pack_oer(pack)),
            len(encode_pack_jer(pack)),
        )
        assert jer > der > oer, f"{name}: der={der} oer={oer} jer={jer}"
