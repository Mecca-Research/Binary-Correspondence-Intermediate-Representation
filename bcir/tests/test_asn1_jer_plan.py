"""J2 — the schema-plan compiler.

`docs/BCIR_ASN1_JSON_ROADMAP.md` §7 states the exit gate in three parts: "Byte-identical
descriptor regeneration; unsupported schema/instruction refusal; Python table-driven/direct
traces agree." Each has a section below, and §5.1's three properties — byte-identical
compilation, fail-closed on unknown features, and "descriptors are data; they contain no
process pointers or executable callbacks when serialized" — are pinned directly.

The most informative test is `test_almost_nothing_is_statically_bounded_and_that_is_what_
jer_is`. §5.1 asks for static capacity "where statically derivable", and for JER that is
four things — a property of the encoding rules, not of this compiler, and the reason J1 has
runtime limits and J3's C interface must take its capacity from the caller.
"""

from __future__ import annotations

import hashlib

from bcir.asn1.codec import Asn1Error
from bcir.asn1.constraints import Size, ValueRange
from bcir.asn1.jer import Array, JerInstructions, JerRules, Name, NameKeyword, Unwrapped, encode_jer
from bcir.asn1.jer_bounded import JerBoundedError, JerErrorCode
from bcir.asn1.jer_plan import (
    FAMILY,
    PLAN_COMPILER,
    PLAN_VERSION,
    PROFILE_BASIC,
    PROFILE_CANONICAL,
    compile_plan,
    decode_with_plan,
    trace_of,
)
from bcir.asn1.schema import (
    Choice,
    Component,
    ObjectSetTable,
    OpenType,
    Primitive,
    Sequence,
    SequenceOf,
)
from bcir.asn1.tags import Universal
from bcir.frontends.asn1.lower import compile_module

#: The ASN.1 schema for `channel.json`, whose fields are the ones
#: `runtime/c/bcir_channel.c` reads today. §5.4 names it as the first integration target,
#: and it is a good one precisely because a hand-written bounded C reader already exists to
#: compare against in J3.
CHANNEL_MODULE = """
Channel DEFINITIONS ::= BEGIN
  ChannelDescriptor ::= SEQUENCE {
      name         UTF8String,
      kind         Kind,
      provenance   Provenance DEFAULT real,
      modeled      BOOLEAN DEFAULT FALSE,
      capabilities SEQUENCE OF Capability }
  Kind ::= ENUMERATED { cpu(0), gpu(1), fpga(2), accelerator(3), storage(4), memory(5) }
  Provenance ::= ENUMERATED { real(0), modeled(1), simulated(2) }
  Capability ::= ENUMERATED { universal(0), data-parallel(1), reduce(2), gather(3),
                              tile(4), matmul(5), stream-unit(6), scalar-stream(7) }
END
"""

_CHANNEL_VALUE = {
    "name": "gpu0",
    "kind": 1,
    "provenance": 0,
    "modeled": False,
    "capabilities": [0, 5],
}


def _channel():
    return compile_module(CHANNEL_MODULE, "channel.asn1").module.types["ChannelDescriptor"]


def _channel_plan(kind=None, **kwargs):
    """Compile a plan for the channel schema.

    `kind` is a parameter rather than always freshly lowered because `JerInstructions` is
    keyed by object identity (X.697 §9.9 inheritance) -- a plan compiled from a *different*
    lowering of the same module would silently see no instructions at all. Passing the same
    type the instructions were assigned to is the caller's obligation, and forgetting it is
    exactly the mistake the identity keying is designed to make visible.
    """
    return compile_plan(
        kind if kind is not None else _channel(),
        module="Channel",
        type_name="ChannelDescriptor",
        source=CHANNEL_MODULE,
        **kwargs,
    )


def _refuses(action, needle: str) -> None:
    try:
        action()
    except (JerBoundedError, Asn1Error) as error:
        assert needle in str(error), f"expected {needle!r} in {error}"
        return
    raise AssertionError(f"expected a refusal mentioning {needle!r}")


# --- gate 1: byte-identical descriptor regeneration --------------------------------------


def test_repeated_compilation_is_byte_identical():
    """§5.1 — "Repeated compilation is byte-identical."

    Not merely equal: the *bytes* must match, because the descriptor's SHA-256 is what
    names it downstream. A serializer that sorted by a dict's iteration order, or embedded
    a memory address, would pass an equality test and fail this one.
    """
    first, second = _channel_plan(), _channel_plan()
    assert first.serialize() == second.serialize()
    assert first.sha256() == second.sha256()
    assert len(first.sha256()) == 64


def test_the_descriptor_carries_every_identity_clause_5_1_names():
    """§5.1's first and sixth bullets."""
    plan = _channel_plan(direct_builder="bcir.runtime.channel")
    text = plan.serialize().decode()
    assert f"plan-version {PLAN_VERSION}" in text
    assert f"compiler {PLAN_COMPILER}" in text
    assert "module Channel" in text and "type ChannelDescriptor" in text
    assert f"family {FAMILY}" in text and f"profile {PROFILE_CANONICAL}" in text
    assert "direct-builder bcir.runtime.channel" in text
    assert plan.source_sha256 == hashlib.sha256(CHANNEL_MODULE.encode()).hexdigest()


def test_the_source_hash_distinguishes_two_schemas_of_the_same_shape():
    """A plan names the schema it came from, not merely a structure: two modules can define
    the same shape and mean different things."""
    other = compile_plan(
        _channel(),
        module="Other",
        type_name="ChannelDescriptor",
        source=CHANNEL_MODULE.replace("Channel DEF", "Other DEF"),
    )
    assert other.source_sha256 != _channel_plan().source_sha256
    assert other.sha256() != _channel_plan().sha256()


def test_the_profile_is_part_of_the_descriptor():
    """§5.3's family/profile naming. A BASIC plan and a canonical plan are different
    descriptors, because they accept different bytes."""
    assert _channel_plan().profile == PROFILE_CANONICAL
    assert _channel_plan(rules=JerRules.BASIC).profile == PROFILE_BASIC
    assert _channel_plan().sha256() != _channel_plan(rules=JerRules.BASIC).sha256()


def test_a_serialized_descriptor_holds_no_process_pointer():
    """§5.1 — "Descriptors are data; they contain no process pointers or executable
    callbacks when serialized."

    The plan does keep a live type reference so `decode_with_plan` can run; what matters is
    that it never reaches the bytes.
    """
    text = _channel_plan().serialize().decode()
    for leak in ("object at 0x", "<bound method", "0x7f", "function "):
        assert leak not in text, leak


# --- gate 2: unsupported schema and instruction refusal (fail closed) --------------------


def test_an_enumerated_without_an_enumeration_is_refused_at_compile_time():
    """§5.1 — "Unknown required descriptor features fail closed."

    X.697 §22.2 encodes the *identifier*, so a bare ENUMERATED has no JER spelling at all.
    Catching it when the plan is built rather than when a value arrives is the difference
    between a schema fault and a runtime surprise.
    """
    kind = Sequence((Component("e", Primitive(Universal.ENUMERATED, "ENUMERATED")),))
    _refuses(lambda: compile_plan(kind, module="M", type_name="T"), "22.2")


def test_an_open_type_is_refused_because_a_static_plan_cannot_name_it():
    """§41 encodes an open type AS its contained type, chosen by a sibling's value at decode
    time, and JER has no hexadecimal fallback the way XER's §8.5 does."""
    inner = Sequence((Component("n", Primitive(Universal.INTEGER, "INTEGER")),), "Inner")
    table = ObjectSetTable("C", ({"&id": 1, "&Type": inner},))
    opened = OpenType(
        "OPEN", table=table, field="&Type", governing=(("id",),), governing_fields=("&id",)
    )
    kind = Sequence(
        (Component("id", Primitive(Universal.INTEGER, "INTEGER")), Component("body", opened))
    )
    _refuses(lambda: compile_plan(kind, module="M", type_name="T"), "41 encodes an open")


def test_two_members_sharing_a_json_name_after_a_rename_are_refused():
    """§16.2 — the final set of names "shall not contain two identical strings". A dispatch
    table with a duplicate key is not a dispatch table."""
    integer = Primitive(Universal.INTEGER, "INTEGER")
    kind = Sequence((Component("a", integer), Component("b", integer, tag=0)))
    instructions = JerInstructions().assign(kind.components[1], Name("a"))
    _refuses(
        lambda: compile_plan(kind, module="M", type_name="T", instructions=instructions), "16.2"
    )


def test_an_unwrapped_choice_that_cannot_be_discriminated_is_refused_when_planned():
    """§19.2.2, moved forward to compile time.

    `jer.py` reports this at decode, which is right for a decoder — §6.6 makes the *final*
    instructions decide conformity. A plan is a contract, though, and a contract that can
    never be decoded is better refused when it is written than when it is used.
    """
    integer = Primitive(Universal.INTEGER, "INTEGER")
    kind = Choice((Component("i", integer, tag=0), Component("j", integer, tag=1)))
    _refuses(
        lambda: compile_plan(
            kind,
            module="M",
            type_name="T",
            instructions=JerInstructions().assign(kind, Unwrapped()),
        ),
        "19.2.2",
    )
    ok = Choice(
        (
            Component("i", integer, tag=0),
            Component("s", Primitive(Universal.UTF8_STRING, "UTF8String"), tag=1),
        )
    )
    plan = compile_plan(
        ok, module="M", type_name="T", instructions=JerInstructions().assign(ok, Unwrapped())
    )
    assert plan.root.kind == "unwrapped-choice"


# --- gate 3: the table-driven and direct traces agree ------------------------------------


def test_the_plan_driven_decode_and_the_direct_decode_agree_on_value_and_trace():
    """§5.2 — a wrapper "must produce the same event trace and diagnostics as the
    table-driven scalar implementation"."""
    plan = _channel_plan()
    value, trace = decode_with_plan(plan, encode_jer(_channel(), _CHANNEL_VALUE))
    assert value == _CHANNEL_VALUE
    assert trace == trace_of(plan, value)
    assert trace[0] == "enter . sequence" and trace[-1] == "leave ."
    # Members are visited in SCHEMA order, so the trace is stable under §27.3.3's freedom
    # about the order they appear in the JSON.
    assert [e.split()[1] for e in trace if e.startswith("member ")] == [
        "./name",
        "./kind",
        "./provenance",
        "./modeled",
        "./capabilities",
    ]


def test_the_trace_follows_the_value_not_the_schema_for_absent_members():
    kind = Sequence(
        (
            Component("x", Primitive(Universal.INTEGER, "INTEGER")),
            Component("y", Primitive(Universal.INTEGER, "INTEGER"), tag=0, optional=True),
        )
    )
    plan = compile_plan(kind, module="M", type_name="T")
    assert "member ./y" not in trace_of(plan, {"x": 1})
    assert "member ./y" in trace_of(plan, {"x": 1, "y": 2})


def test_a_plan_driven_decode_refuses_exactly_what_the_bounded_oracle_refuses():
    """Diagnostics agree too, not merely values — §5.2 asks for both."""
    plan = _channel_plan()
    for document in (
        b'{"kind":"gpu","name":"gpu0","capabilities":[]}',
        b'{"name": "gpu0","kind":"gpu","capabilities":[]}',
        b'{"name":"gpu0","kind":"gpu","capabilities":[]}\n',
    ):
        try:
            decode_with_plan(plan, document)
        except JerBoundedError as error:
            assert error.diagnostic.code is JerErrorCode.NOT_CANONICAL, document
        else:
            raise AssertionError(f"{document!r} decoded under the canonical profile")


def test_a_deserialized_descriptor_cannot_decode_by_itself():
    """§5.1 makes a descriptor data, and data carries no type model, so a plan that has lost
    its live reference says so rather than half-working."""
    plan = _channel_plan()
    import dataclasses

    _refuses(
        lambda: decode_with_plan(dataclasses.replace(plan, _kind=None), b"{}"), "descriptor is data"
    )


# --- §5.1: dispatch, metadata, bounds ----------------------------------------------------


def test_the_dispatch_table_is_sorted_and_complete():
    """A C twin binary-searches it, so it is sorted by name; `members` stays in schema
    order separately, because a decoded value keys by the schema's names."""
    plan = _channel_plan()
    names = [name for name, _index in plan.root.dispatch]
    assert names == sorted(names)
    assert names == sorted(m.name for m in plan.root.members)
    for name, index in plan.root.dispatch:
        assert plan.root.members[index].name == name


def test_required_default_and_extension_metadata_is_recorded_per_member():
    by_name = {m.name: m for m in _channel_plan().root.members}
    assert by_name["name"].required and not by_name["name"].has_default
    assert by_name["provenance"].has_default and not by_name["provenance"].required
    assert by_name["modeled"].has_default and by_name["capabilities"].required
    assert _channel_plan().root.duplicate_policy == "refuse"


def test_recursion_bounds_are_derived():
    """§5.1's "recursion bounds" — the deepest JSON nesting a value can reach."""
    # object -> capabilities array -> the enumerated string inside it.
    assert _channel_plan().root.max_depth == 3
    flat = compile_plan(
        Sequence((Component("x", Primitive(Universal.INTEGER, "INTEGER")),)),
        module="M",
        type_name="T",
    )
    assert flat.root.max_depth == 2


def test_almost_nothing_is_statically_bounded_and_that_is_what_jer_is():
    """§5.1 asks for capacity bounds "where statically derivable". For JER that is four
    things, and the reason is X.697's §7.2.2 constraint blindness.

    `INTEGER (0..255)` and `OCTET STRING (SIZE (4))` are bounded in every binary rail in
    this repo and unbounded here, because §7.2.2 l) and h) hide those constraints from a JER
    encoder. A compiler that reported a bound anyway would be sizing a buffer from a
    constraint the encoder is forbidden to read, so `None` is the correct answer — and it is
    why J3's C interface must take its capacity from the caller.
    """
    kind = Sequence(
        (
            Component("i", Primitive(Universal.INTEGER, "INTEGER", ValueRange(0, 255))),
            Component(
                "o",
                Primitive(Universal.OCTET_STRING, "OCTET STRING", Size(ValueRange(4, 4))),
                tag=0,
            ),
            Component(
                "b", Primitive(Universal.BIT_STRING, "BIT STRING", Size(ValueRange(10, 10))), tag=1
            ),
            Component("f", Primitive(Universal.BOOLEAN, "BOOLEAN"), tag=2),
            Component("n", Primitive(Universal.NULL, "NULL"), tag=3),
        )
    )
    bounds = {
        m.name: m.node.bounded_octets
        for m in compile_plan(kind, module="M", type_name="T").root.members
    }
    assert bounds["i"] is None, "7.2.2 l) hides an integer's value constraint"
    assert bounds["o"] is None, "7.2.2 h) hides a SIZE on an octet string"
    assert bounds["b"] == 2 + 2 * 2, "7.2.1 a) is the one SIZE JER can see"
    assert bounds["f"] == 5 and bounds["n"] == 4
    # A container is bounded only if every member is, so this one is not.
    assert compile_plan(kind, module="M", type_name="T").root.bounded_octets is None
    assert (
        compile_plan(
            Sequence((Component("f", Primitive(Universal.BOOLEAN, "BOOLEAN")),)),
            module="M",
            type_name="T",
        ).root.bounded_octets
        is not None
    )


def test_a_sequence_of_is_never_bounded_because_jer_cannot_see_its_size():
    plan = compile_plan(
        SequenceOf(Primitive(Universal.BOOLEAN, "BOOLEAN")), module="M", type_name="T"
    )
    assert plan.root.bounded_octets is None
    assert plan.root.element.bounded_octets == 5


# --- §5.1: instruction compilation and its hash -------------------------------------------


def test_instructions_are_resolved_into_the_plan_and_hashed():
    kind = _channel()
    instructions = (
        JerInstructions()
        .assign(kind.components[0], Name(NameKeyword.UPPERCASED))
        .assign(kind, Array())
    )
    renamed = _channel_plan(kind, instructions=instructions)
    assert renamed.instruction_hash != _channel_plan().instruction_hash
    assert renamed.sha256() != _channel_plan().sha256()
    # The rename is resolved at compile time, so a decoder matches the wire name directly
    # while the value still keys by the schema's identifier.
    assert renamed.root.members[0].name == "NAME"
    assert renamed.root.members[0].identifier == "name"
    assert renamed.root.kind == "array" and "ARRAY" in renamed.root.instructions


def test_the_instruction_hash_is_over_the_resolved_set_not_the_assignment_order():
    """§13.3 makes a later assignment of a category REPLACE an earlier one, and §7.5.4 says
    the order in which instructions join the set "is not significant". Hashing the resolved
    set is what makes both true of the descriptor as well.
    """
    kind = _channel()
    once = JerInstructions().assign(kind.components[0], Name("label"))
    twice = (
        JerInstructions()
        .assign(kind.components[0], Name("scratch"))
        .assign(kind.components[0], Name("label"))
    )
    assert (
        _channel_plan(kind, instructions=once).instruction_hash
        == _channel_plan(kind, instructions=twice).instruction_hash
    )
    assert (
        _channel_plan(kind, instructions=once).serialize()
        == _channel_plan(kind, instructions=twice).serialize()
    )


def test_a_plan_with_instructions_still_round_trips_through_its_own_decode():
    kind = _channel()
    instructions = (
        JerInstructions()
        .assign(kind.components[0], Name(NameKeyword.UPPERCASED))
        .assign(kind, Array())
    )
    plan = _channel_plan(kind, instructions=instructions)
    document = encode_jer(kind, _CHANNEL_VALUE, instructions=instructions)
    value, trace = decode_with_plan(plan, document)
    assert value == _CHANNEL_VALUE and trace == trace_of(plan, value)
    assert document.startswith(b"[")  # §27.2, the ARRAY form


def test_the_channel_schema_describes_the_fields_the_c_reader_uses():
    """§5.4's first integration target, cross-checked against `runtime/c/bcir_channel.c`:
    it accepts `name`, `kind`, `provenance`, `modeled` and `capabilities`, with six device
    kinds and three provenances. A plan naming a different set describes a different file.
    """
    plan = _channel_plan()
    by_name = {m.name: m for m in plan.root.members}
    assert set(by_name) == {"name", "kind", "provenance", "modeled", "capabilities"}
    assert by_name["kind"].node.enumeration == (
        "cpu",
        "gpu",
        "fpga",
        "accelerator",
        "storage",
        "memory",
    )
    assert by_name["provenance"].node.enumeration == ("real", "modeled", "simulated")
    assert by_name["capabilities"].node.kind == "sequence-of"
    assert by_name["capabilities"].node.element.enumeration[0] == "universal"
    assert by_name["modeled"].node.kind == "boolean"
