"""X.692 part three: the defined syntax read from text, and where an ECN encoding belongs.

Two things are checked here, and they are the same question from two sides.

The first is that `bcir/asn1/BCIR-FrameHeader.ecn` — the §6 gate's workload written in the
notation X.692 defines — produces the octets the Python-assembled objects produce. That is a
second opinion rather than a second spelling: the parser walks clause 23's bracket grammar and
resolves clause 11's class assignments, and if it and `legacy_frame_objects()` ever disagree,
one of them has misread the specification.

The second is the **plan-v6 question**: whether an ECN encoding is a sixth column in
`encode_plan`, carried by a version 6 of that descriptor. The answer is no, and the tests below
are the evidence rather than the assertion — an `EncodePlan` compiled from the frame header's
ASN.1 type has no slot for the wire order or for the `reserved` bits, because both are
properties of an encoding structure and neither is a property of a type.
"""

from bcir.asn1.ecn_syntax import (
    EcnModule, frame_header_module, frame_header_source, parse_module, tokenize,
)
from bcir.asn1.ecn_user import (
    HandleValueKind, IntSpec, PadSpec, encode_with_user, legacy_frame_objects,
    legacy_frame_workload,
)
from bcir.asn1.ecn_param import ParameterKind
from bcir.asn1.encode_plan import compile_encode_plan
from bcir.asn1.tags import Asn1Error


def _encode_from_text(module: EcnModule, value: dict) -> bytes:
    return encode_with_user(module.object_set(), "#Frame-header", value, outer=module.outer())


# --- the round trip -------------------------------------------------------------------------

def test_the_written_specification_produces_the_octets_the_assembled_objects_do():
    """The gate's workload, from text, byte for byte.

    `legacy_frame_workload()` states the target independently of either path — it is the
    layout a real header uses, with the expected octets derived by hand in its docstring — so
    this is three-way agreement rather than two implementations agreeing with each other.
    """
    from bcir.asn1.ecn import CONCATENATION

    _kind, value, expected = legacy_frame_workload()
    assert _encode_from_text(frame_header_module(), value) == expected
    assert encode_with_user(legacy_frame_objects(), CONCATENATION, value,
                            outer=frame_header_module().outer()) == expected


def test_the_transform_survives_the_round_trip_and_is_not_a_constant():
    """A scaled length is only a transform if it moves with the value.

    Encoding one value correctly can be luck — a 4-bit field holding 1010 is also what a
    plain `10` would write. Three values pin the `divide:4`.
    """
    module = frame_header_module()
    for payload, nibble in ((0, 0b0000), (40, 0b1010), (60, 0b1111)):
        octets = _encode_from_text(
            module, {"version": 0, "urgent": False, "payloadOctets": payload})
        assert octets[0] >> 4 == nibble, (payload, octets.hex())


def test_the_module_hashes_and_the_hash_moves_only_when_the_encoding_does():
    """A specification with no canonical form cannot name an artifact. This is that form.

    Reordering the *source* must not change the digest where the order carries no meaning,
    and changing a width must change it. Both directions are checked, because a digest that
    only ever changes is as useless as one that never does.
    """
    module = frame_header_module()
    digest = module.sha256()
    assert len(digest) == 64
    assert frame_header_module().sha256() == digest

    widened = parse_module(frame_header_source().replace(
        "ENCODING-SPACE SIZE 3 MULTIPLE OF bit", "ENCODING-SPACE SIZE 4 MULTIPLE OF bit"))
    assert widened.sha256() != digest

    # Whitespace and comments are not the specification.
    respaced = parse_module(
        "\n".join(line for line in frame_header_source().split("\n")
                  if not line.strip().startswith("--")))
    assert respaced.sha256() == digest


def test_the_handle_notation_parses_and_reaches_the_digest():
    """§22.9.1.2's `[EXHIBITS HANDLE &exhibited-handle AT &Handle-positions
    [AS &handle-value-set]]`, with §21.16.1's six value-set alternatives.

    The digest half is the point. A handle changes what a *decoder* reads and nothing an
    encoder writes, so a serialization that skipped it would give two genuinely different
    specifications one name — which is the failure mode a content-addressed descriptor exists
    to prevent. `SYNTAX_VERSION` moved to 4 for exactly this reason, and to 5 when Annex C's
    parameterized assignments joined it for the same one.
    """
    # §23.7.1's WITH SYNTAX puts EXHIBITS HANDLE after the value encoding and before
    # BIT-REVERSAL, following §23.3.3.1's encoder-action order. `plainVersion` is the only
    # object whose body ends with `ENCODING positive-int`, so this substitution is unique.
    def with_handle(clause: str):
        return parse_module(frame_header_source().replace(
            "ENCODING positive-int\n  }", f"ENCODING positive-int\n      {clause}\n  }}"))

    for clause, kind in (
        ("EXHIBITS HANDLE kind AT { 0, 1 } AS bits:'01'B", HandleValueKind.BITS),
        ("EXHIBITS HANDLE kind AT { 0, 1 } AS number:2", HandleValueKind.NUMBER),
        ("EXHIBITS HANDLE kind AT { 0, 1 } AS range:{0, 1}", HandleValueKind.RANGE),
        ("EXHIBITS HANDLE kind AT { 0, 1 } AS ranges:{{0, 0}, {3, 3}}",
         HandleValueKind.RANGES),
        ("EXHIBITS HANDLE kind AT { 0, 1 }", HandleValueKind.TAG_ANY),
    ):
        module = with_handle(clause)
        handle = module.conditional_int("plainVersion").spec.exhibits
        assert handle is not None and handle.name == "kind", clause
        assert handle.positions == (0, 1), clause
        assert handle.value_set.kind is kind, clause
        assert module.sha256() != frame_header_module().sha256(), clause

    # Two different value sets are two different specifications.
    assert (with_handle("EXHIBITS HANDLE kind AT { 0, 1 } AS range:{0, 1}").sha256()
            != with_handle("EXHIBITS HANDLE kind AT { 0, 1 } AS range:{2, 3}").sha256())
    # §22.9.1.6 makes the positions a SET, so writing them backwards is the same handle.
    assert (with_handle("EXHIBITS HANDLE kind AT { 1, 0 } AS number:2").sha256()
            == with_handle("EXHIBITS HANDLE kind AT { 0, 1 } AS number:2").sha256())


def test_a_handle_value_set_the_choice_does_not_define_is_refused():
    """§21.16.1 has six alternatives, and a seventh spelling would otherwise become
    `tag:any` by default — a set that matches nothing and says so only at write time."""
    try:
        parse_module(frame_header_source().replace(
            "ENCODING positive-int\n  }",
            "ENCODING positive-int\n      "
            "EXHIBITS HANDLE kind AT { 0, 1 } AS integer:3\n  }"))
    except Asn1Error as error:
        assert "21.16.1" in str(error), error
    else:
        raise AssertionError("an unknown HandleValueSet alternative was accepted")


# --- the grammar itself ---------------------------------------------------------------------

def test_a_comment_ends_at_the_next_pair_of_hyphens_as_x680_says():
    """X.680 §12.6: a comment terminates "with a pair of hyphens or at the end of the line".

    Written as a test because it is the kind of rule an implementation quietly gets wrong in
    the permissive direction, and a comment that swallows the rest of a line would silently
    drop an encoding property.
    """
    assert [token.text for token in tokenize("a -- note -- b")] == ["a", "b"]
    assert [token.text for token in tokenize("a -- note\nb")] == ["a", "b"]


def test_a_value_notation_and_its_alternative_are_one_token():
    """`bits:'0'B` is one `Pattern` value, not a name and a string."""
    assert [token.text for token in tokenize("TRUE-PATTERN bits:'0'B")] == [
        "TRUE-PATTERN", "bits:'0'B"]
    assert [token.text for token in tokenize("{ a, b }")] == ["{", "a", ",", "b", "}"]


def test_the_defined_syntax_is_read_in_the_order_the_with_syntax_gives_it():
    """§20.5's bracket structure is load-bearing, so keywords are not order-free.

    `MULTIPLE OF` is nested inside `SIZE`, which is nested inside `ENCODING-SPACE`, in every
    clause 23 `WITH SYNTAX`. A parser that accepted properties in any order would accept
    specifications the notation does not admit — and then have to invent what they mean. Here
    the two are swapped, and the refusal names the keyword that was expected first.
    """
    source = frame_header_source().replace(
        "ENCODING-SPACE SIZE 3 MULTIPLE OF bit",
        "MULTIPLE OF bit ENCODING-SPACE SIZE 3")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "ENCODING-SPACE" in str(error), error
    else:
        raise AssertionError("properties were accepted out of the syntax's order")


def test_an_unimplemented_property_group_is_refused_by_name_and_never_skipped():
    """The failure a permissive parser causes is silent wrong octets, not a crash.

    Each group below is one this repository has not built. Accepting and ignoring any of them
    would produce an encoding that does not match the specification it was handed, which is
    exactly the class of defect the triple-rail design exists to catch — so they are
    recognized, cited and refused.
    """
    space = "ENCODING-SPACE SIZE 3 MULTIPLE OF bit"
    cases = [
        # Groups this repository has not built. Each names what it would need.
        (space, "REPLACE STRUCTURE WITH #Repl", "22.1.2"),
        (space, f"{space} CONTAINED BY x", "22.11"),
        # §23.1 and §23.11's OBJECTS are built; their STRUCTURE notation is not, and the
        # refusal says which half is missing rather than which clause exists. §16.2.12 fixes
        # which clause each half is: `AlternativesStructure` is §16.3 and
        # `ConcatenationStructure` — whose `ConcatComponentPresence` carries the optional
        # marker — is §16.5. These two expectations were the other way round until Annex C was
        # read for slice F, so they are asserted here rather than left to the comment.
        (space, f"{space} ALTERNATIVE DETERMINED BY handle", "16.3"),
        (space, f"{space} PRESENCE DETERMINED BY field-to-be-set USING p", "16.5"),
        # Groups that ARE built, written in a way the clause forbids. These are the more
        # interesting half: a parser that only refused what it had not implemented would
        # accept every one of them. `DETERMINED BY container USING x` used to be on the list
        # above; §21.3.6 is built, so it parses, and naming a field that is not an open
        # container is a write-time refusal instead — see test_asn1_ecn_containers.py.
        (space, f"{space} DETERMINED BY field-to-be-set", "21.3.4"),
        (space, f"ALIGNED TO ANY octet {space}", "22.2.2.2"),
        ("ENCODING positive-int",
         "ENCODING positive-int BIT-REVERSAL reverse-half-units", "22.12.2.3"),
        ("ENCODING positive-int",
         "ENCODING positive-int BIT-REVERSAL sideways", "21.14.1"),
    ]
    for target, clause_text, citation in cases:
        source = frame_header_source().replace(target, clause_text)
        assert source != frame_header_source(), target
        try:
            parse_module(source)
        except Asn1Error as error:
            assert citation in str(error), (clause_text, str(error))
        else:
            raise AssertionError(f"{clause_text!r} was accepted and ignored")


def test_an_integer_object_goes_through_conditional_int_because_the_clause_says_so():
    """§23.6.1 gives `#INT` two properties, and neither is an encoding space.

    An `#INT` object names `#CONDITIONAL-INT` objects; §23.7.1 is where the space, the
    transforms and the value encoding live. Letting `#INT` carry them directly would be
    inventing syntax, which is the thing the citation pass was for.
    """
    source = frame_header_source().replace(
        "versionField #Version ::= { ENCODING plainVersion }",
        "versionField #Version ::= { ENCODING-SPACE SIZE 3 MULTIPLE OF bit }")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "23.6.2.2" in str(error), error
    else:
        raise AssertionError("#INT carried an encoding space of its own")


def test_one_object_per_class_is_enforced_where_the_set_is_formed_not_where_text_is_read():
    """§9.5.2 governs "encoding object set construction", and a module is not a set.

    The frame header's module holds two `#CONDITIONAL-INT` objects, which is legal because
    neither enters the set — §23.6.1 reaches them by name from an `#INT` object. Two objects
    for a class the *structure* names is the violation, and that is what is refused.
    """
    module = frame_header_module()
    conditionals = [name for name, (cls, _spec) in module.objects.items()
                    if cls == "#CONDITIONAL-INT"]
    assert len(conditionals) == 2, conditionals
    assert "#CONDITIONAL-INT" not in module.object_set()

    source = frame_header_source().replace(
        "  pduOuter #OUTER ::= {",
        "  secondVersion #Version ::= { ENCODING plainVersion }\n  pduOuter #OUTER ::= {")
    try:
        parse_module(source).object_set()
    except Asn1Error as error:
        assert "9.5.2" in str(error), error
    else:
        raise AssertionError("a class in the structure carried two objects")


def test_a_transmission_order_that_differs_from_the_type_comes_from_the_structure():
    """§22.10.1.1's `&concatenation-order` is `{textual, tag, random}` and nothing else.

    So the wire order is not a property of the concatenation object — §22.10.3.1 reads
    `textual` from "the ASN.1 type specification **or the ECN structure definition**". Moving
    the field in the §16.5 structure moves it on the wire; the object does not change.
    """
    module = frame_header_module()
    assert module.concatenation().transmission_order() == (
        "payloadOctets", "version", "urgent", "reserved")

    reordered = parse_module(frame_header_source().replace(
        "      payloadOctets  #Scaled-length,\n      version        #Version,",
        "      version        #Version,\n      payloadOctets  #Scaled-length,"))
    assert reordered.concatenation().transmission_order() == (
        "version", "payloadOctets", "urgent", "reserved")
    _kind, value, expected = legacy_frame_workload()
    assert _encode_from_text(reordered, value) != expected


def test_order_tag_and_order_random_are_refused_with_what_they_would_require():
    for order, citation in (("tag", "22.10.2.4"), ("random", "22.10.2.1")):
        source = frame_header_source().replace(
            "CONCATENATION ORDER textual", f"CONCATENATION ORDER {order}")
        try:
            parse_module(source)
        except Asn1Error as error:
            assert citation in str(error), (order, str(error))
        else:
            raise AssertionError(f"ORDER {order} was accepted without its prerequisites")


def test_a_nested_encoding_structure_is_refused_rather_than_flattened():
    """§16.2.1 admits a field that is itself a structure; this rail walks one flat level.

    Flattening it would put a nested structure's fields into its parent's transmission order,
    which is a different encoding from the one written.
    """
    source = frame_header_source().replace(
        "      reserved       #Reserved",
        "      reserved       #Reserved { inner #Reserved }")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "16.2.1" in str(error), error
    else:
        raise AssertionError("a nested structure was flattened into its parent")


# --- the plan-v6 question --------------------------------------------------------------------

def test_the_write_plan_holds_the_type_and_therefore_cannot_hold_this_encoding():
    """**The plan-v6 answer, as evidence.**

    `encode_plan` describes an ASN.1 *type* — tag, members, enumeration, constraint,
    extensibility — and five emitters read the same node and apply their own rule. The
    question was whether an ECN encoding is a sixth such rule, carried by a version 6 of that
    descriptor.

    It is not, and the frame header shows why twice over. Its wire order puts `payloadOctets`
    first, where the plan's members are in the schema's order and `EncodeMember.index` is that
    order; and its `reserved` bits correspond to no ASN.1 component, so there is no member for
    them to be a property of. Both facts belong to a §16.5 encoding structure. Carrying them
    on `EncodeNode` would make a node's meaning depend on which candidate read it, which is
    the one thing that plan's design rules out.

    So an ECN encoding is a *third compilation* of the same schema, with its own version
    counter — the same argument `encode_plan`'s docstring already makes for why a write plan
    is not a read plan.
    """
    kind, _value, _expected = legacy_frame_workload()
    plan = compile_encode_plan(kind, module="FrameEncodings", type_name="FrameHeader")
    module = frame_header_module()

    schema_order = tuple(member.name for member in plan.root.members)
    wire_order = module.concatenation().transmission_order()
    assert schema_order == ("version", "urgent", "payloadOctets")
    assert wire_order[:3] != schema_order

    assert "reserved" not in schema_order
    assert "reserved" in wire_order
    assert isinstance(module.concatenation().fields["reserved"], PadSpec)

    # And the plan is the same plan whichever candidate reads it, which is the property that
    # leaves the ECN facts with nowhere in it to live.
    assert b"reserved" not in plan.serialize()
    assert plan.sha256() == compile_encode_plan(
        kind, module="FrameEncodings", type_name="FrameHeader").sha256()


def test_the_two_descriptors_version_independently():
    """A third compilation gets a third counter, and the digests are not interchangeable."""
    from bcir.asn1.ecn_syntax import SYNTAX_VERSION
    from bcir.asn1.encode_plan import PLAN_VERSION

    kind, _value, _expected = legacy_frame_workload()
    plan = compile_encode_plan(kind, module="FrameEncodings", type_name="FrameHeader")
    module = frame_header_module()
    assert plan.serialize().startswith(f"plan-version {PLAN_VERSION}".encode())
    assert module.serialize().startswith(f"ecn-syntax-version {SYNTAX_VERSION}".encode())
    assert module.sha256() != plan.sha256()


def test_the_transform_is_still_what_no_fixed_candidate_reproduces():
    """The gate's reopening condition, restated against the text-driven path.

    `test_asn1_ecn_user.py` runs this against the Python-assembled objects. Running it here
    too means the *specification* is what no candidate matches, not one particular way of
    building it — so the day a candidate does match, both fail rather than one.
    """
    from bcir.asn1.ecn_user import refuted_by

    kind, value, expected = legacy_frame_workload()
    assert _encode_from_text(frame_header_module(), value) == expected
    for name, (octets, note) in refuted_by(kind, value, expected).items():
        assert octets != expected, (name, note)


# --- refusals that keep the parser honest -----------------------------------------------------

def test_a_module_without_an_end_is_refused():
    try:
        parse_module("M ENCODING-DEFINITIONS ::= BEGIN")
    except Asn1Error as error:
        assert "END" in str(error), error
    else:
        raise AssertionError("a truncated module parsed")


def test_a_class_this_rail_cannot_execute_is_named_rather_than_ignored():
    source = frame_header_source().replace("#Reserved      ::= #PAD",
                                           "#Reserved      ::= #BITS")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "#BITS" in str(error), error
    else:
        raise AssertionError("an unimplemented encoding class parsed")


def test_a_transform_reference_that_names_nothing_is_refused():
    source = frame_header_source().replace("TRANSFORMS { octetsToUnits }",
                                           "TRANSFORMS { noSuchTransform }")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "24.2.4.1" in str(error), error
    else:
        raise AssertionError("a dangling transform reference parsed")


def test_only_one_transform_clause_may_be_used_per_object():
    """§24.1.1's WITH SYNTAX carries the comment "Only one of the following clauses can be
    used", and a second one silently winning would make the first dead text."""
    source = frame_header_source().replace(
        "{ INT-TO-INT divide:4 }", "{ INT-TO-INT divide:4 INT-TO-BITS SIZE 4 }")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "24.1.1" in str(error), error
    else:
        raise AssertionError("two transform clauses parsed into one object")


def test_the_operand_spellings_are_the_clauses_own():
    """§24.3.1 gives `negate` and `subtract` ENUMERATED operands, not integers."""
    for body, wanted in (("INT-TO-INT negate:0", "negate:value"),
                         ("INT-TO-INT subtract:4", "subtract:lower-bound")):
        source = frame_header_source().replace("INT-TO-INT divide:4", body)
        try:
            parse_module(source)
        except Asn1Error as error:
            assert wanted in str(error), (body, str(error))
        else:
            raise AssertionError(f"{body!r} parsed with the wrong operand form")
    assert parse_module(frame_header_source().replace(
        "INT-TO-INT divide:4", "INT-TO-INT negate:value")).sha256()


def test_a_self_delimiting_encoding_space_is_refused_with_what_it_would_need():
    """§21.2.2's DEFAULT is `self-delimiting-values`, which §21.2.7 defines by matching every
    candidate encoding rather than by a width. Defaulting to it silently would make an object
    that states no SIZE encode nothing at all."""
    source = frame_header_source().replace("ENCODING-SPACE SIZE 3 MULTIPLE OF bit",
                                           "ENCODING-SPACE")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "21.2" in str(error), error
    else:
        raise AssertionError("an encoding space with no stated size parsed")


# --- clause 24's nineteen transforms, from text ---------------------------------------------

_ALL_TRANSFORMS = """
Transforms ENCODING-DEFINITIONS ::= BEGIN
  t01 #TRANSFORM ::= { INT-TO-INT divide:4 }
  t02 #TRANSFORM ::= { BOOL-TO-BOOL AS logical:not }
  t03 #TRANSFORM ::= { BOOL-TO-INT AS true-zero }
  t04 #TRANSFORM ::= { INT-TO-BOOL TRUE-IS { 7 } FALSE-IS { 0 } }
  t05 #TRANSFORM ::= { INT-TO-CHARS SIZE 4 PLUS-SIGN TRUE PADDING spaces }
  t06 #TRANSFORM ::= { INT-TO-BITS AS positive-int SIZE 1 MULTIPLE OF octet }
  t07 #TRANSFORM ::= { BITS-TO-INT AS positive-int }
  t08 #TRANSFORM ::= { CHAR-TO-BITS AS mapped CHAR-LIST { "a", "b" } BITS-LIST { '0'B, '1'B } }
  t09 #TRANSFORM ::= { BITS-TO-CHAR AS iso10646 }
  t10 #TRANSFORM ::= { BIT-TO-BITS ZERO-PATTERN bits:'00'B ONE-PATTERN bits:'11'B }
  t11 #TRANSFORM ::= { BITS-TO-BITS SOURCE-LIST { '0'B, '1'B } RESULT-LIST { '1'B, '0'B } }
  t12 #TRANSFORM ::= { CHARS-TO-COMPOSITE-CHAR }
  t13 #TRANSFORM ::= { BITS-TO-COMPOSITE-BITS UNIT octet }
  t14 #TRANSFORM ::= { OCTETS-TO-COMPOSITE-BITS }
  t15 #TRANSFORM ::= { COMPOSITE-CHAR-TO-CHARS }
  t16 #TRANSFORM ::= { COMPOSITE-BITS-TO-BITS }
  t17 #TRANSFORM ::= { COMPOSITE-BITS-TO-OCTETS }
END
"""


def test_every_transform_clause_24_defines_is_reachable_from_the_notation():
    """A transform the model can execute but the notation cannot express is half-built.

    §24.1.1's `WITH SYNTAX` lists the clauses; each one here is a separate branch of the
    parser, and this is what stops a branch from being written and never exercised.
    """
    module = parse_module(_ALL_TRANSFORMS)
    assert len(module.transforms) == 17
    assert module.transforms["t03"].apply(True) == 0        # BOOL-TO-INT AS true-zero
    assert module.transforms["t05"].apply(7) == "  +7"      # INT-TO-CHARS
    assert module.transforms["t10"].apply(1) == (1, 1)      # BIT-TO-BITS


def test_every_transform_has_a_canonical_serialization():
    """The digest is the point of the surface module, so a transform with no serialized form
    silently drops out of it — and two specifications differing only there would hash alike."""
    module = parse_module(_ALL_TRANSFORMS)
    text = module.serialize().decode()
    for name in module.transforms:
        assert f"transform {name} " in text, name
    assert len(module.sha256()) == 64

    # And the digest moves when a transform's properties do.
    other = parse_module(_ALL_TRANSFORMS.replace("AS true-zero", "AS true-one"))
    assert other.sha256() != module.sha256()


def test_a_transform_clause_the_notation_does_not_define_is_refused_by_name():
    """§24.1.1 defines nineteen and all of them are read, so an unknown keyword is a typo
    rather than an unimplemented feature — and the message says so."""
    try:
        parse_module(_ALL_TRANSFORMS.replace("INT-TO-INT divide:4", "INT-TO-FROBNICATE 4"))
    except Asn1Error as error:
        assert "24.1.1" in str(error) and "nineteen" in str(error), error
    else:
        raise AssertionError("an undefined transform clause parsed")


# --- Annex C's parameterized assignments, read from module text ---------------------------

def _with_assignments(*lines: str) -> str:
    """The gate's module with extra assignments before `END`."""
    return frame_header_source().replace("END", "\n".join(lines) + "\nEND")


_LP_STRUCTURE = "#Length-prefixed{<#D>} ::= #CONCATENATION { length #INT, value #D }"


def test_ecn_writes_a_parameter_list_with_two_character_brackets():
    """Annex C.1 rewrites X.683 §8.3's `ParameterList` to `"{<" Parameter "," + ">}"`, and C.4
    does the same to the actual list.

    A parser that reuses X.683's would accept `{#D}` and reject `{<#D>}` — the only spelling
    ECN admits — while citing X.683 correctly throughout. So the brackets are lexed as single
    tokens, and the token stream is asserted directly: `{<` must not decompose into `{`, and
    `>}` must not glue itself to the dummy before it.
    """
    assert [t.text for t in tokenize("#L{<#D>} ::= X")] == [
        "#L", "{<", "#D", ">}", "::=", "X"]
    # X.683's own brackets are still just braces, so the wrong spelling parses as something
    # else entirely rather than as a parameter list — which is exactly why it goes unnoticed.
    assert "{<" not in [t.text for t in tokenize("#L{#D} ::= X")]

    module = parse_module(_with_assignments(_LP_STRUCTURE))
    assignment = module.parameterized["#Length-prefixed"]
    assert assignment.parameters.render() == "{<#D>}"
    assert assignment.body == ("#CONCATENATION", "{", "length", "#INT", ",", "value", "#D", "}")
    # §16.2.12's EncodingStructureDefn is a class AND a braced field list, so a body that
    # stopped at the class would leave the braces to be misread as the next assignment.
    assert module.structure_name == "FrameHeader-structure"


def test_a_governor_may_use_a_dummy_declared_to_its_left_but_only_for_an_object():
    """C.2 modifies X.683 §8.4 so that in a `ParameterizedEncodingObjectAssignment` "the scope
    extends to the `DefinedOrBuiltinEncodingClass` which **precedes** the `::=`", and C.2's
    NOTE gives the shape that needs it. §22.1.2.4 then *requires* that shape: the `ENCODED BY`
    object's governor is the `WITH` structure "instantiated with `#D`".

    So the two clauses are a matched pair — §22.1.2.4 is unwritable without C.2's extension —
    and this is the test that reads one written out in full.
    """
    module = parse_module(_with_assignments(
        _LP_STRUCTURE,
        "lp-object{<#D>} #Length-prefixed{<#D>} ::= { PLACEHOLDER }"))
    encoded_by = module.parameterized["lp-object"]
    assert encoded_by.governor == "#Length-prefixed"
    assert encoded_by.governor_actuals.render() == "{<#D>}"


def test_a_replacement_pair_is_checked_as_it_is_declared_not_when_it_is_used():
    """§22.1.2.2 and §22.1.2.4 restrict the *definitions*, so both can be checked the moment
    both halves are present. Checking early matters because a module may define a pair and
    apply it from an ELM this rail never reads: a pair that could never be instantiated is
    invalid on its own terms whether or not anything here instantiates it."""
    def refused(citation, *lines):
        try:
            parse_module(_with_assignments(*lines))
        except Asn1Error as error:
            assert citation in str(error), (citation, str(error))
            return
        raise AssertionError(f"expected a refusal citing {citation}")

    # §22.1.2.4: the governor is the structure instantiated with the object's OWN dummy.
    refused("22.1.2.4", _LP_STRUCTURE,
            "lp{<#D>} #Length-prefixed{<#Other>} ::= { PLACEHOLDER }")
    # C.3's `{<>}` is a legal ParameterizedReference and still instantiates with nothing.
    refused("22.1.2.4", _LP_STRUCTURE, "lp{<#D>} #Length-prefixed{<>} ::= { PLACEHOLDER }")
    # §22.1.2.2: a single ENCODING CLASS parameter — an object set governor is not one.
    refused("22.1.2.2", "#Bad{<#ENCODINGS:s>} ::= #CONCATENATION { value #INT }",
            "o{<#D>} #Bad{<#D>} ::= { PLACEHOLDER }")
    # §22.1.2.5's biconditional, read off the object's own dummies: a REFERENCE parameter is
    # present exactly when the group has an INSERT AT HEAD, and this pair declares no head-end.
    refused("22.1.2.5", _LP_STRUCTURE,
            "lp{<#D, REFERENCE:at, REFERENCE:also>} #Length-prefixed{<#D>} ::= { P }")


def test_the_governor_forms_c1_admits_and_the_one_it_names_nowhere():
    """C.1's `Governor` is `EncodingClassFieldType | REFERENCE | DefinedOrBuiltinEncodingClass
    | #ENCODINGS | Type`, and its a)-d) rules assign a dummy kind to every one of those but
    `Type` — which is therefore writable and stands for nothing."""
    module = parse_module(_with_assignments(
        "#Any{<#D, REFERENCE:at, #ENCODINGS:objects, #INT:obj, #INT.&size:v>} ::= "
        "#CONCATENATION { value #INT }"))
    params = module.parameterized["#Any"].parameters
    assert params.names() == ("#D", "at", "objects", "obj", "v")
    # Only three of the five governors determine a kind; C.1 shares the other two.
    assert params.kinds() == (ParameterKind.ENCODING_CLASS, ParameterKind.IDENTIFIER,
                              ParameterKind.ENCODING_OBJECT_SET, None, None)

    for source, citation in (
            ("#Bad{<INTEGER:v>} ::= #CONCATENATION { value #INT }", "C.1's Governor"),
            ("#Bad{<#D>} ::= #CONCATENATION { value #INT }\n"
             "#Bad{<#E>} ::= #CONCATENATION { value #INT }", "assigned twice"),
            ("#Bad{<#D>} ::= #CONCATENATION { value #INT }\n"
             "#Worse{<#D", "has no `>}`")):
        try:
            parse_module(_with_assignments(source))
        except Asn1Error as error:
            assert citation in str(error), (citation, str(error))
        else:
            raise AssertionError(f"{source!r} was accepted")


def test_a_parameterized_assignment_reaches_the_digest_governors_and_all():
    """Same argument as the handle's, one clause over. A replacement structure changes what a
    *decoder* reads, so two modules differing only inside one describe different octets and a
    name they shared would be a name that meant two things. `SYNTAX_VERSION` moved to 5.

    Governors are in the serialization too: `render` gives the bare form §22.1.2.2 requires at
    a *use*, and a declaration carries governors that two otherwise-identical modules could
    differ in.
    """
    from bcir.asn1.ecn_syntax import SYNTAX_VERSION

    assert SYNTAX_VERSION == 5
    base = parse_module(_with_assignments(_LP_STRUCTURE))
    assert b"parameterized #Length-prefixed" in base.serialize()

    wider = parse_module(_with_assignments(
        "#Length-prefixed{<#D>} ::= #CONCATENATION { len16 #INT, value #D }"))
    assert base.sha256() != wider.sha256()

    governed = parse_module(_with_assignments(
        "#Length-prefixed{<#ENCODINGS:D>} ::= #CONCATENATION { length #INT, value #D }"))
    assert governed.sha256() != base.sha256()
    assert b"{<#ENCODINGS:D>}" in governed.serialize()


def test_replace_is_still_refused_but_for_a_smaller_reason_than_before():
    """The parameterized structures and objects §22.1 names now parse, and §22.1.2's
    restrictions on them are checked. What is left is the binding between an auxiliary field's
    encoding and the instantiated one (§22.1.2.6, §22.1.1.9) — so the refusal names that,
    rather than the parameterization it used to name."""
    space = "ENCODING-SPACE SIZE 3 MULTIPLE OF bit"
    source = frame_header_source().replace(space, f"{space} REPLACE STRUCTURE WITH #X")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "22.1.2.6" in str(error), str(error)
        assert "{<" in str(error), "the refusal should point at what now parses"
    else:
        raise AssertionError("REPLACE was accepted and ignored")
