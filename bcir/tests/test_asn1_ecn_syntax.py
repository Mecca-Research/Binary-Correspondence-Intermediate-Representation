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

from dataclasses import replace

from bcir.asn1.ecn_syntax import (
    EcnModule, frame_header_module, frame_header_source, parse_module, tokenize,
)
from bcir.asn1.ecn_user import (
    HandleValueKind, IntSpec, PadSpec, encode_with_user, legacy_frame_objects,
    legacy_frame_workload,
)
from bcir.asn1.ecn_param import ParameterKind
from bcir.asn1.ecn_user import (
    AlternativesSpec, OptionalityDetermination, OptionalSpec, ReplaceAction,
)
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


_CONTENTS = """M ENCODING-DEFINITIONS ::= BEGIN
  #Inner   ::= #INT
  #Payload ::= #OCTETS
  cond #CONDITIONAL-INT ::= { ELSE ENCODING-SPACE SIZE 8 MULTIPLE OF bit ENCODING positive-int }
  innerEnc #Inner ::= { ENCODING cond }
  innerSet #ENCODINGS ::= { innerEnc }
  rep #CONDITIONAL-REPETITION ::= { ELSE REPETITION-SPACE DETERMINED BY not-needed }
  payObj #Payload ::= { REPETITION-ENCODING rep %s }
END
"""


def test_an_encoding_object_set_is_assigned_by_union_or_by_builtin_name():
    """§18.1.1's `EncodingObjectSetAssignment`, which §22.11 needed before it could exist.

    Two forms, both in §18.1.1's `EncodingObjectSet`: §18.1.5's braced union of objects and
    other sets, and §18.2.1's seven built-in names. The parser has to tell an object-set
    assignment from an object assignment, and §18.1.2 says what does it — "the
    EncodingObjectSet notation is **governed by the reserved word #ENCODINGS**".
    """
    module = parse_module(_CONTENTS % "CONTENTS-ENCODING innerSet CONTAINING #Inner")
    assert sorted(module.object_sets["innerSet"]) == ["#Inner"]

    # §18.2.1's built-in names resolve to the sets X.690 and X.691 define.
    builtin = parse_module("""M ENCODING-DEFINITIONS ::= BEGIN
  derSet #ENCODINGS ::= DER
END
""")
    assert len(builtin.object_sets["derSet"]) > 1


def test_a_set_may_not_hold_two_objects_of_one_class():
    """§18.1.7: "Encoding objects forming an encoding object set shall all be of distinct
    encoding classes".

    The rule has teeth precisely where §22.11.2 hands the set to a contained type: two objects
    for one class make §9.5.2's lookup ambiguous at the point nothing could recover from it.
    """
    source = _CONTENTS.replace("innerSet #ENCODINGS ::= { innerEnc }",
                               "twin #Inner ::= { ENCODING cond }\n"
                               "  innerSet #ENCODINGS ::= { innerEnc | twin }")
    try:
        parse_module(source % "CONTENTS-ENCODING innerSet CONTAINING #Inner")
    except Asn1Error as error:
        assert "18.1.7" in str(error) and "#Inner" in str(error), str(error)
    else:
        raise AssertionError("a set held two objects of one class")


def test_completed_by_fills_gaps_and_never_overrides():
    """§18.1.8 sends the braced spec through §13.2 as `PrimaryEncodings`, and §9.23.2 makes
    that combination left-biased: the completion supplies "an encoding object for any encoding
    class for which the first set is lacking one". Never the other way round.
    """
    source = _CONTENTS.replace(
        "innerSet #ENCODINGS ::= { innerEnc }",
        "other #Other ::= { ENCODING cond }\n"
        "  fallbackSet #ENCODINGS ::= { other }\n"
        "  innerSet #ENCODINGS ::= { innerEnc } COMPLETED BY fallbackSet").replace(
        "#Inner   ::= #INT", "#Inner   ::= #INT\n  #Other   ::= #INT")
    module = parse_module(source % "CONTENTS-ENCODING innerSet CONTAINING #Inner")
    assert sorted(module.object_sets["innerSet"]) == ["#Inner", "#Other"]
    # The primary's own object survives the merge -- it is not replaced by the completion.
    assert module.object_sets["innerSet"]["#Inner"].name == "innerEnc"


def test_an_object_set_reaches_the_digest():
    """Two modules assigning different sets describe different octets for a contained type."""
    narrow = parse_module(_CONTENTS % "CONTENTS-ENCODING innerSet CONTAINING #Inner")
    assert b"object-set innerSet #Inner=innerEnc" in narrow.serialize()


def test_the_contained_type_group_reads_on_the_keyword_the_clause_actually_uses():
    """§22.11.1.2 spells the group `CONTENTS-ENCODING`, and it is built on that word.

    This asserted a *refusal* until the group was built, and the refusal itself was keyed on
    `CONTAINED` until that was checked against the text — a word X.692 uses only in prose, as
    in "contained type". The test survives the group being built because what it was really
    pinning is the keyword: `CONTENTS-ENCODING` is read, and `CONTAINED` still means nothing.
    """
    module = parse_module(_CONTENTS % "CONTENTS-ENCODING innerSet CONTAINING #Inner")
    spec = module.objects["payObj"][1]
    assert sorted(spec.contents.primary) == ["#Inner"]
    assert spec.contained_class == "#Inner"

    # The word that is not a keyword is still not one.
    try:
        parse_module(_CONTENTS % "CONTAINED BY innerSet")
    except Asn1Error as error:
        assert "22.11" not in str(error), (
            "`CONTAINED` is prose in X.692, not notation; it must not claim §22.11's citation")
    else:
        raise AssertionError("`CONTAINED BY x` parsed as though it meant something")


def test_the_contained_type_group_makes_22_11_2s_choice():
    """§22.11.2's five-row table, reached from module text for the first time.

    The row that matters is the last one, where §22.11.2.2 and §13.2.10.6 a) contradict each
    other: with the group set, an `ENCODED BY` present and `OVERRIDE FALSE`, §13.2.10.6 a) says
    the `ENCODED BY` stands. `ecn_user.ContainedType.select` is where that reading lives, and
    routing the string class through it rather than copying the table is the point — one place
    for a disagreement this repository had to resolve by weight of agreement.
    """
    encoded_by = {"#Inner": "from-ENCODED-BY"}
    containing = {"#Inner": "from-the-container"}

    override = parse_module(
        _CONTENTS % "CONTENTS-ENCODING innerSet OVERRIDE TRUE CONTAINING #Inner"
    ).objects["payObj"][1]
    declines = parse_module(
        _CONTENTS % "CONTENTS-ENCODING innerSet CONTAINING #Inner"
    ).objects["payObj"][1]

    # OVERRIDE TRUE: this group's set wins (§22.11.2.2, §13.2.10.6 b).
    assert sorted(override.contained_objects(containing)) == ["#Inner"]
    assert override.contained_objects(containing) is not encoded_by
    # OVERRIDE FALSE with an ENCODED BY present: §13.2.10.6 a) leaves the ENCODED BY standing.
    declines = replace(declines, encoded_by=encoded_by)
    assert declines.contained_objects(containing) == encoded_by


def test_a_contents_encoding_group_must_name_the_type_it_encodes():
    """§22.11.1.3: the group's purpose is "to determine the encoding of a contained type".

    X.692 takes that type from the ASN.1 `CONTAINING` constraint through clause 12's link. No
    ELM section is readable here, so the class is written beside the group — a **deviation,
    stated**, of the same shape and for the same reason as `#INT`'s `BOUNDS` and `AUXILIARY`.
    Omitting it is refused rather than defaulted: a contained type nobody named is a set
    applied to nothing.
    """
    try:
        parse_module(_CONTENTS % "CONTENTS-ENCODING innerSet")
    except Asn1Error as error:
        assert "22.11.1.3" in str(error) and "CONTAINING" in str(error), str(error)
    else:
        raise AssertionError("a contents encoding with no contained type was accepted")


def test_the_contents_group_names_a_set_the_module_assigns():
    """§18.1.1's reference has to resolve, or the group selects from nothing."""
    try:
        parse_module(_CONTENTS % "CONTENTS-ENCODING noSuchSet CONTAINING #Inner")
    except Asn1Error as error:
        assert "18.1.1" in str(error) and "noSuchSet" in str(error), str(error)
    else:
        raise AssertionError("an undefined object set was accepted")


_REPSTRUCT = """M ENCODING-DEFINITIONS ::= BEGIN
  #Item ::= #INT
  #Rep  ::= #REPETITION
  %s
END
"""


def test_a_repetition_structure_reads_with_and_without_its_identifier():
    """§16.4.1's `RepetitionClass "{" identifier? EncodingStructure "}" Size?`.

    The third `EncodingStructureDefn`, and the one shaped unlike the other two: §16.3 and
    §16.5 take a *list* of `NamedField`s, this takes exactly one `EncodingStructure` whose
    identifier is **optional** — §16.4.2 has it identify "repeated occurrences of the
    `EncodingStructure`", so there is only ever one thing to name.

    That optional identifier has been load-bearing since slice G1 without a structure able to
    exercise it: §17.5.11 makes an `EncodeStructure`'s identifier omitted "if and only if the
    governing encoding constructor is a class in the repetition category with no identifier on
    the repeated element". `ecn_encode.EncodeStructure.unnamed_element` has checked that
    biconditional against a structure no module could write until now.
    """
    named = parse_module(_REPSTRUCT % "Items ::= #Rep { item #Item } (SIZE (0..MAX))")
    assert named.structures["Items"].category == "repetition"
    assert named.structures["Items"].fields == (("item", "#Item"),)

    bare = parse_module(_REPSTRUCT % "Bare ::= #Rep { #Item } (SIZE (4))")
    assert bare.structures["Bare"].fields == (("", "#Item"),)

    # §16.4.1's `Size?` really is optional.
    assert parse_module(_REPSTRUCT % "Plain ::= #Rep { item #Item }").structures[
        "Plain"].size is None


def test_the_repetition_size_bounds_the_number_of_repetitions():
    """§16.4.2: the `Size` specifies "bounds on the number of repetitions" — a count, not a
    value, which is why it lands in `SizeBounds` and not `IntegerBounds`.

    §16.2.10 gives both the range and the fixed form, and §16.2.11 constrains the range twice:
    "MIN shall not be used in `Size`" and the number "shall be non-negative when used in
    `Size`". Both are refused, because a count of repetitions has a floor at zero that a value
    range does not — which is the whole reason the clause says it separately.
    """
    from bcir.asn1.ecn_user import SizeBounds

    assert parse_module(_REPSTRUCT % "A ::= #Rep { i #Item } (SIZE (0..MAX))").structures[
        "A"].size == SizeBounds(0, None)
    assert parse_module(_REPSTRUCT % "B ::= #Rep { i #Item } (SIZE (4))").structures[
        "B"].size == SizeBounds(4, 4)

    for text, cite in (("C ::= #Rep { i #Item } (SIZE (MIN..4))", "16.2.11"),
                       ("D ::= #Rep { i #Item } (SIZE (-1..4))", "16.2.11")):
        try:
            parse_module(_REPSTRUCT % text)
        except Asn1Error as error:
            assert cite in str(error), (text, str(error))
        else:
            raise AssertionError(f"{text!r} was accepted")


def test_a_repetition_structure_holds_exactly_one_encoding_structure():
    """§16.4.1 brackets a single `EncodingStructure`, not a list. A second is not a longer
    repetition — it is a concatenation written under the wrong class."""
    try:
        parse_module(_REPSTRUCT % "E ::= #Rep { a #Item, b #Item }")
    except Asn1Error as error:
        assert "16.4.1" in str(error), str(error)
    else:
        raise AssertionError("a repetition structure took two elements")


def test_the_repetition_size_reaches_the_digest():
    """Two modules differing only in the bound on the repetition count describe different
    octets: §23.14's `#CONDITIONAL-REPETITION` selects on exactly those bounds (§23.2.2.3)."""
    narrow = parse_module(_REPSTRUCT % "F ::= #Rep { i #Item } (SIZE (4))")
    wide = parse_module(_REPSTRUCT % "F ::= #Rep { i #Item } (SIZE (0..MAX))")
    assert narrow.sha256() != wide.sha256()
    assert b"structure F repetition fields 1 size 4..4" in narrow.serialize()


def test_a_builtin_class_this_grammar_cannot_spell_is_not_called_undefined():
    """Seven X.692 classes were being reported as though they did not exist.

    `builtin_of`'s general message — "is not a built-in encoding class and no assignment
    defines it" — is right for a typo and **false** for these: X.692 defines every one, and
    `ecn_user` implements every one. A reader who wrote `#OCTETS` has not misspelled anything;
    they have reached a class this parser has not been taught to spell. Two different mistakes
    with two different fixes, and they were indistinguishable.

    The message also carries the build order, which is the thing that is easy to get wrong
    from outside: §23.2/§23.4/§23.9's string classes take their size from §22.7's repetition
    space rather than from an `ENCODING-SPACE`, so their `WITH SYNTAX` is written in terms of
    `REPETITION-ENCODING(S)` and §23.14's `#CONDITIONAL-REPETITION` has to come first.
    """
    from bcir.asn1.ecn_syntax import _UNREADABLE_CLASSES

    def refusal(class_name: str) -> str:
        try:
            parse_module(f"M ENCODING-DEFINITIONS ::= BEGIN\n  #X ::= {class_name}\nEND\n")
        except Asn1Error as error:
            return str(error)
        raise AssertionError(f"{class_name} was accepted")

    for class_name, (clause, spec) in _UNREADABLE_CLASSES.items():
        message = refusal(class_name)
        assert clause in message and spec in message, (class_name, message)
        assert "is not a built-in encoding class" not in message, (class_name, message)

    # A genuine typo still gets the general message, or the new one would have swallowed it.
    assert "is not a built-in encoding class" in refusal("#NONSENSE")

    # These are the ones `ecn_user` builds and this grammar cannot yet spell. A class in
    # neither table would be one this repository has not modelled at all, and there is no such
    # thing in clause 23 today.
    #
    # The set SHRINKS as classes become readable, which is why it is pinned exactly rather
    # than by membership: §23.14's `#CONDITIONAL-REPETITION` left it when its defined syntax
    # was built, then §23.2's `#BITS` and §23.9's `#OCTETS` when theirs was, and this
    # assertion is what made each of those a deliberate edit instead of a silent one.
    #
    # §23.4's `#CHARS` shares the string classes' WITH SYNTAX and stayed: its repeated element
    # is a character, whose width comes from the ASN.1 type's character set through clause
    # 12's link, where §23.2's bit and §23.9's octet are intrinsic to the class. It is also
    # not on the path to §22.11, whose group appears in §23.2.1's and §23.9.1's syntax and in
    # no other class's.
    assert set(_UNREADABLE_CLASSES) == {"#CHARS", "#NUL", "#TAG"}


def test_an_unimplemented_property_group_is_refused_by_name_and_never_skipped():
    """The failure a permissive parser causes is silent wrong octets, not a crash.

    Each group below is one this repository has not built. Accepting and ignoring any of them
    would produce an encoding that does not match the specification it was handed, which is
    exactly the class of defect the triple-rail design exists to catch — so they are
    recognized, cited and refused.
    """
    space = "ENCODING-SPACE SIZE 3 MULTIPLE OF bit"
    cases = [
        # `CONTENTS-ENCODING` used to be here. It is built now — §18.1's object-set
        # assignments and §23.2/§23.9's string classes are what it was waiting on — so the
        # table holds no unbuilt group at all, and every case below is one that IS built and
        # is written in a way its clause forbids.
        # `REPLACE` used to be here. Its defined syntax is read now, and the fixture above
        # still fails — for a *different* reason: §22.1.2.2 refuses a WITH structure the
        # module never declared. Kept below as a REPLACE test rather than an unimplemented one.
        # §23.11's `PRESENCE` and §23.1's `ALTERNATIVE` both used to sit here and no longer
        # do: §16.5's `ConcatComponentPresence` and §16.3's `AlternativesStructure` are built,
        # so both keywords are read rather than refused. See
        # test_an_optional_encoding_marker_pairs_a_component_with_its_optionality_object and
        # test_an_alternatives_structure_encodes_precisely_one_of_its_named_fields.
        #
        # What that leaves is groups that ARE built and are refused where a clause forbids
        # the way they were written — which is the more interesting half anyway: a parser that
        # only refused what it had not implemented would accept every one of them.
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


_NEST = """BCIR-Nest ENCODING-DEFINITIONS ::= BEGIN
  #Len  ::= #INT
  #Val  ::= #INT
  #Join ::= #CONCATENATION
  %s
END
"""


def test_the_application_point_is_the_first_structure_declared():
    """A repository convention, asserted as one — because X.692 puts the choice out of reach.

    §12.1's encoding link module binds an ASN.1 type to an encoding structure; an
    `ENCODING-DEFINITIONS` module on its own *defines* structures and never says which is
    applied. So the rule here is declaration order, and this pins it: when an ELM section
    becomes readable, this is the assertion that should change.

    First-declared rather than sole-declared is forced by timing. §23's objects resolve their
    structure while the module is still being parsed, so a rule needing the whole module — "the
    one nothing else claims" — would answer differently depending on how far the reader had got.

    Order-sensitivity is the convention's real cost, so it is shown rather than described:
    declaring the head-end structure first makes *it* the application point, and the structure
    the object needs becomes the one nothing reaches.
    """
    clause = ("REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY lp-object "
              "INSERT AT HEAD Head-structure")
    module = parse_module(_replace_module_with_head(clause))
    assert list(module.structures) == ["Payload-structure", "Head-structure"]
    assert module.structure_name == "Payload-structure"
    assert module.structure == (("body", "#Val"),)

    swapped = _replace_module(clause).replace(
        "  Payload-structure ::= #Join { body #Val }",
        "  Head-structure ::= #Join { offset #Len }\n"
        "  Payload-structure ::= #Join { body #Val }")
    try:
        parse_module(swapped)
    except Asn1Error as error:
        # `Head-structure` is now the application point, so `Payload-structure` is the one
        # nothing reaches — and the INSERT AT HEAD names the application point besides.
        assert "22.1.2.7" in str(error), str(error)
    else:
        raise AssertionError("declaration order did not decide the application point")


def test_a_structure_nothing_reaches_is_refused_at_the_modules_end():
    """A declared structure that is neither the application point nor named by anything has
    written octets nobody emits.

    Checked at `END` rather than at the declaration, because that is the first moment every
    claim has been seen — a `REPLACE ... INSERT AT HEAD` may appear after the structure it
    names, and refusing at the declaration would reject a legal module for being ordered
    inconveniently.
    """
    try:
        parse_module(_NEST % "First ::= #Join { a #Len }\n  Orphan ::= #Join { b #Val }")
    except Asn1Error as error:
        assert "Orphan" in str(error) and "22.1.2.7" in str(error), str(error)
    else:
        raise AssertionError("a structure nothing reaches was accepted")


def test_the_nesting_depth_guard_is_the_readers_own_and_says_so():
    """§16.2.1 sets no depth limit, so this one is not a conformance rule and the message says
    which kind of limit it is. Its job is to turn a runaway module into a sentence instead of a
    `RecursionError` raised somewhere inside the tokenizer.
    """
    from bcir.asn1.ecn_syntax import _MAX_STRUCTURE_DEPTH

    levels = _MAX_STRUCTURE_DEPTH + 2
    nest = ("Deep ::= #Join " + "{ f #Join " * levels + "{ x #Len }" + " }" * levels)
    try:
        parse_module(_NEST % nest)
    except Asn1Error as error:
        assert "16.2.1" in str(error) and "reader" in str(error), str(error)
    else:
        raise AssertionError("an unbounded nesting depth was accepted")


def test_a_nested_encoding_structure_is_read_as_a_definition_not_flattened():
    """§16.2.1: "An `EncodingStructure` is either a `DefinedEncodingClass` or an
    `EncodingStructureDefn`" — so a field may be a whole structure, and this reads one.

    The load-bearing assertion is that `head` stays a *single* field of `Outer`. Flattening
    would put `a` and `b` into `Outer`'s own transmission order, which is a different encoding
    from the one written, and it is the failure this refused outright before rather than risk.
    """
    module = parse_module(_NEST % "Outer ::= #Join { head #Join { a #Len, b #Val }, tail #Val }")
    outer = module.structures["Outer"]
    assert outer.fields == (("head", "#Join"), ("tail", "#Val"))
    inner = outer.nested["head"]
    assert inner.fields == (("a", "#Len"), ("b", "#Val"))
    # The nested definition is named by its path, so two structures nesting a `head` each keep
    # their own in the digest instead of colliding on the bare field name.
    assert inner.name == "Outer.head"
    assert inner.category == "concatenation"


def test_a_nested_structure_reaches_the_digest_under_its_path():
    """Two modules differing only inside a nested body describe different octets."""
    narrow = parse_module(_NEST % "Outer ::= #Join { head #Join { a #Len }, tail #Val }")
    wide = parse_module(_NEST % "Outer ::= #Join { head #Join { a #Len, b #Val }, tail #Val }")
    assert narrow.sha256() != wide.sha256()
    assert b"structure Outer.head concatenation fields 1" in narrow.serialize()
    # The parent's field line says the field carries a definition, so a reader of the digest
    # cannot mistake a nested `#Join` field for a plain reference to a `#Join` class.
    assert b"field 0 name head class #Join nested" in narrow.serialize()


def test_encoding_a_nested_structure_is_refused_rather_than_given_its_parents_object():
    """Reading the notation is not encoding it, and this rail says which it has done.

    §16.5.6's application point "proceeds to each of the `EncodingStructure`s" — one object per
    level. This rail builds one object from one flat field list, so applying that object to a
    parent whose field is a whole structure would give the nested fields the *parent's*
    encoding: well-formed octets of the wrong shape, which is exactly what the old outright
    refusal was protecting against. The protection survives the notation becoming readable.
    """
    source = (_NEST % "Outer ::= #Join { head #Join { a #Len }, tail #Val }").replace(
        "END",
        """  cond #CONDITIONAL-INT ::= { ELSE ENCODING-SPACE SIZE 8 MULTIPLE OF bit
                              ENCODING positive-int }
  lenEnc #Len ::= { ENCODING cond }
  valEnc #Val ::= { ENCODING cond }
  joinObj #Join ::= { CONCATENATION ORDER textual ALIGNMENT none }
END""")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "16.5.6" in str(error) and "nests" in str(error), str(error)
    else:
        raise AssertionError("an object was applied to a structure it cannot encode")


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


_STRING = """M ENCODING-DEFINITIONS ::= BEGIN
  #Count   ::= #INT
  #Payload ::= #OCTETS
  #Frame   ::= #CONCATENATION
  Frame-structure ::= #Frame { n #Count, s #Payload }
  cntObj #Count ::= { AUXILIARY ENCODING-SPACE SIZE 8 MULTIPLE OF bit }
  rep #CONDITIONAL-REPETITION ::= { ELSE REPETITION-SPACE DETERMINED BY field-to-be-set USING n }
  payObj #Payload ::= { %s REPETITION-ENCODING rep }
  frameObj #Frame ::= { CONCATENATION ORDER textual ALIGNMENT none }
END
"""


def test_an_octetstring_class_encodes_from_module_text_end_to_end():
    """§23.9's `#OCTETS`, written down and run — step two of the three reaching §22.11.

    The octets are the proof rather than the parse tree: §23.9.2.1 b) has the value
    "considered as a repetition of an octet", §22.7's space writes the count, and the result is
    a length-prefixed octet string that no property of the object states directly. Every part
    of that came from a different clause, which is why the assertion is on bytes.
    """
    from bcir.asn1.ecn_user import encode_with_user

    module = parse_module(_STRING % "")
    assert encode_with_user(module.object_set(), "#Frame", {"s": [1, 2, 3]}) == bytes(
        (3, 1, 2, 3))
    # The count is the number of REPETITIONS, so an empty string is one zero octet.
    assert encode_with_user(module.object_set(), "#Frame", {"s": []}) == bytes((0,))

    spec = module.objects["payObj"][1]
    # §23.9.2.1 b)'s element is intrinsic to the class and never written in the object.
    assert spec.element.width == 8
    assert len(spec.repetition.encodings) == 1


def test_the_repeated_element_comes_from_the_class_not_the_repetition_object():
    """The seam `PendingConditionalRepetition.bind` exists for, now exercised.

    §23.14.1's syntax carries no element; §23.2.2.1 b) and §23.9.2.1 b) supply it, one bit or
    one octet according to which class names the object. So the *same* `#CONDITIONAL-REPETITION`
    object gives a `#BITS` class 1-bit elements and an `#OCTETS` class 8-bit ones, and that is
    what makes deferring it right rather than merely tidy.
    """
    shared = """M ENCODING-DEFINITIONS ::= BEGIN
  #B ::= #BITS
  #O ::= #OCTETS
  rep #CONDITIONAL-REPETITION ::= { ELSE REPETITION-SPACE DETERMINED BY not-needed }
  bObj #B ::= { REPETITION-ENCODING rep }
  oObj #O ::= { REPETITION-ENCODING rep }
END
"""
    module = parse_module(shared)
    assert module.objects["bObj"][1].element.width == 1
    assert module.objects["oObj"][1].element.width == 8


def test_a_string_class_must_say_how_its_repetition_is_encoded():
    """§23.2.2.1 d) and §23.9.2.1 d): one of `REPETITION-ENCODING`/`REPETITION-ENCODINGS`.

    Neither is optional in effect, because the class has no `ENCODING-SPACE` to fall back on —
    an object setting neither describes a value with no size, which is not a smaller encoding
    but no encoding at all.
    """
    try:
        # The `%s` slot goes with the clause being removed, so this fixture takes no argument.
        parse_module(_STRING.replace("{ %s REPETITION-ENCODING rep }", "{ }"))
    except Asn1Error as error:
        assert "23.2.2.1" in str(error), str(error)
    else:
        raise AssertionError("a string class with no repetition encoding was accepted")


def test_both_repetition_spellings_may_not_be_set_at_once():
    """§23.13.2.2 permits exactly one, and §23.2.2.1's NOTE says why the singular exists at
    all: to avoid "a double curly-bracket" for one object. A convenience, not a second
    meaning."""
    try:
        parse_module(_STRING.replace(
            "REPETITION-ENCODING rep }",
            "REPETITION-ENCODING rep REPETITION-ENCODINGS { rep } }") % "")
    except Asn1Error as error:
        assert "23.13.2.2" in str(error), str(error)
    else:
        raise AssertionError("both repetition spellings were accepted")


def test_a_repetition_reference_must_name_a_conditional_repetition_object():
    """A `#BITS` object that took its size from something which never described a repetition
    would produce octets nothing in the module specifies."""
    try:
        parse_module(_STRING.replace("REPETITION-ENCODING rep", "REPETITION-ENCODING cntObj")
                     % "")
    except Asn1Error as error:
        assert "23.2.2.1" in str(error) and "cntObj" in str(error), str(error)
    else:
        raise AssertionError("a non-repetition object was accepted as a repetition encoding")


def test_value_reversal_reaches_the_digest():
    """§23.2.1's `&value-reversal` reverses the order of the *elements*, where §22.12's bit
    reversal reverses bits within a unit. Two different formats, two different properties, and
    a module that sets one describes different octets from a module that does not.
    """
    plain = parse_module(_STRING % "")
    reversed_ = parse_module(_STRING % "VALUE-REVERSAL TRUE")
    assert plain.sha256() != reversed_.sha256()
    assert b"value-reversal 1" in reversed_.serialize()


def test_a_class_this_rail_cannot_execute_is_named_rather_than_ignored():
    """This used `#BITS` until §23.2's notation was built.

    `#CHARS` replaces it because the assertion is about a class the grammar *cannot spell* —
    and §23.4's is the one that is still true of. Its element is a character, whose width comes
    from the ASN.1 type's character set through clause 12's link; §23.2's bit and §23.9's octet
    are intrinsic to the class, which is why those two became readable and this one did not.
    """
    source = frame_header_source().replace("#Reserved      ::= #PAD",
                                           "#Reserved      ::= #CHARS")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "#CHARS" in str(error) and "23.4" in str(error), error
    else:
        raise AssertionError("an unimplemented encoding class parsed")


def test_a_string_class_has_no_encoding_space_and_says_so():
    """§23.2.1 and §23.9.1 give the string categories no `ENCODING-SPACE` group at all.

    Their size comes from the §22.7 repetition space of the `#CONDITIONAL-REPETITION` objects
    they name, which is the whole reason they could not be read until §23.14 was. Writing a
    width on one is not a small mistake — it is asking for a group the class does not have —
    so it is refused by name rather than ignored.

    This is what the `#BITS` half of the test above became once §23.2 was readable: the class
    is no longer unspellable, and the fault moved to the group.
    """
    source = frame_header_source().replace("#Reserved      ::= #PAD",
                                           "#Reserved      ::= #BITS")
    try:
        parse_module(source)
    except Asn1Error as error:
        assert "ENCODING-SPACE" in str(error) and "23.2.1" in str(error), error
    else:
        raise AssertionError("a string class accepted an ENCODING-SPACE")


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

    # 4 for EXHIBITS HANDLE, 5 for Annex C, 6 for §16.5's marker, 7 for §16.3's structure,
    # 8 for a module holding several structures — the serialization emits every one of them,
    # so a module that gains a §22.1.2.7 head-end structure no longer hashes as it did — and
    # 9 for §23.14's `#CONDITIONAL-REPETITION`, a kind of object the digest can now contain,
    # 10 for §23.2's `#BITS` and §23.9's `#OCTETS`, which add two more, 11 for §18.1's
    # encoding object sets plus §22.11's CONTENTS-ENCODING group on the string classes, and
    # 12 for §16.4's RepetitionStructure, whose §16.4.2 Size every structure line now carries.
    assert SYNTAX_VERSION == 12
    base = parse_module(_with_assignments(_LP_STRUCTURE))
    assert b"parameterized #Length-prefixed" in base.serialize()

    wider = parse_module(_with_assignments(
        "#Length-prefixed{<#D>} ::= #CONCATENATION { len16 #INT, value #D }"))
    assert base.sha256() != wider.sha256()

    governed = parse_module(_with_assignments(
        "#Length-prefixed{<#ENCODINGS:D>} ::= #CONCATENATION { length #INT, value #D }"))
    assert governed.sha256() != base.sha256()
    assert b"{<#ENCODINGS:D>}" in governed.serialize()





# --- §17.5.1's EncodeStructure, as a #CONCATENATION object body ---------------------------

_CONCAT_OBJECT = "frameHeader #Frame-header ::= { CONCATENATION ORDER textual ALIGNMENT none }"


def _with_object(body: str, *, extra: str = "") -> str:
    """The gate's module with its `#CONCATENATION` object replaced (and optional additions)."""
    return frame_header_source().replace(_CONCAT_OBJECT, extra + body)


def test_an_encode_structure_of_all_use_sets_describes_the_same_encoding_and_hashes_the_same():
    """§17.5.6 makes `USE-SET` "obtained by applying the `CombinedEncodings`", and in this rail
    the combined set is the one the module forms — so a body that says `USE-SET` everywhere is
    the property-group body written differently.

    The digest agreeing is the assertion that matters. A canonical serialization names *what
    octets the specification describes*, not how it was spelled, so two spellings of one
    encoding must collide — the opposite of the `EXHIBITS HANDLE` and parameterized-assignment
    cases, where the spelling changes what a decoder reads and the hash has to move.
    """
    plain = parse_module(frame_header_source())
    spelled = parse_module(_with_object(
        "frameHeader #Frame-header ::= { ENCODE STRUCTURE { "
        "payloadOctets USE-SET, version USE-SET, reserved USE-SET } WITH frame-set }"))
    assert spelled.concatenation().fields == plain.concatenation().fields
    assert spelled.concatenation().order == plain.concatenation().order
    assert spelled.sha256() == plain.sha256()


def test_naming_an_object_per_component_is_what_this_body_form_buys():
    """§9.5.2 permits at most one encoding object per class *in the object set*, so the
    property-group body reaches every field through its class and two fields of one class
    necessarily share an encoding. §17.5.10's `ComponentEncoding` names an object directly,
    which is a different route to the same field and is not bound by the set.

    So a module with two objects for one class is a specification the old body cannot use and
    this one can. Both halves are asserted, because the point is the contrast.
    """
    second = "wideReserved #Reserved ::= { ENCODING-SPACE SIZE 2 MULTIPLE OF bit }\n  "
    try:
        parse_module(_with_object(_CONCAT_OBJECT, extra=second)).concatenation()
    except Asn1Error as error:
        assert "9.5.2" in str(error), str(error)
    else:
        raise AssertionError("two objects for one class should not form a set")

    named = parse_module(_with_object(
        "frameHeader #Frame-header ::= { ENCODE STRUCTURE { "
        "payloadOctets USE-SET, version USE-SET, reserved wideReserved } WITH frame-set }",
        extra=second))
    assert named.concatenation().fields["reserved"].width == 2


def test_an_object_named_for_a_component_must_be_governed_by_that_components_class():
    """§17.5.13: the `EncodingObject`s "shall be governed by the corresponding encoding classes
    in the component". Without the check, naming an integer object for a boolean field would
    encode a boolean as an integer — well-formed octets of the wrong shape, which no later
    stage would question."""
    for body, citation in (
            ("frameHeader #Frame-header ::= { ENCODE STRUCTURE { payloadOctets USE-SET, "
             "version USE-SET, reserved versionField } WITH frame-set }", "17.5.13"),
            ("frameHeader #Frame-header ::= { ENCODE STRUCTURE { payloadOctets USE-SET, "
             "version USE-SET, reserved nosuchobject } WITH frame-set }", "17.5.13"),
            # §17.5.3: no STRUCTURED WITH and no trailing WITH.
            ("frameHeader #Frame-header ::= { ENCODE STRUCTURE { payloadOctets USE-SET, "
             "version USE-SET, reserved USE-SET } }", "17.5.3"),
            # §17.5.8: the components' own textual order, not the writer's preference.
            ("frameHeader #Frame-header ::= { ENCODE STRUCTURE { version USE-SET, "
             "payloadOctets USE-SET, reserved USE-SET } WITH frame-set }", "17.5.8"),
            # §17.5.10's ComponentEncoding is an identifier AND an encoding.
            ("frameHeader #Frame-header ::= { ENCODE STRUCTURE { payloadOctets, "
             "version USE-SET, reserved USE-SET } WITH frame-set }", "17.5.10")):
        try:
            parse_module(_with_object(body))
        except Asn1Error as error:
            assert citation in str(error), (citation, str(error))
        else:
            raise AssertionError(f"{body!r} was accepted")


def test_a_component_left_out_is_encoded_by_the_object_set_rather_than_being_an_error():
    """§17.5.10: a component with no `ComponentEncoding` is encoded by the `CombinedEncodings`,
    which "shall be present ... and is required, on application to the component, to provide a
    complete encoding of that component".

    So an incomplete list is a legal specification, not a partial one — which is why
    §17.5.8's order rule had to be a subsequence test rather than an equality one.
    """
    partial = parse_module(_with_object(
        "frameHeader #Frame-header ::= { ENCODE STRUCTURE { reserved USE-SET } "
        "WITH frame-set }"))
    plain = parse_module(frame_header_source())
    assert partial.concatenation().fields == plain.concatenation().fields


# --- §16.5's ConcatComponentPresence, and the #OPTIONAL object it names --------------------

_STRUCTURE_CLASS = "  #Frame-header  ::= #CONCATENATION"
_PRESENCE_OBJECT = (
    "  presenceBit #Present ::= { PRESENCE DETERMINED BY field-to-be-set USING flag }\n")


def _with_optional(marker: str = "#Present", presence: str = _PRESENCE_OBJECT) -> str:
    """The gate's module with `reserved` marked `OPTIONAL-ENCODING`.

    Declaration order is load-bearing and worth seeing: the class has to precede the structure
    that references it, and the object has to precede the `#CONCATENATION` object that forms
    the set — which is where §9.5.1 is checked.
    """
    return (frame_header_source()
            .replace(_STRUCTURE_CLASS, "  #Present ::= #OPTIONAL\n" + _STRUCTURE_CLASS)
            .replace("reserved       #Reserved",
                     f"reserved       #Reserved OPTIONAL-ENCODING {marker}")
            .replace(_CONCAT_OBJECT, presence + _CONCAT_OBJECT))


def test_an_optional_encoding_marker_pairs_a_component_with_its_optionality_object():
    """§16.5.1's `ConcatComponentPresence ::= OPTIONAL-ENCODING OptionalClass`, and §16.5.4:
    "the mechanism used to determine whether there is an encoding of the corresponding
    `EncodingStructure` is specified by the encoding object which encodes the `OptionalClass`".

    So the pairing has two owners and neither can do it alone — the mechanism is the
    `#OPTIONAL` object's, the component is the structure's. `optional_wrapped` is the one place
    that knows both, which is why it takes a field name and a spec rather than living on either.
    """
    module = parse_module(_with_optional())
    reserved = module.concatenation().fields["reserved"]
    assert isinstance(reserved, OptionalSpec)
    assert reserved.presence.determination is OptionalityDetermination.FIELD_TO_BE_SET
    assert reserved.presence.reference == "flag"
    # §16.5.3: an UNMARKED component "shall appear precisely once in the encoding", so it is
    # returned untouched — which is what makes the wrap safe to attempt on every field.
    assert not isinstance(module.concatenation().fields["version"], OptionalSpec)
    # The component's own encoding survives inside the wrapper rather than being replaced.
    assert reserved.component.width == 2


def test_the_optional_class_must_be_in_the_optionality_category():
    """§16.5.2: "the `DefinedEncodingClass` in the `OptionalClass` shall be a class in the
    optionality category". A class from any other category names an object that cannot say
    whether the component is present, which is the one thing the marker exists to supply."""
    try:
        parse_module(_with_optional(marker="#Version"))
    except Asn1Error as error:
        assert "16.5.2" in str(error), str(error)
    else:
        raise AssertionError("a non-optionality OptionalClass was accepted")


def test_an_optional_object_that_never_says_how_absence_is_detected_is_refused():
    """§22.5.1.6: the `PRESENCE` specification "is mandatory for it to be set in all places in
    the defined syntax where it is allowed. Defaulting all other parts of this defined syntax
    (e.g., use of `PRESENCE` alone) would not satisfy the above constraints."

    So an `#OPTIONAL` object with no `PRESENCE` is not one taking defaults — it is one that
    never said how a decoder detects absence, and there is no default that could stand in.
    """
    try:
        parse_module(_with_optional(presence="  presenceBit #Present ::= { }\n"))
    except Asn1Error as error:
        assert "22.5.1.6" in str(error), str(error)
    else:
        raise AssertionError("an #OPTIONAL object with no PRESENCE was accepted")


def test_the_marker_reaches_the_digest_because_it_changes_what_a_decoder_reads():
    """A component that may be absent is read differently from one that is always there, so two
    modules differing only in the marker describe different octets. `SYNTAX_VERSION` moved to 6
    for the same reason it moved to 4 for `EXHIBITS HANDLE` and 5 for Annex C.

    This is the *opposite* of §17.5's all-`USE-SET` `EncodeStructure`, which is a second
    spelling of one encoding and deliberately hashes the same. Both are asserted, because the
    distinction between "spelled differently" and "means something different" is the entire
    basis of the digest.
    """
    marked = parse_module(_with_optional())
    assert marked.sha256() != parse_module(frame_header_source()).sha256()
    assert b"optional-encoding #Present" in marked.serialize()
    assert b"optional presence field-to-be-set ref flag" in marked.serialize()


# --- §16.3's AlternativesStructure, the second constructor shape --------------------------

def _alternatives_module(body: str = "num #Small, flag #Flag",
                         obj: str = "ALTERNATIVE DETERMINED BY field-to-be-set USING sel",
                         governor: str = "#Pick") -> str:
    """A module whose one structure is a §16.3 `AlternativesStructure`.

    Written out rather than patched from the gate's module because the *governor* is what this
    slice turns on, and a patch would bury it.
    """
    return f"""BCIR-Alt ENCODING-DEFINITIONS ::= BEGIN
  #Small ::= #INT
  #Flag  ::= #BOOL
  #Pick  ::= #ALTERNATIVES
  #Join  ::= #CONCATENATION
  Choice-structure ::= {governor} {{ {body} }}
  smallCond #CONDITIONAL-INT ::= {{ ELSE ENCODING-SPACE SIZE 8 MULTIPLE OF bit
                                    ENCODING positive-int }}
  smallInt #Small ::= {{ ENCODING smallCond }}
  flagBit  #Flag  ::= {{ ENCODING-SPACE SIZE 1 MULTIPLE OF bit }}
  pick #Pick ::= {{ {obj} }}
END
"""


def test_an_alternatives_structure_encodes_precisely_one_of_its_named_fields():
    """§16.2.12 names three `EncodingStructureDefn`s and two are read now:
    `AlternativesStructure` is §16.3 and `ConcatenationStructure` is §16.5.

    They **share their body** — §16.3.1's `NamedField ::= identifier EncodingStructure` is what
    both are built from — and differ only in what the field list means. §16.3.2: the structure
    "identifies the presence in an encoding of **precisely one** of the `EncodingStructure`s in
    its `NamedFields`", against §16.5.2's zero-or-one for each. Same text, opposite semantics,
    and nothing but the governor's category tells them apart.
    """
    module = parse_module(_alternatives_module())
    spec = next(s for _, s in module.objects.values() if isinstance(s, AlternativesSpec))
    assert list(spec.alternatives) == ["num", "flag"]
    assert spec.order == ("num", "flag")
    assert spec.selection.reference == "sel"
    assert module.structure_category == "alternatives"


def test_the_object_and_the_structure_have_to_agree_on_the_category():
    """§16.3.3 and §16.5.6 both make their structure "an encoding constructor: when an encoding
    object set is applied to this structure ... the application point then proceeds to each of
    the `EncodingStructure`s".

    So the pairing is checked. An `#ALTERNATIVES` object over a concatenation would encode one
    field where all of them belong — a valid encoding of a *different* type, which is the kind
    of mistake that produces well-formed octets and no complaint.
    """
    try:
        parse_module(_alternatives_module(governor="#Join"))
    except Asn1Error as error:
        assert "16.2.12" in str(error), str(error)
    else:
        raise AssertionError("an #ALTERNATIVES object over a concatenation was accepted")


def test_the_optional_marker_is_a_concatenation_tail_and_nothing_else():
    """§16.5.1 hangs `ConcatComponentPresence` off a `ConcatComponent`; §16.3.1's `NamedField`
    has no such tail. §16.3.2 is the reason rather than an accident of the grammar — an
    alternatives structure encodes "precisely one" of its fields, so "this one may be absent"
    would say nothing about it."""
    try:
        parse_module(_alternatives_module(body="num #Small, flag #Flag OPTIONAL-ENCODING #Small"))
    except Asn1Error as error:
        assert "16.5.1" in str(error), str(error)
    else:
        raise AssertionError("an OPTIONAL-ENCODING tail on a NamedField was accepted")


def test_alternative_ordering_has_two_values_where_concatenation_has_three():
    """§22.6.1.1's `&alternative-ordering` is `ENUMERATED {textual, tag}`. `random` is
    §22.10.1.1's, and it would be meaningless here: a CHOICE encodes exactly one alternative,
    so there is no order to randomize. The refusal says *that* rather than listing the enum,
    because a reader who reached for `random` was thinking of the concatenation group.
    """
    for clause, citation in (
            # `random` is refused for not being a value of this type at all...
            ("ALTERNATIVE DETERMINED BY field-to-be-set USING sel ORDER random", "22.6.1.1"),
            # ...while `tag` IS one, parses, and is then refused by §22.6.2.10 for a reason
            # about this module rather than about the enum: "every alternative shall start with
            # an encoding class in the tag category". Two different failures for two different
            # words is the evidence that both values are read rather than both rejected.
            ("ALTERNATIVE DETERMINED BY field-to-be-set USING sel ORDER tag", "22.6.2.10"),
            # §22.6.2.9 makes the group mandatory, exactly as §22.5.1.6 does for PRESENCE.
            ("", "22.6.2.9")):
        try:
            parse_module(_alternatives_module(obj=clause))
        except Asn1Error as error:
            assert citation in str(error), (citation, str(error))
        else:
            raise AssertionError(f"{clause!r} was accepted")


def test_a_marked_component_does_not_change_its_structures_own_category():
    """A regression pin, and the bug is worth naming: §16.5.2's check for the *marker's*
    category reused the variable holding the *structure's*, so one `OPTIONAL-ENCODING` field
    turned its concatenation into an "optional" structure and every later object was rejected
    against it.

    The two categories are different facts about different things and now have different names.
    Asserted through `require_structure`, which is where the damage surfaced.
    """
    module = parse_module(_with_optional())
    assert module.structure_category == "concatenation"
    assert module.require_structure() == module.structure
    # And the marker still did its job, so the fix did not simply drop it.
    assert isinstance(module.concatenation().fields["reserved"], OptionalSpec)


# --- §22.1's REPLACE defined syntax, read from module text --------------------------------

def _replace_module(clause: str) -> str:
    """A module whose concatenation object performs a replacement.

    Every piece §22.1 needs is written out: a parameterized `WITH` structure with a single
    encoding class parameter (§22.1.2.2), an `ENCODED BY` object governed by that structure
    instantiated with its own dummy (§22.1.2.4), and an §17.5.1 `ENCODE STRUCTURE` body naming
    an object per field — which is the specification §22.1.3.5 says the auxiliary values are
    "set according to".
    """
    return f"""BCIR-Rep ENCODING-DEFINITIONS ::= BEGIN
  #Len   ::= #INT
  #Val   ::= #INT
  #Join  ::= #CONCATENATION
  Payload-structure ::= #Join {{ body #Val }}
  #Length-prefixed{{<#D>}} ::= #CONCATENATION {{ length #Len, value #D }}
  lenCond #CONDITIONAL-INT ::= {{ ELSE ENCODING-SPACE SIZE 8 MULTIPLE OF bit
                                  ENCODING positive-int }}
  lenEnc #Len ::= {{ ENCODING lenCond }}
  valCond #CONDITIONAL-INT ::= {{ ELSE ENCODING-SPACE SIZE 16 MULTIPLE OF bit
                                  ENCODING positive-int }}
  valEnc #Val ::= {{ ENCODING valCond }}
  lp-object{{<#D>}} #Length-prefixed{{<#D>}} ::= {{
      ENCODE STRUCTURE {{ length lenEnc, value USE-SET }} WITH combined }}
  joinObj #Join ::= {{ {clause} CONCATENATION ORDER textual ALIGNMENT none }}
END
"""


def _replace_refuses(citation: str, clause: str) -> None:
    try:
        parse_module(_replace_module(clause))
    except Asn1Error as error:
        assert citation in str(error), (citation, str(error))
        return
    raise AssertionError(f"expected a refusal citing {citation} for {clause!r}")


def test_a_replacement_is_built_from_module_text_end_to_end():
    """§22.1's defined syntax, finally readable. The chain it closes is three clauses long:
    §22.1.3.5 says the replacement structure's other fields are "set according to the
    specification in the **replacement structure encoding object**"; §17.5.1's `ENCODE
    STRUCTURE` is that specification; and §22.1.2.6 classifies which fields those are — "all
    fields ... that are not part of the encoding class parameter are auxiliary fields".

    So the dummy field is found by *computation* — the one whose class is the structure's
    parameter — rather than by declaration, and everything else is auxiliary. The transmission
    order is the observable proof: the length field is instantiated around the component and
    written before it.
    """
    module = parse_module(_replace_module(
        "REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY lp-object"))
    spec = module.concatenation()
    replacement = spec.replacement
    assert replacement.action is ReplaceAction.ALL_COMPONENTS
    assert replacement.structure.name == "#Length-prefixed"
    assert replacement.structure.order == ("length", "value")
    assert replacement.structure.dummy == "value"          # §22.1.2.6, computed not declared
    assert set(replacement.structure.auxiliary) == {"length"}
    assert spec.transmission_order() == ("body$length", "body")


def test_exactly_one_of_the_permitted_syntaxes_goes_between_replace_and_with():
    """§22.1.2.1, and it is a closed set of five: §22.1.1.7 lists `STRUCTURE`, `COMPONENT`,
    `ALL COMPONENTS`, `OPTIONALS` and `NON-OPTIONALS`, with §22.1.1.8 making `COMPONENT` "a
    synonym for `REPLACE ALL COMPONENTS`" rather than a sixth action.

    Both failure directions are faults: none of the words, and two of them.
    """
    for clause in ("REPLACE COMPONENT WITH #Length-prefixed ENCODED BY lp-object",
                   "REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY lp-object"):
        module = parse_module(_replace_module(clause))
        # §22.1.1.8's synonym really is one action, not two that behave alike.
        assert module.concatenation().replacement.action is ReplaceAction.ALL_COMPONENTS

    _replace_refuses("22.1.2.1", "REPLACE WITH #Length-prefixed ENCODED BY lp-object")
    _replace_refuses("22.1.2.1",
                     "REPLACE STRUCTURE ALL COMPONENTS WITH #Length-prefixed "
                     "ENCODED BY lp-object")


def test_inside_replace_both_names_are_bare():
    """§22.1.2.2 and §22.1.2.4 close with the same sentence — "only the ... name shall be
    given. They shall not have any parameter list in this use of the names."

    So the structure written `#Length-prefixed{<#D>}` where it is *defined* is
    `#Length-prefixed` here, and copying the definition's spelling is the mistake. C.3's
    `{<>}` is refused too: a legal `ParameterizedReference` elsewhere, still a parameter list
    in this use of the name.
    """
    _replace_refuses("22.1.2.2",
                     "REPLACE ALL COMPONENTS WITH #Length-prefixed{<#D>} ENCODED BY lp-object")
    _replace_refuses("22.1.2.4",
                     "REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY lp-object{<>}")


def test_the_with_structure_and_the_encoded_by_object_are_both_checked_against_the_clause():
    """§22.1.2.2 wants a *parameterized* structure with a single encoding class parameter, and
    §22.1.2.4 wants an object whose governor is that structure. A plain concatenation class is
    neither, and naming one is the natural mistake — it is the class the replacement will be
    applied *to*."""
    _replace_refuses("22.1.2.2", "REPLACE ALL COMPONENTS WITH #Join ENCODED BY lp-object")
    _replace_refuses("22.1.2.4",
                     "REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY valEnc")


def test_auxiliary_fields_with_nothing_to_set_them_are_refused_by_name():
    """§22.1.2.6: the auxiliary fields "shall be set by the encoding of the replacement
    structure". A `WITH` and no `ENCODED BY` leaves `length` unwritten, and an encoder that
    proceeded would emit a field it had no value for."""
    _replace_refuses("22.1.2.6", "REPLACE ALL COMPONENTS WITH #Length-prefixed")


def _replace_module_with_head(
        clause: str, *, head: str = "Head-structure ::= #Join { offset #Len }") -> str:
    """`_replace_module`, plus a second ordinary structure for `INSERT AT HEAD` to name."""
    return _replace_module(clause).replace(
        "  #Length-prefixed", f"  {head}\n  #Length-prefixed")


def test_insert_at_head_reads_a_second_structure_from_module_text():
    """§22.1.2.7, which was refused until a module could hold two ordinary structures.

    The old refusal blamed §13.2 for a limit §13.2 does not impose. The link walks one
    *application point*; nothing in clause 16 caps how many structures a module may **declare**.
    Separating the two is the whole of this change, and the clause became readable with its
    semantics untouched — `ecn_user.HeadEndStructure` and §22.1.3.6's hoisting are as they were.

    §22.1.2.7's own sentences are what the assertions check: the structure has no dummy
    parameters (it is an ordinary `EncodingStructureDefn`), and "all their fields are auxiliary
    fields", each with an encoding object taken from the module under §9.5.2.
    """
    module = parse_module(_replace_module_with_head(
        "REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY lp-object "
        "INSERT AT HEAD Head-structure"))

    assert list(module.structures) == ["Payload-structure", "Head-structure"]
    # The first declaration is the application point; the second is reached by being named.
    assert module.structure_name == "Payload-structure"
    assert module.claimed == {"Head-structure"}

    head_end = module.objects["joinObj"][1].replacement.head_end
    assert head_end.name == "Head-structure"
    assert head_end.order == ("offset",)
    assert set(head_end.auxiliary) == {"offset"}
    # §22.1.3.6 hoists the head-end fields before the component's own, under its name.
    assert [name for name, _ in head_end.expand("body")] == ["body^offset"]


def test_a_head_end_structure_reaches_the_digest():
    """Two modules differing only in the head-end structure describe different octets.

    This is why `SYNTAX_VERSION` moved to 8 rather than the serialization gaining a field
    quietly: before this change the digest covered *the* structure, and a module with two of
    them would have hashed as though the second were not there.
    """
    clause = ("REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY lp-object "
              "INSERT AT HEAD Head-structure")
    narrow = parse_module(_replace_module_with_head(clause))
    wide = parse_module(_replace_module_with_head(
        clause, head="Head-structure ::= #Join { offset #Len, extra #Len }"))
    assert narrow.sha256() != wide.sha256()
    assert b"structure Head-structure concatenation fields 1" in narrow.serialize()


def test_the_head_end_structure_may_not_be_the_application_point_itself():
    """§22.1.2.7 inserts a structure *before the components of* the one being replaced, so
    naming that same structure would insert it before itself.

    Reachable only because both names now resolve: with one structure per module the reference
    could not even be written down.
    """
    _replace_refuses("22.1.2.7",
                     "REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY lp-object "
                     "INSERT AT HEAD Payload-structure")


def test_a_parameterized_structure_is_not_a_head_end_structure():
    """§22.1.2.7's first sentence, and the one property separating the two structures a
    `REPLACE` names: the `WITH` structure **shall** be parameterized (§22.1.2.2) and the
    `INSERT AT HEAD` structure **shall not** be.

    So `#Length-prefixed` is a legal `WITH` and an illegal `INSERT AT HEAD`, and the same name
    in the two positions is the sharpest test of it.
    """
    _replace_refuses("22.1.2.7",
                     "REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY lp-object "
                     "INSERT AT HEAD #Length-prefixed")


def test_insert_at_head_still_may_not_ride_on_replace_structure():
    """§22.1.2.8 outlives the change that made §22.1.2.7 readable.

    Worth its own test because the two used to be indistinguishable from outside: `REPLACE
    STRUCTURE ... INSERT AT HEAD` was refused by §22.1.2.7 for being unreadable, and now gets
    as far as §22.1.2.8, which forbids the *combination*. A rule that only ever fired behind
    another one is a rule nobody has checked.
    """
    try:
        parse_module(_replace_module_with_head(
            "REPLACE STRUCTURE WITH #Length-prefixed ENCODED BY lp-object "
            "INSERT AT HEAD Head-structure"))
    except Asn1Error as error:
        assert "22.1.2.8" in str(error), str(error)
    else:
        raise AssertionError("REPLACE STRUCTURE accepted an INSERT AT HEAD")


def test_a_head_end_structure_no_module_declares_is_named_in_the_refusal():
    """The reference resolves against `structures`, so a typo is a missing name rather than an
    unsupported clause — which is what it looked like before.
    """
    _replace_refuses("22.1.2.7",
                     "REPLACE ALL COMPONENTS WITH #Length-prefixed ENCODED BY lp-object "
                     "INSERT AT HEAD Hed-structure")
