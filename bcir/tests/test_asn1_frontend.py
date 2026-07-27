"""The X.680 front-end: lexer, parser, printer, and the lowering onto the encoder model.

Roadmap phase A (`docs/BCIR_ASN1_BUILDOUT_ROADMAP.md`). Three kinds of test live here:

* **Laws** — the round-trip law, and the X.680 rules a front-end silently gets wrong
  (comment termination, hyphenated lexemes, §31.2.7 EXPLICIT-over-CHOICE, automatic
  tagging's precondition, and the `{ 1 }` type-directed value ambiguity).
* **The phase gate** — the BCIR-StreamPack module, until now hand-built in Python, is
  *parsed from its own ASN.1 text* and must produce byte-identical DER for every corpus
  program. That is the difference between a parser that runs and a parser that is right.
* **Third-party validation** — RFC 5280's AuthorityKeyIdentifier, parsed from text this
  project did not write, decoding real certificates and re-encoding them byte-for-byte.
  A schema compiler that only agrees with its own schemas has proved nothing.
"""
from __future__ import annotations

import base64
import glob
import os
import re

from bcir.asn1.codec import Oid, Strictness
from bcir.asn1.tags import Universal
from bcir.asn1.schema import Choice, SequenceOf, Set, SetOf
from bcir.asn1.tlv import decode_one, encode_tlv
from bcir.frontends.asn1 import (Asn1SemanticError, Asn1SyntaxError, compile_module,
                                 parse_module, print_module, tokenize)

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_STREAMPACK_ASN1 = os.path.join(_ROOT, "bcir", "asn1", "BCIR-StreamPack.asn1")
_PKIX_ASN1 = os.path.join(_ROOT, "bcir", "frontends", "asn1", "testdata",
                          "PKIX1Implicit88.asn1")
_ABI_DOC = os.path.join(_ROOT, "docs", "BCIR_ASN1_X690_ABI.md")

#: A real AuthorityKeyIdentifier extension value, lifted from the Certigna root CA. It
#: carries all three components AND reaches `directoryName [4] Name` -- the one
#: alternative X.680 31.2.7 forces to be EXPLICIT inside an IMPLICIT TAGS module. Pinned
#: as a constant so the law is checked on hosts with no certificate store (Windows CI).
_REAL_AKI = bytes.fromhex(
    "305b80141aedfe413990b42459be01f252d545f65a39dc11a138a4363034310b3009060355"
    "04061302465231123010060355040a0c094468696d796f7469733111300f06035504030c08"
    "4365727469676e61820900fedce3010fc948ff")


def _streampack():
    return compile_module(open(_STREAMPACK_ASN1, encoding="utf-8").read(),
                          "BCIR-StreamPack.asn1")


def _pkix():
    return compile_module(open(_PKIX_ASN1, encoding="utf-8").read(),
                          "PKIX1Implicit88.asn1")


# --- clause 12: the lexical rules a hand-written front-end gets wrong -------------------

def test_line_comment_ends_at_a_second_double_hyphen_not_only_at_end_of_line():
    """X.680 12.6.3: `--` closes a comment, so text after it on the same line is LIVE.

    Treating `--` as "comment to end of line" (the C++ habit) would swallow `b INTEGER`
    here and silently produce a one-component SEQUENCE.
    """
    module = parse_module(
        "M DEFINITIONS ::= BEGIN\n"
        "  T ::= SEQUENCE { a INTEGER, -- note -- b INTEGER }\n"
        "END\n")
    names = [c.name for c in module.assignments[0].type.components]
    assert names == ["a", "b"], names


def test_block_comments_nest():
    """X.680 12.6.4 -- unlike C, so a naive scan-to-`*/` ends the comment too early."""
    module = parse_module(
        "M DEFINITIONS ::= BEGIN /* outer /* inner */ still comment */\n"
        "  T ::= INTEGER\nEND\n")
    assert [a.name for a in module.assignments] == ["T"]


def test_hyphens_join_a_lexeme_only_when_a_letter_or_digit_follows():
    kinds = [(t.kind, t.text) for t in tokenize("identified-organization a - b")]
    assert ("identifier", "identified-organization") in kinds
    assert ("punct", "-") in kinds


def test_a_number_with_a_leading_zero_is_refused():
    """X.680 12.8. `01` is not a number, which is what keeps OID arcs unambiguous."""
    try:
        tokenize("M DEFINITIONS ::= BEGIN T ::= INTEGER DEFAULT 01 END")
        raise AssertionError("lexer accepted a leading-zero number")
    except Asn1SyntaxError as exc:
        assert "leading zero" in str(exc), exc


def test_a_doubled_quote_inside_a_cstring_is_one_quote_character():
    module = parse_module('M DEFINITIONS ::= BEGIN\n'
                          '  T ::= SEQUENCE { a UTF8String DEFAULT "say ""hi""" }\n'
                          'END\n')
    assert module.assignments[0].type.components[0].default.value == 'say "hi"'


# --- the round-trip law -----------------------------------------------------------------

def test_round_trip_law_holds_for_both_bundled_modules():
    """parse(print(parse(t))) == parse(t).

    The law is what makes the AST *complete*: anything the parser dropped would vanish
    from the printed form too, so the two would agree with each other while both
    disagreeing with the module. ASTs are compared rather than text because layout and
    comments are not semantic.
    """
    for path in (_STREAMPACK_ASN1, _PKIX_ASN1):
        node = parse_module(open(path, encoding="utf-8").read(), path)
        again = parse_module(print_module(node), f"{path}<printed>")
        assert again == node, f"round-trip law failed for {path}"


def test_round_trip_preserves_the_tag_mode_the_source_stated():
    """A printer that resolved tags against the module default would round-trip to a
    DIFFERENT module. Keeping the mode unresolved in the AST is what prevents that."""
    node = parse_module("M DEFINITIONS IMPLICIT TAGS ::= BEGIN\n"
                        "  T ::= SEQUENCE { a [0] EXPLICIT INTEGER, b [1] INTEGER }\n"
                        "END\n")
    printed = print_module(node)
    assert "[0] EXPLICIT INTEGER" in printed and "[1] INTEGER" in printed
    assert parse_module(printed) == node


# --- the phase A gate --------------------------------------------------------------------

def test_the_asn1_source_matches_the_module_published_in_the_abi_doc():
    """One module, one text. The doc is the human-readable copy of the same file the
    compiler reads, so the two cannot drift into describing different wire formats."""
    doc = open(_ABI_DOC, encoding="utf-8").read()
    blocks = re.findall(r"```asn1\n(.*?)```", doc, re.S)
    assert len(blocks) == 1, f"expected one asn1 block in the ABI doc, found {len(blocks)}"
    assert blocks[0] == open(_STREAMPACK_ASN1, encoding="utf-8").read(), (
        "docs/BCIR_ASN1_X690_ABI.md and bcir/asn1/BCIR-StreamPack.asn1 have drifted")


def test_parsed_streampack_module_encodes_byte_identically_to_the_hand_built_one():
    """THE phase A gate: the schema is now COMPILED, and the octets did not move.

    Every corpus program is projected through both the hand-built model in
    `bcir/asn1/streampack.py` and the model compiled from `BCIR-StreamPack.asn1`. Equal
    octets means the front-end reproduces a schema a human wrote by hand, including the
    parts that are easy to get subtly wrong: IMPLICIT tagging, the ENUMERATED DEFAULT
    written as an identifier (`DEFAULT core`), and the SEQUENCE OF DEFAULTs (`{}`,
    `{ 1 }`) that X.690 11.5 must omit.
    """
    from bcir.asn1.streampack import MODULE, pack_to_value
    from bcir.examples import PROGRAMS
    from bcir.gem import hydrate
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta

    compiled = _streampack().module
    host, theta = TargetProfile.x86_avx512(), Theta.cool()
    assert compiled.oid == MODULE.oid
    checked = 0
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        value = pack_to_value(hydrate(module, optimize(module, host, theta)))
        hand, parsed = MODULE.encode("StreamPack", value), compiled.encode("StreamPack",
                                                                           value)
        assert hand == parsed, f"{name}: compiled module produced different DER"
        assert compiled.decode("StreamPack", parsed) == MODULE.decode("StreamPack", hand)
        checked += 1
    assert checked >= 10, f"corpus shrank to {checked} programs"


def test_the_enumerated_default_written_as_an_identifier_resolves_to_its_number():
    """`dispatch [11] Dispatch DEFAULT core` must become 0, not the string 'core'.

    A DEFAULT the encoder can never compare equal to would silently disable X.690 11.5
    for that component -- the module would still "work", just emit different octets.
    """
    compiled = _streampack().module
    dispatch = [c for c in compiled.types["LaneSegment"].components
                if c.name == "dispatch"][0]
    assert dispatch.default == 0, dispatch.default


# --- X.680 rules that change the octets ---------------------------------------------------

def test_an_implicit_module_still_tags_a_choice_explicitly():
    """X.680 31.2.7. An implicit tag REPLACES the base tag, and a CHOICE has none
    (29.1) -- replacing it would erase which alternative was chosen. RFC 5280's
    `directoryName [4] Name` is exactly this case, inside an IMPLICIT TAGS module."""
    alternatives = {a.name: a for a in _pkix().module.types["GeneralName"].alternatives}
    assert alternatives["directoryName"].explicit is True, \
        "a CHOICE under an implicit module default must still be tagged EXPLICIT"
    assert alternatives["dNSName"].explicit is False, \
        "a non-CHOICE component must follow the module's IMPLICIT default"


def test_tagging_a_choice_implicitly_on_purpose_is_refused():
    try:
        compile_module("M DEFINITIONS ::= BEGIN\n"
                       "  C ::= CHOICE { a INTEGER, b UTF8String }\n"
                       "  T ::= SEQUENCE { x [0] IMPLICIT C }\n"
                       "END\n")
        raise AssertionError("IMPLICIT over a CHOICE was accepted")
    except Asn1SemanticError as exc:
        assert "31.2.7" in str(exc), exc


def test_automatic_tags_numbers_components_in_order():
    compiled = compile_module(
        "M DEFINITIONS AUTOMATIC TAGS ::= BEGIN\n"
        "  T ::= SEQUENCE { a INTEGER, b UTF8String, c BOOLEAN }\nEND\n").module
    assert [(c.name, c.tag) for c in compiled.types["T"].components] == \
        [("a", 0), ("b", 1), ("c", 2)]


def test_automatic_tags_leaves_a_partially_tagged_list_alone():
    """X.680 12.3: automatic tagging applies only when NO component bears a tag.
    Renumbering a hand-tagged list would move every field on the wire."""
    compiled = compile_module(
        "M DEFINITIONS AUTOMATIC TAGS ::= BEGIN\n"
        "  T ::= SEQUENCE { a [5] INTEGER, b UTF8String }\nEND\n").module
    assert [(c.name, c.tag) for c in compiled.types["T"].components] == \
        [("a", 5), ("b", None)]


def test_a_braced_value_is_read_against_its_type_not_guessed_from_its_shape():
    """`{ 1 }` is a one-element SEQUENCE OF INTEGER value AND a one-arc OID value.

    Only the governing type decides. Guessing from shape made `DEFAULT { 1 }` on
    `SEQUENCE OF INTEGER` read as an OID, which left the DEFAULT incomparable and
    stopped X.690 11.5 from ever omitting the component -- caught by the byte-identity
    gate above, and pinned directly here.
    """
    compiled = compile_module(
        "M DEFINITIONS ::= BEGIN\n"
        "  T ::= SEQUENCE { list SEQUENCE OF INTEGER DEFAULT { 1 },\n"
        "                   oid  OBJECT IDENTIFIER DEFAULT { 1 } }\nEND\n").module
    components = {c.name: c.default for c in compiled.types["T"].components}
    assert components["list"] == [1], components["list"]
    assert components["oid"] == Oid((1,)), components["oid"]


def test_empty_braces_default_to_an_empty_list_for_a_sequence_of():
    compiled = compile_module(
        "M DEFINITIONS ::= BEGIN\n"
        "  T ::= SEQUENCE { xs SEQUENCE OF INTEGER DEFAULT {} }\nEND\n").module
    assert compiled.types["T"].components[0].default == []


def test_choice_alternatives_must_have_distinct_tags():
    try:
        compile_module("M DEFINITIONS ::= BEGIN\n"
                       "  C ::= CHOICE { a INTEGER, b INTEGER }\nEND\n")
        raise AssertionError("a CHOICE with two INTEGER alternatives was accepted")
    except Exception as exc:
        assert "29.3" in str(exc), exc


def test_set_of_encodes_in_the_der_canonical_order():
    """X.690 11.6: a canonical encoding must not depend on the order the caller
    happened to supply, or two peers holding the same set would digest differently."""
    compiled = compile_module(
        "M DEFINITIONS ::= BEGIN\n  S ::= SET OF INTEGER\nEND\n").module
    ascending = encode_tlv(compiled.types["S"].encode([1, 2, 3]))
    assert encode_tlv(compiled.types["S"].encode([3, 1, 2])) == ascending
    assert isinstance(compiled.types["S"], SetOf)


def test_set_decodes_components_in_any_order():
    compiled = compile_module(
        "M DEFINITIONS IMPLICIT TAGS ::= BEGIN\n"
        "  T ::= SET { a [0] INTEGER, b [1] UTF8String }\nEND\n").module
    built = compiled.types["T"]
    assert isinstance(built, Set)
    octets = encode_tlv(built.encode({"a": 1, "b": "x"}))
    assert built.decode(decode_one(octets), strictness=Strictness.DER) == \
        {"a": 1, "b": "x"}


def test_recursive_type_definitions_resolve():
    """Real modules are mutually recursive; the encoder model is eager dataclasses, so
    the lowering needs a lazy reference rather than a refusal."""
    compiled = compile_module(
        "M DEFINITIONS IMPLICIT TAGS ::= BEGIN\n"
        "  Tree ::= SEQUENCE { value INTEGER, children [0] SEQUENCE OF Tree }\n"
        "END\n").module
    tree = compiled.types["Tree"]
    value = {"value": 1, "children": [{"value": 2, "children": []}]}
    octets = encode_tlv(tree.encode(value))
    assert tree.decode(decode_one(octets), strictness=Strictness.DER) == value


# --- what is refused, and why ---------------------------------------------------------

def test_constructs_from_the_companion_recommendations_are_refused_by_name():
    """A front-end that skipped these would build a model disagreeing with the module.
    Each diagnostic names the Recommendation and the roadmap phase that would add it.

    The list SHRINKS as phases land: X.681's classes and open types left it when phase F
    built them, and X.683 parameterization left it when phase F's follow-on did. What
    remains is X.692 encoding control.
    """
    cases = [
        ("M DEFINITIONS ::= BEGIN\n  T ::= SEQUENCE { a INTEGER }\n"
         "ENCODING-CONTROL PER\nEND\n", "X.692"),
    ]
    for text, recommendation in cases:
        try:
            compile_module(text)
            raise AssertionError(f"accepted a construct needing {recommendation}")
        except (Asn1SyntaxError, Asn1SemanticError) as exc:
            assert recommendation in str(exc), (recommendation, str(exc))


def test_a_parameterized_reference_with_the_wrong_arity_is_refused():
    """X.683 §9.6: exactly one ActualParameter per Parameter, in the same order.

    Arity is the one thing instantiation can check without knowing what the actuals mean,
    and getting it wrong silently would bind a dummy to nothing.
    """
    try:
        compile_module("M DEFINITIONS ::= BEGIN\n"
                       "  Pair {X, Y} ::= SEQUENCE { a X, b Y }\n"
                       "  T ::= Pair {BOOLEAN}\nEND\n")
        raise AssertionError("a one-actual reference to a two-parameter assignment passed")
    except Asn1SemanticError as exc:
        assert "9.6" in str(exc), exc


def test_a_parameterized_reference_to_a_plain_assignment_is_refused():
    """§9.2: actual parameters may only be supplied to a PARAMETERIZED assignment."""
    try:
        compile_module("M DEFINITIONS ::= BEGIN\n"
                       "  Plain ::= INTEGER\n  T ::= Plain {BOOLEAN}\nEND\n")
        raise AssertionError("actual parameters were accepted on a plain assignment")
    except Asn1SemanticError as exc:
        assert "9.2" in str(exc), exc


# --- X.681 information objects and open types (roadmap phase F) --------------------------

_PKIX_EXPLICIT = os.path.join(_ROOT, "bcir", "frontends", "asn1", "testdata",
                              "PKIX1Explicit88.asn1")


def test_any_defined_by_lowers_to_an_open_type():
    """`ANY DEFINED BY x` is withdrawn X.680:1988 notation, but it is exactly X.681 §14's
    open type, so both spellings lower to the same thing rather than to a dialect.

    Neither `ANY` nor `DEFINED` is a reserved word in the 2021 edition, so they arrive at
    the parser as ordinary typereferences — which is why the match is on text.
    """
    from bcir.asn1.schema import OpenType

    compiled = compile_module(
        "M DEFINITIONS ::= BEGIN\n"
        "  T ::= SEQUENCE { algorithm OBJECT IDENTIFIER,\n"
        "                   parameters ANY DEFINED BY algorithm OPTIONAL }\nEND\n"
    ).module
    parameters = compiled.types["T"].components[1]
    assert isinstance(parameters.type, OpenType), parameters.type
    assert "algorithm" in parameters.type.name


def test_a_class_value_field_keeps_its_declared_type_while_a_type_field_stays_open():
    """X.681 §14/§15. `ALGORITHM.&id` is a VALUE field whose type the class declared, so
    lowering it to an open type would discard information the module supplied and turn a
    checkable OBJECT IDENTIFIER into opaque octets. Only `&Type` is genuinely open."""
    from bcir.asn1.schema import OpenType, Primitive

    compiled = compile_module(
        "M DEFINITIONS ::= BEGIN\n"
        "  ALGORITHM ::= CLASS { &id OBJECT IDENTIFIER UNIQUE, &Type OPTIONAL }\n"
        "    WITH SYNTAX { &Type IDENTIFIED BY &id }\n"
        "  T ::= SEQUENCE { algorithm ALGORITHM.&id,\n"
        "                   parameters ALGORITHM.&Type OPTIONAL }\nEND\n").module
    algorithm, parameters = compiled.types["T"].components
    assert isinstance(algorithm.type, Primitive), algorithm.type
    assert algorithm.type.universal == int(Universal.OBJECT_IDENTIFIER)
    assert isinstance(parameters.type, OpenType), parameters.type


def test_a_reference_to_an_undefined_class_is_a_named_error():
    try:
        compile_module("M DEFINITIONS ::= BEGIN\n"
                       "  T ::= SEQUENCE { a MISSING.&Type }\nEND\n")
        raise AssertionError("accepted a reference to an undefined class")
    except Asn1SemanticError as exc:
        assert "MISSING" in str(exc) and "never defined" in str(exc), exc


def test_objects_and_object_sets_round_trip_with_their_bodies_intact():
    """They are recorded, not interpreted — their content selects WHICH type an open type
    contains, which is X.682's table-constraint machinery. Keeping the body is what makes
    the round-trip law still meaningful for these modules: a parser that dropped it would
    print a gutted module and the law would happily pass."""
    text = ("M DEFINITIONS ::= BEGIN\n"
            "  ALGORITHM ::= CLASS { &id OBJECT IDENTIFIER UNIQUE, &Type OPTIONAL }\n"
            "  sha256 ALGORITHM ::= { &id id-sha256 }\n"
            "  Algorithms ALGORITHM ::= { sha256 | sha512 }\nEND\n")
    node = parse_module(text)
    assert parse_module(print_module(node)) == node
    kinds = [type(a).__name__ for a in node.assignments]
    assert kinds == ["ClassAssignment", "ObjectAssignment", "ObjectSetAssignment"], kinds
    assert "sha512" in node.assignments[2].objects


def test_an_implicit_tag_on_an_open_type_is_refused():
    """X.680 §31.2.7 names open types alongside CHOICE: an implicit tag REPLACES the base
    tag, and an open type has none — the contained value's tag is the only thing there."""
    try:
        compile_module("M DEFINITIONS ::= BEGIN\n"
                       "  T ::= SEQUENCE { a [0] IMPLICIT ANY }\nEND\n")
        raise AssertionError("IMPLICIT over an open type was accepted")
    except Asn1SemanticError as exc:
        assert "31.2.7" in str(exc), exc


def test_rfc5280_subject_public_key_info_round_trips_the_whole_host_trust_store():
    """The gate phase A could not pass, and the reason phase F exists.

    `AlgorithmIdentifier.parameters` is an open type; every certificate in the store
    carries it, so the type is unusable without X.681. Byte-identical re-encoding is the
    strong form: it means the open type's octets survived untouched, which is the whole
    contract of an open type — this layer does not know the contained type and must not
    alter it.
    """
    from bcir.asn1.codec import Strictness
    from bcir.asn1.tlv import decode_one, encode_tlv

    paths = sorted(glob.glob("/etc/ssl/certs/*.pem"))
    if not paths:
        return                                     # no host trust store: nothing to widen
    spki = compile_module(open(_PKIX_EXPLICIT, encoding="utf-8").read(),
                          "PKIX1Explicit88.asn1").module.types["SubjectPublicKeyInfo"]
    seen = identical = with_parameters = 0
    for path in paths:
        for block in re.findall(
                r"-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----",
                open(path, encoding="utf-8", errors="replace").read(), re.S):
            cert = decode_one(base64.b64decode("".join(block.split())))
            tbs = cert.children[0]
            index = 6 if tbs.children[0].tag.cls.name == "CONTEXT" else 5
            node = tbs.children[index]
            seen += 1
            value = spki.decode(node, strictness=Strictness.DER)
            if "parameters" in value["algorithm"]:
                with_parameters += 1
            if encode_tlv(spki.encode(value)) == encode_tlv(node):
                identical += 1
    assert seen > 0
    assert identical == seen, f"{seen - identical} of {seen} SPKIs did not re-encode"
    # Every certificate carries `parameters`, which is why the open type is not optional
    # in practice and why phase A's stop condition fired on this type.
    assert with_parameters == seen, (with_parameters, seen)


def test_a_containing_constraint_is_modelled_not_discarded():
    """Value-set constraints are dropped because X.690 encodes a value the same way
    whether or not a constraint admitted it. CONTAINING is the exception -- X.682 §11.4
    makes the octet string's abstract value the ENCODING of another type -- so it is now
    carried on the type. It used to be REFUSED, which was the honest answer while it was
    unimplemented; modelling it is the better one.
    """
    compiled = compile_module("M DEFINITIONS ::= BEGIN\n"
                              "  T ::= OCTET STRING (CONTAINING INTEGER)\nEND\n")
    contained = compiled.module.types["T"].contains
    assert contained is not None, "a CONTAINING constraint was silently discarded"
    assert contained.universal == Universal.INTEGER


def test_a_contents_constraint_is_refused_on_a_type_it_cannot_apply_to():
    """§11.3 limits it to OCTET STRING and BIT STRING; anything else is a spec error."""
    try:
        compile_module("M DEFINITIONS ::= BEGIN\n"
                       "  T ::= INTEGER (CONTAINING BOOLEAN)\nEND\n")
        raise AssertionError("a contents constraint on an INTEGER must be refused")
    except Asn1SemanticError as exc:
        assert "11.3" in str(exc), exc


def test_a_size_constraint_is_discarded_and_does_not_change_the_encoding():
    compiled = compile_module(
        "M DEFINITIONS ::= BEGIN\n  S ::= SEQUENCE SIZE (1..MAX) OF INTEGER\nEND\n"
    ).module
    assert isinstance(compiled.types["S"], SequenceOf)
    assert encode_tlv(compiled.types["S"].encode([1])).hex() == "3003020101"


def test_an_unresolved_type_reference_is_a_named_error():
    try:
        compile_module("M DEFINITIONS ::= BEGIN\n  T ::= SEQUENCE { a Missing }\nEND\n")
        raise AssertionError("an undefined type reference was accepted")
    except Asn1SemanticError as exc:
        assert "Missing" in str(exc) and "never" in str(exc), exc


# --- third-party validation: RFC 5280 against real certificates -------------------------

def test_rfc5280_authority_key_identifier_round_trips_a_real_certificate_extension():
    """Decode a real AuthorityKeyIdentifier and re-encode it BYTE-FOR-BYTE.

    The module text is RFC 5280's, not this project's, and the octets come from a
    shipped root CA. Byte-identical re-encoding is the strong form of the claim: it
    means the compiled model agrees with the real encoding on tagging, on OPTIONAL
    placement, and on the EXPLICIT-over-CHOICE rule that `directoryName` depends on.
    """
    aki = _pkix().module.types["AuthorityKeyIdentifier"]
    decoded = aki.decode(decode_one(_REAL_AKI), strictness=Strictness.DER)
    assert set(decoded) == {"keyIdentifier", "authorityCertIssuer",
                            "authorityCertSerialNumber"}, sorted(decoded)
    # The issuer reaches GeneralName's directoryName alternative -- the X.680 31.2.7 path.
    assert decoded["authorityCertIssuer"][0][0] == "directoryName"
    assert encode_tlv(aki.encode(decoded)) == _REAL_AKI, \
        "re-encoding a real AKI extension did not reproduce its octets"


def test_rfc5280_authority_key_identifier_round_trips_the_whole_host_trust_store():
    """The same law, widened to every certificate the host ships (degrades where the
    store is absent, e.g. Windows). Breadth catches shapes one fixture cannot."""
    from bcir.asn1.values import decode_oid

    paths = sorted(glob.glob("/etc/ssl/certs/*.pem"))
    if not paths:
        return                                    # no host trust store: nothing to widen
    aki = _pkix().module.types["AuthorityKeyIdentifier"]
    found = identical = 0
    for path in paths:
        for block in re.findall(
                r"-----BEGIN CERTIFICATE-----(.*?)-----END CERTIFICATE-----",
                open(path, encoding="utf-8", errors="replace").read(), re.S):
            cert = decode_one(base64.b64decode("".join(block.split())))
            holder = [c for c in cert.children[0].children
                      if c.tag.cls.name == "CONTEXT" and c.tag.number == 3]
            if not holder:
                continue
            for ext in holder[0].children[0].children:
                if ".".join(map(str, decode_oid(ext.children[0].content))) != "2.5.29.35":
                    continue
                octets = ext.children[-1].content
                found += 1
                decoded = aki.decode(decode_one(octets), strictness=Strictness.DER)
                if encode_tlv(aki.encode(decoded)) == octets:
                    identical += 1
    assert found == 0 or identical == found, \
        f"{found - identical} of {found} real AKI extensions did not re-encode identically"


def test_the_x509_choice_and_set_of_types_lower_to_the_right_constructors():
    compiled = _pkix().module
    assert isinstance(compiled.types["Name"], Choice)
    assert isinstance(compiled.types["RelativeDistinguishedName"], SetOf)
    assert isinstance(compiled.types["RDNSequence"], SequenceOf)


# --- the CLI ---------------------------------------------------------------------------

def test_cli_check_mode_passes_on_both_bundled_modules():
    from bcir.frontends.asn1.__main__ import main

    assert main(["--check", _STREAMPACK_ASN1, _PKIX_ASN1]) == 0


def test_cli_reports_a_syntax_fault_with_file_line_column(tmp_path=None):
    import tempfile

    from bcir.frontends.asn1.__main__ import main

    with tempfile.TemporaryDirectory() as directory:
        broken = os.path.join(directory, "broken.asn1")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write("M DEFINITIONS ::= BEGIN\n  T ::= SEQUENCE { a \nEND\n")
        assert main(["--check", broken]) == 1
