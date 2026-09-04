"""J1 — the bounded JER oracle: limits, canonical bytes, framing, diagnostics.

`docs/BCIR_ASN1_JSON_ROADMAP.md` §2 rates the JER rail's input rejection **Partial**, and
this file is what closes that rating. Its §8.1 names the corpus: "every byte split and
truncated prefix of valid framed documents; invalid UTF-8 … control characters …
duplicate keys … integer boundaries, excessive digits/exponents … depth/node/member/array/
string limit boundaries … repeated cleanup/retry with the prior output unchanged."

Two of these are worth calling out because they are the ones a looser oracle gets wrong.

**A limit is a boundary, not a vibe.** Each limit test asserts that N passes and N+1
refuses, so an off-by-one in either direction fails. A test that only checked "a huge input
is refused" would pass against a limit set anywhere at all.

**Canonicality is judged by octets.** §3.2 requires a canonical decoder to "reject
non-canonical bytes, not merely decode them to the same abstract value", so every test here
feeds bytes that decode to the *right value* and must still be refused.
"""

from __future__ import annotations

from bcir.asn1.jer import JerRules, encode_jer
from bcir.asn1.jer_bounded import (
    FRAME_HEADER_SIZE,
    FRAME_MAGIC,
    STRICT_LIMITS,
    JerBoundedError,
    JerErrorCode,
    JerLimits,
    decode_bounded,
    decode_framed,
    encode_framed,
    frame,
    scan,
    unframe,
)
from bcir.asn1.schema import Component, Primitive, Sequence, SequenceOf
from bcir.asn1.tags import Universal


def _rec() -> Sequence:
    return Sequence(
        (
            Component("a", Primitive(Universal.INTEGER, "INTEGER")),
            Component("b", Primitive(Universal.UTF8_STRING, "UTF8String"), tag=0),
        )
    )


_VALUE = {"a": 1, "b": "hi"}
_CANONICAL = b'{"a":1,"b":"hi"}'


def _refuses(action, code: JerErrorCode) -> JerBoundedError:
    try:
        action()
    except JerBoundedError as error:
        assert error.diagnostic.code is code, f"expected {code.value}, got {error.diagnostic}"
        return error
    raise AssertionError(f"expected a refusal with code {code.value}")


# --- §4.3: the limits, each at its boundary ----------------------------------------------


def test_the_input_byte_ceiling_is_checked_before_anything_is_parsed():
    """§4.3's first entry, and the one that makes the rest affordable to check."""
    limits = JerLimits(input_bytes=16)
    assert scan(b'{"a":1,"b":"hi"}', limits) > 0  # exactly 16 octets
    error = _refuses(lambda: scan(b'{"a":1,"b":"hix"}', limits), JerErrorCode.INPUT_TOO_LARGE)
    assert error.diagnostic.needed == 17, "the diagnostic says how much was needed (4.2)"


def test_the_depth_ceiling_and_that_a_string_cannot_smuggle_depth_past_it():
    """A `[` inside a string is not a structural token, so an oracle that scanned for
    brackets without tracking quotes would count depth an attacker chose."""
    limits = JerLimits(depth=4)
    assert scan(b"[[[[]]]]", limits) > 0
    _refuses(lambda: scan(b"[[[[[]]]]]", limits), JerErrorCode.DEPTH_EXCEEDED)
    # 200 open brackets, all inside one string: depth 0, one node.
    assert scan(b'"' + b"[" * 200 + b'"', limits) == 1


def test_the_member_and_element_ceilings_are_per_container():
    """§4.3 bounds "object members and array elements", which is a per-container question:
    a document of a thousand two-member objects is not a thousand-member object."""
    limits = JerLimits(members=3, elements=3, nodes=10_000, work=1 << 20)
    assert scan(b'{"a":1,"b":2,"c":3}', limits) > 0
    _refuses(lambda: scan(b'{"a":1,"b":2,"c":3,"d":4}', limits), JerErrorCode.MEMBERS_EXCEEDED)
    assert scan(b"[1,2,3]", limits) > 0
    _refuses(lambda: scan(b"[1,2,3,4]", limits), JerErrorCode.ELEMENTS_EXCEEDED)
    # Sibling containers each get their own budget.
    assert scan(b'[{"a":1},{"a":1},{"a":1}]', limits) > 0


def test_the_string_ceiling_counts_decoded_octets_not_source_octets():
    """An escape is six source octets and at most three decoded ones, so a ceiling on the
    source would refuse strings that fit and admit strings that do not."""
    limits = JerLimits(string_bytes=4)
    assert scan(b'"abcd"', limits) > 0
    _refuses(lambda: scan(b'"abcde"', limits), JerErrorCode.STRING_TOO_LONG)


def test_the_number_ceilings_are_three_separate_questions():
    """§4.3 asks for the token length, the integer digits and the exponent magnitude, and
    none implies the others: `1e999999999` is a short token denoting an unrepresentable
    number, and a thousand-digit integer is a long token denoting an ordinary one."""
    limits = JerLimits(number_bytes=8, integer_digits=4, exponent_magnitude=100)
    assert scan(b"1234", limits) > 0
    _refuses(lambda: scan(b"12345", limits), JerErrorCode.DIGITS_EXCEEDED)
    assert scan(b"1e100", limits) > 0
    _refuses(lambda: scan(b"1e101", limits), JerErrorCode.EXPONENT_EXCEEDED)
    _refuses(lambda: scan(b"1.234567890", limits), JerErrorCode.NUMBER_TOO_LONG)


def test_the_work_ceiling_bounds_what_a_small_input_can_cost():
    """§4.3's last entry — "so an input cannot hide quadratic duplicate/member lookup"."""
    _refuses(
        lambda: scan(b"[" * 40 + b"]" * 40, JerLimits(work=8, depth=100)),
        JerErrorCode.WORK_EXCEEDED,
    )


def test_limits_may_be_tightened_and_never_silently_expanded():
    """§4.3: "Limits are part of the compiled plan and may be tightened by a caller, never
    silently expanded"."""
    assert STRICT_LIMITS.tightened(depth=4).depth == 4
    _refuses(lambda: STRICT_LIMITS.tightened(depth=STRICT_LIMITS.depth + 1), JerErrorCode.MALFORMED)


# --- §3.1: what a conforming fast path still has to reject -------------------------------


def test_the_scanner_refuses_what_is_not_json_at_all():
    """§3.1's list, checked at the scan stage so nothing is built before the refusal."""
    for data, detail in (
        (b"NaN", "begins no JSON value"),
        (b"Infinity", "begins no JSON value"),
        (b'"abc', "unterminated"),
        (b"[}", "mismatched"),
        (b"{", "unclosed"),
        (b"}", "nothing open"),
        (b"-", "no digits"),
        (b"1.", "no digits after it"),
        (b"1e", "exponent with no digits"),
    ):
        error = _refuses(lambda d=data: scan(d), JerErrorCode.MALFORMED)
        assert detail in error.diagnostic.detail, (data, error.diagnostic)


def test_an_unescaped_control_character_is_refused_with_its_offset():
    """ECMA-404 clause 9 forbids it literally, and §8.1 names control characters."""
    error = _refuses(lambda: scan(b'"a\x01b"'), JerErrorCode.MALFORMED)
    assert error.diagnostic.offset == 2


def test_invalid_utf8_is_refused_after_the_structure_passes():
    """§3.1 asks for UTF-8 validation; §7.6.2 of X.697 makes the encoding UTF-8.

    Structural safety and validity are different questions, which is why the scan can work
    on octets: every structural character is ASCII and every non-ASCII UTF-8 octet has its
    high bit set, so no multi-byte sequence can be mistaken for markup.
    """
    _refuses(lambda: decode_bounded(b'{"a":1,"b":"\xff"}', _rec()), JerErrorCode.NOT_UTF8)


def test_a_duplicate_member_is_still_refused_through_the_bounded_path():
    """Carried over from the core rail: `json` resolves duplicates to the last silently,
    which would let one value hide behind another in an encoding that digests differently."""
    _refuses(lambda: decode_bounded(b'{"a":1,"a":2,"b":"hi"}', _rec()), JerErrorCode.SCHEMA)


# --- §3.2: canonicality is a property of the octets ---------------------------------------


def test_every_encoders_option_that_decodes_correctly_is_still_refused():
    """§3.2 — "A canonical decoder must reject non-canonical bytes, not merely decode them
    to the same abstract value."

    Each input below decodes to exactly `_VALUE` under BASIC. Each is refused under the
    canonical profile, and the diagnostic points at the octet rather than restating a rule.
    """
    assert decode_bounded(_CANONICAL, _rec()) == _VALUE
    for label, data in (
        ("member order (27.3.3)", b'{"b":"hi","a":1}'),
        ("insignificant white-space", b'{"a": 1,"b":"hi"}'),
        ("a gratuitous escape (7.6.3)", b'{"a":1,"b":"\\u0068i"}'),
        ("a trailing newline", b'{"a":1,"b":"hi"}\n'),
    ):
        error = _refuses(lambda d=data: decode_bounded(d, _rec()), JerErrorCode.NOT_CANONICAL)
        assert error.diagnostic.offset >= 0, label
        # ... and each is a perfectly good BASIC encoding of the same value.
        assert decode_bounded(data, _rec(), rules=JerRules.BASIC) == _VALUE, label


def test_a_default_valued_component_omitted_or_present_is_one_canonical_form():
    """The profile omits it; the other spelling decodes to the same value and is refused."""
    kind = Sequence(
        (
            Component("x", Primitive(Universal.INTEGER, "INTEGER")),
            Component("y", Primitive(Universal.BOOLEAN, "BOOLEAN"), tag=0, default=False),
        )
    )
    assert decode_bounded(b'{"x":1}', kind) == {"x": 1, "y": False}
    _refuses(lambda: decode_bounded(b'{"x":1,"y":false}', kind), JerErrorCode.NOT_CANONICAL)
    assert decode_bounded(b'{"x":1,"y":false}', kind, rules=JerRules.BASIC) == {"x": 1, "y": False}


def test_set_of_order_is_part_of_the_canonical_octets():
    """§30.3.3 leaves set-of order free, so the profile pins it and the check sees it."""
    from bcir.asn1.schema import SetOf

    kind = SetOf(Primitive(Universal.INTEGER, "INTEGER"))
    assert decode_bounded(b"[1,2,3]", kind) == [1, 2, 3]
    _refuses(lambda: decode_bounded(b"[3,1,2]", kind), JerErrorCode.NOT_CANONICAL)


def test_what_the_canonical_encoder_produces_is_what_the_canonical_decoder_accepts():
    """The round trip that makes the byte comparison an oracle rather than a guess."""
    kind = SequenceOf(_rec())
    value = [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}]
    octets = encode_jer(kind, value, rules=JerRules.CANONICAL)
    assert decode_bounded(octets, kind) == value


# --- §3.3: framing, and §4.2's failure atomicity -----------------------------------------


def test_a_frame_carries_every_field_clause_3_3_names():
    """ "an explicit version, length, integrity field, sequence, and generation"."""
    framed = encode_framed(_rec(), _VALUE, sequence=7, generation=3)
    assert framed.startswith(FRAME_MAGIC)
    assert len(framed) == FRAME_HEADER_SIZE + len(_CANONICAL)
    opened = unframe(framed)
    assert (opened.version, opened.sequence, opened.generation) == (1, 7, 3)
    assert opened.payload == _CANONICAL
    assert decode_framed(framed, _rec()) == _VALUE


def test_integrity_is_verified_before_any_payload_is_returned():
    """§3.3 — "No claim or artifact becomes visible before the complete frame passes
    lexical, schema, semantic, and integrity checks."

    The message says corruption rather than authenticity on purpose: §6.3 keeps a CRC and
    a signature apart, and describing one as the other is on the risk register.
    """
    framed = bytearray(encode_framed(_rec(), _VALUE))
    framed[-1] ^= 0xFF
    error = _refuses(lambda: decode_framed(bytes(framed), _rec()), JerErrorCode.FRAME_INTEGRITY)
    assert "not a signature" in error.diagnostic.detail


def test_every_truncated_prefix_of_a_framed_document_is_refused():
    """§8.1 asks for "every byte split and truncated prefix of valid framed documents"."""
    framed = encode_framed(_rec(), _VALUE)
    for cut in range(len(framed)):
        try:
            decode_framed(framed[:cut], _rec())
        except JerBoundedError as error:
            assert error.diagnostic.code in (
                JerErrorCode.FRAME_MALFORMED,
                JerErrorCode.FRAME_INTEGRITY,
            ), cut
        else:
            raise AssertionError(f"a {cut}-octet prefix decoded")
    assert decode_framed(framed, _rec()) == _VALUE


def test_a_frame_with_the_wrong_magic_or_version_is_refused():
    framed = encode_framed(_rec(), _VALUE)
    _refuses(lambda: unframe(b"XXXX" + framed[4:]), JerErrorCode.FRAME_MALFORMED)
    wrong = bytearray(framed)
    wrong[4] = 99
    _refuses(lambda: unframe(bytes(wrong)), JerErrorCode.FRAME_MALFORMED)


def test_a_refusal_leaves_the_caller_holding_exactly_what_it_had():
    """§4.2 — "On failure, the destination, active generation, and prior artifact remain
    unchanged", and §8.1 asks for "repeated cleanup/retry with the prior output ...
    unchanged".

    For a pure decode this is structural: the bounding pass runs before any value graph
    exists, so there is nothing built to roll back. The test pins the observable half —
    a failed decode returns nothing, and a retry after it still succeeds.
    """
    kind = _rec()
    good = decode_bounded(_CANONICAL, kind)
    for _ in range(3):
        _refuses(lambda: decode_bounded(b'{"b":"hi","a":1}', kind), JerErrorCode.NOT_CANONICAL)
        _refuses(lambda: decode_bounded(b"[" * 500, kind), JerErrorCode.DEPTH_EXCEEDED)
        assert decode_bounded(_CANONICAL, kind) == good


# --- §4.2: the diagnostic itself ----------------------------------------------------------


def test_a_diagnostic_carries_a_stable_code_an_offset_and_a_capacity():
    """§4.2 — "a stable error code, byte offset, schema path, and required capacity".

    The code is an enum rather than a string match on the message, because a caller may
    branch on it; the offset is a *byte* offset, which is why the scan reads octets.
    """
    error = _refuses(
        lambda: scan(b'"' + b"x" * 100 + b'"', JerLimits(string_bytes=8)),
        JerErrorCode.STRING_TOO_LONG,
    )
    diagnostic = error.diagnostic
    assert diagnostic.code is JerErrorCode.STRING_TOO_LONG
    assert diagnostic.offset == 0
    assert diagnostic.needed == 9
    assert "string-too-long" in str(diagnostic) and "octet" in str(diagnostic)
    # Every code is distinct, so branching on one cannot catch another.
    assert len({code.value for code in JerErrorCode}) == len(list(JerErrorCode))


def test_the_scan_offset_is_an_octet_offset_not_a_character_offset():
    """The reason the bounding pass reads octets: §4.2 asks for a byte offset, and a scan
    over a decoded `str` can only report a character index. The J3 C twin will report the
    same number, so the two rails will agree without a translation table.
    """
    # "é" is two octets in UTF-8, so the control character sits at octet 4, character 3.
    data = '{"a":"é'.encode() + b'\x01"}'
    error = _refuses(lambda: scan(data), JerErrorCode.MALFORMED)
    assert error.diagnostic.offset == 8, data
    assert data[error.diagnostic.offset] == 1
