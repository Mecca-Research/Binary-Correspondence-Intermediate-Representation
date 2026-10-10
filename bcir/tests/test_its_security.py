"""The ITS security-envelope study (docs/research/BCIR_ITS_ASN1_BINARY_STUDY.md).

Three things are pinned here, each by numbers rather than by round trips:

* the RECONSTRUCTION of TS 103 097 V1.1.1's binary rules, held to the paper's own published
  lengths (Table VI's binary column) -- every one of the four, exactly;
* the SIZE results the study reports: the same values under BCIR's standard ASN.1 rules, on
  the one-to-one transcription and on the TMSAO schema;
* the ACCOUNTING that explains them: every bit of every envelope attributed to content or to
  one kind of framing, summing exactly to the encoding's length.
"""

from __future__ import annotations

from bcir.asn1 import its_security as its
from bcir.asn1 import its_study as st
from bcir.asn1.tags import Asn1Error


def _refused(fn, *args) -> str:
    try:
        fn(*args)
    except Asn1Error as error:
        return str(error)
    raise AssertionError(f"{fn.__name__}{args!r} was accepted")


# --- the reconstruction ---------------------------------------------------------------------


def test_the_reconstruction_reproduces_every_binary_length_the_paper_published():
    """Table VI's binary column, all four entries. The reconstruction has free choices (field
    order, which header carries what, the certificate's attributes) and these four numbers are
    what pin them -- a reconstruction matching three of four would be a different format."""
    for profile, value in its.envelopes().items():
        assert len(its.encode_binary(value)) == its.PAPER_TABLE_VI["binary"][profile], profile
    # A minimal V1.1.1 authorization ticket is 133 octets: the cert cases' one free length.
    assert len(its.encode_binary(its.authorization_ticket(), "Certificate")) == 133


def test_every_envelope_round_trips_through_the_binary_rules():
    for profile, value in its.envelopes().items():
        assert its.decode_binary(its.encode_binary(value)) == value, profile


def test_intx_is_the_shortest_spelling_and_refuses_every_other():
    """IntX: n leading one bits, a zero, then 7 + 7n value bits. The decoder refuses a longer
    spelling of a value the shorter one carries (Class A: two octet strings for one value)."""
    vectors = {
        0: "00",
        127: "7f",
        128: "8080",
        16383: "bfff",
        16384: "c04000",
        (1 << 56) - 1: "feffffffffffffff",
    }
    for value, octets in vectors.items():
        assert its.encode_intx(value).hex() == octets, value
        assert its.decode_intx(bytes.fromhex(octets), 0) == (value, len(octets) // 2)
    for bad in ("8000", "807f", "c00000", "ff00000000000000", "80", ""):
        _refused(its.decode_intx, bytes.fromhex(bad), 0)
    for value in (-1, 1 << 56, True):
        _refused(its.encode_intx, value)


def test_the_binary_decoder_refuses_every_malformed_envelope():
    raw = its.encode_binary(its.envelope("p1"))
    # Every proper prefix, an octet after the end, an unknown type code, an enum off the root.
    for cut in range(len(raw)):
        _refused(its.decode_binary, raw[:cut])
    _refused(its.decode_binary, raw + b"\x00")
    first_header_code = 3  # protocol_version, security_profile, a one-octet IntX, the code
    assert raw[first_header_code] == its.TYPE_CODES["HeaderField"]["signerInfo"]
    _refused(its.decode_binary, raw[:first_header_code] + b"\x07" + raw[first_header_code + 1 :])
    cert = its.encode_binary(its.authorization_ticket(), "Certificate")
    at = cert.index(bytes([1, 0]))  # subjectType authorizationTicket(1), empty subjectName
    _refused(its.decode_binary, cert[:at] + b"\x09" + cert[at + 1 :], "Certificate")


def test_the_binary_encoder_refuses_a_value_outside_its_field():
    value = its.envelope("p1")
    value["protocolVersion"] = 256
    assert "outside" in _refused(its.encode_binary, value)
    value["protocolVersion"] = True
    _refused(its.encode_binary, value)


# --- the size results -------------------------------------------------------------------------


def test_the_same_values_under_standard_rules_beat_binary_in_every_profile():
    """The study's first result: on the transcription -- V1.1.1's data model, converted one
    construct at a time -- canonical UPER is smaller than the binary encoding in all four
    profiles, where the paper's draft-schema UPER lost three of four."""
    sizes = st.sizes()
    assert [sizes["transcription-uper"][p] for p in its.PROFILES] == [92, 212, 223, 220]
    assert [sizes["binary-v111"][p] for p in its.PROFILES] == [96, 222, 233, 230]
    for p in its.PROFILES:
        assert sizes["transcription-uper"][p] < sizes["binary-v111"][p], p


def test_the_tmsao_schema_beats_binary_in_every_profile_under_uper_and_coer():
    sizes = st.sizes()
    assert [sizes["tmsao-uper"][p] for p in its.PROFILES] == [89, 206, 216, 214]
    assert [sizes["tmsao-coer"][p] for p in its.PROFILES] == [93, 216, 226, 224]
    assert [sizes["tmsao-per-uper"][p] for p in its.PROFILES] == [86, 203, 213, 212]
    for p in its.PROFILES:
        for name in ("tmsao-uper", "tmsao-coer", "tmsao-per-uper"):
            assert sizes[name][p] < sizes["binary-v111"][p], (name, p)


def test_the_paper_average_metric_reproduces_table_vii_and_ranks_the_columns():
    """Table VII's lower limit of the average CAM envelope (10 Hz, certificate at 1 Hz),
    recomputed from Table VI; then the same metric for the study's encodings."""
    avg = st.average_sizes()
    assert round(avg["paper-binary"], 1) == 108.6
    assert round(avg["paper-asn1_uper_draft"], 1) == 103.2
    assert round(avg["paper-protobuf"], 1) == 150.3
    assert round(avg["paper-exi"], 1) == 102.0
    assert round(avg["paper-exi_opt"], 1) == 98.4
    assert round(avg["tmsao-uper"], 1) == 100.7
    assert round(avg["tmsao-per-uper"], 1) == 97.7
    assert avg["tmsao-per-uper"] < avg["paper-exi_opt"] < avg["paper-binary"]


def test_kbcir_selects_the_studys_smallest_encoding_for_the_channel():
    """Phase H's selector (`bcir.asn1.selection`) on the envelope: legality first, canonical
    rules only, then the objective. For the paper's objective -- octets on the channel -- it
    selects canonical UNALIGNED PER on every profile, by exact arithmetic rather than a clock,
    at exactly the study's sizes; with no objective it keeps DER, the pinned degenerate case.
    The latency objectives read Python-oracle timings, so they are not pinned here."""
    from bcir.asn1 import selection as sel

    kind = its.tmsao_type("SecuredMessagePer")
    sizes = st.sizes()["tmsao-per-uper"]
    for profile in its.PROFILES:
        found = sel.measure(kind, its.to_tmsao(its.envelope(profile)), repeats=1)
        assert all(m.legal for m in found), profile  # every rule BCIR speaks carries the value
        chosen = sel.select(found, objective=sel.Objective.WIRE_SIZE)
        assert (chosen.candidate, chosen.octets) == ("CANONICAL-PER-UNALIGNED", sizes[profile])
        assert sel.select(found).candidate == "DER"


# --- the TMSAO schema carries exactly the transcription's values -------------------------------


def test_the_tmsao_field_order_is_v111s_canonical_order():
    """The SEQUENCE that carries a field list must order its components as V1.1.1's prose
    orders the list: signer information first among the headers, the rest by ascending code."""
    for choice in its.FIELD_SETS:
        codes = its.TYPE_CODES[choice]
        order = its.canonical_order(choice)
        assert set(order) == set(codes), choice
        rest = [n for n in order if n != "signerInfo"]
        assert rest == sorted(rest, key=codes.get), choice
        if "signerInfo" in order:
            assert order[0] == "signerInfo"


def test_the_tmsao_mapping_is_a_bijection_on_the_envelopes():
    for profile, value in its.envelopes().items():
        assert its.from_tmsao(its.to_tmsao(value)) == value, profile


def test_each_shaped_integers_root_and_additions_are_v111s_range():
    """The TMSAO module states V1.1.1's range in the schema: each shaped integer's root and
    additional set together are exactly the range the binary rules hold, at both edges
    (`Extensible.known`). The frontend used to drop the additional set, so the model held the
    root alone; the codecs still admit past the range (the relay posture, LangRef §17.3), and the
    mapping is what refuses a value outside it."""
    types = its.tmsao().module.types
    ranges = {
        "IntX": (0, 2**56 - 1),
        "SmallUint8": (0, 2**8 - 1),
        "ShapedTime64": (0, 2**64 - 1),
        "ShapedMessageType": (0, 2**16 - 1),
    }
    for name, (low, high) in ranges.items():
        constraint = types[name].constraint
        assert constraint.additions is not None, name
        known = [constraint.known(v) for v in (low - 1, low, high, high + 1)]
        assert known == [False, True, True, False], (name, known)
        assert constraint.permits(high + 1) and constraint.permits(low - 1), name


def test_the_tmsao_mapping_refuses_a_value_v111_cannot_carry():
    """A shaped integer is extensible, and every BCIR codec admits any value of an extensible
    constraint (the relay posture), so the TMSAO codecs admit integers V1.1.1 has no spelling for
    -- protocolVersion 300 in
    `SecuredMessagePer`, a negative itsAid anywhere -- and `from_tmsao` mapped them to
    transcription values no V1.1.1 encoder writes. Both directions now hold V1.1.1's range
    (the binary rules' bounds), so the mapping is a bijection between V1.1.1's values and
    their images. Both halves: every envelope still maps there and back, at the edges of
    each range too."""
    value = its.envelope("p1cert")
    image = its.to_tmsao(value)
    assert its.from_tmsao(image) == value
    # The edges of V1.1.1's ranges map; one past them is refused, in both directions.
    edges = {"protocolVersion": (0, 255), "securityProfile": (0, 255)}
    for field, (low, high) in edges.items():
        for good in (low, high):
            assert its.from_tmsao(its.to_tmsao(dict(value, **{field: good}))) == dict(
                value, **{field: good}
            )
        for bad in (low - 1, high + 1):
            assert "not a V1.1.1 value" in _refused(its.from_tmsao, dict(image, **{field: bad}))
            assert "not a V1.1.1 value" in _refused(its.to_tmsao, dict(value, **{field: bad}))
    # The shaped header fields of SecuredMessagePer: the message type and the generation time.
    headers = image["headerFields"]
    for field, bad in (("messageType", 65536), ("generationTime", 2**64), ("messageType", -1)):
        assert field in headers, field
        wide = dict(image, headerFields=dict(headers, **{field: bad}))
        assert "not a V1.1.1 value" in _refused(its.from_tmsao, wide), field
    # IntX, shaped in every envelope: the certificate's ITS-AIDs past 56 bits or below zero.
    cert = image["headerFields"]["signerInfo"][1]
    attributes = cert["subjectAttributes"]
    name = next(n for n in attributes if n in ("itsAidList", "itsAidSspList"))
    for bad in (-1, 2**56):
        items = attributes[name]
        changed = (
            [bad, *items[1:]] if name == "itsAidList" else [dict(items[0], itsAid=bad), *items[1:]]
        )
        cert2 = dict(cert, subjectAttributes=dict(attributes, **{name: changed}))
        signer = ("certificate", cert2)
        wide = dict(image, headerFields=dict(headers, signerInfo=signer))
        assert "IntX" in _refused(its.from_tmsao, wide), bad


def test_a_list_the_prose_forbids_is_refused_not_reordered():
    """Reordering would map two binary encodings to one TMSAO value, and dropping a duplicate
    would lose content: both are refusals."""
    value = its.envelope("p2")
    swapped = dict(value, headerFields=list(reversed(value["headerFields"])))
    assert "canonical order" in _refused(its.to_tmsao, swapped)
    doubled = dict(value, headerFields=value["headerFields"] + value["headerFields"][-1:])
    _refused(its.to_tmsao, doubled)


# --- the accounting ---------------------------------------------------------------------------


def test_the_bit_partition_is_exact_for_every_codec_and_profile():
    for codec in (
        "binary-v111",
        "transcription-coer",
        "transcription-uper",
        "tmsao-coer",
        "tmsao-uper",
        "tmsao-per-uper",
    ):
        for profile in its.PROFILES:
            parts = st.framing(codec, profile)
            assert set(parts) == set(st.CATEGORIES)
            assert sum(parts.values()) == 8 * st.sizes()[codec][profile], (codec, profile)


def test_binary_spends_its_extra_octets_on_framing_not_content():
    """On the CAM envelope V1.1.1 frames with 88 bits -- eight one-octet type codes and three
    one-octet lengths -- around 680 bits of fields. The TMSAO schema under UPER frames the SAME
    680 bits with 32 (a presence bitmap with its extension bits, choice indices, one length);
    its PER instance then carries the shaped version, profile, time and message type in 29
    fewer content bits, at the price of four more extension bits."""
    binary = st.framing("binary-v111", "p1")
    assert binary == {"content": 680, "selector": 64, "length": 24, "presence": 0, "padding": 0}
    uper = st.framing("tmsao-uper", "p1")
    assert uper == {"content": 680, "selector": 12, "length": 8, "presence": 12, "padding": 0}
    shaped = st.framing("tmsao-per-uper", "p1")
    assert shaped == {"content": 651, "selector": 12, "length": 8, "presence": 16, "padding": 1}
    assert [st.content_floor(p) for p in its.PROFILES] == [85, 196, 206, 204]


# --- the binary rules' decoder is total ---------------------------------------------------------


def test_a_vector_is_read_against_its_own_end():
    """A vector's elements are read against the vector's end, threaded through the decoder
    rather than cut from the data for each element (which copied the whole prefix every time).
    The trailer vector's length shortened by one leaves its signature an octet short INSIDE
    the vector, and that is refused -- not completed from the octet just after the vector."""
    raw = its.encode_binary(its.envelope("p1"))
    at = next(
        i for i in range(len(raw) - 1, -1, -1) if raw[i] < 0x80 and i + 1 + raw[i] == len(raw)
    )
    short = raw[:at] + bytes([raw[at] - 1]) + raw[at + 1 :]
    assert "truncated" in _refused(its.decode_binary, short)


def test_a_vector_of_empty_elements_is_refused_rather_than_read_forever():
    """An element that takes no octets leaves the vector's length bounding nothing: the decoder
    read the same empty element without end. V1.1.1 has no such element; the decoder is total
    all the same."""
    from bcir.frontends.asn1 import compile_module

    source = "T DEFINITIONS AUTOMATIC TAGS ::= BEGIN\nL ::= SEQUENCE OF Z\nZ ::= SEQUENCE {}\nEND"
    kind = compile_module(source, "t.asn1").module.types["L"]
    assert "took no octets" in _refused(its._decode, kind, bytes([2, 0, 0]), 0)


def test_the_binary_rules_hold_a_size_constraint():
    """A value outside a SIZE is not a value of the type, whichever rule carries it: the binary
    rules refuse it on the way out and on the way in, as the generated byte-rule codec and its
    COER and UPER siblings do (`test_asn1_cgen`)."""
    from bcir.frontends.asn1 import compile_module

    source = (
        "T DEFINITIONS AUTOMATIC TAGS ::= BEGIN\n"
        "S ::= SEQUENCE { a OCTET STRING (SIZE(1..32)), l SEQUENCE (SIZE(1..2)) OF INTEGER (0..255) }\n"
        "END"
    )
    kind = compile_module(source, "t.asn1").module.types["S"]

    def encode(value):
        out = bytearray()
        its._encode(kind, value, out)
        return bytes(out)

    good = {"a": b"abc", "l": [1]}
    assert its._decode(kind, encode(good), 0) == (good, len(encode(good)))
    for bad in ({"a": b""}, {"a": b"x" * 40}, {"l": []}, {"l": [1, 2, 3]}):
        assert "SIZE" in _refused(encode, dict(good, **bad)), bad
    for octets in (
        bytes([40]) + b"x" * 40 + bytes([1, 1]),
        bytes([3]) + b"abc" + bytes([3, 1, 2, 3]),
    ):
        assert "SIZE" in _refused(its._decode, kind, octets, 0), octets.hex()
