"""Dual-rail parity for the X.696 OER decoder.

`runtime/c/bcir_oer.c` is the C twin of the decoding half of `bcir/asn1/oer.py`. OER earns a
real native decoder rather than a scanner because it is the encoding rule in the suite with
the best decode cost — everything octet-aligned, most fields fixed-width words a target can
load directly — which is why the build-out roadmap pairs it with the driver-side fast path.

**It is schema-directed, and that is a law rather than a design choice.** X.696 §6.2:
*"without knowledge of the type of the value encoded, it is not possible to determine the
structure of the encoding"*. There are no tags on the wire outside a CHOICE (§8.7.1) and no
lengths except where a clause asks for one, so there is no schema-free structural pass over
OER — the same law X.691 §7.2 states for PER.

That matters beyond this file: OER's absence from the native cost table was labelled *"no C
decoder exists yet"*, as though it were an ordinary gap. It is the same law. What this
decoder changes is not that a schema-free walk became possible — it did not — but that a
schema-**directed** decode can now be timed against another schema-directed decode, which
is like work compared with like.

The campaign is built from the Python encoder's own output, so the C rail is read against
octets the repository actually produces rather than against hand-written fixtures.

Skips cleanly when no C compiler is visible, exactly as the other C-twin tests do.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile

from bcir.asn1.constraints import Size, ValueRange
from bcir.asn1.oer import OerRules, decode_length, encode_length, encode_oer
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.tags import Asn1Error, Universal

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = ["bcir_oer.c", "test_oer.c"]
_SEED = 20260728

#: `bcir_oer_status`, mirrored so a failure names the status rather than a number.
_STATUS = {0: "OK", 1: "TRUNCATED", 2: "MALFORMED", 3: "RANGE", 4: "OVERFLOW",
           5: "INVALID"}
#: `bcir_oer_kind`.
_INTEGER, _BOOLEAN, _NULL, _FIXED_OCTETS, _VAR_OCTETS = 0, 1, 2, 3, 4


def _build(tmp: str) -> str | None:
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        return None
    out = os.path.join(tmp, "test_oer")
    proc = None
    for std in ("c23", "c2x", "c11"):
        proc = subprocess.run(
            [cc, f"-std={std}", "-O1", "-Wall", "-Wextra", "-Werror", "-I", _C,
             *[os.path.join(_C, name) for name in _SOURCES], "-o", out],
            capture_output=True, text=True)
        if proc.returncode == 0:
            return out
    raise AssertionError(f"the OER twin must build warning-clean:\n{proc.stderr[:3000]}")


def _run(binary: str, lines: list[str]) -> list[str]:
    proc = subprocess.run([binary], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"driver exited {proc.returncode}: {proc.stderr[:2000]}"
    return proc.stdout.splitlines()


def _hex(data: bytes) -> str:
    return data.hex() if data else "-"


# --- §8.6 the length determinant ------------------------------------------------------------


def test_the_length_determinant_agrees_on_every_value_and_form():
    """Short form, long form, and the boundary at 128 where the form changes."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        values = [0, 1, 126, 127, 128, 255, 256, 65535, 65536, 1 << 24, (1 << 32) - 1]
        octets = [encode_length(v) for v in values]
        answers = _run(binary, [f"length {_hex(o)} 0" for o in octets])
        for value, raw, got in zip(values, octets, answers):
            assert got.startswith("OK "), f"{value}: {got}"
            read, end, canonical = got.split()[1:4]
            assert int(read) == value, f"{value}: C read {read}"
            assert int(end) == len(raw), f"{value}: C ended at {end}, encoding is {len(raw)}"
            assert canonical == "1", f"{value}: the canonical encoder's output read as basic"
            # And the Python decoder agrees about both, from the same octets.
            assert decode_length(raw, 0) == (value, len(raw))


def test_a_redundant_leading_zero_is_accepted_and_reported_as_non_canonical():
    """§3.7.12's NOTE against §31.2 — "BASIC in, CANONICAL out", made observable.

    A decoder that silently normalized would let a peer choose the digest by choosing a
    spelling. The C rail accepts the BASIC form and *says* it was not canonical, so a
    caller digesting the input can refuse on its own terms without a second parser.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        # 0x82 0x00 0x80 is 128 written in two octets; 0x81 0x80 is the canonical form.
        basic, canonical = bytes([0x82, 0x00, 0x80]), encode_length(128)
        got_basic, got_canonical = _run(binary, [f"length {_hex(basic)} 0",
                                                 f"length {_hex(canonical)} 0"])
        assert got_basic.split()[1] == "128" and got_basic.split()[3] == "0"
        assert got_canonical.split()[1] == "128" and got_canonical.split()[3] == "1"
        # Python accepts the BASIC spelling too, and reads the same value.
        assert decode_length(basic, 0)[0] == 128


def test_the_long_form_with_a_zero_count_is_malformed_not_truncated():
    """§8.6.5: the octets are all present and they encode nothing.

    Distinguished from a truncation on purpose — "send me more bytes" and "these bytes are
    wrong" call for different responses from a peer.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        got, truncated = _run(binary, ["length 80 0", "length 82ff 0"])
        assert _STATUS[int(got.split()[1])] == "MALFORMED", got
        assert _STATUS[int(truncated.split()[1])] == "TRUNCATED", truncated
        try:
            decode_length(bytes([0x80]), 0)
        except Asn1Error as error:
            assert "8.6.5" in str(error)
        else:
            raise AssertionError("the Python rail accepted a zero-count long form")


# --- §10.3 / §10.4 integers ------------------------------------------------------------------


def _int_type(low, high):
    return Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(low, high))


def test_every_fixed_width_integer_form_round_trips_against_the_python_encoder():
    """§10.3/§10.4's word widths, with the sign chosen by clause 10.2's split.

    The split turns on whether a lower bound EXISTS and is non-negative — not on whether
    the bounds happen to be small — so an 8-bit-looking type with no lower bound is signed.
    """
    cases = [
        (_int_type(0, 255), 1, 0, [0, 1, 200, 255]),
        (_int_type(0, 65535), 2, 0, [0, 4096, 65535]),
        (_int_type(0, 2 ** 32 - 1), 4, 0, [0, 1 << 20, 2 ** 32 - 1]),
        (_int_type(-128, 127), 1, 1, [-128, -1, 0, 127]),
        (_int_type(-32768, 32767), 2, 1, [-32768, -1, 32767]),
        (_int_type(-(2 ** 31), 2 ** 31 - 1), 4, 1, [-(2 ** 31), -1, 2 ** 31 - 1]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        lines, expected = [], []
        for kind, width, signed, values in cases:
            for value in values:
                octets = encode_oer(kind, value, rules=OerRules.CANONICAL)
                assert len(octets) == width, (kind, value, octets)
                lines.append(f"integer {_hex(octets)} 0 {width} {signed}")
                expected.append((value, width))
        for (value, width), got in zip(expected, _run(binary, lines)):
            assert got.startswith("OK "), f"{value}: {got}"
            read, end = got.split()[1:3]
            assert int(read) == value, f"expected {value}, C read {read}"
            assert int(end) == width


def test_the_length_prefixed_integer_form_agrees():
    """§10.3 e) / §10.4 e): a determinant then a variable-size number.

    This is the form an unconstrained INTEGER takes — four times the size of the same
    abstract value in a bounded type, which is the constraint story `oer.py` documents.
    """
    kind = Primitive(Universal.INTEGER, "INTEGER")
    values = [0, 1, -1, 127, -128, 128, -129, 32767, -32768, 2 ** 40, -(2 ** 40)]
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        octets = [encode_oer(kind, v, rules=OerRules.CANONICAL) for v in values]
        answers = _run(binary, [f"integer {_hex(o)} 0 0 1" for o in octets])
        for value, raw, got in zip(values, octets, answers):
            assert got.startswith("OK "), f"{value}: {got}"
            read, end = got.split()[1:3]
            assert int(read) == value, f"expected {value}, C read {read}"
            assert int(end) == len(raw)


def test_an_unsigned_64_bit_value_above_int64_max_is_refused_not_wrapped():
    """Returning a negative number for a positive value is a different value, not a lossy one.

    The alternative — wrapping — would hand the caller a plausible answer that is wrong,
    which is the failure mode this repository refuses everywhere else too.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        big = (b"\xff" * 8).hex()
        got = _run(binary, [f"integer {big} 0 8 0"])[0]
        assert _STATUS[int(got.split()[1])] == "RANGE", got


# --- §16.2 the SEQUENCE preamble --------------------------------------------------------------


def test_the_preamble_bit_order_matches_the_python_encoder():
    """Most significant bit first, one bit per OPTIONAL root component.

    Bit order is the kind of thing two implementations agree on by luck until a type has
    more than one optional component, so this walks every subset of three.
    """
    inner = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
    kind = Sequence((
        Component("a", inner, optional=True),
        Component("b", inner, optional=True),
        Component("c", inner, optional=True),
    ), name="ThreeOptional")
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        lines, expected = [], []
        for mask in range(8):
            value = {}
            if mask & 1:
                value["a"] = 1
            if mask & 2:
                value["b"] = 2
            if mask & 4:
                value["c"] = 3
            octets = encode_oer(kind, value, rules=OerRules.CANONICAL)
            lines.append(f"preamble {_hex(octets)} 0 3")
            expected.append((mask, value))
        for (mask, _value), got in zip(expected, _run(binary, lines)):
            assert got.startswith("OK "), f"{mask}: {got}"
            present, end, canonical = got.split()[1:4]
            assert int(present) == mask, f"mask {mask}: C read {present}"
            assert int(end) == 1 and canonical == "1"


def test_a_type_with_no_optional_component_has_no_preamble():
    """§16.2: no OPTIONAL/DEFAULT root means no preamble octet at all.

    A decoder that always read one would be off by a byte on the most common shape there
    is, and would still decode the FIRST field of many types plausibly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        got = _run(binary, ["preamble 41 0 0"])[0]
        assert got.split()[1:3] == ["0", "0"], got


def test_non_zero_trailing_padding_is_accepted_and_reported_as_non_canonical():
    """§16.2.2 requires zero padding under CANONICAL-OER; BASIC admits the rest."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        clean, padded = _run(binary, ["preamble 80 0 1", "preamble 81 0 1"])
        assert clean.split()[1] == "1" and clean.split()[3] == "1"
        assert padded.split()[1] == "1" and padded.split()[3] == "0", padded


# --- the plan-driven decode -----------------------------------------------------------------------


def _plan(*fields) -> str:
    return ",".join(f"{k}:{w}:{s}:{o}:{f}" for k, w, s, o, f in fields)


def test_a_record_decodes_field_for_field_against_the_python_encoder():
    """The whole point: one schema, the Python encoder's octets, the C decoder's values."""
    byte = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
    word = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(-32768, 32767))
    text = Primitive(Universal.UTF8_STRING, "UTF8String")
    kind = Sequence((
        Component("id", byte),
        Component("delta", word),
        Component("label", text),
        Component("note", text, optional=True),
    ), name="Record")
    plan = _plan((_INTEGER, 1, 0, 0, 0), (_INTEGER, 2, 1, 0, 0),
                 (_VAR_OCTETS, 0, 0, 0, 0), (_VAR_OCTETS, 0, 0, 1, 0))
    values = [
        {"id": 7, "delta": -3, "label": "abc", "note": "n"},
        {"id": 0, "delta": 0, "label": "", "note": ""},
        {"id": 255, "delta": 32767, "label": "x" * 200},
        {"id": 1, "delta": -32768, "label": "unicode-é"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        octets = [encode_oer(kind, v, rules=OerRules.CANONICAL) for v in values]
        answers = _run(binary, [f"sequence {_hex(o)} {plan}" for o in octets])
        for value, raw, got in zip(values, octets, answers):
            assert got.startswith("OK "), f"{value}: {got}"
            parts = got.split()
            assert int(parts[1]) == len(raw), f"{value}: C ended at {parts[1]} of {len(raw)}"
            assert parts[2] == "1", f"{value}: canonical octets read as basic"
            assert parts[3] == f"i{value['id']}"
            assert parts[4] == f"i{value['delta']}"
            assert parts[5] == "s" + (value["label"].encode().hex() or "-")
            if "note" in value:
                assert parts[6] == "s" + (value["note"].encode().hex() or "-")
            else:
                assert parts[6] == "-", f"{value}: an absent component was read as present"


def test_an_absent_optional_component_consumes_no_octets():
    """§16.2. Absence is read from the preamble, never discovered from the contents.

    A decoder that advanced the cursor for an absent component would mis-frame everything
    after it — and would usually still return a value for the next field.
    """
    byte = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
    kind = Sequence((
        Component("maybe", byte, optional=True),
        Component("always", byte),
    ), name="Gapped")
    plan = _plan((_INTEGER, 1, 0, 1, 0), (_INTEGER, 1, 0, 0, 0))
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        with_it = encode_oer(kind, {"maybe": 9, "always": 4}, rules=OerRules.CANONICAL)
        without = encode_oer(kind, {"always": 4}, rules=OerRules.CANONICAL)
        assert len(without) == len(with_it) - 1, "the encoder did not omit the component"
        a, b = _run(binary, [f"sequence {_hex(with_it)} {plan}",
                             f"sequence {_hex(without)} {plan}"])
        assert a.split()[3:5] == ["i9", "i4"], a
        assert b.split()[3:5] == ["-", "i4"], b


def test_a_fixed_size_string_carries_no_length_determinant():
    """§14.1 — the case that makes OER cheap, and the one a decoder gets wrong by adding
    a determinant that is not there."""
    fixed = Primitive(Universal.OCTET_STRING, "OCTET STRING", constraint=Size(ValueRange(4, 4)))
    kind = Sequence((Component("mac", fixed),), name="Fixed")
    plan = _plan((_FIXED_OCTETS, 0, 0, 0, 4))
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        octets = encode_oer(kind, {"mac": b"\xde\xad\xbe\xef"}, rules=OerRules.CANONICAL)
        assert len(octets) == 4, f"a SIZE(4,4) OCTET STRING encoded to {len(octets)} octets"
        got = _run(binary, [f"sequence {_hex(octets)} {plan}"])[0]
        assert got.split()[3] == "sdeadbeef", got


def test_a_plan_the_decoder_cannot_execute_is_refused_before_any_octet_is_read():
    """Fail closed, and fail *early*.

    A partial decode that stopped in the middle would force the caller to tell "the input
    was short" from "the plan was wrong" by inspecting how far it got.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        bad_kind = _run(binary, [f"sequence 00 {_plan((9, 0, 0, 0, 0))}"])[0]
        bad_width = _run(binary, [f"sequence 00 {_plan((_INTEGER, 3, 0, 0, 0))}"])[0]
        for got in (bad_kind, bad_width):
            assert _STATUS[int(got.split()[1])] == "INVALID", got
            assert got.split()[2] == "-1", f"a plan fault reported an octet offset: {got}"


def test_truncation_is_diagnosed_at_every_prefix_of_a_real_encoding():
    """Totality over the shapes that actually occur, not only over random bytes.

    Every proper prefix of a valid encoding must be refused rather than decoded into
    whatever the missing octets would have said. The fuzzer covers arbitrary input; this
    covers the input a peer produces when a connection drops.
    """
    byte = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
    text = Primitive(Universal.UTF8_STRING, "UTF8String")
    kind = Sequence((Component("id", byte), Component("label", text),
                     Component("note", text, optional=True)), name="R")
    plan = _plan((_INTEGER, 1, 0, 0, 0), (_VAR_OCTETS, 0, 0, 0, 0),
                 (_VAR_OCTETS, 0, 0, 1, 0))
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        full = encode_oer(kind, {"id": 3, "label": "hello", "note": "x"},
                          rules=OerRules.CANONICAL)
        lines = [f"sequence {_hex(full[:cut])} {plan}" for cut in range(len(full))]
        for cut, got in enumerate(_run(binary, lines)):
            assert got.startswith("ERR "), f"prefix of {cut} octets decoded: {got}"


def test_a_generated_corpus_keeps_the_two_rails_in_step():
    """Random records through the Python encoder and the C decoder.

    Covers the shapes nobody thought of, which for a length-prefixed format is mostly
    about where the short/long form boundary falls inside a larger structure.
    """
    rng = random.Random(_SEED)
    byte = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
    word = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(-32768, 32767))
    text = Primitive(Universal.UTF8_STRING, "UTF8String")
    kind = Sequence((Component("id", byte), Component("delta", word),
                     Component("label", text), Component("note", text, optional=True)),
                    name="Record")
    plan = _plan((_INTEGER, 1, 0, 0, 0), (_INTEGER, 2, 1, 0, 0),
                 (_VAR_OCTETS, 0, 0, 0, 0), (_VAR_OCTETS, 0, 0, 1, 0))
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        values = []
        for _ in range(150):
            value = {"id": rng.randrange(256), "delta": rng.randrange(-32768, 32768),
                     # Lengths straddling 127 are where the determinant changes form.
                     "label": "a" * rng.choice([0, 1, 126, 127, 128, 129, 300])}
            if rng.random() < 0.5:
                value["note"] = "n" * rng.randrange(0, 130)
            values.append(value)
        octets = [encode_oer(kind, v, rules=OerRules.CANONICAL) for v in values]
        answers = _run(binary, [f"sequence {_hex(o)} {plan}" for o in octets])
        for value, raw, got in zip(values, octets, answers):
            assert got.startswith("OK "), f"{value}: {got}"
            parts = got.split()
            assert int(parts[1]) == len(raw), f"{value}: consumed {parts[1]} of {len(raw)}"
            assert parts[3] == f"i{value['id']}" and parts[4] == f"i{value['delta']}"
            assert parts[5] == "s" + (value["label"].encode().hex() or "-")
            assert parts[6] == ("-" if "note" not in value
                                else "s" + (value["note"].encode().hex() or "-"))
