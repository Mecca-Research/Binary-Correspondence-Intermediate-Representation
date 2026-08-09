"""Dual-rail parity for the plan-driven X.691 PER decoder.

`runtime/c/bcir_per_plan.c` decodes a SEQUENCE against a field table rather than discovering
it from the octets, because X.691 6.2 says a PER encoding carries no identifier and 7.2 says
the schema-free walk does not exist. This file encodes a record with the PYTHON rail and
decodes it with the C twin, comparing every field.

WHY THIS FILE EXISTS, rather than the shell gate alone. `tools/c/check_runtime.sh`'s `#per`
gate compared the twin's output at -O0 against its output at -O3. That catches a
miscompilation and nothing else: both columns come from the SAME decoder, so a decoder that
was wrong about a rule agreed with itself perfectly and the gate passed. Two real defects
lived behind exactly that shape --

  * an octet string that did not begin on an octet boundary was reported at `here / 8`, which
    silently rounds the position DOWN to the octet containing it. In UNALIGNED PER that is the
    ordinary case (15's BOOLEAN is one bit, 11.5.6's constrained integer is as many bits as its
    range needs), so a caller slicing on `offset` got neighbouring bytes rather than the string
    and nothing said so;
  * a string of two octets or fewer sitting mid-octet in ALIGNED PER was refused as MALFORMED,
    though 16.6 places exactly that string in a bit-field with no alignment.

So the campaign here compares DECODED VALUES against what the encoder was given, and locates
each string by the reported bit offset and compares its BYTES. A decoder that loses the
sub-octet position cannot pass by reporting a plausible byte index.

Skips cleanly when no C compiler is visible, exactly as the other C-twin tests do.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from bcir.asn1.constraints import Size, ValueRange
from bcir.asn1.per import PerVariant, encode_per
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.tags import Universal

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = ["bcir_per.c", "bcir_per_plan.c", "test_per_plan.c"]

# bcir_per_kind / bcir_per_bounds, as bcir_per_plan.h numbers them.
_INTEGER, _BOOLEAN, _NULL, _FIXED_OCTETS, _VAR_OCTETS = 0, 1, 2, 3, 4
_UNCONSTRAINED, _SEMI, _CONSTRAINED = 0, 1, 2


def _build(tmp: str) -> str | None:
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        return None
    out = os.path.join(tmp, "test_per_plan")
    proc = None
    for std in ("c23", "c2x", "c11"):
        proc = subprocess.run(
            [cc, f"-std={std}", "-O1", "-Wall", "-Wextra", "-Werror", "-I", _C,
             *[os.path.join(_C, name) for name in _SOURCES], "-o", out],
            capture_output=True, text=True)
        if proc.returncode == 0:
            return out
    raise AssertionError(
        f"the plan-driven PER twin must build warning-clean:\n{proc.stderr[:3000]}")


def _run(binary: str, lines: list[str]) -> list[str]:
    proc = subprocess.run([binary], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"driver exited {proc.returncode}: {proc.stderr[:2000]}"
    return proc.stdout.strip().splitlines()


def _field(kind: int, *, bounds: int = 0, lb: int = 0, ub: int = 0,
           fixed: int = 0, optional: int = 0) -> str:
    return f"{kind}:{bounds}:{lb}:{ub}:{fixed}:{optional}"


def _bits(raw: bytes, bit_offset: int, octets: int) -> bytes:
    """The `octets` octets that begin at `bit_offset`, reassembled across octet boundaries.

    This is the shift a caller must perform for a string PER did not align, and doing it here
    is the point of the test: it proves the reported position actually locates the string.
    """
    out = bytearray()
    for i in range(octets):
        byte = 0
        for b in range(8):
            pos = bit_offset + i * 8 + b
            byte = (byte << 1) | ((raw[pos // 8] >> (7 - (pos % 8))) & 1)
        out.append(byte)
    return bytes(out)


def _parse(line: str) -> tuple[int, list[tuple[int, int, int, int, int]]]:
    """`OK <endbit> <n> present:integer:offset:length:bitoffset ...` -> (endbit, values)."""
    parts = line.split()
    assert parts[0] == "OK", f"the twin refused a conforming encoding: {line}"
    endbit, count = int(parts[1]), int(parts[2])
    values = [tuple(int(x) for x in tok.split(":")) for tok in parts[3:]]
    assert len(values) == count
    return endbit, values


# --- the campaign -------------------------------------------------------------------------

_BYTE = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
_WORD = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 65535))
_NIBBLE = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 9))
_FREE = Primitive(Universal.INTEGER, "INTEGER")
_FLAG = Primitive(Universal.BOOLEAN, "BOOLEAN")
_VOID = Primitive(Universal.NULL, "NULL")


def _octets(size: int) -> Primitive:
    return Primitive(Universal.OCTET_STRING, "OCTET STRING",
                     constraint=Size(ValueRange(size, size)))


def _cases() -> list[tuple[str, str, dict, list[str], list[tuple[str, object]]]]:
    """(label, plan, value, member order, [(name, expected)]) for each record shape.

    Every shape deliberately puts a string somewhere a preceding field has left the cursor
    mid-octet, because that is the position both defects were about.
    """
    cases = []

    # A one-bit BOOLEAN then a 3-octet string: bit 1 in UNALIGNED, and 16.6 does not align a
    # 3-octet string... but 16.6's <=2 rule does not apply either, so ALIGNED pads to 8.
    cases.append((
        "flag+fixed3",
        Sequence((Component("flag", _FLAG), Component("s", _octets(3))), name="A"),
        [_field(_BOOLEAN), _field(_FIXED_OCTETS, fixed=3)],
        {"flag": True, "s": b"abc"}))

    # 16.6's own case: two octets or fewer are placed with NO alignment in EITHER variant, so
    # this string starts at bit 1 in both. The twin used to refuse it outright when aligned.
    cases.append((
        "flag+fixed2 (16.6)",
        Sequence((Component("flag", _FLAG), Component("s", _octets(2))), name="B"),
        [_field(_BOOLEAN), _field(_FIXED_OCTETS, fixed=2)],
        {"flag": False, "s": b"hi"}))
    cases.append((
        "flag+fixed1 (16.6)",
        Sequence((Component("flag", _FLAG), Component("s", _octets(1))), name="C"),
        [_field(_BOOLEAN), _field(_FIXED_OCTETS, fixed=1)],
        {"flag": True, "s": b"!"}))

    # A constrained integer whose range is not a whole number of octets leaves UNALIGNED PER
    # at bit 4, so the string after it straddles three octets rather than two.
    cases.append((
        "nibble+fixed3",
        Sequence((Component("n", _NIBBLE), Component("s", _octets(3))), name="D"),
        [_field(_INTEGER, bounds=_CONSTRAINED, lb=0, ub=9), _field(_FIXED_OCTETS, fixed=3)],
        {"n": 7, "s": b"xyz"}))

    # The original three-field record, which is what the shell gate drives.
    cases.append((
        "byte+flag+fixed3",
        Sequence((Component("id", _BYTE), Component("flag", _FLAG),
                  Component("s", _octets(3))), name="E"),
        [_field(_INTEGER, bounds=_CONSTRAINED, lb=0, ub=255), _field(_BOOLEAN),
         _field(_FIXED_OCTETS, fixed=3)],
        {"id": 42, "flag": True, "s": b"abc"}))

    # NULL contributes no bits at all (clause 19), so it must not move the cursor.
    cases.append((
        "flag+null+fixed2",
        Sequence((Component("flag", _FLAG), Component("v", _VOID),
                  Component("s", _octets(2))), name="F"),
        [_field(_BOOLEAN), _field(_NULL), _field(_FIXED_OCTETS, fixed=2)],
        {"flag": True, "v": None, "s": b"ok"}))

    # 18.2's preamble in both of its states, with the string after it either way.
    for present in (True, False):
        value = {"flag": True, "s": b"pq"}
        if present:
            value["w"] = 4097
        cases.append((
            f"optional-word({'present' if present else 'absent'})+flag+fixed2",
            Sequence((Component("w", _WORD, optional=True), Component("flag", _FLAG),
                      Component("s", _octets(2))), name="G"),
            [_field(_INTEGER, bounds=_CONSTRAINED, lb=0, ub=65535, optional=1),
             _field(_BOOLEAN), _field(_FIXED_OCTETS, fixed=2)],
            value))

    # 13.2.4's unconstrained integer, which is length-determined and octet-aligned in ALIGNED.
    cases.append((
        "flag+unconstrained+fixed2",
        Sequence((Component("flag", _FLAG), Component("i", _FREE),
                  Component("s", _octets(2))), name="H"),
        [_field(_BOOLEAN), _field(_INTEGER, bounds=_UNCONSTRAINED),
         _field(_FIXED_OCTETS, fixed=2)],
        {"flag": False, "i": -300, "s": b"zz"}))

    return cases


def test_plan_decoder_agrees_with_the_python_encoder() -> None:
    """Every field the C twin reports equals what the Python rail encoded.

    Strings are compared by their BYTES, read from the position the twin reported -- which is
    what makes a lost sub-octet offset a failure here rather than a plausible number.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return                                   # no C compiler on this host

        cases = _cases()
        # Both variants in one test rather than two parametrized ones: run_all.py calls each
        # test function with no arguments, so the campaign carries its own axis.
        for variant, aligned in ((PerVariant.UNALIGNED, 0), (PerVariant.ALIGNED, 1)):
            lines, encoded = [], []
            for _label, schema, plan, value in cases:
                raw = encode_per(schema, value, variant=variant)
                lines.append(f"sequence {raw.hex()} {aligned} 0 {','.join(plan)}")
                encoded.append(raw)

            out = _run(binary, lines)
            assert len(out) == len(cases), f"{len(cases)} cases in, {len(out)} answers out"

            for (label, schema, _plan, value), raw, line in zip(cases, encoded, out):
                where = f"{label} {variant.value}"
                _endbit, values = _parse(line)
                assert len(values) == len(schema.components), where
                for component, got in zip(schema.components, values):
                    present, integer, offset, length, bit_offset = got
                    expected = value.get(component.name, None)
                    if component.name not in value:
                        assert present == 0, f"{where}: {component.name} absent but reported present"
                        continue
                    assert present == 1, f"{where}: {component.name} present but reported absent"
                    if isinstance(expected, bool):
                        assert integer == int(expected), f"{where}: {component.name}"
                    elif isinstance(expected, int):
                        assert integer == expected, f"{where}: {component.name}"
                    elif isinstance(expected, bytes):
                        assert length == len(expected), f"{where}: {component.name} length"
                        # The bit offset must LOCATE the string, whatever the octet index says.
                        assert _bits(raw, bit_offset, length) == expected, (
                            f"{where}: {component.name} does not live at bit {bit_offset}")
                        # And the octet index must be honest: present only when it is exact.
                        if bit_offset % 8 == 0:
                            assert offset == bit_offset // 8, f"{where}: {component.name} offset"
                            assert raw[offset:offset + length] == expected, where
                        else:
                            assert offset == -1, (
                                f"{where}: {component.name} starts at bit {bit_offset}, which is "
                                f"no octet slice, yet an octet index {offset} was reported")


def test_a_string_that_starts_mid_octet_reports_no_octet_index() -> None:
    """The regression that the -O0/-O3 comparison could not see.

    Rounding a mid-octet position down to the octet containing it yields an index that looks
    entirely usable and is wrong by `bit_offset % 8` bits. So the contract is that no octet
    index is offered at all in that case, and this pins it directly rather than through a
    record shape that might one day be padded into alignment.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return                                   # no C compiler on this host

        schema = Sequence((Component("flag", _FLAG), Component("s", _octets(3))), name="R")
        plan = f"{_field(_BOOLEAN)},{_field(_FIXED_OCTETS, fixed=3)}"
        raw = encode_per(schema, {"flag": True, "s": b"abc"}, variant=PerVariant.UNALIGNED)

        line = _run(binary, [f"sequence {raw.hex()} 0 0 {plan}"])[0]
        _endbit, values = _parse(line)
        _present, _integer, offset, length, bit_offset = values[1]

        assert bit_offset == 1, "a BOOLEAN is one bit, so the string begins at bit 1"
        assert offset == -1, "a mid-octet string has no octet slice to name"
        assert length == 3
        assert _bits(raw, bit_offset, length) == b"abc"
        # The exact misreading the old code produced: byte 0 for a string that starts at bit 1.
        assert raw[0:3] != b"abc", "this record must not be accidentally octet-aligned"


def test_the_length_determinant_switches_form_at_64k() -> None:
    """X.691 11.9.3.3's constrained length form applies only when `ub` is BELOW 64K.

    At or above it, 11.9.1 says the upper bound stops being usable as a constraint and the
    unconstrained determinant of 11.9.3.6 applies instead. `bcir_per_length` takes its
    `has_ub` flag at face value -- per.py applies the same test at ITS call sites -- so the
    caller owns the rule, and the plan decoder did not apply it. Every OCTET STRING whose SIZE
    upper bound reached 64K was therefore refused, on input its own encoder had produced.

    65535 and 65536 are the two sides of the boundary and 70000 is past it; all three must
    decode, and the bytes must be the ones that went in.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return                                   # no C compiler on this host

        payload = b"abcd"
        bounds = (255, 65535, 65536, 70000)
        for variant, aligned in ((PerVariant.UNALIGNED, 0), (PerVariant.ALIGNED, 1)):
            lines, raws = [], []
            for ub in bounds:
                node = Primitive(Universal.OCTET_STRING, "OCTET STRING",
                                 constraint=Size(ValueRange(0, ub)))
                schema = Sequence((Component("s", node),), name="R")
                raw = encode_per(schema, {"s": payload}, variant=variant)
                lines.append(f"sequence {raw.hex()} {aligned} 0 "
                             f"{_field(_VAR_OCTETS, fixed=ub)}")
                raws.append(raw)

            for ub, raw, line in zip(bounds, raws, _run(binary, lines)):
                where = f"SIZE(0..{ub}) {variant.value}"
                assert line.startswith("OK"), (
                    f"{where}: a conforming encoding was refused ({line})")
                _endbit, values = _parse(line)
                _p, _i, _offset, length, bit_offset = values[0]
                assert length == len(payload), where
                assert _bits(raw, bit_offset, length) == payload, where


def test_sixteen_six_short_strings_decode_in_both_variants() -> None:
    """X.691 16.6: a string of two octets or fewer is placed with NO alignment, in EITHER
    variant. The twin used to refuse exactly that shape in ALIGNED PER as MALFORMED, so a
    conforming encoding produced by BCIR's own encoder could not be read back."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return                                   # no C compiler on this host

        schema = Sequence((Component("flag", _FLAG), Component("s", _octets(2))), name="R")
        plan = f"{_field(_BOOLEAN)},{_field(_FIXED_OCTETS, fixed=2)}"
        for variant, aligned in ((PerVariant.UNALIGNED, 0), (PerVariant.ALIGNED, 1)):
            raw = encode_per(schema, {"flag": True, "s": b"hi"}, variant=variant)
            line = _run(binary, [f"sequence {raw.hex()} {aligned} 0 {plan}"])[0]
            assert line.startswith("OK"), f"{variant.value}: a conforming 16.6 encoding was refused"
            _endbit, values = _parse(line)
            _p, _i, _offset, length, bit_offset = values[1]
            assert _bits(raw, bit_offset, length) == b"hi", variant.value
