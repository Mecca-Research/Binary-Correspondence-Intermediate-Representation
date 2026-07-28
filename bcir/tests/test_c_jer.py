"""Dual-rail parity for the bounded X.697 JER reader (roadmap phase J3).

`runtime/c/bcir_jer.c` is the C twin of `bcir/asn1/jer_bounded.py`: the half of a JER decode
that runs BEFORE any type is consulted, on octets an attacker chose. These tests build the
driver in `runtime/c/test_jer.c` and push the SAME campaign through both rails.

**What is compared, and why each piece.**

*Refusals are compared by code, offset and required capacity* — not merely by the fact of a
refusal. §4.2 asks for "a stable error code, byte offset, schema path, and required
capacity", and a peer that is told only "no" cannot act. A twin that refused the right
documents for the wrong reason would be conformant and useless.

*Acceptances are compared by the whole event trace.* JSON is text, so a superficial "does it
parse" check passes on almost anything; what matters is the exact tokenization — which
member name, which string contents after escape decoding, which raw number token, in what
order, at which byte offsets. A reader that gets the shape right and a string's contents
wrong hands back a value the sender did not mean.

*Number events carry the RAW token.* Nothing in the C rail parses a double, and the trace
proves it: a freestanding reader that called `strtod` would make "the same document" mean
two different values depending on libm and locale. Python's reference trace uses
`parse_float=str` / `parse_int=str`, which preserves the source token exactly, so the two
rails compare octet for octet with no float in sight.

**Where the two rails deliberately differ**, and it is worth being explicit because the
divergence is the interesting part:

- `jer_bounded.scan` is a *bounding* pass with `json.loads` behind it, so it does not check
  that a comma separates two values. The C rail has nothing behind it, so `bcir_jer_parse`
  is a real parser. Stage-1 parity is against `scan`; stage-3 acceptance is against
  `json.loads`.
- The driver's `parse` op runs **all three stages in §4.2's order** — scan, UTF-8, grammar —
  because that is what `decode_bounded` does and what a real caller must do. Comparing
  `bcir_jer_parse` *alone* against `json.loads` would be comparing unlike things: a raw
  `0x80` inside a string literal is well-formed JSON structure, so the parser copies it
  through untouched by design (answering the encoding question in two places would give one
  fault two different offsets), while `json.loads` refuses the document because decoding
  UTF-8 is part of what it does. Only the composed pipeline is comparable to it.
- `bcir_jer_parse` reports `TRAILING_INPUT` where the Python rail reports `SCHEMA`, because
  on that rail the trailing octets are `json.loads`'s complaint rather than the bounding
  pass's. Compared as accept/reject, not by code.
- `BCIR_JER_OVERFLOW` and `BCIR_JER_SINK_REFUSED` have no Python counterpart at all: the
  Python rail allocates and has no sink. They are tested on the C rail alone.

The totality property over arbitrary bytes is the fuzzer's job (`runtime/c/fuzz_jer.c`),
not this file's.

Skips cleanly when no C compiler is visible, exactly as the other C-twin tests do.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import struct
import subprocess
import tempfile
import zlib

from bcir.asn1.jer_bounded import (
    FRAME_HEADER_SIZE, STRICT_LIMITS, JerBoundedError, JerErrorCode, JerLimits, frame, scan,
)

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = ["bcir_jer.c", "test_jer.c", "bcir_runtime.c"]
_SEED = 20260727

#: `bcir_jer_status` -> the `JerErrorCode` it mirrors. The three C-only statuses are absent
#: on purpose: OVERFLOW and SINK_REFUSED have no Python analogue (that rail allocates and
#: has no sink), and INVALID is a NULL-pointer guard a Python signature cannot express.
_STATUS = {
    1: JerErrorCode.INPUT_TOO_LARGE,
    2: JerErrorCode.DEPTH_EXCEEDED,
    3: JerErrorCode.NODES_EXCEEDED,
    4: JerErrorCode.MEMBERS_EXCEEDED,
    5: JerErrorCode.ELEMENTS_EXCEEDED,
    6: JerErrorCode.STRING_TOO_LONG,
    7: JerErrorCode.NUMBER_TOO_LONG,
    8: JerErrorCode.DIGITS_EXCEEDED,
    9: JerErrorCode.EXPONENT_EXCEEDED,
    10: JerErrorCode.WORK_EXCEEDED,
    11: JerErrorCode.MALFORMED,
    12: JerErrorCode.NOT_UTF8,
    13: JerErrorCode.TRAILING_INPUT,
    14: JerErrorCode.FRAME_MALFORMED,
    15: JerErrorCode.FRAME_INTEGRITY,
}
_C_ONLY = {16: "OVERFLOW", 17: "SINK_REFUSED", 18: "INVALID"}


def _build(tmp: str) -> str | None:
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        return None
    out = os.path.join(tmp, "test_jer")
    proc = None
    for std in ("c23", "c2x", "c11"):
        proc = subprocess.run(
            [cc, f"-std={std}", "-O1", "-Wall", "-Wextra", "-Werror", "-I", _C,
             *[os.path.join(_C, name) for name in _SOURCES], "-o", out],
            capture_output=True, text=True)
        if proc.returncode == 0:
            return out
    raise AssertionError(f"the JER twin must build warning-clean:\n{proc.stderr[:3000]}")


def _run(binary: str, lines: list[str]) -> list[str]:
    proc = subprocess.run([binary], input="\n".join(lines) + "\n",
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f"driver exited {proc.returncode}: {proc.stderr[:2000]}"
    return proc.stdout.splitlines()


def _hex(data: bytes) -> str:
    return data.hex() if data else "-"


def _blocks(answers: list[str]) -> list[list[str]]:
    """Split a `parse`/`refuse` run's output into its TRACE ... END blocks."""
    out: list[list[str]] = []
    current: list[str] | None = None
    for line in answers:
        if line == "TRACE":
            current = []
        elif line == "END":
            assert current is not None
            out.append(current)
            current = None
        else:
            assert current is not None, f"a line outside any block: {line!r}"
            current.append(line)
    assert current is None, "an unterminated TRACE block"
    return out


# --- the corpus ----------------------------------------------------------------------------

#: Documents chosen so every branch of both rails is reached: each container shape, each
#: scalar, the trailing-comma and missing-separator forms permissive readers accept, every
#: escape, the surrogate cases, and the truncations one octet short of each.
_DOCS: list[bytes] = [
    b"", b" ", b"\t\n\r ", b"null", b"true", b"false",
    b"0", b"-0", b"1", b"-1", b"10", b"1.5", b"-0.5e+3", b"1E-2", b"1e0",
    b"01", b"-01", b"00", b"1.", b".5", b"-", b"1e", b"1e+", b"1.e3", b"+1",
    b'""', b'"a"', b'"ab\\ncd"', b'"\\"\\\\\\/\\b\\f\\n\\r\\t"', b'"\\u0041"',
    b'"\\u00e9"', b'"\\uD83D\\uDE00"', b'"\\ud800"', b'"\\udc00"', b'"\\ud800\\ud800"',
    b'"\\ud800x"', b'"\\uZZZZ"', b'"\\u041"', b'"\\q"', b'"a', b'"\\', b'"\\u',
    b"[]", b"{}", b"[1]", b"[1,2,3]", b'{"a":1}', b'{"a":1,"b":2}',
    b'{"a":{"b":{"c":[1,[2,[3]]]}}}', b'[[[[[]]]]]',
    b"[1,]", b'{"a":1,}', b"[,]", b"{,}", b"[1 2]", b'{"a" 1}', b'{"a":}', b'{:1}',
    b"[", b"]", b"{", b"}", b"[{", b'{"a"', b'{"a":', b"[1", b"1 2", b"[]]", b"{}}",
    b"nan", b"NaN", b"Infinity", b"-Infinity", b"undefined", b"'a'", b"tru", b"nul",
    b'{"a":1,"a":2}',                              # a duplicate: the schema layer's problem
    b'[null,true,false,0,"",{},[]]',
    "[\"é\", \"中\", \"\U0001f600\"]".encode(),
    b'"\x00"', b'"\x1f"', b'"\x7f"',               # control characters, literal
    b"\xef\xbb\xbf{}",                             # a byte-order mark
    b"\x80", b'"\x80"', b'"\xc3"', b'"\xc0\x80"', b'"\xed\xa0\x80"',
]


class _Object(list):
    """A JSON object, kept distinguishable from an array.

    `object_pairs_hook=list` alone cannot tell `{}` from `[]` — both arrive as an empty
    `list`, and "does it start with a tuple?" is a heuristic that silently fails on exactly
    the empty case. Naming the type removes the guess.
    """


class _Number(str):
    """A number token, kept distinct from a JSON string.

    `parse_float=str` and `parse_int=str` hand back a `str`, which would otherwise be
    indistinguishable from a JSON string — and conflating the two is exactly the confusion
    that made `jer.py`'s `_Raw` accept a number where Table 2 wanted a string. Subclassing
    is safe *here*, unlike there, because this class never leaves the test and the only
    question asked of it is `isinstance`.
    """


def _reference(raw: bytes) -> list[str] | None:
    """Python's event trace for `raw`, in the driver's vocabulary, or None if `json.loads`
    refuses it.

    This is a *serializer over `json`'s own output*, not a second parser — which is what
    keeps it from drifting from the rail it is meant to check. `object_pairs_hook=list`
    preserves member order and duplicates, and `parse_float`/`parse_int` preserve the source
    number token exactly, so no float is constructed on either rail.

    `parse_constant` is made to raise because `json.loads` accepts `NaN`, `Infinity` and
    `-Infinity` by default and ECMA-404 has no such tokens. Leaving the default in would
    have made this reference *more* permissive than the standard and quietly excused a C
    rail that admitted them.

    Byte offsets are the one thing `json` does not report, so they are checked separately by
    `test_the_trace_offsets_point_at_the_construct_they_name` rather than reconstructed here.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        value = json.loads(text, object_pairs_hook=_Object, parse_float=_Number,
                           parse_int=_Number,
                           parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
    except ValueError:
        return None
    out: list[str] = []

    def walk(node) -> None:
        if isinstance(node, _Object):
            out.append("{")
            for name, member in node:
                out.append(f"key {_hex(name.encode())}")
                walk(member)
            out.append("}")
        elif isinstance(node, list):
            out.append("[")
            for element in node:
                walk(element)
            out.append("]")
        elif node is True:
            out.append("true")
        elif node is False:
            out.append("false")
        elif node is None:
            out.append("null")
        elif isinstance(node, _Number):
            out.append(f"num {_hex(str(node).encode())}")
        elif isinstance(node, str):
            try:
                out.append(f"str {_hex(node.encode())}")
            except UnicodeEncodeError:
                # A lone surrogate from a `\uD800` escape. `json` builds it; UTF-8 has no
                # spelling for it; §7.6.2 makes the document UTF-8. The C rail refuses it,
                # and so — since this build — does the Python rail.
                out.append("str <unencodable>")
        else:                                                # pragma: no cover - defensive
            raise AssertionError(f"an unexpected node: {node!r}")

    walk(value)
    return out


def _strip_offsets(block: list[str]) -> list[str]:
    """Drop the byte offsets from a C trace, leaving the vocabulary `_reference` produces."""
    out = []
    for line in block:
        parts = line.split()
        if parts[0] in ("{", "}", "[", "]", "true", "false", "null"):
            out.append(parts[0])
        elif parts[0] in ("key", "str", "num"):
            out.append(f"{parts[0]} {parts[2]}")
        else:
            out.append(line)
    return out


# --- stage 1: the bounding pass ------------------------------------------------------------

def test_the_bounding_pass_diagnoses_identically_on_both_rails():
    """§4.3's limits, compared by code, offset and required capacity.

    This is the one stage where the two rails implement the *same* function, so the
    comparison is exact: every refusal must name the same error code, the same octet, and
    the same "how much would have been enough".
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        cases = [(raw, strict) for raw in _DOCS for strict in (0, 1)]
        answers = _run(binary, [f"scan {strict} {_hex(raw)}" for raw, strict in cases])
        assert len(answers) == len(cases)
        for (raw, strict), got in zip(cases, answers):
            limits = STRICT_LIMITS if strict else JerLimits()
            try:
                nodes = scan(raw, limits)
            except JerBoundedError as error:
                diagnostic = error.diagnostic
                assert got.startswith("ERR "), (
                    f"{raw!r} (strict={strict}): C accepted it ({got}), "
                    f"Python refused it ({diagnostic})")
                code, offset, needed = got.split()[1:4]
                assert _STATUS[int(code)] is diagnostic.code, (
                    f"{raw!r}: C said {_STATUS.get(int(code))}, Python {diagnostic.code}")
                assert int(offset) == diagnostic.offset, (
                    f"{raw!r} ({diagnostic.code.value}): C at octet {offset}, "
                    f"Python at {diagnostic.offset}")
                assert int(needed) == (diagnostic.needed or 0), (
                    f"{raw!r} ({diagnostic.code.value}): C needs {needed}, "
                    f"Python {diagnostic.needed}")
            else:
                assert got == f"OK {nodes}", (
                    f"{raw!r} (strict={strict}): C said {got}, Python counted {nodes} nodes")


def test_every_limit_is_reached_and_named_by_both_rails():
    """One document per §4.3 limit, built to cross exactly that ceiling and no other.

    A limit nobody ever reaches is a limit nobody notices is missing, so each is driven to
    its boundary explicitly rather than hoped for out of the general corpus.
    """
    limits = STRICT_LIMITS
    cases: list[tuple[bytes, JerErrorCode]] = [
        (b"[" * (limits.depth + 1) + b"]" * (limits.depth + 1),
         JerErrorCode.DEPTH_EXCEEDED),
        # Bare separators, not values. `elements` and `nodes` are both 512 under this
        # profile, so 514 scalar elements are 514 nodes and NODES_EXCEEDED fires first —
        # the ceiling under test never gets reached. Commas are counted as elements by the
        # bounding pass and are not nodes, which isolates the one limit this row is for.
        # (The document is not valid JSON; stage 1 is a bounding pass and does not care,
        # and stage 3 is not what this test drives.)
        (b"[" + b"," * (limits.elements + 1) + b"]", JerErrorCode.ELEMENTS_EXCEEDED),
        (b"{" + b"," * (limits.members + 1) + b"}", JerErrorCode.MEMBERS_EXCEEDED),
        (b'"' + b"a" * (limits.string_bytes + 1) + b'"', JerErrorCode.STRING_TOO_LONG),
        (b"1" * (limits.integer_digits + 1), JerErrorCode.DIGITS_EXCEEDED),
        (b"1e" + str(limits.exponent_magnitude + 1).encode(),
         JerErrorCode.EXPONENT_EXCEEDED),
        (b"x" * (limits.input_bytes + 1), JerErrorCode.INPUT_TOO_LARGE),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        answers = _run(binary, [f"scan 1 {_hex(raw)}" for raw, _ in cases])
        for (raw, want), got in zip(cases, answers):
            try:
                scan(raw, limits)
            except JerBoundedError as error:
                assert error.diagnostic.code is want, (
                    f"the Python rail said {error.diagnostic.code} for the {want.value} case")
            else:
                raise AssertionError(f"the Python rail accepted the {want.value} case")
            assert got.startswith("ERR "), f"{want.value}: C accepted it ({got})"
            assert _STATUS[int(got.split()[1])] is want, f"{want.value}: C said {got}"


def test_a_number_token_ceiling_and_a_work_ceiling_are_both_reachable():
    """`number_bytes` and `work` need documents the other limits do not trip first.

    They are separated from the sweep above because reaching them is a construction, not an
    accident: a token long enough to exceed `number_bytes` without exceeding
    `integer_digits` needs its length in the fraction, and exhausting `work` needs many
    small tokens rather than one large one.
    """
    limits = STRICT_LIMITS
    long_fraction = b"0." + b"1" * limits.number_bytes
    much_work = b"[" + b",".join(b"1" for _ in range(400)) + b"]"
    cases = [(long_fraction, JerErrorCode.NUMBER_TOO_LONG),
             (much_work, JerErrorCode.WORK_EXCEEDED)]
    tight = limits.tightened(work=64)
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        answers = _run(binary, [f"scan 1 {_hex(long_fraction)}"])
        try:
            scan(long_fraction, limits)
        except JerBoundedError as error:
            assert error.diagnostic.code is cases[0][1], error.diagnostic
        else:
            raise AssertionError("the Python rail accepted an over-long number token")
        assert _STATUS[int(answers[0].split()[1])] is cases[0][1], answers[0]
        # `work` cannot be driven through the driver's two fixed profiles, so it is checked
        # on the Python rail alone -- the C rail's accounting is compared against it
        # indirectly, by every WORK_EXCEEDED offset in the sweep above.
        try:
            scan(much_work, tight)
        except JerBoundedError as error:
            assert error.diagnostic.code is JerErrorCode.WORK_EXCEEDED, error.diagnostic
        else:
            raise AssertionError("a 64-unit work budget accepted a 400-element array")


def test_a_limit_may_be_tightened_and_never_expanded_on_both_rails():
    """§4.3: limits "are part of the compiled plan and may be tightened by a caller, never
    silently expanded". A struct assignment cannot say that, so both rails check it."""
    fields = ["input_bytes", "depth", "nodes", "members", "elements", "string_bytes",
              "number_bytes", "integer_digits", "exponent_magnitude", "work"]
    base = JerLimits()
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        lines = []
        expected = []
        for field in fields:
            current = getattr(base, field)
            lines.append(f"tighten {field} {current - 1}")
            expected.append(True)
            lines.append(f"tighten {field} {current + 1}")
            expected.append(False)
        answers = _run(binary, lines)
        for field, line, want, got in zip(
                [f for f in fields for _ in (0, 1)], lines, expected, answers):
            assert got.startswith("OK") == want, f"{line}: C said {got}"
            current = getattr(base, field)
            try:
                base.tightened(**{field: current + (-1 if want else 1)})
            except JerBoundedError:
                assert not want, f"the Python rail refused to tighten {field}"
            else:
                assert want, f"the Python rail expanded {field}"


# --- stage 2: the encoding -----------------------------------------------------------------

def test_the_document_utf8_check_names_the_same_octet_on_both_rails():
    """§7.6.2 — and the offset matters as much as the verdict.

    Python's `UnicodeDecodeError.start` is the first octet of the invalid sequence, and the
    C rail reports the same one, so a peer told "not UTF-8 at octet N" gets the same N from
    either rail.
    """
    cases = list(_DOCS) + [
        b"\xc0\x80", b"\xe0\x80\x80", b"\xf0\x80\x80\x80", b"\xed\xa0\x80",
        b"\xf5\x80\x80\x80", b"\x80", b"\xc3", b"\xc3\xa9", b"a\xc3", b"ab\xed\xa0\x80c",
        b"\xf4\x8f\xbf\xbf", b"\xf4\x90\x80\x80",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        answers = _run(binary, [f"utf8doc {_hex(raw)}" for raw in cases])
        for raw, got in zip(cases, answers):
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as error:
                assert got.startswith("ERR "), f"{raw!r}: C accepted it ({got})"
                code, offset = got.split()[1:3]
                assert _STATUS[int(code)] is JerErrorCode.NOT_UTF8, f"{raw!r}: {got}"
                assert int(offset) == error.start, (
                    f"{raw!r}: C at octet {offset}, Python at {error.start}")
            else:
                assert got == "OK", f"{raw!r}: C refused it ({got})"


def test_the_scalar_utf8_decoder_refuses_what_is_not_a_character():
    """Overlong forms, surrogates and anything above U+10FFFF, against Python as oracle.

    Two decoders that disagree about what a byte sequence means is the classic
    validator/consumer split, and it is how an ASCII-looking filter gets bypassed.
    """
    cases = [b"\xc0\x80", b"\xe0\x80\x80", b"\xf0\x80\x80\x80", b"\xed\xa0\x80",
             b"\xf5\x80\x80\x80", b"\x80", b"\xc3", b"\xc3\xa9", b"\xf0\x9f\x98\x80",
             b"\xf4\x8f\xbf\xbf", b"\xf4\x90\x80\x80", b"A", b"\xc1\xbf"]
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        answers = _run(binary, [f"utf8 {_hex(raw)} 0" for raw in cases])
        for raw, got in zip(cases, answers):
            try:
                want = raw.decode("utf-8")
            except UnicodeDecodeError:
                assert got.startswith("ERR"), f"{raw.hex()}: C accepted it ({got})"
            else:
                assert got.startswith("OK"), f"{raw.hex()}: C refused it ({got})"
                code, width = got.split()[1:3]
                assert int(code) == ord(want[0]), f"{raw.hex()}: {got}"
                assert int(width) == len(want[0].encode()), f"{raw.hex()}: {got}"


# --- stage 3: the grammar ------------------------------------------------------------------

def test_the_parser_and_json_loads_accept_exactly_the_same_documents():
    """The grammar, where the C rail has no `json.loads` behind it and must be one.

    Acceptance is compared against `json.loads` rather than against `scan`, because `scan`
    is a bounding pass that deliberately does not check separators. The three documents
    where the two rails legitimately differ are named in the module docstring; none of them
    is an acceptance difference.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        blocks = _blocks(_run(binary, [f"parse 0 {_hex(raw)}" for raw in _DOCS]))
        assert len(blocks) == len(_DOCS)
        for raw, block in zip(_DOCS, blocks):
            want = _reference(raw)
            accepted = block[-1] == "OK"
            if want is None:
                assert not accepted, f"{raw!r}: C accepted what json.loads refused"
            elif "str <unencodable>" in want:
                # A lone surrogate: `json` builds a `str` UTF-8 cannot hold. §7.6.2 makes
                # the document UTF-8, so refusing is correct and accepting is the defect.
                assert not accepted, f"{raw!r}: C accepted a lone surrogate"
                assert _STATUS[int(block[-1].split()[1])] is JerErrorCode.NOT_UTF8, block[-1]
            else:
                assert accepted, f"{raw!r}: C refused what json.loads accepted ({block[-1]})"


def test_the_event_trace_matches_json_loads_value_for_value():
    """The exact tokenization: which name, which decoded contents, which raw number token.

    A reader that gets a document's shape right and a string's contents wrong returns a
    value the sender did not mean, and no structural check catches it.
    """
    accepted = [raw for raw in _DOCS
                if (want := _reference(raw)) is not None and "str <unencodable>" not in want]
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        blocks = _blocks(_run(binary, [f"parse 0 {_hex(raw)}" for raw in accepted]))
        for raw, block in zip(accepted, blocks):
            assert block[-1] == "OK", f"{raw!r}: {block[-1]}"
            assert _strip_offsets(block[:-1]) == _reference(raw), (
                f"{raw!r}:\n  C      {_strip_offsets(block[:-1])}\n"
                f"  Python {_reference(raw)}")


def test_the_trace_offsets_point_at_the_construct_they_name():
    """Every event's offset must address the octet its construct starts at.

    Offsets are what makes a diagnostic actionable, and an event stream with plausible but
    wrong offsets is worse than one with none: it points a reader at the wrong member.
    """
    document = b'{"a": [1, "x"], "bb": {"c": true}}'
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        block = _blocks(_run(binary, [f"parse 0 {_hex(document)}"]))[0]
        assert block[-1] == "OK", block[-1]
        openers = {"{": b"{", "}": b"}", "[": b"[", "]": b"]",
                   "true": b"true", "false": b"false", "null": b"null"}
        for line in block[:-1]:
            parts = line.split()
            offset = int(parts[1])
            if parts[0] in openers:
                assert document[offset:offset + len(openers[parts[0]])] == openers[parts[0]], (
                    f"{line!r} does not address its own construct")
            elif parts[0] in ("key", "str"):
                assert document[offset:offset + 1] == b'"', f"{line!r} is not at a quote"
            else:
                assert document[offset:offset + len(bytes.fromhex(parts[2]))] == \
                    bytes.fromhex(parts[2]), f"{line!r} does not address its own token"


def test_a_sink_can_refuse_mid_walk_and_still_get_a_structured_diagnostic():
    """§4.2's contract has to survive a refusal that comes from the caller, not the input.

    This is the path a schema layer takes: it walks the events, finds a member the type does
    not have, and stops. `BCIR_JER_SINK_REFUSED` carries the sink's own code and the offset
    of the event it refused, so the caller can build the schema path §4.2 also asks for —
    which a reader with no type model cannot fill in itself.
    """
    document = b'{"a": 1, "b": [2, 3]}'
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        blocks = _blocks(_run(binary, [f"refuse {at} {_hex(document)}" for at in range(9)]))
        for at, block in enumerate(blocks):
            assert block[-1].startswith("ERR "), f"refusing at event {at}: {block[-1]}"
            code, offset, sink = block[-1].split()[1:4]
            assert int(code) == 17, f"refusing at event {at}: {block[-1]}"   # SINK_REFUSED
            assert int(sink) == 7, f"the sink's own code was not carried: {block[-1]}"
            assert 0 <= int(offset) < len(document), block[-1]
            assert len(block) == at + 1, (
                f"refusing at event {at} emitted {len(block) - 1} events first")


def test_a_deeper_document_than_the_limit_is_refused_before_the_c_stack_is_touched():
    """Depth is a caller's memory budget, not a property of the C stack.

    A recursive-descent twin would blow the thread's stack on a deeply nested document
    long before any limit fired, and would do it as a crash rather than a diagnostic. The
    parser keeps its nesting in the caller's array, so this refuses cleanly at 65 and
    survives an input a recursive parser would not.
    """
    deep = b"[" * 4096 + b"]" * 4096
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        answers = _run(binary, [f"scan 0 {_hex(deep)}", f"parse 0 {_hex(deep)}"])
        assert answers[0].startswith("ERR "), answers[0]
        assert _STATUS[int(answers[0].split()[1])] is JerErrorCode.DEPTH_EXCEEDED, answers[0]
        assert int(answers[0].split()[3]) == JerLimits().depth + 1, answers[0]
        block = _blocks(answers[1:])[0]
        assert _STATUS[int(block[-1].split()[1])] is JerErrorCode.DEPTH_EXCEEDED, block[-1]


# --- the escape decoder ----------------------------------------------------------------------

_ESCAPES = [
    b"", b"a", b"ab", b"\\n", b"\\t", b"\\r", b"\\b", b"\\f", b"\\/", b"\\\\", b'\\"',
    b"\\u0041", b"\\u00e9", b"\\u4e2d", b"\\uD83D\\uDE00", b"a\\nb\\u0041c",
    b"\\u0000", b"\\u001f", b"\\uffff", b"\\ufffe",
    "é中\U0001f600".encode(),
]


def test_the_escape_decoder_agrees_with_json_character_for_character():
    """ECMA-404 clause 9's nine escapes plus the surrogate pairing §7.6.2 makes mandatory."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        answers = _run(binary, [f"unescape 65536 {_hex(raw)}" for raw in _ESCAPES])
        for raw, got in zip(_ESCAPES, answers):
            want = json.loads(b'"' + raw + b'"')
            assert got.startswith("OK "), f"{raw!r}: {got}"
            payload = got.split(maxsplit=1)[1]
            octets = b"" if payload == "-" else bytes.fromhex(payload)
            assert octets == want.encode(), (
                f"{raw!r}: C produced {octets!r}, json produced {want.encode()!r}")


def test_an_unpaired_surrogate_is_not_utf8_rather_than_malformed():
    """The distinction is the finding this phase produced, so it is pinned by name.

    `"\\ud800"` is well-formed JSON — `json.loads` builds a `str` from it happily — that
    denotes no UTF-8 text at all. §7.6.2 makes a JER document UTF-8, which is why `jer.py`'s
    *encoder* already refused to emit one. The decoder had no matching refusal, so the rail
    could decode a value it could never re-encode; under the canonical profile the
    re-encode then raised an error that escaped `decode_bounded` unstructured, with no code
    and no offset, in direct contradiction of §4.2. Both rails now refuse it in the octet
    pass, and both call it NOT_UTF8 rather than MALFORMED: the JSON is well formed, and it
    is the *encoding* that has no answer.
    """
    cases = [b"\\ud800", b"\\udc00", b"\\ud800\\ud800", b"\\ud800x", b"\\udbff",
             b"\\ud800\\u0041"]
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        answers = _run(binary, [f"unescape 65536 {_hex(raw)}" for raw in cases])
        for raw, got in zip(cases, answers):
            assert got.startswith("ERR "), f"{raw!r}: C accepted it ({got})"
            assert _STATUS[int(got.split()[1])] is JerErrorCode.NOT_UTF8, f"{raw!r}: {got}"
            # And `json` is the contrast: it accepts every one of them.
            assert isinstance(json.loads(b'"' + raw + b'"'), str)
        # The Python rail refuses them in the octet pass, before any value graph exists.
        for raw in cases:
            try:
                scan(b'"' + raw + b'"')
            except JerBoundedError as error:
                assert error.diagnostic.code is JerErrorCode.NOT_UTF8, error.diagnostic
            else:
                raise AssertionError(f"the Python rail accepted {raw!r}")


def test_a_short_output_buffer_reports_the_capacity_it_needed():
    """§4.2's "required capacity", on the one rail that has a caller-owned buffer.

    J2 established that JER hides §7.2.2's SIZE constraints, so a JER reader cannot size its
    buffers from the schema the way every binary rail in this repository does — it must be
    told. That makes "how much would have been enough" load-bearing rather than a nicety,
    and the measuring call (capacity zero) has to produce exactly the figure the writing
    call then fits in.
    """
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        cases = [raw for raw in _ESCAPES if raw]
        measured = _run(binary, [f"unescape 0 {_hex(raw)}" for raw in cases])
        for raw, got in zip(cases, measured):
            want = len(json.loads(b'"' + raw + b'"').encode())
            assert got.startswith("ERR "), f"{raw!r}: {got}"
            code, _offset, needed = got.split()[1:4]
            assert int(code) == 16, f"{raw!r}: {got}"        # OVERFLOW, a C-only status
            assert int(needed) == want, (
                f"{raw!r}: measuring reported {needed}, the value needs {want}")
        # And the figure it reported is exactly enough.
        sizes = [len(json.loads(b'"' + raw + b'"').encode()) for raw in cases]
        exact = _run(binary, [f"unescape {size} {_hex(raw)}"
                              for raw, size in zip(cases, sizes)])
        for raw, got in zip(cases, exact):
            assert got.startswith("OK "), f"{raw!r}: an exact-size buffer was refused ({got})"


# --- §3.3 framing ---------------------------------------------------------------------------

def test_a_frame_is_verified_before_its_payload_is_visible_on_both_rails():
    """§3.3: nothing becomes visible before the frame passes its integrity check.

    Every truncated prefix is refused, a flipped payload octet is refused as an INTEGRITY
    failure and not a malformed one, and a valid frame's fields survive the round trip. The
    CRC is reused from `bcir_runtime.c` and never reimplemented, so the C and Python
    (`zlib.crc32`) rails agree by construction rather than by test.
    """
    payload = b'{"a":1,"b":[true,null]}'
    good = frame(payload, sequence=42, generation=7)
    cases: list[tuple[bytes, str]] = [(good, "ok")]
    cases += [(good[:cut], "short") for cut in range(0, len(good))]
    corrupt = bytearray(good)
    corrupt[FRAME_HEADER_SIZE + 3] ^= 0x01
    cases.append((bytes(corrupt), "integrity"))
    bad_magic = bytearray(good)
    bad_magic[0] = ord("X")
    cases.append((bytes(bad_magic), "magic"))
    bad_version = bytearray(good)
    bad_version[4] = 9
    cases.append((bytes(bad_version), "version"))
    cases.append((good + b"x", "long"))

    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        answers = _run(binary, [f"unframe {_hex(raw)}" for raw, _ in cases])
        for (raw, kind), got in zip(cases, answers):
            if kind == "ok":
                assert got.startswith("OK "), got
                version, sequence, generation, body = got.split()[1:5]
                assert (int(version), int(sequence), int(generation)) == (1, 42, 7), got
                assert bytes.fromhex(body) == payload, got
            elif kind == "integrity":
                assert got.startswith("ERR "), got
                assert _STATUS[int(got.split()[1])] is JerErrorCode.FRAME_INTEGRITY, got
            else:
                assert got.startswith("ERR "), f"{kind}: C accepted a bad frame ({got})"
                assert _STATUS[int(got.split()[1])] is JerErrorCode.FRAME_MALFORMED, \
                    f"{kind}: {got}"


def test_the_two_rails_agree_on_the_frame_header_layout():
    """The header is a fixed 32-octet struct on both rails, and a mismatch there would make
    every field after `magic` mean something different. Checked by construction rather than
    by eye: a frame Python builds is one C reads, field for field."""
    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        rng = random.Random(_SEED)
        cases = [(rng.getrandbits(64), rng.getrandbits(64), bytes(rng.randbytes(n)))
                 for n in (0, 1, 17, 300)]
        built = [frame(body, sequence=seq, generation=gen) for seq, gen, body in cases]
        answers = _run(binary, [f"unframe {_hex(raw)}" for raw in built])
        for (seq, gen, body), got in zip(cases, answers):
            assert got.startswith("OK "), got
            version, sequence, generation, carried = got.split()[1:5]
            assert int(version) == 1, got
            assert int(sequence) == seq, got
            assert int(generation) == gen, got
            assert (b"" if carried == "-" else bytes.fromhex(carried)) == body, got
        # And the layout the C rail reads is the one `struct` writes.
        assert FRAME_HEADER_SIZE == 32
        magic, version, sequence, generation, length, crc = struct.unpack(
            "<4sBxxxQQII", built[2][:FRAME_HEADER_SIZE])
        assert magic == b"BJER" and version == 1
        assert length == len(cases[2][2])
        assert crc == (zlib.crc32(cases[2][2]) & 0xFFFFFFFF)


# --- a randomized sweep -----------------------------------------------------------------------

def test_a_generated_corpus_keeps_the_two_rails_in_step():
    """Random documents, and random mutations of them, through every stage.

    The hand-written corpus above covers the branches somebody thought of. This covers the
    ones nobody did: valid documents built from a grammar, then corrupted one octet at a
    time, which is where a bounds check that is off by one shows up.
    """
    rng = random.Random(_SEED)

    def build(depth: int) -> bytes:
        if depth <= 0 or rng.random() < 0.35:
            return rng.choice([
                b"null", b"true", b"false", b"0", b"-1", b"1.5", b"2e3", b"-0.25e-2",
                b'""', b'"a"', b'"\\n"', b'"\\u0041"', b'"\\uD83D\\uDE00"',
                "\"é中\"".encode(),
            ])
        if rng.random() < 0.5:
            return b"[" + b",".join(build(depth - 1)
                                    for _ in range(rng.randint(0, 4))) + b"]"
        members = [b'"k%d":%s' % (n, build(depth - 1)) for n in range(rng.randint(0, 4))]
        return b"{" + b",".join(members) + b"}"

    corpus: list[bytes] = []
    for _ in range(200):
        document = build(rng.randint(0, 4))
        corpus.append(document)
        if document:
            mutated = bytearray(document)
            at = rng.randrange(len(mutated))
            mutated[at] = rng.randrange(256)
            corpus.append(bytes(mutated))
            corpus.append(document[:rng.randrange(len(document))])

    with tempfile.TemporaryDirectory() as tmp:
        binary = _build(tmp)
        if binary is None:
            return
        scans = _run(binary, [f"scan 0 {_hex(raw)}" for raw in corpus])
        for raw, got in zip(corpus, scans):
            try:
                nodes = scan(raw)
            except JerBoundedError as error:
                assert got.startswith("ERR "), f"{raw!r}: C accepted it ({got})"
                code, offset, needed = got.split()[1:4]
                assert _STATUS[int(code)] is error.diagnostic.code, f"{raw!r}: {got}"
                assert int(offset) == error.diagnostic.offset, f"{raw!r}: {got}"
                assert int(needed) == (error.diagnostic.needed or 0), f"{raw!r}: {got}"
            else:
                assert got == f"OK {nodes}", f"{raw!r}: C said {got}, Python {nodes}"

        blocks = _blocks(_run(binary, [f"parse 0 {_hex(raw)}" for raw in corpus]))
        for raw, block in zip(corpus, blocks):
            want = _reference(raw)
            if want is None or "str <unencodable>" in want:
                assert block[-1] != "OK", f"{raw!r}: C accepted what json.loads refused"
            else:
                assert block[-1] == "OK", f"{raw!r}: C refused it ({block[-1]})"
                assert _strip_offsets(block[:-1]) == want, f"{raw!r}"
