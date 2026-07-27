"""J1 — the bounded JER oracle: limits, canonical-byte validation, framing, diagnostics.

`jer.py` implements X.697. This module implements the part the *roadmap* asks for on top of
it (`docs/BCIR_ASN1_JSON_ROADMAP.md`, phase J1), and it exists because §2 of that roadmap
rates the JER rail's input rejection as **Partial**: the decoder refuses duplicate members
and the non-JSON `NaN`/`Infinity` literals, but it hands the whole input to `json.loads`
with no ceiling on depth, node count, string length or digits, and it accepts non-canonical
bytes under the canonical profile so long as they denote the right value.

Both are fixed here, and the second is the interesting one.

**A canonical decoder must reject non-canonical BYTES.** §3.2: "A canonical decoder must
reject non-canonical bytes, not merely decode them to the same abstract value. Re-encoding
and byte comparison is the initial oracle." That is a stronger claim than it first looks.
`jer.py` already refuses out-of-order members, but member order is one of *six* things the
BCIR profile pins — whitespace, escape spelling, number spelling, DEFAULT omission and
SET OF order are the others — and checking them one at a time is a checklist that drifts
from the encoder. Re-encoding and comparing octets cannot drift: the encoder IS the
definition, so anything it would not have produced is refused by construction. `_canonical`
reports the offset of the first differing octet, so the diagnostic points at the byte rather
than restating the rule.

**Limits are enforced before a value graph exists.** §4.3 requires explicit maxima on input
bytes, depth, nodes, members, elements, string and number-token bytes, digits and exponent
magnitude, and on *total work* "so an input cannot hide quadratic duplicate/member lookup".
`scan` walks the raw octets once and refuses before `json.loads` is ever called, which is
what makes §4.2's "no mutation on failure" trivially true for the whole decode: nothing has
been built yet.

**The scan reads octets, not characters, and that is deliberate.** §4.2 asks diagnostics to
carry a *byte* offset, and a scan over a decoded `str` can only report a character index.
Scanning octets is also what the J3 C twin will do, so the two rails will agree on offsets
rather than needing a translation table. It is safe because every structural character of
JSON is ASCII and every non-ASCII UTF-8 octet has its high bit set, so no multi-byte
sequence can be mistaken for markup — the same property simdjson's structural stage relies
on. UTF-8 *validity* is then checked separately, because structural safety and validity are
different questions.

**Framing is a transaction boundary, not a container format.** §3.3 wants "an explicit
version, length, integrity field, sequence, and generation around each complete document"
with nothing visible "before the complete frame passes lexical, schema, semantic, and
integrity checks". `unframe` therefore verifies the integrity field before returning a
payload at all, and `decode_framed` runs the whole chain — frame, limits, UTF-8, JSON,
schema, canonical bytes — before it returns anything a caller could act on.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, replace
from enum import Enum

from .jer import JerRules, decode_jer, encode_jer
from .schema import Asn1Type
from .tags import Asn1Error


class JerErrorCode(Enum):
    """Stable error codes. §4.2 asks for "a stable error code, byte offset, schema path,
    and required capacity", and stability is the point: a caller may branch on these, so
    they are named rather than derived from a message string."""

    INPUT_TOO_LARGE = "input-too-large"
    DEPTH_EXCEEDED = "depth-exceeded"
    NODES_EXCEEDED = "nodes-exceeded"
    MEMBERS_EXCEEDED = "members-exceeded"
    ELEMENTS_EXCEEDED = "elements-exceeded"
    STRING_TOO_LONG = "string-too-long"
    NUMBER_TOO_LONG = "number-too-long"
    DIGITS_EXCEEDED = "digits-exceeded"
    EXPONENT_EXCEEDED = "exponent-exceeded"
    WORK_EXCEEDED = "work-exceeded"
    MALFORMED = "malformed"
    NOT_UTF8 = "not-utf8"
    TRAILING_INPUT = "trailing-input"
    NOT_CANONICAL = "not-canonical"
    FRAME_MALFORMED = "frame-malformed"
    FRAME_INTEGRITY = "frame-integrity"
    SCHEMA = "schema"


@dataclass(frozen=True)
class JerDiagnostic:
    """§4.2's diagnostic: a stable code, a byte offset, a schema path, and a capacity.

    `needed` answers "how much would have been enough" for a limit failure, which is what
    lets a caller retry with a raised ceiling instead of guessing. It is None when the
    question does not apply.
    """

    code: JerErrorCode
    offset: int = -1
    path: str = ""
    needed: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        parts = [self.code.value]
        if self.offset >= 0:
            parts.append(f"at octet {self.offset}")
        if self.path:
            parts.append(f"at {self.path}")
        if self.needed is not None:
            parts.append(f"needs {self.needed}")
        if self.detail:
            parts.append(self.detail)
        return "JER: " + ", ".join(parts)


class JerBoundedError(Asn1Error):
    """An `Asn1Error` that also carries the structured diagnostic."""

    def __init__(self, diagnostic: JerDiagnostic) -> None:
        super().__init__(str(diagnostic), diagnostic.offset)
        self.diagnostic = diagnostic


def _fail(code: JerErrorCode, offset: int = -1, *, path: str = "",
          needed: int | None = None, detail: str = "") -> JerBoundedError:
    return JerBoundedError(JerDiagnostic(code, offset, path, needed, detail))


@dataclass(frozen=True)
class JerLimits:
    """§4.3's required maxima. Every one of them, with a default a caller may only tighten.

    The defaults are deliberately modest rather than generous: a limit that no realistic
    input reaches is a limit nobody notices is missing. §4.3 says limits "are part of the
    compiled plan and may be tightened by a caller, never silently expanded", which
    `tightened` enforces.
    """

    input_bytes: int = 1 << 20
    depth: int = 64
    nodes: int = 100_000
    members: int = 10_000
    elements: int = 100_000
    string_bytes: int = 1 << 16
    number_bytes: int = 128
    integer_digits: int = 64
    exponent_magnitude: int = 4096
    #: §4.3's last entry. One unit per octet examined plus one per structural event, so a
    #: pathological input cannot buy unbounded work with few bytes.
    work: int = 1 << 24

    def tightened(self, **changes) -> "JerLimits":
        """Return these limits with `changes` applied, refusing any that loosens one."""
        for name, value in changes.items():
            current = getattr(self, name)
            if value > current:
                raise _fail(JerErrorCode.MALFORMED,
                            detail=f"limit {name} may be tightened, not raised from "
                                   f"{current} to {value} (4.3)")
        return replace(self, **changes)


#: §8.1 asks the corpus to cover limit boundaries, so a tiny profile is provided for tests
#: and for callers that know their documents are small control messages.
STRICT_LIMITS = JerLimits(input_bytes=8192, depth=16, nodes=512, members=128,
                          elements=512, string_bytes=1024, number_bytes=40,
                          integer_digits=20, exponent_magnitude=308, work=1 << 18)

_WS = frozenset(b" \t\n\r")
_DIGITS = frozenset(b"0123456789")


def scan(data: bytes, limits: JerLimits = JerLimits()) -> int:
    """Walk the octets once, enforcing every §4.3 limit. Returns the node count.

    This is a *bounding* pass, not a parser: it decides how much work the input may cost and
    refuses beyond that, then leaves the value graph to `json.loads`. It still has to track
    strings and escapes exactly, because a `{` inside a string is not a structural token and
    an input that hid its nesting inside quotes would otherwise walk straight past the depth
    ceiling.
    """
    if len(data) > limits.input_bytes:
        raise _fail(JerErrorCode.INPUT_TOO_LARGE, 0, needed=len(data),
                    detail=f"limit is {limits.input_bytes} octets")
    pos = 0
    end = len(data)
    depth = 0
    nodes = 0
    work = 0
    # One counter per open container, so `members` and `elements` are per-container maxima
    # rather than document-wide totals -- which is what §4.3's wording asks for.
    counts: list[int] = []
    kinds: list[int] = []

    def spend(amount: int, offset: int) -> None:
        nonlocal work
        work += amount
        if work > limits.work:
            raise _fail(JerErrorCode.WORK_EXCEEDED, offset, needed=work,
                        detail=f"limit is {limits.work}")

    while pos < end:
        byte = data[pos]
        spend(1, pos)
        if byte in _WS:
            pos += 1
            continue
        if byte in b"{[":
            depth += 1
            if depth > limits.depth:
                raise _fail(JerErrorCode.DEPTH_EXCEEDED, pos, needed=depth,
                            detail=f"limit is {limits.depth}")
            counts.append(0)
            kinds.append(byte)
            nodes += 1
            if nodes > limits.nodes:
                raise _fail(JerErrorCode.NODES_EXCEEDED, pos, needed=nodes,
                            detail=f"limit is {limits.nodes}")
            pos += 1
            continue
        if byte in b"}]":
            if not kinds:
                raise _fail(JerErrorCode.MALFORMED, pos,
                            detail="a closing bracket with nothing open")
            opened = kinds.pop()
            counts.pop()
            if (opened == 0x7B) != (byte == 0x7D):
                raise _fail(JerErrorCode.MALFORMED, pos,
                            detail="mismatched brackets")
            depth -= 1
            pos += 1
            continue
        if byte == 0x2C:                                     # ","
            if not counts:
                raise _fail(JerErrorCode.MALFORMED, pos,
                            detail="a comma outside any container")
            counts[-1] += 1
            cap = limits.members if kinds[-1] == 0x7B else limits.elements
            code = (JerErrorCode.MEMBERS_EXCEEDED if kinds[-1] == 0x7B
                    else JerErrorCode.ELEMENTS_EXCEEDED)
            if counts[-1] + 1 > cap:
                raise _fail(code, pos, needed=counts[-1] + 1, detail=f"limit is {cap}")
            pos += 1
            continue
        if byte == 0x3A:                                     # ":"
            pos += 1
            continue
        if byte == 0x22:                                     # a string
            pos = _scan_string(data, pos, limits, spend)
            nodes += 1
            if nodes > limits.nodes:
                raise _fail(JerErrorCode.NODES_EXCEEDED, pos, needed=nodes,
                            detail=f"limit is {limits.nodes}")
            continue
        if byte == 0x2D or byte in _DIGITS:                  # a number
            pos = _scan_number(data, pos, limits, spend)
            nodes += 1
            if nodes > limits.nodes:
                raise _fail(JerErrorCode.NODES_EXCEEDED, pos, needed=nodes,
                            detail=f"limit is {limits.nodes}")
            continue
        # `true`, `false`, `null` -- and nothing else. Rejecting here rather than leaving it
        # to `json.loads` is what keeps the non-JSON `NaN`/`Infinity` literals out of the
        # bounded path too (ECMA-404 clause 8 has no such token).
        for literal in (b"true", b"false", b"null"):
            if data.startswith(literal, pos):
                spend(len(literal), pos)
                pos += len(literal)
                nodes += 1
                break
        else:
            raise _fail(JerErrorCode.MALFORMED, pos,
                        detail=f"{data[pos:pos + 12]!r} begins no JSON value")
    if kinds:
        raise _fail(JerErrorCode.MALFORMED, end, detail=f"{len(kinds)} container(s) left "
                                                        f"unclosed at end of input")
    return nodes


def _scan_string(data: bytes, pos: int, limits: JerLimits, spend) -> int:
    """From the opening quote to just past the closing one, counting decoded octets."""
    start = pos
    pos += 1
    decoded = 0
    end = len(data)
    while pos < end:
        byte = data[pos]
        spend(1, pos)
        if byte == 0x22:
            return pos + 1
        if byte == 0x5C:                                     # a backslash escape
            if pos + 1 >= end:
                raise _fail(JerErrorCode.MALFORMED, pos,
                            detail="an escape at end of input")
            following = data[pos + 1]
            if following == 0x75:                            # \\uXXXX
                if pos + 6 > end:
                    raise _fail(JerErrorCode.MALFORMED, pos,
                                detail="a truncated \\u escape")
                pos += 6
                decoded += 3                                 # worst case in UTF-8
            else:
                pos += 2
                decoded += 1
        elif byte < 0x20:
            # ECMA-404 clause 9: a control character may not appear literally in a string.
            raise _fail(JerErrorCode.MALFORMED, pos,
                        detail=f"an unescaped control character U+{byte:04X}")
        else:
            pos += 1
            decoded += 1
        if decoded > limits.string_bytes:
            raise _fail(JerErrorCode.STRING_TOO_LONG, start, needed=decoded,
                        detail=f"limit is {limits.string_bytes}")
    raise _fail(JerErrorCode.MALFORMED, start, detail="an unterminated string")


def _scan_number(data: bytes, pos: int, limits: JerLimits, spend) -> int:
    """Bound the number token, its integer digits and its exponent magnitude.

    The digit and exponent ceilings are separate from the token-length ceiling on purpose:
    `1e999999999` is a short token that denotes a number no ASN.1 real can hold, and
    `1000…0` with a thousand digits is a long token denoting a perfectly ordinary integer.
    §4.3 asks for both, and neither implies the other.
    """
    start = pos
    end = len(data)
    if pos < end and data[pos] == 0x2D:
        pos += 1
    digits = 0
    while pos < end and data[pos] in _DIGITS:
        digits += 1
        pos += 1
        spend(1, pos)
    if digits == 0:
        raise _fail(JerErrorCode.MALFORMED, start, detail="a number with no digits")
    if digits > limits.integer_digits:
        raise _fail(JerErrorCode.DIGITS_EXCEEDED, start, needed=digits,
                    detail=f"limit is {limits.integer_digits}")
    if pos < end and data[pos] == 0x2E:                      # a fraction
        pos += 1
        fraction = 0
        while pos < end and data[pos] in _DIGITS:
            fraction += 1
            pos += 1
            spend(1, pos)
        if fraction == 0:
            raise _fail(JerErrorCode.MALFORMED, pos,
                        detail="a decimal point with no digits after it")
    if pos < end and data[pos] in b"eE":
        pos += 1
        if pos < end and data[pos] in b"+-":
            pos += 1
        exponent_start = pos
        while pos < end and data[pos] in _DIGITS:
            pos += 1
            spend(1, pos)
        if pos == exponent_start:
            raise _fail(JerErrorCode.MALFORMED, pos, detail="an exponent with no digits")
        magnitude = int(data[exponent_start:pos])
        if magnitude > limits.exponent_magnitude:
            raise _fail(JerErrorCode.EXPONENT_EXCEEDED, start, needed=magnitude,
                        detail=f"limit is {limits.exponent_magnitude}")
    if pos - start > limits.number_bytes:
        raise _fail(JerErrorCode.NUMBER_TOO_LONG, start, needed=pos - start,
                    detail=f"limit is {limits.number_bytes}")
    return pos


def _canonical(data: bytes, kind: Asn1Type, value, instructions) -> None:
    """§3.2 — re-encode and compare octets, and point at the first that differs.

    Cheaper checks exist for each individual rule the profile pins, and every one of them
    would be a second definition of canonicality that could drift from the encoder. This
    cannot drift: `encode_jer` under `CANONICAL` *is* the definition.
    """
    again = encode_jer(kind, value, rules=JerRules.CANONICAL, instructions=instructions)
    if again == data:
        return
    offset = next((at for at in range(min(len(again), len(data)))
                   if again[at] != data[at]), min(len(again), len(data)))
    raise _fail(JerErrorCode.NOT_CANONICAL, offset,
                detail=f"the canonical encoding has {again[offset:offset + 12]!r} here, "
                       f"the input has {data[offset:offset + 12]!r}")


def decode_bounded(data: bytes | str, kind: Asn1Type, *,
                   rules: JerRules = JerRules.CANONICAL,
                   limits: JerLimits = JerLimits(),
                   instructions=None):
    """Decode a JER document under explicit limits, in §4.2's order.

    Frame, then structure and limits, then UTF-8, then JSON, then the schema, then canonical
    bytes. Nothing is built until the bounding pass has approved the input, so a refusal at
    any stage leaves the caller exactly as it was — which is the whole of §4.2's "no mutation
    on failure" for a pure decode.
    """
    octets = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    scan(octets, limits)                                     # §4.3, before anything exists
    try:
        octets.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _fail(JerErrorCode.NOT_UTF8, error.start,
                    detail=f"7.6.2 makes the encoding UTF-8: {error.reason}") from None
    try:
        # Always decode with BASIC, even when CANONICAL was asked for. §6.3 makes BASIC the
        # decoder that "shall support all JER encoding alternatives", so this reads the
        # input for what it *is*; the byte comparison below then decides, on its own,
        # whether that was the canonical spelling of it. Splitting the two means exactly
        # one mechanism owns canonicality -- ask `decode_jer` to judge it as well and there
        # are two definitions to keep in step, which is the drift `_canonical` exists to
        # avoid. It also makes the diagnostic uniform: every non-canonical input reports
        # NOT_CANONICAL with the offending octet, not a schema message for some rules and
        # a byte offset for others.
        value = decode_jer(octets, kind, rules=JerRules.BASIC, instructions=instructions)
    except JerBoundedError:
        raise
    except Asn1Error as error:
        raise _fail(JerErrorCode.SCHEMA, getattr(error, "offset", -1),
                    detail=str(error)) from None
    if rules is JerRules.CANONICAL:
        _canonical(octets, kind, value, instructions)
    return value


# --- §3.3 framing -------------------------------------------------------------------------

#: A frame is deliberately tiny and fixed-width: magic, version, sequence, generation,
#: payload length, CRC-32 of the payload. §3.3 names exactly these fields, and nothing here
#: is a container format -- a frame carries one complete document and no more.
FRAME_MAGIC = b"BJER"
FRAME_VERSION = 1
_FRAME_HEADER = struct.Struct("<4sBxxxQQII")
FRAME_HEADER_SIZE = _FRAME_HEADER.size


def frame(payload: bytes, *, sequence: int = 0, generation: int = 0) -> bytes:
    """Wrap one complete document with §3.3's version, sequence, generation, length and CRC."""
    if not isinstance(payload, (bytes, bytearray)):
        raise _fail(JerErrorCode.FRAME_MALFORMED, detail="a frame payload is octets")
    body = bytes(payload)
    return _FRAME_HEADER.pack(FRAME_MAGIC, FRAME_VERSION, sequence, generation,
                              len(body), zlib.crc32(body) & 0xFFFFFFFF) + body


@dataclass(frozen=True)
class Frame:
    version: int
    sequence: int
    generation: int
    payload: bytes


def unframe(data: bytes) -> Frame:
    """Verify the frame *before* returning any payload.

    §3.3: "No claim or artifact becomes visible before the complete frame passes lexical,
    schema, semantic, and integrity checks." Integrity is the part this function owns, and
    it is checked before the payload is handed back rather than alongside it, so a truncated
    or corrupted frame never yields octets a caller might act on.
    """
    octets = bytes(data)
    if len(octets) < FRAME_HEADER_SIZE:
        raise _fail(JerErrorCode.FRAME_MALFORMED, 0, needed=FRAME_HEADER_SIZE,
                    detail=f"a frame header is {FRAME_HEADER_SIZE} octets")
    magic, version, sequence, generation, length, crc = _FRAME_HEADER.unpack(
        octets[:FRAME_HEADER_SIZE])
    if magic != FRAME_MAGIC:
        raise _fail(JerErrorCode.FRAME_MALFORMED, 0,
                    detail=f"expected {FRAME_MAGIC!r}, got {magic!r}")
    if version != FRAME_VERSION:
        raise _fail(JerErrorCode.FRAME_MALFORMED, 4,
                    detail=f"frame version {version} is not {FRAME_VERSION}")
    want = FRAME_HEADER_SIZE + length
    if len(octets) != want:
        raise _fail(JerErrorCode.FRAME_MALFORMED, FRAME_HEADER_SIZE, needed=want,
                    detail=f"the frame declares {length} payload octets and carries "
                           f"{len(octets) - FRAME_HEADER_SIZE}")
    payload = octets[FRAME_HEADER_SIZE:]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        # An integrity failure, never an authenticity one -- §6.3 of the roadmap keeps
        # those separate and so does this message.
        raise _fail(JerErrorCode.FRAME_INTEGRITY, FRAME_HEADER_SIZE,
                    detail="the payload CRC-32 does not match; this detects corruption "
                           "and is not a signature")
    return Frame(version, sequence, generation, payload)


def encode_framed(kind: Asn1Type, value, *, rules: JerRules = JerRules.CANONICAL,
                  sequence: int = 0, generation: int = 0, instructions=None) -> bytes:
    return frame(encode_jer(kind, value, rules=rules, instructions=instructions),
                 sequence=sequence, generation=generation)


def decode_framed(data: bytes, kind: Asn1Type, *,
                  rules: JerRules = JerRules.CANONICAL,
                  limits: JerLimits = JerLimits(), instructions=None):
    """Frame, limits, UTF-8, JSON, schema, canonical bytes — then, and only then, a value."""
    return decode_bounded(unframe(data).payload, kind, rules=rules, limits=limits,
                          instructions=instructions)


__all__ = [
    "FRAME_HEADER_SIZE", "FRAME_MAGIC", "FRAME_VERSION", "STRICT_LIMITS", "Frame",
    "JerBoundedError", "JerDiagnostic", "JerErrorCode", "JerLimits", "decode_bounded",
    "decode_framed", "encode_framed", "frame", "scan", "unframe",
]
