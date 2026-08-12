"""Contents octets for every universal type in X.690 clause 8 (02/2021).

One module per concern would scatter the clause; keeping the whole of clause 8 here
means the normative text and the code stay in reading order, and every function names
the subclause it implements. Each type has an `encode_*` producing the contents octets
in the form DER requires, and a `decode_*` accepting every form BER permits — that
asymmetry *is* the "DER out, BER in" contract, and it lives here rather than in the
caller.

Contents octets only: identifier and length octets belong to `tags`/`length`, and the
structural walk to `tlv`. A few types (bitstring, octetstring, character strings) may
arrive constructed, so their decoders take a `Tlv` rather than raw octets.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .tags import Asn1Error, Tag, TagClass, Universal
from .tlv import Tlv

# --- 8.2 boolean ------------------------------------------------------------------


def encode_boolean(value: bool) -> bytes:
    """§8.2 + §11.1: TRUE is 0xFF (BER allows any non-zero; DER pins all-ones)."""
    return b"\xff" if value else b"\x00"


def decode_boolean(content: bytes, *, der: bool = False) -> bool:
    """§8.2.1: exactly one contents octet. Under DER, §11.1 pins TRUE to 0xFF."""
    if len(content) != 1:
        raise Asn1Error(
            f"boolean contents must be a single octet (X.690 8.2.1), got {len(content)}")
    octet = content[0]
    if der and octet not in (0x00, 0xFF):
        raise Asn1Error(
            f"DER boolean TRUE must be 0xFF (X.690 11.1), got 0x{octet:02x}")
    return octet != 0


# --- 8.3 integer, 8.4 enumerated ---------------------------------------------------


def encode_integer(value: int) -> bytes:
    """§8.3.3: two's complement, §8.3.2: in the fewest possible octets.

    The `+ (value < 0)` is load-bearing: two's complement reaches one value further
    negative than positive at each width, so -128 fits one octet while 128 needs two.
    Taking `bit_length()` of the raw negative value overstates the width at exactly
    those boundaries and emits a redundant leading 0xFF, which §8.3.2 a) forbids.
    """
    width = ((value + (value < 0)).bit_length() // 8) + 1
    return value.to_bytes(width, "big", signed=True)


def decode_integer(content: bytes) -> int:
    """§8.3: reject the padded forms §8.3.2 rules out on every rail.

    §8.3.2's NOTE calls this out as the rule that "ensure[s] an integer value is
    always encoded in the smallest possible number of octets" — it is a BER rule, not
    a DER-only one, so a padded integer is rejected even in permissive mode.
    """
    if not content:
        raise Asn1Error("integer contents must be at least one octet (X.690 8.3.1)")
    if len(content) > 1:
        first, second = content[0], content[1]
        if first == 0x00 and not second & 0x80:
            raise Asn1Error(
                "integer has a redundant leading 0x00 octet (X.690 8.3.2 b)")
        if first == 0xFF and second & 0x80:
            raise Asn1Error(
                "integer has a redundant leading 0xFF octet (X.690 8.3.2 a)")
    return int.from_bytes(content, "big", signed=True)


#: §8.4: an enumerated value is encoded as the integer it is associated with.
encode_enumerated = encode_integer
decode_enumerated = decode_integer


# --- 8.5 real ----------------------------------------------------------------------

#: §8.5.9 special-value contents octets.
REAL_PLUS_INFINITY = 0x40
REAL_MINUS_INFINITY = 0x41
REAL_NOT_A_NUMBER = 0x42
REAL_MINUS_ZERO = 0x43


def encode_real(value: float) -> bytes:
    """§8.5 with the §11.3.1 canonical form: base 2, F = 0, mantissa odd or zero.

    §8.5.2 makes plus zero the empty encoding; §8.5.3 sends minus zero through the
    special-value path. For everything finite and non-zero, §11.3.1 requires the
    mantissa be repeatedly halved until its least significant bit is 1, so that a
    single real value has exactly one encoding.
    """
    if value != value:                                   # NaN
        return bytes([REAL_NOT_A_NUMBER])
    if value == math.inf:
        return bytes([REAL_PLUS_INFINITY])
    if value == -math.inf:
        return bytes([REAL_MINUS_INFINITY])
    if value == 0.0:
        return bytes([REAL_MINUS_ZERO]) if math.copysign(1.0, value) < 0 else b""

    mantissa, exponent = math.frexp(abs(value))          # 0.5 <= mantissa < 1
    mantissa_int = int(mantissa * (1 << 53))
    exponent -= 53
    while mantissa_int and not mantissa_int & 1:         # §11.3.1: make M odd
        mantissa_int >>= 1
        exponent += 1

    exponent_octets = exponent.to_bytes(
        (exponent.bit_length() // 8) + 1, "big", signed=True)
    first = 0x80                                          # §8.5.6 a): binary encoding
    if math.copysign(1.0, value) < 0:
        first |= 0x40                                     # §8.5.7.1: S = -1
    # bits 6-5 = 00 (base 2, §8.5.7.2); bits 4-3 = 00 (F = 0, §11.3.1)
    if len(exponent_octets) <= 3:
        first |= len(exponent_octets) - 1                 # §8.5.7.4 a)-c)
        prefix = bytes([first])
    else:
        first |= 0b11                                     # §8.5.7.4 d)
        prefix = bytes([first, len(exponent_octets)])
    body = mantissa_int.to_bytes((mantissa_int.bit_length() + 7) // 8 or 1, "big")
    return prefix + exponent_octets + body


def decode_real(content: bytes, *, der: bool = False) -> float:
    """§8.5: binary (any of base 2/8/16, any exponent form), decimal, and specials.

    `der` adds clause 11.3, which removes every sender's option §8.5 grants. It is a
    separate argument rather than the default because the BER-in half of the contract must
    keep accepting what §8.5 permits -- the point of checking is to say which clause an
    arriving encoding broke, not to refuse to read it.
    """
    if not content:
        return 0.0                                        # §8.5.2: plus zero
    first = content[0]
    if first & 0x80:
        return _decode_real_binary(content, der=der)
    if first & 0x40:
        return _decode_real_special(content)
    return _decode_real_decimal(content, der=der)


def _decode_real_special(content: bytes) -> float:
    """§8.5.9: exactly one contents octet, one of four assigned values."""
    if len(content) != 1:
        raise Asn1Error(
            "a SpecialRealValue encoding has exactly one contents octet (X.690 8.5.9)")
    octet = content[0]
    if octet == REAL_PLUS_INFINITY:
        return math.inf
    if octet == REAL_MINUS_INFINITY:
        return -math.inf
    if octet == REAL_NOT_A_NUMBER:
        return math.nan
    if octet == REAL_MINUS_ZERO:
        return -0.0
    raise Asn1Error(
        f"0x{octet:02x} is reserved in the SpecialRealValue space (X.690 8.5.9)")


def _decode_real_binary(content: bytes, *, der: bool = False) -> float:
    """§8.5.7: sign, base, binary scaling factor F, and one of four exponent forms."""
    first = content[0]
    sign = -1.0 if first & 0x40 else 1.0
    base_bits = (first >> 4) & 0b11
    if base_bits == 0b11:
        raise Asn1Error("real base bits 11 are reserved (X.690 8.5.7.2)")
    base = (2, 8, 16)[base_bits]
    scale = (first >> 2) & 0b11                           # §8.5.7.3
    form = first & 0b11                                   # §8.5.7.4

    pos = 1
    if form == 0b11:
        if len(content) < 2:
            raise Asn1Error("truncated real exponent-length octet (X.690 8.5.7.4 d)")
        count = content[1]
        if count < 1:
            raise Asn1Error("real exponent octet count must be >= 1 (X.690 8.5.7.4 d)")
        pos = 2
    else:
        count = form + 1
    if pos + count > len(content):
        raise Asn1Error("truncated real exponent octets (X.690 8.5.7.4)")
    exp_octets = content[pos:pos + count]
    man_octets = content[pos + count:]
    exponent = int.from_bytes(exp_octets, "big", signed=True)
    mantissa = int.from_bytes(man_octets, "big") if man_octets else 0
    if der:
        _require_der_binary(base_bits, scale, exp_octets, man_octets)
    # §8.5.7: M = S x N x 2^F, and the encoded value is M x B'^E.
    return sign * mantissa * (2 ** scale) * (float(base) ** exponent)


def _require_der_binary(base_bits: int, scale: int, exp_octets: bytes,
                        man_octets: bytes) -> None:
    """§11.3.1: base 2, F zero, an ODD mantissa, and both fields in the fewest octets.

    §8.5.4 makes base 8 and 16 "a sender's option", and a sender's option is precisely what
    DER removes. The mantissa rule is the one with teeth: §11.3.1's NOTE spells out that
    {M, 2, E} and {M x 2^-n, 2, E + n} are the same real value, so without "M is either 0
    or is odd" every value has unboundedly many encodings -- `80 01 02` and `80 02 01` are
    both 4.0. That is a digest collision on a type BCIR digests.

    §8.5.2 gives plus zero NO contents octets, so a binary encoding whose mantissa is zero
    is a second spelling of a value already spelled exactly once.
    """
    if base_bits != 0:
        raise Asn1Error(
            f"DER uses base 2; this encoding declares base {(2, 8, 16)[base_bits]} "
            f"(X.690 11.3.1, and 8.5.4 makes the base a sender's option BER only)")
    if scale != 0:
        raise Asn1Error(
            f"the binary scaling factor F shall be zero; got {scale} (X.690 11.3.1)")
    if not man_octets or int.from_bytes(man_octets, "big") == 0:
        raise Asn1Error(
            "a zero mantissa spells the value zero, which X.690 8.5.2 encodes with no "
            "contents octets at all (X.690 11.3.1)")
    if int.from_bytes(man_octets, "big") % 2 == 0:
        raise Asn1Error(
            "the mantissa shall be odd, so that one real value has one encoding "
            "(X.690 11.3.1)")
    if man_octets[0] == 0x00:
        raise Asn1Error(
            "the mantissa is not in the fewest octets necessary (X.690 11.3.1)")
    if len(exp_octets) > 1 and exp_octets[0] in (0x00, 0xFF) \
            and (exp_octets[1] & 0x80) == (0x80 if exp_octets[0] == 0xFF else 0x00):
        raise Asn1Error(
            "the exponent is not in the fewest octets necessary (X.690 11.3.1)")


#: §8.5.8 selects an ISO 6093 number representation in bits 6 to 1.
_NR_FORMS = {0b000001: "NR1", 0b000010: "NR2", 0b000011: "NR3"}


#: ISO 6093's three number representations, as X.690 §8.5.8 selects them. The grammars are
#: written out rather than delegated to `float()` because Python's literal grammar is a
#: STRICT SUPERSET of every one of them: it accepts PEP 515 underscores (`1_0`), the words
#: `nan`/`inf`/`Infinity`, surrounding whitespace, and -- via `int`/`float`'s Unicode
#: handling -- non-ASCII decimal digits. Each of those is a byte string that is not a number
#: in ISO 6093 at all, so accepting it means this decoder and a conforming peer disagree
#: about what arrived. Two of those disagreements are also second spellings of a value that
#: X.690 §8.5.9 already encodes exactly once (`NaN`, `inf`), which is what a canonical
#: encoding exists to prevent.
_NR1 = re.compile(r"\A[+-]?[0-9]+\Z")
_NR2 = re.compile(r"\A[+-]?(?:[0-9]+[.,][0-9]*|[0-9]*[.,][0-9]+)\Z")
_NR3 = re.compile(r"\A[+-]?(?:[0-9]+[.,][0-9]*|[0-9]*[.,][0-9]+)[Ee][+-]?[0-9]+\Z")
_NR_GRAMMAR = {0b000001: _NR1, 0b000010: _NR2, 0b000011: _NR3}


def _decode_real_decimal(content: bytes, *, der: bool = False) -> float:
    """§8.5.8: an ISO 6093 NR1/NR2/NR3 field in the octets after the first.

    The declared form is enforced, not merely recorded. Reading the selector and then
    handing the field to a general number parser meant `NR1` -- ISO 6093's *integer* form --
    accepted `1.5e3`, so the selector carried no information and a peer had three spellings
    where the clause gives one.
    """
    form = content[0] & 0b111111
    if form not in _NR_FORMS:
        raise Asn1Error(
            f"ISO 6093 number-representation selector {form:06b} is reserved "
            f"(X.690 8.5.8)")
    try:
        raw = content[1:].decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise Asn1Error(
            "an ISO 6093 field is ASCII; a non-ASCII octet is not a digit "
            "(X.690 8.5.8)") from exc
    if not raw:
        raise Asn1Error("decimal real has an empty ISO 6093 field (X.690 8.5.8)")

    # §11.3.2.2 forbids SPACE outright; ISO 6093 permits leading SPACE otherwise.
    text = raw if der else raw.lstrip(" ")
    if der and raw != raw.strip(" "):
        raise Asn1Error(
            f"decimal real {raw!r} contains SPACE; DER forbids it (X.690 11.3.2.2)")
    if not _NR_GRAMMAR[form].match(text):
        raise Asn1Error(
            f"{text!r} is not an ISO 6093 {_NR_FORMS[form]} field (X.690 8.5.8)")
    if der:
        _require_der_nr3(form, text)
    return float(text.replace(",", "."))                  # ISO 6093 allows the comma


def _require_der_nr3(form: int, text: str) -> None:
    """§11.3.2's five restrictions on the decimal form a DER encoder may write."""
    if form != 0b000011:
        raise Asn1Error(
            f"DER uses the ISO 6093 NR3 form; this encoding declares "
            f"{_NR_FORMS[form]} (X.690 11.3.2.1)")
    if text[0] == "+":
        raise Asn1Error(
            "a non-negative decimal real begins with a digit, not PLUS SIGN "
            "(X.690 11.3.2.3)")
    mantissa, _, exponent = text.replace(",", ".").partition("E" if "E" in text else "e")
    digits = mantissa.lstrip("-")
    whole, _, frac = digits.partition(".")
    if not whole or whole[0] == "0" or not frac or frac[-1] == "0":
        raise Asn1Error(
            f"neither the first nor the last digit of the mantissa may be 0, and the "
            f"last mantissa digit is immediately followed by FULL STOP; got "
            f"{mantissa!r} (X.690 11.3.2.4/11.3.2.5)")
    if exponent == "+0":
        return                                            # §11.3.2.6: the one legal "+0"
    if exponent.startswith("+"):
        raise Asn1Error(
            f"a non-zero exponent does not use PLUS SIGN; got {exponent!r} "
            f"(X.690 11.3.2.6)")
    if exponent.lstrip("-")[0] == "0":
        raise Asn1Error(
            f"a non-zero exponent's first digit is not zero; got {exponent!r} "
            f"(X.690 11.3.2.6)")


def encode_real_decimal(value: float) -> bytes:
    """§8.5.8 + §11.3.2: the NR3 form, no SPACE, normalized as DER requires.

    §11.3.2.4 forbids a leading or trailing zero digit in the mantissa and §11.3.2.6
    forbids a PLUS SIGN or leading zero in a non-zero exponent, so the canonical
    spelling of a decimal real is unique.
    """
    if value == 0.0:
        return b""                                        # §8.5.2 takes precedence
    text = repr(float(value))
    mantissa, _, exp = text.partition("e")
    exponent = int(exp) if exp else 0
    if "." in mantissa:
        whole, _, frac = mantissa.partition(".")
        frac = frac.rstrip("0")
        neg = whole.startswith("-")
        digits = (whole.lstrip("-") + frac).lstrip("0") or "0"
        exponent -= len(frac)
        mantissa = ("-" if neg else "") + digits
    stripped = mantissa.lstrip("-")
    while len(stripped) > 1 and stripped.endswith("0"):   # §11.3.2.4
        stripped = stripped[:-1]
        exponent += 1
    mantissa = ("-" if value < 0 else "") + stripped
    return bytes([0b00000011]) + f"{mantissa}.E{exponent:+d}".replace(
        "+", "+" if exponent == 0 else "").encode("ascii")


# --- 8.6 bitstring -----------------------------------------------------------------


@dataclass(frozen=True)
class BitString:
    """A bitstring: the octets holding the bits, plus how many trailing bits are unused.

    Kept as (octets, unused) rather than a Python int so a value's *length in bits* —
    which X.690 preserves and which callers such as X.509 key usage depend on —
    survives a round trip. `unused` is 0..7 per §8.6.2.2.
    """

    octets: bytes = b""
    unused: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.unused <= 7:
            raise Asn1Error(
                f"unused-bit count must be 0..7 (X.690 8.6.2.2), got {self.unused}")
        if not self.octets and self.unused:
            raise Asn1Error(
                "an empty bitstring must declare zero unused bits (X.690 8.6.2.3)")

    @property
    def bit_length(self) -> int:
        return len(self.octets) * 8 - self.unused

    def __getitem__(self, index: int) -> int:
        if not 0 <= index < self.bit_length:
            raise IndexError(index)
        return (self.octets[index // 8] >> (7 - index % 8)) & 1


def encode_bitstring(value: BitString) -> bytes:
    """§8.6.2 + §11.2.1: the initial unused-bit octet, with unused bits set to zero."""
    if not value.octets:
        return b"\x00"                                    # §8.6.2.3
    body = bytearray(value.octets)
    if value.unused:                                      # §11.2.1
        body[-1] &= (0xFF << value.unused) & 0xFF
    return bytes([value.unused]) + bytes(body)


def decode_bitstring(tlv: Tlv, *, der: bool = False) -> BitString:
    """§8.6: primitive, or the constructed segmentation of §8.6.3/§8.6.4.

    A constructed bitstring is a series of bitstring segments, each of which carries
    its own unused-bit octet; §8.6.4 requires every segment but the last to hold a
    multiple of eight bits, so only the final segment may declare unused bits.
    """
    if tlv.constructed:
        if der:
            raise Asn1Error(
                "DER forbids the constructed form for bitstring (X.690 10.2)",
                tlv.offset)
        octets = bytearray()
        unused = 0
        for i, child in enumerate(tlv.children):
            if child.tag.number != Universal.BIT_STRING or not child.tag.is_universal:
                raise Asn1Error(
                    "a constructed bitstring segment must itself be a bitstring "
                    "(X.690 8.6.4.1 NOTE 2)", child.offset)
            part = decode_bitstring(child, der=der)
            if unused:
                raise Asn1Error(
                    "only the last segment of a constructed bitstring may have unused "
                    "bits (X.690 8.6.4)", child.offset)
            octets += part.octets
            unused = part.unused
            del i
        return BitString(bytes(octets), unused)

    content = tlv.content
    if not content:
        raise Asn1Error(
            "bitstring contents need an initial unused-bit octet (X.690 8.6.2)",
            tlv.offset)
    unused = content[0]
    if unused > 7:
        raise Asn1Error(
            f"unused-bit count must be 0..7 (X.690 8.6.2.2), got {unused}", tlv.offset)
    body = content[1:]
    if not body and unused:
        raise Asn1Error(
            "an empty bitstring must declare zero unused bits (X.690 8.6.2.3)",
            tlv.offset)
    if der and body and body[-1] & ((1 << unused) - 1):
        raise Asn1Error(
            "DER requires each unused bit in the final octet to be zero (X.690 11.2.1)",
            tlv.offset)
    return BitString(body, unused)


# --- 8.7 octetstring, 8.8 null -----------------------------------------------------


def decode_octetstring(tlv: Tlv, *, der: bool = False) -> bytes:
    """§8.7: primitive contents, or the concatenated segments of §8.7.3."""
    if not tlv.constructed:
        return tlv.content
    if der:
        raise Asn1Error(
            "DER forbids the constructed form for octetstring (X.690 10.2)", tlv.offset)
    out = bytearray()
    for child in tlv.children:
        if child.tag.number != Universal.OCTET_STRING or not child.tag.is_universal:
            raise Asn1Error(
                "a constructed octetstring segment must itself be an octetstring "
                "(X.690 8.7.3.2 NOTE 2)", child.offset)
        out += decode_octetstring(child, der=der)
    return bytes(out)


def decode_null(content: bytes) -> None:
    """§8.8.2: the contents octets shall not contain any octets."""
    if content:
        raise Asn1Error(
            f"null contents must be empty (X.690 8.8.2), got {len(content)} octet(s)")
    return None


# --- 8.19 object identifier, 8.20 relative object identifier -----------------------


def _encode_subidentifier(value: int) -> bytes:
    """Base-128, most significant group first, bit 8 set on all but the last octet."""
    groups = [value & 0x7F]
    value >>= 7
    while value:
        groups.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(groups))


def _decode_subidentifiers(content: bytes, what: str) -> list[int]:
    """§8.19.2/§8.20.2, including the "fewest possible octets" rule (no leading 0x80)."""
    out: list[int] = []
    value = 0
    started = False
    for i, octet in enumerate(content):
        if not started and octet == 0x80:
            raise Asn1Error(
                f"{what} subidentifier has a redundant leading 0x80 octet "
                f"(X.690 8.19.2: encoded in the fewest possible octets)", i)
        started = True
        value = (value << 7) | (octet & 0x7F)
        if value > 1 << 128:
            raise Asn1Error(f"{what} subidentifier exceeds the decode bound", i)
        if not octet & 0x80:
            out.append(value)
            value = 0
            started = False
    if started:
        raise Asn1Error(f"{what} ends mid-subidentifier (X.690 8.19.2)", len(content))
    return out


def encode_oid(arcs: tuple[int, ...] | list[int]) -> bytes:
    """§8.19.4: the first two components pack into one subidentifier as X*40 + Y."""
    arcs = tuple(arcs)
    if len(arcs) < 2:
        raise Asn1Error(
            "an object identifier has at least two components (X.690 8.19.4)")
    if any(a < 0 for a in arcs):
        raise Asn1Error("object identifier components must be non-negative")
    first, second = arcs[0], arcs[1]
    if first > 2:
        raise Asn1Error(
            f"the first object identifier component is 0, 1 or 2 (X.690 8.19.4 NOTE), "
            f"got {first}")
    if first < 2 and second >= 40:
        raise Asn1Error(
            f"under arc {first} the second component is 0..39 (X.690 8.19.4 NOTE), "
            f"got {second}")
    out = _encode_subidentifier(first * 40 + second)
    for arc in arcs[2:]:
        out += _encode_subidentifier(arc)
    return out


def decode_oid(content: bytes) -> tuple[int, ...]:
    """§8.19: unpack X*40 + Y back into the first two components."""
    if not content:
        raise Asn1Error("object identifier contents must not be empty (X.690 8.19.2)")
    subs = _decode_subidentifiers(content, "object identifier")
    head = subs[0]
    if head < 40:
        first, second = 0, head
    elif head < 80:
        first, second = 1, head - 40
    else:
        first, second = 2, head - 80
    return (first, second, *subs[1:])


def encode_relative_oid(arcs: tuple[int, ...] | list[int]) -> bytes:
    """§8.20.4: every arc is its own subidentifier — no X*40 + Y packing."""
    if any(a < 0 for a in arcs):
        raise Asn1Error("relative OID arcs must be non-negative")
    return b"".join(_encode_subidentifier(a) for a in arcs)


def decode_relative_oid(content: bytes) -> tuple[int, ...]:
    """§8.20."""
    return tuple(_decode_subidentifiers(content, "relative object identifier"))


# --- 8.21/8.22 OID internationalized resource identifiers --------------------------


def encode_oid_iri(value: str) -> bytes:
    """§8.21.2: the UTF-8 encoding of the XML value notation, no white space."""
    if not value.startswith("/"):
        raise Asn1Error(
            f"an OID-IRI value begins with a solidus (X.680 34.3), got {value!r}")
    return value.encode("utf-8")


def decode_oid_iri(content: bytes) -> str:
    try:
        value = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Asn1Error(f"OID-IRI contents are not valid UTF-8: {exc}") from exc
    if not value.startswith("/"):
        raise Asn1Error("an OID-IRI value begins with a solidus (X.680 34.3)")
    return value


def encode_relative_oid_iri(value: str) -> bytes:
    """§8.22.2: as §8.21 but without the leading solidus."""
    return value.encode("utf-8")


def decode_relative_oid_iri(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Asn1Error(f"RELATIVE-OID-IRI contents are not valid UTF-8: {exc}") from exc


# --- X.680 clause 12's digit productions --------------------------------------------------
#
# Every numeric field in the two TEXT transfer syntaxes -- XER's XMLSignedNumber, realnumber
# and XMLObjectIdentifierValue, JER's object-identifier string -- is spelled with these and
# nothing else. X.680 §12.26 says it outright for an arc: "an arbitrarily long sequence of
# ISO/IEC 10646 characters in the range 0 (DIGIT ZERO) to 9 (DIGIT NINE)".
#
# That has to be checked with a predicate that means the same thing. Python's `str.isdigit()`
# and a regex `\d` are both Unicode-aware, so they answer True for ARABIC-INDIC DIGIT EIGHT
# and FULLWIDTH DIGIT FOUR -- characters the production does not contain. Those rails carry
# UTF-8 by design, so unlike the octet-based rails there is no earlier ASCII decode to catch
# it, and `int()` then converts them happily: `<S>٤٢</S>` decoded to the INTEGER 42.

def is_ascii_digits(text: str) -> bool:
    """True when `text` is one or more characters from DIGIT ZERO to DIGIT NINE."""
    return bool(text) and all("0" <= character <= "9" for character in text)


def is_number_form(text: str) -> bool:
    """§12.8's `number`, and §12.26's arc: ASCII digits, no leading zero unless single.

    "A 'number' shall consist of one or more digits. The first digit shall not be zero
    unless the 'number' is a single digit." Without the second half `1.2.0840` and
    `1.2.840` are two spellings of one object identifier.
    """
    return is_ascii_digits(text) and (len(text) == 1 or text[0] != "0")


# --- 8.23 restricted character strings ---------------------------------------------

#: §8.23.5/§8.23.7/§8.23.8/§8.23.10: how each restricted string maps to octets. The
#: ISO/IEC 2022 registered sets that NumericString..GeneralString draw on are all
#: ASCII-compatible in their assumed G0 (registration 6) state, which is the only
#: state a conforming encoder may use for the types whose Table 3 row forbids explicit
#: escape sequences.
_STRING_CODECS: dict[int, str] = {
    Universal.UTF8_STRING: "utf-8",                       # §8.23.10
    Universal.NUMERIC_STRING: "ascii",                    # §8.23.4 via VisibleString
    Universal.PRINTABLE_STRING: "ascii",
    Universal.IA5_STRING: "ascii",
    Universal.VISIBLE_STRING: "ascii",
    Universal.GRAPHIC_STRING: "utf-8",
    Universal.GENERAL_STRING: "utf-8",
    Universal.TELETEX_STRING: "latin-1",                  # T.61 approximated
    Universal.VIDEOTEX_STRING: "latin-1",
    Universal.UNIVERSAL_STRING: "utf-32-be",              # §8.23.7: 4-octet canonical
    Universal.BMP_STRING: "utf-16-be",                    # §8.23.8: 2-octet BMP
    Universal.OBJECT_DESCRIPTOR: "utf-8",                 # §8.25 -> GraphicString
}

#: X.680 §41.4 Table 10: the PrintableString repertoire, and NumericString's digits
#: plus SPACE. Enforced because these are *restricted* types — a decoder that accepts
#: out-of-repertoire octets silently widens the type.
_PRINTABLE = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '()+,-./:=?")
_NUMERIC = frozenset("0123456789 ")


def encode_string(tag_number: int, value: str) -> bytes:
    """The octet string of §8.23.3 for the given universal character string type."""
    codec = _STRING_CODECS.get(tag_number)
    if codec is None:
        raise Asn1Error(f"UNIVERSAL {tag_number} is not a restricted character string")
    _check_repertoire(tag_number, value)
    try:
        return value.encode(codec)
    except UnicodeEncodeError as exc:
        raise Asn1Error(
            f"value is outside the {Universal(tag_number).name} repertoire: {exc}"
        ) from exc


def decode_string(tlv: Tlv, *, der: bool = False) -> str:
    """§8.23 with §8.23.6's "receivers are required to handle all permitted forms"."""
    number = tlv.tag.number
    codec = _STRING_CODECS.get(number)
    if codec is None:
        raise Asn1Error(f"UNIVERSAL {number} is not a restricted character string",
                        tlv.offset)
    if tlv.constructed and der:
        raise Asn1Error(
            "DER forbids the constructed form for character strings (X.690 10.2)",
            tlv.offset)
    # §8.23.3 encodes every character string "as if it had been declared
    # [UNIVERSAL x] IMPLICIT OCTET STRING", and §8.14.4's implicit tag replaces only the
    # OUTERMOST tag -- so §8.7.3.2's recursion runs over an octetstring and its NOTE 2
    # applies verbatim: the segment tags "are always universal class, number 4". That is
    # exactly what §8.23.5's own worked example encodes (`3a09 04034a6f6e 04026573`).
    octets = (tlv.flatten_content(Universal.OCTET_STRING) if tlv.constructed
              else tlv.content)
    if number in (Universal.UNIVERSAL_STRING, Universal.BMP_STRING):
        width = 4 if number == Universal.UNIVERSAL_STRING else 2
        if len(octets) % width:
            raise Asn1Error(
                f"{Universal(number).name} contents must be a multiple of {width} "
                f"octets (X.690 8.23.{7 if width == 4 else 8})", tlv.offset)
    try:
        value = octets.decode(codec)
    except UnicodeDecodeError as exc:
        raise Asn1Error(
            f"{Universal(number).name} contents are not valid {codec}: {exc}",
            tlv.offset) from exc
    _check_repertoire(number, value, offset=tlv.offset)
    return value


def _check_repertoire(number: int, value: str, offset: int = -1) -> None:
    if number == Universal.PRINTABLE_STRING:
        bad = sorted(set(value) - _PRINTABLE)
        if bad:
            raise Asn1Error(
                f"character(s) {bad!r} are outside the PrintableString repertoire "
                f"(X.680 41.4)", offset)
    elif number == Universal.NUMERIC_STRING:
        bad = sorted(set(value) - _NUMERIC)
        if bad:
            raise Asn1Error(
                f"character(s) {bad!r} are outside the NumericString repertoire "
                f"(X.680 41.2)", offset)
    elif number in (Universal.IA5_STRING, Universal.VISIBLE_STRING):
        limit = 128
        bad = sorted({c for c in value if ord(c) >= limit})
        if bad:
            raise Asn1Error(
                f"character(s) {bad!r} are outside the "
                f"{Universal(number).name} repertoire", offset)


# --- 8.25 useful types: UTCTime and GeneralizedTime --------------------------------

_UTCTIME_DER = re.compile(r"^\d{12}Z$")
_GENTIME_DER = re.compile(r"^\d{14}(\.\d+)?Z$")


def encode_utctime(value: str) -> bytes:
    """§8.25 + §11.8: YYMMDDHHMMSSZ — seconds always present, always UTC."""
    if not _UTCTIME_DER.match(value):
        raise Asn1Error(
            f"DER UTCTime must be YYMMDDHHMMSSZ with seconds present "
            f"(X.690 11.8.1-11.8.2), got {value!r}")
    return value.encode("ascii")


def decode_utctime(content: bytes, *, der: bool = False) -> str:
    try:
        value = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Asn1Error(f"UTCTime contents are not ASCII: {exc}") from exc
    if der and not _UTCTIME_DER.match(value):
        raise Asn1Error(
            f"DER UTCTime must be YYMMDDHHMMSSZ (X.690 11.8), got {value!r}")
    return value


def encode_generalizedtime(value: str) -> bytes:
    """§8.25 + §11.7: seconds present, "Z" terminated, no trailing fractional zeros."""
    if not _GENTIME_DER.match(value):
        raise Asn1Error(
            f"DER GeneralizedTime must be YYYYMMDDHHMMSS[.f]Z (X.690 11.7.1-11.7.2), "
            f"got {value!r}")
    if "." in value and (value.rstrip("Z").endswith("0") or value.endswith(".Z")):
        raise Asn1Error(
            f"DER GeneralizedTime omits trailing fractional zeros and the bare decimal "
            f"point (X.690 11.7.3), got {value!r}")
    return value.encode("ascii")


def decode_generalizedtime(content: bytes, *, der: bool = False) -> str:
    try:
        value = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Asn1Error(f"GeneralizedTime contents are not ASCII: {exc}") from exc
    if der:
        if not _GENTIME_DER.match(value):
            raise Asn1Error(
                f"DER GeneralizedTime must be YYYYMMDDHHMMSS[.f]Z (X.690 11.7), "
                f"got {value!r}")
        if "." in value and value.rstrip("Z").endswith("0"):
            raise Asn1Error(
                f"DER GeneralizedTime omits trailing fractional zeros (X.690 11.7.3), "
                f"got {value!r}")
    return value


# --- 8.26 the TIME type and the useful time types ---------------------------------

#: §8.26.2.2/§8.26.3.2/§8.26.4.2/§8.26.5.2: each type strips a fixed set of characters
#: from the value notation before encoding, so the wire form is a compressed spelling
#: of the notation rather than a distinct grammar.
_TIME_STRIP: dict[int, str] = {
    Universal.TIME: "",
    Universal.DATE: "-",
    Universal.TIME_OF_DAY: ":",
    Universal.DATE_TIME: "-:T",
    Universal.DURATION: "P",
}


def encode_time(tag_number: int, notation: str) -> bytes:
    """§8.26: UTF-8 of the value notation with this type's characters removed."""
    if tag_number not in _TIME_STRIP:
        raise Asn1Error(f"UNIVERSAL {tag_number} is not a TIME-family type")
    text = notation.strip('"')
    for char in _TIME_STRIP[tag_number]:
        text = text.replace(char, "")
    return text.encode("utf-8")


def decode_time(tag_number: int, content: bytes) -> str:
    """§8.26: the stripped notation, returned verbatim.

    Re-inserting the removed separators needs the type's SETTINGS from X.680 §38, which
    is a schema-level concern; the codec hands back exactly what was transmitted so a
    schema layer can rebuild the notation without the codec guessing.
    """
    if tag_number not in _TIME_STRIP:
        raise Asn1Error(f"UNIVERSAL {tag_number} is not a TIME-family type")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Asn1Error(f"TIME contents are not valid UTF-8: {exc}") from exc


__all__ = [
    "BitString", "REAL_MINUS_INFINITY", "REAL_MINUS_ZERO", "REAL_NOT_A_NUMBER",
    "REAL_PLUS_INFINITY", "decode_bitstring", "decode_boolean", "decode_enumerated",
    "decode_generalizedtime", "decode_integer", "decode_null", "decode_octetstring",
    "decode_oid", "decode_oid_iri", "decode_real", "decode_relative_oid",
    "decode_relative_oid_iri", "decode_string", "decode_time", "decode_utctime",
    "encode_bitstring", "encode_boolean", "encode_enumerated",
    "encode_generalizedtime", "encode_integer", "encode_oid", "encode_oid_iri",
    "encode_real", "encode_real_decimal", "encode_relative_oid",
    "encode_relative_oid_iri", "encode_string", "encode_time", "encode_utctime",
    "is_ascii_digits", "is_number_form",
]
