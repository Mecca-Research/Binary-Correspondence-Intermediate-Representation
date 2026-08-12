"""DER conformance over a decoded encoding — X.690 (02/2021) clause 10 + clause 11.

BER is a family of encodings; DER is the single member of that family that makes a
value's octets unique. That uniqueness is why BCIR emits DER and nothing else: the
repo's artifacts are digested, replayed, and compared byte-for-byte, and an encoding
with sender's options cannot support any of that.

This module is the *checker*, kept separate from the encoder on purpose. The encoder
produces DER by construction, so it can never report a violation; the interesting
question is always about bytes that arrived from elsewhere. `der_violations` answers
it structurally — over a `Tlv` tree, without needing the schema — so the BER-in half
of the contract can accept an encoding, re-emit it as DER, and state precisely which
clauses the original broke.

Structural limits: clause 11.5 (no component encoded at its DEFAULT value) and the
DEFAULT/OPTIONAL parts of clause 8.9 need the ASN.1 type definition, so they are
enforced by the schema layer, not here. Everything else in clauses 10 and 11 that a
schema-free walk can see is checked.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tags import Asn1Error, TagClass, Universal
from .tlv import Tlv
from .values import decode_generalizedtime, decode_real, decode_utctime


@dataclass(frozen=True)
class Violation:
    """One DER conformance failure, with the clause that forbids it."""

    clause: str
    message: str
    offset: int

    def __str__(self) -> str:
        return f"X.690 {self.clause}: {self.message} (at octet {self.offset})"


#: Types whose constructed form clause 10.2 forbids outright.
_STRING_TAGS = frozenset({
    Universal.BIT_STRING, Universal.OCTET_STRING, Universal.UTF8_STRING,
    Universal.NUMERIC_STRING, Universal.PRINTABLE_STRING, Universal.TELETEX_STRING,
    Universal.VIDEOTEX_STRING, Universal.IA5_STRING, Universal.GRAPHIC_STRING,
    Universal.VISIBLE_STRING, Universal.GENERAL_STRING, Universal.UNIVERSAL_STRING,
    Universal.BMP_STRING, Universal.OBJECT_DESCRIPTOR,
})


def der_violations(tlv: Tlv) -> list[Violation]:
    """Every clause 10/11 violation in `tlv`, deepest-first within each subtree."""
    out: list[Violation] = []
    _walk(tlv, out)
    return out


def is_der(tlv: Tlv) -> bool:
    return not der_violations(tlv)


def require_der(tlv: Tlv) -> None:
    """Raise `Asn1Error` naming the first violation, or return cleanly."""
    bad = der_violations(tlv)
    if bad:
        raise Asn1Error(
            f"encoding is not DER: {bad[0]}"
            + (f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""), bad[0].offset)


def _walk(tlv: Tlv, out: list[Violation]) -> None:
    # 10.1 — the definite form, in the minimum number of octets.
    if tlv.indefinite:
        out.append(Violation(
            "10.1", f"{tlv.tag} uses the indefinite length form; DER requires the "
                    f"definite form", tlv.offset))
    if tlv.non_minimal_length:
        out.append(Violation(
            "10.1", f"{tlv.tag} length is not encoded in the minimum number of octets",
            tlv.offset))

    if tlv.tag.is_universal:
        _walk_universal(tlv, out)

    for child in tlv.children:
        _walk(child, out)


def _walk_universal(tlv: Tlv, out: list[Violation]) -> None:
    number = tlv.tag.number

    # 10.2 — no constructed strings.
    if tlv.constructed and number in _STRING_TAGS:
        out.append(Violation(
            "10.2", f"{tlv.tag} uses the constructed form; DER forbids it for "
                    f"bitstring, octetstring and restricted character string types",
            tlv.offset))

    # 11.1 — boolean TRUE is all ones.
    if number == Universal.BOOLEAN and not tlv.constructed:
        if len(tlv.content) == 1 and tlv.content[0] not in (0x00, 0xFF):
            out.append(Violation(
                "11.1", f"boolean TRUE is encoded as 0x{tlv.content[0]:02x}; DER "
                        f"requires all eight bits set", tlv.offset))

    # 11.2.1 — every unused bit in a bitstring's final octet is zero.
    if number == Universal.BIT_STRING and not tlv.constructed and tlv.content:
        unused = tlv.content[0]
        body = tlv.content[1:]
        if unused <= 7 and body and body[-1] & ((1 << unused) - 1):
            out.append(Violation(
                "11.2.1", "bitstring has non-zero unused bits in its final octet",
                tlv.offset))

    # 11.6 — SET OF components appear in ascending octet order, shorter encodings
    # padded with trailing zero octets for the comparison only.
    if number == Universal.SET and tlv.constructed and len(tlv.children) > 1:
        encoded = [_reencode(child) for child in tlv.children]
        width = max(len(e) for e in encoded)
        padded = [e.ljust(width, b"\x00") for e in encoded]
        if padded != sorted(padded):
            out.append(Violation(
                "11.6", "set-of components are not in ascending order", tlv.offset))

    # 11.7 / 11.8 — the canonical time spellings.
    if number == Universal.UTC_TIME and not tlv.constructed:
        try:
            decode_utctime(tlv.content, der=True)
        except Asn1Error as exc:
            out.append(Violation("11.8", str(exc), tlv.offset))
    if number == Universal.GENERALIZED_TIME and not tlv.constructed:
        try:
            decode_generalizedtime(tlv.content, der=True)
        except Asn1Error as exc:
            out.append(Violation("11.7", str(exc), tlv.offset))

    # 11.3 — the canonical REAL. Clause 8.5 grants a sender three bases, a scaling factor,
    # a choice of mantissa normalization and three ISO 6093 forms; 11.3 removes all of it.
    # Checked through the value decoder for the same reason the times are: the restrictions
    # are stated over the *decoded* fields, and re-implementing them here would give the
    # checker and the decoder two chances to disagree.
    if number == Universal.REAL and not tlv.constructed:
        try:
            decode_real(tlv.content, der=True)
        except Asn1Error as exc:
            out.append(Violation("11.3", str(exc), tlv.offset))


def _reencode(tlv: Tlv) -> bytes:
    from .tlv import encode_tlv
    return encode_tlv(tlv)


def to_der(tlv: Tlv) -> Tlv:
    """Rewrite a BER tree into its DER form (the BER-in / DER-out conversion).

    Applies the transformations that do not need a schema: collapse constructed
    strings into a single primitive encoding (10.2), normalize boolean TRUE (11.1),
    zero a bitstring's unused bits (11.2.1), and sort SET OF components (11.6). The
    definite minimal length (10.1) falls out of re-encoding. Clause 11.5 is left to
    the schema layer, which is the only layer that knows a component's DEFAULT.
    """
    if tlv.tag.is_universal and tlv.tag.number in _STRING_TAGS and tlv.constructed:
        # A constructed BIT STRING is NOT a raw octet concatenation: §8.6.2.2 gives
        # every segment its own leading unused-bit octet, so the segments join at the
        # bit level and the joined value takes the last segment's unused count.
        # Concatenating the contents octets verbatim would splice those count octets
        # into the value — the failure X.690 §8.6.4.2's own example exposes.
        if tlv.tag.number == Universal.BIT_STRING:
            from .values import decode_bitstring, encode_bitstring
            return Tlv(tlv.tag.as_primitive(),
                       encode_bitstring(decode_bitstring(tlv)), [], offset=tlv.offset)
        return Tlv(tlv.tag.as_primitive(), tlv.flatten_content(), [],
                   offset=tlv.offset)

    if not tlv.constructed:
        content = tlv.content
        if tlv.tag.is_universal:
            if tlv.tag.number == Universal.BOOLEAN and len(content) == 1 and content[0]:
                content = b"\xff"
            elif tlv.tag.number == Universal.BIT_STRING and content:
                unused = content[0]
                if unused <= 7 and len(content) > 1:
                    body = bytearray(content[1:])
                    body[-1] &= (0xFF << unused) & 0xFF
                    content = bytes([unused]) + bytes(body)
        return Tlv(tlv.tag, content, [], offset=tlv.offset)

    children = [to_der(child) for child in tlv.children]
    if tlv.tag.is_universal and tlv.tag.number == Universal.SET:
        from .tlv import encode_tlv
        width = max((len(encode_tlv(c)) for c in children), default=0)
        children = sorted(children, key=lambda c: encode_tlv(c).ljust(width, b"\x00"))
    return Tlv(tlv.tag, b"", children, offset=tlv.offset)


__all__ = ["Violation", "der_violations", "is_der", "require_der", "to_der"]
