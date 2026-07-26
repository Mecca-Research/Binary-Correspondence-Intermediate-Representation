"""The TLV structure of an X.690 encoding — Rec. ITU-T X.690 (02/2021) §8.1.1.

Every encoding is identifier octets, length octets, contents octets, and — only when
the length was indefinite — end-of-contents octets. This module is the *structural*
layer: it walks that shape without interpreting any type's contents, which keeps the
trust-boundary reasoning in one place.

The decoder is deliberately total over arbitrary bytes: any input either yields a
`Tlv` tree or raises `Asn1Error`. Two bounds keep a hostile encoding from turning
recursion into a denial of service — a nesting depth cap and the length bound already
enforced in `length.py`. Neither is in X.690; both are decoder policy, and both are
reported as faults rather than silently truncating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .length import INDEFINITE, decode_length, encode_length
from .tags import EOC, Asn1Error, Tag, TagClass, decode_tag, encode_tag

#: Maximum constructed nesting a decoder will follow. X.690 places no limit; an
#: unbounded decoder recursing on attacker-chosen depth is a stack-exhaustion surface.
DEFAULT_MAX_DEPTH = 64


@dataclass
class Tlv:
    """One decoded encoding.

    A primitive encoding carries `content` (the contents octets verbatim); a
    constructed one carries `children`. `indefinite` records that the *sender* used
    the indefinite form — information DER needs in order to refuse it (§10.1) and
    that would be lost if decoding normalized the two forms together.
    """

    tag: Tag
    content: bytes = b""
    children: list["Tlv"] = field(default_factory=list)
    indefinite: bool = False
    #: The sender used more length octets than necessary (X.690 §8.1.3.5 NOTE 2).
    non_minimal_length: bool = False
    #: Offset of this encoding's first identifier octet in the buffer it came from.
    offset: int = 0

    @property
    def constructed(self) -> bool:
        return self.tag.constructed

    def flatten_content(self) -> bytes:
        """The concatenated contents of a constructed string encoding (§8.6.4/§8.7.3).

        BER lets a bitstring, octetstring or restricted character string be sent as a
        constructed series of fragments; the abstract value is their concatenation.
        Only meaningful for those types — the caller decides whether the segmentation
        was legal for the tag in hand.
        """
        if not self.constructed:
            return self.content
        return b"".join(child.flatten_content() for child in self.children)

    def __str__(self) -> str:
        kind = "constructed" if self.constructed else "primitive"
        if self.constructed:
            return f"{self.tag} ({kind}, {len(self.children)} child(ren))"
        return f"{self.tag} ({kind}, {len(self.content)} octet(s))"


def encode_tlv(tlv: Tlv) -> bytes:
    """Serialize a `Tlv` tree using definite, minimal-length octets throughout.

    The indefinite form is never emitted, even when `tlv.indefinite` is set on a tree
    that was decoded from BER: re-encoding is the point at which BER-in becomes
    DER-out, so the definite form is the only correct choice here.
    """
    if tlv.constructed:
        body = b"".join(encode_tlv(child) for child in tlv.children)
    else:
        body = tlv.content
    return encode_tag(tlv.tag) + encode_length(len(body)) + body


def _is_eoc(data: bytes, pos: int) -> bool:
    return pos + 1 < len(data) and data[pos] == 0x00 and data[pos + 1] == 0x00


def decode_tlv(data: bytes, pos: int = 0, *, max_depth: int = DEFAULT_MAX_DEPTH
               ) -> tuple[Tlv, int]:
    """Decode one encoding at `pos`; return (tlv, next position).

    Trailing bytes after the encoding are the caller's business — use `decode_one`
    when the buffer must contain exactly one encoding and nothing else.
    """
    return _decode(data, pos, max_depth, 0)


def _decode(data: bytes, pos: int, max_depth: int, depth: int) -> tuple[Tlv, int]:
    if depth > max_depth:
        raise Asn1Error(
            f"constructed nesting deeper than the decoder bound ({max_depth})", pos)
    start = pos
    tag, pos = decode_tag(data, pos)
    if tag.cls is TagClass.UNIVERSAL and tag.number == 0:
        raise Asn1Error(
            "unexpected end-of-contents octets: no indefinite-length encoding is open",
            start)
    length, pos = decode_length(data, pos)

    if length.indefinite:
        # §8.1.3.2 a): a primitive encoding must use the definite form.
        if not tag.constructed:
            raise Asn1Error(
                f"{tag} uses the indefinite length form but the encoding is primitive "
                f"(X.690 8.1.3.2 a)", start)
        children: list[Tlv] = []
        while True:
            if pos >= len(data):
                raise Asn1Error(
                    "indefinite-length encoding is not terminated by end-of-contents "
                    "octets", start)
            if _is_eoc(data, pos):
                pos += 2
                break
            child, pos = _decode(data, pos, max_depth, depth + 1)
            children.append(child)
        return Tlv(tag, b"", children, indefinite=True, offset=start), pos

    end = pos + (length.value or 0)
    if end > len(data):
        raise Asn1Error(
            f"{tag} declares {length.value} contents octets but only "
            f"{len(data) - pos} remain", start)

    if not tag.constructed:
        return Tlv(tag, data[pos:end], [], non_minimal_length=length.non_minimal,
                   offset=start), end

    children = []
    inner = pos
    while inner < end:
        child, inner = _decode(data, inner, max_depth, depth + 1)
        children.append(child)
    if inner != end:
        raise Asn1Error(
            f"{tag} children overrun the declared contents length", start)
    return Tlv(tag, b"", children, non_minimal_length=length.non_minimal,
               offset=start), end


def decode_one(data: bytes, *, max_depth: int = DEFAULT_MAX_DEPTH) -> Tlv:
    """Decode a buffer that must hold exactly one encoding.

    Trailing octets are a fault: X.690 §12.1 defines an encoding as a "self-delimiting
    octet string representation", so anything after the first complete encoding is
    either a second value the caller did not ask for or an attempt to smuggle bytes
    past a length-based check.
    """
    tlv, pos = decode_tlv(data, 0, max_depth=max_depth)
    if pos != len(data):
        raise Asn1Error(
            f"{len(data) - pos} trailing octet(s) after a complete encoding", pos)
    return tlv


def iter_tlv(data: bytes, *, max_depth: int = DEFAULT_MAX_DEPTH):
    """Yield each encoding in a buffer holding a concatenated series of them."""
    pos = 0
    while pos < len(data):
        tlv, pos = decode_tlv(data, pos, max_depth=max_depth)
        yield tlv


__all__ = ["DEFAULT_MAX_DEPTH", "EOC", "INDEFINITE", "Tlv", "decode_one", "decode_tlv",
           "encode_tlv", "iter_tlv"]
