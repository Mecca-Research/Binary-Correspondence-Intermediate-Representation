"""The ITS study's size accounting: every octet of every envelope, by encoding and by purpose.

`its_security` gives one abstract value per Table VI envelope and `its_native` times codecs
for it; this module answers WHERE the octets go. Every encoding here is computed by a BCIR
rail from the same value, so the table is reproducible from the repository alone:

* `sizes()` -- octets per envelope for the V1.1.1 byte rule and for UPER, APER, COER and DER
  over both modules (the transcription, and the TMSAO schema), beside the paper's columns;
* `average_sizes()` -- the paper's Table VII metric (a CAM stream at 10 Hz carrying its
  certificate once a second), for every one of those encodings;
* `framing()` -- an exact partition of one envelope's bits into CONTENT (the fields' own
  values) and FRAMING (selectors, lengths and counts, presence and extension bits, padding),
  computed by encoding each subtree on its own with the rule's own encoder and attributing
  the difference between a node and its children to the node's kind.

The partition is what turns "UPER is smaller" into an explanation: the content is the same
octets in every column -- keys, digests, signatures, times -- and the encodings differ ONLY in
framing. V1.1.1 frames with whole octets (a type code before every field, an IntX before every
list); PER frames with bits, and on the TMSAO schema most of its framing is the presence
bitmap that replaced those type codes.
"""

from __future__ import annotations

from collections import Counter

from . import its_security as its
from .tags import Asn1Error

#: The paper's Table VII metric: CAMs at the maximum rate (10 Hz) carrying the certificate at
#: the minimum inclusion rate (1 Hz), so nine envelopes in ten are profile 1 without it.
F_CAM = 10
F_CERT = 1

#: Every encoding `sizes()` measures: (name, module, rule).
ENCODINGS = (
    ("binary-v111", "transcription", "binary"),
    ("transcription-uper", "transcription", "uper"),
    ("transcription-aper", "transcription", "aper"),
    ("transcription-coer", "transcription", "coer"),
    ("transcription-der", "transcription", "der"),
    ("tmsao-uper", "tmsao", "uper"),
    ("tmsao-aper", "tmsao", "aper"),
    ("tmsao-coer", "tmsao", "coer"),
    ("tmsao-der", "tmsao", "der"),
    ("tmsao-per-uper", "tmsao-per", "uper"),
    ("tmsao-per-coer", "tmsao-per", "coer"),
)


def _value(module: str, profile: str):
    value = its.envelope(profile)
    return its.to_tmsao(value) if module.startswith("tmsao") else value


def _type(module: str, name: str = "SecuredMessage"):
    """`tmsao-per` is the TMSAO module's PER-maximal instance (`SecuredMessagePer`): the same
    values, with the version and the profile shaped as well."""
    if module == "tmsao-per":
        return its.tmsao_type("SecuredMessagePer" if name == "SecuredMessage" else name)
    return its.tmsao_type(name) if module == "tmsao" else its.transcription_type(name)


def encode(module: str, rule: str, value, type_name: str = "SecuredMessage") -> bytes:
    """`value` of `module`'s `type_name` under `rule` (binary, uper, aper, coer, der)."""
    from .codec import encode_tlv
    from .oer import encode_oer
    from .per import PerVariant, encode_per

    kind = _type(module, type_name)
    if rule == "binary":
        if module != "transcription":
            raise Asn1Error("the V1.1.1 byte rule encodes the transcription module")
        return its.encode_binary(value, type_name)
    if rule == "uper":
        return encode_per(kind, value)
    if rule == "aper":
        return encode_per(kind, value, variant=PerVariant.ALIGNED)
    if rule == "coer":
        return encode_oer(kind, value)
    if rule == "der":
        return encode_tlv(kind.encode(value))
    raise Asn1Error(f"unknown rule {rule!r}")


def sizes() -> dict:
    """`{encoding name: {profile: octets}}` for every encoding in `ENCODINGS`."""
    return {
        name: {p: len(encode(module, rule, _value(module, p))) for p in its.PROFILES}
        for name, module, rule in ENCODINGS
    }


def average_size(row: dict) -> float:
    """Table VII: `((F_CAM - F_CERT) * s_without + F_CERT * s_with) / F_CAM`, in octets."""
    return ((F_CAM - F_CERT) * row["p1"] + F_CERT * row["p1cert"]) / F_CAM


def average_sizes() -> dict:
    out = {name: average_size(row) for name, row in sizes().items()}
    for name, row in its.PAPER_TABLE_VI.items():
        out[f"paper-{name}"] = average_size(row)
    return out


# --- the bit partition --------------------------------------------------------------------------

#: The framing categories `framing()` reports, beside "content".
CATEGORIES = ("content", "selector", "length", "presence", "padding")


def _subtree_bits(rule: str, kind, value) -> int:
    """The bits `rule` spends on `value` of `kind` encoded on its own, mid-stream -- for
    PER without §11.1's octet padding, since a subtree inside a value is not padded."""
    from .oer import encode_value
    from .per import BitWriter, PerRules, PerVariant
    from .per import _encode as per_encode

    if rule == "binary":
        out = bytearray()
        its._encode(kind, value, out)
        return 8 * len(out)
    if rule == "coer":
        return 8 * len(encode_value(kind, value))
    if rule == "uper":
        writer = BitWriter(PerVariant.UNALIGNED)
        per_encode(writer, kind, value, PerRules.CANONICAL)
        return len(writer)
    raise Asn1Error(f"framing() partitions binary, coer and uper; not {rule!r}")


def _leaf_parts(rule: str, kind, value, total: int) -> tuple[int, int, int]:
    """A primitive's bits as (content, extension bits, length bits).

    A fixed-width field is all value. A variable one is its value's octets (or bits); an IntX
    prefix or a length octet is a length, and PER's extension bit on an extensible INTEGER or
    ENUMERATED is an extension bit -- framing, filed with the presence bits.
    """
    from .oer import _fixed_size, _integer_form
    from .per import _size_bounds, _value_bounds
    from .tags import Universal

    u = kind.universal
    if u == Universal.NULL:
        return 0, 0, 0
    if u == Universal.BOOLEAN:
        return total, 0, 0
    if u == Universal.ENUMERATED:
        ext = 1 if rule == "uper" and getattr(kind, "enum_extensible", False) else 0
        return total - ext, ext, 0
    if u == Universal.OCTET_STRING:
        fixed = (
            (rule == "coer" and _fixed_size(kind) is not None)
            or (rule == "binary" and its._fixed_size(kind) is not None)
            or (rule == "uper" and _size_bounds(kind)[0] == _size_bounds(kind)[1])
        )
        if fixed:
            return total, 0, 0
        return 8 * len(value), 0, total - 8 * len(value)
    if u == Universal.INTEGER:
        if rule == "binary":
            if its._int_width(kind) is not None:
                return total, 0, 0
            octets = len(its.encode_intx(value))  # n leading ones and a zero: n + 1 bits
            return total - octets, 0, octets
        if rule == "coer":
            width, _signed = _integer_form(kind)
            return (total, 0, 0) if width is not None else (total - 8, 0, 8)
        (low, high), ext = _value_bounds(kind)
        if ext:
            inside = low <= value <= high
            return (total - 1, 1, 0) if inside else (total - 9, 1, 8)
        if low is not None and high is not None:
            return total, 0, 0
        return total - 8, 0, 8
    raise Asn1Error(f"framing(): no content rule for {Universal(u).name}")


def _walk(rule: str, kind, value, acc: Counter) -> int:
    from .schema import Choice, Primitive, Sequence, SequenceOf, resolve

    kind = resolve(kind)
    total = _subtree_bits(rule, kind, value)
    if isinstance(kind, Primitive):
        content, extension, length = _leaf_parts(rule, kind, value, total)
        acc["content"] += content
        acc["presence"] += extension
        acc["length"] += length
        return total
    if isinstance(kind, Sequence):
        inner = sum(
            _walk(rule, comp.type, value[comp.name], acc)
            for comp in kind.components
            if comp.name in value
        )
        acc["presence"] += total - inner  # preamble: presence and extension bits
        return total
    if isinstance(kind, SequenceOf):
        inner = sum(_walk(rule, kind.element, item, acc) for item in value)
        acc["length"] += total - inner  # the count, or the octet length, of the list
        return total
    if isinstance(kind, Choice):
        name, inner_value = value
        alt = next(a for a in kind.alternatives if a.name == name)
        inner = _walk(rule, alt.type, inner_value, acc)
        acc["selector"] += total - inner  # a type code, a tag, or an index
        return total
    raise Asn1Error(f"framing(): {type(kind).__name__} is not walked")


def framing(codec: str, profile: str) -> dict:
    """The envelope's bits under `codec` (`binary-v111`, `transcription-uper`,
    `transcription-coer`, `tmsao-uper`, `tmsao-coer`), partitioned into `CATEGORIES`. The
    parts sum to eight times the encoding's length exactly -- `test_its_security` holds it."""
    table = {name: (module, rule) for name, module, rule in ENCODINGS}
    module, rule = table[codec]
    value = _value(module, profile)
    acc: Counter = Counter()
    used = _walk(rule, _type(module), value, acc)
    octets = len(encode(module, rule, value))
    acc["padding"] += 8 * octets - used
    return {name: acc[name] for name in CATEGORIES}


def content_floor(profile: str) -> int:
    """The envelope's content in octets (rounded up): the fields' own values, which every
    encoding of this value carries -- every octet above it, in any column, is framing."""
    return -(-framing("binary-v111", profile)["content"] // 8)


__all__ = [
    "CATEGORIES",
    "ENCODINGS",
    "F_CAM",
    "F_CERT",
    "average_size",
    "average_sizes",
    "content_floor",
    "encode",
    "framing",
    "sizes",
]
