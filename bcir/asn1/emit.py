"""E1 — the format-neutral value stream, and one plan-driven emitter per candidate.

`encode_plan` compiles the schema; this consumes it. `flatten` turns a Python value into the
neutral stream described in that module's docstring, and `emit` turns plan + stream into one
candidate's octets. Every emitter reads the **same** stream, which is what makes their costs
comparable: hand DER its own octets and JER a Python object and the harness measures the
adapters rather than the encodings.

**Parity is against the oracle, per candidate**, and that is the only reason to trust any of
it. `emit(plan, flatten(plan, v), rules=DER)` must equal `encode_tlv(kind.encode(v))`,
`...JER` must equal `encode_jer(kind, v)`, and `...OER` must equal `encode_oer(kind, v)`.
These are not second implementations competing with the oracle — they are the *same* answer
reached through a descriptor, which is the property a C twin then has to reproduce.

**No emitter here re-derives a rule.** Where a clause governs a spelling the clause is named
at the site, because the whole value of a plan-driven encoder is that a reader can check it
against the standard without holding the oracle in their head at the same time.

**A finding the neutral stream exposed, which is an argument for having one.** The oracle's
three encoders disagree about how a Python value spells ASN.1 NULL: `codec` wants its `NULL`
sentinel and refuses `None`; `encode_jer` wants `None` and refuses `NULL`; `encode_oer`
accepts either. **There is no single Python value that can be handed to all three.** That
ambiguity lives in the *value mapping*, not in any encoding — and it is invisible until
something tries to drive every encoder from one input, which is exactly what a matched
comparison must do. The stream has nothing to disagree about, because a NULL contributes zero
octets to it, so all three plan-driven emitters produce the right answer from one value.

Nothing here changes the oracle: the disagreement is pinned by a test rather than papered
over, so whoever unifies the spelling does it deliberately and sees what depended on it.
"""

from __future__ import annotations

import enum

from .encode_plan import EncodeNode, EncodePlan
from .tags import Asn1Error, TagClass, Universal


class EmitRules(enum.Enum):
    """The candidates this harness can emit through a plan."""

    DER = "der"
    BER = "ber"
    JER = "jer"
    COER = "coer"


_TAG_CLASS_BITS = {
    "universal": 0x00, "application": 0x40, "context": 0x80, "private": 0xC0,
}


# --- the neutral value stream -------------------------------------------------------------------


class _Writer:
    def __init__(self) -> None:
        self.out = bytearray()

    def u8(self, value: int) -> None:
        self.out.append(value & 0xFF)

    def u32(self, value: int) -> None:
        self.out += value.to_bytes(4, "big")

    def blob(self, data: bytes) -> None:
        self.u32(len(data))
        self.out += data


class _Reader:
    """Bounds-checked, because the stream is attacker-shaped once a C twin reads it."""

    def __init__(self, data: bytes) -> None:
        self.data, self.at = data, 0

    def u8(self) -> int:
        if self.at + 1 > len(self.data):
            raise Asn1Error(f"value stream truncated at {self.at}: wanted 1 octet")
        self.at += 1
        return self.data[self.at - 1]

    def u32(self) -> int:
        if self.at + 4 > len(self.data):
            raise Asn1Error(f"value stream truncated at {self.at}: wanted 4 octets")
        self.at += 4
        return int.from_bytes(self.data[self.at - 4:self.at], "big")

    def take(self, count: int) -> bytes:
        if self.at + count > len(self.data):
            raise Asn1Error(f"value stream truncated at {self.at}: wanted {count} octets")
        self.at += count
        return self.data[self.at - count:self.at]

    def blob(self) -> bytes:
        return self.take(self.u32())


def _int_octets(value: int) -> bytes:
    """The abstract integer as X.690 §8.3.2 minimal two's complement — never empty."""
    length = max(1, (value.bit_length() + 8) // 8)
    return value.to_bytes(length, "big", signed=True)


def flatten(plan: EncodePlan, value) -> bytes:
    """Project a Python value into the neutral stream, against the plan."""
    writer = _Writer()
    _flatten_node(plan.root, value, writer, plan.type_name)
    return bytes(writer.out)


def _flatten_node(node: EncodeNode, value, writer: _Writer, path: str) -> None:
    kind = node.kind
    if kind == "boolean":
        writer.u8(1 if value else 0)
    elif kind in ("integer", "enumerated"):
        octets = _int_octets(int(value))
        writer.u8(len(octets))
        writer.out += octets
    elif kind == "null":
        return
    elif kind == "octetstring":
        writer.blob(bytes(value))
    elif kind == "string":
        writer.blob(str(value).encode("utf-8"))
    elif kind == "oid":
        writer.blob(".".join(str(arc) for arc in value).encode("ascii"))
    elif kind == "sequence":
        for member in node.members:
            present = member.name in value if isinstance(value, dict) else False
            if member.optional or member.has_default:
                writer.u8(1 if present else 0)
                if not present:
                    continue
            elif not present:
                raise Asn1Error(f"{path}/{member.name} is required and absent")
            _flatten_node(member.node, value[member.name], writer,
                          f"{path}/{member.name}")
    elif kind == "sequence-of":
        items = list(value)
        writer.u32(len(items))
        for index, item in enumerate(items):
            _flatten_node(node.element, item, writer, f"{path}[{index}]")
    elif kind == "choice":
        # The model spells a CHOICE value `(alternative_name, value)`: a bare value would be
        # ambiguous whenever two arms accept the same Python type.
        if not isinstance(value, tuple) or len(value) != 2:
            raise Asn1Error(
                f"{path}: a CHOICE value is an (alternative, value) pair, got {value!r}")
        name, inner = value
        chosen = [m for m in node.members if m.name == name]
        if not chosen:
            raise Asn1Error(f"{path}: {name!r} is not an alternative of this CHOICE")
        writer.u32(chosen[0].index)
        _flatten_node(chosen[0].node, inner, writer, f"{path}/{name}")
    else:
        raise Asn1Error(f"{path}: no flattening rule for plan kind {kind!r}")


# --- X.690: DER and BER ---------------------------------------------------------------------


def _identifier(tag_class: int, constructed: bool, number: int) -> bytes:
    if number < 31:
        return bytes((tag_class | (0x20 if constructed else 0) | number,))
    # X.690 §8.1.2.4: the high-tag-number form, base-128 with the continuation bit.
    body = [number & 0x7F]
    number >>= 7
    while number:
        body.append((number & 0x7F) | 0x80)
        number >>= 7
    return bytes((tag_class | (0x20 if constructed else 0) | 0x1F, *reversed(body)))


def _definite_length(count: int) -> bytes:
    """X.690 §8.1.3.3-§8.1.3.5, and §10.1 requires DER to use the *minimal* form."""
    if count < 0x80:
        return bytes((count,))
    body = count.to_bytes((count.bit_length() + 7) // 8, "big")
    return bytes((0x80 | len(body), *body))


def _tlv(tag_class: int, constructed: bool, number: int, content: bytes, *,
         indefinite: bool) -> bytes:
    head = _identifier(tag_class, constructed, number)
    if indefinite and constructed:
        # X.690 §8.1.3.6: BER may leave the length open and close with an EOC. DER may not
        # (§10.1), which is exactly the difference that makes BER a separate candidate here
        # rather than an alias for DER.
        return head + b"\x80" + content + b"\x00\x00"
    return head + _definite_length(len(content)) + content


def _oid_octets(text: str) -> bytes:
    arcs = [int(part) for part in text.split(".")]
    if len(arcs) < 2:
        raise Asn1Error("an OBJECT IDENTIFIER has at least two arcs (X.690 §8.19.3)")
    # §8.19.4: the first two arcs are combined into one subidentifier.
    body = bytearray()
    for value in [40 * arcs[0] + arcs[1], *arcs[2:]]:
        chunk = [value & 0x7F]
        value >>= 7
        while value:
            chunk.append((value & 0x7F) | 0x80)
            value >>= 7
        body += bytes(reversed(chunk))
    return bytes(body)


def _emit_x690(node: EncodeNode, reader: _Reader, *, indefinite: bool) -> bytes:
    """One node's complete TLV. Content is built first, so the length is always known."""
    kind = node.kind
    if kind == "boolean":
        # §11.1: DER's true is 0xFF exactly, not "any non-zero octet".
        return _tlv(0, False, int(Universal.BOOLEAN),
                    b"\xff" if reader.u8() else b"\x00", indefinite=False)
    if kind in ("integer", "enumerated"):
        universal = Universal.INTEGER if kind == "integer" else Universal.ENUMERATED
        return _tlv(0, False, int(universal), reader.take(reader.u8()), indefinite=False)
    if kind == "null":
        return _tlv(0, False, int(Universal.NULL), b"", indefinite=False)
    if kind == "octetstring":
        return _tlv(0, False, int(Universal.OCTET_STRING), reader.blob(), indefinite=False)
    if kind == "string":
        return _tlv(0, False, node.universal, reader.blob(), indefinite=False)
    if kind == "oid":
        return _tlv(0, False, int(Universal.OBJECT_IDENTIFIER),
                    _oid_octets(reader.blob().decode("ascii")), indefinite=False)
    if kind in ("sequence", "sequence-of"):
        content = bytearray()
        if kind == "sequence":
            for member in node.members:
                if (member.optional or member.has_default) and not reader.u8():
                    continue
                content += _member_x690(member, reader, indefinite=indefinite)
        else:
            for _ in range(reader.u32()):
                content += _emit_x690(node.element, reader, indefinite=indefinite)
        return _tlv(0, True, int(Universal.SEQUENCE), bytes(content), indefinite=indefinite)
    if kind == "choice":
        index = reader.u32()
        chosen = [m for m in node.members if m.index == index]
        if not chosen:
            raise Asn1Error(f"CHOICE index {index} is outside the plan's alternatives")
        return _member_x690(chosen[0], reader, indefinite=indefinite)
    raise Asn1Error(f"no X.690 rule for plan kind {kind!r}")


def _member_x690(member, reader: _Reader, *, indefinite: bool) -> bytes:
    inner = _emit_x690(member.node, reader, indefinite=indefinite)
    if member.tag is None:
        return inner
    bits = _TAG_CLASS_BITS[member.tag_class]
    if member.explicit:
        # §8.14.3: an explicit tag WRAPS the base encoding, so the outer is always
        # constructed whatever the inner was.
        return _tlv(bits, True, member.tag, inner, indefinite=indefinite)
    # §8.14.4: an implicit tag REPLACES the base tag and keeps its constructed bit — which
    # is why the inner encoding has to be taken apart rather than re-wrapped.
    constructed = bool(inner[0] & 0x20)
    body = inner[_identifier_length(inner):]
    return _tlv(bits, constructed, member.tag, _strip_length(body), indefinite=indefinite)


def _identifier_length(data: bytes) -> int:
    if data[0] & 0x1F != 0x1F:
        return 1
    at = 1
    while at < len(data) and data[at] & 0x80:
        at += 1
    return at + 1


def _strip_length(data: bytes) -> bytes:
    first = data[0]
    if first == 0x80:
        return data[1:-2]           # indefinite: drop the 0x80 and the closing EOC
    if first < 0x80:
        return data[1:1 + first]
    count = first & 0x7F
    length = int.from_bytes(data[1:1 + count], "big")
    return data[1 + count:1 + count + length]


# --- X.697: JER -----------------------------------------------------------------------------


def _json_string(text: str) -> str:
    import json
    return json.dumps(text, ensure_ascii=False)


def _emit_jer(node: EncodeNode, reader: _Reader) -> str:
    kind = node.kind
    if kind == "boolean":
        return "true" if reader.u8() else "false"
    if kind in ("integer", "enumerated"):
        return str(int.from_bytes(reader.take(reader.u8()), "big", signed=True))
    if kind == "null":
        return "null"
    if kind == "octetstring":
        # §21: an OCTET STRING is an upper-case hexadecimal string.
        return '"' + reader.blob().hex().upper() + '"'
    if kind == "string":
        return _json_string(reader.blob().decode("utf-8"))
    if kind == "oid":
        return '"' + reader.blob().decode("ascii") + '"'
    if kind == "sequence":
        parts = []
        for member in node.members:
            if (member.optional or member.has_default) and not reader.u8():
                continue
            # §22.2: the member's IDENTIFIER is what a JER document carries — the whole
            # reason this emitter cannot be schema-free.
            parts.append(f"{_json_string(member.name)}:{_emit_jer(member.node, reader)}")
        return "{" + ",".join(parts) + "}"
    if kind == "sequence-of":
        return "[" + ",".join(_emit_jer(node.element, reader)
                              for _ in range(reader.u32())) + "]"
    if kind == "choice":
        index = reader.u32()
        chosen = [m for m in node.members if m.index == index]
        if not chosen:
            raise Asn1Error(f"CHOICE index {index} is outside the plan's alternatives")
        return ("{" + _json_string(chosen[0].name) + ":"
                + _emit_jer(chosen[0].node, reader) + "}")
    raise Asn1Error(f"no JER rule for plan kind {kind!r}")


# --- X.696: OER -----------------------------------------------------------------------------


def _oer_length(count: int) -> bytes:
    """§8.6: short form below 128, else the long form with the octet count."""
    if count < 0x80:
        return bytes((count,))
    body = count.to_bytes((count.bit_length() + 7) // 8, "big")
    return bytes((0x80 | len(body), *body))


def _emit_oer(node: EncodeNode, reader: _Reader) -> bytes:
    kind = node.kind
    if kind == "boolean":
        # §11: any non-zero octet is TRUE, and CANONICAL-OER fixes it at 0xFF.
        return b"\xff" if reader.u8() else b"\x00"
    if kind in ("integer", "enumerated"):
        octets = reader.take(reader.u8())
        # §10.4: an unconstrained integer is length-prefixed two's complement.
        return _oer_length(len(octets)) + octets
    if kind == "null":
        return b""                                   # §12: no octets at all
    if kind == "octetstring":
        data = reader.blob()
        return _oer_length(len(data)) + data          # §15
    if kind == "string":
        data = reader.blob()
        return _oer_length(len(data)) + data          # §27
    if kind == "oid":
        body = _oid_octets(reader.blob().decode("ascii"))
        return _oer_length(len(body)) + body           # §14
    if kind == "sequence":
        # §16.2: a preamble of one bit per OPTIONAL/DEFAULT component, MSB first, padded
        # with zeroes to an octet boundary. §16.2.2 makes the padding zero in CANONICAL-OER.
        optional = [m for m in node.members if m.optional or m.has_default]
        bits: list[int] = []
        body = bytearray()
        present: dict[int, bool] = {}
        for member in node.members:
            if member.optional or member.has_default:
                here = bool(reader.u8())
                present[member.index] = here
                bits.append(1 if here else 0)
                if not here:
                    continue
            body += _emit_oer(member.node, reader)
        preamble = bytearray()
        if optional:
            padded = bits + [0] * (-len(bits) % 8)
            for start in range(0, len(padded), 8):
                octet = 0
                for bit in padded[start:start + 8]:
                    octet = (octet << 1) | bit
                preamble.append(octet)
        return bytes(preamble) + bytes(body)
    if kind == "sequence-of":
        count = reader.u32()
        # §19.1: the quantity is itself a length-prefixed unsigned integer, NOT a fixed
        # word. An empty SEQUENCE OF is `01 00` — one octet of count, whose value is zero —
        # and writing four zero octets instead produced a document a conforming decoder
        # reads as a three-element sequence.
        body = bytearray()
        for _ in range(count):
            body += _emit_oer(node.element, reader)
        quantity = count.to_bytes(max(1, (count.bit_length() + 7) // 8), "big")
        return _oer_length(len(quantity)) + quantity + bytes(body)
    if kind == "choice":
        index = reader.u32()
        chosen = [m for m in node.members if m.index == index]
        if not chosen:
            raise Asn1Error(f"CHOICE index {index} is outside the plan's alternatives")
        # §20.1: the alternative's TAG identifies it — a CHOICE is the one place OER puts
        # a tag on the wire (§8.7.1).
        tag = chosen[0].tag if chosen[0].tag is not None else chosen[0].index
        bits = _TAG_CLASS_BITS[chosen[0].tag_class]
        return bytes((bits | tag,)) + _emit_oer(chosen[0].node, reader)
    raise Asn1Error(f"no OER rule for plan kind {kind!r}")


# --- the one entry point ---------------------------------------------------------------------


def emit(plan: EncodePlan, stream: bytes, *, rules: EmitRules = EmitRules.DER) -> bytes:
    """Plan plus neutral stream, out one candidate's octets. The whole stream must be used.

    A leftover suffix means the stream and the plan disagree about the value's shape, and
    the encoding produced from a prefix would be a *valid* document of the wrong value —
    the failure mode worth refusing loudest.
    """
    reader = _Reader(stream)
    if rules in (EmitRules.DER, EmitRules.BER):
        out = _emit_x690(plan.root, reader, indefinite=rules is EmitRules.BER)
    elif rules is EmitRules.JER:
        out = _emit_jer(plan.root, reader).encode("utf-8")
    elif rules is EmitRules.COER:
        out = _emit_oer(plan.root, reader)
    else:
        raise Asn1Error(f"no emitter for {rules!r}")
    if reader.at != len(stream):
        raise Asn1Error(
            f"the value stream has {len(stream) - reader.at} octets left over after "
            f"{rules.value}; the plan and the stream describe different values")
    return out


__all__ = ["EmitRules", "emit", "flatten"]
