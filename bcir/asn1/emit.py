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
    """The candidates this harness can emit through a plan.

    PER is FOUR rows, not one. X.691's ALIGNED/UNALIGNED split is a genuine cost trade
    rather than a setting — ALIGNED pads so multi-octet fields start on octet boundaries
    (cheaper to read, larger on the wire), UNALIGNED never pads — and the CANONICAL/BASIC
    split decides §19.5's DEFAULT rule. Collapsing them would put one number in the table
    for two different encodings.
    """

    DER = "der"
    BER = "ber"
    JER = "jer"
    COER = "coer"
    CANONICAL_PER_ALIGNED = "cper-a"
    CANONICAL_PER_UNALIGNED = "cper-u"
    BASIC_PER_ALIGNED = "bper-a"
    BASIC_PER_UNALIGNED = "bper-u"


_TAG_CLASS_BITS = {
    "universal": 0x00,
    "application": 0x40,
    "context": 0x80,
    "private": 0xC0,
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
        return int.from_bytes(self.data[self.at - 4 : self.at], "big")

    def take(self, count: int) -> bytes:
        if self.at + count > len(self.data):
            raise Asn1Error(f"value stream truncated at {self.at}: wanted {count} octets")
        self.at += count
        return self.data[self.at - count : self.at]

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


def flatten_value(node: EncodeNode, value, path: str = "") -> bytes:
    """Project one value against ONE node, rather than against a whole plan.

    The plan compiler uses this to record a DEFAULT in stream form. Exposed here rather than
    reimplemented there because the recorded default is compared against a slice of a real
    stream, and two spellings of "what a value looks like" would eventually disagree.
    """
    writer = _Writer()
    _flatten_node(node, value, writer, path or node.type_name)
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
            _flatten_node(member.node, value[member.name], writer, f"{path}/{member.name}")
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
                f"{path}: a CHOICE value is an (alternative, value) pair, got {value!r}"
            )
        name, inner = value
        chosen = [m for m in node.members if m.name == name]
        if not chosen:
            raise Asn1Error(f"{path}: {name!r} is not an alternative of this CHOICE")
        writer.u32(chosen[0].index)
        _flatten_node(chosen[0].node, inner, writer, f"{path}/{name}")
    else:
        raise Asn1Error(f"{path}: no flattening rule for plan kind {kind!r}")


# --- the DEFAULT omission rule, which is per-candidate ------------------------------------------


def _skip_node(node: EncodeNode, reader: _Reader) -> None:
    """Advance the reader past exactly one node's value, emitting nothing.

    The mirror of `_flatten_node`, and the only way to ask "what octets does this member
    occupy" without producing them. A wrong answer here does not corrupt an encoding
    quietly: it desynchronizes the reader, and `emit` refuses a stream with anything left
    over.
    """
    kind = node.kind
    if kind == "boolean":
        reader.u8()
    elif kind in ("integer", "enumerated"):
        reader.take(reader.u8())
    elif kind == "null":
        return
    elif kind in ("octetstring", "string", "oid"):
        reader.blob()
    elif kind == "sequence":
        for member in node.members:
            if (member.optional or member.has_default) and not reader.u8():
                continue
            _skip_node(member.node, reader)
    elif kind == "sequence-of":
        for _ in range(reader.u32()):
            _skip_node(node.element, reader)
    elif kind == "choice":
        index = reader.u32()
        chosen = [m for m in node.members if m.index == index]
        if not chosen:
            raise Asn1Error(f"CHOICE index {index} is outside the plan's alternatives")
        _skip_node(chosen[0].node, reader)
    else:
        raise Asn1Error(f"no skip rule for plan kind {kind!r}")


def _present(member, reader: _Reader, *, omit_default: bool) -> bool:
    """Read a component's presence octet and apply the candidate's DEFAULT rule.

    X.690 §11.5 forbids DER an encoding for a component whose value equals its default;
    X.696 §31.9 and X.697's CJER say the same, and X.691 §19.5 says it for CANONICAL-PER.
    **Plain BER and BASIC-PER do not** — they leave it to the sender, which is precisely the
    implementation freedom the canonical rules exist to remove. So this is not a property of
    the value and cannot live in the format-neutral stream; every candidate asks it
    separately, and `omit_default` is which answer this one gives.

    When the rule applies and the value matches, the component's octets are *consumed and
    discarded*: the stream still describes it, and leaving it unread would leave a suffix
    `emit` correctly refuses.
    """
    if not (member.optional or member.has_default):
        return True
    if not reader.u8():
        return False
    if not (omit_default and member.has_default and member.default_stream):
        return True
    start = reader.at
    _skip_node(member.node, reader)
    if reader.data[start : reader.at] == member.default_stream:
        return False
    reader.at = start
    return True


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


def _tlv(
    tag_class: int, constructed: bool, number: int, content: bytes, *, indefinite: bool
) -> bytes:
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
        return _tlv(
            0, False, int(Universal.BOOLEAN), b"\xff" if reader.u8() else b"\x00", indefinite=False
        )
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
        return _tlv(
            0,
            False,
            int(Universal.OBJECT_IDENTIFIER),
            _oid_octets(reader.blob().decode("ascii")),
            indefinite=False,
        )
    if kind in ("sequence", "sequence-of"):
        content = bytearray()
        if kind == "sequence":
            for member in node.members:
                # X.690 §11 is titled "Restrictions on BER employed by both CER and DER", so
                # §11.5's ban on encoding a component equal to its default binds DER and
                # leaves BER free. That is the same seam §8.1.3.6 opens for the length form,
                # and it is why the two candidates are not aliases.
                if not _present(member, reader, omit_default=not indefinite):
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
    body = inner[_identifier_length(inner) :]
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
        return data[1:-2]  # indefinite: drop the 0x80 and the closing EOC
    if first < 0x80:
        return data[1 : 1 + first]
    count = first & 0x7F
    length = int.from_bytes(data[1 : 1 + count], "big")
    return data[1 + count : 1 + count + length]


# --- X.697: JER -----------------------------------------------------------------------------


def _json_string(text: str) -> str:
    import json

    return json.dumps(text, ensure_ascii=False)


def _emit_jer(node: EncodeNode, reader: _Reader) -> str:
    kind = node.kind
    if kind == "boolean":
        return "true" if reader.u8() else "false"
    if kind == "integer":
        return str(int.from_bytes(reader.take(reader.u8()), "big", signed=True))
    if kind == "enumerated":
        # §22.2: the JSON string denotes "the identifier of the chosen enumeration item".
        # NOT the number — X.690 §8.4 encodes the number and X.697 deliberately does not,
        # which is why the plan has to carry the identifiers at all. Sharing this branch
        # with `integer` emitted `4` where the standard requires `"red"`: a document a JER
        # decoder cannot map back to any enumeration item.
        value = int.from_bytes(reader.take(reader.u8()), "big", signed=True)
        for name, number in node.enumeration:
            if number == value:
                return _json_string(name)
        raise Asn1Error(
            f"{value} names no item of this enumeration, and X.697 §22.1 gives an "
            f"enumerated value no numeric spelling"
        )
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
            # This row is CJER: `encode_jer` defaults to CANONICAL-JER, which omits a
            # component equal to its default. BASIC-JER emits it, and the two really do
            # differ -- the oracle produces `{"a":1,"b":false}` under BASIC.
            if not _present(member, reader, omit_default=True):
                continue
            # §22.2: the member's IDENTIFIER is what a JER document carries — the whole
            # reason this emitter cannot be schema-free.
            parts.append(f"{_json_string(member.name)}:{_emit_jer(member.node, reader)}")
        return "{" + ",".join(parts) + "}"
    if kind == "sequence-of":
        return "[" + ",".join(_emit_jer(node.element, reader) for _ in range(reader.u32())) + "]"
    if kind == "choice":
        index = reader.u32()
        chosen = [m for m in node.members if m.index == index]
        if not chosen:
            raise Asn1Error(f"CHOICE index {index} is outside the plan's alternatives")
        return "{" + _json_string(chosen[0].name) + ":" + _emit_jer(chosen[0].node, reader) + "}"
    raise Asn1Error(f"no JER rule for plan kind {kind!r}")


# --- X.696: OER -----------------------------------------------------------------------------


def _oer_length(count: int) -> bytes:
    """§8.6: short form below 128, else the long form with the octet count."""
    if count < 0x80:
        return bytes((count,))
    body = count.to_bytes((count.bit_length() + 7) // 8, "big")
    return bytes((0x80 | len(body), *body))


#: X.696 §27.2's known-multiplier character types, restricted to the ones this plan compiles
#: as `string`. BMPString and UniversalString are known-multiplier too, but `_LEAF_KIND` has
#: no rule for them, so listing them here would describe a case that cannot arise.
_OER_KNOWN_MULTIPLIER = frozenset(
    {
        int(Universal.NUMERIC_STRING),
        int(Universal.PRINTABLE_STRING),
        int(Universal.VISIBLE_STRING),
        int(Universal.IA5_STRING),
    }
)


def _oer_integer_form(node: EncodeNode) -> tuple[int | None, bool]:
    """§10.3 / §10.4: the fixed word width the constraint selects, and its sign.

    `(None, signed)` is §10.3 e) / §10.4 e), the length-prefixed variable-size form. The
    split between the two clauses is whether a lower bound EXISTS and is non-negative, not
    whether the bounds happen to be small — so a type with no lower bound is signed however
    tight its upper bound is. This mirrors `oer._integer_form` clause for clause, reading the
    plan's recorded bounds where the oracle reads the live constraint.
    """
    constraint = node.constraint
    low = None if constraint is None else constraint.value_low
    high = None if constraint is None else constraint.value_high
    if low is not None and low >= 0:  # §10.2 a) -> §10.3, unsigned
        if high is None:
            return None, False
        for width, limit in ((1, 0xFF), (2, 0xFFFF), (4, 0xFFFFFFFF), (8, 0xFFFFFFFFFFFFFFFF)):
            if high <= limit:
                return width, False
        return None, False  # §10.3 e)
    if low is None or high is None:  # §10.2 b) -> §10.4, signed
        return None, True
    for width in (1, 2, 4, 8):
        bits = width * 8
        if -(1 << (bits - 1)) <= low and high <= (1 << (bits - 1)) - 1:
            return width, True
    return None, True  # §10.4 e)


def _oer_fixed_size(node: EncodeNode) -> int | None:
    """The single length a SIZE constraint fixes, or None when it does not fix one.

    §14.1 and §27.2 turn on the same condition — the effective size constraint's bounds
    being *identical*. A range is not enough: only an exact length lets a decoder find the
    end of the field without a determinant.
    """
    constraint = node.constraint
    if constraint is None or constraint.size_low is None:
        return None
    return constraint.size_low if constraint.size_low == constraint.size_high else None


def _emit_oer(node: EncodeNode, reader: _Reader) -> bytes:
    kind = node.kind
    if kind == "boolean":
        # §11: any non-zero octet is TRUE, and CANONICAL-OER fixes it at 0xFF.
        return b"\xff" if reader.u8() else b"\x00"
    if kind == "integer":
        octets = reader.take(reader.u8())
        width, signed = _oer_integer_form(node)
        if width is None:
            # §10.3 e) / §10.4 e): length-prefixed. The stream carries the minimal *signed*
            # form, which is already right for §10.4; §10.3's unsigned form drops the sign
            # octet a non-negative value may carry.
            value = int.from_bytes(octets, "big", signed=True)
            if not signed:
                octets = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
            return _oer_length(len(octets)) + octets
        value = int.from_bytes(octets, "big", signed=True)
        try:
            return value.to_bytes(width, "big", signed=signed)
        except OverflowError:
            raise Asn1Error(
                f"{value} does not fit the {width}-octet {'signed' if signed else 'unsigned'}"
                f" word its constraint selected (X.696 §10.3/§10.4)"
            ) from None
    if kind == "enumerated":
        # §11 gives ENUMERATED its own form and reads no constraint: the short form below
        # 128, else a long form whose count octet has the top bit set. It is NOT §10's
        # integer, which is why this no longer shares a branch with it.
        value = int.from_bytes(reader.take(reader.u8()), "big", signed=True)
        if 0 <= value < 0x80:
            return bytes((value,))  # §11.3
        octets = _int_octets(value)  # §11.4 long form: SIGNED
        if len(octets) > 0x7F:
            raise Asn1Error("enumerated value needs more than 127 octets (X.696 §11.4)")
        return bytes((0x80 | len(octets),)) + octets
    if kind == "null":
        return b""  # §12: no octets at all
    if kind == "octetstring":
        data = reader.blob()
        fixed = _oer_fixed_size(node)
        if fixed is not None:  # §14.1: no length determinant
            if len(data) != fixed:
                raise Asn1Error(
                    f"SIZE ({fixed}) requires exactly {fixed} octets, got {len(data)} (X.696 §14.1)"
                )
            return data
        return _oer_length(len(data)) + data  # §14.2
    if kind == "string":
        data = reader.blob()
        # §27.2 drops the length determinant only for a KNOWN-MULTIPLIER type whose
        # effective size constraint is a single value: only then does the character count
        # fix the octet count. UTF8String never qualifies (§27.1) — a character costs 1..4
        # octets there, so its length is never implied.
        fixed = _oer_fixed_size(node)
        if fixed is not None and node.universal in _OER_KNOWN_MULTIPLIER:
            # Every repertoire in `_OER_KNOWN_MULTIPLIER` is single-octet, so the stream's
            # UTF-8 octet count IS the character count. A wider known-multiplier type would
            # need the character count instead, and none reaches here.
            if len(data) != fixed:
                raise Asn1Error(
                    f"SIZE ({fixed}) requires exactly {fixed} characters, got {len(data)} "
                    f"(X.696 §27.2)"
                )
            return data
        return _oer_length(len(data)) + data  # §27.3
    if kind == "oid":
        body = _oid_octets(reader.blob().decode("ascii"))
        return _oer_length(len(body)) + body  # §14
    if kind == "sequence":
        # §16.2: a preamble of one bit per OPTIONAL/DEFAULT component, MSB first, padded
        # with zeroes to an octet boundary. §16.2.2 makes the padding zero in CANONICAL-OER.
        optional = [m for m in node.members if m.optional or m.has_default]
        bits: list[int] = []
        body = bytearray()
        present: dict[int, bool] = {}
        for member in node.members:
            if member.optional or member.has_default:
                # §31.9: a DEFAULT component is encoded as ABSENT when it equals its
                # default, and the preamble bit must say so too -- writing a 1 and then no
                # field would misalign every component after it.
                here = _present(member, reader, omit_default=True)
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
                for bit in padded[start : start + 8]:
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


# --- X.691: PER, ALIGNED and UNALIGNED ---------------------------------------------------------
#
# **The bit arithmetic is imported, not retyped.** §11's field encoders — a constrained whole
# number's five cases, the semi-constrained and unconstrained forms, the one/two-octet and
# fragmented length determinants — read no schema at all: they take an integer and a pair of
# bounds. Writing them a second time here would create a second definition of §11 free to
# drift from the one the oracle uses, and the parity test would then be comparing two
# implementations of the same misunderstanding.
#
# What E1 must supply independently is the half that *is* schema-directed: reading bounds,
# extension markers, enumerations and alternative order out of a DESCRIPTOR rather than out of
# a live `Asn1Type`. That is the whole claim being tested, and it is what the C twin then has
# to reproduce with no Python in reach.


#: Canonical tag order for X.691 §23.2, as the plan spells a class.
_TAG_CLASS_ORDER = {"universal": 0, "application": 1, "context": 2, "private": 3}


def _per_effective_tag(member) -> tuple[int, int]:
    """§23.2's sort key for one CHOICE alternative: its outer tag, class first.

    A tagged alternative uses its own tag. An UNTAGGED one uses the tag of its type, and
    §23.3 gives an untagged nested CHOICE the smallest tag among *its* alternatives — which
    is why this recurses rather than reading one field.
    """
    if member.tag is not None:
        return (_TAG_CLASS_ORDER[member.tag_class], member.tag)
    return _per_base_tag(member.node)


def _per_base_tag(node: EncodeNode) -> tuple[int, int]:
    if node.kind == "choice":
        return min(_per_effective_tag(m) for m in node.members)
    # Everything else the plan compiles carries a universal tag, and X.690 Table 1 puts the
    # UNIVERSAL class first, which is why it sorts as 0.
    return (0, node.universal)


def _per_alternatives(node: EncodeNode) -> tuple:
    """The alternatives in §23.2's canonical order — which is what assigns the index."""
    return tuple(sorted(node.members, key=_per_effective_tag))


def _per_value_bounds(node: EncodeNode):
    """§13.1: the extension ROOT's value bounds, and whether an extension bit is emitted."""
    constraint = node.constraint
    if constraint is None:
        return (None, None), False
    return (constraint.root_value_low, constraint.root_value_high), constraint.value_extensible


def _per_size_bounds(node: EncodeNode) -> tuple[int, int | None, bool]:
    """§17.3/§20.4/§30.4's `(lb, ub, extensible)`, in §11.9's shape.

    §11.9.1 makes `lb` default to zero rather than to "unknown": a length is a count, and a
    count has a floor whether or not the schema wrote one down.
    """
    constraint = node.constraint
    if constraint is None:
        return 0, None, False
    low = 0 if constraint.root_size_low is None else int(constraint.root_size_low)
    high = None if constraint.root_size_high is None else int(constraint.root_size_high)
    return low, high, constraint.size_extensible


def _emit_per(node: EncodeNode, reader: _Reader, writer, rules) -> None:
    """One node's field-list. `writer` is `per.BitWriter`, whose variant carries the padding rule."""
    # §11.9.3.4's normally small LENGTH is deliberately absent: it spells §19.8's extension
    # ADDITION bitmap, and `_compile_members` refuses an extension addition outright. An
    # imported name with no call site would look like coverage this rail does not have.
    from .per import (
        _64K,
        _encode_constrained,
        _encode_integer_root,
        _encode_length_and_payload,
        _KNOWN_MULTIPLIER,
    )

    kind = node.kind
    if kind == "boolean":
        writer.put_bit(1 if reader.u8() else 0)  # §12: one bit, no alignment
        return
    if kind == "null":
        return  # §18: no encoding at all
    if kind == "integer":
        value = int.from_bytes(reader.take(reader.u8()), "big", signed=True)
        (low, high), extensible = _per_value_bounds(node)
        if extensible:
            # §13.1: one bit saying whether the value is inside the extension root. Outside,
            # §13.2's constrained forms do not apply and the value goes unconstrained.
            inside = (low is None or value >= low) and (high is None or value <= high)
            writer.put_bit(0 if inside else 1)
            if not inside:
                from .per import _encode_unconstrained

                _encode_unconstrained(writer, value)
                return
        _encode_integer_root(writer, value, low, high)  # §13.2
        return
    if kind == "enumerated":
        value = int.from_bytes(reader.take(reader.u8()), "big", signed=True)
        # §14.1: an ENUMERATED encodes its INDEX into the root sorted ascending, not its
        # value and not its identifier. Three rules, three different readings of one field.
        numbers = sorted({number for _name, number in node.enumeration})
        if value not in numbers:
            raise Asn1Error(f"{value} is not an enumeration value of this type (X.691 §14.1)")
        if node.extensible:
            writer.put_bit(0)  # §14.3
        _encode_constrained(writer, numbers.index(value), 0, len(numbers) - 1)  # §14.2
        return
    if kind == "octetstring":
        data = reader.blob()
        _per_octets(writer, node, data)
        return
    if kind == "string":
        text = reader.blob().decode("utf-8")
        if node.universal in _KNOWN_MULTIPLIER:
            _per_known_multiplier(writer, node, text)  # §30.5
            return
        # §30.6: everything else is its X.690 octets behind an unconstrained length.
        from .values import encode_string

        octets = encode_string(node.universal, text)
        _encode_length_and_payload(
            writer,
            len(octets),
            0,
            None,
            lambda start, stop: (writer.align(), writer.put_octets(octets[start:stop])),
        )
        return
    if kind == "oid":
        octets = _oid_octets(reader.blob().decode("ascii"))
        _encode_length_and_payload(  # §24
            writer,
            len(octets),
            0,
            None,
            lambda start, stop: (writer.align(), writer.put_octets(octets[start:stop])),
        )
        return
    if kind == "sequence":
        # §19.1: the extension bit. This plan refuses extension additions, so the type is
        # extensible with none present and the bit is always zero — but it must still be
        # THERE, because a decoder built from the same schema reads one.
        if node.extensible:
            writer.put_bit(0)
        # §19.2's presence bitmap precedes ALL the components, but the neutral stream
        # INTERLEAVES a presence octet with each value — so the bits cannot simply be
        # collected first: after reading component 0's flag the next stream octet is its
        # value, not component 1's flag. This is the shape OER's preamble already has, and
        # it takes the same answer: the bitmap's WIDTH comes from the plan and no value can
        # change it, so the slots are reserved and each bit is patched as its flag is read.
        # Collecting first desynchronizes the reader, which is how this was found.
        optional = [m for m in node.members if m.optional or m.has_default]
        bitmap_at, seen = len(writer.bits), 0
        for _ in optional:
            writer.put_bit(0)
        for member in node.members:
            # §19.5 is CANONICAL-PER's, and matches what `_present` already decides for the
            # canonical X.690/OER rows; BASIC-PER leaves the sender free.
            here = _present(member, reader, omit_default=_per_canonical(rules))
            if member.optional or member.has_default:
                if here:
                    writer.bits[bitmap_at + seen] = 1
                seen += 1
                if not here:
                    continue
            _emit_per(member.node, reader, writer, rules)  # §19.4
        return
    if kind == "sequence-of":
        count = reader.u32()
        lower, upper, extensible = _per_size_bounds(node)
        if extensible:
            inside = count >= lower and (upper is None or count <= upper)
            writer.put_bit(0 if inside else 1)  # §20.4
            if not inside:
                lower, upper = 0, None
        if upper is not None and lower == upper and upper < _64K:
            if count != upper:
                raise Asn1Error(f"expected exactly {upper} components (X.691 §20.5)")
            for _ in range(count):
                _emit_per(node.element, reader, writer, rules)
            return

        # The elements are consumed from the stream in order, so a fragmenting callback
        # cannot revisit them: the stream is read once, and each slice is emitted as it is
        # reached. `_encode_length_and_payload` calls `emit` with strictly advancing,
        # contiguous ranges, which is exactly what makes that safe.
        def emit_range(start: int, stop: int) -> None:
            for _ in range(stop - start):
                _emit_per(node.element, reader, writer, rules)

        _encode_length_and_payload(writer, count, lower, upper, emit_range)  # §20.6
        return
    if kind == "choice":
        index = reader.u32()
        chosen = [m for m in node.members if m.index == index]
        if not chosen:
            raise Asn1Error(f"CHOICE index {index} is outside the plan's alternatives")
        order = _per_alternatives(node)
        position = order.index(chosen[0])
        if node.extensible:
            writer.put_bit(0)  # §23.5
        if len(order) > 1:  # §23.4: one alternative, no index
            _encode_constrained(writer, position, 0, len(order) - 1)  # §23.6/§23.7
        _emit_per(chosen[0].node, reader, writer, rules)
        return
    raise Asn1Error(f"no PER rule for plan kind {kind!r}")


def _per_canonical(rules) -> bool:
    return rules in (EmitRules.CANONICAL_PER_ALIGNED, EmitRules.CANONICAL_PER_UNALIGNED)


#: The four rules `_emit_per` serves. A frozenset of enum members rather than a lookup into
#: `per`, so deciding *whether* a rule is PER costs no import at all.
_PER_RULES = frozenset(
    {
        EmitRules.CANONICAL_PER_ALIGNED,
        EmitRules.CANONICAL_PER_UNALIGNED,
        EmitRules.BASIC_PER_ALIGNED,
        EmitRules.BASIC_PER_UNALIGNED,
    }
)


def _per_variant(rules):
    """§11.1.3 vs §11.1.4's padding rule, as `per.PerVariant`.

    Resolved lazily because §8's import quarantine keeps research organs off the simple
    path: a module-level import of the PER codec would make every importer of this harness
    pay for it whether or not it ever emits a PER row.
    """
    from .per import PerVariant

    return (
        PerVariant.ALIGNED
        if rules in (EmitRules.CANONICAL_PER_ALIGNED, EmitRules.BASIC_PER_ALIGNED)
        else PerVariant.UNALIGNED
    )


def _per_octets(writer, node: EncodeNode, data: bytes) -> None:
    """§17: octet strings. A fixed short length stays unaligned; the rest take a length."""
    from .per import _64K, _encode_length_and_payload

    lower, upper, extensible = _per_size_bounds(node)
    count = len(data)
    if extensible:
        inside = count >= lower and (upper is None or count <= upper)
        writer.put_bit(0 if inside else 1)  # §17.3
        if not inside:
            lower, upper = 0, None
    if upper is not None and lower == upper:
        if count != upper:
            raise Asn1Error(f"octet string is {count} octets, fixed size is {upper}")
        if upper == 0:
            return  # §17.5
        if upper <= 2:  # §17.6: NOT octet-aligned
            writer.put_octets(data)
            return
        if upper < _64K:  # §17.7
            writer.align()
            writer.put_octets(data)
            return
    _encode_length_and_payload(  # §17.8
        writer,
        count,
        lower,
        upper,
        lambda start, stop: (writer.align(), writer.put_octets(data[start:stop])),
    )


def _per_known_multiplier(writer, node: EncodeNode, text: str) -> None:
    """§30.5: one of the known-multiplier character types."""
    from .per import _64K, _encode_length_and_payload, char_bits_for, default_alphabet

    alphabet = node.constraint.alphabet if node.constraint else ""
    effective = alphabet or default_alphabet(node.universal)
    bits, forward, _ = char_bits_for(node.universal, effective, writer.variant)
    lower, upper, extensible = _per_size_bounds(node)
    count = len(text)
    if extensible:
        # §30.4. The permitted ALPHABET still applies outside the root: §10.3.11 makes an
        # extensible alphabet not PER-visible at all, and a non-extensible one is unaffected
        # by a SIZE marker.
        inside = count >= lower and (upper is None or count <= upper)
        writer.put_bit(0 if inside else 1)
        if not inside:
            lower, upper = 0, None

    def emit_range(start: int, stop: int) -> None:
        # §30.5.6/§30.5.7: align only when the whole field could exceed 16 bits.
        if upper is None or upper * bits > 16:
            writer.align()
        for character in text[start:stop]:
            writer.put_bits(forward[character] if forward else ord(character), bits)

    if upper is not None and lower == upper and upper < _64K:
        if count != upper:
            raise Asn1Error(f"string is {count} characters, fixed size is {upper}")
        if count:
            emit_range(0, count)  # §30.5.6
        return
    _encode_length_and_payload(writer, count, lower, upper, emit_range)  # §30.5.7


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
    elif rules in _PER_RULES:
        from .per import BitWriter

        writer = BitWriter(_per_variant(rules))
        _emit_per(plan.root, reader, writer, rules)
        # §11.1.3.1/§11.1.4: only the OUTERMOST value is padded out to whole octets, which
        # is why this happens here and not inside `_emit_per`.
        out = writer.to_bytes()
    else:
        raise Asn1Error(f"no emitter for {rules!r}")
    if reader.at != len(stream):
        raise Asn1Error(
            f"the value stream has {len(stream) - reader.at} octets left over after "
            f"{rules.value}; the plan and the stream describe different values"
        )
    return out


__all__ = ["EmitRules", "emit", "flatten"]
