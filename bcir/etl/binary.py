"""Binary format / field / record decoding (driver-packet ingestion).

Decodes a packed binary record (e.g. an NVMe SQE header) into named fields. This
is the binary twin of the text parser: bytes -> records -> events, via the same
descriptor-driven correspondence. Byte-aligned fields are supported (sufficient
for NVMe SQE / DDR5 timing / MMIO register maps); non-byte-aligned widths raise.

The descriptors validate at CONSTRUCTION (S0-6, row 21 of the 2026-07/08 assessment): a
field with a zero width or a negative offset, a record whose fields overlap or overrun its
declared size, a format with an unknown endianness -- each is refused where it is written,
not when a packet happens to exercise it. The MLIR twins (`bcir.binary.field` /
`bcir.binary.record` / `bcir.binary.format`) apply the same rules in their op verifiers;
the structural corpus (`bcir/verify/structural_corpus.py`) holds the two rails together.
"""

from __future__ import annotations

from dataclasses import dataclass

FIELD_KINDS = ("u", "s", "f", "bytes")
ENDIANNESS = ("little", "big", "le", "be")


def _require_name(what: str, name: object) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what}: name must be a non-empty string (got {name!r})")


@dataclass(frozen=True)
class BinaryField:
    name: str
    offset_bits: int
    width_bits: int
    kind: str = "u"  # u | s | f | bytes
    semantic: str = ""

    def __post_init__(self) -> None:
        _require_name("binary field", self.name)
        if isinstance(self.offset_bits, bool) or not isinstance(self.offset_bits, int):
            raise ValueError(f"field {self.name!r}: offset_bits must be an integer")
        if isinstance(self.width_bits, bool) or not isinstance(self.width_bits, int):
            raise ValueError(f"field {self.name!r}: width_bits must be an integer")
        if self.offset_bits < 0:
            raise ValueError(
                f"field {self.name!r}: offset_bits must be non-negative (got {self.offset_bits})"
            )
        if self.width_bits < 1:
            raise ValueError(
                f"field {self.name!r}: width_bits must be positive (got {self.width_bits})"
            )
        if self.kind not in FIELD_KINDS:
            raise ValueError(
                f"field {self.name!r}: kind must be one of {'|'.join(FIELD_KINDS)} (got {self.kind!r})"
            )

    @property
    def end_bits(self) -> int:
        return self.offset_bits + self.width_bits


@dataclass(frozen=True)
class BinaryRecord:
    name: str
    fields: tuple[BinaryField, ...]
    size_bits: int = 0

    def __post_init__(self) -> None:
        _require_name("binary record", self.name)
        if not isinstance(self.fields, tuple) or not all(
            isinstance(f, BinaryField) for f in self.fields
        ):
            raise ValueError(f"record {self.name!r}: fields must be a tuple of BinaryField")
        if isinstance(self.size_bits, bool) or not isinstance(self.size_bits, int):
            raise ValueError(f"record {self.name!r}: size_bits must be an integer")
        if self.size_bits < 0:
            raise ValueError(
                f"record {self.name!r}: size_bits must be non-negative (got {self.size_bits})"
            )
        seen: set[str] = set()
        for f in self.fields:
            if f.name in seen:
                raise ValueError(f"record {self.name!r}: duplicate field name {f.name!r}")
            seen.add(f.name)
            if self.size_bits and f.end_bits > self.size_bits:
                raise ValueError(
                    f"record {self.name!r}: field {f.name!r} ends at bit {f.end_bits}, beyond "
                    f"the record's {self.size_bits} bits"
                )
        # Fields are disjoint bit ranges (a union is a different descriptor, not a record).
        ordered = sorted(self.fields, key=lambda f: (f.offset_bits, f.end_bits))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.offset_bits < prev.end_bits:
                raise ValueError(
                    f"record {self.name!r}: fields {prev.name!r} and {cur.name!r} overlap "
                    f"(bits [{prev.offset_bits}, {prev.end_bits}) and "
                    f"[{cur.offset_bits}, {cur.end_bits}))"
                )


@dataclass(frozen=True)
class BinaryFormat:
    name: str
    endianness: str = "little"  # little | big
    alignment_bits: int = 8
    records: tuple[BinaryRecord, ...] = ()

    def __post_init__(self) -> None:
        _require_name("binary format", self.name)
        if self.endianness not in ENDIANNESS:
            raise ValueError(
                f"format {self.name!r}: endianness must be one of {'|'.join(ENDIANNESS)} "
                f"(got {self.endianness!r})"
            )
        if (
            isinstance(self.alignment_bits, bool)
            or not isinstance(self.alignment_bits, int)
            or self.alignment_bits < 1
        ):
            raise ValueError(
                f"format {self.name!r}: alignment_bits must be positive (got {self.alignment_bits})"
            )
        if not isinstance(self.records, tuple) or not all(
            isinstance(r, BinaryRecord) for r in self.records
        ):
            raise ValueError(f"format {self.name!r}: records must be a tuple of BinaryRecord")
        names = [r.name for r in self.records]
        if len(set(names)) != len(names):
            raise ValueError(f"format {self.name!r}: duplicate record names")


def _extract(data: bytes, field: BinaryField, endianness: str) -> int:
    if field.offset_bits % 8 != 0 or field.width_bits % 8 != 0:
        raise ValueError(
            f"field {field.name!r}: only byte-aligned fields are supported "
            f"(offset={field.offset_bits}, width={field.width_bits})"
        )
    start = field.offset_bits // 8
    nbytes = field.width_bits // 8
    chunk = data[start : start + nbytes]
    if len(chunk) != nbytes:
        raise ValueError(f"field {field.name!r}: out of bounds (need {nbytes} bytes at {start})")
    byteorder = "little" if endianness in ("little", "le") else "big"
    return int.from_bytes(chunk, byteorder=byteorder, signed=(field.kind == "s"))


def decode(record: BinaryRecord, data: bytes, endianness: str = "little") -> dict[str, int]:
    """Decode `data` into `{field_name: value}` per the record's field layout."""
    return {f.name: _extract(data, f, endianness) for f in record.fields}


# A reference NVMe Submission Queue Entry header (first 8 bytes), little-endian.
NVME_SQE_HEADER = BinaryRecord(
    name="nvme_sqe_header",
    fields=(
        BinaryField("opcode", offset_bits=0, width_bits=8, kind="u", semantic="command_opcode"),
        BinaryField(
            "command_id", offset_bits=16, width_bits=16, kind="u", semantic="queue_command_id"
        ),
        BinaryField(
            "namespace_id", offset_bits=32, width_bits=32, kind="u", semantic="resource_namespace"
        ),
    ),
    size_bits=64,
)
