"""Binary format / field / record decoding (driver-packet ingestion).

Decodes a packed binary record (e.g. an NVMe SQE header) into named fields. This
is the binary twin of the text parser: bytes -> records -> events, via the same
descriptor-driven correspondence. Byte-aligned fields are supported (sufficient
for NVMe SQE / DDR5 timing / MMIO register maps); non-byte-aligned widths raise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryField:
    name: str
    offset_bits: int
    width_bits: int
    kind: str = "u"        # u | s | f | bytes
    semantic: str = ""


@dataclass(frozen=True)
class BinaryRecord:
    name: str
    fields: tuple[BinaryField, ...]
    size_bits: int = 0


@dataclass(frozen=True)
class BinaryFormat:
    name: str
    endianness: str = "little"     # little | big
    alignment_bits: int = 8
    records: tuple[BinaryRecord, ...] = ()


def _extract(data: bytes, field: BinaryField, endianness: str) -> int:
    if field.offset_bits % 8 != 0 or field.width_bits % 8 != 0:
        raise ValueError(f"field {field.name!r}: only byte-aligned fields are supported "
                         f"(offset={field.offset_bits}, width={field.width_bits})")
    start = field.offset_bits // 8
    nbytes = field.width_bits // 8
    chunk = data[start:start + nbytes]
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
        BinaryField("command_id", offset_bits=16, width_bits=16, kind="u", semantic="queue_command_id"),
        BinaryField("namespace_id", offset_bits=32, width_bits=32, kind="u", semantic="resource_namespace"),
    ),
    size_bits=64,
)
