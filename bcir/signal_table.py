"""The version-zero signal definition table: the Python registry's built-in taxonomy as fixed-width
rows, the ID-range policy, and the generator of its C twin (G15, S3-B).

`bcir.signal_registry` is the normative taxonomy oracle (names, units, metric semantics, stable
numeric IDs 1..15). A resident driver cannot import Python, and a telemetry envelope carries a
signal as a u32 ID, not a name -- so before any driver enters D2 the taxonomy must exist as bytes
both rails agree on. This module is that projection:

  * every built-in `MetricDefinition` becomes one 64-byte little-endian row (`encode_row`), its
    unit, kind, temporality, sampling model, provenance and cost dimension as frozen codes;
  * `emit_c_header` generates `runtime/c/bcir_signal_table.h` from the same rows -- the C table is
    GENERATED, never hand-edited, and `--check` (a test and the C gate) refuses a drifted copy;
  * the ID-range policy: 0 is unassigned/local (never on a wire), ``0x1..0xFFFF`` is BCIR's
    (assigned by the registry), ``0x10000..0x7FFFFFFF`` is vendor-assigned,
    ``0x80000000..0xFFFFFFFE`` is device-local, ``0xFFFFFFFF`` is reserved;
  * `admit_signal` is the unknown-required-signal law: a record whose signal the consumer's
    table does not define is refused when the producer marked it REQUIRED and skipped (and
    counted) otherwise -- the forward-compatibility rule a v0 consumer applies to a newer
    producer, and the fail-closed rule for a signal the consumer must understand.

Version zero (the driver roadmap's rule for pre-D2 structures): the row layout and the codes carry
no compatibility promise until UART and virtio-blk traces revise them; a change is a version bump,
never a reinterpretation. The registry stays the source of truth; this table is its projection.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

from .kbcir.cost import DIMS, MemTier
from .signal_registry import (
    BmcPowerProvider,
    CacheCapacityProvider,
    CpuFreqProvider,
    DieTempProvider,
    FabricBytesProvider,
    GpuPowerProvider,
    HwmonPowerProvider,
    MemBandwidthProvider,
    MetricDefinition,
    MetricKind,
    PmuAvailabilityProvider,
    Provenance,
    RaplEnergyProvider,
    ReliabilityProvider,
    SamplingModel,
    Temporality,
    ThermalPressureProvider,
    ThrottleStateProvider,
    Unit,
)

SIGNAL_TABLE_VERSION = 0
ROW_SIZE = 64
NAME_WIDTH = 40  # NUL-padded ASCII; a name is at most NAME_WIDTH - 1 bytes

# --- the ID-range policy ------------------------------------------------------------------------
SIGNAL_ID_UNASSIGNED = 0
SIGNAL_ID_BCIR = (0x00000001, 0x0000FFFF)
SIGNAL_ID_VENDOR = (0x00010000, 0x7FFFFFFF)
SIGNAL_ID_DEVICE = (0x80000000, 0xFFFFFFFE)
SIGNAL_ID_RESERVED = 0xFFFFFFFF

# --- frozen codes (the row's bytes; index == code) ----------------------------------------------
UNIT_CODES = (
    Unit.NONE,
    Unit.PERCENT,
    Unit.MILLICELSIUS,
    Unit.MICROJOULE,
    Unit.MILLIWATT,
    Unit.MICROWATT,
    Unit.KHZ,
    Unit.BYTES,
    Unit.BYTES_PER_SECOND,
    Unit.BITMASK,
    Unit.COUNT,
    Unit.RATIO_MILLI,
)
KIND_CODES = (MetricKind.GAUGE, MetricKind.COUNTER)
TEMPORALITY_CODES = (Temporality.UNSPECIFIED, Temporality.DELTA, Temporality.CUMULATIVE)
SAMPLING_CODES = (SamplingModel.POLLED, SamplingModel.STREAMED, SamplingModel.EVENT_DRIVEN)
PROVENANCE_CODES = (Provenance.MEASURED, Provenance.MODELED, Provenance.SIMULATED)
COST_DIM_NONE = 0xFF
FLAG_MONOTONIC = 0x01

# id, unit, kind, temporality, flags, sampling, provenance, cost_dim, reserved[5], min_interval
_ROW = struct.Struct("<IBBBBBBB5xQ")
assert _ROW.size + NAME_WIDTH == ROW_SIZE


class SignalTableError(ValueError):
    """A definition cannot be projected into the version-zero table (an unknown code, an ID
    outside the BCIR range, a name too long or not ASCII, a duplicate ID)."""


def signal_range(signal_id: int) -> str:
    """The range a signal ID belongs to: ``unassigned``, ``bcir``, ``vendor``, ``device`` or
    ``reserved``."""
    if type(signal_id) is not int or not 0 <= signal_id <= 0xFFFFFFFF:
        raise SignalTableError("a signal ID is an unsigned 32-bit integer")
    if signal_id == SIGNAL_ID_UNASSIGNED:
        return "unassigned"
    if signal_id == SIGNAL_ID_RESERVED:
        return "reserved"
    if signal_id <= SIGNAL_ID_BCIR[1]:
        return "bcir"
    if signal_id <= SIGNAL_ID_VENDOR[1]:
        return "vendor"
    return "device"


def builtin_definitions() -> tuple[MetricDefinition, ...]:
    """Every built-in definition the registry assigns an ID, in ID order -- the union of the
    default provider set and the channel-selected power providers (IDs 1..15)."""
    providers = (
        ThermalPressureProvider(),
        DieTempProvider(),
        RaplEnergyProvider(),
        CpuFreqProvider(),
        CacheCapacityProvider(MemTier.L1),
        CacheCapacityProvider(MemTier.L2),
        CacheCapacityProvider(MemTier.L3),
        PmuAvailabilityProvider(),
        GpuPowerProvider(),
        BmcPowerProvider(),
        MemBandwidthProvider(),
        FabricBytesProvider(),
        ThrottleStateProvider(),
        ReliabilityProvider(),
        HwmonPowerProvider(),
    )
    defs = sorted((p.definition for p in providers), key=lambda d: d.signal_id)
    ids = [d.signal_id for d in defs]
    if len(set(ids)) != len(ids):
        raise SignalTableError(f"duplicate built-in signal IDs: {ids}")
    return tuple(defs)


def _code(table: tuple, value, what: str) -> int:
    try:
        return table.index(value)
    except ValueError:
        raise SignalTableError(f"{what} {value!r} has no version-zero code") from None


def encode_row(definition: MetricDefinition) -> bytes:
    """One definition as its 64-byte row."""
    errors = definition.validate()
    if errors:
        raise SignalTableError(f"{definition.name}: {'; '.join(errors)}")
    if signal_range(definition.signal_id) != "bcir":
        raise SignalTableError(
            f"{definition.name}: a built-in signal must carry a BCIR-range ID "
            f"(got {definition.signal_id:#x})"
        )
    name = definition.name.encode("ascii", errors="strict") if definition.name.isascii() else None
    if name is None or not 0 < len(name) < NAME_WIDTH or b"\x00" in name:
        raise SignalTableError(f"{definition.name!r}: a name is 1..{NAME_WIDTH - 1} ASCII bytes")
    cost_dim = COST_DIM_NONE if definition.cost_dim is None else DIMS.index(definition.cost_dim)
    row = _ROW.pack(
        definition.signal_id,
        _code(UNIT_CODES, definition.unit, "unit"),
        _code(KIND_CODES, definition.metric_kind, "metric kind"),
        _code(TEMPORALITY_CODES, definition.temporality, "temporality"),
        FLAG_MONOTONIC if definition.monotonic else 0,
        _code(SAMPLING_CODES, definition.sampling_model, "sampling model"),
        _code(PROVENANCE_CODES, definition.provenance, "provenance"),
        cost_dim,
        definition.min_interval_ns,
    )
    return row + name.ljust(NAME_WIDTH, b"\x00")


def encode_table(definitions=None) -> bytes:
    """The table as bytes: rows in strictly ascending ID order."""
    defs = builtin_definitions() if definitions is None else tuple(definitions)
    previous = 0
    out = bytearray()
    for definition in defs:
        if definition.signal_id <= previous:
            raise SignalTableError("table rows must be in strictly ascending signal-ID order")
        previous = definition.signal_id
        out += encode_row(definition)
    return bytes(out)


def decode_row(row: bytes) -> tuple:
    """A row's fields: (id, unit, kind, temporality, flags, sampling, provenance, cost_dim,
    min_interval_ns, name)."""
    if len(row) != ROW_SIZE:
        raise SignalTableError(f"a row is {ROW_SIZE} bytes")
    fields = _ROW.unpack_from(row, 0)
    name = row[_ROW.size :].rstrip(b"\x00").decode("ascii")
    return (*fields, name)


def table_ids(definitions=None) -> frozenset[int]:
    defs = builtin_definitions() if definitions is None else tuple(definitions)
    return frozenset(d.signal_id for d in defs)


def admit_signal(signal_id: int, required: bool, known: frozenset[int] | None = None) -> str:
    """The unknown-required-signal law: ``known`` (the table defines it), ``refused`` (the
    table does not, and the producer marked it REQUIRED) or ``skipped`` (unknown and optional:
    a newer producer's signal an older consumer counts and ignores)."""
    ids = table_ids() if known is None else known
    if signal_id in ids:
        return "known"
    return "refused" if required else "skipped"


# --- the C twin ----------------------------------------------------------------------------------

C_HEADER = "runtime/c/bcir_signal_table.h"


def _c_string(name: str) -> str:
    return '"' + name + '"'


def emit_c_header(definitions=None) -> str:
    """The generated C header: the same rows as a static const table (freestanding C11)."""
    defs = builtin_definitions() if definitions is None else tuple(definitions)
    encode_table(defs)  # every row must project before any C is written
    unit_names = [u.upper() for u in UNIT_CODES]
    lines = [
        "/* GENERATED by `python -m bcir.signal_table --emit` from bcir/signal_registry.py.",
        " * Do not edit: tools/c/check_runtime.sh and bcir/tests/test_signal_table.py refuse a copy",
        " * that differs from the generator. Version zero (docs/kernel/TELEMETRY_ENVELOPE_ABI.md): no",
        " * compatibility promise until UART and virtio-blk traces revise the row. */",
        "#ifndef BCIR_SIGNAL_TABLE_H",
        "#define BCIR_SIGNAL_TABLE_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define BCIR_SIGNAL_TABLE_VERSION {SIGNAL_TABLE_VERSION}u",
        f"#define BCIR_SIGNAL_ROW_SIZE {ROW_SIZE}u",
        f"#define BCIR_SIGNAL_NAME_WIDTH {NAME_WIDTH}u",
        f"#define BCIR_SIGNAL_COUNT {len(defs)}u",
        "",
        "/* The ID-range policy. */",
        f"#define BCIR_SIGNAL_ID_UNASSIGNED 0x{SIGNAL_ID_UNASSIGNED:08X}u",
        f"#define BCIR_SIGNAL_ID_BCIR_FIRST 0x{SIGNAL_ID_BCIR[0]:08X}u",
        f"#define BCIR_SIGNAL_ID_BCIR_LAST 0x{SIGNAL_ID_BCIR[1]:08X}u",
        f"#define BCIR_SIGNAL_ID_VENDOR_FIRST 0x{SIGNAL_ID_VENDOR[0]:08X}u",
        f"#define BCIR_SIGNAL_ID_VENDOR_LAST 0x{SIGNAL_ID_VENDOR[1]:08X}u",
        f"#define BCIR_SIGNAL_ID_DEVICE_FIRST 0x{SIGNAL_ID_DEVICE[0]:08X}u",
        f"#define BCIR_SIGNAL_ID_DEVICE_LAST 0x{SIGNAL_ID_DEVICE[1]:08X}u",
        f"#define BCIR_SIGNAL_ID_RESERVED 0x{SIGNAL_ID_RESERVED:08X}u",
        f"#define BCIR_SIGNAL_COST_DIM_NONE 0x{COST_DIM_NONE:02X}u",
        f"#define BCIR_SIGNAL_FLAG_MONOTONIC 0x{FLAG_MONOTONIC:02X}u",
        "",
        "enum bcir_signal_unit {",
        *[f"  BCIR_SIGNAL_UNIT_{name} = {i}," for i, name in enumerate(unit_names)],
        "};",
        "enum bcir_signal_kind { BCIR_SIGNAL_GAUGE = 0, BCIR_SIGNAL_COUNTER = 1 };",
        "enum bcir_signal_temporality {",
        "  BCIR_SIGNAL_UNSPECIFIED = 0,",
        "  BCIR_SIGNAL_DELTA = 1,",
        "  BCIR_SIGNAL_CUMULATIVE = 2,",
        "};",
        "enum bcir_signal_sampling {",
        "  BCIR_SIGNAL_POLLED = 0,",
        "  BCIR_SIGNAL_STREAMED = 1,",
        "  BCIR_SIGNAL_EVENT_DRIVEN = 2,",
        "};",
        "enum bcir_signal_provenance {",
        "  BCIR_SIGNAL_MEASURED = 0,",
        "  BCIR_SIGNAL_MODELED = 1,",
        "  BCIR_SIGNAL_SIMULATED = 2,",
        "};",
        "",
        "typedef struct bcir_signal_def {",
        "  uint32_t id;",
        "  uint8_t unit, kind, temporality, flags, sampling, provenance, cost_dim;",
        "  uint64_t min_interval_ns;",
        "  char name[BCIR_SIGNAL_NAME_WIDTH];",
        "} bcir_signal_def;",
        "",
        "/* Rows in strictly ascending ID order (a consumer may binary-search them). */",
        "static const bcir_signal_def BCIR_SIGNAL_TABLE[BCIR_SIGNAL_COUNT] = {",
    ]
    for d in defs:
        cost = COST_DIM_NONE if d.cost_dim is None else DIMS.index(d.cost_dim)
        lines.append(
            f"    {{{d.signal_id}u, {UNIT_CODES.index(d.unit)}u, {KIND_CODES.index(d.metric_kind)}u, "
            f"{TEMPORALITY_CODES.index(d.temporality)}u, "
            f"{FLAG_MONOTONIC if d.monotonic else 0}u, "
            f"{SAMPLING_CODES.index(d.sampling_model)}u, "
            f"{PROVENANCE_CODES.index(d.provenance)}u, {cost}u, {d.min_interval_ns}u, "
            f"{_c_string(d.name)}}},"
        )
    lines += ["};", "", "#endif /* BCIR_SIGNAL_TABLE_H */", ""]
    return "\n".join(lines)


def _repo_path(relative: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", relative))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--emit", action="store_true", help=f"write {C_HEADER}")
    action.add_argument("--check", action="store_true", help=f"refuse a drifted {C_HEADER}")
    action.add_argument("--print", action="store_true", help="print the generated header")
    parser.add_argument("--path", default=None, help="the header's path (default: the repo's)")
    args = parser.parse_args(argv)
    text = emit_c_header()
    path = args.path or _repo_path(C_HEADER)
    if args.print:
        sys.stdout.write(text)
        return 0
    if args.emit:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return 0
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            on_disk = fh.read()
    except OSError as exc:
        sys.stderr.write(f"FAIL {path}: {exc}\n")
        return 1
    if on_disk != text:
        sys.stderr.write(
            f"FAIL {path} differs from the generator (python -m bcir.signal_table --emit)\n"
        )
        return 1
    sys.stdout.write(f"PASS {path} matches the generator ({len(builtin_definitions())} rows)\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
