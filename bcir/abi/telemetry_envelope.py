"""TelemetryEnvelopeV0 binary ABI (version zero) -- reference codec (G15, S3-B).

The identity-carrying telemetry record the version-zero triple needs before any driver enters D2:
a NEW versioned frame, not a reinterpretation of BTLM v1 (whose bytes stay frozen and carry no
source, session, generation, clock or loss identity). Layout (little-endian; every field
fixed-width; the normative spec is docs/kernel/TELEMETRY_ENVELOPE_ABI.md and the C view
runtime/c/bcir_telemetry_envelope.h):

    Header (64 bytes, one cache line; every u64 at an 8-aligned offset):
      magic[4]="BTEV"  version:u16=0  flags:u16 (bit 0 REQUIRED)
      kind:u8 @8  schema:u8 @9  clock:u8 @10  unit:u8 @11  size:u16 @12  reserved:u16 @14
      source:u64 @16  session:u64 @24  generation:u32 @32  signal:u32 @36  seq:u32 @40
      lost:u32 @44  timestamp:u64 @48  reserved:u64 @56
    Payload (fixed per kind):
      1 sample   value:i64                                                            (8)
      2 datadna  the frozen 56-byte <7q> DataDNA record (BTLM's record, verbatim)     (56)
    Trailer:
      crc32:u32  zlib CRC-32 of every preceding byte

`source` names the producer instance, `session` changes at every producer (re)start (the restart
boundary BTLM leaves out of band), `generation` binds the measurement to the artifact generation
it was taken under (0: unbound, a host sensor), `signal` is a stable ID from the generated signal
table (bcir.signal_table), `seq` is the (source, session) record sequence modulo 2**32 and `lost`
the records of that stream the producer dropped immediately before this one -- so a consumer can
tell a producer-side drop from a transport loss. `clock`/`unit` name the timestamp's clock.

The wire laws are one predicate applied by `encode_envelope` AND `decode_envelope` and by the C
twin (`bcir_tev_decode`) identically, in the specification's order so the two rails name the same
first violation. Every refusal is a `TelemetryError` whose `status` is the C twin's `bcir_status`
name for the same bytes. Version zero: no compatibility promise until UART and virtio-blk traces
revise the field set; a change is a version bump, never a reinterpretation.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from .streampack_abi import AbiError

ENVELOPE_MAGIC = b"BTEV"
ENVELOPE_VERSION = 0
ENVELOPE_HEADER_SIZE = 64
ENVELOPE_CRC_SIZE = 4
FLAG_REQUIRED = 0x0001

ENVELOPE_KINDS = ("sample", "datadna")  # code = index + 1
PAYLOAD_SIZES = {"sample": 8, "datadna": 56}
ENVELOPE_SIZES = {
    kind: ENVELOPE_HEADER_SIZE + size + ENVELOPE_CRC_SIZE for kind, size in PAYLOAD_SIZES.items()
}
ENVELOPE_MAX = max(ENVELOPE_SIZES.values())

CLOCKS = ("none", "monotonic", "realtime", "boot", "cycles")  # code = index
TIME_UNITS = ("none", "ns", "us", "cycles")  # code = index
CLOCK_UNITS = {
    "none": ("none",),
    "monotonic": ("ns", "us"),
    "realtime": ("ns", "us"),
    "boot": ("ns", "us"),
    "cycles": ("cycles",),
}
DATADNA_FIELDS = ("claim_id", "cycles", "bytes", "misses", "thermal", "voltage", "utilization")

SIGNAL_RESERVED = 0xFFFFFFFF

# magic version flags kind schema clock unit size reserved source session generation signal seq
# lost timestamp reserved
_HEADER = struct.Struct("<4sHHBBBBHHQQIIIIQQ")
assert _HEADER.size == ENVELOPE_HEADER_SIZE
_SAMPLE = struct.Struct("<q")
_DATADNA = struct.Struct("<7q")

_U32 = 0xFFFFFFFF
_U64 = 0xFFFFFFFFFFFFFFFF


class TelemetryError(AbiError):
    """An envelope the wire laws refuse. `status` is the C twin's `bcir_status` name for the
    same bytes (`BCIR_ERR_TRUNCATED`, `_MAGIC`, `_VERSION`, `_RESERVED`, `_TELEMETRY`,
    `_TRAILING`, `_CRC`)."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status


@dataclass(frozen=True)
class TelemetryEnvelope:
    """One telemetry record: a `sample` (one reading of one signal) or a `datadna` record (the
    frozen per-claim execution record, bound to an artifact generation)."""

    kind: str
    source: int
    session: int
    seq: int
    generation: int = 0
    signal: int = 0
    lost: int = 0
    clock: str = "none"
    unit: str = "none"
    timestamp: int = 0
    required: bool = False
    value: int = 0
    record: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        return ENVELOPE_SIZES[self.kind]


def _fail(message: str) -> TelemetryError:
    return TelemetryError("BCIR_ERR_TELEMETRY", message)


def _uint(name: str, value, limit: int) -> None:
    if type(value) is not int or not 0 <= value <= limit:
        raise _fail(f"{name} must be an unsigned {limit.bit_length()}-bit integer, got {value!r}")


def _i64(name: str, value) -> None:
    if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
        raise _fail(f"{name} must be a signed 64-bit integer, got {value!r}")


def validate_envelope(env: TelemetryEnvelope) -> None:
    """The field laws, in the specification's order (every one `BCIR_ERR_TELEMETRY`): the
    encoder refuses exactly what the decoder refuses."""
    if env.kind not in ENVELOPE_KINDS:
        raise _fail(f"unknown kind {env.kind!r}")
    for name, limit in (
        ("source", _U64),
        ("session", _U64),
        ("generation", _U32),
        ("signal", _U32),
        ("seq", _U32),
        ("lost", _U32),
        ("timestamp", _U64),
    ):
        _uint(name, getattr(env, name), limit)
    if type(env.required) is not bool:
        raise _fail("required is a bool")
    if env.clock not in CLOCKS:
        raise _fail(f"unknown clock {env.clock!r}")
    if env.unit not in TIME_UNITS:
        raise _fail(f"unknown time unit {env.unit!r}")
    if env.unit not in CLOCK_UNITS[env.clock]:
        raise _fail(f"clock {env.clock!r} does not count in {env.unit!r}")
    if env.clock == "none" and env.timestamp != 0:
        raise _fail("a record with no clock carries no timestamp")
    if env.source == 0:
        raise _fail("source 0 names no producer")
    if env.session == 0:
        raise _fail("session 0 names no producer session")
    if env.kind == "sample":
        if env.signal == 0 or env.signal == SIGNAL_RESERVED:
            raise _fail("a sample names a signal (not 0, not the reserved ID)")
        _i64("value", env.value)
        if env.record != ():
            raise _fail("a sample carries no DataDNA record")
    else:
        if env.signal != 0:
            raise _fail("a DataDNA record carries its fields, not a signal ID")
        if env.required:
            raise _fail("REQUIRED qualifies a signal; a DataDNA record has none")
        if env.generation == 0:
            raise _fail("a DataDNA record is bound to the artifact generation it measured")
        if env.value != 0:
            raise _fail("a DataDNA record carries no sample value")
        if type(env.record) is not tuple or len(env.record) != len(DATADNA_FIELDS):
            raise _fail(f"a DataDNA record has {len(DATADNA_FIELDS)} fields")
        for name, value in zip(DATADNA_FIELDS, env.record):
            _i64(name, value)


def encode_envelope(env: TelemetryEnvelope) -> bytes:
    """The envelope's bytes. Refuses (`TelemetryError`) whatever `decode_envelope` would."""
    validate_envelope(env)
    size = env.size
    header = _HEADER.pack(
        ENVELOPE_MAGIC,
        ENVELOPE_VERSION,
        FLAG_REQUIRED if env.required else 0,
        ENVELOPE_KINDS.index(env.kind) + 1,
        0,
        CLOCKS.index(env.clock),
        TIME_UNITS.index(env.unit),
        size,
        0,
        env.source,
        env.session,
        env.generation,
        env.signal,
        env.seq,
        env.lost,
        env.timestamp,
        0,
    )
    payload = _SAMPLE.pack(env.value) if env.kind == "sample" else _DATADNA.pack(*env.record)
    body = header + payload
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def decode_envelope(data) -> TelemetryEnvelope:
    """Decode one envelope, applying the wire laws in the specification's order."""
    data = bytes(data)
    if len(data) < ENVELOPE_HEADER_SIZE:
        raise TelemetryError("BCIR_ERR_TRUNCATED", f"{len(data)} bytes is shorter than a header")
    (
        magic,
        version,
        flags,
        kind_code,
        schema,
        clock_code,
        unit_code,
        size,
        reserved16,
        source,
        session,
        generation,
        signal,
        seq,
        lost,
        timestamp,
        reserved64,
    ) = _HEADER.unpack_from(data, 0)
    if magic != ENVELOPE_MAGIC:
        raise TelemetryError("BCIR_ERR_MAGIC", f"magic {magic!r}")
    if version != ENVELOPE_VERSION:
        raise TelemetryError("BCIR_ERR_VERSION", f"version {version} (a v0 reader reads v0)")
    if flags & ~FLAG_REQUIRED:
        raise TelemetryError("BCIR_ERR_RESERVED", f"reserved flag bits {flags:#06x}")
    if not 1 <= kind_code <= len(ENVELOPE_KINDS):
        raise _fail(f"unknown kind code {kind_code}")
    kind = ENVELOPE_KINDS[kind_code - 1]
    if size != ENVELOPE_SIZES[kind]:
        raise _fail(f"a {kind} record is {ENVELOPE_SIZES[kind]} bytes, not {size}")
    if len(data) < size:
        raise TelemetryError("BCIR_ERR_TRUNCATED", f"{len(data)} of {size} bytes")
    if len(data) > size:
        raise TelemetryError("BCIR_ERR_TRAILING", f"{len(data) - size} bytes after the record")
    (crc,) = struct.unpack_from("<I", data, size - ENVELOPE_CRC_SIZE)
    if zlib.crc32(data[: size - ENVELOPE_CRC_SIZE]) & 0xFFFFFFFF != crc:
        raise TelemetryError("BCIR_ERR_CRC", "the CRC does not match the record")
    if reserved16 != 0 or reserved64 != 0:
        raise TelemetryError("BCIR_ERR_RESERVED", "a reserved header field is nonzero")
    if schema != 0:
        raise _fail(f"payload schema {schema} (version zero defines schema 0)")
    if clock_code >= len(CLOCKS):
        raise _fail(f"unknown clock code {clock_code}")
    if unit_code >= len(TIME_UNITS):
        raise _fail(f"unknown time unit code {unit_code}")
    if kind == "sample":
        (value,) = _SAMPLE.unpack_from(data, ENVELOPE_HEADER_SIZE)
        record: tuple[int, ...] = ()
    else:
        value = 0
        record = _DATADNA.unpack_from(data, ENVELOPE_HEADER_SIZE)
    env = TelemetryEnvelope(
        kind=kind,
        source=source,
        session=session,
        seq=seq,
        generation=generation,
        signal=signal,
        lost=lost,
        clock=CLOCKS[clock_code],
        unit=TIME_UNITS[unit_code],
        timestamp=timestamp,
        required=bool(flags & FLAG_REQUIRED),
        value=value,
        record=record,
    )
    validate_envelope(env)
    return env


def datadna_of(env: TelemetryEnvelope):
    """The DataDNA value a `datadna` envelope carries (the RT3 ingest gate still applies)."""
    from ..telemetry import DataDNA

    if env.kind != "datadna":
        raise _fail("only a datadna envelope carries a DataDNA record")
    return DataDNA("", *env.record)
