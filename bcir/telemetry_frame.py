"""T2: the BCIR UART telemetry frame -- a framed, CRC-sealed, resync-able transport.

A self-delimiting, integrity-checked **telemetry frame** that a bare-metal producer
drains from the existing `TelemetryRing` and emits over a byte egress (UART). This is
the Python *reference* rail of a dual-rail (Python reference <-> freestanding C twin)
codec; the C twin is `runtime/c/bcir_telemetry_frame.{c,h}`, and a byte-identical
differential test (`bcir/tests/test_telemetry_frame.py`) pins the two together --
mirroring the StreamPack discipline (`bcir/abi/streampack_abi.py` <->
`runtime/c/bcir_runtime.c`). The frozen frame ABI is documented in
``docs/TELEMETRY_FRAME_ABI.md``.

FROZEN FRAME ABI (v1, little-endian throughout):

    frame := magic[4]      ("BTLM" -- BCIR TeLeMetry; the resync anchor)
           | version:u16   (== 1; a v1 reader rejects a newer version)
           | flags:u16     (reserved, 0)
           | seq:u32       (monotonic frame sequence -- drop/reorder detection)
           | timestamp:u64 (producer timestamp, SyS-T-style; 0 if unavailable)
           | n_records:u16
           | records[n_records]   (each = the 56-byte ``<7q>`` DataDNA record,
                                    REUSING the `TelemetryRing._FMT` ring layout:
                                    claim_id, cycles, bytes, misses, thermal,
                                    voltage, utilization -- so the producer frames
                                    ring records with NO re-encoding of the body)
           | crc32:u32     (zlib-compatible CRC-32 over every preceding frame byte)

The fixed-prefix header is 22 bytes (4+2+2+4+8+2); the trailer CRC is 4 bytes; a frame
carrying ``n`` records is ``22 + 56*n + 4`` bytes. The record stride (56) is reused
verbatim from `TelemetryRing._FMT`, so the wire body of a ring drain is a memcpy.

RESYNC-ON-MAGIC: the decoder can join a stream mid-flight or recover from a corrupted
frame by SCANNING forward to the next ``BTLM`` magic. A garbage prefix, a truncated
frame, or a frame whose CRC fails costs exactly ONE frame (rejected + counted) -- the
decoder resynchronizes on the next magic and recovers every subsequent good frame. The
magic is the resync anchor; per-frame CRC bounds the blast radius of a single corrupt
byte to its own frame.

CRC REUSE (the dual-rail invariant): the Python rail uses ``zlib.crc32(body) &
0xFFFFFFFF``; the C twin uses ``bcir_crc32`` (`runtime/c/bcir_runtime.c`, the same
zlib-compatible reflected poly 0xEDB88320). They MUST agree byte-for-byte -- the
differential test proves it. We do NOT reimplement CRC on either rail.

TWO-TRUTH NOTE: a telemetry frame carries *data* (graded L2/L3 in the telemetry-source
ladder), NEVER a legality verdict. Decoding a frame produces records + an integrity
witness, never a `Diagnostic` and never a verifier verdict; it touches neither
`bcir/verify` nor the cost-vector DIMS. `parse_uart_frames` delegates to the RT3 ingest
gate (`bcir.telemetry.sanitize_events`) so the host decode path produces the SAME
`TelemetryIntegrity` witness as the ring path -- bounding the blast radius (an
out-of-band value can't move Theta; a dropped/blind stream is visible) without ever
crossing into a verdict.

EGRESS-OVER-UART (documented adapter, NOT built here): `encode_frame` returns the frame
bytes; the C producer (`bcir_tf_encode_frame`) writes them to a caller `out` buffer. To
egress over hardware, a 1-line adapter feeds those bytes to the existing MMIO UART
driver -- ``for (size_t i = 0; i < n; i++) uart_send(uart, out[i]);`` (`uart_send`,
`runtime/c/uart_regs.h`). The testable core is the frame bytes (a buffer/callback), not
a hardware poke, so T2 has no hardware dependency.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from .telemetry import DataDNA, TelemetryIntegrity, TelemetryRing, sanitize_events

# --- the frozen frame ABI (mirror bcir_telemetry_frame.h) ------------------------
TELEMETRY_FRAME_MAGIC = b"BTLM"          # BCIR TeLeMetry frame; the resync anchor
TELEMETRY_FRAME_VERSION = 1              # a v1 reader rejects a newer version
TELEMETRY_FRAME_VERSION_MAX = 1

# Reuse the ring's record layout verbatim (claim_id, cycles, bytes, misses, thermal,
# voltage, utilization == 7 x int64 == 56 bytes). The producer drains the ring and
# frames the records with NO re-encoding of the record body.
_RECORD_FMT = TelemetryRing._FMT                 # "<7q"
RECORD_STRIDE = struct.calcsize(_RECORD_FMT)     # 56

# Fixed frame header: magic[4], version:u16, flags:u16, seq:u32, timestamp:u64, n:u16.
_HEADER = struct.Struct("<4sHHIQH")
HEADER_SIZE = _HEADER.size                       # 22
CRC_SIZE = 4
MIN_FRAME_SIZE = HEADER_SIZE + CRC_SIZE          # 26 (an empty, well-formed frame)


class TelemetryFrameError(ValueError):
    """A single telemetry frame failed to decode (bad magic / unsupported version /
    truncation / CRC mismatch). The *stream* decoders (`decode_frames`,
    `parse_uart_frames`) never raise this -- they REJECT-and-count the bad frame and
    resynchronize on the next magic (a corrupt byte costs one frame, not the stream).
    `decode_frame` (the single-frame strict path) raises it so a caller decoding one
    known frame gets a hard failure."""


@dataclass(frozen=True)
class FrameMeta:
    """Per-frame metadata recovered from a decoded frame header (the drop/reorder
    surface). `seq` is the producer's monotonic frame counter; `timestamp` is the
    SyS-T-style producer clock (0 == unavailable)."""

    seq: int = 0
    timestamp: int = 0
    version: int = TELEMETRY_FRAME_VERSION
    flags: int = 0
    n_records: int = 0


def _pack_record(rec: DataDNA) -> bytes:
    """Pack one DataDNA into the 56-byte ``<7q>`` body (the ring layout). Only the seven
    ring-resident numeric fields are on the wire -- the same projection `TelemetryRing`
    makes (segment_id/provenance are not ring-resident)."""
    return struct.pack(_RECORD_FMT, rec.claim_id, rec.cycles, rec.bytes, rec.misses,
                       rec.thermal, rec.voltage, rec.utilization)


def _unpack_record(buf: bytes, off: int) -> DataDNA:
    cid, cyc, byt, mis, th, vo, ut = struct.unpack_from(_RECORD_FMT, buf, off)
    return DataDNA(segment_id="", claim_id=cid, cycles=cyc, bytes=byt, misses=mis,
                   thermal=th, voltage=vo, utilization=ut)


def encode_frame(records, *, seq: int = 0, timestamp: int = 0,
                 version: int = TELEMETRY_FRAME_VERSION) -> bytes:
    """Build ONE telemetry frame: magic + version + flags + seq + timestamp + n +
    records + CRC. Little-endian throughout; the CRC is ``zlib.crc32`` over every
    preceding byte (the same poly the C twin's `bcir_crc32` uses).

    `records` is a list of `DataDNA`; only the seven ring-resident numeric fields land
    on the wire (the 56-byte ``<7q>`` body), so a ring drain frames with no body
    re-encoding. `seq`/`timestamp` ride the header for drop/reorder/age detection."""
    n = len(records)
    if n > 0xFFFF:
        raise TelemetryFrameError(f"too many records for one frame ({n} > 65535)")
    body = bytearray(_HEADER.pack(TELEMETRY_FRAME_MAGIC, version & 0xFFFF, 0,
                                  seq & 0xFFFFFFFF, timestamp & 0xFFFFFFFFFFFFFFFF, n))
    for rec in records:
        body += _pack_record(rec)
    crc = zlib.crc32(bytes(body)) & 0xFFFFFFFF
    body += struct.pack("<I", crc)
    return bytes(body)


def decode_frame(buf: bytes) -> tuple[list[DataDNA], FrameMeta]:
    """Strict single-frame decode: parse exactly one frame at offset 0 and return
    ``(records, FrameMeta)``. Validates magic, version, length, and CRC; raises
    `TelemetryFrameError` (never crashes) on any failure. Use this when decoding one
    known, frame-aligned buffer. For a byte stream that may carry garbage / multiple /
    partial frames, use `decode_frames` (resync) or `parse_uart_frames` (RT3 gate)."""
    recs, meta, consumed = _decode_one(buf, 0, strict=True)
    return recs, meta


def _decode_one(buf, pos: int, *, strict: bool):
    """Try to decode one frame starting exactly at ``buf[pos]``. Returns
    ``(records, meta, consumed_bytes)`` on success. On failure: raise
    `TelemetryFrameError` when ``strict``; else return ``(None, None, 0)`` so the
    stream decoder can resync. NEVER reads out of bounds (every field bounds-checked
    against the real buffer length, like the StreamPack reader)."""
    blen = len(buf)

    def _fail(msg: str):
        if strict:
            raise TelemetryFrameError(msg)
        return None, None, 0

    if pos + HEADER_SIZE + CRC_SIZE > blen:
        return _fail(f"truncated frame at {pos} (need >= {HEADER_SIZE + CRC_SIZE} bytes)")
    magic, version, flags, seq, ts, n = _HEADER.unpack_from(buf, pos)
    if magic != TELEMETRY_FRAME_MAGIC:
        return _fail(f"bad magic {magic!r} (expected {TELEMETRY_FRAME_MAGIC!r})")
    if not (TELEMETRY_FRAME_VERSION <= version <= TELEMETRY_FRAME_VERSION_MAX):
        return _fail(f"unsupported frame version {version} "
                     f"(this reader handles v{TELEMETRY_FRAME_VERSION})")
    frame_len = HEADER_SIZE + n * RECORD_STRIDE + CRC_SIZE
    if pos + frame_len > blen:
        return _fail(f"frame at {pos} claims {n} records but buffer is too short "
                     f"({pos + frame_len} > {blen})")
    body = buf[pos:pos + HEADER_SIZE + n * RECORD_STRIDE]
    stored = struct.unpack_from("<I", buf, pos + HEADER_SIZE + n * RECORD_STRIDE)[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != stored:
        return _fail(f"CRC mismatch at {pos} (corrupt frame)")
    records = [_unpack_record(buf, pos + HEADER_SIZE + i * RECORD_STRIDE)
               for i in range(n)]
    meta = FrameMeta(seq=seq, timestamp=ts, version=version, flags=flags, n_records=n)
    return records, meta, frame_len


@dataclass(frozen=True)
class DecodedFrame:
    records: list[DataDNA]
    meta: FrameMeta


@dataclass(frozen=True)
class FrameStream:
    """The result of decoding a byte stream of frames (resync-aware)."""

    frames: list[DecodedFrame]
    rejected: int                # frames dropped (bad magic-aligned frame / CRC / truncation)

    @property
    def records(self) -> list[DataDNA]:
        """All records across the recovered frames, in stream order."""
        out: list[DataDNA] = []
        for f in self.frames:
            out.extend(f.records)
        return out


def decode_frames(buf: bytes) -> FrameStream:
    """Decode ALL frames in a byte stream, resynchronizing on the ``BTLM`` magic. Joins
    mid-stream / recovers from corruption: a garbage prefix, a truncated frame, or a
    frame whose CRC fails is REJECTED (counted in `rejected`) and the decoder SCANS to
    the next magic and continues. Returns a `FrameStream` (the recovered frames + a
    reject count). Never raises, never crashes on hostile input."""
    frames: list[DecodedFrame] = []
    rejected = 0
    pos = 0
    blen = len(buf)
    while pos + MIN_FRAME_SIZE <= blen:
        if buf[pos:pos + 4] != TELEMETRY_FRAME_MAGIC:
            nxt = buf.find(TELEMETRY_FRAME_MAGIC, pos + 1)
            if nxt == -1:
                break
            pos = nxt
            continue
        recs, meta, consumed = _decode_one(buf, pos, strict=False)
        if recs is None:
            # A magic-aligned frame that failed (CRC / version / truncation): count it
            # and resync on the NEXT magic (a single corrupt frame costs only itself).
            rejected += 1
            nxt = buf.find(TELEMETRY_FRAME_MAGIC, pos + 1)
            if nxt == -1:
                break
            pos = nxt
            continue
        frames.append(DecodedFrame(records=recs, meta=meta))
        pos += consumed
    return FrameStream(frames=frames, rejected=rejected)


def parse_uart_frames(buf: bytes) -> tuple[list[DataDNA], TelemetryIntegrity]:
    """The host decoder that REUSES the RT3 ingest gate: decode the frame stream, then
    delegate to `bcir.telemetry.sanitize_events` so the UART host path produces the SAME
    `TelemetryIntegrity` witness as the ring path. This is the "host decode reuses RT3"
    requirement.

    Flow:
      1. `decode_frames` recovers the good frames (resync on magic, per-frame CRC) and
         counts the frames it had to reject.
      2. Frame-level losses are carried into the witness's `dropped` field: each
         rejected frame is a lost batch of records (a drop/eviction analogue), exactly
         how the ring surfaces an overwrite via `dropped`.
      3. The recovered records pass through `sanitize_events`, which REJECTS-and-counts
         any out-of-band record (a forged ``thermal > 100``, a negative counter, ...)
         and surfaces the strictly-increasing-claim_id (monotonic) signal -- a frame
         seq gap or a reordered/replayed batch makes `monotonic` False (a stale record
         resent after a newer one), exactly as on the ring path.

    Returns ``(accepted_records, TelemetryIntegrity)``. A blind stream (empty / all-
    rejected / all-frames-dropped) sets ``witness.blind`` -- the suppression signal."""
    stream = decode_frames(buf)
    records = stream.records
    accepted, witness = sanitize_events(records, dropped=stream.rejected)
    return accepted, witness


def drain_ring_to_frame(ring: TelemetryRing, *, seq: int = 0,
                        timestamp: int = 0) -> bytes:
    """Convenience producer step: drain the ring and frame the drained records into one
    telemetry frame. Models the bare-metal producer (drain `TelemetryRing`, frame,
    egress). The frame body is the ring's own 56-byte records with no re-encoding."""
    return encode_frame(ring.drain(), seq=seq, timestamp=timestamp)


def frame_length(n_records: int) -> int:
    """Bytes a frame carrying `n_records` occupies on the wire (header + body + CRC)."""
    return HEADER_SIZE + n_records * RECORD_STRIDE + CRC_SIZE
