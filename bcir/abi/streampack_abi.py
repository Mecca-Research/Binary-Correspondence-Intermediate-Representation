"""BCIR StreamPack binary ABI v1 (frozen) + v2/v3 (append-only) -- reference codec.

Layout (little-endian; see docs/kernel/BCIR_STREAMPACK_ABI.md and
runtime/c/bcir_streampack.h for the normative spec):

    Header (64 bytes, cache-line):
      magic[4]="BSPK"  version:u16  flags:u16
      topo_gen:u32  map_gen:u32  data_gen:u32
      n_segments:u32  n_prefetches:u32  n_blocks:u32  n_trace:u32
      [v2] pipeline_depth:u16        (carved from the v1 reserved pad; v1 == 1)
      reserved -> 64 bytes
    Body (sequential, length-prefixed records):
      source_plan:str
      segments[n_segments], prefetches[n_prefetches], blocks[n_blocks], trace[n_trace]
      [v2] prefetch records append buffers:u8 (2 = double-buffer contract)
      [v3] segment records append dispatch:u8 + channel:str
    Trailer:
      crc32:u32  (CRC-32 of every preceding byte)

Strings are u16 length + UTF-8. Integer arrays are u16 count + elements. The
format is frozen at v1 and evolves append-only: v2/v3 only *append* fields (header
pad + record tails), the encoder emits the lowest version that carries the pack
(a pack with neither v2 nor v3 features is byte-identical v1), and this
reader accepts v1 through v3. v1 readers reject newer versions, by contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib

from ..gem.streampack import Block, LaneSegment, Prefetch, StreamPack, TraceNote
from ..model import Lane

ABI_MAGIC = b"BSPK"
ABI_VERSION = 1
ABI_VERSION_MAX = (
    3  # v2: pipeline_depth + prefetch double-buffer; v3: segment dispatch/channel (append-only)
)

# v3 dispatch is a small closed enum -> u8 on the wire (channel stays a free str). "core"/"host"
# are the defaults a v1/v2 pack carries implicitly, so a pack that uses neither encodes as v1/v2.
_DISPATCH_DEFAULT = "core"
_DISPATCH_WIRE = {"core": 0, "pim": 1}  # u8 dispatch code (decoder mirrors)
_DISPATCH_FROM_WIRE = {v: k for k, v in _DISPATCH_WIRE.items()}
_CHANNEL_DEFAULT = "host"

_HEADER = struct.Struct("<4sHHIIIIIII")  # magic, ver, flags, 3 gens, 4 counts
_HEADER_SIZE = 64
_PIPELINE_OFF = _HEADER.size  # v2: u16 appended right after the v1 fields


class AbiError(Exception):
    pass


@dataclass(frozen=True)
class WireSpan:
    """One exact, non-overlapping StreamPack wire region."""

    kind: str
    index: int | None
    name: str
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class StreamPackInspection:
    """Validated pack plus the byte spans used by listing/debug tools."""

    pack: StreamPack
    version: int
    flags: int
    crc32: int
    length: int
    spans: tuple[WireSpan, ...]


class _Writer:
    def __init__(self) -> None:
        self.buf = bytearray()

    def u8(self, v: int) -> None:
        self.buf += struct.pack("<B", _checked_uint("u8", v, 8))

    def u16(self, v: int) -> None:
        self.buf += struct.pack("<H", _checked_uint("u16", v, 16))

    def u32(self, v: int) -> None:
        self.buf += struct.pack("<I", _checked_uint("u32", v, 32))

    def u64(self, v: int) -> None:
        self.buf += struct.pack("<Q", _checked_uint("u64", v, 64))

    def s(self, text: str) -> None:
        if not isinstance(text, str):
            raise AbiError(f"wire string must be str, got {type(text).__name__}")
        raw = text.encode("utf-8")
        self.u16(len(raw))
        self.buf += raw

    def u32_array(self, xs) -> None:
        self.u16(len(xs))
        for x in xs:
            self.u32(x)

    def u64_array(self, xs) -> None:
        self.u16(len(xs))
        for x in xs:
            self.u64(x)

    def s_array(self, xs) -> None:
        self.u16(len(xs))
        for x in xs:
            self.s(x)


def _checked_uint(name: str, value, bits: int) -> int:
    """Return one exactly representable unsigned wire integer; never mask/wrap."""
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << bits):
        raise AbiError(f"{name} value must be an unsigned {bits}-bit integer, got {value!r}")
    return value


def _validate_encode_contract(pack: StreamPack) -> None:
    """Reject a model that the frozen wire cannot represent without changing meaning."""
    _checked_uint("topo_gen", pack.topo_gen, 32)
    _checked_uint("map_gen", pack.map_gen, 32)
    _checked_uint("data_gen", pack.data_gen, 32)
    _checked_uint("n_segments", len(pack.segments), 32)
    _checked_uint("n_prefetches", len(pack.prefetches), 32)
    _checked_uint("n_blocks", len(pack.blocks), 32)
    _checked_uint("n_trace", len(pack.trace_notes), 32)
    depth = _checked_uint("pipeline_depth", pack.pipeline_depth, 16)
    if depth == 0:
        raise AbiError("pipeline_depth must be in [1, 65535]")
    for index, seg in enumerate(pack.segments):
        if isinstance(seg.lane, bool):
            raise AbiError(f"segment[{index}] lane must be a Lane value")
        try:
            Lane(seg.lane)
        except (TypeError, ValueError) as exc:
            raise AbiError(f"segment[{index}] has unknown lane {seg.lane!r}") from exc
        width = _checked_uint(f"segment[{index}].width", seg.width, 32)
        if width == 0 or width & (width - 1):
            raise AbiError(f"segment[{index}] width must be a nonzero power of two, got {width}")
        if not isinstance(seg.dispatch, str) or seg.dispatch not in _DISPATCH_WIRE:
            raise AbiError(
                f"segment[{index}] has unknown dispatch {seg.dispatch!r}; "
                f"expected one of {sorted(_DISPATCH_WIRE)}"
            )
    for index, pf in enumerate(pack.prefetches):
        if (
            not isinstance(pf.buffers, int)
            or isinstance(pf.buffers, bool)
            or pf.buffers not in (1, 2)
        ):
            raise AbiError(f"prefetch[{index}] buffers must be 1 or 2, got {pf.buffers!r}")


class _Reader:
    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.pos = pos

    def _take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise AbiError("truncated StreamPack")
        b = self.data[self.pos : self.pos + n]
        self.pos += n
        return b

    def u8(self) -> int:
        return self._take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self._take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def s(self) -> str:
        try:
            return self._take(self.u16()).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AbiError("StreamPack string is not valid UTF-8") from exc

    def u32_array(self) -> tuple:
        return tuple(self.u32() for _ in range(self.u16()))

    def u64_array(self) -> tuple:
        return tuple(self.u64() for _ in range(self.u16()))

    def s_array(self) -> tuple:
        return tuple(self.s() for _ in range(self.u16()))


# These four writers are the ONE wire-record definition used by both `encode` and the MC1
# inspector. Keeping layout recovery on the writer rail avoids a second parser drifting from the
# frozen ABI whenever an append-only version adds a tail field.
def _write_segment(w: _Writer, seg: LaneSegment, version: int) -> None:
    w.s(seg.name)
    w.u64(seg.claim_id)
    w.u32(seg.phase_id)
    w.u8(int(seg.lane))
    w.u32(seg.width)
    w.u32(0)  # stride_k reserved on the segment record (carried per-claim)
    w.s(seg.opcode)
    w.u32_array(seg.reads)
    w.u32_array(seg.writes)
    w.s(seg.prefetch or "")
    w.s_array(seg.fence_before)
    w.s_array(seg.fence_after)
    if version >= 3:
        w.u8(_DISPATCH_WIRE[seg.dispatch])
        w.s(seg.channel)


def _write_prefetch(w: _Writer, pf: Prefetch, version: int) -> None:
    w.s(pf.name)
    w.u32(pf.distance)
    w.u32_array(pf.targets)
    w.s(pf.hint)
    w.s(pf.pattern)
    if version >= 2:
        w.u8(pf.buffers)


def _write_block(w: _Writer, blk: Block) -> None:
    w.u64(blk.base)
    w.u64(blk.count)
    w.u64_array(blk.strides)


def _write_trace(w: _Writer, note: TraceNote) -> None:
    w.u64(note.claim_id)
    w.u64(note.src_hash)
    w.u64(note.trace_hash)


def encode(pack: StreamPack) -> bytes:
    """Serialize a StreamPack (CRC trailer); emits the lowest carrying version.

    Packs without v2 features (pipeline_depth == 1, no double-buffer prefetch)
    encode byte-identically to the frozen v1 format. A pack that uses v3 segment
    dispatch/channel (any non-default `dispatch`/`channel`) encodes as v3; one that
    uses neither v2 nor v3 features stays byte-identical frozen v1.
    """
    _validate_encode_contract(pack)
    needs_v2 = pack.pipeline_depth > 1 or any(pf.buffers != 1 for pf in pack.prefetches)
    needs_v3 = any(
        seg.dispatch != _DISPATCH_DEFAULT or seg.channel != _CHANNEL_DEFAULT
        for seg in pack.segments
    )
    version = 3 if needs_v3 else (2 if needs_v2 else ABI_VERSION)
    header = _HEADER.pack(
        ABI_MAGIC,
        version,
        0,
        pack.topo_gen,
        pack.map_gen,
        pack.data_gen,
        len(pack.segments),
        len(pack.prefetches),
        len(pack.blocks),
        len(pack.trace_notes),
    )
    if version >= 2:
        header += struct.pack("<H", pack.pipeline_depth)
    header = header + b"\x00" * (_HEADER_SIZE - len(header))

    w = _Writer()
    w.s(pack.source_plan)
    for seg in pack.segments:
        _write_segment(w, seg, version)
    for pf in pack.prefetches:
        _write_prefetch(w, pf, version)
    for blk in pack.blocks:
        _write_block(w, blk)
    for t in pack.trace_notes:
        _write_trace(w, t)

    body = header + bytes(w.buf)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def decode(data: bytes) -> StreamPack:
    """Parse the v1..v3 wire format back into a StreamPack (magic/version/CRC)."""
    if len(data) < _HEADER_SIZE + 4:
        raise AbiError("buffer too small for a StreamPack")
    magic, version, flags, topo, mapg, datag, nseg, npf, nblk, ntr = _HEADER.unpack(
        data[: _HEADER.size]
    )
    if magic != ABI_MAGIC:
        raise AbiError(f"bad magic {magic!r} (expected {ABI_MAGIC!r})")
    if not (ABI_VERSION <= version <= ABI_VERSION_MAX):
        raise AbiError(
            f"unsupported ABI version {version} (this reader handles v{ABI_VERSION}..v{ABI_VERSION_MAX})"
        )
    if flags:
        raise AbiError(f"reserved StreamPack flags must be zero, got 0x{flags:04x}")
    reserved_start = _PIPELINE_OFF + (2 if version >= 2 else 0)
    if any(data[reserved_start:_HEADER_SIZE]):
        raise AbiError("reserved StreamPack header bytes must be zero")
    body, crc = data[:-4], struct.unpack("<I", data[-4:])[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        raise AbiError("CRC mismatch (corrupt StreamPack)")
    depth = struct.unpack_from("<H", data, _PIPELINE_OFF)[0] if version >= 2 else 1
    if depth == 0:
        raise AbiError("pipeline_depth must be in [1, 65535]")

    r = _Reader(data, _HEADER_SIZE)
    pack = StreamPack(
        source_plan=r.s(), topo_gen=topo, map_gen=mapg, data_gen=datag, pipeline_depth=depth
    )
    for _ in range(nseg):
        name = r.s()
        claim_id = r.u64()
        phase_id = r.u32()
        raw_lane = r.u8()
        try:
            lane = Lane(raw_lane)
        except ValueError as exc:
            raise AbiError(f"unknown segment lane code {raw_lane}") from exc
        width = r.u32()
        stride_k = r.u32()
        opcode = r.s()
        # Range gate: width must be a nonzero power of two (docs/kernel/BCIR_STREAMPACK_ABI.md, "BCIR_ERR_WIDTH").
        # The C runtime (bcir_runtime.c seg_range_ok) rejects a non-power-of-two width at decode time; the
        # oracle enforces the same law so a CRC-valid but width-corrupt pack is never accepted here while
        # the deployed runtime refuses it (rail symmetry, like the Lane(...) and dispatch-code raises).
        if width == 0 or (width & (width - 1)):
            raise AbiError(f"segment width must be a nonzero power of two, got {width}")
        if stride_k != 0:
            raise AbiError(f"reserved segment stride_k must be zero, got {stride_k}")
        reads = r.u32_array()
        writes = r.u32_array()
        prefetch = r.s() or None
        fb = r.s_array()
        fa = r.s_array()
        if version >= 3:
            dcode = r.u8()
            if dcode not in _DISPATCH_FROM_WIRE:
                raise AbiError(f"unknown segment dispatch code {dcode} (0=core, 1=pim)")
            dispatch = _DISPATCH_FROM_WIRE[dcode]
            channel = r.s()
        else:
            dispatch, channel = _DISPATCH_DEFAULT, _CHANNEL_DEFAULT
        pack.segments.append(
            LaneSegment(
                name=name,
                claim_id=claim_id,
                phase_id=phase_id,
                lane=lane,
                width=width,
                opcode=opcode,
                reads=reads,
                writes=writes,
                prefetch=prefetch,
                fence_before=fb,
                fence_after=fa,
                dispatch=dispatch,
                channel=channel,
            )
        )
    for index in range(npf):
        name = r.s()
        distance = r.u32()
        targets = r.u32_array()
        hint = r.s()
        pattern = r.s()
        buffers = r.u8() if version >= 2 else 1
        if buffers not in (1, 2):
            raise AbiError(f"prefetch[{index}] buffers must be 1 or 2, got {buffers}")
        pack.prefetches.append(
            Prefetch(
                name=name,
                distance=distance,
                targets=targets,
                hint=hint,
                pattern=pattern,
                buffers=buffers,
            )
        )
    for _ in range(nblk):
        pack.blocks.append(Block(base=r.u64(), count=r.u64(), strides=r.u64_array()))
    for _ in range(ntr):
        pack.trace_notes.append(TraceNote(claim_id=r.u64(), src_hash=r.u64(), trace_hash=r.u64()))
    if r.pos != len(data) - 4:
        raise AbiError(
            f"unexpected trailing body bytes: decoded through offset {r.pos}, "
            f"CRC trailer starts at {len(data) - 4}"
        )
    return pack


def inspect_stream_pack(data: bytes) -> StreamPackInspection:
    """Validate `data` and recover exact record spans for native listing tools.

    Span lengths are generated through the same record writers as `encode`; this is intentionally
    not a second wire parser. `decode` first validates magic/version/CRC/ranges and rejects trailing
    bytes, after which the deterministic writer sizes partition the original blob exactly.
    """
    pack = decode(data)
    _magic, version, flags, *_counts = _HEADER.unpack(data[: _HEADER.size])
    spans: list[WireSpan] = [WireSpan("header", None, "header", 0, _HEADER_SIZE)]
    cursor = _HEADER_SIZE

    def append(kind: str, index: int | None, name: str, write) -> None:
        nonlocal cursor
        writer = _Writer()
        write(writer)
        size = len(writer.buf)
        spans.append(WireSpan(kind, index, name, cursor, size))
        cursor += size

    append("source_plan", None, pack.source_plan, lambda w: w.s(pack.source_plan))
    for index, seg in enumerate(pack.segments):
        append("segment", index, seg.name, lambda w, seg=seg: _write_segment(w, seg, version))
    for index, pf in enumerate(pack.prefetches):
        append("prefetch", index, pf.name, lambda w, pf=pf: _write_prefetch(w, pf, version))
    for index, block in enumerate(pack.blocks):
        append("block", index, f"block{index}", lambda w, block=block: _write_block(w, block))
    for index, note in enumerate(pack.trace_notes):
        append("trace", index, f"claim{note.claim_id}", lambda w, note=note: _write_trace(w, note))

    trailer_offset = len(data) - 4
    if cursor != trailer_offset:
        raise AbiError(
            f"internal layout mismatch: record writers end at {cursor}, "
            f"CRC trailer starts at {trailer_offset}"
        )
    spans.append(WireSpan("crc32", None, "crc32", trailer_offset, 4))
    crc = struct.unpack_from("<I", data, trailer_offset)[0]
    return StreamPackInspection(pack, version, flags, crc, len(data), tuple(spans))
