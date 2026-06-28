"""BCIR StreamPack binary ABI v1 (frozen) + v2 (append-only) -- reference codec.

Layout (little-endian; see docs/BCIR_STREAMPACK_ABI.md and
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
    Trailer:
      crc32:u32  (CRC-32 of every preceding byte)

Strings are u16 length + UTF-8. Integer arrays are u16 count + elements. The
format is frozen at v1 and evolves append-only: v2 only *appends* fields (header
pad + record tails), the encoder emits the lowest version that carries the pack
(a pack with no pipeline/double-buffer contracts is byte-identical v1), and this
reader accepts both. v1 readers reject newer versions, by contract.
"""

from __future__ import annotations

import struct
import zlib

from ..gem.streampack import Block, LaneSegment, Prefetch, StreamPack, TraceNote
from ..model import Lane

ABI_MAGIC = b"BSPK"
ABI_VERSION = 1
ABI_VERSION_MAX = 3     # v2: pipeline_depth + prefetch double-buffer; v3: segment dispatch/channel (append-only)

# v3 dispatch is a small closed enum -> u8 on the wire (channel stays a free str). "core"/"host"
# are the defaults a v1/v2 pack carries implicitly, so a pack that uses neither encodes as v1/v2.
_DISPATCH_DEFAULT = "core"
_DISPATCH_WIRE = {"core": 0, "pim": 1}            # u8 dispatch code (decoder mirrors)
_DISPATCH_FROM_WIRE = {v: k for k, v in _DISPATCH_WIRE.items()}
_CHANNEL_DEFAULT = "host"

_HEADER = struct.Struct("<4sHHIIIIIII")  # magic, ver, flags, 3 gens, 4 counts
_HEADER_SIZE = 64
_PIPELINE_OFF = _HEADER.size            # v2: u16 appended right after the v1 fields


class AbiError(Exception):
    pass


class _Writer:
    def __init__(self) -> None:
        self.buf = bytearray()

    def u8(self, v: int) -> None:
        self.buf += struct.pack("<B", v & 0xFF)

    def u16(self, v: int) -> None:
        self.buf += struct.pack("<H", v & 0xFFFF)

    def u32(self, v: int) -> None:
        self.buf += struct.pack("<I", v & 0xFFFFFFFF)

    def u64(self, v: int) -> None:
        self.buf += struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF)

    def s(self, text: str) -> None:
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


class _Reader:
    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.pos = pos

    def _take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise AbiError("truncated StreamPack")
        b = self.data[self.pos:self.pos + n]
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
        return self._take(self.u16()).decode("utf-8")

    def u32_array(self) -> tuple:
        return tuple(self.u32() for _ in range(self.u16()))

    def u64_array(self) -> tuple:
        return tuple(self.u64() for _ in range(self.u16()))

    def s_array(self) -> tuple:
        return tuple(self.s() for _ in range(self.u16()))


def encode(pack: StreamPack) -> bytes:
    """Serialize a StreamPack (CRC trailer); emits the lowest carrying version.

    Packs without v2 features (pipeline_depth == 1, no double-buffer prefetch)
    encode byte-identically to the frozen v1 format. A pack that uses v3 segment
    dispatch/channel (any non-default `dispatch`/`channel`) encodes as v3; one that
    uses neither v2 nor v3 features stays byte-identical frozen v1.
    """
    needs_v2 = pack.pipeline_depth > 1 or any(pf.buffers != 1 for pf in pack.prefetches)
    needs_v3 = any(seg.dispatch != _DISPATCH_DEFAULT or seg.channel != _CHANNEL_DEFAULT
                   for seg in pack.segments)
    version = 3 if needs_v3 else (2 if needs_v2 else ABI_VERSION)
    header = _HEADER.pack(
        ABI_MAGIC, version, 0,
        pack.topo_gen, pack.map_gen, pack.data_gen,
        len(pack.segments), len(pack.prefetches), len(pack.blocks), len(pack.trace_notes),
    )
    if version >= 2:
        header += struct.pack("<H", max(1, pack.pipeline_depth))
    header = header + b"\x00" * (_HEADER_SIZE - len(header))

    w = _Writer()
    w.s(pack.source_plan)
    for seg in pack.segments:
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
            # v3 append-only segment-record tail: dispatch (u8 enum) + channel (str).
            w.u8(_DISPATCH_WIRE.get(seg.dispatch, 0))
            w.s(seg.channel)
    for pf in pack.prefetches:
        w.s(pf.name)
        w.u32(pf.distance)
        w.u32_array(pf.targets)
        w.s(pf.hint)
        w.s(pf.pattern)
        if version >= 2:
            w.u8(max(1, pf.buffers))  # v2 append-only record tail
    for blk in pack.blocks:
        w.u64(blk.base)
        w.u64(blk.count)
        w.u64_array(blk.strides)
    for t in pack.trace_notes:
        w.u64(t.claim_id)
        w.u64(t.src_hash)
        w.u64(t.trace_hash)

    body = header + bytes(w.buf)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def decode(data: bytes) -> StreamPack:
    """Parse the wire format (v1 or v2) back into a StreamPack (magic/version/CRC)."""
    if len(data) < _HEADER_SIZE + 4:
        raise AbiError("buffer too small for a StreamPack")
    magic, version, _flags, topo, mapg, datag, nseg, npf, nblk, ntr = _HEADER.unpack(
        data[:_HEADER.size])
    if magic != ABI_MAGIC:
        raise AbiError(f"bad magic {magic!r} (expected {ABI_MAGIC!r})")
    if not (ABI_VERSION <= version <= ABI_VERSION_MAX):
        raise AbiError(
            f"unsupported ABI version {version} (this reader handles v{ABI_VERSION}..v{ABI_VERSION_MAX})")
    body, crc = data[:-4], struct.unpack("<I", data[-4:])[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        raise AbiError("CRC mismatch (corrupt StreamPack)")
    depth = struct.unpack_from("<H", data, _PIPELINE_OFF)[0] if version >= 2 else 1

    r = _Reader(data, _HEADER_SIZE)
    pack = StreamPack(source_plan=r.s(), topo_gen=topo, map_gen=mapg, data_gen=datag,
                      pipeline_depth=max(1, depth))
    for _ in range(nseg):
        name = r.s(); claim_id = r.u64(); phase_id = r.u32(); lane = Lane(r.u8())
        width = r.u32(); _stride_k = r.u32(); opcode = r.s()
        reads = r.u32_array(); writes = r.u32_array()
        prefetch = r.s() or None
        fb = r.s_array(); fa = r.s_array()
        if version >= 3:
            dcode = r.u8()
            if dcode not in _DISPATCH_FROM_WIRE:
                raise AbiError(f"unknown segment dispatch code {dcode} (0=core, 1=pim)")
            dispatch = _DISPATCH_FROM_WIRE[dcode]
            channel = r.s()
        else:
            dispatch, channel = _DISPATCH_DEFAULT, _CHANNEL_DEFAULT
        pack.segments.append(LaneSegment(
            name=name, claim_id=claim_id, phase_id=phase_id, lane=lane, width=width,
            opcode=opcode, reads=reads, writes=writes, prefetch=prefetch,
            fence_before=fb, fence_after=fa, dispatch=dispatch, channel=channel))
    for _ in range(npf):
        pack.prefetches.append(Prefetch(
            name=r.s(), distance=r.u32(), targets=r.u32_array(), hint=r.s(), pattern=r.s(),
            buffers=(r.u8() if version >= 2 else 1)))
    for _ in range(nblk):
        pack.blocks.append(Block(base=r.u64(), count=r.u64(), strides=r.u64_array()))
    for _ in range(ntr):
        pack.trace_notes.append(TraceNote(claim_id=r.u64(), src_hash=r.u64(), trace_hash=r.u64()))
    return pack
