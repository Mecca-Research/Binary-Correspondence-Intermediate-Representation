"""BCIR StreamPack binary ABI v1 (frozen) -- reference encoder/decoder.

Layout (little-endian; see docs/BCIR_STREAMPACK_ABI.md and
runtime/c/bcir_streampack.h for the normative spec):

    Header (64 bytes, cache-line):
      magic[4]="BSPK"  version:u16  flags:u16
      topo_gen:u32  map_gen:u32  data_gen:u32
      n_segments:u32  n_prefetches:u32  n_blocks:u32  n_trace:u32
      reserved -> 64 bytes
    Body (sequential, length-prefixed records):
      source_plan:str
      segments[n_segments], prefetches[n_prefetches], blocks[n_blocks], trace[n_trace]
    Trailer:
      crc32:u32  (CRC-32 of every preceding byte)

Strings are u16 length + UTF-8. Integer arrays are u16 count + elements. The format
is frozen at v1: fields are append-only across versions; v1 readers reject newer
major versions.
"""

from __future__ import annotations

import struct
import zlib

from ..gem.streampack import Block, LaneSegment, Prefetch, StreamPack, TraceNote
from ..model import Lane

ABI_MAGIC = b"BSPK"
ABI_VERSION = 1

_HEADER = struct.Struct("<4sHHIIIIIII")  # magic, ver, flags, 3 gens, 4 counts
_HEADER_SIZE = 64


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
    """Serialize a StreamPack to the frozen v1 wire format (with CRC trailer)."""
    header = _HEADER.pack(
        ABI_MAGIC, ABI_VERSION, 0,
        pack.topo_gen, pack.map_gen, pack.data_gen,
        len(pack.segments), len(pack.prefetches), len(pack.blocks), len(pack.trace_notes),
    )
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
    for pf in pack.prefetches:
        w.s(pf.name)
        w.u32(pf.distance)
        w.u32_array(pf.targets)
        w.s(pf.hint)
        w.s(pf.pattern)
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
    """Parse the frozen v1 wire format back into a StreamPack (verifying magic/version/CRC)."""
    if len(data) < _HEADER_SIZE + 4:
        raise AbiError("buffer too small for a StreamPack")
    magic, version, _flags, topo, mapg, datag, nseg, npf, nblk, ntr = _HEADER.unpack(
        data[:_HEADER.size])
    if magic != ABI_MAGIC:
        raise AbiError(f"bad magic {magic!r} (expected {ABI_MAGIC!r})")
    if version != ABI_VERSION:
        raise AbiError(f"unsupported ABI version {version} (this reader is v{ABI_VERSION})")
    body, crc = data[:-4], struct.unpack("<I", data[-4:])[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        raise AbiError("CRC mismatch (corrupt StreamPack)")

    r = _Reader(data, _HEADER_SIZE)
    pack = StreamPack(source_plan=r.s(), topo_gen=topo, map_gen=mapg, data_gen=datag)
    for _ in range(nseg):
        name = r.s(); claim_id = r.u64(); phase_id = r.u32(); lane = Lane(r.u8())
        width = r.u32(); _stride_k = r.u32(); opcode = r.s()
        reads = r.u32_array(); writes = r.u32_array()
        prefetch = r.s() or None
        fb = r.s_array(); fa = r.s_array()
        pack.segments.append(LaneSegment(
            name=name, claim_id=claim_id, phase_id=phase_id, lane=lane, width=width,
            opcode=opcode, reads=reads, writes=writes, prefetch=prefetch,
            fence_before=fb, fence_after=fa))
    for _ in range(npf):
        pack.prefetches.append(Prefetch(
            name=r.s(), distance=r.u32(), targets=r.u32_array(), hint=r.s(), pattern=r.s()))
    for _ in range(nblk):
        pack.blocks.append(Block(base=r.u64(), count=r.u64(), strides=r.u64_array()))
    for _ in range(ntr):
        pack.trace_notes.append(TraceNote(claim_id=r.u64(), src_hash=r.u64(), trace_hash=r.u64()))
    return pack
