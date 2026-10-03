"""The StreamPack decoder's compiled reads, held to the reader they replace (SP-DEC).
`bcir.abi.streampack_abi.decode` once read every field through `_Reader`: a method call, a
bounds check and a `struct.unpack` each. It reads the body in place now -- a local position, each
fixed group through one precompiled layout, each array through the layout of its count -- and
builds the records as slotted value classes (SP-REC). The rule is that none of it may change a
pack or a refusal. `decode_reference` below is the decoder as it was, kept verbatim with its
reader (`_ReferenceReader`); it shares the header constants and `_validate_generation_vector`,
which this slice did not change, and is read only by this grader.
One function, `measure`, grades the row the test and the fault table share:
    streampack.decode.parity   (item, rail) pairs where the compiled decoder's outcome -- the pack,
                               or its refusal: the exception's type and message -- differs from the
                               reference's. The items: every honest pack of the encoder corpus in
                               every spelling (`encode_fixtures.honest_packs`, `spellings`); one
                               pack with every record kind cut at every length, raw and with its
                               CRC remade over the cut; every byte of it replaced by six values,
                               raw and with the CRC remade; and, two faults at once, a replaced
                               byte with the body cut a little past it -- so a law checked out of
                               the reader's order is found, not only a law dropped.
On the parent tree the compiled decoder is the reference itself, so the row reads 0 there too:
it is a guard, and the rows that move are the time and the calls.
"""

from __future__ import annotations

import struct
import zlib

from bcir.abi.streampack_abi import (
    ABI_MAGIC,
    ABI_VERSION,
    ABI_VERSION_MAX,
    _CHANNEL_DEFAULT,
    _DISPATCH_DEFAULT,
    _DISPATCH_FROM_WIRE,
    _GENS_OFF,
    _HEADER,
    _HEADER_SIZE,
    _PIPELINE_OFF,
    AbiError,
    _validate_generation_vector,
    decode,
    encode,
)
from bcir.gem.streampack import Block, Generation, LaneSegment, Prefetch, StreamPack, TraceNote
from bcir.model import Lane

ROWS = ("streampack.decode.parity",)


class _ReferenceReader:
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


def decode_reference(data: bytes) -> StreamPack:
    """Parse the v1..v4 wire format back into a StreamPack (magic/version/CRC)."""
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
    reserved = bytes(data[reserved_start:_HEADER_SIZE])
    if version >= 4:  # v4 carves n_gens out of the pad; the bytes around it stay reserved
        reserved = bytes(data[reserved_start:_GENS_OFF]) + bytes(data[_GENS_OFF + 4 : _HEADER_SIZE])
    if any(reserved):
        raise AbiError("reserved StreamPack header bytes must be zero")
    n_gens = struct.unpack_from("<I", data, _GENS_OFF)[0] if version >= 4 else 0
    body, crc = data[:-4], struct.unpack("<I", data[-4:])[0]
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        raise AbiError("CRC mismatch (corrupt StreamPack)")
    depth = struct.unpack_from("<H", data, _PIPELINE_OFF)[0] if version >= 2 else 1
    if depth == 0:
        raise AbiError("pipeline_depth must be in [1, 65535]")

    r = _ReferenceReader(data, _HEADER_SIZE)
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
    for _ in range(n_gens):
        pack.generations.append(Generation(rid=r.u32(), map_gen=r.u32(), data_gen=r.u32()))
    # The vector's well-formedness is the same predicate the encoder applies (rail symmetry
    # with the C decoder's BCIR_ERR_GENERATION): ascending RIDs, header maxima.
    _validate_generation_vector(pack)
    if r.pos != len(data) - 4:
        raise AbiError(
            f"unexpected trailing body bytes: decoded through offset {r.pos}, "
            f"CRC trailer starts at {len(data) - 4}"
        )
    return pack


# --- the grader ---------------------------------------------------------------------------------

_REPLACEMENTS = (0x00, 0x01, 0x7F, 0x80, 0xFF)
_PAST = 24  # the second fault: the body cut this many bytes past a replaced byte, at most


def outcome(fn, data: bytes) -> tuple:
    """`fn(data)` as a comparable value: the pack, or the refusal's type and message."""
    try:
        return ("pack", fn(data))
    except Exception as exc:  # noqa: BLE001 -- the grader compares every outcome, crashes included
        return ("raise", type(exc).__name__, str(exc))


def sealed(body: bytes) -> bytes:
    """`body` with a CRC remade over it: the field laws behind the CRC are reached."""
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def smallest_with_every_record(packs: list) -> bytes:
    """The wire bytes of the smallest honest pack that carries every record kind, a double-buffer
    prefetch among them (its buffer count is the one byte both 1 and 2 are lawful at, so its
    replacements reach the law on 3)."""
    best = None
    for pack in packs:
        if (
            pack.segments
            and any(pf.buffers == 2 for pf in pack.prefetches)
            and pack.blocks
            and pack.trace_notes
            and pack.generations
        ):
            wire = encode(pack)
            if best is None or len(wire) < len(best):
                best = wire
    if best is None:
        raise AssertionError("no honest pack carries every record kind")
    return best


def items(packs: list):
    """Every graded input, labelled."""
    from bcir.tests.encode_fixtures import spellings

    for pack in packs:
        for label, spelled in spellings(pack):
            try:
                yield f"honest:{label}", encode(spelled)
            except Exception:  # noqa: BLE001 -- a spelling the encoder refuses has no bytes to decode
                continue
    wire = smallest_with_every_record(packs)
    body = wire[:-4]
    for cut in range(len(wire) + 1):
        yield f"cut:{cut}", wire[:cut]
        yield f"cut-sealed:{cut}", sealed(body[:cut])
    for offset in range(len(body)):
        seen = set()
        for value in (*_REPLACEMENTS, body[offset] ^ 0x01, body[offset] ^ 0x02):
            if value in seen:
                continue
            seen.add(value)
            replaced = body[:offset] + bytes((value,)) + body[offset + 1 :]
            yield f"replace:{offset}:{value}", replaced + wire[-4:]
            yield f"replace-sealed:{offset}:{value}", sealed(replaced)
            if value in (0x00, 0xFF):
                for past in range(1, _PAST + 1):
                    if offset + past < len(replaced):
                        yield (
                            f"replace-cut:{offset}:{value}:{past}",
                            sealed(replaced[: offset + past]),
                        )


def measure(seen: dict | None = None) -> dict[str, float]:
    """The row (module docstring), over every item."""
    from bcir.tests.encode_fixtures import honest_packs

    mismatches = 0
    stats = {"items": 0, "accepted": 0, "refused": 0, "messages": set(), "versions": set()}
    for label, data in items(honest_packs()):
        ref = outcome(decode_reference, data)
        new = outcome(decode, data)
        stats["items"] += 1
        if new != ref:
            mismatches += 1
            if seen is not None:
                seen.setdefault("mismatched", []).append((label, ref[:3], new[:3]))
        if ref[0] == "pack":
            stats["accepted"] += 1
            stats["versions"].add(int.from_bytes(data[4:6], "little"))
        else:
            stats["refused"] += 1
            stats["messages"].add(ref[2][:40])
    if seen is not None:
        seen.update(stats)
    return {"streampack.decode.parity": float(mismatches)}
