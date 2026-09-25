"""BCIR StreamPack binary ABI v1 (frozen) + v2/v3/v4 (append-only) -- reference codec.

Layout (little-endian; see docs/kernel/BCIR_STREAMPACK_ABI.md and
runtime/c/bcir_streampack.h for the normative spec):

    Header (64 bytes, cache-line):
      magic[4]="BSPK"  version:u16  flags:u16
      topo_gen:u32  map_gen:u32  data_gen:u32
      n_segments:u32  n_prefetches:u32  n_blocks:u32  n_trace:u32
      [v2] pipeline_depth:u16 @36    (carved from the v1 reserved pad; v1 == 1)
      [v4] n_gens:u32 @40            (carved from the pad; 38..39 stay reserved)
      reserved -> 64 bytes
    Body (sequential, length-prefixed records):
      source_plan:str
      segments[n_segments], prefetches[n_prefetches], blocks[n_blocks], trace[n_trace]
      [v2] prefetch records append buffers:u8 (2 = double-buffer contract)
      [v3] segment records append dispatch:u8 + channel:str
      [v4] generations[n_gens], each rid:u32 map_gen:u32 data_gen:u32, RIDs strictly
           ascending; the header map_gen/data_gen are the vector's maxima (law R11)
    Trailer:
      crc32:u32  (CRC-32 of every preceding byte)

Strings are u16 length + UTF-8. Integer arrays are u16 count + elements. The
format is frozen at v1 and evolves append-only: v2/v3/v4 only *append* fields (header
pad + record tails + a trailing record family), the encoder emits the lowest version
that carries the pack (a pack with no v2/v3/v4 feature is byte-identical v1; one
carrying a generation vector is v4), and this reader accepts v1 through v4. v1 readers
reject newer versions, by contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib

from ..gem.streampack import Block, Generation, LaneSegment, Prefetch, StreamPack, TraceNote
from ..model import Lane

ABI_MAGIC = b"BSPK"
ABI_VERSION = 1
# v2: pipeline_depth + prefetch double-buffer; v3: segment dispatch/channel; v4: the
# per-resource generation vector (R11) -- all append-only.
ABI_VERSION_MAX = 4

# v3 dispatch is a small closed enum -> u8 on the wire (channel stays a free str). "core"/"host"
# are the defaults a v1/v2 pack carries implicitly, so a pack that uses neither encodes as v1/v2.
_DISPATCH_DEFAULT = "core"
_DISPATCH_WIRE = {"core": 0, "pim": 1}  # u8 dispatch code (decoder mirrors)
_DISPATCH_FROM_WIRE = {v: k for k, v in _DISPATCH_WIRE.items()}
_CHANNEL_DEFAULT = "host"

_HEADER = struct.Struct("<4sHHIIIIIII")  # magic, ver, flags, 3 gens, 4 counts
_HEADER_SIZE = 64
_PIPELINE_OFF = _HEADER.size  # v2: u16 appended right after the v1 fields (offset 36)
_GENS_OFF = 40  # v4: n_gens u32 (4-aligned; 38..39 stay reserved)
_GEN_SIZE = 12  # one generation record: rid:u32 map_gen:u32 data_gen:u32


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


def _validate_header_contract(pack: StreamPack) -> None:
    """The header half of the encode contract: the tags, the section counts and the depth."""
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


def _validate_segment(index: int, seg: LaneSegment) -> None:
    """One segment's half of the encode contract (`index` is its place in the pack)."""
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


def _prefetch_buffers_ok(pf: Prefetch) -> bool:
    return isinstance(pf.buffers, int) and not isinstance(pf.buffers, bool) and pf.buffers in (1, 2)


def _validate_prefetch(index: int, pf: Prefetch) -> None:
    """One prefetch's half of the encode contract (`index` is its place in the pack)."""
    if not _prefetch_buffers_ok(pf):
        raise AbiError(f"prefetch[{index}] buffers must be 1 or 2, got {pf.buffers!r}")


def _validate_encode_contract(pack: StreamPack) -> None:
    """Reject a model that the frozen wire cannot represent without changing meaning. The
    header, each record and the generation vector are checked in that order; the delta
    StreamPack (`gem.delta_pack`, GEM+ G18) checks the records it re-emits with the same
    functions, in the same order, so the two refuse a pack with the same first finding."""
    _validate_header_contract(pack)
    # A record whose fields are the exact types in range passes every check below; any other
    # record goes to the full check, which decides it (and names the first finding) as before.
    for index, seg in enumerate(pack.segments):
        lane, width, dispatch = seg.lane, seg.width, seg.dispatch
        if not (
            type(lane) is Lane
            and type(width) is int
            and 0 < width <= _MAX32
            and not width & (width - 1)
            and type(dispatch) is str
            and dispatch in _DISPATCH_WIRE
        ):
            _validate_segment(index, seg)
    for index, pf in enumerate(pack.prefetches):
        buffers = pf.buffers
        if type(buffers) is not int or not (buffers == 1 or buffers == 2):
            _validate_prefetch(index, pf)
    _validate_generation_vector(pack)


def _validate_generation_vector(pack: StreamPack) -> None:
    """The v4 record's well-formedness, shared by the encoder and the decoder: RIDs strictly
    ascending (sorted, unique), every field u32, and the header maxima equal to the
    vector's -- the header tags a v1-v3 reader sees are a SUMMARY of the vector, and two
    sources of truth that may disagree are a second spelling of a stale pack."""
    gens = list(pack.generations)
    _checked_uint("n_gens", len(gens), 32)
    previous = -1
    vec_map = vec_data = -1
    for index, g in enumerate(gens):
        rid, map_gen, data_gen = g.rid, g.map_gen, g.data_gen
        if not (
            type(rid) is int
            and 0 <= rid <= _MAX32
            and type(map_gen) is int
            and 0 <= map_gen <= _MAX32
            and type(data_gen) is int
            and 0 <= data_gen <= _MAX32
        ):
            rid = _checked_uint(f"generation[{index}].rid", rid, 32)
            _checked_uint(f"generation[{index}].map_gen", map_gen, 32)
            _checked_uint(f"generation[{index}].data_gen", data_gen, 32)
        if rid <= previous:
            raise AbiError(
                f"generation vector RIDs must be strictly ascending (generation[{index}] "
                f"rid {rid} after {previous})"
            )
        previous = rid
        vec_map = map_gen if map_gen > vec_map else vec_map
        vec_data = data_gen if data_gen > vec_data else vec_data
    if gens:
        if pack.map_gen != vec_map or pack.data_gen != vec_data:
            raise AbiError(
                f"header map_gen/data_gen ({pack.map_gen}, {pack.data_gen}) must be the "
                f"generation vector's maxima ({vec_map}, {vec_data})"
            )


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


# The record layouts, compiled (SP-ENC). Each record kind has ONE function (`_segment_bytes`,
# `_prefetch_bytes`, `_block_bytes`, `_trace_bytes`, `_generation_bytes`), and it builds the
# record one of two ways:
#   * a PLAIN record -- every field of its exact type, no fence names -- in one `struct` call,
#     through the precompiled layout of its shape (its string lengths and array counts);
#   * any other record, and any plain one a layout cannot carry (`struct` refuses a value out of
#     range), field by field: each field passes on an inline test (an exact `int` in range, an
#     exact `str`) or goes to the check `_Writer` makes (`_checked_uint`, the string check), in
#     wire order, so the refusal -- its type, its message, the field found first -- is
#     `_Writer`'s, and an `int` or `str` subclass the contract accepts is accepted.
# `struct` alone is not the contract: it packs a `bool` and any object with `__index__`, which the
# contract refuses (L4), so a layout only ever sees exact types. `bcir/tests/encode_fixtures.py`
# keeps the `_Writer` encoder verbatim, and the parity row holds these functions to it byte for
# byte and refusal for refusal. They are the ONE wire-record definition used by `encode`, the
# delta StreamPack and the MC1 inspector (through the `_write_*` wrappers below): keeping layout
# recovery on the writer rail avoids a second parser drifting from the frozen ABI whenever an
# append-only version adds a tail field.
_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_SEGMENT_FIXED = struct.Struct("<QIBII")  # claim_id, phase_id, lane, width, stride_k (reserved 0)
_BLOCK_FIXED = struct.Struct("<QQ")  # base, count
_TRACE_RECORD = struct.Struct("<QQQ")  # claim_id, src_hash, trace_hash
_GENERATION_RECORD = struct.Struct("<III")  # rid, map_gen, data_gen
_MAX8, _MAX16, _MAX32, _MAX64 = (1 << 8) - 1, (1 << 16) - 1, (1 << 32) - 1, (1 << 64) - 1
_SMALL = 16  # arrays shorter than this pack through a precompiled layout
_U32_ARRAYS = tuple(struct.Struct(f"<H{n}I") for n in range(_SMALL))
_U64_ARRAYS = tuple(struct.Struct(f"<H{n}Q") for n in range(_SMALL))
_BLOCKS = tuple(struct.Struct(f"<QQH{n}Q") for n in range(_SMALL))  # a plain block, by strides
_EMPTY_ARRAY = _U16.pack(0)
# The layouts of the plain records' shapes, cached up to `_SHAPES_MAX` shapes per kind, so what
# the caches hold is bounded whatever packs pass through (L3).
_SHAPES_MAX = 1024
_SEQUENCES = (tuple, list)
_SEGMENT_V2_SHAPES: dict = {}  # (name, opcode, reads, writes, prefetch) -> layout
_SEGMENT_V3_SHAPES: dict = {}  # ... + channel
_PREFETCH_V1_SHAPES: dict = {}  # (name, targets, hint, pattern) -> layout
_PREFETCH_V2_SHAPES: dict = {}


def _new_layout(shapes: dict, shape: tuple, fmt: str) -> struct.Struct:
    """Compile a record shape's layout, and keep it while the cache has room."""
    layout = struct.Struct(fmt)
    if len(shapes) < _SHAPES_MAX:
        shapes[shape] = layout
    return layout


def _str_bytes(text) -> bytes:
    """A wire string: u16 byte length + UTF-8 (`_Writer.s`)."""
    if type(text) is not str and not isinstance(text, str):
        raise AbiError(f"wire string must be str, got {type(text).__name__}")
    raw = text.encode("utf-8")
    n = len(raw)
    if n > _MAX16:
        _checked_uint("u16", n, 16)
    return _U16.pack(n) + raw


def _u32_array_bytes(xs) -> bytes:
    """u16 count + u32 elements (`_Writer.u32_array`)."""
    n = len(xs)
    if n > _MAX16:
        _checked_uint("u16", n, 16)
    if type(xs) not in _SEQUENCES:
        # Walked once, as `_Writer` walks it: below, an array is walked twice (checked, then
        # packed), and an iterable may yield its items once, or other than `len()` of them.
        return _U16.pack(n) + b"".join([_U32.pack(_checked_uint("u32", x, 32)) for x in xs])
    for x in xs:
        if type(x) is not int or not 0 <= x <= _MAX32:
            _checked_uint("u32", x, 32)
    if n < _SMALL:
        return _U32_ARRAYS[n].pack(n, *xs)
    return struct.pack(f"<H{n}I", n, *xs)


def _u64_array_bytes(xs) -> bytes:
    """u16 count + u64 elements (`_Writer.u64_array`)."""
    n = len(xs)
    if n > _MAX16:
        _checked_uint("u16", n, 16)
    if type(xs) not in _SEQUENCES:  # walked once, as `_Writer` walks it (`_u32_array_bytes`)
        return _U16.pack(n) + b"".join([_U64.pack(_checked_uint("u64", x, 64)) for x in xs])
    for x in xs:
        if type(x) is not int or not 0 <= x <= _MAX64:
            _checked_uint("u64", x, 64)
    if n < _SMALL:
        return _U64_ARRAYS[n].pack(n, *xs)
    return struct.pack(f"<H{n}Q", n, *xs)


def _str_array_bytes(xs) -> bytes:
    """u16 count + wire strings (`_Writer.s_array`), walked once."""
    n = len(xs)
    if not n and type(xs) in _SEQUENCES:  # anything else may yield what its `len()` denies
        return _EMPTY_ARRAY
    if n > _MAX16:
        _checked_uint("u16", n, 16)
    return _U16.pack(n) + b"".join([_str_bytes(x) for x in xs])


def _segment_bytes(seg: LaneSegment, version: int) -> bytes:
    try:
        wire = _plain_segment(seg, version)
    except Exception:  # noqa: BLE001 -- not carried by a layout: the field path decides it
        wire = None
    return wire if wire is not None else _segment_fields(seg, version)


def _plain_segment(seg: LaneSegment, version: int) -> bytes | None:
    """A plain segment's record in one `pack`, or None when the segment is not plain."""
    name, opcode, prefetch = seg.name, seg.opcode, seg.prefetch or ""
    claim_id, phase_id, lane, width = seg.claim_id, seg.phase_id, seg.lane, seg.width
    reads, writes = seg.reads, seg.writes
    fence_before, fence_after = seg.fence_before, seg.fence_after
    if not (
        type(name) is str
        and type(opcode) is str
        and type(prefetch) is str
        and type(claim_id) is int
        and type(phase_id) is int
        and type(lane) is Lane
        and type(width) is int
        and type(reads) in _SEQUENCES
        and type(writes) in _SEQUENCES
        # EMPTY fence arrays, not merely falsy ones: `None` is no array (the wire refuses it)
        and type(fence_before) in _SEQUENCES
        and not fence_before
        and type(fence_after) in _SEQUENCES
        and not fence_after
    ):
        return None
    for x in reads:
        if type(x) is not int:
            return None
    for x in writes:
        if type(x) is not int:
            return None
    name_b, opcode_b, prefetch_b = name.encode(), opcode.encode(), prefetch.encode()
    a, b, r, w, c = len(name_b), len(opcode_b), len(reads), len(writes), len(prefetch_b)
    if version < 3:
        shape = (a, b, r, w, c)
        layout = _SEGMENT_V2_SHAPES.get(shape) or _new_layout(
            _SEGMENT_V2_SHAPES, shape, f"<H{a}sQIBIIH{b}sH{r}IH{w}IH{c}sHH"
        )
        # stride_k is reserved 0, and the two fence arrays are empty
        return layout.pack(
            a, name_b, claim_id, phase_id, int(lane), width, 0, b, opcode_b,
            r, *reads, w, *writes, c, prefetch_b, 0, 0,
        )  # fmt: skip
    channel = seg.channel
    if type(channel) is not str:
        return None
    channel_b = channel.encode()
    d = len(channel_b)
    shape = (a, b, r, w, c, d)
    layout = _SEGMENT_V3_SHAPES.get(shape) or _new_layout(
        _SEGMENT_V3_SHAPES, shape, f"<H{a}sQIBIIH{b}sH{r}IH{w}IH{c}sHHBH{d}s"
    )
    return layout.pack(
        a, name_b, claim_id, phase_id, int(lane), width, 0, b, opcode_b,
        r, *reads, w, *writes, c, prefetch_b, 0, 0, _DISPATCH_WIRE[seg.dispatch], d, channel_b,
    )  # fmt: skip


def _segment_fields(seg: LaneSegment, version: int) -> bytes:
    """The segment record field by field: each field checked in wire order, exactly as `_Writer`
    checks it."""
    name = _str_bytes(seg.name)
    claim_id = seg.claim_id
    if type(claim_id) is not int or not 0 <= claim_id <= _MAX64:
        claim_id = _checked_uint("u64", claim_id, 64)
    phase_id = seg.phase_id
    if type(phase_id) is not int or not 0 <= phase_id <= _MAX32:
        phase_id = _checked_uint("u32", phase_id, 32)
    lane = int(seg.lane)
    if not 0 <= lane <= _MAX8:
        _checked_uint("u8", lane, 8)
    width = seg.width
    if type(width) is not int or not 0 <= width <= _MAX32:
        width = _checked_uint("u32", width, 32)
    parts = [
        name,
        _SEGMENT_FIXED.pack(claim_id, phase_id, lane, width, 0),  # stride_k: carried per claim
        _str_bytes(seg.opcode),
        _u32_array_bytes(seg.reads),
        _u32_array_bytes(seg.writes),
        _str_bytes(seg.prefetch or ""),
        _str_array_bytes(seg.fence_before),
        _str_array_bytes(seg.fence_after),
    ]
    if version >= 3:
        parts.append(_U8.pack(_DISPATCH_WIRE[seg.dispatch]))
        parts.append(_str_bytes(seg.channel))
    return b"".join(parts)


def _prefetch_bytes(pf: Prefetch, version: int) -> bytes:
    try:
        wire = _plain_prefetch(pf, version)
    except Exception:  # noqa: BLE001 -- not carried by a layout: the field path decides it
        wire = None
    return wire if wire is not None else _prefetch_fields(pf, version)


def _plain_prefetch(pf: Prefetch, version: int) -> bytes | None:
    """A plain prefetch's record in one `pack`, or None when the prefetch is not plain."""
    name, hint, pattern, distance, targets = pf.name, pf.hint, pf.pattern, pf.distance, pf.targets
    if not (
        type(name) is str
        and type(hint) is str
        and type(pattern) is str
        and type(distance) is int
        and type(targets) in _SEQUENCES
    ):
        return None
    for x in targets:
        if type(x) is not int:
            return None
    name_b, hint_b, pattern_b = name.encode(), hint.encode(), pattern.encode()
    a, t, h, p = len(name_b), len(targets), len(hint_b), len(pattern_b)
    shape = (a, t, h, p)
    if version < 2:
        layout = _PREFETCH_V1_SHAPES.get(shape) or _new_layout(
            _PREFETCH_V1_SHAPES, shape, f"<H{a}sIH{t}IH{h}sH{p}s"
        )
        return layout.pack(a, name_b, distance, t, *targets, h, hint_b, p, pattern_b)
    buffers = pf.buffers
    if type(buffers) is not int:
        return None
    layout = _PREFETCH_V2_SHAPES.get(shape) or _new_layout(
        _PREFETCH_V2_SHAPES, shape, f"<H{a}sIH{t}IH{h}sH{p}sB"
    )
    return layout.pack(a, name_b, distance, t, *targets, h, hint_b, p, pattern_b, buffers)


def _prefetch_fields(pf: Prefetch, version: int) -> bytes:
    """The prefetch record field by field, exactly as `_Writer` writes it."""
    name = _str_bytes(pf.name)
    distance = pf.distance
    if type(distance) is not int or not 0 <= distance <= _MAX32:
        distance = _checked_uint("u32", distance, 32)
    parts = [
        name,
        _U32.pack(distance),
        _u32_array_bytes(pf.targets),
        _str_bytes(pf.hint),
        _str_bytes(pf.pattern),
    ]
    if version >= 2:
        buffers = pf.buffers
        if type(buffers) is not int or not 0 <= buffers <= _MAX8:
            buffers = _checked_uint("u8", buffers, 8)
        parts.append(_U8.pack(buffers))
    return b"".join(parts)


def _block_bytes(blk: Block) -> bytes:
    base, count, strides = blk.base, blk.count, blk.strides
    if type(base) is int and type(count) is int and type(strides) in _SEQUENCES:
        n = len(strides)
        if n < _SMALL:
            for x in strides:
                if type(x) is not int:
                    break
            else:
                try:
                    return _BLOCKS[n].pack(base, count, n, *strides)
                except struct.error:
                    pass  # a value out of range: the field path refuses it
    return _block_fields(blk)


def _block_fields(blk: Block) -> bytes:
    """The block record field by field, exactly as `_Writer` writes it."""
    base, count = blk.base, blk.count
    if type(base) is not int or not 0 <= base <= _MAX64:
        base = _checked_uint("u64", base, 64)
    if type(count) is not int or not 0 <= count <= _MAX64:
        count = _checked_uint("u64", count, 64)
    return _BLOCK_FIXED.pack(base, count) + _u64_array_bytes(blk.strides)


def _trace_bytes(note: TraceNote) -> bytes:
    claim_id, src_hash, trace_hash = note.claim_id, note.src_hash, note.trace_hash
    if type(claim_id) is not int or not 0 <= claim_id <= _MAX64:
        claim_id = _checked_uint("u64", claim_id, 64)
    if type(src_hash) is not int or not 0 <= src_hash <= _MAX64:
        src_hash = _checked_uint("u64", src_hash, 64)
    if type(trace_hash) is not int or not 0 <= trace_hash <= _MAX64:
        trace_hash = _checked_uint("u64", trace_hash, 64)
    return _TRACE_RECORD.pack(claim_id, src_hash, trace_hash)


def _generation_bytes(g: Generation) -> bytes:
    rid, map_gen, data_gen = g.rid, g.map_gen, g.data_gen
    if type(rid) is not int or not 0 <= rid <= _MAX32:
        rid = _checked_uint("u32", rid, 32)
    if type(map_gen) is not int or not 0 <= map_gen <= _MAX32:
        map_gen = _checked_uint("u32", map_gen, 32)
    if type(data_gen) is not int or not 0 <= data_gen <= _MAX32:
        data_gen = _checked_uint("u32", data_gen, 32)
    return _GENERATION_RECORD.pack(rid, map_gen, data_gen)


def _write_segment(w: _Writer, seg: LaneSegment, version: int) -> None:
    w.buf += _segment_bytes(seg, version)


def _write_prefetch(w: _Writer, pf: Prefetch, version: int) -> None:
    w.buf += _prefetch_bytes(pf, version)


def _write_block(w: _Writer, blk: Block) -> None:
    w.buf += _block_bytes(blk)


def _write_trace(w: _Writer, note: TraceNote) -> None:
    w.buf += _trace_bytes(note)


def _write_generation(w: _Writer, g: Generation) -> None:
    w.buf += _generation_bytes(g)


def _wire_version(needs_v2: bool, needs_v3: bool, needs_v4: bool) -> int:
    """The lowest version that carries a pack: v4 for a generation vector, else v3 for segment
    dispatch/channel, else v2 for pipelining, else the frozen v1."""
    return 4 if needs_v4 else (3 if needs_v3 else (2 if needs_v2 else ABI_VERSION))


def _segment_needs_v3(seg: LaneSegment) -> bool:
    return seg.dispatch != _DISPATCH_DEFAULT or seg.channel != _CHANNEL_DEFAULT


def _encode_header(pack: StreamPack, version: int) -> bytes:
    """The 64-byte header of `pack` at `version`."""
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
    if version >= 4:
        header += b"\x00" * (_GENS_OFF - len(header))  # 38..39 stay reserved
        header += struct.pack("<I", len(pack.generations))
    return header + b"\x00" * (_HEADER_SIZE - len(header))


def _record_bytes(write, record, *version) -> bytes:
    """One record's bytes, exactly as `encode` writes it in place (`write` is one of the
    `_write_*` functions; the version argument for those that take one)."""
    w = _Writer()
    write(w, record, *version)
    return bytes(w.buf)


def encode(pack: StreamPack) -> bytes:
    """Serialize a StreamPack (CRC trailer); emits the lowest carrying version.

    Packs without v2 features (pipeline_depth == 1, no double-buffer prefetch)
    encode byte-identically to the frozen v1 format. A pack that uses v3 segment
    dispatch/channel (any non-default `dispatch`/`channel`) encodes as v3; one that
    carries a per-resource generation vector (every hydrated pack) encodes as v4; one
    that uses none of them stays byte-identical frozen v1.
    """
    _validate_encode_contract(pack)
    needs_v2 = pack.pipeline_depth > 1
    if not needs_v2:
        for pf in pack.prefetches:
            if pf.buffers != 1:
                needs_v2 = True
                break
    needs_v3 = False
    for seg in pack.segments:  # `_segment_needs_v3`, inline: one test per segment, no call
        if seg.dispatch != _DISPATCH_DEFAULT or seg.channel != _CHANNEL_DEFAULT:
            needs_v3 = True
            break
    version = _wire_version(needs_v2, needs_v3, bool(pack.generations))
    parts = [_encode_header(pack, version), _str_bytes(pack.source_plan)]
    parts += [_segment_bytes(seg, version) for seg in pack.segments]
    parts += [_prefetch_bytes(pf, version) for pf in pack.prefetches]
    parts += [_block_bytes(blk) for blk in pack.blocks]
    parts += [_trace_bytes(t) for t in pack.trace_notes]
    if version >= 4:
        parts += [_generation_bytes(g) for g in pack.generations]

    body = b"".join(parts)
    return body + _U32.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode(data: bytes) -> StreamPack:
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
    for index, g in enumerate(pack.generations):
        append("generation", index, f"rid{g.rid}", lambda w, g=g: _write_generation(w, g))

    trailer_offset = len(data) - 4
    if cursor != trailer_offset:
        raise AbiError(
            f"internal layout mismatch: record writers end at {cursor}, "
            f"CRC trailer starts at {trailer_offset}"
        )
    spans.append(WireSpan("crc32", None, "crc32", trailer_offset, 4))
    crc = struct.unpack_from("<I", data, trailer_offset)[0]
    return StreamPackInspection(pack, version, flags, crc, len(data), tuple(spans))
