"""The manifest-of-shards (BSHM, version zero) -- reference codec, split and reassembly (G16, S3-C).

A StreamPack too large to ship as one artifact travels as SHARDS named by digest, bound by one
small manifest. Every shard is itself a StreamPack a node can admit and run on its own -- the
canonical v4 sub-pack of the whole over a contiguous segment range -- and one more pack, the
FRAME, carries everything that is not a segment (the whole's prefetch, block and trace records
under its header). The manifest names the frame and each shard by SHA-256 and the whole by its
length and SHA-256, so the shards reassemble to the whole pack's bytes, byte for byte, or are
refused. Layout (little-endian; every field fixed-width; the normative spec is
docs/kernel/BCIR_SHARD_MANIFEST_ABI.md and the C view runtime/c/bcir_shard_manifest.h):

    Header (64 bytes):
      magic[4]="BSHM"  version:u16=0  flags:u16=0  n_shards:u32 @8  n_segments:u32 @12
      whole_length:u64 @16  frame_length:u64 @24  map_gen:u32 @32  data_gen:u32 @36
      topo_gen:u32 @40  pack_version:u16 @44 (=4)  reserved:u16 @46  n_gens:u32 @48
      reserved[12] @52
    Digests (96 bytes):  whole_sha256[32] @64  frame_sha256[32] @96  registry_digest[32] @128
    Shards (n_shards x 48 bytes) @160:  seg_begin:u32  seg_end:u32  length:u64  sha256[32]
    Trailer:  crc32:u32  zlib CRC-32 of every preceding byte

`registry_digest` is G14's digest of the whole's generation vector, so a node gates the manifest
against its live registry (`bcir.gem.handoff.admit_manifest`) before it fetches a single shard;
each shard carries the whole vector and is admitted again by the one R11 predicate on arrival.
Only a v4 pack with a vector shards: the vector is what every shard is admitted by.

The wire laws are one predicate applied by `encode_manifest` AND `decode_manifest` and by the C
twin (`bcir_shm_decode`) identically, in the specification's order, so the rails name the same
first violation; every reassembly refusal is `BCIR_ERR_SHARD`. Version zero: no compatibility
promise until a cluster exercises it; a change is a version bump, never a reinterpretation.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from .streampack_abi import AbiError, decode, encode, inspect_stream_pack

MANIFEST_MAGIC = b"BSHM"
MANIFEST_VERSION = 0
MANIFEST_HEADER_SIZE = 64
MANIFEST_DIGESTS_SIZE = 96
MANIFEST_ENTRY_SIZE = 48
MANIFEST_CRC_SIZE = 4
MANIFEST_FIXED = MANIFEST_HEADER_SIZE + MANIFEST_DIGESTS_SIZE + MANIFEST_CRC_SIZE  # 164
SHARDS_MAX = 4096  # the declared bound: a manifest is at most 164 + 48 * 4096 bytes
BLOB_MAX = 0xFFFFFFFF  # every length fits a 32-bit target's size_t
PACK_VERSION = 4  # only a v4 pack (one with a generation vector) shards
PACK_HEADER_SIZE = 64


def pack_minimum(n_gens: int) -> int:
    """The smallest v4 pack with `n_gens` vector entries: header, an empty source plan, the
    vector and the CRC -- a lower bound on the frame and on every shard."""
    return PACK_HEADER_SIZE + 2 + 12 * n_gens + 4


# magic version flags n_shards n_segments whole_length frame_length map data topo pack_version
# reserved n_gens reserved
_HEADER = struct.Struct("<4sHHIIQQIIIHHI12s")
assert _HEADER.size == MANIFEST_HEADER_SIZE
_ENTRY = struct.Struct("<IIQ32s")
assert _ENTRY.size == MANIFEST_ENTRY_SIZE
_U32 = 0xFFFFFFFF


class ShardError(AbiError):
    """A manifest, or a shard set, the laws refuse. `status` is the C twin's `bcir_status` name
    for the same bytes."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ShardEntry:
    seg_begin: int
    seg_end: int
    length: int
    sha256: bytes


@dataclass(frozen=True)
class ShardManifest:
    n_segments: int
    whole_length: int
    frame_length: int
    map_gen: int
    data_gen: int
    topo_gen: int
    n_gens: int
    whole_sha256: bytes
    frame_sha256: bytes
    registry_digest: bytes
    shards: tuple[ShardEntry, ...]
    pack_version: int = PACK_VERSION


@dataclass(frozen=True)
class Split:
    """One whole pack as a manifest, a frame and its shards (each a StreamPack's bytes)."""

    manifest: bytes
    frame: bytes
    shards: tuple[bytes, ...]

    def store(self) -> dict[bytes, bytes]:
        """The blobs by SHA-256 -- what a content-addressed store would hold."""
        blobs = {hashlib.sha256(self.frame).digest(): self.frame}
        blobs.update({hashlib.sha256(s).digest(): s for s in self.shards})
        return blobs


# --- the wire laws (one predicate, both directions) ------------------------------------------


def _check_entries(m: ShardManifest) -> None:
    """Laws 9-10 over a manifest's values: the lengths are bounded and the ranges partition
    [0, n_segments) into contiguous, non-empty shards (an empty pack is one empty shard)."""
    minimum = pack_minimum(m.n_gens)
    if m.whole_length > BLOB_MAX or m.frame_length > BLOB_MAX:
        raise ShardError("BCIR_ERR_SHARD", "a length exceeds the 32-bit blob bound")
    if m.frame_length < minimum:
        raise ShardError("BCIR_ERR_SHARD", "the frame is shorter than the smallest v4 pack")
    if m.whole_length < m.frame_length:
        raise ShardError("BCIR_ERR_SHARD", "the whole is shorter than its frame")
    if (m.n_segments == 0) != (m.whole_length == m.frame_length):
        raise ShardError("BCIR_ERR_SHARD", "only a pack with no segments is its own frame")
    expect = 0
    for index, entry in enumerate(m.shards):
        if not minimum <= entry.length <= BLOB_MAX:
            raise ShardError("BCIR_ERR_SHARD", f"shard {index} length is out of bounds")
        if entry.seg_begin != expect:
            raise ShardError(
                "BCIR_ERR_SHARD", f"shard {index} does not start where {index - 1} ends"
            )
        if entry.seg_end < entry.seg_begin or (
            entry.seg_end == entry.seg_begin and not (m.n_segments == 0 and len(m.shards) == 1)
        ):
            raise ShardError("BCIR_ERR_SHARD", f"shard {index} range is empty or reversed")
        expect = entry.seg_end
    if expect != m.n_segments:
        raise ShardError("BCIR_ERR_SHARD", "the shards do not end at the whole's segment count")


def decode_manifest(data: bytes) -> ShardManifest:
    """Parse and validate a manifest (the laws in the specification's order)."""
    data = bytes(data)
    if len(data) < MANIFEST_FIXED:
        raise ShardError("BCIR_ERR_TRUNCATED", "buffer too small for a shard manifest")
    (
        magic,
        version,
        flags,
        n_shards,
        n_segments,
        whole_length,
        frame_length,
        map_gen,
        data_gen,
        topo_gen,
        pack_version,
        reserved16,
        n_gens,
        reserved,
    ) = _HEADER.unpack_from(data, 0)
    if magic != MANIFEST_MAGIC:
        raise ShardError("BCIR_ERR_MAGIC", f"bad magic {magic!r} (expected {MANIFEST_MAGIC!r})")
    if version != MANIFEST_VERSION:
        raise ShardError("BCIR_ERR_VERSION", f"unsupported manifest version {version}")
    (crc,) = struct.unpack_from("<I", data, len(data) - 4)
    if zlib.crc32(data[:-4]) & _U32 != crc:
        raise ShardError("BCIR_ERR_CRC", "CRC mismatch (corrupt manifest)")
    if flags or reserved16 or any(reserved):
        raise ShardError("BCIR_ERR_RESERVED", "reserved manifest fields must be zero")
    if not 1 <= n_shards <= SHARDS_MAX:
        raise ShardError("BCIR_ERR_SHARD", f"n_shards {n_shards} outside 1..{SHARDS_MAX}")
    size = MANIFEST_FIXED + MANIFEST_ENTRY_SIZE * n_shards
    if len(data) < size:
        raise ShardError("BCIR_ERR_TRUNCATED", "the manifest is shorter than its shard count")
    if len(data) > size:
        raise ShardError("BCIR_ERR_TRAILING", "bytes follow the manifest's last shard")
    if pack_version != PACK_VERSION or n_gens == 0:
        raise ShardError("BCIR_ERR_SHARD", "only a v4 pack with a generation vector shards")
    base = MANIFEST_HEADER_SIZE + MANIFEST_DIGESTS_SIZE
    entries = tuple(
        ShardEntry(*_ENTRY.unpack_from(data, base + MANIFEST_ENTRY_SIZE * i))
        for i in range(n_shards)
    )
    m = ShardManifest(
        n_segments=n_segments,
        whole_length=whole_length,
        frame_length=frame_length,
        map_gen=map_gen,
        data_gen=data_gen,
        topo_gen=topo_gen,
        n_gens=n_gens,
        whole_sha256=data[64:96],
        frame_sha256=data[96:128],
        registry_digest=data[128:160],
        shards=entries,
        pack_version=pack_version,
    )
    _check_entries(m)
    return m


def encode_manifest(m: ShardManifest) -> bytes:
    """Serialize a manifest; refuses exactly what `decode_manifest` refuses (the bytes it would
    write are decoded before they are returned)."""
    for name in ("whole_sha256", "frame_sha256", "registry_digest"):
        value = getattr(m, name)
        if not isinstance(value, (bytes, bytearray)) or len(value) != 32:
            raise ShardError("BCIR_ERR_SHARD", f"{name} must be 32 bytes")
    if not 1 <= len(m.shards) <= SHARDS_MAX:
        raise ShardError("BCIR_ERR_SHARD", f"n_shards {len(m.shards)} outside 1..{SHARDS_MAX}")
    try:
        header = _HEADER.pack(
            MANIFEST_MAGIC,
            MANIFEST_VERSION,
            0,
            len(m.shards),
            m.n_segments,
            m.whole_length,
            m.frame_length,
            m.map_gen,
            m.data_gen,
            m.topo_gen,
            m.pack_version,
            0,
            m.n_gens,
            bytes(12),
        )
        body = header + bytes(m.whole_sha256) + bytes(m.frame_sha256) + bytes(m.registry_digest)
        for entry in m.shards:
            if not isinstance(entry.sha256, (bytes, bytearray)) or len(entry.sha256) != 32:
                raise ShardError("BCIR_ERR_SHARD", "a shard digest must be 32 bytes")
            body += _ENTRY.pack(entry.seg_begin, entry.seg_end, entry.length, bytes(entry.sha256))
    except struct.error as exc:
        raise ShardError("BCIR_ERR_SHARD", f"a manifest field is out of range: {exc}") from exc
    data = body + struct.pack("<I", zlib.crc32(body) & _U32)
    decode_manifest(data)  # the encoder refuses what the decoder refuses
    return data


# --- the canonical frame and sub-pack ------------------------------------------------------------


def pack_well_formed(pack) -> bool:
    """The module-free half of R10 the C twin's `bcir_sp_verify_semantic` applies after the
    decode: segment claim ids, trace claim ids and prefetch names unique; every segment traced;
    a named prefetch declared and feeding at least one of the segment's reads."""
    seg_ids = [s.claim_id for s in pack.segments]
    trace_ids = [t.claim_id for t in pack.trace_notes]
    names = [p.name for p in pack.prefetches]
    if len(set(seg_ids)) != len(seg_ids) or len(set(trace_ids)) != len(trace_ids):
        return False
    if len(set(names)) != len(names):
        return False
    traced = set(trace_ids)
    targets = {p.name: set(p.targets) for p in pack.prefetches}
    for seg in pack.segments:
        if seg.claim_id not in traced:
            return False
        if seg.prefetch is not None:
            if seg.prefetch not in targets:
                return False
            if seg.reads and not set(seg.reads) & targets[seg.prefetch]:
                return False
    return True


def hydrated_layout(pack) -> bool:
    """The layout every BCIR hydrator emits, and the only one the manifest shards (so a shard
    is cut in one forward pass, never by a search): one trace record per segment, in segment
    order (S1); and the prefetch records the segments name appear in segment order, each named
    by at most one segment -- unnamed records (a pipelined pack's double-buffer contracts) may
    sit anywhere (S2)."""
    if len(pack.trace_notes) != len(pack.segments):
        return False
    if any(t.claim_id != s.claim_id for t, s in zip(pack.trace_notes, pack.segments)):
        return False
    cursor = 0
    names = [p.name for p in pack.prefetches]
    for seg in pack.segments:
        if seg.prefetch is None:
            continue
        while cursor < len(names) and names[cursor] != seg.prefetch:
            cursor += 1
        if cursor == len(names):
            return False
        cursor += 1
    return True


def _whole(data: bytes):
    """Decode a pack that may shard: canonical v4 with a vector, well-formed, in the hydrated
    layout (else SHARD)."""
    try:
        pack = decode(data)
    except AbiError as exc:
        raise ShardError("BCIR_ERR_SHARD", f"not a well-formed StreamPack: {exc}") from exc
    if not pack.generations or struct.unpack_from("<H", data, 4)[0] != PACK_VERSION:
        raise ShardError("BCIR_ERR_SHARD", "only a v4 pack with a generation vector shards")
    if not pack_well_formed(pack):
        raise ShardError("BCIR_ERR_SHARD", "the pack violates a module-free R10 law")
    if not hydrated_layout(pack):
        raise ShardError("BCIR_ERR_SHARD", "the pack is not in the hydrated layout")
    if encode(pack) != bytes(data):
        raise ShardError("BCIR_ERR_SHARD", "the pack is not in its canonical spelling")
    return pack


def _sub(pack, begin: int, end: int) -> bytes:
    segments = pack.segments[begin:end]
    used = {s.prefetch for s in segments if s.prefetch is not None}
    claims = {s.claim_id for s in segments}
    return encode(
        replace(
            pack,
            segments=list(segments),
            prefetches=[p for p in pack.prefetches if p.name in used],
            blocks=[],
            trace_notes=[t for t in pack.trace_notes if t.claim_id in claims],
        )
    )


def sub_pack(whole: bytes, begin: int, end: int) -> bytes:
    """The canonical v4 sub-pack of `whole` over segments [begin, end): those segments, the
    prefetch records they name and their trace records (each in the whole's order -- which
    the hydrated layout makes segment order), no blocks, and the whole vector under the whole's
    tags and pipeline depth."""
    pack = _whole(whole)
    if not 0 <= begin <= end <= len(pack.segments):
        raise ShardError("BCIR_ERR_SHARD", f"range [{begin}, {end}) is outside the pack")
    return _sub(pack, begin, end)


def _frame(pack) -> bytes:
    """The one spelling of a (decoded) whole's frame -- `frame_of`, `split` and `reassemble` all
    use it, so a defect in it cannot hide on a path the gates do not measure (L14)."""
    return encode(replace(pack, segments=[]))


def frame_of(whole: bytes) -> bytes:
    """The frame: the whole with no segments -- every prefetch, block and trace record and the
    vector, under the whole's tags. A pack with no segments is its own frame."""
    return _frame(_whole(whole))


def partition(n_segments: int, world: int) -> list[tuple[int, int]]:
    """The contiguous balanced partition the seam's `DistributedOrchestrator::shard()` uses:
    ceil(n / world) segments per shard, so there may be fewer shards than ranks. An empty pack
    is one empty shard."""
    world = max(1, world)
    if n_segments == 0:
        return [(0, 0)]
    per = -(-n_segments // world)
    return [(b, min(b + per, n_segments)) for b in range(0, n_segments, per)]


def split(whole: bytes, ranges: Sequence[tuple[int, int]]) -> Split:
    """Shard `whole` over `ranges` (a partition of its segments): the frame, the canonical
    sub-pack of every range and the manifest that binds them by digest."""
    whole = bytes(whole)
    pack = _whole(whole)
    from ..gem.control import registry_digest

    frame = _frame(pack)
    shards = tuple(_sub(pack, b, e) for b, e in ranges)
    manifest = ShardManifest(
        n_segments=len(pack.segments),
        whole_length=len(whole),
        frame_length=len(frame),
        map_gen=pack.map_gen,
        data_gen=pack.data_gen,
        topo_gen=pack.topo_gen,
        n_gens=len(pack.generations),
        whole_sha256=hashlib.sha256(whole).digest(),
        frame_sha256=hashlib.sha256(frame).digest(),
        registry_digest=registry_digest(pack.generations),
        shards=tuple(
            ShardEntry(b, e, len(s), hashlib.sha256(s).digest())
            for (b, e), s in zip(ranges, shards)
        ),
    )
    return Split(encode_manifest(manifest), frame, shards)


def _segment_span(blob: bytes) -> tuple[int, int, int]:
    """(source_plan end, segments end, segment count) of a well-formed pack's bytes."""
    spans = inspect_stream_pack(blob).spans
    plan = next(s for s in spans if s.kind == "source_plan")
    segs = [s for s in spans if s.kind == "segment"]
    return plan.end, (segs[-1].end if segs else plan.end), len(segs)


def reassemble(manifest: bytes, fetch: Callable[[bytes], bytes | None]) -> bytes:
    """The whole pack's bytes from a manifest and a content-addressed `fetch(sha256)`, or
    `ShardError`. The one total predicate: every blob has its declared length and digest and is
    a pack; the whole built from the frame and the shards' segment records has the declared
    length and digest, is a canonical well-formed v4 pack whose tags, vector digest and segment
    count are the manifest's; and the frame and every shard are exactly the ones `split` makes
    of it -- so a node that ran shard r ran precisely segments [begin, end) of this whole."""
    m = decode_manifest(manifest)

    def blob(digest: bytes, length: int, what: str) -> bytes:
        got = fetch(bytes(digest))
        if not isinstance(got, (bytes, bytearray)) or len(got) != length:
            raise ShardError("BCIR_ERR_SHARD", f"{what} is missing or has the wrong length")
        got = bytes(got)
        if hashlib.sha256(got).digest() != bytes(digest):
            raise ShardError("BCIR_ERR_SHARD", f"{what} does not have its declared digest")
        return got

    frame = blob(m.frame_sha256, m.frame_length, "the frame")
    shards = [blob(e.sha256, e.length, f"shard {i}") for i, e in enumerate(m.shards)]
    try:
        plan_end, seg_end, count = _segment_span(frame)
        if count:
            raise ShardError("BCIR_ERR_SHARD", "the frame carries segments")
        parts = [frame[:20], struct.pack("<I", m.n_segments), frame[24:plan_end]]
        for shard in shards:
            s_plan, s_end, _ = _segment_span(shard)
            parts.append(shard[s_plan:s_end])
        parts.append(frame[plan_end:-4])
    except AbiError as exc:
        if isinstance(exc, ShardError):
            raise
        raise ShardError("BCIR_ERR_SHARD", f"a blob is not a well-formed pack: {exc}") from exc
    body = b"".join(parts)
    whole = body + struct.pack("<I", zlib.crc32(body) & _U32)
    if len(whole) != m.whole_length or hashlib.sha256(whole).digest() != m.whole_sha256:
        raise ShardError("BCIR_ERR_SHARD", "the reassembled pack is not the declared whole")
    pack = _whole(whole)
    from ..gem.control import registry_digest

    if (
        len(pack.segments) != m.n_segments
        or (pack.map_gen, pack.data_gen, pack.topo_gen) != (m.map_gen, m.data_gen, m.topo_gen)
        or len(pack.generations) != m.n_gens
        or registry_digest(pack.generations) != m.registry_digest
    ):
        raise ShardError("BCIR_ERR_SHARD", "the manifest's bindings are not the whole's")
    if _frame(pack) != frame:
        raise ShardError("BCIR_ERR_SHARD", "the frame is not the whole's frame")
    for index, (entry, shard) in enumerate(zip(m.shards, shards)):
        if _sub(pack, entry.seg_begin, entry.seg_end) != shard:
            raise ShardError("BCIR_ERR_SHARD", f"shard {index} is not the whole's sub-pack")
    return whole


__all__ = [
    "BLOB_MAX",
    "MANIFEST_ENTRY_SIZE",
    "MANIFEST_FIXED",
    "MANIFEST_HEADER_SIZE",
    "MANIFEST_MAGIC",
    "MANIFEST_VERSION",
    "PACK_VERSION",
    "SHARDS_MAX",
    "ShardEntry",
    "ShardError",
    "ShardManifest",
    "Split",
    "decode_manifest",
    "encode_manifest",
    "frame_of",
    "pack_minimum",
    "hydrated_layout",
    "pack_well_formed",
    "partition",
    "reassemble",
    "split",
    "sub_pack",
]
