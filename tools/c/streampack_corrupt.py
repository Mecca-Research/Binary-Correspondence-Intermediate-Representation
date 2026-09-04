"""CRC-fixing StreamPack semantic-corruption mutator (the adversarial harness).

Produces VALID-CRC StreamPacks that are SEMANTICALLY corrupt -- the byte-identity /
CRC check still passes, but a law of the GEM stream (R10 provenance, R11 generation)
or the lane/width/dispatch range is violated. Each is the kind of pack the freestanding
C decoder/executor used to accept silently; the C semantic verifier
(`bcir_sp_verify_semantic` / `bcir_sp_execute_checked` in runtime/c/bcir_runtime.c +
bcir_exec.c) now REJECTS each with a clean BCIR_ERR_* status. The Python verifier
`bcir/verify::verify_pack` diagnoses the same packs, so the two rails AGREE.

Used by tools/c/check_streampack_semantic.sh (the gate) and bcir/tests/test_abi.py
(the in-process differential). Run standalone:

    python3 -m tools.c.streampack_corrupt <kind> <out.bin>
    kinds: dangling_claim swapped_rid out_of_range_lane out_of_range_width
           unresolved_prefetch invalid_pipeline invalid_buffers stale_generation
           tampered_dispatch trailing_bytes
           reserved_flags reserved_header reserved_stride
           invalid_utf8 duplicate_segment duplicate_prefetch duplicate_trace

The lettered groups are distinct adversarial classes; `--expected-gens` for stale prints
the live (map_gen, data_gen) the C rail should
be told so it can detect the staleness.
"""

from __future__ import annotations

import struct
import sys
import zlib

from bcir.abi import encode
from bcir.abi.streampack_abi import _HEADER_SIZE
from bcir.gem.streampack import LaneSegment, Prefetch, StreamPack, TraceNote
from bcir.model import Lane

# The live (registry) generation a fresh base pack carries -- the C rail is told these as
# the "expected" generation, so only the stale mutation (which bumps them) trips R11.
BASE_MAP_GEN = 7
BASE_DATA_GEN = 19

KINDS = (
    "dangling_claim",  # (a) seg.claim_id redirected away from its trace note
    "swapped_rid",  # (b) prefetch targets redirected to undeclared RIDs (no read fed)
    "out_of_range_lane",  # (c) seg.lane u8 outside the 6 valid lanes
    "out_of_range_width",  # (c) seg.width non-power-of-two
    "unresolved_prefetch",  # (d) seg.prefetch names a prefetch the pack does not declare
    "invalid_pipeline",  # (e) v2 pipeline_depth is zero
    "invalid_buffers",  # (e) v2 prefetch buffers is outside {1,2}
    "stale_generation",  # (f) map_gen/data_gen bumped past the live registry
    "tampered_dispatch",  # (g) v3 dispatch byte flipped to an illegal code (was off-wire)
    "trailing_bytes",  # (h) undeclared suffix inserted before a recomputed CRC
    "reserved_flags",  # (i) reserved header flags are nonzero
    "reserved_header",  # (i) reserved header padding is nonzero
    "reserved_stride",  # (i) reserved segment stride_k is nonzero
    "invalid_utf8",  # (j) source_plan contains a malformed UTF-8 byte
    "duplicate_segment",  # (k) the same claim would execute more than once
    "duplicate_prefetch",  # (k) one name ambiguously binds two prefetch records
    "duplicate_trace",  # (k) one claim ambiguously binds two trace records
)


def _refix_crc(data: bytes) -> bytes:
    """Recompute and overwrite the CRC trailer so a byte-mutated pack still validates."""
    body = data[:-4]
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def _seg0_lane_width_offsets(data: bytes) -> tuple[int, int, int]:
    """Absolute offsets of seg0's lane, width, and reserved stride_k fields."""
    pos = _HEADER_SIZE
    (slen,) = struct.unpack_from("<H", data, pos)
    pos += 2 + slen  # source_plan
    (nlen,) = struct.unpack_from("<H", data, pos)
    pos += 2 + nlen  # seg0.name
    pos += 8 + 4  # claim_id, phase_id
    lane_off = pos
    pos += 1  # lane u8
    width_off = pos
    pos += 4  # width u32
    stride_off = pos  # stride_k u32
    return lane_off, width_off, stride_off


def base_pack() -> StreamPack:
    """A small, valid pack with one segment that declares a prefetch + a trace note."""
    p = StreamPack(source_plan="plan0", topo_gen=1, map_gen=BASE_MAP_GEN, data_gen=BASE_DATA_GEN)
    p.prefetches.append(Prefetch("pf0", 4, (10, 11)))
    p.segments.append(
        LaneSegment(
            name="seg0",
            claim_id=1000,
            phase_id=0,
            lane=Lane.U,
            width=4,
            opcode="f32.add",
            reads=(10, 11),
            writes=(12,),
            prefetch="pf0",
            dispatch="core",
            channel="host",
        )
    )
    p.trace_notes.append(TraceNote(claim_id=1000))
    return p


def _replace_seg0(p: StreamPack, **kw) -> None:
    p.segments[0] = LaneSegment(**{**p.segments[0].__dict__, **kw})


def corrupt(kind: str) -> bytes:
    """Return CRC-VALID, semantically-corrupt StreamPack bytes for `kind`."""
    p = base_pack()
    if kind == "dangling_claim":
        _replace_seg0(p, claim_id=9999)  # no trace note for 9999
        return encode(p)
    if kind == "swapped_rid":
        p.prefetches[0] = Prefetch("pf0", 4, (77777, 88888))  # targets feed no read
        return encode(p)
    if kind == "unresolved_prefetch":
        _replace_seg0(p, prefetch="ghost")  # no prefetch named "ghost"
        return encode(p)
    if kind == "stale_generation":
        p.map_gen = 99
        p.data_gen = 99  # != live (7, 19)
        return encode(p)
    if kind == "invalid_pipeline":
        p.pipeline_depth = 2
        p.prefetches[0] = Prefetch("pf0", 4, (10, 11), buffers=2)
        data = bytearray(encode(p))
        struct.pack_into("<H", data, 36, 0)
        return _refix_crc(bytes(data))
    if kind == "invalid_buffers":
        p.pipeline_depth = 2
        p.prefetches[0] = Prefetch("pf0", 4, (10, 11), buffers=2)
        data = bytearray(encode(p))
        # The pack has no blocks and its one prefetch precedes one fixed-size trace;
        # the prefetch's v2 tail is therefore 24 trace bytes before the CRC.
        data[-4 - 24 - 1] = 3
        return _refix_crc(bytes(data))
    if kind == "out_of_range_lane":
        data = bytearray(encode(p))
        lane_off, _, _ = _seg0_lane_width_offsets(bytes(data))
        data[lane_off] = 9  # only lanes 0..5 are valid
        return _refix_crc(bytes(data))
    if kind == "out_of_range_width":
        data = bytearray(encode(p))
        _, width_off, _ = _seg0_lane_width_offsets(bytes(data))
        struct.pack_into("<I", data, width_off, 3)  # 3 is not a power of two
        return _refix_crc(bytes(data))
    if kind == "tampered_dispatch":
        # encode a legal v3 pack (pim/nvidia_ptx), then flip the on-wire dispatch byte to an
        # illegal code. Pre-v3 this field was OFF the wire (outside the CRC) -- now a tamper is
        # both visible to byte-identity AND range-rejected.
        _replace_seg0(p, dispatch="pim", channel="nvidia_ptx", opcode="reduce.add")
        data = bytearray(encode(p))
        needle = struct.pack("<H", len("nvidia_ptx")) + b"nvidia_ptx"
        idx = bytes(data).find(needle)
        if idx <= 0:
            raise SystemExit("tampered_dispatch: channel marker not found")
        data[idx - 1] = 7  # illegal dispatch (0=core, 1=pim)
        return _refix_crc(bytes(data))
    if kind == "trailing_bytes":
        data = encode(p)
        return _refix_crc(data[:-4] + b"\x00" + data[-4:])
    if kind == "reserved_flags":
        data = bytearray(encode(p))
        struct.pack_into("<H", data, 6, 1)
        return _refix_crc(bytes(data))
    if kind == "reserved_header":
        data = bytearray(encode(p))
        data[63] = 1
        return _refix_crc(bytes(data))
    if kind == "reserved_stride":
        data = bytearray(encode(p))
        _, _, stride_off = _seg0_lane_width_offsets(bytes(data))
        struct.pack_into("<I", data, stride_off, 1)
        return _refix_crc(bytes(data))
    if kind == "invalid_utf8":
        data = bytearray(encode(p))
        plan_len = struct.unpack_from("<H", data, _HEADER_SIZE)[0]
        if not plan_len:
            raise SystemExit("invalid_utf8: source plan is unexpectedly empty")
        data[_HEADER_SIZE + 2] = 0xFF
        return _refix_crc(bytes(data))
    if kind == "duplicate_segment":
        p.segments.append(p.segments[0])
        return encode(p)
    if kind == "duplicate_prefetch":
        p.prefetches.append(p.prefetches[0])
        return encode(p)
    if kind == "duplicate_trace":
        p.trace_notes.append(p.trace_notes[0])
        return encode(p)
    raise SystemExit(f"unknown corruption kind {kind!r} (choose from {', '.join(KINDS)})")


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.stderr.write(f"usage: {argv[0]} <kind> <out.bin>\nkinds: {', '.join(KINDS)}\n")
        return 2
    kind, out = argv[1], argv[2]
    data = corrupt(kind)
    body, crc = data[:-4], struct.unpack("<I", data[-4:])[0]
    assert (zlib.crc32(body) & 0xFFFFFFFF) == crc, "internal: CRC was not fixed"
    with open(out, "wb") as fh:
        fh.write(data)
    # report the expected (live) generation the C rail should be told for the staleness check.
    print(f"{kind} {len(data)} {BASE_MAP_GEN} {BASE_DATA_GEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
