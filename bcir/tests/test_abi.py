"""Frozen StreamPack binary ABI (Phase 7) tests."""

from bcir.abi import ABI_MAGIC, ABI_VERSION, AbiError, decode, encode
from bcir.abi.streampack_abi import ABI_VERSION_MAX
from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.gem.streampack import LaneSegment, Prefetch, StreamPack, TraceNote
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Lane


def _pack():
    m = vector_add(1024)
    return hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))


def test_header_magic_and_version():
    blob = encode(_pack())
    assert blob[:4] == ABI_MAGIC == b"BSPK"
    assert blob[4] == ABI_VERSION == 1
    assert len(blob) >= 64 + 4  # header + crc trailer


def test_round_trip_is_lossless():
    pack = _pack()
    assert decode(encode(pack)) == pack


def test_encoding_is_deterministic():
    pack = _pack()
    assert encode(pack) == encode(pack)        # stable bytes
    assert encode(decode(encode(pack))) == encode(pack)  # idempotent


def test_rich_pack_round_trips():
    pack = StreamPack(source_plan="plan0", topo_gen=1, map_gen=7, data_gen=19)
    pack.prefetches.append(Prefetch("pf0", 4, (10, 11), "T0", "linear"))
    pack.segments.append(LaneSegment(
        name="seg0", claim_id=1000, phase_id=0, lane=Lane.UX, width=16,
        opcode="f32.add", reads=(10, 11), writes=(12,), prefetch="pf0",
        fence_before=(), fence_after=("f0",)))
    pack.trace_notes.append(TraceNote(claim_id=1000, src_hash=42, trace_hash=99))
    assert decode(encode(pack)) == pack


def test_v2_features_round_trip_append_only():
    # A pack carrying pipeline/double-buffer contracts encodes as v2 and
    # round-trips losslessly; the v1 layout is untouched (append-only).
    m = vector_add(1024)
    from bcir.gem import hydrate_pipelined
    pack = hydrate_pipelined(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()),
                             depth=2)
    blob = encode(pack)
    assert blob[4] == 2                       # v2 on the wire
    assert decode(blob) == pack
    assert decode(blob).pipeline_depth == 2


def test_packs_without_v2_features_stay_byte_frozen_v1():
    # The frozen-v1 promise: a pack with no v2 contracts is byte-identical v1.
    pack = _pack()
    assert pack.pipeline_depth == 1 and all(pf.buffers == 1 for pf in pack.prefetches)
    blob = encode(pack)
    assert blob[4] == ABI_VERSION == 1
    assert decode(blob).pipeline_depth == 1


def test_double_buffer_prefetch_round_trips():
    pack = StreamPack(source_plan="plan0", pipeline_depth=2)
    pack.prefetches.append(Prefetch("dbpf0_1", 4, (10, 11), "T1", "double_buffer",
                                    buffers=2))
    out = decode(encode(pack))
    assert out.prefetches[0].buffers == 2
    assert out.pipeline_depth == 2


def _v3_pack():
    """A pack that USES v3 segment dispatch/channel (so it encodes as v3)."""
    pack = StreamPack(source_plan="plan0", topo_gen=1, map_gen=7, data_gen=19)
    pack.prefetches.append(Prefetch("pf0", 4, (10, 11)))
    pack.segments.append(LaneSegment(
        name="seg0", claim_id=1000, phase_id=0, lane=Lane.GGG, width=1,
        opcode="reduce.add", reads=(10, 11), writes=(12,), prefetch="pf0",
        dispatch="pim", channel="nvidia_ptx"))
    pack.trace_notes.append(TraceNote(claim_id=1000))
    return pack


def test_abi_version_max_is_three():
    # v3 (append-only) raised the maximum the reader handles to 3.
    assert ABI_VERSION_MAX == 3


def test_v3_dispatch_channel_round_trips_append_only():
    # A pack carrying a non-default dispatch/channel encodes as v3 and round-trips
    # losslessly; decode(encode(x)) == x and encode is idempotent (byte-identity).
    pack = _v3_pack()
    blob = encode(pack)
    assert blob[4] == 3                                  # v3 on the wire
    out = decode(blob)
    assert out == pack                                   # decode(encode(x)) == x
    assert out.segments[0].dispatch == "pim"
    assert out.segments[0].channel == "nvidia_ptx"
    assert encode(decode(blob)) == blob                  # encode(decode(encode(x))) == encode(x)


def test_packs_without_v3_features_stay_byte_frozen_v1_v2():
    # The append-only promise extends to v3: a pack with every segment at the
    # dispatch/channel defaults ("core"/"host") is byte-identical to its v1/v2 encoding.
    v1 = _pack()
    assert all(s.dispatch == "core" and s.channel == "host" for s in v1.segments)
    assert encode(v1)[4] == ABI_VERSION == 1             # still v1, no v3 tail
    # a v2-feature pack with default dispatch/channel stays v2 (not promoted to v3).
    from bcir.gem import hydrate_pipelined
    m = vector_add(1024)
    v2 = hydrate_pipelined(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()), depth=2)
    assert all(s.dispatch == "core" and s.channel == "host" for s in v2.segments)
    assert encode(v2)[4] == 2


def test_v3_decode_rejects_unknown_dispatch_code():
    # The rail-symmetry guard: an on-wire dispatch byte outside {0=core,1=pim} RAISES on
    # decode (mirrors Lane(bad) raising), instead of silently defaulting.
    import struct
    import zlib
    blob = bytearray(encode(_v3_pack()))
    needle = struct.pack("<H", len("nvidia_ptx")) + b"nvidia_ptx"
    idx = bytes(blob).find(needle)
    assert idx > 0
    blob[idx - 1] = 7                                    # illegal dispatch code
    body = bytes(blob[:-4])
    blob[-4:] = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)   # re-fix CRC
    try:
        decode(bytes(blob))
        assert False, "expected AbiError on an unknown dispatch code"
    except AbiError:
        pass


def test_v3_decode_rejects_out_of_range_lane():
    # The Python decoder already raises on an out-of-range lane via Lane(r.u8()); confirm
    # a CRC-fixed out-of-range lane is rejected (the asymmetry the C rail now mirrors).
    import struct
    import zlib
    from bcir.abi.streampack_abi import _HEADER_SIZE
    blob = bytearray(encode(_pack()))
    pos = _HEADER_SIZE
    (slen,) = struct.unpack_from("<H", blob, pos); pos += 2 + slen      # source_plan
    (nlen,) = struct.unpack_from("<H", blob, pos); pos += 2 + nlen      # seg0.name
    pos += 8 + 4                                                        # claim_id, phase_id
    blob[pos] = 9                                                       # lane (only 0..5 valid)
    body = bytes(blob[:-4])
    blob[-4:] = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    try:
        decode(bytes(blob))
        assert False, "expected a raise on an out-of-range lane"
    except (AbiError, ValueError):
        pass


def test_decode_rejects_bad_magic_version_and_crc():
    blob = bytearray(encode(_pack()))
    bad_magic = bytes(b"XXXX" + blob[4:])
    try:
        decode(bad_magic); assert False, "expected AbiError"
    except AbiError:
        pass
    bad_ver = bytearray(blob); bad_ver[4] = 9
    try:
        decode(bytes(bad_ver)); assert False
    except AbiError:
        pass
    corrupt = bytearray(blob); corrupt[80] ^= 0xFF  # flip a body byte -> CRC fails
    try:
        decode(bytes(corrupt)); assert False
    except AbiError:
        pass
