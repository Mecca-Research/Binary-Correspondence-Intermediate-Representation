"""Frozen StreamPack binary ABI (Phase 7) tests."""

import struct
import zlib
from dataclasses import replace

from bcir.abi import ABI_MAGIC, ABI_VERSION, AbiError, decode, encode
from bcir.abi.streampack_abi import ABI_VERSION_MAX
from bcir.examples import vector_add
from bcir.gem import generation_vector, hydrate
from bcir.gem.streampack import Generation, LaneSegment, Prefetch, StreamPack, TraceNote
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Lane


def _pack():
    m = vector_add(1024)
    return hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))


def _refix(blob: bytearray) -> bytes:
    body = bytes(blob[:-4])
    blob[-4:] = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    return bytes(blob)


def _v1_pack():
    """A hand-built pack using no v2/v3/v4 feature: the frozen v1 encoding."""
    pack = StreamPack(source_plan="plan0", topo_gen=1, map_gen=7, data_gen=19)
    pack.prefetches.append(Prefetch("pf0", 4, (10, 11), "T0", "linear"))
    pack.segments.append(
        LaneSegment(
            name="seg0",
            claim_id=1000,
            phase_id=0,
            lane=Lane.UX,
            width=16,
            opcode="f32.add",
            reads=(10, 11),
            writes=(12,),
            prefetch="pf0",
            fence_before=(),
            fence_after=("f0",),
        )
    )
    pack.trace_notes.append(TraceNote(claim_id=1000, src_hash=42, trace_hash=99))
    return pack


def test_header_magic_and_version():
    blob = encode(_pack())
    assert blob[:4] == ABI_MAGIC == b"BSPK"
    assert ABI_VERSION == 1  # the frozen base version
    assert blob[4] == 4  # a hydrated pack carries the v4 generation vector
    assert encode(_v1_pack())[4] == ABI_VERSION  # a vector-less pack stays frozen v1
    assert len(blob) >= 64 + 4  # header + crc trailer


def test_round_trip_is_lossless():
    pack = _pack()
    assert decode(encode(pack)) == pack


def test_encoding_is_deterministic():
    pack = _pack()
    assert encode(pack) == encode(pack)  # stable bytes
    assert encode(decode(encode(pack))) == encode(pack)  # idempotent


def test_rich_pack_round_trips():
    pack = _v1_pack()
    assert decode(encode(pack)) == pack


def test_v2_features_round_trip_append_only():
    # A pack carrying pipeline/double-buffer contracts encodes as v2 and
    # round-trips losslessly; the v1 layout is untouched (append-only).
    pack = _v1_pack()
    pack.pipeline_depth = 2
    pack.prefetches[0] = replace(pack.prefetches[0], buffers=2)
    blob = encode(pack)
    assert blob[4] == 2  # v2 on the wire
    assert decode(blob) == pack
    assert decode(blob).pipeline_depth == 2
    # A hydrated pipelined pack carries the pipeline AND the generation vector (v4 implies
    # the v2/v3 tails): the depth still round-trips through the v4 header.
    m = vector_add(1024)
    from bcir.gem import hydrate_pipelined

    hydrated = hydrate_pipelined(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()), depth=2)
    blob = encode(hydrated)
    assert blob[4] == 4
    assert decode(blob) == hydrated
    assert decode(blob).pipeline_depth == 2


def test_packs_without_v2_features_stay_byte_frozen_v1():
    # The frozen-v1 promise: a pack with no v2 contracts is byte-identical v1.
    pack = _v1_pack()
    assert pack.pipeline_depth == 1 and all(pf.buffers == 1 for pf in pack.prefetches)
    blob = encode(pack)
    assert blob[4] == ABI_VERSION == 1
    assert decode(blob).pipeline_depth == 1


def test_double_buffer_prefetch_round_trips():
    pack = StreamPack(source_plan="plan0", pipeline_depth=2)
    pack.prefetches.append(Prefetch("dbpf0_1", 4, (10, 11), "T1", "double_buffer", buffers=2))
    out = decode(encode(pack))
    assert out.prefetches[0].buffers == 2
    assert out.pipeline_depth == 2


def _v3_pack():
    """A pack that USES v3 segment dispatch/channel (so it encodes as v3)."""
    pack = StreamPack(source_plan="plan0", topo_gen=1, map_gen=7, data_gen=19)
    pack.prefetches.append(Prefetch("pf0", 4, (10, 11)))
    pack.segments.append(
        LaneSegment(
            name="seg0",
            claim_id=1000,
            phase_id=0,
            lane=Lane.GGG,
            width=1,
            opcode="reduce.add",
            reads=(10, 11),
            writes=(12,),
            prefetch="pf0",
            dispatch="pim",
            channel="nvidia_ptx",
        )
    )
    pack.trace_notes.append(TraceNote(claim_id=1000))
    return pack


def test_abi_version_max_is_four():
    # v4 (append-only: the per-resource generation vector) raised the maximum the reader
    # handles to 4; the frozen base stays 1.
    assert ABI_VERSION_MAX == 4
    assert ABI_VERSION == 1


def test_v3_dispatch_channel_round_trips_append_only():
    # A pack carrying a non-default dispatch/channel encodes as v3 and round-trips
    # losslessly; decode(encode(x)) == x and encode is idempotent (byte-identity).
    pack = _v3_pack()
    blob = encode(pack)
    assert blob[4] == 3  # v3 on the wire
    out = decode(blob)
    assert out == pack  # decode(encode(x)) == x
    assert out.segments[0].dispatch == "pim"
    assert out.segments[0].channel == "nvidia_ptx"
    assert encode(decode(blob)) == blob  # encode(decode(encode(x))) == encode(x)


def test_packs_without_v3_features_stay_byte_frozen_v1_v2():
    # The append-only promise extends to v3: a pack with every segment at the
    # dispatch/channel defaults ("core"/"host") is byte-identical to its v1/v2 encoding.
    v1 = _v1_pack()
    assert all(s.dispatch == "core" and s.channel == "host" for s in v1.segments)
    assert encode(v1)[4] == ABI_VERSION == 1  # still v1, no v3 tail
    # a v2-feature pack with default dispatch/channel stays v2 (not promoted to v3).
    v2 = _v1_pack()
    v2.pipeline_depth = 2
    assert all(s.dispatch == "core" and s.channel == "host" for s in v2.segments)
    assert encode(v2)[4] == 2
    # ... and a v3 pack without a generation vector stays v3 (not promoted to v4).
    assert not _v3_pack().generations
    assert encode(_v3_pack())[4] == 3


def _v4_pack():
    """A hand-built pack that carries a per-resource generation vector (so it encodes as
    v4) over an otherwise-v1 body: two resources at different generations, header maxima."""
    pack = _v1_pack()
    pack.map_gen, pack.data_gen = 7, 19
    pack.generations = [Generation(10, 7, 2), Generation(11, 1, 19), Generation(12, 0, 0)]
    return pack


def test_hydrated_packs_carry_the_generation_vector_as_v4():
    # S0-2 (R11 per resource): hydrate emits one (rid, map_gen, data_gen) per declared
    # resource in RID order, the header maxima are the vector's maxima, and the pack is v4.
    m = vector_add(1024)
    m.resources[10] = replace(m.resources[10], map_gen=3, data_gen=2)
    m.resources[11] = replace(m.resources[11], map_gen=1, data_gen=0)
    pack = hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))
    assert pack.generations == generation_vector(m)
    assert [(g.rid, g.map_gen, g.data_gen) for g in pack.generations] == [
        (10, 3, 2),
        (11, 1, 0),
        (12, 0, 0),
    ]
    assert (pack.map_gen, pack.data_gen) == (3, 2)
    blob = encode(pack)
    assert blob[4] == 4
    assert struct.unpack_from("<I", blob, 40) == (3,)  # n_gens carved from the pad
    assert not any(blob[38:40]) and not any(blob[44:64])  # the pad around it stays reserved
    out = decode(blob)
    assert out == pack
    assert encode(out) == blob


def test_v4_generation_vector_round_trips_append_only():
    # decode(encode(x)) == x, encode is idempotent, and the v4 record is a pure suffix:
    # the vector-less twin's bytes are a prefix-with-header-diff of the v4 encoding, and
    # the vector costs exactly 12 bytes per resource (the wire size of one record).
    pack = _v4_pack()
    blob = encode(pack)
    assert blob[4] == 4
    out = decode(blob)
    assert out == pack
    assert encode(out) == blob
    twin = _v4_pack()
    twin.generations = []
    twin_blob = encode(twin)
    assert twin_blob[4] == ABI_VERSION == 1
    # v4 implies the v2/v3 tails (one `buffers` byte per prefetch; dispatch + channel per
    # segment), then the vector costs exactly 12 bytes per resource as a pure suffix.
    tails = len(pack.prefetches) + sum(1 + 2 + len(s.channel) for s in pack.segments)
    assert len(blob) == len(twin_blob) + tails + 12 * len(pack.generations)
    suffix = b"".join(struct.pack("<III", g.rid, g.map_gen, g.data_gen) for g in pack.generations)
    assert blob[-4 - len(suffix) : -4] == suffix
    # the header differs only in the version and n_gens fields (pipeline_depth is 1 either way).
    assert blob[:4] == twin_blob[:4] and blob[6:36] == twin_blob[6:36]
    assert struct.unpack_from("<H", blob, 36) == (1,) and blob[44:64] == twin_blob[44:64]


def test_encoder_rejects_malformed_generation_vectors():
    """The writer never emits a vector a reader would refuse: RIDs strictly ascending, u32
    fields, and the header maxima equal to the vector's (one source of truth)."""
    unsorted = _v4_pack()
    unsorted.generations = [Generation(11, 1, 19), Generation(10, 7, 2)]
    duplicate = _v4_pack()
    duplicate.generations = [Generation(10, 7, 19), Generation(10, 7, 19)]
    overflow = _v4_pack()
    overflow.generations = [Generation(1 << 32, 7, 19)]
    header_map = _v4_pack()
    header_map.map_gen = 8  # not the vector's maximum
    header_data = _v4_pack()
    header_data.data_gen = 0
    for pack in (unsorted, duplicate, overflow, header_map, header_data):
        try:
            encode(pack)
            raise AssertionError("expected AbiError on a malformed generation vector")
        except AbiError:
            pass


def test_v4_decoder_rejects_malformed_generation_vectors():
    """CRC-valid v4 bytes outside the frozen contract fail the same predicate the encoder
    applies (rail symmetry with the C decoder's BCIR_ERR_GENERATION / TRUNCATED / RESERVED)."""
    from bcir.abi.streampack_abi import _GEN_SIZE

    base = encode(_v4_pack())
    n = len(_v4_pack().generations)
    vec_off = len(base) - 4 - _GEN_SIZE * n
    variants = []
    # RIDs out of order: swap the first two records.
    swapped = bytearray(base)
    first = base[vec_off : vec_off + _GEN_SIZE]
    second = base[vec_off + _GEN_SIZE : vec_off + 2 * _GEN_SIZE]
    swapped[vec_off : vec_off + 2 * _GEN_SIZE] = second + first
    variants.append(_refix(swapped))
    # a duplicate RID: the second record names the first's rid.
    duplicate = bytearray(base)
    struct.pack_into("<I", duplicate, vec_off + _GEN_SIZE, 10)
    variants.append(_refix(duplicate))
    # the header maxima disagree with the vector (two sources of truth).
    stale_header = bytearray(base)
    struct.pack_into("<I", stale_header, 16, 8)  # map_gen @16
    variants.append(_refix(stale_header))
    stale_data = bytearray(base)
    struct.pack_into("<I", stale_data, 20, 0)  # data_gen @20
    variants.append(_refix(stale_data))
    # n_gens promises one record more than the body carries (truncated vector).
    short = bytearray(base)
    struct.pack_into("<I", short, 40, n + 1)
    variants.append(_refix(short))
    # n_gens promises one record fewer: the last record becomes trailing bytes.
    long_ = bytearray(base)
    struct.pack_into("<I", long_, 40, n - 1)
    variants.append(_refix(long_))
    # a v3 header with nonzero bytes at 40..43: reserved until v4 says otherwise.
    v3 = bytearray(encode(_v3_pack()))
    assert v3[4] == 3
    v3[40] = 1
    variants.append(_refix(v3))
    # the bytes around n_gens (38..39, 44..63) stay reserved on v4 too.
    pad = bytearray(base)
    pad[38] = 1
    variants.append(_refix(pad))
    for blob in variants:
        try:
            decode(blob)
            raise AssertionError("expected a refusal of the malformed v4 record")
        except AbiError:
            pass
    assert decode(base) == _v4_pack()  # the unmodified base is clean


def test_v3_decode_rejects_unknown_dispatch_code():
    # The rail-symmetry guard: an on-wire dispatch byte outside {0=core,1=pim} RAISES on
    # decode (mirrors Lane(bad) raising), instead of silently defaulting.
    blob = bytearray(encode(_v3_pack()))
    needle = struct.pack("<H", len("nvidia_ptx")) + b"nvidia_ptx"
    idx = bytes(blob).find(needle)
    assert idx > 0
    blob[idx - 1] = 7  # illegal dispatch code
    body = bytes(blob[:-4])
    blob[-4:] = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)  # re-fix CRC
    try:
        decode(bytes(blob))
        assert False, "expected AbiError on an unknown dispatch code"
    except AbiError:
        pass


def test_encoder_rejects_unrepresentable_or_semantically_invalid_fields():
    """The writer never masks, wraps, or silently substitutes another contract."""
    bad = _v3_pack()
    bad.segments[0] = LaneSegment(**{**bad.segments[0].__dict__, "dispatch": "unknown-device"})
    cases = [bad, StreamPack(pipeline_depth=0), StreamPack(pipeline_depth=1 << 16)]
    unhashable_dispatch = _v3_pack()
    unhashable_dispatch.segments[0] = LaneSegment(
        **{**unhashable_dispatch.segments[0].__dict__, "dispatch": ["pim"]}
    )
    cases.append(unhashable_dispatch)
    invalid_buffers = StreamPack(pipeline_depth=2)
    invalid_buffers.prefetches.append(Prefetch("pf", 1, (), buffers=3))
    cases.append(invalid_buffers)
    overflow_gen = StreamPack(topo_gen=1 << 32)
    cases.append(overflow_gen)
    invalid_width = _v3_pack()
    invalid_width.segments[0] = LaneSegment(**{**invalid_width.segments[0].__dict__, "width": 3})
    cases.append(invalid_width)
    for pack in cases:
        try:
            encode(pack)
            assert False, "expected exact-representability rejection"
        except AbiError:
            pass


def test_decoder_rejects_zero_pipeline_depth_and_invalid_buffer_count():
    """CRC-valid v2 values outside the frozen semantic range fail both constraints."""
    pack = StreamPack(pipeline_depth=2)
    pack.prefetches.append(Prefetch("pf", 1, (), buffers=2))
    original = encode(pack)

    zero_depth = bytearray(original)
    struct.pack_into("<H", zero_depth, 36, 0)
    body = bytes(zero_depth[:-4])
    zero_depth[-4:] = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)

    bad_buffers = bytearray(original)
    bad_buffers[-5] = 3  # the only prefetch is the final body record; its v2 tail is last
    body = bytes(bad_buffers[:-4])
    bad_buffers[-4:] = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)

    for blob in (bytes(zero_depth), bytes(bad_buffers)):
        try:
            decode(blob)
            assert False, "expected v2 semantic-range rejection"
        except AbiError:
            pass


def test_v3_decode_rejects_out_of_range_lane():
    # A CRC-fixed out-of-range lane is rejected through the public ABI error contract
    # (the same semantic refusal enforced by the C rail).
    from bcir.abi.streampack_abi import _HEADER_SIZE

    blob = bytearray(encode(_pack()))
    pos = _HEADER_SIZE
    (slen,) = struct.unpack_from("<H", blob, pos)
    pos += 2 + slen  # source_plan
    (nlen,) = struct.unpack_from("<H", blob, pos)
    pos += 2 + nlen  # seg0.name
    pos += 8 + 4  # claim_id, phase_id
    blob[pos] = 9  # lane (only 0..5 valid)
    body = bytes(blob[:-4])
    blob[-4:] = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    try:
        decode(bytes(blob))
        assert False, "expected a raise on an out-of-range lane"
    except AbiError:
        pass


def test_decode_rejects_bad_magic_version_and_crc():
    blob = bytearray(encode(_pack()))
    bad_magic = bytes(b"XXXX" + blob[4:])
    try:
        decode(bad_magic)
        assert False, "expected AbiError"
    except AbiError:
        pass
    bad_ver = bytearray(blob)
    bad_ver[4] = 9
    try:
        decode(bytes(bad_ver))
        assert False
    except AbiError:
        pass
    corrupt = bytearray(blob)
    corrupt[80] ^= 0xFF  # flip a body byte -> CRC fails
    try:
        decode(bytes(corrupt))
        assert False
    except AbiError:
        pass


def test_decoder_rejects_crc_valid_reserved_fields_and_invalid_utf8():
    """Reserved bits/bytes and stride_k are not extension points until a version says
    so; accepting them creates byte-distinct packs with parser-dependent meaning."""
    from bcir.abi.streampack_abi import _HEADER_SIZE

    def refix(blob: bytearray) -> bytes:
        body = bytes(blob[:-4])
        blob[-4:] = struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
        return bytes(blob)

    base = bytearray(encode(_pack()))
    variants = []
    flags = bytearray(base)
    struct.pack_into("<H", flags, 6, 1)
    variants.append(refix(flags))
    padding = bytearray(base)
    padding[63] = 1
    variants.append(refix(padding))

    stride = bytearray(base)
    pos = _HEADER_SIZE
    (n,) = struct.unpack_from("<H", stride, pos)
    pos += 2 + n  # source plan
    (n,) = struct.unpack_from("<H", stride, pos)
    pos += 2 + n  # segment name
    pos += 8 + 4 + 1 + 4  # id, phase, lane, width
    struct.pack_into("<I", stride, pos, 1)
    variants.append(refix(stride))

    utf8 = bytearray(base)
    plan_len = struct.unpack_from("<H", utf8, _HEADER_SIZE)[0]
    assert plan_len
    utf8[_HEADER_SIZE + 2] = 0xFF
    variants.append(refix(utf8))

    for blob in variants:
        try:
            decode(blob)
            raise AssertionError("expected reserved/UTF-8 refusal")
        except AbiError:
            pass
