"""Frozen StreamPack binary ABI (Phase 7) tests."""

from bcir.abi import ABI_MAGIC, ABI_VERSION, AbiError, decode, encode
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
