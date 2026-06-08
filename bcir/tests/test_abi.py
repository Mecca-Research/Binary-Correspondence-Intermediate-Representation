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
