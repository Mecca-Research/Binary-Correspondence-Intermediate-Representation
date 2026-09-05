"""The C StreamPack encoder (runtime/c/bcir_encode.c): byte-identical write-side parity.

Roadmap (BCIR_MASTER_ROADMAP.md §5.2 / §6): port the StreamPack *encoder* to C so a
driver-resident hydrate can emit the artifact with no Python -- completing the full C
round-trip (decoder + executor already landed). `bcir_sp_reencode` parses a pack and
re-serializes it through the C writer; these tests pin that the re-encoded bytes are
**byte-identical** to `bcir.abi.encode` across the corpus and all ABI versions (v1, v2
with the pipeline/double-buffer tails, v3 dispatch/channel, and v4 with the per-resource
generation vector every hydrated pack carries), plus malformed-input / NOSPACE paths.
"""

from dataclasses import replace
import os
import shutil
import struct
import subprocess
import tempfile
import zlib

from bcir.abi import encode
from bcir.examples import (
    gather_reduce,
    histogram_gather,
    matmul_tiled,
    multi_histogram,
    saxpy_strided,
    scan,
    vector_add,
)
from bcir.gem import hydrate
from bcir.gem.streampack import hydrate_pipelined
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")
AVX = TargetProfile.x86_avx512()
COOL = Theta.cool()


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _build(tmp):
    cc = _cc()
    if cc is None:
        return None
    exe = os.path.join(tmp, "test_encode")
    r = subprocess.run(
        [
            cc,
            "-std=c23",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            _RUNTIME_C,
            os.path.join(_RUNTIME_C, "bcir_encode.c"),
            os.path.join(_RUNTIME_C, "bcir_runtime.c"),
            os.path.join(_RUNTIME_C, "test_encode.c"),
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return exe


def _reencode(exe, tmp, blob: bytes) -> bytes:
    ip, op = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
    with open(ip, "wb") as f:
        f.write(blob)
    if os.path.exists(op):
        os.unlink(op)
    r = subprocess.run([exe, ip, op], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with open(op, "rb") as f:
        return f.read()


def test_c_encoder_builds_freestanding():
    cc = _cc()
    if cc is None:
        return
    for std in ("c11", "c23"):
        r = subprocess.run(
            [
                cc,
                "-ffreestanding",
                "-nostdlib",
                f"-std={std}",
                "-Wall",
                "-Wextra",
                "-I",
                _RUNTIME_C,
                "-c",
                os.path.join(_RUNTIME_C, "bcir_encode.c"),
                "-o",
                os.devnull,
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{std}: {r.stderr}"


def test_reencode_is_byte_identical_across_corpus_v1_to_v4():
    """The headline: C re-encode == bcir.abi.encode, byte-for-byte, on every corpus pack
    in v4 (hydrated: the per-resource generation vector, alone and with the pipeline
    tails), and on their vector-less twins in v1 (frozen) and v2 (pipeline/double-buffer
    tails), plus v3 (dispatch/channel) with and without the vector."""
    if _cc() is None:
        return
    mods = {
        "vector_add": vector_add(),
        "multi_histogram": multi_histogram(),
        "matmul_tiled": matmul_tiled(),
        "scan": scan(),
        "histogram_gather": histogram_gather(),
        "saxpy_strided": saxpy_strided(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        assert exe is not None
        for name, m in mods.items():
            r = optimize(m, AVX, COOL)
            v4, v4p = hydrate(m, r), hydrate_pipelined(m, r, depth=2)
            v1, v2 = hydrate(m, r), hydrate_pipelined(m, r, depth=2)
            v1.generations, v2.generations = [], []  # the vector-less twins
            for label, pack, version in (
                ("v4", v4, 4),
                ("v4-pipelined", v4p, 4),
                ("v1", v1, 1),
                ("v2", v2, 2),
            ):
                blob = encode(pack)
                assert blob[4] == version, f"{name} {label}: wire version {blob[4]}"
                assert _reencode(exe, tmp, blob) == blob, f"{name} {label}: not byte-identical"
        m = gather_reduce()
        pack = hydrate(m, optimize(m, AVX, COOL))
        pack.segments[0] = replace(
            pack.segments[0], opcode="reduce.add", dispatch="pim", channel="hbm_pim"
        )
        blob = encode(pack)
        assert blob[4] == 4  # dispatch/channel AND the vector: v4 implies the v3 tail
        assert _reencode(exe, tmp, blob) == blob
        pack.generations = []
        blob = encode(pack)
        assert blob[4] == 3
        assert _reencode(exe, tmp, blob) == blob


def test_malformed_input_is_rejected_not_crashed():
    if _cc() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        good = encode(hydrate(vector_add(), optimize(vector_add(), AVX, COOL)))
        trailing_body = good[:-4] + b"\x00"
        trailing = trailing_body + struct.pack("<I", zlib.crc32(trailing_body) & 0xFFFFFFFF)
        for bad in (b"", b"BSPK", good[:50], good[:-1], bytes(len(good)), trailing):
            ip = os.path.join(tmp, "bad.bin")
            with open(ip, "wb") as f:
                f.write(bad)
            r = subprocess.run(
                [exe, ip, os.path.join(tmp, "o.bin")], capture_output=True, text=True
            )
            assert r.returncode == 1, (bad[:8], r.returncode, r.stdout, r.stderr)
            assert "ERR" in r.stdout


def test_round_trip_decode_matches():
    """Belt-and-suspenders: the re-encoded bytes also decode back to an equal pack on the
    Python side (the C writer produced a semantically faithful artifact, not just equal
    bytes)."""
    if _cc() is None:
        return
    from bcir.abi import decode

    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        m = multi_histogram()
        pack = hydrate(m, optimize(m, AVX, COOL))
        blob = encode(pack)
        reenc = _reencode(exe, tmp, blob)
        assert decode(reenc).segments == pack.segments
