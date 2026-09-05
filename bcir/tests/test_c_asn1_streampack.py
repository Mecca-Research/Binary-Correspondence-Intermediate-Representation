"""The DER → native StreamPack fast path (roadmap phase D), on the C rail.

`runtime/c/bcir_asn1_streampack.c` reconstructs a native StreamPack artifact directly
from its X.690 DER projection, with no Python anywhere in the reconstruction path — the
direction a driver needs when a peer hands it a projection rather than a native pack.

The law it must satisfy is **byte identity**, not equivalence:

    bcir_asn1_to_streampack(encode_pack(P))  ==  encode(P)

Equivalence would be far weaker and much easier to pass. Byte identity forces the C rail
to re-derive the three things the projection does not carry: the native StreamPack
VERSION (v1/v2/v3 is a function of content, and the module's own `version` field is the
*projection* version, deliberately independent), the reserved `stride_k` the projection
omits by design, and the CRC. Getting any of them wrong changes the digest of an
artifact that is supposed to be frozen.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from bcir.abi import decode, encode
from bcir.asn1.streampack import encode_pack
from bcir.examples import PROGRAMS, vector_add
from bcir.gem import hydrate
from bcir.gem.streampack import LaneSegment, Prefetch, StreamPack, TraceNote
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Lane

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")
_SOURCES = ("bcir_asn1_streampack.c", "bcir_asn1.c", "bcir_runtime.c", "test_asn1_streampack.c")


def _compiler() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _build(tmp: str) -> str | None:
    cc = _compiler()
    if cc is None:
        return None
    exe = os.path.join(tmp, "test_asn1_sp")
    build = subprocess.run(
        [cc, "-std=c23", "-O2", "-Wall", "-Wextra", "-Werror", "-I", _RUNTIME_C]
        + [os.path.join(_RUNTIME_C, name) for name in _SOURCES]
        + ["-o", exe],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    return exe


def _run(exe: str, tmp: str, der: bytes, expected: bytes | None) -> tuple[int, str]:
    der_path = os.path.join(tmp, "in.der")
    with open(der_path, "wb") as handle:
        handle.write(der)
    argv = [exe, der_path]
    if expected is not None:
        want_path = os.path.join(tmp, "want.bin")
        with open(want_path, "wb") as handle:
            handle.write(expected)
        argv.append(want_path)
    result = subprocess.run(argv, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr).strip()


def _corpus_packs():
    host, theta = TargetProfile.x86_avx512(), Theta.cool()
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        yield name, hydrate(module, optimize(module, host, theta))


def test_fast_path_is_freestanding_under_c11_and_c23():
    """It is meant for a driver: no libc, no allocation, no recursion."""
    cc = _compiler()
    if cc is None:
        return
    for std in ("c11", "c23"):
        result = subprocess.run(
            [
                cc,
                "-ffreestanding",
                "-nostdlib",
                f"-std={std}",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                _RUNTIME_C,
                "-c",
                os.path.join(_RUNTIME_C, "bcir_asn1_streampack.c"),
                "-o",
                os.devnull,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"-std={std}: {result.stderr}"


def test_der_projection_reconstructs_byte_identically_for_every_corpus_program():
    """THE phase D law: A3 (additive round trip) proven with no Python in the path."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        checked = 0
        for name, pack in _corpus_packs():
            code, out = _run(exe, tmp, encode_pack(pack), encode(pack))
            assert code == 0, f"{name}: {out}"
            checked += 1
        assert checked >= 10, f"corpus shrank to {checked} programs"


def test_the_reconstruction_picks_the_same_version_the_native_encoder_would():
    """v1/v2/v3/v4 is derived from CONTENT, and the projection does not carry it.

    The module's `version` field is the PROJECTION version and is independent of the
    native one by design, so the C rail has to re-apply the same rule
    `bcir/abi::encode` uses: v2 when anything is pipelined or double-buffered, v3 when
    any segment carries a non-default dispatch or channel, v4 when the pack carries a
    per-resource generation vector (which implies the v2/v3 tails). A rail that guessed
    would still produce a decodable pack -- just not the same octets, and so not the
    same digest.
    """
    from bcir.gem.streampack import Generation

    def build(pipeline_depth=1, buffers=1, dispatch="core", channel="host", generations=False):
        pack = StreamPack(source_plan="plan0", topo_gen=1, map_gen=7, data_gen=19)
        pack.pipeline_depth = pipeline_depth
        if generations:  # maxima (7, 19) == the header tags above
            pack.generations = [Generation(10, 7, 2), Generation(11, 1, 19), Generation(12, 0, 0)]
        pack.prefetches.append(Prefetch("pf0", 4, (10, 11), "T0", "linear", buffers=buffers))
        pack.segments.append(
            LaneSegment(
                name="seg0",
                claim_id=1000,
                phase_id=0,
                lane=Lane.GGG,
                width=16,
                opcode="f32.add",
                reads=(10, 11),
                writes=(12,),
                prefetch="pf0",
                dispatch=dispatch,
                channel=channel,
            )
        )
        pack.trace_notes.append(TraceNote(claim_id=1000))
        return pack

    cases = {
        "v1": build(),
        "v2-pipeline": build(pipeline_depth=2),
        "v2-buffers": build(buffers=2),
        "v3-dispatch": build(dispatch="pim"),
        "v3-channel": build(channel="nvidia_ptx"),
        "v4-generations": build(generations=True),
        "v4-generations-pipelined-pim": build(pipeline_depth=2, dispatch="pim", generations=True),
    }
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        seen = {}
        for label, pack in cases.items():
            native = encode(pack)
            code, out = _run(exe, tmp, encode_pack(pack), native)
            assert code == 0, f"{label}: {out}"
            # Header layout "<4sHH...": the version is the u16 at offset 4. Byte identity
            # was already asserted above, so reading it here names WHICH version the two
            # rails agreed on -- the point being that all three are actually exercised.
            seen[label] = int.from_bytes(native[4:6], "little")
        assert seen == {
            "v1": 1,
            "v2-pipeline": 2,
            "v2-buffers": 2,
            "v3-dispatch": 3,
            "v3-channel": 3,
            "v4-generations": 4,
            "v4-generations-pipelined-pim": 4,
        }, seen


def test_a_malformed_or_ber_only_projection_is_refused_not_partially_reconstructed():
    """The fast path is a trust boundary: it must never emit a partial artifact.

    BER-only spellings are refused rather than normalized. Normalizing would change the
    octets a peer chose, and BCIR digests what it exchanges -- so accepting a
    non-minimal length here would let a peer pick the digest by picking a spelling.
    """
    pack = hydrate(
        vector_add(1024), optimize(vector_add(1024), TargetProfile.x86_avx512(), Theta.cool())
    )
    der = encode_pack(pack)
    bad: list[bytes] = [der[:n] for n in (0, 1, 5, len(der) // 2, len(der) - 1)]
    for index in (0, 1, 3, 10):
        mutated = bytearray(der)
        mutated[index] ^= 0xFF
        bad.append(bytes(mutated))
    if der[1] < 0x80:  # the same value with a non-minimal length: legal BER, not DER
        bad.append(bytes([der[0], 0x81, der[1]]) + der[2:])
    bad.append(der + b"\x00")  # trailing garbage

    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        for index, raw in enumerate(bad):
            code, out = _run(exe, tmp, raw, None)
            assert code != 0, f"malformed input {index} was accepted: {out}"
            assert "AddressSanitizer" not in out and "runtime error" not in out, out


def test_a_reconstructed_pack_decodes_back_to_the_same_value_on_the_python_rail():
    """Closing the loop: C reconstructs, Python decodes, and the plan is unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        if exe is None:
            return
        for name, pack in _corpus_packs():
            native = encode(pack)
            code, out = _run(exe, tmp, encode_pack(pack), native)
            assert code == 0, f"{name}: {out}"
            # The harness proved byte identity; decoding the same octets on the Python
            # rail proves the artifact is a PLAN, not just a matching byte string.
            round_tripped = decode(native)
            assert round_tripped.source_plan == pack.source_plan
            assert len(round_tripped.segments) == len(pack.segments)
