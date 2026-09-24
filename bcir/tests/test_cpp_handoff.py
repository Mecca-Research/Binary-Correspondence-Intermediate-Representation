"""The C<->C++ hand-off seam (`runtime/cpp/`): the contract on real artifacts.

The seam is the boundary between BCIR's deterministic single-node C/IR rail and the C++ layer
ABOVE it (dynamic graph topology, distributed orchestration); the contract is
`docs/languages/CPP_HANDOFF_BOUNDARY.md`. Since G16 (S3-C) the artifact crosses as a borrowed
view over the freestanding C pack table, admitted against the LIVE control plane. This module
runs the seam's own harness (`runtime/cpp/test_orchestrator.cpp`) over artifacts the C/IR path
produces and plane records the Python issuer mints (`handoff_fixtures.seam_artifacts`, the one
minter `tools/cpp/check_handoff.sh` shares): the single-node dispatch equals the direct C walk,
shards run by themselves and reassemble, a builder step freezes and runs, a switch makes the
pack stale, a dead view is refused -- and a corrupted artifact is refused at admission while the
same probe admits the clean one. The G16 rows proper are `test_handoff.py`.

The build is `handoff_fixtures.build_cpp_program`, the source list every C++ harness shares, so
this module cannot drift from the gates (it did once: it kept the pre-G16 list and API). The
quick tier still checks the artifact the seam consumes is producible and decodable; the C++
build runs where both compilers are visible (c-runtime / thorough).
"""

import os
import tempfile

from bcir.abi import decode
from bcir.tests import handoff_fixtures as hf

_EXAMPLES = ("multi_histogram", "vector_add")


def test_artifact_the_seam_consumes_is_producible_and_decodable():
    """Quick-tier coverage (no compiler needed): the StreamPack the C++ seam consumes is produced
    by the existing C/IR path and decodes on the Python rail, and its plane records are the
    three the harness boots from."""
    for example in _EXAMPLES:
        pack, plane = hf.seam_artifacts(example)
        assert pack[:4] == b"BSPK", "the hand-off artifact must be a StreamPack"
        assert decode(pack).segments, "a non-trivial artifact has segments to dispatch"
        n, count = 0, 0
        while n < len(plane):
            n += 4 + int.from_bytes(plane[n : n + 4], "little")
            count += 1
        assert n == len(plane) and count == 3, "a grant and two generation switches"


def _orchestrator(tmp: str) -> str | None:
    return hf.build_cpp_program(
        tmp, os.path.join(hf.CPP_DIR, "test_orchestrator.cpp"), "test_orchestrator"
    )


def _write(tmp: str, name: str, data: bytes) -> str:
    path = os.path.join(tmp, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def test_the_seam_round_trips_and_holds_the_g16_contract():
    """The REAL seam on two artifacts: admitted by the live plane, dispatch == the direct C walk,
    shards by themselves, a frozen builder step, a switch, a dead view."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = _orchestrator(tmp)
        if exe is None:  # quick tier: deferred to where both compilers are visible
            return
        for example in _EXAMPLES:
            pack, plane = hf.seam_artifacts(example)
            run = hf._run([exe, _write(tmp, "pack.bin", pack), _write(tmp, "plane.bin", plane)])
            assert run.returncode == 0, f"{example}: {run.stdout}{run.stderr}"
            assert run.stdout.startswith("OK "), f"unexpected seam output: {run.stdout!r}"


def test_a_corrupted_artifact_is_refused_at_admission_and_the_clean_one_admitted():
    """The two-truth quarantine across the seam: admission carries the plane's verdict (it does
    not re-derive legality), so a corrupted artifact is refused as malformed and never runs --
    and the same probe admits the clean pack, or its refusal would prove nothing (L2)."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = _orchestrator(tmp)
        if exe is None:
            return
        pack, plane = hf.seam_artifacts("multi_histogram")
        plane_path = _write(tmp, "plane.bin", plane)
        clean = hf._run([exe, "--reject", _write(tmp, "clean.bin", pack), plane_path])
        assert (clean.returncode, clean.stdout.strip()) == (1, "ADMITTED"), clean.stdout
        bad = bytearray(pack)
        bad[20] ^= 0xFF  # n_segments: the CRC no longer holds
        rejected = hf._run([exe, "--reject", _write(tmp, "bad.bin", bytes(bad)), plane_path])
        assert (rejected.returncode, rejected.stdout.strip()) == (0, "REJECTED"), rejected.stdout
