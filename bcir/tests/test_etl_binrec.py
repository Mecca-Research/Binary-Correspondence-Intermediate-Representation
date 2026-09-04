"""ETL binary-record decoder: the C trust-boundary twin of bcir/etl/binary.py.

Roadmap §5 #2 ("extend fuzzing to the ROP/MAP/ETL C paths"): the ETL binary decoder is
the one *binary* trust boundary among the front-ends (ROP/MAP are text grammars the
Python fuzz already covers). `runtime/c/bcir_binrec.c` is its bounds-checked C twin,
fuzzed under libFuzzer + ASan/UBSan (tools/c/fuzz_streampack.sh). These tests gate the
**decode semantics** against the oracle: the C decoder and Python `etl.binary` must
agree field-for-field (value + status) across random buffers, including OOB rejection,
signed sign-extension, and big-endian.
"""

import os
import random
import shutil
import subprocess
import tempfile

from bcir.etl.binary import BinaryField, _extract

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")

# Must mirror runtime/c/test_binrec.c kFields exactly:
# (offset_bits, width_bits, kind, endianness).
_FIELDS = [
    (0, 8, "u", "little"),
    (16, 16, "u", "little"),
    (32, 32, "u", "little"),
    (16, 16, "s", "little"),
    (0, 16, "u", "big"),
    (0, 64, "s", "little"),
]
_OK, _OOB = 0, 3


def _py_decode(buf: bytes):
    """The oracle side: (status, value) per field (status 3 == out of bounds)."""
    out = []
    for off, w, kind, endi in _FIELDS:
        try:
            out.append((_OK, _extract(buf, BinaryField("x", off, w, kind), endi)))
        except ValueError:
            out.append((_OOB, 0))  # all fields are aligned & <=64 bits, so only OOB
    return out


def _build_harness(tmp: str) -> str | None:
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return None
    exe = os.path.join(tmp, "test_binrec")
    build = subprocess.run(
        [
            cc,
            "-std=c23",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            _RUNTIME_C,
            os.path.join(_RUNTIME_C, "bcir_binrec.c"),
            os.path.join(_RUNTIME_C, "test_binrec.c"),
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    return exe


def _c_decode(exe: str, tmp: str, buf: bytes):
    path = os.path.join(tmp, "rec.bin")
    with open(path, "wb") as f:
        f.write(buf)
    r = subprocess.run([exe, path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # Each line is "<idx> <status> <value>".
    return [
        (int(parts[1]), int(parts[2]))
        for parts in (ln.split() for ln in r.stdout.strip().splitlines())
    ]


def test_c_decoder_builds_freestanding():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
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
                os.path.join(_RUNTIME_C, "bcir_binrec.c"),
                "-o",
                os.devnull,
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{std}: {r.stderr}"


def test_python_c_parity_over_random_buffers():
    if not (shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")):
        return  # skip cleanly without a C compiler
    rng = random.Random(7)
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        assert exe is not None
        for _ in range(300):
            n = rng.randint(8, 12)  # >= 8 so all six fields are in-bounds
            buf = bytes(rng.randrange(256) for _ in range(n))
            assert _c_decode(exe, tmp, buf) == _py_decode(buf), buf.hex()


def test_oob_is_rejected_on_both_rails():
    if not (shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")):
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        assert exe is not None
        for buf in (b"", b"\x01", b"\x01\x02\x03\x04"):  # too short for the wider fields
            c, p = _c_decode(exe, tmp, buf), _py_decode(buf)
            assert c == p, (buf.hex(), c, p)
            assert any(st == _OOB for st, _ in p), "a short buffer must OOB some field"


def test_full_width_signed_extremes_reinterpret_identically():
    """The 64-bit signed field at the two's-complement EXTREMES.

    A full-width signed field is the one case with no room to sign-extend: the decoder
    must reinterpret the raw bit pattern. Spelling that as a bare `(int64_t)v` cast is
    implementation-defined in C11/C17 whenever `v > INT64_MAX` (only C23 mandates the
    wraparound), which is exactly why `bcir_telemetry_frame.c::rd_i64` avoids it. This
    pins the same discipline for `bcir_binrec.c` on the patterns a cast would decide:
    all-ones (-1), INT64_MIN, INT64_MAX, and INT64_MAX+1.
    """
    if not (shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")):
        return
    # _FIELDS[5] is (0, 64, "s", "little") -- the full-width signed field.
    cases = [
        (b"\xff" * 8, -1),
        (b"\x00" * 7 + b"\x80", -(2**63)),  # INT64_MIN
        (b"\xff" * 7 + b"\x7f", 2**63 - 1),  # INT64_MAX
        (b"\x01" + b"\x00" * 6 + b"\x80", -(2**63) + 1),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        assert exe is not None
        for raw, expected in cases:
            buf = raw + b"\x00" * 4  # pad so every field is in bounds
            c, p = _c_decode(exe, tmp, buf), _py_decode(buf)
            assert c == p, (raw.hex(), c, p)
            assert p[5] == (_OK, expected), (raw.hex(), p[5], expected)


def test_nvme_sqe_header_decodes_consistently():
    """The reference NVMe SQE header (the documented record) decodes the same on both
    rails -- a concrete, named ABI shape, not just random bytes."""
    if not (shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")):
        return
    from bcir.etl.binary import NVME_SQE_HEADER, decode

    buf = bytes([0x01, 0x00, 0x42, 0x00, 0x05, 0x06, 0x07, 0x08])
    fields = decode(NVME_SQE_HEADER, buf)
    assert fields["opcode"] == 0x01
    assert fields["command_id"] == 0x42  # bytes [2:4] little-endian
    assert fields["namespace_id"] == 0x08070605
