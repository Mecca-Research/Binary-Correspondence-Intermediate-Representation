"""A1 packed-Q4, calibration, and real C compute gates."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from bcir.kbcir.lowbit import (
    PackedQ4Tensor,
    SmoothQuantPolicy,
    admit_low_bit_format,
    calibrate_smoothquant,
    calibrated_q4q8_dot,
    pack_signed_int4,
    read_packed_q4,
    unpack_signed_int4,
    write_packed_q4,
)
from bcir.kbcir.quantize import quantize_group
from bcir.toolchain import host_link_args

_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME = _ROOT / "runtime" / "c"
_HASH_A = hashlib.sha256(b"q4-source").hexdigest()
_HASH_B = hashlib.sha256(b"q4-calibration").hexdigest()


def _must_refuse(call, fragment=""):
    try:
        call()
        raise AssertionError("invalid low-bit value was accepted")
    except ValueError as exc:
        assert fragment in str(exc), (fragment, str(exc))


def test_signed_q4_nibbles_are_canonical_and_low_first():
    codes = (-7, -1, 0, 1, 7)
    packed = pack_signed_int4(codes)
    assert packed == bytes((0xF9, 0x10, 0x07))
    assert unpack_signed_int4(packed, len(codes)) == codes
    _must_refuse(lambda: unpack_signed_int4(bytes((0x08,)), 1), "-8")
    _must_refuse(lambda: unpack_signed_int4(bytes((0x10,)), 1), "padding")
    _must_refuse(lambda: pack_signed_int4((8,)), "[-7, 7]")


def test_packed_q4_wire_roundtrip_is_deterministic_compact_and_crc_guarded():
    values = [math.sin(index / 7.0) * (1.0 + index % 11) for index in range(1024)]
    tensor = PackedQ4Tensor.from_values(
        values, (32, 32), source_sha256=_HASH_A, calibration_sha256=_HASH_B
    )
    with tempfile.TemporaryDirectory() as td:
        first = Path(td) / "first.q4"
        second = Path(td) / "second.q4"
        write_packed_q4(first, tensor)
        write_packed_q4(second, tensor)
        assert first.read_bytes() == second.read_bytes()
        assert read_packed_q4(first) == tensor
        assert first.stat().st_size < len(values) * 4 * 0.25
        corrupt = bytearray(first.read_bytes())
        corrupt[-1] ^= 1
        bad = Path(td) / "bad.q4"
        bad.write_bytes(corrupt)
        _must_refuse(lambda: read_packed_q4(bad), "CRC")
        bad.write_bytes(first.read_bytes()[:-1])
        _must_refuse(lambda: read_packed_q4(bad), "offsets")


def test_smoothquant_records_outliers_and_preserves_their_exact_residual():
    activations = [[0.1, 100.0, 0.2, -0.4], [0.2, -90.0, -0.1, 0.3]]
    weights = [[1.0, 0.003, -0.5, 0.25], [0.5, -0.004, 0.75, -0.5]]
    calibration = calibrate_smoothquant(
        activations, weights, SmoothQuantPolicy(outlier_threshold=4.0, max_outlier_channels=1)
    )
    assert calibration.outlier_channels == (1,)
    exact = sum(left * right for left, right in zip(activations[0], weights[0]))
    approximate = calibrated_q4q8_dot(activations[0], weights[0], calibration)
    assert abs(exact - approximate) < 0.2
    _must_refuse(
        lambda: calibrate_smoothquant(
            activations, weights, SmoothQuantPolicy(outlier_threshold=4.0, outlier_mode="reject")
        ),
        "outlier",
    )


def test_calibrated_q4q8_dot_is_bounded_over_random_vectors():
    rng = random.Random(0xA14)
    activations = [[rng.uniform(-2.0, 2.0) for _ in range(64)] for _ in range(8)]
    weights = [[rng.uniform(-1.0, 1.0) for _ in range(64)] for _ in range(4)]
    calibration = calibrate_smoothquant(
        activations, weights, SmoothQuantPolicy(outlier_threshold=100.0)
    )
    for activation in activations:
        for weight in weights:
            exact = sum(left * right for left, right in zip(activation, weight))
            approximate = calibrated_q4q8_dot(activation, weight, calibration)
            assert abs(exact - approximate) < 3.0


def test_format_admission_requires_implementation_provenance_size_and_drift():
    accepted = admit_low_bit_format(
        "bcirq4",
        artifact_bytes=40,
        reference_bytes=100,
        max_logit_error=0.02,
        nll_delta=0.01,
        source_sha256=_HASH_A,
        calibration_sha256=_HASH_B,
        max_size_ratio=0.5,
        max_error=0.05,
        max_nll_delta=0.02,
    )
    assert accepted.accepted and not accepted.reasons
    refused = admit_low_bit_format(
        "nf4",
        artifact_bytes=40,
        reference_bytes=100,
        max_logit_error=0.02,
        nll_delta=0.01,
        source_sha256=_HASH_A,
        calibration_sha256=_HASH_B,
        max_size_ratio=0.5,
        max_error=0.05,
        max_nll_delta=0.02,
    )
    assert not refused.accepted and "not present" in refused.reasons[0]
    _must_refuse(
        lambda: admit_low_bit_format(
            "bcirq4",
            artifact_bytes=40,
            reference_bytes=100,
            max_logit_error=-0.01,
            nll_delta=0.0,
            source_sha256=_HASH_A,
            calibration_sha256=_HASH_B,
            max_size_ratio=0.5,
            max_error=0.05,
            max_nll_delta=0.02,
        ),
        "magnitudes",
    )


def _compiler():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def test_c_q4_q8_scalar_twin_matches_integer_oracle():
    compiler = _compiler()
    if compiler is None:
        return
    rng = random.Random(44)
    q4 = [rng.randint(-7, 7) for _ in range(67)]
    q8 = [rng.randint(-127, 127) for _ in range(67)]
    packed = pack_signed_int4(q4)
    expected = sum(left * right for left, right in zip(q4, q8)) * math.ldexp(1.0, -5)
    packed_init = ",".join(str(value) for value in packed)
    q8_init = ",".join(str(value) for value in q8)
    source = f"""#include <stdio.h>\n#include "bcir_q4_kernel.h"\n
int main(void) {{
  const unsigned char q4[] = {{{packed_init}}};
  const signed char q8[] = {{{q8_init}}};
  printf("%.17g\\n", bcir_q4_q8_dot(q4, q8, {len(q4)}u, -2, -3));
  printf("%d\\n", bcir_q4_codes_validate(q4, (size_t)BCIR_Q4_MAX_ELEMENTS + 1u));
  return 0;
}}\n"""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "test.c"
        exe = Path(td) / ("test.exe" if os.name == "nt" else "test")
        src.write_text(source, "utf-8")
        build = subprocess.run(
            host_link_args(
                [
                    compiler,
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(_RUNTIME),
                    str(src),
                    str(_RUNTIME / "bcir_q4_kernel.c"),
                    "-lm",
                    "-o",
                    str(exe),
                ]
            ),
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run([str(exe)], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        lines = run.stdout.splitlines()
        assert abs(float(lines[0]) - expected) <= 1e-12
        assert lines[1] == "0"


def test_c_q4_avx2_object_contains_real_packed_dot_instruction():
    machine = platform.machine().lower()
    compiler = _compiler()
    objdump = shutil.which("objdump") or shutil.which("llvm-objdump")
    if compiler is None or objdump is None or machine not in ("x86_64", "amd64"):
        return
    with tempfile.TemporaryDirectory() as td:
        obj = Path(td) / "q4.o"
        build = subprocess.run(
            [
                compiler,
                "-std=c11",
                "-O3",
                "-mavx2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(_RUNTIME),
                "-c",
                str(_RUNTIME / "bcir_q4_kernel.c"),
                "-o",
                str(obj),
            ],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        dump = subprocess.run([objdump, "-d", str(obj)], capture_output=True, text=True)
        assert dump.returncode == 0, dump.stderr
        assert "vpmaddwd" in dump.stdout.lower()
