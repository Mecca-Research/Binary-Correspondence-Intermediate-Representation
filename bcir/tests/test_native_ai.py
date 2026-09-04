"""Python-oracle parity and refusal gates for the optional native AI data plane."""

from __future__ import annotations

import atexit
import gc
import hashlib
import math
import random
import shutil
import struct
import tempfile
from pathlib import Path

from bcir.frontends.models.model_microbench import run_bounded_model_microbench_native
from bcir.frontends.models.weights_io import write_q8_decoder
from bcir.kbcir.lowbit import PackedQ4Tensor, pack_signed_int4
from bcir.kbcir.native_ai import NativeAIKernels, NativeOptimizationIndex, NativeQuantizedTensor
from bcir.kbcir.optimization_memory import (
    OptimizationFact,
    OptimizationMemory,
    OptimizationPattern,
    query_optimization_memory,
)
from bcir.kbcir.quantize import quantize_per_group
from bcir.tests.test_model_weights_io import _HASHES, _TOKENS, _model

_CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
_NATIVE = None
_NATIVE_DIRECTORY = None


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _native():
    global _NATIVE, _NATIVE_DIRECTORY
    if _CC is None:
        return None
    if _NATIVE is None:
        _NATIVE_DIRECTORY = tempfile.TemporaryDirectory()
        _NATIVE = NativeAIKernels.build(_NATIVE_DIRECTORY.name, cc=_CC)
        atexit.register(_cleanup_native)
    return _NATIVE


def _cleanup_native():
    global _NATIVE, _NATIVE_DIRECTORY
    directory = _NATIVE_DIRECTORY
    _NATIVE = None
    gc.collect()
    _NATIVE_DIRECTORY = None
    if directory is not None:
        directory.cleanup()


def _oracle(values, group_size: int, bits: int):
    groups = quantize_per_group(values, group_size, bits)
    exponents = tuple(group.scale_exp for group in groups)
    codes = tuple(code for group in groups for code in group.codes)
    encoded = bytes(code & 0xFF for code in codes) if bits == 8 else pack_signed_int4(codes)
    return exponents, encoded


def test_native_quantization_is_byte_identical_for_q8_q4_odd_groups_and_extremes():
    native = _native()
    if native is None:
        return
    rng = random.Random(0xBC1A1)
    fixtures = (
        (),
        (0.0,),
        (math.ldexp(1.0, -1074), -math.ldexp(1.0, -1074)),
        (
            math.ldexp(127.0, -20),
            math.nextafter(math.ldexp(127.0, -20), 0.0),
            math.nextafter(math.ldexp(127.0, -20), math.inf),
        ),
        (
            math.ldexp(7.0, 20),
            math.nextafter(math.ldexp(7.0, 20), 0.0),
            math.nextafter(math.ldexp(7.0, 20), math.inf),
        ),
        tuple(rng.uniform(-100.0, 100.0) for _ in range(31)),
        tuple(rng.uniform(-1.0, 1.0) for _ in range(33)),
        (1e-300, -1e-300, 1e300, -1e300, 0.0),
    )
    for values in fixtures:
        for group_size in (1, 3, 32):
            for bits in (4, 8):
                expected_exponents, expected_codes = _oracle(values, group_size, bits)
                actual = native.quantize(values, group_size=group_size, bits=bits)
                assert actual.exponents == expected_exponents
                assert actual.codes == expected_codes
                assert actual.exponent_bytes_le == struct.pack(
                    f"<{len(expected_exponents)}h", *expected_exponents
                )


def test_native_q8_matrix_kernels_match_increasing_coordinate_oracle_exactly():
    native = _native()
    if native is None:
        return
    rng = random.Random(0xA18)
    inputs = tuple(rng.uniform(-2.0, 2.0) for _ in range(17))
    weights = tuple(rng.uniform(-4.0, 4.0) for _ in range(17 * 11))
    quantized = native.quantize(weights, group_size=13, bits=8)
    signed_codes = tuple(code - 256 if code >= 128 else code for code in quantized.codes)

    expected_matvec = []
    for output in range(11):
        total = 0.0
        for coordinate in range(17):
            index = coordinate * 11 + output
            total += inputs[coordinate] * math.ldexp(
                signed_codes[index], quantized.exponents[index // 13]
            )
        expected_matvec.append(total)
    assert native.q8_matvec(
        inputs, quantized.codes, quantized.exponents, output_count=11, group_size=13
    ) == tuple(expected_matvec)

    expected_rows = []
    for output in range(11):
        total = 0.0
        for coordinate in range(17):
            index = output * 17 + coordinate
            total += inputs[coordinate] * math.ldexp(
                signed_codes[index], quantized.exponents[index // 13]
            )
        expected_rows.append(total)
    assert native.q8_rows_dot(
        inputs, quantized.codes, quantized.exponents, output_count=11, group_size=13
    ) == tuple(expected_rows)


def test_native_boundary_rejects_malformed_numbers_extents_and_symmetric_codes():
    try:
        NativeAIKernels("definitely-missing-native-ai-library")
        raise AssertionError("missing native library was accepted")
    except RuntimeError:
        pass
    malformed_values = (
        (8, 32, 1, (0,), b"\x80"),
        (4, 32, 1, (0,), b"\x08"),
        (4, 32, 1, (0,), b"\x10"),
    )
    for arguments in malformed_values:
        try:
            NativeQuantizedTensor(*arguments)
            raise AssertionError("malformed native quantized value object was accepted")
        except ValueError:
            pass
    native = _native()
    if native is None:
        return
    for values in ((0.0, math.nan), (0.0, math.inf)):
        try:
            native.quantize(values, group_size=32, bits=8)
            raise AssertionError("non-finite native quantization input was accepted")
        except ValueError:
            pass
    bad_calls = (
        lambda: native.quantize((1.0,), group_size=True, bits=8),
        lambda: native.quantize((1.0,), group_size=32, bits=3),
        lambda: native.q8_matvec((1.0,), b"\x01\x02", (0,), output_count=1, group_size=32),
        lambda: native.q8_matvec((1.0,), b"\x80", (0,), output_count=1, group_size=32),
        lambda: native.q8_rows_dot((math.nan,), b"\x01", (0,), output_count=1, group_size=32),
    )
    for call in bad_calls:
        try:
            call()
            raise AssertionError("malformed native request was accepted")
        except (ValueError, RuntimeError):
            pass


def test_native_q4_and_q8_exports_are_identical_to_python_artifacts():
    native = _native()
    if native is None:
        return
    spec, weights = _model()
    values = tuple(math.sin(index / 9.0) for index in range(65))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        python_q8 = root / "python.bcirq8"
        native_q8 = root / "native.bcirq8"
        write_q8_decoder(python_q8, spec, weights, source_hashes=_HASHES, tokenizer_ids=_TOKENS)
        write_q8_decoder(
            native_q8,
            spec,
            weights,
            source_hashes=_HASHES,
            tokenizer_ids=_TOKENS,
            native_kernels=native,
        )
        assert python_q8.read_bytes() == native_q8.read_bytes()
        python_q4 = PackedQ4Tensor.from_values(
            values, (65,), source_sha256=_hash("source"), calibration_sha256=_hash("calibration")
        )
        native_q4 = PackedQ4Tensor.from_values_native(
            values,
            (65,),
            source_sha256=_hash("source"),
            calibration_sha256=_hash("calibration"),
            native_kernels=native,
        )
        assert native_q4 == python_q4


def _optimization_memory() -> OptimizationMemory:
    facts = tuple(
        sorted((OptimizationFact("host", "isa", "x86_64"), OptimizationFact("gpu", "format", "q8")))
    )
    patterns = []
    for index in range(257):
        required = (facts[index & 1],)
        embedding = tuple(
            ((index * 97 + coordinate * 31) % 65536) - 32768 for coordinate in range(17)
        )
        patterns.append(
            OptimizationPattern(
                _hash(f"pattern-{index:04d}"),
                embedding,
                _hash(f"shard-{index}"),
                _hash(f"evidence-{index}"),
                required,
            )
        )
    return OptimizationMemory(
        _hash("model"),
        _hash("embedder"),
        tuple(sorted(patterns, key=lambda row: row.pattern_sha256)),
        facts,
    )


def test_native_q15_index_matches_hard_fact_filtered_python_oracle():
    native = _native()
    if native is None:
        return
    memory = _optimization_memory()
    query = tuple((coordinate * 1009) % 65536 - 32768 for coordinate in range(17))
    index = NativeOptimizationIndex(native, memory)
    assert index.query(query, top_k=13) == query_optimization_memory(memory, query, top_k=13)
    assert index.query(
        query, top_k=13, active_facts=(memory.facts[0],)
    ) == query_optimization_memory(memory, query, top_k=13, active_facts=(memory.facts[0],))


def test_native_model_microbench_is_bounded_and_emits_typed_intervals():
    if _CC is None:
        return
    for weight_format in ("source", "bcirq8-group32"):
        prefill, decode = run_bounded_model_microbench_native(
            dimension=8, repeats=3, weight_format=weight_format, cc=_CC
        )
        assert (prefill.operation, decode.operation) == ("prefill", "decode")
        assert prefill.lower_ns <= prefill.median_ns <= prefill.upper_ns
        assert decode.lower_ns <= decode.median_ns <= decode.upper_ns
        assert prefill.operations == 2 * 8 * 8 * 8
        assert decode.operations == 2 * 8 * 8


def run():
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()


if __name__ == "__main__":
    run()
