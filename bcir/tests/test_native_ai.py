"""Python-oracle parity and refusal gates for the optional native AI data plane."""

from __future__ import annotations

import atexit
import gc
import hashlib
import json
import math
import os
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
        _NATIVE_DIRECTORY = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        _NATIVE = NativeAIKernels.build(_NATIVE_DIRECTORY.name, cc=_CC)
        atexit.register(_cleanup_native)
    return _NATIVE


def _cleanup_native():
    global _NATIVE, _NATIVE_DIRECTORY
    directory = _NATIVE_DIRECTORY
    kernels, _NATIVE = _NATIVE, None
    if kernels is not None:
        # Dropping the reference does not unload the library -- ctypes has no
        # finalizer that does -- so on Windows the directory below would refuse to
        # go. Releasing the handle first is what makes this cleanup a cleanup.
        kernels.close()
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


def _q8_oracle(inputs, signed_codes, exponents, outputs, group, rows):
    """The increasing-coordinate oracle: a matvec (`rows` False: codes [inputs, outputs]) or the
    row dots (`rows` True: codes [outputs, inputs]), each output's chain from +0.0."""
    expected = []
    for output in range(outputs):
        total = 0.0
        for coordinate in range(len(inputs)):
            index = output * len(inputs) + coordinate if rows else coordinate * outputs + output
            total += inputs[coordinate] * math.ldexp(signed_codes[index], exponents[index // group])
        expected.append(total)
    return tuple(expected)


def test_native_q8_instances_match_the_increasing_coordinate_oracle():
    """Q8-LANES: every instance of the Q8 matrix kernels the host admits -- 1 the portable loops, 2
    AVX2, 4 AVX-512 -- is the increasing-coordinate oracle's value for every output, bit for bit:
    at the corpus decoder's projection shapes, at the pair's (columns a multiple of 8, a power-of-two
    group of at least 8: Q8-PAIR) with an odd last row, and at ragged ones (groups straddling rows and
    vectors, a group not a power of two, groups under 8). 0 is the widest admitted; a name outside
    0/1/2/4 is refused before the library, an instance the host does not admit by it."""
    native = _native()
    if native is None:
        return
    instances = native.q8_instances()
    assert instances & 1 and not instances & ~7
    rng = random.Random(0x0A8E5)
    symmetric = tuple(code for code in range(256) if code != 0x80)
    shapes = (
        (128, 64, 32),
        (16, 72, 16),
        (3, 200, 8),
        (2, 136, 64),
        (17, 33, 5),
        (9, 40, 4),
        (24, 48, 48),
        (1, 9, 7),
        (40, 17, 32),
    )
    ran = 0
    for inputs_count, outputs, group in shapes:
        inputs = tuple(rng.uniform(-2.0, 2.0) for _ in range(inputs_count))
        codes = bytes(rng.choice(symmetric) for _ in range(inputs_count * outputs))
        exponents = tuple(rng.randint(-30, 30) for _ in range(-(-len(codes) // group)))
        signed = tuple(code - 256 if code >= 128 else code for code in codes)
        for rows, kernel in ((False, native.q8_matvec), (True, native.q8_rows_dot)):
            expected = _q8_oracle(inputs, signed, exponents, outputs, group, rows)
            for instance in (0, 1, 2, 4):
                if instance and not instances & instance:
                    continue
                got = kernel(
                    inputs,
                    codes,
                    exponents,
                    output_count=outputs,
                    group_size=group,
                    instance=instance,
                )
                assert got == expected, (inputs_count, outputs, group, rows, instance)
                ran += 1
    assert ran >= 2 * len(shapes) * 2  # 0 and 1 at least, on every host
    for refused in (-1, 3, 5, 8, True):
        try:
            native.q8_matvec((1.0,), b"\x01", (0,), output_count=1, group_size=32, instance=refused)
            raise AssertionError(f"instance {refused!r} was accepted")
        except ValueError:
            pass
    for absent in (2, 4):
        if instances & absent:
            continue
        try:
            native.q8_rows_dot(
                (1.0,), b"\x01", (0,), output_count=1, group_size=32, instance=absent
            )
            raise AssertionError(f"instance {absent} the host does not admit was accepted")
        except RuntimeError:
            pass


def test_native_ai_c_harness_c11_and_c23():
    """Q8-LANES: the kernel unit's C harness (runtime/c/test_ai_kernels.c) under C11 and C23, under C23
    capped at AVX2 and with the dispatch compiled out, every build with the instance probe: each
    admitted instance against the portable loops over the decoder's, the block's and ragged shapes,
    exponents past a normal double's, the code 0x80, non-finite outputs zeroing every output and
    refusals writing nothing, and the prevalidated entries on the widest instance. Q8-BOUND: the path
    every vector matvec took -- x times a group's scale once a row and group where the call's ranges
    prove every such product exact, the terms as before elsewhere -- over the lanes test and the
    ranges' edges, each case every byte the portable loops'. Q8-PAIR: the kernel every vector matvec ran -- two
    rows a pass where eight columns' terms lie in one group, the runs elsewhere -- over the lanes test and the
    passes' edges."""
    if _CC is None:
        return
    import re
    import subprocess

    from bcir.toolchain import build_units, host_link_args

    c = Path(__file__).resolve().parents[2] / "runtime" / "c"
    builds = (
        ("c11", [], None),
        ("c2x", [], None),
        ("c2x", ["-DBCIR_AI_MAX_INSTANCE=2"], 3),
        ("c2x", ["-DBCIR_AI_NO_DISPATCH"], 1),
    )
    seen = set()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        for number, (standard, options, cap) in enumerate(builds):
            exe = Path(temporary) / f"ai-kernels-{number}"
            cmd = [
                _CC,
                f"-std={standard}",
                "-DBCIR_AI_INSTANCE_PROBE",
                *options,
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-ffp-contract=off",
                "-I",
                str(c),
                str(c / "test_ai_kernels.c"),
                str(c / "bcir_ai_kernels.c"),
                str(c / "bcir_q4_kernel.c"),
                "-o",
                str(exe),
            ]
            build_units(
                cmd + host_link_args(["-lm"]), f"the AI kernels' {standard} build", timeout=120
            )
            result = subprocess.run(
                [str(exe)], check=True, capture_output=True, text=True, timeout=120
            )
            assert "native AI kernels: all gates passed" in result.stdout
            found = re.search(
                r"Q8-LANES: instances (\d+), (\d+) vector comparisons equal", result.stdout
            )
            assert found, result.stdout
            admitted, compared = int(found.group(1)), int(found.group(2))
            assert admitted & 1 and not admitted & ~7
            if cap is not None:
                assert not admitted & ~cap, (options, admitted)
            # 22 shapes, four exponent ranges, two kernels: 176 comparisons an admitted vector instance
            assert compared == 176 * bin(admitted & 6).count("1"), (options, admitted, compared)
            # Q8-BOUND, an admitted vector instance: the lanes test's 22 shapes with the format's and the narrow
            # exponents proved (44) and with those past a normal double's not (44); the edges' 11 cases, 3 proved
            bound = re.search(
                r"Q8-BOUND: (\d+) matvecs hoisted, (\d+) computed again", result.stdout
            )
            assert bound, result.stdout
            vector = bin(admitted & 6).count("1")
            counts = (int(bound.group(1)), int(bound.group(2)))
            assert counts == (47 * vector, 52 * vector), (options, admitted, counts)
            # Q8-PAIR, two rows a pass where eight columns' terms lie in one group, an admitted vector instance: of the
            # lanes test's 22 shapes 11 and of the edges' 17 shapes 12, each in four exponent ranges
            pair = re.search(
                r"Q8-PAIR: (\d+) matvecs two rows a pass, (\d+) in runs", result.stdout
            )
            assert pair, result.stdout
            counts = (int(pair.group(1)), int(pair.group(2)))
            assert counts == (92 * vector, 64 * vector), (options, admitted, counts)
            seen.add(admitted)
    native = _native()
    if native is not None:
        assert native.q8_instances() in seen  # the default builds admit what the library does


def _asan_compiler() -> str | None:
    """The first C compiler here that builds a program with -fsanitize=address,undefined and runs it, or None."""
    import subprocess

    names = (_CC, shutil.which("clang"), shutil.which("gcc"), shutil.which("cc"))
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        source, exe = Path(temporary) / "probe.c", Path(temporary) / "probe"
        source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
        for cc in dict.fromkeys(name for name in names if name):
            try:
                built = subprocess.run(
                    [cc, "-fsanitize=address,undefined", str(source), "-o", str(exe)],
                    capture_output=True,
                    timeout=120,
                )
                if built.returncode != 0:
                    continue
                ran = subprocess.run([str(exe)], capture_output=True, timeout=60, env=env)
                if ran.returncode == 0:
                    return cc
            except (OSError, subprocess.TimeoutExpired):
                continue
    return None


class _AddressSanitizer:
    """Truthy where a C compiler here runs a program under AddressSanitizer and UBSan, probed once, when first read:
    what `check_tests.py --require` asks of this module when `tools/testing/faults/q8-pair.json` runs the harness's
    sanitizer build, which returns without one."""

    _probed = False
    _cc: str | None = None

    def compiler(self) -> str | None:
        if not _AddressSanitizer._probed:
            _AddressSanitizer._cc = _asan_compiler() if _CC is not None else None
            _AddressSanitizer._probed = True
        return _AddressSanitizer._cc

    def __bool__(self) -> bool:
        return self.compiler() is not None


_ASAN = _AddressSanitizer()


def test_native_ai_c_harness_under_address_sanitizer():
    """Q8-PAIR: the kernel unit's C harness, with the probe, under AddressSanitizer and UBSan. Its edges allocate
    every matrix at its exact size, so a read or write past a matrix's codes, its last group or its outputs -- a
    pass running a step past its rows' end -- stops the run. Returns where no compiler here runs a sanitized
    program."""
    cc = _ASAN.compiler()
    if cc is None:
        return
    import re
    import subprocess

    from bcir.toolchain import build_units, host_link_args

    c = Path(__file__).resolve().parents[2] / "runtime" / "c"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        exe = Path(temporary) / "ai-kernels-asan"
        cmd = [
            cc,
            "-std=c11",
            "-DBCIR_AI_INSTANCE_PROBE",
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-ffp-contract=off",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(c),
            str(c / "test_ai_kernels.c"),
            str(c / "bcir_ai_kernels.c"),
            str(c / "bcir_q4_kernel.c"),
            "-o",
            str(exe),
        ]
        build_units(cmd + host_link_args(["-lm"]), "the AI kernels' sanitizer build", timeout=180)
        env = dict(
            os.environ,
            ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
            UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1",
        )
        result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300, env=env)
        assert result.returncode == 0, (result.returncode, result.stderr[-4000:])
        assert "native AI kernels: all gates passed" in result.stdout, result.stdout
        assert re.search(r"Q8-PAIR: \d+ matvecs two rows a pass, \d+ in runs", result.stdout), (
            result.stdout
        )


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


def _narrow_memory(admitted: int) -> OptimizationMemory:
    """A memory in which exactly `admitted` patterns can satisfy the query's facts."""
    facts = tuple(
        sorted((OptimizationFact("host", "isa", "x86_64"), OptimizationFact("gpu", "format", "q8")))
    )
    patterns = []
    for index in range(24):
        required = (facts[0],) if index < admitted else (facts[1],)
        embedding = tuple(
            ((index * 313 + coordinate * 17) % 65536) - 32768 for coordinate in range(9)
        )
        patterns.append(
            OptimizationPattern(
                _hash(f"narrow-{index:04d}"),
                embedding,
                _hash(f"narrow-shard-{index}"),
                _hash(f"narrow-evidence-{index}"),
                required,
            )
        )
    return OptimizationMemory(
        _hash("narrow-model"),
        _hash("narrow-embedder"),
        tuple(sorted(patterns, key=lambda row: row.pattern_sha256)),
        facts,
    )


def test_native_q15_returns_fewer_matches_than_asked_when_fewer_are_admitted():
    """The short return, on both rails.

    Every other test here admits more patterns than it asks for, so the kernel's
    `count < top_k` exit -- where it writes a shorter result and stops -- had never
    executed under a gate, in C or through this binding. A filter's most likely
    wrong answer is to return the rows it was supposed to remove, and that answer is
    indistinguishable from the right one unless the mask is actually narrower than
    the request.
    """
    native = _native()
    if native is None:
        return
    facts = _narrow_memory(1).facts
    for admitted in (0, 1, 2, 7):
        memory = _narrow_memory(admitted)
        query = tuple((coordinate * 4409) % 65536 - 32768 for coordinate in range(9))
        index = NativeOptimizationIndex(native, memory)
        oracle = query_optimization_memory(memory, query, top_k=13, active_facts=(facts[0],))
        kernel = index.query(query, top_k=13, active_facts=(facts[0],))
        assert len(oracle) == admitted, (admitted, len(oracle))
        assert kernel == oracle, (admitted, kernel, oracle)
    # ... and asking for fewer than are admitted still returns exactly that many.
    memory = _narrow_memory(7)
    query = tuple((coordinate * 4409) % 65536 - 32768 for coordinate in range(9))
    index = NativeOptimizationIndex(native, memory)
    assert len(index.query(query, top_k=3, active_facts=(facts[0],))) == 3


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


def _identity(path: Path) -> tuple[int, int]:
    """What changes if and only if the file was replaced by a build."""
    stat = path.stat()
    return (stat.st_ino, stat.st_mtime_ns)


def _stamp_vouches(library: Path) -> bool:
    """Does the stamp beside `library` validate the bytes now on disk?

    This is what a rebuild promises, and the only thing it promises on every host.
    "The rebuilt library has the same bytes as the first one" is a stronger claim
    that belongs to the *toolchain*: an ELF build is reproducible, a PE carries a
    `TimeDateStamp`, and asserting the stronger one everywhere reports the object
    format as a defect in the cache.

    One predicate rather than the same comparison written out at each site, because
    writing it out per site is how there came to be three of them (L14).
    """
    return NativeAIKernels._reusable(
        library,
        NativeAIKernels._stamp_path(library),
        NativeAIKernels._input_digest(_CC),
    )


def _toolchain_is_reproducible(directory) -> bool:
    """Does compiling the same inputs twice produce the same bytes here?

    On ELF hosts it does, and the stamp's recorded output digest is then a stable
    fact about the source. A Windows PE carries a `TimeDateStamp` in its header, so
    two builds of one unchanged translation unit differ in their bytes -- a property
    of the object format, not of anything BCIR decides.

    S1 does not need reproducibility: the stamp records the digest of whatever that
    build produced, and reuse requires the file on disk to still be *that*. So this
    is a probe rather than a requirement -- it lets the reproducible hosts keep
    asserting the stronger property without the others reporting the object format
    as a defect in the cache. `/Brepro` would make a PE reproducible and is the
    follow-up if that is ever wanted; nothing here depends on it.
    """
    library = Path(directory) / NativeAIKernels._library_name()
    NativeAIKernels.build(directory, cc=_CC, rebuild=True).close()
    first = hashlib.sha256(library.read_bytes()).hexdigest()
    NativeAIKernels.build(directory, cc=_CC, rebuild=True).close()
    return hashlib.sha256(library.read_bytes()).hexdigest() == first


def _build_and_release(directory, **kwargs) -> None:
    """Build, then release the handle before anything touches the file again.

    Windows locks a loaded module: `build`'s own `os.replace` over a library this
    process still has open fails, as does unlinking it or removing the directory it
    lives in. These tests deliberately rebuild into one directory, so each handle is
    released as soon as the build that produced it has been observed. On POSIX this
    changes nothing; it is the difference between running and not running on the
    other host in the matrix.
    """
    NativeAIKernels.build(directory, cc=_CC, **kwargs).close()


def _kernel_workspace():
    """A temp directory for kernel builds, tolerant of a handle the OS still holds.

    `close` releases what this process owns, which is enough for every rebuild
    below. `ignore_cleanup_errors` covers only the last library standing at the end
    of a test: the directory is the OS's to reclaim, and failing a stamp test over
    it would be reporting the platform's file locking as a defect in the stamp.
    """
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def test_native_kernels_release_the_handle_they_claim_to_own():
    """The class docstring says it owns the handle; this is the release half.

    Without it a caller cannot delete or replace the library file on Windows, where
    a loaded module is locked -- so a temporary build directory cannot be cleaned up
    and `build` cannot `os.replace` over a library it has already loaded. Closing is
    final on purpose: every entry point reaches the library through `_library`, so a
    call afterwards raises rather than jumping into an address the loader unmapped.
    """
    if _CC is None:
        return
    with _kernel_workspace() as directory:
        kernels = NativeAIKernels.build(directory, cc=_CC)
        assert kernels.quantize((1.0, -1.0), group_size=2, bits=8) is not None
        kernels.close()
        try:
            kernels.quantize((1.0, -1.0), group_size=2, bits=8)
        except AttributeError:
            pass
        else:
            raise AssertionError("a closed library still served a call")
        kernels.close()  # idempotent: a second release is not a second free
        with NativeAIKernels.build(directory, cc=_CC) as scoped:
            assert scoped.quantize((1.0, -1.0), group_size=2, bits=8) is not None
        try:
            scoped.quantize((1.0, -1.0), group_size=2, bits=8)
        except AttributeError:
            pass
        else:
            raise AssertionError("leaving the context did not release the handle")


def test_native_build_reuses_the_library_when_nothing_changed():
    """The compiler is the cost; the stamp is what stops it running twice for nothing."""
    if _CC is None:
        return
    with _kernel_workspace() as directory:
        _build_and_release(directory)
        library = Path(directory) / NativeAIKernels._library_name()
        stamp = NativeAIKernels._stamp_path(library)
        assert stamp.is_file()
        before = _identity(library)
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        for _ in range(3):
            _build_and_release(directory)
            assert _identity(library) == before
            assert hashlib.sha256(library.read_bytes()).hexdigest() == digest
        # Anti-vacuity: the observation above is only evidence if it CAN change.
        _build_and_release(directory, rebuild=True)
        assert _identity(library) != before
        # Whatever that rebuild produced, the stamp must now vouch for it -- that is
        # the property the cache rests on, and it holds on every host.
        assert _stamp_vouches(library)
        rebuilt = _identity(library)
        _build_and_release(directory)
        assert _identity(library) == rebuilt
        # ... and where the toolchain is reproducible, a forced rebuild of unchanged
        # inputs also gives back the same bytes. A PE embeds a build timestamp, so
        # this is asserted where it is true rather than required everywhere.
        if _toolchain_is_reproducible(directory):
            assert hashlib.sha256(library.read_bytes()).hexdigest() == digest


def test_native_build_recompiles_when_the_recorded_inputs_do_not_match():
    if _CC is None:
        return
    with _kernel_workspace() as directory:
        _build_and_release(directory)
        library = Path(directory) / NativeAIKernels._library_name()
        stamp = NativeAIKernels._stamp_path(library)
        recorded = json.loads(stamp.read_text(encoding="utf-8"))
        recorded["inputs"] = "0" * 64
        stamp.write_text(json.dumps(recorded), encoding="utf-8")
        before = _identity(library)
        _build_and_release(directory)
        assert _identity(library) != before
        assert json.loads(stamp.read_text(encoding="utf-8"))["inputs"] != "0" * 64


def _replace_bytes(path: Path, payload: bytes) -> None:
    """Swap a file's contents by replacing the inode, never by writing through it.

    A built library is mapped executable by every `ctypes.CDLL` still holding it.
    Writing through the path would mutate code under a live mapping -- which
    segfaults the interpreter rather than testing anything. `os.replace` leaves every
    existing mapping on the old inode, which is also how a real concurrent rebuild
    behaves.

    The caller must already have released its handle (see `_build_and_release`):
    POSIX permits the replace either way, but Windows locks a loaded module and
    would refuse it.
    """
    scratch = path.with_name(f".{path.name}.swap")
    scratch.write_bytes(payload)
    os.replace(scratch, path)


def test_native_build_refuses_a_stamp_that_does_not_match_the_library():
    """A stamp is a claim about bytes on disk. Tampered bytes retire the claim."""
    if _CC is None:
        return
    with _kernel_workspace() as directory:
        _build_and_release(directory)
        library = Path(directory) / NativeAIKernels._library_name()
        original = library.read_bytes()
        tampered = original + b"\x00tampered"
        _replace_bytes(library, tampered)
        before = _identity(library)
        _build_and_release(directory)
        assert _identity(library) != before
        # The tampered bytes are gone and a real build stands in their place. Not
        # "the bytes are the original ones": that is the toolchain's reproducibility,
        # not the stamp's promise.
        assert library.read_bytes() != tampered
        assert _stamp_vouches(library)
        NativeAIKernels.load(library).close()


def test_native_build_refuses_a_malformed_stamp():
    """Every unreadable stamp is a rebuild, never a silent reuse (L1: fail closed)."""
    if _CC is None:
        return
    with _kernel_workspace() as directory:
        _build_and_release(directory)
        library = Path(directory) / NativeAIKernels._library_name()
        stamp = NativeAIKernels._stamp_path(library)
        honest = json.loads(stamp.read_text(encoding="utf-8"))
        corruptions = (
            "",
            "not json at all",
            "[]",
            '"a string"',
            "{}",
            '{"schema": "wrong", "inputs": "x", "output": "y"}',
            '{"schema": "bcir-ai-kernels/build-stamp/v1"}',
            '{"schema": "bcir-ai-kernels/build-stamp/v1", "inputs": null, "output": null}',
            # Every other field correct, so ONLY the schema check can reject this one.
            # Without it a future format would be read under this format's rules.
            json.dumps({**honest, "schema": "bcir-ai-kernels/build-stamp/v2"}),
            # Same, isolating the output check: correct schema and inputs, wrong bytes.
            json.dumps({**honest, "output": "0" * 64}),
        )
        for corruption in corruptions:
            stamp.write_text(corruption, encoding="utf-8")
            before = _identity(library)
            _build_and_release(directory)
            assert _identity(library) != before, corruption
            # A rebuild happened, so the stamp it just wrote must vouch for the file
            # it just wrote. Whether those bytes equal the first build's is the
            # toolchain's business, checked once in the reuse test.
            assert _stamp_vouches(library), corruption
        stamp.unlink()
        before = _identity(library)
        _build_and_release(directory)
        assert _identity(library) != before


def test_native_build_rebuilds_when_the_library_is_gone():
    if _CC is None:
        return
    with _kernel_workspace() as directory:
        _build_and_release(directory)
        library = Path(directory) / NativeAIKernels._library_name()
        stamp = NativeAIKernels._stamp_path(library)
        recorded = json.loads(stamp.read_text(encoding="utf-8"))
        library.unlink()
        _build_and_release(directory)
        assert library.is_file()
        rebuilt = json.loads(stamp.read_text(encoding="utf-8"))
        # The inputs did not move, so the recorded inputs must not either. The output
        # digest is whatever this build produced -- identical to the first on a
        # reproducible toolchain, different on one that stamps a build time into the
        # object -- and either way the stamp has to vouch for the file beside it.
        assert rebuilt["inputs"] == recorded["inputs"]
        assert rebuilt["output"] == hashlib.sha256(library.read_bytes()).hexdigest()
        assert _stamp_vouches(library)
        # The recovered library is usable, not merely present.
        NativeAIKernels.load(library).close()


def test_native_build_stamp_covers_every_translation_unit_input():
    """Anti-vacuity (L2): a digest over an empty input set is stable and worthless.

    The freshness check is only as good as the set of files it hashes, so this asserts
    the set is non-empty AND that changing any one member -- the source or any header
    beside it -- moves the digest.
    """
    import bcir.kbcir.native_ai as native_ai

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        headers = [root / "bcir_ai_kernels.h", root / "other.h"]
        source = root / "bcir_ai_kernels.c"
        for path in headers:
            path.write_text("/* header */\n", encoding="utf-8")
        source.write_text("/* source */\n", encoding="utf-8")
        original_include, original_source = native_ai._INCLUDE, native_ai._SOURCE
        native_ai._INCLUDE, native_ai._SOURCE = root, source
        try:
            baseline = NativeAIKernels._input_digest("cc")
            assert NativeAIKernels._input_digest("cc") == baseline
            for path in [*headers, source]:
                previous = path.read_text(encoding="utf-8")
                path.write_text(previous + "/* moved */\n", encoding="utf-8")
                assert NativeAIKernels._input_digest("cc") != baseline, path.name
                path.write_text(previous, encoding="utf-8")
                assert NativeAIKernels._input_digest("cc") == baseline, path.name
            # The compiler is part of the identity too.
            assert NativeAIKernels._input_digest("cc") != NativeAIKernels._input_digest("other-cc")
            # And so is the flag vector.
            saved = native_ai._COMPILE_FLAGS
            native_ai._COMPILE_FLAGS = saved + ("-DEXTRA",)
            try:
                assert NativeAIKernels._input_digest("cc") != baseline
            finally:
                native_ai._COMPILE_FLAGS = saved
        finally:
            native_ai._INCLUDE, native_ai._SOURCE = original_include, original_source


def run():
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()


if __name__ == "__main__":
    run()
