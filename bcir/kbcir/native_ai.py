"""Explicit native data-plane kernels with Python-oracle parity.

This module is opt-in and cold: importing :mod:`bcir` never loads ``ctypes`` or
searches for a compiler.  ``NativeAIKernels.build`` compiles the repository's
portable no-heap C implementation; ``load`` accepts a prebuilt library.  A caller
that requests native execution gets a hard error when the library is unavailable --
there is no silent fallback to the Python oracle.
"""
from __future__ import annotations

import ctypes
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from array import array
from dataclasses import dataclass
from pathlib import Path

from ..toolchain import host_link_args
from .optimization_memory import (OptimizationMatch, OptimizationMemory,
                                  _prepare_optimization_query)

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "runtime" / "c" / "bcir_ai_kernels.c"
_INCLUDE = _ROOT / "runtime" / "c"
_MAX_ELEMENTS = 1 << 26
_BUILD_TIMEOUT_SECONDS = 120
_ABI_VERSION = 1


class _CMatch(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32),
                ("squared_distance", ctypes.c_uint64)]


@dataclass(frozen=True)
class NativeQuantizedTensor:
    bits: int
    group_size: int
    element_count: int
    exponents: tuple[int, ...]
    codes: bytes

    def __post_init__(self) -> None:
        if self.bits not in (4, 8) or type(self.group_size) is not int \
                or not 1 <= self.group_size <= 0xFFFF:
            raise ValueError("native quantized tensor has an invalid format")
        if type(self.element_count) is not int or not 0 <= self.element_count <= _MAX_ELEMENTS:
            raise ValueError("native quantized tensor has an invalid element count")
        groups = 0 if not self.element_count \
            else (self.element_count + self.group_size - 1) // self.group_size
        if len(self.exponents) != groups or any(type(value) is not int or not -300 <= value <= 300
                                                for value in self.exponents):
            raise ValueError("native quantized exponent inventory is malformed")
        expected = self.element_count if self.bits == 8 else (self.element_count + 1) // 2
        if not isinstance(self.codes, bytes) or len(self.codes) != expected:
            raise ValueError("native quantized code inventory is malformed")
        if self.bits == 8 and b"\x80" in self.codes:
            raise ValueError("native Q8 contains forbidden non-symmetric code -128")
        if self.bits == 4:
            for index in range(self.element_count):
                nibble = self.codes[index // 2] >> (4 if index & 1 else 0) & 0xF
                if nibble == 8:
                    raise ValueError("native Q4 contains forbidden non-symmetric code -8")
            if self.element_count & 1 and self.codes[-1] & 0xF0:
                raise ValueError("native Q4 odd-element padding must be zero")

    @property
    def exponent_bytes_le(self) -> bytes:
        return struct.pack(f"<{len(self.exponents)}h", *self.exponents)


class NativeAIKernels:
    """Loaded BCIR native-AI library. The object owns the dynamic-library handle."""

    def __init__(self, path: os.PathLike | str) -> None:
        source = Path(path)
        if not source.is_file():
            raise RuntimeError(f"native AI library does not exist: {source}")
        self.path = source.resolve()
        try:
            self._library = ctypes.CDLL(str(self.path))
        except OSError as exc:
            raise RuntimeError(f"cannot load native AI library {self.path}: {exc}") from exc
        try:
            self._configure()
        except AttributeError as exc:
            raise RuntimeError(f"native AI library has an incompatible ABI: {exc}") from exc

    @classmethod
    def load(cls, path: os.PathLike | str) -> "NativeAIKernels":
        """Load an explicitly built library; absence or ABI mismatch never falls back."""
        return cls(path)

    @staticmethod
    def _library_name() -> str:
        if os.name == "nt":
            return "bcir_ai_kernels.dll"
        if sys.platform == "darwin":
            return "libbcir_ai_kernels.dylib"
        return "libbcir_ai_kernels.so"

    @classmethod
    def build(cls, directory: os.PathLike | str, *, cc: str | None = None) -> "NativeAIKernels":
        """Build and load the checked-in C kernels without writing inside the source tree."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        compiler = cc or os.environ.get("CC") or shutil.which("clang") \
            or shutil.which("cc") or shutil.which("gcc")
        if not compiler or not _SOURCE.is_file():
            raise RuntimeError("native AI kernels require a C11 compiler and runtime/c sources")
        target = target_dir / cls._library_name()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.stem}.tmp-", suffix=target.suffix, dir=target_dir)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            command = [compiler, "-std=c11", "-O3", "-ffp-contract=off", "-Wall",
                       "-Wextra", "-Wpedantic", "-Werror", "-DBCIR_AI_BUILD_SHARED",
                       "-shared"]
            if os.name != "nt":
                command.append("-fPIC")
            command.extend(["-I", str(_INCLUDE), str(_SOURCE), "-o", str(temporary),
                            *host_link_args(["-lm"])])
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=_BUILD_TIMEOUT_SECONDS)
            if result.returncode:
                raise RuntimeError(f"native AI library build failed:\n{result.stderr}")
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return cls(target)

    def _configure(self) -> None:
        u8p = ctypes.POINTER(ctypes.c_uint8)
        i8p = ctypes.POINTER(ctypes.c_int8)
        i16p = ctypes.POINTER(ctypes.c_int16)
        f64p = ctypes.POINTER(ctypes.c_double)
        sizep = ctypes.POINTER(ctypes.c_size_t)
        matchp = ctypes.POINTER(_CMatch)
        self._library.bcir_ai_abi_version.argtypes = []
        self._library.bcir_ai_abi_version.restype = ctypes.c_uint32
        actual_abi = int(self._library.bcir_ai_abi_version())
        if actual_abi != _ABI_VERSION:
            raise RuntimeError(
                f"native AI library ABI {actual_abi} does not match required ABI {_ABI_VERSION}")
        self._library.bcir_ai_quantize_q8_f64.argtypes = [
            f64p, ctypes.c_size_t, ctypes.c_size_t, i16p, ctypes.c_size_t,
            i8p, ctypes.c_size_t]
        self._library.bcir_ai_quantize_q8_f64.restype = ctypes.c_int
        self._library.bcir_ai_quantize_q4_f64.argtypes = [
            f64p, ctypes.c_size_t, ctypes.c_size_t, i16p, ctypes.c_size_t,
            u8p, ctypes.c_size_t]
        self._library.bcir_ai_quantize_q4_f64.restype = ctypes.c_int
        matrix_args = [f64p, ctypes.c_size_t, u8p, ctypes.c_size_t, u8p,
                       ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, f64p]
        self._library.bcir_ai_q8_matvec_f64.argtypes = matrix_args
        self._library.bcir_ai_q8_matvec_f64.restype = ctypes.c_int
        self._library.bcir_ai_q8_rows_dot_f64.argtypes = matrix_args
        self._library.bcir_ai_q8_rows_dot_f64.restype = ctypes.c_int
        self._library.bcir_ai_q15_topk.argtypes = [
            i16p, i16p, u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            matchp, ctypes.c_size_t, sizep]
        self._library.bcir_ai_q15_topk.restype = ctypes.c_int

    @staticmethod
    def _finite_values(values) -> tuple[float, ...]:
        try:
            rows = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("native quantization inputs must be numeric") from exc
        if len(rows) > _MAX_ELEMENTS or any(not math.isfinite(value) for value in rows):
            raise ValueError("native quantization inputs must be bounded and finite")
        return rows

    def quantize(self, values, *, group_size: int, bits: int) -> NativeQuantizedTensor:
        if type(group_size) is not int or not 1 <= group_size <= 0xFFFF:
            raise ValueError("native quantization group_size must be in [1, 65535]")
        if bits not in (4, 8):
            raise ValueError("native quantization supports only Q4 and Q8")
        rows = self._finite_values(values)
        count = len(rows)
        groups = 0 if not count else (count + group_size - 1) // group_size
        values_array = (ctypes.c_double * count)(*rows)
        exponent_array = (ctypes.c_int16 * groups)()
        if bits == 8:
            code_array = (ctypes.c_int8 * count)()
            status = self._library.bcir_ai_quantize_q8_f64(
                values_array, count, group_size, exponent_array, groups,
                code_array, count)
            codes = ctypes.string_at(code_array, count)
        else:
            packed_count = (count + 1) // 2
            code_array = (ctypes.c_uint8 * packed_count)()
            status = self._library.bcir_ai_quantize_q4_f64(
                values_array, count, group_size, exponent_array, groups,
                code_array, packed_count)
            codes = ctypes.string_at(code_array, packed_count)
        if status:
            raise RuntimeError(f"native Q{bits} quantization failed with status {status}")
        return NativeQuantizedTensor(bits, group_size, count,
                                     tuple(int(value) for value in exponent_array), codes)

    @staticmethod
    def _matrix_inputs(input_values, codes, exponents, output_count: int,
                       group_size: int):
        values = NativeAIKernels._finite_values(input_values)
        if (not values or type(output_count) is not int or output_count < 1
                or type(group_size) is not int or not 1 <= group_size <= 0xFFFF
                or not isinstance(codes, (bytes, bytearray, memoryview))):
            raise ValueError("native Q8 matrix arguments are malformed")
        raw_codes = bytes(codes)
        expected = len(values) * output_count
        if expected > _MAX_ELEMENTS or len(raw_codes) != expected:
            raise ValueError("native Q8 matrix code extent is malformed")
        try:
            exponent_values = tuple(exponents)
        except TypeError as exc:
            raise ValueError("native Q8 exponent inventory is malformed") from exc
        groups = (expected + group_size - 1) // group_size
        if len(exponent_values) != groups or any(type(value) is not int or not -300 <= value <= 300
                                                  for value in exponent_values):
            raise ValueError("native Q8 exponent inventory is malformed")
        return values, raw_codes, struct.pack(f"<{groups}h", *exponent_values), groups

    def _matrix(self, function, input_values, codes, exponents, *,
                output_count: int, group_size: int) -> tuple[float, ...]:
        values, raw_codes, exponent_bytes, groups = self._matrix_inputs(
            input_values, codes, exponents, output_count, group_size)
        input_array = (ctypes.c_double * len(values))(*values)
        code_array = (ctypes.c_uint8 * len(raw_codes)).from_buffer_copy(raw_codes)
        exponent_array = (ctypes.c_uint8 * len(exponent_bytes)).from_buffer_copy(exponent_bytes)
        output_array = (ctypes.c_double * output_count)()
        status = function(input_array, len(values), code_array, len(raw_codes),
                          exponent_array, groups, output_count, group_size,
                          output_array)
        if status:
            raise RuntimeError(f"native Q8 matrix kernel failed with status {status}")
        return tuple(float(value) for value in output_array)

    def q8_matvec(self, input_values, codes, exponents, *, output_count: int,
                  group_size: int) -> tuple[float, ...]:
        return self._matrix(self._library.bcir_ai_q8_matvec_f64, input_values,
                            codes, exponents, output_count=output_count,
                            group_size=group_size)

    def q8_rows_dot(self, input_values, codes, exponents, *, output_count: int,
                    group_size: int) -> tuple[float, ...]:
        return self._matrix(self._library.bcir_ai_q8_rows_dot_f64, input_values,
                            codes, exponents, output_count=output_count,
                            group_size=group_size)

    def _q15_topk(self, query: array, patterns: array, eligible: bytes,
                  pattern_count: int, dimension: int, top_k: int) -> tuple[tuple[int, int], ...]:
        query_view = (ctypes.c_int16 * len(query)).from_buffer(query)
        pattern_view = (ctypes.c_int16 * len(patterns)).from_buffer(patterns)
        eligible_view = (ctypes.c_uint8 * len(eligible)).from_buffer_copy(eligible)
        matches = (_CMatch * top_k)()
        count = ctypes.c_size_t(0)
        status = self._library.bcir_ai_q15_topk(
            query_view, pattern_view, eligible_view, pattern_count, dimension,
            top_k, matches, top_k, ctypes.byref(count))
        if status:
            raise RuntimeError(f"native Q15 top-k failed with status {status}")
        return tuple((int(matches[index].index), int(matches[index].squared_distance))
                     for index in range(count.value))


class NativeOptimizationIndex:
    """One-time contiguous native projection of immutable optimization memory."""

    def __init__(self, kernels: NativeAIKernels, memory: OptimizationMemory) -> None:
        if not isinstance(kernels, NativeAIKernels) or not isinstance(memory, OptimizationMemory):
            raise ValueError("native optimization index needs typed kernels and memory")
        self.kernels = kernels
        self.memory = memory
        self._patterns = array("h", (coordinate for pattern in memory.patterns
                                     for coordinate in pattern.embedding_q15))

    def query(self, embedding_q15, *, top_k: int = 1,
              active_facts=None) -> tuple[OptimizationMatch, ...]:
        values, eligible = _prepare_optimization_query(
            self.memory, embedding_q15, top_k=top_k, active_facts=active_facts)
        query = array("h", values)
        matches = self.kernels._q15_topk(
            query, self._patterns, bytes(eligible), len(self.memory.patterns),
            self.memory.dimension, top_k)
        return tuple(OptimizationMatch(
            self.memory.patterns[index].pattern_sha256,
            self.memory.patterns[index].context_shard_sha256, distance)
                     for index, distance in matches)


__all__ = ["NativeAIKernels", "NativeOptimizationIndex", "NativeQuantizedTensor"]
