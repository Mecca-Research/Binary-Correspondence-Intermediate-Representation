"""Explicit native data-plane kernels with Python-oracle parity.

This module is opt-in and cold: importing :mod:`bcir` never loads ``ctypes`` or
searches for a compiler.  ``NativeAIKernels.build`` compiles the repository's
portable no-heap C implementation; ``load`` accepts a prebuilt library.  A caller
that requests native execution gets a hard error when the library is unavailable --
there is no silent fallback to the Python oracle.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
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

from ..toolchain import host_link_args, unit_options
from .optimization_memory import OptimizationMatch, OptimizationMemory, _prepare_optimization_query

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "runtime" / "c" / "bcir_ai_kernels.c"
_INCLUDE = _ROOT / "runtime" / "c"
_MAX_ELEMENTS = 1 << 26
_BUILD_TIMEOUT_SECONDS = 120
_ABI_VERSION = 1
_STAMP_SCHEMA = "bcir-ai-kernels/build-stamp/v1"
_COMPILE_FLAGS = (
    "-std=c11",
    "-O3",
    "-ffp-contract=off",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-DBCIR_AI_BUILD_SHARED",
    "-shared",
)


class _CMatch(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("squared_distance", ctypes.c_uint64)]


class _CLut4Path(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_size_t),
        ("v_packed", ctypes.POINTER(ctypes.c_uint8)),
        ("v_packed_bytes", ctypes.c_size_t),
        ("u_packed", ctypes.POINTER(ctypes.c_uint8)),
        ("u_packed_bytes", ctypes.c_size_t),
        ("g_fp16_le", ctypes.POINTER(ctypes.c_uint8)),
        ("l_fp16_le", ctypes.POINTER(ctypes.c_uint8)),
        ("h_fp16_le", ctypes.POINTER(ctypes.c_uint8)),
    ]


@dataclass(frozen=True)
class NativeQuantizedTensor:
    bits: int
    group_size: int
    element_count: int
    exponents: tuple[int, ...]
    codes: bytes

    def __post_init__(self) -> None:
        if (
            self.bits not in (4, 8)
            or type(self.group_size) is not int
            or not 1 <= self.group_size <= 0xFFFF
        ):
            raise ValueError("native quantized tensor has an invalid format")
        if type(self.element_count) is not int or not 0 <= self.element_count <= _MAX_ELEMENTS:
            raise ValueError("native quantized tensor has an invalid element count")
        groups = (
            0
            if not self.element_count
            else (self.element_count + self.group_size - 1) // self.group_size
        )
        if len(self.exponents) != groups or any(
            type(value) is not int or not -300 <= value <= 300 for value in self.exponents
        ):
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

    def close(self) -> None:
        """Release the dynamic-library handle this object owns.

        The class docstring has always claimed that ownership; this is the other
        half of it. Without a release, a caller cannot delete or replace the library
        file on Windows, where a loaded module is locked: a temporary build
        directory cannot be cleaned up, and `build` cannot `os.replace` over a
        library it has already loaded. On POSIX nothing is locked and closing is
        merely tidy -- the platform that needs this is the one that cannot work
        without it.

        Closing is final and the object is unusable afterwards, which is the point:
        every entry point reaches the library through `self._library`, so a call
        after close raises `AttributeError` on None rather than dereferencing an
        address the loader has unmapped.
        """
        library, self._library = self._library, None
        handle = getattr(library, "_handle", None)
        if handle is None:
            return
        try:
            if os.name == "nt":
                self._free_windows_module(handle)
            else:
                import _ctypes

                _ctypes.dlclose(handle)
        except (OSError, AttributeError, ValueError):
            # A handle the platform declines to release is the platform's to keep.
            # The object is already unusable, which is all `close` promises.
            pass

    @staticmethod
    def _free_windows_module(handle: int) -> None:
        """`FreeLibrary`, through whichever door this interpreter offers.

        `_ctypes.FreeLibrary` is a CPython implementation detail and not promised to
        exist; `kernel32.FreeLibrary` is the documented API and is what actually has
        to succeed, because on Windows an unreleased module keeps its file locked and
        the next `os.replace` over it fails. Trying the private door first and the
        public one after costs nothing and removes the interpreter-version guess.

        The handle is an HMODULE: passed as `c_void_p` so it is not truncated on a
        64-bit host, which an implicit int conversion would risk.
        """
        import _ctypes

        release = getattr(_ctypes, "FreeLibrary", None)
        if release is not None:
            release(handle)
            return
        ctypes.WinDLL("kernel32", use_last_error=True).FreeLibrary(ctypes.c_void_p(handle))

    def __enter__(self) -> "NativeAIKernels":
        return self

    def __exit__(self, *exception_info) -> None:
        self.close()

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

    @staticmethod
    def _compile_command(compiler: str, output: Path) -> list[str]:
        """The exact argv a build would run. One definition, so the stamp cannot drift: the flags,
        then the manifest's options for the kernels' unit under this compiler (M18,
        bcir.toolchain.unit_options -- on x86-64 the JCC-erratum mitigation, without which their
        vector loops ran up to x1.35 slower at some placements on the Cascade Lake core
        KERNEL-ALIGN measured)."""
        own = [f for f in unit_options([_SOURCE.name], compiler) if f not in _COMPILE_FLAGS]
        command = [compiler, *_COMPILE_FLAGS, *own]
        if os.name != "nt":
            command.append("-fPIC")
        command.extend(
            ["-I", str(_INCLUDE), str(_SOURCE), "-o", str(output), *host_link_args(["-lm"])]
        )
        return command

    @staticmethod
    def _input_digest(compiler: str) -> str:
        """Content address of everything that decides the library's bytes.

        Every translation-unit input is hashed by name and content: the kernel source
        and *every* header beside it, which over-approximates the include graph on
        purpose -- an unrelated header change can only force a needless rebuild, never
        permit a stale reuse. The compiler is identified by its resolved path, size and
        modification time rather than by running it each time: a version probe costs ~25 ms
        against a ~1 ms hash, and the recorded output digest below already refuses any
        library whose bytes are not the ones this stamp was written for. (The command's
        manifest options are this compiler's -- its family and machine, asked once a process
        by bcir.toolchain.compiler_identity.)

        Declared scope: a compiler swapped in place at the same path, size and
        nanosecond mtime is not distinguished. Nothing else about the build is out of
        scope -- flags, link arguments, ABI, platform and every input file are in.
        """
        digest = hashlib.sha256()
        digest.update(_STAMP_SCHEMA.encode("utf-8"))
        digest.update(b"\0")
        for name, value in (
            ("abi", str(_ABI_VERSION)),
            ("platform", sys.platform),
            ("os.name", os.name),
            ("flags", "\x1f".join(NativeAIKernels._compile_command(compiler, Path("<out>")))),
        ):
            digest.update(f"{name}={value}".encode())
            digest.update(b"\0")
        try:
            resolved = Path(compiler).resolve()
            stat = resolved.stat()
            identity = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            identity = f"{compiler}|unstattable"
        digest.update(f"cc={identity}".encode())
        digest.update(b"\0")
        inputs = sorted(_INCLUDE.glob("*.h")) + [_SOURCE]
        for path in inputs:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _stamp_path(target: Path) -> Path:
        return target.with_name(target.name + ".stamp")

    @classmethod
    def _reusable(cls, target: Path, stamp_path: Path, inputs: str) -> bool:
        """True only when the stamp proves this exact library came from these inputs.

        Both halves are required. The input digest says the sources and the toolchain
        have not moved; the output digest says the file on disk is still the one the
        build produced. A stamp that is missing, unreadable, malformed, or disagrees
        with either half is not a verdict of "fresh" -- it falls through to a rebuild.
        """
        try:
            recorded = json.loads(stamp_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(recorded, dict):
            return False
        if recorded.get("schema") != _STAMP_SCHEMA:
            return False
        if recorded.get("inputs") != inputs:
            return False
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError:
            return False
        return recorded.get("output") == actual

    @classmethod
    def build(
        cls, directory: os.PathLike | str, *, cc: str | None = None, rebuild: bool = False
    ) -> "NativeAIKernels":
        """Build and load the checked-in C kernels without writing inside the source tree.

        The compiler runs only when the library beside the stamp is not already the one
        these inputs produce. `rebuild=True` compiles unconditionally, which is what the
        gate uses to prove the cached path and the compiled path agree byte for byte.
        """
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        compiler = (
            cc
            or os.environ.get("CC")
            or shutil.which("clang")
            or shutil.which("cc")
            or shutil.which("gcc")
        )
        if not compiler or not _SOURCE.is_file():
            raise RuntimeError("native AI kernels require a C11 compiler and runtime/c sources")
        target = target_dir / cls._library_name()
        stamp_path = cls._stamp_path(target)
        inputs = cls._input_digest(compiler)
        if not rebuild and cls._reusable(target, stamp_path, inputs):
            return cls(target)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.stem}.tmp-", suffix=target.suffix, dir=target_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            command = cls._compile_command(compiler, temporary)
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_SECONDS
            )
            if result.returncode:
                raise RuntimeError(f"native AI library build failed:\n{result.stderr}")
            output = hashlib.sha256(temporary.read_bytes()).hexdigest()
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        # The stamp is published after the library it describes, and atomically, so a
        # reader never sees a stamp vouching for bytes that are not on disk yet.
        cls._write_stamp(stamp_path, inputs=inputs, output=output)
        return cls(target)

    @staticmethod
    def _write_stamp(stamp_path: Path, *, inputs: str, output: str) -> None:
        payload = json.dumps(
            {"schema": _STAMP_SCHEMA, "inputs": inputs, "output": output},
            sort_keys=True,
            separators=(",", ":"),
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{stamp_path.name}.tmp-", dir=stamp_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(temporary_name, stamp_path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

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
                f"native AI library ABI {actual_abi} does not match required ABI {_ABI_VERSION}"
            )
        self._library.bcir_ai_quantize_q8_f64.argtypes = [
            f64p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            i16p,
            ctypes.c_size_t,
            i8p,
            ctypes.c_size_t,
        ]
        self._library.bcir_ai_quantize_q8_f64.restype = ctypes.c_int
        self._library.bcir_ai_quantize_q4_f64.argtypes = [
            f64p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            i16p,
            ctypes.c_size_t,
            u8p,
            ctypes.c_size_t,
        ]
        self._library.bcir_ai_quantize_q4_f64.restype = ctypes.c_int
        matrix_args = [
            f64p,
            ctypes.c_size_t,
            u8p,
            ctypes.c_size_t,
            u8p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            f64p,
        ]
        self._library.bcir_ai_q8_matvec_f64.argtypes = matrix_args
        self._library.bcir_ai_q8_matvec_f64.restype = ctypes.c_int
        self._library.bcir_ai_q8_rows_dot_f64.argtypes = matrix_args
        self._library.bcir_ai_q8_rows_dot_f64.restype = ctypes.c_int
        # Q8-LANES: the kernels' instances, and the prevalidated entries on a named one
        self._library.bcir_ai_q8_instances.argtypes = []
        self._library.bcir_ai_q8_instances.restype = ctypes.c_int
        self._library.bcir_ai_q8_values_validate.argtypes = [
            u8p,
            ctypes.c_size_t,
            u8p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._library.bcir_ai_q8_values_validate.restype = ctypes.c_int
        for entry in ("bcir_ai_q8_matvec_f64_instance", "bcir_ai_q8_rows_dot_f64_instance"):
            getattr(self._library, entry).argtypes = [ctypes.c_int, *matrix_args]
            getattr(self._library, entry).restype = ctypes.c_int
        # DECODE-THREADS: the matvec over a range of its columns
        column_args = [*matrix_args[:-1], ctypes.c_size_t, ctypes.c_size_t, f64p]
        self._library.bcir_ai_q8_matvec_f64_columns.argtypes = column_args
        self._library.bcir_ai_q8_matvec_f64_columns.restype = ctypes.c_int
        self._library.bcir_ai_q8_matvec_f64_columns_instance.argtypes = [ctypes.c_int, *column_args]
        self._library.bcir_ai_q8_matvec_f64_columns_instance.restype = ctypes.c_int
        # HEAD-PACK: the row dots' pack, and the row dots over it
        self._library.bcir_ai_q8_rows_pack_size.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            sizep,
        ]
        self._library.bcir_ai_q8_rows_pack_size.restype = ctypes.c_int
        self._library.bcir_ai_q8_rows_pack.argtypes = [
            u8p,
            ctypes.c_size_t,
            u8p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            u8p,
            ctypes.c_size_t,
        ]
        self._library.bcir_ai_q8_rows_pack.restype = ctypes.c_int
        packed_args = [
            f64p,
            ctypes.c_size_t,
            u8p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            f64p,
        ]
        self._library.bcir_ai_q8_rows_dot_f64_packed.argtypes = packed_args
        self._library.bcir_ai_q8_rows_dot_f64_packed.restype = ctypes.c_int
        self._library.bcir_ai_q8_rows_dot_f64_packed_instance.argtypes = [
            ctypes.c_int,
            *packed_args,
        ]
        self._library.bcir_ai_q8_rows_dot_f64_packed_instance.restype = ctypes.c_int
        self._library.bcir_ai_q15_topk.argtypes = [
            i16p,
            i16p,
            u8p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            matchp,
            ctypes.c_size_t,
            sizep,
        ]
        self._library.bcir_ai_q15_topk.restype = ctypes.c_int
        # LUT4-LANES: packed 1-bit factorized matvec/mb4 kernels (f64 & f32)
        f32p = ctypes.POINTER(ctypes.c_float)
        lut4p = ctypes.POINTER(_CLut4Path)
        self._library.bcir_ai_lut4_instances.argtypes = []
        self._library.bcir_ai_lut4_instances.restype = ctypes.c_int
        self._library.bcir_ai_lut4_packed_bytes.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            sizep,
        ]
        self._library.bcir_ai_lut4_packed_bytes.restype = ctypes.c_int
        self._library.bcir_ai_lut4_pack_signs.argtypes = [
            i8p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            u8p,
            ctypes.c_size_t,
        ]
        self._library.bcir_ai_lut4_pack_signs.restype = ctypes.c_int
        self._library.bcir_ai_lut4_validate_packed.argtypes = [
            u8p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._library.bcir_ai_lut4_validate_packed.restype = ctypes.c_int
        self._library.bcir_ai_lut4_paths_validate.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            lut4p,
            ctypes.c_size_t,
        ]
        self._library.bcir_ai_lut4_paths_validate.restype = ctypes.c_int
        lut4_f64_args = [
            f64p,
            ctypes.c_size_t,
            lut4p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            f64p,
        ]
        self._library.bcir_ai_lut4_matvec_f64.argtypes = lut4_f64_args
        self._library.bcir_ai_lut4_matvec_f64.restype = ctypes.c_int
        self._library.bcir_ai_lut4_matvec_f64_instance.argtypes = [ctypes.c_int, *lut4_f64_args]
        self._library.bcir_ai_lut4_matvec_f64_instance.restype = ctypes.c_int
        lut4_f64_col_args = [
            f64p,
            ctypes.c_size_t,
            lut4p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            f64p,
        ]
        self._library.bcir_ai_lut4_matvec_f64_columns.argtypes = lut4_f64_col_args
        self._library.bcir_ai_lut4_matvec_f64_columns.restype = ctypes.c_int
        self._library.bcir_ai_lut4_matvec_f64_columns_instance.argtypes = [
            ctypes.c_int,
            *lut4_f64_col_args,
        ]
        self._library.bcir_ai_lut4_matvec_f64_columns_instance.restype = ctypes.c_int
        self._library.bcir_ai_lut4_matmul_mb4_f64_instance.argtypes = [
            ctypes.c_int,
            f64p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            lut4p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            f64p,
        ]
        self._library.bcir_ai_lut4_matmul_mb4_f64_instance.restype = ctypes.c_int
        self._library.bcir_ai_lut4_matvec_f32_instance.argtypes = [
            ctypes.c_int,
            f32p,
            ctypes.c_size_t,
            lut4p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            f32p,
        ]
        self._library.bcir_ai_lut4_matvec_f32_instance.restype = ctypes.c_int
        self._library.bcir_ai_lut4_matvec_f32_columns_instance.argtypes = [
            ctypes.c_int,
            f32p,
            ctypes.c_size_t,
            lut4p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            f32p,
        ]
        self._library.bcir_ai_lut4_matvec_f32_columns_instance.restype = ctypes.c_int
        self._library.bcir_ai_lut4_matmul_mb4_f32_instance.argtypes = [
            ctypes.c_int,
            f32p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            lut4p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            f32p,
        ]
        self._library.bcir_ai_lut4_matmul_mb4_f32_instance.restype = ctypes.c_int

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
                values_array, count, group_size, exponent_array, groups, code_array, count
            )
            codes = ctypes.string_at(code_array, count)
        else:
            packed_count = (count + 1) // 2
            code_array = (ctypes.c_uint8 * packed_count)()
            status = self._library.bcir_ai_quantize_q4_f64(
                values_array, count, group_size, exponent_array, groups, code_array, packed_count
            )
            codes = ctypes.string_at(code_array, packed_count)
        if status:
            raise RuntimeError(f"native Q{bits} quantization failed with status {status}")
        return NativeQuantizedTensor(
            bits, group_size, count, tuple(int(value) for value in exponent_array), codes
        )

    @staticmethod
    def _matrix_inputs(input_values, codes, exponents, output_count: int, group_size: int):
        values = NativeAIKernels._finite_values(input_values)
        if (
            not values
            or type(output_count) is not int
            or output_count < 1
            or type(group_size) is not int
            or not 1 <= group_size <= 0xFFFF
            or not isinstance(codes, (bytes, bytearray, memoryview))
        ):
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
        if len(exponent_values) != groups or any(
            type(value) is not int or not -300 <= value <= 300 for value in exponent_values
        ):
            raise ValueError("native Q8 exponent inventory is malformed")
        return values, raw_codes, struct.pack(f"<{groups}h", *exponent_values), groups

    def q8_instances(self) -> int:
        """The Q8 matrix kernels' instances this host admits, as bits: 1 the portable loops (always), 2
        AVX2, 4 AVX-512 (Q8-LANES) -- every one the portable loops' value for every output."""
        return int(self._library.bcir_ai_q8_instances())

    def _matrix(
        self,
        entry: str,
        input_values,
        codes,
        exponents,
        *,
        output_count: int,
        group_size: int,
        instance: int | None = None,
    ) -> tuple[float, ...]:
        values, raw_codes, exponent_bytes, groups = self._matrix_inputs(
            input_values, codes, exponents, output_count, group_size
        )
        input_array = (ctypes.c_double * len(values))(*values)
        code_array = (ctypes.c_uint8 * len(raw_codes)).from_buffer_copy(raw_codes)
        exponent_array = (ctypes.c_uint8 * len(exponent_bytes)).from_buffer_copy(exponent_bytes)
        output_array = (ctypes.c_double * output_count)()
        if instance is not None:
            # the validating entry's own check, then the prevalidated entry on the named instance
            if type(instance) is not int or instance not in (0, 1, 2, 4):
                raise ValueError("native Q8 instance must be 0, 1, 2 or 4")
            status = self._library.bcir_ai_q8_values_validate(
                code_array, len(raw_codes), exponent_array, groups, len(raw_codes), group_size
            )
            if status:
                raise RuntimeError(f"native Q8 matrix kernel failed with status {status}")
            function, extra = getattr(self._library, entry + "_instance"), (instance,)
        else:
            function, extra = getattr(self._library, entry), ()
        status = function(
            *extra,
            input_array,
            len(values),
            code_array,
            len(raw_codes),
            exponent_array,
            groups,
            output_count,
            group_size,
            output_array,
        )
        if status:
            raise RuntimeError(f"native Q8 matrix kernel failed with status {status}")
        return tuple(float(value) for value in output_array)

    def q8_matvec(
        self,
        input_values,
        codes,
        exponents,
        *,
        output_count: int,
        group_size: int,
        instance: int | None = None,
        columns: tuple[int, int] | None = None,
    ) -> tuple[float, ...]:
        """`instance` (Q8-LANES): None the validating entry (the widest instance), else 0, 1, 2 or 4 --
        the prevalidated entry on that instance after the same validation; one the host does not admit
        raises. `columns` (DECODE-THREADS): (begin, end), the outputs begin to end - 1 alone, computed by the
        column entry on the instance named (None the widest) after the same validation -- each the whole call's."""
        if columns is not None:
            return self._matvec_columns(
                input_values,
                codes,
                exponents,
                output_count=output_count,
                group_size=group_size,
                instance=instance,
                columns=columns,
            )
        return self._matrix(
            "bcir_ai_q8_matvec_f64",
            input_values,
            codes,
            exponents,
            output_count=output_count,
            group_size=group_size,
            instance=instance,
        )

    def _matvec_columns(
        self,
        input_values,
        codes,
        exponents,
        *,
        output_count: int,
        group_size: int,
        instance: int | None,
        columns,
    ) -> tuple[float, ...]:
        """DECODE-THREADS: the column entry over [begin, end), after the validating entry's own check."""
        values, raw_codes, exponent_bytes, groups = self._matrix_inputs(
            input_values, codes, exponents, output_count, group_size
        )
        if (
            type(columns) is not tuple
            or len(columns) != 2
            or any(type(edge) is not int for edge in columns)
            or not 0 <= columns[0] < columns[1] <= output_count
        ):
            raise ValueError(
                "native Q8 columns must be (begin, end) with 0 <= begin < end <= output_count"
            )
        if instance is not None and (type(instance) is not int or instance not in (0, 1, 2, 4)):
            raise ValueError("native Q8 instance must be 0, 1, 2 or 4")
        input_array = (ctypes.c_double * len(values))(*values)
        code_array = (ctypes.c_uint8 * len(raw_codes)).from_buffer_copy(raw_codes)
        exponent_array = (ctypes.c_uint8 * len(exponent_bytes)).from_buffer_copy(exponent_bytes)
        output_array = (ctypes.c_double * output_count)()
        status = self._library.bcir_ai_q8_values_validate(
            code_array, len(raw_codes), exponent_array, groups, len(raw_codes), group_size
        )
        if status:
            raise RuntimeError(f"native Q8 matrix kernel failed with status {status}")
        if instance is None:
            function, extra = self._library.bcir_ai_q8_matvec_f64_columns, ()
        else:
            function, extra = self._library.bcir_ai_q8_matvec_f64_columns_instance, (instance,)
        status = function(
            *extra,
            input_array,
            len(values),
            code_array,
            len(raw_codes),
            exponent_array,
            groups,
            output_count,
            group_size,
            columns[0],
            columns[1],
            output_array,
        )
        if status:
            raise RuntimeError(f"native Q8 matrix kernel failed with status {status}")
        return tuple(float(value) for value in output_array[columns[0] : columns[1]])

    def _pack(
        self,
        raw_codes: bytes,
        exponent_bytes: bytes,
        groups: int,
        input_count: int,
        output_count: int,
        group_size: int,
    ):
        """HEAD-PACK: the C library's pack of validated codes and exponents, as a ctypes array of its size."""
        size = ctypes.c_size_t()
        if self._library.bcir_ai_q8_rows_pack_size(
            input_count, output_count, group_size, ctypes.byref(size)
        ):
            raise ValueError(
                "native Q8 rows pack shape is malformed: no pack holds rows that straddle their groups"
            )
        # the layout's own size, ceil(rows / 8) blocks of input_count / group_size segments of 16 + 8 * group_size
        # bytes: a library that sizes a pack otherwise would write past the buffer allocated for it
        if size.value != -(-output_count // 8) * (input_count // group_size) * (
            16 + 8 * group_size
        ):
            raise RuntimeError(
                "native Q8 rows pack size disagrees with the layout bcir_ai_kernels.h states"
            )
        code_array = (ctypes.c_uint8 * len(raw_codes)).from_buffer_copy(raw_codes)
        exponent_array = (ctypes.c_uint8 * len(exponent_bytes)).from_buffer_copy(exponent_bytes)
        pack = (ctypes.c_uint8 * size.value)()
        status = self._library.bcir_ai_q8_rows_pack(
            code_array,
            len(raw_codes),
            exponent_array,
            groups,
            input_count,
            output_count,
            group_size,
            pack,
            size.value,
        )
        if status:
            raise RuntimeError(f"native Q8 rows pack failed with status {status}")
        return pack

    def q8_rows_pack(
        self, codes, exponents, *, input_count: int, output_count: int, group_size: int
    ) -> bytes:
        """HEAD-PACK: `output_count` rows of `input_count` codes, and their groups' exponents, packed for the vector
        row dots -- bcir_ai_kernels.h's layout; input_count a whole number of groups."""
        if type(input_count) is not int or not 1 <= input_count <= _MAX_ELEMENTS:
            raise ValueError("native Q8 matrix arguments are malformed")
        _, raw_codes, exponent_bytes, groups = self._matrix_inputs(
            (0.0,) * input_count, codes, exponents, output_count, group_size
        )
        return bytes(
            self._pack(raw_codes, exponent_bytes, groups, input_count, output_count, group_size)
        )

    def q8_rows_dot(
        self,
        input_values,
        codes,
        exponents,
        *,
        output_count: int,
        group_size: int,
        instance: int | None = None,
        packed: bool = False,
    ) -> tuple[float, ...]:
        """`instance` as q8_matvec's. `packed` (HEAD-PACK): the codes and exponents packed first (q8_rows_pack) and
        the row dots taken over the pack -- every output the same; the input a whole number of groups."""
        if packed is not False and packed is not True:
            raise ValueError("native Q8 packed must be a bool")
        if packed:
            values, raw_codes, exponent_bytes, groups = self._matrix_inputs(
                input_values, codes, exponents, output_count, group_size
            )
            if instance is not None and (type(instance) is not int or instance not in (0, 1, 2, 4)):
                raise ValueError("native Q8 instance must be 0, 1, 2 or 4")
            pack = self._pack(
                raw_codes, exponent_bytes, groups, len(values), output_count, group_size
            )
            input_array = (ctypes.c_double * len(values))(*values)
            output_array = (ctypes.c_double * output_count)()
            if instance is None:
                function, extra = self._library.bcir_ai_q8_rows_dot_f64_packed, ()
            else:
                function, extra = self._library.bcir_ai_q8_rows_dot_f64_packed_instance, (instance,)
            status = function(
                *extra,
                input_array,
                len(values),
                pack,
                len(pack),
                output_count,
                group_size,
                output_array,
            )
            if status:
                raise RuntimeError(f"native Q8 matrix kernel failed with status {status}")
            return tuple(float(value) for value in output_array)
        return self._matrix(
            "bcir_ai_q8_rows_dot_f64",
            input_values,
            codes,
            exponents,
            output_count=output_count,
            group_size=group_size,
            instance=instance,
        )

    def lut4_instances(self) -> int:
        """LUT4-LANES instances this host admits: 1 portable, 2 AVX2, 4 AVX-512 (0 bit drift)."""
        return int(self._library.bcir_ai_lut4_instances())

    def lut4_packed_bytes(self, rows: int, cols: int) -> int:
        if type(rows) is not int or type(cols) is not int or rows < 1 or cols < 1:
            raise ValueError("LUT4 dimensions must be positive integers")
        out_size = ctypes.c_size_t()
        status = self._library.bcir_ai_lut4_packed_bytes(rows, cols, ctypes.byref(out_size))
        if status:
            raise ValueError(f"LUT4 packed size query failed with status {status}")
        return int(out_size.value)

    def lut4_pack_signs(self, signs, *, rows: int, cols: int) -> bytes:
        packed_len = self.lut4_packed_bytes(rows, cols)
        try:
            flat = tuple(int(v) for v in signs)
        except (TypeError, ValueError) as exc:
            raise ValueError("LUT4 signs must be an integer sequence of +1/-1") from exc
        if len(flat) != rows * cols or any(v not in (-1, 1) for v in flat):
            raise ValueError("LUT4 signs must contain rows*cols entries in {-1, +1}")
        sign_arr = (ctypes.c_int8 * len(flat))(*flat)
        out_arr = (ctypes.c_uint8 * packed_len)()
        status = self._library.bcir_ai_lut4_pack_signs(sign_arr, rows, cols, out_arr, packed_len)
        if status:
            raise RuntimeError(f"native LUT4 sign packing failed with status {status}")
        return bytes(out_arr)

    def lut4_validate_packed(self, packed, *, rows: int, cols: int) -> bool:
        if (
            type(rows) is not int
            or type(cols) is not int
            or rows < 1
            or cols < 1
            or not isinstance(packed, (bytes, bytearray, memoryview))
        ):
            return False
        raw = bytes(packed)
        arr = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
        return self._library.bcir_ai_lut4_validate_packed(arr, len(raw), rows, cols) == 0

    def _build_lut4_paths(self, paths, input_count: int, output_count: int):
        path_list = tuple(paths)
        if not 1 <= len(path_list) <= 2:
            raise ValueError("LUT4 matvec requires 1 or 2 factorized paths")
        c_paths = (_CLut4Path * len(path_list))()
        keepalive: list[object] = []
        for idx, item in enumerate(path_list):
            if isinstance(item, dict):
                rank = item["rank"]
                v_raw = bytes(item["v_packed"])
                u_raw = bytes(item["u_packed"])
                g_raw = bytes(item["g_fp16_le"])
                l_raw = bytes(item["l_fp16_le"])
                h_raw = bytes(item["h_fp16_le"])
            else:
                rank = item.rank
                v_raw = bytes(item.v_packed)
                u_raw = bytes(item.u_packed)
                g_raw = bytes(item.g_fp16_le)
                l_raw = bytes(item.l_fp16_le)
                h_raw = bytes(item.h_fp16_le)
            if type(rank) is not int or not 1 <= rank <= 1024:
                raise ValueError("LUT4 path rank must be an integer in [1, 1024]")
            if (
                len(g_raw) != input_count * 2
                or len(l_raw) != rank * 2
                or len(h_raw) != output_count * 2
            ):
                raise ValueError("LUT4 path binary16 scale length mismatch")
            v_arr = (ctypes.c_uint8 * len(v_raw)).from_buffer_copy(v_raw)
            u_arr = (ctypes.c_uint8 * len(u_raw)).from_buffer_copy(u_raw)
            g_arr = (ctypes.c_uint8 * len(g_raw)).from_buffer_copy(g_raw)
            l_arr = (ctypes.c_uint8 * len(l_raw)).from_buffer_copy(l_raw)
            h_arr = (ctypes.c_uint8 * len(h_raw)).from_buffer_copy(h_raw)
            keepalive.extend((v_arr, u_arr, g_arr, l_arr, h_arr))
            c_paths[idx] = _CLut4Path(
                rank=rank,
                v_packed=v_arr,
                v_packed_bytes=len(v_raw),
                u_packed=u_arr,
                u_packed_bytes=len(u_raw),
                g_fp16_le=g_arr,
                l_fp16_le=l_arr,
                h_fp16_le=h_arr,
            )
        status = self._library.bcir_ai_lut4_paths_validate(
            input_count, output_count, c_paths, len(path_list)
        )
        if status:
            raise ValueError(f"native LUT4 path validation failed with status {status}")
        return c_paths, len(path_list), keepalive

    def lut4_matvec(
        self,
        input_values,
        paths,
        *,
        output_count: int,
        instance: int | None = None,
        columns: tuple[int, int] | None = None,
    ) -> tuple[float, ...]:
        """Deterministic FP64 LUT4-LANES matvec with optional instance and column slice."""
        values = self._finite_values(input_values)
        if not values or type(output_count) is not int or output_count < 1:
            raise ValueError("native LUT4 matvec arguments are malformed")
        if instance is not None and (type(instance) is not int or instance not in (0, 1, 2, 4)):
            raise ValueError("native LUT4 instance must be 0, 1, 2 or 4")
        c_paths, n_paths, _keepalive = self._build_lut4_paths(paths, len(values), output_count)
        in_arr = (ctypes.c_double * len(values))(*values)
        out_buf = (ctypes.c_double * output_count)()
        if columns is not None:
            if type(columns) is not tuple or len(columns) != 2:
                raise ValueError("native LUT4 columns must be a 2-tuple (begin, end)")
            col_begin, col_end = columns
            if (
                type(col_begin) is not int
                or type(col_end) is not int
                or col_begin < 0
                or col_begin >= col_end
                or col_end > output_count
            ):
                raise ValueError(
                    "native LUT4 columns must satisfy 0 <= begin < end <= output_count"
                )
            if instance is None:
                status = self._library.bcir_ai_lut4_matvec_f64_columns(
                    in_arr, len(values), c_paths, n_paths, output_count, col_begin, col_end, out_buf
                )
            else:
                status = self._library.bcir_ai_lut4_matvec_f64_columns_instance(
                    instance,
                    in_arr,
                    len(values),
                    c_paths,
                    n_paths,
                    output_count,
                    col_begin,
                    col_end,
                    out_buf,
                )
            if status:
                raise RuntimeError(f"native LUT4 matvec failed with status {status}")
            return tuple(float(v) for v in out_buf[col_begin:col_end])
        if instance is None:
            status = self._library.bcir_ai_lut4_matvec_f64(
                in_arr, len(values), c_paths, n_paths, output_count, out_buf
            )
        else:
            status = self._library.bcir_ai_lut4_matvec_f64_instance(
                instance, in_arr, len(values), c_paths, n_paths, output_count, out_buf
            )
        if status:
            raise RuntimeError(f"native LUT4 matvec failed with status {status}")
        return tuple(float(v) for v in out_buf)

    def lut4_matmul_mb4(
        self,
        inputs,
        paths,
        *,
        output_count: int,
        instance: int = 0,
    ) -> tuple[tuple[float, ...], ...]:
        """Micro-batched (mb4) FP64 LUT4-LANES prefill, bit-identical to single-token matvec."""
        rows = tuple(self._finite_values(row) for row in inputs)
        if not rows or not rows[0] or type(output_count) is not int or output_count < 1:
            raise ValueError("native LUT4 mb4 arguments are malformed")
        input_count = len(rows[0])
        if any(len(row) != input_count for row in rows):
            raise ValueError("native LUT4 mb4 input rows must have equal length")
        if type(instance) is not int or instance not in (0, 1, 2, 4):
            raise ValueError("native LUT4 instance must be 0, 1, 2 or 4")
        c_paths, n_paths, _keepalive = self._build_lut4_paths(paths, input_count, output_count)
        flat = [v for row in rows for v in row]
        in_arr = (ctypes.c_double * len(flat))(*flat)
        out_buf = (ctypes.c_double * (len(rows) * output_count))()
        status = self._library.bcir_ai_lut4_matmul_mb4_f64_instance(
            instance, in_arr, len(rows), input_count, c_paths, n_paths, output_count, out_buf
        )
        if status:
            raise RuntimeError(f"native LUT4 mb4 matmul failed with status {status}")
        return tuple(
            tuple(float(out_buf[t * output_count + j]) for j in range(output_count))
            for t in range(len(rows))
        )

    def lut4_matvec_f32(
        self,
        input_values,
        paths,
        *,
        output_count: int,
        instance: int = 0,
        columns: tuple[int, int] | None = None,
    ) -> tuple[float, ...]:
        """Deterministic FP32 LUT4-LANES matvec (portable / AVX2 VPERMPS / AVX-512 PERMUTEXVAR)."""
        values = self._finite_values(input_values)
        if not values or type(output_count) is not int or output_count < 1:
            raise ValueError("native LUT4 f32 matvec arguments are malformed")
        if type(instance) is not int or instance not in (0, 1, 2, 4):
            raise ValueError("native LUT4 instance must be 0, 1, 2 or 4")
        c_paths, n_paths, _keepalive = self._build_lut4_paths(paths, len(values), output_count)
        in_arr = (ctypes.c_float * len(values))(*values)
        out_buf = (ctypes.c_float * output_count)()
        if columns is not None:
            if type(columns) is not tuple or len(columns) != 2:
                raise ValueError("native LUT4 columns must be a 2-tuple (begin, end)")
            col_begin, col_end = columns
            if (
                type(col_begin) is not int
                or type(col_end) is not int
                or col_begin < 0
                or col_begin >= col_end
                or col_end > output_count
            ):
                raise ValueError(
                    "native LUT4 columns must satisfy 0 <= begin < end <= output_count"
                )
            status = self._library.bcir_ai_lut4_matvec_f32_columns_instance(
                instance,
                in_arr,
                len(values),
                c_paths,
                n_paths,
                output_count,
                col_begin,
                col_end,
                out_buf,
            )
            if status:
                raise RuntimeError(f"native LUT4 f32 matvec failed with status {status}")
            return tuple(float(v) for v in out_buf[col_begin:col_end])
        status = self._library.bcir_ai_lut4_matvec_f32_instance(
            instance, in_arr, len(values), c_paths, n_paths, output_count, out_buf
        )
        if status:
            raise RuntimeError(f"native LUT4 f32 matvec failed with status {status}")
        return tuple(float(v) for v in out_buf)

    def lut4_matmul_mb4_f32(
        self,
        inputs,
        paths,
        *,
        output_count: int,
        instance: int = 0,
    ) -> tuple[tuple[float, ...], ...]:
        """Micro-batched (mb4) FP32 LUT4-LANES prefill, bit-identical to single-token matvec."""
        rows = tuple(self._finite_values(row) for row in inputs)
        if not rows or not rows[0] or type(output_count) is not int or output_count < 1:
            raise ValueError("native LUT4 mb4 f32 arguments are malformed")
        input_count = len(rows[0])
        if any(len(row) != input_count for row in rows):
            raise ValueError("native LUT4 mb4 f32 input rows must have equal length")
        if type(instance) is not int or instance not in (0, 1, 2, 4):
            raise ValueError("native LUT4 instance must be 0, 1, 2 or 4")
        c_paths, n_paths, _keepalive = self._build_lut4_paths(paths, input_count, output_count)
        flat = [v for row in rows for v in row]
        in_arr = (ctypes.c_float * len(flat))(*flat)
        out_buf = (ctypes.c_float * (len(rows) * output_count))()
        status = self._library.bcir_ai_lut4_matmul_mb4_f32_instance(
            instance, in_arr, len(rows), input_count, c_paths, n_paths, output_count, out_buf
        )
        if status:
            raise RuntimeError(f"native LUT4 mb4 f32 matmul failed with status {status}")
        return tuple(
            tuple(float(out_buf[t * output_count + j]) for j in range(output_count))
            for t in range(len(rows))
        )

    def _q15_topk(
        self,
        query: array,
        patterns: array,
        eligible: bytes,
        pattern_count: int,
        dimension: int,
        top_k: int,
    ) -> tuple[tuple[int, int], ...]:
        query_view = (ctypes.c_int16 * len(query)).from_buffer(query)
        pattern_view = (ctypes.c_int16 * len(patterns)).from_buffer(patterns)
        eligible_view = (ctypes.c_uint8 * len(eligible)).from_buffer_copy(eligible)
        matches = (_CMatch * top_k)()
        count = ctypes.c_size_t(0)
        status = self._library.bcir_ai_q15_topk(
            query_view,
            pattern_view,
            eligible_view,
            pattern_count,
            dimension,
            top_k,
            matches,
            top_k,
            ctypes.byref(count),
        )
        if status:
            raise RuntimeError(f"native Q15 top-k failed with status {status}")
        return tuple(
            (int(matches[index].index), int(matches[index].squared_distance))
            for index in range(count.value)
        )


class NativeOptimizationIndex:
    """One-time contiguous native projection of immutable optimization memory."""

    def __init__(self, kernels: NativeAIKernels, memory: OptimizationMemory) -> None:
        if not isinstance(kernels, NativeAIKernels) or not isinstance(memory, OptimizationMemory):
            raise ValueError("native optimization index needs typed kernels and memory")
        self.kernels = kernels
        self.memory = memory
        self._patterns = array(
            "h", (coordinate for pattern in memory.patterns for coordinate in pattern.embedding_q15)
        )

    def query(
        self, embedding_q15, *, top_k: int = 1, active_facts=None
    ) -> tuple[OptimizationMatch, ...]:
        values, eligible = _prepare_optimization_query(
            self.memory, embedding_q15, top_k=top_k, active_facts=active_facts
        )
        query = array("h", values)
        matches = self.kernels._q15_topk(
            query,
            self._patterns,
            bytes(eligible),
            len(self.memory.patterns),
            self.memory.dimension,
            top_k,
        )
        return tuple(
            OptimizationMatch(
                self.memory.patterns[index].pattern_sha256,
                self.memory.patterns[index].context_shard_sha256,
                distance,
            )
            for index, distance in matches
        )


__all__ = ["NativeAIKernels", "NativeOptimizationIndex", "NativeQuantizedTensor"]
