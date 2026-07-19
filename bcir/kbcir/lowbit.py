"""A1 packed Q4 tensor contract, activation calibration, and admission evidence.

``BCIRQ4T v1`` is a little-endian, single-tensor reference format.  Codes are signed
two's-complement nibbles in low-nibble-first order; ``-8`` is forbidden so the oracle's
symmetric ``[-7, 7]`` law remains exact.  Group size is fixed at 32 and each group has
one signed int16 power-of-two exponent.  This does not silently reinterpret BCIRQ8 or
claim whole-decoder Q4 support.
"""
from __future__ import annotations

import hashlib
import math
import os
import statistics
import struct
import tempfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .quantize import QGroup, dequantize, quantize_per_group, scaled_dot


_MAGIC = b"BCIRQ4T\0"
_VERSION = 1
_GROUP_SIZE = 32
_HEADER = struct.Struct("<8sHHIIII4IQQII32s32s28x")
_HEADER_BYTES = 160
_FLAG_SIGNED_SYMMETRIC = 1
_MAX_ELEMENTS = 1 << 26
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def _sha256(value, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower() \
            or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _checked_shape(shape) -> tuple[int, ...]:
    if not isinstance(shape, (tuple, list)) or not 1 <= len(shape) <= 4 \
            or any(type(dim) is not int or not 1 <= dim <= 0xFFFFFFFF for dim in shape):
        raise ValueError("Q4 tensor shape must have one to four positive u32 dimensions")
    count = 1
    for dim in shape:
        count *= dim
        if count > _MAX_ELEMENTS:
            raise ValueError("Q4 tensor element count exceeds the format limit")
    return tuple(shape)


def pack_signed_int4(codes) -> bytes:
    values = tuple(codes)
    if any(type(code) is not int or not -7 <= code <= 7 for code in values):
        raise ValueError("Q4 codes must be integer values in [-7, 7]")
    output = bytearray((len(values) + 1) // 2)
    for index, code in enumerate(values):
        nibble = code & 0xF
        if index & 1:
            output[index // 2] |= nibble << 4
        else:
            output[index // 2] = nibble
    return bytes(output)


def unpack_signed_int4(data, count: int) -> tuple[int, ...]:
    if type(count) is not int or count < 0 or not isinstance(data, (bytes, bytearray, memoryview)) \
            or len(data) != (count + 1) // 2:
        raise ValueError("packed Q4 byte count does not match the element count")
    raw = bytes(data)
    if count & 1 and raw and raw[-1] & 0xF0:
        raise ValueError("packed Q4 odd-element padding nibble must be zero")
    values = []
    for index in range(count):
        nibble = raw[index // 2] >> (4 if index & 1 else 0) & 0xF
        code = nibble - 16 if nibble & 8 else nibble
        if code == -8:
            raise ValueError("packed Q4 contains forbidden non-symmetric code -8")
        values.append(code)
    return tuple(values)


@dataclass(frozen=True)
class PackedQ4Tensor:
    shape: tuple[int, ...]
    exponents: tuple[int, ...]
    packed_codes: bytes
    source_sha256: str
    calibration_sha256: str
    group_size: int = _GROUP_SIZE

    def __post_init__(self) -> None:
        shape = _checked_shape(self.shape); object.__setattr__(self, "shape", shape)
        if self.group_size != _GROUP_SIZE:
            raise ValueError("BCIRQ4T v1 requires group_size=32")
        count = math.prod(shape); groups = (count + _GROUP_SIZE - 1) // _GROUP_SIZE
        if not isinstance(self.exponents, (tuple, list)) or len(self.exponents) != groups \
                or any(type(value) is not int or not -32768 <= value <= 32767
                       for value in self.exponents):
            raise ValueError("Q4 exponent table does not match the tensor groups")
        object.__setattr__(self, "exponents", tuple(self.exponents))
        if not isinstance(self.packed_codes, bytes):
            raise ValueError("packed_codes must be immutable bytes")
        unpack_signed_int4(self.packed_codes, count)
        _sha256(self.source_sha256, "source_sha256")
        _sha256(self.calibration_sha256, "calibration_sha256")

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def codes(self) -> tuple[int, ...]:
        return unpack_signed_int4(self.packed_codes, self.element_count)

    def groups(self) -> tuple[QGroup, ...]:
        codes = self.codes
        return tuple(QGroup(tuple(codes[index:index + self.group_size]), exponent, 4)
                     for index, exponent in zip(range(0, len(codes), self.group_size),
                                                self.exponents))

    def dequantize(self) -> list[float]:
        return dequantize(self.groups())

    @classmethod
    def from_values(cls, values, shape, *, source_sha256: str,
                    calibration_sha256: str) -> "PackedQ4Tensor":
        shape = _checked_shape(shape); values = tuple(values)
        if len(values) != math.prod(shape):
            raise ValueError("Q4 values length does not match shape")
        groups = quantize_per_group(values, _GROUP_SIZE, 4)
        return cls(shape, tuple(group.scale_exp for group in groups),
                   pack_signed_int4(code for group in groups for code in group.codes),
                   source_sha256, calibration_sha256)

    @classmethod
    def from_values_native(cls, values, shape, *, source_sha256: str,
                           calibration_sha256: str, native_kernels) -> "PackedQ4Tensor":
        """Build BCIRQ4T through the explicit C data plane, with no fallback."""
        from .native_ai import NativeAIKernels
        if not isinstance(native_kernels, NativeAIKernels):
            raise ValueError("native_kernels must be a NativeAIKernels instance")
        shape = _checked_shape(shape); values = tuple(values)
        if len(values) != math.prod(shape):
            raise ValueError("Q4 values length does not match shape")
        native = native_kernels.quantize(values, group_size=_GROUP_SIZE, bits=4)
        return cls(shape, native.exponents, native.codes,
                   source_sha256, calibration_sha256)


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _encode_q4(tensor: PackedQ4Tensor) -> bytes:
    if not isinstance(tensor, PackedQ4Tensor):
        raise ValueError("tensor must be a PackedQ4Tensor")
    exponent_bytes = struct.pack(f"<{len(tensor.exponents)}h", *tensor.exponents)
    exponent_offset = _HEADER_BYTES
    code_offset = _align8(exponent_offset + len(exponent_bytes))
    body = exponent_bytes + bytes(code_offset - exponent_offset - len(exponent_bytes)) \
        + tensor.packed_codes
    body_crc = zlib.crc32(body) & 0xFFFFFFFF
    dimensions = (*tensor.shape, *(0 for _ in range(4 - len(tensor.shape))))
    arguments = (_MAGIC, _VERSION, _HEADER_BYTES, _FLAG_SIGNED_SYMMETRIC,
                 len(tensor.shape), tensor.group_size, tensor.element_count, *dimensions,
                 exponent_offset, code_offset, body_crc)
    header = _HEADER.pack(*arguments, 0, bytes.fromhex(tensor.source_sha256),
                          bytes.fromhex(tensor.calibration_sha256))
    header_crc = zlib.crc32(header) & 0xFFFFFFFF
    header = _HEADER.pack(*arguments, header_crc, bytes.fromhex(tensor.source_sha256),
                          bytes.fromhex(tensor.calibration_sha256))
    return header + body


def write_packed_q4(path: os.PathLike | str, tensor: PackedQ4Tensor) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encode_q4(tensor)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_packed_q4(path: os.PathLike | str) -> PackedQ4Tensor:
    source = Path(path)
    try:
        with source.open("rb") as stream:
            raw = stream.read(_MAX_ARTIFACT_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"cannot read Q4 artifact: {exc}") from exc
    if len(raw) < _HEADER_BYTES or len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError("Q4 artifact size is outside the format limits")
    header = raw[:_HEADER_BYTES]
    fields = _HEADER.unpack(header)
    (magic, version, header_bytes, flags, rank, group_size, count, *tail) = fields
    dimensions = tuple(tail[:4]); exponent_offset, code_offset, body_crc, header_crc = tail[4:8]
    source_hash, calibration_hash = tail[8:10]
    if magic != _MAGIC or version != _VERSION or header_bytes != _HEADER_BYTES \
            or flags != _FLAG_SIGNED_SYMMETRIC or group_size != _GROUP_SIZE \
            or not 1 <= rank <= 4 or any(dim != 0 for dim in dimensions[rank:]):
        raise ValueError("unsupported or malformed Q4 header")
    # The header CRC u32 immediately follows body CRC.  Repack instead of relying on
    # a magic byte offset, keeping the validation tied to the declared struct.
    zero_crc = _HEADER.pack(magic, version, header_bytes, flags, rank, group_size, count,
                            *dimensions, exponent_offset, code_offset, body_crc, 0,
                            source_hash, calibration_hash)
    if zlib.crc32(zero_crc) & 0xFFFFFFFF != header_crc:
        raise ValueError("Q4 header CRC mismatch")
    shape = _checked_shape(dimensions[:rank])
    if math.prod(shape) != count:
        raise ValueError("Q4 shape and element count disagree")
    groups = (count + _GROUP_SIZE - 1) // _GROUP_SIZE
    exponent_bytes = groups * 2; code_bytes = (count + 1) // 2
    if exponent_offset != _HEADER_BYTES or code_offset != _align8(exponent_offset + exponent_bytes) \
            or code_offset + code_bytes != len(raw):
        raise ValueError("Q4 offsets or file extent are non-canonical")
    if any(raw[exponent_offset + exponent_bytes:code_offset]):
        raise ValueError("Q4 alignment padding must be zero")
    body = raw[_HEADER_BYTES:]
    if zlib.crc32(body) & 0xFFFFFFFF != body_crc:
        raise ValueError("Q4 body CRC mismatch")
    exponents = struct.unpack(f"<{groups}h", raw[exponent_offset:exponent_offset + exponent_bytes])
    codes = raw[code_offset:]
    return PackedQ4Tensor(shape, exponents, codes, source_hash.hex(), calibration_hash.hex())


@dataclass(frozen=True)
class SmoothQuantPolicy:
    alpha: float = 0.5
    outlier_threshold: float = 8.0
    max_outlier_channels: int = 8
    outlier_mode: str = "fp32_residual"

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)) \
                or not math.isfinite(float(self.alpha)) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("SmoothQuant alpha must be finite in [0, 1]")
        if isinstance(self.outlier_threshold, bool) \
                or not isinstance(self.outlier_threshold, (int, float)) \
                or not math.isfinite(float(self.outlier_threshold)) \
                or self.outlier_threshold <= 1.0:
            raise ValueError("outlier_threshold must be finite and greater than 1")
        if type(self.max_outlier_channels) is not int or self.max_outlier_channels < 0:
            raise ValueError("max_outlier_channels must be a non-negative integer")
        if self.outlier_mode not in ("fp32_residual", "reject"):
            raise ValueError("outlier_mode must be fp32_residual or reject")


@dataclass(frozen=True)
class SmoothQuantCalibration:
    channel_scale_exponents: tuple[int, ...]
    outlier_channels: tuple[int, ...]
    activation_sha256: str
    weight_sha256: str
    policy_sha256: str

    @property
    def digest(self) -> str:
        payload = repr((self.channel_scale_exponents, self.outlier_channels,
                        self.activation_sha256, self.weight_sha256, self.policy_sha256))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @property
    def scales(self) -> tuple[float, ...]:
        return tuple(math.ldexp(1.0, exponent) for exponent in self.channel_scale_exponents)


def _matrix(value, name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a nonempty matrix")
    rows = []
    width = None
    for row in value:
        if not isinstance(row, (tuple, list)) or not row:
            raise ValueError(f"{name} rows must be nonempty sequences")
        values = tuple(float(item) for item in row)
        if any(not math.isfinite(item) for item in values):
            raise ValueError(f"{name} contains a non-finite value")
        if width is None:
            width = len(values)
        if len(values) != width:
            raise ValueError(f"{name} is ragged")
        rows.append(values)
    return tuple(rows)


def _matrix_digest(matrix) -> str:
    digest = hashlib.sha256()
    for row in matrix:
        for value in row:
            digest.update(float(value).hex().encode("ascii")); digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def calibrate_smoothquant(activations, weights,
                          policy: SmoothQuantPolicy = SmoothQuantPolicy()) -> SmoothQuantCalibration:
    activations = _matrix(activations, "activations"); weights = _matrix(weights, "weights")
    if not isinstance(policy, SmoothQuantPolicy) or len(activations[0]) != len(weights[0]):
        raise ValueError("calibration policy or activation/weight channel dimensions are invalid")
    width = len(activations[0])
    activation_max = [max(abs(row[index]) for row in activations) for index in range(width)]
    weight_max = [max(abs(row[index]) for row in weights) for index in range(width)]
    nonzero = [value for value in activation_max if value > 0.0]
    median = statistics.median(nonzero) if nonzero else 1.0
    candidates = [index for index, value in enumerate(activation_max)
                  if value > median * policy.outlier_threshold]
    candidates.sort(key=lambda index: (-activation_max[index], index))
    if candidates and policy.outlier_mode == "reject":
        raise ValueError("activation calibration found outlier channels under reject policy")
    if len(candidates) > policy.max_outlier_channels:
        raise ValueError("activation outlier count exceeds max_outlier_channels")
    exponents = []
    for act, weight in zip(activation_max, weight_max):
        if act == 0.0 or weight == 0.0:
            exponents.append(0); continue
        scale = act ** float(policy.alpha) / weight ** (1.0 - float(policy.alpha))
        exponents.append(max(-32, min(32, int(round(math.log2(scale))))))
    policy_sha = hashlib.sha256(repr(asdict(policy)).encode("ascii")).hexdigest()
    return SmoothQuantCalibration(tuple(exponents), tuple(candidates),
                                  _matrix_digest(activations), _matrix_digest(weights),
                                  policy_sha)


def calibrated_q4q8_dot(activation, weight, calibration: SmoothQuantCalibration) -> float:
    if not isinstance(calibration, SmoothQuantCalibration):
        raise ValueError("calibration must be a SmoothQuantCalibration")
    activation = tuple(float(value) for value in activation)
    weight = tuple(float(value) for value in weight)
    if len(activation) != len(weight) or len(activation) != len(calibration.scales) \
            or any(not math.isfinite(value) for value in activation + weight):
        raise ValueError("dot operands do not match the calibration")
    outliers = set(calibration.outlier_channels); smooth_activation = []; smooth_weight = []
    residual = 0.0
    for index, (left, right, scale) in enumerate(zip(activation, weight, calibration.scales)):
        if index in outliers:
            residual += left * right; smooth_activation.append(0.0); smooth_weight.append(0.0)
        else:
            smooth_activation.append(left / scale); smooth_weight.append(right * scale)
    left_groups = quantize_per_group(smooth_activation, _GROUP_SIZE, 8)
    right_groups = quantize_per_group(smooth_weight, _GROUP_SIZE, 4)
    return residual + scaled_dot(left_groups, right_groups)


@dataclass(frozen=True)
class FormatAdmissionReport:
    format_name: str
    artifact_bytes: int
    reference_bytes: int
    max_logit_error: float
    nll_delta: float
    source_sha256: str
    calibration_sha256: str
    accepted: bool
    reasons: tuple[str, ...]


def admit_low_bit_format(format_name: str, *, artifact_bytes: int, reference_bytes: int,
                         max_logit_error: float, nll_delta: float, source_sha256: str,
                         calibration_sha256: str, max_size_ratio: float,
                         max_error: float, max_nll_delta: float) -> FormatAdmissionReport:
    _sha256(source_sha256, "source_sha256"); _sha256(calibration_sha256, "calibration_sha256")
    if format_name not in ("bcirq4", "mx", "fp8", "nf4", "gguf"):
        raise ValueError("unknown low-bit format family")
    numeric = (artifact_bytes, reference_bytes)
    if any(type(value) is not int or value <= 0 for value in numeric):
        raise ValueError("artifact and reference sizes must be positive integers")
    for value in (max_logit_error, nll_delta, max_size_ratio, max_error, max_nll_delta):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("admission thresholds and measurements must be finite")
    if max_logit_error < 0.0 or max_size_ratio <= 0.0 or max_error < 0.0 \
            or max_nll_delta < 0.0:
        raise ValueError("admission magnitudes must be non-negative and size ratio positive")
    reasons = []
    if format_name != "bcirq4":
        reasons.append("format implementation is not present")
    if artifact_bytes / reference_bytes > max_size_ratio:
        reasons.append("compactness threshold exceeded")
    if abs(max_logit_error) > max_error:
        reasons.append("logit drift threshold exceeded")
    if nll_delta > max_nll_delta:
        reasons.append("NLL delta threshold exceeded")
    return FormatAdmissionReport(format_name, artifact_bytes, reference_bytes,
                                 float(max_logit_error), float(nll_delta), source_sha256,
                                 calibration_sha256, not reasons, tuple(reasons))
