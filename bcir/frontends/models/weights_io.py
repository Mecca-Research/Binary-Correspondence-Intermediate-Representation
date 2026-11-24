"""Deterministic compact Q8 persistence for the reference Llama decoder.

``BCIRQ8 v1`` is deliberately small in scope: Llama/SwiGLU decoder weights,
signed per-group Q8 codes, and power-of-two scales.  It is a storage format,
not a second numerical oracle.  Reading an artifact reconstructs the exact
dequantized values produced by :func:`bcir.kbcir.quantize.quantize_per_group`.

The file is little-endian and self-contained.  A fixed header describes the
model, a fixed-size directory describes every tensor, and three levels of CRC
protect the header, body, and individual tensor payloads.  Source SHA-256
digests tie the derived artifact to its checkpoint, config, and tokenizer.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import stat
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from ...kbcir.quantize import quantize_per_group
from ...kbcir.unsupervised import EmbeddingTable
from .decode import DecoderSpec, DecoderWeights, LayerWeights, check_decoder_weights

if TYPE_CHECKING:
    from ...kbcir.native_ai import NativeAIKernels

MAGIC = b"BCIRQ8\0\0"
VERSION = 1
VERSION_V2 = 2
HEADER_SIZE = 224
DIRECTORY_ENTRY_SIZE = 48
ENDIAN_MARKER = 0x01020304
BITS = 8
DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_DECODED_ELEMENTS = 8_000_000

FLAG_TIED_EMBEDDINGS = 1 << 0
FLAG_HYBRID_LUT4_MLP = 1 << 1

TENSOR_ENC_Q8 = 0
TENSOR_ENC_LUT4_1B = 1
LUT4_MAX_PATHS = 2
LUT4_MAX_RANK = 1024

# Stable tensor ids.  Layer-local ids repeat with a distinct signed layer index.
TENSOR_EMBEDDING = 1
TENSOR_G_ATTN = 10
TENSOR_W_Q = 11
TENSOR_W_K = 12
TENSOR_W_V = 13
TENSOR_W_O = 14
TENSOR_G_FF = 15
TENSOR_W_GATE = 16
TENSOR_W_UP = 17
TENSOR_W_DOWN = 18
TENSOR_G_FINAL = 100
TENSOR_LM_HEAD = 101

# Container-level policy law: LUT4_1B is permitted ONLY on MLP projections.
LUT4_ALLOWED_TENSOR_IDS = frozenset({TENSOR_W_GATE, TENSOR_W_UP, TENSOR_W_DOWN})

_HEADER_PREFIX = struct.Struct("<8sHHIIHBB6I Iiii dd II QQQ II")
_DIRECTORY = struct.Struct("<HhBBHIIIIIIQQ")
assert _HEADER_PREFIX.size == 120
assert _DIRECTORY.size == DIRECTORY_ENTRY_SIZE


def lut4_packed_bytes(rows: int, cols: int) -> int:
    """Exact byte length of a packed 1-bit LUT4 sign matrix [rows, cols].

    Matches ``bcir_ai_lut4_packed_bytes`` bit-for-bit: 32-row blocks (two
    nibbles per byte) over ``ceil(cols / 4)`` 4-column LUT groups.
    """
    if type(rows) is not int or type(cols) is not int or rows < 1 or cols < 1:
        raise ValueError("LUT4 dimensions must be positive integers")
    rows32 = (rows + 31) & ~31
    bpg = rows32 >> 1
    groups4 = (cols + 3) >> 2
    return groups4 * bpg


def pack_lut4_signs(signs, rows: int, cols: int) -> bytes:
    """Pack row-major +1/-1 signs [rows, cols] into canonical LUT4 nibble layout."""
    required = lut4_packed_bytes(rows, cols)
    flat = tuple(int(v) for v in signs)
    if len(flat) != rows * cols or any(v not in (-1, 1) for v in flat):
        raise ValueError("LUT4 signs must contain rows*cols values in {-1, +1}")
    out = bytearray(required)
    bpg = ((rows + 31) & ~31) >> 1
    for r in range(rows):
        blk32 = r >> 5
        lane = r & 15
        is_hi = (r >> 4) & 1
        row_base = r * cols
        slot = (blk32 << 4) + lane
        for c in range(cols):
            if flat[row_base + c] < 0:
                bit = 1 << (c & 3)
                off = (c >> 2) * bpg + slot
                out[off] |= (bit << 4) if is_hi else bit
    return bytes(out)


def validate_lut4_packed(packed: bytes, rows: int, cols: int) -> None:
    """Verify exact byte length and canonical zero padding bits for a packed LUT4 matrix."""
    required = lut4_packed_bytes(rows, cols)
    if not isinstance(packed, (bytes, bytearray, memoryview)) or len(packed) != required:
        raise ValueError(f"LUT4 packed buffer length {len(packed)} != required {required}")
    raw = memoryview(packed)
    rows32 = (rows + 31) & ~31
    bpg = rows32 >> 1
    groups4 = (cols + 3) >> 2
    if rows < rows32:
        for g in range(groups4):
            base = g * bpg
            for r in range(rows, rows32):
                blk32 = r >> 5
                lane = r & 15
                byte = raw[base + (blk32 << 4) + lane]
                nib = (byte >> 4) if ((r >> 4) & 1) else (byte & 0x0F)
                if nib != 0:
                    raise ValueError("LUT4 packed buffer has nonzero padding row bits")
    tail = cols & 3
    if tail:
        pad_mask = 0x0F & ~((1 << tail) - 1)
        base = (groups4 - 1) * bpg
        for idx in range(bpg):
            byte = raw[base + idx]
            if ((byte & 0x0F) & pad_mask) or (((byte >> 4) & 0x0F) & pad_mask):
                raise ValueError("LUT4 packed buffer has nonzero trailing column bits")


def unpack_lut4_signs(packed: bytes, rows: int, cols: int) -> tuple[int, ...]:
    """Unpack canonical LUT4 bytes into row-major +1/-1 signs [rows, cols]."""
    validate_lut4_packed(packed, rows, cols)
    raw = memoryview(packed)
    bpg = ((rows + 31) & ~31) >> 1
    out = [1] * (rows * cols)
    for r in range(rows):
        blk32 = r >> 5
        lane = r & 15
        is_hi = (r >> 4) & 1
        slot = (blk32 << 4) + lane
        row_base = r * cols
        for c in range(cols):
            byte = raw[(c >> 2) * bpg + slot]
            nib = (byte >> 4) if is_hi else (byte & 0x0F)
            if nib & (1 << (c & 3)):
                out[row_base + c] = -1
    return tuple(out)


def encode_fp16_le(values) -> bytes:
    """Pack finite numeric values into little-endian IEEE-754 binary16 bytes."""
    buf = bytearray()
    for val in values:
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError("LUT4 binary16 scale must be finite")
        try:
            packed = struct.pack("<e", fval)
        except OverflowError as exc:
            raise ValueError(f"LUT4 scale {fval!r} overflows IEEE-754 binary16") from exc
        if not math.isfinite(struct.unpack("<e", packed)[0]):
            raise ValueError(f"LUT4 scale {fval!r} rounded to non-finite binary16")
        buf.extend(packed)
    return bytes(buf)


def decode_fp16_le(raw: bytes) -> tuple[float, ...]:
    """Decode little-endian IEEE-754 binary16 bytes, rejecting NaN/Inf."""
    if not isinstance(raw, (bytes, bytearray, memoryview)) or len(raw) % 2:
        raise ValueError("LUT4 binary16 buffer must have even byte length")
    values = tuple(val for (val,) in struct.iter_unpack("<e", raw))
    if any(not math.isfinite(val) for val in values):
        raise ValueError("LUT4 binary16 scales contain non-finite NaN or Inf")
    return values


@dataclass(frozen=True)
class Lut4Path:
    """One packed-1-bit factorized path: W_p = diag(h) @ U @ diag(l) @ V @ diag(g)."""

    rank: int
    v_packed: bytes  # [rank, in_dim]
    u_packed: bytes  # [out_dim, rank]
    g_fp16_le: bytes  # in_dim * 2 bytes
    l_fp16_le: bytes  # rank * 2 bytes
    h_fp16_le: bytes  # out_dim * 2 bytes

    def validate(self, in_dim: int, out_dim: int) -> None:
        if (
            type(self.rank) is not int
            or self.rank < 1
            or self.rank > LUT4_MAX_RANK
            or self.rank > min(in_dim, out_dim)
        ):
            raise ValueError(
                f"LUT4 rank {self.rank} must be in [1, min({in_dim}, {out_dim}, {LUT4_MAX_RANK})]"
            )
        validate_lut4_packed(self.v_packed, self.rank, in_dim)
        validate_lut4_packed(self.u_packed, out_dim, self.rank)
        if (
            len(self.g_fp16_le) != in_dim * 2
            or len(self.l_fp16_le) != self.rank * 2
            or len(self.h_fp16_le) != out_dim * 2
        ):
            raise ValueError("LUT4 binary16 scale byte lengths do not match dimensions")
        decode_fp16_le(self.g_fp16_le)
        decode_fp16_le(self.l_fp16_le)
        decode_fp16_le(self.h_fp16_le)

    @property
    def scales_blob(self) -> bytes:
        return self.g_fp16_le + self.l_fp16_le + self.h_fp16_le

    @property
    def packed_blob(self) -> bytes:
        return self.v_packed + self.u_packed


@dataclass(frozen=True)
class Lut4Tensor:
    """One container-validated MLP projection encoded in BCIRQ8 v2 LUT4_1B."""

    tensor_id: int
    layer: int
    in_dim: int
    out_dim: int
    paths: tuple[Lut4Path, ...]

    def __post_init__(self) -> None:
        if self.tensor_id not in LUT4_ALLOWED_TENSOR_IDS:
            raise ValueError(
                f"LUT4_1B is permitted only on MLP tensors {sorted(LUT4_ALLOWED_TENSOR_IDS)}; "
                f"refused tensor_id={self.tensor_id}"
            )
        if type(self.layer) is not int or not 0 <= self.layer <= 0x7FFF:
            raise ValueError("LUT4_1B tensor layer must be in [0, 32767]")
        if (
            type(self.in_dim) is not int
            or self.in_dim < 1
            or type(self.out_dim) is not int
            or self.out_dim < 1
        ):
            raise ValueError("LUT4_1B dimensions must be positive integers")
        if not isinstance(self.paths, tuple) or not 1 <= len(self.paths) <= LUT4_MAX_PATHS:
            raise ValueError(f"LUT4_1B requires 1..{LUT4_MAX_PATHS} Lut4Path entries")
        for path in self.paths:
            if not isinstance(path, Lut4Path):
                raise ValueError("LUT4_1B paths must be Lut4Path instances")
            path.validate(self.in_dim, self.out_dim)

    @property
    def scales_blob(self) -> bytes:
        return b"".join(p.scales_blob for p in self.paths)

    @property
    def packed_blob(self) -> bytes:
        return b"".join(p.packed_blob for p in self.paths)

    @property
    def scales_bytes(self) -> int:
        return sum(2 * (self.in_dim + p.rank + self.out_dim) for p in self.paths)

    @property
    def packed_bytes(self) -> int:
        return sum(len(p.v_packed) + len(p.u_packed) for p in self.paths)

    @property
    def total_bytes(self) -> int:
        return self.scales_bytes + self.packed_bytes

    @property
    def packed_ranks_u32(self) -> int:
        r0 = self.paths[0].rank
        r1 = self.paths[1].rank if len(self.paths) > 1 else 0
        return r0 | (r1 << 16)


def _lut4_build_table(x0: float, x1: float, x2: float, x3: float) -> tuple[float, ...]:
    s01_pp = x0 + x1
    s01_mp = -x0 + x1
    s01_pm = x0 - x1
    s01_mm = -x0 - x1
    s23_pp = x2 + x3
    s23_mp = -x2 + x3
    s23_pm = x2 - x3
    s23_mm = -x2 - x3
    return (
        s01_pp + s23_pp,
        s01_mp + s23_pp,
        s01_pm + s23_pp,
        s01_mm + s23_pp,
        s01_pp + s23_mp,
        s01_mp + s23_mp,
        s01_pm + s23_mp,
        s01_mm + s23_mp,
        s01_pp + s23_pm,
        s01_mp + s23_pm,
        s01_pm + s23_pm,
        s01_mm + s23_pm,
        s01_pp + s23_mm,
        s01_mp + s23_mm,
        s01_pm + s23_mm,
        s01_mm + s23_mm,
    )


def _lut4_gemv_ref(packed: bytes, rows: int, cols: int, x: list[float]) -> list[float]:
    groups4 = (cols + 3) >> 2
    bpg = ((rows + 31) & ~31) >> 1
    luts = []
    for g in range(groups4):
        c0 = g * 4
        x0 = x[c0] if c0 < cols else 0.0
        x1 = x[c0 + 1] if c0 + 1 < cols else 0.0
        x2 = x[c0 + 2] if c0 + 2 < cols else 0.0
        x3 = x[c0 + 3] if c0 + 3 < cols else 0.0
        luts.append(_lut4_build_table(x0, x1, x2, x3))
    out = [0.0] * rows
    for r in range(rows):
        blk32 = r >> 5
        lane = r & 15
        is_hi = (r >> 4) & 1
        slot = (blk32 << 4) + lane
        acc = 0.0
        for g in range(groups4):
            byte = packed[g * bpg + slot]
            nib = (byte >> 4) if is_hi else (byte & 0x0F)
            acc += luts[g][nib]
        out[r] = acc
    return out


def lut4_matvec_reference(
    input_values, in_dim: int, out_dim: int, paths: tuple[Lut4Path, ...]
) -> tuple[float, ...]:
    """Deterministic Python oracle matching ``bcir_ai_lut4_matvec_f64`` bit-for-bit."""
    x = [float(v) for v in input_values]
    if len(x) != in_dim or any(not math.isfinite(v) for v in x):
        raise ValueError("LUT4 matvec input must have in_dim finite floats")
    out = [0.0] * out_dim
    for path in paths:
        g = decode_fp16_le(path.g_fp16_le)
        lam = decode_fp16_le(path.l_fp16_le)
        h = decode_fp16_le(path.h_fp16_le)
        scaled = [x[i] * g[i] for i in range(in_dim)]
        latent = _lut4_gemv_ref(path.v_packed, path.rank, in_dim, scaled)
        for r in range(path.rank):
            latent[r] *= lam[r]
        proj = _lut4_gemv_ref(path.u_packed, out_dim, path.rank, latent)
        for j in range(out_dim):
            val = out[j] + proj[j] * h[j]
            if not math.isfinite(val):
                raise ValueError("LUT4 matvec produced non-finite output")
            out[j] = val
    return tuple(out)


def reconstruct_lut4_matrix(tensor: Lut4Tensor) -> tuple[float, ...]:
    """Reconstruct dense row-major [in_dim, out_dim] matrix for a Lut4Tensor."""
    in_dim, out_dim = tensor.in_dim, tensor.out_dim
    mat = [0.0] * (in_dim * out_dim)
    for path in tensor.paths:
        g = decode_fp16_le(path.g_fp16_le)
        lam = decode_fp16_le(path.l_fp16_le)
        h = decode_fp16_le(path.h_fp16_le)
        v = unpack_lut4_signs(path.v_packed, path.rank, in_dim)
        u = unpack_lut4_signs(path.u_packed, out_dim, path.rank)
        for i in range(in_dim):
            gi = g[i]
            row_off = i * out_dim
            v_col = [v[r * in_dim + i] * lam[r] * gi for r in range(path.rank)]
            for j in range(out_dim):
                u_off = j * path.rank
                acc = 0.0
                for r in range(path.rank):
                    acc += v_col[r] * u[u_off + r]
                mat[row_off + j] += acc * h[j]
    return tuple(mat)


@dataclass(frozen=True)
class Q8ArtifactMetadata:
    version: int
    group_size: int
    bits: int
    tensor_count: int
    element_count: int
    file_size: int
    body_crc32: int
    artifact_sha256: str
    source_model_sha256: str
    source_config_sha256: str
    tokenizer_sha256: str
    context_length: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    lut4_tensor_count: int = 0


@dataclass(frozen=True)
class _TensorDef:
    tensor_id: int
    layer: int
    rank: int
    dims: tuple[int, int]
    values: tuple


@dataclass(frozen=True)
class _DirectoryEntry:
    tensor_id: int
    layer: int
    rank: int
    flags: int
    reserved: int
    dims: tuple[int, int]
    element_count: int
    group_count: int
    tensor_crc32: int
    exponent_offset: int
    code_offset: int


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _sha_bytes(value: str, field: str) -> bytes:
    if type(value) is not str:
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest") from exc
    if len(raw) != 32:
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    return raw


def _source_digest(source_hashes: Mapping[str, str], key: str) -> str:
    aliases = {
        "model": ("model", "source_model", "source_model_sha256"),
        "config": ("config", "source_config", "source_config_sha256"),
        "tokenizer": ("tokenizer", "tokenizer_sha256"),
    }
    present = [alias for alias in aliases[key] if alias in source_hashes]
    if len(present) > 1:
        raise ValueError(f"source_hashes has ambiguous aliases for {key!r}: {present}")
    if present:
        value = source_hashes[present[0]]
        if type(value) is not str:
            raise ValueError(f"source_hashes {key!r} digest must be a string")
        return value.lower()
    raise ValueError(f"source_hashes is missing {key!r}")


def _tokenizer_value(tokenizer_ids: Mapping[str, int], key: str, default: int = -1) -> int:
    aliases = {
        "bos": ("bos", "bos_token_id"),
        "eos": ("eos", "eos_token_id"),
        "pad": ("pad", "pad_token_id"),
    }
    present = [alias for alias in aliases[key] if alias in tokenizer_ids]
    if len(present) > 1:
        raise ValueError(f"tokenizer_ids has ambiguous aliases for {key!r}: {present}")
    if present:
        value = tokenizer_ids[present[0]]
        if type(value) is not int:
            raise ValueError(f"{key}_token_id must be an integer")
        return value
    return default


def _tensor_defs(spec: DecoderSpec, weights: DecoderWeights) -> list[_TensorDef]:
    if spec.activation != "silu_gate":
        raise ValueError("BCIRQ8 v1 supports only Llama/SwiGLU decoder weights")
    errors = check_decoder_weights(spec, weights)
    if errors:
        raise ValueError("invalid decoder weights: " + "; ".join(errors))
    d, f, kvd = spec.d_model, spec.d_ff, spec.kv_dim
    out = [
        _TensorDef(TENSOR_EMBEDDING, -1, 2, (spec.vocab_size, d), tuple(weights.embedding.table))
    ]
    for layer, lw in enumerate(weights.layers):
        if lw.b1 or lw.b2:
            raise ValueError("BCIRQ8 v1 Llama layers cannot contain MLP biases")
        out.extend(
            (
                _TensorDef(TENSOR_G_ATTN, layer, 1, (d, 1), tuple(lw.g_attn)),
                _TensorDef(TENSOR_W_Q, layer, 2, (d, d), tuple(lw.w_q)),
                _TensorDef(TENSOR_W_K, layer, 2, (d, kvd), tuple(lw.w_k)),
                _TensorDef(TENSOR_W_V, layer, 2, (d, kvd), tuple(lw.w_v)),
                _TensorDef(TENSOR_W_O, layer, 2, (d, d), tuple(lw.w_o)),
                _TensorDef(TENSOR_G_FF, layer, 1, (d, 1), tuple(lw.g_ff)),
                _TensorDef(TENSOR_W_GATE, layer, 2, (d, f), tuple(lw.w_gate)),
                _TensorDef(TENSOR_W_UP, layer, 2, (d, f), tuple(lw.w1)),
                _TensorDef(TENSOR_W_DOWN, layer, 2, (f, d), tuple(lw.w2)),
            )
        )
    out.append(_TensorDef(TENSOR_G_FINAL, -1, 1, (d, 1), tuple(weights.g_final)))
    if not spec.tied_embeddings:
        out.append(_TensorDef(TENSOR_LM_HEAD, -1, 2, (spec.vocab_size, d), tuple(weights.lm_head)))
    return out


def write_q8_decoder(
    path: os.PathLike | str,
    spec: DecoderSpec,
    weights: DecoderWeights,
    *,
    group_size: int = 32,
    source_hashes: Mapping[str, str],
    tokenizer_ids: Mapping[str, int],
    context_length: int | None = None,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_elements: int = DEFAULT_MAX_DECODED_ELEMENTS,
    native_kernels: NativeAIKernels | None = None,
    lut4_tensors: Mapping[tuple[int, int], Lut4Tensor] | None = None,
) -> Q8ArtifactMetadata:
    """Write one atomic, deterministic ``BCIRQ8 v1`` or ``BCIRQ8 v2`` artifact.

    When ``lut4_tensors`` is ``None`` (the default), writes an exact ``BCIRQ8 v1``
    per-group symmetric Q8 artifact bit-for-bit identically to earlier versions.
    When ``lut4_tensors`` is supplied, writes a ``BCIRQ8 v2`` hybrid container
    with ``FLAG_HYBRID_LUT4_MLP`` set, encoding only validated MLP tensors
    (``W_GATE``, ``W_UP``, ``W_DOWN``) in ``LUT4_1B`` while keeping Attention,
    embeddings, RMSNorms, and LM head in deterministic per-group Q8.
    """
    if not isinstance(spec, DecoderSpec):
        raise ValueError("spec must be a DecoderSpec")
    if not isinstance(source_hashes, Mapping) or not isinstance(tokenizer_ids, Mapping):
        raise ValueError("source_hashes and tokenizer_ids must be mappings")
    if type(group_size) is not int or group_size < 1 or group_size > 0xFFFF:
        raise ValueError("group_size must be in [1, 65535]")
    if type(max_bytes) is not int or max_bytes < HEADER_SIZE:
        raise ValueError(f"max_bytes must be an integer >= {HEADER_SIZE}")
    if type(max_elements) is not int or max_elements < 1:
        raise ValueError("max_elements must be an integer >= 1")
    if native_kernels is not None:
        from ...kbcir.native_ai import NativeAIKernels

        if not isinstance(native_kernels, NativeAIKernels):
            raise ValueError("native_kernels must be a NativeAIKernels instance")
    dimensions = {
        "vocab_size": spec.vocab_size,
        "d_model": spec.d_model,
        "n_heads": spec.n_heads,
        "n_kv_heads": spec.kv_heads,
        "n_layers": spec.n_layers,
        "d_ff": spec.d_ff,
    }
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 0xFFFFFFFF:
            raise ValueError(f"{name} must fit a positive u32")
    if spec.n_layers > 0x7FFF:
        raise ValueError("n_layers must fit the BCIRQ8 signed i16 layer field")
    d, f, kvd = spec.d_model, spec.d_ff, spec.kv_dim
    tensor_inventory: list[tuple[int, int, tuple[int, int], int]] = [
        (TENSOR_EMBEDDING, -1, (spec.vocab_size, d), spec.vocab_size * d)
    ]
    for layer in range(spec.n_layers):
        tensor_inventory.extend(
            (
                (TENSOR_G_ATTN, layer, (d, 1), d),
                (TENSOR_W_Q, layer, (d, d), d * d),
                (TENSOR_W_K, layer, (d, kvd), d * kvd),
                (TENSOR_W_V, layer, (d, kvd), d * kvd),
                (TENSOR_W_O, layer, (d, d), d * d),
                (TENSOR_G_FF, layer, (d, 1), d),
                (TENSOR_W_GATE, layer, (d, f), d * f),
                (TENSOR_W_UP, layer, (d, f), d * f),
                (TENSOR_W_DOWN, layer, (f, d), f * d),
            )
        )
    tensor_inventory.append((TENSOR_G_FINAL, -1, (d, 1), d))
    if not spec.tied_embeddings:
        tensor_inventory.append((TENSOR_LM_HEAD, -1, (spec.vocab_size, d), spec.vocab_size * d))

    lut4_map: dict[tuple[int, int], Lut4Tensor] = {}
    if lut4_tensors is not None:
        if not isinstance(lut4_tensors, Mapping) or len(lut4_tensors) == 0:
            raise ValueError(
                "lut4_tensors must be a non-empty mapping for BCIRQ8 v2, or None for BCIRQ8 v1"
            )
        valid_dims = {(tid, lyr): dims for tid, lyr, dims, _ in tensor_inventory}
        for key, lt in lut4_tensors.items():
            if not isinstance(lt, Lut4Tensor):
                raise ValueError(f"lut4_tensors[{key!r}] must be a Lut4Tensor")
            if key != (lt.tensor_id, lt.layer):
                raise ValueError(
                    f"lut4_tensors key {key!r} disagrees with Lut4Tensor {(lt.tensor_id, lt.layer)}"
                )
            if lt.tensor_id not in LUT4_ALLOWED_TENSOR_IDS:
                raise ValueError(
                    f"LUT4_1B is permitted only on MLP tensors; refused tensor_id={lt.tensor_id}"
                )
            if key not in valid_dims:
                raise ValueError(f"lut4_tensors key {key!r} is outside model layer inventory")
            if (lt.in_dim, lt.out_dim) != valid_dims[key]:
                raise ValueError(
                    f"lut4_tensors[{key!r}] shape {(lt.in_dim, lt.out_dim)} != {valid_dims[key]}"
                )
            lut4_map[key] = lt

    element_count = sum(count for _, _, _, count in tensor_inventory)
    if element_count > max_elements:
        raise ValueError(f"BCIRQ8 export has {element_count} elements; limit is {max_elements}")
    predicted_size = _align8(HEADER_SIZE + len(tensor_inventory) * DIRECTORY_ENTRY_SIZE)
    for tid, lyr, _, count in tensor_inventory:
        key = (tid, lyr)
        if key in lut4_map:
            lt = lut4_map[key]
            predicted_size = _align8(predicted_size)
            predicted_size += lt.scales_bytes
            predicted_size = _align8(predicted_size)
            predicted_size += lt.packed_bytes
        else:
            predicted_size = _align8(predicted_size)
            predicted_size += 2 * ((count + group_size - 1) // group_size)
            predicted_size = _align8(predicted_size)
            predicted_size += count
    if predicted_size > max_bytes:
        raise ValueError(f"BCIRQ8 export is {predicted_size} bytes; limit is {max_bytes}")
    if context_length is None:
        context_length = tokenizer_ids.get("context_length", getattr(spec, "context_length", 0))
    if type(context_length) is not int or context_length < 0 or context_length > 0xFFFFFFFF:
        raise ValueError("context_length must fit u32")

    model_sha = _source_digest(source_hashes, "model")
    config_sha = _source_digest(source_hashes, "config")
    tokenizer_sha = _source_digest(source_hashes, "tokenizer")
    source_raw = (
        _sha_bytes(model_sha, "source model digest"),
        _sha_bytes(config_sha, "source config digest"),
        _sha_bytes(tokenizer_sha, "tokenizer digest"),
    )
    bos = _tokenizer_value(tokenizer_ids, "bos")
    eos = _tokenizer_value(tokenizer_ids, "eos")
    pad = _tokenizer_value(tokenizer_ids, "pad")
    for name, value in (("bos", bos), ("eos", eos), ("pad", pad)):
        if not -(1 << 31) <= value < (1 << 31):
            raise ValueError(f"{name}_token_id must fit i32")
        if value != -1 and not 0 <= value < spec.vocab_size:
            raise ValueError(f"{name}_token_id must be -1 or in the model vocabulary")

    tensors = _tensor_defs(spec, weights)
    directory_offset = HEADER_SIZE
    payload_offset = _align8(directory_offset + len(tensors) * DIRECTORY_ENTRY_SIZE)
    payload = bytearray()
    entries: list[_DirectoryEntry] = []

    def pad_payload() -> None:
        payload.extend(
            b"\0" * (_align8(payload_offset + len(payload)) - (payload_offset + len(payload)))
        )

    for tensor in tensors:
        key = (tensor.tensor_id, tensor.layer)
        if key in lut4_map:
            lt = lut4_map[key]
            exponent_bytes = lt.scales_blob
            code_bytes = lt.packed_blob
            entry_flags = TENSOR_ENC_LUT4_1B
            entry_reserved = len(lt.paths)
            group_field = lt.packed_ranks_u32
        else:
            if native_kernels is None:
                groups = quantize_per_group(tensor.values, group_size, BITS)
                exponent_bytes = struct.pack(f"<{len(groups)}h", *(g.scale_exp for g in groups))
                code_values = [code for group in groups for code in group.codes]
                code_bytes = struct.pack(f"<{len(code_values)}b", *code_values)
                group_field = len(groups)
            else:
                native = native_kernels.quantize(tensor.values, group_size=group_size, bits=BITS)
                exponent_bytes = native.exponent_bytes_le
                code_bytes = native.codes
                group_field = len(native.exponents)
            entry_flags = TENSOR_ENC_Q8
            entry_reserved = 0
        pad_payload()
        exponent_offset = payload_offset + len(payload)
        payload.extend(exponent_bytes)
        pad_payload()
        code_offset = payload_offset + len(payload)
        payload.extend(code_bytes)
        crc = zlib.crc32(exponent_bytes)
        crc = zlib.crc32(code_bytes, crc) & 0xFFFFFFFF
        entries.append(
            _DirectoryEntry(
                tensor.tensor_id,
                tensor.layer,
                tensor.rank,
                entry_flags,
                entry_reserved,
                tensor.dims,
                len(tensor.values),
                group_field,
                crc,
                exponent_offset,
                code_offset,
            )
        )

    file_size = payload_offset + len(payload)
    directory = bytearray()
    for entry in entries:
        directory.extend(
            _DIRECTORY.pack(
                entry.tensor_id,
                entry.layer,
                entry.rank,
                entry.flags,
                entry.reserved,
                entry.dims[0],
                entry.dims[1],
                entry.element_count,
                entry.group_count,
                entry.tensor_crc32,
                0,
                entry.exponent_offset,
                entry.code_offset,
            )
        )
    directory.extend(b"\0" * (payload_offset - directory_offset - len(directory)))
    body = bytes(directory) + bytes(payload)
    body_crc = zlib.crc32(body) & 0xFFFFFFFF
    version_out = VERSION_V2 if lut4_map else VERSION
    flags = FLAG_TIED_EMBEDDINGS if spec.tied_embeddings else 0
    if lut4_map:
        flags |= FLAG_HYBRID_LUT4_MLP
    prefix = _HEADER_PREFIX.pack(
        MAGIC,
        version_out,
        HEADER_SIZE,
        ENDIAN_MARKER,
        flags,
        group_size,
        BITS,
        0,
        spec.vocab_size,
        spec.d_model,
        spec.n_heads,
        spec.kv_heads,
        spec.n_layers,
        spec.d_ff,
        context_length,
        bos,
        eos,
        pad,
        float(spec.rope_base),
        float(getattr(spec, "rms_norm_eps", 1e-6)),
        len(entries),
        DIRECTORY_ENTRY_SIZE,
        directory_offset,
        payload_offset,
        file_size,
        body_crc,
        0,
    )
    header = bytearray(prefix + b"".join(source_raw) + b"\0" * 8)
    header_crc = zlib.crc32(header) & 0xFFFFFFFF
    struct.pack_into("<I", header, 116, header_crc)
    artifact = bytes(header) + body
    if len(artifact) != file_size:
        raise AssertionError(f"BCIRQ8 size accounting failed: {len(artifact)} != {file_size}")
    if file_size != predicted_size:
        raise AssertionError(
            f"BCIRQ8 predicted size accounting failed: {file_size} != {predicted_size}"
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(artifact)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

    return Q8ArtifactMetadata(
        version_out,
        group_size,
        BITS,
        len(entries),
        sum(e.element_count for e in entries),
        file_size,
        body_crc,
        hashlib.sha256(artifact).hexdigest(),
        model_sha,
        config_sha,
        tokenizer_sha,
        context_length,
        bos,
        eos,
        pad,
        len(lut4_map),
    )


def _checked_span(
    size: int, offset: int, length: int, field: str, *, lower_bound: int
) -> tuple[int, int]:
    end = offset + length
    if offset < lower_bound or length < 0 or end < offset or end > size:
        raise ValueError(f"{field} span [{offset}, {end}) escapes the BCIRQ8 payload")
    return offset, end


def _read_directory(
    raw: bytes,
    *,
    version: int,
    header_flags: int,
    tensor_count: int,
    directory_offset: int,
    payload_offset: int,
    group_size: int,
) -> tuple[list[_DirectoryEntry], dict[tuple[int, int], Lut4Tensor]]:
    entries: list[_DirectoryEntry] = []
    lut4_tensors: dict[tuple[int, int], Lut4Tensor] = {}
    spans: list[tuple[int, int, str]] = []
    keys: set[tuple[int, int]] = set()
    view = memoryview(raw)
    for index in range(tensor_count):
        off = directory_offset + index * DIRECTORY_ENTRY_SIZE
        fields = _DIRECTORY.unpack_from(raw, off)
        (
            tensor_id,
            layer,
            rank,
            eflags,
            reserved,
            dim0,
            dim1,
            count,
            groups,
            crc,
            reserved2,
            exponent_offset,
            code_offset,
        ) = fields
        if reserved2:
            raise ValueError(f"tensor directory entry {index} has nonzero reserved fields")
        if rank not in (1, 2) or dim0 < 1 or dim1 < 1:
            raise ValueError(f"tensor directory entry {index} has invalid rank/dimensions")
        expected_count = dim0 if rank == 1 else dim0 * dim1
        if count != expected_count:
            raise ValueError(
                f"tensor directory entry {index} element count does not match dimensions"
            )
        if exponent_offset % 8 or code_offset % 8:
            raise ValueError(f"tensor directory entry {index} payload is not 8-byte aligned")
        if eflags == TENSOR_ENC_LUT4_1B:
            if version != VERSION_V2 or not (header_flags & FLAG_HYBRID_LUT4_MLP):
                raise ValueError(
                    f"tensor directory entry {index} uses LUT4_1B outside BCIRQ8 v2 hybrid format"
                )
            if tensor_id not in LUT4_ALLOWED_TENSOR_IDS or rank != 2 or layer < 0:
                raise ValueError(
                    f"LUT4_1B is permitted only on MLP tensors; refused tensor_id={tensor_id}"
                )
            path_count = reserved
            if not 1 <= path_count <= LUT4_MAX_PATHS:
                raise ValueError(
                    f"tensor directory entry {index} has invalid LUT4 path count {path_count}"
                )
            r0 = groups & 0xFFFF
            r1 = (groups >> 16) & 0xFFFF
            ranks = [r0] if path_count == 1 else [r0, r1]
            if path_count == 1 and r1 != 0:
                raise ValueError(
                    f"tensor directory entry {index} has nonzero residual rank on 1-path LUT4"
                )
            max_r = min(dim0, dim1, LUT4_MAX_RANK)
            if any(not 1 <= r <= max_r for r in ranks):
                raise ValueError(
                    f"tensor directory entry {index} has out-of-range LUT4 ranks {ranks}"
                )
            scales_len = sum(2 * (dim0 + r + dim1) for r in ranks)
            packed_len = sum(lut4_packed_bytes(r, dim0) + lut4_packed_bytes(dim1, r) for r in ranks)
            e_span = _checked_span(
                len(raw),
                exponent_offset,
                scales_len,
                f"tensor {index} lut4 scales",
                lower_bound=payload_offset,
            )
            c_span = _checked_span(
                len(raw),
                code_offset,
                packed_len,
                f"tensor {index} lut4 packed",
                lower_bound=payload_offset,
            )
            s_cur = e_span[0]
            p_cur = c_span[0]
            parsed_paths: list[Lut4Path] = []
            for r in ranks:
                g_b = bytes(view[s_cur : s_cur + dim0 * 2])
                s_cur += dim0 * 2
                l_b = bytes(view[s_cur : s_cur + r * 2])
                s_cur += r * 2
                h_b = bytes(view[s_cur : s_cur + dim1 * 2])
                s_cur += dim1 * 2
                v_len = lut4_packed_bytes(r, dim0)
                u_len = lut4_packed_bytes(dim1, r)
                v_b = bytes(view[p_cur : p_cur + v_len])
                p_cur += v_len
                u_b = bytes(view[p_cur : p_cur + u_len])
                p_cur += u_len
                parsed_paths.append(Lut4Path(r, v_b, u_b, g_b, l_b, h_b))
            lt = Lut4Tensor(tensor_id, layer, dim0, dim1, tuple(parsed_paths))
            lut4_tensors[(tensor_id, layer)] = lt
        elif eflags == TENSOR_ENC_Q8:
            if reserved:
                raise ValueError(f"tensor directory entry {index} has nonzero reserved fields")
            expected_groups = (count + group_size - 1) // group_size
            if groups != expected_groups:
                raise ValueError(f"tensor directory entry {index} group count is invalid")
            e_span = _checked_span(
                len(raw),
                exponent_offset,
                groups * 2,
                f"tensor {index} exponents",
                lower_bound=payload_offset,
            )
            c_span = _checked_span(
                len(raw), code_offset, count, f"tensor {index} codes", lower_bound=payload_offset
            )
            exponent_view = view[exponent_offset : exponent_offset + groups * 2]
            if any(
                exponent < -300 or exponent > 300
                for (exponent,) in struct.iter_unpack("<h", exponent_view)
            ):
                raise ValueError(f"tensor directory entry {index} has an invalid Q8 exponent")
            if raw.find(b"\x80", code_offset, code_offset + count) >= 0:
                raise ValueError(f"tensor directory entry {index} uses the non-canonical -128 code")
        else:
            raise ValueError(
                f"tensor directory entry {index} has unsupported encoding flag {eflags}"
            )
        spans.extend(((*e_span, f"tensor {index} exponents"), (*c_span, f"tensor {index} codes")))
        key = (tensor_id, layer)
        if key in keys:
            raise ValueError(f"duplicate tensor id/layer {key}")
        keys.add(key)
        actual = zlib.crc32(view[e_span[0] : e_span[1]])
        actual = zlib.crc32(view[c_span[0] : c_span[1]], actual) & 0xFFFFFFFF
        if actual != crc:
            raise ValueError(f"tensor directory entry {index} CRC mismatch")
        entries.append(
            _DirectoryEntry(
                tensor_id,
                layer,
                rank,
                eflags,
                reserved,
                (dim0, dim1),
                count,
                groups,
                crc,
                exponent_offset,
                code_offset,
            )
        )
    if version == VERSION_V2 and not lut4_tensors:
        raise ValueError("BCIRQ8 v2 artifact must contain at least one LUT4_1B MLP tensor")
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            raise ValueError(f"overlapping tensor payloads: {left[2]} and {right[2]}")
    return entries, lut4_tensors


def _decode_values(raw: bytes, entry: _DirectoryEntry, group_size: int) -> tuple[float, ...]:
    exponent_view = memoryview(raw)[
        entry.exponent_offset : entry.exponent_offset + entry.group_count * 2
    ]
    exponents = tuple(value for (value,) in struct.iter_unpack("<h", exponent_view))
    codes = memoryview(raw)[entry.code_offset : entry.code_offset + entry.element_count]
    return tuple(
        math.ldexp(float(code if code < 128 else code - 256), exponents[index // group_size])
        for index, code in enumerate(codes)
    )


def _expect(
    entries: Mapping[tuple[int, int], _DirectoryEntry],
    tensor_id: int,
    layer: int,
    dims: tuple[int, int],
    rank: int,
) -> _DirectoryEntry:
    key = (tensor_id, layer)
    if key not in entries:
        raise ValueError(f"BCIRQ8 artifact is missing tensor id/layer {key}")
    entry = entries[key]
    if entry.dims != dims or entry.rank != rank:
        raise ValueError(
            f"BCIRQ8 tensor {key} has shape/rank {entry.dims}/{entry.rank}, expected {dims}/{rank}"
        )
    return entry


def read_q8_decoder_v2(
    path: os.PathLike | str,
    *,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_elements: int = DEFAULT_MAX_DECODED_ELEMENTS,
    reconstruct_lut4: bool = True,
) -> tuple[DecoderSpec, DecoderWeights, dict[tuple[int, int], Lut4Tensor], Q8ArtifactMetadata]:
    """Validate and read a ``BCIRQ8 v1`` or ``BCIRQ8 v2`` artifact.

    Returns ``(spec, weights, lut4_tensors, metadata)``.  For ``BCIRQ8 v1``
    artifacts, ``lut4_tensors`` is empty.  For ``BCIRQ8 v2`` hybrid artifacts,
    ``lut4_tensors`` maps ``(tensor_id, layer)`` to validated :class:`Lut4Tensor`
    instances, and when ``reconstruct_lut4`` is True the corresponding dense
    weights are reconstructed in ``weights`` (otherwise zeros fill those MLP
    slots while Attention/embeddings/norms/LM head are decoded normally).
    """
    if type(max_bytes) is not int or max_bytes < HEADER_SIZE:
        raise ValueError(f"max_bytes must be an integer >= {HEADER_SIZE}")
    if type(max_elements) is not int or max_elements < 1:
        raise ValueError("max_elements must be an integer >= 1")
    with Path(path).open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("BCIRQ8 input must be a regular file")
        if info.st_size > max_bytes:
            raise ValueError(f"BCIRQ8 artifact is {info.st_size} bytes; limit is {max_bytes}")
        raw = stream.read(info.st_size + 1)
        if len(raw) != info.st_size:
            raise ValueError("BCIRQ8 artifact changed size while it was being read")
    if len(raw) < HEADER_SIZE:
        raise ValueError("BCIRQ8 artifact is truncated before its fixed header")
    fields = _HEADER_PREFIX.unpack_from(raw)
    (
        magic,
        version,
        header_size,
        endian,
        flags,
        group_size,
        bits,
        reserved,
        vocab,
        d_model,
        n_heads,
        kv_heads,
        n_layers,
        d_ff,
        context_length,
        bos,
        eos,
        pad,
        rope_base,
        rms_norm_eps,
        tensor_count,
        directory_entry_size,
        directory_offset,
        payload_offset,
        file_size,
        body_crc,
        header_crc,
    ) = fields
    if magic != MAGIC:
        raise ValueError("bad BCIRQ8 magic")
    if version not in (VERSION, VERSION_V2) or header_size != HEADER_SIZE:
        raise ValueError(f"unsupported BCIRQ8 version/header {version}/{header_size}")
    if endian != ENDIAN_MARKER or bits != BITS or reserved:
        raise ValueError("unsupported BCIRQ8 endian/bits/reserved fields")
    allowed_flags = (
        FLAG_TIED_EMBEDDINGS
        if version == VERSION
        else (FLAG_TIED_EMBEDDINGS | FLAG_HYBRID_LUT4_MLP)
    )
    if flags & ~allowed_flags:
        raise ValueError("BCIRQ8 header has unknown flags")
    if version == VERSION_V2 and not (flags & FLAG_HYBRID_LUT4_MLP):
        raise ValueError("BCIRQ8 v2 header requires FLAG_HYBRID_LUT4_MLP")
    if any(raw[216:HEADER_SIZE]):
        raise ValueError("BCIRQ8 header has nonzero reserved bytes")
    if group_size < 1 or directory_entry_size != DIRECTORY_ENTRY_SIZE:
        raise ValueError("invalid BCIRQ8 group or directory entry size")
    if file_size != len(raw):
        raise ValueError(f"BCIRQ8 file size field {file_size} != actual {len(raw)}")
    directory_end = directory_offset + tensor_count * DIRECTORY_ENTRY_SIZE
    if (
        directory_offset != HEADER_SIZE
        or directory_end > payload_offset
        or payload_offset > file_size
        or payload_offset % 8
    ):
        raise ValueError("invalid BCIRQ8 directory/payload offsets")
    if any(raw[directory_end:payload_offset]):
        raise ValueError("BCIRQ8 directory padding must be zero")
    header_copy = bytearray(raw[:HEADER_SIZE])
    struct.pack_into("<I", header_copy, 116, 0)
    if zlib.crc32(header_copy) & 0xFFFFFFFF != header_crc:
        raise ValueError("BCIRQ8 header CRC mismatch")
    if zlib.crc32(raw[directory_offset:]) & 0xFFFFFFFF != body_crc:
        raise ValueError("BCIRQ8 body CRC mismatch")
    if min(vocab, d_model, n_heads, kv_heads, n_layers, d_ff) < 1:
        raise ValueError("BCIRQ8 model dimensions must be positive")
    if n_layers > 0x7FFF:
        raise ValueError("BCIRQ8 n_layers exceeds the signed i16 directory field")
    if d_model % n_heads or n_heads % kv_heads or (d_model // n_heads) % 2:
        raise ValueError("BCIRQ8 model has invalid attention head geometry")
    if (
        not math.isfinite(rope_base)
        or rope_base <= 0
        or not math.isfinite(rms_norm_eps)
        or rms_norm_eps <= 0
    ):
        raise ValueError("BCIRQ8 RoPE/RMSNorm parameters are invalid")
    for name, value in (("bos", bos), ("eos", eos), ("pad", pad)):
        if value != -1 and not 0 <= value < vocab:
            raise ValueError(f"BCIRQ8 {name}_token_id is outside the vocabulary")

    parsed, lut4_tensors = _read_directory(
        raw,
        version=version,
        header_flags=flags,
        tensor_count=tensor_count,
        directory_offset=directory_offset,
        payload_offset=payload_offset,
        group_size=group_size,
    )
    element_count = sum(entry.element_count for entry in parsed)
    if element_count > max_elements:
        raise ValueError(f"BCIRQ8 artifact has {element_count} elements; limit is {max_elements}")
    by_key = {(entry.tensor_id, entry.layer): entry for entry in parsed}
    tied = bool(flags & FLAG_TIED_EMBEDDINGS)
    expected_count = 2 + 9 * n_layers + (0 if tied else 1)
    if tensor_count != expected_count:
        raise ValueError(f"BCIRQ8 has {tensor_count} tensors, expected {expected_count}")
    expected_order = [(TENSOR_EMBEDDING, -1)]
    layer_order = (
        TENSOR_G_ATTN,
        TENSOR_W_Q,
        TENSOR_W_K,
        TENSOR_W_V,
        TENSOR_W_O,
        TENSOR_G_FF,
        TENSOR_W_GATE,
        TENSOR_W_UP,
        TENSOR_W_DOWN,
    )
    for layer in range(n_layers):
        expected_order.extend((tensor_id, layer) for tensor_id in layer_order)
    expected_order.append((TENSOR_G_FINAL, -1))
    if not tied:
        expected_order.append((TENSOR_LM_HEAD, -1))
    actual_order = [(entry.tensor_id, entry.layer) for entry in parsed]
    if actual_order != expected_order:
        raise ValueError("BCIRQ8 tensor directory is not in canonical order")
    kv_dim = kv_heads * (d_model // n_heads)

    def values(tid: int, layer: int, dims: tuple[int, int], rank: int) -> tuple[float, ...]:
        entry = _expect(by_key, tid, layer, dims, rank)
        key = (tid, layer)
        if entry.flags == TENSOR_ENC_LUT4_1B:
            if reconstruct_lut4:
                return reconstruct_lut4_matrix(lut4_tensors[key])
            return (0.0,) * entry.element_count
        return _decode_values(raw, entry, group_size)

    embedding_values = values(TENSOR_EMBEDDING, -1, (vocab, d_model), 2)
    layers = []
    for layer in range(n_layers):
        layers.append(
            LayerWeights(
                g_attn=values(TENSOR_G_ATTN, layer, (d_model, 1), 1),
                w_q=values(TENSOR_W_Q, layer, (d_model, d_model), 2),
                w_k=values(TENSOR_W_K, layer, (d_model, kv_dim), 2),
                w_v=values(TENSOR_W_V, layer, (d_model, kv_dim), 2),
                w_o=values(TENSOR_W_O, layer, (d_model, d_model), 2),
                g_ff=values(TENSOR_G_FF, layer, (d_model, 1), 1),
                w1=values(TENSOR_W_UP, layer, (d_model, d_ff), 2),
                b1=(),
                w2=values(TENSOR_W_DOWN, layer, (d_ff, d_model), 2),
                b2=(),
                w_gate=values(TENSOR_W_GATE, layer, (d_model, d_ff), 2),
            )
        )
    g_final = values(TENSOR_G_FINAL, -1, (d_model, 1), 1)
    lm_head = () if tied else values(TENSOR_LM_HEAD, -1, (vocab, d_model), 2)
    spec_kwargs = dict(
        vocab_size=vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        rope_base=rope_base,
        activation="silu_gate",
        n_kv_heads=kv_heads,
        tied_embeddings=tied,
    )
    if "rms_norm_eps" in inspect.signature(DecoderSpec).parameters:
        spec_kwargs["rms_norm_eps"] = rms_norm_eps
    spec = DecoderSpec(**spec_kwargs)
    weights = DecoderWeights(
        embedding=EmbeddingTable(table=embedding_values, n_vocab=vocab, dim=d_model),
        layers=tuple(layers),
        g_final=g_final,
        lm_head=lm_head,
    )
    errors = check_decoder_weights(spec, weights)
    if errors:
        raise ValueError("decoded BCIRQ8 weights are invalid: " + "; ".join(errors))
    model_sha = raw[120:152].hex()
    config_sha = raw[152:184].hex()
    tokenizer_sha = raw[184:216].hex()
    metadata = Q8ArtifactMetadata(
        version,
        group_size,
        bits,
        tensor_count,
        element_count,
        file_size,
        body_crc,
        hashlib.sha256(raw).hexdigest(),
        model_sha,
        config_sha,
        tokenizer_sha,
        context_length,
        bos,
        eos,
        pad,
        len(lut4_tensors),
    )
    return spec, weights, lut4_tensors, metadata


def read_q8_decoder(
    path: os.PathLike | str,
    *,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_elements: int = DEFAULT_MAX_DECODED_ELEMENTS,
) -> tuple[DecoderSpec, DecoderWeights, Q8ArtifactMetadata]:
    """Validate and read a ``BCIRQ8 v1`` or ``BCIRQ8 v2`` artifact."""
    spec, weights, _lut4, metadata = read_q8_decoder_v2(
        path, max_bytes=max_bytes, max_elements=max_elements, reconstruct_lut4=True
    )
    return spec, weights, metadata


__all__ = [
    "BITS",
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "DEFAULT_MAX_DECODED_ELEMENTS",
    "DIRECTORY_ENTRY_SIZE",
    "FLAG_HYBRID_LUT4_MLP",
    "FLAG_TIED_EMBEDDINGS",
    "HEADER_SIZE",
    "LUT4_ALLOWED_TENSOR_IDS",
    "LUT4_MAX_PATHS",
    "LUT4_MAX_RANK",
    "MAGIC",
    "TENSOR_EMBEDDING",
    "TENSOR_ENC_LUT4_1B",
    "TENSOR_ENC_Q8",
    "TENSOR_G_ATTN",
    "TENSOR_G_FF",
    "TENSOR_G_FINAL",
    "TENSOR_LM_HEAD",
    "TENSOR_W_DOWN",
    "TENSOR_W_GATE",
    "TENSOR_W_K",
    "TENSOR_W_O",
    "TENSOR_W_Q",
    "TENSOR_W_UP",
    "TENSOR_W_V",
    "VERSION",
    "VERSION_V2",
    "Lut4Path",
    "Lut4Tensor",
    "Q8ArtifactMetadata",
    "decode_fp16_le",
    "encode_fp16_le",
    "lut4_matvec_reference",
    "lut4_packed_bytes",
    "pack_lut4_signs",
    "read_q8_decoder",
    "read_q8_decoder_v2",
    "reconstruct_lut4_matrix",
    "unpack_lut4_signs",
    "validate_lut4_packed",
    "write_q8_decoder",
]
