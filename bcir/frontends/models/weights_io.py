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
from .decode import (DecoderSpec, DecoderWeights, LayerWeights,
                     check_decoder_weights)

if TYPE_CHECKING:
    from ...kbcir.native_ai import NativeAIKernels

MAGIC = b"BCIRQ8\0\0"
VERSION = 1
HEADER_SIZE = 224
DIRECTORY_ENTRY_SIZE = 48
ENDIAN_MARKER = 0x01020304
BITS = 8
DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_DECODED_ELEMENTS = 8_000_000

FLAG_TIED_EMBEDDINGS = 1 << 0

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

_HEADER_PREFIX = struct.Struct("<8sHHIIHBB6I Iiii dd II QQQ II")
_DIRECTORY = struct.Struct("<HhBBHIIIIIIQQ")
assert _HEADER_PREFIX.size == 120
assert _DIRECTORY.size == DIRECTORY_ENTRY_SIZE


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
    out = [_TensorDef(TENSOR_EMBEDDING, -1, 2, (spec.vocab_size, d),
                      tuple(weights.embedding.table))]
    for layer, lw in enumerate(weights.layers):
        if lw.b1 or lw.b2:
            raise ValueError("BCIRQ8 v1 Llama layers cannot contain MLP biases")
        out.extend((
            _TensorDef(TENSOR_G_ATTN, layer, 1, (d, 1), tuple(lw.g_attn)),
            _TensorDef(TENSOR_W_Q, layer, 2, (d, d), tuple(lw.w_q)),
            _TensorDef(TENSOR_W_K, layer, 2, (d, kvd), tuple(lw.w_k)),
            _TensorDef(TENSOR_W_V, layer, 2, (d, kvd), tuple(lw.w_v)),
            _TensorDef(TENSOR_W_O, layer, 2, (d, d), tuple(lw.w_o)),
            _TensorDef(TENSOR_G_FF, layer, 1, (d, 1), tuple(lw.g_ff)),
            _TensorDef(TENSOR_W_GATE, layer, 2, (d, f), tuple(lw.w_gate)),
            _TensorDef(TENSOR_W_UP, layer, 2, (d, f), tuple(lw.w1)),
            _TensorDef(TENSOR_W_DOWN, layer, 2, (f, d), tuple(lw.w2)),
        ))
    out.append(_TensorDef(TENSOR_G_FINAL, -1, 1, (d, 1), tuple(weights.g_final)))
    if not spec.tied_embeddings:
        out.append(_TensorDef(TENSOR_LM_HEAD, -1, 2, (spec.vocab_size, d),
                              tuple(weights.lm_head)))
    return out


def write_q8_decoder(path: os.PathLike | str, spec: DecoderSpec, weights: DecoderWeights, *,
                     group_size: int = 32, source_hashes: Mapping[str, str],
                     tokenizer_ids: Mapping[str, int], context_length: int | None = None,
                     max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
                     max_elements: int = DEFAULT_MAX_DECODED_ELEMENTS,
                     native_kernels: NativeAIKernels | None = None,
                     ) -> Q8ArtifactMetadata:
    """Write one atomic, deterministic ``BCIRQ8 v1`` artifact.

    ``source_hashes`` accepts ``model``, ``config`` and ``tokenizer`` SHA-256
    values.  ``tokenizer_ids`` accepts ``bos``/``eos``/``pad`` (or the HF
    ``*_token_id`` spellings).  ``context_length`` may be supplied explicitly;
    otherwise ``tokenizer_ids['context_length']`` or ``spec.context_length`` is
    used, falling back to zero for synthetic fixtures. ``max_bytes`` and
    ``max_elements`` bound the in-memory export before any tensor is copied or
    quantized; callers must explicitly opt into larger artifacts. Supplying a
    typed ``NativeAIKernels`` instance moves group quantization into the checked
    C data plane; omission retains the independent Python oracle.
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
        "vocab_size": spec.vocab_size, "d_model": spec.d_model,
        "n_heads": spec.n_heads, "n_kv_heads": spec.kv_heads,
        "n_layers": spec.n_layers, "d_ff": spec.d_ff,
    }
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 0xFFFFFFFF:
            raise ValueError(f"{name} must fit a positive u32")
    if spec.n_layers > 0x7FFF:
        raise ValueError("n_layers must fit the BCIRQ8 signed i16 layer field")
    d, f, kvd = spec.d_model, spec.d_ff, spec.kv_dim
    tensor_counts = [spec.vocab_size * d]
    for _ in range(spec.n_layers):
        tensor_counts.extend((d, d * d, d * kvd, d * kvd, d * d,
                              d, d * f, d * f, f * d))
    tensor_counts.append(d)
    if not spec.tied_embeddings:
        tensor_counts.append(spec.vocab_size * d)
    element_count = sum(tensor_counts)
    if element_count > max_elements:
        raise ValueError(
            f"BCIRQ8 export has {element_count} elements; limit is {max_elements}")
    predicted_size = _align8(HEADER_SIZE + len(tensor_counts) * DIRECTORY_ENTRY_SIZE)
    for count in tensor_counts:
        predicted_size = _align8(predicted_size)
        predicted_size += 2 * ((count + group_size - 1) // group_size)
        predicted_size = _align8(predicted_size)
        predicted_size += count
    if predicted_size > max_bytes:
        raise ValueError(
            f"BCIRQ8 export is {predicted_size} bytes; limit is {max_bytes}")
    if context_length is None:
        context_length = tokenizer_ids.get(
            "context_length", getattr(spec, "context_length", 0))
    if type(context_length) is not int or context_length < 0 or context_length > 0xFFFFFFFF:
        raise ValueError("context_length must fit u32")

    model_sha = _source_digest(source_hashes, "model")
    config_sha = _source_digest(source_hashes, "config")
    tokenizer_sha = _source_digest(source_hashes, "tokenizer")
    source_raw = (_sha_bytes(model_sha, "source model digest"),
                  _sha_bytes(config_sha, "source config digest"),
                  _sha_bytes(tokenizer_sha, "tokenizer digest"))
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
        payload.extend(b"\0" * ((_align8(payload_offset + len(payload))
                                  - (payload_offset + len(payload)))))

    for tensor in tensors:
        if native_kernels is None:
            groups = quantize_per_group(tensor.values, group_size, BITS)
            exponent_bytes = struct.pack(f"<{len(groups)}h",
                                         *(g.scale_exp for g in groups))
            code_values = [code for group in groups for code in group.codes]
            code_bytes = struct.pack(f"<{len(code_values)}b", *code_values)
            group_count = len(groups)
        else:
            native = native_kernels.quantize(tensor.values, group_size=group_size,
                                             bits=BITS)
            exponent_bytes = native.exponent_bytes_le
            code_bytes = native.codes
            group_count = len(native.exponents)
        pad_payload()
        exponent_offset = payload_offset + len(payload)
        payload.extend(exponent_bytes)
        pad_payload()
        code_offset = payload_offset + len(payload)
        payload.extend(code_bytes)
        crc = zlib.crc32(exponent_bytes)
        crc = zlib.crc32(code_bytes, crc) & 0xFFFFFFFF
        entries.append(_DirectoryEntry(
            tensor.tensor_id, tensor.layer, tensor.rank, 0, tensor.dims,
            len(tensor.values), group_count, crc, exponent_offset, code_offset))

    file_size = payload_offset + len(payload)
    directory = bytearray()
    for entry in entries:
        directory.extend(_DIRECTORY.pack(
            entry.tensor_id, entry.layer, entry.rank, entry.flags, 0,
            entry.dims[0], entry.dims[1], entry.element_count, entry.group_count,
            entry.tensor_crc32, 0, entry.exponent_offset, entry.code_offset))
    directory.extend(b"\0" * (payload_offset - directory_offset - len(directory)))
    body = bytes(directory) + bytes(payload)
    body_crc = zlib.crc32(body) & 0xFFFFFFFF
    flags = FLAG_TIED_EMBEDDINGS if spec.tied_embeddings else 0
    prefix = _HEADER_PREFIX.pack(
        MAGIC, VERSION, HEADER_SIZE, ENDIAN_MARKER, flags, group_size, BITS, 0,
        spec.vocab_size, spec.d_model, spec.n_heads, spec.kv_heads, spec.n_layers,
        spec.d_ff, context_length, bos, eos, pad, float(spec.rope_base),
        float(getattr(spec, "rms_norm_eps", 1e-6)), len(entries),
        DIRECTORY_ENTRY_SIZE, directory_offset, payload_offset, file_size,
        body_crc, 0)
    header = bytearray(prefix + b"".join(source_raw) + b"\0" * 8)
    header_crc = zlib.crc32(header) & 0xFFFFFFFF
    struct.pack_into("<I", header, 116, header_crc)
    artifact = bytes(header) + body
    if len(artifact) != file_size:
        raise AssertionError(f"BCIRQ8 size accounting failed: {len(artifact)} != {file_size}")
    if file_size != predicted_size:
        raise AssertionError(
            f"BCIRQ8 predicted size accounting failed: {file_size} != {predicted_size}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                     dir=str(target.parent))
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
        VERSION, group_size, BITS, len(entries), sum(e.element_count for e in entries),
        file_size, body_crc, hashlib.sha256(artifact).hexdigest(), model_sha, config_sha,
        tokenizer_sha, context_length, bos, eos, pad)


def _checked_span(size: int, offset: int, length: int, field: str,
                  *, lower_bound: int) -> tuple[int, int]:
    end = offset + length
    if offset < lower_bound or length < 0 or end < offset or end > size:
        raise ValueError(f"{field} span [{offset}, {end}) escapes the BCIRQ8 payload")
    return offset, end


def _read_directory(raw: bytes, *, tensor_count: int, directory_offset: int,
                    payload_offset: int, group_size: int) -> list[_DirectoryEntry]:
    entries: list[_DirectoryEntry] = []
    spans: list[tuple[int, int, str]] = []
    keys: set[tuple[int, int]] = set()
    for index in range(tensor_count):
        off = directory_offset + index * DIRECTORY_ENTRY_SIZE
        fields = _DIRECTORY.unpack_from(raw, off)
        (tensor_id, layer, rank, flags, reserved, dim0, dim1, count, groups,
         crc, reserved2, exponent_offset, code_offset) = fields
        if reserved or reserved2 or flags:
            raise ValueError(f"tensor directory entry {index} has nonzero reserved fields")
        if rank not in (1, 2) or dim0 < 1 or dim1 < 1:
            raise ValueError(f"tensor directory entry {index} has invalid rank/dimensions")
        expected_count = dim0 if rank == 1 else dim0 * dim1
        if count != expected_count:
            raise ValueError(f"tensor directory entry {index} element count does not match dimensions")
        expected_groups = (count + group_size - 1) // group_size
        if groups != expected_groups:
            raise ValueError(f"tensor directory entry {index} group count is invalid")
        e_span = _checked_span(len(raw), exponent_offset, groups * 2,
                               f"tensor {index} exponents", lower_bound=payload_offset)
        c_span = _checked_span(len(raw), code_offset, count,
                               f"tensor {index} codes", lower_bound=payload_offset)
        if exponent_offset % 8 or code_offset % 8:
            raise ValueError(f"tensor directory entry {index} payload is not 8-byte aligned")
        exponent_view = memoryview(raw)[exponent_offset:exponent_offset + groups * 2]
        if any(exponent < -300 or exponent > 300
               for (exponent,) in struct.iter_unpack("<h", exponent_view)):
            raise ValueError(f"tensor directory entry {index} has an invalid Q8 exponent")
        if raw.find(b"\x80", code_offset, code_offset + count) >= 0:
            raise ValueError(f"tensor directory entry {index} uses the non-canonical -128 code")
        spans.extend(((*e_span, f"tensor {index} exponents"),
                      (*c_span, f"tensor {index} codes")))
        key = (tensor_id, layer)
        if key in keys:
            raise ValueError(f"duplicate tensor id/layer {key}")
        keys.add(key)
        view = memoryview(raw)
        actual = zlib.crc32(view[e_span[0]:e_span[1]])
        actual = zlib.crc32(view[c_span[0]:c_span[1]], actual) & 0xFFFFFFFF
        if actual != crc:
            raise ValueError(f"tensor directory entry {index} CRC mismatch")
        entries.append(_DirectoryEntry(tensor_id, layer, rank, flags, (dim0, dim1),
                                       count, groups, crc, exponent_offset, code_offset))
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            raise ValueError(f"overlapping tensor payloads: {left[2]} and {right[2]}")
    return entries


def _decode_values(raw: bytes, entry: _DirectoryEntry, group_size: int) -> tuple[float, ...]:
    exponent_view = memoryview(raw)[entry.exponent_offset:
                                    entry.exponent_offset + entry.group_count * 2]
    exponents = tuple(value for (value,) in struct.iter_unpack("<h", exponent_view))
    codes = memoryview(raw)[entry.code_offset:entry.code_offset + entry.element_count]
    return tuple(math.ldexp(float(code if code < 128 else code - 256),
                            exponents[index // group_size])
                 for index, code in enumerate(codes))


def _expect(entries: Mapping[tuple[int, int], _DirectoryEntry], tensor_id: int,
            layer: int, dims: tuple[int, int], rank: int) -> _DirectoryEntry:
    key = (tensor_id, layer)
    if key not in entries:
        raise ValueError(f"BCIRQ8 artifact is missing tensor id/layer {key}")
    entry = entries[key]
    if entry.dims != dims or entry.rank != rank:
        raise ValueError(f"BCIRQ8 tensor {key} has shape/rank {entry.dims}/{entry.rank}, "
                         f"expected {dims}/{rank}")
    return entry


def read_q8_decoder(path: os.PathLike | str, *,
                    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
                    max_elements: int = DEFAULT_MAX_DECODED_ELEMENTS
                    ) -> tuple[DecoderSpec, DecoderWeights, Q8ArtifactMetadata]:
    """Validate and read a ``BCIRQ8 v1`` artifact.

    Unknown, missing, duplicated, overlapping, truncated, or corrupt tensors are
    rejected before a decoder object is constructed. The default byte and decoded-
    element ceilings are safe reference limits; larger trusted models require an
    explicit opt-in through ``max_bytes`` and ``max_elements``.
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
            raise ValueError(
                f"BCIRQ8 artifact is {info.st_size} bytes; limit is {max_bytes}")
        raw = stream.read(info.st_size + 1)
        if len(raw) != info.st_size:
            raise ValueError("BCIRQ8 artifact changed size while it was being read")
    if len(raw) < HEADER_SIZE:
        raise ValueError("BCIRQ8 artifact is truncated before its fixed header")
    fields = _HEADER_PREFIX.unpack_from(raw)
    (magic, version, header_size, endian, flags, group_size, bits, reserved,
     vocab, d_model, n_heads, kv_heads, n_layers, d_ff, context_length, bos, eos,
     pad, rope_base, rms_norm_eps, tensor_count, directory_entry_size,
     directory_offset, payload_offset, file_size, body_crc, header_crc) = fields
    if magic != MAGIC:
        raise ValueError("bad BCIRQ8 magic")
    if version != VERSION or header_size != HEADER_SIZE:
        raise ValueError(f"unsupported BCIRQ8 version/header {version}/{header_size}")
    if endian != ENDIAN_MARKER or bits != BITS or reserved:
        raise ValueError("unsupported BCIRQ8 endian/bits/reserved fields")
    if flags & ~FLAG_TIED_EMBEDDINGS:
        raise ValueError("BCIRQ8 header has unknown flags")
    if any(raw[216:HEADER_SIZE]):
        raise ValueError("BCIRQ8 header has nonzero reserved bytes")
    if group_size < 1 or directory_entry_size != DIRECTORY_ENTRY_SIZE:
        raise ValueError("invalid BCIRQ8 group or directory entry size")
    if file_size != len(raw):
        raise ValueError(f"BCIRQ8 file size field {file_size} != actual {len(raw)}")
    directory_end = directory_offset + tensor_count * DIRECTORY_ENTRY_SIZE
    if (directory_offset != HEADER_SIZE or directory_end > payload_offset
            or payload_offset > file_size or payload_offset % 8):
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
    if not math.isfinite(rope_base) or rope_base <= 0 or not math.isfinite(rms_norm_eps) \
            or rms_norm_eps <= 0:
        raise ValueError("BCIRQ8 RoPE/RMSNorm parameters are invalid")
    for name, value in (("bos", bos), ("eos", eos), ("pad", pad)):
        if value != -1 and not 0 <= value < vocab:
            raise ValueError(f"BCIRQ8 {name}_token_id is outside the vocabulary")

    parsed = _read_directory(raw, tensor_count=tensor_count,
                             directory_offset=directory_offset,
                             payload_offset=payload_offset, group_size=group_size)
    element_count = sum(entry.element_count for entry in parsed)
    if element_count > max_elements:
        raise ValueError(
            f"BCIRQ8 artifact has {element_count} elements; limit is {max_elements}")
    by_key = {(entry.tensor_id, entry.layer): entry for entry in parsed}
    tied = bool(flags & FLAG_TIED_EMBEDDINGS)
    expected_count = 2 + 9 * n_layers + (0 if tied else 1)
    if tensor_count != expected_count:
        raise ValueError(f"BCIRQ8 has {tensor_count} tensors, expected {expected_count}")
    expected_order = [(TENSOR_EMBEDDING, -1)]
    layer_order = (TENSOR_G_ATTN, TENSOR_W_Q, TENSOR_W_K, TENSOR_W_V,
                   TENSOR_W_O, TENSOR_G_FF, TENSOR_W_GATE, TENSOR_W_UP,
                   TENSOR_W_DOWN)
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
        return _decode_values(raw, _expect(by_key, tid, layer, dims, rank), group_size)

    embedding_values = values(TENSOR_EMBEDDING, -1, (vocab, d_model), 2)
    layers = []
    for layer in range(n_layers):
        layers.append(LayerWeights(
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
            w_gate=values(TENSOR_W_GATE, layer, (d_model, d_ff), 2)))
    g_final = values(TENSOR_G_FINAL, -1, (d_model, 1), 1)
    lm_head = () if tied else values(TENSOR_LM_HEAD, -1, (vocab, d_model), 2)
    spec_kwargs = dict(vocab_size=vocab, d_model=d_model, n_heads=n_heads,
                       n_layers=n_layers, d_ff=d_ff, rope_base=rope_base,
                       activation="silu_gate", n_kv_heads=kv_heads,
                       tied_embeddings=tied)
    if "rms_norm_eps" in inspect.signature(DecoderSpec).parameters:
        spec_kwargs["rms_norm_eps"] = rms_norm_eps
    spec = DecoderSpec(**spec_kwargs)
    weights = DecoderWeights(
        embedding=EmbeddingTable(table=embedding_values, n_vocab=vocab, dim=d_model),
        layers=tuple(layers), g_final=g_final, lm_head=lm_head)
    errors = check_decoder_weights(spec, weights)
    if errors:
        raise ValueError("decoded BCIRQ8 weights are invalid: " + "; ".join(errors))
    model_sha = raw[120:152].hex()
    config_sha = raw[152:184].hex()
    tokenizer_sha = raw[184:216].hex()
    metadata = Q8ArtifactMetadata(
        version, group_size, bits, tensor_count,
        element_count, file_size, body_crc,
        hashlib.sha256(raw).hexdigest(), model_sha, config_sha, tokenizer_sha,
        context_length, bos, eos, pad)
    return spec, weights, metadata


__all__ = [
    "BITS", "DIRECTORY_ENTRY_SIZE", "HEADER_SIZE", "MAGIC", "VERSION",
    "DEFAULT_MAX_ARTIFACT_BYTES", "DEFAULT_MAX_DECODED_ELEMENTS",
    "Q8ArtifactMetadata", "read_q8_decoder", "write_q8_decoder",
]
