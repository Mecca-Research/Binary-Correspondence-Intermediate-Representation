"""Rung 4 of the open-weight ladder (ML/AI roadmap §7.4): the safetensors TENSOR reader.

Rung 1 (`manifest.py`) reads ONLY the 8-byte length + JSON header (the census contract).
This module reads the weight bytes themselves: `load_tensors` returns every tensor as
Python floats, decoding F64/F32/F16/BF16 with dep-free `struct` arithmetic -- BF16 is a
truncated F32 (the top 16 bits), F16 decodes by the IEEE 754 half-precision layout
(subnormals and infinities included; the values a released checkpoint actually carries).
Layout facts checked, never trusted: every tensor's byte span must sit inside the data
section, spans must not overlap the header, and the element count must match the shape.

Dep-free stdlib on purpose (no numpy): a rung-4 ingest must run wherever the manifest
does. Cost-side: imports no verifier (two-truth)."""

from __future__ import annotations

import os
import struct

from .manifest import _read_safetensors_header

_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}


def _decode_f16(lo: int) -> float:
    """One IEEE 754 binary16 value from its 16-bit pattern (sign/5-exp/10-mantissa)."""
    sign = -1.0 if lo & 0x8000 else 1.0
    exp = (lo >> 10) & 0x1F
    frac = lo & 0x3FF
    if exp == 0:  # subnormal (or zero)
        return sign * frac * 2.0**-24
    if exp == 0x1F:  # inf / nan
        return sign * float("inf") if frac == 0 else float("nan")
    return sign * (1.0 + frac / 1024.0) * 2.0 ** (exp - 15)


def decode_tensor(dtype: str, raw: bytes) -> list[float]:
    """The flat float list of one tensor's bytes (little-endian, the safetensors law)."""
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported tensor dtype {dtype!r} (F64/F32/F16/BF16)")
    if len(raw) % _DTYPE_BYTES[dtype]:
        raise ValueError(
            f"{dtype} tensor byte length {len(raw)} is not a multiple of {_DTYPE_BYTES[dtype]}"
        )
    if dtype == "F64":
        return list(struct.unpack(f"<{len(raw) // 8}d", raw))
    if dtype == "F32":
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if dtype == "BF16":  # the top half of an F32
        n = len(raw) // 2
        halves = struct.unpack(f"<{n}H", raw)
        packed = struct.pack(f"<{n}I", *(h << 16 for h in halves))
        return list(struct.unpack(f"<{n}f", packed))
    if dtype == "F16":
        return [_decode_f16(h) for h in struct.unpack(f"<{len(raw) // 2}H", raw)]
    raise AssertionError("unreachable dtype dispatch")


def load_tensors(path: str, names: list | None = None) -> dict:
    """Every tensor of one safetensors shard (or just `names`) as
    name -> (dtype, shape tuple, flat float list). Validates each tensor's byte span
    against the data section and its element count against the shape -- a lying header
    refuses loudly rather than mis-reading."""
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        header, _metadata, data_off = _read_safetensors_header(f, path, size)
        try:
            selected = None if names is None else set(names)
        except TypeError as exc:
            raise ValueError("selected tensor names must be hashable strings") from exc
        if selected is not None:
            if any(not isinstance(name, str) for name in selected):
                raise ValueError("selected tensor names must be strings")
            missing = selected - set(header)
            if missing:
                raise ValueError(f"requested tensors are absent from shard: {sorted(missing)}")
        out: dict = {}
        for name, info in header.items():
            if selected is not None and name not in selected:
                continue
            dtype = info["dtype"]
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"{name}: unsupported dtype {dtype!r} (F64/F32/F16/BF16)")
            lo, hi = info["data_offsets"]
            n = 1
            for x in info["shape"]:
                n *= x
            if hi - lo != n * _DTYPE_BYTES[dtype]:
                raise ValueError(
                    f"{name}: {hi - lo} bytes != shape {tuple(info['shape'])} "
                    f"x {_DTYPE_BYTES[dtype]}-byte {dtype}"
                )
            f.seek(data_off + lo)
            raw = f.read(hi - lo)
            if len(raw) != hi - lo:
                raise ValueError(f"{name}: shard changed or truncated during tensor read")
            out[name] = (dtype, tuple(info["shape"]), decode_tensor(dtype, raw))
    return out
