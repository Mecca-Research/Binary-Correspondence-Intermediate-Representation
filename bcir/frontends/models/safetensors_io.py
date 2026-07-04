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

import json
import os
import struct

from .manifest import parse_safetensors_header

_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}


def _decode_f16(lo: int) -> float:
    """One IEEE 754 binary16 value from its 16-bit pattern (sign/5-exp/10-mantissa)."""
    sign = -1.0 if lo & 0x8000 else 1.0
    exp = (lo >> 10) & 0x1F
    frac = lo & 0x3FF
    if exp == 0:                                       # subnormal (or zero)
        return sign * frac * 2.0 ** -24
    if exp == 0x1F:                                    # inf / nan
        return sign * float("inf") if frac == 0 else float("nan")
    return sign * (1.0 + frac / 1024.0) * 2.0 ** (exp - 15)


def decode_tensor(dtype: str, raw: bytes) -> list[float]:
    """The flat float list of one tensor's bytes (little-endian, the safetensors law)."""
    if dtype == "F64":
        return list(struct.unpack(f"<{len(raw) // 8}d", raw))
    if dtype == "F32":
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if dtype == "BF16":                                # the top half of an F32
        n = len(raw) // 2
        halves = struct.unpack(f"<{n}H", raw)
        packed = struct.pack(f"<{n}I", *(h << 16 for h in halves))
        return list(struct.unpack(f"<{n}f", packed))
    if dtype == "F16":
        return [_decode_f16(h) for h in struct.unpack(f"<{len(raw) // 2}H", raw)]
    raise ValueError(f"unsupported tensor dtype {dtype!r} (F64/F32/F16/BF16)")


def load_tensors(path: str, names: list | None = None) -> dict:
    """Every tensor of one safetensors shard (or just `names`) as
    name -> (dtype, shape tuple, flat float list). Validates each tensor's byte span
    against the data section and its element count against the shape -- a lying header
    refuses loudly rather than mis-reading."""
    parse_safetensors_header(path)                     # the rung-1 validation law first
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
        header.pop("__metadata__", None)               # rung 1 reads it; the loader reads BYTES
        data_off = 8 + hlen
        out: dict = {}
        for name, info in header.items():
            if names is not None and name not in names:
                continue
            dtype = info["dtype"]
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"{name}: unsupported dtype {dtype!r} (F64/F32/F16/BF16)")
            if "data_offsets" not in info or len(info["data_offsets"]) != 2:
                raise ValueError(f"{name}: missing data_offsets (not a weight-bearing shard)")
            lo, hi = (int(v) for v in info["data_offsets"])
            if not (0 <= lo <= hi and data_off + hi <= size):
                raise ValueError(f"{name}: byte span [{lo}, {hi}) escapes the data section")
            n = 1
            for x in info["shape"]:
                n *= x
            if hi - lo != n * _DTYPE_BYTES[dtype]:
                raise ValueError(f"{name}: {hi - lo} bytes != shape {tuple(info['shape'])} "
                                 f"x {_DTYPE_BYTES[dtype]}-byte {dtype}")
            f.seek(data_off + lo)
            out[name] = (dtype, tuple(info["shape"]), decode_tensor(dtype, f.read(hi - lo)))
    return out
