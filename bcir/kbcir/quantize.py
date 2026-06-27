"""Per-group (block) Q8 <-> float32 quantization bridge -- the A1 inference substrate.

Microscaling-style block floating point (cf. OCP MX / NVFP4): a GROUP of `group_size` values shares
one power-of-two scale ``2**scale_exp`` and is stored as `bits`-wide *signed* integer CODES, so

    real ~= code * 2**scale_exp ,   code in [-code_max, code_max],   code_max = 2**(bits-1) - 1.

A per-GROUP scale localizes dynamic range: a small-magnitude group gets a finer grid than a single
per-tensor scale would, which is the SOTA-standard way to make sub-8-bit quantization near-lossless
(every <=4-bit format -- INT4-g128, OCP-MX block=32, NVFP4 block=16 -- relies on it; see
docs/research/AI_SUBSTRATE_SOTA.md, Pillar 1). The codes are exact-width integers: they are the
`_BitInt(N)` lanes of A1, and dequantization is a shift.

Determinism: the only floats are the bridge's own input/output. The codes, the scale exponents, and the
round-trip ERROR are integer / power-of-two and fully reproducible -- no IEEE-rounding model leaks into
the plan. The static round-trip bound (<= 1 ULP at the realized per-group grid) is what the R17 accuracy
law certifies (``bcir.kbcir.precision.accuracy_bound`` / ``bcir.verify.verify_accuracy``); choosing the
per-group scale that makes that grid *fine in real terms* is this module's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Power-of-two scale exponents are clamped to a generous, physically-ample band so a pathological input
# (e.g. 1e-300) can never underflow the scale to 0.0 and divide-by-zero. +-300 covers every realistic ML
# magnitude with room to spare (float32 tops out near 3.4e38 ~ 2**128).
_EXP_MIN, _EXP_MAX = -300, 300


def code_max(bits: int) -> int:
    """The largest magnitude a `bits`-wide SIGNED code can hold (symmetric range [-code_max, code_max])."""
    if bits < 2:
        raise ValueError(f"bits must be >= 2 (a signed code needs a sign + >=1 magnitude bit); got {bits}")
    return (1 << (bits - 1)) - 1


def _round_half_away(x: float) -> int:
    """Deterministic round-half-away-from-zero (symmetric; NOT Python's banker's `round`)."""
    return int(math.floor(x + 0.5)) if x >= 0.0 else -int(math.floor(-x + 0.5))


def _quantize_code(x: float, cmax: int, scale: float, rounding: str) -> int:
    q = x / scale
    c = int(q) if rounding == "truncate" else _round_half_away(q)   # truncate = toward zero
    return cmax if c > cmax else -cmax if c < -cmax else c          # saturate into the lane


@dataclass(frozen=True)
class QGroup:
    """One quantized block: `bits`-wide signed integer `codes` sharing a power-of-two scale 2**scale_exp.
    ``real ~= code * 2**scale_exp``. Frozen + all-integer, so it is hashable and content-addressable."""

    codes: tuple[int, ...]
    scale_exp: int
    bits: int

    @property
    def scale(self) -> float:
        return math.ldexp(1.0, self.scale_exp)         # 2**scale_exp, exact for in-range exponents

    def dequantize(self) -> list[float]:
        s = self.scale
        return [c * s for c in self.codes]

    def error_bound(self, *, rounding: str = "nearest") -> float:
        """Worst-case per-element round-trip error of THIS group, in REAL units: <= 1/2 step (nearest)
        or < 1 step (truncate), where the step is 2**scale_exp. (Tighter the smaller the group's range.)"""
        return math.ldexp(1.0, self.scale_exp - (1 if rounding == "nearest" else 0))


def quantize_group(values, bits: int, *, rounding: str = "nearest") -> QGroup:
    """Quantize ONE group to a shared power-of-two scale chosen so the largest magnitude just fits the
    `bits`-wide code range (so the max element never saturates -- only the chosen scale, never a clip,
    sets the error)."""
    if rounding not in ("nearest", "truncate"):
        raise ValueError(f"rounding must be 'nearest' or 'truncate'; got {rounding!r}")
    vals = [float(v) for v in values]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("quantize_group: inputs must be finite (no inf/nan at the bridge boundary)")
    cmax = code_max(bits)
    amax = max((abs(v) for v in vals), default=0.0)
    if amax == 0.0:
        return QGroup(codes=tuple(0 for _ in vals), scale_exp=0, bits=bits)
    # smallest power-of-two step 2**e with cmax * 2**e >= amax  ->  e = ceil(log2(amax / cmax)).
    e = max(_EXP_MIN, min(_EXP_MAX, math.ceil(math.log2(amax / cmax))))
    while e < _EXP_MAX and cmax * math.ldexp(1.0, e) < amax:        # fp guard: never let the max saturate
        e += 1
    scale = math.ldexp(1.0, e)
    return QGroup(codes=tuple(_quantize_code(v, cmax, scale, rounding) for v in vals),
                  scale_exp=e, bits=bits)


def quantize_per_group(values, group_size: int, bits: int, *, rounding: str = "nearest") -> list[QGroup]:
    """Quantize `values` in contiguous blocks of `group_size` (the last block may be short). `group_size`
    == len(values) is per-tensor; `group_size` == 1 is per-element. Returns one QGroup per block."""
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1; got {group_size}")
    vals = [float(v) for v in values]
    return [quantize_group(vals[i:i + group_size], bits, rounding=rounding)
            for i in range(0, len(vals), group_size)]


def dequantize(groups) -> list[float]:
    """Reconstruct the float vector from its quantized groups (the float32 side of the bridge)."""
    out: list[float] = []
    for g in groups:
        out.extend(g.dequantize())
    return out


def roundtrip(values, group_size: int, bits: int, *, rounding: str = "nearest") -> list[float]:
    """float -> per-group quantize -> dequantize -> float. The Q8<->float32<->Q8 bridge, end to end."""
    return dequantize(quantize_per_group(values, group_size, bits, rounding=rounding))


def max_abs_error(values, group_size: int, bits: int, *, rounding: str = "nearest") -> float:
    """Measured worst-case round-trip error over the vector (REAL units) -- the empirical quality metric;
    strictly bounded by each group's ``error_bound`` (the property R17 certifies)."""
    deq = roundtrip(values, group_size, bits, rounding=rounding)
    return max((abs(float(v) - d) for v, d in zip(values, deq)), default=0.0)


# --- quantized dot product: the inference primitive (a matmul is a grid of these) ----------------
# Integer-code accumulation is EXACT -- a sum of `bits`-wide code products carries no per-term Q8
# truncation (unlike a fixed-point reduce). So a quantized dot product's only error is the INPUT
# quantization; the accumulation contributes none. The exact integer sum is what the `_BitInt(N)`-lane
# C kernel (lower.c_kernel.emit_quantized_dot_c) computes; the per-group scales are applied here.

def integer_dot(codes_a, codes_b) -> int:
    """Exact integer dot product of two equal-length code vectors (Python big-ints: no overflow, the
    reference the `_BitInt`-lane C kernel's wide accumulator must match within its declared width)."""
    if len(codes_a) != len(codes_b):
        raise ValueError(f"length mismatch: {len(codes_a)} != {len(codes_b)}")
    return sum(int(a) * int(b) for a, b in zip(codes_a, codes_b))


def scaled_dot(groups_a, groups_b) -> float:
    """Dequantized dot product of two identically-grouped quantizations: per group, the EXACT integer
    code dot times the product of the two power-of-two scales, summed over groups (real units)."""
    if len(groups_a) != len(groups_b):
        raise ValueError(f"group count mismatch: {len(groups_a)} != {len(groups_b)}")
    acc = 0.0
    for ga, gb in zip(groups_a, groups_b):
        acc += integer_dot(ga.codes, gb.codes) * math.ldexp(1.0, ga.scale_exp + gb.scale_exp)
    return acc


def quantized_dot(a, b, group_size: int, bits: int, *, rounding: str = "nearest") -> float:
    """float -> per-group quantize BOTH operands -> exact integer code dot per group -> rescale. The end-
    to-end quantized inference dot product; its error vs the exact float dot is bounded by the INPUT
    quantization alone (the accumulation is exact)."""
    qa = quantize_per_group(a, group_size, bits, rounding=rounding)
    qb = quantize_per_group(b, group_size, bits, rounding=rounding)
    return scaled_dot(qa, qb)


def accumulator_bits(lane_bits: int, count: int) -> int:
    """The exact accumulator width for a `count`-term dot product of `lane_bits` codes: a product needs
    2*lane_bits bits, and summing `count` of them needs ceil(log2(count)) carry bits more. This is the
    width the `_BitInt`-lane kernel must declare so the integer accumulation never overflows (= is exact)."""
    return 2 * lane_bits + max(0, (max(1, count) - 1)).bit_length()
