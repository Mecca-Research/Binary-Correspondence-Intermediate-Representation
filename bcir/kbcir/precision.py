"""Deterministic precision / numerical-stability analysis -- the portable subset
mined from the MPAT (Machine Precision Analysis Toolkit) and AEDACI (Approximation
Error Detection & Correction Interface) notes.

Everything here is Q8 fixed-point and integer-exact: no floats, no IEEE rounding
model, no arbitrary-precision bignum. The "error currency" is the **Q8 ULP** -- one
integer tick = 2^-8 in real units; a Q8 op that truncates its low 8 bits loses at
most 1 ULP. Three deterministic tools:

  * **interval error bounds** (`accuracy_bound`, `Interval`) -- propagate integer
    [lo, hi] ranges and a worst-case ULP error count over a claim, giving a provable
    static accuracy bound. This is the producer the cost vector's *accuracy*
    dimension never had; it is **opt-in** and does NOT change the default plan cost,
    so the pinned K_BCIR scores (7808 / 9472) are unaffected.
  * **compensated Q8 reduction** (`compensated_reduce_q8`) -- a residual-carry
    accumulator (the integer analog of Kahan / TwoSum) that is *bit-identical* to the
    int64-exact reduction, versus the naive per-term-truncating accumulator
    (`naive_reduce_q8`) that drifts up to `count` ULP. The measured win, provable:
    `acc*256 + resid == sum(x*w)` is a loop invariant, so the compensated result is
    exactly `(sum x*w) >> 8`.
  * **stability diagnostics** (`cancellation`, `condition_milli`) -- catastrophic-
    cancellation / ill-conditioning signals emitted as two-truth `Graded(value,
    confidence)` propositions: they INFORM but never legislate (the quarantine
    boundary -- a graded value must be `decide`-d before it can be a verdict).

Deliberately NOT ported from the source notes (non-deterministic, float-bound, or
pure bloat for an integer IR): the entire ECC / coding-theory catalogue, ML decoders
and the novel-algorithm zoo, arbitrary-precision bignum, decimal (DPD/BCD), CORDIC,
root-finder / optimizer / quadrature libraries, affine arithmetic, the Newton-Raphson
condition-number *search*, and the runtime hot-patch / kernel-daemon machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

from .twotruth import Graded

Q8 = 256          # 1.0 in Q8 fixed point
ULP = 1           # one Q8 tick = 2^-8 in real units -- the integer error unit


def ulp_distance(a: int, b: int) -> int:
    """Integer error between two Q8 values, in ULPs (ticks)."""
    return abs(a - b)


# --- integer interval arithmetic (exact; no rounding) ----------------------------

@dataclass(frozen=True)
class Interval:
    """An inclusive integer range [lo, hi] of Q8 values. Exact -- the bounds are
    integers and every operation is integer min/max, so propagation is deterministic
    (no rounding to model)."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"interval lo {self.lo} > hi {self.hi}")

    @property
    def width(self) -> int:
        return self.hi - self.lo

    @property
    def mag(self) -> int:
        return max(abs(self.lo), abs(self.hi))

    def __contains__(self, v: int) -> bool:
        return self.lo <= v <= self.hi


def iadd(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo + b.lo, a.hi + b.hi)


def isub(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo - b.hi, a.hi - b.lo)


def imul_q8(a: Interval, b: Interval) -> Interval:
    """Q8 interval multiply (each product right-shifted by 8, conservatively floored)."""
    p = [(a.lo * b.lo) >> 8, (a.lo * b.hi) >> 8, (a.hi * b.lo) >> 8, (a.hi * b.hi) >> 8]
    return Interval(min(p), max(p))


# --- static error bounds (the accuracy-dim producer; opt-in) ---------------------

def reduction_error_bound(count: int, *, compensated: bool = False) -> int:
    """Worst-case accumulated error (ULPs) of a Q8 multiply-accumulate over `count`
    terms. Each naive truncating MAC drops < 1 ULP, so over `count` terms the bound
    is `count` ULP; the residual-carry (compensated) form keeps the dropped low bits,
    so its bound is at most 1 ULP regardless of `count`."""
    n = max(0, count)
    return 1 if compensated else n


def accuracy_bound(claim, *, compensated: bool = False) -> int:
    """Static worst-case error (ULPs) for a claim's Q8 realization -- the deterministic
    producer for the cost vector's accuracy dimension. A `reduce.*` claim accumulates
    `count` terms (bound = count ULP naive, 1 ULP compensated); any other elementwise
    Q8 op truncates at most once (1 ULP). Opt-in: callers feed this into the accuracy
    dim explicitly; it is not wired into the default plan cost."""
    op = getattr(claim, "op", "") or ""
    n = max(1, getattr(claim, "count", 1))
    if op.startswith("reduce."):
        return reduction_error_bound(n, compensated=compensated)
    return ULP


def meets_tolerance(claim, tolerance_ulp: int, *, compensated: bool = False) -> bool:
    """The checkable accuracy contract: the claim's realization is legal at the
    declared tolerance iff its static error bound fits within it (a verifier-law
    shape; oracle-side, opt-in -- MLIR parity is roadmap)."""
    return accuracy_bound(claim, compensated=compensated) <= tolerance_ulp


# --- compensated Q8 reduction (the measured numerical win) -----------------------

def naive_reduce_q8(values, weight: int) -> int:
    """Sum of `(x*weight) >> 8` with per-term truncation -- drifts below the exact
    result by up to `len(values)` ULP (each floor loses < 1 ULP, and they accumulate)."""
    acc = 0
    for x in values:
        acc += (x * weight) >> 8
    return acc


def compensated_reduce_q8(values, weight: int) -> int:
    """Residual-carry MAC (integer Kahan/TwoSum analog): carry the truncated low 8
    bits forward. Loop invariant `acc*256 + resid == sum(x*weight so far)`, so the
    result equals `(sum x*weight) >> 8` exactly -- bit-identical to `exact_reduce_q8`,
    with no int64-wide final accumulator needed."""
    acc, resid = 0, 0
    for x in values:
        full = x * weight + resid
        acc += full >> 8
        resid = full & (Q8 - 1)            # the dropped low 8 bits, carried forward
    return acc


def exact_reduce_q8(values, weight: int) -> int:
    """The int64-exact reference: accumulate full-width, shift once at the end."""
    return sum(x * weight for x in values) >> 8


# --- stability diagnostics (two-truth Graded: inform, never legislate) -----------

def cancellation(a: int, b: int) -> Graded:
    """Catastrophic-cancellation diagnostic for a Q8 subtraction `a - b`, as a
    two-truth `Graded`: confidence falls as the leading bits cancel (bits lost ~
    log2(magnitude / |a-b|)). A graded signal -- it informs (e.g. 'prefer a stable
    reformulation here') but must be `decide`-d before it can ever be a verdict."""
    diff = a - b
    mag = max(abs(a), abs(b), 1)
    total = mag.bit_length()
    lost = total if diff == 0 else max(0, total - abs(diff).bit_length())
    conf = max(0, 1000 - (1000 * lost) // max(1, total))
    return Graded(value=diff, confidence_milli=conf, source="precision.cancellation")


def condition_milli(values) -> int:
    """Relative condition of a sum, in milli: 1000 * Σ|x| / |Σx| (>= 1000; large ⇒
    ill-conditioned / cancellation-prone). Deterministic integer ratio."""
    total = sum(values)
    spread = sum(abs(v) for v in values)
    return (1000 * spread) // max(1, abs(total))
