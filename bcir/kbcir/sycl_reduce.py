"""SYCL backend differential oracle -- the REDUCTION (sum) reference + the Q8 bridge for the SYCL/SPIR-V
GPU channel's ``reduce`` capability.

ARCHITECTURE (the same bright line as ``sycl_saxpy.py`` -- see docs/kernel/SYCL_INTEROP.md): SYCL is a *backend
channel* the planner can route to and a *differential oracle* we use to MEASURE that a heterogeneous device
reproduces BCIR's own deterministic reference. It is NOT a ``c.call.libm:`` external-symbol edge -- it is a
single-source C++ *compiler mode* (``-fsycl``). The kernel is emitted SYCL C++ source (a
``sycl::reduction`` + ``q.submit``), not a call to an external symbol, so there is NO ``c.call.libm:`` label
and NO link-flag rule for SYCL. SYCL's dynamic runtime scheduler lives ABOVE the deterministic C rail (the
G8 boundary) and must NEVER touch the legality path (two-truth).

WHY SUM (the reference-verifiability + op choice): ``out = Sum x[i]`` is the canonical ``sycl::reduction``
op -- the operation that SYCL's reduction abstraction exists to express (``sycl::plus<float>()``). Its
reference is a plain-Python sequential sum (no numpy, no other BCIR code), so the oracle is genuinely
reference-verified.

THE ACCURACY SUBTLETY (this is what distinguishes reduce from SAXPY). SAXPY is per-lane independent, so the
device and the scalar fallback agree to float round-off exactly. A reduction is NOT per-lane independent: a
parallel / tree reduction REORDERS the float additions, and float addition is not associative, so the
SYCL tree-reduction result can differ from a sequential sum by a *reduction-order rounding* term (~ n*eps
on the partial-sum magnitude). Honest framing:

  * the oracle (``reduce_reference``) and the emitted *portable fallback* both do a SEQUENTIAL sum, so they
    agree EXACTLY (bit-identical to float round-off);
  * the SYCL *tree-reduction* device path agrees with the sequential reference only within the reorder
    tolerance ``~ n * eps * max|partial|`` (a few ULP for small n; documented, not faked).

So the bridged bound for a quantized reduction is the R17 input round-trip PLUS the reduction-order term:
the round-trip can perturb each input by at most its per-group error ``e_i``, and a sum of n terms
accumulates those at unit gain (``Sum e_i``), and the reorder term adds ``n * eps * S`` where ``S`` bounds
the partial-sum magnitude. See ``reduce_via_bridge`` and ``reduce_reorder_bound``.

The emitted C++ twin is ``lower.c_kernel.emit_sycl_reduce_c`` (a ``sycl::reduction`` on the ``-fsycl`` path,
a portable scalar C++ sum otherwise). The reference does NOT mutate its input and returns a fresh float, so
the oracle is side-effect-free.
"""

from __future__ import annotations

from .quantize import dequantize, quantize_per_group


def reduce_reference(x: list[float]) -> float:
    """The textbook reduction over the dense (unit-stride) vector ``x`` -- the single source of truth every
    realization is checked against, and the canonical ``sycl::reduction`` op:

      out = Sum_i x[i]   (SEQUENTIAL left-to-right sum, ``sycl::plus<float>()``)

    The input is NOT mutated (the oracle is side-effect-free). The sum is the deterministic LEFT-TO-RIGHT
    order, which the emitted portable fallback matches bit-for-bit; a SYCL tree-reduction reorders the adds
    and so agrees only within ``reduce_reorder_bound`` (float add is non-associative). Raises ``ValueError``
    on an empty vector (a reduction needs at least one element)."""
    n = len(x)
    if n < 1:
        raise ValueError("sycl reduce: need at least one element")
    s = 0.0
    for xi in x:
        s += float(xi)
    return s


def reduce_reorder_bound(x: list[float]) -> float:
    """The reduction-ORDER rounding tolerance: how far a parallel/tree reduction (which reassociates the
    float adds) may differ from the sequential ``reduce_reference``. Float addition is not associative, so a
    reordered sum of ``n`` terms accumulates at most about ``(n - 1) * eps * S`` of rounding, where ``S`` is
    a bound on the magnitude of any partial sum (bounded here by ``Sum |x[i]|``, the worst case) and ``eps``
    is the float32 machine epsilon. This is the term ADDED on top of the R17 input round-trip for a tree
    reduction; the sequential fallback needs none of it (it matches the reference exactly)."""
    n = len(x)
    if n < 1:
        raise ValueError("sycl reduce: need at least one element")
    eps_f32 = 2.0**-23  # float32 machine epsilon (the device/fallback use float32)
    s_abs = sum(abs(float(xi)) for xi in x)  # bounds the magnitude of any partial sum
    return (n - 1) * eps_f32 * s_abs


def reduce_via_bridge(x: list[float], group_size: int = 8, bits: int = 8) -> float:
    """The Q8<->float32<->Q8 bridge wrapped around the SYCL reduction differential oracle -- the SYCL analog
    of ``sycl_saxpy.saxpy_via_bridge`` (integrate the device, don't reinvent the arithmetic). The input
    vector arrives as per-group quantized storage; the bridge dequantizes it to float32 (the R17 round-trip)
    then sums the round-tripped values via ``reduce_reference``. So the oracle reference for the quantized
    path is the reduction of the *round-tripped* inputs.

    Error model: a reduction is a SUM, so two error sources compose. (1) The R17 INPUT round-trip perturbs
    each input by at most its per-group error ``e_i``; a sum of ``n`` terms passes those through at unit
    gain, so the quant contribution is ``Sum e_i`` (bounded by ``n * max_i e_i``). (2) If the realization is
    a tree reduction it also reorders the adds, adding ``reduce_reorder_bound`` on top. The SEQUENTIAL
    reference + the portable fallback share the same add order, so for them only (1) applies; the device
    tree-reduction also pays (2)."""
    xq = dequantize(quantize_per_group(x, group_size, bits))
    return reduce_reference(xq)
