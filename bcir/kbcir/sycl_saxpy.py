"""SYCL backend differential oracle -- the SAXPY reference + the Q8 bridge for a SYCL/SPIR-V GPU channel.

ARCHITECTURE (the bright line you must keep -- see docs/kernel/SYCL_INTEROP.md): SYCL is a *backend channel* the
planner can route to (priced by the K_BCIR cost model on the ``sycl_spirv`` profile) and a *differential
oracle* we use to MEASURE that a heterogeneous device reproduces BCIR's own deterministic reference. It is
NOT a trusted-library FFI edge. Unlike the FIVE Area-B library wraps (BLAS/FFTW/LAPACK/GSL/SLEEF, all routed
through the ``c.call.libm:`` external-symbol edge with a ``-l<lib>`` link rule), SYCL is a single-source C++
*compiler mode* (``-fsycl``: ``icpx -fsycl`` / ``acpp`` / ``clang++ -fsycl``). The kernel is emitted SYCL
C++ source (a ``parallel_for``), not a call to an external symbol -- so there is NO ``c.call.libm:`` label
and NO link-flag rule for SYCL (asserted in bcir/tests/test_sycl_channel.py). SYCL's dynamic runtime
scheduler lives ABOVE the deterministic C rail (the G8 boundary) and must NEVER touch the legality path.

WHY SAXPY (the reference-verifiability + op choice): ``out[i] = a*x[i] + y[i]`` is the canonical
data-parallel op -- the elementwise ``parallel_for`` SYCL exists to express. Its reference is TRIVIALLY
EXACT: a per-element multiply-add in plain Python float (computed here with no numpy and no other BCIR
code), so the oracle is genuinely reference-verified to float round-off. SAXPY is PURE arithmetic (one
multiply + one add per lane -- NO transcendental), so unlike the SLEEF ``exp`` wrap the *kernel itself*
introduces no certified error: the SYCL device path and the portable scalar fallback compute the IDENTICAL
SAXPY to float round-off, and the ONLY boundary error is the R17 Q8<->float32<->Q8 input round-trip.

BCIR owns the CALLING side: the dense contiguous (unit-stride) ``float`` array layout and the Q8<->f32
boundary; the device merely realizes the elementwise map. The emitted C++ twin is
``lower.c_kernel.emit_sycl_saxpy_c`` (a ``parallel_for`` on the ``-fsycl`` path, a portable scalar C++ loop
otherwise). The reference does NOT mutate its inputs and returns a fresh list, so the oracle is
side-effect-free.
"""

from __future__ import annotations

from .quantize import dequantize, quantize_per_group


def saxpy_reference(a: float, x: list[float], y: list[float]) -> list[float]:
    """The textbook SAXPY over the dense (unit-stride) vectors ``x`` and ``y`` -- the single source of
    truth every realization is checked against, and the EXACT elementwise map the SYCL ``parallel_for``
    realizes (and the emitted C++ fallback's scalar loop):

      out[i] = a * x[i] + y[i]

    Inputs are NOT mutated (the oracle is side-effect-free) and a fresh list is returned. The arithmetic is
    one multiply + one add per lane (pure float, no transcendental), so the device and the scalar fallback
    agree to float round-off. Raises ``ValueError`` on an empty vector (a map needs at least one lane) or a
    length mismatch (``len(x) != len(y)``)."""
    n = len(x)
    if n < 1:
        raise ValueError("sycl saxpy: need at least one element")
    if len(y) != n:
        raise ValueError(f"sycl saxpy: x/y length mismatch ({n} vs {len(y)})")
    af = float(a)
    return [af * float(xi) + float(yi) for xi, yi in zip(x, y)]


def saxpy_via_bridge(
    a: float, x: list[float], y: list[float], group_size: int = 8, bits: int = 8
) -> list[float]:
    """The Q8<->float32<->Q8 bridge wrapped around the SYCL SAXPY differential oracle -- the SYCL analog of
    ``vecmath.exp_via_bridge`` / ``matmul.gemm_via_bridge`` (integrate the device, don't reinvent the
    arithmetic). The input vectors arrive as per-group quantized storage; the bridge dequantizes them to
    float32 (the R17 round-trip), then computes the SAXPY on the round-tripped values via
    ``saxpy_reference``. So the oracle reference for the quantized path is the SAXPY of the *round-tripped*
    inputs.

    Error model (tighter than the SLEEF wrap because SAXPY is exact arithmetic): SAXPY has NO transcendental
    -- it is one multiply + one add -- so the ONLY certified boundary error is the INPUT quantization
    (R17-certified by ``quantization_error_bound``). A per-input round-trip error e on x is scaled by ``|a|``
    (the multiply), and the round-trip error on y passes through the add at unit gain; so per element the
    bridged result is perturbed by at most ``(|a| + 1) * e``. The downstream multiply-add is exact (no
    certified error of its own -- like the dense BLAS gemm path), unlike the exp wrap whose transcendental
    amplifies by ``exp(x)``."""
    af = float(a)
    xq = dequantize(quantize_per_group(x, group_size, bits))
    yq = dequantize(quantize_per_group(y, group_size, bits))
    return saxpy_reference(af, xq, yq)
