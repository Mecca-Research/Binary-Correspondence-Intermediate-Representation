"""Area-B breadth (#63) -- the VECTORIZED TRANSCENDENTAL MATH oracle reference + the Q8 bridge wrapped
around a TRUSTED external SLEEF (SIMD-oriented vectorized libm) call (a vectorized elementwise ``exp`` over
a float array; integrate, don't reinvent). The FIFTH and final Area-B library after B5's BLAS ``sgemm`` (a
matmul), B2's FFTW (a spectral transform), #61's LAPACK ``sgesv`` (a linear solve), and #62's GSL
(statistics / special functions) -- SLEEF's distinctive domain is a FAST, VECTORIZED libm (transcendental
math), neither linear algebra, an FFT, nor statistics, so it does not overlap the existing wraps. Wrapped
through the SAME trusted-external ``c.call.libm:`` FFI edge.

WHY VECTORIZED EXP (the reference-verifiability + domain choice): SLEEF's signature contribution is a
vectorized transcendental library -- ``exp`` is the clean, canonical pick (the same elementwise ``expf``
the G1 activation kernels already route through the libm edge, of which SLEEF is the vectorized
alternative). The reference is a TRIVIALLY EXACT, independent per-element ``math.exp`` (computed in the
test with the stdlib, not this module's own code), so the oracle is genuinely reference-verified to float
round-off. The transcendental itself is TRUSTED (SLEEF when linked, plain libm ``expf`` per element
otherwise) -- both compute the identical elementwise ``exp`` to float round-off, exactly as the G1
activation libm edge quarantines its ``expf``.

SLEEF CONVENTIONS (mirrored exactly): the emitted C wraps ``Sleef_expf1_u10`` -- the SCALAR
single-precision ``exp`` at 1.0-ULP accuracy (``f1`` == one f32 lane, ``u10`` == <= 1.0 ULP). The scalar
form is chosen for PORTABILITY: it needs no target-ISA vector type (``__m256``/``svfloat32_t``...), so the
SAME emitted source builds on any ISA, and BCIR owns the calling side (the elementwise loop over the dense
array) -- the vectorization is the trusted library's job (or the compiler's, over the portable per-element
fallback). ``Sleef_expf1_u10`` is the elementwise twin of libm ``expf``; both compute ``e^x`` per lane.

BCIR owns the CALLING side: the dense contiguous (unit-stride) ``float`` array layout and the
Q8<->float32 boundary; the transcendental is DELEGATED to SLEEF (``Sleef_expf1_u10``) when linked, with
the portable reference below (plain libm ``expf`` per element) as the standalone fallback (so CI needs no
SLEEF installed -- exactly the B5 CBLAS / B2 FFTW / #61 LAPACK / #62 GSL pattern). The emitted C twin is
``lower.c_kernel.emit_sleef_exp_c``.

The reference does NOT mutate its input and returns a fresh list, so the oracle is side-effect-free.
Determinism: the only floats are the transform's own input/output; the bridge step at the boundary is the
R17-certified Q8<->float32<->Q8 round-trip (``kbcir.precision.quantization_error_bound``). The
transcendental itself is trusted/exact (SLEEF or the libm reference); the bridge bound is the INPUT
round-trip alone (the transcendental is trusted, exactly like the G1 activation libm edge).
"""

from __future__ import annotations

import math

from .quantize import dequantize, quantize_per_group


def exp_reference(data: list[float]) -> list[float]:
    """The textbook vectorized elementwise ``exp`` over the dense (unit-stride) vector ``data`` -- the
    single source of truth every realization is checked against, and the EXACT elementwise transform SLEEF
    ``Sleef_expf1_u10`` realizes (and the emitted C fallback's per-element ``expf``):

      out[i] = exp(data[i])

    Input is NOT mutated (the oracle is side-effect-free). The transcendental is per-element ``math.exp`` --
    trusted/exact, the same ``e^x`` SLEEF and libm compute, so the wrapped SLEEF call and this reference
    agree to float round-off. Raises on an empty vector (an elementwise map needs at least one lane)."""
    n = len(data)
    if n < 1:
        raise ValueError("sleef exp: need at least one element")
    return [math.exp(float(v)) for v in data]


def exp_via_bridge(data: list[float], group_size: int, bits: int) -> list[float]:
    """Area-B breadth: the Q8<->float32<->Q8 bridge wrapped around a TRUSTED external SLEEF vectorized
    ``exp`` (integrate, don't reinvent) -- the SLEEF analog of ``matmul.gemm_via_bridge`` /
    ``fft.fft_via_bridge`` / ``linsolve.solve_via_bridge`` / ``gsl_kernels.stats_via_bridge``. The data
    arrives as per-group quantized storage; the bridge dequantizes it to float32, a trusted external
    transcendental (SLEEF ``Sleef_expf1_u10``, via the ``c.call.libm:`` edge) is applied per element, and
    BCIR owns the calling side (the dense unit-stride layout + the boundary quantization). So the oracle
    reference for the quantized path is the ``exp`` of the *round-tripped* inputs -- the SOLE certified
    boundary error is the INPUT quantization (R17-certified by ``quantization_error_bound``), the
    transcendental itself being trusted/exact (like the G1 activation libm edge). Mirrors B5/B2/#61/#62
    exactly.

    Conditioning note: ``exp`` amplifies an input perturbation by ``exp(x)`` (its own derivative), so a
    per-input round-trip error e perturbs ``exp(x)`` by ~``exp(x)*e`` -- the bound certified here is the
    INPUT round-trip step alone (keep the bridged data well-scaled). The downstream transcendental is
    trusted/exact and contributes no certified error; this matches the B5/B2/#61/#62 contract where the
    wrapped kernel is exact."""
    dq = dequantize(quantize_per_group(data, group_size, bits))
    return exp_reference(dq)
