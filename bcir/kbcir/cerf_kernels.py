"""Area-B breadth (SEG2) -- the NUMERICALLY-ROBUST SPECIAL-FUNCTION oracle reference + the Q8 bridge
wrapped around a TRUSTED external libcerf (the complex/scaled error-function library) call (the SCALED
COMPLEMENTARY ERROR FUNCTION ``erfcx(x) = e^{x^2} * erfc(x)`` over a float array; integrate, don't
reinvent). The SIXTH Area-B library after B5's BLAS ``sgemm`` (a matmul), B2's FFTW (a spectral
transform), #61's LAPACK ``sgesv`` (a linear solve), #62's GSL (statistics / special functions), and
#63's SLEEF (a fast, vectorized libm) -- libcerf's distinctive domain is NUMERICALLY-ROBUST SPECIAL
FUNCTIONS (the error-function family computed without overflow across the full real line), neither
linear algebra, an FFT, statistics, nor a fast vectorized libm, so it does not overlap the existing
wraps. Wrapped through the SAME trusted-external ``c.call.libm:`` FFI edge.

WHY erfcx (the capability + domain choice): ``erfcx(x) = e^{x^2} * erfc(x)`` is a function the standard
C math library does NOT provide. Computing it naively as ``expf(x*x)*erfcf(x)`` OVERFLOWS for moderately
large x -- ``exp(x^2)`` blows past float range around ``|x| >~ 9`` and past double range around
``|x| >~ 26`` -- yet ``erfcx`` itself stays finite and well-scaled across the whole real line (it decays
like ``1/(x*sqrt(pi))`` for large x). libcerf's ``erfcx`` is numerically robust there (no overflow). So
``erfcx`` is a symbol libm genuinely LACKS (unlike #63's SLEEF exp, which duplicates libm ``expf``) --
this wrap ADDS A CAPABILITY, not just a faster path. The oracle reference is a TRIVIALLY EXACT,
independent ``math.exp(x*x)*math.erfc(x)`` (computed in the test with the stdlib, not this module's own
code), so the oracle is genuinely reference-verified to float round-off WITHIN the tested range. The
function itself is TRUSTED (libcerf ``erfcxf`` when linked, the naive ``expf(x*x)*erfcf(x)`` per element
otherwise) -- both compute the identical ``erfcx`` to float round-off within the bounded test envelope
(``|x| <= 3``, where ``exp(x^2) <= exp(9) ~ 8103`` never overflows and the naive form stays accurate).
The library's advantage -- full-real-line robustness -- lies OUTSIDE the bounded correctness envelope,
EXACTLY as #63's SLEEF advantage (vectorization speed) lies outside its correctness check: the win is on
the calling side + delegation to a trusted library, not a reimplemented special function.

libcerf CONVENTIONS (mirrored exactly): the emitted C wraps ``erfcxf`` -- the float SCALAR form
(``float erfcxf(float)``) from ``<cerf.h>``, the single-precision twin of the dense unit-stride float
layout BCIR owns. The scalar form is chosen for the same reason as #63's ``Sleef_expf1_u10`` (one f32
lane, no target-ISA vector type): BCIR owns the calling side (the elementwise loop over the dense array)
and DELEGATES the special function to libcerf.

BCIR owns the CALLING side: the dense contiguous (unit-stride) ``float`` array layout and the
Q8<->float32 boundary; the special function is DELEGATED to libcerf (``erfcxf``) when linked
(``-DBCIR_USE_CERF -lcerf``), with the portable reference below (the naive ``expf(x*x)*erfcf(x)`` per
element, ``-lm``) as the standalone fallback (so CI needs no libcerf installed -- exactly the B5 CBLAS /
B2 FFTW / #61 LAPACK / #62 GSL / #63 SLEEF pattern). The emitted C twin is
``lower.c_kernel.emit_cerf_erfcx_c``.

The reference does NOT mutate its input and returns a fresh list, so the oracle is side-effect-free.
Determinism: the only floats are the transform's own input/output; the bridge step at the boundary is the
R17-certified Q8<->float32<->Q8 round-trip (``kbcir.precision.quantization_error_bound``). The special
function itself is trusted/exact (libcerf or the naive reference); the bridge bound is the INPUT
round-trip alone (the special function is trusted, exactly like the G1 activation libm edge).
"""

from __future__ import annotations

import math

from .quantize import dequantize, quantize_per_group


def erfcx_reference(data: list[float]) -> list[float]:
    """The textbook scaled complementary error function over the dense (unit-stride) vector ``data`` --
    the single source of truth every realization is checked against, and the EXACT elementwise transform
    libcerf ``erfcxf`` realizes (and the emitted C fallback's per-element ``expf(x*x)*erfcf(x)``):

      out[i] = exp(data[i]*data[i]) * erfc(data[i])     (== erfcx(data[i]))

    Input is NOT mutated (the oracle is side-effect-free). The special function is the definitional
    ``math.exp(v*v)*math.erfc(v)`` -- trusted/exact, the same ``erfcx`` libcerf computes, so the wrapped
    libcerf call and this reference agree to float round-off within the tested range. Raises on an empty
    vector (an elementwise map needs at least one lane)."""
    n = len(data)
    if n < 1:
        raise ValueError("cerf erfcx: need at least one element")
    return [math.exp(float(v) * float(v)) * math.erfc(float(v)) for v in data]


def erfcx_via_bridge(data: list[float], group_size: int, bits: int) -> list[float]:
    """Area-B breadth: the Q8<->float32<->Q8 bridge wrapped around a TRUSTED external libcerf scaled
    complementary error function (integrate, don't reinvent) -- the libcerf analog of
    ``matmul.gemm_via_bridge`` / ``fft.fft_via_bridge`` / ``linsolve.solve_via_bridge`` /
    ``gsl_kernels.stats_via_bridge`` / ``vecmath.exp_via_bridge``. The data arrives as per-group
    quantized storage; the bridge dequantizes it to float32, a trusted external special function (libcerf
    ``erfcxf``, via the ``c.call.libm:`` edge) is applied per element, and BCIR owns the calling side (the
    dense unit-stride layout + the boundary quantization). So the oracle reference for the quantized path
    is the ``erfcx`` of the *round-tripped* inputs -- the SOLE certified boundary error is the INPUT
    quantization (R17-certified by ``quantization_error_bound``), the special function itself being
    trusted/exact (like the G1 activation libm edge). Mirrors B5/B2/#61/#62/#63 exactly.

    Conditioning note: the local Lipschitz constant of ``erfcx`` is its own derivative,
    ``d/dx erfcx(x) = 2*x*erfcx(x) - 2/sqrt(pi)``, so a per-input round-trip error e perturbs
    ``erfcx(x)`` by ~``|2*x*erfcx(x) - 2/sqrt(pi)| * e`` -- the bound certified here is the INPUT
    round-trip step alone (keep the bridged data well-scaled). The downstream special function is
    trusted/exact and contributes no certified error; this matches the B5/B2/#61/#62/#63 contract where
    the wrapped kernel is exact."""
    dq = dequantize(quantize_per_group(data, group_size, bits))
    return erfcx_reference(dq)
