"""SEG2.2 -- the 2-D FFT oracle reference + the Q8 bridge wrapped around a TRUSTED external FFT (integrate,
don't reinvent). The 2-D analog of the B2 1-D FFT wrap (``kbcir.fft``) -- a genuinely NEW numerical
capability (a 2-D spectral transform, the kernel under image/convolution spectral methods, not a 1-D
transform) on the SAME trusted FFTW library and the SAME ``c.call.libm:`` FFI edge B5 used for
``cblas_sgemm``. Same trusted-FFTW-vs-portable-fallback contract, same Q8 bridge, same calling-side
ownership -- only the dimensionality (and so the plan entry point, ``fftwf_plan_dft_2d``) differs.

A 2-D forward DFT over an ``n0 x n1`` array is the separable double sum
``out[(j0,j1)] = sum_{k0,k1} in[(k0,k1)] * exp(-2*pi*i*(j0*k0/n0 + j1*k1/n1))``. Its REALIZATION (FFTW's
plan-based row-then-column radix decomposition vs the naive O((n0*n1)^2) sum) does not change the result
-- both compute the identical transform (FFTW's ``FFTW_FORWARD`` is the ``e^{-2*pi*i*(...)}`` sign
convention, UNNORMALIZED: FFTW does NOT divide by N, and neither does this reference) -- but FFTW is
O(N log N) on silicon. BCIR owns the CALLING side: the interleaved complex, ROW-MAJOR layout (a 2-D array
of ``n0`` rows by ``n1`` cols is ``2*n0*n1`` floats; element ``(r,c)`` has its real part at flat index
``2*(r*n1 + c)`` and its imag at ``2*(r*n1 + c)+1`` -- exactly FFTW's ``fftwf_complex`` row-major layout)
and the Q8<->float32 boundary; the transform is DELEGATED to FFTW when linked, with the portable reference
DFT below as the standalone fallback (so CI needs no FFTW installed -- exactly the B5 CBLAS / B2 1-D FFT
pattern). The emitted C twin is ``lower.c_kernel.emit_fftw_fft2_c``.

Complex values are carried as INTERLEAVED float lists so the oracle, the emitted C kernel, and FFTW all
share one byte-for-byte layout -- no ``complex``-vs-interleaved impedance at the boundary. Determinism: the
only floats are the transform's own input/output; the bridge step at the boundary is the R17-certified
Q8<->float32<->Q8 round-trip (``kbcir.precision.quantization_error_bound``).
"""

from __future__ import annotations

import math

from .quantize import dequantize, quantize_per_group


def dft2_reference(x: list[float], n0: int, n1: int) -> list[float]:
    """The naive forward 2-D DFT, the single source of truth every realization is checked against:
    ``out[(j0,j1)] = sum_{k0,k1} x[(k0,k1)] * exp(-2*pi*i*(j0*k0/n0 + j1*k1/n1))``. `x` is `2*n0*n1`
    interleaved floats in ROW-MAJOR order (element `(r,c)`'s Re at `x[2*(r*n1+c)]`, Im at `x[2*(r*n1+c)+1]`);
    the result is `2*n0*n1` interleaved floats in the same layout. This is the SAME forward transform FFTW's
    ``FFTW_FORWARD`` computes (UNNORMALIZED -- no 1/N), and the same the emitted C fallback computes -- so the
    wrapped FFTW call and this reference agree to float round-off. Side-effect-free."""
    if n0 < 1:
        raise ValueError(f"fft2 rows (n0) must be >= 1; got {n0}")
    if n1 < 1:
        raise ValueError(f"fft2 cols (n1) must be >= 1; got {n1}")
    if len(x) != 2 * n0 * n1:
        raise ValueError(
            f"interleaved input must have 2*n0*n1 = {2 * n0 * n1} floats; got {len(x)}"
        )
    out = [0.0] * (2 * n0 * n1)
    for j0 in range(n0):
        for j1 in range(n1):
            re = 0.0
            im = 0.0
            for k0 in range(n0):
                for k1 in range(n1):
                    # reduced phase PER DIMENSION (matches the C kernel's `(j0*k0)%n0` / `(j1*k1)%n1`):
                    # stable for large dims, and the only floats are the transform's own values.
                    ang = -2.0 * math.pi * (((j0 * k0) % n0) / n0 + ((j1 * k1) % n1) / n1)
                    c = math.cos(ang)
                    s = math.sin(ang)
                    base = 2 * (k0 * n1 + k1)
                    xr = x[base]
                    xi = x[base + 1]
                    re += xr * c - xi * s
                    im += xr * s + xi * c
            o = 2 * (j0 * n1 + j1)
            out[o] = re
            out[o + 1] = im
    return out


def fft2_via_bridge(x: list[float], n0: int, n1: int, group_size: int, bits: int) -> list[float]:
    """SEG2.2: the Q8<->float32<->Q8 bridge wrapped around a TRUSTED external 2-D FFT (integrate, don't
    reinvent) -- the 2-D analog of ``fft.fft_via_bridge``. The interleaved complex input arrives as per-group
    quantized storage; the bridge dequantizes it to float32, a trusted external FFT (FFTW's
    ``fftwf_plan_dft_2d``, via the ``c.call.libm:`` edge) computes the transform, and BCIR owns the calling
    side (the row-major interleaved layout + the boundary quantization). So the oracle reference for the
    quantized path is the 2-D DFT of the *round-tripped* inputs -- the SOLE error vs the true transform is the
    INPUT quantization (R17-certified by ``quantization_error_bound``), the FFT itself being exact/trusted.
    Mirrors the 1-D bridge exactly."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return dft2_reference(dx, n0, n1)
