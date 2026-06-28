"""B2 -- the 1-D FFT oracle reference + the Q8 bridge wrapped around a TRUSTED external FFT (integrate,
don't reinvent). The B2 analog of ``kbcir.matmul`` for a genuinely NEW kernel class -- a spectral
transform, not a matmul -- wrapped through the same trusted-external FFI edge B5 used for ``cblas_sgemm``.

A 1-D forward DFT is ``out[j] = sum_k in[k] * exp(-2*pi*i*j*k/n)``. Its REALIZATION (FFTW's
plan-based radix decomposition vs the naive O(n^2) sum) does not change the result -- both compute the
identical transform (FFTW's ``FFTW_FORWARD`` is the ``e^{-2*pi*i*jk/n}`` sign convention) -- but FFTW is
O(n log n) on silicon. BCIR owns the CALLING side: the interleaved complex layout (``X[2k]=Re,
X[2k+1]=Im``, the ``fftwf_complex`` memory layout) and the Q8<->float32 boundary; the transform is
DELEGATED to FFTW when linked, with the portable reference DFT below as the standalone fallback (so CI
needs no FFTW installed -- exactly the B5 CBLAS pattern). The emitted C twin is
``lower.c_kernel.emit_fftw_fft_c``.

Complex values are carried as INTERLEAVED float lists (``[re0, im0, re1, im1, ...]``) so the oracle, the
emitted C kernel, and FFTW all share one byte-for-byte layout -- no ``complex``-vs-interleaved impedance
at the boundary. Determinism: the only floats are the transform's own input/output; the bridge step at
the boundary is the R17-certified Q8<->float32<->Q8 round-trip (``kbcir.precision.quantization_error_bound``).
"""

from __future__ import annotations

import math

from .quantize import dequantize, quantize_per_group


def dft_reference(x: list[float], n: int) -> list[float]:
    """The naive forward DFT, the single source of truth every realization is checked against:
    ``out[j] = sum_k x[k] * exp(-2*pi*i*j*k/n)``. `x` is `2*n` interleaved floats (`x[2k]`=Re,
    `x[2k+1]`=Im); the result is `2*n` interleaved floats. This is the SAME forward transform FFTW's
    ``FFTW_FORWARD`` computes (and the same the emitted C fallback computes) -- so the wrapped FFTW call
    and this reference agree to float round-off."""
    if n < 1:
        raise ValueError(f"fft length must be >= 1; got {n}")
    if len(x) != 2 * n:
        raise ValueError(f"interleaved input must have 2*n = {2 * n} floats; got {len(x)}")
    out = [0.0] * (2 * n)
    for j in range(n):
        re = 0.0
        im = 0.0
        for k in range(n):
            ang = -2.0 * math.pi * ((j * k) % n) / n      # reduced phase: matches the C kernel's `(j*k)%n`
            c = math.cos(ang)
            s = math.sin(ang)
            xr = x[2 * k]
            xi = x[2 * k + 1]
            re += xr * c - xi * s
            im += xr * s + xi * c
        out[2 * j] = re
        out[2 * j + 1] = im
    return out


def fft_via_bridge(x: list[float], n: int, group_size: int, bits: int) -> list[float]:
    """B2: the Q8<->float32<->Q8 bridge wrapped around a TRUSTED external FFT (integrate, don't reinvent) --
    the FFT analog of ``matmul.gemm_via_bridge`` / ``activation.activation_via_bridge``. The interleaved
    complex input arrives as per-group quantized storage; the bridge dequantizes it to float32, a trusted
    external FFT (FFTW, via the ``c.call.libm:`` edge) computes the transform, and BCIR owns the calling
    side (the interleaved layout + the boundary quantization). So the oracle reference for the quantized
    path is the DFT of the *round-tripped* inputs -- the SOLE error vs the true transform is the INPUT
    quantization (R17-certified by ``quantization_error_bound``), the FFT itself being exact/trusted.
    Mirrors B5 exactly."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return dft_reference(dx, n)
