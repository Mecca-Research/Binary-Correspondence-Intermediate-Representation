"""Area-B breadth (#62) -- the GSL STATISTICS oracle reference + the Q8 bridge wrapped around a TRUSTED
external GSL (GNU Scientific Library) call (mean / variance / standard deviation; integrate, don't
reinvent). A FOURTH, distinct library after B5's BLAS ``sgemm`` (a matmul), B2's FFTW (a spectral
transform), and #61's LAPACK ``sgesv`` (a linear solve) -- GSL's distinctive domain is SPECIAL FUNCTIONS /
STATISTICS, neither linear algebra nor an FFT, so it does not overlap the existing wraps. Wrapped through
the SAME trusted-external ``c.call.libm:`` FFI edge.

WHY STATISTICS (the reference-verifiability choice): the task asks for the GSL kernel that is MOST cleanly
reference-verifiable. GSL statistics has a TRIVIALLY EXACT, independent reference -- the textbook
``mean = (1/n) sum x``, the UNBIASED (sample) ``variance = (1/(n-1)) sum (x-mean)^2`` (GSL's ``n-1``
divisor), and ``sd = sqrt(variance)``. The reference is a plain sum / division (bit-exact-class in float64
against an independent recompute), with the variance/sd square root the SOLE transcendental -- and that
sqrt rides the trusted ``c.call.libm:sqrtf`` edge exactly as the activation/attention kernels quarantine
their transcendental. A special function (``gsl_sf_erf`` etc.) would only be verifiable "to float
round-off" against a libm twin whose Chebyshev round-off differs subtly; statistics gives the cleanest,
genuinely-independent reference. So: a distinct GSL domain AND the strongest reference-verifiability.

GSL CONVENTIONS (mirrored exactly):
  * ``gsl_stats_mean(data, stride, n)``      = (1/n) * sum_i data[i]
  * ``gsl_stats_variance(data, stride, n)``  = (1/(n-1)) * sum_i (data[i] - mean)^2   (UNBIASED: n-1 divisor)
  * ``gsl_stats_sd(data, stride, n)``        = sqrt(gsl_stats_variance(...))
A unit stride is assumed (the dense contiguous vector BCIR owns the layout for). The variance/sd need
``n >= 2`` (the ``n-1`` divisor); the mean needs ``n >= 1``.

BCIR owns the CALLING side: the dense contiguous (unit-stride) layout and the Q8<->float32 boundary; the
statistic is DELEGATED to GSL (``gsl_stats_mean`` / ``gsl_stats_variance`` / ``gsl_stats_sd``) when linked,
with the portable reference below as the standalone fallback (so CI needs no GSL installed -- exactly the
B5 CBLAS / B2 FFTW / #61 LAPACK pattern). The emitted C twin is ``lower.c_kernel.emit_gsl_stats_c``.

The reference does NOT mutate its input and returns a fresh scalar, so the oracle is side-effect-free.
Determinism: the only floats are the statistic's own input/output; the bridge step at the boundary is the
R17-certified Q8<->float32<->Q8 round-trip (``kbcir.precision.quantization_error_bound``). The statistic
itself is trusted/exact (GSL or the reference); the bridge bound is the INPUT round-trip alone.
"""

from __future__ import annotations

import math

from .quantize import dequantize, quantize_per_group

# The statistics this wrap covers, in GSL naming -- the source of truth for both the reference dispatch
# and the emitted C kernel's branch labels.
GSL_STATS = ("mean", "variance", "sd")


def stats_reference(data: list[float], kind: str) -> float:
    """The textbook GSL statistic for the dense (unit-stride) vector ``data`` -- the single source of truth
    every realization is checked against, and the EXACT formula GSL ``gsl_stats_<kind>`` realizes:

      mean      = (1/n) * sum_i data[i]
      variance  = (1/(n-1)) * sum_i (data[i] - mean)^2     (UNBIASED -- GSL's n-1 sample-variance divisor)
      sd        = sqrt(variance)

    `kind` is one of ``GSL_STATS`` ("mean" / "variance" / "sd"). Input is NOT mutated (the oracle is
    side-effect-free). The mean is a pure sum/n -- exact-class against an independent recompute; the
    variance is a pure sum-of-squared-deviations / (n-1); only the sd's square root is transcendental (it
    rides the trusted libm sqrt in the emitted C). Raises on an empty vector (mean) or n<2 (variance/sd,
    the n-1 divisor) -- the honest domain error of a sample statistic."""
    if kind not in GSL_STATS:
        raise ValueError(f"unknown GSL statistic {kind!r}; expected one of {GSL_STATS}")
    n = len(data)
    if n < 1:
        raise ValueError("gsl_stats: need at least one sample for the mean")
    mean = math.fsum(float(v) for v in data) / n  # (1/n) sum x  -- GSL gsl_stats_mean
    if kind == "mean":
        return mean
    if n < 2:
        raise ValueError("gsl_stats: variance/sd need at least two samples (the n-1 divisor)")
    var = math.fsum((float(v) - mean) ** 2 for v in data) / (
        n - 1
    )  # UNBIASED variance (n-1 divisor)
    if kind == "variance":
        return var
    return math.sqrt(var)  # sd = sqrt(variance) -- the only transcendental


def stats_via_bridge(data: list[float], kind: str, group_size: int, bits: int) -> float:
    """Area-B breadth: the Q8<->float32<->Q8 bridge wrapped around a TRUSTED external GSL statistic
    (integrate, don't reinvent) -- the GSL analog of ``matmul.gemm_via_bridge`` / ``fft.fft_via_bridge`` /
    ``linsolve.solve_via_bridge``. The data arrives as per-group quantized storage; the bridge dequantizes
    it to float32, a trusted external statistic (GSL ``gsl_stats_<kind>``, via the ``c.call.libm:`` edge) is
    computed, and BCIR owns the calling side (the dense unit-stride layout + the boundary quantization). So
    the oracle reference for the quantized path is the statistic of the *round-tripped* inputs -- the SOLE
    certified boundary error is the INPUT quantization (R17-certified by ``quantization_error_bound``), the
    statistic itself being trusted/exact. Mirrors B5/B2/#61 exactly.

    Conditioning note: the mean is a 1-Lipschitz average of the inputs, so a per-input round-trip error e
    perturbs the mean by at most e -- the bound certified here is the INPUT round-trip step alone (keep the
    bridged data well-scaled). The downstream statistic is trusted/exact and contributes no certified error;
    this matches the B5/B2/#61 contract where the wrapped kernel is exact."""
    dq = dequantize(quantize_per_group(data, group_size, bits))
    return stats_reference(dq, kind)
