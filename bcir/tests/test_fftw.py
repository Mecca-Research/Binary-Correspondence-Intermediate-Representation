"""B2 -- wrap a trusted FFTW 1-D complex FFT through the c.call.libm: edge, with the A1.1 Q8 bridge at the
boundary (integrate, don't reinvent -- a genuinely NEW kernel class, a spectral transform, mirroring B5's
BLAS wrap). BCIR owns the calling side (the interleaved complex layout + quantization); the transform is
delegated to FFTW when linked, with a portable reference DFT fallback.

Covers: the bridged FFT tracks the float DFT reference within the quantization error (R17-certified input
bound); the emitted C wraps fftwf_plan_dft_1d/execute/destroy with a naive O(n^2) DFT reference fallback;
the fallback path compiles and reproduces the DFT reference under clang (no FFTW needed -- it IS a DFT, so
linked-vs-fallback agree to float round-off); only if an FFTW lib is present, the linked path agrees too;
and the B2 FFTW link-flag rule resolves fftwf_*/fftw_* -> -lfftw3 (with the C twin agreeing). The point:
the win is on the calling side, not a reimplemented FFT."""

import cmath
import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.fft import dft_reference, fft_via_bridge
from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
from bcir.lower.c_kernel import emit_fftw_fft_c
from bcir.model import Claim, Domain, Lane, Opcode, StrideClass
from bcir.toolchain import host_link_args


def _fftw_link():
    """An FFTW lib flag that links, or None (the real-FFTW path then self-skips honestly)."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.c")
        open(src, "w").write("void fftwf_execute();\nint main(void){return 0;}\n")
        if subprocess.run([cc, src, "-lfftw3", "-o", os.path.join(d, "p")],
                          capture_output=True).returncode == 0:
            return "-lfftw3"
    return None


def _interleave(signal):
    """A list of complex numbers -> the interleaved [re0, im0, re1, im1, ...] float layout."""
    out = []
    for z in signal:
        out.append(float(z.real))
        out.append(float(z.imag))
    return out


def _numpy_free_dft(signal):
    """An INDEPENDENT DFT reference using only cmath (no numpy, no BCIR code): the textbook definition
    out[j] = sum_k x[k]*exp(-2*pi*i*j*k/n). The oracle dft_reference must agree with this."""
    n = len(signal)
    return [sum(signal[k] * cmath.exp(-2j * math.pi * j * k / n) for k in range(n)) for j in range(n)]


# --- the oracle DFT reference agrees with an independent textbook DFT ------------------------------

def test_dft_reference_matches_an_independent_dft():
    rng = random.Random(0xB2)
    n = 8
    signal = [complex(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(n)]
    ref = dft_reference(_interleave(signal), n)
    want = _numpy_free_dft(signal)                       # independent of BCIR's code
    for j in range(n):
        assert abs(ref[2 * j] - want[j].real) <= 1e-9, j      # real part
        assert abs(ref[2 * j + 1] - want[j].imag) <= 1e-9, j  # imaginary part


def test_dft_reference_of_a_pure_tone_is_a_single_bin():
    # A non-trivial signal: a pure cosine of frequency 1 over n samples -> energy only in bins 1 and n-1.
    n = 16
    signal = [complex(math.cos(2 * math.pi * t / n), 0.0) for t in range(n)]
    ref = dft_reference(_interleave(signal), n)
    mag = [math.hypot(ref[2 * j], ref[2 * j + 1]) for j in range(n)]
    for j in range(n):
        if j in (1, n - 1):
            assert abs(mag[j] - n / 2) < 1e-6, j         # the tone's two conjugate bins, magnitude n/2
        else:
            assert mag[j] < 1e-6, j                       # every other bin is empty


def test_dft_reference_rejects_malformed_inputs():
    for bad in [([], 0), ([1.0, 2.0], 2), ([1.0, 2.0, 3.0], 1)]:
        try:
            dft_reference(*bad); assert False, f"{bad} should raise"
        except ValueError:
            pass


# --- the oracle bridge: tracks the float DFT reference within the quantization error --------------

def test_bridged_fft_tracks_the_reference_within_quant_error():
    rng = random.Random(0xB2F7)
    n = 16
    signal = [complex(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(n)]
    x = _interleave(signal)
    from bcir.kbcir.quantize import max_abs_error
    q_err = max_abs_error(x, group_size=8, bits=8)        # the R17 input round-trip bound (real units)
    ref = dft_reference(x, n)
    got = fft_via_bridge(x, n, group_size=8, bits=8)
    # A length-n DFT bin is a sum of n unit-magnitude rotations of the inputs, so a per-input error e is
    # amplified by at most n in any output bin (|sum| <= sum|.|). Bound the per-bin error by n*q_err.
    assert all(abs(r - g) <= n * q_err + 1e-6 for r, g in zip(ref, got))


def test_bridge_is_a_clean_roundtrip_then_trusted_fft():
    # fft_via_bridge == DFT reference of the dequantized (round-tripped) inputs -- i.e. the trusted FFT
    # sees exactly the bridged values, so the bridge is the only error source (mirrors gemm_via_bridge).
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(17)
    n = 8
    signal = [complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(n)]
    x = _interleave(signal)
    dx = dequantize(quantize_per_group(x, 8, 8))
    assert fft_via_bridge(x, n, 8, 8) == dft_reference(dx, n)


# --- the emitted wrapper: fftw call + portable reference DFT fallback, BCIR owns the layout -------

def test_emit_wraps_fftw_with_a_reference_fallback():
    c = emit_fftw_fft_c(8, "fft")
    assert "fftwf_plan_dft_1d(8, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE)" in c   # the trusted external
    assert "fftwf_execute(plan)" in c and "fftwf_destroy_plan(plan)" in c
    assert "BCIR_USE_FFTW" in c and "BCIR_FFT_FFTW" in c                         # link-selected
    assert "re += xr * c - xi * s" in c and "im += xr * s + xi * c" in c         # the reference DFT fallback
    for bad in (0, -1):
        try:
            emit_fftw_fft_c(bad); assert False, f"{bad} should raise"
        except ValueError:
            pass


def test_fallback_path_compiles_and_matches_the_reference():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return                                              # quick tier hides the toolchain -> self-skip
    rng = random.Random(0x5EED)
    n = 12
    signal = [complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(n)]
    x = _interleave(signal)
    ref = dft_reference(x, n)
    kernel = emit_fftw_fft_c(n, "fft")
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float in[{2 * n}] = {{{', '.join(f'{v:.6f}f' for v in x)}}};\n"
            f"  float out[{2 * n}];\n  fft(in, out);\n"
            f'  for (int i=0;i<{2 * n};++i) printf("%.6f ", out[i]);\n  return 0;\n}}\n')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "f.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "f")
        # the fallback path needs NO FFTW lib (only -lm for cosf/sinf), exactly as B5's fallback needs no BLAS.
        bld = subprocess.run(host_link_args([cc, "-std=c11", "-O2", src, "-lm", "-o", exe]),
                             capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        got = [float(t) for t in out.stdout.split()]
        # the C fallback IS the same DFT (float32 vs the oracle's float64) -> agree to float round-off.
        assert len(got) == 2 * n
        assert all(abs(g - r) < 1e-3 for g, r in zip(got, ref)), (got, ref)


def test_linked_fftw_path_agrees_when_fftw_is_present():
    lib = _fftw_link()
    if not lib:
        return                                              # no FFTW here -> the real-link path self-skips
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    rng = random.Random(0xF7BA)
    n = 16
    signal = [complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(n)]
    x = _interleave(signal)
    ref = dft_reference(x, n)
    kernel = emit_fftw_fft_c(n, "fft")
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float in[{2 * n}] = {{{', '.join(f'{v:.6f}f' for v in x)}}};\n"
            f"  float out[{2 * n}];\n  fft(in, out);\n"
            f'  for (int i=0;i<{2 * n};++i) printf("%.6f ", out[i]);\n  return 0;\n}}\n')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "f.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "f")
        bld = subprocess.run(host_link_args(
            [cc, "-std=c11", "-O2", "-DBCIR_USE_FFTW", src, lib, "-lm", "-o", exe]),
                             capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        got = [float(t) for t in out.stdout.split()]
        # FFTW computes the identical forward transform (FFTW_FORWARD == e^{-2pi i jk/n}); agree to round-off.
        assert all(abs(g - r) < 1e-2 for g, r in zip(got, ref)), (got, ref)


# --- R17 boundary: the bridge step is certified on the wrapped call -------------------------------

def test_r17_certifies_the_bridge_on_a_quantized_fft_call():
    # a wrapped FFT that consumes quantized lanes carries the bridge's round-trip step in its bound (the
    # trusted FFTW kernel itself is exact); a dense call does not. Mirrors test_blas_gemm exactly.
    def call(qbits):
        return Claim(id=7200, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT, count=1,
                     rd=(70, 71), wr=(72,), op="call.fft", domain=Domain.RAM, quantized_bits=qbits)
    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1     # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1                              # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1                          # the R17 grid bound the bridge is held to


# --- the B2 link-flag rule: an FFTW edge resolves to -lfftw3 (the C twin agrees in check_runtime.sh) --

def test_fftw_link_flag_rule():
    assert library_for_callee("fftwf_execute") == "-lfftw3"          # the B2 rule (the task's assertion)
    assert library_for_callee("fftwf_plan_dft_1d") == "-lfftw3"      # any fftwf_* (single-prec)
    assert library_for_callee("fftw_execute") == "-lfftw3"           # the double-prec fftw_* prefix too
    # no regression on the existing classifications.
    assert library_for_callee("cblas_sgemm") == "-lcblas"           # B5 BLAS still maps
    assert library_for_callee("sqrtf") == "-lm"                     # libm still maps
    assert library_for_callee("free") == NO_FLAG                    # libc-implicit, known (not unknown)
    assert library_for_callee("totally_unknown_fn") is None         # unknown-callee policy unchanged
