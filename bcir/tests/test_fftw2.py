"""SEG2.2 -- wrap a trusted FFTW 2-D complex FFT through the c.call.libm: edge, with the A1.1 Q8 bridge at
the boundary (integrate, don't reinvent -- the 2-D analog of the B2 1-D FFT wrap, a genuinely NEW numerical
capability: a 2-D spectral transform, the kernel under image/convolution spectral methods, not a 1-D
transform). It rides the SAME FFTW library and the SAME `-lfftw3` link rule as the 1-D wrap (`fftwf_*`
already maps there), so NO registry/linkflags change is needed -- only the plan entry point
(`fftwf_plan_dft_2d`) differs. BCIR owns the calling side (the interleaved complex, ROW-MAJOR layout +
quantization); the transform is delegated to FFTW when linked, with a portable reference 2-D DFT fallback.

Covers: the oracle dft2_reference matches an independent cmath 2-D DFT; a separable-composition cross-check
(the direct 2-D DFT == the 1-D dft_reference along axis-1 of each row then along axis-0 of each column -- a
strong independent correctness proof); a single-2D-tone anchor (a pure tone at (a,b) DFTs to one bin of
magnitude n0*n1); the bridged 2-D FFT tracks the float DFT reference within the quantization error
(R17-certified input bound, amplified by <= n0*n1 per bin); the emitted C wraps
fftwf_plan_dft_2d/execute/destroy with a naive O((n0*n1)^2) DFT reference fallback; the fallback path
compiles and reproduces the reference under clang (no FFTW needed -- it IS a 2-D DFT, so linked-vs-fallback
agree to float round-off); only if an FFTW lib is present, the linked path agrees too; and the EXISTING
fftwf_* -> -lfftw3 link-flag rule already classifies the 2-D plan symbol (no new rule). The point: the win
is on the calling side, not a reimplemented FFT."""

import cmath
import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.fft import dft_reference                       # for the separable-composition cross-check
from bcir.kbcir.fft2 import dft2_reference, fft2_via_bridge
from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
from bcir.lower.c_kernel import emit_fftw_fft2_c
from bcir.model import Claim, Domain, Lane, Opcode, StrideClass


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
    """A list of complex numbers -> the interleaved [re0, im0, re1, im1, ...] float layout (row-major when
    `signal` is a flattened 2-D array)."""
    out = []
    for z in signal:
        out.append(float(z.real))
        out.append(float(z.imag))
    return out


def _cmath_2d_dft(grid):
    """An INDEPENDENT 2-D DFT using only cmath (no numpy, no BCIR code): `grid` is an n0-by-n1 list-of-lists
    of complex; returns the n0-by-n1 list-of-lists out[(j0,j1)] = sum_{k0,k1} grid[k0][k1] *
    exp(-2*pi*i*(j0*k0/n0 + j1*k1/n1)). The oracle dft2_reference must agree with this."""
    n0 = len(grid)
    n1 = len(grid[0])
    out = [[0j] * n1 for _ in range(n0)]
    for j0 in range(n0):
        for j1 in range(n1):
            acc = 0j
            for k0 in range(n0):
                for k1 in range(n1):
                    acc += grid[k0][k1] * cmath.exp(-2j * math.pi * (j0 * k0 / n0 + j1 * k1 / n1))
            out[j0][j1] = acc
    return out


def _flatten(grid):
    """An n0-by-n1 list-of-lists of complex -> the row-major interleaved float layout."""
    return _interleave([z for row in grid for z in row])


# --- the oracle DFT reference agrees with an independent textbook 2-D DFT --------------------------

def test_dft2_reference_matches_an_independent_2d_dft():
    rng = random.Random(0xB22)
    n0, n1 = 3, 4
    grid = [[complex(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(n1)] for _ in range(n0)]
    ref = dft2_reference(_flatten(grid), n0, n1)
    want = _cmath_2d_dft(grid)                               # independent of BCIR's code
    for j0 in range(n0):
        for j1 in range(n1):
            o = 2 * (j0 * n1 + j1)
            assert abs(ref[o] - want[j0][j1].real) <= 1e-9, (j0, j1)        # real part
            assert abs(ref[o + 1] - want[j0][j1].imag) <= 1e-9, (j0, j1)    # imaginary part


def test_dft2_reference_equals_separable_1d_composition():
    # A strong independent correctness check: a 2-D DFT is SEPARABLE -- it equals applying the 1-D
    # dft_reference along axis-1 (each row) and then along axis-0 (each column). This composes ONLY the 1-D
    # oracle (a different code path from the direct double sum) and must agree to float round-off.
    rng = random.Random(0x5EBA)
    n0, n1 = 4, 5
    grid = [[complex(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(n1)] for _ in range(n0)]
    direct = dft2_reference(_flatten(grid), n0, n1)

    # step 1: 1-D DFT each ROW (length n1) -> rows[r] is the transformed row r as interleaved floats.
    rows = [dft_reference(_interleave(grid[r]), n1) for r in range(n0)]
    # step 2: 1-D DFT each COLUMN (length n0). Column c is the n0 complex values rows[r][c]; rebuild as
    # complex, transform, and scatter the result back into the flat row-major output.
    out = [0.0] * (2 * n0 * n1)
    for c in range(n1):
        col = [complex(rows[r][2 * c], rows[r][2 * c + 1]) for r in range(n0)]
        tcol = dft_reference(_interleave(col), n0)           # length-n0 1-D DFT down the column
        for r in range(n0):
            o = 2 * (r * n1 + c)
            out[o] = tcol[2 * r]
            out[o + 1] = tcol[2 * r + 1]

    for a, b in zip(direct, out):
        assert abs(a - b) <= 1e-9, (a, b)


def test_dft2_reference_of_a_pure_2d_tone_is_a_single_bin():
    # A pure 2-D tone in[(k0,k1)] = exp(+2*pi*i*(a*k0/n0 + b*k1/n1)) at frequency (a,b) DFTs (with FFTW's
    # e^{-...} sign) to a SINGLE bin of magnitude n0*n1 at (a,b), ~0 elsewhere -- a clean closed-form anchor.
    n0, n1, a, b = 4, 4, 1, 2
    grid = [[cmath.exp(2j * math.pi * (a * k0 / n0 + b * k1 / n1)) for k1 in range(n1)] for k0 in range(n0)]
    ref = dft2_reference(_flatten(grid), n0, n1)
    for j0 in range(n0):
        for j1 in range(n1):
            o = 2 * (j0 * n1 + j1)
            mag = math.hypot(ref[o], ref[o + 1])
            if (j0, j1) == (a, b):
                assert abs(mag - n0 * n1) < 1e-6, (j0, j1)    # the tone's single bin, magnitude n0*n1
            else:
                assert mag < 1e-6, (j0, j1)                   # every other bin is empty


def test_dft2_reference_rejects_malformed_inputs():
    for bad in [([], 0, 0), ([1.0, 2.0], 1, 2), ([1.0, 2.0, 3.0, 4.0], 1, 1), ([1.0, 2.0], 0, 1),
                ([1.0, 2.0], 1, 0)]:
        try:
            dft2_reference(*bad); assert False, f"{bad} should raise"
        except ValueError:
            pass


# --- the oracle bridge: tracks the float DFT reference within the quantization error --------------

def test_bridged_fft2_tracks_the_reference_within_quant_error():
    rng = random.Random(0xB22F7)
    n0, n1 = 4, 4
    grid = [[complex(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(n1)] for _ in range(n0)]
    x = _flatten(grid)
    from bcir.kbcir.quantize import max_abs_error
    q_err = max_abs_error(x, group_size=8, bits=8)            # the R17 input round-trip bound (real units)
    ref = dft2_reference(x, n0, n1)
    got = fft2_via_bridge(x, n0, n1, group_size=8, bits=8)
    # A 2-D DFT bin is a sum of n0*n1 unit-magnitude rotations of the inputs, so a per-input error e is
    # amplified by at most n0*n1 in any output bin (|sum| <= sum|.|). Bound the per-bin error by n0*n1*q_err.
    assert all(abs(r - g) <= n0 * n1 * q_err + 1e-6 for r, g in zip(ref, got))


def test_bridge_is_a_clean_roundtrip_then_trusted_fft():
    # fft2_via_bridge == 2-D DFT reference of the dequantized (round-tripped) inputs -- i.e. the trusted FFT
    # sees exactly the bridged values, so the bridge is the only error source (mirrors the 1-D wrap).
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(172)
    n0, n1 = 3, 5
    grid = [[complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(n1)] for _ in range(n0)]
    x = _flatten(grid)
    dx = dequantize(quantize_per_group(x, 8, 8))
    assert fft2_via_bridge(x, n0, n1, 8, 8) == dft2_reference(dx, n0, n1)


# --- the emitted wrapper: fftw 2-D call + portable reference DFT fallback, BCIR owns the layout ----

def test_emit_wraps_fftw2_with_a_reference_fallback():
    c = emit_fftw_fft2_c(4, 6, "fft2")
    assert "fftwf_plan_dft_2d(4, 6, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE)" in c   # n0 outer, n1 inner
    assert "fftwf_execute(plan)" in c and "fftwf_destroy_plan(plan)" in c
    assert "BCIR_USE_FFTW" in c and "BCIR_FFT2_FFTW" in c                           # link-selected
    assert "re += xr * c - xi * s" in c and "im += xr * s + xi * c" in c            # the reference DFT fallback
    assert "void fft2(const float *in, float *out)" in c                           # the signature
    for bad in ((0, 4), (4, 0), (-1, 4), (4, -1)):
        try:
            emit_fftw_fft2_c(*bad); assert False, f"{bad} should raise"
        except ValueError:
            pass


def test_fallback_path_compiles_and_matches_the_reference():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return                                              # quick tier hides the toolchain -> self-skip
    rng = random.Random(0x5EED2)
    n0, n1 = 3, 5
    grid = [[complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(n1)] for _ in range(n0)]
    x = _flatten(grid)
    ref = dft2_reference(x, n0, n1)
    kernel = emit_fftw_fft2_c(n0, n1, "fft2")
    m = 2 * n0 * n1
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float in[{m}] = {{{', '.join(f'{v:.6f}f' for v in x)}}};\n"
            f"  float out[{m}];\n  fft2(in, out);\n"
            f'  for (int i=0;i<{m};++i) printf("%.6f ", out[i]);\n  return 0;\n}}\n')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "f.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "f")
        # the fallback path needs NO FFTW lib (only -lm for cosf/sinf), exactly as the 1-D fallback.
        bld = subprocess.run([cc, "-std=c11", "-O2", src, "-lm", "-o", exe], capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        got = [float(t) for t in out.stdout.split()]
        # the C fallback IS the same 2-D DFT (float32 vs the oracle's float64) -> agree to float round-off.
        assert len(got) == m
        assert all(abs(g - r) < 1e-3 for g, r in zip(got, ref)), (got, ref)


def test_linked_fftw2_path_agrees_when_fftw_is_present():
    lib = _fftw_link()
    if not lib:
        return                                              # no FFTW here -> the real-link path self-skips
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    rng = random.Random(0xF7BA2)
    n0, n1 = 4, 4
    grid = [[complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(n1)] for _ in range(n0)]
    x = _flatten(grid)
    ref = dft2_reference(x, n0, n1)
    kernel = emit_fftw_fft2_c(n0, n1, "fft2")
    m = 2 * n0 * n1
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float in[{m}] = {{{', '.join(f'{v:.6f}f' for v in x)}}};\n"
            f"  float out[{m}];\n  fft2(in, out);\n"
            f'  for (int i=0;i<{m};++i) printf("%.6f ", out[i]);\n  return 0;\n}}\n')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "f.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "f")
        bld = subprocess.run([cc, "-std=c11", "-O2", "-DBCIR_USE_FFTW", src, lib, "-lm", "-o", exe],
                             capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        got = [float(t) for t in out.stdout.split()]
        # FFTW computes the identical forward transform (FFTW_FORWARD == e^{-2pi i(...)}); agree to round-off.
        assert all(abs(g - r) < 1e-2 for g, r in zip(got, ref)), (got, ref)


# --- R17 boundary: the bridge step is certified on the wrapped call -------------------------------

def test_r17_certifies_the_bridge_on_a_quantized_fft2_call():
    # a wrapped 2-D FFT that consumes quantized lanes carries the bridge's round-trip step in its bound (the
    # trusted FFTW kernel itself is exact); a dense call does not. Mirrors the 1-D test_fftw exactly.
    def call(qbits):
        return Claim(id=7220, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT, count=1,
                     rd=(72, 73), wr=(74,), op="call.fft2", domain=Domain.RAM, quantized_bits=qbits)
    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1     # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1                              # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1                          # the R17 grid bound the bridge is held to


# --- the link-flag rule: the EXISTING fftwf_* -> -lfftw3 rule already covers the 2-D plan symbol ---

def test_fftw2_link_flag_rule_is_the_existing_rule():
    # NO new linkflags rule: the 2-D wrap rides the SAME FFTW library as the 1-D wrap, so the existing
    # fftwf_* prefix rule already classifies the 2-D plan symbol (this is the whole point -- no registry
    # change). Confirm it, plus a small no-regression block.
    assert library_for_callee("fftwf_plan_dft_2d") == "-lfftw3"      # the 2-D plan symbol (existing rule)
    assert library_for_callee("fftwf_execute") == "-lfftw3"          # the shared execute edge
    assert library_for_callee("fftwf_plan_dft_1d") == "-lfftw3"      # the 1-D plan symbol still maps
    # no regression on the existing classifications.
    assert library_for_callee("cblas_sgemm") == "-lcblas"           # B5 BLAS still maps
    assert library_for_callee("sqrtf") == "-lm"                     # libm still maps
    assert library_for_callee("free") == NO_FLAG                    # libc-implicit, known (not unknown)
    assert library_for_callee("totally_unknown_fn") is None         # unknown-callee policy unchanged
