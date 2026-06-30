"""Area-B breadth (SEG2) -- wrap a trusted libcerf (the complex/scaled error-function library)
NUMERICALLY-ROBUST SCALED COMPLEMENTARY ERROR FUNCTION (erfcxf: erfcx(x) = e^{x^2}*erfc(x)) through the
c.call.libm: edge, with the A1.1 Q8 bridge at the boundary (integrate, don't reinvent -- the SIXTH Area-B
library after B5's BLAS sgemm, B2's FFTW, #61's LAPACK sgesv, #62's GSL, and #63's SLEEF: libcerf's
distinctive domain is numerically-robust special functions, the error-function family computed without
overflow across the full real line, neither linear algebra, an FFT, statistics, nor a fast vectorized
libm). BCIR owns the calling side (the dense unit-stride layout + quantization); the special function is
delegated to libcerf when linked, with a portable naive expf(x*x)*erfcf(x)-per-element reference fallback.

Why erfcx (the capability + reference-verifiability choice): erfcx is a function libm does NOT provide --
the naive expf(x*x)*erfcf(x) OVERFLOWS for moderately large x (exp(x^2) past float range around |x|>~9),
yet erfcx itself stays finite across the whole real line, so libcerf ADDS A CAPABILITY (unlike SLEEF's
exp, which duplicates libm expf). The independent check here recomputes erfcx per element with the stdlib
(math.exp(v*v)*math.erfc(v), no BCIR code, no numpy), so the oracle is genuinely reference-verified to
float round-off WITHIN the tested range; the special function itself is trusted (off the deterministic
rail, like the activation expf). KEEP ALL TEST DATA in |x| <= 3 so exp(x*x) <= exp(9) ~ 8103 never
overflows float and the naive reference stays accurate (the library's full-range robustness is its
advantage OUTSIDE this envelope, exactly as SLEEF's vectorization speed is outside its correctness check).

Covers: the oracle erfcx_reference matches an independent per-element stdlib recompute (to float round-off)
incl. the closed-form anchor erfcx(0)=1; the bridged erfcx tracks the reference within the quantization
error (R17-certified input bound, Lipschitz-amplified); the emitted C wraps erfcxf with a portable naive
fallback; the fallback path compiles + runs + is correct (no libcerf needed -- it IS the same erfcx within
|x|<=3, so linked-vs-fallback agree to float round-off vs the independent reference); only if a libcerf lib
is present, the linked path agrees too; and the libcerf link-flag rule resolves erfcx*->-lcerf (with the C
twin agreeing in check_runtime.sh, NO regression on the other six cases). The point: the win is on the
calling side + delegation to a trusted library, not a reimplemented special function."""

import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.cerf_kernels import erfcx_reference, erfcx_via_bridge
from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
from bcir.lower.c_kernel import emit_cerf_erfcx_c
from bcir.model import Claim, Domain, Lane, Opcode, StrideClass


def _cerf_link():
    """A libcerf lib flag set that links (with the cerf header present), or None (the real-libcerf path
    self-skips). libcerf is rarely installed on CI, so this almost always returns None -- the fallback path
    is the one that does the real reference-verification work, exactly as in test_sleef/test_gsl/test_fftw."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.c")
        open(src, "w").write("#include <cerf.h>\n"
                             "int main(void){return (int)erfcxf(0.0f)*0;}\n")
        for lib in (["-lcerf"],):
            if subprocess.run([cc, src, *lib, "-lm", "-o", os.path.join(d, "p")],
                              capture_output=True).returncode == 0:
                return lib
    return None


def _independent_erfcx(data):
    """An INDEPENDENT per-element recompute (no BCIR code, no numpy): out[i] = exp(v*v)*erfc(v) via the
    stdlib (== erfcx(v) within |x|<=3, where exp(v*v) never overflows). erfcx_reference must agree."""
    return [math.exp(v * v) * math.erfc(v) for v in data]


# --- the oracle reference: matches an independent recompute (+ a known closed-form) --------------------

def test_erfcx_reference_matches_an_independent_recompute():
    rng = random.Random(0x64)
    data = [rng.uniform(-3, 3) for _ in range(13)]          # |x| <= 3: exp(x*x) never overflows
    got = erfcx_reference(data)
    want = _independent_erfcx(data)                         # independent of BCIR's code
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert abs(g - w) <= 1e-9 * (1.0 + abs(w)), (g, w)


def test_erfcx_reference_known_closed_form():
    # erfcx(0) = exp(0)*erfc(0) = 1*1 = 1 exactly -- a clean closed-form anchor, independent of BCIR.
    data = [0.0]
    got = erfcx_reference(data)
    want = [1.0]
    for g, w in zip(got, want):
        assert g == w, (g, w)
    assert erfcx_reference([0.0])[0] == 1.0                 # the anchor, stated exactly


def test_erfcx_reference_rejects_empty():
    try:
        erfcx_reference([]); assert False, "empty erfcx should raise"
    except ValueError:
        pass


def test_erfcx_reference_is_side_effect_free():
    data = [3.0, 1.0, -3.0, 1.5, 0.0]
    d0 = list(data)
    erfcx_reference(data)
    assert data == d0


# --- the oracle bridge: tracks the reference within the quantization error -----------------------------

def test_bridged_erfcx_tracks_the_reference_within_quant_error():
    # erfcx amplifies a per-input round-trip error e by ~|d/dx erfcx(x)| = |2*x*erfcx(x) - 2/sqrt(pi)| (its
    # own derivative), so the bound here is the INPUT round-trip (the certified boundary -- the special
    # function itself is trusted/exact on the round-tripped inputs) scaled by the max local Lipschitz
    # constant over the data. Mirrors test_sleef's spread-scaled bound. All data in |x| <= 3.
    rng = random.Random(0x64F7)
    data = [rng.uniform(-3, 3) for _ in range(16)]
    from bcir.kbcir.quantize import max_abs_error
    q_err = max_abs_error(data, group_size=8, bits=8)      # the R17 input round-trip bound (real units)
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
    amp = max(abs(2.0 * v * (math.exp(v * v) * math.erfc(v)) - two_over_sqrt_pi)
              for v in data) + 1.0                         # erfcx' = 2x*erfcx(x) - 2/sqrt(pi)
    ref = erfcx_reference(data)
    got = erfcx_via_bridge(data, group_size=8, bits=8)
    for r, g in zip(ref, got):
        assert abs(r - g) <= amp * q_err + 1e-6, (r, g, q_err)


def test_bridge_is_a_clean_roundtrip_then_trusted_special_function():
    # erfcx_via_bridge == reference erfcx of the dequantized (round-tripped) inputs -- i.e. the trusted
    # special function sees exactly the bridged values, so the bridge is the only error source (mirrors
    # B5/B2/#61/#62/#63). The special function adds no certified error (it is trusted, like the activation
    # expf).
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(641)
    data = [rng.uniform(-2, 2) for _ in range(12)]
    dd = dequantize(quantize_per_group(data, 8, 8))
    assert erfcx_via_bridge(data, 8, 8) == erfcx_reference(dd)


# --- the emitted wrapper: erfcxf + portable naive expf*erfcf fallback, BCIR owns the layout ------------

def test_emit_wraps_cerf_with_a_libm_fallback():
    c = emit_cerf_erfcx_c(8, "verfcx")
    assert "erfcxf(data[i])" in c                                            # the trusted external kernel
    assert "BCIR_USE_CERF" in c and "BCIR_ERFCX_CERF" in c                  # link-selected
    assert "#include <cerf.h>" in c                                         # the libcerf header on the linked path
    assert "expf(data[i]*data[i])" in c                                     # the portable naive twin (exp(x^2))
    assert "erfcf(data[i])" in c                                            # ... times erfc(x)
    assert "#include <math.h>" in c                                        # libm on the fallback path
    assert "c.call.libm:erfcxf" in c                                       # the FFI edge label (the libcerf callee)
    assert "c.call.libm:expf" in c                                         # the fallback's libm edge label (exp)
    assert "c.call.libm:erfcf" in c                                        # the fallback's libm edge label (erfc)
    # the signature is the dense elementwise (data, out) map.
    assert "void verfcx(const float *data, float *out)" in c
    try:
        emit_cerf_erfcx_c(0, "verfcx"); assert False, "n=0 should raise"
    except ValueError:
        pass


def _run_erfcx_kernel(cc, kernel, data, define=None, libs=None):
    """Compile the emitted erfcx kernel + a main that calls it on `data` and prints each result; return the
    parsed float list."""
    n = len(data)
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float data[{n}] = {{{', '.join(f'{v:.8f}f' for v in data)}}};\n"
            f"  float out[{n}];\n  verfcx(data, out);\n"
            f"  for (int i = 0; i < {n}; ++i) printf(\"%.8f\\n\", (double)out[i]);\n"
            f"  return 0;\n}}\n")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "s")
        cmd = [cc, "-std=c11", "-O2"]
        if define:
            cmd.append(define)
        cmd += [src]
        if libs:
            cmd += libs
        cmd += ["-lm", "-o", exe]                          # -lm: expf/erfcf in the fallback path
        bld = subprocess.run(cmd, capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return [float(v) for v in out.stdout.split()]


def test_fallback_path_compiles_runs_and_is_correct():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return                                              # quick tier hides the toolchain -> self-skip
    data = [2.5, -1.0, 3.0, 0.5, 0.0, -2.0, 1.0, -0.25, 1.75, -3.0]   # |x| <= 3: naive form is safe
    ref = erfcx_reference(data)
    want = _independent_erfcx(data)                         # the genuine independent check (stdlib)
    got = _run_erfcx_kernel(cc, emit_cerf_erfcx_c(len(data), "verfcx"), data)   # NO libcerf lib
    assert len(got) == len(data)
    for g, r, w in zip(got, ref, want):
        # the C fallback IS the same erfcx (float32 vs the oracle's float64) -> agree to float round-off.
        assert abs(g - r) < 1e-3 * (1.0 + abs(r)), (g, r)
        assert abs(g - w) < 1e-3 * (1.0 + abs(w)), (g, w)


def test_linked_cerf_path_agrees_when_cerf_is_present():
    libs = _cerf_link()
    if not libs:
        return                                              # no libcerf here -> the real-link path self-skips
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    data = [2.5, -1.0, 3.0, 0.5, 0.0, -2.0, 1.0, -0.25, 1.75, -3.0]
    ref = erfcx_reference(data)
    got = _run_erfcx_kernel(cc, emit_cerf_erfcx_c(len(data), "verfcx"), data,
                            define="-DBCIR_USE_CERF", libs=libs)
    for g, r in zip(got, ref):
        # libcerf computes the identical erfcx (and within |x|<=3 the naive ref is accurate too); agree.
        assert abs(g - r) < 1e-2 * (1.0 + abs(r)), (g, r)


# --- R17 boundary: the bridge step is certified on the wrapped call ------------------------------------

def test_r17_certifies_the_bridge_on_a_quantized_erfcx_call():
    # a wrapped special function that consumes quantized lanes carries the bridge's round-trip step in its
    # bound (the trusted libcerf kernel itself is exact/trusted); a dense call does not. Mirrors
    # test_sleef/test_gsl/test_lapack/test_fftw/test_blas.
    def call(qbits):
        return Claim(id=6400, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT, count=1,
                     rd=(64,), wr=(65,), op="call.erfcx", domain=Domain.RAM, quantized_bits=qbits)
    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1     # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1                              # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1                          # the R17 grid bound the bridge is held to


# --- the libcerf link-flag rule: an erfcx edge resolves to -lcerf (the C twin agrees in check_runtime.sh) -

def test_cerf_link_flag_rule():
    assert library_for_callee("erfcxf") == "-lcerf"                 # the rule (the wrapper's actual callee)
    assert library_for_callee("erfcx") == "-lcerf"                  # the bare/double form too
    # no regression on the existing classifications (all six other library cases + libm + libc + unknown).
    assert library_for_callee("Sleef_expf1_u10") == "-lsleef"      # #63 SLEEF still maps
    assert library_for_callee("gsl_stats_mean") == "-lgsl"         # #62 GSL still maps
    assert library_for_callee("LAPACKE_sgesv") == "-llapack"       # #61 LAPACK still maps
    assert library_for_callee("fftwf_execute") == "-lfftw3"        # B2 FFTW still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"          # B5 BLAS still maps
    assert library_for_callee("expf") == "-lm"                     # libm still maps (the erfcx fallback's twin)
    assert library_for_callee("free") == NO_FLAG                   # libc-implicit, known (not unknown)
    assert library_for_callee("totally_unknown_fn") is None        # unknown-callee policy unchanged
