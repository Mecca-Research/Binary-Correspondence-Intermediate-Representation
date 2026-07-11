"""Area-B breadth (#62) -- wrap a trusted GSL (GNU Scientific Library) statistic (gsl_stats_mean /
gsl_stats_variance / gsl_stats_sd) through the c.call.libm: edge, with the A1.1 Q8 bridge at the boundary
(integrate, don't reinvent -- a FOURTH, distinct library after B5's BLAS sgemm, B2's FFTW, and #61's LAPACK
sgesv: GSL's special-functions / statistics domain, neither linear algebra nor an FFT). BCIR owns the
calling side (the dense unit-stride layout + quantization); the statistic is delegated to GSL when linked,
with a portable textbook reference fallback.

Why statistics (the reference-verifiability choice): GSL statistics has a TRIVIALLY EXACT, independent
reference -- mean = sum/n, variance = sum((x-mean)^2)/(n-1) (GSL's unbiased n-1 divisor), sd = sqrt(var).
The independent check here recomputes those formulas with no BCIR code (and no numpy), so the oracle is
genuinely reference-verified; only the sd's sqrt is transcendental (it rides c.call.libm:sqrtf).

Covers: the oracle stats_reference matches an independent recompute (exact for mean/variance, float
round-off for the sd sqrt) incl. a known closed-form case; the bridged statistic tracks the reference
within the quantization error (R17-certified input bound); the emitted C wraps gsl_stats_* with a portable
reference fallback; the fallback path compiles + runs + is correct (no GSL needed -- it IS the same formula,
so linked-vs-fallback agree to float round-off); only if a GSL lib is present, the linked path agrees too;
and the GSL link-flag rule resolves gsl_* -> -lgsl (with the C twin agreeing in check_runtime.sh). The
point: the win is on the calling side, not a reimplemented statistic."""

import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.gsl_kernels import GSL_STATS, stats_reference, stats_via_bridge
from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
from bcir.lower.c_kernel import emit_gsl_stats_c
from bcir.model import Claim, Domain, Lane, Opcode, StrideClass
from bcir.toolchain import host_link_args


def _gsl_link():
    """A GSL lib flag set that links (with the gsl headers present), or None (the real-GSL path self-skips)."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.c")
        open(src, "w").write("#include <gsl/gsl_statistics.h>\n"
                             "int main(void){double x[2]={1,2}; return (int)gsl_stats_mean(x,1,2)*0;}\n")
        for lib in (["-lgsl", "-lgslcblas"], ["-lgsl"]):
            if subprocess.run(host_link_args([cc, src, *lib, "-lm", "-o", os.path.join(d, "p")]),
                              capture_output=True).returncode == 0:
                return lib
    return None


def _independent_stats(data, kind):
    """An INDEPENDENT recompute of the GSL statistic (no BCIR code, no numpy): mean = sum/n, variance =
    sum((x-mean)^2)/(n-1) (the unbiased n-1 divisor GSL uses), sd = sqrt(variance). stats_reference must
    agree with this."""
    n = len(data)
    mean = sum(data) / n
    if kind == "mean":
        return mean
    var = sum((v - mean) ** 2 for v in data) / (n - 1)
    return var if kind == "variance" else math.sqrt(var)


# --- the oracle reference: matches an independent recompute (+ a known closed-form) --------------------

def test_stats_reference_matches_an_independent_recompute():
    rng = random.Random(0x62)
    data = [rng.uniform(-5, 5) for _ in range(11)]
    for kind in GSL_STATS:
        got = stats_reference(data, kind)
        want = _independent_stats(data, kind)             # independent of BCIR's code
        assert abs(got - want) <= 1e-9, (kind, got, want)


def test_stats_reference_known_closed_form():
    # data = 1..10: mean = 5.5; unbiased (n-1) variance of 1..10 is exactly 110/12 = 9.1666...; sd its sqrt.
    data = [float(i) for i in range(1, 11)]
    assert abs(stats_reference(data, "mean") - 5.5) <= 1e-12
    assert abs(stats_reference(data, "variance") - (110.0 / 12.0)) <= 1e-12
    assert abs(stats_reference(data, "sd") - math.sqrt(110.0 / 12.0)) <= 1e-12


def test_variance_uses_the_unbiased_n_minus_1_divisor():
    # GSL's gsl_stats_variance is the UNBIASED (sample) variance: it divides by n-1, NOT n. Pin that: for
    # [0, 2] the n-1 variance is ((0-1)^2+(2-1)^2)/1 = 2.0 (the n-divisor "population" form would give 1.0).
    assert abs(stats_reference([0.0, 2.0], "variance") - 2.0) <= 1e-12
    assert abs(stats_reference([0.0, 2.0], "sd") - math.sqrt(2.0)) <= 1e-12


def test_stats_reference_rejects_malformed_inputs():
    try:
        stats_reference([], "mean"); assert False, "empty mean should raise"
    except ValueError:
        pass
    for bad_kind in ("variance", "sd"):
        try:
            stats_reference([1.0], bad_kind); assert False, f"n<2 {bad_kind} should raise"
        except ValueError:
            pass
    try:
        stats_reference([1.0, 2.0], "median"); assert False, "unknown kind should raise"
    except ValueError:
        pass


def test_stats_reference_is_side_effect_free():
    data = [3.0, 1.0, 4.0, 1.0, 5.0]
    d0 = list(data)
    for kind in GSL_STATS:
        stats_reference(data, kind)
    assert data == d0


# --- the oracle bridge: tracks the reference within the quantization error -----------------------------

def test_bridged_stats_tracks_the_reference_within_quant_error():
    # The mean is a 1-Lipschitz average, so a per-input round-trip error e perturbs it by at most e; the
    # variance/sd are bounded by a generous multiple of the input round-trip (the certified boundary -- the
    # statistic itself is trusted/exact on the round-tripped inputs).
    rng = random.Random(0x62F7)
    data = [rng.uniform(-4, 4) for _ in range(16)]
    from bcir.kbcir.quantize import max_abs_error
    q_err = max_abs_error(data, group_size=8, bits=8)     # the R17 input round-trip bound (real units)
    spread = max(abs(v) for v in data) + 1.0
    for kind in GSL_STATS:
        ref = stats_reference(data, kind)
        got = stats_via_bridge(data, kind, group_size=8, bits=8)
        # mean: |delta| <= q_err. variance/sd: bound by a generous spread-scaled multiple of q_err.
        amp = 1.0 + 1e-6 if kind == "mean" else 8.0 * spread
        assert abs(ref - got) <= amp * q_err + 1e-6, (kind, ref, got, q_err)


def test_bridge_is_a_clean_roundtrip_then_trusted_stat():
    # stats_via_bridge == reference statistic of the dequantized (round-tripped) inputs -- i.e. the trusted
    # statistic sees exactly the bridged values, so the bridge is the only error source (mirrors B5/B2/#61).
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(173)
    data = [rng.uniform(-3, 3) for _ in range(12)]
    dd = dequantize(quantize_per_group(data, 8, 8))
    for kind in GSL_STATS:
        assert stats_via_bridge(data, kind, 8, 8) == stats_reference(dd, kind)


# --- the emitted wrapper: gsl_stats_* + portable reference fallback, BCIR owns the layout --------------

def test_emit_wraps_gsl_with_a_reference_fallback():
    c = emit_gsl_stats_c("sd", 8, "stat")
    assert "gsl_stats_sd(data, 1, 8)" in c                                  # the trusted external kernel
    assert "BCIR_USE_GSL" in c and "BCIR_STATS_GSL" in c                    # link-selected
    assert "#include <gsl/gsl_statistics.h>" in c                          # the GSL header on the linked path
    assert "float d = data[i] - mean; ss += d * d;" in c                   # the unbiased-variance fallback
    assert "ss / (float)(8 - 1)" in c                                      # the n-1 divisor
    assert "sqrtf(var)" in c                                               # sd's transcendental on the libm edge
    # mean has no transcendental (no sqrtf); variance has the squared-deviation fallback but no sqrt.
    assert "sqrtf" not in emit_gsl_stats_c("mean", 4, "stat")
    assert "sqrtf" not in emit_gsl_stats_c("variance", 4, "stat")
    assert "gsl_stats_mean(data, 1, 4)" in emit_gsl_stats_c("mean", 4, "stat")
    for bad in [("mean", 0), ("variance", 1), ("sd", 1), ("median", 4)]:
        try:
            emit_gsl_stats_c(*bad); assert False, f"{bad} should raise"
        except ValueError:
            pass


def _run_stats_kernel(cc, kernel, data, define=None, libs=None):
    """Compile the emitted statistic kernel + a main that calls it on `data` and prints the float result;
    return the parsed value."""
    n = len(data)
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float data[{n}] = {{{', '.join(f'{v:.8f}f' for v in data)}}};\n"
            f"  printf(\"%.8f\\n\", (double)stat(data));\n  return 0;\n}}\n")
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
        cmd += ["-lm", "-o", exe]                          # -lm: sqrtf in the sd fallback path
        bld = subprocess.run(host_link_args(cmd), capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return float(out.stdout.strip())


def test_fallback_path_compiles_runs_and_is_correct():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return                                              # quick tier hides the toolchain -> self-skip
    data = [2.5, -1.0, 3.0, 4.5, 0.0, -2.0, 1.0, 6.0, 3.5, -0.5]
    for kind in GSL_STATS:
        ref = stats_reference(data, kind)
        want = _independent_stats(data, kind)              # the genuine independent check
        got = _run_stats_kernel(cc, emit_gsl_stats_c(kind, len(data), "stat"), data)  # NO GSL lib
        # the C fallback IS the same formula (float32 vs the oracle's float64) -> agree to float round-off.
        assert abs(got - ref) < 1e-3, (kind, got, ref)
        assert abs(got - want) < 1e-3, (kind, got, want)


def test_linked_gsl_path_agrees_when_gsl_is_present():
    libs = _gsl_link()
    if not libs:
        return                                              # no GSL here -> the real-link path self-skips
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    data = [2.5, -1.0, 3.0, 4.5, 0.0, -2.0, 1.0, 6.0, 3.5, -0.5]
    for kind in GSL_STATS:
        ref = stats_reference(data, kind)
        got = _run_stats_kernel(cc, emit_gsl_stats_c(kind, len(data), "stat"), data,
                                define="-DBCIR_USE_GSL", libs=libs)
        # GSL computes the identical statistic; agree to float round-off.
        assert abs(got - ref) < 1e-2, (kind, got, ref)


# --- R17 boundary: the bridge step is certified on the wrapped call ------------------------------------

def test_r17_certifies_the_bridge_on_a_quantized_stats_call():
    # a wrapped statistic that consumes quantized lanes carries the bridge's round-trip step in its bound
    # (the trusted GSL kernel itself is exact/trusted); a dense call does not. Mirrors test_lapack/fftw/blas.
    def call(qbits):
        return Claim(id=6200, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT, count=1,
                     rd=(60, 61), wr=(62,), op="call.stats", domain=Domain.RAM, quantized_bits=qbits)
    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1     # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1                              # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1                          # the R17 grid bound the bridge is held to


# --- the GSL link-flag rule: a GSL edge resolves to -lgsl (the C twin agrees in check_runtime.sh) ------

def test_gsl_link_flag_rule():
    assert library_for_callee("gsl_stats_mean") == "-lgsl"          # the rule (the wrapper's actual callee)
    assert library_for_callee("gsl_stats_variance") == "-lgsl"      # any gsl_*
    assert library_for_callee("gsl_stats_sd") == "-lgsl"
    assert library_for_callee("gsl_sf_erf") == "-lgsl"              # a special-function gsl_* too
    # no regression on the existing classifications.
    assert library_for_callee("LAPACKE_sgesv") == "-llapack"        # #61 LAPACK still maps
    assert library_for_callee("fftwf_execute") == "-lfftw3"         # B2 FFTW still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"           # B5 BLAS still maps
    assert library_for_callee("sqrtf") == "-lm"                     # libm still maps
    assert library_for_callee("free") == NO_FLAG                    # libc-implicit, known (not unknown)
    assert library_for_callee("totally_unknown_fn") is None         # unknown-callee policy unchanged
