"""Area-B breadth (#63) -- wrap a trusted SLEEF (SIMD-oriented vectorized libm) VECTORIZED ELEMENTWISE EXP
(Sleef_expf1_u10) through the c.call.libm: edge, with the A1.1 Q8 bridge at the boundary (integrate, don't
reinvent -- the FIFTH and final Area-B library after B5's BLAS sgemm, B2's FFTW, #61's LAPACK sgesv, and
#62's GSL: SLEEF's distinctive domain is vectorized transcendental math, a fast vectorized libm, neither
linear algebra, an FFT, nor statistics). BCIR owns the calling side (the dense unit-stride layout +
quantization); the transcendental is delegated to SLEEF when linked, with a portable libm-expf-per-element
reference fallback.

Why vectorized exp (the reference-verifiability + domain choice): exp is SLEEF's canonical vectorized
transcendental, and the SAME elementwise expf the G1 activation kernels already route through the libm edge
(SLEEF is its vectorized alternative). The independent check here recomputes exp per element with the
stdlib (math.exp, no BCIR code, no numpy), so the oracle is genuinely reference-verified to float round-off;
the transcendental itself is trusted (off the deterministic rail, like the activation expf).

Covers: the oracle exp_reference matches an independent per-element math.exp recompute (to float round-off)
incl. a known closed-form case; the bridged exp tracks the reference within the quantization error
(R17-certified input bound, exp-amplified); the emitted C wraps Sleef_expf1_u10 with a portable expf
fallback; the fallback path compiles + runs + is correct (no SLEEF needed -- it IS the same exp, so
linked-vs-fallback agree to float round-off vs the independent reference); only if a SLEEF lib is present,
the linked path agrees too; and the SLEEF link-flag rule resolves Sleef_* -> -lsleef (with the C twin
agreeing in check_runtime.sh, NO regression on the other five cases). The point: the win is on the calling
side, not a reimplemented transcendental."""

import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
from bcir.toolchain import host_link_args
from bcir.kbcir.vecmath import exp_reference, exp_via_bridge
from bcir.lower.c_kernel import emit_sleef_exp_c
from bcir.model import Claim, Domain, Lane, Opcode, StrideClass


def _sleef_link():
    """A SLEEF lib flag set that links (with the sleef header present), or None (the real-SLEEF path
    self-skips). SLEEF is rarely installed on CI, so this almost always returns None -- the fallback path
    is the one that does the real reference-verification work, exactly as in test_gsl/test_fftw."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.c")
        open(src, "w").write(
            "#include <sleef.h>\nint main(void){return (int)Sleef_expf1_u10(0.0f)*0;}\n"
        )
        for lib in (["-lsleef"],):
            if (
                subprocess.run(
                    host_link_args([cc, src, *lib, "-lm", "-o", os.path.join(d, "p")]),
                    capture_output=True,
                ).returncode
                == 0
            ):
                return lib
    return None


def _independent_exp(data):
    """An INDEPENDENT per-element recompute (no BCIR code, no numpy): out[i] = exp(data[i]) via the stdlib.
    exp_reference must agree with this."""
    return [math.exp(v) for v in data]


# --- the oracle reference: matches an independent recompute (+ a known closed-form) --------------------


def test_exp_reference_matches_an_independent_recompute():
    rng = random.Random(0x63)
    data = [rng.uniform(-4, 4) for _ in range(13)]
    got = exp_reference(data)
    want = _independent_exp(data)  # independent of BCIR's code
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert abs(g - w) <= 1e-9 * (1.0 + abs(w)), (g, w)


def test_exp_reference_known_closed_form():
    # exp(0)=1, exp(1)=e, exp(ln 2)=2, exp(-1)=1/e -- closed-form anchors, independent of BCIR.
    data = [0.0, 1.0, math.log(2.0), -1.0]
    got = exp_reference(data)
    want = [1.0, math.e, 2.0, 1.0 / math.e]
    for g, w in zip(got, want):
        assert abs(g - w) <= 1e-12, (g, w)


def test_exp_reference_rejects_empty():
    try:
        exp_reference([])
        assert False, "empty exp should raise"
    except ValueError:
        pass


def test_exp_reference_is_side_effect_free():
    data = [3.0, 1.0, -4.0, 1.5, 0.0]
    d0 = list(data)
    exp_reference(data)
    assert data == d0


# --- the oracle bridge: tracks the reference within the quantization error -----------------------------


def test_bridged_exp_tracks_the_reference_within_quant_error():
    # exp amplifies a per-input round-trip error e by ~exp(x) (its own derivative), so the bound here is the
    # INPUT round-trip (the certified boundary -- the transcendental itself is trusted/exact on the
    # round-tripped inputs) scaled by the max exp(x) over the data. Mirrors test_gsl's spread-scaled bound.
    rng = random.Random(0x63F7)
    data = [rng.uniform(-3, 3) for _ in range(16)]
    from bcir.kbcir.quantize import max_abs_error

    q_err = max_abs_error(data, group_size=8, bits=8)  # the R17 input round-trip bound (real units)
    amp = max(math.exp(v) for v in data) + 1.0  # exp is its own derivative -> exp(x)-scaled
    ref = exp_reference(data)
    got = exp_via_bridge(data, group_size=8, bits=8)
    for r, g in zip(ref, got):
        assert abs(r - g) <= amp * q_err + 1e-6, (r, g, q_err)


def test_bridge_is_a_clean_roundtrip_then_trusted_transcendental():
    # exp_via_bridge == reference exp of the dequantized (round-tripped) inputs -- i.e. the trusted
    # transcendental sees exactly the bridged values, so the bridge is the only error source (mirrors
    # B5/B2/#61/#62). The transcendental adds no certified error (it is trusted, like the activation expf).
    from bcir.kbcir.quantize import dequantize, quantize_per_group

    rng = random.Random(631)
    data = [rng.uniform(-2, 2) for _ in range(12)]
    dd = dequantize(quantize_per_group(data, 8, 8))
    assert exp_via_bridge(data, 8, 8) == exp_reference(dd)


# --- the emitted wrapper: Sleef_expf1_u10 + portable expf fallback, BCIR owns the layout ---------------


def test_emit_wraps_sleef_with_a_libm_fallback():
    c = emit_sleef_exp_c(8, "vexp")
    assert "Sleef_expf1_u10(data[i])" in c  # the trusted external kernel
    assert "BCIR_USE_SLEEF" in c and "BCIR_VEXP_SLEEF" in c  # link-selected
    assert "#include <sleef.h>" in c  # the SLEEF header on the linked path
    assert "expf(data[i])" in c  # the portable reference twin
    assert "#include <math.h>" in c  # libm on the fallback path
    assert "c.call.libm:Sleef_expf1_u10" in c  # the FFI edge label (the SLEEF callee)
    assert "c.call.libm:expf" in c  # the fallback's libm edge label
    # the signature is the dense elementwise (data, out) map.
    assert "void vexp(const float *data, float *out)" in c
    try:
        emit_sleef_exp_c(0, "vexp")
        assert False, "n=0 should raise"
    except ValueError:
        pass


def _run_exp_kernel(cc, kernel, data, define=None, libs=None):
    """Compile the emitted exp kernel + a main that calls it on `data` and prints each result; return the
    parsed float list."""
    n = len(data)
    main = (
        f"\n#include <stdio.h>\nint main(void){{\n"
        f"  float data[{n}] = {{{', '.join(f'{v:.8f}f' for v in data)}}};\n"
        f"  float out[{n}];\n  vexp(data, out);\n"
        f'  for (int i = 0; i < {n}; ++i) printf("%.8f\\n", (double)out[i]);\n'
        f"  return 0;\n}}\n"
    )
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
        cmd += ["-lm", "-o", exe]  # -lm: expf in the fallback path
        bld = subprocess.run(host_link_args(cmd), capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return [float(v) for v in out.stdout.split()]


def test_fallback_path_compiles_runs_and_is_correct():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    data = [2.5, -1.0, 3.0, 0.5, 0.0, -2.0, 1.0, -0.25, 1.75, -3.0]
    ref = exp_reference(data)
    want = _independent_exp(data)  # the genuine independent check (stdlib math.exp)
    got = _run_exp_kernel(cc, emit_sleef_exp_c(len(data), "vexp"), data)  # NO SLEEF lib
    assert len(got) == len(data)
    for g, r, w in zip(got, ref, want):
        # the C fallback IS the same exp (float32 vs the oracle's float64) -> agree to float round-off.
        assert abs(g - r) < 1e-3 * (1.0 + abs(r)), (g, r)
        assert abs(g - w) < 1e-3 * (1.0 + abs(w)), (g, w)


def test_linked_sleef_path_agrees_when_sleef_is_present():
    libs = _sleef_link()
    if not libs:
        return  # no SLEEF here -> the real-link path self-skips
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    data = [2.5, -1.0, 3.0, 0.5, 0.0, -2.0, 1.0, -0.25, 1.75, -3.0]
    ref = exp_reference(data)
    got = _run_exp_kernel(
        cc, emit_sleef_exp_c(len(data), "vexp"), data, define="-DBCIR_USE_SLEEF", libs=libs
    )
    for g, r in zip(got, ref):
        # SLEEF computes the identical exp (1.0 ULP); agree to float round-off.
        assert abs(g - r) < 1e-2 * (1.0 + abs(r)), (g, r)


# --- R17 boundary: the bridge step is certified on the wrapped call ------------------------------------


def test_r17_certifies_the_bridge_on_a_quantized_exp_call():
    # a wrapped transcendental that consumes quantized lanes carries the bridge's round-trip step in its
    # bound (the trusted SLEEF kernel itself is exact/trusted); a dense call does not. Mirrors
    # test_gsl/test_lapack/test_fftw/test_blas.
    def call(qbits):
        return Claim(
            id=6300,
            opcode=Opcode.MUL,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1,
            rd=(63,),
            wr=(64,),
            op="call.vexp",
            domain=Domain.RAM,
            quantized_bits=qbits,
        )

    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1  # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1  # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1  # the R17 grid bound the bridge is held to


# --- the SLEEF link-flag rule: a SLEEF edge resolves to -lsleef (the C twin agrees in check_runtime.sh) -


def test_sleef_link_flag_rule():
    assert (
        library_for_callee("Sleef_expf1_u10") == "-lsleef"
    )  # the rule (the wrapper's actual callee)
    assert library_for_callee("Sleef_sinf1_u10") == "-lsleef"  # any Sleef_*
    assert library_for_callee("Sleef_tanhf1_u10") == "-lsleef"  # another vectorized transcendental
    # no regression on the existing classifications (all five other library cases + libm + libc + unknown).
    assert library_for_callee("gsl_stats_mean") == "-lgsl"  # #62 GSL still maps
    assert library_for_callee("LAPACKE_sgesv") == "-llapack"  # #61 LAPACK still maps
    assert library_for_callee("fftwf_execute") == "-lfftw3"  # B2 FFTW still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"  # B5 BLAS still maps
    assert library_for_callee("expf") == "-lm"  # libm still maps (the SLEEF fallback's twin)
    assert library_for_callee("free") == NO_FLAG  # libc-implicit, known (not unknown)
    assert library_for_callee("totally_unknown_fn") is None  # unknown-callee policy unchanged
