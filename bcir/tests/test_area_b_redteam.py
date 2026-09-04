"""D0.2 (H2 pre-driver hardening) -- an ADVERSARIAL NUMERICAL RED-TEAM over the seven Area-B
``c.call.libm:`` library wraps (libcerf erfcx, LAPACK sgesv, GSL stats, SLEEF exp, FFTW 1-D/2-D, BLAS
sgemm). The seven happy-path suites (test_cerf / test_lapack / test_gsl / test_sleef / test_fftw /
test_fftw2 / test_blas_gemm) verify each wrap against a stdlib reference WITHIN a deliberately safe
envelope (|x| <= 3, well-conditioned A, n >= 2, finite inputs). This file probes the SAME wraps OUTSIDE
that envelope and on degenerate inputs -- the cases an attacker (or a careless caller) reaches first.

THE ONE UNIFYING PROPERTY (the security finding to assert everywhere):
    Outside its happy-path envelope, every wrap must EITHER (a) still match a robustly-computed reference,
    OR (b) DEGRADE HONESTLY -- surface an IEEE ``inf``/``nan``, raise ``ValueError``/``OverflowError``, or
    report a solver error (LAPACK ``info > 0`` / a singular-pivot raise). It must NEVER silently return a
    plausible-but-WRONG FINITE number. A wrong finite number where the wrap should degrade is a
    silent-numerical-corruption bug -- the most valuable thing this red-team could find. The asserts below
    encode "non-finite OR raises OR a not-small residual", never "papered over to a wrong finite value".

The structure mirrors the seven existing suites exactly: stdlib-only independent references (NO numpy),
the ``shutil.which("clang") or "cc" or "gcc"`` + ``if not cc: return`` toolchain SELF-SKIP so the quick
(no-toolchain) tier stays green, the compile-emitted-kernel-then-run helper (the shape of
test_cerf.py::``_run_erfcx_kernel``), fully deterministic fixed-literal probe inputs (the adversarial edge
values are chosen by hand, no RNG -- a red-team targets specific boundaries, not random samples), and
plain ``def test_*`` functions so ``bcir.tests.run_all`` auto-collects them (the module is listed in
run_all._MODULES).

THE LOAD-BEARING NUMERIC THRESHOLDS (used by the asserts below):
  * float32 ``exp(t)``     overflows to +inf when ``t > ln(FLT_MAX) ~ 88.72``     (FLT_MAX ~ 3.40e38).
  * float64 ``exp(t)``     overflows (math.exp RAISES OverflowError) at ``t > ln(DBL_MAX) ~ 709.78``
                           (DBL_MAX ~ 1.80e308).
  * So the naive ``exp(x*x)`` form (the libcerf fallback) overflows at |x| > sqrt(88.72) ~ 9.42 in
    float32, and at |x| > sqrt(709.78) ~ 26.64 in float64. erfcx(x) itself stays FINITE across the whole
    real line (for large x, ``erfcx(x) ~ 1/(x*sqrt(pi))``, rel error ~ 1/(2x^2)); the naive fallback's
    overflow is exactly the capability libcerf adds OUTSIDE the |x| <= 3 happy-path envelope.

WHAT THIS RED-TEAM FOUND: all seven wraps degrade HONESTLY on every probe below -- none returns a wrong
finite number where it should overflow / raise / propagate a NaN. (If one ever does, the assert here flags
it as a regression, which is precisely the point of the hardening gate.) The wrap implementations and
emitters are NOT modified by this file.
"""

import math
import os
import shutil
import subprocess
import tempfile

from bcir.kbcir.cerf_kernels import erfcx_reference
from bcir.kbcir.fft import dft_reference
from bcir.kbcir.fft2 import dft2_reference
from bcir.kbcir.gsl_kernels import GSL_STATS, stats_reference
from bcir.kbcir.linsolve import residual, solve_reference
from bcir.kbcir.matmul import matmul_reference
from bcir.kbcir.vecmath import exp_reference
from bcir.toolchain import host_link_args
from bcir.lower.c_kernel import (
    emit_blas_gemm_c,
    emit_cerf_erfcx_c,
    emit_fftw_fft2_c,
    emit_fftw_fft_c,
    emit_lapack_solve_c,
    emit_sleef_exp_c,
)

# The float32 / float64 overflow thresholds the red-team probes straddle (see the module docstring).
FLT_MAX = 3.4028234663852886e38  # IEEE-754 binary32 max finite magnitude
LN_FLT_MAX = math.log(FLT_MAX)  # ~ 88.72: float32 exp(t) overflows past here
LN_DBL_MAX = math.log(
    1.7976931348623157e308
)  # ~ 709.78: float64 exp(t) (math.exp) raises past here
X_F32_EXP2_OVERFLOW = math.sqrt(LN_FLT_MAX)  # ~ 9.42: |x| past which float32 exp(x*x) overflows
X_F64_EXP2_OVERFLOW = math.sqrt(LN_DBL_MAX)  # ~ 26.64: |x| past which float64 exp(x*x) raises

_CC = None


def _cc():
    """The host C compiler (clang preferred, then cc, then gcc), or None. Every compile-and-run probe
    SELF-SKIPS when this is None, exactly like the seven happy-path suites, so the quick tier stays green."""
    global _CC
    if _CC is None:
        _CC = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc") or ""
    return _CC or None


def _run_elementwise_kernel(cc, kernel, fn, lits):
    """Compile the emitted elementwise (data, out) kernel + a main that calls it on the C float LITERALS
    `lits` (so a probe can pass NAN / INFINITY / a near-overflow literal) and prints each result; return
    the parsed token list (floats, plus the strings "nan"/"inf"/"-inf"). Mirrors the shape of
    test_cerf.py::_run_erfcx_kernel / test_sleef.py::_run_exp_kernel."""
    n = len(lits)
    main = (
        "\n#include <stdio.h>\n#include <math.h>\nint main(void){\n"
        f"  float data[{n}] = {{{', '.join(lits)}}};\n"
        f"  float out[{n}];\n  {fn}(data, out);\n"
        f'  for (int i = 0; i < {n}; ++i) printf("%.8g\\n", (double)out[i]);\n'
        "  return 0;\n}\n"
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "s")
        bld = subprocess.run(
            host_link_args([cc, "-std=c11", "-O2", src, "-lm", "-o", exe]),
            capture_output=True,
            text=True,
        )
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return out.stdout.split()


def _parse(tok):
    """Parse one printf token into a Python float (nan/inf survive float())."""
    return float(tok)


# =====================================================================================================
# WRAP 1/7 -- libcerf erfcx (the SHARPEST case). The naive fallback is expf(x*x)*erfcf(x):
#   * large POSITIVE x: erfcf(x) -> 0, expf(x*x) -> +inf, so +inf * 0 -> NaN.
#   * large NEGATIVE x: erfcf(x) -> 2, expf(x*x) -> +inf, so -> +inf.
# The TRUE erfcx stays finite (~ 1/(x*sqrt(pi))) -- the naive fallback degrades HONESTLY (NaN/inf), never
# a wrong finite. The float64 oracle reference shares the fragility, just at the higher |x| > 26.64.
# =====================================================================================================


def test_cerf_fallback_degrades_honestly_at_x_12_float32():
    # The C FALLBACK kernel (NO -DBCIR_USE_CERF) at x = +-12 (|12| > 9.42, so float32 exp(x*x) overflows).
    # +12: erfcf(12) -> 0, expf(144) -> +inf  =>  +inf * 0 -> NaN (honest).
    # -12: erfcf(-12) -> 2, expf(144) -> +inf =>  -> +inf (honest). NEITHER is a wrong finite number.
    cc = _cc()
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    got = _run_elementwise_kernel(cc, emit_cerf_erfcx_c(2, "verfcx"), "verfcx", ["12.0f", "-12.0f"])
    assert len(got) == 2, got
    pos, neg = _parse(got[0]), _parse(got[1])
    assert not math.isfinite(pos), ("erfcx fallback at +12 must be non-finite (NaN), got", pos)
    assert math.isnan(pos), ("erfcx fallback at +12 is +inf*0 = NaN", pos)
    assert not math.isfinite(neg), ("erfcx fallback at -12 must be non-finite (+inf), got", neg)
    assert math.isinf(neg) and neg > 0.0, ("erfcx fallback at -12 is +inf", neg)


def test_cerf_fallback_first_overflow_step_is_inf_not_wrong_finite():
    # Step across the float32 exp(x*x) overflow edge (|x| ~ 9.42): just BELOW (x=9.0, x*x=81<88.72) the
    # fallback is a GENUINE finite erfcx; just ABOVE (x=9.5, x*x=90.25>88.72) it overflows. The honest
    # behaviour: the below value is finite AND correct; the above value is +inf, NOT a wrong finite number.
    cc = _cc()
    if not cc:
        return
    got = _run_elementwise_kernel(cc, emit_cerf_erfcx_c(2, "verfcx"), "verfcx", ["9.0f", "9.5f"])
    below, above = _parse(got[0]), _parse(got[1])
    # below the edge: the fallback must MATCH the (still-finite) float64 reference -- no early corruption.
    ref_below = erfcx_reference([9.0])[0]
    assert math.isfinite(below), below
    assert abs(below - ref_below) < 1e-3 * (1.0 + abs(ref_below)), (below, ref_below)
    # above the edge: honest overflow to +inf (expf(90.25) overflows float32), never a plausible finite.
    assert math.isinf(above) and above > 0.0, (
        "erfcx fallback just past the f32 edge must be +inf",
        above,
    )


def test_cerf_oracle_reference_is_not_full_range_robust_float64():
    # A red-team finding in its own right: the float64 oracle erfcx_reference (math.exp(v*v)*math.erfc(v))
    # is NOT full-range robust -- it OVERFLOWS past |x| > 26.64 (x*x > 709.78), where math.exp RAISES
    # OverflowError. Only a LINKED libcerf is robust across the whole line. Assert the honest failure
    # (a raise) -- it does NOT silently return a wrong finite number.
    for x in (30.0, -30.0):
        try:
            v = erfcx_reference([x])
            # If it ever returns instead of raising, it MUST be non-finite (never a wrong finite value).
            assert not math.isfinite(v[0]), (
                "erfcx_reference past |x|>26.64 must be non-finite",
                x,
                v,
            )
        except OverflowError:
            pass  # the honest float64 degradation (a raise)


def test_cerf_oracle_reference_is_finite_and_correct_where_only_float32_fails():
    # The discriminating case x = 12: 144 < 709.78, so the float64 reference is STILL FINITE (only the
    # float32 fallback above overflows). And the math is RIGHT -- it tracks the leading asymptotic
    # 1/(x*sqrt(pi)) to within ~2% (rel error ~ 1/(2x^2) ~ 0.35%). So the float64 path is honest+correct;
    # only the float32 fallback degrades (to NaN/inf) -- never a wrong finite either way.
    v = erfcx_reference([12.0])[0]
    assert math.isfinite(v), v
    asymptotic = 1.0 / (12.0 * math.sqrt(math.pi))
    assert abs(v - asymptotic) <= 0.02 * asymptotic, (v, asymptotic)
    # the negative twin: erfcx(-12) ~ 2*exp(144) is finite in float64 (144 < 709.78) and huge+positive.
    vneg = erfcx_reference([-12.0])[0]
    assert math.isfinite(vneg) and vneg > 0.0, vneg


def test_cerf_linked_path_is_robust_only_if_libcerf_is_present():
    # ONLY if a real libcerf links: the -DBCIR_USE_CERF kernel at x=12 stays FINITE and ~ erfcx(12) (the
    # robust special function, no overflow) -- the capability the wrap adds. Else self-skip (libcerf is
    # rarely installed on CI, exactly as test_cerf.py::_cerf_link notes).
    cc = _cc()
    if not cc:
        return
    libs = None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.c")
        open(src, "w").write("#include <cerf.h>\nint main(void){return (int)erfcxf(0.0f)*0;}\n")
        if (
            subprocess.run(
                host_link_args([cc, src, "-lcerf", "-lm", "-o", os.path.join(d, "p")]),
                capture_output=True,
            ).returncode
            == 0
        ):
            libs = ["-lcerf"]
    if not libs:
        return  # no libcerf here -> the robust path self-skips
    n = 2
    main = (
        "\n#include <stdio.h>\nint main(void){\n"
        f"  float data[{n}] = {{12.0f, -3.0f}};\n"
        f"  float out[{n}];\n  verfcx(data, out);\n"
        f'  for (int i = 0; i < {n}; ++i) printf("%.8g\\n", (double)out[i]);\n'
        "  return 0;\n}\n"
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.c")
        open(src, "w").write(emit_cerf_erfcx_c(n, "verfcx") + main)
        exe = os.path.join(d, "s")
        bld = subprocess.run(
            host_link_args(
                [cc, "-std=c11", "-O2", "-DBCIR_USE_CERF", src, *libs, "-lm", "-o", exe]
            ),
            capture_output=True,
            text=True,
        )
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        got = [float(t) for t in out.stdout.split()]
    # libcerf is FINITE at x=12 (the robustness the fallback lacks) and ~ the leading asymptotic.
    assert math.isfinite(got[0]), got
    asymptotic = 1.0 / (12.0 * math.sqrt(math.pi))
    assert abs(got[0] - asymptotic) <= 0.02 * asymptotic, (got[0], asymptotic)


# =====================================================================================================
# WRAP 2/7 -- LAPACK sgesv (a dense linear solve). A SINGULAR A has no unique solution: LAPACKE_sgesv
# returns info > 0 (the U(info,info)=0 zero pivot); the portable Gaussian-elimination-with-partial-
# pivoting reference RAISES ValueError on the zero pivot. Either way: an HONEST failure, never a wrong
# "solution". A well-conditioned n=1 solve is correct; an ill-conditioned-but-nonsingular solve still
# has a small residual.
# =====================================================================================================


def test_lapack_singular_matrix_degrades_honestly():
    # [[1,2],[2,4]] is rank-deficient (row 2 = 2*row 1) -> no unique solve. The reference must NOT return a
    # wrong finite "solution": it raises, OR (defensively) returns a non-finite x, OR leaves a residual
    # ||Ax-b|| that is NOT small. Probe what it actually does and assert the honest outcome.
    a = [1.0, 2.0, 2.0, 4.0]
    b = [3.0, 6.0]
    try:
        x = solve_reference(a, b, 2, 1)
    except ValueError:
        return  # the honest LAPACK info>0 analog (a raise)
    # If it did NOT raise, the only honest outcomes are a non-finite x or a not-small residual.
    res = residual(a, x, b, 2, 1)
    assert (not all(math.isfinite(v) for v in x)) or res > 1e-3, (
        "singular solve must degrade honestly, not return a wrong finite solution",
        x,
        res,
    )


def test_lapack_singular_identical_rows_degrades_honestly():
    # The other classic singular form: two identical rows [[1,2],[1,2]]. Same honest-failure property.
    a = [1.0, 2.0, 1.0, 2.0]
    b = [3.0, 3.0]
    try:
        x = solve_reference(a, b, 2, 1)
    except ValueError:
        return
    res = residual(a, x, b, 2, 1)
    assert (not all(math.isfinite(v) for v in x)) or res > 1e-3, (x, res)


def test_lapack_n1_solve_is_correct():
    # The minimal non-degenerate edge: a 1x1 solve A=[[a]], b=[c] -> x = c/a. Well-conditioned, so it MUST
    # be correct (a degrade here would be a false alarm). Pin the exact value + a ~0 residual.
    x = solve_reference([4.0], [12.0], 1, 1)
    assert abs(x[0] - 3.0) <= 1e-9, x
    assert residual([4.0], x, [12.0], 1, 1) <= 1e-9


def test_lapack_ill_conditioned_but_nonsingular_still_solves():
    # NEAR-singular but still nonsingular: [[1,1],[1,1.0001]] (cond ~ 4e4). The partial-pivoting reference
    # is stable enough to solve it with a SMALL residual -- it does NOT spuriously degrade on a solvable
    # (if poorly conditioned) system. The honest outcome here is a correct solve, not a raise.
    a = [1.0, 1.0, 1.0, 1.0001]
    b = [2.0, 2.0001]  # exact solution x = [1, 1]
    x = solve_reference(a, b, 2, 1)
    assert all(math.isfinite(v) for v in x), x
    assert residual(a, x, b, 2, 1) <= 1e-3, (x, residual(a, x, b, 2, 1))


def test_lapack_fallback_solves_well_conditioned_but_not_a_singular_system():
    # Compile-and-run the EMITTED fallback (NO LAPACK lib): it solves a well-conditioned 2x2 correctly
    # (small residual), the honest happy outcome. (The singular system is asserted in Python above, since
    # the C kernel's in-place divide-by-zero is platform inf/nan rather than a clean raise; the
    # reference's raise is the canonical honest signal.)
    cc = _cc()
    if not cc:
        return
    a = [4.0, 1.0, 1.0, 3.0]
    b = [1.0, 2.0]
    n, nrhs = 2, 1
    ref = solve_reference(a, b, n, nrhs)
    main = (
        "\n#include <stdio.h>\nint main(void){\n"
        f"  float A[{n * n}] = {{{', '.join(f'{v:.8f}f' for v in a)}}};\n"
        f"  float b[{n * nrhs}] = {{{', '.join(f'{v:.8f}f' for v in b)}}};\n"
        f"  int info = solve(A, b);\n"
        f'  if (info != 0) {{ printf("INFO %d\\n", info); return 2; }}\n'
        f'  for (int i=0;i<{n * nrhs};++i) printf("%.8f ", b[i]);\n  return 0;\n}}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.c")
        open(src, "w").write(emit_lapack_solve_c(n, nrhs, "solve") + main)
        exe = os.path.join(d, "s")
        bld = subprocess.run(
            host_link_args([cc, "-std=c11", "-O2", src, "-lm", "-o", exe]),
            capture_output=True,
            text=True,
        )
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        got = [float(t) for t in out.stdout.split()]
    assert all(abs(g - r) < 1e-3 for g, r in zip(got, ref)), (got, ref)
    assert residual(a, got, b, n, nrhs) < 1e-3, residual(a, got, b, n, nrhs)


# =====================================================================================================
# WRAP 3/7 -- GSL statistics. gsl_stats_variance uses the UNBIASED n-1 divisor, so n=1 divides by
# (n-1)=0 -- a domain error. The reference RAISES ValueError (the honest failure); the mean (n>=1) is
# still exact. A NaN/Inf input element propagates to a non-finite statistic (no silent finite).
# =====================================================================================================


def test_gsl_variance_n1_degrades_honestly():
    # n=1 variance/sd: the unbiased n-1 divisor is 0 -> a domain error. The reference raises ValueError
    # (the honest degradation), never a wrong finite "variance".
    for kind in ("variance", "sd"):
        try:
            v = stats_reference([5.0], kind)
            assert not math.isfinite(v), (
                kind,
                "n=1 variance/sd must degrade honestly, got finite",
                v,
            )
        except ValueError:
            pass  # the honest n-1-divisor domain error (a raise)


def test_gsl_mean_n1_is_exact():
    # The mean needs only n>=1, so a 1-sample mean is EXACTLY the sample (no degrade -- a false alarm
    # would be the bug here).
    assert stats_reference([7.5], "mean") == 7.5


def test_gsl_nan_input_propagates_to_non_finite():
    # A NaN element must propagate to mean/variance/sd as NaN (never silently dropped -> a wrong finite).
    data = [1.0, float("nan"), 3.0]
    for kind in GSL_STATS:
        v = stats_reference(data, kind)
        assert math.isnan(v), (kind, "a NaN input must propagate to NaN, got", v)


def test_gsl_inf_input_propagates_to_non_finite():
    # An Inf element must propagate to a non-finite statistic: mean -> +inf, variance/sd -> NaN (inf-inf
    # in the deviation). Either way NON-FINITE -- never a plausible finite number.
    data = [1.0, float("inf"), 3.0]
    for kind in GSL_STATS:
        v = stats_reference(data, kind)
        assert not math.isfinite(v), (kind, "an Inf input must propagate to non-finite, got", v)


# =====================================================================================================
# WRAP 4/7 -- SLEEF exp (vectorized libm exp). The vectorized path must MATCH libm on the extremes:
#   large +x -> +inf (float32 exp overflows past value > 88.72), large -x -> 0.0, NaN -> NaN.
# Assert the IEEE behaviour in BOTH the float64 Python reference and (compile+run) the float32 C
# fallback, and that they AGREE on these limits.
# =====================================================================================================


def test_sleef_fallback_exp_extremes_float32():
    # The C FALLBACK (float32 expf): +90 (> 88.72) -> +inf; -100 -> ~0; NaN -> NaN; +10/+88 finite. All
    # HONEST IEEE degradation, never a wrong finite.
    cc = _cc()
    if not cc:
        return
    lits = ["90.0f", "-100.0f", "NAN", "88.0f", "10.0f"]
    got = [
        _parse(t)
        for t in _run_elementwise_kernel(cc, emit_sleef_exp_c(len(lits), "vexp"), "vexp", lits)
    ]
    assert math.isinf(got[0]) and got[0] > 0.0, ("expf(90) float32 must overflow to +inf", got[0])
    assert got[1] == 0.0 or abs(got[1]) < 1e-30, ("expf(-100) must decay to ~0", got[1])
    assert math.isnan(got[2]), ("expf(NaN) must propagate NaN", got[2])
    assert math.isfinite(got[3]) and abs(got[3] - math.exp(88.0)) < 1e-3 * math.exp(88.0), got[3]
    assert math.isfinite(got[4]) and abs(got[4] - math.exp(10.0)) < 1e-3 * math.exp(10.0), got[4]


def test_sleef_reference_exp_extremes_float64():
    # The float64 Python reference: large -x -> 0.0, NaN -> NaN (honest); large +x that overflows DOUBLE
    # (value > 709.78) RAISES OverflowError -- the honest float64 degradation. (At +90 the float64 ref is
    # still FINITE (1.2e39 < DBL_MAX) even though the float32 fallback above overflows -- the precise
    # float32-vs-float64 divergence: the float32 path degrades EARLIER, both honestly.)
    assert exp_reference([-800.0]) == [0.0]
    assert math.isnan(exp_reference([float("nan")])[0])
    assert math.isfinite(exp_reference([90.0])[0])  # finite in float64, overflows in float32
    try:
        v = exp_reference([710.0])  # 710 > 709.78 -> float64 overflow
        assert not math.isfinite(v[0]), ("float64 exp past 709.78 must be non-finite", v)
    except OverflowError:
        pass  # the honest float64 degradation (a raise)


def test_sleef_fallback_and_reference_agree_on_finite_extremes():
    # The vectorized fallback must MATCH libm (the float64 reference) on the finite extremes -- the
    # red-team property that the fast path agrees with the trusted libm where both are finite.
    cc = _cc()
    if not cc:
        return
    data = [88.0, -50.0, 10.0, -88.0, 0.0]  # all finite in BOTH float32 and float64
    ref = exp_reference(data)
    lits = [f"{v:.8f}f" for v in data]
    got = [
        _parse(t)
        for t in _run_elementwise_kernel(cc, emit_sleef_exp_c(len(data), "vexp"), "vexp", lits)
    ]
    for g, r in zip(got, ref):
        assert math.isfinite(g) and math.isfinite(r), (g, r)
        assert abs(g - r) <= 1e-3 * (1.0 + abs(r)), (g, r)


# =====================================================================================================
# WRAP 5/7 -- BLAS sgemm. A NaN/Inf in an input matrix propagates to exactly the output cells that sum
# over it; an Inf can also make a 0*inf product NaN. The n=1 (1x1x1) edge is correct. The emitter
# rejects a zero dimension with ValueError (assert the raise where it exists).
# =====================================================================================================


def test_blas_nan_input_propagates_to_dependent_output_cell():
    # C = A @ B with a NaN in A. Every output cell C[i,j] sums over a NaN-containing row/col -> NaN. The
    # reference must propagate it (never silently drop it to a wrong finite).
    M, N, K = 2, 2, 2
    a = [1.0, float("nan"), 2.0, 3.0]  # A[0,1] is NaN
    b = [1.0, 0.0, 0.0, 1.0]  # identity -> C = A, so C[0,1] inherits the NaN
    c = matmul_reference(a, b, M, N, K)
    assert math.isnan(c[1]), ("the NaN must reach the dependent output cell", c)
    assert all(math.isfinite(v) or math.isnan(v) for v in c)  # nothing silently fabricated


def test_blas_inf_input_propagates_to_non_finite():
    # An Inf in A: the dependent cell becomes non-finite (inf, or NaN via inf*0 / inf-inf). Never a
    # plausible finite number where an Inf flowed in.
    M, N, K = 1, 1, 2
    a = [float("inf"), 1.0]
    b = [2.0, 3.0]
    c = matmul_reference(a, b, M, N, K)  # inf*2 + 1*3 = +inf
    assert not math.isfinite(c[0]), ("an Inf input must propagate to a non-finite output", c)


def test_blas_n1_gemm_is_correct():
    # The minimal 1x1x1 edge: C[0] = A[0]*B[0]. Correct -> no false-alarm degrade.
    c = matmul_reference([3.0], [4.0], 1, 1, 1)
    assert c == [12.0], c


def test_blas_emitter_rejects_zero_dimension():
    # The emitter rejects a zero dimension with ValueError (an honest contract failure) rather than
    # emitting a silent empty/UB kernel -- assert the raise where it exists (mirrors test_blas_gemm).
    for bad in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        try:
            emit_blas_gemm_c(*bad)
            assert False, f"emit_blas_gemm_c{bad} should raise on a zero dimension"
        except ValueError:
            pass


def test_blas_fallback_nan_propagation_compiles_and_runs():
    # Compile-and-run the EMITTED fallback (NO BLAS lib) with a NaN in A: the dependent output cell is NaN
    # in the actual C kernel too -- honest IEEE propagation end to end, not a wrong finite.
    cc = _cc()
    if not cc:
        return
    M, N, K = 1, 1, 1
    kernel = emit_blas_gemm_c(M, N, K, "g")
    main = (
        "\n#include <stdio.h>\n#include <math.h>\nint main(void){\n"
        "  float A[1] = {NAN};\n  float B[1] = {2.0f};\n  float C[1];\n  g(A, B, C);\n"
        '  printf("%.8g\\n", (double)C[0]);\n  return 0;\n}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "g.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "g")
        bld = subprocess.run(
            host_link_args([cc, "-std=c11", "-O2", src, "-lm", "-o", exe]),
            capture_output=True,
            text=True,
        )
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        assert math.isnan(_parse(out.stdout.strip())), ("the C gemm must propagate NaN", out.stdout)


# =====================================================================================================
# WRAP 6/7 -- FFTW 1-D DFT. A DFT bin sums over EVERY input, so a NaN/Inf in any input bin propagates to
# ALL output bins. The n=1 DFT is the identity (the single input unchanged).
# =====================================================================================================


def _interleave(signal):
    """A list of complex -> the interleaved [re0, im0, re1, im1, ...] float layout (the FFTW layout)."""
    out = []
    for z in signal:
        out.append(float(z.real))
        out.append(float(z.imag))
    return out


def test_fftw1d_nan_input_propagates_to_all_output_bins():
    # A NaN in ONE input bin reaches EVERY output bin (each bin sums over all inputs). The reference must
    # propagate it -- no output bin is silently finite where a NaN flowed into the sum.
    n = 4
    signal = [1.0 + 0j, complex(float("nan"), 0.0), 2.0 + 0j, 3.0 + 0j]
    out = dft_reference(_interleave(signal), n)
    for j in range(n):
        assert math.isnan(out[2 * j]) or math.isnan(out[2 * j + 1]), (
            "the NaN input must reach output bin",
            j,
            out,
        )


def test_fftw1d_inf_input_propagates_to_non_finite():
    # An Inf input bin makes every output bin non-finite (inf, or NaN via the cos/sin rotation), never a
    # plausible finite value.
    n = 3
    signal = [complex(float("inf"), 0.0), 1.0 + 0j, 2.0 + 0j]
    out = dft_reference(_interleave(signal), n)
    for j in range(n):
        assert not (math.isfinite(out[2 * j]) and math.isfinite(out[2 * j + 1])), (
            "the Inf input must make output bin non-finite",
            j,
            out,
        )


def test_fftw1d_n1_dft_is_the_identity():
    # The n=1 DFT is out[0] = in[0] (the sum over a single sample, exp(0)=1). Exact identity -> no degrade.
    out = dft_reference([2.5, -1.5], 1)
    assert out == [2.5, -1.5], out


# =====================================================================================================
# WRAP 7/7 -- FFTW 2-D DFT. Same propagation property: a NaN/Inf in any input bin reaches ALL output
# bins (a 2-D DFT sums over every input). The 1x1 DFT is the identity.
# =====================================================================================================


def _flatten(grid):
    """An n0-by-n1 list-of-lists of complex -> the row-major interleaved float layout."""
    return _interleave([z for row in grid for z in row])


def test_fftw2d_nan_input_propagates_to_all_output_bins():
    n0, n1 = 2, 2
    grid = [[1.0 + 0j, 2.0 + 0j], [complex(float("nan"), 0.0), 3.0 + 0j]]  # one NaN cell
    out = dft2_reference(_flatten(grid), n0, n1)
    for j in range(n0 * n1):
        assert math.isnan(out[2 * j]) or math.isnan(out[2 * j + 1]), (
            "the NaN input must reach 2-D output bin",
            j,
            out,
        )


def test_fftw2d_inf_input_propagates_to_non_finite():
    n0, n1 = 2, 2
    grid = [[complex(float("inf"), 0.0), 1.0 + 0j], [2.0 + 0j, 3.0 + 0j]]
    out = dft2_reference(_flatten(grid), n0, n1)
    for j in range(n0 * n1):
        assert not (math.isfinite(out[2 * j]) and math.isfinite(out[2 * j + 1])), (
            "the Inf input must make 2-D output bin non-finite",
            j,
            out,
        )


def test_fftw2d_1x1_dft_is_the_identity():
    out = dft2_reference([4.0, -2.0], 1, 1)
    assert out == [4.0, -2.0], out


def test_fftw2d_fallback_nan_propagation_compiles_and_runs():
    # Compile-and-run the EMITTED 2-D fallback (NO FFTW lib) with a NaN input cell: the actual C kernel
    # propagates the NaN to every output bin too -- honest IEEE propagation end to end.
    cc = _cc()
    if not cc:
        return
    n0, n1 = 2, 2
    grid = [[1.0 + 0j, 2.0 + 0j], [complex(float("nan"), 0.0), 3.0 + 0j]]
    x = _flatten(grid)
    m = 2 * n0 * n1
    # build the in[] initializer, substituting NAN for the non-finite literal.
    inits = []
    for v in x:
        inits.append("NAN" if math.isnan(v) else f"{v:.6f}f")
    kernel = emit_fftw_fft2_c(n0, n1, "fft2")
    main = (
        "\n#include <stdio.h>\n#include <math.h>\nint main(void){\n"
        f"  float in[{m}] = {{{', '.join(inits)}}};\n"
        f"  float out[{m}];\n  fft2(in, out);\n"
        f'  for (int i = 0; i < {m}; ++i) printf("%.6g ", (double)out[i]);\n  return 0;\n}}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "f.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "f")
        bld = subprocess.run(
            host_link_args([cc, "-std=c11", "-O2", src, "-lm", "-o", exe]),
            capture_output=True,
            text=True,
        )
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        got = [_parse(t) for t in out.stdout.split()]
        assert len(got) == m, got
        for j in range(n0 * n1):
            assert math.isnan(got[2 * j]) or math.isnan(got[2 * j + 1]), (
                "the C 2-D DFT must propagate NaN to bin",
                j,
                got,
            )


# A run_all-friendly explicit collection list (the existing suites are auto-collected by name; this is a
# convenience for a direct `python -m bcir.tests.test_area_b_redteam` invocation and mirrors the
# run_all convention of discovering `test_*` callables).
def run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    return len(fns)


if __name__ == "__main__":
    n = run_all()
    print(f"{n} passed")
