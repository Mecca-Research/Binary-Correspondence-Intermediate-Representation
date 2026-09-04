"""E1 (ML-breadth) -- the OVERDETERMINED ordinary-least-squares (OLS) wrap: generalize the square dense solve
(linsolve / LAPACK sgesv) to linear regression (minimize ||A x - b||_2) via the SAME Area-B "integrate, don't
reinvent" pattern. The OLS source of truth (ols_reference) forms the NORMAL EQUATIONS G = A^T A, c = A^T b and
REUSES linsolve.solve_reference for the inner square solve; the emitted C delegates to LAPACK's QR-based sgels
when linked, with a portable normal-equations fallback. BCIR owns the calling side; the bridge is the R17 input
round-trip.

Covers: ols_reference recovers a KNOWN x on a CONSISTENT overdetermined system (b = A x_true -> recovers
x_true) and, on an INCONSISTENT system, the solution satisfies the OLS optimality condition
normal_equation_residual ~ 0 (the best fit, ||A x - b|| > 0); the bridged solve tracks the reference within the
R17 input bound (well-conditioned A); the emitted C wraps LAPACKE_sgels with a normal-equations fallback (the
c.call.libm:LAPACKE_sgels label + the signature + n<1/m<n raise); the fallback compiles + runs + matches
ols_reference to float round-off (no LAPACK needed); only if a LAPACK lib is present, the linked path agrees;
the LAPACK link-flag rule resolves LAPACKE_sgels -> -llapack (no regression on the other five rules + libm +
libc + unknown); and the module imports no verifier / emits no Diagnostic (the two-truth quarantine). The point:
the win is on the calling side, not a reimplemented least-squares solver -- and the honest conditioning note
(normal equations square cond(A); QR / sgels is the more stable path) holds.
"""

import ast
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.ols import normal_equation_residual, ols_reference, ols_via_bridge
from bcir.kbcir.precision import quantization_error_bound
from bcir.lower.c_kernel import emit_lapack_ols_c
from bcir.toolchain import host_link_args


def _lapack_link():
    """A LAPACK lib flag that links (with lapacke.h present), or None (the real-LAPACK path self-skips)."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.c")
        open(src, "w").write(
            "#include <lapacke.h>\nint main(void){return (int)LAPACK_ROW_MAJOR*0;}\n"
        )
        for lib in (["-llapacke", "-llapack"], ["-llapacke"], ["-llapack"]):
            if (
                subprocess.run(
                    [cc, src, *lib, "-o", os.path.join(d, "p")], capture_output=True
                ).returncode
                == 0
            ):
                return lib
    return None


def _design(m: int, n: int):
    """A WELL-CONDITIONED overdetermined m x n design matrix (row-major). Column 0 is a bias (all 1s); the
    remaining columns are GENUINELY independent functions of the row index (not all affine in i -- a basis of
    distinct shapes: x, x^2-ish, alternating, ...), so A has full column rank n and A^T A is well-conditioned
    (the normal equations stay stable). Returns the flat row-major list."""
    import math

    def col(i: int, j: int) -> float:
        # j>=1: a small library of linearly-INDEPENDENT column shapes (a polynomial/Fourier-ish basis).
        if j == 1:
            return float(i)  # linear
        if j == 2:
            return float(i * i) * 0.1  # quadratic (independent of linear + bias)
        if j == 3:
            return math.sin(0.7 * i + 0.3)  # oscillatory
        return float(i**j) * (0.05 ** (j - 1)) + (1.0 if i % 2 else -1.0) * 0.5

    a = []
    for i in range(m):
        a.append(1.0)  # bias column
        a.extend(col(i, j) for j in range(1, n))
    return a


def _matvec(a, x, m, n, nrhs=1):
    """A x (m x nrhs), row-major -- an INDEPENDENT product (no BCIR code) used to build a consistent b."""
    out = [0.0] * (m * nrhs)
    for i in range(m):
        for r in range(nrhs):
            s = 0.0
            for j in range(n):
                s += a[i * n + j] * x[j * nrhs + r]
            out[i * nrhs + r] = s
    return out


# --- the oracle reference: recovers a known x on a consistent system; optimal on an inconsistent one --------


def test_ols_recovers_a_known_x_on_a_consistent_overdetermined_system():
    # m > n, b = A x_true exactly -> OLS recovers x_true to round-off (the fit IS exact, residual ~0).
    m, n = 8, 3
    a = _design(m, n)
    x_true = [2.0, -1.5, 0.75]  # n coefficients to recover
    b = _matvec(a, x_true, m, n)  # consistent: b lies in the column space of A
    x = ols_reference(a, b, m, n)
    assert len(x) == n
    assert all(abs(x[j] - x_true[j]) <= 1e-7 for j in range(n)), (x, x_true)
    # the genuine independent check: the OLS optimality residual A^T(A x - b) ~ 0.
    assert normal_equation_residual(a, x, b, m, n) <= 1e-6, normal_equation_residual(a, x, b, m, n)


def test_ols_fits_y_equals_2x_plus_1_a_line_of_best_fit():
    # The textbook linear regression: fit y = c1*x + c0 from points on the line (m=8 points, n=2 coeffs
    # [c0, c1]). Design rows [1, x_i]; recovers [1, 2] exactly (consistent), residual ~0.
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    m, n = len(xs), 2
    a = [v for x in xs for v in (1.0, x)]  # row-major [1, x_i]
    b = [2.0 * x + 1.0 for x in xs]  # exactly on the line y = 2x + 1
    coef = ols_reference(a, b, m, n)  # coef = [c0, c1]
    assert abs(coef[0] - 1.0) <= 1e-7 and abs(coef[1] - 2.0) <= 1e-7, coef
    assert normal_equation_residual(a, coef, b, m, n) <= 1e-6


def test_ols_is_optimal_on_an_inconsistent_system_residual_zero_but_fit_imperfect():
    # An INCONSISTENT system (b NOT in the column space): the OLS solution does NOT make ||A x - b|| = 0, but
    # it DOES make the optimality residual A^T(A x - b) ~ 0 (the best fit). So normal_equation_residual ~ 0
    # while the per-row fit error ||A x - b|| is strictly positive.
    m, n = 5, 2
    a = [
        1.0,
        0.0,  # rows: [1, x]
        1.0,
        1.0,
        1.0,
        2.0,
        1.0,
        3.0,
        1.0,
        4.0,
    ]
    b = [1.0, 0.0, 3.0, 2.0, 5.0]  # noisy / not collinear -> no exact fit
    x = ols_reference(a, b, m, n)
    # optimality: A^T(A x - b) ~ 0 (the genuine independent verifier, recomputed directly).
    assert normal_equation_residual(a, x, b, m, n) <= 1e-5, normal_equation_residual(a, x, b, m, n)
    # but the fit is NOT exact -- some row residual is meaningfully nonzero (inconsistent system).
    r = _matvec(a, x, m, n)
    worst_row = max(abs(r[i] - b[i]) for i in range(m))
    assert worst_row > 0.1, worst_row


def test_ols_supports_multiple_right_hand_sides():
    # nrhs > 1: two consistent regressions at once (the sgels nrhs feature). Each rhs recovers its own x_true.
    m, n, nrhs = 6, 2, 2
    a = _design(m, n)
    x_true = [
        3.0,
        1.0,  # rhs0 = [3,1] ... laid out n x nrhs row-major
        -2.0,
        0.5,
    ]  # ... per column: x[j*nrhs + r]
    b = _matvec(a, x_true, m, n, nrhs)
    x = ols_reference(a, b, m, n, nrhs)
    assert all(abs(x[k] - x_true[k]) <= 1e-6 for k in range(n * nrhs)), (x, x_true)
    assert normal_equation_residual(a, x, b, m, n, nrhs) <= 1e-5


def test_ols_reference_rejects_a_rank_deficient_design():
    # Two identical columns -> A is column-rank-deficient -> G = A^T A is singular -> no unique minimizer.
    # The honest failure (LAPACK sgels info > 0), surfaced by the reused solve_reference raising on G.
    m, n = 4, 2
    a = [
        1.0,
        1.0,  # column 0 == column 1 -> rank 1
        2.0,
        2.0,
        3.0,
        3.0,
        4.0,
        4.0,
    ]
    b = [1.0, 2.0, 3.0, 4.0]
    try:
        ols_reference(a, b, m, n)
        assert False, "rank-deficient design (singular normal matrix) should raise"
    except ValueError:
        pass


def test_ols_reference_rejects_malformed_inputs():
    # n<1, m<n (underdetermined), bad a/b lengths, nrhs<1 -- each the honest dimension failure.
    bad = [
        ([1.0], [1.0], 1, 0, 1),  # n < 1
        (_design(2, 3), [1.0, 2.0], 2, 3, 1),  # m < n (underdetermined)
        ([1.0, 2.0, 3.0], [1.0, 2.0], 3, 2, 1),  # len(a) != m*n
        (_design(4, 2), [1.0, 2.0, 3.0], 4, 2, 1),  # len(b) != m*nrhs
        (_design(4, 2), [0.0] * 4, 4, 2, 0),  # nrhs < 1
    ]
    for args in bad:
        try:
            ols_reference(*args)
            assert False, f"{args[2:]} should raise"
        except ValueError:
            pass


def test_ols_reference_is_side_effect_free():
    # The oracle builds the normal matrix into fresh buffers (no in-place overwrite of the caller's data).
    m, n = 5, 2
    a = _design(m, n)
    b = _matvec(a, [1.0, 2.0], m, n)
    a0, b0 = list(a), list(b)
    ols_reference(a, b, m, n)
    assert a == a0 and b == b0


# --- the oracle bridge: tracks the reference within the R17 input round-trip bound ------------------------


def test_bridged_ols_tracks_the_reference_within_quant_error():
    # A WELL-CONDITIONED design (bias + spread columns): OLS via the normal equations amplifies the input
    # round-trip by ~cond(A)^2, so we keep cond(A) modest and bound per-output error by a generous multiple of
    # the R17 input round-trip error (the certified boundary; the solve is trusted/exact on the bridged data).
    rng = random.Random(0xE1)
    m, n = 10, 3
    a = []
    for i in range(m):
        a.append(1.0)  # bias column
        for j in range(1, n):
            a.append(
                rng.uniform(-2.0, 2.0) + 5.0 * j
            )  # spread, well-separated -> well-conditioned A^T A
    x_true = [rng.uniform(-3, 3) for _ in range(n)]
    b = _matvec(a, x_true, m, n)
    from bcir.kbcir.quantize import max_abs_error

    q_err = max_abs_error(
        a + b, group_size=8, bits=8
    )  # the R17 input round-trip bound (real units)
    ref = ols_reference(a, b, m, n)
    got = ols_via_bridge(a, b, m, n, 1, group_size=8, bits=8)
    amp = 200.0  # conservative cond(A)^2-style amplification
    assert all(abs(r - g) <= amp * q_err + 1e-4 for r, g in zip(ref, got)), (ref, got, q_err)


def test_bridge_is_a_clean_roundtrip_then_trusted_ols():
    # ols_via_bridge == ols_reference of the dequantized (round-tripped) inputs -- the trusted solver sees
    # exactly the bridged values, so the bridge is the only error source (mirrors linsolve.solve_via_bridge).
    from bcir.kbcir.quantize import dequantize, quantize_per_group

    rng = random.Random(0xE1A)
    m, n = 7, 2
    a = []
    for i in range(m):
        a.extend([1.0, rng.uniform(-1.0, 1.0) + 3.0 * (i + 1)])
    b = [rng.uniform(-3, 3) for _ in range(m)]
    da = dequantize(quantize_per_group(a, 8, 8))
    db = dequantize(quantize_per_group(b, 8, 8))
    assert ols_via_bridge(a, b, m, n, 1, 8, 8) == ols_reference(da, db, m, n)


# --- the emitted wrapper: LAPACKE_sgels + portable normal-equations fallback, BCIR owns the layout --------


def test_emit_wraps_lapack_sgels_with_a_normal_equations_fallback():
    c = emit_lapack_ols_c(8, 3, 1, "ols")
    assert (
        "LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', 8, 3, 1, qa, 3, bx, 1)" in c
    )  # the trusted QR least-squares kernel
    assert "c.call.libm:LAPACKE_sgels" in c  # the LAPACK callee edge label
    assert "BCIR_USE_LAPACK" in c and "BCIR_OLS_LAPACK" in c  # link-selected
    assert "G[p * N + q] = s;" in c  # the normal-equations G = A^T A fallback
    assert (
        "if (v > best) { best = v; pivot = row; }" in c
    )  # partial pivoting (numerically necessary)
    assert "#include <lapacke.h>" in c  # the LAPACKE header on the linked path
    assert (
        "void ols(const float *A, const float *b, float *x)" in c
    )  # the signature bakes m,n,nrhs in
    # n<1 / m<n raise (the honest dimension failures).
    for bad in [(0, 0), (3, 5), (4, 0)]:  # n<1 (m=0), m<n (3<5), nrhs handled below
        try:
            emit_lapack_ols_c(*bad)
            assert False, f"{bad} should raise"
        except ValueError:
            pass
    try:
        emit_lapack_ols_c(4, 2, 0)
        assert False, "nrhs<1 should raise"
    except ValueError:
        pass


def _run_ols_kernel(cc, kernel, a, b, m, n, nrhs, define=None, libs=None):
    """Compile the emitted OLS kernel + a main that calls it on (copies of) A,b and prints x; return the
    parsed solution list. The kernel does NOT mutate A/b (it copies into sgels-shaped scratch internally)."""
    main = (
        f"\n#include <stdio.h>\nint main(void){{\n"
        f"  float A[{m * n}] = {{{', '.join(f'{v:.8f}f' for v in a)}}};\n"
        f"  float b[{m * nrhs}] = {{{', '.join(f'{v:.8f}f' for v in b)}}};\n"
        f"  float x[{n * nrhs}] = {{0}};\n"
        f"  ols(A, b, x);\n"
        f'  for (int i=0;i<{n * nrhs};++i) printf("%.8f ", x[i]);\n  return 0;\n}}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "o.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "o")
        cmd = [cc, "-std=c11", "-O2"]
        if define:
            cmd.append(define)
        cmd += [src]
        if libs:
            cmd += libs
        cmd += ["-lm", "-o", exe]  # -lm: fabsf in the fallback path
        bld = subprocess.run(host_link_args(cmd), capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return [float(t) for t in out.stdout.split()]


def test_fallback_path_compiles_runs_and_matches_the_reference():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    m, n = 8, 3
    a = _design(m, n)
    x_true = [2.0, -1.5, 0.75]
    b = _matvec(a, x_true, m, n)
    ref = ols_reference(a, b, m, n)
    got = _run_ols_kernel(cc, emit_lapack_ols_c(m, n, 1, "ols"), a, b, m, n, 1)  # NO LAPACK lib
    assert len(got) == n
    # the C fallback IS the same normal-equations OLS (float32 vs the oracle's float64) -> float round-off ...
    assert all(abs(g - r) < 1e-3 for g, r in zip(got, ref)), (got, ref)
    # ... and the genuine independent check: the optimality residual A^T(A x - b) ~ 0.
    assert normal_equation_residual(a, got, b, m, n) < 1e-2, normal_equation_residual(
        a, got, b, m, n
    )


def test_fallback_path_handles_multiple_right_hand_sides():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return
    m, n, nrhs = 6, 2, 2
    a = _design(m, n)
    x_true = [3.0, 1.0, -2.0, 0.5]
    b = _matvec(a, x_true, m, n, nrhs)
    ref = ols_reference(a, b, m, n, nrhs)
    got = _run_ols_kernel(cc, emit_lapack_ols_c(m, n, nrhs, "ols"), a, b, m, n, nrhs)
    assert all(abs(g - r) < 1e-3 for g, r in zip(got, ref)), (got, ref)
    assert normal_equation_residual(a, got, b, m, n, nrhs) < 1e-2


def test_linked_lapack_path_agrees_when_lapack_is_present():
    libs = _lapack_link()
    if not libs:
        return  # no LAPACK here -> the real-link path self-skips
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    m, n = 8, 3
    a = _design(m, n)
    x_true = [2.0, -1.5, 0.75]
    b = _matvec(a, x_true, m, n)
    ref = ols_reference(a, b, m, n)
    got = _run_ols_kernel(
        cc, emit_lapack_ols_c(m, n, 1, "ols"), a, b, m, n, 1, define="-DBCIR_USE_LAPACK", libs=libs
    )
    # sgels (QR) and the normal-equations reference find the SAME minimizer on a well-conditioned A; agree
    # to float round-off (the honest conditioning note: QR is the more stable path, equal here).
    assert all(abs(g - r) < 1e-2 for g, r in zip(got, ref)), (got, ref)
    assert normal_equation_residual(a, got, b, m, n) < 1e-2


# --- R17 boundary: the bridge step is the INPUT round-trip alone (the solve is trusted/exact) -------------


def test_r17_bound_is_the_input_roundtrip_for_the_quantized_ols():
    # The certified boundary error of a quantized OLS call is the bridge's round-trip step (the trusted
    # least-squares solve itself is exact/trusted -- it contributes no certified error). The R17 grid bound
    # the bridge is held to is 1 ULP, exactly as the linsolve/BLAS/FFTW wraps.
    assert quantization_error_bound() == 1


# --- the LAPACK link-flag rule: LAPACKE_sgels -> -llapack (REUSES the existing rule; no regression) -------


def test_ols_link_flag_rule_reuses_the_existing_lapack_rule():
    # LAPACKE_sgels MATCHES the existing LAPACKE_* rule, so NO linkflags change is needed -- verify it.
    assert (
        library_for_callee("LAPACKE_sgels") == "-llapack"
    )  # the OLS callee (the wrapper's actual symbol)
    assert library_for_callee("LAPACKE_sgesv") == "-llapack"  # the square-solve sibling still maps
    assert (
        library_for_callee("sgels_") == "-llapack" or library_for_callee("sgels_") is None
    )  # sgels not in the Fortran set; either way no surprise flag
    # no regression on the other five library rules + libm + libc + unknown.
    assert library_for_callee("gsl_stats_mean") == "-lgsl"  # #62 GSL still maps
    assert library_for_callee("Sleef_expf1_u10") == "-lsleef"  # #63 SLEEF still maps
    assert library_for_callee("fftwf_execute") == "-lfftw3"  # B2 FFTW still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"  # B5 BLAS still maps
    assert library_for_callee("sqrtf") == "-lm"  # libm still maps
    assert library_for_callee("free") == NO_FLAG  # libc-implicit, known (not unknown)
    assert library_for_callee("totally_unknown_fn") is None  # unknown-callee policy unchanged


# --- the two-truth quarantine: the OLS module imports no verifier / emits no Diagnostic -------------------


def test_ols_touches_no_verifier_and_emits_no_diagnostic():
    # OLS is a cost-side / oracle quantity (off the legality path), never an R-law legality verdict. Inspect
    # the IMPORTED module object: no verify / Diagnostic / Graded names bound, and no verifier module imported.
    import bcir.kbcir.ols as ols_mod

    bound = set(vars(ols_mod))
    assert not (bound & {"verify_quarantine", "Diagnostic", "Graded", "decide"}), sorted(bound)
    tree = ast.parse(open(ols_mod.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)
    # the returned values are plain floats (no graded/confidence wrapper).
    x = ols_reference([1.0, 0.0, 1.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0], 3, 2)
    assert isinstance(x, list) and all(isinstance(v, float) for v in x)
    assert isinstance(
        normal_equation_residual([1.0, 0.0, 1.0, 1.0, 1.0, 2.0], x, [1.0, 2.0, 3.0], 3, 2), float
    )


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
