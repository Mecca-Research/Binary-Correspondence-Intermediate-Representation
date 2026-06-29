"""E2 (ML-breadth) -- the PRINCIPAL COMPONENT ANALYSIS (PCA) wrap: generalize the OLS shape "form a symmetric
matrix then solve" (ols / LAPACK sgels) to "form a symmetric matrix then EIGENDECOMPOSE" via the SAME Area-B
"integrate, don't reinvent" pattern. The PCA source of truth (pca_reference) centers the data, forms the
symmetric covariance C = (1/(m-ddof)) Xc^T Xc, and feeds C to a deterministic JACOBI eigensolver (net-new --
nothing existed to reuse, the analog of E1 reusing the trusted square solve); the emitted C delegates to
LAPACK's ssyev when linked, with a portable Jacobi fallback. BCIR owns the calling side; the bridge is the R17
input round-trip.

Covers: pca_reference recovers KNOWN eigenpairs of a hand-built symmetric matrix and the dominant direction of
a dataset spread along a known axis (first PC ~ that axis, explained-variance ratio dominant); eigen_residual ~
0 and orthonormality_residual ~ 0 on the reference output (the genuine independent checks); eigenvalues sorted
descending; sum ~ trace (variance preserved); the sign convention is applied deterministically; pca_via_bridge
tracks the reference within the R17 input bound on a well-conditioned, well-separated input AND equals
pca_reference of the round-tripped data (clean bridge); dimension/validation errors raise; center_columns /
covariance_matrix / pca_reference are side-effect-free; the emitted C wraps LAPACKE_ssyev with a Jacobi
fallback (the c.call.libm:LAPACKE_ssyev label + the BCIR_USE_LAPACK guard + the Jacobi marker + the signature +
n<1 raise); the fallback compiles + runs + recovers known eigenvalues to float round-off (no LAPACK needed);
only if a LAPACK lib is present, the linked ssyev path agrees; the LAPACK link-flag rule resolves
LAPACKE_ssyev -> -llapack (no regression on the other rules); and the module imports no verifier / emits no
Diagnostic (the two-truth quarantine). The point: the win is on the calling side, not a reimplemented
eigensolver -- and the honest note (ssyev vs Jacobi differ in realization, agree on well-separated eigenvalues;
degenerate eigenvalues make eigenVECTORS non-unique) holds.
"""

import ast
import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.pca import (center_columns, covariance_matrix, eigen_residual,
                            orthonormality_residual, pca_reference, pca_via_bridge,
                            trace_residual)
from bcir.kbcir.precision import quantization_error_bound
from bcir.lower.c_kernel import emit_lapack_eigh_c


def _lapack_link():
    """A LAPACK lib flag that links (with lapacke.h present), or None (the real-LAPACK path self-skips)."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.c")
        open(src, "w").write("#include <lapacke.h>\nint main(void){return (int)LAPACK_ROW_MAJOR*0;}\n")
        for lib in (["-llapacke", "-llapack"], ["-llapacke"], ["-llapack"]):
            if subprocess.run([cc, src, *lib, "-o", os.path.join(d, "p")],
                              capture_output=True).returncode == 0:
                return lib
    return None


def _axis_dataset(m: int, axis, spread: float, jitter: float, seed: int):
    """A WELL-CONDITIONED m x n dataset whose points lie (mostly) along `axis` (a unit n-vector): each sample
    is t_i * spread * axis + small isotropic jitter, with t_i spread out so the variance ALONG `axis` dominates
    every orthogonal direction (a well-separated dominant eigenvalue -> a unique first PC). Returns the flat
    row-major m x n list."""
    n = len(axis)
    rng = random.Random(seed)
    ts = [(-1.0 + 2.0 * i / (m - 1)) for i in range(m)]   # spread along the axis: -1 .. 1
    x = []
    for i in range(m):
        for j in range(n):
            x.append(ts[i] * spread * axis[j] + jitter * (rng.random() - 0.5))
    return x


# --- the oracle reference: recovers known eigenpairs / a known dominant direction -------------------------

def test_pca_recovers_known_eigenpairs_of_a_hand_built_symmetric_matrix():
    # Build data so the covariance is a KNOWN diagonal matrix with distinct entries: feature j is an
    # INDEPENDENT signal (a distinct orthogonal pattern over the m samples) scaled by [3, 2, 1] along the
    # standard axes. Independent (orthogonal) columns -> the covariance is diag(s_j^2 * var_j) with distinct
    # entries, so the eigenvalues are those diagonal entries (DESCENDING) and the eigenvectors are the standard
    # basis vectors e0, e1, e2. (Using the SAME signal per feature would make the columns collinear -> a
    # rank-1 covariance with a single nonzero eigenvalue; distinct per-feature signals avoid that.)
    m, n = 12, 3
    scales = [3.0, 2.0, 1.0]
    # three mutually-orthogonal sample patterns (cos / cos(2.) / cos(3.) over the sample index are linearly
    # independent and near-orthogonal over a full set of samples) -> a near-diagonal covariance.
    patterns = [[math.cos((j + 1) * (i + 0.5) * math.pi / m) for i in range(m)] for j in range(n)]
    x = [scales[j] * patterns[j][i] for i in range(m) for j in range(n)]
    vals, vecs = pca_reference(x, m, n)
    assert len(vals) == n and len(vecs) == n * n
    # eigenvalues descending and well-separated (ratio (3:2:1)^2 = 9:4:1, since each pattern has equal variance).
    assert vals[0] > vals[1] > vals[2] > 0.0, vals
    # the eigenVECTORS are the standard basis (axis-aligned independent features), sign convention -> the
    # leading 1 positive.
    for t in range(n):
        for j in range(n):
            want = 1.0 if j == t else 0.0
            assert abs(abs(vecs[t * n + j]) - want) <= 1e-6, (t, j, vecs)
        # sign convention: the largest-magnitude entry (the diagonal 1) is positive.
        assert vecs[t * n + t] > 0.0, (t, vecs)
    # the eigenvalue ratio matches the variance ratio of the scaled features (9:4:1).
    assert abs(vals[0] / vals[2] - 9.0) <= 1e-4 and abs(vals[1] / vals[2] - 4.0) <= 1e-4, vals


def test_pca_finds_the_known_dominant_direction_of_a_spread_dataset():
    # Points spread along a known (non-axis-aligned) unit direction with small jitter -> the FIRST principal
    # component is ~ that axis and its explained-variance RATIO dominates.
    axis = [0.6, 0.8]                                     # a known unit direction (0.6^2 + 0.8^2 = 1)
    m, n = 40, 2
    x = _axis_dataset(m, axis, spread=5.0, jitter=0.02, seed=0xE2)
    vals, vecs = pca_reference(x, m, n)
    pc0 = [vecs[0], vecs[1]]
    # first PC aligns with `axis` (up to the sign convention -- |cos| ~ 1).
    cos = abs(pc0[0] * axis[0] + pc0[1] * axis[1])
    assert cos > 0.999, (pc0, axis, cos)
    # the dominant variance ratio: lambda0 / (lambda0 + lambda1) ~ 1 (the spread is along one direction).
    ratio = vals[0] / (vals[0] + vals[1])
    assert ratio > 0.99, (vals, ratio)


def test_eigen_and_orthonormality_residuals_are_zero_on_the_reference_output():
    # The genuine INDEPENDENT checks (recomputed directly from C and the eigenpairs, not via the eig path):
    # C v = lambda v for every pair (eigen_residual ~ 0) and the components are orthonormal (V^T V = I).
    m, n = 30, 4
    rng = random.Random(0xE2A)
    # a well-conditioned dataset with distinct feature scales -> distinct (well-separated) eigenvalues.
    scales = [4.0, 3.0, 2.0, 1.0]
    x = [scales[j] * (rng.random() - 0.5) for _ in range(m) for j in range(n)]
    c = covariance_matrix(x, m, n)
    vals, vecs = pca_reference(x, m, n)
    assert eigen_residual(c, vals, vecs, n) <= 1e-9, eigen_residual(c, vals, vecs, n)
    assert orthonormality_residual(vecs, n) <= 1e-9, orthonormality_residual(vecs, n)


def test_eigenvalues_sorted_descending_and_sum_equals_trace():
    # Eigenvalues come back DESCENDING; their sum equals trace(C) (total variance preserved when k == n).
    m, n = 20, 3
    rng = random.Random(0xE2B)
    x = [rng.uniform(-2.0, 2.0) for _ in range(m * n)]
    c = covariance_matrix(x, m, n)
    vals, _ = pca_reference(x, m, n)
    assert all(vals[t] >= vals[t + 1] for t in range(n - 1)), vals      # descending
    assert trace_residual(c, vals, n) <= 1e-9, trace_residual(c, vals, n)  # sum(eigvals) ~ trace(C)


def test_top_k_truncation_keeps_the_largest_k_components():
    # k < n keeps the top-k descending eigenpairs (the largest-variance directions). Independent (orthogonal
    # DCT) per-feature patterns scaled by [4,3,2,1] -> a full-rank covariance with four distinct eigenvalues.
    m, n, k = 16, 4, 2
    scales = [4.0, 3.0, 2.0, 1.0]
    patterns = [[math.cos((j + 1) * (i + 0.5) * math.pi / m) for i in range(m)] for j in range(n)]
    x = [scales[j] * patterns[j][i] for i in range(m) for j in range(n)]
    full_vals, _ = pca_reference(x, m, n)
    vals, vecs = pca_reference(x, m, n, k)
    assert len(vals) == k and len(vecs) == k * n
    assert all(abs(vals[t] - full_vals[t]) <= 1e-9 for t in range(k)), (vals, full_vals)
    assert orthonormality_residual(vecs, n, k) <= 1e-9


def test_sign_convention_is_deterministic_and_largest_magnitude_entry_positive():
    # The sign convention (largest-magnitude entry positive) is applied + reproducible: two runs are identical,
    # and the dominant coordinate of every returned eigenvector is >= 0.
    m, n = 24, 3
    rng = random.Random(0xE2C)
    scales = [3.0, 2.0, 1.0]
    x = [scales[j] * (rng.random() - 0.5) for _ in range(m) for j in range(n)]
    vals1, vecs1 = pca_reference(x, m, n)
    vals2, vecs2 = pca_reference(x, m, n)
    assert vecs1 == vecs2 and vals1 == vals2                # fully deterministic
    for t in range(n):
        comp = [vecs1[t * n + j] for j in range(n)]
        bj = max(range(n), key=lambda j: abs(comp[j]))      # the largest-magnitude entry
        assert comp[bj] > 0.0, (t, comp)


def test_pca_reference_rejects_malformed_inputs():
    # n<1, m<1, bad x length, k<1, k>n -- each the honest dimension failure.
    bad = [
        ([1.0], 1, 0, None),                                 # n < 1
        ([], 0, 2, None),                                    # m < 1
        ([1.0, 2.0, 3.0], 3, 2, None),                       # len(x) != m*n
        ([1.0, 2.0, 3.0, 4.0], 2, 2, 0),                     # k < 1
        ([1.0, 2.0, 3.0, 4.0], 2, 2, 3),                     # k > n
    ]
    for args in bad:
        try:
            pca_reference(*args)
            assert False, f"{args[1:]} should raise"
        except ValueError:
            pass


def test_covariance_rejects_bad_ddof():
    # ddof >= m (a non-positive divisor) and ddof < 0 are the honest failures.
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]                       # m=3, n=2
    for ddof in (3, 4, -1):
        try:
            covariance_matrix(x, 3, 2, ddof=ddof)
            assert False, f"ddof={ddof} should raise"
        except ValueError:
            pass
    # ddof=0 (population) and ddof=1 (sample, default) both succeed and differ by the m/(m-1) factor.
    cpop = covariance_matrix(x, 3, 2, ddof=0)
    csamp = covariance_matrix(x, 3, 2, ddof=1)
    assert all(abs(csamp[i] * (3 - 1) - cpop[i] * 3) <= 1e-9 for i in range(4)), (cpop, csamp)


def test_pca_reference_center_and_covariance_are_side_effect_free():
    # The oracle builds fresh buffers (no in-place overwrite of the caller's data).
    m, n = 6, 3
    rng = random.Random(0xE2D)
    x = [rng.uniform(-1, 1) for _ in range(m * n)]
    x0 = list(x)
    center_columns(x, m, n)
    covariance_matrix(x, m, n)
    pca_reference(x, m, n)
    assert x == x0


# --- the oracle bridge: tracks the reference within the R17 input round-trip bound ------------------------

def test_bridged_pca_tracks_the_reference_within_quant_error():
    # A WELL-CONDITIONED dataset with a WELL-SEPARATED dominant eigenvalue (spread along one axis): PCA forms
    # C = Xc^T Xc / (m-1), which (like OLS's normal equations) squares the input sensitivity, so we bound the
    # per-eigenvalue error by a generous multiple of the R17 input round-trip error (the certified boundary;
    # the eig is trusted/exact on the bridged data). Eigenvectors of a well-separated spectrum are stable too.
    axis = [0.6, 0.8]
    m, n = 50, 2
    x = _axis_dataset(m, axis, spread=6.0, jitter=0.05, seed=0xE2E)
    from bcir.kbcir.quantize import max_abs_error
    q_err = max_abs_error(x, group_size=8, bits=8)            # the R17 input round-trip bound (real units)
    ref_vals, ref_vecs = pca_reference(x, m, n)
    got_vals, got_vecs = pca_via_bridge(x, m, n, None, group_size=8, bits=8)
    amp = 500.0                                               # conservative Xc^T Xc-style amplification
    assert all(abs(r - g) <= amp * q_err + 1e-3 for r, g in zip(ref_vals, got_vals)), (ref_vals, got_vals, q_err)
    # the dominant eigenVECTOR (well-separated -> stable) tracks too (cosine ~ 1).
    cos0 = abs(sum(ref_vecs[j] * got_vecs[j] for j in range(n)))
    assert cos0 > 0.99, (ref_vecs, got_vecs, cos0)


def test_bridge_is_a_clean_roundtrip_then_trusted_pca():
    # pca_via_bridge == pca_reference of the dequantized (round-tripped) input -- the trusted eig sees exactly
    # the bridged values, so the bridge is the only error source (mirrors ols_via_bridge / solve_via_bridge).
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(0xE2F)
    m, n = 18, 3
    x = [rng.uniform(-3, 3) for _ in range(m * n)]
    dx = dequantize(quantize_per_group(x, 8, 8))
    bridged = pca_via_bridge(x, m, n, None, 8, 8)
    direct = pca_reference(dx, m, n)
    assert bridged == direct


# --- the emitted wrapper: LAPACKE_ssyev + portable Jacobi fallback, BCIR owns the layout ------------------

def test_emit_wraps_lapack_ssyev_with_a_jacobi_fallback():
    c = emit_lapack_eigh_c(4, "eigh")
    assert "LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', 4, scratch, 4, val)" in c   # the trusted symmetric eig
    assert "c.call.libm:LAPACKE_ssyev" in c                                       # the LAPACK callee edge label
    assert "BCIR_USE_LAPACK" in c and "BCIR_EIGH_LAPACK" in c                      # link-selected
    assert "JACOBI rotation" in c                                                  # the Jacobi fallback marker
    assert "copysignf(1.0f, theta)" in c                                           # the Jacobi rotation angle
    assert "#include <lapacke.h>" in c                                             # the LAPACKE header on the linked path
    assert "void eigh(const float *C, float *eigvals, float *eigvecs)" in c        # the signature bakes n in
    assert "DESCENDING" in c                                                       # the sort convention
    # n<1 raises (the honest dimension failure).
    for bad in (0, -1):
        try:
            emit_lapack_eigh_c(bad)
            assert False, f"n={bad} should raise"
        except ValueError:
            pass


def _run_eigh_kernel(cc, kernel, c, n, define=None, libs=None):
    """Compile the emitted eigh kernel + a main that calls it on (a copy of) the symmetric C and prints the
    eigenvalues then the eigenvectors (row-major); return (eigvals, eigvecs). The kernel does NOT mutate C."""
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float C[{n * n}] = {{{', '.join(f'{v:.8f}f' for v in c)}}};\n"
            f"  float vals[{n}] = {{0}};\n"
            f"  float vecs[{n * n}] = {{0}};\n"
            f"  eigh(C, vals, vecs);\n"
            f'  for (int i=0;i<{n};++i) printf("%.8f ", vals[i]);\n'
            f'  for (int i=0;i<{n * n};++i) printf("%.8f ", vecs[i]);\n  return 0;\n}}\n')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "e.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "e")
        cmd = [cc, "-std=c11", "-O2"]
        if define:
            cmd.append(define)
        cmd += [src]
        if libs:
            cmd += libs
        cmd += ["-lm", "-o", exe]                              # -lm: sqrtf/fabsf/copysignf in the Jacobi path
        bld = subprocess.run(cmd, capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        nums = [float(t) for t in out.stdout.split()]
        return nums[:n], nums[n:]


def test_fallback_path_compiles_runs_and_matches_the_reference():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return                                                # quick tier hides the toolchain -> self-skip
    # A symmetric covariance with WELL-SEPARATED eigenvalues: independent (orthogonal DCT) per-feature patterns
    # scaled by [3,2,1] -> a full-rank, near-diagonal covariance (distinct eigenvalues 9:4:1).
    m, n = 12, 3
    scales = [3.0, 2.0, 1.0]
    patterns = [[math.cos((j + 1) * (i + 0.5) * math.pi / m) for i in range(m)] for j in range(n)]
    x = [scales[j] * patterns[j][i] for i in range(m) for j in range(n)]
    c = covariance_matrix(x, m, n)
    ref_vals, ref_vecs = pca_reference(x, m, n)
    got_vals, got_vecs = _run_eigh_kernel(cc, emit_lapack_eigh_c(n, "eigh"), c, n)  # NO LAPACK lib
    assert len(got_vals) == n and len(got_vecs) == n * n
    # the C Jacobi fallback IS the same eig (float32 vs the oracle's float64) -> float round-off ...
    assert all(abs(g - r) < 1e-3 for g, r in zip(got_vals, ref_vals)), (got_vals, ref_vals)
    assert all(abs(g - r) < 1e-3 for g, r in zip(got_vecs, ref_vecs)), (got_vecs, ref_vecs)
    # ... and the genuine independent checks: C v = lambda v and V^T V = I on the C output.
    assert eigen_residual(c, got_vals, got_vecs, n) < 1e-2, eigen_residual(c, got_vals, got_vecs, n)
    assert orthonormality_residual(got_vecs, n) < 1e-2, orthonormality_residual(got_vecs, n)


def test_fallback_recovers_a_hand_built_diagonal_spectrum():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return
    # A hand-built DIAGONAL symmetric matrix with distinct entries: the eigenvalues ARE the diagonal
    # (descending) and the eigenvectors are the standard basis. The C fallback must recover them.
    n = 3
    c = [5.0, 0.0, 0.0,
         0.0, 3.0, 0.0,
         0.0, 0.0, 1.0]
    got_vals, got_vecs = _run_eigh_kernel(cc, emit_lapack_eigh_c(n, "eigh"), c, n)
    assert all(abs(g - w) < 1e-4 for g, w in zip(got_vals, [5.0, 3.0, 1.0])), got_vals
    for t in range(n):
        for j in range(n):
            want = 1.0 if j == t else 0.0
            assert abs(abs(got_vecs[t * n + j]) - want) < 1e-4, (t, j, got_vecs)
        assert got_vecs[t * n + t] > 0.0, (t, got_vecs)       # sign convention


def test_linked_lapack_path_agrees_when_lapack_is_present():
    libs = _lapack_link()
    if not libs:
        return                                                # no LAPACK here -> the real-link path self-skips
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    m, n = 12, 3
    scales = [3.0, 2.0, 1.0]
    patterns = [[math.cos((j + 1) * (i + 0.5) * math.pi / m) for i in range(m)] for j in range(n)]
    x = [scales[j] * patterns[j][i] for i in range(m) for j in range(n)]
    c = covariance_matrix(x, m, n)
    ref_vals, ref_vecs = pca_reference(x, m, n)
    got_vals, got_vecs = _run_eigh_kernel(cc, emit_lapack_eigh_c(n, "eigh"), c, n,
                                          define="-DBCIR_USE_LAPACK", libs=libs)
    # ssyev (Householder + QR) and the Jacobi reference find the SAME spectrum on well-separated eigenvalues;
    # agree to float round-off (the honest note: different realization, same decomposition).
    assert all(abs(g - r) < 1e-3 for g, r in zip(got_vals, ref_vals)), (got_vals, ref_vals)
    assert all(abs(g - r) < 1e-2 for g, r in zip(got_vecs, ref_vecs)), (got_vecs, ref_vecs)
    assert eigen_residual(c, got_vals, got_vecs, n) < 1e-2


# --- R17 boundary: the bridge step is the INPUT round-trip alone (the eig is trusted/exact) ---------------

def test_r17_bound_is_the_input_roundtrip_for_the_quantized_pca():
    # The certified boundary error of a quantized PCA call is the bridge's round-trip step (the trusted
    # eigendecomposition itself is exact/trusted -- it contributes no certified error). The R17 grid bound the
    # bridge is held to is 1 ULP, exactly as the OLS/linsolve/BLAS/FFTW wraps.
    assert quantization_error_bound() == 1


# --- the LAPACK link-flag rule: LAPACKE_ssyev -> -llapack (REUSES the existing rule; no regression) -------

def test_pca_link_flag_rule_reuses_the_existing_lapack_rule():
    # LAPACKE_ssyev MATCHES the existing LAPACKE_* rule, so NO linkflags change is needed -- verify it.
    assert library_for_callee("LAPACKE_ssyev") == "-llapack"        # the PCA callee (the wrapper's actual symbol)
    assert library_for_callee("LAPACKE_sgels") == "-llapack"        # the E1 OLS sibling still maps
    assert library_for_callee("LAPACKE_sgesv") == "-llapack"        # the square-solve sibling still maps
    # no regression on the other library rules + libm + libc + unknown.
    assert library_for_callee("gsl_stats_mean") == "-lgsl"          # #62 GSL still maps
    assert library_for_callee("Sleef_expf1_u10") == "-lsleef"       # #63 SLEEF still maps
    assert library_for_callee("fftwf_execute") == "-lfftw3"         # B2 FFTW still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"           # B5 BLAS still maps
    assert library_for_callee("sqrtf") == "-lm"                     # libm still maps
    assert library_for_callee("free") == NO_FLAG                    # libc-implicit, known (not unknown)
    assert library_for_callee("totally_unknown_fn") is None         # unknown-callee policy unchanged


# --- the two-truth quarantine: the PCA module imports no verifier / emits no Diagnostic -------------------

def test_pca_touches_no_verifier_and_emits_no_diagnostic():
    # PCA is a cost-side / oracle quantity (off the legality path), never an R-law legality verdict. Inspect
    # the IMPORTED module object: no verify / Diagnostic / Graded names bound, and no verifier module imported.
    import bcir.kbcir.pca as pca_mod
    bound = set(vars(pca_mod))
    assert not (bound & {"verify_quarantine", "Diagnostic", "Graded", "decide"}), sorted(bound)
    tree = ast.parse(open(pca_mod.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)
    # the returned values are plain floats (no graded/confidence wrapper).
    x = [3.0, 0.0, -3.0, 0.0, 0.0, 2.0, 0.0, -2.0]            # m=4, n=2
    vals, vecs = pca_reference(x, 4, 2)
    assert isinstance(vals, list) and all(isinstance(v, float) for v in vals)
    assert isinstance(vecs, list) and all(isinstance(v, float) for v in vecs)
    c = covariance_matrix(x, 4, 2)
    assert isinstance(eigen_residual(c, vals, vecs, 2), float)
    assert isinstance(orthonormality_residual(vecs, 2), float)


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
