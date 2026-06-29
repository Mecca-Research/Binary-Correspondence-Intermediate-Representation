"""E2 (ML-breadth) -- PRINCIPAL COMPONENT ANALYSIS (PCA): the eigendecomposition oracle reference + the Q8
bridge, the SECOND slice of the ML-breadth ladder after E1 (OLS). PCA GENERALIZES the OLS shape "form a
symmetric matrix then SOLVE it" into "form a symmetric matrix then EIGENDECOMPOSE it": OLS builds the normal
matrix ``G = A^T A`` and solves ``G x = c``; PCA builds the (centered) sample covariance ``C = (1/(m-ddof))
Xc^T Xc`` and eigendecomposes the symmetric ``C`` for the principal directions (eigenvectors) and their
explained variances (eigenvalues). Both are the SAME "symmetric Gram matrix from a data matrix" object; OLS
asks for a solve, PCA asks for the spectrum.

THE NET-NEW REFERENCE (why this is not a reuse like E1's ``linsolve.solve_reference``): E1 REUSED the trusted
square Gaussian-elimination solve because one already existed. There is NO eigen/SVD/Jacobi solver anywhere in
the repo to reuse, so the symmetric eigendecomposition reference IS net-new here. It uses the classic JACOBI
ROTATION algorithm for real symmetric matrices: a deterministic cyclic sweep of plane rotations that drives the
off-diagonal entries to zero, accumulating the rotations into the eigenvector matrix. Jacobi is the natural
"source of truth" eig: it is simple, fully deterministic (a fixed sweep order, no randomness, no pivoting
heuristic), and converges to full machine precision on a symmetric matrix. It is the trusted-eig analog of the
trusted Gaussian-elimination solve E1 leaned on -- except implemented here because nothing existed.

THE HONEST NOTE (the source of truth vs the trusted kernel realize the eig DIFFERENTLY): the emitted C twin
(``lower.c_kernel.emit_lapack_eigh_c``) prefers LAPACK's ``ssyev`` (Householder tridiagonalization + implicit-QR
/ divide-and-conquer) when linked; this Python reference and the C fallback use Jacobi. The two methods differ
in REALIZATION but compute the SAME spectral decomposition and AGREE to float round-off on a symmetric matrix
with WELL-SEPARATED eigenvalues. The honest caveat: a DEGENERATE (repeated) eigenvalue makes the eigenVECTORS
non-unique -- any orthonormal basis of the eigenspace is a valid answer, so Jacobi and ssyev can legitimately
return DIFFERENT bases there (not a bug). The tests therefore use DISTINCT, well-separated eigenvalues, exactly
as E1 used a well-conditioned A.

DETERMINISM CONVENTIONS (so the reference and the C path are byte-comparable):
  * EIGENVALUE ORDERING -- the eigenpairs are sorted by eigenvalue DESCENDING (the PCA convention: the first
    principal component has the largest variance), carrying each eigenvector along with its eigenvalue.
  * EIGENVECTOR SIGN -- an eigenvector is defined only up to sign (v and -v are both unit eigenvectors). A
    deterministic convention fixes it: the entry of LARGEST MAGNITUDE in each eigenvector is forced positive
    (ties broken by the lowest index). The SAME convention is applied in the emitted C, so the two agree.

Layout (row-major throughout): ``x`` is the m x n data matrix (m samples, n features; ``x[i*n + j]`` is sample
i, feature j). ``center_columns`` subtracts the per-feature mean; ``covariance_matrix`` returns the n x n
symmetric ``C`` row-major (``C[p*n + q]``); ``pca_reference`` returns ``(eigvals, eigvecs)`` where ``eigvals``
is the length-k list of explained variances (descending) and ``eigvecs`` is a flat ``k*n`` row-major list of
the k principal directions (``eigvecs[t*n + j]`` is component t's j-th coordinate). All functions return FRESH
buffers and do NOT mutate their inputs (side-effect-free oracle).

Determinism: the only floats are the analysis's own input/output; the bridge step at the boundary is the
R17-certified Q8<->float32<->Q8 round-trip (``kbcir.precision.quantization_error_bound``). The
eigendecomposition itself is trusted/exact (LAPACK or the Jacobi reference); the bridge bound certified at the
boundary is the INPUT round-trip alone (the eig contributes no certified error), exactly the ``ols_via_bridge``
/ ``solve_via_bridge`` contract. This module is OFF the legality path: it imports no verifier, emits no
Diagnostic, and returns plain floats (no graded/confidence wrapper).
"""

from __future__ import annotations

import math

from .quantize import dequantize, quantize_per_group

# Jacobi sweep controls (deterministic; no randomness). A symmetric n x n matrix converges in O(log n) cyclic
# sweeps; the cap is generous and the off-diagonal threshold is the real stopping rule. Both are fixed so the
# reference is fully reproducible.
_JACOBI_MAX_SWEEPS = 100
_JACOBI_OFFDIAG_EPS = 1e-300        # stop when the off-diagonal Frobenius norm is below this (full precision)


def _offdiag_norm(a: list[list[float]], n: int) -> float:
    """The Frobenius norm of the strictly-upper off-diagonal of the symmetric working matrix ``a`` (the Jacobi
    convergence measure -- it decreases monotonically as rotations zero out off-diagonal mass)."""
    s = 0.0
    for p in range(n):
        for q in range(p + 1, n):
            s += a[p][q] * a[p][q]
    return math.sqrt(2.0 * s)        # symmetric -> the lower triangle mirrors the upper


def _jacobi_eigh(c: list[float], n: int,
                 max_sweeps: int = _JACOBI_MAX_SWEEPS,
                 offdiag_eps: float = _JACOBI_OFFDIAG_EPS):
    """Symmetric eigendecomposition by the classic JACOBI ROTATION algorithm -- the deterministic source of
    truth for the spectrum of the real symmetric ``c`` (n x n, row-major). Returns ``(eigvals, eigvecs)``: a
    length-n list of eigenvalues and an n x n list-of-rows ``eigvecs`` whose ROW t is the unit eigenvector for
    ``eigvals[t]`` (i.e. ``eigvecs[t][j]`` is coordinate j of eigenvector t). NOT sorted and NOT sign-fixed
    here -- ``pca_reference`` applies the descending sort + sign convention.

    The algorithm: maintain a working symmetric matrix ``a`` (a copy of ``c``) and an accumulated rotation
    matrix ``v`` (initially the identity). In a fixed CYCLIC sweep over every off-diagonal pair (p, q) with
    p < q, apply the Givens/Jacobi plane rotation that ZEROES ``a[p][q]`` (and ``a[q][p]``): the rotation angle
    is ``theta = (a[q][q] - a[p][p]) / (2 a[p][q])`` and ``t = sign(theta)/(|theta| + sqrt(theta^2 + 1))`` (the
    smaller root, for numerical stability), ``c_ = 1/sqrt(t^2+1)``, ``s = t c_``. The rotation is applied to
    both ``a`` (a similarity transform A <- J^T A J, so the spectrum is preserved exactly) and ``v`` (V <- V J,
    accumulating the eigenvectors). Sweeps repeat until the off-diagonal norm is below ``offdiag_eps`` or the
    sweep cap is hit. On convergence the diagonal of ``a`` holds the eigenvalues and the columns of ``v`` hold
    the eigenvectors; this returns ``eigvecs[t]`` = column t of ``v`` (the t-th eigenvector as a row).

    Deterministic: the cyclic sweep order, the fixed cap, and the threshold are all fixed -- no randomness, no
    data-dependent ordering. Converges to full machine precision on a symmetric matrix (the off-diagonal mass
    falls quadratically once the rotations localize)."""
    # Working copies (list-of-rows), so the input ``c`` survives unchanged (side-effect-free).
    a = [[float(c[p * n + q]) for q in range(n)] for p in range(n)]
    v = [[1.0 if p == q else 0.0 for q in range(n)] for p in range(n)]   # accumulated rotations (identity)

    for _ in range(max_sweeps):
        if _offdiag_norm(a, n) <= offdiag_eps:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if apq == 0.0:
                    continue                                   # already zero -> no rotation needed
                # The rotation that annihilates a[p][q] (the smaller-root form for stability).
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                tval = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                cc = 1.0 / math.sqrt(tval * tval + 1.0)        # cos
                ss = tval * cc                                 # sin
                # Apply the similarity transform A <- J^T A J (rows/cols p,q only), preserving symmetry.
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = cc * akp - ss * akq
                    a[k][q] = ss * akp + cc * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = cc * apk - ss * aqk
                    a[q][k] = ss * apk + cc * aqk
                a[p][q] = 0.0                                  # force the annihilated entries exactly to zero
                a[q][p] = 0.0
                # Accumulate the rotation into the eigenvector matrix V <- V J.
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = cc * vkp - ss * vkq
                    v[k][q] = ss * vkp + cc * vkq

    eigvals = [a[i][i] for i in range(n)]                      # the diagonal holds the eigenvalues
    eigvecs = [[v[r][t] for r in range(n)] for t in range(n)]  # eigvecs[t] = column t of V (eigenvector t)
    return eigvals, eigvecs


def _column_means(x: list[float], m: int, n: int) -> list[float]:
    """Per-feature (per-column) mean of the m x n row-major data ``x`` (length-n list)."""
    means = [0.0] * n
    for i in range(m):
        for j in range(n):
            means[j] += float(x[i * n + j])
    return [mu / m for mu in means]


def center_columns(x: list[float], m: int, n: int) -> list[float]:
    """Center each FEATURE (column) of the m x n row-major data ``x``: subtract the per-column mean from every
    entry, returning a FRESH centered ``Xc`` (same m x n row-major layout). Side-effect-free -- the input ``x``
    is not mutated. Centering is the first PCA step: the covariance is about variation around the mean, so each
    feature is shifted to zero mean before the covariance is formed. Validates dims (n>=1, m>=1, len(x)==m*n)."""
    if n < 1:
        raise ValueError(f"pca feature dimension must be >= 1; got n={n}")
    if m < 1:
        raise ValueError(f"pca needs at least one sample m >= 1; got m={m}")
    if len(x) != m * n:
        raise ValueError(f"data matrix must have m*n = {m * n} entries; got {len(x)}")
    means = _column_means(x, m, n)
    return [float(x[i * n + j]) - means[j] for i in range(m) for j in range(n)]


def covariance_matrix(x: list[float], m: int, n: int, ddof: int = 1) -> list[float]:
    """The n x n SYMMETRIC sample covariance ``C = (1/(m-ddof)) Xc^T Xc`` (row-major, ``C[p*n + q]``),
    recomputed directly from the centered data. REUSES ``center_columns`` for the centering step. ``ddof`` is
    the delta degrees of freedom: ``ddof=1`` (the default) is the UNBIASED sample covariance (the ``m-1``
    divisor, matching ``gsl_stats_variance`` / numpy ``np.cov`` defaults); ``ddof=0`` is the population
    covariance (the ``m`` divisor). Symmetric positive-semidefinite by construction. Side-effect-free (builds a
    fresh buffer from a fresh centered copy). Validates dims and ``ddof < m`` (a non-positive divisor is the
    honest failure -- too few samples for the requested degrees of freedom)."""
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0; got {ddof}")
    if ddof >= m:
        raise ValueError(f"covariance needs ddof < m (a positive divisor m-ddof); got ddof={ddof}, m={m}")
    xc = center_columns(x, m, n)                               # validates n>=1, m>=1, len(x)==m*n
    denom = float(m - ddof)
    cmat = [0.0] * (n * n)
    for p in range(n):
        for q in range(n):
            acc = 0.0
            for i in range(m):
                acc += xc[i * n + p] * xc[i * n + q]
            cmat[p * n + q] = acc / denom
    return cmat


def _apply_sign_convention(vec: list[float], n: int) -> list[float]:
    """Force the entry of LARGEST MAGNITUDE positive (ties broken by lowest index) -- the deterministic
    eigenvector sign convention. v and -v are both unit eigenvectors; this picks one reproducibly so the
    reference and the emitted C path agree. Returns a fresh, possibly sign-flipped, copy."""
    best_idx = 0
    best_mag = abs(vec[0])
    for j in range(1, n):
        if abs(vec[j]) > best_mag:
            best_mag = abs(vec[j])
            best_idx = j
    flip = vec[best_idx] < 0.0
    return [-v for v in vec] if flip else list(vec)


def pca_reference(x: list[float], m: int, n: int, k: int | None = None):
    """The PCA source of truth: center -> covariance -> symmetric Jacobi eig -> sort eigenpairs by eigenvalue
    DESCENDING -> apply the sign convention -> return the top-k principal components and explained variances.
    ``x`` is the m x n row-major data (m samples, n features); ``k`` is the number of components to keep
    (default ``n`` -- all of them). Returns ``(eigvals, eigvecs)``: ``eigvals`` is the length-k list of
    explained variances (descending), ``eigvecs`` is the flat ``k*n`` row-major list of components
    (``eigvecs[t*n + j]`` is component t's j-th coordinate). Side-effect-free.

    The pipeline mirrors ``ols_reference`` exactly in spirit: there OLS built ``G = A^T A`` and REUSED the
    trusted square solve; here PCA builds the symmetric covariance ``C`` and feeds it to the trusted Jacobi eig
    (``_jacobi_eigh``). The descending sort is the PCA convention (largest variance first = the first principal
    component); the sign convention (largest-magnitude entry positive) makes each returned eigenvector
    reproducible and comparable to the C path.

    Conditioning / degeneracy (honest): Jacobi converges to full precision on a symmetric matrix, and on
    WELL-SEPARATED eigenvalues the eigenvectors are unique up to sign (which the sign convention fixes). A
    repeated eigenvalue makes the eigenVECTORS non-unique (any orthonormal basis of the eigenspace is valid),
    so two correct eigensolvers can return different bases there -- callers comparing eigenvectors should use
    distinct eigenvalues. Raises on bad dims (n<1, m<1, len mismatch, k<1, k>n) -- the honest failures."""
    if n < 1:
        raise ValueError(f"pca feature dimension must be >= 1; got n={n}")
    if m < 1:
        raise ValueError(f"pca needs at least one sample m >= 1; got m={m}")
    if len(x) != m * n:
        raise ValueError(f"data matrix must have m*n = {m * n} entries; got {len(x)}")
    if k is None:
        k = n
    if k < 1:
        raise ValueError(f"number of components k must be >= 1; got {k}")
    if k > n:
        raise ValueError(f"cannot keep k={k} components of an n={n}-feature space (k > n)")

    cmat = covariance_matrix(x, m, n, ddof=1)                  # the symmetric covariance (ddof=1 default)
    vals, vecs = _jacobi_eigh(cmat, n)                         # the trusted symmetric eig (Jacobi)
    # Sort eigenpairs by eigenvalue DESCENDING (PCA convention), carrying eigenvectors along; ties broken by
    # the original index so the order is fully deterministic.
    order = sorted(range(n), key=lambda t: (-vals[t], t))
    out_vals: list[float] = []
    out_vecs: list[float] = []
    for t in order[:k]:
        out_vals.append(vals[t])
        out_vecs.extend(_apply_sign_convention(vecs[t], n))    # deterministic sign per component
    return out_vals, out_vecs


def eigen_residual(c: list[float], eigvals: list[float], eigvecs: list[float],
                   n: int, k: int | None = None) -> float:
    """The defining-eigen-equation residual ``max |C v - lambda v|`` over the returned (lambda, v) pairs (real
    units) -- the INDEPENDENT correctness check a symmetric eigendecomposition must satisfy. At a correct
    decomposition ``C v = lambda v`` for every eigenpair, so this is ~0. Recomputed DIRECTLY from ``C`` and the
    eigenpairs (NOT via ``_jacobi_eigh`` / the eig code path), so it is a genuine independent verifier --
    exactly the role ``normal_equation_residual`` plays for OLS. ``c`` is the n x n symmetric matrix row-major;
    ``eigvecs`` is the flat ``k*n`` row-major component list, ``eigvals`` the matching length-k values."""
    if k is None:
        k = len(eigvals)
    worst = 0.0
    for t in range(k):
        lam = float(eigvals[t])
        for p in range(n):
            acc = 0.0                                          # (C v)_p, recomputed directly
            for q in range(n):
                acc += float(c[p * n + q]) * float(eigvecs[t * n + q])
            worst = max(worst, abs(acc - lam * float(eigvecs[t * n + p])))
    return worst


def orthonormality_residual(eigvecs: list[float], n: int, k: int | None = None) -> float:
    """The orthonormality residual ``max |V^T V - I|`` over the k returned components (dimensionless) -- the
    second INDEPENDENT check: the principal components are ORTHONORMAL (mutually orthogonal unit vectors), a
    property of the eigenvectors of a symmetric matrix. So every pairwise dot product is ~0 and every
    self-dot ~1; this is ~0 at a correct decomposition. Recomputed DIRECTLY from the returned eigenvectors
    (independent of the eig code path). ``eigvecs`` is the flat ``k*n`` row-major component list."""
    if k is None:
        k = len(eigvecs) // n if n else 0
    worst = 0.0
    for s in range(k):
        for t in range(k):
            dot = 0.0
            for j in range(n):
                dot += float(eigvecs[s * n + j]) * float(eigvecs[t * n + j])
            expected = 1.0 if s == t else 0.0
            worst = max(worst, abs(dot - expected))
    return worst


def trace_residual(c: list[float], eigvals: list[float], n: int) -> float:
    """Total-variance check: ``|sum(eigvals) - trace(C)|`` (real units). The trace of a matrix equals the sum
    of its eigenvalues, so when k == n (all components kept) the sum of explained variances equals the total
    variance ``trace(C) = sum_p C[p][p]`` -- variance is preserved by the decomposition. ~0 at a correct full
    decomposition. (With k < n it is the variance DROPPED by truncation, not ~0 -- so this helper is meaningful
    as a check only when all n eigenvalues are passed.)"""
    tr = sum(float(c[p * n + p]) for p in range(n))
    return abs(sum(float(v) for v in eigvals) - tr)


def pca_via_bridge(x: list[float], m: int, n: int, k: int | None,
                   group_size: int, bits: int):
    """E2: the Q8<->float32<->Q8 bridge wrapped around a TRUSTED eigendecomposition (integrate, don't
    reinvent) -- the PCA analog of ``ols_via_bridge`` / ``solve_via_bridge``. The DATA matrix arrives as
    per-group quantized storage; the bridge dequantizes it to float32, then ``pca_reference`` centers, forms the
    covariance, and runs the trusted Jacobi eig (the emitted C delegates to LAPACK ``ssyev`` via the
    ``c.call.libm:`` edge when linked). BCIR owns the calling side (the row-major layout + the boundary
    quantization). So the oracle reference for the quantized path is the PCA of the *round-tripped* input data
    -- the SOLE certified boundary error is the INPUT quantization (R17-certified by
    ``quantization_error_bound``), the eigendecomposition itself being trusted/exact. Mirrors OLS exactly.

    Conditioning note: PCA forms ``C = Xc^T Xc / (m-ddof)``, which (like OLS's normal equations) squares the
    sensitivity of the centered data, so the bridge bound certified here is the INPUT round-trip step alone --
    keep the bridged data WELL-CONDITIONED with WELL-SEPARATED eigenvalues (a near-degenerate spectrum makes the
    eigenVECTORS ill-determined, independent of the bridge). The downstream eig is trusted/exact and contributes
    no certified error; this matches the OLS/linsolve/BLAS/FFTW contract where the wrapped kernel is exact."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return pca_reference(dx, m, n, k)
