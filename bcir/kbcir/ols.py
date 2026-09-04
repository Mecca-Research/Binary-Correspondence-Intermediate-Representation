"""E1 (ML-breadth) -- the OVERDETERMINED ordinary-least-squares (OLS) oracle reference + the Q8 bridge,
wrapped around a TRUSTED external LAPACK least-squares driver via the SAME Area-B "integrate, don't reinvent"
pattern ``linsolve`` (LAPACK ``sgesv``) / BLAS ``sgemm`` / FFTW / GSL / SLEEF already use. OLS GENERALIZES the
square dense solve: given ``A`` (m x n, m >= n) and ``b``, find ``x`` minimizing ``||A x - b||_2`` (the line/
plane of best fit -- linear regression). The square solve (``linsolve``) is the m == n special case; OLS is its
overdetermined generalization, so this REUSES ``linsolve.solve_reference`` for the inner square solve.

Like every Area-B wrap, BCIR owns the CALLING side: the ROW-MAJOR ``A[m*n]`` / ``b[m*nrhs]`` layout and the
Q8<->float32 boundary; the least-squares solve is DELEGATED to LAPACK's QR driver ``sgels`` when linked, with a
portable normal-equations fallback as the standalone path (so CI needs no LAPACK -- exactly the linsolve/BLAS/
FFTW pattern). The emitted C twin is ``lower.c_kernel.emit_lapack_ols_c``.

THE HONEST CONDITIONING NOTE (the source of truth vs the trusted kernel realize OLS differently):
The OLS source of truth here forms the NORMAL EQUATIONS -- ``G = A^T A`` (n x n) and ``c = A^T b`` (n x nrhs),
then solves ``G x = c`` (the unique minimizer of ``||A x - b||_2``, since ``d/dx ||A x - b||^2 = 0`` <=>
``A^T(A x - b) = 0`` <=> ``A^T A x = A^T b``). The normal equations are the textbook OLS reference and reuse the
trusted ``linsolve.solve_reference`` exactly, but they SQUARE the conditioning: ``cond(A^T A) = cond(A)^2``, so
on an ill-conditioned ``A`` they lose roughly twice the digits a QR factorization would. The EMITTED C path
therefore prefers LAPACK's QR-based ``sgels`` (which works on ``A`` directly -- ``cond(A)``, not ``cond(A)^2``);
its portable fallback uses the same normal equations as this reference. Both compute the SAME least-squares
minimizer and AGREE to float round-off on a WELL-CONDITIONED system (which the tests deliberately use). The
honest caveat: on an ill-conditioned ``A`` the QR path is the more accurate; the normal-equations reference/
fallback is the simple, exact-on-well-conditioned twin.

Layout (mirrors LAPACKE_sgels with ``LAPACK_ROW_MAJOR``): ``a`` is the ``m*n`` design matrix row-major
(``a[i*n + j]``); ``b`` is ``m*nrhs`` row-major (``b[i*nrhs + r]`` is row i of right-hand side r). The result x
has the ``n*nrhs`` row-major layout. The reference returns a FRESH x and does NOT mutate its inputs (it copies
into the normal matrix internally), so the oracle is side-effect-free; the in-place/workspace overwrite contract
is a property of the EMITTED C kernel (documented there), tested by giving the C kernel scratch copies.

Determinism: the only floats are the solve's own input/output; the bridge step at the boundary is the
R17-certified Q8<->float32<->Q8 round-trip (``kbcir.precision.quantization_error_bound``). The least-squares
solve itself is trusted/exact (LAPACK or the reference); the bridge bound certified at the boundary is the INPUT
round-trip alone (the solve contributes no certified error), exactly the ``solve_via_bridge`` contract.
"""

from __future__ import annotations

from .linsolve import solve_reference
from .quantize import dequantize, quantize_per_group


def _normal_matrix(a: list[float], m: int, n: int) -> list[float]:
    """``G = A^T A`` (n x n, row-major), recomputed directly from the m x n row-major ``a``. Symmetric
    positive-semidefinite by construction; positive-definite iff A has full column rank n."""
    g = [0.0] * (n * n)
    for p in range(n):
        for q in range(n):
            acc = 0.0
            for i in range(m):
                acc += float(a[i * n + p]) * float(a[i * n + q])
            g[p * n + q] = acc
    return g


def _normal_rhs(a: list[float], b: list[float], m: int, n: int, nrhs: int) -> list[float]:
    """``c = A^T b`` (n x nrhs, row-major), recomputed directly from the row-major ``a`` and ``b``."""
    c = [0.0] * (n * nrhs)
    for p in range(n):
        for r in range(nrhs):
            acc = 0.0
            for i in range(m):
                acc += float(a[i * n + p]) * float(b[i * nrhs + r])
            c[p * nrhs + r] = acc
    return c


def ols_reference(a: list[float], b: list[float], m: int, n: int, nrhs: int = 1) -> list[float]:
    """The OLS source of truth via the NORMAL EQUATIONS: minimize ``||A x - b||_2`` over the overdetermined
    (m >= n) system. Forms ``G = A^T A`` (n x n) and ``c = A^T b`` (n x nrhs), then solves the square normal
    system ``G x = c`` by REUSING ``linsolve.solve_reference`` (the trusted Gaussian-elimination-with-partial-
    pivoting solve). ``a`` is m x n row-major (``a[i*n+j]``); ``b`` is m x nrhs row-major (``b[i*nrhs+r]``); the
    result x has the n x nrhs row-major layout. Inputs are NOT mutated (the normal matrix/rhs are built into
    fresh buffers), so the oracle is side-effect-free.

    Conditioning (honest): the normal equations square ``cond(A)`` (less stable than QR on an ill-conditioned
    A) -- see the module docstring. They are the textbook OLS reference and exact on a well-conditioned A.
    Raises if ``G`` is rank-deficient (a singular normal matrix -- A is column-rank-deficient, no unique
    minimizer), the honest failure of an OLS, exactly as ``solve_reference`` raises on a singular system."""
    if n < 1:
        raise ValueError(f"ols column dimension must be >= 1; got n={n}")
    if m < n:
        raise ValueError(f"ols needs an overdetermined system m >= n; got m={m} < n={n}")
    if nrhs < 1:
        raise ValueError(f"nrhs must be >= 1; got {nrhs}")
    if len(a) != m * n:
        raise ValueError(f"design matrix must have m*n = {m * n} entries; got {len(a)}")
    if len(b) != m * nrhs:
        raise ValueError(f"right-hand side must have m*nrhs = {m * nrhs} entries; got {len(b)}")

    g = _normal_matrix(a, m, n)  # G = A^T A   (n x n)
    c = _normal_rhs(a, b, m, n, nrhs)  # c = A^T b   (n x nrhs)
    # Reuse the trusted square solve -- raises ValueError on a singular G (rank-deficient A): the honest fail.
    return solve_reference(g, c, n, nrhs)


def normal_equation_residual(
    a: list[float], x: list[float], b: list[float], m: int, n: int, nrhs: int = 1
) -> float:
    """The OLS optimality residual ``max|A^T(A x - b)|`` (real units) -- the INDEPENDENT correctness check an
    OLS solution must satisfy. At the least-squares optimum the gradient ``d/dx ||A x - b||^2 = 2 A^T(A x - b)``
    is zero, so ``A^T(A x - b) = 0`` -- this is ~0 even when ``||A x - b|| > 0`` (an inconsistent system: the
    fit is best, not exact). Recomputed DIRECTLY from A, x, b (NOT via ``ols_reference`` / the normal matrix),
    so it is a genuine independent verifier: form the per-row residual ``r = A x - b`` (m x nrhs), then check
    ``A^T r`` (n x nrhs)."""
    # r = A x - b   (m x nrhs), recomputed directly.
    r = [0.0] * (m * nrhs)
    for i in range(m):
        for rr in range(nrhs):
            acc = 0.0
            for j in range(n):
                acc += float(a[i * n + j]) * float(x[j * nrhs + rr])
            r[i * nrhs + rr] = acc - float(b[i * nrhs + rr])
    # max | A^T r |   (n x nrhs), recomputed directly.
    worst = 0.0
    for p in range(n):
        for rr in range(nrhs):
            acc = 0.0
            for i in range(m):
                acc += float(a[i * n + p]) * r[i * nrhs + rr]
            worst = max(worst, abs(acc))
    return worst


def ols_via_bridge(
    a: list[float], b: list[float], m: int, n: int, nrhs: int, group_size: int, bits: int
) -> list[float]:
    """E1: the Q8<->float32<->Q8 bridge wrapped around a TRUSTED external least-squares solve (integrate, don't
    reinvent) -- the OLS analog of ``linsolve.solve_via_bridge``. The design matrix and right-hand side arrive
    as per-group quantized storage; the bridge dequantizes them to float32, a trusted external solver (LAPACK
    ``sgels``, via the ``c.call.libm:`` edge) finds the least-squares x, and BCIR owns the calling side (the
    row-major layout + the boundary quantization). So the oracle reference for the quantized path is the OLS of
    the *round-tripped* inputs -- the SOLE certified boundary error is the INPUT quantization (R17-certified by
    ``quantization_error_bound``), the least-squares solve itself being trusted/exact. Mirrors linsolve exactly.

    Conditioning note: OLS via the normal equations is more conditioning-sensitive than a square solve (it
    amplifies an input perturbation by ~``cond(A)^2``), so the bridge bound certified here is the INPUT
    round-trip step alone -- keep the bridged system WELL-CONDITIONED. The downstream solve is trusted/exact and
    contributes no certified error; this matches the linsolve/BLAS/FFTW contract where the wrapped kernel is
    exact."""
    da = dequantize(quantize_per_group(a, group_size, bits))
    db = dequantize(quantize_per_group(b, group_size, bits))
    return ols_reference(da, db, m, n, nrhs)
