"""B-breadth (#61) -- the dense LINEAR SOLVE oracle reference + the Q8 bridge wrapped around a TRUSTED
external LAPACK ``sgesv`` (solve A x = b via LU with partial pivoting; integrate, don't reinvent). A THIRD,
distinct kernel class after B5's BLAS ``sgemm`` (a matmul) and B2's FFTW (a spectral transform) -- a direct
SOLVER, wrapped through the SAME trusted-external ``c.call.libm:`` FFI edge.

A dense linear solve ``A x = b`` factorizes ``A = P L U`` (LU with partial pivoting) and then forward/back
substitutes. Its REALIZATION (LAPACK's blocked, cache-aware ``sgesv`` vs the naive Gaussian elimination
below) does not change the result -- both solve the identical system, to float round-off -- but LAPACK is
the numerically-hardened, fast kernel on silicon. BCIR owns the CALLING side: the ROW-MAJOR ``A[n*n]`` /
``b[n*nrhs]`` layout, the in-place contract (``sgesv`` overwrites A with its LU factors and b with the
solution x), and the Q8<->float32 boundary; the solve is DELEGATED to LAPACK when linked, with the portable
reference solver below as the standalone fallback (so CI needs no LAPACK installed -- exactly the B5 CBLAS
and B2 FFTW pattern). The emitted C twin is ``lower.c_kernel.emit_lapack_solve_c``.

PARTIAL PIVOTING is numerically necessary, not a flourish: plain Gaussian elimination without pivot
selection divides by a possibly-tiny (or zero) diagonal and is numerically unstable. The reference selects
the largest-magnitude pivot in each column (the same partial-pivoting strategy LAPACK's ``sgesv`` uses), so
the fallback and the trusted kernel agree to float round-off on a well-conditioned system.

Layout (mirrors LAPACKE_sgesv with ``LAPACK_ROW_MAJOR``): ``a`` is the ``n*n`` coefficient matrix row-major
(``a[i*n + j]``); ``b`` is ``n*nrhs`` row-major (``b[i*nrhs + r]`` is row i of right-hand side r). The
result x has the same ``n*nrhs`` row-major layout. The reference returns a FRESH x and does NOT mutate its
inputs (it copies A/b internally), so the oracle is side-effect-free; the in-place overwrite contract is a
property of the EMITTED C kernel (documented there), tested by giving the C kernel scratch copies.

Determinism: the only floats are the solve's own input/output; the bridge step at the boundary is the
R17-certified Q8<->float32<->Q8 round-trip (``kbcir.precision.quantization_error_bound``). The solve itself
is trusted/exact (LAPACK or the reference); the bridge bound is the INPUT round-trip alone.
"""

from __future__ import annotations

from .quantize import dequantize, quantize_per_group


def solve_reference(a: list[float], b: list[float], n: int, nrhs: int = 1) -> list[float]:
    """The naive dense linear solve A x = b via Gaussian elimination with PARTIAL PIVOTING -- the single
    source of truth every realization is checked against, and the EXACT algorithm class LAPACK ``sgesv``
    realizes (LU with partial pivoting). `a` is the `n*n` coefficient matrix row-major (`a[i*n+j]`); `b` is
    the `n*nrhs` right-hand side(s) row-major (`b[i*nrhs+r]`); the result x has the same `n*nrhs` row-major
    layout. Inputs are NOT mutated (working copies are made internally), so the oracle is side-effect-free
    -- the in-place overwrite contract is a property of the wrapped C kernel, not the reference.

    Partial pivoting (swap in the largest-magnitude pivot per column) is numerically necessary: it is what
    makes the elimination stable, and it is exactly what ``sgesv`` does -- so this reference and the trusted
    LAPACK kernel agree to float round-off on a well-conditioned system. Raises if A is singular (a zero
    pivot column), the honest failure of a solve (LAPACK returns ``info > 0`` for the same condition)."""
    if n < 1:
        raise ValueError(f"solve dimension must be >= 1; got {n}")
    if nrhs < 1:
        raise ValueError(f"nrhs must be >= 1; got {nrhs}")
    if len(a) != n * n:
        raise ValueError(f"coefficient matrix must have n*n = {n * n} entries; got {len(a)}")
    if len(b) != n * nrhs:
        raise ValueError(f"right-hand side must have n*nrhs = {n * nrhs} entries; got {len(b)}")

    # Working copies (row-major), so the inputs survive unchanged (side-effect-free oracle).
    m = [[float(a[i * n + j]) for j in range(n)] for i in range(n)]
    x = [[float(b[i * nrhs + r]) for r in range(nrhs)] for i in range(n)]

    for col in range(n):
        # Partial pivot: the row (at or below `col`) with the largest-magnitude entry in this column.
        pivot = col
        best = abs(m[col][col])
        for row in range(col + 1, n):
            if abs(m[row][col]) > best:
                best = abs(m[row][col])
                pivot = row
        if best == 0.0:
            raise ValueError(f"singular matrix: zero pivot in column {col} (no unique solution)")
        if pivot != col:  # swap the pivot row into place (both A and the RHS)
            m[col], m[pivot] = m[pivot], m[col]
            x[col], x[pivot] = x[pivot], x[col]
        # Eliminate column `col` below the pivot.
        inv = 1.0 / m[col][col]
        for row in range(col + 1, n):
            factor = m[row][col] * inv
            if factor != 0.0:
                for j in range(col, n):
                    m[row][j] -= factor * m[col][j]
                for r in range(nrhs):
                    x[row][r] -= factor * x[col][r]

    # Back substitution: solve the upper-triangular system for each right-hand side.
    for row in range(n - 1, -1, -1):
        for r in range(nrhs):
            s = x[row][r]
            for j in range(row + 1, n):
                s -= m[row][j] * x[j][r]
            x[row][r] = s / m[row][row]

    # Flatten back to the n*nrhs row-major result.
    return [x[i][r] for i in range(n) for r in range(nrhs)]


def residual(a: list[float], x: list[float], b: list[float], n: int, nrhs: int = 1) -> float:
    """The max-abs residual ``max|A x - b|`` (real units) -- the independent correctness check a solve must
    satisfy. For a well-conditioned system the solver makes this ~0 (float round-off). Pure: does not depend
    on `solve_reference` (it recomputes A@x directly), so it is a genuine independent verifier."""
    worst = 0.0
    for i in range(n):
        for r in range(nrhs):
            acc = 0.0
            for j in range(n):
                acc += float(a[i * n + j]) * float(x[j * nrhs + r])
            worst = max(worst, abs(acc - float(b[i * nrhs + r])))
    return worst


def solve_via_bridge(
    a: list[float], b: list[float], n: int, nrhs: int, group_size: int, bits: int
) -> list[float]:
    """B-breadth: the Q8<->float32<->Q8 bridge wrapped around a TRUSTED external linear solve (integrate,
    don't reinvent) -- the solve analog of ``matmul.gemm_via_bridge`` / ``fft.fft_via_bridge``. The
    coefficient matrix and right-hand side arrive as per-group quantized storage; the bridge dequantizes
    them to float32, a trusted external solver (LAPACK ``sgesv``, via the ``c.call.libm:`` edge) solves the
    system, and BCIR owns the calling side (the row-major layout + the boundary quantization). So the oracle
    reference for the quantized path is the SOLVE of the *round-tripped* inputs -- the SOLE certified
    boundary error is the INPUT quantization (R17-certified by ``quantization_error_bound``), the solve
    itself being trusted/exact. Mirrors B5/B2 exactly.

    Conditioning note: a linear solve is more conditioning-sensitive than a gemm (the solve amplifies an
    input perturbation by ~the condition number of A), so the bridge bound certified here is the INPUT
    round-trip step alone -- keep the bridged system well-conditioned. The downstream solve is trusted/exact
    and contributes no certified error; this matches the B5/B2 contract where the wrapped kernel is exact."""
    da = dequantize(quantize_per_group(a, group_size, bits))
    db = dequantize(quantize_per_group(b, group_size, bits))
    return solve_reference(da, db, n, nrhs)
