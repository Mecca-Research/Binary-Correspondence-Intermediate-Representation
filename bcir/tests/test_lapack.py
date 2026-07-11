"""B-breadth (#61) -- wrap a trusted LAPACK dense linear solve (sgesv: A x = b via LU with partial pivoting)
through the c.call.libm: edge, with the A1.1 Q8 bridge at the boundary (integrate, don't reinvent -- a
THIRD, distinct kernel class after B5's BLAS sgemm and B2's FFTW: a direct SOLVER, not a matmul or a
spectral transform). BCIR owns the calling side (the row-major layout + the in-place sgesv contract +
quantization); the solve is delegated to LAPACKE_sgesv when linked, with a portable reference solver
(Gaussian elimination with partial pivoting) fallback.

Covers: the oracle solve_reference produces a correct solution (small residual ||Ax-b||) and matches an
independent solver to float round-off, including an nrhs>1 case; the bridged solve tracks the reference
within the quantization error (R17-certified input bound); the emitted C wraps LAPACKE_sgesv with a
partial-pivoting reference fallback; the fallback path compiles + runs + solves correctly (the residual
check, no LAPACK needed -- it IS the same solver, so linked-vs-fallback agree to float round-off); only if
a LAPACK lib is present, the linked path agrees too; and the LAPACK link-flag rule resolves LAPACKE_*/
sgesv_ -> -llapack (with the C twin agreeing in check_runtime.sh). The point: the win is on the calling
side, not a reimplemented solver."""

import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.linsolve import residual, solve_reference, solve_via_bridge
from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
from bcir.lower.c_kernel import emit_lapack_solve_c
from bcir.model import Claim, Domain, Lane, Opcode, StrideClass
from bcir.toolchain import host_link_args


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


def _independent_solve(a, b, n, nrhs):
    """An INDEPENDENT dense solve (its own Gaussian elimination with partial pivoting; no BCIR code, no
    numpy): build the augmented matrix, eliminate, back-substitute. solve_reference must agree with this."""
    aug = [[float(a[i * n + j]) for j in range(n)] + [float(b[i * nrhs + r]) for r in range(nrhs)]
           for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda row: abs(aug[row][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        for row in range(col + 1, n):
            f = aug[row][col] / aug[col][col]
            for j in range(col, n + nrhs):
                aug[row][j] -= f * aug[col][j]
    x = [[0.0] * nrhs for _ in range(n)]
    for row in range(n - 1, -1, -1):
        for r in range(nrhs):
            s = aug[row][n + r]
            for j in range(row + 1, n):
                s -= aug[row][j] * x[j][r]
            x[row][r] = s / aug[row][row]
    return [x[i][r] for i in range(n) for r in range(nrhs)]


# --- the oracle reference: a correct solution (small residual) matching an independent solver ----------

def test_solve_reference_matches_an_independent_solver_and_has_tiny_residual():
    # A well-conditioned non-trivial 4x4 system (diagonally dominant -> well-conditioned).
    a = [10.0, 1.0, -2.0, 0.0,
         1.0, 8.0, 1.0, -1.0,
         -2.0, 1.0, 12.0, 2.0,
         0.0, -1.0, 2.0, 9.0]
    b = [7.0, -3.0, 14.0, 5.0]
    n, nrhs = 4, 1
    x = solve_reference(a, b, n, nrhs)
    want = _independent_solve(a, b, n, nrhs)             # independent of BCIR's code
    assert all(abs(x[i] - want[i]) <= 1e-9 for i in range(n)), (x, want)
    assert residual(a, x, b, n, nrhs) <= 1e-9            # A x ~ b (the genuine independent check)


def test_solve_reference_supports_multiple_right_hand_sides():
    # n>=3 with nrhs>1: solve A X = B for two right-hand sides at once (the sgesv nrhs feature).
    a = [4.0, 1.0, 0.0,
         1.0, 5.0, 2.0,
         0.0, 2.0, 6.0]
    b = [6.0, 1.0,                                        # row 0: rhs0=6, rhs1=1
         12.0, 0.0,                                       # row 1
         8.0, 3.0]                                        # row 2
    n, nrhs = 3, 2
    x = solve_reference(a, b, n, nrhs)
    want = _independent_solve(a, b, n, nrhs)
    assert all(abs(x[i] - want[i]) <= 1e-9 for i in range(n * nrhs)), (x, want)
    assert residual(a, x, b, n, nrhs) <= 1e-9


def test_solve_reference_needs_pivoting_to_stay_stable():
    # A system whose top-left pivot is zero: naive (no-pivot) elimination would divide by zero. Partial
    # pivoting swaps in a nonzero pivot, so the reference solves it cleanly. x = [1, 2] for this system.
    a = [0.0, 1.0,
         1.0, 1.0]
    b = [2.0, 3.0]
    x = solve_reference(a, b, 2, 1)
    assert abs(x[0] - 1.0) <= 1e-9 and abs(x[1] - 2.0) <= 1e-9, x
    assert residual(a, x, b, 2, 1) <= 1e-9


def test_solve_reference_rejects_a_singular_matrix():
    # Two identical rows -> rank-deficient, no unique solution. The honest failure (LAPACK info>0).
    try:
        solve_reference([1.0, 2.0, 1.0, 2.0], [3.0, 3.0], 2, 1)
        assert False, "singular matrix should raise"
    except ValueError:
        pass


def test_solve_reference_rejects_malformed_inputs():
    for bad in [([1.0], [1.0], 0, 1), ([1.0, 2.0, 3.0], [1.0], 2, 1), ([1.0], [1.0, 2.0], 1, 1)]:
        try:
            solve_reference(*bad); assert False, f"{bad} should raise"
        except ValueError:
            pass
    try:
        solve_reference([1.0], [1.0], 1, 0); assert False, "nrhs<1 should raise"
    except ValueError:
        pass


def test_solve_reference_is_side_effect_free():
    # The oracle copies its inputs internally (the in-place overwrite is the C kernel's contract, not the
    # reference's), so A and b survive unchanged after a solve.
    a = [3.0, 1.0, 1.0, 2.0]
    b = [9.0, 8.0]
    a0, b0 = list(a), list(b)
    solve_reference(a, b, 2, 1)
    assert a == a0 and b == b0


# --- the oracle bridge: tracks the reference within the quantization error -----------------------------

def test_bridged_solve_tracks_the_reference_within_quant_error():
    # A WELL-CONDITIONED system (diagonally dominant): a solve amplifies the input round-trip by ~cond(A),
    # so we keep cond(A) small and bound the per-output error generously by it. The bridge is the ONLY error
    # source (the solve is trusted/exact on the round-tripped inputs).
    rng = random.Random(0x61)
    n, nrhs = 4, 1
    a = []
    for i in range(n):
        for j in range(n):
            a.append((20.0 if i == j else rng.uniform(-1, 1)))   # strongly diagonally dominant
    b = [rng.uniform(-5, 5) for _ in range(n * nrhs)]
    from bcir.kbcir.quantize import max_abs_error
    q_err = max_abs_error(a + b, group_size=8, bits=8)            # the R17 input round-trip bound (real units)
    ref = solve_reference(a, b, n, nrhs)
    got = solve_via_bridge(a, b, n, nrhs, group_size=8, bits=8)
    # Conservative amplification: for a diagonally dominant A the solution sensitivity is modest; bound the
    # per-output deviation by a generous multiple of the input round-trip error (the certified boundary).
    amp = 50.0
    assert all(abs(r - g) <= amp * q_err + 1e-6 for r, g in zip(ref, got)), (ref, got, q_err)


def test_bridge_is_a_clean_roundtrip_then_trusted_solve():
    # solve_via_bridge == reference solve of the dequantized (round-tripped) inputs -- i.e. the trusted
    # solver sees exactly the bridged values, so the bridge is the only error source (mirrors B5/B2).
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(173)
    n, nrhs = 3, 1
    a = [(15.0 if i // n == i % n else rng.uniform(-1, 1)) for i in range(n * n)]
    b = [rng.uniform(-3, 3) for _ in range(n * nrhs)]
    da = dequantize(quantize_per_group(a, 8, 8))
    db = dequantize(quantize_per_group(b, 8, 8))
    assert solve_via_bridge(a, b, n, nrhs, 8, 8) == solve_reference(da, db, n, nrhs)


# --- the emitted wrapper: LAPACKE_sgesv + portable reference fallback, BCIR owns the layout ------------

def test_emit_wraps_lapack_with_a_reference_fallback():
    c = emit_lapack_solve_c(4, 1, "solve")
    assert "LAPACKE_sgesv(LAPACK_ROW_MAJOR, 4, 1, A, 4, ipiv, b, 1)" in c   # the trusted external kernel
    assert "BCIR_USE_LAPACK" in c and "BCIR_SOLVE_LAPACK" in c              # link-selected
    assert "float factor = A[row * N + col] * inv;" in c                   # the partial-pivot elimination fallback
    assert "if (v > best) { best = v; pivot = row; }" in c                 # partial pivoting (numerically necessary)
    assert "#include <lapacke.h>" in c                                     # the LAPACKE header on the linked path
    for bad in [(0, 1), (1, 0)]:
        try:
            emit_lapack_solve_c(*bad); assert False, f"{bad} should raise"
        except ValueError:
            pass


def _run_solve_kernel(cc, kernel, a, b, n, nrhs, define=None, libs=None):
    """Compile the emitted solver + a main that calls it on (a copy of) A,b and prints x; return the
    parsed solution list. The kernel overwrites b in place with x (the sgesv contract)."""
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float A[{n * n}] = {{{', '.join(f'{v:.8f}f' for v in a)}}};\n"
            f"  float b[{n * nrhs}] = {{{', '.join(f'{v:.8f}f' for v in b)}}};\n"
            f"  int info = solve(A, b);\n"
            f'  if (info != 0) {{ printf("INFO %d\\n", info); return 2; }}\n'
            f'  for (int i=0;i<{n * nrhs};++i) printf("%.8f ", b[i]);\n  return 0;\n}}\n')
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
        cmd += ["-lm", "-o", exe]                          # -lm: fabsf in the fallback path
        bld = subprocess.run(host_link_args(cmd), capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return [float(t) for t in out.stdout.split()]


def test_fallback_path_compiles_runs_and_solves_correctly():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return                                              # quick tier hides the toolchain -> self-skip
    a = [10.0, 2.0, -1.0,
         -3.0, 9.0, 1.0,
         1.0, -2.0, 8.0]
    b = [11.0, 7.0, 7.0]
    n, nrhs = 3, 1
    ref = solve_reference(a, b, n, nrhs)
    got = _run_solve_kernel(cc, emit_lapack_solve_c(n, nrhs, "solve"), a, b, n, nrhs)  # NO LAPACK lib
    assert len(got) == n * nrhs
    # the C fallback IS the same solver (float32 vs the oracle's float64) -> agree to float round-off ...
    assert all(abs(g - r) < 1e-3 for g, r in zip(got, ref)), (got, ref)
    # ... and the genuine independent check: the residual ||A x - b|| is ~0.
    assert residual(a, got, b, n, nrhs) < 1e-3, residual(a, got, b, n, nrhs)


def test_fallback_path_solves_multiple_right_hand_sides():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return
    a = [6.0, 1.0, 2.0,
         1.0, 7.0, 1.0,
         2.0, 1.0, 9.0]
    b = [10.0, 1.0,
         9.0, 2.0,
         12.0, 3.0]
    n, nrhs = 3, 2
    ref = solve_reference(a, b, n, nrhs)
    got = _run_solve_kernel(cc, emit_lapack_solve_c(n, nrhs, "solve"), a, b, n, nrhs)
    assert all(abs(g - r) < 1e-3 for g, r in zip(got, ref)), (got, ref)
    assert residual(a, got, b, n, nrhs) < 1e-3


def test_linked_lapack_path_agrees_when_lapack_is_present():
    libs = _lapack_link()
    if not libs:
        return                                              # no LAPACK here -> the real-link path self-skips
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    a = [10.0, 2.0, -1.0,
         -3.0, 9.0, 1.0,
         1.0, -2.0, 8.0]
    b = [11.0, 7.0, 7.0]
    n, nrhs = 3, 1
    ref = solve_reference(a, b, n, nrhs)
    got = _run_solve_kernel(cc, emit_lapack_solve_c(n, nrhs, "solve"), a, b, n, nrhs,
                            define="-DBCIR_USE_LAPACK", libs=libs)
    # LAPACK solves the identical system (LU + partial pivoting); agree to float round-off.
    assert all(abs(g - r) < 1e-2 for g, r in zip(got, ref)), (got, ref)
    assert residual(a, got, b, n, nrhs) < 1e-2


# --- R17 boundary: the bridge step is certified on the wrapped call ------------------------------------

def test_r17_certifies_the_bridge_on_a_quantized_solve_call():
    # a wrapped solve that consumes quantized lanes carries the bridge's round-trip step in its bound (the
    # trusted solver itself is exact/trusted); a dense call does not. Mirrors test_blas_gemm / test_fftw.
    def call(qbits):
        return Claim(id=6100, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT, count=1,
                     rd=(60, 61), wr=(62,), op="call.solve", domain=Domain.RAM, quantized_bits=qbits)
    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1     # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1                              # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1                          # the R17 grid bound the bridge is held to


# --- the LAPACK link-flag rule: a LAPACK edge resolves to -llapack (the C twin agrees in check_runtime.sh) --

def test_lapack_link_flag_rule():
    assert library_for_callee("LAPACKE_sgesv") == "-llapack"        # the rule (the wrapper's actual callee)
    assert library_for_callee("LAPACKE_dgesv") == "-llapack"        # any LAPACKE_*
    assert library_for_callee("sgesv_") == "-llapack"               # the Fortran-ABI driver symbol
    assert library_for_callee("dgetrf_") == "-llapack"              # another Fortran LU/solve driver
    # no regression on the existing classifications.
    assert library_for_callee("fftwf_execute") == "-lfftw3"         # B2 FFTW still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"           # B5 BLAS still maps
    assert library_for_callee("sqrtf") == "-lm"                     # libm still maps
    assert library_for_callee("free") == NO_FLAG                    # libc-implicit, known (not unknown)
    assert library_for_callee("totally_unknown_fn") is None         # unknown-callee policy unchanged
    assert library_for_callee("foo_") is None                       # a non-LAPACK trailing-underscore symbol
