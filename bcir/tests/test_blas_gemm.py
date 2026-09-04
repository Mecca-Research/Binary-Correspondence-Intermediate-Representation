"""B5 -- wrap a trusted BLAS sgemm through the c.call.libm: edge, with the A1.1 Q8 bridge at the
boundary (integrate, don't reinvent). BCIR owns the calling side (row-major layout + quantization); the
kernel is delegated to CBLAS when linked, with a portable reference fallback.

Covers: the bridged gemm tracks the float reference within the quantization error (R17-certified input
bound); the emitted C wraps cblas_sgemm with a row-major reference fallback; the fallback path compiles
and reproduces the reference under clang (no BLAS needed); and -- only if a CBLAS lib is present -- the
linked CBLAS path agrees too. The point: the win is on the calling side, not a reimplemented kernel."""

import os
import random
import shutil
import subprocess
import tempfile

from bcir.kbcir.matmul import gemm_via_bridge, matmul_reference
from bcir.kbcir.precision import accuracy_bound
from bcir.lower.c_kernel import emit_blas_gemm_c
from bcir.model import Claim, Domain, Lane, Opcode, StrideClass


def _cblas_link():
    """A CBLAS lib flag that links, or None (the real-BLAS path then self-skips honestly)."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.c")
        open(src, "w").write(
            "#include <stddef.h>\nvoid cblas_sgemm();\nint main(void){return 0;}\n"
        )
        for lib in ("-lcblas", "-lblas", "-lopenblas"):
            if (
                subprocess.run(
                    [cc, src, lib, "-o", os.path.join(d, "p")], capture_output=True
                ).returncode
                == 0
            ):
                return lib
    return None


# --- the oracle bridge: tracks the float reference within the quantization error -----------------


def test_bridged_gemm_tracks_the_reference_within_quant_error():
    rng = random.Random(0xB5)
    M, N, K = 4, 4, 16
    a = [rng.uniform(-2, 2) for _ in range(M * K)]
    b = [rng.uniform(-2, 2) for _ in range(K * N)]
    ref = matmul_reference(a, b, M, N, K)
    got = gemm_via_bridge(a, b, M, N, K, group_size=K, bits=8)
    assert all(abs(r - g) <= 0.5 * K + 1e-6 for r, g in zip(ref, got))


def test_bridge_is_a_clean_roundtrip_then_trusted_gemm():
    # gemm_via_bridge == reference matmul of the dequantized (round-tripped) inputs -- i.e. the trusted
    # float gemm sees exactly the bridged values, so the bridge is the only error source.
    from bcir.kbcir.quantize import dequantize, quantize_per_group

    rng = random.Random(11)
    M, N, K = 3, 5, 8
    a = [rng.uniform(-3, 3) for _ in range(M * K)]
    b = [rng.uniform(-3, 3) for _ in range(K * N)]
    da = dequantize(quantize_per_group(a, K, 8))
    db = dequantize(quantize_per_group(b, K, 8))
    assert gemm_via_bridge(a, b, M, N, K, K, 8) == matmul_reference(da, db, M, N, K)


# --- the emitted wrapper: cblas call + portable reference fallback, BCIR owns the layout ----------


def test_emit_wraps_cblas_with_a_reference_fallback():
    c = emit_blas_gemm_c(8, 8, 8, "g")
    assert (
        "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans" in c
    )  # the trusted external kernel
    assert "BCIR_USE_CBLAS" in c and "BCIR_GEMM_CBLAS" in c  # link-selected
    assert "s += A[i * 8 + k] * B[k * 8 + j]" in c  # the row-major reference fallback
    for bad in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        try:
            emit_blas_gemm_c(*bad)
            assert False, f"{bad} should raise"
        except ValueError:
            pass


def test_fallback_path_compiles_and_matches_the_reference():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    rng = random.Random(0x5EED)
    M, N, K = 3, 4, 5
    a = [float(rng.randint(-3, 3)) for _ in range(M * K)]  # integer-valued -> float sums exact
    b = [float(rng.randint(-3, 3)) for _ in range(K * N)]
    ref = matmul_reference(a, b, M, N, K)
    kernel = emit_blas_gemm_c(M, N, K, "g")
    main = (
        f"\n#include <stdio.h>\nint main(void){{\n"
        f"  float A[{M * K}] = {{{', '.join(f'{x:.1f}f' for x in a)}}};\n"
        f"  float B[{K * N}] = {{{', '.join(f'{x:.1f}f' for x in b)}}};\n"
        f"  float C[{M * N}];\n  g(A, B, C);\n"
        f'  for (int i=0;i<{M * N};++i) printf("%.1f ", C[i]);\n  return 0;\n}}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "g.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "g")
        bld = subprocess.run(
            [cc, "-std=c11", "-O2", src, "-o", exe], capture_output=True, text=True
        )
        assert bld.returncode == 0, bld.stderr  # the fallback path needs NO BLAS lib
        out = subprocess.run([exe], capture_output=True, text=True)
        got = [float(x) for x in out.stdout.split()]
        assert got == [round(r, 1) for r in ref], (got, ref)


def test_linked_cblas_path_agrees_when_a_blas_is_present():
    lib = _cblas_link()
    if not lib:
        return  # no CBLAS here -> the real-link path self-skips
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    rng = random.Random(0xCB1A5)
    M, N, K = 4, 4, 4
    a = [float(rng.randint(-3, 3)) for _ in range(M * K)]
    b = [float(rng.randint(-3, 3)) for _ in range(K * N)]
    ref = matmul_reference(a, b, M, N, K)
    kernel = emit_blas_gemm_c(M, N, K, "g")
    main = (
        f"\n#include <stdio.h>\nint main(void){{\n"
        f"  float A[{M * K}] = {{{', '.join(f'{x:.1f}f' for x in a)}}};\n"
        f"  float B[{K * N}] = {{{', '.join(f'{x:.1f}f' for x in b)}}};\n"
        f"  float C[{M * N}];\n  g(A, B, C);\n"
        f'  for (int i=0;i<{M * N};++i) printf("%.1f ", C[i]);\n  return 0;\n}}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "g.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "g")
        bld = subprocess.run(
            [cc, "-std=c11", "-O2", "-DBCIR_USE_CBLAS", src, lib, "-o", exe],
            capture_output=True,
            text=True,
        )
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        got = [float(x) for x in out.stdout.split()]
        assert all(abs(g - round(r, 1)) <= 0.05 for g, r in zip(got, ref)), (got, ref)


# --- R17 boundary: the bridge step is certified on the wrapped call -------------------------------


def test_r17_certifies_the_bridge_on_a_quantized_gemm_call():
    # a wrapped gemm that consumes quantized lanes carries the bridge's round-trip step in its bound
    # (the trusted float kernel itself is exact); a dense call does not.
    def call(qbits):
        return Claim(
            id=7000,
            opcode=Opcode.MUL,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1,
            rd=(50, 51),
            wr=(52,),
            op="call.gemm",
            domain=Domain.RAM,
            quantized_bits=qbits,
        )

    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1  # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1  # dense call: the op's own 1 ULP
