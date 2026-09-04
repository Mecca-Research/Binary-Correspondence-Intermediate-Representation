"""B3 -- calling-side tuning for a WRAPPED kernel, priced by the EXISTING K_BCIR cost model.

The roadmap thesis for the wrapped-library work (B5 BLAS gemm, B2 FFTW fft): BCIR's value is optimizing the
CALLING side (layout, prefetch, tiling, channel selection) around a TRUSTED kernel it does NOT reimplement.
B5/B2 hard-code those params (``emit_blas_gemm_c`` always emits ``CblasRowMajor``, no prefetch, no channel).
B3 makes them a COST-MODEL-PRICED PLAN: BCIR chooses {major-order, caller block, prefetch distance, channel}
through the SAME cost machinery, records the choice in a ``CallingSideCertificate``, and -- THE GATE THAT
MATTERS -- the tuned call computes the IDENTICAL numeric result as the untuned wrapped call.

Covers:
  * the cost model SELECTS each calling-side param (block from plan_matmul; major-order from cost_of's
    traffic terms; prefetch from the _PREFETCH_FACTOR memory discount; channel from optimize-on-profile),
    each priced through the EXISTING model, not a bespoke discount;
  * the certificate records the chosen plan + cost before/after + the modeled win, with a clean NO-OP when
    tuning is unprofitable (a small square on a host-only tower);
  * REFERENCE-INVARIANCE (the load-bearing gate): every tuned emit -- row-major, column-major, and
    prefetch -- computes the IDENTICAL C as the untuned ``emit_blas_gemm_c`` reference (compile+run, bit-
    exact for integer-valued inputs); the column-major C^T=B^T A^T identity is proved numerically (the
    linked-CBLAS path self-skips when no BLAS is present, exactly as the B5 tests do);
  * a genuine priced channel win (the GPU channel under a hot Theta, via the cost model's existing thermal
    coupling) and a genuine priced prefetch win (a memory-bound shape).

Distinct from G3: this is narrowly the wrapped kernel's OWN calling-side params (the cblas call's major-
order + leading dim, the caller's block, the __builtin_prefetch hints, the channel) -- NOT a general
SoA<->AoS transform of tensor resources. All deterministic + pure-Python on the oracle rail; no MLIR change.
"""

import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import replace

from bcir.channels import CHANNELS
from bcir.kbcir.calling_side import (
    COL_MAJOR,
    ROW_MAJOR,
    CallingSideCertificate,
    CallingSidePlan,
    _PREFETCH_FACTOR,
    _order_cost,
    certify_calling_side,
    plan_calling_side,
    tune_wrapped_gemm,
)
from bcir.kbcir.cost import MEMORY, IDENTITY_FACTOR, TargetProfile, Theta
from bcir.kbcir.matmul import TilePlan, cost_of, plan_matmul
from bcir.lower.c_kernel import emit_blas_gemm_c, emit_tuned_gemm_c
from bcir.kbcir.matmul import matmul_reference

AVX = TargetProfile.x86_avx512()
COOL = Theta.cool()
HOT = Theta.hot()


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


# --- the plan: each calling-side param is the cost model's PRICED choice ---------------------------


def test_plan_block_is_the_existing_tile_search():
    # the caller-side block IS matmul.plan_matmul's TilePlan (reused unchanged), not a new search.
    for M, N, K in [(8, 8, 8), (256, 256, 4), (64, 16, 64)]:
        plan = plan_calling_side(M, N, K, AVX)
        assert plan.block == plan_matmul(M, N, K, AVX)
        assert 1 <= plan.block.tile_m <= M and 1 <= plan.block.tile_n <= N


def test_plan_major_order_is_priced_through_cost_of():
    # major-order is the lower-bottleneck order under the EXISTING cost_of traffic model; row wins ties
    # (the B5 baseline). The chosen major is exactly the argmin of _order_cost over {row, col}.
    for M, N, K in [(8, 8, 8), (256, 16, 256), (16, 256, 256), (300, 12, 300)]:
        plan = plan_calling_side(M, N, K, AVX)
        row_c = _order_cost(M, N, K, plan.block, ROW_MAJOR, AVX)
        col_c = _order_cost(M, N, K, plan.block, COL_MAJOR, AVX)
        expected = COL_MAJOR if col_c < row_c else ROW_MAJOR
        assert plan.major == expected, (M, N, K, row_c, col_c)


def test_prefetch_factor_is_a_single_axis_memory_coupling_like_deforestation():
    # the prefetch win is priced through CostVector.couple with a single-axis Q8 memory discount -- the
    # SAME shape as realize._DEFOREST_FACTOR (NOT a bespoke scalar). Discounts ONLY memory, identity else.
    assert _PREFETCH_FACTOR[MEMORY] < 256
    assert all(f == 256 for i, f in enumerate(_PREFETCH_FACTOR) if i != MEMORY)
    assert len(_PREFETCH_FACTOR) == len(IDENTITY_FACTOR)


def test_prefetch_triggers_only_when_there_is_a_next_block():
    # a single-block call (the whole M fits one tile) has nothing to prefetch -> distance 0 (a no-op).
    small = plan_calling_side(8, 8, 8, AVX)
    assert small.block.tile_m == 8 and not small.prefetches  # one block along M
    # a shape whose chosen block tiles M leaves blocks to hide -> distance 1.
    big = plan_calling_side(256, 256, 256, AVX)
    assert big.block.tile_m < 256 and big.prefetches and big.prefetch_distance == 1


def test_channel_is_priced_by_the_existing_optimize_on_profile():
    # cool tower: cpu and gpu price the dense-gemm tile equally (width capped at 16), so the tie stays on
    # the host CPU channel (off-host is adopted only on a STRICT win) -- the clean channel no-op.
    tower = [CHANNELS["x86_avx512"], CHANNELS["nvidia_ptx"]]
    plan = plan_calling_side(256, 256, 256, AVX, channels=tower, theta=COOL)
    assert plan.channel_kind == "cpu" and not plan.off_host


def test_channel_win_is_the_cost_models_thermal_coupling_on_a_hot_machine():
    # a HOT machine penalizes wide SIMD (the AVX-512 downclock the cost model already prices); the GPU
    # channel's lower thermal/power density makes its width-16 tile genuinely cheaper -> the cost model
    # routes off-host. This is the EXISTING thermal-coupling mechanism, not a B3-specific discount.
    tower = [CHANNELS["x86_avx512"], CHANNELS["nvidia_ptx"]]
    plan, cert = tune_wrapped_gemm(256, 256, 256, target=AVX, channels=tower, theta=HOT)
    assert plan.channel == "nvidia_ptx" and plan.channel_kind == "gpu" and plan.off_host
    assert cert.gain > 0 and not cert.is_noop  # the priced channel win


# --- the certificate: proof-carrying record + the clean no-op -------------------------------------


def test_certificate_is_a_clean_noop_when_tuning_is_unprofitable():
    # a small square on a host-only tower: row-major optimal, one block (no prefetch), host channel -> a
    # clean no-op (gain 0), mirroring fusion/bundle's no-op shape.
    plan, cert = tune_wrapped_gemm(8, 8, 8, target=AVX)
    assert isinstance(cert, CallingSideCertificate)
    assert plan.major == ROW_MAJOR and not plan.prefetches and not plan.off_host
    assert cert.gain == 0 and cert.is_noop


def test_certificate_records_the_prefetch_win_on_a_memory_bound_shape():
    # a memory-bound gemm (small K -> few MACs, lots of C traffic): memory is the bottleneck, so the
    # prefetch's memory-axis discount genuinely lowers the priced cost. The certificate records it.
    plan, cert = tune_wrapped_gemm(256, 256, 4, target=AVX)
    assert plan.prefetches
    assert cert.tuned_cost < cert.untuned_cost and cert.gain > 0
    assert not cert.is_noop


def test_certificate_records_before_after_and_baseline():
    plan, cert = tune_wrapped_gemm(256, 256, 4, target=AVX)
    assert cert.untuned_major == ROW_MAJOR and cert.untuned_channel == "host"
    assert cert.gain == cert.untuned_cost - cert.tuned_cost
    assert cert.plan is plan


def test_is_deterministic():
    tower = [CHANNELS["x86_avx512"], CHANNELS["nvidia_ptx"]]
    a_plan, a_cert = tune_wrapped_gemm(256, 256, 4, target=AVX, channels=tower, theta=HOT)
    b_plan, b_cert = tune_wrapped_gemm(256, 256, 4, target=AVX, channels=tower, theta=HOT)
    assert a_plan == b_plan
    assert (a_cert.untuned_cost, a_cert.tuned_cost, a_cert.gain) == (
        b_cert.untuned_cost,
        b_cert.tuned_cost,
        b_cert.gain,
    )


# --- the emitted tuned call: the chosen params appear, structure preserved ------------------------


def test_emit_row_major_carries_the_row_major_call_and_structure():
    plan = plan_calling_side(8, 8, 8, AVX)  # row-major, no prefetch
    c = emit_tuned_gemm_c(plan, "g")
    assert "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans" in c
    assert "8, 8, 8, 1.0f, A, 8, B, 8, 0.0f, C, 8" in c  # row-major leading dims
    assert "BCIR_USE_CBLAS" in c and "BCIR_GEMM_CBLAS" in c  # link-selected, structure preserved
    assert "s += A[i * 8 + k] * B[k * 8 + j]" in c  # the row-major reference fallback
    assert "channel: host (cpu)" in c
    assert "__builtin_prefetch" not in c  # no prefetch in this plan


def test_emit_column_major_uses_the_swapped_call_and_matching_leading_dim():
    base = plan_calling_side(6, 5, 7, AVX)
    plan = replace(base, major=COL_MAJOR, prefetch_distance=1)
    c = emit_tuned_gemm_c(plan, "g")
    # column-major computes C^T = B^T A^T into the SAME buffer: B first (lda=N), A second (ldb=K), C ldc=N.
    assert "cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans" in c
    assert (
        "5, 6, 7, 1.0f, B, 5, A, 7, 0.0f, C, 5" in c
    )  # m=N, n=M, k=K; B(lda=N=5), A(ldb=K=7), C(ldc=N=5)
    assert "__builtin_prefetch" in c  # the prefetch hint
    # the fallback is ALWAYS the row-major reference math (layout-independent) -- so it honors the same C.
    assert "s += A[i * 7 + k] * B[k * 5 + j]" in c


def test_emit_rejects_bad_dims():
    for bad in [(0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        plan = CallingSidePlan(
            M=bad[0],
            N=bad[1],
            K=bad[2],
            major=ROW_MAJOR,
            block=TilePlan(1, 1, 1, "ijk", 0, 0, 0, True),
            prefetch_distance=0,
            channel="host",
            channel_kind="cpu",
        )
        try:
            emit_tuned_gemm_c(plan, "g")
            assert False, f"{bad} should raise"
        except ValueError:
            pass


# --- the column-major C^T=B^T A^T identity: proved numerically (models the cblas ColMajor call) ----


def test_column_major_identity_equals_the_row_major_reference():
    # exactly what `cblas_sgemm(CblasColMajor, NoTrans, NoTrans, N, M, K, B, N, A, K, C, N)` computes:
    # ColMajor C[r + c*ldc] = sum_p first[r+p*lda]*second[p+c*ldb], first=B(lda=N), second=A(ldb=K).
    def cblas_colmajor(A, B, M, N, K):
        m, n, k, lda, ldb, ldc = N, M, K, N, K, N
        C = [0.0] * (M * N)
        for r in range(m):
            for c in range(n):
                s = 0.0
                for p in range(k):
                    s += B[r + p * lda] * A[p + c * ldb]
                C[r + c * ldc] = s
        return C

    rng = random.Random(0xC0F)
    for _ in range(400):
        M, N, K = rng.randint(1, 8), rng.randint(1, 8), rng.randint(1, 8)
        a = [rng.uniform(-3, 3) for _ in range(M * K)]
        b = [rng.uniform(-3, 3) for _ in range(K * N)]
        ref = matmul_reference(a, b, M, N, K)
        cm = cblas_colmajor(a, b, M, N, K)
        assert all(abs(x - y) < 1e-9 for x, y in zip(ref, cm)), (M, N, K)


# --- REFERENCE-INVARIANCE (the gate that matters): tuned emit == untuned reference, compile+run ----


def _run_gemm(kernel: str, fn: str, M: int, N: int, K: int, a, b, fmt: str = ".1f"):
    """Compile a gemm kernel + a tiny main that prints C[i], run it, return the floats."""
    cc = _cc()
    main = (
        f"\n#include <stdio.h>\nint main(void){{\n"
        f"  float A[{M * K}] = {{{', '.join(f'{x:.6f}f' for x in a)}}};\n"
        f"  float B[{K * N}] = {{{', '.join(f'{x:.6f}f' for x in b)}}};\n"
        f"  float C[{M * N}];\n  {fn}(A, B, C);\n"
        f'  for (int i=0;i<{M * N};++i) printf("%{fmt} ", C[i]);\n  return 0;\n}}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "g.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "g")
        bld = subprocess.run(
            [cc, "-std=c11", "-O2", "-Wall", "-Wextra", src, "-o", exe],
            capture_output=True,
            text=True,
        )
        assert bld.returncode == 0, bld.stderr  # the fallback path needs NO BLAS lib
        out = subprocess.run([exe], capture_output=True, text=True)
        return [float(x) for x in out.stdout.split()]


def test_tuned_row_major_is_bit_exact_to_the_untuned_reference():
    cc = _cc()
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    rng = random.Random(0xB3A)
    M, N, K = 5, 6, 7
    a = [float(rng.randint(-4, 4)) for _ in range(M * K)]  # integer-valued -> float sums exact
    b = [float(rng.randint(-4, 4)) for _ in range(K * N)]
    untuned = _run_gemm(emit_blas_gemm_c(M, N, K, "g"), "g", M, N, K, a, b)
    plan = plan_calling_side(M, N, K, AVX)
    tuned = _run_gemm(emit_tuned_gemm_c(plan, "g"), "g", M, N, K, a, b)
    assert tuned == untuned  # the tuned row-major call == untuned reference


def test_tuned_column_major_and_prefetch_are_bit_exact_to_the_untuned_reference():
    # THE LOAD-BEARING INVARIANCE: column-major + prefetch computes the IDENTICAL C as the untuned row-
    # major reference. (The fallback path -- exercised here, no BLAS in the sandbox -- always runs the
    # row-major reference math; the column-major C^T=B^T A^T identity for the linked cblas call is proved
    # in test_column_major_identity_equals_the_row_major_reference. A prefetch hint is a pure cache hint
    # with 0 numeric effect.)
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0xB3C)
    M, N, K = 6, 5, 7
    a = [float(rng.randint(-4, 4)) for _ in range(M * K)]
    b = [float(rng.randint(-4, 4)) for _ in range(K * N)]
    untuned = _run_gemm(emit_blas_gemm_c(M, N, K, "g"), "g", M, N, K, a, b)
    base = plan_calling_side(M, N, K, AVX)
    for plan in (
        base,
        replace(base, major=COL_MAJOR),  # column-major only
        replace(base, prefetch_distance=1),  # prefetch only
        replace(base, major=COL_MAJOR, prefetch_distance=1),
    ):  # both
        tuned = _run_gemm(emit_tuned_gemm_c(plan, "g"), "g", M, N, K, a, b)
        assert tuned == untuned, (plan.major, plan.prefetch_distance)


def test_a_channel_choice_does_not_change_the_numeric_result():
    # a different execution channel computes the same value -- the channel annotation rides the C but the
    # math is identical (the fallback is channel-independent).
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0xB3D)
    M, N, K = 4, 4, 5
    a = [float(rng.randint(-3, 3)) for _ in range(M * K)]
    b = [float(rng.randint(-3, 3)) for _ in range(K * N)]
    untuned = _run_gemm(emit_blas_gemm_c(M, N, K, "g"), "g", M, N, K, a, b)
    host_plan = plan_calling_side(M, N, K, AVX)
    gpu_plan = replace(host_plan, channel="nvidia_ptx", channel_kind="gpu")
    tuned = _run_gemm(emit_tuned_gemm_c(gpu_plan, "g"), "g", M, N, K, a, b)
    assert tuned == untuned
    assert "channel: nvidia_ptx (gpu)" in emit_tuned_gemm_c(gpu_plan, "g")


# --- the linked CBLAS path agrees too (column-major included) -- self-skips when no BLAS present ---


def _cblas_link():
    cc = _cc()
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


def test_linked_cblas_column_major_agrees_when_a_blas_is_present():
    lib = _cblas_link()
    if not lib:
        return  # no CBLAS here -> the real-link path self-skips
    cc = _cc()
    rng = random.Random(0xCB13)
    M, N, K = 4, 4, 4
    a = [float(rng.randint(-3, 3)) for _ in range(M * K)]
    b = [float(rng.randint(-3, 3)) for _ in range(K * N)]
    ref = matmul_reference(a, b, M, N, K)
    base = plan_calling_side(M, N, K, AVX)
    plan = replace(base, major=COL_MAJOR)  # exercise the column-major cblas call
    kernel = emit_tuned_gemm_c(plan, "g")
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
