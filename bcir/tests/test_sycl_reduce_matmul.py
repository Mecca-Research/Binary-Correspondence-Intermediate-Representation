"""S3: round out the SYCL oracle -- the channel's OTHER two declared capabilities (``reduce`` + ``matmul``).

The ``sycl_spirv`` channel declares ``{data_parallel, matmul, reduce}``; ``data_parallel`` (SAXPY) is
exercised by test_sycl_channel / test_sycl_dispatch. This adds the SAME proven pattern for the other two:

  * REDUCE (``out = Sum x[i]``, the canonical ``sycl::reduction`` op): a deterministic sequential-sum
    reference (``kbcir.sycl_reduce.reduce_reference``) + an R17 Q8 bridge (``reduce_via_bridge``), the
    emitted single-source C++ kernel (``emit_sycl_reduce_c``: a ``sycl::reduction`` device path + a portable
    sequential-sum fallback), and a resident dispatcher round-trip (``SyclDispatcher.run_reduce``).
  * MATMUL (``C = A . B``): REUSES the existing B1/B5 ``kbcir.matmul.matmul_reference`` /
    ``gemm_via_bridge`` as the source of truth (no second matmul math), the emitted single-source C++ kernel
    (``emit_sycl_matmul_c``: a 2-D ``parallel_for`` device path + a portable triple-loop fallback), and a
    dispatcher round-trip (``SyclDispatcher.run_matmul``).

HONEST DEPTH (same discipline as test_sycl_channel / test_sycl_dispatch): there is NO SYCL toolchain / GPU
on CI. The portable C++ fallback compile+run is REAL and runs here; the ``-fsycl`` device path self-skips.
The reduce device path is a TREE reduction whose reordered float adds agree with the sequential reference
only within ``reduce_reorder_bound`` (~ n*eps*sum|x|) -- documented, never faked. SYCL is a compiler MODE
(-fsycl), NOT a ``c.call.libm:`` -l<lib> edge (re-asserted)."""

import os
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.matmul import gemm_via_bridge, matmul_reference
from bcir.kbcir.quantize import max_abs_error
from bcir.kbcir.sycl_reduce import reduce_reference, reduce_reorder_bound, reduce_via_bridge
from bcir.lower.c_kernel import emit_sycl_matmul_c, emit_sycl_reduce_c
from bcir.lower.sycl_dispatch import (
    ChannelDispatcher, DispatchUnavailable, SyclDispatcher, sycl_cxx,
)

# fixed data, reused across the kernel + dispatcher checks.
_X = [1.0, 2.0, 3.0, 0.5, -1.0, 4.0, -2.5, 0.0]
# a small matmul: A is 2x3, B is 3x4 -> C is 2x4 (m=2, k=3, n=4).
_M, _K, _N = 2, 3, 4
_A = [1.0, 2.0, 3.0,
      4.0, 5.0, 6.0]                                       # 2x3
_B = [1.0, 0.0, 2.0, 1.0,
      0.0, 1.0, 1.0, 2.0,
      3.0, 1.0, 0.0, 1.0]                                  # 3x4


def _independent_sum(x):
    """An INDEPENDENT recompute (no BCIR code, no numpy): out = sum(x[i]) via stdlib float add."""
    s = 0.0
    for v in x:
        s += v
    return s


def _independent_matmul(a, b, m, k, n):
    """An INDEPENDENT row-major matmul (no BCIR code, no numpy): C[i,j] = sum_kk A[i,kk]*B[kk,j]."""
    c = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a[i * k + kk] * b[kk * n + j]
            c[i * n + j] = s
    return c


# ============================ REDUCE ============================

# --- (1) the reduce oracle reference: matches an independent recompute + a known closed-form -----------

def test_reduce_reference_matches_an_independent_recompute():
    import random
    rng = random.Random(0x5EED)
    x = [rng.uniform(-5, 5) for _ in range(19)]
    got = reduce_reference(x)
    want = _independent_sum(x)                              # independent of BCIR's code
    assert abs(got - want) <= 1e-12 * (1.0 + abs(want)), (got, want)


def test_reduce_reference_known_closed_form():
    # sum([1,2,3,4,5]) = 15; sum of a single element is itself; an all-zero vector sums to 0.
    assert reduce_reference([1.0, 2.0, 3.0, 4.0, 5.0]) == 15.0
    assert reduce_reference([42.0]) == 42.0
    assert reduce_reference([0.0, 0.0, 0.0]) == 0.0
    # a closed-form anchor: sum(1..100) = 100*101/2 = 5050.
    assert reduce_reference([float(i) for i in range(1, 101)]) == 5050.0


def test_reduce_reference_rejects_empty():
    try:
        reduce_reference([])
        assert False, "empty reduce should raise"
    except ValueError:
        pass
    try:
        reduce_reorder_bound([])
        assert False, "empty reorder-bound should raise"
    except ValueError:
        pass


def test_reduce_reference_is_side_effect_free():
    x = [3.0, 1.0, -4.0, 1.5, 0.0]
    x0 = list(x)
    reduce_reference(x)
    reduce_via_bridge(x)
    assert x == x0


# --- (2) the reduce bridge: tracks the reference within the R17 + reorder bound ------------------------

def test_bridged_reduce_tracks_the_reference_within_quant_plus_reorder():
    # A reduction is a SUM, so the R17 input round-trip (per-input error e_i) accumulates at unit gain over
    # n terms (bound n*max_i e_i). reduce_via_bridge + reduce_reference share the SEQUENTIAL add order, so
    # NO reorder term applies between them -- only the input round-trip does. (The reorder term is the
    # ADDITIONAL slack a device TREE reduction would need; tested against reduce_reorder_bound below.)
    import random
    rng = random.Random(0x5E2D)
    x = [rng.uniform(-4, 4) for _ in range(16)]
    e = max_abs_error(x, group_size=8, bits=8)              # R17 per-input round-trip bound (real units)
    bound = len(x) * e + 1e-6                               # n terms summed at unit gain
    ref = reduce_reference(x)
    got = reduce_via_bridge(x, group_size=8, bits=8)
    assert abs(ref - got) <= bound, (ref, got, e)


def test_bridge_is_a_clean_roundtrip_then_exact_sum():
    # reduce_via_bridge == reduce_reference of the dequantized (round-tripped) inputs -- the sum sees
    # exactly the bridged values, so the bridge is the only error source (the add order is the reference's).
    import random
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(0x5E3)
    x = [rng.uniform(-2, 2) for _ in range(12)]
    xd = dequantize(quantize_per_group(x, 8, 8))
    assert reduce_via_bridge(x, 8, 8) == reduce_reference(xd)


def test_reduce_reorder_bound_is_a_nonneg_n_eps_term():
    # the reorder tolerance is ~ (n-1)*eps*sum|x| -- non-negative, zero for a 1-element reduction (nothing
    # to reorder), and growing with both n and the magnitude of the partial sums.
    assert reduce_reorder_bound([7.0]) == 0.0
    b = reduce_reorder_bound(_X)
    assert b >= 0.0
    assert reduce_reorder_bound([10.0] * 8) > reduce_reorder_bound([0.1] * 8)


# --- (3) the emitted reduce kernel: a SYCL reduction + a portable fallback, NOT a c.call.libm: edge ----

def test_emit_reduce_wraps_a_sycl_reduction_with_a_scalar_fallback():
    c = emit_sycl_reduce_c(8, "bcir_reduce")
    assert "sycl::reduction" in c                                         # the SYCL device reduction
    assert "sycl::plus<float>()" in c                                     # the canonical reduce op
    assert "parallel_for" in c                                           # the device reduction kernel
    assert "#if defined(BCIR_USE_SYCL)" in c                             # toolchain-selected device path
    assert "#include <sycl/sycl.hpp>" in c                              # the SYCL header on the device path
    assert "s += x[i];" in c                                             # the portable sequential-sum fallback
    assert "out[0] = s;" in c                                            # the scalar result
    # the signature bakes n in: (x, out), C++ (no extern "C").
    assert "void bcir_reduce(const float *x, float *out)" in c
    # CRITICAL: SYCL is a compiler MODE, not a c.call.libm: edge.
    assert "c.call.libm:" not in c
    try:
        emit_sycl_reduce_c(0, "bcir_reduce")
        assert False, "n<1 should raise"
    except ValueError:
        pass


# ============================ MATMUL ============================

# --- (4) the matmul oracle: REUSE the existing matmul_reference (independent recompute + closed form) --

def test_matmul_reference_matches_an_independent_recompute():
    # we REUSE kbcir.matmul.matmul_reference (B1/B5) as the source of truth -- no second matmul math.
    # matmul_reference's arg order is (a, b, M, N, K); our (m, k, n) maps to M=m, N=n, K=k.
    import random
    rng = random.Random(0xB1B1)
    m, k, n = 3, 5, 4
    a = [rng.uniform(-3, 3) for _ in range(m * k)]
    b = [rng.uniform(-3, 3) for _ in range(k * n)]
    got = matmul_reference(a, b, m, n, k)
    want = _independent_matmul(a, b, m, k, n)              # independent of BCIR's code
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert abs(g - w) <= 1e-9 * (1.0 + abs(w)), (g, w)


def test_matmul_reference_known_closed_form():
    # identity: A . I = A. A is 2x2, I is 2x2.
    a = [1.0, 2.0, 3.0, 4.0]
    ident = [1.0, 0.0, 0.0, 1.0]
    assert matmul_reference(a, ident, 2, 2, 2) == a
    # our fixed _A . _B closed form (2x3 . 3x4):
    #   row0 = [1,2,3]; col0 of B = [1,0,3] -> 1*1+2*0+3*3 = 10
    #   C[0,0]=10, C[0,1]=1*0+2*1+3*1=5, C[0,2]=1*2+2*1+3*0=4, C[0,3]=1*1+2*2+3*1=8
    c = matmul_reference(_A, _B, _M, _N, _K)
    assert c[0] == 10.0 and c[1] == 5.0 and c[2] == 4.0 and c[3] == 8.0


def test_matmul_reference_rejects_bad_dims():
    # emit_sycl_matmul_c is the dim-validating surface for the SYCL realization.
    for bad in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        try:
            emit_sycl_matmul_c(*bad)
            assert False, f"bad dims {bad} should raise"
        except ValueError:
            pass


def test_gemm_via_bridge_tracks_the_reference_within_quant_error():
    # REUSE the existing R17 wrapper gemm_via_bridge (no new matmul bridge): it is matmul_reference of the
    # round-tripped inputs, so the error vs the reference is the input quantization alone (matmul is exact
    # MAC, no transcendental). gemm_via_bridge's arg order is (a, b, M, N, K, gs, bits).
    import random
    rng = random.Random(0xB52D)
    m, k, n = 4, 6, 3
    a = [rng.uniform(-3, 3) for _ in range(m * k)]
    b = [rng.uniform(-3, 3) for _ in range(k * n)]
    ea = max_abs_error(a, 8, 8)
    eb = max_abs_error(b, 8, 8)
    # |C - C'| per output <= sum_kk |a*eb + b*ea + ea*eb|; a loose k-term bound suffices for the contract.
    amax = max((abs(v) for v in a), default=0.0)
    bmax = max((abs(v) for v in b), default=0.0)
    bound = k * (amax * eb + bmax * ea + ea * eb) + 1e-5
    ref = matmul_reference(a, b, m, n, k)
    got = gemm_via_bridge(a, b, m, n, k, 8, 8)
    for r, g in zip(ref, got):
        assert abs(r - g) <= bound, (r, g, ea, eb)


# --- (5) the emitted matmul kernel: a 2-D parallel_for + a portable fallback, NOT a c.call.libm: edge --

def test_emit_matmul_wraps_a_2d_parallel_for_with_a_triple_loop_fallback():
    c = emit_sycl_matmul_c(_M, _K, _N, "bcir_matmul")
    assert "parallel_for" in c                                            # the SYCL device kernel
    assert "sycl::range<2>" in c                                          # the 2-D output grid
    assert "sycl::id<2>" in c                                             # the (i, j) work-item index
    assert "#if defined(BCIR_USE_SYCL)" in c                             # toolchain-selected device path
    assert "#include <sycl/sycl.hpp>" in c                              # the SYCL header on the device path
    # the portable triple-loop fallback (the IDENTICAL row-major product as matmul_reference).
    assert f"for (std::size_t kk = 0; kk < {_K}; ++kk)" in c
    # the signature: (A, B, C), C++ (no extern "C").
    assert "void bcir_matmul(const float *A, const float *B, float *C)" in c
    # CRITICAL: SYCL is a compiler MODE, not a c.call.libm: edge (distinct from the B5 CBLAS gemm wrap).
    assert "c.call.libm:" not in c
    assert "cblas_sgemm" not in c                                        # not the FFI gemm


# ============================ the fallback compiles + runs + matches ============================

def _cxx():
    """A host C++ compiler (g++/clang++/c++), or None (the fallback compile/run self-skips)."""
    return shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")


def _run_kernel(cxx, kernel, fn_call_main, define=None, extra=None):
    """Compile the emitted kernel (as C++) + a provided ``main`` and return the parsed float list. Mirrors
    test_sycl_channel's _run_saxpy_kernel."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.cpp")
        with open(src, "w") as f:
            f.write(kernel + fn_call_main)
        exe = os.path.join(d, "k")
        cmd = [cxx, "-std=c++17", "-O2"]
        if define:
            cmd.append(define)
        if extra:
            cmd += extra
        cmd += [src, "-o", exe]
        bld = subprocess.run(cmd, capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return [float(v) for v in out.stdout.split()]


def _reduce_main(x):
    xb = ", ".join(f"{v:.8f}f" for v in x)
    return (f"\n#include <cstdio>\nint main(void){{\n"
            f"  float x[{len(x)}] = {{{xb}}};\n  float out;\n  bcir_reduce(x, &out);\n"
            f"  printf(\"%.8f\\n\", (double)out);\n  return 0;\n}}\n")


def _matmul_main(a, b, m, k, n):
    ab = ", ".join(f"{v:.8f}f" for v in a)
    bb = ", ".join(f"{v:.8f}f" for v in b)
    return (f"\n#include <cstdio>\nint main(void){{\n"
            f"  float A[{m * k}] = {{{ab}}};\n  float B[{k * n}] = {{{bb}}};\n  float C[{m * n}];\n"
            f"  bcir_matmul(A, B, C);\n"
            f"  for (int i = 0; i < {m * n}; ++i) printf(\"%.8f\\n\", (double)C[i]);\n"
            f"  return 0;\n}}\n")


def test_reduce_fallback_path_compiles_runs_and_is_correct():
    cxx = _cxx()
    if not cxx:
        return                                              # quick tier hides the toolchain -> self-skip
    ref = reduce_reference(_X)
    want = _independent_sum(_X)
    got = _run_kernel(cxx, emit_sycl_reduce_c(len(_X), "bcir_reduce"), _reduce_main(_X))   # NO -fsycl
    assert len(got) == 1
    # the C++ fallback is the SAME sequential sum (float32 vs the oracle's float64) -> float round-off.
    assert abs(got[0] - ref) < 1e-4 * (1.0 + abs(ref)), (got[0], ref)
    assert abs(got[0] - want) < 1e-4 * (1.0 + abs(want)), (got[0], want)


def test_matmul_fallback_path_compiles_runs_and_is_correct():
    cxx = _cxx()
    if not cxx:
        return                                              # self-skip
    ref = matmul_reference(_A, _B, _M, _N, _K)
    want = _independent_matmul(_A, _B, _M, _K, _N)
    got = _run_kernel(cxx, emit_sycl_matmul_c(_M, _K, _N, "bcir_matmul"),
                      _matmul_main(_A, _B, _M, _K, _N))     # NO -fsycl
    assert len(got) == _M * _N
    for g, r, w in zip(got, ref, want):
        assert abs(g - r) < 1e-4 * (1.0 + abs(r)), (g, r)
        assert abs(g - w) < 1e-4 * (1.0 + abs(w)), (g, w)


# --- the -fsycl device path runs + agrees IF a real SYCL compiler probes OK, else self-skips ----------

def test_reduce_device_path_agrees_when_a_sycl_compiler_is_present():
    sy = sycl_cxx()
    if not sy:
        return                                              # no SYCL toolchain -> the device path self-skips
    cxx, extra = sy
    ref = reduce_reference(_X)
    got = _run_kernel(cxx, emit_sycl_reduce_c(len(_X), "bcir_reduce"), _reduce_main(_X),
                      define="-DBCIR_USE_SYCL", extra=extra)
    # the SYCL TREE reduction reorders the float adds -> agrees within the R17-free reorder tolerance plus a
    # loose float slack (NOT bit-identical to the sequential reference, by design).
    tol = reduce_reorder_bound(_X) + 1e-2 * (1.0 + abs(ref))
    assert abs(got[0] - ref) <= tol, (got[0], ref, tol)


def test_matmul_device_path_agrees_when_a_sycl_compiler_is_present():
    sy = sycl_cxx()
    if not sy:
        return                                              # no SYCL toolchain -> the device path self-skips
    cxx, extra = sy
    ref = matmul_reference(_A, _B, _M, _N, _K)
    got = _run_kernel(cxx, emit_sycl_matmul_c(_M, _K, _N, "bcir_matmul"),
                      _matmul_main(_A, _B, _M, _K, _N), define="-DBCIR_USE_SYCL", extra=extra)
    for g, r in zip(got, ref):
        # each output's k-dot has the same accumulation order on both paths -> agree to float round-off.
        assert abs(g - r) < 1e-2 * (1.0 + abs(r)), (g, r)


# ============================ the resident dispatcher round-trip ============================

def test_dispatcher_run_reduce_matches_reference_and_reports_mode():
    disp = SyclDispatcher()
    assert isinstance(disp, ChannelDispatcher)
    try:
        got = disp.run_reduce(_X)
    except DispatchUnavailable:
        return                                              # no C++ compiler -> self-skip
    finally:
        disp.close()
    ref = reduce_reference(_X)
    # the dispatcher uses the portable fallback on CI (sequential sum == reference); a device tree reduction
    # would agree within the reorder tolerance. Cover both honestly.
    tol = 1e-4 * (1.0 + abs(ref)) if disp.mode == "fallback" \
        else reduce_reorder_bound(_X) + 1e-2 * (1.0 + abs(ref))
    assert abs(got - ref) <= tol, (got, ref, disp.mode)
    assert disp.mode in ("fallback", "sycl-device")
    if sycl_cxx() is None:
        assert disp.mode == "fallback"                      # the expected CI mode


def test_dispatcher_run_matmul_matches_reference_and_reports_mode():
    disp = SyclDispatcher()
    try:
        got = disp.run_matmul(_A, _B, _M, _K, _N)
    except DispatchUnavailable:
        return                                              # no C++ compiler -> self-skip
    finally:
        disp.close()
    ref = matmul_reference(_A, _B, _M, _N, _K)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert abs(g - r) < 1e-4 * (1.0 + abs(r)), (g, r)
    assert disp.mode in ("fallback", "sycl-device")
    if sycl_cxx() is None:
        assert disp.mode == "fallback"


def test_dispatcher_validates_reduce_and_matmul_inputs():
    # input validation is PURE and happens before any compile (no toolchain needed).
    disp = SyclDispatcher()
    try:
        disp.run_reduce([])
        assert False, "empty reduce should raise ValueError"
    except ValueError:
        pass
    for bad in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        try:
            disp.run_matmul([0.0], [0.0], *bad)
            assert False, f"bad matmul dims {bad} should raise"
        except ValueError:
            pass
    # operand length mismatch (A length != m*k) also raises before any compile.
    try:
        disp.run_matmul([1.0, 2.0], _B, _M, _K, _N)
        assert False, "A length mismatch should raise"
    except ValueError:
        pass
    disp.close()


# ============================ the architectural boundary: SYCL is NOT a link edge ============================

def test_sycl_reduce_matmul_is_not_a_link_flag_edge():
    # CRITICAL re-assert (mirrors test_sycl_channel / test_sycl_dispatch): the emitted reduce + matmul kernel
    # names resolve to None (SYCL is a compiler MODE -fsycl, NOT a c.call.libm: -l<lib> edge); no link rule.
    assert library_for_callee("bcir_reduce") is None
    assert library_for_callee("bcir_matmul") is None
    # the existing five library rules (+ libm + libc-implicit) still resolve -- no regression.
    assert library_for_callee("cblas_sgemm") == "-lcblas"
    assert library_for_callee("fftwf_execute") == "-lfftw3"
    assert library_for_callee("LAPACKE_sgesv") == "-llapack"
    assert library_for_callee("gsl_stats_mean") == "-lgsl"
    assert library_for_callee("Sleef_expf1_u10") == "-lsleef"
    assert library_for_callee("expf") == "-lm"
    assert library_for_callee("free") == NO_FLAG


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
    sys.exit(0)
