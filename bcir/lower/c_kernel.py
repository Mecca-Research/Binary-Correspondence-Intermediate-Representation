"""Portable C23 lowering for elementwise BCIR claims (the C kernel backend).

The K_BCIR-selected realization in the GEM StreamPack becomes a clean, portable
**C23** kernel: the chosen lane width drives the loop structure, the elementwise
op is emitted on the contracted element type, pointers are `restrict`-qualified
(the non-aliasing contract), and a scalar tail keeps the kernel bounds-safe for
any trip count. C is the universal kernel lingua franca (CUDA C / HIP / OpenCL C
/ ISPC / plain CPU C); this is the lowering that a driver-resident toolchain can
finish without BCIR-native instruction selection.

Library-first: `emit_kernel_c` is a pure `(plan) -> C string` function, reusable
both AOT (compiled + self-checked here by `compile_and_run_c`) and driver-embedded
(emit, hand to the resident compiler). `emit_selfcheck_c` wraps the kernel with a
self-checking `main` for the AOT path.

C23 conformance (ISO/IEC 9899:2023): `restrict` on the pointer parameters
(6.7.3.1, the aliasing guarantee), file-scope `static_assert` on the element size
(6.7.11), `#pragma STDC FP_CONTRACT OFF` for reproducible float results (7.12.2 --
no fused-multiply-add contraction, so the self-check's exact compare holds),
`<stddef.h>` `size_t` trip counts, and `<stdint.h>` `int32_t` for the FP-less
(eBPF-style) element type. Emit with `-std=c23` (clang) or `-std=c2x` (gcc). The
R12 lowering contract -- lane geometry, bounds, precision -- is checked by
`verify.verify_c_lowering`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

from ..gem import hydrate
from ..model import Module, Opcode, StrideClass
from ..kbcir.realize import RealizationResult
from .llvm import find_elementwise

# Opcode -> C operator (the elementwise binary ops the lowering supports).
C_OP = {Opcode.ADD: "+", Opcode.SUB: "-", Opcode.MUL: "*"}


def _ctype(elem: str) -> str:
    return "int32_t" if elem == "i32" else "float"


def _selected_segment(module: Module, result: RealizationResult, claim):
    """The hydrated StreamPack lane segment for `claim` (the GEM artifact the C
    kernel lowers from), or None if the pack has no matching segment."""
    pack = hydrate(module, result)
    for seg in pack.segments:
        if seg.claim_id == claim.id:
            return seg
    return None


def emit_kernel_c(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel",
                  elem: str = "f32", width_override: int | None = None,
                  hw_width: int | None = None) -> str:
    """Emit a portable C23 kernel for the selected elementwise realization, driven
    by the GEM StreamPack segment (lane width -> loop structure, op, resources).
    `elem` is "f32" (float) or "i32" (int32_t for FP-less targets); `width_override`
    forces a width. Pure: no I/O, reusable AOT or driver-embedded.

    **Width-aware lowering** (`hw_width` = the target's widest lane,
    `HProfile.vector_width`): the selected width is the K_BCIR plan's lane decision,
    but how it is realized depends on whether it is the *full* hardware lane or a
    deliberate sub-maximal throttle:

      * `w == hw_width` (the go-fast case): the width is a *floor*. Emit the
        **idiomatic** loop and let the resident compiler vectorize to >= the lane
        and interleave as it sees fit -- the planner picked the strategy, the
        compiler owns instruction selection. (Hand-blocking pins it at exactly `w`
        per step. On bandwidth-bound elementwise kernels the two are
        measured-neutral -- loop form is within noise -- so this is a correctness /
        division-of-labor refinement, not a guaranteed speedup; the floor matters
        for compute-bound kernels and stops the lowering from second-guessing isel.)
      * `1 < w < hw_width` (a thermal/power throttle, e.g. a hot-machine AVX-512
        downclock dodge): emit a **fixed-trip width-`w`** loop so the compiler
        vectorizes to exactly `w`, physically honoring the sub-maximal lane.
      * `w == 1`: a pure scalar loop.

    When `hw_width` is not given the lowering conservatively caps at `w` (the literal
    geometry-encoding form) -- the same code as before this change, so callers that
    do not know the target are unaffected. The R12 contract
    (`verify.verify_c_lowering`, which takes the same `hw_width`) checks the rule."""
    claim, cand = find_elementwise(module, result)
    seg = _selected_segment(module, result, claim)
    w = int(width_override) if width_override else int(seg.width if seg else cand.width)
    w = w if w >= 1 else 1
    op = C_OP[claim.opcode]
    ctype = _ctype(elem)

    # restrict is sound only when the read and write resources are disjoint (no
    # aliasing). The elementwise contract (2 reads, 1 write) makes this the norm.
    reads = tuple(seg.reads) if seg else tuple(claim.rd)
    writes = tuple(seg.writes) if seg else tuple(claim.wr)
    rqual = " restrict" if not (set(reads) & set(writes)) else ""

    includes = "#include <stddef.h>\n"
    fp_pragma = ""
    if elem == "i32":
        includes += "#include <stdint.h>\n"
    else:
        # reproducible float: disallow contraction so a + b is a single rounding.
        fp_pragma = "#pragma STDC FP_CONTRACT OFF\n"

    # Full hardware lane (go-fast) vs deliberate sub-maximal throttle vs scalar.
    full_lane = hw_width is not None and w == int(hw_width)
    if w == 1:
        geom = "scalar"
    elif full_lane:
        geom = "full hardware lane: idiomatic loop, the compiler vectorizes to >= width"
    else:
        geom = "throttled lane: capped loop honors the sub-maximal width"

    head = (
        f"/* BCIR -> portable C23 kernel (lower_contract). op={claim.op or op} "
        f"lane={cand.lane.name} width={w} elem={ctype} "
        f"(K_BCIR-selected; {geom}; StreamPack {seg.name if seg else '-'}). */\n"
        f"{includes}"
        f'static_assert(sizeof({ctype}) == 4, "BCIR {ctype} kernel needs a 4-byte element");\n'
        f"{fp_pragma}\n"
    )
    sig = (f"void {fn_name}(const {ctype} *{rqual} A, const {ctype} *{rqual} B,\n"
           f"             {ctype} *{rqual} C, size_t n)")

    if w == 1 or full_lane:
        # Idiomatic loop: scalar (w==1) or the full-lane go-fast form. The compiler
        # realizes the lane and interleaves freely (the width is a floor); a
        # hand-blocked loop would pin it at exactly w. Measured-neutral on
        # bandwidth-bound kernels -- this is the correct division of labor, not a
        # speedup claim.
        body = (f"{sig} {{\n"
                f"  for (size_t i = 0; i < n; ++i)\n"
                f"    C[i] = A[i] {op} B[i];\n"
                f"}}\n")
    else:
        # Sub-maximal throttle: a fixed-trip width-w inner loop the compiler
        # vectorizes to exactly w (honoring the deliberate sub-maximal lane), plus a
        # bounds-safe scalar tail.
        body = (
            f"{sig} {{\n"
            f"  size_t i = 0;\n"
            f"  /* width-{w} lane: a fixed-trip inner loop capped at the selected "
            f"(sub-maximal) width -- honors the thermal/power throttle */\n"
            f"  for (; i + {w}u <= n; i += {w}u)\n"
            f"    for (size_t j = 0; j < {w}u; ++j)\n"
            f"      C[i + j] = A[i + j] {op} B[i + j];\n"
            f"  /* scalar tail: bounds-safe for any trip count (the strict bounds contract) */\n"
            f"  for (; i < n; ++i)\n"
            f"    C[i] = A[i] {op} B[i];\n"
            f"}}\n"
        )
    return head + body


# --- Q-fixed lane arithmetic with exact-width _BitInt(N) (C23) --------------------

def _std_int_for(bits: int) -> str:
    """The smallest standard signed integer type holding >= `bits` value bits -- the
    C11 fallback when _BitInt(N) is unavailable."""
    for w in (8, 16, 32, 64):
        if bits <= w:
            return f"int{w}_t"
    raise ValueError(f"no standard integer type for {bits} bits (max 64)")


def emit_qfixed_kernel_c(module: Module, result: RealizationResult,
                         fn_name: str = "bcir_qfixed", lane_bits: int = 16,
                         frac_bits: int = 8) -> str:
    """Emit a **Q-fixed** elementwise kernel whose lanes are *exactly* `lane_bits`
    wide, using C23 `_BitInt(N)` (ISO/IEC 9899:2023 6.2.5) -- the place _BitInt pays.

    The K_BCIR cost algebra is Q8 fixed point (256 == x1.0); the deterministic
    integer/Q-fixed execution path is exactly where exact-width lanes matter. A
    Q(`lane_bits-frac_bits`).`frac_bits` lane multiply is `(a * b) >> frac_bits`; the
    product needs `2*lane_bits` bits, so a standard `int16_t` lane would silently
    promote to `int` (32-bit) and a 12- or 24-bit lane has *no* standard type at all.
    `_BitInt(N)` gives the exact storage width the frozen ABI declares and a
    well-defined `2N`-bit product accumulator -- no promotion surprises, deterministic
    wraparound across hosts.

      * MUL -> Q-fixed scaled multiply `(int_2N)a * b >> frac_bits` narrowed to N bits
        (the same `>> 8` coupling the cost model does, generalized to N/Q).
      * ADD/SUB -> same-scale lane add/sub, computed in the 2N accumulator and narrowed
        (so an N-bit overflow wraps deterministically, not via promotion).

    Portable by construction: the lane/accumulator types are a `_BitInt(N)`/`_BitInt(2N)`
    pair under C23 (when `__BITINT_MAXWIDTH__` admits 2N) and the smallest standard
    int pair otherwise, selected by the preprocessor -- so the *same source* builds and
    gives bit-identical results under `-std=c23` and `-std=c11` (the AOT self-check
    asserts both agree). Pure: no I/O, reusable AOT or driver-embedded."""
    if not (1 <= frac_bits < lane_bits):
        raise ValueError(f"need 1 <= frac_bits ({frac_bits}) < lane_bits ({lane_bits})")
    if 2 * lane_bits > 64:
        raise ValueError(f"lane_bits {lane_bits}: 2*lane_bits must be <= 64 for the fallback")
    claim, cand = find_elementwise(module, result)
    op = C_OP[claim.opcode]
    is_mul = claim.opcode == Opcode.MUL
    n2 = 2 * lane_bits
    lane_std = _std_int_for(lane_bits)
    acc_std = _std_int_for(n2)
    qm, qn = lane_bits - frac_bits, frac_bits

    reads = tuple(claim.rd)
    writes = tuple(claim.wr)
    rqual = " restrict" if not (set(reads) & set(writes)) else ""

    # MUL rescales by >> frac_bits; ADD/SUB keep the Q scale (no shift).
    combine = (f"((q_acc_t)A[i] * (q_acc_t)B[i]) >> {qn}" if is_mul
               else f"(q_acc_t)A[i] {op} (q_acc_t)B[i]")

    return (
        f"/* BCIR -> Q-fixed C23 kernel (exact-width _BitInt). op={claim.op or op} "
        f"format=Q{qm}.{qn} lane_bits={lane_bits} (K_BCIR-selected width={cand.width}; "
        f"the {'scaled multiply' if is_mul else 'same-scale add/sub'} of the integer "
        f"execution path). */\n"
        "#include <stddef.h>\n#include <stdint.h>\n"
        f"#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L \\\n"
        f"    && defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= {n2}\n"
        f"  typedef _BitInt({lane_bits}) q_lane_t;   /* exact {lane_bits}-bit Q-fixed lane */\n"
        f"  typedef _BitInt({n2}) q_acc_t;     /* exact {n2}-bit product accumulator */\n"
        f"  #define BCIR_QFIXED_BITINT 1\n"
        f"#else\n"
        f"  typedef {lane_std} q_lane_t;     /* portable fallback: smallest std int >= {lane_bits} bits */\n"
        f"  typedef {acc_std} q_acc_t;\n"
        f"  #define BCIR_QFIXED_BITINT 0\n"
        f"#endif\n"
        f'_Static_assert(BCIR_QFIXED_BITINT || sizeof(q_acc_t) * 8 >= {n2},\n'
        f'               "BCIR Q-fixed accumulator must hold the {n2}-bit product");\n'
        f"\n"
        f"void {fn_name}(const q_lane_t *{rqual} A, const q_lane_t *{rqual} B,\n"
        f"             q_lane_t *{rqual} C, size_t n) {{\n"
        f"  for (size_t i = 0; i < n; ++i)\n"
        f"    C[i] = (q_lane_t)({combine});\n"
        f"}}\n"
    )


def emit_quantized_dot_c(lane_bits: int, count: int, fn_name: str = "bcir_qdot") -> str:
    """Emit a quantized integer DOT PRODUCT over exact-width `_BitInt(lane_bits)` code lanes -- the A1
    inference primitive (a matmul is a grid of these; B1/B5 build on it). The accumulation is INTEGER and
    EXACT: a `2*lane_bits + ceil(log2(count))`-bit accumulator holds the full sum of code products with no
    per-term truncation, so -- unlike a Q8 fixed-point reduce -- it adds ZERO error (the quantized dot's
    only error is the input quantization; ``bcir.kbcir.quantize.accumulator_bits`` is this width).

    Portable like the Q-fixed kernel: the lane/accumulator are a `_BitInt(N)`/`_BitInt(acc)` pair under
    C23 (when ``__BITINT_MAXWIDTH__`` admits the accumulator) and the smallest standard ints otherwise,
    selected by the preprocessor -- so the same source builds bit-identically under -std=c23 and -std=c11.
    The dequant per-group scale is applied by the caller (oracle ``quantize.scaled_dot``); this kernel is
    the pure exact-integer core."""
    if lane_bits < 2:
        raise ValueError(f"lane_bits must be >= 2; got {lane_bits}")
    from ..kbcir.quantize import accumulator_bits          # the exact-accumulation width contract
    acc_bits = accumulator_bits(lane_bits, count)
    if acc_bits > 64:
        raise ValueError(f"acc_bits {acc_bits} (lane_bits={lane_bits}, count={count}) exceeds the 64-bit "
                         f"standard-int fallback ceiling; the C23 _BitInt path would still hold it")
    lane_std, acc_std = _std_int_for(lane_bits), _std_int_for(acc_bits)
    return (
        f"/* BCIR -> quantized integer dot product (exact-width _BitInt lanes). lane_bits={lane_bits} "
        f"count={count} acc_bits={acc_bits} (the accumulation is exact -> zero reduction error; only the "
        f"input quantization contributes). */\n"
        "#include <stddef.h>\n#include <stdint.h>\n"
        f"#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L \\\n"
        f"    && defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= {acc_bits}\n"
        f"  typedef _BitInt({lane_bits}) q_lane_t;   /* exact {lane_bits}-bit code lane */\n"
        f"  typedef _BitInt({acc_bits}) q_acc_t;     /* exact {acc_bits}-bit dot accumulator */\n"
        f"  #define BCIR_QDOT_BITINT 1\n"
        f"#else\n"
        f"  typedef {lane_std} q_lane_t;     /* portable fallback: smallest std int >= {lane_bits} bits */\n"
        f"  typedef {acc_std} q_acc_t;\n"
        f"  #define BCIR_QDOT_BITINT 0\n"
        f"#endif\n"
        f'_Static_assert(BCIR_QDOT_BITINT || sizeof(q_acc_t) * 8 >= {acc_bits},\n'
        f'               "BCIR quantized-dot accumulator must hold the {acc_bits}-bit exact sum");\n'
        f"\n"
        f"int64_t {fn_name}(const q_lane_t * restrict A, const q_lane_t * restrict B, size_t n) {{\n"
        f"  q_acc_t acc = 0;\n"
        f"  for (size_t i = 0; i < n; ++i)\n"
        f"    acc += (q_acc_t)A[i] * (q_acc_t)B[i];     /* exact integer MAC -- no truncation */\n"
        f"  return (int64_t)acc;\n"
        f"}}\n"
    )


def emit_blas_gemm_c(M: int, N: int, K: int, fn_name: str = "bcir_gemm") -> str:
    """B5: wrap a TRUSTED external BLAS sgemm through the `c.call.libm:` FFI edge -- "integrate, don't
    reinvent". BCIR owns the CALLING side: it fixes the row-major C = A@B layout and (with the A1.1 Q8
    bridge at the boundary) the precision, and DELEGATES the kernel to `cblas_sgemm` when CBLAS is linked
    (`-DBCIR_USE_CBLAS -lcblas`), with a portable reference triple-loop fallback selected by the
    preprocessor when it is not. Both paths compute the identical row-major product, so the same source is
    correct linked or standalone -- the win is on the calling side (layout / quant / fusion), not a
    reimplemented kernel. Dims are baked in (a planned claim knows its shape)."""
    if M < 1 or N < 1 or K < 1:
        raise ValueError(f"gemm dims must be >= 1; got M={M} N={N} K={K}")
    return (
        f"/* BCIR -> external trusted GEMM via the c.call.libm: edge (integrate, don't reinvent). "
        f"row-major C[{M}x{N}] = A[{M}x{K}] @ B[{K}x{N}]; CBLAS sgemm when linked, reference fallback "
        f"otherwise -- BCIR owns the calling side (layout + the Q8<->f32 boundary). */\n"
        "#include <stddef.h>\n"
        "#if defined(BCIR_USE_CBLAS)\n"
        "  #include <cblas.h>\n"
        "  #define BCIR_GEMM_CBLAS 1\n"
        "#else\n"
        "  #define BCIR_GEMM_CBLAS 0\n"
        "#endif\n"
        f"void {fn_name}(const float *A, const float *B, float *C) {{\n"
        f"#if BCIR_GEMM_CBLAS\n"
        f"  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,\n"
        f"              {M}, {N}, {K}, 1.0f, A, {K}, B, {N}, 0.0f, C, {N});\n"
        f"#else\n"
        f"  for (size_t i = 0; i < {M}; ++i)\n"
        f"    for (size_t j = 0; j < {N}; ++j) {{\n"
        f"      float s = 0.0f;\n"
        f"      for (size_t k = 0; k < {K}; ++k) s += A[i * {K} + k] * B[k * {N} + j];\n"
        f"      C[i * {N} + j] = s;\n"
        f"    }}\n"
        f"#endif\n"
        f"}}\n"
    )


def emit_tuned_gemm_c(plan, fn_name: str = "bcir_gemm_tuned") -> str:
    """B3: emit a wrapped BLAS sgemm whose CALLING-SIDE params are the cost model's PRICED PLAN, not
    hard-coded -- the tuned variant of ``emit_blas_gemm_c``. ``plan`` is a ``kbcir.calling_side.Calling
    SidePlan`` carrying the chosen {major-order, caller block, prefetch distance, channel}. The emitted C
    computes the IDENTICAL row-major C[MxN] = A[MxK] @ B[KxN] as the untuned ``emit_blas_gemm_c`` -- calling
    -side tuning never changes the math; that is the whole point (the invariance the tests prove).

    What the plan turns into emitted C:
      * **major-order** -- ``CblasRowMajor`` (lda=K, ldb=N, ldc=N) OR ``CblasColMajor`` via the textbook
        C^T = B^T A^T identity: ``cblas_sgemm(CblasColMajor, NoTrans, NoTrans, N, M, K, 1, B, N, A, K, 0,
        C, N)`` writes the SAME row-major C buffer (a row-major MxN matrix IS a column-major NxM matrix in
        memory), so the column-major call is numerically identical, byte-for-byte, to the row-major call.
      * **prefetch** -- when ``plan.prefetches``, the portable fallback loop emits ``__builtin_prefetch``
        hints for the NEXT row block of A (distance ``plan.prefetch_distance`` blocks ahead); a prefetch
        hint changes NOTHING numerically (it is a pure cache hint), so the result is unchanged. cblas owns
        its own prefetching internally, so the hint rides the fallback path where the loop is open.
      * **channel** -- emitted as a ``/* channel: <name> (<kind>) */`` annotation (the routed execution
        channel from ``channels.orchestrate``); a different channel computes the same value.

    Keeps the ``#if defined(BCIR_USE_CBLAS)`` + portable fallback structure: the fallback honors the SAME
    row-major result for either major-order (the math is layout-independent), so the same source is correct
    linked or standalone. Dims are baked in from the plan (a planned claim knows its shape)."""
    M, N, K = plan.M, plan.N, plan.K
    if M < 1 or N < 1 or K < 1:
        raise ValueError(f"gemm dims must be >= 1; got M={M} N={N} K={K}")
    major = "CblasColMajor" if not plan.is_row_major else "CblasRowMajor"
    # The chosen cblas call. Column-major passes B first (the C^T=B^T A^T swap) into the SAME C buffer.
    if plan.is_row_major:
        cblas_call = (f"  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,\n"
                      f"              {M}, {N}, {K}, 1.0f, A, {K}, B, {N}, 0.0f, C, {N});\n")
    else:
        cblas_call = (f"  /* column-major C^T = B^T A^T into the SAME row-major C buffer (identical result) */\n"
                      f"  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,\n"
                      f"              {N}, {M}, {K}, 1.0f, B, {N}, A, {K}, 0.0f, C, {N});\n")
    # The portable fallback: ALWAYS the row-major reference math (layout-independent), with the chosen
    # prefetch hints. A `__builtin_prefetch` is a pure cache hint -- zero numeric effect.
    blk = max(1, plan.block.tile_m)
    if plan.prefetches:
        pf = (f"      /* B3 prefetch: pull the next row block of A (distance {plan.prefetch_distance} "
              f"block(s)) -- a pure cache hint, 0 numeric effect */\n"
              f"      if (i + {plan.prefetch_distance * blk}u < {M}u)\n"
              f"        __builtin_prefetch(&A[(i + {plan.prefetch_distance * blk}u) * {K}u], 0, 1);\n")
    else:
        pf = ""
    fallback = (f"  for (size_t i = 0; i < {M}; ++i) {{\n"
                f"{pf}"
                f"    for (size_t j = 0; j < {N}; ++j) {{\n"
                f"      float s = 0.0f;\n"
                f"      for (size_t k = 0; k < {K}; ++k) s += A[i * {K} + k] * B[k * {N} + j];\n"
                f"      C[i * {N} + j] = s;\n"
                f"    }}\n"
                f"  }}\n")
    pf_note = (f"prefetch dist={plan.prefetch_distance}" if plan.prefetches else "no prefetch")
    return (
        f"/* BCIR -> B3 calling-side-tuned wrapped GEMM (cost-model-priced plan; integrate, don't reinvent). "
        f"row-major C[{M}x{N}] = A[{M}x{K}] @ B[{K}x{N}]; major-order={major}, {pf_note}, "
        f"channel={plan.channel} ({plan.channel_kind}). The tuned call computes the IDENTICAL C as the "
        f"untuned row-major reference -- calling-side tuning never changes the math. */\n"
        f"/* channel: {plan.channel} ({plan.channel_kind}) */\n"
        "#include <stddef.h>\n"
        "#if defined(BCIR_USE_CBLAS)\n"
        "  #include <cblas.h>\n"
        "  #define BCIR_GEMM_CBLAS 1\n"
        "#else\n"
        "  #define BCIR_GEMM_CBLAS 0\n"
        "#endif\n"
        f"void {fn_name}(const float *A, const float *B, float *C) {{\n"
        f"#if BCIR_GEMM_CBLAS\n"
        f"{cblas_call}"
        f"#else\n"
        f"{fallback}"
        f"#endif\n"
        f"}}\n"
    )


def emit_fftw_fft_c(n: int, fn_name: str = "bcir_fft") -> str:
    """B2: wrap a TRUSTED external FFTW 1-D complex FFT through the `c.call.libm:` FFI edge -- "integrate,
    don't reinvent", a genuinely NEW kernel class (a spectral transform, not a matmul). BCIR owns the
    CALLING side: it fixes the interleaved complex layout `X[2k]=Re, X[2k+1]=Im` and (with the A1.1 Q8
    bridge at the boundary) the precision, and DELEGATES the transform to FFTW's plan API
    (`fftwf_plan_dft_1d` + `fftwf_execute` + `fftwf_destroy_plan`) when linked (`-DBCIR_USE_FFTW -lfftw3`),
    with a portable reference DFT (the naive O(n^2) `sum_k x[k]*exp(-2*pi*i*j*k/n)`) selected by the
    preprocessor when it is not. BOTH paths compute the IDENTICAL forward transform (FFTW's
    FFTW_FORWARD == the `e^{-2*pi*i*jk/n}` sign convention), so the same source is correct linked or
    standalone -- the win is on the calling side (layout / quant), not a reimplemented FFT. `n` is baked
    in (a planned claim knows its length). The interface is a float[2*n] in -> float[2*n] out, interleaved
    complex, exactly the `fftwf_complex` memory layout (so the wrapped call is a pointer cast, no copy)."""
    if n < 1:
        raise ValueError(f"fft length must be >= 1; got {n}")
    return (
        f"/* BCIR -> external trusted 1-D FFT via the c.call.libm: edge (integrate, don't reinvent). "
        f"forward complex DFT of length {n}; interleaved layout X[2k]=Re,X[2k+1]=Im. FFTW "
        f"fftwf_plan_dft_1d(FFTW_FORWARD) when linked, naive O(n^2) reference DFT otherwise -- both "
        f"compute the identical transform; BCIR owns the calling side (layout + the Q8<->f32 boundary). */\n"
        "#include <stddef.h>\n"
        "#if defined(BCIR_USE_FFTW)\n"
        "  #include <fftw3.h>\n"
        "  #define BCIR_FFT_FFTW 1\n"
        "#else\n"
        "  #include <math.h>\n"
        "  #define BCIR_FFT_FFTW 0\n"
        "#endif\n"
        f"void {fn_name}(const float *in, float *out) {{\n"
        f"#if BCIR_FFT_FFTW\n"
        f"  /* The trusted external kernel. `in`/`out` are interleaved complex == fftwf_complex layout; the\n"
        f"     plan is created/executed/destroyed around the single transform (n={n} baked in). */\n"
        f"  fftwf_complex *fin = (fftwf_complex *)(void *)in;\n"
        f"  fftwf_complex *fout = (fftwf_complex *)out;\n"
        f"  fftwf_plan plan = fftwf_plan_dft_1d({n}, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE);\n"
        f"  fftwf_execute(plan);                  /* c.call.libm:fftwf_execute */\n"
        f"  fftwf_destroy_plan(plan);\n"
        f"#else\n"
        f"  /* Portable reference DFT: out[j] = sum_k in[k] * exp(-2*pi*i*j*k/n) -- the SAME forward\n"
        f"     transform FFTW computes, so linked and standalone agree to float round-off. */\n"
        f"  const float two_pi = 6.28318530717958647692f;\n"
        f"  for (size_t j = 0; j < {n}; ++j) {{\n"
        f"    float re = 0.0f, im = 0.0f;\n"
        f"    for (size_t k = 0; k < {n}; ++k) {{\n"
        f"      float ang = -two_pi * (float)((j * k) % {n}) / (float){n};   /* reduced phase: stable for large n */\n"
        f"      float c = cosf(ang), s = sinf(ang);                          /* c.call.libm:cosf/sinf */\n"
        f"      float xr = in[2 * k], xi = in[2 * k + 1];\n"
        f"      re += xr * c - xi * s;\n"
        f"      im += xr * s + xi * c;\n"
        f"    }}\n"
        f"    out[2 * j] = re;\n"
        f"    out[2 * j + 1] = im;\n"
        f"  }}\n"
        f"#endif\n"
        f"}}\n"
    )


def emit_lapack_solve_c(n: int, nrhs: int = 1, fn_name: str = "bcir_solve") -> str:
    """B-breadth (#61): wrap a TRUSTED external LAPACK dense linear solve through the `c.call.libm:` FFI edge
    -- "integrate, don't reinvent", a THIRD distinct kernel class (a direct SOLVER, not a matmul or a
    spectral transform; mirrors B5's BLAS sgemm and B2's FFTW). Solves `A x = b` via LU with PARTIAL
    PIVOTING. BCIR owns the CALLING side: it fixes the ROW-MAJOR layout and (with the A1.1 Q8 bridge at the
    boundary) the precision, and DELEGATES the solve to `LAPACKE_sgesv` (LAPACK_ROW_MAJOR) when linked
    (`-DBCIR_USE_LAPACK -llapacke -llapack`), with a portable reference Gaussian-elimination-with-partial-
    pivoting fallback selected by the preprocessor when it is not. BOTH paths solve the IDENTICAL system to
    the same result (to float round-off), so the same source is correct linked or standalone -- the win is
    on the calling side (layout / quant), not a reimplemented solver. `n`/`nrhs` are baked in (a planned
    claim knows its shape).

    LAYOUT + IN-PLACE CONTRACT (mirrors LAPACKE_sgesv with LAPACK_ROW_MAJOR exactly): `A` is the `n*n`
    coefficient matrix row-major (`A[i*n+j]`), `b` is `n*nrhs` row-major (`b[i*nrhs+r]`). `sgesv` is
    DESTRUCTIVE -- it OVERWRITES A with its L and U factors and OVERWRITES b WITH THE SOLUTION x, in place.
    Both paths here honor that contract byte-for-byte: on return, A holds the LU factors (its original
    entries are gone) and `b` holds the solution x (same `n*nrhs` row-major layout). A caller that needs the
    original A/b must copy them before the call. Returns 0 on success, or a positive pivot index (1-based)
    if A is singular -- the same `info` convention LAPACK uses (U[i,i] exactly 0 -> no unique solution)."""
    if n < 1:
        raise ValueError(f"solve dimension must be >= 1; got {n}")
    if nrhs < 1:
        raise ValueError(f"nrhs must be >= 1; got {nrhs}")
    return (
        f"/* BCIR -> external trusted dense linear solve via the c.call.libm: edge (integrate, don't "
        f"reinvent). solve A x = b (LU + partial pivoting); A is n*n={n}x{n} row-major, b is n*nrhs="
        f"{n}x{nrhs} row-major, OVERWRITTEN with x in place (the sgesv contract). LAPACKE_sgesv "
        f"(LAPACK_ROW_MAJOR) when linked, reference Gaussian-elimination-with-partial-pivoting fallback "
        f"otherwise -- both solve the identical system; BCIR owns the calling side (layout + the Q8<->f32 "
        f"boundary). Returns 0 on success, or the 1-based singular pivot index. */\n"
        "#include <stddef.h>\n"
        "#if defined(BCIR_USE_LAPACK)\n"
        "  #include <lapacke.h>\n"
        "  #define BCIR_SOLVE_LAPACK 1\n"
        "#else\n"
        "  #include <math.h>\n"
        "  #define BCIR_SOLVE_LAPACK 0\n"
        "#endif\n"
        f"int {fn_name}(float *A, float *b) {{\n"
        f"#if BCIR_SOLVE_LAPACK\n"
        f"  /* The trusted external kernel. LAPACK_ROW_MAJOR matches BCIR's row-major A/b layout; sgesv\n"
        f"     overwrites A with LU and b with x in place (n={n}, nrhs={nrhs} baked in). */\n"
        f"  lapack_int ipiv[{n}];\n"
        f"  lapack_int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, {n}, {nrhs}, A, {n}, ipiv, b, {nrhs});  /* c.call.libm:LAPACKE_sgesv */\n"
        f"  return (int)info;\n"
        f"#else\n"
        f"  /* Portable reference solver: Gaussian elimination with PARTIAL PIVOTING -- the SAME LU-with-\n"
        f"     partial-pivoting algorithm sgesv realizes, so linked and standalone agree to float round-off.\n"
        f"     Overwrites A (with the eliminated/LU state) and b (with x) in place, honoring the sgesv\n"
        f"     contract. Partial pivoting (largest-magnitude pivot per column) is numerically necessary. */\n"
        f"  const size_t N = {n}u, NR = {nrhs}u;\n"
        f"  for (size_t col = 0; col < N; ++col) {{\n"
        f"    size_t pivot = col;\n"
        f"    float best = fabsf(A[col * N + col]);\n"
        f"    for (size_t row = col + 1; row < N; ++row) {{    /* select the largest-magnitude pivot */\n"
        f"      float v = fabsf(A[row * N + col]);\n"
        f"      if (v > best) {{ best = v; pivot = row; }}\n"
        f"    }}\n"
        f"    if (best == 0.0f) return (int)(col + 1);          /* singular: 1-based pivot index, like sgesv info */\n"
        f"    if (pivot != col) {{                              /* swap the pivot row into place (A and b) */\n"
        f"      for (size_t j = 0; j < N; ++j) {{ float t = A[col * N + j]; A[col * N + j] = A[pivot * N + j]; A[pivot * N + j] = t; }}\n"
        f"      for (size_t r = 0; r < NR; ++r) {{ float t = b[col * NR + r]; b[col * NR + r] = b[pivot * NR + r]; b[pivot * NR + r] = t; }}\n"
        f"    }}\n"
        f"    float inv = 1.0f / A[col * N + col];\n"
        f"    for (size_t row = col + 1; row < N; ++row) {{     /* eliminate column `col` below the pivot */\n"
        f"      float factor = A[row * N + col] * inv;\n"
        f"      if (factor != 0.0f) {{\n"
        f"        for (size_t j = col; j < N; ++j) A[row * N + j] -= factor * A[col * N + j];\n"
        f"        for (size_t r = 0; r < NR; ++r) b[row * NR + r] -= factor * b[col * NR + r];\n"
        f"      }}\n"
        f"    }}\n"
        f"  }}\n"
        f"  for (size_t row = N; row-- > 0; ) {{               /* back substitution per right-hand side */\n"
        f"    for (size_t r = 0; r < NR; ++r) {{\n"
        f"      float s = b[row * NR + r];\n"
        f"      for (size_t j = row + 1; j < N; ++j) s -= A[row * N + j] * b[j * NR + r];\n"
        f"      b[row * NR + r] = s / A[row * N + row];\n"
        f"    }}\n"
        f"  }}\n"
        f"  return 0;\n"
        f"#endif\n"
        f"}}\n"
    )


def emit_lapack_ols_c(m: int, n: int, nrhs: int = 1, fn_name: str = "bcir_ols") -> str:
    """E1 (ML-breadth): wrap a TRUSTED external LAPACK OVERDETERMINED least-squares solve through the
    `c.call.libm:` FFI edge -- "integrate, don't reinvent", GENERALIZING ``emit_lapack_solve_c`` (the square
    ``sgesv`` solve) to OLS linear regression. Finds the ``x`` minimizing ``||A x - b||_2`` for an m x n design
    matrix A (m >= n) and right-hand side b. BCIR owns the CALLING side: it fixes the ROW-MAJOR layout and (with
    the A1.1 Q8 bridge at the boundary) the precision, and DELEGATES the least-squares solve to
    ``LAPACKE_sgels`` (the QR-based driver, LAPACK_ROW_MAJOR, trans='N') when linked
    (`-DBCIR_USE_LAPACK -llapack`), with a portable normal-equations fallback selected by the preprocessor when
    it is not. Both compute the IDENTICAL least-squares ``x`` to float round-off ON A WELL-CONDITIONED SYSTEM,
    so the same source is correct linked or standalone -- the win is on the calling side (layout / quant), not a
    reimplemented solver. `m`/`n`/`nrhs` are baked in (a planned claim knows its shape).

    HONEST CONDITIONING NOTE (why the two paths differ in REALIZATION): the linked path uses ``sgels`` (a QR
    factorization of A directly -- conditioning ~``cond(A)``), the more numerically stable least-squares method.
    The portable fallback forms the NORMAL EQUATIONS ``G = A^T A`` (n x n), ``c = A^T b`` (n x nrhs) and solves
    ``G x = c`` by the SAME Gaussian-elimination-with-partial-pivoting as ``emit_lapack_solve_c``'s fallback;
    this squares the conditioning (~``cond(A)^2``) and is the textbook OLS twin of ``kbcir.ols.ols_reference``.
    On a WELL-CONDITIONED A (which the tests use) both agree to float round-off; on an ill-conditioned A the QR
    path is the more accurate (the honest caveat). CI needs no LAPACK -- the fallback is the full standalone OLS.

    LAYOUT + IN-PLACE/WORKSPACE CONTRACT (mirrors LAPACKE_sgels with LAPACK_ROW_MAJOR exactly): `A` is the m x n
    design matrix row-major (`A[i*n+j]`), `b` is the right-hand side. ``sgels`` is DESTRUCTIVE -- it OVERWRITES
    A with its QR factorization and OVERWRITES b with the result, in place; b must be sized for the LARGER of
    the m-row input and the n-row solution (LAPACKE_sgels's ``ldb = max(m,n)``-row contract: b is an
    ``max(m,n) x nrhs`` buffer, the first m rows the input, the first n rows the solution x on return). LAPACKE
    allocates its own QR workspace. This emitter's signature takes a separate ``x`` output buffer (n x nrhs) and
    does NOT mutate A/b: the linked path copies A and b into ``sgels``-shaped scratch internally, runs sgels,
    and copies the first n result rows into x; the fallback writes x from the normal-equations solve. So the
    CALLER's A/b survive (the destructive sgels contract is honored on internal scratch, not the caller's data)
    -- the kernel is ``void {fn}(const float *A, const float *b, float *x)``. Returns nothing; a singular normal
    matrix in the fallback (rank-deficient A) leaves x at the partial-elimination state (the honest analog of
    sgels's ``info > 0`` "A is rank-deficient")."""
    if m < 1 or n < 1 or nrhs < 1:
        raise ValueError(f"ols dims must be >= 1; got m={m} n={n} nrhs={nrhs}")
    if m < n:
        raise ValueError(f"ols needs an overdetermined system m >= n; got m={m} < n={n}")
    ldb = max(m, n)
    return (
        f"/* BCIR -> external trusted OVERDETERMINED least-squares (OLS) via the c.call.libm: edge (integrate, "
        f"don't reinvent; E1 generalizes the square sgesv solve to linear regression). minimize ||A x - b||_2; "
        f"A is m*n={m}x{n} row-major, b is m*nrhs={m}x{nrhs} row-major, x is n*nrhs={n}x{nrhs} row-major. "
        f"LAPACKE_sgels (QR, LAPACK_ROW_MAJOR, trans='N') when linked -- conditioning ~cond(A); reference "
        f"NORMAL-EQUATIONS (G=A^T A) fallback otherwise -- conditioning ~cond(A)^2. Both agree to float "
        f"round-off on a well-conditioned A; BCIR owns the calling side (layout + the Q8<->f32 boundary). */\n"
        "#include <stddef.h>\n"
        "#if defined(BCIR_USE_LAPACK)\n"
        "  #include <lapacke.h>\n"
        "  #include <string.h>\n"
        "  #define BCIR_OLS_LAPACK 1\n"
        "#else\n"
        "  #include <math.h>\n"
        "  #define BCIR_OLS_LAPACK 0\n"
        "#endif\n"
        f"void {fn_name}(const float *A, const float *b, float *x) {{\n"
        f"#if BCIR_OLS_LAPACK\n"
        f"  /* The trusted external kernel: QR-based least squares (the more stable, ~cond(A) method).\n"
        f"     sgels is DESTRUCTIVE + uses an ldb=max(m,n)={ldb}-row b buffer (the input in the first m rows,\n"
        f"     the solution in the first n rows on return), so copy the caller's A/b into sgels-shaped scratch\n"
        f"     (the caller's data survives), run sgels, then copy the first n solution rows into x. */\n"
        f"  float qa[{m * n}];\n"
        f"  float bx[{ldb * nrhs}];\n"
        f"  memcpy(qa, A, sizeof qa);\n"
        f"  for (size_t i = 0; i < {ldb}u * {nrhs}u; ++i) bx[i] = 0.0f;\n"
        f"  for (size_t i = 0; i < {m}u; ++i)\n"
        f"    for (size_t r = 0; r < {nrhs}u; ++r) bx[i * {nrhs}u + r] = b[i * {nrhs}u + r];\n"
        f"  LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', {m}, {n}, {nrhs}, qa, {n}, bx, {nrhs});  /* c.call.libm:LAPACKE_sgels */\n"
        f"  for (size_t i = 0; i < {n}u; ++i)\n"
        f"    for (size_t r = 0; r < {nrhs}u; ++r) x[i * {nrhs}u + r] = bx[i * {nrhs}u + r];\n"
        f"#else\n"
        f"  /* Portable reference: the NORMAL EQUATIONS G x = c, G = A^T A (n x n), c = A^T b (n x nrhs),\n"
        f"     solved by the SAME Gaussian elimination with PARTIAL PIVOTING as emit_lapack_solve_c's fallback.\n"
        f"     The textbook OLS twin of kbcir.ols.ols_reference; squares the conditioning (~cond(A)^2) but is\n"
        f"     exact on a well-conditioned A, so it agrees with the QR path to float round-off there. */\n"
        f"  const size_t M = {m}u, N = {n}u, NR = {nrhs}u;\n"
        f"  float G[{n * n}], c[{n * nrhs}];\n"
        f"  for (size_t p = 0; p < N; ++p) {{\n"
        f"    for (size_t q = 0; q < N; ++q) {{                 /* G = A^T A */\n"
        f"      float s = 0.0f;\n"
        f"      for (size_t i = 0; i < M; ++i) s += A[i * N + p] * A[i * N + q];\n"
        f"      G[p * N + q] = s;\n"
        f"    }}\n"
        f"    for (size_t r = 0; r < NR; ++r) {{                /* c = A^T b */\n"
        f"      float s = 0.0f;\n"
        f"      for (size_t i = 0; i < M; ++i) s += A[i * N + p] * b[i * NR + r];\n"
        f"      c[p * NR + r] = s;\n"
        f"    }}\n"
        f"  }}\n"
        f"  for (size_t col = 0; col < N; ++col) {{            /* solve G x = c in place on (G, c) */\n"
        f"    size_t pivot = col;\n"
        f"    float best = fabsf(G[col * N + col]);\n"
        f"    for (size_t row = col + 1; row < N; ++row) {{    /* partial pivoting (numerically necessary) */\n"
        f"      float v = fabsf(G[row * N + col]);\n"
        f"      if (v > best) {{ best = v; pivot = row; }}\n"
        f"    }}\n"
        f"    if (best == 0.0f) {{ for (size_t r = 0; r < NR; ++r) x[col * NR + r] = 0.0f; continue; }}  /* singular normal matrix: rank-deficient A (sgels info>0 analog) */\n"
        f"    if (pivot != col) {{\n"
        f"      for (size_t j = 0; j < N; ++j) {{ float t = G[col * N + j]; G[col * N + j] = G[pivot * N + j]; G[pivot * N + j] = t; }}\n"
        f"      for (size_t r = 0; r < NR; ++r) {{ float t = c[col * NR + r]; c[col * NR + r] = c[pivot * NR + r]; c[pivot * NR + r] = t; }}\n"
        f"    }}\n"
        f"    float inv = 1.0f / G[col * N + col];\n"
        f"    for (size_t row = col + 1; row < N; ++row) {{\n"
        f"      float factor = G[row * N + col] * inv;\n"
        f"      if (factor != 0.0f) {{\n"
        f"        for (size_t j = col; j < N; ++j) G[row * N + j] -= factor * G[col * N + j];\n"
        f"        for (size_t r = 0; r < NR; ++r) c[row * NR + r] -= factor * c[col * NR + r];\n"
        f"      }}\n"
        f"    }}\n"
        f"  }}\n"
        f"  for (size_t row = N; row-- > 0; ) {{               /* back substitution -> x */\n"
        f"    for (size_t r = 0; r < NR; ++r) {{\n"
        f"      float s = c[row * NR + r];\n"
        f"      for (size_t j = row + 1; j < N; ++j) s -= G[row * N + j] * x[j * NR + r];\n"
        f"      x[row * NR + r] = s / G[row * N + row];\n"
        f"    }}\n"
        f"  }}\n"
        f"#endif\n"
        f"}}\n"
    )


def emit_lapack_eigh_c(n: int, fn_name: str = "bcir_eigh") -> str:
    """E2 (ML-breadth): wrap a TRUSTED external LAPACK SYMMETRIC EIGENDECOMPOSITION through the
    `c.call.libm:` FFI edge -- "integrate, don't reinvent", the PCA sibling of ``emit_lapack_ols_c``. Where E1's
    OLS forms a symmetric Gram matrix and SOLVES it, E2's PCA forms a symmetric covariance and EIGENDECOMPOSES
    it; this emitter is the eigensolver half. Given the n x n SYMMETRIC matrix ``C`` (row-major; the covariance
    from ``kbcir.pca.covariance_matrix``), it writes the n eigenvalues and n eigenvectors, sorted by eigenvalue
    DESCENDING (the PCA convention -- first principal component = largest variance) with the same sign
    convention as the Python reference. BCIR owns the CALLING side: it fixes the ROW-MAJOR layout and (with the
    A1.1 Q8 bridge at the boundary) the precision, and DELEGATES the eig to ``LAPACKE_ssyev`` (jobz='V',
    uplo='U', LAPACK_ROW_MAJOR) when linked (`-DBCIR_USE_LAPACK -llapack`), with a portable JACOBI fallback (the
    SAME algorithm as ``kbcir.pca._jacobi_eigh``) selected by the preprocessor when it is not. ``n`` is baked in
    (a planned claim knows its shape). CI needs NO LAPACK -- the Jacobi fallback is the full standalone path.

    HONEST CONDITIONING NOTE (why the two paths differ in REALIZATION): the linked path uses ``ssyev``
    (Householder tridiagonalization followed by an implicit-QR / divide-and-conquer iteration on the
    tridiagonal -- the numerically-hardened LAPACK eig). The portable fallback uses the classic JACOBI rotation
    sweep (the textbook symmetric eig, the C twin of ``kbcir.pca._jacobi_eigh``). The two methods differ in
    REALIZATION but compute the SAME spectral decomposition of a symmetric matrix and AGREE to float round-off
    on WELL-SEPARATED eigenvalues (which the tests use). The honest caveat: a DEGENERATE (repeated) eigenvalue
    makes the eigenVECTORS non-unique -- any orthonormal basis of the eigenspace is a valid answer, so ssyev and
    Jacobi can legitimately return DIFFERENT bases there (this is not a bug; tests must use DISTINCT eigenvalues
    to compare eigenvectors). Both honor the SAME post-processing: eigenvalues sorted DESCENDING (ssyev returns
    them ASCENDING, so the linked path reverses), and each eigenvector's largest-magnitude entry forced positive
    (the sign convention -- ties broken by the lowest index -- so the two paths produce identical signs).

    LAYOUT + SCRATCH CONTRACT (mirrors LAPACKE_ssyev with LAPACK_ROW_MAJOR): ``C`` is the n x n symmetric matrix
    row-major (``C[p*n+q]``). ``ssyev`` is DESTRUCTIVE -- it OVERWRITES its matrix argument with the eigenVECTORS
    (jobz='V') -- so, exactly as ``emit_lapack_ols_c`` copies for ``sgels``, this emitter copies the caller's
    ``C`` into internal scratch, runs ssyev there, and writes the (sorted, sign-fixed) results into the separate
    ``eigvals`` / ``eigvecs`` output buffers. So the CALLER's ``C`` is NOT mutated. The signature is
    ``void {fn}(const float *C, float *eigvals, float *eigvecs)``: ``eigvals`` is a length-n buffer (descending),
    ``eigvecs`` is an n*n row-major buffer whose ROW t (``eigvecs[t*n + j]``) is component t's j-th coordinate --
    the same flat layout ``kbcir.pca.pca_reference`` returns. Returns nothing."""
    if n < 1:
        raise ValueError(f"eigh dimension must be >= 1; got {n}")
    return (
        f"/* BCIR -> external trusted SYMMETRIC EIGENDECOMPOSITION (PCA) via the c.call.libm: edge (integrate, "
        f"don't reinvent; E2 -- the eig sibling of E1's OLS solve). symmetric C[{n}x{n}] row-major -> n "
        f"eigenvalues (DESCENDING) + n eigenvectors (row t = component t), sign convention: largest-magnitude "
        f"entry positive. LAPACKE_ssyev (jobz='V', uplo='U', LAPACK_ROW_MAJOR) when linked -- Householder + "
        f"implicit-QR; reference JACOBI rotation fallback otherwise -- the C twin of kbcir.pca._jacobi_eigh. "
        f"Both agree to float round-off on well-separated eigenvalues; BCIR owns the calling side (layout + the "
        f"Q8<->f32 boundary). Degenerate eigenvalues make eigenVECTORS non-unique (use distinct ones). */\n"
        "#include <stddef.h>\n"
        "#include <math.h>\n"
        "#if defined(BCIR_USE_LAPACK)\n"
        "  #include <lapacke.h>\n"
        "  #include <string.h>\n"
        "  #define BCIR_EIGH_LAPACK 1\n"
        "#else\n"
        "  #define BCIR_EIGH_LAPACK 0\n"
        "#endif\n"
        f"void {fn_name}(const float *C, float *eigvals, float *eigvecs) {{\n"
        f"  const size_t N = {n}u;\n"
        f"  float val[{n}];        /* eigenvalues, pre-sort (ssyev: ascending; Jacobi: unsorted) */\n"
        f"  float vec[{n}][{n}];   /* vec[t] = eigenvector t (a row) */\n"
        f"#if BCIR_EIGH_LAPACK\n"
        f"  /* The trusted external kernel: ssyev (jobz='V', uplo='U'). ssyev OVERWRITES its matrix argument\n"
        f"     with the eigenvectors and is DESTRUCTIVE, so copy the caller's C into scratch (the caller's C\n"
        f"     survives), run ssyev, then read the eigenvalues (ASCENDING) and eigenvector ROWS out. With\n"
        f"     LAPACK_ROW_MAJOR + uplo='U', on return scratch's ROW t holds eigenvector t. */\n"
        f"  float scratch[{n * n}];\n"
        f"  memcpy(scratch, C, sizeof scratch);\n"
        f"  LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', {n}, scratch, {n}, val);  /* c.call.libm:LAPACKE_ssyev */\n"
        f"  for (size_t t = 0; t < N; ++t)\n"
        f"    for (size_t j = 0; j < N; ++j) vec[t][j] = scratch[t * N + j];\n"
        f"#else\n"
        f"  /* Portable reference: the classic JACOBI rotation sweep -- the C twin of kbcir.pca._jacobi_eigh.\n"
        f"     Maintain a working symmetric matrix a (a copy of C) and an accumulated rotation matrix v; each\n"
        f"     cyclic sweep applies the plane rotation that zeroes a[p][q] (a similarity transform, so the\n"
        f"     spectrum is preserved) and accumulates it into v. Converges to float precision on a symmetric\n"
        f"     matrix. Deterministic (fixed sweep order + cap), so it agrees with ssyev to round-off on\n"
        f"     well-separated eigenvalues. */\n"
        f"  float a[{n}][{n}], v[{n}][{n}];\n"
        f"  for (size_t p = 0; p < N; ++p)\n"
        f"    for (size_t q = 0; q < N; ++q) {{ a[p][q] = C[p * N + q]; v[p][q] = (p == q) ? 1.0f : 0.0f; }}\n"
        f"  for (int sweep = 0; sweep < 100; ++sweep) {{\n"
        f"    float off = 0.0f;                               /* off-diagonal Frobenius norm (convergence) */\n"
        f"    for (size_t p = 0; p < N; ++p)\n"
        f"      for (size_t q = p + 1; q < N; ++q) off += a[p][q] * a[p][q];\n"
        f"    if (sqrtf(2.0f * off) <= 1e-20f) break;          /* converged to float precision */\n"
        f"    for (size_t p = 0; p < N; ++p) {{\n"
        f"      for (size_t q = p + 1; q < N; ++q) {{\n"
        f"        float apq = a[p][q];\n"
        f"        if (apq == 0.0f) continue;                   /* already zero -> no rotation */\n"
        f"        float theta = (a[q][q] - a[p][p]) / (2.0f * apq);\n"
        f"        float tt = copysignf(1.0f, theta) / (fabsf(theta) + sqrtf(theta * theta + 1.0f));\n"
        f"        float cc = 1.0f / sqrtf(tt * tt + 1.0f);     /* cos */\n"
        f"        float ss = tt * cc;                          /* sin */\n"
        f"        for (size_t kk = 0; kk < N; ++kk) {{         /* A <- A J (columns p,q) */\n"
        f"          float akp = a[kk][p], akq = a[kk][q];\n"
        f"          a[kk][p] = cc * akp - ss * akq;\n"
        f"          a[kk][q] = ss * akp + cc * akq;\n"
        f"        }}\n"
        f"        for (size_t kk = 0; kk < N; ++kk) {{         /* A <- J^T A (rows p,q) */\n"
        f"          float apk = a[p][kk], aqk = a[q][kk];\n"
        f"          a[p][kk] = cc * apk - ss * aqk;\n"
        f"          a[q][kk] = ss * apk + cc * aqk;\n"
        f"        }}\n"
        f"        a[p][q] = 0.0f; a[q][p] = 0.0f;              /* force the annihilated entries to zero */\n"
        f"        for (size_t kk = 0; kk < N; ++kk) {{         /* V <- V J (accumulate the rotation) */\n"
        f"          float vkp = v[kk][p], vkq = v[kk][q];\n"
        f"          v[kk][p] = cc * vkp - ss * vkq;\n"
        f"          v[kk][q] = ss * vkp + cc * vkq;\n"
        f"        }}\n"
        f"      }}\n"
        f"    }}\n"
        f"  }}\n"
        f"  for (size_t t = 0; t < N; ++t) {{                  /* val[t] = a[t][t]; vec[t] = column t of V */\n"
        f"    val[t] = a[t][t];\n"
        f"    for (size_t r = 0; r < N; ++r) vec[t][r] = v[r][t];\n"
        f"  }}\n"
        f"#endif\n"
        f"  /* Common post-processing (BOTH paths): sort eigenpairs by eigenvalue DESCENDING (PCA convention --\n"
        f"     ssyev returns ascending, Jacobi unsorted; a selection sort that carries vectors along is fine for\n"
        f"     these small n), then force each eigenvector's largest-magnitude entry positive (the sign\n"
        f"     convention, ties to the lowest index) -- so the linked and fallback outputs are identical. */\n"
        f"  size_t idx[{n}];\n"
        f"  for (size_t t = 0; t < N; ++t) idx[t] = t;\n"
        f"  for (size_t s = 0; s < N; ++s) {{                  /* selection sort idx[] by val[] DESCENDING */\n"
        f"    size_t best = s;\n"
        f"    for (size_t t = s + 1; t < N; ++t)\n"
        f"      if (val[idx[t]] > val[idx[best]]) best = t;     /* strict > keeps stable order on ties */\n"
        f"    size_t tmp = idx[s]; idx[s] = idx[best]; idx[best] = tmp;\n"
        f"  }}\n"
        f"  for (size_t s = 0; s < N; ++s) {{\n"
        f"    size_t t = idx[s];\n"
        f"    eigvals[s] = val[t];\n"
        f"    size_t bj = 0; float bmag = fabsf(vec[t][0]);    /* largest-magnitude entry -> sign convention */\n"
        f"    for (size_t j = 1; j < N; ++j) {{ float mg = fabsf(vec[t][j]); if (mg > bmag) {{ bmag = mg; bj = j; }} }}\n"
        f"    float sign = (vec[t][bj] < 0.0f) ? -1.0f : 1.0f;\n"
        f"    for (size_t j = 0; j < N; ++j) eigvecs[s * N + j] = sign * vec[t][j];\n"
        f"  }}\n"
        f"}}\n"
    )


def emit_gsl_stats_c(kind: str, n: int, fn_name: str = "bcir_stats") -> str:
    """Area-B breadth (#62): wrap a TRUSTED external GSL (GNU Scientific Library) STATISTIC through the
    `c.call.libm:` FFI edge -- "integrate, don't reinvent", a FOURTH distinct library after B5's BLAS sgemm
    (a matmul), B2's FFTW (a spectral transform), and #61's LAPACK sgesv (a linear solve). GSL's distinctive
    domain is SPECIAL FUNCTIONS / STATISTICS, so this does not overlap the existing wraps. `kind` is one of
    "mean" / "variance" / "sd" -> ``gsl_stats_mean`` / ``gsl_stats_variance`` / ``gsl_stats_sd``.

    BCIR owns the CALLING side: it fixes the DENSE CONTIGUOUS (unit-stride) layout and (with the A1.1 Q8
    bridge at the boundary) the precision, and DELEGATES the statistic to GSL when linked
    (`-DBCIR_USE_GSL -lgsl -lgslcblas`; gsl depends on a cblas), with a portable reference fallback selected
    by the preprocessor when it is not. BOTH paths compute the IDENTICAL statistic (to float round-off), so
    the same source is correct linked or standalone -- CI needs no GSL installed.

    GSL CONTRACT (mirrored exactly): ``gsl_stats_mean(data, stride=1, n)`` = (1/n) sum data[i];
    ``gsl_stats_variance(data, 1, n)`` = (1/(n-1)) sum (data[i]-mean)^2 -- the UNBIASED (sample) variance
    with the ``n-1`` divisor; ``gsl_stats_sd(data, 1, n)`` = sqrt(variance). The mean/variance are exact
    sum/divide (NO transcendental); only the sd's square root rides the trusted ``c.call.libm:sqrtf`` edge
    (link -lm), quarantining the transcendental exactly as the activation/attention kernels do. The kernel
    takes a ``const float *data`` and returns the ``float`` statistic. `n` is baked in (a planned claim
    knows its length); variance/sd require n>=2 (the n-1 divisor), the mean n>=1."""
    if kind not in ("mean", "variance", "sd"):
        raise ValueError(f"unknown GSL statistic {kind!r}; expected 'mean', 'variance', or 'sd'")
    if n < 1:
        raise ValueError(f"stats length must be >= 1; got {n}")
    if kind in ("variance", "sd") and n < 2:
        raise ValueError(f"{kind} needs n >= 2 (the n-1 divisor); got {n}")
    gsl_call = f"gsl_stats_{kind}"
    # The GSL header for these is <gsl/gsl_statistics.h> (the gsl_stats_* family). The reference fallback
    # needs <math.h> only for the sd's sqrtf.
    head = (
        f"/* BCIR -> external trusted GSL statistic via the c.call.libm: edge (integrate, don't reinvent; a "
        f"FOURTH library -- GSL's special-functions/statistics domain). gsl_stats_{kind} over a dense "
        f"unit-stride float[{n}]; {gsl_call} when linked (-lgsl -lgslcblas), portable reference fallback "
        f"otherwise -- both compute the identical statistic to float round-off. BCIR owns the calling side "
        f"(layout + the Q8<->f32 boundary). */\n"
        "#include <stddef.h>\n"
        "#if defined(BCIR_USE_GSL)\n"
        "  #include <gsl/gsl_statistics.h>\n"
        "  #define BCIR_STATS_GSL 1\n"
        "#else\n"
        "  #include <math.h>\n"
        "  #define BCIR_STATS_GSL 0\n"
        "#endif\n"
    )
    # The trusted GSL call. gsl_stats_* take a (data, stride, n) triple; BCIR fixes stride=1 (dense).
    linked = (f"  /* The trusted external kernel: {gsl_call}(data, stride=1, n={n}). */\n"
              f"  return (float){gsl_call}(data, 1, {n});  /* c.call.libm:{gsl_call} */\n")
    # The portable reference: the IDENTICAL textbook statistic GSL computes (two-pass mean then, for
    # variance/sd, the unbiased sum-of-squared-deviations / (n-1)). Float math, so it agrees with the linked
    # GSL (double-internally) to float round-off, exactly as the FFTW/LAPACK fallbacks do.
    if kind == "mean":
        fallback = (f"  /* gsl_stats_mean reference: (1/n) sum data[i]. */\n"
                    f"  float sum = 0.0f;\n"
                    f"  for (size_t i = 0; i < {n}; ++i) sum += data[i];\n"
                    f"  return sum / (float){n};\n")
    else:
        var_body = (f"  /* gsl_stats_variance reference: (1/(n-1)) sum (data[i]-mean)^2 -- the UNBIASED\n"
                    f"     (sample) variance, GSL's n-1 divisor. Two-pass: mean, then squared deviations. */\n"
                    f"  float sum = 0.0f;\n"
                    f"  for (size_t i = 0; i < {n}; ++i) sum += data[i];\n"
                    f"  float mean = sum / (float){n};\n"
                    f"  float ss = 0.0f;\n"
                    f"  for (size_t i = 0; i < {n}; ++i) {{ float d = data[i] - mean; ss += d * d; }}\n"
                    f"  float var = ss / (float)({n} - 1);\n")
        if kind == "variance":
            fallback = var_body + "  return var;\n"
        else:  # sd = sqrt(variance) -- the only transcendental, on the c.call.libm:sqrtf edge
            fallback = var_body + "  return sqrtf(var);   /* c.call.libm:sqrtf -- the sole transcendental */\n"
    return (
        head
        + f"float {fn_name}(const float *data) {{\n"
        + "#if BCIR_STATS_GSL\n"
        + linked
        + "#else\n"
        + fallback
        + "#endif\n"
        + "}\n"
    )


def emit_sleef_exp_c(n: int, fn_name: str = "bcir_vexp") -> str:
    """Area-B breadth (#63): wrap a TRUSTED external SLEEF (SIMD-oriented vectorized libm) VECTORIZED
    ELEMENTWISE EXP through the `c.call.libm:` FFI edge -- "integrate, don't reinvent", the FIFTH and final
    Area-B library after B5's BLAS sgemm (a matmul), B2's FFTW (a spectral transform), #61's LAPACK sgesv (a
    linear solve), and #62's GSL (statistics). SLEEF's distinctive domain is VECTORIZED TRANSCENDENTAL MATH
    (a fast vectorized libm), so this does not overlap the existing wraps. The kernel maps `out[i] =
    exp(data[i])` over a dense unit-stride float[`n`] array.

    BCIR owns the CALLING side: it fixes the dense contiguous (unit-stride) layout and (with the A1.1 Q8
    bridge at the boundary) the precision, and DELEGATES the transcendental to SLEEF when linked
    (`-DBCIR_USE_SLEEF -lsleef`), with a portable reference fallback (plain libm `expf` per element, -lm)
    selected by the preprocessor when it is not. BOTH paths compute the IDENTICAL elementwise transform (to
    float round-off -- both are `exp`), so the same source is correct linked or standalone -- CI needs no
    SLEEF installed. The win is on the calling side (the elementwise loop / layout / quant), not a
    reimplemented transcendental.

    PORTABILITY (the function choice): the linked path calls `Sleef_expf1_u10` -- SLEEF's SCALAR
    single-precision exp at 1.0-ULP accuracy (`f1` == one f32 lane, `u10` == <= 1.0 ULP). The scalar form
    needs NO target-ISA vector type (no `__m256`/`svfloat32_t`), so the SAME emitted source builds on any
    ISA; BCIR owns the elementwise loop and the compiler/library owns the vectorization. `Sleef_expf1_u10`
    is the elementwise twin of libm `expf` (both `e^x` per lane) -- the SAME transcendental the G1
    activation kernels already route through the libm edge, of which SLEEF is the vectorized alternative.
    The transcendental stays fully OFF the deterministic legality rail (SLEEF/libm are opaque/trusted, like
    any external edge), exactly as the activation/attention `expf`. `n` is baked in (a planned claim knows
    its length); n >= 1 (an elementwise map needs at least one lane). The contract: `data`/`out` are dense
    `float[n]` (out may alias data for an in-place map; both paths read data[i] before writing out[i])."""
    if n < 1:
        raise ValueError(f"sleef exp length must be >= 1; got {n}")
    return (
        f"/* BCIR -> external trusted SLEEF vectorized exp via the c.call.libm: edge (integrate, don't "
        f"reinvent; a FIFTH library -- SLEEF's vectorized-transcendental-math domain). out[i] = exp(data[i]) "
        f"over a dense unit-stride float[{n}]; Sleef_expf1_u10 (scalar f32, 1.0 ULP) when linked (-lsleef), "
        f"portable libm expf-per-element fallback otherwise (-lm) -- both compute the identical elementwise "
        f"exp to float round-off. BCIR owns the calling side (the elementwise layout + the Q8<->f32 "
        f"boundary); the transcendental is trusted (off the deterministic rail, like the G1 activation "
        f"expf). */\n"
        "#include <stddef.h>\n"
        "#if defined(BCIR_USE_SLEEF)\n"
        "  #include <sleef.h>\n"
        "  #define BCIR_VEXP_SLEEF 1\n"
        "#else\n"
        "  #include <math.h>\n"
        "  #define BCIR_VEXP_SLEEF 0\n"
        "#endif\n"
        f"void {fn_name}(const float *data, float *out) {{\n"
        f"  for (size_t i = 0; i < {n}; ++i)\n"
        f"#if BCIR_VEXP_SLEEF\n"
        f"    out[i] = Sleef_expf1_u10(data[i]);   /* c.call.libm:Sleef_expf1_u10 -- the trusted vectorized libm */\n"
        f"#else\n"
        f"    out[i] = expf(data[i]);              /* c.call.libm:expf -- the portable reference twin (-lm) */\n"
        f"#endif\n"
        f"}}\n"
    )


def emit_sycl_saxpy_c(n: int, fn_name: str = "bcir_saxpy") -> str:
    """SYCL backend differential oracle: emit a SINGLE-SOURCE C++ SAXPY kernel ``out[i] = a*x[i] + y[i]`` over
    a dense unit-stride ``float[n]`` -- a SYCL device ``parallel_for`` when compiled with a SYCL toolchain
    (``-fsycl``, ``BCIR_USE_SYCL``), a portable scalar C++ fallback otherwise. BOTH paths compute the IDENTICAL
    SAXPY to float round-off (SAXPY is exact multiply-add, no transcendental), so the same source is correct
    on the device or standalone -- CI needs no SYCL compiler installed.

    CRITICAL -- this is NOT a ``c.call.libm:`` edge. The five Area-B library wraps (BLAS/FFTW/LAPACK/GSL/SLEEF)
    each call an external symbol through the ``c.call.libm:`` FFI edge and carry a ``-l<lib>`` link rule. SYCL
    is fundamentally different: it is a single-source C++ *compiler MODE* (``-fsycl``: ``icpx -fsycl`` / ``acpp``
    / ``clang++ -fsycl``), and the kernel is emitted SYCL C++ source (a ``parallel_for``), NOT a call to a
    linkable symbol. So this emitter mints NO ``c.call.libm:`` label and there is NO link-flag rule for SYCL in
    bcir/frontends/cfront/linkflags.py (a SYCL-ish symbol resolves to None/unknown -- asserted in
    test_sycl_channel.py). SYCL lives on the C++ side of the G8 boundary (a host runtime / device programming
    model ABOVE the deterministic C rail); this kernel is C++ so it is emitted as C++ either way (no
    ``extern "C"``), consistent with that placement.

    BCIR owns the CALLING side: the dense contiguous (unit-stride) ``float`` array layout and (with the A1.1 Q8
    bridge at the boundary) the precision; the device merely realizes the elementwise map. The signature bakes
    ``n`` in (a planned claim knows its length): ``void {fn}(float a, const float *x, const float *y, float
    *out)``; ``out`` may not alias x/y on the device path (a ``parallel_for`` writes out[i] from a*x[i]+y[i]).
    ``n >= 1`` (a map needs at least one lane)."""
    if n < 1:
        raise ValueError(f"sycl saxpy length must be >= 1; got {n}")
    return (
        f"/* BCIR -> SYCL device (parallel_for) differential oracle: a*x+y over a dense unit-stride "
        f"float[{n}]; the SYCL device path when compiled with a SYCL toolchain (-fsycl, BCIR_USE_SYCL), a "
        f"portable scalar C++ fallback otherwise -- both compute the identical SAXPY to float round-off. BCIR "
        f"owns the calling side (dense layout + the Q8<->f32 boundary). NOTE: SYCL is a compiler MODE "
        f"(-fsycl), NOT a 'c.call.libm' -l<lib> external-symbol edge. */\n"
        "#include <cstddef>\n"
        "#if defined(BCIR_USE_SYCL)\n"
        "  #include <sycl/sycl.hpp>\n"
        "  #define BCIR_SAXPY_SYCL 1\n"
        "#else\n"
        "  #define BCIR_SAXPY_SYCL 0\n"
        "#endif\n"
        f"void {fn_name}(float a, const float *x, const float *y, float *out) {{\n"
        "#if BCIR_SAXPY_SYCL\n"
        f"  /* SYCL device path: a single queue, USM device pointers, one parallel_for over the {n} lanes. */\n"
        "  sycl::queue q;\n"
        f"  const std::size_t n = {n};\n"
        "  float *dx = sycl::malloc_device<float>(n, q);\n"
        "  float *dy = sycl::malloc_device<float>(n, q);\n"
        "  float *dout = sycl::malloc_device<float>(n, q);\n"
        "  q.memcpy(dx, x, n * sizeof(float)).wait();\n"
        "  q.memcpy(dy, y, n * sizeof(float)).wait();\n"
        "  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {\n"
        "    dout[i] = a * dx[i] + dy[i];   /* out[i] = a*x[i] + y[i] */\n"
        "  }).wait();\n"
        "  q.memcpy(out, dout, n * sizeof(float)).wait();\n"
        "  sycl::free(dx, q); sycl::free(dy, q); sycl::free(dout, q);\n"
        "#else\n"
        f"  /* Portable scalar C++ fallback (no SYCL header, plain g++/clang++): the IDENTICAL SAXPY. */\n"
        f"  for (std::size_t i = 0; i < {n}; ++i)\n"
        f"    out[i] = a * x[i] + y[i];\n"
        "#endif\n"
        "}\n"
    )


def emit_sycl_reduce_c(n: int, fn_name: str = "bcir_reduce") -> str:
    """SYCL backend differential oracle (the ``reduce`` capability): emit a SINGLE-SOURCE C++ reduction
    kernel ``out = Sum_i x[i]`` over a dense unit-stride ``float[n]`` -- a SYCL device ``sycl::reduction``
    (``sycl::plus<float>()``) in a ``q.submit`` when compiled with a SYCL toolchain (``-fsycl``,
    ``BCIR_USE_SYCL``), a portable scalar C++ sequential sum otherwise. The signature bakes ``n`` in:
    ``void {fn}(const float *x, float *out)`` -- ``out`` is a 1-element result. ``n >= 1`` (a reduction
    needs at least one element).

    THE ACCURACY SUBTLETY -- unlike SAXPY, the two paths are NOT bit-identical. The portable fallback does a
    SEQUENTIAL left-to-right sum, which matches ``kbcir.sycl_reduce.reduce_reference`` EXACTLY; the SYCL
    ``sycl::reduction`` is a tree/parallel reduction that REORDERS the float adds (float add is not
    associative), so the device path agrees with the sequential reference only within the documented reorder
    tolerance (``kbcir.sycl_reduce.reduce_reorder_bound`` ~ ``n*eps*Sum|x|``). Both are honest realizations
    of ``Sum x[i]``; only their add ORDER differs.

    CRITICAL -- this is NOT a ``c.call.libm:`` edge. Like ``emit_sycl_saxpy_c``, this is a single-source C++
    *compiler MODE* (``-fsycl``) and the kernel is emitted SYCL C++ source (a ``sycl::reduction``), NOT a
    call to a linkable symbol -- so it mints NO ``c.call.libm:`` label and adds NO link-flag rule (a
    SYCL-ish symbol resolves to None/unknown). SYCL lives on the C++ side of the G8 boundary; the kernel is
    emitted as C++ either way (no ``extern "C"``)."""
    if n < 1:
        raise ValueError(f"sycl reduce length must be >= 1; got {n}")
    return (
        f"/* BCIR -> SYCL device (sycl::reduction) differential oracle: out = sum(x[i]) over a dense "
        f"unit-stride float[{n}]; the SYCL device path (sycl::reduction + sycl::plus<float>()) when "
        f"compiled with a SYCL toolchain (-fsycl, BCIR_USE_SYCL), a portable scalar C++ SEQUENTIAL sum "
        f"otherwise. NOTE: the device tree-reduction REORDERS the float adds, so it matches the sequential "
        f"reference only within ~n*eps*sum|x| (float add is non-associative); the fallback's sequential sum "
        f"matches exactly. SYCL is a compiler MODE (-fsycl), NOT a 'c.call.libm' -l<lib> edge. */\n"
        "#include <cstddef>\n"
        "#if defined(BCIR_USE_SYCL)\n"
        "  #include <sycl/sycl.hpp>\n"
        "  #define BCIR_REDUCE_SYCL 1\n"
        "#else\n"
        "  #define BCIR_REDUCE_SYCL 0\n"
        "#endif\n"
        f"void {fn_name}(const float *x, float *out) {{\n"
        "#if BCIR_REDUCE_SYCL\n"
        f"  /* SYCL device path: a queue, USM device pointers, one reduction over the {n} lanes with "
        f"sycl::plus. */\n"
        "  sycl::queue q;\n"
        f"  const std::size_t n = {n};\n"
        "  float *dx = sycl::malloc_device<float>(n, q);\n"
        "  float *dsum = sycl::malloc_device<float>(1, q);\n"
        "  q.memcpy(dx, x, n * sizeof(float)).wait();\n"
        "  q.memset(dsum, 0, sizeof(float)).wait();\n"
        "  q.submit([&](sycl::handler &h) {\n"
        "    auto r = sycl::reduction(dsum, sycl::plus<float>());\n"
        "    h.parallel_for(sycl::range<1>(n), r, [=](sycl::id<1> i, auto &acc) {\n"
        "      acc += dx[i];                  /* out += x[i] -- tree reduction (reordered adds) */\n"
        "    });\n"
        "  }).wait();\n"
        "  q.memcpy(out, dsum, sizeof(float)).wait();\n"
        "  sycl::free(dx, q); sycl::free(dsum, q);\n"
        "#else\n"
        f"  /* Portable scalar C++ fallback (no SYCL header): the SEQUENTIAL left-to-right sum -- matches "
        f"the BCIR reference exactly. */\n"
        "  float s = 0.0f;\n"
        f"  for (std::size_t i = 0; i < {n}; ++i)\n"
        "    s += x[i];\n"
        "  out[0] = s;\n"
        "#endif\n"
        "}\n"
    )


def emit_sycl_matmul_c(m: int, k: int, n: int, fn_name: str = "bcir_matmul") -> str:
    """SYCL backend differential oracle (the ``matmul`` capability): emit a SINGLE-SOURCE C++ matmul kernel
    ``C = A . B`` (row-major A[m x k], B[k x n], C[m x n]) -- a SYCL device 2-D ``parallel_for`` over
    ``sycl::range<2>(m, n)`` accumulating over ``k`` when compiled with a SYCL toolchain (``-fsycl``,
    ``BCIR_USE_SYCL``), a portable triple-loop C++ matmul otherwise. The signature bakes the dims in:
    ``void {fn}(const float *A, const float *B, float *C)``. ``m, k, n >= 1``.

    This matches ``kbcir.matmul.matmul_reference``'s layout + accumulation EXACTLY: row-major, each output
    ``C[i*n + j]`` is the sequential ``k``-order dot of A's row ``i`` and B's column ``j`` -- so the
    differential is clean (the realization reorders nothing per output; each output's k-sum is the same
    left-to-right order on both paths). For integer-valued inputs the device and fallback are bit-identical
    to the reference; for general floats they agree to float round-off (the dot accumulation order is the
    same, so there is no reduce-style reorder term between paths).

    CRITICAL -- this is NOT a ``c.call.libm:`` edge (and is distinct from the B5 ``emit_blas_gemm_c`` CBLAS
    wrap, which DOES call ``cblas_sgemm`` through the ``c.call.libm:`` FFI edge). Here the math is emitted as
    SYCL C++ source (a ``parallel_for``) in a single-source *compiler MODE* (``-fsycl``), NOT a call to a
    linkable symbol -- so it mints NO ``c.call.libm:`` label and adds NO link-flag rule (a SYCL-ish symbol
    resolves to None/unknown). SYCL lives on the C++ side of the G8 boundary; emitted as C++ either way (no
    ``extern "C"``)."""
    if m < 1 or k < 1 or n < 1:
        raise ValueError(f"sycl matmul dims must be >= 1; got m={m} k={k} n={n}")
    return (
        f"/* BCIR -> SYCL device (2-D parallel_for) differential oracle: row-major C[{m}x{n}] = "
        f"A[{m}x{k}] . B[{k}x{n}]; the SYCL device path (parallel_for over sycl::range<2>({m}, {n}), each "
        f"work-item accumulates the k-dot) when compiled with a SYCL toolchain (-fsycl, BCIR_USE_SYCL), a "
        f"portable triple-loop C++ matmul otherwise -- both reproduce kbcir.matmul.matmul_reference (same "
        f"row-major layout + sequential k-accumulation). NOTE: SYCL is a compiler MODE (-fsycl), NOT a "
        f"'c.call.libm' -l<lib> edge (unlike the B5 CBLAS gemm wrap). */\n"
        "#include <cstddef>\n"
        "#if defined(BCIR_USE_SYCL)\n"
        "  #include <sycl/sycl.hpp>\n"
        "  #define BCIR_MATMUL_SYCL 1\n"
        "#else\n"
        "  #define BCIR_MATMUL_SYCL 0\n"
        "#endif\n"
        f"void {fn_name}(const float *A, const float *B, float *C) {{\n"
        "#if BCIR_MATMUL_SYCL\n"
        f"  /* SYCL device path: USM device pointers + one 2-D parallel_for over the {m}x{n} output grid; "
        f"each (i, j) work-item accumulates the length-{k} dot in k-order. */\n"
        "  sycl::queue q;\n"
        f"  const std::size_t M = {m}, K = {k}, N = {n};\n"
        "  float *dA = sycl::malloc_device<float>(M * K, q);\n"
        "  float *dB = sycl::malloc_device<float>(K * N, q);\n"
        "  float *dC = sycl::malloc_device<float>(M * N, q);\n"
        "  q.memcpy(dA, A, M * K * sizeof(float)).wait();\n"
        "  q.memcpy(dB, B, K * N * sizeof(float)).wait();\n"
        "  q.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> idx) {\n"
        "    const std::size_t i = idx[0], j = idx[1];\n"
        "    float s = 0.0f;\n"
        "    for (std::size_t kk = 0; kk < K; ++kk)\n"
        "      s += dA[i * K + kk] * dB[kk * N + j];   /* C[i,j] = sum_k A[i,k]*B[k,j] */\n"
        "    dC[i * N + j] = s;\n"
        "  }).wait();\n"
        "  q.memcpy(C, dC, M * N * sizeof(float)).wait();\n"
        "  sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);\n"
        "#else\n"
        f"  /* Portable triple-loop C++ fallback (no SYCL header): the IDENTICAL row-major product. */\n"
        f"  for (std::size_t i = 0; i < {m}; ++i)\n"
        f"    for (std::size_t j = 0; j < {n}; ++j) {{\n"
        "      float s = 0.0f;\n"
        f"      for (std::size_t kk = 0; kk < {k}; ++kk)\n"
        f"        s += A[i * {k} + kk] * B[kk * {n} + j];\n"
        f"      C[i * {n} + j] = s;\n"
        "    }\n"
        "#endif\n"
        "}\n"
    )


# --- G7: gem.conv kernel (a 2-D conv == a structured matmul; reference-verified vs the naive direct conv) -
# A 2-D single-group convolution lowered to portable C. Following the B5 "BCIR owns the calling side"
# pattern, the emitted kernel computes the IDENTICAL result as the naive direct conv (kbcir.conv.
# conv2d_reference) -- the realization (direct stream vs im2col+gemm) is a cost-model choice that never
# changes the math, so the emit is the clean, transparent direct loop the reference defines (bit-exact for
# integer/exact inputs, float round-off otherwise). The conv arithmetic is pure multiply-accumulate -- NO
# transcendental, so it stays on the deterministic rail (no libm; like the exact relu / the gemm fallback).

def emit_conv2d_c(in_c: int, in_h: int, in_w: int, out_c: int, kh: int, kw: int,
                  stride: int = 1, pad: int = 0, fn_name: str = "bcir_conv2d") -> str:
    """Emit a portable C kernel for a single-group 2-D convolution (cross-correlation, the ML convention),
    reproducing ``kbcir.conv.conv2d_reference`` exactly. Layout is NCHW-flat: ``in`` is C_in*H*W row-major,
    ``W`` is C_out*C_in*Kh*Kw row-major, ``out`` is C_out*out_h*out_w row-major. Zero padding is implicit
    (an index outside the H x W frame contributes 0). Dims are baked in (a planned claim knows its shape).

    The kernel is the direct loop -- a pure multiply-accumulate, NO transcendental, NO libm (the conv stays
    on the deterministic rail, like the exact relu / the gemm reference fallback). The im2col+gemm
    realization (kbcir.conv) computes the IDENTICAL products, so this transparent direct emit IS the
    reference both realizations are verified against."""
    if min(in_c, in_h, in_w, out_c, kh, kw) < 1:
        raise ValueError(f"conv dims must be >= 1; got C_in={in_c} H={in_h} W={in_w} C_out={out_c} "
                         f"Kh={kh} Kw={kw}")
    if stride < 1 or pad < 0:
        raise ValueError(f"conv needs stride >= 1 (got {stride}) and pad >= 0 (got {pad})")
    out_h = (in_h + 2 * pad - kh) // stride + 1
    out_w = (in_w + 2 * pad - kw) // stride + 1
    if out_h < 1 or out_w < 1:
        raise ValueError(f"conv output is empty (out_h={out_h}, out_w={out_w})")
    return (
        f"/* BCIR -> gem.conv 2-D (single-group cross-correlation; reference-verified vs the naive direct "
        f"conv). NCHW-flat: in[{in_c}x{in_h}x{in_w}], W[{out_c}x{in_c}x{kh}x{kw}], out[{out_c}x{out_h}x"
        f"{out_w}]; stride={stride} pad={pad}. Pure MAC -- no transcendental, deterministic rail. */\n"
        "#include <stddef.h>\n"
        f"void {fn_name}(const float *restrict in, const float *restrict W, float *restrict out) {{\n"
        f"  for (size_t oc = 0; oc < {out_c}; ++oc)\n"
        f"    for (long oy = 0; oy < {out_h}; ++oy)\n"
        f"      for (long ox = 0; ox < {out_w}; ++ox) {{\n"
        f"        float s = 0.0f;\n"
        f"        for (size_t ic = 0; ic < {in_c}; ++ic)\n"
        f"          for (long ky = 0; ky < {kh}; ++ky) {{\n"
        f"            long iy = oy * {stride} + ky - {pad};\n"
        f"            if (iy < 0 || iy >= {in_h}) continue;        /* implicit zero padding */\n"
        f"            for (long kx = 0; kx < {kw}; ++kx) {{\n"
        f"              long ix = ox * {stride} + kx - {pad};\n"
        f"              if (ix < 0 || ix >= {in_w}) continue;\n"
        f"              float wv = W[((oc * {in_c} + ic) * {kh} + ky) * {kw} + kx];\n"
        f"              s += in[(ic * {in_h} + iy) * {in_w} + ix] * wv;\n"
        f"            }}\n"
        f"          }}\n"
        f"        out[(oc * {out_h} + oy) * {out_w} + ox] = s;\n"
        f"      }}\n"
        f"}}\n"
    )


# --- G7: gem.attention kernel (single-head scaled-dot-product; the softmax exp rides the c.call.libm: edge) -
# Single-head scaled-dot-product attention out = softmax(Q@K^T/sqrt(d))@V, lowered to portable C and
# reproducing kbcir.attention.attention_reference. Attention decomposes into the existing ops: two matmuls
# + a scale (exact, deterministic) + the EXISTING softmax (its exp is the only transcendental). The kernel
# routes that exp through the trusted ``c.call.libm:`` edge (`<math.h>` expf, -lm) exactly as the activation
# softmax / B5 do -- the transcendental stays OFF the deterministic legality rail. So the emit matches the
# reference bit-exactly on the matmul/scale, to float round-off through the softmax.

def emit_attention_c(seq_len: int, d_k: int, fn_name: str = "bcir_attention") -> str:
    """Emit a portable C kernel for single-head scaled-dot-product attention ``out = softmax(Q@K^T/
    sqrt(d_k))@V``, reproducing ``kbcir.attention.attention_reference``. Q, K, V, out are each row-major
    ``seq_len x d_k``. The softmax is the numerically-stable reduce-max -> exp -> normalize form (the SAME
    as the activation softmax / inference kernel), and its exp rides the trusted ``c.call.libm:`` edge
    (expf, link -lm). The two matmuls + the 1/sqrt(d_k) scale are exact/deterministic. Dims baked in.

    BCIR owns the calling side (the seq x d_k layout, the scale, the stable softmax); the only external is
    the trusted libm expf -- so the emit equals the reference bit-exactly on the matmul/scale and to float
    round-off through the softmax (the quarantine boundary, exactly as activation.softmax)."""
    if seq_len < 1 or d_k < 1:
        raise ValueError(f"attention dims must be >= 1; got seq_len={seq_len} d_k={d_k}")
    return (
        f"/* BCIR -> gem.attention single-head scaled-dot-product: out = softmax(Q@K^T/sqrt(d))@V. "
        f"seq_len={seq_len} d_k={d_k}; Q,K,V,out are seq_len x d_k row-major. The two matmuls + scale are "
        f"exact; the softmax exp rides the c.call.libm: edge (expf, -lm) -- the quarantine boundary. */\n"
        "#include <stddef.h>\n#include <math.h>\n"
        f"void {fn_name}(const float *restrict Q, const float *restrict K,\n"
        f"             const float *restrict V, float *restrict out) {{\n"
        f"  const size_t n = {seq_len}u, d = {d_k}u;\n"
        f"  const float scale = 1.0f / sqrtf((float)d);   /* 1/sqrt(d_k), the scaled-dot normalizer */\n"
        f"  float A[{seq_len} * {seq_len}];               /* the seq x seq attention weights */\n"
        f"  for (size_t i = 0; i < n; ++i) {{\n"
        f"    /* row i of the scaled scores S' = (Q @ K^T) * scale */\n"
        f"    float m = -INFINITY;\n"
        f"    for (size_t j = 0; j < n; ++j) {{\n"
        f"      float s = 0.0f;\n"
        f"      for (size_t t = 0; t < d; ++t) s += Q[i * d + t] * K[j * d + t];   /* Q_i . K_j */\n"
        f"      s *= scale;\n"
        f"      A[i * n + j] = s;\n"
        f"      if (s > m) m = s;                          /* reduce-max (softmax stability shift) */\n"
        f"    }}\n"
        f"    float sum = 0.0f;\n"
        f"    for (size_t j = 0; j < n; ++j) {{ A[i * n + j] = expf(A[i * n + j] - m); sum += A[i * n + j]; }}  /* c.call.libm:expf */\n"
        f"    if (sum == 0.0f) sum = 1.0f;\n"
        f"    for (size_t j = 0; j < n; ++j) A[i * n + j] /= sum;   /* normalize -> softmax row */\n"
        f"    /* out row i = A_i @ V (the context: seq x d_k) */\n"
        f"    for (size_t t = 0; t < d; ++t) {{\n"
        f"      float c = 0.0f;\n"
        f"      for (size_t j = 0; j < n; ++j) c += A[i * n + j] * V[j * d + t];\n"
        f"      out[i * d + t] = c;\n"
        f"    }}\n"
        f"  }}\n"
        f"}}\n"
    )


def emit_layernorm_c(rows: int, dim: int, fn_name: str = "bcir_layernorm") -> str:
    """E3 (ML-breadth): emit a portable C LAYERNORM kernel -- the ONE new numeric primitive of the full
    Transformer block (``bcir/kbcir/transformer.py``). Per-ROW over the ``dim`` feature axis, reproducing
    ``kbcir.transformer.layernorm_reference``:

        mean = (1/dim) * sum_c x[r,c]
        var  = (1/dim) * sum_c (x[r,c] - mean)^2          (POPULATION variance: /dim, the layernorm convention)
        out[r,c] = gamma[c] * (x[r,c] - mean) / sqrtf(var + eps) + beta[c]

    ``x``/``out`` are ``rows x dim`` row-major; ``gamma``/``beta`` are length ``dim``. The ONLY transcendental
    is the ``1/sqrtf(var + eps)`` -- it rides the trusted ``c.call.libm:sqrtf`` FFI edge (link ``-lm``, which the
    cfront link rail ALREADY maps: ``library_for_callee("sqrtf") == "-lm"`` -- NO linkflags change). The mean /
    variance sums and the per-channel scale/shift are exact/deterministic. So this is the layernorm analog of
    the GSL ``sd`` wrap and the attention softmax: a portable, self-contained kernel whose single external is
    the trusted libm rsqrt, matching the oracle reference to float round-off (float32 C vs float64 oracle).

    WHY THIS IS THE ONLY NEW KERNEL (the honest-depth contract): a transformer block is a COMPOSITION of ops
    BCIR already ships -- the per-head matmuls + scale + softmax are the EXISTING ``emit_attention_c`` single-head
    attention C twin, the projections / feed-forward are the EXISTING matmul emitters, and the residual adds are
    plain arithmetic. Only the layernorm had no C twin, so the block's executable seam is this one new kernel
    plus the already-tested existing twins. ``dim==1`` is degenerate (variance 0, so the normalized term is 0 +
    beta); it is allowed (eps keeps the sqrt well-defined), but layernorm over a single channel is uninformative.

    Signature ``void {fn}(const float *x, const float *gamma, const float *beta, float eps, float *out)`` with
    ``rows``/``dim`` baked in (a planned claim knows its shape). BCIR owns the calling side (the row-major layout,
    the population /dim variance, the eps floor); the trusted libm ``sqrtf`` is the sole external. ``out`` may NOT
    alias ``x`` (the kernel reads each row's x while writing out -- give distinct buffers)."""
    if rows < 1:
        raise ValueError(f"layernorm rows must be >= 1; got {rows}")
    if dim < 1:
        raise ValueError(f"layernorm dim must be >= 1; got {dim}")
    return (
        f"/* BCIR -> gem.layernorm (E3 Transformer block's one NEW numeric primitive): per-row over the dim "
        f"feature axis, out = gamma*(x-mean)/sqrtf(var+eps) + beta. rows={rows} dim={dim}; x,out are rows x dim "
        f"row-major, gamma/beta length dim. POPULATION variance (/dim, the layernorm rule). The only "
        f"transcendental is the 1/sqrtf(var+eps) -- it rides the c.call.libm: edge (sqrtf, -lm; already mapped, "
        f"no linkflags change). The mean/variance sums + the scale/shift are exact. Matches "
        f"kbcir.transformer.layernorm_reference to float round-off. */\n"
        "#include <stddef.h>\n#include <math.h>\n"
        f"void {fn_name}(const float *restrict x, const float *restrict gamma,\n"
        f"             const float *restrict beta, float eps, float *restrict out) {{\n"
        f"  const size_t rows = {rows}u, dim = {dim}u;\n"
        f"  for (size_t r = 0; r < rows; ++r) {{\n"
        f"    const float *xr = x + r * dim;\n"
        f"    float mean = 0.0f;\n"
        f"    for (size_t c = 0; c < dim; ++c) mean += xr[c];\n"
        f"    mean /= (float)dim;                            /* row mean */\n"
        f"    float var = 0.0f;\n"
        f"    for (size_t c = 0; c < dim; ++c) {{ float d = xr[c] - mean; var += d * d; }}\n"
        f"    var /= (float)dim;                             /* POPULATION variance (/dim, the layernorm rule) */\n"
        f"    float inv = 1.0f / sqrtf(var + eps);           /* the rsqrt -- the c.call.libm:sqrtf edge */\n"
        f"    for (size_t c = 0; c < dim; ++c)\n"
        f"      out[r * dim + c] = gamma[c] * (xr[c] - mean) * inv + beta[c];   /* scale + shift (exact) */\n"
        f"  }}\n"
        f"}}\n"
    )


def emit_lstm_cell_c(input_dim: int, hidden_dim: int, fn_name: str = "bcir_lstm_cell") -> str:
    """E4 (ML-breadth): emit a portable C LSTM CELL forward -- the recurrent-cell executable seam, reproducing
    ``kbcir.recurrent.lstm_cell_reference``. ONE step of a standard LSTM, per hidden unit ``u`` with gate
    pre-activations ``a_g = W_g x + U_g h_prev + b_g``:

        f = sigmoid(a_f) ; i = sigmoid(a_i) ; o = sigmoid(a_o) ; g = tanhf(a_g)
        c[u] = f * c_prev[u] + i * g
        h[u] = o * tanhf(c[u])

    ``x`` is length ``input_dim``; ``h_prev`` / ``c_prev`` / ``h`` / ``c`` are length ``hidden_dim``; each gate's
    ``W_*`` is ``hidden_dim x input_dim`` row-major and ``U_*`` is ``hidden_dim x hidden_dim`` row-major, with a
    length-``hidden_dim`` bias ``b_*`` -- the gates are passed grouped (f, i, o, g) in that order. The ONLY
    transcendentals are ``tanhf`` and the ``expf`` inside the sigmoid -- BOTH ride the trusted ``c.call.libm:``
    FFI edge (link ``-lm``, which the cfront link rail ALREADY maps: ``library_for_callee("tanhf") ==
    library_for_callee("expf") == "-lm"`` -- NO linkflags change). The gate matmuls / adds and the elementwise
    products are exact/deterministic. So this is the recurrent-cell analog of the layernorm / attention-softmax
    wraps: a portable, self-contained kernel whose only externals are the trusted libm tanh/exp, matching the
    oracle reference (``lstm_cell_reference``) to float round-off (float32 C vs float64 oracle).

    WHY LSTM IS THE C SEAM (the honest-depth contract): of E4's two tiers, Tier A (the closed-set relu-RNN) is
    already lowerable through the EXISTING autodiff/closed-set machinery (relu = select, exact, no libm), so the
    net-new executable seam is the TRANSCENDENTAL tier -- and the LSTM cell is its canonical, gate-rich
    representative (four sigmoids/tanh + the cell-state recurrence). The sigmoid is the numerically-guarded
    ``x >= 0 ? 1/(1+expf(-x)) : ex/(1+ex)`` form (mirrors ``recurrent.sigmoid``) so large ``|x|`` never overflows.

    Signature ``void {fn}(const float *x, const float *h_prev, const float *c_prev, const float *Wf, const float
    *Uf, const float *bf, ... (i, o, g) ..., float *h, float *c)`` with ``input_dim`` / ``hidden_dim`` baked in (a
    planned claim knows its shape). BCIR owns the calling side (the row-major layout, the gate order, the guarded
    sigmoid); the trusted libm ``tanhf`` / ``expf`` are the sole externals. ``h`` / ``c`` are distinct output
    buffers and may NOT alias the inputs."""
    if input_dim < 1 or hidden_dim < 1:
        raise ValueError(f"lstm cell dims must be >= 1; got input_dim={input_dim} hidden_dim={hidden_dim}")
    return (
        f"/* BCIR -> gem.recurrent LSTM cell (E4 recurrent-cell C seam): one step, per unit "
        f"f=sigmoid(Wf x+Uf h+bf), i,o likewise, g=tanhf(Wg x+Ug h+bg), c=f*c_prev+i*g, h=o*tanhf(c). "
        f"input_dim={input_dim} hidden_dim={hidden_dim}; x len input_dim, h_prev/c_prev/h/c len hidden_dim, "
        f"W_* hidden_dim x input_dim, U_* hidden_dim x hidden_dim, b_* len hidden_dim (gates f,i,o,g). The only "
        f"transcendentals are tanhf + the expf in the sigmoid -- both ride the c.call.libm: edge (tanhf/expf, "
        f"-lm; already mapped, no linkflags change). The gate matmuls/adds + products are exact. Matches "
        f"kbcir.recurrent.lstm_cell_reference to float round-off. */\n"
        "#include <stddef.h>\n#include <math.h>\n"
        "/* numerically-guarded logistic sigmoid (no overflow for large |x|; mirrors recurrent.sigmoid). */\n"
        f"static float {fn_name}_sigmoid(float z) {{\n"
        f"  if (z >= 0.0f) {{ float e = expf(-z); return 1.0f / (1.0f + e); }}   /* c.call.libm:expf */\n"
        f"  float e = expf(z); return e / (1.0f + e);                          /* c.call.libm:expf */\n"
        f"}}\n"
        f"void {fn_name}(const float *restrict x, const float *restrict h_prev,\n"
        f"             const float *restrict c_prev,\n"
        f"             const float *restrict Wf, const float *restrict Uf, const float *restrict bf,\n"
        f"             const float *restrict Wi, const float *restrict Ui, const float *restrict bi,\n"
        f"             const float *restrict Wo, const float *restrict Uo, const float *restrict bo,\n"
        f"             const float *restrict Wg, const float *restrict Ug, const float *restrict bg,\n"
        f"             float *restrict h, float *restrict c) {{\n"
        f"  const size_t di = {input_dim}u, dh = {hidden_dim}u;\n"
        f"  for (size_t u = 0; u < dh; ++u) {{\n"
        f"    float af = bf[u], ai = bi[u], ao = bo[u], ag = bg[u];\n"
        f"    for (size_t j = 0; j < di; ++j) {{        /* W_* x : the input-weight matmul rows (exact) */\n"
        f"      float xj = x[j];\n"
        f"      af += Wf[u * di + j] * xj; ai += Wi[u * di + j] * xj;\n"
        f"      ao += Wo[u * di + j] * xj; ag += Wg[u * di + j] * xj;\n"
        f"    }}\n"
        f"    for (size_t k = 0; k < dh; ++k) {{        /* U_* h_prev : the recurrent-weight matmul rows (exact) */\n"
        f"      float hk = h_prev[k];\n"
        f"      af += Uf[u * dh + k] * hk; ai += Ui[u * dh + k] * hk;\n"
        f"      ao += Uo[u * dh + k] * hk; ag += Ug[u * dh + k] * hk;\n"
        f"    }}\n"
        f"    float fgate = {fn_name}_sigmoid(af);     /* forget gate  */\n"
        f"    float igate = {fn_name}_sigmoid(ai);     /* input gate   */\n"
        f"    float ogate = {fn_name}_sigmoid(ao);     /* output gate  */\n"
        f"    float gcand = tanhf(ag);                  /* candidate -- c.call.libm:tanhf */\n"
        f"    float cu = fgate * c_prev[u] + igate * gcand;     /* new cell state (exact given the gates) */\n"
        f"    c[u] = cu;\n"
        f"    h[u] = ogate * tanhf(cu);                 /* new hidden state -- c.call.libm:tanhf */\n"
        f"  }}\n"
        f"}}\n"
    )


# --- E5: classical-ML PREDICT-path kernels (the baked-model fixed-shape predict; G5 baked-weights pattern) ---
# Classical ML splits sharply: TRAINING (tree induction, the SVM QP solve, NB fitting) is iterative/
# combinatorial -- a POOR fit for BCIR's fixed-shape claim model (library/Python). PREDICT over a BAKED model is
# the opposite: a deterministic, fixed-shape kernel -- the G5 baked-weights inference pattern. These two emitters
# show the Area-B pattern covers BOTH a transcendental predictor (RBF-SVM: the only external is expf on the
# c.call.libm: edge, -lm; the dot/distance sums are exact) and an exact one (the tree: pure comparisons + a leaf
# return, NO transcendental). BCIR owns the calling side (the row-major layout, the baked params); libsvm is the
# canonical SVM library in the framing only -- the decision function is emitted self-contained.

def emit_svm_rbf_predict_c(n_sv: int, n_feat: int, fn_name: str = "bcir_svm_rbf") -> str:
    """E5 (ML-breadth): emit a portable C RBF-SVM DECISION-FUNCTION predict kernel -- the classical-ML
    transcendental-predictor seam, reproducing ``kbcir.classical.svm_decision_rbf``:

        f(x) = Sum_i alpha_y[i] * expf(-gamma * ||x - SV[i]||^2) + b

    ``x`` is length ``n_feat``; ``SV`` is ``n_sv x n_feat`` row-major (``SV[i*n_feat + j]``); ``alpha_y`` is the
    per-SV dual coefficient ALREADY MULTIPLIED by the label (``alpha_i * y_i``), length ``n_sv``; ``b`` is the
    bias; ``gamma > 0`` is the RBF width. The ONLY transcendental is the ``expf`` -- it rides the trusted
    ``c.call.libm:`` FFI edge (link ``-lm``, which the cfront link rail ALREADY maps:
    ``library_for_callee("expf") == "-lm"`` -- NO linkflags change). The squared distance ``||x - SV[i]||^2`` and
    the dual-weighted sum are EXACT/deterministic. ``expf(-gamma*d^2)`` underflows to 0 for a far SV (fine -- a
    distant support vector contributes ~0); there is no overflow (the argument is <= 0). So this is the
    classical-ML analog of the LSTM / attention-softmax wraps: a portable, self-contained kernel whose only
    external is the trusted libm exp, matching the oracle reference (``svm_decision_rbf``) to float round-off
    (float32 C vs float64 oracle). This IS libsvm's RBF ``svm_predict`` -- emitted DIRECTLY (no libsvm
    dependency), per the "integrate, don't reinvent" framing (libsvm is the canonical library; the math is small
    + self-contained so we emit it).

    Signature ``float {fn}(const float *x, const float *SV, const float *alpha_y, float b, float gamma)`` with
    ``n_sv`` / ``n_feat`` baked in (a planned claim knows its shape). The caller takes ``sign()`` of the result
    for the class (``svm_predict``)."""
    if n_sv < 1 or n_feat < 1:
        raise ValueError(f"svm rbf dims must be >= 1; got n_sv={n_sv} n_feat={n_feat}")
    return (
        f"/* BCIR -> classical-ML PREDICT (E5): RBF-SVM decision function (G5 baked-model fixed-shape kernel). "
        f"f(x)=Sum_i alpha_y[i]*expf(-gamma*||x-SV[i]||^2)+b. n_sv={n_sv} n_feat={n_feat}; x len n_feat, SV n_sv x "
        f"n_feat row-major, alpha_y len n_sv (=alpha_i*y_i), b bias, gamma>0. The ONLY transcendental is expf -- "
        f"it rides the c.call.libm: edge (expf, -lm; already mapped, no linkflags change); the squared distances "
        f"+ the dual-weighted sum are exact. This is libsvm's RBF svm_predict, emitted self-contained. Matches "
        f"kbcir.classical.svm_decision_rbf to float round-off. */\n"
        "#include <stddef.h>\n#include <math.h>\n"
        f"float {fn_name}(const float *restrict x, const float *restrict SV,\n"
        f"             const float *restrict alpha_y, float b, float gamma) {{\n"
        f"  const size_t n_sv = {n_sv}u, n_feat = {n_feat}u;\n"
        f"  float acc = b;\n"
        f"  for (size_t i = 0; i < n_sv; ++i) {{\n"
        f"    float d2 = 0.0f;                          /* ||x - SV[i]||^2 (exact) */\n"
        f"    for (size_t j = 0; j < n_feat; ++j) {{\n"
        f"      float d = x[j] - SV[i * n_feat + j];\n"
        f"      d2 += d * d;\n"
        f"    }}\n"
        f"    acc += alpha_y[i] * expf(-gamma * d2);    /* the RBF kernel -- c.call.libm:expf (the sole transcendental) */\n"
        f"  }}\n"
        f"  return acc;\n"
        f"}}\n"
    )


def emit_tree_predict_c(n_nodes: int, n_feat: int, fn_name: str = "bcir_tree") -> str:
    """E5 (ML-breadth): emit a portable C DECISION-TREE predict kernel -- the classical-ML EXACT-predictor seam
    (the counterpart to the RBF-SVM transcendental one), reproducing ``kbcir.classical.tree_predict``. Traverse
    the baked flat-array tree from the root (node 0): at an INTERNAL node go ``left`` if ``x[feature] <=
    threshold`` else ``right``; stop at a LEAF (``feature[node] == -1``) and return its ``leaf_value``.

    EXACT -- pure comparisons + a leaf-constant return, NO transcendental, NO libm (integer/float-exact, 0 ULP).
    So this kernel needs NO ``-lm`` and mints NO ``c.call.libm:`` edge -- it is the EXACT half of the Area-B
    pattern (the RBF-SVM kernel is the transcendental half), showing the classical-ML predict path covers both.
    The baked arrays are passed in: ``feature`` (``int``, ``-1`` = leaf), ``threshold`` / ``leaf_value``
    (``float``), ``left`` / ``right`` (``int`` child node ids), each length ``n_nodes``; ``x`` is length
    ``n_feat``. A depth bound (``n_nodes + 1`` hops) guards a malformed cyclic tree (a well-formed CART tree's
    children have a strictly larger node id, so it terminates well within the bound). Matches ``tree_predict``
    EXACTLY (it is the same comparisons + leaf return -- integer/float-exact, not merely to round-off).

    Signature ``float {fn}(const float *x, const int *feature, const float *threshold, const int *left, const int
    *right, const float *leaf_value)`` with ``n_nodes`` / ``n_feat`` baked in. The caller maps the returned leaf
    value to a class id (or uses it directly for regression)."""
    if n_nodes < 1 or n_feat < 1:
        raise ValueError(f"tree dims must be >= 1; got n_nodes={n_nodes} n_feat={n_feat}")
    return (
        f"/* BCIR -> classical-ML PREDICT (E5): decision-tree threshold traversal (G5 baked-model fixed-shape "
        f"kernel; the EXACT half of the Area-B pattern -- NO transcendental, NO libm). From the root, go left if "
        f"x[feature[node]] <= threshold[node] else right, until feature[node]==-1 (a leaf) -> return "
        f"leaf_value[node]. n_nodes={n_nodes} n_feat={n_feat}; x len n_feat, the tree arrays len n_nodes. Pure "
        f"comparisons + a leaf return (integer/float-exact, 0 ULP). Matches kbcir.classical.tree_predict "
        f"exactly. */\n"
        "#include <stddef.h>\n"
        f"float {fn_name}(const float *restrict x, const int *restrict feature,\n"
        f"             const float *restrict threshold, const int *restrict left,\n"
        f"             const int *restrict right, const float *restrict leaf_value) {{\n"
        f"  const size_t n_nodes = {n_nodes}u;\n"
        f"  size_t node = 0u;\n"
        f"  for (size_t hop = 0; hop <= n_nodes; ++hop) {{   /* bounded: at most n_nodes hops to a leaf */\n"
        f"    if (feature[node] == -1) return leaf_value[node];   /* a leaf -- the prediction (exact) */\n"
        f"    if (x[feature[node]] <= threshold[node]) node = (size_t)left[node];\n"
        f"    else node = (size_t)right[node];\n"
        f"  }}\n"
        f"  return leaf_value[node];   /* depth-bound fallthrough (malformed tree); return the current node's value */\n"
        f"}}\n"
    )


# --- E6: unsupervised-learning kernel (K-means nearest-centroid ASSIGN: the exact predict kernel) ----
# E6 (unsupervised + the data pipeline) mirrors E5's train-vs-predict split: K-means FIT is a bounded iterative
# optimization (Lloyd's, library/Python-shaped), while ASSIGN over BAKED centroids is the fixed-shape PREDICT
# kernel -- the G5 baked-weights pattern. The assign kernel is the EXACT half of the Area-B pattern (like the
# decision tree): pure subtract/multiply/add + comparisons (the squared distance + an argmin), NO transcendental,
# NO libm -- so the C-vs-oracle check is INTEGER-EXACT (the same argmin cluster id), not merely to round-off. C
# twin of kbcir.unsupervised.kmeans_assign.

def emit_kmeans_assign_c(k: int, n_feat: int, fn_name: str = "bcir_kmeans_assign") -> str:
    """E6 (ML-breadth): emit a portable C K-MEANS nearest-centroid ASSIGN kernel -- the unsupervised-learning
    EXACT-predictor seam (the analog of E5's exact decision-tree kernel), reproducing
    ``kbcir.unsupervised.kmeans_assign``:

        argmin_c  ||x - centroid[c]||^2        (the nearest centroid by SQUARED Euclidean distance)

    ``x`` is length ``n_feat``; ``centroids`` is ``k x n_feat`` row-major (``centroids[c*n_feat + j]``). Ranking
    on the squared distance is IDENTICAL to ranking on distance (``sqrt`` is monotone), so the assignment needs
    NO transcendental -- pure subtract/multiply/add + a ``<`` comparison. DETERMINISTIC tie-break: the LOWEST
    centroid index wins on a distance tie (the strict ``<`` only updates the best on a STRICTLY smaller
    distance). EXACT -- 0 ULP, NO libm, needs NO ``-lm`` and mints NO ``c.call.libm:`` edge (it is the EXACT
    half of the Area-B pattern, like the decision tree). Returns an ``int`` cluster id, so the C-vs-oracle check
    is INTEGER-EXACT (the same argmin) as long as the test point is off an exact equidistant tie -- a float32
    (C) vs float64 (oracle) difference can never flip the integer argmin away from a tie.

    Signature ``int {fn}(const float *x, const float *centroids)`` with ``k`` / ``n_feat`` baked in (a planned
    claim knows its shape). The caller uses the returned cluster id directly. Matches ``kmeans_assign``
    exactly."""
    if k < 1 or n_feat < 1:
        raise ValueError(f"kmeans assign dims must be >= 1; got k={k} n_feat={n_feat}")
    return (
        f"/* BCIR -> unsupervised K-means ASSIGN (E6): nearest-centroid argmin (G5 baked-model fixed-shape "
        f"kernel; the EXACT half of the Area-B pattern -- NO transcendental, NO libm). argmin_c "
        f"||x-centroid[c]||^2 by squared Euclidean distance (= ranking on distance, sqrt monotone -> no "
        f"transcendental). k={k} n_feat={n_feat}; x len n_feat, centroids k x n_feat row-major. Tie-break: "
        f"lowest centroid index (strict < updates only on a STRICTLY smaller distance). Returns an int cluster "
        f"id -- integer-exact vs the oracle (same argmin). Matches kbcir.unsupervised.kmeans_assign exactly. */\n"
        "#include <stddef.h>\n"
        f"int {fn_name}(const float *restrict x, const float *restrict centroids) {{\n"
        f"  const size_t k = {k}u, n_feat = {n_feat}u;\n"
        f"  int best_c = 0;\n"
        f"  float best_d = 0.0f;                          /* ||x - centroid[0]||^2 (exact) */\n"
        f"  for (size_t j = 0; j < n_feat; ++j) {{ float d = x[j] - centroids[j]; best_d += d * d; }}\n"
        f"  for (size_t c = 1; c < k; ++c) {{\n"
        f"    float dist = 0.0f;                          /* ||x - centroid[c]||^2 (exact) */\n"
        f"    for (size_t j = 0; j < n_feat; ++j) {{\n"
        f"      float d = x[j] - centroids[c * n_feat + j];\n"
        f"      dist += d * d;\n"
        f"    }}\n"
        f"    if (dist < best_d) {{ best_d = dist; best_c = (int)c; }}   /* strict < : a tie keeps the LOWER index */\n"
        f"  }}\n"
        f"  return best_c;\n"
        f"}}\n"
    )


# --- G1: gem.activation kernels (relu exact / the transcendental four via the c.call.libm: edge) ----
# The activation analog of the B5 BLAS wrap. relu is integer/Q-fixed CLEAN -- a pure max(0,x), emitted
# with NO transcendental and NO libm dependency (exact, 0 ULP, valid for f32 OR i32). The transcendental
# four (sigmoid/tanh/gelu/softmax) route their exp/tanh through the SAME `c.call.libm:` FFI edge B5 used
# for cblas_sgemm: the C kernel `#include <math.h>` and calls the trusted `expf`/`tanhf` (link with -lm),
# keeping the transcendental fully OFF the deterministic legality rail (libm is opaque/trusted, like any
# external edge). All are f32 (the libm edge returns float); relu additionally supports i32.

def emit_relu_kernel_c(n: int, fn_name: str = "bcir_relu", elem: str = "f32") -> str:
    """Emit the EXACT relu kernel `C[i] = max(0, A[i])` -- the integer/Q-fixed-clean activation. No
    transcendental, no libm, 0-ULP exact on f32 OR i32 (a pure comparison/select the compiler lowers to a
    branchless max). This is the activation that stays on the deterministic rail."""
    if n < 1:
        raise ValueError(f"relu length must be >= 1; got {n}")
    ctype = _ctype(elem)
    includes = "#include <stddef.h>\n" + ("#include <stdint.h>\n" if elem == "i32" else "")
    zero = "0" if elem == "i32" else "0.0f"
    return (
        f"/* BCIR -> gem.activation relu (EXACT, integer/Q-fixed clean: max(0,x), 0 ULP, no "
        f"transcendental). elem={ctype} n={n}. */\n"
        f"{includes}"
        f"void {fn_name}(const {ctype} *restrict A, {ctype} *restrict C, size_t n) {{\n"
        f"  for (size_t i = 0; i < n; ++i)\n"
        f"    C[i] = A[i] > {zero} ? A[i] : {zero};\n"
        f"}}\n"
    )


def emit_activation_kernel_c(kind: str, n: int, fn_name: str = "bcir_activation",
                             axis_len: int | None = None) -> str:
    """Emit a gem.activation C kernel. ``relu`` -> the exact `emit_relu_kernel_c` (f32). The transcendental
    four route their exp/tanh through the trusted ``c.call.libm:`` edge (`<math.h>` `expf`/`tanhf`, -lm),
    exactly as B5's `emit_blas_gemm_c` wraps `cblas_sgemm` -- the transcendental is the trusted external,
    BCIR owns the calling side (the elementwise/last-axis layout). All transcendental kernels are f32.

    Shapes/forms (each matches the oracle reference in kbcir.activation):
      sigmoid: C[i] = 1/(1+expf(-A[i]))
      tanh:    C[i] = tanhf(A[i])
      gelu:    C[i] = 0.5*A[i]*(1+tanhf(sqrt(2/pi)*(A[i]+0.044715*A[i]^3)))   (the GPT/BERT tanh form)
      softmax: per row of `axis_len`: m=max(row); e=expf(x-m); C=e/sum(e)     (the stable reduce-max form)
    """
    from ..kbcir.activation import _check_kind, libm_edges
    _check_kind(kind)
    if n < 1:
        raise ValueError(f"activation length must be >= 1; got {n}")
    if kind == "relu":
        return emit_relu_kernel_c(n, fn_name, elem="f32")

    edges = libm_edges(kind)                       # the trusted libm symbols this kernel calls
    head = (f"/* BCIR -> gem.activation {kind} via the c.call.libm: edge ({', '.join(edges)}; trusted "
            f"external, BCIR owns the calling side -- the B5 wrap pattern). f32 n={n}. */\n"
            "#include <stddef.h>\n#include <math.h>\n")

    if kind == "sigmoid":
        body = (f"void {fn_name}(const float *restrict A, float *restrict C, size_t n) {{\n"
                f"  for (size_t i = 0; i < n; ++i)\n"
                f"    C[i] = 1.0f / (1.0f + expf(-A[i]));   /* c.call.libm:expf */\n"
                f"}}\n")
    elif kind == "tanh":
        body = (f"void {fn_name}(const float *restrict A, float *restrict C, size_t n) {{\n"
                f"  for (size_t i = 0; i < n; ++i)\n"
                f"    C[i] = tanhf(A[i]);                   /* c.call.libm:tanhf */\n"
                f"}}\n")
    elif kind == "gelu":
        body = (f"void {fn_name}(const float *restrict A, float *restrict C, size_t n) {{\n"
                f"  const float c = sqrtf(2.0f / 3.14159265358979323846f);\n"
                f"  for (size_t i = 0; i < n; ++i) {{\n"
                f"    float x = A[i];\n"
                f"    C[i] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));  /* c.call.libm:tanhf */\n"
                f"  }}\n"
                f"}}\n")
    else:  # softmax
        ax = axis_len if axis_len is not None else n
        if ax < 1 or n % ax != 0:
            raise ValueError(f"softmax: axis_len {ax} must be >= 1 and divide n {n}")
        body = (f"void {fn_name}(const float *restrict A, float *restrict C, size_t n) {{\n"
                f"  const size_t ax = {ax}u;   /* the last-axis (reduction) length */\n"
                f"  for (size_t r = 0; r < n; r += ax) {{\n"
                f"    float m = A[r];\n"
                f"    for (size_t j = 1; j < ax; ++j) if (A[r + j] > m) m = A[r + j];  /* reduce-max */\n"
                f"    float s = 0.0f;\n"
                f"    for (size_t j = 0; j < ax; ++j) {{ C[r + j] = expf(A[r + j] - m); s += C[r + j]; }}  /* c.call.libm:expf */\n"
                f"    if (s == 0.0f) s = 1.0f;\n"
                f"    for (size_t j = 0; j < ax; ++j) C[r + j] /= s;                  /* normalize */\n"
                f"  }}\n"
                f"}}\n")
    return head + body


# --- G2: fused matmul+activation epilogue kernel (the inline activation, no intermediate buffer) -----
# The optimizer (kbcir.fusion) chooses this fused form when it prices cheaper than the two-pass
# (matmul-then-activation) plan -- the win is eliding the intermediate C-buffer's write+read. The kernel
# applies the activation to each matmul output element s INLINE, just before the store: C[i*N+j] = act(s).
# relu stays the EXACT clean-rail max(0,s); the transcendentals inline the SAME trusted libm call
# (expf/tanhf) the standalone activation kernel uses (the c.call.libm: edge, quarantine rules carried over).
# softmax is NOT supported here (a last-axis row reduction can not be an inline per-element epilogue -- it
# is scoped out of fusion in kbcir.fusion).

def _inline_activation_expr(kind: str, var: str) -> tuple[str, bool]:
    """The C expression applying activation `kind` INLINE to the float scalar `var` (the matmul output
    element), plus whether it needs ``<math.h>``. relu -> the exact ``(var > 0.0f ? var : 0.0f)`` (no libm);
    the transcendentals -> the SAME expf/tanhf form the standalone ``emit_activation_kernel_c`` uses, so the
    fused result equals the unfused reference to float round-off. softmax is rejected (non-elementwise)."""
    if kind == "relu":
        return f"({var} > 0.0f ? {var} : 0.0f)", False
    if kind == "sigmoid":
        return f"(1.0f / (1.0f + expf(-({var}))))", True            # c.call.libm:expf
    if kind == "tanh":
        return f"tanhf({var})", True                                # c.call.libm:tanhf
    if kind == "gelu":
        # the GPT/BERT tanh-approximation GELU, identical to emit_activation_kernel_c's gelu form.
        return (f"(0.5f * ({var}) * (1.0f + tanhf(0.7978845608028654f * "
                f"(({var}) + 0.044715f * ({var}) * ({var}) * ({var})))))"), True   # c.call.libm:tanhf
    raise ValueError(f"{kind!r} is not an inline-fusible activation (softmax is a row reduction; relu/"
                     f"sigmoid/tanh/gelu fuse)")


def emit_matmul_activation_c(M: int, N: int, K: int, kind: str = "relu",
                             fn_name: str = "bcir_matmul_act") -> str:
    """G2: a FUSED matmul -> activation epilogue kernel -- ``C[i,j] = act( (A@B)[i,j] )`` with the activation
    applied INLINE as each output element is produced and NO intermediate buffer materialized (the whole
    point of the fusion: the un-activated product never round-trips to memory). This is what the tropical
    optimizer (kbcir.fusion) emits when it prices the fused plan cheaper than matmul-then-activation.

    relu inlines the EXACT clean-rail ``max(0,s)`` (no libm); sigmoid/tanh/gelu inline the SAME trusted
    ``c.call.libm:`` call (expf/tanhf) the standalone activation kernel uses, so the fused output EQUALS the
    unfused reference (bit-exact for relu, float round-off for the transcendentals).

    The matmul part uses the REFERENCE triple loop (NOT the ``#if defined(BCIR_USE_CBLAS)`` cblas path of
    ``emit_blas_gemm_c``): cblas writes the WHOLE C buffer before returning, so there is no per-element seam
    to splice the inline activation into -- the fusion's defining property (no intermediate materialized) is
    only expressible on the open reference loop, where ``s`` is in a register exactly when the activation
    needs it. A cblas-backed variant would have to either materialize C and run a second pass (the UNFUSED
    plan we are beating) or use a cblas epilogue callback (not portable C23). So the fused path is the
    reference loop by construction -- documented here, the trade the fusion makes."""
    if M < 1 or N < 1 or K < 1:
        raise ValueError(f"matmul+act dims must be >= 1; got M={M} N={N} K={K}")
    expr, needs_math = _inline_activation_expr(kind, "s")
    includes = "#include <stddef.h>\n" + ("#include <math.h>\n" if needs_math else "")
    note = (f"relu: EXACT inline max(0,s), no libm" if kind == "relu"
            else f"{kind}: inline via the c.call.libm: edge (the same trusted call the standalone kernel "
                 f"uses); fused == unfused reference to float round-off")
    return (
        f"/* BCIR -> FUSED gem.matmul + gem.activation({kind}) epilogue (optimizer's priced choice; "
        f"kbcir.fusion). row-major C[{M}x{N}] = act(A[{M}x{K}] @ B[{K}x{N}]); the activation is applied "
        f"INLINE per output element -- NO intermediate buffer materialized. {note}. */\n"
        f"{includes}"
        f"void {fn_name}(const float *restrict A, const float *restrict B, float *restrict C) {{\n"
        f"  for (size_t i = 0; i < {M}; ++i)\n"
        f"    for (size_t j = 0; j < {N}; ++j) {{\n"
        f"      float s = 0.0f;\n"
        f"      for (size_t k = 0; k < {K}; ++k) s += A[i * {K} + k] * B[k * {N} + j];\n"
        f"      C[i * {N} + j] = {expr};   /* the inline activation epilogue (no separate pass) */\n"
        f"    }}\n"
        f"}}\n"
    )


def emit_qfixed_selfcheck_c(module: Module, result: RealizationResult,
                            fn_name: str = "bcir_qfixed", lane_bits: int = 16,
                            frac_bits: int = 8) -> str:
    """Wrap the Q-fixed kernel with a self-checking `main`: it computes the reference
    in a 64-bit accumulator and asserts the kernel matches at the selected count and a
    tail-exercising size. In-range inputs keep the Q-fixed result within `lane_bits`,
    so the `_BitInt(N)` and standard-int fallback builds are bit-identical."""
    claim, _ = find_elementwise(module, result)
    n = max(1, claim.count)
    is_mul = claim.opcode == Opcode.MUL
    op = C_OP[claim.opcode]
    kernel = emit_qfixed_kernel_c(module, result, fn_name, lane_bits, frac_bits)
    ref = (f"((int64_t)A[i] * (int64_t)B[i]) >> {frac_bits}" if is_mul
           else f"(int64_t)A[i] {op} (int64_t)B[i]")
    # Inputs use ~half the lane so the product>>frac and the sum stay within lane_bits.
    half = lane_bits // 2
    return (
        "#include <stddef.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n\n"
        + kernel
        + f"""
static int check(size_t n) {{
  q_lane_t *A = malloc(n * sizeof *A), *B = malloc(n * sizeof *B), *C = malloc(n * sizeof *C);
  if (!A || !B || !C) {{ free(A); free(B); free(C); return 2; }}
  for (size_t i = 0; i < n; ++i) {{
    A[i] = (q_lane_t)((int64_t)(i % {1 << half}) - {1 << (half - 1)});
    B[i] = (q_lane_t)((int64_t)((i * 3 + 1) % {1 << half}) - {1 << (half - 1)});
    C[i] = (q_lane_t)0;
  }}
  {fn_name}(A, B, C, n);
  int rc = 0;
  for (size_t i = 0; i < n; ++i) {{
    int64_t want = {ref};
    if ((int64_t)C[i] != want) {{
      printf("FAIL n=%zu i=%zu got %lld want %lld\\n", n, i, (long long)C[i], (long long)want);
      rc = 1; break;
    }}
  }}
  free(A); free(B); free(C);
  return rc;
}}

int main(void) {{
  if (check({n}u)) return 1;        /* the selected count */
  if (check({n}u + 7u)) return 1;   /* a tail-exercising size */
  printf("OK {fn_name} bitint=%d\\n", BCIR_QFIXED_BITINT);
  return 0;
}}
"""
    )


def compile_and_run_qfixed_c(module: Module, result: RealizationResult,
                             fn_name: str = "bcir_qfixed", lane_bits: int = 16,
                             frac_bits: int = 8, std: str = "c23",
                             workdir: str | None = None) -> tuple[bool, str]:
    """Emit the Q-fixed self-check, compile under `-std=<std>` (c23 exercises the
    `_BitInt(N)` lanes; c11 exercises the portable fallback), run, and confirm it
    printed OK. Returns (ok, combined_output)."""
    cc = _which("clang") or _which("cc") or _which("gcc")
    if cc is None:
        return False, "no C compiler on PATH"
    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-qfixed-")
    try:
        src = os.path.join(workdir, "qfixed_check.c")
        exe = os.path.join(workdir, "qprog")
        with open(src, "w") as f:
            f.write(emit_qfixed_selfcheck_c(module, result, fn_name, lane_bits, frac_bits))
        build = subprocess.run([cc, f"-std={std}", "-O2", "-Wall", "-Wextra", src, "-o", exe],
                               capture_output=True, text=True)
        if build.returncode != 0:
            return False, "Q-fixed build failed:\n" + build.stdout + build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        ok = run.returncode == 0 and "OK" in run.stdout
        return ok, run.stdout + run.stderr
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def emit_gather_kernel_c(module: Module, result: RealizationResult,
                         fn_name: str = "bcir_gather", elem: str = "f32") -> str:
    """Emit the **gather** realization the cost model AVOIDS: an indexed read
    `C[i] = A[idx[i]] op B[i]` (an extra `const long *restrict idx`). This is the
    form a gather/scatter-unaware lowering would use; it pays `gather_penalty`
    (indexed loads + cache misses) on silicon. BCIR's cost model prefers the
    direct realization (`emit_kernel_c`) whenever the access is not random."""
    claim, _ = find_elementwise(module, result)
    op = C_OP[claim.opcode]
    ctype = _ctype(elem)
    includes = "#include <stddef.h>\n"
    fp_pragma = ""
    if elem == "i32":
        includes += "#include <stdint.h>\n"
    else:
        fp_pragma = "#pragma STDC FP_CONTRACT OFF\n"
    return (
        f"/* BCIR -> gather realization (cost-model-AVOIDED; pays gather_penalty). "
        f"op={claim.op or op} elem={ctype}. */\n"
        f"{includes}"
        f'static_assert(sizeof({ctype}) == 4, "BCIR {ctype} gather needs a 4-byte element");\n'
        f"{fp_pragma}\n"
        f"void {fn_name}(const {ctype} *restrict A, const {ctype} *restrict B,\n"
        f"             {ctype} *restrict C, const long *restrict idx, size_t n) {{\n"
        f"  for (size_t i = 0; i < n; ++i)\n"
        f"    C[i] = A[idx[i]] {op} B[i];\n"
        f"}}\n"
    )


def find_reduce(module: Module, result: RealizationResult) -> tuple:
    """Return (claim, candidate) for the selected `reduce.gather` claim (a
    reducible-permutation reduction with a blocked-vs-gather choice)."""
    by_claim = result.by_claim()
    for ph in module.phases:
        for claim in ph.claims:
            if claim.op == "reduce.gather":
                cand = by_claim.get(claim.id)
                if cand is not None:
                    return claim, cand
    raise NotImplementedError("no reduce.gather claim selected in this plan")


# --- CIM/PIM-aware spatial partitioning (optimize_spatial) ------------------------

@dataclass(frozen=True)
class SpatialBinding:
    claim_id: int
    target: str                # "pim" | "core"
    transport_bytes_saved: int  # operand bytes that stay in memory (host transport skipped)


@dataclass(frozen=True)
class SpatialPlan:
    bindings: tuple
    pim_capable: bool

    @property
    def offloaded(self) -> tuple:
        return tuple(b.claim_id for b in self.bindings if b.target == "pim")

    @property
    def total_bytes_saved(self) -> int:
        return sum(b.transport_bytes_saved for b in self.bindings)

    @property
    def is_noop(self) -> bool:
        return not self.offloaded


def is_pim_target(h) -> bool:
    """A target is PIM/CIM-capable iff its ISA features advertise `pim` (e.g.
    `TargetProfile(..., isa_features=frozenset({"pim"}))`)."""
    return "pim" in getattr(h, "isa_features", frozenset())


def optimize_spatial(module: Module, result: RealizationResult, h) -> SpatialPlan:
    """Spatial layout pass: on a **PIM-capable** target, bind a reduction straight to
    the memory controller -- the reduce runs in memory and only the scalar result
    crosses the bus, skipping the host<-device transport of the read stream -- when
    `cim_decision` says it wins. Everything else stays on the core. The reported
    saving is the operand bytes that never move (`count * elem_bytes`).

    This is **modeled**: there is no real PIM device in the sandbox, so it does not
    emit device-controller code -- it produces the binding decision and the
    transport-avoided figure. On a non-PIM target every claim stays on the core (a
    clean no-op). Next-phase work (a real PIM target + emitter) is tracked in
    docs/HARDWARE_VALIDATION.md."""
    from ..gem.cim import cim_decision    # lazy: keep gem.cim off the simple emit path
    pim = is_pim_target(h)
    by_claim = result.by_claim()
    bindings: list[SpatialBinding] = []
    for ph in module.phases:
        for claim in ph.claims:
            if claim.id not in by_claim:
                continue
            target, saved = "core", 0
            if pim and (claim.op or "").startswith("reduce.") and claim.rd:
                res = module.resource(claim.rd[0])
                eb = res.elem_bytes if res else 4
                if cim_decision(claim.op, claim.count, eb, h).offload:
                    target, saved = "pim", claim.count * eb   # read stream stays in memory
            bindings.append(SpatialBinding(claim.id, target, saved))
    return SpatialPlan(bindings=tuple(bindings), pim_capable=pim)


def emit_reduce_c(module: Module, result: RealizationResult, fn_name: str = "bcir_reduce",
                  elem: str = "i32", gather: bool | None = None) -> str:
    """Lower a `reduce.gather` claim to C: `acc = sum_i T[i]` (blocked) or
    `acc = sum_i T[idx[i]]` (gather), per the K_BCIR-selected realization. Integer
    by default -- integer addition is associative, so the blocked (sequential) and
    gather (permuted) sums are bit-identical, which is what makes the gather
    avoidance a *correct* transformation (the win is pure: same answer, no random
    access). `gather=None` follows the selected candidate."""
    claim, cand = find_reduce(module, result)
    if gather is None:
        gather = cand.name == "gather"
    ctype = _ctype(elem)
    includes = "#include <stddef.h>\n" + ("#include <stdint.h>\n" if elem == "i32" else "")
    idx_param = ", const long *restrict idx" if gather else ""
    read = "T[idx[i]]" if gather else "T[i]"
    mode = "gather" if gather else "blocked"
    return (
        f"/* BCIR -> reduce.gather lowering ({mode}; K_BCIR-selected={cand.name}). "
        f"integer sum is order-independent, so blocked == gather exactly. */\n"
        f"{includes}\n"
        f"{ctype} {fn_name}(const {ctype} *restrict T{idx_param}, size_t n) {{\n"
        f"  {ctype} acc = 0;\n"
        f"  for (size_t i = 0; i < n; ++i) acc += {read};\n"
        f"  return acc;\n"
        f"}}\n"
    )


def emit_compensated_reduce_c(fn_name: str = "bcir_reduce_comp") -> str:
    """Emit a **compensated** Q8 reduction kernel (`precision="compensated"`) -- the
    C lowering of `kbcir.precision.compensated_reduce_q8`. A Q8 multiply-accumulate
    `acc = (sum_i T[i]*weight) >> 8` with a *residual carry*: the truncated low 8 bits of
    each term are carried forward (the integer Kahan/TwoSum analog), so the loop invariant
    `acc*256 + resid == sum(T[i]*weight so far)` holds and the result is **bit-identical to
    the int64-exact reduction** -- with no wide final accumulator. The naive per-term-
    truncating form (`emit_reduce_c`) drifts below it by up to `n` ULP.

    Determinism note: requires arithmetic (floor) right shift on signed ints -- the C23
    semantics, and universal on real targets -- so `full >> 8` matches the oracle's `>>`."""
    return (
        f"/* BCIR -> compensated Q8 reduction (precision=\"compensated\"; residual-carry MAC). "
        f"acc*256 + resid == sum(T[i]*weight) so far -> bit-identical to the int64-exact "
        f"reduction (vs naive per-term truncation, which drifts up to n ULP). */\n"
        "#include <stddef.h>\n#include <stdint.h>\n"
        '_Static_assert((-256 >> 8) == -1, "BCIR compensated reduction needs arithmetic '
        '(floor) signed right shift");\n'
        f"int32_t {fn_name}(const int32_t *restrict T, int32_t weight, size_t n) {{\n"
        f"  int32_t acc = 0, resid = 0;   /* resid is the carried low 8 bits, in [0, 255] */\n"
        f"  for (size_t i = 0; i < n; ++i) {{\n"
        f"    int64_t full = (int64_t)T[i] * (int64_t)weight + resid;\n"
        f"    acc += (int32_t)(full >> 8);\n"
        f"    resid = (int32_t)(full & 0xFF);\n"
        f"  }}\n"
        f"  return acc;\n"
        f"}}\n"
    )


def emit_compensated_selfcheck_c(fn_name: str = "bcir_reduce_comp", n: int = 1000) -> str:
    """Wrap the compensated kernel with a self-checking `main`: it computes the
    compensated, naive, and int64-exact reductions over the same Q8 data and asserts the
    compensated result is **bit-identical to exact**, the naive form **drifts** below it
    (demonstrating the win), and the drift never exceeds the `n`-ULP bound the accuracy
    contract relies on. The inputs (weight < 1.0 in Q8) guarantee per-term truncation, so
    the naive accumulator visibly loses precision."""
    kernel = emit_compensated_reduce_c(fn_name)
    return (
        "#include <stddef.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n\n"
        + kernel
        + f"""
static int64_t naive(const int32_t *T, int32_t w, size_t n) {{
  int64_t acc = 0;                          /* per-term truncation: acc += (T[i]*w) >> 8 */
  for (size_t i = 0; i < n; ++i) acc += ((int64_t)T[i] * w) >> 8;
  return acc;
}}
static int64_t exact(const int32_t *T, int32_t w, size_t n) {{
  int64_t s = 0;                            /* full width, shift once at the end */
  for (size_t i = 0; i < n; ++i) s += (int64_t)T[i] * w;
  return s >> 8;
}}

static int check(size_t n) {{
  int32_t *T = malloc(n * sizeof *T);
  if (!T) return 2;
  for (size_t i = 0; i < n; ++i) T[i] = (int32_t)(1 + (i % 5));   /* small Q8 values */
  const int32_t w = 200;                    /* 0.78 in Q8 -> every term truncates */
  int64_t comp = {fn_name}(T, w, n), nv = naive(T, w, n), ex = exact(T, w, n);
  free(T);
  if (comp != ex) {{ printf("FAIL n=%zu comp=%lld exact=%lld\\n", n, (long long)comp, (long long)ex); return 1; }}
  if (nv > ex) {{ printf("FAIL n=%zu naive=%lld > exact=%lld\\n", n, (long long)nv, (long long)ex); return 1; }}
  if (ex - nv > (int64_t)n) {{ printf("FAIL n=%zu drift=%lld > n\\n", n, (long long)(ex - nv)); return 1; }}
  return (ex - nv > 0) ? 0 : 3;             /* the naive form must visibly drift */
}}

int main(void) {{
  if (check({n}u)) return 1;
  if (check({n}u + 7u)) return 1;
  puts("OK {fn_name}");
  return 0;
}}
"""
    )


def compile_and_run_compensated_c(fn_name: str = "bcir_reduce_comp", n: int = 1000,
                                  std: str = "c23", workdir: str | None = None) -> tuple[bool, str]:
    """Emit the compensated-reduction self-check, compile under `-std=<std>`, run, and
    confirm it printed OK (compensated == exact, naive drifts within the n-ULP bound)."""
    cc = _which("clang") or _which("cc") or _which("gcc")
    if cc is None:
        return False, "no C compiler on PATH"
    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-comp-")
    try:
        src = os.path.join(workdir, "comp_check.c")
        exe = os.path.join(workdir, "cprog")
        with open(src, "w") as f:
            f.write(emit_compensated_selfcheck_c(fn_name, n))
        build = subprocess.run([cc, f"-std={std}", "-O2", "-Wall", "-Wextra", src, "-o", exe],
                               capture_output=True, text=True)
        if build.returncode != 0:
            return False, "compensated build failed:\n" + build.stdout + build.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        ok = run.returncode == 0 and "OK" in run.stdout
        return ok, run.stdout + run.stderr
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def find_strided(module: Module, result: RealizationResult) -> tuple:
    """Return (claim, candidate) for the selected STRIDED claim (a strided-vs-gather
    choice -- the cost model picks the direct strided realization over GGG)."""
    by_claim = result.by_claim()
    for ph in module.phases:
        for claim in ph.claims:
            if claim.stride_class == StrideClass.STRIDED:
                cand = by_claim.get(claim.id)
                if cand is not None:
                    return claim, cand
    raise NotImplementedError("no strided claim selected in this plan")


def emit_strided_c(module: Module, result: RealizationResult, fn_name: str = "bcir_strided",
                   elem: str = "f32", gather: bool | None = None) -> str:
    """Lower a STRIDED claim `Y[i] = X[i*k]` to C: the **direct strided** access
    (the cost model's pick) or the **gather** alternative `Y[i] = X[idx[i]]` (a
    gather-unaware lowering, idx[i] = i*k -- the same elements through indexed
    loads). A non-reduction gather avoidance: BCIR knows the access is strided, so
    it emits direct addressing and avoids the slower gather instruction.
    `gather=None` follows the selected candidate."""
    claim, cand = find_strided(module, result)
    if gather is None:
        gather = cand.name == "gather"
    k = max(1, claim.stride_k)
    ctype = _ctype(elem)
    inc = "#include <stddef.h>\n" + ("#include <stdint.h>\n" if elem == "i32" else "")
    if gather:
        return (
            f"/* BCIR -> strided lowering (gather; K_BCIR-selected={cand.name}). */\n"
            f"{inc}\n"
            f"void {fn_name}(const {ctype} *restrict X, const long *restrict idx,\n"
            f"             {ctype} *restrict Y, size_t n) {{\n"
            f"  for (size_t i = 0; i < n; ++i) Y[i] = X[idx[i]];\n"
            f"}}\n"
        )
    return (
        f"/* BCIR -> strided lowering (direct, stride {k}; K_BCIR-selected={cand.name}). */\n"
        f"{inc}\n"
        f"void {fn_name}(const {ctype} *restrict X, {ctype} *restrict Y, size_t n, size_t k) {{\n"
        f"  for (size_t i = 0; i < n; ++i) Y[i] = X[i * k];\n"
        f"}}\n"
    )


def emit_header_c(fn_name: str = "bcir_kernel", elem: str = "f32") -> str:
    """A freestanding C23 header declaring the kernel ABI -- the stable contract a
    driver/runtime compiles the emitted kernel against (no BCIR dependency)."""
    ctype = _ctype(elem)
    guard = f"BCIR_{fn_name.upper()}_H"
    includes = "#include <stddef.h>\n"
    if elem == "i32":
        includes += "#include <stdint.h>\n"
    return (
        f"/* BCIR kernel ABI (freestanding, generated). Stable contract for a "
        f"resident toolchain. */\n"
        f"#ifndef {guard}\n#define {guard}\n"
        f"{includes}\n"
        f"/* Elementwise C = A op B over n elements; A,B,C are non-overlapping. */\n"
        f"void {fn_name}(const {ctype} *restrict A, const {ctype} *restrict B,\n"
        f"             {ctype} *restrict C, size_t n);\n"
        f"#endif /* {guard} */\n"
    )


def emit_selfcheck_c(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel",
                     elem: str = "f32", hw_width: int | None = None) -> str:
    """Wrap the kernel with a self-checking C23 `main` (the AOT path). Checks the
    selected count and a non-divisible size (count + 7) to exercise the tail.
    `hw_width` is forwarded so the self-check validates the exact deployed kernel
    (the go-fast form at the full lane)."""
    claim, _ = find_elementwise(module, result)
    n = max(1, claim.count)
    op = C_OP[claim.opcode]
    ctype = _ctype(elem)
    kernel = emit_kernel_c(module, result, fn_name, elem, hw_width=hw_width)
    if elem == "i32":
        init = "A[i] = (int32_t)i; B[i] = (int32_t)(2u * i); C[i] = -1;"
        fmt = "%d"
    else:
        init = "A[i] = (float)i; B[i] = 2.0f * (float)i; C[i] = -1.0f;"
        fmt = "%g"

    return (
        "#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n\n"
        + kernel
        + f"""
static int check(size_t n) {{
  {ctype} *A = malloc(n * sizeof *A), *B = malloc(n * sizeof *B), *C = malloc(n * sizeof *C);
  if (!A || !B || !C) {{ free(A); free(B); free(C); return 2; }}
  for (size_t i = 0; i < n; ++i) {{ {init} }}
  {fn_name}(A, B, C, n);
  int rc = 0;
  for (size_t i = 0; i < n; ++i) {{
    {ctype} want = A[i] {op} B[i];
    if (C[i] != want) {{ printf("FAIL n=%zu i=%zu got {fmt} want {fmt}\\n", n, i, C[i], want); rc = 1; break; }}
  }}
  free(A); free(B); free(C);
  return rc;
}}

int main(void) {{
  if (check({n}u)) return 1;        /* the selected count */
  if (check({n}u + 7u)) return 1;   /* a non-divisible size: exercises the scalar tail */
  puts("OK {fn_name}");
  return 0;
}}
"""
    )


def compile_and_run_c(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel",
                      elem: str = "f32", workdir: str | None = None,
                      hw_width: int | None = None) -> tuple[bool, str]:
    """Emit the self-checking C, compile it (C23: clang -std=c23, else gcc -std=c2x),
    run, and check it printed OK. Returns (ok, combined_output). `hw_width` selects
    the width-aware form so the self-check validates the deployed kernel."""
    cc = _which("clang") or _which("cc") or _which("gcc")
    if cc is None:
        return False, "no C compiler on PATH"

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-ckernel-")
    try:
        src = os.path.join(workdir, "kernel_check.c")
        exe = os.path.join(workdir, "prog")
        with open(src, "w") as f:
            f.write(emit_selfcheck_c(module, result, fn_name, elem, hw_width=hw_width))
        build = None
        for std in ("-std=c23", "-std=c2x"):
            build = subprocess.run([cc, std, "-O2", "-Wall", "-Wextra", src, "-o", exe],
                                   capture_output=True, text=True)
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return False, "C build failed:\n" + (build.stdout + build.stderr if build else "")
        run = subprocess.run([exe], capture_output=True, text=True)
        ok = run.returncode == 0 and "OK" in run.stdout
        return ok, run.stdout + run.stderr
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def _which(name: str) -> str | None:
    from shutil import which
    return which(name)
