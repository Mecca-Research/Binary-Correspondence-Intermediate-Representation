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
