/* Native float32 tensor kernels. Row-major spans are caller-owned, non-overlapping
 * except the documented in-place vector/RoPE operations. Dimensions and capacities
 * are admitted by bcir_decoder_plan before these hot kernels are called. No heap,
 * global state, fast-math, framework dependency or device discovery; the one host
 * query is the CPU's instruction-set support, which only chooses among bit-identical
 * variants of the same kernels. Providers complete all writes before returning and
 * never throw across the C ABI. */
#ifndef BCIR_TENSOR_H
#define BCIR_TENSOR_H
#include <stddef.h>
#ifndef BCIR_DT_API
#if defined(_WIN32) && defined(BCIR_DT_BUILD_SHARED)
#define BCIR_DT_API __declspec(dllexport)
#else
#define BCIR_DT_API
#endif
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef void (*bcir_tensor_mm_fn)(void *ctx, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c);
typedef void (*bcir_tensor_silu_fn)(void *ctx, size_t n, const float *x, float *y);
typedef struct bcir_tensor_provider {
  void *ctx;
  bcir_tensor_mm_fn mm;       /* NULL selects portable C; same transpose/alpha/beta contract */
  bcir_tensor_silu_fn silu;   /* NULL selects stable libm C; computes x*sigmoid(x), same VJP */
} bcir_tensor_provider;
BCIR_DT_API void bcir_tensor_mm(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c);
/* A caller-owned deterministic thread pool (NDT-THREADS). A kernel that takes one splits its work
 * into tasks whose partition is a function of the shapes alone, never of the thread count, and
 * every output element -- and every reduction -- is produced by one task in the serial order, so
 * the result is bit-identical at every thread count and to the serial kernel (a NULL pool, or a
 * pool of one). The pool lives in memory the caller provides -- bcir_tensor_pool_size bytes,
 * aligned to max_align_t -- so nothing here allocates; its threads are OS resources, started by
 * init and joined by stop, after which the memory is the caller's again. pthreads on POSIX,
 * Win32 threads on Windows; elsewhere only threads == 1 has a size. A pool is used by one caller
 * at a time, like the state that names it. */
typedef struct bcir_tensor_pool bcir_tensor_pool;
BCIR_DT_API int bcir_tensor_pool_rails(void);   /* 1 pthreads, 2 Win32, 0 serial only */
BCIR_DT_API size_t bcir_tensor_pool_size(unsigned threads); /* 0: a count this host refuses */
BCIR_DT_API bcir_tensor_pool *bcir_tensor_pool_init(void *memory, size_t bytes, unsigned threads);
BCIR_DT_API void bcir_tensor_pool_stop(bcir_tensor_pool *pool); /* joins; NULL or stopped: no-op */
BCIR_DT_API unsigned bcir_tensor_pool_threads(const bcir_tensor_pool *pool); /* NULL: 1 */
/* Run `tasks` tasks of `fn` over the pool's threads -- task t on thread t mod threads, each
 * thread its tasks in ascending order -- and return when every one has finished; a NULL pool,
 * one thread or one task runs them in order on the caller. A caller's partition keeps every
 * output element, and every reduction, with one task, as the kernels below do. */
typedef void (*bcir_tensor_task_fn)(void *ctx, size_t task);
BCIR_DT_API void bcir_tensor_pool_run(bcir_tensor_pool *pool, bcir_tensor_task_fn fn, void *ctx,
    size_t tasks);
BCIR_DT_API void bcir_tensor_mm_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c);
BCIR_DT_API void bcir_tensor_silu_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y);
BCIR_DT_API void bcir_tensor_swiglu_backward_pool(bcir_tensor_pool *pool, size_t n,
    const float *gate, const float *up, const float *dy, float *dg, float *du);
BCIR_DT_API void bcir_tensor_rms_pool(bcir_tensor_pool *pool, size_t rows, size_t width,
    float eps, const float *x, const float *g, float *y);
BCIR_DT_API void bcir_tensor_rope_pool(bcir_tensor_pool *pool, size_t batch, size_t time,
    size_t heads, size_t dim, double base, float *x, int inverse);
/* y[i] += x[i] and y[i] *= x[i] over elements, as the trainer's residual and gate paths do. */
BCIR_DT_API void bcir_tensor_add_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y);
BCIR_DT_API void bcir_tensor_mul_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y);
BCIR_DT_API void bcir_tensor_attention_pool(bcir_tensor_pool *pool, size_t batch, size_t time,
    size_t heads, size_t kvheads, size_t dim, const float *q, const float *k, const float *v,
    float *p, float *y);
BCIR_DT_API void bcir_tensor_attention_backward_inplace_pool(bcir_tensor_pool *pool,
    size_t batch, size_t time, size_t heads, size_t kvheads, size_t dim, const float *q,
    const float *k, const float *v, float *p, const float *dy, float *dq, float *dk, float *dv);
/* The fused contract, v2 (NDT-FMA): c[i][j] starts at its beta-scaled value and then, for l
 * ascending, c = fma(alpha*A(i,l), B(l,j), c) -- the product alpha*A rounded as in v1, the term's
 * product and its addition rounded once together (C99 fmaf, an IEEE fusedMultiplyAdd). The same
 * nest as v1 with one operation per term in place of two, so a host with FMA hardware runs it at
 * twice the arithmetic rate; the portable instance calls fmaf, the AVX2+FMA and AVX-512
 * instances fuse in registers, all bit-identical to the fmaf loop. v1 stays the default of
 * every state, row and record; a state selects v2 by its contract field (bcir_decoder_train.h), and a
 * state trained under v2 is not the v1 state. */
BCIR_DT_API void bcir_tensor_mm_fused(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c);
BCIR_DT_API void bcir_tensor_mm_fused_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m,
    size_t n, size_t k, float alpha, const float *a, const float *b, float beta, float *c);
/* The fused instances this host can run: bit 1 portable (fmaf), bit 2 AVX2+FMA, bit 4 AVX-512F;
 * bcir_tensor_mm_fused_variant runs one, or returns 0 and writes nothing. */
BCIR_DT_API int bcir_tensor_mm_fused_variants(void);
BCIR_DT_API int bcir_tensor_mm_fused_variant(int variant, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c);
/* The blocked fused contract, v3 (NDT-KBLOCK): c[i][j] starts at its beta-scaled value; the
 * depth [0, k) is split into blocks by one rule -- while terms remain, a whole block of
 * BCIR_MM_KBLOCK terms when at least two blocks' worth remain, else half the remainder rounded
 * up to a multiple of BCIR_MM_KBLOCK_UNROLL when more than one block's worth remains, else the
 * remainder -- and for each block in order P = the fused chain of its terms from zero
 * (P = fma(A(i,l), B(l,j), P) for l ascending), then c = c + alpha*P (the product rounded, then
 * the addition). It is the arithmetic of a level-3 BLAS whose kernel accumulates each K panel
 * in registers and adds it to C: OpenBLAS's sgemm computes exactly this on its SkylakeX target
 * (Q 448, unroll 16) for alpha 1, so a provider that matches it on a differential can be
 * admitted under v3 (NativeDecoder.use_cblas), and a state trained under v3 is one state with
 * the provider or without it. For k <= BCIR_MM_KBLOCK, beta 0 and alpha 1 the result is v2's;
 * otherwise its own. bcir_tensor_kblock_split writes the rule's block lengths for k in order
 * (up to capacity) and returns their count. The instances are v3's own (NDT-BLOCK): the
 * generic one the plain-C block, the AVX-512 and AVX2 ones intrinsic register blocks of 8 x 48
 * and 6 x 16 accumulators over a tile packed by vector, held to the same chain bit for bit. */
#define BCIR_MM_KBLOCK 448
#define BCIR_MM_KBLOCK_UNROLL 16
BCIR_DT_API size_t bcir_tensor_kblock_split(size_t k, size_t *lengths, size_t capacity);
BCIR_DT_API void bcir_tensor_mm_kblock(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c);
BCIR_DT_API void bcir_tensor_mm_kblock_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m,
    size_t n, size_t k, float alpha, const float *a, const float *b, float beta, float *c);
BCIR_DT_API int bcir_tensor_mm_kblock_variants(void);
BCIR_DT_API int bcir_tensor_mm_kblock_variant(int variant, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c);
/* The BCIR exponential (NDT-EXP): one operation sequence on every instance -- a Cody-Waite
 * reduction x = k ln2 + r, a degree-7 polynomial in r, 2^k as two exact power-of-two scalings
 * (one rounding into the subnormals), the three special cases selected at the end -- so the
 * scalar, AVX2 and AVX-512 instances agree bit for bit, with no libm and no fma; within 1 ulp
 * of exp evaluated in double and rounded to binary32 (bcir_tensor_exp_sweep measures the bound).
 * SiLU, the SwiGLU VJP and the attention softmax under their v2 contracts use it (the sigmoid is
 * 1/(1+e) for x >= 0 and e*(1/(1+e)) for x < 0, e = exp(-|x|)); v1 keeps libm's expf. */
BCIR_DT_API float bcir_tensor_expf(float x);
BCIR_DT_API void bcir_tensor_exp(size_t n, const float *x, float *y);  /* the widest instance */
BCIR_DT_API int bcir_tensor_exp_variant(int variant, size_t n, const float *x, float *y);
/* Every binary32 in [-104, 89] at `stride` (a power of two) through every instance this host
 * runs: `mismatch` counts the inputs where two instances differ, `ulp` the largest distance from
 * exp in double rounded to binary32. Returns the inputs swept; 0 for a stride that is not a
 * power of two. */
BCIR_DT_API size_t bcir_tensor_exp_sweep(unsigned stride, size_t *mismatch, unsigned *ulp);
BCIR_DT_API void bcir_tensor_silu_v2_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y);
BCIR_DT_API void bcir_tensor_swiglu_backward_v2_pool(bcir_tensor_pool *pool, size_t n,
    const float *gate, const float *up, const float *dy, float *dg, float *du);
BCIR_DT_API void bcir_tensor_attention_v2_pool(bcir_tensor_pool *pool, size_t batch, size_t time,
    size_t heads, size_t kvheads, size_t dim, const float *q, const float *k, const float *v,
    float *p, float *y);
/* The portable GEMM's instruction-set variants: bit 1 portable C, bit 2 AVX2, bit 4 AVX-512F
 * (x86-64 Linux with GCC or Clang). Every variant computes the same rounded operations in the
 * same order, so all of them are bit-identical; bcir_tensor_mm runs the widest one the host
 * supports. bcir_tensor_mm_variant runs one variant and returns it, or returns 0 and writes
 * nothing when the host cannot run it. */
BCIR_DT_API int bcir_tensor_mm_variants(void);
BCIR_DT_API int bcir_tensor_mm_variant(int variant, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c);
/* Optional LP64 CBLAS adapter: ctx points to a caller-owned function pointer.
 * No BLAS headers or link dependency. The adapter uses portable C on shapes beyond
 * the LP64 provider's INT_MAX dimensions. */
typedef void (*bcir_cblas_sgemm_fn)(int, int, int, int, int, int, float,
    const float *, int, const float *, int, float, float *, int);
BCIR_DT_API void bcir_tensor_cblas_mm(void *ctx, int ta, int tb, size_t m, size_t n, size_t k,
    float alpha, const float *a, const float *b, float beta, float *c);
/* y = x W^T, W is out x in; dx overwritten or accumulated, dW accumulated. */
BCIR_DT_API void bcir_tensor_linear(const bcir_tensor_provider *p, size_t rows, size_t in,
    size_t out, const float *x, const float *w, float *y);
BCIR_DT_API void bcir_tensor_linear_backward(const bcir_tensor_provider *p, size_t rows,
    size_t in, size_t out, const float *x, const float *w, const float *dy,
    float *dx, float *dw, float dx_beta);
BCIR_DT_API void bcir_tensor_silu(void *ctx, size_t n, const float *x, float *y);
BCIR_DT_API void bcir_tensor_swiglu_backward(size_t n, const float *gate, const float *up,
    const float *dy, float *dg, float *du);
BCIR_DT_API void bcir_tensor_rms(size_t rows, size_t width, float eps, const float *x,
    const float *g, float *y);
BCIR_DT_API void bcir_tensor_rms_backward(size_t rows, size_t width, float eps, const float *x,
    const float *g, const float *dy, float *dx, float *dg);
/* Half-split Llama RoPE, [batch,time,head,channel]; inverse applies its transpose. */
BCIR_DT_API void bcir_tensor_rope(size_t batch, size_t time, size_t heads, size_t dim,
    double base, float *x, int inverse);
/* p: [batch,query-head,time,time]; causal future entries are exactly zero.
 * q/k/v: [batch,time,head,channel]. Backward sums all GQA groups into dk/dv. */
BCIR_DT_API void bcir_tensor_attention(size_t batch, size_t time, size_t heads, size_t kvheads,
    size_t dim, const float *q, const float *k, const float *v, float *p, float *y);
BCIR_DT_API void bcir_tensor_attention_backward(size_t batch, size_t time, size_t heads,
    size_t kvheads, size_t dim, const float *q, const float *k, const float *v,
    const float *p, const float *dy, float *dq, float *dk, float *dv);
/* The same dq, dk and dv, bit for bit, through the GEMM kernel; p is rewritten in place with
 * the score adjoint (p_ij (dp_ij - dot_i) / sqrt(dim)), so it no longer holds probabilities. */
BCIR_DT_API void bcir_tensor_attention_backward_inplace(size_t batch, size_t time,
    size_t heads, size_t kvheads, size_t dim, const float *q, const float *k, const float *v,
    float *p, const float *dy, float *dq, float *dk, float *dv);
#ifdef __cplusplus
}
#endif
#endif
