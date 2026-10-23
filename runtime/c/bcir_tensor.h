/* Native float32 tensor kernels. Row-major spans are caller-owned, non-overlapping
 * except the documented in-place vector/RoPE operations. Dimensions and capacities
 * are admitted by bcir_decoder_plan before these hot kernels are called. No heap,
 * global state, fast-math, framework dependency or device discovery; the one host
 * query is the CPU's instruction-set support, which only chooses among bit-identical
 * variants of the same kernels (and, for the bf16 contract's AMX instance, the OS's tile
 * permission and an admission differential, asked once and kept: see NDT-BF16 below).
 * Providers complete all writes before returning and never throw across the C ABI. */
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
/* Run `tasks` tasks of `fn` over the pool's threads -- each thread claiming the next unclaimed
 * task, in ascending order, where the compiler has GCC's or Clang's atomics, task t on thread
 * t mod threads elsewhere -- and return when every one has finished; a NULL pool, one thread or
 * one task runs them in order on the caller. Which thread runs a task is not part of the
 * contract: a caller's partition keeps every output element, and every reduction, with one task,
 * as the kernels below do. */
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
/* The RMSNorm VJP on the pool (NDT-REDUCE): rows in ranges of BCIR_ROW_CHUNK for dx and each row's
 * 1/rms, kept in `inv` (rows floats the caller owns for the call, disjoint from the others), then
 * columns in ranges of BCIR_COL_CHUNK for the gamma adjoint, each column's sum over the rows in row
 * order -- bcir_tensor_rms_backward's result, bit for bit, at every thread count. */
BCIR_DT_API void bcir_tensor_rms_backward_pool(bcir_tensor_pool *pool, size_t rows, size_t width,
    float eps, const float *x, const float *g, const float *dy, float *dx, float *dg, float *inv);
/* The same VJP with its two writes chosen (NDT-RMS-ACC, NDT-ZERO), every value the call above's:
 * BCIR_TENSOR_RMSB_ADD adds each element of dx to the value dx holds -- dx + (the VJP's element),
 * one rounded sum, which is the call above into a buffer t followed by bcir_tensor_add_pool(t, dx)
 * -- and BCIR_TENSOR_RMSB_FIRST starts each column's gamma adjoint at +0.0 instead of at the value
 * dg holds, as a zero fill of dg before the call would (dg is then written, never read). */
#define BCIR_TENSOR_RMSB_ADD 1u
#define BCIR_TENSOR_RMSB_FIRST 2u
BCIR_DT_API void bcir_tensor_rms_backward_into_pool(bcir_tensor_pool *pool, size_t rows,
    size_t width, float eps, const float *x, const float *g, const float *dy, float *dx, float *dg,
    float *inv, unsigned how);
BCIR_DT_API void bcir_tensor_rope_pool(bcir_tensor_pool *pool, size_t batch, size_t time,
    size_t heads, size_t dim, double base, float *x, int inverse);
/* y[i] += x[i] and y[i] *= x[i] over elements, as the trainer's residual and gate paths do. */
BCIR_DT_API void bcir_tensor_add_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y);
BCIR_DT_API void bcir_tensor_mul_pool(bcir_tensor_pool *pool, size_t n, const float *x, float *y);
/* y[i] = silu(x[i]) * u[i] in one pass (NDT-SWIGLU-FWD): the SiLU rounded to binary32 as
 * bcir_tensor_silu_pool computes it (libm's expf; the _v2 entry: the v2 SiLU, as
 * bcir_tensor_silu_v2_pool), then its product with u rounded -- the SiLU into y followed by
 * bcir_tensor_mul_pool(u, y), bit for bit, with one read of x and u and one write of y. Each
 * element reads x[i] and u[i] before it writes y[i]; y must not partly overlap x or u. */
BCIR_DT_API void bcir_tensor_silu_mul_pool(bcir_tensor_pool *pool, size_t n, const float *x,
    const float *u, float *y);
BCIR_DT_API void bcir_tensor_attention_pool(bcir_tensor_pool *pool, size_t batch, size_t time,
    size_t heads, size_t kvheads, size_t dim, const float *q, const float *k, const float *v,
    float *p, float *y);
/* The forward's writers with their verdict (NDT-SCAN-W): each computes what the entry it names
 * computes, bit for bit, and sets *bad to 1 when some element of the regions below is not finite
 * after the call, leaving *bad as it was otherwise -- so a caller ORs its calls' verdicts into one.
 * The tasks that wrote a region read it back while it is in their core's cache. A region's verdict
 * may come from another region its value reaches: the RMSNorm's regions are x and y, and it reads
 * y, since an infinite or NaN element of a row makes the row's 1/rms 0 or NaN and that element of
 * y NaN; the SwiGLU product's are x, u and y, and it reads y, since a non-finite gate makes the
 * SiLU infinite or NaN (-inf times the sigmoid's 0 is NaN) and a non-finite factor makes the
 * product so; the attention's are q, k and v (batch x time x heads or kvheads x dim), p and y, and
 * it reads k and y -- a query that is not finite makes every score of its row infinite or NaN and
 * so the row's probabilities NaN, and a NaN probability or a value that is not finite enters a
 * context sum, every key's own row taking its term; a key is the region the softmax can hide (a
 * key whose scores are all -inf weighs 0 in every row), so the keys are read themselves. */
BCIR_DT_API void bcir_tensor_rms_scan_pool(bcir_tensor_pool *pool, size_t rows, size_t width,
    float eps, const float *x, const float *g, float *y, int *bad);
BCIR_DT_API void bcir_tensor_silu_mul_scan_pool(bcir_tensor_pool *pool, size_t n, const float *x,
    const float *u, float *y, int *bad);
BCIR_DT_API void bcir_tensor_silu_mul_v2_scan_pool(bcir_tensor_pool *pool, size_t n,
    const float *x, const float *u, float *y, int *bad);
BCIR_DT_API void bcir_tensor_attention_scan_pool(bcir_tensor_pool *pool, size_t batch, size_t time,
    size_t heads, size_t kvheads, size_t dim, const float *q, const float *k, const float *v,
    float *p, float *y, int *bad);
BCIR_DT_API void bcir_tensor_attention_v2_scan_pool(bcir_tensor_pool *pool, size_t batch,
    size_t time, size_t heads, size_t kvheads, size_t dim, const float *q, const float *k,
    const float *v, float *p, float *y, int *bad);
BCIR_DT_API void bcir_tensor_attention_backward_inplace_pool(bcir_tensor_pool *pool,
    size_t batch, size_t time, size_t heads, size_t kvheads, size_t dim, const float *q,
    const float *k, const float *v, float *p, const float *dy, float *dq, float *dk, float *dv);
/* The attention under a contract the caller names (ATTN-FMA): BCIR_TENSOR_ATTENTION_EXP puts the
 * forward's softmax under the BCIR exponential (bcir_tensor_attention_v2_scan_pool's), and
 * BCIR_TENSOR_ATTENTION_FUSED every multiply-add of the products -- the scores q_i . k_j, the
 * context's p_ij v_j, the backward's probability adjoints dy_i . v_j, its dot sum p_ij dp_ij and
 * the dq, dk and dv sums -- one IEEE fusedMultiplyAdd (C99 fmaf: the product unrounded, one
 * rounding) in place of v1's rounded product and rounded sum, over the same terms in the same
 * order from the same start; the scaling, the softmax and the score adjoint p_ij (dp_ij - dot_i)
 * times the scale 1/sqrtf(dim) are v1's. Without FUSED each entry is its v1 entry above, bit for
 * bit; with it every instance (portable fmaf, AVX2+FMA and AVX-512 FMA, chosen as the fused GEMM's
 * are) is bit-identical to the fmaf loops at every thread count. BCIR_TENSOR_ATTENTION_SCRATCH
 * (INFER-ATTN) makes p the forward's scratch instead of its output, for a forward no backward
 * follows: each (batch, query head) runs its rows a strip at a time -- scores, softmax, context --
 * in its own slot at the start of p (bcir_tensor_attention_scratch_floats below), so a strip stays
 * in the core's cache, and y and the verdict are the stored attention's bit for bit; p's contents
 * after it are unspecified and no backward may read them (the bf16 entry below takes the bit
 * alike). Other bits are ignored, and the backward ignores EXP and SCRATCH. bad is the forward's
 * verdict as above, or NULL. */
#define BCIR_TENSOR_ATTENTION_EXP 1u
#define BCIR_TENSOR_ATTENTION_FUSED 2u
#define BCIR_TENSOR_ATTENTION_SCRATCH 8u
/* The floats of p a forward under BCIR_TENSOR_ATTENTION_SCRATCH writes, at its start: a slot of
 * whole rows a (batch, query head) -- at least 32 rows or all of them, about 64K floats -- the
 * strips running through it. A caller may lend just these; SIZE_MAX where the count overflows. */
BCIR_DT_API size_t bcir_tensor_attention_scratch_floats(size_t batch, size_t time, size_t heads);
BCIR_DT_API void bcir_tensor_attention_contract_scan_pool(bcir_tensor_pool *pool,
    unsigned contract, size_t batch, size_t time, size_t heads, size_t kvheads, size_t dim,
    const float *q, const float *k, const float *v, float *p, float *y, int *bad);
BCIR_DT_API void bcir_tensor_attention_backward_contract_pool(bcir_tensor_pool *pool,
    unsigned contract, size_t batch, size_t time, size_t heads, size_t kvheads, size_t dim,
    const float *q, const float *k, const float *v, float *p, const float *dy, float *dq,
    float *dk, float *dv);
/* The attention on bfloat16 operands (BF16-ATTN, BCIR_TENSOR_ATTENTION_BF16): its six products --
 * the scores S = Q K^T and the context Y = P V, the backward's dV = P^T dY, dP = dY V^T, dQ = dS K and
 * dK = dS^T Q -- under the bf16 GEMM contract below (the operands rounded to bfloat16, each 32-term
 * depth tile's two lanes from +0.0 added to c, flushed as the AMX unit flushes), each element over
 * the depth tiles the causal mask reaches from its 32-row block: a query block's key tiles 0 .. its
 * own (Y, dQ), a key block's query tiles from its own on (dV, dK), every channel tile (S, dP; their
 * blocks past the diagonal are not computed). A tile's terms past the mask are the +0.0
 * probabilities and score adjoints p holds there. The scaling, the softmax
 * (BCIR_TENSOR_ATTENTION_EXP and BCIR_TENSOR_ATTENTION_SCRATCH as above; other bits ignored), the
 * dot and the score adjoint p_ij (dp_ij - dot_i) scale are v1's in v1's order; the future
 * probabilities and adjoints are +0.0; a KV head's dk and dv take its query heads in order, the
 * first from +0.0; the forward's verdict is as above. `variant` names the
 * products' instance -- 1 portable, 4 AVX-512, 8 AMX, 0 the widest bcir_tensor_mm_bf16_variants
 * admits -- every instance the same bits at every thread count. The entries pack each task's
 * operands into the workspace -- one region a worker of the pool where the workers are no more than
 * the tasks, else one a task: bcir_tensor_attention_bf16_pool_workspace bytes for a pool of that
 * many threads (a NULL pool one), at most bcir_tensor_attention_bf16_workspace (one region a task,
 * the most any pool uses; a function of the shapes alone) -- and return the instance they ran, or
 * 0 -- writing nothing -- for an instance not admitted, kvheads not dividing heads, or a smaller
 * workspace. No value depends on which region a task packs into (WS-PLAN). */
#define BCIR_TENSOR_ATTENTION_BF16 4u
BCIR_DT_API size_t bcir_tensor_attention_bf16_workspace(size_t batch, size_t time, size_t heads,
    size_t kvheads, size_t dim);
BCIR_DT_API size_t bcir_tensor_attention_bf16_pool_workspace(size_t batch, size_t time,
    size_t heads, size_t kvheads, size_t dim, unsigned threads);
BCIR_DT_API int bcir_tensor_attention_bf16_scan_pool(bcir_tensor_pool *pool, int variant,
    unsigned contract, size_t batch, size_t time, size_t heads, size_t kvheads, size_t dim,
    const float *q, const float *k, const float *v, float *p, float *y, int *bad, void *workspace,
    size_t bytes);
BCIR_DT_API int bcir_tensor_attention_bf16_backward_pool(bcir_tensor_pool *pool, int variant,
    size_t batch, size_t time, size_t heads, size_t kvheads, size_t dim, const float *q,
    const float *k, const float *v, const float *p, const float *dy, float *dq, float *dk,
    float *dv, void *workspace, size_t bytes);
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
/* Two GEMMs as one pool job (NDT-SCHED-0): the first's tasks then the second's, each GEMM's
 * partition exactly its own pool entry's, so every output element keeps its depth order and the
 * results are the two entries' bit for bit; one barrier in place of two, and a narrow product's
 * idle threads take the other's tasks. For two products whose writes are disjoint -- a linear's
 * two VJPs, dx and dW, which read one adjoint. `contract` is 0 for v1, 1 for the fused v2, 2 for
 * the blocked v3 (the three entries above, in that order); a NULL pool or one thread runs the
 * first then the second serially. */
typedef struct bcir_tensor_mm_call {
  int ta, tb; size_t m, n, k; float alpha; const float *a, *b; float beta; float *c;
} bcir_tensor_mm_call;
BCIR_DT_API void bcir_tensor_mm_pair_pool(bcir_tensor_pool *pool, int contract,
    const bcir_tensor_mm_call *x, const bcir_tensor_mm_call *y);
/* GEMMs as one pool job (NDT-SCHED-1): `nchain` calls that write one C -- the same c, m and n --
 * run in order per tile, each tile's task scaling its elements by the next call's beta and adding
 * that call's terms, as the calls would one after another; then `ny` calls whose writes are
 * disjoint from the chain's and from each other, each with its own partition (the pair job's
 * second GEMM, generalized). The results are the calls' run one after another in the order given,
 * bit for bit: every element receives the same operations in the same order. At most
 * BCIR_TENSOR_MM_GROUP calls of each kind; a NULL pool or one thread, a count past it, or a chain
 * whose calls do not share c, m and n runs the calls serially in that order -- the chain, then the
 * rest. */
#define BCIR_TENSOR_MM_GROUP 4
BCIR_DT_API void bcir_tensor_mm_group_pool(bcir_tensor_pool *pool, int contract,
    const bcir_tensor_mm_call *chain, size_t nchain, const bcir_tensor_mm_call *y, size_t ny);
BCIR_DT_API int bcir_tensor_mm_kblock_variants(void);
BCIR_DT_API int bcir_tensor_mm_kblock_variant(int variant, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c);
/* The bf16 contract (NDT-BF16): a GEMM on bfloat16 operands accumulated in binary32 as the AMX
 * unit's TDPBF16PS accumulates them (BF16-0's record pins the instruction bit for bit), so the
 * AMX, AVX-512 and portable instances agree on every input that is not NaN. For each element:
 *   a(i,l) = bf16(alpha*A(i,l)), the product rounded to binary32 first, and b(l,j) = bf16(B(l,j)),
 *   where bf16 keeps a binary32's upper 16 bits rounded to nearest, ties to even, takes a
 *   subnormal to the zero of its sign and a NaN to a quiet NaN (bcir_tensor_bf16_bits;
 *   VCVTNEPS2BF16's rounding);
 *   c = beta*C, rounded (+0.0 under beta 0, C never read); the depth in blocks of
 *   BCIR_MM_BF16_BLOCK terms from l = 0, the last padded with +0.0 terms; for each block
 *   E = O = +0.0, then over its 16 pairs in order E = fma(a(i,2p), b(2p,j), E) and
 *   O = fma(a(i,2p+1), b(2p+1,j), O), each one rounding, a result that is tiny (below 2^-126 when
 *   rounded to 24 bits with the exponent unbounded) flushed to the zero of its sign; then
 *   c = daz(c) + (E + O), each sum flushed when subnormal, daz(c) the zero of c's sign when c is
 *   subnormal. A NaN result is a quiet NaN; which one is not part of the contract.
 * A state trained under it is its own (BCIR_DT_CONTRACT_BF16). The operands are packed first into
 * workspace the caller lends -- A as 16-row x 32-depth tiles, B as 16-pair x 16-column tiles of its
 * depth pairs interleaved, rows, columns and depths padded with +0.0 to multiples of 32 -- one
 * layout on every instance; bcir_tensor_mm_bf16_workspace gives the bytes one GEMM packs whole,
 * bcir_tensor_mm_bf16_group_workspace a group's (operands the calls share packed once). Less
 * workspace is not a different result: a group that does not fit runs its calls one at a time and
 * a call that does not fit runs in chunks of rows and columns, every element the same operations
 * in the same order; below one 32-row strip of A and one 32-column strip of B, 64 bytes of
 * alignment besides, an entry writes nothing and returns 0. Instances: bit 1 portable C, bit 4
 * AVX-512F (the flushes by MXCSR's FTZ and DAZ, set around its kernel only), bit 8 AMX -- offered
 * where the CPU has AMX-TILE and AMX-BF16, the OS enables their state, Linux grants the process
 * the tile permission (arch_prctl ARCH_REQ_XCOMP_PERM) and a differential against the portable
 * kernel on the contract's edges agrees bit for bit; asked once per process, the answer kept in
 * one atomic word, the unit's only state between calls. The entries return 1 (the _variant entry
 * the instance it ran) or 0, nothing written: workspace below the minimum, an instance the host
 * cannot run. */
#define BCIR_MM_BF16_BLOCK 32
BCIR_DT_API unsigned bcir_tensor_bf16_bits(float x);
BCIR_DT_API size_t bcir_tensor_mm_bf16_workspace(size_t m, size_t n, size_t k);
BCIR_DT_API size_t bcir_tensor_mm_bf16_group_workspace(const bcir_tensor_mm_call *chain,
    size_t nchain, const bcir_tensor_mm_call *y, size_t ny);
BCIR_DT_API int bcir_tensor_mm_bf16_variants(void);
BCIR_DT_API int bcir_tensor_mm_bf16_variant(int variant, int ta, int tb, size_t m, size_t n,
    size_t k, float alpha, const float *a, const float *b, float beta, float *c, void *workspace,
    size_t bytes);
BCIR_DT_API int bcir_tensor_mm_bf16_pool(bcir_tensor_pool *pool, int ta, int tb, size_t m,
    size_t n, size_t k, float alpha, const float *a, const float *b, float beta, float *c,
    void *workspace, size_t bytes);
/* bcir_tensor_mm_group_pool's job under the bf16 contract: the chain's calls into one C in order,
 * then the rest, each element's operations those of the calls run one after another. */
BCIR_DT_API int bcir_tensor_mm_bf16_group_pool(bcir_tensor_pool *pool,
    const bcir_tensor_mm_call *chain, size_t nchain, const bcir_tensor_mm_call *y, size_t ny,
    void *workspace, size_t bytes);
/* The bf16 contract's operands written by their producers (BF16-PANELS). A tensor only bf16 GEMMs
 * read can be stored as the packed panel such a GEMM reads -- each element the conversion of its
 * binary32 value (bcir_tensor_bf16_bits), in the tiles the entries' packs write for a whole call --
 * so the GEMM skips that pack and computes the packed call's product bit for bit. A NaN value's
 * element is a quiet NaN, which one not part of the contract: which NaN an operation on two NaNs
 * propagates is the compiled code's, so the writers' copy of a kernel and the entries' can hold
 * different ones. An r x c row-major tensor (r and c multiples of 32: no padding) has three panels,
 * each of r*c elements:
 *   as A    rows r, depth c, in 16-row x 32-depth tiles (a linear's input in x W^T, an adjoint in
 *           its input adjoint dy W);
 *   as B    depth r, columns c, in 16-pair x 16-column tiles of its depth pairs (an activation in
 *           its weight adjoint dy^T x);
 *   as A^T  rows c, depth r, as A's tiles (an adjoint in its weight adjoint, the transposed A).
 * bcir_tensor_mm_bf16_group_panels_pool is bcir_tensor_mm_bf16_group_pool with pre[q] (q over the
 * chain, then the rest) naming call q's A panel and B panel, each NULL where the entry packs that
 * operand as before (pre NULL: none); a panel is never shared between calls. A call with a panel
 * runs whole in the one job: m, n and k multiples of 32, alpha 1 where A is a panel (the A pack
 * converts alpha times A; a panel is the operand alone), and the workspace holding the group's other
 * packs whole, or the entry returns 0, nothing written; bcir_tensor_mm_bf16_group_panels_workspace
 * is that whole, the bytes the group packs with its panels (WS-PLAN: its rule is the entry's -- a
 * panel neither packed nor shared). The writers: SiLU(x) * u (v2 non-zero: the
 * BCIR exponential's, bcir_tensor_silu_mul_v2_scan_pool's values; else v1's libm SiLU) as A and as B,
 * *bad set when a binary32 value is not finite (the scan entries' verdict); the SwiGLU VJP's two
 * adjoints (bcir_tensor_swiglu_backward_v2_pool's or v1's values) each as A and as A^T. They return
 * 1, or 0 with nothing written when r or c is not a multiple of 32. */
typedef struct bcir_tensor_bf16_panels { const unsigned short *a, *b; } bcir_tensor_bf16_panels;
BCIR_DT_API int bcir_tensor_mm_bf16_group_panels_pool(bcir_tensor_pool *pool,
    const bcir_tensor_mm_call *chain, size_t nchain, const bcir_tensor_mm_call *y, size_t ny,
    const bcir_tensor_bf16_panels *pre, void *workspace, size_t bytes);
BCIR_DT_API size_t bcir_tensor_mm_bf16_group_panels_workspace(const bcir_tensor_mm_call *chain,
    size_t nchain, const bcir_tensor_mm_call *y, size_t ny, const bcir_tensor_bf16_panels *pre);
BCIR_DT_API int bcir_tensor_silu_mul_panels_pool(bcir_tensor_pool *pool, int v2, size_t r,
    size_t c, const float *x, const float *u, unsigned short *a, unsigned short *b, int *bad);
BCIR_DT_API int bcir_tensor_swiglu_backward_panels_pool(bcir_tensor_pool *pool, int v2, size_t r,
    size_t c, const float *g, const float *u, const float *dy, unsigned short *dga,
    unsigned short *dgt, unsigned short *dua, unsigned short *dut);
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
BCIR_DT_API void bcir_tensor_silu_mul_v2_pool(bcir_tensor_pool *pool, size_t n, const float *x,
    const float *u, float *y);
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
/* PX-3b (DISPATCH): the instance a job runs and the threads it runs on, as a planner names them.
 *
 * bcir_tensor_pool_cap(pool, workers) has the jobs after it run on the caller and workers 1 ..
 * workers-1 of the pool -- 0, or a count at or past the pool's, all of them -- and returns the cap
 * in force before (0: all); bcir_tensor_pool_workers is what the next job runs on (a NULL pool 1).
 * Every partition is a function of the shapes alone, so a job's result is the same bits at every
 * cap; a worker past the cap reports each job without claiming a task (the reports stay the
 * barrier), a cap of one runs every job on the caller, and what a job's workspace must hold (the
 * bf16 entries' regions) stays a function of the pool's threads. A NULL pool: no effect, 0.
 *
 * The GEMM and attention entries with the instance named: `variant` 0 the widest the family admits
 * (the plain entries above), else one bit of it -- bcir_tensor_mm_contract_variants(contract): v1's
 * (contract 0) bcir_tensor_mm_variants, the fused and blocked contracts' (1, 2)
 * bcir_tensor_mm_fused_variants, the bf16 contract's (3) bcir_tensor_mm_bf16_variants; the binary32
 * attention's products the fused GEMM's under BCIR_TENSOR_ATTENTION_FUSED, v1's otherwise. Each
 * returns the instance it ran, or 0 having written nothing -- an instance the family does not admit,
 * a contract outside 0..3, panels outside the bf16 contract, or what its plain entry refuses -- and
 * its results are the plain entry's bit for bit, every instance of a family the same bits. */
BCIR_DT_API unsigned bcir_tensor_pool_cap(bcir_tensor_pool *pool, unsigned workers);
BCIR_DT_API unsigned bcir_tensor_pool_workers(const bcir_tensor_pool *pool);
BCIR_DT_API int bcir_tensor_mm_contract_variants(int contract);
BCIR_DT_API int bcir_tensor_mm_contract_variant_pool(bcir_tensor_pool *pool, int contract,
    int variant, int ta, int tb, size_t m, size_t n, size_t k, float alpha, const float *a,
    const float *b, float beta, float *c, void *workspace, size_t bytes);
BCIR_DT_API int bcir_tensor_mm_pair_variant_pool(bcir_tensor_pool *pool, int contract, int variant,
    const bcir_tensor_mm_call *x, const bcir_tensor_mm_call *y);
BCIR_DT_API int bcir_tensor_mm_group_variant_pool(bcir_tensor_pool *pool, int contract, int variant,
    const bcir_tensor_mm_call *chain, size_t nchain, const bcir_tensor_mm_call *y, size_t ny,
    const bcir_tensor_bf16_panels *pre, void *workspace, size_t bytes);
BCIR_DT_API int bcir_tensor_attention_contract_variant_scan_pool(bcir_tensor_pool *pool,
    int variant, unsigned contract, size_t batch, size_t time, size_t heads, size_t kvheads,
    size_t dim, const float *q, const float *k, const float *v, float *p, float *y, int *bad);
BCIR_DT_API int bcir_tensor_attention_backward_contract_variant_pool(bcir_tensor_pool *pool,
    int variant, unsigned contract, size_t batch, size_t time, size_t heads, size_t kvheads,
    size_t dim, const float *q, const float *k, const float *v, float *p, const float *dy,
    float *dq, float *dk, float *dv);
/* ATT-CONTIG: the binary32 attention entries above with a workspace the caller lends. Where a row of
 * q, k or v holds more than one head and the context reaches BCIR_ATT_CONTIG_FROM positions (1024 on
 * the measuring hosts; no value depends on it), each task copies its head's rows into a region of the
 * workspace -- the forward's q, k and v rows and its context, the backward's K and V rows and their
 * adjoints and each query head's Q, dY and dQ -- runs on them and copies its results back: values
 * moved, none computed. One region a worker of the pool where the workers are no more than the
 * forward's tasks (batch x heads), else one a task: bcir_tensor_attention_pool_workspace bytes for a
 * pool of that many threads (a NULL pool one), at most bcir_tensor_attention_workspace (one region a
 * task, a function of the shapes alone), both 0 where no task copies (or kvheads not dividing heads).
 * Less workspace is not a different result: with less than the pool's need, or none, every task runs
 * on the rows in place. Each returns what its entry without the workspace returns, and every result
 * is that entry's bit for bit, at every thread count and with any workspace. */
BCIR_DT_API size_t bcir_tensor_attention_workspace(size_t batch, size_t time, size_t heads,
    size_t kvheads, size_t dim);
BCIR_DT_API size_t bcir_tensor_attention_pool_workspace(size_t batch, size_t time, size_t heads,
    size_t kvheads, size_t dim, unsigned threads);
BCIR_DT_API int bcir_tensor_attention_contract_variant_scan_ws_pool(bcir_tensor_pool *pool,
    int variant, unsigned contract, size_t batch, size_t time, size_t heads, size_t kvheads,
    size_t dim, const float *q, const float *k, const float *v, float *p, float *y, int *bad,
    void *workspace, size_t bytes);
BCIR_DT_API int bcir_tensor_attention_backward_contract_variant_ws_pool(bcir_tensor_pool *pool,
    int variant, unsigned contract, size_t batch, size_t time, size_t heads, size_t kvheads,
    size_t dim, const float *q, const float *k, const float *v, float *p, const float *dy,
    float *dq, float *dk, float *dv, void *workspace, size_t bytes);
#ifdef __cplusplus
}
#endif
#endif
