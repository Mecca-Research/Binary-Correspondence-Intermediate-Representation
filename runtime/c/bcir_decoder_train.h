/* Complete dense Llama/SwiGLU training in C. No allocation in plan, forward,
 * backward, optimizer or multi-step run. FP32 weights/activations; FP64 loss/norm.
 * HF row-major projections and half-split RoPE (ingest permutes only at export).
 * The checked state owns no memory; arenas must be disjoint and float aligned.
 * Caller serializes access; tensor providers must honor bcir_tensor.h contracts.
 * The state ends in its thread pool (ABI 2, NDT-THREADS): NULL or a bcir_tensor_pool the
 * caller owns, run by every BCIR kernel the provider does not replace. Each kernel's
 * partition is fixed by the shapes, so the state after any number of steps is bit-identical
 * at every thread count and to the serial state; a provider threads its own kernels.
 * `contract` (ABI 3, NDT-FMA) names the arithmetic contract BCIR's GEMM keeps: 0 is v1 (one
 * rounded multiply and one rounded add per term), BCIR_DT_CONTRACT_FUSED is v2 (one fmaf per
 * term, k ascending -- bit-identical wherever fmaf is an IEEE fusedMultiplyAdd). A provider's
 * GEMM keeps its own. BCIR_DT_CONTRACT_EXP (NDT-EXP) puts SiLU, the SwiGLU VJP and the attention
 * softmax under their v2 contracts (the BCIR exponential, no libm); a provider's SiLU keeps its
 * own. BCIR_DT_CONTRACT_REDUCE (NDT-REDUCE) puts the clipping norm under a fixed-shape reduction:
 * the gradient in chunks of bcir_decoder_reduce_chunk elements (a function of the count alone),
 * each chunk's squares summed in index order in double from zero, the chunk sums added in chunk
 * order from zero -- so the update's one pass runs on the pool; without the bit the norm is the one
 * index-order chain, serial. BCIR_DT_CONTRACT_BF16 (ABI 4, NDT-BF16) puts BCIR's GEMMs under the
 * bf16 contract (bcir_tensor.h: operands rounded to bfloat16, accumulated as the AMX unit's
 * TDPBF16PS accumulates them, every instance bit for bit); the state then lends `workspace`, at
 * least bcir_decoder_workspace_floor bytes disjoint from the arenas and the state, for the packed
 * operands (BCIR_DT_CAPACITY below that, BCIR_DT_INVALID without one or overlapping), and a
 * provider's GEMM, as under every contract, keeps its own. Any workspace from the floor up trains
 * the same state bit for bit (WS-PLAN): bcir_decoder_workspace_need is what the step packs, and
 * below it a group runs its calls one at a time and a call in chunks (bcir_tensor.h), the SwiGLU
 * block without its panels below the bytes its panel groups pack -- slower, never different; a
 * product that refuses the workspace fails the step with BCIR_DT_CAPACITY, nothing committed. Under
 * any contract the binary32 attention copies its tasks' head rows into a lent workspace from
 * bcir_tensor_attention_pool_workspace's bytes on (ATT-CONTIG: long contexts whose rows hold several
 * heads), and runs them in place below that or without one -- the same values; a state that lends a
 * workspace without the bf16 contract lends it disjoint from the arenas and the state
 * (BCIR_DT_INVALID otherwise), at any size, none at all the default. One GEMM contract at a time: FUSED,
 * KBLOCK and BF16 exclude one another. BCIR_DT_CONTRACT_ATTN (ATTN-FMA) puts the attention's
 * products under the fused contract (bcir_tensor.h's BCIR_TENSOR_ATTENTION_FUSED: every
 * multiply-add of the scores, the context and the backward's products one fmaf, the same terms in
 * the same order), with any GEMM contract or none. BCIR_DT_CONTRACT_ATTNBF16 (BF16-ATTN) puts them on
 * bfloat16 operands instead (bcir_tensor.h's bf16 attention: each product the bf16 GEMM contract's
 * over the depth tiles the causal mask reaches, the softmax and the score adjoint v1's), with the
 * bf16 GEMM contract alone: it packs into the same workspace, which bcir_decoder_workspace_bytes
 * then sizes for it too, and excludes BCIR_DT_CONTRACT_ATTN. Other bits are refused. A state trained under one contract
 * is not another's: the checkpoint names it. Whatever the contract, the loss's rows, the RMSNorm gamma
 * adjoints and the embedding scatter run on the pool in the serial order (NDT-REDUCE), so every
 * loss and state is the serial one, bit for bit, at every thread count.
 * `recompute` (ABI 5, REMAT) names layers whose forward cache the backward recomputes instead of
 * keeping: bit l for layer l (layers past 64 are always kept; a bit past the last layer is
 * refused). The kept layers hold their slots of the activation arena in layer order and the
 * recomputed ones share one slot after them, so the arena needs bcir_decoder_activation_count
 * floats, the logits its last; the backward re-runs a recomputed layer's forward from its input
 * (the residual stream keeps every layer's) up to the hidden activation, the last value its
 * backward reads, except the highest such layer, whose forward cache the shared slot still holds.
 * The kernels' partitions are the shapes', so the recomputed values are the forward's bit for bit
 * and the state is the all-kept state's under every recompute set.
 * The plan's three counts are the arena sums of the step program in
 * bcir/hosted/models/native_program.py (NDT-GEM), held to it by train.plan.mismatch, and so is
 * bcir_decoder_activation_count under a recompute set. The scratch arena's last span (ROPE-TABLE,
 * time x head-dim floats) is the rotation's table: each forward fills it with the cosines and sines
 * bcir_tensor_rope_pool evaluates on every call (bcir_tensor_rope_table_pool) and every rotation of
 * its step reads them there (bcir_tensor_rope_lent_pool), so the step's values are the same bit for
 * bit with one evaluation of them where each rotation made its own. */
#ifndef BCIR_DECODER_TRAIN_H
#define BCIR_DECODER_TRAIN_H
#include "bcir_tensor.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct bcir_decoder_spec {
  uint32_t vocab,width,heads,kvheads,layers,ff,batch,time,tied;
  double rope_base,rms_eps;
} bcir_decoder_spec;
typedef struct bcir_decoder_plan {
  bcir_decoder_spec spec;
  size_t parameters,activations,scratch;
} bcir_decoder_plan;
typedef struct bcir_decoder_state {
  bcir_decoder_plan plan;
  float *weights,*gradients,*moment1,*moment2,*activations,*scratch;
  size_t weight_capacity,gradient_capacity,moment1_capacity,moment2_capacity;
  size_t activation_capacity,scratch_capacity;
  uint64_t step;
  double beta1_power,beta2_power;
  bcir_tensor_provider provider;
  bcir_tensor_pool *pool;
  uint32_t contract;
  void *workspace;          /* ABI 4 (NDT-BF16): the bf16 contract's packs; under any contract the
                             * binary32 attention's copies (ATT-CONTIG) */
  size_t workspace_bytes;
  uint64_t recompute;       /* ABI 5 (REMAT): bit l recomputes layer l's cache in the backward */
} bcir_decoder_state;
typedef struct bcir_decoder_adamw {
  double lr,beta1,beta2,epsilon,weight_decay,grad_clip;
} bcir_decoder_adamw;
typedef struct bcir_decoder_event { double loss,grad_norm; uint64_t step; } bcir_decoder_event;
enum { BCIR_DT_OK=0, BCIR_DT_INVALID=1, BCIR_DT_CAPACITY=2, BCIR_DT_NUMERIC=3 };
enum { BCIR_DT_CONTRACT_FUSED=1u, BCIR_DT_CONTRACT_EXP=2u, BCIR_DT_CONTRACT_KBLOCK=4u,
  BCIR_DT_CONTRACT_REDUCE=8u, BCIR_DT_CONTRACT_BF16=16u, BCIR_DT_CONTRACT_ATTN=32u,
  BCIR_DT_CONTRACT_ATTNBF16=64u };
/* 1: the state through its provider; 2: the state ends in its pool (NDT-THREADS); 3: and in its
 * contract (NDT-FMA, NDT-EXP, NDT-KBLOCK, NDT-REDUCE, ATTN-FMA: the same field, more bits); 4: and in the
 * bf16 contract's workspace (NDT-BF16); 5: and in its recompute set (REMAT). */
BCIR_DT_API uint32_t bcir_decoder_abi_version(void);
/* The step's workspace (WS-PLAN), every figure 0 unless `contract` names BF16 -- the binary32
 * attention's copies excepted (ATT-CONTIG, below) -- and under BF16 SIZE_MAX for a plan
 * bcir_decoder_make_plan refuses (0 without it). need: the bytes at which every GEMM group of
 * the step packs whole -- the most any group packs, counted by the tensor unit's own rule
 * (bcir_tensor_mm_bf16_group_panels_workspace: an operand two calls read alike packed once, a
 * panel not at all) over the groups as the step issues them, the SwiGLU block's from its panels
 * where the shapes admit them (the rows, the width and the hidden width multiples of 32: a state
 * whose provider replaces the GEMM or SiLU, or whose panel slots are not 64-byte aligned, runs that
 * block's groups without panels, in chunks where they exceed this) -- and under the bf16 attention
 * its regions for a pool of `threads` workers (bcir_tensor_attention_bf16_pool_workspace; 0: any
 * pool), else the binary32 attention's (bcir_tensor_attention_pool_workspace, ATT-CONTIG; under a
 * contract without BF16 the need is these alone, 0 where its tasks copy nothing). floor: the least a
 * state admits (valid() asks it for the state's pool): one 32-row strip of A and one 32-column strip
 * of B at the step's deepest call, and the bf16 attention's regions for the pool; 0 without BF16.
 * bytes: the need on any pool. */
BCIR_DT_API size_t bcir_decoder_workspace_need(const bcir_decoder_plan *plan, uint32_t contract,
    unsigned threads);
BCIR_DT_API size_t bcir_decoder_workspace_floor(const bcir_decoder_plan *plan, uint32_t contract,
    unsigned threads);
BCIR_DT_API size_t bcir_decoder_workspace_bytes(const bcir_decoder_plan *plan, uint32_t contract);
/* The least workspace at which the SwiGLU block runs with its panels (WS-PLAN): what its panel
 * groups pack, each of which runs whole; 0 unless `contract` names BF16 and the shapes admit the
 * panels, SIZE_MAX for a plan bcir_decoder_make_plan refuses. */
BCIR_DT_API size_t bcir_decoder_workspace_panels(const bcir_decoder_plan *plan, uint32_t contract);
/* The activation arena's floats under a recompute set (REMAT): the plan's count less one cache slot
 * for each recomputed layer but the one slot they share -- plan->activations for none. SIZE_MAX
 * for a plan bcir_decoder_make_plan refuses or a bit past its last layer. */
BCIR_DT_API size_t bcir_decoder_activation_count(const bcir_decoder_plan *plan, uint64_t recompute);
/* The reduce contract's chunk for `count` elements: 1024 doubled until the chunks number at most
 * 1024 (so 1024 up to 2^20 elements, 16384 for the 9.84M decoder). The update's first pass runs
 * one task per chunk under every contract; only under BCIR_DT_CONTRACT_REDUCE do the chunks reach
 * the norm. */
BCIR_DT_API size_t bcir_decoder_reduce_chunk(size_t count);
/* The clipping norm's square as the update's first pass computes it under the state's contract
 * -- the scaled gradient's squares in one index-order chain, or the reduce contract's chunk sums
 * in chunk order -- with the pass's verdict: BCIR_DT_NUMERIC on a non-finite element of the
 * gradient, weights or moments (a negative second moment included), nothing written then; the
 * update reports its square root. For the oracle's emulation, which compares the sum before the
 * root rounds a difference away. */
BCIR_DT_API int bcir_decoder_gradient_norm(bcir_decoder_state *s,double gradient_scale,
    double *squared);
/* Checked shape/products; failure leaves *plan untouched. Parameter order:
 * embedding; each layer: attn_norm,q,k,v,o,ff_norm,gate,up,down;
 * final_norm; untied head. Retain caches; reuse adjoints across all layers.
 * Capacities are float element counts, not bytes. */
BCIR_DT_API int bcir_decoder_make_plan(const bcir_decoder_spec *spec,bcir_decoder_plan *plan);
/* Forward stores logits at the end of the activation arena. No backward reads its cache (the
 * backward runs its own forward), so its attention keeps no probabilities (INFER-ATTN): each
 * layer's probability region is the attention's scratch, its contents after it unspecified, and
 * every other value -- the logits, the verdict -- is the backward's forward's bit for bit.
 * Backward computes mean causal-token CE and every parameter gradient including tied-head
 * accumulation; it consumes the forward cache it recomputes, leaving each layer's attention
 * probabilities replaced by their score adjoint (under BCIR_DT_CONTRACT_ATTNBF16, as they were) --
 * the binary32 attention's at j <= i: its forward runs under BCIR_TENSOR_ATTENTION_LOWER (P-LOWER),
 * so a row's columns past its diagonal are +0.0 or what the region held before, and under
 * BCIR_TENSOR_ATTENTION_BANDS (P-BANDS), so where the host and the shapes admit it the region holds
 * 16-row bands with the rows in the lanes (bcir_tensor.h), a band's keys past its last row as the
 * region held them.
 * IDs: batch*time, strictly [0,vocab), rejected before arena mutation.
 * accumulate=1 adds another mean batch gradient. */
BCIR_DT_API int bcir_decoder_forward(bcir_decoder_state *s,const uint32_t *tokens,size_t count);
BCIR_DT_API int bcir_decoder_loss_backward(bcir_decoder_state *s,const uint32_t *tokens,
    const uint32_t *targets,size_t count,int accumulate,double *loss);
/* Atomic preflight: invalid/non-finite inputs or overflowing updates leave weights,
 * moments and step unchanged. Scale before global-L2 clipping. */
BCIR_DT_API int bcir_decoder_update(bcir_decoder_state *s,const bcir_decoder_adamw *opt,
    double gradient_scale,double *grad_norm);
/* Native loop, optional per-update rates; validate all IDs/outputs before any update.
 * A later numeric failure preserves completed steps and reports completed count.
 * A call's first step scans the weights before its forward; the steps after it skip that scan,
 * the weights being the call's own last commit -- every candidate admitted, finite -- with no
 * caller code run since (FACTS-RUN): one call of several steps scans once. */
BCIR_DT_API int bcir_decoder_run(bcir_decoder_state *s,const bcir_decoder_adamw *opt,
    const uint32_t *tokens,const uint32_t *targets,size_t token_count,
    const double *rates,size_t steps,bcir_decoder_event *events,
    size_t event_capacity,size_t *completed);
/* Average accumulation consecutive micro-batches; one clip and update per step. */
BCIR_DT_API int bcir_decoder_run_accum(bcir_decoder_state *s,const bcir_decoder_adamw *opt,
    const uint32_t *tokens,const uint32_t *targets,size_t token_count,
    const double *rates,size_t steps,size_t accumulation,bcir_decoder_event *events,
    size_t event_capacity,size_t *completed);
/* PX-2 (PACK): bcir_decoder_run_accum's run from a launch program -- the planner's decisions (which
 * calls share a job, each GEMM's entry, the SwiGLU block's panels, the forward a recomputed layer
 * re-runs, the scans the run's facts leave out, where each operand sits) carried as data and run by
 * the trainer's own kernels (bcir/hosted/models/native_launch.py states the program for a trainer's
 * decisions). A program is `records` records of BCIR_DT_RECORD_WORDS words, the layout PX-1's launch
 * trace records: kind, launch (a GEMM job's calls share it; every other record is a job of one; each
 * job's launch exceeds the one before), flags, eight dimensions, ten operands as (space, byte offset,
 * bytes) -- spaces 1-4 the parameter arenas in the state's order, 5 activations, 6 scratch, 7 the
 * workspace (whole: the bf16 contract's, and the binary32 attention's where the state lends bytes),
 * 8 the micro-batch's tokens or targets (whole, offset 0); an absent one (0, 0, 0) -- and seven
 * spare words, zero but the two PX-3b names below. A kernel record runs its call as
 * the step runs it, so a program's trace is the program; PASS, PREFLIGHT and COMMIT are trace
 * records only. ROPE-TABLE: a ROPE record (batch, time, heads, dim, inverse) reads its table as its
 * second operand, time x dim floats, which a ROPE_TABLE record (time, dim) writes at the plan's
 * rope_base. A scan of a parameter arena fails the run (BCIR_DT_NUMERIC) on a value not
 * finite, a scan of activations or scratch reports to the next FORWARD check; a scan of the whole
 * weights, and an update, make the weights known finite, and no other record reads them before
 * they are (FACTS-RUN). The host's records: CHECK -- FORWARD: a product refused its workspace,
 * BCIR_DT_CAPACITY, else a value the forward's writers or scans found not finite, BCIR_DT_NUMERIC;
 * RAN: a refused product, BCIR_DT_CAPACITY;
 * LOSS: the same, then the micro-batch's loss not finite, BCIR_DT_NUMERIC, else it closes the
 * micro-batch (its tokens are the next ones), its mean adding a 1/accumulation share to the step's
 * loss -- and UPDATE (the update's pass record, its kind UPDATE): bcir_decoder_update's update with
 * the gradient scaled by 1/accumulation and the step's rate, its event the step's.
 * The whole program is checked before anything runs, BCIR_DT_INVALID with nothing written: a record
 * of no kind or with a field outside its grammar, an operand outside the spaces its use admits
 * (kernels read the weights, gradients, activations and scratch and write the last three), outside
 * its space or of another size than its dimensions name, a written operand meeting another operand
 * of its job (one GEMM chain's shared C excepted), flags other than the state's contract names, a
 * GEMM entry or job another than the state's kernels take (bcir_tensor.h's group and panel rules
 * included), a provider's call without the provider or BCIR's SiLU with one, the weights read
 * before they are known, a micro-batch not closed by LOSS after exactly one loss record, an update
 * not after exactly `accumulation` micro-batches, or a program not `steps` such steps -- after the
 * arguments' checks of bcir_decoder_run_accum, whose outputs, ordering of verdicts and completed
 * count this has; the program itself lies outside the state, the events and the count. The check
 * holds each call to its kernel's contract and the run to its step's shape; that the operands'
 * places compute the step's gradient is the program's, which train.plan.program.mismatch holds.
 * PX-4 (PLACE): here the activation and scratch arenas are as long as the caller lent them (their
 * capacities, any length, where the fixed step needs the plan's spans: BCIR_DT_CAPACITY there), and
 * every operand lies inside its arena's lent length -- so a state whose two transient arenas a
 * placement sized (bcir/hosted/models/native_pack.py's place_by_lifetime) runs the programs placed
 * there; the arguments and the program lie outside those lengths.
 * PX-3b (DISPATCH): two of the seven spare words name how a job runs, alike in every record of it.
 * Word 41 is the instance -- 0 the widest the job's family admits (what the step runs), else one bit
 * bcir_tensor_mm_contract_variants admits for the state's GEMM contract, or for an attention the
 * instance family its products take (bcir_tensor.h) -- and only a GEMM job of BCIR's entries or an
 * attention (ATTN, ATTN_VJP) may name one. Word 42 is the job's workers -- 0 all the pool's threads,
 * else 1 .. the pool's threads (bcir_tensor_pool_cap) -- on any record but CHECK. Every instance of
 * a family and every cap is the same bits, so a program of any is the program of none, bit for bit;
 * a word outside its grammar is BCIR_DT_INVALID with nothing written. A trace records in them the
 * instance a GEMM or attention launch ran (the entry's return) and the workers its job ran on. A
 * run leaves the pool's cap as it found it, however it ends. */
#define BCIR_DT_RECORD_WORDS 48
#define BCIR_DT_WORD_INSTANCE 41
#define BCIR_DT_WORD_WORKERS 42
enum { BCIR_DT_REC_SCAN=1, BCIR_DT_REC_ZERO, BCIR_DT_REC_EMBED, BCIR_DT_REC_RMS, BCIR_DT_REC_GEMM,
  BCIR_DT_REC_ROPE, BCIR_DT_REC_ATTN, BCIR_DT_REC_ATTN_VJP, BCIR_DT_REC_ADD, BCIR_DT_REC_SILU_MUL,
  BCIR_DT_REC_SILU_MUL_PANELS, BCIR_DT_REC_SWIGLU_VJP, BCIR_DT_REC_SWIGLU_VJP_PANELS,
  BCIR_DT_REC_RMS_VJP, BCIR_DT_REC_LOSS, BCIR_DT_REC_SCATTER, BCIR_DT_REC_PASS,
  BCIR_DT_REC_PREFLIGHT, BCIR_DT_REC_COMMIT, BCIR_DT_REC_MUL, BCIR_DT_REC_PROVIDER_SILU,
  BCIR_DT_REC_CHECK, BCIR_DT_REC_UPDATE, BCIR_DT_REC_ROPE_TABLE };
enum { BCIR_DT_CHECK_FORWARD=1u, BCIR_DT_CHECK_RAN=2u, BCIR_DT_CHECK_LOSS=4u };
BCIR_DT_API int bcir_decoder_run_program(bcir_decoder_state *s,const bcir_decoder_adamw *opt,
    const uint64_t *program,size_t records,const uint32_t *tokens,const uint32_t *targets,
    size_t token_count,const double *rates,size_t steps,size_t accumulation,
    bcir_decoder_event *events,size_t event_capacity,size_t *completed);
#ifdef __cplusplus
}
#endif
#endif
