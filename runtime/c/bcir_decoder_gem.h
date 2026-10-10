/*===- bcir_decoder_gem.h - a decoder program, executed through GEM ------===
 *
 * QUAL-2. A decoder program (bcir/frontends/models/decoder_program.py: one claim per decoder
 * operation per token, planned by K_BCIR and hydrated into a StreamPack) runs here: GEM's
 * deterministic executor (bcir_sp_execute) dispatches the pack's claims, and each claim runs
 * the kernel its op string names on the operands its RIDs name -- the kernels
 * bcir_llama_ops.h shares with the monolithic runner, so the two compute bit-identical logits.
 *
 * The pack is the program. bcir_dgem_load decodes every segment and block into a claim table
 * and checks each claim against the RID scheme below and the model's tensor shapes before
 * anything runs; a pack that is not a decoder program for this model is refused whole. The
 * tape's capacity and the prompt's length are the program's: read from its argmax claims.
 * During the run every position, cache row and tape slot a claim names is checked against
 * the data flow so far (an embedding reads a written tape slot, an append extends its
 * layer's cache by exactly one row, attention reads the rows appended, an argmax appends at
 * the tape's end), so no index a hostile pack supplies reaches memory unchecked.
 *
 * Memory: the model and the pack are borrowed and must outlive the interpreter, unchanged; every
 * buffer it allocates (claim table, executor scratch, tape, workspace) is owned and released by
 * bcir_dgem_free, through the bcir_host_allocator given to bcir_dgem_load. Nothing is sized from
 * the pack's header before the semantic walk has read every record it declares.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_DECODER_GEM_H
#define BCIR_DECODER_GEM_H

#include "bcir_exec.h"
#include "bcir_llama_ops.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The RID scheme (an ABI; decoder_program.py spells the same values). */
enum {
  BCIR_DGEM_RID_X = 1, BCIR_DGEM_RID_H = 2, BCIR_DGEM_RID_Q = 3, BCIR_DGEM_RID_K = 4,
  BCIR_DGEM_RID_V = 5, BCIR_DGEM_RID_QR = 6, BCIR_DGEM_RID_KR = 7, BCIR_DGEM_RID_CTX = 8,
  BCIR_DGEM_RID_H2 = 9, BCIR_DGEM_RID_GATE = 10, BCIR_DGEM_RID_UP = 11, BCIR_DGEM_RID_FF = 12,
  BCIR_DGEM_RID_FINAL = 13, BCIR_DGEM_RID_LOGITS = 14, BCIR_DGEM_RID_TOK = 16,
  BCIR_DGEM_RID_EMBED = 32, BCIR_DGEM_RID_G_FINAL = 33, BCIR_DGEM_RID_HEAD = 34,
  BCIR_DGEM_LAYER_BASE = 256, BCIR_DGEM_LAYER_STRIDE = 16, BCIR_DGEM_MAX_LAYERS = 1024
};
enum {
  BCIR_DGEM_SLOT_G_ATTN = 0, BCIR_DGEM_SLOT_WQ = 1, BCIR_DGEM_SLOT_WK = 2,
  BCIR_DGEM_SLOT_WV = 3, BCIR_DGEM_SLOT_WO = 4, BCIR_DGEM_SLOT_G_FF = 5,
  BCIR_DGEM_SLOT_W_GATE = 6, BCIR_DGEM_SLOT_W_UP = 7, BCIR_DGEM_SLOT_W_DOWN = 8,
  BCIR_DGEM_SLOT_K_CACHE = 9, BCIR_DGEM_SLOT_V_CACHE = 10
};

/* The operations, in decoder_program.OPS order. */
typedef enum bcir_dgem_op {
  BCIR_DGEM_EMBED = 0, BCIR_DGEM_RMSNORM, BCIR_DGEM_MATVEC, BCIR_DGEM_ROPE,
  BCIR_DGEM_KV_APPEND, BCIR_DGEM_ATTENTION, BCIR_DGEM_MATVEC_ADD, BCIR_DGEM_SWIGLU,
  BCIR_DGEM_HEAD, BCIR_DGEM_ARGMAX, BCIR_DGEM_N_OPS
} bcir_dgem_op;

/* The op string of `op` ("dec.embed", ...), or "" out of range. */
const char *bcir_dgem_op_name(bcir_dgem_op op);

/* Refusals: 0 is success. */
#define BCIR_DGEM_E_ARG (-1)     /* NULL or inconsistent argument */
#define BCIR_DGEM_E_PACK (-2)    /* the StreamPack is malformed (pack_status holds why) */
#define BCIR_DGEM_E_PROGRAM (-3) /* a claim is not a decoder operation of this model */
#define BCIR_DGEM_E_MODEL (-4)   /* the model is not a ready Llama/SwiGLU BCIRQ8 decoder */
#define BCIR_DGEM_E_MEMORY (-5)  /* the workspace budget or an allocation */
#define BCIR_DGEM_E_ORDER (-6)   /* a claim ran outside its data-flow order */
#define BCIR_DGEM_E_KERNEL (-7)  /* a kernel refused its operands */

/* Per-operation counters, accumulated over every run since load (or reset). Octets are the
 * operands each kernel reads and writes: BCIRQ8 weights at a code an element and two
 * exponent octets a group, activations and cache rows in double, tape slots in int32 --
 * decoder_program.claim_traffic's model, counted at execution. */
typedef struct bcir_dgem_stats {
  uint64_t claims[BCIR_DGEM_N_OPS];
  uint64_t bytes_read[BCIR_DGEM_N_OPS];
  uint64_t bytes_written[BCIR_DGEM_N_OPS];
  uint64_t ns[BCIR_DGEM_N_OPS]; /* only with a clock installed */
} bcir_dgem_stats;

/* Optional hooks: a monotonic clock in nanoseconds (per-operation time), and a call after
 * every argmax with the tape position it wrote (per-token latency). */
typedef uint64_t (*bcir_dgem_clock_fn)(void *ctx);
typedef void (*bcir_dgem_token_fn)(void *ctx, size_t position, int32_t token);

struct bcir_dgem_claim; /* the decoded claim table's row (private) */

typedef struct bcir_dgem {
  const bcir_q8_model *model;      /* borrowed */
  const uint8_t *pack;             /* borrowed */
  size_t pack_len;
  bcir_status pack_status;         /* the runtime's verdict when E_PACK */
  struct bcir_dgem_claim *claims;  /* owned: indexed by claim id */
  size_t n_claims;
  bcir_exec_item *scratch;         /* owned: n_claims */
  bcir_phase_stat *phases;         /* owned: n_claims (an upper bound on distinct phases) */
  int32_t *tape;                   /* owned: capacity */
  uint32_t *kv_rows;               /* owned: rows appended, per layer */
  double *final_row;               /* owned: the FINAL resource (d_model) */
  size_t capacity, prompt_len, filled, next_position, current;
  size_t final_at, head_at;        /* the position whose final row / logits are current */
  bcir_llama_ws ws;                /* owned */
  bcir_dgem_stats stats;
  bcir_dgem_clock_fn clock; void *clock_ctx;
  bcir_dgem_token_fn on_token; void *token_ctx;
  int error;                       /* the first refusal inside a run */
  bcir_host_allocator allocator;
} bcir_dgem;

/* Decode and check the program; size and allocate the interpreter. Runs nothing. */
int bcir_dgem_load(bcir_dgem *g, const bcir_q8_model *model, const uint8_t *pack, size_t len,
                   const bcir_host_allocator *allocator);

/* The program's tape: positions it holds, and how many of them are the prompt. */
size_t bcir_dgem_capacity(const bcir_dgem *g);
size_t bcir_dgem_prompt_len(const bcir_dgem *g);

/* Execute the loaded program through GEM on `prompt` (exactly bcir_dgem_prompt_len ids).
 * `generated` receives capacity - prompt_len ids; `final_logits`, when non-NULL, the
 * vocab_size scores that chose the last. Statistics accumulate in g->stats. */
int bcir_dgem_run(bcir_dgem *g, const int32_t *prompt, size_t prompt_len, int32_t *generated,
                  size_t max_new, double *final_logits);

/* Zero the statistics. */
void bcir_dgem_reset_stats(bcir_dgem *g);

/* Release everything the interpreter owns and zero it; safe on a zeroed interpreter. */
void bcir_dgem_free(bcir_dgem *g);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_DECODER_GEM_H */
