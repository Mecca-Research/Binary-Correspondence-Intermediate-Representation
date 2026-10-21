/*===- bcir_decoder_pack.h - the native training step from a StreamPack (PX-2b) -------===
 *
 * PX-2b (PACK): the launch program bcir_decoder_run_program runs (PX-2a), taken from BCIR's own
 * artifacts instead of words -- a StreamPack GEM hydrated from a K_BCIR-planned module of the
 * program's calls, and the ExecutionPlanV1 minted from the same realization, whose lifetimes place
 * every value the claims name (bcir/hosted/models/native_pack.py states both for a state and a
 * planner's decisions). The pack carries one claim per kernel call and one phase per launch (a GEMM
 * job's calls share theirs); each claim's rd and wr list the RIDs of its read and written operands
 * in operand order (an operand read and written in place in both), its block's base the micro-batch
 * of an embed, a loss or a scatter (else 0) and its count the call's work. Its name says what the
 * claim is, from a closed vocabulary (native_pack.OPS and the GEMM grammar
 * train.gemm.<alone|pair|chain|rest>.<nt|nn|tn>[.acc][.pa|.pb]); everything else its record holds
 * -- every dimension, every flag, every operand's place -- is derived here from the state and the
 * plan.
 *
 * bcir_decoder_lower_pack verifies the two artifacts with their own verifiers -- bcir_ep_check_pack's
 * laws, in one walk of the plan that hands the lowering its lifetimes (bcir_ep_check_pack_lifetimes):
 * the pack's provenance and range laws; the plan's wire laws, among them the alias law -- no two
 * lifetimes of one bank live at once on overlapping bytes; the pack the plan's lowering, one claim
 * per step in step order; one generation vector -- then lowers each claim GEM's executor dispatches
 * (bcir_sp_execute, in its order: the phases in the pack's order, a phase's claims by id) to one
 * record of `program`, BCIR_DT_RECORD_WORDS words, in that order. A phase's rank in the dispatch is
 * its launch (rank + 1) and the time a lifetime's phase span (its half-open ticks) is read in:
 * every value a claim names must be live at the claim's rank -- with the alias law, no claim can
 * write a byte another value still holds. The banks are the state's: weights, gradients, moment1,
 * moment2, activations and scratch (spaces 1-6), workspace (7, the bf16 workspace), tokens and
 * targets (8, the micro-batch's). Every claim is lowered and checked before any record is written,
 * so a refused pack writes nothing to `program` (*records 0): BCIR_DT_INVALID with `why` the
 * verifiers' bcir_status (1..25), ahead of any reason of the lowering's own, or a BCIR_DP_* reason
 * below; BCIR_DT_CAPACITY when `work` or `program` is short of bcir_decoder_pack_sizes' figures.
 * `work` is scratch: its bytes are the lowering's. The lowered program is the planner's, checked
 * again whole by bcir_decoder_run_program before anything runs (its records' grammar, operands and
 * job shapes are its check's, not this one's).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_DECODER_PACK_H
#define BCIR_DECODER_PACK_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_decoder_train.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Why a pack was refused, past the verifiers' own statuses:
 *   OP        a claim's name outside the vocabulary
 *   CLAIMS    claim ids not 0..n-1 once each, blocks not one a segment, a claim not dispatched
 *   RID       a claim names a RID no lifetime places
 *   BANK      a lifetime in a bank the state has not, or an operand in a bank its slot takes not
 *   LIVE      a claim names a value not live at the claim's rank
 *   SLOTS     rd / wr not the claim's operands: too few, too many, an in-place operand split
 *   RUN       a run of tensors not one contiguous span of one bank
 *   SHAPE     a dimension the operands do not determine (a size not a whole multiple)
 *   BLOCK     a block's base not the claim's micro-batch, or its count not the call's work
 *   LIVENESS  a plan whose lifetimes live in the schedule's ticks, not the phases' */
enum {
  BCIR_DP_OP = 101,
  BCIR_DP_CLAIMS,
  BCIR_DP_RID,
  BCIR_DP_BANK,
  BCIR_DP_LIVE,
  BCIR_DP_SLOTS,
  BCIR_DP_RUN,
  BCIR_DP_SHAPE,
  BCIR_DP_BLOCK,
  BCIR_DP_LIVENESS
};

/* The records and the work bytes bcir_decoder_lower_pack needs for this pack and plan (the pack's
 * semantic laws and the plan's header checked first, so a count is one the bytes hold):
 * BCIR_DT_OK, or BCIR_DT_INVALID with `why` the verifier's bcir_status and nothing else written. */
BCIR_DT_API int bcir_decoder_pack_sizes(const uint8_t *pack, size_t pack_len, const uint8_t *plan,
                                        size_t plan_len, size_t *records, size_t *work_bytes,
                                        int *why);

/* The program the pack states for state `s`, lowered through the plan (header comment): `work` is
 * the caller's scratch (8-byte aligned, `work_bytes` long), `program` holds `capacity` records. */
BCIR_DT_API int bcir_decoder_lower_pack(const bcir_decoder_state *s, const uint8_t *pack,
                                        size_t pack_len, const uint8_t *plan, size_t plan_len,
                                        void *work, size_t work_bytes, uint64_t *program,
                                        size_t capacity, size_t *records, int *why);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_DECODER_PACK_H */
