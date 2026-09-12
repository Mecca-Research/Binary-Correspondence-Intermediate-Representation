/*===- bcir_execution_plan.h - BCIR ExecutionPlanV1 binary ABI (frozen v1) ----===
 *
 * The normative C view of the plan-as-bytes wire format (GEM+ roadmap G11, staged plan
 * S1-C): the plan a StreamPack was derived from and every reader prices. The Python
 * reference codec is bcir/abi/execution_plan_abi.py; docs/kernel/BCIR_EXECUTION_PLAN_ABI.md
 * is the prose spec. The freestanding C runtime (bcir_runtime.c) decodes and verifies it
 * with no libc dependency, exactly as it does the StreamPack.
 *
 * Wire format (little-endian; the StreamPack's conventions -- str := u16 length + UTF-8):
 *   Header (64 bytes, cache-line aligned)  -- bcir_ep_header (every u64 8-aligned)
 *   Body (sequential, length-prefixed records):
 *     source_plan : str
 *     steps[n_steps]           := claim_id:u64 phase_id:u32 candidate:str lane:u8 width:u32
 *                                 cost:i64 stream:u32 start:u64 duration:u64
 *     lifetimes[n_lifetimes]   := rid:u32 bank:str offset:u64 size:u64 alignment:u32
 *                                 first_phase:u32 last_phase:u32      (RIDs strictly ascending)
 *                                 [v2: first_tick:u64 last_tick:u64]  (half-open liveness in the
 *                                 header's liveness domain; a v1 record reads
 *                                 [first_phase, last_phase + 1))
 *     moves[n_moves]           := rid:u32 src_bank:str dst_bank:str offset:u64 size:u64
 *                                 route:str kind:u8 coherence:u8 map_gen:u32 data_gen:u32
 *                                 after_claim:u64 before_claim:u64
 *     generations[n_gens]      := rid:u32 map_gen:u32 data_gen:u32    (RIDs strictly ascending;
 *                                 bcir_generation_view, the StreamPack v4 record)
 *   Trailer: u32 CRC-32 of every preceding byte.
 *
 * The format is frozen at v1; fields are append-only across versions and a v1 reader
 * rejects a newer version and refuses nonzero reserved bytes. v2 (G5, S1-D) carves the
 * liveness byte out of the header pad (offset 9: 0 = phase positions, 1 = the placement's
 * ticks) and appends the tick tail to the lifetime record; encoders emit the lowest carrying
 * version, so a phase-liveness plan with default ticks is byte-identical v1.
 *
 * The wire laws (bcir_ep_verify; the Python codec applies the same predicate): mode legal;
 * streams >= 1 and 1 <= knee <= streams; per step a legal lane, a nonzero power-of-two
 * width, a stream below `streams` or the tail sentinel, duration == max(0, cost),
 * start + duration <= makespan; claim ids unique; lifetimes with ascending RIDs, a
 * power-of-two alignment the offset honors, size >= 1, first_phase <= last_phase,
 * last_tick > first_tick, and no two lifetimes of one bank live at once at overlapping
 * addresses (the alias law by bytes, BCIR_ERR_PLAN); moves with legal kind/coherence codes
 * and size >= 1; an ascending generation vector; the declared records consume the body
 * exactly (BCIR_ERR_TRAILING otherwise).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_EXECUTION_PLAN_H
#define BCIR_EXECUTION_PLAN_H

#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_EP_MAGIC       "BPLN"   /* bytes 0..3 of the header */
#define BCIR_EP_VERSION     1
#define BCIR_EP_VERSION_MAX 2   /* v2: the liveness byte + the lifetime tick tail (G5) */
#define BCIR_EP_HEADER_SIZE 64

/* The decoupled GGG/random tail's stream on the wire (the scheduler's TAIL_STREAM, -1). */
#define BCIR_EP_TAIL_STREAM 0xFFFFFFFFu

/* The placement the plan carries (bcir/gem/schedule.py::SCHEDULE_MODES). */
typedef enum bcir_ep_mode {
  BCIR_EP_MODE_EFT    = 0,   /* phase-barriered LPT/EFT dispatch */
  BCIR_EP_MODE_TOKENS = 1    /* token-pipelined dispatch */
} bcir_ep_mode;
#define BCIR_EP_MODE_MAX 1

/* The domain the lifetimes' ticks live in (v2; a v1 plan reads BCIR_EP_LIVENESS_PHASE). */
typedef enum bcir_ep_liveness {
  BCIR_EP_LIVENESS_PHASE    = 0,   /* topological phase positions: [first_phase, last_phase + 1) */
  BCIR_EP_LIVENESS_SCHEDULE = 1    /* the placement's own ticks (schedule-aware liveness, G5) */
} bcir_ep_liveness;
#define BCIR_EP_LIVENESS_MAX 1

/* G8 movement-edge kinds and coherence actions (closed sets; a code outside is refused). */
typedef enum bcir_ep_move_kind {
  BCIR_EP_MOVE_DIRECT = 0, BCIR_EP_MOVE_PEER = 1, BCIR_EP_MOVE_STAGED = 2,
  BCIR_EP_MOVE_REMATERIALIZED = 3, BCIR_EP_MOVE_COMPRESSED = 4, BCIR_EP_MOVE_EVICTED = 5
} bcir_ep_move_kind;
#define BCIR_EP_MOVE_KIND_MAX 5
typedef enum bcir_ep_coherence {
  BCIR_EP_COHERENCE_NONE = 0, BCIR_EP_COHERENCE_FLUSH = 1,
  BCIR_EP_COHERENCE_INVALIDATE = 2, BCIR_EP_COHERENCE_WRITEBACK = 3
} bcir_ep_coherence;
#define BCIR_EP_COHERENCE_MAX 3

/* The 64-byte header IS the wire layout: the three u64 fields sit at 40/48/56. */
typedef struct bcir_ep_header {
  uint8_t  magic[4];        /* "BPLN" */
  uint16_t version;         /* 1..BCIR_EP_VERSION_MAX */
  uint16_t flags;           /* reserved (0) */
  uint8_t  mode;            /* bcir_ep_mode */
  uint8_t  liveness;        /* @9 v2: bcir_ep_liveness (reads PHASE on a v1 plan; reserved there) */
  uint8_t  reserved0[2];    /* 10..11 (0) */
  uint32_t streams;         /* @12 affinity domains the plan was placed on (the tail is extra) */
  uint32_t knee;            /* @16 the bandwidth knee the dispatch clamped to; 1..streams */
  uint32_t n_steps;         /* @20 record counts in the body */
  uint32_t n_lifetimes;     /* @24 */
  uint32_t n_moves;         /* @28 */
  uint32_t n_gens;          /* @32 */
  uint8_t  reserved1[4];    /* 36..39 (0) */
  uint64_t makespan;        /* @40 */
  uint64_t module_hash;     /* @48 R13 hash_module of the goal graph the plan realizes */
  uint64_t target_hash;     /* @56 R13 hash_target of the target it was placed for (0 = none) */
} bcir_ep_header;

BCIR_STATIC_ASSERT(sizeof(bcir_ep_header) == BCIR_EP_HEADER_SIZE,
                   "BCIR ExecutionPlan header must be exactly 64 bytes (frozen ABI v1)");

/* Zero-copy views of the body records (pointers into the buffer). */
typedef struct bcir_ep_step_view {
  uint64_t claim_id;
  uint32_t phase_id;
  const char *candidate; uint16_t candidate_len;
  uint8_t  lane;            /* bcir_lane */
  uint32_t width;
  int64_t  cost;            /* the plan's own step cost (signed) */
  uint32_t stream;          /* affinity domain, or BCIR_EP_TAIL_STREAM */
  uint64_t start;
  uint64_t duration;        /* == max(0, cost) */
} bcir_ep_step_view;

typedef struct bcir_ep_lifetime_view {
  uint32_t rid;
  const char *bank; uint16_t bank_len;
  uint64_t offset;
  uint64_t size;
  uint32_t alignment;
  uint32_t first_phase;
  uint32_t last_phase;
  uint64_t first_tick;      /* v2: the half-open liveness interval in the header's domain; */
  uint64_t last_tick;       /*     a v1 record reads [first_phase, last_phase + 1)          */
} bcir_ep_lifetime_view;

typedef struct bcir_ep_move_view {
  uint32_t rid;
  const char *src_bank; uint16_t src_len;
  const char *dst_bank; uint16_t dst_len;
  uint64_t offset;
  uint64_t size;
  const char *route; uint16_t route_len;
  uint8_t  kind;            /* bcir_ep_move_kind */
  uint8_t  coherence;       /* bcir_ep_coherence */
  uint32_t map_gen;
  uint32_t data_gen;
  uint64_t after_claim;     /* the overlap window: after this claim finishes (0 = none) */
  uint64_t before_claim;    /* ... and before this claim starts (0 = none) */
} bcir_ep_move_view;

typedef int (*bcir_ep_step_fn)(const bcir_ep_step_view *step, void *ctx);
typedef int (*bcir_ep_lifetime_fn)(const bcir_ep_lifetime_view *lifetime, void *ctx);
typedef int (*bcir_ep_move_fn)(const bcir_ep_move_view *move, void *ctx);

/* Validate magic + version + reserved bytes + CRC and copy the header out. A trust boundary:
 * every field is bounds-checked; a malformed/hostile buffer returns a status, never an OOB
 * read (fuzzed alongside the StreamPack decoder, runtime/c/fuzz_streampack.c). */
BCIR_NODISCARD bcir_status bcir_ep_validate(const uint8_t *BCIR_RESTRICT data, size_t len,
                                            bcir_ep_header *BCIR_RESTRICT hdr);

/* The wire laws over the whole body (see the header comment), ending with exact body
 * consumption. BCIR_ERR_PLAN names a record that violates a plan law; BCIR_ERR_LANE /
 * BCIR_ERR_WIDTH the range gate; BCIR_ERR_PROVENANCE a duplicated claim; BCIR_ERR_GENERATION
 * an unsorted vector; BCIR_ERR_OVERFLOW an interval past the 64-bit line; BCIR_ERR_TRAILING
 * undeclared bytes before the CRC. Mirrors bcir/abi/execution_plan_abi.py::validate_plan. */
BCIR_NODISCARD bcir_status bcir_ep_verify(const uint8_t *BCIR_RESTRICT data, size_t len);

/* The plan's name (the pack's `source_plan` must equal it); validates first. */
BCIR_NODISCARD const char *bcir_ep_source_plan(const uint8_t *BCIR_RESTRICT data, size_t len,
                                               uint16_t *BCIR_RESTRICT out_len);

/* Verify (bcir_ep_verify), then walk one record family in order, invoking `fn`; return
 * nonzero from `fn` to stop early. Every declared record is still parsed. */
BCIR_NODISCARD bcir_status bcir_ep_for_each_step(const uint8_t *BCIR_RESTRICT data, size_t len,
                                                 bcir_ep_step_fn fn, void *ctx);
BCIR_NODISCARD bcir_status bcir_ep_for_each_lifetime(const uint8_t *BCIR_RESTRICT data,
                                                     size_t len, bcir_ep_lifetime_fn fn,
                                                     void *ctx);
BCIR_NODISCARD bcir_status bcir_ep_for_each_move(const uint8_t *BCIR_RESTRICT data, size_t len,
                                                 bcir_ep_move_fn fn, void *ctx);
BCIR_NODISCARD bcir_status bcir_ep_for_each_generation(const uint8_t *BCIR_RESTRICT data,
                                                       size_t len, bcir_gen_fn fn, void *ctx);

/* R11 for the plan: `live[0..n_live)` is the caller's live registry (any order). The plan
 * is STALE (BCIR_ERR_STALE) unless every vector entry names a live resource with exactly
 * its generation AND every live resource has an entry -- the same predicate as
 * bcir_sp_check_generation_vector, so a plan minted under an older vector is refused
 * before the pack derived from it is. A plan with no vector is stale against any registry
 * that declares resources. */
BCIR_NODISCARD bcir_status bcir_ep_check_generation_vector(
    const uint8_t *BCIR_RESTRICT plan, size_t plan_len,
    const bcir_generation_view *BCIR_RESTRICT live, size_t n_live);

/* R10/R11 across the two artifacts: the pack `pack[0..pack_len)` is the lowering of the
 * plan `plan[0..plan_len)`. Both are verified (their statuses pass through); then the
 * pack's `source_plan` must equal the plan's, the pack must carry exactly one segment per
 * plan step, in step order, with the step's claim, phase, lane and width
 * (BCIR_ERR_PROVENANCE otherwise), and the two generation vectors must be identical entry
 * for entry -- a pack whose plan carries an older (or a different) vector for any resource
 * is STALE (BCIR_ERR_STALE). Mirrors bcir/verify::verify_execution_plan(pack=...). */
BCIR_NODISCARD bcir_status bcir_ep_check_pack(const uint8_t *BCIR_RESTRICT plan, size_t plan_len,
                                              const uint8_t *BCIR_RESTRICT pack, size_t pack_len);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BCIR_EXECUTION_PLAN_H */
