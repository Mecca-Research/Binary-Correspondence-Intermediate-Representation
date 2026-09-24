/*===- bcir_handoff.h - the data-plane hand-off: a pack table with explicit lifetime ===
 *
 * The freestanding half of the C <-> C++ seam (GEM+ roadmap G16, staged plan S3-C). The Python
 * reference is bcir/gem/handoff.py; the C++ RAII seam over it is runtime/cpp/bcir_handoff.hpp;
 * the contract is docs/languages/CPP_HANDOFF_BOUNDARY.md. The rails decide every operation
 * identically -- outcome, handle, resident generation and the table's and the plane's state
 * digests after every step (tools/perf/gemplus_baseline.py --group handoff).
 *
 * BORROWED VIEWS WITH EXPLICIT LIFETIME. A table owns `n_slots` fixed-size slots of one arena (the
 * caller's storage: the driver memory class, handles plus offsets). A producer reserves a slot,
 * writes the artifact into it ONCE through bcir_ho_reserved and commits it: frozen, and no API
 * writes it again. Consumers borrow it through a handle (slot index, epoch). Releasing the owner
 * advances the epoch, so every handle and view of that incarnation is refused from then on
 * (BCIR_ERR_LIFETIME) -- a view that outlives its owner is refused, never read. A borrow already
 * in progress keeps its bytes: a pinned slot RETIRES and its last return frees it, so a slot is
 * never reused under a reader. An epoch never wraps: a slot released at the last epoch leaves
 * service for good (EXHAUSTED). Returning a borrow exactly once is the caller's obligation (the
 * C++ Borrow guard is move-only and returns in its destructor); the table refuses a return it
 * can prove wrong (no pin, another incarnation).
 *
 * GENERATION GATING AT ADMIT(). bcir_ho_admit asks the LIVE control plane (G14) -- its one
 * predicate, bcir_ctl_admit_pack, against the installed registry -- and records the resident
 * generation and registry digest it admitted at. bcir_ho_dispatch runs only an artifact admitted
 * in this incarnation at the resident generation of the same registry (a switch between admit and
 * dispatch makes it stale), and runs the walk as a PHASE of the plane (bcir_ctl_enter /
 * bcir_ctl_leave): a switch requested mid-dispatch is deferred to the boundary, and new work is
 * refused while the plane drains.
 *
 * No internal synchronization: the owner serializes every operation (the C++ PackArena holds a
 * mutex around them; a borrowed view is read outside it). Freestanding: <stddef.h> + <stdint.h>
 * (through bcir_runtime.h), no heap, no libc.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_HANDOFF_H
#define BCIR_HANDOFF_H

#include "bcir_control_plane.h"
#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_HO_SLOTS_MAX   65535u
#define BCIR_HO_EPOCH_FIRST 1u          /* a zeroed handle {0, 0} is never live */
#define BCIR_HO_EPOCH_MAX   0xFFFFFFFFu
#define BCIR_HO_PINS_MAX    0xFFFFFFFFu

typedef enum bcir_ho_slot_state {
  BCIR_HO_FREE      = 0,
  BCIR_HO_BUILDING  = 1,  /* reserved: the producer is writing */
  BCIR_HO_FROZEN    = 2,  /* committed: immutable, borrowable */
  BCIR_HO_RETIRED   = 3,  /* released while borrowed: freed by the last return */
  BCIR_HO_EXHAUSTED = 4   /* released at the last epoch: out of service for good */
} bcir_ho_slot_state;

typedef enum bcir_ho_verdict { BCIR_HO_APPLIED = 0, BCIR_HO_REFUSED = 1 } bcir_ho_verdict;

typedef enum bcir_ho_refusal {
  BCIR_HO_REFUSAL_NONE       = 0,
  BCIR_HO_REFUSAL_LIFETIME   = 1,  /* no live slot at the handle's epoch; a return that was no borrow */
  BCIR_HO_REFUSAL_FULL       = 2,  /* no free slot */
  BCIR_HO_REFUSAL_CAPACITY   = 3,  /* a length outside [1, slot capacity] */
  BCIR_HO_REFUSAL_STALE      = 4,  /* the plane refused it as stale / an admission of a past generation */
  BCIR_HO_REFUSAL_UNADMITTED = 5,  /* dispatch with no admission in this incarnation */
  BCIR_HO_REFUSAL_MALFORMED  = 6,  /* the plane refused it as malformed (its own status passes) */
  BCIR_HO_REFUSAL_DRAINING   = 7,  /* the plane refused the dispatch phase while draining */
  BCIR_HO_REFUSAL_EXHAUSTED  = 8,  /* a pin or plane counter at its bound */
  BCIR_HO_REFUSAL_WALK       = 9   /* the segment walk failed (its status passes) */
} bcir_ho_refusal;
#define BCIR_HO_REFUSAL_MAX 9

typedef struct bcir_ho_handle {
  uint32_t index;
  uint32_t epoch;
} bcir_ho_handle;

/* A borrow in progress: valid until returned. */
typedef struct bcir_ho_view {
  uint32_t index;
  uint32_t epoch;
  const uint8_t *data;
  size_t len;
} bcir_ho_view;

typedef struct bcir_ho_slot {
  uint32_t epoch;
  uint8_t state;                /* bcir_ho_slot_state */
  uint8_t admitted;
  uint16_t reserved;
  uint32_t pins;
  uint32_t admitted_generation;
  uint8_t admitted_registry[32];
  size_t length;
} bcir_ho_slot;

typedef struct bcir_ho_table {
  bcir_ho_slot *slots;   /* the caller's n_slots entries */
  uint32_t n_slots;
  size_t capacity;       /* bytes per slot */
  uint8_t *arena;        /* the caller's n_slots * capacity bytes */
} bcir_ho_table;

typedef struct bcir_ho_outcome {
  uint8_t verdict;       /* bcir_ho_verdict */
  uint8_t refusal;       /* bcir_ho_refusal */
  uint16_t reserved;
  uint32_t generation;   /* the plane's resident generation afterwards (0: no plane involved) */
  bcir_ho_handle handle; /* the handle issued (reserve) or acted on; {0, 0} otherwise */
  bcir_status status;
} bcir_ho_outcome;

/* A table over the caller's storage: 1..65535 slots of 1..2^32-1 bytes each, the arena at least
 * n_slots * slot_capacity bytes (computed without overflow). Every slot starts FREE at epoch 1.
 * BCIR_ERR_NOSPACE on a short arena or a bad shape; the table is zeroed on failure. */
BCIR_NODISCARD bcir_status bcir_ho_init(bcir_ho_table *BCIR_RESTRICT table,
                                        bcir_ho_slot *BCIR_RESTRICT slots, uint32_t n_slots,
                                        uint8_t *BCIR_RESTRICT arena, size_t arena_len,
                                        size_t slot_capacity);

/* The producer. `reserve` takes the lowest FREE slot (BUILDING); `reserved` returns its write
 * region (BCIR_ERR_LIFETIME unless the handle names a building slot); `commit` freezes its first
 * `length` bytes; `abort` frees it with nothing published and its epoch advanced. */
BCIR_NODISCARD bcir_ho_outcome bcir_ho_reserve(bcir_ho_table *table);
BCIR_NODISCARD bcir_status bcir_ho_reserved(const bcir_ho_table *BCIR_RESTRICT table,
                                            bcir_ho_handle handle, uint8_t **BCIR_RESTRICT data,
                                            size_t *BCIR_RESTRICT capacity);
BCIR_NODISCARD bcir_ho_outcome bcir_ho_commit(bcir_ho_table *table, bcir_ho_handle handle,
                                              size_t length);
BCIR_NODISCARD bcir_ho_outcome bcir_ho_abort(bcir_ho_table *table, bcir_ho_handle handle);

/* The consumers. `borrow` pins a FROZEN slot of the handle's epoch and hands back its bytes;
 * `give_back` returns a borrow (a frozen slot of its epoch, or one retired or exhausted from it);
 * `release` ends the incarnation -- every handle and view of it is refused from now on. */
BCIR_NODISCARD bcir_ho_outcome bcir_ho_borrow(bcir_ho_table *BCIR_RESTRICT table,
                                              bcir_ho_handle handle,
                                              bcir_ho_view *BCIR_RESTRICT view);
BCIR_NODISCARD bcir_ho_outcome bcir_ho_give_back(bcir_ho_table *BCIR_RESTRICT table,
                                                 const bcir_ho_view *BCIR_RESTRICT view);
BCIR_NODISCARD bcir_ho_outcome bcir_ho_release(bcir_ho_table *table, bcir_ho_handle handle);

/* The plane. `admit`: borrow, bcir_ctl_admit_pack on the live plane, return; applied records
 * the resident generation and registry digest, a refusal revokes any earlier admission.
 * `dispatch`: borrow; refused unless admitted in this incarnation at the plane's resident
 * generation of the same registry; bcir_ctl_enter (refused while draining); walk the segments
 * through `fn` (bcir_sp_for_each_segment); bcir_ctl_leave (a deferred switch lands here, once);
 * return. `admit_manifest`: a shard manifest's tags and registry digest against the installed
 * registry, before any shard is fetched. */
BCIR_NODISCARD bcir_ho_outcome bcir_ho_admit(bcir_ho_table *BCIR_RESTRICT table,
                                             bcir_ho_handle handle,
                                             bcir_ctl_state *BCIR_RESTRICT plane);
BCIR_NODISCARD bcir_ho_outcome bcir_ho_dispatch(bcir_ho_table *BCIR_RESTRICT table,
                                                bcir_ho_handle handle,
                                                bcir_ctl_state *BCIR_RESTRICT plane,
                                                bcir_seg_fn fn, void *ctx);
BCIR_NODISCARD bcir_ho_outcome bcir_ho_admit_manifest(const bcir_ctl_state *BCIR_RESTRICT plane,
                                                      const uint8_t *BCIR_RESTRICT manifest,
                                                      size_t len);

/* SHA-256 over the canonical serialization of the table (per slot: epoch, state, admitted, pins,
 * admitted generation, the held length, the admitted registry and the SHA-256 of the held bytes)
 * -- the value the rails compare after every operation. */
void bcir_ho_state_digest(const bcir_ho_table *BCIR_RESTRICT table, uint8_t out[32]);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BCIR_HANDOFF_H */
