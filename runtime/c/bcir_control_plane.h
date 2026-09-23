/*===- bcir_control_plane.h - BCIR ControlRecordV1 ABI + the resident plane ---===
 *
 * The normative C view of the control plane as bytes (GEM+ roadmap G14, staged plan S3-A):
 * lease, generation, quiescence, activation, rollback and cancellation as small, fixed,
 * versioned records that a resident runtime decides by their bytes. The Python reference
 * codec is bcir/abi/control_abi.py and the reference plane bcir/gem/control.py;
 * docs/kernel/BCIR_CONTROL_PLANE_ABI.md is the prose spec. The rails agree record for record
 * (Python encode -> C decode -> Python re-encode, byte-identical) and decision for decision
 * (every scenario's verdicts, refusals, statuses and resident state digests), held by the
 * G14 rows (tools/perf/gemplus_baseline.py --group control) and tools/c/check_runtime.sh.
 *
 * Wire format (little-endian; every field fixed-width -- there is no variable-length field):
 *   ControlRecordV1 := header(64) || body(bcir_ctl_body_bytes(version, kind)) || mac[32] || crc32
 *     mac   = HMAC-SHA256(K, header || body): K is the plane's root key on a lease grant
 *             (lease == 0), bcir_ctl_lease_key(root, lease) otherwise
 *     crc32 = CRC-32 (bcir_crc32, zlib-compatible) of every preceding byte
 *   The CRC is the corruption gate a keyless reader applies; the MAC is authority. An HMAC
 *   proves possession of a key, not the identity of a signer -- the field is `mac`, never
 *   `signature`.
 *
 * Freestanding: <stddef.h> + <stdint.h> (through bcir_runtime.h and bcir_sha256.h), no heap,
 * no libc. The plane is a fixed-size value the caller owns (eight leases, one 192-byte pending
 * slot). This is a frozen BCIR artifact format, NOT the BCIR UAPI: the driver roadmap keeps
 * the UAPI unfrozen, and a future ioctl or IPC adapter marshals these values.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_CONTROL_PLANE_H
#define BCIR_CONTROL_PLANE_H

#include "bcir_runtime.h"
#include "bcir_sha256.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_CTL_MAGIC        "BCTL"  /* bytes 0..3 of the header */
#define BCIR_CTL_VERSION      1
#define BCIR_CTL_VERSION_MAX  1
#define BCIR_CTL_HEADER_SIZE  64u
#define BCIR_CTL_MAC_SIZE     32u
#define BCIR_CTL_TRAILER_SIZE 36u     /* mac[32] || crc32 */
#define BCIR_CTL_MIN_BYTES    (BCIR_CTL_HEADER_SIZE + BCIR_CTL_TRAILER_SIZE)
/* The declared bound on any record (the largest v1 record is 164 bytes); three cache lines,
 * the slot stride the telemetry/control ring (G15) may adopt. */
#define BCIR_CTL_RECORD_MAX   192u
#define BCIR_CTL_LEASE_CAPACITY 8u    /* the resident lease table is a fixed array */
#define BCIR_CTL_KEY_MIN      16u     /* a plane's root key: 128 bits at least ... */
#define BCIR_CTL_KEY_MAX      64u     /* ... one HMAC block at most */

/* The six kinds (wire codes). Switches -- generation, activate, rollback -- move the resident
 * generation; the others act without moving it. */
typedef enum bcir_ctl_kind {
  BCIR_CTL_LEASE      = 1,  /* the root grants a lease: a capability mask over a boundary window */
  BCIR_CTL_GENERATION = 2,  /* install a registry state (the generation vector's maxima + digest) */
  BCIR_CTL_QUIESCE    = 3,  /* drain: new work is refused until a switch lands or the deadline */
  BCIR_CTL_ACTIVATE   = 4,  /* install an artifact in place of the live one */
  BCIR_CTL_ROLLBACK   = 5,  /* restore the previous artifact (the generation moves FORWARD) */
  BCIR_CTL_CANCEL     = 6   /* withdraw the issuer's own pending switch */
} bcir_ctl_kind;
#define BCIR_CTL_KIND_MAX 6

/* What `subject` names. */
typedef enum bcir_ctl_scope {
  BCIR_CTL_SCOPE_MODULE   = 0,
  BCIR_CTL_SCOPE_RESOURCE = 1,
  BCIR_CTL_SCOPE_MAPPING  = 2,
  BCIR_CTL_SCOPE_SESSION  = 3,
  BCIR_CTL_SCOPE_CHANNEL  = 4
} bcir_ctl_scope;
#define BCIR_CTL_SCOPE_MAX 4

/* The capability a record exercises is exactly its kind's bit; a lease may grant any of them
 * except the lease bit (v1 has no delegation). */
#define BCIR_CTL_CAP(kind)     (UINT64_C(1) << ((kind) - 1))
#define BCIR_CTL_CAP_ALL       UINT64_C(0x3F)
#define BCIR_CTL_CAP_GRANTABLE UINT64_C(0x3E)

/* Why a record was issued: a closed set per kind, 0 = none everywhere.
 *   lease: none | generation: remap 1, rewrite 2, topology 3 | quiesce: activation 1,
 *   rollback 2, teardown 3 | activate: promotion 1, repair 2 | rollback: correctness 1,
 *   health 2, policy 3 | cancel: withdrawn 1, superseded 2 */
#define BCIR_CTL_REASON_NONE 0

/* The 64-byte header IS the wire layout (one cache line, every u64 at an 8-aligned offset).
 * Decoders read it field by field, little-endian; the struct is the decoded view. */
typedef struct bcir_ctl_header {
  uint8_t  magic[4];     /* "BCTL" */
  uint16_t version;      /* @4  1..BCIR_CTL_VERSION_MAX */
  uint16_t flags;        /* @6  reserved (0) */
  uint8_t  kind;         /* @8  bcir_ctl_kind */
  uint8_t  scope;        /* @9  bcir_ctl_scope */
  uint8_t  reason;       /* @10 the kind's closed reason code */
  uint8_t  reserved0;    /* @11 reserved (0) */
  uint32_t body_len;     /* @12 == bcir_ctl_body_bytes(version, kind): a keyless reader bounds
                          *     the record before it trusts the kind table */
  uint32_t generation;   /* @16 the generation the record asserts (the NEW one for a switch) */
  uint32_t expect;       /* @20 the resident generation the issuer witnessed (compare-and-swap) */
  uint64_t capability;   /* @24 == BCIR_CTL_CAP(kind) */
  uint64_t boundary;     /* @32 the earliest phase/event boundary the record may take effect at */
  uint64_t sequence;     /* @40 the issuer's monotone sequence under its lease (>= 1) */
  uint64_t lease;        /* @48 the lease it is issued under; 0 only on a lease grant */
  uint64_t subject;      /* @56 the handle it acts on (`scope` says which kind of handle) */
} bcir_ctl_header;

BCIR_STATIC_ASSERT(sizeof(bcir_ctl_header) == BCIR_CTL_HEADER_SIZE,
                   "BCIR ControlRecordV1 header must be exactly 64 bytes (frozen ABI v1)");

/* The fixed bodies (decoded views; the wire is little-endian, fields in this order). */
typedef struct bcir_ctl_lease_body {       /* kind 1, 40 bytes */
  uint64_t lease_id;       /* never 0, never reused */
  uint64_t granted;        /* nonempty subset of BCIR_CTL_CAP_GRANTABLE */
  uint64_t issued_epoch;   /* the lease is valid for boundaries [issued_epoch, expiry_epoch) */
  uint64_t expiry_epoch;
  uint64_t holder;         /* the principal (never 0) */
} bcir_ctl_lease_body;
typedef struct bcir_ctl_generation_body {  /* kind 2, 48 bytes */
  uint32_t map_gen;        /* the per-resource vector's maxima ... */
  uint32_t data_gen;
  uint32_t topo_gen;       /* ... the topology generation ... */
  uint32_t reserved;       /* (0: BCIR_ERR_RESERVED otherwise) */
  uint8_t  registry_digest[32];  /* ... and bcir_ctl_registry_digest of the vector itself */
} bcir_ctl_generation_body;
typedef struct bcir_ctl_quiesce_body {     /* kind 3, 8 bytes */
  uint64_t drain_deadline; /* >= the record's boundary */
} bcir_ctl_quiesce_body;
typedef struct bcir_ctl_activate_body {    /* kind 4, 64 bytes */
  uint8_t artifact_sha256[32];   /* nonzero, distinct from previous_sha256 */
  uint8_t previous_sha256[32];   /* the live artifact it replaces (all-zero = none) */
} bcir_ctl_activate_body;
typedef struct bcir_ctl_rollback_body {    /* kind 5, 64 bytes */
  uint8_t restore_sha256[32];    /* the previous artifact, restored */
  uint8_t rollback_token[32];    /* bcir_ctl_rollback_token of the activation it undoes */
} bcir_ctl_rollback_body;
typedef struct bcir_ctl_cancel_body {      /* kind 6, 16 bytes */
  uint64_t first_sequence; /* 1 <= first <= last < the cancel's own sequence */
  uint64_t last_sequence;
} bcir_ctl_cancel_body;

BCIR_STATIC_ASSERT(sizeof(bcir_ctl_lease_body) == 40, "lease body is 40 bytes (frozen v1)");
BCIR_STATIC_ASSERT(sizeof(bcir_ctl_generation_body) == 48, "generation body is 48 bytes");
BCIR_STATIC_ASSERT(sizeof(bcir_ctl_quiesce_body) == 8, "quiesce body is 8 bytes");
BCIR_STATIC_ASSERT(sizeof(bcir_ctl_activate_body) == 64, "activate body is 64 bytes");
BCIR_STATIC_ASSERT(sizeof(bcir_ctl_rollback_body) == 64, "rollback body is 64 bytes");
BCIR_STATIC_ASSERT(sizeof(bcir_ctl_cancel_body) == 16, "cancel body is 16 bytes");

/* One decoded record. */
typedef struct bcir_ctl_record {
  bcir_ctl_header hdr;
  union {
    bcir_ctl_lease_body lease;
    bcir_ctl_generation_body generation;
    bcir_ctl_quiesce_body quiesce;
    bcir_ctl_activate_body activate;
    bcir_ctl_rollback_body rollback;
    bcir_ctl_cancel_body cancel;
  } body;
  uint8_t mac[32];
} bcir_ctl_record;

/* The body length of `kind` at `version` (the append-only table); 0 when there is none. */
uint32_t bcir_ctl_body_bytes(uint16_t version, uint8_t kind);

/* The keyless framing laws 1-8 (length, magic, version, reserved header bytes, kind,
 * body_len, exact length, CRC) and a copy of the header. A trust boundary: every read is
 * bounds-checked, so a hostile buffer returns a status and is never read out of bounds. */
BCIR_NODISCARD bcir_status bcir_ctl_validate(const uint8_t *BCIR_RESTRICT data, size_t len,
                                             bcir_ctl_header *BCIR_RESTRICT hdr);

/* Every keyless wire law, 1-11, in the specification's order (so the rails name the same
 * first violation): the framing laws, then the header field laws (BCIR_ERR_CONTROL), the
 * generation body's reserved word (BCIR_ERR_RESERVED), the body laws (BCIR_ERR_CONTROL) and
 * a MAC that is not all zero (BCIR_ERR_MAC). `out` (may be NULL) receives the decoded record
 * on BCIR_OK. Mirrors bcir/abi/control_abi.py::decode_control. */
BCIR_NODISCARD bcir_status bcir_ctl_decode(const uint8_t *BCIR_RESTRICT data, size_t len,
                                           bcir_ctl_record *BCIR_RESTRICT out);
BCIR_NODISCARD bcir_status bcir_ctl_verify(const uint8_t *BCIR_RESTRICT data, size_t len);

/* The keyed check: the record verifies (its status passes through) and its MAC is the one
 * `key` produces over header || body, compared in constant time (BCIR_ERR_MAC otherwise).
 * `key` is the root key for a lease grant and the lease's key otherwise; NULL is valid only
 * with key_len 0, and an invalid pair is refused (BCIR_ERR_MAC) without being read. */
BCIR_NODISCARD bcir_status bcir_ctl_check_mac(const uint8_t *BCIR_RESTRICT data, size_t len,
                                              const uint8_t *BCIR_RESTRICT key, size_t key_len);

/* The capability-scoped key a lease holder receives:
 * HMAC-SHA256(root, "BCTL/lease/v1" || 0x00 || u64le(lease_id)). A NULL root is valid only
 * with root_len 0; an invalid pair is refused (BCIR_ERR_CONTROL, `out` zeroed), never read as
 * the empty key. */
BCIR_NODISCARD bcir_status bcir_ctl_lease_key(const uint8_t *BCIR_RESTRICT root, size_t root_len,
                                              uint64_t lease_id, uint8_t out[32]);

/* SHA-256("BCTL/registry/v1" || 0x00 || vector), the vector as rid:u32 map_gen:u32
 * data_gen:u32 per resource -- exactly the StreamPack v4 / ExecutionPlanV1 generation tail.
 * RIDs must be strictly ascending (BCIR_ERR_GENERATION otherwise; `out` is then untouched). */
BCIR_NODISCARD bcir_status bcir_ctl_registry_digest(const bcir_generation_view *BCIR_RESTRICT vector,
                                                    size_t n, uint8_t out[32]);

/* The token an activation mints: SHA-256("BCTL/token/v1" || 0x00 || header || body) of a
 * verified activate record (its status passes through; BCIR_ERR_CONTROL for another kind).
 * Public by construction -- a binding of which activation a rollback undoes, not a secret. */
BCIR_NODISCARD bcir_status bcir_ctl_rollback_token(const uint8_t *BCIR_RESTRICT data, size_t len,
                                                   uint8_t out[32]);

/* --- the resident plane ------------------------------------------------------------------ */

/* How an operation ended. `none`: succeeded with no record decided. */
typedef enum bcir_ctl_verdict {
  BCIR_CTL_APPLIED  = 0,
  BCIR_CTL_DEFERRED = 1,  /* a switch held for its boundary: success-with-a-decision, not an error */
  BCIR_CTL_REFUSED  = 2,
  BCIR_CTL_NONE     = 3
} bcir_ctl_verdict;

/* Why the plane refused -- the same closed set, in the same order, as
 * bcir/gem/control.py::REFUSALS. */
typedef enum bcir_ctl_refusal {
  BCIR_CTL_REFUSAL_NONE       = 0,
  BCIR_CTL_REFUSAL_MALFORMED  = 1,   /* a wire law (the status names it) */
  BCIR_CTL_REFUSAL_LEASE      = 2,   /* no such lease */
  BCIR_CTL_REFUSAL_MAC        = 3,   /* the MAC is not the root's (grant) or the lease's */
  BCIR_CTL_REFUSAL_SUBJECT    = 4,   /* minted for another (scope, subject) */
  BCIR_CTL_REFUSAL_EXPIRED    = 5,   /* outside its lease's window; a grant or drain already past */
  BCIR_CTL_REFUSAL_CAPABILITY = 6,   /* the lease does not grant the record's capability */
  BCIR_CTL_REFUSAL_REPLAY     = 7,   /* sequence not above the issuer's last accepted one */
  BCIR_CTL_REFUSAL_STALE      = 8,   /* minted against a generation the plane has left */
  BCIR_CTL_REFUSAL_AHEAD      = 9,   /* minted against a generation the plane never reached */
  BCIR_CTL_REFUSAL_EARLY      = 10,  /* a non-switch whose boundary is ahead of the plane's */
  BCIR_CTL_REFUSAL_PENDING    = 11,  /* a switch while another is pending */
  BCIR_CTL_REFUSAL_MISMATCH   = 12,  /* the content witness disagrees with the resident state */
  BCIR_CTL_REFUSAL_NOTHING    = 13,  /* nothing to roll back / no pending switch to cancel */
  BCIR_CTL_REFUSAL_FULL       = 14,  /* the lease table is full of live leases */
  BCIR_CTL_REFUSAL_DUPLICATE  = 15,  /* a lease id that is not above every id ever granted */
  BCIR_CTL_REFUSAL_DRAINING   = 16,  /* enter while draining */
  BCIR_CTL_REFUSAL_BUSY       = 17,  /* advance while a phase is in flight */
  BCIR_CTL_REFUSAL_IDLE       = 18,  /* leave with nothing in flight */
  BCIR_CTL_REFUSAL_EXHAUSTED  = 19   /* a counter would wrap */
} bcir_ctl_refusal;
#define BCIR_CTL_REFUSAL_MAX 19

typedef struct bcir_ctl_outcome {
  uint8_t     verdict;     /* bcir_ctl_verdict */
  uint8_t     refusal;     /* bcir_ctl_refusal */
  uint8_t     kind;        /* the decided record's kind; 0 when none was decided */
  uint8_t     reserved;
  uint32_t    generation;  /* the resident generation afterwards */
  uint64_t    sequence;    /* the decided record's sequence; 0 when none */
  bcir_status status;      /* the wire law's status on a malformed record, BCIR_ERR_MAC on a MAC
                            * refusal, the artifact's own status on an admission */
} bcir_ctl_outcome;

typedef struct bcir_ctl_lease_entry {
  uint64_t lease_id;
  uint64_t granted;
  uint64_t issued;         /* valid for boundaries [issued, expiry) */
  uint64_t expiry;
  uint64_t holder;
  uint64_t last_sequence;  /* the last accepted record's sequence under this lease */
  uint8_t  key[32];        /* bcir_ctl_lease_key(root, lease_id), derived once when the grant is
                            * applied (G15/S3-B: a leased record's MAC is checked against it
                            * instead of re-deriving it per record). Not state: the state digest
                            * does not cover it, and it is wiped when the lease leaves the table. */
} bcir_ctl_lease_entry;

/* One handle's resident control state: the mirror of bcir/gem/control.py::ControlPlane.
 * It holds the root key the way the authority does; an issuer holds only its lease key. */
typedef struct bcir_ctl_state {
  uint8_t  key[BCIR_CTL_KEY_MAX];
  uint32_t key_len;
  uint8_t  scope;            /* the plane's identity: (scope, subject) */
  uint8_t  draining;
  uint8_t  has_registry;
  uint8_t  reserved0;
  uint64_t subject;
  uint32_t generation;       /* the resident generation (starts at 0) */
  uint32_t reg_map_gen;      /* the installed registry state (when has_registry) */
  uint32_t reg_data_gen;
  uint32_t reg_topo_gen;
  uint8_t  reg_digest[32];
  uint8_t  artifact[32];     /* the live artifact, its rollback target and the live */
  uint8_t  previous[32];     /* activation's token (all-zero = none) */
  uint8_t  token[32];
  uint64_t boundary;         /* phase/event boundaries crossed */
  uint64_t in_flight;        /* phases in progress */
  uint64_t drain_deadline;
  uint64_t root_sequence;    /* the root's last accepted sequence (its grants) */
  uint64_t last_lease_id;    /* the largest lease id ever granted (ids never recur) */
  uint32_t n_leases;
  uint32_t pending_len;      /* 0 = no pending switch */
  bcir_ctl_lease_entry leases[BCIR_CTL_LEASE_CAPACITY];  /* ascending by lease_id */
  uint8_t  pending[BCIR_CTL_RECORD_MAX];                 /* the deferred switch's own verified bytes */
} bcir_ctl_state;

/* A fresh plane for the handle (scope, subject) under a 16..64-byte root key
 * (BCIR_ERR_CONTROL otherwise, and the state is left zeroed). */
BCIR_NODISCARD bcir_status bcir_ctl_init(bcir_ctl_state *BCIR_RESTRICT state,
                                         const uint8_t *BCIR_RESTRICT key, size_t key_len,
                                         uint8_t scope, uint64_t subject);

/* Decide one record: applied, deferred or refused, the checks in the specification's order
 * (malformed, lease, mac, subject, expired, capability, replay, stale/ahead, the kind laws,
 * timing). A refused record changes nothing -- not a sequence counter, not a lease, not the
 * pending slot. A switch applies only at a boundary: with a phase in flight, or before its
 * `boundary`, it is deferred -- its sequence consumed and its bytes held; the resident
 * generation does not move. */
BCIR_NODISCARD bcir_ctl_outcome bcir_ctl_submit(bcir_ctl_state *BCIR_RESTRICT state,
                                                const uint8_t *BCIR_RESTRICT data, size_t len);

/* The boundaries. `enter` counts a phase in (refused while draining); `leave` counts one out,
 * and the leave that returns the plane to quiescence crosses a boundary; `advance` crosses an
 * event boundary while nothing is in flight. Crossing: the counter advances (never wraps), a
 * drain past its deadline lapses, and a pending switch that is due is re-checked against its
 * lease's window and applied -- once, there -- or refused there and reported. */
BCIR_NODISCARD bcir_ctl_outcome bcir_ctl_enter(bcir_ctl_state *state);
BCIR_NODISCARD bcir_ctl_outcome bcir_ctl_leave(bcir_ctl_state *state);
BCIR_NODISCARD bcir_ctl_outcome bcir_ctl_advance(bcir_ctl_state *state);

/* The data plane, by bytes. A StreamPack (bcir_sp_verify_semantic first, its status passing
 * through as a malformed refusal) is admitted only if its header maxima and topo_gen are the
 * installed registry's and its generation vector digests to the installed registry_digest; an
 * ExecutionPlanV1 (bcir_ep_verify first) only if its vector does. With no registry installed
 * nothing is admitted. Stale artifacts are refused with BCIR_ERR_STALE. */
BCIR_NODISCARD bcir_ctl_outcome bcir_ctl_admit_pack(bcir_ctl_state *BCIR_RESTRICT state,
                                                    const uint8_t *BCIR_RESTRICT data, size_t len);
BCIR_NODISCARD bcir_ctl_outcome bcir_ctl_admit_plan(bcir_ctl_state *BCIR_RESTRICT state,
                                                    const uint8_t *BCIR_RESTRICT data, size_t len);

/* SHA-256 over the canonical serialization of the resident state (the key excluded): the
 * value two rails compare after every operation. */
void bcir_ctl_state_digest(const bcir_ctl_state *BCIR_RESTRICT state, uint8_t out[32]);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BCIR_CONTROL_PLANE_H */
