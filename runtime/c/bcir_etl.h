/*===- bcir_etl.h - the event transducer in C (the C twin of bcir/etl) -----===
 *
 * PORT-4 (N4 of the native port audit): the Event Transduction Layer's three oracle modules, freestanding --
 *   bcir_fsm_*    a finite-state transducer over a symbol stream      (bcir/etl/fsm.py)
 *   bcir_event_*  the event-stream, event-kind and event descriptors   (bcir/etl/events.py)
 *   bcir_rop_*    the mini-ROP grammar: text -> tokens -> claims       (bcir/etl/parse.py)
 * Driver packets and device text are untrusted, so this is a trust boundary: every input is bounded by its length,
 * every output goes to a buffer the caller sized, nothing allocates, and every path returns a status -- the verdict
 * the oracle gives, or the capacity the caller did not provide. Each refusal names the element or offset the
 * oracle's exception names, so a caller can spell it (`bcir_etl_tool.c` spells each in the oracle's words, and
 * bcir/tests/test_c_etl.py holds the two rails to one answer).
 *
 * Depends only on <stddef.h> and <stdint.h>. Strings are (pointer, length) pairs, never NUL-terminated, so an
 * oracle string holding a NUL is one here too.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_ETL_H
#define BCIR_ETL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_etl_status {
  BCIR_ETL_OK = 0,
  BCIR_ETL_ERR_INVALID = 1,          /* a NULL pointer with a nonzero count, or no output */
  BCIR_ETL_ERR_CAPACITY = 2,         /* an output buffer the caller sized too small */
  /* bcir_fsm_compile: the oracle's ValueError, in the order it raises them */
  BCIR_ETL_ERR_STATE_NAME = 10,      /* a state of an empty name */
  BCIR_ETL_ERR_STATE_KIND = 11,      /* a state accepting and an error both */
  BCIR_ETL_ERR_TX_FIELD = 12,        /* a transition of an empty src, dst or on (`field` says which) */
  BCIR_ETL_ERR_STATE_DUP = 13,       /* a state's name another state has */
  BCIR_ETL_ERR_TX_ENDPOINT = 14,     /* a transition naming no declared state (`field` 0 src, 1 dst) */
  BCIR_ETL_ERR_TX_DUP = 15,          /* a second transition on one (state, symbol) */
  BCIR_ETL_ERR_START = 16,           /* a start state not declared */
  /* bcir_event_*_check */
  BCIR_ETL_ERR_STREAM_NAME = 20,
  BCIR_ETL_ERR_STREAM_KIND = 21,     /* not text|binary|telemetry|packet|driver|token */
  BCIR_ETL_ERR_STREAM_ENCODING = 22, /* not utf8|bytes|le|be|records|custom */
  BCIR_ETL_ERR_STREAM_BITS = 23,     /* element_bits < 1 */
  BCIR_ETL_ERR_STREAM_WINDOW = 24,   /* max_window < 1 */
  BCIR_ETL_ERR_KIND_NAME = 25,       /* an event kind of an empty name */
  BCIR_ETL_ERR_EVENT_KIND = 26,      /* an event of an empty kind */
  BCIR_ETL_ERR_EVENT_SPAN = 27,      /* an event span not 0 <= begin <= end */
  /* bcir_rop_parse */
  BCIR_ETL_ERR_LEX = 30,             /* a character no token rule takes, at `pos` */
  BCIR_ETL_ERR_EXPECT = 31,          /* `expected X, got <token>`: X a token kind or a keyword */
  BCIR_ETL_ERR_SEPARATOR = 32,       /* `expected ';' or '}', got <token>` */
  BCIR_ETL_ERR_STATEMENT = 33,       /* an unknown statement keyword */
  BCIR_ETL_ERR_LANE = 34,            /* an unknown lane name */
  BCIR_ETL_ERR_STRIDE = 35,          /* an unknown stride name */
  BCIR_ETL_ERR_COUNT = 36,           /* a count past the signed 64-bit wire domain */
  BCIR_ETL_ERR_REPEATED = 37         /* a statement a claim already made */
} bcir_etl_status;

typedef struct bcir_etl_str { const char *s; size_t n; } bcir_etl_str;

/* --- the finite-state transducer (bcir/etl/fsm.py) ------------------------------------------------------------ */
typedef struct bcir_fsm_state { bcir_etl_str name; uint8_t accepting, error; } bcir_fsm_state;
/* `action` of length 0 captures nothing, as the oracle's None and "" do; `on` "*" is the catch-all edge. */
typedef struct bcir_fsm_transition { bcir_etl_str src, dst, on, action; } bcir_fsm_transition;
typedef struct bcir_fsm {
  const bcir_fsm_state *states; size_t n_states;
  const bcir_fsm_transition *tx; size_t n_tx;
  uint32_t *src, *dst;               /* each transition's state indexes, in arrays of n_tx the caller provides */
  size_t start;
} bcir_fsm;
/* The oracle's construction, every check in its order: each state's own fields, then each transition's, then a
 * repeated state name, then each transition's endpoints and (state, symbol) key, then the start. On a refusal,
 * *where is the state or transition it names and *field which end or field (else 0). */
bcir_etl_status bcir_fsm_compile(bcir_fsm *m, const bcir_fsm_state *states, size_t n_states,
                                 const bcir_fsm_transition *tx, size_t n_tx, bcir_etl_str start,
                                 uint32_t *src, uint32_t *dst, size_t *where, int *field);
/* One step of a run: transition `tx` taken on symbol `sym` (an index of the input). The oracle's trace is
 * (states[src[tx]], symbols[sym], states[dst[tx]]); its captures are the steps whose transition has an action. */
typedef struct bcir_fsm_step { uint32_t tx; size_t sym; } bcir_fsm_step;
/* Run `symbols` from the start: a symbol takes the edge on it, else the catch-all; a symbol with neither, or one
 * read in an error state, stops the run unaccepted there. Accepted when the run ends in an accepting state.
 * `trace` holds up to `cap` steps (n is always enough); a run that needs more is BCIR_ETL_ERR_CAPACITY. */
bcir_etl_status bcir_fsm_run(const bcir_fsm *m, const bcir_etl_str *symbols, size_t n,
                             bcir_fsm_step *trace, size_t cap, size_t *n_steps,
                             size_t *final_state, int *accepted);

/* --- the event descriptors (bcir/etl/events.py) --------------------------------------------------------------- */
bcir_etl_status bcir_event_stream_check(bcir_etl_str name, bcir_etl_str kind, bcir_etl_str encoding,
                                        int64_t element_bits, int64_t max_window);
bcir_etl_status bcir_event_kind_check(bcir_etl_str name);
bcir_etl_status bcir_event_check(bcir_etl_str kind, int64_t begin, int64_t end);

/* --- the mini-ROP grammar (bcir/etl/parse.py) ----------------------------------------------------------------- */
typedef enum bcir_rop_tok {
  BCIR_ROP_LBRACE = 0, BCIR_ROP_RBRACE, BCIR_ROP_SEMI, BCIR_ROP_COMMA, BCIR_ROP_INT, BCIR_ROP_IDENT, BCIR_ROP_EOF
} bcir_rop_tok;
const char *bcir_rop_tok_name(bcir_rop_tok kind);      /* "LBRACE", ..., "EOF": the oracle's token kinds */
typedef struct bcir_rop_resource { size_t at, n; uint32_t rid; } bcir_rop_resource;   /* text[at, at+n) */
typedef struct bcir_rop_claim {
  uint32_t id;                       /* 1000 + its index */
  size_t name_at, name_n;            /* `rop.<name>`: the name, text[name_at, name_at+name_n) */
  size_t rd_first, n_rd, wr_first, n_wr;   /* its reads and writes: rids[rd_first ...], rids[wr_first ...] */
  uint64_t count;                    /* 1 unless it says */
  uint8_t lane, stride;              /* bcir.model.Lane / StrideClass values: U and UNIT unless it says */
} bcir_rop_claim;
typedef struct bcir_rop_out {
  bcir_rop_resource *res; size_t cap_res, n_res;       /* rid 10 + its index, in the order first named */
  bcir_rop_claim *claims; size_t cap_claims, n_claims;
  uint32_t *rids; size_t cap_rids, n_rids;
  /* a refusal: the offset the oracle names, the token there (kind, text[tok_at, tok_at+tok_n)), and what was
   * expected -- a token kind (`want_kind` >= 0) or the keyword `claim` (`want_kind` -1) */
  size_t pos, tok_at, tok_n; int tok_kind, want_kind;
} bcir_rop_out;
/* Parse `text` (`n` octets): `program := claim+`, each `claim IDENT { stmt* }`, a statement `reads idlist`,
 * `writes idlist`, `count INT`, `lane IDENT` or `stride IDENT`, `;`-separated (optional before `}`). A statement
 * once per claim; a count within the signed 64-bit wire domain; a lane or stride named as the model names one, in
 * any case. */
bcir_etl_status bcir_rop_parse(const char *text, size_t n, bcir_rop_out *out);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_ETL_H */
