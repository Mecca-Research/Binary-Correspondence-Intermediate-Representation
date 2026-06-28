#ifndef BCIR_QUARANTINE_H
#define BCIR_QUARANTINE_H
/* Bounds-quarantine runtime ABI (§5.12). The C frontend promotes a known-extent local/static array access
 * to `masked` (runtime-bounds-checked); the emitted C guards the index and, on an out-of-bounds value,
 * calls `bcir_bounds_quarantine(rid, index, extent, site)`. The WEAK default here records the access
 * provenance into a fixed ring (the debugger reads it via `bcir_quarantine_report`) and aborts -- fail-fast.
 * The ML-layer / debugger OVERRIDES the weak symbol with a STRONG definition to consult the graded-truth
 * quarantine and recover / log / apply policy (a follow-on; the only sanctioned two-truth crossing is a
 * recorded `decide`). In-bounds accesses never call it, so a promoted access is behaviour-identical to the
 * raw `a[i]` for any defined input. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* The OOB / decide ring totals (bcir_oob_count, bcir_decide_count) are incremented from the bounds-handler,
 * which a multithreaded program may enter from several threads at once. A plain `count++` is a read-modify-
 * write that races: two concurrent OOB events can land in the SAME ring slot and lose an audit record, and
 * the running total can scramble. We make the counters ATOMIC so each event claims a distinct slot and the
 * total is exact under concurrency. Where C11 atomics are available (the hosted runtime + the recovery
 * override are compiled hosted) we use `_Atomic unsigned long` + atomic_fetch_add_explicit. A freestanding
 * build without <stdatomic.h> falls back to a plain `volatile unsigned long` under a DOCUMENTED single-
 * threaded contract -- the freestanding emitted unit is single-threaded by construction (no thread spawn in
 * the emitted C), so the non-atomic path is correct there; the hosted ABI (where overrides + real threads
 * live) always gets the atomic form. BCIR_OOB_COUNTER_ATOMIC reflects which path is active. */
#if !defined(__STDC_NO_ATOMICS__) && \
    (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) && __has_include(<stdatomic.h>)
  #include <stdatomic.h>
  #define BCIR_OOB_COUNTER_ATOMIC 1
  typedef _Atomic unsigned long bcir_oob_counter;
  /* return the PRE-increment value (the slot this event owns) and advance the total by one, atomically. */
  static inline unsigned long bcir_counter_next(bcir_oob_counter *c) {
      return atomic_fetch_add_explicit(c, 1ul, memory_order_relaxed);
  }
  static inline unsigned long bcir_counter_load(const bcir_oob_counter *c) {
      return atomic_load_explicit((bcir_oob_counter *)c, memory_order_relaxed);
  }
#else
  #define BCIR_OOB_COUNTER_ATOMIC 0
  typedef volatile unsigned long bcir_oob_counter;   /* single-threaded contract (see note above) */
  static inline unsigned long bcir_counter_next(bcir_oob_counter *c) { return (*c)++; }
  static inline unsigned long bcir_counter_load(const bcir_oob_counter *c) { return *c; }
#endif

/* The out-of-bounds handler for a READ (load index site). `rid` is the numeric resource provenance (stable
 * per translation unit) and `site` is a static "<function>:<array>" source-site handle the debugger /
 * ML-layer resolves -- the site->source table realized inline as a string literal the frontend emits at the
 * guard. RETURNS the index the access should use: the WEAK default records + aborts (it never returns); a
 * STRONG override may RECOVER a read by returning an in-bounds index -- but only through a recorded `decide`
 * (see bcir_decision). Clamping a READ is sound: it redirects the LOAD to a valid element (a defined value,
 * no state mutated). */
size_t bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent, const char *site);

/* The out-of-bounds handler for a WRITE (store index site). An OOB STORE must NEVER be clamped: clamping
 * would silently redirect the store to a VALID element (a[extent-1]), corrupting live data the program owns
 * -- an isolation violation, not a recovery. So this handler does NOT return an index for the store to land
 * on; a policy may consult the graded-truth quarantine and RECORD a `decide`, but the only legal classical
 * outcomes are ABORT (fail-fast) or REJECT -- never a silent redirect. It is declared `noreturn`: an OOB
 * write always terminates the access. The WEAK default records the provenance + aborts; a STRONG override
 * may record a richer `decide` first, but it too must not return. (Contrast bcir_bounds_quarantine, which
 * MAY return a clamped index because a read clamp mutates nothing.) */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn))
#endif
void bcir_bounds_quarantine_write(uint64_t rid, uint64_t index, uint64_t extent, const char *site);

/* The checked index used by the emitted C for a READ (`x = a[i]`): `i` if in-bounds, else run the read
 * handler and use the index it RETURNS (the weak default aborts and never returns; a recovery override
 * returns a clamped in-bounds index). A single evaluation of `i` is assumed (the frontend feeds a temp). */
#define BCIR_CHK(rid, i, n, site) \
    ((uint64_t)(i) < (uint64_t)(n) ? (size_t)(i) \
     : bcir_bounds_quarantine((uint64_t)(rid), (uint64_t)(i), (uint64_t)(n), (site)))

/* The checked index used by the emitted C for a WRITE (`a[i] = v`): `i` if in-bounds, else run the WRITE
 * handler -- which is `noreturn` (abort / reject), so the store NEVER lands on a redirected element. Because
 * the handler cannot return, the false branch's value is unreachable; we still spell `(size_t)(i)` there so
 * the conditional has a single common type (the comma keeps the call's void result out of the value
 * position). The expansion stays an rvalue usable as `a[BCIR_CHK_W(...)]`. A single evaluation of `i` is
 * assumed (the frontend feeds a temp). */
#define BCIR_CHK_W(rid, i, n, site) \
    ((uint64_t)(i) < (uint64_t)(n) ? (size_t)(i) \
     : (bcir_bounds_quarantine_write((uint64_t)(rid), (uint64_t)(i), (uint64_t)(n), (site)), (size_t)(i)))

/* --- rid -> extent/domain correspondence at the guard boundary: a DOCUMENTED freestanding limit ---------
 * The guard TRUSTS the inline `n` (extent) and treats `rid` as opaque provenance: it never resolves `rid`
 * to a declared extent/domain to cross-check `n`. So a CORRUPTED emitted-C with a wrong `n` (a widened bound
 * past the real storage), or an `rid` whose resource is actually MMIO/device memory dressed up as a RAM
 * array, would pass the guard. The MLIR `-bcir-verify` rail closes the *design-time* domain gap (R3 now
 * rejects an MMIO/NVM resource reached as another domain; R10 rejects cross-lane write aliases), but the
 * FREESTANDING emitted C has NO registry to consult at the access -- there is no rid->{extent,domain} table
 * in the translation unit (the analogue of the StreamPack runtime's R10 provenance is absent from a bare
 * --emit-c unit). We do NOT fake a registry that is not there.
 *
 * What IS feasible and tamper-evident WITHOUT a registry, for a KNOWN-EXTENT local/static array: the inline
 * `n` is emitted from the SAME source as the array's `[N]` declaration, so the two can be tied by a
 * COMPILE-TIME assertion `_Static_assert(n == sizeof(a)/sizeof(a[0]))`. A tampered `n` (or a decl/extent
 * mismatch) then fails to COMPILE -- the extent becomes tamper-evident at the rid->storage boundary with no
 * runtime cost and no registry. BCIR_EXTENT_ASSERT realizes this; the emitter MAY place it once per
 * known-extent guarded array (a future emit wiring, gated by the dual-rail byte-parity). It does NOT cover:
 *   - RECOVERED-EXTENT pointers (the ptr_extent path): `n` is a runtime variable with no `sizeof` relation,
 *     so the assert does not apply -- a corrupt recovered count is a genuine freestanding gap.
 *   - DOMAIN (the MMIO-as-RAM `rid`): the freestanding unit has no domain tag per rid to assert against;
 *     this is enforced only on the MLIR rail (R3) and the StreamPack runtime (R10), NOT at this guard.
 *
 * Reconciliation of RT4's dropped c.load bound `ub`: that `ub` is the LOAD UNIT SIZE (unit_bytes), a
 * cost-model/lowering hint on `c.load.imm[1]`, NOT a bounds upper-bound -- it is rail-divergent metadata
 * deliberately DROPPED from the canonical digest (see bcir_cfront.c vn_imm, "--canon byte-identity"), so it
 * is neither enforced nor tamper-evident, and was never a bounds guarantee. The actual bounds contract is
 * the `masked` promotion + this BCIR_CHK/BCIR_CHK_W guard; `ub` plays no bounds role. */
#define BCIR_EXTENT_ASSERT(arr, n) \
    _Static_assert((n) == sizeof(arr) / sizeof((arr)[0]), \
                   "BCIR §5.12: guard extent disagrees with array storage (tampered n or decl)")

/* A record of one out-of-bounds access, for the debugger / ML-layer to inspect. `site` points at the
 * emitted "<function>:<array>" string literal (static storage; valid for the program lifetime). */
typedef struct { uint64_t rid, index, extent; const char *site; } bcir_oob_record;
#define BCIR_OOB_RING 64

/* The debugger trace surface: the fixed ring of the most recent OOB events and the running total, plus a
 * reader that prints them (oldest-retained first) to `f`. The handler may be overridden, but these stay the
 * observability layer -- reading the ring never crosses the two-truth line into a legality verdict. */
extern bcir_oob_record bcir_oob_ring[BCIR_OOB_RING];
extern bcir_oob_counter bcir_oob_count;             /* atomic when available; see bcir_oob_counter above */
void bcir_quarantine_report(FILE *f);

/* Append one event to the ring (the weak default uses it; an override may reuse it before applying policy,
 * so the trace stays populated regardless of which handler is linked). */
void bcir_oob_record_event(uint64_t rid, uint64_t index, uint64_t extent, const char *site);

/* --- the two-truth crossing: a RECORDED `decide` (§5.12 + LANGREF §14) -------------------------------
 * The weak default aborts; a recovery override turns an OOB access into a CLASSICAL action (abort / clamp).
 * That action is a graded->classical crossing -- the ML-layer's recovery confidence (graded truth) must
 * never silently become the access. The ONLY sanctioned crossing is a `decide`: collapse a graded
 * proposition `(action, confidence)` at a FROZEN threshold into a classical value, and RECORD it. This ring
 * is the C twin of `kbcir.twotruth.Decision` -- the auditable trail R13 (`verify.verify_quarantine`) speaks
 * of, so the crossing is never silent. Reading it never decides legality; it is pure observation. */
typedef enum { BCIR_RECOVER_ABORT = 0, BCIR_RECOVER_CLAMP = 1 } bcir_recover_action;

typedef struct {
    uint64_t rid, index, extent;     /* the OOB access this decision is about */
    const char *site;                /* the "<func>:<array>" handle */
    uint32_t confidence_milli;       /* the graded proposition's confidence, Q-milli [0,1000] */
    uint32_t threshold_milli;        /* the FROZEN decide threshold, Q-milli [0,1000] */
    int admitted;                    /* confidence >= threshold: the graded truth cleared the threshold */
    int action;                      /* the classical bcir_recover_action applied (abort / clamp) */
    size_t recovered_index;          /* the in-bounds index the access used (0 if it aborted) */
} bcir_decision;
#define BCIR_DECIDE_RING 64

/* The decide-audit surface: the ring of the most recent recovery crossings + the running total, and a
 * reader (oldest-retained first). An override appends via bcir_decide_record_event; the weak default never
 * crosses (it aborts unconditionally, with no graded input), so it records nothing here. */
extern bcir_decision bcir_decide_ring[BCIR_DECIDE_RING];
extern bcir_oob_counter bcir_decide_count;          /* atomic when available; see bcir_oob_counter above */
void bcir_decide_report(FILE *f);
void bcir_decide_record_event(uint64_t rid, uint64_t index, uint64_t extent, const char *site,
                              uint32_t confidence_milli, uint32_t threshold_milli,
                              int admitted, int action, size_t recovered_index);

#endif /* BCIR_QUARANTINE_H */
