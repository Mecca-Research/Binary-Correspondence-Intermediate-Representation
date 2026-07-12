/*===- bcir_cfront.h - the BCIR plug-in C frontend (C twin of bcir/frontends/cfront) ===
 *
 * The production C implementation of the C frontend: it ingests driver/kernel C
 * (fixed-width integer expressions, struct/union layout, bitfields, volatile/MMIO
 * register access -- the register-map subset) and lowers it to the BCIR claim
 * graph (bcir_cir.h), the *same* IR the oracle reasons over. It runs a verifier
 * (R1-R8 subset) and emits verified C, so a driver build embeds a real BCIR C
 * compiler with no Python.
 *
 * This is the dual-rail port of the Python prototype in bcir/frontends/cfront/:
 * once a stage is validated in the oracle, the real implementation lives here and
 * a Python<->C parity test (bcir/tests/test_c_cfront.py) gates the two rails.
 *
 * Host compiler tool: uses libc. The IR + emitted C it produces are freestanding.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_CFRONT_H
#define BCIR_CFRONT_H

#include "bcir_cir.h"
#include "bcir_host_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bcir_cfront_result {
  bcir_unit unit;          /* the lowered translation unit (functions + call graph) */
  int ok;                  /* R1-R8 + R18 verifier clean */
  int emitted_ok;          /* emitted[] is complete; false means the graph is valid but its optional
                            * verified-C text exceeded the fixed result capacity (never partial) */
  char diag[256];          /* first diagnostic (empty when ok) */
  char emitted[32768];     /* faithful emitted C for every bcir_<fn> (the C.2 output seam) */
  bcir_host_allocator _allocator; /* private owner used by idempotent free */
  uint32_t _owner_tag;
} bcir_cfront_result;

/* Re-entrant hosted compiler context. The context owns parser caches and a
 * per-operation scratch arena; returned results own their IR separately. */
typedef struct bcir_cfront_context {
  bcir_host_allocator allocator;
  void *state;
} bcir_cfront_context;

int bcir_cfront_context_init(bcir_cfront_context *context,
                             const bcir_host_allocator *allocator);
void bcir_cfront_context_reset(bcir_cfront_context *context);
void bcir_cfront_context_destroy(bcir_cfront_context *context);

/* Initialize an output to the only valid empty state. Safe before every first use. */
void bcir_cfront_result_init(bcir_cfront_result *out);

/* Context-based, re-entrant entries. Inputs are borrowed; `out` must first be
 * initialized with bcir_cfront_result_init (or be a prior result from this API).
 * A prior owned result is released before reuse. On success `out` owns the unit
 * until bcir_cfront_free; on failure it is empty except diag. */
int bcir_cfront_compile_context(bcir_cfront_context *context, const char *src,
                                bcir_cfront_result *out);
int bcir_cfront_compile_target_context(bcir_cfront_context *context,
                                       const char *src, const char *target,
                                       bcir_cfront_result *out);

/* NON-THREAD-SAFE compatibility wrapper over one process-static context. It
 * initializes `out` for legacy callers; release each returned result before
 * passing the same object to another compatibility call.
 * Compile one C translation unit (the L1-L5 + L3/L4 subset) into the claim graph,
 * verify it (R1-R8 + R18 call-graph), and emit faithful C when it fits. Returns 0 on
 * graph success, nonzero on a parse/lowering error (diag set). `ok` reflects the verifier;
 * `emitted_ok` must be checked before consuming `emitted` (a too-large artifact is empty). */
int bcir_cfront_compile(const char *src, bcir_cfront_result *out);

/* NON-THREAD-SAFE compatibility wrapper. As above, but lay the unit out for `target`'s data model (the C twin of frontends/cfront/abi.py:
 * one of x86_64-linux, aarch64-linux, riscv64-linux, x86_64-windows, i386-linux). `target` NULL
 * selects the host (x86_64-linux LP64); an unknown name returns nonzero with diag set. `long`, the
 * pointer, and the `size_t`-class types follow the selected model; everything else is fixed by C. */
int bcir_cfront_compile_target(const char *src, const char *target, bcir_cfront_result *out);

/* Release every owned allocation and restore the valid empty state. Idempotent. */
void bcir_cfront_free(bcir_cfront_result *out);

/* A canonical, RID-independent structural summary of the entry function's claim graph --
 * the Python<->C dual-rail parity key (bcir/tests/test_c_cfront.py computes the same from
 * the oracle). Writes "funcs=N claims=N mmio=N bf=N const=N binop=N call=N repro=N ok=1
 * digest=<16-hex>" (repro = the count of C23 [[reproducible]]/[[unsequenced]]-hinted functions in
 * the unit; digest = the cross-rail per-claim structural digest, see bcir_cfront_digest). */
void bcir_cfront_summary(const bcir_unit *u, int ok, char *buf, size_t n);

/* The cross-rail PER-CLAIM STRUCTURAL DIGEST (the count->structural parity fix): an FNV-1a (64-bit)
 * hash of a canonical, language-independent serialization of every function's claim DATAFLOW -- the
 * sorted multiset of each claim's value-number record, plus a per-function OBSERVABLE-OUTPUT anchor.
 * Catches count-PRESERVING corruptions the 9-count summary misses: operand swaps incl. NON-COMMUTATIVE
 * operand REVERSAL, op substitutions, call redirects, c.const tampers, c.cast width changes, WRONG
 * struct member / bitfield (member offset + bit off/width/sign folded from the imm), and SINK-write
 * redirects (return-temp / store target, via the anchor). The Python oracle (bcir.verify.cfront_
 * structural_digest) builds the same records + hash, so the two rails produce a BYTE-IDENTICAL digest.
 * Per-claim record: "<op-base>|<opcode-int>|<read value-numbers>|<semantic imm>|<dom-int>";
 * anchor: "ret=<return-value VN>|stores=<dest-VN->value-VN;...>". */
uint64_t bcir_cfront_digest(const bcir_unit *u);

/* Allocator-injected canonical analysis forms. The allocator is borrowed for
 * the operation and every temporary is released before return. */
uint64_t bcir_cfront_digest_with_allocator(const bcir_unit *u,
                                           const bcir_host_allocator *allocator);

/* The raw canonical serialization the digest hashes (text, NOT hashed) -- the byte-identity proof:
 * the Python cfront_structural_canon must equal this byte-for-byte on the corpus, so the digests
 * match. Writes the per-function sorted records, '@'-separated. */
void bcir_cfront_canon(const bcir_unit *u, char *buf, size_t n);
void bcir_cfront_canon_with_allocator(const bcir_unit *u, char *buf, size_t n,
                                      const bcir_host_allocator *allocator);

/* The module-scope effect / commutation analysis (the C twin of pipeline.own_footprint + commute):
 * for each function a `fn=<name> reads=<globals|-> writes=<globals|->` line (its alias/effect
 * footprint over file-scope globals, callee effects folded in transitively, names sorted), then a
 * `commute <a> <b> = 0|1` line per function pair (1 iff their footprints don't conflict -- two
 * readers commute, a writer conflicts with any reader/writer of the same global). */
void bcir_cfront_effects(const bcir_unit *u, char *buf, size_t n);
void bcir_cfront_effects_with_allocator(const bcir_unit *u, char *buf, size_t n,
                                        const bcir_host_allocator *allocator);

/* B1: derive the linker flags a unit's external-call edges need (e.g. `c.call.libm:sqrt` -> `-lm`),
 * written as one space-separated, deduped, STABLY-SORTED line to `buf` (empty for a pure-integer
 * unit). Deterministic/reproducible. The byte-identical C twin of bcir/frontends/cfront/linkflags.py
 * (derive_link_flags + format_link_flags); the parity is gated in bcir/tests/test_c_cfront.py +
 * tools/c/check_runtime.sh. The callee->library mapping (bcir_lib_for_callee, the extension point for
 * roadmap B2's FFTW/LAPACK/GSL/SLEEF wraps) lives in bcir_cfront.c. */
void bcir_cfront_link_flags(const bcir_unit *u, char *buf, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_CFRONT_H */
