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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bcir_cfront_result {
  bcir_unit unit;          /* the lowered translation unit (functions + call graph) */
  int ok;                  /* R1-R8 + R18 verifier clean */
  char diag[256];          /* first diagnostic (empty when ok) */
  char emitted[32768];     /* faithful emitted C for every bcir_<fn> (the C.2 output seam) */
} bcir_cfront_result;

/* Compile one C translation unit (the L1-L5 + L3/L4 subset) into the claim graph,
 * verify it (R1-R8 + R18 call-graph), and emit faithful C. Returns 0 on success,
 * nonzero on a parse/lowering error (diag set). `ok` reflects the verifier. */
int bcir_cfront_compile(const char *src, bcir_cfront_result *out);

/* As above, but lay the unit out for `target`'s data model (the C twin of frontends/cfront/abi.py:
 * one of x86_64-linux, aarch64-linux, riscv64-linux, x86_64-windows, i386-linux). `target` NULL
 * selects the host (x86_64-linux LP64); an unknown name returns nonzero with diag set. `long`, the
 * pointer, and the `size_t`-class types follow the selected model; everything else is fixed by C. */
int bcir_cfront_compile_target(const char *src, const char *target, bcir_cfront_result *out);

/* Release the heap arrays the result holds. */
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

/* The raw canonical serialization the digest hashes (text, NOT hashed) -- the byte-identity proof:
 * the Python cfront_structural_canon must equal this byte-for-byte on the corpus, so the digests
 * match. Writes the per-function sorted records, '@'-separated. */
void bcir_cfront_canon(const bcir_unit *u, char *buf, size_t n);

/* The module-scope effect / commutation analysis (the C twin of pipeline.own_footprint + commute):
 * for each function a `fn=<name> reads=<globals|-> writes=<globals|->` line (its alias/effect
 * footprint over file-scope globals, callee effects folded in transitively, names sorted), then a
 * `commute <a> <b> = 0|1` line per function pair (1 iff their footprints don't conflict -- two
 * readers commute, a writer conflicts with any reader/writer of the same global). */
void bcir_cfront_effects(const bcir_unit *u, char *buf, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_CFRONT_H */
