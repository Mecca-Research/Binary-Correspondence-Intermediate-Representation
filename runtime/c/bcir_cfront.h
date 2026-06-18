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
  bcir_func func;          /* the lowered function (the entry) */
  int ok;                  /* R1-R8 verifier clean */
  char diag[256];          /* first diagnostic (empty when ok) */
  char emitted[16384];     /* emitted verified C (the C.2 output seam) */
} bcir_cfront_result;

/* Compile one C translation unit (the register-map subset) into the claim graph,
 * verify it, and emit verified C. Returns 0 on success (parsed + verified), nonzero
 * on a parse/lowering error (diag set). `ok` reflects the R1-R8 verifier. */
int bcir_cfront_compile(const char *src, bcir_cfront_result *out);

/* Release the heap arrays the result holds. */
void bcir_cfront_free(bcir_cfront_result *out);

/* A canonical, RID-independent structural summary of the lowered claim graph -- the
 * Python<->C dual-rail parity key (bcir/tests/test_c_cfront.py computes the same from
 * the oracle's lowering). Writes e.g. "claims=23 mmio=1 bf=3 const=5 binop=7 ok=1". */
void bcir_cfront_summary(const bcir_func *f, int ok, char *buf, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_CFRONT_H */
