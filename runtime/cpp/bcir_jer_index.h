/*===- bcir_jer_index.h - the hosted structural index for the bounding pass -===
 *
 * J5's row covers UTF-8 validation. This is the other half 1's pipeline names, the "optional
 * hosted SIMD structural index", and 7.4 records why it is a different problem from the
 * UTF-8 rail rather than more of the same.
 *
 * WHY IT IS NOT THE UTF-8 RAIL'S SHAPE. `bcir_jer_validate_utf8` is a whole-document
 * function with no cost budget: skipping an ASCII run is semantically free, because the
 * answer is a property of the octets alone. `bcir_jer_scan` is neither. It charges one work
 * unit per octet against 4.3's ceiling, and BCIR_JER_WORK_EXCEEDED carries the EXACT octet
 * at which the budget ran out -- so the scan's cost is observable output. A vector pass that
 * skipped a run without charging for it would accept documents the scalar rail rejects.
 *
 * WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT. `bcir_jer_scan`'s loop is a DISPATCH:
 * skip whitespace, recognise a structural octet, or hand off to a token scanner. Only the
 * dispatch is vectorizable -- finding the next octet that is not whitespace, or the end of a
 * plain string body. The token scanners are where the semantics live: 4.3's string_bytes and
 * number_digits limits, escape validity, the exponent ceiling.
 *
 * So this file rebuilds the DISPATCH and reuses the token scanners verbatim, through
 * `bcir_jer_scan_cursor`. It is a second dispatch loop, not a second scanner, and that is
 * the whole design: 4.1 makes the scalar rail authoritative, and 8's table names a second
 * semantics rail as the risk this must not take.
 *
 * THIS VERSION IS SCALAR. It exists first so the equivalence harness can prove the seam is
 * SUFFICIENT -- that a loop built only from the public cursor reproduces `bcir_jer_scan`'s
 * status, offset and node count over the whole corpus -- before any SIMD is introduced. A
 * differential that only starts existing alongside the optimization cannot tell you which of
 * the two broke it.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_JER_INDEX_H
#define BCIR_JER_INDEX_H

#include "bcir_jer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The same contract as `bcir_jer_scan`, argument for argument and diagnostic for diagnostic.
 * Identical is the requirement, not equivalent: 4.2 promises a stable code AND a byte
 * offset, and an offset off by one still sends a caller to the wrong octet. */
bcir_jer_status bcir_jer_index_scan(const uint8_t *data, size_t len,
                                    const bcir_jer_limits *limits,
                                    bcir_jer_level *stack, size_t stack_entries,
                                    uint64_t *nodes, bcir_jer_diag *diag);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_JER_INDEX_H */
