/*===- bcir_encode.h - freestanding StreamPack encoder (write side) -------===
 *
 * The C write-side twin of bcir/abi/streampack_abi.py::encode -- so a driver-resident
 * hydrate can emit the StreamPack artifact with **no Python and no libc**, completing the
 * full C round-trip (the decoder is bcir_runtime.c, the executor bcir_exec.c).
 *
 * `bcir_sp_reencode` parses a StreamPack field-by-field and re-serializes it through the
 * encoder's value-based write primitives into a caller buffer, recomputing the CRC. By
 * construction the output is **byte-identical** to the (frozen v1 / append-only v2)
 * encoding, so a round-trip against a Python-encoded pack proves the writer primitives +
 * the record layout match the ABI exactly (the parity gate is
 * bcir/tests/test_c_encoder.py). The same write primitives a driver uses to emit a pack
 * it built in memory; the round-trip is the testable, self-contained form.
 *
 * Bounds-checked end to end: a malformed input returns a status (never an OOB read), and
 * the writer never exceeds the caller buffer (BCIR_ERR_NOSPACE). Fuzzed under libFuzzer +
 * ASan/UBSan (runtime/c/fuzz_encode.c).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_ENCODE_H
#define BCIR_ENCODE_H

#include "bcir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Re-serialize the StreamPack `in[0..in_len)` into `out[0..out_cap)`, writing the encoded
 * length to *out_len. The output equals the frozen-ABI encoding of the same pack (byte-
 * identical to bcir.abi.encode). Returns BCIR_ERR_* on a malformed input and
 * BCIR_ERR_NOSPACE if `out_cap` is too small (out_cap >= in_len always suffices). */
BCIR_NODISCARD bcir_status bcir_sp_reencode(const uint8_t *BCIR_RESTRICT in, size_t in_len,
                                            uint8_t *BCIR_RESTRICT out, size_t out_cap,
                                            size_t *BCIR_RESTRICT out_len);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_ENCODE_H */
