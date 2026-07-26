/*===- fuzz_per.c - libFuzzer entry for the X.691 PER decoding primitives ---===
 *
 * The totality contract: for ANY input bytes and ANY bounds, every bcir_per_* entry point
 * returns a status and never reads outside the buffer. There is no "valid input" notion
 * here -- clause 11's decoders are the layer that takes an attacker-supplied width, count
 * or fragment header and advances a cursor with it, so the property under test is that no
 * such value can walk the cursor out of the buffer.
 *
 * The bounds are derived FROM the input rather than fixed, so the fuzzer explores the
 * range-selection branches of 11.5 (bit-field, one-octet, two-octet, indefinite) instead
 * of hammering one of them.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "bcir_per.h"

static void drive(const uint8_t *data, size_t len, bcir_per_variant variant,
                  int64_t lb, int64_t ub) {
  bcir_per_reader r;
  int64_t signed_out = 0;
  uint64_t unsigned_out = 0;
  int more = 0;

  if (bcir_per_reader_init(&r, data, len, variant) != BCIR_PER_OK) return;
  (void)bcir_per_constrained(&r, lb, ub, &signed_out);

  if (bcir_per_reader_init(&r, data, len, variant) != BCIR_PER_OK) return;
  (void)bcir_per_semi_constrained(&r, lb, &signed_out);

  if (bcir_per_reader_init(&r, data, len, variant) != BCIR_PER_OK) return;
  (void)bcir_per_unconstrained(&r, &signed_out);

  if (bcir_per_reader_init(&r, data, len, variant) != BCIR_PER_OK) return;
  (void)bcir_per_normally_small(&r, &unsigned_out);

  if (bcir_per_reader_init(&r, data, len, variant) != BCIR_PER_OK) return;
  (void)bcir_per_normally_small_length(&r, &unsigned_out);

  if (bcir_per_reader_init(&r, data, len, variant) != BCIR_PER_OK) return;
  (void)bcir_per_length(&r, 0, 0, 0, &unsigned_out, &more);

  if (bcir_per_reader_init(&r, data, len, variant) != BCIR_PER_OK) return;
  (void)bcir_per_length(&r, lb < 0 ? 0 : lb, ub < 0 ? 0 : ub, 1, &unsigned_out, &more);

  /* A length loop: keep reading while the determinant says a fragment follows. The bound
   * is what proves the loop terminates on hostile input rather than spinning. */
  if (bcir_per_reader_init(&r, data, len, variant) == BCIR_PER_OK) {
    int guard = 0;
    do {
      if (bcir_per_length(&r, 0, 0, 0, &unsigned_out, &more) != BCIR_PER_OK) break;
    } while (more && ++guard < 64);
  }

  /* Every bit width, including the 0 and 64 edges. */
  if (bcir_per_reader_init(&r, data, len, variant) == BCIR_PER_OK) {
    unsigned width;
    for (width = 0u; width <= 64u; width++) {
      if (bcir_per_get_bits(&r, width, &unsigned_out) != BCIR_PER_OK) break;
    }
  }

  if (bcir_per_reader_init(&r, data, len, variant) == BCIR_PER_OK) {
    (void)bcir_per_align(&r);
  }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  int64_t lb = 0, ub = 0;
  const uint8_t *body = data;
  size_t body_len = size;

  /* The first sixteen bytes, when present, seed the bounds; the rest is the encoding. */
  if (size >= 16u) {
    uint64_t raw_lb = 0, raw_ub = 0;
    memcpy(&raw_lb, data, 8);
    memcpy(&raw_ub, data + 8, 8);
    lb = (int64_t)raw_lb;
    ub = (int64_t)raw_ub;
    body = data + 16;
    body_len = size - 16u;
  }

  drive(body, body_len, BCIR_PER_UNALIGNED, lb, ub);
  drive(body, body_len, BCIR_PER_ALIGNED, lb, ub);
  /* Bounds in the opposite order exercise the lb > ub refusal rather than skipping it. */
  drive(body, body_len, BCIR_PER_ALIGNED, ub, lb);
  return 0;
}
