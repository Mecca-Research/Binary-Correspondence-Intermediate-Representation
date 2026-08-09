/*===- fuzz_per_plan.c - libFuzzer entry for the plan-driven X.691 PER decoder ---------------===
 *
 * The totality contract: for ANY octets and ANY plan, bcir_per_decode_sequence returns a
 * status, never reads outside [data, data + len), and never writes outside the caller's
 * output array.
 *
 * THE PLAN IS FUZZED TOO, for the reason fuzz_oer.c gives: X.691 6.2 says a PER encoding
 * carries no identifier, so this decoder is driven by a caller-supplied field table as well
 * as by octets, and a table that disagrees with the document is the shape that walks a cursor
 * out of bounds. It is reachable whenever a descriptor and a document come from different
 * places, which for a driver reading a manifest is always.
 *
 * PER ADDS A SECOND AXIS OER DOES NOT HAVE: `aligned` and `extensible` ride on the CALL
 * rather than on a field, and ALIGNED differs from UNALIGNED at every field boundary. So both
 * are derived from the input and every plan is driven through all four combinations -- a
 * decoder correct on one variant can be wrong on the other in a way a single-variant campaign
 * would never show.
 *
 * The bounds are drawn from the octets as full 64-bit values, so 11.5's range-selection
 * branches (bit-field / one-octet / two-octet / indefinite) and 13.2's three integer shapes
 * are all reachable rather than one being hammered.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_per_plan.h"

#define MAX_FIELDS 16

/* Eight octets of the input as a 64-bit quantity, so a bound can be any value including the
 * ones that make 11.5.7.4's indefinite-length form and the RANGE refusals reachable. */
static int64_t pick64(const uint8_t *buffer, size_t len, size_t at) {
  uint64_t value = 0u;
  size_t i;
  for (i = 0; i < 8u; i++) value = (value << 8) | buffer[(at + i) % len];
  return (int64_t)value;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t buffer[1 << 13];
  static bcir_per_field fields[MAX_FIELDS];
  static bcir_per_value values[MAX_FIELDS];
  size_t len = size > sizeof(buffer) ? sizeof(buffer) : size;
  size_t at, count, end = 0;
  int aligned, extensible;

  for (at = 0; at < len; at++) buffer[at] = data[at];

  if (len == 0) {
    /* A zero-length document is a legal call, and a plan against it must refuse rather
     * than read: every field is truncated at bit zero. */
    fields[0].kind = BCIR_PER_K_BOOLEAN;
    fields[0].bounds = BCIR_PER_B_UNCONSTRAINED;
    fields[0].lb = 0; fields[0].ub = 0; fields[0].fixed_len = 0; fields[0].optional = 0;
    (void)bcir_per_decode_sequence(buffer, 0, fields, 1, 0, 0, values, &end);
    (void)bcir_per_decode_sequence(buffer, 0, fields, 0, 1, 1, values, &end);
    return 0;
  }

  count = (size_t)(buffer[0] % (MAX_FIELDS + 1));
  for (at = 0; at < count; at++) {
    uint8_t seed = buffer[(at + 1u) % len];
    /* 5 is past the last kind, so the decoder's own refusal path is reached too. */
    fields[at].kind = (bcir_per_kind)(seed % 6u);
    fields[at].bounds = (bcir_per_bounds)((seed >> 3) % 4u);   /* 3 is out of range */
    fields[at].lb = pick64(buffer, len, (at + 2u) % len);
    fields[at].ub = pick64(buffer, len, (at + 3u) % len);
    /* Both a plausible SIZE and an enormous one: a fixed length far past the document is
     * exactly the plan/document mismatch this target exists to reach. */
    fields[at].fixed_len = (seed & 0x40u)
        ? (uint32_t)buffer[(at + 4u) % len] * 0x01000000u
        : (uint32_t)buffer[(at + 4u) % len];
    fields[at].optional = (uint8_t)((seed >> 2) & 1u);
  }

  /* Both variants and both extensibility settings: `aligned` and `extensible` are properties
   * of the CALL, so they are a dimension of the plan surface rather than of a field. */
  for (aligned = 0; aligned <= 1; aligned++) {
    for (extensible = 0; extensible <= 1; extensible++) {
      (void)bcir_per_decode_sequence(buffer, len, fields, count, aligned, extensible,
                                     values, &end);
      /* A count larger than the plan the caller filled in must still stay inside `values`. */
      (void)bcir_per_decode_sequence(buffer, len, fields, MAX_FIELDS, aligned, extensible,
                                     values, &end);
      /* `end_bit` is optional; a NULL there must not be dereferenced. */
      (void)bcir_per_decode_sequence(buffer, len, fields, count, aligned, extensible,
                                     values, 0);
    }
  }

  /* The argument-validation edges, which no octet can reach on its own. */
  (void)bcir_per_decode_sequence(0, len, fields, count, 0, 0, values, &end);
  (void)bcir_per_decode_sequence(buffer, len, 0, count, 0, 0, values, &end);
  (void)bcir_per_decode_sequence(buffer, len, fields, count, 0, 0, 0, &end);
  return 0;
}
