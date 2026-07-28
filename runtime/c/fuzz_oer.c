/*===- fuzz_oer.c - libFuzzer entry for the X.696 OER decoder ---------------===
 *
 * The totality contract: for ANY octets and ANY plan, every bcir_oer_* entry point returns
 * a status and never reads outside the buffer nor writes outside the caller's output.
 *
 * THE PLAN IS FUZZED TOO, and that is the point of this target. Every other decoder in this
 * repository is driven by the input alone; OER is driven by the input AND a caller-supplied
 * field table, because X.696 6.2 makes a schema-free walk impossible. That doubles the
 * attack surface: a plan whose declared widths and lengths disagree with the octets is
 * exactly the shape that walks a cursor out of bounds, and it is reachable whenever a
 * descriptor and a document come from different places -- which, for a driver reading a
 * manifest, is always.
 *
 * So the field table is derived FROM the input rather than fixed, including its length
 * fields, so the fuzzer reaches plans a well-formed compiler would never emit.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_oer.h"

#define MAX_FIELDS 16

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t buffer[1 << 14];
  static bcir_oer_field fields[MAX_FIELDS];
  static bcir_oer_value values[MAX_FIELDS];
  bcir_oer_diag diag;
  size_t len = size > sizeof(buffer) ? sizeof(buffer) : size;
  size_t at;
  size_t count;
  size_t end = 0;
  int canonical = 0;

  for (at = 0; at < len; at++) buffer[at] = data[at];

  /* The primitives, at every offset including the at-end and past-end edges. */
  for (at = 0; at <= len + 1; at++) {
    uint64_t value = 0;
    int64_t signed_value = 0;
    size_t stop = 0;
    (void)bcir_oer_length(buffer, len, at, &value, &stop, &canonical, &diag);
    (void)bcir_oer_preamble(buffer, len, at, len ? (buffer[0] % 70u) : 0u, &value, &stop,
                            &canonical, &diag);
    (void)bcir_oer_integer(buffer, len, at, 0, 1, &signed_value, &stop, &diag);
    (void)bcir_oer_integer(buffer, len, at, len ? (1u << (buffer[len - 1] % 4u)) : 1u,
                           len ? (buffer[0] & 1) : 0, &signed_value, &stop, &diag);
    /* A width the plan checker must refuse rather than read. */
    (void)bcir_oer_integer(buffer, len, at, 3, 0, &signed_value, &stop, &diag);
    if (at > 64) break;                        /* the interior is covered by the walk */
  }

  if (len == 0) {
    (void)bcir_oer_decode_sequence(buffer, 0, 0, fields, 0, values, &end, &canonical,
                                   &diag);
    return 0;
  }

  /* A plan derived from the input: kinds, widths, optionality and fixed lengths all come
   * from octets an attacker chose, so a mismatch between plan and document is reachable. */
  count = (size_t)(buffer[0] % (MAX_FIELDS + 1));
  for (at = 0; at < count; at++) {
    uint8_t seed = buffer[(at + 1) % len];
    fields[at].kind = (bcir_oer_kind)(seed % 6u);   /* 5 is out of range: must be refused */
    fields[at].width = (uint8_t)(1u << (seed % 4u));
    fields[at].is_signed = (uint8_t)(seed & 1u);
    fields[at].optional = (uint8_t)((seed >> 1) & 1u);
    fields[at].fixed_len = (uint32_t)buffer[(at + 2) % len] * 4u;
    if ((seed & 0x30u) == 0x30u) fields[at].width = 0;      /* the length-prefixed form */
  }
  (void)bcir_oer_decode_sequence(buffer, len, 0, fields, count, values, &end, &canonical,
                                 &diag);
  /* And starting inside the buffer, so a plan meets a document that does not begin here. */
  (void)bcir_oer_decode_sequence(buffer, len, (size_t)buffer[0] % (len + 1), fields, count,
                                 values, &end, &canonical, &diag);
  /* A count larger than the caller's output array must be a refusal, not a write past it. */
  (void)bcir_oer_decode_sequence(buffer, len, 0, fields, MAX_FIELDS, values, &end,
                                 &canonical, &diag);
  return 0;
}
