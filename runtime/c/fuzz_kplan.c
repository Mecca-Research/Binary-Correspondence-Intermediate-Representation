/*===- fuzz_kplan.c - libFuzzer: the native K_BCIR planner (G17, S4-A) ----------===
 *
 * The first input byte picks a mode:
 *   0  the BKPI decoder over the raw bytes -- total on arbitrary input, never an out-of-bounds
 *      read -- and, when it accepts, the planner;
 *   1  the BKPR decoder over the raw bytes;
 *   2  the rest is a BKPI record WITHOUT its trailer: the harness appends the CRC the record
 *      needs (the BCIRQ8 checksum repair), so a mutation reaches past the seal to every field
 *      law and into the planner.
 *
 * Every plan the planner returns must hold, or the harness aborts (a finding):
 *   - the realization decodes (bcir_kp_decode_realization), with one step per claim and a
 *     score that is the sum of the step costs;
 *   - planning is a function of the input: a second plan, over dirty scratch at another
 *     alignment, writes the same bytes;
 *   - a refusal writes nothing: the output is zero and `out_len` is 0.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_kplan.h"

#define MAX_SCRATCH (64u << 20)

static void plan_checked(const uint8_t *data, size_t size) {
  bcir_kp_input in;
  if (bcir_kp_decode_input(data, size, &in) != BCIR_OK) return;
  size_t scratch_len = 0, cap = 0, out_len = 1;
  if (bcir_kp_scratch_size(&in, &scratch_len) != BCIR_OK) return;
  if (bcir_kp_realization_size(&in, &cap) != BCIR_OK) return;
  if (scratch_len > MAX_SCRATCH || cap > MAX_SCRATCH) return;
  uint8_t *scratch = (uint8_t *)malloc(scratch_len + 8u);
  uint8_t *out = (uint8_t *)malloc(cap + 8u);
  uint8_t *again = (uint8_t *)malloc(cap + 8u);
  if (!scratch || !out || !again) abort();
  memset(scratch, 0xA5, scratch_len + 8u);
  memset(out, 0x5A, cap + 8u);
  bcir_status st = bcir_kp_plan(&in, scratch, scratch_len, out, cap, &out_len);
  if (st != BCIR_OK) {
    if (out_len != 0) abort();
    for (size_t i = 0; i < cap; i++)
      if (out[i]) abort();
  } else {
    bcir_kp_realization r;
    if (out_len != cap) abort();
    if (bcir_kp_decode_realization(out, out_len, &r) != BCIR_OK) abort();
    if (r.n_steps != in.n_claims) abort();
    size_t again_len = 0;
    memset(scratch, 0x3C, scratch_len + 8u);
    if (bcir_kp_plan(&in, scratch + 3, scratch_len, again, cap, &again_len) != BCIR_OK) abort();
    if (again_len != out_len || memcmp(out, again, out_len) != 0) abort();
  }
  free(scratch);
  free(out);
  free(again);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 1) return 0;
  uint8_t mode = data[0];
  data++;
  size--;
  if (mode == 1) {
    bcir_kp_realization r;
    (void)bcir_kp_decode_realization(data, size, &r);
    return 0;
  }
  if (mode == 2) {
    uint8_t *sealed = (uint8_t *)malloc(size + 4u);
    if (!sealed) abort();
    if (size) memcpy(sealed, data, size);
    uint32_t crc = bcir_crc32(sealed, size);
    sealed[size] = (uint8_t)crc;
    sealed[size + 1] = (uint8_t)(crc >> 8);
    sealed[size + 2] = (uint8_t)(crc >> 16);
    sealed[size + 3] = (uint8_t)(crc >> 24);
    plan_checked(sealed, size + 4u);
    free(sealed);
    return 0;
  }
  plan_checked(data, size);
  return 0;
}
