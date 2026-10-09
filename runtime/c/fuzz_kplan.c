/*===- fuzz_kplan.c - libFuzzer: the native K_BCIR planner (G17, S4-A) ----------===
 *
 * The first input byte picks a mode:
 *   0  the BKPI decoder over the raw bytes -- total on arbitrary input, never an out-of-bounds
 *      read -- and, when it accepts, the planner;
 *   1  the BKPR decoder over the raw bytes;
 *   2  the rest is a BKPI record WITHOUT its trailer: the harness appends the CRC the record
 *      needs (the BCIRQ8 checksum repair), so a mutation reaches past the seal to every field
 *      law and into the planner;
 *   3  the BKPB (binding) decoder over the raw bytes (CXX4).
 *
 * Every plan the planner returns must hold, or the harness aborts (a finding):
 *   - the realization decodes (bcir_kp_decode_realization), with one step per claim and a
 *     score that is the sum of the step costs;
 *   - planning is a function of the input: a second plan, over dirty scratch at another
 *     alignment, writes the same bytes;
 *   - a refusal writes nothing: the output is zero and `out_len` is 0.
 *
 * And every plan the planner returns must hydrate (CXX4) under a binding consistent with its
 * input (RID i + 1 at index i, a generation for each declared resource): into a StreamPack the
 * runtime's own reader accepts in full, the same bytes over dirty scratch at another alignment
 * -- or be refused only for what a StreamPack cannot carry (a width that is not a power of two,
 * a negative block field), writing nothing.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_kplan.h"

#define MAX_SCRATCH (64u << 20)

static void put32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

/* The plan of `in` hydrated under a binding consistent with it; aborts on a broken property. */
static void hydrate_checked(const bcir_kp_input *in, const uint8_t *plan, size_t plan_len) {
  bcir_kp_realization r;
  if (bcir_kp_decode_realization(plan, plan_len, &r) != BCIR_OK) abort();
  uint32_t n_gens = 0;
  for (uint32_t i = 0; i < in->n_resources; i++)
    if (in->data[in->off_resources + BCIR_KP_RESOURCE_SIZE * (size_t)i]) n_gens++;
  size_t blen = BCIR_KP_BINDING_HEADER_SIZE + 4u * (size_t)in->n_resources +
                12u * (size_t)n_gens + 4u + BCIR_KP_CRC_SIZE;
  uint8_t *bind = (uint8_t *)calloc(blen, 1);
  if (!bind) abort();
  memcpy(bind, "BKPB", 4);
  put32(bind + 8, in->n_resources);
  put32(bind + 12, n_gens);
  put32(bind + 16, 1u);
  put32(bind + 20, 4u);
  size_t at = BCIR_KP_BINDING_HEADER_SIZE;
  for (uint32_t i = 0; i < in->n_resources; i++, at += 4u) put32(bind + at, i + 1u);
  for (uint32_t i = 0; i < in->n_resources; i++) {
    if (!in->data[in->off_resources + BCIR_KP_RESOURCE_SIZE * (size_t)i]) continue;
    put32(bind + at, i + 1u);
    put32(bind + at + 4u, i);
    put32(bind + at + 8u, i ^ 1u);
    at += 12u;
  }
  memcpy(bind + at, "fuzz", 4);
  put32(bind + at + 4u, bcir_crc32(bind, at + 4u));
  bcir_kp_binding b;
  if (bcir_kp_decode_binding(bind, blen, &b) != BCIR_OK) abort();
  size_t scratch_len = 0, cap = 0, out_len = 1;
  if (bcir_kp_hydrate_scratch_size(in, &b, &scratch_len) != BCIR_OK) abort();
  if (scratch_len > MAX_SCRATCH) {
    free(bind);
    return;
  }
  uint8_t *scratch = (uint8_t *)malloc(scratch_len + 8u);
  if (!scratch) abort();
  memset(scratch, 0xA5, scratch_len + 8u);
  bcir_status st = bcir_kp_hydrate_size(in, &r, &b, scratch, scratch_len, &cap);
  if (st != BCIR_OK) {
    if (st != BCIR_ERR_WIDTH && st != BCIR_ERR_OVERFLOW) abort();
    if (cap != 0) abort();
  } else if (cap <= MAX_SCRATCH) {
    uint8_t *out = (uint8_t *)malloc(cap + 8u), *again = (uint8_t *)malloc(cap + 8u);
    if (!out || !again) abort();
    if (bcir_kp_hydrate(in, &r, &b, scratch, scratch_len, out, cap, &out_len) != BCIR_OK) abort();
    bcir_streampack_header hdr;
    if (out_len != cap || bcir_sp_validate(out, out_len, &hdr) != BCIR_OK ||
        bcir_sp_verify_semantic(out, out_len, hdr.map_gen, hdr.data_gen) != BCIR_OK)
      abort();
    size_t again_len = 0;
    memset(scratch, 0x3C, scratch_len + 8u);
    if (bcir_kp_hydrate(in, &r, &b, scratch + 5, scratch_len, again, cap, &again_len) != BCIR_OK ||
        again_len != out_len || memcmp(out, again, out_len) != 0)
      abort();
    free(out);
    free(again);
  }
  free(scratch);
  free(bind);
}

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
    hydrate_checked(&in, out, out_len);
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
  if (mode == 3) {
    bcir_kp_binding b;
    (void)bcir_kp_decode_binding(data, size, &b);
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
