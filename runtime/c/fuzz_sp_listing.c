/*===- fuzz_sp_listing.c - libFuzzer harness for the MC1 walk and listing ----===
 *
 * bcir_sp_for_each_record and the listing/dump over it (bcir_sp_listing.c) read an untrusted
 * StreamPack. Each input is driven raw and RESEALED (its CRC recomputed, so a mutation reaches the
 * record laws instead of dying on the checksum), and the harness aborts unless:
 *   - the walk's spans partition an accepted pack exactly, in wire order, and every view it hands over
 *     lies inside the pack (touched, so ASan bounds it);
 *   - the listing and the dump return the walk's own verdict, and a refused pack reaches no sink;
 *   - a pack the semantic verifier accepts the walk accepts (R10 adds laws, never removes one);
 *   - a sink that refuses its second chunk stops the format call with BCIR_ERR_NOSPACE.
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 \
 *       fuzz_sp_listing.c bcir_sp_listing.c bcir_runtime.c -I . -o fuzz_sp_listing
 *   ./fuzz_sp_listing -runs=1000000           (see tools/c/fuzz_streampack.sh)
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_sp_listing.h"

typedef struct { size_t next, records; uint64_t acc; int broken; } spans_t;

static void touch(spans_t *s, const char *p, uint16_t n) {
  if (n) s->acc += (uint8_t)p[0] + (uint8_t)p[n - 1];
}

static int spans(const bcir_sp_record *r, void *ctx) {
  spans_t *s = (spans_t *)ctx;
  if (r->offset != s->next || r->length == 0) s->broken = 1;
  s->next = r->offset + r->length;
  s->records++;
  const char *t;
  uint16_t n;
  bcir_sp_strings it;
  switch (r->kind) {
  case BCIR_SP_RECORD_SOURCE_PLAN: touch(s, r->source_plan, r->source_plan_len); break;
  case BCIR_SP_RECORD_SEGMENT:
    touch(s, r->segment.name, r->segment.name_len);
    touch(s, r->segment.opcode, r->segment.opcode_len);
    touch(s, r->segment.prefetch, r->segment.prefetch_len);
    touch(s, r->segment.channel, r->segment.channel_len);
    for (uint16_t i = 0; i < r->segment.n_reads; i++) s->acc += bcir_seg_read_rid(&r->segment, i);
    for (uint16_t i = 0; i < r->segment.n_writes; i++) s->acc += bcir_seg_write_rid(&r->segment, i);
    it = r->fence_before;
    while (bcir_sp_strings_next(&it, &t, &n)) touch(s, t, n);
    it = r->fence_after;
    while (bcir_sp_strings_next(&it, &t, &n)) touch(s, t, n);
    break;
  case BCIR_SP_RECORD_PREFETCH:
    touch(s, r->prefetch.name, r->prefetch.name_len);
    touch(s, r->prefetch.hint, r->prefetch.hint_len);
    touch(s, r->prefetch.pattern, r->prefetch.pattern_len);
    for (uint16_t i = 0; i < r->prefetch.n_targets; i++) s->acc += bcir_prefetch_target(&r->prefetch, i);
    break;
  case BCIR_SP_RECORD_BLOCK:
    for (uint16_t i = 0; i < r->block.n_strides; i++) s->acc += bcir_block_stride(&r->block, i);
    break;
  default: break;
  }
  return 0;
}

typedef struct { size_t calls, refuse_after; } sink_t;

static int sink(void *ctx, const char *text, size_t len) {
  sink_t *s = (sink_t *)ctx;
  s->calls++;
  if (len == 0 || len > BCIR_LISTING_CHUNK) abort();
  (void)text[len - 1];
  return s->refuse_after && s->calls > s->refuse_after;
}

static void drive(const uint8_t *data, size_t size, uint32_t width) {
  spans_t s = {0, 0, 0, 0};
  bcir_status walked = bcir_sp_for_each_record(data, size, spans, &s);
  if (walked == BCIR_OK && (s.broken || s.next != size || s.records < 3u)) abort();
  if (walked != BCIR_OK && s.records) abort();  /* a refused pack shows no record */
  if (bcir_sp_verify_semantic(data, size, UINT32_MAX, UINT32_MAX) == BCIR_OK && walked != BCIR_OK)
    abort();

  size_t need = 0;
  if (bcir_sp_listing_scratch_bytes(data, size, &need) != walked) abort();
  if (walked != BCIR_OK && need) abort();
  uint8_t *scratch = (uint8_t *)malloc(need ? need : 1u);
  if (!scratch) abort();
  sink_t all = {0, 0};
  if (bcir_sp_format_listing(data, size, scratch, need, sink, &all) != walked) abort();
  if (walked != BCIR_OK && all.calls) abort();
  sink_t dump = {0, 0};
  if (bcir_sp_format_hexdump(data, size, width, sink, &dump) != walked) abort();
  if (walked != BCIR_OK && dump.calls) abort();
  if (walked == BCIR_OK) {
    sink_t refusing = {0, 1};
    bcir_status st = bcir_sp_format_listing(data, size, scratch, need, sink, &refusing);
    if (st != (all.calls > 1u ? BCIR_ERR_NOSPACE : BCIR_OK)) abort();
    refusing.calls = 0;
    st = bcir_sp_format_hexdump(data, size, width, sink, &refusing);
    if (st != (dump.calls > 1u ? BCIR_ERR_NOSPACE : BCIR_OK)) abort();
  }
  free(scratch);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint32_t width = size ? 1u + (uint32_t)(data[size - 1] % BCIR_HEXDUMP_WIDTH_MAX) : 16u;
  drive(data, size, width);
  if (size >= (size_t)BCIR_STREAMPACK_HEADER_SIZE + 4u && size <= (1u << 20)) {
    uint8_t *sealed = (uint8_t *)malloc(size);
    if (!sealed) abort();
    memcpy(sealed, data, size);
    uint32_t crc = bcir_crc32(sealed, size - 4u);
    for (int i = 0; i < 4; i++) sealed[size - 4u + (size_t)i] = (uint8_t)(crc >> (8 * i));
    drive(sealed, size, width);
    free(sealed);
  }
  return 0;
}
