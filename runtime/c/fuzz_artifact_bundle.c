/* libFuzzer harness for the allocation-free BCAB reader, selector and listing.
 *
 * Each input is driven raw and RESEALED (its body CRC, embedded SHA-256 and header CRC recomputed,
 * as bcir/tests/test_artifact_bundle.py's _reseal does), so a mutation reaches the directory and
 * payload laws instead of dying on a checksum. The harness aborts unless the listing and the dump
 * return the reader's own verdict, a refused bundle reaches no sink, a sink that refuses its second
 * chunk stops the call with BCIR_AB_ERR_OUTPUT, and every text and StreamPack payload of an accepted
 * bundle has its views (PORT-1). */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_ab_listing.h"
#include "bcir_runtime.h"
#include "bcir_sha256.h"

typedef struct { size_t calls, refuse_after; } sink_t;

static int sink(void *ctx, const char *text, size_t len) {
  sink_t *s = (sink_t *)ctx;
  s->calls++;
  if (len == 0 || len > BCIR_LISTING_CHUNK) abort();
  (void)text[len - 1];
  return s->refuse_after && s->calls > s->refuse_after;
}

static void drive(const uint8_t *data, size_t size, uint32_t width) {
  bcir_ab_view view;
  bcir_ab_status opened = bcir_ab_open(data, size, &view);
  sink_t all = {0, 0}, dump = {0, 0};
  if (bcir_ab_format_listing(data, size, sink, &all) != opened) abort();
  if (bcir_ab_format_hexdump(data, size, width, sink, &dump) != opened) abort();
  if (opened != BCIR_AB_OK) {
    if (all.calls || dump.calls) abort();  /* a refused bundle prints nothing */
    return;
  }
  sink_t refusing = {0, 1};
  if (bcir_ab_format_listing(data, size, sink, &refusing) !=
      (all.calls > 1u ? BCIR_AB_ERR_OUTPUT : BCIR_AB_OK))
    abort();
  refusing.calls = 0;
  if (bcir_ab_format_hexdump(data, size, width, sink, &refusing) !=
      (dump.calls > 1u ? BCIR_AB_ERR_OUTPUT : BCIR_AB_OK))
    abort();

  uint64_t accumulator = view.generation + view.provenance_digest;
  for (uint32_t index = 0; index < view.count; ++index) {
    bcir_ab_entry entry;
    if (bcir_ab_get(&view, index, &entry) != BCIR_AB_OK) abort();  /* open checked every entry */
    accumulator += entry.payload_size + entry.payload_crc32;
    if (entry.payload_size) accumulator += entry.payload[entry.payload_size - 1u];
    sink_t views = {0, 0};
    if (entry.kind == BCIR_AB_STREAM_PACK) {
      if (bcir_ab_pack_mnemonics(entry.payload, (size_t)entry.payload_size, sink, &views) !=
          BCIR_AB_OK)
        abort();  /* a bundle admits only a pack the decoder takes */
    } else if (entry.format == BCIR_AB_FMT_TEXT) {
      if (bcir_ab_text_listing(entry.payload, (size_t)entry.payload_size, sink, &views) !=
              BCIR_AB_OK ||
          bcir_ab_text_mnemonics(entry.payload, (size_t)entry.payload_size, sink, &views) !=
              BCIR_AB_OK)
        abort();
    }
  }
  bcir_ab_envelope envelope = {0};
  envelope.features = "avx2,simd128";
  envelope.endianness = BCIR_AB_ENDIAN_LITTLE;
  envelope.pointer_bits = 64;
  envelope.require_r12 = 1;
  uint32_t selected = BCIR_AB_NO_INDEX;
  (void)bcir_ab_select(&view, &envelope, NULL, &selected);
  if (accumulator == UINT64_MAX && selected == 0) abort();  /* keeps the reads live */
}

static void put32(uint8_t *p, uint32_t v) {
  for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint32_t width = size ? 1u + (uint32_t)(data[0] % 64u) : 16u;
  drive(data, size, width);
  if (size >= BCIR_AB_HEADER_SIZE && size <= (1u << 20)) {
    uint8_t *sealed = (uint8_t *)malloc(size);
    if (!sealed) abort();
    memcpy(sealed, data, size);
    put32(sealed + 76, bcir_crc32(sealed + BCIR_AB_HEADER_SIZE, size - BCIR_AB_HEADER_SIZE));
    memset(sealed + 80, 0, 36);
    bcir_sha256 sha;
    bcir_sha256_init(&sha);
    bcir_sha256_update(&sha, sealed, size);
    bcir_sha256_final(&sha, sealed + 84);
    put32(sealed + 80, bcir_crc32(sealed, BCIR_AB_HEADER_SIZE));
    drive(sealed, size, width);
    free(sealed);
  }
  return 0;
}
