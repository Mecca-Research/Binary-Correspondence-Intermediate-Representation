/*===- bcir_sp_listing.c - StreamPack listing and hexadecimal dump (MC1) ------===
 *
 * Freestanding: only <stddef.h>/<stdint.h> through the runtime header and the shared text writer
 * (bcir_listing_text.h), no heap, no libc. Every line
 * is the one bcir/abi/streampack_tool.py prints, field for field; the comments name the Python
 * expression each piece spells. Held octet for octet to the oracle by
 * bcir/tests/test_c_streampack_listing.py, fuzzed under libFuzzer + ASan/UBSan
 * (runtime/c/fuzz_sp_listing.c).
 *===----------------------------------------------------------------------===*/
#include "bcir_sp_listing.h"

/* repr(list(strings)) spelled with bcir_lt_quoted: "[" + ", ".join(...) + "]". */
static void put_quoted_list(bcir_lt_out *o, bcir_sp_strings strings) {
  const char *s;
  uint16_t n;
  int first = 1;
  bcir_lt_c(o, '[');
  while (bcir_sp_strings_next(&strings, &s, &n)) {
    if (!first) bcir_lt_put(o, ", ", 2);
    bcir_lt_quoted(o, s, n);
    first = 0;
  }
  bcir_lt_c(o, ']');
}

/* Little-endian loads and stores over the walk's checked octets and the borrowed scratch. */
static uint32_t ld32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t ld64(const uint8_t *p) { return (uint64_t)ld32(p) | ((uint64_t)ld32(p + 4) << 32); }
static void st32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void st64(uint8_t *p, uint64_t v) { st32(p, (uint32_t)v); st32(p + 4, (uint32_t)(v >> 32)); }

/* _rids(values): "[" + ",".join(f"r{rid}") + "]", over little-endian u32s. */
static void put_rids(bcir_lt_out *o, const uint8_t *rids, uint16_t count) {
  bcir_lt_c(o, '[');
  for (uint16_t i = 0; i < count; i++) {
    if (i) bcir_lt_c(o, ',');
    bcir_lt_c(o, 'r');
    bcir_lt_dec(o, ld32(rids + (size_t)i * 4u));
  }
  bcir_lt_c(o, ']');
}

/* f"  @{span.offset:08x}+{span.length:<4} " -- every record line's span prefix. */
static void put_span(bcir_lt_out *o, const bcir_sp_record *r) {
  bcir_lt_put(o, "  @", 3);
  bcir_lt_hex(o, r->offset, 8);
  bcir_lt_c(o, '+');
  bcir_lt_dec_left(o, r->length, 4);
  bcir_lt_c(o, ' ');
}

static const char *const LANE[] = {"u", "ux", "t", "ggg", "a", "h"};  /* Lane.name.lower() */
static const char *const DISPATCH[] = {"core", "pim"};                /* _DISPATCH_FROM_WIRE */

/* --- the trace index: the claim-to-note map, sorted, in borrowed scratch --------------------
 * Twelve octets an entry: the claim id (u64) and the trace record's index (u32). Sorted by (claim,
 * index), the last entry of a claim is its last trace record -- the note `traced` keeps, since a
 * later key overwrites an earlier one in a dict. Heapsort: in place, no recursion (R18). */

#define IX_ENTRY 12u

static int ix_less(const uint8_t *ix, size_t a, size_t b) {
  uint64_t ca = ld64(ix + a * IX_ENTRY), cb = ld64(ix + b * IX_ENTRY);
  if (ca != cb) return ca < cb;
  return ld32(ix + a * IX_ENTRY + 8u) < ld32(ix + b * IX_ENTRY + 8u);
}
static void ix_swap(uint8_t *ix, size_t a, size_t b) {
  for (size_t k = 0; k < IX_ENTRY; k++) {
    uint8_t t = ix[a * IX_ENTRY + k];
    ix[a * IX_ENTRY + k] = ix[b * IX_ENTRY + k];
    ix[b * IX_ENTRY + k] = t;
  }
}
static void ix_sift(uint8_t *ix, size_t root, size_t n) {
  for (;;) {
    size_t child = 2u * root + 1u;
    if (child >= n) return;
    if (child + 1u < n && ix_less(ix, child, child + 1u)) child++;
    if (!ix_less(ix, root, child)) return;
    ix_swap(ix, root, child);
    root = child;
  }
}
static void ix_sort(uint8_t *ix, size_t n) {
  if (n < 2u) return;
  for (size_t i = n / 2u; i-- > 0u;) ix_sift(ix, i, n);
  for (size_t end = n - 1u; end > 0u; end--) {
    ix_swap(ix, 0, end);
    ix_sift(ix, 0, end);
  }
}
/* The trace record index of `claim`'s last note, or -1 when the claim is untraced. */
static int64_t ix_find(const uint8_t *ix, size_t n, uint64_t claim) {
  size_t lo = 0, hi = n;  /* the first entry whose claim is past `claim` */
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2u;
    if (ld64(ix + mid * IX_ENTRY) <= claim) lo = mid + 1u; else hi = mid;
  }
  if (lo == 0u || ld64(ix + (lo - 1u) * IX_ENTRY) != claim) return -1;
  return (int64_t)ld32(ix + (lo - 1u) * IX_ENTRY + 8u);
}

/* --- the listing ------------------------------------------------------------------------- */

typedef struct {
  bcir_lt_out *o;
  const uint8_t *data;
  size_t len;
  uint8_t *ix;     /* the trace index (scratch) */
  size_t n_ix;
  size_t trace0;   /* the offset of trace[0]: the records are 24 octets, back to back */
  int next;        /* the next section heading to print: 0 prefetches, 1 blocks, 2 generations */
} listing_t;

static int index_trace(const bcir_sp_record *r, void *ctx) {
  listing_t *l = (listing_t *)ctx;
  if (r->kind != BCIR_SP_RECORD_TRACE) return 0;
  if (r->index == 0u) l->trace0 = r->offset;
  st64(l->ix + (size_t)r->index * IX_ENTRY, r->trace.claim_id);
  st32(l->ix + (size_t)r->index * IX_ENTRY + 8u, r->index);
  return 0;
}

/* Every heading before section `upto` not yet printed (a section with no records still has one). */
static void headings(listing_t *l, int upto) {
  static const char *const HEADING[] = {"prefetches:\n", "blocks:\n", "generation vector:\n"};
  for (; l->next <= upto && l->next < 3; l->next++) bcir_lt_s(l->o, HEADING[l->next]);
}

static void list_segment(listing_t *l, const bcir_sp_record *r) {
  bcir_lt_out *o = l->o;
  const bcir_segment_view *s = &r->segment;
  put_span(o, r);
  bcir_lt_put(o, s->name, s->name_len);                 /* {seg.name}: {seg.opcode} */
  bcir_lt_put(o, ": ", 2);
  bcir_lt_put(o, s->opcode, s->opcode_len);
  bcir_lt_s(o, " claim="); bcir_lt_dec(o, s->claim_id);
  bcir_lt_s(o, " phase="); bcir_lt_dec(o, s->phase_id);
  bcir_lt_s(o, " lane="); bcir_lt_s(o, LANE[s->lane]);
  bcir_lt_s(o, " width="); bcir_lt_dec(o, s->width);
  bcir_lt_s(o, " rd="); put_rids(o, s->reads, s->n_reads);
  bcir_lt_s(o, " wr="); put_rids(o, s->writes, s->n_writes);
  bcir_lt_s(o, " dispatch="); bcir_lt_s(o, DISPATCH[s->dispatch]);
  bcir_lt_s(o, " channel="); bcir_lt_put(o, s->channel, s->channel_len);
  bcir_lt_s(o, " prefetch=");
  if (s->prefetch_len) bcir_lt_put(o, s->prefetch, s->prefetch_len); else bcir_lt_c(o, '-');
  int64_t note = ix_find(l->ix, l->n_ix, s->claim_id);
  if (note < 0) {
    bcir_lt_s(o, " untraced\n");
  } else {
    const uint8_t *t = l->data + l->trace0 + (size_t)note * 24u;
    bcir_lt_s(o, " src=0x"); bcir_lt_hex(o, ld64(t + 8), 16);
    bcir_lt_s(o, " trace=0x"); bcir_lt_hex(o, ld64(t + 16), 16);
    bcir_lt_c(o, '\n');
  }
  if (r->fence_before.count || r->fence_after.count) {
    bcir_lt_s(o, "    fences before="); put_quoted_list(o, r->fence_before);
    bcir_lt_s(o, " after="); put_quoted_list(o, r->fence_after);
    bcir_lt_c(o, '\n');
  }
}

static int list_record(const bcir_sp_record *r, void *ctx) {
  listing_t *l = (listing_t *)ctx;
  bcir_lt_out *o = l->o;
  const bcir_streampack_header *h = r->header;
  switch (r->kind) {
  case BCIR_SP_RECORD_HEADER:
    bcir_lt_s(o, "BSPK v"); bcir_lt_dec(o, h->version);
    bcir_lt_s(o, " bytes="); bcir_lt_dec(o, l->len);
    bcir_lt_s(o, " flags=0x"); bcir_lt_hex(o, h->flags, 4);
    bcir_lt_s(o, "\ngenerations topo="); bcir_lt_dec(o, h->topo_gen);
    bcir_lt_s(o, " map="); bcir_lt_dec(o, h->map_gen);
    bcir_lt_s(o, " data="); bcir_lt_dec(o, h->data_gen);
    bcir_lt_s(o, " pipeline_depth="); bcir_lt_dec(o, h->pipeline_depth);
    bcir_lt_s(o, "\nrecords segments="); bcir_lt_dec(o, h->n_segments);
    bcir_lt_s(o, " prefetches="); bcir_lt_dec(o, h->n_prefetches);
    bcir_lt_s(o, " blocks="); bcir_lt_dec(o, h->n_blocks);
    bcir_lt_s(o, " trace="); bcir_lt_dec(o, h->n_trace);
    bcir_lt_s(o, " generations="); bcir_lt_dec(o, h->n_gens);
    bcir_lt_c(o, '\n');
    break;
  case BCIR_SP_RECORD_SOURCE_PLAN:
    bcir_lt_s(o, "source_plan ");
    bcir_lt_quoted(o, r->source_plan, r->source_plan_len);
    bcir_lt_s(o, "\nsegments:\n");
    break;
  case BCIR_SP_RECORD_SEGMENT:
    list_segment(l, r);
    break;
  case BCIR_SP_RECORD_PREFETCH: {
    const bcir_prefetch_view *p = &r->prefetch;
    headings(l, 0);
    put_span(o, r);
    bcir_lt_put(o, p->name, p->name_len);
    bcir_lt_s(o, ": distance="); bcir_lt_dec(o, p->distance);
    bcir_lt_s(o, " targets="); put_rids(o, p->targets, p->n_targets);
    bcir_lt_s(o, " hint="); bcir_lt_put(o, p->hint, p->hint_len);
    bcir_lt_s(o, " pattern="); bcir_lt_put(o, p->pattern, p->pattern_len);
    bcir_lt_s(o, " buffers="); bcir_lt_dec(o, p->buffers);
    bcir_lt_c(o, '\n');
    break;
  }
  case BCIR_SP_RECORD_BLOCK:
    headings(l, 1);
    put_span(o, r);
    bcir_lt_s(o, "block"); bcir_lt_dec(o, r->index);
    bcir_lt_s(o, ": base="); bcir_lt_dec(o, r->block.base);
    bcir_lt_s(o, " count="); bcir_lt_dec(o, r->block.count);
    bcir_lt_s(o, " strides=[");                     /* repr(list(block.strides)) */
    for (uint16_t i = 0; i < r->block.n_strides; i++) {
      if (i) bcir_lt_put(o, ", ", 2);
      bcir_lt_dec(o, bcir_block_stride(&r->block, i));
    }
    bcir_lt_s(o, "]\n");
    break;
  case BCIR_SP_RECORD_TRACE:
    break;  /* shown on its segment's line */
  case BCIR_SP_RECORD_GENERATION:
    headings(l, 2);
    put_span(o, r);
    bcir_lt_c(o, 'r'); bcir_lt_dec(o, r->generation.rid);
    bcir_lt_s(o, ": map="); bcir_lt_dec(o, r->generation.map_gen);
    bcir_lt_s(o, " data="); bcir_lt_dec(o, r->generation.data_gen);
    bcir_lt_c(o, '\n');
    break;
  case BCIR_SP_RECORD_CRC32:
    headings(l, 2);
    bcir_lt_s(o, "crc32 @"); bcir_lt_hex(o, r->offset, 8);
    bcir_lt_s(o, " = 0x"); bcir_lt_hex(o, r->crc32, 8);
    bcir_lt_s(o, " valid\n");
    break;
  }
  return o->failed;
}

/* What a checked pack's listing borrows: 12 octets a trace record. The walk has held every declared
 * trace record to the body (24 octets each), so the product is below half the pack's length. */
static bcir_status listing_need(const bcir_sp_checked *pack, size_t *need) {
  bcir_status st = bcir_sp_checked_for_each_record(pack, 0, 0);
  *need = st == BCIR_OK ? (size_t)pack->hdr.n_trace * IX_ENTRY : 0u;
  return st;
}

bcir_status bcir_sp_listing_scratch_bytes(const uint8_t *BCIR_RESTRICT data, size_t len,
                                          size_t *BCIR_RESTRICT need) {
  size_t n = 0;
  bcir_sp_checked pack;
  bcir_status st = bcir_sp_check(data, len, &pack);
  if (st == BCIR_OK) st = listing_need(&pack, &n);
  if (need) *need = n;
  return st;
}

bcir_status bcir_sp_format_listing(const uint8_t *BCIR_RESTRICT data, size_t len,
                                   uint8_t *BCIR_RESTRICT scratch, size_t scratch_len,
                                   bcir_sink_fn sink, void *ctx) {
  size_t need;
  bcir_sp_checked pack;
  bcir_status st = bcir_sp_check(data, len, &pack);   /* the CRC, once */
  if (st == BCIR_OK) st = listing_need(&pack, &need);
  if (st != BCIR_OK) return st;
  if (!sink || scratch_len < need || (need && !scratch)) return BCIR_ERR_NOSPACE;
  bcir_lt_out o;
  bcir_lt_open(&o, sink, ctx);
  listing_t l;
  l.o = &o; l.data = data; l.len = len; l.ix = scratch; l.n_ix = need / IX_ENTRY;
  l.trace0 = 0; l.next = 0;
  if (need) {
    st = bcir_sp_checked_for_each_record(&pack, index_trace, &l);
    if (st != BCIR_OK) return st;
    ix_sort(l.ix, l.n_ix);
  }
  st = bcir_sp_checked_for_each_record(&pack, list_record, &l);
  if (st != BCIR_OK) return st;
  bcir_lt_flush(&o);
  return o.failed ? BCIR_ERR_NOSPACE : BCIR_OK;
}

/* --- the dump ---------------------------------------------------------------------------- */

typedef struct {
  bcir_lt_out *o;
  const uint8_t *data;
  uint32_t width;
} dump_t;

static const char *const KIND[] = {"header", "source_plan", "segment", "prefetch",
                                   "block", "trace", "generation", "crc32"};

static int dump_record(const bcir_sp_record *r, void *ctx) {
  dump_t *d = (dump_t *)ctx;
  bcir_lt_out *o = d->o;
  /* f"# {span.kind}{index} {span.name!r} offset=0x{span.offset:08x} length={span.length}" */
  bcir_lt_s(o, "# "); bcir_lt_s(o, KIND[r->kind]);
  int indexed = r->kind != BCIR_SP_RECORD_HEADER && r->kind != BCIR_SP_RECORD_SOURCE_PLAN &&
                r->kind != BCIR_SP_RECORD_CRC32;
  if (indexed) { bcir_lt_c(o, '['); bcir_lt_dec(o, r->index); bcir_lt_c(o, ']'); }
  bcir_lt_c(o, ' ');
  switch (r->kind) {
  case BCIR_SP_RECORD_HEADER: bcir_lt_s(o, "'header'"); break;
  case BCIR_SP_RECORD_SOURCE_PLAN: bcir_lt_quoted(o, r->source_plan, r->source_plan_len); break;
  case BCIR_SP_RECORD_SEGMENT: bcir_lt_quoted(o, r->segment.name, r->segment.name_len); break;
  case BCIR_SP_RECORD_PREFETCH: bcir_lt_quoted(o, r->prefetch.name, r->prefetch.name_len); break;
  case BCIR_SP_RECORD_BLOCK: bcir_lt_s(o, "'block"); bcir_lt_dec(o, r->index); bcir_lt_c(o, '\''); break;
  case BCIR_SP_RECORD_TRACE: bcir_lt_s(o, "'claim"); bcir_lt_dec(o, r->trace.claim_id); bcir_lt_c(o, '\''); break;
  case BCIR_SP_RECORD_GENERATION: bcir_lt_s(o, "'rid"); bcir_lt_dec(o, r->generation.rid); bcir_lt_c(o, '\''); break;
  case BCIR_SP_RECORD_CRC32: bcir_lt_s(o, "'crc32'"); break;
  }
  bcir_lt_s(o, " offset=0x"); bcir_lt_hex(o, r->offset, 8);
  bcir_lt_s(o, " length="); bcir_lt_dec(o, r->length);
  bcir_lt_c(o, '\n');
  bcir_lt_rows(o, d->data, r->offset, r->length, d->width);  /* the span's rows */
  return o->failed;
}

bcir_status bcir_sp_format_hexdump(const uint8_t *BCIR_RESTRICT data, size_t len, uint32_t width,
                                   bcir_sink_fn sink, void *ctx) {
  if (width < 1u || width > BCIR_HEXDUMP_WIDTH_MAX) return BCIR_ERR_OVERFLOW;
  if (!sink) return BCIR_ERR_NOSPACE;
  bcir_status st;
  bcir_lt_out o;
  bcir_lt_open(&o, sink, ctx);
  dump_t d;
  d.o = &o; d.data = data; d.width = width;
  st = bcir_sp_for_each_record(data, len, dump_record, &d);
  if (st != BCIR_OK) return st;
  bcir_lt_flush(&o);
  return o.failed ? BCIR_ERR_NOSPACE : BCIR_OK;
}
