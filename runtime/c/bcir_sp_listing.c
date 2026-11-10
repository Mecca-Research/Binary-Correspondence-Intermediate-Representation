/*===- bcir_sp_listing.c - StreamPack listing and hexadecimal dump (MC1) ------===
 *
 * Freestanding: only <stddef.h>/<stdint.h> through the runtime header, no heap, no libc. Every line
 * is the one bcir/abi/streampack_tool.py prints, field for field; the comments name the Python
 * expression each piece spells. Held octet for octet to the oracle by
 * bcir/tests/test_c_streampack_listing.py, fuzzed under libFuzzer + ASan/UBSan
 * (runtime/c/fuzz_sp_listing.c).
 *===----------------------------------------------------------------------===*/
#include "bcir_sp_listing.h"

/* --- the output: a chunk buffer in front of the sink ----------------------------------------- */

typedef struct {
  char buf[BCIR_LISTING_CHUNK];
  size_t n;
  bcir_sink_fn sink;
  void *ctx;
  int failed;  /* the sink refused: nothing more is offered to it */
} out_t;

static void out_flush(out_t *o) {
  if (!o->failed && o->n && o->sink(o->ctx, o->buf, o->n)) o->failed = 1;
  o->n = 0;
}

static void put(out_t *o, const char *s, size_t n) {
  while (n && !o->failed) {
    size_t room = sizeof o->buf - o->n;
    size_t take = n < room ? n : room;
    for (size_t i = 0; i < take; i++) o->buf[o->n + i] = s[i];
    o->n += take; s += take; n -= take;
    if (o->n == sizeof o->buf) out_flush(o);
  }
}

static void put_c(out_t *o, char ch) { put(o, &ch, 1); }

static void put_s(out_t *o, const char *s) {
  size_t n = 0;
  while (s[n]) n++;
  put(o, s, n);
}

static const char HEX[] = "0123456789abcdef";

/* f"{v:0{digits}x}": lowercase, at least `digits` wide, never truncated. */
static void put_hex(out_t *o, uint64_t v, unsigned digits) {
  char t[16];
  unsigned n = 0;
  do { t[n++] = HEX[v & 0xFu]; v >>= 4; } while (v && n < 16u);
  while (n < digits && n < 16u) t[n++] = '0';
  while (n) put_c(o, t[--n]);
}

/* f"{v}" for an unsigned integer, and f"{v:<width}": left-aligned, space-padded, never truncated. */
static unsigned dec(char t[20], uint64_t v) {
  unsigned n = 0;
  do { t[n++] = (char)('0' + (int)(v % 10u)); v /= 10u; } while (v);
  return n;  /* reversed */
}
static void put_dec(out_t *o, uint64_t v) {
  char t[20];
  unsigned n = dec(t, v);
  while (n) put_c(o, t[--n]);
}
static void put_dec_left(out_t *o, uint64_t v, unsigned width) {
  char t[20];
  unsigned n = dec(t, v), shown = n;
  while (n) put_c(o, t[--n]);
  for (; shown < width; shown++) put_c(o, ' ');
}

/* ascii(text) for a str whose octets the walk checked as UTF-8 (see the header for why ascii and not
 * repr): the quote is " when the text holds ' and no ", else '; the quote and the backslash are
 * escaped, \t \n \r by name, every other code point below U+0020, U+007F and every one past it by
 * number. */
static void put_quoted(out_t *o, const char *text, size_t n) {
  const uint8_t *s = (const uint8_t *)text;
  int single = 0, dbl = 0;
  for (size_t i = 0; i < n; i++) {
    if (s[i] == '\'') single = 1;
    if (s[i] == '"') dbl = 1;
  }
  char quote = single && !dbl ? '"' : '\'';
  put_c(o, quote);
  size_t i = 0;
  while (i < n) {
    uint32_t cp = s[i];
    size_t extra = cp < 0x80u ? 0u : cp < 0xE0u ? 1u : cp < 0xF0u ? 2u : 3u;
    if (extra) cp &= extra == 1u ? 0x1Fu : extra == 2u ? 0x0Fu : 0x07u;
    i++;
    for (size_t k = 0; k < extra && i < n; k++) cp = (cp << 6) | (uint32_t)(s[i++] & 0x3Fu);
    if (cp == (uint32_t)(uint8_t)quote || cp == '\\') {
      put_c(o, '\\'); put_c(o, (char)cp);
    } else if (cp == '\t') {
      put(o, "\\t", 2);
    } else if (cp == '\n') {
      put(o, "\\n", 2);
    } else if (cp == '\r') {
      put(o, "\\r", 2);
    } else if (cp >= 0x20u && cp < 0x7Fu) {
      put_c(o, (char)cp);
    } else if (cp < 0x100u) {
      put(o, "\\x", 2); put_hex(o, cp, 2);
    } else if (cp < 0x10000u) {
      put(o, "\\u", 2); put_hex(o, cp, 4);
    } else {
      put(o, "\\U", 2); put_hex(o, cp, 8);
    }
  }
  put_c(o, quote);
}

/* repr(list(strings)) spelled with put_quoted: "[" + ", ".join(...) + "]". */
static void put_quoted_list(out_t *o, bcir_sp_strings strings) {
  const char *s;
  uint16_t n;
  int first = 1;
  put_c(o, '[');
  while (bcir_sp_strings_next(&strings, &s, &n)) {
    if (!first) put(o, ", ", 2);
    put_quoted(o, s, n);
    first = 0;
  }
  put_c(o, ']');
}

/* _rids(values): "[" + ",".join(f"r{rid}") + "]", over little-endian u32s. */
static uint32_t ld32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t ld64(const uint8_t *p) { return (uint64_t)ld32(p) | ((uint64_t)ld32(p + 4) << 32); }
static void st32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void st64(uint8_t *p, uint64_t v) { st32(p, (uint32_t)v); st32(p + 4, (uint32_t)(v >> 32)); }

static void put_rids(out_t *o, const uint8_t *rids, uint16_t count) {
  put_c(o, '[');
  for (uint16_t i = 0; i < count; i++) {
    if (i) put_c(o, ',');
    put_c(o, 'r');
    put_dec(o, ld32(rids + (size_t)i * 4u));
  }
  put_c(o, ']');
}

/* f"  @{span.offset:08x}+{span.length:<4} " -- every record line's span prefix. */
static void put_span(out_t *o, const bcir_sp_record *r) {
  put(o, "  @", 3);
  put_hex(o, r->offset, 8);
  put_c(o, '+');
  put_dec_left(o, r->length, 4);
  put_c(o, ' ');
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
  out_t *o;
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
  for (; l->next <= upto && l->next < 3; l->next++) put_s(l->o, HEADING[l->next]);
}

static void list_segment(listing_t *l, const bcir_sp_record *r) {
  out_t *o = l->o;
  const bcir_segment_view *s = &r->segment;
  put_span(o, r);
  put(o, s->name, s->name_len);                 /* {seg.name}: {seg.opcode} */
  put(o, ": ", 2);
  put(o, s->opcode, s->opcode_len);
  put_s(o, " claim="); put_dec(o, s->claim_id);
  put_s(o, " phase="); put_dec(o, s->phase_id);
  put_s(o, " lane="); put_s(o, LANE[s->lane]);
  put_s(o, " width="); put_dec(o, s->width);
  put_s(o, " rd="); put_rids(o, s->reads, s->n_reads);
  put_s(o, " wr="); put_rids(o, s->writes, s->n_writes);
  put_s(o, " dispatch="); put_s(o, DISPATCH[s->dispatch]);
  put_s(o, " channel="); put(o, s->channel, s->channel_len);
  put_s(o, " prefetch=");
  if (s->prefetch_len) put(o, s->prefetch, s->prefetch_len); else put_c(o, '-');
  int64_t note = ix_find(l->ix, l->n_ix, s->claim_id);
  if (note < 0) {
    put_s(o, " untraced\n");
  } else {
    const uint8_t *t = l->data + l->trace0 + (size_t)note * 24u;
    put_s(o, " src=0x"); put_hex(o, ld64(t + 8), 16);
    put_s(o, " trace=0x"); put_hex(o, ld64(t + 16), 16);
    put_c(o, '\n');
  }
  if (r->fence_before.count || r->fence_after.count) {
    put_s(o, "    fences before="); put_quoted_list(o, r->fence_before);
    put_s(o, " after="); put_quoted_list(o, r->fence_after);
    put_c(o, '\n');
  }
}

static int list_record(const bcir_sp_record *r, void *ctx) {
  listing_t *l = (listing_t *)ctx;
  out_t *o = l->o;
  const bcir_streampack_header *h = r->header;
  switch (r->kind) {
  case BCIR_SP_RECORD_HEADER:
    put_s(o, "BSPK v"); put_dec(o, h->version);
    put_s(o, " bytes="); put_dec(o, l->len);
    put_s(o, " flags=0x"); put_hex(o, h->flags, 4);
    put_s(o, "\ngenerations topo="); put_dec(o, h->topo_gen);
    put_s(o, " map="); put_dec(o, h->map_gen);
    put_s(o, " data="); put_dec(o, h->data_gen);
    put_s(o, " pipeline_depth="); put_dec(o, h->pipeline_depth);
    put_s(o, "\nrecords segments="); put_dec(o, h->n_segments);
    put_s(o, " prefetches="); put_dec(o, h->n_prefetches);
    put_s(o, " blocks="); put_dec(o, h->n_blocks);
    put_s(o, " trace="); put_dec(o, h->n_trace);
    put_s(o, " generations="); put_dec(o, h->n_gens);
    put_c(o, '\n');
    break;
  case BCIR_SP_RECORD_SOURCE_PLAN:
    put_s(o, "source_plan ");
    put_quoted(o, r->source_plan, r->source_plan_len);
    put_s(o, "\nsegments:\n");
    break;
  case BCIR_SP_RECORD_SEGMENT:
    list_segment(l, r);
    break;
  case BCIR_SP_RECORD_PREFETCH: {
    const bcir_prefetch_view *p = &r->prefetch;
    headings(l, 0);
    put_span(o, r);
    put(o, p->name, p->name_len);
    put_s(o, ": distance="); put_dec(o, p->distance);
    put_s(o, " targets="); put_rids(o, p->targets, p->n_targets);
    put_s(o, " hint="); put(o, p->hint, p->hint_len);
    put_s(o, " pattern="); put(o, p->pattern, p->pattern_len);
    put_s(o, " buffers="); put_dec(o, p->buffers);
    put_c(o, '\n');
    break;
  }
  case BCIR_SP_RECORD_BLOCK:
    headings(l, 1);
    put_span(o, r);
    put_s(o, "block"); put_dec(o, r->index);
    put_s(o, ": base="); put_dec(o, r->block.base);
    put_s(o, " count="); put_dec(o, r->block.count);
    put_s(o, " strides=[");                     /* repr(list(block.strides)) */
    for (uint16_t i = 0; i < r->block.n_strides; i++) {
      if (i) put(o, ", ", 2);
      put_dec(o, bcir_block_stride(&r->block, i));
    }
    put_s(o, "]\n");
    break;
  case BCIR_SP_RECORD_TRACE:
    break;  /* shown on its segment's line */
  case BCIR_SP_RECORD_GENERATION:
    headings(l, 2);
    put_span(o, r);
    put_c(o, 'r'); put_dec(o, r->generation.rid);
    put_s(o, ": map="); put_dec(o, r->generation.map_gen);
    put_s(o, " data="); put_dec(o, r->generation.data_gen);
    put_c(o, '\n');
    break;
  case BCIR_SP_RECORD_CRC32:
    headings(l, 2);
    put_s(o, "crc32 @"); put_hex(o, r->offset, 8);
    put_s(o, " = 0x"); put_hex(o, r->crc32, 8);
    put_s(o, " valid\n");
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
  out_t o;
  o.n = 0; o.sink = sink; o.ctx = ctx; o.failed = 0;
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
  out_flush(&o);
  return o.failed ? BCIR_ERR_NOSPACE : BCIR_OK;
}

/* --- the dump ---------------------------------------------------------------------------- */

typedef struct {
  out_t *o;
  const uint8_t *data;
  uint32_t width;
} dump_t;

static const char *const KIND[] = {"header", "source_plan", "segment", "prefetch",
                                   "block", "trace", "generation", "crc32"};

static int dump_record(const bcir_sp_record *r, void *ctx) {
  dump_t *d = (dump_t *)ctx;
  out_t *o = d->o;
  /* f"# {span.kind}{index} {span.name!r} offset=0x{span.offset:08x} length={span.length}" */
  put_s(o, "# "); put_s(o, KIND[r->kind]);
  int indexed = r->kind != BCIR_SP_RECORD_HEADER && r->kind != BCIR_SP_RECORD_SOURCE_PLAN &&
                r->kind != BCIR_SP_RECORD_CRC32;
  if (indexed) { put_c(o, '['); put_dec(o, r->index); put_c(o, ']'); }
  put_c(o, ' ');
  switch (r->kind) {
  case BCIR_SP_RECORD_HEADER: put_s(o, "'header'"); break;
  case BCIR_SP_RECORD_SOURCE_PLAN: put_quoted(o, r->source_plan, r->source_plan_len); break;
  case BCIR_SP_RECORD_SEGMENT: put_quoted(o, r->segment.name, r->segment.name_len); break;
  case BCIR_SP_RECORD_PREFETCH: put_quoted(o, r->prefetch.name, r->prefetch.name_len); break;
  case BCIR_SP_RECORD_BLOCK: put_s(o, "'block"); put_dec(o, r->index); put_c(o, '\''); break;
  case BCIR_SP_RECORD_TRACE: put_s(o, "'claim"); put_dec(o, r->trace.claim_id); put_c(o, '\''); break;
  case BCIR_SP_RECORD_GENERATION: put_s(o, "'rid"); put_dec(o, r->generation.rid); put_c(o, '\''); break;
  case BCIR_SP_RECORD_CRC32: put_s(o, "'crc32'"); break;
  }
  put_s(o, " offset=0x"); put_hex(o, r->offset, 8);
  put_s(o, " length="); put_dec(o, r->length);
  put_c(o, '\n');
  /* f"{offset:08x}  {hex_bytes:<{width * 3 - 1}}  |{ascii_bytes}|" per row of the span */
  const size_t end = r->offset + r->length;
  for (size_t at = r->offset; at < end && !o->failed; at += d->width) {
    size_t n = end - at < d->width ? end - at : d->width;
    put_hex(o, at, 8);
    put(o, "  ", 2);
    for (size_t i = 0; i < n; i++) {
      if (i) put_c(o, ' ');
      put_hex(o, d->data[at + i], 2);
    }
    for (size_t pad = n * 3u - 1u; pad < (size_t)d->width * 3u - 1u; pad++) put_c(o, ' ');
    put(o, "  |", 3);
    for (size_t i = 0; i < n; i++) {
      uint8_t b = d->data[at + i];
      put_c(o, b >= 32u && b < 127u ? (char)b : '.');
    }
    put(o, "|\n", 2);
  }
  return o->failed;
}

bcir_status bcir_sp_format_hexdump(const uint8_t *BCIR_RESTRICT data, size_t len, uint32_t width,
                                   bcir_sink_fn sink, void *ctx) {
  if (width < 1u || width > BCIR_HEXDUMP_WIDTH_MAX) return BCIR_ERR_OVERFLOW;
  if (!sink) return BCIR_ERR_NOSPACE;
  bcir_status st;
  out_t o;
  o.n = 0; o.sink = sink; o.ctx = ctx; o.failed = 0;
  dump_t d;
  d.o = &o; d.data = data; d.width = width;
  st = bcir_sp_for_each_record(data, len, dump_record, &d);
  if (st != BCIR_OK) return st;
  out_flush(&o);
  return o.failed ? BCIR_ERR_NOSPACE : BCIR_OK;
}
