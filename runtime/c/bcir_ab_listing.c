/*===- bcir_ab_listing.c - BCAB listing and hexadecimal dump (MC1) ------------===
 *
 * Freestanding: the BCAB reader's view and entries, the shared SHA-256, the listings' one text
 * writer. Every line is the one bcir/abi/artifact_tool.py prints, field for field. Held octet for
 * octet to the oracle by bcir/tests/test_c_bundle_listing.py, fuzzed (runtime/c/fuzz_ab_listing.c).
 *===----------------------------------------------------------------------===*/
#include "bcir_ab_listing.h"
#include "bcir_runtime.h"
#include "bcir_sha256.h"

/* ArtifactKind / ArtifactFormat / Endianness .name.lower(), by wire value (bcir_ab_open admits no
 * other value). */
static const char *const KIND[26] = {
    "", "stream_pack", "elf_object", "elf_shared", "coff_object", "macho_object", "archive",
    "wasm", "llvm_bitcode", "llvm_ir", "ptx", "cubin", "spirv", "jvm_class", "cil", "c_source",
    "cpp_source", "sycl_source", "assembly", "elf_executable", "pe_executable", "pe_shared",
    "macho_executable", "macho_shared", "raw_binary", "execution_plan"};
static const char *const FORMAT[14] = {
    "none", "stream_pack", "elf", "coff", "macho", "archive", "wasm", "llvm_bitcode", "text",
    "spirv", "jvm_class", "pe", "raw", "execution_plan"};
static const char *const ENDIAN[3] = {"neutral", "little", "big"};

static const char *name_of(const char *const *names, unsigned count, unsigned value) {
  return value < count ? names[value] : "?";
}

static void put_digest(bcir_lt_out *o, const uint8_t digest[32]) {
  for (unsigned i = 0; i < 32u; i++) bcir_lt_hex(o, digest[i], 2);
}

static int zero_digest(const uint8_t digest[32]) {
  for (unsigned i = 0; i < 32u; i++)
    if (digest[i]) return 0;
  return 1;
}

/* f"{value or '-'}" for a NUL-terminated field. */
static void put_or_dash(bcir_lt_out *o, const char *value) {
  if (value[0]) bcir_lt_s(o, value); else bcir_lt_c(o, '-');
}

/* _flags(variant): exec, r12, portable, debug in that order, comma-joined, or "-". */
static void put_flags(bcir_lt_out *o, uint16_t flags) {
  static const struct { uint16_t bit; const char *name; } order[] = {
      {BCIR_AB_FLAG_EXECUTABLE, "exec"}, {BCIR_AB_FLAG_R12_ATTESTED, "r12"},
      {BCIR_AB_FLAG_PORTABLE, "portable"}, {BCIR_AB_FLAG_DEBUG, "debug"}};
  int any = 0;
  for (unsigned i = 0; i < 4u; i++) {
    if (!(flags & order[i].bit)) continue;
    if (any) bcir_lt_c(o, ',');
    bcir_lt_s(o, order[i].name);
    any = 1;
  }
  if (!any) bcir_lt_c(o, '-');
}

/* The variant id of `index`, or "-" for BCIR_AB_NO_INDEX. */
static void put_id_of(bcir_lt_out *o, const bcir_ab_view *v, uint32_t index) {
  bcir_ab_entry e;
  if (index == BCIR_AB_NO_INDEX || bcir_ab_get(v, index, &e) != BCIR_AB_OK) {
    bcir_lt_c(o, '-');
    return;
  }
  bcir_lt_s(o, e.variant_id);
}

static bcir_ab_status finish(bcir_lt_out *o) {
  bcir_lt_flush(o);
  return o->failed ? BCIR_AB_ERR_OUTPUT : BCIR_AB_OK;
}

bcir_ab_status bcir_ab_format_listing(const uint8_t *data, size_t len, bcir_sink_fn sink,
                                      void *ctx) {
  bcir_ab_view v;
  bcir_ab_status st = bcir_ab_open(data, len, &v);
  if (st != BCIR_AB_OK) return st;
  if (!sink) return BCIR_AB_ERR_INVALID;
  uint8_t digest[32];
  bcir_sha256 sha;
  bcir_sha256_init(&sha);
  bcir_sha256_update(&sha, data, len);
  bcir_sha256_final(&sha, digest);

  bcir_lt_out o;
  bcir_lt_open(&o, sink, ctx);
  bcir_lt_s(&o, "BCAB v1 bytes="); bcir_lt_dec(&o, v.length);
  bcir_lt_s(&o, " entries="); bcir_lt_dec(&o, v.count);
  bcir_lt_s(&o, "\nsha256="); put_digest(&o, digest);
  bcir_lt_s(&o, " embedded_sha256="); put_digest(&o, v.embedded_sha256);
  bcir_lt_s(&o, "\ncrc32 body=0x"); bcir_lt_hex(&o, v.body_crc32, 8);
  bcir_lt_s(&o, " header=0x"); bcir_lt_hex(&o, v.header_crc32, 8);
  bcir_lt_s(&o, "\nprovenance=0x"); bcir_lt_hex(&o, v.provenance_digest, 16);
  bcir_lt_s(&o, " generation="); bcir_lt_dec(&o, v.generation);
  bcir_lt_s(&o, " root="); put_id_of(&o, &v, v.root_index);
  bcir_lt_s(&o, " default="); put_id_of(&o, &v, v.default_index);
  bcir_lt_s(&o, "\nvariants:\n");
  for (uint32_t i = 0; i < v.count && !o.failed; i++) {
    bcir_ab_entry e;
    if (bcir_ab_get(&v, i, &e) != BCIR_AB_OK) return BCIR_AB_ERR_INVALID;  /* open checked it */
    bcir_lt_s(&o, "  ["); bcir_lt_dec(&o, i); bcir_lt_s(&o, "] "); bcir_lt_s(&o, e.variant_id);
    bcir_lt_s(&o, " kind="); bcir_lt_s(&o, name_of(KIND, 26u, e.kind));
    bcir_lt_s(&o, " format="); bcir_lt_s(&o, name_of(FORMAT, 14u, e.format));
    bcir_lt_s(&o, " @"); bcir_lt_hex(&o, e.payload_offset, 8);
    bcir_lt_c(&o, '+'); bcir_lt_dec(&o, e.payload_size);
    bcir_lt_s(&o, " priority="); bcir_lt_sdec(&o, e.priority);
    bcir_lt_s(&o, " flags="); put_flags(&o, e.flags);
    bcir_lt_s(&o, "\n      target triple="); put_or_dash(&o, e.triple);
    bcir_lt_s(&o, " arch="); put_or_dash(&o, e.architecture);
    bcir_lt_s(&o, " os_abi="); put_or_dash(&o, e.os_abi);
    bcir_lt_s(&o, " channel="); put_or_dash(&o, e.channel);
    bcir_lt_s(&o, " endian="); bcir_lt_s(&o, name_of(ENDIAN, 3u, e.endianness));
    bcir_lt_s(&o, " bits=");
    if (e.pointer_bits) bcir_lt_dec(&o, e.pointer_bits); else bcir_lt_c(&o, '-');
    bcir_lt_s(&o, " machine=");
    if (e.machine) bcir_lt_dec(&o, e.machine); else bcir_lt_c(&o, '-');
    bcir_lt_s(&o, "\n      features require="); put_or_dash(&o, e.required_features);
    bcir_lt_s(&o, " prohibit="); put_or_dash(&o, e.prohibited_features);
    bcir_lt_s(&o, " entry="); put_or_dash(&o, e.entry_symbol);
    bcir_lt_s(&o, " cal_gen="); bcir_lt_dec(&o, e.cal_gen);
    bcir_lt_s(&o, "\n      crc32=0x"); bcir_lt_hex(&o, e.payload_crc32, 8);
    bcir_lt_s(&o, " sha256="); put_digest(&o, e.payload_sha256);
    bcir_lt_s(&o, " manifest=");
    if (zero_digest(e.target_manifest_sha256)) bcir_lt_c(&o, '-');
    else put_digest(&o, e.target_manifest_sha256);
    bcir_lt_c(&o, '\n');
  }
  return finish(&o);
}

/* One span's heading: f"# {kind}{index} {name!r} offset=0x{offset:08x} length={length}". `name`
 * and `prefix` are ASCII (bcir_ab_open admits printable ASCII ids only). */
static void heading(bcir_lt_out *o, const char *kind, int64_t index, const char *prefix,
                    const char *name, uint64_t offset, uint64_t length) {
  char text[64];
  size_t n = 0;
  for (const char *p = prefix; *p && n < sizeof text - 1u; p++) text[n++] = *p;
  for (const char *p = name; *p && n < sizeof text - 1u; p++) text[n++] = *p;
  bcir_lt_s(o, "# "); bcir_lt_s(o, kind);
  if (index >= 0) { bcir_lt_c(o, '['); bcir_lt_dec(o, (uint64_t)index); bcir_lt_c(o, ']'); }
  bcir_lt_c(o, ' ');
  bcir_lt_quoted(o, text, n);
  bcir_lt_s(o, " offset=0x"); bcir_lt_hex(o, offset, 8);
  bcir_lt_s(o, " length="); bcir_lt_dec(o, length);
  bcir_lt_c(o, '\n');
}

bcir_ab_status bcir_ab_format_hexdump(const uint8_t *data, size_t len, uint32_t width,
                                      bcir_sink_fn sink, void *ctx) {
  if (width < 1u || width > 64u) return BCIR_AB_ERR_INVALID;
  bcir_ab_view v;
  bcir_ab_status st = bcir_ab_open(data, len, &v);
  if (st != BCIR_AB_OK) return st;
  if (!sink) return BCIR_AB_ERR_INVALID;
  bcir_lt_out o;
  bcir_lt_open(&o, sink, ctx);
  /* The spans in offset order (the oracle sorts them): the header, the directory entries, then
   * each payload after the zero padding that aligns it, then the final padding. */
  heading(&o, "header", -1, "", "header", 0, BCIR_AB_HEADER_SIZE);
  bcir_lt_rows(&o, data, 0, BCIR_AB_HEADER_SIZE, width);
  for (uint32_t i = 0; i < v.count && !o.failed; i++) {
    bcir_ab_entry e;
    if (bcir_ab_get(&v, i, &e) != BCIR_AB_OK) return BCIR_AB_ERR_INVALID;
    size_t at = BCIR_AB_HEADER_SIZE + (size_t)i * BCIR_AB_ENTRY_SIZE;
    heading(&o, "directory", i, "", e.variant_id, at, BCIR_AB_ENTRY_SIZE);
    bcir_lt_rows(&o, data, at, BCIR_AB_ENTRY_SIZE, width);
  }
  size_t previous = BCIR_AB_HEADER_SIZE + (size_t)v.count * BCIR_AB_ENTRY_SIZE;
  for (uint32_t i = 0; i < v.count && !o.failed; i++) {
    bcir_ab_entry e;
    if (bcir_ab_get(&v, i, &e) != BCIR_AB_OK) return BCIR_AB_ERR_INVALID;
    size_t at = (size_t)e.payload_offset, size = (size_t)e.payload_size;
    if (at > previous) {
      heading(&o, "padding", -1, "before:", e.variant_id, previous, at - previous);
      bcir_lt_rows(&o, data, previous, at - previous, width);
    }
    heading(&o, "payload", i, "", e.variant_id, at, size);
    bcir_lt_rows(&o, data, at, size, width);
    previous = at + size;
  }
  if (len > previous && !o.failed) {
    heading(&o, "padding", -1, "", "final", previous, len - previous);
    bcir_lt_rows(&o, data, previous, len - previous, width);
  }
  return finish(&o);
}

/* --- the payload views ------------------------------------------------------------------- */

bcir_ab_status bcir_ab_text_listing(const uint8_t *text, size_t len, bcir_sink_fn sink, void *ctx) {
  if (!sink || (!text && len)) return BCIR_AB_ERR_INVALID;
  bcir_lt_out o;
  bcir_lt_open(&o, sink, ctx);
  bcir_lt_put(&o, (const char *)text, len);
  if (!len || text[len - 1u] != '\n') bcir_lt_c(&o, '\n');
  return finish(&o);
}

/* One code point of well-formed UTF-8 at text[*at], advancing *at. */
static uint32_t next_cp(const uint8_t *text, size_t len, size_t *at) {
  uint32_t cp = text[(*at)++];
  size_t extra = cp < 0x80u ? 0u : cp < 0xE0u ? 1u : cp < 0xF0u ? 2u : 3u;
  if (extra) cp &= extra == 1u ? 0x1Fu : extra == 2u ? 0x0Fu : 0x07u;
  for (size_t k = 0; k < extra && *at < len; k++) cp = (cp << 6) | (uint32_t)(text[(*at)++] & 0x3Fu);
  return cp;
}

/* str.isspace(): the same 29 code points on every interpreter the package declares (3.11 to 3.15). */
static int py_space(uint32_t cp) {
  return (cp >= 0x09u && cp <= 0x0Du) || (cp >= 0x1Cu && cp <= 0x20u) || cp == 0x85u ||
         cp == 0xA0u || cp == 0x1680u || (cp >= 0x2000u && cp <= 0x200Au) || cp == 0x2028u ||
         cp == 0x2029u || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u;
}

/* str.splitlines()'s line boundaries ("\r\n" is one, handled by the caller). */
static int py_break(uint32_t cp) {
  return (cp >= 0x0Au && cp <= 0x0Du) || (cp >= 0x1Cu && cp <= 0x1Eu) || cp == 0x85u ||
         cp == 0x2028u || cp == 0x2029u;
}

/* The first word of one line text[start, end), when the stripped line is no comment. */
static void line_mnemonic(bcir_lt_out *o, const uint8_t *text, size_t start, size_t end) {
  size_t at = start, word = end, word_end = end;
  while (at < end) {                      /* the leading edge: str.strip()'s left half */
    size_t here = at;
    if (!py_space(next_cp(text, end, &at))) { word = here; break; }
  }
  if (word == end) return;                /* an empty or all-whitespace line */
  const char *w = (const char *)text + word;
  size_t room = end - word;
  if (w[0] == '#' || w[0] == ';' || w[0] == '.' || (room >= 2u && w[0] == '/' && w[1] == '/'))
    return;
  at = word;
  while (at < end) {                      /* the word: up to the first whitespace, str.split() */
    size_t here = at;
    if (py_space(next_cp(text, end, &at))) { word_end = here; break; }
  }
  bcir_lt_put(o, w, word_end - word);
  bcir_lt_c(o, '\n');
}

bcir_ab_status bcir_ab_text_mnemonics(const uint8_t *text, size_t len, bcir_sink_fn sink,
                                      void *ctx) {
  if (!sink || (!text && len)) return BCIR_AB_ERR_INVALID;
  bcir_lt_out o;
  bcir_lt_open(&o, sink, ctx);
  size_t start = 0, at = 0;
  while (at < len && !o.failed) {
    size_t here = at;
    uint32_t cp = next_cp(text, len, &at);
    if (!py_break(cp)) continue;
    line_mnemonic(&o, text, start, here);
    if (cp == '\r' && at < len && text[at] == '\n') at++;
    start = at;
  }
  if (start < len && !o.failed) line_mnemonic(&o, text, start, len);  /* the last, unterminated */
  return finish(&o);
}

static int opcode_line(const bcir_segment_view *seg, void *ctx) {
  bcir_lt_out *o = (bcir_lt_out *)ctx;
  bcir_lt_put(o, seg->opcode, seg->opcode_len);
  bcir_lt_c(o, '\n');
  return o->failed;
}

bcir_ab_status bcir_ab_pack_mnemonics(const uint8_t *pack, size_t len, bcir_sink_fn sink,
                                      void *ctx) {
  if (!sink) return BCIR_AB_ERR_INVALID;
  /* decode(payload)'s laws over the whole pack before the first opcode is offered */
  bcir_streampack_header h;
  if (bcir_sp_validate(pack, len, &h) != BCIR_OK ||
      bcir_sp_for_each_record(pack, len, 0, 0) != BCIR_OK)
    return BCIR_AB_ERR_PAYLOAD;
  bcir_lt_out o;
  bcir_lt_open(&o, sink, ctx);
  if (h.n_segments == 0u) {
    bcir_lt_c(&o, '\n');  /* "\n".join([]) + "\n" */
  } else if (bcir_sp_for_each_segment(pack, len, opcode_line, &o) != BCIR_OK) {
    return BCIR_AB_ERR_PAYLOAD;
  }
  return finish(&o);
}
