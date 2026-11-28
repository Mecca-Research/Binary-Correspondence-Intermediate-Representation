/* CSUB-2b: a block-scope static table of string literals, behaviour-checked on both rails.
 * A string literal is the address of its array -- an address constant (C11 6.6p9) -- and so an entry of a static's
 * image wherever C converts that address without a diagnostic: into a pointer to plain `char`, at any qualification,
 * from a literal of no prefix. The runtime's units name their enumerations this way (`bcir_artifact_bundle.c`'s
 * `names[]`, `bcir_decoder_pack.c`'s role and panel suffixes, `bcir_jer.c`'s literal names, `bcir_sp_listing.c`'s
 * headings). Both rails render each entry by one canonical spelling, whatever escapes and pieces the source spells
 * it with, so the image -- and the digest over it -- is the same on both. */
#include <stdint.h>
#include <stddef.h>

static uint32_t str_len(const char *s) {
  uint32_t n = 0;
  while (s[n]) n++;
  return n;
}
static uint32_t str_sum(const char *s) {
  uint32_t h = 2166136261u;
  for (uint32_t i = 0; s[i]; i++) h = (h ^ (uint32_t)(unsigned char)s[i]) * 16777619u;
  return h;
}

/* the runtime's shape: a `const char *const` table indexed by a code, a fallback for codes past it */
const char *csub2_name_of(uint32_t k) {
  static const char *const names[] = {"ok", "invalid", "size", "magic", "version", "layout"};
  return k < sizeof names / sizeof names[0] ? names[k] : "?";
}

uint32_t csub2_names(uint32_t k) {
  return str_len(csub2_name_of(k)) * 131u + str_sum(csub2_name_of(k % 7u));
}

/* every escape C spells, pieces, an empty literal, a `?` pair the emit must not spell as a trigraph (`-std=c11` reads
 * one), a null entry and an inferred size */
uint32_t csub2_escapes(uint32_t k) {
  static const char *const t[] = {"\a\b\f\n\r\t\v\x27\"\\\?\101\x41", "", "a" "b", "\x4" "1", "?\?=", "\1\12\123x",
                                  0, ("par")};
  const char *s = t[k % (sizeof t / sizeof t[0])];
  return s ? str_len(s) * 7u + str_sum(s) : 99u;
}

/* records of a string and a code, braced, by brace elision and by designators, with a gap */
struct kw {
  const char *text;
  uint32_t id;
};
uint32_t csub2_keywords(uint32_t k) {
  static const struct kw kws[] = {{"if", 1u}, {"else", 2u}, "while", 3u, [5] = {.id = 9u, .text = "for"}};
  const struct kw *e = &kws[k % 6u];
  return (e->text ? str_sum(e->text) : 5u) + e->id * 1000u;
}

/* a scalar static, two declarators, a thread-local one, a typedef'd pointer, a `char *` pointee, a later store over
 * an earlier one, a union's pointer member and a member beside a bit-field */
typedef const char *str_t;
union word {
  uint64_t u;
  const char *s;
};
struct tagged {
  uint32_t a : 4;
  const char *p;
};
uint32_t csub2_scalars(uint32_t k) {
  static const char *const p = "pea", *const q = "queue";
  static _Thread_local const char *tl = "thread";
  static const str_t tt[] = {"alpha", "beta"};
  static char *const mut[] = {"mutable"};
  static const char *last[2] = {[0] = "gone", [0] = 0, [1] = 0, [1] = "kept"};
  static union word un = {.s = "union"};
  static const struct tagged tg = {3u, "bits"};
  return str_sum(p) + str_len(q) + str_sum(tl) * 3u + str_sum(tt[k & 1u]) + str_len(mut[0]) +
         (last[0] == 0) * 11u + str_sum(last[1]) + str_sum(un.s) + tg.a + str_sum(tg.p);
}

/* The entry the harness drives: every function above, folded. */
uint32_t csub2_strtab(uint32_t x, uint32_t y) {
  return csub2_names(x) * 3u + csub2_escapes(y) * 5u + csub2_keywords(x ^ y) * 7u + csub2_scalars(y) * 11u;
}
