/* Static local tables (CF-STATICTAB): a `static` array, struct or union with a brace initializer -- the embedded
 * lookup table, CRC table and persistent counter. C initializes a static once, before the program runs, so both
 * rails fold its initializer to constants with the initializer walk (brace elision, designators, strings) and
 * emit it in the declaration -- `static T t[N] = {...};` -- never as stores at each call. Both had refused every
 * such table (the oracle: "static initializer is not a constant expression"; the twin: "non-constant enum
 * initializer"). The fold is C's own arithmetic, each operation in its type: `~0u` in a 64-bit table is
 * 0xFFFFFFFF, `-1` is all ones. Each function's statics persist across calls, so the harness's repeated calls
 * tell a table initialized once from one stored at each call. A static pointer or function pointer keeps its
 * storage class too (the twin had declared a static pointer as an uninitialized automatic one). */
#include <stdint.h>

struct stt_cfg { uint16_t a; uint8_t b; };
struct stt_pt { int16_t x, y; };
struct stt_in { uint8_t x, y; };
struct stt_out { struct stt_in a; uint16_t b; struct stt_in c[2]; };
union stt_u { uint32_t w; uint8_t b[4]; };
struct stt_bf { uint32_t a : 3, b : 5; int32_t c : 4; };
enum stt_color { STT_RED = 1, STT_GREEN = 4, STT_BLUE = 9 };
struct stt_node { uint32_t v; struct stt_node *next; };
typedef uint32_t (*stt_op)(uint32_t);

uint32_t stt_gv[4] = {9u, 8u, 7u, 6u};
uint32_t stt_words[2] = {11u, 13u};
struct stt_node stt_na;
struct stt_node stt_nb;

static uint32_t stt_twice(uint32_t x) { return x * 2u; }
static uint32_t stt_thrice(uint32_t x) { return x * 3u; }

uint32_t stt_crc_step(uint32_t i) {         /* a lookup table */
  static const uint8_t tbl[4] = {3u, 5u, 7u, 11u};
  return tbl[i % 4u];
}
uint32_t stt_counter(uint32_t i) {          /* partially initialized, and persistent: the rest is zero */
  static uint32_t hist[3] = {1u, 2u};
  hist[i % 3u] += i;
  return hist[0] + hist[1] * 3u + hist[2] * 5u;
}
uint32_t stt_conf(uint32_t i) {             /* a struct */
  static const struct stt_cfg c = {7, 9};
  return c.a + c.b + i;
}
uint32_t stt_const_first(uint32_t i) {      /* `const static`: static in any specifier position */
  const static uint16_t t[3] = {100u, 200u, 300u};
  return t[i % 3u];
}
uint32_t stt_inferred(uint32_t i) {         /* `[]` takes its extent from the list */
  static uint16_t t[] = {10, 20, 30, 40, 50};
  t[i % 5u] = (uint16_t)(t[i % 5u] + 1u);
  return t[i % 5u] + (uint32_t)sizeof t;
}
uint32_t stt_designated(uint32_t i) {       /* designators, the entry after one continuing past it */
  static const uint8_t t[6] = {[2] = 5, 7, [5] = 9};
  return t[i % 6u] * 3u + t[3];
}
uint32_t stt_elided(uint32_t i) {           /* brace elision into the rows of a 2-D table */
  static const uint8_t m[2][2] = {1, 2, 3, 4};
  return m[i % 2u][(i / 2u) % 2u] + m[1][1];
}
uint32_t stt_rows(uint32_t i) {             /* braced rows, a designated one, and a row count taken from the list */
  static uint8_t m[][3] = {{1, 2, 3}, [2] = {7, 8}, {10}};
  m[i % 4u][i % 3u] += 1u;
  return m[i % 4u][i % 3u] + (uint32_t)sizeof m;
}
uint32_t stt_string(uint32_t i) {           /* a string literal sizes and fills a character table */
  static const char s[] = "abc";
  return (uint32_t)s[i % 4u] + (uint32_t)sizeof s;
}
uint32_t stt_nested(uint32_t i) {           /* a struct of structs, with nested braces */
  static struct stt_out o = {{1, 2}, 300, {{3, 4}, {5, 6}}};
  o.c[i % 2u].y = (uint8_t)(o.c[i % 2u].y + i);
  return o.a.x + o.a.y + o.b + o.c[0].y + o.c[1].y * 3u;
}
uint32_t stt_points(uint32_t i) {           /* an array of structs with negative members, the last one zero */
  static struct stt_pt pts[3] = {{1, -2}, {3, -4}};
  pts[i % 3u].x = (int16_t)(pts[i % 3u].x + 1);
  return (uint32_t)(pts[i % 3u].x * 10 + pts[i % 3u].y);
}
uint32_t stt_union(uint32_t i) {            /* a union by its first member, and one by a designated other */
  static const union stt_u v = {0x01020304u};
  static union stt_u d = {.b = {9, 8}};
  d.b[i % 4u] = (uint8_t)(d.b[i % 4u] + 1u);
  return v.w + d.b[0] + d.b[1] * 3u + d.b[2] * 5u + d.b[3] * 7u;
}
uint32_t stt_negatives(uint32_t i) {        /* negative constants in narrow and 64-bit signed tables */
  static const int8_t t[4] = {-1, -128, 127, -5};
  static int64_t w[2] = {-5, -9223372036854775807 - 1};
  w[0] -= (int64_t)i;
  return (uint32_t)(int32_t)t[i % 4u] + (uint32_t)((uint64_t)w[0] >> 32) + (uint32_t)((uint64_t)w[1] >> 60);
}
uint32_t stt_constexpr(uint32_t i) {        /* constant expressions: a shift, sizeof, enumerators, a cast, ~0u */
  static const uint32_t t[6] = {1u << 3, sizeof(uint32_t), STT_BLUE, STT_GREEN * 2 + 1, (uint8_t)300, ~0u};
  return t[i % 6u];
}
uint32_t stt_wide(uint32_t i) {             /* a 64-bit table takes C's types: ~0u is 32 bits wide, -1 all 64 */
  static const uint64_t t[3] = {~0u, -1, 1ull << 40};
  return (uint32_t)(t[i % 3u] >> 8) ^ (uint32_t)(t[i % 3u] >> 40);
}
uint32_t stt_bitfields(uint32_t i) {        /* bit-fields, a signed one negative */
  static const struct stt_bf s = {5, 17, -3};
  return s.a + s.b * 10u + (uint32_t)(s.c + 8) * 100u + i;
}
uint32_t stt_flags(uint32_t i) {            /* _Bool elements normalize: 2 is 1 */
  static const _Bool t[3] = {2, 0, 1};
  return t[i % 3u] + t[0] * 2u;
}
uint32_t stt_floats(uint32_t i) {           /* a float table of integer constants */
  static float t[3] = {1, 2, -3};
  t[i % 3u] += 0.5f;
  return (uint32_t)(int32_t)(t[i % 3u] * 4.0f);
}
uint32_t stt_colors(uint32_t i) {           /* enumerators in an enum-typed table */
  static const enum stt_color pal[3] = {STT_BLUE, STT_RED, STT_GREEN};
  return (uint32_t)pal[i % 3u];
}
uint32_t stt_scalars(uint32_t i) {          /* a scalar static takes the same fold: C's own conversions */
  static int64_t n = -5;
  static unsigned u = -1;
  static uint32_t s = sizeof(uint16_t) * 3u;
  static uint32_t b = {7u};
  n -= (int64_t)i;
  u -= i;
  s += i;
  b ^= i;
  return (uint32_t)((uint64_t)n >> 32) + u + s * 3u + b * 5u;
}
uint32_t stt_in_loop(uint32_t i) {          /* declared in a loop body: initialized once, not per iteration */
  uint32_t s = 0;
  for (uint32_t k = 0; k < 3u; k++) {
    static uint32_t hits[3] = {10, 20, 30};
    hits[k] += i;
    s += hits[k];
  }
  return s;
}
uint32_t stt_two_scopes(uint32_t i) {       /* two scopes' statics of one name are two objects */
  uint32_t r;
  if (i & 1u) {
    static uint32_t n = 5;
    n += i;
    r = n;
  } else {
    static uint32_t n = 7;
    n += i;
    r = n * 3u;
  }
  return r;
}
uint32_t stt_pointer(uint32_t s) {          /* a static pointer persists: it steps through the table call by call */
  static uint32_t *sp = 0;
  static uint32_t *sq;
  if (sp == 0) sp = stt_gv; else sp += 1;
  if (sp == stt_gv + 4) sp = stt_gv;
  return *sp + (sq ? 1u : 0u) + s;
}
uint32_t stt_struct_pointer(uint32_t s) {   /* a static pointer to a struct, alternating between two */
  static struct stt_node *np;
  stt_na.v = 7u;
  stt_nb.v = 5u;
  if (np == &stt_na) np = &stt_nb; else np = &stt_na;
  return np->v + s;
}
uint32_t stt_void_pointer(uint32_t s) {     /* a static `void *` */
  static void *vp;
  uint32_t *q = (uint32_t *)vp;
  if (q == 0) q = stt_words; else q += 1;
  if (q == stt_words + 2) q = stt_words;
  vp = q;
  return *q + s;
}
uint32_t stt_function_pointer(uint32_t s) { /* static function pointers, a typedef'd one and a declarator's:
                                              * which function each holds alternates call by call */
  static stt_op fp = 0;
  static uint32_t (*fq)(uint32_t);
  stt_op n = stt_twice;
  if (fp == n) n = stt_thrice;
  fp = n;
  if (fq != n) fq = n; else fq = stt_twice;
  return (fp == stt_twice) * 3u + (fq == stt_thrice) * 5u + s;
}
uint32_t stt_entry(uint32_t i) {
  uint32_t r = stt_crc_step(i);
  r += stt_counter(i) * 3u;
  r += stt_conf(i) * 5u;
  r += stt_const_first(i) * 7u;
  r += stt_inferred(i) * 11u;
  r += stt_designated(i) * 13u;
  r += stt_elided(i) * 17u;
  r += stt_rows(i) * 19u;
  r += stt_string(i) * 23u;
  r += stt_nested(i) * 29u;
  r += stt_points(i) * 31u;
  r += stt_union(i) * 37u;
  r += stt_negatives(i) * 41u;
  r += stt_constexpr(i) * 43u;
  r += stt_wide(i) * 47u;
  r += stt_bitfields(i) * 53u;
  r += stt_flags(i) * 59u;
  r += stt_floats(i) * 61u;
  r += stt_colors(i) * 67u;
  r += stt_scalars(i) * 71u;
  r += stt_in_loop(i) * 73u;
  r += stt_two_scopes(i) * 79u;
  r += stt_pointer(i) * 83u;
  r += stt_struct_pointer(i) * 89u;
  r += stt_void_pointer(i) * 97u;
  r += stt_function_pointer(i) * 101u;
  return r;
}
