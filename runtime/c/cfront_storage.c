/* Declaration specifiers in any order (C11 6.7p1, CF-STORAGE): `static` is a static wherever it is written
 * -- first, after the type (`uint32_t static n`), after a qualifier (`volatile static`), after a typedef name,
 * a struct tag or an `unsigned` run -- and a qualifier after the type qualifies it as it does before one.
 * (Both rails had kept only a LEADING `static`: `volatile static uint32_t n` became an uninitialized local.
 * A block-scope `extern` is refused on both, never bound as a new local.) Each function's static persists
 * across calls, so the harness's repeated calls tell a static from a local. Both rails lower them the same
 * way, and each emit returns what the original does. */
#include <stdint.h>

struct st_pair { uint32_t a, b; };
typedef uint32_t st_word;

uint32_t st_first(uint32_t i) {                     /* the spelling that always worked */
  static uint32_t n;
  n += i;
  return n;
}
uint32_t st_after_type(uint32_t i) {                /* after the type */
  uint32_t static n;
  n += i;
  return n;
}
uint32_t st_after_qualifier(uint32_t i) {           /* after a qualifier */
  volatile static uint32_t n;
  n = n + i;
  return n;
}
uint32_t st_after_unsigned(uint32_t i) {            /* inside a keyword run, with an initializer */
  unsigned static n = 7u;
  n += i;
  return n;
}
uint32_t st_after_typedef(uint32_t i) {             /* after a typedef name */
  st_word static n = 3u;
  n ^= i;
  return n;
}
uint32_t st_array(uint32_t i) {                     /* a static array, spelled after its element type */
  uint8_t static a[4];
  a[i % 4u] += (uint8_t)i;
  return a[0] + a[1] * 3u + a[2] * 5u + a[3] * 7u;
}
uint32_t st_struct(uint32_t i) {                    /* after a struct tag */
  struct st_pair static p;
  p.a += i;
  p.b = p.a * 2u;
  return p.b;
}
uint32_t st_qualifiers(uint32_t i) {                /* const / volatile after the type */
  uint32_t const c = 3u;
  uint32_t volatile v = i;
  v = v + c;
  return v;
}
uint32_t st_pointer(uint32_t i) {                   /* `T volatile *p` points at volatile T */
  static uint32_t cell;
  uint32_t volatile *p = &cell;
  *p = *p + i;
  return cell;
}
uint32_t st_entry(uint32_t i) {
  uint32_t r = st_first(i);
  r += st_after_type(i) * 3u;
  r += st_after_qualifier(i) * 5u;
  r += st_after_unsigned(i) * 7u;
  r += st_after_typedef(i) * 11u;
  r += st_array(i) * 13u;
  r += st_struct(i) * 17u;
  r += st_qualifiers(i) * 19u;
  r += st_pointer(i) * 23u;
  return r;
}
