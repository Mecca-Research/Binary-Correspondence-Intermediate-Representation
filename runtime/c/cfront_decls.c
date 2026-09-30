/* Declaring functions faithfully (CF-DECLS): callees defined after their callers -- `static` or not -- declared
 * first by a prototype, as C99 requires (C11 6.5.1p2), prototypes leaving parameters unnamed (a pointer to
 * `const`, an array, a function pointer), a function named as a value and a call in a `sizeof` operand before the
 * callee's definition. Each rail's emit declares the unit's functions a function calls ahead of it, so every emit
 * compiles and runs as the original with no declaration supplied by hand. A call made before any declaration of
 * its callee is refused on both rails (the test's inline units); `cfront_decls_link.c` holds the callees another
 * unit defines. */
#include <stdint.h>

static uint32_t dc_later(const uint32_t *p, uint32_t s);         /* static, defined after its caller */
uint32_t dc_wide(uint32_t s);                                     /* external linkage, defined after its caller */
static uint64_t dc_big(const uint32_t *, uint32_t);               /* unnamed parameters, a 64-bit result */
static uint32_t dc_apply(uint32_t (*)(uint32_t), uint32_t);       /* an unnamed function-pointer parameter */
static uint32_t dc_sum(const uint32_t [], uint32_t n);            /* an unnamed array parameter */
static uint32_t dc_twice(uint32_t v);                             /* named as a value before its definition */

static uint32_t dc_tab[3] = {5u, 7u, 11u};

uint32_t dc_entry(uint32_t s) {                                   /* every callee is defined below it */
  dc_tab[0] = s;
  uint64_t b = dc_big(dc_tab, s);
  return dc_later(dc_tab, s) + dc_wide(s) + (uint32_t)(b >> 7) + dc_apply(dc_twice, s) + dc_sum(dc_tab, 3u)
         + (uint32_t)sizeof(dc_big(dc_tab, s));
}

static uint32_t dc_later(const uint32_t *p, uint32_t s) {
  if (p) return p[0] + s * 3u;
  return s ^ 0x3Cu;
}
uint32_t dc_wide(uint32_t s) { return dc_later(0, s ^ 0x55u) + 1u; }
static uint64_t dc_big(const uint32_t *p, uint32_t s) { return ((uint64_t)p[1] << 32) | (uint64_t)(s + p[2]); }
static uint32_t dc_apply(uint32_t (*fn)(uint32_t), uint32_t s) { return fn(s) + fn(s + 1u); }
static uint32_t dc_sum(const uint32_t a[], uint32_t n) {
  uint32_t k = 0;
  for (uint32_t i = 0; i < n; i++) k += a[i];
  return k;
}
static uint32_t dc_twice(uint32_t v) { return v * 2u + 1u; }
