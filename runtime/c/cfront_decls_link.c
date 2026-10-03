/* Prototyped callees another unit defines (CF-DECLS): each emit declares them `extern` as the prototypes do -- a
 * pointer or array parameter to `const` keeps its `const`, an unnamed function-pointer parameter is spelled
 * `RET (*)(PARAMS)` -- where both rails dropped the `const`, a declaration that conflicts with the original's in one
 * translation unit, and the oracle spelled a function-pointer parameter by its name. The test's driver defines
 * the callees. */
#include <stdint.h>

uint32_t dl_read(const uint32_t p[], uint32_t (*)(uint32_t), uint32_t s);   /* a const array, an unnamed funcptr */
uint64_t dl_mix(const uint8_t *, uint32_t);                                   /* a const pointer, unnamed */

static uint32_t dl_twice(uint32_t v) { return v * 2u + 1u; }
static uint32_t dl_tab[3] = {5u, 7u, 11u};
static uint8_t dl_bytes[4] = {1u, 2u, 3u, 4u};

uint32_t dl_entry(uint32_t s) {
  dl_tab[1] = s;
  return dl_read(dl_tab, dl_twice, s) + (uint32_t)(dl_mix(dl_bytes, s) >> 3);
}
