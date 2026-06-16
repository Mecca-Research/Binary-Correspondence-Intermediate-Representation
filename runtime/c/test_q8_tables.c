/*===- test_q8_tables.c - check the frozen Q8 tier table (embed or fallback) ===
 *
 * Includes the GENERATED bcir_q8_tables.h (#embed-baked on C23 toolchains that
 * support it, identical fallback bytes otherwise) and asserts the factors match the
 * oracle (bcir/kbcir/cost.py MemoryHierarchy.default()). Prints "OK q8 embed=<0|1>".
 *
 *   clang -std=c23 -I runtime/c runtime/c/test_q8_tables.c -o t && ./t
 *===----------------------------------------------------------------------===*/
#include <stdio.h>

#include "bcir_q8_tables.h"

int main(void) {
  struct { int tier; uint16_t bw, lat; } want[] = {
    {BCIR_Q8_L1, 16, 16},     {BCIR_Q8_L2, 32, 48},   {BCIR_Q8_L3, 96, 96},
    {BCIR_Q8_DRAM, 256, 256}, {BCIR_Q8_HBM, 64, 192}, {BCIR_Q8_CXL, 384, 512},
    {BCIR_Q8_SSD, 1024, 4096},
  };
  for (unsigned i = 0; i < sizeof want / sizeof want[0]; ++i) {
    uint16_t bw = bcir_q8_tier_bw(want[i].tier), lat = bcir_q8_tier_lat(want[i].tier);
    if (bw != want[i].bw || lat != want[i].lat) {
      printf("FAIL tier=%d bw=%u/%u lat=%u/%u\n", want[i].tier, bw, want[i].bw, lat, want[i].lat);
      return 1;
    }
  }
  /* DRAM is the x1.0 baseline; HBM bandwidth is cheaper (lower factor) than DRAM. */
  if (bcir_q8_tier_bw(BCIR_Q8_DRAM) != BCIR_Q8_ONE) return 1;
  if (!(bcir_q8_tier_bw(BCIR_Q8_HBM) < bcir_q8_tier_bw(BCIR_Q8_DRAM))) return 1;
  printf("OK q8 embed=%d ntiers=%d\n", BCIR_Q8_EMBED, BCIR_Q8_NTIERS);
  return 0;
}
