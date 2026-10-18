/* BF16-0 (NDT-BF16): VCVTNE2PS2BF16 (two sources), the AMX host's pack conversion, against
 * bcir_tensor.c's bf16_of on every binary32, both source positions, and under MXCSR RZ+FTZ+DAZ on a
 * sample (README.json).
 *
 *   gcc -O2 cvt2.c && ./a.out */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static uint16_t bf16_of(uint32_t u) {
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (uint16_t)((u >> 16) | 0x0040u);
  if (!(u & 0x7f800000u)) return (uint16_t)((u >> 16) & 0x8000u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__attribute__((target("avx512f,avx512bf16"))) int main(void) {
  const __m512i iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  long bad = 0;
  for (uint64_t base = 0; base < (1ull << 32); base += 32) {
    __m512i lo = _mm512_add_epi32(_mm512_set1_epi32((int)(uint32_t)base), iota);
    __m512i hi = _mm512_add_epi32(lo, _mm512_set1_epi32(16));
    __m512bh h = _mm512_cvtne2ps_pbh(_mm512_castsi512_ps(hi), _mm512_castsi512_ps(lo));
    uint16_t out[32]; memcpy(out, &h, 64);
    for (int i = 0; i < 32; i++) bad += out[i] != bf16_of((uint32_t)base + (uint32_t)i);
  }
  long d = 0; unsigned int old = _mm_getcsr(); uint32_t s = 1;
  for (int k = 0; k < 1 << 20; k++) {
    s = s * 1664525u + 1013904223u;
    __m512 v = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_set1_epi32((int)s), iota));
    __m512bh h0 = _mm512_cvtne2ps_pbh(v, v);
    _mm_setcsr((old & ~0x6000u) | 0x6000u | 0x8040u);
    __m512bh h1 = _mm512_cvtne2ps_pbh(v, v);
    _mm_setcsr(old);
    d += memcmp(&h0, &h1, 64) != 0;
  }
  printf("VCVTNE2PS2BF16 vs bf16_of over all 2^32 inputs: %ld mismatches; under MXCSR RZ+FTZ+DAZ %ld of %d samples differ\n", bad, d, 1 << 20);
  return bad != 0;
}
