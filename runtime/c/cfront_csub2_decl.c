/* CSUB-2a: two declarations the runtime's units write, behaviour-checked on both rails.
 * - A function-pointer type whose parameter is declared as an array, as `bcir_shard_manifest.h`'s fetch callback
 *   declares `const uint8_t digest[32]`: the parameter is a pointer to the element (C11 6.7.6.3p7), as a definition's
 *   array parameter is -- in a typedef, in a member and in a local's inline declarator, sized or not.
 * - `_Alignas(N)` / `alignas(N)` on an object (C11 6.7.5), as `bcir_tensor.c` aligns `_Alignas(64) float bt[...]`:
 *   locals, several declarators off one specifier, the strictest of two specifiers, a block static, and file-scope
 *   objects of every storage class. The alignment is the object's, and each rail's emit keeps it: each function
 *   reads the address of what it aligns, which a dropped specifier would leave on the stack's 16-byte boundary. */
#include <stdint.h>
#include <stddef.h>
#include <stdalign.h>

typedef uint32_t (*digest_fn)(const uint8_t digest[8], uint32_t n);
typedef uint32_t (*rows_fn)(const uint32_t v[], uint32_t n);
struct fetcher {
  digest_fn fetch;
  uint32_t (*fold)(uint32_t seed, const uint8_t block[4]);
};

static uint32_t sum_digest(const uint8_t digest[8], uint32_t n) {
  uint32_t s = n;
  for (uint32_t i = 0; i < 8u; i++) s = s * 31u + digest[i];
  return s;
}
static uint32_t sum_rows(const uint32_t v[], uint32_t n) {
  uint32_t s = 0;
  for (uint32_t i = 0; i < n; i++) s += v[i] ^ i;
  return s;
}
static uint32_t fold4(uint32_t seed, const uint8_t block[4]) {
  return seed ^ ((uint32_t)block[0] | (uint32_t)block[1] << 8 | (uint32_t)block[2] << 16 | (uint32_t)block[3] << 24);
}

uint32_t csub2_fparray(uint32_t x) {
  /* static: an array handed to a call through a function pointer leaves its activation, and the escape corpus
   * (`escape_fixtures`) holds only local arrays the analysis proves private */
  static uint8_t d[8];
  static uint32_t v[3];
  for (uint32_t i = 0; i < 8u; i++) d[i] = (uint8_t)(x + i * 7u);
  v[0] = x;
  v[1] = x + 1u;
  v[2] = x * 3u;
  digest_fn f = sum_digest;
  rows_fn g = sum_rows;
  uint32_t (*h)(uint32_t seed, const uint8_t block[]) = fold4;
  /* the member's declarator is read as the typedef's is: two pointers, laid out as Clang lays them out */
  return f(d, x) + g(v, 3u) + h(x, d) + h(x, d + 4) + (uint32_t)sizeof(struct fetcher);
}

static _Alignas(4096) uint8_t g_page[64] = {1, 2, 3};
_Alignas(256) const uint32_t g_words[4] = {5, 6, 7, 8};
alignas(512) static const uint16_t g_halves[2] = {7, 9};

static uint32_t aligned_to(const void *p, uintptr_t n) { return (uint32_t)(((uintptr_t)p & (n - 1u)) == 0u); }

struct blk {
  uint8_t b[48];
};

uint32_t csub2_align_local(uint32_t x) {
  /* a struct and scalars, whose addresses the witness reads: a local array whose address leaves its function is
   * no private one, which the escape corpus holds every local array to be (the emit spells an array's alignment
   * as it spells these: `test_alignment_specifiers_and_array_parameters_of_function_types_on_both_rails`) */
  _Alignas(256) struct blk buf;
  _Alignas(512) uint32_t w = x, z = x + 1u;
  alignas(16) alignas(1024) uint64_t two = (uint64_t)x * 3u;
  for (uint32_t i = 0; i < 48u; i++) buf.b[i] = (uint8_t)(x + i);
  return buf.b[7] + w + z + (uint32_t)two + aligned_to(&buf, 256u) * 1000u + aligned_to(&w, 512u) * 2000u +
         aligned_to(&z, 512u) * 4000u + aligned_to(&two, 1024u) * 8000u;
}

uint32_t csub2_align_static(uint32_t x) {
  static _Alignas(2048) uint32_t calls;
  calls += 1u;   /* each rail's function its own static: the harness calls the original, then the emit */
  return calls + g_page[x % 3u] + g_halves[x & 1u] + g_words[x & 3u] + aligned_to(&calls, 2048u) * 100u +
         aligned_to(g_page, 4096u) * 200u + aligned_to(g_words, 256u) * 400u + aligned_to(g_halves, 512u) * 800u;
}

/* The entry the harness drives: every function above, folded. */
uint32_t csub2_decl(uint32_t x, uint32_t y) {
  return csub2_fparray(x) * 3u + csub2_align_local(y) * 5u + csub2_align_static(x ^ y) * 7u;
}
