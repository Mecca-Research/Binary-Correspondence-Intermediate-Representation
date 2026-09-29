/* `__attribute__((packed))` written AFTER a struct's closing brace (CF-TRAILPACK) -- the common spelling. The
 * C twin read it only there, for the aggregate's alignment, after it had already placed every member at its
 * natural offset: `struct Pk { uint8_t c; uint64_t n; } __attribute__((packed));` put `n` at 8 and folded
 * `sizeof` to 16 where Clang and the oracle say 1 and 9, so every access to it read and wrote the wrong
 * bytes. The attributes after the `}` are now read before the members are laid out; a trailing `aligned(N)`
 * raises the aggregate's alignment, as it did.
 *
 * Every member is written before it is read, so no function depends on the harness's random fill. */
#include <stdint.h>

struct In { uint8_t a; uint32_t b; } __attribute__((packed));
struct Pk {
  uint8_t c;
  uint64_t n;                      /* misaligned: offset 1 */
  uint32_t lo : 3;
  uint32_t hi : 30;                /* straddles a byte boundary, packed bit by bit */
  struct In in;                    /* a nested packed struct: 5 bytes, alignment 1 */
  uint16_t arr[3];                 /* an array member at an odd offset */
} __attribute__((packed));
struct Lead { uint8_t c; uint64_t n; };                     /* unpacked, for contrast */
struct __attribute__((packed)) Pl { uint8_t c; uint64_t n; };   /* the leading spelling, which already worked */
struct Wide { uint8_t c; uint32_t n; } __attribute__((packed, aligned(4)));
typedef struct TPt { uint8_t c; uint32_t n; uint16_t h; } __attribute__((packed)) TP;   /* tagged: the harness
                                                                                 * names a parameter's struct */

uint64_t tp_members(struct Pk *p, uint64_t v, uint32_t i) {
  p->c = 1u;
  p->n = v;
  p->lo = 5u;
  p->hi = (uint32_t)v;
  p->in.a = 2u;
  p->in.b = (uint32_t)(v >> 7);
  p->arr[i % 3u] = (uint16_t)v;
  return p->n + p->c + p->lo + p->hi + p->in.a + p->in.b + p->arr[i % 3u];
}

uint32_t tp_sizes(void) {
  return (uint32_t)sizeof(struct Pk) + 100u * (uint32_t)sizeof(struct Lead) + 10000u * (uint32_t)sizeof(struct Pl)
         + 1000000u * (uint32_t)sizeof(struct Wide);
}

uint32_t tp_typedef(TP *t, uint32_t v) {
  t->c = 3u;
  t->n = v;                        /* offset 1 */
  t->h = (uint16_t)(v >> 3);       /* offset 5 */
  return t->n + t->c + t->h + (uint32_t)sizeof(TP) + 10u * (uint32_t)_Alignof(TP);
}

/* The harness runs the unit's LAST function against the original. */
uint64_t tp_all(struct Pk *p, TP *t, uint64_t v, uint32_t i) {
  uint64_t r = tp_members(p, v, i);
  r += tp_sizes();
  r += tp_typedef(t, (uint32_t)v);
  return r;
}
