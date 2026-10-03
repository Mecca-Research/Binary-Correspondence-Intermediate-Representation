/* Forms the two rails lowered to different claim graphs (CF-RAILSPLIT): a logical not (a value, a condition, a
 * status-register polling loop -- the twin gave `!` another opcode than the oracle's `_UN`), a bare dereference
 * statement (the twin kept its store probe's claims before lowering the statement again), the address of an
 * element of an allocated buffer (the twin counted `&p[i]` as taking `p`'s address, so it did not recover the
 * allocation's extent), and a VLA of volatile elements (the twin laid it out as ordinary memory and failed R3).
 * Both rails lower each the same way, and each emit returns what the original does. */
#include <stdint.h>
#include <stdlib.h>

uint32_t rs_not(uint32_t x) {                       /* `!` as a value, twice, and in arithmetic */
  uint32_t a = !x;
  uint32_t b = !!x;
  return a + b * 2u + (!(x & 4u)) * 4u;
}
uint32_t rs_if_not(uint32_t x) {                    /* `!` as a condition */
  if (!x) return 7u;
  if (!(x & 1u)) return 3u;
  return 1u;
}
uint32_t rs_poll(uint32_t s) {                      /* a status poll: loop while the ready bit is clear */
  volatile uint32_t st[4] = {s & 6u, s & 2u, s | 1u, 1u};
  uint32_t n = 0u;
  while (!(st[n & 3u] & 1u)) {
    n++;
    if (n > 5u) break;
  }
  return n;
}
uint32_t rs_bare_deref(uint32_t s) {                /* a dereference read and discarded, as a statement */
  volatile uint32_t r[2] = {s, s + 1u};
  *(r + 1);
  *r;
  return r[0] + r[1];
}
uint32_t rs_elem_addr(uint32_t n) {                 /* the address of an element of an allocated buffer */
  uint32_t *p = malloc(4u * ((n & 7u) + 2u));
  if (p == 0) return 0u;
  uint32_t *q = &p[1];
  *q = n ^ 7u;
  p[0] = 1u;
  uint32_t r = p[0] + p[1];
  free(p);
  return r;
}
uint32_t rs_vla_volatile(uint32_t n) {              /* a VLA of volatile elements, one- and two-dimensional */
  uint32_t k = (n & 3u) + 2u;
  volatile uint32_t va[k];
  volatile uint16_t vm[k][3];
  va[1] = n;
  vm[1][2] = (uint16_t)(n + 5u);
  return va[1] + vm[1][2] * 3u;
}
uint32_t rs_entry(uint32_t s) {
  return rs_not(s) + rs_if_not(s) * 3u + rs_poll(s) * 5u + rs_bare_deref(s) * 7u + rs_elem_addr(s) * 11u +
         rs_vla_volatile(s) * 13u;
}
