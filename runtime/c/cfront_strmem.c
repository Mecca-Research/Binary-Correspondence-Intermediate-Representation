/* <string.h> memory routines as external libc edges (CF-RTWIDE): `memcpy`, `memmove` and `memset` each lower on
 * both rails to one `c.call.libm:` edge -- verbatim, opaque to R18, linked from libc with no flag -- where both
 * rails had refused the unit as a call to an undefined function. The emit spells every plain memory access as a
 * `memcpy`, so a re-parsed emit calls it. Each routine returns its destination. The objects are structs, read
 * by member, so the unit holds no bounds-guarded access and its emit is in the round trip's re-parsed set. */
#include <stdint.h>
#include <string.h>

struct sm_rec { uint32_t id; uint16_t w[3]; uint8_t tag; };
struct sm_buf { uint8_t b[8]; };
struct sm_words { uint32_t v[4]; };

uint32_t sm_copy(uint32_t s) {                      /* whole structs, byte for byte */
  struct sm_rec a, b;
  struct sm_words src, dst;
  a.id = s; a.w[0] = (uint16_t)s; a.w[1] = 7u; a.w[2] = (uint16_t)(s >> 3); a.tag = (uint8_t)(s ^ 0x5Au);
  src.v[0] = s; src.v[1] = s * 3u; src.v[2] = s ^ 9u; src.v[3] = s >> 2;
  memcpy(&b, &a, sizeof b);
  memcpy(&dst, &src, sizeof dst);
  return b.id + b.w[0] + b.w[1] + b.w[2] + b.tag + dst.v[0] + dst.v[3];
}

uint32_t sm_move(uint32_t s) {                      /* overlapping ranges: memmove reads before it writes */
  struct sm_buf r;
  r.b[0] = (uint8_t)s; r.b[1] = (uint8_t)(s + 1u); r.b[2] = (uint8_t)(s + 2u); r.b[3] = (uint8_t)(s + 3u);
  r.b[4] = (uint8_t)(s + 4u); r.b[5] = (uint8_t)(s + 5u); r.b[6] = (uint8_t)(s + 6u); r.b[7] = (uint8_t)(s + 7u);
  memmove(&r.b[2], &r.b[0], 5);
  memmove(&r.b[0], &r.b[3], 4);
  return r.b[0] + 3u * r.b[2] + 5u * r.b[6] + 7u * r.b[7];
}

uint32_t sm_fill(uint32_t s) {                      /* memset: every byte the value's low eight bits */
  struct sm_words w;
  struct sm_rec r;
  memset(&w, (int)(s & 0xFFu), sizeof w);
  memset(&r, 0, sizeof r);
  r.tag = (uint8_t)s;
  return w.v[0] ^ w.v[2] ^ r.id ^ r.w[1] ^ r.tag;
}

uint32_t sm_result(uint32_t s) {                    /* each routine returns its destination */
  uint32_t a = s, b = 0;
  struct sm_buf c;
  uint32_t *p = memcpy(&b, &a, sizeof b);
  uint8_t *q = memset(&c, 3, sizeof c);
  return *p + q[2] + (uint32_t)(p == &b);
}

uint32_t sm_entry(uint32_t s) { return sm_copy(s) + sm_move(s) + sm_fill(s) + sm_result(s); }
