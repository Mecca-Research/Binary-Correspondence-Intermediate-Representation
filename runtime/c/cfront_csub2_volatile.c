/* CSUB-2e: volatile objects stepped, and reached through a pointer sum -- `bcir_quarantine.h`'s counter,
 * `return (*c)++;` of a `volatile unsigned long *` -- behaviour-checked on both rails. A step of a volatile scalar
 * is one read and one write: through a pointer (`(*c)++`, `++*c`, `c[k]++`, `*(c + k)`), of a volatile array's
 * element, of a volatile member or a plain one of a device struct, of a volatile struct's member, member array
 * element or array element, and of a named volatile object. A pointer sum of a pointer to volatile storage is a
 * pointer to volatile storage, read and stored through. The storage is block statics, set at each call: the
 * harness runs the original and then the emit, and the escape corpus holds every local array private. */
#include <stdint.h>

typedef volatile unsigned long ctr;
/* `bcir_quarantine.h`'s counter, as written: the slot an event owns, and the total advanced by one */
static inline unsigned long next(ctr *c) { return (*c)++; }

struct regs {
  volatile uint32_t status;
  uint32_t shadow;
  volatile uint16_t ring[4];
};
struct inner {
  uint32_t x;
};
struct cell {
  uint32_t n;
  uint32_t a[2];
  struct inner in;
};

/* through a pointer to volatile storage: postfix, prefix, an element, a pointer sum */
static uint32_t through(volatile uint32_t *c, uint32_t k) {
  uint32_t r = (*c)++;
  r += ++*c;
  r += c[k & 3u]++;
  r += ++c[(k + 1u) & 3u];
  r += (*(c + (k & 3u)))--;
  --*(c + 3u - (k & 3u));
  *(c + 1u + (k & 1u)) = r & 0xffu;
  r += *(k % 3u + c);
  volatile uint32_t *q = c + (k & 1u);
  *q += 3u;
  return r + *q;
}

/* a device struct's members, and a volatile struct's */
static uint32_t members(struct regs *p, volatile struct cell *q, uint32_t k) {
  uint32_t r = p->status++;
  r += ++p->shadow;
  p->shadow--;
  r += p->ring[k & 3u]++;
  r += q->n++;
  r += q->a[k & 1u]++;
  r += ++q->in.x;
  r += q[1].n--;
  r += q[1].a[1]++;
  return r + p->status + p->shadow + p->ring[k & 3u];
}

/* named volatile objects */
static uint32_t named(uint32_t k) {
  static volatile uint32_t ticks;
  static volatile uint8_t small[4];
  ticks = k;
  for (uint32_t i = 0; i < 4u; i++) small[i] = (uint8_t)(k + i * 77u);
  ticks++;
  --ticks;
  ticks += 5u;
  uint32_t r = small[k & 3u]++;
  small[1]--;
  r += ticks++;
  return r + ticks + small[1] + small[k & 3u];
}

uint32_t csub2_volatile(uint32_t x, uint32_t y) {
  static volatile uint32_t w[4];
  static ctr counter;
  static struct regs dev;
  static volatile struct cell cells[2];
  for (uint32_t i = 0; i < 4u; i++) {
    w[i] = x + i * y;
    dev.ring[i] = (uint16_t)(x * 3u + i);
  }
  counter = x;
  dev.status = y;
  dev.shadow = x ^ y;
  for (uint32_t i = 0; i < 2u; i++) {
    cells[i].n = x + i;
    cells[i].a[0] = y + i;
    cells[i].a[1] = x * y + i;
    cells[i].in.x = y ^ i;
  }
  uint32_t r = through(w, y);
  unsigned long s0 = next(&counter), s1 = next(&counter);
  r += (uint32_t)s0 + (uint32_t)s1 * 3u;
  r += members(&dev, cells, x);
  r += named(x ^ y);
  return r + w[0] * 7u + w[3] + (uint32_t)counter;
}
