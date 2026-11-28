/* CSUB-2d: a pointer-to-array member, `T (*m)[N]` -- `bcir_cir.h`'s `char (*calls)[BCIR_CIR_NAME]`, the call table every
 * unit that includes it declares -- behaviour-checked on both rails. A pointer to a row of N scalars, laid out as a
 * pointer: its elements read, stored, stepped, compound-assigned and addressed by the row (`f->calls[k][j]`), the row
 * sized (`sizeof *f->calls`), the pointer tested for null, copied member to member, passed to a row-pointer parameter
 * and taken from a two-dimensional array; through a struct value, an element of a member array, a loaded pointer and a
 * union. The storage is block statics, set at each call: the harness runs the original and then the emit, and the
 * escape corpus holds every local array private. */
#include <stdint.h>
#include <stddef.h>

#define NAME 8

struct func {
  uint32_t n;
  char (*calls)[NAME];
  int n_calls;
  uint16_t (*w)[3];
  struct func *next;
};
struct unit {
  struct func funcs[2];
  struct func *cur;
};
union cell {
  uint16_t (*w)[3];
  uintptr_t raw;
};

static uint32_t rowsum(uint16_t (*r)[3], uint32_t k) { return r[k][0] + r[k][1] * 3u + r[k][2] * 5u; }

/* the R18 call table: a callee's name stored into its row a byte at a time, and found again */
static int put_call(struct func *f, const char *name) {
  if (!f->calls) return -1;
  int k = f->n_calls++;
  uint32_t j = 0;
  for (; j + 1u < NAME && name[j]; j++) f->calls[k][j] = name[j];
  for (; j < NAME; j++) f->calls[k][j] = 0;
  return k;
}
static int same(const struct func *f, int k, const char *name) {
  for (uint32_t j = 0; j < NAME; j++) {
    if (f->calls[k][j] != name[j]) return 0;
    if (!name[j]) return 1;
  }
  return 1;
}
static int find_call(const struct func *f, const char *name) {
  for (int k = 0; k < f->n_calls; k++)
    if (same(f, k, name)) return k;
  return -1;
}

/* an element stored, compound-assigned, stepped, assigned as a value and addressed, and the row passed on */
static uint32_t weights(struct func *f, uint32_t k, uint16_t v) {
  f->w[k][0] = v;
  f->w[k][1] += (uint16_t)(v * 3u);
  f->w[k][2]++;
  uint16_t a = ++f->w[k][2];
  uint16_t b = (f->w[k][1] -= 2u);
  uint16_t *p = &f->w[k][1];
  *p ^= 0x55u;
  return a + b * 7u + rowsum(f->w, k) + (uint32_t)(&f->w[k][2] - &f->w[0][0]);
}

/* the pointer's size, the row's and an element's */
static uint32_t sizes(const struct func *f) {
  return (uint32_t)(sizeof f->calls * 1000u + sizeof *f->calls * 100u + sizeof f->calls[0] * 10u + sizeof f->calls[0][0] +
                    sizeof *f->w * 7u + sizeof(struct func) * 3u);
}

/* tested for null and copied member to member; through an element of a member array, a loaded pointer and a value */
static uint32_t copies(struct unit *u, uint32_t i) {
  struct func *a = &u->funcs[i & 1u], *b = &u->funcs[(i + 1u) & 1u];
  uint32_t r = (a->calls == 0) + (b->w != 0) * 2u;
  b->w = a->w;
  b->calls = a->calls;
  r += (b->w == a->w) * 4u;
  u->funcs[1].w[1][2] = (uint16_t)(i + 40u);
  u->cur = b;
  u->cur->w[0][1] += 3u;
  u->cur->next->calls[0][1] = 'q';
  struct func s = *a;
  return r + s.w[1][2] + u->funcs[0].w[0][1] + (uint32_t)u->cur->next->calls[0][1];
}

static uint32_t cells(union cell *c, uint32_t k) {
  c->w[k][0] = 11u;
  c->w[k][2] = (uint16_t)(c->w[k][0] + k);
  return c->w[k][0] + c->w[k][2] * 3u;
}

uint32_t csub2_rowptr(uint32_t x, uint32_t y) {
  static char names[4][NAME];
  static char spare[2][NAME];
  static uint16_t wt[3][3];
  static uint16_t wt2[3][3];
  static struct unit u;
  static struct func g;
  for (uint32_t k = 0; k < 3u; k++)
    for (uint32_t j = 0; j < 3u; j++) {
      wt[k][j] = (uint16_t)(x + k * 3u + j);
      wt2[k][j] = (uint16_t)(y ^ (k + j));
    }
  for (uint32_t j = 0; j < NAME; j++) spare[0][j] = spare[1][j] = 0;
  u.funcs[0].calls = names;
  u.funcs[0].n_calls = 0;
  u.funcs[0].w = wt;
  u.funcs[0].next = &g;
  u.funcs[1].calls = 0;
  u.funcs[1].n_calls = 0;
  u.funcs[1].w = wt2;
  u.funcs[1].next = &g;
  g.calls = spare;
  g.w = wt2;
  g.next = &g;
  struct func *f = &u.funcs[0];
  uint32_t r = (uint32_t)(put_call(f, "merge") + 1) * 3u + (uint32_t)(put_call(f, "rank") + 1) * 5u;
  r += (uint32_t)(put_call(f, (x & 1u) ? "split" : "merge") + 1) * 7u;
  r += (uint32_t)(find_call(f, "rank") + 2) * 11u + (uint32_t)(find_call(f, "absent") + 2) * 13u;
  r += (uint32_t)(put_call(&u.funcs[1], "none") + 2) * 17u;
  r += weights(f, y % 3u, (uint16_t)x) * 19u + sizes(f) * 23u;
  r += copies(&u, x) * 29u;
  union cell c;
  c.w = wt;
  r += cells(&c, x % 3u) * 31u;
  return r + (uint32_t)names[1][0] + wt[2][2] + (uint32_t)spare[0][1];
}
