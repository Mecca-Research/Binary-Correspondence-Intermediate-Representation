/* The rail splits CF-SPLIT2 closed: forms the cfront rails lowered to different claim graphs, or that one rail
 * lowered and the other refused. `p = p + n` was a `c.ptradd` on the oracle and a sum and a copy on the twin; the
 * twin refused an increment or an assignment used as a value through a pointer the lvalue loads (`h.next->v++`,
 * `x = (h.next->v = 5)`), through a pointer parameter (`p[i]++`) or a dereference (`(*p)++`, `++*p`), the
 * address of such an object (`&h.next->v`) and a call through a parenthesized callee (`(fp)(x)`); the oracle
 * refused a dereference of any pointer value but a name, a member or a call (`*&a`, `*(c ? &a : &b)`, `*p++`) and
 * a statement `++h.next->v;`; and the twin took `*q->a` of a member array as the array's address and a load through
 * it, where the oracle loads the first element. Each lowers to one claim graph now, and runs as the original does. */
#include <stdint.h>

struct sp_node { uint32_t v; uint8_t c; int16_t w; unsigned bits : 5; struct sp_node *next; uint32_t *arr; };
struct sp_holder { struct sp_node *next; };
struct sp_ops { uint32_t (*fn)(uint32_t); };
struct sp_atom { _Atomic uint32_t n; _Atomic uint32_t *slots; struct sp_atom *self; };
struct sp_in { uint32_t x; uint16_t b[3]; };
struct sp_rows { uint32_t n; uint32_t a[4]; uint8_t c[2]; struct sp_in in; };

static struct sp_node sp_n1;
static struct sp_node sp_n2;
static uint32_t sp_buf[8];
static struct sp_atom sp_at;
static _Atomic uint32_t sp_slots[4];
static struct sp_rows sp_rw;

static uint32_t sp_twice(uint32_t x) { return 2u * x + 1u; }

static void sp_link(uint32_t s) {                   /* two nodes that point at each other, and the buffer */
  sp_n1.v = s; sp_n1.c = (uint8_t)(s + 250u); sp_n1.w = (int16_t)(s & 0x7FFFu); sp_n1.bits = s & 31u;
  sp_n1.next = &sp_n2; sp_n1.arr = sp_buf;
  sp_n2.v = s ^ 5u; sp_n2.c = (uint8_t)s; sp_n2.w = -3; sp_n2.bits = 30u;
  sp_n2.next = &sp_n1; sp_n2.arr = sp_buf + 4;
  for (uint32_t i = 0; i < 8u; i++) sp_buf[i] = s + i;
}

uint32_t sp_ptr_sum(uint32_t s) {                   /* p = p + n: a pointer sum, then a copy, on both rails */
  sp_link(s);
  uint32_t *p = sp_buf;
  uint32_t *q = sp_buf + 7;
  uint32_t acc = 0;
  p = p + 1;
  acc += *p;
  p = p + (s & 1u);
  acc += *p;
  q = q - 2;
  acc += *q;
  p = 1 + p;
  acc += *p;
  p += 1;
  p++;
  acc += *p;
  return acc;
}

uint32_t sp_chain_steps(uint32_t s) {               /* ++/-- through the pointer the lvalue loads */
  struct sp_holder h;
  sp_link(s);
  h.next = &sp_n1;
  uint32_t a = h.next->v++;
  uint32_t b = ++h.next->v;
  h.next->next->v--;
  --h.next->next->v;
  ++h.next->v;
  uint32_t c = h.next->c++;                         /* a uint8_t: wraps at 256 */
  uint32_t d = ++h.next->c;
  int32_t e = h.next->w--;
  int32_t f = --h.next->next->w;
  uint32_t g = h.next->bits++;                      /* a 5-bit field: wraps at 32 */
  uint32_t k = ++h.next->next->bits;
  uint32_t m = h.next->arr[s & 3u]++;
  uint32_t n = ++h.next->next->arr[1];
  return a * 3u + b + c + d + (uint32_t)e + (uint32_t)f + g + k + m + n + sp_n1.v + sp_n2.v + sp_n1.c
         + sp_n1.bits + sp_n2.bits + sp_buf[s & 3u] + sp_buf[5];
}

uint32_t sp_chain_values(uint32_t s) {              /* `=` and OP= used as a value through the loaded pointer */
  struct sp_holder h;
  sp_link(s);
  h.next = &sp_n1;
  uint32_t a = (h.next->v = s + 3u);
  uint32_t b = (h.next->v += 5u);
  uint32_t c = (h.next->next->v ^= s);
  uint32_t d = (h.next->c = (uint8_t)(s + 7u));
  uint32_t e = (h.next->c += 250u);                 /* narrows: the stored byte, re-read */
  uint32_t f = (h.next->arr[s & 3u] = s * 7u);
  uint32_t g = (h.next->next->arr[2] -= 9u);
  uint32_t k = (h.next->bits = s);                  /* the 5-bit field keeps s & 31 */
  return a + b + c + d + e + f + g + k + sp_n1.v + sp_n2.v + sp_buf[s & 3u] + sp_buf[6];
}

uint32_t sp_param_elems(uint32_t *p, uint32_t i) {  /* p[i] of a pointer parameter, stepped and assigned */
  p[i]++;
  ++p[i + 1u];
  uint32_t a = p[i]++;
  uint32_t b = ++p[i];
  uint32_t c = (p[i] = a + 7u);
  uint32_t d = (p[i + 1u] -= 3u);
  p[i]--;
  return a + b + c + d + p[i] + p[i + 1u];
}

uint32_t sp_param_bytes(uint8_t *p, uint32_t i) {   /* ... a uint8_t element narrows on every store */
  uint32_t a = ++p[i];
  uint32_t b = (p[i] += 300u);
  uint32_t c = p[i]--;
  uint32_t d = (p[i] = (uint8_t)(i + 250u));
  return a + b + c + d + p[i];
}

uint32_t sp_derefs(uint32_t s) {                    /* *e of any pointer value, stepped and assigned */
  uint32_t a = s;
  uint32_t b = s ^ 9u;
  uint32_t *p = &a;
  (*p)++;
  ++*p;
  uint32_t x = (*p)--;
  uint32_t y = --*p;
  uint32_t z = *&a + *&b;
  *&b = s + 11u;
  uint32_t w = *(s & 1u ? &a : &b);
  *(s & 2u ? &a : &b) += 4u;
  (*&a)++;
  ++*&b;
  sp_link(s);
  uint32_t *q = sp_buf;
  uint32_t v = *q++;
  v += *q++;
  v += *(q - 1);
  ++*(q + 0);
  uint32_t u = (*(q + 1))++;
  return a + b + x + y + z + w + v + u + *q + sp_buf[3];
}

uint32_t sp_addresses(uint32_t s) {                 /* &obj through the pointer the lvalue loads */
  struct sp_holder h;
  sp_link(s);
  h.next = &sp_n1;
  uint32_t *pv = &h.next->v;
  uint32_t *pn = &h.next->next->v;
  uint32_t *pe = &h.next->arr[s & 3u];
  uint8_t *pc = &h.next->next->c;
  *pv += 1u;
  *pn ^= s;
  *pe = s * 3u;
  *pc = (uint8_t)(*pc + 200u);
  return *pv + *pn + *pe + *pc + sp_n1.v + sp_n2.v + sp_buf[s & 3u];
}

uint32_t sp_paren_calls(uint32_t s) {               /* a call through a parenthesized callee */
  uint32_t (*fp)(uint32_t) = sp_twice;
  struct sp_ops o;
  o.fn = sp_twice;
  return (fp)(s) + ((fp))(s + 1u) + (sp_twice)(s + 2u) + (o.fn)(s + 3u) + fp(s + 4u);
}

uint32_t sp_atomics(uint32_t s) {                   /* an `_Atomic` object through the loaded pointer */
  struct sp_atom *a = &sp_at;
  a->self = &sp_at;
  a->slots = sp_slots;
  a->n = s;
  sp_slots[1] = s ^ 3u;
  uint32_t x = a->self->n++;
  uint32_t y = ++a->self->n;
  uint32_t z = (a->self->n += 3u);
  uint32_t w = a->self->slots[1]++;
  uint32_t v = (a->self->slots[1] = s + 2u);
  uint32_t t = --a->self->slots[1];
  return x + y + z + w + v + t + a->n + sp_slots[1];
}

uint32_t sp_member_arrays(uint32_t s) {             /* `*q->a`: the member array's first element, in place */
  struct sp_rows *q = &sp_rw;
  struct sp_rows l;
  for (uint32_t i = 0; i < 4u; i++) sp_rw.a[i] = s + i;
  sp_rw.c[0] = (uint8_t)s;
  sp_rw.c[1] = 7u;
  sp_rw.in.x = s;
  sp_rw.in.b[0] = (uint16_t)(s >> 3);
  l = sp_rw;
  uint32_t a = *q->a;
  *q->a += 3u;
  ++*q->a;
  uint32_t b = (*q->a)++;
  uint32_t c = (*q->a = s ^ 0x55u);
  uint32_t d = (*q->c += 250u);                     /* a uint8_t element narrows */
  *q->in.b = (uint16_t)(s * 3u);
  uint32_t e = ++*q->in.b;
  uint32_t f = *l.a + *l.in.b + *l.c;
  uint8_t *pc = &*q->c;
  return a + b + c + d + e + f + *pc + sp_rw.a[0] + sp_rw.c[0] + sp_rw.in.b[0];
}

uint32_t sp_entry(uint32_t s) {                     /* every form once */
  static uint32_t cells[4];
  cells[s & 3u] = s;
  return sp_ptr_sum(s) + sp_chain_steps(s) + sp_chain_values(s) + sp_param_elems(cells, s & 1u)
         + sp_derefs(s) + sp_addresses(s) + sp_paren_calls(s) + sp_atomics(s) + sp_member_arrays(s);
}
