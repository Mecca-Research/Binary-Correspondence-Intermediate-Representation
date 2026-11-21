/* CSUB-2: a struct or union a call returns through a prototype -- `bcir_handoff.c`'s `c = bcir_ctl_admit_pack(...)`, a
 * callee another unit defines -- and a pointer one returns, followed with `->` and `[]`, behaviour-checked on both
 * rails. The aggregate is assigned, initialized from, read a member of in place (`pr_make(s).b`), passed on by value,
 * stored into an element and chosen by `?:`. The test's driver defines the callees. */
#include <stdint.h>
#include <stddef.h>

typedef struct {
  uint8_t ok;
  uint16_t why;
  uint64_t gen;
} pr_outcome;
struct pr_pair {
  uint32_t a, b;
};
union pr_word {
  uint32_t w;
  uint8_t b[4];
};

pr_outcome pr_admit(const uint8_t *data, size_t len);
struct pr_pair pr_make(uint32_t x);
union pr_word pr_word_of(uint32_t x);
struct pr_pair *pr_find(uint32_t x);
uint32_t *pr_table(void);

/* bcir_handoff.c's admit: the outcome assigned, then its members tested */
uint32_t pr_admit_entry(uint32_t s) {
  static uint8_t buf[8];
  for (uint32_t i = 0; i < 8u; i++) buf[i] = (uint8_t)(s + i);
  pr_outcome c;
  c = pr_admit(buf, s & 7u);
  if (!c.ok) return c.why;
  return (uint32_t)c.gen + c.why;
}

static uint32_t pr_sum(struct pr_pair p) { return p.a * 3u + p.b; }

/* initialized from, read a member of in place, passed by value, stored into an element, chosen by `?:` */
uint32_t pr_pair_entry(uint32_t s) {
  struct pr_pair p = pr_make(s);
  struct pr_pair q[2];
  q[1] = pr_make(s + 1u);
  struct pr_pair r = (s & 1u) ? pr_make(s ^ 3u) : q[1];
  return p.a + pr_make(s).b + pr_sum(pr_make(s ^ 5u)) * 7u + q[1].b + r.a * 11u;
}

/* a union returned, and a member of it read in place */
uint32_t pr_word_entry(uint32_t s) {
  union pr_word u = pr_word_of(s);
  return u.b[0] + u.b[3] * 5u + pr_word_of(s + 2u).w;
}

/* a pointer a prototyped callee returns, followed with `->` and `[]` */
uint32_t pr_ptr_entry(uint32_t s) { return pr_find(s)->a + pr_find(s + 1u)[0].b * 3u + pr_table()[s & 3u] * 5u; }
