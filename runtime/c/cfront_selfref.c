/* Self-referential and forward-referenced structs (C11 6.2.5p22, 6.7.2.3, CF-SELFREF): a struct holds a
 * pointer to its own type before its definition closes; a pointer to a struct declared later -- or only ever
 * declared -- is an ordinary pointer, its target laid out when (and if) the struct is completed; `struct t;`
 * declares the tag; a typedef may name a pointer to an incomplete struct; a function may take and return
 * pointers to such structs, and a call's result is dereferenced in place; `typeof` a pointer member to the
 * struct being defined names the completed struct, so a subscript through it steps one whole element. Both
 * rails lay them out the same way, and each emit returns what the original does. */
#include <stdint.h>

struct node { uint32_t v; struct node *next; };
struct a_side;                                      /* declared here, defined after its first use */
struct b_side { struct a_side *peer; uint16_t w; };
struct a_side { struct b_side *peer; uint32_t v; };
struct opaque;                                      /* never defined: used only through a pointer */
typedef struct opaque *handle_t;
struct tree { struct tree *left, *right; uint8_t key; };

struct node *sr_advance(struct node *n, uint32_t k) {   /* a pointer to the struct, returned */
  for (uint32_t j = 0u; j < k; j++) {
    if (n->next) n = n->next;
  }
  return n;
}
uint32_t sr_list(uint32_t s) {                      /* a list linked through the struct's own pointer */
  struct node c = {s * 3u, 0};
  struct node b = {s + 2u, &c};
  struct node a = {s, &b};
  uint32_t sum = 0u;
  for (struct node *p = &a; p; p = p->next) sum = sum * 31u + p->v;
  return sum + sr_advance(&a, s % 4u)->v * 7u + (uint32_t)sizeof(struct node);
}
uint32_t sr_mutual(uint32_t s) {                    /* two structs that point at each other */
  struct a_side a;
  struct b_side b;
  a.peer = &b;
  a.v = s;
  b.peer = &a;
  b.w = (uint16_t)(s >> 1);
  struct b_side *pb = a.peer;
  struct a_side *pa = pb->peer;
  return pa->v + pb->w * 3u + (uint32_t)sizeof(struct a_side) * 5u;
}
uint32_t sr_opaque(handle_t h, struct opaque *o, uint32_t s) {   /* pointers to a struct never defined */
  handle_t k = h;
  return (k == o ? 1u : 0u) + (h ? 2u : 0u) + s;
}
uint32_t sr_tree(uint32_t s) {                      /* two pointers to its own type */
  struct tree l = {0, 0, (uint8_t)(s + 1u)};
  struct tree r = {0, 0, (uint8_t)(s + 2u)};
  struct tree t = {&l, &r, (uint8_t)s};
  return t.key + t.left->key * 3u + t.right->key * 5u + (t.left->left ? 7u : 0u);
}
uint32_t sr_typeof_next(uint32_t s) {              /* `typeof` a member that points at its own struct */
  struct node arr[3] = {{s, 0}, {s + 5u, 0}, {s * 3u, 0}};
  struct node h = {0u, arr};
  typeof(h.next) q = h.next;                        /* `struct node *`: q[1] is the next whole node */
  return q[1].v + q[2].v * 3u;
}
uint32_t sr_entry(uint32_t s) {
  uint32_t x = s;
  return sr_list(s) + sr_mutual(s) * 3u + sr_opaque((handle_t)&x, (struct opaque *)&x, s) * 5u + sr_tree(s) * 7u;
}
