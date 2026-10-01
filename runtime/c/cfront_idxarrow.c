/* Elements that are pointers (CF-IDXARROW): of an array of pointers, through a `T **`, of a file-scope array of
 * pointers. The twin had read `pp[1][0].f` -- assigned, compounded or stepped as a value -- at the flat index `1*1+0`
 * of `pp`, a struct's field inside the pointer array; typed `pp + 1` and `arr + 1` as a `T *` and `&arr[i]` as a
 * `T *` four bytes apart, so their emits did not compile or compared unequal pointers; and read a file-scope array of
 * pointers as an array of integers. The oracle lowered each. Both rails lower them to one claim graph, and each
 * emit returns what the original does. A member access straight through such an element, `arr[i]->f`, is refused on
 * both (the oracle has no subscript base; the twin had read the pointer's slot as the struct). The objects pointed
 * at live at file scope -- the G10 escape rows count a local whose address is stored -- and a function that writes
 * one sets it first, so the original and the emit read the same. */
#include <stdint.h>

struct ix_s { uint32_t v; uint32_t w; };
typedef struct ix_s *ix_sp;
typedef struct ix_s **ix_spp;
static int32_t ix_gx = -5, ix_gy = 6;
static int32_t *ix_garr[2] = {&ix_gx, &ix_gy};
static struct ix_s ix_ga = {7u, 8u}, ix_gb = {9u, 10u};
static struct ix_s *ix_gsp[2] = {&ix_ga, &ix_gb};
static char ix_c0 = 1, ix_c1 = 3;
static double ix_d0 = 1.5, ix_d1 = 2.5;
static void *ix_gva[2] = {&ix_gx, &ix_gy};

uint32_t ix_chain_values(uint32_t s) {   /* `pp[i][j].f` assigned, compounded and stepped as a value */
  ix_ga.v = s;
  ix_gb.w = 2u;
  struct ix_s *arr[2] = {&ix_ga, &ix_gb};
  struct ix_s **pp = arr;
  uint32_t r = (pp[1][0].w = s + 9u);
  r += (pp[0][0].v += 3u);
  r += pp[1][0].w++;
  r += ++pp[0][0].v;
  pp[1][0].w--;
  return r + ix_ga.v + ix_gb.w * 3u + arr[1][0].w * 5u;
}
uint32_t ix_steps(uint32_t s) {          /* `pp + 1`, `pp++`, `pp += 0` of a `T **`; an array of pointers decayed */
  ix_gx = (int32_t)s;
  int32_t *arr[2] = {&ix_gx, &ix_gy};
  int32_t **pp = arr;
  int32_t **q = pp + 1;
  int32_t **r = arr + 1;
  int32_t **t = (s & 1u) ? q : arr;
  pp++;
  pp += 0;
  return (uint32_t)(**q + **r * 3 + **t * 5 + **pp * 7);
}
uint32_t ix_addresses(uint32_t s) {      /* `&arr[i]` of an array of pointers: a `T **`, a pointer apart */
  ix_gx = (int32_t)s;
  int32_t **q = &ix_garr[1];
  void **vq = &ix_gva[s & 1u];
  return (uint32_t)**q + (q == ix_garr + 1) * 3u + (vq == ix_gva + (s & 1u)) * 5u + (uint32_t)*(int32_t *)*vq * 7u;
}
uint32_t ix_kinds(uint32_t s) {          /* `char **`, `double **` and a pointer to struct pointers */
  char *cs[2] = {&ix_c0, &ix_c1};
  char **cp = cs + 1;
  double *ds[2] = {&ix_d0, &ix_d1};
  double **dq = ds + 1;
  struct ix_s *ss[2] = {&ix_ga, &ix_gb};
  struct ix_s **sq = ss + 1;
  struct ix_s *z = *sq;
  return (uint32_t)**cp + (uint32_t)(**dq * 2.0) + z->w * 3u + s;
}
uint32_t ix_typedefs(uint32_t s) {       /* a pointer to a struct-pointer typedef, and a typedef of two: `struct ix_s **` */
  ix_sp arr[2] = {&ix_ga, &ix_gb};
  ix_sp *pp = arr;
  ix_sp q = pp[1];
  ix_sp *r = pp + 1;
  ix_sp z = *r;
  ix_spp t = ix_gsp;
  struct ix_s *u = t[1];
  return q->v + z->w * 3u + (uint32_t)sizeof(*pp) * 5u + u->v * 7u + (uint32_t)sizeof(*t) + s;
}
uint32_t ix_globals(uint32_t s) {        /* a file-scope array of pointers decays to a `T **` */
  int32_t **pp = ix_garr + 1;
  int32_t *p0 = *ix_garr;
  struct ix_s *z = *ix_gsp;
  struct ix_s **zz = &ix_gsp[1];
  struct ix_s *zb = *zz;
  return (uint32_t)(**pp + *p0 * 3 + **ix_garr * 5) + z->v + zb->w * 7u + s;
}
uint32_t ix_entry(uint32_t s) {
  return ix_chain_values(s) + ix_steps(s) * 3u + ix_addresses(s) * 5u + ix_kinds(s) * 7u + ix_typedefs(s) * 11u
         + ix_globals(s) * 13u;
}
