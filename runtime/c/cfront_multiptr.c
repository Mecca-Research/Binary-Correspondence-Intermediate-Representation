/* Per-declarator pointer / array shape in a multi-declarator declaration (#multiptr): in `int *p, q;`
 * the `*` binds to the DECLARATOR, not the type-specifier -- p is `int*`, q is `int`. The oracle
 * already typed each declarator individually; the C twin folded `*` into the shared specifier, so
 * `int *p, q;` mis-typed q as an 8-byte pointer (its store/load moved 8 bytes, not 4) and `int *p, *q;`
 * was rejected outright. Now the twin parses the base specifier once (p_type_base) and applies each
 * declarator's own leading `*`s on a fresh copy of it (apply_stars), for both local declarations and
 * struct members. Driven by a differential that uses each trailing declarator AS a scalar -- an 8-byte
 * store would clobber an adjacent field / read back wrong. */

struct Mix { int *p, q; long *lp, m; };   /* a pointer + a scalar, twice: q and m must stay 4/8-byte scalars */

int md_local_mixed(int n) {               /* `int *p, q;` -- p is a pointer, q is a scalar int */
  int *p, q;
  int x = n * 3 + 1;
  p = &x; q = n - 7;
  return *p + q;
}
int md_local_two_ptr(int a, int b) {      /* `int *p, *q;` -- both pointers (was rejected by the twin) */
  int x = a, y = b;
  int *p, *q;
  p = &x; q = &y;
  return *p * 2 + *q;
}
int md_local_ptr_arr(int n) {             /* `int *p, arr[3];` -- a pointer then an array declarator */
  int v = n + 5;
  int *p, arr[3];
  p = &v; arr[0] = n; arr[1] = n + 1; arr[2] = n + 2;
  return *p + arr[0] + arr[1] + arr[2];
}
long md_struct(struct Mix *s, int n) {    /* the struct's q / m are scalars, not pointers */
  int x = n;
  long y = (long)n * 7 - 3;
  s->p = &x; s->q = n + 11; s->lp = &y; s->m = (long)n - 9;
  return (long)(*(s->p)) + s->q + *(s->lp) + s->m;
}
