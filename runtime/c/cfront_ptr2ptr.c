/* Pointer-to-pointer (#ptr2ptr): `int **pp`, the double dereference `**pp`, a store through two levels
 * `*pp = q` (the output-parameter idiom) and `**pp = v`, and `int **pp = &p` built by address-of-a-
 * pointer. Both rails modeled `int **` as a single `int *` (no indirection depth in the type model), so
 * `*pp` read 4 bytes (the base int) where the pointee is an 8-byte pointer, `**pp` fell back entirely,
 * and `*pp = q` truncated the stored pointer. Now the type carries a pointer DEPTH: `*pp` on a `T**`
 * loads a `T*` (pointer_size bytes), `**pp` derefs that to a `T`, address-of a `T*` yields a `T**`, and
 * a store through `**` moves a full pointer. Slice 3 of the pointer-value model.
 *
 * Driven by a bespoke harness (the generic one fills a pointee with random bytes -- invalid call/deref
 * targets for a double pointer), which builds real `x / &x / &&x` chains. */

int p2_read(int **pp) {
  return **pp;                                     /* double dereference: load the pointer, then the int */
}
int *p2_get(int **pp) {
  return *pp;                                      /* one level: `*pp` is a pointer (pointer_size load) */
}
void p2_set(int **pp, int *q) {
  *pp = q;                                         /* the output-parameter idiom: store a pointer through ** */
}
int p2_store_through(int **pp, int v) {
  **pp = v;                                        /* store a value through two levels of indirection */
  return **pp;
}
int p2_rmw(int **pp, int d) {
  **pp += d;                                       /* read-modify-write through `**` */
  return **pp;
}
int p2_local(int seed) {
  int x = seed * 3 + 1;
  int *p = &x;
  int **pp = &p;                                   /* address-of a pointer -> a `int **` local */
  return **pp + 7;
}
