/* Deref-through a loaded pointer field (#fieldderef): reading/writing *through* a pointer that lives in
 * a struct field -- `*(s->p)`, the chain `s->mid->k` (member access through a loaded pointer-to-struct
 * field, and the two-hop `s->mid->leaf->x`), and the subscript `s->p[i]` (== `*(s->p + i)`). Both rails
 * resolved a member used as a *base* to the enclosing struct's address + the field type, so a deref read
 * the struct's own bytes instead of loading the pointer first; `s->mid->k` and `s->p[i]` fell back. Now a
 * pointer-valued field used as a base is LOADED (a full pointer_size load at the field's offset) and the
 * loaded pointer becomes the new base -- the same move as `*q` on a `**pp`. Slice 2b of the pointer-value
 * model: slice 2 stored/loaded the field itself; this dereferences and chains it.
 *
 * Driven by a bespoke harness that builds real `s->p = &x` chains (the generic equivalence harness fills
 * a pointee with random bytes -- an invalid deref target). `q` sits at a non-zero, padded offset and the
 * chain hops two pointer levels, so the field offsets and the per-hop loads are all exercised. */
struct Leaf { int x; long y; };                    /* defined first: a forward, non-self-referential chain */
struct Mid  { struct Leaf *leaf; int k; };
struct Box  { struct Mid *mid; int *p; int n; long *q; };   /* p @8, n @16, q @24 (past padding) */

int fd_read(struct Box *s) {
  return *(s->p);                                  /* deref the loaded pointer field */
}
void fd_write(struct Box *s, int v) {
  *(s->p) = v * 3 + 1;                             /* store through the loaded pointer field */
}
long fd_qread(struct Box *s) {
  return *(s->q);                                  /* a long* field at offset 24: proves the field offset */
}
int fd_index(struct Box *s, int i) {
  return s->p[i];                                  /* subscript the loaded pointer field: *(s->p + i) */
}
void fd_index_set(struct Box *s, int i, int v) {
  s->p[i] = v - 5;                                 /* indexed store through the loaded pointer field */
}
int fd_rmw(struct Box *s, int d) {
  *(s->p) += d;                                    /* read-modify-write through the loaded field */
  return *(s->p);
}
int fd_chain1(struct Box *s) {
  return s->mid->k;                                /* member access THROUGH a loaded pointer-to-struct field */
}
void fd_chain1_set(struct Box *s, int v) {
  s->mid->k = v;                                   /* store through a chained pointer field */
}
int fd_chain1_rmw(struct Box *s, int d) {
  s->mid->k += d;                                  /* compound store through a chained pointer field */
  return s->mid->k;
}
int fd_chain2(struct Box *s) {
  return s->mid->leaf->x;                          /* a two-hop pointer chain: Box -> Mid* -> Leaf* -> x */
}
long fd_chain2_long(struct Box *s) {
  return s->mid->leaf->y;                          /* a wide (long) member at the end of a two-hop chain */
}
void fd_chain2_set(struct Box *s, int v) {
  s->mid->leaf->x = v;                             /* store at the end of a two-hop pointer chain */
}
