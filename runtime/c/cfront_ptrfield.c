/* Pointer values stored into / loaded from a struct field (#ptrfield): `s->p = q;` and `return s->p;`.
 * The C twin modeled a pointer struct member as 4 bytes -- so the struct LAYOUT itself was wrong (an
 * adjacent field overlapped the high half of the pointer on a 64-bit target) AND a store truncated the
 * 8-byte pointer to 4 (`uint32_t _v = q`). The oracle already laid pointer members out at pointer_size
 * (8) bytes, matching Clang; this brings the twin into agreement: a pointer member occupies pointer_size,
 * its store/load moves the full pointer, and the loaded value carries its real `T *` type.
 *
 * The harness fills the pointee struct with random bytes then the function overwrites the fields, so no
 * garbage pointer is dereferenced -- each function stores and reads back, comparing the returned pointer
 * (an offset into a buffer) or the adjacent scalar (which a 4-byte pointer model would clobber). (Slice 2
 * of the pointer-value model: deref-through and chaining a loaded pointer field -- `*(s->p)`, `s->t->a`
 * -- are slice 2b.) */
struct Node { int *p; int n; long *q; };          /* two pointers + a scalar: stresses the 8-byte offsets */

int *pf_store_return(struct Node *s, int *v) {
  s->p = v;                                        /* store an 8-byte pointer into a field... */
  return s->p;                                     /* ...and read it back (round-trips through 8 bytes) */
}
unsigned pf_layout(struct Node *s, int *v, int k) {
  s->n = k;                                        /* write the scalar FIRST, then the pointer: a 4-byte */
  s->p = v;                                        /* pointer model puts `n` at offset 4, so the 8-byte */
  return (unsigned)s->n;                           /* store of p (0..7) would clobber it -- the fix keeps n at 8 */
}
long *pf_long(struct Node *s, long *v) {
  s->q = v;                                        /* a long* field (8-byte pointee), at the aligned offset */
  return s->q;
}
unsigned pf_both(struct Node *s, int *v, long *w, int k) {
  s->n = k;                                        /* scalar first, then both pointers: the scalar between */
  s->p = v; s->q = w;                              /* two 8-byte pointers survives only if p (0..7) and */
  return (unsigned)s->n;                           /* q (16..23) don't overlap n (8) -- i.e. the 8-byte layout */
}
