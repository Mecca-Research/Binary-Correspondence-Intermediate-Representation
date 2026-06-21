/* Pointer locals -- address-of a local into a typed pointer variable (#ptrlocal): `T *p = &x;` then
 * read/write `*p`. The C twin emitted a pointer local as a plain `uint32_t`, so `p = &x;` was an invalid
 * pointer-to-integer assignment (a hard error under C23) AND truncated the 8-byte address to 32 bits on a
 * 64-bit target -- a miscompile where it compiled at all. The oracle, which emits `T *p`, was correct.
 * Now the twin carries the pointee (width, signedness, struct tag) on the pointer resource and emits the
 * real `T *p`, with the deref reading exactly the pointee width (so a `signed char *` deref sign-extends). */
struct Pair { int x; int y; };

unsigned pl_store(unsigned a, unsigned b) {
  int c = (int)a - (int)b;
  int *p = &c;
  *p = (int)b * 2;                                 /* mutate the local through the pointer */
  return (unsigned)c;
}
unsigned pl_read(unsigned a, unsigned b) {
  int c = (int)a - (int)b;
  int *p = &c;
  return (unsigned)(*p + 1);
}
unsigned pl_schar(unsigned a, unsigned b) {
  signed char c = (signed char)((int)a - (int)b);
  signed char *p = &c;
  return (*p < 0) ? 1u : 0u;                        /* a sub-int pointee deref sign-extends */
}
unsigned pl_compound(unsigned a, unsigned b) {
  int c = (int)a - (int)b;
  int *p = &c;
  *p += 100;                                        /* read-modify-write through the pointer */
  return (unsigned)c;
}
unsigned pl_struct(unsigned a, unsigned b) {
  struct Pair s; s.x = (int)a - (int)b; s.y = (int)b;
  struct Pair *p = &s;
  p->x += 5;                                        /* member access through a struct pointer */
  return (unsigned)(p->x + p->y);
}
unsigned pl_reassign(unsigned a, unsigned b) {
  int x = (int)a, y = (int)b;
  int *p = &x;
  p = &y;                                           /* a pointer local reassigned to another address */
  return (unsigned)(*p);
}
