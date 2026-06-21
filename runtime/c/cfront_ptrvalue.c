/* Pointer values returned from a function (#ptrvalue): pointer arithmetic `p + i` used as an rvalue
 * yields a pointer that is returned by value. The C twin used to type the `p + i` temporary as a plain
 * uint32_t -- an invalid pointer-from-integer return under C23 AND an 8-byte-pointer truncation on a
 * 64-bit target (a miscompile where it compiled at all). The oracle, which carries the pointee type,
 * was correct. Now the twin clones the pointee (width/sign/tag) onto the arithmetic result, emitting a
 * real pointee-scaled `T *t = p + i`. The harness compares the RETURNED pointers (offsets into a shared
 * buffer), so each function keeps its index in-bounds. (Slice 1 of the pointer-value model: returning a
 * pointer; storing one into a struct field and pointer-to-pointer are the follow-on slices.) */
int *pv_advance(int *p, unsigned n) {
  return p + n;                                   /* pointer + int -> pointer */
}
int *pv_commute(int *p, unsigned n) {
  return (int)n + p;                              /* int + pointer (commuted) -> pointer */
}
int *pv_back(int *p, unsigned n) {
  return (p + 200) - (n % 100u);                  /* pointer - int -> pointer (offset kept in [101,200]) */
}
unsigned *pv_uadvance(unsigned *p, unsigned n) {
  return p + n;                                   /* unsigned-pointee pointer arithmetic */
}
short *pv_sadvance(short *p, unsigned n) {
  return p + n;                                   /* sub-int pointee: the stride is sizeof(short), not 4 */
}
int *pv_chain(int *p, unsigned n) {
  int *q = p + n;                                 /* a pointer rvalue bound to a local, then advanced */
  return q + 1;
}
