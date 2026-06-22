/* Nested structs (#neststruct): a struct member that is itself a struct -- read/written via `o.in.x` /
 * `o->in.x` chains (the frontend flattens the member-access offset), and INITIALISED with a nested brace
 * `{ { inner.. }, .. }`. The brace init recurses through struct nesting on both rails (offset-based member
 * stores, the same path as an array member); a `_Bool` nested member still normalizes, an inner array
 * member still lands element-by-element. Both rails agree + match Clang on size/offsets/values. */

struct Inner { unsigned a; short b; _Bool f; };
struct Outer { struct Inner in; unsigned tag; };

/* by-value: nested member reads. */
unsigned outer_read(struct Outer o) {
  return o.in.a + (unsigned)o.in.b + (unsigned)o.in.f + o.tag;
}
/* via pointer: nested member writes (incl. a `_Bool` nested member that normalizes). */
unsigned outer_write(struct Outer *o, unsigned x) {
  o->in.a = x;
  o->in.b = (short)(x + 1u);
  o->in.f = x & 4u;
  o->tag = x ^ 7u;
  return o->in.a + (unsigned)o->in.b + (unsigned)o->in.f + o->tag;
}

/* a deeper nest with an inner ARRAY member, initialised end-to-end by nested braces. */
struct Grid  { unsigned cells[3]; _Bool on; };
struct Frame { struct Grid g; struct Inner hdr; unsigned id; };
unsigned frame_make(unsigned x) {
  struct Frame f = { { { x, x + 1u, x + 2u }, x & 1u }, { x, (short)x, x & 8u }, x + 100u };
  unsigned r = f.id + (unsigned)f.g.on + f.hdr.a + (unsigned)f.hdr.b + (unsigned)f.hdr.f;
  for (unsigned k = 0u; k < 3u; k++) r += f.g.cells[k];
  return r;
}
