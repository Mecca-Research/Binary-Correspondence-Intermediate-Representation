/* Nested struct member access (#nestmember): `o.pos.lo` / `dev->ctrl.flags` -- a struct-in-struct /
 * sub-register-block. Both rails flatten the `.`/`->` chain to a single offset access (matching Clang's
 * layout), so byte offsets + reads / writes / compound-assign / nested bitfields all agree. The C twin's
 * `field` now carries a sub-struct index, and a descent helper accumulates the offset through each hop. */
struct Inner { unsigned lo, hi; };
struct Outer { struct Inner pos; unsigned tag; };
unsigned outer_sum(unsigned a) {
  struct Outer o;
  o.pos.lo = a; o.pos.hi = a * 2u; o.tag = a ^ 7u;
  o.pos.lo += 3u;                        /* nested compound assignment */
  return o.pos.lo + o.pos.hi + o.tag;
}
struct Bits { unsigned mode : 4, lvl : 4; };
struct Dev { struct Bits ctrl; unsigned status; };
unsigned dev_pack(unsigned v) {
  struct Dev d;
  d.ctrl.mode = v; d.ctrl.lvl = v >> 2; d.status = v << 1;   /* nested bitfields + a plain member */
  return d.ctrl.mode + d.ctrl.lvl + d.status;
}
