/* C23 `_BitInt(N)` BITFIELDS (#bitintbitfield): a bit-precise bitfield `_BitInt(N) m : W` (2<=N<=64,
 * 1<=W<=N) packs into the `_BitInt(N)` STORAGE UNIT (its 1/2/4/8-byte slot) LSB-first, byte-identical to a
 * standard-integer bitfield of that storage size -- verified == Clang (sizeof / _Alignof / bit-layout) by
 * the layout differential in test_c_cfront.py. The member access reads/writes the W bits; the read value
 * undergoes C23 integer promotion keyed on W: W<=32 promotes to `int` (so `s.f + s.f` is `int`), W>32 keeps
 * `_BitInt(N)` (int can't hold it) -- matching Clang. The emit prints `_BitInt(N) m : W;` faithfully via the
 * original source, so Clang lays it out the same way in BOTH the original and the re-emitted bcir_*.
 *
 * Arithmetic is on UNSIGNED lanes (wrap-around defined, full-range seeding without UB); the entry RETURNS a
 * computed `_BitInt(12)` the harness compares. */

/* two `_BitInt(12)` bitfields packed into one 2-byte (`_BitInt(12)`-sized) storage unit, LSB-first */
struct packed12 { unsigned _BitInt(12) a : 5; unsigned _BitInt(12) b : 6; };

/* a `_BitInt(12)` bitfield next to a STANDARD-int bitfield (mixed packing, same 2-byte unit family) */
struct mixed12 { unsigned _BitInt(12) lo : 5; unsigned int hi : 6; };

/* a WIDE (W>32) `_BitInt(64)` bitfield: it does NOT promote in arithmetic -- stays `_BitInt(64)`. */
struct wide64 { unsigned _BitInt(64) x : 40; unsigned _BitInt(64) y : 20; };

/* read two narrow `_BitInt(12)` bitfields, sum (each promotes to int per C23, W=5/6 <= 32), narrow to 12. */
unsigned _BitInt(12) bf_read_narrow(struct packed12 s) {
  unsigned _BitInt(12) r = (unsigned _BitInt(12))(s.a + s.b);   /* the int sum narrows into the 12-bit lane */
  return r;
}

/* WRITE a `_BitInt(12)` bitfield, then read it back (round-trips through the 5-bit field of the unit). */
unsigned _BitInt(12) bf_write_read(unsigned _BitInt(12) v) {
  struct packed12 s; s.a = v; s.b = (unsigned _BitInt(12))0;
  return s.a;                                            /* the stored low 5 bits, typed `_BitInt(12)` */
}

/* a WIDE bitfield: `s.x` (W=40 > 32) stays `_BitInt(64)`, so same-type arithmetic stays 64-bit (no promote). */
unsigned _BitInt(64) bf_wide(struct wide64 s) {
  unsigned _BitInt(64) w = s.x + (unsigned _BitInt(64))1;   /* ubi64 + ubi64 -> ubi64 (the wide field, no promote) */
  return w;
}

/* the ENTRY: build a packed struct from the inputs, read its bitfields back, compose a 12-bit result. */
unsigned _BitInt(12) bitint_bitfield_entry(unsigned _BitInt(12) a, unsigned _BitInt(12) b) {
  struct packed12 s; s.a = a; s.b = b;                   /* write both bitfields of one storage unit */
  struct mixed12 m; m.lo = a; m.hi = (unsigned)b;        /* a bit-int bitfield + a standard bitfield */
  unsigned _BitInt(12) r = bf_read_narrow(s);            /* read them back, sum-and-narrow */
  unsigned _BitInt(12) t = (unsigned _BitInt(12))(m.lo + m.hi);   /* mixed-kind bitfield read, narrow */
  return r + t + bf_write_read(a);                       /* same-type 12-bit adds (wrap) */
}
