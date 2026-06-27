/* C23 `_BitInt(N)` as STRUCT / UNION MEMBERS (#bitintmember): a PLAIN (non-bitfield) `_BitInt(N)` member
 * is first-class -- read AND write, with same-type `_BitInt(N)` arithmetic on the loaded member value. The
 * headline is again a NON-standard lane width (`_BitInt(12)`) that has no fixed-width type, alongside a
 * `_BitInt(64)`. Clang lays a `_BitInt(N)`, 2<=N<=64, in the smallest power-of-two byte storage unit >= N
 * bits (so `_BitInt(12)` is 2-byte/2-aligned, exactly like uint16_t); the frontend's `bitint` CType already
 * carries that storage width + alignment, so the layout (sizeof/offsetof) MATCHES Clang -- the member
 * load/store goes through the storage width typed `_BitInt(N)`, and the emit spells `_BitInt(N)` /
 * `unsigned _BitInt(N)` faithfully, keeping the original source and the re-emitted bcir_* behaviour-equivalent.
 *
 * A `_BitInt` BITFIELD (`_BitInt(N) m:W`), mixing a `_BitInt` member with a standard int VARIABLE, and
 * mixing DIFFERENT `_BitInt` widths all STAY routed to fallback (see test_cfront_fallback.py) -- only the
 * PLAIN member is in scope here.
 *
 * Arithmetic stays on UNSIGNED lanes (wrap-around is well defined, so the equivalence harness may seed the
 * full value range without signed-overflow UB). The entry takes a struct BY VALUE and RETURNS a computed
 * unsigned `_BitInt(12)`, so Clang runs both binaries on seeded inputs across the 12/64-bit members. */

/* a struct with a NON-standard-width `_BitInt(12)` member (2-byte storage), a wide `_BitInt(64)` member,
 * and an ordinary `int` -- so the member offsets exercise the mixed (sub-word + word + 8-byte) layout. */
struct bipair {
  int tag;                       /* an ordinary member: pins offset 0, the `_BitInt(12)` lands after it */
  unsigned _BitInt(12) lo;       /* a 12-bit lane: 2-byte/2-aligned storage (== uint16_t), exact 12-bit value */
  unsigned _BitInt(64) hi;       /* a 64-bit lane: 8-byte/8-aligned storage, a width that coincides with a
                                  * standard type but stays a distinct `_BitInt` (no promotion) */
};

/* READ two `_BitInt` members and combine them with SAME-type `_BitInt(12)` arithmetic (the loaded `s.lo`
 * keeps its `_BitInt(12)` type -- it does NOT promote, so `s.lo + (unsigned _BitInt(12))k` wraps at 12 bits). */
unsigned _BitInt(12) bipair_fold(struct bipair s) {
  unsigned _BitInt(12) a = s.lo;                       /* read a 12-bit member */
  unsigned _BitInt(12) b = (unsigned _BitInt(12))s.hi; /* narrow the 64-bit member into the 12-bit lane */
  unsigned _BitInt(12) c = a + b;                      /* same-type 12-bit add (wraps mod 2^12) */
  return c;
}

/* WRITE `_BitInt` members of a LOCAL struct, then READ them back -- the member STORE path at the storage
 * width. The values written are same-type `_BitInt` arithmetic results, so the store/load round-trips the
 * exact N-bit value. */
unsigned _BitInt(12) bipair_rw(unsigned _BitInt(12) x, unsigned _BitInt(64) y) {
  struct bipair s;
  s.tag = 0;
  s.lo = x + (unsigned _BitInt(12))1;     /* WRITE a 12-bit member (a same-type add, wraps at 12 bits) */
  s.hi = y * (unsigned _BitInt(64))3;     /* WRITE a 64-bit member (a same-type multiply, wraps at 64 bits) */
  unsigned _BitInt(12) r = s.lo + (unsigned _BitInt(12))s.hi;   /* read BOTH back; combine at 12 bits */
  return r;
}

/* the ENTRY (the LAST function -- the one the equivalence harness runs): take the struct BY VALUE, fold its
 * members, then exercise the read/write helper, and return a computed `_BitInt(12)` the harness compares
 * across both binaries on seeded inputs. */
unsigned _BitInt(12) bitint_member_entry(struct bipair s) {
  unsigned _BitInt(12) f = bipair_fold(s);                         /* read members + 12-bit arithmetic */
  unsigned _BitInt(12) g = bipair_rw(f, s.hi);                     /* write members + read-back */
  unsigned _BitInt(12) r = f + g - (unsigned _BitInt(12))7;        /* same-type 12-bit arithmetic (wraps) */
  return r;
}
