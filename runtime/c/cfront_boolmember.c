/* _Bool struct-member store normalization (#boolmember): a store into a `_Bool` MEMBER (or a `_Bool`
 * array element) must normalize any nonzero value to 1 (§6.3.1.2). The member/element store path now lands
 * a `_Bool _v = value` temp (which converts), not a raw low-byte copy. cfront_boolnorm.c already covers the
 * LOCAL-scalar case; a struct member / member-array element is a distinct slot path that previously copied
 * the unnormalized byte (so `s.flag = 2` read back as 2, not 1) -- a miscompile vs Clang on both rails. */

/* scalar `_Bool` members, set by member assignment (the `s.m = v` store path). */
struct Flags { unsigned tag; _Bool a; _Bool b; unsigned tail; };
unsigned flags_set(unsigned x) {
  struct Flags f;
  f.tag = x;
  f.a = x;            /* x != 0  -> 1 */
  f.b = x & 2u;       /* (x&2)!=0 -> 1 */
  f.tail = x ^ 3u;
  return f.tag + (unsigned)f.a * 10u + (unsigned)f.b * 100u + f.tail;
}

/* a `_Bool` ARRAY member initialised with a nested brace (each element normalizes) + read back. */
struct Bits { _Bool v[4]; unsigned n; };
unsigned bits_pack(unsigned x) {
  struct Bits s = { { x, x & 4u, 0u, x + 1u }, x };   /* element stores normalize to 0/1 */
  s.v[x & 3u] = x | 1u;                               /* a runtime element store (also normalizes) */
  unsigned r = s.n;
  for (unsigned k = 0u; k < 4u; k++) r += (unsigned)s.v[k] << k;
  return r;
}
