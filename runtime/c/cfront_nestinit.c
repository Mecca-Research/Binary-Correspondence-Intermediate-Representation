/* Nested-brace aggregate initialisers (#nestinit): a struct whose member is an ARRAY initialised with a
 * NESTED brace `{ m0, { e0,e1,e2,e3 }, m1 }` -- exercised as a local decl, a compound literal, and a
 * struct return BY VALUE. Each element store lands at `&obj + member_off + i*elem_size` (the same
 * offset-based member store as a runtime `s.arr[i] = v`), riding the `= {0}` zero baseline; a float
 * element converts (double->float) rather than reinterpreting. Both rails match Clang on size + values.
 * (Before this, an array-bearing struct could only be a by-value/pointer PARAMETER -- a local/return of
 * one routed to the LLVM fallback because the nested brace was not parsed.) */

/* (1) a LOCAL struct decl whose `int` array member takes a nested positional brace. */
struct Frame { unsigned id; int samp[4]; unsigned tail; };
unsigned frame_mix(unsigned a) {
  struct Frame f = { a, { (int)a, (int)(a + 1u), -(int)(a & 7u), 5 }, a ^ 0x55u };
  unsigned s = f.id + f.tail;
  for (unsigned k = 0u; k < 4u; k++) s += (unsigned)f.samp[k];
  return s;
}

/* (2) a COMPOUND LITERAL with a nested `float` array brace (each element double->float converted). */
struct Mixed { unsigned n; float w[3]; };
unsigned mixed_use(unsigned a) {
  struct Mixed m = (struct Mixed){ a, { (float)(a & 15u), 0.5f, 2.5f } };
  return m.n + (unsigned)(m.w[0] + m.w[1] + m.w[2]);
}

/* (3) a struct RETURN BY VALUE built from a nested-brace compound literal -- a padding-free struct so the
 * behaviour harness's whole-object memcmp of the returned value is exact. */
struct Pack { unsigned tag; unsigned body[4]; };
struct Pack pack_make(unsigned a) {
  return (struct Pack){ a + 1u, { a, a << 1u, a + 7u, a & 0xFu } };
}
