/* A BITFIELD member assignment used as a VALUE (#bfassignexpr): `(s.bits = v) + 1`, compound `(s.bits += v)`.
 * The value of the assignment is the masked / sign-extended STORED field -- a re-read via bf.get (the plain
 * `=` path already re-reads; the compound path re-reads too because a bitfield narrows to its bit width).
 * Extends the lvalue-as-value forms (#lvassignexpr / #narrowcompound) to bitfield targets. Local structs,
 * deterministic; the masking/sign-extension is exercised (`v` overflows the field width). */

struct BF { unsigned a : 3; unsigned b : 5; int c : 6; };

unsigned bf_plain(unsigned v) {                 /* (s.b = v) masks to 5 bits, used as a value */
    struct BF s; s.a = 0u; s.b = 0u; s.c = 0;
    unsigned r = (s.b = v) + 1u;                /* value == (v & 31u), then + 1 */
    return r + s.b + s.a;
}

unsigned bf_compound(unsigned v) {              /* (s.b += v) masks to 5 bits, used as a value */
    struct BF s; s.a = 0u; s.b = 10u; s.c = 0;
    unsigned r = (s.b += v) * 2u;               /* value == ((10u + v) & 31u), then * 2 */
    return r + s.b;
}

int bf_signed(int v) {                          /* a SIGNED bitfield: (s.c = v) sign-extends from 6 bits */
    struct BF s; s.a = 0u; s.b = 0u; s.c = 0;
    int r = (s.c = v) - 1;                       /* value == sign-extend-6(v), then - 1 */
    return r + s.c;
}

unsigned bf_chain(unsigned v) {                 /* two bitfield assignments combined as values */
    struct BF s; s.a = 0u; s.b = 0u; s.c = 0;
    unsigned r = (s.b = v) + (s.a = v);         /* (v&31) + (v&7) */
    return r + s.a + s.b;
}
