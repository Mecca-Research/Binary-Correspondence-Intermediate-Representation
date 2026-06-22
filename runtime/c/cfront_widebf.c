/* Wide bitfields (#widebf): a bitfield in a 64-bit storage unit (`long long` / `unsigned long long : N`,
 * N up to 64). The frontend's bitfield ACCESS was 32-bit-hardcoded -- the read truncated a >32-bit field to
 * 32 bits, and the read-modify-write store used a 32-bit clear mask + a `uint32_t` result, so writing one
 * field zeroed bits 32-63 of the unit and clobbered any adjacent field living there. Both rails now size the
 * unit load + `bf.get`/`bf.set` to the declared type width (64-bit), so a wide field round-trips and
 * preserves its neighbours; the result keeps the promoted type (a >32-bit field stays its 64-bit declared
 * type, int/unsigned cannot hold it). Narrow bitfields (<= 32 bits, int unit) are unchanged. */
struct Wide {
    unsigned long long lo  : 20;
    unsigned long long mid : 40;   /* bits 20-59 -- straddles the 32-bit boundary */
    unsigned long long tail : 4;   /* bits 60-63 -- must survive a write to `mid` */
    long long sgn : 36;            /* a SIGNED wide field (its own unit), sign-extended on read */
    unsigned narrow : 5;           /* a narrow field (<= 32, the unchanged path) */
};
unsigned long long wide_rmw(struct Wide *w, unsigned long long x)
{
    w->lo  = x;
    w->mid = x * 3ull;             /* a value needing > 32 bits */
    w->tail = x;
    w->sgn = -(long long)(x & 0xFFFFull);
    w->narrow = (unsigned)x;
    return w->lo + w->mid + w->tail + (unsigned long long)w->sgn + w->narrow;
}
