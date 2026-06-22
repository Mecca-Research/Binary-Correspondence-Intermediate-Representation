/* Packed bitfields (#packedbf): `struct __attribute__((packed))` with bitfields. Packed bitfields pack
 * bit-by-bit with NO storage-unit reservation (Clang/GCC), so a field sits at the running bit cursor and
 * may STRADDLE byte/word boundaries -- e.g. `unsigned char tag; unsigned a:5; unsigned b:9; unsigned char z`
 * is 4 bytes (a/b in bits 8-21, z at byte 3), not the 6 the old `sizeof(T)`-unit-reservation gave. Each
 * field's access unit spans only the bytes it covers (`ceil((bit_off+w)/8)`), read into a uint wide enough
 * to hold it (a field straddling into bits >= 32 needs a 64-bit unit) and written back over only those
 * spanned bytes -- so a read-modify-write of one field preserves its neighbours (incl. a field sharing the
 * straddled bytes). Both rails match Clang on size/offset (the LAYOUT differential) AND on the values. */

struct __attribute__((packed)) Pk {
    unsigned char  tag;
    unsigned       a : 5;     /* bits 8-12  */
    unsigned       b : 9;     /* bits 13-21 (straddles bytes 1-2) */
    unsigned char  mid;
    int            s : 13;    /* a SIGNED packed field, straddles bytes */
    unsigned long long w : 40;/* a WIDE packed field (> 32 bits, 64-bit unit), straddles many bytes */
    unsigned       t : 3;
    unsigned char  z;
};
unsigned long long packed_rmw(struct Pk *p, unsigned long long x)
{
    p->tag = (unsigned char)x;
    p->a   = (unsigned)x;
    p->b   = (unsigned)(x >> 1);
    p->mid = (unsigned char)(x >> 2);
    p->s   = (int)x - 9;
    p->w   = x * 5ull;
    p->t   = (unsigned)x;
    p->z   = (unsigned char)(x + 1u);
    return (unsigned long long)p->tag + p->a + p->b + p->mid
         + (unsigned long long)(long long)p->s + p->w + p->t + p->z;
}
