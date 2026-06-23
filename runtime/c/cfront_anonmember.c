/* Anonymous struct/union members (#anonmember, C11 6.7.2.1): a `struct {...};` or `union {...};` with NO
 * member name inside a struct -- its members are PROMOTED into the enclosing struct's namespace, so they are
 * read/written directly as `p->x` (not `p->some_name.x`). An anonymous union's members overlap at the union's
 * offset (writing `kind`/`u32` and reading `i32` aliases the bytes); an anonymous struct's members lay out
 * in-place. Both rails FLATTEN the anonymous aggregate's leaf fields into the parent's field table at shifted
 * offsets while the parent's size/alignment still reserves the aggregate as a unit -- so layout matches Clang
 * (sizeof/offsetof) and a read-modify-write of each promoted field lands at the right offset. A NAMED inline
 * aggregate (`struct {...} sub;`) stays a normal nested member (`p->sub.a`). Anonymous members may nest. */

struct Anon {
    uint8_t  tag;
    union {                 /* anonymous union: lo / hi / both alias the same 4 bytes */
        uint32_t both;
        struct { uint16_t lo, hi; };   /* an anonymous STRUCT nested in the anonymous union */
    };
    struct {                /* anonymous struct: a and b lay out in place after the union */
        int32_t a;
        int16_t b;
    };
    struct { int32_t x, y; } pt;       /* a NAMED inline aggregate -> a normal nested member `p->pt.x` */
    uint8_t  z;
};

int64_t anon_rmw(struct Anon *p, int32_t v)
{
    p->tag  = (uint8_t)v;
    p->both = (uint32_t)(v * 7);       /* write the whole 4 bytes ... */
    p->hi   = (uint16_t)(v + 3);       /* ... then overwrite the high half (aliases `both` bits 16-31) */
    p->a    = v - 5;
    p->b    = (int16_t)(v * 2);
    p->pt.x = v + 1;
    p->pt.y = v - 1;
    p->z    = (uint8_t)(v + 9);
    return (int64_t)p->tag + p->both + p->lo + p->hi
         + p->a + p->b + p->pt.x + p->pt.y + p->z;
}
