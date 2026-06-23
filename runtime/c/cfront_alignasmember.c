/* Over-aligned struct members (#alignasmember): `_Alignas(N)` / C23 `alignas(N)` / `__attribute__((aligned(N)))`
 * on a struct member raises THAT member's alignment (it can only increase it). The member's offset is rounded
 * up to the requested alignment, the aggregate's own alignment becomes the max over its members (so the struct
 * is itself over-aligned), and the size is rounded up to that -- e.g. `char a; _Alignas(16) int b; char c;` puts
 * b at offset 16, c at 20, and sizeof == 32 (align 16), NOT the 12 the natural layout would give. An explicit
 * member alignment survives even inside a packed struct. Both rails compute this layout like Clang (the LAYOUT
 * differential checks each sizeof/offsetof) and a read-modify-write of each member lands at the right offset.
 * The integer-constant form is supported; `_Alignas(type-name)` routes to fallback on both rails (consistently,
 * not silently mis-aligned). */

struct Aligned {
    uint8_t            tag;
    _Alignas(16) int32_t  cacheline;   /* offset 16 (over-aligned), forces struct align 16 */
    uint8_t            mid;
    alignas(8) int64_t    wide;         /* C23 keyword spelling, 8-aligned */
    __attribute__((aligned(4))) int16_t pair[2];  /* GNU spelling, 4-aligned array member */
    uint8_t            z;
};

int64_t aligned_rmw(struct Aligned *p, int32_t x)
{
    p->tag       = (uint8_t)x;
    p->cacheline = x * 3;
    p->mid       = (uint8_t)(x + 1);
    p->wide      = (int64_t)x * 1000;
    p->pair[0]   = (int16_t)x;
    p->pair[1]   = (int16_t)(x - 7);
    p->z         = (uint8_t)(x + 2);
    return (int64_t)p->tag + p->cacheline + p->mid + p->wide
         + p->pair[0] + p->pair[1] + p->z;
}
