/* L8: struct return-by-value + a struct-by-value parameter (calling-convention ABI). */
struct vec2 { uint32_t x; uint32_t y; };
struct vec2 l8_swap(struct vec2 p)
{
    struct vec2 r;
    r.x = p.y + 1u;
    r.y = p.x * 3u;
    return r;
}
