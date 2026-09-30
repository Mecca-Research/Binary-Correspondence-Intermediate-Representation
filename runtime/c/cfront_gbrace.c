/* CF-GBRACE, CF-TLS: a file-scope initializer is walked as C walks the current object, and thread storage is kept.
 * Arrays of structs, rows, character tables, a union and a braced scalar are initialized at file scope with nested
 * braces, brace elision and designators, as a local is. The oracle refused a nested brace in a global's initializer
 * (`{{1u, 2u}, {3u, 4u}}`), and both rails sized an unsized global by its top-level entries, so `gb_pts` below was
 * seven elements, not six, and each access past the fourth of the twin's `gb_recs` reached the quarantine. A
 * `_Thread_local` static local was a plain `static` in both emits: one object every thread shares. Everything a
 * function reads is at file scope or its own static, so no local is lent to a call. */
#include <stdint.h>

struct gb_pt { uint32_t x, y; };
struct gb_rec { uint32_t k; uint8_t v[3]; };
struct gb_seg { struct gb_pt a, b; };
union gb_word { uint32_t w; uint8_t b[4]; };

struct gb_pt gb_pts[] = {{1u, 2u}, {3u, 4u}, 5u, 6u, [5].y = 9u};             /* six elements */
uint32_t gb_rows[][3] = {{1u, 2u, 3u}, 4u, 5u, 6u, [3] = {7u}};              /* four rows */
char gb_names[][6] = {"reset", "read", {'w', 'r'}};                           /* three names */
struct gb_rec gb_recs[] = {{1u, {2u, 3u, 4u}}, {5u, "ab"}, 6u, 7u, 8u, 9u};   /* three records */
struct gb_seg gb_segs[2] = {{{1u, 2u}, {3u, 4u}}, {.b.y = 8u, .a = {5u}}};
uint8_t gb_cube[][2][2] = {1u, 2u, 3u, 4u, 5u};                              /* two planes */
union gb_word gb_word = {.b = {1u, 2u, 3u, 4u}};
uint32_t gb_one = {7u};
uint32_t gb_cast = (uint32_t)(sizeof(struct gb_pt) * 3u);                     /* a cast, not a call */
const struct gb_pt gb_origin = {.y = 3u};
_Thread_local uint32_t gb_tls = 5u;

uint32_t gb_pt_at(uint32_t i)
{
    return gb_pts[i % 6u].x * 16u + gb_pts[i % 6u].y;
}

/* each unsized global's extent, as the walk found it */
uint32_t gb_counts(void)
{
    return (uint32_t)(sizeof gb_pts / sizeof gb_pts[0]) * 1000u + (uint32_t)(sizeof gb_rows / sizeof gb_rows[0]) * 100u
           + (uint32_t)(sizeof gb_names / sizeof gb_names[0]) * 10u + (uint32_t)(sizeof gb_recs / sizeof gb_recs[0])
           + (uint32_t)(sizeof gb_cube / sizeof gb_cube[0]) * 10000u;
}

uint32_t gb_row_at(uint32_t i, uint32_t j)
{
    return gb_rows[i % 4u][j % 3u];
}

uint32_t gb_name_at(uint32_t i, uint32_t j)
{
    return (uint32_t)gb_names[i % 3u][j % 6u];
}

uint32_t gb_rec_at(uint32_t i, uint32_t j)
{
    return gb_recs[i % 3u].k * 256u + gb_recs[i % 3u].v[j % 3u];
}

uint32_t gb_seg_at(uint32_t i)
{
    uint32_t n = i & 1u;
    return gb_segs[n].a.x + gb_segs[n].a.y * 3u + gb_segs[n].b.x * 5u + gb_segs[n].b.y * 7u;
}

uint32_t gb_cube_at(uint32_t i)
{
    return gb_cube[i & 1u][(i >> 1) & 1u][(i >> 2) & 1u];
}

uint32_t gb_scalars(void)
{
    return gb_word.b[0] + gb_word.b[3] * 16u + gb_one * 256u + gb_origin.x * 4096u + gb_origin.y * 65536u
           + (gb_cast << 20);
}

/* thread storage: each thread its own object, at its initial value when the thread starts */
uint32_t gb_tls_bump(uint32_t s)
{
    gb_tls += s;
    return gb_tls;
}

uint32_t gb_tls_count(uint32_t s)
{
    static _Thread_local uint32_t n = 3u;
    n += s;
    return n;
}

uint32_t gb_tls_table(uint32_t s)
{
    _Thread_local static uint32_t t[3] = {1u, 2u};
    t[s % 3u] += s;
    return t[0] + t[1] * 3u + t[2] * 5u;
}

uint32_t gb_tls_point(uint32_t s)
{
    static struct gb_pt _Thread_local p = {.y = 4u};
    p.x += s;
    p.y ^= s;
    return p.x + p.y;
}

uint32_t gb_tls_zero(uint32_t s)
{
    static _Thread_local uint64_t z;
    z = z * 3u + s;
    return (uint32_t)(z >> 7) ^ (uint32_t)z;
}

uint32_t gb_entry(uint32_t s)
{
    return gb_pt_at(s) + gb_counts() + gb_row_at(s, s >> 2) + gb_name_at(s, s >> 1) + gb_rec_at(s, s >> 3)
           + gb_seg_at(s) + gb_cube_at(s) + gb_scalars();
}
