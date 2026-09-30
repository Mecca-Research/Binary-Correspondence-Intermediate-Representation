/* CF-GARRAY: file-scope arrays and character tables pass through cfront as local ones do. A multi-dimensional global
 * passed to a row-pointer parameter `T (*p)[N]` is its first element's address in both emits: each passed it by name,
 * whose type is still `T[A][N]`, where the emitted parameter is the flat `T *`, so Clang and GCC refused the call. A
 * character array sized by its string literal lowers on the twin, which had refused it as a `non-constant enum
 * initializer`. A declaration listing several objects -- arrays, scalars, a pointer, structs, and the struct it
 * defines -- declares each of them on both rails, which had refused the `,`. Everything here is at file scope, so no
 * local array is lent to a call. */
#include <stdint.h>

struct ga_pt { uint32_t x, y; };

uint32_t ga_m[2][3];                             /* a 2-D global, passed to row-pointer parameters */
uint16_t ga_w[3][2][2];                          /* a 3-D one */
char ga_name[] = "bcir";                         /* a character table sized by its literal: 5 */
char ga_pad[8] = "ab";                           /* ... in a larger array, the rest zero */
unsigned char ga_bytes[] = "\x01\x02" "\x03";    /* concatenated, with escapes: 4 */
uint32_t ga_a[3], ga_b[2];                       /* a list of arrays */
uint32_t ga_x, ga_y = 7u, *ga_p, ga_z[2] = {1u, 2u};   /* scalars, a pointer and an initialized array */
static uint64_t ga_s1 = 5u, ga_s2;               /* a static list */
struct ga_pt ga_u, ga_v = {3u, 4u}, ga_arr[2];   /* a list of structs */
static struct ga_ctr { uint32_t hits; uint16_t tag; } ga_c1, ga_c2 = {9u, 1u};   /* the struct it defines */

static uint32_t ga_rows(uint32_t (*p)[3], uint32_t i)
{
    return p[i & 1u][2] * 3u + p[0][i % 3u];
}

static uint32_t ga_cols(uint32_t m[][3], uint32_t i)
{
    m[1][i % 3u] += 1u;
    return m[1][0] + m[1][1] + m[1][2];
}

static uint32_t ga_deep(uint16_t (*p)[2][2], uint32_t i)
{
    return (uint32_t)p[i % 3u][1][i & 1u] + (uint32_t)p[2][0][0];
}

static uint32_t ga_first(uint32_t *p, uint32_t n)
{
    uint32_t k = 0u;
    for (uint32_t i = 0u; i < n; i++)
        k = k * 31u + p[i];
    return k;
}

/* a 2-D and a 3-D global passed where the callee takes a row pointer */
uint32_t ga_pass(uint32_t s)
{
    for (uint32_t i = 0u; i < 2u; i++)
        for (uint32_t j = 0u; j < 3u; j++)
            ga_m[i][j] = s * (i + 1u) + j;
    for (uint32_t i = 0u; i < 3u; i++) {
        ga_w[i][0][0] = (uint16_t)(s + i);
        ga_w[i][0][1] = (uint16_t)(s ^ i);
        ga_w[i][1][0] = (uint16_t)(s >> i);
        ga_w[i][1][1] = (uint16_t)(i * 7u);
    }
    uint32_t k = ga_rows(ga_m, s) + ga_cols(ga_m, s + 1u);
    return k ^ ga_deep(ga_w, s);
}

/* the character tables: sized by their literals, read and written by index */
uint32_t ga_chars(uint32_t s)
{
    uint32_t k = (uint32_t)sizeof ga_name * 100u + (uint32_t)sizeof ga_pad * 10u + (uint32_t)sizeof ga_bytes;
    k += (uint32_t)ga_name[s % 5u] + (uint32_t)ga_pad[s % 8u] + ga_bytes[s % 4u];
    ga_pad[2 + s % 5u] = (char)('a' + s % 26u);
    return k + (uint32_t)ga_pad[2 + (s + 1u) % 5u];
}

/* the lists: each declarator its own object and type */
uint32_t ga_lists(uint32_t s)
{
    ga_a[s % 3u] = s;
    ga_b[s & 1u] = s + 1u;
    ga_x = s ^ ga_y;
    ga_p = s & 1u ? &ga_x : &ga_z[1];
    ga_s2 += (uint64_t)s;
    uint32_t k = ga_a[0] + ga_a[1] * 3u + ga_a[2] * 5u + ga_b[0] + ga_b[1] * 7u + *ga_p;
    k += ga_first(ga_a, 3u) + ga_first(ga_z, 2u);
    return k + (uint32_t)ga_s1 + (uint32_t)(ga_s2 & 0xFFu) + (uint32_t)sizeof ga_z;
}

/* the struct lists, and the objects of the struct a declaration defines */
uint32_t ga_structs(uint32_t s)
{
    ga_u.x = s;
    ga_u.y = s >> 1;
    ga_arr[s & 1u].x = ga_v.x + s;
    ga_arr[1].y = ga_v.y;
    ga_c1.hits += s;
    ga_c1.tag = (uint16_t)(s & 0xFFFFu);
    ga_c2.hits += 1u;
    return ga_u.x + ga_u.y + ga_arr[0].x + ga_arr[1].x + ga_arr[1].y + ga_c1.hits + ga_c1.tag + ga_c2.hits + ga_c2.tag +
           (uint32_t)sizeof ga_c1;
}

uint32_t ga_entry(uint32_t s)
{
    return ga_pass(s) ^ ga_chars(s) ^ ga_lists(s) ^ ga_structs(s);
}
