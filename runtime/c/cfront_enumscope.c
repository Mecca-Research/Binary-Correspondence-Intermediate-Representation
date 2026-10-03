/* CF-ENUMSCOPE: a name declared in a block -- a local, a parameter, a loop's own declaration -- hides a file-scope
 * enumerator of the same name (C11 6.2.1p4) from the end of its declarator (6.2.1p7) to the end of its block; outside
 * that block the name is the enumerator again. Both rails read every enumerator through one visibility rule, so the
 * hidden one is never folded in place of the object that hides it. */
#include <stdint.h>

enum { N = 4, K = 3, W = 2 };

/* a local hides N: the VLA's extent, its sizeof and the sum read the local, never 4 */
static uint32_t es_local(uint32_t s)
{
    uint32_t N = (s & 1u) + 1u;
    uint32_t a[N];
    a[0] = s;
    return a[0] + (uint32_t)sizeof a + N;
}

/* a parameter hides K; the enumerator W stays visible */
static uint32_t es_param(uint32_t K, uint32_t s)
{
    return K * 10u + s + W;
}

/* an inner block's local hides N to the block's end; after it N is the enumerator again */
static uint32_t es_block(uint32_t s)
{
    uint32_t r = N;
    {
        uint32_t N = s ^ 5u;
        r += N;
    }
    return r + N;
}

/* a loop's own declaration hides K for the loop; after it K is the enumerator */
static uint32_t es_loop(uint32_t s)
{
    uint32_t r = 0u;
    for (uint32_t K = 0u; K < (s & 7u); K++)
        r += K;
    return r + K;
}

/* a local's scope begins after its declarator: `sizeof N` in its own initializer is the local's 8, not an int's 4 */
static uint32_t es_self(uint32_t s)
{
    uint64_t N = sizeof N;
    return (uint32_t)N + s;
}

/* a local array hides W: `sizeof W` is the array's size */
static uint32_t es_sizeof(uint32_t s)
{
    uint16_t W[5] = {1u, 2u, 3u, 4u, 5u};
    return (uint32_t)sizeof W + W[s % 5u];
}

/* a parameter hides N in this function only ... */
static uint32_t es_pn(uint32_t N)
{
    return N + 1u;
}

/* ... so at file scope, after it, N is the enumerator: an enumerator's value and a table's dimension */
enum { M = N + 1 };
static const uint32_t es_tab[N] = {7u, 11u, 13u, 17u};

static uint32_t es_after(uint32_t s)
{
    switch (s & 7u) {
    case N:
        return es_tab[s & 3u] + M;
    case K:
        return 100u;
    default:
        return M + (uint32_t)sizeof es_tab;
    }
}

uint32_t enumscope(uint32_t s, uint32_t t)
{
    return es_local(s) + es_param(t, s) + es_block(s) + es_loop(t) + es_self(s) + es_sizeof(t) + es_pn(s) +
           es_after(s ^ t);
}
