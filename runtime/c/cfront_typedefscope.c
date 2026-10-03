/* CF-TYPEDEFSCOPE: a name declared in a block -- a local, a parameter, a loop's own declaration, an enumerator --
 * hides a file-scope typedef of the same name (C11 6.2.1p4) from the end of its declarator (6.2.1p7) to the end of
 * its block, so there it starts no declaration, no cast and no `sizeof` type-name: `T = T * 3u;` assigns, `T * x;`
 * multiplies, `(T) - s` subtracts and `sizeof(B)` sizes the object. Outside that block the name is the typedef again.
 * Both rails read every typedef name through one visibility rule; the twin's token pre-pass keeps the names a block's
 * declarations bind, to the end of that block. Each emit declares its locals up front, so none takes a name the emit
 * spells for something else: a global, a function it calls (as the emit calls it, `bcir_f`), a libc routine (`memcpy`
 * spells every member store), a typedef, a standard type name, the store helper `_v` -- nor another object's name. */
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef uint32_t T;
typedef uint8_t B;
typedef uint32_t (*F)(uint32_t);
typedef struct { uint32_t a; } S;
uint32_t td_g = 5u;
uint32_t td_g_2 = 7u;

static uint32_t td_twice(uint32_t v)
{
    return v * 2u + 1u;
}

/* a local hides T: each statement that starts with it is an expression */
static uint32_t td_assign(uint32_t s)
{
    uint32_t T = s;
    T = T * 3u;
    T += 1u;
    T++;
    return T;
}

/* `T * x;` multiplies (its value discarded), `(T) - s` subtracts, `(T) * x` multiplies, and `sizeof(B)` and
 * `sizeof B` size the uint64_t object, not the uint8_t typedef */
static uint32_t td_operands(uint32_t s)
{
    uint64_t B = s;
    uint32_t T = s + 1u, x = s ^ 3u;
    T * x;
    return (T) - s + (uint32_t)sizeof(B) * 100u + (uint32_t)sizeof B + (T) * x + (uint32_t)B;
}

/* a parameter hides T for the whole body */
static uint32_t td_param(uint32_t T, uint32_t s)
{
    T = T + s;
    return T * 2u;
}

/* an inner block's local hides T to the block's end; after it, T declares again */
static uint32_t td_block(uint32_t s)
{
    {
        uint32_t T = s ^ 9u;
        T <<= 1;
        s = T;
    }
    T v = (T)(s) + 1u;
    return v;
}

/* a loop's own declaration hides T for the loop; after it, T declares again */
static uint32_t td_loop(uint32_t s)
{
    uint32_t r = 0u;
    for (uint32_t T = 0u; T < (s & 7u); T++)
        r += T;
    T z = r * 3u;
    return z;
}

/* a block's enumerator hides T: `T * s` multiplies by 5 */
static uint32_t td_enum(uint32_t s)
{
    enum { T = 5 };
    return T * s + (T);
}

/* a local's own initializer reads the local: its sizeof is 4, not the typedef's 1 */
static uint32_t td_init(uint32_t s)
{
    uint32_t B = (uint32_t)sizeof(B);
    return B + s;
}

/* a local array and a local pointer named T, subscripted and dereferenced, parenthesized too */
static uint32_t td_array(uint32_t s)
{
    uint32_t T[2] = {s, s + 1u};
    T[1] ^= 4u;
    return (T)[1] + T[0];
}

static uint32_t td_pointer(uint32_t s)
{
    uint32_t v = s, *T = &v;
    *T += 2u;
    (*T)++;
    return *T;
}

/* a function pointer named F calls through itself, parenthesized too; after its block, F declares one */
static uint32_t td_call(uint32_t s)
{
    {
        uint32_t (*F)(uint32_t) = td_twice;
        s = F(s) + (F)(s);
    }
    F g = td_twice;
    return g(s);
}

/* a struct's typedef hidden in a block, then declaring again */
static uint32_t td_struct(uint32_t s)
{
    {
        uint32_t S = s;
        S = S + 1u;
        s = S;
    }
    S v;
    v.a = s;
    return v.a;
}

/* a local hides T only in its own function: this one declares with T */
static uint32_t td_next(uint32_t s)
{
    T x = s * 7u;
    return x;
}

/* every local is declared up front in the emitted C, so a block's local may not keep a file-scope name the function
 * reads after the block: a global (`td_g`, 5 again after the block) or a function it calls (`td_twice`) -- nor take,
 * renamed, the name of another global it reads (`td_g_2`) */
static uint32_t td_shadow(uint32_t s)
{
    {
        uint32_t td_g = s;
        s = td_g + 1u;
    }
    {
        uint32_t td_twice = s ^ 1u;
        s = td_twice;
    }
    return s + td_g + td_g_2 + td_twice(s);
}

/* `typeof` of the object, not of the typedef */
static uint32_t td_typeof(uint32_t s)
{
    uint64_t B = s;
    typeof(B) y = B;
    return (uint32_t)sizeof(y) + (uint32_t)y;
}

/* a parameter and a local declared with the typedef they name: the type is read before the declarator (`T T`), and
 * the name is the object's from there on */
static uint32_t td_selfname(T T)
{
    T += 2u;
    return T * 3u;
}

static uint32_t td_selflocal(uint32_t s)
{
    T T = s ^ 5u;
    T <<= 1;
    return T;
}

/* a member named as a typedef is a member, read through its struct: the typedef still declares beside it */
struct td_m {
    uint32_t T;
};

static uint32_t td_member(uint32_t s)
{
    struct td_m x;
    x.T = s;
    T y = x.T + 1u;
    return y;
}

/* a block's local named as a global the function writes after the block: the write is the global's */
uint32_t td_w = 5u;

static uint32_t td_write(uint32_t s)
{
    {
        uint32_t td_w = s ^ 7u;
        s = td_w;
    }
    td_w = s + 1u;
    return s;
}

/* blocks' locals named as the emit spells a function the function calls after them (`bcir_td_twice`) and the library
 * routines it calls (`memcpy`, `memset`, `fabs`) -- `memcpy` also as every member store, `p.a = memcpy;` among them */
struct td_pair { uint32_t a, b; };

static uint32_t td_spelled(uint32_t s)
{
    struct td_pair p, q;
    {
        uint32_t bcir_td_twice = s ^ 3u;
        s = bcir_td_twice + 1u;
    }
    {
        uint32_t memcpy = s >> 2;
        p.a = memcpy;
    }
    p.b = s * 5u;
    memcpy(&q, &p, sizeof q);
    {
        uint32_t memset = s & 7u;
        s = memset + q.a;
    }
    memset(&p, 0, sizeof p);
    {
        uint32_t fabs = s & 15u;
        s = fabs + q.b;
    }
    return td_twice(s) + q.b + p.a + (uint32_t)fabs((double)(s & 0xffu) - 100.0);
}

/* parameters named as the store helper `_v` and as `memcpy`, each stored whole into a member: the emit renames them,
 * or its `{ uint32_t _v = _v; memcpy(...); }` would store the helper itself */
static uint32_t td_helper(uint32_t _v, uint32_t memcpy)
{
    struct td_pair p;
    p.a = _v;
    p.b = memcpy;
    return p.a * 3u + p.b;
}

/* two blocks' `x` beside a local `x_2`, two blocks' `td_k` beside a typedef `td_k_2` the function declares with, and a
 * block's `size_t`: the second of each takes a name no other object and no type has */
typedef struct { uint32_t a; } td_k_2;

static uint32_t td_suffix(uint32_t s)
{
    { uint32_t x = s ^ 1u; s = x; }
    { uint32_t x = s ^ 2u; s = x + 3u; }
    uint32_t x_2 = s + 9u;
    { uint32_t td_k = s ^ 5u; s = td_k; }
    { uint32_t td_k = s * 3u; s = td_k; }
    td_k_2 v;
    v.a = s;
    {
        uint32_t size_t = s ^ 3u;
        s = size_t + 1u;
    }
    return s + x_2 + v.a + (uint32_t)sizeof(s);
}

/* the entry: every function above from one value */
uint32_t typedefscope(uint32_t s)
{
    return td_assign(s) + td_operands(s) + td_param(s, 3u) + td_block(s) + td_loop(s) + td_enum(s) + td_init(s) +
           td_array(s) + td_pointer(s) + td_call(s) + td_struct(s) + td_next(s) + td_shadow(s) + td_typeof(s) +
           td_selfname(s) + td_selflocal(s) + td_member(s) + td_write(s) + td_spelled(s) + td_helper(s, s ^ 0x55u) +
           td_suffix(s);
}
