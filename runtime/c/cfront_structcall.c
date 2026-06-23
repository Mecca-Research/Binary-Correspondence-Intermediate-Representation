/* Struct-returning call RESULT used by value (#structcall): calling an in-unit struct-returning function and
 * using the result -- `struct P p = mk(x);`, `g(mk(x))` (pass a call result by value), `mk(x).field` (member
 * of the result), and a passthrough `return mk(x);`. The result is a by-value struct TEMPORARY; both rails
 * previously typed it as a flat `uint32_t` (emitting invalid C `uint32_t t = mk(x);` -- a struct assigned to
 * an int), so any caller using the result build-failed. Now the c.call result keeps the aggregate CType, so it
 * is declared `struct P t = mk(x);` and copies / passes / member-accesses correctly, matching Clang. */

struct P { int x; short y; long z; };

static struct P mk(int a)
{
    struct P p = { a, (short)(a + 1), (long)a * 10 };
    return p;
}

static int use_by_val(struct P p)            /* a struct-by-value parameter consuming a call result */
{
    return p.x + p.y + (int)p.z;
}

static struct P passthru(int a)              /* a function that returns another's struct result */
{
    return mk(a);
}

int structcall(int x)
{
    struct P p = mk(x);                      /* assign a call result to a local */
    int s = p.x + p.y + (int)p.z;
    s += mk(x + 1).x + mk(x + 2).y;          /* member of a call result */
    s += use_by_val(mk(x + 3));              /* pass a call result by value */
    s += passthru(x + 4).x;                  /* member of a passthrough result */
    return s;
}
