/* Address-of a member-array element (#addrofarr) -- the deferred follow-on of #addrof: `&s.arr[i]`,
 * `&s->arr[i]`, and the multi-dimensional `&s.m[i][j]`. The element lands at `&base + member_off +
 * lin*elem_size` (`(char *)s` for a pointer base, `(char *)&s` for a value), reusing the same row-major
 * flatten the member-array LOAD path uses. Both rails resolve it through the shared lvalue/base-pointer
 * logic, Clang-equivalent. (A further descent -- `&s.arr[i].field` array-of-structs, or `&s.ptrarr[i]` --
 * stays a both-rails follow-on.) The writes are IDEMPOTENT so the equivalence harness, which calls the
 * original then the bcir_* twin on the SAME buffer, sees a deterministic result. The entry is POINTER-based
 * (a `struct *` the harness seeds as a buffer); the value-struct case rides parity + the cross-target gate
 * (the generic harness can't seed a by-value struct with array members). */

struct Grid { unsigned a[4]; unsigned m[2][3]; };

static void setk(unsigned *p, unsigned k) { *p = k; }

unsigned ga_value(struct Grid g, unsigned i)         /* &value-struct member-array element (`(char *)&g + ..`) */
{
    setk(&g.a[i & 3u], 7u);
    return g.a[i & 3u];
}

unsigned ga_2d(struct Grid *g, unsigned i, unsigned j)   /* &pointer-struct 2-D member-array element */
{
    setk(&g->m[i & 1u][j % 3u], 3u);
    return g->m[i & 1u][j % 3u];
}

unsigned ga_entry(struct Grid *gp, unsigned i, unsigned j)   /* the entry: pointer member-array elements */
{
    setk(&gp->a[i & 3u], 5u);
    setk(&gp->m[i & 1u][j % 3u], 8u);
    return gp->a[i & 3u] + gp->m[i & 1u][j % 3u] + gp->a[0];
}
