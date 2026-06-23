/* General address-of `&` (#addrof) -- the address of an lvalue as a value: an out-param, a pointer
 * initializer, a swap target. Was limited to `&local`/`&param`/`&(compound literal)`; `&s->member` and
 * `&*p` were a both-rails divergence -- the ORACLE SILENTLY MISCOMPILED them (`&s->m` computed `&s + off`,
 * the address of the pointer VARIABLE, not `(char *)s + off`; `&*q` took the address of a loaded COPY) while
 * the twin fell back -- and `&arr[i]` failed on both. Now `&var`, `&s.member`, `&s->member`, `&*p`, and
 * `&arr[i]` all lower on both rails through the same base-pointer logic loads/stores use (a POINTER base
 * decays to `(char *)s`, a VALUE base is addressed as `(char *)&s`), Clang-equivalent. (A member-array /
 * array-of-structs element address, and a pointer/array/bit-field member, stay a both-rails follow-on.)
 * The writes here are IDEMPOTENT (`*p = const`) so the equivalence harness -- which calls the original then
 * the bcir_* twin on the SAME buffers -- sees a deterministic result. */

struct Pt { unsigned x, y; };

static void setk(unsigned *p, unsigned k) { *p = k; }     /* an out-param write through a pointer */

unsigned amember(struct Pt s)             { setk(&s.x, 7u);  return s.x + s.y; }     /* &value-struct member */
unsigned apmember(struct Pt *s)           { setk(&s->x, 7u); return s->x + s->y; }   /* &pointer-struct member */
unsigned aderef(unsigned *q)              { setk(&*q, 9u);   return *q; }             /* &*p == p */
unsigned aindex(unsigned *a, unsigned i)  { setk(&a[i & 7u], 5u); return a[i & 7u]; } /* &arr[i] (element addr) */

unsigned addrof_entry(struct Pt p, struct Pt *sp, unsigned *buf, unsigned i)
{
    struct Pt loc = p;
    setk(&loc.y, 11u);                    /* &local.member (value struct) */
    unsigned t = i;
    setk(&t, 3u);                         /* &local scalar */
    setk(&sp->y, 9u);                     /* &s->member (pointer struct) -- the fixed path */
    setk(&buf[i & 7u], 5u);               /* &arr[i] element address */
    setk(&*buf, 2u);                      /* &*p == buf (buf[0]) */
    return loc.x + loc.y + t + sp->x + sp->y + buf[i & 7u] + buf[0];
}
