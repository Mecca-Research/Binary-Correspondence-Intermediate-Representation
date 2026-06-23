/* Array-of-structs members (#aostruct): a struct member that is an ARRAY of structs -- `struct In arr[N];` --
 * accessed element-then-field as `p->arr[i].field` (read AND write, plain and compound). The decisive layout
 * fact: the element STRIDE is `sizeof(struct In)` but the access copies a FIELD-sized value at
 * `member_off(arr) + i*sizeof(In) + offsetof(field)`. Both rails lower this to ONE strided load/store (the
 * member-array claim gains an element-stride imm distinct from the field copy size), so it matches Clang on
 * every target. The nested-brace initializer `{ { {a,b}, {c,d} }, z }` builds each element struct in place. */

struct Pt { int x; short y; signed char tag; float w; };

struct Scene {
    int        n;
    struct Pt  pts[4];        /* the array-of-structs member */
    long       total;
};

double aostruct_rmw(struct Scene *p, int v)
{
    unsigned i = (unsigned)v & 3u;            /* a RUNTIME element index */
    p->n          = v;
    p->pts[0].x   = v;
    p->pts[1].x   = v * 2;
    p->pts[0].x  += 5;                          /* compound store to an element field */
    p->pts[i].y   = (short)(v + 1);
    p->pts[2].tag = (signed char)(v - 3);       /* a signed sub-int element field */
    p->pts[i].w   = (float)v * 0.5f;            /* a float element field */
    p->pts[3].w  += 1.0f;
    p->total      = (long)v * 100;
    return (double)p->n + p->pts[0].x + p->pts[1].x
         + p->pts[i].y + p->pts[2].tag + p->pts[i].w + p->pts[3].w + p->total;
}

/* The nested-brace initializer of an array-of-structs member (built as a by-value local + read back). */
int aostruct_init(int v)
{
    struct Scene s = { v, { { v, (short)v, (signed char)v, 1.0f },
                            { v + 1, (short)(v + 1), (signed char)(v + 1), 2.0f } }, (long)v };
    s.pts[0].x += 10;
    return s.n + s.pts[0].x + s.pts[1].y + s.pts[1].tag + (int)s.total;
}
