/* Increment/decrement operators in EXPRESSION position as a VALUE (#incdecexpr): postfix `a++` / `a--` yield
 * the OLD value, prefix `++a` / `--a` yield the NEW value. The lvalue is resolved ONCE (a side-effecting index
 * runs once). Covers a named local, a struct member, an array element, a bitfield, a sub-int (narrowing)
 * local, a pointer step, and the comma operator. The statement forms `a++;` already worked; this adds the
 * value form. (NOTE: adjacent operators like `a - -b` / `a++ + ++a` mis-lex because the preprocessor strips
 * the separating space -- a pre-existing limitation, identical on both rails; not exercised here.) */

struct S { unsigned x, y; };
struct B { unsigned x : 5, y : 3; };

unsigned post_inc(unsigned a) { unsigned x = a++; return x * 100u + a; }   /* old=a, then a+1 */
unsigned pre_inc(unsigned a)  { unsigned x = ++a; return x * 100u + a; }   /* new=a+1 */
unsigned post_dec(unsigned a) { unsigned x = a--; return x * 100u + a; }
int      signed_dec(int a)    { int x = a--; return x * 100 + a; }         /* a signed step */

unsigned mem_inc(unsigned v) {                                            /* a struct member */
    struct S s;
    s.x = v;
    unsigned r = s.x++;
    return r * 100u + s.x;
}

unsigned arr_inc(unsigned i, unsigned v) {                                /* an array element */
    unsigned a[4];
    a[i & 3u] = v;
    unsigned r = a[i & 3u]++;
    return r * 100u + a[i & 3u];
}

unsigned bf_inc(unsigned v) {                                            /* a bitfield (mask/shift RMW) */
    struct B b;
    b.x = v & 31u;
    unsigned r = b.x++;
    return r * 100u + b.x;
}

unsigned narrow_inc(unsigned char c) {                                   /* a sub-int local -> the prefix re-read */
    unsigned d = ++c;
    return d * 1000u + c;
}

unsigned comma_inc(unsigned a, unsigned b) {                            /* the comma operator with a step */
    return (a++, b) + a;
}

unsigned ptr_inc(unsigned *base, unsigned i) {                          /* a pointer steps by one element */
    unsigned *p = base + (i & 3u);
    unsigned *q = p++;
    return (unsigned)(q == base + (i & 3u)) + 2u * (unsigned)(p == base + (i & 3u) + 1);
}
