/* An ARRAY-OF-STRUCTS element field, or a MEMBER-ARRAY element, used as a VALUE (#aosassignexpr) -- the last
 * lvalue-as-value form. `(a[i].f = v) + 1`, compound `(a[i].f += v) * 2`, member-array `(s.arr[i] = v) + 3`.
 * The lvalue is a STRIDED member: resolved ONCE (index + element stride + field offset), stored, then
 * re-read (the value). Combined with the narrow re-read for a sub-int field. Local storage, deterministic,
 * in-bounds for every input. */

struct AP { unsigned x, y; };

unsigned aos_plain(unsigned i, unsigned v) {        /* an AOS element field as a value */
    struct AP a[4];
    a[i & 3u].x = 0u;
    unsigned r = (a[i & 3u].x = v) + 1u;
    return r + a[i & 3u].x;
}

unsigned aos_compound(unsigned i, unsigned v) {     /* compound on an AOS element field */
    struct AP a[4];
    a[i & 3u].y = 10u;
    unsigned r = (a[i & 3u].y += v) * 2u;
    return r + a[i & 3u].y;
}

struct MA { unsigned arr[4]; };

unsigned ma_elem(unsigned i, unsigned v) {          /* a member-array element as a value */
    struct MA s;
    s.arr[i & 3u] = 0u;
    unsigned r = (s.arr[i & 3u] = v) + 3u;
    return r + s.arr[i & 3u];
}

struct NP { unsigned char c; unsigned x; };

unsigned aos_narrow(unsigned i, unsigned v) {       /* a NARROW field of an AOS element: strided + truncating */
    struct NP a[4];
    a[i & 3u].c = 0u;
    unsigned r = (a[i & 3u].c += v) * 2u;           /* value == (unsigned char)(0 + v), strided + narrow */
    return r + a[i & 3u].c;
}
