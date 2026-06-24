/* The value of a COMPOUND assignment `lv OP= rhs` used as a VALUE is the STORED (narrowed) value, not the
 * un-narrowed binop result (#narrowcompound). When the target is NARROWER than the promoted arithmetic
 * result -- a sub-int `unsigned char` / `unsigned short` member, array element, or deref -- the store
 * truncates, so the value must be a re-read (exactly like the plain `=` path), NOT the raw sum. Returning the
 * sum was a both-rails SILENT MISCOMPILE (clean, wrong) -- untriggered by the fuzzer. The truncation is
 * exercised here (the target starts near its max and `v` overflows the narrow width); unsigned wraparound is
 * well-defined, so the two rails AND Clang agree. (A full-width target needs no re-read -- unchanged.) */

struct NC { unsigned char c; };
unsigned nc_member(unsigned v) {                /* unsigned char member: (s.c += v) truncates to 8 bits */
    struct NC s;
    s.c = 200u;
    unsigned r = (s.c += v) * 3u;               /* value == (unsigned char)(200u + v), then * 3 */
    return r + s.c;
}

unsigned nc_array(unsigned i, unsigned v) {     /* unsigned short array element: truncates to 16 bits */
    unsigned short a[4] = {0};
    a[i & 3u] = 60000u;
    unsigned r = (a[i & 3u] += v) + 7u;         /* value == (unsigned short)(60000u + v) */
    return r + a[i & 3u];
}

unsigned nc_deref(unsigned v) {                 /* unsigned char deref */
    unsigned char y = 250u;
    unsigned char *p = &y;
    unsigned r = (*p += v) * 2u;                /* value == (unsigned char)(250u + v) */
    return r + y;
}

unsigned nc_full(unsigned i, unsigned v) {      /* a FULL-width target: no re-read, value == the sum (control) */
    unsigned a[4] = {0};
    a[i & 3u] = 1000u;
    unsigned r = (a[i & 3u] += v) * 2u;
    return r + a[i & 3u];
}
