/* Address-of follow-ons (#addroffollow): a PLAIN-base array-of-structs element-field address `&arr[i].field`
 * (the indexed base is a bare array/pointer param or a local array -- NOT a struct member; the member form
 * &s.arr[i].field is already #addrofaos) and a POINTER-member address `&s->ptr`. Both were both-rails
 * fallbacks. The element address is `(char*)arr + i*sizeof(elem) + field_off`; the pointer-member address is
 * `(char*)s + ptr_off`. Write-through the returned address too. (An ARRAY member `&s->arr` stays a fallback --
 * its result is a pointer-TO-ARRAY, which needs `(*)[]` declarators.) */

struct P { unsigned a, b; };

unsigned aos_read(struct P *arr, unsigned i) {       /* &arr[i].field via a POINTER-param base, read */
    unsigned *p = &arr[i & 3u].b;
    return *p + 1u;
}

unsigned aos_write(unsigned i, unsigned v) {         /* &arr[i].field via a LOCAL-array base, write-through */
    struct P arr[4];
    unsigned *p = &arr[i & 3u].a;
    *p = v;                                          /* store through the element-field address */
    return arr[i & 3u].a;                            /* == v, read back the written field */
}

struct S { unsigned *ptr; unsigned n; };

unsigned pm_rw(struct S *s, unsigned *q) {           /* &s->ptr -- the address of a POINTER member, read + write */
    s->ptr = q;
    unsigned **pp = &s->ptr;
    *pp = q + 1;                                     /* store through the pointer-member's address */
    return (unsigned)(s->ptr == q + 1);
}
