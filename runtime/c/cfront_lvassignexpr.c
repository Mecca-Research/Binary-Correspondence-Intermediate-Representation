/* Lvalue-assignment-as-a-value, extended forms (#lvassignexpr): an assignment whose target is an ARRAY
 * ELEMENT `a[i]`, a pointer DEREF `*p`, or a NESTED struct member `o.in.x` -- used as a VALUE (a
 * sub-expression: `(a[i]=v)+1`, `(*p=v)*2`, `a[0]=b[0]=v`). The lvalue is resolved ONCE; the expression
 * yields the stored/converted value (a re-read of the once-resolved lvalue, == the assigned value for a
 * non-volatile scalar). Extends the single-level-scalar-member case (#memassignexpr). LOCAL storage so each
 * call has fresh state; deterministic + in-bounds for every input. */

unsigned lv_arr(unsigned i, unsigned v) {           /* an array element as a value: (a[i]=v)+1 */
    unsigned a[8] = {0};
    unsigned r = (a[i & 7u] = v) + 1u;
    return r + a[i & 7u];
}

unsigned lv_deref(unsigned v) {                     /* a pointer deref as a value: (*p=v)*2 */
    unsigned y = 0u;
    unsigned *p = &y;
    unsigned r = (*p = v) * 2u;
    return r + y;
}

struct LVI { unsigned x; };
struct LVO { struct LVI in; };
unsigned lv_nested(unsigned v) {                    /* a nested member as a value: (o.in.x=v)+3 */
    struct LVO o;
    unsigned r = (o.in.x = v) + 3u;
    return r + o.in.x;
}

unsigned lv_compound(unsigned i, unsigned v) {      /* a compound array assignment as a value: (a[i]+=5)*2 */
    unsigned a[8] = {0};
    a[i & 7u] = v;
    unsigned r = (a[i & 7u] += 5u) * 2u;
    return r + a[i & 7u];
}

unsigned lv_chain(unsigned v) {                     /* chained array-element assignment: a[0]=b[0]=v */
    unsigned a[4] = {0};
    unsigned b[4] = {0};
    unsigned r = (a[0] = b[0] = v) + 7u;
    return r + a[0] + b[0];
}
