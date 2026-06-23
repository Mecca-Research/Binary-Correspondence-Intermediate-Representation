/* Member-lvalue assignment as an EXPRESSION (#memassignexpr): an assignment to a plain SCALAR struct member
 * yields a value, so it appears as a sub-expression -- `(p->x = v) + 1`, chained `p->x = p->y = v`, in a
 * condition `if ((p->flag = check()))`, compound `(p->x += 5) * 2`. PR #455 added this only for a NAMED
 * VARIABLE target; a member lvalue used as a value previously fell back on both rails. The value is the
 * STORED/CONVERTED member value (so `(char_member = 300)` yields the (char) value) -- the twin stores then
 * RE-READS the member, matching the oracle's named-local semantics and Clang. (An array element / nested
 * member / deref / bitfield / array-of-structs target as a value stays a follow-on -- both rails fall back.) */

struct Reg {
    int          x;
    signed char  c;        /* a NARROW member -- the assigned value is the (signed char) truncation */
    short        y;
    _Bool        flag;     /* a _Bool member -- normalized to 0/1 on store, the value reads it back */
    int          z;
};

int memassignexpr(struct Reg *p, int v)
{
    int s = (p->x = v) + 1;                 /* member assignment as a value */
    s += (p->c = v * 9);                    /* narrowing: value is (signed char)(v*9) */
    p->y = p->z = v - 3;                    /* chained member assignment (right-associative) */
    s += p->y + p->z;
    if ((p->flag = v & 1))                  /* member assignment in a condition */
        s += 100;
    s += (p->x += 5) * 2;                   /* compound member assignment as a value */
    return s + p->x + p->c;
}
