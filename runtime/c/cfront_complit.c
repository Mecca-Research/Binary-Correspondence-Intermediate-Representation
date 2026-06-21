/* Compound literals (#complit): a C99 compound literal `( type-name ){ initializer-list }` is an
 * anonymous object of `type` with automatic storage. Both rails materialize a nameless local (`_cl<N>`)
 * and initialize it exactly like a braced declarator -- a struct/union reuses the `= {0}` zero baseline
 * + a store per initialized member (positional or `.field=` designated, any order, with the rest
 * zero-filling); a scalar `(int){v}` copies the single value in. Supported in rvalue position (a
 * by-value struct argument, a scalar value, a member initializer), under `&` (a pointer to the
 * temporary), and with direct postfix on the literal -- `(struct P){...}.field` reads a member off the
 * materialized temporary like any struct lvalue (the parser threads the literal through the postfix
 * tail; the lowering already treats it as a base). Array literals `(int[]){...}` are a deferred
 * follow-on. Driven by a value differential == Clang on both rails. */

struct Pt { int x, y; };
struct Rg { int lo, hi, mid; };
struct Wide { int tag; long val; };

static int pt_sum(struct Pt p)  { return p.x * 100 + p.y; }
static int rg_span(struct Rg r) { return (r.hi - r.lo) * 10 + r.mid; }

int cl_byval(int a, int b)       { return pt_sum((struct Pt){a, b}); }              /* by-value struct arg */
int cl_designated(int a, int b)  { return pt_sum((struct Pt){.y = a, .x = b}); }    /* designators, out of order */
int cl_partial(int a)            { return rg_span((struct Rg){.hi = a, .lo = 1}); } /* .mid zero-fills (§6.7.10) */
int cl_scalar(int a)             { return (int){a * 3} + 7; }                       /* a scalar literal value */
int cl_addr_scalar(int a)        { int *q = &(int){a + 5}; return *q * 2; }         /* &(int){...} -- ptr to temp */
int cl_addr_struct(int a, int b) { struct Pt *q = &(struct Pt){a, b}; return q->x - q->y; } /* &(struct){...} */
int cl_nested(int a)             { return pt_sum((struct Pt){ (int){a}, a + 1 }); } /* a literal inside a literal */

/* direct postfix `.field` on the literal -- read a member straight off the materialized temporary */
int cl_dot(int a, int b)         { return (struct Pt){a, b}.x * 100 + (struct Pt){a, b}.y; }       /* .x / .y */
int cl_dot_desig(int a, int b)   { return (struct Pt){.y = a, .x = b}.x - (struct Pt){.y=a,.x=b}.y; } /* designated then .f */
int cl_dot_part(int a)           { return (struct Rg){.hi = a, .lo = 1}.mid + (struct Rg){.hi=a,.lo=1}.hi; } /* .mid zero-fills */
int cl_dot_wide(int a)           { return (int)(((struct Wide){a, (long)a * 1000000L + a}.val) >> 20); } /* a long field stays 64-bit */
