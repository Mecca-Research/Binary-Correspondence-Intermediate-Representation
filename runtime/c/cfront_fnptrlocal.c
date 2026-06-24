/* A function-pointer LOCAL VARIABLE `RET (*f)(PARAMS) = fn;` (#fnptrlocal). The oracle parses + lowers it, but
 * the twin's local-decl path only handled funcptr struct MEMBERS and TYPEDEFs -- a funcptr LOCAL fell into the
 * comma-declarator loop and failed ("expected declarator name"), a latent oracle-vs-twin divergence. The
 * funcptr is set to a fixture function and called (an indirect call); the result types by the funcptr's
 * return (#signedfnptr). Self-contained + deterministic. */

static unsigned add1(unsigned x) { return x + 1u; }
static unsigned dbl(unsigned x)  { return x * 2u; }
static int      neg(int x)       { return -x - 1; }

unsigned use_local_fp(unsigned x) {              /* declare, call, reassign, call again */
    unsigned (*f)(unsigned) = add1;
    unsigned r = f(x);
    f = dbl;
    return r * 10u + f(x);
}

unsigned cond_local_fp(unsigned x, unsigned which) {   /* a value-selected funcptr local */
    unsigned (*f)(unsigned) = add1;
    if (which & 1u) f = dbl;
    return f(x);
}

int signed_local_fp(int x) {                     /* a SIGNED-return funcptr local -> arithmetic >> */
    int (*f)(int) = neg;
    int r = f(x);
    return r >> 1;
}
