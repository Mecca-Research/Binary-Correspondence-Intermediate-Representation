/* Funcptr dispatch through a loaded pointer (#fnptrchain): calling a function-pointer struct member
 * reached THROUGH a loaded pointer-to-struct field -- `d->ops->fn(args)`, and the two-hop
 * `s->dev->ops->fn(args)`. The direct `o->fn(args)` already fused member access + call into one
 * `c.call.imember:<field>` claim; this extends the fusion to a funcptr member of a struct reached via a
 * loaded pointer field. The postfix pointer chain from the deref-through slice (#fieldderef) now
 * recognizes a `(` after a member as a fused indirect call whose base is the loaded pointer, so the
 * emit is `ptr->fn(args)`. The oracle already lowered this; this brings the C twin into agreement. Each
 * call is an indirect dispatch -- R18 leaves it an opaque external edge (no callee-resolution edge).
 * The remaining `o.p->fn()` follow-on after slice 2b closed `o.p->v`; driven by a bespoke harness that
 * wires real operation tables. */
typedef int (*binop)(int, int);
struct Ops { binop add; binop sub; binop mul; };
struct Dev { struct Ops *ops; int tag; };
struct Sys { struct Dev *dev; int id; };

int fc_add(struct Dev *d, int a, int b) {
  return d->ops->add(a, b);                         /* funcptr member through a one-hop loaded pointer */
}
int fc_combo(struct Dev *d, int a, int b) {
  return d->ops->add(a, b) * 2 + d->ops->sub(a, b); /* two dispatches composed in one expression */
}
int fc_twohop(struct Sys *s, int a, int b) {
  return s->dev->ops->mul(a, b);                    /* a two-hop pointer chain to the funcptr member */
}
