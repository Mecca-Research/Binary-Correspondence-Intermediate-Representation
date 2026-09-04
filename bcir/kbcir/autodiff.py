"""B3: reverse-mode automatic differentiation as content-addressed graph rewrites.

This is the math oracle for slice B3 of the ML/AI-integration roadmap (see
``docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md``, section 6). It realizes the central B3
thesis in the smallest honest form: a differentiable expression DAG of first-order
primitives, **content-addressed (hash-consed)** so structurally identical
subexpressions are *one shared node*, and reverse-mode AD expressed as a set of
**local backward rewrite rules** -- one adjoint (vector-Jacobian product) rule per
primitive -- that walk the forward DAG and *build* an accumulated adjoint graph.
Because the adjoint accumulation is itself content-addressed, a subexpression
shared in the forward graph contributes a *single shared adjoint* (deduped, not
recomputed) -- a stronger, global form of common-subexpression reuse than a local
per-instruction cache. That global structural sharing is the property the ML/AI
roadmap's B3 closure register highlights as BCIR's distinctive contribution.

TERMINOLOGY. The framing is **content-addressed,
confluent local GRAPH REWRITES** in the tradition of 2-categorical / string-diagram
(PROP / symmetric-monoidal-category) rewriting -- *not* an operadic two-cell account (the established structure is monoidal-category
rewriting). Each primitive's backward rule is a local rewrite; the rules are
*confluent* (the diamond property), so the gradient is independent of the order in
which adjoint contributions are accumulated. One clean way to read reverse mode is
"**linearize then transpose**": evaluate the forward DAG to fix the point of
linearization, take the local linear (Jacobian) map of each primitive at that point,
and transpose the composite linear map -- the adjoint pass below is exactly the
transpose, accumulated by content key.

Intellectual antecedents (cited as direct prior art, whose soundness / confluence /
structural-sharing arguments this module reuses):

  * Alvarez-Picallo, Ghica, Sprunger & Zanasi, "Functorial String Diagrams for
    Reverse-Mode AD" (arXiv:2107.13433, CSL 2023): reverse-mode AD as **local rewrite
    rules on string diagrams**, implemented as double-pushout hypergraph rewriting,
    **proven sound** against reverse-derivative-category semantics, with the
    **diamond/confluence** property (differentiation order does not change the answer)
    and canonical structural **sharing/dedup** falling out of the hypergraph
    representation -- the same content-addressing argument, arrived at independently.
  * Conal Elliott, "The Simple Essence of Automatic Differentiation"
    (arXiv:1804.00746, ICFP 2018): AD as a structure-preserving functor.
  * Radul, Paszke, Frostig, Johnson & Maclaurin, "Decomposing Reverse-Mode AD"
    (arXiv:2105.09469): reverse = forward-mode **linearization then transposition**,
    the most rewrite-like account in the mainstream literature (how JAX implements
    ``grad``).

The correctness gate is a numerical one: every gradient is checked against a
**central-difference finite-difference** gradient within a tolerance (the hard gate),
and against the exact symbolic gradient for closed-form cases. Sharing is *measurable*
(``unique_node_count``); confluence/determinism is *tested* (two accumulation orders,
plus byte-identical replay).

CONTROL FLOW + HIGHER-ORDER (the B3 extension slice). Three additions keep the same
discipline:

  * ``select(cond, a, b)`` -- a differentiable conditional (JAX ``lax.select`` / C ``?:``;
    predicate = sign test of ``cond``'s forward value). Its VJP routes the adjoint entirely
    to the selected branch (0 to the other, 0 to ``cond``) -- the standard a.e. convention.
  * ``unroll_scan`` -- a bounded loop/scan UNROLLED into the same first-order primitives (no
    new functional primitive), so the existing reverse pass differentiates through it:
    reverse-mode over a bounded loop is backprop-through-time.
  * SYMBOLIC backward rules (``_BACKWARD_SYM``) that REWRITE INTO NEW GRAPH NODES instead of
    computing floats, so the gradient is itself a DAG. ``grad_graph`` returns gradient
    *nodes*; ``hessian`` differentiates those nodes AGAIN (reverse-over-reverse) and is gated
    against ``second_difference_hessian``.

HONEST LIMITATION (the precise boundary; faithful to the roadmap's B3 research basis, which cites
arXiv:2107.13433, 1804.00746, 2105.09469). Reverse-over-reverse is sound HERE for one
specific reason: every primitive's VJP is itself expressible in the SAME closed primitive
set (+, -, *, /, neg, dot, select, exp, log, sqrt, tanh, sin, cos), so the gradient graph is an ordinary DAG that can be
differentiated again. This closure is what makes ``hessian`` honest -- and it is exactly
where the general story is known to strain. Two boundaries:

  (a) CLOSURE is required. The admitted transcendental VJPs are now expressed in the
      same vocabulary; a foreign primitive whose VJP is not, or a genuinely dynamic
      higher-order functional / unbounded
      data-dependent recursion, breaks the closure: the gradient would not be a DAG in this
      set and could not be re-differentiated by this machinery. The B3 closure register flags this
      directly -- reverse-derivative categories "do not natively support higher-order
      functions" (arXiv:1910.07065), and higher-order/control-flow/mutation coverage
      is named as the open AD risk. ``unroll_scan`` deliberately handles only BOUNDED loops
      (a finite composition of primitives) for this reason.
  (b) ``select``'s exact second derivative carries a DISTRIBUTIONAL (Dirac-delta) term at
      the switch boundary ``fval[cond] == 0`` -- the derivative of the step. The a.e.
      convention here (matching JAX/PyTorch) DROPS that term. So a ``hessian`` taken through
      a ``select`` is correct ALMOST EVERYWHERE but NOT at the boundary: away from the switch
      it matches the second difference; at/near the switch the true Hessian has a delta the
      numbers cannot see and this code does not claim. A test pins both halves of this (away:
      matches; boundary: caveat asserted, not papered over).

Deterministic, pure-Python, dependency-free. This is a **research organ** and is kept
*cold* (off the hot plan->emit import path; see ``tools/perf/import_graph.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --- the content-addressed differentiable DAG ------------------------------------
#
# A node is identified by a structural CONTENT KEY: (op, operand-content-keys,
# constant payload). Two structurally identical subexpressions therefore hash-cons
# to THE SAME node id -- they are shared, not duplicated. We reuse the spirit of
# bcir.kbcir.egraph's hashconsing (a memo from content key -> id) but specialize it
# to a *differentiable* primitive set with a forward evaluator and a per-primitive
# backward (adjoint) rule.

# Primitive arities. ``const`` / ``var`` are leaves; the rest are first-order ops.
# ``select`` is the differentiable conditional (arity 3: predicate, then-branch, else-branch).
_ARITY = {
    "const": 0,
    "var": 0,
    "neg": 1,
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "select": 3,
    "exp": 1,
    "log": 1,
    "sqrt": 1,
    "tanh": 1,
    "sin": 1,
    "cos": 1,
    "dot": None,
}  # dot is n-ary (variadic)


@dataclass(frozen=True)
class Node:
    """A hash-consed node of the expression DAG.

    ``key`` is the structural content key (op + operand keys + constant), so equal
    structure == equal key == one interned node. ``args`` are operand node ids
    (ints), ``const`` carries a leaf's constant (a float for ``const``; a name for
    ``var``)."""

    nid: int
    op: str
    args: tuple  # operand node ids
    const: object = None
    key: tuple = ()


def _content_key(op: str, args: tuple, const) -> tuple:
    """The deterministic structural content key for a node. Floats are kept as-is
    (exact value identity); the key is a plain tuple so it is hashable + reproducible
    and two builders produce byte-identical keys for the same structure."""
    return (op, const, tuple(args))


class Tape:
    """A content-addressed builder for the differentiable DAG (the hash-cons store).

    Every constructor interns by content key, so a subexpression that appears twice
    -- e.g. ``t = mul(a, b)`` used in both ``mul(t, c)`` and ``mul(t, d)`` -- is a
    SINGLE node referenced twice, never two copies. ``unique_node_count`` exposes how
    many distinct nodes the DAG holds, which is what makes the structural sharing
    *measurable* (and strictly fewer than a naive tree's node count)."""

    def __init__(self) -> None:
        self._nodes: list[Node] = []
        self._memo: dict[tuple, int] = {}  # content key -> node id (the hash-cons)

    # -- the interning core --
    def _intern(self, op: str, args: tuple, const=None) -> int:
        key = _content_key(op, args, const)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        nid = len(self._nodes)
        self._nodes.append(Node(nid, op, tuple(args), const, key))
        self._memo[key] = nid
        return nid

    def node(self, nid: int) -> Node:
        return self._nodes[nid]

    # -- leaves --
    def const(self, c: float) -> int:
        """A literal constant. Its gradient w.r.t. any input is zero (a leaf rule)."""
        return self._intern("const", (), float(c))

    def var(self, name: str) -> int:
        """A named input variable -- a differentiation target."""
        return self._intern("var", (), name)

    # -- first-order primitives (each carries a local backward rule, see _BACKWARD) --
    def add(self, a: int, b: int) -> int:
        return self._intern("add", (a, b))

    def sub(self, a: int, b: int) -> int:
        return self._intern("sub", (a, b))

    def mul(self, a: int, b: int) -> int:
        return self._intern("mul", (a, b))

    def neg(self, a: int) -> int:
        return self._intern("neg", (a,))

    def div(self, a: int, b: int) -> int:
        """a / b (b must be non-zero at the linearization point)."""
        return self._intern("div", (a, b))

    def exp(self, a: int) -> int:
        return self._intern("exp", (a,))

    def log(self, a: int) -> int:
        return self._intern("log", (a,))

    def sqrt(self, a: int) -> int:
        return self._intern("sqrt", (a,))

    def tanh(self, a: int) -> int:
        return self._intern("tanh", (a,))

    def sin(self, a: int) -> int:
        return self._intern("sin", (a,))

    def cos(self, a: int) -> int:
        return self._intern("cos", (a,))

    # -- the control-flow primitive: a differentiable conditional (JAX lax.select / C ?:) --
    def select(self, cond: int, a: int, b: int) -> int:
        """select(cond, a, b) = a if fval[cond] > 0 else b -- the differentiable
        conditional / ``where``.

        Semantics match JAX ``lax.select`` and the C ternary ``?:``: the predicate is the
        SIGN TEST of ``cond``'s forward value (``cond > 0`` picks ``a``, else ``b``). It is
        a *piecewise-constant selector*: on either side of the switch boundary the output
        equals one branch exactly, so its derivative w.r.t. the selected branch is 1, w.r.t.
        the other branch is 0, and w.r.t. ``cond`` is 0 almost everywhere (the standard a.e.
        VJP convention -- see :func:`_bw_select`). This is what lets bounded data-dependent
        control flow enter the content-addressed DAG without leaving the first-order
        primitive set: a ``select`` is just one more local rewrite rule."""
        return self._intern("select", (cond, a, b))

    # -- the composite that ties to the AI substrate: a dot product (sum of products) --
    def dot(self, us: tuple, vs: tuple) -> int:
        """dot(u, v) = sum_i u_i * v_i -- a sum of products, the scalar core of the A1.2
        quantized dot / B1 gem.matmul (here exact float, the math oracle path). Built as
        a single variadic primitive so its adjoint is one local rule (grad_u_i += g*v_i,
        grad_v_i += g*u_i) rather than a hand-unrolled tree -- the rule that ties B3 to
        the matmul substrate."""
        us, vs = tuple(us), tuple(vs)
        if len(us) != len(vs):
            raise ValueError(f"dot length mismatch: {len(us)} != {len(vs)}")
        # args = u-ids followed by v-ids; const records the split so backward can recover it.
        return self._intern("dot", us + vs, ("dot", len(us)))

    @property
    def unique_node_count(self) -> int:
        """How many distinct (hash-consed) nodes the DAG holds. Strictly fewer than a
        naive tree would materialize whenever a subexpression is shared -- the
        content-addressing dedup made measurable."""
        return len(self._nodes)

    def node_count(self) -> int:
        return len(self._nodes)


# --- forward evaluation ----------------------------------------------------------


def evaluate(tape: Tape, root: int, env: dict) -> float:
    """Evaluate the DAG rooted at ``root`` under ``env`` (var name -> float).
    Deterministic; memoized over node ids so a shared subexpression is evaluated once
    (the forward analogue of the shared-adjoint property)."""
    cache: dict[int, float] = {}
    for nid in _topo_order(tape, root):
        n = tape.node(nid)
        op = n.op
        if op == "const":
            cache[nid] = n.const
        elif op == "var":
            if n.const not in env:
                raise KeyError(f"unbound var {n.const!r}")
            cache[nid] = float(env[n.const])
        elif op == "neg":
            cache[nid] = -cache[n.args[0]]
        elif op == "add":
            cache[nid] = cache[n.args[0]] + cache[n.args[1]]
        elif op == "sub":
            cache[nid] = cache[n.args[0]] - cache[n.args[1]]
        elif op == "mul":
            cache[nid] = cache[n.args[0]] * cache[n.args[1]]
        elif op == "div":
            cache[nid] = cache[n.args[0]] / cache[n.args[1]]
        elif op == "exp":
            cache[nid] = math.exp(cache[n.args[0]])
        elif op == "log":
            cache[nid] = math.log(cache[n.args[0]])
        elif op == "sqrt":
            cache[nid] = math.sqrt(cache[n.args[0]])
        elif op == "tanh":
            cache[nid] = math.tanh(cache[n.args[0]])
        elif op == "sin":
            cache[nid] = math.sin(cache[n.args[0]])
        elif op == "cos":
            cache[nid] = math.cos(cache[n.args[0]])
        elif op == "select":
            # predicate is the SIGN TEST of the cond node's forward value (JAX lax.select).
            cond, a, b = n.args
            cache[nid] = cache[a] if cache[cond] > 0 else cache[b]
        elif op == "dot":
            k = n.const[1]
            us, vs = n.args[:k], n.args[k:]
            cache[nid] = sum(cache[u] * cache[v] for u, v in zip(us, vs))
        else:  # pragma: no cover - defensive; _ARITY is the source of truth
            raise ValueError(f"unknown op {op!r}")
    return cache[root]


def _topo_order(tape: Tape, root: int) -> list[int]:
    """Node ids reachable from ``root`` in dependency order (operands before users).
    Deterministic (iterative post-order over the operand lists); shared nodes appear
    exactly once -- the single visit that the shared-adjoint accumulation relies on."""
    order: list[int] = []
    seen: set[int] = set()
    # iterative post-order: push (nid, expanded?) frames so a node is emitted only
    # after all its operands, and exactly once.
    stack = [(root, False)]
    while stack:
        nid, expanded = stack.pop()
        if expanded:
            if nid not in seen:
                seen.add(nid)
                order.append(nid)
            continue
        if nid in seen:
            continue
        stack.append((nid, True))
        for a in tape.node(nid).args:
            if a not in seen:
                stack.append((a, False))
    return order


# --- reverse-mode AD as local backward rewrite rules -----------------------------
#
# Each primitive z = op(a, b, ...) carries a LOCAL backward rule: given the adjoint
# (cotangent) grad_z flowing into z, it contributes to each operand's adjoint. This
# is the per-primitive rewrite rule -- the vector-Jacobian product / transpose of the
# primitive's local linear map. Reverse mode is: evaluate forward (fix the
# linearization point), then visit nodes in REVERSE topological order, applying each
# node's local rule to accumulate operand adjoints. Accumulation is BY NODE ID, so a
# subexpression shared in the forward graph receives a SINGLE accumulated adjoint
# (the global structural sharing / dedup) and its rule fires once.
#
# A rule is a function (forward_values, grad_z) -> [(operand_nid, contribution), ...].
# It is "local" in the literal sense: it sees only the node's own forward operand
# values and the incoming adjoint -- never the rest of the graph. That locality is
# what makes the rules confluent (composable in any order).


def _bw_neg(fvals, args, const, gz):
    # z = -a  ->  grad_a += -gz
    return [(args[0], -gz)]


def _bw_add(fvals, args, const, gz):
    # z = a + b  ->  grad_a += gz ; grad_b += gz
    return [(args[0], gz), (args[1], gz)]


def _bw_sub(fvals, args, const, gz):
    # z = a - b  ->  grad_a += gz ; grad_b += -gz
    return [(args[0], gz), (args[1], -gz)]


def _bw_mul(fvals, args, const, gz):
    # z = a * b  ->  grad_a += gz * b ; grad_b += gz * a (the canonical product rule)
    a, b = args
    return [(a, gz * fvals[b]), (b, gz * fvals[a])]


def _bw_div(fvals, args, const, gz):
    # z = a / b  ->  grad_a += gz / b ; grad_b += -gz * a / b**2 (the quotient rule)
    a, b = args
    bv = fvals[b]
    return [(a, gz / bv), (b, -gz * fvals[a] / (bv * bv))]


def _bw_dot(fvals, args, const, gz):
    # z = sum_i u_i * v_i  ->  grad_u_i += gz * v_i ; grad_v_i += gz * u_i
    # (the dot's local adjoint -- the same product-rule shape, vectorized; the tie to
    # the matmul substrate).
    k = const[1]
    us, vs = args[:k], args[k:]
    out = []
    for u, v in zip(us, vs):
        out.append((u, gz * fvals[v]))
        out.append((v, gz * fvals[u]))
    return out


def _bw_select(fvals, args, const, gz):
    # z = a if cond > 0 else b. The VJP of a piecewise-constant selector (the standard
    # a.e. convention, identical to JAX/PyTorch ``where``): the incoming adjoint flows
    # ENTIRELY to the SELECTED branch, 0 to the other, and 0 to ``cond``. The predicate is
    # locally constant on each side of the switch, so its local Jacobian w.r.t. cond is
    # zero almost everywhere; the measure-zero boundary delta (an exact distributional
    # term at fval[cond] == 0) is DROPPED -- exactly what the mainstream frameworks do, and
    # the boundary caveat the honest-limitation note pins.
    cond, a, b = args
    if fvals[cond] > 0:
        return [(a, gz), (b, 0.0), (cond, 0.0)]
    return [(a, 0.0), (b, gz), (cond, 0.0)]


def _bw_exp(fvals, args, const, gz):
    return [(args[0], gz * math.exp(fvals[args[0]]))]


def _bw_log(fvals, args, const, gz):
    return [(args[0], gz / fvals[args[0]])]


def _bw_sqrt(fvals, args, const, gz):
    return [(args[0], gz / (2.0 * math.sqrt(fvals[args[0]])))]


def _bw_tanh(fvals, args, const, gz):
    value = math.tanh(fvals[args[0]])
    return [(args[0], gz * (1.0 - value * value))]


def _bw_sin(fvals, args, const, gz):
    return [(args[0], gz * math.cos(fvals[args[0]]))]


def _bw_cos(fvals, args, const, gz):
    return [(args[0], -gz * math.sin(fvals[args[0]]))]


# the rewrite-rule table: one local backward rule per primitive (leaves have none --
# const and var are differentiation boundaries, const contributing zero by definition).
_BACKWARD = {
    "neg": _bw_neg,
    "add": _bw_add,
    "sub": _bw_sub,
    "mul": _bw_mul,
    "div": _bw_div,
    "dot": _bw_dot,
    "select": _bw_select,
    "exp": _bw_exp,
    "log": _bw_log,
    "sqrt": _bw_sqrt,
    "tanh": _bw_tanh,
    "sin": _bw_sin,
    "cos": _bw_cos,
}


@dataclass(frozen=True)
class GradResult:
    """The gradient of an output w.r.t. requested inputs, plus the cheap-gradient
    accounting that demonstrates reverse mode touches O(1)x the forward op count."""

    grads: dict  # var name -> d(output)/d(var)
    value: float  # the forward value of the output (computed en route)
    forward_ops: int  # primitive ops in the forward DAG (reachable, deduped)
    backward_ops: int  # primitive rule firings in the reverse pass

    @property
    def cheap_gradient_ratio(self) -> float:
        """backward / forward op count. The Baur-Strassen "cheap gradient principle"
        says this is a small constant (O(1)), not growing with graph size -- exposed
        to *demonstrate*, not to assert a hard bound."""
        return self.backward_ops / max(1, self.forward_ops)


def _accumulate_adjoints(tape: Tape, root: int, order, fvals: dict) -> tuple[dict, int]:
    """The reverse pass: visit ``order`` (a reverse-topological order of the forward
    DAG) and apply each node's local backward rule, accumulating adjoints BY NODE ID.

    Because accumulation is keyed on node id, a forward-shared node has ONE adjoint
    slot: its incoming contributions sum into that slot and its own rule fires exactly
    once (the global, content-addressed adjoint sharing). ``order`` is a parameter so a
    test can feed a DIFFERENT valid reverse order and confirm the result is identical
    (confluence / the diamond property)."""
    adj: dict[int, float] = {root: 1.0}  # seed: d(output)/d(output) = 1
    firings = 0
    for nid in order:
        gz = adj.get(nid, 0.0)
        if gz == 0.0:
            continue  # no adjoint flows here -> rule is a no-op
        n = tape.node(nid)
        rule = _BACKWARD.get(n.op)
        if rule is None:
            continue  # a leaf (const/var): differentiation boundary
        firings += 1
        for operand, contribution in rule(fvals, n.args, n.const, gz):
            adj[operand] = adj.get(operand, 0.0) + contribution
    return adj, firings


def _reverse_order(order: list[int]) -> list[int]:
    """A valid reverse-topological visiting order: the forward topo order reversed
    (users before operands). Any order satisfying "a node is visited before its
    operands" is valid and -- by confluence -- yields the same adjoints."""
    return list(reversed(order))


def reverse_orders(tape: Tape, output: int) -> tuple[list[int], list[int]]:
    """Two DIFFERENT but both-valid reverse-topological visiting orders for the DAG
    rooted at ``output`` (each visits a node before any of its operands). Confluence
    (the diamond property) says :func:`grad` returns the same adjoints under either --
    which a test asserts. The two are derived independently: one from the reversed
    forward topo order, one by sorting on node DEPTH descending (deepest-from-root
    first), so they genuinely differ in the interior order whenever the DAG is not a
    chain."""
    topo = _topo_order(tape, output)
    order_a = _reverse_order(topo)

    depth: dict[int, int] = {}
    for nid in topo:  # operands precede users in topo, so this fills bottom-up
        args = tape.node(nid).args
        depth[nid] = 0 if not args else 1 + max(depth[a] for a in args)
    # depth descending, node id as a stable tie-break: a node's operands are strictly
    # shallower, so they always come later -> a valid reverse-topological order.
    order_b = sorted(topo, key=lambda nid: (-depth[nid], nid))
    return order_a, order_b


def grad(tape: Tape, output: int, inputs, *, order: list[int] | None = None) -> GradResult:
    """Reverse-mode gradient of ``output`` w.r.t. each name in ``inputs``, computed by
    applying the per-primitive local backward rewrite rules over the content-addressed
    DAG.

    Steps (linearize-then-transpose, read as a rewrite):
      1. forward-evaluate to fix the linearization point (the operand values each local
         Jacobian needs);
      2. visit nodes in reverse-topological order, firing each primitive's local adjoint
         rule and accumulating contributions BY NODE ID (the transpose of the composite
         linear map; shared nodes => one shared adjoint);
      3. read off the adjoint at each requested input ``var``.

    ``inputs`` is an iterable of var names *or* an ``env`` dict (its keys are used);
    ``env`` for the forward values is taken from ``inputs`` if it is a dict, else must
    be supplied via ``grad_at``. ``order`` lets a caller inject an alternative valid
    reverse order to exercise confluence."""
    if not isinstance(inputs, dict):
        raise TypeError("grad() needs an env dict (var name -> value); use grad_at for names + env")
    env = inputs
    topo = _topo_order(tape, output)
    fvals = _forward_values(tape, topo, env)
    rev = order if order is not None else _reverse_order(topo)
    adj, firings = _accumulate_adjoints(tape, output, rev, fvals)
    # map each var name back to the node id that carries it, read its adjoint (0 if the
    # input never reaches the output -- an unused input has zero gradient).
    name_to_nid = {tape.node(nid).const: nid for nid in topo if tape.node(nid).op == "var"}
    grads = {name: adj.get(name_to_nid.get(name, -1), 0.0) for name in env}
    forward_ops = sum(1 for nid in topo if tape.node(nid).op in _BACKWARD)
    return GradResult(
        grads=grads, value=fvals[output], forward_ops=forward_ops, backward_ops=firings
    )


def grad_at(tape: Tape, output: int, env: dict, *, order: list[int] | None = None) -> GradResult:
    """Explicit (output, env) form of :func:`grad` -- gradient at the point ``env``."""
    return grad(tape, output, env, order=order)


def _forward_values(tape: Tape, topo: list[int], env: dict) -> dict:
    """Forward values for every node in ``topo`` (the linearization point). Separated
    from :func:`evaluate` so the reverse pass can reuse the per-node table."""
    cache: dict[int, float] = {}
    for nid in topo:
        n = tape.node(nid)
        op = n.op
        if op == "const":
            cache[nid] = n.const
        elif op == "var":
            if n.const not in env:
                raise KeyError(f"unbound var {n.const!r}")
            cache[nid] = float(env[n.const])
        elif op == "neg":
            cache[nid] = -cache[n.args[0]]
        elif op == "add":
            cache[nid] = cache[n.args[0]] + cache[n.args[1]]
        elif op == "sub":
            cache[nid] = cache[n.args[0]] - cache[n.args[1]]
        elif op == "mul":
            cache[nid] = cache[n.args[0]] * cache[n.args[1]]
        elif op == "div":
            cache[nid] = cache[n.args[0]] / cache[n.args[1]]
        elif op == "exp":
            cache[nid] = math.exp(cache[n.args[0]])
        elif op == "log":
            cache[nid] = math.log(cache[n.args[0]])
        elif op == "sqrt":
            cache[nid] = math.sqrt(cache[n.args[0]])
        elif op == "tanh":
            cache[nid] = math.tanh(cache[n.args[0]])
        elif op == "sin":
            cache[nid] = math.sin(cache[n.args[0]])
        elif op == "cos":
            cache[nid] = math.cos(cache[n.args[0]])
        elif op == "select":
            cond, a, b = n.args
            cache[nid] = cache[a] if cache[cond] > 0 else cache[b]
        elif op == "dot":
            k = n.const[1]
            us, vs = n.args[:k], n.args[k:]
            cache[nid] = sum(cache[u] * cache[v] for u, v in zip(us, vs))
        else:  # pragma: no cover
            raise ValueError(f"unknown op {op!r}")
    return cache


# --- the finite-difference correctness gate --------------------------------------


def finite_difference_grad(tape: Tape, output: int, env: dict, *, eps: float = 1e-6) -> dict:
    """Central-difference numerical gradient: for each var, (f(x+eps) - f(x-eps)) / (2 eps).
    This is the HARD correctness gate -- the analytically computed :func:`grad` must match
    this within a tolerance, on every function tested (including the fuzz)."""
    out: dict[str, float] = {}
    for name in env:
        plus = dict(env)
        plus[name] = env[name] + eps
        minus = dict(env)
        minus[name] = env[name] - eps
        out[name] = (evaluate(tape, output, plus) - evaluate(tape, output, minus)) / (2 * eps)
    return out


def gradients_match(a: dict, b: dict, *, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Whether two gradient dicts agree within a relative+absolute tolerance (used to
    compare analytic grad vs finite-difference / symbolic)."""
    if set(a) != set(b):
        return False
    for name in a:
        if abs(a[name] - b[name]) > atol + rtol * abs(b[name]):
            return False
    return True


def max_grad_error(a: dict, b: dict) -> float:
    """The largest absolute disagreement between two gradient dicts (for reporting the
    exact failing case rather than just a pass/fail)."""
    return max((abs(a[name] - b.get(name, 0.0)) for name in a), default=0.0)


# --- a tiny helper to count a naive (un-shared) tree size, for the dedup assertion ---


def naive_tree_node_count(tape: Tape, root: int) -> int:
    """How many nodes a NAIVE tree (no sharing) would materialize for ``root`` -- i.e.
    expand every shared subexpression into its own copy. Compared against
    ``tape.unique_node_count`` to *measure* the content-addressing dedup: whenever a
    subexpression is shared, the hash-consed DAG holds strictly fewer nodes."""
    memo: dict[int, int] = {}

    def count(nid: int) -> int:
        if nid in memo:
            return memo[nid]
        n = tape.node(nid)
        total = 1 + sum(count(a) for a in n.args)
        memo[nid] = total
        return total

    return count(root)


# --- bounded control flow: unrolling a scan into the content-addressed DAG -----------
#
# A bounded loop / scan is differentiated by UNROLLING it into the existing first-order
# primitive set -- no new functional primitive, no closure capture in the IR. The
# unrolled body is plain add/sub/mul/div/neg/dot/select, so the EXISTING reverse pass
# differentiates straight through it: reverse-mode through a bounded loop is exactly
# backprop-through-time (BPTT). Because the builder is content-addressed, any step body
# that is structurally identical across iterations is hash-consed to one shared node,
# and -- by the global shared-adjoint property -- contributes a single shared adjoint.


def unroll_scan(tape: Tape, step, init: int, xs: tuple) -> int:
    """Fold ``carry = step(tape, carry, x)`` over ``xs`` and return the final carry node id.

    ``step(tape, carry_nid, x_nid) -> nid`` is a Python callable that builds ONE step of a
    recurrence out of existing primitives (e.g. ``carry * w + x``); ``init`` is the initial
    carry node id; ``xs`` is a tuple of input node ids fed one per step. This is the honest
    way to put a bounded loop into the DAG: we unroll the loop body into ordinary nodes, so

      d(final_carry)/d(input)  ==  reverse-mode AD over the unrolled DAG  ==  BPTT,

    with no extra machinery -- :func:`grad` already differentiates the result, and the
    finite-difference gate already checks it. (Unbounded / data-dependent trip counts are
    out of scope here: that is the genuinely-higher-order case the limitation note flags;
    a *bounded* scan is just a finite composition of primitives.)"""
    carry = init
    for x in xs:
        carry = step(tape, carry, x)
    return carry


# --- higher-order: SYMBOLIC (graph-valued) backward rules ----------------------------
#
# To get SECOND derivatives honestly, the backward pass must REWRITE INTO NEW GRAPH NODES
# rather than computing floats -- this is the literal "AD as graph rewrites" thesis. A
# symbolic adjoint rule for z = op(a, b, ...) takes the tape, the operand node ids, and
# the incoming-adjoint NODE id ``gz`` and BUILDS the contribution to each operand's adjoint
# as a new Tape node (e.g. for mul(a, b): grad_a contribution node = tape.mul(gz, b)). The
# accumulated gradient is then an ordinary node in the SAME DAG, in the SAME closed
# primitive set -- so it can be fed back into the very same machinery and differentiated
# AGAIN (reverse-over-reverse), which is what makes :func:`hessian` sound here. The closure
# property -- every primitive's VJP is expressible in {+, -, *, /, neg, dot, select} -- is
# exactly the boundary the honest-limitation note pins (see the module docstring).
#
# A symbolic rule mirrors its numeric twin in _BACKWARD but returns
# [(operand_nid, contribution_nid), ...] -- node ids, not floats.


def _sbw_neg(tape, args, const, gz):
    # z = -a  ->  grad_a += -gz
    return [(args[0], tape.neg(gz))]


def _sbw_add(tape, args, const, gz):
    # z = a + b  ->  grad_a += gz ; grad_b += gz
    return [(args[0], gz), (args[1], gz)]


def _sbw_sub(tape, args, const, gz):
    # z = a - b  ->  grad_a += gz ; grad_b += -gz
    return [(args[0], gz), (args[1], tape.neg(gz))]


def _sbw_mul(tape, args, const, gz):
    # z = a * b  ->  grad_a += gz * b ; grad_b += gz * a (product rule, as graph nodes)
    a, b = args
    return [(a, tape.mul(gz, b)), (b, tape.mul(gz, a))]


def _sbw_div(tape, args, const, gz):
    # z = a / b  ->  grad_a += gz / b ; grad_b += -(gz * a) / (b * b)  (quotient rule).
    # Built ENTIRELY from the closed primitive set so the gradient graph is differentiable
    # again: neg(div(mul(gz, a), mul(b, b))).
    a, b = args
    grad_a = tape.div(gz, b)
    grad_b = tape.neg(tape.div(tape.mul(gz, a), tape.mul(b, b)))
    return [(a, grad_a), (b, grad_b)]


def _sbw_dot(tape, args, const, gz):
    # z = sum_i u_i * v_i  ->  grad_u_i += gz * v_i ; grad_v_i += gz * u_i (as graph nodes).
    k = const[1]
    us, vs = args[:k], args[k:]
    out = []
    for u, v in zip(us, vs):
        out.append((u, tape.mul(gz, v)))
        out.append((v, tape.mul(gz, u)))
    return out


def _sbw_select(tape, args, const, gz):
    # z = a if cond > 0 else b. The symbolic VJP keeps the SAME a.e. convention as the
    # numeric rule, but expresses the branch routing AS A select node so it stays a
    # graph-valued, re-differentiable expression: grad_a = select(cond, gz, 0),
    # grad_b = select(cond, 0, gz), grad_cond = 0. (The boundary delta is dropped here too;
    # see the honest-limitation note -- the SECOND derivative through a select is a.e.
    # correct but misses the distributional term at fval[cond] == 0.)
    cond, a, b = args
    zero = tape.const(0.0)
    grad_a = tape.select(cond, gz, zero)
    grad_b = tape.select(cond, zero, gz)
    return [(a, grad_a), (b, grad_b), (cond, zero)]


def _sbw_exp(tape, args, const, gz):
    return [(args[0], tape.mul(gz, tape.exp(args[0])))]


def _sbw_log(tape, args, const, gz):
    return [(args[0], tape.div(gz, args[0]))]


def _sbw_sqrt(tape, args, const, gz):
    denominator = tape.mul(tape.const(2.0), tape.sqrt(args[0]))
    return [(args[0], tape.div(gz, denominator))]


def _sbw_tanh(tape, args, const, gz):
    value = tape.tanh(args[0])
    derivative = tape.sub(tape.const(1.0), tape.mul(value, value))
    return [(args[0], tape.mul(gz, derivative))]


def _sbw_sin(tape, args, const, gz):
    return [(args[0], tape.mul(gz, tape.cos(args[0])))]


def _sbw_cos(tape, args, const, gz):
    return [(args[0], tape.neg(tape.mul(gz, tape.sin(args[0]))))]


# the SYMBOLIC rewrite-rule table -- one graph-valued backward rule per primitive,
# mirroring _BACKWARD. Each builds new Tape nodes in the SAME closed primitive set, so the
# gradient graph is an ordinary DAG that can be differentiated again.
_BACKWARD_SYM = {
    "neg": _sbw_neg,
    "add": _sbw_add,
    "sub": _sbw_sub,
    "mul": _sbw_mul,
    "div": _sbw_div,
    "dot": _sbw_dot,
    "select": _sbw_select,
    "exp": _sbw_exp,
    "log": _sbw_log,
    "sqrt": _sbw_sqrt,
    "tanh": _sbw_tanh,
    "sin": _sbw_sin,
    "cos": _sbw_cos,
}


def grad_graph(tape: Tape, output: int, inputs) -> dict:
    """SYMBOLIC reverse-mode gradient: return, for each input var name, the NODE ID of its
    gradient EXPRESSION (not a float).

    This is reverse mode done as pure graph rewriting: seed the output adjoint with a
    ``const(1.0)`` node, visit the forward DAG in reverse-topological order, and for each
    node fire its SYMBOLIC backward rule (:data:`_BACKWARD_SYM`), BUILDING new Tape nodes
    for each operand's adjoint contribution. Multiple contributions into the same node are
    combined with ``tape.add`` nodes (kept content-addressed/deterministic, so the gradient
    graph is reproducible and shares structure). The result is an ordinary DAG in the SAME
    closed primitive set -- which is exactly why it can be differentiated AGAIN to get a
    Hessian (see :func:`hessian`).

    ``inputs`` is an iterable of var names *or* an ``env`` dict (its keys are used). For a
    var that does not reach the output, the gradient is a fresh ``const(0.0)`` node (an
    unused input has identically-zero gradient). Evaluating the returned nodes at a point
    must reproduce the NUMERIC :func:`grad` at that point -- the symbolic and numeric rails
    agree (a test pins this)."""
    names = list(inputs.keys()) if isinstance(inputs, dict) else list(inputs)
    topo = _topo_order(tape, output)
    order = _reverse_order(topo)
    # adjoint table: node id -> NODE ID of its accumulated adjoint expression.
    adj: dict[int, int] = {output: tape.const(1.0)}  # seed d(output)/d(output) = 1
    for nid in order:
        gz = adj.get(nid)
        if gz is None:
            continue  # no adjoint flows here
        n = tape.node(nid)
        rule = _BACKWARD_SYM.get(n.op)
        if rule is None:
            continue  # a leaf (const/var)
        for operand, contribution in rule(tape, n.args, n.const, gz):
            if operand in adj:
                adj[operand] = tape.add(adj[operand], contribution)  # combine contributions
            else:
                adj[operand] = contribution
    name_to_nid = {tape.node(nid).const: nid for nid in topo if tape.node(nid).op == "var"}
    out: dict[str, int] = {}
    for name in names:
        vnid = name_to_nid.get(name)
        out[name] = adj[vnid] if (vnid is not None and vnid in adj) else tape.const(0.0)
    return out


def hessian(tape: Tape, output: int, inputs, env: dict) -> dict:
    """The Hessian H[i, j] = d^2(output) / (d x_i d x_j), evaluated at ``env`` (floats).

    The second derivative is obtained HONESTLY by differentiating the GRADIENT GRAPH: call
    :func:`grad_graph` to get, for each input x_i, the node id g_i of d(output)/d(x_i) as an
    ordinary expression in the closed primitive set; then differentiate EACH g_i again with
    the numeric :func:`grad`, giving d g_i / d x_j = d^2(output)/(d x_i d x_j). This is
    reverse-over-reverse: it is sound precisely because every primitive's VJP lives in the
    same primitive set, so the gradient graph is just another DAG (see the module
    docstring's limitation note for the exact boundary of this argument).

    Returns ``{(name_i, name_j): float}`` for every ordered pair of input names. The
    central-second-difference :func:`second_difference_hessian` is the hard numeric gate; a
    symmetric closed form gives H[i, j] == H[j, i] (which a test checks)."""
    names = list(inputs.keys()) if isinstance(inputs, dict) else list(inputs)
    gnodes = grad_graph(tape, output, names)
    out: dict[tuple, float] = {}
    for ni in names:
        gi = gnodes[ni]
        row = grad(tape, gi, env).grads  # d g_i / d x_j for all j, at env
        for nj in names:
            out[(ni, nj)] = row.get(nj, 0.0)
    return out


def second_difference_hessian(tape: Tape, output: int, env: dict, *, eps: float = 1e-4) -> dict:
    """Central SECOND-difference numeric Hessian at ``env`` -- the HARD gate for
    :func:`hessian`.

    Diagonal entries use the standard three-point second difference

        H[i, i] ~ (f(x + e_i) - 2 f(x) + f(x - e_i)) / eps^2,

    off-diagonal entries the standard four-point central stencil

        H[i, j] ~ (f(x+e_i+e_j) - f(x+e_i-e_j) - f(x-e_i+e_j) + f(x-e_i-e_j)) / (4 eps^2).

    Second differences are noisier than the first differences used for the gradient gate
    (they divide by eps^2 and subtract nearly-equal numbers), so the matching tolerance
    (:func:`hessians_match`) is looser and ``eps`` is chosen larger than the gradient gate's
    -- still tight enough to catch a wrong second-derivative rule on the closed forms tested.
    Returns ``{(name_i, name_j): float}``."""
    names = list(env.keys())
    base = evaluate(tape, output, env)

    def shifted(deltas: dict) -> float:
        e = dict(env)
        for nm, d in deltas.items():
            e[nm] = env[nm] + d
        return evaluate(tape, output, e)

    out: dict[tuple, float] = {}
    for i in names:
        fpp = shifted({i: +eps})
        fmm = shifted({i: -eps})
        out[(i, i)] = (fpp - 2.0 * base + fmm) / (eps * eps)
        for j in names:
            if j == i:
                continue
            f_pp = shifted({i: +eps, j: +eps})
            f_pm = shifted({i: +eps, j: -eps})
            f_mp = shifted({i: -eps, j: +eps})
            f_mm = shifted({i: -eps, j: -eps})
            out[(i, j)] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
    return out


def hessians_match(a: dict, b: dict, *, rtol: float = 2e-2, atol: float = 2e-3) -> bool:
    """Whether two Hessian dicts (keyed by ``(name_i, name_j)``) agree within a
    relative+absolute tolerance. The tolerance is deliberately LOOSER than
    :func:`gradients_match`: a central second difference divides by eps^2 and subtracts
    nearly-equal forward values, so it carries more numerical noise than a first difference.
    The eps/tol pair is chosen so closed forms pass cleanly while a wrong second-derivative
    rule still trips the gate."""
    if set(a) != set(b):
        return False
    for k in a:
        if abs(a[k] - b[k]) > atol + rtol * abs(b[k]):
            return False
    return True


def max_hessian_error(a: dict, b: dict) -> float:
    """The largest absolute disagreement between two Hessian dicts (for reporting the exact
    failing entry rather than a bare pass/fail)."""
    return max((abs(a[k] - b.get(k, 0.0)) for k in a), default=0.0)


# --- the closure proof: d/dx : ClosedSet -> DAG(ClosedSet) ---------------------------
#
# This section FORMALIZES and machine-PROVES the property the module's honest-limitation
# note states in prose and the whole reverse-over-reverse story relies on: the adjoint
# operator set is CLOSED. Differentiating any expression built from the primitive set
# {const, var, neg, add, sub, mul, div, dot, select, exp, log, sqrt, tanh, sin, cos}
# produces a gradient DAG that stays in the SAME set -- the adjoint never introduces a
# foreign op kind or dynamic functional primitive. Together with hash-consing, that closure is exactly what gives the
# adjoint DAG a CANONICAL FORM over a fixed vocabulary -- the property a future
# ``gem.autodiff`` law op will serialize/verify against.
#
# The proof is a REAL check, not a hardcoded list: it derives the closed set from ``_ARITY``
# (the single source of truth), then for EVERY differentiable primitive it actually APPLIES
# that primitive's symbolic VJP rule (``_BACKWARD_SYM``) to symbolic adjoint + primal inputs
# via a real ``Tape``, walks every node the rule EMITS, and asserts each emitted op kind is
# in the closed set. Because the rule emits ordinary ``Tape`` nodes, the emitted vocabulary
# is OBSERVED, not asserted -- if a rule ever started emitting a foreign op, this trips. The
# numeric twin ``_BACKWARD`` is tied to the same proof by checking it computes exactly the
# value of the (proven-closed) symbolic DAG (see :func:`numeric_matches_symbolic_vjp`).

#: The closed primitive set, DERIVED from ``_ARITY`` so it is a single source of truth and
#: cannot drift from the ops the ``Tape`` can build. ``lower.autodiff_kernel._LOWERABLE`` is
#: asserted equal to this (see :func:`closed_set_agrees_with_lowerable`).
CLOSED_SET = frozenset(_ARITY)

#: The two leaves (differentiation boundaries): they have arity 0 and carry NO backward rule.
_LEAVES = frozenset(op for op, ar in _ARITY.items() if ar == 0)


def differentiable_ops() -> frozenset:
    """The differentiable primitives: every op that can appear as a NON-LEAF node and so
    carries a local VJP rule -- i.e. the closed set minus the two leaves (``const``/``var``).
    This is the set the registry-completeness check expects a backward rule for, derived from
    ``_ARITY`` (not hand-listed), so it tracks the primitive set automatically."""
    return CLOSED_SET - _LEAVES


def _canonical_symbolic_inputs(tape: "Tape", op: str) -> tuple:
    """Build representative SYMBOLIC inputs for applying ``op``'s VJP rule on ``tape``: a
    tuple ``(args, const, gz)`` of operand node ids, the op's ``const`` payload, and the
    incoming-adjoint node id ``gz``. Inputs are fresh ``var`` nodes (the linearization point
    is irrelevant to which OP KINDS the rule emits -- the rule's structure is the same for any
    inputs), so the emitted vocabulary the closure check observes is exactly the rule's.

    ``dot`` is variadic, so a representative arity (k=2 -> 4 operands) is used with the same
    ``const=("dot", k)`` payload the ``Tape.dot`` builder records; every fixed-arity op uses
    its ``_ARITY`` count."""
    gz = tape.var("__gz__")
    if op == "dot":
        k = 2  # a representative arity; the rule shape is k-independent
        args = tuple(tape.var(f"__d{i}__") for i in range(2 * k))
        return args, ("dot", k), gz
    arity = _ARITY[op]
    args = tuple(tape.var(f"__a{i}__") for i in range(arity))
    return args, None, gz


def emitted_ops_for(op: str) -> frozenset:
    """APPLY ``op``'s symbolic VJP rule (``_BACKWARD_SYM[op]``) to canonical symbolic inputs
    and return the set of op kinds of EVERY node the rule emits (each contribution node and
    everything reachable from it on the fresh part of the tape). This is the heart of the
    closure proof: the emitted vocabulary is OBSERVED by walking the actual emitted nodes, so
    it cannot drift from the rule. The returned set is what :func:`closure_report` asserts is a
    subset of :data:`CLOSED_SET`."""
    rule = _BACKWARD_SYM[op]
    tape = Tape()
    args, const, gz = _canonical_symbolic_inputs(tape, op)
    base = tape.node_count()  # everything from here on is what the RULE emitted
    contributions = rule(tape, args, const, gz)
    # the op kind of every contribution node and everything it reaches (the gradient subgraph
    # the rule built). Walk reachability so a multi-node contribution (e.g. div's nested
    # neg(div(mul(...), mul(...)))) is fully inspected.
    roots = [cnid for _operand, cnid in contributions]
    ops: set[str] = set()
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        n = tape.node(nid)
        # only nodes the rule freshly emitted carry NEW op kinds; the input leaves (< base) are
        # the symbolic primals/adjoint and are not part of the emitted vocabulary. We still
        # record their op (var) for completeness of the reachable set but they are in CLOSED_SET.
        ops.add(n.op)
        for a in n.args:
            if a not in seen:
                stack.append(a)
    return frozenset(ops)


def closure_report() -> dict:
    """The machine-checked closure proof for the SYMBOLIC adjoint (``_BACKWARD_SYM``), as a
    per-primitive map ``op -> frozenset(emitted op kinds)``. For every differentiable primitive
    this APPLIES its VJP rule and OBSERVES the op kinds it emits (:func:`emitted_ops_for`). The
    caller asserts every value is a subset of :data:`CLOSED_SET` -- i.e. ``d/dx`` maps the closed
    set into a DAG over the SAME closed set. Because ``_BACKWARD_SYM`` is the graph-valued twin
    used for reverse-over-reverse, this also proves arbitrary-order AD (the Hessian and beyond)
    stays in the set: the gradient DAG is itself differentiable by the same machinery."""
    return {op: emitted_ops_for(op) for op in sorted(differentiable_ops())}


def numeric_matches_symbolic_vjp(op: str, *, seed: int = 0) -> bool:
    """Tie the NUMERIC backward rule ``_BACKWARD[op]`` to the proven-closed symbolic DAG: at a
    random linearization point, the float contributions the numeric rule returns must EQUAL the
    values obtained by evaluating the symbolic rule's emitted (closed-set) nodes. The numeric
    rule computes only ``+ - * / neg`` of its float inputs -- the same arithmetic the symbolic
    rule materializes as closed-set nodes -- so agreement here certifies the numeric adjoint's
    operations are exactly the closed-set ones (it has no separate vocabulary to escape into).
    Deterministic (seeded), exact (the closed-form rules are float-exact, no tolerance)."""
    import random

    rng = random.Random(seed ^ (hash(op) & 0xFFFF))
    tape = Tape()
    args, const, gz = _canonical_symbolic_inputs(tape, op)
    # a random, well-conditioned point (denominators away from zero for div/dot stability).
    env = {tape.node(nid).const: rng.uniform(0.5, 2.0) for nid in args}
    env[tape.node(gz).const] = rng.uniform(0.5, 2.0)
    fvals = {nid: env[tape.node(nid).const] for nid in (*args, gz)}
    gz_val = fvals[gz]
    numeric = _BACKWARD[op](fvals, args, const, gz_val)
    symbolic = _BACKWARD_SYM[op](tape, args, const, gz)
    # both return [(operand, contribution), ...] in the same operand order; compare the value of
    # each numeric float contribution against the evaluated symbolic-node contribution.
    if len(numeric) != len(symbolic):
        return False
    for (n_operand, n_val), (s_operand, s_nid) in zip(numeric, symbolic):
        if n_operand != s_operand:
            return False
        if evaluate(tape, s_nid, env) != n_val:
            return False
    return True


def registry_completeness() -> dict:
    """Prove the BIJECTION between the differentiable primitives and each backward registry --
    no missing rule, no orphan/dead rule -- for BOTH the numeric ``_BACKWARD`` and the symbolic
    ``_BACKWARD_SYM`` twin. Returns a report dict:

      * ``differentiable``       -- the differentiable op set (closed set minus leaves);
      * ``backward_keys`` / ``backward_sym_keys`` -- the registry key sets;
      * ``backward_complete``    -- ``_BACKWARD`` keys == differentiable set (bijection);
      * ``backward_sym_complete``-- ``_BACKWARD_SYM`` keys == differentiable set (bijection);
      * ``missing_*`` / ``orphan_*`` -- the exact discrepancies (empty on success), so a failure
        names the drifting op rather than a bare False.

    A leaf (``const``/``var``) must have NO rule (it is a differentiation boundary); a
    differentiable op must have EXACTLY one. The caller asserts both ``*_complete`` and that no
    leaf leaked into either registry."""
    diff = differentiable_ops()
    bw = frozenset(_BACKWARD)
    bws = frozenset(_BACKWARD_SYM)
    return {
        "differentiable": diff,
        "backward_keys": bw,
        "backward_sym_keys": bws,
        "backward_complete": bw == diff,
        "backward_sym_complete": bws == diff,
        "missing_backward": diff - bw,
        "orphan_backward": bw - diff,
        "missing_backward_sym": diff - bws,
        "orphan_backward_sym": bws - diff,
        "leaves_in_backward": _LEAVES & bw,
        "leaves_in_backward_sym": _LEAVES & bws,
    }


def closed_set_agrees_with_lowerable() -> bool:
    """Assert the two definitions of "the closed set" AGREE -- :data:`CLOSED_SET` (derived here
    from ``_ARITY``) and ``lower.autodiff_kernel._LOWERABLE`` (the set the G6 C-kernel emitter
    enforces). A single source of truth: if a primitive is added to the oracle, this trips until
    the lowerable set is updated too, so the closure proof and the lowering can never silently
    diverge on what the closed vocabulary is."""
    from ..lower.autodiff_kernel import _LOWERABLE

    return CLOSED_SET == frozenset(_LOWERABLE)
