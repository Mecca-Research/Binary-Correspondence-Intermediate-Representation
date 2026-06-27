"""B3: reverse-mode automatic differentiation as content-addressed graph rewrites.

This is the math oracle for slice B3 of the ML/AI-integration roadmap (see
``docs/research/AI_SUBSTRATE_SOTA.md``, Pillar 3). It realizes the central B3
thesis in the smallest honest form: a differentiable expression DAG of first-order
primitives, **content-addressed (hash-consed)** so structurally identical
subexpressions are *one shared node*, and reverse-mode AD expressed as a set of
**local backward rewrite rules** -- one adjoint (vector-Jacobian product) rule per
primitive -- that walk the forward DAG and *build* an accumulated adjoint graph.
Because the adjoint accumulation is itself content-addressed, a subexpression
shared in the forward graph contributes a *single shared adjoint* (deduped, not
recomputed) -- a stronger, global form of common-subexpression reuse than a local
per-instruction cache. That global structural sharing is the property the SOTA
scan highlights as BCIR's distinctive contribution.

TERMINOLOGY (the SOTA scan corrected this). The framing is **content-addressed,
confluent local GRAPH REWRITES** in the tradition of 2-categorical / string-diagram
(PROP / symmetric-monoidal-category) rewriting -- *not* "operad" (no prior art models
reverse-mode AD as operad 2-cells; the established structure is monoidal-category
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

Deterministic, pure-Python, dependency-free. This is a **research organ** and is kept
*cold* (off the hot plan->emit import path; see ``tools/perf/import_graph.py``).
"""

from __future__ import annotations

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
_ARITY = {"const": 0, "var": 0, "neg": 1,
          "add": 2, "sub": 2, "mul": 2, "div": 2, "dot": None}  # dot is n-ary (variadic)


@dataclass(frozen=True)
class Node:
    """A hash-consed node of the expression DAG.

    ``key`` is the structural content key (op + operand keys + constant), so equal
    structure == equal key == one interned node. ``args`` are operand node ids
    (ints), ``const`` carries a leaf's constant (a float for ``const``; a name for
    ``var``)."""

    nid: int
    op: str
    args: tuple          # operand node ids
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
        self._memo: dict[tuple, int] = {}     # content key -> node id (the hash-cons)

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


# the rewrite-rule table: one local backward rule per primitive (leaves have none --
# const and var are differentiation boundaries, const contributing zero by definition).
_BACKWARD = {
    "neg": _bw_neg, "add": _bw_add, "sub": _bw_sub,
    "mul": _bw_mul, "div": _bw_div, "dot": _bw_dot,
}


@dataclass(frozen=True)
class GradResult:
    """The gradient of an output w.r.t. requested inputs, plus the cheap-gradient
    accounting that demonstrates reverse mode touches O(1)x the forward op count."""

    grads: dict           # var name -> d(output)/d(var)
    value: float          # the forward value of the output (computed en route)
    forward_ops: int      # primitive ops in the forward DAG (reachable, deduped)
    backward_ops: int     # primitive rule firings in the reverse pass

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
    adj: dict[int, float] = {root: 1.0}    # seed: d(output)/d(output) = 1
    firings = 0
    for nid in order:
        gz = adj.get(nid, 0.0)
        if gz == 0.0:
            continue                       # no adjoint flows here -> rule is a no-op
        n = tape.node(nid)
        rule = _BACKWARD.get(n.op)
        if rule is None:
            continue                       # a leaf (const/var): differentiation boundary
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
    for nid in topo:                          # operands precede users in topo, so this fills bottom-up
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
    return GradResult(grads=grads, value=fvals[output],
                      forward_ops=forward_ops, backward_ops=firings)


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
        plus = dict(env); plus[name] = env[name] + eps
        minus = dict(env); minus[name] = env[name] - eps
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
