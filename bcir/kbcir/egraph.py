"""bcir.egraph -- the building-blocks engine: e-classes, rewrites, extraction.

We have decomposition (the phase DAG, claims) and partition (lanes, the GGG
quarantine, candidate enumeration). This is the *composition* engine: the
fundamental building blocks (atoms) compose via operators into higher-order
patterns, equivalent forms are explored by rewriting, and the cheapest
composition is extracted under the K_BCIR cost model. It is an e-graph
(equality-saturation, egg-style), and it is the direct realization of the
liked/unliked-pair model:

  * **Liked pairs** = e-classes. An atom (a const, a resource) is its own class;
    by hashconsing, two references to the same building block land in the *same*
    class -- the identity relation `a = a`, the "memory module" that stores a
    consistent representation. Common subexpressions are liked pairs found for
    free (CSE).
  * **Unliked pairs** = operators (+, -, x) and the rewrites they enable. An
    operator synthesizes a new element from differing inputs; a rewrite proves two
    forms equal and *merges* their classes (congruence closure), folding an
    unliked result back toward a simpler liked attractor (`x + 0 -> x`,
    `x*b + x*c -> x*(b+c)`).
  * **Resolution** = saturation + extraction. Applying all legal rewrites to a
    fixpoint builds the space of equivalent compositions; extraction picks the
    min-cost representative per class -- the optimized building block.

Complexity is bounded exactly as the model warns it must be: a rewrite is legal
only if it preserves semantics (the e-graph proves equality by construction) and
does not increase the selected cost (extraction returns the min, so the result is
always <= the input). Saturation runs to a fixpoint or a budget -- the guard on
the "abundance of unliked pairs." Witnessed by R9 (`bcir.egraph.extract`:
optimized_cost <= original_cost).

Deterministic, integer, dependency-free. This is plan-time (L1) work; its own
cost is governed by the L1 throttle (`kbcir.throttle`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- expressions (the building blocks) -------------------------------------------


@dataclass(frozen=True)
class Expr:
    """An expression tree: leaf ("const",val) / ("var",name) or op ("add"/"sub"/"mul")."""

    op: str
    val: object = None
    args: tuple = ()


def C(v: int) -> Expr:
    return Expr("const", v, ())


def V(name) -> Expr:
    return Expr("var", name, ())


def Add(a: Expr, b: Expr) -> Expr:
    return Expr("add", None, (a, b))


def Sub(a: Expr, b: Expr) -> Expr:
    return Expr("sub", None, (a, b))


def Mul(a: Expr, b: Expr) -> Expr:
    return Expr("mul", None, (a, b))


# default op costs (mul costlier than add, so factoring pays). Alignable with the
# K_BCIR opclass; the relative order is what drives extraction.
DEFAULT_COSTS = {"const": 0, "var": 0, "add": 1, "sub": 1, "mul": 2}


# --- the e-graph -----------------------------------------------------------------


@dataclass
class SaturationStats:
    iterations: int = 0
    merges: int = 0
    enodes: int = 0
    saturated: bool = False  # reached a fixpoint within budget?


class EGraph:
    """A hashconsed e-graph with union-find + congruence closure (rebuild)."""

    def __init__(self):
        self._parent: dict[int, int] = {}
        self._next = 0
        # enode id -> (op, val, child class-ids); class root -> set of enode ids
        self._enodes: dict[int, tuple] = {}
        self._class_nodes: dict[int, set] = {}
        self._memo: dict[tuple, int] = {}  # canonical enode key -> class id
        self._enode_next = 0

    # union-find
    def find(self, c: int) -> int:
        parent = self._parent  # localize the hot dict lookup
        root = c
        while parent[root] != root:
            root = parent[root]
        while parent[c] != root:
            parent[c], c = root, parent[c]
        return root

    def _new_class(self) -> int:
        cid = self._next
        self._next += 1
        self._parent[cid] = cid
        self._class_nodes[cid] = set()
        return cid

    def _canon_key(self, eid: int) -> tuple:
        op, val, children = self._enodes[eid]
        return (op, val, tuple(self.find(c) for c in children))

    def add(self, expr: Expr) -> int:
        children = tuple(self.add(a) for a in expr.args)
        key = (expr.op, expr.val, tuple(self.find(c) for c in children))
        if key in self._memo:
            return self.find(self._memo[key])
        cid = self._new_class()
        eid = self._enode_next
        self._enode_next += 1
        self._enodes[eid] = (expr.op, expr.val, list(children))
        self._class_nodes[cid].add(eid)
        self._memo[key] = cid
        self._enode_of_class = getattr(self, "_enode_of_class", {})
        self._enode_of_class[eid] = cid
        return cid

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return 0
        self._parent[rb] = ra
        self._class_nodes[ra] |= self._class_nodes.pop(rb)
        return 1

    def rebuild(self) -> int:
        """Restore congruence: enodes whose canonical keys coincide name the same
        class and are merged. Returns the number of merges (loops to a fixpoint)."""
        total = 0
        changed = True
        while changed:
            changed = False
            seen: dict[tuple, int] = {}
            for eid in list(self._enodes):
                cls = self.find(self._enode_of_class[eid])
                self._enode_of_class[eid] = cls
                key = self._canon_key(eid)
                if key in seen and self.find(seen[key]) != cls:
                    if self.union(seen[key], cls):
                        total += 1
                        changed = True
                else:
                    seen[key] = cls
        # refresh the memo to canonical keys
        self._memo = {
            self._canon_key(eid): self.find(self._enode_of_class[eid]) for eid in self._enodes
        }
        return total

    def classes(self) -> dict:
        """root class id -> list of (op, val, child-roots) canonical enodes."""
        find = self.find  # localize hot lookups (called O(enodes))
        eoc = self._enode_of_class
        out: dict[int, list] = {}
        for eid, (op, val, children) in self._enodes.items():
            out.setdefault(find(eoc[eid]), []).append((op, val, tuple(find(c) for c in children)))
        return out

    def class_has_const(self, cid: int, value: int) -> bool:
        cid = self.find(cid)
        for eid in self._class_nodes[cid]:
            op, val, _ = self._enodes[eid]
            if op == "const" and val == value:
                return True
        return False

    # --- extraction (min-cost representative per class) --------------------------

    def extract(self, root: int, costs: dict = None):
        """Return (best_expr, cost) for `root` under `costs` -- the cheapest
        composition. Fixpoint over class costs (handles shared/cyclic structure)."""
        costs = costs or DEFAULT_COSTS
        classes = self.classes()
        INF = float("inf")
        best_cost = {c: INF for c in classes}
        best_node = {c: None for c in classes}
        changed = True
        while changed:
            changed = False
            for c, nodes in classes.items():
                for op, val, children in nodes:
                    cc = costs.get(op, 1) + sum(
                        best_cost.get(self.find(ch), INF) for ch in children
                    )
                    if cc < best_cost[c]:
                        best_cost[c], best_node[c] = cc, (op, val, children)
                        changed = True

        def build(c):
            op, val, children = best_node[self.find(c)]
            return Expr(op, val, tuple(build(ch) for ch in children))

        r = self.find(root)
        return build(r), best_cost[r]


# --- rewrites (the unliked operators that merge classes) -------------------------


def rw_identities(eg: EGraph) -> int:
    """x+0->x, 0+x->x, x*1->x, 1*x->x, x*0->0, x-x->0 (identity elimination)."""
    merges = 0
    for c, nodes in list(eg.classes().items()):
        for op, val, ch in nodes:
            if op == "add" and len(ch) == 2:
                if eg.class_has_const(ch[1], 0):
                    merges += eg.union(c, ch[0])
                elif eg.class_has_const(ch[0], 0):
                    merges += eg.union(c, ch[1])
            elif op == "mul" and len(ch) == 2:
                if eg.class_has_const(ch[1], 1):
                    merges += eg.union(c, ch[0])
                elif eg.class_has_const(ch[0], 1):
                    merges += eg.union(c, ch[1])
                elif eg.class_has_const(ch[0], 0) or eg.class_has_const(ch[1], 0):
                    merges += eg.union(c, eg.add(C(0)))
            elif op == "sub" and len(ch) == 2 and eg.find(ch[0]) == eg.find(ch[1]):
                merges += eg.union(c, eg.add(C(0)))
    return merges


def rw_constant_fold(eg: EGraph) -> int:
    """1+1->2, 1-1->0, 2*3->6: resolve an unliked pair of constants into a new
    liked atom (the synthesized element that joins the system's memory)."""
    merges = 0
    for c, nodes in list(eg.classes().items()):
        for op, val, ch in nodes:
            if len(ch) != 2:
                continue
            a, b = _const_value(eg, ch[0]), _const_value(eg, ch[1])
            if a is None or b is None:
                continue
            r = {"add": a + b, "sub": a - b, "mul": a * b}.get(op)
            if r is not None:
                merges += eg.union(c, eg.add(C(r)))
    return merges


def rw_factor(eg: EGraph) -> int:
    """x*b + x*c -> x*(b+c) (distributivity / fusion): recompose shared building
    blocks into a cheaper form."""
    merges = 0
    classes = eg.classes()
    muls = {
        c: [tuple(n[2]) for n in nodes if n[0] == "mul" and len(n[2]) == 2]
        for c, nodes in classes.items()
    }
    for c, nodes in list(classes.items()):
        for op, val, ch in nodes:
            if op != "add" or len(ch) != 2:
                continue
            for a1, b1 in muls.get(eg.find(ch[0]), []):
                for a2, b2 in muls.get(eg.find(ch[1]), []):
                    pairs = [(a1, b1, a2, b2), (a1, b1, b2, a2), (b1, a1, a2, b2), (b1, a1, b2, a2)]
                    for f1, o1, f2, o2 in pairs:
                        if eg.find(f1) == eg.find(f2):
                            summ = eg.add_class_sum(o1, o2)
                            factored = eg.add_mul(f1, summ)
                            merges += eg.union(c, factored)
                            break
    return merges


# helpers that add structural nodes by class id (not Expr) ------------------------


def _const_class(eg: EGraph, value: int):
    for c, nodes in eg.classes().items():
        for op, val, _ch in nodes:
            if op == "const" and val == value:
                return c
    return None


def _const_value(eg: EGraph, cid: int):
    for eid in eg._class_nodes[eg.find(cid)]:
        op, val, _ = eg._enodes[eid]
        if op == "const":
            return val
    return None


# class-id-level constructors (used by rewrites) ----------------------------------


def _add_node(eg: EGraph, op: str, children: tuple) -> int:
    key = (op, None, tuple(eg.find(c) for c in children))
    if key in eg._memo:
        return eg.find(eg._memo[key])
    cid = eg._new_class()
    eid = eg._enode_next
    eg._enode_next += 1
    eg._enodes[eid] = (op, None, list(children))
    eg._class_nodes[cid].add(eid)
    eg._enode_of_class[eid] = cid
    eg._memo[key] = cid
    return cid


EGraph.add_class_sum = lambda self, a, b: _add_node(self, "add", (a, b))
EGraph.add_mul = lambda self, a, b: _add_node(self, "mul", (a, b))


DEFAULT_RULES = [rw_constant_fold, rw_identities, rw_factor]


def saturate(eg: EGraph, rules=None, budget: int = 30) -> SaturationStats:
    """Apply rewrites to a fixpoint (or `budget` iterations -- the blow-up guard)."""
    rules = rules or DEFAULT_RULES
    st = SaturationStats()
    for _ in range(budget):
        st.iterations += 1
        m = 0
        for rule in rules:
            m += rule(eg)
        m += eg.rebuild()
        st.merges += m
        if m == 0:
            st.saturated = True
            break
    st.enodes = len(eg._enodes)
    return st


# --- the end-to-end optimizer (build -> saturate -> extract) ---------------------


@dataclass(frozen=True)
class EGraphResult:
    original_cost: int
    optimized_cost: int
    optimized: Expr
    iterations: int
    enodes: int
    saturated: bool

    @property
    def improved(self) -> int:
        return self.original_cost - self.optimized_cost  # >= 0 (rewrites never raise cost)


def _expr_cost(expr: Expr, costs: dict) -> int:
    """The cost of an expression as written -- op cost plus the cost of each
    argument. Identical to extracting an unsaturated e-graph of `expr` (each class
    has one enode), but without building a throwaway graph: the input's price."""
    return costs.get(expr.op, 1) + sum(_expr_cost(a, costs) for a in expr.args)


def optimize_expr(expr: Expr, rules=None, costs: dict = None, budget: int = 30) -> EGraphResult:
    """Optimize an expression through the e-graph; the result cost is always
    <= the original (R9: a rewrite never increases the selected cost)."""
    costs = costs or DEFAULT_COSTS
    original_cost = _expr_cost(expr, costs)

    eg = EGraph()
    root = eg.add(expr)
    st = saturate(eg, rules, budget)
    best, opt_cost = eg.extract(root, costs)
    return EGraphResult(
        original_cost=int(original_cost),
        optimized_cost=int(opt_cost),
        optimized=best,
        iterations=st.iterations,
        enodes=st.enodes,
        saturated=st.saturated,
    )


# --- BCIR bridge: claims compose as expressions; CSE is the liked-pair memory ----


def module_exprs(module) -> dict:
    """Map a module's binary-elementwise claims to expressions over operand RIDs.
    Two claims with the same op + operands share an e-class (CSE = liked pair)."""
    _OP = {
        "vector.add": "add",
        "vector.sub": "sub",
        "vector.axpy": "mul",
        "add": "add",
        "sub": "sub",
        "mul": "mul",
    }
    out = {}
    for ph in module.phases:
        for c in ph.claims:
            op = _OP.get(c.op)
            if op and len(c.rd) == 2:
                out[c.id] = Expr(op, None, (V(c.rd[0]), V(c.rd[1])))
            elif op and len(c.rd) == 1:
                out[c.id] = V(c.rd[0])
    return out


def shared_blocks(module) -> int:
    """How many distinct building blocks (e-classes) a module's claims reduce to
    after CSE -- fewer than the claim count means shared liked pairs were found."""
    eg = EGraph()
    for expr in module_exprs(module).values():
        eg.add(expr)
    eg.rebuild()
    return len(eg.classes())


# --- the resident (persistent) e-graph: re-extract on telemetry, no re-parse -----


@dataclass
class ResidentEGraph:
    """A *living* e-graph: built and saturated once, then kept resident so that when
    telemetry reports a bottleneck the planner can **pivot** to an equivalent, more
    efficient sub-graph by re-extracting under reweighted costs -- WITHOUT
    re-parsing the source or re-saturating. The equivalence classes are already
    proven; a pivot is just a fresh min-cost extraction over the standing graph.

    `pivot(bump=...)` raises the cost of a reported-bottleneck op (e.g. telemetry
    says `mul` stalls) and re-extracts; if the class holds a cheaper equivalent
    under the new costs, extraction returns it. Gains-only: extraction always
    returns the min, so a pivot never raises the selected cost. The standing graph
    is not mutated (`resident_size` is invariant across pivots), proving no rebuild.
    """

    eg: EGraph
    root: int
    costs: dict
    stats: SaturationStats
    pivots: int = 0
    _built_enodes: int = 0

    @classmethod
    def build(
        cls, expr: Expr, rules=None, costs: dict = None, budget: int = 30
    ) -> "ResidentEGraph":
        eg = EGraph()
        root = eg.add(expr)
        st = saturate(eg, rules, budget)
        return cls(
            eg=eg,
            root=root,
            costs=dict(costs or DEFAULT_COSTS),
            stats=st,
            _built_enodes=len(eg._enodes),
        )

    def current(self) -> EGraphResult:
        """Extract the best composition for the resident root under the live costs."""
        best, cost = self.eg.extract(self.root, self.costs)
        return EGraphResult(
            original_cost=int(cost),
            optimized_cost=int(cost),
            optimized=best,
            iterations=0,
            enodes=len(self.eg._enodes),
            saturated=self.stats.saturated,
        )

    def pivot(self, *, costs: dict = None, bump: dict = None) -> EGraphResult:
        """Re-cost and re-extract over the standing graph (no rebuild, no re-parse).
        `costs` replaces entries; `bump` adds to op costs (a telemetry bottleneck)."""
        new = dict(self.costs)
        if costs:
            new.update(costs)
        for op, delta in (bump or {}).items():
            new[op] = new.get(op, 1) + delta
        self.costs = new
        self.pivots += 1
        return self.current()

    @property
    def resident_size(self) -> int:
        """Enodes held resident; invariant across pivots (the graph is not rebuilt)."""
        return len(self.eg._enodes)

    @property
    def rebuilt_since_build(self) -> bool:
        return self.resident_size != self._built_enodes
