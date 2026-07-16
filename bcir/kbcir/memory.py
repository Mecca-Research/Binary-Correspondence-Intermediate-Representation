"""bcir.kbcir.memory -- Memory modules: the fixpoint law a = Lim(Res(U)).

The Axiom of Memory Modules. The e-graph (`kbcir.egraph`) is the *composition*
engine: it builds the space of equivalent forms of a universe U of building
blocks by **resolution** -- one round Res(.) of rewriting + congruence rebuild --
and extracts the min-cost representative. Resolution is *monotone* (it only adds
equalities; a merged class never un-merges) over a *bounded* lattice (finitely
many e-nodes under a terminating rule set), so by Knaster-Tarski/Kleene the
iteration Res^k(U) converges to a least fixpoint

    Lim(Res(U)) = Res^infinity(U) = the smallest X with Res(X) = X   (== saturation).

A **memory module** is the extraction of that fixpoint, *frozen* and
*generation-tagged*:

    memory = Extract(Lim(Res(U)))   -- a saturated, immutable, gen-tagged artifact.

This is the liked-pair identity `a = a` realized at module scope: a memory module
is the *attractor* of resolution. Two properties make it the right thing to pin
into a generation, and both are the keystone tying Phase 21 (the e-graph engine)
to Phase 20 (the provenance manifest):

  * **Idempotence.** `Res(Lim(Res(U))) = Lim(Res(U))`: re-resolving a memory
    module yields itself. Freezing a fixpoint is *stable*; the artifact is its
    own canonical form. (`reresolve`, `is_idempotent`.)
  * **The admissibility law (the fixpoint witness).** An artifact may be frozen
    into a generation **only if** resolution reached the fixpoint --
    `saturated == True`. A *budget cutoff* is a partial resolution `Res^k(U)` for
    some finite `k < infinity`; freezing it would pin a non-canonical,
    still-improvable representative as "memory," and -- because a cutoff is
    order/budget dependent while the fixpoint is canonical (confluence) -- two
    runs that cut off differently need not agree. That breaks the determinism the
    provenance manifest depends on (manifest equality => identical plan). Hence

        saturated == True   ==>   admissible as memory.

    The saturation witness *is* the admitting certificate for entry into a
    generation, witnessed by R13 (`verify.verify_memory`).

Deterministic, integer, dependency-free. Producing a memory module is L1/L2
plan-time work (`saturate` runs at L1; freezing/generation-tagging is an L2
checkpoint act); nothing here runs on the L0 hot path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .._artifact_json import strict_json_loads
from .egraph import (
    DEFAULT_COSTS,
    DEFAULT_RULES,
    EGraph,
    EGraphResult,
    Expr,
    module_exprs,
    optimize_expr,
    saturate,
)
from .provenance import _fnv

# Default plan-time saturation budget. Generous: a memory module must reach the
# *fixpoint* within it, not merely run out of rounds (that is the whole law).
DEFAULT_BUDGET = 64

# A synthetic root op used to fold a module's claim forest into a single universe
# so the whole module resolves to one fixpoint. It is inert under the rewrites
# (they touch add/sub/mul only) and free under extraction.
_MODULE_OP = "module"
_MODULE_COSTS = {**DEFAULT_COSTS, _MODULE_OP: 0}


class UnsaturatedMemory(Exception):
    """Refused to freeze: resolution did not reach its fixpoint within budget, so
    the extraction is a partial Res^k(U), not Lim(Res(U)) -- not admissible as
    memory. Raise the budget or fix the rule set (it must terminate)."""


# --- the canonical content fingerprint -------------------------------------------

def _canon(expr: Expr) -> tuple:
    """A canonical nested tuple for an Expr tree (op, val, children...)."""
    return (expr.op, expr.val, tuple(_canon(a) for a in expr.args))


def fingerprint(expr: Expr, cost: int) -> int:
    """Content hash of an extracted representative + its cost (FNV-1a, i63-masked;
    the same hash the provenance manifest uses, so fingerprints interoperate)."""
    return _fnv("memory", _canon(expr), int(cost))


# --- the memory module -----------------------------------------------------------

@dataclass(frozen=True)
class MemoryModule:
    """A frozen, generation-tagged extraction of a resolution fixpoint:
    `a = Lim(Res(U))`. `saturated` is the fixpoint witness; `admissible` is the
    law (saturated AND generation-tagged)."""

    expr: Expr            # Extract(Lim(Res(U))) -- the canonical min-cost representative
    cost: int             # its K_BCIR cost (extraction returns the minimum)
    fingerprint: int      # content hash of (expr, cost)
    generation: int       # the generation tag; immutable within this generation (>= 1)
    iterations: int       # rounds of Res(.) applied to reach the fixpoint
    enodes: int           # |Lim(Res(U))| -- the saturated e-graph size
    saturated: bool       # the fixpoint witness: Res(state) == state within budget

    @property
    def admissible(self) -> bool:
        """The fixpoint law: a memory module is a *saturated* extraction, frozen
        AND generation-tagged. A budget cutoff (saturated == False) or an
        untagged artifact (generation < 1) is not memory."""
        return self.saturated and self.generation >= 1

    def to_json(self) -> str:
        d = asdict(self)
        d["expr"] = _canon(self.expr)
        return json.dumps(d, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "MemoryModule":
        d = strict_json_loads(text, "memory module")
        fields = {"expr", "cost", "fingerprint", "generation", "iterations",
                  "enodes", "saturated"}
        if not isinstance(d, dict) or set(d) != fields:
            raise ValueError(f"memory-module fields must be exactly {sorted(fields)}")

        def nonnegative_i63(key: str) -> int:
            value = d[key]
            if (isinstance(value, bool) or not isinstance(value, int) or
                    not 0 <= value <= (1 << 63) - 1):
                raise ValueError(f"memory-module {key} must be a non-negative i63")
            return value

        cost = nonnegative_i63("cost")
        stored_fingerprint = nonnegative_i63("fingerprint")
        generation = nonnegative_i63("generation")
        iterations = nonnegative_i63("iterations")
        enodes = nonnegative_i63("enodes")
        if not isinstance(d["saturated"], bool):
            raise ValueError("memory-module saturated must be a boolean")
        computed_cost = _validate_canonical_expr(d["expr"])
        if computed_cost != cost:
            raise ValueError(
                f"memory-module cost {cost} does not match expression cost {computed_cost}")
        expr = _uncanon(d["expr"])
        computed_fingerprint = fingerprint(expr, cost)
        if stored_fingerprint != computed_fingerprint:
            raise ValueError(
                f"memory-module fingerprint {stored_fingerprint} does not match content "
                f"fingerprint {computed_fingerprint}")
        return MemoryModule(expr=expr, cost=cost, fingerprint=stored_fingerprint,
                            generation=generation, iterations=iterations, enodes=enodes,
                            saturated=d["saturated"])


def _validate_canonical_expr(root) -> int:
    """Validate a persisted expression without recursively trusting its shape.

    Returns its canonical default K_BCIR cost.  The limits bound both decoder
    memory and the e-graph work a caller can trigger after loading the artifact.
    """
    stack = [(root, 1)]
    nodes = 0
    cost = 0
    op_cost = {"const": 0, "var": 0, "add": 1, "sub": 1, "mul": 2,
               _MODULE_OP: 0}
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > 65536:
            raise ValueError("memory-module expression exceeds 65536 nodes")
        if depth > 256:
            raise ValueError("memory-module expression exceeds depth 256")
        if not isinstance(node, list) or len(node) != 3:
            raise ValueError("memory-module expression nodes must be [op, value, children]")
        op, value, children = node
        if op not in op_cost:
            raise ValueError(f"memory-module expression has unsupported op {op!r}")
        if not isinstance(children, list):
            raise ValueError("memory-module expression children must be an array")
        if op == "const":
            if (isinstance(value, bool) or not isinstance(value, int) or
                    not -(1 << 63) <= value <= (1 << 63) - 1 or children):
                raise ValueError("memory-module const must be a signed i64 leaf")
        elif op == "var":
            valid_string = (isinstance(value, str) and value and len(value) <= 4096 and
                            not any(ord(ch) < 0x20 for ch in value))
            valid_int = (not isinstance(value, bool) and isinstance(value, int) and
                         -(1 << 63) <= value <= (1 << 63) - 1)
            if not (valid_string or valid_int) or children:
                raise ValueError("memory-module var must be a bounded scalar leaf")
        elif op == _MODULE_OP:
            if (depth != 1 or not isinstance(value, str) or not value or len(value) > 4096 or
                    any(ord(ch) < 0x20 for ch in value)):
                raise ValueError("memory-module module op must be a bounded root")
        elif value is not None or len(children) != 2:
            raise ValueError(f"memory-module {op} must have null value and two children")
        cost += op_cost[op]
        if cost > (1 << 63) - 1:
            raise ValueError("memory-module expression cost overflows i63")
        stack.extend((child, depth + 1) for child in reversed(children))
    return cost


def _uncanon(t) -> Expr:
    op, val, children = t[0], t[1], t[2]
    return Expr(op, val, tuple(_uncanon(c) for c in children))


# --- producers (resolve -> freeze) -----------------------------------------------

def from_result(result: EGraphResult, generation: int = 1) -> MemoryModule:
    """Build a memory module from an already-computed e-graph result, carrying its
    saturation witness through faithfully (admissible iff it saturated + tagged)."""
    return MemoryModule(
        expr=result.optimized, cost=int(result.optimized_cost),
        fingerprint=fingerprint(result.optimized, result.optimized_cost),
        generation=int(generation), iterations=int(result.iterations),
        enodes=int(result.enodes), saturated=bool(result.saturated))


def try_freeze(expr: Expr, *, generation: int = 1, rules=None, costs: dict = None,
               budget: int = DEFAULT_BUDGET) -> MemoryModule:
    """Resolve `expr` to (attempted) saturation and freeze the extraction --
    *without* enforcing the fixpoint law. The returned module reports `saturated`
    honestly; `admissible` tells you whether it earns a generation. Use this to
    build a candidate for the verifier to judge; use `freeze` to enforce the law
    at the producer."""
    result = optimize_expr(expr, rules=rules, costs=costs, budget=budget)
    return from_result(result, generation)


def freeze(expr: Expr, *, generation: int = 1, rules=None, costs: dict = None,
           budget: int = DEFAULT_BUDGET) -> MemoryModule:
    """Resolve and freeze `expr` into a memory module, *enforcing* the fixpoint
    law: raises `UnsaturatedMemory` if resolution hit the budget without reaching
    its fixpoint (you tried to freeze a partial Res^k(U), not Lim(Res(U)))."""
    mm = try_freeze(expr, generation=generation, rules=rules, costs=costs, budget=budget)
    if not mm.saturated:
        raise UnsaturatedMemory(
            f"resolution did not reach a fixpoint in {mm.iterations} round(s) "
            f"(budget {budget}); the extraction is a cutoff Res^k(U), not "
            f"Lim(Res(U)) -- not admissible as memory")
    return mm


def _module_universe(module) -> Expr:
    """Fold a module's binary-elementwise claim forest into a single universe: a
    synthetic root over every claim expression, so the whole module resolves to
    one fixpoint (shared subexpressions become liked pairs -- CSE)."""
    exprs = module_exprs(module)
    args = tuple(exprs[cid] for cid in sorted(exprs))
    return Expr(_MODULE_OP, module.name, args)


def freeze_module(module, *, generation: int = 1, rules=None,
                  budget: int = DEFAULT_BUDGET) -> MemoryModule:
    """Freeze a whole BCIR module's resolved claim forest as one memory module:
    `Extract(Lim(Res(U)))` over the module's universe of claim expressions.
    Enforces the fixpoint law (raises `UnsaturatedMemory` on a budget cutoff)."""
    return freeze(_module_universe(module), generation=generation, rules=rules,
                  costs=_MODULE_COSTS, budget=budget)


# --- idempotence (the memory module is its own attractor) ------------------------

def reresolve(mm: MemoryModule, *, rules=None, costs: dict = None,
              budget: int = DEFAULT_BUDGET) -> MemoryModule:
    """Re-resolve a memory module's stored representative: `Res(Lim(Res(U)))`. The
    axiom predicts this returns the module itself (same structure, immediately
    saturated) -- the idempotence/attractor property."""
    costs = costs if costs is not None else _costs_for(mm.expr)
    return try_freeze(mm.expr, generation=mm.generation, rules=rules, costs=costs,
                      budget=budget)


def is_idempotent(mm: MemoryModule, *, rules=None, costs: dict = None,
                  budget: int = DEFAULT_BUDGET) -> bool:
    """True iff re-resolving the module yields itself, saturated -- a runnable
    witness that the stored artifact is a genuine fixpoint `Res(Lim) = Lim`."""
    re = reresolve(mm, rules=rules, costs=costs, budget=budget)
    return re.saturated and re.expr == mm.expr


def _costs_for(expr: Expr) -> dict:
    """Pick the cost map a representative was resolved under (module roots are
    free; everything else uses the default op costs)."""
    return _MODULE_COSTS if expr.op == _MODULE_OP else DEFAULT_COSTS


# --- the provenance bridge (Phase 21 -> Phase 20) --------------------------------

def memory_artifacts(mm: MemoryModule) -> tuple:
    """The artifact tags a memory module contributes to a provenance manifest: its
    generation and content fingerprint. Chaining these into the manifest is what
    makes a frozen memory module part of a plan's commit hash."""
    return (("mem_gen", mm.generation), ("mem_fp", mm.fingerprint))
