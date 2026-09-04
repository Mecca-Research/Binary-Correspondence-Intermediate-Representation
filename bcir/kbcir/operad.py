"""bcir.kbcir.operad -- the enriched-operad memory interface (the higher
intelligence layer over the memory modules).

This is the interpretive layer that sits *on top of* the deterministic spine
(R1-R13) and the frozen memory modules (`kbcir.memory`, `a = Lim(Res(U))`). The
memory module already IS an operad: its e-nodes are operations, the operators
(add/sub/mul) are the composition `γ`, the atoms (const/var) are the identity `η`,
and the extraction tree is the operad's operation tree. This module *enriches*
that operad with **labels** and **indexes** -- a navigable, traceable,
content-addressed structure that makes the memory mathematically and
computationally intelligent without touching the hot path.

The enriched operad `O_L = ((O_L(n)), γ_L, η_L, L, I)`:

  * **L (labeling).** A hierarchical, descriptive label per operation,
    `L(op) = (L1, L2, ..., Lm)` -- e.g. `("MEMORY","op","mul")`. Composition
    preserves labels: `L(γ_L(...)) = f_L(L(parent), L(child_1), ...)`.
  * **I (indexing).** A content-addressed index per operation -- the FNV
    fingerprint of `(name, label, child indexes)`. Composition keeps it
    consistent: `I(γ_L(...)) = f_I(...)`. Content addressing (not random UUIDs)
    is the discipline that keeps the layer deterministic: two structurally
    identical operations get the *same* index, so CSE / the liked-pair identity
    `a = a` falls out for free, and reproducibility (manifest equality) is
    preserved.

**Where this lives in the layer stack.** Labels and indexes are *interpretive*
metadata, quarantined on the graded side of the two-truth separation (§14): they
may *inform* planning, retrieval, and debugging but are **never** read by the
R-laws. The lower IR (the StreamPack, the realized plan) carries decisions, not
labels -- exactly the "disable labeling on the hot path" case. So this whole
module is conditionally activatable (`enable_labeling` / `enable_indexing`),
matching the cost/benefit tiering: full on the memory interface, selective in
pipelines, off on the hot path.

Integrity is checkable: `verify.verify_enriched` witnesses label consistency,
content-addressed index uniqueness, and mapping integrity (`Trace` never
dangles) -- the higher layer's own R13-style law, the analog of `verify_memory`
for the enriched structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .egraph import Expr
from .provenance import _fnv

_NO_INDEX = 0


# --- labeling / indexing functions (the preservation laws f_L, f_I) --------------


def f_label(parent_label: tuple, child_labels: tuple) -> tuple:
    """Label preservation: the composed operation's label is a systematic function
    of its constituents' labels. The operator's own hierarchical label dominates;
    the children's *head categories* are appended so the label genuinely depends on
    what it composed (and stays deterministic)."""
    heads = tuple(cl[0] if cl else "?" for cl in child_labels)
    return tuple(parent_label) + ((heads,) if heads else ())


def f_index(name: str, label: tuple, child_indexes: tuple) -> int:
    """Index consistency: a content-addressed index over (name, label, children).
    Deterministic and composition-aware -- identical content => identical index
    (CSE), so the index is the liked-pair identity made explicit. Uses the same
    FNV the manifest/memory use, so indexes interoperate with fingerprints."""
    return _fnv("op", name, tuple(label), tuple(child_indexes))


# --- the enriched operation + 2-cell ---------------------------------------------


@dataclass(frozen=True)
class EnrichedOp:
    """An n-ary operation enriched with a label and a content-addressed index, plus
    its children (the SourceMap: index -> child indexes) and its local position
    within its parent's composition (the composite index `(parent, local)`)."""

    name: str
    arity: int
    label: tuple  # hierarchical label (L1, L2, ...) -- () when labeling is off
    index: int  # content-addressed index -- _NO_INDEX when indexing is off
    children: tuple = ()  # child indexes (the operation-tree edges)
    local: int = 0  # position within the parent's child list


@dataclass(frozen=True)
class TwoCell:
    """A 2-morphism: a transformation between operations (an e-graph rewrite made
    explicit), carrying the rule that justifies it. `src ⇒ dst` by `rule`."""

    src: int
    dst: int
    rule: tuple  # the rewrite's hierarchical label, e.g. ("REWRITE","factor")


# --- the enriched operad (the system) --------------------------------------------


class EnrichedOperad:
    """A content-addressed operad of labeled, indexed operations with traceback.

    Conditionally activatable: `enable_labeling` / `enable_indexing` toggle the
    metadata per the cost/benefit tiering (full on the memory interface, off on
    the hot path). With indexing off, operations are keyed by insertion order."""

    def __init__(self, enable_labeling: bool = True, enable_indexing: bool = True):
        self.enable_labeling = enable_labeling
        self.enable_indexing = enable_indexing
        self.ops: dict[int, EnrichedOp] = {}
        self.cells: list[TwoCell] = []
        self._counter = 0

    # add / compose (γ_L, η_L) ----------------------------------------------------

    def add(
        self, name: str, arity: int = 0, label: tuple = None, children: tuple = (), local: int = 0
    ) -> int:
        """Add an operation (η_L for a leaf, γ_L for a composite). The index is
        content-addressed over (name, label, children); identical operations
        collapse to one entry -- CSE / the liked-pair memory."""
        label = self._resolve_label(name, label)
        index = self._resolve_index(name, label, tuple(children))
        # content addressing: an identical op is already present (a liked pair).
        if index in self.ops:
            return index
        self.ops[index] = EnrichedOp(
            name=name, arity=arity, label=label, index=index, children=tuple(children), local=local
        )
        return index

    def compose(self, name: str, children: tuple, label: tuple = None, local: int = 0) -> int:
        """γ_L: compose `children` under operation `name`, preserving labels
        (`f_label`) and indexes (`f_index`)."""
        child_labels = tuple(self.ops[c].label for c in children if c in self.ops)
        base = self._resolve_label(name, label)
        composed_label = f_label(base, child_labels) if self.enable_labeling else ()
        return self.add(
            name, arity=len(children), label=composed_label, children=tuple(children), local=local
        )

    def record_rewrite(self, src: int, dst: int, rule) -> TwoCell:
        """Record a 2-morphism (a rewrite) between two operations -- transformation
        tracking with label inheritance (the higher-category layer)."""
        cell = TwoCell(
            src=src, dst=dst, rule=tuple(rule) if not isinstance(rule, str) else ("REWRITE", rule)
        )
        self.cells.append(cell)
        return cell

    # queries / mapping -----------------------------------------------------------

    def get(self, index: int) -> EnrichedOp:
        return self.ops.get(index)

    def source_map(self) -> dict:
        """index -> child indexes (the operation-tree edges E)."""
        return {i: op.children for i, op in self.ops.items()}

    def sources(self, index: int) -> tuple:
        """The immediate constituent operations of `index`."""
        op = self.ops.get(index)
        return op.children if op else ()

    def trace(self, index: int) -> list:
        """Reverse mapping: `index` and all its transitive constituent operations,
        deterministic preorder, each once -- map any operation back to its
        sources."""
        out: list = []
        seen: set = set()

        def walk(i: int) -> None:
            op = self.ops.get(i)
            if op is None or i in seen:
                return
            seen.add(i)
            out.append(op)
            for c in op.children:
                walk(c)

        walk(index)
        return out

    # validation (the integrity rules) -------------------------------------------

    def validate(self, index: int) -> bool:
        """An operation is well-formed iff (when active) its label is a non-empty
        hierarchical tuple, its content index agrees with its content, and every
        child resolves."""
        op = self.ops.get(index)
        if op is None:
            return False
        if self.enable_labeling and (not isinstance(op.label, tuple) or not op.label):
            return False
        if self.enable_indexing and f_index(op.name, op.label, op.children) != op.index:
            return False
        return all(c in self.ops for c in op.children)

    def problems(self) -> list:
        """Every (index, reason) integrity violation -- the basis of verify_enriched."""
        out: list = []
        for i, op in self.ops.items():
            if self.enable_labeling and (not isinstance(op.label, tuple) or not op.label):
                out.append((i, "label is not a non-empty hierarchical tuple"))
            if self.enable_indexing and f_index(op.name, op.label, op.children) != op.index:
                out.append((i, "content-addressed index does not match content"))
            for c in op.children:
                if c not in self.ops:
                    out.append((i, f"child index {c} does not resolve (dangling source)"))
        return out

    # internals -------------------------------------------------------------------

    def _resolve_label(self, name: str, label: tuple) -> tuple:
        if not self.enable_labeling:
            return ()
        if label is not None:
            return tuple(label)
        return ("OP", name)

    def _resolve_index(self, name: str, label: tuple, children: tuple) -> int:
        if not self.enable_indexing:
            self._counter += 1
            return self._counter
        return f_index(name, label, children)


# --- the bridge: lift a memory module into an enriched operad ---------------------


def _op_label(expr: Expr) -> tuple:
    """The hierarchical label for a memory operation, by kind."""
    if expr.op == "const":
        return ("MEMORY", "const", str(expr.val))
    if expr.op == "var":
        return ("MEMORY", "var", str(expr.val))
    if expr.op == "module":
        return ("MEMORY", "module", str(expr.val))
    return ("MEMORY", "op", expr.op)


def enrich_memory(memory, *, enable_labeling: bool = True, enable_indexing: bool = True):
    """Lift a frozen memory module (`kbcir.memory.MemoryModule`, whose `.expr` is a
    resolved operation tree) into an enriched operad: every operation gains a
    hierarchical label and a content-addressed index, and the tree's edges become
    the SourceMap. Returns `(operad, root_index)`.

    This is the higher-intelligence view of memory: the frozen fixpoint becomes a
    labeled, indexed, traceable structure -- queryable by label, navigable by
    index, with `trace` recovering the sources of any node -- while the memory
    module itself stays the deterministic, generation-tagged artifact."""
    operad = EnrichedOperad(enable_labeling=enable_labeling, enable_indexing=enable_indexing)

    def build(expr: Expr, local: int = 0) -> int:
        child_indexes = tuple(build(a, i) for i, a in enumerate(expr.args))
        return operad.add(
            expr.op,
            arity=len(expr.args),
            label=_op_label(expr),
            children=child_indexes,
            local=local,
        )

    root = build(memory.expr)
    return operad, root
