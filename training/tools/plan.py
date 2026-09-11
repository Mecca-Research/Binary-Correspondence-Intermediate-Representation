"""Retrieval as a planned query: legality first, then a price on BCIR's twelve axes.

`search_chunks.py` used to execute one plan. The backend was a flag, the row set was
always every row, and the text of every chunk was materialized whether or not the
ranking wanted it. Nothing chose anything, so nothing could explain why.

This module makes the choice explicit, in the shape BCIR already uses for a
different question. `bcir/asn1/selection.py` picks an ASN.1 encoding rule by
enumerating candidates, discarding the ones that are not *legal* for the value at
hand, and minimizing over a named cost axis -- with `Objective.WIRE_SIZE` spelled
`"memory"` and `Objective.ENCODE_LATENCY` spelled `"compute.encode"`, because the
objectives are the cost vector's own axes. Choosing between the exact Q15 scan, the
C kernel and the lossy Q8 view is the same shape of decision, over the same twelve
axes, one of which is `accuracy`.

**Legality is decided before cost, and never from a measurement.** The four rules
below are structural properties of a plan, checked the way the verifier's R-laws
are checked: a plan that fails one is not a more expensive plan, it is not a plan.
In particular no amount of speed makes the Q8 view legal for a request that asked
for the exact ranking -- that is `accuracy`, and it is a legality question first
and a priced axis second.

**The prices are modeled, and say so.** The constants in `CostModel` are calibrated
from wall-clock measurements on one host; they are here to *order* plans, not to
predict a duration. Nothing in this module gates on them, no claim of optimality is
made from them (TMSAO-4: heuristic, no claim), and a plan's chosen-ness is always
explainable as "this axis, this number, these candidates".

Declared scope of the predicate language: a conjunction of `column op value` terms
with `op` one of `eq`, `ne`, `in`, `prefix`. No disjunction across columns, no
arithmetic, no user expressions. Answering the next soundness question by adding an
expression evaluator is how a retrieval tool becomes an interpreter nobody asked
for; the boundary is stated here so the answer can point at it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bcir.kbcir.cost import DIMS, CostVector  # noqa: E402  (path set above)

import catalog as catalog_module  # noqa: E402


class PlanError(ValueError):
    """A request that cannot be turned into a plan. Never a silently narrowed one."""


# --------------------------------------------------------------------------
# Objectives -- the cost vector's own axis names, as in bcir/asn1/selection.py
# --------------------------------------------------------------------------


class Objective(Enum):
    """What the caller is short of. The value names the axis it minimizes.

    `EXACTNESS` is not "no objective": it is the named objective whose answer is the
    definition -- the exact Q15 ranking, whatever it costs. It is the degenerate case
    this rail pins, the way `Objective.NONE` pins DER in the ASN.1 selector.
    """

    EXACTNESS = "accuracy"
    LATENCY = "compute"
    FOOTPRINT = "memory"
    STARTUP = "compile"


#: Per-objective weight vectors. One axis carries the objective's weight; the rest
#: carry 1, so a tie on the named axis is broken by total work rather than by
#: declaration order alone.
def _weights(primary: str) -> tuple[int, ...]:
    return tuple(1024 if name == primary else 1 for name in DIMS)


# --------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------

OPERATORS = ("eq", "ne", "in", "prefix")

#: How `--where` spells each operator. Longest first, so `!=` is not read as `=`.
_SPELLINGS = (("!=", "ne"), ("^=", "prefix"), ("=", "eq"))


@dataclass(frozen=True)
class Predicate:
    """One `column op value` term. Total, closed, and comparable.

    `values` is always a tuple, even for `eq`, so the four operators share one shape
    and a caller never has to ask which arity it is holding.
    """

    column: str
    op: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.op not in OPERATORS:
            raise PlanError(f"unknown operator {self.op!r}; the language is {', '.join(OPERATORS)}")
        if not self.values:
            raise PlanError(f"predicate on {self.column!r} has no value")
        if self.op in ("eq", "ne", "prefix") and len(self.values) != 1:
            raise PlanError(f"operator {self.op!r} takes one value, got {len(self.values)}")

    def __str__(self) -> str:
        if self.op == "eq":
            return f"{self.column} = {self.values[0]}"
        if self.op == "ne":
            return f"{self.column} != {self.values[0]}"
        if self.op == "prefix":
            return f"{self.column} ^= {self.values[0]}"
        return f"{self.column} in ({', '.join(self.values)})"


def parse_predicate(text: str) -> Predicate:
    """Parse one `--where` term.

    `subject=llvm`            -> eq
    `subject!=llvm`           -> ne
    `source_path^=training/`  -> prefix
    `subject=llvm,data`       -> in, because a comma-separated right side is a set

    A term with no operator is an error rather than a guess. Guessing here would
    silently widen a query the caller meant to narrow.
    """
    for spelling, op in _SPELLINGS:
        index = text.find(spelling)
        if index > 0:
            column = text[:index].strip()
            raw = text[index + len(spelling) :].strip()
            if not column:
                raise PlanError(f"predicate {text!r} has no column")
            if op == "eq" and "," in raw:
                values = tuple(part.strip() for part in raw.split(",") if part.strip())
                if not values:
                    raise PlanError(f"predicate {text!r} has no value")
                return Predicate(column, "in", values)
            return Predicate(column, op, (raw,))
    raise PlanError(
        f"predicate {text!r} names no operator; write column=value, column!=value, "
        "column^=prefix, or column=one,two for a set"
    )


# --------------------------------------------------------------------------
# Selection -- which rows, and how many the catalog says that is
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """The rows a predicate conjunction admits, plus what the catalog knew in advance.

    `rows` is None when every row is admitted -- the unfiltered case, kept distinct
    from "a filter that happens to match everything" so a plan can price the mask it
    would have to build.

    `estimated` is what the statistics said before any postings were read, and
    `estimate_exact` says whether that number was a count or a bound. Both are
    recorded because the estimate is what the *planner* used, and a plan that was
    chosen on a bound should be explainable as having been chosen on a bound.
    """

    rows: tuple[int, ...] | None
    estimated: int
    estimate_exact: bool
    predicates: tuple[Predicate, ...] = ()

    @property
    def admitted(self) -> int:
        return self.estimated if self.rows is None else len(self.rows)


def estimate(catalog, predicate: Predicate) -> tuple[int, bool]:
    """How many rows a single predicate admits, and whether that count is exact."""
    if predicate.op == "prefix":
        if predicate.column != catalog_module.PATH_COLUMN:
            raise PlanError(
                f"prefix is only defined on {catalog_module.PATH_COLUMN}, not {predicate.column!r}"
            )
        return catalog.count_prefix(predicate.values[0])
    if predicate.op == "eq":
        count = catalog.count_equals(predicate.column, predicate.values[0])
        if count is None:
            raise PlanError(_unindexed(catalog, predicate.column))
        return count, True
    if predicate.op == "in":
        total = 0
        for value in predicate.values:
            count = catalog.count_equals(predicate.column, value)
            if count is None:
                raise PlanError(_unindexed(catalog, predicate.column))
            total += count
        return total, True
    count = catalog.count_equals(predicate.column, predicate.values[0])
    if count is None:
        raise PlanError(_unindexed(catalog, predicate.column))
    return catalog.rows_total - count, True


def _unindexed(catalog, column: str) -> str:
    indexed = ", ".join(catalog.indexed_columns()) + f", {catalog_module.PATH_COLUMN}"
    return f"column {column!r} is not indexed; the catalog indexes {indexed}"


def select(catalog, predicates) -> Selection:
    """Resolve a conjunction of predicates to the exact rows it admits.

    The estimate is computed first, from statistics alone, and the *smallest*
    predicate is evaluated first so the intersection shrinks fastest -- the only
    ordering decision here, and it is made from exact counts rather than a guess.

    Evaluation is exact even where the estimate was a bound: a prefix the index did
    not count still resolves by walking distinct paths, which is over paths, not over
    rows. Pricing may use a bound; answering never does.
    """
    predicates = tuple(predicates)
    if not predicates:
        return Selection(None, catalog.rows_total, True, ())

    estimates = []
    for predicate in predicates:
        count, exact = estimate(catalog, predicate)
        estimates.append((count, exact, predicate))
    estimated = min(count for count, _, _ in estimates)
    estimate_exact = all(exact for _, exact, _ in estimates)

    order = sorted(range(len(estimates)), key=lambda i: (estimates[i][0], i))
    admitted: set[int] | None = None
    for index in order:
        _, _, predicate = estimates[index]
        rows = _rows_for(catalog, predicate)
        admitted = rows if admitted is None else (admitted & rows)
        if not admitted:
            break
    resolved = tuple(sorted(admitted or ()))
    return Selection(resolved, estimated, estimate_exact, predicates)


def _rows_for(catalog, predicate: Predicate) -> set[int]:
    if predicate.op == "prefix":
        return set(catalog.rows_with_prefix(predicate.values[0]))
    if predicate.op == "eq":
        rows = catalog.rows_equal(predicate.column, predicate.values[0])
        if rows is None:
            raise PlanError(_unindexed(catalog, predicate.column))
        return set(rows)
    if predicate.op == "in":
        found: set[int] = set()
        for value in predicate.values:
            rows = catalog.rows_equal(predicate.column, value)
            if rows is None:
                raise PlanError(_unindexed(catalog, predicate.column))
            found |= set(rows)
        return found
    rows = catalog.rows_equal(predicate.column, predicate.values[0])
    if rows is None:
        raise PlanError(_unindexed(catalog, predicate.column))
    return set(range(catalog.rows_total)) - set(rows)


# --------------------------------------------------------------------------
# The cost model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Modeled unit costs, in nanoseconds of work on the calibration host.

    Every field was measured, and the measurement is named beside it. They are here
    to *order* plans; nothing gates on them, and no optimality is claimed from them.
    A host where they are wrong chooses a slower plan, never a wrong answer -- which
    is why legality is decided without them.
    """

    #: One thousand Q15 multiply-accumulates, per backend. Measured as the full-corpus
    #: scan (2,012 x 512) divided by 1,007 thousand-MAC units.
    ns_per_kmac_reference: int = 53_700
    ns_per_kmac_native: int = 884
    ns_per_kmac_q8: int = 4_727

    #: Materializing chunk text. `full` is one parse of every record (the plan that
    #: reads the whole table); `seek` is one recorded span; `open` is per file touched.
    ns_per_row_full_scan: int = 22_400
    ns_per_row_seek: int = 8_000
    ns_per_file_open: int = 20_000

    #: Building a derived column over the whole set, per row.
    ns_per_row_derived: int = 31_000

    #: Producing the kernel: a compiler invocation, or the stamped cache path.
    ns_compile_kernel: int = 281_900_000
    ns_load_cached_kernel: int = 1_900_000

    #: Quantizing the corpus matrix into BCIRQ8, per row. No q8 artifact is persisted,
    #: so this is paid per process by any plan that uses that view.
    ns_per_row_q8_quantize: int = 4_000

    #: Bytes are counted for the `memory` axis directly; this scales them into the
    #: same integer space as the nanosecond fields so one weight vector spans both.
    memory_units_per_byte: int = 1

    #: What a lossy ranking costs on the `accuracy` axis, per requested result. Zero
    #: for the exact backends by construction -- they are the definition.
    accuracy_penalty_per_result: int = 1_000_000

    #: What a two-implementation differential costs on `verification`: it is the
    #: second scan, plus the comparison, and it buys the bit-equality claim.
    verification_units_per_row: int = 1_000


DEFAULT_MODEL = CostModel()

#: Where the defaults came from, so a reader can re-derive or challenge them.
CALIBRATION = {
    "method": "wall clock, warm, best of 3-5, one query against lexical-hash-v1",
    "corpus": "2,012 rows x 512 dims",
    "class": "wall -- indicative, never gating (docs/PERFORMANCE_AUDIT.md)",
    "note": "these order plans; they do not predict a duration and nothing gates on them",
}


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------

BACKENDS = ("reference", "native", "q8", "both")

#: Which backends compute the exact Q15 ranking that is this rail's definition.
EXACT_BACKENDS = frozenset({"reference", "native", "both"})

#: Which backends need a compiled kernel.
COMPILED_BACKENDS = frozenset({"native", "q8", "both"})


@dataclass(frozen=True)
class Plan:
    """One way to answer the request, with its verdict and its price."""

    backend: str
    selection: Selection
    top_k: int
    materialize: str  # "seek" | "full" | "none"
    kernel_cached: bool
    legal: bool
    refusals: tuple[str, ...] = ()
    cost: CostVector = field(default_factory=CostVector.zero)

    @property
    def scanned(self) -> int:
        return self.selection.admitted

    def label(self) -> str:
        where = "" if self.selection.rows is None else f" +where({self.scanned})"
        return f"{self.backend}{where} /{self.materialize}"


# -- legality ---------------------------------------------------------------
#
# Four rules, checked before any cost is computed, each a structural property of
# the plan rather than a measurement of it. The names are the vocabulary an
# explanation uses, so a refusal is always attributable to one rule.

LEGALITY_RULES = ("exactness", "availability", "coverage", "materialization")


def legality(
    plan_backend: str,
    selection: Selection,
    *,
    require_exact: bool,
    available_backends: frozenset,
    rows_total: int,
    materialize: str,
    want_text: bool,
) -> tuple[str, ...]:
    """Every rule this plan breaks, in rule order. Empty means legal."""
    refusals = []
    if require_exact and plan_backend not in EXACT_BACKENDS:
        refusals.append(
            f"exactness: {plan_backend} is a different arithmetic, and the request "
            "asked for the exact Q15 ranking"
        )
    if plan_backend not in available_backends:
        refusals.append(f"availability: the {plan_backend} backend is not reachable on this host")
    if selection.rows is not None:
        if any(not 0 <= row < rows_total for row in selection.rows):
            refusals.append("coverage: the selection names a row outside the set")
    if want_text and materialize == "none":
        refusals.append("materialization: the request wants text and this plan reads none")
    if not want_text and materialize == "full":
        refusals.append(
            "materialization: the request wants no text and this plan reads every row's"
        )
    return tuple(refusals)


# -- pricing ----------------------------------------------------------------


def price(
    backend: str,
    selection: Selection,
    *,
    top_k: int,
    dim: int,
    rows_total: int,
    materialize: str,
    kernel_cached: bool,
    files_touched: int,
    model: CostModel = DEFAULT_MODEL,
) -> CostVector:
    """The twelve-axis cost of running this plan once, in modeled units.

    Only the axes a single-host in-process scan actually spends on are non-zero.
    `fabric`, `sync`, `thermal`, `power`, `reliability`, `security` and `contention`
    are zero here and are left zero rather than invented: an axis with a fabricated
    number is worse than an axis with none, because it will be minimized against.
    """
    scanned = selection.admitted
    # `bcir_ai_q8_rows_dot_f64` is a rows-dot, not a top-k: it takes no eligibility
    # mask, so a q8 plan scores every row whatever the predicate admitted and filters
    # afterwards. Pricing it as though the predicate narrowed its scan would make the
    # planner prefer it for exactly the queries where it helps least.
    scan_rows = rows_total if backend == "q8" else scanned
    kmacs = max(0, (scan_rows * dim + 1023) // 1024)

    per_kmac = {
        "reference": model.ns_per_kmac_reference,
        "native": model.ns_per_kmac_native,
        "q8": model.ns_per_kmac_q8,
        "both": model.ns_per_kmac_reference + model.ns_per_kmac_native,
    }[backend]
    compute = kmacs * per_kmac

    # A pure-Python scan reads its rows through a derived column; the C backends
    # hand the packed codes to the kernel and build nothing.
    if backend in ("reference", "both"):
        compute += rows_total * model.ns_per_row_derived
    if backend == "q8":
        compute += rows_total * model.ns_per_row_q8_quantize

    if materialize == "full":
        compute += rows_total * model.ns_per_row_full_scan
        memory_bytes = rows_total * 1024
    elif materialize == "seek":
        compute += top_k * model.ns_per_row_seek + files_touched * model.ns_per_file_open
        memory_bytes = top_k * 1024
    else:
        memory_bytes = 0

    memory_bytes += scan_rows * dim * 2  # the codes the scan actually reads
    if selection.rows is not None:
        memory_bytes += rows_total  # the eligibility mask, one byte per row

    compile_units = (
        (model.ns_load_cached_kernel if kernel_cached else model.ns_compile_kernel)
        if backend in COMPILED_BACKENDS
        else 0
    )
    accuracy = 0 if backend in EXACT_BACKENDS else top_k * model.accuracy_penalty_per_result
    verification = scanned * model.verification_units_per_row if backend == "both" else 0

    return CostVector.of(
        compute=compute,
        memory=memory_bytes * model.memory_units_per_byte,
        compile=compile_units,
        accuracy=accuracy,
        verification=verification,
    )


# -- enumeration and choice -------------------------------------------------


def candidates(
    catalog,
    selection: Selection,
    *,
    top_k: int,
    dim: int,
    want_text: bool,
    require_exact: bool,
    available_backends: frozenset,
    kernel_cached: bool,
    files_touched: int,
    model: CostModel = DEFAULT_MODEL,
) -> list[Plan]:
    """Every plan considered, legal or not, in declaration order.

    Illegal plans are kept rather than dropped, so an explanation can say *why* a
    plan the reader expected is not the one that ran. A planner that silently omits
    its rejects cannot be argued with.
    """
    plans = []
    materializations = ("seek", "full") if want_text else ("none",)
    for backend in BACKENDS:
        for materialize in materializations:
            refusals = legality(
                backend,
                selection,
                require_exact=require_exact,
                available_backends=available_backends,
                rows_total=catalog.rows_total,
                materialize=materialize,
                want_text=want_text,
            )
            plans.append(
                Plan(
                    backend=backend,
                    selection=selection,
                    top_k=top_k,
                    materialize=materialize,
                    kernel_cached=kernel_cached,
                    legal=not refusals,
                    refusals=refusals,
                    cost=price(
                        backend,
                        selection,
                        top_k=top_k,
                        dim=dim,
                        rows_total=catalog.rows_total,
                        materialize=materialize,
                        kernel_cached=kernel_cached,
                        files_touched=files_touched,
                        model=model,
                    ),
                )
            )
    return plans


def choose(plans, objective: Objective) -> Plan:
    """The cheapest legal plan on the objective's axis. Legality first, always.

    Ties break by the scalarized cost and then by declaration order, so the same
    request against the same catalog picks the same plan on every host -- a planner
    whose choice wobbles cannot be gated.
    """
    legal = [plan for plan in plans if plan.legal]
    if not legal:
        raise PlanError(
            "no legal plan for this request:\n"
            + "\n".join(f"  {plan.label()}: {'; '.join(plan.refusals)}" for plan in plans)
        )
    weights = _weights(objective.value)
    order = {id(plan): index for index, plan in enumerate(plans)}
    return min(legal, key=lambda plan: (plan.cost.dot(weights), order[id(plan)]))


# -- explanation ------------------------------------------------------------


def explain(plans, chosen: Plan, objective: Objective, *, selection: Selection) -> str:
    """EXPLAIN: what was considered, what it would cost, and why this one won."""
    lines = [
        f"EXPLAIN  objective={objective.name.lower()} (minimize {objective.value})",
    ]
    if selection.predicates:
        terms = " AND ".join(str(predicate) for predicate in selection.predicates)
        exactness = "exact" if selection.estimate_exact else "bound"
        lines.append(
            f"  WHERE {terms}\n"
            f"    catalog estimated {selection.estimated} row(s) [{exactness}], "
            f"resolved {selection.admitted}"
        )
    else:
        lines.append(f"  WHERE (none) -- every one of {selection.estimated} row(s)")
    widest = max(len(plan.label()) for plan in plans)
    weights = _weights(objective.value)
    lines.append(f"  {'plan'.ljust(widest)}  {'scalar':>14}  {objective.value:>14}  verdict")
    for plan in plans:
        scalar = plan.cost.dot(weights)
        axis = plan.cost.as_dict()[objective.value]
        if not plan.legal:
            verdict = "REFUSED " + plan.refusals[0].split(":", 1)[0]
        elif plan is chosen:
            verdict = "CHOSEN"
        else:
            verdict = "legal"
        lines.append(f"  {plan.label().ljust(widest)}  {scalar:14d}  {axis:14d}  {verdict}")
    for plan in plans:
        if not plan.legal:
            for refusal in plan.refusals:
                lines.append(f"    {plan.label()}: {refusal}")
    nonzero = {name: value for name, value in chosen.cost.as_dict().items() if value}
    lines.append(
        "  chosen cost vector: "
        + ", ".join(f"{name}={value}" for name, value in sorted(nonzero.items()))
    )
    lines.append(f"  ({CALIBRATION['class']}; {CALIBRATION['note']})")
    return "\n".join(lines)
