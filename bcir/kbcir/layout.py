"""G3: cost-model-driven SoA<->AoS layout pivot for a multi-field tensor resource.

``Resource.layout`` defaults to ``"soa"`` (``model/graph.py``) and -- until this module -- was NEVER
analyzed or transformed: the MLIR lowering hard-coded ``#bcir.layout<soa>``, the C emit assumed a single
flat array, and no pass ever asked whether a Structure-of-Arrays or an Array-of-Structures layout is
CHEAPER for the claims that touch a given resource. G3 builds that pivot: it analyzes a multi-field
resource's access pattern, PRICES SoA vs AoS through the EXISTING K_BCIR cost model (the SAME memory/stride
terms ``realize.candidates_for`` scores every other claim by -- NOT a bespoke discount), chooses the
minimum-cost layout as a priced plan, and records it in a proof-carrying ``LayoutCertificate``. The hard
invariant the tests discharge: a layout change NEVER changes the computed values -- it only changes the
memory addressing / access cost (``lower.layout_kernel`` proves SoA result == AoS result, bit-exact).

THE CONCEPT (the two layouts of an F-field record array of N records):

  * **SoA** (structure-of-arrays): each field is its OWN contiguous array (F arrays of N). A claim that
    SWEEPS ONE FIELD across all records walks one array at UNIT stride -- vectorizable, cheap. A claim that
    touches ALL F fields of one record must GATHER across the F arrays -- costlier (the fields of a record
    are F arrays apart).
  * **AoS** (array-of-structures): the records are interleaved (one array of N records, each F fields wide).
    A WHOLE-RECORD access is contiguous/UNIT -- cheap. A SINGLE-FIELD SWEEP is STRIDED (stride = the record
    width F) -- it skips F-1 fields between the values it wants, so it cannot vectorize and pays the stride
    penalty.

So the two access shapes have MIRRORED costs under the two layouts, and which layout wins is exactly which
access SHAPE dominates the workload. We do not assert that preference -- we PRICE it.

THE LOAD-BEARING DESIGN CONSTRAINT -- the layout choice is the cost model's PRICED decision, NOT a hardcoded
preference. We reuse the EXISTING machinery and let it price the choice:

  * A single-field sweep is modeled as a claim with ``stride_class = UNIT`` under its FAVORABLE layout (SoA)
    and ``stride_class = STRIDED, stride_k = F`` under its UNFAVORABLE layout (AoS) -- and a whole-record
    access is the exact mirror (UNIT under AoS, STRIDED under SoA). ``realize.candidates_for`` / ``_cost``
    then price each shape: a UNIT claim pays ``base_overhead * 1`` per access and vectorizes to the lane
    width; a STRIDED claim pays ``base_overhead * min(stride_k, cacheline/elem_bytes)`` AND is pinned scalar
    (no vector candidate for STRIDED), so its memory term is genuinely, monotonically larger. That delta IS
    the SoA-vs-AoS price -- the optimizer's own ``_stride_penalty`` term, applied by the optimizer's own
    candidate algebra. NOTHING in this module invents a discount; it sets up the two claim SHAPES the cost
    model already knows how to score and sums the cheapest candidate's memory cost across the workload.

  * The per-layout cost of the resource is ``sum over the claims that touch it`` of (that claim's cheapest
    memory cost under the layout) weighted by the claim's ``count`` (records swept). The minimum-cost layout
    wins; ties keep the DECLARED/default layout (SoA), so a balanced or single-field workload stays a clean
    no-op on the default.

LEGALITY / no-op (mirroring ``fusion``/``calling_side``):
  * a resource with ONE field (``fields <= 1``) has no SoA/AoS distinction -- a clean no-op (keep SoA).
  * a resource no claim touches, or an all-single-field-sweep workload, prices SoA <= AoS -> keep SoA.
  * the pivot to AoS is adopted ONLY on a STRICT cost win (whole-record access dominates).

REFERENCE-INVARIANCE (proved in test_layout_pivot.py + ``lower.layout_kernel``): the SAME multi-field
computation emitted under SoA and under AoS produces BIT-IDENTICAL results -- only the addressing
(``soa_index`` = field*N + r  vs  ``aos_index`` = r*F + field) differs, never the value. That invariance is
the correctness backbone; the certificate records only the COST decision.

Scope -- DISTINCT from B3 (calling_side.py): B3 tunes the CALLING side of a WRAPPED kernel (cblas
major-order / prefetch / channel) around a trusted library call. G3 is the GENERAL memory layout of a
multi-field tensor/struct resource IN THE CLAIM GRAPH -- no wrapped call, no calling side. The two share the
"price a choice through the existing model, certify it, prove the math is unchanged" shape but touch
disjoint mechanisms. No new globally-numbered R-law; no MLIR C++ change (the oracle prototype + the Python
emitter, as B1/B5's MLIR wiring was kept separate -- an MLIR ``#bcir.layout`` pass is the follow-on).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Claim, Lane, Opcode, StrideClass
from .cost import MEMORY, TargetProfile
from .realize import candidates_for

# Layout tokens -- the spelling matches ``Resource.layout`` (model/graph.py) and the MLIR
# ``#bcir.layout<...>`` attribute, so a chosen layout flows straight into the resource + the emit.
SOA = "soa"   # structure-of-arrays: one contiguous array per field (the declared default)
AOS = "aos"   # array-of-structures: records interleaved, each F fields wide

# The two access shapes a claim can have over a multi-field resource.
SINGLE_FIELD = "single_field_sweep"   # one field, all records   (UNIT under SoA, STRIDED under AoS)
WHOLE_RECORD = "whole_record"         # all F fields, per record  (UNIT under AoS, STRIDED under SoA)


@dataclass(frozen=True)
class FieldAccess:
    """How ONE claim accesses a multi-field resource: a single-field sweep or a whole-record access, over
    ``records`` records of an ``fields``-field layout. ``shape`` is SINGLE_FIELD or WHOLE_RECORD."""

    claim_id: int
    shape: str               # SINGLE_FIELD | WHOLE_RECORD
    records: int             # how many records this claim sweeps (the cost-model `count`)
    fields: int              # the record width F (fields per record)

    def stride_class_under(self, layout: str) -> StrideClass:
        """The access's stride class under ``layout`` -- the cost-model shape the EXISTING ``candidates_for``
        prices. UNIT when the layout makes the access contiguous, STRIDED (step = F) when it interleaves it:

          * SINGLE_FIELD: contiguous under SoA (the field's own array), strided under AoS (step over F).
          * WHOLE_RECORD: contiguous under AoS (the record is local), strided under SoA (fields F arrays apart).
        """
        contiguous = (layout == SOA and self.shape == SINGLE_FIELD) or \
                     (layout == AOS and self.shape == WHOLE_RECORD)
        return StrideClass.UNIT if contiguous else StrideClass.STRIDED

    def stride_k_under(self, layout: str) -> int:
        """The element stride under ``layout``: 1 when contiguous, the record width F when interleaved (the
        single-field sweep skips F-1 fields per step under AoS; the whole-record gather is F arrays apart
        under SoA -- modeled with the same F stride so the cost model prices the locality loss)."""
        return 1 if self.stride_class_under(layout) == StrideClass.UNIT else max(2, self.fields)


@dataclass(frozen=True)
class LayoutPlan:
    """The cost-model-priced layout for ONE multi-field resource. ``layout`` is the chosen SoA/AoS (the
    minimum-cost one); ``soa_cost``/``aos_cost`` are the priced memory cost of the whole workload under each
    layout (summed over the claims that touch the resource, through the EXISTING cost model). Applying the
    plan changes ONLY addressing -- never the computed values (the invariance the tests prove)."""

    rid: int
    fields: int              # the record width F (1 => single-field, a no-op resource)
    layout: str              # the chosen layout: "soa" | "aos"
    soa_cost: int            # the workload's priced memory cost under SoA
    aos_cost: int            # ... and under AoS
    accesses: tuple[FieldAccess, ...]

    @property
    def is_soa(self) -> bool:
        return self.layout == SOA

    @property
    def is_aos(self) -> bool:
        return self.layout == AOS

    @property
    def chosen_cost(self) -> int:
        return self.aos_cost if self.is_aos else self.soa_cost

    @property
    def alt_cost(self) -> int:
        return self.soa_cost if self.is_aos else self.aos_cost


@dataclass(frozen=True)
class LayoutCertificate:
    """The proof-carrying record of one layout decision -- modeled on ``calling_side.CallingSideCertificate``
    / ``fusion.FusionCertificate``: the resource, the chosen layout, the SoA-vs-AoS cost, and the modeled
    win. A clean NO-OP (keep the declared/default SoA, gain 0) when SoA is already optimal or the resource
    has one field. The certificate asserts NOTHING about the numeric result -- that invariance is the
    separate, stronger gate the tests + ``lower.layout_kernel`` discharge (SoA result == AoS result,
    bit-exact)."""

    plan: LayoutPlan
    declared_layout: str     # the resource's declared/default layout ("soa") -- the baseline we pivot from

    @property
    def rid(self) -> int:
        return self.plan.rid

    @property
    def chosen_layout(self) -> str:
        return self.plan.layout

    @property
    def soa_cost(self) -> int:
        return self.plan.soa_cost

    @property
    def aos_cost(self) -> int:
        return self.plan.aos_cost

    @property
    def gain(self) -> int:
        """The modeled memory-cost reduction (>= 0): declared-layout cost minus the chosen-layout cost. The
        win is pure addressing -- the computed values are unchanged (the separate, stronger invariance)."""
        declared_cost = self.soa_cost if self.declared_layout == SOA else self.aos_cost
        return declared_cost - self.plan.chosen_cost

    @property
    def is_noop(self) -> bool:
        """True iff the pivot changed nothing the cost model prices: the chosen layout IS the declared
        default (SoA) and there was no cost reduction -- the clean no-op the design requires (a single-field
        resource, an SoA-optimal / tied workload, or a one-field resource)."""
        return self.chosen_layout == self.declared_layout and self.gain == 0


def _access_memory_cost(access: FieldAccess, layout: str, target: TargetProfile) -> int:
    """The priced memory cost of ONE access under ONE layout, through the EXISTING cost model. We build the
    claim SHAPE the layout induces (UNIT or STRIDED with stride_k = F) and ask ``realize.candidates_for`` for
    its candidates, taking the CHEAPEST candidate's memory axis -- exactly the min over realizations the
    optimizer itself does. The whole point: a UNIT access vectorizes + pays ``base_overhead*1``, a STRIDED
    one is pinned scalar + pays ``base_overhead*min(F, cacheline/elem)``, so the memory term is genuinely,
    monotonically larger under the unfavorable layout. This is the optimizer's OWN ``_stride_penalty`` term,
    not a bespoke layout discount."""
    sc = access.stride_class_under(layout)
    sk = access.stride_k_under(layout)
    claim = Claim(id=access.claim_id, opcode=Opcode.ADD, lane=Lane.U, stride_class=sc,
                  stride_k=sk, count=max(1, access.records), rd=(1, 2), wr=(3,))
    cands = candidates_for(claim, target)
    return min(c.base.v[MEMORY] for c in cands)


def _layout_cost(accesses: tuple[FieldAccess, ...], layout: str, target: TargetProfile) -> int:
    """The workload's total priced memory cost under ``layout``: the sum over the accesses of each access's
    cheapest memory cost (``_access_memory_cost``). Summing the per-claim minimum is the min,+ composition
    the realize optimizer uses ALONG a plan -- here over the claims that share the resource."""
    return sum(_access_memory_cost(a, layout, target) for a in accesses)


def plan_layout(rid: int, fields: int, accesses: tuple[FieldAccess, ...] | list[FieldAccess],
                target: TargetProfile | None = None) -> LayoutPlan:
    """K_BCIR's deterministic SoA<->AoS choice for one multi-field resource. Prices the whole workload
    (``accesses``) under SoA and under AoS through the EXISTING cost model (``_layout_cost`` ->
    ``candidates_for``'s memory/stride terms) and picks the MINIMUM-cost layout. Ties keep the declared
    default SoA (so a balanced or single-field-dominated workload stays a no-op). A resource with one field
    (``fields <= 1``) has no SoA/AoS distinction -> SoA, both costs equal (a clean no-op).

    Deterministic + pure-Python: reproducible for a given (rid, fields, accesses, target)."""
    target = target or TargetProfile.for_host()
    accesses = tuple(accesses)
    if fields <= 1 or not accesses:
        # No multi-field distinction (or nothing touches it): SoA, equal costs, a clean no-op.
        cost = _layout_cost(accesses, SOA, target) if accesses else 0
        return LayoutPlan(rid=rid, fields=max(1, fields), layout=SOA,
                          soa_cost=cost, aos_cost=cost, accesses=accesses)
    soa_cost = _layout_cost(accesses, SOA, target)
    aos_cost = _layout_cost(accesses, AOS, target)
    # min,+ selection over the two layouts; AoS adopted ONLY on a STRICT win (ties keep the default SoA).
    layout = AOS if aos_cost < soa_cost else SOA
    return LayoutPlan(rid=rid, fields=fields, layout=layout,
                      soa_cost=soa_cost, aos_cost=aos_cost, accesses=accesses)


def certify_layout(plan: LayoutPlan, declared_layout: str = SOA) -> LayoutCertificate:
    """Emit the proof-carrying ``LayoutCertificate`` for a chosen plan: the resource, the chosen layout, the
    SoA-vs-AoS cost, and the modeled win (the declared-layout cost minus the chosen-layout cost), all through
    the existing cost model. A clean no-op certificate (gain 0) when SoA is already optimal. The NUMERIC
    invariance (SoA result == AoS result) is the separate, stronger gate the tests + ``lower.layout_kernel``
    prove."""
    return LayoutCertificate(plan=plan, declared_layout=declared_layout)


def analyze_resource_accesses(module, rid: int, fields: int) -> tuple[FieldAccess, ...]:
    """Classify how each claim in ``module`` accesses resource ``rid`` into the two field-access shapes the
    layout pivot prices. The shape is read from the claim's stride class: a claim declared ``WHOLE`` (its
    op string carries ``:whole_record``, or it is TILE-shaped over the record) accesses all F fields of each
    record -- a WHOLE_RECORD access; anything else sweeping the resource is a SINGLE_FIELD sweep (the common
    elementwise case -- one field across the records). ``records`` is the claim's ``count``.

    This is the access-pattern ANALYSIS step: it does not change the graph, it reads each claim's declared
    intent and produces the ``FieldAccess`` list ``plan_layout`` prices. A claim that does not touch ``rid``
    is skipped."""
    out: list[FieldAccess] = []
    for ph in module.phases:
        for claim in ph.claims:
            if rid not in claim.io_rids():
                continue
            op = claim.op or ""
            whole = op.endswith(":whole_record") or ":whole_record" in op or \
                claim.stride_class == StrideClass.TILE
            shape = WHOLE_RECORD if whole else SINGLE_FIELD
            out.append(FieldAccess(claim_id=claim.id, shape=shape,
                                   records=max(1, claim.count), fields=fields))
    return tuple(out)


def pivot_resource_layout(module, rid: int, fields: int, target: TargetProfile | None = None
                          ) -> tuple[LayoutPlan, LayoutCertificate]:
    """The one-call G3 entry point for a resource living in a ``module``: analyze how the module's claims
    access resource ``rid`` (``analyze_resource_accesses``), price SoA vs AoS (``plan_layout``), and certify
    the choice (``certify_layout``). Returns the (plan, certificate). The plan's chosen layout is what the
    emit honors (``lower.layout_kernel.emit_layout_kernel_c`` / the Resource's ``layout`` field); the
    certificate is the proof-carrying record. A single-field-dominated workload yields a clean SoA no-op; a
    whole-record-dominated one pivots to AoS with a positive gain."""
    accesses = analyze_resource_accesses(module, rid, fields)
    declared = module.resource(rid).layout if module.resource(rid) is not None else SOA
    plan = plan_layout(rid, fields, accesses, target)
    cert = certify_layout(plan, declared_layout=declared)
    return plan, cert
