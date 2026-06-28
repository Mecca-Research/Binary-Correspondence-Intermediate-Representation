"""G4 (Pillar 3a): a cache-line / memory-bank-conflict predictor as a FROZEN, informs-only cost SIGNAL.

The vision-alignment audit (docs/VISION_ALIGNMENT_AUDIT.md, Pillar 3a) flags that BCIR's ``memory`` cost is
a STATIC Q8 formula (bandwidth/latency factors + a stride penalty) with NO model that PREDICTS cache-line
utilization or memory-bank conflicts from a claim's access SHAPE. The vision wants a "Layer-1 AI" that
"analyzes the mathematical graph shapes and predicts cache-line utilization and memory-bank conflicts." This
module is that predictor in BCIR's disciplined form: a small, deterministic, analytic model that computes a
cache-line-utilization + bank-conflict cost SIGNAL from a claim's (stride, element size, count) and FROZEN Q8
cache/bank parameters, and feeds it into the cost model's plan ranking on the CONTENTION axis.

THE TWO-TRUTH / QUARANTINE CONSTRAINT (the non-negotiable discipline -- see ``twotruth.py``): a predicted
cache/bank cost is a GRADED signal. It may *inform* which realization is cheapest (rank plans), but it MUST
NEVER change an R1-R21 legality verdict and MUST NEVER gate legality. Like the existing learned organs (the
``allocator`` placer, ``sensing``/``calibrate``, ``microbench`` ratios) it is deterministic, frozen to Q8,
and stays STRICTLY off the legality path. The verifier (``verify.verify_plan``) checks lane legality (R9) and
12-d cost completeness + non-negativity (R8) -- it NEVER reads the CONTENTION axis, so a conflict-prone plan
and a conflict-free plan are EXACTLY as legal. The predictor only moves the CONTENTION cost dimension; a
legality verdict is identical with or without it. ``test_cache_predict.py`` discharges that proof.

OPT-IN / GAINS-ONLY (so existing pinned conformance scores are UNPERTURBED): NOTHING here touches
``realize._cost`` / ``candidates_for`` / ``optimize``. The base cost vectors the default planner builds NEVER
set the CONTENTION axis (grep confirms: no ``contention=`` in any base cost). This predictor is a SEPARATE,
opt-in cost contribution: a caller that wants the signal calls ``contention_cost(...)`` / ``rank_realizations
(...)`` explicitly and ADDS it to a candidate's cost vector. By default it is never consulted, so every pinned
score in the suite stays byte-identical -- exactly the opt-in shape of the allocator / sensing / dvfs organs
(the ``test_perf`` guard keeps them off the simple path).

THE ANALYTIC MODEL (deterministic, FROZEN Q8 constants -- the "lightweight deterministic predictor"):

  * **cache-line-utilization waste.** A contiguous (unit-stride) sweep packs ``cacheline/elem_bytes`` useful
    elements into every line it touches -- full utilization, zero waste. A strided/gather access touches MORE
    cache lines per useful element: each step lands ``stride*elem_bytes`` bytes away, so once the step exceeds
    a line it touches a fresh line PER element while using only one element of it. We price the waste as the
    Q8 ratio ``lines_touched / useful_lines`` minus 1.0 (a unit sweep => ratio 1.0 => 0 waste); it saturates
    at ``elems_per_line`` (one line per element -- a full gather), the classic worst case.

  * **bank conflicts.** Interleaved memory has ``banks`` (== ``TargetProfile.mem_channels``, the concurrent
    bandwidth-bound streams) and a consecutive element advances one bank. An access whose element stride is a
    MULTIPLE of the bank count (the classic power-of-2 stride == a multiple of the #banks) keeps hitting the
    SAME bank -- the accesses SERIALIZE instead of spreading across banks. We price the conflict by the number
    of distinct banks the stride reaches, ``banks / gcd(stride, banks)``: a unit / coprime stride reaches all
    ``banks`` banks (no conflict -> 0 penalty), a stride sharing a factor g with the bank count reaches only
    ``banks/g`` banks and serializes by the missing factor (the conflict degree). A stride == #banks reaches 1
    bank -> maximal serialization.

Both terms are scaled by the access ``count`` (more elements => more lines wasted / more serialized accesses)
and summed into one CONTENTION cost. Everything is integer Q8; the same (claim, params) always yields the
same cost (a hard BCIR reproducibility requirement).

A trained/calibrated refinement could later REPLACE the analytic constants (``cacheline``, ``banks``, the
waste/conflict weights) with measured Q8 values under ``cost.cal_gen`` -- EXACTLY as the cost ratios in
``microbench``/``calibrate`` already do (the ``CachePredictor`` carries a ``gen`` provenance tag for that). We
SHIP the analytic frozen model; the calibrated path is the documented follow-on.

Scope: oracle prototype + the priced signal + the quarantine proof. An MLIR cost-pass port (a
``bcir.contention`` cost producer feeding ``BCIRCostModel``) mirrors this analytic model with the SAME frozen
constants -- the separate G4-MLIR follow-up. No new globally-numbered R-law; no MLIR C++ change.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from ..model import Claim, StrideClass
from .cost import CONTENTION, CostVector, TargetProfile

Q8 = 256  # 1.0 in Q8 fixed point


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# --- the access shape the predictor reasons over --------------------------------------------------

@dataclass(frozen=True)
class AccessShape:
    """The plan-time access SHAPE the predictor prices: a claim touches ``count`` elements of
    ``elem_bytes`` each at element stride ``stride`` (1 == contiguous/unit sweep). This is read off a claim's
    declared geometry (``from_claim``) -- no new model field; the SHAPE is the cost-model lever."""

    stride: int              # element stride (1 == unit/contiguous; >1 == strided; large == gather-like)
    elem_bytes: int          # bytes per element
    count: int               # elements touched

    @staticmethod
    def from_claim(claim: Claim, target: TargetProfile) -> "AccessShape":
        """The access shape a claim declares, mapped onto (stride, elem_bytes, count) the predictor prices:

          * UNIT / SCALAR / TILE  -> stride 1 (contiguous: full cache-line utilization, all banks reached);
          * STRIDED               -> the claim's ``stride_k`` (the element step it skips by);
          * CACHELINE             -> stride 1 cache-line-locally, but indexed within the line (modeled unit);
          * RANDOM                -> a gather: modeled at a stride that touches a fresh line per element
                                     (``cacheline/elem_bytes``), the worst-case utilization the gather pays.

        Deterministic; reads only the claim's declared fields + the target's ``elem_bytes``."""
        sc = claim.stride_class
        eb = max(1, target.elem_bytes)
        count = max(1, claim.count)
        if sc == StrideClass.STRIDED:
            stride = max(1, claim.stride_k)
        elif sc == StrideClass.RANDOM:
            # a gather touches an unpredictable line per element -> model the worst-case line-per-element step.
            stride = max(1, target.cacheline // eb)
        else:
            # UNIT / SCALAR / TILE / CACHELINE: contiguous (or cache-line-local) -> unit stride.
            stride = 1
        return AccessShape(stride=stride, elem_bytes=eb, count=count)


# --- the frozen Q8 predictor (analytic; cache-line waste + bank conflict) -------------------------

@dataclass(frozen=True)
class CachePredictor:
    """A FROZEN, deterministic cache-line / bank-conflict predictor (the L1 analytic artifact).

    FROZEN Q8 parameters (the cache/bank geometry the analytic model reasons over):
      * ``cacheline``  -- the cache-line size in bytes (how many contiguous bytes one fetch brings in);
      * ``banks``      -- the interleaved memory-bank / channel count (== ``TargetProfile.mem_channels``);
      * ``waste_weight`` / ``conflict_weight`` -- Q8 scales mixing the two terms into the CONTENTION cost.

    ``gen`` is the R13-style provenance generation (0 = the seeded analytic constants; >=1 would tag a
    calibrated refinement that replaced them under ``cost.cal_gen``). Integer-only forward: the same shape +
    params always yield the same Q8 cost (reproducible, host-independent)."""

    cacheline: int = 64          # bytes per cache line (frozen analytic constant)
    banks: int = 4               # interleaved bank/channel count (== mem_channels)
    waste_weight: int = Q8       # Q8 scale on the cache-line-utilization-waste term (x1.0)
    conflict_weight: int = Q8    # Q8 scale on the bank-conflict term (x1.0)
    gen: int = 0                 # provenance: 0 = seeded analytic; >=1 = a calibrated refinement

    @staticmethod
    def for_target(target: TargetProfile) -> "CachePredictor":
        """The predictor whose FROZEN geometry matches a target: its cache-line size and its bank/channel
        count (``mem_channels`` -- the concurrent bandwidth-bound streams that ARE the banks). Seeded
        analytic constants (gen 0); a calibrated table would bump ``gen`` and replace the weights under
        ``cost.cal_gen``."""
        return CachePredictor(cacheline=max(1, target.cacheline),
                              banks=max(1, target.mem_channels), gen=0)

    def _elems_per_line(self, elem_bytes: int) -> int:
        """How many useful elements a single cache line holds (>=1)."""
        return max(1, self.cacheline // max(1, elem_bytes))

    def line_waste_q8(self, shape: AccessShape) -> int:
        """The cache-line-utilization WASTE in Q8 (0 == perfect utilization, no waste).

        A line holds ``epl = cacheline/elem_bytes`` useful elements. A unit-stride sweep packs all ``epl`` of
        them into every line -> utilization 1.0 -> waste 0. A strided access of element step ``s`` lands a
        fresh line every ``ceil(epl / s)`` ... more precisely it touches ``min(epl, s)`` distinct lines'-worth
        of "reach" per ``epl`` elements: the fraction of each fetched line actually used is ``1 / min(s, epl)``
        (a step of ``s`` skips ``s-1`` elements, so only 1 of every ``min(s, epl)`` line-resident elements is
        wanted). The WASTE is ``lines_touched/useful_lines - 1`` == ``min(s, epl) - 1`` in Q8 units, saturating
        at ``epl - 1`` (one line per element -- a full gather). Unit stride => 0; the larger the stride the
        more lines are dragged in per useful element."""
        epl = self._elems_per_line(shape.elem_bytes)
        reach = min(max(1, shape.stride), epl)     # distinct lines touched per useful element (>=1)
        return (reach - 1) * Q8                      # Q8 waste ratio above the unit-sweep baseline

    def bank_conflict_q8(self, shape: AccessShape) -> int:
        """The bank-conflict serialization degree in Q8 (0 == conflict-free, reaches all banks).

        Interleaved memory has ``banks`` banks; consecutive elements advance one bank. A stride ``s`` reaches
        ``banks / gcd(s, banks)`` distinct banks: a unit / coprime stride reaches all ``banks`` (no conflict),
        a stride sharing factor ``g`` with the bank count reaches only ``banks/g`` -- it SERIALIZES by the
        missing factor ``g`` (the classic power-of-2 stride == a multiple of #banks hits ONE bank, g == banks).
        We price the conflict by that serialization factor minus 1 (conflict-free => 0): ``g - 1`` in Q8 (g =
        ``gcd(s, banks)``, so ``banks/reached_banks``). Maximal at ``banks - 1`` (a stride == #banks, one bank,
        fully serialized)."""
        s = max(1, shape.stride)
        g = gcd(s, self.banks)          # the shared factor: banks reached == banks/g, serialization == g
        return (g - 1) * Q8             # Q8 conflict degree above the conflict-free (all-banks) baseline

    def contention_q8(self, shape: AccessShape) -> int:
        """The combined per-element CONTENTION signal in Q8: the weighted sum of the cache-line waste and the
        bank-conflict degree. Both are 0 for a unit-stride sweep, so a contiguous access has 0 contention."""
        waste = (self.line_waste_q8(shape) * self.waste_weight) >> 8
        conflict = (self.bank_conflict_q8(shape) * self.conflict_weight) >> 8
        return waste + conflict

    def contention_cost(self, shape: AccessShape) -> int:
        """The integer CONTENTION cost for the WHOLE access: the per-element Q8 contention scaled by the
        element ``count`` (more elements => more wasted lines / more serialized accesses), brought down from
        Q8 to an integer cost the planner sums on the CONTENTION axis. Deterministic; >= 0; 0 for a unit
        sweep (a clean no-op signal on a contiguous access)."""
        per_elem = self.contention_q8(shape)
        return (per_elem * shape.count) >> 8

    def cost_vector(self, shape: AccessShape) -> CostVector:
        """The CONTENTION cost as a CostVector (zero on every other axis) -- the additive contribution a
        caller ADDS to a candidate's base cost to consult the signal. It touches ONLY the CONTENTION axis, so
        adding it can never perturb the memory/compute/etc. axes the pinned scores are checked against."""
        return CostVector.of(contention=self.contention_cost(shape))

    def predict_claim(self, claim: Claim, target: TargetProfile) -> int:
        """The CONTENTION cost the predictor assigns a claim's declared access shape (the one-call entry):
        read the shape (``AccessShape.from_claim``) and price it. Deterministic + frozen."""
        return self.contention_cost(AccessShape.from_claim(claim, target))

    @property
    def fingerprint(self) -> int:
        """An FNV-1a fingerprint of the frozen parameters (R13 provenance: the same predictor identity =>
        the same predictions)."""
        h = 1469598103934665603
        for v in (self.cacheline, self.banks, self.waste_weight, self.conflict_weight, self.gen):
            h = ((h ^ (v & 0xFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h


# --- the proof-carrying record + the opt-in ranking helper ----------------------------------------

@dataclass(frozen=True)
class ContentionPrediction:
    """The proof-carrying record of ONE contention prediction -- modeled on the ``layout``/``fusion``
    certificates: the access shape, its cache-line waste, its bank-conflict degree, and the combined
    CONTENTION cost. A clean NO-OP (waste 0, conflict 0, cost 0) for a unit-stride sweep. The record asserts
    NOTHING about legality -- it is a GRADED cost signal that only informs plan ranking."""

    shape: AccessShape
    line_waste_q8: int       # Q8 cache-line-utilization waste (0 == perfect)
    bank_conflict_q8: int    # Q8 bank-conflict serialization degree (0 == conflict-free)
    contention_cost: int     # the combined integer CONTENTION cost (scaled by count)
    gen: int                 # the predictor generation that produced it (provenance)

    @property
    def is_conflict_free(self) -> bool:
        """True iff the access neither wastes cache lines nor collides on a bank (a unit-stride sweep)."""
        return self.line_waste_q8 == 0 and self.bank_conflict_q8 == 0

    @property
    def is_noop(self) -> bool:
        """True iff the prediction adds nothing the planner prices -- a contiguous, conflict-free access."""
        return self.contention_cost == 0


def predict(claim: Claim, target: TargetProfile,
            predictor: CachePredictor | None = None) -> ContentionPrediction:
    """Predict a claim's CONTENTION signal: read its access shape and price the cache-line waste + bank
    conflict through the frozen analytic predictor, returning the proof-carrying ``ContentionPrediction``.
    Deterministic + opt-in -- callers consult it explicitly; it NEVER runs on the default plan path. The
    predictor defaults to the one matching ``target``'s frozen cache/bank geometry."""
    pred = predictor or CachePredictor.for_target(target)
    shape = AccessShape.from_claim(claim, target)
    return ContentionPrediction(
        shape=shape,
        line_waste_q8=pred.line_waste_q8(shape),
        bank_conflict_q8=pred.bank_conflict_q8(shape),
        contention_cost=pred.contention_cost(shape),
        gen=pred.gen,
    )


def rank_realizations(candidates, claim: Claim, target: TargetProfile,
                      predictor: CachePredictor | None = None):
    """Rank a claim's candidate realizations by their base scalar cost PLUS the predicted CONTENTION cost,
    so a conflict-prone realization ranks more expensive and a conflict-free one is preferred. This is the
    OPT-IN consultation: the caller passes the candidates (e.g. from ``realize.candidates_for``) and gets them
    back sorted cheapest-first under the contention-augmented cost. It does NOT mutate the candidates or the
    base cost vectors (so the default planner -- which never calls this -- is untouched and every pinned score
    is identical); it returns ``(candidate, base_cost, contention_cost, total)`` tuples for inspection.

    Each candidate is priced by the SHAPE its lane implies: a scalar/strided candidate inherits the claim's
    declared stride, a gather candidate (lane GGG) is priced as the worst-case line-per-element gather (the
    same RANDOM mapping ``AccessShape.from_claim`` uses), so the contention signal distinguishes a strided
    realization from a gathered one of the SAME claim. The base cost is the candidate's summed cost-vector
    magnitude (an order-preserving scalar); the contention cost is ADDED on top. Deterministic: ties break by
    the candidate name for a stable order."""
    from ..model import Lane

    pred = predictor or CachePredictor.for_target(target)
    ranked = []
    for cand in candidates:
        # the realization's effective access shape: a gather lane pays the worst-case gather contention;
        # everything else inherits the claim's declared stride shape.
        if cand.lane == Lane.GGG:
            eb = max(1, target.elem_bytes)
            shape = AccessShape(stride=max(1, target.cacheline // eb), elem_bytes=eb,
                                count=max(1, claim.count))
        else:
            shape = AccessShape.from_claim(claim, target)
        base_cost = sum(cand.base.v)
        cont = pred.contention_cost(shape)
        ranked.append((cand, base_cost, cont, base_cost + cont))
    ranked.sort(key=lambda t: (t[3], t[0].name))
    return ranked
