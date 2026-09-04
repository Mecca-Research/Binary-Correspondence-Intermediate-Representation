"""G2: autonomous matmul+activation epilogue fusion, priced by the EXISTING tropical optimizer.

The canonical "matmul -> activation" epilogue: a ``gem.matmul`` whose output feeds a ``gem.activation``
(relu/gelu/...). Fusing them applies the activation INLINE as each matmul output element is produced, so the
intermediate C-buffer never materializes -- the matmul output is activated on the fly and only the final
tensor is written. This is the activation analog of the deforestation/CSE fusion ``realize.py`` already
prices for adjacent ELEMENTWISE claims, lifted to the matmul producer / activation consumer pair.

THE LOAD-BEARING DESIGN CONSTRAINT -- fusion is the optimizer's PRICED CHOICE, not a hardcoded rewrite.
We do NOT invent a bespoke "matmul+activation discount". Instead we lean on the fact ``realize.py`` ALREADY
prices the win: when the activation reads the operand the matmul produced **in the same phase**,
``fused_candidates`` couples the activation's cost by ``_DEFOREST_FACTOR`` (the x0.75 memory discount for
eliding the intermediate's write+read round-trip). So the FUSED plan is simply the two claims in one phase
(the intermediate is deforested), and the UNFUSED plan is the same two claims separated by a phase barrier
(the barrier materializes the intermediate -- the comment in ``fused_candidates`` is explicit: "A barrier
between phases materializes intermediates, so both credits are intra-phase only"). We score BOTH through the
unchanged ``optimize`` and adopt fusion iff it strictly wins. The discount is the optimizer's, applied by
the optimizer's own algebra -- this module only sets up the two candidate module shapes and compares them.

LEGALITY (when fusion is NOT applied -- a clean no-op, mirroring ``optimize_bundled``):
  * the activation must be the SOLE consumer of the matmul output (no other claim reads the un-activated
    intermediate -- if something else needs the raw product, the buffer must materialize and fusing would be
    unsound). This is the producer->consumer single-reader rule.
  * the activation must be ELEMENTWISE. ``softmax`` is a last-axis ROW REDUCTION (reduce-max -> exp ->
    normalize needs the whole row before any output element is final), so it CANNOT be applied inline as each
    matmul element drops out -- it is scoped OUT of epilogue fusion here (a documented non-fusible kind), the
    elementwise four (relu/sigmoid/tanh/gelu) fuse.
  * no RAW/WAR/WAW conflict beyond the intended producer->consumer edge (the matmul output is read only by
    the activation; the activation's output is fresh).
  * fusion is adopted only if the priced fused score is STRICTLY LESS than the unfused score (it is a no-op
    when it does not help -- e.g. a degenerate shape where the discount rounds away).

Every adopted fusion emits a ``FusionCertificate`` (modeled on ``bundle.BundleCertificate``): the claims
fused, the before/after plan score, and the win -- a proof-carrying record of the decision.

Scope: this is the oracle realization decision + the fused C kernel emit + verification. Wiring it into an
MLIR ``gem.fused_matmul_activation`` law op is the separate G2-MLIR follow-up (kept separate, as matmul.py /
activation.py's MLIR wiring was). NO new globally-numbered R-law.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..model import Claim, Module, Phase
from .activation import ACTIVATIONS
from .cost import HProfile, Theta
from .realize import RealizationResult, optimize
from .weights import PERF, Policy

# The matmul / activation op tags the detector keys on (the planned-claim op strings matmul.py and
# activation.py declare; the example builders use "gem.matmul" / "gem.activation"). The activation kind
# rides the op string as a ``gem.activation:<kind>`` suffix -- the established BCIR convention for op-string
# payload (cf. ``c.call.libm:expf``, ``c.cast:``), so the kind is carried with NO model change and the bare
# ``gem.activation`` (kind-less) still matches as relu.
_MATMUL_OPS = ("gem.matmul", "linalg.matmul")
_ACTIVATION_OP = "gem.activation"


def _is_activation_op(op: str | None) -> bool:
    """True iff `op` is a gem.activation claim (bare ``gem.activation`` or the ``gem.activation:<kind>``
    suffixed form)."""
    return op == _ACTIVATION_OP or (op or "").startswith(_ACTIVATION_OP + ":")


# Epilogue-fusible activation kinds: the ELEMENTWISE family. softmax is a last-axis reduction (it needs the
# whole row before any output element is final), so it can NOT be applied inline as each matmul element is
# produced -- it is scoped out of epilogue fusion (the documented non-fusible kind).
FUSIBLE_ACTIVATIONS = tuple(k for k in ACTIVATIONS if k != "softmax")
NON_FUSIBLE_ACTIVATIONS = ("softmax",)


def is_fusible_activation(kind: str) -> bool:
    """True iff `kind` is an ELEMENTWISE activation that can be applied inline per matmul output element.
    softmax (a last-axis row reduction) is NOT fusible: its normalizer needs the whole row, so no single
    output element is final until the row's reduce-max + sum are done -- it cannot be an inline epilogue."""
    return kind in FUSIBLE_ACTIVATIONS


@dataclass(frozen=True)
class MatmulActivationEpilogue:
    """A detected matmul -> activation epilogue in one phase: the matmul produces `inter_rid`, which the
    activation (sole consumer) reads and writes to `out_rid`. `act_kind` is the activation family member."""

    phase_id: int
    matmul_id: int
    activation_id: int
    inter_rid: int  # the intermediate (un-activated matmul output) buffer that fusion elides
    out_rid: int  # the activation's output
    act_kind: str  # relu / sigmoid / tanh / gelu / softmax (softmax is non-fusible)

    @property
    def fusible(self) -> bool:
        """The activation kind is elementwise (softmax -- a row reduction -- is excluded)."""
        return is_fusible_activation(self.act_kind)


@dataclass(frozen=True)
class FusionCertificate:
    """The proof-carrying record of one adopted matmul+activation epilogue fusion -- modeled on
    ``bundle.BundleCertificate``: which claims fused, the plan score before (unfused: the intermediate
    materializes at a phase barrier) and after (fused: the intermediate is deforested), and the win."""

    epilogue: MatmulActivationEpilogue
    unfused_score: (
        int  # plan score with the intermediate MATERIALIZED (matmul, then a separate pass)
    )
    fused_score: int  # ... and with it DEFOREST-elided (the inline epilogue)
    act_kind: str

    @property
    def gain(self) -> int:
        return self.unfused_score - self.fused_score

    @property
    def matmul_id(self) -> int:
        return self.epilogue.matmul_id

    @property
    def activation_id(self) -> int:
        return self.epilogue.activation_id


def _act_kind(claim: Claim) -> str:
    """The activation family kind a gem.activation claim declares, parsed from the ``gem.activation:<kind>``
    op-string suffix (the detector reads it to apply the softmax scope-out + the kernel emit). A bare
    ``gem.activation`` with no suffix defaults to ``relu`` (the exact/clean-rail activation)."""
    op = claim.op or ""
    if op.startswith(_ACTIVATION_OP + ":"):
        kind = op.split(":", 1)[1]
        if kind in ACTIVATIONS:
            return kind
    return "relu"


def find_matmul_activation_epilogues(module: Module) -> list[MatmulActivationEpilogue]:
    """Detect every matmul -> activation epilogue: a ``gem.matmul`` claim whose single write RID is read by
    exactly ONE ``gem.activation`` claim in the SAME phase, and read by NO other claim (the activation is the
    SOLE consumer of the un-activated intermediate -- the producer->consumer single-reader legality rule).

    This is the precise legality boundary: if any other claim also reads the matmul's output, that reader
    needs the raw (un-activated) product, so the buffer MUST materialize and the inline epilogue is illegal;
    such a pair is not returned. softmax activations ARE detected (so callers can see them) but flagged
    non-fusible via ``.fusible`` (a last-axis reduction can not be an inline epilogue)."""
    epilogues: list[MatmulActivationEpilogue] = []
    for ph in module.phases:
        # All readers of each rid in this phase (the single-consumer legality check).
        readers: dict[int, list[Claim]] = {}
        for c in ph.claims:
            for rid in set(c.rd):
                readers.setdefault(rid, []).append(c)
        for mm in ph.claims:
            if (mm.op or "") not in _MATMUL_OPS or len(mm.wr) != 1:
                continue
            inter = mm.wr[0]
            consumers = readers.get(inter, [])
            # SOLE consumer + it is a gem.activation. >1 reader -> the raw product is needed elsewhere,
            # so the intermediate must materialize: not a legal inline epilogue.
            if len(consumers) != 1:
                continue
            act = consumers[0]
            if not _is_activation_op(act.op) or len(act.wr) != 1:
                continue
            # The activation must not also write the intermediate (it reads it, writes a fresh output).
            if inter in act.wr:
                continue
            epilogues.append(
                MatmulActivationEpilogue(
                    phase_id=ph.phase_id,
                    matmul_id=mm.id,
                    activation_id=act.id,
                    inter_rid=inter,
                    out_rid=act.wr[0],
                    act_kind=_act_kind(act),
                )
            )
    return epilogues


def _split_phase_at_barrier(module: Module, epi: MatmulActivationEpilogue) -> Module:
    """The UNFUSED module shape: a deep copy where the activation is moved into a NEW downstream phase that
    depends on the matmul's phase -- so a phase BARRIER sits between producer and consumer. The barrier
    materializes the intermediate (``fused_candidates``: "A barrier between phases materializes
    intermediates"), so the deforestation discount does NOT apply -- this is exactly the un-fused cost
    (matmul writes the full C buffer, the activation re-reads it in a separate pass). All other claims keep
    their phase. The fused module is the original (producer + consumer in one phase -> deforested)."""
    m = copy.deepcopy(module)
    src_phase = next(p for p in m.phases if p.phase_id == epi.phase_id)
    act = next(c for c in src_phase.claims if c.id == epi.activation_id)
    src_phase.claims = [c for c in src_phase.claims if c.id != epi.activation_id]
    # A fresh phase id past every existing one; it depends on the matmul's phase (RAW on the intermediate).
    new_pid = max(p.phase_id for p in m.phases) + 1
    m.add_phase(Phase(phase_id=new_pid, deps=(epi.phase_id,), claims=[act]))
    return m


def optimize_fused(
    module: Module, h: HProfile, theta: Theta, policy: Policy = PERF
) -> tuple[RealizationResult, list[FusionCertificate]]:
    """Score every detected matmul -> activation epilogue FUSED vs UNFUSED through the unchanged tropical
    optimizer and adopt the fusion iff it strictly wins -- the autonomous, priced epilogue-fusion decision.

    For each legal (sole-consumer, ELEMENTWISE) epilogue:
      * the FUSED plan is the module as given -- producer and consumer in one phase, so ``optimize`` /
        ``fused_candidates`` applies the ``_DEFOREST_FACTOR`` discount to the activation (the intermediate
        never round-trips: the inline epilogue);
      * the UNFUSED plan splits the activation past a phase BARRIER, which materializes the intermediate
        (the discount vanishes -- matmul writes the full buffer, the activation re-reads it).
    Fusion is adopted iff ``fused_score < unfused_score``; each adopted fusion emits a ``FusionCertificate``.

    Returns the (fused) ``RealizationResult`` and one certificate per ADOPTED fusion. A clean NO-OP (the
    base plan, empty certificates) when there is no epilogue, when the only epilogue is non-fusible (softmax)
    or illegal (the intermediate has another reader), or when fusion does not strictly lower the score.
    Mirrors ``bundle.optimize_bundled``'s shape exactly."""
    result = optimize(module, h, theta, policy)
    certs: list[FusionCertificate] = []
    for epi in find_matmul_activation_epilogues(module):
        if not epi.fusible:  # softmax: a row reduction, not an inline epilogue -> skip
            continue
        fused_score = result.score  # the module as given: producer+consumer same phase
        unfused = _split_phase_at_barrier(module, epi)
        unfused_score = optimize(unfused, h, theta, policy).score
        if fused_score < unfused_score:  # the priced win -- adopt the fusion
            certs.append(FusionCertificate(epi, unfused_score, fused_score, epi.act_kind))
    return result, certs
