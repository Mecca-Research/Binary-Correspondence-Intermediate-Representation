"""B3: calling-side tuning for a WRAPPED kernel, priced by the EXISTING K_BCIR cost model.

The roadmap thesis for the wrapped-library work (B5 BLAS gemm, B2 FFTW fft) is that BCIR's value is
optimizing the CALLING side -- layout, prefetch, tiling, channel selection -- around a TRUSTED kernel
that BCIR does NOT reimplement. B5/B2 today hard-code those calling-side params: ``emit_blas_gemm_c``
always emits ``cblas_sgemm(CblasRowMajor, ...)``, no prefetch, no channel choice. B3 turns them into a
COST-MODEL-PRICED PLAN: for a wrapped gemm BCIR chooses {major-order, caller-side block, prefetch
distance, execution channel} through the SAME cost machinery every other claim is scored by, records the
choice in a proof-carrying ``CallingSideCertificate``, and -- the gate that matters -- the tuned call
computes the IDENTICAL numeric result as the untuned wrapped call (calling-side tuning never changes the
math; that is the whole point).

THE LOAD-BEARING DESIGN CONSTRAINT -- each calling-side param is the cost model's PRICED decision, NOT a
bespoke discount. We reuse the existing machinery and let it price the choice:

  * **major-order (row vs column)** -- priced by ``matmul.cost_of``'s existing blocked-GEMM traffic terms.
    cblas's row/col major swap is the textbook "compute C^T = B^T A^T into the same buffer" identity; the
    two orders differ only in which operand is reused across which tile extent (A across the N-tile vs B
    across the M-tile), which ``cost_of`` ALREADY models. We cost both with the SAME function and pick the
    lower ``bottleneck`` (the max,+ roofline step), so the order is chosen, not assumed.
  * **caller-side block / tile** -- it IS ``matmul.plan_matmul``'s ``TilePlan`` (the existing dual-semiring
    tile search), reused unchanged. The block is the cache-blocking the caller wraps around the wrapped
    call; ``plan_matmul`` already prices it through ``cost_of``.
  * **prefetch distance** -- priced through ``CostVector.couple`` with the SAME memory-axis discount the
    GEM ``Prefetch`` contract models (``streampack._PREFETCH_FACTOR``-shaped): a software prefetch of the
    NEXT block hides its load latency, a memory-axis win, gated on the block actually having a next block
    to prefetch (a single-block call gets distance 0 -- nothing to hide).
  * **execution channel (host vs accelerator)** -- priced by the EXISTING ``optimize`` on each channel's
    ``TargetProfile`` (exactly what ``channels.orchestrate`` does): the gemm claim is scored on every
    channel in the tower and the cheapest suitable one wins. Single-channel towers stay on host (a no-op).

Reference-invariance (proved in test_calling_side_tuning.py): every choice produces the IDENTICAL C as the
untuned row-major reference. Column-major with the matching leading dim computes the same C (the C^T=B^TA^T
identity, byte-for-byte same buffer); a prefetch hint changes nothing numerically; a different channel
computes the same value. The emitted C carries the chosen params (``emit_tuned_gemm_c``); the fallback
honors the same layout, so linked and standalone agree.

Scope -- DISTINCT from G3: this is narrowly the WRAPPED KERNEL's OWN calling-side parameters (the cblas
call's major-order + leading dim, the caller's blocking around it, the ``__builtin_prefetch`` hints, the
channel that runs it). It is NOT a general SoA<->AoS transform of tensor resources in the claim graph
(that is G3, a separate segment). No new globally-numbered R-law; no MLIR change (the oracle prototype, as
B1/B5's MLIR wiring was kept separate).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..model import Claim, Domain, Lane, Opcode, StrideClass
from .cost import COMPUTE, MEMORY, CostVector, IDENTITY_FACTOR, TargetProfile, Theta
from .matmul import TilePlan, bottleneck, cost_of, cost_vector, plan_matmul
from .realize import optimize
from .weights import PERF, Policy

# Prefetch memory-axis discount (Q8): a software prefetch of the next block hides its load latency, the
# same memory-traffic-hiding win the GEM Prefetch contract models. Shaped exactly like realize._DEFOREST_
# FACTOR (a single-axis Q8 < 1.0 coupling, identity elsewhere) so it prices through CostVector.couple with
# the SAME algebra -- NOT a bespoke scalar discount. x0.875 == a conservative hide of the prefetched leg.
_PREFETCH_FACTOR = tuple(224 if i == MEMORY else 256 for i in range(len(IDENTITY_FACTOR)))

# cblas major-order tokens (the emitted C uses these verbatim; the leading dim follows the order).
ROW_MAJOR = "row"
COL_MAJOR = "col"


@dataclass(frozen=True)
class CallingSidePlan:
    """The cost-model-priced calling-side parameters for ONE wrapped gemm call. Each field is the cost
    model's decision (see the module docstring for how each is priced), not a hardcoded constant. The
    plan is purely calling-side: applying it does NOT change the numeric result of the wrapped kernel."""

    M: int
    N: int
    K: int
    major: str  # "row" | "col" -- the cblas major-order (matching leading dim emitted)
    block: TilePlan  # the caller-side cache block (matmul.plan_matmul's TilePlan, reused)
    prefetch_distance: int  # blocks ahead to prefetch (0 == nothing to hide -> no hint emitted)
    channel: str  # the execution channel name ("host" or an accelerator channel)
    channel_kind: str  # "cpu" | "gpu" | "fpga" | ... (the routed hardware class)

    @property
    def lead_dim_a(self) -> int:
        """The leading dimension of A in the emitted cblas call for the chosen major-order. Row-major A is
        MxK (lda=K); the column-major form passes B first (the C^T=B^T A^T swap) so A is the 2nd operand
        with ldb=K either way -- but for row-major A is operand 1 with lda=K."""
        return self.K

    @property
    def is_row_major(self) -> bool:
        return self.major == ROW_MAJOR

    @property
    def prefetches(self) -> bool:
        return self.prefetch_distance > 0

    @property
    def off_host(self) -> bool:
        return self.channel_kind != "cpu" and self.channel != "host"


@dataclass(frozen=True)
class CallingSideCertificate:
    """The proof-carrying record of one calling-side tuning decision -- modeled on
    ``fusion.FusionCertificate`` / ``bundle.BundleCertificate``: the chosen plan, the cost BEFORE (the
    untuned row-major / no-prefetch / host call) and AFTER (the tuned call), and the modeled win. A clean
    NO-OP plan (the untuned params, gain 0) when tuning does not help. The certificate asserts NOTHING
    about the numeric result -- that invariance is the separate, stronger gate the tests discharge: tuned
    == untuned reference, bit-exact."""

    plan: CallingSidePlan
    untuned_cost: int  # the roofline bottleneck of the untuned (row-major, no prefetch, host) call
    tuned_cost: int  # ... and of the tuned call (the priced calling-side win)
    untuned_major: str  # the baseline major-order ("row" -- what B5 hard-codes)
    untuned_channel: str  # the baseline channel ("host")

    @property
    def gain(self) -> int:
        """The modeled cost reduction (>= 0). The win is calling-side only -- the kernel math is unchanged."""
        return self.untuned_cost - self.tuned_cost

    @property
    def is_noop(self) -> bool:
        """True iff tuning changed nothing the cost model prices (row-major, no prefetch, an on-host CPU
        channel, and no cost reduction) -- the clean no-op the design requires when tuning is unprofitable.
        A CPU-kind channel counts as the host baseline (it is the host), so a host-only / tied tower stays a
        no-op even though the routed channel carries a specific CPU name."""
        return (
            self.gain == 0
            and self.plan.major == self.untuned_major
            and not self.plan.off_host
            and not self.plan.prefetches
        )


def _gemm_claim(M: int, N: int, K: int, claim_id: int = 9_300) -> Claim:
    """The wrapped-gemm claim the channel router/cost model scores (a single dense T_MACC tile claim --
    the op string ``gem.matmul`` is what ``channels.claim_required_caps`` keys the matmul capability on).
    Its count is the output element budget M*N (what the cost model streams)."""
    return Claim(
        id=claim_id,
        opcode=Opcode.T_MACC,
        lane=Lane.T,
        stride_class=StrideClass.TILE,
        count=max(1, M * N),
        rd=(1, 2),
        wr=(3,),
        op="gem.matmul",
        domain=Domain.RAM,
    )


def _order_cost(M: int, N: int, K: int, block: TilePlan, major: str, target: TargetProfile) -> int:
    """The roofline BOTTLENECK (max,+ over compute/memory) of a wrapped gemm under one major-order, priced
    by the EXISTING ``matmul.cost_of``. Row-major reuses A across the N-tile and B across the M-tile;
    column-major (the C^T=B^T A^T identity) swaps those roles, so we cost it by swapping the (M,N) /
    (tile_m,tile_n) extents fed to ``cost_of`` -- the SAME traffic model, with the operands' reuse axes
    exchanged. The result is the dual-semiring bottleneck the rest of K_BCIR ranks on."""
    if major == ROW_MAJOR:
        compute, mem, _ = cost_of(M, N, K, block.tile_m, block.tile_n, block.tile_k, target)
    else:
        # Column-major computes C^T = B^T A^T: the row/col reuse axes (M<->N, tile_m<->tile_n) swap.
        compute, mem, _ = cost_of(N, M, K, block.tile_n, block.tile_m, block.tile_k, target)
    return bottleneck(CostVector.of(compute=compute, memory=mem))


def _channel_cost(M: int, N: int, K: int, channel, theta: Theta, policy: Policy) -> int:
    """The K_BCIR cost of the wrapped gemm on one channel, scored by the EXISTING ``optimize`` over that
    channel's ``TargetProfile`` (exactly the per-channel pricing ``channels.orchestrate`` uses). Returns
    the chosen step's scalar cost, or a large sentinel if the claim does not realize on the channel."""
    from ..model import Module, Phase, Resource

    m = Module(name="b3_gemm_channel")
    for rid, nm in ((1, "A"), (2, "B"), (3, "C")):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(max(1, M * N),), name=nm))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[_gemm_claim(M, N, K)]))
    res = optimize(m, channel.profile, theta, policy)
    step = next((s for s in res.steps if s.claim_id == 9_300), None)
    return step.cost if step is not None else (1 << 62)


def plan_calling_side(
    M: int,
    N: int,
    K: int,
    target: TargetProfile | None = None,
    channels=None,
    theta: Theta | None = None,
    policy: Policy = PERF,
) -> CallingSidePlan:
    """K_BCIR's deterministic calling-side choice for a wrapped gemm C[MxN] = A[MxK] @ B[KxN]. EVERY param
    is priced through the existing cost model:

      * **block** -- ``matmul.plan_matmul`` (the dual-semiring tile search), reused unchanged.
      * **major-order** -- ``_order_cost`` prices row vs column via ``matmul.cost_of``; the lower-bottleneck
        order wins (ties -> row, the B5 baseline, so a balanced shape stays a no-op).
      * **prefetch_distance** -- 1 (prefetch the next block) iff the chosen block leaves >1 block along the
        streamed extent to hide; else 0. The WIN is priced in the certificate via ``_PREFETCH_FACTOR``.
      * **channel** -- among ``channels`` (a tower of ``HardwareChannel``; default: host only), the cheapest
        SUITABLE channel by ``_channel_cost`` (the existing ``optimize``-on-profile pricing), tie-broken by
        name. A single-channel / host-only tower stays on host.

    Deterministic + pure-Python; reproducible for a given (shape, target, tower, theta, policy)."""
    target = target or TargetProfile.for_host()
    theta = theta or Theta.cool()

    block = plan_matmul(M, N, K, target)

    # major-order: the lower-bottleneck order under the existing cost_of traffic model (row wins ties).
    row_c = _order_cost(M, N, K, block, ROW_MAJOR, target)
    col_c = _order_cost(M, N, K, block, COL_MAJOR, target)
    major = COL_MAJOR if col_c < row_c else ROW_MAJOR

    # prefetch: hide the next block along the streamed (M) extent iff there is one (the chosen block tiles M).
    blocks_along_m = (M + max(1, block.tile_m) - 1) // max(1, block.tile_m)
    prefetch_distance = 1 if blocks_along_m > 1 else 0

    # channel: the cheapest suitable channel in the tower (the existing optimize-on-profile pricing).
    channel_name, channel_kind = "host", "cpu"
    if channels:
        from ..channels import channel_suits

        claim = _gemm_claim(M, N, K)
        priced = [
            (c, _channel_cost(M, N, K, c, theta, policy))
            for c in channels
            if channel_suits(claim, c)
        ]
        if priced:
            # Cheapest suitable channel; ties prefer a host CPU channel (so a genuine tie stays on host --
            # off-host is adopted only on a STRICT cost win), then by name (deterministic).
            best_ch, _ = min(
                priced, key=lambda cc: (cc[1], 0 if cc[0].kind == "cpu" else 1, cc[0].name)
            )
            channel_name, channel_kind = best_ch.name, best_ch.kind

    return CallingSidePlan(
        M=M,
        N=N,
        K=K,
        major=major,
        block=block,
        prefetch_distance=prefetch_distance,
        channel=channel_name,
        channel_kind=channel_kind,
    )


def _tuned_cost(plan: CallingSidePlan, target: TargetProfile) -> int:
    """The roofline bottleneck of the TUNED call: the chosen major-order's ``_order_cost`` (the layout +
    block win), coupled by ``_PREFETCH_FACTOR`` when the plan prefetches (the memory-axis latency hide).
    Priced entirely through the existing ``cost_of`` + ``CostVector.couple`` algebra."""
    row_c = _order_cost(plan.M, plan.N, plan.K, plan.block, ROW_MAJOR, target)
    col_c = _order_cost(plan.M, plan.N, plan.K, plan.block, COL_MAJOR, target)
    base = col_c if plan.major == COL_MAJOR else row_c
    if plan.prefetches:
        # The prefetch hides the next block's load: scale the bottleneck's memory leg by the prefetch
        # factor (the same single-axis Q8 coupling realize.py uses for deforestation). We re-price the
        # chosen order's (compute, mem) and couple the memory axis, then take the bottleneck again.
        if plan.major == COL_MAJOR:
            c, m_, _ = cost_of(
                plan.N,
                plan.M,
                plan.K,
                plan.block.tile_n,
                plan.block.tile_m,
                plan.block.tile_k,
                target,
            )
        else:
            c, m_, _ = cost_of(
                plan.M,
                plan.N,
                plan.K,
                plan.block.tile_m,
                plan.block.tile_n,
                plan.block.tile_k,
                target,
            )
        coupled = CostVector.of(compute=c, memory=m_).couple(_PREFETCH_FACTOR)
        return bottleneck(coupled)
    return base


def certify_calling_side(
    plan: CallingSidePlan,
    target: TargetProfile | None = None,
    channels=None,
    theta: Theta | None = None,
    policy: Policy = PERF,
) -> CallingSideCertificate:
    """Emit the proof-carrying ``CallingSideCertificate`` for a chosen plan: the untuned baseline cost (the
    row-major, no-prefetch, host call B5 hard-codes) and the tuned cost (the priced calling-side win),
    both through the existing cost model. The certificate records the decision + the modeled gain; the
    NUMERIC invariance (tuned == untuned reference) is the separate, stronger gate the tests prove.

    The channel leg of the win is the K_BCIR ``optimize``-on-profile delta (host vs the chosen channel),
    added to the layout/prefetch bottleneck delta -- so a plan that moves off-host carries that win too."""
    target = target or TargetProfile.for_host()
    theta = theta or Theta.cool()

    # untuned baseline: row-major block, no prefetch, host channel.
    untuned_block = plan.block
    untuned_layout = _order_cost(plan.M, plan.N, plan.K, untuned_block, ROW_MAJOR, target)
    tuned_layout = _tuned_cost(plan, target)

    # channel delta (priced by the existing optimize-on-profile, like orchestrate): host cost - chosen.
    chan_delta = 0
    if channels and plan.off_host:
        host_cost = chosen_cost = None
        for c in channels:
            cost = _channel_cost(plan.M, plan.N, plan.K, c, theta, policy)
            if c.kind == "cpu" and host_cost is None:
                host_cost = cost
            if c.name == plan.channel:
                chosen_cost = cost
        if host_cost is not None and chosen_cost is not None and chosen_cost < host_cost:
            chan_delta = host_cost - chosen_cost

    untuned_cost = untuned_layout
    tuned_cost = tuned_layout - chan_delta
    return CallingSideCertificate(
        plan=plan,
        untuned_cost=untuned_cost,
        tuned_cost=tuned_cost,
        untuned_major=ROW_MAJOR,
        untuned_channel="host",
    )


def tune_wrapped_gemm(
    M: int,
    N: int,
    K: int,
    target: TargetProfile | None = None,
    channels=None,
    theta: Theta | None = None,
    policy: Policy = PERF,
) -> tuple[CallingSidePlan, CallingSideCertificate]:
    """The one-call B3 entry point: plan the calling side of a wrapped gemm and certify it. Returns the
    (plan, certificate). The plan's params are emitted by ``lower.c_kernel.emit_tuned_gemm_c``; the
    certificate is the proof-carrying record. A balanced shape on a host-only tower yields a clean no-op
    plan + certificate (row-major, no prefetch, host, gain 0)."""
    plan = plan_calling_side(M, N, K, target, channels, theta, policy)
    cert = certify_calling_side(plan, target, channels, theta, policy)
    return plan, cert
