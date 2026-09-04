#!/usr/bin/env python3
"""The numbers GEM+ has to beat, and the protocol for deciding whether it did.

    python tools/perf/gemplus_baseline.py --list          # the frozen baseline
    python tools/perf/gemplus_baseline.py --compare       # re-measure and grade
    python tools/perf/gemplus_baseline.py --compare --scale 2 --repeats 5
    python tools/perf/gemplus_baseline.py --json out.json # machine-readable verdicts

WHY A FROZEN BASELINE. The 2026-08-12 architecture and performance audit
(`docs/research/BCIR_TMSAO_ARCHITECTURE_AND_PERFORMANCE_REPORT.md`) measured this repository
on a real machine and wrote the numbers down. Without them, "GEM+ made the planner faster"
is an opinion. With them it is a comparison, and a comparison is the only thing that can be
wrong -- which is the point.

THE THREE VERDICTS, and why the third one exists.

  GAIN        the metric improved past the noise band. Report the magnitude AND the
              remaining headroom: `gap_to_bound` says how much of the theoretical win is
              still on the table, because a 5% gain on an operation with a proved 60%
              floor is a different result from a 5% gain that closes the gap.
  NO-CHANGE   the metric did not move. This is a FINDING, not a pass. The roadmap requires
              an investigation: either the slice does not touch this path (then the metric
              was mis-assigned), the win was cancelled by a cost elsewhere (then the model
              is incomplete), or the operation is already at its bound (then say so and
              retire the metric).
  REGRESSION  the metric got worse. Blocks the slice.

THE HEADROOM COLUMN IS THE TMSAO CONNECTION. Every row that has a known lower bound carries
it. `bound` is a *measured or proved* floor from the audit -- an exact-solver optimum, an
Omega(n) argument, a hardware roofline -- never a guess. The distance between the incumbent
and the maximum valid lower bound is exactly what TMSAO-2 calls the optimality gap, so this
table is the certificate's evidence in miniature: closing a gap here is closing it there.

WHAT THIS IS NOT. The audit host was WSL on a Ryzen 5 2600 with no PMU or RAPL, and the
report says plainly that its timings "are not silicon performance certificates". Neither are
these. A ratio measured here establishes an ALGORITHMIC trend -- a quadratic sweep becoming
linear is visible through a factor of noise; a 3% constant-factor change is not. Rows whose
`kind` is `ratio` or `count` are host-independent and are the ones that can be gated hard.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: Where the numbers came from. Cited on every report so a reader can check them.
SOURCE = "docs/research/BCIR_TMSAO_ARCHITECTURE_AND_PERFORMANCE_REPORT.md"
BASELINE_HOST = (
    "WSL2, AMD Ryzen 5 2600 (6c/12t, AVX2, 8 MiB L3), 7.7 GiB RAM, "
    "Python 3.10.12, GCC 11.4, Clang/MLIR 22.1.8, no PMU/RAPL"
)

#: How much a wall-clock row must move before it counts. The audit took medians of five
#: repetitions on a virtualized host with no frequency control; anything under this is not
#: distinguishable from the machine. Ratio and count rows have their own tighter bands.
WALL_CLOCK_NOISE = 0.15

#: What each kind of row has to move before the movement means anything. The `ratio` band is
#: wide because a ratio of two timings inherits both timings' variance; `exact` is tight
#: because a solver's optimum does not vary at all.
_DEFAULT_NOISE = {"wall": WALL_CLOCK_NOISE, "ratio": 0.25, "exact": 0.02}


class Metric:
    """One thing GEM+ is expected to move, with the floor it is moving toward.

    `lower_is_better` is explicit rather than inferred from the name: `gap_to_bound` on a
    throughput row and on a latency row point in opposite directions, and guessing which
    from a string is how a dashboard ends up celebrating a regression.
    """

    __slots__ = (
        "key",
        "group",
        "what",
        "baseline",
        "unit",
        "kind",
        "bound",
        "bound_source",
        "lower_is_better",
        "slice_owner",
        "noise",
    )

    def __init__(
        self,
        key,
        group,
        what,
        baseline,
        unit,
        kind,
        *,
        bound=None,
        bound_source="",
        lower_is_better=True,
        slice_owner="",
        noise=None,
    ):
        self.key = key
        self.group = group
        self.what = what
        self.baseline = baseline
        self.unit = unit
        # "wall"  absolute time -- host-dependent, INDICATIVE off the baseline host.
        # "ratio" a ratio of two TIMED quantities -- cancels the machine's speed but not its
        #         run-to-run variance, so it is host-portable with a wide band. Measured at
        #         68.95 and then 76.29 on the same host minutes apart, which is where this
        #         band comes from rather than from a guess.
        # "exact" a counted or solved quantity -- an exact solver's optimum, a fraction over
        #         a fixed corpus, a byte extent. Deterministic, zero variance, gated hard.
        self.kind = kind
        self.bound = bound
        self.bound_source = bound_source
        self.lower_is_better = lower_is_better
        self.slice_owner = slice_owner  # the GEM+ slice that must move this
        self.noise = noise if noise is not None else _DEFAULT_NOISE[kind]

    def headroom(self, value: float) -> float | None:
        """How much of the theoretical win is still unclaimed, as a fraction.

        0.0 means the incumbent sits on its proved floor and this metric is finished --
        which is a legitimate and useful outcome to be able to state. `None` means no bound
        is known yet, and that is itself a roadmap item: an optimality claim cannot be made
        for a row whose floor nobody has computed.
        """
        if self.bound is None or value is None:
            return None
        if self.lower_is_better:
            if value <= self.bound:
                return 0.0
            return (value - self.bound) / value
        if value >= self.bound:
            return 0.0
        return (self.bound - value) / self.bound

    def verdict(self, value: float | None, *, same_host: bool = False) -> str:
        """Grade `value`, refusing to grade a wall-clock row measured on another machine.

        The first run of this harness measured `optimize_scheduled.512` at 1,974 ms against
        a 1,704 ms baseline and called it a 15.9% REGRESSION. It was not one: the baseline
        host is a Ryzen 5 2600 on WSL with Python 3.10, and the measuring host was neither.
        The same run measured the SLOWDOWN RATIO at 68.95 against a baseline of 69.2 --
        0.4% apart, across two unrelated machines.

        That pair is the whole argument for how this table is gated. A ratio between two
        operations timed in the same process cancels the machine out; an absolute
        millisecond does not, and comparing one across hosts manufactures verdicts in both
        directions. So a wall row off the baseline host is reported INDICATIVE and never
        blocks a slice, while ratio and count rows are graded everywhere.
        """
        if value is None:
            return "NOT-MEASURED"
        if self.kind == "wall" and not same_host:
            return "INDICATIVE"
        change = self.improvement(value)
        if change > self.noise:
            return "GAIN"
        if change < -self.noise:
            return "REGRESSION"
        return "NO-CHANGE"

    def improvement(self, value: float) -> float:
        """Signed fractional improvement over the baseline, positive = better."""
        base = self.baseline
        if base == 0:
            return 0.0
        return (base - value) / abs(base) if self.lower_is_better else (value - base) / abs(base)


# --- the frozen baseline -------------------------------------------------------------------
#
# Every number below is quoted from the report named in SOURCE. Where the report gives a
# proved optimum from an exact solver, it becomes `bound` -- those are the rows where a
# TMSAO-2 gap statement is available today.

METRICS: tuple[Metric, ...] = (
    # --- §6.3: the quadratic sweep. The clearest algorithmic target in the whole report.
    Metric(
        "optimize_scheduled.512",
        "planner",
        "optimize_scheduled at 512 claims",
        1703.66,
        "ms",
        "wall",
        bound=24.61,
        bound_source="serial optimize at 512 claims (§6.3) -- the sweep "
        "should approach O(claims x candidates), i.e. the "
        "serial cost times a small constant",
        slice_owner="G2",
    ),
    Metric(
        "optimize_scheduled.256",
        "planner",
        "optimize_scheduled at 256 claims",
        435.73,
        "ms",
        "wall",
        bound=12.19,
        bound_source="serial optimize at 256 claims (§6.3)",
        slice_owner="G2",
    ),
    Metric(
        "optimize_scheduled.slowdown.512",
        "planner",
        "optimize_scheduled / serial optimize at 512",
        69.2,
        "x",
        "ratio",
        bound=4.0,
        bound_source="a linear sweep over a constant candidate set should "
        "cost a small multiple of the serial pass, not a "
        "growing one (§6.3)",
        slice_owner="G2",
    ),
    Metric(
        "optimize_scheduled.quality",
        "planner",
        "one-sweep makespan / exact makespan, 4-claim 3-candidate fixture",
        1.00696,
        "x",
        "exact",
        bound=1.0,
        bound_source="exhaustive enumeration of the same fixture (§6.3)",
        slice_owner="G4",
    ),
    # --- §6.1: HEFT-lite against an exact branch-and-bound scheduler.
    Metric(
        "eft.suboptimal.2domains",
        "scheduler",
        "fraction of 1,716 six-job instances where EFT is suboptimal, 2 domains",
        0.1107,
        "fraction",
        "exact",
        bound=0.0,
        bound_source="exact branch-and-bound over the same corpus (§6.1)",
        slice_owner="G4",
    ),
    Metric(
        "eft.worst.2domains",
        "scheduler",
        "worst EFT/optimal makespan ratio, 2 domains",
        1.1333,
        "x",
        "exact",
        bound=1.0,
        bound_source="exact branch-and-bound (§6.1)",
        slice_owner="G4",
    ),
    Metric(
        "eft.mean.2domains",
        "scheduler",
        "mean EFT/optimal makespan ratio, 2 domains",
        1.0078,
        "x",
        "exact",
        bound=1.0,
        bound_source="exact branch-and-bound (§6.1)",
        slice_owner="G4",
    ),
    # --- §6.2: two implementations that do not describe the same schedule. This one is a
    # CORRECTNESS metric wearing a performance costume: the target is agreement, not speed.
    Metric(
        "pricing.eft.divergence",
        "scheduler",
        "price_scheduled makespan / schedule_eft makespan on the §6.2 fixture",
        51200 / 25700,
        "x",
        "exact",
        bound=1.0,
        bound_source="one canonical schedule artifact read by both (§6.2) -- "
        "any value but 1.0 means the objective and the executor "
        "disagree about what the plan is",
        slice_owner="G1",
    ),
    # --- §6.4: first-fit against exact backtracking.
    Metric(
        "memory.suboptimal.fraction",
        "memory",
        "fraction of 500 seven-resource fixtures where first-fit is suboptimal",
        0.386,
        "fraction",
        "exact",
        bound=0.0,
        bound_source="exact integer backtracking (§6.4)",
        slice_owner="G5",
    ),
    Metric(
        "memory.worst.ratio",
        "memory",
        "worst first-fit extent / proved optimum",
        21 / 13,
        "x",
        "exact",
        bound=1.0,
        bound_source="exact integer backtracking, 21 vs 13 units (§6.4)",
        slice_owner="G5",
    ),
    Metric(
        "memory.real.bytes",
        "memory",
        "real planner extent on the §6.4 fixture at 64-byte alignment",
        1344,
        "bytes",
        "exact",
        bound=832,
        bound_source="exact layout on the same fixture (§6.4)",
        slice_owner="G5",
    ),
    # --- §5.2: the digest recomputation the profile found. Three hashes of one immutable
    # module is pure overhead, and it is the largest single line in the profile.
    Metric(
        "static_memory.plan.2048",
        "memory",
        "plan_static_memory (includes verify) at 2,048 resources",
        301.02,
        "ms",
        "wall",
        bound=88.05,
        bound_source="the module digest alone at the same size (§5.2) -- "
        "one canonical digest computed once is the floor the "
        "planner cannot go below while it still hashes",
        slice_owner="G0",
    ),
    Metric(
        "static_memory.digest.2048",
        "memory",
        "module digest at 2,048 resources",
        88.05,
        "ms",
        "wall",
        slice_owner="G0",
    ),
    Metric(
        "static_memory.verify.2048",
        "memory",
        "external verify at 2,048 resources",
        157.88,
        "ms",
        "wall",
        bound=88.05,
        bound_source="an identity-bound API lets an independent verifier "
        "reuse a proved digest instead of recomputing (§5.2)",
        slice_owner="G0",
    ),
    # --- §5.1: the deterministic audit. These are the end-to-end rows; they move only when
    # a slice changes something real, which makes them the honest integration signal.
    Metric(
        "audit.kbcir-streampack.scale4",
        "audit",
        "K_BCIR->StreamPack at scale 4 (4,096 claims)",
        275.22,
        "ms",
        "wall",
        slice_owner="G2",
    ),
    Metric(
        "audit.static-lifetime-planner.scale4",
        "audit",
        "static lifetime planner at scale 4 (2,048 resources)",
        566.35,
        "ms",
        "wall",
        slice_owner="G0",
    ),
    Metric(
        "audit.mixed-wave-token-eft.scale4",
        "audit",
        "mixed wave/token/EFT at scale 4 (2,048 claims)",
        48.05,
        "ms",
        "wall",
        slice_owner="G1",
    ),
    Metric(
        "audit.iterative-phase-dag.scale4",
        "audit",
        "iterative phase DAG at scale 4 (2,048 claims)",
        34.62,
        "ms",
        "wall",
        slice_owner="G3",
    ),
    # --- §4.4: the native structural wins. These are what BCIR is FOR, and the roadmap must
    # not regress them while making the planner faster.
    Metric(
        "native.gather-avoidance",
        "native",
        "blocked realization vs gather form",
        5.58,
        "x",
        "ratio",
        lower_is_better=False,
        bound=6.05,
        bound_source="the upper end of the observed 5.58-6.05x band (§4.4)",
        slice_owner="G6",
    ),
    Metric(
        "native.blocked-reduction",
        "native",
        "blocked reduction vs naive",
        11.68,
        "x",
        "ratio",
        lower_is_better=False,
        bound=11.72,
        bound_source="the upper end of the observed band (§4.4)",
        slice_owner="G6",
    ),
    Metric(
        "native.direct-stride",
        "native",
        "direct strided access vs gather form",
        1.27,
        "x",
        "ratio",
        lower_is_better=False,
        bound=1.33,
        bound_source="the upper end of the observed band (§4.4)",
        slice_owner="G6",
    ),
    Metric(
        "native.dense-parity",
        "native",
        "dense streaming BCIR vs equivalent compiler loop",
        0.98,
        "x",
        "ratio",
        lower_is_better=False,
        bound=1.01,
        bound_source="the observed 0.98-1.01x parity band (§4.4). Parity is "
        "the CORRECT result here -- when LLVM sees the same "
        "affine loop and alias facts, matching it is the floor, "
        "and a claimed win above this band needs a structural "
        "reason before it is believed",
        slice_owner="G6",
    ),
    # --- S0-A (2026-09-04): rows a slice added. These two are NOT quoted from the report: the
    # report's K_BCIR->StreamPack case verified its plan with no scope at all
    # (`verify_plan(module, result)`), so the numbers did not exist to quote. Each carries the
    # value the same code path measured on the tree BEFORE the slice (#758), frozen the same
    # way -- a slice may add a row only together with its own pre-slice measurement.
    Metric(
        "verify.plan.r9.vacuous",
        "verifier",
        "fraction of the audit fixture's steps whose FORGED cost R9 accepts",
        1.0,
        "fraction",
        "exact",
        bound=0.0,
        bound_source="R9 and the planner price a step through ONE predicate "
        "(`realize.step_cost`), so a forged cost on any step of "
        "the planner's own plan is a diagnostic and no step is "
        "left to accept on trust (laws.md L2, L14)",
        slice_owner="S0-A",
    ),
    Metric(
        "verify.plan.scope.overhead",
        "verifier",
        "scope-aware verify_plan / optimize, same fixture (4,096 claims)",
        1.07,
        "x",
        "ratio",
        slice_owner="G17",
    ),
)

_BY_KEY = {metric.key: metric for metric in METRICS}


# --- measuring the current tree -------------------------------------------------------------


def measure_audit(scale: int, repeats: int) -> dict[str, float]:
    """Re-run the deterministic audit and return the `audit.*` rows in milliseconds."""
    from bcir.performance_audit import run_tmsao_audit

    report = run_tmsao_audit(scale=scale, repeats=repeats)
    out: dict[str, float] = {}
    for sample in report.samples:
        out[f"audit.{sample.name}.scale{scale}"] = sample.median_ns / 1e6
    return out


def measure_planner() -> dict[str, float]:
    """The §6.3 sweep, at the two sizes the report tabulates."""
    import time

    from bcir.kbcir import TARGETS, optimize
    from bcir.kbcir.cost import Theta
    from bcir.kbcir.weights import PERF
    from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass

    from bcir.gem.overlap import optimize_scheduled

    host = TARGETS[sorted(TARGETS)[0]]
    out: dict[str, float] = {}
    for count in (256, 512):
        module = Module(name=f"sweep{count}")
        module.add_resource(Resource(rid=1, shape=(64,)))
        module.add_phase(
            Phase(
                phase_id=0,
                claims=[
                    Claim(
                        id=index + 1,
                        opcode=Opcode.ADD,
                        lane=Lane.U,
                        stride_class=StrideClass.UNIT,
                        count=64,
                        rd=(1,),
                        wr=(1,),
                        op="vector.add",
                    )
                    for index in range(count)
                ],
            )
        )
        for label, fn in (("optimize_scheduled", optimize_scheduled), ("serial", optimize)):
            start = time.perf_counter()
            fn(module, host, Theta.cool(), PERF)
            elapsed = (time.perf_counter() - start) * 1e3
            if label == "optimize_scheduled":
                out[f"optimize_scheduled.{count}"] = elapsed
            else:
                out[f"_serial.{count}"] = elapsed
    if "optimize_scheduled.512" in out and out.get("_serial.512"):
        out["optimize_scheduled.slowdown.512"] = out["optimize_scheduled.512"] / out["_serial.512"]
    return {k: v for k, v in out.items() if not k.startswith("_")}


def measure_exact() -> dict[str, float]:
    """The deterministic rows: solver optima and fixed-corpus fractions.

    These are the rows that gate. They involve no timing, so they are identical on every
    host and in every run -- if one of them moves, the compiler's DECISIONS changed, which
    is exactly the signal a roadmap slice is supposed to produce.
    """
    out: dict[str, float] = {}

    # §6.2: price_scheduled against schedule_eft. The report measured 51,200 against
    # 25,700 on four independent claims across TWO domains, and this fixture reproduces that
    # ratio to every digit the report prints -- 1.9922178988 against 51200/25700 = 1.9922 --
    # while its absolute costs are 4x the report's. That is the `ratio` vs `wall`
    # classification demonstrating itself: the structure carries across scales and machines,
    # the milliseconds do not.
    #
    # Three details are load-bearing, each found by getting it wrong first:
    #   * the claims must touch DISTINCT resources, or they conflict and both schedulers
    #     serialize them, giving a ratio of 1.0 and hiding the defect;
    #   * the target must expose exactly TWO domains, because the divergence comes from
    #     round-robin binning putting both large claims in one bin while EFT splits them --
    #     with eight domains both schedulers find the same answer;
    #   * the costs must come from the real cost model, because `price_scheduled` recomputes
    #     from the module and ignores costs supplied on the steps.
    try:
        from dataclasses import replace

        from bcir.gem.overlap import price_scheduled
        from bcir.gem.schedule import schedule_eft
        from bcir.kbcir import TARGETS, optimize
        from bcir.kbcir.cost import Theta
        from bcir.kbcir.weights import PERF
        from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass

        host = replace(TARGETS["x86_avx2"], affinity_domains=2)
        module = Module(name="divergence")
        for rid in range(1, 9):
            module.add_resource(Resource(rid=rid, shape=(65536,)))
        module.add_phase(
            Phase(
                phase_id=0,
                claims=[
                    Claim(
                        id=index + 1,
                        opcode=Opcode.ADD,
                        lane=Lane.U,
                        stride_class=StrideClass.UNIT,
                        count=count,
                        rd=(2 * index + 1,),
                        wr=(2 * index + 2,),
                        op="vector.add",
                    )
                    for index, count in enumerate((16384, 64, 16384, 64))
                ],
            )
        )
        result = optimize(module, host, Theta.cool(), PERF)
        durations = {step.claim_id: step.cost for step in result.steps}
        priced = price_scheduled(module, result, host, Theta.cool(), PERF)
        eft = schedule_eft(module, durations, host)
        if eft.makespan:
            out["pricing.eft.divergence"] = priced.makespan / eft.makespan
    except Exception as exc:  # pragma: no cover - shape probe
        sys.stderr.write(f"[baseline] pricing/EFT divergence unavailable: {exc}\n")

    return out


def measure_verifier() -> dict[str, float]:
    """The S0-A rows: can R9 fire on the planner's own plan, and what does firing cost.

    Both come from ONE fixture -- the audit's K_BCIR->StreamPack module at scale 4 (matmul
    128x128, tile 8: 4,096 claims), the module `audit.kbcir-streampack.scale4` times -- so
    the ratio row explains that row's movement instead of restating it.

    The vacuity row does not ask the verifier to agree with the planner: a shared predicate
    agrees with itself by construction, and a row that can only read 0.0 measures nothing.
    It forges EVERY step's cost by one unit and counts the steps the verifier still accepts.
    Before S0-A `verify_plan` took no scope, so that was every step (1.0); anything above
    0.0 now means R9 has stopped re-deriving some step -- the vacuity L2 forbids, and the
    one regression this row exists to catch.

    The overhead row is the price of the fix, as a same-process ratio so it carries across
    hosts: scope-aware verification currently re-derives the planner's whole offer
    (`fused_candidates`) to price each chosen step. G17's single-candidate re-derivation is
    the slice that must move it; no floor is recorded because the 3-5x it projects is a
    projection, not a proved bound.
    """
    import statistics
    import time
    from dataclasses import replace

    from bcir.examples import matmul_tiled
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.realize import optimize
    from bcir.kbcir.weights import PERF
    from bcir.verify import verify_plan

    out: dict[str, float] = {}
    module = matmul_tiled(n=128, tile=8)
    host, theta = TargetProfile.x86_avx2(), Theta.mem_bound()

    def median_ms(fn, repeats=3):
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1e3)
        return statistics.median(samples)

    result = optimize(module, host, theta, PERF)
    plan_ms = median_ms(lambda: optimize(module, host, theta, PERF))
    verify_ms = median_ms(lambda: verify_plan(module, result, host, theta=theta, policy=PERF))
    if plan_ms:
        out["verify.plan.scope.overhead"] = verify_ms / plan_ms

    forged = replace(
        result,
        score=result.score + len(result.steps),
        steps=tuple(replace(s, cost=s.cost + 1) for s in result.steps),
    )
    flagged = sum(
        1
        for d in verify_plan(module, forged, host, theta=theta, policy=PERF)
        if d.law == "R9" and "does not re-derive" in d.message
    )
    if result.steps:
        out["verify.plan.r9.vacuous"] = 1.0 - flagged / len(result.steps)
    return out


_MEASURERS = {
    "audit": measure_audit,
    "planner": measure_planner,
    "exact": measure_exact,
    "verifier": measure_verifier,
}


def measure(scale: int, repeats: int, groups: set[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, fn in _MEASURERS.items():
        if groups and name not in groups:
            continue
        try:
            out.update(fn(scale, repeats) if name == "audit" else fn())
        except Exception as exc:  # pragma: no cover - host variance
            sys.stderr.write(f"[baseline] {name} measurement unavailable: {exc}\n")
    return out


# --- reporting -------------------------------------------------------------------------------


def _fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    if unit in ("ms",):
        return f"{value:,.2f}"
    if unit in ("bytes", "count"):
        return f"{value:,.0f}"
    return f"{value:.4f}"


def on_baseline_host() -> bool:
    """Whether wall-clock rows may be graded rather than merely reported.

    Deliberately strict, and deliberately not inferred from a CPU model string alone: the
    baseline was taken under WSL with a particular Python and no frequency control, and any
    one of those differing is enough to make a millisecond comparison meaningless. Set
    BCIR_BASELINE_HOST=1 to assert you are reproducing the report's environment.
    """
    return os.environ.get("BCIR_BASELINE_HOST") == "1"


def compare(measured: dict[str, float], *, same_host: bool | None = None) -> list[dict]:
    if same_host is None:
        same_host = on_baseline_host()
    rows = []
    for metric in METRICS:
        value = measured.get(metric.key)
        headroom = metric.headroom(value if value is not None else metric.baseline)
        rows.append(
            {
                "key": metric.key,
                "group": metric.group,
                "slice": metric.slice_owner,
                "what": metric.what,
                "unit": metric.unit,
                "kind": metric.kind,
                "baseline": metric.baseline,
                "measured": value,
                "verdict": metric.verdict(value, same_host=same_host),
                "improvement": metric.improvement(value) if value is not None else None,
                "bound": metric.bound,
                "bound_source": metric.bound_source,
                "headroom": headroom,
            }
        )
    return rows


def render(rows: list[dict]) -> str:
    lines = [
        f"BCIR GEM+ baseline comparison  (source: {SOURCE})",
        f"baseline host: {BASELINE_HOST}",
        f"this host:     {platform.platform()}, Python {platform.python_version()}",
        "",
        f"{'metric':<40} {'slice':<6} {'baseline':>12} {'now':>12} {'verdict':<12} {'headroom':>9}",
        "-" * 96,
    ]
    for row in rows:
        head = "-" if row["headroom"] is None else f"{row['headroom'] * 100:.1f}%"
        lines.append(
            f"{row['key']:<40} {row['slice']:<6} "
            f"{_fmt(row['baseline'], row['unit']):>12} "
            f"{_fmt(row['measured'], row['unit']):>12} "
            f"{row['verdict']:<12} {head:>9}"
        )

    graded = [r for r in rows if r["verdict"] not in ("NOT-MEASURED", "INDICATIVE")]
    indicative = [r for r in rows if r["verdict"] == "INDICATIVE"]
    gains = [r for r in graded if r["verdict"] == "GAIN"]
    flat = [r for r in graded if r["verdict"] == "NO-CHANGE"]
    bad = [r for r in graded if r["verdict"] == "REGRESSION"]
    lines += [
        "",
        f"{len(gains)} gain, {len(flat)} no-change, {len(bad)} regression, "
        f"{len(indicative)} indicative, "
        f"{len(rows) - len(graded) - len(indicative)} not measured",
    ]
    if indicative:
        lines += [
            "",
            "INDICATIVE rows are wall-clock measured off the baseline host, so they "
            "are reported and not graded.",
            "Re-run with BCIR_BASELINE_HOST=1 on the report's environment to grade "
            "them; the ratio rows above are host-portable and gate everywhere.",
        ]
    if flat:
        lines += [
            "",
            "NO-CHANGE rows are findings, not passes. For each, the roadmap requires one of:",
            "  (a) the slice does not touch this path -- the metric was mis-assigned;",
            "  (b) the win was cancelled elsewhere -- the cost model is incomplete;",
            "  (c) the operation is already at its bound -- say so and retire the row.",
        ]
        for row in flat:
            lines.append(f"  - {row['key']} (slice {row['slice'] or '?'})")
    if bad:
        lines += ["", "REGRESSIONS block the slice:"]
        for row in bad:
            lines.append(f"  - {row['key']}: {row['improvement'] * 100:+.1f}%")
    return "\n".join(lines)


def render_list() -> str:
    lines = [f"{len(METRICS)} frozen metrics from {SOURCE}", ""]
    for group in sorted({m.group for m in METRICS}):
        lines.append(f"[{group}]")
        for metric in [m for m in METRICS if m.group == group]:
            floor = (
                "no bound computed yet"
                if metric.bound is None
                else f"bound {_fmt(metric.bound, metric.unit)} {metric.unit}"
            )
            lines.append(
                f"  {metric.key:<40} {_fmt(metric.baseline, metric.unit):>12} "
                f"{metric.unit:<8} slice={metric.slice_owner or '?':<4} {floor}"
            )
            if metric.bound_source:
                lines.append(f"      floor: {metric.bound_source}")
        lines.append("")
    unbounded = [m.key for m in METRICS if m.bound is None]
    if unbounded:
        lines += [
            "Rows with no lower bound yet -- no optimality claim is available for "
            "these until one is computed:"
        ]
        lines += [f"  - {key}" for key in unbounded]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the frozen baseline")
    parser.add_argument("--compare", action="store_true", help="re-measure and grade")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="limit measurement to a group (audit, planner, exact, verifier)",
    )
    parser.add_argument("--json", help="write the verdicts to a JSON file")
    args = parser.parse_args(argv)

    if args.list or not args.compare:
        print(render_list())
        return 0

    measured = measure(args.scale, args.repeats, set(args.group) or None)
    rows = compare(measured)
    print(render(rows))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "source": SOURCE,
                    "baseline_host": BASELINE_HOST,
                    "host": platform.platform(),
                    "scale": args.scale,
                    "repeats": args.repeats,
                    "rows": rows,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
    return 1 if any(r["verdict"] == "REGRESSION" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
