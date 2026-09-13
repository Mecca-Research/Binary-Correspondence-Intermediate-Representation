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
    # --- G2, the general case (S2-A): many step-shortening trials. The report's fixture has
    # none (the serial optimum is already the shortest step everywhere), so the sweep's cost
    # there is its fixed overhead; `bcir/tests/sweep_fixtures.general_fixture(16, 16)` has one
    # trial per pair of claims (512 claims, 256 trials) under ENERGY on x86_avx2. The baselines
    # are the parent tree (S1-D, 2026-09-12) on the harness host, every trial re-placing the
    # whole module; they are not the report's, which never measured this case.
    Metric(
        "optimize_scheduled.general.512",
        "planner",
        "optimize_scheduled at 512 claims, 256 step-shortening trials (16 phases)",
        605.5,
        "ms",
        "wall",
        bound=109.0,
        bound_source="8x the serial pass on the same fixture and host (13.6 ms): the G2 "
        "target ratio applied to the general case",
        slice_owner="G2",
    ),
    Metric(
        "optimize_scheduled.general.slowdown.512",
        "planner",
        "optimize_scheduled / serial optimize, the general case at 512 claims",
        44.5,
        "x",
        "ratio",
        bound=8.0,
        bound_source="the G2 target: a sweep of step-shortening trials priced at a small "
        "multiple of the serial pass (roadmap G2)",
        slice_owner="G2",
    ),
    Metric(
        "sweep.replacement.fraction",
        "planner",
        "claims re-placed per trial / claims, the general case at 512 claims",
        1.0,
        "x",
        "exact",
        bound=0.0625,
        bound_source="the affected phase and nothing else: 1/16 of the module on the "
        "16-phase fixture (a checkpointed replay can re-place less)",
        slice_owner="G2",
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
    # The report's three-domain row (§6.1), frozen at its numbers; and the exact rail (G4 /
    # S2-B): the corpus proved instance by instance, so `solver.unproved.fraction` is the share
    # the bounded search could not close within its budget and `solver.gap.p95` the 95th
    # percentile of the certified gap `(U - L) / U` over the corpus -- the Stage 2 exit rows.
    # Baselines are the parent tree: no exact rail, so nothing was proved (1.0) and the only
    # gap anyone could state was the heuristic's own against the section 6.1 oracle.
    Metric(
        "eft.suboptimal.3domains",
        "scheduler",
        "fraction of 1,716 six-job instances where EFT is suboptimal, 3 domains",
        18 / 1716,
        "fraction",
        "exact",
        bound=0.0,
        bound_source="exact branch-and-bound over the same corpus (§6.1)",
        slice_owner="G4",
    ),
    Metric(
        "eft.worst.3domains",
        "scheduler",
        "worst EFT/optimal makespan ratio, 3 domains",
        7 / 6,
        "x",
        "exact",
        bound=1.0,
        bound_source="exact branch-and-bound (§6.1)",
        slice_owner="G4",
    ),
    Metric(
        "solver.unproved.fraction",
        "scheduler",
        "six-job instances (2 and 3 domains) the exact rail did not prove within its budget",
        1.0,
        "fraction",
        "exact",
        bound=0.0,
        bound_source="every instance of the corpus closed (stop reason optimal); the "
        "parent tree had no exact rail, so nothing was proved",
        slice_owner="G4",
    ),
    Metric(
        "solver.gap.p95",
        "scheduler",
        "95th percentile of the certified relative gap (U - L) / U over the six-job corpus",
        0.0625,
        "fraction",
        "exact",
        bound=0.0,
        bound_source="the incumbent proved optimal on every instance; the baseline is the "
        "heuristic's own p95 gap against the §6.1 oracle over the pooled 2- and 3-domain "
        "corpus on the parent tree (1/16: two domains, 190 of 1,716 instances off)",
        slice_owner="G4",
    ),
    # --- G12 (S2-C): the dispatch law and resumable search. Counted over a fixed corpus of
    # interruption points (solver x fixture x budget split); the baselines are the parent
    # tree, where no solver had a resume parameter (every point unavailable), the exact layout
    # refused a zero budget (no incumbent at that point) and no certificate recorded a dispatch.
    Metric(
        "search.resume.unavailable",
        "dispatch",
        "interruption points whose resumed run is unavailable or differs from the uninterrupted run",
        393,
        "count",
        "exact",
        bound=0.0,
        bound_source="every proof-rail solver returns a content-addressed state whose continuation "
        "equals the uninterrupted run (G12); the parent tree had none",
        slice_owner="G12",
    ),
    Metric(
        "dispatch.incumbent.missing",
        "dispatch",
        "interruption points (budget 0 included) without a legal incumbent, every solver",
        26,
        "count",
        "exact",
        bound=0.0,
        bound_source="an incumbent first: the fast rail's answer stands at every interruption point; "
        "the parent tree's exact layout refused a zero budget on each memory fixture",
        slice_owner="G12",
    ),
    Metric(
        "dispatch.unrecorded",
        "dispatch",
        "corpus certificates without a dispatch record (rail, solver, units, budget, stop reason, bound source)",
        12,
        "count",
        "exact",
        bound=0.0,
        bound_source="every certificate names the rail that ran (G12); the parent tree's twelve carried none",
        slice_owner="G12",
    ),
    # --- G6 (S2-D): typed regions and the objective registry. Counted over a fixed corpus
    # (the 12 programs, the 2x3 general fixture, 20 generated and 20 random modules: 542
    # claims); the baselines are the parent tree, where no claim was covered by a region with
    # a verified expansion and the two objective names the law rail shares existed without
    # verified laws.
    Metric(
        "regions.unexpanded.claims",
        "regions",
        "corpus claims not covered by a verified region whose expansion is the module claim for claim",
        542,
        "count",
        "exact",
        bound=0.0,
        bound_source="every claim sits in exactly one recognized, verified region and the region graph "
        "expands to the module (G6); the parent tree had no region layer",
        slice_owner="G6",
    ),
    Metric(
        "objectives.unverified",
        "regions",
        "objective registry entries admitted without their laws proved (closure, identities, order)",
        2,
        "count",
        "exact",
        bound=0.0,
        bound_source="the registry admits an objective only when verify_objective passes (G6); the "
        "parent tree named min_plus and max_plus with no laws attached",
        slice_owner="G6",
    ),
    # --- §6.2: two implementations that do not describe the same schedule. This one is a
    # CORRECTNESS metric wearing a performance costume: the target is agreement, not speed.
    # G1 (S1-A) landed the one artifact: `price_scheduled` reads `schedule_plan`, which is the
    # placement `schedule_eft` returns, so the row reads exactly 1.0; the retired wave pricer
    # is kept as `price_waves_legacy` and `measure_legacy_divergence` still reproduces the
    # report's 1.9922 on this fixture (the witness that the fixture exhibits what it tracks).
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
    # --- §6.4: first-fit against exact backtracking. Measured (G5 / S1-D) over the corpus in
    # `bcir/tests/memory_fixtures.py`: 500 deterministic seven-resource fixtures of the
    # report's shape (the report's own corpus is not in the tree; first-fit is suboptimal on
    # 40.4% of this one against the report's 38.6%, worst ratio 1.6x) and `WORST_FIXTURE`,
    # which reproduces the report's worst case exactly (21 against 13 units; 1,344 against
    # 832 bytes at 64-byte alignment through the real planner). The rows measure the ENGAGED
    # rail: the bounded exact solver behind first-fit, every fixture solved to a proved
    # optimum within its work-unit budget.
    Metric(
        "memory.suboptimal.fraction",
        "memory",
        "fraction of the 500 seven-resource fixtures where the engaged layout is above the "
        "proved optimum",
        0.386,
        "fraction",
        "exact",
        bound=0.0,
        bound_source="exact integer backtracking (§6.4); the bounded exact solver "
        "(`static_memory.exact_layout`) proves every fixture of the corpus",
        slice_owner="G5",
    ),
    Metric(
        "memory.worst.ratio",
        "memory",
        "worst engaged extent / proved optimum over the corpus",
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
        "real planner extent on the §6.4 worst-case fixture at 64-byte alignment",
        1344,
        "bytes",
        "exact",
        bound=832,
        bound_source="exact layout on the same fixture (§6.4): 13 lines of 64 bytes",
        slice_owner="G5",
    ),
    # --- §5.2: the digest recomputation the profile found. Three hashes of one immutable
    # module is pure overhead, and it is the largest single line in the profile. G3 (S1-B)
    # owns these: the module identity is computed once per revision from an iterative
    # canonical stream and shared through `provenance.module_identity` / `digest_of`, and
    # the count row is the exact gate -- the verifier keeps its right to recompute at a
    # trust boundary, so the count is one, never zero.
    Metric(
        "static_memory.digests.2048",
        "memory",
        "full module digests over plan_static_memory + one identity-bound external verify",
        3.0,
        "count",
        "exact",
        bound=1.0,
        bound_source="one canonical digest computed once (§5.2): the planner hashes, "
        "the verifier validates the identity by content, the client reuses it",
        slice_owner="G3",
    ),
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
        slice_owner="G3",
    ),
    Metric(
        "static_memory.digest.2048",
        "memory",
        "module digest at 2,048 resources",
        88.05,
        "ms",
        "wall",
        slice_owner="G3",
    ),
    Metric(
        "static_memory.verify.2048",
        "memory",
        "identity-bound external verify at 2,048 resources",
        157.88,
        "ms",
        "wall",
        bound=88.05,
        bound_source="an identity-bound API lets an independent verifier "
        "reuse a proved digest instead of recomputing (§5.2)",
        slice_owner="G3",
    ),
    # --- G11 (S1-C): the plan as bytes. Before the slice the canonical plan was Python objects:
    # nothing round-tripped, no second rail could read it, a stale or malformed plan had no
    # bytes to be refused by -- so on the parent tree EVERY fixture of every gate fails, and
    # the rows count those failures over fixed corpora (26 corpus plans = 12 programs x 2
    # placements + the audit fixture x 2; 27 reader fixtures = the plans + the static-memory
    # fixture; 6 stale (fixture, rail) pairs; 39 malformed (variant, rail) pairs). The bound
    # is zero and each row is graded exactly. The three rows that need the C twin are
    # NOT-MEASURED without a C compiler, never estimated from the Python rail alone.
    Metric(
        "plan.abi.mismatches",
        "plan",
        "corpus plans whose Python encode -> C decode -> Python re-encode is NOT byte-identical",
        26,
        "count",
        "exact",
        bound=0,
        bound_source="every plan of the corpus survives the C twin byte for byte "
        "(the G11 gate `plan.abi.roundtrip`)",
        slice_owner="G11",
    ),
    Metric(
        "plan.readers.disagreements",
        "plan",
        "reader fixtures on which pricing, both executors, static memory or the StreamPack "
        "lowering, reading the plan, do NOT reproduce their in-memory traces",
        27,
        "count",
        "exact",
        bound=0,
        bound_source="one artifact, four readers, identical traces "
        "(the G11 gate `plan.readers.agree`)",
        slice_owner="G11",
    ),
    Metric(
        "plan.stale.accepted",
        "plan",
        "(stale-vector fixture, rail) pairs where a plan minted under an older vector, or a "
        "pack older than its plan, is ACCEPTED",
        6,
        "count",
        "exact",
        bound=0,
        bound_source="R11 by bytes on the Python and C rails (G11 gate)",
        slice_owner="G11",
    ),
    Metric(
        "plan.malformed.accepted",
        "plan",
        "(malformed-plan variant, rail) pairs -- truncated, duplicated, out-of-order, unknown "
        "claim, bad mode/stream/duration, unsorted vector, trailing bytes, CRC -- ACCEPTED",
        39,
        "count",
        "exact",
        bound=0,
        bound_source="every variant refused on every rail that can see it (G11 gate)",
        slice_owner="G11",
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

    # The general case (G2 / S2-A): one step-shortening trial per pair under ENERGY on
    # x86_avx2 (the widths the fixture is built for), 16 phases x 16 pairs = 512 claims.
    from bcir.kbcir.weights import ENERGY
    from bcir.tests.sweep_fixtures import general_fixture

    general = general_fixture(16, 16)
    target = TARGETS["x86_avx2"]
    stats: dict = {}
    start = time.perf_counter()
    optimize_scheduled(general, target, Theta.cool(), ENERGY, stats=stats)
    out["optimize_scheduled.general.512"] = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    optimize(general, target, Theta.cool(), ENERGY)
    serial = (time.perf_counter() - start) * 1e3
    if serial:
        out["optimize_scheduled.general.slowdown.512"] = (
            out["optimize_scheduled.general.512"] / serial
        )
    if stats.get("trials"):
        out["sweep.replacement.fraction"] = stats["pops"] / (stats["trials"] * stats["claims"])
    return {k: v for k, v in out.items() if not k.startswith("_")}


def divergence_fixture():
    """The §6.2 fixture: four independent claims across TWO domains, planned by the real cost
    model. Returns (module, host, plan, durations).

    The report measured `price_scheduled` at 51,200 against `schedule_eft` at 25,700 on four
    independent claims with durations [25600, 100, 25600, 100] over two domains; this fixture
    reproduces that ratio to every digit the report prints -- 1.9922178988 against 51200/25700
    = 1.9922 -- with the retired pricer, while its absolute costs are 4x the report's. That is
    the `ratio` vs `wall` classification demonstrating itself: the structure carries across
    scales and machines, the milliseconds do not.

    Three details are load-bearing, each found by getting it wrong first:
      * the claims must touch DISTINCT resources, or they conflict and both schedulers
        serialize them, giving a ratio of 1.0 and hiding the defect;
      * the target must expose exactly TWO domains, because the divergence came from the wave
        pricer's round-robin binning putting both large claims in one bin while EFT split them
        -- with eight domains both found the same answer;
      * the costs must come from the real cost model, because the pricers re-derive them from
        the module and ignore costs supplied on the steps.
    """
    from dataclasses import replace

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
    return module, host, result, durations


def measure_legacy_divergence() -> float | None:
    """The retired wave pricer (`gem.overlap.price_waves_legacy`) against the executor on the
    §6.2 fixture: the number the report measured, kept so the fixture is proven to still
    exhibit the divergence it exists to track. Not a graded row -- the pricer it measures is
    read by nothing but this witness."""
    try:
        from bcir.gem.overlap import price_waves_legacy
        from bcir.gem.schedule import schedule_eft
        from bcir.kbcir.cost import Theta
        from bcir.kbcir.weights import PERF

        module, host, result, durations = divergence_fixture()
        legacy = price_waves_legacy(module, result, host, Theta.cool(), PERF)
        eft = schedule_eft(module, durations, host)
        return legacy.makespan / eft.makespan if eft.makespan else None
    except Exception as exc:  # pragma: no cover - shape probe
        sys.stderr.write(f"[baseline] legacy divergence witness unavailable: {exc}\n")
        return None


def measure_exact() -> dict[str, float]:
    """The deterministic rows: solver optima and fixed-corpus fractions.

    These are the rows that gate. They involve no timing, so they are identical on every
    host and in every run -- if one of them moves, the compiler's DECISIONS changed, which
    is exactly the signal a roadmap slice is supposed to produce.
    """
    out: dict[str, float] = {}

    # §6.2: price_scheduled against schedule_eft on `divergence_fixture`. Since G1 (S1-A) the
    # price READS the executor's placement (`gem.schedule.schedule_plan`), so the row is 1.0
    # by construction; `measure_legacy_divergence` keeps the report's 1.9922 as the witness.
    try:
        from bcir.gem.overlap import price_scheduled
        from bcir.gem.schedule import schedule_eft
        from bcir.kbcir.cost import Theta
        from bcir.kbcir.weights import PERF

        module, host, result, durations = divergence_fixture()
        priced = price_scheduled(module, result, host, Theta.cool(), PERF)
        eft = schedule_eft(module, durations, host)
        if eft.makespan:
            out["pricing.eft.divergence"] = priced.makespan / eft.makespan
    except Exception as exc:  # pragma: no cover - shape probe
        sys.stderr.write(f"[baseline] pricing/EFT divergence unavailable: {exc}\n")

    return out


def measure_scheduler() -> dict[str, float]:
    """The exact rail (G4 / S2-B) over the report's §6.1 corpus and the §6.3 shape.

    The `eft.*` rows read the PROOF RAIL: for every instance the bounded exact search starts
    from the heuristic's placement and reports its incumbent, so the rows are the incumbent
    against the independent partition oracle (`exact_fixtures.partition_optimum`) -- zero
    suboptimal, ratio 1.0 -- while `eft.heuristic.*` keeps the heuristic's own numbers as the
    witness that the corpus still exhibits what the report measured. `solver.unproved.fraction`
    and `solver.gap.p95` are the Stage 2 exit rows; `optimize_scheduled.quality` is the
    one-sweep re-selection against the exhaustive enumeration over `exact_fixtures.quality_corpus`
    (worst ratio).
    """
    from bcir.gem.exact import exact_schedule, exact_selection
    from bcir.kbcir import TARGETS
    from bcir.kbcir.cost import Theta
    from bcir.kbcir.weights import ENERGY, PERF
    from bcir.tests.exact_fixtures import (
        partition_optimum,
        quality_corpus,
        six_job_corpus,
        six_job_module,
        six_job_target,
    )

    out: dict[str, float] = {}
    module = six_job_module()
    corpus = six_job_corpus()
    unproved = 0
    gaps: list[float] = []
    for domains in (2, 3):
        target = six_job_target(domains)
        suboptimal = heuristic_suboptimal = 0
        worst = heuristic_worst = 1.0
        mean = heuristic_mean = 0.0
        for durs in corpus:
            durations = {index + 1: d for index, d in enumerate(durs)}
            certified = exact_schedule(module, durations, target)
            optimum = partition_optimum(durs, domains)
            ratio = certified.incumbent / optimum
            heuristic = certified.heuristic / optimum
            suboptimal += ratio > 1
            heuristic_suboptimal += heuristic > 1
            worst, heuristic_worst = max(worst, ratio), max(heuristic_worst, heuristic)
            mean += ratio
            heuristic_mean += heuristic
            unproved += certified.stop_reason != "optimal"
            gaps.append(certified.incumbent_gap["relative"])
        out[f"eft.suboptimal.{domains}domains"] = suboptimal / len(corpus)
        out[f"eft.worst.{domains}domains"] = worst
        out[f"eft.mean.{domains}domains"] = mean / len(corpus)
        out[f"eft.heuristic.suboptimal.{domains}domains"] = heuristic_suboptimal / len(corpus)
        out[f"eft.heuristic.worst.{domains}domains"] = heuristic_worst
        out[f"eft.heuristic.mean.{domains}domains"] = heuristic_mean / len(corpus)
    out["solver.unproved.fraction"] = unproved / (2 * len(corpus))
    gaps.sort()
    out["solver.gap.p95"] = gaps[min(len(gaps) - 1, int(round(0.95 * (len(gaps) - 1))))]

    worst_quality = 1.0
    for _name, small in quality_corpus():
        for policy in (PERF, ENERGY):
            selection = exact_selection(small, TARGETS["x86_avx512"], Theta.cool(), policy)
            worst_quality = max(worst_quality, selection.ratio)
    out["optimize_scheduled.quality"] = worst_quality
    return out


def measure_dispatch() -> dict[str, float]:
    """The G12 rows (S2-C) over a fixed corpus of interruption points: the six-job schedule
    corpus (every 20th instance, 3 splits), the memory corpus (every 20th fixture and the
    witness, 3 splits) and the section 6.3 selection corpus (10 modules, 3 splits)."""
    from bcir.examples import PROGRAMS
    from bcir.gem.dispatch import DispatchRequest, solve_memory, solve_schedule, solve_selection
    from bcir.gem.exact import certify_schedule, exact_schedule, exact_selection
    from bcir.kbcir import TARGETS, optimize
    from bcir.kbcir.cost import Theta
    from bcir.kbcir.static_memory import exact_layout
    from bcir.kbcir.weights import PERF
    from bcir.tests.exact_fixtures import (
        quality_corpus,
        six_job_corpus,
        six_job_module,
        six_job_target,
    )
    from bcir.tests.memory_fixtures import WORST_FIXTURE, corpus, items_of

    unavailable = missing = 0

    def same_schedule(a, b):
        return (
            a.incumbent,
            a.lower_bound,
            a.stop_reason,
            a.expansions,
            [(s.claim_id, s.domain, s.start, s.finish) for s in a.schedule.slots],
        ) == (
            b.incumbent,
            b.lower_bound,
            b.stop_reason,
            b.expansions,
            [(s.claim_id, s.domain, s.start, s.finish) for s in b.schedule.slots],
        )

    module, target = six_job_module(), six_job_target(2)
    for durs in six_job_corpus()[::20]:
        durations = {i + 1: d for i, d in enumerate(durs)}
        full = exact_schedule(module, durations, target, budget=5000)
        total = max(1, full.expansions)
        for b1 in (0, 1, total // 2):
            try:
                first = exact_schedule(module, durations, target, budget=b1)
                second = exact_schedule(
                    module, durations, target, budget=total - b1, resume=first.state
                )
                if not same_schedule(
                    second, exact_schedule(module, durations, target, budget=total)
                ):
                    unavailable += 1
            except Exception:
                unavailable += 1
            try:
                result, _record = solve_schedule(
                    DispatchRequest("schedule", 6, "TMSAO-1", b1), module, durations, target
                )
                if not getattr(result, "slots", None) and not getattr(
                    getattr(result, "schedule", None), "slots", None
                ):
                    missing += 1
            except Exception:
                missing += 1
    for rows in [items_of(f) for _seed, f in list(corpus())[::20]] + [items_of(WORST_FIXTURE)]:
        full = exact_layout(rows, 200_000)
        total = max(1, full.expansions)
        for b1 in (0, 1, total // 2):
            try:
                if b1 == 0:
                    solve_memory(DispatchRequest("memory", len(rows), "TMSAO-1", 0), rows)
                    unavailable += 0
                else:
                    first = exact_layout(rows, b1)
                    second = exact_layout(rows, max(1, total - b1), resume=first.state)
                    both = exact_layout(rows, b1 + max(1, total - b1))
                    if (second.offsets, second.extent, second.stop_reason, second.expansions) != (
                        both.offsets,
                        both.extent,
                        both.stop_reason,
                        both.expansions,
                    ):
                        unavailable += 1
            except Exception:
                unavailable += 1
            try:
                layout, _record = solve_memory(
                    DispatchRequest("memory", len(rows), "TMSAO-1", b1), rows
                )
                if set(layout.offsets) != {row.rid for row in rows}:
                    missing += 1
            except Exception:
                missing += 1
    for _name, small in quality_corpus(seeds=0)[:10]:
        h = TARGETS["x86_avx512"]
        full = exact_selection(small, h, Theta.cool(), PERF)
        total = max(1, full.assignments)
        for b1 in (0, 1, total // 2):
            try:
                if b1 == 0:
                    solve_selection(
                        DispatchRequest("selection", 4, "TMSAO-1", 0), small, h, Theta.cool(), PERF
                    )
                else:
                    first = exact_selection(small, h, Theta.cool(), PERF, limit=b1)
                    second = exact_selection(
                        small, h, Theta.cool(), PERF, limit=total - b1, resume=first.state
                    )
                    if (second.optimum, second.widths, second.assignments, second.stop_reason) != (
                        full.optimum,
                        full.widths,
                        full.assignments,
                        full.stop_reason,
                    ):
                        unavailable += 1
            except Exception:
                unavailable += 1
            try:
                solve_selection(
                    DispatchRequest("selection", 4, "TMSAO-1", b1), small, h, Theta.cool(), PERF
                )
            except Exception:
                missing += 1
    unrecorded = 0
    for _name, build in sorted(PROGRAMS.items()):
        program = build()
        certificate = certify_schedule(
            program,
            optimize(program, TARGETS["x86_avx512"], Theta.cool(), PERF),
            TARGETS["x86_avx512"],
            Theta.cool(),
            PERF,
        )
        if certificate.to_dict().get("dispatch") is None:
            unrecorded += 1
    return {
        "search.resume.unavailable": float(unavailable),
        "dispatch.incumbent.missing": float(missing),
        "dispatch.unrecorded": float(unrecorded),
    }


def measure_regions() -> dict[str, float]:
    """The G6 exact rows (S2-D) over the region corpus."""
    import random

    from bcir.examples import PROGRAMS
    from bcir.kbcir.differential import gen_module
    from bcir.kbcir.objectives import registry, verify_objective
    from bcir.kbcir.realize import _flatten
    from bcir.kbcir.regions import expand, region_graph, verify_region
    from bcir.tests.sweep_fixtures import general_fixture, random_module

    modules = [build() for _name, build in sorted(PROGRAMS.items())]
    modules.append(general_fixture(2, 3))
    modules += [gen_module(random.Random(seed)) for seed in range(20)]
    modules += [random_module(seed) for seed in range(20)]
    unexpanded = 0
    for module in modules:
        claims = [(pid, claim.id) for pid, claim in _flatten(module)]
        try:
            graph = region_graph(module)
            if any(verify_region(region, module) for region in graph.regions):
                raise ValueError("a region failed its verifier")
            expanded = [(pid, claim.id) for pid, claim in expand(graph, module)]
        except Exception:
            unexpanded += len(claims)
            continue
        covered = set(graph.by_claim())
        unexpanded += sum(1 for pid, cid in claims if cid not in covered)
        if expanded != claims:
            unexpanded += len(claims)
    unverified = 0
    try:
        for entry in registry().values():
            unverified += bool(verify_objective(entry))
    except Exception:
        unverified += 6
    return {
        "regions.unexpanded.claims": float(unexpanded),
        "objectives.unverified": float(unverified),
    }


def measure_native() -> dict[str, float]:
    """The §4.4 native structural wins through the measured-evidence rail (`bcir.bench`),
    compiled and timed on this host -- guardrails, not targets (G6). A ratio here is a
    same-host number: the report's baselines were measured on other silicon, and this host's
    tenancy is what `kbcir.microbench.host_attestation` says it is. Empty when no C
    toolchain is available."""
    from bcir.bench import bench_available, compare, compare_gather, compare_reduce, compare_strided

    if not bench_available():
        return {}
    out: dict[str, float] = {}
    gather = compare_gather("vector_add", opt="-O2", n=1 << 16, reps=30)
    if gather.speedup_milli:
        out["native.gather-avoidance"] = gather.speedup_milli / 1000
    reduce = compare_reduce("gather_reduce", opt="-O2", n=1 << 20, reps=30)
    if reduce.speedup_milli:
        out["native.blocked-reduction"] = reduce.speedup_milli / 1000
    strided = compare_strided("saxpy_strided", opt="-O2", n=1 << 22, reps=30)
    if strided.speedup_milli:
        out["native.direct-stride"] = strided.speedup_milli / 1000
    dense = compare("vector_add", opt="-O3", n=1 << 20, reps=100)
    if dense.speedup_milli:
        out["native.dense-parity"] = dense.speedup_milli / 1000
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


def measure_digest() -> dict[str, float]:
    """The G3 rows (§5.2): the module digest, the static-memory plan (which verifies), and
    an external verify, all over the audit's static-memory fixture at scale 4 -- 2,048
    resources, 3,972 claims, the report's own probe -- plus the exact count of full digests
    the plan-and-verify chain computes.

    The external verify is the identity-bound one the row's bound names: the client holds
    the module's `ModuleIdentity` and the verifier validates it against the module's content
    (the canonical stream) instead of re-hashing. A verifier at a trust boundary still
    recomputes -- `hash_module` is that primitive, and the count row would read 2 if the
    identity were ever refused on this immutable fixture.
    """
    import statistics
    import time

    from bcir.kbcir.provenance import digest_stats, hash_module, module_identity
    from bcir.kbcir.static_memory import plan_static_memory, verify_static_memory_plan
    from bcir.performance_audit import _AuditHardware, static_memory_module

    out: dict[str, float] = {}
    module = static_memory_module(4)
    hardware = _AuditHardware()
    bindings = {rid: "ram" for rid in module.resources}

    def median_ms(fn, repeats=5):
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1e3)
        return statistics.median(samples)

    # The exact gate first, on a fresh identity: plan (with its internal verify) and one
    # identity-bound external verify must compute the digest exactly once.
    module.touch()
    before = digest_stats()["hash_module"]
    plan = plan_static_memory(module, bindings, hardware)
    identity = module_identity(module)
    errors = verify_static_memory_plan(plan, module, bindings, hardware, identity=identity)
    if errors:
        raise AssertionError(f"the static-memory plan failed its external verify: {errors}")
    out["static_memory.digests.2048"] = float(digest_stats()["hash_module"] - before)

    out["static_memory.digest.2048"] = median_ms(lambda: hash_module(module))
    out["static_memory.plan.2048"] = median_ms(
        lambda: (module.touch(), plan_static_memory(module, bindings, hardware))
    )
    out["static_memory.verify.2048"] = median_ms(
        lambda: verify_static_memory_plan(plan, module, bindings, hardware, identity=identity)
    )
    return out


def plan_fixtures():
    """The corpus plans the G11 rows are measured over: every corpus program under the
    x86_avx512/cool profile and the audit's K_BCIR->StreamPack fixture at scale 1 (matmul_tiled
    n=32, 64 claims) under x86_avx2/mem_bound, each in both placements."""
    from bcir.examples import PROGRAMS, matmul_tiled
    from bcir.gem.execution_plan import plan_from_realization
    from bcir.kbcir.cost import TargetProfile, Theta
    from bcir.kbcir.realize import optimize

    rows = [
        (name, build(), TargetProfile.x86_avx512(), Theta.cool())
        for name, build in sorted(PROGRAMS.items())
    ]
    rows.append(
        (
            "audit.kbcir-streampack.1",
            matmul_tiled(n=32, tile=8),
            TargetProfile.x86_avx2(),
            Theta.mem_bound(),
        )
    )
    for name, module, target, theta in rows:
        result = optimize(module, target, theta)
        for mode in ("eft", "tokens"):
            plan = plan_from_realization(module, result, target, mode, plan="plan0")
            yield f"{name}/{mode}", module, target, theta, result, plan


def _plan_harness():
    """Build the C plan harness (runtime/c/test_execution_plan.c); None without a compiler."""
    import shutil
    import subprocess
    import tempfile

    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return None, None
    c_dir = os.path.join(ROOT, "runtime", "c")
    tmp = tempfile.mkdtemp(prefix="bcir-plan-")
    exe = os.path.join(tmp, "test_execution_plan")
    build = subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            os.path.join(c_dir, "bcir_runtime.c"),
            os.path.join(c_dir, "test_execution_plan.c"),
            "-I",
            c_dir,
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        sys.stderr.write(f"[baseline] plan harness build failed: {build.stderr}\n")
        return None, tmp
    return exe, tmp


def measure_plan() -> dict[str, float]:
    """The G11 rows (S1-C): the plan as bytes, counted failures over fixed corpora.

    `plan.abi.mismatches`       corpus plans whose Python encode -> C decode (the harness's
                                record dump) -> Python re-encode is not byte-identical.
    `plan.readers.disagreements` reader fixtures where the pricer's makespan and placement,
                                both executors' placements (and the re-placement from the
                                plan's own step costs), the static memory plan's lifetimes or
                                the StreamPack lowering, read from the plan, do not reproduce
                                their in-memory traces.
    `plan.stale.accepted`       (stale-vector fixture, rail) pairs accepted.
    `plan.malformed.accepted`   (malformed variant, rail) pairs accepted.
    Every row is 0 at the bound; on the parent tree every fixture failed because no plan
    bytes existed (the baselines).
    """
    import shutil
    import subprocess

    from bcir.abi import decode_plan, encode, encode_plan
    from bcir.gem.execution_plan import realization_of, schedule_of
    from bcir.gem.overlap import price_scheduled
    from bcir.gem.schedule import schedule_plan
    from bcir.gem.streampack import hydrate
    from bcir.kbcir.weights import PERF
    from bcir.tests.plan_fixtures import (
        c_roundtrip,
        malformed_variants,
        parse_c_dump,
        stale_fixtures,
        static_memory_lifetimes_agree,
    )
    from bcir.verify import verify_execution_plan

    out: dict[str, float] = {}
    fixtures = list(plan_fixtures())

    # readers agree (Python rail only: the readers are Python organs).
    agree = 0
    for _name, module, target, theta, result, plan in fixtures:
        blob = encode_plan(plan)
        read = decode_plan(blob)
        priced = price_scheduled(module, result, target, theta, PERF, plan.mode)
        ok = priced.makespan == read.makespan
        ok = (
            ok
            and {s.claim_id: (s.domain, s.start, s.finish) for s in priced.schedule.slots}
            == read.slot_map()
        )
        placed = schedule_plan(module, result, target, plan.mode)
        ok = (
            ok
            and {s.claim_id: (s.domain, s.start, s.finish) for s in placed.slots} == read.slot_map()
        )
        again = schedule_plan(module, realization_of(read), target, plan.mode)
        ok = (
            ok
            and {s.claim_id: (s.domain, s.start, s.finish) for s in again.slots} == read.slot_map()
        )
        ok = ok and schedule_of(read).makespan == placed.makespan
        ok = ok and encode(hydrate(module, realization_of(read), read.source_plan)) == encode(
            hydrate(module, result, plan.source_plan)
        )
        ok = ok and not verify_execution_plan(module, read, target=target, result=result)
        agree += bool(ok)
    total = len(fixtures) + 1
    agree += bool(static_memory_lifetimes_agree())
    out["plan.readers.disagreements"] = float(total - agree)

    exe, tmp = _plan_harness()
    try:
        if exe is not None:
            identical = 0
            for _name, _module, _target, _theta, _result, plan in fixtures:
                blob = encode_plan(plan)
                identical += encode_plan(parse_c_dump(c_roundtrip(exe, tmp, blob))) == blob
            out["plan.abi.mismatches"] = float(len(fixtures) - identical)
            refused, count = stale_fixtures(exe, tmp)
            out["plan.stale.accepted"] = float(count - refused)
            refused, count = malformed_variants(exe, tmp)
            out["plan.malformed.accepted"] = float(count - refused)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    _ = subprocess  # the harness is driven by the shared fixture module
    return out


def measure_memory() -> dict[str, float]:
    """The G5 rows (S1-D): the static memory planner's engaged layout against the proved
    optimum, over the section 6.4 corpus and its worst-case witness, through the REAL planner
    (`kbcir.static_memory.plan_static_memory`, 64-byte lines and alignment) with the bounded
    exact solver engaged (`layout="exact"`). A fixture the solver cannot prove within its
    budget counts as suboptimal (a stated gap, never a claimed optimum)."""
    from bcir.kbcir.static_memory import plan_static_memory
    from bcir.performance_audit import _AuditHardware
    from bcir.tests.memory_fixtures import WORST_FIXTURE, corpus, module_of, unit_layouts

    hardware = _AuditHardware()
    suboptimal = 0
    worst = 1.0
    count = 0
    for seed, rows in corpus():
        module, bindings = module_of(rows, f"corpus-{seed}")
        plan = plan_static_memory(module, bindings, hardware, layout="exact")
        bank = plan.banks[0]
        _first_fit, proved = unit_layouts(rows)
        optimum = proved.extent * 64
        if bank.stop_reason != "optimal" or proved.stop_reason != "optimal":
            suboptimal += 1
            worst = max(worst, bank.extent_bytes / max(optimum, 1))
        else:
            suboptimal += bank.extent_bytes > optimum
            worst = max(worst, bank.extent_bytes / optimum)
        count += 1
    out = {
        "memory.suboptimal.fraction": suboptimal / count,
        "memory.worst.ratio": worst,
    }
    module, bindings = module_of(WORST_FIXTURE, "section-6.4-worst")
    plan = plan_static_memory(module, bindings, hardware, layout="exact")
    out["memory.real.bytes"] = float(plan.banks[0].extent_bytes)
    return out


_MEASURERS = {
    "audit": measure_audit,
    "planner": measure_planner,
    "scheduler": measure_scheduler,
    "dispatch": measure_dispatch,
    "regions": measure_regions,
    "native": measure_native,
    "exact": measure_exact,
    "verifier": measure_verifier,
    "digest": measure_digest,
    "plan": measure_plan,
    "memory": measure_memory,
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
        help="limit measurement to a group (audit, planner, scheduler, dispatch, regions, native, exact, verifier, digest, plan, memory)",
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
        with open(args.json, "w", encoding="utf-8", newline="\n") as handle:
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
