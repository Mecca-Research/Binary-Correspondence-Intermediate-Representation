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
        "host_dependent",
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
        host_dependent=False,
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
        # A ratio is host-portable only when BOTH sides are timed in the same process on the
        # same machine, so the machine cancels (see `verdict`). A ratio between two separately
        # COMPILED kernels does not cancel it: it measures a microarchitectural property --
        # this host's gather penalty, its store-forwarding, its prefetchers -- and reading it
        # against another host's number manufactures verdicts in both directions. Such a row
        # is measured and reported everywhere, and graded only on the baseline host.
        self.host_dependent = host_dependent

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

        `host_dependent` extends that rule to the ratios the argument above does NOT cover.
        It holds for a ratio of two operations timed in ONE process, where the machine
        divides out. It fails for a ratio of two separately compiled kernels: `native.*`
        divides a gather realization by a blocked one, which is a measurement OF the host's
        gather penalty, not a measurement that cancels it. The §4.4 band was taken on the
        report's machine and this one reproducibly sits below it, so grading the row across
        hosts made `--group native --compare` exit nonzero on roughly one run in thirty --
        a gate firing on the machine rather than on the code (S2-D shipped it that way).
        """
        if value is None:
            return "NOT-MEASURED"
        if not same_host and (self.kind == "wall" or self.host_dependent):
            return "INDICATIVE"
        change = self.improvement(value)
        if change > self.noise:
            return "GAIN"
        if change < -self.noise:
            return "REGRESSION"
        return "NO-CHANGE"

    def improvement(self, value: float) -> float:
        """Signed fractional improvement over the baseline, positive = better.

        A baseline of zero has no fraction to move by, and this returned 0.0 for it -- so a
        guard row held at zero (a fact that must stay false, S5-A's `alias.llvm.false_noalias`)
        could climb to any count and still grade NO-CHANGE. Any move off a zero baseline is the
        whole of it: 100% in the direction it went."""
        base = self.baseline
        if base == 0:
            if value == 0:
                return 0.0
            return -1.0 if (value > 0) == self.lower_is_better else 1.0
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
    # --- G13 (S2-E): the workload component W, the measured-candidate corpus and the replay
    # gate. Counted over the 12 corpus programs; the baselines are the parent tree, where no
    # certificate could declare a workload (every scope collided), the portfolio admitted any
    # certificate with one clean episode whatever the log held, and a TMSAO-3 request went to
    # the fast rail whatever evidence existed.
    Metric(
        "scope.workload.collisions",
        "workload",
        "pairs of distinct workloads on one corpus program whose certificate scopes share a digest "
        "(12 programs x 3 workloads: 36 pairs)",
        36,
        "count",
        "exact",
        bound=0.0,
        bound_source="W is a component of the scope: two workloads on one program never share a "
        "digest (G13); the parent tree could not declare one",
        slice_owner="G13",
    ),
    Metric(
        "replay.subset.admitted",
        "workload",
        "promotions the portfolio admits on a certificate covering fewer episodes than the corpus "
        "logs (12 programs x 2 candidates, 3 logged episodes, a one-episode certificate)",
        24,
        "count",
        "exact",
        bound=0.0,
        bound_source="a corpus certificate must replay every logged episode (G13); the parent "
        "tree's certificate carried no corpus and admitted any clean subset",
        slice_owner="G13",
    ),
    Metric(
        "dispatch.measured.unavailable",
        "workload",
        "TMSAO-3 requests with a declared workload and corpus evidence that the law sends to the "
        "fast rail (12 programs)",
        12,
        "count",
        "exact",
        bound=0.0,
        bound_source="the measured rail is dispatched over a declared W and existing evidence "
        "(G13); the parent tree's law sent every TMSAO-3 request to the fast rail",
        slice_owner="G13",
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
    # --- G14 (S3-A): the control plane as bytes. Before the slice lease, generation, quiesce,
    # activate, rollback and cancel were prose, an `admit(map_gen, data_gen)` argument and two
    # methods that raise: no record had bytes, no rail could refuse one, a mid-phase switch was
    # refused rather than deferred. So on the parent tree every fixture fails by absence, and
    # the rows count those failures over fixed corpora (bcir/tests/control_fixtures.py): 29
    # corpus records; 41 malformed variants x 2 rails; 12 stale fixtures x 2 rails + the three
    # Python-only boundaries -- the trusted loader and context-shard activation held on the
    # parent (measured: the fixture's own checks, run against its modules) and the verifier's
    # had no entry point; 8 mid-phase fixtures x 2 rails; 32 transition/authority/witness
    # scenarios x 2 rails; 52 two-rail traces. Every row is exact and bounded at zero; all six
    # need the C twin and are NOT-MEASURED without a C compiler, never estimated from the
    # Python rail alone.
    Metric(
        "control.abi.mismatches",
        "control",
        "corpus records whose Python encode -> C decode -> Python re-encode is NOT byte-identical, "
        "or whose honest MAC either rail's keyed check refuses",
        29,
        "count",
        "exact",
        bound=0,
        bound_source="every record of every kind, scope and reason survives the C twin byte for "
        "byte (the G14 gate `control.record.bytes`)",
        slice_owner="G14",
    ),
    Metric(
        "control.malformed.accepted",
        "control",
        "(malformed variant, rail) pairs -- one per wire law: truncated, trailing, magic, "
        "version, reserved, kind, body length, CRC, scope, reason, capability, sequence, lease, "
        "the generation witness, every body law, a zero MAC -- NOT refused with the declared status",
        82,
        "count",
        "exact",
        bound=0,
        bound_source="an unknown version, a reserved or trailing byte and every other wire law "
        "refused on both rails with the same status (G14 gate)",
        slice_owner="G14",
    ),
    Metric(
        "control.stale.accepted",
        "control",
        "(stale fixture, rail) pairs where a record, pack or plan minted against a generation the "
        "plane has left is NOT refused, plus Python-only boundaries (loader, context shard, "
        "verifier) that do not hold",
        25,
        "count",
        "exact",
        bound=0,
        bound_source="a stale generation is refused by bytes at every boundary (G14 gate)",
        slice_owner="G14",
    ),
    Metric(
        "control.deferred.lost",
        "control",
        "(mid-phase fixture, rail) pairs where a switch requested mid-phase or before its "
        "boundary is NOT deferred and then decided exactly once at the boundary",
        16,
        "count",
        "exact",
        bound=0,
        bound_source="a mid-phase switch is deferred, never applied early, never applied twice, "
        "never lost (G14 gate `quiescent switch`)",
        slice_owner="G14",
    ),
    Metric(
        "control.decisions.nonconforming",
        "control",
        "(transition, authority or refusal-witness scenario, rail) pairs whose plane decisions "
        "are NOT the specification's -- every kind's transition, every forgery, every refusal code",
        64,
        "count",
        "exact",
        bound=0,
        bound_source="every record decided by its bytes as docs/kernel/BCIR_CONTROL_PLANE_ABI.md "
        "orders it, and every refusal the plane can name witnessed (L22)",
        slice_owner="G14",
    ),
    Metric(
        "control.traces.divergent",
        "control",
        "scenarios whose two rails disagree in any verdict, refusal, status or resident state "
        "digest after any operation",
        52,
        "count",
        "exact",
        bound=0,
        bound_source="one plane, two realizations, identical traces (the C twin decides every "
        "record as the oracle does)",
        slice_owner="G14",
    ),
    # --- G15 (S3-B): the live SPSC ring and the version-zero triple (the generated signal table,
    # TelemetryEnvelopeV0, the ring itself). Before the slice the only shared telemetry ring was a
    # quiescent snapshot (a head and slots, no tail, no publication protocol, no loss count, no
    # epoch): measured on the parent, a reader behind by a lap received the survivors and no count,
    # and a slot read mid-rewrite came back with the fields of two records and no error. There was
    # no oracle, no C twin, no envelope and no generated table, so every fixture fails by absence and
    # the rows count those failures over fixed corpora (bcir/tests/ring_fixtures.py): 15 table rows
    # + the generated header + 13 envelopes; 23 malformed envelopes x 2 rails + 33 malformed
    # scenarios x 2 rails; 70 scripted scenarios x 2 rails (loss, torn); 11 continuity fixtures x 2;
    # 5 stale fixtures x 2; 70 two-rail traces; the 52 G14 scenarios through a live control ring x 2
    # rails; 6 concurrent C runs. Every row is exact and bounded at zero and needs the C twin: without
    # a C compiler the group is NOT-MEASURED, and without POSIX threads/processes the concurrent row is.
    Metric(
        "ring.abi.mismatches",
        "ring",
        "generated signal-table rows whose C bytes differ from Python's, the generated header's "
        "drift, and corpus envelopes whose Python encode -> C decode -> re-encode is not identical",
        29,
        "count",
        "exact",
        bound=0,
        bound_source="the taxonomy and the envelope are the same bytes on both rails",
        slice_owner="G15",
    ),
    Metric(
        "ring.malformed.accepted",
        "ring",
        "(malformed variant, rail) pairs NOT refused with the declared status: every envelope wire "
        "law, every geometry law, owner and slot corruption, the unknown-required-signal law",
        112,
        "count",
        "exact",
        bound=0,
        bound_source="every law refuses its own variant on both rails",
        slice_owner="G15",
    ),
    Metric(
        "ring.loss.unaccounted",
        "ring",
        "(scenario, rail) pairs whose committed accounting is not exactly what the rail's verdicts "
        "reported (published == delivered + lost + stale; refused counted) or not the declared numbers",
        140,
        "count",
        "exact",
        bound=0,
        bound_source="records lost to overwrite are counted exactly; a consumer that falls behind "
        "sees the count (the G15 gate `ring.loss.accounting`)",
        slice_owner="G15",
    ),
    Metric(
        "ring.torn.delivered",
        "ring",
        "(scenario, rail) pairs that delivered a record not byte-identical to what was published "
        "at its position",
        140,
        "count",
        "exact",
        bound=0,
        bound_source="a consumer never sees a torn record (the G15 gate)",
        slice_owner="G15",
    ),
    Metric(
        "ring.sequence.misreported",
        "ring",
        "(continuity fixture, rail) pairs whose (missing, reordered, duplicated) is not the declared "
        "triple",
        22,
        "count",
        "exact",
        bound=0,
        bound_source="gaps, reorders and duplicates reported as the frame ABI reports them (the "
        "G15 sequence-continuity gate; one predicate, SequenceTracker)",
        slice_owner="G15",
    ),
    Metric(
        "ring.stale.accepted",
        "ring",
        "(stale fixture, rail) pairs not refused as declared: a deposed producer or consumer, a "
        "record stamped with a deposed epoch, telemetry of a generation the plane has left",
        10,
        "count",
        "exact",
        bound=0,
        bound_source="stale generations refused at every boundary (the Stage 3 exit)",
        slice_owner="G15",
    ),
    Metric(
        "ring.traces.divergent",
        "ring",
        "scenarios whose two rails' traces differ in any line (verdict, status, position, count, "
        "epoch, payload, or the region's CRC after the operation)",
        70,
        "count",
        "exact",
        bound=0,
        bound_source="one ring, two realizations, the same bytes after every step",
        slice_owner="G15",
    ),
    Metric(
        "ring.control.divergent",
        "ring",
        "(G14 control scenario, rail) pairs whose plane trace through a live control ring differs "
        "from the direct trace, plus scenarios whose two rails' ring traces differ",
        104,
        "count",
        "exact",
        bound=0,
        bound_source="the ring is a transport, never a decision (control records second)",
        slice_owner="G15",
    ),
    Metric(
        "ring.concurrent.violations",
        "ring",
        "concurrent C runs (threads; processes with a peer SIGKILLed and taken over) that tore, "
        "failed to account, or broke continuity",
        6,
        "count",
        "exact",
        bound=0,
        bound_source="the same laws with the peers truly concurrent and dying",
        slice_owner="G15",
    ),
    # The G15 ratio: two threads streaming DataDNA envelopes (124 bytes) through a backpressure
    # ring against one thread memcpy-ing the same bytes. Not measurable on the parent (no ring), so
    # the baseline is the slice's first measurement on the reference host of this program (4 vCPU
    # virtualized, clang 18; medians of 5): 26.0x -- the ring pays cross-core cache-line transfers
    # that memcpy never pays. On that host the ratio is dominated by where the two threads land:
    # pinned per vCPU pair it ranged ~13-49x within the hour (a minimal unchecked Lamport queue
    # moving the same bytes, ~1.1-6x), so the row is reported, never graded, and no code change is
    # attributable to it there. host_dependent: it measures the host's placement and coherence
    # latency, not only the code.
    Metric(
        "ring.throughput",
        "ring",
        "ring time per record / memcpy time per record, the same 124-byte envelopes (two threads "
        "vs one)",
        26.0,
        "x",
        "ratio",
        bound=1.0,
        bound_source="the memcpy floor of the same bytes (a two-thread handoff cannot reach it)",
        slice_owner="G15",
        host_dependent=True,
    ),
    # --- G16 (S3-C): the data-plane hand-off. On the parent (1ee34676) the C++ seam took a raw
    # (pointer, length) artifact: admit() gated nothing by default and compared header maxima when
    # handed numbers, dispatch() consulted no admission, a view outlived its buffer silently, the
    # dynamic-graph backend was a stub and no manifest-of-shards existed. The pack table, the freeze,
    # the manifest and the C++ RAII types did not exist on any rail, so every fixture fails by
    # absence; the rows count those failures over fixed corpora (bcir/tests/handoff_fixtures.py: 40
    # scenarios, 296 operations on three rails -- 3 C-only; 187 freeze cases; 73 split cases, 8
    # refused splits, 25 malformed manifests, 9 tampered sets, 42 shard runs, 10 C++ witnesses; the
    # Stage 3 exit flow's 30 declared outcomes and its evidence line on three rails).
    # Every row is exact and bounded at zero; a rail without its compiler is NOT-MEASURED.
    Metric(
        "handoff.stale.admitted",
        "handoff",
        "(stale admission, rail) pairs not refused as stale: a pack older than the live registry -- the maxima moved, a resource moved under unchanged maxima (map or data), no vector, a resource undeclared or missing, another program with coinciding maxima, the topology moved -- and a stale shard manifest",
        42,
        "count",
        "exact",
        bound=0,
        bound_source="admit() refuses a pack older than the registry generation (the G16 gate); the parent's own seam admitted 9 of 9 by default and 8 of 9 handed the live maxima",
        slice_owner="G16",
    ),
    Metric(
        "handoff.stale.dispatched",
        "handoff",
        "(dispatch the specification refuses, rail) pairs not refused as declared: never admitted, admitted in a generation the plane has left (a switch or an activation between admit and dispatch), a draining plane",
        90,
        "count",
        "exact",
        bound=0,
        bound_source="only what was admitted at the resident generation of the installed registry runs (the parent's dispatch consulted no admission at all)",
        slice_owner="G16",
    ),
    Metric(
        "handoff.fresh.refused",
        "handoff",
        "(fresh admission or dispatch, rail) pairs refused: the honest half the stale rows must not buy with over-refusal",
        90,
        "count",
        "exact",
        bound=0,
        bound_source="a current pack is admitted and runs",
        slice_owner="G16",
    ),
    Metric(
        "handoff.lifetime.unrefused",
        "handoff",
        "(access through a dead handle or view, rail) pairs not refused BCIR_ERR_LIFETIME -- released, reused by the next step's same-length pack, aborted, never issued, a returned borrow -- plus the C++ lifetime witnesses (an owner out of scope, an arena gone, a moved owner, a double return, a foreign arena)",
        68,
        "count",
        "exact",
        bound=0,
        bound_source="a view that outlives its owner is refused (the G16 gate); on the parent a reused buffer was dispatched through the old view and a freed one read freed memory (ASan)",
        slice_owner="G16",
    ),
    Metric(
        "handoff.copies",
        "handoff",
        "segment views the C++ seam's dispatches read outside the arena slot the view names, over every scenario and shard run, plus the moved-once witness",
        80,
        "count",
        "exact",
        bound=0,
        bound_source="the hand-off moves bytes once: written into the slot, read in place (the G16 gate)",
        slice_owner="G16",
    ),
    Metric(
        "handoff.builder.violations",
        "handoff",
        "(builder step, rail) pairs not deciding as declared -- a step frozen, admitted and run; a step frozen before a registry switch refused after it; every freeze law refused transactionally -- plus freeze cases whose C bytes or refusal differ from the oracle's",
        422,
        "count",
        "exact",
        bound=0,
        bound_source="the dynamic-graph builder freezes a fresh StreamPack per step through the C/IR rail",
        slice_owner="G16",
    ),
    Metric(
        "handoff.decisions.nonconforming",
        "handoff",
        "every other declared (operation, rail) outcome not as declared: the table's capacity, fullness and bounds (a full table, an exhausted epoch or pin count), a malformed admission, the producer's own steps",
        349,
        "count",
        "exact",
        bound=0,
        bound_source="the table decides every operation as specified",
        slice_owner="G16",
    ),
    Metric(
        "handoff.shards.mismatches",
        "handoff",
        "split cases whose manifest, frame or shards differ between the oracle, the C twin and the C++ seam's cut; wholes a rail does not reassemble byte for byte; packs outside the hydrated layout not refused",
        392,
        "count",
        "exact",
        bound=0,
        bound_source="shards by digest reassemble to the whole-pack bytes (the G16 gate), identically on every rail",
        slice_owner="G16",
    ),
    Metric(
        "handoff.shards.malformed.accepted",
        "handoff",
        "(malformed manifest or tampered shard set, rail) pairs not refused with the declared status: one variant per manifest wire law; a missing, forged, foreign or non-canonical shard; a manifest whose tags or registry digest lie",
        68,
        "count",
        "exact",
        bound=0,
        bound_source="every law refuses its own variant on both rails; only the declared whole ever reassembles",
        slice_owner="G16",
    ),
    Metric(
        "handoff.reentry.divergent",
        "handoff",
        "(whole, world, rail) cases whose shards, each admitted by the live plane and run by itself, do not reproduce the whole's dispatch",
        115,
        "count",
        "exact",
        bound=0,
        bound_source="the per-rank re-entry reproduces the single-node dispatch exactly (the L5 preparation: only the network is missing)",
        slice_owner="G16",
    ),
    Metric(
        "handoff.traces.divergent",
        "handoff",
        "scenarios whose native trace (C twin, C++ seam) differs from the oracle's in any line: outcome, handle, generation, claims, the table's and the plane's digests",
        77,
        "count",
        "exact",
        bound=0,
        bound_source="one hand-off, three realizations, the same state after every step",
        slice_owner="G16",
    ),
    # The Stage 3 exit gate (the staged plan, section 6): one artifact generation flows plan ->
    # control -> data -> telemetry -> evidence on the loopback, and after the switch the old
    # generation is offered at every boundary. The flow (runtime/c/test_stage3.h; the oracle is
    # handoff_fixtures.run_stage3_python) needs the pack table, so on the parent it fails by
    # absence on every rail; the parent's real seam admitted and ran the stale pack besides.
    Metric(
        "handoff.stage3.stale.accepted",
        "handoff",
        "(boundary, rail) pairs at which the Stage 3 exit flow accepted the old generation after the switch: a control record witnessed against it, its plan, its pack's dispatch and admission, its shard manifest, a late telemetry sample bound to it",
        18,
        "count",
        "exact",
        bound=0,
        bound_source="stale generations refused at every boundary (the Stage 3 exit gate)",
        slice_owner="G16",
    ),
    Metric(
        "handoff.stage3.flow.divergent",
        "handoff",
        "(declared step, rail) pairs of the Stage 3 exit flow not as declared -- the records carried by the control ring, both plans, both packs stored, admitted, gated and run, their telemetry through the two-slot ring into the intake -- plus evidence laws that do not reconcile (ring loss = intake gaps, one stale record, every delivered record decided, every record carried) and native flows that differ from the oracle's",
        75,
        "count",
        "exact",
        bound=0,
        bound_source="one generation flows plan -> control -> data -> telemetry -> evidence, identically on every rail",
        slice_owner="G16",
    ),
    # The G16 ratio: the seam's dispatch of an admitted pack over the direct C walk of the same bytes
    # (one thread, the same callback; the 120-segment synthetic pack). Measured on the parent's own
    # seam (its dispatch(): shard() re-validated the whole pack, then the walk validated it again),
    # medians of 7 rounds of 2,000 on the reference host: 1.93-1.98x, baseline 1.95x. The G16 seam
    # checks the view's lifetime, the admission's generation and registry and the plane phase per
    # dispatch -- all O(1) -- and moves the full verification and the registry digest to admit(),
    # once: ~1.00-1.02x. A single-core ratio of the same work, so the band holds across hosts.
    Metric(
        "handoff.dispatch.overhead",
        "handoff",
        "seam dispatch time / direct C walk time, the same admitted 120-segment pack and callback",
        1.95,
        "x",
        "ratio",
        bound=1.0,
        bound_source="the direct C walk of the same bytes: every per-dispatch check is O(1)",
        slice_owner="G16",
    ),
    # --- G17 (S4-A): the compact planner and its native twin. The planner's offer became compact
    # indexed arrays behind the same API (`realize.fused_offer`), held to the pre-G17 planner kept
    # verbatim as `realize_reference`, and the native planner (`runtime/c/bcir_kplan.c`) is held
    # to both byte for byte (bcir/tests/planner_fixtures.py::measure, which the tests and
    # tools/c/check_runtime.sh grade the same way). RED is the parent (731373df) with every new
    # entry point made to raise over the final corpora, built first (L25).
    Metric(
        "planner.parity",
        "kplan",
        "(case, comparison) pairs where the compact planner differs from the pre-G17 planner -- the plan, its BKPR bytes, and on the fixed corpus the ExecutionPlanV1 bytes -- or the native planner from the compact one (its BKPR bytes, or the same refusal), over the fixed corpus under every target, Theta and policy and 240 generated modules under every target",
        8430,
        "count",
        "exact",
        bound=0,
        bound_source="identical plan bytes before and after, and Python versus native (roadmap G17)",
        slice_owner="G17",
    ),
    Metric(
        "planner.malformed.accepted",
        "kplan",
        "(malformed record, rail) pairs not refused with the declared status: one BKPI variant per wire and planning law, one BKPR variant per wire law (54 + 20), on the Python codec and the C twin",
        148,
        "count",
        "exact",
        bound=0,
        bound_source="every law refuses its own violation with one status on both rails",
        slice_owner="G17",
    ),
    Metric(
        "planner.r9.misjudged",
        "kplan",
        "(plan, call) pairs R9 misjudges through the planner's offer: the planner's own plan of a legal module refused, or a forgery of a field a step carries (name, width, base, lane -- including a plain int -- cost, phase, an unhashable name or phase) accepted or answered with a traceback; 14 legal modules x 2 scopes, 396 forgeries, each graded with the scope and (all but a forged cost) without it",
        232,
        "count",
        "exact",
        bound=0,
        bound_source="every forgery refused with a diagnostic and every honest plan accepted; the parent raised on 176 forgery verdicts (a plain-int lane, an unhashable phase) and accepted 56 (a first step's phase was never bound to its claim)",
        slice_owner="G17",
    ),
    # The 2026-09-04 profile's 6.06 M was the whole K_BCIR->StreamPack chain at scale 8 before
    # S0-A made R9 re-derive the planner's offer; this row is the planner alone, the thing G17
    # rewrites, on the parent under CPython 3.11.15 (call counts differ between interpreters: the
    # tests compare two planners in one process instead, and state the factor).
    Metric(
        "planner.calls",
        "kplan",
        "calls (cProfile total, builtins included) planning the audit's K_BCIR->StreamPack fixture at scale 8 (32,768 claims), CPython 3.11",
        3419172,
        "calls",
        "exact",
        slice_owner="G17",
    ),
    Metric(
        "planner.native.scale4",
        "kplan",
        "native planner (bcir_kplan.c, -O2) median time per plan of the audit fixture at scale 4 (4,096 claims), BKPI decoded once",
        3.41,
        "ms",
        "wall",
        slice_owner="G17",
    ),
    # --- G18 (S4-B): the K_BCIR -> StreamPack chain advanced by declared deltas. `DeltaChain` holds
    # the incremental plan (`kbcir.delta`), the delta StreamPack (`gem.delta_pack`) and the
    # incremental verdict (`verify.delta`); every link is held to the chain run from scratch on the
    # module the delta declares (bcir/tests/delta_fixtures.py::measure, which the tests and
    # tools/perf/check_delta.py grade the same way) over 258 cases of 8 steps (a build and 7
    # deltas). RED is the slice's parent (825888e9, the S4-A head) running the final corpus: the
    # mechanisms are absent there, so every comparison they own fails and a delta costs the chain
    # from scratch.
    Metric(
        "planner.delta.parity",
        "delta",
        "(case, step) pairs where the chain's incremental plan differs from optimize() of the declared module -- the steps, their costs, the score, the BKPR bytes or refusal -- or its module from the declared one; 258 cases x (a build + 7 deltas)",
        2064,
        "count",
        "exact",
        bound=0,
        bound_source="the plan a delta advances to is optimize() of the module it declares, byte for byte (roadmap G18)",
        slice_owner="G18",
    ),
    Metric(
        "pack.delta.identity",
        "delta",
        "(case, step) pairs where the delta StreamPack differs from encode(hydrate_pipelined()) of the declared module -- the pack, its bytes, or the refusal with its message; 258 cases x 8 steps, a wire refusal (beside an edit that must survive it) and its repair in every case that can carry one",
        2064,
        "count",
        "exact",
        bound=0,
        bound_source="the re-emitted pack is the hydrated pack, byte for byte, and refuses what encode refuses",
        slice_owner="G18",
    ),
    Metric(
        "verify.delta.identity",
        "delta",
        "(case, step, rail) triples where the incremental verdict differs from verify() + verify_plan() + verify_pack(): the chain's own links, and a rail handing the verdict each step's plan and pack with a plan and a pack field forged (17 forgery kinds, the stale plan and pack of the step before among them); 258 cases x 8 steps x 2 rails",
        4128,
        "count",
        "exact",
        bound=0,
        bound_source="the verdict re-derived per unit is the full verdict, over honest and forged inputs alike",
        slice_owner="G18",
    ),
    Metric(
        "delta.malformed.accepted",
        "delta",
        "(malformed input, rail) pairs not refused with DeltaError before anything moved: 15 malformed deltas on apply_delta, IncrementalPlan.apply and DeltaChain.apply (each followed by an honest delta that must reproduce the chain from scratch), a module declaring a claim id twice on 4 rails, a module changed outside a delta on 2",
        51,
        "count",
        "exact",
        bound=0,
        bound_source="every delta the chain does not admit is refused before anything moves, on every rail",
        slice_owner="G18",
    ),
    Metric(
        "kbcir-streampack.delta",
        "delta",
        "median time of one DeltaChain.apply (a one-claim count edit) / median time of the chain from scratch on the module it declares, audit fixture at scale 4 (4,096 claims), one process",
        1.0,
        "x",
        "ratio",
        slice_owner="G18",
    ),
    Metric(
        "kbcir-streampack.delta.calls",
        "delta",
        "calls (cProfile total, builtins included) of one one-claim delta of the audit fixture at scale 8 (32,768 claims), CPython 3.11: on the parent, the chain from scratch",
        10855666,
        "calls",
        "exact",
        slice_owner="G18",
    ),
    # --- G9 (S5-A): the declared alias facts carried the rest of the way to LLVM. Each row counts
    # failures over the fixed corpus of bcir/tests/alias_fixtures.py::measure, which the tests and
    # tools/perf/check_alias.py grade the same way: 504 lawful kernels (7 RID partitions x 3 ops x
    # 2 element types x 4 widths x 3 contracts) on the LLVM and C backends, the gather form, the
    # hot-shape specialist, the ABI header and the Q-fixed kernel once per plan, 112 modules the
    # subset must refuse, 290 one-fact flips per rail, 5 self-checks per plan and 21 forgery kinds.
    # RED is the slice's parent (ad4ebff0, the S4-B head) running the final corpus.
    Metric(
        "alias.noalias.mismatch",
        "alias",
        "pointer parameters whose no-alias assertion (LLVM noalias, C restrict) is not the RID partition's -- false on a shared resource or dropped on an exclusive one -- over the LLVM kernel and the C kernel, gather form, specialist, ABI header and Q-fixed kernel",
        648,
        "count",
        "exact",
        bound=0,
        bound_source="noalias / restrict exactly where the declared RIDs prove a pointer exclusive, on every emitter (roadmap G9)",
        slice_owner="G9",
    ),
    Metric(
        "alias.scope.mismatch",
        "alias",
        "LLVM memory accesses whose !alias.scope / !noalias do not encode the RID partition (one scope per resource in one domain; its own in !alias.scope, every other in !noalias); 2,646 accesses",
        2646,
        "count",
        "exact",
        bound=0,
        bound_source="alias scopes derived from the RID partition on every access (roadmap G9)",
        slice_owner="G9",
    ),
    Metric(
        "alias.tbaa.mismatch",
        "alias",
        "LLVM memory accesses without the TBAA tag of the declared element type -- the tag clang gives the same C type; 2,646 accesses",
        2646,
        "count",
        "exact",
        bound=0,
        bound_source="TBAA from the declared element type on every access (roadmap G9)",
        slice_owner="G9",
    ),
    Metric(
        "alias.volatile.mismatch",
        "alias",
        "LLVM memory accesses and C pointer parameters (kernel, gather form, specialist, header, Q-fixed kernel) whose volatility is not the claim's",
        1533,
        "count",
        "exact",
        bound=0,
        bound_source="volatile carried through to LLVM, not fenced in BCIR only (roadmap G9)",
        slice_owner="G9",
    ),
    Metric(
        "alias.fence.mismatch",
        "alias",
        "kernels whose fences are not the hazard's (seq_cst first and last for a barriered claim, none for a unique one), LLVM and every C emitter with a body",
        742,
        "count",
        "exact",
        bound=0,
        bound_source="the hazard contract realized in the kernel R12 already required it of",
        slice_owner="G9",
    ),
    Metric(
        "alias.refusal.accepted",
        "alias",
        "(module, emitter) pairs lowered that the elementwise subset must refuse: an atomic or unknown hazard, an operand resource declaring an element size the kernel does not address or not declared at all; 112 modules on 6 emitters (the Q-fixed kernel, whose lanes read no declared element, on the hazards only)",
        602,
        "count",
        "exact",
        bound=0,
        bound_source="the subset refuses what it does not generate instead of lowering it to plain accesses",
        slice_owner="G9",
    ),
    Metric(
        "alias.differential.silent",
        "alias",
        "(module, module', emitter) triples differing in exactly one declared fact (the RID partition, volatility, the hazard, an operand's element size, the element type) whose emitted facts are identical",
        1072,
        "count",
        "exact",
        bound=0,
        bound_source="two modules differing only in an alias fact differ in the emitted facts (roadmap G9)",
        slice_owner="G9",
    ),
    Metric(
        "alias.harness.unaliased",
        "alias",
        "self-checks (LLVM AOT/JIT harness, C and Q-fixed self-checks, WASM node harness) that do not bind one buffer per declared resource; 105 over 21 plans",
        75,
        "count",
        "exact",
        bound=0,
        bound_source="every self-check runs the kernel with the aliasing the claim declares",
        slice_owner="G9",
    ),
    Metric(
        "alias.r12.rejected",
        "alias",
        "(kernel, backend) pairs the emitter produced and its own lowering law R12 rejects; 504 kernels on 2 backends",
        336,
        "count",
        "exact",
        bound=0,
        bound_source="the emitter and its lowering law agree on every lawful kernel",
        slice_owner="G9",
    ),
    Metric(
        "alias.r12.forgery.accepted",
        "alias",
        "forged kernels (one fact dropped, added or contradicted; 21 forgery kinds over 84 kernels on 2 backends) to which R12 raises nothing it did not raise to the honest one",
        1084,
        "count",
        "exact",
        bound=0,
        bound_source="R12 holds every emitted alias fact to the declaration",
        slice_owner="G9",
    ),
    Metric(
        "alias.llvm.false_noalias",
        "alias",
        "access pairs (one at least a store) through two positions naming one resource that LLVM's default alias analysis proves NoAlias (opt aa-eval, 84 kernels); needs a coherent LLVM toolset",
        0,
        "count",
        "exact",
        bound=0,
        bound_source="no fact the declaration contradicts -- held at zero since the landed half of G9",
        slice_owner="G9",
    ),
    Metric(
        "alias.llvm.scope_facts.missing",
        "alias",
        "access pairs (one at least a store) through positions naming distinct resources that LLVM's scoped-noalias analysis ALONE does not prove NoAlias (opt aa-eval, 84 kernels); needs a coherent LLVM toolset",
        300,
        "count",
        "exact",
        bound=0,
        bound_source="every declared-disjoint pair proved by the scopes alone: present, and effective",
        slice_owner="G9",
    ),
    Metric(
        "alias.backends.disagree",
        "alias",
        "(kernel, fact family) pairs where clang's IR for the C kernel carries other noalias parameters, TBAA types, volatility or fences than the LLVM kernel; 84 kernels; needs a coherent LLVM toolset",
        132,
        "count",
        "exact",
        bound=0,
        bound_source="the two paths to LLVM carry the same declared facts",
        slice_owner="G9",
    ),
    Metric(
        "alias.llvm.caller_memory",
        "alias",
        "accesses to a C caller's own long data around the kernel LLVM inlined into it (clang -O2 caller, llvm-link, opt -O2) that contradict the declared facts: kept across a unique kernel whose TBAA proves them disjoint, or dropped across a barriered kernel whose fences order them; 84 callers; needs a coherent LLVM toolset",
        56,
        "count",
        "exact",
        bound=0,
        bound_source="the declared facts are what LLVM needs to remove the caller's redundant store and reload -- and what stops it at a barrier",
        slice_owner="G9",
    ),
    Metric(
        "alias.exec.failed",
        "alias",
        "(kernel, runner) pairs whose self-check fails with the declared aliasing bound: 14 kernels (every partition, the ops rotating) on LLVM AOT and JIT, the C and Q-fixed kernels and WASM under node -- each runner the host has; RED is the node harness computing `+` for every op",
        9,
        "count",
        "exact",
        bound=0,
        bound_source="every kernel runs correctly under the aliasing it declares, on every runner",
        slice_owner="G9",
    ),
    # --- SP-ENC: the StreamPack encoder's record layouts compiled (#790's recommendation 1).
    # The parity row counts failures over the fixed corpus of bcir/tests/encode_fixtures.py, which
    # the tests and tools/perf/check_encode.py grade the same way; the encoder before the slice is
    # kept verbatim as its reference, so the row is a guard that reads 0 on the parent as well.
    # The rows that move are the calls (exact) and the time (ratio, same process) of `encode`, and
    # the calls of the K_BCIR -> StreamPack chain from scratch, where `encode` was 74%.
    Metric(
        "streampack.encode.parity",
        "encode",
        "(item, rail) pairs where the compiled encoder's bytes or refusal (the exception's type and message) differ from the encoder before SP-ENC, kept verbatim: every honest pack of the corpus in every wire version and spelling, every field of every record forged one and two at a time, and every forged record through the record functions directly",
        0,
        "count",
        "exact",
        bound=0,
        bound_source="byte for byte and refusal for refusal the encoder it replaces -- the frozen ABI's reference codec does not move",
        slice_owner="SP-ENC",
    ),
    Metric(
        "streampack.encode.calls",
        "encode",
        "calls (cProfile total, builtins included) of one encode of the audit fixture's pipelined StreamPack at scale 8 (32,768 claims, 5.76 MB, wire v4), CPython 3.11",
        7543875,
        "calls",
        "exact",
        slice_owner="SP-ENC",
    ),
    Metric(
        "streampack.encode",
        "encode",
        "median time of encode / median time of the encoder before SP-ENC (kept verbatim), the audit fixture's pipelined StreamPack at scale 4, interleaved in one process",
        1.0,
        "x",
        "ratio",
        slice_owner="SP-ENC",
    ),
    Metric(
        "kbcir-streampack.full.calls",
        "encode",
        "calls (cProfile total, builtins included) of the K_BCIR -> StreamPack chain from scratch -- plan, hydrate, encode and the three verdicts -- on the audit fixture's one-claim delta module at scale 8 (32,768 claims), CPython 3.11",
        10110215,
        "calls",
        "exact",
        slice_owner="SP-ENC",
    ),
    # --- G10 (S5-B): escape analysis, indirect-call narrowing and the effect footprint made
    # sound. Every row is counted by bcir/tests/escape_fixtures.py::measure, which the tests and
    # tools/perf/check_escape.py grade the same way: the cfront corpus (173 units, 39 declared-
    # extent local arrays, 22 indirect sites), 24 generated units whose verdicts and targets are
    # known by construction, 386 forms, and a dynamic witness that runs every drivable pair of
    # functions in both orders. RED is the parent (8d3aab84, #792's merge) judged by the same
    # fixtures: it proved nothing private, narrowed no call, and called 119 diverging pairs
    # commuting (its footprint recorded every store as a read). The corpus counts are what is
    # NOT yet proved or narrowed, so each falls from the parent's total toward a proved floor --
    # and, the analysis being sound, a value below its floor is itself a defect (the gate holds
    # both sides).
    Metric(
        "escape.unproved",
        "escape",
        "declared-extent local arrays of the cfront corpus (the roadmap's escape candidates, 39) not proved to stay in their activation -- the candidates less those proved nonescaping",
        39,
        "count",
        "exact",
        bound=0,
        bound_source="every candidate: the analysis is sound (escape.verdict.mismatch, effects.commute.unsound), so each one it proves private is private, and it proves all 39",
        slice_owner="G10",
    ),
    Metric(
        "icall.unknown",
        "escape",
        "indirect call sites of the cfront corpus (22) not narrowed to known functions of the unit: each may reach code the unit cannot see",
        22,
        "count",
        "exact",
        bound=15,
        bound_source="the open world: 15 sites call a function pointer, or a member of a struct, that an exported function takes as a parameter -- another unit may pass anything, so no sound analysis narrows them",
        slice_owner="G10",
    ),
    Metric(
        "icall.unresolved",
        "escape",
        "indirect call sites of the cfront corpus (22) not resolved to exactly one function of the unit",
        22,
        "count",
        "exact",
        bound=16,
        bound_source="the 15 open-world sites, and cond_local_fp's pointer, which the value of `which` sets to either of two functions; use_local_fp's two sites resolve only under a flow-sensitive analysis",
        slice_owner="G10",
    ),
    Metric(
        "escape.verdict.mismatch",
        "escape",
        "named locals of 24 generated units (73) whose escape verdict is not the one they have by construction -- read in place (nonescaping), lent to a reader (lent), captured into a global (escaping); no verdict counts",
        73,
        "count",
        "exact",
        bound=0,
        bound_source="the verdict each local has by construction",
        slice_owner="G10",
    ),
    Metric(
        "icall.target.mismatch",
        "escape",
        "indirect call sites of 24 generated units (20) whose narrowed target set is not the one they have by construction; not narrowed counts",
        20,
        "count",
        "exact",
        bound=0,
        bound_source="the target set each site has by construction",
        slice_owner="G10",
    ),
    Metric(
        "effects.commute.unsound",
        "escape",
        "function pairs of the cfront corpus and the 24 generated units that CompileResult.commute calls commuting while the dynamic witness shows the two orders diverge (every drivable pair, 1,867 decided, both orders in separate processes over shared buffers); needs a C compiler",
        119,
        "count",
        "exact",
        bound=0,
        bound_source="a footprint that is sound: every divergence the witness can show is a conflict",
        slice_owner="G10",
    ),
    Metric(
        "effects.parity.mismatch",
        "escape",
        "units of the cfront corpus, the 24 generated units and the 4 forms units (200 compared; a unit the twin refuses counts, except the pinned preprocessor limits) whose bcir-cc --emit-effects differs from the oracle's footprint report; needs a C compiler and the checkout's runtime/c",
        24,
        "count",
        "exact",
        bound=0,
        bound_source="one analysis, both rails, byte for byte",
        slice_owner="G10",
    ),
    Metric(
        "escape.parity.mismatch",
        "escape",
        "units of the cfront corpus, the 24 generated units and the 4 forms units (200 compared) whose bcir-cc --emit-escape differs from the oracle's escape report (on the parent neither rail had one); needs a C compiler and the checkout's runtime/c",
        200,
        "count",
        "exact",
        bound=0,
        bound_source="one analysis, both rails, byte for byte",
        slice_owner="G10",
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
        host_dependent=True,  # a ratio of two COMPILED kernels: it measures this host, not the code
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
        host_dependent=True,  # a ratio of two COMPILED kernels: it measures this host, not the code
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
        host_dependent=True,  # a ratio of two COMPILED kernels: it measures this host, not the code
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
        host_dependent=True,  # a ratio of two COMPILED kernels: it measures this host, not the code
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


def _bounded_work(result) -> int:
    """A bounded body whose cost follows the plan's widths: what the harness executes as the
    plan, so the corpus holds real samples of a real body on this host."""
    return sum(max(1, step.candidate.width) for step in result.steps)


def measure_workload() -> dict[str, float]:
    """The G13 exact rows (S2-E) over the 12 corpus programs."""
    from bcir.examples import PROGRAMS
    from bcir.gem.dispatch import DispatchRequest, dispatch
    from bcir.gem.exact import certify_schedule
    from bcir.kbcir import TARGETS, optimize
    from bcir.kbcir.cost import Theta
    from bcir.kbcir.measured import MeasuredCorpus, measure_plans, scope_key
    from bcir.kbcir.portfolio import PolicyPortfolio, ReplayCertificate
    from bcir.kbcir.weights import ENERGY, PERF, THROUGHPUT
    from bcir.kbcir.workload import workload_for

    h = TARGETS["x86_avx2"]
    theta = Theta.cool()
    collisions = subset = unavailable = 0
    for _name, build in sorted(PROGRAMS.items()):
        module = build()
        workloads = (
            workload_for(module, service_level="latency", latency_ns=1_000_000),
            workload_for(module, batch=8, service_level="throughput", throughput_per_s=1_000),
            workload_for(module),
        )
        result = optimize(module, h, theta, PERF)
        scopes = [
            certify_schedule(module, result, h, theta, PERF, requested="TMSAO-4", workload=w).scope
            for w in workloads
        ]
        collisions += sum(scopes[i] == scopes[j] for i in range(3) for j in range(i + 1, 3))
        # the corpus: three logged episodes, every policy measured on a bounded body
        corpus = MeasuredCorpus()
        for episode in (Theta.cool(), Theta.hot(), Theta.mem_bound()):
            for entry in measure_plans(
                module,
                h,
                episode,
                (PERF, ENERGY, THROUGHPUT),
                _bounded_work,
                workload=workloads[0],
                source_commit="5b45fdca",
                repeats=2,
            ):
                corpus.append(entry)
        program, target, digest = scope_key(module, h, workloads[0])
        logged = len(corpus.episodes(program, target, digest))
        for candidate in (ENERGY, THROUGHPUT):
            portfolio = PolicyPortfolio.default()
            partial = ReplayCertificate(
                candidate.name, PERF.name, 1, 0, corpus=corpus.head, logged=logged
            )
            try:
                portfolio.promote(PERF.name, candidate, partial)
                subset += 1
            except ValueError:
                pass
        evidence = len(corpus.lookup(program, target, digest))
        size = sum(len(phase.claims) for phase in module.phases)
        decision = dispatch(
            DispatchRequest("selection", size, "TMSAO-3", 1, workload=digest, evidence=evidence)
        )
        unavailable += decision.rail != "measured"
    return {
        "scope.workload.collisions": float(collisions),
        "replay.subset.admitted": float(subset),
        "dispatch.measured.unavailable": float(unavailable),
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


def measure_control() -> dict[str, float]:
    """The G14 rows (S3-A): the control plane as bytes, counted failures over fixed corpora on
    both rails (bcir/tests/control_fixtures.py::measure, which the tests and
    tools/c/check_runtime.sh grade the same way). Every row needs the C twin: without a C
    compiler the group is NOT-MEASURED rather than estimated from one rail."""
    import shutil
    import tempfile

    from bcir.tests.control_fixtures import build_harness, measure

    tmp = tempfile.mkdtemp(prefix="bcir-control-")
    try:
        exe = build_harness(tmp)
        return measure(exe, tmp) if exe is not None else {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def measure_ring() -> dict[str, float]:
    """The G15 rows (S3-B): the live ring, the envelope and the generated table, counted failures
    over fixed corpora on both rails (bcir/tests/ring_fixtures.py::measure, which the tests and
    tools/c/check_runtime.sh grade the same way), and the throughput ratio. Every row needs the C
    twin: without a C compiler the group is NOT-MEASURED rather than estimated from one rail."""
    import shutil
    import tempfile

    from bcir.tests.ring_fixtures import build_harness, measure, measure_throughput

    tmp = tempfile.mkdtemp(prefix="bcir-ring-")
    try:
        exe = build_harness(tmp)
        if exe is None:
            return {}
        out = measure(exe, tmp)
        out.update(measure_throughput(exe))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def measure_handoff() -> dict[str, float]:
    """The G16 rows (S3-C): the data-plane hand-off, counted failures over fixed corpora on the
    oracle, the C twin and the C++ seam (bcir/tests/handoff_fixtures.py::measure, which the tests,
    tools/c/check_runtime.sh and tools/cpp/check_handoff.sh grade the same way). The rows need
    both native rails: without the compilers the group is NOT-MEASURED, never one rail's guess."""
    import shutil
    import tempfile

    from bcir.tests.handoff_fixtures import (
        build_cpp_harness,
        build_harness,
        measure,
        measure_overhead,
    )

    tmp = tempfile.mkdtemp(prefix="bcir-handoff-")
    try:
        exe, cpp = build_harness(tmp), build_cpp_harness(tmp)
        if exe is None or cpp is None:
            return {}
        out = measure(exe, cpp, tmp)
        out.update(measure_overhead(cpp, tmp))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def measure_kplan() -> dict[str, float]:
    """The G17 rows (S4-A): the compact planner against the pre-G17 planner and the native planner
    against both (bcir/tests/planner_fixtures.py::measure, which the tests and
    tools/c/check_runtime.sh grade the same way), the planner's call count and the native plan's
    time. The exact parity rows need the C twin: without a C compiler they are NOT-MEASURED, never
    estimated from one rail; the call count needs nothing but the interpreter."""
    import shutil
    import tempfile

    from bcir.kbcir import realize
    from bcir.tests.planner_fixtures import build_harness, call_count, measure, native_ms

    out = {"planner.calls": float(call_count(realize.optimize))}
    tmp = tempfile.mkdtemp(prefix="bcir-kplan-")
    try:
        exe = build_harness(tmp)
        if exe is not None:
            out.update(measure(exe, tmp))
            out["planner.native.scale4"] = native_ms(exe, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def measure_delta() -> dict[str, float]:
    """The G18 rows (S4-B): the chain advanced by declared deltas against the chain from scratch
    (bcir/tests/delta_fixtures.py::measure, which the tests and tools/perf/check_delta.py grade the
    same way), the time ratio of a one-claim delta at scale 4 and its call count at scale 8. Pure
    Python: every row is measured wherever the interpreter runs."""
    from bcir.tests.delta_fixtures import delta_calls, delta_ratio, measure

    out = measure()
    out["kbcir-streampack.delta"] = delta_ratio()
    out["kbcir-streampack.delta.calls"] = float(delta_calls()[0])
    return out


def measure_alias() -> dict[str, float]:
    """The G9 rows (S5-A): the declared alias facts on every emitter of the elementwise kernel,
    counted failures over fixed corpora (bcir/tests/alias_fixtures.py::measure, which the tests
    and tools/perf/check_alias.py grade the same way). The text rows need only the interpreter;
    the rows LLVM judges need a coherent clang/llvm-link/opt and are NOT-MEASURED without one,
    never zero by default."""
    from bcir.tests.alias_fixtures import measure

    return measure(llvm=True)


def measure_encode() -> dict[str, float]:
    """The SP-ENC rows: the compiled StreamPack encoder held to the encoder it replaces
    (bcir/tests/encode_fixtures.py::measure, which the tests and tools/perf/check_encode.py grade
    the same way), its calls at scale 8 and its time against the reference at scale 4, and the
    calls of the chain from scratch at scale 8. Pure Python: measured wherever the interpreter
    runs."""
    from bcir.tests.delta_fixtures import full_calls
    from bcir.tests.encode_fixtures import encode_calls, encode_ratio, measure

    out = measure()
    out["streampack.encode.calls"] = float(encode_calls()[0])
    out["streampack.encode"] = encode_ratio()
    out["kbcir-streampack.full.calls"] = float(full_calls())
    return out


def measure_escape() -> dict[str, float]:
    """The G10 rows (S5-B): the escape analysis, indirect-call narrowing and effect footprint of
    the cfront frontend (bcir/tests/escape_fixtures.py::measure, which the tests and
    tools/perf/check_escape.py grade the same way). The corpus and truth rows need only the
    interpreter; the witness and parity rows need a C compiler (and the parity rows the
    checkout's runtime/c) and are NOT-MEASURED without one, never zero by default."""
    from bcir.tests.escape_fixtures import measure

    return measure()


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
    "workload": measure_workload,
    "native": measure_native,
    "exact": measure_exact,
    "verifier": measure_verifier,
    "digest": measure_digest,
    "plan": measure_plan,
    "control": measure_control,
    "ring": measure_ring,
    "handoff": measure_handoff,
    "kplan": measure_kplan,
    "delta": measure_delta,
    "alias": measure_alias,
    "encode": measure_encode,
    "escape": measure_escape,
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
    if unit in ("bytes", "count", "calls"):
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
            "INDICATIVE rows are measured off the baseline host and are reported, not "
            "graded: wall-clock rows, and the host-dependent ratios whose two sides are "
            "separately compiled kernels rather than one process (the native.* band).",
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
        help="limit measurement to a group (audit, planner, scheduler, dispatch, regions, workload, native, exact, verifier, digest, plan, control, ring, handoff, kplan, delta, memory)",
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
