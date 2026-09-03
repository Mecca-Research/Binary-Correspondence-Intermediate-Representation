# BCIR TMSAO core architecture and operational performance audit
<!-- allow-law-ranges -->

**Audit date:** 2026-07-29
**Repository revision:** `7d1b8d7a58bf19823b6d62320293460b00a7aa33` (`origin/main`, post-PR #683)
**Scope:** research, source inspection, bounded x86 validation, operational/data-structure
performance, exact small-instance differentials, and comparison with current compiler and
systems research.
**Change policy:** no BCIR source or documentation was changed during this audit.

## 1. Executive answer

BCIR should **not replace GEM**. GEM is a sound and unusually useful semantic spine for
recording claims, dependencies, resource identity, phases, hazards, legality, provenance,
and target-independent execution intent. Keeping a graph as the conservative fallback is
also reasonable: the graph is a compile-time truth, while StreamPack/native lowering can
erase graph traversal from the hot path.

BCIR has **not demonstrated Theoretical Maximum System Architecture Optimization
(TMSAO)**, however, and tropical min-plus is **not a universal master algorithm**.
The current tropical rail is exact for the problem it actually constructs: selecting a path
through a layered candidate DAG under additive, integer/Q8 costs, with an RCSP/Pareto
variant for additive resource caps. That is an excellent deterministic kernel. It does not
jointly solve arbitrary scheduling, placement, fusion, memory allocation, communication,
or data-dependent control. Those interactions make the true optimization problem
non-additive and commonly NP-hard.

The most important result of this audit is therefore architectural:

> **Keep GEM as the proof-carrying semantic graph; evolve K_BCIR from “one tropical
> optimizer” into a verified solver portfolio over a hierarchical, region-aware decision
> model. Bind the chosen schedule, memory layout, transfers, and execution artifact into
> one certificate.**

This is an optimization of the core, not a demolition of it.

The implementation is already strong in deterministic legality, reproducible costs,
candidate filtering, bounded RCSP/Pareto selection, StreamPack hydration, C/LLVM fallback,
and parity testing. The main gaps are:

1. `price_scheduled`, `schedule_eft`, and token execution do not use the same scheduling
   model.
2. Candidate choice, schedule, and static memory are optimized separately.
3. The schedule-aware candidate sweep is heuristic and scales quadratically.
4. The static allocator is deterministic first-fit, not an optimal layout solver.
5. Static liveness is phase-based, while token scheduling can overlap phases.
6. Cost calibration lacks bare-metal PMU/energy evidence and one native microbenchmark has
   a stride/provenance validity defect.
7. The MLIR rail lowers selected GEM matmul work to concrete `scf.for` loops, but it does
   not yet expose target schedules through MLIR Transform IR or model regular regions with
   Affine IR.

These are tractable extensions. They do not invalidate GEM, the R-laws, or StreamPack.

## 2. What “TMSAO” can and cannot mean

There is no workload-independent program that is simultaneously optimal for every CMOS
machine, input distribution, thermal state, reliability target, and objective. Even a
single target admits conflicting objectives: latency, throughput, energy, temperature,
memory, accuracy, compilation time, code size, and lifetime wear can disagree.

For BCIR, a defensible TMSAO claim should mean:

- the legal search space is explicitly bounded and complete for the declared abstraction;
- all hard constraints are checked independently of learned or measured cost;
- the objective corresponds to the schedule and layout that will actually execute;
- exact rails report proof of optimality;
- heuristic rails report a lower bound, incumbent, optimality gap, budget, and provenance;
- measured terms include target, toolchain, workload, environment, samples, uncertainty,
  and generation;
- the emitted artifact is cryptographically bound to the selected realization, schedule,
  memory map, transfers, and target generation;
- runtime adaptation occurs only through verified generation changes and rollback.

Under that definition, “theoretical maximum” is always **relative to a declared model and
search boundary**. BCIR can prove “optimal among these legal candidates under this target
model,” but cannot honestly prove “globally optimal software for the entire CMOS universe.”

## 3. Audit method and limitations

### 3.1 Host and resource controls

The audit ran in WSL on:

- AMD Ryzen 5 2600, 6 cores / 12 threads, AVX2, 8 MiB L3;
- 7.7 GiB RAM and 2 GiB swap;
- NVIDIA GTX 1660 SUPER, deliberately unused;
- Python 3.10.12;
- GCC 11.4;
- Clang/LLVM/MLIR 22.1.8 from the installed apt.llvm.org snapshot.

Every local workload was bounded:

- at most two test workers;
- one OpenMP/BLAS/Rayon numerical thread;
- serial targeted probes pinned to one logical CPU where practical;
- no GPU work, large-model inference/training, QEMU, ARM emulation, unbounded fuzzing, or
  sustained stress loops;
- Linux `/tmp` worktree and temporary files for the targeted probes.

WSL does not expose the PMU, RAPL, or a trustworthy bare-metal thermal/power envelope here.
Consequently, timings in this report establish implementation behavior and reveal major
algorithmic trends; they are **not silicon performance certificates**.

### 3.2 Validation baseline

The default `/usr/bin/clang` is Clang 14. It produced 64 failures because that compiler
does not accept the requested C23 spelling. This was a tool-version limitation, not a BCIR
semantic regression.

With `/usr/lib/llvm-22/bin` first in `PATH`, the complete bounded silicon-degrade inventory
reported:

```text
2,839 passed
0 failed
elapsed 7:58.78
maximum RSS 1,197,820 KiB
swap used 0
```

The worktree remained clean and `git diff --check` passed. The validation log is
`/tmp/bcir-clang22-silicon-degrade.log`.

### 3.3 Performance methods

The audit combined:

- BCIR's deterministic 13-group performance audit at scale 1, 2, and 4, five repetitions
  each;
- C/Clang native structural comparisons;
- `cProfile` on the scale-2 audit;
- exhaustive exact scheduling over all 1,716 nondecreasing six-job duration multisets with
  values 1–8;
- exact backtracking for 500 bounded seven-resource static layouts;
- exhaustive candidate assignments for a bounded schedule-aware K_BCIR fixture;
- direct consistency checks between scheduled pricing, EFT execution, phase liveness, and
  token-DAG execution;
- source-backed complexity analysis.

Timing is deliberately not a CI pass/fail gate in the deterministic audit. Result digests
and correctness invariants are the gate.

## 4. Current architecture: what is genuinely strong

### 4.1 GEM is a good semantic and verification substrate

GEM captures facts that ordinary compiler DAGs often scatter across analyses:

- explicit resources and domains;
- read/write claims and hazards;
- phase and token dependencies;
- lane and stride intent;
- target and telemetry generations;
- cost candidates and certificates;
- deterministic verifier laws.

A DAG fallback is appropriate when BCIR cannot prove stronger structure. It preserves
ordering and ownership information without forcing speculative lowering. The important
performance rule is that the DAG should guide compilation; the resulting StreamPack or
native artifact should be flat, compact, and directly dispatchable.

### 4.2 K_BCIR's exact boundary is useful and implementable

`bcir/kbcir/realize.py` constructs candidate columns and runs a min-plus shortest path.
`bcir/kbcir/rcsp.py` keeps non-dominated labels over scalar score and selected dimensions
of the 12-axis cost vector:

```text
compute, memory, fabric, sync, compile, thermal,
power, reliability, security, accuracy, contention, verification
```

Within that layered additive graph, the result is deterministic and exact. This is a
valuable building block because it is:

- easy to independently verify;
- stable under integer/Q8 arithmetic;
- compatible with hard caps and Pareto enumeration;
- safe to use as a fast exact rail inside a larger optimizer.

### 4.3 StreamPack and flat planners scale well

The targeted flat K_BCIR/StreamPack probe used independent claims with two candidates each:

| Claims | Candidate optimization | Hydration | Plan verification | Pack verification |
|---:|---:|---:|---:|---:|
| 128 | 6.76 ms | 0.99 ms | 0.15 ms | 0.26 ms |
| 256 | 13.54 ms | 2.11 ms | 0.29 ms | 0.56 ms |
| 512 | 30.06 ms | 3.90 ms | 0.54 ms | 1.04 ms |
| 1,024 | 51.65 ms | 8.58 ms | 1.11 ms | 2.39 ms |
| 2,048 | 106.07 ms | 19.04 ms | 2.31 ms | 5.25 ms |

The path is approximately linear. At the largest point it processed roughly 19,000
candidate-selected claims/s in Python and hydrated more than 100,000 claims/s. This is
good evidence that the basic graph-to-flat-artifact path is not suffering an asymptotic
collapse.

### 4.4 Structural transformations produce real native wins

Three bounded Clang 22 trend runs showed:

| Native comparison | Observed ratio |
|---|---:|
| Dense streaming BCIR vs equivalent compiler loop | 0.98–1.01× |
| Dense L1 | 1.00× |
| Gather avoidance | 5.58–6.05× |
| Blocked reduction | 11.68–11.72× |
| Direct strided access vs gather form | 1.27–1.33× |

Matching Clang on equivalent dense loops is the expected result. The large wins occur when
BCIR preserves enough structure to avoid a gather or select a blocked reduction. This
supports the project's central engineering intuition: BCIR should find structural
realizations that a late generic backend cannot infer, then let LLVM/GCC perform ordinary
instruction selection and register allocation.

It does **not** show that graph representation itself makes arbitrary algorithms faster.
The win comes from a proved transformation into a better data path.

## 5. Full operational and data-structure performance

### 5.1 Deterministic audit results

Median time over five repetitions:

| Group | Scale 1 | Scale 2 | Scale 4 | Scale-4 / scale-1 |
|---|---:|---:|---:|---:|
| Iterative phase DAG (512→2,048 claims) | 5.99 ms | 14.59 ms | 34.62 ms | 5.78× |
| Mixed wave/token/EFT (512→2,048 claims) | 11.64 ms | 21.75 ms | 48.05 ms | 4.13× |
| K_BCIR→StreamPack (64→4,096 claims) | 3.84 ms | 30.81 ms | 275.22 ms | 71.67× |
| Static lifetime planner (512→2,048 resources) | 110.53 ms | 231.31 ms | 566.35 ms | 5.12× |
| Bounded telemetry ring (512→2,048 entries) | 10.59 ms | 21.34 ms | 42.68 ms | 4.03× |
| Q8/Q4 blocks (4,096→16,384 values) | 13.73 ms | 26.22 ms | 54.20 ms | 3.95× |
| K-means/KNN/scaler/embedding (256→1,024 samples) | 15.68 ms | 35.82 ms | 63.47 ms | 4.05× |
| Tiled matmul (24→96 dimension) | 8.39 ms | 50.83 ms | 332.27 ms | 39.60× |
| OLS/PCA (64→256 rows) | 2.63 ms | 5.91 ms | 9.20 ms | 3.50× |
| Transformer block (8→32 sequence) | 3.53 ms | 7.46 ms | 18.10 ms | 5.12× |
| LSTM/GRU (16→64 steps) | 2.70 ms | 5.37 ms | 10.89 ms | 4.04× |
| Autodiff/Adam (32→128 examples plus scaled epochs) | 16.75 ms | 65.08 ms | 267.00 ms | 15.94× |
| Bounded MCTS (64→256 simulations) | 0.42 ms | 0.81 ms | 1.86 ms | 4.41× |

Interpretation:

- Telemetry, quantization, recurrent layers, and the composite unsupervised path are close
  to linear in the work represented by the fixture.
- The graph and scheduler paths are slightly superlinear but remain operational at
  thousands of claims in the Python oracle.
- The K_BCIR→StreamPack fixture increases claims by 64× from scale 1 to scale 4; a 71.7×
  time increase is close to linear in claims.
- Matmul's approximately cubic growth is expected when dimension grows 4×.
- The transformer result reflects attention's sequence interaction plus Python overhead.
- The training fixture scales both examples and training work, so its near-quadratic
  result is expected from the fixture definition; it is not evidence of a production
  training kernel.
- These ML measurements validate semantics and gross complexity. Python oracle throughput
  is not a claim about optimized C/CUDA model execution.

### 5.2 Profiled bottleneck

The scale-2 profile executed about 4.8 million Python calls. The largest cumulative costs
were:

- static memory planning and verification: about 1.14 s under instrumentation;
- module digest/hash work: about 0.65 s;
- recursive canonical flattening: more than one million calls;
- live-alias verification;
- then training, matmul, and K_BCIR/StreamPack work.

A focused static-memory probe separated the costs:

| Resources | Claims | Phase liveness | Module digest | Plan (includes verify) | External verify |
|---:|---:|---:|---:|---:|---:|
| 256 | 492 | 0.41 ms | 13.15 ms | 41.47 ms | 16.37 ms |
| 512 | 989 | 0.82 ms | 21.30 ms | 70.40 ms | 34.96 ms |
| 1,024 | 1,984 | 1.86 ms | 45.12 ms | 146.12 ms | 72.96 ms |
| 2,048 | 3,972 | 3.35 ms | 88.05 ms | 301.02 ms | 157.88 ms |

`plan_static_memory` hashes the module and then invokes the verifier; the verifier hashes
again. A client that independently verifies hashes a third time. Independent verification
must remain, but immutable module identity should be computed once from an iterative
canonical stream and reused through an identity-bound API. The verifier should still
recompute when identity is not trusted or when crossing a trust boundary.

## 6. Exact differentials: where the current optimizer is not optimal

### 6.1 EFT scheduling

`schedule_eft` accurately calls itself “HEFT-lite”: per phase it uses LPT priority,
earliest-finish placement, locality tie-breaking, and a bandwidth-knee clamp.

Against an exact branch-and-bound scheduler over every nondecreasing six-job duration
multiset with values 1–8:

| Domains | Instances | Suboptimal | Mean BCIR/optimal | Worst BCIR/optimal |
|---:|---:|---:|---:|---:|
| 2 | 1,716 | 190 (11.07%) | 1.0078 | 17/15 = 1.1333 |
| 3 | 1,716 | 18 (1.05%) | 1.0013 | 7/6 = 1.1667 |

The heuristic is generally good on this bounded independent-job corpus, but it cannot be
called globally exact. This is normal: heterogeneous precedence scheduling is hard, and
HEFT was designed as a low-complexity heuristic, not a universal proof procedure
([original HEFT paper](https://doi.org/10.1109/71.993206)).

### 6.2 Scheduled pricing is not operational EFT pricing

`price_scheduled`:

- groups claims into fixed waves;
- assigns wave members round-robin to affinity bins;
- prices each bin as a serial chain.

`schedule_eft`:

- orders ready work by duration;
- places it on the earliest-finishing domain;
- includes locality and bandwidth-knee behavior.

On the same four independent selected claims with durations `[25600, 100, 25600, 100]`
and two domains:

```text
price_scheduled makespan = 51,200
schedule_eft makespan    = 25,700
serial sum               = 51,400
```

The nearly 2× disagreement is not timing noise; it is a different algorithm. Therefore,
the LangRef's `M(π,Θ)` cannot simultaneously denote both implementations.

The fix is conceptual before it is algorithmic: a selected plan must carry one canonical
schedule artifact, and the objective, verifier, memory planner, and executor must all read
that artifact.

### 6.3 Schedule-aware candidate selection

`optimize_scheduled` begins with the serial min-plus optimum and performs one coordinate
sweep. Each trial copies the assignment and recomputes the whole makespan.

Measured scaling:

| Claims | Serial `optimize` | `optimize_scheduled` | Slowdown |
|---:|---:|---:|---:|
| 32 | 1.66 ms | 11.12 ms | 6.7× |
| 64 | 3.19 ms | 34.78 ms | 10.9× |
| 128 | 6.15 ms | 121.74 ms | 19.8× |
| 256 | 12.19 ms | 435.73 ms | 35.8× |
| 512 | 24.61 ms | 1,703.66 ms | 69.2× |

This is effectively quadratic for a constant number of candidates.

An exhaustive 4-claim, 3-candidate fixture found:

```text
one-sweep plan: all vec16                         makespan 55,552
exact plan:     vec16, vec16, vec8, vec8          makespan 55,168
ratio: 1.00696
```

The immediate engineering fix is incremental delta pricing: cache per-phase/bin
contributions and update only the affected chain. That should make one sweep close to
`O(claims × candidates)`. A bounded exact or CP-SAT rail should handle small/high-value
graphs and provide an optimality gap for the heuristic rail. The official
[OR-Tools scheduling documentation](https://developers.google.com/optimization/scheduling)
demonstrates the relevant CP-SAT job-shop formulation; BCIR does not need to make that
hosted solver part of legality.

### 6.4 Static memory allocation

The planner computes phase-based lifetimes and uses aligned first-fit. It is deterministic
and safe under its own lifetime model, but not space-optimal.

Across 500 deterministic random seven-resource fixtures, exact integer backtracking found:

- first-fit was suboptimal in 193 cases (38.6%);
- mean absolute gap was 0.994 size units;
- worst fixture used 21 units versus a proved 13-unit optimum, a 1.615× ratio.

The same fixture through the real BCIR planner with 64-byte alignment used 1,344 bytes;
the exact layout used 832 bytes.

BCIR should retain first-fit as the predictable fast path, then invoke a bounded exact
layout solver when:

- the fast layout violates capacity;
- peak pressure exceeds a threshold;
- the artifact is high-value or repeatedly deployed;
- a user requests a bounded optimality proof.

Every result should record a concurrent-live lower bound, achieved extent, and gap.

### 6.5 Phase liveness and token scheduling can disagree

`live_intervals` uses topological phase positions. `execute_tokens` explicitly allows
independent claims in later phases to overlap earlier phases.

A two-resource test demonstrates the consequence:

- resource A is used only in phase 0;
- resource B is used only in phase 1;
- phase-liveness aliases both at offset 0;
- barrier scheduling runs them sequentially;
- token scheduling legally overlaps the independent claims on different domains.

If the static offsets were composed with that token schedule, the two live resources
would alias.

The current repository does not appear to feed this static memory map into token execution
or StreamPack offsets, so this audit did **not** establish a deployed memory-corruption
path. It establishes a hard future integration condition:

> Static memory plans must be computed from the final schedule's time intervals, or the
> verifier must prove that the schedule refines the phase-lifetime order used by the plan.

### 6.6 The LangRef overstates the exact rail

The LangRef says the constrained series-parallel equation is solved exactly by RCSP label
dominance. Source inspection shows that RCSP labels minimize additive `score` and additive
resources over the layered candidate DAG. They do not incorporate the later wave/EFT
schedule, first-fit memory layout, or joint placement.

The exact statement should be:

- RCSP is exact over the declared layered candidate graph for additive score and caps;
- `price_scheduled` is a deterministic derived price under its fixed wave/bin model;
- `optimize_scheduled` is a one-sweep heuristic;
- `schedule_eft` is a HEFT-lite execution scheduler;
- static memory is deterministic first-fit;
- the composed global problem is not presently solved exactly.

This is a documentation precision gap, not a reason to discard the implementation.

## 7. Native measurement validity finding

The native microbenchmark defaults to:

```c
n = 1 << 22;
stride = 16;
index = (k * stride) % n;
```

Because `gcd(2^22, 16) = 16`, the strided pass visits only `n/16` unique elements and
repeats that cycle 16 times. The nominal 32 MiB double buffer therefore has a 2 MiB unique
working set for that regime. This contradicts the intended all-elements/cache-defeating
interpretation.

The program also emits the literal provenance string `native microbench (bare-metal)` even
under WSL.

A bounded comparison between the default size and the coprime size 4,194,303 confirmed
that the issue is methodological; the observed ratios moved modestly on this host, but the
default still does not measure what its provenance claims.

Required correction:

- choose a stride coprime to `n`, or generate a true full-cycle permutation;
- record unique elements and working-set bytes;
- emit raw sample timings as well as Q8 ratios;
- report robust intervals and outlier policy;
- attest OS/virtualization, CPU, compiler, frequency policy, PMU/RAPL availability, and
  target generation;
- reserve “bare-metal” for an environment that proves it.

Compiler-integrated profiling is an appropriate future direction. KPerfIR reports a
multi-level compiler-centric GPU profiling IR with 8.2% overhead and 2% relative error in
its evaluation ([OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/guan)).
BCIR should borrow the separation of measurement IR from legality, not its GPU-specific
implementation wholesale.

## 8. Is a DAG the right fallback?

Yes—provided BCIR distinguishes **semantic fallback** from **optimization structure**.

A plain DAG is excellent for:

- conservative dependency and effect ordering;
- topological legality checks;
- deterministic graph transformations;
- identifying parallel readiness;
- serialization into a flat execution plan.

A plain DAG is insufficient or inefficient for:

- loops without expansion;
- affine iteration spaces and dependence distances;
- fixed-rate streams;
- async event/state-machine behavior;
- n-ary fusion and shared-data relationships;
- alias sets and memory interference;
- recursive or data-dependent control;
- hardware hierarchy below an operator.

The answer is not to abandon the DAG. It is to give DAG nodes **typed regions** with
stronger local mathematics:

- affine/polyhedral regions for regular loops and memory maps;
- SDF/CSDF regions for fixed-rate streaming;
- timed event/max-plus regions for latency and throughput;
- tensor index-map regions for virtual movement and fusion;
- state-machine/Petri-style regions for asynchronous events;
- opaque claim DAGs as the universal conservative fallback.

MLIR already demonstrates why this separation works. Its
[Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/) represents fine-grained
transformation control separately from payload IR, while the
[Affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/) provides restricted
polyhedral structures, DMA, prefetch, parallel loops, and analyzable index maps. BCIR can
interoperate with those abstractions without making MLIR the source of R-law legality.

For fixed-rate channels, synchronous dataflow can determine valid periodic schedules at
compile time and remove runtime scheduling overhead
([Lee and Messerschmitt, 1987](https://ptolemy.berkeley.edu/publications/papers/87/staticscheduling/)).
For timing regions, min-plus alone is incomplete: timed event graphs naturally use
max-plus for lower/earliest timing and min-plus for upper/latest constraints
([event-graph analysis](https://arxiv.org/abs/2003.04703)).

## 9. Is tropical min-plus the master algorithm?

No. It is one of the most useful exact kernels in the portfolio.

Min-plus is the right algebra when:

- alternatives combine by minimum;
- serial path costs add;
- edge costs are fixed for the represented state;
- all important interactions have been encoded in state or edge costs.

It is not sufficient when:

- parallel completion combines by maximum;
- an assignment changes contention for other assignments;
- memory placement changes communication cost;
- fusion changes the graph and kernel candidate set;
- probability or expected behavior matters;
- the objective is a Pareto set rather than a scalar;
- feasibility involves cumulative or disjunctive resources;
- control is cyclic or input-dependent.

The GraphBLAS standard is useful evidence: it defines graph computation over many
user-selectable semirings, including min-plus, max-plus, plus-times, and Boolean forms;
no one semiring subsumes every graph algorithm
([GraphBLAS C API 2.1](https://graphblas.org/docs/GraphBLAS_API_C_v2.1.0.pdf)).

BCIR should therefore keep “K_BCIR” as the optimization umbrella and name the current
min-plus path as one solver rail. A semiring registry can support:

- min-plus for additive candidate paths;
- max-plus for critical-path/earliest-finish timing;
- min-max for bottleneck objectives;
- Boolean reachability and legality summaries;
- lexicographic and Pareto orders for non-substitutable objectives;
- probabilistic/Markov models only where probabilities are measured and declared.

The phrase “tropical geometry” should be used only when BCIR actually operates on tropical
polytopes, varieties, or geometric constructions. Most current code is tropical algebra
and dynamic programming. Accurate terminology makes the strongest claims more credible.

## 10. What current state-of-the-art systems add

No surveyed system replaces all of BCIR. Several solve narrower optimization subproblems
more deeply and should inform GEM+.

### 10.1 Algebra plus schedules plus hardware hierarchy

[Mirage](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) uses multilevel
µGraphs spanning kernel, thread-block, and thread levels. It searches coordinated algebraic
and schedule transformations and uses abstraction-based pruning plus probabilistic
equivalence checking. Its core lesson for BCIR is that a single operator-level graph misses
cross-level transformations.

[Nautilus](https://arxiv.org/abs/2604.14825) jointly applies high-level expression
rewrites and tile scheduling and reports automatically discovering
FlashAttention-3-like kernels. This reinforces the need for a structured tensor region
and a search space that can change both algorithm and schedule.

Equality saturation remains valuable for enumerating equivalent programs without
destructive phase ordering. The [egg paper](https://arxiv.org/abs/2004.03082) demonstrates
an efficient e-graph foundation. BCIR already has a bounded e-graph rail; the missing work
is a richer, typed rewrite vocabulary and a cost extractor coupled to the final schedule
and memory model.

### 10.2 Joint scheduling and memory

[COSMA](https://arxiv.org/abs/2311.18246) formulates operator schedule, memory allocation,
and tensor replacement together and reports optimal solutions for its bounded DNN
accelerator formulation. That is direct evidence that BCIR's current decomposition can
miss global optima.

[Twill](https://www.usenix.org/conference/osdi26/presentation/soi) jointly formulates
software pipelining and warp specialization for a restricted class and uses constraint
solvers to prove optimal schedules, including expert FlashAttention schedules on Hopper
and Blackwell. BCIR should use the same principle: exact solvers are practical and powerful
inside a declared region, not across an unbounded universal graph.

[Welder](https://www.usenix.org/system/files/osdi23-shi.pdf) uses a tile graph to coordinate
fusion and memory access. [Timeloop](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation),
[MAESTRO](https://arxiv.org/abs/1805.02566), and
[Accelergy](https://accelergy.mit.edu/paper.pdf) demonstrate target-aware mapping,
data-centric reuse analysis, occupancy, action counts, and energy projection. These are
better models for regular tensor regions than pricing every operator as an independent
flat claim.

### 10.3 Data movement as a first-class transformation

The Hong-Kung red/blue pebble model provides machine-independent I/O lower bounds
([original paper](https://doi.org/10.1145/800076.802486)). BCIR should put such lower
bounds beside measured costs so a plan can report how far it is from the theoretical
communication floor.

[Marvel](https://arxiv.org/abs/2002.07752) separates off-chip and on-chip mapping because
off-chip movement is vastly more expensive.
[VTC](https://www.usenix.org/conference/osdi26/presentation/hu-muyan) represents movement
through virtual-tensor index maps and reports up to 1.93× performance and 60% inference
memory savings by eliminating unnecessary physical movement.

For BCIR, this argues for:

- index-map/virtual-view resources;
- movement-elimination proofs;
- physical materialization only at verified escape points;
- schedule-aware lifetime and rematerialization decisions;
- I/O lower bounds in plan certificates.

### 10.4 Persistent flat execution

[MPK](https://www.usenix.org/conference/osdi26/presentation/cheng) lowers tensor programs
to SM-level task graphs executed inside a persistent mega-kernel and reports up to 1.7×
lower end-to-end inference latency. This is compatible with BCIR's goal: use graphs to
derive a schedule, then execute a flattened low-overhead stream. BCIR should treat
persistent-kernel execution as a target realization, not make every target use it.

### 10.5 Search and autotuning

[Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) combines hierarchical
search, evolutionary exploration, and a learned cost model.
[MetaSchedule](https://arxiv.org/abs/2205.13603) exposes composable probabilistic search
spaces. [TensorIR](https://arxiv.org/abs/2207.04296) supplies tensorized loop abstractions.
[TASO](https://www.cs.cmu.edu/~zhihaoj2/papers/sosp19.pdf) combines verified graph
substitutions with cost search.

BCIR's stronger safety boundary should remain:

- learned models propose or rank candidates;
- legality and equivalence are deterministic;
- measurements update only generation-tagged cost artifacts;
- promotion requires replay, bounds, and rollback.

## 11. Proposed GEM+ / K_BCIR architecture

```text
Semantic source / model / driver schema
                  |
                  v
      R-law proof-carrying GEM graph
      (resources, effects, hazards, tokens)
                  |
                  v
       Typed hierarchical regions
  affine | tensor-map | SDF | timed-event |
  state-machine | opaque conservative DAG
                  |
                  v
     Candidate/equality decision graph
  algorithm, fusion, tiling, layout, movement,
  recompute, precision, backend, persistence
                  |
                  v
         Joint execution decision IR
  schedule + placement + memory + transfers +
       synchronization + thermal policy
                  |
                  v
            Solver portfolio
  min-plus | RCSP/Pareto | max-plus | DP |
  bounded exact/CP-SAT | heuristic | measured
                  |
                  v
       Independently verified plan bundle
  lower bounds, incumbent, gap, provenance,
  target generation, schedule, static layout
                  |
                  v
        StreamPack / BCAB / native object
        (flat execution; no graph hot path)
```

### 11.1 Preserve two truths

BCIR's best design principle is the separation of:

1. **semantic/legality truth**, which must not depend on performance telemetry; and
2. **realization/cost truth**, which may use measured and learned evidence.

GEM+ must preserve this. A solver timeout, bad model, stale table, or missing PMU must
never make an illegal plan legal.

### 11.2 Introduce one canonical execution plan

The content-addressed plan should bind:

- semantic module digest and R-law result;
- target, telemetry, calibration, and policy generations;
- candidate assignment;
- exact dependency/token graph used by execution;
- concrete schedule slots and affinity domains;
- resource-to-bank placement and aligned offsets;
- prefetch, transfer, barrier, eviction, and rematerialization operations;
- StreamPack/BCAB/native artifact digest;
- objective terms and units;
- exact/quantized/approximate classification;
- solver identity, search budget, lower bound, incumbent, and gap;
- measurement samples and confidence interval where applicable.

The verifier should reject a plan whose memory lifetimes do not cover its schedule or whose
price is computed from a different schedule.

### 11.3 Use a portfolio, not one solver

Recommended dispatch:

| Region/problem | Default fast rail | Proof/high-value rail |
|---|---|---|
| Additive layered candidates | current min-plus | same rail is exact |
| Additive caps/Pareto | current RCSP labels | bounded full Pareto/certificate |
| Independent or precedence jobs | EFT/list schedule | branch-and-bound or CP-SAT |
| Static memory | aligned first-fit | exact bounded layout/CP-SAT |
| Fixed-rate streams | SDF periodic schedule | balance/repetition-vector proof |
| Timed event regions | max-plus/min-plus analysis | cycle-time/bound certificate |
| Affine tensor loops | Transform/Affine schedule | polyhedral/tile search |
| Joint tensor schedule/memory | heuristic decomposition | bounded ILP/CP-SAT |
| Large candidate spaces | learned/evolutionary proposal | exhaustive sampled oracle |

The system should select the cheapest solver that can meet the requested proof level and
compile-time budget.

## 12. Prioritized implementation program

### P0 — make the objective and evidence truthful

1. **Unify schedule semantics.**
   - Define one schedule IR/artifact.
   - Make pricing, execution, lifetime, and verification consume it.
   - Rename legacy fixed-wave pricing if retained.
   - Correct the LangRef exactness statement.

2. **Make memory schedule-aware before composition.**
   - Derive live intervals from scheduled first/last use.
   - Verify transfer/prefetch extension of lifetimes.
   - Reject token overlap against a phase-only memory plan.

3. **Repair native calibration.**
   - Full-cycle stride/permutation.
   - Honest environment provenance.
   - Raw samples, intervals, working-set census, and counter availability.

Acceptance:

- the four-job pricing/EFT disagreement becomes impossible;
- the two-phase alias fixture is rejected or receives disjoint storage;
- virtualized runs cannot claim bare-metal provenance;
- old artifacts remain readable only with explicit legacy semantics.

### P1 — remove avoidable planner overhead and expose quality

4. **Incremental schedule pricing.**
   - Cache bin/phase contributions and predecessor couplings.
   - Reprice a changed candidate locally.
   - Target near-linear one-sweep scaling.

5. **Bounded exact scheduler and memory solver.**
   - Add dependency-free branch-and-bound for small fixtures.
   - Optionally use a hosted CP-SAT adapter for larger offline plans.
   - Emit lower bound, incumbent, and gap.

6. **Digest and canonicalization optimization.**
   - Iterative canonical stream instead of recursive flattening.
   - Immutable module digest cache.
   - Preserve independent verifier recomputation at trust boundaries.

Acceptance:

- 512-claim schedule-aware selection is no longer quadratic;
- all enumerated six-job schedules and seven-resource layouts match exact or report a gap;
- the 13-unit memory fixture is solved exactly on the proof rail;
- digest caching cannot survive mutation or cross-module substitution.

### P2 — add structured regions and joint decisions

7. **Export target schedules through MLIR Transform IR.**
   - Keep BCIR legality in BCIR dialects.
   - Represent schedule choices separately from payload IR.
   - Start with tiled GEMM and fused elementwise chains.

8. **Add Affine/tensor-map regions.**
   - Preserve loop domains, index maps, reuse, DMA/prefetch, and virtual views.
   - Lower to SCF/vector/LLVM only after scheduling.

9. **Add SDF and timed-event regions.**
   - Use fixed-rate periodic schedules where rates are proved.
   - Use max-plus/min-plus timing for event graphs.
   - Fall back to opaque DAG when preconditions fail.

10. **Jointly optimize schedule, memory, movement, and fusion.**
    - Begin with small tensor subgraphs.
    - Compare decomposition against exhaustive/constraint-solved candidates.
    - Add red/blue-pebble and capacity lower bounds.

Acceptance:

- at least two targets and two workload classes;
- exhaustive measured candidate comparison for bounded regions;
- exact schedule/memory certificate on small cases;
- no regression in R-law or StreamPack parity.

### P3 — establish silicon-grounded optimality claims

11. **Calibrate on controlled bare-metal targets.**
    - At minimum one x86 CPU and one materially different target
      (ARM, GPU, or accelerator).
    - PMU counters, bandwidth, frequency, temperature, energy, and toolchain provenance.

12. **Separate model error from search error.**
    - Compare predicted ranking with measured candidates.
    - Report confidence intervals.
    - Report optimizer regret under both predicted and measured objectives.

13. **Promote only generation-bound tables.**
    - Immutable, content-addressed measurements.
    - Quiescent activation, replay corpus, rollback.

No “TMSAO achieved” claim should be made before this phase has repeatable multi-target
evidence.

## 13. Decisions

### Keep

- GEM as the semantic/effect/resource graph;
- R1–R24 legality independent of cost;
- deterministic Q8/integer cost artifacts;
- min-plus candidate selection;
- RCSP/Pareto caps;
- StreamPack as a flat execution artifact;
- LLVM/GCC/vendor backends for mature low-level code generation;
- learned models only as proposals under deterministic gates.

### Change

- “single exact central optimizer” into a solver portfolio;
- phase-only memory planning into final-schedule-aware planning;
- fixed-wave scheduled pricing into canonical-schedule pricing;
- repeated whole-module hashing into immutable identity propagation;
- generic flat claims into typed hierarchical regions where structure is proved;
- unqualified optimality language into model-bounded certificates.

### Do not do

- do not rip out GEM;
- do not force every workload into affine, SDF, or tensor form;
- do not execute a dynamic graph interpreter on simple sequential hot paths;
- do not claim min-plus optimizes arbitrary algorithms merely because they can be encoded
  as graphs;
- do not add a learned legality path;
- do not equate CRC/hash integrity with authenticity;
- do not use WSL timings as bare-metal energy/thermal evidence.

## 14. Final assessment

BCIR is closer to a strong TMSAO **framework** than to a completed TMSAO **optimizer**.
Its differentiator is not that it has discovered one universal algorithm. Its
differentiator is the possibility of combining:

- proof-carrying semantic correspondence;
- explicit physical resources;
- multiple mathematical scheduling models;
- deterministic and learned candidate generation;
- measured target evidence;
- flat portable execution artifacts;
- rollback-safe adaptation.

The current core is sufficiently solid to preserve. The next leap is to make the
composition honest and joint:

```text
candidate choice ≠ schedule ≠ memory ≠ transfer ≠ execution
```

until BCIR binds and verifies them as one plan.

If that work is completed, GEM becomes the stable semantic substrate, tropical min-plus
remains the fast exact selector where its assumptions hold, and stronger region-specific
models handle the cases where they do not. That is a more credible route to state of the
art than replacing the core with another single abstraction.

## Appendix A — evidence artifacts

- Full Clang 22 test log: `/tmp/bcir-clang22-silicon-degrade.log`
- Deterministic performance audits:
  - `/tmp/bcir-tmsao-scale1.json`
  - `/tmp/bcir-tmsao-scale2.json`
  - `/tmp/bcir-tmsao-scale4.json`
- Clang 22 native trend samples:
  - `/tmp/bcir-tmsao-clang22-trend-v2.jsonl`
- Clean audit worktree:
  - `/tmp/BCIR-tmsao-audit-20260729`

## Appendix B — principal BCIR source anchors

- `docs/BCIR_LANGREF.md:42` — central K_BCIR equation and current exactness wording
- `bcir/kbcir/cost.py:33` — 12-axis cost vector
- `bcir/kbcir/realize.py:307` — layered min-plus optimizer
- `bcir/kbcir/rcsp.py:140` — additive constrained label DP
- `bcir/gem/overlap.py:129` — one-sweep schedule-aware candidate optimizer
- `bcir/gem/schedule.py:151` — phase-barriered HEFT-lite scheduler
- `bcir/gem/schedule.py:194` — token-DAG scheduler
- `bcir/kbcir/allocator.py:208` — phase-based live intervals
- `bcir/kbcir/static_memory.py:311` — aligned first-fit static memory plan
- `runtime/c/bcir_microbench.c:121` — native microbenchmark defaults/provenance

## Appendix C — additional primary references

- [MLIR Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
- [MLIR Affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- [GraphBLAS C API 2.1](https://graphblas.org/docs/GraphBLAS_API_C_v2.1.0.pdf)
- [HEFT](https://doi.org/10.1109/71.993206)
- [Hong-Kung I/O complexity](https://doi.org/10.1145/800076.802486)
- [Mirage, OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi)
- [KPerfIR, OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/guan)
- [VTC, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/hu-muyan)
- [MPK, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/cheng)
- [Twill, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/soi)
- [Nautilus](https://arxiv.org/abs/2604.14825)
- [COSMA](https://arxiv.org/abs/2311.18246)
- [Welder](https://www.usenix.org/system/files/osdi23-shi.pdf)
- [Timeloop](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation)
- [MAESTRO](https://arxiv.org/abs/1805.02566)
- [Accelergy](https://accelergy.mit.edu/paper.pdf)
- [TensorIR](https://arxiv.org/abs/2207.04296)
- [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng)
- [TASO](https://www.cs.cmu.edu/~zhihaoj2/papers/sosp19.pdf)
- [MetaSchedule](https://arxiv.org/abs/2205.13603)
- [egg](https://arxiv.org/abs/2004.03082)
