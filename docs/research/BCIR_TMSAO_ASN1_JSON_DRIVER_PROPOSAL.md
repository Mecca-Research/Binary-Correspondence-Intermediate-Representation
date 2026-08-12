# BCIR TMSAO, GEM+, ASN.1, and driver architecture proposal

> Source-backed architecture review: 2026-08-10.
>
> Repository baseline: main at 8965916, including merged PR #739. This document is a
> proposal and gap register. A statement is “landed” only when the cited repository
> implementation exists. Future algorithms, driver behavior, and performance targets are
> requirements, not current capabilities.

> **Audit status.** The correctness-closure defects this proposal's P0 stage depends on were
> independently reproduced and largely fixed on 2026-08-12; see
> [`BCIR_SECURITY_AUDIT_2026-08-12.md`](BCIR_SECURITY_AUDIT_2026-08-12.md). Fifteen defects
> confirmed, fourteen fixed. The one left open is the provenance digest's coverage, which is a
> two-rail change and is the first item of the GEM+ scope work.

## 1. Executive verdict

BCIR does not need its core ripped out. Its strongest foundations remain useful:

- the claim/resource graph is a conservative, portable semantic fallback;
- the R1–R25 law rail separates legality from optimization;
- GEM makes hazards and executable order explicit;
- K_BCIR makes realization choice target- and policy-dependent;
- StreamPack freezes a compact, verified execution order;
- native C and MLIR/C++ rails provide independent checks against the Python oracle.

The present system is nevertheless not a complete Theoretical Maximum System Architecture
Optimization engine. Tropical min-plus solves one important problem exactly: a minimum
additive-cost path through the legal candidates represented by a layered DAG. It does not, by
itself, solve joint scheduling, placement, memory layout, register allocation, fusion,
communication overlap, cache conflicts, thermal response, or stochastic control. Encoding all
of that history into a shortest-path state is possible in principle and usually combinatorial
in practice.

The correct evolution is **GEM+**, not a replacement with one new universal formalism:

1. keep the opaque claim DAG as the always-legal fallback;
2. recognize typed regions with stronger local mathematics;
3. place equivalent realizations in a versioned candidate/equivalence graph;
4. select a bounded solver portfolio by region and proof obligation;
5. publish one content-addressed execution plan that jointly records all interacting choices;
6. certify the incumbent against explicit lower bounds and uncertainty.

ASN.1 should become BCIR’s strongest external schema and encoding portfolio, but not its sole
internal binary or runtime execution format. JER is UTF-8 JSON text and belongs on build,
configuration, control, and diagnostic planes. DER, COER, canonical PER, and declared ECN
realizations are compact interchange choices. Verified claims, StreamPack, BCAB, and native
objects remain execution artifacts.

The Python oracle should also be narrowed, not discarded. Stable, repeatedly executed
compiler and control-plane work should migrate to the existing MLIR C++23 optimizer core.
Python remains the executable specification, differential oracle, research surface, test
generator, and hosted AI/model layer. C remains the freestanding and fixed-contract runtime.

## 2. Operational definition of TMSAO

> **Theoretical Maximum System Architecture Optimization is the best proven or measured legal
> realization within a declared, content-addressed scope, with a reported optimality gap.**

This is an operational engineering definition, not a claim that one program is universally
optimal. There is no workload-independent realization that simultaneously minimizes latency,
energy, temperature, memory, code size, compilation time, wear, and error for every CMOS
machine and input distribution. Even on one target, these objectives conflict.

### 2.1 Scope identity

A TMSAO result is meaningful only under:

    S = digest(P, H, W, Theta, A, B, O, M, U, G)

where:

- **P — program contract:** BCIR graph, inputs, laws, semantics, precision, and admitted
  approximation;
- **H — hardware:** topology, instruction and capability sets, memory banks, links, capacities,
  and independently identified devices;
- **W — workload:** shapes, input distribution, concurrency, service levels, and planning
  horizon;
- **Theta — operating state:** firmware, microcode, driver, OS, clocks, thermal state,
  contention, and wear;
- **A — admitted search space:** transformations, kernels, libraries, schedules, and candidate
  boundary;
- **B — hard bounds:** capacity, security, reliability, temperature, power, accuracy, and
  policy constraints;
- **O — objective relation:** lexicographic, Pareto, robust, constrained, or explicitly
  scalarized;
- **M — measurement protocol:** warm-up, repetitions, counters, environment, outlier policy,
  and timing source;
- **U — uncertainty:** confidence or prediction coverage, quantization, sensor error, and
  unobserved-state treatment;
- **G — generations:** hardware profile, firmware, driver, calibration, model, schema, and
  artifact generations.

Changing any member invalidates reuse unless the artifact declares and proves a safe
projection. A driver profile reduces ambiguity; it cannot assert zero ambiguity. Hidden
firmware, silicon variation, sensor resolution, neighboring traffic, and input-dependent
execution remain measured or bounded variables in Theta and U.

### 2.2 Computational form of the Axiom of Modules

The scope equation turns the proposed module quadruple into a computable contract:

    Module X = (M, R_M, P_M, T_M)

| Axiom component | GEM+ realization |
|---|---|
| Underlying set M | regions, claims, resources, candidates, schedules, placements, artifacts |
| Labeled relationships R_M | depends-on, equivalent-to, conflicts-with, communicates-with, contains, refines, invalidates |
| Dynamic parameters P_M | H, W, Theta, policy, calibration, limits, solver budgets, generation |
| Topology or structure T_M | graph topology, typed-region algebra, memory/fabric topology, partial orders, metrics |

The quadruple supplies the mathematical vocabulary; the content digest, finite bounds,
verifier, measurement protocol, and certificate make it operational.

### 2.3 Certificate levels

The former letter classes are renamed to ordered levels so the proof strength is visible:

| Level | Required evidence | Permitted claim |
|---|---|---|
| **1 — exact** | complete finite search or a sound proof; lower bound equals incumbent; deterministic legality and replay | optimal within S |
| **2 — bounded** | sound lower bound, legal incumbent, explicit finite gap, and solver termination/budget record | no worse than the reported gap within S |
| **3 — measured** | declared finite candidate set, controlled target measurements, robust interval, winning incumbent, and replay | best measured admitted realization; no claim outside that set |
| **4 — heuristic** | legal replayable incumbent and full provenance, but no useful lower bound or complete measured comparison | candidate found; optimality unknown |

Every certificate records absolute and relative gaps when defined:

    absolute_gap = incumbent - lower_bound
    relative_gap = absolute_gap / max(abs(incumbent), epsilon)

For Pareto objectives the certificate records a lower-bound set or dominance envelope rather
than inventing one scalar gap. For stochastic objectives it records the distribution source,
coverage, robust statistic, and chance or worst-case constraint.

A component certificate does not automatically compose into a system certificate. A locally
optimal layout can prevent later fusion; a memory-minimal schedule can reduce parallelism; a
fast kernel can increase shared-memory pressure. Components therefore expose a parametric or
Pareto interface summary. The composer must prove compatibility or downgrade the resulting
level and report the remaining global gap.

## 3. Source-backed state at PR #739

### 3.1 What is already strong

The current Python rail contains real optimization machinery:

- additive 12-axis costs and target profiles in
  [cost.py](../../bcir/kbcir/cost.py);
- layered candidate realization and min-plus coupling in
  [realize.py](../../bcir/kbcir/realize.py);
- duration-aware EFT, asynchronous tokens, and overlap pricing in
  [schedule.py](../../bcir/gem/schedule.py),
  [async_tokens.py](../../bcir/gem/async_tokens.py), and
  [overlap.py](../../bcir/gem/overlap.py);
- resource-constrained shortest-path labels and Pareto pruning in
  [rcsp.py](../../bcir/kbcir/rcsp.py);
- bounded equality saturation in [egraph.py](../../bcir/kbcir/egraph.py);
- deterministic memory planning and independent alias verification in
  [static_memory.py](../../bcir/kbcir/static_memory.py) and
  [allocator.py](../../bcir/kbcir/allocator.py);
- HAM routes, next-use eviction, content shards, DMA descriptions, and replay in
  [ham.py](../../bcir/kbcir/ham.py), [context_shard.py](../../bcir/kbcir/context_shard.py),
  and [dma.py](../../bcir/kbcir/dma.py);
- tile, fusion, layout, cache, throttle, calibration, learned-prior, PUCT, and model-planning
  organs throughout [bcir/kbcir](../../bcir/kbcir/).

The MLIR C++23 rail is no longer a thin legality shell. It contains deterministic cost,
fusion, coupled-plan, overlap/max-plus, RCSP/Pareto, composition, replay, EFT, asynchronous,
power, allocation-pool, and tensor lowering passes under
[mlir/lib/passes](../../mlir/lib/passes/). LLVM, Clang, and vendor libraries remain
general-purpose code-generation providers rather than BCIR legality authorities.

The [bounded performance audit](../PERFORMANCE_AUDIT.md) has already removed several major
avoidable costs: recursive graph traversal, pairwise hazard discovery on independent claims,
quadratic first-fit rescanning, and inner-loop Python row copies. That is meaningful progress,
but it is not a global-optimality proof.

### 3.2 Confirmed architectural gaps

The current model is still split across related but non-identical decisions:

- priced realization, EFT scheduling, and token execution do not share one canonical schedule;
- candidate choice, scheduling, placement, and static memory are optimized in separate stages;
- schedule-aware candidate selection is a local sweep that repeatedly evaluates makespan;
- static memory uses deterministic lowest-address aligned first-fit rather than an optimal
  bounded layout solver;
- static liveness is topological/phase based, while token execution can overlap phases;
- [schedule_artifact.py](../../bcir/kbcir/schedule_artifact.py) is a useful matmul-specific
  artifact, not a whole-program transformation and execution plan;
- [semiring.py](../../bcir/kbcir/semiring.py) provides min-plus DAG shortest path, not a
  semiring/objective registry;
- the generic [graph model](../../bcir/model/graph.py) has phases, claims, and resources but
  no typed-region contract or versioned candidate/equivalence graph;
- physical PMU, energy, thermal, memory-controller, and accelerator evidence is incomplete.

These are integration gaps, not evidence that the claim DAG or min-plus implementation should
be deleted.

### 3.3 Scaling evidence and its correct interpretation

Linear work is not automatically a failure. Reading N claims, validating N directory entries,
or emitting B StreamPack bytes has lower bounds of Omega(N) or Omega(B). A TMSAO effort may
lower constants, reduce the admitted input, reuse previous work, compress output, or avoid
materialization; it cannot honestly promise sublinear time while consuming and emitting the
same uncached information.

The earlier operational sweep supplied near-linear Q8/Q4 and OLS/PCA scaling and much larger
growth for autodiff, tiled matmul, and K_BCIR-to-StreamPack. Those observations are useful
triage, but raw samples and the exact environment were not committed. They remain Level 4
evidence until reproduced by the repository audit.

Likewise, a dense BCIR-to-compiler-loop ratio around 0.98–1.01 is parity, not a theorem that
the ratio can be pushed above one. When LLVM already sees the same affine loop and alias
facts, matching its generated loop is the expected floor. BCIR earns a structural win only
when it possesses additional legal information: equivalent layout, avoidable gather,
cross-operator fusion, bounded accuracy, placement, or whole-graph reuse.

The present strided microbenchmark has a validity defect:
[microbench.py](../../bcir/kbcir/microbench.py) constructs
“(i * stride) mod n” with default stride 16 and power-of-two n. It visits only n / 16 unique
positions. Before those values calibrate TMSAO, the benchmark must use a stride coprime to n
or construct a proven full-cycle permutation, record unique elements and working-set bytes,
retain raw timings, define an outlier rule, and attest the timing and target environment.

## 4. GEM+: the canonical architecture

### 4.1 Typed regions over a conservative DAG

Every program remains representable as claims and resources. A recognizer may refine a
verified subgraph into one of these region types:

| Region | Stronger local model | Primary decisions |
|---|---|---|
| affine/polyhedral | affine sets, maps, dependences, schedules | loop transforms, tile, vectorize, locality |
| SDF/CSDF | repetition vectors and fixed/cyclo-static rates | bounded buffers, periodic schedule, throughput |
| tensor-index | index maps, layouts, contraction/fusion algebra | virtual movement, layout, fusion, tiling |
| timed-event | event graph and max-plus timing | critical paths, initiation interval, throughput cycles |
| state-machine | explicit states, transitions, capabilities, generations | asynchronous events, cancellation, restart |
| opaque claim DAG | existing hazards, bounds, and order | universal conservative fallback |

Refinement is additive. Failure to recognize or prove a region returns to the claim DAG;
it never silently guesses a stronger algebra.

### 4.2 Versioned candidate/equivalence graph

The layered candidate columns become a general graph whose nodes are legal realizations and
whose typed edges mean:

- semantic equivalence;
- permitted approximation with a declared error contract;
- refinement or lowering;
- layout or representation conversion;
- schedule/placement compatibility;
- invalidation by target or generation;
- composition with an adjacent region.

Each node carries content identity, required capabilities, proof references, resource effects,
shape predicates, and a parametric 12-axis cost. Each edge declares whether it preserves exact
values, bounded error, identity/provenance, ordering, and replay. Cycles are allowed for
exploration; the selected plan is acyclic in execution dependencies.

### 4.3 Algebra and objective registry

Tropical min-plus remains the default for additive path composition, but it is one registered
algebra:

| Algebra/order | Use |
|---|---|
| min-plus | additive realization and routing paths |
| max-plus | critical paths, timed events, pipeline cycle time |
| min-max | bottleneck links, peak pressure, worst-stage objective |
| Boolean | reachability, capability, legality, feature summaries |
| lexicographic | non-negotiable priority order after hard feasibility |
| Pareto | non-substitutable objectives and composition surfaces |
| stochastic/robust | measured distributions, prediction intervals, worst-case or chance constraints |

The registry must define identity, composition, comparison, overflow behavior, units, and
whether pruning is sound. A weighted sum is legal only for commensurate, policy-authorized
utilities. Security, correctness, capacity, and minimum accuracy are constraints, not negative
costs that a latency win can purchase away.

### 4.4 One canonical execution plan

GEM+ introduces a versioned, content-addressed plan containing:

- scope digest S and all generation identities;
- typed regions and fallback boundaries;
- candidate/equivalence selections and legality/proof references;
- schedule slots, events, dependencies, cancellation points, and deadlines;
- device, core, bank, register-class, and address placement;
- actual scheduled lifetimes, allocations, alignment, alias groups, and peak live sets;
- movement routes, copies, virtual views, prefetch, eviction, rematerialization, and barriers;
- layout, fusion, tiling, vectorization, precision, library, and code-generation selections;
- twelve-axis cost vectors, hard constraints, objective relation, and policy;
- lower bounds, incumbent, intervals, certificate level, and optimality gaps;
- output artifacts, hashes, replay inputs, fallbacks, and rollback generations.

StreamPack is lowered from this plan. It is not a second independent planner. MLIR Transform
IR can carry reusable target transformations, while LLVM VPlan can inform vectorization-plan
interchange; neither becomes BCIR’s universal semantics. The MLIR
[Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/) explicitly represents
fine-grained transformations and alternatives, and
[VPlan](https://llvm.org/docs/VectorizationPlan.html) represents candidate vectorization plans.
GEM+ should export/import the relevant decisions rather than duplicate LLVM’s final
instruction selection and register allocation.

## 5. Solver portfolio and lower-bound stack

### 5.1 Portfolio

The first implementation should reuse current organs and add bounded adapters:

| Problem | Primary solver | Verification/fallback |
|---|---|---|
| additive layered candidates/routes | current min-plus DP | exhaustive tiny graph |
| constrained candidate paths | current RCSP labels/Pareto | complete bounded enumeration |
| equality and local fusion/layout | current bounded e-graph | deterministic extraction replay |
| independent/mildly coupled tasks | current EFT/HEFT-lite | exact tiny schedule oracle |
| regular affine regions | affine dependence/schedule search | MLIR Affine legality |
| SDF/CSDF regions | repetition-vector and periodic scheduling | balance equations, bounded buffers |
| timed-event regions | max-plus longest path/cycle mean | event replay |
| joint schedule/place/memory | bounded CP-SAT/ILP or branch-and-bound adapter | independent plan verifier |
| large coupled regions | anytime beam/A*/large-neighborhood/decomposition | legal incumbent at every stop |
| uncertain target state | robust interval or scenario optimization | declared distributions/scenarios |
| candidate generation | learned prior/PUCT/residual prediction | deterministic law and plan verifier |

External CP-SAT/ILP is a hosted optional reference, not a freestanding dependency. COSMA is
useful evidence that bounded accelerator graphs can jointly optimize scheduling, allocation,
and replacement through an exact model
([paper](https://arxiv.org/abs/2311.18246)). BCIR should use such a model as a small-instance
oracle and high-value-region solver, then measure heuristic gaps continuously. The classical
SDF formulation supplies static periodic scheduling for fixed-rate streams
([Lee and Messerschmitt](https://doi.org/10.1109/TC.1987.5009446)); QEMU, event-driven
drivers, and general programs still require different regions.

### 5.2 Mandatory lower bounds

Each plan reports every applicable bound:

- semantic work and critical-path bound;
- resource/work bound by execution class;
- Roofline compute/bandwidth bound;
- communication cut, link-capacity, and congestion bound;
- peak live-set and bank-capacity bound;
- interval-graph/clique and alignment-aware allocation bound;
- register/shared-memory occupancy bound where the backend exposes it;
- encoding-size and required-output bound;
- thermal/power-envelope bound over the declared horizon;
- service/deadline/network-calculus bound for bounded streams.

Unavailable bounds are marked unavailable, not zero. The bound stack is independently
checkable and cannot call the same heuristic that produced the incumbent.

### 5.3 Joint optimization without combinatorial denial

Driver profiles narrow the state space but cannot make a hard problem disappear. GEM+ uses:

1. region recognition and exact local simplification;
2. capability and hard-bound pruning;
3. dominance/Pareto pruning;
4. decomposition by weak coupling and cut boundaries;
5. iterative schedule–placement–memory refinement;
6. exact solving for small/high-value regions;
7. anytime legal incumbents for the remainder;
8. lower-bound tracking and explicit stop budgets.

Potential decomposition techniques include Lagrangian relaxation, Benders-style cuts,
column generation, and large-neighborhood search, but each enters only after a deterministic
small-instance oracle demonstrates its gap and failure behavior.

## 6. Scaling, scheduling, and memory program

### 6.1 Make incremental work the default

The path below linear rebuild time is not a faster full scan; it is avoiding a full rebuild:

- content-address every region, candidate set, proof, measurement table, and plan fragment;
- invalidate by dependency and generation rather than by whole program;
- compile repeated graph families into bounded parameterized templates;
- store sparse deltas for unchanged claims, resources, schedules, and StreamPack blocks;
- memoize region recognition, legality, lower bounds, and Pareto fronts;
- retain resident plan state and native artifacts across invocations;
- stream validation and lowering without materializing duplicate object graphs;
- compress only where decompression plus transfer is measured cheaper;
- use lazy hydration and demand-driven code generation for unreachable or cold variants.

Every cache hit must verify all identities in S. “Same target name” is not enough; firmware,
driver, thermal policy, calibration generation, and workload predicates must match or declare a
safe projection.

### 6.2 Canonical schedule and liveness

The immediate correctness fix is one schedule model:

1. candidate effects create tasks, resources, events, and movement;
2. the scheduler assigns explicit start/end or partial-order event intervals;
3. memory liveness derives from those actual intervals, including overlap and asynchronous
   completion;
4. allocation and bank placement feed conflicts, spills, and movement back to scheduling;
5. iteration stops at a fixed point, exact solve, budget, or certified gap;
6. StreamPack carries the selected events and barriers.

Phase-topological liveness remains a conservative fallback. It may over-allocate but must never
alias values whose scheduled lifetimes overlap.

### 6.3 Static allocation tiers

Use three allocation modes:

- **linear-time deterministic fallback:** current coalescing first-fit;
- **bounded improvement:** interval coloring, bank-aware best fit, local compaction, and
  conflict-directed repair;
- **exact small-region oracle:** alignment-aware CP-SAT/ILP or branch-and-bound with peak
  address/bank capacity objective.

The independent alias verifier accepts all three. An “optimal memory” certificate includes the
live-set lower bound, alignment fragmentation bound, bank constraints, incumbent peak, and gap.

### 6.4 Data access and dense-loop work

For dense affine loops, first improve information rather than fight the backend:

- preserve restrict/no-alias, alignment, extent, and divisibility facts;
- export affine maps and vectorization candidates to MLIR/LLVM;
- fuse adjacent memory-bound operators when register/shared-memory bounds permit;
- specialize shapes only when code-size and compile-time policy admit it;
- compare vendor library, generated loop, persistent kernel, and fallback candidates.

For strided/gather work:

- prove whether the map is injective, a permutation, periodic, or many-to-one;
- use loop interchange, blocking, structure-of-arrays, and cache-line grouping;
- price native gather only after target measurement;
- preserve reduction order unless an approximation contract permits reassociation;
- distinguish fewer unique elements from a real locality improvement.

The existing direct-stride versus gather win is useful because BCIR knows the index relation.
It is not evidence that every irregular access can be flattened.

### 6.5 HAM, Semantic Swap, and heterogeneous memory

HAM becomes the plan’s physical/logical movement layer, while Semantic Swap becomes one
candidate family for eviction, rematerialization, and persistence. The joint plan must model:

- CPU registers/cache/DRAM, accelerator registers/SRAM/HBM, fabric, and SSD/NVMe banks;
- capacity, bandwidth, latency, concurrency, transfer granularity, coherence, and endurance;
- tensor/shard ownership, dirty state, generations, and recovery;
- prefetch issue, transfer overlap, completion events, eviction, and rematerialization;
- hard pinning, security/capability domains, and forbidden routes;
- uncertainty and thermal response for sustained traffic.

The comparison boundary is:

| System/idea | What BCIR should learn | What BCIR must add |
|---|---|---|
| [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) | partition and offload model state across GPU, CPU, and NVMe | generalize beyond training tensors; prove lifetime, route, and recovery |
| [Hugging Face Accelerate device maps](https://huggingface.co/docs/accelerate/main/concept_guides/big_model_inference) | payload-free sizing and explicit GPU/CPU/disk placement | measured schedules, transfer overlap, certificates, no silent fallback |
| [Apple Metal unified-memory capability](https://developer.apple.com/documentation/metal/mtldevice/hasunifiedmemory) | distinguish physically unified and discrete memory | profile coherence and contention; never assume “unified” means free movement |
| [GraphBLAS](https://graphblas.org/) | sparse graph algebra over declared semirings | use as a typed-region/provider interface, not universal execution semantics |
| [Mirage](https://arxiv.org/abs/2405.05751) | jointly search algebra, schedule, and multi-level GPU kernels | deterministic exact/parity rail where required and target-independent fallback |
| [Nautilus](https://arxiv.org/abs/2604.14825) | preserve regular structure while auto-scheduling tiled kernels | admit only after independent benchmark/provenance gates |
| [Event Tensor](https://arxiv.org/abs/2604.13327) | first-class tile events and persistent dynamic megakernels | map into timed-event regions and bounded cancellation/recovery |
| persistent e-graphs | retain equivalences across incremental replans | generation-aware eviction and bounded memory |
| MLIR Transform/VPlan | explicit, replayable transformation choices | connect to the canonical BCIR plan without making them law authorities |
| hardware protection keys | cheap page-domain switching where available | capability checks and fallback; keys are not spatial memory safety |

Names such as “MPK” are overloaded; this document distinguishes Mirage Persistent Kernel from
Linux/CPU memory protection keys. Any future provider must pin the project, revision, license,
target, and semantic contract before comparison.

### 6.6 SYCL and provider boundary

SYCL remains a portable hosted provider and hardware-discovery surface. It does not define
legality and does not erase vendor-specific effects. The canonical plan can target a SYCL
queue, CUDA stream, HIP queue, CPU pool, or freestanding channel through one event/movement
contract. A provider must report:

- devices, memory kinds, links, queue ordering, and atomic/coherence capabilities;
- allocation, copy, prefetch, kernel, event, cancellation, and teardown behavior;
- native handle generation and ownership;
- measured intervals and unavailable counters;
- exact failure instead of silently moving work to another backend.

## 7. ASN.1 through PR #739 and its GEM+ role

### 7.1 What is now built

The ASN.1 implementation is materially ahead of the earlier PR #707 analysis. Current source
and tests show:

- X.680 front end, X.681 objects, X.682 constraints, and X.683 parameterization;
- BER-in/DER-out, aligned and unaligned PER, OER/COER, XER, and JER;
- X.692 built-in and user-defined ECN, including links, value mapping, parameterization,
  constructors, contained types, and transforms;
- Python oracle implementations, freestanding C twins for the bounded wire rails, and C++
  JER structural-index/SIMD helpers;
- plan-driven PER C decoding;
- R24 for ASN.1 and R25 for ECN on the MLIR verifier rail;
- native encoding-selection measurements, intervals, generation-tagged calibration, and
  certificates;
- StreamPack and ArtifactBundle additive projections.

“Three parts” is BCIR’s implementation taxonomy—the class/object model, user-defined
encoding behavior, and clause 20 defined-syntax ingestion—not a claim that X.692 is formally
published as three numbered parts. The supplied 2021 recommendation’s contents confirm the
single standard’s integrated model: parameterization, replacement structures, value mappings,
contained types, defined syntax, and the clause 23 class-specific notations depend on one
another.

The detailed inventory is in the
[ASN.1 build-out roadmap](../BCIR_ASN1_BUILDOUT_ROADMAP.md), normative language is in
[LangRef section 17](../BCIR_LANGREF.md), and the latest external comparison is
[BCIR’s ASN.1 compiler comparison](../BCIR_ASN1_COMPILER_COMPARISON.md).

“All X.692 parts are built” does not mean the entire ASN.1 ecosystem is closed. The current
front end deliberately refuses or lacks:

- COMPONENTS OF inlining;
- selection types;
- WITH SUCCESSORS and WITH DESCENDANTS imports;
- a large real-protocol grammar corpus;
- schema-directed valid-value generation;
- an expanded adversarial/security corpus;
- ICD/bit-layout reporting and a polished user CLI;
- target PMU/energy calibration beyond the admitted measurements.

The ECN parser has no remaining unsupported keyword group, but unreadable class notations such
as CHARS, NUL, and TAG still require an explicit supported/refused contract. These gaps should
remain visible rather than allowing “standard complete” to mean different things in different
documents.

Two roadmap headings also lag their own body: “baseline through PR #670” and “built-in model
landed; user-defined half reopened” no longer summarize the implementation through #739.
That is documentation debt, not missing code, and should be reconciled in the next ASN.1
roadmap maintenance pass.

### 7.2 ASN.1 is a portfolio, not the universal internal binary

BCIR should adopt ASN.1 as the default **external contract language when a durable,
cross-language, constrained schema is needed**. It should not force every internal object or
hot execution path through ASN.1.

| Plane | Preferred representation |
|---|---|
| schema, compatibility, external wire contract | ASN.1 modules plus constraints/instructions |
| human/configuration/control | bounded JER or other explicitly admitted text projection |
| canonical compact interchange | DER, COER, canonical PER, or a selected declared ECN |
| optimizer semantics | typed GEM+ regions, claims/resources, candidate graph |
| executable order | canonical execution plan lowered to StreamPack |
| deployable package | BCAB and native/backend objects |
| telemetry evidence | frozen frame ABI; optional JER export for inspection |

CBOR, FlatBuffers, Protocol Buffers, MessagePack, Thrift, Ion, EXI, and other formats may enter
as measured provider candidates when a workload requires them. They are not all equivalent:
some provide schemas, some canonical encodings, some zero-copy table access, and some only a
wire representation. ASN.1 earns a privileged contract role from its constraint and encoding
family, not from a claim that one encoding always wins.

### 7.3 JER boundary

JER remains JSON text. Schema specialization removes type guessing and can avoid a DOM and
heap allocation; it does not remove UTF-8, string escape, number, duplicate-key, depth,
length, and schema validation. A structural tape or SIMD scanner is an index and validation
accelerator, not proof that variable values sit at fixed offsets.

The safe pipeline is:

    ASN.1 schema + bounded JER bytes
      -> scalar UTF-8 and structural validation
      -> optional SIMD structural index
      -> generated schema-specialized event parser
      -> transactional typed value or claim builder
      -> R24/R25 and ordinary BCIR laws
      -> GEM+ candidate graph and K_BCIR selection
      -> canonical plan
      -> StreamPack / BCAB / native artifact

Privileged execution does not parse JSON. A speculative “happy path” may remove repeated work
only after a complete validation result for the exact immutable bytes is bound to the schema
and profile digest. Hardware faults are not a substitute for validation and transactional
rollback.

### 7.4 ASN.1 encoding as a typed GEM+ region

Each serialization region exposes candidates keyed by:

- schema/module digest and selected type;
- encoding family, standardized rule, canonical/private profile, and ECN identities;
- constraints, JER instructions, ECN link/module/parameter/value-mapping identities;
- encode/decode direction and schema-directed versus schema-free decode kind;
- canonicality, compactness, alignment, scratch, latency, energy, and streaming behavior;
- target generation and measurement interval.

K_BCIR compares these candidates under the existing twelve axes. Cache-line and stride effects
belong within memory/contention; there is no thirteenth cost dimension. A selection certificate
records the lower bound, measured candidate set, intervals, hard constraints, chosen encoding,
and plan/artifact identities.

ECN metaprogramming remains compile-time or controlled-load-time generation of immutable,
verified descriptors and code. It is not in-flight self-modifying kernel logic. An encoding
switch produces a new generation activated at a quiescent boundary with compatibility checks
and rollback.

### 7.5 Network and telecom correspondence

ASN.1, PER, and ECN can express internet and telecom structures through one abstract value, but
that does not make physical media interchangeable for free. Zero-copy is legal only when:

- source and destination layouts are proved identical or represented by a virtual view;
- ownership, lifetime, alignment, coherency, and DMA/IOMMU domains agree;
- endianness, bit ordering, framing, integrity, and retransmission semantics agree;
- no security boundary requires a copy or revalidation.

Otherwise GEM+ emits explicit unpack, transform, copy, scatter/gather, and barrier tasks and
prices them. Fragmenting one logical stream across Ethernet, Wi-Fi, LTE, or LoRa requires a
transport policy, sequencing, congestion/reliability model, security association, and
reassembly contract above the encoding rule.

### 7.6 Parallel driver tracks

The first comparison must be userspace/simulator based:

1. freeze one device schema and physical register/packet layout;
2. implement the conventional C/Linux reference;
3. compile the schema-authored path into the same direct RuntimeChannel behavior;
4. compare commands, state transitions, faults, and teardown;
5. then compare latency, traffic, cache behavior, and code size;
6. only after parity, test separate virtual devices or sequential binding in BCIR-Linux.

BCIR may optimize access order, staging, coalescing, packet alternatives, and verified command
fusion. It cannot repack device-defined MMIO registers. Two drivers must never bind
simultaneously to one physical device.

## 8. Python-to-C++ migration roadmap

### 8.1 Boundary

Current approximate source scale makes the need clear without making language the objective:
the Python package contains roughly 96,000 non-test lines, runtime C roughly 31,000, runtime
C++ roughly 1,800, and MLIR C++/definitions roughly 15,000. Much of the optimizer already has
a real C++23 implementation. A blind rewrite would create two new oracles and years of drift.

Use this placement rule:

- **C:** freestanding, allocation-free/fixed-allocation runtime, wire codecs, loaders, kernels,
  driver edges, and stable ABI;
- **C++23/MLIR:** immutable artifact ownership, candidate graphs, canonical plans, solver
  adapters, incremental caches, hosted ASN.1 compilation, and performance-sensitive
  orchestration;
- **Python:** executable semantics, differential reference, research, fuzz/test generation,
  hosted model training, analysis, visualization, and provider-neutral experimentation.

### 8.2 Migration slices

| Slice | Work | Promotion gate |
|---|---|---|
| CXX0 inventory | profile imports, allocations, graph sizes, and duplicated Python/C++ behavior | committed call/ownership map and benchmarks |
| CXX1 artifact core | immutable IDs, typed regions, candidate graph, canonical plan schema, C ABI views | deterministic byte identity and malformed-input refusal |
| CXX2 plan builder | compose existing MLIR cost/fusion/RCSP/EFT passes through one plan builder/replayer | Python/C++ differential parity over generated graphs |
| CXX3 joint solvers | scheduled liveness, bank placement, exact tiny oracle, bounded anytime portfolio | legal incumbent at interruption; gap certificates |
| CXX4 incremental engine | dependency invalidation, memoized regions/frontiers, delta StreamPack | full rebuild identity and adversarial cache invalidation |
| CXX5 ASN.1 host compiler | immutable schema/encoding descriptors, JER event parser generation, ECN plan integration | Python/C/C++ wire and claim parity |
| CXX6 memory/providers | HAM/Semantic Swap/SYCL/provider orchestration and event ownership | failure, cancellation, peer/device loss, replay |
| CXX7 retirement | remove only Python duplicates with stable native coverage | import quarantine, compatibility notice, no lost oracle |

The first candidates to migrate are the integration bottlenecks: generic graph traversal and
region recognition, candidate/equivalence construction, whole-plan composition, scheduled
liveness/allocation, incremental artifact caching, and hosted ASN.1 descriptor generation.

Do not migrate experimental model architectures, calibration analysis, small reference
algorithms, test generators, or readable legality oracles merely to increase C++ line count.
Python APIs can become thin bindings after native stability. Existing direct Python entry
points remain until deprecation and byte/semantic compatibility gates are complete.

### 8.3 Ownership and failure contract

Every C++ public object is immutable after publication or has one explicit owner. Builders are
transactional. Allocation failure, solver cancellation, invalid input, or provider loss cannot
replace the previous valid artifact. C views are borrowed with explicit lifetime. Python
bindings own reference-counted handles, never raw cross-language pointers. Serialization is
versioned, bounded, deterministic, and independently parsed.

## 9. Hardware profiles and the native measurement rig

### 9.1 What this host proves

The 2026-08-10 probe observed:

- WSL2 kernel 6.18.33.2 with Microsoft hypervisor;
- AMD Ryzen 5 2600, six cores/twelve threads exposed to the guest;
- only breakpoint, kprobe, msr, software, tracepoint, and uprobe event sources—no CPU PMU;
- perf_event_paranoid 2, empty powercap, and no perf userspace tool;
- NVIDIA GeForce GTX 1660 SUPER, Windows KMD 610.88 and WSL user-mode 610.57.01;
- QEMU 6.2 for x86-64 and AArch64.

This is enough for functional x86/CUDA checks and QEMU architecture/boot/driver tests. It is
not enough for physical CPU PMU, package energy, memory-controller, or bare-metal thermal
certificates. QEMU’s own documentation states that instruction-count mode is deterministic but
not cycle accurate and instruction count can be poorly correlated with hardware performance
([QEMU invocation](https://www.qemu.org/docs/master/system/invocation.html)).

NVIDIA supports Turing in current Nsight Compute and permits WSL profiling when host counter
access is enabled, but WSL retains managed-memory/NVML limitations and NVIDIA has documented
WSL timestamp caveats. GPU-only measurements may become Level 3 for the declared WSL target;
they cannot be silently combined with unavailable physical CPU counters into a whole-system
Level 1/2 certificate
([Nsight requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html),
[CUDA on WSL constraints](https://docs.nvidia.com/cuda/archive/13.0.2/wsl-user-guide/index.html)).

### 9.2 Recommended Linux environment

Install a current supported x86-64 Linux distribution directly on a spare SSD or dedicated
partition in UEFI mode. Fedora Workstation or Debian stable are suitable; BCIR should remain
distro-neutral. A live USB is useful for probing but a persistent native install is required
for pinned tools, kernel parameters, repeatable storage, and NVIDIA modules. Ubuntu is not a
requirement.

Required setup:

1. preserve Windows recovery keys and backups; use a separate disk where practical;
2. install native Linux with the distribution kernel, matching headers, compiler, perf,
   cpupower/kernel tools, sensors, and QEMU;
3. install the NVIDIA Linux driver through one supported packaging path; do not mix the WSL
   proxy driver with native Linux packages;
4. keep Secure Boot enabled with signed modules, or document a deliberate lab-only change;
5. record UEFI/BIOS, microcode, kernel, mitigations, driver, CUDA, compiler, and library hashes;
6. probe available PMUs under sysfs and verify cycles/instructions/cache events with perf;
7. grant narrowly scoped CAP_PERFMON where possible rather than global root or permanently
   opening perf_event_paranoid
   ([kernel perf security](https://docs.kernel.org/6.2/admin-guide/perf-security.html));
8. probe powercap and hwmon. The older Ryzen model may not expose the amd_energy driver’s
   documented supported model set; unavailable package energy requires a calibrated external
   wall meter, not an estimate
   ([amd_energy](https://www.kernel.org/doc/html/v5.11/hwmon/amd_energy.html),
   [powercap](https://docs.kernel.org/6.8/power/powercap/powercap.html));
9. use AMD uProf/IBS/PMC only where the tool reports support, retaining raw data and tool
   version ([AMD uProf](https://www.amd.com/en/developer/uprof/uprof-performance-analysis.html));
10. establish pinned-core and whole-system profiles separately; use cpusets/affinity and IRQ
    placement before considering more invasive isolation
    ([kernel CPU isolation](https://docs.kernel.org/admin-guide/cpu-isolation.html));
11. control or record governor, boost, clocks, fan curve, ambient temperature, warm-up,
    throttling, and background services
    ([CPU frequency scaling](https://docs.kernel.org/6.17/admin-guide/pm/cpufreq.html));
12. collect raw wall time, counters, energy, temperature, frequency, migrations, faults, and
    context switches with prediction intervals and a declared outlier policy.

Run two profiles rather than hiding the tradeoff: a fixed/reproducible clock profile for model
calibration and a normal governor/boost profile for deployment behavior. Never label a VM,
container, WSL run, or QEMU run “bare metal.”

### 9.3 Target profile generation

A profile is not a handwritten constant table. The driver/measurement agent emits:

- topology/capability inventory and inaccessible features;
- capacity and hard-policy limits;
- microbenchmark corpus identity and raw samples;
- response surfaces for shape, concurrency, placement, frequency, and temperature;
- prediction intervals and observed nonlinear interactions;
- calibration/firmware/driver generation and expiry triggers;
- safe conservative defaults and refusal conditions.

Telemetry can convert hidden behavior into measured covariates—frequency throttling, thermal
state, contention, errors—but never turns an unobservable state into certainty. Learned
residuals propose or price candidates; deterministic limits and final verification remain
authoritative.

## 10. API, database, service, and IPC architecture

BCIR should split APIs by responsibility, not create microservices inside the optimizer:

| Plane | Operations | Runtime rule |
|---|---|---|
| command/write | submit schema, candidate, measurement, build, activation, cancellation | authenticated, idempotent identity, transactional |
| query/read | inspect artifacts, plans, certificates, status, telemetry evidence | immutable snapshot or generation |
| data | StreamPack, tensors, compiled artifacts, bulk telemetry | bounded binary channel, backpressure |
| control | leases, generations, capability negotiation, quiescence, rollback | small versioned messages |
| evidence | raw samples, provenance, logs, certificates | append-only/content-addressed |

An artifact registry and measurement store may use a database; execution must not depend on an
online query. A build coordinator, telemetry collector, or fleet calibration service may be a
microservice; the compiler library remains embeddable and deterministic. A service bus/event
hub may distribute immutable artifact and generation events, never replace direct
RuntimeChannel semantics. MCP can expose tools to an AI/operator control plane but has no place
in the privileged execution path or law definition.

Read/write separation is valuable as command-query separation and immutable snapshots. It is
not a reason to duplicate every low-level ABI call. The stable direct driver contract remains
open/claim/map/submit/sync/event/cancel/close, with explicit ownership and generations.

## 11. Driver, kernel, FPGA, and SASOS implications

### 11.1 Driver profiles

Each driver package should eventually contribute:

- authoritative register/packet/command schema and immutable physical layouts;
- assembler, disassembler, verifier, simulator, and differential oracle;
- direct RuntimeChannel binding and Linux/native adapters;
- topology/capability and measurement provider;
- bounded candidate library and code-generation constraints;
- telemetry registry mappings and replay corpus;
- target-response surfaces, generation triggers, and TMSAO certificates.

Profiles reduce impossible candidates early and provide realistic bounds. They do not permit
the optimizer to extrapolate beyond measured capacity or hide uncertainty.

### 11.2 Security boundary

Schema constraints and verified IR can eliminate classes of malformed layout and command
errors, but they do not alone provide Rust-like temporal safety, authenticity, DMA isolation,
interrupt safety, speculative-execution containment, or secure live replacement.

A same-address-space design requires:

- unforgeable capabilities;
- spatial and temporal memory safety;
- control-flow and code-integrity enforcement;
- DMA/IOMMU containment;
- bounded interrupt and fault behavior;
- authenticated artifacts and rollback protection;
- revocation, quiescence, and generation-safe replacement;
- hardware or independently verified enforcement when the compiler is compromised.

On conventional x86/Linux, rings, page tables, processes, IOMMUs, and signed modules remain
defense in depth. Memory protection keys can reduce page-permission switching cost, but they
are limited hardware resources and do not protect instruction fetch on x86
([kernel pkeys](https://cdn.kernel.org/doc/html/latest/core-api/protection-keys.html)).

FPGAs, capability machines, AI accelerators, and custom fabrics may expose a better match:
tagged memory, region capabilities, synthesized bounds, isolated DMA, and command processors
that accept verified StreamPack-like work. BCIR should target those through explicit capability
profiles, not infer that Unix privilege separation is obsolete before equivalent enforcement
exists.

### 11.3 JIT kernels

JIT compilation selects and instantiates preverified templates under a bounded plan. It does
not permit arbitrary runtime kernel-code generation in privileged space. New code is verified,
content-addressed, signed where required, installed as a new generation, activated only at a
quiescent boundary, and retained with a rollback path.

## 12. Prioritized implementation program

### P0 — definitions, validity, and a common artifact

1. Freeze the TMSAO scope and Level 1–4 certificate schemas.
2. Fix the stride benchmark validity defect; retain raw samples and environment attestations.
3. Add exact small-instance schedule and allocation oracles.
4. Define typed regions, candidate/equivalence graph, and canonical plan schemas.
5. Reconcile pricing, EFT, token execution, and scheduled lifetime terminology.
6. Pin all current performance claims to reproducible reports or mark them historical.

**Gate:** deterministic schemas and digests; old claim DAG round-trips unchanged; invalid
profiles and false bare-metal claims are refused.

### P1 — GEM+ integration

1. Implement opaque, affine, tensor-index, timed-event, SDF/CSDF, and state-machine region
   interfaces with conservative fallback.
2. Add the algebra/objective registry and hard-constraint hierarchy.
3. Build one C++23 canonical plan builder over existing MLIR optimizer passes.
4. Derive actual scheduled lifetimes and connect allocation/movement feedback.
5. Export transformation choices through MLIR Transform-compatible artifacts and vector
   candidates through a VPlan-compatible boundary where feasible.
6. Add lower-bound stack and exact-versus-HEFT/first-fit gap reports.

**Gate:** Python/C++ parity; legal incumbent under cancellation; exhaustive agreement on
bounded generated instances; no regression in existing StreamPack.

### P2 — incremental plans and heterogeneous memory

1. Add dependency-granular invalidation and persistent candidate/proof/frontier caches.
2. Add delta plans, parameterized templates, lazy hydration, and delta StreamPack generation.
3. Integrate HAM, Semantic Swap, prefetch, eviction, rematerialization, and transfer overlap
   into the canonical plan.
4. Add SYCL/CUDA/CPU provider capabilities with exact failure and generation ownership.
5. Benchmark library/generated/persistent candidates on controlled targets.

**Gate:** delta and clean rebuild are byte-identical; injected stale generations cannot run;
peak memory, traffic, and time certificates are independently checked.

### P3 — ASN.1 and C++ hosted compilation

1. Represent encoding family/profile/ECN identity as a typed serialization region.
2. Port immutable descriptor and transactional plan construction to C++ while retaining
   Python/C wire oracles.
3. Complete the ranked ASN.1 compiler-comparison gaps, beginning with real grammars and valid
   value generation.
4. Integrate measured encoding candidates into whole-plan movement and memory costs.
5. Build the C/Linux and schema-authored userspace/simulator driver comparison.

**Gate:** abstract-value and emitted-command parity; malformed JER/ECN cannot mutate output;
physical layouts remain exact; every selected encoding has canonical/provenance evidence.

### P4 — physical drivers and kernel evidence

1. Calibrate one native x86 CPU and the GTX 1660 SUPER separately, then jointly where counters
   support it.
2. Add QEMU functional rails for x86-64/AArch64/RISC-V without using them as performance truth.
3. Build UART and queue/DMA lifecycle drivers through the direct interface.
4. Run sequential/virtual-device Linux comparisons before kernel fork changes.
5. Promote proven contracts into BCIR-Linux and later a native kernel.

**Gate:** direct/Linux behavior parity; teardown, saturation, cancel, peer/device loss, restart,
hotplug, suspend/resume, and rollback tests; signed provenance and controlled physical
performance report.

## 13. Risk register and decision rules

| Risk | Consequence | Control |
|---|---|---|
| “master algorithm” overclaim | wrong solver and hidden global gap | region/solver registry and certificate level |
| state explosion | compile-time or memory failure | bounded budgets, decomposition, anytime legal incumbent |
| scalarization hides policy | security/accuracy traded for speed | hard constraints, lexicographic/Pareto objectives |
| driver profile treated as truth | unsafe extrapolation | intervals, generation invalidation, refusal outside support |
| schedule/memory drift | overlap aliases or excess traffic | one canonical plan and independent verifier |
| cache makes stale plan look valid | wrong target execution | full S identity and dependency invalidation |
| Python/C++ semantic fork | false parity and maintenance cost | byte/semantic differential gates before retirement |
| ASN.1 becomes mandatory hot path | latency and complexity | external-contract/encoding portfolio boundary |
| JER called binary/zero-parse | skipped validation and memory bugs | bounded scalar correctness rail; SIMD only as accelerator |
| ECN rewrites physical MMIO | device corruption | physical layouts immutable; optimize access/packet alternatives |
| QEMU/WSL called bare metal | invalid cost tables | provenance refusal and certificate downgrade |
| learned optimizer controls legality | nondeterminism and unsafe plans | learning proposes; verifier decides |
| SASOS removes defense in depth | one bug compromises whole machine | capabilities, hardware isolation, signed generations |

Admit a new abstraction, solver, library, serialization format, or service only when it:

1. serves a named region/workload not already covered;
2. has a precise semantic and ownership contract;
3. has a bounded fallback;
4. is compared on correctness, cost, uncertainty, and complexity;
5. improves the incumbent or closes a proof gap;
6. preserves dependency-free/freestanding paths where required.

## 14. Final answer

BCIR’s core is close to the right *shape* for TMSAO: legality-first claims, explicit resources,
target profiles, multiple realizations, measurement, and content-addressed artifacts. It is
not yet the theoretical maximum because its current planner decomposes decisions that interact
and its evidence lacks a complete lower-bound/gap and physical-measurement rail.

Tropical min-plus has achieved “master primitive” status for additive layered selection, not
“master algorithm” status for computing. GEM’s DAG remains the safest universal fallback, not
the best local representation for every region. GEM+ supplies the missing architecture by
combining typed local mathematics, a solver portfolio, joint plan state, lower bounds,
uncertainty, and replay.

The highest-value next move is therefore P0/P1: fix measurement validity, freeze the scope and
certificate contracts, define the candidate graph and canonical plan, and join schedule,
placement, memory, and movement in the existing MLIR C++23 rail. That improves the parts
without discarding the core.

## 15. Primary references

- MLIR, [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/) and
  [Affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- LLVM, [VPlan](https://llvm.org/docs/VectorizationPlan.html)
- COSMA, [joint scheduling, allocation, and tensor replacement](https://arxiv.org/abs/2311.18246)
- Lee and Messerschmitt,
  [static scheduling of synchronous dataflow](https://doi.org/10.1109/TC.1987.5009446)
- Willsey et al., [egg/equality saturation](https://arxiv.org/abs/2004.03082)
- Mirage, [multi-level tensor superoptimization](https://arxiv.org/abs/2405.05751)
- Event Tensor, [dynamic persistent-kernel abstraction](https://arxiv.org/abs/2604.13327)
- Nautilus, [auto-scheduling tiled GPU kernels](https://arxiv.org/abs/2604.14825)
- DeepSpeed, [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
- GraphBLAS, [standard algebraic graph building blocks](https://graphblas.org/)
- Linux kernel,
  [perf security](https://docs.kernel.org/6.2/admin-guide/perf-security.html),
  [CPU isolation](https://docs.kernel.org/admin-guide/cpu-isolation.html), and
  [powercap](https://docs.kernel.org/6.8/power/powercap/powercap.html)
- QEMU, [instruction-count limitations](https://www.qemu.org/docs/master/system/invocation.html)
- NVIDIA, [Nsight Compute support](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/gpu-support.html)
- AMD, [uProf performance analysis](https://www.amd.com/en/developer/uprof/uprof-performance-analysis.html)
