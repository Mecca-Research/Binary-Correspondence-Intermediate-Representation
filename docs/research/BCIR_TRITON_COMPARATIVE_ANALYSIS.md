# BCIR and Triton: architecture, integration, and a compiled CPU pilot

**Updated 2026-10-19. Research, not a capability or performance guarantee.**

BCIR's strongest opportunity is to couple program-level legality, placement, memory and
execution choices with high-quality kernel implementations. Triton's strongest delivered
capability is turning tile programs into efficient GPU kernels. These are different levels of
optimization. BCIR's native C training kernels have also advanced substantially since the
2026-07-04 version of this document; calling all of BCIR's ML execution a Python reference is
now inaccurate.

**Recommendation:** retain BCIR's C execution core and make Triton an optional resident compiler
and accelerator backend above that boundary. Establish a measured interface before considering
an upstream extension, a portable layout-algebra module, or an out-of-tree backend. A wholesale
fork is unnecessary at this stage. Neither system has been shown to subsume the other.

The companion [three-way architecture and IR analysis](BCIR_TRITON_PYTORCH_ARCHITECTURE_2026-10-19.md)
extends this study to PyTorch's current FX/ATen and Inductor representations, MLIR trade-offs,
runtime/driver boundaries and a competitive assessment blueprint. It adds source analysis,
including a direct IR comparison; it does not add performance measurements.

The subsequent [backend optimality test plan and kernel gates](BCIR_BACKEND_OPTIMALITY_AND_KERNEL_GATES_2026-10-20.md)
specify eager/compiled/IR/kernel/runtime comparisons, target qualification, telemetry and
driver/IPC/boot entry criteria. Pending #820 has since progressed through PX-2b pack/plan
execution at `b6c50f1`; this CPU pilot remains pinned to main and excludes that pending work.
The [data/AI driver and IPC qualification report](BCIR_DRIVER_IPC_AI_QUALIFICATION_2026-10-20.md)
extends those gates with per-device workload evidence and the physical HAM/native SYCL sequence.

**Scope of this CPU study:** architecture and a few initial kernel comparisons. Full
comparative model training remains **deferred until PR #820 is completed and merged**. No
PyTorch performance comparison or full-model training experiment was performed for this study.

## 1. Evidence boundary and version pins

| Item | Examined version / status |
|---|---|
| BCIR measured source | `12aa6dc5ba422418742f49e4bfc00aafdc17e07f` on `main`; no runtime modifications |
| Production Triton | Latest published upstream release checked: **3.8.0**, 2026-08-28; source `c01b6774b1865984607d89d89d3a10833de92037` [T1] |
| Executed Triton backend | Official **experimental Triton-CPU 3.8.0+cpu** snapshot, published 2026-09-22; source `5a4885bf9b0b349a525b490a1df25f449f12d4e9` [T2] |
| CPU release LLVM pin | `62b7cf9623fc310525f39ed69aaecc318a909731`, build 2; distinct from BCIR's newest-toolchain validation requirement |
| PR #820 | Open, draft, unmerged at inspection; head `af1d935be77854c0d0fee2b8846714281864cd7c`; base is the measured BCIR pin [B8] |
| Execution host | AMD EPYC 9V74 VM, LLVM identifies `znver4`; GCC 13.3.0, Python 3.12.14, NumPy 2.3.5; logical CPUs 0–1 used |
| Available hardware | CPU only. No CUDA, ROCm, SYCL device execution, GPU, AMX measurements or PMU evidence |

The CPU wheel was downloaded from the official release and its bytes verified against the
release asset's SHA-256:
`693f657501e6d791e889d5fd880d3b84a9bf4d8f3087dbb685534bb8533361fa`.
The two Triton repositories/releases must not be conflated: the CPU snapshot is useful for
this overlap experiment, but it does **not** represent the performance of Triton's production
NVIDIA/AMD backends. A normal `pip install triton` is not this CPU snapshot.

Evidence is separated throughout:

- **Implemented:** inspected at the pinned source, with existing tests where named.
- **Measured here:** the reproducible CPU pilot in §6, with raw samples and numerical checks.
- **Pending:** PR #820's design and reported measurements, not re-measured or treated as merged.
- **Proposed:** integration and optimization work in §§8–9; not new runtime functionality.

## 2. Where optimization happens

Triton's main authoring model is a Python-embedded, explicitly tiled SPMD kernel DSL. A program
instance operates on blocks of values, using pointers, masks, reductions and dot operations.
The compiler maps those logical blocks to the target's execution and memory resources. Gluon
provides a lower-level authoring route with explicit layouts, shared memory, warp specialization
and target-specific operations [T3]. Python authoring does not mean the device computes those
operations in Python.

For the inspected NVIDIA backend, the important pipeline is:

`Triton DSL → TTIR → TritonGPU IR → LLVM IR → PTX → ptxas → cubin`.

AMD has its own target lowering and device-image pipeline. Triton owns substantial
layout assignment, tensor-core selection, shared-memory handling, pipelining and synchronization
logic **before** the final target compiler/assembler handoff. Delegating instruction selection
does not make this work redundant. Triton-CPU instead lowers through its CPU dialects to
LLVM and a native shared object, with a C++/OpenMP launcher [T4].

BCIR has several cooperating execution paths, rather than one already unified compiler for
all tensor computation:

| Layer | BCIR today | Triton today | Architectural implication |
|---|---|---|---|
| Program representation | Registry/resources, typed claims and phases; Python conformance oracle plus compiled MLIR law rail | Tile kernels in TTIR and target-oriented IR; host integrations compose kernels | Compare a graph planner with a graph planner, and a kernel with a kernel |
| Legality | R-law checks, explicit contracts, cross-rail and negative witnesses | Operation verifiers, analyses, compiler tests and specialized checking tools | Different assurance scopes; neither is a proof that every generated machine instruction is correct |
| Candidate selection | Deterministic integer cost vectors, budgets, bounded/exact searches and learned ordering | Compiler heuristics; explicit launch/tile configurations; optional measured autotuning and pruning | Analytical screening and hardware measurement can cooperate |
| Scheduling and memory | ExecutionPlan placements, generations, lifetimes and moves; exact small-scope options and fallbacks | Intra-kernel layout, shared-memory allocation, synchronization and pipelining; caller controls graph execution | Whole-program arenas and per-kernel shared memory solve different problems |
| Native ML execution | C tensor kernels, deterministic pool, decoder forward/backward, optimizer and caller-owned arenas | User-written kernels; kernel libraries and host/framework integrations | The native C runtime is now a meaningful implemented comparison surface |
| GPU execution depth | Elementwise PTX emission checks; SYCL dispatch seam with portable fallback and optional device path | Executable NVIDIA/AMD backends, tensor operations, asynchronous movement and profiling | Producing PTX text or running a C++ fallback is insufficient GPU qualification |
| Integration boundary | C core; selective C++ framework/accelerator boundary; optional Python tooling | Python frontend/runtime plus C++/MLIR compiler and vendor/runtime interfaces | Triton can be optional tooling without becoming a core Python dependency |

Relevant BCIR anchors: [oracle/law parity](../PARITY.md),
[cost vectors](../../bcir/kbcir/cost.py),
[ExecutionPlan](../../bcir/gem/execution_plan.py),
[native tensor contracts](../../runtime/c/bcir_tensor.h), and
[native decoder](../machine-learning/BCIR_NATIVE_DECODER_TRAINING.md).

**An important integration gap:** the native decoder's direct C loop does not issue a
StreamPack claim for every tensor operation. Its program description and arena counts do not
establish that every K_BCIR transformation already selects or rewrites each native kernel.
This pilot directly calls existing C kernels; it does not measure GEM+ scheduling, StreamPack,
ASN.1, or BCIR-generated MLIR code. Those require separate interventions and measurements.

## 3. The differences that matter beyond vendor support

### 3.1 Layout is executable mathematics

Triton's LinearLayout machinery represents maps between bit-indexed execution/storage
coordinates and logical tensor coordinates over **GF(2)**. This enables reasoning about
layout conversion, broadcasts and replication as well as distributed register/shared-memory
layouts [T5]. Gluon exposes part of this control to the author.

BCIR's [SoA/AoS planner](../../bcir/kbcir/layout.py) and integer-affine access maps address useful
but different layout questions. An integer-affine address `a*i+b` does not automatically
represent a bitwise XOR bank swizzle. Neither representation is generally a superset of the
other. Also, `B^-1 A` is valid only when the required inverse exists: real layouts can be
non-injective because of replication. Triton's implementation explicitly handles preimages
and `invertAndCompose`; a BCIR port must preserve that distinction.

A portable BCIR extension should distinguish integer-affine maps, GF(2) maps, masks/tails,
replication and physical address spaces. It then needs executable conversions, legality
witnesses and a conversion-cost model. Merely storing a layout descriptor does not remove a
shuffle, shared-memory round trip, transpose, or redundant operand pack.

### 3.2 Kernel fusion and whole-program fusion are different

Triton makes custom fused tensor programs relatively direct to express. The actual cost includes
register pressure, shared-memory footprint, occupancy, spills, instruction dependencies and
padded work. A larger fused kernel can be slower than two smaller ones.

BCIR can make fusion, placement, lifetime reuse and movement decisions at a broader level.
However, this advantage reaches hardware only when the selected graph has executable kernels
and the implementation exposes resource feedback. Eliminating graph nodes without generating
a correspondingly fused implementation does not eliminate launches or memory traffic.

The useful coupling is: BCIR proposes a legal partition; Triton compiles each candidate;
compiled resource use and measured latency feed back into selection. Reject infeasible resource
budgets before execution. Keep a measured cost ranking separate from the legality verdict.

### 3.3 Cost models, search and the meaning of optimal

BCIR's [tile prior](../../bcir/kbcir/tile_prior.py) orders a declared search and can stop under an
admissible bound. Its certificate concerns the modeled candidate set and cost function. The
[execution scope](../../bcir/kbcir/scope.py) identifies what such a claim is about. Current
[exact_plan](../../bcir/gem/execution_plan.py) can carry a solver's improved placement into the
plan that runs, subject to placement and memory-binding checks; the July document predates
this work.

That does not prove that the model's chosen tile is fastest on a GPU, or that separate schedule
and memory optima form a joint optimum. BCIR still needs calibrated target profiles and evidence
that selected decisions reach the executed implementation.

Triton's autotuner benchmarks configured candidates, caches choices in process, can persist
results with `cache_results`, and provides pruning/performance-model hooks. The 3.8 release adds
selection/timing/cache-status listener support [T1, T6]. It is inaccurate to say it always pays
for tuning again, or that Triton contains no performance modeling. Conversely, a best measured
configuration is not a proof over every possible kernel.

A sensible BCIR policy is to use analytical bounds to reduce the candidate set, then spend a
bounded measurement budget where uncertainty can change the decision. Record calibration
uncertainty and amortize compilation/tuning over the expected workload horizon.

### 3.4 Numerical contracts are part of the architecture

BCIR's native C library now has separate rounded-product FP32, fused FP32, blocked fused FP32,
and BF16 contracts, plus vector exponential and attention variants. Its deterministic pool
keeps each contract's operation order independent of the worker count [B1]. These are
implemented capabilities, not a promise of all-purpose bit parity with external kernels.

Triton supports explicit dot input precision and mixed-precision device operations. Its dot
API documents TF32 behavior and possible truncation; comparisons must name `input_precision`,
accumulator type, fusion settings and backend [T7]. FP32 storage alone is insufficient to
establish matched arithmetic.

The pilot admits a **tolerance-based FP32 comparison**: no fast math, no quantization or TF32;
FMA is allowed for GEMM. RMS reductions and exponentials can differ, and GEMM accumulation
order is not asserted identical. A backend replacing a BCIR contract needs either exact
contract conformance or an explicitly admitted approximation contract. Passing this study's
finite test inputs does not qualify it as an exact provider.

### 3.5 Verification and reproducibility need narrower claims

Triton 3.8 includes FpSan, broader ConSan support and experimental GSan [T1]. FpSan compares
symbolic computation using instrumented payload algebra; it is **not** an IEEE rounding,
NaN, exception or subnormal oracle [T8]. Its existence corrects the old document's blanket
“no verification” claim without turning it into a proof of general compiler correctness.

Similarly, BCIR law checks and dual-rail witnesses establish specific invariants under their
stated domains. They do not imply every optimized native/device binary has a general semantic
proof. Both need negative witnesses, numerical stress cases and real backend execution.

Triton's compilation cache has source, compiler/backend, option and environment identities;
it is not an unversioned opaque cache [T9]. BCIR offers a broader plan/generation/artifact
record. Its [BCAB envelope](../kernel/BCIR_ARTIFACT_BUNDLE_ABI.md) can carry unchanged standard
images and exact-byte hashes. CRCs and hashes must not be described as authenticity signatures.
A cached compiled kernel and an auditable application-level execution plan remain different
artifacts.

## 4. What has changed on the BCIR side

The pinned tree has native SIMD/FMA/blocked GEMM implementations, BF16 packing and AMX admission
on supported hosts, vector nonlinearities, pooled forward/backward kernels and complete native
decoder training. The [training roadmap](BCIR_TRAINING_ACCELERATION_ROADMAP.md) and the
[native runtime contracts](../../runtime/c/bcir_tensor.h) describe these surfaces; their prior
model-performance results are not reproduced as Triton comparisons here.

The GEM+ representation has also broadened: [regions](../../bcir/kbcir/regions.py) now recognize
affine, synchronous/cyclo-static dataflow, timed, 2-D nest and opaque cases. Exact scheduling
and [schedule-bound static memory](../../bcir/kbcir/static_memory.py) are implemented within
bounded scopes. These facts supersede old gap lists, while preserving the distinction between
recognized mathematical structure and target-specific kernel generation.

GPU depth remains the main execution gap, not merely a sponsorship gap:

- [`gem.matmul_buffer` lowering](../../mlir/lib/passes/BCIRLowerGemMatmulBufferPass.cpp) builds
  tiled scalar `scf.for` loops with loads/stores and multiply/add; it does not implement a
  tensor-core/shared-memory GPU pipeline.
- The [`nvptx64` target](../../bcir/codegen/targets.py) describes PTX text validation, including
  an assembly marker. That is not a device-resident training backend.
- The [SYCL seam](../kernel/SYCL_INTEROP.md) has real host dispatch and differential coverage,
  with explicit `sycl-device` / `fallback` / `unavailable` modes. Device submission still needs
  its toolchain and hardware; a successful fallback must retain its label.
- BCAB's ability to hold device images does not supply their compiler, driver, loader or
  synchronization implementation.

### PR #820: relevant, pending, and excluded from this benchmark

The inspected draft adds per-layer activation recomputation, a shared recompute-cache slot,
exact BF16 group-workspace accounting, and a budgeted choice of recomputation/workspace tiers
(state ABI 5). It distinguishes workspace **floor**, **panel threshold**, and **full need**.
This is directly relevant to cross-kernel memory planning and to keeping operands in useful
packed forms [B8].

At the 2026-10-20 refresh, pending head `b6c50f1` also has PX-1 launch witnessing, PX-2a
checked program execution and PX-2b StreamPack/ExecutionPlanV1 lowering into that executable
program. PX-3–PX-5 remain open at that pin; the
[qualification report](BCIR_DRIVER_IPC_AI_QUALIFICATION_2026-10-20.md#1-current-source-and-continuity-evidence)
retains the source and status boundary. None of this changes the pilot's measured source.

Its reported memory/performance frontier is evidence from that PR's own host and protocol.
It was not checked out, run, or blended into the CPU measurements below. A future Triton
adapter should expose its own scratch/persistent-pack needs to this planner rather than hiding
allocations behind an opaque kernel call. Full-model conclusions wait for the merged source.

## 5. Comparison dimensions that remain unmeasured

| Question | Present evidence | Required comparison |
|---|---|---|
| GPU tensor-core throughput | Triton implementation/source; no local GPU | Same GPU, precision, resident inputs and permitted math; executed BCIR GPU path |
| GPU fusion/occupancy tradeoff | Architecture analysis | Fusion partitions, registers/spills, shared memory, occupancy and total latency |
| Whole-model training | Intentionally deferred | Merged #820, identical recipe/initialization, forward/backward/update and held-out quality |
| Cross-kernel arenas / recomputation | BCIR implementation; #820 pending | Match memory cap, recompute policy and live allocations across complete executions |
| Transfers and asynchronous overlap | BCIR model/seams and Triton mechanisms | Real transfer events, fences, ownership and makespan on the same device |
| ASN.1 / StreamPack contribution | Not exercised by the pilot | Isolated plan-load/dispatch/replay intervention, including amortization |
| ARM / Snapdragon / distributed scale | No such hardware used | Separate native target/driver and communication qualification |

ASN.1 and StreamPack can reduce control-plane parsing, improve validated transfer/dispatch, or
carry a compact plan. They do not automatically reduce the cost of a floating-point multiply.
Tensors still need layouts and residency that the selected kernel actually consumes. Likewise,
UEFI independence and C23 adoption do not establish GPU throughput without an execution path.

## 6. Executed pilot: native C vs experimental Triton-CPU

The reproducer is [`tools/perf/triton_cpu_compare.py`](../../tools/perf/triton_cpu_compare.py).
Raw evidence: [session A](measurements/2026-10-19/triton-cpu/session-a.json),
[session B](measurements/2026-10-19/triton-cpu/session-b.json), and
[provenance / reproduction](measurements/2026-10-19/triton-cpu/README.md).
The [archived harness](measurements/2026-10-19/triton-cpu/measured-harness.py.txt) matches the
hash in both sessions. The current reproducer adds explicit UTF-8/LF output settings to the
two JSON writes; its kernel and timing code are unchanged.

The protocol uses ten shapes at one and two workers, including non-power-of-two tails. Each
Triton configuration is compiled and checked before timing; vectors try blocks 256/1024/4096,
GEMM tries three tile configurations, and RMS uses the next-power-of-two row width. Five tuning
batches select the Triton configuration; **21 separate randomized paired measurement rounds**
then compare it with the existing BCIR implementations. No new optimized C kernel was written.

GCC builds the existing tensor unit with `-O3 -fno-fast-math -ffp-contract=off -pthread`; explicit
BCIR FMA kernels still execute FMA. Triton uses the CPU backend, `ukernels=None`, no external
BLAS, `enable_fast_math=False`, and fusion only for GEMM. Its default SLEEF vector-library
option is retained. Both sides use preallocated NumPy-backed FP32 buffers and cached integer
addresses. Timing includes synchronous host launch and the called kernel; it excludes buffer
allocation, reference calculation, reset, compilation and tuning. Triton's bound compiled
launcher is used rather than re-running Python JIT specialization on every sample.

These are repeated-buffer warm timings, not streaming DRAM bandwidth or kernel-only device
timings. The CPU affinity is shared by the two implementations. `OMP_WAIT_POLICY=PASSIVE`,
`GOMP_SPINCOUNT=0`, and a 2 ms pause outside each timed batch allow inactive pools to park.
Two-worker results therefore describe this coexistence profile; they do not claim optimal
standalone OpenMP settings. A preliminary smoke run exposed severe pool interference with
spinning defaults; those exploratory timings are excluded. Pointer preparation was also moved
outside both timed boundaries before freezing **protocol v2**, used by both published sessions.

The C rows use vector exponential **v2** for SwiGLU and **kblock** for GEMM. The libm SwiGLU and
sequential fused GEMM alternatives remain in the raw files. These are explicit arithmetic
choices, not silent substitutions. The study does not include XSMM/oneDNN tuning, expert
hand-optimized Triton CPU kernels, production GPU kernels, autograd or model training.

### 1 worker

| Workload | BCIR µs, A / B | Triton-CPU µs, A / B | BCIR/Triton ratio [95% interval], A / B |
|---|---:|---:|---|
| Residual add 4,096 | 1.23 / 1.17 | 2.83 / 2.41 | 0.433 [0.396, 0.472] / 0.486 [0.453, 0.509] |
| Residual add 1,048,579 | 195.68 / 181.76 | 194.12 / 185.94 | 1.008 [0.933, 1.085] / 0.978 [0.909, 1.037] |
| SwiGLU 4,096 | 3.96 / 3.79 | 5.00 / 4.31 | 0.791 [0.750, 0.902] / 0.879 [0.848, 0.926] |
| SwiGLU 1,048,579 | 942.14 / 863.30 | 1213.06 / 901.70 | 0.777 [0.713, 0.901] / 0.957 [0.873, 1.048] |
| RMSNorm 64 × 128 | 3.95 / 3.13 | 9.61 / 8.25 | 0.411 [0.294, 0.569] / 0.380 [0.343, 0.418] |
| RMSNorm 256 × 257 | 18.09 / 20.58 | 122.97 / 141.42 | 0.147 [0.139, 0.152] / 0.146 [0.140, 0.169] |
| RMSNorm 128 × 1,024 | 43.69 / 54.70 | 143.14 / 159.80 | 0.305 [0.297, 0.312] / 0.342 [0.324, 0.361] |
| GEMM (M,N,K) 128 × 128 × 128 | 53.34 / 52.11 | 76.30 / 74.19 | 0.699 [0.635, 0.752] / 0.702 [0.673, 0.731] |
| GEMM (M,N,K) 256 × 256 × 256 | 420.65 / 391.79 | 644.91 / 515.26 | 0.652 [0.601, 0.830] / 0.760 [0.713, 0.798] |
| GEMM (M,N,K) 129 × 257 × 255 | 205.17 / 189.17 | 371.79 / 343.09 | 0.552 [0.494, 0.617] / 0.551 [0.521, 0.581] |

### 2 workers

| Workload | BCIR µs, A / B | Triton-CPU µs, A / B | BCIR/Triton ratio [95% interval], A / B |
|---|---:|---:|---|
| Residual add 4,096 | 1.09 / 1.13 | 4.45 / 4.63 | 0.245 [0.238, 0.255] / 0.245 [0.234, 0.257] |
| Residual add 1,048,579 | 115.05 / 153.19 | 146.87 / 156.43 | 0.783 [0.653, 1.067] / 0.979 [0.772, 1.007] |
| SwiGLU 4,096 | 3.98 / 3.86 | 6.47 / 6.09 | 0.615 [0.582, 0.649] / 0.633 [0.594, 0.682] |
| SwiGLU 1,048,579 | 541.74 / 519.46 | 641.92 / 506.78 | 0.844 [0.719, 1.058] / 1.025 [0.874, 1.164] |
| RMSNorm 64 × 128 | 3.05 / 2.99 | 23.66 / 22.48 | 0.129 [0.126, 0.130] / 0.133 [0.114, 0.133] |
| RMSNorm 256 × 257 | 13.28 / 15.84 | 91.81 / 84.44 | 0.145 [0.125, 0.168] / 0.188 [0.142, 0.191] |
| RMSNorm 128 × 1,024 | 40.73 / 27.89 | 101.96 / 97.05 | 0.399 [0.298, 0.544] / 0.287 [0.272, 0.302] |
| GEMM (M,N,K) 128 × 128 × 128 | 42.93 / 41.09 | 71.61 / 70.97 | 0.600 [0.518, 0.687] / 0.579 [0.481, 0.716] |
| GEMM (M,N,K) 256 × 256 × 256 | 209.89 / 233.12 | 313.03 / 368.94 | 0.671 [0.502, 0.909] / 0.632 [0.529, 0.730] |
| GEMM (M,N,K) 129 × 257 × 255 | 112.23 / 182.01 | 309.12 / 334.87 | 0.363 [0.334, 0.540] / 0.544 [0.541, 0.551] |


A ratio below one means the named BCIR C kernel had lower measured latency in that row.
Intervals are paired bootstrap intervals over rounds, descriptive for this shared VM. They do
not capture all cross-session variation, and no wall-time threshold is proposed as a CI gate.
There is no geometric-mean “BCIR beats Triton” headline across unrelated shapes/backends.

Across two sessions: **40 workload/worker cases**, 96 candidate compilations/launches,
and all numerical checks passed. The maximum absolute errors against float64 across both
implementations and sessions were: residual add 2.38e-07, SwiGLU 5.01e-07,
RMSNorm 1.67e-06, and GEMM 1.35e-05. These are observed errors, not universal bounds.

The measured BCIR C library build took 6.55 / 6.29 s (A / B).
Triton's summed first-call times across the declared candidate set were
26.04 / 25.03 s. These include specialization/compilation/loading/first execution and
are **not** an equal-work compiler-speed comparison. Tuning and repeated-measurement costs are
separate from those first calls. All selected GEMM configurations were 16 × 32 × 32; this
small candidate set is not a globally best Triton CPU implementation.

All published candidate and final-output checks passed against float64 references under the
recorded per-element tolerances. Residual add is checked more tightly than the transcendental
and reduction paths. The raw files retain maximum absolute errors, scale-normalized errors,
BCIR–Triton differences, the inputs' seeds, chosen configurations, every timing observation,
compiler metadata and selected IR hashes. They do not establish bit parity, gradient accuracy,
recall, language-model quality or behavior over all non-finite inputs.

Memory reporting is deliberately limited to known input/output spans and BCIR pool storage.
It excludes compiler caches, stack/register spills, launcher allocations, allocator arenas and
process RSS. **No measured peak-memory superiority is claimed from these kernel tests.**

## 7. What the pilot suggests, and what to improve

1. **Keep the native C path.** Its fixed-call overhead and specialized reductions are useful
   CPU assets. The Triton-CPU launcher source allocates a grid-coordinate array for multi-program
   launches and uses OpenMP; BCIR's caller-owned pool has a different overhead profile [B1, T4].
   This is a plausible contributor to the small-call gap, not an isolated causal measurement.
2. **Use the library's optimized contracts deliberately.** The existing vector exponential is
   much faster than the libm SwiGLU route on these inputs. Retain both numerical contracts and
   their accuracy tests; select explicitly rather than globally enabling fast math. The blocked
   GEMM deserves comparison with the sequential fused variant, not just the original rounded-
   product implementation.
3. **Treat large elementwise work as a bandwidth/launch problem.** The largest residual-add
   case is much closer than small-call RMS. Test cache-size crossings, alignment, NUMA placement,
   independent buffers and matched idle policy before changing the kernel algorithm.
4. **Improve Triton CPU authoring before generalizing a gap.** Row grouping, explicit tiling,
   masks and vectorization choices can matter greatly. Three GEMM tiles and a row-wise RMS kernel
   are an initial search, not a survey of the best possible CPU implementations. Compile and
   launch overhead need separate attribution.
5. **Make the graph/native connection measurable.** Publish which selected layouts, packs,
   fused groups and arena decisions actually reach C calls. Reuse packed operands across legal
   consumers, price transposes and tails, and account for pool barriers. PR #820's shared-panel
   and workspace accounting are relevant groundwork, not a measured gain in this study.
6. **Use a fresh target-specific GPU study for Triton's core strength.** The CPU pilot provides
   no quantitative estimate of the gap on tensor-core GEMM, Flash-style attention, low-precision
   arithmetic, persistent kernels or device-side asynchronous transfers.

## 8. Integration blueprint: make the two optimization levels cooperate

A proposed Triton backend should consume a bounded kernel contract and return a compiled
artifact plus enough metadata for BCIR to price and execute it safely. The current synchronous
C provider ABI is not a complete asynchronous device ABI: host pointers cannot be silently
reinterpreted as device addresses, and completion/ownership cannot disappear at the boundary.
Keep allocation/driver/framework integration above the C core, using C++ where the vendor API
requires it.

| Contract area | Required fields / behavior | Why it matters |
|---|---|---|
| Semantics | Operation, shapes/strides, input/output alias rules, masks, side effects, precision/accumulation/reduction contract, admitted tolerance | A faster variant must implement the admitted operation |
| Target | Architecture/features, address spaces, driver/runtime/compiler versions and compatibility | One artifact is not a universal GPU binary |
| Resources | Alignment, input/output residency, shared/global scratch, persistent packs, register/spill metadata where available | The graph planner must see hidden memory and occupancy costs |
| Execution | Grid, launch parameters, stream/event dependencies, completion and error semantics | Supports actual overlap without changing ordering or lifetime laws |
| Identity | Source/configuration/backend/compiler hashes, output artifact digest, plan/scope/generation binding | Separates repeatable compilation from auditable application execution |
| Calibration | Cold compilation/tuning, warm latency, transfers, uncertainty, workload horizon | Supports honest amortization and bounded empirical refinement |

Use [BCAB](../kernel/BCIR_ARTIFACT_BUNDLE_ABI.md) to carry compatible standard images unchanged;
do not invent a replacement device object format. An artifact hash is byte identity, while a
law witness is semantic evidence: the adapter must carry both without claiming they are the
same thing.

The optimization target is a coupled decision over **partition, backend, precision, layout,
packing, schedule, workspace and recomputation**, subject to legality and resource budgets.
A useful objective is measured makespan at an admitted memory/quality envelope, with compile
cost amortized over repeated use. Optimizing each variable independently can miss a jointly
better program. This is a development direction, not a claim that the present implementation
already solves the full joint search.

## 9. Prioritized work, with acceptance evidence

| Priority | Work | Acceptance evidence |
|---|---|---|
| 1 | Freeze this source-backed comparison and CPU pilot | Reproducer, pins, raw samples, numerical contracts and explicit hardware skips |
| 2 | Minimal optional Triton resident-backend prototype | One real kernel, typed contract, deterministic artifact identity, refusal of incompatible shape/precision/target, and correct execution on actual hardware |
| 3 | Measure composition overhead separately | Standalone Triton vs BCIR dispatching the **same** Triton artifact; distinguish launch/transfer/plan cost from kernel code quality |
| 4 | Connect graph decisions to executed native/device kernels | Trace selected fusion/layout/pack/arena into launches; demonstrate one intervention improves total latency at the same memory and quality contract |
| 5 | Add portable GF(2) layout reasoning alongside integer-affine maps | Broadcast/non-injective cases, inverse preconditions, masked tails, executable conversions and cross-rail witnesses |
| 6 | Couple analytical selection with bounded hardware calibration | Held-out shapes, prediction/regret/error record, compilation horizon and a legal fallback on stale/missing measurements |
| 7 | Expand target-specific kernel portfolio | GEMM/transposes/reductions/nonlinearities, then attention forward/backward, low precision and long-context memory under explicit contracts |
| 8 | Full training comparison after #820 merges | Protocol in §10; no reuse of old training ratios as Triton evidence |

`triton-shared`/standard-MLIR ingestion can be a later, explicitly subsetted experiment. Lowering
a Triton program to generic linalg does not preserve every GPU layout, asynchronous instruction
or hardware-specific semantic. An out-of-tree backend is a larger compiler-maintenance
commitment than calling a resident backend; it should follow a demonstrated missing requirement.
A fork is justified only by a concrete blocked capability with a funded maintenance plan.

## 10. Deferred full-training protocol

After PR #820 is complete and merged, freeze the merged BCIR SHA and the Triton target/toolchain
before any model comparison. Triton is a kernel compiler, not a complete training framework;
there must be an explicitly described host/autodiff/optimizer harness around the comparator.
Keep PyTorch comparisons shelved unless separately requested.

Use two experiments, because they answer different questions:

- **Matched recipe:** identical model, initialization, batches, optimizer, update count,
  precision/approximation contract and memory cap. Check forward outputs, backward gradients,
  optimizer/state evolution and held-out quality before timing. This tests execution machinery.
- **Best admitted recipe per system:** equal hardware, memory, tuning budget and quality target,
  allowing documented architecture/layout/recompute differences. Report quality versus time
  and memory frontiers. This tests what each ecosystem can build, without attributing a model
  change to a faster compiler.

First isolate kernel forward/backward and update components. Then measure complete steps,
including synchronization and real movement, with precompiled warm throughput **and** cold
start/compilation. Record live arena/scratch/packing memory separately from process RSS and
external-library caches. Use multiple paired seeds for learning/quality and repeated randomized
timing rounds. Include irregular and variable lengths, finite/non-finite numerical stress,
checkpoint/resume, and memory-constrained recomputation. Never compare CPU latency with GPU
latency as evidence of compiler superiority.

On a GPU, BCIR orchestrating Triton kernels may beat another graph plan while using the same
compiler. Attribute that result to partitioning, scheduling, memory or movement as demonstrated;
it does not mean Triton's own generated machine code was outperformed by an unrelated backend.

## Sources and implementation anchors

All source references were inspected for this update. BCIR relative links refer to the measured
base unless explicitly labeled pending. Upstream source links are release-pinned; live docs
are supplemental and may move.

- **[T1]** [Triton 3.8 release notes](https://github.com/triton-lang/triton/releases/tag/v3.8.0),
  [pinned NVIDIA compiler pipeline](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/third_party/nvidia/backend/compiler.py),
  [AMD pipeline](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/third_party/amd/backend/compiler.py).
- **[T2]** [Experimental Triton-CPU release](https://github.com/triton-lang/triton-cpu/releases/tag/v3.8.0),
  [CPU source pin](https://github.com/triton-lang/triton-cpu/tree/5a4885bf9b0b349a525b490a1df25f449f12d4e9),
  [LLVM pin](https://github.com/triton-lang/triton-cpu/blob/5a4885bf9b0b349a525b490a1df25f449f12d4e9/cmake/llvm-info.json).
- **[T3]** [Gluon overview](https://triton-lang.org/main/gluon/index.html),
  [layout tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html).
- **[T4]** Triton-CPU [compiler/options](https://github.com/triton-lang/triton-cpu/blob/5a4885bf9b0b349a525b490a1df25f449f12d4e9/third_party/cpu/backend/compiler.py)
  and [driver/launcher](https://github.com/triton-lang/triton-cpu/blob/5a4885bf9b0b349a525b490a1df25f449f12d4e9/third_party/cpu/backend/driver.py).
- **[T5]** [LinearLayout definition and GF(2) mathematics](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/include/triton/Tools/LinearLayout.h).
- **[T6]** [Autotuner implementation](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/python/triton/runtime/autotuner.py),
  [autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html).
- **[T7]** [dot precision API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).
- **[T8]** [FpSan semantics and limitations](https://triton-lang.org/main/programming-guide/chapter-3/fpsan.html),
  [debugging and sanitizers](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html).
- **[T9]** [Compilation pipeline/cache lookup](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/python/triton/compiler/compiler.py),
  [cache keys](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/python/triton/runtime/cache.py).
- **[B1]** [C tensor implementation](../../runtime/c/bcir_tensor.c) and [contracts](../../runtime/c/bcir_tensor.h).
- **[B2]** [Scope](../../bcir/kbcir/scope.py), [cost](../../bcir/kbcir/cost.py),
  [tile prior](../../bcir/kbcir/tile_prior.py), [region carrier](../../bcir/kbcir/regions.py).
- **[B3]** [ExecutionPlan](../../bcir/gem/execution_plan.py),
  [plan ABI](../../bcir/abi/execution_plan_abi.py), [static memory](../../bcir/kbcir/static_memory.py).
- **[B4]** [MLIR matmul lowering](../../mlir/lib/passes/BCIRLowerGemMatmulBufferPass.cpp),
  [codegen targets](../../bcir/codegen/targets.py).
- **[B5]** [SYCL interop](../kernel/SYCL_INTEROP.md),
  [resident dispatcher](../../bcir/lower/sycl_dispatch.py).
- **[B6]** [BCAB](../kernel/BCIR_ARTIFACT_BUNDLE_ABI.md),
  [C++ handoff boundary](../languages/CPP_HANDOFF_BOUNDARY.md).
- **[B7]** [Native training](../machine-learning/BCIR_NATIVE_DECODER_TRAINING.md),
  [training program description](../../bcir/hosted/models/native_program.py).
- **[B8]** **Pending:** [PR #820](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/pull/820),
  inspected at `af1d935be77854c0d0fee2b8846714281864cd7c`. Its findings are not CPU-pilot results.
