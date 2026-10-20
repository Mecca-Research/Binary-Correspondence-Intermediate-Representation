# BCIR, Triton and PyTorch: representations, hardware control and an optimization strategy

**2026-10-19; test/driver blueprint extended 2026-10-20 UTC. Source-based architectural research; no new performance experiment.**

BCIR has a credible systems opportunity: make semantic legality, graph partitioning, kernel
selection, memory, recomputation and dispatch cooperate through explicit contracts. Its native
C kernels already justify further investigation. The available measurements do **not** yet
establish a superior IR or a performance ceiling above Triton and PyTorch. That requires showing
which compiler decisions changed the executed program and why those changes improved it.

The strongest current competitors are more capable than several premises of this comparison
suggest. Triton has substantial target-specific MLIR optimization, including register layouts,
tensor-core lowering and asynchronous pipelines. PyTorch has multiple IRs, a graph compiler,
memory planning and replaceable backends; it can use Triton inside its compiled execution path.
Consequently, the interesting comparison is between particular layers and complete execution
paths, with BCIR's broader integration tested as a hypothesis.

This is a companion to the [BCIR–Triton analysis and CPU pilot](BCIR_TRITON_COMPARATIVE_ANALYSIS.md).
It extends the architecture discussion to PyTorch without reopening full model benchmarking.

The [backend optimality test plan and kernel entry gates](BCIR_BACKEND_OPTIMALITY_AND_KERNEL_GATES_2026-10-20.md)
now specifies comparator routes, sixteen experiments, the gap register, transformation-admission
rules and parallel driver/IPC/boot milestones. It also records pending #820's newer PX-1 trace
work and the PX-2–PX-6 follow-up ladder.

## 1. Evidence, versions and scope

| Project | Inspected source | Evidence boundary |
|---|---|---|
| BCIR | `12aa6dc5ba422418742f49e4bfc00aafdc17e07f`, the main revision used by the CPU pilot | Python conformance oracle, compiled MLIR law/optimizer, C runtime and training paths; these are distinct execution surfaces |
| Triton | Published release 3.8.0, `c01b6774b1865984607d89d89d3a10833de92037` | Production NVIDIA/AMD source inspected; no GPU execution on this study host |
| Triton-CPU | Experimental 3.8.0+cpu, `5a4885bf9b0b349a525b490a1df25f449f12d4e9` | Only the separately documented CPU pilot was executed |
| PyTorch | Latest published release checked: 2.14.1, `5c4886908584029761b579af026dcfb627c84070`, published 2026-09-30 | Source inspection of capture, autodiff, Inductor IR/scheduler, memory planning and code generation; PyTorch was not installed or benchmarked for this extension |
| BCIR PR #820 | Open, draft, unmerged; refreshed 2026-10-20 at `6c01fcad1c2ace951a077956d8d0058b6490b28f` | Per-layer recomputation, exact BF16 workspace coupling and PX-1 launch tracing; pending branch evidence, not measurements repeated here |

Full comparative model training remains deferred until [#820](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/pull/820)
is completed and merged. Source pins above are independent inspection points, **not a tested
compatibility matrix**. In particular, PyTorch 2.14.1 pins Triton commit
`675c59878aa2280b31f722aaf42b825fcee21de8`; this report does not assert that installing
upstream Triton 3.8 into that PyTorch release is supported [P1, P2]. Live documentation was
checked on the report date; release-pinned implementation takes precedence.

The pilot called existing BCIR C functions directly. It did not execute BCIR-generated MLIR,
GEM+ planning, StreamPack or ASN.1. Its useful kernel results cannot identify the IR as their
cause. Likewise, earlier model figures and the proposed 3–80× range are not newly reproduced
measurements in this document.

## 2. Compare the actual optimization layers

| Question | BCIR | Triton | PyTorch |
|---|---|---|---|
| Main authoring/semantic surface | Typed resources, claims, phases and region contracts; native model/training APIs | Python-embedded tiled SPMD kernel language; explicit-layout Gluon route | Imperative tensor programs, modules, autograd and training ecosystem; captured graphs when compiled |
| Important representations | Oracle objects and artifacts; ODS MLIR law; structural IRDL projection; ExecutionPlan/StreamPack; C runtime state | TTIR, TritonGPU and target dialects; LLVM IR and device images | FX graphs with operator semantics, functional ATen/Prims forms, joint forward/backward graphs, Inductor loop/buffer/scheduler IR; backend representations |
| Main decision scale | Program/claim planning, resource constraints, scheduling, lifetimes and movement; native C kernels on a separate path | Tile-to-thread/register/shared-memory mapping and kernel implementation | Model graph capture, differentiation/partitioning, fusion, memory and code generation; eager/library fallback where appropriate |
| Delivered hardware optimization | Native CPU kernels and target-specific paths; qualified scope varies by rail and target | Deep NVIDIA/AMD GPU kernel compilation; CPU is a separate experimental backend | CPU code generation and libraries, GPU code generation including Triton, accelerator/library dispatch and distributed infrastructure |
| Present integration limit | Broad plans are not yet shown to control every tensor operation in native decoder execution | A kernel compiler does not automatically own the model graph, data pipeline, optimizer or OS driver | Graph breaks, unsupported operations and opaque external calls can limit optimization scope; many layers and specializations have engineering costs |
| Differentiating opportunity | Explicit, inspectable coupling across domains and bounded resources with a small C execution core | Hardware-efficient custom kernels with substantial reusable compiler machinery | Broad operator/autograd coverage, established training/deployment workflows and compiler integration |

PyTorch and Triton are often **parts of one stack**, so beating a standalone Triton example
and beating compiled PyTorch answer different questions. A BCIR plan using the same Triton
kernel could improve orchestration while retaining Triton's kernel quality. That is a useful
BCIR result if the experiment identifies the changed scheduling or movement decisions.

### Claims that need correction

| Premise | Finding from the inspected implementations | Consequence for BCIR |
|---|---|---|
| MLIR/LLVM dependence shows Triton's IR is underdeveloped | Triton performs extensive domain-specific transformations before LLVM; MLIR provides infrastructure for defining those representations | Judge executed transformations and generated resources, rather than compiler dependency count |
| PyTorch cannot define or change its computation model through an IR | FX/operator IRs, Inductor IR and custom compiler backends provide several representation and transformation boundaries [P3–P7] | A BCIR backend/importer can be an integration route; replacing all of PyTorch is unnecessary to test BCIR planning |
| PyTorch's age fixes it into a pre-MLIR design | Its current compilation stack substantially changes how captured programs run | Compare current eager and compiled paths separately |
| No C prevents Triton from reaching drivers | Triton includes `third_party/nvidia/backend/driver.c`, which loads CUDA modules and launches kernels [T3] | A C runtime interface is feasible; implementation language does not determine device access |
| OS protection is Triton's central optimization defect | Host/device dispatch is asynchronous; user-mode queues and graph submission already amortize control overhead [H1–H3] | Measure submission, synchronization and residency before proposing an OS replacement for throughput |
| More C, fewer compiler layers, or a different build system implies faster execution | Source language shares and build orchestration do not identify runtime work or emitted instructions | Inspect binaries, execution traces, allocations, bytes moved and numerical contracts |

The approximate repository language percentages are deliberately not used as capability
measurements. TableGen and MLIR definitions describe semantics and compiler structure; Python
can generate native code; C++ can implement a kernel compiler or a native runtime. GitHub's
language accounting is also sensitive to generated files and repository contents. CMake can
generate Make/Ninja builds, and shell scripts can orchestrate either. These are maintenance and
reproducibility choices, rather than evidence about the model of computation.

## 3. How advanced is Triton's MLIR implementation compared with BCIR's?

**For production GPU tensor-kernel lowering, Triton's implementation is substantially deeper
than BCIR's inspected MLIR execution path.** BCIR has broader program-level law and planning
concepts. Breadth of represented domains and depth of target code generation are separate axes.

| MLIR dimension | Triton 3.8 | BCIR at the inspected main revision |
|---|---|---|
| Dialect purpose | Tensor/pointer operations, distributed GPU layouts, shared memory and target operations | Resource/claim semantics, verifier laws, planning, scheduling, encoding and target contracts |
| Layout semantics | GF(2) linear layouts across registers, lanes, warps and blocks; layout conversions, replication and target encodings | Useful resource and integer-affine layout/planning machinery; not an equivalent GPU layout-conversion stack |
| Executable GPU transformations | Coalescing, layout-conversion elimination, matmul acceleration, software pipelining, warp specialization, tensor-memory/shared-memory and barrier handling | Compiled cost, plan, overlap, budget and verifier passes; GPU backend maturity is much narrower than these planning concepts |
| Matrix lowering | Target tensor-core operations and associated operand, memory and synchronization transformations | `gem.matmul_buffer` lowers to tiled scalar `scf.for` loops with `memref.load/store` and `arith.mulf/addf`; no tile-private accumulator in that lowering |
| Verification | Operation verifiers, compiler analyses/tests and specialized sanitizers; not a proof of all IEEE numerical behavior | Cross-rail law/cost/artifact parity and negative witnesses within declared scopes; not a proof of all native/device code |
| Meaning of portable representation | Shared compiler infrastructure with target-specific implementations and capabilities | ODS compiled law plus a pure-data structural IRDL projection; loading that projection does not run the C++ optimizer or prove semantic parity by itself |
| Hardware evidence in this study | Upstream GPU capabilities are source-inspected; experimental CPU overlap measured | Existing native C kernels measured; no experiment connecting their speed to a new MLIR transformation |

The NVIDIA backend explicitly assembles passes such as `add_accelerate_matmul`,
`add_remove_layout_conversions`, `add_pipeline`, `add_warp_specialize` and
`add_optimize_tmem_layouts`. It subsequently lowers to LLVM IR, emits PTX, and invokes `ptxas`
to produce a device binary. These passes encode substantial GPU knowledge before final
instruction selection/assembly [T1]. Gluon exposes more layout and execution decisions to
authors [T2].

The contrast with BCIR is concrete. Its [matmul-buffer pass](../../mlir/lib/passes/BCIRLowerGemMatmulBufferPass.cpp)
emits an executable-style tiled loop nest and handles ragged tile bounds. It reads and writes
the output for each scalar reduction iteration in the emitted IR. Downstream optimizers may
improve that code, but the pass itself is not equivalent to Triton's tensor-core, register-tile
and asynchronous GPU pipeline. BCIR's fast hand-written C GEMMs cannot be attributed to this
pass without compiling and measuring this route.

BCIR's [compiled pass family](../../mlir/README.md) is nevertheless substantial: cost
recomputation, coupled planning, constrained search, overlap, scheduling, pool planning,
explanation and replay are real implemented work. The [parity ledger](../PARITY.md) also
records which pieces have a law/C twin and which remain oracle-only. It should be the capability
ledger for claims about cross-rail coverage; neither operation counts nor parser round-trips
replace it.

### Why the layout mathematics matters

An integer-affine map `a*i+b` and an XOR bank swizzle belong to different algebraic domains.
Triton's GF(2) machinery supports layouts that BCIR's integer-affine carrier does not
automatically subsume. Conversely, a GPU bit-layout algebra is not a whole-program resource
and lifetime model. Replicated layouts may be non-injective, so conversion needs preimages or
an appropriate composed map, rather than assuming every layout has an inverse [T4].

The useful BCIR extension is an explicit distinction among logical shape/strides, affine
address maps, GF(2) distribution/swizzles, masks, replication and address spaces. Its cost
must include actual conversion work: shuffles, shared-memory round trips, synchronization,
padding and persistent packs. That converts mathematical expressivity into an executable
optimization contract.

## 4. What PyTorch does instead of a mandatory MLIR middle layer

PyTorch's usual Inductor path does not require a direct, universal MLIR representation for the
whole model. It still has a multi-level compiler, and its Triton GPU path uses MLIR indirectly.
The inspected stack supports these boundaries [P3–P8]:

| Stage | Representation and decisions | Relevant limitation |
|---|---|---|
| Eager execution | ATen/dispatcher, autograd and native/library implementations execute tensor operations | Per-operation host overhead and intermediate materialization can matter; no assumption that every operation is Python arithmetic |
| TorchDynamo capture | Python bytecode analysis captures FX graphs; guards and symbolic shapes determine reusable specialization | Unsupported Python/side effects can cause graph breaks, recompilation or refusal depending on settings |
| AOTAutograd / AOTDispatcher | Functionalization and decompositions support joint forward/backward graphs, partitioning and recomputation | Mutation, aliasing, RNG/effects and differentiation semantics constrain legal rewrites |
| TorchInductor | Its own pointwise/reduction loop IR, layouts/buffers, dependency/fusion scheduling, memory planning and backend selection | Compiler heuristics and target coverage are imperfect; internal interfaces change |
| Kernel implementation | C++/OpenMP CPU code, Triton GPU code, target-specific generators and external library kernels | Best lowering depends on shapes, target, precision, fusion and library availability |
| Deployment | `torch.export` captures normalized operator graphs and shape constraints; AOTInductor can package exported inference for non-Python use | Exportability and supported deployment operators are separate requirements; this does not make all training a standalone export artifact |

This is directly visible in the pinned source:

- `torch/_inductor/ir.py` defines `Loops`, `Pointwise`, `Reduction`, `Layout`,
  `FixedLayout`, `FlexibleLayout`, `Buffer`, `ComputedBuffer` and `ExternKernel` [P5].
- `scheduler.py` represents dependencies, fusion and external-kernel scheduling;
  `codegen/memory_planning.py` computes buffer groups/live ranges and allocates pools [P6].
- `torch/_functorch/partitioners.py` implements `min_cut_rematerialization_partition` and
  activation-memory-budget handling [P7]. Recomputation/memory trade-offs are already an active
  PyTorch capability, rather than an unoccupied architectural category.
- `codegen/common.py` registers device scheduling/wrapper code generation. The default CPU
  C++ path and CUDA combined scheduling are not the only configurable implementations [P8].
  Registration in source is not evidence of equal maturity across all of those backends.

PyTorch also has CPU cache/register/thread blocking and autotuning machinery. Its C++ GEMM
templates explicitly account for microkernels and L1/L2 blocking [P9]. Therefore, a blanket
claim that it lacks low-level hardware awareness is inaccurate. An opaque external library
can still restrict cross-call fusion and visibility; the specific dispatched implementation
must be identified before drawing that conclusion.

### Benefits and costs of this choice

PyTorch's Python-based compiler IRs make graph rewriting and integration convenient in its
existing ecosystem. It can evolve operator decompositions, symbolic shape handling and
backends without first adopting a common MLIR dialect family for every graph. It retains
access to MLIR where useful through Triton or projects such as Torch-MLIR [P10].

The cost is maintaining several representations and their semantic bridges: aliasing/effects,
guards, operator decompositions, graph partitions, scheduler IR and device implementations.
Some graph-level invariants and transformation infrastructure must be implemented there
rather than inherited from MLIR. A direct MLIR route would inherit useful infrastructure but
would still need correct PyTorch capture, autodiff, effects, operator coverage and lowering.
Adopting MLIR does not make those problems disappear.

PyTorch is extensible through custom operators, graph passes, decompositions and
`torch.compile(..., backend=...)`. Backends can receive AOTAutograd's forward/backward graphs
[P4]. A BCIR importer could use that boundary while leaving the C execution core independent.
That would need an explicit supported subset, tensor ownership/stride rules, mutation/aliasing
handling, RNG/effect ordering, shape guards and refusal/fallback behavior. Internal Inductor
hooks should not be presented as stable public APIs.

Neither custom backends nor MLIR dialects permit arbitrary semantic changes while claiming
equivalence. A different Grassmannian head, loss, approximation or precision rule is a model
or numerical-contract change. It needs its own quality experiment even when a compiler can
represent it cleanly.

### Direct comparison of BCIR's and PyTorch's IRs

There are two useful direct comparisons: **semantic graph against semantic graph**, and
**lowered optimization representation against lowered optimization representation**. FX is
principally a graph container; its operators and metadata supply much of the meaning.
Functional ATen supplies a more constrained operator contract. Inductor IR exposes loop,
indexing, layout and buffer decisions at a lower level. Comparing only the word "IR" would
hide these differences.

| IR property | BCIR's corresponding representation | PyTorch's corresponding representation | Finding / question to test |
|---|---|---|---|
| Values and storage identity | Registry-indexed `Resource`s; claims name read/write resources, phases and dependencies | FX node values/arguments; ATen operator contracts; Inductor named buffers/layouts and indexed memory dependencies | BCIR makes resource identity explicit early; functional values and later bufferization can also support lifetime/alias analysis |
| Mutation, effects and ordering | Claim hazards, volatile/fence behavior, generations and optional lifetime annotations | Functionalization, effect/RNG handling, operator schemas and scheduler dependency/mutation handling | Both constrain reordering; compare actual supported effects and refusal witnesses rather than calling either graph effect-free |
| Tensor operations and gradients | ML operation/plan families plus native tensor/decoder forward/backward implementations, with rail-specific coverage | Broad ATen operators, decomposition rules, autograd-generated backward graphs and joint-graph partitioning | PyTorch has broader delivered training/compiler coverage; BCIR native backward coverage is not automatically graph-level differentiation coverage |
| Shapes and indexing | Resource shapes, access maps and claim counts; compositional dynamic counts can be static upper bounds | Symbolic sizes/strides and guarded specialization; Inductor symbolic indexing dependencies | A bounded dynamic cost contract is different from compiling symbolic extents; compare variable shapes, guards, recompiles and fallback behavior |
| Local computation models | Recognized affine, SDF/cyclostatic, timed, 2-D nest and opaque regions | Operator graph plus specialized higher-order/control representations; Inductor loop/reduction IR and backend kernels | BCIR offers explicit non-tensor models; the current region carrier expands back to the original claims, so recognition alone does not demonstrate new execution transformations |
| Optimization objective | Explicit integer cost vectors, policies/budgets, candidate selection and scope-relative certificates | Heuristics, dependency/fusion/memory scheduling, template/library choices and measured autotuning | BCIR exposes a promising decision contract; compare feasible candidate coverage and measured ranking error, not just availability of an argmin |
| Memory planning | Explicit bank/offset/lifetime/movement records in an ExecutionPlan, with heuristic/exact small-scope options | Inductor buffer groups, reuse, live ranges, memory-aware ordering and allocation pools; forward/backward rematerialization elsewhere in the stack | Both plan memory. Test whether the selected plan includes all actual scratch/packs and enforces the same budget in execution |
| Representation of the selected execution | Plan/StreamPack artifacts and C execution contracts; native decoder still has a separate direct loop | Scheduled buffers/kernels and generated wrapper code call compiled/library implementations | BCIR needs the missing plan-to-native coupling demonstrated; PyTorch's generated wrappers are an existing execution bridge |
| Verification and explanation | Registered laws, oracle/law/C parity, scoped certificates, explanation/replay | Operator constraints, capture guards, dependency analyses, compiler assertions/tests and execution validation | Compare each checked property and its scope; neither row establishes complete equivalence of arbitrary final machine code |
| Extensibility boundary | Oracle model, ODS operations/passes, optional IRDL projection, runtime/provider ABIs | FX passes, decompositions, custom operators, custom compiler backends and internal codegen interfaces | Both are extensible. Measure implementation effort and semantic/ABI obligations for one concrete new operation |

The BCIR side is grounded in [the claim/resource model](../../bcir/model/graph.py),
[regions](../../bcir/kbcir/regions.py), [compositional bounds](../../bcir/kbcir/compose.py)
and [ExecutionPlan](../../bcir/gem/execution_plan.py). Optional timing or lifetime annotations
only constrain a frontend that actually supplies them. The PyTorch side is grounded in
[FX nodes][FXNODE], [Inductor IR][P5], [memory dependencies][P12], the scheduler, pool planner
and joint-graph partitioner. Entries describe inspected mechanisms, not new timing results.

A productive first common program is a SwiGLU block:
`U = X @ Wg; V = X @ Wu; Y = silu(U) * V`, followed by its backward computation.
Keep shapes, strides, FP behavior and inputs fixed. Compare the semantic graphs, admitted
partitions, shared-input packing, pointwise fusion, temporary lifetimes and emitted calls.
Inspect rather than assume whether each system fuses, repacks or retains an intermediate.
Include an irregular shape, a non-contiguous input and a bounded variable-length case.

Use two routes when executing that future comparison:

1. Give both planners the same supported kernel/provider portfolio and measure graph/plan
   construction, verification, scheduling, packing and execution. This isolates the effect of
   planning more effectively than changing the kernel compiler at the same time.
2. Allow each system its own qualified code generation/providers under the same numerical,
   tuning and resource budget. This measures the complete implementation, including backend
   quality. Preserve graph/IR dumps and final calls/artifact identities for attribution.

Compiler-only timing can measure capture/import, lowering, planning and code generation
separately, with graph size/depth/fan-out varied independently. Measure peak compiler memory,
IR/artifact size, legal transformation coverage and optimization quality as well as time.
Do not equate smaller serialized plan bytes with a smaller training activation arena. Likewise,
timing Python object construction against parsing a prebuilt MLIR file would include different
work unless the input boundary is matched.

This direct analysis gives a testable BCIR proposition: its explicit resource and constrained
planning model might produce better time/memory choices on the shared subset. It also exposes
the current alternative explanation for performance gaps: missing executable transformations
or weaker kernels. The appropriate result is a property-by-property comparison and a measured
frontier, rather than declaring one whole IR universally superior.

## 5. Advantages and limits of Triton's MLIR-heavy approach

MLIR is an extensible compiler framework, not one fixed machine model. A dialect supplies
operations/types/attributes and semantics; transformations decide how they map to lower
levels. Several dialects can coexist during progressive lowering [M1, M2]. BCIR itself uses
this design in its compiled law rail.

| Advantage | What it buys Triton | Remaining cost or limit |
|---|---|---|
| Typed, multi-level representations | Preserve tensor/layout/target information until the corresponding optimization is useful | Lowering too early can lose information; retaining incompatible representations creates bridging work |
| Shared infrastructure | SSA/regions, parsing, verifiers, rewrite/conversion machinery, analysis and testing utilities | Domain legality, good rewrites and useful cost models still require substantial implementation |
| Explicit GPU dialects | Reason about distributed registers, shared memory, tensor operations and asynchronous execution | Target-specific layout and synchronization rules require ongoing hardware work |
| LLVM/vendor toolchain handoff | Reuse instruction-selection, optimization, object/device tooling and hardware assemblers | Version pins, final-code opacity, toolchain regressions and compile-time costs remain |
| DSL and Gluon abstraction choices | Offer both productive tile authoring and more explicit hardware mapping | No abstraction automatically finds the best tile/partition; autotuning/search can be expensive |

Triton's longer-term constraint is its **optimization scope and coverage**. A kernel compiler
does not automatically optimize end-to-end data ingestion, an entire training state, device
placement, cross-kernel lifetimes or distributed communication. Host/framework integrations
can provide those layers. It is therefore a composable building block rather than a complete
training/OS system.

Other limits are practical: layout conversions can be expensive; fusion can increase register
pressure and spills; dynamic sizes create padding/specialization choices; backend support is
uneven; and final instruction selection may still need target-specific fixes. These are
engineering challenges shared with competing compilers. They do not imply an inherent
inability to extend Triton's IR for future hardware.

## 6. Drivers, protection boundaries and hardware intimacy

Three different things are often called a kernel or driver:

| Component | Responsibility | Relevance here |
|---|---|---|
| GPU compute kernel | Device instructions executed by GPU workgroups/warps | Triton generates these; such a kernel is not an x86 ring-0 OS kernel |
| User-mode runtime/driver interface | Load device images, manage allocations/streams/events and submit work | Triton's C helper calls CUDA driver APIs; other integrations can use the same device images and runtime boundary |
| OS kernel-mode device driver | Device setup, memory mappings, isolation, faults, scheduling and hardware/firmware coordination | A native BCIR OS needs this functionality or an explicit trusted-device alternative; it is not supplied merely by having a C ML library |

The absence of C would not prohibit a C ABI or native device access, and Triton in fact has C
code at this interface. Writing a runtime in C can help BCIR control dependencies and memory;
it does not confer extra hardware privilege or eliminate the need for a device protocol.

OS protection is also not a per-arithmetic-operation tax. GPU execution can remain resident and
asynchronous between submissions. AMD's ROCr documentation explicitly describes user-mode
queues enabling low-latency dispatch [H1]. CUDA Graphs describe batches/dependencies as a
workflow for repeated submission [H2]. Profiling distinguishes host API overhead, queueing,
device execution and synchronization [H3]. None of these requires removing Linux.

BCIR can investigate a specialized runtime or independent OS, but performance attribution
requires measuring the relevant boundary. First separate host submission time, device event
time, copies/page migration, barriers, allocator activity and full critical-path latency.
Compare resident queues, captured graphs, batched launches and persistent work before spending
a driver rewrite on an overhead already amortized by the incumbent.

Device compute, HBM traffic and interconnect costs remain even if host submission becomes
free. A specialized OS could help latency predictability, resource policy, boot footprint or
dedicated-device workloads. Its throughput benefit is workload-dependent and unmeasured here.
Sharing compiler/runtime contracts is a reasonable integration goal; sharing one address
space is neither necessary for those contracts nor sufficient for peak throughput.

## 7. What the small-workload speedups imply about TMSAO

The compare/analyze/update loop is valuable when it identifies a mechanism and preserves the
numerical contract. A small-call win can come from fewer allocations, dispatch costs, better
specialization, SIMD kernels or different thread behavior. It need not come from a more
expressive IR. At larger sizes, dense math throughput, packing, cache capacity, memory
bandwidth and parallel scheduling can dominate instead.

Under the simple Amdahl model, speeding up a fraction `p` of the original elapsed time by a
factor `r`, with the remainder unchanged and no overlap change, gives:

`S = 1 / ((1-p) + p/r)`.

- An 80× improvement to 10% of a step produces about **1.11×** overall speedup.
- An 8× improvement to 80% produces about **3.33×** overall speedup.

These are calculated examples, not BCIR measurements. For asynchronous systems, use the
critical-path trace rather than adding overlapping kernel durations to estimate `p`.

### The useful optimization problem

For a declared graph/workload, target and numerical contract, choose a plan `pi` from a legal
candidate set to minimize an objective such as elapsed time, subject to peak memory, quality,
resource and compilation budgets. Choices can include partitioning, implementations, layouts,
packing reuse, recomputation, placement, precision and dispatch. The interactions matter:

- Fusion can remove memory traffic but reduce occupancy through increased register pressure.
- Keeping a packed weight can save repeated work but extend its lifetime and consume an arena.
- Recomputation can free activations, enabling a larger useful batch; it also consumes compute.
- A precision change can reduce traffic while changing convergence or numerical safeguards.
- Overlap can hide transfers while increasing simultaneous buffer occupancy and contention.

For fixed required arithmetic `F` and traffic `Q` under a declared algorithm, a useful lower
bound includes `max(F/P, Q/B)`, where `P` and `B` bound compute throughput and bandwidth.
Dependency span and communication can add other lower bounds. These are not automatically
tight: achievable throughput varies with shapes, occupancy, cache behavior and concurrent
work. A model-relative proof of optimum does not establish a physical hardware optimum.

TMSAO should therefore be reported as a **scope and remaining gap**, rather than one global
completion percentage. Name the legal transformations, supported operators/shapes, calibration,
search budget, lower bound and observed result. Report where the cost model misranks a candidate.
The [BCIR execution scope](../../bcir/kbcir/scope.py),
[cost model](../../bcir/kbcir/cost.py) and [memory planner](../../bcir/kbcir/static_memory.py)
provide useful foundations for this discipline.

Compilation is part of the objective. If candidate B adds compilation/tuning time `DeltaC` and
saves `DeltaT > 0` per execution, its simple break-even count is `N > DeltaC/DeltaT` before
accounting for other changes. A short-lived app and repeated training can rationally select
different plans. Cache identity must include target, compiler, semantics and shape assumptions.

## 8. Current directions and what they suggest for BCIR

This table distinguishes shipped/source-visible mechanisms from recommendations. It is not
a forecast that any one project will dominate.

| Observed direction | Primary evidence | BCIR strategy inferred from it |
|---|---|---|
| Tile abstractions coexist with explicit hardware mapping | Triton/Gluon; NVIDIA's CUDA 13.2 cuTile work [T1, T2, H4] | Preserve high-level algebra and allow explicit target tile/layout contracts at the right boundary; keep optional vendor backends |
| Memory and recomputation are first-class compiler decisions | PyTorch joint-graph partitioner and memory planner [P6, P7]; pending BCIR #820 | Couple activation lifetimes with real packing/workspace needs, then include those decisions in the executed plan |
| Host orchestration is reduced through residency, queues and graph submission | ROCr queues and CUDA Graphs [H1, H2] | Build asynchronous events/residency into the provider interface before assuming an OS rewrite is needed |
| Target specialization and multiple implementations persist | Triton's target pipelines; Inductor CPU templates and backend registry [T1, P8, P9] | One semantic operation should admit C, compiler-generated and optional vendor implementations with explicit capability/precision predicates |
| Capture/export and deployment are separate design concerns | PyTorch custom backends, `torch.export` and AOTInductor [P4, P11] | Provide optional graph import and ahead-of-time artifacts while preserving a dependency-light C runtime |

The likely competitive pressure is **more integration inside existing ecosystems**, alongside
more compiler choices. BCIR should plan against incumbents that also improve recomputation,
layouts, profiling and autotuning. The defensible contribution would be demonstrated cross-domain
decisions, transparent costs/contracts and portability, rather than assuming the other projects
will remain siloed.

Low precision, heterogeneous devices and distributed execution expand the search space.
Represent storage, accumulation, approximation/error policy and device communication explicitly;
test training quality rather than ranking a changed precision contract on latency alone.
Learned or generated candidate selection can help search, but must remain constrained by
legality and held-out measurements. Neither this report nor the current CPU pilot demonstrates
multi-device training or multi-trillion-parameter scalability.

## 9. A gateable architectural update blueprint

The order below prioritizes causal evidence and useful integration. It proposes work rather
than claiming these interfaces have been implemented by this documentation update.

The detailed [OT test catalogue and BG entry gates](BCIR_BACKEND_OPTIMALITY_AND_KERNEL_GATES_2026-10-20.md)
refine this order. Pending #820 now implements PX-1 trace-to-program checking. Continue with
its PX-2 streamed C execution and PX-3 plan-selected intervention after merge; trace matching
alone is not the complete plan-driven executor. Bounded UART/boot research and a hosted
accelerator provider can proceed alongside IR/training optimization under their own gates.

| Priority / proposed slice | Concrete change | Acceptance evidence |
|---|---|---|
| 1. Trace to plan-driven execution | Review pending PX-1 launch/arena witnesses; implement PX-2/PX-3 so the admitted per-call choices drive native execution, retaining the direct loop as reference | Complete trace/state parity first; then change one legal plan decision and show the corresponding implementation, allocation, movement or schedule changes and measured effect |
| 2. Memory/packing/recompute coupling | After #820 merges, expose its exact workspace tiers and recompute sets through the same planning contract as native execution; avoid duplicate accounting across pools/packs | Budget admitted equals observed owned storage for the declared scope; time/memory frontier at fixed full micro-batch and numerical contract; checkpoint/resume and stress refusals |
| 3. Kernel portfolio | Inventory forward/backward/update coverage; add or improve measured bottlenecks; retain reference C, tuned C and optional vendor/compiler providers | Operator and gradient/state checks, irregular/tail cases, cold/warm resource measurements and representative held-out shapes; provider selected under declared predicates |
| 4. Resident accelerator boundary | Extend the synchronous C provider seam with device/context, ownership, address space, streams/events, async copies, scratch, persistent packs, resource metadata and versioned device-image identity | First execute the same compiled kernel directly and through BCIR with identical resident buffers; isolate orchestration overhead before comparing alternative kernels |
| 5. Layout and lowering depth | Add the required GPU layout algebra and conversions, register/tile accumulators and target-specific memory/synchronization lowering, or delegate them to a qualified backend | Compile and execute representative kernels on real target hardware; inspect spills, shared memory and conversions; reject unsupported layouts/targets explicitly |
| 6. Optional model-graph import | Admit a declared functional ATen/FX subset through a separate integration package; reuse forward/backward capture where useful | Imported graphs have checked effects/aliases/shapes and supported operations; refused nodes are visible; end-to-end tests identify which BCIR decisions execute |
| 7. Calibrated search | Combine legality, analytical pruning, compile/resource feedback and bounded measurement; optimize the appropriate time/memory/quality objective | Report ranking error, tuning/compile cost and cache behavior; demonstrate gains on held-out workloads and more than one qualified target |

Keep execution and arena management in C where that serves BCIR's core requirements. Use C++
at framework/compiler/accelerator boundaries when required by those APIs. A Python graph-import
or tuning tool need not become a runtime dependency. Follow the existing
[C++ handoff boundary](../languages/CPP_HANDOFF_BOUNDARY.md).

Do not couple this ladder's first success to independent UEFI boot or a complete vendor driver.
Those are separate useful milestones with their own device bring-up and validation criteria.
The first result should be a narrow, reproducible demonstration that a BCIR decision improves
an existing hardware execution path at fixed semantics.

## 10. Make competitive assessment continuous and diagnostic

A comparison registry should define experiments once and preserve failures, ties and unsupported
cases alongside wins. This is a proposed engineering process, not a scheduled task or an
implemented benchmark service.

### Required record for every result

| Record | Required fields |
|---|---|
| Identity | Repository/compiler/provider commits, built artifact hash, target/driver/runtime, benchmark protocol and random seeds |
| Semantics | Operation/model recipe, input distribution and shapes/strides, storage/accumulation types, reduction/approximation policy, effects/RNG and quality tolerance |
| Execution | Actual provider/lowering path, fallback labels, thread/affinity settings, device residency, transfer/queue/synchronization boundaries, compile-cache state |
| Resources | Parameter/state/activation/scratch/pack bytes, simultaneously live device allocations, process/reserved memory where relevant, compile time and tuning budget |
| Outcome | Numerical and training-state checks, held-out quality, paired latency samples and uncertainty, throughput, failures/refusals; energy only when actually measurable |
| Attribution | Selected plan, executed trace, baseline and changed decision, expected mechanism, observed bytes/launches/allocations or hardware counters |

Keep memory scopes separate. Arena counts, process RSS, device allocated memory and allocator
reserved memory are different quantities. The CPU pilot has payload/pool accounting, not a
measured whole-process peak-memory comparison.

### Comparison ladder

| Level | Comparator | What it can establish |
|---|---|---|
| Primitive | Same semantic kernel, target, shape and precision | Implementation quality, including tails and backward; direct-call latency is a distinct metric from device-only time |
| Same artifact | Direct provider dispatch versus BCIR dispatch of identical compiled code/buffers | Orchestration, residency and host overhead attributable to BCIR |
| Graph block | Same normalization/MLP/attention block and numerical contract | Fusion, packing reuse, layout conversion and intermediate-memory effects |
| Full training, deferred | Matched model, initialization, batches, optimizer and quality contract versus qualified compiled PyTorch/other training harness | Complete training machinery, including memory constraints, synchronization and state evolution |
| Best admitted recipe, deferred | Equal hardware, memory/tuning budget and quality target with documented architectural freedom | Ecosystem capability and quality/time/memory frontier; model changes are not compiler-speed evidence |
| System/runtime | Identical device kernels/workload with measured host/runtime changes | Benefit of specialized queues, drivers or OS policy; distinct from kernel math quality |

Use small correctness and representative performance checks during kernel/planner changes;
reserve a broader frozen matrix for qualification. On real hardware, include launch-bound,
cache-resident, bandwidth-bound and dense-math regimes, variable lengths, contended/resident
conditions and supported precision modes. Pin tuning budgets, report cold and warm behavior,
and randomize paired rounds. Select held-out shapes before tuning to expose overfitting.

For a regression, identify whether time moved into code generation, packing, arithmetic,
memory traffic, recomputation, dispatch or synchronization. An A/B intervention should vary
one decision where practical; a coupled change needs its own explanation. Fit a calibrated
ranking model only after verifying which implementation ran. Gate correctness and refusals
deterministically; use timing confidence bands and appropriate target isolation for performance
rather than a noisy VM threshold that repeatedly flips.

The strategic milestone is a repeatable win on a named workload/target/quality/memory contract,
with a demonstrated mechanism and no hidden fallback. Expand the supported matrix from there.

## Sources and implementation anchors

BCIR links refer to the main source pin in §1. The companion CPU report and its raw artifacts
remain the authority for those measurements. Source was read locally from release checkouts;
no timing or performance claim is inferred from source line counts.

- **[B1]** BCIR [parity ledger](../PARITY.md), [MLIR implementation](../../mlir/README.md),
  [matmul-buffer lowering](../../mlir/lib/passes/BCIRLowerGemMatmulBufferPass.cpp),
  [native training scope](../machine-learning/BCIR_NATIVE_DECODER_TRAINING.md),
  [native program description](../../bcir/hosted/models/native_program.py),
  [target emission](../../bcir/codegen/targets.py) and [SYCL boundary](../kernel/SYCL_INTEROP.md).
- **[B2]** Pending [#820](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/pull/820),
  refreshed at `6c01fcad1c2ace951a077956d8d0058b6490b28f` for PX-1 and the updated test plan;
  the initial three-way inspection was at `b0ce73f89d890ff0c8ee3d42da6ca1dfbc37101e`,
  and the CPU pilot's earlier inspection was at `af1d935be77854c0d0fee2b8846714281864cd7c`.
- **[T1]** Triton [3.8 release](https://github.com/triton-lang/triton/releases/tag/v3.8.0),
  [pinned NVIDIA pipeline](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/third_party/nvidia/backend/compiler.py),
  [GPU transforms](https://github.com/triton-lang/triton/tree/c01b6774b1865984607d89d89d3a10833de92037/lib/Dialect/TritonGPU/Transforms)
  and [NVIDIA transforms](https://github.com/triton-lang/triton/tree/c01b6774b1865984607d89d89d3a10833de92037/lib/Dialect/TritonNvidiaGPU/Transforms).
- **[T2]** [Gluon overview](https://triton-lang.org/main/gluon/index.html) and
  [layout tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html).
- **[T3]** Triton [C driver interface](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/third_party/nvidia/backend/driver.c).
- **[T4]** Triton [LinearLayout and GF(2)](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/include/triton/Tools/LinearLayout.h).
- **[P1]** [PyTorch 2.14.1 release](https://github.com/pytorch/pytorch/releases/tag/v2.14.1)
  and [source pin](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070).
- **[P2]** PyTorch's [Triton commit pin](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/.ci/docker/ci_commit_pins/triton.txt).
- **[P3]** [Compiler overview](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html),
  [ATen/Prims IR definitions](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/torch.compiler_ir.md)
  and [dynamic shapes](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html).
- **[P4]** [Custom backends, including AOTAutograd](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/torch.compiler_custom_backends.md).
- **[P5]** [Inductor IR](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/ir.py).
- **[P6]** [Scheduler](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/scheduler.py),
  [memory-aware scheduling](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/memory.py)
  and [pool/live-range planning](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/codegen/memory_planning.py).
- **[P7]** [Joint-graph partitioner](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_functorch/partitioners.py),
  [activation-checkpointing techniques](https://pytorch.org/blog/activation-checkpointing-techniques/).
- **[P8]** [Device code-generation registry](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/codegen/common.py),
  [Inductor compiler](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/compile_fx.py).
- **[P9]** [C++ GEMM templates](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/codegen/cpp_gemm_template.py)
  and [CPU autotune tutorial](https://docs.pytorch.org/tutorials/unstable/max_autotune_on_CPU_tutorial.html).
- **[P10]** [Torch-MLIR](https://github.com/llvm/torch-mlir) and its
  [FX importer](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/extras/fx_importer.py),
  an optional integration project rather than PyTorch's mandatory default compiler.
- **[P11]** [Export contract](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/export/__init__.py),
  [AOTInductor](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_inductor.html).
- **[P12]** [FX node semantics](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/fx/node.py)
  and [Inductor indexed dependencies](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/dependencies.py).
- **[M1]** [MLIR rationale](https://mlir.llvm.org/docs/Rationale/Rationale/).
- **[M2]** [Linalg rationale and progressive lowering](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/).
- **[H1]** [ROCr runtime and user-mode queues](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/what-is-rocr-runtime.html).
- **[H2]** [CUDA Graphs and batched submission](https://developer.nvidia.com/blog/a-guide-to-cuda-graphs-in-gromacs-2023/).
- **[H3]** [NVIDIA's overhead/latency profiling distinctions](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/).
- **[H4]** [CUDA 13.2 and cuTile](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/).

[P5]: https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/ir.py
[P12]: https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/dependencies.py
[FXNODE]: https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/fx/node.py
