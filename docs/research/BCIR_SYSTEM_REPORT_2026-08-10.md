<!-- allow-law-ranges -->
# BCIR comprehensive system report (snapshot at HEAD 997511de, 2026-08-10)

> **Superseded snapshot.** This report was appended to `BCIR_LANGREF.md` by a direct commit on
> 2026-08-11 and relocated here on 2026-09-03 because it is a dated inventory, not law. Its
> counts are frozen at that HEAD; the live inventory is generated [`STATUS.md`](../STATUS.md),
> the capability snapshot is [`REPO_CURRENT_STATE_AUDIT.md`](../REPO_CURRENT_STATE_AUDIT.md), and the
> 2026-09-03 whole-system analysis is
> [`BCIR_SYSTEM_ANALYSIS_2026-09-03.md`](BCIR_SYSTEM_ANALYSIS_2026-09-03.md). Section numbers below
> are the report's own and do not refer to LangRef sections.


**Report basis:** fresh, source-level reconstruction from the current repository rather than reliance on the compacted conversation.

## 1. Provenance and scope

| Item | Verified state |
|---|---|
| Repository | `Mecca-Research/Binary-Correspondence-Intermediate-Representation` |
| Branch | `main` |
| Local `HEAD` | `997511de91c59328759c6eec29602ab34344e206` |
| `origin/main` | `997511de91c59328759c6eec29602ab34344e206` |
| Working tree | Clean |
| Commit date | 2026-08-10 |
| Latest commit | `Merge pull request #740 from Mecca-Research/agent/tmsao-gemplus-architecture` |

The latest commit primarily introduces:

```text
docs/research/BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md
```

That document is a **research architecture proposal**. Its GEM+, TMSAO, service, IPC, driver, and distributed-runtime designs must not be confused with landed production implementations.

This report covers the execution-bearing system:

- semantic model and verifier laws;
- K_BCIR planning and optimization;
- GEM scheduling and execution;
- StreamPack and artifact formats;
- frontends and ETL;
- ASN.1;
- MLIR/C++;
- freestanding and hosted C;
- lowering and target code generation;
- hardware channels and memory planning;
- telemetry and calibration;
- AI, machine learning, model ingestion, inference, and training;
- security boundaries;
- testing, CI, documentation, and LLVM education;
- the latest GEM+/TMSAO proposal as a proposal.

It deliberately does **not** prioritize shortcomings or prescribe next development work yet.

---

# 2. What BCIR is

BCIR is not one conventional compiler IR. It is a multi-rail system for:

1. expressing a computation as a conservative graph of typed resource claims;
2. checking static legality independently of optimization;
3. enumerating legal realizations of those claims;
4. selecting a realization under target, runtime, cost, and budget constraints;
5. scheduling and freezing the selected result as a portable execution artifact;
6. lowering that artifact or plan to C, LLVM, MLIR, object code, WASM, and bounded stack-machine formats;
7. executing through native runtimes;
8. collecting telemetry;
9. admitting measured information back into future planning without allowing measurement or learning to redefine legality.

The central planning formulation is:

\[
K_{\text{BCIR}}(G \mid H,\Theta)
=
\min_{\pi} C_H(\pi,\Theta)
\]

where:

- \(G\) is the semantic claim graph;
- \(H\) is the target or substrate profile;
- \(\Theta\) is live runtime state;
- \(\pi\) is a path through legal realization candidates;
- \(C_H\) is a multidimensional, context-coupled integer cost.

The design’s most important separation is:

> **Legality is deterministic. Cost, telemetry, measurements, and learned models may rank legal alternatives, but they cannot make an illegal alternative legal.**

Primary sources:

- `docs/BCIR_LANGREF.md`
- `bcir/model/`
- `bcir/verify/`
- `bcir/kbcir/`
- `bcir/gem/`
- `mlir/lib/passes/BCIRVerifyPass.cpp`

---

# 3. Authority hierarchy

BCIR has several implementations that serve different purposes.

## 3.1 Normative language and law rail

The authoritative specification consists principally of:

- `docs/BCIR_LANGREF.md`;
- production MLIR ODS definitions;
- `-bcir-verify` in `BCIRVerifyPass.cpp`;
- frozen ABI documents under `docs/kernel/`.

The MLIR verifier is the production static-law rail.

## 3.2 Python executable conformance oracle

The Python package under `bcir/` is the broadest executable implementation. It contains:

- the semantic graph;
- candidate generation;
- cost model;
- optimizer;
- schedulers;
- StreamPack reference codec;
- ASN.1 reference implementation;
- telemetry and calibration;
- model and ML research organs;
- reference numerical implementations;
- differential and fuzz generators.

It is broader than the production C and MLIR implementations and is often the first rail on which a capability is made executable.

## 3.3 Freestanding and hosted C

`runtime/c/` supplies:

- no-libc/freestanding codecs and execution components;
- hosted compiler and ownership components where diagnostics or allocation are necessary;
- C twins of selected Python semantics;
- bounded trust-boundary implementations;
- native AI and model kernels.

It is intentionally narrower than the Python oracle.

## 3.4 C++ adapter layer

`runtime/cpp/` provides hosted adapters:

- BCAB RAII wrapper;
- JER SIMD and structural-index acceleration;
- a real single-node StreamPack orchestration seam;
- explicit dynamic-graph and distributed orchestration stubs.

## 3.5 Educational rail

`llvm-training/` is an independent educational and evaluation corpus. It teaches and tests LLVM/MLIR concepts but does not define BCIR legality.

## 3.6 Research rail

`docs/research/` contains proposals, audits, measurements, and architecture investigations. These become system capabilities only when backed by source and tests elsewhere.

---

# 4. Mechanical repository inventory

The fresh checkout contains **1,750 tracked files**.

| Tree | Tracked files | Role |
|---|---:|---|
| `llvm-training/` | 612 | LLVM/MLIR curriculum, exercises, fixtures, autograder |
| `bcir/` | 514 | Python oracle, tests, frontends, optimization, models |
| `runtime/` | 314 | C and C++ production/runtime rails |
| `mlir/` | 186 | ODS dialect, verifier, passes, examples, fixtures |
| `docs/` | 50 | normative, status, architecture, roadmap, research |
| `tools/` | 49 | validation, code generation, C, MLIR, model and performance gates |
| `channels/` | 5 | channel manifests and examples |
| `.github/` | 6 | CI and scheduled validation |

Selected subtrees:

| Subtree | Tracked files |
|---|---:|
| `bcir/tests/` | 250 |
| `bcir/kbcir/` | 80 |
| `bcir/frontends/` | 56 |
| `bcir/asn1/` | 42 |
| `bcir/hosted/` | 22 |
| `runtime/c/` | 303 |
| `runtime/cpp/` | 11 |
| `mlir/test/` | 117 |

A current ODS scan found:

- **133 operation definitions**
- **37 registered passes**

Operation-family count:

| ODS family | Operations |
|---|---:|
| Core | 27 |
| K_BCIR | 20 |
| GEM | 19 |
| Verify | 12 |
| ASN.1 | 7 |
| ECN | 7 |
| Binary format | 7 |
| Lowering contract | 6 |
| Optimization | 6 |
| Transducer | 6 |
| Event | 4 |
| Parse | 4 |
| Memory | 3 |
| Async | 2 |
| Trace | 2 |
| Target | 1 |
| **Total** | **133** |

---

# 5. End-to-end architecture

The principal execution path is:

```text
program / source / schema / manifest / binary record
        │
        ▼
frontend or programmatic graph construction
        │
        ▼
BCIR resources + claims + explicit phase DAG
        │
        ▼
R1–R25 legality verification
        │
        ▼
legal candidate enumeration
        │
        ▼
K_BCIR optimization under H, Θ, policy and optional budgets
        │
        ▼
selected plan + candidate costs + provenance/certificates
        │
        ▼
GEM scheduling and StreamPack hydration
        │
        ├── portable C / LLVM IR / MLIR / object / WASM / JVM / CIL
        ├── BCAB multi-backend bundle
        └── freestanding StreamPack execution
                │
                ▼
telemetry / measurement / calibration evidence
                │
                ▼
frozen generation-tagged table for a later replan
```

## 5.1 Stable embeddable facade

`bcir/api.py` exposes a compact library interface:

- `build_artifact(...)`
- `compile_kernel(...)`
- `KernelArtifact`

A resulting artifact carries:

- portable C source;
- freestanding C header;
- selected width and operation;
- K_BCIR score;
- calibration generation;
- plan-manifest digest;
- R12 lowering attestation;
- budget and feasibility metadata;
- diagnostics.

A fresh end-to-end AOT check was exercised through this API and returned:

```text
EMBEDDABLE_API_AOT_OK
```

## 5.2 Concrete fresh example

For `vector_add` on the `x86_avx512` profile under cool/latency policy, BCIR produced:

```text
scalar score = 32768
vec8 score   = 9472
vec16 score  = 7808
selected     = vec16
```

It then produced:

- one StreamPack segment;
- generated BCIR MLIR;
- portable C23;
- a provenance digest;
- a successful compiled self-check.

This demonstrates the actual source-to-plan-to-lowering path rather than only a documented architecture.

---

# 6. Semantic model

Primary sources:

- `bcir/model/graph.py`
- `bcir/model/lanes.py`
- `bcir/model/opcodes.py`

## 6.1 Registry-first resources

Memory and state are represented as registry entries rather than unrestricted raw pointers.

A resource records information such as:

- RID;
- name;
- domain;
- count or shape;
- layout and access form;
- generation;
- optional timing/lifetime facts.

Claims refer to resources through RIDs.

This enables:

- explicit identity;
- generation checks;
- provenance;
- resource-domain legality;
- deterministic serialization;
- stale-artifact rejection.

## 6.2 Claims

A claim expresses an operation together with its effects:

- claim ID;
- opcode;
- phase;
- read resources;
- write resources;
- element count;
- lane;
- stride class and stride;
- domain;
- hazard mode;
- bounds policy;
- optional timing;
- optional lifetime;
- optional tensor shape or lowering metadata on richer rails.

The claim graph describes **what must correspond**, while candidate realizations describe **how it may be executed**.

## 6.3 Explicit ordering

Ordering is not inferred from textual source order.

BCIR uses:

- an explicit phase DAG;
- explicit phase dependencies;
- read/write hazards;
- optional fine-grained async tokens;
- barriers as ordering edges.

## 6.4 Lane taxonomy

The normative lane enum is:

```text
U, UX, T, GGG, A, H
```

The system distinguishes regular/vectorizable, tiled, sparse/random, and specialized access classes. In particular:

- `T` is associated with tiled execution;
- `GGG` is the irregular/random access quarantine and scheduling tail;
- U/UX/T regular work can be scheduled separately from irreducible random work.

## 6.5 Stride classes

```text
SCALAR
UNIT
STRIDED
CACHELINE
TILE
RANDOM
```

## 6.6 Resource domains

```text
RAM
VRAM
NVM
MMIO
CXL
HBM
```

These are semantic domains, not proof that every physical device is installed.

## 6.7 Core opcodes

The base model includes:

```text
NOP
LOAD
STORE
ADD
SUB
MUL
ATOMIC_ADD
ATOMIC_SUB
ATOMIC_XOR
CMPXCHG
BARRIER
PHASE_ENTER
PHASE_LEAVE
GGG_LOAD
GGG_STORE
T_MACC
GEM_DISPATCH
PROV_NOTE
```

Higher-level tensor and format operations are represented on the MLIR and specialized Python rails.

---

# 7. Verifier laws R1–R25

BCIR currently defines twenty-five named law classes.

| Law | Meaning |
|---|---|
| **R1** | Registry uniqueness |
| **R2** | Registry/resource resolution |
| **R3** | Domain legality |
| **R4** | Phase-DAG legality |
| **R5** | Hazard and ordering legality |
| **R6** | Lane legality |
| **R7** | Bounds legality |
| **R8** | Cost completeness |
| **R9** | Plan legality |
| **R10** | Stream provenance |
| **R11** | Generation validity |
| **R12** | Lowering-contract legality |
| **R13** | Policy and decision provenance |
| **R14** | CIM/PIM dispatch legality |
| **R15** | DVFS clock legality |
| **R16** | Allocator/tier placement legality |
| **R17** | Numerical-accuracy contract |
| **R18** | Compositional call-graph integrity |
| **R19** | Synchronous timing legality |
| **R20** | Clock-domain crossing legality |
| **R21** | Pointer lifetime: use-after-free/double-free |
| **R22** | Shape consistency |
| **R23** | Dtype compatibility |
| **R24** | ASN.1 schema and encoding-rule legality |
| **R25** | X.692 ECN definition legality |

## 7.1 Important details

### R5: hazards, volatile and barriers

- Volatile access is represented structurally.
- Volatile claims require ordered/barriered semantics.
- Barriers prevent reordering and inhibit fusion across the barrier.
- Atomic RMW and compare-exchange carry explicit memory ordering on MLIR.

### R10–R13: artifact identity

BCIR checks:

- source-plan provenance;
- topology/mapping/data generations;
- lowering support;
- target identity;
- policy and calibration generation;
- hashes/digests of planning inputs.

R13 recomputes and cross-checks manifest components rather than trusting a supplied digest.

### R14–R17: smart-lowering laws

- PIM dispatch is restricted to eligible reduction forms.
- DVFS uses bounded Q8 clock factors and forbids inappropriate PIM overclocking.
- allocator placement observes tier capacities;
- numerical error must remain within the declared tolerance.

These laws are vacuous when their optional structures are absent.

### R18

Every call must resolve, and the call graph is acyclic. Recursive planning is not silently approximated.

### R19/R20

Optional timing metadata supports:

- clock frequency;
- setup/hold margin;
- synchronous and mixed synchronization;
- cross-domain RAW dependency checks.

### R21

The compiler can detect:

- use after free;
- double free;
- revalidation after assignment/allocation.

For the C frontend, policy can be:

```text
advisory
fallback
reject
```

### R22/R23

These cover tensor seams such as:

- matmul → activation;
- conv/attention → activation;
- embedding → RMSNorm;
- RoPE → attention;
- KV cache → GQA attention;
- native device tile multiples.

### R24

The law rail checks static ASN.1 properties before a value exists, including:

- module OID validity;
- legal universal tags;
- primitive/constructor consistency;
- sequence/set-of element structure;
- unique component tags;
- OPTIONAL versus DEFAULT;
- DEFAULT value presence;
- SET discriminability;
- canonical emission;
- transcode source and target rules;
- additive projections.

### R25

R25 governs combinations of ECN structures and properties whose invalidity is statically decidable. Current sources cite and enforce the statically decidable X.692 subclauses catalogued in
[`BCIR_ASN1_BUILDOUT_ROADMAP.md`](../BCIR_ASN1_BUILDOUT_ROADMAP.md) §4 (27 of them pinned to the fixture
that trips them by `test_asn1_ecn_law_parity.py`), including conditions involving:

- encoding object/class uniqueness;
- range-condition comparisons;
- alignment and start pointers;
- determinant usage;
- transform placement;
- bit-reversal units;
- conditional encoding objects;
- replacement structure rules;
- identification handles;
- repetition and constructor rules.

Primary source: `docs/BCIR_LANGREF.md:166-378`.

---

# 8. K_BCIR optimizer

Primary sources:

- `bcir/kbcir/cost.py`
- `bcir/kbcir/realize.py`
- `bcir/kbcir/rcsp.py`
- `bcir/kbcir/provenance.py`
- `bcir/kbcir/compose.py`
- `bcir/kbcir/egraph.py`

## 8.1 Twelve-dimensional cost vector

Every candidate can carry:

```text
compute
memory
fabric
sync
compile
thermal
power
reliability
security
accuracy
contention
verification
```

Costs are integer-valued. Context coupling uses fixed-point/Q8-style arithmetic rather than host floating point in the planning hot path.

## 8.2 Substrate profile \(H\)

A target profile describes facts such as:

- architecture/triple;
- ISA features;
- lane widths;
- vector width;
- cache line;
- gather penalty;
- affinity domains;
- memory channels;
- thermal and power density;
- base overhead;
- memory hierarchy;
- calibration generation.

Built-in cost-model profiles:

```text
x86_avx2
x86_avx512
arm64_neon
arm64_sve
riscv_rvv
nvidia_ptx
```

## 8.3 Runtime state \(\Theta\)

Theta carries runtime pressures such as:

- thermal;
- power;
- memory pressure;
- contention.

Standard examples include cool, hot, and memory-bound states. Real telemetry can be folded into a later generation through a calibrated table.

## 8.4 Candidate generation

For each claim, BCIR can enumerate legal realizations such as:

- scalar;
- vector width \(W\);
- strided;
- gather;
- UX bucket;
- tiled;
- fused variants;
- target-specific options.

An unsupported candidate is not assigned a high cost; it is removed from the legal candidate set.

## 8.5 Min-plus selection

`realize.optimize` constructs a layered realization DAG and performs deterministic min-plus selection. Context-dependent factors can account for:

- adjacent shared operands;
- fusion;
- target lane width;
- thermal pressure;
- memory tier;
- path context.

## 8.6 Constrained optimization

`rcsp.py` implements resource-constrained shortest-path selection:

\[
\min M(\pi)
\quad\text{subject to}\quad
R(\pi,\Theta)\le B(H,\Theta)
\]

Capabilities include:

- hard dimension budgets;
- feasibility checks;
- dominance pruning;
- Pareto-plan enumeration;
- explicit `Infeasible` refusal.

This is stronger than merely changing scalarization weights.

## 8.7 Overlap-aware planning

`gem/overlap.py` prices the actual concurrent schedule rather than only a serial sum:

- parallel work on separate domains combines by maximum;
- time-sharing combines additively;
- context coupling can change the best plan once scheduling is considered.

## 8.8 Composition and control flow

`compose.py` extends planning to a bounded region tree:

- leaf;
- sequence;
- conditional;
- call;
- function summary.

It supports conservative effects, undefined-callee refusal, and acyclic calls.

## 8.9 E-graph composition

`egraph.py` provides:

- e-classes;
- expression nodes;
- equality rewrites;
- constant folding;
- factorization;
- congruence rebuilding;
- saturation statistics;
- minimum-cost extraction;
- shared-block/CSE analysis.

`memory.py` wraps bounded saturation into a content-addressed “memory module” with fixpoint and idempotence checks.

## 8.10 Provenance and replay

A `ProvenanceManifest` binds:

- module;
- target;
- Theta;
- policy;
- calibration table;
- decision-rule generations and fingerprints.

The system supports deterministic replay and tests manifest equality against reproduced plans.

## 8.11 Calibration and learned guidance

Implemented mechanisms include:

- host microbenchmarking;
- quantized/frozen calibration tables;
- EWMA and linear calibrators;
- adaptive policy selection;
- regret ledgers;
- tile priors;
- channel priors;
- proposal-order accelerators;
- bounded MoE/GNN research;
- optimization-memory lookup.

The design rule remains:

> A learned mechanism may propose or reorder candidates; deterministic search and verification establish the result.

---

# 9. GEM scheduling and execution

Primary sources:

- `bcir/gem/streampack.py`
- `bcir/gem/concurrency.py`
- `bcir/gem/schedule.py`
- `bcir/gem/async_tokens.py`
- `bcir/gem/execute.py`
- `bcir/gem/dvfs.py`
- `bcir/gem/cim.py`

## 9.1 StreamPack hydration

GEM turns a selected plan into a hot executable `StreamPack`.

The dormant artifact is the semantic graph and plan. The hot artifact contains selected, ordered execution segments.

## 9.2 Deterministic execution

`execute.py` runs:

- phases in topological order;
- claims in deterministic claim-ID order where deterministic mode applies;
- optional per-claim kernels;
- per-phase telemetry.

## 9.3 Concurrent wave scheduler

`concurrency.py` forms waves:

- independent claims share a wave;
- RAW/WAR/WAW conflicts serialize;
- work is assigned across affinity domains;
- irregular GGG work can be placed into a separate tail stream.

## 9.4 Duration-aware scheduling

`schedule.py` adds:

- duration-aware LPT priority;
- earliest-finish-time placement;
- locality;
- memory-bandwidth knee calculations;
- event-driven scheduling;
- deterministic tie breaking.

## 9.5 Async token graph

`async_tokens.py` emits explicit dependencies:

- each claim forks a completion token;
- conflicting later claims await the relevant tokens;
- independent claims await nothing.

This permits fine-grained pipelining without adding unnecessary phase-wide barriers.

## 9.6 DVFS and power

The DVFS planner:

- classifies compute-bound versus memory-bound phases;
- proposes target clock factors;
- observes thermal and power constraints;
- can quantize plans to available silicon states;
- separates proposed plans from actual actuation results.

## 9.7 CIM/PIM

Eligible reduction segments can be annotated for PIM-style dispatch. This changes where a segment is intended to execute, but does not itself prove the presence of physical processing-in-memory hardware.

## 9.8 Power-rail scheduling

The scheduler can reason about concurrent phase power and produce power-rail decisions rather than treating every legal concurrent launch as thermally free.

---

# 10. StreamPack ABI

Primary sources:

- `bcir/abi/streampack_abi.py`
- `runtime/c/bcir_streampack.h`
- `runtime/c/bcir_runtime.c`
- `docs/kernel/BCIR_STREAMPACK_ABI.md`

## 10.1 Format identity

```text
magic:       BSPK
base version: 1
maximum supported version: 3
header size: 64 bytes
endianness: little-endian
```

## 10.2 Versioning

- **v1** is the frozen base ABI.
- **v2** adds pipeline/double-buffer information through reserved header space.
- **v3** adds segment dispatch/channel information.
- Changes are append-only and versioned.

## 10.3 Contents

A pack contains:

- source plan/provenance text;
- generation numbers:
  - topology;
  - mapping;
  - data;
- lane segments;
- prefetch records;
- blocks;
- trace notes;
- pipeline depth;
- per-segment dispatch and channel in later versions;
- integrity trailer/CRC.

## 10.4 Validation

The reference and C implementations check:

- magic;
- version;
- record boundaries;
- count/range arithmetic;
- CRC;
- trailing bytes;
- semantic consistency;
- generation where a live registry is available.

## 10.5 Tooling

`bcir-pack` provides:

```text
bcir-pack dis
bcir-pack hexdump
```

It validates before interpretation and supports a maximum input-size limit.

## 10.6 Native round trip

The C rail includes:

- decoder;
- encoder;
- claim graph + plan hydrator;
- deterministic executor;
- semantic verifier;
- Python/C byte-identity tests.

---

# 11. BCAB multi-backend artifact bundle

Primary sources:

- `bcir/abi/artifact_bundle.py`
- `runtime/c/bcir_artifact_bundle.{h,c}`
- `runtime/cpp/bcir_artifact_bundle.hpp`
- `docs/kernel/BCIR_ARTIFACT_BUNDLE_ABI.md`

## 11.1 Wire contract

```text
magic: BCAB
version: 1
header: 128 bytes
directory entry: 448 bytes
maximum entries: 1024
default hard size ceiling: 1 GiB
```

BCAB is a deterministic container. It does not replace standard payload formats.

## 11.2 Supported artifact kinds

```text
STREAM_PACK
ELF_OBJECT
ELF_SHARED
COFF_OBJECT
MACHO_OBJECT
ARCHIVE
WASM
LLVM_BITCODE
LLVM_IR
PTX
CUBIN
SPIRV
JVM_CLASS
CIL
C_SOURCE
CPP_SOURCE
SYCL_SOURCE
ASSEMBLY
ELF_EXECUTABLE
PE_EXECUTABLE
PE_SHARED
MACHO_EXECUTABLE
MACHO_SHARED
RAW_BINARY
```

## 11.3 Compatibility selection

A selection envelope can constrain:

- target triple;
- architecture;
- OS ABI;
- channel;
- feature set;
- accepted kind and format masks;
- endianness;
- pointer width;
- machine ID;
- target-manifest SHA-256;
- calibration generation;
- R12 requirement;
- debug allowance.

## 11.4 Tooling

`bcir-bundle` supports:

```text
list
hexdump
extract
select
dis
mnemonics
to-der
from-der
to-oer
from-oer
```

Validation precedes listing, extraction, or disassembly.

## 11.5 C++ wrapper

`ArtifactBundleView` provides:

- RAII-style error mapping;
- borrowed immutable payload views;
- indexed access;
- default or envelope-constrained selection.

---

# 12. Frontends and event transduction

## 12.1 MAP frontend

`bcir/frontends/map.py` reads a terse macro-assembly-like form containing:

- resource declarations;
- domains;
- operations;
- reads/writes;
- counts;
- lanes;
- strides.

It lowers directly into a BCIR `Module`.

## 12.2 ROP frontend

`bcir/frontends/rop.py` reads a declarative brace-delimited registry-first language containing:

- modules;
- resources;
- phases;
- claims;
- reads and writes;
- lane and stride metadata.

## 12.3 Event Transduction Layer

`bcir/etl/` provides a more general ingestion architecture:

- regex lexer;
- recursive-descent ROP parser;
- typed event streams;
- token-driven finite-state transducers;
- structured binary record descriptors;
- packed-field decoding.

The binary record decoder currently supports byte-aligned fields and fails closed on unsupported non-byte-aligned layouts.

The intended correspondence is:

```text
text / bytes / packet / telemetry
    -> tokens or fields
    -> events
    -> BCIR claims
    -> normal verification and planning
```

## 12.4 C frontend

Primary sources:

- `bcir/frontends/cfront/`
- `runtime/c/bcir_cfront.{h,c}`
- `runtime/c/bcir_cpp.{h,c}`
- `docs/languages/CFRONT_GUIDE.md`
- `docs/languages/C_MEMORY_DISCIPLINE.md`

The C frontend is a bounded driver/kernel-oriented C compiler frontend, not a claim to arbitrary ISO C completeness.

Its supported architecture includes:

- C11, C17 and C23/C2x mode selection;
- preprocessing;
- include paths;
- object/function/variadic macros;
- stringizing and token pasting;
- C23 `__VA_OPT__`;
- conditional compilation;
- dependency generation;
- compile-database ingestion;
- multi-file/project verdicts;
- target ABI selection;
- Clang-style and JSON diagnostics;
- syntax-only mode;
- LLVM fallback signaling;
- R21 policy;
- generated C;
- generated self-checks;
- link flags;
- linkable multi-translation-unit output.

The semantic subset includes driver-oriented constructs such as:

- fixed-width integer operations;
- pointers and arrays within documented bounds;
- structures, unions and bitfields;
- volatile/MMIO accesses;
- atomics;
- fences;
- register-map patterns;
- external library edges;
- source locations and diagnostics.

The Python frontend also contains explicit modeling for GNU inline assembly and port-I/O intrinsics. The native C twin has its own supported subset and explicit portable acquire/release fence emission; unsupported constructs are not silently assigned semantics.

## 12.5 Bounds quarantine

The C lowering can emit guarded access and call:

```c
bcir_bounds_quarantine(...)
```

Two behaviors exist:

- a weak abort-oriented default;
- a strong policy-driven recovery reference capable of actions such as clamping.

This is a controlled runtime policy seam, not an optimizer license to ignore bounds.

---

# 13. ASN.1 subsystem

The ASN.1 implementation is a complete subsystem with independent schema, transfer-syntax, law, runtime, artifact, measurement, and admission layers.

Primary trees:

```text
bcir/asn1/
bcir/frontends/asn1/
runtime/c/bcir_asn1*
runtime/c/bcir_per*
runtime/c/bcir_oer*
runtime/c/bcir_jer*
runtime/c/bcir_xer*
mlir/include/BCIR/BCIRAsn1Ops.td
mlir/include/BCIR/BCIREcnOps.td
```

## 13.1 Schema and semantic model

The Python schema layer supports a documented X.680 subset including:

- primitive types;
- tagged/prefixed types;
- SEQUENCE, SET, CHOICE;
- SEQUENCE OF and SET OF;
- OPTIONAL and DEFAULT;
- constraints;
- imports and assignments;
- open types;
- object identifiers;
- restricted and unrestricted string/time families on their relevant rails.

Unsupported notation fails rather than being approximated.

## 13.2 X.681/X.682/X.683

Implemented capabilities include:

- information object classes;
- `WITH SYNTAX`;
- objects and object sets;
- associated tables;
- table constraints;
- component-relation constraints;
- open-type resolution from a governing sibling;
- user-defined constraints;
- contents constraints;
- parameterized types;
- parameterized objects and object sets;
- structural actual-parameter substitution.

Resolution of an open type is enrichment: original octets remain available, and a resolved value is attached when a matching table row exists.

## 13.3 X.690 BER and DER

The X.690 rail implements the broad clause-8 value surface, including:

- BOOLEAN;
- INTEGER;
- ENUMERATED;
- REAL;
- BIT STRING;
- OCTET STRING;
- NULL;
- OID and relative OID;
- OID-IRI forms;
- SEQUENCE/SET and OF variants;
- CHOICE;
- explicit and implicit tags;
- open types;
- character strings;
- useful/time types.

Policy:

```text
emit: DER
accept canonical storage: DER
accept foreign input: BER
normalize: BER -> DER
```

CER is accepted as BER input where structurally valid but is deliberately not an emission candidate.

## 13.4 X.691 PER

Implemented on the Python rail:

- aligned and unaligned PER;
- BASIC and CANONICAL variants;
- constrained, semi-constrained and unconstrained integers;
- normally-small values;
- length determinants;
- 16K fragmentation;
- sequence/choice/collections;
- extension markers;
- extension addition groups;
- open-type wrappers;
- constraint roots and extension bits;
- permitted alphabets for supported types.

The implementation is validated against the standard’s Annex A examples.

The C side contains:

- a bounded bit reader;
- alignment;
- whole-number and length primitives;
- a schema-directed narrow plan decoder.

PER is not schema-free, by design of the standard.

Documented PER exclusions include types/clauses not required by the current BCIR schemas, such as selected REAL, BIT STRING, EMBEDDED PDV, and unrestricted-character-string paths.

## 13.5 X.696 OER/COER

Implemented:

- BASIC OER input;
- CANONICAL OER emission;
- constraint-dependent fixed-width forms;
- presence preambles;
- canonical SET ordering;
- collection counts;
- supported primitive and constructor types;
- Annex A byte-level validation.

The C decoder is schema-directed, as required by X.696.

## 13.6 X.693 XER/CXER

Implemented on the Python rail:

- BASIC XER;
- canonical XER emission;
- tag processing;
- escaping;
- supported primitive and composite values.

The C twin is intentionally the lexical/trust-boundary layer:

- tag scanning;
- bounded lexical checks;
- `xmlcstring` escaping.

Extended XER is not claimed.

## 13.7 X.697 JER

Implemented capabilities include:

- clauses 20–41 over the supported model;
- BASIC JER;
- BCIR canonical JER profile;
- ARRAY;
- BASE64;
- NAME;
- OBJECT;
- TEXT;
- UNWRAPPED;
- instruction precedence;
- deterministic emission;
- malformed-input rejection.

X.697 defines no standard canonical JER. Therefore:

> `BCIR canonical JER profile` is a private, versioned deterministic profile and carries no invented standards OID.

### Bounded JER reader

The Python bounded reader performs:

1. input and structural bound checks;
2. UTF-8 validation;
3. JSON grammar validation;
4. schema decoding;
5. optional canonical re-encoding and byte comparison.

It returns stable diagnostics including:

- error code;
- byte offset;
- required capacity.

### Framing

The optional BCIR JER frame carries:

- version;
- sequence;
- generation;
- length;
- CRC-32.

A frame must validate before its payload is exposed.

### Native C JER

The C implementation provides:

- no allocation;
- no recursion;
- caller-owned stack and scratch;
- bounded structural scanning;
- UTF-8 validation;
- event parsing;
- raw number-token handling;
- deterministic diagnostics.

Canonical-byte and full schema legality remain on the Python/compiled-plan rails rather than being independently reinvented in the scalar scanner.

### C++ JER acceleration

The hosted C++ layer provides:

- SSE2;
- AVX2;
- AArch64 NEON;
- scalar fallback;
- structural indexing;
- ASCII/UTF-8 fast paths.

The vector path decides where blocks or runs end; the scalar C implementation remains responsible for semantic rejection and exact diagnostic offsets.

## 13.8 X.692 ECN

The ECN implementation now includes the three major portions:

1. class/object/object-set and built-in encoding model;
2. user-defined encoding objects;
3. surface syntax and encoding links.

Capabilities include:

- encoding classes and objects;
- encoding object sets;
- EDM and ELM;
- built-in BER/PER sets;
- bit-level encoding spaces;
- justification;
- padding;
- transmission order;
- determinants;
- start pointers;
- conditions;
- replacement;
- bit reversal;
- identification handles;
- repetition;
- alternatives;
- optionality;
- concatenation;
- containment;
- value mappings;
- encoding links;
- parameterization;
- clause-24 transforms.

The current syntax compiler reports:

```text
ECN syntax version: 13
```

ECN has a production MLIR family and R25 verifier checks; it is no longer only a Python-side notation experiment.

## 13.9 Encoding selection

`selection.py` treats transfer syntax as a realization choice.

The fixed comparison set has ten candidates.

Canonical/selectable:

```text
DER
CANONICAL-PER-UNALIGNED
CANONICAL-PER-ALIGNED
COER
JER-BCIR-CANONICAL
```

Decode/noncanonical targets:

```text
BER
BASIC-PER-UNALIGNED
BASIC-PER-ALIGNED
BASIC-OER
JER
```

Selection objectives include:

```text
none/status quo
wire size
encode latency
decode latency
```

The rules are:

1. representability and round trip first;
2. cost second;
3. noncanonical formats cannot become digested emission choices.

Exact wire length and measured timing are stored separately.

## 13.10 Certified ASN.1 selection

`certified.py` adds:

- distribution-free timing intervals;
- exact integer coverage reporting;
- generation-tagged cost tables;
- candidate indistinguishability;
- deterministic tie-breaking;
- native-table provenance;
- target/corpus identity;
- Pareto and RCSP selection;
- multi-stage byte budgets;
- certificate records.

Timing intervals that overlap do not fabricate a winner. Exact wire size can serve as a deterministic tie-break.

## 13.11 ASN.1 compiled forms

The subsystem has several representations for different purposes:

- `schema.py`: semantic types and constraints;
- `dialect.py`: Python dialect representation;
- `program.py`: canonical program representation;
- `graph.py`: flat node/edge graph capable of cycles without recursive JSON nesting;
- `surface.py`: sparse textual/presentational view;
- `manifest.py`: schema-bound channel/device/selection manifests;
- `encode_plan.py`: schema-directed write plan, currently plan version 5;
- `jer_plan.py`: JER read/schema plan, version 1;
- `staged.py`: candidate admission/install model.

The graph’s canonical JER is content-addressed. Presentation metadata is kept separate so formatting changes do not change semantic identity.

## 13.12 ASN.1 artifact projections

BCIR projects semantic StreamPack and BCAB values into standard encoding families.

For StreamPack:

- native ABI remains authoritative and frozen;
- ASN.1 projection is additive;
- DER can reconstruct native bytes through freestanding C;
- the reconstructed native result is byte-identical, including re-derived version, reserved fields, and CRC.

For BCAB:

- DER/BER;
- OER/COER;
- canonical aligned/unaligned PER;

can round-trip through semantic projections while rebuilding native offsets and integrity fields canonically.

## 13.13 Staged self-modification model

`staged.py` models:

```text
running program
 -> canonical JER proposal as data
 -> schema and R-law verification
 -> K_BCIR selection
 -> compilation
 -> signing/admission
 -> generation-tagged quiescent install
 -> rollback
```

Security properties modeled:

- proposal is data, not executable memory;
- verification happens before compilation;
- compilation calls are observable;
- digest and authority signature are distinct;
- install checks signatures again;
- stale generations are refused;
- in-flight calls block installation and rollback;
- rollback advances generation.

The signature is currently HMAC-SHA256. It demonstrates shared-key admission authority, not deployment-grade asymmetric signer identity.

## 13.14 ASN.1 test surface

The focused inventory established approximately:

```text
50 ASN.1-focused test modules
1,001 ASN.1-focused test functions
```

Coverage includes positive, negative, parity, canonicality, differential, native C, C++, and law-fixture testing.

---

# 14. MLIR implementation

Primary sources:

```text
mlir/include/BCIR/
mlir/lib/
mlir/test/
tools/wsl/
tools/irdl/
```

## 14.1 Dialect families

The current ODS tree defines operation families for:

- core modules, registries, resources, phases and claims;
- K_BCIR plans, paths, candidates, policies and Theta;
- GEM segments and tensor operations;
- verification/provenance records;
- ASN.1;
- ECN;
- parsing;
- finite-state transducers;
- events;
- binary formats;
- async tokens;
- memory;
- optimization;
- lowering contracts;
- targets;
- traces.

## 14.2 Registered passes

The 37 passes are:

### Fundamental and conversion

```text
Verify
PromoteLanes
ConvertToLLVM
```

### Planning and evidence

```text
ClassifyLanes
SelectRealization
CostModel
Plan
Overlap
OverlapOptimize
Sense
Rcsp
RcspPlan
Bundle
Explain
Replay
Compose
```

### Scheduling and smart lowering

```text
Cim
Dvfs
ScheduleEft
Async
PowerRail
AllocPool
Batch
Schedule
LowerToLLVM
CacheContention
LayoutPivot
```

### Tensor cost passes

```text
GemMatmulCost
GemActivationCost
GemConvCost
GemAttentionCost
```

### Tensor lowering/fusion

```text
LowerGemMatmul
LowerGemMatmulBuffer
LowerGemActivation
LowerGemConv
LowerGemAttention
FuseMatmulActivation
```

## 14.3 Production verification

`-bcir-verify` checks the structurally decidable form of R1–R25.

The Python oracle-to-MLIR bridge emits candidate paths and a selected path while allowing the MLIR rail to recompute:

- lane classification;
- legal candidate selection;
- score;
- lowering state.

## 14.4 Tooling and corpus

The MLIR rail includes:

- TableGen validation;
- a compiled `bcir-opt`;
- IRDL projection usable with stock `mlir-opt`;
- positive examples;
- verifier-negative fixtures;
- pass tests;
- lowering tests;
- bytecode round trips;
- FileCheck assertions;
- Python-generated differential corpora;
- target capability matrices.

---

# 15. Lowering and code generation

## 15.1 Portable C23

`bcir/lower/c_kernel.py` is the broad native source backend.

It supports:

- selected lane-aware elementwise kernels;
- bounds-safe tails;
- `restrict` contracts;
- floating-point contraction policy;
- quantized kernels;
- reductions;
- gather/strided work;
- tensor kernels;
- external numerical-provider wrappers;
- self-check harnesses.

The selected full hardware lane may be represented as an idiomatic loop that the resident compiler vectorizes rather than as hard-coded architecture intrinsics.

## 15.2 Textual LLVM IR

`bcir/lower/llvm.py` supports a deliberately bounded subset:

- one selected executable claim;
- two reads;
- one write;
- add/sub/mul;
- scalar or vector LLVM IR;
- AOT compilation and execution.

Unsupported arbitrary graphs are refused rather than truncated.

## 15.3 JIT

`bcir/lower/jit.py`:

- emits the same kernel and self-check as AOT;
- compiles to LLVM IR;
- links with `llvm-link`;
- runs with `lli`.

## 15.4 WASM

`bcir/lower/wasm.py`:

- compiles the bounded kernel through Clang’s `wasm32` target;
- links with `wasm-ld`;
- validates the module;
- executes through Node where available.

## 15.5 Stack-machine lowering

`stackify.py` lowers a small expression graph once to generic stack operations, then maps it to:

- WASM;
- JVM bytecode;
- .NET CIL.

`jvm_class.py` assembles the supported JVM subset directly. CIL tests use `ilasm` and Mono where available.

## 15.6 `llc` target code generation

Codegen descriptors exist for:

| Target | Output |
|---|---|
| `x86_64` | ELF object |
| `aarch64` | ELF object |
| `riscv64` | ELF object |
| `nvptx64` | PTX assembly |
| `bpf` | verifier-oriented scalar ELF object |
| `spirv64` | best-effort SPIR-V assembly |
| `c` | portable C fallback |

SPIR-V depends on an LLVM build containing the relevant backend. Its absence is reported, not replaced by a dummy artifact.

## 15.7 Numerical/tensor emitters

The C lowering contains emitters or wrappers for:

- GEMM and tuned GEMM;
- FFT and 2-D FFT;
- dense linear solve;
- OLS;
- eigensolver/PCA paths;
- GSL statistics;
- SLEEF vector exponential;
- libcerf;
- convolution;
- attention;
- layer normalization;
- recurrent cells;
- SVM;
- trees;
- K-means;
- activation;
- matmul+activation fusion;
- SYCL saxpy/reduce/matmul.

External libraries remain trusted provider edges rather than reimplemented algorithms.

## 15.8 Specialist synthesis

`specialist.py` can produce shape-baked, loop-unrolled C for hot, fixed shapes, while retaining generic kernels for dynamic shapes.

---

# 16. Native C runtime and compiler

`runtime/c/` is split between freestanding and hosted components.

## 16.1 Freestanding philosophy

Core components generally use:

- fixed-width integers;
- caller-owned buffers;
- no hidden heap;
- bounded iteration;
- explicit capacities;
- total status returns;
- checked arithmetic;
- no recursion for attacker-controlled wire structure.

Hosted components introduce allocation or diagnostics only through explicit contracts.

## 16.2 Core components

| Component | Capability |
|---|---|
| `bcir_streampack` | Frozen ABI definitions |
| `bcir_runtime` | StreamPack decode and semantic walk |
| `bcir_encode` | StreamPack encoding |
| `bcir_exec` | Deterministic execution |
| `bcir_hydrate` | C graph+plan to StreamPack |
| `bcir_cir` | Native claim graph |
| `bcir_plan` | Minimal native plan |
| `bcir_verify` | Native verifier subset and diagnostics |
| `bcir_provenance` | Digest twin |
| `bcir_channel` | Channel manifest/routing |
| `bcir_runtime_channel` | In-process driver-hook ABI |
| `bcir_artifact_bundle` | BCAB reader/selector |
| `bcir_binrec` | ETL packed-record decoder |
| `bcir_telemetry_frame` | UART frame codec |
| `bcir_cfront` | Native C frontend |
| `bcir_cpp` | Native preprocessor |
| `bcir_diag` | Source diagnostics |
| `bcir_quarantine` | Bounds policy |
| `bcir_asn1` | X.690 BER/DER |
| `bcir_per` | PER primitives |
| `bcir_per_plan` | Schema-directed PER subset |
| `bcir_oer` | OER subset |
| `bcir_jer` | Bounded JER scanner/parser |
| `bcir_xer` | XER lexical layer |
| `bcir_emit` | Plan-driven ASN.1 encoding |
| `bcir_asn1_streampack` | DER to native StreamPack |
| `bcir_q8_model` | BCIRQ8 loader |
| `bcir_q4_kernel` | Q4/Q8 kernels |
| `bcir_ai_kernels` | Native AI primitives |
| `bcir_decode` | LLM decode-stage kernels |
| `bcir_llama` | Standalone greedy inference |
| `bcir_train` | Native training-stage kernels |
| `bcir_x86_interrupt` | x86 interrupt-frame ABI |

## 16.3 Native verifier coverage

The C verifier documents:

- R1–R8 over the graph;
- R9 over the plan;
- R10–R11 over StreamPack;
- R12 lowering support;
- R13 provenance;
- relevant R14–R18 behavior for its subset;
- advisory R21 lifetime walking.

The full richer R19–R25 law surface remains principally the Python/MLIR responsibility.

## 16.4 Runtime channel ABI

The direct in-process channel interface supports:

```text
init
reset
open
claim
map
submit
sync
next_event
close
```

A loopback implementation exists.

This is an in-process vtable, not an IPC, kernel, or DMA transport.

---

# 17. C++ handoff

## 17.1 Single-node orchestrator

`SingleNodeOrchestrator` is real:

- admits immutable StreamPack through the C verifier;
- creates one whole-pack shard;
- re-enters the freestanding C segment walk;
- returns claim dispatch order;
- is tested against the direct-C path.

## 17.2 Dynamic graph orchestrator

The class and interface exist, but dispatch deliberately raises an error. It documents how a future mutable C++ graph builder would freeze results back into StreamPack.

## 17.3 Distributed orchestrator

The class provides deterministic segment-range sharding logic, but real dispatch is a stub. No MPI/NCCL dependency or cluster runtime is shipped.

Thus:

> C++ single-node handoff is executable; dynamic and distributed execution are explicit non-silent placeholders.

---

# 18. Hardware channels

Primary sources:

- `bcir/channels.py`
- `bcir/channel_plugin.py`
- `channels/*.json`
- `bcir/lower/sycl_dispatch.py`

## 18.1 Built-in channels

```text
x86_avx2
x86_avx512
arm64_neon
arm64_sve
riscv_rvv
nvidia_ptx
fpga_systolic
hbm_pim
nvme_stream
```

The channel set is broader than the six core cost profiles because some channels are modeled extension points rather than CPU target profiles.

## 18.2 Channel manifest

A `channel.json` records sections for:

- format version;
- name and kind;
- provenance;
- modeled status;
- architecture match;
- capabilities;
- code generation;
- runtime hooks;
- calibration;
- target profile.

Channels can be registered through:

- JSON manifests;
- Python entry points;
- core registration.

Duplicate replacement is refused.

## 18.3 Routing

The router derives required capabilities from a claim and selects a suitable channel. It does not route an operation to a channel that lacks the necessary declared capability.

## 18.4 Current execution reality

- Host CPU C/LLVM execution is real.
- Cross-target object/PTX generation is real where LLVM has the backend.
- SYCL has a resident dispatch path when a suitable compiler/runtime is installed.
- FPGA, HBM/PIM, and NVMe channels are modeled planning surfaces without physical device proof in this validation.
- PTX generation is not GPU execution.
- eBPF object generation is not kernel attachment.
- SPIR-V generation is toolchain-dependent.

---

# 19. Device manifests, static memory, and HAM

## 19.1 Device manifest

`device_manifest.py` models immutable hardware facts:

- banks;
- capacities;
- tiers;
- native tile sizes;
- interconnect distance matrix;
- target identity.

Checks include:

- positive capacities and tiles;
- consistent parallel arrays;
- square distance matrix;
- zero diagonal;
- positive off-diagonal;
- symmetry;
- probe agreement;
- explicit bank moves.

Runtime probing may veto disagreement but does not rewrite the static schema.

## 19.2 Static memory planning

`static_memory.py` assigns touched resources to offsets in declared banks.

Capabilities:

- bank binding;
- byte offsets;
- phase-lifetime analysis;
- reuse for disjoint lifetimes;
- overlap refusal for simultaneously live resources;
- independent plan verification.

This is a static compiler artifact, not a runtime allocator.

## 19.3 Hierarchical Access Memory

`ham.py` plans semantic resource movement across memory banks.

Inputs:

- hardware envelope;
- declared directed links;
- capacities;
- resources and access workload;
- optional policy identity.

Outputs:

- movement actions;
- residency actions;
- bank peaks;
- execution plan;
- replay evidence;
- lowered BCIR graph;
- StreamPack.

A route exists only if every required directed link is explicitly present.

HAM does **not** itself:

- issue DMA;
- configure CXL;
- establish peer-to-peer transfers;
- prove physical topology;
- perform live driver operations.

CLI:

```text
bcir-ham-plan --hardware ... --workload ... --plan-out ... [--pack-out ...]
```

---

# 20. Telemetry, silicon signals, and calibration

## 20.1 DataDNA telemetry

`bcir/telemetry.py` records per-segment evidence such as:

- cycles;
- bytes;
- misses;
- thermal pressure;
- voltage pressure;
- utilization;
- provenance;
- generation.

Sinks include:

- null;
- in-memory list;
- validating wrapper;
- file;
- broker;
- shared ring;
- durable log;
- Kafka-oriented adapter.

## 20.2 Telemetry ring and durability

The ring tracks:

- publication;
- overwrite/loss evidence;
- sequence;
- capacity;
- validation.

Durable inputs use bounded, strict parsing and path-identity protections.

## 20.3 UART telemetry frame

```text
magic: BTLM
version: 1
header: 22 bytes
```

The frame is:

- self-delimiting;
- sequence-bearing;
- timestamped;
- CRC-sealed;
- stream-resynchronizable.

Python and C twins are byte-compared.

## 20.4 Signal registry

The signal-provider registry gives typed definitions for:

- thermal pressure;
- die temperature;
- RAPL energy;
- CPU frequency;
- cache capacity;
- PMU availability;
- GPU power;
- BMC power;
- memory bandwidth;
- fabric bytes;
- throttle state;
- reliability;
- hwmon power.

Each reading carries units, provenance, sampling model, and temporality.

## 20.5 Real-host probes

`silicon.py` can inspect, when exposed by the host:

- cache topology;
- cache capacities;
- cpufreq;
- PMU counters;
- RAPL energy;
- on-die thermal;
- OS counters.

Missing signals are reported as unavailable rather than synthesized as measured values.

## 20.6 Derived metrics and export

`telemetry_metrics.py` calculates:

- min/max/average/RMS/count;
- trigger-to-target spans;
- plan-cost sensitivity;
- sampling budgets.

`telemetry_export.py` provides:

- Prometheus text;
- scrape output;
- OTLP-shaped metrics and JSON;
- push adapters;
- Redfish metric reports.

## 20.7 Calibration loop

Telemetry can feed:

```text
measure
 -> validate
 -> quantize
 -> freeze generation-tagged table
 -> replan
 -> compare/replay
```

It cannot change an in-flight plan or relax a law.

---

# 21. Numerical, tensor, and ML capabilities

The Python K_BCIR rail contains a broad numerical and ML reference layer.

## 21.1 Quantization

### Q8

- per-group signed codes;
- shared power-of-two scale;
- dequantization;
- round-trip error;
- integer dot;
- scaled dot;
- accumulator-width analysis.

### Q4

- packed two’s-complement nibbles;
- low-nibble-first format;
- forbidden `-8` to preserve symmetric range;
- group size 32;
- power-of-two scale exponent;
- SmoothQuant-style calibration;
- Q4/Q8 dot;
- format admission evidence.

## 21.2 Tensor primitives

Implemented reference/planning families:

- matmul;
- activation:
  - ReLU;
  - sigmoid;
  - tanh;
  - GELU;
  - softmax;
- 2-D convolution;
- scaled dot-product attention;
- grouped-query attention in the model decoder;
- layer normalization;
- RMSNorm;
- RoPE;
- recurrent cells:
  - RNN;
  - LSTM;
  - GRU;
- transformer encoder block;
- matmul+activation fusion;
- SoA/AoS layout pivot.

Each planned family separates:

- reference result;
- candidate realization;
- cost;
- verification.

## 21.3 Numerical-provider integrations

BCIR wraps rather than reimplements selected mature libraries:

- BLAS GEMM;
- FFTW 1-D and 2-D FFT;
- LAPACK dense solve;
- LAPACK OLS;
- eigensolver/PCA paths;
- GSL statistics;
- SLEEF vector math;
- libcerf.

Provider availability and measured performance are planning evidence. They do not define mathematical or BCIR legality.

## 21.4 Classical and unsupervised ML

Reference capabilities include:

- KNN classification/regression;
- decision-tree prediction;
- linear/RBF SVM;
- Gaussian Naive Bayes;
- K-means;
- standard scaling;
- min/max scaling;
- K-fold partitioning;
- autoencoder forward path;
- embedding lookup;
- PCA;
- OLS.

Training versus prediction/transform capability is explicitly separated where only one side has native lowering.

## 21.5 Automatic differentiation

`autodiff.py` implements:

- content-addressed/hash-consed expression DAG;
- forward evaluation;
- reverse-mode differentiation;
- symbolic gradient graph;
- numerical gradient comparison;
- Hessians;
- second-difference Hessian checks;
- closed differentiable primitive registry;
- lowering-closure checks.

`autodiff_program.py` adds:

- admission;
- quarantine reasons;
- scalar mutation functionalization;
- bounded loop tracing;
- defunctionalization;
- rematerialization plans;
- AD-order comparisons.

Programs outside the admitted subset are refused with structured reasons.

## 21.6 Losses and optimizers

Losses include:

- MSE;
- softmax cross-entropy;
- binary cross-entropy with logits;
- hinge loss.

Optimizers include:

- SGD;
- momentum;
- RMSProp;
- Adam.

There are both reference implementations and portable C update emitters.

## 21.7 Training

`training.py` provides a bounded supervised loop with:

- minibatching;
- train/validation split;
- early stopping;
- classification and regression metrics;
- linear and MLP reference models.

`train_graph.py` turns a training step into explicit BCIR claims:

```text
forward
 -> activation
 -> loss
 -> reduce
 -> backward
 -> update
```

It supports:

- planned runs;
- StreamPack hydration;
- scheduling;
- streaming;
- certificates.

`lower/autodiff_kernel.py` emits forward/backward/SGD C. `runtime/c/bcir_train` supplies selected native stage kernels.

---

# 22. Open-weight model stack

Primary sources:

```text
bcir/frontends/models/
bcir/hosted/models/
runtime/c/bcir_q8_model*
runtime/c/bcir_decode*
runtime/c/bcir_llama*
```

## 22.1 Manifest-first ingestion

`manifest.py` builds a model manifest before loading weights.

It records:

- architecture;
- license;
- tokenizer reference;
- shard inventory;
- shard hashes;
- dtype histogram;
- parameter count;
- context length.

Safetensors headers are read without reading payload bytes.

## 22.2 Header-only tensor inventory

`inventory.py` records:

- tensor names;
- dtypes;
- shapes;
- physical offsets/spans;
- shard layout;
- a header/layout digest.

The header digest is planning identity, not a substitute for content hashes.

## 22.3 Tokenizers

Dependency-free references exist for:

- GPT-2/Qwen-style byte-level BPE;
- special tokens;
- lossless byte/unicode mapping;
- chat rendering;
- SentencePiece protobuf reading;
- score-based SentencePiece BPE.

## 22.4 Reference decoder

A bounded Llama/Gemma-style dense decoder includes:

- embeddings;
- pre-norm layers;
- Q/K/V projections;
- RoPE;
- causal attention;
- grouped-query attention;
- KV cache;
- feed-forward/SwiGLU-style path;
- RMSNorm;
- logits;
- greedy decode.

## 22.5 Real-weight ingestion

The HF ingestion layer converts:

- `config.json`;
- Safetensors tensor layouts;
- `[out,in]` linear orientation;
- head geometry;
- RoPE interleaving;
- model tensor naming;

into the reference decoder’s weight structure.

## 22.6 Quantized artifacts

Capabilities include:

- per-group Q8 model quantization;
- drift records;
- deterministic BCIRQ8 persistence;
- standalone C loader;
- native decode kernels;
- standalone greedy inference.

## 22.7 Serving

`serve.py` models generation as a proof-carrying planned artifact:

- prefill and decode phases;
- session module;
- generation certificate;
- token DFA;
- stream events;
- planned `generate()`.

## 22.8 Paged KV and batching

`paged_kv.py` provides:

- page-table-backed KV resources;
- per-page generations;
- eviction rules;
- live-session protections;
- session admission;
- continuous-batching claim graphs;
- batch certificates.

## 22.9 Model capacity and placement assessment

`bcir-model-assess` can operate without opening weights.

Inputs:

- tensor inventory;
- hardware envelope;
- workload.

Outputs:

- format-size estimates;
- training-state sizes;
- bank peaks;
- placement candidates;
- prediction intervals;
- cost report;
- selected execution plan;
- verified StreamPack.

Optional bounded native or Python-oracle microbench records can be attached.

---

# 23. Hosted model and training laboratory

The hosted layer is opt-in and allows PyTorch and numerical dependencies without contaminating the dependency-free core import path.

## 23.1 Hosted Llama model

Capabilities:

- independent PyTorch Llama reference;
- RMSNorm;
- attention;
- MLP;
- decoder layers;
- backbone and language model;
- deterministic AdamW training;
- generation-based checkpoints;
- pickle-free checkpoint handling;
- deterministic export back through BCIR’s strict ingestion boundary.

## 23.2 Training stages

Bounded hosted stages include:

- supervised fine-tuning;
- embedding distillation;
- reward-model training;
- DPO;
- PPO;
- GAE;
- reasoning SFT;
- small supervised MLP/GRU/transformer models.

## 23.3 Corpus pipeline

The data pipeline provides:

- raw and prepared document identities;
- deterministic preparation;
- provenance-preserving output;
- prepared-corpus manifests;
- byte-fallback BPE training for small fixtures.

## 23.4 Adaptive transformer research

The dependency-free semantic rail and hosted PyTorch rail cover bounded versions of:

- tied-depth/looped models;
- variable-width schedules;
- reference-plus-sliding attention;
- exogenous anchors;
- coarse-to-fine multi-patch processing.

These are research contracts, not production kernels or latency evidence.

## 23.5 Byte-native research

Implemented bounded research surfaces include:

- raw-byte vocabulary and lossless references;
- Byte Latent Transformer shapes;
- patch policies;
- block denoising;
- speculative verification;
- diffusion draft/verify;
- MambaByte selective SSM;
- byteified transplant plans;
- measured-ingest choice;
- tiny hosted training.

Again, these are bounded semantic and hosted reference implementations.

## 23.6 Sequence-interface research

Capabilities include:

- cross-tokenizer alignment;
- projected log probabilities;
- tokenizer expansion;
- continued-BPE initialization;
- finite scalar quantization;
- causal-series codec;
- progressive growth stages;
- hosted row freezing and staged training.

## 23.7 Provider contracts

The hosted training provider layer separates:

- teacher inference/embedding providers;
- remote compute providers.

A teacher may produce immutable targets; it is not treated as a remote gradient source. A remote compute provider executes BCIR-owned code and returns BCIR-owned artifacts.

## 23.8 Hardware policy learning

A bounded PyTorch GNN/Transformer can learn priors for hardware-plan search. Promotion still requires:

- deterministic candidate legality;
- static-memory safety;
- StreamPack validity;
- measured-versus-simulated provenance;
- promotion certificate.

---

# 24. Performance and TMSAO audit

`bcir/performance_audit.py` defines a bounded cross-organ audit.

It exercises representative:

- graph;
- optimizer;
- scheduler;
- StreamPack;
- memory;
- telemetry;
- quantization;
- ML;
- hardware-search;

operations and records deterministic correctness plus host-relative timing.

It does **not** claim to prove theoretical maximum system performance. Shared-host timings are informational unless run on a controlled performance rig.

`bcir-tmsao-audit` exposes this rail.

The latest research proposal gives TMSAO a broader formal interpretation involving typed regions, objective registries, lower bounds, and whole-system certificates. That broader formulation is not yet the current executable optimizer.

---

# 25. Security architecture

Primary sources:

- `SECURITY.md`
- `docs/research/BCIR_SECURITY_THREAT_MODEL.md`
- `docs/research/BCIR_SECURITY_RED_TEAM_AUDIT_2026-07-15.md`

## 25.1 Security objective

Hostile source, model, manifest, telemetry, and wire input should either:

- become one bounded, unambiguous, verified artifact; or
- fail before externally visible mutation.

## 25.2 Core invariants

1. Parse once and interpret once.
2. Reject duplicate or ambiguous identity.
3. Bound bytes, recursion, geometry and work before allocation or mutation.
4. Preflight before commit.
5. Make ownership explicit.
6. Check generations and content identity at consumption.
7. Serialize temporal transitions.
8. Keep legality independent of telemetry and learning.
9. Do not carry process pointers or mutable Python objects across future privilege boundaries.

## 25.3 Input protections

Current implementations use combinations of:

- strict duplicate-key JSON parsing;
- non-finite-number rejection;
- exact schemas;
- checked add/multiply/alignment;
- span and overlap checks;
- UTF-8 validation;
- count/depth/size ceilings;
- CRC and digest checks;
- generation checks;
- no-follow/stable-file identity checks;
- private temporary directories;
- subprocess timeouts;
- compiler/test worker limits.

## 25.4 Ownership

- Freestanding C is generally heap-free.
- Hosted C uses explicit allocator injection and cleanup contracts.
- Borrowed views document required backing lifetimes.
- C++ wrappers preserve borrowed immutable payload semantics.
- destroy/reset operations are intended to be idempotent.

## 25.5 Current privilege boundary

The repository contains no current:

- resident BCIR kernel module;
- BCIR device node;
- privileged BCIR service;
- cross-process BCIR IPC transport;
- DMA/IOMMU implementation;
- network listener;
- setuid transition.

RuntimeChannel is a same-process vtable.

Consequently, BCIR is not a sandbox: Python plugins, resident compilers, and native libraries execute with the caller’s authority.

## 25.6 Fuzzing and sanitizers

The C trust-boundary suite covers formats including:

- StreamPack;
- binary records;
- telemetry frames;
- BCIRQ8;
- BCAB;
- BER/DER;
- DER-to-StreamPack;
- PER;
- OER;
- JER;
- plan-driven encoding.

CI includes ASan, UBSan, LSan, static analysis, and libFuzzer campaigns. A scheduled deep sweep adds Valgrind, cppcheck, scan-build, larger frontend campaigns, and longer fuzzing.

---

# 26. CLI and operator surfaces

Installed Python console scripts:

```text
bcir
bcir-pack
bcir-bundle
bcir-registry
bcir-model-assess
bcir-asn1c
bcir-ham-plan
bcir-tmsao-audit
```

Additional module driver:

```text
python -m bcir.frontends.cfront
```

## 26.1 Main `bcir` CLI

Capabilities include:

- program selection;
- target selection;
- Theta and policy;
- LLVM emission;
- MLIR emission;
- C emission;
- AOT execution;
- JIT execution;
- WASM execution;
- wave schedule;
- EFT schedule;
- async token schedule;
- RCSP budgets;
- overlap pricing;
- calibration table loading;
- regret report;
- MoE research;
- proposal acceleration;
- provenance manifest;
- e-graph analysis;
- soft-temperature plan distribution;
- target code generation;
- host calibration;
- silicon probing;
- benchmarking;
- joint bundle optimization;
- explain records;
- replay;
- witness reduction.

## 26.2 Registry operator

`bcir-registry` provides governed in-process inspection and mutation. A write increments `data_gen`, thereby making already hydrated StreamPacks stale under R11.

## 26.3 ASN.1 compiler driver

`bcir-asn1c` supports:

- list;
- print;
- check;
- DER encode/decode;
- BER-tolerant decode;
- JER encode/decode;
- BASIC versus BCIR-canonical JER;
- framed JER;
- DER/JER transcode;
- hex/raw I/O;
- source-position diagnostics.

---

# 27. Testing and CI evidence

## 27.1 Local thorough Python run

The fresh checkout discovered **3,280 tests**.

Initial thorough result:

```text
3278 passed, 2 failed
```

The two failures were:

1. WASM execution under an obsolete Node.js runtime;
2. the test-runner default-timeout self-test while the validation process had explicitly overridden that timeout.

After:

- upgrading Node to `22.23.2`;
- removing the timeout override for the self-test;

the targeted rerun reported:

```text
FAILED_TESTS_RERUN_PASS 2/2
```

The exact evidence is therefore:

> One complete aggregate run produced 3,278 passes and two environment-induced failures, and both failures passed after correcting the environment.

There was not a second single-command aggregate `3280/3280` run.

## 27.2 ASN.1 result

Neither aggregate failure was an ASN.1 test, so no registered focused ASN.1 failure was reported in the complete run.

## 27.3 Toolchain

The coherent Linux environment has access to:

```text
gcc
g++
cmake
ninja
clang
clang++
lli
llvm-link
llc
llvm-as
opt
mlir-opt
mlir-tblgen
FileCheck
wasm-ld
node
mono
ilasm
valgrind
ccache
jq
llvm-bolt
```

LLVM/Clang/MLIR:

```text
22.1.8
```

The coherent prefix is:

```text
/usr/lib/llvm-22/bin
```

with:

```text
LLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm
MLIR_DIR=/usr/lib/llvm-22/lib/cmake/mlir
```

## 27.4 GitHub Actions

The exact commit’s CI run concluded successfully across 13 matrix-expanded jobs, including:

- two Python oracle shards;
- C runtime;
- C analysis;
- Ubuntu and Windows host portability;
- Ubuntu and Windows hosted model gates;
- LLVM training;
- native AArch64 oracle;
- native AArch64 C runtime;
- LLVM/MLIR 22 rail;
- documentation governance.

## 27.5 Local validation boundary

Not freshly completed as standalone local aggregate gates in this checkout:

- full MLIR-22 build/pass aggregate;
- standalone full sanitizer sweep;
- standalone Valgrind deep sweep;
- standalone long libFuzzer campaign.

The exact commit’s CI is evidence for those remote gates but is kept distinct from local execution.

---

# 28. LLVM training and education

`llvm-training/` contains twenty numbered topic areas:

```text
00 foundations
01 syntax
02 types
03 constants
04 memory
05 control flow
06 metadata
07 optimization
08 pitfalls
09 vectorization
10 grammar
11 concurrency
12 backend/JIT
13 advanced IR
14 MLIR bridge
15 binary analysis
16 exception handling
17 new pass manager
18 MLIR lowering to LLVM
19 hardware-aware work
```

Additional infrastructure includes:

- exercise manifests;
- prompt templates;
- reference solutions;
- invalid examples;
- adversarial fixtures;
- deterministic variants;
- an autograder;
- deterministic dataset export;
- binary-analysis evidence;
- CSV schema checks;
- opaque-pointer validation;
- opt-diff goldens;
- lit tests;
- MLIR examples;
- BCIR mapping exercises;
- `llc` smoke;
- none/thin/full LTO matrix;
- optional BOLT smoke;
- CMake aggregate quality targets.

This is both curriculum and compiler-toolchain regression corpus, but it does not define the semantics of the BCIR production dialect.

---

# 29. Documentation and governance

The repository has several documentation classes:

## Normative/current

- `docs/BCIR_LANGREF.md`
- `docs/STATUS.md`
- `docs/PARITY.md`
- ABI documents under `docs/kernel/`
- active language guides

## Generated governance

CI checks:

- `STATUS.md` regeneration;
- relative links;
- references to retired paths;
- summary claims against source;
- hot/cold import quarantine.

## Roadmaps

Roadmaps may mix:

- historical plans;
- delivered phase notes;
- current boundaries;
- future gates.

Their “built” claims should be cross-checked against source and tests.

## Research

Research documents include:

- performance studies;
- threat models;
- architecture proposals;
- comparison studies;
- GEM+/TMSAO design.

They are evidence and design input, not automatically product law.

---

# 30. Latest GEM+/TMSAO architecture proposal

The latest commit proposes an integrated future architecture.

## 30.1 Typed regions

Rather than optimizing isolated claims with partly separate organs, GEM+ would place claims into typed regions such as:

- compute;
- memory;
- encoding;
- scheduling;
- placement;
- communication.

## 30.2 Candidate/equivalence graph

It proposes a versioned graph carrying:

- semantic alternatives;
- candidate implementations;
- transformations;
- equivalence relationships;
- provenance;
- lower bounds.

## 30.3 Objective registry

Proposed algebraic objectives include:

- min-plus;
- max-plus;
- min-max;
- Boolean;
- Pareto;
- robust objectives.

## 30.4 One canonical execution plan

The target is one content-addressed plan binding:

- candidate choice;
- schedule;
- channel;
- placement;
- memory;
- transfer syntax;
- evidence;
- certificate.

## 30.5 Solver portfolio

The proposal calls for a solver portfolio with:

- exact methods where bounded;
- admissible heuristics;
- lower bounds;
- Pareto pruning;
- decomposition;
- incremental recomputation;
- explicit certificate levels.

## 30.6 Scaling and memory

Proposed work includes:

- incremental planning;
- canonical liveness schedule;
- tiered static allocation;
- dense-loop optimization;
- integrated HAM/Semantic Swap;
- provider boundaries.

## 30.7 ASN.1 role

ASN.1 encoding choice would become a typed GEM+ region rather than a separate selection subsystem. JER would remain a source/control/load-plane format, not a hot-path replacement for StreamPack.

## 30.8 C++ and system services

The proposal also sketches:

- Python-to-C++ migration;
- API/database/service decomposition;
- IPC;
- native measurement rigs;
- driver and kernel profiles;
- FPGA/SASOS implications.

All of this is currently proposal-level unless an individual component already exists elsewhere, such as HAM, static memory, ASN.1 selection, or the single-node C++ handoff.

---

# 31. Capability and maturity matrix

Legend:

- **Normative** — production law or frozen ABI.
- **Executable oracle** — real reference implementation.
- **Native** — real C/C++ implementation.
- **Tool-dependent** — executable when the resident compiler/runtime exists.
- **Modeled** — executable planning model without corresponding device execution.
- **Hosted research** — bounded experimental implementation.
- **Proposal** — documentation/design only.

| Subsystem | Status |
|---|---|
| Resource/claim/phase model | Executable oracle; represented in MLIR; native C subset |
| R1–R25 verification | Normative MLIR; broad Python twin; narrower C twin |
| K_BCIR min-plus planning | Executable oracle and MLIR pass family |
| RCSP/Pareto planning | Executable oracle and MLIR representation/passes |
| Composition/e-graph | Executable oracle |
| Provenance/replay | Executable oracle, MLIR, selected C twin |
| GEM scheduling | Executable oracle |
| StreamPack v1–v3 | Frozen ABI; Python and freestanding C |
| BCAB v1 | Frozen ABI; Python, C and C++ |
| MAP/ROP | Executable frontends |
| ETL/transducers | Executable oracle; binary C twin |
| C frontend | Executable Python and native C supported subsets |
| Portable C lowering | Native/tool-dependent and exercised |
| Textual LLVM lowering | Executable bounded subset |
| AOT/JIT/WASM | Tool-dependent and tested |
| JVM/CIL | Tool-dependent bounded subsets |
| Cross-target `llc` | Tool-dependent |
| ASN.1 schema/encodings | Broad executable oracle |
| ASN.1 R24/R25 | Normative MLIR law rail |
| ASN.1 native codecs | Native bounded format-specific twins |
| ASN.1 certified selection | Executable oracle plus native measurements |
| Static memory | Executable verified planner |
| HAM | Executable metadata planner; no physical transfer engine |
| Device manifest | Executable static schema |
| Telemetry | Executable Python; C frame/ring twins |
| Silicon probes | Real where host exposes signals; honest unavailable state otherwise |
| CPU channels | Executable |
| SYCL | Tool-dependent hosted execution |
| GPU/PTX | Real artifact generation; no local physical execution evidence |
| FPGA/PIM/NVMe | Modeled channels |
| Quantization/math/tensors | Executable oracle; many C paths |
| Autodiff/training | Executable oracle; bounded C lowering |
| Open-weight ingestion | Executable |
| Standalone BCIRQ8 inference | Native |
| Hosted PyTorch model lab | Hosted research and CI-gated |
| Byte/adaptive/sequence research | Hosted research |
| C++ single-node orchestration | Native and real |
| C++ dynamic/distributed orchestration | Explicit stubs |
| Resident kernel/driver/IPC | Not implemented |
| GEM+ unified planner | Proposal |
| Whole-system optimality certificate | Proposal |

---

# 32. What was accomplished in this reconstruction

I re-established the report from the repository and produced a new mechanical evidence set under the ignored validation directory:

```text
build/validation/system-report-repo-inventory.json
build/validation/system-report-pygount.txt
build/validation/system-report-deep-inventory.json
build/validation/system-report-python-api.json
build/validation/system-report-cli-help.txt
build/validation/system-report-vector-add-pipeline.txt
build/validation/system-report-vector-add.mlir
build/validation/system-report-vector-add.c
build/validation/system-report-embeddable-api.txt
```

I also:

- refreshed and froze Git provenance;
- confirmed the checkout is clean;
- inventoried all tracked top-level and major subsystem trees;
- enumerated current ODS operation families;
- enumerated all 37 MLIR passes;
- inventoried Python public structures and module descriptions;
- exercised all primary CLI help surfaces;
- traced the end-to-end architecture;
- generated a real vector-add plan, MLIR, and C output;
- compiled and self-checked the embeddable API’s native artifact;
- re-audited the complete ASN.1 architecture;
- separated production, native, modeled, hosted-research, educational, and proposal surfaces.

No product source file was modified.

---

# 33. Overall characterization

At the current commit, BCIR is best understood as six connected systems:

1. **A registry-first semantic IR and R1–R25 legality system.**
2. **A multidimensional realization optimizer with budgets, provenance, replay, and bounded learned guidance.**
3. **A GEM scheduling and frozen StreamPack execution architecture.**
4. **A multi-backend compiler/runtime stack spanning Python, MLIR/C++, portable C, LLVM, WASM, and bounded stack formats.**
5. **A broad ASN.1 compiler/selection ecosystem covering X.680–X.697 subsets, R24/R25 law, native twins, and certified encoding choice.**
6. **A large ML/model and heterogeneous-hardware research-to-runtime platform, with clear distinctions between native CPU execution, hosted experiments, modeled channels, and unimplemented physical drivers.**

The implementation is far beyond a documentation-only IR. It contains real compilers, optimizers, codecs, native runtimes, artifacts, model loaders, inference kernels, training references, telemetry, and extensive differential validation.

At the same time, the repository usually records the crucial maturity boundary directly:

- Python breadth is not automatically native breadth.
- Code generation is not device execution.
- A modeled hardware channel is not hardware evidence.
- A hosted research oracle is not a production kernel.
- A research proposal is not a landed architecture.
- A green remote CI result is not a fresh local hardware result.
- A content digest is not signer authority.
- JER is JSON text, not a binary hot-path format.
- The existing collection of optimizers is not yet the unified GEM+ whole-system solver proposed by the latest commit.

This completes the comprehensive capability and architecture reconstruction. The next phase can now focus specifically on **current shortcomings, architectural inconsistencies, missing native coverage, and a prioritized development program**, without having to rediscover the system first.
