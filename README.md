# Binary-Correspondence-Intermediate-Representation

BCIR is an IR-first compiler, runtime and model-development system built on one optimization
problem and one execution model: **Kolmogorov Binary Correspondence Intermediate Representation
optimization (K_BCIR)** and the **Graph Execution Model (GEM)**. A program is a registry of
*claims* over typed resources, ordered in phases and typed by execution lane. BCIR proves which
physical realizations of each claim are legal, prices only those with an integer cost algebra, and
hydrates the selected plan into a **StreamPack** that every rail executes and verifies the same way.

```
K_BCIR(G | H, Θ) = min_{π ∈ Legal(G,H)}  M(π, Θ)
                   subject to  R(π, Θ) ⪯ B(H, Θ)

M(π,Θ) = makespan of π's canonical schedule artifact — (max,+) over parallel, (min,+)⊗ over series
R(π,Θ) = additive resource vector (energy, traffic, …)
B(H,Θ) = live budgets (thermal cap, power cap, bandwidth)
```

Legality comes before cost: the verifier laws R1–R25 refuse a candidate before K_BCIR prices it.
Learned or measured data may rank and calibrate, but it never decides legality. Every artifact is
content-addressed and immutable within a generation. BCIR is a planning, verification, artifact
and runtime layer *above* resident toolchains: LLVM, GCC and vendor stacks remain its backends.

> **Two separate things live in this repository.** `bcir/`, `mlir/` and `runtime/` **are** the
> BCIR IR and its runtime. [`training/`](training) is a corpus for humans, agents and model
> training, and it is **not** part of the IR; neither depends on the other. See
> [`AGENTS.md`](AGENTS.md) and [`docs/BCIR_Repo_Structure.md`](docs/BCIR_Repo_Structure.md).

Counts (tests, MLIR operations, passes, runtime sources, verifier-law fixtures) live only in the
generated [`docs/STATUS.md`](docs/STATUS.md). This page says what exists and where; it does not
restate how many.

## Contents

1. [Languages and build makeup](#1-languages-and-build-makeup)
2. [Vision and architecture of the IR](#2-vision-and-architecture-of-the-ir)
3. [ASN.1 binary encodings and JSON](#3-asn1-binary-encodings-and-json)
4. [AI training — what is built](#4-ai-training--what-is-built)
5. [Roadmaps](#5-roadmaps)
6. [Documentation map](#documentation-map) · [License](#license)

---

## 1. Languages and build makeup

BCIR is realized as four cooperating rails. A change earns a rail through **differential parity,
not assertion** ([`docs/PARITY.md`](docs/PARITY.md)): the Python oracle states what the semantics
are, the MLIR rail states what is legal, and the C and C++ rails must produce the same bytes and
verdicts as the oracle.

| Rail | Language | Path | Role |
|---|---|---|---|
| Python oracle | Python ≥ 3.11, no runtime dependencies | [`bcir/`](bcir) | the executable conformance reference; every semantic decision is made here first |
| MLIR law | C++23 (C++26 where the compiler has it) and TableGen/ODS, on MLIR/LLVM 23 (22 still in CI) | [`mlir/`](mlir) | the IR law: the `bcir` dialect family, the compiled `bcir-opt`, verifier and optimizer passes, and a pure-data IRDL projection for stock `mlir-opt` |
| C runtime | C23, with C11 builds of the portable cores | [`runtime/c/`](runtime/c) | the production rail: freestanding codecs and executors, hosted compiler and model tools, driver adapters |
| C++ seam | C++17 | [`runtime/cpp/`](runtime/cpp) | a narrow orchestration layer above the C ABI, and the SIMD JSON structural index |

### The Python oracle (`bcir/`)

The oracle installs as the `bcir` package with no third-party dependencies. Optional extras are
opt-in and never on the verifier's path: `model-lab` (PyTorch, Safetensors, NumPy) for the hosted
model laboratory, `telemetry-kafka` for a Kafka sink, and `dev` for the pinned `ruff` and
`pre-commit`.

| Package | What it holds |
|---|---|
| [`bcir/model/`](bcir/model) | BCIR-0..2: the claim graph, lanes, opcodes, resources and phases |
| [`bcir/kbcir/`](bcir/kbcir) | BCIR-3: K_BCIR — the cost algebra, target profiles, the min-plus optimizer and the GEM+ solvers, bounds and certificates (§2) |
| [`bcir/gem/`](bcir/gem) | BCIR-4: GEM — the canonical schedule, StreamPack hydration, the deterministic and concurrent executors, the execution plan, the dispatch law, the exact solvers |
| [`bcir/verify/`](bcir/verify) | the runnable reference of the verifier laws R1–R25, alias facts, poison facts and delta re-verification |
| [`bcir/lower/`](bcir/lower) | BCIR-5: lowering to LLVM IR (AOT through Clang, JIT through `lli`), C, WebAssembly and MLIR, with the declared alias and poison facts |
| [`bcir/abi/`](bcir/abi) | the byte ABIs: StreamPack, the BCAB artifact bundle, ExecutionPlanV1, the plan signature (Ed25519), the control-plane record, the shard manifest, the telemetry envelope |
| [`bcir/asn1/`](bcir/asn1), [`bcir/frontends/asn1/`](bcir/frontends/asn1) | the ASN.1 portfolio and its schema compiler (§3) |
| [`bcir/frontends/`](bcir/frontends) | the C front end (`cfront`), the ROP and MAP front ends, and model ingest and payload-free model assessment |
| [`bcir/etl/`](bcir/etl) | M5 event transduction: events, the FSM, the parser and the binary decoder |
| [`bcir/make/`](bcir/make) | BCIR Make: the `BCIRfile` grammar, its DAG model and laws, and the task runner |
| [`bcir/hosted/`](bcir/hosted) | opt-in hosted references: the model laboratory, staged training and the CP-SAT adapter (§4) |
| `bcir/telemetry*.py`, [`bcir/channels.py`](bcir/channels.py) | the telemetry schema, frame and export; the hardware channels, extended by plugin manifests in [`channels/`](channels) |
| [`bcir/tests/`](bcir/tests) | the conformance suite, run in a bounded `quick` tier and a `thorough` tier |

Installed commands: `bcir` (plan, schedule, lower and run an example), `bcir-pack` (StreamPack),
`bcir-bundle` (BCAB), `bcir-registry`, `bcir-model-assess`, `bcir-asn1c` (the ASN.1 compiler),
`bcir-make`, `bcir-ham-plan` and `bcir-performance-audit` (`bcir-tmsao-audit` is its older alias,
and says that it issues no TMSAO certificate).

### The C runtime (`runtime/c/`)

Every C source belongs to one library, tool, harness or fuzzer named in
[`runtime/manifest.json`](runtime/manifest.json), the one source list that CMake, BCIR Make and
every gate read. The code follows three enforced memory classes
([`docs/languages/C_MEMORY_DISCIPLINE.md`](docs/languages/C_MEMORY_DISCIPLINE.md)): freestanding
code that never allocates, hosted tools with injected allocators (tested by failing each allocation
in turn), and driver adapters that cross boundaries with handles and offsets, never raw pointers.
Every trust-boundary decoder has a libFuzzer target run under ASan and UBSan.

| Library | What it does |
|---|---|
| `bcir_base` | the runtime core, SHA-256, Ed25519 and plan signatures |
| `bcir_streampack` | the plan reader, hydration, the GEM executor, the StreamPack encoder and decoder, binary records, the logistic-training stage kernels, telemetry frames and provenance |
| `bcir_plane` | the control-plane records, the live SPSC ring, the telemetry envelope, the data-plane hand-off, the shard manifest and the native K_BCIR planner |
| `bcir_artifact` | the allocation-free BCAB bundle reader and selector |
| `bcir_asn1` | the ASN.1 twins: X.690, PER (with the bit-oriented writer and the plan-driven decoder), OER, XER, the bounded JER reader, and the StreamPack ASN.1 module |
| `bcir_cfront` | the C front-end twin: preprocessor, parser and lowering, verifier and diagnostics |
| `bcir_quarantine` | the bounds-quarantine runtime that the C front end's runtime-checked array accesses call |
| `bcir_models` | BCIRQ8 inference: the Q8 model, Q4 kernel, AI kernels, the Llama decoder and the decoder as a GEM program |
| `bcir_decoder_training` | the native tensor core and the decoder trainer (forward, backward, AdamW) |
| `bcir_channel`, `bcir_driver` | the hardware-channel table and the RuntimeChannel v1 loopback driver adapter |

Tools: `bcir-cc` (the C compiler driver over the front-end twin), `bcir-llama` (standalone BCIRQ8
inference), `bcir-qualify` (the qualification rails, §4), `bcir-make` (the C twin of BCIR Make),
and the cost-calibration, ASN.1 and AI micro-benchmarks.

### The C++ seam (`runtime/cpp/`)

`bcir_seam` orchestrates above the C ABI. Artifacts cross it as borrowed views admitted by the live
control plane. The single-node and dynamic-graph backends are real, as are the distributed partition
and the manifest of shards; only cross-node dispatch is an explicit stub. `bcir_jer_index` is the J5
SIMD structural index for JSON: AVX2 or SSE2 chosen at run time on x86-64, NEON on AArch64, scalar
elsewhere. SYCL is a backend channel and differential oracle, never on the legality path: a SAXPY
kernel the oracle emits is compiled and run against BCIR's reference by its own gate.

### The build

- **CMake** ([`CMakeLists.txt`](CMakeLists.txt), [`cmake/`](cmake),
  [`CMakePresets.json`](CMakePresets.json)) builds everything from the manifest. Configure presets:
  `default`, `gcc`, `clang`, `clang23`, `asan`, `ubsan`, `tsan`, `fuzzer` and `mlir`. Test presets:
  `build` (manifest reconciliation and build parity), `quick` (`build` plus the oracle's quick
  tier), `c`, `fuzz`, `mlir`, `section` and `all`. The tree installs and exports
  `find_package(BCIR)` with versioned headers.
- **BCIR Make** (`bcir-make`, oracle in [`bcir/make/`](bcir/make) with a C twin) runs the
  repository's gates as a `BCIRfile` DAG with content-addressed reuse, a two-worker scheduler, an
  artifact cache and pinned tools. Its runs agree with CMake's, section by section.
- **Gates** ([`tools/`](tools)): `tools/c/check_runtime.sh` and its per-section scripts (strict
  warnings, sanitizers, byte-identity), the MLIR and IRDL scripts, the docs-governance checks, the
  security rails and the performance harnesses.
- **Toolchains.** CI's default is Clang 18 and GCC 13. Every change is also judged on the newest
  LLVM/Clang/MLIR 23 and Node 24 (`tools/local/check_latest.sh`). CI runs x86-64, native AArch64,
  Windows, the MLIR rail on LLVM 22 and 23, and a qualification job on both architectures
  ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

### Quickstart

```bash
# The oracle (no third-party dependencies): plan, schedule, lower and run an example program.
python -m bcir.run vector_add --target x86_avx512 --theta cool       # K_BCIR picks vec16
python -m bcir.run vector_add --target nvidia_ptx --explain          # why each candidate won or lost
python -m bcir.run vector_add --target x86_avx512 --schedule --jit   # canonical schedule + LLVM JIT (lli)
python -m bcir.run vector_add --target x86_avx512 --wasm             # WebAssembly, run through Node
python -m bcir.run vector_add --budget thermal=700 --overlap         # RCSP budgets and the makespan M(π,Θ)
python -m bcir.run vector_add --egraph                               # the building-blocks rewrite engine
python -m bcir.run multi_histogram --target nvidia_ptx --emit-mlir   # emit the plan for the MLIR law rail
python -m bcir.kbcir.differential -n 5000                            # generated Python <-> MLIR parity

# Tests and gates (two workers at most on a local host).
python -m bcir.tests.run_all --tier quick -j 2
bash tools/c/check_runtime.sh
cmake --preset default && cmake --build --preset default && ctest --preset quick

# The compiled MLIR dialect (needs an LLVM/MLIR 23 or 22 toolset).
bash tools/wsl/build_mlir.sh && bash tools/wsl/check_ods_examples.sh
bash tools/irdl/check_corpus.sh
```

[`CONTRIBUTING.md`](CONTRIBUTING.md) lists the full pre-PR gate.

---

## 2. Vision and architecture of the IR

### The vision

BCIR treats compilation as a search for the cheapest **legal** correspondence between a
computation and a machine, under the machine's live state. The search runs over representation
and machine-state space as a shortest-path problem in a tropical semiring; it is not a one-shot
source-to-binary pipeline. Five rules hold everywhere:

1. **Legality precedes optimization.** The R-laws refuse; only then does K_BCIR price.
2. **One semantic truth, many realizations.** The oracle defines it; MLIR, C and C++ are held to
   it by generated differentials, bit-exact scores and byte-identical artifacts.
3. **Frozen ABIs stay frozen.** StreamPack v1, BCAB v1 and the telemetry frame grow only by
   appended versions or additive projections.
4. **DER out, BER in.** BCIR never emits an encoding whose octets a peer may choose.
5. **Unsupported work fails honestly.** A skip is never reported as a pass, and a modeled number
   is never presented as a measured one.

### The levels

| Level | Name | Question |
|---|---|---|
| BCIR-0 | Semantic claim graph | What computation or state transformation is intended? |
| BCIR-1 | Shaped data graph | Which tensors, columns, buffers, sparse maps, records and layouts exist? |
| BCIR-2 | Registry and placement candidates | Where can resources live, and which domain constraints apply? |
| BCIR-3 | K_BCIR correspondence plan | Which realization path π is selected under H and Θ? |
| BCIR-4 | GEM stream IR | Which lane schedule, StreamPack, fences and prefetch contracts execute? |
| BCIR-5 | Target lowering | LLVM, vector, GPU, SPIR-V, PTX, WASM, or a future BCIR-native binary |

([`docs/BCIR_LANGREF.md` §1–§2](docs/BCIR_LANGREF.md#1-the-multi-level-ir))

### The core engine: K_BCIR → GEM → StreamPack

- **K_BCIR** prices each legal candidate on a 12-axis integer/Q8 cost vector under the pressure Θ
  and RCSP resource budgets B, over target profiles and pluggable hardware channels (CPU, GPU,
  FPGA, memory and storage). The search is min-plus. A temperature dial gives a soft plan
  distribution whose T = 0 limit is the tropical optimum. Cost tables are measured and frozen
  (Bayesian posterior with conformal intervals) and applied per target.
- **GEM** reads one canonical schedule artifact, shared by the price, both executors and the law
  rail's passes, and hydrates the plan into a StreamPack. The oracle's deterministic and concurrent
  executors and the native C executor (`bcir_exec.c`) are held to the same results and verdicts.
  Since G11 the plan itself is bytes (`ExecutionPlanV1`), bound to its pack. Since G21 it can be
  signed: a detached Ed25519 statement binds plan, scope, module, pack and certificate.
- **StreamPack** is the execution artifact. Version 1 is frozen; v2–v4 append to it (v4 adds the
  per-resource generation vector). Its DER projection is the `BCIR-StreamPack` ASN.1 module, and
  artifacts travel in **BCAB** bundles. Contracts:
  [`docs/kernel/BCIR_STREAMPACK_ABI.md`](docs/kernel/BCIR_STREAMPACK_ABI.md),
  [`docs/kernel/BCIR_EXECUTION_PLAN_ABI.md`](docs/kernel/BCIR_EXECUTION_PLAN_ABI.md),
  [`docs/kernel/BCIR_ARTIFACT_BUNDLE_ABI.md`](docs/kernel/BCIR_ARTIFACT_BUNDLE_ABI.md).

### Algorithms and optimizations around the engine

These are landed. The GEM+ program
([`docs/research/BCIR_GEMPLUS_ROADMAP.md`](docs/research/BCIR_GEMPLUS_ROADMAP.md)) holds them to a
frozen baseline harness. Its rows are `exact`, which gate; `ratio`, which gate with a wide band;
or `wall`, which is indicative only. Every certificate names its class on the **TMSAO ladder**:
TMSAO-1 is an exact optimum, TMSAO-2 a bounded gap, TMSAO-3 the best measured, and TMSAO-4 a
heuristic with no claim.

- **Search and selection.** Min-plus planning with RCSP budgets. The building-blocks e-graph runs
  the rewrite laws (LangRef §11). A propose-and-verify accelerator reaches the same optimum with
  fewer expansions. A learned mixture-of-experts gate routes inside the two-truth quarantine.
  Regret and replay accounting. A provenance manifest whose replay reproduces the plan.
- **Scheduling.** Wave, token and EFT schedulers over the phase DAG, and a hazard DAG of data
  hazards and fenced barriers. Incremental EFT re-placement (G2). Bounded exact branch-and-bound
  schedules with a named lower-bound stack, issuing the first TMSAO-1/TMSAO-2 certificates (G4).
  The proved optimum is the plan that runs (G19). A dispatch law with work-unit budgets and
  resumable solvers (G12). The depth of optimization is priced by its value, and straight-line
  modules are delegated to LLVM (G27).
- **Memory and movement.**
  - Allocation: schedule-aware liveness with a bounded exact layout (G5); a dependency-free exact
    CSP and the joint schedule × memory optimum, with an optional CP-SAT adapter (G28).
  - Movement and aliasing: data movement as a first-class transformation (G8); declared alias
    facts exported to LLVM as `noalias`, scopes and TBAA (G9); escape analysis and indirect-call
    narrowing (G10).
  - Bounds: red-blue pebble and hierarchical-roofline bounds, and alignment-aware memory bounds
    with an incumbent layout portfolio (G25/G26).
- **Regions and objectives.** Typed regions and a semiring registry admitted only with their laws
  proved (G6). Expected cost over Markov loops (G22). SDF/CSDF dataflow and max-plus cycle time by
  Karp's algorithm (G23/G24). Polyhedral nests to depth two (G30). The matmul tiling plan exported
  as an MLIR Transform script (G31).
- **Identity, proof and verification.** ExecutionScopeV1 and the TMSAO ladder (G0). The module
  digest computed once per revision (G3). Incremental re-verification and delta StreamPack (G18).
  No-wrap and in-bounds facts proved and exported to LLVM as poison flags under R12 (G29).
- **Runtime planes.** The control-plane record ABI (G14), the live shared ring (G15), the
  data-plane hand-off (G16), and a compact native planner that matches the oracle byte for byte
  (G17).
- **Measurement.** The declared workload `W` and an append-only measured corpus with replay
  certificates (G13). The repaired native measurement rig (G7). `tools/perf/ab_audit.py`, which
  closes every slice with a before/after audit.
- **ML primitives as claims.** Exact-width low-bit lanes and group-32 quantization, deterministic
  matmul schedule search, closed-set reverse-mode AD, and library wrappers (FFT, LAPACK, GSL),
  each lowered and held to the oracle.

### Legality: the verifier laws R1–R25

Every law has a runnable oracle in [`bcir/verify/`](bcir/verify), a rule on the MLIR rail, and a
negative fixture
([LangRef §10](docs/BCIR_LANGREF.md#10-verifier-laws-r1r25)).

| Laws | What they hold |
|---|---|
| R1–R2 | registry uniqueness and resolution |
| R3–R7 | domain, phase-DAG, hazard, lane and bounds legality |
| R8–R12 | cost completeness, plan legality, stream provenance, generation validity, lowering legality |
| R13 | policy provenance: the module digest, recomputed at every trust boundary |
| R14–R18 | CIM/PIM dispatch, the DVFS clock, allocator placement, the accuracy contract, a resolved and acyclic call graph |
| R19–R20 | synchronous timing and clock-domain crossing |
| R21 | pointer lifetime: use-after-free and double-free |
| R22–R23 | shape and dtype agreement at GEM seams |
| R24–R25 | ASN.1 encoding-rule legality and X.692 ECN encoding-definition legality |

The LangRef also fixes the module, memory, claim, phase, lane, cost and stream laws with the
naked-pointer policy (§3–§9), the rewrite laws (§11), the lowering contracts (§12), learning
placement (§13) and the two-truth separation (§14). New laws ship *vacuous by default*: the whole
existing corpus must verify byte-identically with the new law in place.

---

## 3. ASN.1 binary encodings and JSON

ASN.1 is BCIR's **external contract language**. A StreamPack, a BCAB bundle, an execution plan or
a telemetry frame crosses a boundary where the peer may not be BCIR. X.680's abstract syntax plus a
named transfer syntax states that boundary once, and every rail checks it
([LangRef §17](docs/BCIR_LANGREF.md#17-asn1-in-bcir),
[`docs/BCIR_ASN1_X690_ABI.md`](docs/BCIR_ASN1_X690_ABI.md)).

| Rail | Artifact | What it answers |
|---|---|---|
| Python oracle | [`bcir/asn1/`](bcir/asn1), [`bcir/frontends/asn1/`](bcir/frontends/asn1) | what the octets are, executably |
| MLIR law | `bcir.asn1.*` (R24), `bcir.ecn.*` (R25) | what a schema may legally say, statically |
| C twins | `runtime/c/bcir_{asn1,per,per_plan,oer,xer,jer}.c` | what a freestanding reader can read |

**Transfer syntaxes.** Canonicality is the selection criterion: a rule with no canonical variant
may be decoded but is never chosen for emission, because a selected encoding becomes a digested
artifact.

| Recommendation | Rules | Canonical member |
|---|---|---|
| X.690 | BER, CER, DER | DER (CER input is refused by profile) |
| X.691 | PER aligned / unaligned × basic / canonical | canonical PER |
| X.696 | OER, COER | COER |
| X.693 | XER, CXER | CXER |
| X.697 | JER | BCIR's canonical JER |

**The front end.** `bcir-asn1c` compiles X.680 modules into BCIR's type model and prints them back
losslessly. It covers:

- X.681–X.683: information objects, table constraints and hygienic parameterization;
- constraints, including PER-visibility;
- recursive types and tagged assignments;
- open types.

Extensibility follows one **relay posture** on every rule (LangRef §17.3):

- an extensible constraint keeps its additional value set but never refuses a value on encode;
- an extension addition, mandatory or not, may be absent, because the value is one of an earlier
  version;
- a version bracket is one addition, flattened on every rule but PER and OER.

The octets of these postures are checked against two independent codecs, asn1tools and pycrate,
not just round-tripped.

**X.692 ECN** is built in all three parts: the model, the user-defined encodings, and the defined
syntax. Its refusal list is empty, and R25 holds ECN definitions on the law rail.

**Selection and generated code.**
- *Selection.* K_BCIR selects an encoding rule per value, target and budget. Certified intervals
  come from measured native tables; two targets are calibrated, and both are core clusters of one
  phone. RCSP octet budgets bound the choice.
- *Generated codecs.* [`bcir/asn1/cgen.py`](bcir/asn1/cgen.py) compiles a schema into C codecs for
  COER and UPER.
- *Evidence.* The ITS security-envelope study
  ([`docs/research/BCIR_ITS_ASN1_BINARY_STUDY.md`](docs/research/BCIR_ITS_ASN1_BINARY_STUDY.md))
  reconstructs ETSI TS 103 097 V1.1.1's binary rules and measures BCIR's ASN.1 rules against them
  for size, time and memory.

### JSON

- **JER (X.697)** is landed as the J1–J6 ladder
  ([`docs/BCIR_ASN1_JSON_ROADMAP.md`](docs/BCIR_ASN1_JSON_ROADMAP.md)):
  - J1, a bounded reader and its oracle;
  - J2, an immutable compiled schema plan;
  - J3, a freestanding C twin;
  - J4, an MLIR rule family;
  - J5, a hosted SIMD structural index;
  - J6, certified K_BCIR selection with real target calibration.

  Encoding instructions (`ARRAY`, `UNWRAPPED`, `NAME`, `BASE64`, …) and semantic size bounds are
  built. JER is a build, control, configuration and load-plane format. A latency-sensitive path
  consumes verified claims, StreamPack, BCAB or native objects, never JSON text.
- **JSON as a program representation**
  ([`docs/BCIR_JSON_PROGRAM_REPRESENTATION.md`](docs/BCIR_JSON_PROGRAM_REPRESENTATION.md)) is
  landed as P1–P6:
  - P1, a graph form with integer-index edges and cycle-safe content addressing;
  - P2, the `bcir.asn1.*` dialect projected through it;
  - P3, phases, claims, resources, timing and lifetime carried structurally;
  - P4, cost-graph execution over a caller's semiring, with Karp's minimum mean cycle;
  - P5, a lossless sparse text view whose presentation stays outside the canonical form;
  - P6, staged self-modification, where a proposal reaches an artifact only through the trusted
    loader.

---

## 4. AI training — what is built

Everything in this section exists today and is gated. Hosted pieces are opt-in (`model-lab`
extra) and stay out of the verifier, the dependency-free oracle and the freestanding C runtime.

### The model training harness

- **Hosted model laboratory** ([`bcir/hosted/models/`](bcir/hosted/models)):
  - *Model.* An independent Llama-family decoder driven by `DecoderSpec`: RMSNorm, RoPE, GQA,
    causal attention, SwiGLU, tied or untied heads.
  - *Training.* `train_hosted` uses AdamW only, with explicit device and precision.
  - *Checkpoints.* Pickle-free generations, validated before any state changes, and published
    atomically.
  - *Telemetry.* Informative only, never steering.
  - *Gate.* An always-on gate closes the loop: random weights → training → safe checkpoint →
    strict ingest → BCIRQ8 → standalone C, with matching IDs and logits.
- **Staged training** ([`bcir/hosted/training/`](bcir/hosted/training)):
  - *Data.* Corpus cleaning with license admission, exact deduplication and content-addressed
    splits.
  - *Tokenizer.* A dependency-free byte-fallback BPE tokenizer.
  - *Objectives.* SFT; Bradley–Terry reward training; DPO against a frozen reference; PPO with
    GAE; verified-reasoning SFT; relational embedding distillation.
  - *Models.* Bounded MLP, GRU and encoder confirmation models.
  - *Ledger.* An append-only, content-addressed pipeline ledger.
  - *Providers.* Teacher and remote-compute provider interfaces, with offline implementations only.
- **Payload-free model assessment** (`bcir-model-assess`) reads a checkpoint's headers, never its
  weights. It produces a tensor inventory, a hardware envelope and a workload spec, then a capacity
  and placement report and a verified StreamPack.
- **Architecture laboratories.**
  - An adaptive transformer, a byte-native model and a sequence-interface / constructive-growth
    lab.
  - A hardware-policy RL gate: telemetry → PUCT → a verified plan.
  - A pinned real-model gate that downloads checksum-pinned files.

### The C implementation

- **Native decoder trainer**
  ([`docs/machine-learning/BCIR_NATIVE_DECODER_TRAINING.md`](docs/machine-learning/BCIR_NATIVE_DECODER_TRAINING.md)):
  - *Scope.* A complete FP32 Llama/SwiGLU forward, backward and AdamW loop in C11, also built as
    C23, behind a stdlib-only Python binding (`NativeDecoder`).
  - *Kernels.* A blocked GEMM that is bit-identical to the reference loop, and causal attention
    with GQA backward.
  - *Checkpoints.* Caller-owned arenas with no allocation in C, and SHA-256 checkpoints.
  - *Parity.* Checked against an independent PyTorch model, gradient by gradient.
  - *As a GEM+ program.* One training step is also stated as a GEM+ program, and frozen-harness rows
    hold the C rail to it.
- **BCIRQ8 v1** ([LangRef §16](docs/BCIR_LANGREF.md#16-bcirq8-v1-decoder-artifact-contract)) is the
  decoder artifact contract. `bcir-llama` runs standalone BCIRQ8 inference in C (`bcir_q8_model`,
  the Q4 kernel, the AI kernels), and a pinned TinyLlama parity gate holds it to the oracle.
- **Qualification on a concrete target** ([`docs/BCIR_QUALIFICATION.md`](docs/BCIR_QUALIFICATION.md)).
  `build_lab_model.py` builds a BCIR-native decoder from the repository's own docs: corpus, BPE
  tokenizer, native C training and BCIRQ8 export. `run_qualification.py` then checks:
  - the decoder as a K_BCIR-planned BCIR program with R1–R25 clean;
  - the plan and pack round-tripping DER, OER and JER;
  - GEM's C execution matching the monolithic runner bit for bit;
  - counted memory traffic matching planned traffic;
  - simulated cache traffic per rail.

  It runs as a CI job on x86-64 and native AArch64.
- **Lowering** ([`bcir/lower/`](bcir/lower), LangRef §12): a supported elementwise claim lowers to
  LLVM IR for AOT compilation with Clang and JIT through `lli`, carrying the declared alias and
  poison facts; arbitrary graphs are refused rather than truncated. The same package emits C: a
  reverse-mode AD forward/backward kernel with an SGD step, and a baked-weights end-to-end
  inference kernel, each checked against the oracle.

### The training corpus (`training/`)

The corpus serves three audiences: people learning, agents doing a task, and models being trained.
It has its own normative reference, [`training/TRAINING_LANGREF.md`](training/TRAINING_LANGREF.md),
whose tools are its conformance oracle. Its tiers are hand-written prose, derived retrieval chunks
and embeddings, and derived distillation records.

`training/tools/export_training_examples.py` emits BCIR `SFTExample`s and preference pairs that
the verifier decides. The retrieval rail became a database through the landed S1–S19 ladder: a
query planner, generations, zone maps, constraints and transactions
([`training/DATABASE_ROADMAP.md`](training/DATABASE_ROADMAP.md)).

The open subject is [`training/llvm/`](training/llvm), an agent context pack for LLVM IR grounded
in the LLVM LangRef. Its chapters run from foundations through the MLIR bridge, the backend, the
JIT and the new pass manager. Every standalone example assembles with a modern `llvm-as`, and a
lit-gated aggregate checks them. The other subjects are scoped and planned.

---

## 5. Roadmaps

Roadmaps own the future; [`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md) owns the past.
[`docs/BCIR_MASTER_ROADMAP.md`](docs/BCIR_MASTER_ROADMAP.md) orders the programs below.

### In development

| Program | Roadmap | Where it stands |
|---|---|---|
| GEM+ / TMSAO | [`docs/research/BCIR_GEMPLUS_ROADMAP.md`](docs/research/BCIR_GEMPLUS_ROADMAP.md) | G0–G18 and the 2026-10-06 completion ladder are landed. Open: lay the native training step out by lifetime (NDT-MEM), the rest of the C ASN.1 performance work, CXX3. TMSAO-3 (S6) needs two physical targets with PMU counters, and is an explicit hardware skip until then. |
| ML / AI integration | [`docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md`](docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md) | Phases A–F. Open: packed INT2–INT6 compute, activation quantization, measured hardware schedule evidence, general higher-order and control-flow AD, open-weight model ingestion (GLM, Gemma, Qwen) and serving. Next model for qualification: BCIR-TinyStories-32M. |
| ASN.1 and JSON | [`docs/BCIR_ASN1_BUILDOUT_ROADMAP.md`](docs/BCIR_ASN1_BUILDOUT_ROADMAP.md), [`docs/BCIR_ASN1_JSON_ROADMAP.md`](docs/BCIR_ASN1_JSON_ROADMAP.md) | The portfolio is complete on its documented subsets; J7 (hardware counters) is blocked on device access, not design. |
| Build | [`docs/BCIR_BUILD_ROADMAP.md`](docs/BCIR_BUILD_ROADMAP.md) | BUILD-1–8 are landed: CMake/CTest over the manifest, section scripts, install and export, the dependency index, the MLIR rail under CMake, BCIR Make and its C twin. |
| Training corpus | [`training/ROADMAP.md`](training/ROADMAP.md) | Phase 1 completes `llvm/`; Phases 2–8 open the hardware, systems, low-level, formats, languages, data and backends subjects. |

### Future: databases, drivers and kernels

- **Databases.**
  - The retrieval database (S1–S19 landed) states when it grows next, such as ANN indexing past
    about 50,000 rows, wire protocols and authentication, as reopening conditions in
    [`training/DATABASE_ROADMAP.md`](training/DATABASE_ROADMAP.md).
  - A repository-owned **Unicode 17.0 database** is planned in
    [`training/UNICODE_ROADMAP.md`](training/UNICODE_ROADMAP.md): pinned UCD sources;
    normalization, collation, security and IDNA engines gated by the standard's own conformance
    suites. No slice has landed yet.
  - A **small language model** that consults that database through a hybrid SQL/vector lookup is
    planned in [`training/SLM_ROADMAP.md`](training/SLM_ROADMAP.md). No slice has landed yet.
- **Drivers**
  ([`docs/kernel/BCIR_DRIVER_KERNEL_ROADMAP.md`](docs/kernel/BCIR_DRIVER_KERNEL_ROADMAP.md)).
  Every driver climbs the same D0–D7 ladder, from research contract to hardware promotion, and
  evidence comes first:
  - order: the RuntimeChannel loopback (landed), then the 16550/16750 UART, virtio-console,
    virtio-blk, virtio-net, emulated PCIe (e1000, NVMe), and finally physical and accelerator
    families;
  - the BCIR UAPI v1 freezes only after UART and virtio-blk prove both device classes;
  - the UART has a detailed blueprint
    ([`docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md`](docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md)),
    and AMD accelerators have their own track
    ([`docs/kernel/BCIR_AMD_AI_DRIVER_ROADMAP.md`](docs/kernel/BCIR_AMD_AI_DRIVER_ROADMAP.md)).
- **Kernels.**
  - BCIR-Linux is an experimental fork that serves as a compatibility oracle.
  - A native-kernel track runs in parallel: UEFI entry, then physical and virtual memory, ACPI and
    Device Tree, timers and interrupts, PCIe and MSI-X, DMA and IOMMU, then device adapters in
    evidence order.
  - On top come slim native IPC and a JIT microkernel.
  - The verified memory-fabric control plane (GDS, P2PDMA, CXL, semantic storage) is in
    [`docs/kernel/BCIR_HAM_MEMORY_FABRIC.md`](docs/kernel/BCIR_HAM_MEMORY_FABRIC.md).
  - The typed x86 long-mode entry, descriptor and segment operations and an interrupt-frame
    trampoline are already built and checked by assembling them.
- **Hardware access.** Live PMU counters, TMSAO-3 and driver bring-up need hardware this
  repository's hosts do not expose
  ([`docs/BCIR_TARGET_ACCESS.md`](docs/BCIR_TARGET_ACCESS.md)). Those gates are recorded as
  explicit skips, never simulated.

---

## Documentation map

- [`docs/BCIR_LANGREF.md`](docs/BCIR_LANGREF.md) — the normative language reference: levels, the
  central equation, laws R1–R25, rewrite and lowering contracts, BCIRQ8 v1, ASN.1 and ECN.
- [`docs/PARITY.md`](docs/PARITY.md) — the Python ↔ MLIR law contract and the Python ↔ C
  artifact and runtime parity ledger.
- [`docs/STATUS.md`](docs/STATUS.md) — the generated inventory: tests, ODS operations, passes,
  runtime sources and verifier-law fixture tags.
- [`docs/REPO_CURRENT_STATE_AUDIT.md`](docs/REPO_CURRENT_STATE_AUDIT.md) — what is implemented
  today, rail by rail.
- [`docs/ONBOARDING_DEEP_DIVE.md`](docs/ONBOARDING_DEEP_DIVE.md) — a guided reading path through
  the system.
- [`docs/PERFORMANCE_AUDIT.md`](docs/PERFORMANCE_AUDIT.md) — the TMSAO methodology, measured
  bottlenecks and hardware-gated limits.
- [`docs/BCIR_QUALIFICATION.md`](docs/BCIR_QUALIFICATION.md) — the end-to-end qualification of a
  BCIR-native model.
- [`docs/security/laws.md`](docs/security/laws.md) — the gate-authoring laws behind every
  assurance rail.
- [`docs/languages/CFRONT_GUIDE.md`](docs/languages/CFRONT_GUIDE.md) — the `bcir-cfront` C front
  end: CLI, diagnostics, the target ABI matrix and the supported subset.
- [`docs/machine-learning/THIRD_PARTY_MODELS.md`](docs/machine-learning/THIRD_PARTY_MODELS.md) —
  pinned model provenance and licenses.
- [`docs/DEVELOPMENT_HISTORY.md`](docs/DEVELOPMENT_HISTORY.md) — how it was built: the method, the
  PR-era arc and the changelog.

The Python package is the **executable conformance oracle**; the MLIR dialect is the **law** it
must agree with. LLVM and Clang are backends, not the conceptual center.

## License

This project is licensed under the **BCIR Non-Commercial License, Version 1.0**
(`LicenseRef-BCIR-NC-1.0`) — see [`LICENSE`](LICENSE) for the full terms.

The code is open and free for **open-source use, development, modification,
free redistribution, and private use**. **Commercial use, commercial
distribution, and patent use are not permitted** without explicit prior
written permission from Mecca-Research; contact the maintainers through this
repository to request a commercial license. No trademark rights are granted,
and the software is provided without warranty or liability.

In the standard sense of the terms this is a **source-available, non-commercial**
license, not an OSI-approved open-source license: "open-source use" above is the
license's own permitted purpose (using the Work in and with open-source projects),
not a statement that the license itself is open source.
