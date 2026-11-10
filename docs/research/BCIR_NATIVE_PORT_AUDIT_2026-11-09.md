# What the Python oracle still holds that belongs in C or C++, and where K_BCIR's solver should grow (2026-11-09)

> Dated, non-normative audit of `main` at 779d500 (after #825, with #826's roadmaps).
>
> It answers two questions the owner asked:
> 1. Which components of the Python oracle should already have a C or C++ realization in the
>    current build and do not?
> 2. Should the K_BCIR solver grow in C or in C++, and should more components be built in C++
>    at all, given that the C++ rail has stayed thin throughout the build?
>
> The placement rule it applies is the migration proposal's
> ([`BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md`](BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md) §8.1)
> and the C ↔ C++ boundary's ([`../languages/CPP_HANDOFF_BOUNDARY.md`](../languages/CPP_HANDOFF_BOUNDARY.md)).
> Every "has a twin" and "has none" below was read from the source, not from a summary.

## 0. The answers

**1. Fifty-four oracle modules (about 35,000 of the oracle's 162,000 non-test lines) should gain
a native realization.** None of them breaks a native artifact that runs today; Tier 1 limits
what those artifacts can do (native inference, for one, cannot read text). They fall into three
tiers:
- **Tier 1, needed by a native path the current build already has or has committed to**
  (16 modules): the MC1 StreamPack and BCAB disassembler, the two tokenizers, the event
  transducer, and ECN's executable codecs.
- **Tier 2, the C++ toolchain** (31 modules): the GEM+ solver portfolio, the incremental
  engine, the ASN.1 host compiler, the native kernel lowering and the memory fabric. These
  remove Python from the offline toolchain; the runtime does not wait on them.
- **Tier 3, gated on a consumer that does not exist yet** (7 modules): paged KV and serving,
  the device manifest, DMA and event phases, telemetry export and budgeted selection.

Everything else the inventory lists as Python-only stays in Python, by the placement rule, or
already has a realization the inventory does not count (§2).

**2. Grow the K_BCIR solver in both languages, split by where it runs, not by preference.**
- **The online core stays in C.** `bcir_kplan.c` is already the native K_BCIR planner. It is
  freestanding, heap-free and libc-free, takes caller-lent scratch, does exact 128-bit
  arithmetic and reproduces the Python plan byte for byte. That is what a driver, a
  microkernel or a live re-plan at a quiescent boundary needs, and C++ would add nothing
  there.
- **The offline solver portfolio goes to C++23.** That means the exact schedulers, the CSP/ILP
  rail, the joint schedule × memory optimum, regions and dataflow, exact static memory, data
  movement and the incremental engine. They are search over growing graphs. In C each one
  needs hand-written hash tables, heaps, dynamic arrays and graph stores, which is where the
  memory-safety defects of this repository have lived.
- **C++ does not cost the memory control.** Written as arena-backed C++23 (injected
  `std::pmr` resources, `std::expected` instead of exceptions, no global constructors, a C ABI
  façade), it keeps the same control C gives and adds RAII teardown and typed ownership.
- **This does not break the boundary's rule that C++ never computes a verdict.** The solver
  proposes plans; the C and MLIR verifiers, which share no code with it, decide them.

So: **keep the Python/C route for everything that runs on the target, and start building the
hosted toolchain in C++.** Do not move existing C to C++. §4 gives the reasoning and the rules.

## 1. Method and the baseline

| Tree | What | Lines |
|---|---|---|
| `bcir/` (tests excluded) | the oracle: 313 modules | 161,986 |
| `runtime/c/` | the C rail: libraries and tools / everything with tests, fuzzers and C-front fixtures | 55,060 / 81,590 |
| `runtime/cpp/` | the C++17 seam: hand-off (G16), the orchestrator, the JER SIMD and structural index (J5), with tests | 3,572 |
| `mlir/lib`, `mlir/tools` | the C++23 law rail: the dialect, `bcir-opt` and its passes | 13,181 |

The method:
1. Every oracle module was cross-referenced against every C, C++ and MLIR source that names
   it, together with how many oracle modules and tools import it and its docstring
   (a script over `git ls-files`).
2. The generated oracle inventory ([`../BCIR_ORACLE_INVENTORY.md`](../BCIR_ORACLE_INVENTORY.md),
   CXX0) supplied the C twins.
3. Each candidate was then checked in the code: its consumers, what the nearest native unit
   implements, and what `PARITY.md` declares.

## 2. What is already native, including what the inventory misses

The inventory counts **45 modules with a C twin**: a C unit counts only when its leading comment
names the module. Read from the sources, the native coverage is wider:

- **The MLIR rail holds C++23 realizations the inventory never reads.** It holds the 12-axis
  cost model (`-bcir-cost-model`), the min-plus plan (`-bcir-plan`) and the budgeted selections
  (`-bcir-rcsp`, `-bcir-rcsp-plan`). It also holds the EFT schedule (`-bcir-schedule-eft`) with
  scheduled pricing and its re-selection sweep (`-bcir-overlap`, `-bcir-overlap-optimize`).
  Beside those are bundle, fusion, layout pivot, sensing, CIM/DVFS, cache contention, the
  allocation pool, the GEM operator costs and lowerings, and the verifier over R1–R25. The A/B
  audit's `sched_eft` and `sched_tokens` rows read "Python-only" in the inventory, but
  `PARITY.md` holds them to `-bcir-schedule-eft` and `-bcir-async`. `sched_waves`'s concurrency
  model has its law-rail form as the lane segment's `affinity` and `unroll` attributes.
- **48 of the 270 modules listed as Python-only have an MLIR counterpart**: a dialect op, a law
  or a pass.
- **16 more sit under a package-level C twin**: `bcir_cfront.c` mirrors the C front end and
  `bcir_make.c` mirrors BCIR Make.
- **Some C units mirror a module without naming it.**
  - `bcir_artifact_bundle.c` is BCAB v1's reader and variant selection: `bcir_ab_open`,
    `bcir_ab_get`, `bcir_ab_select`.
  - `bcir_q4_kernel.c` is `kbcir.lowbit`'s packed Q4 (the BCIRQ4T v1 format).
- **The C emitter encodes JER**, not only the parser. `bcir_emit.c` writes X.697 text, so JER
  runs natively in both directions.

**The inventory itself is the first thing to fix (PORT-0, §5).** It should read `runtime/cpp/`
and `mlir/lib/` beside `runtime/c/`. The C units that mirror a module without naming it should
name it.

## 3. The components that should be native and are not

### Tier 1 — a native path already needs them

| # | Component (oracle) | Lines | Why native | Language, and why | Slot |
|---|---|---|---|---|---|
| N1 | **MC1 disassembly and listing**: `abi.streampack_tool`, `abi.artifact_tool` | 616 | The master roadmap's backend rail lists "a freestanding C twin for MC1 disassembly" as its first open step (§4.2). The C readers it would sit on exist: `bcir_sp_*` and the BCAB view. | **C**, freestanding: the readers are, and a listing over a borrowed view needs no heap | PORT-1; rides between phase slices |
| N2 | **Tokenizers**: `frontends.models.tokenizer` (byte-level BPE), `frontends.models.spm` (SentencePiece) | 511 | Native inference takes token ids only (`bcir_llama_cli --prompt-ids`). The Q8 model header records the tokenizer's SHA-256 but carries no tokenizer, so a standalone C deployment cannot read text. | **C**, hosted-allocator class; table-driven, bounded, fuzzed at the trust boundary | PORT-2; precedes SLM-1's `byte_bpe.v2` and the AI system's fused text pipeline (AL3) |
| N3 | **ECN's executable codecs**: the nine `asn1.ecn*` modules | 12,157 | No C encoder or decoder exists for an ECN-defined encoding. R25 checks ECN's legality on the law rail, but nothing executes it natively. The interface phase's packet codecs (IF-24) are ECN modules. | **C, generated** by `cgen` rather than hand-ported. The generator stays Python; its output is C. | IF-5 gains the ECN backend (this audit adds it to the interface roadmap) |
| N4 | **The event transducer**: `etl.fsm`, `etl.parse`, `etl.events` | 381 | Driver-packet ingestion (GEM-E, M5). C has `bcir_binrec.c` for binary records but no executable transducer or grammar-to-claims parser; MLIR has only the ops. | **C**, freestanding: fixed-state machines over bounded input | PORT-4; with the UART event-driven slices and driver row 1 |

**The C verifier's law surface** is a fifth Tier-1 item, scoped rather than a module.
`bcir_verify.c` is declared over the C front end's subset and covers its laws:
- R1–R18, where R14–R17 are vacuous for scalar integer code;
- R21, advisory.

R19 and R20 (synchronous timing, clock-domain crossing) and R22–R25 run on the oracle and the
law rail only. R19/R20 matter as soon as driver code with timing claims is compiled by `bcir-cc`
without the oracle, so they are **PORT-3, with driver row 1** (the event timers). R22–R25 join
the C path when tensor programs or generated ASN.1 codecs do (IF-6).

### Tier 2 — the C++ toolchain (the migration proposal's CXX ladder)

These are planning-time, hosted and graph-heavy. The runtime consumes their output as data, the
way the native trainer runs PX-3's choices and PX-4's placement. They do not block a native
artifact; they remove Python from the offline toolchain.

| # | Component (oracle) | Lines | CXX slice | Notes |
|---|---|---|---|---|
| N5 | **The GEM+ solver portfolio**:<br>• `gem.exact`, `gem.dispatch`, `gem.diff`;<br>• `kbcir.csp`, `kbcir.joint`;<br>• `kbcir.regions`, `kbcir.dataflow`, `kbcir.expectation`, `kbcir.io_bounds`, `kbcir.objectives`, `kbcir.scope`;<br>• `kbcir.static_memory`, `kbcir.movement` | 8,334 | CXX2, CXX3 | One C++23 library with a C ABI façade in the planner ABI's style: records in, a plan and a certificate out. G12's work-unit budgets carry over unchanged. |
| N6 | **The incremental engine**: `kbcir.delta`, `gem.delta_chain`, `gem.delta_pack`, `verify.delta` | 1,797 | CXX4 | `PARITY.md` declares it oracle-only today: its outputs are the full chain's, which have native twins. A native engine is for re-planning cost at scale. |
| N7 | **The ASN.1 host compiler**: `frontends.asn1` (lexer, parser, AST, lowering, printer) | 4,438 | CXX5 | The A/B audit times `compile_module` (`asn1_compile`) and has no native row. Port only after IF-1 and IF-2 settle the front end; porting a front end that still refuses three constructs would port the refusals. |
| N8 | **Native kernel lowering**: `lower.llvm`, `lower.alias_facts`, `lower.poison`, with `verify.alias` and `verify.poison` | 1,495 | — | `PARITY.md`: "the law rail emits no LLVM kernel". `-convert-bcir-to-llvm` covers only the machine edge (inline assembly, MMIO, atomics, interrupt frames). A BCIR-to-LLVM-dialect lowering in the MLIR rail is the native path, and the JIT microkernel's small runtime codegen slices (driver roadmap §7.5) need it. The Python emitters (`c_kernel`, `wasm`, `jvm_class`) stay as generators and oracles. Behind a written GO/STOP. |
| N9 | **The memory fabric**: `kbcir.ham`, `kbcir.context_shard`, `kbcir.optimization_memory` | 2,289 | CXX6 | Physical HAM is gated on driver evidence (HMF-D0–D5); the planning half can move earlier. |

### Tier 3 — gated on a consumer that does not exist yet

| # | Component (oracle) | Lines | Language | Its consumer |
|---|---|---|---|---|
| N10 | `frontends.models.paged_kv`, `frontends.models.serve` | 1,123 | C | Native serving. Production serving is open in the master roadmap; `bcir_llama.c` already runs greedy generation. |
| N11 | `kbcir.device_manifest` | 534 | C | A JIT-microkernel instance binding device manifests at instantiation (driver roadmap §7.5). Today drivers read generated headers (`uart16550_regs.h`). |
| N12 | `kbcir.dma`, `kbcir.events` | 397 | C | Driver row 2, DMA and the ownership engine, and its blueprint |
| N13 | `telemetry_export` (Prometheus, OTLP and Redfish-shaped adapters) | 910 | C or C++ | A native telemetry or ledger service (the AI system's ledger role) |
| N14 | `kbcir.rcsp` (budgeted selection under live Θ) | 206 | C | On-target re-planning under a budget. Until then `-bcir-rcsp` holds the law, and the C planner covers the unconstrained core. |

### What stays in Python

The remainder, by the placement rule's own list:
- executable semantics and readable legality oracles (`verify`, `model`, the `asn1` schema model);
- research architectures and the ML-breadth references (`kbcir.transformer`, `recurrent`,
  `byte_latent`, `adaptive_transformer`);
- learned priors and calibration (`tile_prior`, `channel_prior`, `bayescal`, `calibrate`,
  `calibloop`, `moegate`);
- hosted training (`bcir.hosted`);
- test and fuzz generators (`kbcir.differential`, `kbcir.fuzz`, `verify.structural_corpus`);
- analysis and reports (`performance_audit`, `telemetry_replay`, `bench`);
- host tooling (`toolchain`, `run`, `bcir.make`'s runner);
- every code generator whose output is C or another target, including `asn1.cgen` and the
  `lower` emitters.

The migration proposal is explicit that these are not to be moved "merely to increase C++ line
count".

## 4. C or C++ for K_BCIR, and whether to grow C++

### 4.1 Where K_BCIR runs today

| Realization | What it is | Role |
|---|---|---|
| `bcir/kbcir` (Python) | the cost algebra, the realization optimizer, RCSP and the whole GEM+ portfolio | the oracle |
| `bcir_kplan.c` (C, 1,623 lines) | the compact planner (G17): canonical phase order, the fused offer with CSE and deforestation discounts, the coupled min-plus path, exact 128-bit arithmetic, the measured-price table (PX-3b-2) and the hydrate to a StreamPack | the native core, freestanding, byte-identical to the oracle (`planner.parity`, `hydrate.native.parity`) |
| `mlir/lib/passes` (C++23) | the cost model, plan, RCSP and Pareto, scheduling, overlap and the operator costs as `bcir-opt` passes | the law rail: an independent third implementation the other two must agree with |

### 4.2 The deciding facts

- **Where the code runs decides the language.** On a target, a driver or a microkernel, or in
  a re-plan at a quiescent boundary, the planner has no allocator, no libc and no unwinder. C
  is the only option, and it exists. Offline, in the toolchain, the planner may allocate freely
  but must be correct over large, growing graphs.
- **The offline solvers are container code.** Branch and bound, the CSP rail's propagation,
  e-graph saturation, region trees, the static planner's portfolio and the delta's dependency
  cones each need dynamic graphs, hash maps, priority queues and memo tables. Freestanding C
  would rebuild each of those per solver. This repository's own record is that the defects live
  in that code: the security laws' L3 (bounds where the resource commits) and L14 (one
  predicate, not five local repairs) came from it.
- **C++23 keeps the memory control that matters.**
  - Allocation goes through an injected `std::pmr::memory_resource`, the C++ spelling of the
    C rail's injected `bcir_host_allocator`. An arena per solve gives one release point, and a
    failing resource drives the same allocation-failure campaign the C rail runs.
  - `std::expected` carries refusals as values with the C rail's status vocabulary, so control
    flow stays explicit and no exception crosses the C ABI.
  - RAII makes idempotent teardown a property of the types, where C has to test it, and move-only
    owners make a lifetime a type error. The seam's `PackOwner`/`Borrow` already does this.
  - What C++ costs, and how each is kept out: hidden allocation (`pmr` containers only, no
    `std::function` in hot loops), global constructors (none), template bloat (explicit
    instantiation), RTTI and exceptions (off at the façade).
- **Independence is kept by checking, not by duplicating search.** The boundary says C++
  "never computes a verdict"; the solver proposes and does not decide. A C++ solver's plan is
  accepted only after the C verifier (R9 over the plan, R10/R11 over the pack) and the law rail
  re-derive it, and those share no code with the solver. Its optimality certificate is checked
  against its own stated bound by the existing certificate checks. It is the same arrangement
  as "learning proposes; certificates dispose" (master invariant 3), applied to search.
- **One native realization per component.** A solver library must not become a fourth copy of
  an algorithm that already has a C twin and an MLIR pass. The core planner stays C only. The
  C++ library takes only what has no native form today: Tier 2's N5–N9. The MLIR passes stay
  the independent law and do not link the library, or the law rail would share bugs with the
  thing it checks.

### 4.3 Why the C++ rail stayed thin, and why that changes now

The build went bottom-up:
- frozen ABIs, codecs, kernels, the runtime, the C front end and drivers, all of them C's
  territory;
- a C++ seam held to the one thing C lacks, dynamic and distributed orchestration.

That was correct for what was built. The work now queued is C++'s territory by the placement
rule:
- the interface phase's ASN.1 host compiler (N7);
- the GEM+ portfolio leaving Python (N5, N6);
- the memory fabric's planning (N9);
- a native kernel lowering (N8).

Keeping that work in Python keeps Python in the toolchain. Writing it in freestanding C builds
containers by hand for code that never runs on a target.

### 4.4 The rules for growing C++

1. **C++ stays hosted.** Nothing freestanding, nothing in a driver, nothing on the
   deterministic artifact path. The C units do not move.
2. **One library, a C ABI façade.** Records in and records out, in the planner ABI's style
   (BKPI/BKPR), so a C or Python caller never sees a C++ type. It is standalone like the seam,
   not part of the MLIR cmake, and built by BCIR Make and CMake from `runtime/manifest.json`.
3. **Arena memory and value errors**: `pmr` only, `std::expected` at every fallible call, no
   exception across the façade, no global state.
4. **Parity before trust.** Each solver reproduces the oracle over a generated corpus before
   any certificate names it. It runs under ASan/UBSan, gets a fuzzer at the façade and joins
   the allocation-failure campaign, exactly as a C twin does.
5. **Its plans are checked, never trusted.** The C and MLIR verifiers decide them.
6. **C++23 on the CI floor.** The CI-default GCC 13 and Clang 18 cover C++23's `std::expected`
   and `pmr`, and the MLIR rail already builds as C++23. The seam's C++17 stays as it is.

## 5. The order

| Slice | What | When |
|---|---|---|
| **PORT-0** | The inventory reads `runtime/cpp/` and `mlir/lib/` and counts their twins. C units that mirror a module without naming it name it (`bcir_artifact_bundle.c`, `bcir_q4_kernel.c`). The A/B audit's Python-only rows show their MLIR pass where one exists. | First; small, tooling only |
| **PORT-1** | N1, MC1 disassembly in C | Between interface-phase slices |
| **PORT-2** | N2, the tokenizers in C | Between slices; before SLM-1 |
| **PORT-3** | The C verifier's R19/R20 | With driver row 1 |
| **PORT-4** | N4, the event transducer in C | With the UART event-driven slices |
| ECN in `cgen` | N3 | Inside the interface phase, IF-5 |
| **PORT-5** | The C++ solver library's foundation: the façade, the arena and error conventions, and the first solver (`gem.exact`) at parity | After the interface phase's stage A |
| **PORT-6 to PORT-9** | N5's remaining solvers (CXX3), the incremental engine (N6, CXX4), the memory fabric's planning (N9, CXX6) and N7, the ASN.1 host compiler (CXX5) | Alternating with the foundation and driver work |
| **PORT-10** | N8, a native kernel lowering in MLIR | Behind a GO/STOP written after PORT-5 |
| Tier 3 | N10–N14 | Each with its consumer |

## 6. Threats to this audit

- **A reference is not a twin.** A native file naming a module can be a comment. Every Tier-1
  and Tier-2 row was checked in the code. The two counts in §2 (48 MLIR counterparts, 16
  package-level twins) are references, labeled as such.
- **Module granularity hides partial twins.** A module with a twin can still have functions
  without one. This audit names partial coverage only where it was checked: the C verifier's
  laws, BCAB's writer (which stays Python, build-time) and JER (complete in both directions).
- **Line counts measure size, not effort.** A port's cost is its parity corpus and gates as
  much as its code.
