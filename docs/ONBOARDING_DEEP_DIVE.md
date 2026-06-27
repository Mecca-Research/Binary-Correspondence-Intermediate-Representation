<!-- allow-retired-paths -->
# BCIR Repository Deep Dive — Onboarding Synthesis

A single-document, up-to-speed synthesis of the **Binary Correspondence Intermediate
Representation** repo: what it is, how every subsystem works, how it was built (the
512-PR arc), its current state, and the ordered next development steps. It is a
*reading* of the tree as it stands — the normative law remains `docs/BCIR_LANGREF.md`
+ the `mlir/` dialect; the live counts remain `docs/STATUS.md` (generated). Where this
doc and those disagree, they win.

> Scope note. This is an onboarding/orientation artifact, not a normative spec. It
> links out to the authoritative docs for every claim. Counts are deliberately not
> hard-coded in prose here (they live in `docs/STATUS.md`).

---

## 0. The one-paragraph thesis

BCIR is a **registry-first, phase-ordered, lane-typed, cost-governed correspondence
IR**. `K_BCIR` is the IR-level optimization *calculus* that selects a legal physical
realization path; `GEM` is the execution IR that hydrates the selected path into a
streamed lane schedule; `MLIR` is the bootstrap framework used to define, verify,
rewrite, and lower BCIR. The central equation:

```
K_BCIR(G | H, Θ) = min_{π ∈ Legal(G,H)}  M(π, Θ)    subject to    R(π, Θ) ⪯ B(H, Θ)
```

- `M(π,Θ)` — schedule-aware makespan: series composition accumulates with `(min,+) ⊗`,
  parallel co-execution combines with `max` → `(max,+)` over the wave/token DAG.
- `R(π,Θ)` — the additive 12-dimensional integer/Q8 resource ledger `Σ Tᵢ ⊗ fᵢ(π)`.
- `B(H,Θ)` — live budgets (thermal/power caps). **Θ changes what is *legal*, not just
  what is fast**: a hot machine makes wide SIMD *infeasible*, so vec8 becomes the
  correct plan, not merely the cheaper one.

Pragmatic positioning (from `docs/BCIR_MASTER_ROADMAP.md` §1): BCIR is a **cost-governed
planning + verification layer *above* LLVM, designed to live inside a driver/runtime —
explicitly *not* a Clang replacement.** It models four things LLVM does not: cost as a
first-class IR object, Θ-feasibility, a principled ML boundary (the two-truth
quarantine), and provenance/reproducibility as obligations. It **matches** Clang on
dense kernels (it delegates instruction selection to the resident backend) and **wins**
only where it exploits program *intent* the backend lacks: gather/scatter avoidance,
reorderable reductions, stride knowledge, and budget feasibility.

---

## 1. The repository is two separate things

| Tree | What it is |
|------|-----------|
| `bcir/` (Python) | the **executable conformance oracle** — runnable today, dependency-free |
| `mlir/` (C++/TableGen) | the **IR law** — the ODS dialect family, the compiled `bcir-opt`, the IRDL projection |
| `runtime/c/` (C) | the freestanding **C twin** — a no-Python compiler/runtime that must byte-agree with the oracle |
| `tools/` | validation scripts (tblgen / IRDL round-trip / build+check `bcir-opt` / docs governance / silicon rig) |
| `docs/` | LangRef, the master + ML/AI roadmaps, PARITY, STATUS, the repo-structure record |
| `llvm-training/` | a **separate** LLVM/MLIR teaching corpus for agents — **NOT part of the IR** |

`bcir/` is the conformance reference; `mlir/` is the law it must agree with
(`docs/PARITY.md`); `runtime/c/` is a second conformance rail. The earlier C++
skeleton (`ir/surface`, `ir/core`, `ir/llvm`, `ir/runtime`) was the first milestone and
has been **retired** — its semantics are subsumed by the oracle + the compiled dialect.
See `docs/BCIR_Repo_Structure.md` and `AGENTS.md`.

### The six IR levels

```
BCIR-0  Semantic Claim Graph        — what transformation is intended      (model/)
BCIR-1  Shaped Data Graph           — tensors/buffers/layouts that exist
BCIR-2  Registry/Placement Graph    — where resources may live
BCIR-3  K_BCIR Correspondence Plan  — which realization path π under H, Θ   (kbcir/)  ← the optimizer
BCIR-4  GEM Stream IR               — StreamPack, waves, fences             (gem/)
BCIR-5  Target Lowering IR          — LLVM/PTX/WASM/C23                      (lower/, codegen/)
```

### The dual-rail spine

Every capability exists on **two independent implementations that must agree**: the
Python **oracle** and either the MLIR **law** or the C **twin**. Agreement is proven,
not asserted — by bit-exact pinned scores, by *generated adversarial differential*
campaigns (two genuinely distinct algorithms over one cost model agreeing across
thousands of random modules), and by Clang behaviour-equivalence for the C frontend.

---

## 2. The semantic model (`bcir/model/`)

Registry-first: memory is addressed by integer **RID**, never a raw pointer; ordering is
an explicit **phase DAG**, never textual order.

- **`Resource`** (`graph.py`): `rid`, `domain` (RAM/VRAM/NVM/MMIO/CXL/HBM), `elem_bytes`,
  `shape`, `layout` (soa/aos/aosoa/blocked), `access` (`flat` O(1) vs **`ham`** O(log n)),
  `priority` (CXL hotness), and `map_gen`/`data_gen` generation tags. `.count` = product
  of shape.
- **`Claim`** (the primitive op): `opcode`, `lane`, `stride_class`, `count`, `stride_k`,
  `rd`/`wr` RID tuples, `hazard` (unique/atomic/barriered), `verify` (none/bounds/exact/
  hash), `bounds` (strict/masked/assumed_safe), the semantic `op` string (`"vector.add"`,
  `"reduce.gather"`), `precision`/`tolerance_ulp` (R17), `dynamic` (count is a static
  upper bound), and two **optional, None-defaulting** annotations — `timing` (R19/R20) and
  `lifetime` (R21) — that keep the entire scalar/C subset unconstrained.
- **`Phase`**: `phase_id` + `deps` (the DAG) + `claims`. A barrier between phases
  materializes intermediates, so fusion/CSE credits are intra-phase only.
- **`Module`**: `{rid: Resource}` + `[Phase]`; `add_resource` enforces R1 (RID uniqueness).

**Lanes are execution *geometry*, not vector hints** (`lanes.py` — integer values are
*normative* and must match `mlir/include/BCIR/BCIRAttrs.td`): `U=0` unit/affine, `UX=1`
cacheline-indexed, `T=2` tile, `GGG=3` gather/scatter ("always legal, must be
minimized"), `A=4` atomic, `H=5` hazard/control. `StrideClass`: SCALAR/UNIT/STRIDED/
CACHELINE/TILE/RANDOM. `Domain`: RAM/VRAM/NVM/MMIO/CXL/HBM.

---

## 3. The K_BCIR optimizer core (`bcir/kbcir/`)

### The cost algebra (`cost.py`)

- **`CostVector`** — a frozen **12-dimensional integer tuple**: `compute, memory, fabric,
  sync, compile, thermal, power, reliability, security, accuracy, contention,
  verification`. This vector is **parity-locked and never extended** — there is no 13th
  axis; new signals (e.g. timing) ride existing axes plus context factors.
- **`couple(factor)`** — the `⊗` operator: `(c·f) >> 8` per dim, a Q8 context scale
  (`256` = ×1.0, *not* scalar multiply). This is the `fᵢ(π)` of the equation: a
  transition's realized cost depends on placement/fusion/thermal, not just the opcode.
- **`dot(weights)`** — scalarize the 12-vector into the single integer the shortest path
  minimizes.
- **`Theta`** — 8 normalized 0..100 live pressures (thermal/power/mem_pressure/...);
  drives *both* the scalarization weights *and* the per-step coupling.
- **`TargetProfile` (H)** — an **open container** of target constants (lane_widths,
  gather_penalty, base_overhead, thermal/power_density, affinity_domains, mem_channels,
  a `MemoryHierarchy` of Q8 `{bw, lat}` tiers vs DRAM=256). New targets are *factories,
  not optimizer code* (`x86_avx2/avx512`, `arm64_neon/sve`, `nvidia_ptx`, `riscv_rvv`,
  `for_host()`).

### Legal realization + the shortest path (`realize.py`, `semiring.py`)

`candidates_for(claim, H)` enumerates *only provably-legal* lowerings keyed on stride
class (RANDOM → gather only; never silently bucket a random gather into a cheaper lane;
`reduce.gather` → blocked vs gather, blocked wins by avoiding `gather_penalty`).
`fused_candidates` bakes intra-phase **CSE/value-numbering** and **producer→consumer
deforestation** (×0.75 memory) into the base costs, shared by every rail so they price
identically. `_context_factor(prev, cand, Θ)` is the *path-dependent* coupling — fusion
locality (memory ×0.75 when a wide candidate shares a read with its wide predecessor) and
the **AVX-512 downclock** (thermal/power ×1.25 when Θ.thermal ≥ 60 and width ≥ 16).
`semiring.dag_shortest_path` solves the layered min-plus DAG in one forward relaxation
(node-id order is topological). The worked anchor that recurs in ~15 PRs and is
FileCheck-pinned on the MLIR rail: `vector_add` on AVX-512 cool Θ → **vec16, score 7808**;
under a 700 thermal/power cap → **vec8, 9472**.

### The constrained + soft rails

- **RCSP** (`rcsp.py`) — `optimize_constrained` solves `min M(π) s.t. R(π,Θ) ⪯ B` by
  label-dominance DP over the same candidate DAG; `Budget.unbounded()` reproduces
  `optimize` exactly (parity). `pareto_plans` recovers the full non-dominated front,
  including non-convex points no weight vector can reach.
- **Soft temperature** (`softdp.py`) — `F_T = −T·log Σ exp(−score/T)` is the Gibbs free
  energy over plans; `T→0` *delegates to integer `optimize`* (bit-identical, the certified
  rail). `T>0` is a differentiable posterior (`∂F_T/∂w = E_π[C]`) — the compiler as a
  learnable layer; an L2/L3 offline organ that anneals + freezes to a T=0 integer table.

### Physicalization (`gem/overlap.py`, `allocator.py`, `mapping.py`)

`gem.overlap.price_scheduled` computes `M(π,Θ)`: claims on distinct affinity domains
combine with **max**, in-bin claims serialize with `(min,+)` (fusion discount only against
the actual in-bin predecessor), the GGG tail decouples as `max(waves, tail)`, phases serial.
`allocator.place()` is an intent-aware "smart malloc" (gains-only: admit a move only if it
fits capacity *and* strictly lowers modeled memory cost).

---

## 4. The intelligence layer — CT5 learned organs (`bcir/kbcir/`)

**Governing law: learning never touches truth.** The L0–L3 placement (LangRef §13): L0
hot path = inference prohibited; L1 plan-time = frozen Q8 tables only; L2 checkpoints =
portfolio + replay gate; L3 = measured, human-actuated meta-policy. Every learned organ
either changes *work* (search order), *selects among already-certified artifacts*, or runs
*offline and freezes to integers + a generation tag* before reaching the deterministic
rail. `Q8 = 256` is the universal quantization unit.

- **`twotruth.py`** — the **two-truth quarantine (MOPC)**: classical truth `v` (binary
  R-law verdicts, "no 0.7 legal") is quarantined from graded truth `(v, w)` (posteriors,
  intervals, regret stored in Q-milli). A graded proposition may *inform* but never
  *become* a legality verdict; the only sanctioned crossing is a recorded `decide()` at a
  frozen threshold. `assert_classical` raises `GradedTruthLeak` at the R-law boundary.
- **`provenance.py` → `proof.py`** — the provenance spine: a `ProvenanceManifest` is "the
  commit hash of a plan" (FNV-1a, i63-masked for MLIR parity, chaining `m_module/m_target/
  m_theta/m_policy` + artifact generation tags). **Manifest equality ⇒ identical plan.**
  `proof.explain/replay/reduce` add per-claim rationale + rewrite certificates and verify
  bit-for-bit reproduction. R13 witnesses the whole chain.
- The organs (each with its honesty mechanism):
  - **`accel.py`** — propose-verify search accelerator: a learned candidate *ordering* for
    branch-and-bound that **provably returns the same optimum** (order-invariant). "Even a
    bad ranker is safe — it changes effort, not the result."
  - **`moegate.py`** — a GNN/MoE router over the claim graph that selects among
    *already-certified* policy experts; deployed only behind a counterfactual replay gate
    (admit iff it never prices worse than the incumbent). "The ensemble of specialized
    compilers."
  - **`egraph.py` / `memory.py` / `operad.py`** — the composition engine (e-graph + CSE +
    equality saturation; R9: a rewrite never raises cost), the fixpoint law
    `memory = Extract(Lim(Res(U)))` (frozen only if `saturated`), and the enriched-operad
    label/index layer (content-addressed by the same FNV, off the hot path).
  - **`calibrate.py` / `calibloop.py` / `bayescal.py` / `microbench.py`** — the
    measure→quantize→freeze→apply→certify loop. `bayescal` ships a *certified* conformal
    ±δ error bar; `MeasuredReplanCertificate.measured` is True *only* with a real silicon
    signal (PMU/RAPL/thermal), never a fabricated win.
  - **`regret.py`** — the L3 instrument (not actuator): a principled **MDL/BIC** retune
    trigger `ΔL = data_fit − (k/2)·ln N > 0` replaces a magic threshold.
  - **`sensing.py`** — spend telemetry only where uncertain (a-priori from the ranker
    margin, a-posteriori from observed variance).
  - **`differential.py` / `fuzz.py`** — the *continuous proofs*: generated adversarial
    Python↔MLIR parity and trust-boundary fuzzing (graceful-rejection vs logic-crash).
  - **`bundle.py`, `portfolio.py`, `weights.py`** — joint multi-claim reorder (with
    certificates), the L2 incumbent + replay gate, and the Θ-modulated scalarization
    weights the dial turns.

---

## 5. GEM, ETL, frontends, lowering (`bcir/gem`, `etl`, `frontends`, `lower`)

- **GEM (BCIR-4)** — `streampack.hydrate` turns a chosen plan into the hot executable
  artifact (one `LaneSegment` per chosen step, with prefetch/block/trace records, plus
  `dispatch=core|pim` and `channel=host|...` for heterogeneous towers). `execute.py` is the
  deterministic phase-sliced executor (topo phase order, ascending claim id within a phase).
  `concurrency.py` forms CT2 waves with a decoupled GGG tail; `schedule.py` adds HEFT-lite
  duration-aware scheduling, token-DAG execution, locality affinity, and a bandwidth-knee
  clamp; `cim.py`/`dvfs.py` add gains-only PIM offload + phase-aware DVFS.
- **ETL / M5 (`etl/`)** — text/binary → claims: a regex lexer + recursive-descent ROP-claim
  grammar (`parse.py`), an FST transducer (`fsm.py`), and a byte-aligned binary-record
  decoder (`binary.py`, e.g. NVMe SQE headers). Mirrored by the MLIR `event.*/fsm.*/
  parse.*/binary.*` op families.
- **Frontends (CT3)** — `rop.py` (registry-first declarative), `map.py` (macro-assembly),
  and the `cfront/` C frontend (see §7).
- **Lowering (BCIR-5)** — *one portable artifact, many backends*: `llvm.py` emits legal
  SSA-only LLVM IR run via clang AOT or `lli` JIT; `wasm.py` compiles to wasm32 + runs via
  node; `c_kernel.py` emits portable C23 (with `_BitInt(N)` Q-fixed lanes, compensated
  reductions, the `restrict` non-aliasing contract); `mlir.py` is the **Python→law bridge**
  (`to_mlir` emits the exact pass-corpus surface so parity is *generated*, not curated);
  `stackify.py` linearizes to WASM/JVM/CIL. The hazard→ordering law lives in
  `memory_model.py` (`unique→monotonic`, `atomic→acq_rel`, `barriered→seq_cst`).

---

## 6. The verifier laws (`bcir/verify/`, `mlir/lib/passes/BCIRVerifyPass.cpp`)

**R1–R18 are first-class on both rails** (Python oracle + `-bcir-verify`), each with a
negative `-verify-diagnostics` MLIR case:

| | | |
|---|---|---|
| R1 registry uniqueness | R2 resolution | R3 domain legality (MMIO write needs ordered hazard; HAM illegal on MMIO) |
| R4 phase-DAG acyclicity | R5 hazard legality | R6 lane legality (the stride→lane table) |
| R7 bounds legality (`reduce.*` writes extent 1) | R8 cost completeness | R9 plan legality + budget feasibility |
| R10 stream provenance | R11 generation validity | R12 lowering legality + MOPC support-preservation |
| R13 policy provenance (recomputes the FNV digest + cross-checks every component vs the IR) | R14 CIM/PIM dispatch | R15 DVFS clock |
| R16 allocator placement | R17 accuracy contract (forces compensated realization) | R18 compositional call-graph (no recursion) |

**R19/R20/R21 are *emerging model laws*** — enforced on the oracle rail today (R21 also
advisory in the C twin) but **beyond the stable MLIR law table**, so `gen_status.py` still
reports R1–R18 first-class. They are driven by optional, None-defaulting claim metadata
(the non-disturbance invariant), excluded from the R13 digest, and run separately from the
pass/fail verdict — so adding them changes no existing score, plan, or verdict:
- **R19** synchronous-timing legality (internal consistency of an optional `Timing` block).
- **R20** clock-domain-crossing (a RAW dep across clock domains must be synchronized).
- **R21** pointer-lifetime: use-after-free / double-free over an optional `Lifetime`
  ({use, alloc, free} + epoch); **already load-bearing for C heap on both rails**.

The test harness (`bcir/tests/run_all.py`) gates campaigns by **tier**: `quick` (default,
gates the C compiler off via a monkeypatched `shutil.which` — so 19 dual-rail C tests
report `skip:no-cc`), `c-runtime`, `silicon-degrade`, `thorough` (CI mode via
`BCIR_THOROUGH=1`, nothing gated → the full suite passes).

---

## 7. The MLIR dialect — the law (`mlir/`)

The ODS/TableGen family is the normative IR; `bcir-opt` is a real `mlir-opt` clone that
parses it, runs the R1–R18 laws, and **recomputes** the K_BCIR optimizer core from first
principles, FileCheck-pinned to the oracle's constants.

- **Encoding** — the whole IR is a **nested symbol-table tree**, not SSA dataflow.
  Container ops (`module`, `registry`, `kbcir.plan`, `gem.stream_pack`, the M5 grammar/
  fsm/binary/func containers) carry `[Symbol, SymbolTable, NoTerminator]`; a claim is a
  symbol carrying all contract attributes + `FlatSymbolRef` operands. A `Symbol` op may
  not produce an SSA result (an MLIR-22 verifier constraint), so the legacy handle types
  are vestigial — resources/paths/packs are addressed by name.
- **Op families** (~85 ops across 10 layered `.td` files): Core (BCIR-0..2), Target, Mem
  (CT1), KBCIR (BCIR-3 planning), GEM (BCIR-4), Trace, Verify (R-laws as IR certificates),
  Opt, LoweringContract (M3/BCIR-5), the M5 Event/Transducer/Parse/BinaryFormat families,
  and Async.
- **Passes + pipelines** (25 registered `-bcir-*` flags) — the canonical order is **verify
  → promote/optimize → lower**. `BCIRCostModel.h` is the header-only C++23 port of the cost
  algebra shared by `-bcir-cost-model / -plan / -rcsp* / -overlap / -compose`; a per-module
  `PlanAnalysis` is computed *once* and shared. `-convert-bcir-to-llvm` is intentionally
  *partial* (`compute→llvm.fadd…`, `barrier→llvm.fence`) — backend codegen is delegated to
  the resident toolchain, exactly like the C twin.
- **The IRDL portability rail** (`mlir/irdl/bcir.irdl.mlir`) — a pure-data, structural-only
  projection loaded by *stock* `mlir-opt --irdl-file=`; uses underscore-flattened names
  (MLIR-22 IRDL forbids dots), and carries no R-laws (the dotted ODS taxonomy is the source
  of truth).
- **How parity is proven** (`docs/PARITY.md`) — enum-value + 12-d-order parity; concept
  cross-map; *recompute-from-first-principles* (the law reproduces 7808/528384/126976/9472
  from the claim + capability alone); two genuinely distinct algorithms over one cost model
  (`differential.law_select` ↔ `-bcir-select-realization`); a generated adversarial campaign
  (≥1500 modules, shrinking on mismatch); 68 `expected-error` negatives; and a
  **byte-identical provenance digest** (the law re-derives the oracle's content hashes from
  the IR alone — the strongest parity claim).

---

## 8. The C frontend / driver (`runtime/c/`, `bcir-cfront` / `bcir-cc`)

`bcir-cfront` is a **verified-subset C23 compiler** that lowers driver/kernel C to the same
claim graph the oracle reasons over (so R1–R18 + the cost model apply unchanged), verifies
it, and emits **behaviour-equivalent C** that Clang then compiles. What it cannot compile it
hands off via a graceful **`--fallback`** route-to-LLVM contract.

- **Dual rail** — the Python oracle (`bcir/frontends/cfront/`) is the conformance reference;
  the C twin (`runtime/c/bcir_cfront.c`, a 4500-line *no-AST, lower-while-you-parse*
  recursive-descent compiler) is the production rail. A feature is "done" only when both
  rails produce a byte-identical claim-graph summary *and* both emits match Clang
  compile-and-run on UB-free-by-construction inputs.
- **The pipeline** (`bcir_cc.c`): `file.c → bcir_cpp (preprocess) → bcir_cfront (lex/parse/
  lower/R1–R18 verify/C.2 attest) → [bcir_plan → bcir_hydrate] StreamPack`. Driver flags
  mirror `cc`: `-I/-D/-U/-std/-E/--target/--fallback/--emit-{c,claimgraph,pack,effects}`.
- **The supported subset is broad**: fixed-width + core ints with a true `(width,
  signedness)` value model, `_Bool`/plain-`char`, floats/`long double`/`_Complex`, multi-
  level pointers (8-byte), arrays incl. 2-D/3-D, struct/union with per-target ABI-faithful
  layout (Itanium bit-cursor bitfields, packed, over-aligned, anonymous members), `enum`,
  `typedef`, `typeof`, `_Generic`, statement expressions, the full compound-literal surface
  (1-D + multi-dim, scalar + aggregate), computed goto, function-pointer locals/members/
  params, atomics (`<stdatomic.h>` + GCC builtins), VLAs (1-D + multi-dim, runtime sizeof,
  bounds-masked), variadics, `<math.h>`/`<complex.h>`/malloc-family as opaque typed external
  edges, and a full L7 preprocessor (`__VA_OPT__`, `#embed`, `__has_*`). **`_Decimal*` is
  *blocked*** — Clang 18 cannot compile it, so it is un-validatable under the
  Clang-equivalence methodology.
- **The safety machinery (§5.12)** — `access_bnd` promotes an indexed access
  `assumed_safe → masked` when the extent is statically recoverable (a sized local/static
  array, or a *stable* malloc/calloc pointer whose count is recovered/snapshotted into a
  hidden immutable `__bcir_extK`), emitting `a[BCIR_CHK(rid, i, n, site)]`. R7 +
  `verify_cfront_lowering` assert every masked claim is discharged by exactly one guard.
- **Gating** — `tools/c/check_runtime.sh` (the fixed-fixture parity + ABI + fallback +
  effects + compile→execute gate) and `tools/c/fuzz_cfront.py` (a 4-axis differential
  fuzzer: outcome + claim-summary parity + `sizeof`/`offsetof` layout + Clang behaviour).

---

## 9. The C runtime — StreamPack, channels, native-object gate (`runtime/c/`)

- **StreamPack ABI** (`bcir_streampack.h`, `docs/BCIR_STREAMPACK_ABI.md`) — BCIR's "WASM
  analog": a 64-byte cache-line header (`static_assert`-locked) + length-prefixed
  segment/prefetch/block/trace records + a zlib CRC-32 trailer, little-endian. **v1 frozen,
  v2 append-only** (a v1 walker stays correct on a v2 pack; encoders emit the lowest
  carrying version). Three views — the prose spec, `bcir/abi/streampack_abi.py`, and the C
  header — must byte-agree.
- **The freestanding executor** (`bcir_runtime.c` decoder + `bcir_exec.c` executor + the
  `bcir_encode.c` re-encoder + `bcir_hydrate.c` producer) closes a **no-Python, no-libc**
  round-trip a driver runs end-to-end. The three trust-boundary decoders are fuzzed under
  libFuzzer + ASan/UBSan (any malformed input returns a status, never an OOB read).
- **The 9-channel heterogeneous tower** (`bcir_channel.c`, `bcir/channels.py`,
  `docs/HETEROGENEOUS_CHANNELS.md`) — six real arch backends (`x86_avx512/avx2`,
  `arm64_neon/sve`, `riscv_rvv`, `nvidia_ptx`) + three *modeled* future backends
  (`fpga_systolic`, `nvme_stream`, `hbm_pim`). `orchestrate()` plans a module across the
  tower (matmul→GPU, scatter→HBM/PIM, near-storage reduce→NVMe) into *one* StreamPack whose
  per-segment `channel` tag the executor dispatches on. A `HardwareChannel` isolates
  everything arch-specific (the fix for the class of bug where the x86 `perf_event_open`
  syscall number silently fired on ARM).
- **Bounds quarantine** (`bcir_quarantine.c` + `_recover.c`) — `BCIR_CHK` is
  `i < n ? i : bcir_bounds_quarantine(...)` (in-bounds never calls the handler). The weak
  default records-and-aborts; the reference recovery override is the **two-truth crossing
  made concrete**: a frozen per-site policy table → an audited `decide` (clamp iff admitted)
  → a recorded crossing in a decide-ring. The C twin of `kbcir.twotruth.Decision`.
- **The native-object gate** (`docs/BCIR_NATIVE_OBJECT_GATE.md`) — **DEFERRED, correctly.**
  BCIR does *not* hand-roll isel/regalloc/ELF emission. The chosen path emits C23 or LLVM IR
  and hands it to the resident backend; `codegen_object_c` is verified end-to-end to real
  ELF for **eBPF, x86-64, and aarch64 (Raspberry Pi 5)**. Native isel stays behind explicit
  GO criteria (G1 no resident backend + G2 measured ≥2× economics) that no seeded target
  meets today; revisit for a bare PIM/CIM controller or a driver-resident eBPF JIT.

---

## 10. The development history (the 512-PR arc)

The repo is heavily agent-driven (OpenAI Codex for the training corpus; Claude Code for the
system) with a disciplined, high-cadence rhythm. The arc, oldest → newest:

- **#2–#101** — *genesis + pivot.* A complete C++ BCIR compiler skeleton built in one day,
  then a pivot to "LLVM-IR as the semantic source of truth" (a 64-byte `BcirClaimV1`, the
  `runtime/llvm/` seed), then a large CI-gated `llvm-training/` agent corpus. Ends mid-thrash
  on a deep MLIR bridge.
- **#102–#201** — *the real system begins.* PR **#153** restructures the repo and realizes
  the central equation as a runnable optimizer (the Python oracle + the MLIR/IRDL dialect
  law + `PARITY.md`; `vector_add = 7808` is pinned). The legacy C++ `ir/` skeleton is
  **retired** (#157). The verifier grows 4→**R1–R13** ("proven, not trusted," #163), the
  TMSAO generalization adds the constrained series-parallel equation + RCSP + GEM
  wave-overlap, and the full learning roadmap lands (temperature dial, MDL retune, Bayesian/
  conformal cost model, MoE gate, search accelerator) — all "learn offline, freeze to the
  certified rail." First measured wins: gather avoidance ~6.5–16×, budget feasibility as a
  correctness property.
- **#202–#301** — *port to the law rail + open the C frontier.* The entire deterministic
  optimizer is ported to **C++23 on the MLIR rail, bit-exact** ("recompute, don't trust");
  generated adversarial differential parity replaces curated pins; laws grow to **R1–R18**;
  a full no-Python StreamPack round-trip + a `bcir-cc` C compiler appear; BCIR goes
  hardware-agnostic with ARM first-class and the heterogeneous channel architecture. The
  MLIR toolchain moves to gating **LLVM 22** (obtained via **conda-forge** when CI's
  `apt.llvm.org` egress was blocked — #246).
- **#302–#401** — *the C frontend, completed + hardened.* The preprocessor, lexer/parser
  literal/float/libm surface, and a "Phase-4 general replacement compiler" (Clang-grade
  diagnostics, an ABI matrix, optimizer-correctness analyses, fuzzing). A long bug-hunt:
  *compiling and running* the emitted C against Clang surfaces a tail of latent miscompiles
  (scope leaks, signedness in an unsigned value model), each fixed and locked behind a new
  differential gate.
- **#402–#501** — *near-complete C11/C23.* The 4-byte→8-byte pointer model closed; variadics,
  `_Generic`, `typeof`, `_Complex`, VLAs, computed goto; a differential fuzzer that found
  ~18+ distinct bugs; and three **forward tracks** seeded (RTL/synchronous-timing R19/R20,
  the naked-pointer R21 safety system, "C as a substrate") — all *additive, vacuous,
  digest-excluded*.
- **#502–#512** — *the close.* The entire array-compound-literal surface completed dual-rail
  (with three pre-existing silent miscompiles on the *regular* declarator path fixed first —
  "the literal guard never comes off an unproven foundation"); a **fourth parity axis
  (storage-extent)** added to the harness; then the pivot to forward planning — roadmap
  §5.14 (the MLIR-catch-up + freestanding-C23-driver arc, release 0.3b) and the ML/AI
  integration roadmap (Phase A–F). **#512 is the most recent PR** (2026-06-26).

The connective tissue across all 512: **oracle-first prototyping, bit-exact parity gates,
generated adversarial differentials, gains-only opt-in additions that never disturb the
pinned spine, and an unusual culture of intellectual honesty** — retracting false wins,
re-diagnosing prior PRs, and letting negative findings redirect the roadmap.

---

## 11. Current state & verification (validated this session)

Everything below was *run*, not inferred. The repo is in a healthy, fully-green state.

| Check | Result |
|---|---|
| Python conformance suite (`BCIR_THOROUGH=1 python -m bcir.tests.run_all`) | **878 passed, 0 failed** |
| (default `quick` tier) | 859 passed; the 19 "failures" are `skip:no-cc` artifacts of the tier's deliberate compiler gate — **not real failures** |
| LLVM-training: examples / invalid-fixtures / opaque-pointers / bcir-mapping / csv-schema / manifests / dataset | all **pass** |
| LLVM-training autograder self-test (`grade-exercises.py --self-test`) | **700/700 (100%)**, full confidence |
| LLVM-training autograder unit tests | **14/14 OK** |
| EVAL self-test (`llvm-training/EVAL.md`, 30 questions + path prompts) | answered **30/30**, band 27–30 (see §12) |

**The MLIR rail — previously skipped, now fully validated.** This environment ships LLVM 18
with no MLIR tools by default; LLVM 22 (CI's gating version) is pulled from `apt.llvm.org`,
which this environment's egress policy **denies**. Ubuntu noble's own (allowed) mirror
carries the full **LLVM/MLIR 20.1.2** toolkit, and the repo's build scripts are
version-agnostic (they pick the highest installed `/usr/lib/llvm-*`), so installing it
unblocks the entire rail:

| MLIR-rail check | Result on LLVM/MLIR 20.1.2 |
|---|---|
| `bcir-opt` build (`tools/wsl/build_mlir.sh`) | builds; all 25 `-bcir-*` passes registered |
| ODS generators (`tblgen_check.sh`) | pass |
| pretty-ODS corpus + FileCheck (`check_ods_examples.sh`) | pass |
| verify/promote/lower passes (`check_passes.sh`) | pass |
| MLIR-bytecode round-trip (`check_bytecode.sh`) | pass |
| IRDL projection on stock `mlir-opt` (`check_corpus.sh`) | pass |
| training MLIR tiers (`verify-mlir-examples.sh --require-tools`) | **Tier 4** reached (mlir-translate→llvm-as→opt-verify), expected `bcir.unlowered` conversion-failure demonstrated |

> The exact-version path to LLVM 22 in this environment is **conda-forge** (`mlir=22`,
> the same workaround CI history used in PR #246) — but the rail validates identically on
> 20, since the MLIR verifier "has only tightened since LLVM 19" and the dialect already
> carries the `SymbolTable` traits that constraint requires.

**The single genuinely-deferred result in the whole repo** is a *measured* (not modeled)
bare-metal replan win: the software path is push-button (`tools/silicon/measure_replan.sh`
prints `rig-ready: YES/NO`) and CI-exercised in degrade mode, lighting up the moment a host
with PMU + RAPL + a userspace cpufreq governor runs the runbook. This is the "intelligence
ahead of substrate" risk and the top differentiator (`docs/HARDWARE_VALIDATION.md`).

---

## 12. LLVM training — completion record

The `llvm-training/` corpus is a context/reference pack (not fine-tuning data) that teaches
agents LLVM/MLIR/BCIR mechanics before they touch the IR: 00–19 chaptered lessons, a recipe
index, ~42 exercises (write/repair/predict/review/MLIR/backend-JIT/binary-analysis/
BCIR-lowering families) + templates, a deterministic stdlib autograder with per-exercise
JSON manifests and tiered MLIR validation (0–4), a `bcir-mapping/` guide, and a 30-question
EVAL self-test.

Completion evidence (this session): the full verification suite is green (§11), the
autograder self-test scores 100% with full confidence, and the EVAL self-test was answered
**30/30** — grounded in the actual corpus files (not memory), with concrete IR claims
spot-checked against the live LLVM/MLIR 20.1.2 toolchain (e.g. the array-of-structs GEP
`getelementptr inbounds %Entry, ptr %base, i64 %index, i32 1` assembles with `llvm-as`; the
HAM-hint prefetch `immarg` literals pass `opt -passes=verify`). The completed self-test is
preserved at `llvm-training/eval/EVAL_COMPLETED.md`.

---

## 13. The roadmap — ordered next development steps

From `docs/BCIR_MASTER_ROADMAP.md` (§5.14, §6, §7) and `docs/BCIR_ML_AI_INTEGRATION_ROADMAP.md`.
The roadmap is **dependency-ordered** — building out of order creates debt.

**The next deliverable is release `0.3b`: the MLIR-catch-up + freestanding-C23-driver arc
(§5.14)**, which closes the "law trails the oracle" gap. Phased:

0. **Hygiene** — honest test tiers + an explicit `c-runtime` CI gate (label the `skip:no-cc`
   quick-tier results correctly); document the naked-pointer policy in LangRef + the
   C-frontend guide.
1. **Promote R19/R20/R21 to first-class** — the six R14–R18-pattern artifacts, each
   parity-gated: (a) `#bcir.timing` + `#bcir.lifetime` `OptionalAttr` on `bcir.claim`;
   (b) `verifyR19/R20/R21` in `BCIRVerifyPass.cpp`; (c) LangRef §10 sections; (d) a negative
   FileCheck case; (e) promote the C-twin R21 advisory → verdict; (f) widen `gen_status.py`
   from `range(1,19)` to `range(1,22)` so status reports **R1–R21**.
2. **Extend MLIR for the *law-bearing* C semantics only** (the filter: a feature gets MLIR
   only if it affects effects/aliasing/lifetime/provenance/ABI/volatile-atomic ordering/
   timing/target/verification/cost). **Add:** object lifetime (the R21 attr), volatile
   access, atomic RMW/CAS (`#bcir.mem_ordering`), function-pointer/indirect-call effect set,
   pointer extent-provenance, and a target ABI/calling-convention `lower_contract`. **Keep
   frontend-only:** storage/linkage, C initializer semantics, source-location spelling.
3. **Ship the freestanding-C23-driver release** — multi-file project mode + per-project
   verdict, `compile_commands.json`, dependency output (`-M/-MM/-MF`), real
   UAPI/CMSIS/PCIe/NVMe/ACPI register-map fixtures, R1–R21 clean with per-file fallback +
   emitted C/object artifacts. **The first externally-usable BCIR compiler deliverable.**

**The strategic frontend arc** (§5.7): Phase C (the solid C frontend + a *generalized*
self-verifying C backend for an arbitrary claim graph — the keystone) → Phase D (drivers /
the Hardware Description Layer: import Linux register maps / PCIe / ACPI; per-channel resident
drivers via `channel.json`) → Phase F (full frontends: C++ then Python). Phase M (selective
ML ops) runs throttled-parallel, never blocking the keystone.

**The ML/AI program** (`docs/BCIR_ML_AI_INTEGRATION_ROADMAP.md`, order A→B→(C∥D)→E→F):
- **A** — max out C23 as the inference substrate (`_BitInt(N)` quantized weights, `<stdbit.h>`
  fixed-point math, `[[unsequenced]]`/`[[reproducible]]` as fusion-legality signals).
- **B** — tensor ops as *claims* (`gem.matmul/conv/attention`), letting the existing min-plus
  search choose layout/tile/loop-order over new cost-vector dimensions (shape/dtype become the
  next R-laws after R21); gradients as operad 2-cells; integrate ATLAS/GSL/FFTW/BLAS-LAPACK via
  the existing `c.call.libm` seam — **do not rebuild XLA/TF**.
- **C/D/E/F** — data/memory organs (tabular streaming → tensors; a BCIR vector DB from
  *materialized HAM* + HDF5/LMDB; the operad index as the vector key) → language reach (Fortran
  via `ISO_C_BINDING`) → ML-guided hardware deployment → higher cognition (a deterministic
  tokenizer; the BFD object backbone; the L3 meta-policy closing on itself, staying graded +
  human-actuated).

**The first concrete, gateable slices**: A1 (`_BitInt(N)` end-to-end + the unsequenced
fusion signal) + B1 (one `gem.matmul` op, oracle + MLIR law, K_BCIR choosing the tile/loop) +
B5 (wrap one BLAS `gemm` through `c.call.libm`) + C1 (a column-oriented streaming buffer).

**Deferred (gated):** the measured real-silicon replan win (§5.4, top differentiator); native
isel (§5.5, behind the GO/STOP gate); and the learned organs / offline calibration / generators
stay Python *by design* (§5.6 — porting them would violate the two-truth quarantine).

**Release ladder** (§7): `0.2` reproducible compiler ✅ · `0.3` measured adaptive compiler ◑ ·
**`0.3b` freestanding-C23-driver + law catch-up ☐ (NEXT)** · `0.4a` proof-carrying mechanism ✅ ·
`0.4b` proof-carrying contract ☐ · `1.0` ☐ (stable ABI, ≥2 measured hardware targets, one
external frontend).

---

## 14. The disciplines that hold it all together

1. **Dual-rail parity as law** — oracle ↔ MLIR (recompute-from-first-principles + generated
   adversarial differential) and oracle ↔ C twin (claim-summary parity + Clang behaviour).
2. **Two-truth quarantine / L0–L3** — a graded proposition may inform but never *be* a
   legality verdict; learning never runs on the hot path.
3. **Prototype-then-port** — prototype in the Python oracle, then *stop and build the real
   thing* on the production rail (MLIR for law, C for runtime), parity-gated.
4. **Provenance as the spine** — every plan is a replayable manifest; manifest equality ⇒
   identical plan; nothing is globally immutable, but everything is immutable within its
   generation.
5. **Additive, vacuous, digest-excluded** — new laws/metadata default to absent, early-return
   when absent, and stay out of the provenance digest, so they provably disturb nothing.
6. **Gains-only, opt-in, off the hot path** — every "smart" organ is lazy-loaded and deviates
   from the baseline only when the modeled cost strictly improves.
7. **Honesty over theater** — refuse false/unmeasurable wins, report negative findings, and
   let them redirect the roadmap; the harness must measure what matters (the storage-extent
   axis was added the moment behaviour-equivalent over-sizing proved invisible).

---

## 15. Where to find things (file index)

| You want… | Look in |
|---|---|
| the semantic model (lanes/opcodes/claims) | `bcir/model/` |
| the cost algebra + optimizer | `bcir/kbcir/cost.py`, `realize.py`, `semiring.py`, `rcsp.py`, `softdp.py` |
| the learned organs + two-truth + provenance | `bcir/kbcir/{accel,moegate,egraph,memory,operad,calibrate,bayescal,regret,twotruth,provenance,proof}.py` |
| GEM execution / scheduling / StreamPack | `bcir/gem/` |
| ETL / event transduction | `bcir/etl/` |
| ROP / MAP / C frontends | `bcir/frontends/` |
| LLVM/JIT/WASM/MLIR/C lowering | `bcir/lower/` |
| the verifier laws (Python) | `bcir/verify/__init__.py` |
| the MLIR dialect (the law) | `mlir/include/BCIR/*.td`, `mlir/lib/`, `mlir/tools/bcir-opt.cpp` |
| the verify/cost-model/plan passes | `mlir/lib/passes/`, `mlir/lib/BCIRCostModel.h` |
| the IRDL projection | `mlir/irdl/bcir.irdl.mlir` |
| the C frontend + driver | `runtime/c/bcir_cfront.c`, `bcir_cc.c`, `bcir_cir.h`, `bcir_cpp.c` |
| the C runtime + StreamPack + channels | `runtime/c/bcir_{runtime,exec,encode,hydrate,streampack,channel,quarantine}.*` |
| the C verifier twin | `runtime/c/bcir_verify.c` |
| the CLI entry point | `bcir/run.py` |
| MLIR-rail build/validate scripts | `tools/wsl/`, `tools/irdl/` |
| the C parity/fuzz gates | `tools/c/check_runtime.sh`, `tools/c/fuzz_cfront.py` |
| the silicon rig runbook | `tools/silicon/measure_replan.sh` |
| **the law / roadmaps / parity / status** | `docs/BCIR_LANGREF.md`, `docs/BCIR_MASTER_ROADMAP.md`, `docs/BCIR_ML_AI_INTEGRATION_ROADMAP.md`, `docs/PARITY.md`, `docs/STATUS.md` |
| the LLVM training corpus | `llvm-training/` (start at `START_HERE.md` → `INDEX.md` → `RECIPES.md`) |
