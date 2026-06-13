# BCIR Build Blueprint v0.2 — Target-Open, Heterogeneous, AI-Guided

Companion to [`BCIR_LANGREF.md`](BCIR_LANGREF.md) (the law). BCIR is a **container
for any target** — x86, ARM, RISC-V, GPU — so hardware-specific compilers/drivers
can later be built from firmware/ISA/opcode/registry tables. Like WASM, one
portable BCIR-4 StreamPack feeds both **AOT** and **JIT** kernels; the optimizer
that picks realizations is AI-guidable (CT4).

## Capability tracks

| Track | Capability | Status |
|---|---|---|
| **CT1** | Target-open container + memory hierarchy (HBM / HAM / CXL) | oracle done; law authored + validated |
| **CT2** | Mixed-stride concurrent graph exec + cache unroll + thread→cache affinity | oracle done (`bcir/gem/concurrency.py`); law: lane-segment `affinity`/`unroll` |
| **CT3** | ROP & MAP performance paradigms (front-ends → claims) | oracle done (`bcir/frontends/{rop,map}.py`) |
| **CT4** | ML-guided "data DNA" telemetry loop (thermal/voltage → exec mgmt) | oracle done (`bcir/telemetry.py`, `bcir/kbcir/calibrate.py`); law: `bcir.trace.data_dna` |
| **CT5** | AOT + JIT + WASM backends per target (WASM-like agnosticism) | oracle done — AOT (clang) + JIT (lli) + **WASM** (`bcir/lower/wasm.py`, runs via node) |

## CT1 — done (oracle) / authored (law)

- **Open container.** `kbcir.cost.TargetProfile` / `bcir.target.capability` is a
  data descriptor for any target (`triple`, `isa_features`, `lane_widths`
  warp/scalable-aware, `affinity_domains`, a `MemoryHierarchy`). Adding a target
  is a **factory/data entry, not optimizer code**. Seeded: `x86_avx2`,
  `x86_avx512`, `arm64_neon`, `arm64_sve`, `nvidia_ptx`; `riscv_rvv` proves
  extensibility. Demonstrated: the same `vector_add` realizes as **vec32** (GPU
  warp), **vec4** (ARM NEON), **vec16** (x86 AVX-512), **vec8** (AVX2).
- **Memory hierarchy (C_mem = Σ_levels).** A resource's `Domain` selects a `Tier`
  (`L1/L2/L3/DRAM/HBM/CXL/SSD`); the K_BCIR `memory` cost scales by that tier's
  bandwidth + latency factors (Q-fixed vs DRAM, so RAM costs exactly as before).
  **HBM** ≈4× bandwidth → cheaper memory cost. **HAM** (`access="ham"`) turns
  random addressing into an O(log n) walk → beats flat gather. **CXL** semantic
  swap is a priority-aware tier between DRAM and SSD.

## Dual-rail architecture (locked)

```
Track A: ODS / TableGen / bcir-opt
  Full BCIR dialect, pretty syntax, R1-R12 verifier, optimizer, proof discharge.

Track B: IRDL projection / stock mlir-opt
  Pure-data BCIR projection, generic syntax, structural verification, portability
  proof, no BCIR-authored compiled code.
```

The IRDL rail (`mlir/irdl/bcir.irdl.mlir`) is the **passport**: it proves BCIR
artifacts can be shipped as data and loaded by a standard prebuilt engine. It is
structural only (no `irdl.c_pred`, which would require compiled C++ and block
runtime registration). Deep semantics stay in the ODS rail + the `bcir/` oracle.

## CT2–CT5 — built in the oracle

- **CT2** `bcir/gem/concurrency.py`: the phase DAG becomes concurrent waves —
  independent claims co-execute, conflicting claims serialize, the GGG/random tail
  is decoupled so it never stalls the U/UX/T stream, and each wave's claims pin
  round-robin to the target's affinity domains (oversubscription -> `contention`).
- **CT3** `bcir/frontends/{rop,map}.py`: a registry-first declarative ROP front-end
  and a terse MAP macro-assembly front-end, both parsing text -> verified BCIR
  claims that feed K_BCIR/GEM (reusing `bcir/verify`).
- **CT4** `bcir/telemetry.py` + `bcir/kbcir/calibrate.py`: the "data DNA" schema
  (cycles/bytes/misses/thermal/**voltage**/utilization + provenance) over a
  `TelemetrySink` (null/list/file; Kafka is the intended broker backend), an EWMA
  cost-calibrator that folds telemetry back into Θ, an adaptive policy selector,
  and a `rehydrate_decide` (keep/patch/repack/replan). Θ carries a `voltage` axis.
- **CT5** `bcir/lower/jit.py`: in-process JIT over the *same* StreamPack lowering —
  emit kernel + harness, compile to IR, `llvm-link`, run with `lli`. One portable
  artifact, two backends (AOT clang + JIT lli).
- **M5 Event Transduction Layer** (shipped earlier): `bcir.event.*`, `bcir.fsm.*`,
  `bcir.parse.*`, `bcir.binary.*` make text/binary/packet/telemetry ingestion all
  instances of the same correspondence machinery.

### Phase 9 (done): real per-target codegen
- `bcir/codegen/`: BCIR → LLVM IR → real artifacts via `llc`. Seeded
  `bcir.target.lower_contract` targets, each validated: **aarch64** (ARM) and
  **riscv64** (cross-targets, ELF objects), **nvptx64** (GPU PTX asm), **bpf**
  (eBPF — an integer-only scalar kernel, since eBPF has no FP), **x86_64**, and a
  portable **C-source fallback** (compiles anywhere). SPIR-V is a registered
  descriptor that reports cleanly when no SPIR-V backend is built into `llc`.
- **Scope, stated plainly:** per-target codegen is a *data-driven routing layer
  over the LLVM toolchain* (`llc`/`clang`/`lli` as subprocesses). BCIR-native
  instruction selection, register allocation, and linking are **future work**;
  the `bcir.target.lower_contract` descriptors are the seam where they land.
- The float kernel emitter gained an `elem`/`width_override` so FP-less targets
  (eBPF) get an integer scalar kernel. CLI: `python -m bcir.run vector_add --codegen all`.

### Phase 8 (done): runtime + concurrency + memory model
- **Freestanding C StreamPack runtime** (`runtime/c/bcir_runtime.{h,c}`): loads the
  frozen ABI with **no libc** (only `<stddef.h>`/`<stdint.h>`), bitwise CRC-32,
  zero-copy segment walk. A cross-language parity test (Python encodes → C decodes)
  gates the ABI (`tools/c/check_runtime.sh`).
- **`!bcir.token` async model**: `bcir.async.fork` (launch a claim, yield a token)
  + `bcir.async.await` (join) in the dialect + IRDL; `bcir/gem/async_tokens.py`
  computes the explicit fork/await dependency plan (each claim awaits its earlier
  conflicts; independent claims await nothing).
- **Memory model (atomics → LLVM ordering)**: `BCIR_MemOrdering` enum + a barrier
  `ordering` attr; `-convert-bcir-to-llvm` lowers `bcir.barrier {ordering}` to the
  matching `llvm.fence`; `bcir/lower/memory_model.py` is the normative
  hazard→ordering / ordering→LLVM map (clamped to the fence-legal set).

### Phase 7 (done): portable artifact + WASM + stackify
- **Frozen StreamPack binary ABI v1** (`bcir/abi/streampack_abi.py`,
  `runtime/c/bcir_streampack.h`, `docs/BCIR_STREAMPACK_ABI.md`) — the portable
  artifact, with a CRC trailer and a lossless round-trip.
- **WASM** deployment via the LLVM path (`bcir/lower/wasm.py`): the K_BCIR-selected
  kernel compiles to `.wasm` (`clang --target=wasm32` + wasm-ld) and **runs via
  node**, self-checking — one artifact, a second portable backend.
- **Generic stackify** (`bcir/lower/stackify.py`): register-form `Expr` →
  postfix stack-op sequence → thin `to_wasm/to_jvm/to_cil` encoders — the shared
  foundation for the stack-machine bytecode targets (WASM / JVM / CIL).

### Done since (LangRef M3 + CT4 depth)
- **The propose-verify search accelerator** (Phase 19, LangRef §13): `kbcir.accel`
  is the safest place learning touches the (hot-ish) plan-time search — a learned
  candidate ordering speeds the exact RCSP search but **provably returns the same
  optimum**, because the optimum is invariant to candidate visitation order.
  `optimize_ordered` is an exact branch-and-bound with an *admissible* suffix
  bound + budget feasibility pruning; a good order (greedy, or the learned
  `LearnedRanker` trained on the exact optimizer's own choices, frozen to Q8)
  tightens the incumbent earlier — it finds the optimum as the *first* complete
  plan and prunes more (fewer expansions) — while any order, even worst-first,
  returns the exact optimum. The `accelerator_certificate` checks equivalence to
  `optimize_constrained` (mismatches must be 0), witnessed by **R13**
  (`bcir.kbcir.search_accel`: admitted ⇒ zero mismatches; a deployed accelerator
  must be certified exact). The network proposes an order; the verifier disposes.
  CLI: `python -m bcir.run --accel`.
- **The learned MoE gate — a GNN over the claim graph** (Phase 18, LangRef §13):
  `kbcir.moegate` is the literal "ensemble of specialized compilers" — a one-layer
  Graph Neural Network over the claim graph (message passing + hardtanh embed +
  mean readout) with a softmax routing head, replacing the hand-coded
  `classify(Θ)` with a *learned* router over the certified portfolio experts.
  Trained by exact softmax-cross-entropy SGD (hand-rolled, no autograd, no deps)
  on the **regret ledger** (`ledger_labels`: the label is the hindsight-best
  expert). It is the *safe* learning regime — it only selects among
  already-certified experts, never emits a table — and deploys **frozen** to a Q8
  integer gate (`freeze`/`FrozenGate`, exact integer routing via a hardtanh clamp,
  deterministic across hosts) only behind an admitting **replay certificate**
  (`gate_replay_gate`: no regression vs `classify` under M(π,Θ)). Witnessed by
  **R13** (`bcir.kbcir.moe_gate`: routes a known portfolio, admitted ⇒ zero
  regressions, a deployed gate must have passed). CLI: `python -m bcir.run --moe`.
- **Bayesian cost model with conformal error bars** (Phase 17, LangRef §13):
  `kbcir.bayescal` upgrades the point microbench table to a posterior with a
  certified interval — a conjugate-Gaussian (VI-exact) posterior over each Q8
  ratio (`gaussian_update`), a distribution-free split-conformal `±δ` at a
  stated coverage (`conformal_delta`, Vovk), and likelihood-free **ABC**
  (`abc_calibrate`) that uses the GEM/`optimize` forward model as the simulator
  (accept a proposed table iff its *simulated* plan score lands within epsilon
  of the observed). The frozen `BayesianCalibratedProfile` applies like a point
  table (Q8) and additionally carries the conformal `δ`; the conformal guarantee
  is witnessed under R8/R13 (`bcir.kbcir.calibration` `coverage_milli` /
  `random_delta_q8`: coverage a real probability, `δ ≥ 0`, no interval from ≤ 1
  sample). CLI: `python -m bcir.kbcir.microbench --bayes`.
- **MDL / Bayesian-evidence retune law** (Phase 16, LangRef §13): the boundary
  dashboard's retune trigger is now the principled two-part code
  `ΔL = Σ regret_i/best_i − (k/2)·ln(N)` (the BIC large-sample Bayesian
  evidence, Schwarz 1978) instead of a magic regret-rate threshold — a swap is
  recommended only when the accumulated relative regret pays for the
  model-complexity cost of specifying and certifying it (few episodes of small
  regret never flag; sustained or large regret does, and the *same* per-episode
  regret flips keep→retune as evidence accumulates). `kbcir.regret` carries
  `data_fit_nats`/`complexity_nats`/`evidence_margin`; the verdict is folded
  into **R13** on both rails (`verify_provenance` verdicts obligation +
  `bcir.kbcir.regret_ledger` `data_fit_milli`/`complexity_milli`/`verdict`):
  a verdict is illegal unless consistent with its MDL margin
  (retune ⟺ data_fit > complexity).
- **The temperature dial** (Phase 15, LangRef §8/§13): `kbcir.softdp` is the
  soft, differentiable twin of `optimize` — a log-sum-exp dynamic program over
  the same legal candidate DAG whose `T → 0` limit recovers the tropical
  optimizer *exactly* (at `T = 0` it delegates to the integer `optimize`, so
  every pinned score is bit-identical). At `T > 0` it returns the Gibbs free
  energy `F_T`, per-claim plan marginals, and the expected cost vector — with
  an exact analytic gradient `∂F_T/∂w = E_π[C]` (no autograd), making the
  optimizer a learnable layer. The single abstraction behind the
  Solomonoff/Bayesian/softmax lineages: learn at `T > 0` offline, anneal and
  freeze to a `T = 0` table for the certified hot path. Law:
  `bcir.kbcir.soft_select` (R9: `F ≤` score; `T = 0 ⇒ F ==` score). CLI:
  `bcir.run --soft-temp T`.
- **R13 policy provenance + the regret ledger** (Phase 14, LangRef §10/§13):
  the law that makes rule swaps witnessable — a promoted portfolio entry
  requires its admitting replay certificate, a calibrated profile must present
  its frozen table with matching generation and constants (no silent drift),
  and regret-ledger books must balance (`verify.verify_provenance`,
  `-bcir-verify` R13, `bcir.verify.policy_provenance`). The **regret ledger**
  (`kbcir.regret`) is the L3 instrument: per-rule hindsight regret under one
  neutral yardstick, with `boundary_report` rendering keep/retune verdicts —
  the dashboard that says where the heuristic/learned boundary belongs. An
  instrument, never an actuator: flagged rules go through the L2 gate, R13
  witnesses the chain, actuation stays human. CLI: `bcir.run --regret`.
- **Physics-anchored calibration + learning placement** (Phase 13, LangRef §13):
  the L1 microbenchmark harness (`kbcir.microbench`) measures streaming/strided/
  random/compute regimes with deterministic access orders, quantizes to Q8
  ratios (stream = 256 by definition), and freezes generation-tagged tables
  that substitute the seeded constants (`CalibratedProfile.apply`; the
  checked-in ratio-1 reference table reproduces them exactly — vec16 @ 7808
  survives table application). The L2 portfolio (`kbcir.portfolio`) holds
  frozen, generation-tagged gain schedules selected by a deterministic
  workload-class table; swaps deploy only behind the **replay gate** — a
  counterfactual no-regression certificate judged on the incumbent's M(π,Θ).
  The **L0 prohibition** (no learned inference on the hot path) is normative.
  Law: `bcir.kbcir.calibration` / `portfolio` / `replay_certificate` +
  capability `cal_gen`, verified under R8/R9.
- **Duration-aware GEM scheduling + StreamPack v2** (Phase 12): the five GEM
  upgrades — `gem.schedule.schedule_eft` (HEFT-lite: LPT priority + earliest
  finish time, hazard producers serialize by claim id), `execute_tokens` (the
  `!bcir.token` DAG replaces phase barriers, so independent later-phase claims
  overlap — pipelined phases fall out of the awaits), locality-aware affinity
  (earliest-finish ties prefer the domain holding the claim's operands),
  the bandwidth-knee clamp (`bandwidth_knee(H)` from `TargetProfile.mem_channels`;
  bandwidth-class claims queue past the roofline knee), and **StreamPack ABI v2**
  (append-only: header `pipeline_depth`, prefetch `buffers`; double-buffer
  contracts emitted by `hydrate_pipelined`; the freestanding C runtime decodes
  both versions and packs without v2 features stay byte-identical frozen v1).
  Law: `bcir.gem.schedule` + v2 attrs on `stream_pack`/`prefetch`/`lane_segment`,
  verified under R9/R10.
- **The constrained series-parallel equation** (Phase 11): the LangRef central
  equation is now `min M(π,Θ) s.t. R(π,Θ) ⪯ B(H,Θ)` with the scalarized
  tropical form as its documented degenerate case. Runnable on both rails:
  `kbcir.rcsp` (budgets, label dominance, Pareto fronts — a 700 thermal cap
  makes vec16 infeasible and selects vec8 at the pinned score 9472) and
  `gem.overlap` (M(π,Θ): wave-parallel max, in-bin series chaining with true
  schedule-predecessor coupling, decoupled-tail overlap, plus the one-sweep
  select→schedule→re-price iteration). The law carries `bcir.kbcir.budget` /
  `bcir.kbcir.scheduled_price`, and `-bcir-verify` enforces budget feasibility
  and price consistency under R9.
- **Compiled `bcir-opt`** (`mlir/lib/BCIRDialect.cpp` + `mlir/tools/bcir-opt.cpp`):
  the dialect builds and the *pretty* ODS corpus parses/verifies/FileCheck-round-trips
  through it on LLVM 18 (CI `mlir-rail-validate`).
- **Verifier laws R1–R12 complete on both rails** (Phase 10): the oracle runs the
  full chain (`verify` R1–R7, `verify_plan` R8–R9, `verify_pack` R10–R11,
  `verify_lowering` R12 — the lowered kernel is checked against the selected
  lane geometry, bounds guard, hazard fence, precision, and a no-invented-
  instructions whitelist); the MLIR `-bcir-verify` pass enforces all twelve
  structurally, negative-tested per law (`mlir/test/passes/verify_laws*.mlir`).
- **Real ML calibrator** — `kbcir.calibrate.LinearCalibrator`, an online linear-model
  SGD that learns to predict thermal pressure from telemetry features (behind the
  same interface as `EwmaCalibrator`).
- **Kafka `TelemetrySink`** — `telemetry.KafkaSink` (injectable producer +
  `connect()` lazy kafka-python backend).

### Still forward
C++ ports of the five declared GEM passes (`bcir-classify-lanes`,
`bcir-select-realization`, `bcir-batch`, `bcir-schedule`, `bcir-lower-to-llvm`)
mirroring the oracle; BCIR-native instruction selection + register allocation +
linking behind the `bcir.target.lower_contract` seam (GPU via a
PTX/`gpu`-dialect path); a native (C-runtime) microbench backend filling the
same frozen-table schema with bare-metal numbers; L2 portfolio entries learned
offline (e.g. Bayesian optimization) on real telemetry, deployed through the
replay gate; a live Kafka broker deployment; automating the L3 boundary flip
behind the replay gate + R13 (deliberately deferred: measured, certified,
human-actuated — LangRef §13).

## Non-regression rules

Determinism (integer/Q-fixed only) · back-compat (`HProfile = TargetProfile`
alias + factories) · no invented LLVM instructions · atomics never rewritten into
load/op/store pseudo-atomics · IRDL rail stays C++-free.
