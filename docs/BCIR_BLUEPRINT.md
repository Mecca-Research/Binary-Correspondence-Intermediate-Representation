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
- **Compiled `bcir-opt`** (`mlir/lib/BCIRDialect.cpp` + `mlir/tools/bcir-opt.cpp`):
  the dialect builds and the *pretty* ODS corpus parses/verifies/FileCheck-round-trips
  through it on LLVM 18 (CI `mlir-rail-validate`).
- **Real ML calibrator** — `kbcir.calibrate.LinearCalibrator`, an online linear-model
  SGD that learns to predict thermal pressure from telemetry features (behind the
  same interface as `EwmaCalibrator`).
- **Kafka `TelemetrySink`** — `telemetry.KafkaSink` (injectable producer +
  `connect()` lazy kafka-python backend).

### Still forward
Per-target `bcir.target.lower_contract` codegen (ARM/RISC-V via clang
cross-targets; GPU via a PTX/`gpu`-dialect path) using the compiled dialect; a
trained ML model behind `LinearCalibrator`; a live Kafka broker deployment.

## Non-regression rules

Determinism (integer/Q-fixed only) · back-compat (`HProfile = TargetProfile`
alias + factories) · no invented LLVM instructions · atomics never rewritten into
load/op/store pseudo-atomics · IRDL rail stays C++-free.
