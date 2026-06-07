# BCIR Build Blueprint v0.2 — Target-Open, Heterogeneous, AI-Guided

Companion to [`BCIR_LANGREF.md`](BCIR_LANGREF.md) (the law). BCIR is a **container
for any target** — x86, ARM, RISC-V, GPU — so hardware-specific compilers/drivers
can later be built from firmware/ISA/opcode/registry tables. Like WASM, one
portable BCIR-4 StreamPack feeds both **AOT** and **JIT** kernels; the optimizer
that picks realizations is AI-guidable (CT4).

## Capability tracks

| Track | Capability | Status |
|---|---|---|
| **CT1** | Target-open container + memory hierarchy (HBM / HAM / CXL) | oracle done; law authored |
| **CT2** | Mixed-stride concurrent graph exec + cache unroll + thread→cache affinity | planned |
| **CT3** | ROP & MAP performance paradigms (front-ends → claims) | planned |
| **CT4** | ML-guided "data DNA" telemetry loop (thermal/voltage → exec mgmt) | interface/schema only |
| **CT5** | AOT + JIT backends per target (WASM-like agnosticism) | AOT (clang) done; JIT planned |

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

## Forward tracks (CT2–CT5) and the M5+ transduction layer

- **CT2** async dependency tokens (co-schedule U/UX/T with the GGG tail
  decoupled), cache-unroll rewrites, thread→cache affinity (cache-thrash in the
  `contention` dim).
- **CT3** ROP & MAP front-ends → claims, reusing `bcir/verify` + GEM.
- **CT4** "data DNA" telemetry schema (cycles/bytes/misses/thermal/**voltage**/
  utilization + provenance), a pluggable ML cost-calibrator, adaptive policy;
  Kafka is the transport (interface only now). Θ already carries a `voltage` axis.
- **CT5** in-process JIT over the same StreamPack; per-target
  `bcir.target.lower_contract`; ARM/RISC-V via clang cross-targets; GPU via a
  PTX/`gpu`-dialect path once the MLIR C++ dialect library lands.
- **M5 Event Transduction Layer** (next organ): `bcir.event.*`, `bcir.fsm.*`,
  `bcir.parse.*`, `bcir.binary.*` make text/binary/packet/telemetry ingestion all
  instances of the same correspondence machinery — *parser generator = grammar →
  event machine; GEM-E = event machine → streams → target*.

## Non-regression rules

Determinism (integer/Q-fixed only) · back-compat (`HProfile = TargetProfile`
alias + factories) · no invented LLVM instructions · atomics never rewritten into
load/op/store pseudo-atomics · IRDL rail stays C++-free.
