# BCIR Language Reference — v0.1 (normative)

BCIR is the IR *law*; MLIR is the forge used to express, verify, transform, and
lower that law during bootstrap; LLVM/Clang are backends and interoperability
targets, **not** the conceptual center.

> **Source-of-truth rule.** BCIR semantics live in this document and in the
> dialect definitions under `mlir/`. The Python package under `bcir/` is the
> **executable conformance oracle** — it must agree with this document, never
> override it (see [`PARITY.md`](PARITY.md)). C++ may implement the engine; it may
> not become the definition of BCIR.

## 0. Stance

BCIR is a multi-level IR specification:
`syntax · types · attributes · operations · verification laws · rewrite laws ·
cost laws · execution laws · lowering contracts`. It is a **correspondence IR**:
it preserves the chain from semantic intent to physical execution and asks not
merely whether a program is structurally valid but whether the computation is
physically, temporally, and contractually **executable**.

## 1. The multi-level IR

| Level | Name | Question |
|---|---|---|
| BCIR-0 | Semantic Claim Graph | What computation/state transformation is intended? |
| BCIR-1 | Shaped Data Graph | What tensors/columns/buffers/sparse maps/records/layouts exist? |
| BCIR-2 | Registry / Placement Candidate Graph | Where can resources live; what domain constraints apply? |
| BCIR-3 | K_BCIR Correspondence Plan | Which realization path π is selected under H and Θ? |
| BCIR-4 | GEM Stream IR | What lane schedule, StreamPack, fences, prefetch contracts execute? |
| BCIR-5 | Target Lowering IR | LLVM / vector / GPU / SPIR-V / PTX / WASM / future BCIR-native binary. |

`K_BCIR(G | H, Θ) = min_π Σ_i T_i ⊗ f_i(π)` maps BCIR-2 → BCIR-3: a shortest-path
search over representation and machine-state space, not a one-shot
source→binary pipeline.

## 2. Central equation

```
K_BCIR(G | H, Θ) = min_π  Σ_i  T_i ⊗ f_i(π)  =  min_π  C_H(π, Θ)
```

- `G` — the goal graph (a BCIR program).
- `π` — a realization plan (lane/stride class, batching, schedule, prefetch).
- `T_i` — base cost of a primitive (a 12-d cost vector).
- `f_i(π)` — path-context coupling (placement, fusion, precision, thermal).
- `⊗` — element-wise coupling over cost dimensions (a resource tensor), **not**
  scalar multiply.
- selection — `score = w(H,Θ,phase,policy) · (T_i ⊗ f_i(π))`, minimized by a
  tropical (min,+) shortest path over a realization DAG of **legal** candidates.

## 3–9. Laws (summary)

- **Module law (§3).** A module is a registry-governed execution universe;
  registries precede claims; plans derive from legal claims; a GEM stream may not
  exist without an originating BCIR plan.
- **Registry-first memory (§4).** Raw pointers are outlawed at the core level.
  `Address = (RID, layout, domain, offset, generation)`.
- **Claim law (§5).** The primitive object is the *claim*, not the instruction:
  `op + resources + contract + phase + cost + verification + ≥1 legal realization`.
- **Phase DAG (§6).** Execution order is a phase graph (acyclic), not textual
  order.
- **Lane law (§7).** Lanes are execution-geometry types: `U` unit/stride, `UX`
  cacheline-local, `T` tile, `GGG` gather/scatter (always legal, must be
  minimized), `A` atomic, `H` hazard/provenance.
- **K_BCIR cost (§8).** Cost is *in the IR* — a 12-d `costvec`
  (compute, memory, fabric, sync, compile, thermal, power, reliability, security,
  accuracy, contention, verification). Illegal paths are rejected before scoring;
  Pareto pruning precedes scalar selection; the selected path hydrates GEM.
- **GEM Stream IR (§9).** The StreamPack is the hot artifact; the BCIR graph is
  the dormant semantic artifact. A pack retains provenance and generation tags
  and is rehydrated (patch/repack/replan) on mismatch.

## 10. Verifier laws (R1–R12)

R1 registry uniqueness · R2 registry resolution · R3 domain legality ·
R4 phase-DAG legality · R5 hazard legality · R6 lane legality · R7 bounds
legality · R8 cost completeness · R9 plan legality · R10 stream provenance ·
R11 generation validity · R12 lowering legality. Encoded as IR via the
`bcir.verify.*` op family; the runnable subset lives in `bcir/verify`.

## 11. Rewrite laws

Lane promotion (`GGG→UX→U(k)→U`), tile formation, layout (`AoS→SoA→AoSoA`),
prefetch introduction, GGG quarantine. A rewrite is legal **only if** it does not
increase the selected K_BCIR cost (or strictly improves legality) and the module
still passes R1–R12. Encoded via `bcir.opt.*`.

## 12. Lowering contracts

BCIR-4 → BCIR-5 lowering is governed by R12: each lowered op preserves the BCIR
semantic (lane geometry, bounds, hazard, precision) or carries an explicit
discharge in `bcir.trace`. LLVM is the **first** backend, not the center.
Encoded via `bcir.isa.*` / `bcir.target.lower_contract`.

## 15. Milestone map

1. LangRef v0.1 — this document. ✔
2. Declarative dialect definitions — `mlir/include/BCIR/*.td`, `mlir/passes/*.td`. ◑ (authored; build pending toolchain)
3. Verifier-first compiler (`bcir-verify-*`). ○
4. Rewrite laws. ◑ (law-as-IR authored)
5. K_BCIR planner — candidate-path/costvec/selected-path IR. ◑ (runnable in `bcir/`)
6. GEM hydration — GraphPlan/LanePlan/StreamPack IR. ◑ (runnable in `bcir/`)
7. LLVM as first backend. ◑ (oracle AOT path runs via clang)

Until the MLIR toolchain exists on this host, the oracle (`bcir/`, runnable via
`python -m bcir.run`) demonstrates Milestones 5–7 in miniature and is the
conformance reference for the dialects.

## 16. Thesis

> BCIR is a registry-first, phase-ordered, lane-typed, cost-governed
> correspondence IR. K_BCIR is the IR-level optimization calculus that selects
> legal physical realization paths. GEM is the execution IR that hydrates
> selected correspondence paths into streamed lane schedules. MLIR is the
> bootstrap framework used to define, verify, rewrite, and lower BCIR until BCIR
> has enough mass to become its own compiler toolchain.
