# The C ↔ C++ hand-off boundary (the contract) + a compilable seam scaffold

> **Status: design-first contract + a minimal, compilable, tested seam scaffold.**
> This document defines the boundary precisely; the seam is exercised by a real,
> standalone C++17 scaffold under [`runtime/cpp/`](../../runtime/cpp) with a round-trip
> smoke test gated by [`tools/cpp/check_handoff.sh`](../../tools/cpp/check_handoff.sh)
> (wired into [`tools/c/check_runtime.sh`](../../tools/c/check_runtime.sh)). It addresses
> Pillar 5d of [`VISION_ALIGNMENT_AUDIT.md`](../VISION_ALIGNMENT_AUDIT.md) and frames the
> boundary in the L0–L3 / two-truth placement law of
> [`BCIR_LANGREF.md`](../BCIR_LANGREF.md) §13 and the invariants in
> [`BCIR_MASTER_ROADMAP.md`](../BCIR_MASTER_ROADMAP.md) §1.

## Honest depth (read this first)

> **The deliverable is the *designed contract* + a *compilable seam with a single-node
> reference implementation and a round-trip smoke test* — NOT a real MPI/NCCL cluster
> integration.** A real distributed/dynamic backend needs multi-node hardware we do not
> have and would be untested debt. So:
>
> - **REAL (compiles + runs + is gated):** the `Orchestrator` interface, the
>   `SingleNodeOrchestrator` reference (it consumes the artifact and re-enters the
>   existing freestanding C kernels), the round-trip identity smoke test, and the
>   *sharding logic* of the distributed backend (a pure segment-stream partition that
>   needs no cluster).
> - **STUB (documented, marked, fails loudly):** the `DynamicGraphOrchestrator` and
>   `DistributedOrchestrator` *dispatch* paths. They describe exactly what a real
>   implementation would do but are not built; dispatching them throws a `HandoffError`,
>   so dead code can never silently masquerade as working. There is **no real MPI/NCCL
>   dependency** and **no dynamic/distributed logic on the deterministic legality path**.

---

## Why a boundary at all

BCIR's deterministic rail is **single-node by design**. Its whole value proposition —
the R1–R23 laws, the provenance digest, the byte-identical Python↔C parity, the
two-truth quarantine — rests on a *statically known, bounded, deterministic* graph that
freezes to a self-contained artifact (the StreamPack, BCIR's "WASM analog"; see
[`BCIR_STREAMPACK_ABI.md`](../kernel/BCIR_STREAMPACK_ABI.md)). Two whole classes of real ML work
break that premise:

1. **Dynamic graph topology** — nodes/edges created *at runtime*: an RL agent spawning
   network nodes as it explores; a transformer allocating mixed-length token graphs on
   the fly. The graph shape is not known when the rail is built.
2. **Distributed multi-node orchestration** — the work is too big for one node: MPI/NCCL
   collectives, networking, failure recovery, job queuing, thread pooling. This is a
   *software stack above* the kernels, not a kernel.

Neither belongs on the deterministic rail. Both are exactly the **"high-level software
abstraction that consumes frozen artifacts"** the L0–L3 placement law (§3 of the master
roadmap, the *one-line rule*) puts in **C++ ABOVE** the rail. This document defines that
above-the-rail boundary so the single-node limit is **explicit, contracted, and seamed**
rather than an undocumented hole.

---

## What STAYS on the C/IR rail (below the boundary)

Everything the rail is already good at — unchanged:

- **Single-node, statically-bounded / known-shape graphs.** A claim graph whose nodes,
  edges, and shapes are fixed at build time.
- **Deterministic inference (G5)** and the **bounded training slice (G6)** — the forward
  pass and the bounded backward pass that lower to the freestanding C kernels.
- **All R-laws + provenance.** R1–R23 verification, the R13 provenance digest, R10/R11
  StreamPack semantics, the R17 Q8↔f32↔Q8 bridge — the verdict-bearing legality path.
- **The frozen artifact production.** The C/IR rail *emits* the StreamPack (via
  `bcir_encode.c` / `bcir/abi`) — the serialized, CRC-sealed, self-contained executable.

This is the **L0/L1 deterministic core** in two-truth terms: integer/Q-fixed, on the
decision/execution path, in C (`runtime/c/`) or MLIR/C++ law.

## What CROSSES to C++ (above the boundary)

- **Dynamic graph topology** — runtime node/edge creation (RL node spawning,
  dynamic/mixed-length token graphs). The *graph builder* is a mutable, OO, RAII-managed
  C++ object; each step it materializes a **fresh** claim graph and **freezes it to a
  StreamPack** via the C/IR rail. What crosses *down* is always an immutable artifact.
- **Distributed multi-node orchestration** — MPI/NCCL, networking, failure recovery, job
  queue, thread pool. The C++ orchestrator decides placement/topology across nodes and
  dispatches shards; each node runs the existing single-node C/IR rail.

This is the **L2/L3 high-level layer**: it *schedules, shards, retries, replicates* — it
never computes a verdict and never alters a frozen artifact.

---

## The seam

### The artifact

The unit that crosses the boundary is the **frozen StreamPack** (or, equivalently, a
serialized claim-graph manifest — the same shape the
[`channel.json`](../../bcir/channel_plugin.py) plugin seam uses to describe a backend). It
is the natural hand-off because it is already **self-contained, serialized, CRC-sealed,
and semantically verifiable** ([`BCIR_STREAMPACK_ABI.md`](../kernel/BCIR_STREAMPACK_ABI.md)). The
C/IR rail produces it; the C++ orchestrator consumes it.

The C++ orchestrator is the **multi-node generalization of the existing
[heterogeneous-channel](../kernel/HETEROGENEOUS_CHANNELS.md) routing seam**: where `route_claim`
routes a claim to a *backend* on one node, the orchestrator routes/shards work across
*nodes*, each node running the existing single-node C/IR rail.

### The data contract — what C++ MAY and MAY NOT do

> The C++ side receives the artifact as an **immutable byte view** (read access, never
> write access).

It **MAY**: schedule, shard (partition the segment stream), place shards on nodes, retry
a failed shard, replicate a shard for fault tolerance, reduce per-shard results.

It **MAY NEVER**:

- **alter a frozen artifact's bytes or semantics** — the StreamPack is CRC-sealed; a
  mutation breaks the CRC and is rejected by the C decoder at the boundary;
- **become an R-law verdict.** Legality is the C/IR rail's verdict. The orchestrator
  *asks* the authority (`admit()` delegates to `bcir_sp_verify_semantic`) and *carries*
  the verdict across — it does **not** re-derive legality. An orchestration failure (a
  node died, a retry exhausted, a backend unbuilt) is an **operational fault** surfaced
  as a C++ exception (`HandoffError`), never a legality verdict.

**The two-truth quarantine extends across the boundary.** Below the line: the
deterministic L0/L1 verdict (a `bcir_status` from the C verifier). Above the line: the
graded L2/L3 placement/retry decisions. They never mix — a placement decision can no more
become an R-law verdict than a learned organ can write the legality path. The artifact is
the airlock: a verdict travels *up* inside the artifact's status; an orchestration
decision never travels *down* into the artifact's bytes.

### Re-entry

After the C++ orchestrator decides placement, it **calls back into the C/IR single-node
kernels per shard / per node** — the *re-entry*. In the scaffold this is
`bcir_sp_for_each_segment` (the existing freestanding C decoder in
[`bcir_runtime.c`](../../runtime/c/bcir_runtime.c)); in a real distributed run each node
runs its own `SingleNodeOrchestrator` over its shard. The rail stays the authority; C++ is
only the dispatcher.

### Diagram

```
                         ABOVE THE LINE  —  C++  (L2/L3, graded: schedule/shard/retry)
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │   Orchestrator (abstract)                                                       │
   │     • admit(artifact)  ── delegates legality to the C/IR verifier (no re-derive)│
   │     • shard(artifact)  ── placement/topology decision (read-only artifact)      │
   │     • dispatch(...)    ── re-enter the C kernels per shard; retry/replicate      │
   │                                                                                 │
   │   SingleNodeOrchestrator (REAL)   DynamicGraph (STUB)   Distributed (STUB)      │
   │        one shard, node 0          re-freeze each step    shard across ranks      │
   │             │                      → StreamPack          (sharding REAL,         │
   │             │                      → single-node          dispatch needs         │
   │             │                                              MPI/NCCL: stub)       │
   └─────────────┼───────────────────────────────────────────────────────────────────┘
                 │                ▲                              ▲
   frozen        │  re-entry      │ verdict (bcir_status)        │ frozen artifact
   StreamPack    │  (per shard)   │ carried UP, never re-derived │ produced by the rail
   (immutable)   ▼                │                              │
   ┌─────────────────────────────┴──────────────────────────────┴──────────────────┐
   │           BELOW THE LINE  —  C / IR  (L0/L1, deterministic: the authority)      │
   │   bcir_sp_verify_semantic (R10/R11 + range)   bcir_sp_for_each_segment (kernels)│
   │   single-node, known-shape, R1–R23 + provenance, byte-identical Python↔C parity │
   └────────────────────────────────────────────────────────────────────────────────┘
```

The airlock is the horizontal line: only a **frozen artifact** goes down and only a
**status verdict** comes up. Nothing graded crosses down into the bytes.

### Failure / retry / idempotence semantics

- **Idempotence.** The artifact bytes are immutable and the C walk is deterministic, so a
  re-dispatch recomputes the **identical** result. A retry never accumulates state. The
  smoke test asserts this directly (`dispatch(..., max_retries=3)` == the first run).
- **Retry.** `dispatch(data, len, max_retries)` re-attempts a failed shard up to
  `max_retries` times. The single-node reference has no transient faults, so it converges
  immediately; the loop demonstrates the contract a real distributed backend uses (re-place
  the shard on a healthy node, then re-dispatch the *same immutable artifact*).
- **Failure is operational, not legal.** A C-rail legality failure arrives as a non-OK
  `bcir_status` inside the result (carried up). An *orchestration* failure (unbuilt
  backend, exhausted retries, dead node) is a `HandoffError` exception — handled by the
  C++ layer (retry/replicate/abort), never reaching back to rewrite the artifact.
- **Replication** (a real distributed concern) is a placement decision: dispatch the same
  immutable shard to N nodes and take the first success — safe precisely because the
  artifact is immutable and the result is deterministic.

---

## Why C++ (and not C / IR)

Dynamic topology and distributed software stacks need exactly the abstractions the **flat,
registry-oriented C rail deliberately lacks**:

- **OO + virtual dispatch** — a backend hierarchy (`Orchestrator` → single-node /
  dynamic / distributed) selected at runtime by workload shape. The C rail is a flat
  registry of structs, not a class hierarchy.
- **The STL** — `std::vector` of shards, `std::unique_ptr` ownership, dynamic containers
  for a graph that grows at runtime. The freestanding C rail has no allocator and no
  containers by design (it links with `-ffreestanding -nostdlib`).
- **Exceptions + RAII** — failure recovery, resource cleanup across node failures, and the
  operational-fault channel (`HandoffError`) that is explicitly *not* a verdict. The C
  rail has no exceptions and fails by `bcir_status` return codes.

These are the "high-level software abstraction" half of the placement law: deterministic
integer kernels stay in C; the dynamic, allocating, exception-handling orchestration stack
goes to C++ above them. The boundary keeps the rail freestanding and verdict-bearing while
the complexity lives where it belongs.

---

## The scaffold (what is built)

| File | Role |
|---|---|
| [`runtime/cpp/bcir_orchestrator.hpp`](../../runtime/cpp/bcir_orchestrator.hpp) | The seam: the `Orchestrator` abstract base, the `Shard`/`DispatchResult`/`HandoffError` data contract, the single-node REAL backend + the two STUB backends, and the factory. |
| [`runtime/cpp/bcir_orchestrator.cpp`](../../runtime/cpp/bcir_orchestrator.cpp) | Implementation. `SingleNodeOrchestrator` re-enters the existing C decoder; the stubs document a real impl and throw. |
| [`runtime/cpp/test_orchestrator.cpp`](../../runtime/cpp/test_orchestrator.cpp) | The round-trip smoke driver: asserts the seam's dispatch order == the **direct** C/IR decode of the same artifact (round-trip identity), plus the admit/shard/stub contract surfaces. |
| [`tools/cpp/check_handoff.sh`](../../tools/cpp/check_handoff.sh) | Standalone build+run gate: compiles the C++17 scaffold (linked only against the freestanding C runtime), produces a real StreamPack via the existing C/IR path, round-trips it, and checks a corrupted artifact is rejected at the boundary. Wired into [`tools/c/check_runtime.sh`](../../tools/c/check_runtime.sh). |

The scaffold is **plain C++17, standalone**: it links only against
[`runtime/c/bcir_runtime.c`](../../runtime/c/bcir_runtime.c) (the existing freestanding
decoder) and is **not** part of the MLIR/LLVM cmake. The MLIR/C++ law rail under
[`mlir/`](../../mlir) is untouched.

---

## Risks / follow-ups (what a real distributed/dynamic implementation would add)

- **A real dynamic-graph builder** — a mutable C++ claim-graph object (RL node spawning /
  mixed-length token graphs) that materializes and **freezes** a fresh StreamPack each
  step via the C/IR rail. The seam already specifies the contract; the builder is the work.
- **A real MPI/NCCL backend** — collectives, a communicator/rank model, a job queue, a
  thread pool, networking. This needs a multi-node cluster + an MPI/NCCL dependency
  (deliberately not added here). The `shard()` partition logic is already real and tested;
  only the cross-node dispatch + reduce is owed.
- **Failure recovery + replication policy** — heartbeats, node-death detection,
  shard re-placement, replication factor. The retry/idempotence contract is specified and
  smoke-tested single-node; a real impl wires it to a cluster membership service.
- **Generation gating at the boundary** — `admit()` already accepts an expected
  `map_gen`/`data_gen`; a real deployment must pass the live registry generation so a
  **stale** artifact (R11) is rejected before placement.
- **A manifest variant of the artifact** — for graphs too large to ship as one StreamPack,
  a serialized claim-graph *manifest* (the same schema the `channel.json` plugin uses) that
  references shard packs by digest. The seam is artifact-shaped already; this is a format
  addition, not a contract change.
