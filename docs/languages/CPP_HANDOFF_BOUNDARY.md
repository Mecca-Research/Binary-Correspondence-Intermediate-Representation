# The C ↔ C++ hand-off boundary (the contract) and the seam that holds it

> **Status: the contract, realized (G16, staged plan S3-C).** This document defines the
> boundary. The seam under [`runtime/cpp/`](../../runtime/cpp) holds it at run time over the
> freestanding C pack table ([`BCIR_DATA_PLANE_HANDOFF.md`](../kernel/BCIR_DATA_PLANE_HANDOFF.md)),
> and [`tools/cpp/check_handoff.sh`](../../tools/cpp/check_handoff.sh) gates it (wired into
> [`tools/c/check_runtime.sh`](../../tools/c/check_runtime.sh)). It addresses Pillar 5d of
> [`VISION_ALIGNMENT_AUDIT.md`](../VISION_ALIGNMENT_AUDIT.md), and it frames the boundary in the
> L0–L3 / two-truth placement law of [`BCIR_LANGREF.md`](../BCIR_LANGREF.md) §13 and the
> invariants in [`BCIR_MASTER_ROADMAP.md`](../BCIR_MASTER_ROADMAP.md) §1.

## Honest depth (read this first)

> **Real, and gated:**
>
> - the artifact crossing as a **borrowed view with an explicit lifetime** (`PackArena`,
>   `Reservation`, `PackOwner`, `PackView`, `Borrow` over the C pack table);
> - **admission against the live control plane** (`admit()` is not virtual);
> - **dispatch** of what was admitted at the resident generation, as a plane phase;
> - the `SingleNodeOrchestrator`;
> - the **dynamic-graph backend**: `GraphBuilder` freezes a fresh StreamPack per step through the
>   C/IR rail, straight into an arena slot;
> - the distributed backend's **partition and manifest-of-shards**: every shard is a pack a node
>   admits and runs by itself, and the set reassembles to the whole by digest.
>
> **Stub (documented, marked, fails loudly):** the distributed backend's **cross-node
> dispatch**. It needs MPI/NCCL and a cluster we deliberately do not add; dispatching it throws
> `HandoffError`, so dead code never masquerades as working. There is **no MPI/NCCL dependency**,
> and **no dynamic or distributed logic on the deterministic legality path**.

---

## Why a boundary at all

BCIR's deterministic rail is **single-node by design**. Its whole value proposition rests on a
*statically known, bounded, deterministic* graph that freezes to a self-contained artifact: the
R1–R25 laws, the provenance digest, the byte-identical Python↔C parity and the two-truth
quarantine. The artifact is the StreamPack, BCIR's "WASM analog" (see
[`BCIR_STREAMPACK_ABI.md`](../kernel/BCIR_STREAMPACK_ABI.md)). Two classes of real ML work break
that premise:

1. **Dynamic graph topology**: nodes and edges created *at runtime*. Examples are an RL agent
   spawning network nodes as it explores, and a transformer allocating mixed-length token graphs
   on the fly.
2. **Distributed multi-node orchestration**: MPI/NCCL collectives, networking, failure recovery,
   job queuing — a *software stack above* the kernels.

Neither belongs on the deterministic rail. Both are the **"high-level software abstraction that
consumes frozen artifacts"** that the L0–L3 placement law puts in **C++ above** the rail. This
document defines that boundary, so the single-node limit is explicit, contracted and seamed.

## What stays on the C/IR rail (below the boundary)

- **Single-node, statically bounded graphs**, deterministic inference and the bounded training
  slice.
- **All R-laws and provenance.** R1–R25, the R13 provenance digest, R10/R11 StreamPack semantics:
  the path that bears the verdict.
- **The frozen artifact's production.** The C/IR rail emits the StreamPack: `bcir_encode.c` /
  `bcir/abi`, and, per dynamic step, `bcir_hydrate_generations`.
- **The data plane's decisions.** The pack table (`bcir_handoff.h`) decides every reservation,
  borrow, release, admission and dispatch, and the control plane (`bcir_control_plane.h`) decides
  every admission's legality. Both are freestanding C twins of the Python oracles, byte for byte.

## What crosses to C++ (above the boundary)

- **Dynamic graph topology.** The `GraphBuilder` is a mutable, RAII-managed C++ object. Each step
  it freezes a **fresh** claim graph into a StreamPack through the C/IR rail. What crosses down is
  always an immutable artifact.
- **Distributed orchestration.** Partition, cut, placement, retry and replication decisions
  across nodes; each node runs the single-node C/IR rail.
- **Lifetime as types.** The rules the table enforces at run time (write once, borrow, release,
  refuse the dead) are also the *shape of the C++ code*: move-only owners, weak views, RAII pins.

This is the **L2/L3 layer**. It schedules, shards, retries and replicates. It never computes a
verdict and never alters a frozen artifact.

---

## The seam

### The artifact, and how it crosses

The unit that crosses is the **frozen StreamPack**. It is self-contained, CRC-sealed and
semantically verifiable. It crosses as bytes **written once into an arena slot and read in place
by every consumer**:

| Type | Role |
|---|---|
| `PackArena` | Owns the table's slots and arena; a `shared_ptr`, so views hold it weakly. One mutex serializes every table operation, so an arena may be shared across threads. A borrowed view is read outside the mutex. |
| `Reservation` | A producer's slot. It is written **once** through `data()`/`write()`, then `commit()` produces a `PackOwner`, or `abort()` drops it. Destroyed uncommitted, it aborts, so nothing half-built is ever published. |
| `PackOwner` | Move-only owner of one frozen artifact. `release()`, or the destructor, ends the incarnation: every view of it is refused from then on with `BCIR_ERR_LIFETIME`. |
| `PackView` | Copyable and non-owning. It may outlive its owner or its arena; then every borrow is **refused**. It never reads freed or reused bytes. |
| `Borrow` | A move-only RAII pin. `data()`/`size()` stay valid for its lifetime, even if the owner releases meanwhile (the slot retires, and its last borrow frees it). It is returned exactly once, by `give_back()` or the destructor. |

A C++ handle cannot be forged, and a borrow cannot be copied. So the C harness's four
C-only operations have no C++ spelling: a forged handle, a poked epoch or pin count, and a copied
borrow. Those laws are graded on the Python and C rails. The C++ seam gets its own witnesses
instead: a view after release, after its owner's scope, or after its arena; a reused slot's old
view; a borrow that outlives the release; a reservation that aborts; a moved owner; a double
return; a foreign arena; and bytes moved exactly once.

### The data contract: what C++ MAY and MAY NOT do

It **MAY**: schedule, partition the segment stream, cut shards, place shards on nodes, retry a
transient fault, replicate a shard, and reduce per-shard results.

It **MAY NEVER**:

- **alter a frozen artifact's bytes or semantics**. No API writes a committed slot. A pack is
  immutable until it is released, and after release it is unreachable;
- **become an R-law verdict**. `Orchestrator::admit()` is *not virtual*. It is the live plane's
  one predicate: `bcir_ctl_admit_pack` against the installed registry, through the table. No
  backend can override legality, and **no caller-supplied generation numbers** exist anywhere in
  the seam. A refusal (stale, malformed, lifetime, unadmitted, draining) is a `bcir_status` in a
  `HandoffResult`, carried up. An orchestration failure (an unbuilt transport, exhausted retries)
  is a `HandoffError` exception, never a verdict.

**The two-truth quarantine extends across the boundary.** Below the line is the deterministic
verdict. Above it are graded placement and retry decisions, and the two never mix. The artifact is
the airlock: a verdict travels *up* inside a result, and an orchestration decision never travels
*down* into the artifact's bytes.

### Generation gating

The resident plane holds the admission, not the caller. `admit()` records the resident
generation **and the registry digest** it admitted at. `dispatch()` runs the artifact only if both
still hold, and it runs it as a phase of the plane, so a switch requested mid-dispatch lands at
the phase boundary. A switch between admit and dispatch makes the artifact stale; re-admission
refuses it, and that refusal revokes the admission. A plane that restarts and reaches the same
generation number under another registry is caught by the registry binding.

### Re-entry

After placement, the orchestrator calls back into the C/IR single-node kernels: that call is the
*re-entry*. `dispatch()` walks the admitted slot's segments **in place**
(`bcir_sp_for_each_segment` over the borrowed bytes; the gate counts the segment views that point
outside the slot, and the count is 0). In a distributed run, each rank admits its own shard and
runs its own `SingleNodeOrchestrator`. The rail stays the authority, and C++ only dispatches.

### Diagram

```
                         ABOVE THE LINE  —  C++  (L2/L3, graded: schedule/shard/retry)
   ┌───────────────────────────────────────────────────────────────────────────────┐
   │ Orchestrator: admit(view, plane) [not virtual] · shard(view) · dispatch(view)  │
   │   SingleNode (REAL)   DynamicGraph (REAL: step = freeze→admit→run→release)     │
   │   Distributed (REAL partition + cut() manifest-of-shards; dispatch STUB)        │
   │ PackArena ─ Reservation ─▶ PackOwner ─▶ PackView ─▶ Borrow   (RAII lifetimes)   │
   └──────────────┬───────────────────────▲───────────────────────▲─────────────────┘
      written     │  re-entry in place     │ verdict (bcir_status) │ frozen artifact:
      ONCE into   │  (borrowed bytes)      │ carried UP            │ rail-produced,
      a slot      ▼                        │                       │ per step or whole
   ┌──────────────────────────────────────┴───────────────────────┴─────────────────┐
   │        BELOW THE LINE  —  C / IR  (L0/L1, deterministic: the authority)           │
   │  bcir_ho_* pack table (epochs, pins, admission record)  bcir_ctl_* live plane     │
   │  bcir_hydrate_generations (per-step freeze)  bcir_shm_* manifest-of-shards        │
   │  bcir_sp_for_each_segment (kernels)   byte-identical to the Python oracles        │
   └──────────────────────────────────────────────────────────────────────────────────┘
```

### Failure, retry and idempotence

- **Idempotence.** The bytes are immutable and the walk is deterministic, so a re-dispatch
  recomputes the identical result.
- **Refusals are not retried.** A dead view, a missing admission at the resident generation or a
  draining plane is a *verdict*. Retrying the same immutable bytes cannot change it.
- **Retry is for transient faults.** The contract a real distributed backend uses is to re-place
  the shard on a healthy node, then re-dispatch the same immutable artifact.
- **Replication** is a placement decision: dispatch the same shard to N nodes and take the first
  success. That is safe because the artifact is immutable and the result deterministic.

---

## Why C++ (and not C / IR)

Dynamic topology and distributed stacks need what the flat, freestanding C rail deliberately
lacks:

- OO and virtual dispatch: a backend hierarchy selected at runtime;
- the STL and ownership types: a graph that grows at runtime, and shard sets;
- exceptions and RAII: failure recovery, and a lifetime that the type system enforces.

The integer kernels and every decision stay in C. The allocating, exception-handling
orchestration stack goes to C++ above them.

---

## What is built

| File | Role |
|---|---|
| [`runtime/c/bcir_handoff.h`](../../runtime/c/bcir_handoff.h) / `.c` | The freestanding pack table: epochs, pins, retirement, the admission record (generation and registry), dispatch as a plane phase, the manifest gate, the state digest. |
| [`runtime/c/bcir_shard_manifest.h`](../../runtime/c/bcir_shard_manifest.h) / `.c` | The manifest-of-shards ([`BCIR_SHARD_MANIFEST_ABI.md`](../kernel/BCIR_SHARD_MANIFEST_ABI.md)). |
| `bcir_hydrate_generations` in [`runtime/c/bcir_hydrate.h`](../../runtime/c/bcir_hydrate.h) | The per-step freeze: a v4 pack bound to the live registry. |
| [`runtime/cpp/bcir_handoff.hpp`](../../runtime/cpp/bcir_handoff.hpp) / `.cpp` | The RAII types, `admit_view`/`dispatch_view`, `GraphBuilder`, `cut_shards`/`reassemble_shards`. |
| [`runtime/cpp/bcir_orchestrator.hpp`](../../runtime/cpp/bcir_orchestrator.hpp) / `.cpp` | `Orchestrator` (non-virtual `admit`), the single-node and dynamic-graph backends, the distributed partition and `cut()`, the factory. |
| [`runtime/cpp/test_orchestrator.cpp`](../../runtime/cpp/test_orchestrator.cpp) | The contract on real artifacts from the C/IR path and the plane's records, minted by `handoff_fixtures.seam_artifacts`, the one minter the gate and `test_cpp_handoff.py` share. It checks that dispatch equals the direct C walk, shards run by themselves and reassemble, a builder step freezes and runs, a switch makes the artifact stale, and a dead view is refused. `--reject` refuses a corrupted artifact at admission, and the same probe admits the clean one. |
| [`runtime/cpp/test_handoff.cpp`](../../runtime/cpp/test_handoff.cpp) | The C++ rail of the G16 rows: every scenario it can spell, traced line for line against the oracle; shard runs and re-entry; the lifetime witnesses; the Stage 3 exit flow through the seam (`--stage3`); the overhead benchmark. |
| [`tools/cpp/check_handoff.sh`](../../tools/cpp/check_handoff.sh) | The gate: the standalone C++17 build under `-Wpedantic -Werror`; the contract on real artifacts; a corrupted artifact refused at admission; every row at zero; a double-return mutant that must fire; everything again under ASan and UBSan; the BCAB wrapper. |

The seam is **plain C++17, standalone**. It links only the freestanding C units and is **not**
part of the MLIR/LLVM cmake.

---

## Risks and follow-ups (what remains, deliberately)

- **A real MPI/NCCL backend.** Collectives, a communicator and rank model, a job queue, a thread
  pool. The partition, the shards, the manifest, per-rank admission and re-entry are real and
  gated. Only the transport and the cross-rank reduction are owed, and they need a cluster and a
  dependency not added here.
- **Failure recovery and replication policy.** Heartbeats, node-death detection, re-placement.
  The idempotence contract is specified; a real implementation wires it to cluster membership.
- **A shared-memory pack table.** The table lives in one address space. A table shared across
  processes (the live ring's model) needs its own ordering argument, and is a later slice.
