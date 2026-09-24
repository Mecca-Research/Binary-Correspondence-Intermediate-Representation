# BCIR data-plane hand-off: the pack table, the per-step freeze and the Stage 3 exit flow

The data plane is **how a frozen artifact moves from the component that produces it to the ones
that run it** (GEM+ roadmap G16, staged plan S3-C). The component is the C/IR rail, the
dynamic-graph builder, or a node receiving shards. The artifact is **written once and read in
place**. Every read goes through a **borrowed view with an explicit lifetime**, and a view that
outlives its owner is refused, never read. **Admission** is the live control plane's predicate
against the installed registry. **Dispatch** runs only what was admitted at the resident
generation of that registry, as a phase of the plane.

Before this slice, the C++ seam took a raw `(pointer, length)` pair. Measured on the parent
(1ee34676, `admit(data, len)` as its callers used it):

- `admit()` gated nothing by default: 9 of 9 stale packs were admitted. Handed the live maxima,
  it compared header numbers and still admitted 8 of 9.
- `dispatch()` consulted no admission, so every stale pack ran.
- A view into a reused buffer dispatched the next step's bytes without any error. A view into a
  freed buffer read freed memory (ASan).
- The dynamic-graph backend was a stub, and no manifest-of-shards existed.

The executable oracle is [`bcir/gem/handoff.py`](../../bcir/gem/handoff.py), which holds
`PackTable`, `freeze_claims` and `admit_manifest`. The production rail is the freestanding C twin
[`runtime/c/bcir_handoff.h`](../../runtime/c/bcir_handoff.h) / `bcir_handoff.c` (no heap, no
libc, the driver memory class: handles plus offsets into the caller's storage), together with
`bcir_hydrate_generations` in [`bcir_hydrate.h`](../../runtime/c/bcir_hydrate.h). The C++ RAII
seam over it is [`runtime/cpp/bcir_handoff.hpp`](../../runtime/cpp/bcir_handoff.hpp), and its
contract is [`CPP_HANDOFF_BOUNDARY.md`](../languages/CPP_HANDOFF_BOUNDARY.md). Shards have their
own specification, [`BCIR_SHARD_MANIFEST_ABI.md`](BCIR_SHARD_MANIFEST_ABI.md). The three rails
decide every operation identically: the outcome, the handle, the resident generation, and the
table's and plane's state digests after every step.

## The pack table

A table owns `n_slots` (1..65535) fixed-size slots of one arena, both in the caller's storage.
A **handle** is `(slot index, epoch)`. Epochs start at 1, so the zeroed handle `{0, 0}` is never
live, and they **never wrap**.

| State | Meaning |
|---|---|
| `free` | available to `reserve` |
| `building` | reserved: the producer is writing through `reserved()` |
| `frozen` | committed: immutable and borrowable |
| `retired` | released while pinned; the last give-back frees it |
| `exhausted` | released at the last epoch; out of service for good |

Operations:

- **Producer.** `reserve` takes the lowest free slot. The producer writes the artifact **once**
  through `reserved()`, which returns the write region only for a building slot's handle.
  `commit(length)` freezes the first `length` bytes, where `1 ≤ length ≤ capacity`. `abort` frees
  the slot with nothing published and advances its epoch. No API writes a frozen slot again.
- **Consumers.** `borrow` pins a frozen slot at the handle's epoch and hands back its bytes.
  `give_back` returns a borrow: to a frozen slot of its epoch, or to one retired or exhausted from
  that epoch. The table refuses a give-back it can prove wrong: no pin, another incarnation, or
  the zeroed view. `release` ends the incarnation by advancing the epoch, so every handle and view
  of it is refused from then on with `BCIR_ERR_LIFETIME`. A pinned slot **retires** instead of
  freeing, so a borrow in progress keeps its bytes and a slot is never reused under a reader.
- **Plane.**
  - `admit` borrows the artifact, asks the **live** plane's one predicate
    (`bcir_ctl_admit_pack` against the installed registry, the R11 law of
    [`BCIR_CONTROL_PLANE_ABI.md`](BCIR_CONTROL_PLANE_ABI.md)), and gives the borrow back. When the
    plane applies it, the slot records the resident generation **and the registry digest** it was
    admitted at. A refusal revokes any earlier admission.
  - `dispatch` borrows the artifact. It is refused unless the slot was admitted in this
    incarnation, at the plane's resident generation, **of the same registry**. It then enters a
    plane phase (`bcir_ctl_enter`, refused while draining), walks the segments in place
    (`bcir_sp_for_each_segment`), and leaves the phase (`bcir_ctl_leave`). A switch requested
    mid-dispatch lands at that boundary, once.

The registry binding matters because the generation number alone does not name a registry. A
plane that restarts reaches generation 1 again, possibly under another registry. The scenario
`admission/restart-other-registry` witnesses the refusal on all three rails, and
`admission/restart-same-registry` witnesses that a restart under the same registry still admits.
So the binding is exactly (generation, registry), never the plane's identity.

**Refusals** (wire code = index): `none`, `lifetime`, `full`, `capacity`, `stale`,
`unadmitted`, `malformed`, `draining`, `exhausted`, `walk`. Their statuses are:

| Refusal | Status |
|---|---|
| `lifetime` | `BCIR_ERR_LIFETIME` (23, appended by S3-C) |
| `full` | `BCIR_ERR_FULL` |
| `capacity` | `BCIR_ERR_NOSPACE` |
| `stale`, `unadmitted` | `BCIR_ERR_STALE` |
| `draining` | `BCIR_ERR_CONTROL` |
| `exhausted` | `BCIR_ERR_OVERFLOW` |
| `malformed`, `walk` | the plane's or the walk's own status, carried |

A pin count or an epoch at its bound refuses; it never wraps.

**The state digest** is what the rails compare after every operation. It is the SHA-256 of
`"BHOF/state/v0\0"`, then `n_slots:u32 capacity:u64`, then per slot:

- `epoch:u32 state:u8 admitted:u8 pins:u32 admitted_generation:u32 held_length:u64`;
- the admitted registry (32 bytes);
- the SHA-256 of the held bytes, or 32 zero bytes. A slot holds bytes when it is frozen or
  retired, or exhausted with pins.

The table has no internal synchronization. Its owner serializes every operation: the C++
`PackArena` holds a mutex around each one, and reads a borrowed view outside the mutex.

## The per-step freeze (the dynamic-graph builder)

`bcir_hydrate_generations(f, plan, topo_gen, gens, n_gens, buf, cap, out_len)` freezes one
step's claim graph as a **StreamPack v4 bound to the live registry**. The segments and trace
records are the ones `bcir_hydrate` writes. The generation vector `gens` is appended, its maxima
become the header's `map_gen`/`data_gen`, `topo_gen` is the registry's, the pipeline depth is 1,
and every segment carries the v3 defaults (dispatch core, channel `host`). The oracle,
`freeze_claims`, is byte-identical. The laws, in order:

1. the arguments → `BCIR_ERR_NOSPACE`
2. a plan that does not cover the function → `BCIR_ERR_PROVENANCE`
3. no vector, or RIDs not strictly ascending → `BCIR_ERR_GENERATION` (a frozen step binds to a
   registry)
4. per claim:
   - read/write counts over their bounds, or a label that is not ≤ 31 printable ASCII characters
     → `BCIR_ERR_PROVENANCE`
   - ids not strictly ascending across all claims → `BCIR_ERR_PROVENANCE`
   - NOP claims are skipped for the checks that follow
   - a lane over H → `BCIR_ERR_LANE`
   - a plan step naming another claim → `BCIR_ERR_PROVENANCE`
   - a width that is not a power of two → `BCIR_ERR_WIDTH`
   - **any RID read or written that the vector does not declare** → `BCIR_ERR_PROVENANCE`: a
     step that touches a resource its registry does not carry is refused at the freeze, not at
     execution
5. the sizes → `BCIR_ERR_OVERFLOW` / `BCIR_ERR_NOSPACE`

The freeze checks everything before it writes anything, so on failure nothing is written. The
C++ `GraphBuilder::freeze` freezes **straight into a reserved slot**, the only write the
artifact's bytes ever get. `DynamicGraphOrchestrator::step` then admits the step against the live
plane and runs it through the single-node path. The pack it emits is in the hydrated layout, so
it may shard.

`bcir_hydrate` itself is unchanged: its v1 bytes stay byte-identical.

## The Stage 3 exit flow

The staged plan's Stage 3 exit gate is:

> one artifact generation flows plan → control → data → telemetry → evidence on the loopback
> with generation handles end to end; stale generations refused at every boundary.

That gate is one scripted flow, written once for the oracle
(`bcir/tests/handoff_fixtures.py::run_stage3_python`) and once for both native rails
([`runtime/c/test_stage3.h`](../../runtime/c/test_stage3.h)). The C harness drives it through
the pack table directly; the C++ harness drives it through the seam's arena, owners and views. All
three print the same lines. The program is `multi_histogram`, with four claims, placed for
AVX-512. Generation b moves one resource's map generation past the maxima.

1. **Control.** A lease and generation a travel as ControlRecordV1 bytes through a live CONTROL
   ring (BACKPRESSURE) into the resident plane.
2. **Plan.** Generation a's ExecutionPlanV1 is admitted by the plane.
3. **Data.** Generation a's StreamPack is written once into a slot, admitted against the plane,
   its shard manifest is gated, and it is dispatched in place.
4. **Telemetry.** Each event (the admission, each claim run) becomes one `TelemetryEnvelopeV0`
   sample bound to **the generation the data plane admitted**. The samples go through a live
   OVERWRITE telemetry ring of two slots into the host intake. The four claim samples are drained
   once, so the ring loses two of them and says so.
5. **The switch.** Generation b is carried by the control ring, and the intake follows the plane.
6. **The old generation, at every boundary.** Each of these is refused by its own law, with its
   own status:
   - a record witnessed against generation 1 (the plane's compare-and-swap);
   - generation a's plan (the plane);
   - its pack's dispatch and admission (the table);
   - its manifest (the manifest gate);
   - a late sample bound to it (the intake).
7. **Generation b flows**, and the retired artifact's handle is dead too.
8. **Evidence.** One line reconciles the flow:
   - the ring's loss equals the intake's gap count, exactly;
   - there is one stale record;
   - every delivered record was decided;
   - every control record was carried;
   - the plane is at generation 2;
   - the accepted count is the declared one;
   - the plane's and the table's state digests are included.

Its rows are `handoff.stage3.stale.accepted` (18 → 0: six boundaries on three rails) and
`handoff.stage3.flow.divergent` (75 → 0). The fault table injects a defect into each **other**
boundary the flow crosses — the plane's compare-and-swap, the intake's stale law, the ring's loss
count — and the flow's rows catch each one. So the flow is evidence about every boundary it
crosses, not only the data plane's.

## Gates

- **Rows** (`tools/perf/gemplus_baseline.py --group handoff`, graded by
  `tools/c/check_handoff.py` → `bcir/tests/handoff_fixtures.py::measure`; the rails are absent on
  the parent → the rails here):

  | Row | Parent → here |
  |---|---|
  | `handoff.stale.admitted` | 42 → 0 |
  | `handoff.stale.dispatched` | 90 → 0 |
  | `handoff.fresh.refused` | 90 → 0 |
  | `handoff.lifetime.unrefused` | 68 → 0 |
  | `handoff.copies` | 80 → 0 |
  | `handoff.builder.violations` | 422 → 0 |
  | `handoff.decisions.nonconforming` | 349 → 0 |
  | `handoff.traces.divergent` | 77 → 0 |

  The shard rows and the Stage 3 rows are listed above. `handoff.dispatch.overhead`, a timed
  ratio, went from 1.95× to about 1.01×: every per-dispatch check is O(1), and full verification
  moved to `admit()`, once.
- **C:** the `tools/c/check_runtime.sh` G16 section:
  - freestanding C11 and C23 builds under `-Wconversion -Wpedantic`;
  - the API's fail-closed laws;
  - every row at zero on the oracle and the C twin, with `-O0 == -O3 ==` the oracle;
  - a mutant with the dispatch-generation law removed, which the rows must fail.
- **C++:** `tools/cpp/check_handoff.sh`:
  - the standalone seam built with `-Wpedantic -Werror`;
  - the contract on real artifacts;
  - every row at zero on the oracle and the C++ seam;
  - a double-return mutant;
  - everything under ASan and UBSan.
- **Fuzz:** `fuzz_handoff.c` mode 2 runs table scripts under invariants: exact pins, retired
  slots have pins, epochs are monotone, borrowed bytes are immutable. Mode 3 runs the freeze: a
  refusal writes nothing, and a success verifies, is v4, and binds its vector.
- **Faults:** `tools/testing/faults/handoff.json`.

## Not claimed

- No cross-node transport: the distributed backend's dispatch remains a declared stub.
- No zero-copy across processes. The table lives in one address space; a shared-memory table is a
  later slice with its own ordering argument.
- No on-stack replacement: a frozen artifact is never modified, only released.
