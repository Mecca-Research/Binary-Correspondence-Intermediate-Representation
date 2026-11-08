# BCIR AI system: one knowledge plane, its planner, and its processes

> **Status: design (2026-11-08); no AIS slice has landed.** This roadmap defines how the hybrid
> database ([`training/DATABASE_ROADMAP.md`](../training/DATABASE_ROADMAP.md)), the Unicode 17.0
> database ([`training/UNICODE_ROADMAP.md`](../training/UNICODE_ROADMAP.md)) and the
> small-language-model laboratory ([`training/SLM_ROADMAP.md`](../training/SLM_ROADMAP.md)) work
> as one system rather than three. It covers what K_BCIR and GEM+ do in that system, which
> algorithms it needs beyond those roadmaps and the ML program, and how the system becomes a set of
> operating-system processes on the kernel's IPC. The processes serve three workloads: building and
> querying the learning database, language processing, and automated metaprogramming (agent
> coding).
>
> It is a companion roadmap in the repository's authority order. It changes no law (R1–R25), no
> frozen ABI and no landed slice; every mechanism it names is either landed (with the slice that
> landed it) or labeled proposed. Its slices (`AIS-n`) start **after** the foundation phase's exit
> ([`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §3.1), alternating with the driver order;
> AIS-0, this document, is the only one in the foundation phase.

## 1. The answer in brief

- **One system means five shared things:**
  - one identity, a knowledge epoch that binds every generation a result read;
  - one planner, K_BCIR's legality before cost on the 12-axis `CostVector`;
  - one artifact path: module → K_BCIR plan → GEM+ schedule → StreamPack → executor;
  - one certificate family, DER envelopes and TMSAO certificates;
  - one data plane, immutable generations read as borrowed views.

  The three programs each already use most of these. What is missing is the epoch, and query and
  text processing taking the same artifact path the training step already takes (PX-2b).
- **K_BCIR and GEM+ need no new laws and no new cost axes.** They need:
  - new claim families: query operators, text operators, channel placement;
  - the landed region kinds put to new use: SDF and timed regions (G23/G24) for the
    producer–consumer pipeline between services, and expected cost (G22) for a lookup cache;
  - the `fabric` and `sync` axes populated, which the lookup's price leaves at zero today;
  - the knowledge epoch declared in the scope identity's existing generations component (G0's
    `G`), so no new component and no scope version.
- **Six algorithms are needed beyond the Unicode roadmap and the ML system** (§4.3, AL1–AL6):
  - equi-joins over the catalog's sorted columns and postings;
  - batched multi-query lookup with shared scans;
  - a fused, streaming native text pipeline (UTF-8 → NFC → segmentation → BPE);
  - lookup caching priced by expected cost;
  - channel-mode selection between the three IPC modes;
  - an append-only stream generation for data produced while training runs.

  Approximate nearest neighbours stay a measured decision at UC-10's scale, not a default.
- **The OS model is eight service roles on the IPC program's four primitives** (§5).
  Control travels as control records (G14) on SPSC rings (G15). The six v1 kinds already move
  an epoch and lease a buffer; request and completion kinds are appended in a v2. A record
  carries digests, offsets and capabilities, never payloads. Data travels as immutable
  generation files mapped read-only, and as batch buffers whose ownership moves. Each role is a JIT-microkernel catalog entry: a
  preverified service bundle (driver roadmap §7.5).
- **It is buildable before native kernel work.** Rung I1 (one process, in-process
  RuntimeChannel) and rung I2 (separate Linux processes, shared regions, futex and eventfd
  semantics) need no driver. Native IPC (I4/I5) follows the driver ladder as already planned,
  and so do the device data paths: NVMe for generations, DMA leases for zero-copy batches,
  timers for deadlines.

## 2. What already exists, and where the seams are

| Piece | Landed | Where |
|---|---|---|
| A legality-first query planner priced on K_BCIR's `CostVector` | S1–S19 | `training/tools/plan.py`, `training/tools/db/` |
| Content-addressed generations, one transaction per publish, a `CURRENT` pointer | S6, S13 | `training/tools/generations.py`, `catalog.py` |
| The Q15 top-k under a predicate mask; Q8 matvec and row dots | native AI ABI v1 | `runtime/c/bcir_ai_kernels.c` |
| The lookup's legality rules, price, DER certificate and L0–L3 placement | specified, not landed | SLM_ROADMAP §2.7–§2.11 (SLM-2–SLM-5) |
| The Unicode generation: pinned UCD, records, the one UAX #15 implementation | specified, not landed | UNICODE_ROADMAP §5, §10 |
| The native training step as a K_BCIR-planned, GEM-hydrated StreamPack program, run from data, its memory placed by lifetime | NDT-GEM, PX-1 to PX-5 | `research/BCIR_GEMPLUS_ROADMAP.md` §9 |
| Control records, the live SPSC ring, the data-plane hand-off of borrowed views | G14, G15, G16 | [`kernel/BCIR_CONTROL_PLANE_ABI.md`](kernel/BCIR_CONTROL_PLANE_ABI.md), [`kernel/BCIR_LIVE_RING_ABI.md`](kernel/BCIR_LIVE_RING_ABI.md) |
| Exact solvers and lower bounds (TMSAO-1/2), joint schedule × memory, expected cost, SDF and timed regions, I/O bounds, value-priced optimization depth, signed plans | G4, G28, G22, G23/G24, G25/G26, G27, G21 | `bcir/kbcir/`, `bcir/gem/` |
| Bounded reasoning search with pluggable `verify`; the append-only training-pipeline ledger | landed | `bcir/hosted/training/` |
| Bounded subprocess execution and the subject-neutral grading rail | landed | `training/tools/safe_process.py`, `grading.py` |
| Context shards and dual optimization memory | contract and bounded oracle | ML roadmap §1.4.1; [`kernel/BCIR_HAM_MEMORY_FABRIC.md`](kernel/BCIR_HAM_MEMORY_FABRIC.md) |
| Intent → claim-graph synthesis under law | design | ML roadmap §8.6 |
| The IPC program: four primitives, the channel tuple, rungs I1–I8 | planned | driver roadmap §7.4 |

**The three seams this roadmap closes:**
1. **The query planner is a second planner.** `plan.py` prices on `CostVector` but plans outside
   the K_BCIR → GEM+ → StreamPack path the training step now takes. Its plans therefore carry no
   TMSAO certificate above TMSAO-4, and they cannot be scheduled together with the step that
   consumes them.
2. **Generations bind pairwise, not as one.** SLM-2's `generation` rule checks that a plan's
   catalog, sets, projection and engines agree. No single record names everything a result read:
   the Unicode version, the catalog, the sets, the tokenizer, the checkpoint and the plan baseline.
3. **The engines are corpus-side.** The database and the Unicode rail live under
   `training/tools/`, and the IR may not import them (`AGENTS.md`, the one-way corpus law). A
   service the OS runs is IR-side, so each engine the services need gains an IR-side twin. It
   lives in `bcir/` with a `runtime/c` form where the hot path needs it, and is proven equal by
   a parity gate that lives corpus-side: `training/tools` may import `bcir`, never the reverse.
   UC-4 is specified in this shape for normalization (`bcir/hosted/training/unicode_normalize.py`).
   The corpus data crosses as data.

## 3. The unified system

### 3.1 One identity: the knowledge epoch (AIS-1, proposed)

A **knowledge epoch** is one DER record binding the generations that every result in it read:

```
KnowledgeEpochV1 ::= SEQUENCE {
  unicode      GenerationRef,     -- the UCD generation (UC-1..UC-8; version in `extra`)
  catalogs     SEQUENCE OF GenerationRef,   -- the chunk table, the character table (UC-3), ...
  sets         SEQUENCE OF SetRef,          -- embedding sets bound by ids-in-order
  tokenizer    Digest,            -- bcir.byte_bpe.v2 (SLM-1)
  projection   Digest OPTIONAL,   -- the frozen query projection (SLM-5)
  checkpoint   GenerationRef OPTIONAL,      -- the model generation
  planBaseline Digest             -- training/plans/baseline-v1.json's digest
}
```

The ASN.1 sketch is illustrative; the module is UC-8's `BCIR-Unicode`, extended append-only, or
the ABI document's choice at AIS-1. The rules:
- **Every certificate binds an epoch.** A lookup envelope (SLM-4), a training-ledger stage, a
  checkpoint, an inference record and an agent action each name the epoch digest.
- **The epoch generalizes SLM-2's `generation` rule.** A plan reading two epochs is refused
  unless the plan is a declared migration.
- **Epochs change only at quiescent boundaries**, as every generation does today: artifacts
  are immutable within a generation (the master roadmap's invariant 4).
- **The epoch is declared in the scope identity.** `ExecutionScopeV1` (G0) already has a
  generations component, `G` ("generations of hardware profile, calibration, firmware, driver,
  model, artifacts"). The epoch digest goes there, so a TMSAO certificate states which data it
  holds for. There is no new component and no new scope version.

### 3.2 One planner, one artifact path

Every operation the system performs is a BCIR module planned by K_BCIR and hydrated by GEM+.
Training already is (NDT-GEM, PX-2b). The table names the claim family for each workload:

| Workload | Claim family | Status |
|---|---|---|
| A training step | kernel calls, launches, arena spans | landed (PX-2b) |
| A lookup | `resolve` (predicate → selection), `intersect` (postings), `seek` (sorted index), `topk` (Q15 kernel, masked), `fuse` (shards, two spaces), `fetch` (late materialization) | proposed (AIS-2) |
| A text batch | `decode` (UTF-8, bounded), `nfc` (quick-check then full), `segment` (UAX #29), `bpe` (byte_bpe.v2), `features` (character records) | proposed (AIS-5) |
| An inference request | lookup, prompt assembly, decoder steps (L0) | proposed (AIS-6) |
| An agent action | retrieve, propose, build, verify, record | proposed (AIS-8) |

The pipeline is the same for each workload:
1. The R-laws and the appended structural rules refuse first.
2. K_BCIR prices the legal candidates.
3. GEM+ schedules the chosen plan.
4. The result hydrates into a StreamPack.
5. The Python oracle and the C twin execute it, their outputs compared.

`plan.py` stays the corpus-side oracle for lookups; the IR-side query module matches it plan for
plan and row for row (the parity gate, §2 seam 3). What this buys:
- one schedule for a training step together with the lookups it consumes (G23's SDF rates);
- TMSAO-1/2 certificates on small query plans from the exact solvers (G4, G28);
- signed plans (G21) for every service request.

### 3.3 One certificate family

Certificates are DER out, BER in. Each records its legal and refused candidates and the chosen
one, separated from cost:

| Certificate | Where it comes from | Status |
|---|---|---|
| The `SelectionEnvelope` | the ASN.1 rail (`bcir/asn1/manifest.py`) | landed |
| The `LookupEnvelope` | SLM-4 | specified |
| The plan statement | `PlanStatementV1`, G21 | landed |
| The TMSAO certificate | G4 onward | landed |
| The training ledger's stage records | `TrainingPipelineLedger` | landed |
| The knowledge epoch | AIS-1 | proposed |

A measured sidecar stays outside the digest and labeled measured. This is what full traceability
means operationally: any number the system reports carries the epoch, the plan and the
certificate that produced it, and replays byte-identically on another host (SLM-12's rule, made
system-wide).

### 3.4 One data plane

Generations are immutable, so they are the zero-copy unit:
- **Shared, not copied.** Columns, postings, sorted indexes, Q15 codes, character records and
  checkpoints are written once and read through mapped, read-only views. G16 hands borrowed
  views across the control plane. Readers share one mapping and take no locks.
- **Swapped, not edited.** A new epoch is published (S13), readers move at a quiescent
  boundary, and the old epoch is retired when its last reader leaves.
- **Ownership moves, bytes do not.** Batch buffers between services move by transfer of
  ownership, the lease of driver roadmap §5.5, never by copying.

## 4. K_BCIR and GEM+ in the system: what changes, and the new algorithms

### 4.1 What stays

The law and the price stay as they are:
- the R-laws and legality before cost;
- the 12 axes: `compute`, `memory`, `compile`, `accuracy`, `verification`, `fabric`, `sync`,
  `reliability`, `security`, `thermal`, `power`, `contention`;
- the TMSAO ladder;
- the two-truth quarantine (learned proposers rank and propose; they never certify);
- the L0–L3 placement ladder.

### 4.2 What K_BCIR and GEM+ gain (no new laws)

- **New claim families** (§3.2's table): query operators, text operators and channel placement.
  Each is a registry entry with its legality and its price; none is a new R-law.
- **Populated axes.** The lookup's price leaves `fabric`, `sync`, `reliability` and `security`
  at zero today, and rightly: one process, no boundary. Across processes they become real:
  - `fabric` is the bytes crossing a protection or device boundary;
  - `sync` is the wakeups, barriers and doorbells: the `T_sync + T_wake` terms of the driver
    roadmap's `T_IPC`;
  - `security` is the capability checks a channel mode keeps;
  - `reliability` is a restart's recovery work.
- **Region kinds put to work.**
  - G23's SDF region with its FIFO bounds models the text → store → train pipeline: its balance
    equations fix batch sizes, and its FIFO bounds fix ring sizes.
  - G24's max-plus cycle time is the pipeline's throughput floor, so a ring or batch choice
    carries a certificate.
  - G22's expectation semiring prices a cache from a measured `BranchProfile`.
  - G25/G26's I/O bounds price a scan against its compulsory loads.
- **Scope identity.** The knowledge epoch is declared in `ExecutionScopeV1`'s existing `G`
  component (§3.1).

### 4.3 The algorithms this system needs

None of these is in the Unicode roadmap or in the ML system as built:

| Id | Algorithm | Why it is needed | Exactness | Slice |
|---|---|---|---|---|
| AL1 | **Equi-joins over the catalog**: merge join over `order.bin`'s sorted columns, hash or postings join over indexed columns. Where the key is an indexed column, the join's size is computed exactly from the two sides' value counts, Σ_v count_L(v) · count_R(v), as S15 computes conjunctions exactly | UC-3 makes a second table; SLM-10's lexical graph is three tables (characters, lexemes, edges) whose traversals are joins | exact; legality refuses a join whose key types differ or whose generation differs | AIS-4 |
| AL2 | **Batched multi-query lookup**: one masked kernel pass over the union of a batch's selections, a per-query top-k and an exact split of the shared scan's cost | a training step issues one lookup per sequence; B separate scans re-read the same codes B times | exact: each query's result equals its solo result, which the parity gate checks | AIS-3 |
| AL3 | **A fused streaming text pipeline in C**: UTF-8 validation, NFC quick-check falling back to full composition, UAX #29 boundaries and byte_bpe.v2 merges in one pass over a borrowed input, no intermediate strings, tables generated from the Unicode generation | the native data plane has "no string or code-point primitive" (SLM_ROADMAP §1.3); tokenization in Python bounds every text-heavy workload | exact: byte-identical to UC-4, UC-8 and SLM-1's oracles on their conformance suites | AIS-5 |
| AL4 | **Lookup caching priced by expected cost**: a result cache keyed by (epoch, request digest), admission decided by G22's expectation over a measured `BranchProfile` | retrieval-augmented training repeats queries on every pass over its data; agents repeat retrievals within a task | the cache is exact (content-addressed); its admission rule is priced, not learned | AIS-3 |
| AL5 | **Channel-mode selection**: same-domain call, shared region or protected message channel, chosen per channel by K_BCIR (driver roadmap §7.4's three x86 modes) | the IPC program names the choice; this states it as a claim with legality and price | legality is structural; the price is modeled until measured | AIS-7 |
| AL6 | **Stream generations**: an append-only generation for data produced while training or agents run (agent transcripts, accepted patches, new judged queries), sealed into a normal generation at a boundary | continual learning and agent coding produce data mid-run; today every generation is a whole publish | exact; a stream is never read as sealed | AIS-9 |

Two decisions are measured rather than assumed:
- **Approximate nearest neighbours** reopen only when UC-10's row count arrives
  (DATABASE_ROADMAP's reopening condition, ~50,000 rows). Even then they enter as a
  non-`EXACTNESS` backend the `exactness` rule already fences.
- **Learned proposers** stay at L2: the reasoning proposer, the code proposer and the routing
  gate. Each is promoted only past a written gate of the kind
  [`training/LEARNED_EMBEDDING_GATE.md`](../training/LEARNED_EMBEDDING_GATE.md) holds.

## 5. The OS processes, on the kernel's IPC

### 5.1 Eight service roles

Each role is a JIT-microkernel catalog entry, a preverified service bundle (driver roadmap §7.5).
Its manifest carries:
- its code identity, its epoch reads, its channels and its capability routes;
- its resource budget, including cores, huge-page arenas and the NUMA node;
- its rollback generation.

Heavy optimization happens offline. A role starts from an immutable image and binds only
declared configuration.

| Role | Does | Owns | Reads | Hot path |
|---|---|---|---|---|
| **text** | Unicode validation, normalization, segmentation, tokenization, character features | nothing mutable; one Unicode generation pinned | UTF-8 batches | AL3 (C) |
| **store** | catalog queries, lookups, joins, publishes | the single writer of each catalog (S13); many readers | generations (mapped read-only) | the Q15 and Q8 kernels, AL1, AL2 |
| **train** | the native decoder trainer and the hosted stages | the training arenas (NDT-PAGES, NDT-MEM), the checkpoint stream | batches from text and store; lookups from store | the NDT kernels (C) |
| **infer** | BCIRQ8 inference (L0), prompt assembly | the KV cache | checkpoints; lookups from store (L1) | `bcir_llama` (C) |
| **plan** | K_BCIR and GEM+ planning, certificates | the plan cache | epochs; the workload component `W` | offline; online only within G12's work-unit budgets and G27's value |
| **verify** | the R-law verifier (`bcir-opt -bcir-verify`), the C front (`bcir-cc`), tests in a bounded sandbox (`safe_process`'s bounds as a service) | nothing mutable | modules, sources | the verifier rails |
| **ledger** | provenance and telemetry: envelopes, certificates, epochs, the data-DNA loop | the append-only ledger | everything's records | append-only writes |
| **agent** | metaprogramming orchestration: intent → retrieve → propose → build → verify → record | an agent's stream generation (AL6) | store, infer, verify | `bounded_reasoning_search` |

### 5.2 How the roles talk: the four primitives

Every channel is the IPC program's tuple `(id, generation, producer, consumer, region, ring,
event, rights, ordering, failure policy)`, admitted once:
- **Control** is the control record (G14) on per-pair SPSC rings (G15). A record carries a
  kind, sequence, generation, capability and integrity metadata, never a payload.
  - **v1's six kinds already run the system's lifecycle.** `lease` grants a buffer. `quiesce`,
    `generation` and `activate` move a service to a new epoch at a boundary, with `activate`
    naming the epoch's digest as its artifact (proposed). `rollback` restores the previous
    epoch, and `cancel` withdraws earlier requests.
  - **Requests and completions are new kinds**: a lookup, a text batch, a build. They are
    appended in a v2 of the record, as the frozen-ABI rule requires, fixed-size like v1's, and
    carry content digests, region offsets and a lease id (proposed, AIS-6).
- **Bulk data** sits in **shared regions**:
  - generations mapped read-only into every reader;
  - batch and result buffers leased with a transfer-of-ownership record.
- **Wake-ups** are **doorbells** past a threshold, and **waitable words** (futex semantics) when
  a ring is empty or full.
- **Batched requests** use **SQ/CQ rings** with registered buffers: lookups from train, builds
  from agent.

The steady state of a lookup from the trainer:
1. train writes a v2 request record (the epoch digest, the request digest, a leased query
   buffer) to its store ring and rings the doorbell past the batch threshold.
2. store resolves the batch with AL2 and writes the top-k row ids into the leased result buffer.
3. store returns ownership with a completion record carrying the `LookupEnvelope` digest.
4. ledger receives the envelope.

No byte of a generation is copied, and nothing is serialized beyond the record.

### 5.3 The workloads as process graphs

- **Building the learning database.**
  1. text normalizes, segments and tokenizes the sources.
  2. store builds chunk and character tables, then sets.
  3. store publishes an epoch.
  4. plan re-records the plan baseline.
  5. ledger seals the epoch.

  Throughput is G24's cycle time over the text → store pipeline.
- **Querying it for deep learning.**
  - train ↔ store lookups (AL2, AL4) inside the training step's schedule;
  - checkpoints stream to store as generations;
  - one epoch per run unless a migration is declared.
- **Language processing.** text → infer, with infer ↔ store lookups at L1. The L0 decoder stays
  lookup-free (SLM_ROADMAP §2.11).
- **Automated metaprogramming (agent coding).**
  1. agent takes an intent and retrieves from store.
  2. infer proposes a candidate: a patch, or a claim graph under ML roadmap §8.6.
  3. verify builds it and checks it: the R-laws, `bcir-cc`, the tests, in the bounded sandbox.
  4. `bounded_reasoning_search` accepts or rejects. This is SLM-6 generalized: the engines are
     the `verify` callbacks, and a candidate that fails the law does not exist.
  5. agent records the accepted candidate to its stream generation (AL6), and ledger binds it to
     the epoch.

  The repository's own corpus, verifiers and grading rail are this workload's first data, and
  the one-way corpus law holds: the corpus is read as data.

### 5.4 Scheduling and memory policy for AI processes

The OS policy is itself a K_BCIR choice, priced on `contention`, `memory` and `fabric`:
- cores partitioned by role (the trainer's pool kept off store's and text's cores);
- huge-page arenas (NDT-PAGES);
- NUMA placement of each generation's mapping near its heaviest reader;
- ring sizes and batch sizes from G23/G24;
- wake thresholds from the measured `T_IPC` terms.

On Linux these use stock mechanisms first: affinity, cgroups, `madvise`, memfd, futex and
eventfd. They are measured against the comparators of the IPC rungs. The native microkernel
inherits the same manifests (driver roadmap §7.5) once I5 and the driver evidence exist. This is
the boundary the master roadmap's §3.2 draws: in-process memory pressure is the trainer's own
work; placement, pinning, wake-ups and cross-process copies are this layer's.

## 6. The slices

One gateable slice per PR, slice id in the title, RED before GREEN, a fault entry per new check,
exact gate output in the PR body. A slice that adds an IR-side twin of a corpus-side engine lands
its parity gate corpus-side.

| Slice | What | Depends on | Gate |
|---|---|---|---|
| **AIS-0** | this design | — | docs governance |
| **AIS-1** | the knowledge epoch: the record, its binding in every certificate, its declaration in `ExecutionScopeV1`'s `G` component | SLM-4, UC-8 | an epoch-mixed plan is refused; every certificate kind names one epoch; the scope digest moves when the epoch does and only then |
| **AIS-2** | the lookup as a StreamPack program: the query claim family, K_BCIR planning, GEM+ hydration, the oracle and the C twin | SLM-5, AIS-1 | parity with `plan.py` plan for plan and row for row over the plan baseline's requests; TMSAO-1/2 on the small plans the exact solvers reach |
| **AIS-3** | batched multi-query lookup (AL2) and the expected-cost cache (AL4) | AIS-2 | each query's result equals its solo result; the cache never serves another epoch; the admission price reproduces from the recorded profile |
| **AIS-4** | equi-joins (AL1) | AIS-2, UC-3 | join results equal the nested-loop reference; the computed size equals the joined row count on every indexed key; mistyped and cross-generation joins are refused |
| **AIS-5** | the fused native text pipeline (AL3) | UC-8, SLM-1 | byte-identical to the oracles on `NormalizationTest`, `GraphemeBreakTest` and `WordBreakTest`, and to SLM-1's tokenizer over the corpus; fuzzed under ASan and UBSan at the trust boundary |
| **AIS-6** | services at rung I1: text, store, train, infer and ledger in one process over RuntimeChannel; the control record's v2 request and completion kinds, on both rails; the pipeline priced as G23/G24 regions | AIS-2, AIS-5 | results identical to the direct calls; every v1 record still decodes byte-identically and every malformed v2 record is refused on both rails; the cycle-time certificate reproduces |
| **AIS-7** | services at rung I2: separate processes, read-only generation mappings, leased buffers, channel-mode selection (AL5) | AIS-6; DMA's lease laws (driver order 2) | rung I2's comparator (memfd + futex2 + eventfd); zero copies of generation bytes, counted; peer-death and stale-generation semantics |
| **AIS-8** | the agent loop: verify as a service, `bounded_reasoning_search` over code proposals, the sandbox's bounds | AIS-6, SLM-6 | a candidate failing any verify engine is never accepted, even when a witness gives it the best score; every accepted action names its epoch and certificate |
| **AIS-9** | stream generations (AL6); I3 asynchronous channels; the roles as JIT-microkernel catalog entries | AIS-7, AIS-8 | a stream is never read as sealed; the I3 comparator (io_uring); each bundle instantiates only from its signed manifest |

Native rungs I4 and I5, NVMe-backed generations (HMF-D4) and device transfers (HMF-D1/D2) follow
the driver ladder (driver roadmap §5.6) and are not slices here.

## 7. Order, relative to everything else

1. **Now: the foundation phase's fifteen slices** (master roadmap §3.1, starting at UC-1). AIS-0
   is the only AIS slice in it.
2. **Then the driver order resumes** with the event timers, DMA and the virtio queue proof,
   alternating one slice at a time with AIS-1 to AIS-6. Those six need no driver; AIS-7 waits for
   DMA's lease laws.
3. **The deferred Unicode and SLM slices** (UC-9–UC-11, SLM-8–SLM-13) take the remaining
   alternate turns. UC-10's scale is where AL1, AL2 and the approximate-neighbour decision get
   measured.
4. **AIS-7 to AIS-9 run with driver orders 2–7**: DMA, virtio, ACPI, PCIe, NVMe and the I1–I4
   IPC rungs. Rung I5 and the native processes come with the master kernel (order 13).

## 8. What this roadmap does not claim

- **No speed is claimed for any design here.** Every gain is measured by the slice that claims
  it, on the classes of the GEM+ measurement discipline (`exact`, `ratio`, `wall`), and a
  virtualized host's numbers stay TMSAO-4.
- **No learned component certifies anything.** Proposers, rankers and routers rank and propose;
  the verifier and K_BCIR's legality decide.
- **The corpus never becomes a dependency of the IR.** Every engine a service runs is IR-side,
  with its corpus-side twin as the oracle.
- **No code from a project under confidentiality terms is used**, and no external dataset is
  admitted without its license review and a pinned revision (UNICODE_ROADMAP §7.4's rule).
