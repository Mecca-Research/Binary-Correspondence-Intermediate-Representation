# BCIR Hierarchical Access Memory and Model-Artifact Fabric

> **Status:** provider-neutral compiler/runtime foundation landed; physical DMA, CXL,
> storage-controller, and cache-controller bindings are gated driver/kernel work.
>
> **Contract owner:** this document owns semantic resource movement, context-shard
> activation, and the boundary between the existing K_BCIR model and future memory-fabric
> adapters. The broader build order remains in
> [`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md).

## 1. Decision

BCIR adopts the useful core of the HAM proposal without claiming hardware that does not exist:

1. Treat model weights, adapters, plans, telemetry corpora, and indexes as content-addressed
   semantic resources with explicit dependencies, home banks, sizes, priorities, and generations.
2. Plan movement only over directed links declared in a `HardwareEnvelope`. An undeclared
   SSD→VRAM edge is not inferred from device names or marketing capability.
3. Distinguish a direct peer route, a direct DMA route involving host memory, and a staged route
   with an observable host-memory bounce. Functional fallback never inherits a direct-path
   performance claim.
4. Independently replay residency, capacity, version freshness, write invalidation, write-back,
   route cost, and final cleanup before lowering the same actions to BCIR claims and StreamPack.
5. Keep learned models advisory and off the legality path. Exact planning and hard facts dispose
   every candidate; learning may order bounded candidates for future generations.
6. Reference existing BCIRQ8, Safetensors, frozen-Q8-table, and StreamPack payloads from strict
   context-shard manifests instead of defining a generic new tensor format.
7. Defer GPUDirect/cuFile, PCI P2PDMA, dma-buf, CXL region management, NVMe firmware, and
   memory-controller integration to driver/kernel milestones with physical evidence.

This is a control-plane and artifact-contract landing. It performs no large inference, does not
configure a device, and does not move payload bytes.

## 2. What the proposal got right—and what required correction

Direct storage/accelerator paths are real but conditional. NVIDIA documents GPUDirect Storage as
an explicit DMA path between storage and GPU memory, while also documenting alignment, BAR,
filesystem, driver, allocation-type, topology, and compatibility-mode cases that stage data
through CPU memory. The API is CPU-issued and best suited to coarse streaming I/O; it is not a
promise that arbitrary cache-line traffic bypasses the host on every machine. See the official
[GDS overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html) and
[design guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html).

Linux likewise permits PCI peer-to-peer DMA only where topology is safe. The kernel blocks
undefined cross-host-bridge routing by default, and a dma-buf importer accepting peer resources
must handle resources without ordinary `struct page` backing plus explicit fences, reservations,
reference counts, and mapping invalidation. These are driver properties, not compiler guesses:

- [Linux PCI P2PDMA](https://docs.kernel.org/driver-api/pci/p2pdma.html)
- [Linux dma-buf sharing and synchronization](https://docs.kernel.org/driver-api/dma-buf.html)

CXL is a coherent interconnect and tier/pool substrate, not an automatic SSD→GPU bypass. Linux
constructs CXL ports, decoders, regions, DAX devices, and optional system-RAM exposure; device
types and currently supported management surfaces differ. BCIR must consume those observed,
versioned capabilities through an adapter rather than invent a flat universal memory:

- [Linux CXL device types and protocols](https://docs.kernel.org/driver-api/cxl/devices/device-types.html)
- [Linux CXL driver operation](https://docs.kernel.org/driver-api/cxl/linux/cxl-driver.html)
- [Linux CXL DAX operation](https://docs.kernel.org/driver-api/cxl/linux/dax-driver.html)

Filesystem DAX removes page-cache copies for memory-like storage but has architecture, mapping,
`struct page`, RDMA, and long-stall restrictions. It is not interchangeable with GDS or generic
peer DMA; see the [Linux DAX contract](https://docs.kernel.org/filesystems/dax.html).

The proposal's learned-cache ideas remain research inputs, not replacement policy claims.
[LeCaR](https://www.usenix.org/conference/hotstorage18/presentation/vietri) shows that online
learning can combine eviction policies, while learned indexes model key position rather than
making exact lookup unnecessary ([Kraska et al.](https://arxiv.org/abs/1712.01208)). BCIR therefore
uses deterministic trace-aware eviction for a known static access sequence and treats learned
scores only as future bounded-search priors.

For cloud-to-edge adaptation, knowledge distillation transfers supervised targets rather than
cloud gradient access ([Hinton, Vinyals, and Dean](https://arxiv.org/abs/1503.02531)); LoRA freezes
base weights and learns low-rank updates ([Hu et al.](https://arxiv.org/abs/2106.09685)). BCIR
records such outputs as attested payloads. It does not call an inference/embedding API a remote
backpropagation channel.

## 3. Source-backed inventory

| Proposal surface | State before this landing | State after this landing | Honest boundary |
|---|---|---|---|
| HAM/CXL vocabulary | `Resource.access="ham"`, priority, memory domains; MLIR `mem.ham`/`mem.cxl_swap` descriptors | Retained | Descriptors alone do not move data |
| Bank topology and movement | `DeviceManifest`, directed `HardwareEnvelope` links, `move_claim`, DMA descriptors | Reused by generic HAM routing | A link is caller-declared capability evidence, not a live probe |
| Model layer streaming | Exact resident/layer-stream/CPU-GPU candidates with move/prefetch/barrier/evict lowering | Reused; HAM generalizes the control contract beyond decoder layers | Target execution remains adapter work |
| Static/paged memory | Exact phase-lifetime address reuse and paged KV generations/eviction | Retained | Static allocation and HAM residency are distinct contracts |
| Tiny policy | Availability-aware telemetry Transformer, topology GNN, reward/DPO/PPO, bounded PUCT | Reused as advisory policy identity | Six simulated episodes cannot certify live promotion |
| Semantic resource tree | Missing | `HAMResource` dependency DAG, home bank, size, type, priority, mutability, and initial payload hash | Metadata only; no embedded payload |
| Route planner | Model-specific only | Directed shortest-cost route, lookahead prefetch, explicit eviction, mutable write invalidation/write-back | No physical DMA submission |
| Independent dynamic verifier | Missing | Replays routes, transient/persistent capacity, residency, generations, version lineage, and transfer summaries | Version lineage is not a payload hash after a write |
| Executable artifact | Model plans only | `HAMExecutionPlan` plus verified BCIR module and StreamPack | Existing StreamPack ABI is unchanged |
| Context shards | Provider/training artifacts existed, no activation envelope | Strict manifest/catalog, payload verification, quiescent activation, rollback token, HAM conversion | No in-flight attachment or opaque executable blob |
| Dual memory | Batch K-means and GNN infrastructure; no combined store | Exact bounded Q15 retrieval plus hard-fact veto oracle | Not a production ANN/CPG database |
| GDS/P2PDMA/CXL/NVMe binding | Missing | Still missing, explicitly sequenced in §9 | Requires kernel/vendor support and real hardware |

Repository component counts remain generated in [`../STATUS.md`](../STATUS.md).

## 4. Semantic metadata and planning contract

`HAMWorkload` is strict canonical JSON and contains no weight payload:

- the exact `HardwareEnvelope.digest`;
- a sorted, acyclic `HAMResource` inventory;
- a contiguous access sequence;
- a bounded lookahead distance and route-hop limit;
- an optional frozen learned-policy digest that affects provenance, never legality.

Each resource declares one pinned `home_bank`. Immutable resources retain that canonical copy.
Mutable resources may be written only when marked mutable; a write advances the generation,
invalidates every stale copy explicitly, and writes the new generation back to the home bank
before the phase can finish. The resulting `version_sha256` is a deterministic lineage identity
derived from the prior identity and write event. It does **not** claim to hash unseen output bytes.

Dependencies are semantic prerequisites. Before an access, the dependency closure and target
resource must all be current in the requested bank. This makes the proposed “tree branch” an
explicit DAG rather than an opaque neural embedding.

### 4.1 Routing

The planner uses only directed `LinkEnvelope` edges. Edge cost is:

```text
latency_ns + ceil(resource_bytes * 1_000_000_000 / bandwidth_bytes_per_second)
```

The lowest-cost bounded simple path wins with deterministic hop/path tie breaking. Routes are
classified as:

- `direct-peer`: one declared edge and neither endpoint uses the host channel;
- `direct-dma`: one declared edge with host memory as an endpoint;
- `staged`: two or more edges; `host_bounce=true` only when a host-channel bank is intermediate.

A compatible staged path may execute correctly but does not count toward direct-peer bytes.

### 4.2 Residency and eviction

The compiler has the complete bounded access sequence, so it uses exact next-use information
rather than pretending a runtime model can see the future. Pinned home copies are never eviction
candidates. For non-home copies, capacity pressure evicts the farthest next use with priority,
size, and resource-ID tie breaking. Lookahead prefetch protects earlier/nearer prefetched data;
when a later optional prefetch cannot fit without displacing it, the later prefetch is skipped and
the eventual demand remains fail-closed.

Every transfer records logical bytes, hop-multiplied wire bytes, predicted link time, route class,
and host-bounce status. Transient staged capacity contributes to the peak for each intermediate
bank. Plans finish with exactly one current home copy per resource.

## 5. Lowering and verification

`lower_ham_workload()` performs four independent stages:

1. Build the semantic action stream.
2. Replay the stream from the original home state and reject any residency, generation, route,
   or capacity mismatch.
3. Lower transfers, prefetches, barriers, accesses, invalidations, and evictions to ordinary BCIR
   claims; run K_BCIR realization; hydrate a channel-tagged StreamPack.
4. Run R-law, smart-lowering, bank-move, provenance, and a second artifact replay gate.

The CLI is metadata-only:

```bash
bcir-ham-plan \
  --hardware hardware.json \
  --workload ham-workload.json \
  --plan-out ham-plan.json \
  --pack-out ham-plan.bspk
```

It reports `payload_bytes_read: 0` and uses atomic output replacement. The StreamPack wire format
is unchanged; HAM actions lower through its existing claims, prefetch records, phases, channels,
generations, CRC, and provenance.

## 6. Context shards: transport crystallized artifacts, not “neural teleportation”

`ContextShardManifest` references exactly one existing payload and binds:

- payload format, byte count, and SHA-256;
- exact/quantized/approximate classification;
- base-model, hardware-envelope, selector, provenance, and certificate hashes;
- immutable source revision and artifact generation;
- a positive rank only for LoRA.

Allowed pairs are deliberately narrow: BCIRQ8 v1, frozen BCIR Q8 tables, Safetensors LoRA/weight
deltas, and verified StreamPack execution plans. ONNX describes executable graphs and GGUF is a
model container; neither is treated as a universal embedding or update protocol.

A canonical catalog maps a selector to ordered immutable shard identities. Activation requires
an exact selector/manifest/catalog match, advances beyond the catalog generation, waits for a
quiescent boundary, and emits a rollback token bound to the previous and next identities. The
artifact may become a `HAMResource`, but no API in this landing attaches weights to a running
model or mutates an in-flight StreamPack.

## 7. Dual-rail optimization memory

`OptimizationMemory` is the portable oracle for the proposed System-1/System-2 boundary:

- **Fuzzy rail:** bounded Q15 embeddings and exact squared-distance search with stable tie breaks.
- **Hard rail:** canonical `(subject, predicate, value)` facts covering hardware and program
  constraints.
- **Join rule:** a pattern is invisible unless every required fact is active. Similarity ranks
  only the remaining candidates.

The representation is intentionally smaller than a production vector database or Code Property
Graph. The original CPG work combines AST, control-flow, and program-dependence graphs
([Yamaguchi et al.](https://www.ieee-security.org/TC/SP2014/papers/ModelingandDiscoveringVulnerabilitieswithCodePropertyGraphs.pdf)); a later adapter must preserve
that distinction and BCIR's hardware facts. A FAISS/ScaNN-class approximate index may replace the
bounded exact retrieval only after recall, deterministic-filtering, corruption, and resource
gates pass against this oracle. See the primary
[FAISS paper](https://arxiv.org/abs/1702.08734) and
[ScaNN implementation reference](https://github.com/google-research/google-research/tree/master/scann).

The fast execution rail never performs a database query. A background control plane may retrieve
and certify a shard for a later generation.

## 8. Tiny-model placement

The proposal's six names are roles, not a requirement for six separately trained networks:

| Proposed role | Current BCIR substrate | Next evidence |
|---|---|---|
| Page-graph predictor | HAM dependency/access traces and lookahead oracle | Compare a frozen predictor against exact trace replay; storage firmware is §9 work |
| Cache eviction model | Exact next-use compiler policy; channel/frozen-Q8 priors | Compare learned ordering with deterministic baselines on real traces; no controller replacement claim |
| Phase router | Existing GNN/Transformer hardware policy and bounded PUCT | Real two-target episodes and measured-only promotion |
| Telemetry regressor | DataDNA/telemetry tokens, calibration, reward model | Real PMU/device corpus with availability/loss accounting |
| Mixed-stride pack model | `StridedView`, DMA descriptor lowering, GEM schedules | Measured driver DMA corpus and exhaustive legal pack comparison |
| Topology auditor | Existing memory-topology GNN | Physical topology transfer evidence and frozen deployment artifact |

Offline K-means/scalers, classical models, reward/DPO/PPO, Q8 priors, and MCTS already exist. KDE,
online K-means, tabular SARSA/Q-learning, edge-conditioned GNN variants, or a new tiny model are
added only when a measured role has a fixture, baseline, bounded state/action space, and exact
admission gate. None belongs in CPU microcode or controller firmware from this repository.

## 9. Driver/kernel/firmware sequence

The following work is deliberately **not** implemented by this landing:

| Gate | Hardware-owned work | Prerequisite and acceptance evidence |
|---|---|---|
| **HMF-D0 capability adapter** | Read supported vendor/Linux topology and produce an attested `HardwareEnvelope` | Probe may veto, never silently steer; direct/staged classification agrees with an independent topology tool |
| **HMF-D1 GDS userspace adapter** | Optional cuFile file/buffer/stream registration and submit/cancel/status binding | Supported NVIDIA GPU/filesystem/driver; direct and compatibility modes separately measured; alignment/BAR/error corpus |
| **HMF-D2 Linux peer-memory bridge** | PCI P2PDMA and/or dma-buf exporter/importer, fences, reservations, invalidation, IOMMU isolation | Same-hierarchy/cross-root negative tests, peer death, hot-unplug, cancellation, stale generation, KASAN/KCSAN/lockdep |
| **HMF-D3 CXL tier adapter** | Enumerate ports/decoders/regions, devdax/kmem policy, hotplug and poison/error handling | Stock-Linux baseline, NUMA/tier measurements, restart/hot-remove, capacity-generation and persistence tests |
| **HMF-D4 NVMe semantic store** | Filesystem/index/ZNS layout, durability, recovery, integrity, optional computational-storage interface | NVMe/virtio-blk driver maturity, power-loss model, append/recovery and corruption corpus; no controller model required initially |
| **HMF-D5 controller co-design** | Any cache/NVMe/CXL-controller inference engine or interrupt-avoidance mailbox | Vendor/FPGA research platform, bounded WCET/power proof, safe fallback, firmware signing/update/recovery, real benefit over host control plane |

The NVM Express ZNS command set provides sequentially written zones and host-managed ordering; it
does not supply BCIR semantic metadata by itself. Any semantic store must define its own durable
index and recovery protocol over a supported block/filesystem interface. See the official
[NVMe ZNS specification page](https://nvmexpress.org/specification/nvme-zoned-namespaces-zns-command-set-specification/).

These gates live in the driver/kernel program because they require privileged configuration,
kernel lifetime/synchronization, firmware recovery, or physical topology. Until they pass, HAM
plans are simulator/compiler artifacts and hardware links are user-supplied evidence.

## 10. Acceptance and promotion

Every HAM change must retain deterministic tests for:

- direct, direct-DMA, staged, host-bounce, missing-route, and route-hop cases;
- dependency cycles, unknown banks/resources, immutable writes, malformed/duplicate JSON;
- capacity saturation, protected lookahead, deterministic eviction, and pinned home copies;
- stale generations, mutable invalidation/write-back, final cleanup, and plan tampering;
- exact peak/byte/latency re-derivation and StreamPack/R-law verification;
- context-shard format pairing, payload hash/size, catalog binding, quiescence, rollback, and
  atomic persistence;
- fuzzy-near-but-illegal retrieval being rejected by hard facts.

Physical promotion additionally requires direct-versus-staged parity, measured bandwidth/latency
intervals, CPU utilization, power/thermal impact, IOMMU/isolation evidence, cancellation and peer
death, and rollback on every supported target. No simulated result promotes a live route or shard.
