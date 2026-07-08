# BCIR Game-Optimization Roadmap — legendary game techniques, mapped to GEM / K_BCIR / StreamPack (2026-07-08)

**What this is**: a research note + build plan that mines the *gold-standard optimization lore of
games* — RollerCoaster Tycoon's assembly, Doom's BSP, Quake's fast inverse square root, direct
IEEE-754 bit manipulation, Elite/Frontier procedural generation from a seed, Crash Bandicoot's
low-memory CD streaming, Elite Dangerous / No Man's Sky deterministic universes, Factorio's
cache-friendly layouts and `fork()`+copy-on-write background save — and separates the parts BCIR
can legally import from the parts that would quietly break its guarantees. The imports land in
three places: **GEM** (the StreamPack scheduler / paging / DMA / prefetch hot path), **K_BCIR**
(the tropical cost model, the e-graph, the realization search), and the **StreamPack ABI**
(the frozen artifact, generation tags, provenance/replay). This doc is the reference for the
game-optimization work; the per-slice code lands as follow-on build waves under the ML/AI
integration roadmap's process.

> **The honest reframe (read this first).** The request asked to "manipulate binary / floating-point
> for speed **AND perfect accuracy**." That is **half true, and the split is the entire thesis of
> this document.** There are two disjoint families of game bit-trick, and BCIR treats them under
> two disjoint disciplines:
>
> - **EXACT family** — tricks that exploit the *encoding* without approximating the *value*, so the
>   result is **bit-identical** to its reference. Multiply/divide by 2ᵏ as an integer shift or an
>   IEEE-754 exponent-add (`ldexp`/`scalbn`); sign/abs/copysign as XOR/AND masks; the total-order
>   integer comparison of floats; `frexp`/`ldexp` field decomposition; Doom's 16.16 fixed-point;
>   deterministic integer lockstep; A\*/Dijkstra/JPS optimal-cost search; flow-field integration;
>   SoA↔AoS reindexing; arena allocation; `fork()`+COW page snapshots; Roaring bitsets;
>   content-addressed replay. These are **legal, value-invariant K_BCIR/GEM/StreamPack
>   realizations**, gated by a bit-identity proof. "Perfect accuracy" is **true** here.
>
> - **APPROXIMATE family** — tricks that trade accuracy for speed and are **NOT** bit-exact. The
>   Quake fast inverse square root (~0.175 % peak relative error after one Newton step) and its
>   bit-log/exp2/pow parent family; lossy MegaTexture HD-Photo→DXT transcode; float-based lockstep
>   (AoE II HD FPU-pinning) which is only fragile per-platform quasi-determinism. For these,
>   **"perfect accuracy" is FALSE** — and BCIR must never dress them up as exact. They split again:
>   *numeric* error (fast-rsqrt) can ride a **new** certified relative-error contract that informs
>   `cost.accuracy` and is quarantined from the legality verdict by two-truth; *combinatorial* error
>   (HPA\* path-suboptimality, navmesh polygonization) is path-**value** divergence that R17 cannot
>   bound at all, and lossy transcode is orders beyond ≤1 ULP — both are **do-not-import** into the
>   exact core.
>
> Every "already in BCIR" claim below is scoped honestly, and the myth-flags are carried through:
> RCT was **~99 %** assembly (not 100 %) and **did** ship patches ("bug-free / never patched" is
> myth); the fast-rsqrt constant `0x5f3759df` was **not** written by Carmack; BCIR is
> **zero-storage-REPRODUCIBLE**, not zero-storage; generation tags are staleness **identity**,
> orthogonal to copy-on-write; Factorio's real headline cache win was **software prefetch** over fat
> AoS objects, **not** SoA packing.

---

## 1. The exact-vs-approximate split — the load-bearing thesis

BCIR's whole legality model rests on **value-invariance**: a realization is legal only if it is
provably bit-identical to its reference (`matmul_tiled == reference`, `SoA == AoS`), and **cost,
learning, and telemetry are quarantined** from that verdict by two-truth (`twotruth.py:99`). A
game bit-trick can therefore enter the *legal* rail **only if it is exact**. Anything approximate
must enter through a *different* door — a numeric accuracy contract that two-truth keeps out of the
verdict — or not at all.

| Trick | Family | Bit-identical? | BCIR door | Status |
|---|---|---|---|---|
| `x*2^k`, `x/2^k` (unsigned) → shift; float → exponent-add (`ldexp`/`scalbn`) | **exact** | yes (within range) | value-invariant K_BCIR rewrite | build-exact |
| Signed `x/2^k` (the trap) | **exact** *iff* `((1<<k)-1)` bias added | yes with bias; **NO** as bare `>>k` | value-invariant rewrite + bias proof | build-exact |
| sign / abs / copysign via XOR/AND mask | **exact** | yes (all values, incl. NaN) | value-invariant rewrite | build-exact |
| IEEE-754 total-order integer compare | **exact** | yes (domain: NaN unordered, ±0) | value-invariant rewrite + `layout.py` radix sort | build-exact |
| `frexp`/`ldexp` field split | **exact** | yes | value-invariant rewrite | build-exact |
| Doom 16.16 fixed-point (`FixedMul = (a*b)>>16`) | **exact** | yes | already a special case of `quantize.py` block FP | already-in-bcir |
| A\*/Dijkstra/JPS/JPS+/goal-bounding | **exact** | yes (provably optimal-cost) | tropical (min,+) search K_BCIR already runs | build-exact (extend solver) |
| Flow-field integration (discrete Eikonal) | **exact** | yes (the integration; steering is out of scope) | GEM stencil kernel | build-exact |
| SoA↔AoS reindexing | **exact** | yes (`LayoutCertificate`) | `layout.py` priced pivot | already-in-bcir |
| Roaring bitsets / Doom solidsegs | **exact** | yes (lossless) | `_BitInt(N)` packed sets, priced pivot | build-exact |
| Seed + integer recurrence (Elite Tribonacci) | **exact** | yes | provenance/replay (`ProvenanceManifest`) | already-in-bcir |
| `fork()`+COW page snapshot | **exact** | yes (a page snapshot *is* the memory) | provenance/replay + BCIR-IPC (physical) | build-exact |
| **Quake fast inverse sqrt** `0x5f3759df` + 1 Newton step | **approximate** | **NO** (~0.175 % ≈ 10³–10⁴ ULPs) | do-not-import literal; PATTERN behind a **new** rel-error contract, two-truth-quarantined | build-approx-under-a-new-contract |
| bit-log / exp2 / pow (fast-rsqrt parent family) | **approximate** | NO | same new rel-error contract | build-approx |
| HPA\* / navmesh polygonization | **approximate (combinatorial)** | NO (path-**value** divergence) | **do-not-import** (R17 cannot bound); at most a cost-side heuristic verified vs exact search | do-not-import |
| MegaTexture HD-Photo→DXT transcode | **lossy** | NO (orders beyond ≤1 ULP) | **do-not-import** into exact core; use `quantize.py` bounded compression instead | do-not-import |
| Float lockstep (AoE II HD FPU-pin) | **approximate** | NO (fragile per-platform) | float generator must never issue a verdict (two-truth) | do-not-import (in the legality path) |

**Why the fast-rsqrt PATTERN, not the literal.** The one-step Quake sequence has a proven **peak
relative error of 1.752 × 10⁻³ (0.175 %)** — roughly 2⁻⁹·², i.e. **10³–10⁴ ULPs** for float32
operands near 1.0. R17 / `precision.py` is a **Q8-integer, provably ≤1 ULP** law that **explicitly
excludes** Newton-Raphson and any float rounding model, so **fast-rsqrt does NOT fit R17** — wiring
it there would silently corrupt the accuracy invariant. It needs a **NEW certified max-relative-error
(float) contract kind**, separate from R17's ≤1-ULP instance. And separately: the **resident-compiler
gate already lowers `1/sqrt` to hardware `rsqrtss`/`vrsqrtps` via clang** — faster *and* more accurate
than the 1990s hack — so the **literal `0x5f3759df` is do-not-import**; only the **pattern** (an
approx op behind a rel-error contract, quarantined from legality) is worth building.

---

## 2. The overlap — what BCIR already embodies (map, don't re-build)

Most of the *exact* game lore is **already** BCIR machinery. Importing it would re-derive what
exists; the honest move is to **map** it, cite the surface, and reserve build effort for the real
gaps. Anchored surfaces:

| Game technique | BCIR surface (already built) | Anchor |
|---|---|---|
| Demoscene/console cycle-budget accounting (every op priced, "hot placement costs more") | 12-d integer `CostVector` + `couple`/`dot` (no floats in the hot path = deterministic lockstep accounting) | `bcir/kbcir/cost.py:44`, `:67`, `:71` |
| Console/streaming memory pyramid (VRAM→RAM→disc), LOD budget | `MemoryHierarchy` L1/L2/L3/DRAM/HBM/CXL/SSD tiers, `bw_factor`/`lat_factor` | `cost.py:154`, `:166`, `:141` |
| Offline shader/permutation baking → cheapest legal variant | `realize.candidates_for` + `optimize()` min-plus shortest path over a layered candidate DAG | `realize.py:137`, `:321`, `:356` |
| Draw-call batching / render-pass fusion (never recompute, never round-trip) | `fused_candidates` value-numbered CSE + producer→consumer deforestation | `realize.py:263`, `:251`, `:202` |
| "Sort particles by cell, then sweep" (kill the random scatter) | `reduce.gather` blocked realization beats scatter penalty; HAM O(log n) | `realize.py:159`, `:151` |
| Bit-squashing / block-compressed assets (BC/DXT, shared-exponent) | `quantize_group` `_BitInt(N)` block FP, dequant = shift | `quantize.py:73`, `:63`, `:50` |
| Deterministic fixed-point lockstep MACs (RTS sim / replay) | `integer_dot` exact accumulation + provably-wide `accumulator_bits` | `quantize.py:131`, `:159`, `:150` |
| The fast-rsqrt accuracy **contract** (bounded, known worst-case) | R17 `precision.py` Q8-ULP static bounds + compensated (Kahan) reduce + `meets_tolerance` | `precision.py:100`, `:109`, `:147`, `:129` |
| Data-Oriented Design SoA/AoS cache packing (Mike Acton / ECS) | `layout.py` priced SoA↔AoS pivot + bit-exact `LayoutCertificate` | `layout.py:206`, `:183`, `:140` |
| Learned move-ordering + opening book / transposition table + alpha-beta cutoff | `FrozenTilePrior` guided search with admissible provable early-exit | `tile_prior.py:147`, `:85`, `:163` |
| COW savestate generation counters; cooked/baked asset bundle version stamps | StreamPack hot artifact + `topo_gen`/`map_gen`/`data_gen` + `provenance_ok` | `gem/streampack.py:66`, `:72`, `:78` |
| Versioned save-file / demo-replay binary format with CRC (cartridge saves, Quake `.dem`) | StreamPack binary ABI: frozen v1 + append-only v2/v3, CRC-32 trailer, lowest-carrying-version encode | `abi/streampack_abi.py:128`, `:186`, `:197` |
| Frame job-graph / render-graph scheduler (fiber jobs) with roofline cap | Duration-aware EFT waves + token-DAG pipelining + bandwidth-knee clamp + locality affinity | `gem/schedule.py:118`, `:157`, `:63`, `:79` |
| Double buffering / page-flip + prefetch-ahead streaming | `hydrate_pipelined` double-buffer prefetch (`buffers=2`) | `gem/streampack.py:111`, `:135` |
| Virtual-memory paging / megatexture world streaming | `PagedKV` page table over registry Resources, per-page `data_gen`, capacity law + eviction with use-after-free guard | `frontends/models/paged_kv.py:64`, `:103`, `:119` |
| DMA/blitter chains (Amiga blitter, PS2 VIF/DMA) | `dma_descriptors` strided-view pair → scatter-gather ring, unit-stride coalescing, priced fragmentation | `kbcir/dma.py:82`, `:51`, `:126` |
| Fixed console/platform capability manifest (VRAM tiling/swizzle constraints, veto-not-steer probes) | `DeviceManifest` static schema, `StridedView` as the only allocation currency, native-tile refusal at plan time | `kbcir/device_manifest.py:60`, `:194`, `:234` |
| Interrupt-driven console I/O (raster/HBlank IRQ, DMA-complete IRQ, audio refill) | A1/B1 event phases: async entry, explicit arm/mask claims, interrupt-context ordering seam | `kbcir/events.py:63`, `:41`, `:48` |
| Seeded procedural gen + deterministic replay ("the commit hash of a plan") | `ProvenanceManifest` + `replay` (content-hash of module/target/theta/policy/artifact-gens) | `kbcir/provenance.py:96`, `:200`, `:110` |
| Server-authoritative state vs client-side prediction (cosmetic informs, never decides) | two-truth quarantine: classical legality vs graded confidence, crossed only by a recorded `decide()` | `kbcir/twotruth.py:99`, `:119`, `:55` |
| Peephole/algebraic strength reduction; LOD-swap to a cheaper equivalent subgraph | e-graph equality saturation + rewrites; `ResidentEGraph` re-extract pivot — **the host for the bit-shift rewrite** | `kbcir/egraph.py:319`, `:254`, `:406`, `:316` |

**The six genuine gaps** (everything else above is already built): (1) the **bit-shift
strength-reduction rewrite** — `egraph.py:316` `DEFAULT_RULES` and `realize.py:137` are the hosts,
the rule and the "shift" opclass are absent; (2) a **seeded metaprogramming generator** — the
plan→emit path is fixed per plan, a `(seed, generator)→kernel-family` replayable descriptor is
absent; (3) **bit-packed frontier sets** over the scheduler/e-graph worklists — the bit-pack
primitive exists (`quantize.py`), the bitset frontier data structure does not; (4) an
**approximate-op class** admitted only under a certified tolerance gate — R17 exists, the candidate
that spends it does not; (5) the **physical `fork()`/COW snapshot verb** — the generation tags and
stale-detection exist, the fork-and-reconcile verb does not; (6) **extreme-low-memory profiles** —
the `TargetProfile` factory container (`cost.py:219`) is deliberately open, the tiny MCU profile +
capacity-pressure policy are not present.

---

## 3. Per-game principles — the full ledger

Each row: the game technique, the durable lesson, the BCIR mechanism it maps to, the target
subsystem, and an honest status (`already-in-bcir` / `build-exact` / `build-approx-under-a-contract`
/ `do-not-import`).

### 3.1 Bit-level / floating-point (the exact core, and the one approximation)

| # | Game / technique | Lesson | BCIR mechanism | Target | Status |
|---|---|---|---|---|---|
| 1 | **Doom (1993)** 16.16 fixed-point (`fixed_t`, `FixedMul`) | Power-of-two-scaled block integers give deterministic, cross-CPU bit-reproducible math; `FixedMul = (a*b)>>16` is accumulate-then-shift | `quantize.py` per-group 2^scale signed `_BitInt(N)` with exact-integer accumulate-then-shift is a **superset** of 16.16; truncation is R17's ≤1 ULP | quantize/precision + provenance/replay | already-in-bcir |
| 2 | **IEEE-754** `x*2^k` / `x/2^k` | `x*2^k` = shift (int) or exponent-add (float); bit-exact within range | New value-invariant `Candidate` in `realize.candidates_for` priced lower on COMPUTE; `rw_strength_reduce` in `egraph.py DEFAULT_RULES`; clang folds to `shr`/`lea`/`scalbn` | K_BCIR (egraph + realize) | build-exact |
| 3 | **IEEE-754** signed `x/2^k` — *the value-invariance trap* | Signed `x/2^k` is **NOT** `x>>k`: `>>` rounds to −∞, C division truncates to zero; the exact rewrite adds `((1<<k)-1)` bias for negatives | The canonical two-truth test: the cheaper naive-shift candidate is **illegal regardless of cost**, and the bit-identity proof MUST reject it. Only *unsigned* div-by-2ᵏ is a plain shift | K_BCIR | build-exact |
| 4 | **IEEE-754** sign/abs/copysign, `frexp`/`ldexp`, total-order compare | Sign-bit XOR/AND masks, exponent-mantissa split, and the total-order key flip `bits ^ (0x80000000 | -(bits>>31))` are bit-identical and unlock **radix sort of float resources** | Strength-reduction rewrites emitted as C23 `bit_cast`+mask (clang → `andps`/`xorps`); ordered-compare makes a float field radix-sortable via `layout.py`. Caveats carried in the proof: NaN unordered, −0.0 ≠ +0.0 as ints | K_BCIR + layout | build-exact |
| 5 | **Quake III (1999)** fast inverse square root | **APPROXIMATE** (~0.175 % peak after one Newton step); can NEVER be value-invariant and does NOT fit R17's ≤1-ULP law; clang already lowers `1/sqrt` to `rsqrtss` (faster *and* more accurate) | Do-not-import the literal `0x5f3759df`. Only the **pattern** survives: an approx op behind a **new** certified max-relative-error (float) contract feeding `cost.accuracy`, two-truth-quarantined. Myth-flag: not Carmack's (Tarolli/Walsh/Kahan-Ng) | quantize/precision (new contract kind) | do-not-import literal; build-approx-under-a-new-contract for the pattern |

### 3.2 Spatial structure & pathfinding (exact search is *already* K_BCIR's tropical solver)

| # | Game / technique | Lesson | BCIR mechanism | Target | Status |
|---|---|---|---|---|---|
| 6 | **Doom BSP** baked into the WAD `NODES` lump | Build a spatial partition **offline once**, traverse cheap; exact front-to-back order, zero runtime sort. (O(log n) is **point-location only**; full-scene traversal is linear in polys) | GEM's phase DAG is already a precomputed topo-ordered partition — the structural twin. A baked index becomes a frozen StreamPack (gen tags + CRC + `ProvenanceManifest`); an edit invalidates it like R11 stale-pack. `realize.py` already turns HAM random access into `ceil(log2 n)` point-location | StreamPack + device-manifest/HAM | already-in-bcir (map, don't build) |
| 7 | **A\*/Dijkstra/JPS/JPS+/Goal-Bounding** | Exact (min,+) shortest path with an admissible lower-bound heuristic for pruning; all return the provably optimal-**cost** path | **This IS** the tropical search K_BCIR runs — `realize.py` π\* = argmin via `semiring.py::dag_shortest_path` ("agrees with Dijkstra"). Gap: it handles only a **layered** DAG; nav graphs are cyclic, so add a PQ frontier + bit-packed closed set. A\*'s admissible `h(n)` prunes `realize.py`'s own candidate search (two-truth: prunes cost only) | K_BCIR | build-exact (extend the existing solver) |
| 8 | **HPA\* / navmesh** (Recast/Detour) | **APPROXIMATE** by construction (one transition point per entrance; polygonized walkable space) — changes the path **value** | Violates value-invariance → not a legal realization. Error is **combinatorial**, not ULP-numeric, so R17 cannot certify it either. At most a cost-side planning heuristic verified against the exact search — never the verdict | K_BCIR | do-not-import |
| 9 | **Flow-field RTS** (Supreme Commander 2) | One exact Dijkstra/BFS integration field (discrete Eikonal) built once, read O(1) per agent — compute-heavy build amortized over thousands of agents | The one place pathfinding is a **kernel**, not a search: a wavefront/stencil pass on the phase DAG + A2 DMA rings. Integration is exact; downstream steering interpolation is approximate and **out of BCIR scope** | GEM | build-exact |

### 3.3 Procedural generation / metaprogramming from a seed

| # | Game / technique | Lesson | BCIR mechanism | Target | Status |
|---|---|---|---|---|---|
| 10 | **Elite (1984) / Frontier** — seed + deterministic recurrence | Store a tiny seed + generator, regenerate on demand; Elite's Tribonacci twist (`s2' = s0+s1+s2 mod 2^16`) + 8-bit ROL is **integer-exact**. This is deterministic **reproducibility** | `provenance.py` `ProvenanceManifest` is "the commit hash of a plan" — stores only a 63-bit FNV digest + 4 component hashes + score + widths; `replay()` re-runs `optimize()` and proves bit-identity (R13). **Honesty:** BCIR is zero-storage-**reproducible**, not zero-storage — the StreamPack is materialized by design | provenance/replay | already-in-bcir (the exemplar) |
| 11 | **Stellar Forge / No Man's Sky** — float generators | On-demand "what exists here?" but **float** physics/noise is deterministic per-platform only, not cross-arch bit-identical; "zero storage / perfect determinism" is a storage claim, and the NMS superformula is disputed | Admissible only as cost/telemetry or design analogy; a float generator must **never** issue a reproducibility verdict (two-truth). Coordinate-seeded lazy KV/tile regen is build-approx-under-R17 only if the page generator is an **integer** recurrence | provenance/replay + GEM | do-not-import (float determinism in the legality path) |
| 12 | **.kkrieger / fr-08** — store the creation history | Persist the operator DAG (recipe), regenerate ~200 MB assets from ~96 KB. Individual float ops are per-platform, but "store the generator" is the durable idea | Already BCIR's content-addressed claim/rewrite DAG + e-graph CSE + `ProvenanceManifest`. **To-build:** a registered `(seed, generator)→Module` descriptor whose identity **is** the module hash, seeding specialized C23/LLVM kernel families replayable bit-for-bit. kkrunchy-style opaque packer is **do-not-import** (defeats provenance transparency) | StreamPack + GraphSeed generator | already-in-bcir (architecture); build-exact (formalized GraphSeed) |

### 3.4 Memory frugality & streaming (Crash Bandicoot's real lesson)

| # | Game / technique | Lesson | BCIR mechanism | Target | Status |
|---|---|---|---|---|---|
| 13 | **Crash Bandicoot (1996)** — 64K page streaming + opcode DMA | **EXACT** demand-driven data movement + an **APPROXIMATE** locality *prediction*; a mispredict costs a stall, never a wrong value. (Myth: it was **software-managed streaming**, NOT hardware MMU paging) | `paged_kv.py` `PagedKV` is already a page table proven bit-for-bit vs the non-paged reference; A2 DMA descriptor rings (`dma.py`) **are** Crash's opcode DMA made IR-native (one atomic MMIO doorbell, completion = A1 event) — the honest replacement for hand-DMA under the resident-compiler gate | GEM | already-in-bcir (mechanism); build-exact (generalize PagedKV→PagedResource) |
| 14 | **RAGE / id Tech 5 MegaTexture** | **EXACT** sub-mechanisms (page-table indirection, feedback-driven residency, tile transport) but **LOSSY** parts: HD-Photo→DXT transcode and mip-LOD popping. "Unlimited perfect detail" is false at the mechanism level | Page-table + residency map onto PagedResource + the tiered `MemoryHierarchy` (SSD "semantic-swap", CXL). The lossy transcode is orders beyond ≤1 ULP — **do-not-import** into the exact core; use `quantize.py` per-group packing (build-approx-under-R17) for bounded compression instead | GEM + quantize/precision | build-exact (indirection/residency); do-not-import (lossy transcode) |
| 15 | **Factorio** — software prefetch over fat AoS objects | **HEADLINE win is SOFTWARE PREFETCH** (−128..+384 B = 8 cache lines) over AoS entities **larger than a cache line** — 9–13 % UPS. **NOT** the SoA packing commonly assumed | `gem/streampack.py` `Prefetch` records (`double_buffer`, `buffers=2`) + `pipeline_depth` cover the mechanism; **missing** is a cost-priced prefetch-**distance** auto-tuner emitted as `__builtin_prefetch` (resident-compiler gate, no asm). Perf-only telemetry, trivially value-invariant | GEM | already-in-bcir (double-buffer); build-exact (distance auto-tuner) |
| 16 | **Factorio** — `fork()`+COW background save | `fork()` gives the child a consistent frozen snapshot; the kernel COW-copies only mutated 4096-B pages while the parent keeps simulating — zero micro-stutter, bit-identical | **SPLIT:** the logical versioned-snapshot **identity** is already StreamPack `topo/map/data_gen` + R11 stale + R13 replay (staleness detection, **orthogonal to COW**). The **physical** `fork()`/`madvise` page-COW of the resident registry at a generation boundary is the genuine to-build (libc syscall, Linux/macOS, needs spare RAM). Do **NOT** lean on the unverified "wave-15 Linux Master Kernel fork peer" | provenance/replay + BCIR-IPC | already-in-bcir (logical identity); build-exact (physical fork/COW) |

### 3.5 Data layout, determinism & set representation

| # | Game / technique | Lesson | BCIR mechanism | Target | Status |
|---|---|---|---|---|---|
| 17 | **Mike Acton DOD / ECS** — SoA↔AoS | Single-field sweep wants SoA (unit-stride, vectorizable); whole-record wants AoS; the layout is a value-invariant reindexing that should be **priced, not asserted** | `layout.py` already prices both poles through the SAME `realize` stride/memory terms, adopts AoS only on a strict win, and emits a `LayoutCertificate` proving SoA == AoS bit-exact. **Honesty:** there is **no standalone "cacheline" dimension** in the 12-d `CostVector` — locality lives inside the MEMORY axis (`min(stride_k, cacheline/elem_bytes)`) | layout + K_BCIR | already-in-bcir |
| 18 | **StarCraft / AoE** — deterministic lockstep | Fixed-point integer sim is genuinely bit-exact and portable; float lockstep (AoE II HD FPU-pinning) is fragile quasi-determinism. Send commands, checksum state, replay bit-for-bit | `cost.py`'s "no floats in the hot path" invariant + `quantize` `integer_dot` exact accumulation + StreamPack CRC/gen tags/R13 replay **are** the checksum-and-replay machinery. The command-stream **transport** itself is do-not-import (netcode, not an IR mechanism) | quantize/precision + provenance/replay | already-in-bcir |
| 19 | **Roaring bitmaps / Doom `solidsegs`** | Bit-packed sets with native AND/OR/ANDNOT; per-2¹⁶-chunk array/bitmap/run container chosen by density; lossless/exact | Frontier/closed/occupancy sets pack via `quantize` `_BitInt(N)`; the hybrid container choice is a value-invariant representation pivot priced like `layout.py`'s SoA/AoS (`LayoutCertificate`). Emit word-parallel ops as IR, hand popcount/vector isel to clang | quantize/precision + layout | build-exact |

### 3.6 The correctness route (why "embrace assembly" is an anti-pattern here)

| # | Game / technique | Lesson | BCIR mechanism | Target | Status |
|---|---|---|---|---|---|
| 20 | **RollerCoaster Tycoon (1999)** — ~99 % x86 asm | Stability came from **determinism + a frozen artifact**, not from assembly; "bug-free / never-patched" is myth (patches to v1.08.187 / v1.10.026) | Resident-compiler gate: BCIR emits C23/LLVM IR and hands isel to clang/llc — **never hand-rolls asm**. The determinism/frugality **spirit** is already K_BCIR strength-reduction + `quantize` `_BitInt(N)` + `layout` SoA/AoS. The method is an explicit **anti-pattern** | other (reframe) | do-not-import (method); already-in-bcir (spirit) |
| 21 | **seL4 (SOSP 2009)** — machine-checked correctness | The principled route to "stable / no-patching" is **proof**, not asm; seL4 proved correctness **without owning isel** (base proof trusted the C compiler). TCB = ~600 asm lines + boot + hardware + compiler | BCIR's value-invariance proofs are seL4-style refinement at the realization level; R1–R23 laws + two-truth are the guard. "Bug-free" = correct-relative-to-spec **modulo a declared TCB**. seL4 later closed the compiler gap via translation validation (PLDI 2013) — the model for a future BCIR IR→machine-code validation pass | other (verification rails) | build-exact (proof-carrying realization discipline) |
| 22 | **SQLite / TigerBeetle / NASA Power of Ten** | Exhaustive testing (100 % MC/DC + differential + fuzz), static allocation + deterministic simulation testing (VOPR), and static-analyzability rules are what asm was a crude proxy for | `bcir.kbcir.differential` already does Python↔MLIR parity; extend with MC/DC gating + an IR/plan mutation fuzzer (replay-gated). Plan-time worst-case sizing freezes HAM/StreamPack (no runtime alloc). A VOPR-style fault injector over the GEM scheduler/DMA rings checks replay stays bit-identical. Codify Power-of-Ten as explicit R-laws | provenance/replay + GEM + device-manifest/HAM | build-exact |
| 23 | **Kaze SM64** — smaller code on a memory-bound target | On a memory-bound target, prefer **smaller** code (inlining/unrolling **hurt** via icache misses); fixed-point sine LUT over float trig | Surface an icache/code-size cost dimension so the tropical optimizer prefers smaller code when memory/fabric terms dominate — turns a hand-won heuristic into a **priced** decision. LUT memoization = `realize.py` materialize-vs-recompute (exact for enumerable domains; a transcendental LUT is build-approx-under-R17) | K_BCIR | build-exact |

---

## 4. Lessons applied to **GEM** (the StreamPack hot path)

GEM already owns the streaming/scheduling machinery the console/streaming games pioneered. The
imports deepen it; none touches numerics.

1. **A2 DMA descriptor rings ARE Crash Bandicoot's opcode-level DMA, made IR-native and law-clean**
   (`dma.py:82`). Strided-view pairs compile to a scatter-gather ring, unit-stride runs coalesce
   (contiguous copy = **1** descriptor), fragmentation is priced per-descriptor, the doorbell is one
   atomic volatile MMIO write, and completion is an A1 event phase — the honest resident-compiler-gate
   replacement for Andy Gavin's hand-written MIPS. *(already-in-bcir)*
2. **Generalize rung-7 `PagedKV` into a general `PagedResource` streaming store** — a page table of
   registry Resources over a `DeviceManifest` bank, construction-time D-R4 refusal, capacity law,
   `data_gen`-per-write, eviction = `map_gen` bump + view free, admission = append phases — for **any**
   large array streamed from a slow tier (Crash 64K pages / id Tech tile cache). Value-invariance holds
   by construction: **paging is a registry story, not a math story.** *(build-exact refactor)*
3. **A flow-field integration kernel** is the one place a pathfinding op is a **kernel**, not a search:
   one exact Dijkstra/BFS wavefront (discrete Eikonal) built as a stencil pass on the phase DAG + A2
   rings, then read O(1) per agent. *(build-exact)*
4. **Deepen the `gem.prefetch` double-buffer contract** (`hydrate_pipelined`: `buffers=2`,
   `pattern=double_buffer`, `pipeline_depth`) into a configurable N-deep demand-streaming ring pointing
   at the SSD/CXL tier — append-only and provably numerically inert (the calling side proves "a prefetch
   hint changes nothing numerically"), so the overlap win is **pure schedule**. *(build-exact)*
5. **Add a cost-priced prefetch-DISTANCE / lookahead auto-tuner** emitted as `__builtin_prefetch` — this
   is Factorio's **real** headline cache win (software prefetch over fat AoS, −128..+384 B window,
   9–13 % UPS), which is **NOT** the SoA packing `layout.py` already covers. Perf-only telemetry, never
   touches legality. *(build-exact)*
6. **Stand up a TigerBeetle VOPR-style deterministic fault-injecting simulator** over the GEM phase DAG,
   A1/B1 event phases, A2 DMA rings, and paged-KV streaming, checking replay stays bit-identical under
   injected sync/fabric/storage faults; fault outcomes **inform, never legislate** (two-truth).
   *(build-exact)*

---

## 5. Lessons applied to **K_BCIR** (the tropical cost model, e-graph, realization search)

The deepest realization of this whole exercise: **most game "AI" search is already K_BCIR's tropical
(min,+) machinery.** The imports extend the *same* solver — they do not add a new subsystem.

1. **Pathfinding is NOT a new subsystem** — it is the (min,+) shortest path K_BCIR already runs.
   `realize.py` π\* = argmin over the candidate DAG via `semiring.py::dag_shortest_path` ("agrees with
   Dijkstra"). The only gap: that solver handles a **layered** DAG; navigation graphs are cyclic, so
   extend the same semiring with a **priority-queue frontier + bit-packed closed set** (general
   Dijkstra/A\*), still pure min-plus, still value-invariant. *(build-exact)*
2. **A\*'s admissible heuristic `h(n)`** is a never-overestimating lower bound on the remaining scalarized
   `CostVector`; adding one lets `realize.py` skip provably-dominated branches **without changing π\***
   (Hart-Nilsson-Raphael optimality) — a genuine speedup of K_BCIR's **own** search that respects
   two-truth (prunes cost search only). *(build-exact)*
3. **Add `rw_strength_reduce` to `egraph.py DEFAULT_RULES` and a cheaper "shift" candidate in
   `realize.candidates_for`:** `x*2^k → shift`/exponent-add, sign/abs → mask, float total-order flip →
   integer compare. Every rewrite carries a **bit-identity proof**; clang does final isel; overflow/
   subnormal boundaries are legality **side-conditions**. *(build-exact)*
4. **The signed div-by-2ᵏ case is the canonical value-invariance / two-truth demonstration:** the naive
   `x>>k` candidate is **illegal** for negative `x` (`>>` rounds to −∞, C division truncates to zero)
   **regardless of how cheap it is** — the exact rewrite must add the `((1<<k)-1)` bias, and the proof
   MUST reject the cheaper-but-wrong candidate. Only *unsigned* div-by-2ᵏ is a plain shift. *(build-exact)*
5. **Represent the scheduler's ready/visited/frontier claim sets and `egraph.saturate`'s worklist as
   packed `_BitInt`/bitword sets** (the chess-bitboard / BFS-frontier trick) reusing the `quantize.py`
   bit-packing lane, with **Roaring-style** per-chunk array/bitmap/run container selection priced like
   `layout.py`'s pivot — replacing Python `set()` churn in the hot planning loop. *(build-exact)*
6. **Surface an icache/code-size cost dimension** so the tropical optimizer can pick smaller code (no
   unroll/inline) when the memory/fabric terms dominate — Kaze's N64 "smaller beats faster on a
   memory-bound target" turned into a priced decision. **Honesty:** there is no standalone "cacheline"
   axis among the 12 dims (compute/memory/fabric/sync/compile/thermal/power/reliability/security/
   accuracy/contention/verification); cache locality lives inside the MEMORY axis. *(build-exact)*
7. **Carmack surface-caching/PVS and LUT memoization ARE `realize.py` materialize-vs-recompute + e-graph
   CSE** choosing to materialize when the memory term beats recompute — exact for integer/enumerable
   domains; a transcendental LUT is instead build-approx-under-R17. *(build-exact)*

---

## 6. Lessons applied to the **StreamPack ABI** (frozen artifact, generation tags, provenance/replay)

The procedural-generation and save-file games map onto the StreamPack's frozen-artifact discipline —
but with two honesty corrections carried in bold.

1. **`ProvenanceManifest` + `replay()` IS Elite's store-seed-regenerate-artifact at compiler-plan
   scope:** it stores only a 63-bit FNV digest + 4 component hashes + score + widths and **never** the
   realized pack; `replay` re-runs `optimize()` and proves bit-exact identity (R13). **Honesty
   correction:** this is deterministic **reproducibility**, not zero-storage — the StreamPack is a
   materialized frozen hot artifact **by design**. An Elite-style zero-storage mode (regenerate the pack
   on demand, never materialize it) is a **distinct design point** BCIR has the machinery for but does
   not currently embody. *(already-in-bcir)*
2. **Baked spatial indices** — Doom's BSP in the WAD `NODES` lump, JPS+ jump-distance maps, Recast
   navmesh, goal-bounding boxes — carry as **frozen StreamPacks** with `topo_gen`/`map_gen`/`data_gen` +
   CRC + `ProvenanceManifest`; a level/graph edit invalidates exactly like R11 stale-pack and R13 replay
   reproduces identical paths. *(build-exact)*
3. **The LOGICAL half of Factorio's `fork`+COW save already exists:** `topo/map/data_gen` generation tags
   + R11 stale + R13 replay are a snapshot-at-gen-N that stays distinct from the advancing live gen and is
   rehydratable. **Honesty:** generation tags are staleness **IDENTITY, ORTHOGONAL** to COW's lazy
   page-sharing — do **not** claim the tag *is* a COW mechanism. The **physical** zero-stutter
   `fork()`/`madvise` page-COW of the resident registry is the genuine to-build (BCIR-IPC), and the
   "wave-15 Linux Master Kernel fork peer" is **unverified** — do not depend on it. *(already-in-bcir
   logical; build-exact physical)*
4. **StreamPack CRC trailer + append-only versioning + generation tags + R13 replay** are the versioned
   save / demo-replay format (cartridge saves, Quake `.dem`) with byte-identical lowest-carrying-version
   encode and corruption caught by checksum — the modern form of what RCT's asm and AoE's fixed-point
   achieved. *(already-in-bcir)*
5. **Formalize a registered `(seed, generator) → Module` GraphSeed descriptor** whose identity **is** the
   module hash, seeding specialized C23/LLVM kernel families replayable bit-for-bit (the .kkrieger "store
   the recipe, not the bytes" idea, made integer-exact). Do **NOT** import opaque exe-crunchers
   (kkrunchy/UPX) — they defeat CRC/generation-tag provenance transparency. *(build-exact generator;
   do-not-import packer)*

---

## 7. Ranked build slices

Effort: **S** = small/self-contained (reuse existing engine), **M** = medium, **L** = large. Every
slice is value-invariant or two-truth-quarantined; none violates the resident-compiler gate.

| # | Slice | Target | Effort | Goal |
|---|---|---|---|---|
| **G1** | **Shift/exponent-add strength-reduction rewrite** (with signed-divide bias guard) | K_BCIR (egraph + realize) | **S** | Add `rw_strength_reduce` to `egraph.py DEFAULT_RULES` + a "shift" opclass candidate in `realize.candidates_for` for MUL/DIV-by-2ᵏ, sign/abs masks, and the float total-order flip; each carries a bit-identity proof, and the signed div-by-2ᵏ proof reproduces the `((1<<k)-1)` bias so the naive-shift candidate is **rejected as illegal**. The engine and cost model exist; only the rule + opclass are missing. *(The canonical value-invariance test — build this first.)* |
| **G2** | **Cyclic min-plus pathfinder + admissible-heuristic pruning** | K_BCIR | **M** | Extend `semiring.py::dag_shortest_path` from a layered DAG to a general cyclic graph via a PQ frontier + bit-packed closed set (Dijkstra/A\*/JPS home), and add an admissible cost-lower-bound `h(n)` that prunes `realize.py`'s own candidate search **without changing π\***. Pure min-plus, value-invariant, two-truth-clean. *(Highest leverage — it also speeds up the compiler's own search.)* |
| **G3** | **Bit-packed frontier sets + Roaring container selection** | K_BCIR + layout | **M** | Replace Python `set()` churn in `schedule.py` ready/frontier and `egraph.saturate` worklist with packed `_BitInt`/bitword sets and Roaring-style per-chunk array/bitmap/run container choice priced like the SoA/AoS pivot; word-parallel ops emitted as IR, popcount isel to clang. Bit-exact membership. |
| **G4** | **Generalize `PagedKV → PagedResource` + demand-fetch/spill scheduler** | GEM + device-manifest/HAM | **L** | Lift `PagedKV`'s KV-specific numerics into a general streaming store for any large array backed by a slow tier, and add a residency/spill policy over the tiered `MemoryHierarchy` (SSD "semantic-swap" / CXL) under `Theta.mem_pressure`, every move priced by the distance matrix. Data moved is exact; policy cost is two-truth-quarantined. *(Flagship GEM slice.)* |
| **G5** | **Prefetch-distance auto-tuner** (Factorio's real cache win) | GEM | **S** | Cost-price a per-stream prefetch distance/degree, emitted as `__builtin_prefetch` (resident-compiler gate, no asm), and deepen `hydrate_pipelined` into an N-deep ring from the SSD/CXL tier. Perf-only, numerically inert, trivially value-invariant. |
| **G6** | **Physical `fork()`+COW background-snapshot API** | provenance/replay + BCIR-IPC | **M** | A `snapshot()`/`fork()` verb that `fork()`/`madvise` page-COW-clones the resident registry at a generation boundary so a background re-plan/save/verify runs on a frozen gen-N snapshot while the live registry advances to N+1, reconciled by R11 generation compare. Bit-identical (a page snapshot **is** the memory); Linux/macOS, needs spare RAM. Scope strictly to `fork()`/`madvise` over the registry — **drop** the unverified "Linux Master Kernel peer" framing. |
| **G7** | **Seeded GraphSeed metaprogramming generator via replay** | StreamPack + K_BCIR | **M** | Formalize a registered `(seed, generator) → Module` descriptor whose identity is the module hash, using a `ProvenanceManifest` digest as the deterministic seed to emit specialized C23/LLVM kernel families replayable bit-for-bit through `replay()`. Integer-exact; behind the resident-compiler gate. Do not import opaque packers. |
| **G8** | **Relative-error approximate-op contract** (the fast-rsqrt PATTERN, not the literal) | quantize/precision | **M** | Add a **NEW** certified max-relative-error (float) accuracy-contract kind — **distinct** from R17's Q8 ≤1-ULP integer law — plus an APPROXIMATE candidate class in `realize.candidates_for` admitted **only** when the contract certifies the claim's declared tolerance, charging the accuracy axis and strictly quarantined from legality by two-truth. Build only if this contract type is added; the literal `0x5f3759df` stays do-not-import (clang `rsqrtss` is faster and more accurate). *(Ship a design doc before any code.)* |
| **G9** | **Extreme-low-memory / edge profiles + static residency plan** | device-manifest/HAM + other | **M** | Add tiny `TargetProfile` factories and `DeviceManifest` banks for MCU/embedded targets (small L1, no HBM, NVM-swap dominant), a memory-first `Policy` weighting capacity/memory axes, and a compile-time static overlay/residency schedule + arena over a fixed bank (the bank-switching ancestor) so there is **zero runtime allocation** — lowered to C23/LLVM for the MCU. Bit-identical to the fully-resident reference. |
| **G10** | **seL4/SQLite/TigerBeetle verification rail** (the reframe of "embrace asm") | provenance/replay + device-manifest/HAM + other | **L** | Elevate every realization pass to carry a machine-checkable equality obligation to its reference (seL4-style refinement); extend `bcir.kbcir.differential` with MC/DC coverage gating + an IR/plan mutation fuzzer (replay-gated); add plan-time worst-case static sizing so the hot rail has zero dynamic allocation; codify Power-of-Ten as explicit R-laws. **This — not hand-rolled asm — is the route to "stable / low-patch."** |
| **G11** | **Baked-index StreamPack + offline-partition wiring** | StreamPack + GEM | **S** | Wrap baked spatial indices (BSP/navmesh/JPS+ jump maps/goal-bounding boxes) as frozen StreamPacks with generation tags + CRC + `ProvenanceManifest` and map the BSP-as-precomputed-partition onto the existing phase DAG (a **mapping, not a build**, since GEM's DAG is already a topo-ordered partition and HAM already gives O(log n) point-location). |

### Sequencing

1. **Land the S-effort, self-contained exact wins first** — **G1** (shift/exponent-add strength
   reduction, with the signed-divide bias guard as the headline value-invariance test) and **G5**
   (prefetch-distance auto-tuner). Both reuse existing engines and prove the exact-family thesis
   end-to-end.
2. **Build G2 next** (cyclic min-plus pathfinder + admissible-heuristic pruning), since it doubles as a
   speedup for `realize.py`'s **own** candidate search — highest leverage because it improves the
   compiler *while* adding the A\*/Dijkstra/JPS capability.
3. **Specify G8's new float relative-error contract as a design doc BEFORE any code**, explicitly
   separated from R17's Q8-ULP law, with the fast-rsqrt literal marked do-not-import and only the
   quarantined pattern buildable — get sign-off that it never touches the legality verdict.
4. **Prototype G6** scoped strictly to `fork()`/`madvise` over the registry + R11 generation reconcile
   (drop the "Linux Master Kernel peer" framing); measure zero-stutter background re-plan on a Linux
   target with spare RAM.
5. **Generalize G4** (`PagedKV → PagedResource` + demand-fetch/spill) as the flagship L-effort GEM slice,
   then wire **G11** (baked spatial indices as frozen StreamPacks) on top of it.
6. **Stand up G10 incrementally:** MC/DC coverage gating + IR/plan mutation fuzzer in
   `bcir.kbcir.differential` first, then the VOPR-style fault injector over the GEM scheduler/DMA rings,
   then plan-time static worst-case sizing.
7. **Add an explicit rounding-convention + float-domain side-condition field** to `LayoutCertificate` /
   value-invariance proofs so exact rewrites carry their overflow/subnormal/signed-shift boundaries — a
   **prerequisite for merging any G1 strength-reduction candidate**.

---

## 8. Risks & myth-flags

Each is a real failure mode the adversarial verification pass surfaced. Every one is a way a naive
import would silently break a BCIR guarantee.

1. **Contract-kind sprawl.** Fast-rsqrt needs a genuinely **new** float relative-error contract, **NOT**
   R17's Q8 ≤1-ULP law (0.175 % ≈ 10³–10⁴ ULPs). Wiring approximate ops to R17 directly **silently
   corrupts the accuracy invariant**; the new contract must be a separate, explicitly-quarantined kind or
   the item stays do-not-import.
2. **Signed-shift correctness trap.** Shipping `x>>k` for signed divide-by-2ᵏ is an off-by-one for every
   negative dividend (`>>` rounds to −∞, C truncates to zero). The bias-correction proof is **mandatory**;
   the naive candidate is the poster child for value-invariance rejecting a cheaper-but-wrong realization.
3. **Float-exactness boundary conditions.** Exponent-add mul/div by 2ᵏ is bit-exact **only within the
   normal range** — overflow to `inf`, gradual underflow into subnormals, and NaN/±0 propagation must
   match spec-conformant `ldexp`, or a naive "increment the exponent field" rewrite is wrong at the
   boundary. The value-invariance proof must carry the **domain side-condition**.
4. **Over-claiming "already in BCIR."** (a) BCIR is zero-storage-**REPRODUCIBLE**, not zero-storage — the
   StreamPack is materialized; (b) generation tags are staleness **IDENTITY, ORTHOGONAL** to COW — the
   physical fork/COW is real to-build; (c) `layout.py` covers only the **layout-choice half** of Factorio,
   **NOT** software prefetch (its actual headline win). Presenting these as done leaves real gaps unbuilt.
5. **Fabricated anchors.** The "wave-15 BCIR-IPC Linux Master Kernel fork peer" has **no basis** in the
   provided BCIR machinery. Building on it risks anchoring the fork/COW slice to a non-existent component;
   scope the slice to `fork()`/`madvise` over the registry + provenance instead.
6. **Combinatorial-vs-numeric confusion.** HPA\*/navmesh error is path-**VALUE** (combinatorial), not
   ULP-numeric, so R17 cannot bound it and build-approx-under-R17 does **NOT** apply — treating them as
   R17-gatable approximations is a category error. They are do-not-import as legal realizations.
7. **Compiler in the TCB.** Like base-2009 seL4, BCIR's value-invariance is proven at the IR/reference
   level and hands isel to clang/llc, so correctness is "up to compiler trust", not down to the metal.
   Marketing "bug-free" without a translation-validation pass (seL4's PLDI-2013 route) overstates the
   guarantee.
8. **Resident-compiler-gate drift.** Any temptation to hand-roll popcount/AVX for frontiers, MIPS-style
   DMA, or the shift/mask sequences **violates the gate** and breaks the value-invariance proof chain;
   everything must stay C23/LLVM IR with isel deferred to clang/llc.
9. **Fixed-point/rounding convention in certificates.** A signed fixed-point right-shift carries the same
   round-toward-neg-inf vs truncation distinction as integer divide; the `LayoutCertificate` /
   value-invariance proof needs an explicit **rounding-convention note** or two "equivalent" realizations
   can disagree by a ULP.

### Myth-flags carried through (historical honesty)

- **Fast inverse square root** was **not** written by John Carmack. He popularized it via the 2005 Quake
  III source release; the lineage traces to **Gary Tarolli** (SGI/3dfx), with roots via **Greg Walsh** and
  earlier Cleve Moler / William Kahan-era numerics. Magic constant `0x5f3759df`; Lomont's slightly better
  `0x5f375a86`. Peak relative error **1.752 × 10⁻³ (0.175 %)** after one Newton step — **approximate, not
  exact.**
- **RollerCoaster Tycoon** was **~99 %** x86 assembly (the last ~1 % was C for the Windows/DirectX
  interface), **not** 100 %, and **did** ship patches (base v1.08.187, expansion v1.10.026) — the
  "bug-free / never-patched" ideal is **myth**. Sawyer's stability came from single-expert authorship +
  determinism + a frozen artifact, not from asm's magic.
- **Crash Bandicoot** used **software-managed streaming**, **not** hardware MMU paging. A locality
  mispredict cost a stall, never a wrong value.
- **BSP** is O(log n) for **point-location only**; full-scene front-to-back traversal is linear in polys.
- **BCIR is zero-storage-REPRODUCIBLE, not zero-storage** — the StreamPack is a materialized frozen hot
  artifact by design. Elite is the exemplar of the *reproducibility* pillar, **not** proof that BCIR is
  storage-free.
- **Factorio's** headline cache win was **software prefetch over fat AoS objects**, not the SoA packing
  commonly assumed; true SoA splitting shows up mainly in the 2024 Space Age work. `layout.py` owns the
  layout-choice half; the prefetch win is a real gap (**G5**).
- There is **no standalone "cacheline" dimension** in the 12-d `CostVector`; locality lives inside the
  **MEMORY** axis.

---

## 9. The bottom line

Game-optimization lore imports into BCIR **only after** the exact-vs-approximate honesty split:

- **Exact** bit/layout/search tricks become value-invariant, cost-priced K_BCIR/GEM/StreamPack
  machinery — and **much of it is already built** (§2). The genuine new exact work is a short list:
  the shift rewrite (**G1**), the cyclic pathfinder (**G2**), bit-packed frontiers (**G3**),
  `PagedResource` (**G4**), the prefetch tuner (**G5**), the physical fork/COW verb (**G6**), the
  GraphSeed generator (**G7**), edge profiles (**G9**), the verification rail (**G10**), and the
  baked-index wiring (**G11**).
- The **approximate** family (fast-rsqrt, HPA\*/navmesh, lossy MegaTexture transcode, float lockstep) is
  either **two-truth-quarantined behind a NEW relative-error contract** (**G8** — the fast-rsqrt
  *pattern*, never the literal) or **do-not-import** — and **never** dressed up as "perfect accuracy."
- "Embrace assembly for stability" (RCT) is an **anti-pattern** under the resident-compiler gate; the
  principled route to "stable / no-patching" is BCIR's **verification rails** — R1–R23, two-truth,
  value-invariance proofs, deterministic provenance — the route **seL4 proved exists** (**G10**), not
  hand-rolled asm.

The user's "binary manipulation for speed **and** perfect accuracy" is honored **exactly** where it is
true (the exact family) and honestly bounded everywhere it is not (the approximate family) — which is the
only way to keep BCIR's one law intact.

---

### Cross-references

- [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) — the authoritative roadmap; this doc is listed in its reference set.
- [`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](BCIR_ML_AI_INTEGRATION_ROADMAP.md) — the process the per-slice build waves land under.
- [`BCIR_WHOLE_MODEL_REFERENCE.md`](BCIR_WHOLE_MODEL_REFERENCE.md) — the rung-8 capstone (same "migrate the idea, wrap it in BCIR's discipline" pattern).
- [`BCIR_STREAMPACK_ABI.md`](BCIR_STREAMPACK_ABI.md) — the frozen binary ABI the StreamPack lessons (§6) build on.
- [`BCIR_LANGREF.md`](BCIR_LANGREF.md) — the R1–R23 law spec (R11 stale-pack, R13 replay, R17 accuracy).
