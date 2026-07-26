# BCIR Language Reference — v0.2.0 (normative)

BCIR is the IR *law*; MLIR is the forge used to express, verify, transform, and
lower that law during bootstrap; LLVM/Clang are backends and interoperability
targets, **not** the conceptual center.

This revision describes package version `0.2.0`. The `0.3b` release notes are an
unreleased draft and do not change this document's version or compatibility claims.

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

The normative form is **constrained series-parallel**:

```
K_BCIR(G | H, Θ) = min_{π ∈ Legal(G,H)}  M(π, Θ)    subject to    R(π, Θ) ⪯ B(H, Θ)
```

- `G` — the goal graph (a BCIR program); `π` — a realization plan (lane/stride
  class, batching, schedule, prefetch).
- `M(π, Θ)` — the **schedule-aware price**: series composition (claims chained
  on one affinity domain, the decoupled GGG tail, successive waves and phases)
  accumulates with (min,+) ⊗; parallel composition (claims co-executing in a
  CT2 wave on distinct domains) combines with **max**. A transition's coupling
  `f_i` applies against its *actual* schedule predecessor — a fusion discount
  belongs only to claims that really run back-to-back. M is (max,+) over the
  wave/token DAG (`gem.overlap.price_scheduled`).
- `R(π, Θ)` — the additive resource ledger, `Σ_i T_i ⊗ f_i(π)` per dimension
  (`T_i` a 12-d cost vector; `⊗` element-wise Q8 coupling, **not** scalar
  multiply).
- `B(H, Θ)` — live budgets (thermal cap, power cap, …), Θ-dependent: a hot
  machine makes wide SIMD *infeasible*, not merely expensive.
- Solved exactly by label dominance over the layered DAG of **legal**
  candidates (RCSP, `kbcir.rcsp.optimize_constrained`); Pareto-optimal plans
  that no weight vector can reach are recovered by `kbcir.rcsp.pareto_plans`.
  All arithmetic is integer/Q8 and deterministic.

**Degenerate case (the scalarized default rail).** With no budgets (B = ∞) and
serial composition (one domain, textual chaining), M reduces to the weighted
sum and selection to the tropical (min,+) shortest path:

```
K_BCIR(G | H, Θ) = min_π  Σ_i  T_i ⊗ f_i(π)  =  min_π  C_H(π, Θ)
```

with `score = w(H,Θ,phase,policy) · (T_i ⊗ f_i(π))`. This degenerate form is
the default rail (`kbcir.realize.optimize`) and the worked-example constants
(vec16, score 7808) are pinned to it; the constrained rail reproduces it
exactly under an unbounded budget. Budget feasibility and scheduled-price
consistency (makespan + overlap_gain = serial) are verifier obligations under
law R9 (`bcir.kbcir.budget`, `bcir.kbcir.scheduled_price`, `-bcir-verify`).

**Temperature (the soft generalization).** The tropical (min,+) selection is the
zero-temperature limit of a log-sum-exp dynamic program:

```
F_T(G | H, Θ) = −T · log Σ_{π ∈ Legal} exp( −score(π) / T )   →   min_π score(π)   as T → 0
```

`F_T` is the Gibbs free energy over realization plans; at `T > 0` it yields a
*posterior over plans* (per-claim marginals, an expected cost vector) and is
**differentiable** — `∂F_T/∂w = E_π[C]`, the expected sufficient statistic — so
the optimizer becomes a learnable layer (`kbcir.softdp`). This is an L2/L3
offline organ (LangRef §13): learning happens at `T > 0`, then anneals and
**freezes** to a `T = 0` integer table for the certified hot path. At `T = 0` it
delegates to `optimize` and is bit-exact; the degenerate-case law (`F_T ≤`
hard score, with equality at `T = 0`) is a verifier obligation under R9
(`bcir.kbcir.soft_select`).

## 3–9. Laws (summary)

- **Module law (§3).** A module is a registry-governed execution universe;
  registries precede claims; plans derive from legal claims; a GEM stream may not
  exist without an originating BCIR plan.
- **Registry-first memory (§4).** Raw pointers are outlawed at the core level.
  `Address = (RID, layout, domain, offset, generation)`.
- **Claim law (§5).** The primitive object is the *claim*, not the instruction:
  `op + resources + contract + phase + cost + verification + ≥1 legal realization`.
- **Phase DAG (§6).** Execution order is a phase graph (acyclic), not textual
  order. A phase may carry an **event source** (`Phase.event`, Part VII A1): a
  non-empty source names the interrupt/event that TRIGGERS the phase — first-class
  asynchronous entry. The EV laws govern it on both rails (`kbcir/events.py`, the
  R3/EV seam in `-bcir-verify`): **EV1** an event phase declares no phase deps
  (hazards + masking order it, never program order); **EV2** the source must be armed
  by an explicit `irq.unmask:<src>` claim in the program flow (enablement is a claim
  over the controller resource, never implicit); **EV3** (the interrupt-context
  ordering seam) a resource written by an event phase may be touched by program
  claims only inside a masked window (`irq.mask:<src>` … `irq.unmask:<src>`) or as a
  `Lane.A` atomic. `event` defaults to `""` and is digest-excluded — the entire
  pre-A1 corpus is untouched (non-disturbance, measured).
- **Lane law (§7).** Lanes are execution-geometry types: `U` unit/stride, `UX`
  cacheline-local, `T` tile, `GGG` gather/scatter (always legal, must be
  minimized), `A` atomic, `H` hazard/provenance.
- **K_BCIR cost (§8).** Cost is *in the IR* — a 12-d `costvec`
  (compute, memory, fabric, sync, compile, thermal, power, reliability, security,
  accuracy, contention, verification). Illegal paths are rejected before scoring;
  Pareto pruning precedes scalar selection; the selected path hydrates GEM.
- **GEM Stream IR (§9).** The StreamPack is the hot artifact; the BCIR graph is
  the dormant semantic artifact. A pack retains provenance and generation tags
  and is rehydrated (patch/repack/replan) on mismatch. Scheduling is
  duration-aware (`bcir.gem.schedule`): EFT waves with locality affinity and the
  bandwidth knee, or the `!bcir.token` DAG. StreamPack v1 is frozen; v2 adds
  pipelining/double-buffer records and v3 adds segment dispatch/channel metadata under the
  append-only rules in [`BCIR_STREAMPACK_ABI.md`](kernel/BCIR_STREAMPACK_ABI.md).

### The naked-pointer policy (normative, §4)

C-frontend pointers enter the registry-first model (§4) as pointer *resources*, and every
dereference lowers to a load/store claim carrying a `bounds` strength. The policy is:

- **Known / recoverable extent → checked (`masked`).** An indexed access whose extent the
  frontend can recover — a local/static array's declared shape, or a `malloc`/`calloc`
  element count (a stable name, or a side-effect-free expression snapshotted at the
  allocation) — is promoted `assumed_safe → masked` and carries the `verify=bounds`
  contract (R7), discharged at runtime by the `BCIR_CHK` quarantine guard: transparent
  in-bounds; out-of-bounds, a provenance-recorded quarantine (abort by default; a strong
  override may recover only through the recorded two-truth decide, §14).
- **Unknown naked pointer → `assumed_safe` (trusted).** No runtime check is emitted; the
  access is trusted to land in its allocation.
- **`malloc`/`free` → optional R21 lifetime diagnostics.** Allocation/free events are
  stamped so R21 (§10) surfaces use-after-free / double-free — advisory by default,
  promotable to a fallback/reject verdict (`--r21`,
  [`CFRONT_GUIDE.md`](languages/CFRONT_GUIDE.md)).
- **No silent proof of unknown extents.** BCIR never fabricates a bound it cannot
  recover; the unprovable keeps the `--fallback`/quarantine contract.

Both C-front rails enforce this identically (oracle `lower.py::_access_bounds`/`_bind_extent`,
twin `bcir_cfront.c`; the bounds decision is part of the R13 digest, so a one-rail split
is a hard parity failure). The supported surface and ownership discipline are
[`CFRONT_GUIDE.md`](languages/CFRONT_GUIDE.md) and
[`C_MEMORY_DISCIPLINE.md`](languages/C_MEMORY_DISCIPLINE.md).

## 10. Verifier laws (R1–R24)

R1 registry uniqueness · R2 registry resolution · R3 domain legality ·
R4 phase-DAG legality · R5 hazard legality · R6 lane legality · R7 bounds
legality · R8 cost completeness · R9 plan legality · R10 stream provenance ·
R11 generation validity · R12 lowering legality · **R13 policy provenance** ·
**R14 CIM/PIM dispatch** (PIM only for `reduce.*`) · **R15 DVFS clock** (Q8 ∈
[64,512]; no PIM overclock) · **R16 allocator placement** (L1 ≤ 64 KiB / L2 ≤ 4 MiB) ·
**R17 accuracy contract** (a claim's static Q8-ULP error bound ≤ its declared tolerance;
a long `reduce.*` is bounded by `count` ULP naive but 1 ULP compensated, so a tight
tolerance forces `precision="compensated"`) · **R18 compositional call graph** (every
`kbcir.call` resolves to a `kbcir.func` and the call graph is acyclic — no recursion;
the law-rail form of `compose.plan_composite`'s undefined-callee + recursion rejections).
R14–R18 are first-class `-bcir-verify` laws, dual-rail with
`verify.{verify_cim,verify_dvfs,verify_allocator,verify_accuracy}`. R13 additionally
**recomputes** a manifest's digest from its component hashes (byte-identical to
`provenance._digest`) and **cross-checks every component hash** — `m_theta` / `m_policy` /
`m_target` / `m_module` — against the in-IR `kbcir.theta` / `kbcir.policy` (unfolded
base) / `target.capability` / `bcir.module` (`hash_*`), so neither the digest nor any input
identity is taken on trust. —
every decision rule in force (gain schedule, cost table) carries a generation
tag and an admitting certificate: a promoted portfolio entry requires its
replay certificate, a calibrated profile must present its frozen table with
matching generation and constants, a regret ledger's books must balance. Rule
swaps are never silent. Encoded as IR via the `bcir.verify.*` op family. The
runnable full set lives in `bcir/verify`, one entry point per correspondence
artifact — `verify(module)` R1–R8(static), `verify_plan` R8–R9, `verify_pack`
R10–R11, `verify_lowering` R12, `verify_provenance` R13 — and the MLIR-native
`-bcir-verify` pass enforces the structurally checkable form of all of R1–R24
on the dialect.

**Timing + lifetime laws (R19/R20/R21).** Three further laws over the
register-transfer / naked-pointer-safety tracks. They are driven by *optional*
claim metadata (absent by default, so the entire scalar / C-frontend subset is
unconstrained — the non-disturbance invariant, exactly as R14–R17 are vacuous for
it), now carried on the dialect as the `#bcir.timing` / `#bcir.lifetime` attributes
(`OptionalAttr` on `bcir.claim` / `load` / `store`):

- **R19 (synchronous-timing legality)** and **R20 (clock-domain-crossing)** —
  over the optional `#bcir.timing` block (`model.graph.Timing`, §5.11): a declared
  timing block must be internally consistent — a valid `sync_type`; non-negative
  `latency_cycles` / `setup_hold_margin` / `clock_frequency_mhz`; a synchronous
  claim carries a positive clock; the setup/hold margin fits within the stage
  latency (R19) — and a RAW dependency that crosses clock domains must be
  synchronized (the consumer declares `sync_type='mixed'` or a barriered hazard),
  else it is an unguarded crossing (R20).
- **R21 (pointer-lifetime legality: use-after-free / double-free)** — over the
  optional `#bcir.lifetime` annotation (`model.graph.Lifetime`, this §10): walking the
  claim order against the freed set, a read of a freed-and-not-reallocated resource
  is a use-after-free, a `free` of an already-freed resource is a double-free, and a
  write (reassignment / `alloc`) re-validates.

**ASN.1 encoding-rule legality (R24).** Over the `bcir.asn1.*` schema operations
(§17's ASN.1 profile, [`BCIR_ASN1_X690_ABI.md`](BCIR_ASN1_X690_ABI.md)). R24 checks the
faults decidable from the **type alone**, before any value exists — which is why they
belong on the law rail rather than in the oracle's encoder: a SET whose components
share a tag is undecodable for every value it could ever hold, and X.680 §24.4/§25.3
say so about the type. The rules: encoding rules must be `der` on a module and on an
`asn1.encode` (X.690 clause 10 + 11 — BER and CER leave the octets to the sender, and
BCIR digests what it emits); a module OID must be well-formed under X.690 §8.19.4
(root arc 0–2, second arc 0–39 beneath arcs 0 and 1); a universal tag number must be
assigned by X.680 Table 1 (0, 15 and 37+ are reserved); a primitive names a universal
tag and a constructor does not; a `sequence_of`/`set_of` names an element type and
nothing else does; component tags within a type are distinct; OPTIONAL and DEFAULT are
exclusive (X.680 §25.5) and a DEFAULT carries its value (needed for §11.5's omission);
a SET does not mix tagged and untagged components (it is order-free on the wire, X.690
§8.11.2); a `strict_der` decode does not also declare it accepts BER; and an
`asn1.projection` is marked **additive** — a projection that replaced a frozen wire
format would invalidate every digest taken over the native octets, so the IR refuses to
express one. Vacuous for IR with no `bcir.asn1.*` operation (the non-disturbance
invariant R14–R23 also hold to). Oracle twins: `asn1.schema` and `asn1.der`.

**Shape + dtype laws (R22/R23) — the D2 promotion.** The E3–E6 `check_*` validators'
"structurally valid tensors" guarantee, made law over the `gem.*` tensor claims:

- **R22 (shape consistency)** — a producer→consumer **gem seam** hands over one
  tensor, so both ends must agree: on the MLIR rail, a `gem.activation` adjacent to a
  `gem.matmul` (the fusion contract) must declare a shape extent equal to the matmul's
  `m*n`; on the oracle rail, `verify.verify_shape` checks the written→read `count`
  handover between gem claims. Spec-level shape/extent/kind rules from the E3–E6
  checkers ride the same number via `verify.verify_ml_spec`.
- **R23 (dtype compatibility)** — the dtype handover at a gem seam: an activation
  epilogue adjacent to a `gem.conv` / `gem.attention` must declare the producer's
  dtype; the E3–E6 quarantine dtype rules (a transcendental needs `f32`) ride the
  same number at the spec level (`verify_ml_spec`). The model-rail `Claim` carries no
  dtype, so structural R23 is MLIR-side by design.

Both are **vacuous by default** (no gem seam / no spec checked — the whole scalar /
C-frontend corpus is untouched), with negative `-verify-diagnostics` cases in
`mlir/test/passes/verify_shape_dtype.mlir` and the oracle suite in
`bcir/tests/test_shape_dtype_laws.py`.

The same R22/R23 adjacency discipline extends to the **rung-5 LLM decode ops**
(open-weight ladder §7.4: `bcir.gem.embedding`, `bcir.gem.rmsnorm`, `bcir.gem.rope` —
the law-rail records of the rung-3 reference decoder's stages). Op-level laws:
positive extents everywhere; `gamma_len == dim` on the normalizer; RoPE's `dim` must
be **even** (the rotation pairs channels `2k/2k+1`); rmsnorm/rope are `f32` (their
sqrt/cos/sin ride the trusted libm edge — the attention-softmax quarantine rule).
Seam laws: an `embedding → rmsnorm` pair must agree on the gathered extent
(`rows*dim == n_ids*dim`, R22) and hand the dtype over (R23); a `rope → attention`
pair must agree on the head width (`d_k == rope dim`, R22 — a mismatch means the
rotation straddled head boundaries) and the dtype (R23). Vacuous with no adjacent
pair; negatives in `mlir/test/passes/verify_llm_ops.mlir`; oracle twin
`bcir/frontends/models/decode.py` (`decoder_layer_reference` composes exactly this
chain).

The rung's remaining ops complete the same discipline: **`bcir.gem.gqa_attention`**
(grouped-query attention — `n_heads` query heads over `n_kv_heads` shared K/V heads;
`n_kv_heads == n_heads` is plain MHA) and **`bcir.gem.kv_cache`** (the per-layer,
per-kv-head roped K/V row store; `capacity = 0` is the unbounded reference posture).
Op-level laws: `n_kv_heads` must divide `n_heads` and sit in `[1, n_heads]` (GQA
shares **whole** head groups — the constraint `DecoderSpec` enforces oracle-side);
GQA is `f32` (its softmax rides the libm edge); a nonzero-capacity cache refuses
`pos > capacity` (the paged-serving over-fill lie). Seam laws: a
`rope → gqa_attention` pair rides the plain-attention `d_k` handover (R22) + dtype
(R23); a `kv_cache → gqa_attention` pair must agree on the shared head geometry
(`n_kv_heads` and `d_k`, R22) and the dtype (R23). Oracle twins:
`gqa_attention_reference` and `KVCache`/`decode_with_kv_cache` (bit-for-bit against
naive recompute); C twins: `runtime/c/bcir_decode.c::bcir_gqa_attention` /
`bcir_gqa_attention_row` (the cached row is the full recompute's last row **bitwise**,
one shared kernel path).

The **driver-seam hardening laws** (Phase D, the D-R rules) ride the same numbers:
**`bcir.device_manifest`** is the static hardware schema (D-R1) — bank names/tiers/
capacities/native tiles + the row-major Q8 interconnect distance matrix, verified
internally consistent (parallel arrays; positive capacities/tiles; a square,
zero-diagonal, positive-off-diagonal, **symmetric** distance matrix — an interconnect
is a metric). The seam law: a `gem.matmul` adjacent to a manifest must submit tiles
that are **multiples of the device's native tile** (R22 — a 15×15 tile against a
16-native device is runtime fragmentation, refused at compile time). Discovery is
veto-not-steer by construction: no adaptive re-planning op exists. Oracle twins:
`kbcir/device_manifest.py` (`check_device_manifest`, `check_strided_view`,
`check_bank_moves` — memory-tier crossings need an explicit `mem.move.*`; MMIO exempt,
its ordering is the R3 rail — and `probe_agree`); negatives in
`mlir/test/passes/verify_device_manifest.mlir`.

R19/R20/R21 are **first-class `-bcir-verify` MLIR laws**, dual-rail with the
oracle's `verify.verify_timing` / `verify.verify_lifetime` (run through
`verify.verify_smart_lowering` alongside R14–R17 and R22), each with a negative
`-verify-diagnostics` case in `mlir/test/passes/verify_timing_lifetime.mlir`, so the
generated status ([`STATUS.md`](STATUS.md)) now reports the first-class set as
**R1–R24**. R21 detection runs on both driver rails; it is *advisory* by default
(surfaced, never gates), and the `bcir-cc` / `bcir-cfront` drivers expose a `--r21`
policy — `advisory` (default) · `fallback` (route the unit to the LLVM backend,
exit 2) · `reject` (a hard verify error, exit 1) — so a detected use-after-free /
double-free can gate the production compile, with the two rails drawing the same
exit code (the parity gate in `tools/c/check_runtime.sh`). Remaining pointer-safety work
is the bounds-promotion of array parameters under a dominating-bound proof and the
offline ML policy table.

**Volatile & atomic ordering are structural.** A
volatile-qualified access is a first-class **`volatile` qualifier on the claim rail**
(`Claim.volatile` / the MLIR `is_volatile` claim attr — distinct from
`bcir.volatile_load/store`, the lowered integer-address MMIO accessors), not a
resource string tag: **R5 requires an ordered (atomic/barriered) hazard on a volatile
claim**, and the optimizer fences it exactly like `barriered` (`bundle._conflict` /
`-bcir-bundle` never reorder or bundle across it). The C frontend stamps it on every
MMIO access (`domain=MMIO, hazard=barriered, volatile=true`), so the law holds by
construction; the qualifier is digest-excluded (R13) and false by default
(non-disturbance). Atomic RMW/CAS are likewise **first-class MLIR ops** —
`bcir.atomic_rmw` (`add|sub|xor|exchange`) and `bcir.atomic_cas` (strong/`weak`) —
carrying the existing `#bcir.mem_ordering` attr (absent = `seq_cst`; a CAS derives
its failure ordering by the LLVM strongest-failure rule) and lowering to
`llvm.atomicrmw` / `llvm.cmpxchg`, lifting the cfront `c.atomic.*` / `c.c11atom.*` /
`c.cmpxchg.*` opcode-string claims into structures the ordering law sees
(`atomic_ops*.mlir`, `verify_volatile.mlir`, `test_volatile_atomic_law.py`). The
remaining Phase 2 areas — indirect-call callee type+effect, pointer
extent-provenance, the ABI contract op — are the next increments.

**Barriers are first-class ordering edges (ASM3b).** A `barriered`-hazard claim is
not only un-reorderable itself — it *fences other claims*. No claim may be reordered
across a barrier, and a producer→consumer pair spanning one does **not** receive the
deforestation (fusion) cost discount. This is enforced in the optimizer, not the
verdict laws: `bundle._conflict` treats a barriered claim as conflicting with every
other claim (so `find_bundles` / `_legal_reorder` never move a claim past it), and
`realize.fused_candidates` skips the ×0.75 memory deforestation factor when the
consumer is barriered or a shared operand was produced by a barriered producer (the
fence forces the intermediate to materialize). The MLIR cost model
(`BCIRCostModel.h::fusedColumns`) mirrors the discount-skip **byte-for-byte** — an
oracle/MLIR divergence here would be an R13 breach — pinned by
`mlir/test/passes/cost_model_barrier.mlir`. The scope is **every** `barriered` claim:
memory fences, MMIO loads/stores, port-I/O, and volatile/`"memory"`-clobber inline
asm. This is a **structural** property, **not** a new verdict R-law (barriers stay
off the legality value-path); `verify.verify_barrier_ordering` checks a realized plan
respects the fence, *advisory* and out of the frontend verdict like R21. It is a safe
no-op on any module carrying no barriered claim.

**Order-parameterized memory fences (SEG6.1).** The cfront recognizes the
memory-fence intrinsics and lowers them to `barriered` `BARRIER` claims whose
**kind** rides the op string — `c.fence` (full / seq_cst), `c.fence.acquire` (load
fence), `c.fence.release` (store fence) — which `emit.py` realizes as the per-ISA
hardware barrier behind `--target` (x86 `mfence`/`lfence`/`sfence`; aarch64 `dmb
ish`/`ishld`/`ishst`; riscv64 `fence rw,rw`/`r,rw`/`rw,w`). The order-taking forms
`__atomic_thread_fence(order)` (GCC/Clang) and `atomic_thread_fence(order)` (C11
`<stdatomic.h>`) now **parse their `memory_order` argument** and route to the kind it
implies, recognizing both the C11 `memory_order_*` constants and the GCC `__ATOMIC_*`
macro spellings (which share the integer values relaxed=0, consume=1, acquire=2,
release=3, acq_rel=4, seq_cst=5): `acquire`/`consume` → `c.fence.acquire`, `release`
→ `c.fence.release`, and `seq_cst`/`acq_rel`/`relaxed` — plus any **non-constant** or
unrecognized order — fold conservatively to the full `c.fence` (a stronger fence never
under-synchronizes, so the fold is sound and never crashes the lowering). The
`acquire`/`release`/`seq_cst` orderings carry end-to-end to the MLIR `BCIR_MemOrdering`
enum and lower to `llvm.fence acquire`/`release`/`seq_cst`
(`mlir/test/passes/memory_ordering.mlir`); the BCIR↔LLVM ordering map is the identity
mirrored by `bcir/lower/memory_model.py`. The full-fence op string stays the
backward-compatible `c.fence` so the existing `__atomic_thread_fence(5)` /
`__sync_synchronize` claims are unchanged.

**C-twin dual-rail fence parity (SEG7).** The byte-identical C reimplementation
(`runtime/c/bcir_cfront.c`) that backs the dual-rail digest parity gate mirrors the
SEG6.1 *lowering* exactly: the order-taking forms route the fence **kind** by the first
argument's order (an unshadowed `memory_order_*` / `__ATOMIC_*` name resolves to an int
const claim identical to a same-valued literal, with the same env→func→constant shadow
precedence as the oracle), so the dual-rail **digest** now covers `acquire`/`release`/
`seq_cst` symmetrically — verified by the extended `runtime/c/cfront_atomic.c` corpus
under `bcir/tests/test_c_cfront.py` / `tools/c/check_runtime.sh`. The C twin re-emits
**portable** C (`__atomic_thread_fence(__ATOMIC_ACQUIRE/RELEASE/SEQ_CST)`); the per-ISA
fence asm is Python-only by design, so the C twin carries no per-ISA emit.

## 11. Rewrite laws (the building-blocks engine)

Lane promotion (`GGG→UX→U(k)→U`), tile formation, layout (`AoS→SoA→AoSoA`),
prefetch introduction, GGG quarantine. A rewrite is legal **only if** it does not
increase the selected K_BCIR cost (or strictly improves legality) and the module
still passes R1–R12. Encoded via `bcir.opt.*`.

The **composition** engine that applies them is an e-graph / equality saturation
(`kbcir.egraph`, `bcir.egraph.extract`) — the realization of the liked/unliked-pair
model. **Liked pairs** are e-classes: an atom or a shared subexpression is a class
(the identity `a = a`, the memory module), and hashconsing finds common
subexpressions for free (CSE). **Unliked pairs** are operators and the rewrites
they enable: a rewrite proves two forms equal and *merges* their classes
(congruence closure), folding an unliked result toward a simpler liked attractor
(`x+0→x`, `1+1→2`, `a*b+a*c→a*(b+c)`). **Resolution** `Res(·)` is one round of
rewriting + congruence rebuild; **extraction** picks the min-cost representative
per class. Because extraction returns the minimum, the optimized cost is always
≤ the input cost — an R9 obligation.

**The Axiom of Memory Modules — `a = Lim(Res(U))`.** Resolution is *monotone* (a
merged class never un-merges) over a *bounded* lattice (finitely many e-nodes
under a terminating rule set), so by Knaster–Tarski/Kleene the iteration
`Res^k(U)` converges to a least fixpoint

```
Lim(Res(U)) = Res^∞(U) = the smallest X with Res(X) = X      (≡ saturation).
```

A **memory module** is the extraction of that fixpoint, *frozen* and
*generation-tagged*: `memory = Extract(Lim(Res(U)))` (`kbcir.memory`,
`bcir.kbcir.memory_module`). The **admissibility law** (the fixpoint witness) is
the bridge from the e-graph engine (§11) to the provenance spine (§13): an
artifact may be frozen into a generation **only if** resolution reached its
fixpoint —

```
saturated == True   ⇒   admissible as memory.
```

A *budget cutoff* is a partial `Res^k(U)`, `k < ∞`; freezing it pins a
non-canonical, still-improvable representative as "memory," and because a cutoff
is budget/order dependent while the fixpoint is canonical (confluence), two runs
that cut off differently need not agree — breaking the determinism the manifest
depends on. Idempotence `Res(Lim(Res(U))) = Lim(Res(U))` makes a memory module
its own attractor — the `a = a` identity at module scope — so re-resolving a
frozen module reproduces it; the verifier exploits this for tamper-evidence
(`verify.verify_memory` independently re-resolves rather than trusting the
recorded witness, the analog of `verify_manifest` recomputing the digest).
Witnessed by R13.

Nothing is globally immutable: a liked pair holds *within* a generation; across
generations it is an unliked pair resolved by rehydration, and the provenance
manifest pins identity. A frozen memory module's generation + content fingerprint
chain into that manifest (`manifest_for(..., memory=…)`), so an admissible
(saturated) extraction is itself part of a plan's commit hash.

**`bcir.gem.autodiff` — the closed-set autodiff DAG as a law object (B3).** The
same content-addressed / hash-consed machinery underwrites a *differentiable* DAG:
B3 (`bcir/kbcir/autodiff.py`) models reverse-mode AD as local backward rewrites
over an expression DAG built from a **closed vocabulary**
`{const, var, neg, add, sub, mul, div, dot, select}` (`autodiff.CLOSED_SET`,
derived from `_ARITY` as the single source of truth). The closure property —
differentiating any expression in that set produces a gradient DAG that stays in
the *same* set, no foreign op kind ever appears — is what lets the MLIR law rail
carry the forward DAG as a **first-class serialized op** with a fixed-vocabulary
verifier. `bcir.gem.autodiff` is an attribute-carrying (no-SSA-operand) plan op,
exactly like the sibling `gem.matmul`/`activation` plan records. It serializes one
hash-consed forward `Tape` as **parallel per-node arrays** (the node count =
`opcodes.size()`):

- `n_inputs` — the number of `var` leaves (function inputs);
- `opcodes` — per node, the op kind as a fixed code (`0=const 1=var 2=neg 3=add
  4=sub 5=mul 6=div 7=dot 8=select`, mirroring `_ARITY`'s order);
- `arities` — per node, its operand count (fixed by `_ARITY` except the **variadic
  `dot`**, an even `2k` operands carried per node);
- `arg_base` + `args` — the flattened operand node-indices; node `i`'s operands are
  `args[arg_base[i] : arg_base[i]+arities[i]]`;
- `consts` — per node a payload: a `const`'s integer value, a `var`'s input slot in
  `[0, n_inputs)`, or the `-1` sentinel for a non-leaf;
- `output` — the result node index.

Its verifier (`hasVerifier`) is the **closed-set law** — the op-level structural
well-formedness check the sibling gem ops run inline, **not** a new
globally-numbered R-law: (1) *closed vocabulary* — every opcode is in
`{const..select}` (a foreign code is rejected, the law the closure proof
underwrites); (2) *arity* — each node's operand count matches `_ARITY` (the
variadic `dot` carries an even `2k` arity); (3) *DAG / index bounds* — every
operand index references a **strictly earlier** node (`< i`, guaranteeing
acyclicity + topological order; no forward / out-of-range refs), a `var`'s slot is
in `[0, n_inputs)`, and `output` is a valid node index; (4) *payload consistency* —
a leaf carries a payload, a non-leaf the `-1` sentinel, and the parallel arrays are
equal length with `args` holding exactly `sum(arities)` operands. (Phase 2a: the op
+ verifier + round-trip only; the lowering/cost pass is a follow-up. The op carries
the DAG *structure* — op/operands/leaf-slot — with the `const` value as an integer
payload; a float `const` value is out of scope for this structural law object.)

**`bcir.asm` — verbatim inline assembly as a law op (ASM1, SEG8.1).** GNU extended
inline asm (`__asm__`/`__asm__ __volatile__`) is a **trusted opaque effect edge**: on
the cfront rail it is the `c.asm:` / `c.asm.volatile:` claim
(`bcir/frontends/cfront/lower.py::_asm_stmt` + `AsmInfo`); `bcir.asm` is its
**MLIR-rail twin**, closing the gap where inline asm was cfront-only (the MLIR rail had
`bcir.barrier` for fences but nothing for inline asm). It carries the assembly
**verbatim** (ISA-neutral pass-through), the closest sibling op being `bcir.barrier` —
a `"memory"`-clobber / `volatile` form is an ordering fence. Fields (mirroring
`AsmInfo`):

- `asm_template` — the asm template string in the user's (cfront/C source) **GCC operand
  syntax** (e.g. `"mfence"`, `"movl %1, %0"`). The lowering **translates** the GCC operand
  syntax to LLVM-IR `$`-syntax before storing it on `llvm.inline_asm` (`%%`→`%`,
  `%<letter><N>`→`${N:<letter>}`, `%<N>`→`$N` incl. multi-digit, a literal `$`→`$$`; exotic
  `%[name]`/`%P`/`%=` are **rejected** at lowering with a clear diagnostic), since a verbatim
  GCC `%w1` fails `llc: invalid register name`. Named `asm_template`, not `template`, since
  `template` is a C++ keyword the ODS-generated accessors would collide with;
- `is_volatile` — a `__asm__ __volatile__` (a side-effecting edge that must not be
  DCE'd); the lowering *always* marks `has_side_effects` (asm is conservatively
  side-effecting), so this records the source spelling for round-trip;
- `out_constraints` / `in_constraints` — the per-operand constraints, each output
  already an output constraint (e.g. `"=r"`), each input a plain constraint (e.g.
  `"r"`, `"Nd"`);
- `clobbers` — the clobber list, **bare** (e.g. `"memory"`, `"cc"`); the lowering
  renders each as a `~{...}` constraint **entry** (LLVM has no separate clobber field);
- operands `$args` are **outputs-then-inputs** in the cfront SSA order (the first
  `out_constraints.size()` operands are the output lvalue SSA values); results
  `$results` are one per output constraint.

Its verifier (`hasVerifier`) is the op-level structural well-formedness check (**not** a
new globally-numbered R-law): `args.size() == out_constraints + in_constraints` and
`results.size() == out_constraints`. The lowering (`-convert-bcir-to-llvm`) emits
`llvm.inline_asm` — a **single** LLVM constraint string built as the out constraints,
then the in constraints, then each clobber as a `~{<clobber>}` entry, all comma-joined
(e.g. outs `["=r"]`, ins `["r"]`, clobbers `["memory"]` → `"=r,r,~{memory}"`),
`has_side_effects`, default (AT&T) dialect. The LLVM `call asm` operand list is the
**input operands only** (a write-only `"="` output is the *result*, never an asm-call
argument), so the lowering passes `args[out_constraints.size():]`. **Lowering scope (the
first slice):** 0 or 1 result, write-only `"="` outputs; a multi-output asm (LLVM
returns a struct needing `extractvalue` unpacking) and read-write `"+"` outputs are a
follow-on (SEG8.x) — the lowering rejects `results.size() > 1` **and** any `"+"`
out-constraint with a clear diagnostic rather than shipping wrong/invalid LLVM. (A `"+"`
output is read-write, not a `"="` result: it lowers to `llvm.inline_asm … "+r,r" %x :
(i32) -> i32`, which opt/llc reject — *“inline asm without outputs must return void”* —
so the full `"+"` handling, the tied input + matching constraint, is the same SEG8.x
follow-on.) This establishes the `llvm.inline_asm` lowering the **SEG8.2** port-I/O op
(`bcir.portio`, per-ISA x86 `in`/`out` emitted *as* inline asm) reuses. The lowering is
**assemble-checked** end-to-end (one coherently resolved `mlir-translate` / `llc`
toolset, ending in `llc -filetype=obj`), not just FileCheck text. (Tests:
`mlir/test/passes/inline_asm_roundtrip.mlir`, `inline_asm.mlir`,
`inline_asm_verify_neg.mlir`, `inline_asm_lower_neg.mlir`,
`asm_lowering_smoke.mlir`.)

**`bcir.portio` — x86 port-mapped I/O as a law op (ASM2, SEG8.2).** The x86
`in`/`out` instructions (port-mapped I/O) are a **trusted opaque effect edge**: on
the cfront rail this is the `c.portio.in.{b,w,l}:` / `c.portio.out.{b,w,l}:` claim
(`bcir/frontends/cfront/lower.py::_portio` + `emit.py::_portio_stmt`); `bcir.portio`
is its **MLIR-rail twin**, **reusing** the `llvm.inline_asm` lowering `bcir.asm`
established (SEG8.1) — port I/O *is* x86 inline asm. Port I/O is volatile + ordered
(barriered), off the legality value-path, and **x86-only** (ARM/RISC-V have no port
I/O, only MMIO; the cfront claim raises an honest diagnostic on a non-x86 target).
Fields:

- `direction` — `in` (a READ: the port is the input, the read value is the result) or
  `out` (a void WRITE: value-then-port are the inputs, no result), an `#bcir.port_dir`
  enum mirroring the cfront `c.portio.in.*` / `c.portio.out.*` suffix;
- `width` — the access width in **bits**, one of `{8, 16, 32}` (verifier-checked),
  mapping to the GCC operand-size modifier **b=8 / w=16 / l=32** (the cfront `{b,w,l}`
  suffix);
- operands keyed off `direction`: `in` takes **one** operand `$port`; `out` takes
  **two**, `$value, $port` (value first — the Linux `out(value, port)` order);
- results keyed off `direction`: `in` produces **one** result (the read value); `out`
  produces **none**.

Its verifier (`hasVerifier`) is the op-level structural well-formedness check (**not**
a new globally-numbered R-law): `width ∈ {8,16,32}`; `in` ⇒ one operand + one result;
`out` ⇒ two operands + zero results; the in-result / out-value integer width equals the
op width (`i8`/`i16`/`i32`) and the port is an integer. The lowering
(`-convert-bcir-to-llvm`) selects the x86 template from `(direction, width)` in
**LLVM-IR operand syntax** (verified to assemble via `llc` to `inb %dx,%al` etc.) —
`in`: `inb ${1:w}, ${0:b}` / `inw ${1:w}, ${0:w}` / `inl ${1:w}, ${0:k}`; `out`:
`outb ${0:b}, ${1:w}` / `outw ${0:w}, ${1:w}` / `outl ${0:k}, ${1:w}`, where the
accumulator is `${0:b}`/`${0:w}`/`${0:k}` (al/ax/eax) and the port is `${1:w}` (the
16-bit `dx`) — and emits `llvm.inline_asm` via the **same generic attribute-list builder**
as `bcir.asm` (so it compiles identically on LLVM-20 and the CI's LLVM-22, where the
positional `InlineAsmOp` builder gained a `tail_call_kind` parameter). The constraint
string is `"={ax},N{dx}"` for `in` (output `={ax}`, then input `N{dx}`; the call operand
is the port only — the `={ax}` output is the *result*) and `"{ax},N{dx}"` for `out` (two
inputs `value, port`, no output). The **fully-qualified register names** are what LLVM’s
x86 backend needs: cfront’s GCC `%w1`/`=a,Nd` spellings are correct on the C→clang path
but `llc` rejects them on this MLIR→LLVM path (*“invalid register name”* / *“couldn’t
allocate output register”*) — clang’s frontend does the `%b0`→`${0:b}`, `=a`→`={ax}`,
`Nd`→`N{dx}` translation, which the frontend-less MLIR rail emits directly. `has_side_effects`
is **always** set (port I/O is volatile, never DCE'd/reordered — like the cfront
`__volatile__`). The lowering is **assemble-checked** end-to-end (one coherently resolved
`mlir-translate | llc` toolset), not just FileCheck text. Like `bcir.asm`, it does
**not** need the Python oracle's MLIR emitter to produce it yet (the oracle→MLIR wiring
is a later increment).
(Tests: `mlir/test/passes/portio_roundtrip.mlir`, `portio.mlir`, `portio_verify_neg.mlir`,
`asm_lowering_smoke.mlir`.)

**`bcir.volatile_load` / `bcir.volatile_store` — first-class MMIO on the law rail
(SEG8.2 / D1.2).** A memory-mapped device-register access is an **ordered volatile**
read/write. On the cfront rail this is a `volatile`-qualified register: the resource is
marked `domain=Domain.MMIO, access="volatile"` (`bcir/frontends/cfront/lower.py`) and
`emit.py` renders the ordered volatile access `*(volatile T *)(ptr + off)`. Until now the
MLIR rail carried MMIO only as the `Domain.MMIO` **enum value** on a claim/resource, with
device access riding the barrier/hazard machinery; these two ops make the MMIO **accessor**
first-class, as it already is in cfront emit. They are the **lowered** accessor — the
post-bounds-check device access — so they take the resolved integer register **address**
(mirroring the cfront emit `*(volatile T*)(intaddr)`); the RID/extent/bounds contract (R3)
stays at the `bcir.claim`/`bcir.load` layer, exactly as in cfront (the claim carries the
isolation law; the emit is a raw ordered volatile access).

- `bcir.volatile_load $addr : iN -> T` — `$addr` is the device-register address (a
  **signless integer**, e.g. an `i64` physical address); the result `$value` is the
  register's natural width (`i8`/`i16`/`i32`/…).
- `bcir.volatile_store $value, $addr : T, iN` — the void counterpart.

The lowering (`-convert-bcir-to-llvm`) emits `llvm.inttoptr $addr` to an opaque
`!llvm.ptr` then a **volatile** `llvm.load` / `llvm.store` (`volatile` set via the generated
setter, not a positional builder arg, so it is stable across LLVM-20 and the CI's LLVM-22).
`volatile` is **always** set — a device read/write must never be elided, duplicated, or
reordered with other volatile accesses (the MMIO guarantee, matching the cfront `volatile`).
The verifier requires `$value` to be a **scalar** hardware-register type (an integer or
float — a vector/index/struct device register is rejected) and `$addr` to be an integer of
at least **pointer width (≥ 32 bits)** so a too-narrow address is not silently zero-extended
into the device pointer. Like `bcir.asm`/`bcir.portio`, the oracle→MLIR wiring is a later
increment. (Tests:
`mlir/test/passes/volatile_mmio_roundtrip.mlir`, `volatile_mmio.mlir`,
`volatile_mmio_verify_neg.mlir`.)

**`bcir.creg_read` / `bcir.creg_write` — x86-64 control-register access (boot/CPU-state
asm edge, D1.3).** Reading/writing a control register (`CR0`/`CR2`/`CR3`/`CR4`/`CR8`) has
no C value-semantics — it is the **irreducible-assembly** tier (a trusted opaque effect
edge, like `bcir.asm`/`bcir.portio`). **Unlike** those two, it has **no cfront dual-rail
twin** (the cfront rail expresses control-register access only as a raw `c.asm` blob), so
the template is authored **directly in LLVM-IR inline-asm syntax** (`$0` operand,
single-`%` register) — empirically the exact form clang emits for
`__asm__("mov %%cr3, %0" : "=r"(v))`. (`bcir.asm` and `bcir.portio` carry GCC operand
syntax and the lowering **translates** it to this LLVM-IR `${N:mod}` form; the creg
templates, having no cfront dual-rail twin, are simply authored in the LLVM-IR form
directly. All three are now **assemble-checked** through `mlir-translate | llc`, not just
FileCheck text.) x86-only.

- `bcir.creg_read <crN> -> i64` — `reg` is a `#bcir.ctrl_reg` enum
  (`cr0`/`cr2`/`cr3`/`cr4`/`cr8`); the result is the 64-bit register value (control
  registers are 64-bit in long mode, verifier-enforced).
- `bcir.creg_write <crN>, $value : i64` — the void counterpart.

The lowering (`-convert-bcir-to-llvm`) emits `llvm.inline_asm has_side_effects` with
`"mov %<crN>, $0"`, `"=r,~{memory}"` for a read (returns `i64`) and `"mov $0, %<crN>"`,
`"r,~{memory}"` for a write (no result). The `~{memory}` clobber is carried on **both** the
read and the write: a control register reflects/controls the live memory system — a `CR3`
write reloads the page tables and flushes the TLB, `CR0`/`CR4` toggle paging/protection, and
a `CR2`/`CR3` *read* observes that live state — so each access is an **ordering point** vs
ordinary memory ops (`has_side_effects` alone only orders vs other side-effecting asm, not vs
plain loads/stores). `has_side_effects` is **always** set (a control register is not a pure
value: `CR2` holds the faulting address, `CR3` the live page-table base). Built via the
**same generic attribute-list `InlineAsmOp` builder** as `bcir.asm` (LLVM-20/22 stable).
(Tests: `mlir/test/passes/creg_roundtrip.mlir`, `creg.mlir`, `creg_verify_neg.mlir`.)

**`bcir.msr_read` / `bcir.msr_write` — x86-64 model-specific-register access (boot/CPU-state
asm edge, D1.4).** The same Tier-1 trusted-opaque-edge family as `bcir.creg_*` (no C
value-semantics, **no cfront dual-rail twin** — the cfront rail expresses MSR access only
as a raw `c.asm` blob — so the template is authored directly in LLVM-IR inline-asm syntax).
**Unlike `creg`**, the register is **not** an enum but a **runtime index** (`rdmsr`/`wrmsr`
select the MSR numbered by `ECX`), and the 64-bit value is **split across `EDX:EAX`**
(high:low). x86-only.

- `bcir.msr_read $index : i32 -> i64` — `index` is the MSR number (placed in `ECX`); the
  result is the 64-bit MSR value.
- `bcir.msr_write $index, $value : i32, i64` — the void counterpart.

The lowering (`-convert-bcir-to-llvm`) keeps the op surface a **clean 64-bit register** (the
`EDX:EAX` split is an ISA detail of `rdmsr`/`wrmsr`, not part of the op contract): a read
lowers to a **multi-output** `llvm.inline_asm has_side_effects "rdmsr", "={ax},={dx},{cx},
~{memory}"` returning `!llvm.struct<(i32, i32)>`, then **reassembles** the i64 as
`(zext hi) << 32 | zext lo` (the kernel `native_read_msr` idiom); a write **splits** the i64
into its low/high halves (`trunc` / `lshr`+`trunc`) and feeds `llvm.inline_asm
has_side_effects "wrmsr", "{cx},{ax},{dx},~{memory}"` `(index, low, high)` with no result.
The `~{memory}` clobber is carried on **both** the read and the write: an MSR is live CPU
state (`IA32_EFER` toggles long mode, `IA32_APIC_BASE` relocates the local APIC, the
`SYSENTER`/`STAR` MSRs install syscall entry points), so each access is an **ordering point**
vs ordinary memory ops, exactly as for `creg`. `has_side_effects` is **always** set. Both
lowerings are **assemble-checked** through `mlir-translate | llc` (the `rdmsr`/`wrmsr`
opcodes appear in the emitted object), not just FileCheck text. (Tests:
`mlir/test/passes/msr_roundtrip.mlir`, `msr.mlir`, and the assemble-smoke fixture
`asm_lowering_smoke.mlir`.)

**`bcir.entry` — x86-64 long-mode C handoff (ASM4).** This top-level symbol op is the
irreducible no-prologue edge below the verified C driver rail:

```mlir
bcir.entry @bcir_boot stack @bcir_stack_top target @bcir_kernel_main
```

Its lowering appends LLVM module assembly that loads the stack-top symbol into `RSP`, aligns
down to 16 bytes, pushes a zero sentinel so the tail-jumped SysV C entry observes ordinary
function-entry alignment, clears `RBP` and DF, and jumps to the `noreturn` C target. It is not
a compiler-generated naked function: LLVM deliberately constrains naked-function IR, while
module assembly preserves the exact entry sequence. All three symbols must match the unambiguous
`[A-Za-z_][A-Za-z0-9_.$]*` assembler subset, preventing directive injection and AT&T
immediate-prefix ambiguity.

This op starts in **x86-64 long mode**. It does not implement a reset vector, A20, real/protected/
long-mode transitions, page tables, relocation, UEFI/bootloader protocol, or AP startup. Those
remain explicit platform boot-stub work and must not be inferred from the `entry` name.

**`bcir.descriptor_load`, `bcir.task_register_load`, and `bcir.segment_reload` — x86
descriptor/segment state (ASM5).** These typed side-effecting operations remove raw-template
ambiguity at the boot edge:

- `bcir.descriptor_load <gdt|idt>, %addr : i64` emits `lgdt` or `lidt` through a pointer to
  the architectural ten-byte pseudo-descriptor;
- `bcir.task_register_load %selector : i16` emits `ltr` with a structurally 16-bit selector;
- `bcir.segment_reload %data, %code : i16, i16` loads DS/ES/SS and uses a unique-label far
  return to reload CS. FS/GS bases are intentionally managed through their MSRs.

Each is `has_side_effects` with a memory clobber. The lowering emits real object code and the
gate disassembles it with `llvm-objdump`; tests never execute these privileged instructions.

**`bcir.interrupt_trampoline` — normal x86-64 interrupt/trap entry (ASM6).** A top-level
trampoline normalizes the processor frame, saves every integer GPR, calls a verified C body with
one borrowed frame pointer in RDI, restores state, removes `vector,error_code`, and returns with
`iretq`:

```mlir
bcir.interrupt_trampoline @irq32 vector 32 handler @dispatch {
  swapgs_on_user = true
}
```

The vector is in `[0,255]`, except that the ordinary op unconditionally refuses vectors
1 (#DB), 2 (NMI), 8 (#DF), 18 (#MC), and 29 (AMD #VC). Linux likewise gives #VC an
IST/nesting-specific entry rather than its ordinary IDT path. For accepted vectors, the lowering owns the
hardware-error-code set `{10,11,12,13,14,17,21,30}`; all others receive a synthetic
zero before the vector word. Callers cannot supply an error-code boolean that would shift
the return frame.
The fixed frame exposed by [`bcir_x86_interrupt.h`](../runtime/c/bcir_x86_interrupt.h) is:

```text
r15 r14 r13 r12 r11 r10 r9 r8 rdi rsi rbp rdx rcx rbx rax
vector error_code rip cs rflags rsp ss
```

The frame is always 176 bytes. AMD64 long mode pushes saved `SS:RSP` for every interrupt and
`iretq` consumes them even on a same-CPL return; they are not an optional user-only tail. This is
pinned against the [AMD64 system-programming interrupt contract](https://docs.amd.com/v/u/en-US/24593_3.44_APM_Vol2),
not inferred from CPL. The trampoline owns the frame and the C handler borrows it for the call. A
handler may edit saved return state but may not change the saved CS RPL or retain the pointer. C is compiled with
`-mno-red-zone -mgeneral-regs-only`;
SIMD/FPU state is outside this ABI until an explicit XSAVE policy lands. Editing RIP/CS while CET
shadow stacks are active also requires a corresponding shadow-stack update policy; otherwise #CP may
veto the return.

The shim starts with `cli`. It accepts only original CPL0/CPL3 frames, records the original
CS RPL in private callee-saved state, and traps with `ud2` if the C body changes that RPL;
this preserves the selector-validation, return-protection, and `swapgs` policy chosen on entry. When
`swapgs_on_user` is true, exact CPL3 entry executes `swapgs`, fences before C, and pairs the
exit from the private entry state. Editing other permitted return fields therefore cannot
desynchronize GS. A separate paranoid GS/IST/nesting operation, plus CR3/PTI policy where
applicable, is required for the five refused vectors.

This normal edge does not emit Linux-style alternatives for SMAP `clac`, CET/IBT, CR3/PTI, or
entry-side speculation mitigations. A platform must keep those features disabled/known-safe or add a
separately verified feature-specific entry policy before binding this op to a production IDT.

`x86_driver_edges.mlir` pins parsing/lowering and unsafe-symbol/policy refusals;
`asm_lowering_smoke.mlir` plus `tools/wsl/check_asm_lowering.sh` emits and disassembles a real
object, checks all eight accepted hardware-error vectors, the ordinary no-error control,
the five unconditional refusals, stack handoff, descriptor/segment instructions, `cli`,
private RPL/`swapgs` pairing, `lfence`, the `ud2` class guard, and `iretq`.

## 12. Lowering contracts

BCIR-4 → BCIR-5 lowering is governed by R12: each lowered op preserves the BCIR
semantic (lane geometry, bounds, hazard, precision) or carries an explicit
discharge in `bcir.trace`. LLVM is the **first** backend, not the center.
Encoded via `bcir.isa.*` / `bcir.target.lower_contract`.

**Modular Mapping Functions (`kbcir.mapping`).** A lowering — and any
representation change — is a mapping function `f` between cost-bearing
representations, and R12 imposes two further laws on it:

- **Objective-support preservation.** `Supp(J)` is the set of cost dimensions on
  which the objective `J` is nonzero — *where the objective matters*. A legal map
  must carry that support forward,

  ```
  f(Supp(J)) ⊆ Supp(J')
  ```

  so a lowering may sharpen, rescale, or fuse a cost but may not silently **drop**
  a dimension that mattered (lose the thermal / security / accuracy / verification
  term) unless it carries an explicit discharge — the same escape R12 already
  grants bounds/hazard/precision. The objective's footprint is an invariant of
  legal lowering (`verify.verify_support_preservation`, R12).
- **Commutativity / path independence.** If two conversion paths reach the same
  target — a direct map `Φ` and a two-step `Ψ` then `Λ` — they must agree:

  ```
  Λ ∘ Ψ = Φ
  ```

  A result may not depend on which legal path produced it. This is the
  PARITY/manifest discipline generalized to **any** representation rail:
  oracle↔MLIR parity, manifest replay (`reproduces`), JSON round-trips, and any
  future rail are instances of one commuting-square law
  (`verify.verify_commutativity`, R12).

## 13. Learning placement (normative policy)

Learning and measurement enter BCIR only where decisions are slow enough to
amortize inference, reversible at the next checkpoint, and produce artifacts
the R-laws can check. The placement criterion is the **amortization
inequality** — place learning at a layer only if

```
E[improvement per decision]  >>  decision rate x inference cost
```

— stratified by timescale, **never by importance**:

- **L0 (the hot path — PROHIBITED).** No learned inference executes on the hot
  path: lane-promotion *application*, prefetch issue, bounds masks, the
  StreamPack ABI, stackify, fences. At hot-path rates even a nanosecond-scale
  inference swamps what it optimizes. Decisions are **compiled out**: the
  binary artifact carries decisions, never models. This prohibition is law,
  not guidance.
- **L1 (plan time — frozen tables only).** Learning and measurement supply
  *inputs to exact search, never the search*: cost tables T_i, tier factors,
  gather penalties, coupling factors — produced offline, **quantized to
  integer Q8, frozen, generation-tagged** (`bcir.kbcir.calibration`,
  `kbcir.microbench.CalibratedProfile`, `cal_gen` on the target capability).
  The table may be a point estimate (microbench) or a **Bayesian posterior with
  a certified conformal error bar** (`kbcir.bayescal`): a conjugate-Gaussian
  (VI-exact) posterior over each ratio + a distribution-free split-conformal
  `±δ` at a stated coverage, optionally inferred likelihood-free by **ABC** with
  the GEM/`optimize` forward model as the simulator. The frozen artifact is
  still Q8 integers plus an integer `δ`; the conformal guarantee lets later
  selection be made *robust* over the credible interval. Plan-time "inference"
  is a table lookup; determinism and the pinned scores are preserved by
  construction. The verifier gates table well-formedness — and the conformal
  guarantee (coverage in (0,1), `δ ≥ 0`, no interval from ≤ 1 sample) — under
  R8/R13. The loop is **closed and certified** (`kbcir.calibloop`): measure →
  freeze → apply → replan emits a `CalibrationCertificate` whose **win** is the
  measured cost of *not* recalibrating (the stale plan, faithfully rescored on
  the machine telemetry reports, minus the recalibrated optimum); it is
  admissible only when `cal_gen ≥ 1` and `win ≥ 0` (R13,
  `verify.verify_calibration`). Measurement stays offline (L2/L3); the frozen
  table and every downstream decision are integer and reproducible.
- **L2 (checkpoints — portfolio + replay gate).** Gain schedules (policy
  weight vectors, thresholds) adapt only at checkpoints, only as members of a
  **portfolio of frozen, generation-tagged policies**
  (`bcir.kbcir.portfolio`), selected at plan time by a router. The router may be
  the deterministic workload-class table (`classify`) or a **learned
  Mixture-of-Experts gate** — a GNN over the claim graph trained on the regret
  ledger (`kbcir.moegate`, `bcir.kbcir.moe_gate`). The gate is the *safe*
  learning regime: it only *selects among already-certified experts*, never
  emits a table or policy; it deploys **frozen** (Q8 integer routing,
  deterministic across hosts) and only behind an admitting replay certificate.
  A schedule swap or a gate deployment requires that **replay certificate**
  (`bcir.kbcir.replay_certificate`): counterfactual replay on logged Θ episodes,
  judged by the incumbent's scheduled metric M(π,Θ), zero regressions over ≥ 1
  episodes (verified under R9/R13). The network proposes a route; the verifier
  disposes. Shadow → canary → promote; never silent.
- **L3 (the meta-policy — measured, human-actuated).** Where the
  heuristic/learned boundary itself sits is a measured question: the **regret
  ledger** (`kbcir.regret`, `bcir.kbcir.regret_ledger`) continuously books each
  deployed rule's gap to the hindsight-best alternative under one neutral
  yardstick. The retune trigger is **not a magic threshold** but the **MDL /
  Bayesian-evidence** two-part code: a swap is recommended iff it shortens the
  total description length,

  ```
  ΔL = Σ_i regret_i/best_i  −  (k/2)·ln(N)  >  0
       \___ data fit (saving) _/   \__ BIC complexity _/
  ```

  i.e. the accumulated *relative* regret (the bits the deployed rule wastes)
  must outweigh the model-complexity penalty of specifying and certifying the
  swap (the large-sample Bayesian evidence, Schwarz 1978). Few episodes of small
  regret never flag — that would be overfitting noise — while sustained or large
  regret does. The verdict is a recommendation, never an actuation: a flagged
  rule is a *candidate* for retuning, the swap still goes through the L2 replay
  gate, and **R13 (policy provenance)** witnesses the chain *and the evidence* —
  a verdict is illegal unless it is consistent with its MDL margin (retune ⟺
  data_fit > complexity). Actuation stays human by policy; any future automation
  of the flip must run behind both the gate and R13.

**Provenance is the spine.** Every decision rule in force is frozen and
generation-tagged; a **provenance manifest** (`kbcir.provenance`,
`bcir.kbcir.provenance_manifest`) chains a plan's inputs and those generations
into a single content hash — the commit hash of a plan. Manifest equality ⇒ an
identical plan, so the constantly-updating computation DAG is reproducible and
debuggable: an immutable plan is a *committed* manifest (a closed branch), a
candidate under evaluation is an *open* branch, and `diff` reports which
generation moved between two runs. Nothing is globally immutable, but everything
is immutable *within its generation*. R13 (`verify.verify_manifest`) requires a
deployed plan's manifest to reproduce its recorded score and shape on replay.

The **memory module** (§11, `kbcir.memory`) is the e-graph's contribution to this
spine: a frozen, generation-tagged *saturated* extraction `a = Lim(Res(U))`. Its
admissibility law — `saturated == True ⇒ admissible`, the fixpoint witness as the
admitting certificate — is what lets it earn a generation tag at all; a budget
cutoff `Res^k(U)` may not be frozen. An admissible module's generation +
fingerprint chain into the manifest (`manifest_for(..., memory=…)`), and R13
(`verify.verify_memory`, folded into `verify_provenance`) independently
re-resolves the stored representative to confirm it is a genuine fixpoint before
admitting it. This ties the building-blocks engine (the e-graph) to the
version-DAG spine (the manifest) with one checkable law.

The legality laws (R1–R12), lane semantics, and hazard contracts are **never
learnable**: they are laws, not preferences.

## 14. The two-truth separation (MOPC)

What makes §13 *enforceable* rather than aspirational is that BCIR carries **two
distinct kinds of truth** and quarantines them apart (`kbcir.twotruth`):

- **Classical truth `v`** — deterministic, binary, generation-independent: the
  legality verdicts of the R-laws. A claim is legal or it is not; a manifest
  reproduces or it does not; a memory module is a fixpoint or it is not. There is
  no "0.7 legal." This is the only truth `verify.*` speaks (a `Diagnostic` carries
  no confidence).
- **Graded truth `(v, w)`** — a *graded proposition*: a value carried with a
  confidence `w ∈ [0,1]`. This is the learned/measured machinery — the softdp
  plan posterior (§2), the bayescal conformal interval (§13 L1), the regret
  ledger's evidence (§13 L3). It answers *which legal plan is best*, never
  *whether a plan is legal*.

**The quarantine (the single most important discipline, enforced not stated): a
graded proposition may inform but never become a legality verdict.** Graded truth
is kept out of the verifier. The only sanctioned crossing is a `decide` — an
explicit, *recorded* collapse of a graded proposition to a classical value at a
frozen threshold (the anneal/freeze of §2/§13 made auditable). The crossing is
never silent, and **R13** (`verify.verify_quarantine`) is the guard that no
confidence-weighted value reaches the R-laws except as the classical value of a
recorded decision. The graded algebra (`g_and`/`g_or`/`g_not` — "dynamic truth
tables that learn") lives entirely on the graded side: it proposes, and the
classical laws dispose. This is the safe way to import learned dynamic truth —
keep it out of the verifier.

## 15. The enriched-operad memory interface (the higher intelligence layer)

The memory module (§11, `a = Lim(Res(U))`) is already an **operad**: its e-nodes
are operations, the operators are the composition `γ`, the atoms are the identity
`η`, and the extraction tree is the operad's operation tree. The higher
intelligence layer **enriches** that operad with labels and indexes
(`kbcir.operad`, `O_L = ((O_L(n)), γ_L, η_L, L, I)`) to make memory navigable,
traceable, and queryable — without touching the deterministic spine:

- **Labeling `L`.** A hierarchical, descriptive label per operation
  (`L(op) = (L1, L2, …)`, e.g. `("MEMORY","op","mul")`). Composition preserves it,
  `L(γ_L(…)) = f_L(L(parent), L(children…))` (`f_label`).
- **Indexing `I`.** A **content-addressed** index — the FNV fingerprint of
  `(name, label, child indexes)` — kept consistent under composition
  `I(γ_L(…)) = f_I(…)` (`f_index`). Content addressing (**not** random UUIDs) is
  the discipline that keeps the layer deterministic: structurally identical
  operations get the *same* index, so CSE / the liked-pair identity `a = a` falls
  out for free and reproducibility is preserved.
- **Trace.** Reverse mapping from any operation to its constituent sources (the
  operation tree `T = (V,E)`, the `SourceMap`); rewrites are recorded as
  **2-cells** (the higher-category layer: transformations between operations).

**Where it sits.** Labels and indexes are *interpretive* metadata, quarantined on
the graded side of §14: they may **inform** planning, retrieval, and debugging but
are never read by the R-laws. The lower IR (StreamPack, realized plan) carries
decisions, not labels — so the layer is conditionally activatable
(`enable_labeling` / `enable_indexing`), matching the cost/benefit tiering: full
on the memory interface, selective in pipelines, off on the hot path. Its own
integrity (label consistency, content-addressed index uniqueness, mapping
integrity) is checkable under R13 (`verify.verify_enriched`) — the analog of
`verify_memory` for the enriched structure. `enrich_memory` lifts a frozen memory
module into this operad: the deterministic fixpoint, made intelligent.

## 16. BCIRQ8 v1 decoder-artifact contract

BCIRQ8 is the deterministic, weight-only signed-int8 persistence format for BCIR's
reference Llama/SwiGLU decoder. It is a model artifact, not a second numerical oracle:
reading it reconstructs exactly the per-group values produced by
`bcir.kbcir.quantize.quantize_per_group`. Python owns the canonical writer/reader in
`bcir.frontends.models.weights_io`; the portable hosted-C loader is
`runtime/c/bcir_q8_model.{h,c}`. The standalone `bcir-llama` realization accepts
verified token IDs; raw-text tokenization remains outside the C executable.

### 16.1 Scalar encoding and file invariants

- All integers and IEEE-754 binary64 fields are **little-endian**. Magic is the eight
  bytes `BCIRQ8\0\0`; version is `1`; the endian marker is `0x01020304`.
- Each weight code is signed int8 in the canonical symmetric range `[-127,127]`;
  `-128` is invalid. Each contiguous group has one signed-int16 power-of-two exponent
  `e`, currently constrained by both readers to `[-300,300]`. Element `i` reconstructs
  as `ldexp(code[i], exponent[i/group_size])`.
- `group_size` is in `[1,65535]`; the pinned real-model gate uses 32. BCIRQ8 v1 fixes
  `bits=8`. Generic Q4 and other low-bit oracle experiments do **not** change this wire
  format; they require a new, explicitly versioned artifact contract.
- The header is 224 bytes and each directory entry is 48 bytes. The directory starts at
  byte 224. The payload and every exponent/code span are eight-byte aligned; all
  alignment padding and reserved fields are zero.
- The file contains weights only. KV cache, RoPE inverse frequencies, activations, and
  tokenizer text machinery are runtime state. Ingest validates the auxiliary RoPE
  inverse-frequency tensor, then readers reconstruct it from `rope_base`.

### 16.2 Fixed 224-byte header

| Offset | Width | Field |
|---:|---:|---|
| 0 | 8 | magic `BCIRQ8\0\0` |
| 8 | 2 | `version` (`1`) |
| 10 | 2 | `header_size` (`224`) |
| 12 | 4 | endian marker `0x01020304` |
| 16 | 4 | flags; bit 0 means tied embedding/LM head; all other bits reserved |
| 20 | 2 | `group_size` |
| 22 | 1 | `bits` (`8`) |
| 23 | 1 | reserved zero |
| 24 | 24 | six `u32`: vocabulary, model width, query heads, KV heads, layers, FFN width |
| 48 | 4 | context length (`u32`) |
| 52 | 12 | BOS, EOS, and PAD token IDs (`i32`; `-1` may denote absent) |
| 64 | 8 | RoPE base (`f64`, finite and positive) |
| 72 | 8 | RMSNorm epsilon (`f64`, finite and positive) |
| 80 | 4 | tensor count |
| 84 | 4 | directory-entry size (`48`) |
| 88 | 8 | directory offset (`224`) |
| 96 | 8 | aligned payload offset |
| 104 | 8 | exact file size |
| 112 | 4 | CRC-32 of bytes `[directory_offset,file_size)` |
| 116 | 4 | header CRC-32, computed with this field zero |
| 120 | 32 | source checkpoint SHA-256 |
| 152 | 32 | source config SHA-256 |
| 184 | 32 | tokenizer SHA-256 |
| 216 | 8 | reserved zero |

Model geometry is valid only when all six dimensions are positive, `d_model` is
divisible by `n_heads`, `n_heads` is divisible by `n_kv_heads`, and head width is even.
The C loader additionally bounds the signed layer index representation.

### 16.3 Fixed 48-byte tensor directory

| Offset | Width | Field |
|---:|---:|---|
| 0 | 2 | tensor ID (`u16`) |
| 2 | 2 | layer (`i16`; `-1` for global tensors) |
| 4 | 1 | rank (`1` or `2`) |
| 5 | 1 | flags (zero in v1) |
| 6 | 2 | reserved zero |
| 8 | 4 | dimension 0 |
| 12 | 4 | dimension 1 (`1` for rank 1) |
| 16 | 4 | element count |
| 20 | 4 | group count, exactly `ceil(element_count/group_size)` |
| 24 | 4 | tensor CRC-32 over exponent bytes followed by code bytes |
| 28 | 4 | reserved zero |
| 32 | 8 | aligned exponent-array offset |
| 40 | 8 | aligned code-array offset |

Readers require exact dimensions/counts, unique `(tensor_id,layer)` keys, bounded and
non-overlapping spans, canonical CRCs, and this directory order:

1. embedding (`id=1`, global);
2. for each layer in ascending order: attention norm `10`, Q/K/V/O projections
   `11/12/13/14`, FFN norm `15`, gate/up/down projections `16/17/18`;
3. final norm (`100`, global);
4. LM head (`101`, global) only when embeddings are untied.

The exact tensor count is therefore `2 + 9*n_layers + (0 if tied else 1)`. Unknown,
missing, duplicated, reordered, overlapping, unaligned, truncated, non-canonical, or
CRC-invalid data is rejected before a decoder object is constructed.

### 16.4 Interfaces, ownership, and reproducibility gate

```python
write_q8_decoder(path, spec, weights, group_size=32,
                 source_hashes=hashes, tokenizer_ids=ids)
spec, weights, metadata = read_q8_decoder(path)
```

```c
void bcir_q8_model_init(bcir_q8_model *);
int bcir_q8_model_load(const char *, bcir_q8_model *, char *, size_t);
int bcir_q8_model_load_with_allocator(const char *, bcir_q8_model *,
                                      char *, size_t,
                                      const bcir_host_allocator *);
void bcir_q8_model_free(bcir_q8_model *);
int bcir_llama_generate_greedy(const bcir_q8_model *, const int32_t *, size_t,
                               size_t, int32_t *, double *);
```

The Python writer uses a same-directory temporary file, flushes and `fsync`s it, and
atomically replaces the destination. Equal inputs produce byte-identical artifacts. The C
loader zero-initializes outputs on reported failure, supports allocator injection, and has
an idempotent destroy contract for initialized/owned models.

`python tools/models/run_real_model_gate.py` is the always-on composition gate; `--offline`
requires an already verified cache. It publishes only
`build/model-gate/parity-report.json`. Checkpoints, tokenizers, logits, executables, and
derived BCIRQ8 weights remain local cache/build products and must not be committed. Model
provenance and license pins live in
[`THIRD_PARTY_MODELS.md`](machine-learning/THIRD_PARTY_MODELS.md).

## 17. Conformance profiles and external-contract boundary

This reference defines semantics; an implementation must name the profile it supports
and reject work outside it. The current profiles are:

- **MLIR law profile:** ODS/TableGen dialects plus `-bcir-verify` R1–R24 and the
  documented optimizer/GEM passes. A successful parse is not a lowering guarantee.
- **Python oracle profile:** executable semantic, planning, GEM, model, and codec
  reference. It is the differential oracle, not an alternative normative syntax.
- **C-front profile:** the driver-oriented C subset documented by
  [`CFRONT_GUIDE.md`](languages/CFRONT_GUIDE.md), with the explicitly scoped verifier
  surface and route-to-resident-compiler fallback. It must not claim complete ISO C23.
- **LLVM AOT/JIT profile:** exactly one selected, two-read/one-write elementwise
  add/sub/mul claim on the Python path. Additional executable claims are a hard error.
  MLIR `bcir-aot` is partial preparation and may leave mixed BCIR/GEM/LLVM operations.
- **x86 ordinary-entry profile:** long-mode C handoff, descriptor/segment operations,
  and the accepted ordinary interrupt/trap vectors described in §11. It excludes reset
  transition, NMI/IST/paranoid entry, and unmodeled feature policy.
- **BCIRQ8 decoder-artifact profile:** the complete v1 contract in §16, including
  canonical order, CRC/bounds checks, provenance hashes, and tied/untied heads.
- **ASN.1 / X.690 interop profile:** the whole of X.690 (02/2021) clause 8 over the
  X.680 (02/2021) tag assignments, restricted by clauses 10 and 11 on emission —
  **DER out, BER in**. CER is accepted on input and never emitted: §9.1 makes the
  indefinite length form mandatory for constructed encodings, which no digested,
  frozen artifact can carry. X.681/X.682/X.683 (information objects, constraints,
  parameterization) and X.691/X.692/X.693 (PER, ECN, XER) are outside the profile.
  A second set of encoding rules is built: **X.696 OER** (`bcir/asn1/oer.py`), under the
  same posture — **CANONICAL-OER out, BASIC-OER in** — over the same type model, which is
  the concrete form of the claim that encoding rules are a realization choice and not part
  of the schema. It is validated against X.696 Annex A's own worked example.
  Module TEXT is compiled by the X.680 front-end (`bcir/frontends/asn1/`, the
  `bcir-asn1c` CLI): a `.asn1` module lowers to the same type model the encoder uses,
  so a peer's schema is consumed rather than transcribed. The front-end covers the
  clause 13 module structure and the clause 16-31 types the profile can encode, and
  REFUSES what it cannot express -- an `ANY DEFINED BY` or an information object class
  names X.681 rather than being skipped, because a front-end that silently dropped one
  would build a type model disagreeing with the module it just read.

The following versioned contracts are adjacent to BCIR semantics but have dedicated
normative documents because their byte/lifecycle evolution is independent:

- StreamPack v1–v3: [`BCIR_STREAMPACK_ABI.md`](kernel/BCIR_STREAMPACK_ABI.md), whose
  native octets stay frozen; its **additive** ASN.1 module and DER projection live in
  [`BCIR_ASN1_X690_ABI.md`](BCIR_ASN1_X690_ABI.md);
- BTLM telemetry frame and signal meaning:
  [`TELEMETRY_FRAME_ABI.md`](kernel/TELEMETRY_FRAME_ABI.md) and
  [`SIGNAL_REGISTRY.md`](kernel/SIGNAL_REGISTRY.md);
- direct RuntimeChannel and future driver UAPI/IPC:
  [`BCIR_DRIVER_KERNEL_ROADMAP.md`](kernel/BCIR_DRIVER_KERNEL_ROADMAP.md).

RuntimeChannel v1 is an in-process hook contract, not a stable Linux userspace ABI. No
pointer may cross a future process/kernel boundary; adapters use fixed-width structures,
byte offsets, generation-tagged handles, explicit ownership, cancellation, timeout,
restart, and backpressure semantics.

Conformance is evidenced only by the applicable executed gates. Generated
[`STATUS.md`](STATUS.md) is an inventory and a clean skip records missing capability; it
does not turn an unexecuted architecture, toolchain, transport, or device path into a
supported one.

## 18. Thesis

> BCIR is a registry-first, phase-ordered, lane-typed, cost-governed
> correspondence IR. K_BCIR is the IR-level optimization calculus that selects
> legal physical realization paths. GEM is the execution IR that hydrates
> selected correspondence paths into streamed lane schedules. MLIR is the
> bootstrap framework used to define, verify, rewrite, and lower BCIR until BCIR
> has enough mass to become its own compiler toolchain.
