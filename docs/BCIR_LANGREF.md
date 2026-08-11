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
- **Artifact selection envelope.** BCAB v1 binds a StreamPack to exact standard backend
  payloads and selects by target/feature/manifest/calibration/R12 policy. It is separately
  versioned because it does not change StreamPack or BCIR semantics; see
  [`BCIR_ARTIFACT_BUNDLE_ABI.md`](kernel/BCIR_ARTIFACT_BUNDLE_ABI.md).

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

## 10. Verifier laws (R1–R25)

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
`-bcir-verify` pass enforces the structurally checkable form of all of R1–R25
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
(§17's ASN.1 rail and §18's profile, [`BCIR_ASN1_X690_ABI.md`](BCIR_ASN1_X690_ABI.md)). R24 checks the
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
§8.11.2); a `strict_der` or `strict_canonical` decode does not also declare it accepts
a non-canonical syntax, and `strict_der` is not claimed outside X.690; a
`bcir.asn1.transcode` emits a canonical target, does not transcode a syntax to itself,
and reads a canonical source when it claims `preserve_value`; and an
`asn1.projection` is marked **additive** — a projection that replaced a frozen wire
format would invalidate every digest taken over the native octets, so the IR refuses to
express one. Vacuous for IR with no `bcir.asn1.*` operation (the non-disturbance
invariant R14–R23 also hold to). Oracle twins: `asn1.schema` and `asn1.der`.

`#bcir.asn1_rules` names **every transfer syntax the repository speaks** — the X.690
three, X.691's four PER variants, X.696's OER/COER, X.693's XER/CXER, and X.697's JER
plus BCIR's canonical JER profile. It is one enum rather than a (family, profile) pair
of attributes, because a transfer syntax is what a peer either speaks or does not;
family, canonicality and PER alignment are **derived** from it in `BCIRDialect.cpp`, so
no two attributes can disagree about one syntax. `ber`/`cer`/`der` keep integer values
0/1/2, so the extension is additive and pre-existing artifacts are unaffected.

R24's emission law is stated over the derived predicate: **BCIR emits only a syntax
whose octets are a function of the abstract value**, because it digests what it emits.
DER is the X.690 member of that set; CANONICAL-PER, COER, CXER and BCIR's canonical JER
are the others. `cer` fails it despite its name — X.690 §9.1 makes the indefinite length
form mandatory for constructed CER encodings, so a CER artifact is not byte-stable.
`strict_canonical` is the family-neutral spelling of `strict_der`, which R24 now holds
to X.690. `bcir.asn1.transcode` names one value in two syntaxes.

JER *instruction* legality and compiled-descriptor identity are still future work in
[`BCIR_ASN1_JSON_ROADMAP.md`](BCIR_ASN1_JSON_ROADMAP.md).

**X.692 ECN encoding-definition legality (R25).** Over the `bcir.ecn.*` operations, which
mirror [`ecn_syntax.py`](../bcir/asn1/ecn_syntax.py)'s `EcnModule`: a module holds encoding
classes assigned from built-in ones (clause 11), one §16.5 concatenation structure whose
textual order is the transmission order, and the encoding objects that realize those classes.

R25 exists for a sharper version of R24's reason. An encoding definition module is written
once and applied to *many* types, so a fault that only fires on the right value can sit in one
indefinitely. Every rule below is a statement about the specification, decidable with no value
in hand: §9.5.2's one object per class in a set; §21.11.5's Comparison required by the last
three range conditions and forbidden to the first five; §22.2.2.2's `ALIGNED TO ANY` requiring
a start pointer; §22.8.2.2's `USING` present **if and only if** the determination is not
`not-needed`, with §22.8.2.3 and §22.8.2.5 confining each transform list to one determination;
§22.12.2.3 and §21.14.5 on the unit a bit reversal divides; §23.7.2.4's one of `IF`/`IF-ALL`/
`ELSE`; §23.7.2.7's `subtract:lower-bound` only under a condition that guarantees a lower
bound; and §22.1.2.8's `REPLACE STRUCTURE` forbidding `INSERT AT HEAD` and requiring
`ENCODED BY`. Negative fixtures are in `mlir/test/passes/verify_ecn.mlir`, one per rule, with
two positive modules so an over-firing law is caught too.

The bit-level *values* — patterns, widths, transform operands — stay in the oracle, for the
same reason the ASN.1 ops carry a type's shape and not its values: "this pattern is 0101" is
not a proposition that can be false. What the IR holds is every property whose *combination*
with another property can be.

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
**R1–R25**. R21 detection runs on both driver rails; it is *advisory* by default
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

## 17. ASN.1 in BCIR

ASN.1 is BCIR's **external contract language**. A StreamPack, a BCAB bundle and a
telemetry frame all cross a boundary where the peer is not BCIR, and X.680's abstract
syntax plus a named transfer syntax is how that boundary is stated once and checked on
every rail. This section describes what is built; §18's profile bullet states the
interoperability boundary, and the exclusions live in
[`BCIR_ASN1_BUILDOUT_ROADMAP.md`](BCIR_ASN1_BUILDOUT_ROADMAP.md).

### 17.1 Three rails, one specification

The same ASN.1 fact is carried on all three of BCIR's rails, and a disagreement between
them is a bug in whichever rail is wrong — never a tolerance:

| Rail | Artifact | What it answers |
|---|---|---|
| Python oracle | `bcir/asn1/`, `bcir/frontends/asn1/` | what the octets ARE, executably |
| MLIR law | `bcir.asn1.*` (R24), `bcir.ecn.*` (R25) | what a schema may legally SAY, statically |
| C twin | `runtime/c/bcir_{asn1,per,per_plan,oer,jer,xer}.c` | what a freestanding driver can READ |

The C twins are differentials, not reimplementations: each is driven by a corpus the
Python rail encoded, and the gate compares field values, not just exit codes. Two of this
build's sharper defects were found exactly there — a decoder that reported a mid-octet PER
string at the octet *containing* it, and an MLIR verifier rule reading a neighbouring
attribute group — and both were invisible to a gate that compared a rail against itself.

### 17.2 Transfer syntaxes

`#bcir.asn1_rules` names every transfer syntax the repository speaks. **Canonicality is the
selection criterion**: a rule with no canonical variant may be decoded but never chosen for
emission, because a selected encoding becomes a digested artifact.

<!-- claim: asn1-c-twins-exist -->
| Recommendation | Rules | Canonical member | C twin |
|---|---|---|---|
| X.690 | BER, CER, DER | DER | `bcir_asn1.c` |
| X.691 | PER aligned/unaligned × basic/canonical | CANONICAL-PER | `bcir_per.c`, `bcir_per_plan.c` |
| X.696 | OER, COER | COER | `bcir_oer.c` |
| X.693 | XER, CXER | CXER | `bcir_xer.c` |
| X.697 | JER | BCIR's canonical JER | `bcir_jer.c` |

`cer` fails the canonicality test despite its name: X.690 §9.1's indefinite-length
constructed form gives one abstract value more than one octet string.

**Two decode tables, never merged.** X.691 §7.2 and X.696 §6.2 deny PER and OER a
schema-*free* decode permanently — that is a property of the rules, not a missing feature —
so the schema-free and schema-directed tables are separate, and a certificate records which
one a measurement came from (`decode_kind`). Collapsing them would let a rule that cannot be
walked without a schema appear to compete with one that can.

### 17.3 The front end and the object layer

Module text compiles through `bcir/frontends/asn1/` (the `bcir-asn1c` CLI): X.680 types,
tags, constraints and extension markers, with unsupported notation failing **closed**.

- **X.681** information objects, object sets, and open types resolved through their
  associated tables.
- **X.682** table, component-relation and user-defined constraints. These are deliberately
  *not* PER-visible (X.691 §10.3.4/§10.3.5), so they narrow no field's width — a verifier
  reads them, no encoder does.
- **X.683** parameterization, including the `{<`/`>}` spelling ECN's Annex C rewrites it
  into.

Constraints are load-bearing rather than advisory: for PER and OER a PER-visible constraint
*is* the width. `INTEGER (0..255)` occupies eight bits with no length determinant; the same
type unconstrained costs a determinant plus a minimum-octets two's-complement field.

### 17.4 X.692 ECN — all three parts

ECN is where an encoding stops being a fixed rule and becomes a **specification with its own
digest**. All three parts are built: the class/object/object-set model with EDM/ELM and the
seven built-in BER/PER sets (`ecn.py`); the user-defined half — bit-level encoding spaces,
justification, `#PAD`, stated transmission order, clause 24's nineteen `#TRANSFORM`s and
`#OUTER` (`ecn_user.py`); and clause 20's defined syntax read from an
`ENCODING-DEFINITIONS` module (`ecn_syntax.py`), so a specification can be hashed rather
than assembled in Python.

`_UNSUPPORTED_KEYWORDS` holds **no unbuilt group**. Every row remaining is a group that *is*
built and is refused where a clause forbids the way it was written — which is the useful
kind of refusal, since it cites the subclause that says so.

ECN answers a question the other rules cannot: a scaled-length frame header that none of
DER, canonical PER, COER or CJER reproduces. Canonical PER lands on the same octet *count*
and different octets, so the gap is expressiveness rather than size — and a test fails if any
candidate ever matches.

### 17.5 Plans, and why an ECN encoding is not a sixth column

`encode_plan.py` compiles a schema into a deterministic descriptor the C emitter executes,
and `jer_plan.py`/`graph.py` do the same for the JSON rail. An ECN encoding is **not** a
sixth column in that plan: the plan describes an ASN.1 *type*, while a frame header's wire
order and reserved bits are properties of an encoding *structure*. `EncodeNode` has a slot
for neither, and adding them would make a node's meaning depend on which candidate read it.
It is a third compilation with its own version counter and its own digest.

### 17.6 Encoding selection, and the two-truth boundary

`K_BCIR` may select encoding rules the way it selects a lane width, governed by three laws
that this section restates because they are where encoding selection most easily goes wrong:

- **legality first** — an encoding is a candidate only if the abstract value is
  representable in it. That is a verifier question and never a cost question.
- **two-truth (§14)** — a measured encode/decode cost is *graded* truth. It informs
  selection; it never becomes a legality verdict.
- **canonical or excluded** — see §17.2.

Selections are certified (`certified.py`): a certificate names the evidence, and a frozen
cost table names measurements rather than a string. On this host both of the selection gates
are answered, and the two axes disagree — COER wins encode while CANONICAL-PER-ALIGNED wins
schema-directed decode. A single-row table would have hidden that, which is the reason there
are two.

### 17.7 The C twins' contract

Each twin is **freestanding** — `<stddef.h>` and `<stdint.h>` only, no allocation, no libc,
no recursion — and **total**: for any octets and any plan, every entry point returns a status
and never reads outside the buffer it was given. A width, count or fragment header that would
run past the end is a diagnosed refusal, not a read.

Nothing is copied. A decoded string is reported as an offset and a length into the caller's
own buffer, because the caller already owns it and a decoder that copied would be choosing an
allocation policy on the caller's behalf. Where PER leaves a string on a non-octet boundary —
X.691 §16.6 permits exactly that, in both variants — the exact bit offset is reported and the
octet index is withheld, since an index rounded down to the containing octet names plausible
bytes that are off by a few bits.

The twins are gated under `-Werror` at `-std=c11` and `-std=c2x`, compared at `-O0` against
`-O3`, and fuzzed under ASan/UBSan with the **plan** fuzzed alongside the octets — a
descriptor and a document that came from different places is the ordinary case, not the
exotic one.

## 18. Conformance profiles and external-contract boundary

This reference defines semantics; an implementation must name the profile it supports
and reject work outside it. The current profiles are:

- **MLIR law profile:** ODS/TableGen dialects plus `-bcir-verify` R1–R25 and the
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
- **ASN.1 interoperability profiles:** the MLIR law profile is the whole of X.690
  (02/2021) clause 8 over X.680 tag assignments, restricted by clauses 10 and 11 on
  emission — **DER out, BER in**. CER is accepted as BER input and never emitted.
  The shared Python schema/oracle additionally implements the documented subsets of
  X.681 information objects, X.682 constraints/table resolution, X.683
  parameterization, canonical aligned/unaligned PER, COER/OER, CXER/XER, Python-oracle
  JER with all six X.697 instruction families, and <!-- claim: ecn-three-parts-built -->**all three parts of ECN** — the
  built-in model, the user-defined half, and clause 20's defined syntax read from module
  text (§17.4). Each profile
  has its own explicit exclusions and C coverage in
  [`BCIR_ASN1_BUILDOUT_ROADMAP.md`](BCIR_ASN1_BUILDOUT_ROADMAP.md); the MLIR
  `asn1_rules` attribute names every one of them, and R24's emission check is stated
  over canonicality rather than over X.690.
  Module text is compiled by the X.680 front end (`bcir/frontends/asn1/`, the
  `bcir-asn1c` CLI), and unsupported notation fails closed. JER remains JSON text and has
  no *standardized* canonical variant — BCIR emits its own, which is why §17.2 names it
  that way rather than "CJER". <!-- claim: jer-has-all-three-rails -->It **does** now have all three rails: the scalar C twin
  (`bcir_jer.c`), the MLIR family/profile inside `asn1_rules`, and the direct-claims plan
  (`jer_plan.py`); [`BCIR_ASN1_JSON_ROADMAP.md`](BCIR_ASN1_JSON_ROADMAP.md) owns what
  remains of that promotion path.
- **BCAB artifact-bundle profile:** canonical multi-image directory, integrity checks,
  standard payload identity, deterministic compatibility selection, and Python/C/C++/MLIR
  parity. Its additive ASN.1 projection provides DER/BER, COER/OER, and canonical
  aligned/unaligned PER transfer syntaxes without changing native BCAB kind IDs or
  bytes. Payload assembler, linker, loader, and ISA semantics remain external standards.

The following versioned contracts are adjacent to BCIR semantics but have dedicated
normative documents because their byte/lifecycle evolution is independent:

- StreamPack v1–v3: [`BCIR_STREAMPACK_ABI.md`](kernel/BCIR_STREAMPACK_ABI.md), whose
  native octets stay frozen; its **additive** ASN.1 module and DER projection live in
  [`BCIR_ASN1_X690_ABI.md`](BCIR_ASN1_X690_ABI.md);
- Artifact Bundle v1: [`BCIR_ARTIFACT_BUNDLE_ABI.md`](kernel/BCIR_ARTIFACT_BUNDLE_ABI.md);
  its additive ASN.1 projection is specified by
  [`BCIR_ASN1_X690_ABI.md`](BCIR_ASN1_X690_ABI.md);
- schema-bound JER compilation:
  [`BCIR_ASN1_JSON_ROADMAP.md`](BCIR_ASN1_JSON_ROADMAP.md), which keeps JSON off the
  privileged execution path;
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

## 19. Thesis

> BCIR is a registry-first, phase-ordered, lane-typed, cost-governed
> correspondence IR. K_BCIR is the IR-level optimization calculus that selects
> legal physical realization paths. GEM is the execution IR that hydrates
> selected correspondence paths into streamed lane schedules. MLIR is the
> bootstrap framework used to define, verify, rewrite, and lower BCIR until BCIR
> has enough mass to become its own compiler toolchain.


# BCIR comprehensive system report

**Report basis:** fresh, source-level reconstruction from the current repository rather than reliance on the compacted conversation.

## 1. Provenance and scope

| Item | Verified state |
|---|---|
| Repository | `Mecca-Research/Binary-Correspondence-Intermediate-Representation` |
| Branch | `main` |
| Local `HEAD` | `997511de91c59328759c6eec29602ab34344e206` |
| `origin/main` | `997511de91c59328759c6eec29602ab34344e206` |
| Working tree | Clean |
| Commit date | 2026-08-10 |
| Latest commit | `Merge pull request #740 from Mecca-Research/agent/tmsao-gemplus-architecture` |

The latest commit primarily introduces:

```text
docs/research/BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md
```

That document is a **research architecture proposal**. Its GEM+, TMSAO, service, IPC, driver, and distributed-runtime designs must not be confused with landed production implementations.

This report covers the execution-bearing system:

- semantic model and verifier laws;
- K_BCIR planning and optimization;
- GEM scheduling and execution;
- StreamPack and artifact formats;
- frontends and ETL;
- ASN.1;
- MLIR/C++;
- freestanding and hosted C;
- lowering and target code generation;
- hardware channels and memory planning;
- telemetry and calibration;
- AI, machine learning, model ingestion, inference, and training;
- security boundaries;
- testing, CI, documentation, and LLVM education;
- the latest GEM+/TMSAO proposal as a proposal.

It deliberately does **not** prioritize shortcomings or prescribe next development work yet.

---

# 2. What BCIR is

BCIR is not one conventional compiler IR. It is a multi-rail system for:

1. expressing a computation as a conservative graph of typed resource claims;
2. checking static legality independently of optimization;
3. enumerating legal realizations of those claims;
4. selecting a realization under target, runtime, cost, and budget constraints;
5. scheduling and freezing the selected result as a portable execution artifact;
6. lowering that artifact or plan to C, LLVM, MLIR, object code, WASM, and bounded stack-machine formats;
7. executing through native runtimes;
8. collecting telemetry;
9. admitting measured information back into future planning without allowing measurement or learning to redefine legality.

The central planning formulation is:

\[
K_{\text{BCIR}}(G \mid H,\Theta)
=
\min_{\pi} C_H(\pi,\Theta)
\]

where:

- \(G\) is the semantic claim graph;
- \(H\) is the target or substrate profile;
- \(\Theta\) is live runtime state;
- \(\pi\) is a path through legal realization candidates;
- \(C_H\) is a multidimensional, context-coupled integer cost.

The design’s most important separation is:

> **Legality is deterministic. Cost, telemetry, measurements, and learned models may rank legal alternatives, but they cannot make an illegal alternative legal.**

Primary sources:

- `docs/BCIR_LANGREF.md`
- `bcir/model/`
- `bcir/verify/`
- `bcir/kbcir/`
- `bcir/gem/`
- `mlir/lib/passes/BCIRVerifyPass.cpp`

---

# 3. Authority hierarchy

BCIR has several implementations that serve different purposes.

## 3.1 Normative language and law rail

The authoritative specification consists principally of:

- `docs/BCIR_LANGREF.md`;
- production MLIR ODS definitions;
- `-bcir-verify` in `BCIRVerifyPass.cpp`;
- frozen ABI documents under `docs/kernel/`.

The MLIR verifier is the production static-law rail.

## 3.2 Python executable conformance oracle

The Python package under `bcir/` is the broadest executable implementation. It contains:

- the semantic graph;
- candidate generation;
- cost model;
- optimizer;
- schedulers;
- StreamPack reference codec;
- ASN.1 reference implementation;
- telemetry and calibration;
- model and ML research organs;
- reference numerical implementations;
- differential and fuzz generators.

It is broader than the production C and MLIR implementations and is often the first rail on which a capability is made executable.

## 3.3 Freestanding and hosted C

`runtime/c/` supplies:

- no-libc/freestanding codecs and execution components;
- hosted compiler and ownership components where diagnostics or allocation are necessary;
- C twins of selected Python semantics;
- bounded trust-boundary implementations;
- native AI and model kernels.

It is intentionally narrower than the Python oracle.

## 3.4 C++ adapter layer

`runtime/cpp/` provides hosted adapters:

- BCAB RAII wrapper;
- JER SIMD and structural-index acceleration;
- a real single-node StreamPack orchestration seam;
- explicit dynamic-graph and distributed orchestration stubs.

## 3.5 Educational rail

`llvm-training/` is an independent educational and evaluation corpus. It teaches and tests LLVM/MLIR concepts but does not define BCIR legality.

## 3.6 Research rail

`docs/research/` contains proposals, audits, measurements, and architecture investigations. These become system capabilities only when backed by source and tests elsewhere.

---

# 4. Mechanical repository inventory

The fresh checkout contains **1,750 tracked files**.

| Tree | Tracked files | Role |
|---|---:|---|
| `llvm-training/` | 612 | LLVM/MLIR curriculum, exercises, fixtures, autograder |
| `bcir/` | 514 | Python oracle, tests, frontends, optimization, models |
| `runtime/` | 314 | C and C++ production/runtime rails |
| `mlir/` | 186 | ODS dialect, verifier, passes, examples, fixtures |
| `docs/` | 50 | normative, status, architecture, roadmap, research |
| `tools/` | 49 | validation, code generation, C, MLIR, model and performance gates |
| `channels/` | 5 | channel manifests and examples |
| `.github/` | 6 | CI and scheduled validation |

Selected subtrees:

| Subtree | Tracked files |
|---|---:|
| `bcir/tests/` | 250 |
| `bcir/kbcir/` | 80 |
| `bcir/frontends/` | 56 |
| `bcir/asn1/` | 42 |
| `bcir/hosted/` | 22 |
| `runtime/c/` | 303 |
| `runtime/cpp/` | 11 |
| `mlir/test/` | 117 |

A current ODS scan found:

- **133 operation definitions**
- **37 registered passes**

Operation-family count:

| ODS family | Operations |
|---|---:|
| Core | 27 |
| K_BCIR | 20 |
| GEM | 19 |
| Verify | 12 |
| ASN.1 | 7 |
| ECN | 7 |
| Binary format | 7 |
| Lowering contract | 6 |
| Optimization | 6 |
| Transducer | 6 |
| Event | 4 |
| Parse | 4 |
| Memory | 3 |
| Async | 2 |
| Trace | 2 |
| Target | 1 |
| **Total** | **133** |

---

# 5. End-to-end architecture

The principal execution path is:

```text
program / source / schema / manifest / binary record
        │
        ▼
frontend or programmatic graph construction
        │
        ▼
BCIR resources + claims + explicit phase DAG
        │
        ▼
R1–R25 legality verification
        │
        ▼
legal candidate enumeration
        │
        ▼
K_BCIR optimization under H, Θ, policy and optional budgets
        │
        ▼
selected plan + candidate costs + provenance/certificates
        │
        ▼
GEM scheduling and StreamPack hydration
        │
        ├── portable C / LLVM IR / MLIR / object / WASM / JVM / CIL
        ├── BCAB multi-backend bundle
        └── freestanding StreamPack execution
                │
                ▼
telemetry / measurement / calibration evidence
                │
                ▼
frozen generation-tagged table for a later replan
```

## 5.1 Stable embeddable facade

`bcir/api.py` exposes a compact library interface:

- `build_artifact(...)`
- `compile_kernel(...)`
- `KernelArtifact`

A resulting artifact carries:

- portable C source;
- freestanding C header;
- selected width and operation;
- K_BCIR score;
- calibration generation;
- plan-manifest digest;
- R12 lowering attestation;
- budget and feasibility metadata;
- diagnostics.

A fresh end-to-end AOT check was exercised through this API and returned:

```text
EMBEDDABLE_API_AOT_OK
```

## 5.2 Concrete fresh example

For `vector_add` on the `x86_avx512` profile under cool/latency policy, BCIR produced:

```text
scalar score = 32768
vec8 score   = 9472
vec16 score  = 7808
selected     = vec16
```

It then produced:

- one StreamPack segment;
- generated BCIR MLIR;
- portable C23;
- a provenance digest;
- a successful compiled self-check.

This demonstrates the actual source-to-plan-to-lowering path rather than only a documented architecture.

---

# 6. Semantic model

Primary sources:

- `bcir/model/graph.py`
- `bcir/model/lanes.py`
- `bcir/model/opcodes.py`

## 6.1 Registry-first resources

Memory and state are represented as registry entries rather than unrestricted raw pointers.

A resource records information such as:

- RID;
- name;
- domain;
- count or shape;
- layout and access form;
- generation;
- optional timing/lifetime facts.

Claims refer to resources through RIDs.

This enables:

- explicit identity;
- generation checks;
- provenance;
- resource-domain legality;
- deterministic serialization;
- stale-artifact rejection.

## 6.2 Claims

A claim expresses an operation together with its effects:

- claim ID;
- opcode;
- phase;
- read resources;
- write resources;
- element count;
- lane;
- stride class and stride;
- domain;
- hazard mode;
- bounds policy;
- optional timing;
- optional lifetime;
- optional tensor shape or lowering metadata on richer rails.

The claim graph describes **what must correspond**, while candidate realizations describe **how it may be executed**.

## 6.3 Explicit ordering

Ordering is not inferred from textual source order.

BCIR uses:

- an explicit phase DAG;
- explicit phase dependencies;
- read/write hazards;
- optional fine-grained async tokens;
- barriers as ordering edges.

## 6.4 Lane taxonomy

The normative lane enum is:

```text
U, UX, T, GGG, A, H
```

The system distinguishes regular/vectorizable, tiled, sparse/random, and specialized access classes. In particular:

- `T` is associated with tiled execution;
- `GGG` is the irregular/random access quarantine and scheduling tail;
- U/UX/T regular work can be scheduled separately from irreducible random work.

## 6.5 Stride classes

```text
SCALAR
UNIT
STRIDED
CACHELINE
TILE
RANDOM
```

## 6.6 Resource domains

```text
RAM
VRAM
NVM
MMIO
CXL
HBM
```

These are semantic domains, not proof that every physical device is installed.

## 6.7 Core opcodes

The base model includes:

```text
NOP
LOAD
STORE
ADD
SUB
MUL
ATOMIC_ADD
ATOMIC_SUB
ATOMIC_XOR
CMPXCHG
BARRIER
PHASE_ENTER
PHASE_LEAVE
GGG_LOAD
GGG_STORE
T_MACC
GEM_DISPATCH
PROV_NOTE
```

Higher-level tensor and format operations are represented on the MLIR and specialized Python rails.

---

# 7. Verifier laws R1–R25

BCIR currently defines twenty-five named law classes.

| Law | Meaning |
|---|---|
| **R1** | Registry uniqueness |
| **R2** | Registry/resource resolution |
| **R3** | Domain legality |
| **R4** | Phase-DAG legality |
| **R5** | Hazard and ordering legality |
| **R6** | Lane legality |
| **R7** | Bounds legality |
| **R8** | Cost completeness |
| **R9** | Plan legality |
| **R10** | Stream provenance |
| **R11** | Generation validity |
| **R12** | Lowering-contract legality |
| **R13** | Policy and decision provenance |
| **R14** | CIM/PIM dispatch legality |
| **R15** | DVFS clock legality |
| **R16** | Allocator/tier placement legality |
| **R17** | Numerical-accuracy contract |
| **R18** | Compositional call-graph integrity |
| **R19** | Synchronous timing legality |
| **R20** | Clock-domain crossing legality |
| **R21** | Pointer lifetime: use-after-free/double-free |
| **R22** | Shape consistency |
| **R23** | Dtype compatibility |
| **R24** | ASN.1 schema and encoding-rule legality |
| **R25** | X.692 ECN definition legality |

## 7.1 Important details

### R5: hazards, volatile and barriers

- Volatile access is represented structurally.
- Volatile claims require ordered/barriered semantics.
- Barriers prevent reordering and inhibit fusion across the barrier.
- Atomic RMW and compare-exchange carry explicit memory ordering on MLIR.

### R10–R13: artifact identity

BCIR checks:

- source-plan provenance;
- topology/mapping/data generations;
- lowering support;
- target identity;
- policy and calibration generation;
- hashes/digests of planning inputs.

R13 recomputes and cross-checks manifest components rather than trusting a supplied digest.

### R14–R17: smart-lowering laws

- PIM dispatch is restricted to eligible reduction forms.
- DVFS uses bounded Q8 clock factors and forbids inappropriate PIM overclocking.
- allocator placement observes tier capacities;
- numerical error must remain within the declared tolerance.

These laws are vacuous when their optional structures are absent.

### R18

Every call must resolve, and the call graph is acyclic. Recursive planning is not silently approximated.

### R19/R20

Optional timing metadata supports:

- clock frequency;
- setup/hold margin;
- synchronous and mixed synchronization;
- cross-domain RAW dependency checks.

### R21

The compiler can detect:

- use after free;
- double free;
- revalidation after assignment/allocation.

For the C frontend, policy can be:

```text
advisory
fallback
reject
```

### R22/R23

These cover tensor seams such as:

- matmul → activation;
- conv/attention → activation;
- embedding → RMSNorm;
- RoPE → attention;
- KV cache → GQA attention;
- native device tile multiples.

### R24

The law rail checks static ASN.1 properties before a value exists, including:

- module OID validity;
- legal universal tags;
- primitive/constructor consistency;
- sequence/set-of element structure;
- unique component tags;
- OPTIONAL versus DEFAULT;
- DEFAULT value presence;
- SET discriminability;
- canonical emission;
- transcode source and target rules;
- additive projections.

### R25

R25 governs combinations of ECN structures and properties whose invalidity is statically decidable. Current sources cite and enforce dozens of X.692 subclauses, including conditions involving:

- encoding object/class uniqueness;
- range-condition comparisons;
- alignment and start pointers;
- determinant usage;
- transform placement;
- bit-reversal units;
- conditional encoding objects;
- replacement structure rules;
- identification handles;
- repetition and constructor rules.

Primary source: `docs/BCIR_LANGREF.md:166-378`.

---

# 8. K_BCIR optimizer

Primary sources:

- `bcir/kbcir/cost.py`
- `bcir/kbcir/realize.py`
- `bcir/kbcir/rcsp.py`
- `bcir/kbcir/provenance.py`
- `bcir/kbcir/compose.py`
- `bcir/kbcir/egraph.py`

## 8.1 Twelve-dimensional cost vector

Every candidate can carry:

```text
compute
memory
fabric
sync
compile
thermal
power
reliability
security
accuracy
contention
verification
```

Costs are integer-valued. Context coupling uses fixed-point/Q8-style arithmetic rather than host floating point in the planning hot path.

## 8.2 Substrate profile \(H\)

A target profile describes facts such as:

- architecture/triple;
- ISA features;
- lane widths;
- vector width;
- cache line;
- gather penalty;
- affinity domains;
- memory channels;
- thermal and power density;
- base overhead;
- memory hierarchy;
- calibration generation.

Built-in cost-model profiles:

```text
x86_avx2
x86_avx512
arm64_neon
arm64_sve
riscv_rvv
nvidia_ptx
```

## 8.3 Runtime state \(\Theta\)

Theta carries runtime pressures such as:

- thermal;
- power;
- memory pressure;
- contention.

Standard examples include cool, hot, and memory-bound states. Real telemetry can be folded into a later generation through a calibrated table.

## 8.4 Candidate generation

For each claim, BCIR can enumerate legal realizations such as:

- scalar;
- vector width \(W\);
- strided;
- gather;
- UX bucket;
- tiled;
- fused variants;
- target-specific options.

An unsupported candidate is not assigned a high cost; it is removed from the legal candidate set.

## 8.5 Min-plus selection

`realize.optimize` constructs a layered realization DAG and performs deterministic min-plus selection. Context-dependent factors can account for:

- adjacent shared operands;
- fusion;
- target lane width;
- thermal pressure;
- memory tier;
- path context.

## 8.6 Constrained optimization

`rcsp.py` implements resource-constrained shortest-path selection:

\[
\min M(\pi)
\quad\text{subject to}\quad
R(\pi,\Theta)\le B(H,\Theta)
\]

Capabilities include:

- hard dimension budgets;
- feasibility checks;
- dominance pruning;
- Pareto-plan enumeration;
- explicit `Infeasible` refusal.

This is stronger than merely changing scalarization weights.

## 8.7 Overlap-aware planning

`gem/overlap.py` prices the actual concurrent schedule rather than only a serial sum:

- parallel work on separate domains combines by maximum;
- time-sharing combines additively;
- context coupling can change the best plan once scheduling is considered.

## 8.8 Composition and control flow

`compose.py` extends planning to a bounded region tree:

- leaf;
- sequence;
- conditional;
- call;
- function summary.

It supports conservative effects, undefined-callee refusal, and acyclic calls.

## 8.9 E-graph composition

`egraph.py` provides:

- e-classes;
- expression nodes;
- equality rewrites;
- constant folding;
- factorization;
- congruence rebuilding;
- saturation statistics;
- minimum-cost extraction;
- shared-block/CSE analysis.

`memory.py` wraps bounded saturation into a content-addressed “memory module” with fixpoint and idempotence checks.

## 8.10 Provenance and replay

A `ProvenanceManifest` binds:

- module;
- target;
- Theta;
- policy;
- calibration table;
- decision-rule generations and fingerprints.

The system supports deterministic replay and tests manifest equality against reproduced plans.

## 8.11 Calibration and learned guidance

Implemented mechanisms include:

- host microbenchmarking;
- quantized/frozen calibration tables;
- EWMA and linear calibrators;
- adaptive policy selection;
- regret ledgers;
- tile priors;
- channel priors;
- proposal-order accelerators;
- bounded MoE/GNN research;
- optimization-memory lookup.

The design rule remains:

> A learned mechanism may propose or reorder candidates; deterministic search and verification establish the result.

---

# 9. GEM scheduling and execution

Primary sources:

- `bcir/gem/streampack.py`
- `bcir/gem/concurrency.py`
- `bcir/gem/schedule.py`
- `bcir/gem/async_tokens.py`
- `bcir/gem/execute.py`
- `bcir/gem/dvfs.py`
- `bcir/gem/cim.py`

## 9.1 StreamPack hydration

GEM turns a selected plan into a hot executable `StreamPack`.

The dormant artifact is the semantic graph and plan. The hot artifact contains selected, ordered execution segments.

## 9.2 Deterministic execution

`execute.py` runs:

- phases in topological order;
- claims in deterministic claim-ID order where deterministic mode applies;
- optional per-claim kernels;
- per-phase telemetry.

## 9.3 Concurrent wave scheduler

`concurrency.py` forms waves:

- independent claims share a wave;
- RAW/WAR/WAW conflicts serialize;
- work is assigned across affinity domains;
- irregular GGG work can be placed into a separate tail stream.

## 9.4 Duration-aware scheduling

`schedule.py` adds:

- duration-aware LPT priority;
- earliest-finish-time placement;
- locality;
- memory-bandwidth knee calculations;
- event-driven scheduling;
- deterministic tie breaking.

## 9.5 Async token graph

`async_tokens.py` emits explicit dependencies:

- each claim forks a completion token;
- conflicting later claims await the relevant tokens;
- independent claims await nothing.

This permits fine-grained pipelining without adding unnecessary phase-wide barriers.

## 9.6 DVFS and power

The DVFS planner:

- classifies compute-bound versus memory-bound phases;
- proposes target clock factors;
- observes thermal and power constraints;
- can quantize plans to available silicon states;
- separates proposed plans from actual actuation results.

## 9.7 CIM/PIM

Eligible reduction segments can be annotated for PIM-style dispatch. This changes where a segment is intended to execute, but does not itself prove the presence of physical processing-in-memory hardware.

## 9.8 Power-rail scheduling

The scheduler can reason about concurrent phase power and produce power-rail decisions rather than treating every legal concurrent launch as thermally free.

---

# 10. StreamPack ABI

Primary sources:

- `bcir/abi/streampack_abi.py`
- `runtime/c/bcir_streampack.h`
- `runtime/c/bcir_runtime.c`
- `docs/kernel/BCIR_STREAMPACK_ABI.md`

## 10.1 Format identity

```text
magic:       BSPK
base version: 1
maximum supported version: 3
header size: 64 bytes
endianness: little-endian
```

## 10.2 Versioning

- **v1** is the frozen base ABI.
- **v2** adds pipeline/double-buffer information through reserved header space.
- **v3** adds segment dispatch/channel information.
- Changes are append-only and versioned.

## 10.3 Contents

A pack contains:

- source plan/provenance text;
- generation numbers:
  - topology;
  - mapping;
  - data;
- lane segments;
- prefetch records;
- blocks;
- trace notes;
- pipeline depth;
- per-segment dispatch and channel in later versions;
- integrity trailer/CRC.

## 10.4 Validation

The reference and C implementations check:

- magic;
- version;
- record boundaries;
- count/range arithmetic;
- CRC;
- trailing bytes;
- semantic consistency;
- generation where a live registry is available.

## 10.5 Tooling

`bcir-pack` provides:

```text
bcir-pack dis
bcir-pack hexdump
```

It validates before interpretation and supports a maximum input-size limit.

## 10.6 Native round trip

The C rail includes:

- decoder;
- encoder;
- claim graph + plan hydrator;
- deterministic executor;
- semantic verifier;
- Python/C byte-identity tests.

---

# 11. BCAB multi-backend artifact bundle

Primary sources:

- `bcir/abi/artifact_bundle.py`
- `runtime/c/bcir_artifact_bundle.{h,c}`
- `runtime/cpp/bcir_artifact_bundle.hpp`
- `docs/kernel/BCIR_ARTIFACT_BUNDLE_ABI.md`

## 11.1 Wire contract

```text
magic: BCAB
version: 1
header: 128 bytes
directory entry: 448 bytes
maximum entries: 1024
default hard size ceiling: 1 GiB
```

BCAB is a deterministic container. It does not replace standard payload formats.

## 11.2 Supported artifact kinds

```text
STREAM_PACK
ELF_OBJECT
ELF_SHARED
COFF_OBJECT
MACHO_OBJECT
ARCHIVE
WASM
LLVM_BITCODE
LLVM_IR
PTX
CUBIN
SPIRV
JVM_CLASS
CIL
C_SOURCE
CPP_SOURCE
SYCL_SOURCE
ASSEMBLY
ELF_EXECUTABLE
PE_EXECUTABLE
PE_SHARED
MACHO_EXECUTABLE
MACHO_SHARED
RAW_BINARY
```

## 11.3 Compatibility selection

A selection envelope can constrain:

- target triple;
- architecture;
- OS ABI;
- channel;
- feature set;
- accepted kind and format masks;
- endianness;
- pointer width;
- machine ID;
- target-manifest SHA-256;
- calibration generation;
- R12 requirement;
- debug allowance.

## 11.4 Tooling

`bcir-bundle` supports:

```text
list
hexdump
extract
select
dis
mnemonics
to-der
from-der
to-oer
from-oer
```

Validation precedes listing, extraction, or disassembly.

## 11.5 C++ wrapper

`ArtifactBundleView` provides:

- RAII-style error mapping;
- borrowed immutable payload views;
- indexed access;
- default or envelope-constrained selection.

---

# 12. Frontends and event transduction

## 12.1 MAP frontend

`bcir/frontends/map.py` reads a terse macro-assembly-like form containing:

- resource declarations;
- domains;
- operations;
- reads/writes;
- counts;
- lanes;
- strides.

It lowers directly into a BCIR `Module`.

## 12.2 ROP frontend

`bcir/frontends/rop.py` reads a declarative brace-delimited registry-first language containing:

- modules;
- resources;
- phases;
- claims;
- reads and writes;
- lane and stride metadata.

## 12.3 Event Transduction Layer

`bcir/etl/` provides a more general ingestion architecture:

- regex lexer;
- recursive-descent ROP parser;
- typed event streams;
- token-driven finite-state transducers;
- structured binary record descriptors;
- packed-field decoding.

The binary record decoder currently supports byte-aligned fields and fails closed on unsupported non-byte-aligned layouts.

The intended correspondence is:

```text
text / bytes / packet / telemetry
    -> tokens or fields
    -> events
    -> BCIR claims
    -> normal verification and planning
```

## 12.4 C frontend

Primary sources:

- `bcir/frontends/cfront/`
- `runtime/c/bcir_cfront.{h,c}`
- `runtime/c/bcir_cpp.{h,c}`
- `docs/languages/CFRONT_GUIDE.md`
- `docs/languages/C_MEMORY_DISCIPLINE.md`

The C frontend is a bounded driver/kernel-oriented C compiler frontend, not a claim to arbitrary ISO C completeness.

Its supported architecture includes:

- C11, C17 and C23/C2x mode selection;
- preprocessing;
- include paths;
- object/function/variadic macros;
- stringizing and token pasting;
- C23 `__VA_OPT__`;
- conditional compilation;
- dependency generation;
- compile-database ingestion;
- multi-file/project verdicts;
- target ABI selection;
- Clang-style and JSON diagnostics;
- syntax-only mode;
- LLVM fallback signaling;
- R21 policy;
- generated C;
- generated self-checks;
- link flags;
- linkable multi-translation-unit output.

The semantic subset includes driver-oriented constructs such as:

- fixed-width integer operations;
- pointers and arrays within documented bounds;
- structures, unions and bitfields;
- volatile/MMIO accesses;
- atomics;
- fences;
- register-map patterns;
- external library edges;
- source locations and diagnostics.

The Python frontend also contains explicit modeling for GNU inline assembly and port-I/O intrinsics. The native C twin has its own supported subset and explicit portable acquire/release fence emission; unsupported constructs are not silently assigned semantics.

## 12.5 Bounds quarantine

The C lowering can emit guarded access and call:

```c
bcir_bounds_quarantine(...)
```

Two behaviors exist:

- a weak abort-oriented default;
- a strong policy-driven recovery reference capable of actions such as clamping.

This is a controlled runtime policy seam, not an optimizer license to ignore bounds.

---

# 13. ASN.1 subsystem

The ASN.1 implementation is a complete subsystem with independent schema, transfer-syntax, law, runtime, artifact, measurement, and admission layers.

Primary trees:

```text
bcir/asn1/
bcir/frontends/asn1/
runtime/c/bcir_asn1*
runtime/c/bcir_per*
runtime/c/bcir_oer*
runtime/c/bcir_jer*
runtime/c/bcir_xer*
mlir/include/BCIR/BCIRAsn1Ops.td
mlir/include/BCIR/BCIREcnOps.td
```

## 13.1 Schema and semantic model

The Python schema layer supports a documented X.680 subset including:

- primitive types;
- tagged/prefixed types;
- SEQUENCE, SET, CHOICE;
- SEQUENCE OF and SET OF;
- OPTIONAL and DEFAULT;
- constraints;
- imports and assignments;
- open types;
- object identifiers;
- restricted and unrestricted string/time families on their relevant rails.

Unsupported notation fails rather than being approximated.

## 13.2 X.681/X.682/X.683

Implemented capabilities include:

- information object classes;
- `WITH SYNTAX`;
- objects and object sets;
- associated tables;
- table constraints;
- component-relation constraints;
- open-type resolution from a governing sibling;
- user-defined constraints;
- contents constraints;
- parameterized types;
- parameterized objects and object sets;
- structural actual-parameter substitution.

Resolution of an open type is enrichment: original octets remain available, and a resolved value is attached when a matching table row exists.

## 13.3 X.690 BER and DER

The X.690 rail implements the broad clause-8 value surface, including:

- BOOLEAN;
- INTEGER;
- ENUMERATED;
- REAL;
- BIT STRING;
- OCTET STRING;
- NULL;
- OID and relative OID;
- OID-IRI forms;
- SEQUENCE/SET and OF variants;
- CHOICE;
- explicit and implicit tags;
- open types;
- character strings;
- useful/time types.

Policy:

```text
emit: DER
accept canonical storage: DER
accept foreign input: BER
normalize: BER -> DER
```

CER is accepted as BER input where structurally valid but is deliberately not an emission candidate.

## 13.4 X.691 PER

Implemented on the Python rail:

- aligned and unaligned PER;
- BASIC and CANONICAL variants;
- constrained, semi-constrained and unconstrained integers;
- normally-small values;
- length determinants;
- 16K fragmentation;
- sequence/choice/collections;
- extension markers;
- extension addition groups;
- open-type wrappers;
- constraint roots and extension bits;
- permitted alphabets for supported types.

The implementation is validated against the standard’s Annex A examples.

The C side contains:

- a bounded bit reader;
- alignment;
- whole-number and length primitives;
- a schema-directed narrow plan decoder.

PER is not schema-free, by design of the standard.

Documented PER exclusions include types/clauses not required by the current BCIR schemas, such as selected REAL, BIT STRING, EMBEDDED PDV, and unrestricted-character-string paths.

## 13.5 X.696 OER/COER

Implemented:

- BASIC OER input;
- CANONICAL OER emission;
- constraint-dependent fixed-width forms;
- presence preambles;
- canonical SET ordering;
- collection counts;
- supported primitive and constructor types;
- Annex A byte-level validation.

The C decoder is schema-directed, as required by X.696.

## 13.6 X.693 XER/CXER

Implemented on the Python rail:

- BASIC XER;
- canonical XER emission;
- tag processing;
- escaping;
- supported primitive and composite values.

The C twin is intentionally the lexical/trust-boundary layer:

- tag scanning;
- bounded lexical checks;
- `xmlcstring` escaping.

Extended XER is not claimed.

## 13.7 X.697 JER

Implemented capabilities include:

- clauses 20–41 over the supported model;
- BASIC JER;
- BCIR canonical JER profile;
- ARRAY;
- BASE64;
- NAME;
- OBJECT;
- TEXT;
- UNWRAPPED;
- instruction precedence;
- deterministic emission;
- malformed-input rejection.

X.697 defines no standard canonical JER. Therefore:

> `BCIR canonical JER profile` is a private, versioned deterministic profile and carries no invented standards OID.

### Bounded JER reader

The Python bounded reader performs:

1. input and structural bound checks;
2. UTF-8 validation;
3. JSON grammar validation;
4. schema decoding;
5. optional canonical re-encoding and byte comparison.

It returns stable diagnostics including:

- error code;
- byte offset;
- required capacity.

### Framing

The optional BCIR JER frame carries:

- version;
- sequence;
- generation;
- length;
- CRC-32.

A frame must validate before its payload is exposed.

### Native C JER

The C implementation provides:

- no allocation;
- no recursion;
- caller-owned stack and scratch;
- bounded structural scanning;
- UTF-8 validation;
- event parsing;
- raw number-token handling;
- deterministic diagnostics.

Canonical-byte and full schema legality remain on the Python/compiled-plan rails rather than being independently reinvented in the scalar scanner.

### C++ JER acceleration

The hosted C++ layer provides:

- SSE2;
- AVX2;
- AArch64 NEON;
- scalar fallback;
- structural indexing;
- ASCII/UTF-8 fast paths.

The vector path decides where blocks or runs end; the scalar C implementation remains responsible for semantic rejection and exact diagnostic offsets.

## 13.8 X.692 ECN

The ECN implementation now includes the three major portions:

1. class/object/object-set and built-in encoding model;
2. user-defined encoding objects;
3. surface syntax and encoding links.

Capabilities include:

- encoding classes and objects;
- encoding object sets;
- EDM and ELM;
- built-in BER/PER sets;
- bit-level encoding spaces;
- justification;
- padding;
- transmission order;
- determinants;
- start pointers;
- conditions;
- replacement;
- bit reversal;
- identification handles;
- repetition;
- alternatives;
- optionality;
- concatenation;
- containment;
- value mappings;
- encoding links;
- parameterization;
- clause-24 transforms.

The current syntax compiler reports:

```text
ECN syntax version: 13
```

ECN has a production MLIR family and R25 verifier checks; it is no longer only a Python-side notation experiment.

## 13.9 Encoding selection

`selection.py` treats transfer syntax as a realization choice.

The fixed comparison set has ten candidates.

Canonical/selectable:

```text
DER
CANONICAL-PER-UNALIGNED
CANONICAL-PER-ALIGNED
COER
JER-BCIR-CANONICAL
```

Decode/noncanonical targets:

```text
BER
BASIC-PER-UNALIGNED
BASIC-PER-ALIGNED
BASIC-OER
JER
```

Selection objectives include:

```text
none/status quo
wire size
encode latency
decode latency
```

The rules are:

1. representability and round trip first;
2. cost second;
3. noncanonical formats cannot become digested emission choices.

Exact wire length and measured timing are stored separately.

## 13.10 Certified ASN.1 selection

`certified.py` adds:

- distribution-free timing intervals;
- exact integer coverage reporting;
- generation-tagged cost tables;
- candidate indistinguishability;
- deterministic tie-breaking;
- native-table provenance;
- target/corpus identity;
- Pareto and RCSP selection;
- multi-stage byte budgets;
- certificate records.

Timing intervals that overlap do not fabricate a winner. Exact wire size can serve as a deterministic tie-break.

## 13.11 ASN.1 compiled forms

The subsystem has several representations for different purposes:

- `schema.py`: semantic types and constraints;
- `dialect.py`: Python dialect representation;
- `program.py`: canonical program representation;
- `graph.py`: flat node/edge graph capable of cycles without recursive JSON nesting;
- `surface.py`: sparse textual/presentational view;
- `manifest.py`: schema-bound channel/device/selection manifests;
- `encode_plan.py`: schema-directed write plan, currently plan version 5;
- `jer_plan.py`: JER read/schema plan, version 1;
- `staged.py`: candidate admission/install model.

The graph’s canonical JER is content-addressed. Presentation metadata is kept separate so formatting changes do not change semantic identity.

## 13.12 ASN.1 artifact projections

BCIR projects semantic StreamPack and BCAB values into standard encoding families.

For StreamPack:

- native ABI remains authoritative and frozen;
- ASN.1 projection is additive;
- DER can reconstruct native bytes through freestanding C;
- the reconstructed native result is byte-identical, including re-derived version, reserved fields, and CRC.

For BCAB:

- DER/BER;
- OER/COER;
- canonical aligned/unaligned PER;

can round-trip through semantic projections while rebuilding native offsets and integrity fields canonically.

## 13.13 Staged self-modification model

`staged.py` models:

```text
running program
 -> canonical JER proposal as data
 -> schema and R-law verification
 -> K_BCIR selection
 -> compilation
 -> signing/admission
 -> generation-tagged quiescent install
 -> rollback
```

Security properties modeled:

- proposal is data, not executable memory;
- verification happens before compilation;
- compilation calls are observable;
- digest and authority signature are distinct;
- install checks signatures again;
- stale generations are refused;
- in-flight calls block installation and rollback;
- rollback advances generation.

The signature is currently HMAC-SHA256. It demonstrates shared-key admission authority, not deployment-grade asymmetric signer identity.

## 13.14 ASN.1 test surface

The focused inventory established approximately:

```text
50 ASN.1-focused test modules
1,001 ASN.1-focused test functions
```

Coverage includes positive, negative, parity, canonicality, differential, native C, C++, and law-fixture testing.

---

# 14. MLIR implementation

Primary sources:

```text
mlir/include/BCIR/
mlir/lib/
mlir/test/
tools/wsl/
tools/irdl/
```

## 14.1 Dialect families

The current ODS tree defines operation families for:

- core modules, registries, resources, phases and claims;
- K_BCIR plans, paths, candidates, policies and Theta;
- GEM segments and tensor operations;
- verification/provenance records;
- ASN.1;
- ECN;
- parsing;
- finite-state transducers;
- events;
- binary formats;
- async tokens;
- memory;
- optimization;
- lowering contracts;
- targets;
- traces.

## 14.2 Registered passes

The 37 passes are:

### Fundamental and conversion

```text
Verify
PromoteLanes
ConvertToLLVM
```

### Planning and evidence

```text
ClassifyLanes
SelectRealization
CostModel
Plan
Overlap
OverlapOptimize
Sense
Rcsp
RcspPlan
Bundle
Explain
Replay
Compose
```

### Scheduling and smart lowering

```text
Cim
Dvfs
ScheduleEft
Async
PowerRail
AllocPool
Batch
Schedule
LowerToLLVM
CacheContention
LayoutPivot
```

### Tensor cost passes

```text
GemMatmulCost
GemActivationCost
GemConvCost
GemAttentionCost
```

### Tensor lowering/fusion

```text
LowerGemMatmul
LowerGemMatmulBuffer
LowerGemActivation
LowerGemConv
LowerGemAttention
FuseMatmulActivation
```

## 14.3 Production verification

`-bcir-verify` checks the structurally decidable form of R1–R25.

The Python oracle-to-MLIR bridge emits candidate paths and a selected path while allowing the MLIR rail to recompute:

- lane classification;
- legal candidate selection;
- score;
- lowering state.

## 14.4 Tooling and corpus

The MLIR rail includes:

- TableGen validation;
- a compiled `bcir-opt`;
- IRDL projection usable with stock `mlir-opt`;
- positive examples;
- verifier-negative fixtures;
- pass tests;
- lowering tests;
- bytecode round trips;
- FileCheck assertions;
- Python-generated differential corpora;
- target capability matrices.

---

# 15. Lowering and code generation

## 15.1 Portable C23

`bcir/lower/c_kernel.py` is the broad native source backend.

It supports:

- selected lane-aware elementwise kernels;
- bounds-safe tails;
- `restrict` contracts;
- floating-point contraction policy;
- quantized kernels;
- reductions;
- gather/strided work;
- tensor kernels;
- external numerical-provider wrappers;
- self-check harnesses.

The selected full hardware lane may be represented as an idiomatic loop that the resident compiler vectorizes rather than as hard-coded architecture intrinsics.

## 15.2 Textual LLVM IR

`bcir/lower/llvm.py` supports a deliberately bounded subset:

- one selected executable claim;
- two reads;
- one write;
- add/sub/mul;
- scalar or vector LLVM IR;
- AOT compilation and execution.

Unsupported arbitrary graphs are refused rather than truncated.

## 15.3 JIT

`bcir/lower/jit.py`:

- emits the same kernel and self-check as AOT;
- compiles to LLVM IR;
- links with `llvm-link`;
- runs with `lli`.

## 15.4 WASM

`bcir/lower/wasm.py`:

- compiles the bounded kernel through Clang’s `wasm32` target;
- links with `wasm-ld`;
- validates the module;
- executes through Node where available.

## 15.5 Stack-machine lowering

`stackify.py` lowers a small expression graph once to generic stack operations, then maps it to:

- WASM;
- JVM bytecode;
- .NET CIL.

`jvm_class.py` assembles the supported JVM subset directly. CIL tests use `ilasm` and Mono where available.

## 15.6 `llc` target code generation

Codegen descriptors exist for:

| Target | Output |
|---|---|
| `x86_64` | ELF object |
| `aarch64` | ELF object |
| `riscv64` | ELF object |
| `nvptx64` | PTX assembly |
| `bpf` | verifier-oriented scalar ELF object |
| `spirv64` | best-effort SPIR-V assembly |
| `c` | portable C fallback |

SPIR-V depends on an LLVM build containing the relevant backend. Its absence is reported, not replaced by a dummy artifact.

## 15.7 Numerical/tensor emitters

The C lowering contains emitters or wrappers for:

- GEMM and tuned GEMM;
- FFT and 2-D FFT;
- dense linear solve;
- OLS;
- eigensolver/PCA paths;
- GSL statistics;
- SLEEF vector exponential;
- libcerf;
- convolution;
- attention;
- layer normalization;
- recurrent cells;
- SVM;
- trees;
- K-means;
- activation;
- matmul+activation fusion;
- SYCL saxpy/reduce/matmul.

External libraries remain trusted provider edges rather than reimplemented algorithms.

## 15.8 Specialist synthesis

`specialist.py` can produce shape-baked, loop-unrolled C for hot, fixed shapes, while retaining generic kernels for dynamic shapes.

---

# 16. Native C runtime and compiler

`runtime/c/` is split between freestanding and hosted components.

## 16.1 Freestanding philosophy

Core components generally use:

- fixed-width integers;
- caller-owned buffers;
- no hidden heap;
- bounded iteration;
- explicit capacities;
- total status returns;
- checked arithmetic;
- no recursion for attacker-controlled wire structure.

Hosted components introduce allocation or diagnostics only through explicit contracts.

## 16.2 Core components

| Component | Capability |
|---|---|
| `bcir_streampack` | Frozen ABI definitions |
| `bcir_runtime` | StreamPack decode and semantic walk |
| `bcir_encode` | StreamPack encoding |
| `bcir_exec` | Deterministic execution |
| `bcir_hydrate` | C graph+plan to StreamPack |
| `bcir_cir` | Native claim graph |
| `bcir_plan` | Minimal native plan |
| `bcir_verify` | Native verifier subset and diagnostics |
| `bcir_provenance` | Digest twin |
| `bcir_channel` | Channel manifest/routing |
| `bcir_runtime_channel` | In-process driver-hook ABI |
| `bcir_artifact_bundle` | BCAB reader/selector |
| `bcir_binrec` | ETL packed-record decoder |
| `bcir_telemetry_frame` | UART frame codec |
| `bcir_cfront` | Native C frontend |
| `bcir_cpp` | Native preprocessor |
| `bcir_diag` | Source diagnostics |
| `bcir_quarantine` | Bounds policy |
| `bcir_asn1` | X.690 BER/DER |
| `bcir_per` | PER primitives |
| `bcir_per_plan` | Schema-directed PER subset |
| `bcir_oer` | OER subset |
| `bcir_jer` | Bounded JER scanner/parser |
| `bcir_xer` | XER lexical layer |
| `bcir_emit` | Plan-driven ASN.1 encoding |
| `bcir_asn1_streampack` | DER to native StreamPack |
| `bcir_q8_model` | BCIRQ8 loader |
| `bcir_q4_kernel` | Q4/Q8 kernels |
| `bcir_ai_kernels` | Native AI primitives |
| `bcir_decode` | LLM decode-stage kernels |
| `bcir_llama` | Standalone greedy inference |
| `bcir_train` | Native training-stage kernels |
| `bcir_x86_interrupt` | x86 interrupt-frame ABI |

## 16.3 Native verifier coverage

The C verifier documents:

- R1–R8 over the graph;
- R9 over the plan;
- R10–R11 over StreamPack;
- R12 lowering support;
- R13 provenance;
- relevant R14–R18 behavior for its subset;
- advisory R21 lifetime walking.

The full richer R19–R25 law surface remains principally the Python/MLIR responsibility.

## 16.4 Runtime channel ABI

The direct in-process channel interface supports:

```text
init
reset
open
claim
map
submit
sync
next_event
close
```

A loopback implementation exists.

This is an in-process vtable, not an IPC, kernel, or DMA transport.

---

# 17. C++ handoff

## 17.1 Single-node orchestrator

`SingleNodeOrchestrator` is real:

- admits immutable StreamPack through the C verifier;
- creates one whole-pack shard;
- re-enters the freestanding C segment walk;
- returns claim dispatch order;
- is tested against the direct-C path.

## 17.2 Dynamic graph orchestrator

The class and interface exist, but dispatch deliberately raises an error. It documents how a future mutable C++ graph builder would freeze results back into StreamPack.

## 17.3 Distributed orchestrator

The class provides deterministic segment-range sharding logic, but real dispatch is a stub. No MPI/NCCL dependency or cluster runtime is shipped.

Thus:

> C++ single-node handoff is executable; dynamic and distributed execution are explicit non-silent placeholders.

---

# 18. Hardware channels

Primary sources:

- `bcir/channels.py`
- `bcir/channel_plugin.py`
- `channels/*.json`
- `bcir/lower/sycl_dispatch.py`

## 18.1 Built-in channels

```text
x86_avx2
x86_avx512
arm64_neon
arm64_sve
riscv_rvv
nvidia_ptx
fpga_systolic
hbm_pim
nvme_stream
```

The channel set is broader than the six core cost profiles because some channels are modeled extension points rather than CPU target profiles.

## 18.2 Channel manifest

A `channel.json` records sections for:

- format version;
- name and kind;
- provenance;
- modeled status;
- architecture match;
- capabilities;
- code generation;
- runtime hooks;
- calibration;
- target profile.

Channels can be registered through:

- JSON manifests;
- Python entry points;
- core registration.

Duplicate replacement is refused.

## 18.3 Routing

The router derives required capabilities from a claim and selects a suitable channel. It does not route an operation to a channel that lacks the necessary declared capability.

## 18.4 Current execution reality

- Host CPU C/LLVM execution is real.
- Cross-target object/PTX generation is real where LLVM has the backend.
- SYCL has a resident dispatch path when a suitable compiler/runtime is installed.
- FPGA, HBM/PIM, and NVMe channels are modeled planning surfaces without physical device proof in this validation.
- PTX generation is not GPU execution.
- eBPF object generation is not kernel attachment.
- SPIR-V generation is toolchain-dependent.

---

# 19. Device manifests, static memory, and HAM

## 19.1 Device manifest

`device_manifest.py` models immutable hardware facts:

- banks;
- capacities;
- tiers;
- native tile sizes;
- interconnect distance matrix;
- target identity.

Checks include:

- positive capacities and tiles;
- consistent parallel arrays;
- square distance matrix;
- zero diagonal;
- positive off-diagonal;
- symmetry;
- probe agreement;
- explicit bank moves.

Runtime probing may veto disagreement but does not rewrite the static schema.

## 19.2 Static memory planning

`static_memory.py` assigns touched resources to offsets in declared banks.

Capabilities:

- bank binding;
- byte offsets;
- phase-lifetime analysis;
- reuse for disjoint lifetimes;
- overlap refusal for simultaneously live resources;
- independent plan verification.

This is a static compiler artifact, not a runtime allocator.

## 19.3 Hierarchical Access Memory

`ham.py` plans semantic resource movement across memory banks.

Inputs:

- hardware envelope;
- declared directed links;
- capacities;
- resources and access workload;
- optional policy identity.

Outputs:

- movement actions;
- residency actions;
- bank peaks;
- execution plan;
- replay evidence;
- lowered BCIR graph;
- StreamPack.

A route exists only if every required directed link is explicitly present.

HAM does **not** itself:

- issue DMA;
- configure CXL;
- establish peer-to-peer transfers;
- prove physical topology;
- perform live driver operations.

CLI:

```text
bcir-ham-plan --hardware ... --workload ... --plan-out ... [--pack-out ...]
```

---

# 20. Telemetry, silicon signals, and calibration

## 20.1 DataDNA telemetry

`bcir/telemetry.py` records per-segment evidence such as:

- cycles;
- bytes;
- misses;
- thermal pressure;
- voltage pressure;
- utilization;
- provenance;
- generation.

Sinks include:

- null;
- in-memory list;
- validating wrapper;
- file;
- broker;
- shared ring;
- durable log;
- Kafka-oriented adapter.

## 20.2 Telemetry ring and durability

The ring tracks:

- publication;
- overwrite/loss evidence;
- sequence;
- capacity;
- validation.

Durable inputs use bounded, strict parsing and path-identity protections.

## 20.3 UART telemetry frame

```text
magic: BTLM
version: 1
header: 22 bytes
```

The frame is:

- self-delimiting;
- sequence-bearing;
- timestamped;
- CRC-sealed;
- stream-resynchronizable.

Python and C twins are byte-compared.

## 20.4 Signal registry

The signal-provider registry gives typed definitions for:

- thermal pressure;
- die temperature;
- RAPL energy;
- CPU frequency;
- cache capacity;
- PMU availability;
- GPU power;
- BMC power;
- memory bandwidth;
- fabric bytes;
- throttle state;
- reliability;
- hwmon power.

Each reading carries units, provenance, sampling model, and temporality.

## 20.5 Real-host probes

`silicon.py` can inspect, when exposed by the host:

- cache topology;
- cache capacities;
- cpufreq;
- PMU counters;
- RAPL energy;
- on-die thermal;
- OS counters.

Missing signals are reported as unavailable rather than synthesized as measured values.

## 20.6 Derived metrics and export

`telemetry_metrics.py` calculates:

- min/max/average/RMS/count;
- trigger-to-target spans;
- plan-cost sensitivity;
- sampling budgets.

`telemetry_export.py` provides:

- Prometheus text;
- scrape output;
- OTLP-shaped metrics and JSON;
- push adapters;
- Redfish metric reports.

## 20.7 Calibration loop

Telemetry can feed:

```text
measure
 -> validate
 -> quantize
 -> freeze generation-tagged table
 -> replan
 -> compare/replay
```

It cannot change an in-flight plan or relax a law.

---

# 21. Numerical, tensor, and ML capabilities

The Python K_BCIR rail contains a broad numerical and ML reference layer.

## 21.1 Quantization

### Q8

- per-group signed codes;
- shared power-of-two scale;
- dequantization;
- round-trip error;
- integer dot;
- scaled dot;
- accumulator-width analysis.

### Q4

- packed two’s-complement nibbles;
- low-nibble-first format;
- forbidden `-8` to preserve symmetric range;
- group size 32;
- power-of-two scale exponent;
- SmoothQuant-style calibration;
- Q4/Q8 dot;
- format admission evidence.

## 21.2 Tensor primitives

Implemented reference/planning families:

- matmul;
- activation:
  - ReLU;
  - sigmoid;
  - tanh;
  - GELU;
  - softmax;
- 2-D convolution;
- scaled dot-product attention;
- grouped-query attention in the model decoder;
- layer normalization;
- RMSNorm;
- RoPE;
- recurrent cells:
  - RNN;
  - LSTM;
  - GRU;
- transformer encoder block;
- matmul+activation fusion;
- SoA/AoS layout pivot.

Each planned family separates:

- reference result;
- candidate realization;
- cost;
- verification.

## 21.3 Numerical-provider integrations

BCIR wraps rather than reimplements selected mature libraries:

- BLAS GEMM;
- FFTW 1-D and 2-D FFT;
- LAPACK dense solve;
- LAPACK OLS;
- eigensolver/PCA paths;
- GSL statistics;
- SLEEF vector math;
- libcerf.

Provider availability and measured performance are planning evidence. They do not define mathematical or BCIR legality.

## 21.4 Classical and unsupervised ML

Reference capabilities include:

- KNN classification/regression;
- decision-tree prediction;
- linear/RBF SVM;
- Gaussian Naive Bayes;
- K-means;
- standard scaling;
- min/max scaling;
- K-fold partitioning;
- autoencoder forward path;
- embedding lookup;
- PCA;
- OLS.

Training versus prediction/transform capability is explicitly separated where only one side has native lowering.

## 21.5 Automatic differentiation

`autodiff.py` implements:

- content-addressed/hash-consed expression DAG;
- forward evaluation;
- reverse-mode differentiation;
- symbolic gradient graph;
- numerical gradient comparison;
- Hessians;
- second-difference Hessian checks;
- closed differentiable primitive registry;
- lowering-closure checks.

`autodiff_program.py` adds:

- admission;
- quarantine reasons;
- scalar mutation functionalization;
- bounded loop tracing;
- defunctionalization;
- rematerialization plans;
- AD-order comparisons.

Programs outside the admitted subset are refused with structured reasons.

## 21.6 Losses and optimizers

Losses include:

- MSE;
- softmax cross-entropy;
- binary cross-entropy with logits;
- hinge loss.

Optimizers include:

- SGD;
- momentum;
- RMSProp;
- Adam.

There are both reference implementations and portable C update emitters.

## 21.7 Training

`training.py` provides a bounded supervised loop with:

- minibatching;
- train/validation split;
- early stopping;
- classification and regression metrics;
- linear and MLP reference models.

`train_graph.py` turns a training step into explicit BCIR claims:

```text
forward
 -> activation
 -> loss
 -> reduce
 -> backward
 -> update
```

It supports:

- planned runs;
- StreamPack hydration;
- scheduling;
- streaming;
- certificates.

`lower/autodiff_kernel.py` emits forward/backward/SGD C. `runtime/c/bcir_train` supplies selected native stage kernels.

---

# 22. Open-weight model stack

Primary sources:

```text
bcir/frontends/models/
bcir/hosted/models/
runtime/c/bcir_q8_model*
runtime/c/bcir_decode*
runtime/c/bcir_llama*
```

## 22.1 Manifest-first ingestion

`manifest.py` builds a model manifest before loading weights.

It records:

- architecture;
- license;
- tokenizer reference;
- shard inventory;
- shard hashes;
- dtype histogram;
- parameter count;
- context length.

Safetensors headers are read without reading payload bytes.

## 22.2 Header-only tensor inventory

`inventory.py` records:

- tensor names;
- dtypes;
- shapes;
- physical offsets/spans;
- shard layout;
- a header/layout digest.

The header digest is planning identity, not a substitute for content hashes.

## 22.3 Tokenizers

Dependency-free references exist for:

- GPT-2/Qwen-style byte-level BPE;
- special tokens;
- lossless byte/unicode mapping;
- chat rendering;
- SentencePiece protobuf reading;
- score-based SentencePiece BPE.

## 22.4 Reference decoder

A bounded Llama/Gemma-style dense decoder includes:

- embeddings;
- pre-norm layers;
- Q/K/V projections;
- RoPE;
- causal attention;
- grouped-query attention;
- KV cache;
- feed-forward/SwiGLU-style path;
- RMSNorm;
- logits;
- greedy decode.

## 22.5 Real-weight ingestion

The HF ingestion layer converts:

- `config.json`;
- Safetensors tensor layouts;
- `[out,in]` linear orientation;
- head geometry;
- RoPE interleaving;
- model tensor naming;

into the reference decoder’s weight structure.

## 22.6 Quantized artifacts

Capabilities include:

- per-group Q8 model quantization;
- drift records;
- deterministic BCIRQ8 persistence;
- standalone C loader;
- native decode kernels;
- standalone greedy inference.

## 22.7 Serving

`serve.py` models generation as a proof-carrying planned artifact:

- prefill and decode phases;
- session module;
- generation certificate;
- token DFA;
- stream events;
- planned `generate()`.

## 22.8 Paged KV and batching

`paged_kv.py` provides:

- page-table-backed KV resources;
- per-page generations;
- eviction rules;
- live-session protections;
- session admission;
- continuous-batching claim graphs;
- batch certificates.

## 22.9 Model capacity and placement assessment

`bcir-model-assess` can operate without opening weights.

Inputs:

- tensor inventory;
- hardware envelope;
- workload.

Outputs:

- format-size estimates;
- training-state sizes;
- bank peaks;
- placement candidates;
- prediction intervals;
- cost report;
- selected execution plan;
- verified StreamPack.

Optional bounded native or Python-oracle microbench records can be attached.

---

# 23. Hosted model and training laboratory

The hosted layer is opt-in and allows PyTorch and numerical dependencies without contaminating the dependency-free core import path.

## 23.1 Hosted Llama model

Capabilities:

- independent PyTorch Llama reference;
- RMSNorm;
- attention;
- MLP;
- decoder layers;
- backbone and language model;
- deterministic AdamW training;
- generation-based checkpoints;
- pickle-free checkpoint handling;
- deterministic export back through BCIR’s strict ingestion boundary.

## 23.2 Training stages

Bounded hosted stages include:

- supervised fine-tuning;
- embedding distillation;
- reward-model training;
- DPO;
- PPO;
- GAE;
- reasoning SFT;
- small supervised MLP/GRU/transformer models.

## 23.3 Corpus pipeline

The data pipeline provides:

- raw and prepared document identities;
- deterministic preparation;
- provenance-preserving output;
- prepared-corpus manifests;
- byte-fallback BPE training for small fixtures.

## 23.4 Adaptive transformer research

The dependency-free semantic rail and hosted PyTorch rail cover bounded versions of:

- tied-depth/looped models;
- variable-width schedules;
- reference-plus-sliding attention;
- exogenous anchors;
- coarse-to-fine multi-patch processing.

These are research contracts, not production kernels or latency evidence.

## 23.5 Byte-native research

Implemented bounded research surfaces include:

- raw-byte vocabulary and lossless references;
- Byte Latent Transformer shapes;
- patch policies;
- block denoising;
- speculative verification;
- diffusion draft/verify;
- MambaByte selective SSM;
- byteified transplant plans;
- measured-ingest choice;
- tiny hosted training.

Again, these are bounded semantic and hosted reference implementations.

## 23.6 Sequence-interface research

Capabilities include:

- cross-tokenizer alignment;
- projected log probabilities;
- tokenizer expansion;
- continued-BPE initialization;
- finite scalar quantization;
- causal-series codec;
- progressive growth stages;
- hosted row freezing and staged training.

## 23.7 Provider contracts

The hosted training provider layer separates:

- teacher inference/embedding providers;
- remote compute providers.

A teacher may produce immutable targets; it is not treated as a remote gradient source. A remote compute provider executes BCIR-owned code and returns BCIR-owned artifacts.

## 23.8 Hardware policy learning

A bounded PyTorch GNN/Transformer can learn priors for hardware-plan search. Promotion still requires:

- deterministic candidate legality;
- static-memory safety;
- StreamPack validity;
- measured-versus-simulated provenance;
- promotion certificate.

---

# 24. Performance and TMSAO audit

`bcir/performance_audit.py` defines a bounded cross-organ audit.

It exercises representative:

- graph;
- optimizer;
- scheduler;
- StreamPack;
- memory;
- telemetry;
- quantization;
- ML;
- hardware-search;

operations and records deterministic correctness plus host-relative timing.

It does **not** claim to prove theoretical maximum system performance. Shared-host timings are informational unless run on a controlled performance rig.

`bcir-tmsao-audit` exposes this rail.

The latest research proposal gives TMSAO a broader formal interpretation involving typed regions, objective registries, lower bounds, and whole-system certificates. That broader formulation is not yet the current executable optimizer.

---

# 25. Security architecture

Primary sources:

- `SECURITY.md`
- `docs/research/BCIR_SECURITY_THREAT_MODEL.md`
- `docs/research/BCIR_SECURITY_RED_TEAM_AUDIT_2026-07-15.md`

## 25.1 Security objective

Hostile source, model, manifest, telemetry, and wire input should either:

- become one bounded, unambiguous, verified artifact; or
- fail before externally visible mutation.

## 25.2 Core invariants

1. Parse once and interpret once.
2. Reject duplicate or ambiguous identity.
3. Bound bytes, recursion, geometry and work before allocation or mutation.
4. Preflight before commit.
5. Make ownership explicit.
6. Check generations and content identity at consumption.
7. Serialize temporal transitions.
8. Keep legality independent of telemetry and learning.
9. Do not carry process pointers or mutable Python objects across future privilege boundaries.

## 25.3 Input protections

Current implementations use combinations of:

- strict duplicate-key JSON parsing;
- non-finite-number rejection;
- exact schemas;
- checked add/multiply/alignment;
- span and overlap checks;
- UTF-8 validation;
- count/depth/size ceilings;
- CRC and digest checks;
- generation checks;
- no-follow/stable-file identity checks;
- private temporary directories;
- subprocess timeouts;
- compiler/test worker limits.

## 25.4 Ownership

- Freestanding C is generally heap-free.
- Hosted C uses explicit allocator injection and cleanup contracts.
- Borrowed views document required backing lifetimes.
- C++ wrappers preserve borrowed immutable payload semantics.
- destroy/reset operations are intended to be idempotent.

## 25.5 Current privilege boundary

The repository contains no current:

- resident BCIR kernel module;
- BCIR device node;
- privileged BCIR service;
- cross-process BCIR IPC transport;
- DMA/IOMMU implementation;
- network listener;
- setuid transition.

RuntimeChannel is a same-process vtable.

Consequently, BCIR is not a sandbox: Python plugins, resident compilers, and native libraries execute with the caller’s authority.

## 25.6 Fuzzing and sanitizers

The C trust-boundary suite covers formats including:

- StreamPack;
- binary records;
- telemetry frames;
- BCIRQ8;
- BCAB;
- BER/DER;
- DER-to-StreamPack;
- PER;
- OER;
- JER;
- plan-driven encoding.

CI includes ASan, UBSan, LSan, static analysis, and libFuzzer campaigns. A scheduled deep sweep adds Valgrind, cppcheck, scan-build, larger frontend campaigns, and longer fuzzing.

---

# 26. CLI and operator surfaces

Installed Python console scripts:

```text
bcir
bcir-pack
bcir-bundle
bcir-registry
bcir-model-assess
bcir-asn1c
bcir-ham-plan
bcir-tmsao-audit
```

Additional module driver:

```text
python -m bcir.frontends.cfront
```

## 26.1 Main `bcir` CLI

Capabilities include:

- program selection;
- target selection;
- Theta and policy;
- LLVM emission;
- MLIR emission;
- C emission;
- AOT execution;
- JIT execution;
- WASM execution;
- wave schedule;
- EFT schedule;
- async token schedule;
- RCSP budgets;
- overlap pricing;
- calibration table loading;
- regret report;
- MoE research;
- proposal acceleration;
- provenance manifest;
- e-graph analysis;
- soft-temperature plan distribution;
- target code generation;
- host calibration;
- silicon probing;
- benchmarking;
- joint bundle optimization;
- explain records;
- replay;
- witness reduction.

## 26.2 Registry operator

`bcir-registry` provides governed in-process inspection and mutation. A write increments `data_gen`, thereby making already hydrated StreamPacks stale under R11.

## 26.3 ASN.1 compiler driver

`bcir-asn1c` supports:

- list;
- print;
- check;
- DER encode/decode;
- BER-tolerant decode;
- JER encode/decode;
- BASIC versus BCIR-canonical JER;
- framed JER;
- DER/JER transcode;
- hex/raw I/O;
- source-position diagnostics.

---

# 27. Testing and CI evidence

## 27.1 Local thorough Python run

The fresh checkout discovered **3,280 tests**.

Initial thorough result:

```text
3278 passed, 2 failed
```

The two failures were:

1. WASM execution under an obsolete Node.js runtime;
2. the test-runner default-timeout self-test while the validation process had explicitly overridden that timeout.

After:

- upgrading Node to `22.23.2`;
- removing the timeout override for the self-test;

the targeted rerun reported:

```text
FAILED_TESTS_RERUN_PASS 2/2
```

The exact evidence is therefore:

> One complete aggregate run produced 3,278 passes and two environment-induced failures, and both failures passed after correcting the environment.

There was not a second single-command aggregate `3280/3280` run.

## 27.2 ASN.1 result

Neither aggregate failure was an ASN.1 test, so no registered focused ASN.1 failure was reported in the complete run.

## 27.3 Toolchain

The coherent Linux environment has access to:

```text
gcc
g++
cmake
ninja
clang
clang++
lli
llvm-link
llc
llvm-as
opt
mlir-opt
mlir-tblgen
FileCheck
wasm-ld
node
mono
ilasm
valgrind
ccache
jq
llvm-bolt
```

LLVM/Clang/MLIR:

```text
22.1.8
```

The coherent prefix is:

```text
/usr/lib/llvm-22/bin
```

with:

```text
LLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm
MLIR_DIR=/usr/lib/llvm-22/lib/cmake/mlir
```

## 27.4 GitHub Actions

The exact commit’s CI run concluded successfully across 13 matrix-expanded jobs, including:

- two Python oracle shards;
- C runtime;
- C analysis;
- Ubuntu and Windows host portability;
- Ubuntu and Windows hosted model gates;
- LLVM training;
- native AArch64 oracle;
- native AArch64 C runtime;
- LLVM/MLIR 22 rail;
- documentation governance.

## 27.5 Local validation boundary

Not freshly completed as standalone local aggregate gates in this checkout:

- full MLIR-22 build/pass aggregate;
- standalone full sanitizer sweep;
- standalone Valgrind deep sweep;
- standalone long libFuzzer campaign.

The exact commit’s CI is evidence for those remote gates but is kept distinct from local execution.

---

# 28. LLVM training and education

`llvm-training/` contains twenty numbered topic areas:

```text
00 foundations
01 syntax
02 types
03 constants
04 memory
05 control flow
06 metadata
07 optimization
08 pitfalls
09 vectorization
10 grammar
11 concurrency
12 backend/JIT
13 advanced IR
14 MLIR bridge
15 binary analysis
16 exception handling
17 new pass manager
18 MLIR lowering to LLVM
19 hardware-aware work
```

Additional infrastructure includes:

- exercise manifests;
- prompt templates;
- reference solutions;
- invalid examples;
- adversarial fixtures;
- deterministic variants;
- an autograder;
- deterministic dataset export;
- binary-analysis evidence;
- CSV schema checks;
- opaque-pointer validation;
- opt-diff goldens;
- lit tests;
- MLIR examples;
- BCIR mapping exercises;
- `llc` smoke;
- none/thin/full LTO matrix;
- optional BOLT smoke;
- CMake aggregate quality targets.

This is both curriculum and compiler-toolchain regression corpus, but it does not define the semantics of the BCIR production dialect.

---

# 29. Documentation and governance

The repository has several documentation classes:

## Normative/current

- `docs/BCIR_LANGREF.md`
- `docs/STATUS.md`
- `docs/PARITY.md`
- ABI documents under `docs/kernel/`
- active language guides

## Generated governance

CI checks:

- `STATUS.md` regeneration;
- relative links;
- references to retired paths;
- summary claims against source;
- hot/cold import quarantine.

## Roadmaps

Roadmaps may mix:

- historical plans;
- delivered phase notes;
- current boundaries;
- future gates.

Their “built” claims should be cross-checked against source and tests.

## Research

Research documents include:

- performance studies;
- threat models;
- architecture proposals;
- comparison studies;
- GEM+/TMSAO design.

They are evidence and design input, not automatically product law.

---

# 30. Latest GEM+/TMSAO architecture proposal

The latest commit proposes an integrated future architecture.

## 30.1 Typed regions

Rather than optimizing isolated claims with partly separate organs, GEM+ would place claims into typed regions such as:

- compute;
- memory;
- encoding;
- scheduling;
- placement;
- communication.

## 30.2 Candidate/equivalence graph

It proposes a versioned graph carrying:

- semantic alternatives;
- candidate implementations;
- transformations;
- equivalence relationships;
- provenance;
- lower bounds.

## 30.3 Objective registry

Proposed algebraic objectives include:

- min-plus;
- max-plus;
- min-max;
- Boolean;
- Pareto;
- robust objectives.

## 30.4 One canonical execution plan

The target is one content-addressed plan binding:

- candidate choice;
- schedule;
- channel;
- placement;
- memory;
- transfer syntax;
- evidence;
- certificate.

## 30.5 Solver portfolio

The proposal calls for a solver portfolio with:

- exact methods where bounded;
- admissible heuristics;
- lower bounds;
- Pareto pruning;
- decomposition;
- incremental recomputation;
- explicit certificate levels.

## 30.6 Scaling and memory

Proposed work includes:

- incremental planning;
- canonical liveness schedule;
- tiered static allocation;
- dense-loop optimization;
- integrated HAM/Semantic Swap;
- provider boundaries.

## 30.7 ASN.1 role

ASN.1 encoding choice would become a typed GEM+ region rather than a separate selection subsystem. JER would remain a source/control/load-plane format, not a hot-path replacement for StreamPack.

## 30.8 C++ and system services

The proposal also sketches:

- Python-to-C++ migration;
- API/database/service decomposition;
- IPC;
- native measurement rigs;
- driver and kernel profiles;
- FPGA/SASOS implications.

All of this is currently proposal-level unless an individual component already exists elsewhere, such as HAM, static memory, ASN.1 selection, or the single-node C++ handoff.

---

# 31. Capability and maturity matrix

Legend:

- **Normative** — production law or frozen ABI.
- **Executable oracle** — real reference implementation.
- **Native** — real C/C++ implementation.
- **Tool-dependent** — executable when the resident compiler/runtime exists.
- **Modeled** — executable planning model without corresponding device execution.
- **Hosted research** — bounded experimental implementation.
- **Proposal** — documentation/design only.

| Subsystem | Status |
|---|---|
| Resource/claim/phase model | Executable oracle; represented in MLIR; native C subset |
| R1–R25 verification | Normative MLIR; broad Python twin; narrower C twin |
| K_BCIR min-plus planning | Executable oracle and MLIR pass family |
| RCSP/Pareto planning | Executable oracle and MLIR representation/passes |
| Composition/e-graph | Executable oracle |
| Provenance/replay | Executable oracle, MLIR, selected C twin |
| GEM scheduling | Executable oracle |
| StreamPack v1–v3 | Frozen ABI; Python and freestanding C |
| BCAB v1 | Frozen ABI; Python, C and C++ |
| MAP/ROP | Executable frontends |
| ETL/transducers | Executable oracle; binary C twin |
| C frontend | Executable Python and native C supported subsets |
| Portable C lowering | Native/tool-dependent and exercised |
| Textual LLVM lowering | Executable bounded subset |
| AOT/JIT/WASM | Tool-dependent and tested |
| JVM/CIL | Tool-dependent bounded subsets |
| Cross-target `llc` | Tool-dependent |
| ASN.1 schema/encodings | Broad executable oracle |
| ASN.1 R24/R25 | Normative MLIR law rail |
| ASN.1 native codecs | Native bounded format-specific twins |
| ASN.1 certified selection | Executable oracle plus native measurements |
| Static memory | Executable verified planner |
| HAM | Executable metadata planner; no physical transfer engine |
| Device manifest | Executable static schema |
| Telemetry | Executable Python; C frame/ring twins |
| Silicon probes | Real where host exposes signals; honest unavailable state otherwise |
| CPU channels | Executable |
| SYCL | Tool-dependent hosted execution |
| GPU/PTX | Real artifact generation; no local physical execution evidence |
| FPGA/PIM/NVMe | Modeled channels |
| Quantization/math/tensors | Executable oracle; many C paths |
| Autodiff/training | Executable oracle; bounded C lowering |
| Open-weight ingestion | Executable |
| Standalone BCIRQ8 inference | Native |
| Hosted PyTorch model lab | Hosted research and CI-gated |
| Byte/adaptive/sequence research | Hosted research |
| C++ single-node orchestration | Native and real |
| C++ dynamic/distributed orchestration | Explicit stubs |
| Resident kernel/driver/IPC | Not implemented |
| GEM+ unified planner | Proposal |
| Whole-system optimality certificate | Proposal |

---

# 32. What was accomplished in this reconstruction

I re-established the report from the repository and produced a new mechanical evidence set under the ignored validation directory:

```text
build/validation/system-report-repo-inventory.json
build/validation/system-report-pygount.txt
build/validation/system-report-deep-inventory.json
build/validation/system-report-python-api.json
build/validation/system-report-cli-help.txt
build/validation/system-report-vector-add-pipeline.txt
build/validation/system-report-vector-add.mlir
build/validation/system-report-vector-add.c
build/validation/system-report-embeddable-api.txt
```

I also:

- refreshed and froze Git provenance;
- confirmed the checkout is clean;
- inventoried all tracked top-level and major subsystem trees;
- enumerated current ODS operation families;
- enumerated all 37 MLIR passes;
- inventoried Python public structures and module descriptions;
- exercised all primary CLI help surfaces;
- traced the end-to-end architecture;
- generated a real vector-add plan, MLIR, and C output;
- compiled and self-checked the embeddable API’s native artifact;
- re-audited the complete ASN.1 architecture;
- separated production, native, modeled, hosted-research, educational, and proposal surfaces.

No product source file was modified.

---

# 33. Overall characterization

At the current commit, BCIR is best understood as six connected systems:

1. **A registry-first semantic IR and R1–R25 legality system.**
2. **A multidimensional realization optimizer with budgets, provenance, replay, and bounded learned guidance.**
3. **A GEM scheduling and frozen StreamPack execution architecture.**
4. **A multi-backend compiler/runtime stack spanning Python, MLIR/C++, portable C, LLVM, WASM, and bounded stack formats.**
5. **A broad ASN.1 compiler/selection ecosystem covering X.680–X.697 subsets, R24/R25 law, native twins, and certified encoding choice.**
6. **A large ML/model and heterogeneous-hardware research-to-runtime platform, with clear distinctions between native CPU execution, hosted experiments, modeled channels, and unimplemented physical drivers.**

The implementation is far beyond a documentation-only IR. It contains real compilers, optimizers, codecs, native runtimes, artifacts, model loaders, inference kernels, training references, telemetry, and extensive differential validation.

At the same time, the repository usually records the crucial maturity boundary directly:

- Python breadth is not automatically native breadth.
- Code generation is not device execution.
- A modeled hardware channel is not hardware evidence.
- A hosted research oracle is not a production kernel.
- A research proposal is not a landed architecture.
- A green remote CI result is not a fresh local hardware result.
- A content digest is not signer authority.
- JER is JSON text, not a binary hot-path format.
- The existing collection of optimizers is not yet the unified GEM+ whole-system solver proposed by the latest commit.

This completes the comprehensive capability and architecture reconstruction. The next phase can now focus specifically on **current shortcomings, architectural inconsistencies, missing native coverage, and a prioritized development program**, without having to rediscover the system first.
