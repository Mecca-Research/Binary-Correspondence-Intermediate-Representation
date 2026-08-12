# Advanced compiler techniques: what BCIR needs, what it has, and what it must not build

A triage of nineteen advanced compilation strategies against BCIR as it stands at `610dd01`
and against the [GEM+ roadmap](BCIR_GEMPLUS_ROADMAP.md). Every "already have it" below is
backed by a file in this tree, because a roadmap that claims coverage it does not have is
worse than one that admits a gap.

The techniques sort into **four** categories, not two, and the third is the one that changes
what the roadmap should say.

| | Category | Count |
|---|---|---|
| **A** | Already built | 4 |
| **B** | Already in the roadmap | 5 |
| **C** | **Belongs to LLVM — BCIR's job is to *supply the facts*, not the pass** | 6 |
| **D** | Genuinely new, and worth adding | 4 |

---

## The framing that decides most of the list

BCIR is not a code generator and does not own a backend. The performance report's §4.4 measured
what actually produces wins:

| Native comparison | Ratio |
|---|---:|
| Dense streaming BCIR vs equivalent compiler loop | **0.98–1.01×** |
| Gather avoidance | **5.58–6.05×** |
| Blocked reduction | **11.68–11.72×** |
| Direct strided access vs gather | 1.27–1.33× |

Parity on the dense loop is the *expected* result — when LLVM sees the same affine loop and the
same alias facts, matching it is the floor. The 5.6× and 11.7× wins come from BCIR preserving
enough structure to prove a **better data path** before LLVM ever sees the code.

That is the whole division of labour, and it settles category **C**: every technique on the list
that operates *below* the point where BCIR hands off — register allocation, post-RA scheduling,
cache-line padding, store-to-load forwarding, instruction-level modulo scheduling — is work LLVM
already does well, has done for twenty years, and does with information BCIR does not have
(register pressure after allocation, the exact scheduling model of the subtarget). Reimplementing
any of them means owning their correctness for a worse result.

**But category C is not "ignore it".** For each of those six, BCIR holds a *declared* fact that
LLVM must otherwise *infer*, and inference is exactly where a backend gives up and takes the
conservative path. BCIR's contribution is to hand LLVM the fact. That reframing turns six
"don't build" entries into one concrete roadmap slice, and it is where the largest unclaimed win
in this document sits.

---

## A. Already built

### A1. Linear allocation arenas — `bcir/kbcir/static_memory.py`

`plan_static_memory` computes lifetimes, assigns byte offsets inside banks, and emits a
`StaticMemoryPlan` with a module digest and a hardware digest. That *is* an arena: one linear
extent, offsets assigned at compile time, no allocator at runtime. It is deterministic and
verified.

It is not *optimal* — first-fit was suboptimal on 38.6% of 500 exact-solver fixtures, worst case
1.615× — and that gap is roadmap **G5**, which already exists with those numbers as its gate.

### A2. Vectorized lexing — `runtime/cpp/bcir_jer_simd.cpp`

The JER structural index, with scalar/SSE2/AVX2/NEON tiers, exact status/offset/work-budget
parity against the scalar rail, and shared token scanners. Landed as J5, deliberately
non-default because its *performance* admission is narrower than its *correctness* admission.

The one thing to preserve: the scalar bounded C rail stays the correctness authority. A
structural index accelerates whitespace dispatch; it is not a licence to skip validation. That
boundary is already written into `bcir_jer.h`.

### A3. Progressive dialect lowering — the MLIR rail

`bcir-optimize` → `bcir-hydrate` → `bcir-lower-llvm` → `bcir-aot`, with `bcir.*`, `bcir.asn1.*`
and `bcir.ecn.*` dialects and R1–R25 verified at the top. This is textbook progressive lowering,
and it is what the law rail *is*.

### A4. Quiescent generation switch — `bcir/asn1/staged.py`

The OSR-shaped half of tiered execution: `enter`/`leave` quiescence, generation-tagged `install`,
signature verification, and `rollback`. What it is *not* is on-stack replacement — see D4.

---

## B. Already in the GEM+ roadmap

| Technique | Slice | Gate already written |
|---|---|---|
| **Pluto-like cache tiling** | G6 (affine/polyhedral region) | integer sets/maps, dependence polyhedra; native rows must not regress |
| **Schedule trees** | G6 + G1 | one canonical schedule artifact is G1's `pricing.eft.divergence` → 1.0 |
| **Equality saturation / e-graphs** | G6 (equivalence-graph region) | e-classes plus side conditions, bounded extraction |
| **ILP-based extraction** | G4 (solver portfolio) | CP-SAT/MILP for small high-value regions, with a reported gap |
| **AutoFDO / hardware PGO** | G7 + P3 | PMU counters, raw samples, generation-tagged frozen tables |

Two notes where the roadmap is already more careful than the technique list:

**Equality saturation needs bounded extraction, and the roadmap says so.** An e-graph grows
without limit and the extraction problem is NP-hard in general. G6 requires every region to
supply a *refusal condition*, which for an e-graph means a node/time budget and a deterministic
fallback — otherwise saturation becomes a way to make compilation unbounded rather than optimal.

**Souper-style rule synthesis is deliberately not adopted.** Auto-synthesized rewrite rules are
proved correct by an SMT solver *against the synthesizer's own semantics*. BCIR's semantics are
R1–R25 plus the three-rail agreement, so a synthesized rule would need proving against those, and
a rule that is correct in Z3's model but violates R5's hazard contract is exactly the class of
defect the security audit spent two passes removing. Candidate *generation* by search is fine and
is in G4; candidate *legality* stays with the laws.

---

## C. LLVM's job — BCIR's job is to supply the fact

This is the reframing. For each, BCIR already holds a declared fact that LLVM must otherwise
infer conservatively.

### C1. Memory disambiguation / advanced alias analysis — **BCIR has better than AA, and throws it away**

An alias analysis *infers* what may alias. BCIR does not have to infer: every claim declares its
read and write RIDs, its `hazard` (`unique` | `atomic` | `barriered`), its `bounds`
(`strict` | `masked` | `assumed_safe`), and whether it is `volatile`. That is **declared
aliasing**, which is strictly stronger than any inferred result.

And today the LLVM lowering does this (`bcir/lower/llvm.py`):

```llvm
define void @bcir_kernel(ptr noalias %A, ptr noalias %B, ptr noalias %C, i64 %n)
```

`noalias` on **every** pointer, unconditionally — including graphs where a claim legally reads
and writes the same RID in place. The performance report flagged it, and it is worse than
leaving the facts out: it is a *false* fact. LLVM will reorder on the strength of it.

**This is the single largest unclaimed win in this document**, and it is small: emit `noalias`
only where the RIDs are provably disjoint, and emit TBAA metadata / `!alias.scope` /
`!noalias` scopes derived from the RID partition where they are not. LLVM's OoO scheduling,
load/store reordering and vectorizer all improve on real alias facts, and BCIR is currently
supplying a blanket assertion instead.

### C2. Store-to-load forwarding

The hardware fast path fires when a store and a subsequent load match in address and size. BCIR
knows both — the claim carries `count`, `offset`, `stride_k` and the resource's element size. It
should keep matched-size access patterns matched when it picks a realization, and stop there.
Emitting the store-to-load-forwarding-friendly *instruction sequence* is the backend's job.

### C3. Vector selective masking

Already half-declared: `bounds="masked"` exists and the C front end assigns it for a known-extent
array or a recovered `malloc` count. That is the *fact* — "this access is bounds-checked by a
mask, not a branch". Lowering it to AVX-512 `%k` registers is LLVM's; BCIR's job is to carry the
fact to the point where LLVM can use it, and to price the masked realization honestly in the cost
model.

### C4. Chaitin-Briggs graph colouring vs greedy allocation

**Do not build.** BCIR emits LLVM IR with SSA values and lets LLVM's greedy allocator run. A
Chaitin-Briggs implementation inside BCIR would be a second register allocator competing with a
mature one, on a target BCIR does not model at that resolution. The *interesting* related
question — how many live values a plan forces — belongs in G4's lower-bound stack as a **register
pressure bound**, which is a cost input, not an allocator.

### C5. Post-RA scheduling, and C6. cache-line alignment / code padding

**Do not build.** Both need post-allocation register state and the subtarget's exact issue model.
BCIR has neither and should not acquire them. What BCIR *can* contribute is loop-header
identification and hot/cold structure via G7's profile data, which is what LLVM's alignment
heuristics consume anyway.

### Software pipelining (modulo scheduling)

Split. Instruction-level modulo scheduling is LLVM's (`-enable-pipeliner`). But the *decision* to
overlap iterations — how many stages, how much buffering, whether the dependence distance permits
it — is a **schedule-tree** question and lands in G6's SDF/CSDF region, where repetition vectors
and bounded FIFOs are exactly the right model. BCIR chooses the pipeline depth; LLVM emits it.

---

## D. Genuinely new — the roadmap additions

Four items are not covered anywhere, and each earns its place.

### D1. Export declared alias facts to LLVM — **new slice G9**

The C1 fix, promoted to a slice because it has a measurable gate and the largest expected effect
per line changed.

- Emit `noalias` only on provably disjoint RIDs. Correctness: today's blanket `noalias` is a
  false assertion on any in-place graph.
- Derive `!alias.scope`/`!noalias` scope metadata from the RID partition.
- Emit TBAA from the resource's declared element type.
- Carry `volatile` through as `volatile` rather than fencing it only in BCIR's own optimizer.

**Gate:** the `native.*` rows must not regress, and a new `exact` row — *no `noalias` is emitted
on a pointer pair sharing a RID* — gates it. That row is a correctness check, and it is the
reason this is a slice rather than a tweak.

### D2. Block-local vs global SSA — **a modelling question, answered "neither, and here is why"**

BCIR's claim graph is not SSA and should not become SSA. A claim is an *effect on named
resources*, not a value definition; `rd`/`wr` are RIDs, not SSA edges. The C front end already
maps SSA values to scalar `Resource`s at the boundary, which is the correct place for the
translation.

What the SSA question is *really* asking is whether BCIR's dependence information is
block-local or whole-module — and there the answer is a real gap: **phase-based liveness is
coarser than the token schedule permits**. That is already G5, and it is the same finding from a
different direction. Recording the answer here stops the question being re-asked.

### D3. Escape analysis and interprocedural context — **new slice G10**

ThinLTO and CHA-based devirtualization are *not* directly applicable: a BCIR `Module` is
whole-program by construction, so there is no link step to be thin about, and there are no
vtables. But two of the underlying capabilities are missing and are worth having:

- **Escape analysis.** The C front end already records `bounds_provenance`
  (`declared_extent` | `recovered_count` | `snapshot_extent` | …). A resource whose provenance is
  `declared_extent` and whose RID never crosses a call boundary provably does not escape — which
  licenses stack placement (an `alloca`, or a bank-local arena slot) instead of a heap resource.
  This composes directly with A1's arena.
- **Indirect-call resolution.** `callee_sig` already carries a declared callee *type* on
  `c.call.indirect`. Narrowing it to a single admitted target — CHA's job, done over BCIR's own
  declared call graph rather than a class hierarchy — turns an opaque effect edge into a known
  one, which unblocks fusion and reordering across it.

**Gate:** `exact` rows — the count of resources proved non-escaping on the cfront corpus, and the
count of indirect edges resolved to one target. Both are deterministic and gate on any host.

### D4. On-stack replacement — **explicitly out of scope, and the reason matters**

BCIR has the *safe* half already (A4: quiescent generation switch with rollback). True OSR —
hot-swapping a frame mid-iteration — requires mapping live state between two compiled forms while
the stack is live, which means an unverified transition in the middle of a plan. That is
incompatible with the certificate model: the executing artifact would stop being the attested one
partway through.

The roadmap's answer stays *quiescent* replacement: a new generation activates at a boundary
where no plan is in flight. Recorded here so the decision is a decision rather than an omission.

---

## Summary: what changes in the roadmap

Two new slices and one recorded decision:

| | Addition | Gate |
|---|---|---|
| **G9** | Export declared alias facts to LLVM (`noalias`, alias scopes, TBAA, volatile) | `exact`: no `noalias` on a shared RID; `native.*` must not regress |
| **G10** | Escape analysis + indirect-call target narrowing | `exact`: resources proved non-escaping; indirect edges resolved |
| — | OSR out of scope; quiescent generation switch is the answer | — |

G9 goes early — it is small, it fixes a false assertion the compiler currently emits, and every
later slice's `native.*` measurements are taken through it. G10 follows G6, since escape analysis
over typed regions is stronger than over the opaque claim DAG.

Everything else on the list is already built (A), already scheduled (B), or deliberately left to
LLVM with BCIR supplying the facts (C).
