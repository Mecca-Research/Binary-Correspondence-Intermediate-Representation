# BCIR — ML / AI Integration Roadmap

> **Companion to [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md).** That document is the authoritative
> phase ladder (Phase C → M → D → F → L), the capability tracks (CT1–CT5), the release ladder, and the
> two-truth placement law. This document **expands its Phase M (selective ML ops) and Phase L (the ML library
> / ecosystem)** into a single, dependency-ordered ML/AI program, and slots every proposed integration onto
> the existing substrate. It does **not** restate the spine, the organs, or the quarantine — it references
> them ([`BCIR_LANGREF.md`](BCIR_LANGREF.md) §13 for L0–L3, [`HETEROGENEOUS_CHANNELS.md`](HETEROGENEOUS_CHANNELS.md)
> for the channel model). Everything here is held to the same discipline that built the rest of BCIR:
> **prototype in the Python oracle → port to the MLIR/C law, parity-gated; learned/float stays Python and
> freezes to Q8; nothing graded ever touches the deterministic hot path.**

---

## 0. Stance — why an IR becomes intelligence

The textbook definition of an IR — *"a data structure used internally by compilers to bridge high-level source
and low-level machine code; modular, portable, optimizable"* — is operationally true and **purposeless**. It
describes features, not a reason to exist. BCIR's reason to exist is the thesis below.

**Reference State is the defining characteristic of a useful machine intelligence.** A model that only maps
inputs to outputs has no *body* — no standing relationship to the physical substrate it runs on. BCIR gives
the intelligence a body: its computational organs (the memory hierarchy CT1, the wave scheduler CT2, the
calibrated cost tables CT4, the heterogeneous channel tower) **are** the reference state. The IR is the mind's
coupling to that body. Intelligence does not float in from a dataset; it **floods in from the reference state**
when the system asks, of every program:

- *What state transformation is intended?* → the **semantic claim graph** (BCIR-0).
- *What tensors / columns / buffers / sparse maps / records / layouts exist?* → the **shaped data graph** + the
  registry (BCIR-1/2).
- *Where can resources live; what domain constraints apply?* → the **placement candidate graph** + R3/R16.
- *Which realization path π is selected under H and Θ?* → **K_BCIR** (the tropical planner).
- *What lane schedule, StreamPack, fences, prefetch contracts execute?* → **GEM**.

The graph is an intelligence **scaffold**: start from the lowest level, work with its rules (the R-laws), and
structure emerges in accordance to Landauer's Principle. The intelligence is the **feedback loop of optimization on the computational
reference state** — and that loop is the foundation we claim for all future AI built this way.

**This is deliberately not neuromorphic.** Brain-mimicking event-driven chips activate only on changing
stimuli — a poor fit for processing static, massive datasets, and a poor return against CMOS. We do not want
AI that mimics a human; we want AI that does **what humans cannot**: hold and transform vast state, deterministic
and reproducible, an *augmentation* of human intelligence — an engineered, alien ally, built from math,
programming, and optimization in feedback loops of learning and evolution. The "life-like" trait we keep is
**Reference State**, not biology.

**The logic foundation is tropical.** Intelligence here is shortest-path / most-thermal-optimal / most-efficient
pathway selection over the legal candidate DAG:

> `G` — the goal graph (a BCIR program). `π` — a realization plan (lane/stride class, batching, schedule,
> prefetch). `M(π,Θ)` — the schedule-aware price ((min,+) series ⊗, max parallel, over the wave/token DAG).
> `R(π,Θ)` — the additive 12-d resource ledger (Σ Tᵢ ⊗ fᵢ, element-wise Q8 coupling). `B(H,Θ)` — live budgets
> (thermal/power caps, Θ-dependent). Solved exactly by RCSP label dominance; Pareto-optimal plans no weight
> vector can reach recovered by `kbcir.rcsp.pareto_plans`. **All arithmetic integer/Q8 and deterministic.**

**The non-negotiables (what keeps this engineering, not dreaming).** Every layer below obeys:
1. **The two-truth quarantine** (LangRef §13, `bcir/kbcir/twotruth.py`, `verify_quarantine` R13). *Classical
   truth `v`* — binary legality (R1–R21); there is no "0.7 legal." *Graded truth `(v,w)`* — a value with
   confidence `w∈[0,1]`, the learned/measured machinery (softdp posterior, bayescal interval, regret evidence)
   that answers *which legal plan is best*, never *whether a plan is legal*. Graded may **inform** but never
   **become** a verdict; the only sanctioned crossing is an explicit `decide()` at a frozen threshold.
2. **The L0–L3 placement law.** L0 (hot path) — learned inference **prohibited**, decisions compiled out. L1
   (plan time) — frozen Q8 tables only. L2 (checkpoints) — portfolio + replay gate. L3 (meta-policy) —
   measured, human-actuated, gated by `ΔL = Σ regretᵢ/bestᵢ − (k/2)·ln(N) > 0`.
3. **Provenance is the spine.** Every decision rule in force is frozen + generation-tagged; the manifest (R13)
   is the commit hash of a plan; the memory module (`a = Lim(Res(U))`, e-graph saturation) is a frozen,
   generation-tagged extraction.
4. **Prototype-then-port + the `--fallback` / Clang-equivalence gate** — no new op ships without an
   oracle↔law↔C parity test, and BCIR never silently changes a program's observable behaviour.

Everything in §2 is an addition **under** these four laws.

---

## 1. The intelligence already in BCIR (the substrate this builds on)

This roadmap adds net-new ML capability on top of a substrate that is already an intelligence engine. The
audit (do not rebuild any of this — extend it):

**The deterministic spine (the body's reflexes).** BCIR-0..5; the claim graph; K_BCIR (`-bcir-cost-model`,
`-bcir-plan`, `-bcir-rcsp`, the min-plus shortest path + Pareto); GEM (`bcir/gem/execute.py`, the phase-ordered
wave executor); the **R1–R21** verifier (dual-rail MLIR/C++/C + Python; R19/R20/R21 first-class since the
[`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §5.14 promotion). **Reusable as-is.**

**The learned organs already present (CT5, all Python, all freeze to Q8).** Each has a named growth axis —
this is what *"continuous development at every level"* means concretely:

| Organ (`bcir/kbcir/`) | Learns | Tier | Continuous-development axis |
|---|---|---|---|
| `microbench` | Q8 cost ratios (gather penalty, base overhead) | L1 | more access regimes; host thermal-noise models |
| `bayescal` | Gaussian posterior + conformal ±δ on cost | L1/L2 | ABC using `optimize` as the simulator |
| `egraph` / `memory` | liked/unliked-pair rewrites; saturated fixpoint `Lim(Res(U))` | L1/L2 | rule synthesis; saturation heuristics + budgets |
| `operad` | hierarchical labels `L`, content-addressed index `I` (FNV), trace, 2-cells | L1/L2 (gated) | **the 2-cell rewrite algebra (proposed, not yet built)** |
| `portfolio` | workload-class thresholds + gain schedules | L1/L2 | adaptive thresholds; class synthesis |
| `moegate` | a GNN router over the claim graph → expert selection | L2→L1 | attention heads; exploration temperature |
| `accel` | a logistic ranker for branch-and-bound candidate order | L2→L1 | feature engineering; ensemble/boosted rankers |
| `softdp` | finite-T plan posterior (free energy `−T log Σ exp(−score/T)`) | L2/L3 | annealing schedules; distillation targets |
| `regret` | hindsight regret + the MDL evidence margin `ΔL` | L3 | bandit (UCB/Thompson); cross-hardware transfer |
| `calibloop` / `throttle` | certified replan win; per-component amortization | L2/L3 | online recalibration triggers; tier-crossing budgets |
| `provenance` | manifest digest + version DAG (R13) | L2 | cross-generation plan tracing; causality |

The reasoning layers — **e-graph saturation** (the memory module, human-level "is this the same expression?"
reasoning made foundational to optimization) and the **operad** (labels/index/trace that make memory
navigable, traceable, queryable without touching the spine) — are the seeds the higher ML layers grow from.
The reference-state metrics — the **12-d cost vector**, the **thermal/power budgets**, and the recently added
**clock/timing** signal (R19/R20, `Timing` in `bcir/model/graph.py`) — are the axes the ML layers optimize over.

**The net-new boundary (what this roadmap actually builds).** Tensor ops, ML graphs, gradient machinery, and
the training loop are **absent today** (the frontend is scalar/elementwise; `lower/c_kernel.py` emits
elementwise C only). But the integration *seams* exist and are clean: the **libm/FFI edge** (`c.call.libm:` —
61 math names, malloc/free with extent tracking) is the path for BLAS/LAPACK/FFTW; the **K_BCIR search** already
prices `UNIT/STRIDE/GATHER/TILE` stride classes and a `gather_penalty`; the **R17 accuracy law** + compensated
Q8 reduction already model numerical precision; the **telemetry ring + calibration loop** already close
(measure → freeze → replan → certify) and need only an ML trainer fed in.

---

## 2. The ordered build-out

Six phases, dependency-ordered. The through-line: **a verifiable C inference substrate → ML primitives over
it → data/memory organs to feed them → more language sources → ML-guided hardware deployment → higher
cognition.** Each phase states what it *builds on*, its *L0–L3 / two-truth placement*, and its *parity gate*.
Phases A–B are the immediate, gateable work; E–F are horizons. ML ops (Phase B) run **throttled in parallel**
with the C/driver work and **never block the keystone** (master roadmap Phase M discipline).

### Phase A — The C inference substrate (the body's native instruction set)
*Extends master-roadmap Phase C.0. C is not "a driver language" here — it is the unrestricted, hardware-native
AI inference pathway with no framework tax.*

- **A1 — Max out C23 as the inference pathway.** *Done already* (build on): `#embed`, `_Atomic`/`<stdatomic.h>`,
  `_Alignof`, `typeof`/`_Generic`, VLAs, bitfields, `_Complex`, compound/designated initializers, statement
  expressions, `_Decimal*` (recognized). *Net-new, ML-load-bearing*: lower `_BitInt(N)` (exact-width quantized
  weights); `<stdbit.h>` + `<stdckdint.h>` (safe bit/overflow math for fixed-point activations); the
  optimization attributes **`[[unsequenced]]` / `[[reproducible]]` / `[[gnu::*]]`** elevated from lexed-only to
  **legality/cost signals** (an unsequenced kernel unlocks reorder/fusion the cost model can price); `assume`
  expressions feeding the verifier; native decimal floats as Q-fixed activation tables. Each is a Phase-C.0
  enabler with a fixture + Clang-equivalence gate.
- **A2 — Native matrix/tensor abstractions + safe-pointer polymorphism.** Build a first-class matrix/vector
  type for the C subset (over the existing multi-dim array + aggregate lowering) and use the **extent-provenance
  pointer model** (the `masked`/`assumed_safe` decision, §5.12) to give *polymorphism without errors*: a typed,
  bounds-recoverable pointer is the safe substrate for "C that reads like assembly, with automatic safety."
- **A3 — The test → quarantine → study → fix → optimize loop (the first practical CS layer).** Wire the
  natural debugging layers into ML + optimization: the runtime quarantine (`BCIR_CHK` / `bcir_bounds_quarantine`)
  + the three-way differential fuzzer (`tools/c/fuzz_cfront.py`) + the **regret ledger** become an *error-handling
  policy* — detect → quarantine → study (root-cause via the differential shrinker) → fix → re-optimize, with the
  policy chosen per computational environment. Lives L3 (observational) → frozen L1 policy.
- **A4 — Macro Assembly Programming for memory consolidation.** Extend the MAP frontend (CT3) + the allocator
  pool-plan (`-bcir-alloc-pool`, liveness arenas) with low-level ML so thousands of lines of hand-allocation
  collapse into engine-optimized, pool-planned claims.

### Phase B — The ML primitive layer (tensor ops as claims) — *this is Phase M, deepened*
*Builds on Phase A + the K_BCIR search + the operad/e-graph. Prototype each op in the oracle, port to an MLIR
law, parity-gate. Throttled, parallel to C/driver work.*

- **B1 — Tensor ops as BCIR claims.** Add `gem.matmul` / `gem.conv` / `gem.attention` ops carrying shape/rank/
  dtype metadata; let the **existing min-plus search decide the realization** (row- vs column-major, tile sizes,
  loop order) by encoding the choice as extra cost-vector dimensions — *reuse* `optimize()`, the stride classes,
  and `gather_penalty`. New laws: shape consistency, dtype compatibility (the next first-class R-laws after R21).
- **B2 — Activation specialization + fusion.** Activations fuse with the producing op via the **existing bundle/
  fusion/deforestation optimizer** — no new fusion engine, just new fusible ops.
- **B3 — Gradient machinery as graph transformations.** Forward and backward passes are **rewrites recorded as
  operad 2-cells** (this is the concrete use for the proposed 2-cell rewrite algebra in §1) — autodiff becomes a
  traceable, content-addressed transformation over the claim graph, reusing the e-graph + the SourceMap. Oracle
  prototype → MLIR law.
  - **Landed so far — the M/E build record.** The Phase-B ML substrate below is **BUILT**: each slice an
    oracle-first, parity-gated, PR-sized landing, all **off the legality path** (no verifier touched, no
    `Diagnostic` emitted), pure-Python oracle + emitted-C twins gated in `tools/c/check_runtime.sh`,
    deterministic given the seed. The per-slice build narratives are summarized in
    [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md); the definitive detail is the code + tests.

    | Slice | What shipped | Where |
    |---|---|---|
    | B3 Phase 1 | Reverse-mode autodiff as content-addressed rewrites over a **closed primitive set** `{const, var, neg, add, sub, mul, div, dot, select}`, the closure **machine-proven** (no adjoint rule ever emits a foreign op, so reverse-over-reverse stays in-vocabulary — the canonical-form property the `gem.autodiff` law op relies on) | `bcir/kbcir/autodiff.py`, `bcir/tests/test_autodiff_closure.py` |
    | M1 | The loss head: closed-set MSE built into the `Tape`; transcendental losses (softmax-CE, BCE-with-logits, hinge) return closed-form **gradient seeds** so a parameter gradient never carries a transcendental | `bcir/kbcir/losses.py` |
    | M2 | Adaptive optimizers — momentum, RMSprop, Adam (bias-corrected) — as a side-effect-free reference oracle + emitted C steps (RMSprop/Adam ride the trusted libm `sqrtf` edge; SGD/momentum are pure arithmetic) | `bcir/lower/optimizers.py` |
    | M3 | The epoch / mini-batch training loop (deterministic LCG shuffle, train/val split, metrics, early stop) — **the Tier-1 trio (M1–M3) is complete: BCIR trains logistic regression and a small MLP end-to-end** | `bcir/kbcir/training.py` |
    | E1 / E2 | OLS + PCA on the Area-B "integrate, don't reinvent" wrap pattern (LAPACK `sgels` / `ssyev` when linked, portable normal-equations / Jacobi fallbacks, independent optimality/eigen verifiers, R17 input bridges) | `bcir/kbcir/ols.py`, `bcir/kbcir/pca.py` |
    | E3 | A full Transformer encoder block (multi-head, batched, causal-masked, POST-/PRE-LN) **composed** from the existing attention/softmax/matmul references; LayerNorm is the one net-new primitive | `bcir/kbcir/transformer.py` |
    | E4 | RNN / LSTM / GRU — a two-tier design forced by the autodiff closure: a closed-set relu-RNN trainable end-to-end via the existing `unroll_scan` + `grad` (literal BPTT, finite-difference-verified), and LSTM/GRU cells with closed-form gate gradients on the libm edge | `bcir/kbcir/recurrent.py` |
    | E5 | Classical-ML **PREDICT** wraps (KNN / decision tree / SVM / Gaussian-NB) — **the train-vs-predict research finding E7 cites**: training (tree induction, the SVM dual QP, NB fitting) is iterative/combinatorial and stays library/Python-side; predict over a **baked** model is the fixed-shape G5 kernel pattern BCIR owns | `bcir/kbcir/classical.py` |
    | E6 | Unsupervised + the data pipeline: K-means (deterministic Lloyd, exact assign kernel), Standard/MinMax scalers, CV folds, an autoencoder, embedding lookup — all reusing E5/M1/M3 pieces | `bcir/kbcir/unsupervised.py` |
    | E7 | The language-placement **capstone**: every ML/numeric component classified into the Python / C / MLIR / C++ hierarchy by five criteria (two-truth, train-vs-predict, exact-vs-transcendental, planned-claim cost, the C++ boundary) | [`ML_LANGUAGE_PLACEMENT_ANALYSIS.md`](ML_LANGUAGE_PLACEMENT_ANALYSIS.md) |
- **B4 — Hybrid tropical + selective gradient.** Reframe training: the **tropical planner finds the structure**
  (layout, schedule, fusion — exact, deterministic), **gradient steps tune the weights** (graded side). Many
  training problems become tropical optimization + a few gradient steps. `softdp` (the finite-T posterior) and
  the `regret` ledger are the learned guidance — always graded, never on L0.
- **B5 — Integrate existing C numerical libraries as intrinsics (do NOT rebuild XLA/TF).** Wrap **ATLAS, GSL,
  FFTW, OpenBLAS/LAPACK, SLEEF/libmvec** through the *existing* `c.call.libm:{xgemm,...}` edge; `bcir-cc --emit-c`
  links `-lblas`/`-lfftw3`. BCIR's value is optimizing the **calling side** (layout, prefetch, fusion, tiling,
  channel selection) around a trusted kernel. Precision bridges Q8 → trained float32 → Q8, certified by the R17
  accuracy law. *Compress what we need, integrate the rest.*

### Phase C — Data + memory organs (feeding the ML) — *extends CT1 / CT3*
*Builds on the ETL/binary-record frontends, the telemetry ring, and the memory-tier cost model (which today is
cost-only — this phase materializes it).*

- **C1 — Tabular streaming → tensor ops.** Remodel a **FreeTDS**-style row source into a column-oriented
  streaming buffer; streamed rows become a tensor stream over `!bcir.token` + GEM waves — *accelerated* tabular
  learning instead of sequential epochs. Reuses the binary-record decoder + the telemetry ring; net-new: the
  column buffer + windowed-aggregate ops.
- **C2 — The BCIR vector database.** **Materialize the HAM** O(log n) cost abstraction into a real hierarchical
  index (today HAM is priced but never allocated), persisted through **HDF5** + **LMDB**, with embeddings from
  Phase B. The **content-addressed operad index** (the FNV fingerprint) is the natural vector key — CSE and the
  liked-pair identity `a = a` give dedup and reproducibility for free. Combine with the existing memory tiers
  (L1..SSD/CXL) so hot vectors promote and cold vectors demote under the same cost model.
- **C3 — Cloud deployment.** Promote the telemetry **ring → a real Kafka producer** (net-new C; the Python
  abstraction exists); use **Zarr** chunked arrays as the StreamPack-friendly tensor-on-disk format; target
  **WASM** as a channel for portable deployment. *Assess vs alternatives* (Arrow/Parquet for columnar; Arrow
  Flight vs Kafka for transport) before committing — the StreamPack ABI and the channel model are the fixed
  points.

### Phase D — Language reach (more goal-graph sources) — *extends Phase F*
- **D1 — Fortran, immediate: GCC/Flang fallback.** Compile Fortran with Flang/GCC → object/static lib; BCIR
  **calls it via the standard ABI / `ISO_C_BINDING`** over the existing `c.call` + `--fallback` seam, and
  optimizes the *calling* side (layout, prefetch, fusion). No Fortran frontend required to get value.
- **D2 — Fortran, later: selective deep integration.** Treat only the high-value subsets as **BCIR intrinsics /
  specialized ops** (the same pattern as B5): Fortran array sections, intrinsic math, BLAS/LAPACK-shaped kernels,
  and HPC `do` loops with known trip counts over contiguous arrays. Not a full language frontend — targeted
  intrinsics. (C++ and the Python lifter remain master-roadmap Phase F.)

### Phase E — Hardware-native deployment (the body's organs) — *this is master-roadmap Phase D, now ML-guided*
*Deliberately after Phase B: drivers and JIT microkernels are far more powerful once the ML layer can guide
them (thermal/power/clock optimization, adaptive unrolling, best-ISA selection).*

- **E1 — JIT kernels + ML-guided code generation.** Wire **Intel IPP** (and equivalents) as channel-backed JIT
  kernels; the **calibration loop + `moegate` router + telemetry** make data-driven choices of vector
  instruction set, cache-tiling strategy, register file, execution unit, and clock — *learning over time* which
  combination gives the best energy/performance on the current silicon (the recursive-intelligence seed). Extends
  naturally to thermal derating, power-domain decisions, adaptive unrolling.
- **E2 — ML SMBIOS/UEFI boot + the kernel triage.** Build the boot/discovery layer **before** drivers: SMBIOS +
  UEFI feed the **`channel.json` profile** (the reference-state bootstrap — the body learning its own organs);
  ML decides boot/driver configuration. Explicitly plan the **Linux-master-kernel ABI/IPC triage**: which ABI
  and inter-process-communication contracts to keep vs re-derive through BCIR.
- **E3 — Drivers / JIT microkernels (ML-guided).** The `drivers/` JIT generator + per-channel resident drivers
  (master-roadmap Phase D), now consuming the ML guidance from E1 to give each hardware **channel** its real,
  measured-calibrated driver. Closes the heterogeneous-tower loop (FPGA/NVMe/HBM-PIM become driver-backed).

### Phase F — Higher cognition (NLP + recursive intelligence) — *extends Phase L*
*The payoff layer, on top of compilers + kernels + the ML library.*

- **F1 — The NLP base: a deterministic token generator.** After codegen is solid, pair an **ML FreeType**
  interface with the **Unicode database** as the base tokenizer: glyph/grapheme/codepoint structure → tokens,
  with the **operad content-addressed index** as the stable token fingerprint (reproducible, CSE-friendly). This
  is the substrate for an NLP system that inherits BCIR's determinism + provenance.
- **F2 — The file-creation backbone.** Use the **Binary File Descriptor (BFD)** library as the object/file
  emission backbone, wired into the native-object gate ([`BCIR_NATIVE_OBJECT_GATE.md`](BCIR_NATIVE_OBJECT_GATE.md))
  — ML + BCIR deciding object layout/section placement.
- **F3 — Recursive intelligence (the capstone).** The L3 meta-policy loop closes **on itself**: the regret
  ledger + the `ΔL` trigger applied to the *optimizer's own* policies, so the system self-optimizes its
  optimization strategies — the point where genuine reference-state self-understanding emerges. It stays **L3,
  graded, human-actuated** by the quarantine: the system *recommends* changes to itself; a human (or a frozen,
  replay-certified gate) actuates. No recursion is ever allowed onto the L0 hot path.

---

## 3. The continuous-development discipline (how every layer keeps evolving)

This is the engineering method that makes "continuous development at every level" real rather than aspirational.
Every new ML capability in §2 obeys it:

1. **Place it on the L0–L3 ladder first.** Where does the *training* live (L2/L3, float, Python)? What *freezes
   to Q8* (the deployed artifact)? What, if anything, crosses to L1 (a frozen table) — and never to L0. SGD,
   autodiff, and posterior inference are L2/L3 graded machinery; only their frozen Q8 outputs deploy.
2. **Prototype in the oracle, then port to the law.** A tensor op, a gradient rewrite, a new R-law — first a
   Python oracle prototype (cheap iteration, the conformance reference), then the MLIR/C++ law and/or the C
   twin, gated by an oracle↔law↔C parity test. The oracle then **freezes** (the prototype-then-port discipline,
   master roadmap §3).
3. **Close the provenance/regret feedback loop.** Measure (telemetry ring) → book regret (`ΔL`) → recalibrate
   (`calibloop`) → re-freeze a generation-tagged table → R13 manifest. This loop *is* the continuous-development
   engine; every organ in §1 plugs its growth axis into it.
4. **Hold the two-truth invariant.** A new capability may produce graded `(v,w)` guidance, but legality stays
   classical (the R-laws + `verify_quarantine`). A confidence never becomes a verdict; a diagnostic never
   carries a confidence.

---

## 4. Capability-track placement (do we need CT6 / CT7?)

The ideas in §2 mostly extend existing tracks: **CT1** (memory) gains the materialized HAM + the vector DB
(C2) and the tensor-on-disk formats (C3); **CT3** (frontends) gains tabular streaming (C1), the Fortran seam
(D), and the FreeType/Unicode tokenizer (F1); **CT4** (calibration) gains the ML-guided JIT loop (E1); **CT5**
(learning) gains gradient machinery (B3) and the recursive meta-policy (F3). Two genuinely new groupings are
worth promoting to their own tracks once Phase B lands:

- **CT6 — ML primitives.** Tensor/gradient/training ops as first-class claim-graph + cost-model citizens
  (B1–B4), with their own R-laws (shape/dtype/accuracy).
- **CT7 — Data & persistence organs.** Streaming sources, the vector DB, and the cloud transport (C1–C3) as a
  named track distinct from the static ETL frontends.

Recommendation: develop them inside CT1/CT3/CT5 through Phase B/C, and formalize CT6/CT7 in the master roadmap
once the first tensor op and the first vector-DB slice are dual-rail and measured.

---

## 5. Risk register / honest boundaries (out of the dreamy potentials)

- **Substrate–intelligence inversion.** A rich learned/tensor stack over a backend whose real-silicon win is
  still *modeled, not measured* (master roadmap §5.4, deferred pending a rig). *Mitigation:* every tensor op is
  held to the **same Clang-equivalence + measured-replan gate** as the rest of BCIR; ML ops are throttled and
  never block the keystone; the quarantine keeps the learned side off L0.
- **"Don't rebuild XLA/TensorFlow."** The discipline is **integrate, don't reinvent** — wrap ATLAS/GSL/FFTW/
  BLAS-LAPACK (and later cuBLAS/XNNPACK) through the libm/FFI seam and win on the *calling side*. Re-deriving a
  framework violates the "compress only what we need" rule (master roadmap Phase L).
- **Float on the deterministic path is forbidden.** ML trains in float on the graded/Python side and **freezes
  to Q8** to deploy; the R17 accuracy law is the bridge that makes the frozen artifact certifiable.
- **Neuromorphic is explicitly rejected.** The reference-state-optimization approach on CMOS, not event-driven
  biological mimicry, is the bet. The "life-like" trait is Reference State, not spiking.
- **Near-term vs vision.** F1–F3 (NLP, BFD, recursive self-optimization) and E2 (ML boot) are **horizons** —
  named here for order, not started. The immediate, gateable work is **A1–A2 + B1 + B5** (a C23-maxed substrate,
  one tensor op with K_BCIR tile search, one BLAS kernel wrapped via the libm seam).

---

## 6. Where to start (the first concrete, gateable slices)

> **Status (2026-06-28, see [`VISION_ALIGNMENT_AUDIT.md`](VISION_ALIGNMENT_AUDIT.md)).** Slices 1–3 below
> (A1 `_BitInt(N)` + fusion attributes, B1 `gem.matmul` + MLIR law, B5 one BLAS `gemm` wrap) are **DONE**.
> The active frontier is **finishing the B5/Area-B library-integration breadth** — the calling-side win the
> roadmap is built on — before C1.

In dependency order, the lowest-risk entry points — each a single PR-sized, oracle-first, parity-gated slice:

1. ✅ **A1 slice — DONE.** `_BitInt(N)` lowered end-to-end; `[[unsequenced]]`/`[[reproducible]]` elevated to a
   fusion-legality signal the cost model prices. (C23 substrate, dual-rail, Clang-equivalence gated.)
2. ✅ **B1 slice — DONE.** One `gem.matmul` op (oracle + MLIR law), the K_BCIR search choosing tile size / loop
   order over new cost-vector dimensions; verified against a reference C matmul.
3. ✅ **B5 slice (first kernel) — DONE.** One BLAS `gemm` wrapped through the `c.call.libm:` edge with the
   R17-certified Q8↔f32↔Q8 bridge; portable reference fallback.

**The active Area-B frontier — finish library integration (B5 breadth, the calling-side win):**

4. **B1-link slice** — `bcir-cc --emit-c` emits the **automatic link flags** (`-lblas`/`-lfftw3`/…) implied by
   the `c.call.libm:` edges a unit actually uses, so a wrapped kernel links end-to-end with no manual flags.
5. **B2 slice** — wrap a **new** C math library (FFTW *or* LAPACK *or* GSL *or* SLEEF) through the same edge,
   B5-style, with the R17 bridge at the seam and a portable fallback.
6. **B3 slice** — **calling-side tuning** (layout / tiling / prefetch / channel selection) around a wrapped
   kernel — the first concrete step toward the Pillar-3 layout intelligence the audit flags as missing.
7. **Area-B breadth** — ATLAS / GSL / FFTW / OpenBLAS-LAPACK / SLEEF wrapped through the same edge.
8. **C1 slice** — a column-oriented streaming buffer over the binary-record decoder, feeding a B1 matmul as a
   tensor stream.

Each slice deepens master-roadmap **Phase M**; the data/driver/cognition phases (C–F) follow as the substrate
proves out. The order is **A → B → (C ∥ D) → E → F**, with B throttled-parallel to the ongoing C-frontend /
freestanding-driver work (§5.14), and the two-truth quarantine + prototype-then-port discipline applied at
every step.

---

## 7. Open-weight model ingestion (GLM / Gemma / Qwen) — the LLM-serving horizon

> Moved here from `OPENAI_BCIR_INTEGRATION_RESEARCH.md` (which now covers only the OpenAI
> product-surface integration): open weights are an **ML/AI-substrate** program, not an OpenAI
> integration. They change the problem from "GPT as a remote teacher" to "the model is an artifact
> BCIR may own, inspect, quantize, place, and serve." BCIR is conceptually well suited to this —
> its core job is turning a semantic computation into a legal, costed, target-aware realization
> with telemetry and replay — but BCIR currently has **ML primitives and small-model
> training/inference** (§2 Phase B), not a full LLM runtime that can load modern checkpoint
> formats. This section is a **horizon track** in the same spirit as Phase F: named for order,
> gated behind the Phase B/C substrate.

### 7.1 Model-family fit

| Open-weight family | Fit for BCIR now | Why | Main difficulty |
|---|---|---|---|
| GLM-5.2-class heavy models | Research / cluster-scale target | Strong open-weight coding/agent model; useful as a local teacher or high-end endpoint if the deployment stack already exists | Very large memory/KV-cache, tensor/expert parallelism, long-context attention, production scheduler, tokenizer/checkpoint compatibility, safety and license review |
| Gemma 4-class models | Best practical first target | Open weights for responsible commercial use; positioned for advanced reasoning/agentic workloads and optimized deployment across hardware classes | Exact tokenizer, weight-layout importer, attention/RoPE/norm kernels, quantization and eval harness |
| Qwen open-weight models | Practical first/second target, especially coder/agent variants | Widely used open-weight coding/reasoning models with deployment recipes; smaller dense/MoE variants realistic for local or hosted BCIR endpoints | Architecture variants, chat templates, tokenizer edge cases, MoE/expert routing, license/version matrix |

The practical recommendation: **start with a smaller Gemma/Qwen dense instruct model**, prove the
checkpoint → BCIR manifest → quantized inference → telemetry → eval loop, then add larger
Qwen/Gemma variants, and treat GLM-5.2-class models as a scale-out target once BCIR has sharding,
KV-cache management, and production serving.

Sources: GLM-5.2 announcement (<https://z.ai/blog/glm-5.2>), GLM-5 repository
(<https://github.com/zai-org/GLM-5>), Gemma 4 model overview (<https://ai.google.dev/gemma/docs/core>),
Google DeepMind Gemma 4 page (<https://deepmind.google/models/gemma/gemma-4/>), Gemma open-weight
library (<https://github.com/google-deepmind/gemma>), Qwen3 announcement
(<https://qwenlm.github.io/blog/qwen3/>), Qwen3.5 announcement (<https://qwen.ai/blog?id=qwen3.5>),
Qwen3.6 repository (<https://github.com/QwenLM/Qwen3.6>).

### 7.2 What BCIR already has (the §1/§2 substrate, restated for this track)

- **Tensor/math primitives:** matmul, activation, softmax, attention, transformer block references,
  layernorm, recurrent models, classical models, quantization, losses, optimizers, autodiff.
- **Training scaffolding:** deterministic datasets/mini-batches, train/validation splits, supervised
  training loops, metrics, early stopping, optimizer state (the M1–M3 trio).
- **Lowering paths:** C kernels, LLVM/JIT/AOT hooks, SYCL dispatch, WASM, specialist lowerings,
  target/channel descriptions.
- **Optimization and placement:** K_BCIR cost vectors, target profiles, RCSP, telemetry,
  calibration, regret, portfolio routing, provenance manifests.
- **Safety/correctness gates:** the R-laws, the two-truth quarantine, parity discipline, fuzzing,
  replay, C/LLVM equivalence checks, telemetry integrity, docs/training separation.

Enough for **small BCIR-native endpoint models** and **pieces of LLM inference** — not yet for
drop-in loading of a modern open-weight chat model.

### 7.3 What is missing to plug in open weights

| Missing layer | What must be built | Why it matters |
|---|---|---|
| Checkpoint importer | Load `safetensors`/GGUF/HF shard layouts; map tensor names/shapes/dtypes to a BCIR `ModelManifest`; validate hashes/licenses | A trustworthy bridge from external weights into content-addressed artifacts |
| Tokenizer + chat-template rail | BPE/SentencePiece compatibility, special/tool-call tokens, chat templates, detokenization tests | An LLM endpoint is wrong if tokenization or prompt formatting drifts from the model contract |
| LLM graph dialect | First-class ops for embedding, RMSNorm/LayerNorm variants, RoPE/ALiBi, grouped-query attention, sliding-window attention, MoE routing, KV-cache read/write, logits head, sampling | The E3 transformer is an oracle composition, not a complete modern decoder-only LLM dialect |
| KV-cache + serving runtime | Paged KV cache, prefill/decode split, continuous batching, speculative-decoding hooks, streaming tokens, cancellation, multi-session state | Production endpoints are dominated by decode scheduling and KV memory, not one-shot matmul |
| Quantization formats | Weight-only int4/int8, activation quantization, per-channel/per-group scales, GGUF/AWQ/GPTQ/FP8-style adapters, accuracy-law (R17) extensions | Open models are practical only when quantized and accuracy-bounded |
| Parallel placement | Tensor/pipeline/expert parallelism, CPU/GPU/NPU offload, a multi-device channel cost model | GLM-class models require scale-out; even smaller models benefit from heterogeneous placement |
| Kernel library | Fused QKV, attention kernels, RoPE, RMSNorm, gated MLP/SwiGLU/GELU, dequantized GEMM, MoE dispatch, logits/sampling kernels | The existing references need production kernels and law parity |
| Endpoint API | An OpenAI-compatible `/v1/chat/completions` or Responses-like adapter, streaming, tool-calling schema, structured outputs, auth/quota/rate limits | Makes BCIR-owned models usable by existing agent tooling |
| Eval + safety harness | Per-model eval packs, jailbreak/prompt-injection tests, license/safety metadata, red-team corpora, hallucination/faithfulness checks | Open weights remove provider-side guardrails; BCIR must own the deployment safety envelope |

### 7.4 A staged implementation path

1. ✅ **Manifest-only ingestion — LANDED** (`bcir/frontends/models/manifest.py`,
   `test_model_manifest.py`): `ModelManifest` records — architecture, license, tokenizer ref,
   weight-shard inventory + streamed sha256 hashes, dtype census, parameter count, context
   length — built from the safetensors HEADERS + config only (the weight bytes are hashed for
   integrity, never interpreted), deterministic (canonical-JSON digest, ingestion-order-free),
   JSON round-tripping, loud on malformed shards. Dep-free stdlib. *(Built before any weight
   loading or decode kernels, per the contract.)*
2. ✅ **Tokenizer parity — LANDED** (`bcir/frontends/models/tokenizer.py`,
   `test_model_tokenizer.py`): a dep-free byte-level BPE reference (the HF `tokenizer.json`
   shape: byte alphabet, rank-based merges, specials never split), LOSSLESS round-trip over
   arbitrary unicode by construction, golden ids over a hand-computed mini fixture, the
   Gemma-style chat template as a pinned named fixture, and the tokenizer sha256 tied into
   `ModelManifest.tokenizer_digest` (the wrong tokenizer for a model is detected by hash).
   Byte-for-byte parity against a specific released model lands when its real tokenizer.json
   is ingested (the loader accepts the real shape; the exact pre-tokenizer regex is per-model).
3. ✅ **Reference decode — LANDED** (`bcir/frontends/models/decode.py`, `test_model_decode.py`):
   a slow, dependency-light dense-decoder reference (the Gemma/Llama pre-norm shape) COMPOSED
   from the existing oracle pieces — `embedding_lookup` → per layer [`rmsnorm_reference` →
   Q/K/V `matmul_reference` → `rope_reference` per head → causal `scores_reference` +
   `softmax_reference` → W_o + residual → RMSNorm → `feedforward_reference` + residual] →
   final RMSNorm → tied-embedding logits → greedy argmax. Two decode paths, one truth:
   naive full recompute AND the incremental **KV-cache twin** emit the same ids BIT-FOR-BIT
   (the E3 reference-vs-realization pattern); causality pinned (a later token never moves an
   earlier row, exactly); and the ladder ties — the synthetic model's shard census (rung 1
   `param_count`) equals `decoder_param_count(spec)` and the manifest carries the rung-2
   tokenizer digest that encoded the decoded prompt. Byte parity against a released
   checkpoint lands when its real weights are ingested (rung 4's quantized artifact).
4. ✅ **Quantized inference artifact — LANDED** (`bcir/frontends/models/quantized.py`,
   `test_model_quantized.py`): the Q8↔float32 bridge wrapped around the trusted rung-3
   decoder (the `attention_via_bridge` pattern) — the artifact IS the reference decoder over
   per-group round-tripped weights, so the sole certified error is the R17-bounded weight
   quantization (≤1 ULP per value at each group's grid). The drift discipline: a
   deterministic `DriftRecord` per prompt fixture — both paths' greedy ids, max teacher-forced
   logit drift along the float trajectory, mean-NLL (perplexity proxy) under both models,
   the R17 stamp. Measured on the synthetic fixture: Q8 ~2.7e-3 max logit drift (~20×
   tighter than Q4), and a greedy flip is *recorded, never hidden* (`ids_match`). Real-weight
   ingestion (a released tiny checkpoint through manifest → tokenizer → decode → quantize)
   is the remaining half, gated on rung 5's law rail for the LLM ops.
5. ◑ **C/MLIR law rail — FIRST SLICE LANDED** (`verify_llm_ops.mlir`): ODS ops for the rung-3
   decoder's LLM-specific stages — `bcir.gem.embedding` / `bcir.gem.rmsnorm` / `bcir.gem.rope` —
   with op-level laws (positive extents; `gamma_len == dim`; RoPE's **even-dim** pairing law; the
   f32 libm-edge quarantine rule on rmsnorm/rope) and the D2 adjacency seams in `-bcir-verify`:
   embedding→rmsnorm extent + dtype handover (R22/R23), rope→attention head-width `d_k == dim` +
   dtype (R22/R23) — exactly the chain `decoder_layer_reference` composes, with negatives.
   *Remaining:* GQA/KV-cache ops + the C-twin kernels (prototype-then-port).
6. **Serving endpoint** — streaming decode, schema-constrained tool-call output, telemetry frames,
   replay manifests.
7. **Scale-out** — continuous batching, paged KV, multi-device placement, expert/tensor
   parallelism for larger Qwen/Gemma and eventually GLM-class models.
8. **Fine-tune/adapt** — LoRA/QLoRA-style adapters as first-class artifacts before full-parameter
   training; adapters frozen with the same provenance and eval gates as kernels.

**Endpoint gates (when models serve production traffic):** shadow-mode deployment before live
routing; confidence/uncertainty thresholds that escalate to a frontier model or human review;
drift monitoring + periodic replay against frozen evals; and the hard separation of endpoint
predictions from BCIR legality verdicts (the two-truth quarantine, §0).

### 7.5 Bottom line

BCIR is **architecturally well suited** to open weights — it already thinks in typed graphs,
lowering, costed placement, telemetry, quantization, parity, and provenance — but it is **not yet
a plug-and-play LLM inference engine**. The credible path is not GLM-first; it is a small
Gemma/Qwen dense model through the manifest → tokenizer → reference-decode → quantized-artifact
ladder, lowered into BCIR kernels and exposed as a guarded endpoint. After that, heavier models
are an engineering problem (sharding, KV memory, kernel performance, safety operations), not a
conceptual mismatch.

---

## 8. Feasibility audit — the deeper-integration program (2026-07-02)

> A technical audit answering: why wrap numerical libraries instead of adopting a full ML
> framework; how BCIR goes deeper than §2 currently proposes; how telemetry stays
> instant/minimal-overhead and feeds recursive learning; how *all* operations feed one
> recursive-optimization ecosystem without degrading cached learned artifacts; and how BCIR
> becomes a vertically-integrated model-creation/training/deployment system and an AI
> metaprogramming ecosystem. Verdicts are conservative and every mechanism is named against
> existing machinery. Draws on the C23 self-assembly analysis (typeof/`_BitInt`/`#embed`/
> `constexpr` compile-time training structures) from the prior-project research corpus.

### 8.1 Why wrapping beats importing a full ML library (the technical case)

**The mechanism.** An Area-B wrap is a *claim* whose op string names a trusted external
kernel (`c.call.libm:cblas_sgemm`, `:fftwf_plan_dft_2d`, `:LAPACKE_sgels`, …). That one edge
buys, mechanically:
1. **A typed, costed seam.** The claim carries shape/stride/dtype and a 12-d cost vector, so
   the K_BCIR planner prices *around* the kernel — layout (SoA↔AoS pivot), tiling, prefetch,
   fusion of producers/consumers, channel placement, and Θ-feasibility — exactly the things a
   framework's opaque executor decides internally and invisibly.
2. **A certified numerical boundary.** The R17 Q8↔f32↔Q8 bridge bounds the *only* error BCIR
   introduces (the input round-trip); the kernel interior is trusted-exact, and an
   *independent verifier* (normal-equation residual, eigen residual, …) checks the result
   without re-deriving the algorithm.
3. **Deterministic linkage + portability.** `linkflags.py`/`bcir_lib_for_callee` derive
   `-lblas/-lfftw3/-llapack/…` from the claim graph itself, and every wrap ships a portable
   fallback (normal equations, Jacobi eig) so CI needs no vendor library.
4. **Legality isolation.** The wrap can never become a verdict: it is off the legality path
   by construction (the two-truth line), so importing a numerical bug cannot corrupt R-law
   verification.

**Versus a full framework (XLA/TF/PyTorch).** A framework import would bring its own graph
representation, scheduler, memory planner, and runtime — precisely the organs BCIR *is*.
The result would be two planners fighting over the same decisions with no shared cost model,
a dependency mass that breaks the dependency-free-oracle and freestanding-C properties, and
an opaque legality story. The E5/E7 research finding generalizes this: the *iterative/
combinatorial* halves (framework training loops, autotuners) are a poor fit for the
fixed-shape claim model and belong outside; the *fixed-shape* halves (kernels) are exactly
what claims express. **Verdict: the wrap-and-optimize-the-calling-side route is the optimized
route for kernel interiors** — with one honest caveat: wraps alone do not give whole-graph
ML optimization. That is what B1's tensor-ops-as-claims provides (BCIR owns the graph,
wraps own the leaf kernels); the two are complements, not alternatives.

### 8.2 Going deeper than the current proposal

The §2 program stops at "tensor ops as claims + wrapped kernels + a Python training loop."
The audit finds five deepening moves, all quarantine-compatible:

- ◑ **D1 — Training as a planned graph, not a Python loop. FIRST SLICE LANDED**
  (`bcir/kbcir/train_graph.py`, `test_train_graph.py`): one training step is six chained
  first-class claim phases (forward `gem.matmul` → `gem.activation` → `gem.loss` →
  `reduce.loss_mean` → `gem.autodiff` → `gem.opt_step`) — law-clean under R1–R22, priced by the
  tropical optimizer (realized in stage order), composed over steps via `kbcir.compose` (a run
  is a `Seq`; series-summed cost), RCSP-budget-feasible-or-not BEFORE execution,
  R13-deterministic, and structurally bridged to the M3 loop (`steps_for` = epochs ×
  batches/epoch = one update claim per optimizer step). ✅ **Step 2 LANDED**: `hydrate_train_step` lowers the
  selected realization to a StreamPack (R10 provenance / R11 tags clean, segments == the plan's
  realized order) and `train_planned` runs REAL training with the GEM executor dispatching the
  six numeric stage kernels per step — one executed step matches the closed-form logistic
  reference to 1e-12, the run converges under the shared gate, and every epoch commits a
  replayable ProvenanceManifest (epoch + pack generations as artifact tags; digests distinct,
  `diff` == artifacts, `replay` exact). ✅ **Step 3 LANDED**: the binary StreamPack rail
  (`test_train_pack_exec.py`) — the hydrated train-step pack encodes to the binary ABI
  (`bcir.abi.encode`) and executes through the C executor (`runtime/c/bcir_exec.c`) with the
  oracle's exact dispatch order / phase order / per-phase telemetry, and the C order equals the
  claim order every executed step of a real convergent `train_planned` run used — one step of
  training is a no-Python hot artifact. ✅ **Step 4 LANDED**: the train step COMPUTES in C
  (`runtime/c/bcir_train.c` + `test_train.c`, gated by `test_train_c_kernels.py`) — the six
  numeric stage kernels (kernel-for-kernel twins of `_step_kernels`: ascending-index sums, the
  guarded two-branch sigmoid, the eps-clamped BCE) behind `bcir_exec`'s per-claim callback,
  the whole loop (epochs × batches → dispatch → compute) in C over the binary pack. The
  differential gate: per-epoch losses + trained weights == `train_planned` to ≤1e-12, the
  first step's dispatch order is the executor's [1..6], and the C curve passes the SAME shared
  convergence gate as the oracle run — **training as a C artifact, no Python in the loop**.
  *Next:* overlap/EFT scheduling of the stage streams. The autodiff closure proof is the
  enabler: the gradient DAG has a fixed vocabulary, so it hydrates to a StreamPack like any
  program.
- ✅ **D2 — Shape/dtype as first-class R-laws (R22/R23). LANDED.** `check_transformer`/
  `check_classical`-style checkers were op-level and advisory; shape consistency and dtype
  compatibility are now numbered laws via the R19–R21 six-artifact pattern: **R22** checks the
  `gem.*` producer→consumer SEAM on both rails (oracle `verify.verify_shape` over the count
  handover; MLIR `verifyR22` over matmul→activation extent, the fusion adjacency contract) and
  **R23** the dtype handover (conv/attention→activation on MLIR; the E3–E6 quarantine dtype rules
  at the spec level via `verify.verify_ml_spec`, which promotes every checker message to R22/R23).
  Negative MLIR cases in `verify_shape_dtype.mlir`; oracle suite `test_shape_dtype_laws.py`;
  `gen_status` sweeps R1–R23. Vacuous-by-default (non-disturbance). This is the "structurally
  valid tensors" guarantee (§8.4) made law.
- ◑ **D3 — Learned cost-model priors at L1. FIRST SLICE LANDED** (`kbcir/tile_prior.py`,
  `test_tile_prior.py`): the accel-ranker precedent generalized to the L1 tile search — a
  logistic prior over CHEAP tile features (no `cost_of` to rank), trained offline on the exact
  search's own choices under the (calibrated) `TargetProfile`, frozen to a Q8 integer table,
  used only to ORDER `plan_matmul`'s search with a PROOF-gated early exit (the compute term is
  tile-independent, so a cache-fitting candidate at the bottleneck floor is unbeatable).
  `TilePriorCertificate` is the safety witness: guided == exhaustive on (fits_cache,
  bottleneck) over held-out shapes, mismatches 0, **71% fewer nodes costed** (gate ≥40%); under
  a bandwidth-starved calibration the proof never fires and the search honestly degenerates to
  exhaustive (still exact). Calibloop wiring is by construction — train against the measured
  profile the loop froze; `plan_matmul` itself untouched (opt-in, vacuous by default). *Next:*
  channel-choice priors + per-shape-class tables persisted alongside the calibloop's cal_gen.
- **D4 — E-graph rule synthesis (the operad 2-cell algebra).** Learn *candidate* rewrites
  from liked/unliked pair statistics; each learned rule is admitted only with a machine-
  checkable equivalence certificate (the egraph extract cost proof), keeping learning out of
  legality.
- **D5 — The LLM dialect (§7.3)** — the ceiling-raiser: embedding/RMSNorm/RoPE/GQA/KV-cache
  ops as claims makes decode scheduling a K_BCIR problem (prefill/decode split ≈ phase
  structure; paged KV ≈ registry-first resources with generations; continuous batching ≈
  wave scheduling). BCIR's scheduler vocabulary already matches the shape of the problem.

### 8.3 Telemetry: instant reads, minimal overhead, on the learning loop

**Already built and load-bearing:** the preallocated `TelemetryRing` (fixed 56-byte `<7q>`
records, `pack_into`, non-blocking drop-and-count — a write is a bounded store, never an
allocation or a lock); the zero-copy shared-memory ring with a validated header (forged
headers rejected); reject-don't-clamp ingest (`sanitize_events` + integrity witness); and
**uncertainty-gated sensing** (`sensing.py`: sample only where per-path cost variance or
ranker confidence says measurement pays — the single most important overhead lever).

**The policy this audit fixes:** (1) L0 writes are fixed-cost ring stores only — no
formatting, no branching beyond the mask, drop-and-count under pressure (backpressure must
never propagate into the hot path); (2) *reading* is never on the hot path — drains happen at
L2 checkpoints (phase barriers, plan boundaries) or on a separate consumer (the D2.1 UART
frame path); (3) sampling rate is itself a planned, priced decision (the sensing gate),
budgeted through the existing verification/`compile` axes and enforced by `perf_budget`
regression gates; (4) provenance tags ride every record so synthetic never masquerades as
measured. **Feedback wiring (exists, needs closing at scale):** ring → Θ folding
(calibrators) → replan → `CalibrationCertificate` (win ≥ 0) → regret ledger → MDL ΔL trigger
→ retrain/refreeze → R13 generation bump. Every arrow is implemented; the missing piece is
running the loop *continuously* (calibloop as a resident service per channel) rather than
push-button — a Phase-E1 slice.

### 8.4 One recursive ecosystem without degraded loss (the cached-artifact validity law)

The question — how can *all* operations feed recursive reinforced learning while cached
summary learning tensors stay structurally valid — is answered by generalizing the discipline
BCIR already applies to every learned organ:

1. **Every cached learned artifact is a frozen, generation-tagged, fingerprinted object**
   (the FrozenCalibrator/FrozenGate/FrozenRanker/CalibratedProfile pattern): content-addressed
   (FNV, the operad index), schema-versioned, R13-witnessed. A "summary tensor" (a distilled
   dataset, a compressed experience buffer, a teacher-trace digest) enters the ecosystem only
   as such an artifact — never as mutable state.
2. **Structural validity is checkable, not assumed:** shape/dtype law (D2) on the artifact's
   declared schema; the **idempotence gate** from the memory module (`Lim(Res(U))`:
   re-resolving a saturated artifact must be a fixpoint — if re-summarizing a summary changes
   it, it was not converged and is refused admission); and strictly-validating `from_json`
   loaders that reject forged or drifted artifacts.
3. **No silent replacement — supersession with a gate.** A new generation replaces the old
   only through the replay gate (counterfactual no-regression on held-out episodes under the
   frozen neutral judge) — the mechanism that already gates portfolio and MoE promotion.
   Generations are append-only lineage (the provenance DAG), so degradation is *diagnosable*
   (diff two generations) and *reversible* (roll back a generation), never compounding.
4. **Degradation detection is the regret ledger's job:** hindsight regret against the frozen
   judge accumulates per artifact; the MDL boundary `Σ regretᵢ/bestᵢ > (k/2)·ln N` is the
   trigger to retrain from rawer data (the anti-collapse valve against summarizing summaries
   of summaries — teacher-blind-spot inheritance is bounded by always keeping the raw-episode
   tier retrievable and periodically re-distilling from it, not from the cache).
5. **The two-truth quarantine is the global stability theorem:** because no graded artifact
   can become a legality verdict, a degraded learned cache can at worst make plans *slower*,
   never *wrong* — the ecosystem can afford aggressive recursion because its failure mode is
   bounded to cost, not correctness.

### 8.5 The vertically-integrated system (data → dataset → model → training → deployment)

Feasible as a staged bootstrap; every seam exists, ordered by how much new machinery each
source needs:

| Rung | Source | Lands through | Status |
|---|---|---|---|
| 1 | **Telemetry** (self-supervised: cost, schedule, thermal) | ring → episodes → calibrators/rankers | **real today** — the only fully-closed loop |
| 2 | **Built-in tables** (Unicode DB → the F1 tokenizer; Q8/ISA/training tables via `#embed`) | frozen compile-time datasets (the C23 `#embed`/`constexpr` self-assembly pattern) | ABI machinery exists; F1 unstarted |
| 3 | **User input / intent** (ROP/MAP/C sources, CLI episodes) | frontends → claim graphs + provenance | frontends real; intent-mining unstarted |
| 4 | **RAG / vector store** | Phase C2: materialized HAM + operad content-address as the key; HDF5/LMDB persistence | priced-but-not-materialized |
| 5 | **Wikipedia / web scraping** | the ETL rail (parse/FSM/binary) + C1 streaming, with license/provenance tags and the reject-don't-clamp ingest posture | ETL seed exists; scale organs are Phase C |
| 6 | **Frontier-model APIs** (cloud teachers) | typed TrainingSession/Episode records, schema-gated (`OPENAI_BCIR_INTEGRATION_RESEARCH.md` §3.8) | designed, unbuilt |
| 7 | **Local open weights + trainer models** | §7 manifest → tokenizer parity → reference decode → quantized artifact ladder | designed, unbuilt |

The integration law: **every source produces the same thing** — provenance-tagged, typed
episodes/artifacts that the deterministic gates (schema → verifier → parity → replay) admit
or reject. Model creation (§2 Phase B primitives + D1 training graphs), training (M1–M3
generalized to planned graphs), and deployment (`api.build_artifact` → R12-attested kernels;
§7 endpoints) then share one artifact pipeline. **Verdict: feasible; the binding constraints
are the Phase-C data organs and (for rungs 6–7) the serving/eval harness — not the IR.**

### 8.6 AI metaprogramming: user intent forging data structures

The mechanism is **intent → claim-graph synthesis under law**:
- The *target language* already exists: registry-first Resources (whose `layout` soa/aos/
  aosoa/blocked, `access` flat/ham, tiering and priority ARE data-structure decisions the
  cost model prices) + compose region trees + the rewrite algebra.
- The *search space* is the e-graph + operad 2-cells (equivalence-preserving rewrites with
  extraction certificates); the *proposal policy* is learned (MoE gate / ranker class,
  frozen); the *objective* is K_BCIR itself; the *boundary* is the verifier — a synthesized
  structure that fails R-laws simply does not exist.
- On the C rail, the C23 self-assembly toolkit from the prior-project analysis (typeof-generic
  module templates, `_BitInt` exact-width state, `constexpr` rule tables, `#embed` baked
  corpora) is the compile-time materialization of a chosen structure.

So "user intent forges data structure pathways" = an intent frontend (natural language or
declarative spec → goal graph, the instruction-compiler pattern) + a learned proposer emitting
candidate Resource/Claim graphs + verify-plan-measure selecting the winner and freezing it
with provenance. Each stage exists or has a named precedent; the net-new piece is the intent
frontend and the proposer training corpus (which rungs 3/6 of §8.5 supply). **Feasible as a
Phase-F-adjacent track; the quarantine keeps synthesis proposals from ever self-certifying.**

### 8.7 Erasing the line between programming and intelligence

The unification is not rhetorical — it is the claim graph as the *single representation* for
both: a program is a claim graph whose realization is chosen by optimization; a model is a
claim graph whose parameters are chosen by optimization. Traditional layers (MLPs, attention
heads, recurrent cells) integrate as **first-class claims** (G1/G7/E3/E4 already are),
scheduled, fused, placed, budgeted, and verified identically to any other computation — and
conversely, ordinary code becomes *differentiable through selection* (softdp's `dF/dw = E[C]`
makes the compiler's own choices a gradient surface). The two-truth split is what makes the
identity safe: legality stays classical on both sides; choice-under-cost is graded on both
sides. What to build to make it real rather than latent: D1 (training as planned graphs),
D2 (shape/dtype law), and the §8.6 intent loop.

### 8.8 Verdicts + sequencing

| Proposal | Verdict | Gate |
|---|---|---|
| Wrap-not-import (8.1) | **Confirmed optimal**; keep, extend breadth | existing R17 + independent verifiers |
| Training as planned graphs (D1) | Feasible now (oracle→law port) | six-artifact + parity |
| Shape/dtype laws (D2) | ✅ Landed (R22/R23) | R19–R21 promotion pattern |
| Learned cost priors (D3) | Feasible now | accel-certificate pattern (0 mismatches) |
| Resident calibloop service (8.3) | Feasible now (host); measured win still rig-gated | perf_budget + provenance |
| Summary-artifact law (8.4) | Feasible now — mostly codifying existing discipline | replay gate + idempotence |
| Data organs at scale (8.5 rungs 4–5) | Phase C, real engineering | C1/C2 slices |
| Cloud-teacher + open-weight rungs (8.5 rungs 6–7) | Designed; gated on serving/eval harness | §7.4 ladder |
| Intent-synthesis ecosystem (8.6) | Horizon (Phase F-adjacent); seed slices possible now | quarantine + certificates |
| Rule synthesis (D4) | Research-side | equivalence certificates |

Ordering: **D2 → D1 → D3 → resident calibloop → 8.4 codification → C1/C2 → 8.6 seed** —
each PR-sized, oracle-first, parity-gated, per the house discipline.
