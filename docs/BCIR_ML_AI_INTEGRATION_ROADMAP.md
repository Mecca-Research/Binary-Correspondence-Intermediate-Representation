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
wave executor); the R1–R18 verifier (dual-rail MLIR/C++/C + Python) with R19/R20/R21 emerging
([`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §5.14). **Reusable as-is.**

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
  - **M1 — loss-function library BUILT** (`bcir/kbcir/losses.py`, `bcir/tests/test_losses.py`). The loss head
    that turns the existing matmul + activation + autodiff + SGD into trainable supervised models (forward →
    loss → backward → param grads, end-to-end). It implements a **two-path design forced by the autodiff
    closure property** (`bcir/kbcir/autodiff.py`'s docstring, boundary (a): a primitive whose VJP needs
    exp/log breaks the closed set `{const,var,neg,add,sub,mul,div,dot,select}`):
    - **Closed-set loss (MSE)** — built ENTIRELY into the `Tape` as `(1/n)·dot(e, e)` with `e_i = pred_i −
      target_i` (sub + dot + scale), so the EXISTING `autodiff.grad` differentiates and
      `lower/autodiff_kernel.py::emit_autodiff_kernel_c` lowers the whole model→loss DAG for free (verified:
      grad == finite-difference; lowered to C, matched the oracle to round-off).
    - **Transcendental losses (softmax-CE, BCE-with-logits)** — the forward value needs `log`/`exp` (the libm
      side, like `activation.py`), so each returns `(loss_value, grad_logits)` with the famous CLOSED-FORM
      gradient that lives in the closed set (`softmax(logits) − onehot`; `sigmoid(logits) − target`, reusing
      the G1 `softmax_reference`/`sigmoid_reference`). That vector SEEDS the model's backward, so the
      parameter gradient never carries a transcendental. Numerically stable forms: log-sum-exp (subtract
      `max`) for softmax-CE, `max(z,0) − z·y + log(1+exp(−|z|))` for BCE. A `hinge` (SVM) `(value, grad)` is
      included on the same seed path. This unlocks **logistic regression** (BCE) and **multiclass
      classification** (softmax-CE); MSE unlocks the regression head. Pure-Python oracle, off the legality
      path (a loss is cost/optimization-side, never an R-law verdict).
  - **M2 — adaptive optimizers BUILT** (`bcir/lower/optimizers.py`, `bcir/tests/test_optimizers.py`). The
    minimal SGD (`autodiff_kernel.py::emit_sgd_step_c`, `params[i] -= lr·grad[i]`) generalized into the
    standard adaptive family — **momentum (heavy-ball)**, **RMSprop**, **Adam (with bias correction)** — as a
    reference oracle (pure Python, side-effect-free `(params, state) → (params′, state′)`) + an emitted C
    step that mirrors `emit_sgd_step_c` (an in-place `void` over the param vector + state buffers, compiled
    and run by a tempdir harness). The exact conventions: momentum `v = β·v + g; p −= lr·v` (raw grad into
    the decayed velocity, β=0 ⇒ plain SGD); RMSprop `s = β·s + (1−β)·g²; p −= lr·g/(√s + ε)`; Adam
    `t += 1; m = β₁m + (1−β₁)g; v = β₂v + (1−β₂)g²; m̂ = m/(1−β₁ᵗ); v̂ = v/(1−β₂ᵗ); p −= lr·m̂/(√v̂ + ε)`
    (the **bias correction** — the t-dependent divisors — is the distinctive feature and matters most in the
    first few steps; `t` round-trips as state, `int *t` in the C step). **Which ride libm `sqrtf`** (the
    honest `c.call.libm:` edge): SGD and momentum are **pure arithmetic** (`<stddef.h>` only, no `-lm`);
    **RMSprop and Adam need a square root** → `#include <math.h>`, `sqrtf`, link `-lm` (the harness links it
    for those two only). Verified: each reference step matches an independent hand computation (Adam's bias
    correction pinned at t=1 AND t=2); the emitted C step matches the reference to float round-off over
    several steps (Adam's m/v/t state round-trips); each optimizer drives a convex MSE (a linear model, known
    minimum at (w,b)=(2,1)) to ~0 with near-monotone descent, and Adam's per-coordinate scaling beats plain
    SGD on an ill-conditioned variant. Off the legality path (cost/optimization-side, never an R-law verdict;
    touches no verifier, emits no `Diagnostic`). **One step is the primitive** — M3 wires the repeated loop.
  - **M3 — training loop BUILT** (`bcir/kbcir/training.py`, `bcir/tests/test_training.py`). The
    epoch / mini-batch TRAINING LOOP that composes the trio into end-to-end supervised learning: a
    `Dataset(X, y)` abstraction, a DETERMINISTIC seed-keyed shuffle (`minibatches`, a stdlib LCG — no numpy,
    no `random`), a disjoint `train_val_split`, eval metrics (`accuracy` argmax/threshold, `mse_metric`,
    `binary_f1` with the full confusion matrix), an `EarlyStop` patience hook, and
    `train(model, params0, dataset, *, loss, optimizer, epochs, batch_size, lr, val, metrics, early_stop,
    seed) -> TrainResult`. **How it composes** (the same two-path split M1 is forced into by the autodiff
    closure): the `model` callable builds a fresh `Tape` forward per batch; for a **closed-set loss (MSE)**
    the loop builds `mse(...)` INTO the Tape and the EXISTING `autodiff.grad` gives `dL/dparam` directly (no
    seed); for a **transcendental loss (BCE, softmax-CE)** the loop takes M1's closed-form `grad_logits` and
    CHAINS it — `dL/dparam = Σ_k grad_logits[k]·d(logit_k)/dparam` (run `grad` on each logit, scale by the
    seed) — so the transcendental lives only in the monitored loss value, never in the parameter gradient.
    The per-parameter gradient then drives the M2 optimizer rule selected by name (`"adam"` + hypers), the
    loop managing its per-param state (velocity / squared-grad EMA / Adam's m,v,t) across steps. It is the
    oracle GENERALIZATION of `autodiff_kernel.oracle_train` (the single-DAG forward→backward→SGD reference) to
    epochs / mini-batch / arbitrary M1 loss / arbitrary M2 optimizer / held-out val / metrics / early stop.
    Verified: `minibatches` is a deterministic full-coverage permutation (ragged last batch); the metrics
    match a hand-computed confusion matrix; **logistic regression** (BCE + Adam on a linearly-separable set)
    reaches **100% train + 100% val accuracy** in ≤ 40 epochs with a near-monotone loss and identical
    final loss/params across same-seed runs; a **2-layer MLP** (hidden relu via the closed-set `select`,
    BCE + Adam) on a NON-linearly-separable XOR set reaches **100% train accuracy**, clearing the **linear
    model's ceiling (~0.51) by +0.49** — the hidden layer learns the nonlinearity; softmax-CE multiclass and
    the MSE closed-set regression converge too; early stop fires when val plateaus. Off the legality path
    (cost/optimization-side, never an R-law verdict; touches no verifier, emits no `Diagnostic`; the model
    graphs stay in the closed lowerable primitive set). Pure-Python oracle, deterministic given the seed.
  - **Tier-1 trio (M1–M3) COMPLETE: BCIR trains logistic regression + an MLP end-to-end.** M1 (loss head) +
    M2 (adaptive optimizers) + M3 (the training loop) close the loop that turns BCIR's existing matmul +
    activation + reverse-mode autodiff into supervised learning — forward → loss → backward → optimizer step,
    over epochs and mini-batches, to high accuracy on toy datasets — entirely as a pure-Python oracle off the
    legality path. The capstone demonstration trains **logistic regression** (the BCE closed-form seed path)
    and a **small MLP** (the hidden relu, clearing a linear ceiling) end-to-end and deterministically.
  - **E1 — OLS (ordinary least squares) BUILT** (`bcir/kbcir/ols.py`, `bcir/tests/test_ols.py`; the emitted C
    twin `bcir/lower/c_kernel.py::emit_lapack_ols_c`). The first of an ML-breadth series, built on the **exact
    Area-B "integrate, don't reinvent" wrap pattern** the `linsolve` (LAPACK `sgesv`) / BLAS / FFTW / GSL /
    SLEEF kernels use. OLS **generalizes the square dense solve to overdetermined linear regression**: given
    `A` (m×n, m≥n) and `b`, find `x` minimizing `‖A·x − b‖₂` (the line/plane of best fit). The source of truth
    `ols_reference` forms the **normal equations** `G = AᵀA`, `c = Aᵀb` and **REUSES `linsolve.solve_reference`**
    for the inner square solve; `normal_equation_residual` is an *independent* verifier (`max|Aᵀ(A·x − b)|` — the
    optimality condition `AᵀA x = Aᵀb`, ~0 at the optimum even when `‖A·x−b‖ > 0` on an inconsistent system);
    `ols_via_bridge` is the Q8↔f32↔Q8 round-trip then reference. **Honest conditioning note:** the normal
    equations *square* `cond(A)` (less stable), so the emitted C path delegates to LAPACK's **QR-based `sgels`**
    (better conditioned, ~`cond(A)`) when linked (`-DBCIR_USE_LAPACK -llapack` — `LAPACKE_sgels` rides the
    *existing* `LAPACKE_*`→`-llapack` rule, **no linkflags change**), with the portable normal-equations
    fallback otherwise (CI needs no LAPACK); they agree to float round-off on a **well-conditioned** system. The
    R17 bridge bound is the **input round-trip alone**; the solve is trusted/exact. Verified: recovers a known
    `x` on a consistent overdetermined system (e.g. fits `y = 2x + 1`), is optimal on an inconsistent one, the
    bridge tracks the reference within the R17 bound, the fallback **compiles + runs + recovers** the known `x`
    (the `#ols` runtime probe), and the module **touches no verifier / emits no `Diagnostic`** (off the legality
    path). Pure-Python oracle, deterministic.
  - **E2 — PCA (principal component analysis) BUILT** (`bcir/kbcir/pca.py`, `bcir/tests/test_pca.py`; the
    emitted C twin `bcir/lower/c_kernel.py::emit_lapack_eigh_c`). The second ML-breadth slice, built on the same
    **Area-B "integrate, don't reinvent" wrap pattern** as E1. PCA **generalizes the OLS shape "form a symmetric
    matrix then SOLVE" into "form a symmetric matrix then EIGENDECOMPOSE"**: given data `X` (m samples × n
    features), `pca_reference` **centers** each feature, forms the **symmetric sample covariance** `C =
    (1/(m−ddof))·Xcᵀ·Xc` (ddof=1 default), and **eigendecomposes** `C` for the principal directions
    (eigenvectors) and explained variances (eigenvalues), sorted **descending** (largest variance first). Where
    E1 *reused* the trusted square solve, the symmetric eig is **net-new** (nothing existed to reuse): the
    deterministic **Jacobi rotation** algorithm (`_jacobi_eigh`) is the trusted-eig source of truth — the analog
    of E1 leaning on Gaussian elimination. Two *independent* verifiers (recomputed directly from `C` and the
    eigenpairs, not via the eig path): `eigen_residual` (`max|C·v − λ·v|` — the defining eigen equation, ~0) and
    `orthonormality_residual` (`max|VᵀV − I|` — the components are orthonormal), with a `trace_residual`
    total-variance check (Σλ ≈ trace(C)); `pca_via_bridge` is the Q8↔f32↔Q8 input round-trip then reference.
    **Honest note:** the emitted C path delegates to LAPACK's **`ssyev`** (Householder + implicit-QR /
    divide-and-conquer) when linked (`-DBCIR_USE_LAPACK -llapack` — `LAPACKE_ssyev` rides the *existing*
    `LAPACKE_*`→`-llapack` rule, **no linkflags change**), with the portable **Jacobi** fallback otherwise (CI
    needs no LAPACK); the two differ in *realization* but agree to float round-off on **well-separated**
    eigenvalues (a degenerate/repeated eigenvalue makes the eigenVECTORS non-unique — any orthonormal basis of
    the eigenspace is valid — so tests use **distinct** eigenvalues). A deterministic **sign convention**
    (largest-magnitude entry positive) is applied in both the Python and the C path so they are byte-comparable.
    The R17 bridge bound is the **input round-trip alone**; the eig is trusted/exact. Verified: recovers known
    eigenpairs of a hand-built spectrum and the dominant direction of a spread dataset, the two residuals are
    ~0, eigenvalues descending and Σλ ≈ trace, the fallback **compiles + runs + recovers** `diag(5,3,1)` (the
    `#pca` runtime probe), and the module **touches no verifier / emits no `Diagnostic`** (off the legality
    path). Pure-Python oracle, deterministic.
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
