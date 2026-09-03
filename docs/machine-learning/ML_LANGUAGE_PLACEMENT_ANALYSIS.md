# BCIR — ML Language-Placement Analysis (E7, the capstone)

> **The deliverable the ML-breadth ladder was built to answer.** Across E1 (OLS), E2 (PCA), E3
> (the Transformer block), E4 (RNN/LSTM/GRU), E5 (classical-ML predict), E6 (unsupervised +
> pipeline) — plus the M-trio (losses/optimizers/training), the G-series `gem.*` ops, and the
> Area-B library wraps — one question kept recurring: *which language does each ML/numeric
> component genuinely belong to?* This document answers it concretely: it classifies every
> component into a four-language hierarchy (**Python / C / MLIR / C++**) with the determining
> reason, then gives the migration map. It is a grounded analysis — every placement is checked
> against the actual module before it is asserted (the inventory in §4 cites real files and
> function names). The framing is **not invented here**: E5's `classical.py` and E6's
> `unsupervised.py` docstrings already name "the research finding the E7 capstone cites," and
> this document is that capstone.
>
> Companion to [`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](BCIR_ML_AI_INTEGRATION_ROADMAP.md) (the
> phase ladder), [`VISION_ALIGNMENT_AUDIT.md`](../VISION_ALIGNMENT_AUDIT.md) (the pillar audit),
> [`CPP_HANDOFF_BOUNDARY.md`](../languages/CPP_HANDOFF_BOUNDARY.md) (the C++ boundary), and
> [`PARITY.md`](../PARITY.md) (the dual-rail oracle↔law contract). Counts are in
> [`STATUS.md`](../STATUS.md); prose links there rather than hard-coding numbers.

---

## 1. Executive summary — the thesis

**A component's genuine language is not a matter of taste; it is determined by where it falls on
five orthogonal criteria** that emerged across the E-slices and the substrate. The dominant
pattern is a single split, observed independently in E5 and E6 and named in both docstrings:

> **The train/predict (fit/transform) split**, crossed with the **exact/transcendental** axis
> and the **legality/cost** axis.

Read concretely:

- **The iterative/combinatorial halves stay in Python.** Anything with no fixed dataflow —
  decision-tree induction, the SVM dual quadratic program, K-means' Lloyd iteration, the
  gradient training loop, the autodiff `Tape` itself — has data-dependent control flow,
  convergence loops, and variable-length intermediate structures. It is the oracle / source of
  truth, and it is a *poor fit for a fixed-shape, planned-claim model* (E5's exact words).
- **The fixed-shape PREDICT/INFERENCE/TRANSFORM halves lower to C.** Once a model is a baked set
  of constants (tree thresholds, SVM support vectors, NB mean/var, scaler `(mean,std)`, K-means
  centroids, a trained weight matrix), evaluating it is a deterministic, data-independent-control-flow
  kernel — the **G5 baked-weights pattern** — emitted by `bcir/lower/c_kernel.py` and run on the
  honest C rail.
- **The tensor-op *claims* live on the MLIR law rail.** `gem.matmul` / `activation` / `conv` /
  `attention` / `fused_matmul_activation` / `layout_pivot` / `contention` are first-class planned
  claims with a 12-D CostVector, the dual-semiring K_BCIR search, and `hasVerifier` op-level laws —
  a real compiler IR is the right home for structural/legality reasoning + cost.
- **The legality verdict is multi-railed (Python ↔ C ↔ MLIR).** The current MLIR law rail is
  R1–R25; the C frontend twin implements its explicitly scoped R1–R18 subset, while the Python
  oracle supplies the applicable semantic checks. Crucially, **no ML module touches the verdict**
  — the two-truth quarantine holds (verified in §4.0).
- **The performance/runtime boundary is C++.** Where C's abstractions run out — dynamic graphs,
  distributed orchestration (the G8 hand-off), and the SYCL single-source compiler mode (`-fsycl`)
  — the kernel/dispatcher lowers to C++.

The hierarchy, in one stack:

```
   Python   the ORACLE: source of truth, iterative/combinatorial TRAIN/FIT halves,
            the autodiff Tape, the planners + cost model, the bridges, research organs.
            (Defines correctness; holds the parts with no fixed dataflow.)
      │   freezes to Q8 / bakes constants
      ▼
   C        the dual-rail TWIN + the fixed-shape PREDICT/INFERENCE/TRANSFORM kernels
            (emit_*_c) + the c.call.libm: edge (exp/log/tanh/sqrt) + the Area-B wraps.
            (The honest executable rail.)
      │   (cost/legality reasoning lifted up)
      ▼
   MLIR     the LAW rail: the gem.* tensor-op claims + R1–R25 verifier laws + CostVectors.
            (Structural/legality reasoning + cost search.)
      │   (where C's abstractions run out)
      ▼
   C++      the performance/runtime boundary (G8): the hand-off scaffold + the SYCL backend.
            (Templates / RAII / a runtime / vendor SDKs.)
```

The rest of this document defines the five criteria (§2), the four tiers (§3), the per-component
classification table (§4), and the migration map (§5).

---

## 2. The five placement criteria

Each criterion is one axis. A component's genuine language is read off where it falls on all five.

### Criterion 1 — Legality/decision path vs cost/oracle path (the two-truth quarantine)

The **decision path** — the R1–R25 MLIR law rail, applicable Python checks, and the C frontend's
scoped R1–R18 twin — is deterministic, integer/Q8, and multi-railed: the Python oracle
is the conformance reference where surfaces overlap, the C twin is byte-identical there, and the MLIR pass is the law (see
[`PARITY.md`](../PARITY.md)). Telemetry and learned signals inform plan *cost* (they feed `theta` and
the CostVector) but **never** a legality verdict — a confidence can never become a verdict; a
diagnostic never carries a confidence.

**Consequence for placement:** anything on the legality path must exist as the full dual rail
(Python + C + MLIR). Anything purely on the cost/oracle side has more freedom. Every ML module in
the inventory is provably off the legality path — they import no verifier and emit no `Diagnostic`
(their own docstrings assert it; verified in §4.0). That is *why* they are free to live wherever
the other four criteria put them.

### Criterion 2 — Fixed-shape deterministic kernel vs iterative/combinatorial (the train-vs-predict split)

This is the central E5/E6 finding. A **fixed-shape, data-independent-control-flow** kernel
(PREDICT / INFERENCE / TRANSFORM over baked params) lowers cleanly to C/MLIR. An
**iterative/combinatorial** pass has no fixed dataflow and stays in Python/library. E5's
`bcir/kbcir/classical.py` docstring states it verbatim:

> "Classical ML splits SHARPLY into two halves with OPPOSITE structure: … TRAINING — decision-tree
> INDUCTION …, the SVM dual QUADRATIC-PROGRAM solve, Naive-Bayes FITTING … are ITERATIVE /
> COMBINATORIAL optimization: no fixed dataflow, data-dependent control flow, convergence loops,
> variable-length intermediate structures. That is a POOR fit for BCIR's fixed-shape, planned-claim
> model — it belongs in a LIBRARY / Python …, NOT as a BCIR claim. … PREDICT / INFERENCE — the
> OPPOSITE. Once trained, the model is a FIXED, BAKED set of constants … a DETERMINISTIC,
> FIXED-SHAPE kernel — exactly the G5 baked-weights inference pattern."

E6's `bcir/kbcir/unsupervised.py` carries the same split as **fit vs transform** (K-means' Lloyd
iteration / a scaler's statistical pass are FIT; the nearest-centroid assign / exact scaler
transform are TRANSFORM). This split is the single biggest determinant in the table.

### Criterion 3 — Exact closed-set arithmetic vs transcendental (the closed-set-vs-libm boundary)

The B3 autodiff `Tape` (`bcir/kbcir/autodiff.py`) has a **closed differentiable primitive set**
`{const, var, neg, add, sub, mul, div, dot, select}` — no transcendentals. Because
`relu(x) = select(x, x, 0)` is in the set, relu is exact, re-differentiable, and lowers to C/MLIR
on the deterministic rail with no accuracy contract. But the autodiff docstring's own honest
limitation (boundary (a)) is load-bearing: *"A primitive whose VJP is NOT in the primitive set
(e.g. one needing exp/log/a transcendental) … breaks the closure: the gradient would not be a DAG
in this set and could not be re-differentiated by this machinery."* So **transcendentals
(exp/log/tanh/sqrt) ride the trusted `c.call.libm:` edge** — still C, but the libm kernel is
opaque/trusted (off the deterministic legality rail). This is why E4's recurrent cells and M1's
losses are *two-tier*: relu-RNN / MSE are closed-set (built into the Tape), while LSTM/GRU and
softmax-CE/BCE supply a **closed-form gradient SEED** so the transcendental lives only in the
monitored value, never in the parameter gradient.

### Criterion 4 — Planned tensor claim with a cost model (the `gem.*` ops)

`matmul` / `conv` / `attention` / `activation` / `fusion` / `layout` are first-class **planned
claims**. Their realization is a *free choice* the dual-semiring K_BCIR cost search picks (min,+
over the path cost; max,+ for the roofline bottleneck), and every realization reproduces the same
result. They carry a 12-D CostVector (`bcir/kbcir/cost.py` `DIMS`: compute, memory, fabric, sync,
compile, thermal, power, reliability, security, accuracy, contention, verification) and an op-level
`hasVerifier` law. The MLIR `gem.*` ops in `mlir/include/BCIR/BCIRGEMOps.td` *are* that law rail —
`gem.matmul` "the law-rail record of the B1 plan," `gem.activation` carrying "THE QUARANTINE SPLIT"
(relu clean / transcendentals on the libm edge), `gem.contention` informs-only on the CONTENTION
axis. A real compiler IR + passes is the right home for structural/legality reasoning + cost.

### Criterion 5 — Performance/runtime boundary needing C++ abstractions (the G8 hand-off)

Where C's flat, registry-oriented, freestanding rail runs out of abstractions — OO + virtual
dispatch, the STL, exceptions + RAII, a runtime, vendor SDKs — the kernel lowers to C++. Two
concrete crossings exist: the **C↔C++ hand-off scaffold** ([`CPP_HANDOFF_BOUNDARY.md`](../languages/CPP_HANDOFF_BOUNDARY.md),
`runtime/cpp/`) for dynamic-graph topology and distributed (MPI/NCCL) orchestration; and the
**SYCL backend**, which is a *single-source C++ compiler mode* (`-fsycl`), **not** a `c.call.libm:`
FFI edge — `bcir/kbcir/sycl_saxpy.py` is explicit that "SYCL's dynamic runtime scheduler lives
ABOVE the deterministic C rail (the G8 boundary) and must NEVER touch the legality path." The C++
side may schedule/shard/retry but may never alter a frozen artifact or become an R-law verdict —
the two-truth quarantine extends across the boundary.

---

## 3. The four language tiers

### Python — the ORACLE / source of truth / iterative-or-combinatorial trainers / planning

Python defines correctness and holds everything with no fixed dataflow. It is where every
`*_reference` source of truth lives, where the autodiff `Tape` + `grad`/`grad_graph`/`unroll_scan`
live, where the planners + the 12-D cost model live, and — decisively — where the **FIT/TRAIN
halves** live: tree induction, the SVM dual QP, K-means' Lloyd iteration, the gradient training
loop (`training.py`), and the bridges (`*_via_bridge`). It freezes to Q8 (or bakes constants) to
deploy; it never runs on the deterministic hot path (the L0–L3 placement law).

### C — the dual-rail TWIN + fixed-shape PREDICT kernels + the libm edge + Area-B wraps

C is the honest executable rail. It holds the **verifier C twin** (`runtime/c/bcir_verify.c`, the
dual rail), the **fixed-shape PREDICT/INFERENCE/TRANSFORM kernels** emitted by
`bcir/lower/c_kernel.py` + `autodiff_kernel.py` + `inference.py`, the **`c.call.libm:` edge** for
the single transcendental each kernel needs (exp/log/tanh/sqrt), and the **Area-B library wraps**
(BLAS/LAPACK/FFTW/GSL/SLEEF) where BCIR owns the *calling side* around a trusted external kernel.

### MLIR — the LAW rail: `gem.*` tensor-op claims + R1–R25 verifier laws + CostVectors

MLIR is structural/legality reasoning + cost. It holds the `gem.*` planned-tensor-op claims
(`BCIRGEMOps.td`), the R1–R25 verifier laws (`-bcir-verify`), and the CostVector passes — the law
the Python oracle is gated against (PARITY).

### C++ — the performance/runtime boundary (G8)

C++ holds what needs templates/RAII/a runtime/vendor SDKs above the C rail: the hand-off
orchestrator scaffold (`runtime/cpp/`) and the SYCL backend (`emit_sycl_*` C++ source, `-fsycl`).
It schedules/shards/dispatches; it never computes a verdict and never alters a frozen artifact.

---

## 4. The classification table (the heart)

### 4.0 — The quarantine cross-check (Criterion 1, verified)

Before classifying, the load-bearing fact: **none of the ML/numeric modules touches the legality
path.** A grep across `bcir/kbcir/{ols,pca,transformer,recurrent,classical,unsupervised,losses,
training,matmul,activation,autodiff}.py` for verifier imports / `Diagnostic` finds only docstring
assertions of the *absence* (e.g. `transformer.py`: "it imports NO verifier, emits NO Diagnostic";
`losses.py`: "This module touches no verifier and emits no Diagnostic"). The verifier itself
(`bcir/verify/__init__.py`, `Diagnostic`, `verify()/verify_all()`) is the only place laws live, and
it is dual-railed to `runtime/c/bcir_verify.c` and the MLIR `-bcir-verify` pass. This is *why* the
ML tier is free to be placed by Criteria 2–5: it is all on the cost/oracle side of the quarantine.
(The one `bcir/kbcir/` module that *does* import the verifier is `differential.py` — but that is the
planner's cost-validation oracle, not an ML component: it *calls* `verify()/verify_plan()` to gate a
plan, which is exactly the sanctioned direction. No ML kernel emits a verdict.)

> Columns: **Component** · **Current home** · **Genuine home** · **Criterion** · **Justification.**
> "Current home" is where the code lives today; "genuine home" is where each *piece* belongs (often
> a split). Criterion numbers refer to §2.

### 4.1 — Substrate

| Component (file) | Current home | Genuine home | Crit. | Justification |
|---|---|---|---|---|
| `autodiff.py` — the `Tape`, closed set `{const,var,neg,add,sub,mul,div,dot,select}`, `grad`/`grad_graph`/`unroll_scan`/`hessian` | Python | **Python** (the Tape + reverse-mode rewrites); the *closed-set forward+backward DAG* lowers to **C** via `emit_autodiff_kernel_c` | 2, 3 | The hash-consed rewrite engine is iterative/structural (stays Python); a concrete closed-set DAG is fixed-shape and lowers to a forward+backward C kernel. Transcendental VJPs are out of closure (Crit 3) → must be seeded, not differentiated. |
| `matmul.py` — `matmul_reference`, `TilePlan`, `plan_matmul`, `cost_vector` | Python | **Python** (oracle + planner) → **MLIR** (`gem.matmul` law op) → **C** (`emit_blas_gemm_c`/`emit_tuned_gemm_c` realization) | 4 | A matmul is a planned tensor claim: the realization (tile/loop order) is the dual-semiring cost choice. The plan record is the MLIR law; the executable is C. |
| `quantize.py` — `QGroup`, `scaled_dot`, `quantized_dot`, `compensated_reduce_q8` | Python | **Python** oracle + **C** twin (the Q8 bridge / compensated reduction emit to `emit_compensated_reduce_c`) | 1, 3 | The Q8↔f32↔Q8 bridge is the R17 boundary; exact integer accumulation lowers to C bit-exactly. On the cost/accuracy axis, not legality. |
| `precision.py` — `Interval`, `accuracy_bound`, `meets_tolerance` | Python | **Python** (producer of the accuracy cost axis) ↔ **MLIR** R17 law | 1, 4 | Static ULP bounds feed the CostVector accuracy dimension; R17 is the dual-rail law. `meets_tolerance` is verifier-shaped but opt-in/cost-side. |
| `cost.py` — 12-D `CostVector` (`DIMS`), `TargetProfile`, `MemTier` | Python | **Python** ↔ **MLIR** (12-D parity, PARITY.md) | 4 | The cost algebra is identical in both rails (PARITY pins the dimension order). Pure planning/cost. |
| `semiring.py` — `dag_shortest_path` (min-plus), the dual semiring | Python | **Python** ↔ **MLIR** (K_BCIR `-bcir-rcsp`/`-bcir-plan`) | 4 | Min-plus shortest-path + max-plus roofline is the planner; a deterministic cost search, dual-railed. |
| The verifier (`bcir/verify/`, `verify()`, `Diagnostic`) | Python | **Python ⊕ scoped C ⊕ MLIR** | 1 | The legality verdict: MLIR carries R1–R25, the C frontend carries R1–R18, and Python checks each applicable oracle surface. |

### 4.2 — G-series `gem.*` ops (the planned tensor claims)

All of these have a Python oracle/planner (`bcir/kbcir/*.py`), a C emitter
(`bcir/lower/c_kernel.py`), and a built MLIR law op (`mlir/include/BCIR/BCIRGEMOps.td`). Criterion 4
places the *claim* on MLIR; Criterion 3 places the *kernel* on C (exact) or the libm edge
(transcendental).

| Component | Current home | Genuine home | Crit. | Justification |
|---|---|---|---|---|
| `matmul.py` → `gem.matmul` | Py + C + MLIR | claim → **MLIR**; kernel → **C** | 4 | K_BCIR tile/loop plan is the MLIR law; exact GEMM arithmetic lowers to C. |
| `activation.py` → `gem.activation` (relu/sigmoid/tanh/gelu/softmax) | Py + C + MLIR | claim → **MLIR**; **relu → C exact**, **sigmoid/tanh/gelu/softmax → C `c.call.libm:` edge (expf/tanhf)** | 3, 4 | The op's own "QUARANTINE SPLIT": relu is 0-ULP clean (may be i32/f32); transcendentals route libm → need f32. |
| `conv.py` → `gem.conv` | Py + C + MLIR | claim → **MLIR**; kernel → **C** (`emit_conv2d_c`) | 4 | A conv is a *structured matmul* (im2col); priced through the matmul roofline, direct-vs-im2col is the cost choice. |
| `attention.py` → `gem.attention` (single-head scaled-dot-product) | Py + C + MLIR | claim → **MLIR**; kernel → **C** (`emit_attention_c`), **softmax `exp` → libm edge** | 3, 4 | Decomposes into two `gem.matmul`s + a `gem.activation` softmax; the matmuls are exact, the softmax rides libm. (Multi-head/batched/masked is the E3 composition, below.) |
| `fusion.py` → `gem.fused_matmul_activation` | Py + C + MLIR | **MLIR** (the deforestation-priced fusion decision) + **C** (`emit_matmul_activation_c`) | 4 | Sole-consumer epilogue fusion is the optimizer's priced choice; the op carries the `FusionCertificate` (unfused/fused/gain). Softmax is non-fusible (row reduction) and rejected. |
| `layout.py` → `gem.layout_pivot` (SoA↔AoS) | Py + C + MLIR | **MLIR** (the priced layout choice) + **C** (`layout_kernel`, address-invariant) | 4 | Layout changes addresses, never values (bit-exact); which layout wins is priced via the stride-penalty terms, informs-only. |
| `inference.py` — `emit_inference_kernel_c` (G5 baked weights) | Python emitter → C | **C** (the baked-weights fixed-shape kernel) | 2 | The canonical fixed-shape PREDICT pattern: weights baked as `static const` (`#embed`/literal), fused single-pass, R17-bounded. The whole E5/E6 PREDICT tier inherits this. |

> Honest note on the `gem.*` Python files: their docstrings call the MLIR wiring a "separate
> follow-up," but the MLIR law ops themselves are *built* — `BCIRGEMOps.td` defines `gem.matmul`,
> `gem.activation`, `gem.conv`, `gem.attention`, `gem.fused_matmul_activation`, `gem.layout_pivot`,
> `gem.contention`, each `hasVerifier` — and [`VISION_ALIGNMENT_AUDIT.md`](../VISION_ALIGNMENT_AUDIT.md)
> (pillars 3a/3b/3c/5c) records them as "ported to the MLIR law rail." So the claim genuinely lives
> on MLIR today; the Python file is its oracle prototype.

### 4.3 — Area-B library wraps (the `c.call.libm:` edge)

BCIR owns the calling side (row-major layout, Q8 boundary, in-place contract) around a trusted
external kernel; a portable reference fallback keeps CI free of the library. All emit through
`bcir/lower/c_kernel.py`.

| Component (file) | Current home | Genuine home | Crit. | Justification |
|---|---|---|---|---|
| `linsolve.py` — `solve_reference` (dense LU, partial pivot) → LAPACK `sgesv` | Python + C | oracle → **Python**; kernel → **C** (`emit_lapack_solve_c`, delegates to `LAPACKE_sgesv`) | 2, 1 | Fixed-shape solve; LAPACK is the hardened realization, the Gaussian-elimination reference is the portable twin. The solve is exact arithmetic (no transcendental). |
| `gsl_kernels.py` — `stats_reference` (mean/variance/sd) → GSL | Python + C | oracle → **Python**; kernel → **C** (`emit_gsl_stats_c`); **sd's `sqrt` → libm** | 3 | Fixed-shape statistic; `sd` is the only transcendental (sqrt on the libm edge). |
| The SLEEF / FFTW / BLAS emitters (`emit_sleef_exp_c`, `emit_fftw_fft_c`, `emit_blas_gemm_c`) + linkflags | Python emitter → C | **C** (the `c.call.libm:` edge + `-l<lib>` link rule) | 3 | The five Area-B wraps (BLAS/FFTW/LAPACK/GSL/SLEEF) all route through the external-symbol edge; SLEEF's vectorized `exp` is the canonical transcendental wrap. |

### 4.4 — M-trio (losses / optimizers / training)

| Component | Current home | Genuine home | Crit. | Justification |
|---|---|---|---|---|
| `losses.py` — `mse` (closed-set into Tape) | Python | **Python** oracle → **C** (lowered by `emit_autodiff_kernel_c` for free, as closed-set nodes) | 3 | `MSE = (1/n)·dot(e,e)` is sub+dot+scale, all in the closed set → the existing autodiff differentiates/lowers it. |
| `losses.py` — `softmax_cross_entropy`, `binary_cross_entropy_with_logits`, `hinge` (transcendental, closed-form grad seed) | Python | forward value → **Python/C libm** (`exp`/`log`); **gradient → closed-set seed** | 3 | The forward needs log/exp (libm), so it cannot be a re-differentiable Tape node; the closed-form `grad_logits` (`softmax−onehot` / `sigmoid−target`) seeds the closed-set backward — the transcendental never enters the parameter gradient. |
| `lower/optimizers.py` — `sgd_step`, `momentum_step` (pure arithmetic) | Python + C | **C** (`emit_sgd_step_c` in `autodiff_kernel.py` — the pre-existing G6 step; `emit_momentum_step_c` in `optimizers.py`; no `-lm`) | 3 | One step is a fixed-shape in-place update over the param vector; SGD/momentum are pure arithmetic (`<stddef.h>` only). |
| `lower/optimizers.py` — `rmsprop_step`, `adam_step` (sqrt) | Python + C | **C** (`emit_rmsprop_step_c`, `emit_adam_step_c`) **on the libm edge** (`sqrtf`, `-lm`) | 3 | The `√s`/`√v̂` divisor is the one transcendental → `#include <math.h>`, `sqrtf`, link `-lm`; the bias-correction `t`-divisors stay arithmetic. |
| `training.py` — the epoch/mini-batch loop: `train(...)`, `Dataset`, `minibatches`, `_lcg_permutation`, `EarlyStop`, metrics | Python | **Python** (must stay) | 2 | The training loop is the iterative half: a convergence loop over epochs/mini-batches with a seed-keyed shuffle and early-stop — no fixed dataflow. It composes M1+M2+autodiff; the *steps* it drives are C, the *loop* is Python. |

### 4.5 — E1–E6 (the ML-breadth ladder) and the emitted kernels

| Component | Current home | Genuine home | Crit. | Justification |
|---|---|---|---|---|
| **E1 OLS** `ols.py` — `ols_reference` (normal equations, reuses `solve_reference`) + `emit_lapack_ols_c` | Python + C | oracle → **Python**; kernel → **C** (delegates to LAPACK QR `sgels`, portable normal-equations fallback) | 2 | Overdetermined regression is a fixed-shape solve; the C path uses the better-conditioned QR. Exact arithmetic; R17 bridge = input round-trip only. |
| **E2 PCA** `pca.py` — `pca_reference`, `_jacobi_eigh` + `emit_lapack_eigh_c` | Python + C | oracle (+ the net-new Jacobi eig) → **Python**; kernel → **C** (delegates to LAPACK `ssyev`, portable Jacobi fallback) | 2 | "Form a symmetric matrix then eigendecompose" is fixed-shape; the iterative Jacobi *fit* stays Python, the eig *result* is a C kernel. No transcendental (covariance + Jacobi rotations are exact). |
| **E3 Transformer block** `transformer.py` — `transformer_block_reference`, `multihead_attention_reference`, `layernorm_reference`, `feedforward_reference` + `emit_layernorm_c` | Python + C | composition → reuses **MLIR** `gem.*` claims; net-new **`layernorm` → C** (`emit_layernorm_c`, **`sqrt` → libm**) | 2, 3, 4 | A block is a COMPOSITION of existing claims (matmul/softmax/attention); the only net-new numeric primitive is LayerNorm, whose `sqrt` rides the libm edge. The matmuls are exact. |
| **E4 Tier A relu-RNN** `recurrent.py` — `rnn_relu_step`/`rnn_relu_unroll` (closed-set, BPTT via `unroll_scan`+`grad`) | Python + C | closed-set forward+BPTT → **Python** Tape → lowers to **C** | 3 | Built from only closed primitives (relu = `select`); the unrolled DAG's `grad` IS backprop-through-time. Re-differentiable, exact. |
| **E4 Tier B LSTM/GRU** `recurrent.py` — `lstm_cell_reference`/`gru_cell_reference` + `lstm_cell_grads`/`gru_cell_grads` + `emit_lstm_cell_c` | Python + C | forward → **C** libm edge (`tanhf`/`expf` guarded sigmoid); **gradient → closed-form seed** (Python) | 3 | `tanh`/`sigmoid` are out of closure → the cell is the M1 seed treatment: numeric forward on the libm edge + closed-form analytic gradients (`σ'=σ(1−σ)`, `tanh'=1−tanh²`). |
| **E5 classical TRAIN** — tree induction, SVM dual QP, NB fit, KNN "fit" | (not built — library) | **Python / library** (scikit-learn, libsvm) | 2 | The iterative/combinatorial half: no fixed dataflow, data-dependent control flow, convergence loops. *BCIR does not try to own it* (E5's words). |
| **E5 classical PREDICT** `classical.py` — `knn_classify`/`knn_regress`, `tree_predict`, `svm_decision_linear`/`svm_decision_rbf`, `nb_predict` + `emit_svm_rbf_predict_c` + `emit_tree_predict_c` | Python + C | **C** (baked-model fixed-shape kernels): KNN/tree/linear-SVM **exact**; **RBF-SVM `exp` → libm**; Gaussian-NB **`log` baked at bake time** | 2, 3 | PREDICT over baked constants = the G5 pattern. KNN ranks on *squared* distance (sqrt monotone → no transcendental); tree is exact comparisons; the NB `log(2π·σ²)` normaliser is data-independent → precomputed, so runtime logs nothing. |
| **E6 unsupervised FIT** `unsupervised.py` — `kmeans_fit` (Lloyd), `standard_scaler_fit`/`minmax_fit` | Python | **Python** | 2 | The fit/train half: Lloyd's iteration is a bounded convergence loop; a scaler's statistical pass reduces over the whole set. `StandardScaler`'s `sqrt` is a *fit-time* transcendental that bakes into the std. |
| **E6 unsupervised PREDICT/TRANSFORM** `unsupervised.py` — `kmeans_assign`, `standard_scaler_transform`/`minmax_transform`, `autoencoder_forward`, `embedding_lookup` + `emit_kmeans_assign_c` | Python + C | **C** (baked fixed-shape kernels) | 2, 3 | `kmeans_assign` reuses the exact squared distance (no transcendental; `emit_kmeans_assign_c` returns an `int`, no `-lm`); scaler transform is exact division (the sqrt was at fit); the autoencoder forward reuses matmul+activation; embedding is an exact row gather. |
| **E6 CV folds** `unsupervised.py` — `k_fold_indices` (reuses `training._lcg_permutation`) | Python | **Python** | 2 | A deterministic index partition built on the training LCG shuffle — a planning/data utility, no kernel. |

### 4.6 — The C++ boundary (G8)

| Component | Current home | Genuine home | Crit. | Justification |
|---|---|---|---|---|
| The hand-off orchestrator (`runtime/cpp/bcir_orchestrator.{hpp,cpp}`) | C++ scaffold | **C++** | 5 | Dynamic-graph topology + distributed (MPI/NCCL) orchestration need OO/virtual dispatch, the STL, exceptions+RAII — the abstractions the flat C rail deliberately lacks. It consumes a frozen artifact, never alters it ([`CPP_HANDOFF_BOUNDARY.md`](../languages/CPP_HANDOFF_BOUNDARY.md)). |
| The SYCL backend — `sycl_saxpy.py`/`sycl_reduce.py` (the oracle modules) + `emit_sycl_*_c` (saxpy/reduce/matmul) + `lower/sycl_dispatch.py` | Python oracle + C++ emit | oracle → **Python**; kernel → **C++** (`-fsycl` single-source, `parallel_for`); dispatcher → **C++** above G8 | 5 | SYCL is a C++ *compiler mode*, **not** a `c.call.libm:` edge (no link-flag rule). Its dynamic runtime scheduler lives above the deterministic C rail and must never touch the legality path. Portable scalar fallback does the real work on CI. |

---

## 5. The migration map

This section is explicit and honest: what *could* migrate down (and the gate), what *must* stay,
and what is already correctly placed. "Down" means Python → C/MLIR/C++ (toward the executable/law
rails).

### 5.1 — Already correctly placed (no migration owed)

- **The `gem.*` tensor-op claims** are already tri-railed (Python oracle + C emitter + MLIR law op
  in `BCIRGEMOps.td`). Criterion 4 is satisfied: the claim is on MLIR, the kernel on C.
- **The verifier** is already the full dual/tri rail (Python ↔ `bcir_verify.c` ↔ `-bcir-verify`),
  as Criterion 1 demands.
- **The Q8 bridge / R17** is dual-railed (`quantize.py`/`precision.py` ↔ MLIR R17 law), with the
  compensated reduction bit-exact to the C twin.
- **Every E1–E6 PREDICT kernel** already has its C twin emitter (`emit_lapack_ols_c`,
  `emit_lapack_eigh_c`, `emit_layernorm_c`, `emit_lstm_cell_c`, `emit_svm_rbf_predict_c`,
  `emit_tree_predict_c`, `emit_kmeans_assign_c`) and is reference-verified + runtime-probed.

### 5.2 — Could migrate down — and the gate for each

- **The `gem.*` Python planners → MLIR cost passes.** The MLIR CostVector passes
  (`-bcir-gem-{matmul,activation,conv,attention}-cost`) are **built**; what remains is the policy
  decision to *subsume* the Python `plan_matmul`/`plan_activation`/… oracles entirely. **Gate:** the
  oracle is kept as the conformance reference (PARITY discipline) — it migrates only when the MLIR
  pass is proven byte-identical and the oracle's role as the cheap-iteration prototype is no longer
  needed. Today the split (oracle prototype + law op) is deliberate, not debt.
- **The autodiff *closed-set* DAG → C/MLIR. ✅ landed.** A concrete forward+backward DAG lowers to C
  (`emit_autodiff_kernel_c`) **and** the MLIR `gem.autodiff` law op exists (`BCIRGEMOps.td`, carrying
  the serialized closed-set DAG). **Gate (stands for any extension):** Criterion 3 —
  this is sound *only* for the closed primitive set; the moment a VJP needs a transcendental, closure
  breaks and the seed treatment (M1/E4 Tier B) is required. So the lowered surface covers the
  closed-set core only.
- **The Area-B `*_reference` fallbacks → broader library breadth.** LAPACK/GSL/FFTW/SLEEF wraps are
  in; ATLAS/OpenBLAS and additional kernels are the remaining Area-B breadth (the active frontier per
  the roadmap §6 and the audit backlog). **Gate:** each new wrap needs the R17 bridge at the seam + a
  portable fallback + a parity gate — pure breadth, not new capability.
- **The SYCL device path → resident driver.** The `-fsycl` path self-skips on CI (no device); the
  portable scalar fallback does the real work. **Gate:** a real heterogeneous device plus a resident
  driver under the master roadmap's §4.3/§4.4 rails before the device path is load-bearing. Stays
  above G8 regardless.
- **The C++ dynamic/distributed backends.** The hand-off contract + single-node seam are real; the
  `DynamicGraphOrchestrator`/`DistributedOrchestrator` dispatch paths are documented STUBS. **Gate:**
  multi-node hardware + an MPI/NCCL dependency (deliberately not added). The *sharding logic* is
  already real; only cross-node dispatch+reduce is owed.

### 5.3 — Must stay in Python (the oracle + the iterative/combinatorial trainers)

These are not "not yet migrated" — they are placed correctly and **cannot** move down, by
Criterion 2 or 3:

- **The autodiff `Tape` engine** (hash-consing, the local backward rewrite rules,
  `grad_graph`/`hessian`). It is a structural graph-rewrite engine, not a fixed-shape kernel.
- **The gradient training loop** (`training.py`): epochs, mini-batches, the seed-keyed shuffle,
  early stop — a convergence loop with no fixed dataflow. The *steps* it drives are C; the *loop* is
  Python.
- **Every TRAIN/FIT half**: decision-tree induction, the SVM dual QP, Naive-Bayes fitting, K-means'
  Lloyd iteration. E5 states it directly — these are "a POOR fit for BCIR's fixed-shape,
  planned-claim model … it belongs in a LIBRARY / Python, NOT as a BCIR claim. BCIR does not try to
  own it." (K-means' Lloyd `kmeans_fit` is bounded+deterministic so it *can* live in the oracle, but
  its iterative nature is exactly the fit/predict seam — the *assign* kernel is what lowers to C.)
- **Every `*_reference` source of truth and `*_via_bridge`**: by the prototype-then-port discipline,
  the oracle defines correctness and *freezes*; it is the conformance reference the C/MLIR rails are
  gated against. It stays by design.
- **The transcendental *forward* values** (softmax-CE/BCE loss values, LSTM/GRU gate forwards): they
  ride the libm edge when emitted, but their re-differentiation is impossible in the closed set
  (Criterion 3), so the gradient is always the closed-form seed computed in Python.

---

## 6. Conclusion — the clean hierarchy

The language an ML/numeric component genuinely belongs to is determined by the five criteria, and
the answer is a clean four-tier hierarchy:

- **Python** — the oracle and source of truth; the autodiff `Tape` and the reverse-mode rewrite
  engine; the planners + the 12-D cost model; **the iterative/combinatorial TRAIN/FIT halves**
  (tree induction, the SVM dual QP, K-means' Lloyd iteration, the gradient training loop); every
  `*_reference` and `*_via_bridge` bridge. *It defines correctness and holds the parts with no fixed
  dataflow; it freezes to Q8 / bakes constants to deploy.*

- **C** — the dual-rail verifier twin (`bcir_verify.c`); **the fixed-shape PREDICT/INFERENCE/TRANSFORM
  kernels** (the `emit_*_c` family, the G5 baked-weights pattern); the `c.call.libm:` edge for the
  single transcendental each kernel needs (exp/log/tanh/sqrt); the five Area-B library wraps
  (BLAS/LAPACK/FFTW/GSL/SLEEF). *The honest executable rail.*

- **MLIR** — the law rail: the `gem.*` planned tensor-op claims (matmul/activation/conv/attention/
  fusion/layout/contention), the R1–R25 verifier laws, and the CostVectors. *Structural/legality
  reasoning + the dual-semiring cost search.*

- **C++** — the performance/runtime boundary (G8): the dynamic-graph + distributed hand-off
  orchestrator scaffold, and the SYCL single-source (`-fsycl`) backend. *Templates / RAII / a
  runtime / vendor SDKs, above the rail — it schedules and dispatches but never computes a verdict or
  alters a frozen artifact.*

The dominant pattern, stated once: **train stays up (Python), predict lowers down (C); exact lowers
to the deterministic rail, transcendental rides the trusted libm edge; the tensor-op claim is the
MLIR law, the legality verdict is the dual rail, and the dynamic/distributed runtime is C++.** Every
ML module sits on the cost/oracle side of the two-truth quarantine — which is precisely what frees
each piece to live where these criteria put it.
