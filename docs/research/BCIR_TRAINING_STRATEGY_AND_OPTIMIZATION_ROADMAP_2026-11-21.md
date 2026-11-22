# Training Strategy, Kernel Systems, and Weight-Space Optimization Roadmap (2026-11-21)

> Dated 2026-11-21, non-normative engineering roadmap. Captures concrete training-strategy,
> numerical-hygiene, native-C kernel/memory, and weight-space engineering improvements directly
> exposed by auditing all six model-training archives in
> [`Mecca-Research/Modular-AI`](https://github.com/Mecca-Research/Modular-AI)
> (`BCIR Model Training/{gp-global, wedge-next, wedge-integrated, wedge-query-qware, wedge-pr818, BCIR-50M-Study}`)
> against BCIR's production native C11 trainer ([`bcir_decoder_train.h`](../../runtime/c/bcir_decoder_train.h),
> [`bcir_tensor.h`](../../runtime/c/bcir_tensor.h), ABI 5).
>
> Companion architecture roadmap:
> [BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md](BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md).

Every slice below cites the exact measurement or raw-log anomaly that motivates it, defines its
numerical/provenance contract, and specifies a falsifiable acceptance gate.

---

## Part A. Training Strategy & Optimization Hygiene (`TRAIN-OPT-1` – `TRAIN-OPT-4`)

### `TRAIN-OPT-1`: Fix 99.78% Gradient-Clip Saturation via $\mu$P / Depth-Scaled Initialization & Per-Scale LR

**Observed defect (`BCIR-50M-Study/results/train-*.json`):**
Auditing every per-update event record across all 10 runs (`6,000` updates) reveals that **5,987 of
6,000 updates (99.78%)** hit the `grad_clip = 1.0` ceiling — including **600/600 (100%)** of every
update in `reference10m`, `wide10m`, `wide50m`, and `deep50m` (step-1 norms `10.36–18.25`, medians
`1.50–1.97`, last-50-step means `1.41–1.68`; `pilot1m` spiked to `30.42` at step 50). Two root
causes interact:
1. [`NativeDecoder.__init__`](../../bcir/hosted/models/native.py) initializes every parameter
   tensor with uniform $\mathcal{N}(0, 0.02^2)$, without residual-branch depth scaling
   ($1/\sqrt{2L}$ on `o_proj` and `down_proj`) or width scaling ($1/\sqrt{d}$). (A related omission
   appeared in `gp-global/experiment.py`, where geometric `gate_g` skipped $1/\sqrt{2L}$ scaling.)
2. All five scales ($d \in [192, 768]$, $L \in [2, 12]$) shared a single unscaled `lr = 6e-4` and
   `grad_clip = 1.0`, forcing AdamW to run almost entirely on direction-only normalized gradients
   and distorting effective step sizes across width and depth.

**Proposed engineering change:**
- Add an explicit versioned initialization policy (`init_policy = "legacy_002" | "residual_scaled" | "mup_v1"`)
  carried in the checkpoint metadata and training recipe (preserving `"legacy_002"` for exact
  regression against existing checkpoints):
  - `"residual_scaled"`: $\sigma = 1/\sqrt{d_{\text{in}}}$ for hidden projections, scaled by
    $1/\sqrt{2L}$ on residual output projections (`o_proj`, `down_proj`), embeddings at $\sigma = 0.02$.
  - `"mup_v1"` (Maximal Update Parameterization): width-scaled initialization ($\sigma \propto 1/\sqrt{d_{\text{in}}}$)
    and per-tensor LR multipliers ($\eta_{\text{hidden}} = \eta_0 \cdot (d_{\text{base}} / d)$,
    $\eta_{\text{embed/norm}} = \eta_0$) so hyperparameters tuned at `pilot1m` transfer directly to
    `wide10m` and `wide50m`.
- Add a mandatory `grad_clip_fraction` telemetry metric to every training summary JSON.

**Acceptance gate:** On paired `pilot1m` → `wide10m` → `wide50m` runs (2+ seeds), post-warmup
(step > 50) gradient-clip saturation drops from **99.8% to `< 5%`**, with strictly lower held-out
CE at matched 600-step and 6,000-step budgets and zero NaN/Inf events.

---

### `TRAIN-OPT-2`: Upstream Physical Causal Suffix Trimming & Exact Masked-Target Loss

**Observed win (`wedge-integrated/results/summary.json`):**
Physically executing only the active causal prefix `x[:, :T_active]` on variable-length batches
eliminated **41.67%** of physical positions (`921,600 → 537,600`) and delivered a **1.603×**
training speedup (`22.05 → 13.75 s`) with zero recall loss (`94.01%` at $T=1024$, `64.58%` at
$T=2048$). Production [`bcir_decoder_train.c`](../../runtime/c/bcir_decoder_train.c) currently
requires dense targets strictly in `[0, vocab)` and a single fixed `time` per plan.

**Proposed engineering change:**
- Add `BCIR_DT_TARGET_IGNORE = UINT32_MAX` support to `bcir_decoder_loss_backward` behind an
  explicit contract flag, refusing all-masked microbatches with `BCIR_DT_INVALID` before mutating
  state (preserving `wedge-integrated`'s fail-closed contract).
- Support **dynamic prefix-length binding (`active_time <= plan.spec.time`)** inside
  `bcir_decoder_run_accum` and `bcir_decoder_run_program` using the single pre-allocated max-shape
  workspace (zero reallocation), admitted **only** when the verifier certifies that every layer in
  the plan is strictly causal and free of cross-position RNG dependencies.
- Add **global-denominator microbatch loss normalization** (`sum(loss_m) / sum(supervised_m)`) so
  mixed-length / packed microbatches with unequal supervised token counts accumulate exact gradients
  instead of forcing equal supervised counts per microbatch.
- Document the exact `~3.6e-8` backward gradient shift under `BCIR_DT_CONTRACT_KBLOCK` (caused by
  K-block row re-partitioning `[448,288,288]` vs `[256,256]` in `bcir_tensor.c`) as a named
  non-bitwise execution equivalence, separate from bitwise-identical transformations.

**Acceptance gate:** Forward logits and loss match padded execution with `0.0` error; normalized
gradients match independent PyTorch AD to `< 1e-6` rel-L2 across unequal-supervision microbatches;
variable-length ordered recall reproduces `>= 1.5×` wall-time speedup on production C11.

---

### `TRAIN-OPT-3`: Multi-Stage Curricula & Compute-Proportional Token Budgets

**Observed gaps (`BCIR-50M-Study` §3, `wedge-next` §6, `wedge-query-qware` §4):**
- `BCIR-50M-Study` trained `wide50m` on only **153,600 bytes** (`0.00304` tokens/parameter — ~6,500×
  below compute-optimal) at `T = 64` (~12 words), causing `wide10m` to tie `wide50m` at `4.75×`
  lower compute.
- `wedge-integrated` / `wedge-query-qware` jumped directly from 2 pairs to 4 pairs at step 300/900,
  leaving even `attention32` stuck at `1/k` candidate-value guessing (`23.5%` on 4 pairs, `5.08%`
  swap).
- `wedge-integrated` §6 showed that a short 40-step long-context tail (`T=1024`) improved $T=2048$
  recall (`63.0% → 91.2%`), while 40 extra `T=128` steps already yielded `86.5%` at `1/8` the cost.

**Proposed engineering change:**
- Establish a **three-tier standard training budget protocol** for all future `Modular-AI` studies:
  1. *Tier S (diagnostic, 526k)*: `2,000–4,000` updates (`1M–4M` tokens), progressive `1 → 2 → 4 → 8`-pair
     KV curriculum, balanced variable lengths `{64, 128, 256}` + optional 10% `T=512/1024` tail.
  2. *Tier M (10M scaling)*: **$\ge 20\text{M}$ tokens** (`~2` tok/param) at $T \in \{256, 512\}$,
     comparing both **raw UTF-8 bytes** (and Arch-5 byte patches) and **BCIR BPE (`bcir.byte_bpe`)**.
  3. *Tier L (50M scaling)*: **$\ge 50\text{M}$–$100\text{M}$ tokens** (`1–2` tok/param) comparing
     candidates at **both equal tokens and equal wall-clock time** (exercising `ATT-CONTIG` at
     $T \ge 1024$ and `KSPLIT` weight-adjoint depth splitting!).

---

### `TRAIN-OPT-4`: Eliminate 15.6s Python-Loop Weight Initialization & Peak-RSS Doubling

**Observed defects (`BCIR-50M-Study/results/benchmark.json`, `results/weights-wide50m-*.json`):**
1. [`NativeDecoder.__init__`](../../bcir/hosted/models/native.py) populates `weights` via a pure
   Python scalar loop (`rng.gauss(0.0, 0.02)`), taking **15.62 s** on `wide50m` (`14.97 s` on
   `deep50m`, `3.02 s` on `reference10m`) — equal to **27 full 50M training steps** — even when
   the caller immediately restores a saved checkpoint!
2. During `weight_space.py` evaluation, cloning full 50M weights plus buffering full test logit
   tensors (`64 × 4 × 64 × 260 × 4 B`) doubled process peak RSS from **`1,061 MiB` → `2,181 MiB`**.

**Proposed engineering change:**
- Add `init="skip"` and a validated `NativeDecoder.from_checkpoint(path, ...)` constructor that
  allocates zeroed/uninitialized native arenas and loads verified `.npz` arrays directly into the
  C buffer (while preserving bit-exact legacy Python RNG initialization under `init="legacy"`).
- Add a deterministic C11 Philox/SplitMix64 Box-Muller initializer (`bcir_decoder_init_weights`)
  for new `"residual_scaled"` / `"mup_v1"` recipes — cutting fresh 50M initialization from
  **15.6 s to `< 50 ms`** (>300× speedup) with exact cross-platform reproducibility.
- Stream per-batch KL / top-1 / CE reference comparisons in `weight_space` evaluation harnesses
  instead of materializing whole-test-set logit arrays in RAM.

---

## Part B. Native C Kernel, Memory & Hardware-Efficiency Optimizations (`TRAIN-OPT-5` – `TRAIN-OPT-10`)

### `TRAIN-OPT-5`: Online-Softmax IO-Aware Tiled Training Attention (`BCIR-FlashAttn-CPU`)

**Observed gap (flagged in all 5 `wedge-*` reports & `BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md`):**
BCIR's `INFER-ATTN` (`BCIR_TENSOR_ATTENTION_SCRATCH` in [`bcir_tensor.h`](../../runtime/c/bcir_tensor.h))
already runs inference attention in cache-resident row strips without storing the $B H T^2$
probability matrix. During **training backward**, however, `bcir_decoder_train.c` still materializes
the full $B H T^2$ probability tensor `p` (`78.5 MiB` at 526k B1/T1024; `322 MiB` at 526k B4/T1024;
`6.0 GiB` at 50M B4/T2048). Every comparison between linear/wedge mixers and native attention is
currently confounded by this missing baseline.

**Proposed engineering change:**
- Add `BCIR_TENSOR_ATTENTION_TILED` (and `BCIR_DT_CONTRACT_ATTN_TILED`):
  - **Forward**: tile Q in blocks of $B_r = 32$ and K/V in blocks of $B_c = 64$ using online safe
    softmax (FlashAttention-2 rescaling order, strictly ascending key blocks so every output element
    is deterministic across thread counts!). Store ONLY the $O(B H T)$ log-sum-exp vector `lse`
    (FP64 or FP32) instead of the $O(B H T^2)$ probability matrix `p`.
  - **Backward**: recompute each $(B_r, B_c)$ score/probability tile in L1/L2 workspace using `lse`
    and $D_i = \sum_d dY_{i,d} O_{i,d}$, accumulating `dQ`, `dK`, `dV` in deterministic block order
    using BCIR's `ATTN-FMA` or `BF16-ATTN` tile kernels.
- **Expected impact:** Cuts 526k `attention` B1/T1024 arena from **`78.53 MiB` → `~18–20 MiB`**
  (eliminating the 32 MiB accumulation confound on `attention32`!) and cuts `wide50m` B4/T2048
  attention probability storage from **`6.00 GiB` → `3.0 MiB`** (`lse` only — **2,000×** smaller!).

**Acceptance gate:** Forward/backward agrees with reference FP64 attention to `< 1e-6` rel-L2, is
bitwise identical across 1..N threads, fits `attention` B4/T256 and B4/T512 inside 32 MiB without
microbatch accumulation, and matches or beats stored `P-BANDS` speed for $T \ge 512$.

---

### `TRAIN-OPT-6`: Native `AVX512_BF16` (`VDPBF16PS`) GEMM Contract for Non-AMX Hosts

**Observed defect (`wedge-pr818/results/paired-timing.json`, `vendor-new/bcir_tensor.c:1312-1352`):**
On AMD EPYC 9V74 (Zen 4, which has `AVX512_BF16` hardware instructions but no Intel AMX),
`BCIR_DT_CONTRACT_BF16` (contract 26) runs **0.877× slower** than FP32 `kblock` on `query_full`
(`30.95 vs 28.48 ms`) and **0.960×** on `attention`. That happens because `NDT-BF16` is pinned
bit-for-bit to Intel AMX `TDPBF16PS` (separate even/odd accumulators + MXCSR DAZ/FTZ flush), forcing
non-AMX CPUs onto FP32 FMA emulation instead of hardware `VDPBF16PS`.

**Proposed engineering change:**
- Keep `BCIR_DT_CONTRACT_BF16` unchanged for bit-exact Intel AMX reproducibility.
- Introduce an explicitly distinct, named arithmetic contract **`BCIR_DT_CONTRACT_BF16_VDP`**:
  - Defined mathematically by the IEEE-754 pair-dot accumulation semantics of x86 `VDPBF16PS` /
    ARMv8.6 `BFMMLA` (paired BF16-to-FP32 dot product summed directly onto the accumulator in
    32-depth blocks, with a bit-identical portable C reference!).
  - Unlocks native 2× BF16 vector throughput on AMD EPYC (Zen 4/5) and Intel Cooper/Sapphire/Granite
    Rapids non-AMX paths without silently breaking `NDT-BF16` checkpoint identity.

**Acceptance gate:** Bitwise identity between portable C and `VDPBF16PS` AVX-512-BF16 instances
across edge-case fuzzing; paired AB/BA speedup **$> 1.35\times$** over FP32 `kblock` on AMD EPYC
9V74 with matched held-out language and recall quality.

---

### `TRAIN-OPT-7`: Persistent Optimizer-State Compression (`16 B/param → 6–8 B/param`)

**Observed bottleneck (`BCIR-50M-Study` §3, `results/benchmark.json`):**
At `wide50m` (`50.54M` params), `771.25 MiB` out of `877.52 MiB` (**87.9%**) is the four persistent
FP32 arrays (`weights`, `gradients`, `moment1`, `moment2` = `16 B/param`). Activation recomputation
(`REMAT`) can only reclaim a fraction of the remaining `106.27 MiB`. On an 8 GiB CPU/edge host,
scaling to 150M–300M parameters is blocked entirely by persistent AdamW state, not activations.

**Proposed engineering change (two selectable, named optimizer contracts):**
1. **Blockwise Q8 AdamW Moments (`BCIR_DT_OPT_Q8_MOMENTS`)**: Keep FP32 master weights (`4 B/p`)
   and FP32/BF16 gradients (`2–4 B/p`), but store `moment1` (signed int8) and `moment2` (unsigned
   int8 / dynamic-exponent log-scale) in fixed 256-element blocks with FP32 block scales — dropping
   persistent state from **`16 B/param` → `8.06 B/param`** (or `6.06 B/param` with BF16 gradient
   buffer), saving **~385–480 MiB** at 50M and enabling **~150M–200M** parameter training inside a
   2 GiB native buffer budget!
2. **Row/Column-Factored Second Moment (`BCIR_DT_OPT_ADAFACTOR`)**: Factor $m_2 \in \mathbb{R}^{m\times n}$
   into row/column vectors $r \in \mathbb{R}^m, c \in \mathbb{R}^n$ (`~8.02 B/param` with FP32
   weights + grads + `m1`, zero quantization noise, trivial deterministic C11 reduction).

**Acceptance gate:** Matched 3-seed training on `wide10m` and `wide50m` within `<= 0.005` nats of
FP32 AdamW at equal token budget, bit-exact across thread counts and checkpoint resume, and
verified `<= 55%` persistent buffer footprint.

---

### `TRAIN-OPT-8`: Complete `PX-3b` Dispatch Coverage, Multi-Thread Scaling & Calibration Discipline

**Observed gaps (`BCIR-50M-Study/results/dispatch-*.json`, `wedge-integrated/results/calibration.json`,
`wedge-pr818/results/paired-timing.json`):**
- `BCIR-50M-Study` proved `PX-3b` per-job clock probing (`BCIR_DT_TIME_PROBE`) works cleanly at 50M
  with `run_program` parity, but only probed 1 site family (`dispatch:down.fwd`) at 2 threads.
- `wedge-integrated`'s unguarded 3-rep median selector cost `30.85 s` and misranked plans on a noisy
  VM; `wedge-pr818` showed `MADV_HUGEPAGE` slowed T512 (`0.959×`) despite verified `63 MiB` THP.
- `wedge-next` showed scalar custom-mixer kernels and batch-only scans scaled only `1.21×` from 1T
  to 4T at `B=1`.

**Proposed engineering change:**
- Extend 50M dispatch calibration to all `DISPATCH_KINDS` (`qkv`, `o`, `gateup`, `down`, `attention`,
  and backward VJPs) across `threads = {2, 4, 8}`, keeping BCIR's `native_choose.py` 95% Student-$t$
  interval acceptance rule (never adopting an alternative whose CI crosses `256 = 1.0×`).
- Charge calibration wall time against an explicit amortization threshold (skipping autotuning for
  short runs where calibration exceeds 10% of total training time, falling back to the declared
  deterministic heuristic).
- Make `MADV_HUGEPAGE` opt-in / profile-gated rather than unconditional for medium working sets
  (`< 128 MiB`) where separate aligned mappings outperform a single THP slab on EPYC.
- Require every new custom mixer kernel under `MIXER-EXT` to parallelize over `(B, H)` (or chunks)
  and use AVX2/AVX-512 SIMD kernels for nonlinear gates/scans rather than scalar loops.

---

### `TRAIN-OPT-9`: Per-Lifetime Dirty-Writeback HAM Modeling & Tighter BF16 Panel Alias Proofs

**Observed gaps (`wedge-next/results/ham.json`, `wedge-pr818/results/buffer-inventory.json`):**
- `wedge-next`'s HAM proof-of-concept modeled an entire model working set as one read-only resource
  (`-77.7%` transfers at 64 MiB bank), ignoring dirty gradient/moment writebacks and per-layer
  activation streaming.
- `wedge-pr818` disabled `BF16-PANELS` (`BCIR_DT_NO_PANELS=1`) on compact rematerialized layouts
  because `ff_remat` cross-aliases `dg`/`up` and `du`/`gate`, which was only verified for
  elementwise FP32 writers.

**Proposed engineering change:**
- Split HAM training plans into distinct read-only (`weights`, `rope_table`), write-once/consume
  (`layer_cache[l]`), and read-modify-write (`gradients`, `moment1`, `moment2`) resources so HAM
  accurately prices `REMAT` recomputation against dirty host-RAM / CXL / HBM transfer cost in
  BCIR's 12D cost vector.
- Add an alias-safe BF16 panel layout for `REMAT` layers (staging panel tiles in `workspace` or
  disjoint scratch offsets proven by `place_by_lifetime`) and qualify `BF16-PANELS` under `REMAT`
  with red-zone + UBSan parity gates.

---

### `TRAIN-OPT-10`: Split-Repository Training Data & Provenance Architecture (`Mecca-Research/Modular-AI`)

**Operational principle:**
Keep `Binary-Correspondence-Intermediate-Representation` lean (operational compiler, MLIR dialect,
native C/C++ runtime, verification gates, and compact engineering records) while hosting heavy
training archives, multi-seed `.npz` checkpoints, raw per-step logs, and training datasets in
[`Mecca-Research/Modular-AI`](https://github.com/Mecca-Research/Modular-AI) under
`BCIR Model Training/`.

**Standard archive contract for every future training study uploaded to `Modular-AI`:**
1. Frozen `protocol.json` (plus any pre-test `extension-protocol.json`) hashed before training begins.
2. Zero-pickle numeric checkpoint storage (`np.savez` / `allow_pickle=False`), split into separate
   `*_weights.zip` and `*_optimizer.zip` archives above 200 MB with a standalone `restore_checkpoint.py`
   round-trip verifier (adopting `BCIR-50M-Study`'s proven delivery format).
3. `SOURCE_MANIFEST.json`, `results/host.json`, `results/build.json`, `results/data.json`, and
   `results/final-audit.json` pinning exact BCIR commit SHA, C source hashes, dataset SHA-256, and
   hardware/toolchain status.
4. Active BCIR docs reference `Modular-AI` archive names + commit/manifest hashes rather than
   copying `.npz` binaries or frozen study trees into `BCIR`.

---

## Part C. Formalizing Weight-Space Engineering & Circuit Compilation (`WSP-1` – `WSP-4`)

`BCIR-50M-Study` §5 separated four mechanisms that are routinely conflated under "weight-space
programming": exact circuit compilation (Exp A: `65,536/65,536`), blind structural fault repair
(Exp B: `≤ 5.5e-7` nats, `0` false positives), functional gauge permutation + moment transport
(Exp C: `≤ 1.05e-5` logit drift), and clean-model weight-only SVD deflation (Exp D: `0/4` test
improvements). BCIR should formalize the useful primitives behind strict provenance laws without
overclaiming weight-only pretraining:

### `WSP-1`: Versioned `WeightPatch` Provenance Contract
Define a structured, zero-pickle transformation record (JSON + optional DER envelope via
`bcir/asn1/`) for any post-hoc weight modification:
- **Identity**: `base_checkpoint_sha256`, `bcir_commit`, `numerical_contract`, `patch_kind`
  (`"structural_repair" | "gauge_permutation" | "low_rank_edit" | "circuit_graft"`).
- **Target & Representation**: exact tensor names, axis correspondence (e.g., coupled
  `gate_proj[row] / up_proj[row] / down_proj[col]`), changed coordinate count, and either exact
  sparse replacement values or compact low-rank factors `(u, v, alpha)`.
- **Optimizer-State Policy (mandatory, from Exp C)**:
  - `"transport"`: permitted **only** for mathematically proven gauge-preserving transforms with
    explicit moment permutation/transform rules verified against a 1-step unpermuted reference.
  - `"inference_only"` or `"reset_moments"`: required for all non-gauge edits (Exp B / Exp D),
    refusing silent continuation with stale Adam moments.
- **Verification & Rollback**: selection split hash & forward-pass compute budget, held-out CE
  delta, KL divergence from base distribution, top-1 flip rate, and inverse/rollback hash.

### `WSP-2`: Zero-Inference Structural Weight-Health Sentinel
Generalize `BCIR-50M-Study` Exp B into an optional, low-overhead diagnostic check
(`bcir.hosted.models.weight_health`) runnable on saved checkpoints or periodic training hooks
(`1–41 ms` at 10M–50M, zero forward passes):
- Flags extreme column/row norm outliers (`> 8× median`), dead SwiGLU columns, NaN/Inf/subnormal
  contamination, and broken tied-embedding symmetry.
- Explicitly positioned as a **telemetry & suspect-tensor recovery diagnostic** — never a
  replacement for SHA-256 checkpoint integrity verification (when a trusted backup exists) and never
  claiming semantic error correction.

### `WSP-3`: Activation-Guided Behavioral Edit Discipline (Avoiding Exp D's Overfitting Trap)
`BCIR-50M-Study` Exp D proved that weight-only spectral deflation selected on 8 validation batches
(2,048 tokens) overfits validation (`3/4` selected $\alpha=0.3$, `0/4` improved test CE) while
perturbing 99.999% of matrix coordinates. Any future clean-model or behavioral weight-editing
proposal in BCIR must enforce:
1. **Measured behavioral diagnosis first** (activation-space / residual-stream probe on a concrete
   measured failure mode such as repetition loops or factual recall failure, not blind weight SVD).
2. **Adequate selection split + untouched holdout** ($\ge 16\text{k}$ selection tokens, separate
   held-out test and general-language collateral KL ceiling $\le 10^{-3}$).
3. **Matched-norm random control AND compute-matched SGD/LoRA control** reported side by side.

### `WSP-4`: Verified Neural Circuit Compilation & Symbolic Routing Bridge
`BCIR-50M-Study` Exp A demonstrated that a compiled 50-parameter `3→8→2` ReLU full adder achieves
`65,536/65,536` exact 8-bit arithmetic whereas 1/3 identical Adam controls collapsed to `10.01%`.
Combined with `wedge-integrated`'s 68-byte C exact KV registry (`1,568/1,568`), this reinforces
[`training/SLM_ROADMAP.md`](../../training/SLM_ROADMAP.md) §2.6 (`SLM-6`):
- Route schema-known exact associations and deterministic arithmetic/logic to **verified symbolic
  engines or compiled exact micro-circuits** behind BCIR's `verify` callback boundary, rather than
  burning neural capacity on tasks where SGD exhibits dead-neuron variance.
- Reserve learned neural weights for ambiguous, unstructured, and compositional generalization, and
  benchmark compiled sub-circuits honestly against plain C/compiler controls (as Exp A did).

---

## Priority Execution Matrix

| Priority | Slice | Primary Metric Targeted | Dependencies |
|---:|---|---|---|
| **1** | `TRAIN-OPT-1` ($\mu$P + residual init) & `TRAIN-OPT-4` (fast C init / `init="skip"`) | Fix 99.8% clip saturation; cut 50M init `15.6s → <0.05s` | None (immediate on `main`) |
| **2** | `TRAIN-OPT-5` (`BCIR-FlashAttn-CPU` tiled training attention) | Cut attention B4/T2048 prob storage `2,000×`; remove `attention32` confound | `bcir_tensor.c` `INFER-ATTN` |
| **3** | `TRAIN-OPT-2` (causal suffix trimming + masked-target loss) | `~1.6×` speedup on variable-length training; clean multi-length batches | `bcir_decoder_train.c` ABI 6 |
| **4** | `MIXER-EXT` + `Arch-1` (`BCIR-DeltaNet`) & `Arch-2` (`BCIR-Hybrid-SWA`) | Solve 4-pair KV binding (`>80%`) + `T2048` recall + byte LM simultaneously | ABI 6 + `TRAIN-OPT-1..2` |
| **5** | `TRAIN-OPT-6` (`AVX512_BF16` `VDPBF16PS` contract) & `TRAIN-OPT-7` (Q8/Adafactor state) | Turn AMD EPYC BF16 `0.877× → >1.35×`; halve persistent state (`16 → 8 B/p`) | `bcir_tensor.c` / `bcir_decoder_train.c` |
| **6** | `TRAIN-OPT-3` (Tier M/L 10M–50M token scaling) + `Arch-4/5/6` + `WSP-1..4` | True scaling laws (`>= 1` tok/param), `4×` byte-patch throughput, auditable patches | Priorities 1–5 + `Modular-AI` archive contract |
