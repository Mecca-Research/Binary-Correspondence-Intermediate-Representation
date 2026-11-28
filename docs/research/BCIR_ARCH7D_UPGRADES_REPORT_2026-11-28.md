# BCIR `Arch7d` 7-Upgrade Performance Report & Architecture Decisions (`AdaGC`/Kahan, Output-Boundary, `FWP-Snapshot`, Co-Evolution, `CIG`+Turbo `SIM`, `ES`/`MeZO`+Critic, `WeightWatcher`/`Tracr`/`IWIBench`)

**Date:** 2026-11-28
**Status:** Production ABI 6 C11 + Python implementation for Upgrade 1 (`runtime/c/bcir_precision.{h,c}`, `runtime/c/bcir_mixer_ext.{h,c}`, `bcir/hosted/models/mixer_ext.py`, `runtime/c/test_mixer_ext.c`, `bcir/tests/test_mixer_ext.py`) verified with `0`-bit-drift `1T == 2T`, `BCIROPT1` v2 resume, Clang 23 ASan+UBSan+Static Analyzer (`18/18` units clean), `MEMORY_DISCIPLINE_SANITIZE=1`, and Python 3.15 (`4,865/4,865` tests); complete bare-metal ISO C11 + Python empirical evaluation across all 7 upgrades archived in [`Mecca-Research/Modular-AI` PR #232](https://github.com/Mecca-Research/Modular-AI/pull/232) under `BCIR Model Training/arch7d-upgrades-study/` (pinned commit `616caec3ef875c394c65266cd554893c4b21cbcb`, 36 SHA-256-verified files in `results/MANIFEST.json`, `1.5 MiB`, LF-only).
**Host & Toolchain Provenance (`P6`):** `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 physical cores, `avx2`, `fma`, no `avx512f`, no AMX), `Linux 6.18.57 x86_64`, `cc (Ubuntu 13.3.0)` (`-std=c11 -O3 -mavx2 -ffp-contract=off -fno-math-errno -Wall -Wextra -Wpedantic -Werror`) and `clang 23.1.6` + `Python 3.15.2` for BCIR latest-toolchain validation.
**Related BCIR Roadmaps & Reports:**
- [BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md](BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md) (`MIXER-EXT`, `Arch-1..7`, 3-Tier Control-Surface Law)
- [BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md](BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md) (`TRAIN-OPT-1..10`, `WSP-1..4`, `SIM-1..3`, `P1–P6`)
- [BCIR_ARCH7D_10M_PERFORMANCE_REPORT_2026-11-27.md](BCIR_ARCH7D_10M_PERFORMANCE_REPORT_2026-11-27.md) (`BCIR-Arch7d-10M` reference build & tracked `9.87 MiB` `BCIRQ8 v1` checkpoint)
- [BCIR_WSP_SIM_HYBRID_REPORT_2026-11-27.md](BCIR_WSP_SIM_HYBRID_REPORT_2026-11-27.md) (Prior `18`-arm `WSP`/`SSP-Live`/`SIM` hybrid study, `Modular-AI` PR #231 `d823c20`)
- [BCIR_PRECISION_FRONTIER_AND_ARCH_STUDY_2026-11-22.md](BCIR_PRECISION_FRONTIER_AND_ARCH_STUDY_2026-11-22.md) (`50M` precision frontier, `COMP-1`, ABI 6 `1.024M` pilot)

---

## 0. Executive Summary: Bibliography Audit → 7 Empirical Verdicts

Auditing **all 67 references** in Google's *Self-Improvement Memory* (`arXiv:2611.18856`), **all 42 BibTeX references + 11 November 2026 follow-ups** in MIT CSAIL's *Sparks of Weight-Space Programming*, and the independent Nov 16 Claude WSP replication (`arXiv:2611.20953`) surfaced **seven high-leverage upgrades** across BCIR's training optimizer, stateful `GDN-2` runtime, co-evolution law, continual `SIM` orchestrator, zero-backprop adaptation layer, and `WSP-2`/`WSP-4` qualification gates. Every upgrade was implemented in ISO C11 + Python against BCIR's production ABI 6 runtime and `BCIR-Arch7d-10M` (`10.037M` params, `3× GDN-2 + 1× G1-NoPE-MLA + Canon-AC + mHC C=2`), with Gate 0 parity (`max_abs_logit_diff = 1.24e-5`, `episode_restore_max_diff = 0.0`, `comp1_parity_max_diff = 0.0`, zero Clang 23 ASan/UBSan findings):

| # | Upgrade & Literature Anchor | Measured Headline Result | Honest Negative / Boundary Finding | BCIR Production Decision |
|---:|---|---|---|---|
| **1** | **Per-Tensor `AdaGC` + FP32 Kahan + Per-Group `cos(g_t, m_{t-1})` (`2502.11034`, `2610.33826`)** | `AdaGC` + `clip=3.0` beats global-only on `2/2` `Arch7d_2M` seeds (`2.28800` vs `2.28916`) & `10M` cont. (`1.43840` vs `1.43851`) at matched speed (`895.8` vs `890.0 tok/s`, `+531.5 KiB` Kahan at 10M) | Removing global `clip=3.0` regresses under full-batch OOD spikes (`+0.0241` nats); raw 10-step reactive OPAL LR throttling starves noisy `B=4` batches (`+0.0197` nats) | **SHIPPED IN ABI 6** (`bcir_precision.{h,c}`): compose `AdaGC` *under* `clip=3.0`; Kahan opt-in on `embed`/`norm`; keep `cos(g,m)` diagnostic-only at small batch |
| **2** | **Output-Boundary Control (`final_norm` gain + sparse logit bias; `2512.12895`, `2611.20953`)** | Reproduces `2611.20953`: `13` boundary scalars (`gain=0.75` + `12` loop-token biases) jump `27.43 → 89.55` (`+62.12`, loop `90.6% → 9.4%`) at `0` latency cost | `final_norm_gain` alone is mathematically argmax-invariant (`+0.00` score, `+0.092` dCE); static always-on boundary costs `0.134` KL on clean prose | **ADOPT (ECHO-GATED):** Wire behind echo-detector trigger (`ctrl_apply_always=0`) in Tier 1 decode runtime so clean prose pays `0.000` KL |
| **3** | **`BCIR-FWP-Snapshot` Revisable `GDN-2` Fast-Weight State $S_t$ (`2102.11174`, `2610.12668`)** | `0.750 MiB` FP64 (`0.375 MiB` FP32, `134 us` save / `131 us` restore) rollback + rank-1 delta erasure jumps `27.43 → 76.56` (`loop 90.6% → 28.1%`) & recovers `96.8%` of mid-stall prefix damage (`+0.00222` vs `+0.13692` Blind Reset, `S_t cos=0.970`) | Full episode rollback (`S_t` + `L3` MLA KV + `Canon-AC` ring) scales with `T` (`1.81–2.56 MiB` at `T=64..256`, `425–601 us`); requires layout-fingerprint validation (verified fail-closed) | **ADOPT (PHASE M3):** Standardize `BCIR-FWP-Snapshot` (`FWP1`/`FWS1`) with layout fingerprint in `bcir_q8_model` / `Arch-7d` streaming decode |
| **4** | **On-Policy Localized Correction for Weight + Harness Co-Evolution (`2605.27276`, `2609.09134`)** | `OnPolicy_Localized_Anchored` (`68.75%` `BCIR_EXT_IGNORE` mask + 50% corpus anchor) wins co-evolved score (`71.54`, `+15.75` = `1.77×` OffPolicy's `+8.91`) and is the **only** arm clearing `P4` (`+0.0213 <= +0.04`) | Unanchored on-policy SFT collapses (`+0.125–0.223` dCE, `-16.6` to `-17.9` with-harness); anchored co-evolution specializes to the active harness (`noH` `9.96` vs OffPolicy `17.62`) | **ADOPT LAW (`COEV-1`):** Require `BCIR_EXT_IGNORE` failure-span masking + `>= 50%` pretraining-corpus replay anchor on all weight+harness updates |
| **5** | **Factorized Core + Turbo Overlays & `CIG` Graph `bcir.sim.cig.v2` (`2609.14857`, `2610.15150`, `2609.40330`, `2610.05039`)** | Across 4 conflicting shifts (`Prose → Struct → Short → COMP-1`), `Factorized_Harmony_CIG` hits `98.71` final mean with **`0.00` forgetting** (`25` evals, `106.1 s` vs Monolithic `95.30` / `8.73` max forgetting / `48` evals / `185.4 s`); zero-label `TurboRouter` hits `98.57` (`93.75%` route acc, `0.04` max forgetting — `213×` cut) | Flat `SIM v1` cuts search evals (`30` vs `48`) but still suffers full `8.73` monolithic forgetting on conflicting regimes | **UPGRADE `SIM` → `bcir.sim.cig.v2`:** Replace single monolithic harness with `universal_core + regime overlays` + telemetry-driven `TurboRouter` + typed `CIG` graph |
| **6** | **Forward-Only `ES`/`MeZO`, Matched-Budget `Best-of-K` & Idea-Level Critic (`2509.24372`, `2305.17333`, `2610.08993`, `2607.12227`)** | Inference-RAM `ES` beats `MeZO`/`Random` on 520-param Tier-2 (`36.71` vs `34.63`/`31.05`, `dCE=-0.00019`) & recovers `-0.01608` nats + `+3.72` pts on `COMP-1` 12 scales; `0.18 ms` Idea-Level Critic hits **Spearman `ρ=0.827`** on held-out D4 (Top-3 = `100%` oracle, **`85%` C11 rollouts saved / `6.67×` faster**) | Zero-edit `Best-of-4` crushes static weight `ES` (`98.98` vs `36.71`, `0.0%` loops, `0.0` KL) whenever `4.4×` per-query latency (`1356` vs `310 ms/p`) is affordable | **ADOPT DUAL POLICY:** Gate `CIG` proposals with Idea-Level Critic; use `Best-of-4` when `>= 4.4×` test-time budget exists, `ES` when `1×` edge latency & inference-only RAM bind |
| **7** | **Data-Free `WeightWatcher` ESD $\alpha$ + `Tracr` Circuits + `IWIBench` 5-Task Suite (Nature Comms 2021, `2301.05062`, `2611.08647`)** | Compiled `Tracr` parity/Dyck-1 hit **`100.0%` exact** vs `51.2%`/`98.1%` 3-seed Adam; **`IWI5` (data-free 12-scalar Frobenius RMS restoration on `COMP-1`) recovers `-0.11107` nats (`PPL 8.19 → 7.33`) with `0` forward passes**; `IWI1` (`3` `L3.bg` scalars) gains `+6.69` (`dCE=-0.00085`); `IWI4` repairs `18×` column spike (`KL=0.00002`) | **Muon ESD Discovery:** Muon orthogonalization preserves full spectral rank (`48.2` @ 2M, `92.7` @ 10M) & raises $\alpha$ to `5.7–7.4` (classical SGD `[2,4]` HT-SR does NOT apply to Muon, nor detects `COMP-1` PPL doubling); `IWI2`/`IWI3` show null/reversal limits of naive projection steering | **ADOPT IN `WSP-2`/`WSP-4` & `COMP-1`:** Add data-free Frobenius RMS restoration to `COMP-1` PTQ default; log optimizer-calibrated ESD $\alpha$/`srank` in `WSP-2` (never replacing `P4`); keep `Tracr` + `IWIBench` gates |

---

## 1. Upgrade 1: Production ABI 6 `AdaGC`, FP32 Kahan Compensated Rounding & Per-Group Telemetry (Experiment A)

### 1.1 Production C11 + Python Implementation (`BCIR` ABI 6)

Upgrade 1 is implemented directly in BCIR's production ABI 6 runtime with **100% backward compatibility** (`enable = 0` defaults preserve byte-for-byte ABI 6 behavior and `BCIROPT1` v1 serialization):

- **Headers & Runtime:** [`runtime/c/bcir_precision.h`](../../runtime/c/bcir_precision.h), [`runtime/c/bcir_precision.c`](../../runtime/c/bcir_precision.c), [`runtime/c/bcir_mixer_ext.h`](../../runtime/c/bcir_mixer_ext.h), [`runtime/c/bcir_mixer_ext.c`](../../runtime/c/bcir_mixer_ext.c), [`bcir/hosted/models/mixer_ext.py`](../../bcir/hosted/models/mixer_ext.py).
- **Deterministic Parallelism:** `opt_tensor_stats_worker` runs 1-task-per-shape with fixed ascending element reduction, maintaining `0` bit drift between `1T` and `2T..NT` across all 7 native optimizers (`FP32`, `Q8`, `Q8+BF16Grad`, `Adafactor`, `Adafactor+BF16Grad`, `Muon-FP32`, `Q8-Muon+BF16Grad`).
- **Kahan Compensation & `BCIROPT1` v2:** `opt_kahan_add_delta` preserves sub-ULP updates on FP32 `embed`/`lm_head`/`norm` tensors across all 7 optimizers; `BCIROPT1` v2 atomically serializes/validates/commits AdaGC EMA norms, clip-hit counters, and Kahan residual buffers with `0`-bit-drift mid-run resume.
- **Verification (`runtime/c/test_mixer_ext.c`, `bcir/tests/test_mixer_ext.py`):** `PASS upgrade1.adagc_kahan_telemetry kahan_subulp=ok 1T==2T_bits=0 v2_resume_bits=0`; zero leaks/UB under Clang 23 ASan+UBSan and `MEMORY_DISCIPLINE_SANITIZE=1`; `0` diagnostics on Clang 23 `--analyze`; `4,865/4,865` passed on Python 3.15.2.

### 1.2 Empirical Results (`expA_adagc_kahan_pergroup.json`)

**2-Seed `Arch7d_2M` (`Opt_Q8_Muon_BF16Grad`, `80` steps, `40,960` tok, `B=4, T=128`, `grad_clip=3.0`):**

| Regime | Seed 42 Test CE | Seed 137 Test CE | Mean Test CE (Std) | Mean Test PPL | Pure `train_loop` tok/s | Kahan RAM | AdaGC Hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Baseline_GlobalGC` (`clip=3.0`) | `2.28859` | `2.28972` | `2.28916 (0.00056)` | `9.867` | `890.0` | `0.0 KiB` | `0` |
| **`AdaGC_Only` + `clip=3.0`** | **`2.28750`** | **`2.28850`** | **`2.28800 (0.00050)`** | **`9.855`** | **`895.8`** | **`0.0 KiB`** | `1,545` |
| `AdaGC_Kahan_StaticLR` (`wv×0.75`, `fn×0.5`) | `2.28792` | `2.29095` | `2.28944 (0.00151)` | `9.869` | `897.2` | `+522.25 KiB` | `1,475` |
| `AdaGC_Kahan_ReactiveOPAL` (raw 10-step) | `2.30192` | `2.31588` | `2.30890 (0.00698)` | `10.064` | `881.1` | `+522.25 KiB` | `1,465` |

**`BCIR-Arch7d-10M` `16`-Step Continuation (`B=2, T=128`, `Opt_Q8_Muon_BF16Grad`):**
- `Baseline_GlobalGC`: `final_test_ce = 1.43851` (`PPL 4.2144`), `68.199 MiB`, `141.9 tok/s`.
- `Upgrade1_AdaGC_Kahan_PerGroup`: **`final_test_ce = 1.43840` (`PPL 4.2140`)**, `68.718 MiB` (**`+531.5 KiB` / `+0.76%`** for Kahan on `embed`+`norm`), `141.6 tok/s`, `814` per-tensor AdaGC hits.
- Per-group final $\cos(g_t, m_{t-1})$ telemetry cleanly separates structural stability strata: `canon_mhc = +0.196` (highly coherent) vs `gdn2_gates = -0.022`, `attn_proj = -0.029`, `embed = -0.034`, `mlp = -0.042`, `value_proj = -0.043`, `norm = -0.046`, and **`g1_gate = -0.132`** (highest sign-flip oscillation on the global anchor gate).

**Honest Negative Findings:**
1. **Synthetic OOD Spike Stress Test (`50` steps, OOD byte batches at steps `15, 30, 42`):** Disabling the global safety ceiling (`grad_clip = 30.0`) under `AdaGC_PerTensor` worsened test CE (`2.44726` vs `GlobalGC_3.0` `2.42317`, `+0.0241` nats) because early warmup (`gnorm 6–8`) inflated EMA baselines before step 15. **Law:** Always compose `AdaGC` *under* `grad_clip = 3.0`.
2. **Naive Reactive OPAL LR Throttling (`B=4`):** Raw mini-batch `cos(g_t, m_{t-1})` fluctuates negative frequently on small byte batches, starving LR (`+0.0197` nats regression). Keep `cos(g_t, m_{t-1})` **diagnostic-only** unless EMA-smoothed at large batch sizes (`N >= 8192`).

---

## 2. Upgrades 2 & 3: Output-Boundary Control & `BCIR-FWP-Snapshot` on `BCIR-Arch7d-10M` (Experiment B)

### 2.1 Upgrade 2: Output-Boundary Control (`expB1_output_boundary.csv`)

Tuned strictly on `12` WikiText-2 validation `evolve` prompts and evaluated on `32` untouched held-out test prompts + `12` test CE/KL sequences:

| Arm | Boundary Scalars | Test Score | Loop Rate | `dist4` | Test `ΔCE` | `KL` vs Base | ms/prompt (2T) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Clean_Base` | `0` | `27.43` | `90.6%` | `0.455` | `+0.00000` | `0.00000` | `302.6` |
| `Boundary_FinalNorm_Only` (`gain=0.70`) | `1` | `27.43 (+0.00)` | `90.6%` | `0.455` | `+0.09244` | `0.08719` | `293.0` |
| `Boundary_CtrlBias_Only` (`12` loop-toks, `-0.80`) | `12` | `71.44 (+44.01)` | `34.4%` | `0.772` | `+0.03692` | `0.04101` | `306.2` |
| **`Boundary_FinalNorm_Plus_CtrlBias` (`g=0.75, -0.80`)** | **`13`** | **`89.55 (+62.12)`** | **`9.4%`** | **`0.885`** | `+0.13441` | `0.13413` | `302.7` |
| `SSP_Live_GDN2_Only` | `0` | `23.95 (-3.48)` | `90.6%` | `0.385` | `+0.00958` | `0.00649` | `296.1` |
| `Composed_Boundary_Plus_SSP_Live` | `13` | `84.25 (+56.82)` | `15.6%` | `0.841` | `+0.14233` | `0.13973` | `303.6` |

**Mechanistic Confirmation of `arXiv:2611.20953`:** Uniform `final_norm_gain` alone is mathematically argmax-invariant under greedy decoding (`+0.00` score across all 5 evolve grid points `0.70..1.35`, while shifting calibration CE `+0.092` nats). Paired with `12` sparse control-token biases, however, `g = 0.75 < 1.0` amplifies effective bias-to-logit ratio (`argmax(g·z + b) = argmax(z + b/g)`), jumping score `+18.11` pts *above* `CtrlBias_Only` (`71.44 → 89.55`, loop `90.6% → 9.4%`) at zero latency cost. Because static unconditional activation incurs `0.134` KL on clean text, production BCIR gates Output-Boundary Control strictly on online echo triggers (`ctrl_apply_always = 0`).

### 2.2 Upgrade 3: `BCIR-FWP-Snapshot` Revisable `GDN-2` Fast-Weight State $S_t$ (`expB2_fwp_snapshot.csv`)

- **Measured Snapshot Footprint & Latency (`BCIR-Arch7d-10M`, `3 × 8 × 64 × 64` `GDN-2` fast-weight state $S_t$):**
  - Compact `GDN-2` fast weights (`FWS1`): **`786,464 B` (`0.750 MiB`) FP64 lossless** (`134.1 us` save / `130.6 us` restore); **`393,248 B` (`0.375 MiB`) FP32 2× compressed** (`max_logit_diff = 1.91e-6`).
  - Full episode snapshot (`FWP1` = `S_t` + `L3` MLA KV + `Canon-AC` ring + 64-bit layout-fingerprint header): `1.81 / 2.06 / 2.56 MiB` at `T = 64 / 128 / 256` (`425–601 us` restore, bit-exact `0.0` diff, fail-closed on architecture mismatch verified under Clang 23 ASan).
- **Stall Recovery Rollouts (`32` untouched held-out test prompts, `0` weight edits, `0` logit biases):**

| Episode Recovery Mode | Test Score | Loop Rate | `dist4` | Mean Rollbacks | `S_t` Retain Cosine |
|---|---:|---:|---:|---:|---:|
| `Mode0_Forward_Only` | `27.43` | `90.6%` | `0.455` | `0.00` | `1.0000` |
| `Mode1_Blind_Full_Reset` (wipes prefix on stall) | `30.66 (+3.23)` | `93.8%` | `0.551` | `6.00` | `1.0000` |
| `Mode2_FWP_Snapshot_Rollback` (bit-exact rollback + taboo) | `72.80 (+45.37)` | `34.4%` | `0.800` | `5.50` | `1.0000` |
| **`Mode3_FWP_Rollback_Plus_Revise` (rollback + rank-1 delta erasure)** | **`76.56 (+49.13)`** | **`28.1%`** | **`0.812`** | `5.56` | **`0.9697`** |

- **Direct Prefix-Memory Retention Probe (`16` WikiText-2 passages after forced 16-token stall excursion):**
  - `Oracle_NoStall`: `1.45177` CE (`4.2707` PPL)
  - `Forward_AfterStall`: `1.52065` CE (`+0.06888` nats stall pollution)
  - `Blind_Full_Reset`: `1.58869` CE (`+0.13692` nats prefix amnesia — `2×` worse than not resetting!)
  - **`FWP_Rollback`**: **`1.45177` CE (`+0.00000` — bit-exact Oracle)**
  - **`FWP_Rollback_Revise`**: **`1.45399` CE (`+0.00222` nats — recovers `96.8%` of stall damage and `98.4%` of blind-reset amnesia while outperforming pure Rollback on breakout `76.56` vs `72.80`!)**.

---

## 3. Upgrade 4: On-Policy Localized Correction Law for Weight + Harness Co-Evolution (Experiment C)

Matched `24`-step `Opt_Q8_Muon_BF16Grad + AdaGC` post-training from an identical `Arch7d_2M` checkpoint (`120` pretrain steps, `test_ce = 2.12786`, `noH = 13.29`, `withH = 55.79` inside active Tier-1 `SSP-Live` harness; `32/32` validation rollouts contained real failing spans; localized mask masked `68.75%` of non-failing tokens via `BCIR_EXT_IGNORE`):

| Co-Evolution Law | Test CE (`ΔCE` vs Base) | Test PPL | `noH` Score | `withH` Score (`Δ`) | `withH` Loop % | Clears `P4` (`<= +0.04`)? |
|---|---:|---:|---:|---:|---:|---|
| `Base_Student` | `2.12786 (+0.00000)` | `8.397` | `13.29` | `55.79 (+0.00)` | `62.5%` | Reference |
| `OffPolicy_FullImitation` (10M teacher seqs) | `2.20595 (+0.07809)` | `9.079` | `17.62` | `64.70 (+8.91)` | `46.9%` | **NO** |
| `OnPolicy_FullHarness` (no replay) | `2.35078 (+0.22292)` | `10.494` | `4.07` | `37.94 (-17.85)` | `93.8%` | **NO (collapse)** |
| `OnPolicy_Localized_NoReplay` (`68.75%` masked) | `2.25309 (+0.12523)` | `9.517` | `12.22` | `39.24 (-16.55)` | `90.6%` | **NO** |
| **`OnPolicy_Localized_Anchored` (`50%` corpus anchor)** | **`2.14918 (+0.02132)`** | **`8.578`** | `9.96` | **`71.54 (+15.75)`** | **`34.4%`** | **YES (`3.66×` less drift)** |

**Established Law (`COEV-1`):** Unanchored on-policy SFT collapses catastrophically (`+0.125–0.223` dCE, `-16.6` to `-17.9` with-harness). Pairing `BCIR_EXT_IGNORE` localized failure-span masking with `50%` pretraining-corpus anchoring is the **only** law that passes BCIR's `P4` `+0.04` drift gate (`+0.0213`) while achieving **`1.77×`** the co-evolved with-harness improvement of off-policy imitation (`+15.75` vs `+8.91`).

---

## 4. Upgrade 5: Factorized Universal-Core + Turbo Instance-Routed Overlays & `CIG` (`bcir.sim.cig.v2`, Experiment D)

Sequential 4-domain continual evolution (`D1 Prose → D2 Structured → D3 ShortPrefix → D4 COMP-1 1-bit`) on real `BCIR-Arch7d-10M` with calibrated real-entropy fidelity bands (matching `BCIR-Arch7d-10M`'s `1.32`-nat clean baseline entropy) and **strictly isolated cold caches per method**:

| Method | D1 Prose | D2 Struct | D3 Short | D4 COMP-1 | Final Mean | Mean / Max Forgetting | Uncached Evals | Cold Wall | Route Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Monolithic_Transfer` | `90.33` | `97.66` | `94.70` | `98.53` | `95.30` | `4.50` / `8.73` | `48` | `185.4 s` | — |
| `Flat_SIM_v1` | `90.33` | `97.66` | `94.70` | `98.53` | `95.30` | `4.28` / `8.73` | `30` | `137.1 s` | — |
| **`Factorized_Harmony_CIG`** | **`99.14`** | `97.50` | **`99.47`** | **`98.73`** | **`98.71`** | **`0.00` / `0.00`** | **`25`** | **`106.1 s`** | `100.0%` |
| **`Factorized_CIG_TurboRouter`** | **`99.10`** | `96.98` | **`99.47`** | **`98.73`** | **`98.57`** | **`0.01` / `0.04`** | `43` | `164.7 s` | **`93.75%`** (0 labels) |

**Architectural Upgrade (`bcir.sim.v1 → bcir.sim.cig.v2`):** Flat `SIM v1` prunes traps cleanly (`30` vs `48` evals, `1.35×` faster cold wall) but hits the exact same `8.73` monolithic forgetting wall when `D4 COMP-1` overwrites `D1 Prose`. Factorizing into a shared `universal_core` plus per-regime overlays driven by a telemetry-only `CIG` graph (`Evidence → Hypothesis → Intervention → Outcome`, zero domain-string heuristics) eliminates forgetting (`0.00`), cuts uncached search evals **`1.92×` (`25` vs `48`)** and cold wall **`1.75×` (`106.1` vs `185.4 s`)**, and generalizes to zero-label per-prompt `TurboRouter` at `93.75%` routing accuracy (`0.04` max forgetting — **`213×` reduction**).

---

## 5. Upgrade 6: Forward-Only `ES`/`MeZO` (Inference RAM), Matched-Budget `Best-of-K` Search & Idea-Level Critic (Experiment E)

### 5.1 Forward-Only Tier 2 `ES`/`MeZO` vs Matched-Budget `Best-of-K` Search (`expE1_es_mezo_vs_search.csv`)

Optimizing the `520`-param Tier-2 surface (`L3.bg` `G1` + `final_norm` = `0.00518%` of `10M`) strictly inside inference-only RAM (`0` backward passes, matched `32` evolve rollouts), plus `24`-rollout `ES` on established `COMP-1` `12` MLP scale factors, evaluated on `24` untouched held-out test prompts:

| Arm | Evolve Rollouts | Test Score | Loop % | Test `ΔCE` | `KL` vs Base | Serving Latency | Memory Regime |
|---|---:|---:|---:|---:|---:|---:|---|
| `Clean_Base_Q8` | `0` | `31.28` | `79.2%` | `+0.00000` | `0.00000` | `310.5 ms/p` (`1.00×`) | Inference only |
| `RandomSearch_LowDim520` | `32` | `31.05 (-0.23)` | `79.2%` | `+0.00041` | `0.00036` | `311.4 ms/p` | Inference only |
| `MeZO_Antithetic_LowDim520` | `32` | `34.63 (+3.35)` | `75.0%` | `+0.00006` | `0.00011` | `312.4 ms/p` | Inference only |
| **`ES_Antithetic_LowDim520`** | `32` | **`36.71 (+5.43)`** | **`70.8%`** | **`-0.00019`** | **`0.00016`** | `~310 ms/p` (`388.7` jitter) | Inference only |
| **`TestTimeSearch_BestOf4`** | `0` | **`98.98 (+67.69)`** | **`0.0%`** | **`+0.00000`** | **`0.00000`** | `1355.6 ms/p` (`4.37×`) | Inference only (0 edits) |
| **`TestTimeSearch_BestOf8`** | `0` | **`99.52 (+68.24)`** | **`0.0%`** | **`+0.00000`** | **`0.00000`** | `2456.5 ms/p` (`7.91×`) | Inference only (0 edits) |
| `COMP1_1bit_Base` | `0` | `62.47` | `45.8%` | `+0.00000` | `0.00000` | `304.1 ms/p` | Inference only (`COMP-1`) |
| **`COMP1_1bit_ES_12Scales`** | `24` | **`66.19 (+3.72)`** | **`41.7%`** | **`-0.01608`** | `0.00743` | `301.8 ms/p` | Inference only (`COMP-1`) |

### 5.2 Idea-Level Critic Pre-Screening (`expE2_idea_critic.csv`, `arXiv:2610.08993`)

Trained strictly on `D1..D3` (`60` real C11 proposals, zero `D4` visibility) and evaluated on strictly held-out `D4 COMP-1` (`20` proposals):
- **Fit + score cost:** `0.179 ms` (closed-form deterministic Ridge regression).
- **Held-out D4 tie-aware Spearman rank correlation:** **`ρ = 0.8271`**.
- **Verification savings:** Top-3 verified proposals recover **`100.0%` of the brute-force oracle optimum (`99.015`)**, cutting C11 verification rollouts by **`85.0%` (`6.67×` speedup, `6.06 s` vs `39.77 s`)**.

---

## 6. Upgrade 7: Data-Free `WeightWatcher` ESD $\alpha$, `Tracr` Circuits & `IWIBench` 5-Task Suite (Experiment F)

### 6.1 Data-Free `WeightWatcher` CSN Power-Law ESD $\alpha$ in `WSP-2` (`expF1_weightwatcher.csv`, `0` forward passes)

| Model / Optimizer / Precision | Test PPL | Mean ESD $\alpha$ | MLP / Attn $\alpha$ | Mean `stable_rank` | Mean `r90` | Audit ms |
|---|---:|---:|---:|---:|---:|---:|
| `Arch7d_2M_Untrained_Init` | `277.31` | `5.577` | `6.974 / 5.098` | `48.36` | `0.706` | `399.6` |
| `Arch7d_2M_AdamW_FP32` (`80` steps) | `11.90` | `5.478` | `6.860 / 5.004` | **`39.43` (`-18.5%`)** | `0.703` | `384.7` |
| `Arch7d_2M_Q8Muon` (`80` steps) | `10.11` | `5.868` | `7.460 / 5.322` | **`48.20` (preserved)** | `0.706` | `380.9` |
| `Arch7d_2M_Q8Muon_AdaGC` (`80` steps) | `10.12` | `5.713` | `7.355 / 5.151` | `48.15` | `0.706` | `385.7` |
| `GQA_Full_2M_Q8Muon` (`80` steps) | `12.43` | `6.539` | `7.964 / 5.471` | `70.87` | `0.630` | `381.1` |
| `BCIR_Arch7d_10M_BCIRQ8` (`1.114M` tok) | `4.13` | **`6.798`** | `7.352 / 6.608` | `92.73` | `0.697` | `2224.8` |
| `BCIR_Arch7d_10M_COMP1_1bit` (PTQ) | `8.25` | `6.807` | `7.387 / 6.608` | `90.66` | `0.697` | `2267.3` |

**Crucial Negative & Diagnostic Discovery:** Newton-Schulz polar orthogonalization in Muon explicitly prevents singular-value heavy-tail collapse (`stable_rank` remains `48.2` vs `48.4` init while AdamW contracts `-18.5%` to `39.4`), pushing Muon checkpoints to **`α ≈ 5.7–7.4`** even at `PPL 4.13`. Furthermore, `COMP-1` PTQ doubles PPL (`4.13 → 8.25`) while barely shifting $\alpha$ (`6.798 → 6.807`). Therefore, **classical SGD `WeightWatcher` `[2, 4]` bounds must NEVER be applied as pass/fail gates on Muon checkpoints or used to bypass `P4` held-out qualification**.

### 6.2 `Tracr`/`RASP` Compiled Exact Micro-Circuit Gate (`WSP-4`)

| Exact Deterministic Circuit | Inputs / Sequences | Compiled Exact Acc (0 steps) | Eval Wall | 3-Seed Matched Adam Control |
|---|---:|---:|---:|---:|
| **12-Bit Binary Parity** (`182` ReLU-basis params) | `4,096` exhaustive | **`100.00%`** | `1.04 ms` | `51.22%` (`[50.68, 52.34, 50.63]%`) |
| **Dyck-1 Nested Bracket Balance** (`T=16`, causal linear attn + ReLU) | `2,048` seqs | **`100.00%`** | `0.85 ms` | `98.08%` (`[97.95, 97.95, 98.34]%`) |

### 6.3 `IWIBench` 5-Task Gradient-Free Intra-Weight Injection Suite on `BCIR-Arch7d-10M` (`expF3_iwibench.csv`)

Every task emitted a provenance-hashed `bcir.wsp1.weight_patch.v1` manifest:

| Task | Target Locus (`%` of 10M) | Metric | Before → After | Collateral `ΔCE` | Collateral `KL` | Outcome |
|---|---|---|---:|---:|---:|---|
| `IWI1_LoopSuppression_L3bg_G1` | `L3.bg` (`3` scalars, `0.00003%`) | Rollout Score | **`22.75 → 29.44`** (`+6.69`) | **`-0.00085`** | `0.00017` | **PASS** |
| `IWI2_TabooToken_Eq_L3wdownRank1` | `L3.wdown` (`16,384` coords, `0.163%`) | Taboo `'='` % | `0.521% → 0.521%` | `-0.00074` | `0.00008` | **NULL** (use Upgrade 2 bias) |
| `IWI3_FormatDelim_FinalNormSteer` | `final_norm` (`512` coords, `0.005%`) | Delim `'.'/'\n'` % | `1.107% → 0.716%` | `-0.00003` | `0.00080` | **REVERSAL** (RMSNorm coupling) |
| `IWI4_BlindFaultRepair_L2wdown` | `L2.wdown` (`512` coords, `0.005%`, `0` fwd) | Corrupt → Repaired CE | **`1.38998 → 1.38847`** | `+0.00046` vs clean | **`0.00002`** | **PASS** |
| **`IWI5_COMP1_DataFree_RMSCalib`** | **`12` MLP scales (`0.00012%`, `0` fwd, `0` data)** | **`COMP-1` Test CE** | **`2.10254 → 1.99147`** | **`-0.11107` (`-0.86` PPL!)** | `0.13223` | **MAJOR DISCOVERY** |

**Immediate `COMP-1` Deployment Bonus (`IWI5`):** Analytically restoring the exact per-matrix Frobenius norm $\|W_{\text{orig}}\|_F / \|W_{\text{COMP1}}\|_F$ across the `12` wide `BCIR-Arch7d-10M` MLP matrices (`12` FP16 scalars = `24` bytes, `0` forward passes, `0` calibration tokens) immediately recovers **`-0.11107` nats (`PPL 8.19 → 7.33`) — `16.4%` of the entire Muon `COMP-1` PTQ gap before a single step of QAT or ES!** This zero-cost analytical step should run automatically at the end of every `COMP-1` Dual-SVID + Joint-ITQ export.
