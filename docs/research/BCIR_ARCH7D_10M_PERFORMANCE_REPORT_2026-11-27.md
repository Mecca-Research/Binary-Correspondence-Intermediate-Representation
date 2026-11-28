# `BCIR-Arch7d-10M` Performance Report & Reference Architecture Configuration

**Date:** 2026-11-27
**Status:** Executed bare-metal C11 + Python ctypes 6-stage empirical study on production BCIR ABI 6 `MIXER-EXT` (`runtime/c/{bcir_tensor,bcir_precision,bcir_mixer_ext}.c`, zero duplicated C code); all stage scripts, per-step CSV telemetry (`1,024` logged steps across Stages 1–4), Gate 0 proofs, multi-panel figures, SHA-256 manifest, and runnable deterministic `BCIRQ8 v1` checkpoint (`9.87 MiB`, SHA-256 `bc67f32a2a1a...`) archived in [`Mecca-Research/Modular-AI` PR #230](https://github.com/Mecca-Research/Modular-AI/pull/230) under `BCIR Model Training/arch7d-10m-training-study/` (pinned commit `f5b9e26dd008990b68f43dcb91434770f513ea7a`).
**Host & Toolchain Provenance (`P6`):** `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 physical cores, `avx2`, `fma`, no `avx512f`, no AMX, `7.8 GiB` RAM), `Linux 6.18.57 x86_64`, `cc (Ubuntu 13.3.0)` (`-std=c11 -O3 -ffp-contract=off -fno-math-errno -Wall -Wextra -Wpedantic -Werror`). Cross-compiler bit-exact determinism verified (`sha=95dfa233b413`, `0` bit drift) across GCC `-O3`, GCC `-O3 -march=native`, and Clang 23 `-O3 -march=native`.
**Related BCIR Roadmaps:**
- [BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md](BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md) (`MIXER-EXT`, `Arch-1..7`)
- [BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md](BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md) (`TRAIN-OPT-1..10`, `P1–P6`)
- [BCIR_PRECISION_FRONTIER_AND_ARCH_STUDY_2026-11-22.md](BCIR_PRECISION_FRONTIER_AND_ARCH_STUDY_2026-11-22.md) (`50M` precision frontier, `COMP-1`, ABI 6 `1.024M` pilot)

---

## 1. Reference Architecture Configuration: `BCIR-Arch7d-10M`

Synthesizing the `littlebit-sub1bit-study` (`3ec7ee5`), `precision-arch-study` (`8cc3f23`), and `mixer-ext-abi6-study` (`529f3f2` / `c707873`) findings and validated by the Stage 0 10M sizing + micro-ablation sweep, the canonical **10M-parameter BCIR reference architecture** is **`BCIR-Arch7d-10M`** (`Arch7d_10M_B_ff928_gr32_dc128`, **`10,036,768` parameters**):

```c
/* Canonical BCIR-Arch7d-10M ABI 6 configuration (runtime/c/bcir_mixer_ext.h) */
bcir_ext_spec spec = {
  .vocab          = 256u,    /* raw UTF-8 bytes */
  .width          = 512u,    /* clears BCIR P4 width gate: min(m, n) >= 512 on 12/12 MLPs */
  .heads          = 8u,      /* head_dim dh = 64 */
  .kvheads        = 2u,
  .layers         = 4u,      /* 3:1 hybrid block */
  .ff             = 928u,    /* SwiGLU FFN; Stage 0 winner via mla_dc=128 -> ff=928 reallocation */
  .tied           = 1u,      /* tied byte embedding and lm_head */
  .rope_base      = 10000.0,
  .rms_eps        = 1e-5,
  .layer_mixer    = { BCIR_MIXER_GDN2, BCIR_MIXER_GDN2, BCIR_MIXER_GDN2, BCIR_MIXER_MLA },
  .layer_use_rope = { 0u, 0u, 0u, 0u }, /* NoPE-MLA eliminates RoPE extrapolation aliasing */
  .delta_chunk    = 16u,     /* FP64 chunkwise WY recurrence */
  .mla_dc         = 128u,    /* compressed KV latent: cuts T=4096 state 33% (3.75 -> 2.75 MiB) */
  .mla_dr         = 32u,
  .init_policy    = BCIR_INIT_MUP_V1,
  .mup_d_base     = 128u,
  .gate_mode      = BCIR_GATE_G1_HEADWISE, /* arXiv:2505.06708 */
  .qk_norm        = 1u,
  .canon_mode     = BCIR_CANON_AC,         /* arXiv:2512.17351v2 (causal conv1d k=4 pre-mixer & pre-MLP) */
  .mhc_streams    = 2u,                    /* arXiv:2512.24880v2 (Sinkhorn-Knopp Birkhoff, t_max=20) */
  .online_softmax = 1u,                    /* TRAIN-OPT-5 O(T) LSE tiled attention */
  .gate_rank      = 32u,                   /* compact GDN-2 decay/write low-rank bottleneck */
};
/* Default Training Stack: BCIR_OPT_Q8_MUON_BF16_GRAD (7.125 B/p, 68.20 MiB persistent, grad_clip=3.0) */
```

### Parameter, Memory & Deployment Footprint Breakdown

| Resource Metric | `BCIR-Arch7d-10M` (`10.037M`) | `Arch2_Hybrid_10M` (`9.860M`) | `GQA_Full_10M` (`9.835M`) | `NarrowDeep d=384 L=8` (`12.321M`) |
|---|---:|---:|---:|---:|
| Parameter Split (`Attn/Mixer` / `MLP` / `Other`) | `4.18M` / `5.70M` / `0.16M` | `3.83M` / `5.90M` / `0.13M` | `2.62M` / `7.08M` / `0.14M` | `5.11M` / `7.08M` / `0.13M` |
| FP32 Master Weights (`MiB`) | `38.29` | `37.61` | `37.52` | `47.00` |
| Persistent RAM (`Opt_FP32_AdamW`, `16.00 B/p`) | `153.15` | `150.45` | `150.07` | `188.01` |
| Persistent RAM (`Opt_Adafactor`, `12.02 B/p`) | `115.09` | `112.99` | `112.70` | `141.36` |
| **Persistent RAM (`Opt_Q8_Muon_BF16Grad`, `7.13 B/p`)** | **`68.20` (`-55.5%`)** | `66.93` | `66.74` | `83.30` |
| Training Arena (`B=8, T=256`, `online_softmax=1`) | `850.7 MiB` | `520.9 MiB` | `373.5 MiB` | `1,137.1 MiB` |
| Training Arena (`B=4, T=512`, `online_softmax=1`) | **`829.8 MiB`** | `507.6 MiB` | `363.2 MiB` | `1,116.7 MiB` |
| `T=4096` Streaming State (`KV + Recurrent`) | **`2.75 MiB` (`5.82×` vs GQA)** | `4.75 MiB` | `16.00 MiB` | `5.12 MiB` |
| `P4` Width Gate (`min(m, n) >= 512`) | **PASS (`12/12` MLPs)** | PASS (`12/12`) | PASS (`12/12`) | **FAIL (`0/24` MLPs)** |
| `BCIRQ8 v1` Total `T=4096` Footprint (`MiB`) | **`12.62`** (`9.87` wts) | `14.45` | `25.67` | `17.24` |
| `COMP-1` Total `T=4096` (`r=16`, FP16 / FP32 scales) | **`7.79` / `7.86 MiB`** | `9.45` / `9.52 MiB` | `19.67` / `19.75 MiB` | `17.24` / `17.24 MiB` |

**Stage 0 Structural Findings:**
1. **Why `d=512, L=4` strictly dominates `d=384, L=8`:** Narrow-deep `d=384, L=8` fails the `P4` width gate on all 24 MLP matrices (`0` eligible), bloating its `T=4096` `COMP-1` footprint to `17.24 MiB` (`2.21×` larger than `BCIR-Arch7d-10M`'s `7.79 MiB`) and increasing training arena RAM by `+33.7%`.
2. **Latent-to-FFN Budget Trade-off:** In the Stage 0 32-step micro-ablation (`32,768` tok, `B=8, T=128`, `Opt_Q8_Muon_BF16Grad`), reallocating `~100k` parameters from `mla_dc` (`192 → 128`) into SwiGLU width (`ff = 896 → 928`) beat both `ff=896, dc=192, gr=32` (`val_t128` `11.40` vs `11.51`, zero-shot `val_t256` `10.60` vs `10.73`) and `ff=832, dc=192, gr=64` (`11.48` / `10.63`), while shrinking `T=4096` KV+Recurrent state by `33%` (`3.75 → 2.75 MiB`).

---

## 2. Progressive 3-Stage Curriculum Training (`1,114,112` Real Tokens)

Trained on raw UTF-8 WikiText-2 bytes (`10.94M`-byte train stream) using a unified global cosine LR schedule (`warmup=40`, `max_lr=3.0e-3`, `min_lr=2.0e-4` over `640` global steps) with `Opt_Q8_Muon_BF16Grad` (`beta1=0.95, beta2=0.95, wd=0.01, grad_clip=3.0`) and **0-bit-drift cross-stage optimizer state handoff** (`10.77 MiB` serialized state persisted via crash-atomic generation checkpoints across mid-stage chunks and `T=128 → T=256 → T=512` context transitions):

| Stage | Context / Batch (`N`) | Global Steps | Tokens (`Stage` / `Cum.`) | Wall (`s`) / `tok/s` | Med `ms/st` | Grad Norm (`Mean` / `Max`) | Clip % | Val (`CE` / `PPL` / `BPB` / `Acc`) | Test `T=128` (`PPL`) | Test `T=256` (`PPL`) | Test `T=512` (`PPL`) | Max `\|Act\|` / Sink |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stage 1** | `T=128, B=8` (`1,024`) | `0 → 192` | `196.6k` / `0.197M` | `648.2` / `303.3` | `3,174` | `3.09` / `9.78` | `26.6%` | `1.8806` / `6.56` / `2.713` / `47.1%` | `6.84` | `6.74` | `6.65` | `1.80` / `0.041` |
| **Stage 2** | `T=256, B=8` (`2,048`) | `192 → 512` | `655.4k` / `0.852M` | `1,841.5` / `355.9` | `5,358` | `1.49` / `2.81` | **`0.0%`** | `1.4373` / `4.21` / `2.074` / `58.3%` | `4.58` | `4.47` | `4.46` | `2.13` / `0.025` |
| **Stage 3** | `T=512, B=4` (`2,048`) | `512 → 640` | `262.1k` / **`1.114M`** | `810.0` / `323.6` | `5,962` | `1.53` / `2.44` | **`0.0%`** | **`1.3998` / `4.05` / `2.019` / `58.9%`** | **`4.44`** | **`4.32`** | **`4.30`** | **`2.17` / `0.013`** |

**Scale-Up Efficiency vs `1.22M` `pilot1m` (`PR #229`):**
- At `1.024M` tokens (`T=256`), `1.22M` `Arch-7d + Q8-Muon` reached untouched test **`PPL 4.81`** (`1.5716` CE).
- Scaling to `10.04M` `BCIR-Arch7d-10M` passed that mark by `655k` tokens (`val PPL 4.47`), reached untouched test **`PPL 4.47`** (`T=256`) at `852k` tokens, and finished Stage 3 (`1.114M` tokens) at untouched test **`PPL 4.32` (`T=256`) / `4.30` (`T=512`)** — **`-0.1092` nats (`-10.2%` PPL)** better with **`0.0%` clip hits across all 448 steps of Stages 2 & 3** and zero nonfinite events.
- Under `TRAIN-OPT-5` (`online_softmax=1`), doubling sequence length from `T=256 (B=8)` to `T=512 (B=4)` at fixed `N=2,048` **reduced** total training arena RAM (`850.7 → 829.8 MiB`) while holding `91%` of `T=256` throughput (`323.6` vs `355.9 tok/s`).

---

## 3. Parameter-Matched 10M Controls: Architecture × Optimizer × Curriculum Co-Design

Controlled 4-way benchmark at identical `B=8, T=256`, `96` steps (`196,608` real tokens), compared directly against Stage 1 progressive `T=128` (`192 × 1,024 = 196,608` tokens):

| Model + Optimizer (`196,608` Tokens) | Params | Pers. RAM (`MiB`) | `T=4K` State | Val CE (`T=256`) | Test `T=256` (`CE` / `PPL` / `Acc`) | Zero-Shot `T=512` (`PPL`) | `BCIRQ8` `Delta` | `COMP-1` PTQ `Delta` | Max `\|Act\|` | Wall (`s`) / `tok/s` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `GQA_Full_10M + Adafactor` | `9.835M` | `112.7` | `16.00 MiB` | `2.2308` | `2.2574` / `9.56` / `35.2%` | `9.53` | `-0.0001` | `+0.1367` | `5.39` | `389.5` / `505` |
| `Arch7d_10M + Adafactor` | `10.037M` | `115.1` | **`2.75 MiB`** | `2.0916` | `2.1267` / `8.39` / `38.8%` | `8.34` | `+0.0003` | `+0.1884` | `2.31` | `437.5` / `449` |
| `Arch2_Hybrid_10M + Adafactor` | `9.860M` | `113.0` | `4.75 MiB` | `2.0425` | `2.0738` / `7.95` / `40.4%` | `7.92` | `+0.0003` | `+0.3237` | `3.54` | **`359.6` / `547`** |
| **`Arch7d_10M + Q8-Muon (Flat T=256)`** | `10.037M` | **`68.2`** | **`2.75 MiB`** | **`2.0185`** | **`2.0593` / `7.84` / `41.0%`** | **`7.84`** | **`-0.0000`** | `+0.4721` | **`1.40`** | `575.4` / `342` |
| **`Arch7d_10M + Q8-Muon (Stage 1 T=128)`** | `10.037M` | **`68.2`** | **`2.75 MiB`** | **`1.8551`** | **`1.9082` / `6.74` / `46.0%`** | **`6.65`** | `+0.0002` | — | `1.80` | `648.2` / `303` |

**Two Operational Laws Established at 10M Scale:**
1. **`Arch-7d` × `Opt_Q8_Muon_BF16Grad` Co-Design Law:** Under diagonal `Adafactor` at 96 steps, `Arch-7d` (`8.39` PPL) warms up slightly slower than `Arch-2` (`7.95`); pairing `Arch-7d` with **`Opt_Q8_Muon_BF16Grad`** flips the ranking to `#1` (`7.84` PPL) while cutting persistent RAM `40%` (`115.1 → 68.2 MiB`), cutting peak activations `2.52×` vs `Arch-2` (`1.40` vs `3.54`), and passing `Arch7d + Adafactor`'s full 96-step validation loss at step 72 (`419.6 s` vs `425.0 s` — winning on **both equal tokens AND equal wall time**!).
2. **Progressive Context Curriculum Multiplier:** On `GDN-2 + NoPE-MLA`, spending the first `196,608` tokens at `T=128` (`192` Muon steps) instead of flat `T=256` (`96` Muon steps) improves zero-shot `T=256` untouched test CE by **`-0.1511` nats (`6.74` vs `7.84` PPL)** and `T=512` test CE by **`-0.1633` nats (`6.65` vs `7.84` PPL)** in only `1.13×` wall time (`648.2 s` vs `575.4 s`).

---

## 4. `4×` Zero-Shot Context Extrapolation & Structural Forensics

### 4.1 Untouched Test Scaling (`T=64 → T=2048`) & Per-Position Adaptation

| Horizon `T` | Regime | Test CE | Test PPL | Test BPB | Top-1 Acc | Max `\|Act\|` | Attn Sink | Inference State (`MiB`) | Eval Arena (`MiB`) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `64` | In-distribution | `1.5042` | `4.50` | `2.170` | `55.9%` | `2.15` | `0.067` | `0.78` | `204.2` |
| `128` | In-distribution | `1.4897` | `4.44` | `2.149` | `57.0%` | `2.17` | `0.044` | `0.81` | `293.0` |
| `256` | In-distribution | `1.4538` | `4.28` | `2.097` | `57.4%` | `2.19` | `0.025` | `0.88` | `471.2` |
| `512` | Max Train `T` | `1.4580` | `4.30` | `2.103` | `57.8%` | `2.17` | `0.013` | `1.00` | `486.0` |
| **`1024`** | **Zero-Shot `2×`** | **`1.4560`** | **`4.29`** | **`2.101`** | `57.3%` | `2.17` | `0.007` | **`1.25`** | `488.3` |
| **`2048`** | **Zero-Shot `4×`** | **`1.4192`** | **`4.13`** | **`2.048`** | **`58.5%`** | **`2.13`** | **`0.005`** | **`1.75`** | `854.0` |

- **Per-position in-context gain (`T=512`):** `pos 0–15` (`1.6511` CE / `5.21` PPL / `52.3%` acc) → `pos 384–511` (`1.4170` CE / `4.12` PPL / `58.8%` acc, **`-0.2341` nats**). On zero-shot `T=1024`, positions `512–767` sustain `1.4185` CE (`4.13` PPL, `57.7%` acc) with zero extrapolation drift.

### 4.2 Per-Layer Spectral, `Canon-AC` & `mHC` Structural Forensics

- **Muon Spectral Conditioning (`Init → Stage 3`):** Total Frobenius norm stays virtually flat (`215.00 → 216.94`, `+0.9%`), MLP mean 90%-energy rank holds at **`64.50%`** (`stable_rank = 167.58`), and mixer/attention `r90` holds at **`73.48%`**. Distributing energy uniformly across singular values maximizes representational capacity per parameter while making converged Muon MLPs inherently harder to truncate post-hoc to 1 bit without QAT (explaining the `+0.47` to `+0.68` nat PTQ delta vs Adafactor's `+0.19` at 197k tokens).
- **Learned `Canon-AC` Causal Kernels:** All 8 `conv1d k=4` layers converge to clean monotonically decaying byte-lag filters (`Tap -1` `0.085–0.148` RMS > `Tap -2` `0.080–0.143` > `Tap -3` `0.067–0.115` > `Tap -0` `0.037–0.049`), with pre-MLP `Canon-C` consistently `~1.5×` stronger than pre-mixer `Canon-A`.
- **Learned `mHC` `C=2` Birkhoff Matrices:** All 8 residual stream mixers remain strictly doubly-stochastic (`diag ≈ 0.953–0.961`, `off-diag ≈ 0.039–0.047`), holding `max_abs_act` flat at `2.13–2.19` across all horizons `T=64..2048`.

---

## 5. Deployment Policy (`P1–P6`) Qualification & Exact Differentiable `COMP-1` Factor-QAT

| Deployment Policy (`T=256` Untouched Test) | Weights (`MiB`) | MLP `bpw` | vs FP32 | `T=4096` Total (`MiB`) | Test CE (`Delta vs FP32`) | Test PPL | `P3`/`P4` Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `FP32_Ref` | `38.29` | `32.00` | `1.00×` | `41.04` | `1.4677` (`+0.0000`) | `4.34` | Reference |
| **`BCIRQ8_v1_g32`** (tracked artifact) | **`9.87`** | `8.25` | **`3.88×`** | **`12.62`** | **`1.4679` (`+0.0001`)** | **`4.34`** | **`ADMITTED_Q8_V1`** |
| `COMP1_PTQ_r16_fp16scale` | `5.04` | `1.1460` | `7.59×` | `7.79` | `2.1460` (`+0.6782`) | `8.55` | `PROMOTE_TO_Q8` (PTQ) |
| `COMP1_PTQ_r32_fp32scale` | `5.14` | `1.2931` | `7.45×` | `7.89` | `2.0914` (`+0.6237`) | `8.10` | `PROMOTE_TO_Q8` (PTQ) |
| **`COMP1_FactorQAT_r16_fp16s`** (`32` st, `117s`) | **`5.04`** | **`1.1460`** | **`7.59×`** | **`7.79`** | **`1.8556` (`+0.3879`, `-0.2903` vs PTQ)** | **`6.40`** | Needs longer KD-QAT for `<= +0.05` |

**Critical QAT Engineering Discovery:**
- Naive sign-flipping STE-QAT (re-running Dual-SVID + Procrustes ITQ every 4 steps at LR `1.2e-3`) suffered a `+1.54` nat binary sign-flip shock (`val CE 2.07 → 3.62 → 2.25`).
- Replacing it with **Exact Differentiable Scale-Factor `COMP-1` QAT** (freezing non-MLP tensors at `BCIRQ8 v1`, freezing binary signs `{B, Bu, Bv}`, and backpropagating exact chain-rule gradients from `bcir_ext_loss_backward` directly onto continuous FP16-rounded scales `(alpha_r, alpha_c, h_u, g_v, lam)`) produced **strictly monotonic recovery with zero shock**: `val PPL 7.96 → 6.46 → 6.12 → 5.97 → 5.91`, recovering **`-0.2903` nats (`42.8%` of the entire 1-bit PTQ gap)** on untouched test (`8.55 → 6.40` PPL) in just `32` steps (`32,768` tokens, `116.6 s`). Under strict `+0.05` nat deployment gates, BCIR's `P3`/`P4` engine automatically admits `BCIRQ8 v1` (`9.87 MiB`, `+0.0001` nats) until full 200–500 step Factor-QAT + logit KD closes the remaining gap.

---

## 6. Calibrated Multi-Core Projections & Actionable Architecture Updates

From measured 2-core Haswell AVX2 throughput (`355.9 tok/s` @ `T=256`, `323.6 tok/s` @ `T=512`, `0` bit drift):
- **Tier M (`10M` tokens, `~1.0` tok/param):** `~7.8 hrs` (2-core AVX2), **`~1.2 hrs` (16-core AVX2 est., `6.5×`)**, **`~0.43 hrs` (32-core AVX-512 est., `18×`)**.
- **Chinchilla-Optimal (`200M` tokens, `20` tok/param):** **`~8.7 hrs` (32-core AVX-512 est.)**.

**Immediate Architecture Plan Updates Driven by This Report:**
1. **Standardize `BCIR-Arch7d-10M` (`d=512, L=4, ff=928, mla_dc=128, gate_rank=32`)** as the canonical 10M reference configuration and apply the compact-latent (`dc = d/4`) + wide-SwiGLU + `d >= 512` `P4` sizing rule when scaling to 50M (`d=768, L=8, mla_dc=192, gate_rank=48`).
2. **Mandate Progressive Context Curriculum + Cross-Stage `Opt_Q8_Muon_BF16Grad` Handoff** (`T=128 → T=256 → T=512`) as BCIR's standard Tier M/L training recipe (`-0.1511` nats free gain at equal early tokens + `55.5%` persistent RAM reduction + `0.0%` post-warmup clip rate).
3. **Adopt Exact Differentiable Scale-Factor `COMP-1` QAT** (never naive periodic re-ITQ STE) as the official `COMP-1` QAT primitive for closing the Muon-trained 1-bit MLP gap, while shipping **`BCIRQ8 v1`** (`9.87 MiB`, `+0.0001` nats) immediately for near-lossless edge deployment.
4. **Next Native Runtime Priority:** Lower `BCIR-Arch7d-10M` (`GDN-2` + `NoPE-MLA` + `Canon-AC` + `mHC`) onto BCIR's `BCIRQ8 v1` / `BCIRQ8 v2 LUT4-LANES` inference decode engine (`runtime/c/bcir_q8_model.{h,c}` + `bcir_ai_kernels.{h,c}`) so trained `Arch-7d` checkpoints serve natively at `12.62 MiB` (`BCIRQ8 v1`) / `7.79 MiB` (`COMP-1`) total `T=4096` footprint.
