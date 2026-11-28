# BCIR Weight-Space Programming, State-Space Programming & Self-Improvement Memory (`SIM`) Performance Report & 3-Tier Hybrid Architecture Configuration

**Date:** 2026-11-27
**Status:** Executed bare-metal ISO C11 + Python ctypes empirical study linking directly against unmodified BCIR ABI 6 `runtime/c/{bcir_tensor,bcir_precision,bcir_mixer_ext}.c` (commit `7da46cfc73d16cf086fba8834b3e6baf5919419a`) paired with a Gate-0-verified stateful incremental decode, activation-contrast probe, and live State-Space Programming (`SSP-Live`) C11 companion (`<= 1.29e-5` max abs logit diff vs `bcir_ext_forward`). All scripts, per-step CSVs (`480` training steps), `18`-arm locus audit telemetry, `WSP-1` `WeightPatch` manifests, `WSP-2` health sentinel records, five full `bcir.sim.v1` JSON ledgers, `10`-seed paired CRSI logs, tracked `BCIRQ8 v1` 2M checkpoints, five figures, and SHA-256 `MANIFEST.json` are archived in [`Mecca-Research/Modular-AI` PR #231](https://github.com/Mecca-Research/Modular-AI/pull/231) under `BCIR Model Training/wsp-sim-hybrid-study/` (pinned commit `d823c202c211ac4bb31c89ef35b398984a9ac4d1`).
**Host & Toolchain Provenance (`P6`):** `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 physical cores, `avx2`, `fma`, no `avx512f`, no AMX), `Linux 6.18.57 x86_64`, `cc (Ubuntu 13.3.0)` (`-std=c11 -O3 -mavx2 -ffp-contract=off -fno-math-errno -Wall -Wextra -Wpedantic -Werror`).
**Related BCIR Roadmaps & Reports:**
- [BCIR_ARCH7D_10M_PERFORMANCE_REPORT_2026-11-27.md](BCIR_ARCH7D_10M_PERFORMANCE_REPORT_2026-11-27.md) (Base trained `BCIR-Arch7d-10M` reference model & tracked `9.87 MiB` `BCIRQ8 v1` artifact)
- [BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md](BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md) (`MIXER-EXT`, `Arch-1..7`)
- [BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md](BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md) (`TRAIN-OPT-1..10`, `WSP-1..4`, `SIM-1..3`, `P1–P6`)
- [BCIR_PRECISION_FRONTIER_AND_ARCH_STUDY_2026-11-22.md](BCIR_PRECISION_FRONTIER_AND_ARCH_STUDY_2026-11-22.md) (`50M` precision frontier, `COMP-1`, ABI 6 `1.024M` pilot)

---

## 1. Re-Reviewing MIT CSAIL Weight-Space Programming vs `BCIR-50M-Study`

Re-examining Takehi et al. (*Sparks of Weight-Space Programming*, MIT CSAIL, Nov 2, 2026) and its released logs (`RikiyaT/lm_understand-logs` `exp8_math/README.md` & `gemma_followup/README.md`) alongside our November 4 `BCIR-50M-Study` (`Modular-AI/BCIR Model Training/BCIR-50M-Study/weight_space.py`) exposes **an overlooked architectural isomorphism** and **four concrete reasons why `BCIR-50M-Study` Exp D scored `0/4` on clean-model edits**:

### 1.1 The Overlooked `Qwen3.5-0.8B` ↔ `BCIR-Arch7d` Structural Isomorphism

Inspecting `Qwen/Qwen3.5-0.8B/config.json` verbatim confirms that the model Astra successfully edited in Takehi et al. Experiment 8 is **not** a uniform GQA transformer — it is a **`3:1` hybrid (`18× linear_attention [Gated DeltaNet] + 6× full_attention`) with `full_attention_interval = 4`, `attn_output_gate = true` (`G1` SDPA sigmoid gate), and `linear_conv_kernel_dim = 4`**:

| Architectural Property | `Qwen3.5-0.8B` (`Takehi et al.` Exp 8) | `BCIR-Arch7d-10M` (`PR #230`) | `BCIR-50M-Study` (`Nov 4, 2026`) |
|---|---|---|---|
| Hybrid Sequence Mixer Layout | `3:1` `linear_attention` (GDN) + `full_attention` (`24L`) | `3:1` `GDN-2` (`L0–L2`) + global `NoPE-MLA` (`L3`) | Uniform `GQA+RoPE` (`6L`/`8L`, no hybrid split) |
| Full-Attention Stratum Indices | **`{3, 7, 11, 15, 19, 23}` (`≡ 3 mod 4`)** | **`L3` (`3 ≡ 3 mod 4`)** | None (attention smeared across every layer) |
| Output Attention Gate (`G1`) | `attn_output_gate = true` | `BCIR_GATE_G1_HEADWISE` (`L0–L3`, `8` scalars/layer) | Absent |
| Short Causal Conv1d | `linear_conv_kernel_dim = 4` | `BCIR_CANON_AC` (`k = 4` pre-mixer & pre-MLP) | Absent |
| Winning WSP Edit Locus | `p25c`: **`L15`** Head 7; `p50c`: **`L15` & `L19`** `mlp.down_proj` top-64 cols (`0.016%` params) | **`L3` (`NoPE-MLA`)** `wdown` top-64 cols (`0.326%`) & **`L3.bg` `G1`** (`3` scalars = `0.00003%`) | Arbitrary middle layer `n_layers // 2` (`L3`/`L4`), dense `99.999%` cols |

Both of Astra's winning edits in Experiment 8 (`p25c` on Layer 15; `p50c` on Layers 15 and 19, lifting MATH-500 `7.9% → 48.1%`, GSM8K `13.8% → 67.1%`, and cutting loop rate `93.0% → 33.1%`) converged **exclusively onto the deep `full_attention` layers (`15 ≡ 3, 19 ≡ 3 mod 4`)** — never onto any of the 18 linear-attention (DeltaNet) layers.

### 1.2 Four Compounding Limitations in `BCIR-50M-Study` Exp D

1. **Wrong Architecture Family:** Uniform `GQA+RoPE` lacks the recurrent-vs-global functional bottleneck (`3:1 GDN-2 + NoPE-MLA`) and lacks `G1` headwise sigmoid output gates (`L3.bg`).
2. **Wrong Diagnostic Signal:** `BCIR-50M-Study` Exp D ran 20 weight-only power iterations on a single middle `mlp.down_proj` with zero activation contrast — whereas Takehi et al. Exp 7 proved weight-only trait probing failed (`0/4`) until activation/attention probes were enabled, and Exp 8's non-looping Gemma 3 270M control (`8.5%` cut rate vs Qwen's `93.0%`) proved WSP requires a **measured, low-dimensional behavioral failure mode**.
3. **Dense Instead of Sparse Support:** Deflating `100%` of matrix columns touched `1,572,858` coordinates at 50M (`99.999%` of the matrix) versus Takehi et al. `p50c`'s top-64 columns (`0.016%` of model weights).
4. **Raw Teacher-Forced CE Selection Objective:** Selecting $\alpha \in \{0, 0.1, 0.3, 1.0\}$ strictly by minimum teacher-forced cross-entropy over `8` validation batches (`2,048` tokens) automatically rejects any edit that trades `+0.0006` nats of teacher-forced CE for major autoregressive repetition reduction.

---

## 2. Stage 1: Gate 0 Parity & 18-Arm Cross-Locus Audit on Real `BCIR-Arch7d-10M`

### 2.1 Gate 0 Verification & Organic Behavioral Failure Diagnosis

Before any edit experiment, our ISO C11 stateful incremental decoder, activation-contrast probe, and live State-Space Programming engine (`src/bcir_stateful_probe.{h,c}`) passed **Gate 0** against unmodified BCIR `bcir_ext_forward` across all three architecture families:
- **`BCIR-Arch7d-10M`** (`10.037M` trained `BCIRQ8 v1`, `GDN-2 + NoPE-MLA + G1 + QK-Norm + Canon-AC + mHC`): `max_abs_diff = 1.29e-5` (**PASS**).
- **`GQA+RoPE`** (`2L, d=128`): `max_abs_diff = 9.83e-7` (**PASS**).
- **`Arch-2 Hybrid`** (`3× DeltaNet + 1× GQA+RoPE`): `max_abs_diff = 1.56e-6` (**PASS**).
- **`WSP-2` Zero-Inference Health Sentinel** on `BCIR-Arch7d-10M`: **PASS** (`0` nonfinites, `0` outliers `> 8× median`, `wdown` max/median ratios `1.14–1.43`, `3.7 ms`).

On real WikiText-2 prefixes (`prompt_len = 48`, `gen_len = 80`), the trained `BCIR-Arch7d-10M` checkpoint exhibits an **organic `100.0%`-incidence greedy repetition-loop attractor** (`"of the second of the song of the song of the song..."`, `rep4 = 65.7%`, `distinct-4 = 34.3%` on `32` untouched held-out test prefixes = `2,560` generated bytes). Running **100% gradient-free activation-contrast probing** (`0` backward passes, strictly on `12` disjoint WikiText-2 validation `evolve` rollouts, `600` looping tokens vs `360` clean tokens, `17.2 s`) maps the internal loop mechanism across all 4 layers:

| Layer & Mixer Stratum | Residual Contrast $\|\mu_{\text{loop}} - \mu_{\text{clean}}\|_2$ | Mean `G1` Gate (Loop / Clean) | Recurrent State Echo $\cos(S_t, S_{t-1})$ | Max Global Repeat-Copy Attention Mass | Top Diagnosed Head |
|---|---:|---:|---:|---:|---:|
| `L0` (`GDN-2` recurrent) | `1.867` | `0.658` / `0.654` | `0.755` | — | `H3` |
| `L1` (`GDN-2` recurrent) | `3.080` | `0.433` / `0.439` | **`0.880`** | — | `H4` |
| `L2` (`GDN-2` recurrent) | `4.700` | `0.587` / `0.578` | `0.841` | — | `H6` |
| **`L3` (`NoPE-MLA` global)** | **`6.121` (`3.28×` vs `L0`)** | `0.404` / `0.403` | — | **`0.135` (`H0`)** | **`H0`** |

The loop signal builds inside the recurrent `GDN-2` state matrices (`cos(S_t, S_{t-1}) = 0.841–0.880` limit cycles in `L1–L2`) and amplifies monotonically into **`L3` (`NoPE-MLA`)**, where residual contrast peaks (`6.121`) and global Head `H0` locks `13.5%` attention mass onto matching prior bytes.

### 2.2 Complete 18-Arm Cross-Locus Benchmark (Strict Val Selection → Single Untouched Test Eval)

Every hyperparameter was selected strictly on WikiText-2 validation (`evolve`, `12` prefixes + `3,072` CE/KL bytes) and evaluated once on strictly disjoint WikiText-2 test (`32` prefixes = `2,560` generated bytes + `8,192` teacher-forced CE/KL bytes). Composite score (`0–100`) combines non-looping rate, `distinct-4`, and non-rep8 penalized exponentially by `max(0, ΔCE)` and `max(0, KL - 0.015)`:

| # | Arm | Architectural Tier / Locus | Test Score ↑ | Loop % ↓ | Rep-4 % ↓ | Dist-4 ↑ | Test `ΔCE` | Test `KL` ↓ | Top-1 Agree | Edited Coords (`%`) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | `Baseline_Clean` | Reference (`BCIRQ8 v1`) | `20.18` | `100.0%` | `65.7%` | `34.3%` | `+0.0000` | `0.00000` | `1.0000` | `0` (`0.00000%`) |
| 02 | `Control_50M_ExpD_BlindSVD_L2` | 50M Exp D (`L2` dense `α=1.0`) | `32.80` | `100.0%` | `45.9%` | `54.1%` | `+0.0006` | `0.00295` | `0.9597` | `475,136` (`4.734%`) |
| 03 | `WSP_L0_GDN2_wdown_SparseTopK` | Tier 2 WSP (`L0 k=128 α=1.2`) | `18.57` | `100.0%` | `67.5%` | `32.6%` | `+0.0027` | `0.00348` | `0.9613` | `65,536` (`0.653%`) |
| 04 | `WSP_L1_GDN2_wdown_SparseTopK` | Tier 2 WSP (`L1 k=64 α=1.4`) | `17.14` | `100.0%` | `70.3%` | `29.7%` | `-0.0000` | `0.00234` | `0.9689` | `32,768` (`0.326%`) |
| 05 | `WSP_L2_GDN2_wdown_SparseTopK` | Tier 2 WSP (`L2 k=128 α=1.2`) | `27.23` | `100.0%` | `54.9%` | `45.1%` | `+0.0012` | `0.00366` | `0.9630` | `65,536` (`0.653%`) |
| **06** | **`WSP_L3_MLA_wdown_SparseTopK`** | **Tier 2 WSP (`L3 k=64 α=1.4`)** | **`27.63`** | `100.0%` | **`53.1%`** | **`46.9%`** | `+0.0076` | `0.01175` | `0.9368` | **`32,768` (`0.326%`)** |
| 07 | `WSP_L3_MLA_wdown_DenseRank1` | WSP Ablation (`L3 k=928 α=0.4`) | `24.45` | `100.0%` | `59.3%` | `40.7%` | `+0.0007` | `0.00214` | `0.9670` | `475,135` (`4.734%`) |
| **08** | **`WSP_L3_MLA_G1_HeadGate`** | **Tier 2 WSP (`L3.bg H0,2,3 -1.2`)** | **`21.15`** | `100.0%` | `63.9%` | `36.1%` | `+0.0002` | **`0.00031`** | `0.9861` | **`3` (`0.00003%`)** |
| 09 | `WSP_L3_MLA_HeadOut_Scale` | Tier 2 WSP (`L3.wo H0 ×0.0`) | `20.66` | `100.0%` | `64.8%` | `35.1%` | `-0.0003` | `0.00039` | `0.9839` | `32,430` (`0.323%`) |
| 10 | `SSP_Weight_GDN2_StateGates` | Tier 2 SSP-W (`L0-2 ba/be`) | `19.12` | `100.0%` | `67.4%` | `32.6%` | `+0.0004` | `0.00029` | `0.9883` | `192` (`0.0019%`) |
| 11 | `WSP_CanonAC_Taps` | Tier 2 WSP (`Canon-A/C ×0.85`) | `19.25` | `100.0%` | `66.0%` | `34.0%` | `+0.0048` | `0.00341` | `0.9628` | `12,133` (`0.121%`) |
| 12 | `WSP_mHC_Streams` | Tier 2 WSP (`mhc_hres Δ=-0.4`) | `20.18` | `100.0%` | `65.7%` | `34.3%` | `-0.0000` | `0.00000` | `1.0000` | `16` (`0.00016%`) |
| **13** | **`SSP_Live_GDN2_EchoDamp`** | **Tier 1 SSP (`S_t` damp + erase)** | **`30.48`** | **`96.9%`** | **`51.9%`** | **`48.1%`** | **`+0.0000`** | **`0.00000`** | **`1.0000`** | **`0` (`0.00000%`)** |
| 14 | `SSP_Live_MLA_RepeatBias` | Tier 1 SSP (`L3` repeat bias) | `20.31` | `100.0%` | `65.5%` | `34.5%` | `+0.0000` | `0.00000` | `1.0000` | `0` (`0.00000%`) |
| **15** | **`Harness_Decoding_NgramGuard`** | **Tier 1 Harness (`ng4=5.0, p=.15`)** | **`78.93`** | **`40.6%`** | **`6.7%`** | **`93.3%`** | **`+0.0000`** | **`0.00000`** | **`1.0000`** | **`0` (`0.00000%`)** |
| 16 | `Hybrid_Tier2WSP_plus_Tier1SSP` | Hybrid (`L3.wdown + SSP + ng5`) | `65.80` | `59.4%` | `12.3%` | `87.7%` | `+0.0076` | `0.01175` | `0.9368` | `32,768` (`0.326%`) |
| 17 | `Control_Random_MatchedNorm` | `WSP-3` Ctrl (`L3` rand, `L2=.582`) | `20.17` | `100.0%` | `65.7%` | `34.3%` | `+0.0001` | `0.00005` | `0.9962` | `32,768` (`0.326%`) |
| 18 | `Control_SGD_MicroFT_8steps` | `WSP-3` Ctrl (`8st Q8-Muon, 26.2s`) | `34.51` | `96.9%` | `44.0%` | `56.0%` | **`-0.0067`** | `0.00494` | `0.9427` | `10,036,499` (`100%`) |

### 2.3 Five Concrete Engineering Laws From Stage 1

1. **Activation-guided sparse `L3.wdown` works directionally — matched-norm random has zero effect:** Arm 06 (`L3 NoPE-MLA` top-64 columns, `0.326%` params) improves score **`+7.45`** (`20.18 → 27.63`, `rep4 65.7% → 53.1%`), strictly outperforming early recurrent layers `L0` (`18.57`, worse) and `L1` (`17.14`, worse), outperforming **dense `L3.wdown` (`24.45` at `14.5×` more weights)** using the exact same activation vector $u_3$, and beating the **matched-norm random control (`20.17`, `-0.01`)** cleanly.
2. **`BCIR-Arch7d`'s `G1` headwise output gate (`L3.bg`) is an ultra-sparse `10,810×` upgrade over `p25c` head zeroing:** Modifying just **`3` scalar biases (`0.00003%` of model params)** in `L3.bg` outperforms zeroing `32,430` columns of `L3.wo` (`21.15` vs `20.66`, lower KL `0.00031` vs `0.00039`).
3. **Why static `GDN-2` gate weight edits fail while adaptive live `SSP` succeeds:** Permanently shifting `GDN-2` `ba_chan`/`be_chan` weights (Arm 10) degrades clean working memory (`19.12 < 20.18`). Applying the **exact same state-damping / erase-boost intervention adaptively at inference time (`SSP-Live`, Arm 13)** — triggered only when a 4-gram echo occurs — leaps to **`30.48` (`+10.30` pts, `0` weight mutations, `0.0000` KL)**, beating every single-layer sparse WSP weight edit.
4. **Structural traps (`Canon-AC`, `mHC`, `L0–L1` FFN) & why `50M Exp D` appeared inert:** Scaling `Canon-AC` delayed taps degrades both behavior (`19.25`) and CE (`+0.0048`); DeepSeek `mHC` Birkhoff doubly-stochastic streams (`Arm 12`) are structurally invariant (`20.18`, `KL=0.00000`). Re-evaluating `50M Exp D` blind SVD (`Arm 02`) under a behavioral repetition objective instead of raw teacher-forced CE shows `α=1.0` actually lowers `rep4` (`65.7% → 45.9%`, `32.80`), confirming that `50M Exp D`'s raw-CE selection gate rejected behaviorally active edits — yet blind SVD still mutates `475,136` coordinates (`4.73%`) and leaves `100%` of sequences looping.
5. **Do not stack persistent weight mutations onto ephemeral decoding attractors — and never mistake WSP for pretraining:** Adaptive `Tier 1 NgramGuard` (Arm 15) reaches **`78.93` (`loop 100% → 40.6%`, `dist4 93.3%`, `0` KL)**, beating naive hybrid stacking (Arm 16, `65.80`) which paid `L3.wdown`'s `+0.0076` CE / `0.01175` KL tax unnecessarily. Meanwhile, just 8 native C11 `Opt_Q8_Muon` steps (`26.2 s`, Arm 18) both **improved held-out CE (`-0.0067` nats)** and reached `34.51`, outperforming every static WSP edit.

---

## 3. Stage 2: Matched 3-Architecture Training & Editability Localization (`163,840` Tokens)

Training three parameter-matched `~2.6M` 4-layer models (`d=256, h=4, kvh=2, B=8, T=128`, `160` steps = `163,840` WikiText-2 bytes, `Opt_Q8_Muon_BF16Grad`, identical batch order & cosine LR `4e-3 → 4e-4`) directly compares uniform `GQA` against `3:1` hybrids for BOTH language modeling and post-training WSP/SSP editability:

| Architecture | Params | Pers. RAM | `T=4096` State | `tok/s` | Val PPL | Test PPL | Peak `\|Act\|` | `L0` ΔScore | `L1` ΔScore | `L2` ΔScore | `L3` ΔScore | `SSP-Live` ΔScore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Arch7d_2M`** (`3×GDN2 + NoPE-MLA + G1 + Canon + mHC`) | `2,671,504` | `18.40 MiB` | **`1.38 MiB`** | `1,142` | **`6.70`** | **`6.84`** | **`1.50`** | **`+6.06`** | `-3.74` | `-4.49` | `-4.11` | **`+2.76`** (`0` KL) |
| **`Arch2_Hybrid_2M`** (`3×DeltaNet + GQA+RoPE`) | `2,629,912` | `18.05 MiB` | `4.38 MiB` | `1,334` | `6.77` | **`6.84`** | `1.94` | `+0.07` | `+0.12` | `-1.21` | **`+3.19`** | **`+6.02`** (`0` KL) |
| **`GQA_Full_2M`** (`4×GQA+RoPE`) | `2,722,048` | `18.66 MiB` | `16.00 MiB` | `1,527` | `8.09` | `8.16` | `2.02` | `-7.77` | `-2.77` | `-0.55` | `-3.54` | `+0.56` (`0` KL) |

**Two Structural Takeaways:**
1. **Uniform `GQA_Full_2M` fails WSP on `0/4` layers (`-7.77` to `-0.55`) and yields negligible `SSP-Live` gain (`+0.56`):** Because uniform GQA lacks a recurrent state-space stratum (`GDN-2`/`DeltaNet`) and distributes attention diffusely across all 4 layers, sparse single-layer edits fail everywhere — reproducing `BCIR-50M-Study`'s negative result on uniform `GQA`. Both `3:1` hybrids (`Arch7d_2M`, `Arch2_Hybrid_2M`) cut test PPL by `-1.32` (`6.84` vs `8.16`), shrink `T=4096` state `3.65–11.6×`, and support clean sparse WSP (`+3.19` to `+6.06`) and zero-KL `SSP-Live` (`+2.76` to `+6.02`).
2. **Cross-Scale Locus Shift & Established `COMP-1` Regime Shift:** On early-trained `Arch7d_2M` (`164k` tokens), the highest-leverage WSP layer sits at `L0` (`+6.06`), whereas at `BCIR-Arch7d-10M` (`1.114M` tokens) it shifts cleanly to the global `L3 NoPE-MLA` anchor (`+7.45`). Similarly, quantizing `BCIR-Arch7d-10M` MLPs via BCIR's established `COMP-1` Dual-SVID + Joint-ITQ signed-residual 1-bit PTQ (`r=16, FP16 scales`, bit-exact parity `max_diff = 0.0` with `arch7d-10m-training-study/src/common_10m.py`, `5.04 MiB` weights, `5.14 s`) shifts test `CE 1.5175 → 2.1575` (`PPL 4.56 → 8.65`, `+0.6399` nats) while altering greedy loop dynamics (`loop 100.0% → 90.6%`, `dist4 34.3% → 67.3%`). These real regime shifts directly motivate **Tier 3 Self-Improvement Memory (`SIM`)**.

---

## 4. Stage 3: Google Self-Improvement Memory (`SIM`, `arXiv:2611.18856`) Across Real BCIR Shifts

We implemented BCIR's problem-grouped **`SIM` Ledger (`bcir.sim.v1`)** following Yu et al. (`arXiv:2611.18856` §3.2 & Appendix D.1–D.3): `SIM` sits on the **outer-loop optimizer (Proposer + Selector)** rather than inside the policy context window, recording `{problem_id, root_cause_hypothesis, action_family, locus, rationale, predicted_outcome [pre-eval], observed_outcome [fixed/missed/regressed/KL], accepted, provenance}` and retaining **all failed and regressing attempts**.

We ran strictly matched **4-condition CRSI ablations** (`Cold_Start (H_0, ∅)`, `Harness_Transfer (H_S, ∅)`, `Problem_Only (H_S, M_S^c)`, and `Full_SIM (H_S, M_S)`) with identical round/candidate budgets, **strictly isolated cold evaluation caches per condition**, and strict WikiText-2 validation (`evolve`) vs untouched test (`heldout`) isolation across **5 real BCIR continual shifts** plus an **`n = 10` paired-seed reproducibility study**:

### 4.1 Five Real BCIR Continual Transitions (Untouched Held-Out Target Score, Source Retention & Cold Wall)

| Transition | Continual Shift | `Cold_Start` | `Harness_Transfer` | `Problem_Only` | **`Full_SIM`** | `SIM − Transfer` | Loop (`SIM` / `Tr`) | Source Retention (`SIM` / `Tr`) | Cold Wall (`SIM` / `Tr`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `T1a` (`10M Prose → Structured`) | Cross-Dataset | `94.12` | `95.27` | `95.46` | **`99.47`** | **`+4.20`** | **`0%`** / `8%` | `95.52` / `99.33` | `72.1s` / `68.0s` |
| `T1b` (`10M Prose → LongCtx`) | Cross-Dataset | `93.03` | `99.37` | `99.37` | **`99.37`** | `+0.00` | **`0%`** / `0%` | `99.52` / `99.52` | `84.0s` / `83.1s` |
| `T2a` (`Arch7d 2M → 10M`) | Cross-Scale | `27.79` | `98.95` | `98.95` | **`98.95`** | `+0.00` | **`0%`** / `0%` | `99.57` / `99.57` | `65.1s` / `67.6s` |
| `T2b` (`GQA-2M → Arch7d-10M`) | **Cross-Family** | `30.44` | `91.84` | `89.58` | **`98.09`** | **`+6.24`** | **`0%`** / `8%` | **`78.28`** / `8.41` | **`65.6s`** / `89.9s` (`1.37×`) |
| `T3` (`10M Q8 v1 → COMP-1 1b`) | **Cross-Precision** | `53.23` | `47.42` | `43.45` | **`95.07`** | **`+47.65`** | **`8%`** / `92%` | **`97.45`** / `21.26` | **`82.7s`** / `106.1s` (`1.28×`) |
| **5-Shift Mean** | — | `59.72` | `86.57` | `85.36` | **`98.19`** | **`+11.62`** | **`1.7%`** / `21.7%` | **`94.07`** / `65.62` | **`73.9s`** / `82.9s` (`1.12×`) |

- **`Full_SIM` placed #1 (or tied #1 at the `99+` ceiling) on `5/5` real BCIR transitions**, cutting mean held-out loop rate **`21.7% → 1.7%`** (`0%` on 4/5 shifts, `8%` vs `92%` on `T3`), raising mean source retention **`65.62 → 94.07` (`+28.45` pts)**, and lowering mean cold-cache wall time (`73.9s` vs `82.9s` Transfer / `94.7s` Cold Start).
- On the **Cross-Family policy shift (`T2b`: `GQA-2M → Arch7d-10M`, direct BCIR analogue of Opus → Gemini in `arXiv:2611.18856` §4.3.1)**, `Harness_Transfer` degraded below its own Round-0 transfer (`93.41 → 91.84`) and suffered catastrophic source collapse (`ret = 8.41`), while `Problem_Only` dropped further (`89.58`, `-2.26` below Transfer) — replicating `arXiv:2611.18856`'s finding that diagnosis labels without rationale + predicted/observed failure outcomes are insufficient. `Full_SIM` jumped to **`98.09` (`+6.24` over Transfer, `+8.51` over Problem-Only)** with **`9.3×` stronger source retention (`78.28` vs `8.41`)** and **`1.37×` faster cold-cache runtime (`65.6s / 13 evals` vs `89.9s / 15 evals`)** because it avoided weight-mutating candidates requiring extra teacher-forced CE/KL passes.
- On the **Cross-Precision deployment shift (`T3`: `BCIRQ8 v1 → established COMP-1 1-bit`)**, blind `Harness_Transfer` (`47.42`, `92%` loops) and `Problem_Only` (`43.45`, `100%` loops) both suffered **negative transfer** below `Cold_Start` (`53.23`). Using its failure/KL memory and precision-regime provenance, `Full_SIM` rescued adaptation to **`95.07` (`+47.65` over Transfer, `+51.62` over Problem-Only, `-0.0267` nats `COMP-1` CE recovery, `4.58×` stronger source retention `97.45` vs `21.26`, and `1.28×` faster cold-cache runtime `82.7s / 16 evals` vs `106.1s / 18 evals`)**.

### 4.2 `n = 10` Paired-Seed Statistical Reproducibility & Honest Small-Split Variance Limit (`S10`)

Running `10` independent source→target seeds on `Arch7d_2M` (`D1 Prose → D2 Structured`, `n = 6` Evolve prompts, `12` Held-Out prompts):
- **Condition Ordering:** **`Full_SIM` (`81.71`)** > `Problem_Only` (`76.73`) > `Harness_Transfer` (`72.79`) > `Cold_Start` (`69.46`).
- **Paired Statistics (`Full_SIM − Harness_Transfer`):** `6/10` wins (including `+41.12` and `+41.66` rescues on pairs 5 and 9 where Transfer and Problem-Only collapsed to `100%` loops), mean **`+8.92` pts**, `std = 20.27`, `95% CI [-5.58, +23.42]`, `t(9) = 1.392`, two-sided `p = 0.197`, `Cohen's d_z = 0.44`.
- **Source Retention (`D1`):** **`Full_SIM = 85.12`** vs `Harness_Transfer = 75.39` (**`+9.74` pts**).
- **Trap-Locus Budget Suppression:** `Full_SIM` reduced wasted evaluations on `Canon-AC` by **`2.6×`** (`14.9% → 5.7%`), `mHC` by **`2.1×`** (`7.9% → 3.8%`), and static `GDN-2` gates by **`2.0×`** (`12.9% → 6.7%`), redirecting **`3.7×`** more budget (`37.1%` vs `9.9%`) to the zero-KL winning mechanism.
- **Honest Scope Disclosure:** With only `n = 6` Evolve prompts, a single sequence flipping between looping and non-looping swings Evolve score by `±7.5` points, preventing the 10-seed paired test from clearing $p < 0.05$ (`p = 0.197`). This empirically validates BCIR's `WSP-3` / `SIM` governance rule: production `SIM` loops must pair problem-grouped ledgers with adequately sized Evolve splits (`>= 30–50` tasks) and per-trace semantic proposers (as in `arXiv:2611.18856`).

---

## 5. Architecture Blueprint: Where State-Space Programming, Weight-Space Programming & `SIM` Fit in BCIR

Synthesizing Stages 1–3 with `BCIR-Arch7d-10M` (merged `PR #230` `8039838`), `BCIR-50M-Study` (Nov 4, 2026), and `COMP-1` (`PR #228`) locks in BCIR's **3-Tier Self-Improving Hybrid Architecture**:

| Architectural Tier | Mechanism & Artifact | Scope & Collateral KL | Optimal BCIR Hybrid Locus | Governance Contract |
|---|---|---|---|---|
| **Tier 1: Live State-Space Programming (`SSP-Live`) & Decoding Harness** | Adaptive online `GDN-2` FP64 state damping (`S_t`), `alpha`/`erase` runtime shifts, `NoPE-MLA` repeat-key logit bias, and n-gram/presence decoding guard | Per-session / ephemeral (**`0` weight mutations, `ΔCE = 0.0000`, `KL = 0.0000` on clean text**) | **Recurrent `L0–L2` `GDN-2` state `S_t`** (breaks limit cycles, `+10.30` pts) + **Decoding Logit Guard** (`+58.75` pts, `loop 100% → 40.6%`) | Preferred first-line control for runtime/decoding attractors; deterministic C11 (`<= 1.29e-5` parity) |
| **Tier 2: Surgical Weight-Space Programming (`WSP-1..4`)** | Activation-contrast sparse top-$k$ `wdown` deflation, `G1` scalar head-gate biases, `WSP-2` outlier norm repair, `COMP-1` Scale-Factor calibration, and `WSP-4` compiled symbolic circuits | Persistent across sessions (`0.00003%–0.33%` coords, bounded `KL <= 0.015`, `inference_only` unless gauge-transported) | **Global `NoPE-MLA` Anchor Layers (`l ≡ 3 mod 4`, i.e., `L3` in 4L / `L3,L7` in 8L 50M):** `L{l}.wdown` top-$k$ cols (`+7.45`) & **`L{l}.bg` `G1` gates (`3` scalars)**; `COMP-1` MLP scales. **NEVER `Canon-AC`, `mHC`, static global `GDN-2` gates, or blind dense SVD.** | `WSP-1` `bcir.wsp1.weight_patch.v1` SHA-256 + `WSP-2` sentinel + `WSP-3` held-out KL ceiling, matched-norm random & SGD controls |
| **Tier 3: Outer-Loop Meta-Memory (`SIM-1..3`, `arXiv:2611.18856`)** | Problem-grouped `bcir.sim.v1` JSON ledger read by the **Proposer + Selector** (NOT policy context) across dataset, scale (`2M → 10M → 50M`), family (`GQA → Arch7d`), and precision (`BCIRQ8 → COMP-1`) shifts | Persistent across continual optimization cycles (`+4.2` to `+47.7` pts across shifts, `1.28–1.37×` cold-cache speedup on hard shifts, `2.0–2.6×` trap suppression, up to `9.3×` higher source retention) | **Outer-Loop CRSI Orchestrator** governing BOTH Tier 1 (`SSP-Live` / decoding) AND Tier 2 (`WSP` `WeightPatch` / `COMP-1` scale calibration) | Strict Evolve (`>= 30` tasks) vs Held-Out split isolation; pre-eval `predicted_outcome`; zero deletion of failed/regressing attempts |

### Recommended Configuration Update for Next-Stage `BCIR-Arch7d-10M` / `50M` Builds

1. **Preserve the `3:1` Hybrid Stratum (`GDN-2 × 3 + NoPE-MLA × 1` with `G1` + `QK-Norm` + `Canon-AC` + `mHC C=2`)** across both `10M` (`L=4`, global anchor `L3`) and `50M` (`d=768, L=8`, global anchors `L3, L7`) — it simultaneously maximizes language quality, `4×` zero-shot extrapolation, `T=4K` memory efficiency, and dual-tier `SSP-Live` + `L{3,7}` WSP controllability.
2. **Lower `bcir_stateful_probe`'s `GDN-2` + `NoPE-MLA` + `SSP-Live` echo-damping hooks into BCIR's Phase M3 native C11 inference runtime** (`runtime/c/bcir_q8_model.{h,c}` / `bcir_llama.c`) so production serving gains zero-KL runtime loop suppression on `BCIRQ8 v1` and `COMP-1` deployments.
3. **Standardize `bcir.wsp1.weight_patch.v1` and `bcir.sim.v1` under BCIR's hosted toolchain** (alongside `bcir/frontends/models/deploy_policy.py`), restricting persistent `WSP` edits to diagnosed global-attention anchors (`l ≡ 3 mod 4`) and `COMP-1` quantization recovery while routing runtime attractors to Tier 1 `SSP-Live` / decoding harnesses and exact deterministic logic to `WSP-4` / `SLM-6` symbolic engines.

---

## 6. Primary References

- Takehi et al., *Sparks of Weight-Space Programming* (MIT CSAIL, Nov 2, 2026): https://rikiyat.github.io/blog/weight-space-programming/ and logs https://github.com/RikiyaT/lm_understand-logs
- Yu et al., *Self-Improvement Memory Enables Continual Recursive Self-Improvement* (Google Cloud AI / Google DeepMind / PSU, Nov 13, 2026): https://arxiv.org/abs/2611.18856
- Qwen Team, `Qwen/Qwen3.5-0.8B` architecture configuration (`full_attention_interval: 4`, `attn_output_gate: true`, `linear_conv_kernel_dim: 4`): https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json
- Prior `BCIR-50M-Study` (Nov 4, 2026): `Mecca-Research/Modular-AI` `BCIR Model Training/BCIR-50M-Study/`
- Pinned WSP / SSP / SIM Hybrid Study Archive (`wsp-sim-hybrid-study`, PR #231, commit `d823c202c211ac4bb31c89ef35b398984a9ac4d1`): https://github.com/Mecca-Research/Modular-AI/tree/d823c202c211ac4bb31c89ef35b398984a9ac4d1/BCIR%20Model%20Training/wsp-sim-hybrid-study
