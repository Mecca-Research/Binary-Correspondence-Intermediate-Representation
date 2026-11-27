# BCIR Floating-Point Precision Frontier (50M), Region-Policy `COMP-1`, and `MIXER-EXT` Comparative Architecture Study

**Date:** 2026-11-22 (updated 2026-11-24 with production `COMP-1` / `P1–P6` & Exp 5 baseline; updated 2026-11-27 with production ABI 6 `MIXER-EXT`, `Q8-Muon`, beyond-the-list `Arch-7` ladder, and `1.024M`-token `T=256` pilot)
**Status:** Executed bare-metal C11 + Python ctypes empirical studies; all raw logs, Gate 0 proofs, multi-seed curves, checkpoints (`<= 24 MiB`), SHA-256 provenance manifests, multi-panel figures, and permanent `bcir.deploy_baseline.v1` checkpoint archived in [`Mecca-Research/Modular-AI`](https://github.com/Mecca-Research/Modular-AI) under `BCIR Model Training/precision-arch-study/` (merged [PR #228](https://github.com/Mecca-Research/Modular-AI/pull/228) at `8cc3f234be0bc84009b79f152ab55995ada9a499`) and `BCIR Model Training/mixer-ext-abi6-study/` ([PR #229](https://github.com/Mecca-Research/Modular-AI/pull/229) at `e458a859e190430827319b0e725b325241fbd787`).
**Host & Toolchain Provenance (`P6`):** Exp 1–4: `Intel(R) Xeon(R) 6985P-C` (2 cores, `avx512f`, `avx512_bf16`, `avx512_fp16`, `avx512_vnni`, no AMX exposed), `clang 23.1.6` (`-O3 -std=c11 -march=native -ffp-contract=off -Wall -Wextra -Werror`), linked against BCIR `runtime/c/bcir_tensor.c` and `runtime/c/bcir_decoder_train.c` at commit `15c7287e93a968795cf27c31caa7ee5e79cd48f9`. Exp 5 & §8 (ABI 6 `MIXER-EXT`): `Intel(R) Xeon(R) CPU @ 2.30GHz` (2 cores, `avx2_fma`, no `avx512f`), `clang 23.1.6`, linked directly against production BCIR `runtime/c/{bcir_q8_model,bcir_decode,bcir_ai_kernels,bcir_llama,bcir_tensor,bcir_precision,bcir_mixer_ext}.c`.
**Related BCIR Roadmaps:**
- [BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md](BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md) (`TRAIN-OPT-1..10`, Precision Policy v0.1 `P1–P6`)
- [BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md](BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md) (`MIXER-EXT`, `Arch-1..6`, `COMP-1`)
- [BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md](BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md) (LittleBit / LittleBit-2 sub-1-bit QAT and packed LUT4 microbenchmarks)

---

## 0. Executive Summary: Empirical Answers to the Six Research Questions

1. **Optimizer-state storage dominates training RAM more than inference weight precision.** At `50.54M` parameters (`wide50m`), standard FP32 AdamW requires `771.20 MiB` (`16.00 B/param` = `192.8 MiB` FP32 weights + `578.4 MiB` gradients and moments). Compressing optimizer states + gradients via `Opt_Q8_BF16Grad` (`8.05 B/param`, `387.86 MiB`) saves **`383.34 MiB` (`-49.7%`)** — nearly **`2.0×` the entire FP32 weight matrix** — while staying within **`-0.0009` nats** of FP32 on untouched WikiText-2 test CE at `50.54M` (`2.4603` vs `2.4613`), **`+0.0001` nats** at `6.16M`, and **`+0.0024` nats** at `0.86M` with zero nonfinite events. By contrast, compressing `50.54M` inference weights all the way from FP32 (`192.4 MiB`) to `0.32 BPW` (`2.33 MiB`) saves `190.1 MiB` (half as much RAM) and incurs `+1.0415` nats without QAT. On CPUs without hardware matrix units, `Opt_Adafactor` (`12.01 B/param`, saves `192.33 MiB`) is the overall Pareto winner, outperforming FP32 at equal tokens (`-0.0106` nats at `50.54M`) while tracking FP32 wall speed (`825.6` vs `733.5 ms/step`).
2. **Completed `BF16` / `FP8` / `FP4` precision frontier at `50.54M` exposes equal-token vs equal-time separation and confirms `P1`–`P3`.** Every tested compute format (`BF16_NDT`, `BF16_VDP`, `FP8 E4M3/E5M2`, `MXFP4 E2M1`, `Asym MXFP4/FP8`, and region-mixed policies) converges with zero nonfinite events and clears the `P4` `+0.04` nat (`+4.08%` PPL) gate at `50.54M` under equal tokens (`Delta CE` in `[-0.0106, +0.0095]` nats). However, on a non-AMX host (`Xeon 6985P-C`), `fp32_kblock` (`273.3 GF/s` 2T) outperforms `bf16_ndt` (`161.2 GF/s`) and `bf16_vdp_avx512` (`121.0 GF/s`), so at **equal wall time (`94.8 s`)** `FP32_Ref` (`2.4225` val CE) and `Opt_Adafactor` (`2.4400`) beat every emulated low-precision compute path (`2.4673–2.6524`). Silicon probing across both Sapphire Rapids and Granite Rapids pins the hardware `VDPBF16PS` contract to `ftz(fmaf(a0, b0, ftz(fmaf(a1, b1, acc))))` (`0` bit mismatches), validating `Rule P1` (never auto-enable BF16 from a CPU feature bit alone).
3. **Attention and MLP must be governed as separate quantization-policy regions in both training and deployment.** During narrow-width training (`pilot1m`, `d=192`), `Compute_MXFP4_Uniform` fails the `+0.04` nat gate (`+0.0481` nats), whereas keeping Attention in FP32 (`Region_FP32Attn_MXFP4MLP`: `+0.0141` nats) or FP8 (`Region_FP8Attn_MXFP4MLP_Q8Opt` at `8.05 B/param`: `+0.0138` nats) recovers `>70%` of the error. At `50.54M`, keeping Attention in FP32 cuts `Rule P3` conditional FP64 guard triggers **`36.4×`** (`364 → 10`). During deployment at `50.54M`, `BCIRQ8_Uniform` (`8.25 BPW`, deterministic lossy with `|w - w_hat| <= 2^{e-1}` and `0.0` bound violations) holds `Delta CE = -0.0002` at `46.4 tok/s` (`3.15×` FP32); compressing MLPs to Packed-1b LUT4 (`COMP-1`) drops artifact size to `14.07 MiB` (`164.8 tok/s`, `11.2×` FP32), while additionally binarizing Attention (`LB_Uniform`) adds `+0.3366` nats of extra damage. On `wedge128_kv4_strong`, uniform LittleBit PTQ collapses `kv_single` from `100.0%` to `73.4%` (reaching only `92.2%` after 160-step KD-QAT), whereas `COMP-1` (`BCIRQ8` Attention + LittleBit MLP) recovers **`100.0%`** exact recall at `3.2–4.8×` lower KL (`0.0127–0.0192`).
4. **Matrix width and aspect ratio — not total parameter count — govern the sub-1-bit storage floor, yet a low floor never guarantees accuracy.** Matching `wide_shallow_6m` (`d=512, ff=1536, L=2`, `6,162,944` params) against `narrow_deep_6m` (`d=256, ff=736, L=8`, `6,164,736` params — `0.029%` parameter difference) at identical relative rank fraction (`9.375%/path`), `wide_shallow_6m` achieves **`27%` lower realized BPW** (`0.37` vs `0.47 BPW`), `2.02×` lower `r=1` scale floor (`0.0443` vs `0.0896 BPW`), `1.69×` faster decode (`3,637` vs `2,146 tok/s`), and `-0.141` nats lower `COMP-1` PTQ error. Nevertheless, across all four scales (`0.53M → 50.54M`), even as the MLP `r=1` storage floor plunges `6.6×` (`0.200 → 0.030 BPW`), **90%-energy spectral rank remains `61–73%` of `min(m, n)`** (`stable_rank = 216.1` at `50.54M`), causing `+0.43` to `+1.04` nats PTQ collapse without QAT.
5. **Low-bit weight compression (`COMP-1`) and KV-cache optimization (`Arch-4 MLA` / `Arch-2 Hybrid-SWA` / `Arch-1 DeltaNet`) are orthogonal, composable problems.** In the `wide10m` `2×2` matrix at `T=4096`, applying `COMP-1` to `GQA_Full` shrinks weights `11.9×` (`37.26 → 3.14 MiB`) but leaves `8.00 MiB` of uncompressed KV cache (`71.8%` of total `11.14 MiB` inference RAM). Composing `COMP-1` with `Arch-4 MLA` (decoupled RoPE, exact global context) cuts KV cache `1.60×` vs 4:1 GQA (`5.00 MiB`, `8.98 MiB` total; `3.20×` vs 3:1 GQA at `50M`); composing with `Arch-2 Hybrid-SWA` cuts KV cache `3.90×` (`2.05 MiB`, `5.19 MiB` total — `8.72×` vs FP32 GQA); composing with `Arch-1 DeltaNet` eliminates KV cache completely (`0.00 MiB` KV + `0.50 MiB` $O(1)$ recurrent state, `5.22 MiB` total).
6. **`TRAIN-OPT-1/2/4` and `MIXER-EXT` establish a decisive new architecture lead for `Arch-2 Hybrid-DN+GQA` and `Arch-1 DeltaNet`.** `TRAIN-OPT-1` (`mup_v1`) lowers held-out PPL at all three scales (`10.54 → 9.69` on `pilot1m`, `11.31 → 10.60` on `wide512`, `15.71 → 14.31` on `wide50m`), while `grad_clip=3.0` drops post-warmup clip saturation from `100.0%` to `0.0–42.6%`. `TRAIN-OPT-2` delivers `2.54×` real speedup (`max_loss_diff = 3.02e-7`). Across `MIXER-EXT` on `wide10m` WikiText-2 (`2` seeds), **`Arch-2 Hybrid-DN+GQA`** (`3:1` DeltaNet + Full GQA) leads all models at **`9.26` PPL (`-0.1483` nats / `-13.8%` PPL vs `GQA_Full` `10.74`)** while also achieving the highest length extrapolation (`ext6 = 33.6%`). Pure **`Arch-1 DeltaNet`** achieves **`9.35` PPL (`-0.1386` nats vs `GQA_Full`)** with `0.00 MiB` KV cache and **`3.8×` higher long-filler retention (`44.0%` vs `11.5%` `SWA_Only`)** past the SWA horizon (`filler=20 > W=16`). **`Arch-2 Hybrid-SWA`** beats `GQA_Full` on both seeds (`10.65` vs `10.74` PPL) at faster step latency (`157.3` vs `159.6 ms/step`).

**Scope & Token-Exposure Caveat:** All training experiments here (`30.7k–76.8k` tokens per run; and even subsequent `50M–100M`-token runs) are **initial research qualification gates** measuring numerical stability, optimizer/compute tradeoffs, region sensitivity, and architectural inductive bias. They are orders of magnitude below converged general-purpose LLM pretraining exposure (`>= 20` tokens/parameter) and do not establish final converged scaling exponents.

---

## 1. Gate 0 Verification & `P1` Numerical Contracts

Before running any training or deployment benchmark, `build/test_gate0_all` (`results/gate0_verify_output.txt`) plus Python whole-model deploy parity (`bcir_deploy_prefill_seq` vs `bcir_ext_forward`) passed **15/15** C11 checks and **6/6** deploy parity checks with `-Wall -Wextra -Werror` and `-ffp-contract=off`:

| Verification Gate | Checked Invariant | Measured Result | Verdict |
|---|---|---:|---|
| `[VDP]` `VDPBF16PS` portable vs `AVX512_BF16` (`64×128×96`) | `ftz(fmaf(a0,b0, ftz(fmaf(a1,b1, acc))))` bit-identity | `0` bit mismatches | **PASS** |
| `[WY]` `Arch-1 DeltaNet` Chunkwise WY (`C=16`, `FP64`) vs token recurrence (`T=73`) | Exact state & output identity | `max_abs = 0.000e+00` | **PASS** |
| `[MLA]` `Arch-4 MLA` absorbed decode vs unabsorbed attention (`T=32`) | Latent absorption $q_{\text{abs}} = W_{UK}^\top q^C$ | `max_abs = 2.067e-08` | **PASS** |
| `[BCIR]` `MIXER-EXT` GQA vs unmodified `bcir_decoder_train.c` | Forward loss & backward gradients (`kblock+attn+reduce`) | `loss_diff = 0.000e+00`, `grad_diff = 0.000e+00` | **PASS** |
| `[FD]` Central finite-diff AD (`GQA`, `SWA`, `DeltaNet`, `Hybrid-SWA`, `MLA`) | `60–99` coordinates across `20–33` tensors per family | `max_abs <= 3.70e-5`, `max_rel <= 1.72e-2` | **PASS** |
| `[OPT2]` All-masked refusal & physical suffix trimming (`10/16`) | Refuse `sup==0`; bit-identical trimmed vs padded grads | `refuse=1`, `loss_diff = 0.000e+00`, `grad_diff = 0.000e+00` | **PASS** |
| `[THR]` 1T vs 2T hybrid step (`SWA + DeltaNet + MLA`) | Deterministic thread-pool partition invariance | `0` bit diffs | **PASS** |
| `[OPT7]` All 5 optimizer-state contracts (`16.00 → 8.05 B/param`) | 5-step convergence & zero nonfinites | `loss 3.4595 → 3.1783–3.2090` | **PASS** |
| `[DEP]` Whole-model streaming deploy vs `bcir_ext_forward` (5 mixers) | `GQA`, `SWA`, `DeltaNet`, `Hybrid-SWA+DN`, `MLA-absorbed` | `max_abs_logit_diff <= 5.66e-7` | **PASS** |
| `[LUT4]` Whole-model `COMP-1` AVX-512 vs portable packed LUT4 & Q8 bound | `bcir_deploy_prefill_seq` + $|w-\hat{w}| \le 2^{e-1}$ | `avx_diff = 0.000e+00`, `q8_viol = 0.000e+00` | **PASS** |

---

## 2. Experiment 1: Operational Foundations (`TRAIN-OPT-1`, `TRAIN-OPT-2`, `TRAIN-OPT-4`)

### 2.1 `TRAIN-OPT-4` Native C11 Initialization & `TRAIN-OPT-2` Suffix Trimming

| Scale | Params | C11 Ziggurat (`ms`) | `skip_init=1` (`ms`) | Python NumPy (`ms`) | Python stdlib est. (`ms`) | Speedup vs stdlib |
|---|---:|---:|---:|---:|---:|---:|
| `pilot1m` (`d=192, L=2`) | `861,120` | `10.8` | `6.2` | `16.6` | `189.1` | `17.6×` (`30.5×` skip) |
| `wide512` (`d=512, L=2`) | `6,162,944` | `73.9` | `43.5` | `203.0` | `1,534.7` | `20.8×` (`35.3×` skip) |
| `wide50m` (`d=768, L=8`) | `50,541,312` | `2,412.4` | `446.8` | `2,111.2` | `12,914.7` | `5.4×` (`28.9×` skip) |

*(Note: `c11_ziggurat_total_ms` at `50.54M` includes first-touch OS page faults across `771.2 MiB` of persistent state and avoids Python NumPy's `2×` peak-RSS doubling from allocating an external `192.8 MiB` staging array; `skip_init=1` accelerates checkpoint-load startup to `446.8 ms`.)*

On 4-pair KV sequences (`B=8, T=64`, active suffix `19.0/64`), **`TRAIN-OPT-2`** physical suffix trimming cuts median step latency from **`32.39 ms/step → 12.74 ms/step` (`2.54×` wall speedup)** across 25 AdamW steps with `max_loss_diff = 3.02e-7` and `max_weight_diff = 3.98e-6`.

### 2.2 `TRAIN-OPT-1`: Initialization Policy & Gradient-Clip Scaling Across Three Scales (2 Seeds)

| Scale (`steps`, `lr`) | Policy (`grad_clip`) | Mean Test CE | Seed Spread | Mean Test PPL | Median Grad Norm | Post-Warmup Clip Rate | Median `ms/step` |
|---|---|---:|---:|---:|---:|---:|---:|
| `pilot1m` (`300st`, `1.5e-3`) | `legacy_002` (`1.0`) | `2.3547` | `0.0126` | `10.54` | `1.28` | `98.7%` | `21.1` |
| `pilot1m` (`300st`, `1.5e-3`) | `residual_scaled` (`1.0`) | `2.2709` | `0.0372` | `9.69` | `1.33` | `100.0%` | `19.3` |
| `pilot1m` (`300st`, `1.5e-3`) | `mup_v1` (`1.0`) | `2.2709` | `0.0372` | `9.69` | `1.33` | `100.0%` | `19.0` |
| `pilot1m` (`300st`, `1.5e-3`) | `mup_v1` (`3.0`) | `2.2829` | `0.0515` | `9.81` | `1.37` | **`0.0%`** | `20.9` |
| `wide512` (`150st`, `1.2e-3`) | `legacy_002` (`1.0`) | `2.4252` | `0.0081` | `11.31` | `1.83` | `100.0%` | `98.2` |
| `wide512` (`150st`, `1.2e-3`) | `residual_scaled` (`1.0`) | `2.3989` | `0.0289` | `11.01` | `1.83` | `100.0%` | `106.3` |
| `wide512` (`150st`, `1.2e-3`) | `mup_v1` (`1.0`) | **`2.3610`** | `0.0083` | **`10.60`** | `2.12` | `100.0%` | `105.6` |
| `wide512` (`150st`, `1.2e-3`) | `mup_v1` (`3.0`) | `2.3664` | `0.0202` | `10.66` | `2.09` | **`2.6%`** | `104.2` |
| `wide50m` (`60st`, `1.0e-3`) | `legacy_002` (`1.0`) | `2.7542` | `0.0190` | `15.71` | `2.42` | `100.0%` | `747.5` |
| `wide50m` (`60st`, `1.0e-3`) | `residual_scaled` (`1.0`) | `2.6626` | `0.0180` | `14.34` | `2.37` | `100.0%` | `783.8` |
| `wide50m` (`60st`, `1.0e-3`) | `mup_v1` (`1.0`) | **`2.6611`** | `0.0137` | **`14.31`** | `2.91` | `100.0%` | `744.5` |
| `wide50m` (`60st`, `1.0e-3`) | `mup_v1` (`3.0`) | `2.6642` | `0.0179` | `14.36` | `2.98` | **`42.6%`** | `743.3` |

**Diagnosis:** Natural post-warmup gradient norm in raw-byte tied-embedding decoders scales approximately with $\sqrt{L}$ (`1.37` at 2L `pilot1m`, `2.09` at 2L `wide512`, `2.98` at 8L `wide50m`). Therefore `mup_v1` alone improves held-out quality (`-0.064` to `-0.093` nats) by fixing initialization variance and per-width LR scaling (`d_base / d`), while raising `grad_clip` from `1.0` to `3.0` (or depth-scaling to $\propto \sqrt{L}$) is separately required to remove gradient-clip saturation (`100.0% → 0.0% / 2.6% / 42.6%`).

---

## 3. Experiment 2: Completed Precision Frontier up to `50.54M` (`P1–P6`)

### 3.1 Isolated 2-Thread GEMM Microbenchmarks (`wide50m` Shapes, `Xeon 6985P-C`)

| `P1` GEMM Contract | `256×2048×768` (`ms` / `GF/s`) | `256×768×768` (`ms` / `GF/s`) | `256×768×2048` (`ms` / `GF/s`) | Rel. Frob. Err vs FP32 |
|---|---:|---:|---:|---:|
| `fp32_kblock` | **`2.95` / `273.3`** | **`1.05` / `288.5`** | **`2.82` / `285.6`** | `0.00000` |
| `bf16_ndt` (AVX-512F, no AMX) | `5.00` / `161.2` | `2.05` / `147.2` | `4.38` / `184.0` | `0.00235` |
| `bf16_vdp_avx512` (`VDPBF16PS`) | `6.65` / `121.0` | `2.11` / `143.1` | `5.54` / `145.3` | `0.00235` |
| `bf16_vdp_portable` (bit-exact ref) | `940.26` / `0.9` | `348.99` / `0.9` | `938.73` / `0.9` | `0.00235` |
| `bcirq8_g32` (`w_hat = q * 2^e`) | `6.32` / `127.3` | `2.43` / `124.5` | `6.52` / `123.5` | `0.01178` |
| `fp8_e4m3_blk32` | `10.66` / `75.6` | `3.96` / `76.3` | `11.05` / `72.9` | `0.03394` |
| `fp8_e5m2_blk32` | `10.49` / `76.8` | `3.95` / `76.5` | `10.71` / `75.2` | `0.06783` |
| `mxfp4_e2m1_blk32` | `5.34` / `150.9` | `2.02` / `149.4` | `5.60` / `143.9` | `0.16313` |

### 3.2 `50.54M` (`wide50m`) Complete 10-Policy Precision Frontier (Equal-Token & Equal-Time)

All 10 policies ran `120` steps (`B=4, T=64`, `mup_v1`, `grad_clip=3.0`, `lr=1.0e-3`, validation every `30` steps, validation-only selection, single untouched test evaluation, `0` nonfinites across every policy):

| Policy (`P2` Stage Assignment) | Persistent State (`MiB` / `B/p`) | Saved vs FP32 | Test CE (120st) | `Delta CE` vs FP32 | PPL Ratio $e^{\Delta\text{CE}}$ | Equal-Time Val CE (`94.8s`) | Median `ms/step` | `P3` FP64 Promos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `FP32_Ref` | `771.2` / `16.00` | `0.0 MiB` | `2.4613` | `+0.0000` | `1.0000` | **`2.4225`** | **`733.5`** | `9` |
| `Opt_Adafactor` | `578.9` / `12.01` | `192.3 MiB` | **`2.4507`** | **`-0.0106`** | **`0.9895`** | `2.4400` | `825.6` | `1` |
| `Compute_FP8_E4M3_E5M2` | `771.2` / `16.00` | `0.0 MiB` | `2.4597` | `-0.0016` | `0.9984` | `2.6015` | `1,363.9` | `0` |
| `Opt_Q8_BF16Grad` | **`387.9` / `8.05`** | **`383.3 MiB`** | `2.4603` | `-0.0009` | `0.9990` | `2.6023` | `1,357.1` | `1` |
| `Compute_BF16_VDP` | `771.2` / `16.00` | `0.0 MiB` | `2.4622` | `+0.0009` | `1.0009` | `2.4994` | `1,006.3` | `10` |
| `Compute_BF16_NDT` | `771.2` / `16.00` | `0.0 MiB` | `2.4624` | `+0.0011` | `1.0011` | `2.4673` | `887.3` | `6` |
| `Compute_MXFP4_Uniform` | `771.2` / `16.00` | `0.0 MiB` | `2.4659` | `+0.0046` | `1.0046` | `2.4846` | `948.3` | **`364`** |
| `Region_FP8Attn_MXFP4MLP_Q8Opt` | **`387.9` / `8.05`** | **`383.3 MiB`** | `2.4674` | `+0.0061` | `1.0062` | `2.6524` | `1,785.5` | `130` |
| `Region_FP32Attn_MXFP4MLP` | `771.2` / `16.00` | `0.0 MiB` | `2.4682` | `+0.0069` | `1.0069` | `2.5134` | `1,026.6` | **`10`** |
| `Compute_Asym_MXFP4_FP8` | `771.2` / `16.00` | `0.0 MiB` | `2.4708` | `+0.0095` | `1.0095` | `2.5489` | `1,177.0` | `0` |

### 3.3 Cross-Scale Width Effect (`pilot1m` `d=192` vs `wide512` `d=512` vs `wide50m` `d=768`)

| Policy | `pilot1m` (`0.86M`, 2 seeds) `Delta CE` | `wide512` (`6.16M`, 2 seeds) `Delta CE` | `wide50m` (`50.54M`) `Delta CE` | `P4` Gate (`<= +0.04` nats) |
|---|---:|---:|---:|---|
| `Opt_Adafactor` | `-0.0117` | `+0.0013` | `-0.0106` | **PASS (all 3 scales)** |
| `Opt_Q8_BF16Grad` (`8.05 B/p`) | `+0.0024` | `+0.0001` | `-0.0009` | **PASS (all 3 scales)** |
| `Compute_BF16_VDP` | `-0.0032` | `-0.0001` | `+0.0009` | **PASS (all 3 scales)** |
| `Compute_FP8_E4M3_E5M2` | `+0.0073` | `+0.0036` | `-0.0016` | **PASS (all 3 scales)** |
| `Compute_MXFP4_Uniform` | **`+0.0481` (FAIL)** | — | `+0.0046` (`364` FP64 promos) | **FAIL at `d=192`** |
| `Compute_Asym_MXFP4_FP8` | `+0.0304` | `+0.0078` | `+0.0095` | Marginal at `d=192` |
| `Region_FP32Attn_MXFP4MLP` | **`+0.0141`** | `+0.0077` | `+0.0069` (`10` FP64 promos) | **PASS (all 3 scales)** |
| `Region_FP8Attn_MXFP4MLP_Q8Opt` | **`+0.0138`** | `+0.0088` | `+0.0061` | **PASS (all 3 scales)** |

---

## 4. Experiment 3: Geometry, Spectral Rank, Whole-Model `50.54M` `COMP-1`, and Circuits

### 4.1 Parameter-Matched `6.16M` Geometry & Multi-Scale Spectral Audit

Matching `wide_shallow_6m` (`d=512, ff=1536, L=2`, `6,162,944` params) against `narrow_deep_6m` (`d=256, ff=736, L=8`, `6,164,736` params, `0.029%` diff) on identical WikiText-2 batches:

| Model / Region | Mean `r=1` Storage Floor (`BPW`) | Mean `r90` Spectral Rank (`% min_dim`) | Mean Stable Rank | Matched-Rank `LB_Uniform` `BPW` / Decode | `COMP-1` PTQ `BPW` / `Delta CE` / Decode |
|---|---:|---:|---:|---:|---:|
| `wedge128` (`0.53M`, MLP) | `0.1997` | `61.2%` | `18.4` | `0.87 BPW` (`r=16 res`) | `3.76 BPW` / `kv1 95.3%` PTQ |
| `narrow_deep_6m` (`6.165M`, MLP) | `0.0896` | `70.4%` | `65.7` | `0.47 BPW` / `2,146 tok/s` | `2.44 BPW` / `+0.5747` / `944 tok/s` |
| `wide_shallow_6m` (`6.163M`, MLP) | **`0.0443`** | `72.6%` | `149.4` | **`0.37 BPW`** / **`3,637 tok/s`** | **`2.06 BPW`** / **`+0.4335`** / **`1,426 tok/s`** |
| `wide50m` (`50.54M`, MLP) | **`0.0304`** | `70.8%` | `216.1` | **`0.32 BPW`** / **`680.2 tok/s`** | **`2.28 BPW`** / `+0.7049` / **`164.8 tok/s`** |

### 4.2 Whole-Model `50.54M` Deployment Matrix & `768×2048` Packed-1b LUT4 Batch Crossover

Whole-model streaming evaluation on untouched WikiText-2 test (`wide50m_fp32_120st.npz`, `avx_diff = 0.000e+00`, `q8_viol = 0.000e+00` everywhere):

| `50.54M` Deployment Policy | Body BPW | Artifact (`MiB`) | Test CE | `Delta CE` vs FP32 | Streaming Decode (`us/tok` / `tok/s`) | Speedup vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| `FP32_Ref` | `32.00` | `192.40` | `2.5013` | `+0.0000` | `67,928.5` / `14.7` | `1.00×` |
| `FP16_Uniform` | `16.00` | `96.40` | `2.5013` | `+0.0000` | `62,015.8` / `16.1` | `1.10×` |
| `BCIRQ8_Uniform` (lossy g32) | `8.25` | `49.90` | `2.5012` | **`-0.0002`** | `21,562.7` / `46.4` | **`3.15×`** |
| `COMP1_Q8Attn_LBMLP_r64_res` (PTQ) | `2.28` | `14.07` | `3.2062` | `+0.7049` (needs QAT) | `6,069.2` / `164.8` | **`11.20×`** |
| `COMP1_Q8Attn_LBMLP_r64_1p` (PTQ) | `2.17` | `13.42` | `3.2793` | `+0.7780` (needs QAT) | `6,033.7` / `165.7` | `11.26×` |
| `LB_Uniform_r64_res` (PTQ) | `0.32` | `2.33` | `3.5429` | `+1.0415` (needs QAT) | `1,470.2` / `680.2` | `46.27×` |

Isolated 2-thread `wide50m` MLP (`768×2048, r=64 res = 0.36 BPW`, `Xeon 6985P-C`) batch crossover (`us`):

| Batch $N$ | `Dense FP32` | `Dense FP16` | `Dense BCIRQ8` | `LowRank FP32` | `Packed1b LUT4` | `Packed1b mb4` | Winner |
|---:|---:|---:|---:|---:|---:|---:|---|
| `1` | `172.4` | `126.4` | `77.2` | `31.1` | **`4.8`** | `4.9` | `Packed1b LUT4` (`35.9×` vs FP32) |
| `4` | `190.5` | `546.7` | `325.6` | `40.4` | `19.4` | **`16.1`** | `Packed1b mb4` (`11.8×` vs FP32) |
| `16` | `311.0` | `2,187.2` | `1,463.6` | `350.5` | `87.1` | **`62.9`** | `Packed1b mb4` (`4.9×` vs FP32) |
| `32` | `475.6` | `4,656.5` | `2,610.7` | **`135.2`** | `165.2` | `140.9` | **Crossover `N ≈ 32`** (`mb4` still `3.4×` vs FP32) |
| `64` | `786.0` | `9,051.1` | `5,263.0` | **`226.5`** | `362.4` | `264.9` | `LowRank FP32` (`mb4` `3.0×` vs FP32) |
| `128` | `1,399.5` | `17,945.9` | `10,011.2` | **`420.7`** | `792.9` | `688.1` | `LowRank FP32` (`mb4` `2.0×` vs FP32) |
| `256` | `2,740.0` | `34,922.5` | `20,450.1` | **`688.2`** | `1,586.7` | `1,488.8` | `LowRank FP32` (`mb4` `1.8×` vs FP32) |

### 4.3 Exact Associative-Recall Circuit Preservation (`wedge128_kv4_strong`, PTQ vs 160-Step KD-QAT)

| Policy (`wedge128`, `525,696` params) | Body BPW | `kv_single` Exact Match | KD-QAT Final KL | Notes |
|---|---:|---:|---:|---|
| `FP32_Ref` / `FP16_Uniform` | `32.00` / `16.00` | `100.0%` | — | Teacher reference |
| `BCIRQ8_Uniform` (lossy g32) | `8.25` | `100.0%` | — | Matches `kv1`; differs by 1/64 sequence on `extrap_6pair` (`17.2%` vs `18.8%`) — confirming lossy contract |
| `COMP1_Q8Attn_LBMLP_r16_res` (PTQ) | `3.76` | `95.3%` | — | Attention preserved in Q8 |
| `COMP1_Q8Attn_LBMLP_r16_res_QAT` | `3.76` | **`100.0%`** | `0.0192` | Full circuit recovery via 160st KD-QAT |
| `COMP1_Q8Attn_LBMLP_r8_res_QAT` | `3.64` | **`100.0%`** | **`0.0127`** | Full recovery at `4.8×` lower KL than Uniform LB |
| `LB_Uniform_r16_res` (PTQ) | `0.87` | `73.4%` | — | Attention binarization collapses circuit (`-26.6 pp`) |
| `LB_Uniform_r16_res_QAT` | `0.87` | `92.2%` | `0.0611` | Remains `-7.8 pp` short after 160st KD-QAT |

---

## 5. Experiment 4: `MIXER-EXT` Architecture Comparison & `2×2` Weight × KV Orthogonality

### 5.1 Associative Recall, Counterfactuals, Long-Distance Filler (`> W`), and Length Extrapolation

| Architecture (`4L`, `SWA W=16`) | `Exp 4A` `kv1` (2 seeds) | `Exp 4A2` `mq4` (4-pair) | `Exp 4A2` `value_swap` | `Exp 4A2` `filler=20 (>W)` | `Exp 4A2` `extrap_6pair` | `4A2` `ms/step` |
|---|---:|---:|---:|---:|---:|---:|
| `GQA_Full` | `94.2%` | `55.9%` | `54.1%` | `17.4%` | `27.2%` | `29.9` |
| `SWA_Only` (`W=16`) | `91.1%` | **`83.2%`** | **`85.4%`** | **`11.5%` (window wall)** | `18.4%` | **`24.6`** |
| `Arch-1 DeltaNet` (`WY FP64`) | `93.0%` | `44.9%` | `44.9%` | **`44.0%` (`3.8×` SWA)** | `31.1%` | `37.4` |
| `Arch-2 Hybrid-SWA` (`3:1`) | `90.3%` | `64.5%` | `62.3%` | `12.5%` | `24.5%` | `25.1` |
| `Arch-2 Hybrid-DN+GQA` (`3:1`) | `71.9%` | `61.1%` | `62.5%` | `25.8%` (`2.2×` SWA) | **`33.6%`** | `35.9` |
| `Arch-4 MLA` (decoupled RoPE) | `93.4%` | `46.3%` | `45.1%` | `15.1%` | `30.5%` | `37.2` |

### 5.2 WikiText-2 `wide10m` (`~9.84M–11.44M`, 2 Seeds, 160 Steps) & `2×2` Orthogonality at `T=4096`

| Architecture | Params | Test CE (2 seeds) | Test PPL | `Delta CE` vs GQA | Train `ms/step` | FP32 (`W + KV@4k`) | `BCIRQ8` (`W + KV@4k` / `Delta CE`) | `COMP-1` (`W + KV@4k` / PTQ `Delta CE`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `GQA_Full` | `9.84M` | `2.3740` | `10.74` | `+0.0000` | `159.6` | `45.26 MiB` (`8.00 KV`) | `17.80 MiB` / `-0.0000` | `11.14 MiB` (`71.8%` KV!) / `+0.3562` |
| `Arch-2 Hybrid-SWA` (`3:1`) | `9.84M` | `2.3651` | `10.65` | `-0.0089` | **`157.3`** | `39.31 MiB` (`2.05 KV`) | **`11.85 MiB`** / `-0.0001` | **`5.19 MiB`** (`8.72×` cut) / `+0.4514` |
| `Arch-4 MLA` (dec. RoPE) | `10.69M` | `2.4049` | `11.08` | `+0.0309` | `186.3` | `45.51 MiB` (`5.00 KV`) | `15.64 MiB` / `+0.0001` | `8.98 MiB` (`5.07×` cut) / `+0.4921` |
| `Arch-1 DeltaNet` | `11.44M` | `2.2354` | `9.35` | **`-0.1386`** | `235.2` | `43.88 MiB` (`0.00 KV`, `0.50 Rec`) | **`11.88 MiB`** / `-0.0001` | **`5.22 MiB`** / `+1.2663` (needs QAT) |
| **`Arch-2 Hybrid-DN+GQA`** | `11.04M` | **`2.2257`** | **`9.26`** | **`-0.1483`** | `211.7` | `44.23 MiB` (`2.00 KV`, `0.38 Rec`) | `13.36 MiB` / `0.0000` | `6.70 MiB` / `+1.0348` (needs QAT) |

**Key Architectural & Systems Takeaway:**
- **`Arch-2 Hybrid-DN+GQA`** (`3:1` DeltaNet + Full GQA) and **`Arch-1 DeltaNet`** outperform `GQA_Full` by `-0.139` to `-0.148` nats (`-12.9%` to `-13.8%` PPL) on raw-byte WikiText-2 while slashing long-context KV cache (`3.36×` to `∞`).
- **`BCIRQ8_Uniform`** transfers cleanly across **every** `MIXER-EXT` family (`|Delta CE| <= 0.0001` nats) and delivers `1.29–1.95×` streaming decode speedups (`208–250 tok/s`).
- Conversely, zero-QAT `COMP-1` PTQ degrades `DeltaNet` (`+1.266` nats) and `Hybrid-DN+GQA` (`+1.035` nats) `~3×` harder than `GQA_Full` (`+0.356` nats), confirming that **sub-1-bit deployment on high-information-density DeltaNet hybrids requires dedicated KD-QAT and full `P4` qualification**.

---

## 6. Reproduction & Artifact Integrity (`P6`)

Every number, table, and plot in Sections 1–5 and 7 is reproducible directly from merged `Mecca-Research/Modular-AI` [PR #228](https://github.com/Mecca-Research/Modular-AI/pull/228) at commit `8cc3f234be0bc84009b79f152ab55995ada9a499` (`BCIR Model Training/precision-arch-study/`):

```bash
cd "BCIR Model Training/precision-arch-study"
make && make verify
python src/exp1_train_opt_1_4.py --part small && python src/exp1_train_opt_1_4.py --part 50m && python src/exp1_train_opt_1_4.py --part merge
python src/exp2_precision_frontier.py --part micro && python src/exp2_precision_frontier.py --part a_pilot # ... through --part merge
python src/exp3_littlebit_comp1_50m.py --part a && python src/exp3_littlebit_comp1_50m.py --part b && python src/exp3_littlebit_comp1_50m.py --part c && python src/exp3_littlebit_comp1_50m.py --part merge
python src/exp4_mixer_ext_architectures.py --part a && python src/exp4_mixer_ext_architectures.py --part a2 && python src/exp4_mixer_ext_architectures.py --part b1 && python src/exp4_mixer_ext_architectures.py --part b2 && python src/exp4_mixer_ext_architectures.py --part merge
python src/make_study_plots.py
```

All tracked Exp 1–5 artifacts (and the SHA-256 digest of the omitted 193 MiB deterministic `wide50m_fp32_120st.npz` teacher) are hashed in `results/MANIFEST.json` at merged commit `8cc3f234be0bc84009b79f152ab55995ada9a499`.

---

## 7. Production BCIR Deployment Policy (`COMP-1` / `P1–P6`) Implementation & Permanent Baseline Checkpoint (`Exp 5`)

The deployment policy validated by Experiments 1–4 is implemented directly in BCIR and anchored by a permanent `P6` baseline checkpoint (`bcir.deploy_baseline.v1`) so upcoming floating-point approaches and native `MIXER-EXT` architectures can be compared against a reproducible production reference.

### 7.1 Production BCIR Implementation Surface

| Layer | Implemented Files | Contract & Enforced Invariants |
|---|---|---|
| Native C11 `LUT4-LANES` | `runtime/c/bcir_ai_kernels.{h,c}`, `runtime/c/test_ai_kernels.c`, `bcir/kbcir/native_ai.py` | Packed 1.0-bit 2-nibbles/byte factorized matvec (`portable=1`, `AVX2=2`, `AVX-512=4`, `0` bit drift, `-ffp-contract=off`, zero heap allocations, `DECODE-THREADS` column slicing, `mb4` micro-batched prefill, `f64` & `f32`) |
| `BCIRQ8 v2` hybrid container | `runtime/c/bcir_q8_model.{h,c}`, `runtime/c/bcir_llama.c`, `runtime/c/bcir_decoder_gem.c`, `runtime/c/fuzz_q8_model.c`, `bcir/frontends/models/weights_io.py` | Preserves `BCIRQ8 v1` bit-for-bit; enforces at container level that `LUT4_1B` (`BCIR_Q8_TENSOR_ENC_LUT4_1B = 1`) is permitted **only** on MLP tensors (`W_GATE`, `W_UP`, `W_DOWN`) while keeping Attention (`W_Q/W_K/W_V/W_O`), embeddings, RMSNorms, and `lm_head` in deterministic lossy per-group Q8 (`\|w - q \cdot 2^e\| \le 2^{e-1}`), with 3-level CRC32, finite binary16 scale checks, canonical zero padding, and full `bcir-qualify` GEM parity |
| Qualification Engine (`P1–P6`) | `bcir/frontends/models/deploy_policy.py`, `bcir/tests/test_deploy_policy.py` | Dep-free stdlib (+ lazy optional NumPy SVD fast path, cold-quarantined in `tools/perf/import_graph.py`): explicit `PrecisionContract` (`P1`), width gate (`min_lut4_dim=512`, `P2`), selective `P3` promotion / fail-closed refusal, two-stage Q8-Attention + hybrid qualification (`P4`), Q8 bounded-error verifier, Dual-SVID + Joint-ITQ factorizer, and strict `bcir.deploy_baseline.v1` JSON baseline loader/saver (`P6`) |

### 7.2 Experiment 5 Verified Baseline Checkpoint (`8cc3f234be0bc84009b79f152ab55995ada9a499`)

Executed via `src/exp5_deploy_policy_baseline.py` and `src/bcir_native_deploy_bench.c` (linked directly against BCIR's production `runtime/c/{bcir_q8_model,bcir_decode,bcir_ai_kernels,bcir_llama,bcir_tensor}.c`) on a 2-core `Intel(R) Xeon(R) CPU @ 2.30GHz` (`avx2_fma`, no `avx512f`, `clang 23.1.6`, `-ffp-contract=off`), saved and schema-validated in `results/deploy_policy_baseline_checkpoint.json` (with frozen Q8 attention during KD-QAT and fail-closed nonfinite checks):

| Model / Stage | Artifact Size | vs `BCIRQ8 v1` | Untouched Test CE (`Delta CE`) | Test PPL | `2T` Decode (`tok/s`) | `1T` vs `2T` Drift | `bcir-qualify` GEM | `P1–P6` Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **`wide512`** (`6.16M`, `d=512, L=2`, 160st) `Default_FP32_Ref` | `23.51 MiB` (FP32) | — | `2.3542` (`+0.0000`) | `10.53` | — | — | — | Training ref (`16.00 B/p`, `199.9 ms/st`) |
| `wide512` `Default_Adafactor` (`12.01 B/p`) | `23.51 MiB` (FP32) | — | `2.3525` (`-0.0017`) | `10.51` | — | — | — | Training Pareto (`203.4 ms/st`) |
| `wide512` `Default_Q8_BF16Grad` (`8.05 B/p`) | `23.51 MiB` (FP32) | — | `2.3545` (`+0.0003`) | `10.53` | — | — | — | Memory-opt ref (`366.4 ms/st`) |
| `wide512` `BCIRQ8 v1` (`Q8 g=32`, 0 bound viols) | `6.25 MiB` | `1.00×` | `2.3545` (`+0.0000`) | `10.53` | `538.6` | `0` bits | — | Stage-1 Q8 PASS (`max_norm_err=1.000`) |
| `wide512` `BCIRQ8 v2 COMP-1` PTQ (`1.042` MLP bpw) | **`2.05 MiB`** | **`3.05×` smaller** | `2.4953` (`+0.1408`) | `12.13` | `303.6` | `0` bits | **PASS** (`114` claims) | `ADMITTED_HYBRID_V2` |
| **`wide512` `BCIRQ8 v2 COMP-1` + 80st KD-QAT** (`28.1s`) | **`2.05 MiB`** | **`3.05×` smaller** | **`2.3641` (`+0.0096`)** | **`10.63`** | `306.9` | `0` bits | **PASS** (`114` claims) | **`ADMITTED_HYBRID_V2`** (`KL=0.0031`, frozen Q8 Attn) |
| **`wide50m`** (`50.54M`, `d=768, L=8`, 60st `Q8_BF16Grad`) | `387.9 MiB` train (`8.05 B/p`) | — | `2.6920` (`+0.0000`) | `14.76` | — | — | — | Training ref (`2982.6 ms/st`, `0` nonfinites) |
| `wide50m` `BCIRQ8 v1` (`Q8 g=32`, 0 bound viols) | `51.22 MiB` | `1.00×` | `2.6920` (`+0.0000`) | `14.76` | `67.9` | `0` bits | — | Stage-1 Q8 PASS (`50.54M` elems verified) |
| `wide50m` `BCIRQ8 v2 COMP-1` PTQ (zero-QAT) | **`17.65 MiB`** | **`2.90×` smaller** | `3.0156` (`+0.3237`) | `20.40` | `37.8` | `0` bits | — | **`PROMOTED_Q8_V1`** (`P4` catches `+0.3237 > 0.20`, `P3` promotes to `v1`) |
| **`wedge128_kv4_strong`** (`0.53M`, `d=128 < 512`) | `547.0 KiB` (`v1`) | Width-gated | Exact match on 6/6 circuits (`kv_single 100.0%`) | — | `4,138.9` | `0` bits | **PASS** (`206` claims) | **`PROMOTED_Q8_V1`** (`W_Q` LUT4 refused) |

**Production `LUT4-LANES` vs `Q8-LANES` Kernel Sweep (`Exp 5D`, 2-Core Haswell AVX2 VM, `0` Bit Drift Across All 16 Points):**
- Across `wide512` (`512×1536`, `1536×512`) and `wide50m` (`768×2048`, `2048×768`) MLP shapes at `N ∈ {1, 4, 16, 64}` (`1.026–1.070` bpw, **`7.95×–8.28×` weight compression vs Q8**), production `LUT4-LANES` FP32 (`VPERMPS + VBLENDVPS`) achieves **`4.30×–5.47×` speedup** over `Q8-LANES` (`62.4` vs `296.6 us` on `wide512` up `N=1`; `104.5` vs `571.6 us` on `wide50m` up `N=1`).
- Because `bcir_llama` accumulates in exact FP64 and AVX2 lacks an in-register 16-entry FP64 permute (`lut4_accum32_f64_avx2` falls back to `_mm256_i32gather_pd`), FP64 `LUT4-LANES` on Haswell AVX2 runs at `0.45×–0.54×` of `Q8-LANES`, whereas AVX-512 hosts dispatch `_mm512_permutex2var_pd` with zero gathers (and `Exp 3` demonstrated `3.55×` whole-model `COMP-1` decode speedup over `BCIRQ8` on `Xeon 6985P-C` AVX-512).

---

## 8. Production ABI 6 `MIXER-EXT` Runtime, `Q8-Muon` Frontier, Beyond-the-List `Arch-7` Ladder & `1.024M`-Token Pilot (`PR #229`)

To advance directly from the Experiment 1–5 baseline to the next native sequence-mixer architecture and training stack — and push beyond the initial roadmap list — BCIR now ships the production ISO C11 ABI 6 `MIXER-EXT` + Default Training Stack runtime (`runtime/c/bcir_precision.{h,c}`, `runtime/c/bcir_mixer_ext.{h,c}`, `runtime/c/test_mixer_ext.c`, `bcir/hosted/models/mixer_ext.py`, `bcir/tests/test_mixer_ext.py`), preserving ABI 5 (`runtime/c/bcir_decoder_train.{h,c}`) untouched. Complete logs, JSON artifacts, plots, and the verified `1.024M`-token checkpoint are archived in [`Mecca-Research/Modular-AI` PR #229](https://github.com/Mecca-Research/Modular-AI/pull/229) (`BCIR Model Training/mixer-ext-abi6-study/`, pinned at commit `e458a859e190430827319b0e725b325241fbd787`).

### 8.1 Production ABI 6 Implementation & Gate 0 Verification

- **6 Native Mixers (`layer_mixer[l]`):** `GQA (0)`, `SWA (1)`, `DeltaNet (2)`, **Kimi Delta Attention `KDA` (`3`, `arXiv:2510.26692`)**, `MLA` RoPE/NoPE (`4`), and **Gated DeltaNet-2 `GDN-2` (`5`, `arXiv:2605.22791v2`)** with FP64 chunkwise WY solvers (`max_abs = 0.000e+00` DeltaNet, `3.55e-15` KDA, `2.67e-15` GDN-2 vs token recurrence) and verified MLA absorption (`<= 2.05e-8`).
- **5 Orthogonal Frontier Toggles (default `0`):** `gate_mode` (`G1` SDPA output sigmoid gate, `arXiv:2505.06708`), `qk_norm` (per-head Q/K RMSNorm), `canon_mode` (`Canon-A/C` causal `conv1d k=4`, `arXiv:2512.17351v2`), `mhc_streams` (`C=2, 4` DeepSeek `mHC` Sinkhorn-Knopp Birkhoff doubly-stochastic residual streams, `arXiv:2512.24880v2`, `t_max=20`), and `online_softmax` (`TRAIN-OPT-5` `O(T)` LSE tiled attention for `GQA` & `MLA`, reducing `GQA` prob buffers **`512.0×`** at `T=1024`: `512.00 → 1.00 MiB`).
- **7 Native Optimizers & 0-Bit-Drift Resume (`bcir_precision.{h,c}`):** Adds **`BCIR_OPT_MUON_FP32` (`12.03 B/p`)** and **`BCIR_OPT_Q8_MUON_BF16_GRAD` (`7.07 B/p`, `Muon-8L/AdamW-32 + BF16Grad`, `arXiv:2509.23106v3`)** alongside `FP32 AdamW`, `Q8`, `Q8+BF16Grad`, and `Adafactor`, plus fail-closed binary optimizer-state serialization (`bcir_opt_{state_bytes,save_state,load_state}` / `bcir_ext_{opt_state_bytes,save_opt_state,load_opt_state}`). Verified `0` bit drift for both `1T == 2T` AND mid-run checkpoint resume across all 7 optimizers, zero hot-path `malloc`, 0 Clang 23 static-analyzer warnings, and clean under GCC `-std=c11 -Werror` and Clang 23 `-std=c23 -fsanitize=address,undefined` (`detect_leaks=1`).

### 8.2 Optimizer Frontier & Muon Batch-Amortization Law (`Exp A`, `e458a85`)

| Scale / Policy | Persistent (`MiB` / `B/p`) | Saved vs FP32 | Mean Test CE (2 seeds) | Mean Test PPL | `Delta CE` vs FP32 | Median `ms/step` |
|---|---:|---:|---:|---:|---:|---:|
| `wide50m` (`50.74M`) `Opt_FP32_AdamW` | `774.20` / `16.00` | `0.00 MiB` | — | — | — | `1,539.5` |
| `wide50m` `Opt_Adafactor` | `581.12` / `12.01` | `193.08 MiB` | — | — | — | `1,615.3` |
| `wide50m` `Opt_Q8_BF16Grad` | `389.37` / `8.05` | `384.83 MiB` | — | — | — | `2,985.8` |
| **`wide50m` `Opt_Q8_Muon_BF16Grad`** | **`342.17` / `7.07`** | **`432.03 MiB (-55.8%)`** | — | — | — | — |
| `wide10m 3:1 DN+GQA` `Opt_FP32_AdamW` (`120st`) | `206.45` / `16.00` | `0.00 MiB` | `2.3030` | `10.00` | `+0.0000` | `501.0` |
| `wide10m 3:1 DN+GQA` `Opt_Adafactor` | `155.02` / `12.02` | `51.43 MiB` | `2.3153` | `10.13` | `+0.0123` | `514.0` |
| `wide10m 3:1 DN+GQA` `Opt_Q8_BF16Grad` | `103.83` / `8.05` | `102.62 MiB` | `2.3008` | `9.98` | `-0.0023` | `1,001.1` |
| **`wide10m 3:1 DN+GQA` `Opt_Q8_Muon_BF16Grad`** | **`92.30` / `7.15`** | **`114.15 MiB (2.24×)`** | **`2.2110`** | **`9.12`** | **`-0.0920`** | `2,553.1` |

- On `wide512 GQA`, Q8 momentum quantization matches FP32 Muon within **`+0.0004` nats** on both seeds (`2.3629` vs `2.3624`), validating `arXiv:2509.23106v3` Theorem 3.
- **CPU Batch-Amortization Rule:** Newton-Schulz fixed cost (`~940 ms` on `wide512`) slows tiny `N=256` updates by `5.92×`, amortizing to `2.23×` at `N=1024`, `1.62×` at `N=2048`, and `~1.15×` at `N=8192`. Use `Opt_Adafactor`/`Opt_Q8_BF16Grad` for small `N <= 512`, and **`Opt_Q8_Muon_BF16Grad`** once `N >= 2048` (via larger batch or gradient accumulation).

### 8.3 10-Architecture ABI 6 Ablation Ladder (`Exp B`, `wide10m` 2 Seeds, 160 Steps, `Opt_Adafactor`, Real Procrustes Joint-ITQ + Dual-SVID `COMP-1` PTQ)

| Architecture | Params | Test CE (`Delta vs GQA`) | Test PPL | `BCIRQ8` `Delta` | `COMP-1` PTQ `Delta` | Max `\|Act\|` | `T=4096` State | `B1` `mq4` / `fill20` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Arch7d` (`GDN2+G1NoPEMLA+CanonAC+mHC2`)** | `14.30M` | **`2.2526` (`-0.0927`)** | **`9.51`** | **`-0.0001`** | `+0.1086` | `10.62` | **`3.75 MiB`** | `20.8%` / `15.3%` (`act 9.06`) |
| `Arch2_Hybrid_DN_GQA` | `13.53M` | `2.2622` (`-0.0830`) | `9.60` | `+0.0001` | `+0.2214` | `16.28` | `4.75 MiB` | `15.7%` / `11.9%` |
| `Arch1_DeltaNet` | `13.93M` | `2.2651` (`-0.0802`) | `9.63` | `+0.0003` | `+0.1297` | **`7.62`** | **`1.00 MiB`** | `18.1%` / `16.9%` |
| `Arch7c` (`GDN2+G1NoPEMLA+CanonAC`) | `14.30M` | `2.2662` (`-0.0791`) | `9.64` | `+0.0001` | `+0.1105` | `10.87` | `3.75 MiB` | `20.8%` / `19.4%` |
| `Arch7b` (`GDN2+G1NoPEMLA`) | `14.28M` | `2.2796` (`-0.0657`) | `9.77` | `+0.0001` | `+0.1043` | `9.93` | `3.75 MiB` | `22.7%` / `15.3%` |
| `Arch7a` (`KDA+NoPEMLA`) | `13.88M` | `2.3136` (`-0.0316`) | `10.11` | `+0.0003` | `+0.1166` | `13.96` | `3.75 MiB` | **`25.9%` / `20.9%`** |
| `GQA_Full_G1_QKNorm` | `12.34M` | `2.3364` (`-0.0089`) | `10.34` | `+0.0001` | `+0.1282` | `14.13` | `16.00 MiB` | `20.8%` / `17.5%` |
| `GQA_Full_CanonAC` | `12.34M` | `2.3428` (`-0.0025`) | `10.41` | `-0.0000` | `+0.0960` | `14.29` | `16.00 MiB` | `25.5%` / `19.1%` (`ord8 16.0%`) |
| `GQA_Full` (baseline) | `12.33M` | `2.3453` (`+0.0000`) | `10.44` | `+0.0001` | **`+0.0885`** | `14.09` | `16.00 MiB` | `20.4%` / `18.4%` |
| `Arch4_MLA` (decoupled RoPE) | `13.57M` | `2.3972` (`+0.0519`) | `10.99` | `-0.0001` | `+0.1203` | `11.00` | `14.00 MiB` | `21.8%` / `15.9%` |

Every beyond-the-list component stacked monotonically (`Arch7a → 7b`: `-0.0340` nats; `7b → 7c`: `-0.0134`; `7c → 7d`: `-0.0136`), overtaking `Arch2_Hybrid_DN_GQA` while cutting `T=4096` state `4.27×` vs GQA (`3.75 MiB`) and cutting zero-QAT Dual-SVID + Joint-ITQ `COMP-1` PTQ error **`2.04×`** (`+0.1086` vs `+0.2214`).

### 8.4 Real `1.024M`-Token `T=256` Long-Horizon Pilot (`Exp C`, `online_softmax=1`, `B=8`)

| Model + Optimizer (`1.024M` tok @ `T=256`, `pilot1m`) | Params | Persistent RAM (`MiB`) | Untouched Test CE | Untouched Test PPL | `BCIRQ8` `Delta` | Max `\|Act\|` | Wall Time (`2-core`) | Throughput (`tok/s`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `GQA_Full` + `Opt_Adafactor` | `1.00M` | `11.55` | `1.8666` | `6.47` | `+0.0004` | `10.42` | `526.5 s` | `1,945` |
| `Arch-2 Hybrid-DN+GQA` + `Opt_Adafactor` | `1.09M` | `12.48` | `1.7138` | `5.55` | `+0.0006` | `8.87` | **`380.3 s`** | **`2,693`** |
| `Arch-7d` + `Opt_Adafactor` | `1.22M` | `14.03` | `1.6542` | `5.23` | `+0.0003` | `3.79` | `475.7 s` | `2,153` |
| **`Arch-7d` + `Opt_Q8_Muon_BF16Grad`** | `1.22M` | **`8.76`** | **`1.5716`** | **`4.81`** | `+0.0005` | **`2.15`** | `511.0 s` | `2,004` |

At `N=2,048` tokens/update over `1,024,000` real tokens (`~0.85–1.0` tokens/param), **`Arch-7d + Opt_Q8_Muon_BF16Grad`** achieves a **triple Pareto win**: **`-0.2949` nats (`-25.5%` PPL)** below `GQA_Full`, **`-0.1422` nats** below `Arch-2`, faster wall time than `GQA_Full` (`511.0 s` vs `526.5 s`), lowest persistent RAM (`8.76 MiB`), and **`4.84×` lower peak activation amplitude** (`2.15` vs `10.42`). Atomic SHA-256 checkpoint reload parity (`9.93e-8` CE) and live mid-run 0-bit-drift training resume (`4 + 4 == 8` steps with periodic in-loop recovery checkpoints + full optimizer state restore across `Opt_Adafactor` and `Opt_Q8_Muon_BF16Grad`, `test_ce_abs_diff = 0.0e+00`) are verified in `results/exp_c_tier_ml_pilot.json`; full Tier M (`10M`) and Tier L (`50M`) runs remain queued on 16+ core Xeon/EPYC hosts via `src/run_tier_ml_qual.py`.
