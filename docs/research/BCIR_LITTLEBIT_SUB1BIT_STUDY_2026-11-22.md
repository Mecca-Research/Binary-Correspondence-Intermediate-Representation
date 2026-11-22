# Bare-Metal C Evaluation of LittleBit Sub-1-Bit Latent Factorization (2026-11-22)

> Dated 2026-11-22, non-normative empirical study. Evaluates **LittleBit** (Lee et al., NeurIPS
> 2025, `arXiv:2506.13771v5`) and **LittleBit-2** (`arXiv:2603.00042v2`, Joint-ITQ) directly on
> BCIR-generated models and BCIR's native C11 tensor runtime
> ([`bcir_tensor.h`](../../runtime/c/bcir_tensor.h), [`bcir_decoder_train.h`](../../runtime/c/bcir_decoder_train.h),
> [`bcir_q8_model.h`](../../runtime/c/bcir_q8_model.h), [`bcir_ai_kernels.h`](../../runtime/c/bcir_ai_kernels.h))
> with zero PyTorch or framework dependency during QAT, distillation, or bit-packed inference.
>
> Companion documents:
> - Architecture build roadmap (updated with `COMP-1` and LittleBit evidence):
>   [BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md](BCIR_MODEL_ARCHITECTURE_ROADMAP_2026-11-21.md)
> - Training strategy, kernel, and deployment roadmap:
>   [BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md](BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md)
> - Whole-model Q8 inference and BCIRQ8 v1 contract:
>   [`BCIR_WHOLE_MODEL_REFERENCE.md`](../machine-learning/BCIR_WHOLE_MODEL_REFERENCE.md)

---

## 0. Executive Verdict: Six Empirical Laws

Implementing LittleBit and LittleBit-2 directly over `bcir_tensor.c` eliminates Python/framework
correspondence gaps and exposes both the genuine compression-speed wins and the structural limits
of sub-1-bit latent matrix factorization. Six measured laws govern how BCIR should use (and not
use) this technique:

1. **Sub-1-bit latent binary factorization decisively beats matched-byte low-rank FP16, and
   fixing three bare-metal bottlenecks in the published recipe slashes `0.31 BPW` PPL by `-1.42`.**
   On `BCIR-wide512` (`5.50M` params, 4:1 GQA, `FP32` test PPL `7.09`, `BCIRQ8` `7.09`, full
   1-bit `OneBit` `7.27` at `1.050 BPW`), published `LittleBit-2 (ITQ Res)` reaches **`7.26` PPL at
   `0.697 BPW`** (beating full 1-bit `OneBit` at 34% fewer bits), **`7.32` PPL at `0.545 BPW`**
   (`+0.032` nats over FP32 at `58.7×` body compression), **`8.43` PPL at `0.301 BPW`**, and
   **`9.17` PPL at `0.205 BPW`** (`156×` body compression), while matched-byte `Tiny-Rank FP16 + KD`
   degrades to `9.32`, `11.78`, `20.70`, and `27.85` PPL respectively. Replacing the published
   `SmoothSign(k=100) + λ=10·InterMSE` recipe with BCIR's ablation-derived **`Joint-ITQ +
   Clipped-STE + Output KL (λ = 0) + GQA kv_factor = 2.0`** recipe drops `0.311 BPW` PPL from
   **`8.75 → 7.33`** (`-1.42` PPL, `3.0×` less sign thrashing, `1.54×` faster QAT), and hybrid
   `COMP-1` (`BCIRQ8` Attention + `0.30b` MLP) reaches **`7.26` PPL** (`+0.024` nats over FP32).
2. **FP16 multi-scale vectors impose a hard $O(1/d)$ scale floor that blocks extreme sub-1-bit
   compression on narrow models and thin GQA K/V projections.**
   Before `rank = 1` is allocated, dual-path FP16 row/column scales (`h, g` × 2 paths) consume
   $\text{BPW}_{\text{floor}} = 32(1/d_{\text{in}} + 1/d_{\text{out}})$: **`0.425 BPW`** at
   $d=128$, **`0.273 BPW`** at $d=192$, and **`0.313 BPW`** on 4:1 GQA `K/V` (`128×512`) at
   $d=512$, versus `0.0123 BPW` at Llama2-7B and `0.0061 BPW` at 70B. Below $d \approx 256$,
   duplicating scales starves rank so strongly that **1-path strictly beats 2-path Residual**
   (`BCIR-wedge128`: `10.06` vs `10.59` PPL on LM; `100.0%` vs `83.6%` on ordered `value_swap`);
   at $d \ge 512$, 2-path Residual flips to a large win whenever target BPW exceeds `2×` the scale
   floor.
3. **Unquantized embeddings dominate total-model BPW on BPE LLMs — whereas BCIR's 260-byte
   vocabulary avoids the embedding trap by `12×`.**
   At `0.10 BPW` transformer body, `Llama3-8B` (`V = 128,256`, `13.09%` of parameters in FP16
   embeddings + head) inflates to **`2.181` total BPW** (`21.8×` headline), whereas tied
   byte-native `BCIR-wide50m` (`V = 260`, `0.42%` embeddings) achieves **`0.169–0.179` total BPW**.
   Conversely, LittleBit's latent KV caching *reverses* under GQA + 2-path Residual + `kv_factor=4`
   (`0.61×` on `Llama3-8B`, `0.55×` on `Llama2-70B`, `0.89×` on `BCIR-wide50m` at `0.55 BPW` —
   *expanding* KV cache) and does not commute with RoPE without MLA-style decoupled RoPE.
4. **Zero-shot sub-1-bit PTQ breaks exact attention circuits while sparing SwiGLU MLPs; QAT+KD
   recovers `100%` exact circuits where matched-byte low-rank FP16 fails.**
   On saved `BCIR-wedge128` checkpoints (`100%` FP32 teachers), zero-shot Dual-SVID PTQ collapses
   `kv_single` to `0.0%` (all layers) and `9.4%` (`AttnOnly`), whereas **`MLPonly` retains `100.0%`
   with zero QAT steps**. After QAT+KD at `0.75 BPW`, `LittleBit-1P` recovers **`100.0%` exact
   accuracy** on `kv_single`, `ordered`, and `ordered value_swap` (`both_correct = 100.0%`), while
   matched-byte `Tiny-Rank FP16` stalls at `72.7%` (`kv_single`), `55.5%` (`ordered`), and `28.9%`
   (`value_swap`). `BCIRQ8` (`8.50 BPW`, lossy group-32 quantization via `quantize_per_group`)
   achieved near-lossless retention on these checkpoints (`100.0%` zero-shot on
   `kv_single`/`ordered`/`value_swap`, `+0.00004` nat LM CE delta at `0.0078` relative weight
   Frobenius error, and `in_cand` `99.2% → 98.4%` on `kv_curriculum_4pair`).
5. **Naive bit-packing loses on CPU; exact 1.0-bit/weight 2-nibbles-per-byte AVX-512 LUT4
   delivers `18–225×` decode speedups (`N=1`), and the prefill crossover is shape- and
   kernel-dependent.**
   On Intel Xeon Platinum 8481C (Sapphire Rapids, 1 thread), portable bit-packing and AVX-512
   Sign-XOR lose to dense FP32/Q8 at small widths and during prefill (`N=256`). Packing two 4-bit
   LUT nibbles per byte (exact `1.0 bit/weight`, halving DRAM/L3 traffic) with `_mm512_permutexvar_ps`
   accelerates `N=1` decode by **`18.1×`** vs FP16 (`14.9×` vs Q8) on `BCIR-wide50m` (`5.8 us`),
   **`71.4×`** (`31.5×` vs Q8) on `Llama2-7B 0.30b` (`152.6 us`), and **`224.5×`** (`130.0×` vs
   Q8) on `Llama2-70B 0.10b` (`296.7 us`, L3-resident). Across batched prefill (`N = 4..256`),
   4-token weight-reusing `packed1b_LUT4_mb4` beats dense `BCIR FP32 kblock` everywhere
   (`2.6×–8.0×` faster at `N = 64–256`); against unpacked rank-matched `LowRank-FP32`, the
   crossover shifts strongly with shape and rank (`N ≈ 64–128` on `wide512`/`wide50m 0.30b`,
   `N > 256` on `wide50m 0.55b`, and `N > 128` on `Llama2-7B 0.30b/0.55b`, where `mb4` stays
   `1.96×–2.50×` faster).
6. **Sub-1-bit QAT+KD carries a heavy training compute and RAM tax (`2.0–3.1×` slower, `1.63×` RAM).**
   On `BCIR-wide512` (2 threads native C), `LittleBit_Res_0.55_KD` runs at **`228.4 ms/step`**
   (`1.97×` slower than Dense FP32 `115.9 ms`, `4.96×` slower than `LowRank FP16 KD` `46.0 ms`)
   and requires **`216.7 MiB`** (`1.63×` Dense FP32 `133.1 MiB`); `1.00 BPW` reaches `356.7 ms/step`
   (`3.08×`). LittleBit is strictly a post-training deployment compressor, never a pretraining
   accelerator.

---

## 1. Native C Engine, Reproducible Bundle & 3-Gate Verification

The complete native-C implementation (`src/bcir_littlebit.h`, `bcir_littlebit_core.c`,
`bcir_littlebit_bench.c`, `bench_lut4_packed.c`), portable `Makefile`, Python drivers
(`src/lb_harness.py`, `pillar1..5*.py`, `pillar5_packed_batched_bench.py`), trained FP32 teacher
checkpoints (`checkpoints/*.npz`), raw JSON/text outputs (`results/*.json`,
`results/verify_gates_output.txt`), and SHA-256 provenance manifest (`results/MANIFEST.json`,
pinning every source file, trained teacher, saved `../wedge-pr818/checkpoints/*.npz` checkpoint,
and `../gp-global/data/wikitext2-*.parquet` split) are archived in the companion repository at
pinned commit **`31bebc39a446b7b61f2a826c8a02a40cff88e0e5`**:

- Pinned bundle:
  https://github.com/Mecca-Research/Modular-AI/tree/31bebc39a446b7b61f2a826c8a02a40cff88e0e5/BCIR%20Model%20Training/littlebit-sub1bit-study
- BCIR base commit linked during build and verification: `3fdb708d73371a6e0e1abf7dde70b7239fc2fc06`

All measurements were executed on Intel Xeon Platinum 8481C (`avx512f`, `avx512_bf16`,
`avx512_fp16`, `avx512_vnni`, `avx512_vpopcntdq`), 2 cores, 7.8 GiB RAM, compiled with Clang 23.1.6
(`-O3 -std=c11 -D_POSIX_C_SOURCE=200809L -march=native -ffp-contract=off`) directly against BCIR's
[`runtime/c/bcir_tensor.c`](../../runtime/c/bcir_tensor.c) and
[`runtime/c/bcir_decoder_train.c`](../../runtime/c/bcir_decoder_train.c) (k-blocked GEMM, fused
RMSNorm, RoPE, causal GQA forward/backward, SwiGLU):

```bash
cd "BCIR Model Training/littlebit-sub1bit-study"
export BCIR_REPO=/path/to/Binary-Correspondence-Intermediate-Representation
make verify          # builds libbcir_littlebit.so + bcir_decoder_train.so and runs Gates 1-3
make bench-packed    # runs batched Packed-1b LUT4 (token-major & mb4) prefill/decode sweep
./build/bench_lut4_packed
```

All three verification gates passed on a clean build (`results/verify_gates_output.txt`):

| Verification Gate | Check Scope | Measured Error (`-ffp-contract=off`) | Verdict |
|---|---|---|---|
| **Gate 1: Dense BCIR Parity** | `LB_MODE_DENSE_FP32` vs unmodified `bcir_decoder_train.so` (forward loss + all parameter gradients) | `loss_diff = 0.00e+00`, `max_grad_diff = 0.00e+00` | **PASS** |
| **Gate 2: Finite-Diff Adjoints** | Central finite differences (`eps = 2e-3`, stratified spans + top gradients) across `LOWRANK_FP16`, `ONEBIT_FULL`, `LITTLEBIT_1P`, `LITTLEBIT_RES` + `KL + 10*InterMSE` KD | `max_abs_err <= 5.45e-5`; `max_rel_err` `5.86e-4` (`RES+KD`) to `4.54e-3` (`ONEBIT`) | **PASS** |
| **Gate 3: Bit-Packed Kernels** | `PORTABLE_BITPACK`, `AVX512_SIGNXOR`, `AVX512_LUT4`, `AVX512_LUT4_PACKED1B`, `AVX512_LUT4_PACKED1B_MB4`, `DENSE_FP16_STREAM`, `DENSE_Q8_BCIR` vs exact FP32 reference (plus non-multiple `130×258, r=13, N=1,7` and production shapes up to `N=64`) | `0.00e+00` (portable/Q8) to `8.94e-8` (AVX-512 LUT4 / Packed-1b / mb4) and `1.49e-7` (FP16); `<= 4.47e-7` at `768×2048, N=64` | **PASS** |

Formulations match LittleBit v5 (Appendix D BPW, Eqs. 6–8 Dual-SVID, SmoothSign $k=100$,
Eq. 11 $\mathcal{L}_{\text{KL}} + 10\,\mathcal{L}_{\text{inter-MSE}}$, Proposition 1 factorized
execution) and LittleBit-2 (Algorithm 1 50-iteration Joint-ITQ orthogonal Procrustes rotation
preserving $U'V'^\top = (U'R)(V'R)^\top$).

---

## 2. Analytical Bounds: Scale Floor, Total-Model BPW & KV-Cache Reversal

For a linear layer $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with 2-path Residual
factorization (ranks $r_1 = r_2 = r$) and FP16 scale vectors (`h, g, l` per path), exact bit cost
is:

$$\text{BPW}_{\text{res}}(r) = \frac{2r(d_{\text{in}} + d_{\text{out}}) + 32(d_{\text{in}} + d_{\text{out}}) + 32r}{d_{\text{in}} d_{\text{out}}}, \qquad \text{BPW}_{\text{floor}} = 32\left(\frac{1}{d_{\text{in}}} + \frac{1}{d_{\text{out}}}\right)$$

| Model | $d$ | $d_{kv}$ | $d_{\text{ff}}$ | Emb % | Body Floor (Res / 1P) | K/V Floor (Res) | At `0.10b` Target: Body / Total BPW (Clamped/7) | At `0.55b` Target: KV Ratio (`kv=1` / `kv=4`) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `BCIR-wedge128` (0.53M, MHA) | 128 | 128 | 256 | 6.50% | `0.4250` / `0.2125` | `0.5000` | `0.453` / `1.464` (`7/7`) | `64.00×` / `16.00×` (clamped $r=1$) |
| `BCIR-pilot192` (0.86M, GQA 2:1) | 192 | 96 | 512 | 5.90% | `0.2727` / `0.1364` | `0.5000` | `0.290` / `1.218` (`7/7`) | `48.00×` / `12.00×` (clamped $r=1$) |
| `BCIR-mid384` (3.25M, GQA 3:1) | 384 | 128 | 1024 | 3.13% | `0.1380` / `0.0690` | `0.3333` | `0.147` / `0.643` (`7/7`) | `6.40×` / `1.60×` |
| `BCIR-wide512` (5.50M, GQA 4:1) | 512 | 128 | 1536 | 2.20% | `0.1005` / `0.0503` | `0.3125` | `0.115` / `0.465` (`4/7`) | `5.82×` / `1.46×` |
| `BCIR-wide50m` (50.5M, GQA 3:1) | 768 | 256 | 2048 | 0.42% | `0.0690` / `0.0345` | `0.1667` | `0.102` / **`0.169`** (`2/7`) | `3.56×` / **`0.89×` (expands)** |
| `OPT-1.3B` (MHA) | 2048 | 2048 | 8192 | 6.01% | `0.0225` / `0.0112` | `0.0312` | `0.099` / `1.056` (`0/7`) | `3.88×` / **`0.97×` (expands)** |
| `Llama2-7B` (MHA) | 4096 | 4096 | 11008 | 3.89% | `0.0123` / `0.0062` | `0.0156` | `0.100` / `0.719` (`0/7`) | `3.75×` / **`0.94×` (expands)** |
| `Llama3-8B` (GQA 4:1) | 4096 | 1024 | 14336 | 13.09% | `0.0120` / `0.0060` | `0.0391` | `0.100` / **`2.181`** (`0/7`) | `2.46×` / **`0.61×` (expands)** |
| `Llama2-70B` (GQA 8:1) | 8192 | 1024 | 28672 | 0.76% | `0.0061` / `0.0030` | `0.0352` | `0.100` / `0.221` (`0/7`) | `2.20×` / **`0.55×` (expands)** |

**Structural RoPE incompatibility for latent KV caching:** Caching pre-up-projection latent keys
$z_t = (x_t \odot g) V_{\text{sign}} \odot \ell \in \mathbb{R}^{2r}$ cannot commute with
post-projection Rotary Position Embeddings $\mathrm{RoPE}(U_{\text{sign}}^\top z_t)$ unless RoPE
is explicitly decoupled onto a separate non-factorized key subspace — which is precisely what
`Arch-4` (`BCIR-MLA`) provides natively without binary quantization loss.

---

## 3. Spectral Break-Even & Joint-ITQ Conditioning Audit

Auditing all trained projection matrices across four BCIR FP32 teachers (saved `Modular-AI`
`wedge128` LM and `kv_single`, plus freshly trained `pilot192` and `wide512` teachers on WikiText-2
raw UTF-8 bytes) exposes both the spectral regime where binary factorization beats low-rank FP16
and the mechanism behind LittleBit-2's Joint-ITQ improvement:

| BCIR Teacher | Projs | Mean $\gamma$ | Mean `erank%` | Row Distortion $\lambda$ (Raw → ITQ) | Max $\lambda$ (Raw → ITQ) | Kurtosis (Raw → ITQ) | Zero-Shot `0.55 BPW` Rel-Frob Err: FP16 / LB1P / LBRes / ITQ1P / ITQRes | `OneBit` / `Q8` Err |
|---|---:|---:|---:|---|---|---|---|---|
| `wedge128_lm` (0.53M) | 21 | 0.397 | 65.0% | `0.349 → 0.273` | `0.622 → 0.540` | `3.01 → 2.45` | `0.965` / `0.895` / `0.930` / **`0.869`** / `0.925` | `0.598` / `0.0078` |
| `wedge128_kv1` (0.53M) | 21 | 0.393 | 65.5% | `0.350 → 0.273` | `0.638 → 0.544` | `3.02 → 2.46` | `0.968` / `0.897` / `0.932` / **`0.871`** / `0.927` | `0.599` / `0.0077` |
| `pilot192_lm` (0.86M) | 14 | 0.693 | 26.4% | `0.395 → 0.263` | `0.692 → 0.440` | `4.28 → 2.42` | `0.855` / `0.797` / `0.769` / **`0.704`** / `0.731` | `0.590` / `0.0078` |
| `wide512_lm` (5.50M) | 14 | 0.436 | 43.7% | `0.387 → 0.259` | `0.676 → 0.383` | `4.14 → 2.32` | `0.852` / `0.834` / `0.801` / `0.754` / **`0.751`** | `0.598` / `0.0080` |

Key spectral findings:
- **Joint-ITQ works on 100% of audited matrices (70/70):** Orthogonal Procrustes rotation reduces
  mean row distortion $\lambda$ by `22–33%` and collapses super-Gaussian kurtosis (`4.14–4.28`) to
  sub-Gaussian hypercube-aligned factors (`2.32–2.45`).
- **Exact width crossover predicted from Frobenius error alone:** At $d=128$, `ITQ-1P` (`0.869`)
  beats `ITQ-Res` (`0.925`) by `0.056`; at $d=512$, `ITQ-Res` (`0.751`) overtakes `ITQ-1P`
  (`0.754`), and on steep-spectrum layers (`L1.o`, `L1.gate`), `LBv1-1P` actually loses to
  `Tiny-Rank FP16` (`0.828` vs `0.778`) until Residual + Joint-ITQ flips it (`0.697`).

---

## 4. Sub-1-Bit Accuracy Frontier & Exact-Circuit Stress Test

### 4.1 Language Modeling Frontier (`BCIR-wide512` 5.50M & `BCIR-pilot192` 0.86M)

Evaluated on 16,384 held-out WikiText-2 test bytes (64 batches × B4 × T64) after native-C QAT+KD
(`KL + 10·InterMSE`, `SmoothSign k=100`, cosine LR `8e-4`, 120 steps on `wide512`, 300 steps on
`pilot192`):

| Variant (`wide512_5.50M`) | Body BPW | Total BPW (FP16 emb) | Deploy KiB | Layer-0 Ranks `(q, k, mlp)` | PTQ Test PPL | QAT Test PPL | Test CE | Byte Acc | Step ms (2T) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `FP32_teacher` | 32.000 | 31.648 | 23,817 | dense | 7.09 | **7.09** | 1.9588 | 43.12% | — |
| `BCIR_Q8_PTQ` (`BCIRQ8` g32) | 8.500 | 8.500 | 6,397 | dense Q8 | 7.09 | **7.09** | 1.9588 | 43.11% | — |
| `OneBit_1.0_KD` | 1.050 | 1.379 | 1,038 | full 1b | 7.66 | 7.27 | 1.9833 | 41.99% | 341.0 |
| `TinyRankFP16_1.00` | 0.997 | 1.327 | 999 | `(16, 6, 24)` | 12.62 | 7.94 | 2.0721 | 39.47% | 48.6 |
| `LBv1_Res_1.00` | 0.997 | 1.327 | 999 | `(110×2, 34×2, 174×2)` | 10.83 | 7.39 | 2.0005 | 41.42% | 387.8 |
| `LBv2_ITQ_Res_1.00` | 0.997 | 1.327 | 999 | `(110×2, 34×2, 174×2)` | **7.51** | **7.21** | **1.9751** | **42.63%** | 344.2 |
| `TinyRankFP16_0.70` | 0.668 | 1.006 | 757 | `(11, 4, 16)` | 47.58 | 9.32 | 2.2320 | 35.83% | 39.9 |
| `LBv1_Res_0.70` | 0.697 | 1.034 | 778 | `(72×2, 19×2, 117×2)` | 10.77 | 7.47 | 2.0112 | 41.12% | 265.2 |
| `LBv2_ITQ_Res_0.70` | 0.697 | 1.034 | 778 | `(72×2, 19×2, 117×2)` | **7.83** | **7.26** | **1.9825** | **42.18%** | 261.3 |
| `TinyRankFP16_0.55` | 0.531 | 0.872 | 656 | `(8, 3, 13)` | 54.72 | 11.78 | 2.4660 | 31.71% | 38.4 |
| `LBv1_Res_0.55` | 0.545 | 0.885 | 666 | `(53×2, 11×2, 88×2)` | 11.67 | 7.56 | 2.0226 | 40.77% | 218.0 |
| `LBv2_ITQ_Res_0.55` | 0.545 | 0.885 | 666 | `(53×2, 11×2, 88×2)` | **8.05** | **7.32** | **1.9906** | **41.78%** | 212.8 |
| `TinyRankFP16_0.30` | 0.279 | 0.625 | 470 | `(4, 1, 7)` | 116.06 | 20.70 | 3.0300 | 24.71% | 36.6 |
| `LBv1_Res_0.30` | 0.301 | 0.647 | 487 | `(22×2, 2×2, 41×2)` | 20.01 | 8.75 | 2.1693 | 36.95% | 146.9 |
| `LBv2_ITQ_Res_0.30` | 0.301 | 0.647 | 487 | `(22×2, 2×2, 41×2)` | **16.52** | **8.43** | **2.1317** | **37.75%** | 149.9 |
| `TinyRankFP16_0.20` | 0.170 | 0.518 | 390 | `(3, 1, 4)` | 78.76 | 27.85 | 3.3270 | 22.72% | 36.3 |
| `LBv1_Res_0.20` | 0.205 | 0.553 | 416 | `(9×2, 2×2, 22×2)` | 24.61 | 9.55 | 2.2561 | 35.01% | 123.0 |
| `LBv2_ITQ_Res_0.20` | 0.205 | 0.553 | 416 | `(9×2, 2×2, 22×2)` | **23.09** | **9.17** | **2.2161** | **35.53%** | 121.4 |

On narrower `pilot192_0.86M` (`FP32` `6.87`, `Q8` `6.87`, `OneBit` `7.03` at `1.136 BPW`), the
$O(1/d)$ scale floor manifests directly:
- `0.990 BPW`: `TinyRankFP16 = 12.28`, `LBv1_Res = 7.34` (PTQ `17.26`), **`LBv2_ITQ_Res = 7.14`** (PTQ `8.11`).
- `0.695 BPW`: `TinyRankFP16 = 16.16`, `LBv1_Res = 7.64` (PTQ `23.55`), **`LBv2_ITQ_Res = 7.43`** (PTQ `12.12`).
- `0.538 BPW`: `TinyRankFP16 = 23.53`, `LBv1_Res = 8.78` (PTQ `35.97`), **`LBv2_ITQ_Res = 8.66`** (PTQ `32.59`).
- `0.372 BPW` (scale-floor clamped at `r_q=2, r_k=2, r_mlp=8`): cliff to **`12.63` / `12.70`** (`+0.615` nats vs `+0.173` nats on `wide512` at an *even lower* `0.301 BPW`!).

### 4.2 Exact Associative Binding & Ordered Counterfactuals (`BCIR-wedge128`, Saved `Modular-AI` Teachers)

Evaluated on saved `Modular-AI` (`wedge-pr818`) FP32 attention checkpoints (`d=128, L=3, MHA`):

| Variant (`BCIR-wedge128`) | Body BPW | LM Test PPL (PTQ → QAT) | `kv_single` Acc (PTQ → QAT) | `ordered` Acc (PTQ → QAT) | `ordered` `value_swap` Both-Correct |
|---|---:|---:|---:|---:|---:|
| `FP32_teacher` | 32.000 | `9.21 → 9.21` | `100.0% → 100.0%` | `100.0% → 100.0%` | `100.0%` |
| `BCIR_Q8_PTQ` (`BCIRQ8` g32) | 8.500 | `9.21 → 9.21` | `100.0% → 100.0%` | `100.0% → 100.0%` | `100.0%` |
| `OneBit_Full_KD` | 1.212 | `17.02 → 9.42` | `83.6% → 100.0%` | `95.3% → 100.0%` | `100.0%` |
| `TinyRankFP16_0.75` | 0.750 | `36.96 → 13.01` | `0.0% → 72.7%` | `5.5% → 55.5%` | **`28.9%`** |
| `LBv1_1P_0.75` | 0.749 | `48.45 → 10.36` | `0.0% → 100.0%` | `6.2% → 100.0%` | **`100.0%`** |
| `LBv1_Res_0.75` | 0.738 | `55.54 → 10.86` | `0.0% → 100.0%` | `7.8% → 100.0%` | `100.0%` |
| `LBv2_ITQ_1P_0.75` | 0.749 | `25.56 →` **`10.06`** | `9.4% → 100.0%` | `10.2% → 100.0%` | **`100.0%`** |
| `LBv2_ITQ_Res_0.75` | 0.738 | `29.99 → 10.59` | `0.0% → 100.0%` | `16.4% → 95.3%` | **`83.6%`** |
| `LBv1_MLPonly_0.75` | 13.249 | — | **`100.0%`** `→ 100.0%` | `20.3% → 100.0%` | `100.0%` |
| `LBv1_AttnOnly_0.75` | 19.499 | — | **`9.4%`** `→ 100.0%` | `10.2% → 100.0%` | `100.0%` |

---

## 5. Controlled Ablation Matrix (`BCIR-wide512` 5.50M GQA 4:1)

All ablations hold dataset, batch schedule (`B4×T64`), optimizer (`AdamW`, cosine `8e-4`, 120
steps), and teacher (`wide512_5.50M`, `FP32` test PPL `7.09`) identical.

### 5.1 A1: Residual Compensation vs 1-Path (Matched Bit Budget)

| Target BPW | Realized Body BPW (1P / Res) | 1-Path PTQ → QAT PPL | 2-Path Residual PTQ → QAT PPL | Residual Gain ($\Delta$ PPL / $\Delta$ CE) |
|---:|---|---:|---:|---:|
| `0.80` | `0.797` / `0.797` | `20.51 → 8.40` | `10.75 →` **`7.45`** | **`-0.95` PPL (`-0.119` nats)** |
| `0.55` | `0.548` / `0.545` | `21.21 → 8.60` | `11.67 →` **`7.56`** | **`-1.04` PPL (`-0.130` nats)** |
| `0.30` | `0.298` / `0.301` | `23.34 → 9.13` | `20.01 →` **`8.75`** | **`-0.38` PPL (`-0.042` nats)** |
| `0.20` | `0.199` / `0.205` | `27.23 → 10.71` | `24.61 →` **`9.55`** | **`-1.16` PPL (`-0.115` nats)** |

**Width-dependent phase transition:** At $d=512$, 2-path Residual beats 1-path at *every* tested
bitrate (`0.80` down to `0.20 BPW`) because the second scale-compensated basis captures an
orthogonal quantization error profile while adding only `+0.050 BPW` scale overhead. On narrow
`BCIR-wedge128` ($d=128$, §4.2), duplicating scales costs `+0.213 BPW` and cuts total rank by
`>53%`, flipping the winner back to **1-path** (`10.06` vs `10.59` PPL; `100.0%` vs `83.6%` on
ordered `value_swap`).

### 5.2 A2: Multi-Scale Compensation (`h, g, l` at `0.30 BPW`)

| Scale Configuration | Realized Body BPW | Init Frob Err | PTQ PPL | QAT Test PPL | Test CE |
|---|---:|---:|---:|---:|---:|
| Full `(h, g, l)` | `0.301` | `0.859` | `20.01` | **`8.75`** | `2.1693` |
| Drop latent `l` (`h, g` only) | `0.299` | `0.874` | `20.12` | `8.86` | `2.1815` (`+0.012` nats) |
| No scales (`h = g = l = 1`) | `0.199` | `141.998` | `306.60` | **`34.44`** | `3.5393` (`+1.370` nats) |

Row/column scales `(h, g)` carry `>99%` of the compensation value (without them, unscaled binary
factorization collapses to `34.44` PPL); the latent vector `l` adds a small, cheap (`0.002 BPW`)
`-0.11` PPL refinement.

### 5.3 A3: SmoothSign vs STE & The Joint-ITQ Gradient-Freezing Interaction (`0.30 BPW`)

| Surrogate + Init | Init Frob | PTQ PPL | QAT Test PPL | Test CE | Sign Flip Rate | Active Grad Frac (`actG`) | Step ms (2T) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Dual-SVID + SmoothSign(k=100)` | `0.859` | `20.01` | `8.75` | `2.1693` | `0.00182` | `53.1%` | `152.2 ms` |
| `Dual-SVID + SmoothSign(k=10)` | `0.859` | `20.01` | `8.77` | `2.1710` | `0.00190` | `100.0%` | `173.5 ms` |
| `Dual-SVID + Clipped-STE` | `0.859` | `20.01` | `8.79` | `2.1742` | `0.00190` | `100.0%` | `147.0 ms` |
| `Dual-SVID + Unclipped-STE` | `0.859` | `20.01` | `8.79` | `2.1742` | `0.00190` | `100.0%` | `147.1 ms` |
| `Joint-ITQ + SmoothSign(k=100)` | `0.836` | `16.52` | `8.43` | `2.1317` | `0.00141` | **`48.1%`** | `179.0 ms` |
| **`Joint-ITQ + Clipped-STE`** | `0.836` | `16.52` | **`8.41`** | **`2.1290`** | `0.00148` | **`100.0%`** | **`146.7 ms`** (`1.22×` faster) |

**Key mechanistic finding:** Without Joint-ITQ, `SmoothSign(k=100)` offers only a `0.0049`-nat edge
over plain `Clipped-STE` (`8.75` vs `8.79`). **With LittleBit-2 Joint-ITQ**, orthogonal rotation
pushes latent weights away from zero toward `±1` hypercube corners — freezing **`51.9%`** of
`SmoothSign(k=100)` gradients on `wide512` (`69.9%` on `pilot192`, `81.5%` on `wedge128`) outside
its narrow `|x| < 0.03` support. Switching `Joint-ITQ` to **`Clipped-STE`** restores `100%` active
gradients, improves QAT PPL (`8.41` vs `8.43`), and eliminates `tanh()` backward cost for a
**`1.22×` faster native-C QAT step** (`146.7` vs `179.0 ms`).

### 5.4 A4: Initialization & Training Objective (`0.30 BPW`, 120 Steps Unless Noted)

| Initialization + Objective | PTQ PPL | QAT Test PPL | Test CE | Sign Flip Rate | Step ms (2T) | Key Finding |
|---|---:|---:|---:|---:|---:|---|
| `From-Scratch Random + CE` | `364.79` | `12.82` | `2.5509` | `0.00151` | `94.5 ms` | Fails as from-scratch pretraining (`+4.70` PPL vs KL) |
| `Naive SVD + KL + 10·InterMSE` | `45.65` | `10.77` | `2.3768` | `0.00125` | `149.0 ms` | Dual-SVID rank-1 scale init is load-bearing (`+2.02` PPL) |
| `Dual-SVID + CE only` (no teacher!) | `20.01` | `8.44` | `2.1330` | `0.00106` | `133.8 ms` | Beats paper Eq. 11 (`8.75`) with **61% less RAM** (`83.6 MiB`) |
| **`Dual-SVID + Output KL (λ = 0)`** | `20.01` | **`8.12`** | **`2.0940`** | **`0.00110`** | `154.2 ms` | **Beats Eq. 11 (`8.75`) by `-0.63` PPL & even 300-step Eq. 11 (`8.16`)!** |
| `Dual-SVID + KL + 10·InterMSE` (Eq. 11) | `20.01` | `8.75` | `2.1693` | `0.00182` | `152.6 ms` | Heavy `10·InterMSE` boosts sign thrashing by **+65%** |
| `Dual-SVID + 0.5CE + 0.5KL + 10·InterMSE` | `20.01` | `8.79` | `2.1735` | `0.00182` | `156.7 ms` | Same `+65%` sign thrashing from `10·InterMSE` |
| `Dual-SVID + Eq. 11 (300 steps, 2.5×)` | `20.01` | `8.16` | `2.0990` | `0.00186` | `153.3 ms` | Still slightly behind 120-step `Output KL (λ = 0)` (`8.12`) |

**Key objective finding:** At extreme sub-1-bit compression (`0.30 BPW`), LittleBit Eq. 11's heavy
`λ = 10` intermediate residual-stream MSE over-constrains the binary lattice to mimic the exact
coordinate basis of the FP32 teacher, increasing sign-flip thrashing by **+65%** (`0.00110 →
0.00182`) and hurting held-out PPL by **`+0.63`** (`8.12 → 8.75`). Removing `InterMSE` (`Output KL`
only, `λ = 0`) converges faster in `120` steps (`8.12`) than `Eq. 11` does in `300` steps (`8.16`).

### 5.5 A5: GQA `kv_factor` Allocation (`wide512` 4:1 GQA at `0.30 BPW`)

| GQA `kv_factor` | Realized Body BPW | $\Delta$ Body BPW | Init Frob | PTQ PPL | QAT Test PPL | Test CE |
|---:|---:|---:|---:|---:|---:|---:|
| `1.0` (uniform — `K/V` clamped $r=2$) | `0.301` | `+0.000` | `0.859` | `20.01` | `8.75` | `2.1693` |
| **`2.0`** (de-clamps thin `128×512` `K/V`) | `0.311` | **`+0.010`** | `0.835` | **`10.98`** | **`7.70`** | **`2.0408`** (`-1.05` PPL!) |
| `4.0` (paper GQA default; hurts KV cache) | `0.338` | `+0.037` | `0.807` | `10.99` | `7.64` | `2.0336` (`-0.06` marginal) |

On 4:1 GQA, `K/V` matrices are only `4.3%` of body parameters yet suffer a `3.1×` higher scale
floor (`0.3125 BPW`). Uniform `0.30 BPW` starves `K/V` to `r = 2`; `kv_factor = 2.0` costs just
**`+0.010 BPW`** body-wide while halving PTQ PPL (`20.01 → 10.98`) and cutting QAT PPL by
**`-1.05` PPL** (`8.75 → 7.70`), whereas `kv_factor = 4.0` buys only `-0.06` PPL while expanding
latent KV cache (§2).

### 5.6 A6: Layer Sensitivity — MLP-Only vs Attention-Only (`0.30 BPW`)

| Quantized Scope | Compressed Param Share | PTQ PPL | QAT Test PPL | $\Delta$ CE vs FP32 (`1.9588`) | Relative Damage / Parameter |
|---|---:|---:|---:|---:|---:|
| **`MLP_only` (`gate, up, down`)** | **`78.3%`** (`4.72M`) | **`13.04`** | **`7.35`** | **`+0.0355` nats** | **`1.0×` (baseline)** |
| `Attn_only` (`q, k, v, o`) | `21.7%` (`1.31M`) | `17.40` | `8.02` | `+0.1235` nats | **`12.5×` more sensitive** |
| `All_layers` (`q..down`) | `100.0%` (`6.03M`) | `20.01` | `8.75` | `+0.2105` nats | `4.6×` |

**Per-parameter sensitivity proof for `COMP-1`:** Attention projections suffer **`12.5×` more
cross-entropy degradation per compressed parameter** than SwiGLU MLP projections at `0.30 BPW`
(`+0.1235` nats from `21.7%` of parameters vs `+0.0355` nats from `78.3%`). Combined with §4.2's
`100.0%` zero-shot `kv_single` retention on `MLP_only`, this quantitatively establishes the
hybrid `Q8-Attention + Sub-1-Bit MLP` design boundary.

### 5.7 A7: End-to-End Validation of BCIR's Stacked Findings & `COMP-1` Hybrid (`0.30 BPW`)

Combining the three failure-mode repairs uncovered in A3 (`Clipped-STE` replacing `SmoothSign(k=100)`
under Joint-ITQ), A4 (`Output KL λ = 0` replacing `10·InterMSE`), and A5/A6 (`kv_factor = 2.0` or
`BCIRQ8` Attention) yields a dramatic simultaneous gain in accuracy, sign stability, and QAT speed:

| Recipe (`wide512_5.50M`, identical 120 QAT steps) | Effective Body BPW | PTQ PPL | QAT Test PPL | Test CE | Sign Flip Rate | Step ms (2T) |
|---|---:|---:|---:|---:|---:|---:|
| `FP32_teacher` / `BCIRQ8` g32 PTQ | `32.00` / `8.50` | `7.09` | `7.09` | `1.9588` | — | `115.9 ms` (FP32) |
| Published `LittleBit-v1` (Eq. 11: `Dual-SVID + SmoothSign k=100 + λ=10`) | `0.301` | `20.01` | `8.75` | `2.1693` | `0.00182` | `152.6 ms` |
| Published `LittleBit-2` (`Joint-ITQ + SmoothSign k=100 + λ=10`) | `0.301` | `16.52` | `8.43` | `2.1317` | `0.00141` | `179.0 ms` |
| **`BCIR All-Layer Synthesis` (`Joint-ITQ + Clipped-STE + KL λ=0 + kv=2`)** | **`0.311`** (`103×` vs FP32) | **`8.51`** | **`7.33`** | **`1.9923`** (`+0.034` nats) | **`0.00060`** (`3.0×` lower!) | **`116.3 ms`** (`1.54×` faster!) |
| **`BCIR COMP-1 Hybrid` (`BCIRQ8` g32 Attn + `0.30b` ITQ+STE+KL MLP)** | **`2.08`** (`15.4×` vs FP32) | **`8.17`** | **`7.26`** | **`1.9828`** (`+0.024` nats) | **`0.00063`** | **`114.4 ms`** |

Fixing the three bare-metal bottlenecks drops all-layer `0.31 BPW` PPL by **`-1.42` PPL (`8.75 →
7.33`)** — matching `0.545 BPW` LittleBit-2 (`7.32`) at **43% fewer bits** and `1.54×` faster QAT —
while `COMP-1` Hybrid reaches **`7.26` PPL** (beating full-rank 1-bit `OneBit` `7.27`) while
keeping attention in per-group Q8 (`BCIRQ8` g32, verified per-checkpoint against the FP32 teacher).

---

## 6. Bare-Metal C Inference & QAT Training Overhead

### 6.1 Single-Thread AVX-512 Inference Microbenchmark (Sapphire Rapids)

#### Decode (`N = 1`) — Standalone Packed-1b vs Dense & Bit-Packed Baselines

Comparing BCIR's production `kblock` FP32 GEMM, `bf16` GEMM, AVX-512 FP16 stream, Q8 g32 (`8.25b`
1B-exp/32 microbench; `BCIRQ8 v1` uses `8.50b` int16-exp/32), rank-matched `LowRank-FP32`, portable
bit-packed C, AVX-512 masked Sign-XOR, unpacked 4-bit LUT (`1 B/nibble`), and **exact
1.0-bit/weight 2-nibbles-per-byte AVX-512 LUT4 (`packed1b_LUT4`, `bench_lut4_packed.txt`)**:

| Shape (`din×dout`) | $N$ | BPW | $r$/path | BCIR FP32 | FP16 Stream | Q8 g32 | LowRank FP32 | AVX-512 SignXOR | Unpacked LUT4 | **Packed-1b LUT4** | **Packed-1b vs FP16 / Q8** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `wedge128` `128×256` | 1 | 0.55 | 8 | 5.4 us | 1.8 us | 2.3 us | 1.6 us | 2.9 us | 0.5 us | — | `3.6×` / `4.5×` (LUT4) |
| `wide512` `512×1536` | 1 | 0.30 | 41 | 152.8 us | 41.7 us | 43.3 us | 24.5 us | 72.3 us | 3.8 us | **3.7 us** | **`11.3×`** / **`11.7×`** |
| `wide50m` `768×2048` | 1 | 0.30 | 67 | 342.9 us | 104.9 us | 86.2 us | 58.9 us | 28.3 us | 7.1 us | **5.8 us** (63 KiB) | **`18.1×`** / **`14.9×`** |
| `Llama2-7B` `4096×11008` | 1 | 0.55 | 804 | 23,813.7 us | 12,085.9 us | 4,498.6 us | 13,677.7 us | 1,788.2 us | 559.5 us | **337.4 us** (3.0 MiB) | **`35.8×`** / **`13.3×`** |
| `Llama2-7B` `4096×11008` | 1 | 0.30 | 431 | 23,352.6 us | 10,893.7 us | 4,802.9 us | 5,487.3 us | 1,452.1 us | 238.3 us | **152.6 us** (1.6 MiB) | **`71.4×`** / **`31.5×`** |
| `Llama2-7B` `4096×11008` | 1 | 0.10 | 133 | 18,900.9 us | 9,076.9 us | 4,590.6 us | 969.3 us | 256.1 us | 64.5 us | **51.9 us** (585 KiB) | **`174.9×`** / **`88.5×`** |
| `Llama2-70B` `8192×28672` | 1 | 0.55 | 1735 | 135,630.4 us | 67,568.4 us | 38,391.8 us | 82,517.7 us | 4,819.2 us | 6,353.7 us | **2,023.9 us** (15.8 MiB) | **`33.4×`** / **`19.0×`** (`3.14×` vs unpacked!) |
| `Llama2-70B` `8192×28672` | 1 | 0.30 | 939 | 135,866.3 us | 67,555.3 us | 38,634.6 us | 43,438.1 us | 5,410.0 us | 2,778.8 us | **1,055.3 us** (8.6 MiB) | **`64.0×`** / **`36.6×`** |
| `Llama2-70B` `8192×28672` | 1 | 0.10 | 302 | 131,830.4 us | 66,621.2 us | 38,577.8 us | 11,397.2 us | 3,296.2 us | 510.2 us | **296.7 us** (2.9 MiB) | **`224.5×`** / **`130.0×`** |

#### Batched Prefill Crossover (`N = 1..256`) — Direct Packed-1b LUT4 & 4-Token Micro-Batch (`mb4`) Benchmark

Benchmarking exact 1.0-bit/weight `packed1b_LUT4` (token-major) and 4-token weight-reusing
`packed1b_LUT4_mb4` (same 1.0-bit/weight storage, amortizing each 16-byte load and nibble unpack
across 4 tokens; Gate-3 `max_abs_err <= 4.47e-7`) directly inside `bcir_littlebit_bench.c`
(`results/bench_lut4_packed_batched.json`) against BCIR's `kblock` FP32 GEMM (`BCIR FP32`) and
unpacked rank-matched `LowRank-FP32` (`2`-step `bcir_tensor_mm_kblock_pool`):

| Shape (`din×dout`) | BPW ($r$/path) | $N$ | BCIR FP32 | LowRank FP32 | Unpacked LUT4 | **Packed-1b (1-tok)** | **Packed-1b (`mb4`)** | **LowRank / `mb4`** | **FP32 / `mb4`** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wide512` `512×1536` | 0.30 (41) | 1 | 151.4 us | 27.0 us | 3.8 us | **3.7 us** | 3.7 us | `7.27×` | `40.9×` |
| `wide512` `512×1536` | 0.30 (41) | 16 | 342.7 us | 68.2 us | 67.5 us | 70.8 us | **59.6 us** | `1.14×` | `5.75×` |
| `wide512` `512×1536` | 0.30 (41) | 32 | 517.3 us | **102.0 us** | 132.2 us | 145.2 us | 107.5 us | `0.95×` | `4.81×` |
| `wide512` `512×1536` | 0.30 (41) | 64 | 906.5 us | 242.8 us | 273.3 us | 284.6 us | **235.0 us** | `1.03×` | `3.86×` |
| `wide512` `512×1536` | 0.30 (41) | 128 | 1,587.5 us | **369.5 us** | 541.6 us | 569.5 us | 495.8 us | **`0.75×`** | `3.20×` |
| `wide512` `512×1536` | 0.30 (41) | 256 | 3,052.2 us | **743.6 us** | 1,119.0 us | 1,161.5 us | 1,037.2 us | **`0.72×`** | `2.94×` |
| `wide50m` `768×2048` | 0.30 (67) | 1 | 336.4 us | 60.3 us | 7.4 us | **6.9 us** | 7.0 us | `8.68×` | `48.1×` |
| `wide50m` `768×2048` | 0.30 (67) | 16 | 680.9 us | 154.5 us | 134.0 us | 133.2 us | **110.5 us** | `1.40×` | `6.16×` |
| `wide50m` `768×2048` | 0.30 (67) | 32 | 1,030.3 us | 238.2 us | 261.5 us | 264.7 us | **214.4 us** | `1.11×` | `4.81×` |
| `wide50m` `768×2048` | 0.30 (67) | 64 | 1,722.8 us | 426.7 us | 517.4 us | 484.3 us | **426.4 us** | **`1.00×`** (tie) | `4.04×` |
| `wide50m` `768×2048` | 0.30 (67) | 128 | 3,171.0 us | **817.7 us** | 999.6 us | 1,003.0 us | 850.0 us | **`0.96×`** | `3.73×` |
| `wide50m` `768×2048` | 0.30 (67) | 256 | 6,151.2 us | **1,553.7 us** | 2,129.0 us | 1,973.3 us | 1,699.2 us | **`0.91×`** | `3.62×` |
| `wide50m` `768×2048` | 0.55 (136) | 64 | 1,749.2 us | 851.8 us | 776.6 us | 778.0 us | **615.7 us** | **`1.38×`** | `2.84×` |
| `wide50m` `768×2048` | 0.55 (136) | 256 | 6,084.3 us | 2,997.3 us | 3,339.8 us | 3,066.3 us | **2,364.6 us** | **`1.27×`** | `2.57×` |
| `Llama2-7B` `4096×11008` | 0.30 (431) | 1 | 18,501.5 us | 4,492.4 us | 240.4 us | **159.2 us** | 159.9 us | `28.09×` | `115.7×` |
| `Llama2-7B` `4096×11008` | 0.30 (431) | 16 | 30,089.8 us | 9,057.4 us | 3,053.0 us | 2,784.9 us | **2,118.9 us** | **`4.27×`** | `14.2×` |
| `Llama2-7B` `4096×11008` | 0.30 (431) | 64 | 61,225.2 us | 17,721.6 us | 11,908.7 us | 10,633.5 us | **7,651.0 us** | **`2.32×`** | `8.00×` |
| `Llama2-7B` `4096×11008` | 0.30 (431) | 128 | 105,036.8 us | 31,185.6 us | 23,674.8 us | 21,232.4 us | **15,946.1 us** | **`1.96×`** | `6.59×` |
| `Llama2-7B` `4096×11008` | 0.55 (804) | 64 | 61,396.3 us | 33,359.4 us | 25,799.5 us | 18,781.7 us | **13,367.1 us** | **`2.50×`** | `4.59×` |
| `Llama2-70B` `8192×28672` | 0.30 (939) | 16 | 162,781.7 us | 48,058.4 us | 25,152.8 us | 17,675.7 us | **11,306.7 us** | **`4.25×`** | `14.4×` |

**Key dispatch finding:** Against dense `BCIR FP32 kblock` (or `BF16`), `packed1b_LUT4_mb4` wins
across every tested batch size (`N = 1..256`). Against an unpacked rank-matched `LowRank-FP32` GEMM
(which requires `32×` dequantized weight scratch), the crossover `N*(din, dout, r, kernel, backend)`
depends strongly on shape, rank, and weight reuse: token-major `packed1b_LUT4` crosses at `N = 16`
(`wide512 0.30b`) or `N = 32` (`wide50m 0.30b`), whereas `packed1b_LUT4_mb4` ties at `N = 64` and
crosses at `N >= 128` on `wide50m 0.30b`, stays `1.27×` faster through `N = 256` on `wide50m 0.55b`,
and remains `1.96×–2.50×` faster through `N = 64–128` on `Llama2-7B`. No single fixed `N` cutoff
generalizes across shapes or backends; `COMP-1` must calibrate the crossover per `(din, dout, r,
backend, threads)`.

### 6.2 Native-C QAT Training Step Time & Memory Tax (2 Threads, B4/T64)

| Model | Variant | Step ms (2T) | vs Dense FP32 | Student Arena | Student + Teacher RAM | Deploy Artifact |
|---|---|---:|---:|---:|---:|---:|
| `pilot192_0.86M` | `Dense_FP32_CE` | 17.62 ms | `1.00×` | 26.38 MiB | 26.38 MiB | 3.191 MiB |
| `pilot192_0.86M` | `LowRank_FP16_0.55_KD` | 11.59 ms | `0.66×` (faster) | 14.45 MiB | 40.82 MiB | 0.150 MiB |
| `pilot192_0.86M` | `LittleBit_Res_0.55_CE` | 33.68 ms | `1.91×` slower | 17.64 MiB | 17.64 MiB | 0.150 MiB |
| `pilot192_0.86M` | `LittleBit_Res_0.55_KD` | 40.26 ms | `2.28×` slower | 17.64 MiB | 44.02 MiB | 0.150 MiB |
| `pilot192_0.86M` | `LittleBit_Res_1.00_KD` | 56.92 ms | `3.23×` slower | 23.26 MiB | 49.64 MiB | 0.193 MiB |
| `wide512_5.50M` | `Dense_FP32_CE` | 115.86 ms | `1.00×` | 133.13 MiB | 133.13 MiB | 23.259 MiB |
| `wide512_5.50M` | `LowRank_FP16_0.55_KD` | 46.02 ms | `0.40×` (faster) | 44.41 MiB | 177.53 MiB | 0.646 MiB |
| `wide512_5.50M` | `LittleBit_Res_0.55_CE` | 204.56 ms | `1.77×` slower | 83.57 MiB | 83.57 MiB | 0.650 MiB |
| `wide512_5.50M` | `LittleBit_Res_0.55_KD` | 228.35 ms | **`1.97×` slower** | 83.57 MiB | **216.70 MiB (`1.63×`)** | 0.650 MiB |
| `wide512_5.50M` | `LittleBit_Res_0.30_KD` | 160.64 ms | `1.39×` slower | 60.78 MiB | 193.90 MiB (`1.46×`) | 0.476 MiB |
| `wide512_5.50M` | `LittleBit_Res_1.00_KD` | 356.70 ms | **`3.08×` slower** | 126.11 MiB | **259.24 MiB (`1.95×`)** | 0.975 MiB |

---

## 7. BCIR Integration Decision & `COMP-1` Proposal

These measurements establish a clear separation of concerns for BCIR's roadmap:

1. **Do NOT adopt LittleBit for pretraining or narrow models ($d < 512$).** Pretraining stays on
   BCIR's native FP32/BF16 blocked GEMMs and `TRAIN-OPT-1..7`. Narrow models ($d \le 256$) deploy on
   existing `BCIRQ8` (`8.50 BPW`, lossy per-group Q8 via `quantize_per_group` with near-lossless
   retention on the tested checkpoints — `<= 0.00004` nat CE delta and `100%` `kv_single`/`ordered`
   recall — and per-checkpoint qualification).
2. **Do NOT adopt LittleBit's latent KV-cache replacement.** Use `Arch-4` (`BCIR-MLA` with
   decoupled RoPE) for exact-attention KV compression and `Arch-1`/`Arch-2` (`BCIR-DeltaNet` /
   `BCIR-Hybrid-SWA`) for fixed-state global memory.
3. **Adopt `COMP-1` (Hybrid Q8-Attention + Sub-1-Bit Joint-ITQ Packed-LUT4 MLP Deployment) for
   wide models ($d \ge 512$, `wide10m`, `wide50m`, `BCIR-MoE`, and ingested 1B–70B models):**
   - Keep embeddings, RMSNorms, and sensitive attention (`W_q, W_k, W_v, W_o`, or MLA projections) in
     `BCIRQ8` (`bcir_q8_model.h`, verified per-checkpoint for CE/logit drift and exact-circuit recall
     against the FP32 teacher), while compressing the dominant SwiGLU MLP tensors (`W_gate, W_up,
     W_down` — `~67–75%` of dense weights and `~85%+` of `BCIR-MoE` expert weights!) to `0.30–0.55
     BPW` via Dual-SVID + Joint-ITQ + short post-training KD QAT.
   - Store binary factors in the verified **exact 1.0-bit/weight 2-nibbles-per-byte layout** under
     an append-only `BCIRQ8 v2` tensor encoding tag with per-tensor CRC32 verification.
   - Implement a deterministic `LUT4-LANES` kernel family beside `Q8-LANES` in
     [`runtime/c/bcir_ai_kernels.h`](../../runtime/c/bcir_ai_kernels.h) (`portable` / `AVX2` /
     `AVX-512` bit-identical via fixed ascending 4-weight group accumulation, supporting both
     single-token decode and 4-token weight-reusing micro-batch `mb4`, with `DECODE-THREADS` column
     slicing). Keep any prefill fallback threshold `N*(din, dout, r, backend, threads)` to unpacked
     low-rank GEMM explicitly provisional and calibrated per shape and backend rather than globally
     fixed.

---

## 8. Primary References

- Lee et al., *LittleBit: Ultra Low-Bit Quantization via Latent Factorization* (NeurIPS 2025, v5):
  https://arxiv.org/abs/2506.13771
- Lee et al., *LittleBit-2: Joint Iterative Quantization for Sub-1-Bit Latent Factorization*
  (ICML 2026, v2): https://arxiv.org/abs/2603.00042
- Pinned reproducible native-C experiment bundle, trained FP32 teachers, raw JSON/text outputs,
  3-gate verification log, and SHA-256 provenance manifest (covering `wedge-pr818` checkpoints and
  `gp-global/data/` WikiText-2 splits):
  https://github.com/Mecca-Research/Modular-AI/tree/31bebc39a446b7b61f2a826c8a02a40cff88e0e5/BCIR%20Model%20Training/littlebit-sub1bit-study
  ([PR #227](https://github.com/Mecca-Research/Modular-AI/pull/227),
  [`results/MANIFEST.json`](https://github.com/Mecca-Research/Modular-AI/blob/31bebc39a446b7b61f2a826c8a02a40cff88e0e5/BCIR%20Model%20Training/littlebit-sub1bit-study/results/MANIFEST.json))
