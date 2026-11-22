# Model Architecture Roadmap: Beyond the Global Wedge (2026-11-21)

> Dated 2026-11-21, non-normative research and build roadmap. Synthesizes all six native-C training
> archives retained in [`Mecca-Research/Modular-AI`](https://github.com/Mecca-Research/Modular-AI)
> under `BCIR Model Training/` (`gp-global`, `wedge-next`, `wedge-integrated`, `wedge-query-qware`,
> `wedge-pr818`, and `BCIR-50M-Study` — 180+ training runs, >140,000 native-C updates, 211+
> SHA-verified checkpoints) and specifies the next architecture candidates to implement, measure,
> and gate on BCIR's native C11 trainer ([`bcir_decoder_train.h`](../../runtime/c/bcir_decoder_train.h),
> [`bcir_tensor.h`](../../runtime/c/bcir_tensor.h)).
>
> Companion documents:
> - Training strategy, kernel, memory, and weight-space optimizations:
>   [BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md](BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md)
> - Prior geometric-head review and runtime optimizations (GEO-0 through §40):
>   [BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md](BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md)
> - Native training acceleration ladder:
>   [BCIR_TRAINING_ACCELERATION_ROADMAP.md](BCIR_TRAINING_ACCELERATION_ROADMAP.md)
> - Small-language-model laboratory and byte-native pipeline:
>   [`training/SLM_ROADMAP.md`](../../training/SLM_ROADMAP.md)
> - Bare-metal C LittleBit / LittleBit-2 sub-1-bit compression study (`COMP-1`):
>   [BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md](BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md)

The GEM+ measurement discipline
([BCIR_GEMPLUS_ROADMAP.md §0](BCIR_GEMPLUS_ROADMAP.md#0-the-measurement-discipline-and-why-it-comes-first))
applies to every proposal here: no architecture is promoted on arena size or single-task recall
alone, every new operator carries an independent PyTorch AD reference plus bitwise thread-count
and resume parity, and every capability claim is gated by paired held-out tests and counterfactuals.

---

## 0. Six Structural Laws Established by the Training Archives & Sub-1-Bit Study

The six archives in `Mecca-Research/Modular-AI` (`BCIR Model Training/`) and the companion
bare-metal C LittleBit/LittleBit-2 study
([BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md](BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md))
provide an unusually clean controlled evidence base because every run shares BCIR's deterministic
C11 tensor kernels, fixed random batch hashes, and explicit memory accounting. Six concrete
architectural laws emerge:

1. **Single-vector prefix compression hits a hard write-side associative-binding wall.**
   In `wedge-integrated`, `wedge-query-qware`, and `wedge-pr818`, the rank-22 unweighted prefix sum
   $v_t = (\sum_{j<t} z_j)/\sqrt{\max(t,1)}$ fails even **single-pair** key-value retrieval
   (`5.34%` global, `5.53%` query-contracted vs `100.00%` native `attention32`; chance is `6.25%`)
   and scores `0.00%` prediction change on 4-pair `value_swap` counterfactuals. Adding the
   query-plane interior product $c_j = \sum_i q_i P_{ij}$ lifts ordered recall at $T=2048$ from
   `69.27%` to `89.06%` (and `95.31%` in BF16), and raises `query_change` sensitivity from
   `19.01%` to `31.64%`, yet leaves 4-pair accuracy at `7.29%` (`0.13%` both-correct). Two
   structural causes explain this:
   - *Unselective write*: sparse 3-token triples `(258, key, value)` drown in 64–256 random filler
     tokens because the prefix sum has no learned input-dependent write gate and no local
     adjacency convolution to bind `(key, value)` before aggregation.
   - *Lossy readout*: normalized $c = q^\top P = (v - \langle q,v\rangle q)/\|v - \langle q,v\rangle q\|$
     discards the parallel dot-product projection $\langle q,v\rangle$ and prefix norm $\|v\|$, and
     reuses $W_{\text{reduce}}$ instead of an independent query projector.
2. **Static local/global mixing trades language against long-range recall.**
   Pure global wedge worsens WikiText-2 byte CE by `+0.29` nats relative to attention (`2.4568` vs
   `2.1628` in `gp-global`; `2.4544` vs `2.1540` in `wedge-integrated`). A parameter-free 50/50
   local/global prefix blend closes `78–85%` of that language gap (`2.2203–2.2310`) but collapses
   $T=1024$ recall from `82.03%` down to `44.79%` (`wedge-next`). Language locality and long-range
   memory must live on **separate, gated or interleaved channels**, not a fixed equal average.
3. **Position encoding must match the mixer's algebraic structure.**
   Applying RoPE inside an additive global prefix sum causes destructive phase interference across
   sequence lengths (`gp-global`: $T=1024$ recall `3.65%` with RoPE vs `85.16%` without RoPE,
   `96.09%` with variable-length training). Conversely, attention needs positional structure for
   local order. Global compressed states should be **position-free (NoPE) or decay-gated**, while
   local/exact attention carries decoupled positional embeddings.
4. **Dense projections dominate runtime (Amdahl ceiling), and thinning the mixer hurts extrapolation.**
   At `B1/T1024` (`wedge-next/results/profile-1thread.json`), dense linear GEMMs take **70.9%** of
   1-thread step time (`59.6%` at 4 threads), while wedge geometry plus scans take only **7.04%**
   (1T) / **13.16%** (4T) — capping infinitely fast geometry at `1.076×` (1T) / `1.152×` (4T).
   Meanwhile, thinning the wedge readout via `Q=32` Tucker factorization (`lowrank32`), product
   subspaces (`multi12`), or post-training SVD (`Q=32/64`) cuts $T=1024$ recall from `82–88%` to
   `18–65%` and can even *increase* the activation arena (`lowrank32_budget`: `32.06 MiB` vs
   `29.36 MiB`). Next architectures must cast their **core mixing FLOPs into BCIR's blocked GEMM
   engines** (`kblock` / `bf16` AMX & AVX-512) rather than relying on scalar serial scans.
5. **Active-parameter scaling without sparsity or patching hits CPU compute and persistent-state walls.**
   In `BCIR-50M-Study`, `wide50m` (`50.54M` params, 8L, $d=768$) runs cleanly in native C (`440 tok/s`
   on 2 threads, `877.5 MiB` buffers), with `wide10m` (`9.84M`, 4L, $d=512$) beating deeper
   `reference10m` (`6L, d=384`) on quality, speed, and buffers. However, `771.25 MiB` (**87.9%**)
   of `wide50m` memory is persistent FP32 state (`16 B/param`), raw UTF-8 byte context $T=64$ spans
   only ~12 words, and Experiment B showed individual `wide50m` MLP columns can be amplified
   `256×` with `-0.000045` CE impact — signaling high MLP activation sparsity that conditional
   compute (MoE) or byte-latent patching can exploit.
6. **Sub-1-bit latent matrix factorization is a wide-MLP deployment multiplier, not a narrow-model
   or training replacement.**
   Native-C evaluation of LittleBit / LittleBit-2 (`BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md`)
   establishes four quantitative boundaries: (a) FP16 multi-scale vectors impose a hard
   $32(1/d_{\text{in}} + 1/d_{\text{out}})$ floor (`0.425 BPW` at $d=128$, `0.313 BPW` on 4:1 GQA
   `K/V` at $d=512$), restricting extreme sub-1-bit gains to wide layers ($d \ge 512$); (b) at
   $d=512$, Joint-ITQ 2-path Residual reaches `7.32` PPL at `0.545 BPW` (`+0.032` nats vs `7.09`
   FP32) and beats matched-byte low-rank FP16 (`11.78` PPL) by `>4` PPL, while zero-shot PTQ breaks
   attention circuits (`0.0–9.4%` `kv_single`) yet spares SwiGLU MLPs (`100.0%` zero-shot);
   (c) exact 1.0-bit/weight 2-nibbles-per-byte AVX-512 LUT4 (`_mm512_permutexvar_ps`) accelerates
   `N=1` decode by `18.1×` (`wide50m 0.30b`) to `224.5×` (`70B 0.10b`) over FP16 (`14.9–130×` over
   BCIR Q8), while prefill ($N \ge 64$) crosses back to blocked FMA; and (d) QAT+KD imposes a
   `2.0–3.1×` training-step slowdown and `1.63×` RAM tax, keeping it strictly post-training.

---

## 1. Prerequisite Infrastructure: Pluggable Sequence-Mixer ABI (`MIXER-EXT`, ABI 6)

All five `wedge-*` archives had to perform textual C source grafting onto `bcir_decoder_train.c`
because [`bcir_decoder_spec`](../../runtime/c/bcir_decoder_train.h) is hardwired to uniform causal
GQA across every layer. Before landing any new model family, BCIR should add a minimal,
zero-allocation mixer extension boundary:

- **Per-layer mixer kind table** (`uint8_t layer_mixer[64]` in ABI 6 spec, defaulting to all
  `BCIR_DT_MIXER_GQA` so ABI 5 plans stay 100% bit-identical):
  - `BCIR_DT_MIXER_GQA = 0`: existing causal GQA (full context).
  - `BCIR_DT_MIXER_SWA = 1`: sliding-window causal GQA with window $W$ (`O(B H T W)` probs/scratch).
  - `BCIR_DT_MIXER_DELTANET = 2`: chunkwise gated delta-rule associative memory (Arch-1).
  - `BCIR_DT_MIXER_KEYED_WEDGE = 3`: write-gated symmetric+antisymmetric bilinear memory (Arch-3).
  - `BCIR_DT_MIXER_MLA = 4`: multi-head latent attention with decoupled RoPE (Arch-4).
- **Unified per-mixer contract callbacks** (static table, zero heap, compatible with `REMAT`,
  `WS-PLAN`, `PX-2/3b/4` `bcir_decoder_run_program`, and `train.plan.mismatch`):
  1. `param_count(spec, layer)` and explicit parameter layout descriptors (so `NativeDecoder`
     initialization, checkpoint serialization, and Weight-Space permutation/patch tools know exact
     tensor/axis identities).
  2. `activation_floats(spec, layer)` and `scratch_floats(spec, layer)` (allowing heterogeneous
     layer schedules such as 3:1 SWA + DeltaNet to share the single `REMAT` recompute slot sized to
     $\max_{l \in \text{recompute}} \text{act}(l)$).
  3. `workspace_need(spec, layer, contract, threads)` and `workspace_floor(...)` (integrating
     chunked GEMM packing into `bcir_decoder_workspace_need`).
  4. `inference_state_bytes(spec, layer)` (exposing exact streaming KV or recurrent matrix state
     size to HAM and the C inference decoder `bcir_decode.c`).
- **Launch-program record opcodes** (`BCIR_DT_REC_CONV1D`, `BCIR_DT_REC_CONV1D_VJP`,
  `BCIR_DT_REC_DELTA_CHUNK`, `BCIR_DT_REC_DELTA_CHUNK_VJP`, `BCIR_DT_REC_SWA`,
  `BCIR_DT_REC_SWA_VJP`, `BCIR_DT_REC_MOE_ROUTE`) validated by `bcir_decoder_run_program` before
  any buffer write, preserving the R1–R25 verifier and `place_by_lifetime` guarantees.

---

## 2. Six Concrete Architecture Build Proposals

The global wedge was one geometric point in a much broader design space of sub-quadratic,
hybrid, and hardware-sympathetic sequence models. The six candidates below are ordered by expected
return on BCIR's current native C11 CPU/AMX/AVX-512 stack and directly target the verified failure
modes of the six `Modular-AI` studies.

### Summary Comparison Matrix

| ID | Architecture | Primary Gap Solved | Core Mixing Complexity (Train / Infer State) | Fits BCIR Blocked GEMM / AMX? | Priority & Target Scale |
|---|---|---|---|---|---|
| **Arch-1** | **Chunkwise Gated DeltaNet** (`BCIR-DeltaNet`) | Write-side KV binding, filler suppression, multi-key overwrite | $O(B T (d^2 + C d))$ / $O(h d_k d_v)$ fixed | **Yes (>90% GEMM via WY chunks)** | **P0 (Research)** · 526k → 10M → 50M |
| **Arch-2** | **Hybrid Sliding-Window GQA + Associative Global** (`BCIR-Hybrid-SWA`) | Local LM vs global recall trade-off without 50/50 interference | $O(B T (W d + d^2))$ / $O(W d_{kv} + h d_k d_v)$ | **Yes (SWA + DeltaNet/GQA)** | **P0 (Production)** · 526k → 10M → 50M |
| **Arch-3** | **Write-Gated Keyed-Wedge** (Sym + Antisym Bilinear Memory) | Repairs all 3 geometric flaws of `wedge-query-qware` | $O(B T (d^2 + C r^2))$ / $O(h r^2)$ fixed | **Yes (chunked $K^\top V$ GEMMs)** | **P1 (Geometric)** · 526k → 10M |
| **Arch-4** | **Multi-Head Latent Attention** (`BCIR-MLA`, Decoupled RoPE) | Exact attention quality with 4–8× smaller KV cache & NoPE content | $O(B T^2 d_h)$ / $O(T (d_c + d_R))$ | **Yes (`ATT-CONTIG`/`BF16-ATTN`)** | **P1 (Exact Attn)** · 10M → 50M |
| **Arch-5** | **Byte-Latent Patch Decoder** (`BCIR-BLT`) | 4× effective byte context & 16× less core attention FLOPs | Core runs at $T/4$ latent patches | **Yes (standard GQA/SWA core)** | **P1 (Byte/SLM)** · 10M → 50M |
| **Arch-6** | **Deterministic Fixed-Capacity Sparse SwiGLU MoE** (`BCIR-MoE`) | 50M capacity at ~15M active FLOPs/token; exploits MLP sparsity | Active FLOPs ~30% of dense 50M | **Yes (static expert GEMM groups)** | **P2 (Scale)** · 50M total / 15M active |

---

### Arch-1: Chunkwise Gated DeltaNet (`BCIR-DeltaNet`) — *P0 Associative-Memory Candidate*

**Motivation:** Directly cures the exact write-side failure diagnosed in `wedge-query-qware` §4 and
`wedge-pr818` §6.1 while retaining linear $O(T)$ training memory and fixed-size streaming state.

**Equations (per head $h$, head dimensions $d_k, d_v$):**
1. **Short local causal convolution (`k=4`) before mixing** (captures adjacent `(258, key, value)`
   triples and local byte n-grams — only $4 \cdot d$ params/layer):
   $$\tilde q_t, \tilde k_t, \tilde v_t = \mathrm{SiLU}\!\left(\mathrm{CausalConv1D}_{k=4}(W_q h_t, W_k h_t, W_v h_t)\right)$$
   $$q_t = \frac{\tilde q_t}{\max(\|\tilde q_t\|_2, 10^{-6})}, \qquad k_t = \frac{\tilde k_t}{\max(\|\tilde k_t\|_2, 10^{-6})}$$
   *(Unit $L_2$ normalization on $q_t, k_t$ guarantees $\beta_t k_t k_t^\top$ has spectral radius $\le \beta_t$, preventing state explosion without wedge Gram-determinant cancellation.)*
2. **Input-dependent decay $\alpha_t \in (0,1)$ and write/erase strength $\beta_t \in (0, 2)$**
   (Householder reflection regime $\beta_t \in (0,2)$ enables exact key-value overwriting and state
   rotation, while $\beta_t \to 0$ suppresses filler tokens):
   $$\alpha_t = \sigma(W_\alpha h_t + b_\alpha), \qquad \beta_t = 2\,\sigma(W_\beta h_t + b_\beta)$$
3. **Gated Delta-Rule matrix recurrence** on associative state $S_t \in \mathbb{R}^{d_k \times d_v}$:
   $$S_t = \alpha_t S_{t-1}\left(I - \beta_t k_t k_t^\top\right) + \beta_t k_t v_t^\top = \alpha_t S_{t-1} + \beta_t k_t \left(v_t^\top - k_t^\top S_{t-1}\right), \qquad o_t = S_t^\top q_t$$
   followed by per-head RMSNorm and an output SwiGLU/sigmoid gate $m_t = W_o(\mathrm{SiLU}(W_g h_t) \odot \mathrm{RMSNorm}(o_t))$.

**BCIR Native C Lowering (WY Chunkwise Parallel Form, chunk size $C = 32$ or $64$):**
- Partition $T$ into $T/C$ chunks of $C=32$ tokens (matching `BCIR_MM_BF16_BLOCK = 32` and AVX-512
  16-lane tiling!).
- Within each chunk, solve the tiny $32\times 32$ strictly lower-triangular WY system
  $T_{\text{wy}} = (I + \mathrm{strictLower}(\mathrm{diag}(\beta)\,(K K^\top \odot \Gamma)))^{-1}$
  in FP64 (guarantees exact conditioning, deterministic forward-substitution in $O(C^2)$ per head,
  zero matrix-inversion instability).
- Compute pseudo-values $U = T_{\text{wy}}(\mathrm{diag}(\beta) V)$ and keys $W = T_{\text{wy}}(\mathrm{diag}(\beta\gamma) K)$, then update inter-chunk state via **two standard blocked GEMMs** (`bcir_tensor_mm_kblock_pool` / `bcir_tensor_mm_bf16_pool`):
  $$S_{[c+1]} = \Gamma_c S_{[c]} + K_{[c]}^\top \left(U_{[c]} - W_{[c]} S_{[c]}\right), \qquad O_{[c]} = (Q_{[c]} K_{[c]}^\top \odot M_{\text{causal}})\left(U_{[c]} - W_{[c]} S_{[c]}\right) + Q_{[c]} S_{[c]}$$
- **Why this fits BCIR dramatically better than `gp-global`:** Over **90%** of mixer FLOPs run inside
  BCIR's existing AVX-512 / AMX `kblock` and `bf16` GEMM kernels; parallelization spans `(B, H)`
  even at `B = 1` (fixing `wedge-next`'s 1.21× serial-scan ceiling); training caches only chunk
  boundary states $S_{[c]}$ (`O(B H (T/C) d_k d_v)` — **8× smaller** than per-token states and
  trivially recomputable under `REMAT`!).

**Matched Test Recipes:**
- **526k parity test:** $L=3, d=128, h=4, d_k=d_v=32, d_{\text{ff}}=248$ (matched to `525,696` ±0.5%; fixed streaming state = **$16\text{ KiB/layer}$** vs `78.5 MiB` attention training arena at $T=1024$).
- **10M & 50M scale tests:** matched directly to `wide10m` ($L=4, d=512, h=8, d_k=d_v=64$, `128 KiB/layer` state) and `wide50m` ($L=8, d=768, h=12, d_k=d_v=64$, `192 KiB/layer` state).

**Falsifiable GO/STOP Gate:** Must achieve **`kv_single >= 99%`**, **4-pair `kv_curriculum >= 80%`**,
**`value_swap` both-correct `>= 75%`**, **ordered $T=2048 \ge 92\%$**, and **WikiText-2 byte CE
$\le 2.17$** at the 526k matched 32 MiB budget (beating both `query_full` and `attention32` on the
4-pair binding suite).

---

### Arch-2: Interleaved Sliding-Window GQA + Global Associative Mixer (`BCIR-Hybrid-SWA`) — *P0 Production-Scale Candidate*

**Motivation:** `wedge-next` and `wedge-integrated` proved that exact local attention dominates
byte/token language modeling (`2.15–2.17` vs `2.45`), while full quadratic causal attention blows
up activation memory at long contexts (`78.5 MiB` at 526k B1/T1024; `1.50 GiB` of probs alone at
50M B4/T2048). Interleaving **Sliding-Window Attention (SWA)** with **Global Associative Memory**
(following the Jamba / Samba / Qwen3-Next / Gemma-3 hybrid pattern) resolves both bottlenecks at
once without the static 50/50 interference that crippled `wedge-next`'s equal hybrid.

**Layer Schedule (parameter-matched to `wide10m` and `wide50m`):**
- **4-layer (`wide10m` scale, 3:1 schedule):** `[SWA(W=128), SWA(W=128), DeltaNet(NoPE), SWA(W=128)]`
  (ablation control: replace `DeltaNet` at layer 2 with 1 Full-GQA(NoPE) layer).
- **8-layer (`wide50m` scale, 3:1 schedule):** `[SWA, SWA, DeltaNet, SWA, SWA, SWA, DeltaNet, SWA]`
  (or `[SWA, DeltaNet] × 4`, 1:1 schedule, compared at identical parameter & wall-time budget).
- **12-layer (`deep50m` scale):** 3:1 `[SWA, SWA, SWA, DeltaNet] × 3`.

**Exact Memory Win (Sliding-Window Kernel `BCIR_DT_MIXER_SWA`):**
- A banded causal window of width $W \in \{64, 128\}$ (multiple of 32 for `BF16-ATTN` & `P-BANDS`)
  stores only $B \cdot H \cdot T \cdot W$ probabilities instead of $B \cdot H \cdot T^2$.
- At `wide50m` ($L=8, H=12$) with `B=4, T=2048`:
  - Full attention probs (all 8 layers): $8 \times 4 \times 12 \times 2048^2 \times 4\text{ B} = \mathbf{6.00\text{ GiB}}$ (exceeds the 4 GiB native admission cap!).
  - 6 SWA($W=128$) + 2 DeltaNet($C=32$) layers: $6 \times 4 \times 12 \times 2048 \times 128 \times 4\text{ B} + 2 \times 4 \times 12 \times 64 \times 64^2 \times 4\text{ B} = \mathbf{288\text{ MiB} + 96\text{ MiB} = 384\text{ MiB}}$ — a **15.6× reduction** in mixer activation storage with zero accuracy penalty on local syntax and full global recall via the DeltaNet layers!
- Uses RoPE on SWA layers (strictly within window $W$, where RoPE never extrapolates out of
  distribution!) and **NoPE on DeltaNet / periodic global layers** (directly leveraging `gp-global`'s
  position-ablation discovery).

**Falsifiable GO/STOP Gate:** At 526k and 10M/50M scales, must **match or beat** full-GQA WikiText-2
CE at $T=64$ and $T=512$, beat full-GQA on `T=1024/2048` ordered & KV recall, and fit `B4/T2048`
training inside the 4 GiB native buffer budget at 50M without falling below `0.85×` full-GQA step
speed at $T=256$.

---

### Arch-3: Write-Gated Keyed-Wedge (`Keyed-Wedge` — Symmetric + Antisymmetric Bilinear Memory) — *P1 Geometric Evolution*

**Motivation:** Before retiring or superseding the Grassmann/Plücker line of research, isolate
whether **exterior-product (antisymmetric) algebra** genuinely adds value over plain symmetric
linear attention *once the write-side bottleneck and lossy normalization are fixed*.

**Three Targeted Repairs to `wedge-query-qware`:**
1. **Local adjacency conv (`k=4`) + learned write gate $w_t = \sigma(W_w h_t)$ + exponential decay
   $\gamma_t = \sigma(W_\gamma h_t + b_\gamma)$:** replaces the unweighted prefix sum so filler
   tokens receive $w_t \approx 0$ and adjacent `(key, value)` pairs enter the write path together.
2. **Matrix-valued bilinear state decomposed into symmetric and antisymmetric (2-form) parts**
   ($r = 24$ per head, $h=2$ or single head $r=32$):
   $$M_t = \gamma_t M_{t-1} + w_t (k_t \otimes v_t) = M_t^{\text{sym}} + M_t^{\text{wedge}}$$
   $$M_t^{\text{sym}} = \gamma_t M_{t-1}^{\text{sym}} + \tfrac{w_t}{2}(k_t v_t^\top + v_t k_t^\top), \qquad M_t^{\text{wedge}} = \gamma_t M_{t-1}^{\text{wedge}} + \tfrac{w_t}{2}(k_t \wedge v_t)$$
   Notice that $M_t^{\text{wedge}}$ is a **general (higher-rank) multivector / 2-form** in
   $\bigwedge^2 \mathbb{R}^r$, NOT restricted to a single rank-2 decomposable plane! It can
   simultaneously store up to $\lfloor r/2 \rfloor$ orthogonal oriented pairs without collapsing
   them into one plane.
3. **Independent query projector $q_t = W_q h_t$ contracting BOTH channels without unit-norm
   cancellation:**
   $$c_t^{\text{sym}} = M_t^{\text{sym}} q_t, \qquad c_t^{\text{wedge}} = (M_t^{\text{wedge}})^\top q_t, \qquad g_t = W_{\text{sym}}\,\mathrm{RMSNorm}(c_t^{\text{sym}}) + W_{\text{wedge}}\,\mathrm{RMSNorm}(c_t^{\text{wedge}})$$
   Using $\mathrm{RMSNorm}$ on the contracted output vector $c_t \in \mathbb{R}^r$ instead of
   dividing $u \wedge v$ by $\max(\|u \wedge v\|, 10^{-6})$ **completely eliminates**:
   - the `global_core.h` $10^6$ clamped-norm gradient spike (`344k–595k` norm blowup),
   - the FP32 common-mode cancellation floor (`0.0262` L2 distortion), and
   - the loss of $\langle q, k \rangle$ parallel alignment and prefix magnitude.

**Controlled Scientific Ablation (same params, data, seeds, optimizer):**
Compare three toggles on the exact same `kv_single`, `kv_curriculum`, `value_swap`, `ordered`,
and WikiText-2 suite: (a) **symmetric-only** ($M^{\text{sym}}$), (b) **wedge-only**
($M^{\text{wedge}}$), and (c) **combined** ($M^{\text{sym}} + M^{\text{wedge}}$). This settles
definitively whether antisymmetric 2-forms improve order-sensitive retrieval and swap resistance
over symmetric linear attention when both have proper write gates.

---

### Arch-4: Multi-Head Latent Attention (`BCIR-MLA`, Low-Rank KV + Decoupled RoPE) — *P1 Exact-Attention Compressor*

**Motivation:** For workloads requiring exact quadratic attention (no recurrent compression loss at
all), KV-cache size and memory bandwidth dominate long-context inference (`bcir_decode.c`) and
activation storage. `BCIR-50M-Study` already validated GQA (`3:1` on `wide50m`, `5:1` on `deep50m`).
Multi-Head Latent Attention pushes compression further while applying `gp-global`'s positional
finding directly to exact attention.

**Equations:**
- Compress hidden state into a single shared KV latent $c_t^{KV} = W_{DKV} h_t \in \mathbb{R}^{d_c}$
  (e.g., $d_c = 128$ at `wide50m` $d=768$ — **6× smaller** than full $K, V$, **2× smaller** than
  4-KV-head GQA!) and up-project content keys/values $k_t^C = W_{UK} c_t^{KV}$, $v_t^C = W_{UV} c_t^{KV}$.
- **Decoupled RoPE**: carry position strictly on a small shared RoPE key $k_t^R = \mathrm{RoPE}(W_{KR} h_t) \in \mathbb{R}^{d_R}$ ($d_R = 32$) and per-head RoPE queries $q_{t,h}^R = \mathrm{RoPE}(W_{QR,h} h_t)$, concatenating $q_{t,h} = [q_{t,h}^C; q_{t,h}^R]$ and $k_{t,h} = [k_{t,h}^C; k_t^R]$.
- **Inference weight absorption in `bcir_decode.c`:** At decode time, $W_{UK}$ absorbs into $W_Q$
  and $W_{UV}$ absorbs into $W_O$, so the decoder caches ONLY `[c_t^{KV}; k_t^R]` (`160` floats/token/layer
  at 50M vs `512` floats for 4-KV-head GQA — **3.2× smaller KV cache** and **3.2× less KV DRAM
  bandwidth** during autoregressive Q8/FP32 generation!).

**BCIR Integration:** Reuses BCIR's existing `bcir_tensor_mm_kblock_pool`, `bcir_tensor_rope_table_pool`
(over $d_R$ only), `bcir_tensor_attention_contract_variant_scan_ws_pool` (`ATT-CONTIG` / `P-BANDS` /
`BF16-ATTN`), and upcoming tiled online-softmax training attention (`TRAIN-OPT-5`).

**Why `BCIR-MLA` strictly supersedes LittleBit latent KV caching:** The LittleBit study
([BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md](BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md) §2)
proved that caching binary-factorized pre-up-projection KV latents *expands* KV cache under GQA +
2-path Residual + `kv_factor=4` (`0.89×` on `wide50m`, `0.61×` on `Llama3-8B`, `0.55×` on
`Llama2-70B` at `0.55 BPW`) and cannot commute with post-projection RoPE. `BCIR-MLA` solves exact
KV compression cleanly via decoupled RoPE and weight absorption with zero sub-1-bit attention
degradation.

---

### Arch-5: Byte-Latent Patch Decoder (`BCIR-BLT`) — *P1 Byte-Native Scale Multiplier*

**Motivation:** Every study in `Modular-AI` trained directly on raw UTF-8 bytes (260-token vocab).
In `BCIR-50M-Study`, training `wide50m` at `T = 64` raw bytes gave the 50.5M model only ~12 English
words per sequence while spending 50.5M parameter multiplications on every single ASCII character!
Meanwhile, [`training/SLM_ROADMAP.md`](../../training/SLM_ROADMAP.md) §3.6 (`SLM-7b`) already
identifies grapheme/byte-native patching as the alternative to BPE.

**Architecture (3-stage hierarchical decoder, 100% deterministic C11):**
1. **Local Byte Encoder (1 lightweight layer, $d_{\text{byte}} = 256$, SWA $W=32$ or `conv1d`):**
   Processes raw bytes $x_{0..T-1}$ and pools fixed **4-byte patches** (Stage 1: deterministic, static
   shape $T_p = T/4$, zero ragged dynamic allocation!) — or Stage 2 **grapheme/delimiter-aligned
   patches** bounded to max patch length $P_{\max}=8$ using Unicode UAX #29 boundaries from UC-6 —
   into patch embeddings $e_0, \dots, e_{T_p - 1} \in \mathbb{R}^{d_{\text{core}}}$.
2. **Deep Latent Core (`wide10m` or `wide50m`, GQA or Arch-2 Hybrid-SWA):**
   Runs on $T_p = T/4$ latent patches! For `T = 256` raw bytes, the 50M core executes only
   **$T_p = 64$ steps** — **4× higher byte throughput/second** and **16× lower quadratic attention
   cost** than running `wide50m` at raw byte length `256`!
3. **Local Byte Decoder (1 lightweight causal cross-patch/SWA layer, $d_{\text{byte}} = 256$):**
   Conditions on the causal previous patch representation $z_{\lfloor t/4 \rfloor - 1}$ plus the
   preceding bytes within the current patch to predict the next raw UTF-8 byte `0..255`.

**Why this matters for BCIR:** Eliminates BPE tokenizer dependency while making 50M byte-level
training 3–4× faster in bytes/sec on CPU, and feeds directly into `SLM-7b` once Unicode UAX #29
segmentation lands.

**Falsifiable GO/STOP Gate:** Matched to `wide10m` and `wide50m` on **equal wall-clock training
time** AND **equal raw training bytes** over WikiText-2 raw bytes at $T=256$ and $T=512$; must
achieve strictly lower test bits-per-byte (BPB) at equal wall time with zero tokenizer artifacts.

---

### Arch-6: Deterministic Fixed-Capacity Sparse SwiGLU MoE (`BCIR-MoE`) — *P2 Compute-Sparse 50M+ Scaling*

**Motivation:** Three direct empirical signals:
- Dense `wide50m` took **4.75× longer** per token than `wide10m` (`440` vs `2,091 tok/s`), because
  every token multiplied all 50.54M weights (`BCIR-50M-Study`).
- In Experiment B, multiplying an entire middle-layer `wide50m` `down_proj` column by `256×` moved
  test CE by `-0.000045` nats (0.26% top-1 flips), confirming massive token-level MLP sparsity.
- In [BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md](BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md) §4.2,
  SwiGLU MLP layers absorbed sub-1-bit factorization with **100.0% zero-shot `kv_single` circuit
  retention** (while attention collapsed to `9.4%`). Because routed experts are `>80%` of `BCIR-MoE`
  total bytes, pairing `Arch-6` with `COMP-1` post-training expert compression directly removes MoE's
  cold-expert DRAM footprint bottleneck at decode time.

**BCIR-Safe Deterministic MoE Design (Zero Heap, Bit-Exact Across Thread Counts):**
- Replace each dense SwiGLU MLP ($d_{\text{ff}} = 2048$ in `wide50m`) with **1 shared always-on
  expert** ($d_{\text{ff,shared}} = 512$) + **$E = 8$ routed experts** ($d_{\text{ff,exp}} = 512$ each,
  top-$k = 2$ active per token $\Rightarrow$ total active $d_{\text{ff}} = 1536$, or $E=8$, $d_{\text{ff,exp}}=384$,
  top-2 + shared `384` $\Rightarrow$ ~**16–20M active params** out of **~50M total params**!).
- **Static Fixed-Capacity Buffer Contract (`capacity = ceil(factor * k * B * T / E)`, `factor = 1.25`,
  rounded up to multiple of 32 for `BF16-PANELS`!):**
  - Pre-allocate exact static token-dispatch buffers in the activation arena (`E × capacity × d`);
    no dynamic allocation, 100% compatible with BCIR's `place_by_lifetime` static solver and
    `bcir_decoder_make_plan`!
  - Deterministic tie-breaking by ascending expert index and token index guarantees bit-identical
    routing across 1..N threads.
  - Aux-loss-free bias load balancing (updating per-expert router bias outside AdamW gradients on
    fixed step cadence) avoids auxiliary-loss gradient interference.
  - Each expert's SwiGLU runs as a standard batched `bcir_tensor_mm_kblock_pool` / `bf16` call over
    `(capacity, d_ff_exp, d)` — preserving full AVX-512 / AMX vector efficiency!

**Falsifiable GO/STOP Gate:** Compare **50M-total / 16M-active `BCIR-MoE`** against dense `wide10m`
and dense `wide50m` at **equal wall-clock time** and **equal training tokens** (≥10M tokens); must
beat dense `wide50m` by **$\ge 2.2\times$ tokens/sec** while achieving lower held-out CE than
`wide10m` at matched active FLOPs (tracking expert load-balance utilization $\ge 85\%$).

---

### Deployment Compression Track (`COMP-1`): Hybrid Q8-Attention + Sub-1-Bit Joint-ITQ Packed-LUT4 MLP

**Motivation ([BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md](BCIR_LITTLEBIT_SUB1BIT_STUDY_2026-11-22.md)):**
Standard `BCIRQ8 v1` (`runtime/c/bcir_q8_model.h`) is lossless at `8.50 BPW` (group-32 int16 exponents) and remains the
right deployment format for narrow models ($d < 512$) and exact-circuit attention layers. For wide
models ($d \ge 512$ — `wide10m`, `wide50m`, `BCIR-MoE`, and ingested 1B–70B checkpoints), SwiGLU
MLP weights represent `67–85%` of total bytes and dominate autoregressive decode DRAM traffic while
tolerating extreme sub-1-bit latent factorization far better than attention.

**Specification (Post-Training Export Extension to `BCIRQ8` & `bcir_ai_kernels.h`):**
1. **Strict width & layer admission gates:** Apply sub-1-bit compression **only** to wide linear
   projections with $d_{\text{in}}, d_{\text{out}} \ge 512$ (avoiding the $32(1/d_{\text{in}} + 1/d_{\text{out}})$
   scale floor). Because attention suffers **`12.5×` more CE damage per parameter** than SwiGLU MLP
   at `0.30 BPW` (`+0.1235` nats from `21.7%` params vs `+0.0355` nats from `78.3%`), default policy
   keeps embeddings, RMSNorms, and attention/MLA projections in lossless `BCIRQ8` (`8.50 BPW`) while
   compressing SwiGLU MLPs (`W_gate, W_up, W_down` and MoE experts) to `0.30–0.55 BPW` via
   **Dual-SVID + 50-iteration Joint-ITQ + `Clipped-STE` + `Output KL (λ = 0)`** QAT (validated on
   `wide512`: `7.26` PPL hybrid at `2.08` effective body BPW vs `7.09` FP32, and `7.33` PPL all-layer
   at `0.311 BPW` with `kv_factor = 2.0`, beating published Eq. 11 `8.75` PPL by `-1.42` PPL at
   `1.54×` faster QAT). Use 1-path if ever forced onto narrow $d \le 256$ layers; use 2-path
   Residual for $d \ge 512$.
2. **Exact 1.0-bit/weight 2-nibbles-per-byte `BCIRQ8 v2` tensor encoding:** Pack 4-weight binary
   LUT nibbles for rows `[r0..r0+15]` in low nibbles and `[r0+16..r0+31]` in high nibbles of the
   same 16 bytes (verified to halve DRAM/L3 traffic and deliver `3.14×` speedup over 1-byte/nibble
   on 70B layers), protected by per-tensor CRC32 and source SHA-256 provenance matching `BCIRQ8 v1`.
3. **Deterministic `LUT4-LANES` kernel contract & prefill/decode dispatch:** Add `LUT4-LANES`
   (`portable` / `AVX2` / `AVX-512` bit-identical via fixed ascending 4-weight group order, plus
   `DECODE-THREADS` column slicing) beside `Q8-LANES` in `runtime/c/bcir_ai_kernels.h`. Dispatch
   `LUT4-LANES` for autoregressive decode ($N \le 16$, measured `14.9–130×` faster than Q8), and
   switch to blocked FP32/BF16 GEMM for batched prefill ($N \ge 64$, where FMA overtakes LUT4).

**Falsifiable GO/STOP Gate:** On `wide10m` / `wide50m` (`d >= 512`), `COMP-1` must achieve
**`>= 8×` end-to-end MLP weight reduction** vs FP32 (`>= 3×` total artifact reduction vs `BCIRQ8`),
**`>= 5×` single-token decode speedup** over `BCIRQ8` on AVX-512 hosts, `0` bit drift between
portable and AVX-512 `LUT4-LANES` instances, and `<= 0.04` nat WikiText-2 CE drift with `>= 99%`
exact associative recall retention.

---

## 3. Unified Architecture Qualification Ladder (`ARCH-GATE`)

Every architecture built under this roadmap must clear the **same six-stage gate** before promotion
from experimental overlay to registered BCIR model family (preventing the single-task shortcuts
caught in `gp-global` and `wedge-next`):

| Gate | Scope | Mandatory Pass Criterion |
|---|---|---|
| **G1. Numerical & ABI Law** | PyTorch AD oracle (incl. duplicate prefix & all-masked refusal), 1T vs 4T state identity, checkpoint resume, UBSan, `REMAT` parity, `run_program` parity | Max grad rel-L2 `< 1e-6`; 0 bit drift across threads/resume/REMAT; 0 UBSan findings; 0 heap allocs in step |
| **G2. Associative Binding** | `kv_single` (1-pair), `kv_curriculum` (1/2/4/8-pair), `value_swap`, `query_change` at $T \in \{128, 256, 512, 1024\}$ | `kv_single >= 99%`; 4-pair `>= 80%`; `value_swap` both-correct `>= 75%`; `query_change` both-correct `>= 75%` |
| **G3. Ordered Extrapolation** | Shared-alphabet `0..15` ordered recall trained at variable $T \in \{64,128,256\}$, evaluated at $T \in \{512, 1024, 2048, 4096\}$ + `T512` swap | `T1024 >= 98%`; `T2048 >= 90%`; `T512` swap both-correct `>= 98%` |
| **G4. Language Quality** | WikiText-2 raw bytes (and BPE where matched) at $T \in \{64, 256, 512, 1024\}$ under matched controlled buffers AND matched wall time | Match or beat `attention32` / `wide10m` CE at equal wall time; no degradation at $T=512/1024$ |
| **G5. Multi-Scale Exposure** | 10M (`wide10m`) and 50M (`wide50m`) runs with **$\ge 10\text{M}$–$50\text{M}$ training tokens** ($\ge 0.2$–$1.0$ tok/param, $\ge 3$ seeds, proper $\mu$P/LR tuning) | Separate equal-token and equal-wall-time curves; document gradient-clip saturation rate `< 5%` after warmup |
| **G6. Systems Pareto** | Native controlled buffers (`16·P + act + scratch + ws`), process peak RSS, step ms (1T/4T/8T), inference state B/layer, and time-to-quality | Report full 5-tuple; never claim a win from activation arena size alone |

---

## 4. Phased Build Sequence

1. **Phase M1 (`MIXER-EXT`, ABI 6):** Land the pluggable per-layer mixer interface and launch-program
   opcodes (`bcir_decoder_train.h`, `native_program.py`, `native_pack.py`), keeping ABI 5 GQA
   bit-identical.
2. **Phase M2 (`Arch-1` + `Arch-2` + `Arch-3` at 526k):** Implement `BCIR-DeltaNet`,
   `BCIR-Hybrid-SWA`, and `Keyed-Wedge` in native C11 and run the complete G1–G4 ladder against
   `attention32` and `query_full` on identical paired seeds. Promote only architectures clearing G2
   and G4.
3. **Phase M3 (`Arch-4` + `Arch-5` at 10M):** Implement `BCIR-MLA` and `BCIR-BLT` (4-byte static
   patches first, UAX #29 dynamic patches after UC-6 / `SLM-7b`) and benchmark equal-wall-time byte
   LM + KV-cache decode bandwidth against `wide10m`.
4. **Phase M4 (`Arch-2` / `Arch-1` / `Arch-6` at 50M+):** Scale the M2/M3 winners plus
   `BCIR-MoE` to 50M parameters under the upgraded training/kernel stack
   ([`TRAIN-OPT-1..8`](BCIR_TRAINING_STRATEGY_AND_OPTIMIZATION_ROADMAP_2026-11-21.md)) with
   $\ge 10\text{M}$ training tokens and archive all raw checkpoints/logs in
   `Mecca-Research/Modular-AI` under `BCIR Model Training/`.
5. **Phase M5 (`COMP-1` Deployment Export):** Add `BCIRQ8 v2` hybrid Q8-Attention + Sub-1-Bit
   Joint-ITQ Packed-LUT4 MLP export and deterministic `LUT4-LANES` kernels (`bcir_ai_kernels.h`,
   `bcir_llama.c`) targeting `wide10m`, `wide50m`, `BCIR-MoE`, and ingested open-weight models
   ($d \ge 512$).

---

## 5. Primary Technical References

- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule* (2024): https://arxiv.org/abs/2412.06464
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (NeurIPS 2024): https://arxiv.org/abs/2406.06484
- Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training* (ICML 2024): https://arxiv.org/abs/2312.06635
- Lieber et al., *Jamba: A Hybrid Transformer-Mamba Language Model* (2024): https://arxiv.org/abs/2403.19887
- Ren et al., *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling* (2024): https://arxiv.org/abs/2406.07522
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (MLA & MoE, 2024): https://arxiv.org/abs/2405.04434
- Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens* (2024): https://arxiv.org/abs/2412.09871
- Lee et al., *LittleBit: Ultra Low-Bit Quantization via Latent Factorization* (NeurIPS 2025, v5): https://arxiv.org/abs/2506.13771
- Lee et al., *LittleBit-2: Joint Iterative Quantization for Sub-1-Bit Latent Factorization* (ICML 2026, v2): https://arxiv.org/abs/2603.00042
- Archived BCIR model-training evidence (`gp-global`, `wedge-next`, `wedge-integrated`, `wedge-query-qware`, `wedge-pr818`, `BCIR-50M-Study`): https://github.com/Mecca-Research/Modular-AI
