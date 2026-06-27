# AI-Substrate SOTA Scan — design input for A1 / B1 / B5

> **Status: research note (non-normative).** Design input for the ML/AI-integration
> slices in `BCIR_ML_AI_INTEGRATION_ROADMAP.md`:
> **A1** (`_BitInt(N)` exact-width lanes + the Q8↔float32↔Q8 bridge under the R17 accuracy law),
> **B1** (`gem.matmul` as a claim whose tiling/loop-order/layout is chosen by K_BCIR's analytic cost search),
> and **B5** (wrap trusted C math libraries through the `c.call.libm:` seam).
> Forward-looking notes on **B3** (autodiff as graph rewriting) are included because they change A1/B1 design choices now.
>
> Produced 2026-06-27 from a multi-source web-research scan (three parallel research
> agents, one per pillar). Every load-bearing claim is cited inline. Where the SOTA
> is genuinely contested, that is **flagged**, not smoothed over. This note records
> *what the field does and where BCIR sits relative to it* — it does not change any
> normative law or count.

---

## TL;DR — the five things that change the design

1. **A1 — add per-group/block scaling under R17.** The integer + formal-error-bound bet is *defensible and differentiated* (Qualcomm shows INT8 ≥ FP8 on dedicated inference silicon; a static error *guarantee* is something the ML-quant literature does not offer). But **every** sub-4-bit SOTA format is near-lossless *only* with group/block scales (INT4-g128, OCP-MX block=32, NVFP4 block=16, GGUF super-block=256). A single per-tensor Q8 scale is competitive at 8-bit and **a gap at 4-bit**. Group scaling is mechanically compatible with a fixed-point ULP bound — close this first.

2. **A1 — `_BitInt(N)` must lower to *packed sub-byte compute lanes*, not padded power-of-two storage**, or the low-bit edge is lost. Exact-width integer lanes are a genuinely good substrate for INT3/INT5/ternary that most ISAs can only express via packing — but only if the compute path is real.

3. **B1 — the exact thesis is already published and viable.** *tritonBLAS* (AMD, 2025) does deterministic, zero-autotuning analytic GEMM tile selection at **94.7 % of exhaustive autotuning**, 100 % reproducible, µs-scale, retargeting by recalibrating hardware constants. Frame B1 as *"~90–95 % of autotuned, deterministic, tuning-farm-free,"* **not** *"beat Ansor's peak."*

4. **B1 — min,+ alone is insufficient; the roofline bottleneck is max,+.** Analytic GEMM models that hit ~95 % use a *bottleneck* (`max` over competing rooflines) latency model, not an additive one. B1 needs a **dual-semiring** cost: min,+ to sum path/setup costs, **max,+ to model the binding bottleneck** (compute vs. each memory level). **K_BCIR already carries both** (min,+ optimize + max,+ overlap) — this is an *alignment to exploit*, not a gap to build.

5. **B3 — fix the terminology and budget for the Enzyme tax.** The "operad 2-cell" framing should be **"2-categorical / string-diagram (PROP) rewriting"** — the established math (Alvarez-Picallo et al., CSL 2023) is monoidal-category rewriting, *not* operadic; claiming "operad" without separate justification is a credibility risk. And Enzyme's headline result — differentiating *optimized* IR is **~4.5× faster** than differentiating unoptimized IR — predicts a real performance tax if B3 differentiates the high-level pre-optimization claim graph. Plan multi-level AD (Enzyme-MLIR style) or differentiate-high/optimize-low (JAX style), and **measure the gap early.**

**Net:** BCIR's three bets — bounded integer quantization, deterministic analytic scheduling, and verified content-addressed AD rewriting — are each *one defensible point on a contested frontier*, not behind the curve. The wins are determinism, reproducibility, retargeting-by-recalibration, and formal guarantees the ML stack lacks. The honest costs are: the last ~5–25 % of compute-bound GPU peak (B1), the ≤4-bit/activation-outlier regime unless group-scaling + a smoothing story are added (A1), and a performance tax + higher-order/mutation coverage work for AD (B3). All three costs are *scoped and known*, which is the point of doing this scan before writing code.

---

## How this maps to the roadmap

| Slice | What it builds | What this scan tells you to do |
|---|---|---|
| **A1** | `_BitInt(N)` → exact-width quantized lanes; Q8↔f32↔Q8 bridge under R17 | Keep the integer + formal-bound bet; **add per-group scaling**; ensure packed sub-byte compute; scope Q8 to weight-only/well-conditioned, plan an activation-outlier story separately |
| **B1** | `gem.matmul` realized by K_BCIR analytic cost search, verified vs. reference C | Target ~90–95 % of autotuned + 100 % reproducible; use **dual-semiring (min,+ & max,+)**; emit an MLIR transform-dialect schedule; deliberately *do not* build a measured autotuner |
| **B3** | gradients as content-addressed reversible graph rewrites over the claim graph | Cite Alvarez-Picallo et al. (CSL 2023) as direct antecedent; **rename "operad"→"2-categorical/PROP rewriting"**; plan for the Enzyme post-optimization tax; budget for higher-order/mutation coverage |
| **B5** | wrap BLAS/LAPACK/FFTW/GSL/SLEEF via `c.call.libm:` | Discipline is SOTA-endorsed (win on the calling side); the R17-certified Q8↔f32↔Q8 bridge is the differentiator at the FFI seam |

The dual-rail discipline (Python oracle → MLIR/C++ law rail, Clang-equivalence-gated) is the right risk management for all of these — the categorical/analytic literature gives *semantics and bounds*, not a fast implementation.

---

## Pillar 1 — Quantization

Neural-network quantization replaces high-precision (FP32/FP16/BF16) tensors with low-bit representations to cut memory traffic, footprint, and (where hardware supports low-bit arithmetic) compute.

A note on confidence: the *mechanisms* below are well-established and primary-sourced. The *comparative* claims (FP8 vs INT8, FP4 vs INT4, "which method is best") are genuinely contested and depend heavily on bit-width, weight-only vs weight+activation, model family, and granularity. Those are flagged inline.

### 1. PTQ vs QAT

**Post-training quantization (PTQ)** quantizes an already-trained model using only a small *calibration* set (typically 128–512 unlabeled sequences) and no gradient updates — minutes to a few GPU-hours even at 175B parameters. **Quantization-aware training (QAT)** simulates quantization (fake-quant ops, straight-through estimator) during training/fine-tuning so the weights adapt to the rounding.

For ≥8-bit, and for 4-bit weight-only on large models, PTQ alone often retains >99 % of baseline accuracy, so QAT is usually not worth its cost. QAT's advantage appears in the **low-bit regime** (≤4-bit weight+activation, or ≤3-bit weight-only), and especially on *reasoning* tasks, where PTQ shows severe degradation (arXiv:2409.11055). Vendors now push **quantization-aware distillation (QAD)** to recover NVFP4 accuracy (NVIDIA, arXiv:2601.20088). Error criterion: PTQ minimizes a *local* layer-wise reconstruction error on calibration data; QAT minimizes the *global* task loss under simulated quantization.

### 2. Weight-only vs weight+activation; INT8 and INT4

**Weight-only** (e.g. W4A16) shrinks the dominant memory cost and accelerates the *memory-bound* autoregressive decode stage (AWQ, arXiv:2306.00978). It does **not** speed up compute-bound prefill/large-batch GEMMs. **Weight+activation** (e.g. W8A8) lets the *matmul itself* run in low precision (INT8/FP8 tensor cores) — harder because activations carry outliers (SmoothQuant, arXiv:2211.10438).

- **INT8**: essentially solved for inference (W8A8 with outlier handling reaches FP16 accuracy; INT8 weight-only is trivially lossless).
- **INT4**: the contested frontier. INT4 *weight-only* is near-lossless with GPTQ/AWQ **plus per-group scaling**; INT4 *activations* (W4A4) remain hard and usually need rotation/smoothing (granularity survey, arXiv:2507.17417).

### 3. Key LLM PTQ methods (what error each minimizes)

- **LLM.int8()** (Dettmers et al., NeurIPS 2022, arXiv:2208.07339): W8A8 via vector-wise absmax + **mixed-precision decomposition** — the ~0.1 % emergent-outlier feature dims (appearing near the 6.7B "phase shift") run in FP16, the rest in INT8. The mixed path is *not* hardware-friendly and is often slower than FP16 though memory drops ~2×.
- **SmoothQuant** (Xiao et al., arXiv:2211.10438): migrate activation-outlier difficulty into weights via a per-channel smoothing factor `s_j = max(|X_j|)^α / max(|W_j|)^{1-α}` folded into the prior layer → *fully-INT8* W8A8 (hardware-friendly). ~512 calibration sentences; `α≈0.5`.
- **GPTQ** (Frantar et al., ICLR 2023, arXiv:2210.17323): weight-only 3–4 bit minimizing layer-wise output error `‖WX − ŴX‖²` with **inverse-Hessian** second-order info; ~128 calibration segments; can **overfit the calibration set**.
- **AWQ** (Lin et al., MLSys 2024, arXiv:2306.00978): weight-only, **no Hessian** — protect the ~0.1–1 % salient channels (salience set by *activation* magnitude) via per-channel scaling found by small grid search; ~10× more data-efficient than GPTQ. *"AWQ > GPTQ" is a weak, contested ordering* (arXiv:2409.11055).

### 4. Granularity: per-tensor vs per-channel vs per-group/block

Finer granularity localizes dynamic range → lower error, at the cost of more scale storage. **Per-group/block (groups of 32/64/128) is the de-facto requirement for INT4 and below** — reducing intra-group variance is what makes 4-bit weight-only near-lossless ("INT4-g128"), and block scaling is the foundation of MX/NVFP4 (arXiv:2510.16805). Consensus is strong here.

### 5. Microscaling (OCP MX) and FP8

**FP8** (E4M3 = more precision/less range, weights+activations; E5M2 = more range, gradients): native tensor-core support since **Hopper (H100)**, continuing on **Blackwell**, ~2× FP16 throughput. FP8's exponent gives dynamic range that *naturally absorbs outliers* — its main edge over INT8 for activations.

**OCP Microscaling (MX)** (OCP MX v1.0; arXiv:2310.10537) — block floating point: a block of **k=32** shares one **E8M0** power-of-two scale, with low-bit payloads: MXFP8 (E4M3/E5M2), MXFP6 (E3M2/E2M3), MXFP4 (E2M1), MXINT8. **Blackwell** accelerates MXFP4/6/8 with on-the-fly dequant; MXFP4 ≈ 2× the throughput of MXFP8 (arXiv:2512.02189). **NVFP4** refines MXFP4 with **two-level scaling**: E2M1 elements in **blocks of 16** with an FP8-E4M3 per-block scale + a per-tensor FP32 scale → finer adaptation, near-FP8 accuracy at ~1.8× smaller footprint (NVIDIA; arXiv:2509.25149).

**Where float/MX beats INT — and where it's contested:** for **weight+activation**, low-bit *float* tends to beat INT at equal bit-width (the exponent handles outliers; FP > INT for W4A4, arXiv:2305.12356). For **weight-only**, INT4 and NVFP4 are *similar*, and INT can win at ≤3-bit/small group. **Contested flag:** Qualcomm (arXiv:2303.17951) argues the *opposite* for dedicated inference hardware — an FP8-E4 MAC is ~53–183 % larger in gate count than INT8, and after QAT INT8 matches or beats FP8 on well-behaved networks. So *"FP8 > INT8" is format-and-context-dependent* — the single most important nuance for a hardware-cost-governed compiler.

### 6. Block floating point, GGUF k-quants, NF4/double-quant

- **GGUF k-quants (llama.cpp)** — the dominant CPU/edge format: **super-blocks of 256** = 8×32 sub-blocks, hierarchical FP16 super-block scale + 6-bit per-sub-block scales *and* mins (asymmetric). `Q4_K_M` ≈ 4.5 bits/weight, ~75 % smaller, minimal quality loss. **No formal error bound — empirically tuned** (arXiv:2601.14277).
- **QLoRA: NF4 + double quantization** (Dettmers et al., NeurIPS 2023, arXiv:2305.14314): **NF4** is a *quantile* type information-theoretically optimal for zero-mean Gaussian weights; **double quantization** quantizes the scales themselves (saving ~0.37–0.5 bits/param). Criterion: distributional optimality under a Gaussian prior, not a per-tensor error bound.

### Design implications for BCIR (slice A1)

**Aligned with SOTA (genuine fit):**
- **Integer fixed-point is a first-class, hardware-efficient choice.** Qualcomm (arXiv:2303.17951) explicitly argues the integer paradigm is the better accuracy/silicon trade-off for dedicated inference (integer MACs 1.5–2.8× cheaper in gates than FP8). For a *cost-governed* IR this is defensible — **EDGE** for cost-modeled targets.
- **Exact-width `_BitInt(N)` lanes generalize per-group integer quant.** Native arbitrary-width integer lanes cleanly express INT3/INT5/ternary that GPTQ/k-quants exploit but most ISAs express only via packing. **EDGE** *if* BCIR emits packed sub-byte GEMM lanes; **GAP** if `_BitInt(N)` only legalizes to padded power-of-two storage with no low-bit compute path.
- **A formal per-claim error bound is rare and valuable.** Essentially every SOTA method certifies quality *empirically*; R17's static Q8-ULP bound (and the count-ULP-naive vs 1-ULP-compensated reduction rule) is closest to classical fixed-point DSP error analysis, which the ML-quant literature abandoned. **EDGE**, provided the bound is *tight enough to be useful*.

**Divergent (honest gaps / scoping risks):**
- **SOTA trends to low-bit *float*/*microscaling*, not pure fixed-point Q8.** Blackwell throughput leadership is MXFP4/NVFP4/FP8 with two-level block scales. A single global Q8 scale is coarser and lacks the exponent that absorbs activation outliers — exactly what SmoothQuant/LLM.int8() worked *around*. **GAP** for transformer activation paths without an outlier/smoothing story; **not a gap** for well-conditioned/DSP-style or weight-only work.
- **Granularity.** A per-claim (≈per-tensor) Q8 scale is competitive at 8-bit but **not at 4-bit**. Adding **per-group scaling under R17** is the highest-leverage extension and is mechanically compatible with a ULP bound. **Closeable GAP.**
- **FP32 as the bridge pivot** is sound for *certification* (FP32 is de-facto ground truth) and fits the "wrap trusted C math" discipline; becomes a **GAP** only if FP32 is also the runtime compute path on throughput-critical kernels. **Neutral-to-EDGE.**

**Net:** aligned with the integer-quantization camp and the universal reliance on group-scaled low-bit integers, *differentiated* by a static error *guarantee*; divergence from the float/MX mainstream is real but **scoped** — EDGE for cost-governed/well-conditioned/weight-centric/verification-critical work, GAP for ≤4-bit and transformer-activation regimes until (a) per-group scaling and (b) an outlier/smoothing path are added under R17.

**Low-confidence/contested:** (1) FP8>INT8 and FP4>INT4 are *not* settled — don't design against either as law; (2) AWQ vs GPTQ is a weak ordering; (3) MXFP4 vs NVFP4 vs INT4 for *weight-only* is still actively benchmarked; (4) vendor accuracy headlines (NVFP4) are optimistic — treat as such.

---

## Pillar 2 — Tensor scheduling / tiling

*Choosing how a tensor op (especially `matmul`) is realized — tile sizes, loop order, layout, fusion — and what that implies for BCIR's deterministic analytic tropical cost search in B1.*

### 2.1 Polyhedral compilation (Pluto, isl)

**Pluto** (Bondhugula et al., PLDI 2008) finds tiling hyperplanes via an ILP embedding a *linear* cost function (minimize dependence distance), tiling for locality + parallelism — deterministically, no measurement; machinery from **isl**. Documented limits are exactly BCIR's concern: Pluto's objective "does not consider … data sizes, the complexity of control … and the characteristics of the hardware," and it is **size-agnostic — same schedule regardless of loop extents** (LOOPer, arXiv:2403.11522). That is the central weakness of purely-analytic affine models: they capture reuse *structure* but not the quantitative interaction with a specific cache hierarchy or shape.

### 2.2 Learned cost models (TVM/Ansor, Halide)

- **Ansor** (Zheng et al., OSDI 2020, arXiv:2006.06762): samples complete programs, tunes with evolutionary search + a learned GBDT cost model retrained on **on-device measurements**; up to 3.8×/2.6×/1.7× over prior SOTA — but needs ~1,000 trials/op and **thousands–tens of thousands of measurements per network**, hours of search.
- **Halide** brackets the spectrum: **Mullapudi 2016** is *purely analytic* (bounds analysis, no autotuning); **Adams 2019** uses a *learned* cost model + beam search, ~2× faster *without* autotuning and >2× *with* — the "with/without" gap is the key data point: a good model gets most of the way, measurement adds an increment.
- **LOOPer** (arXiv:2403.11522) puts a learned model on polyhedral transforms and beats Pluto's linear cost by **1.42× geomean** — evidence the *cost model*, not the transform space, was Pluto's bottleneck.

### 2.3 Triton, CUTLASS hierarchical tiling

**Triton** performance hinges on tunable BLOCK_SIZE/num_warps/num_stages selected by `@triton.autotune`, which **exhaustively compiles + benchmarks candidates at runtime**. **CUTLASS** encodes the GEMM hierarchy as templates — **threadblock → warp → thread tiles** with software pipelining — expressed via **CuTe** layout algebra. This hierarchical block/warp/thread decomposition *is* the realization space B1's `gem.matmul` must navigate.

### 2.4 MLIR linalg + transform dialect

MLIR **linalg** is the de-facto tensor-codegen testbed (tile-the-consumer/fuse-the-producer, lower to vector/affine). **Key architectural point:** the **transform dialect is a *mechanism* for applying a schedule, not a cost model** — *what* schedule is chosen by an external driver (heuristic/hand-written/autotuner). PEAK (LCPC 2023) and IREE drive transform-dialect schedules with reuse analysis *plus* an autotuner. **This is the natural integration point for a BCIR-style analytic planner: produce a transform-dialect schedule, lower through linalg/vector, optionally refine by measurement.**

### 2.5 Roofline: when analytic predicts right, and when it fails

The **roofline model** (Williams et al., CACM 2009) bounds `P = min(P_peak, I × b_S)`. The sharpest evidence on analytic success/failure is Ernst et al.'s "Warpspeed" (arXiv:2204.14242): it **reliably separates good from bad configs** (picked a 96 %-of-peak config for a 3D stencil) but **cannot resolve the top of the ranking** (its top pick was only the 12th-fastest measured), and **degrades on complex kernels** (only identifies the *worst* configs for a two-phase LBM kernel). Generalizable: **analytic cost models are strong in the memory-bound/roofline-clear regime and for coarse class selection, weak at fine selection among good configs and on kernels with latency/occupancy/pipeline effects the model omits** (Ansor §2 concurs).

### 2.6 The contested analytic-vs-learned debate (2025)

- **Pro-analytic — tritonBLAS** (AMD, arXiv:2512.04226, Dec 2025): a **deterministic, zero-autotuning** analytic model selects Triton GEMM tile hierarchies from calibrated hardware constants. Across 150k GEMM shapes on MI300X: **94.7 % of exhaustive autotuning, zero tuning time**; selection in **50–80 µs vs. 10–50 s** (5–6 orders faster, shape-independent); on real Llama-3 shapes ~**13.9 % slower** than autotuned vendor `torch.matmul` (occasionally faster); **retargets MI300X→MI350X by changing only the constants.** Essentially a published instance of the B1 thesis.
- **Pro-autotuning — Ringlein et al.** (IBM, arXiv:2505.03780, 2025): on attention/RMSNorm kernels, autotuned Triton is up to **2.3×** over vendor libs, and **reusing one GPU's best config on another drops to as little as 7 %**. For *complex* kernels the best config is hardware-specific and not recoverable by a static heuristic.

> **Contested flags.** Headline numbers are workload/hardware-specific single-paper results — directional, not constants. tritonBLAS is **GEMM-only on AMD CDNA**, excludes attention/fused kernels; Ringlein's pro-autotuning result is *on* attention — they partly describe **different op classes**, and that boundary is genuinely unsettled. "Analytic ≈ 95 % of autotuned" should be read *for GEMM in roofline-clear regimes*, not for fused GPU kernels.

### Design implications for BCIR (slice B1)

- **(a) Quality vs. peak.** Consistent envelope for analytic, no-measurement GEMM selection: **~90–95 % of autotuned peak** in the roofline-clear/memory-bound regime (tritonBLAS 94.7 %; Ernst 96 %-of-peak; LOOPer model-guided ≈75–92 % of oracle). The last ~5–25 % needs measured autotuning, and the gap *widens* for complex/compute-bound/fused kernels.
- **Load-bearing semiring note.** The analytic models that hit 94.7 % use a *bottleneck/roofline* latency model — `max` over competing rooflines. **A min,+ (additive shortest-path) search is the wrong shape for the bottleneck; the bottleneck is max,+.** B1 needs **both** semirings (min,+ to compose path/setup costs; max,+ for the binding bottleneck among compute vs. each memory level). **K_BCIR already has both** (min,+ optimize + max,+ overlap) — exploit this; it is the single most important alignment.
- **(b) Reproducibility/determinism — clear EDGE.** Analytic selection is bit-for-bit reproducible; autotuned schedules are per-process, re-tuned each run, rarely cached, fragile across environments (Ringlein). For a cost-governed *verified* IR, determinism is architecturally aligned, not just convenient.
- **(c) Retargetability — EDGE *in form*, GAP *in matched peak*.** Recalibrate a small constant vector per target (tritonBLAS MI300X→MI350X). BCIR's cost-vector approach inherits this — *cheap, principled retargeting, no tuning farm* — but only if the cost vector is **re-derived per target** (Ringlein's 7 % warns against reusing one config), and it still lands at the analytic ~90–95 % ceiling on hard kernels.
- **(d) Cost of not autotuning.** Avoided: a tuning farm, hours/target, per-shape recompilation. Paid: the ~5–25 % peak gap on hard kernels and *zero* ability to find the non-intuitive winner measurement reveals (Ernst's true-fastest was 12th analytically).

**Net:** an EDGE for the regime B1 targets (a `gem.matmul` chosen for layout/tiling/fusion/loop-order, verified vs. reference C, on memory-bound/roofline-clear shapes), a GAP against autotuned peak for complex compute-bound fused GPU kernels. tritonBLAS shows the *exact* B1 thesis is viable at ~95 % + 100 % reproducible. **Frame B1 as: deterministic, reproducible, tuning-farm-free schedules capturing ~90–95 % of the value with µs-scale planning — leave the last few percent of compute-bound GPU peak to a measured layer BCIR deliberately does not build** (consistent with "integrate, don't reinvent"). Two risks to flag: (1) min,+ alone is insufficient — plan dual-semiring; (2) the "~95 %" is GEMM/memory-bound evidence — do **not** assume it generalizes to fused/attention kernels.

---

## Pillar 3 — Autodiff as graph rewriting (forward-looking for B3)

The canonical reference is Baydin, Pearlmutter, Radul & Siskind, *AD in ML: a Survey* (arXiv:1502.05767, JMLR 2017). What matters for BCIR is less "how to take a derivative" and more "what *representation* you differentiate, and whether the transformation is a verifiable, replayable object."

### 1. Forward vs reverse mode

Forward propagates a tangent (one Jacobian column/pass; wins when inputs ≪ outputs); reverse propagates an adjoint backward (one row/pass; wins when outputs ≪ inputs — the ML case). Reverse mode's "cheap gradient principle" (Baur–Strassen): a full gradient costs a small constant multiple of the forward eval in *operations*, but **memory proportional to computation length** (the forward trace must be retained) — the central engineering problem, motivating checkpointing. A BCIR-relevant decomposition: **reverse = forward-mode linearization then transposition** of the linear map (Radul et al., *Decomposing Reverse-Mode AD*, arXiv:2105.09469) — exactly how JAX implements `grad`, and the most "rewrite-like" account in the mainstream literature.

### 2. Tracing AD (JAX, PyTorch)

**JAX** traces to `jaxpr` (small functional SSA IR); AD/`vmap`/JIT are *composable transformations on jaxprs*, XLA lowers (Frostig et al., MLSys 2018). Control flow must be staged through primitives — relevant because BCIR's claim graph is similarly explicit. **PyTorch autograd** is operator-overloading with a dynamic tape — a content graph of the executed trace, but **ephemeral, untyped-by-law, recreated each call**, with no verification or cross-run dedup. BCIR wants the tape to be a *first-class, content-addressed, verified artifact*.

### 3. Source-transformation AD

**Tapenade** (Hascoët & Pascual, ACM TOMS 2013): mature source-to-source for Fortran/C with activity / to-be-recorded analyses BCIR will need analogues of. **Zygote** (Innes, arXiv:1810.07951): reverse-mode source-to-source on Julia's **SSA-form IR** — the key move, giving principled control flow / recursion / mutation / higher-order handling without unrolling. The SSA-rewrite framing is the closest mainstream precedent for "AD as a transformation over a structured program graph." (*Diffractor* = under-documented successor — **low confidence**.)

### 4. IR-level AD: Enzyme (the most important data point)

**Enzyme** (Moses & Churavy, NeurIPS 2020, arXiv:2010.01709) synthesizes gradients of statically-analyzable **LLVM IR** (so it differentiates anything lowering to LLVM). Pipeline: **type analysis** (recover element types behind untyped pointers), **activity analysis**, **synthesis** (per-instruction adjoints + a cache/tape with use-analysis). **Headline result, load-bearing for BCIR:** AD *after* optimization is a **geomean 4.5× faster** than AD before optimization (e.g. LICM hoists an `O(N)` call out of a loop so its adjoint is also hoisted, `O(N)` vs `O(N²)` gradient). **Limitations:** needs IR for everything (no runtime-loaded code), statically-deducible types (TBAA on, no unions of differing types), no exception adjoints at publication. An **Enzyme-MLIR** path runs AD at multiple abstraction levels via differentiation interfaces — the single most architecturally-relevant comparator for BCIR's MLIR law rail (capability claims **medium-confidence** — thinner peer-reviewed coverage).

### 5. Checkpointing / rematerialization

Reverse-mode memory grows with computation length → trade recompute for storage. Foundational: **revolve** (Griewank & Walther, ACM TOMS 2000) — provably optimal log-depth checkpoint schedules. In DL: **Checkmate** (MILP, arXiv:1910.02653), **Dynamic Tensor Rematerialization** (online/greedy, arXiv:2006.09616). Enzyme recomputes by default, caching only when forced. **BCIR-relevant:** in a content-addressed graph, "recompute vs cache" *is* "re-derive a claim from its content hash vs materialize it" — checkpointing becomes a *replay/dedup* decision, a genuine conceptual fit (and a place BCIR could contribute) — though **no prior art** does checkpointing over a content-addressed claim graph specifically.

### 6. The categorical / structured view (the evidence base for the framing)

- **Supported:** Conal Elliott, *The Simple Essence of AD* (ICFP 2018, arXiv:1804.00746) — AD is a *structure-preserving functor* (a 1-functor, not 2-cells/operads). Lens/optics AD: *Reverse Derivative Categories* (arXiv:1910.07065) + *Categorical Foundations of Gradient-Based Learning* (arXiv:2103.01931) + *Deep Learning with Parametric Lenses* (arXiv:2404.00408) give a principled "forward claim ↔ backward claim" pairing — but reverse-derivative categories **do not natively support higher-order functions**.
- **Closest real prior art:** Alvarez-Picallo, Ghica, Sprunger & Zanasi, *Functorial String Diagrams for Reverse-Mode AD* (arXiv:2107.13433, CSL 2023) — expresses reverse-mode AD as **local rewrite rules on string diagrams**, implements it as **double-pushout (DPO) hypergraph rewriting**, and **proves soundness** against reverse-derivative-category semantics. Crucially: the rewrite system has the **diamond/confluence property** (so differentiation order doesn't change the answer), and the hypergraph representation makes equivalent diagrams **share one canonical form** — *structural sharing/dedup falls out of the representation*, exactly BCIR's content-addressing argument, arrived at independently.

**Honest assessment of the "operad/2-cell" framing.** *Supported:* "AD as a functor / confluent local graph rewrites with subgraph sharing / lens." *Speculative / not found in prior art:* **no paper models AD as 2-cells of an *operad***, nor requires *reversibility* in BCIR's sense, nor frames AD rewrites as morphisms of a *verified content-addressed claim graph with provenance*. The existing rewriting work is naturally (2-)categorical, so "2-cell" language is defensible, but **the specific *operadic* packaging is BCIR's own synthesis** — the established structure is symmetric-monoidal / cartesian-closed **PROP / category** rewriting, *not* operads. **Soften "operad"→"(monoidal) 2-categorical / string-diagram rewriting," or justify "operad" independently.**

### Design implications for BCIR (slice B3)

- **Vs Enzyme:** the 4.5× (AD-after-optimization) result is a direct challenge. If B3 records gradient rewrites over the high-level *pre-optimization* claim graph, it risks the exact `O(N)`→`O(N²)` pathology Enzyme exists to avoid. Mitigations with precedent: (i) **multi-level AD** (Enzyme-MLIR style — also fire on lowered/optimized claims); or (ii) **differentiate-high / optimize-low** (JAX/XLA — rely on rewrites preserving enough algebra for late optimization). **Measure this gap early — it is the project's biggest AD performance risk.**
- **Vs categorical/lens AD:** BCIR is a *natural home* for these ideas. Genuine wins from verification + content-addressing + replay: **traceability & deterministic replay** of each diff step (no mainstream system offers this); **correctness-by-construction** (the soundness-via-rewrite-rules result + the confluence/diamond property formally back "differentiation order doesn't change the answer"); **global dedup** of shared subgraphs (stronger than Enzyme's local cache-reuse); **replay-as-checkpointing** (principled, provenance-aware rematerialization).
- **Risks:** the Enzyme performance tax (above); **coverage of higher-order functions** (reverse-derivative categories don't support them natively), **mutation, and control flow** — exactly where the clean categorical story strains and Enzyme/Zygote do real, un-glamorous work; and the engineering surface (a verified IR + content store + from-scratch AD engine vs. Enzyme reusing all of LLVM). The dual-rail discipline is sensible risk management.

**Bottom line:** *"AD as content-addressed, confluent local graph rewrites with structural sharing, given a sound rewrite-rule semantics against reverse-derivative-category laws"* is **well-supported** (Elliott 1804.00746; Cockett et al. 1910.07065; Alvarez-Picallo et al. 2107.13433 — cite as direct antecedents, reuse their soundness/confluence arguments). *"AD as reversible **operad** 2-cells with a 21-law verifier, replay, and provenance"* is **BCIR's own bet** — the 2-categorical-rewriting reading is defensible, but the operadic packaging, reversibility requirement, and verified-provenance layer are **contributions to validate, not settled results.**

---

## Contested / low-confidence register

- **FP8 > INT8 / FP4 > INT4** — *not settled*; format-, granularity-, and workload-dependent (Qualcomm arXiv:2303.17951 argues INT wins on dedicated silicon). Do not design against either as law.
- **AWQ > GPTQ** — weak empirical ordering, small and model-dependent margins.
- **MXFP4 vs NVFP4 vs INT4 (weight-only)** — actively benchmarked (2025–26 papers diverge); vendor accuracy headlines optimistic.
- **"Analytic ≈ 95 % of autotuned"** — well-supported *for GEMM in roofline-clear regimes* (tritonBLAS); **not** demonstrated for fused/attention kernels, where Ringlein (arXiv:2505.03780) says measured autotuning still dominates. The op-class boundary is genuinely unsettled.
- **min,+ vs max,+ for B1** — that the bottleneck is max,+ is med-high confidence (inferred from tritonBLAS's latency formulation); the exact dual-semiring formulation for BCIR is unverified and should be prototyped in the oracle first.
- **"operad" as the correct structure for AD rewrites** — *low confidence*; the literature is PROP/monoidal/2-category, not operad. Diffractor specifics and detailed Enzyme-MLIR capabilities — low/medium confidence (thin peer-reviewed coverage).

## Key sources

**Quantization:** LLM.int8() (2208.07339), SmoothQuant (2211.10438), GPTQ (2210.17323), AWQ (2306.00978), OCP MX v1.0 + Microscaling (2310.10537), NVFP4 pretraining (2509.25149), QLoRA/NF4 (2305.14314), FP8-vs-INT8 on dedicated HW (2303.17951), granularity survey (2507.17417), GGUF k-quant eval (2601.14277).
**Scheduling:** Pluto (PLDI 2008), Ansor (2006.06762), Halide-learned (Adams 2019), Mullapudi (TOG 2016), LOOPer (2403.11522), MLIR transform dialect (2409.03864), roofline (CACM 2009), Warpspeed (2204.14242), tritonBLAS (2512.04226), "GPU Performance Portability needs Autotuning" (2505.03780), CUTLASS/CuTe docs.
**Autodiff:** AD survey (1502.05767), Decomposing Reverse-Mode AD (2105.09469), JAX tracing (MLSys 2018), Zygote (1810.07951), Tapenade (TOMS 2013), Enzyme (2010.01709), revolve (TOMS 2000), Checkmate (1910.02653), DTR (2006.09616), Simple Essence of AD (1804.00746), Reverse Derivative Categories (1910.07065), Parametric Lenses (2404.00408), Functorial String Diagrams for Reverse-Mode AD (2107.13433).
