# BCIR full global head with query-aware geometric contraction

Research implementation and controlled CPU study · 18 October 2026

**The full head and its query-aware contraction are implemented in C and tested. The contraction does not establish useful associative learning at this training budget.** The retained full wedge route is augmented with an explicit query–plane interior product. Both the original-size and nearly parameter-matched versions remain near the 16-label chance level on random key/value association; attention learns single-pair value retrieval but also falls short of reliable multi-pair binding.

The result is task-dependent. This experiment tests one precise contraction, one small model scale and a declared data/optimization budget.

**Long ordered recall does improve:** at length 2048, the full-size contraction rises to **89.06%**, versus **69.27%** baseline; the matched version reaches **85.42%**. At length 1024, the matched version reaches 100% on all three selected models. This supports an ordered-recall candidate, not a solved associative-memory architecture.

The frozen study completed **51 from-scratch training runs, 41,400 quality-training updates and 51 retained selected checkpoints**. Three independently trained seeds are 20261018, 20261019 and 20261020. All checkpoints restore with their recorded model/optimizer digests, and all paired input/source checks pass. Each task trains separate weights; this is not one jointly trained model.

## 1. What was built

For the existing full rank 22 global branch, let

\[
u_t=z_t,\qquad v_t=\frac{\sum_{j<t}z_j}{\sqrt{\max(t,1)}},\qquad
p_{ij}=\frac{u_i v_j-u_j v_i}{\|u\wedge v\|},\ i<j.
\]

The full 231-coordinate dense projection remains. With the numerical safeguards below, the new channel is

\[
q=\frac{u}{\max(\|u\|,10^{-6})},\qquad
P_{ij}=p_{ij},\quad P_{ji}=-p_{ij},\qquad
c_j=\sum_i q_iP_{ij},\qquad
g=W_p p+W_c c.
\]

The sign is **qᵀP**, not Pq. Away from clamping/quantization, this yields the normalized component of v orthogonal to q. It does not restore the parallel component, prefix magnitude or associations that were never encoded in the prefix representation. The current query uses the existing learned reduced frame; this experiment does not test an independent query projector or a key/value matrix memory.

`query_core.h` implements the packed contraction, its direct query VJP, its packed-plane VJP and native pool dispatch without allocation. Backpropagation includes both paths: through p and through q. The VJP adds

\[
\bar p_{ij}\mathrel{+}=q_i\bar c_j-q_j\bar c_i,
\qquad \bar q=P\bar c,
\qquad \bar u\mathrel{+}=\frac{\bar q-\hat u(\hat u^\mathsf T\bar q)}{\|u\|}
\]

above the query floor; below it the denominator is constant. The native backward reconstructs a row's packed feature locally when necessary, so compact execution may reuse the former packed-feature arena. It then combines the geometry adjoints before the existing reverse prefix scan.

`build_query.py` produces compact cached, frame/FF reconstruction and combined layer/frame/FF reconstruction variants. Native plans count the new cached 22-vector per row/layer and contraction-gradient scratch. `policy.py` retains one persistent model/optimizer state and reusable workspace, with real resizing and budget admission. All linear, tensor, backward and AdamW arithmetic runs in C. Python generates records, binds plans and records experiments. PyTorch supplies an independent AD reference, not the training-throughput engine.

| Model | FF width | Parameters | Change from baseline |
| --- | --- | --- | --- |
| Full global baseline | 257 | 525,696 | +0.000% |
| Full global + contraction, FF257 | 257 | 534,144 | +1.607% |
| Matched + contraction, FF250 | 250 | 526,080 | +0.073% |
| Native attention32 | 256 | 525,696 | +0.000% |

Width 128, depth 3, vocabulary 260, tied embeddings, RMSNorm and SwiGLU are shared. Global arms retain rank 22 and no RoPE. Attention has 4 heads and RoPE. Per-parameter name/seed initialization makes shared tensors identical; Wc is initialized at standard deviation 1.0, the same unit-feature scale as Wp. No candidate-specific hyperparameter search was performed.

The full-size addition tests the new route with unchanged FF257. The FF250 variant adds only 384 total parameters relative to the 525696-parameter baseline, so a gain cannot simply be credited to an extra8448 parameters. FF reallocation itself remains a trade-off, rather than an exactly identical architecture.

## 2. Numerical and implementation gates

A repeated-prefix check exposed a pre-existing sensitivity in the original area-clamp convention. Identical leading tokens create an absent plane. FP32 forward outputs can agree closely while very large intermediate derivatives cancel differently through shared weights. In the recorded two-layer baseline example, the C gradient norm is about 344205 and its relative disagreement with Torch is 1.26. The query head exhibits the same inherited issue. This is a numerical/derivative-conditioning finding, not a diagnosis that the new contraction VJP is wrong.

The already available precision safeguard is therefore common to **both** baseline and contraction study arms: when the frame Gram determinant is at most 1e-4 of the squared-norm product, pair arithmetic/VJP use FP64. A precise area at most 1e-6 yields zero feature and zero pair VJP. The query has its separate 1e-6 norm floor. Ordinary feature/gradient buffers remain FP32. The safeguarded duplicate-prefix baseline's gradient norm is 2.81 and C/Torch relative error is 4.45e-7.

This explicitly changes the absent-plane numerical convention; it is not a proof of smoothness at the cutoff, a remedy for FP32 input quantization or a universal conditioning guarantee. Guarded near-collinear and tiny-query cases are tested independently. The common guard keeps the association comparison focused on the contraction. All six old ordered/multi-pair baseline checkpoints reproduce the previous study's selected weights and moments bitwise; the language baselines are newly trained under the common guarded convention and are not bitwise continuations of the old unguarded ones.

Validation passes include:

- Independent full-model forward/backward at width 32 and width 128, including FF257/250 and duplicate leading tokens, across all three query layouts. Maximum gradient relative L2 error: **7.33e-07**; maximum logit error: **6.26e-07**.
- Packed-contraction AD checks with ordinary, tiny, large and zero queries; full prefix-geometry AD and directional finite differences. Guarded geometry maximum gradient relative error: **1.33e-07**.
- Exact causal-prefix output equality when future tokens change; bitwise state equality across shape/layout transitions and one/four threads; exact checkpoint replay.
- Exact-capacity buffer red zones and focused UBSan for compact and combined reconstruction; independent baseline, hybrid and attention controls.
- Microbatch accumulation normalized according to the native SUM-of-means contract: gradient relative difference **8.54e-08**, first updated-weight difference **4.28e-09**.
- Query-head budget changes 96→32→20→64 MiB at B4/T1024, fixed microbatch 1, preserve persistent storage and produce bitwise-identical optimizer state to the ample cached reference. The20 MiB plan actually uses about 18.97 MiB; 8 MiB is refused before allocation.

## 3. Frozen learning protocol

| Task | Updates/run | Data |
| --- | --- | --- |
| Single-pair value retrieval | 900 | B4; 1 random key/value triple; balanced lengths 64/128/256 |
| Multi-pair association | 900 | B4; 2 pairs for 300 updates, then 4; balanced64/128/256 |
| Ordered first-value recall | 900 | B4; targets and distractors share tokens0–15; balanced64/128/256 |
| Raw WikiText-2 byte modeling | 600 | Lengths 32/64/128; batches 8/4/2; exactly 256 supervised bytes/update |

Synthetic associations have unique random keys 32–47 and values 0–15, explicit triples `(258, key, value)`, filler 64–95, and a final query marker/key. Validation always tests the declared task: 1 pair for the single-pair arm; 4 pairs for the curriculum arm. Data are paired across model candidates. Same optimizer/rates are used: AdamW, peak LR1e-3,30-update warmup, cosine floor 1e-4, betas 0.9/0.95, epsilon 1e-8, decay 0.1, norm clip 1.0. The lowest aggregate validation CE at 100-update intervals selects a checkpoint. Final tests never select checkpoints or revise the recipe.

Every training arm has the same 32 MiB **controlled-buffer** envelope. Global/query/hybrid retain the full effective microbatch; attention uses the largest fitting microbatch and accumulation, including micro 2×2 at T256. This is the stronger matched-storage comparator. It is the retained native probability-materializing attention implementation, not a claim about every optimized attention implementation.

The bound counts four FP32 persistent arrays, activation/scratch workspace, staging, pool control, alignment slack and named C records. It excludes code, thread stacks, Python metadata, data, checkpoint/result copies and allocator retention. It is not process RSS or an OS memory cap. Fresh test inference has a common separate 512 MiB envelope, and performs no updates.

Primary association evaluation combines T128 and T256: 256 examples/length per selected seed checkpoint,512 per seed. Larger-length,1/2/4/8-pair and counterfactual results are secondary. Samples are shared across candidates and training seeds. Tables give the mean of three trained models; seed ranges are descriptive, not confidence intervals or independent pooled-example replication.

## 4. Association result: the new readout does not solve binding

| Model | Single-pair accuracy | Four-pair accuracy | Four-pair CE |
| --- | --- | --- | --- |
| Full global baseline | 5.34% | 7.49% | 2.78050 |
| Full global + contraction, FF257 | 5.53% | 7.29% | 2.78217 |
| Matched + contraction, FF250 | 5.53% | 7.23% | 2.78114 |
| Native attention32 | 100.00% | 23.50% | 2.21274 |

Unconditional guessing over 16 labels averages **6.25%**. An oracle that knows the 4 candidate values but guesses their assignment averages25%; that is a different reference. Attention’s single-pair success verifies that the objective/data can support value retrieval at this budget, while its four-pair result still does not establish reliable binding.

| Contraction comparison | Accuracy change vs baseline | Three-seed changes |
| --- | --- | --- |
| kv_single: Full global + contraction, FF257 | +0.195 points | +0.000, +0.000, +0.586 |
| kv_single: Matched + contraction, FF250 | +0.195 points | +0.000, +0.000, +0.586 |
| kv_curriculum: Full global + contraction, FF257 | -0.195 points | +0.391, -0.391, -0.586 |
| kv_curriculum: Matched + contraction, FF250 | -0.260 points | +0.000, -0.195, -0.586 |

The relevant finding is absence of a useful retrieval gain, not whether a small finite-sample fluctuation is exactly zero. Neither additional parameters nor near-matched FF reallocation produces reliable association learning here.

| Four-pair test | Model | Accuracy | Prediction is one of the 4 values |
| --- | --- | --- | --- |
| T128 | Full global baseline | 7.68% | 26.82% |
| T128 | Full global + contraction, FF257 | 7.68% | 27.60% |
| T128 | Matched + contraction, FF250 | 7.68% | 26.56% |
| T128 | Native attention32 | 20.83% | 98.70% |
| T256 | Full global baseline | 7.29% | 24.61% |
| T256 | Full global + contraction, FF257 | 6.90% | 25.78% |
| T256 | Matched + contraction, FF250 | 6.77% | 25.52% |
| T256 | Native attention32 | 26.17% | 98.96% |
| T1024 | Full global baseline | 6.25% | 29.17% |
| T1024 | Full global + contraction, FF257 | 6.77% | 29.17% |
| T1024 | Matched + contraction, FF250 | 6.25% | 29.17% |
| T1024 | Native attention32 | 21.35% | 86.98% |

The candidate-set rate distinguishes access to any plausible value from retrieval of the queried assignment. Longer contexts and 8-pair extrapolations are retained in `results/summary.json` and per-case predictions, including weak results.

| Counterfactual at T128 | Model | Both versions correct | Prediction changes |
| --- | --- | --- | --- |
| value_swap | Full global baseline | 0.00% | 0.00% |
| value_swap | Full global + contraction, FF257 | 0.00% | 0.00% |
| value_swap | Matched + contraction, FF250 | 0.00% | 0.00% |
| value_swap | Native attention32 | 5.08% | 34.64% |
| query_change | Full global baseline | 0.13% | 19.01% |
| query_change | Full global + contraction, FF257 | 0.13% | 31.64% |
| query_change | Matched + contraction, FF250 | 0.13% | 15.36% |
| query_change | Native attention32 | 1.43% | 21.74% |

Value swaps preserve the complete token multiset and query key while changing the correct value. Query changes leave the stored triples unchanged and ask for another present key. Both-correct rates require appropriate responses to both versions. The analogous T512 tests are retained. These probes prevent a favorable average candidate-value score from being mistaken for assignment learning.

In the global-family four-pair T128 swap tests, predictions never change, so no example pair has both answers correct. The full-size contraction does increase response sensitivity to query-key changes (31.64% prediction changes, versus19.01% baseline), but both-correct accuracy remains0.13%. Query sensitivity alone is insufficient: the model still fails to map the new key to its value. These are observed argmax behaviors on the finite test, not proofs that every internal state or logit is invariant.

The result is consistent with an unresolved learned write/binding bottleneck. The contraction reads the same compressed plane and current frame. If earlier layers have not encoded which value belongs to which key before aggregation, the readout cannot recover that missing assignment. This is an architectural hypothesis supported by the failure pattern, not a representation-impossibility proof; more training, better optimization or a different learned feature construction could change it.

## 5. Ordered recall and language controls

| Ordered recall | T128 | T256 | T512 | T1024 | T2048 |
| --- | --- | --- | --- | --- | --- |
| Full global baseline | 100.00% | 100.00% | 99.61% | 95.31% | 69.27% |
| Full global + contraction, FF257 | 100.00% | 100.00% | 99.87% | 99.48% | 89.06% |
| Matched + contraction, FF250 | 100.00% | 100.00% | 99.87% | 100.00% | 85.42% |
| Native attention32 | 100.00% | 100.00% | 100.00% | 100.00% | 93.23% |

Ordered recall uses a genuinely order-sensitive shared target/distractor vocabulary. The tables report fresh data: 256 examples/seed at T128/256/512,128 at T1024 and 64 at T2048. Training ends at T256, so longer lengths test extrapolation. This control is distinct from keyed association and cannot substitute for it.

The full-size contraction gains **19.79 percentage points** at T2048; the matched version gains **16.15 points**. The full-size variant's three-seed range is 87.50–90.63%; the matched variant spans68.75–95.31%, so its longer-context stability is less convincing even though it wins T1024. Attention32 remains ahead in mean T2048 accuracy. These small long-length samples need replication.

| Model | First checked 99% validation step | Training time to crossing | T512 swap both-correct |
| --- | --- | --- | --- |
| Full global baseline | 700.0 | 11.325 s | 98.96% |
| Full global + contraction, FF257 | 500.0 | 8.485 s | 99.87% |
| Matched + contraction, FF250 | 500.0 | 8.736 s | 99.87% |
| Native attention32 | 266.7 | 4.219 s | 100.00% |

Crossings are observed every 100 updates and are not a sustained-accuracy certificate. This is time to a task-specific validation threshold; it must be considered alongside total update time and held-out extrapolation.

| Language model | CE, nats/byte | T512 CE |
| --- | --- | --- |
| Full global baseline | 2.45864 | 2.48507 |
| Full global + contraction, FF257 | 2.45671 | 2.48540 |
| Matched + contraction, FF250 | 2.45631 | 2.48511 |
| Separate language hybrid | 2.22206 | 2.27388 |
| Native attention32 | 2.17280 | 2.24546 |

The language aggregate weights the actual test bytes at T32/64/128 (14336 bytes per seed). Each run trains 153600 bytes from the raw train split. The local/global language hybrid remains a separate candidate and checkpoint family; no hybrid recall result is implied. Small CE changes need broader scale/seed replication before promotion.

The contraction improves aggregate byte CE by only about 0.002 nats/byte; T512 is essentially unchanged. The separate language hybrid produces a much larger gain and still trails attention. The new query route does not replace that language-focused candidate.

## 6. Execution and memory cost

| Multi-pair training | Controlled buffers | 900-update execution | ms/update |
| --- | --- | --- | --- |
| Full global baseline | 26.085 MiB | 14.725 s | 16.361 |
| Full global + contraction, FF257 | 26.472 MiB | 15.928 s | 17.698 |
| Matched + contraction, FF250 | 26.185 MiB | 16.276 s | 18.084 |
| Native attention32 | 29.296 MiB | 15.193 s | 16.881 |

Training execution includes host update preparation/binding/staging and C updates, excludes validation/data generation/startup/builds, and averages three seeds. There is no borrowed timing calibration or claim that a noisy short lookup finds the global optimum. The controller prefers the larger fitting microbatch and cheaper admitted reconstruction tier. All candidates see the same physical positions and supervision within each task.

The contraction consumes additional cache/parameters and small GEMMs/pool barriers. Its execution cost is worth paying only for a measured capability or convergence advantage. The full head’s low-storage property is retained, but comparison to attention must grant attention accumulation too.

This study runs native BCIR-derived attention and global kernels. It does **not** rerun a PyTorch training-throughput benchmark, establish a new speedup over PyTorch or qualify accelerators/bare metal.

## 7. Recommended next steps

1. Keep the full-head contraction as an explicit experimental recipe, with its tested numerical contract and checkpoint identity. Promote it only for tasks whose held-out quality/time-to-quality justify its cost; do not make it the default associative-memory mechanism.
2. Target **write-side binding** next: construct a local triple/adjacency-aware feature before global aggregation, or a small explicit key/value matrix state with learned write/read interfaces. Retain the plane for complementary global context. Compare against the same one-pair, four-pair, value-swap and query-change tests.
3. Separate the hypotheses: independent query projection, area/parallel-coordinate side channels, and a 1→2→4→8-pair curriculum should be individually named and parameter/data matched. More updates are a legitimate next study, not an explanation already proven by these 900-update failures.
4. Keep the common precision guard and budgeted C execution. For the useful ordered-recall candidate, fuse its feature/VJP work and group compatible readout VJPs to reduce barriers, then recheck gradients and trajectories. These are proposed optimizations; no speed gain from them is claimed here.
5. Retain the separate language hybrid and strong attention32 reference. Before claiming general architecture superiority, expand scale, data, seeds, stronger attention baselines and hardware qualification.

The key distinction is between a correct geometric program and a useful learned memory. The new implementation closes the equations-to-code gap for this hypothesis. The controlled association tests show that this contraction alone does not close the binding gap.

## 8. Reproduction, provenance and limits

The recovered integrated archive was SHA-256 verified (`e3899d3a52ab283e2aa7976fdc7c1e6bb95566563ceb3b97ffa808596caf2731`), and all 205 listed source/data/checkpoint entries matched their manifest. This study is self-contained: pinned C/tensor sources, exact ctypes ABI, generated C, build commands, independent oracle, protocols, raw events, test predictions/labels,51 checkpoints and source manifests are included. See `README.md` for clean-run commands.

Native foundations are pinned to BCIR `aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d`; inspected upstream main remains `be6e377a2eb417cb4bfe3de5243ca09961ba1833`. The host is AMD EPYC 9V74, eight-core quota/8 GiB container; experiments use4 native pool threads. GCC 13.3/C11 builds use `-O3`, no fast-math and disabled implicit FP contraction. Python 3.12, NumPy 2.3.5, PyArrow 26.0.0 and CPU Torch 2.14.1 support the study. Builds, model jobs and heavy checks were serialized.

The repository's requested newest LLVM/Clang/MLIR 23.1.3 and Python 3.15 were unavailable; Node 24.19 was behind24.22. Full repository CI, native ARM/Windows, accelerators, UEFI and physical hardware-memory movement are unqualified. This is isolated research C and a host controller, not a merged/registered production IR/MLIR operation. No repository source, commit or PR was changed.

Synthetic tasks and a small byte model do not predict large-language-model capability. Three seeds and limited training do not prove representational impossibility, universal superiority or statistical equivalence. Test data are paired and fixed, not thousands of independent training replications. Negative association results, full per-seed curves and the numerical stress findings are retained.

Source references:

- BCIR pinned native foundation: https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/tree/aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d
- Inspected main: https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/tree/be6e377a2eb417cb4bfe3de5243ca09961ba1833
- Dataset split hashes/URLs: `DATA_PROVENANCE.md` and `protocol.json`.
- Numerical evidence: `results/validation.json`, `results/conditioning-check.json`, `results/precision-budget-probe.json`.
- Learning evidence: `results/train-*.json`, `results/eval-*.json`, `results/summary.json`, `results/baseline-continuity.json`.
- Final source/checkpoint/pairing audit: `results/final-audit.json`.

All numeric outcomes in this report derive from the retained runs, not claimed figures in a research paper.
