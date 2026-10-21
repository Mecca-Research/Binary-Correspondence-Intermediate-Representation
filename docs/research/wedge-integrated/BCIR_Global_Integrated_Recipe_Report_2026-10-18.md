# BCIR full global recipe: integrated execution, memory budgets and variable-length learning

Research report · 18 October 2026

**The combined recipe is implemented and tested.** It retains the full rank-22 global wedge head, uses one reusable workspace, chooses cached/recomputed execution from a declared buffer budget, and physically trims causally unused suffixes. On matched ordered-recall data, trimming reduces training execution from **22.05 to 13.75 seconds: 1.60× faster**, without reducing head size. Variable-length learning improves the harder ordered-recall result at T1024 from **78.91% to 94.01%**.

The stronger comparisons also change the interpretation of the earlier results. Variable-length attention reaches 100% at T1024. When attention is also allowed accumulation inside 32 MiB, global’s controlled-buffer advantage is approximately **11%**, with similar observed training execution time. The global architecture has not established a universal performance or capability advantage.

The session completed **33 from-scratch training runs and 27,000 quality-training updates**, nine continuation arms totaling 360 further updates, a 240-update deterministic replay for archival verification, performance calibration/probes, independent gradients, counterfactual retrieval tests and precision audits. **42 selected/final checkpoints** are retained, hash-verified and restored. Each task trains separate model weights; this is not one jointly trained model demonstrating every capability.

## 1. Recovered foundation and environment

The starting report was read from its current saved version. Its 327-file experiment bundle was recovered, checked against SHA-256 `27e08abb19750ee3f8d33267d7f36603deaa3fbe6b3dc960a18e45411357ebca`, and verified file-by-file against its manifest.

Native model/tensor sources remain pinned to BCIR `aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d`. The inspected main repository is `be6e377a2eb417cb4bfe3de5243ca09961ba1833`, unchanged from the prior study. The new code is an isolated research controller and generated C portfolio, rather than a production BCIR IR/MLIR registration or merged runtime update. No repository commit, push or PR was made.

The host again reports AMD EPYC 9V74, affinity CPUs 0–8, an eight-core quota and an 8 GiB memory limit. Training and policy measurements use four native pool threads. Heavy builds/model jobs were serialized. GCC 13.3 builds C11 with `-O3`, floating contraction disabled outside the explicitly contracted kernels, and no fast-math reassociation. Python 3.12.14, NumPy 2.5.4, PyArrow 26.0.0 and CPU PyTorch 2.14.1 support the study. PyTorch supplies independent automatic differentiation; no PyTorch training-throughput benchmark was run.

The base global model has **525,696 parameters**, width 128, three layers, FF257, rank 22, 231 packed wedge coordinates, tied 260-token embeddings, RMSNorm and no RoPE. Native causal attention uses width/depth/vocabulary matched, FF256 and RoPE, also totaling 525,696 parameters. The separate language hybrid retains the existing equal local/global prefix blend and full geometric head.

## 2. What was built

### A reusable native execution controller

`policy.py` queries the actual C `make_plan` entry for every admitted shape/layout. It compares candidate storage and measured update cost, then binds the chosen C plan. All tensor operations, backward kernels and AdamW updates remain in C.

`Runner` owns one set of weights, gradients and two moments, one aligned workspace, staged inputs/targets and one native pool. Activations and scratch are disjoint views into the workspace. Switching physical length or reconstruction tier rebinds those views while preserving optimizer step and bias powers. Budget changes release old work/staging arrays before replacement; the persistent model arrays remain the same storage.

Plans carry model configuration, numerical family, effective/microbatch sizes, accumulation count, context, source hash, footprint, calibration status and stable content-derived IDs. The module source is checked on a new binding. The controller returns copied inference outputs; retaining borrowed internal activation views across resizing is outside its supported API.

The global portfolio includes cached/grouped execution, compact scratch, frame/FF reconstruction, whole-layer reconstruction and combined layer/frame/FF reconstruction. Compact guarded and language-hybrid variants are also compiled. Numerical families stay separate: a resize may change layout, but cannot silently change the model or its numerical contract.

### What the memory budget means

The **controlled-buffer bound** includes the four persistent FP32 arrays, unified activation/scratch allocation, alignment slack, the native pool-control buffer, staged effective-update inputs/targets and specified C call records. It excludes thread stacks, code, Python/NumPy wrapper metadata, datasets, checkpoint copies, inference-result copies and allocator-retained pages. It is **not a process RSS limit**. The budget probe records RSS separately to make this distinction inspectable.

The persistent array term alone is 16 bytes per parameter, approximately 8.02 MiB here. The runtime must reserve headroom for excluded resources before treating its remaining capacity as an array budget. At this scale, an admitted native layout is a verified storage calculation for the named buffers, not a guarantee about all operating-system memory consumption.

The MiB budgets are imposed workload-buffer envelopes on the 8 GiB host. No operating-system-enforced 20 MiB process limit or physical accelerator-memory limit was used.

### Physical variable-length execution

Ordered recall uses homogeneous batch lengths 64, 128 or 256, shuffled in a balanced schedule. The padded control processes T256 every time but places the supervised query at its logical end. The trimmed recipe executes exactly the corresponding prefix. Both use the same examples, labels, order, effective batch 4, optimizer and 900 updates.

Physical positions fall from **921,600 to 537,600**, a **41.67% reduction**. This is actual smaller C tensor execution, rather than a padding mask over an unchanged tensor. It is not arbitrary sequence packing: each example retains its own causal prefix reset.

The transformation relies on the declared causal mixer/attention, token-wise normalization and FF operations, no cross-example state, and no dropout/RNG indexing dependency. A general compiler must establish those properties before dropping a suffix. It cannot infer this optimization for an opaque or bidirectional operation merely from ignored loss positions.

## 3. Protocol and validation

The main 30-run protocol was frozen before training. Three seeds are 20261018, 20261019 and 20261020. Checkpoints are selected by aggregate validation CE at 100-update intervals, never by final tests.

| Task | Main cases | Updates/run | Training distribution |
|---|---|---:|---|
| Ordered first-value recall | Fixed global; variable padded global; variable trimmed global; guarded global; attention | 900 | All values/distractors share tokens 0–15; variable lengths 64/128/256; B4 |
| Random key/value association | Global and attention | 900 | Unique random keys/values; two pairs for 300 updates, then four; variable lengths 64/128/256; B4 |
| WikiText-2 raw byte language modeling | Global; separate local/global hybrid; attention | 600 | Lengths 32/64/128, batches 8/4/2; exactly 256 supervised bytes/update |

The main quality-controlled runs keep the full effective microbatch. Global and guarded recall use a 32 MiB controlled-buffer budget; full-microbatch attention uses 64 MiB. Language cases all use 32 MiB and 153,600 training bytes each. The subsequent three-run attention32 stage grants attention accumulation under the same 32 MiB limit, with identical data, rates, update counts and validation. Its protocol and fresh paired tests are separate from the main study.

Focused checks establish:

* Bitwise agreement with the pinned cached native model at each update while physical contexts and reconstruction libraries change; identical replay after checkpoint resume; same-input dynamic-shape state parity at one/four threads.
* Budget transitions 96 → 32 → 20 → 64 MiB preserve persistent storage identity and optimizer state, with fixed-microbatch reference updates bitwise identical.
* Trimming gives identical checked logits/loss. Gradient relative L2 differences are about **3.6e-8**, because changing row counts changes the blocked weight-reduction partition. It is mathematically equivalent under the causal contract, but not bitwise-identical training.
* Independent full-model gradients for global, guarded and hybrid compact models have relative errors below **7e-7** in these checks. UBSan covers compact, combined reconstruction, guarded and hybrid execution.
* All-masked updates and unequal supervised counts across microbatches are refused without changing model/optimizer state. The latter avoids incorrectly averaging differently normalized microbatch losses.

BCIR leaves the sum of microbatch-mean gradients in the gradient array and applies `1/accumulation` inside the optimizer. Independent accumulation checks account for that contract: normalized gradient relative errors against Torch are below **7.4e-7**. At full attention shape B4/T256, changing microbatch 4 to2 changes the first normalized gradient by **1.82e-7 relative L2** and the first updated weights by **2.94e-9 relative L2**. This does not imply equality after a long nonconvex training trajectory.

## 4. Main integrated results

Training execution includes per-update input preparation, binding/staging and C updates; it excludes dataset generation, validation, one-time calibration and startup. Values below are three-seed means. These are short CPU experiments, with measurable shared-host variability.

| Ordered-recall recipe | Controlled buffers reserved | Training execution | Physical positions |
|---|---:|---:|---:|
| Variable global, padded | 26.09 MiB | 22.05 s | 921,600 |
| Variable global, trimmed | **26.09 MiB** | **13.75 s** | **537,600** |
| Variable guarded global, trimmed | 26.09 MiB | 12.70 s | 537,600 |
| Variable attention, full microbatch | 50.56 MiB | 11.96 s | 537,600 |
| Variable attention, 32 MiB budget with accumulation | **29.30 MiB** | **13.40 s** | **537,600** |

The trimmed global gain is **1.603×** over its matched padded control. The same workspace is reserved for the largest admitted training/validation shape, so trimming reduces active work without requiring per-update allocation. The guarded timing is not evidence that FP64 guards accelerate training: it uses separately calibrated layouts and a different execution order on a noisy host.

At equal full microbatch, global reserves 48.4% less controlled storage than attention. Once attention can accumulate, the difference narrows to **11.0% less than attention32**. Training times are similar enough that this study does not establish a clear global throughput advantage over the matched-storage attention recipe.

This attention comparator retains the native probability-materializing implementation. An IO-aware exact-attention implementation could reduce its footprint further; this session does not benchmark that stronger baseline [6].

### Recall when target and distractors share the vocabulary

Main held-out samples are shared across model cases and seeds. Each point through T1024 uses 128 examples per seed; T2048 uses only 32 per seed. Means describe three seeds, not a confidence interval.

| Training / model | T128 | T256 | T512 | T1024 | T2048 |
|---|---:|---:|---:|---:|---:|
| Fixed T128 global | 100.00% | 100.00% | 99.22% | 78.91% | 26.04% |
| Variable padded global | 100.00% | 100.00% | 99.48% | 94.01% | 64.58% |
| Variable trimmed global | 100.00% | 100.00% | 99.48% | **94.01%** | **64.58%** |
| Variable guarded global | 100.00% | 100.00% | 99.48% | 94.01% | 64.58% |
| Variable attention, full microbatch | 100.00% | 100.00% | 100.00% | **100.00%** | **98.96%** |

Variable training improves global T1024 recall by **15.10 percentage points** and T2048 by **38.54 points**. Padded and trimmed recipes have the same observed recall here and nearly equal CE; their final weights differ by approximately 1e-6 to 1.4e-5 relative L2. Guarded and ordinary trimmed global finish with **bitwise-identical model/optimizer digests for all three seeds** on this ordinary training distribution.

At T512, swapping the first value with a different distractor preserves the token multiset and changes the correct answer. Trimmed global answers both versions correctly on **99.48%** of the paired tests; attention achieves 100%. This is stronger evidence of order-sensitive behavior than the previous disjoint-vocabulary shortcut task.

Attention still reaches the 99% aggregate-validation threshold sooner: mean training execution is roughly **3.11 seconds**, versus **10.76 seconds** for trimmed global. Crossings are observed only every 100 updates. The storage-focused global branch is useful, but update cost must not replace time-to-quality.

### Fairer attention storage and numerical sensitivity

Fresh matched-storage tests after checkpoint selection:

| Recipe | T256 | T1024 | T2048 |
|---|---:|---:|---:|
| Retained global32 | 100.00% | 94.53% | 65.63% |
| Retained full-microbatch attention64 | 100.00% | 100.00% | 100.00% |
| New attention32 with accumulation | 100.00% | 100.00% | 89.58% |

The T2048 point uses 32 samples/seed. Both attention variants perform perfectly in-distribution, but their long extrapolation differs. After 900 updates their weights differ by approximately 0.11–0.13 relative L2, despite the very small checked first-update rounding difference. Correct gradient normalization was independently verified. This is consistent with diverging nonconvex trajectories; the three-seed pilot does not establish that accumulation generally harms extrapolation. It does establish that execution/numerical changes need length-generalization checks as well as local gradient checks.

## 5. Choosing recomputation under available storage

Calibration uses three repetitions of train-byte inputs, one warmup and three timed effective updates per candidate. It took **30.85 seconds**. Candidate selection minimizes median observed update wall time among feasible plans. Held-out performance uses new inputs and six repetitions, each with two warmups and four timed effective updates. B4/T1024 is fixed; microbatch/accumulation can change.

| Budget | Calibrated selection | Controlled buffers | Held-out ms/effective update | Simple heuristic ms/update |
|---:|---|---:|---:|---:|
| 20 MiB | Combined reconstruction, micro1 ×4 | 18.76 MiB | 133.23 | 124.09 |
| 24 MiB | Combined reconstruction, micro1 ×4 | 18.76 MiB | 129.81 | 133.35 |
| 32 MiB | Frame/FF reconstruction, micro1 ×4 | 26.11 MiB | 106.49 | 120.77 |
| 48 MiB | Frame/FF reconstruction, micro1 ×4 | 26.11 MiB | 107.08 | 113.64 |
| 64 MiB | Frame/FF reconstruction, micro1 ×4 | 26.11 MiB | 108.44 | 113.66 |
| 96 MiB | Frame/FF reconstruction, micro1 ×4 | 26.11 MiB | 110.12 | 99.92 |
| 160 MiB | Cached/grouped, micro4 ×1 | 125.49 MiB | 94.57 | 87.18 |

The simple heuristic prefers the largest fitting microbatch, then the cheaper reconstruction tier. The calibrated policy can select a smaller microbatch even when a larger one fits. All admitted allocations stay within their declared storage bounds; 8, 16 and 18 MiB are refused before model-array allocation for this shape.

**The speed selector is not reliably optimal.** It beats the simple heuristic at some budgets and loses at others. At20 and 160 MiB the policies select the same plan yet show different measured medians, exposing substantial timing noise. The 96 MiB lookup also misses the faster held-out larger-microbatch choice. Consequently these decisions deserve a **best-measured** label, not an optimality certificate. A production selector needs longer/pinned calibration windows, uncertainty intervals, near-tie handling and explicit calibration amortization.

The durable gain is strict admission plus executable alternatives, rather than a claim that a tiny timing table solves all scheduling. On another CPU the supplied CLI avoids using the incompatible CPU-model cost table and falls back to the declared heuristic until recalibrated. Matching the CPU model still does not guarantee identical quota, frequency or memory behavior.

The 30.85-second calibration is longer than the 8.29 seconds saved by trimming one 900-update global run. Trimming itself does not require timing calibration. Keep deterministic storage admission and causal trimming available without autotuning; charge calibration against the additional performance it actually produces and amortize it across sufficiently many updates or reused recipes.

## 6. Continuation through changing budgets and a longer-context tail

Three selected global checkpoints each receive 40 further T1024 updates at LR 1e-4. One arm uses ample storage and full microbatch 4; the other changes budgets every 10 updates: **160 → 64 → 32 → 20 MiB**. The data and effective batch 4 remain identical. Both run to completion, with equal optimizer step counters.

Adaptive versus ample final weights differ by at most **1.34e-8 relative L2**, and the tested predictions agree. The continuation is deterministically replayed with the same final state digests, and all final states are archived. A new short-tail control receives 40 T128 updates using the same first values/labels and rate. It uses one-eighth as many physical input positions as the long tail, so it is an update-matched, not compute-matched control.

Fresh evaluation after all 40 updates, with no test-based checkpoint selection:

| Global checkpoint | T1024 | T2048 | T4096 |
|---|---:|---:|---:|
| Before continuation | 95.31% | 63.02% | 35.42% |
| 40 extra T128 updates | 98.70% | 86.46% | 52.08% |
| 40 extra T1024 updates, ample storage | **99.74%** | **91.15%** | 48.96% |
| Same T1024 updates, adaptive storage | **99.74%** | **91.15%** | 48.96% |

These are separate fresh samples from the main tables: 128/64/32 examples per seed at T1024/2048/4096. After the long tail, T1024 is a trained context, not an extrapolation result.

Extra optimization alone accounts for much of the gain. The long tail adds about one percentage point at T1024 and 4.69 points at T2048 relative to the short tail, but does not win at T4096. Mean execution is 0.52 seconds for the short tail, 3.62 seconds for ample long execution and 4.10 seconds for adaptive long execution, excluding evaluation and resize overhead outside update timing. A longer-context tail is a useful optional curriculum stage, not a universal improvement. T4096 remains a clear recall gap.

## 7. Precision guards: useful contract, limited ordinary-data effect

The prior FP64 fallback is integrated into compact and reconstructed layouts. Audit-only atomic counters are compiled separately and excluded from training/timing variants.

For the trained guarded global model, ordinary inference selects the guard for exactly the initial absent prefix in these samples: **0.78125% at T128 and 0.097656% at T1024**. A 90%-repetition stress distribution raises the T1024 mean only to approximately 0.11224%. The initial absent plane remains 0.097656%. Contextualized deeper-layer vectors are not automatically nearly collinear merely because input tokens repeat.

The guarded and ordinary training checkpoints match bitwise on all three seeds here, so this study supplies no ordinary-data recall gain from the guard. Keep it as an explicit numerical option for conditioning stress and audit its rate. The earlier constructed near-collinear cancellation result remains a primitive result; it should not be repackaged as a measured accuracy gain.

## 8. The separate language candidate

All three language cases receive 153,600 raw training bytes, balanced physical lengths 32/64/128 and the same optimizer/validation convention. Paired test aggregation uses 14,336 bytes per seed across those lengths.

| Language architecture | Test CE, nats/byte | Controlled buffers | Mean training execution |
|---|---:|---:|---:|
| Full global | 2.45435 | 15.36 MiB | 4.31 s |
| Equal local/global blend | **2.22032** | **14.72 MiB** | 4.09 s |
| Native attention | **2.15401** | 17.16 MiB | 3.88 s |

The hybrid closes approximately **77.9% of this global-to-attention CE gap**, while attention remains better. At T512 the respective CEs are 2.48842, 2.27969 and 2.24762. This supports keeping the language blend as its own candidate. It does not demonstrate simultaneous recall and language quality in one set of weights, nor remove the earlier hybrid recall trade-off.

The two global-family language recipes have the same parameter count and admit the same layout sizes. Their small reservation difference in this table comes from selecting cached versus compact execution, rather than a smaller hybrid architecture.

## 9. Associative binding remains the major capability gap

The key/value task uses explicit triples `(258,key32..47,value0..15)`, unique random keys/values and filler64..95. The final query specifies one key. A successful model must retrieve the associated value rather than the first or most common value. Four candidate values are present in the main test. An oracle that identifies that set but guesses its assignment would average25%; unconditional random choice among16 labels averages6.25%.

| Four-pair recall | T128 | T256 | T1024 | T2048 |
|---|---:|---:|---:|---:|
| Global | 6.51% | 6.77% | 4.95% | 0.00% |
| Attention | 27.34% | 25.00% | 21.35% | 12.50% |

At T256 with one pair, global achieves 6.51% while attention achieves 99.22%; with two pairs they achieve 5.99% and 52.60%. At T128, swapping two values while keeping the token multiset and query key fixed gives paired both-correct rates of **0% global and 5.21% attention**. Neither model establishes reliable arbitrary key/value binding at this budget. Attention’s results are largely compatible with candidate-value access without robust assignment retrieval. More training and alternative objectives may change this; these results are not a representational-impossibility proof.

An independent C grammar-known parser/registry reference answers **all 1,568 held-out/control queries correctly**, including swaps and 1/2/4/8-pair cases. Its per-stream table occupies 68 bytes and allocates nothing. This verifies labels and provides an exact structured-input reference. It is **not a trained neural model**, does not learn the grammar and is not a fair general-language speed or intelligence comparison.

For BCIR’s integrated lab, the distinction is useful: schema-known symbolic associations can live in typed registries with exact lookup. Learned models can recognize/produce those operations for less structured inputs. The present neural task results do not yet justify delegating all such memory to the global head’s weights/prefix summary.

## 10. The next architectural hypothesis to test

The global branch now has an execution recipe worth retaining. Its next model change should target the exposed query-dependent readout problem, rather than shrink the head again.

A specific multilinear hypothesis is a query-aware interior-product side channel. Let P be the full antisymmetric matrix of normalized packed wedge coordinates, and let q = u/||u||. Then, away from the absent-plane region,

\[
c_j=\sum_i q_i P_{ij}
\quad=\quad
\frac{v_j-\langle q,v\rangle q_j}
{\|v-\langle q,v\rangle q\|}.
\]

The current dense readout sees p; a candidate can retain that full route and add

\[
g=W_p p+W_c c.
\]

This explicitly contracts the current query frame with the plane instead of requiring subsequent gates/MLPs to learn that interaction implicitly. The formula still loses the prefix component parallel to q; it does not reconstruct arbitrary v or guarantee retrieval. Optional area/parallel-coordinate features are further hypotheses with their own conditioning and normalization contracts.

This is a proposed inductive bias, not an established causal diagnosis of the one-pair failure.

At rank 22,width128, Wc adds 2,816 parameters/layer. Reducing FF257 to250 removes 2,688/layer, yielding a nearly parameter-matched **526,080-parameter** recipe, +0.073%. Test both the unchanged-FF and matched-budget versions to separate the side channel from FF reallocation. This model has **not been implemented or shown superior in this session**. Its next gate should be native C/Torch gradients, near-collinear cases, the one-pair failure, four-pair counterfactual binding, ordered recall and language under fixed data/budgets.

A second candidate is a small explicit keyed matrix/registry state with a learned write/read interface, while retaining the full global plane for other context. That is a more direct inductive bias for association, but it adds state and requires trained recognition/update semantics. Neither proposal should inherit the current recipe’s memory or quality claims without measurement.

## 11. BCIR integration blueprint

| Component | Implemented research evidence | Next production integration |
|---|---|---|
| Model/numerical registry | Separate full-global, guarded, hybrid and attention families; stable recipe/plan IDs | Carry these identities and checkpoint compatibility through the IR/training recipe |
| Causal length bound | Actual prefix trimming with paired data and gradient checks | Admit a last-supervised-position bound only for certified causal regions; opaque fallback retains padding |
| Workspace planning | Actual C plan queries and one reusable allocation | Have the verified execution plan drive offsets, aliases and kernel ordering in native training |
| Memory pressure | Real budget-dependent reconstruction, accumulation and resizing | Integrate hardware-envelope capacity and runtime headroom; distinguish controlled storage from RSS |
| Cost selection | Measured lookup plus held-out comparison to a simple heuristic | Longer calibration, confidence/near-tie handling, amortization and hardware/version invalidation |
| Loss normalization | Equal supervision counts enforced across microbatches | Add explicit global-denominator/weighted accumulation before mixed-length packing with unequal active counts |
| Reproducibility | Frozen main protocol, separately declared follow-ups,42 checkpoints and replay digests | Preserve state/trace provenance and qualify the full supported toolchain/hardware matrix |
| Capability routing | Ordered recall, binding, language and exact schema-known reference separated | Route supported structured associations to registries; gate neural retrieval on actual held-out performance |

The new high-value compiler fact is that a causally unreachable masked suffix is dead computation. This couples semantic bounds to shape, allocation and execution cost. The wedge-specific frame reconstruction remains useful, while suffix trimming, general rematerialization and budget-aware accumulation are broader compiler/runtime techniques that can benefit other models too.

Checkpoint/recomputation techniques have established prior art: Chen et al. analyze sublinear-memory training, while Checkmate formulates rematerialization schedule selection with profiled hardware costs and optimization [1,2]. This controller explores a small portfolio of hand-generated plans; it is not a replacement for a general joint schedule/layout/rematerialization solver or a global TMSAO proof.

## 12. Recommended working recipe and remaining qualifications

For continued global-head research, use the full rank 22 head, batch-homogeneous variable lengths, causal trimming, compact workspace admission and budget-dependent reconstruction. Preserve effective batch and loss weighting; test each numerical convention at the required lengths. Treat precision guards as an option with measured coverage, and the language hybrid as a separate checkpoint family. A short refinement tail is cheap; a longer-context tail can improve nearer long contexts but costs more tokens and does not fix T4096.

Use attention32 as the matched-storage quality reference, and the exact registry only for its known schema. The current evidence supports a working memory-focused global branch, not a universal replacement for attention, arbitrary associative memory or mature libraries.

All42 archived checkpoints restore correctly. Frozen main sources, paired training/test hashes, admitted footprint calculations and fixed-microbatch execution-state equality pass the final audit. Independent gradient, accumulation, resume, counterfactual, refusal and focused UBSan checks pass. The supplied single-run CLI and JSON recipe make the implementation concrete and reusable.

The newest requested LLVM/Clang/MLIR23 and Python3.15 are absent; Node24 is behind the requested point release. Full BCIR CI, native ARM/Windows, accelerator, UEFI and bare-metal qualification were not run. No physical HAM/DMA transfer was measured in this session. This research artifact is not a production integration or a new PyTorch throughput result.

## Sources and evidence

1. Chen et al., *Training Deep Nets with Sublinear Memory Cost*: https://arxiv.org/abs/1604.06174
2. Jain et al., *Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization*: https://arxiv.org/abs/1910.02653
3. Pinned BCIR foundation: https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/tree/aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d
4. Inspected BCIR main revision: https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/tree/be6e377a2eb417cb4bfe3de5243ca09961ba1833
5. Data provenance and split hashes: `DATA_PROVENANCE.md`, `protocol.json`.
6. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*: https://arxiv.org/abs/2205.14135
7. Audited numerical aggregates: `results/summary.json`; source/checkpoint checks: `results/final-audit.json`.

All reported experiment measurements derive from retained native runs, not benchmark figures from the cited papers. The bundle includes raw repetitions, per-seed results, protocols, generated C, independent references, checkpoint arrays and reproduction instructions.
