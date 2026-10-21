# BCIR global wedge: memory, recall and the next architecture frontier

Research experiment · 17 October 2026

**The strongest result is a portfolio of execution plans for the same model.** Native buffer reuse and recomputation reduced the global model’s activation/scratch arena from **30.875 MiB to 10.703 MiB**, with identical tested logits, gradients, optimizer state and resumed runs. The smallest arena costs additional computation. The best model architecture also depends on the task: the original global wedge retained long recall better than the thinner multilinear variants; a local/global blend improved language loss but weakened recall.

This study completed **57 new native-C learning runs and 28,200 learning updates**, plus execution benchmarks, independent automatic-differentiation checks, checkpoint restoration, precision stress tests and verified BCIR memory-planner scenarios. It establishes useful conditional bounds and implementation choices. It does not establish a globally optimal architecture or a new PyTorch speed result.

## 1. Baseline, provenance and comparison limits

The recovered report, `BCIR_Grassmann_Global_Report_2026-10-17.md`, supplied the starting implementation and checkpoints. Its experiment archive was recovered and verified against SHA-256 `0d137f15f2e1cdccffaffb100ae818b7f95bbfb757415cd227a0f1d2dd1737c9` before extraction.

The earlier **48.97 ms/update** measurement was on an **Intel Xeon Platinum 8573C**. This session runs on an **AMD EPYC 9V74**, with an eight-core CPU quota, affinity allowing CPUs 0–8 and an 8 GiB memory limit. One-thread timings are pinned to CPU 0; four-thread timings use the native BCIR pool. The old 49 ms result remains a historical reference for its original host, rather than a portable latency target. This study remeasures baselines within each alternating timing block.

The native tensor/training foundation is pinned to BCIR commit `aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d`. The HAM and exact-layout planners are from main commit `be6e377a2eb417cb4bfe3de5243ca09961ba1833`. The repositories are unchanged. The new C kernels and scripts are isolated research artifacts; they are not yet registered BCIR/MLIR operators or a GEM+ training-plan execution path.

The baseline is a **525,696-parameter** byte decoder: width 128, three layers, tied 260-token embeddings, RMSNorm, SwiGLU width 257, and a rank-22 global wedge without RoPE. The attention comparator has the same parameter count, width/depth/vocabulary and SwiGLU width 256, with causal attention and RoPE. Both train through BCIR’s native C loop and AdamW. PyTorch is an independent correctness reference here; no training-throughput comparison with PyTorch was run.

Builds use GCC 13.3, C11, `-O3`, contraction disabled, and no fast-math reassociation. The existing BCIR tensor backend already contains runtime-selected AVX2/AVX-512 matrix kernels. This study does not assume it was running scalar generic GEMM. Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, PyArrow 26.0.0 and CPU PyTorch 2.14.1 support the experiments and references.

### What was fixed before training

`protocol.json` pins the six main recipes, seeds, data hashes, optimizer, budgets and source hashes. Three seeds are used: 20261017, 20261018 and 20261019. Validation CE at 100-update intervals selects checkpoints; held-out test results do not select among checkpoints. The six-run hybrid follow-up has a separate protocol, one fixed equal blend and fresh paired test samples. It is exploratory rather than an independently replicated final design.

| Task | Models × seeds | Budget per run | Evaluation |
|---|---:|---|---|
| WikiText-2 raw byte language modeling | 6 × 3 | 600 updates, B4/T64; 153,600 supervised bytes | 16,384 test bytes per run |
| Fixed-length first-token recall | 6 × 3 | 400 updates, B8/T128; 3,200 examples | 2,048 test examples per run |
| Variable-length recall | 3 × 3 | 400 updates, B4/physical T256; 1,600 examples | Query length sampled from 64/128/256 |
| Ordered recall with overlapping vocabularies | 2 × 3 | 600 updates, B8/T128; 4,800 examples | 2,048 test examples per run |
| Exploratory hybrid language/recall | 2 tasks × 3 | Same respective budgets as above | New paired test samples |

Variable-length training computes the entire padded T256 tensor. It is not a packed/ragged execution optimization. Fixed and variable recall use the same number of physical input positions per update, but different numbers of examples and supervised query positions. Length-stress evaluations use 128 fresh samples per seed and length. Reported means and ranges describe these three runs; they are not population confidence intervals.

## 2. The geometry and what can be preserved exactly

For normalized layer input h, the model computes z = W_reduce h. Its strictly earlier prefix is

\[
v_t=\frac{\sum_{j<t}z_j}{\sqrt{\max(t,1)}},\qquad
p_{t,ij}=\frac{z_{t,i}v_{t,j}-z_{t,j}v_{t,i}}
{\max(\|z_t\wedge v_t\|_2,10^{-6})},\quad i<j.
\]

There are 231 packed coordinates at rank 22. A dense readout projects p to width 128; a learned elementwise gate combines that readout with h. Residual and feed-forward processing follow.

The prefix and reverse-adjoint scans already remove direct quadratic prefix rebuilding. The earlier **12.2× result at T4096** concerns that geometric forward/backward primitive, not the entire training update. It cannot be multiplied into a whole-model speedup a second time.

A decomposable wedge can be regenerated from its two vectors: **44 frame floats instead of 231 packed floats**, an 81.0% representation reduction before considering the rest of the model. The existing VJP already requires those vectors. This makes discarding and reconstructing p especially attractive: it removes an additional cache without requiring an approximate subspace projection.

This is an execution transformation. Restricting a learned readout to lower rank or replacing one subspace with several smaller ones changes the model class. Those changes require new learning and quality checks.

Two further mathematical constraints matter:

* A normalized wedge is invariant to positive rescaling of its prefix away from the epsilon region. Thus division by sqrt(t) does not, by itself, provide a reliable length signal. Removing that division changes FP32 rounding and epsilon behavior, so it is not one of this study’s bitwise-equivalent transformations.
* In the first layer, an unweighted prefix sum is invariant to permutation of earlier reduced inputs. Deeper causal layers can encode order through their contextualized inputs. Long receptive field alone is therefore insufficient evidence of general positional retrieval.

## 3. Exact execution improvements and the memory frontier

Implemented transformations include removal of unused scratch, consumption of logits in-place by the loss gradient, grouped linears that share an input, pooled gates, lifetime-based scratch reuse, frame reconstruction, feed-forward reconstruction and whole-layer reconstruction. Whole-layer reconstruction keeps residual boundaries but shares one layer cache. Its final combined variant also discards packed frame features and FF gate/up intermediates, and avoids a redundant prefix scan when the cached prefix remains valid.

**B1/T1024, 525,696 parameters; medians from the final six-repetition alternating block:**

| Execution plan | Activation + scratch arena | One pinned CPU, ms/update | Four-thread pool, ms/update |
|---|---:|---:|---:|
| Original global | 30.875 MiB | 54.22 | 25.95 |
| Grouping/gates and scratch reuse | 26.785 MiB | 52.75 | 25.84 |
| Frame + FF reconstruction and scratch reuse | 18.055 MiB | 56.37 | 26.09 |
| Whole-layer reconstruction and scratch reuse | 13.613 MiB | 70.52 | 27.49 |
| Whole-layer + frame + FF reconstruction | **10.703 MiB** | **70.39** | **32.92** |

All plans add **8.021 MiB** for weights, gradients and two FP32 AdamW moments. Arena measurements are planned array sizes, not process RSS, cache misses or off-chip traffic. Pool storage, Python/data buffers and allocator overhead are outside these numbers. The new loss path consumes logits during backward; callers needing logits afterwards must copy them before backward or request a different buffer contract.

The smallest arena is **65.3% smaller than the original global arena**. It is **86.4% smaller than the 78.531 MiB arena of the retained native attention comparator**. At one CPU it takes about 30% more update time than the original global model in this block. These figures describe a memory/time trade-off, not a speedup from minimizing every allocation.

The earlier, separately retained execution block measured original global at 52.43 ms and native attention at 124.17 ms on one CPU. Repeated four-thread results vary appreciably: for example, the final combined reconstruction plan ranged from 25.67 to 38.59 ms. The bundle retains every repetition and both timing blocks; small percentage latency changes should be treated as provisional on this shared host.

The attention arena is not a lower bound for attention. An IO-aware exact implementation such as FlashAttention can avoid storing the full quadratic attention matrix. Any broader architectural claim must compare against that stronger memory baseline too [1].

### Using the savings for larger microbatches

A separate real C-loop experiment fixes effective B4/T1024 and a **64 MiB native-buffer budget**, including one input/target pair. It selects the largest fitting microbatch and uses accumulation for the remainder. Four repetitions, four native threads:

| Plan | Microbatch × accumulation | Native buffers + one input pair | Median ms/effective update |
|---|---:|---:|---:|
| Original global | 1 × 4 | 38.904 MiB | 101.05 |
| Grouped/gated global | 1 × 4 | 37.389 MiB | 104.14 |
| Scratch reuse | 2 × 2 | 61.607 MiB | 96.20 |
| Frame + FF reconstruction | 2 × 2 | 44.146 MiB | **89.25** |
| Whole-layer reconstruction | 4 × 1 | 62.506 MiB | 113.24 |

Frame/FF reconstruction produces about **13.2% more effective-update throughput** than the original in this constrained experiment. The plan fitting the largest microbatch is slower. Different microbatch shapes change rounded gradient accumulation order; identical examples and effective optimizer-update counts do not imply bitwise-identical states across those shapes. This is a declared buffer budget, not evidence that the host or all attention implementations run out of memory at 64 MiB.

## 4. How BCIR HAM and static memory planning benefit

HAM does not need a new kind of allocator to benefit. It needs accurate resource sizes, lifetimes, aliases, reconstruction costs and the schedule under which those claims hold.

The study exercised the **actual BCIR HAM lowering and verifier**, using two independent read-only resources, each sized as one model’s entire native working set. Six accesses alternate between the jobs in a declared fast bank. Resources start in host RAM; links are declared as 16 GB/s plus 1 microsecond. Every admitted plan passes `verify_ham_plan` and is retained with its digest.

**64 MiB fast-bank scenario — planner predictions, no hardware transfer executed:**

| Plan | One job’s native working set | Predicted transfer volume | Capacity evictions | Predicted transfer delay |
|---|---:|---:|---:|---:|
| Original global | 38.90 MiB | 233.38 MiB | 4 | 15.301 ms |
| Scratch reuse | 34.81 MiB | 208.84 MiB | 4 | 13.693 ms |
| Frame + FF reconstruction | 26.08 MiB | **52.15 MiB** | **0** | **3.420 ms** |
| Whole-layer reconstruction | 21.63 MiB | 43.27 MiB | 0 | 2.838 ms |

Two original jobs do not fit simultaneously; two frame/FF jobs do. Crossing that threshold yields **77.7% fewer transfer bytes and approximately 4.47× lower modeled transfer delay** for this access sequence. At a bank large enough to hold both original jobs, the capacity-eviction advantage disappears and the remaining reduction follows the smaller cold-load sizes. The native attention working set in this fixture does not fit a 64 MiB bank.

These resources are deliberately conservative whole allocations, treated read-only. Actual training writes gradients/moments, may need dirty writebacks, and should split resources by real lifetime. Consequently the prediction is not a measured GPU, CXL or HBM speedup, nor a complete training transfer plan.

For static allocation, 14 logical scratch lifetimes were passed to BCIR’s real first-fit and exact-layout solvers. First-fit required **4,206,592 bytes**. Exact layout reached the simultaneous-liveness lower bound of **3,682,304 bytes**, with `stop_reason="optimal"`; the manually generated C offsets were checked independently for overlap. That is an optimal placement **for this fixture’s serialized lifetimes and disjoint VJP-buffer requirements**. Changing alias permissions, kernels or concurrency changes the optimization problem.

The appropriate BCIR integration is to expose a frame as a reconstructible resource, keep a readout’s weight VJP before reuse of its feature buffer, and jointly validate the execution schedule and layout. A parallel scheduler cannot reuse the serialized layout unchanged. HAM should compare reconstruction cost with transfer/retention cost under the calibrated hardware envelope. ASN.1 can describe those contracts and artifacts; repeatedly serializing dense floating-point activations is not established as a hot-path optimization here.

## 5. Multilinear architecture experiments

The earlier multilinear work supplies useful distinctions: exact factorization versus truncation, basis/gauge freedom, mode-wise contraction and the cost of representing intermediate results. This study implemented two concrete restrictions in native C, with independent Torch gradients.

**Output-mode factorization:** view the antisymmetric readout as W[d,i,j], and factor its packed matrix as C[D,Q] B[Q,231], with Q32. This is a Tucker-style factorization of one mode, not a completed MPCA training algorithm. It removes parameters and matrix work but adds an N×Q intermediate.

**Product of subspaces:** replace one rank-22 plane with two independently normalized rank-12 planes. Packed feature count falls from 231 to 132. Gr(2,22) has intrinsic dimension 40, as does Gr(2,12) × Gr(2,12), but these are different spaces and learned functions. Equal dimension is not a lossless conversion. The C implementation assigns a plane with norm ≤1e-6 zero feature and zero VJP; this explicit absent-plane policy resolved a gradient disagreement at duplicated initial tokens before training began.

Each family also has a parameter-budget variant that spends the saved parameters on a wider FF layer. Main tests use the same three seeds, training batches, optimizer and respective budgets.

| Recipe | Parameters | One CPU T1024 ms/update¹ | Arena at T1024 | Byte-language CE² | Fixed T128 recall² | T1024 recall after fixed training³ |
|---|---:|---:|---:|---:|---:|---:|
| Original global, grouped/gated | 525,696 | 53.58 | 29.359 MiB | 2.4560 | 100.00% | **82.03%** |
| Q32 readout, original FF | 471,456 | 51.45 | 29.859 MiB | 2.4539 | 99.98% | 49.48% |
| Q32 readout, FF304 | 525,600 | 53.55 | 32.062 MiB | 2.4567 | 93.64% | 54.17% |
| Two rank-12 planes, original FF | 488,448 | **49.72** | 27.875 MiB | 2.4521 | 100.00% | 47.14% |
| Two rank-12 planes, FF289 | 525,312 | 52.69 | 29.375 MiB | 2.4535 | 100.00% | 64.58% |
| Native causal attention | 525,696 | 125.18 | 78.531 MiB | **2.1696** | 100.00% | 72.92% |

¹ Separate architecture timing block, six repetitions, B1/T1024. ² Mean of three newly trained checkpoints; language evaluation is T64. ³ Mean on 128 shared fresh recall examples per seed, evaluated at T1024. No per-recipe hyperparameter search.

The thinner designs are useful speed/parameter candidates, but their short-length scores hide substantial extrapolation losses. Reallocating saved parameters into FF width is not consistently beneficial. Q32 also demonstrates that fewer parameters do not necessarily mean a smaller activation arena.

### Post-training compression is a different question

A separate SVD study factorized the three retained older dense-recall checkpoints without fine tuning. It transfers weights only; AdamW states are not migrated.

| Readout rank | Recall at T128 | T512 | T1024 |
|---|---:|---:|---:|
| Dense | 100.00% | 100.00% | 88.54% |
| 16 | 16.15% | 9.38% | 12.24% |
| 32 | 64.06% | 20.31% | 18.23% |
| 64 | 100.00% | 80.73% | 61.72% |
| 128 | 100.00% | 100.00% | 88.54% |

Full rank preserves these observed accuracies but increases parameter count; it is not a compression win. Truncation seriously damages long recall. A mathematically valid low-rank approximation is not automatically a task-preserving model optimization. Tensor-train and other multilinear neural parameterizations are established research directions [2]; useful BCIR work would add structured contraction choices, optimizer-state semantics and measured quality/cost contracts rather than infer superiority from the factorization alone.

## 6. Retaining recall: length, shortcuts and time to quality

For disjoint-vocabulary recall, the target value is token 0–15, distractors are 16–31, BOS is 256 and the final query is 257. The task admits a content-isolation shortcut. Ordered recall instead draws the first value and every distractor from 0–15, so identifying the target requires order information.

| Training / model | Recall T128 | T512 | T1024 |
|---|---:|---:|---:|
| Fixed T128 / global | 100.00% | 100.00% | 82.03% |
| Variable 64/128/256 / global | 99.74% | **98.70%** | **96.09%** |
| Variable / Q32 budget | 71.61% | 70.83% | 61.46% |
| Variable / two-plane budget | 93.75% | 91.93% | 82.29% |
| Fixed T128 / attention | 100.00% | 100.00% | 72.92% |
| Ordered T128 / global, ordered test | 100.00% | 99.74% | 72.40% |
| Ordered T128 / attention, ordered test | 100.00% | 99.74% | 63.28% |

Variable-length training is the clearest measured improvement to original-global extrapolation. It improves T1024 recall by **14.06 percentage points** on the shared length-stress sample set. This is a training-distribution result, not a faster execution kernel or proof of universal retrieval superiority. Variable-length attention training was not run.

Changing the fixed-trained models’ distractors to the ordered vocabulary without retraining exposes the shortcut: at T128, global averages **0.26%**, attention **17.19%**, and the thinner wedge models 0% on this OOD test. Retraining on ordered examples gets both global and attention to 100% at T128. These results are a strong reason to gate associative recall, repeated values, positional retrieval and distractor shifts separately.

Attention also reaches quality faster on the training tasks. At the coarse 100-update validation checkpoints, all global seeds first reach 99% fixed recall at update 300, versus update 100 for attention. Mean measured training time to that checkpoint is **7.46 seconds versus 2.23 seconds**. For ordered recall it is update 400 versus 100, **9.44 versus 2.02 seconds**. Evaluation overhead is excluded, timings use four threads, and actual threshold crossings lie somewhere between checkpoints. A cheaper long-context update does not imply faster training to a target capability.

## 7. Exploratory local/global blend

After the isolated global and multilinear comparisons, a single parameter-free blend was implemented:

\[
v_t=\tfrac12\frac{\sum_{j<t}z_j}{\sqrt{\max(t,1)}}+
\tfrac12 z_{t-1},\qquad p_t=\operatorname{normalize}(z_t\wedge v_t).
\]

The local term is zero at t0. Both terms share the current factor, so the result remains one decomposable wedge; it is not an arbitrary average of unrelated planes. Its C reverse scan includes the additional local adjoint. Independent full-model automatic differentiation and one/four-thread state parity pass.

Fresh paired WikiText-2 test windows, after unchanged training/validation budgets:

| Architecture | CE, nats/byte | Byte accuracy |
|---|---:|---:|
| Global only | 2.4683 | 29.50% |
| Equal-blend hybrid | **2.2310** | **36.80%** |
| Native attention | 2.1900 | 38.00% |

The blend closes approximately **85.2% of this global-to-attention CE gap**, while retaining the same parameter count and 29.359 MiB arena. In the final one-core timing block it takes 54.10 ms versus 52.47 ms for the grouped global baseline. This is promising evidence that a local channel addresses part of the language-modeling gap.

Its recall cost is substantial: fresh fixed-T128 recall averages **98.06% versus 100%** for global; at T1024 the separate length-stress set falls to **44.79% versus 82.03%**. Therefore the equal blend should remain a language-focused candidate. A gated residual local branch, length-conditioned mixture or independent global/local channels deserves a new ablation. Their advantages have not been demonstrated here, and adding parameters must be controlled.

## 8. Nearly collinear vectors and precision

Normalization becomes ill-conditioned as wedge area approaches zero. Better arithmetic reduces cancellation error; it cannot restore direction information lost by exact collinearity or by rounding inputs to FP32.

The implemented guard estimates the Gram determinant in FP64. When `(uu*vv - uv²) <= 1e-4*uu*vv`, pair products, norm and VJP accumulation use FP64 before returning FP32 features/adjoints. A plane with norm ≤1e-6 has zero feature and zero VJP. This is a **new numerical contract**, rather than a bitwise-equivalent execution plan.

In the stress case v := v + 1e6 u, using the same quantized FP32 inputs:

| Measurement | Original FP32 pair arithmetic | Guarded arithmetic |
|---|---:|---:|
| Maximum Plücker relation residual | ~3.17e-3 | **2.53e-9** |
| Feature / VJP agreement with the corresponding FP64-input-arithmetic reference | Cancellation error present | Zero measured feature and relative VJP error in this case |

The full-model reference check, including duplicated initial tokens, has maximum gradient error **2.46e-7** and relative gradient L2 error **5.78e-7**. Exact-collinearity zero-feature/zero-VJP behavior is checked. On the retained fixed-global checkpoints, guarded T1024 recall remains 82.03%; better residuals do not fix this model’s recall limitation.

Final timing-block medians are 52.97 ms guarded versus 52.47 ms ordinary on one core, and 21.53 versus 20.57 ms at four threads. The observed one-core difference is about 1%, with overlapping ranges; it is not a guaranteed overhead bound. Guard-branch frequency was not instrumented.

Before promotion, measure fallback frequency, wedge-area distributions, clipping/refusal rates and accuracy on repeated-token and adversarially near-collinear inputs. A zero-VJP dead plane can also inhibit learning. Ridge normalization, orthogonalized frames or a learned degeneracy bypass change derivatives and need their own training protocol. Lower-precision storage should be evaluated after these safeguards, not assumed safe from the memory result.

## 9. What “Theoretical Maximum System Architecture Optimization” can mean here

There is no justified single theoretical minimum in milliseconds for this model on this virtualized host. A useful TMSAO claim must name its semantics, allowed transformations, hardware envelope, memory budget and quality requirements.

**A proved local endpoint:** the 3,682,304-byte scratch placement matches the liveness lower bound for the declared fixture. The 10.703 MiB full arena is only a feasible measured implementation; a globally minimal training arena has not been proved. Fused loss, additional in-place VJPs, norm regeneration, tiled readouts or different checkpoint schedules may lower it further.

**A measured bottleneck bound:** the one-core profile of the grouped global model is:

| Region | ms/update | Share of whole step |
|---|---:|---:|
| Forward matrix operations | 13.595 | 24.6% |
| Backward matrix operations | 25.612 | 46.3% |
| Geometry forward + backward, including scans | 3.890 | **7.0%** |
| Activation/norm backward | 2.444 | 4.4% |
| Other work, including loss, optimizer and remaining scans | 9.755 | 17.6% |
| Total instrumented update | 55.296 | 100% |

With all other work unchanged, making geometry infinitely fast gives at most **1.076× whole-step speedup**. At four threads the corresponding geometry-only ceiling is **1.152×**. Halving all measured matrix-operation time would instead predict roughly **1.55×** one-core step speedup, under the same fixed-other-work assumption. These are Amdahl bounds for this profile, not predictions that such optimizations are achievable.

The model issues approximately **3.224 billion linear-matmul FLOPs per B1/T1024 update**, excluding scalar, normalization and optimizer work. Its small geometric representation does not remove dense width×width, FF or vocabulary projections. The next speed work should target their exact shapes, packing reuse, pair/group scheduling, vector tails, loss/optimizer passes and memory movement.

For a calibrated hardware envelope, a lower bound can combine critical path and compute/traffic terms:

\[
t\;\ge\;\max\left(t_{\rm critical\ path},\;F/C_{\rm attainable},\;
Q_{\rm offchip}/B_{\rm attainable}\right).
\]

The Roofline traffic term requires actual memory traffic after cache effects, not arena capacity [3]. Neither counters nor a bare-metal bandwidth/compute calibration were collected, so this study supplies no numerical roofline optimum. Communication, launch/barrier and state-update constraints also need explicit treatment.

Algorithmically, scans remove direct O(T²r) prefix work, leaving O(BTr²) feature work and dense model operations. Streaming inference may keep O(Br) prefix state per layer, but training still needs residuals, adjoints, optimizer state and a checkpoint/reconstruction strategy. An O(r) prefix state is not an O(r)-memory training system.

The correct objective is a constrained frontier: minimize **time to specified held-out quality**, with limits on memory and numerical failures and minimum recall across required lengths/tasks. Maximum update rate, smallest arena and highest recall need not occur at the same point. Hardware-aware linear-attention work provides established examples of these state/parallelism/IO trade-offs [4].

## 10. Recommended BCIR updates, in order

| Priority | Update | Evidence and acceptance condition |
|---|---|---|
| 1 | Promote semantic buffer lifetimes and reconstruction choices into the training plan | Carry explicit alias/use-after-consume rules, frame/prefix provenance, kernel order and workspace sizes; verify the selected layout actually drives C execution. Preserve the tested exact-model contract. |
| 2 | Keep original global capacity while optimizing dense kernels and runtime overhead | Geometry is only 7% of the one-core step. Benchmark representative reduce/readout/gate/FF/head shapes, forward and both VJPs, including ragged dimensions and transpose cases. The existing AVX kernels are the starting point. |
| 3 | Select recomputation by calibrated memory/time cost | Offer original, compact, frame/FF and layer reconstruction plans. Include microbatch/accumulation and HAM transfers in the same objective; the 64 MiB experiment shows why smallest arena alone is insufficient. |
| 4 | Make variable-length and vocabulary-shift recall mandatory quality gates | Original global reached 96.09% at T1024 after variable-length training, while thinner variants lost recall. Add random key/value retrieval, multiple queries and repeated values before claiming general memory capability. |
| 5 | Admit precision safeguards as a separately named numerical contract | Test guard thresholds, absent-plane semantics, gradient/resume behavior and data-dependent overhead. Preserve FP32-input limitations in documentation. |
| 6 | Develop a language-focused local branch without silently replacing global recall | The equal blend closes most of the measured byte-loss gap but fails the long-recall gate. Compare additive/gated dual channels with controlled parameter/FLOP budgets and variable-length training. |
| 7 | Add multilinear structure only where quality/cost supports it | Q32 and product subspaces are useful ablations, not drop-in equivalents. Next test train-only mode bases, antisymmetric tensor contractions, adaptive rank or periodic dense layers; log rank selection and state conversion. |
| 8 | Compare stronger execution baselines and scale | Use IO-aware attention, compiled PyTorch/JAX, larger decoders and real accelerator hardware with matched tasks, optimizers, precision, warmup, time-to-quality and memory definitions. |

A direct skew-bilinear readout is a specific compiler opportunity:

\[
g_d=\frac{\sum_{i<j} W_{dij}(u_i v_j-u_j v_i)}{\|u\wedge v\|}.
\]

The compiler can choose packed-feature GEMM, tiled feature regeneration or a fused skew contraction. The formula alone does not make the fused contraction faster: it may give up efficient GEMM reuse or change reduction order. The new frame-reconstruction variant supplies an exact reference and a measured memory target for that next experiment.

The earlier multilinear subspace-learning work can extend this to mode-wise bases and structured readouts. Fit bases on training data only; distinguish full-rank basis changes from truncation; account for contraction intermediates and optimizer moments. MPCA resources provide a starting point for actual multilinear subspace methods [5]. This study did not implement a full MPCA/TT training stack or establish a geometry-specific advantage over all such methods.

## 11. Verification, reproducibility and remaining qualifications

* Ten main exact execution variants pass bitwise logits, loss, gradients, five-update state and resume comparisons at contexts 9/64, width 32, three layers, B2 and one/four threads. The additional combined endpoint passes the same checks. Full-width timing runs also agree in final state across all exact variants, repetitions and thread counts at their shared shapes.
* Six main architectures, the hybrid and the guarded model have independent full-model automatic-differentiation comparisons. Causality and all-masked refusal are checked for the main architectures.
* UBSan passes complete native updates for frame/FF, layer, combined-layer, low-rank and multi-subspace implementations. This is not an exhaustive sanitizer/fuzz campaign.
* All 57 new learning checkpoints are SHA-verified and restored; six reused baseline result references are also verified. Frozen source hashes and paired training-input hashes remain intact. Inference-only SVD weights are separately marked and include source-checkpoint hashes.
* Actual HAM artifacts are verified; exact-layout claims are restricted to the declared lifetime fixture. No GPU/DMA transfer, FPGA, ARM, UEFI or bare-metal test was performed.
* Repository `git diff --check` and clean-worktree checks pass. No repository source change, commit, push or PR was made.
* LLVM/Clang/MLIR 23 and Python 3.15 are absent; Node 24 is behind the repository’s requested version. The latest-toolchain matrix, full repository CI, Windows, native ARM and accelerator qualification remain explicit skips. GCC13/Python3.12 research checks do not substitute for those gates.

The accompanying bundle includes generated C, native sources/license, independent references, frozen protocols, all repetitions, profiles, verified memory artifacts, data hashes/parquets, checkpoints, SVD weights, validation outputs and reproduction instructions. `results/summary.json` and `results/final-audit.json` make the principal numerical claims inspectable.

## Sources

1. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. Primary paper: https://arxiv.org/abs/2205.14135
2. Novikov et al., *Tensorizing Neural Networks*. NeurIPS 2015 primary proceedings: https://papers.nips.cc/paper_files/paper/2015/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html
3. Williams, Waterman and Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. Berkeley primary technical report: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf
4. Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training*. Primary paper: https://arxiv.org/abs/2312.06635
5. Author-lab MPCA/UMPCA resources and publications: https://www.dsp.utoronto.ca/projects/biometrics/
6. BCIR HAM planner at the tested main revision: https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/blob/be6e377a2eb417cb4bfe3de5243ca09961ba1833/bcir/kbcir/ham.py
7. BCIR static-memory planner at the tested main revision: https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/blob/be6e377a2eb417cb4bfe3de5243ca09961ba1833/bcir/kbcir/static_memory.py

All experiment tables in this report derive from the retained local measurement files, not the cited papers’ benchmark numbers.
