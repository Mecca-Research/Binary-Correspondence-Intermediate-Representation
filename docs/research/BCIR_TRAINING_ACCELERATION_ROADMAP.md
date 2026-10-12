# Training acceleration: the ladder from behind PyTorch to ahead of it

> Dated 2026-10-11, non-normative, the plan of record for the native trainer's next slices. It
> follows the [2026-10-10 assessment](BCIR_GEMPLUS_ASN1_ML_PERFORMANCE_2026-10-10.md) and the
> [2026-10-11 measurements](BCIR_NATIVE_TRAINING_PERFORMANCE_2026-10-11.md), reads their records
> under [`measurements/`](measurements/), and states what each remaining millisecond is made of,
> which contract bounds it, and the slice that moves it. The GEM+ roadmap's measurement discipline
> ([§0](BCIR_GEMPLUS_ROADMAP.md#0-the-measurement-discipline-and-why-it-comes-first)) applies to
> every number here: `exact` rows gate, `ratio` rows carry a band, `wall` rows are indicative, and a
> before/after is one host, one session, interleaved. Counts live in the generated
> [`STATUS.md`](../STATUS.md).

## 0. The answer

- **The gap is one contract and one thread.** At the 9.84M decoder's training shape (B=8, T=128),
  BCIR's own kernels train a step in 1,632 ms against eager PyTorch's 762 on one thread, and PyTorch
  trains it in 257 ms on this class of VM's four cores while BCIR's kernels run one (§1). Of the
  870 ms between the one-thread steps, 853 are the GEMM, and the GEMM is bound by its contract: one
  rounded multiply and one rounded add per term issues two vector instructions where MKL's fused
  multiply-add issues one, so the kernel's ceiling is half of MKL's and it already runs at 73% of
  its own (§2). Everything else in the step is faster on BCIR than on PyTorch.
- **Two declared changes lift both bounds without giving up determinism.** A second GEMM contract,
  `fused` (one IEEE fused multiply-add per term, the same ascending-`k` order), is bit-identical on
  every host with an IEEE `fma` and doubles the ceiling (NDT-FMA). A thread pool that partitions
  every kernel so that each output element is still produced by one thread in the same operation
  order makes the training state independent of the thread count, bit for bit, under either
  contract (NDT-THREADS). Neither is an approximation; each is a witness.
- **The arithmetic reaches eager PyTorch at four threads and lands within 15% of it on one.** §2's
  budget puts the step at about 880 ms on one thread after the fused contract and a BCIR-owned
  exponential (PyTorch: 762), and at 255-285 ms on four (PyTorch: 257). Beating the four-thread
  step is the last 10%: a micro-kernel at MKL's efficiency, the two VJPs of a layer run
  concurrently, and the plan choosing per call. Those are slices with rows, not promises (§3).
- **The training corpus is a one-way build dependency of the IR (TRAIN-RAIL, landed).** The IR's
  build has a `train` rail that reads `training/**/*.md` and `docs/**/*.md` as the pinned corpus
  of the BCIR-native lab model and trains it on the native trainer (`tools/models/train_rail.sh`:
  CTest's `train-corpus`, BCIR Make's `train.corpus`, the qualification job); nothing under
  `bcir/`, `mlir/` or `runtime/` imports or compiles anything from `training/`, and `training/`
  keeps its own gates (§5). The optimizations below then reach model training by construction, because the model the
  build trains is trained by the kernels the build gates.
- **What stays declared.** No GPU, no PMU, no native-ARM timing, no BF16 or AMX on this host (its
  CPU has neither), one seed per backend. A clean skip is reported as a skip (§6).

## 1. Where a step's time goes

The `breakdown` stage of [`train_comparison.py`](measurements/2026-10-11/train_comparison.py)
times one step at the comparison's shape part by part (medians of five rounds, one thread, GCC
13, the kernel as it ships; [`breakdown.json`](measurements/2026-10-11/breakdown.json)). The
PyTorch column is `torch.profiler`'s self CPU time by operator family. These are the 2026-10-11
report's numbers on its host (a four-core Intel Xeon VM at 2.1 GHz, AVX-512); NDT-MEASURE re-takes
every row on the host of each slice's before/after, and no number crosses hosts.

| One step, B=8, T=128, 9,836,928 parameters (ms) | BCIR, own kernels | BCIR + OpenBLAS | eager PyTorch |
| --- | ---: | ---: | ---: |
| GEMMs, forward (16 shapes, 20.1 GFLOP) | 429 | 184 | 476 for both directions (`aten::mm`, 60.4 GFLOP) |
| GEMMs, backward (`dx` and `dW`, 40.3 GFLOP) | 900 | 382 | |
| attention forward, six layers | 51 | 51 | 21 (flash SDPA) |
| attention backward | 90 | 90 | 39 |
| SiLU forward (libm `expf` per element) | 48 | 49 | 174 for every other operator of the forward and backward |
| SwiGLU VJP (`expf` again) | 54 | 43 | |
| RMSNorm forward + VJP, RoPE forward + inverse | 18 | 19 | |
| AdamW with the clipping norm | 55 | 69 | 55 (AdamW 46, clipping 9) |
| **one step** | **1,632** | **885** | **762** |

Three facts in that table carry the plan:

1. **The GEMM is 81% of BCIR's step and 853 of the 870 ms it trails by.** Its rate is 45 GFLOP/s
   over the step's sixteen shapes (48.8 timed alone); MKL's is 127. OpenBLAS behind the CBLAS seam
   runs the same shapes at 107 GFLOP/s, and with it the step is 885 ms: the rest of BCIR's step is
   already faster than PyTorch's (319 ms against 286 for everything but the GEMM, of which 81 is
   attention's deficit and the rest BCIR's lead).
2. **The exponential is 102 ms.** SiLU and the SwiGLU VJP call libm's `expf` once per element
   (8 x 128 x 1,024 x 6 layers, twice), and the attention forward's softmax calls it 3.2 million
   times a step (B·H·T(T+1)/2 per layer). Both kernels are otherwise a handful of multiplies.
3. **One thread.** On the four cores of that VM, eager PyTorch's step is 257 ms (2.9x its one-thread
   step) and BCIR with a four-thread OpenBLAS 542 ms, because only its linears were threaded. BCIR's
   own kernels stay at 1,569 ms there ([`context.json`](measurements/2026-10-11/context.json)).

And at longer contexts the attention's stored probabilities grow as B·H·T²: one layer goes from
faster than PyTorch's flash kernel at T=64 to 1.87x slower at T=1,024, and the arena from 323 to
503 MiB (the report's §5.8). That is a separate slice (NDT-ATTN) because it moves nothing at the
comparison's T=128.

## 2. The ceilings, and the budget they leave

**The GEMM under each contract.** The report's host runs AVX-512 at 2.1 GHz with two FMA ports.
With fused multiply-adds its single-core peak is 2 x 16 lanes x 2 flops = 64 flops per cycle, 134
GFLOP/s; MKL's 127 is 95% of it. Without fusion each term costs a multiply and an add on those same
two ports, 32 flops per cycle, 67 GFLOP/s; BCIR's 48.8 is 73% of that. So under today's contract a
perfect kernel gains at most 1.37x, and no kernel under it reaches MKL. Under the fused contract the
ceiling is MKL's; at the kernel's present efficiency the step's GEMMs would take about 670 ms
instead of 1,329, and at MKL's efficiency about 520.

Why the fused contract keeps the guarantee that matters: IEEE 754 defines `fma(a, b, c)` as the
exactly rounded `a*b + c`, so a kernel that evaluates `c = fma(alpha*a, b, c)` for `l` ascending
performs one specified rounding per term. Every host with a hardware FMA (x86-64 FMA3, every
AArch64, POWER) and every libm whose `fmaf` is correctly rounded (glibc, musl, the UCRT) produces
the same bits. What changes is the result against today's contract, by one rounding per term; the
two are two contracts, not an approximation of one, and the trainer records which one trained a
state. The Python oracle emulates the fused contract exactly (an exact rational product and sum,
rounded once to binary32), as it emulates today's.

**Threads.** Four cores give at most 4x to compute-bound kernels and less to memory-bound ones.
The GEMM, attention and SiLU are compute-bound at this shape; the AdamW passes and the norm pass
stream 157-197 MB each and share one memory system. PyTorch's 2.9x on the whole step is the
reference point for what the VM's cores deliver.

**The exponential.** A deterministic `expf` -- a Cody-Waite range reduction and a fixed-degree
polynomial, every operation rounded, the same operation sequence in scalar and in every vector
width -- costs 10-14 flops per element against libm's branchy 40-60 cycles, and removes the one
place where "bit-identical on every host" currently depends on which libm the host links. Its
error against the correctly rounded value is stated as a bound, not hidden: it is a declared
second contract of the SiLU, SwiGLU and softmax kernels, as the fused contract is of the GEMM.

**The budget.** Composing the levers at the ceilings above, before any of them is built:

| One step (ms) | today, v1, 1 thread | fused + exp, 1 thread | the same, 4 threads | at MKL's GEMM efficiency, 4 threads |
| --- | ---: | ---: | ---: | ---: |
| GEMMs | 1,329 | ~670 | ~185 | ~145 |
| attention | 141 | ~110 | ~32 | ~32 |
| SiLU, SwiGLU VJP, norms, RoPE | 120 | ~30 | ~12 | ~12 |
| AdamW and the clipping norm | 55 | 55 | ~30 | ~30 |
| loss, embedding, residuals | ~10 | ~10 | ~10 | ~10 |
| **step** | **1,632** | **~875** | **~270** | **~230** |
| eager PyTorch, same threads | 762 | 762 | 257 | 257 |

The arithmetic says: parity within 15% on one thread, parity at four, and ahead of eager PyTorch
at four only with the GEMM at MKL's efficiency or with the structural levers of §3 (concurrent
VJPs, the plan choosing per call, one pass fewer in the update). It is an expectation, and each
slice replaces one row of it with a measurement before the next slice is designed.

## 3. The ladder

One gated slice per entry, the slice ID in the PR title, every number in a PR body with its host,
compiler, contract and thread count, and the before/after audit of the GEM+ roadmap's §0.4 in every
PR. P0 is the integration the rest flows through, P1 is the two bounds, P2 widens, P3 needs
hardware.

| Slice | Content | What it holds | Gate | Status |
|---|---|---|---|---|
| **TRAIN-0** (P0) | this ladder; the one-way corpus law (§5) in `AGENTS.md`, the structure doc and `training/README.md` | -- | docs governance | this PR |
| **TRAIN-RAIL** (P0) | `training/` as the IR's data, one way: the corpus recipe `tools/models/configs/bcir-corpus-860k.json` (`docs/**/*.md` with `training/**/*.md`; the docs recipe's model, tokenizer and licence, its own split salt) and the rail `tools/models/train_rail.sh` -- CTest's `train-corpus` (label `train`, `BCIR_TRAIN_STEPS`, 40 by default), BCIR Make's task `train.corpus` (the same script and count through `run_section.sh`, reading the recipe, the trainer's two C units, `bcir/`, `docs/` and `training/`, so a corpus edit retrains and a runtime edit elsewhere does not), and the qualification job (200 steps, then qualified as the docs model was) | the IR's libraries and tools build without the corpus: the task is the one reader of the trees; a recipe that selects nothing is refused before any work; only tracked files are read, each under the recipe's licence; a dirty corpus tree is recorded in the card (`corpus_tree_dirty`) beside every file's bytes and digest | `train.corpus.unselected` 328 -> 0 (the tracked Markdown files of both roots the recipe does not select, read without a training run; the docs recipe reads 328); the witness exits 1 on the docs recipe and 2 on a manifest it cannot read; the script exits 2 before any work on what it cannot run; one script, recipe, root pair and step count in every place the rail lives (`test_train_rail.py`) | **landed** |
| **NDT-THREADS** (P1) | a caller-owned deterministic thread pool (`bcir_tensor_pool`: pthreads on POSIX, Win32 threads on Windows, serial elsewhere); the GEMM partitioned over (column block, row range) so every output element keeps its depth order; attention over (batch, KV head) with the query heads of one KV head in order, so the shared `dk`/`dv` accumulate as the reference does; SiLU, the VJPs, RMSNorm and the AdamW commit over elements and rows; the gradient-norm pass serial (index order, double); ABI v2: the state names its pool | **the training state is identical at every thread count and to today's single-thread state**, under v1 and under v2 | `train.state.threads.mismatch` 0 (three steps of the threads bench -- vocab 250, width 72, 4 heads over 2 KV heads, 2 layers, batch 2 x 40 tokens -- at 2, 3 and 4 threads against one; `tools/testing/faults/ndt-threads.json`: a GEMM row partition that stops before the ragged tail and a runner that drops each thread's last task turn it red); the `identity` stage's 20-step SHA-256 at 1, 2, 3 and 4 threads is `6c0419e85d345904...`, the report's serial digest, on every instruction-set variant and a repeat -- 16 rows, one state ([identity-threads.json](measurements/2026-10-12/identity-threads.json), 2026-10-12, GCC 13.3); the C harness holds every pool kernel and three training steps at threads 1..4 to the serial kernels and the scalar references; the runtime gate's `decoder_train_tsan` section holds the partitions race-free under ThreadSanitizer and fires on one KV head's query heads over two threads (the manifest's `test_decoder_train_race` mutation) | **landed** |
| **NDT-FMA** (P1) | the fused contract: `bcir_tensor_mm_fused` (`c = beta*c`, then `c = fma(alpha*a(i,l), b(l,j), c)` for `l` ascending), a portable instance through `fmaf`, AVX2+FMA and AVX-512 instances fusing in registers (the same register block and packed tile as v1; the wider block and the per-panel packing of the transposed-A operand stay open, measured below); the state selects it (ABI 3 `contract`, `NativeDecoder(contract="fused")`); the checkpoint and every record name the contract | bit-identical across hosts and compilers with an IEEE `fma`; the oracle's exact emulation | `train.mm.fused.mismatch` 0 over the 20 GEMM cases against an exact rational emulation rounded once per term (`round_f32`, never through a double: a tie resolved by a double rounds wrongly); the C harness holds every fused instance to the `fmaf` ascending-`k` loop, serially and on a pool, and the fused state apart from v1's; `ndt-fma.json`: a term rounded twice, a fused entry running v1, a trainer ignoring its contract, an undefined bit admitted; measured: v2 against v1 at one thread on the 9.84M decoder's 13 GEMM shapes, best of 7: work-weighted 46.9 -> 66.9 GFLOP/s, x1.43 (x1.33 to x1.51 per shape; [gemm-fused-rates.json](measurements/2026-10-12/gemm-fused-rates.json), GCC 13.3, AVX-512 instance, indicative) -- short of the arithmetic's x2 because the kernel is not only its multiply-adds (packing, loads and the register block's stores stay), which the wider block and the per-panel packing above are for | **landed** |
| **NDT-EXP** (P2) | a BCIR-owned `expf`: one operation sequence, scalar and vectorized, no libm; SiLU, the SwiGLU VJP and the attention softmax under a declared v2 of their contracts; v1 keeps libm | identical scalar, AVX2 and AVX-512 results; a stated ulp bound against the correctly rounded value over the whole binary32 range | `train.exp.mismatch` 0 over 2^24 inputs across instances; `train.exp.ulp.max` as a stated row; RED: a polynomial term dropped, a reduction constant off | open |
| **NDT-SCHED** (P2) | the GEM+ step program drives the step: `training_step_program` emits an `ExecutionPlanV1` the C step reads -- per call the contract, the instance and the thread split chosen from the measured-candidate corpus (G13), and the two VJPs of a linear (`dx`, `dW`: independent reads, disjoint writes) scheduled concurrently where the narrow K/V projections leave cores idle | the plan that runs is the plan that was verified (G19's law extended to the trainer); R9 prices it | `train.plan.executed.mismatch` 0 (the trace of the C step equals the plan); the step's time against the plan's roofline; RED: a plan naming a variant the host lacks is refused before any kernel runs | open |
| **NDT-ATTN** (P2) | tiled attention: the forward keeps one running maximum and log-sum-exp per row and stores no probabilities; the backward recomputes each tile's probabilities from the stored log-sum-exp; fixed tile order, so deterministic; a declared v2 of the attention contract | O(B·H·T) memory in place of O(B·H·T²); at T=1,024 a layer within PyTorch's flash path | `train.attention.tiled.mismatch` against the scalar reference under the v2 contract; the `context` stage's per-layer times and arena bytes at T=64..1,024 | open |
| **NDT-NORM** (P3) | one pass fewer in the update where the order allows: the finiteness check and the maxima gathered as the backward writes each gradient, the norm's index-order sum kept | the same state | the `breakdown` stage's optimizer row | candidate |
| **NDT-MEM** (P2) | the transient arena laid out by lifetime (the GEM+ roadmap's open item) | `train.memory.transient` 1.326 -> 1.0 | as stated there | open |
| **NDT-BF16 / AMX** (P3) | BF16 computation with an explicit accumulation and master-weight policy, AMX tiles where the host has them | quality parity across seeds | hardware-gated: this host has neither `avx512_bf16` nor AMX; an explicit skip, never simulated | hardware-gated |

Order of work (TRAIN-0, TRAIN-RAIL, NDT-THREADS and NDT-FMA landed 2026-10-12): TRAIN-0 -> TRAIN-RAIL -> NDT-THREADS -> NDT-FMA -> NDT-EXP -> NDT-MEASURE (the
comparison re-run) -> NDT-SCHED -> NDT-ATTN. THREADS precedes FMA because it changes no
arithmetic, so its witness is the strongest kind (the existing state digest reproduced at every
thread count) and it lands on the contract every existing gate already holds. FMA is then a
provider choice on a kernel that is already partitioned. EXP is independent of both and small.
SCHED needs the per-call choices that THREADS and FMA create.

### 3.1 What each slice may not do

- Change today's contract. v1 stays the default of every existing gate, row and model card; v2 is
  selected, named and digested. A state trained under v2 is not compared to a v1 digest.
- Thread a reduction without a fixed partition. A sum whose grouping depends on the thread count
  is a different number at each count; the ladder admits only partitions in which each output
  element, or each reduction, is produced by one thread in the serial order.
- Call libm from a kernel whose contract says BCIR owns the arithmetic (v2 of SiLU/softmax).
- Report a vendor BLAS number as BCIR's. The CBLAS seam stays, measured and labeled, as the
  comparison's `bcir-openblas` backend is.
- Claim a per-step speedup without the same model: every comparison trains from the same weights,
  batches and rates, and reports the final test cross-entropy beside the step time.

## 4. The protocol: test, evaluate, update, retest, reevaluate

Five instruments, each already in the tree, each run on one host in one session with the sides
interleaved; a slice is measured with all of them before and after, and its PR quotes the outputs.

| Level | Instrument | What it reads | Gates? |
|---|---|---|---|
| kernel | `train_comparison.py kernels --base <main>` | every GEMM shape of the step (16), both attention directions, AdamW, per instance; output hashes before against after | hashes equal under one contract: **exact** |
| state | `train_comparison.py identity --steps 20 --cc <each>` (NDT-THREADS adds `--threads`) | the 20-step state SHA-256 across compilers, instances and thread counts | one SHA per contract: **exact** |
| step | `train_comparison.py breakdown` and `baseline`, at 1, 2 and 4 threads | the step part by part, BCIR against eager PyTorch; the step before against after | **wall**, indicative |
| run | `train_comparison.py train --task lm --backend <each> --threads <n>` | 1,000 steps: time to each validation cross-entropy, final test cross-entropy, accuracy and words completed, peak RSS, the final state's SHA-256 | the model equal to PyTorch's (CE within 0.001 nats): **exact** in intent; the times indicative |
| row | `tools/perf/gemplus_baseline.py --compare --group train`, `tools/perf/ab_audit.py` | `train.step.ms` and its roofline floor; the new rows above; the before/after audit of every PR | the exact rows gate; the ratio rows within their bands |

Rules the protocol keeps:

- A number without its host, compiler, contract and thread count is not a result.
- The PyTorch side runs at the same thread count as BCIR's, from the same weights and batches, and
  every run's record carries the BLAS it linked (MKL on the report's host) and the final state's
  digest.
- No row from this ladder becomes a CI threshold. CI holds the exact rows (mismatch counts,
  digests); the wall rows are read by a person.
- Honest skips are named: PMU counters, energy, GPU, native-ARM timing (CI's AArch64 job runs the
  identity gates, not a benchmark), BF16/AMX.

## 5. The one-way corpus law

Until this ladder, `training/` was "never a build dependency of BCIR" and "neither may depend on
the other" (`AGENTS.md` §2, `docs/BCIR_Repo_Structure.md`, `training/README.md`). The law
protected two things: the IR's semantics must not need corpus edits, and the corpus must stay
usable without the IR. Both survive the amendment, which is one-way and data-only:

1. **The IR's `train` rail reads `training/` as data.** The lab-model recipe selects
   `training/**/*.md` beside `docs/**/*.md` through the same `hosted.training.data` edge the corpus
   standard already builds (one tracked file, one `RawDocument`, one licence, one digest). The rail
   is a CTest label (`train`), a BCIR Make task and the qualification job; `cmake --build` of the
   IR's libraries and tools needs nothing from `training/`.
2. **Nothing under `bcir/`, `mlir/` or `runtime/` imports or compiles anything from `training/`.**
   The oracle stays dependency-free; the corpus's Python stays its own.
3. **`training/` never imports the IR except through the lazy edges its standard names** (the
   hosted stages its `verify_ml_components.py` exercises, exactly as today).
4. **A change to the corpus changes the model the rail trains, and the model card says so**
   (`corpus_tree_dirty`, the per-file digests). A change to the IR's semantics still must not
   require a corpus change.

The amendment is in `AGENTS.md` §2, the structure doc's table and the corpus README's scope note;
`CONTRIBUTING.md` and `training/ROADMAP.md` restate it.

## 6. What this ladder will not claim

- No speedup from a model trained differently. Every comparison is the same model: the same
  weights, batches and rates on every backend, the final test cross-entropy beside the time.
- No "bit-identical" across contracts. v1 and v2 are two contracts; identity is claimed within
  one, across hosts, compilers, instances and thread counts, and is a witness each time.
- No optimality. The GEMM's rate is compared to measured MKL and OpenBLAS and to the arithmetic
  ceiling of its contract on this host, never certified (TMSAO-4, as everything the oracle emits).
- No GPU, PMU, energy, native-ARM timing, BF16 or AMX result from this host.
- No step time as a CI threshold.
