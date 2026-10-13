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
- **Measured (NDT-MEASURE, 2026-10-12).** The first three slices landed and were re-measured
  together ([BCIR_NDT_MEASURE_2026-10-12.md](BCIR_NDT_MEASURE_2026-10-12.md)). The budget held for
  the elementwise kernels and was wrong for the GEMM, for the threads and for the softmax under
  `exp` (its row loop called the exponential element by element, so the attention forward was slower
  than under libm; NDT-ATTN's first item, landed the same day, puts the row through the vector
  instance: 79 to 55 ms at one thread, the digest unchanged): the
  fused contract gains x1.24 on the step's GEMMs, not 2x, because the kernel's
  packing, loads and stores are the same under both contracts; three threads give x1.71
  on the step and four x2.03, not 3x, because the 64-row tasks sit behind one barrier
  per GEMM and the reductions stay serial. On three threads the 9.84M decoder trains at 819 ms a
  step under `fused+exp` against 989 under v1, the same model to four decimals; eager PyTorch's 774
  on one thread is the 10-11 host's number. §2's measured table and §3's two candidate slices
  (NDT-BLOCK, NDT-REDUCE) carry the correction.
- **The kernel at the provider's rate (NDT-BLOCK, landed 2026-10-12).** NDT-KBLOCK's gap between
  BCIR's own v3 kernel and the admitted provider was the schedule, not the arithmetic. With the
  AVX-512 and AVX2 instances rewritten as intrinsic register blocks (8 x 48, 6 x 16) over a tile
  packed by vector, the kernel reads 99.4 GFLOP/s work-weighted at one thread on the step's shapes
  against OpenBLAS's 88.8 in the same process and 218.9 on the pool at four against its 219.1,
  bit-identical to the chain on every instance; the 9.84M decoder trains at 516 ms a step on four
  threads on BCIR's kernel alone (753 before; the admitted provider 616) and 974 on one (1,425),
  in the same state (the report's §8). What stays open is named in the row: the transposed-A
  operand's packing, and the two shapes still behind the provider.
- **The reductions on the pool (NDT-REDUCE, landed 2026-10-13).** The loss's rows, the gamma
  VJP's row sums and the embedding scatter now run as tasks in the serial order, so they thread
  under every contract without moving a bit; the update's first pass and preflight are one task
  per fixed chunk (NDT-NORM's content, landed with it); and the clipping norm has a declared
  fixed-chunk contract, the `reduce` bit, which the oracle emulates exactly. In three same-session
  pairs at four threads the update reads 40.7 / 49.7 / 47.5 ms under `kblock+exp` against
  31.8 / 47.2 / 39.6 under `kblock+exp+reduce` (-9.0 / -2.6 / -7.9; 53 in NDT-BLOCK's record on a
  faster day), the gamma VJP row 11.0 / 14.3 / 12.4 against 6.7 / 8.8 / 11.8; the step itself
  is unresolved on a host whose unchanged GEMM rows drifted 20% within the session, and
  the 1,000-step `kblock+exp+reduce` run's 625.8 ms a step on four threads (516 the day before)
  is that host's state, the same held-out model, its own state.
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

**A third contract, found by measuring (NDT-KBLOCK, landed 2026-10-12).** OpenBLAS's `sgemm` on
this host's SkylakeX target computes a K-blocked fused chain -- the depth split by its level-3
driver's rule (Q 448, unroll 16: whole 448-blocks while two remain, then half the rest rounded
up to 16, then the rest), each block's fused chain from zero added to c in order -- bit for bit,
for every one of the trainer's shapes (its small-matrix path, below about 10^6 multiply-adds,
keeps other orders). Declared as BCIR's blocked fused contract (v3) and implemented by BCIR's
own kernel on every host, it makes OpenBLAS an admissible provider: `use_cblas` admits it by a
differential over the trainer's shapes at beta 0 and 1, refuses it otherwise, and the state is
one state with the provider or without it. Under v3 the GEMM's ceiling is the provider's rate
(87-103 GFLOP/s at one thread here, 241-266 at four), with BCIR's own v3 kernel at 56-76 at one
thread; the measurement report's §7 has the step and the 1,000-step run under it.

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

**Measured (NDT-MEASURE, 2026-10-12).** The `bcir-breakdown` step on a host whose v1 one-thread
step is 1,859.9 ms (the 1,632 above was the 10-11 host's; the rows are the report's §3, the
GEMM row the forward and backward GEMMs together, the last row the step less its kernels):

| One step (ms) | v1, 1 thread | `fused+exp`, 1 thread | `fused+exp`, 3 threads | `fused+exp`, 4 threads |
| --- | ---: | ---: | ---: | ---: |
| GEMMs | 1,415.7 | 1,081.3 | 597.7 | 479.2 |
| attention | 150.9 | 167.6 | 81.7 | 71.0 |
| SiLU, SwiGLU VJP, norms, RoPE | 147.2 | 45.2 | 31.8 | 28.7 |
| AdamW and the clipping norm | 76.5 | 78.3 | 64.2 | 59.4 |
| loss, embedding, residuals (the rest) | 69.2 | 134.1 | 105.2 | 102.8 |
| **step** | **1,859.9** | **1,506.9** | **881.5** | **741.7** |
| eager PyTorch, the 10-11 host | 762 (1 thread) | -- | -- | ~257 (4 threads, the 10-11 context stage) |

Read against the expectation: the GEMM row is where it was wrong -- x1.24 from
the contract instead of 2x, and x2.26 from four threads instead of 3.6x -- the elementwise row
is where it held, and the attention row did not move at one thread (the `exp` softmax's scalar
row loop; NDT-ATTN's first item, landed the same day, takes the forward from 79 to 55 ms). The
step's three residuals each name a slice
(§3): the kernel's register block and packing (NDT-BLOCK), the task partition and the two VJPs
run concurrently (NDT-SCHED), and the serial reductions (NDT-REDUCE, with NDT-NORM for the
update's clipping pass).

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
| **NDT-EXP** (P2) | a BCIR-owned `expf` (`bcir_tensor_expf`): one operation sequence -- a Cody-Waite reduction, a degree-7 polynomial, 2^k as two exact power-of-two scalings, the special cases selected at the end -- scalar and vectorized (AVX2, AVX-512 instances), no libm, no fma; SiLU, the SwiGLU VJP and the attention softmax under a declared v2 of their contracts (the `exp` contract bit, `NativeDecoder(contract="exp")`, with `"fused+exp"` for both v2 contracts); v1 keeps libm | identical scalar, AVX2 and AVX-512 results; a stated ulp bound against exp in double rounded to binary32 over every binary32 in [-104, 89] sampled at stride 2^8 | `train.exp.mismatch` 0 over 2^24 inputs across instances; `train.exp.ulp.max` 1, the stated row, held exactly; the C harness: the instances, the special values, SiLU/SwiGLU VJP/softmax v2 against their definitions and apart from v1, the four contracts' states each their own and each identical at 1..4 threads; `ndt-exp.json`: the polynomial's highest terms dropped, the reduction's low constant dropped, an instance on libm, the v2 softmax on libm, a trainer ignoring the bit | **landed** |
| **NDT-SCHED** (P2) | **first item landed (NDT-SCHED-0, 2026-10-13):** the two VJPs of a linear (`dx`, `dW`: one adjoint read, disjoint writes) as one pool job (`bcir_tensor_mm_pair_pool`: the first GEMM's tasks then the second's, each with its own partition, so every output element keeps its depth order and the results are the two entries' bit for bit) -- one barrier in place of two for each of the step's 43 linears, and the narrow K/V projections' idle threads take the other product's tasks; a provider's GEMM runs the two one after the other as before. Then the rest: `training_step_program` emits an `ExecutionPlanV1` the C step reads -- per call the contract, the instance and the thread split chosen from the measured-candidate corpus (G13) | the plan that runs is the plan that was verified (G19's law extended to the trainer); R9 prices it; the pair job's results the two entries' bit for bit (the C harness holds every shape pair under the three GEMM contracts, both orders, and the twelve contracts' states at one to four threads through the trainer's own pairs) | `train.plan.executed.mismatch` 0 (the trace of the C step equals the plan); the step's time against the plan's roofline; RED: a plan naming a variant the host lacks is refused before any kernel runs; `ndt-threads.json` NT9-NT10 (a pair split at the first GEMM's column count, the pair's serial path skips the first GEMM) | first item **landed**: measured in 3 same-session pairs at four threads under `kblock+exp+reduce` -- the backward pass 631.9 / 633.8 / 622.4 -> 601.0 / 586.3 / 594.6 ms (-30.9 / -47.5 / -27.8), the GEMM-backward row timed alone 292.7 / 285.9 / 266.6 -> 238.0 / 254.8 / 282.6 (-54.6 / -31.0 / +16.0), the step 684.2 / 679.4 / 659.3 -> 645.1 / 681.7 / 668.1 (-39.2 / +2.4 / +8.8); the backward pass (the `loss_backward` stage, the forward recomputed inside it, as the trainer runs it -- the one stage the pair job changes) is below the two-call stage in 3 of 3 pairs, by -30.9 / -47.5 / -27.8 ms (631.9 / 633.8 / 622.4 against 601.0 / 586.3 / 594.6), and the GEMM-backward row timed alone by -54.6 / -31.0 / +16.0; the step's own cell (-39.2 / +2.4 / +8.8) is within this host's run-to-run noise, as NDT-REDUCE's was -- the one barrier fewer per linear and the narrow projections' tasks shared are worth about 5% of the backward at four threads here; the rest open |
| **NDT-ATTN** (P2) | **first item landed (2026-10-12):** the v2 softmax's row through the vector exponential after the max-subtraction (NDT-MEASURE's finding: the scalar row loop cost the attention forward a quarter more than libm; the same values, so the `exp` digest stays -- 79 to 55 ms at one thread, [attn0/](measurements/2026-10-12/attn0/)); then tiled attention: the forward keeps one running maximum and log-sum-exp per row and stores no probabilities; the backward recomputes each tile's probabilities from the stored log-sum-exp; fixed tile order, so deterministic; a declared v2 of the attention contract | O(B·H·T) memory in place of O(B·H·T²); at T=1,024 a layer within PyTorch's flash path | `train.attention.tiled.mismatch` against the scalar reference under the v2 contract; the `context` stage's per-layer times and arena bytes at T=64..1,024 | open |
| **NDT-NORM** (P3) | the update's first pass on the pool where the order allows: the three state streams' maxima and finiteness (associative) as pool tasks, the gradient norm's index-order chain kept serial -- measured before building ([update-passes.json](measurements/2026-10-12/update-passes.json), 2026-10-12): the four-thread update is 53 ms, 41 of them the first pass (16 the gradient's chain, 25 the weight and moment streams), 12 the pooled commit (33 serial); the pooled first pass would take it to about 34 ms, under 4% of the four-thread step with the admitted provider | the same state (the chain, the maxima and the verdicts are the same values) | the `bcir-breakdown` stage's update row: 53 -> ~34 ms at four threads | **landed with NDT-REDUCE (2026-10-13):** the first pass and the preflight are one task per fixed chunk under every contract, the chain serial where the contract keeps it -- the update row 47.5 ms at four threads under `kblock+exp` in the same session as the reduce pair (§9 of the report; 53.1 in §8's record on a faster day) |
| **NDT-MEM** (P2) | the transient arena laid out by lifetime (the GEM+ roadmap's open item) | `train.memory.transient` 1.326 -> 1.0 | as stated there | open |
| **NDT-MEASURE** (P1) | the comparison re-run on the pool and the contracts ([BCIR_NDT_MEASURE_2026-10-12.md](BCIR_NDT_MEASURE_2026-10-12.md)): `identity` per contract and thread count, `bcir-breakdown` at 1, 3 and 4 threads under each contract, the 1,000-step language-model run on three and four threads under v1 and `fused+exp` | every state its contract's own and identical at 1..4 threads; the 1,000-step v1 run on three threads ends in the 10-11 one-thread run's state (`aed5ffbf…`) | the exact rows; the step read against §2 row by row (§2's measured table) -- the fused contract gains 1.2x on the step's GEMMs, not 2x, and three threads 1.7x on the step, not 3x, which names the next slices: the register block and packing (NDT-BLOCK), the partition and the concurrent VJPs (NDT-SCHED), the serial reductions (NDT-REDUCE) | **landed** |
| **NDT-KBLOCK** (P1) | the blocked fused GEMM contract, v3: the depth split by one rule (whole 448-blocks while two remain, then half the rest rounded up to a multiple of 16, then the rest), each block's fused chain from zero added to c as c + alpha*P; BCIR's own kernel on every host (the partial in registers across a whole block, the fused instances' widths); the state's `kblock` bit (`NativeDecoder(contract="kblock" | "kblock+exp")`, one GEMM contract at a time); a CBLAS provider admitted under v3 only by a differential over the trainer's own GEMM shapes at beta 0 and 1 (`use_cblas`), refused otherwise | the same state with the provider or without it; OpenBLAS's `sgemm` is the contract bit for bit on its SkylakeX target for the trainer's shapes (its small-matrix path is not, and is refused where a trainer's shapes are small) | `train.mm.kblock.mismatch` 0 of 40 against the exact emulation, with the split rule on both rails; `train.mm.provider.unsound` 0 wherever the wheel is importable -- the verdict is the differential's on a host that admits (the Intel host, 0 of 30 differences) and on one that refuses (PR #817's AMD EPYC host, 14 of 30; `train.mm.provider.differences` records the host's count, reported, never graded off the baseline host); the C harness holds every instance to the K-blocked fmaf chains, serially and on a pool, equal to v2 exactly where the contract says it is; the six contracts' states each their own and each identical at 1..4 threads; `ndt-kblock.json`: the rule without its halving, a chain from c, a fused add, a trainer or a one-thread pool entry on the fused kernel, an admission that does not read its differential; measured: the 1,000-step `kblock+exp` run ends in one state (`d7fc618c…`) on BCIR's own kernel and on OpenBLAS admitted, at one and at four threads; 655 ms a step with the provider on four threads (753 on BCIR's kernel), 1,066 on one (1,425); BCIR's own v3 kernel faster than v2 at one thread (the step's GEMMs 1,081 to 952 ms); with the provider's workers sleeping between calls (`OPENBLAS_THREAD_TIMEOUT=4`) 616 ms a step on four threads (the step by kernel 527) -- the report's §7 | **landed** |
| **NDT-BLOCK** (P1) | the v3 kernel's register block and packing under the same contract (the result is fixed by the contract, so only the schedule changed): the AVX-512 and AVX2 instances written in intrinsics -- 8 x 48 and 6 x 16 accumulators, every A term one broadcast from memory, the tile three (two) vector loads per depth, so eleven loads per 24 fused multiply-adds and a third of the old 4 x 64 block's tile traffic from L2 per fma; the tile packed by vector (rows of B by masked loads, columns of B^T by 16 x 16 register transposes); the transposed-A operand's next line prefetched eight depths ahead; the pool's tasks 128 rows by one column block of the instance's width (the tile packed half as often); the generic instance keeps the plain-C block. The per-panel packing of the transposed-A operand stays open: the serial path has no buffer for it under the unit's no-allocation law, and the two shapes it would help most stay behind the provider in the record: `dW` with 1,024 rows and depth 1,024, whose A lines alias one cache set, reads 73.6 GFLOP/s against OpenBLAS's 89.6 (31.2 before the prefetch), and the depth-192 product 67.4 against 91.1, where one short K block leaves the per-call costs uncovered | `train.mm.kblock.mismatch` stays 0, now over every instance the host runs (the AVX-512 and AVX2 instances are their own code); every state digest stays | **landed (2026-10-12):** the kernel-rate record ([block/gemm-rates-block.json](measurements/2026-10-12/block/gemm-rates-block.json), best of 7, the step's 15 shapes, GCC 13.3): work-weighted 61.2 -> 99.4 GFLOP/s at one thread against OpenBLAS's 88.8 in the same process, and 135.9 -> 218.9 on the pool at four against OpenBLAS's 219.1 at four with its workers sleeping -- BCIR's own kernel at or past the admitted provider's rate on this host; the `identity` stage's six contract/thread rows on the widest, AVX2 and portable builds end in the kblock states of §7 ([block/identity-block.json](measurements/2026-10-12/block/identity-block.json)); the `bcir-breakdown` GEMM rows 569 ms at one thread and 290 at four (952 and 531 before); the 1,000-step `kblock+exp` run on BCIR's kernel ends in `d7fc618c…` at 516 ms a step on four threads (753 before) and 974 on one (1,425) -- the report's §8; `ndt-kblock.json` NK7-NK10 (the AVX-512 instance's last row, a task's column block narrower than its register block, a transposed tile's ragged columns, a ragged block's fused epilogue) |
| **NDT-REDUCE** (P3) | the index-order reductions the pool left serial: the loss's rows, the gamma VJP's row sums and the embedding scatter as tasks in the serial order (rows whose terms the caller adds in row order; column ranges whose sums run over the rows in row order), bit-identical under every contract; the update's first pass and preflight one task per chunk of `bcir_decoder_reduce_chunk` elements (1,024 doubled until at most 1,024 chunks), verdicts and maxima exact; the clipping norm under a declared contract -- the `reduce` bit: each chunk's squares summed in index order in double from zero, the chunk sums added in chunk order from zero, a shape that is a function of the parameter count alone (`NativeDecoder(contract="<any>+reduce")`, `"reduce"` for v1; the checkpoint names it); the oracle emulates both norms operation for operation and `bcir_decoder_gradient_norm` reports the square the update takes its root of | the loss and the state the serial ones at every thread count under every contract; the reduce contract its own norm, identical at every thread count, emulated exactly -- its state coincides with the chain's while the float rounding of every update absorbs the norm's last bits and is its own from the first rounding boundary crossed (within twenty steps on the 9.84M decoder) | `train.norm.reduce.mismatch` 0 of 6 norms and 8 chunk counts; `train.state.threads.mismatch` 0 under v1 and under `reduce`; the C harness's `reduction_references` (the pooled loss and scatter against serial references written apart from the pool), `pool_identity` over twelve contracts with the losses and norms the steps report, `adamw_reference` under both norm contracts on a three-chunk fixture; `ndt-reduce.json` NR1-NR11 (a chunk following the thread count, sums added from the last chunk, the bit ignored, the loss's terms or the scatter's rows walked from the last, the gamma adjoint's rows reversed or its last column chunk dropped, one chunk's verdict read, the loss's ragged tail dropped, a rule that never doubles, a chunk summed from its last element), 11 of 11 caught | **landed (2026-10-13):** measured ([measurements/2026-10-13/reduce/](measurements/2026-10-13/reduce/), the report's §9): in three same-session pairs at four threads (each contract's breakdown alternated with the other's, minutes apart, so both see one host) the update reads 40.7 / 49.7 / 47.5 ms under `kblock+exp` -- the first pass pooled, the chain serial, NDT-NORM's content -- against 31.8 / 47.2 / 39.6 under `kblock+exp+reduce`, below it in every pair (-9.0 / -2.6 / -7.9), with §8's 53.1 the day before on a faster host; the gamma VJP row 11.0 / 14.3 / 12.4 against 6.7 / 8.8 / 11.8 (-4.3 / -5.5 / -0.6; 9.2 in §8); the step itself is not resolved on this day's host, whose unchanged GEMM rows drifted 20% within the session, more than the slice's effect on the step; the 1,000-step `kblock+exp+reduce` run at 625.8 ms a step on four threads (1419.6 on one), ending in `5137e013…`, its own, not §7's `d7fc618c…` (the norm's last bits crossed a float rounding boundary within the first steps, as the 20-step identity rows already show, and the state is the reduce contract's from there), and the one-thread run ends in the same state, the held-out model 1.5003 / 1.6319 |
| **NDT-BF16 / AMX** (P3) | BF16 computation with an explicit accumulation and master-weight policy, AMX tiles where the host has them | quality parity across seeds | hardware-gated: this host has neither `avx512_bf16` nor AMX; an explicit skip, never simulated | hardware-gated |

Order of work (TRAIN-0, TRAIN-RAIL, NDT-THREADS, NDT-FMA, NDT-EXP and NDT-MEASURE landed 2026-10-12): TRAIN-0 -> TRAIN-RAIL -> NDT-THREADS -> NDT-FMA -> NDT-EXP -> NDT-MEASURE (the
comparison re-run) -> NDT-KBLOCK (the provider contract the measurement found) -> NDT-BLOCK (landed 2026-10-12: the kernel at the provider's rate) -> NDT-REDUCE (landed 2026-10-13: the
reductions on the pool, NDT-NORM with it) -> NDT-SCHED (its first item, the two VJPs as one job, landed
2026-10-13) -> NDT-ATTN, with
NDT-MEASURE's reading (§2's measured table) setting SCHED's and BLOCK's targets. THREADS precedes FMA because it changes no
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
| state | `train_comparison.py identity --steps 20 --cc <each> --threads 1,2,3,4 --contract <each>` (`--variants` names the builds) | the 20-step state SHA-256 across compilers, instances, thread counts and contracts | one SHA per contract: **exact** |
| step | `train_comparison.py breakdown` and `baseline` (BCIR against eager PyTorch); `bcir-breakdown --threads <n> --contract <c>` (the step by kernel through the entry points the trainer calls under the run's pool and contracts, no PyTorch needed), at 1, 3 and 4 threads | the step part by part; the step before against after | **wall**, indicative |
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
