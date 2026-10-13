# Native C decoder training

BCIR now has an opt-in, complete FP32 Llama/SwiGLU decoder trainer. The portable
execution core is C11, also tested as C23. Python prepares token batches, initializes
or imports weights, and saves checkpoints; the multi-step forward/backward/AdamW loop
executes in C. PyTorch remains an independent optional reference.

## Implemented contract

| Component | Native implementation |
|---|---|
| Tensor linear algebra | Register-blocked row-major GEMM: a KCxNR tile of B packed once per (column, depth) block into cache-line-aligned storage and reused by every row, an MRxNR block of C kept in registers across the tile's depth; bit-identical to the ascending-k loop on every input. Portable C everywhere, plus AVX2 and AVX-512 variants on x86-64 Linux (GCC or Clang) chosen per call from the CPU, all bit-identical (`bcir_tensor_mm_variant` runs one). Transpose variants, linear forward, input and accumulated weight VJPs; optional explicit LP64 CBLAS adapter. Two contracts (NDT-FMA): v1, one rounded multiply and one rounded add per term, the default of every gate, row and record; and the fused v2, one IEEE `fmaf` per term for k ascending (a portable `fmaf` instance, AVX2+FMA and AVX-512 instances fusing in registers, all bit-identical to the `fmaf` loop), selected by the state's `contract` (ABI 3) and named in the checkpoint. Three GEMM contracts, selected by the state: v1 (a rounded multiply and a rounded add per term), the fused v2 (one `fmaf` per term, NDT-FMA) and the blocked fused v3 (NDT-KBLOCK: the depth split by one rule into blocks of up to 448, each block's fused chain from zero added to c in order -- the arithmetic OpenBLAS's `sgemm` performs on its SkylakeX target, so that provider is admitted under v3 by a differential over the trainer's own shapes, and the state is one state with it or without it) |
| Vector math | Stable sigmoid/SiLU, SwiGLU VJP, RMSNorm and gamma VJP, half-split RoPE and its transpose; standard libm without fast-math. Under the `exp` contract (NDT-EXP) SiLU, the SwiGLU VJP and the attention softmax take their exponential from BCIR's own `bcir_tensor_expf` -- one operation sequence on the scalar, AVX2 and AVX-512 instances, no libm, no fma, within 1 ulp of exp in double rounded to binary32 over 2^24 sampled inputs (`train.exp.ulp.max`, stated) |
| Attention | Causal softmax attention as two masked products per (batch, head) through the GEMM kernel; a backward that rewrites the layer's probabilities in place with their score adjoint and feeds the GEMM kernel (no workspace beyond them); Q/K/V backward, GQA group-gradient accumulation, output projection backward. Both are bit-identical to the scalar reference loops, which `bcir_tensor_attention_backward` keeps |
| Decoder | Embedding scatter-add backward, both residual paths, both per-layer norms, gate/up/down projections, final norm, tied or untied vocabulary head, mean token cross-entropy; the loss's rows, the gamma VJPs' row sums and the embedding scatter run on the pool in the serial order (NDT-REDUCE: rows whose terms the caller adds in row order, column ranges whose sums run over the rows in row order), so every loss and state is the serial one at every thread count under every contract |
| Optimizer | AdamW moments, constant-time bias powers, decoupled decay, global L2 clipping, mean micro-batch accumulation, per-update learning rates. The atomic preflight is skipped only when the magnitudes the first pass records prove it would admit every element; the passes carry the same AVX2/AVX-512 variants and stay bit-identical The first pass and the preflight run on the pool one task per chunk of `bcir_decoder_reduce_chunk` elements (1,024 doubled until at most 1,024 chunks), their verdicts and maxima exact whatever the partition; the clipping norm is the contract's -- the one index-order chain in double, or under the `reduce` contract (NDT-REDUCE) the chunks' sums added in chunk order from zero, a declared reduction whose shape is a function of the parameter count alone; `bcir_decoder_gradient_norm` reports the square the update takes its root of |
| Threads | A deterministic pool in memory the caller provides (`bcir_tensor_pool`: `size`/`init`/`stop`, no allocation in the unit; pthreads on POSIX, Win32 threads on Windows, one thread elsewhere; ABI 2 names it in the state, NDT-THREADS). The GEMM is partitioned over (column block, 64-row range) so every output element keeps its depth order; attention over (batch, query head) forward and (batch, KV head) backward with the query heads of one KV head in order, so the shared `dk`/`dv` accumulate as the reference does; SiLU, the SwiGLU VJP, the residual add and gate multiply, RMSNorm, RoPE and the AdamW commit over fixed chunks of elements, rows and positions. The loss, the embedding scatter, the gamma VJP's row sum and the update's norm pass stay serial (index-order reductions). The state after any number of steps is bit-identical at every thread count and to the one-thread trainer; a provider's GEMM or SiLU is threaded as the provider is, the pool runs the rest |
| Buffer plan | Checked element/byte products, exact sizes for the selected retained-cache algorithm, caller-owned disjoint arenas, one adjoint workspace reused across all layers; no allocation in C |
| Checkpoint | Atomic JSON + little-endian FP32 weights/moments; SHA-256 covers configuration, counter, bias powers and payload; bounded strict resume, no pickle |
| Deployment | Existing strict HF tensor ingest performs projection transposes and RoPE permutation; `decoder_weights()` feeds the existing Q8 writer and decoder |

The parameter ABI follows the existing hosted Llama's HF names and output-by-input
matrices. The scalar Tape emitter remains available; this path uses bounded tensor
loops and analytic VJPs, so source size does not grow with parameter count.

The entry points are [`bcir_decoder_train.h`](../../runtime/c/bcir_decoder_train.h),
[`bcir_tensor.h`](../../runtime/c/bcir_tensor.h), and the cold, stdlib-only
[`NativeDecoder`](../../bcir/hosted/models/native.py) binding. Sources, C11/C23 harnesses
and the shared gate section are registered in the runtime build manifest.

## Using the trainer

```python
from bcir.frontends.models.decode import DecoderSpec
from bcir.hosted.models.native import NativeDecoder, NativeAdamW

spec = DecoderSpec(32, 16, 4, 2, 24, activation="silu_gate", n_kv_heads=1)
library = NativeDecoder.build("build/native-decoder")
trainer = NativeDecoder(library, spec, batch_size=1, context_length=4)
events = trainer.run(
    [1, 2, 3, 4] * 40, [2, 3, 4, 5] * 40,
    steps=40, optimizer=NativeAdamW(lr=0.01),
)
trainer.save("build/native-decoder/trained.bcirdt")
resumed = NativeDecoder.resume(library, "build/native-decoder/trained.bcirdt")
```

Flat inputs and targets have `steps * gradient_accumulation * batch * context`
IDs. Micro-batches are consecutive within each update. `backward(accumulate=True)`
and `update(gradient_scale=...)` also expose individual stages. The existing
`SequenceTokenSource.batch()` supplies input/shifted-target rows; corpus provenance,
tokenization, validation splits and sampling remain caller responsibilities.

`load_state_dict()` accepts flat finite spans under the HF tensor names;
`state_dict(gradients=True)` exposes every parameter gradient. Checkpoint between
optimizer updates: activations and gradients recompute on resume. Restoring a
different execution provider can change rounding; exact resume is gated on the same
provider (the pool's thread count does not reach the state). Supply the same optimizer options and subsequent
learning-rate schedule on resume; these are caller inputs rather than checkpoint
fields. Checkpoint hashes detect corruption and do not authenticate an artifact.

`NativeDecoder(..., contract="fused")` trains under the fused GEMM contract (v2; `"v1"` by default),
`contract="exp"` under the BCIR exponential, `"fused+exp"` under both, `"kblock"` under the blocked
fused GEMM contract (v3), and any of these with `+reduce` (`"reduce"` alone for v1) under the reduce
contract, whose clipping norm is the fixed-chunk sum the update's one pooled pass computes
(NDT-REDUCE);
the checkpoint names the contract and `resume` restores it, since a state trained under one is not
the other's. `NativeDecoder(..., threads=n)` runs BCIR's kernels on a pool of `n` threads (one by default;
`close()` releases them, and `resume(..., threads=n)` continues a checkpoint on any count, since the
state does not depend on it). An explicit provider is selected with
`trainer.use_cblas(library_path, symbol="cblas_sgemm")`. Configure its thread count
before measuring; the pool keeps running every kernel but the GEMM. Missing providers fail visibly. There is no automatic PyTorch
fallback. The C ABI accepts native GEMM and SiLU function pointers: C++ belongs in a
framework/vendor/SYCL adapter when that integration requires it. The core contains
no C++ objects, accelerator discovery or framework runtime.

## The train rail (TRAIN-RAIL)

The IR's build trains this decoder on the repository's own text. `tools/models/train_rail.sh`
runs the corpus recipe `tools/models/configs/bcir-corpus-860k.json` -- every tracked Markdown file
of `docs/` and `training/`, read as data under the one-way corpus law of `AGENTS.md` §2 (the IR
never imports the corpus; the corpus keeps its own gates) -- through `tools/models/build_lab_model.py`:
the corpus standard's preparation, BCIR's BPE, this trainer compiled with `CC`, the BCIRQ8 export
and its sample. It ends by reading the corpus witness, `tools/models/train_rail.py`: every tracked
Markdown file of both trees is in what the model trained on (`train.corpus.unselected` 0, the
rail's exact GEM+ row; the docs recipe alone reads 328). CTest runs it as `train-corpus`
(`ctest --preset train`; `BCIR_TRAIN_STEPS`, 40 by default, the bounded smoke), BCIR Make as the
task `train.corpus` with the same count (a corpus edit retrains; a runtime edit elsewhere does not),
and CI's qualification job for 200 steps before qualifying the model. The full run is the recipe's
own step count: `bash tools/models/train_rail.sh build/corpus-model`. The optimizations of
[the training-acceleration roadmap](../research/BCIR_TRAINING_ACCELERATION_ROADMAP.md) reach this
model by construction, because the rail trains it on this trainer.

Which contract and how many threads is the recipe's `native` block (TRAIN-POLICY, 2026-10-13).
The contract is the model's: it changes the bits, so the recipe -- the thing a model is
reproducible from, with its commit -- names it, and the shipped recipes name the fast one,
`kblock+exp+reduce`. The threads are the host's: the state is the serial one at every count under
every contract, so `"threads": "host"` trains on the CPUs the process may run on (its affinity
mask, else the CPU count; `BCIR_TRAIN_THREADS` overrides it with a positive integer and refuses
anything else), and an integer trains on that many. Where a CTest entry declares fewer processors
than the host has, the entry's trainer takes what it declares: `train-corpus` two, the BCIR Make
gate's train task one (its two workers are its two processors); CI's qualification job runs the
rail itself, on the runner's CPUs. `build_lab_model.py` checks the block with the
rest of the recipe before any work (a contract the trainer does not admit, a thread count that is
neither `"host"` nor positive) and the model card records the contract, the threads, the policy
and the widest kernel instance the host ran (`avx512f`, `avx2-fma`, `portable` -- provenance, not
a parameter: the contract's result is the same on each). The tests hold the law on the built
model: the tiny recipe at another thread count is the same checkpoint byte for byte, under another
contract another one. On this host the 40-step smoke trained at 2706 tokens/s under
`v1` on one thread, at 3705 under the fast contract on one thread and at 6690
on the host's 4 (avx512f; 30.3 -> 12.2 s of training
inside a rail run of 314 -> 286 s, the rest of which is the corpus, the
tokenizer, the validation batches and the export -- the next thing to measure), its checkpoint at one
thread and at 4 the same ([policy/](../research/measurements/2026-10-13/policy/)).

## The step as a GEM+ program (NDT-GEM)

[`native_program.training_step_program`](../../bcir/hosted/models/native_program.py) states one
training step from the spec alone, in the vocabulary GEM+'s bounds read: every span of the six
arenas in the C plan's order, the values those spans hold (an overwrite starts one, an
accumulation keeps the one it reads), and one step per kernel call in the C rail's order with the
values it reads and writes, its GEMMs and its counted work. Counted work is two units per
multiply-add of the contractions every implementation performs: each linear's product and its two
VJPs, and the causal attention's six contractions. Normalization, softmax, RoPE, SwiGLU, the loss
and the optimizer are not counted, so the count bounds the step's work from below.

Four rows of the frozen GEM+ harness (`python tools/perf/gemplus_baseline.py --compare --group
train`, graded by [`native_decoder_fixtures`](../../bcir/tests/native_decoder_fixtures.py)) hold
the C rail to it:

| Row | Kind | What it holds | Reading |
|---|---|---|---|
| `train.plan.mismatch` | exact | the program's arena sums against `bcir_decoder_make_plan` over a six-spec corpus (tied and untied, MHA and GQA, context one, tile-crossing widths) | 0 of 36 (spec, arena) pairs; the parent checked the 24 parameter pairs only |
| `train.mm.mismatch` | exact | `bcir_tensor_mm` against an independent binary32 emulation of its contract, bit for bit, on 20 cases (every transpose pair, the tiles' edges) | 0 of 20, and 0 on the parent kernel: the contract both keep |
| `train.memory.transient` | exact | the bytes the C plan reserves for activations and workspace over G26's concurrent-live bound on the same values | 1.326x: 999,424 bytes held over a 753,664-byte bound at the bench shape |
| `train.mm.fused.mismatch` | exact | `bcir_tensor_mm_fused` against an exact rational emulation of the fused contract rounded once per term, bit for bit, on the same 20 cases | 0 of 20 (2026-10-12, NDT-FMA) |
| `train.mm.kblock.mismatch` | exact | `bcir_tensor_mm_kblock` against an exact rational emulation of the blocked fused contract (v3) on 40 cases (the 20 fused cases and 20 deeper ones crossing the split rule's three cases, an inexact alpha, beta 1), plus the depths where the C split rule and the Python one disagree | 0 of 40 (2026-10-12, NDT-KBLOCK) |
| `train.mm.provider.unsound` | exact | whether `NativeDecoder.use_cblas`'s verdict on OpenBLAS (scipy_openblas32) over the comparison decoder disagrees with the differential it is defined by -- admitted though a case differs, refused though none does, or refused with another count; 0 on a host that admits (the 2026-10-12 Intel host) and on one that refuses (PR #817's AMD EPYC host, 14 of 30) alike; measured where the wheel is importable | 0 |
| `train.mm.provider.differences` | exact, host-dependent | of the comparison decoder's 15 step shapes at beta 0 and 1, the cases where OpenBLAS's `sgemm` (one thread) differs in any bit from `bcir_tensor_mm_kblock`: the wheel's target kernels' arithmetic, not BCIR's -- reported everywhere, graded only on the baseline host (0 of 30 on the Intel host's SkylakeX kernels; 14 of 30 on PR #817's AMD host, where the trainer refuses the provider) | 0 (Intel), 14 (AMD) |
| `train.exp.mismatch` | exact | inputs among 2^24 where two instruction-set instances of the BCIR exponential differ | 0 (2026-10-12, NDT-EXP) |
| `train.exp.ulp.max` | exact, stated | the exponential's largest distance from exp in double rounded to binary32 over the same inputs | 1 ulp (2026-10-12, NDT-EXP) |
| `train.state.threads.mismatch` | exact | the state (weights, both moments, the step, the bias powers) after three steps of the threads bench (vocab 250, width 72, 4 heads over 2 KV heads, 2 layers, batch 2 x 40 tokens) on two, three and four threads of the pool, and the losses and clipping norms those steps report, against the one-thread run's on the same library, seeded weights, tokens and rates, under v1 and under the reduce contract | 0 of 6 (2026-10-12, NDT-THREADS; the reduce contract and the reported losses and norms 2026-10-13) |
| `train.norm.reduce.mismatch` | exact | the clipping norm's square the trainer's first pass computes (`bcir_decoder_gradient_norm`) and the root each update reports, under v1 and under the reduce contract, against the oracle's emulation of that contract from the gradients the backward left (three steps of the threads bench, bit for bit), plus the counts around the chunk rule's first doubling and the 9.84M decoder's on which the C rule and the Python one disagree | 0 of 6 norms and 8 counts (2026-10-13, NDT-REDUCE) |
| `train.step.ms` | wall, floor in the same run | one native step of the bench decoder against G26's roofline compute term: the counted work at the best rate the same kernel reaches on any of the step's own GEMM shapes | 3.4 ms against a 0.78 ms floor measured beside it (2026-10-11, clang 18; the same host read 4.6 against 1.3 before NDT-CLANG, and NDT-GEM's reading was 6.9 against 2.8) |

The step's floor is a bound on this host's kernel (TMSAO-3), not on the silicon: its peak is a
rate the kernel was measured to reach, not a declared hardware peak. The memory terms of the
roofline are not applied to the timed loop, because the bench step's working set stays
cache-resident from one step to the next and first-touch traffic is not the loop's.
`TrainingStepProgram.roofline` takes declared peaks and memory levels for a bound on a machine.

The memory row records headroom, not a change: GEM+'s incumbent layout portfolio lays the same
values out at the bound, so a plan that places the transient values by lifetime instead of one
span per buffer would hold 245,760 bytes (25%) less at the bench shape. That changes the C ABI's fixed
spans, so it is its own slice (NDT-MEM, open).

## Evidence and scope

`tools/models/test_native_decoder.py` compares native logits, every named gradient,
and multi-step clipped AdamW against the independent hosted PyTorch model. It covers
two-layer tied/GQA, untied/MHA, and context-one cases. The math SDPA reference is used
for gradient trajectories: fused SDPA can leave rounding noise in theoretically zero
context-one Q/K gradients, which Adam's small epsilon amplifies. The timed reference
uses PyTorch's default eager path and `torch.optim.AdamW` without foreach/fused updates.

`runtime/c/test_decoder_train.c` checks the GEMM contract bit for bit against the
ascending-k loop over every transpose pair, the register blocks' and tiles' edges and every
instruction-set variant the host can run; attention forward and the in-place backward against
the scalar reference loops (finite and non-finite operands); the AdamW update against the
reference update on both its bounded and preflight paths; finite differences for every parameter,
gradient accumulation, invalid IDs/capacities/aliasing, atomic numerical update
rejection and falling loss in the complete native loop. It runs under C11 and
C23, and again with the kernels held to their portable and AVX2 variants, via
`tools/c/sections/decoder_train.sh`; the Python oracle also exercises it. The library is built
with `-O3 -ffp-contract=off -fno-math-errno` (the manifest's options for the unit and
`NativeDecoder.build` alike): `-fno-math-errno` lets the AdamW passes vectorize `sqrt` and changes
no result.
The hosted CI cells require the tensor differential, including exact native resume
and training-to-Q8-to-C-inference parity.

The [2026-10-11 report](../research/BCIR_NATIVE_TRAINING_PERFORMANCE_2026-10-11.md) measures this
trainer's GEMM, attention and AdamW kernel by kernel against the ones they replaced. It also trains
a 9.84M-parameter decoder on this trainer and on PyTorch (eager and compiled) from identical
weights, batches and learning rates. It reports time per step, held-out quality on a byte-level
language model of the repository's documentation and on a copy task, time to quality and memory,
and where each framework's step spends its time.

The training corpus as the trainer's data (TRAIN-RAIL) and the deterministic thread pool
(NDT-THREADS) landed on 2026-10-12. The pool's evidence is three-fold: `runtime/c/test_decoder_train.c`
holds every pool kernel (GEMM shapes crossing the row tasks and every column-block width under
every transpose pair, causal GQA attention forward and in-place backward, SiLU, the SwiGLU VJP,
RMSNorm, RoPE, the residual add and gate multiply, the task runner) and three complete training
steps at one to four threads to the serial kernels and the scalar references, bit for bit; the
runtime gate's `decoder_train_tsan` section runs that harness under ThreadSanitizer and holds the
same build of a pool whose attention backward puts one KV head's query heads on two threads to a
race report (the manifest's `test_decoder_train_race` mutation); and `train.state.threads.mismatch`
holds the Python trainer's state across thread counts, with `tools/testing/faults/ndt-threads.json`
the injected partition faults each witness catches. The fused GEMM contract landed the same day (NDT-FMA): `gemm_fused_contract` in the C harness holds
every fused instance to the `fmaf` loop, `train.mm.fused.mismatch` holds the kernel to an exact rational
emulation rounded once per term (`round_f32`, never through a double), and
`tools/testing/faults/ndt-fma.json` the injected faults. The BCIR-owned exponential landed too (NDT-EXP): `exp_contract` in the C harness, the two rows
above, and `tools/testing/faults/ndt-exp.json`. The three were re-measured together the same day
([BCIR_NDT_MEASURE_2026-10-12.md](../research/BCIR_NDT_MEASURE_2026-10-12.md)): every state its
contract's own and identical at one to four threads, the 1,000-step v1 run on three threads ending
in the 10-11 one-thread run's state byte for byte, and the 9.84M decoder's step on three threads at
819 ms under `fused+exp` against 989 under v1 (the same model to four decimals), with the step's time
by kernel at 1, 3 and 4 threads read against the roadmap's budget row by row. The blocked fused
contract (v3, NDT-KBLOCK) landed the same day, from that measurement's one surprise: OpenBLAS's
`sgemm` was bit-identical to the fused contract wherever one K block covered the depth, and a
K-blocked form of it everywhere else. `gemm_kblock_contract` in the C harness holds every instance
to the K-blocked `fmaf` chains and the split rule on ten depths, `train.mm.kblock.mismatch` holds
the kernel to its exact emulation, and `use_cblas` admits a CBLAS provider under v3 only by a
differential over the trainer's own shapes -- the first provider that is a BCIR contract bit for
bit rather than its own arithmetic; `tools/testing/faults/ndt-kblock.json` holds the injected
faults. The admission is per host: OpenBLAS's SkylakeX kernels compute the contract (0 of 30
cases differ on the 2026-10-12 Intel host, admitted), its kernels on PR #817's AMD EPYC host do
not (14 of 30, refused, the state on BCIR's kernels), and what the rows hold is the guard, not
the wheel: `train.mm.provider.unsound` is 0 on both (the verdict is the differential's) and
`train.mm.provider.differences` is the host's count, reported and never graded off the baseline
host (PR817-P0). A provider whose
worker threads spin between calls contends with the pool's kernels: OpenBLAS's spin for 2^28
cycles after each call cost the four-thread step a fifth until `OPENBLAS_THREAD_TIMEOUT` was set
low before the library loaded (the comparison harness sets it; the report's §7 measures it). What
comes next for
this trainer -- the plan driving the step's dispatch, tiled attention --
is the ladder of
[the training-acceleration roadmap](../research/BCIR_TRAINING_ACCELERATION_ROADMAP.md).

The v3 kernel's own rate (NDT-BLOCK, 2026-10-12) is the provider's. Its AVX-512 and AVX2
instances are intrinsic register blocks of 8 x 48 and 6 x 16 accumulators over a tile packed by
vector, with a transposed A prefetched a few depths ahead and the pool's tasks 128 rows by one
column block of the instance's width; the generic instance keeps the plain-C block, and every
instance is held to the K-blocked `fmaf` chain by the C harness and by `train.mm.kblock.mismatch`,
which now runs per instance. On the step's 15 GEMM shapes the kernel reads 99 GFLOP/s
work-weighted at one thread against OpenBLAS's 89 in the same process and 219 on the pool at four
against OpenBLAS's 219 ([the record](../research/measurements/2026-10-12/block/gemm-rates-block.json);
the report's §8). The result is the contract's, so the state digests of §7 are unchanged.

The reductions the pool left serial run on it since NDT-REDUCE (2026-10-13). The loss's rows,
the gamma VJPs' row sums and the embedding scatter are tasks in the serial order -- rows whose
terms the caller adds in row order, column ranges whose sums run over the rows in row order --
so every loss and state is the serial one at every thread count under every contract; the
update's first pass and preflight are one task per chunk of `bcir_decoder_reduce_chunk` elements
(1,024 doubled until at most 1,024 chunks), their verdicts and maxima exact, and the clipping
norm is the contract's: the one index-order chain under the contracts that were, or under the
`reduce` contract the chunks' sums added in chunk order from zero, a declared reduction whose
shape is a function of the parameter count alone, so the first pass is pooled whole. The reduce
bit changes the norm's last bits and nothing else; the float rounding of every update absorbs
that difference for a while (the harness's three steps, the threads bench's three), and once a
rounding boundary is crossed the state is the reduce contract's own (on the 9.84M decoder within
twenty steps); the contract's witness is the norm, which the oracle emulates operation for
operation, and `bcir_decoder_gradient_norm` reports the square the update takes its root of so the
row compares the sum before the root rounds a difference away. `reduction_references` in the C
harness holds the pooled loss and scatter to serial references written apart from the pool,
`pool_identity` holds the twelve contracts' states, losses and norms at one to four threads to the
serial run and the gamma VJP to the serial kernel, `adamw_reference` holds the update under both
norm contracts to a reference of each, serially and pooled, `train.norm.reduce.mismatch` holds
the norms to the oracle's emulation with the chunk rule on both rails, and
`tools/testing/faults/ndt-reduce.json` holds eleven injected faults (a chunk following the thread
count, the terms or the rows walked from the last, a chunk's verdict dropped, a ragged tail
dropped, a rule that never doubles). Measured on the 9.84M decoder
([the record](../research/measurements/2026-10-13/reduce/), the report's §9): in three same-session pairs at four threads (each contract's breakdown alternated with the other's, minutes apart, so both see one host) the update reads 40.7 / 49.7 / 47.5 ms under `kblock+exp` -- the first pass pooled, the chain serial, NDT-NORM's content -- against 31.8 / 47.2 / 39.6 under `kblock+exp+reduce`, below it in every pair (-9.0 / -2.6 / -7.9), with §8's 53.1 the day before on a faster host; the gamma VJP row 11.0 / 14.3 / 12.4 against 6.7 / 8.8 / 11.8 (-4.3 / -5.5 / -0.6; 9.2 in §8); the step itself is not resolved on this day's host, whose unchanged GEMM rows drifted 20% within the session, more than the slice's effect on the step; the
1,000-step `kblock+exp+reduce` run trains at 625.8 ms a step on four threads (1419.6
on one) and ends in `5137e013…`, its own, not §7's `d7fc618c…` (the norm's last bits crossed a float rounding boundary within the first steps, as the 20-step identity rows already show, and the state is the reduce contract's from there), and the one-thread run ends in the same state, the held-out model 1.5003 / 1.6319
nats per byte as before.

The two VJPs of a linear run as one pool job since NDT-SCHED-0 (2026-10-13): `dx = dy W` and
`dW += dy^T x` read one adjoint and write apart, so `bcir_tensor_mm_pair_pool` runs the first
GEMM's tasks and then the second's in one `pool_run`, each with the partition its own entry would
have used, so every output element keeps its depth order and the results are the two entries' bit
for bit (the C harness holds every shape pair under the three GEMM contracts in both orders, and
the trainer's twelve contracts at one to four threads); a provider's GEMM keeps its two calls.
One barrier in place of two for each of the step's 43 linears, and the narrow K/V projections'
idle threads take the other product's tasks. Measured in 3 same-session pairs at four
threads under `kblock+exp+reduce`: the GEMM-backward row 292.7 / 285.9 / 266.6 -> 238.0 / 254.8 / 282.6
ms (-54.6 / -31.0 / +16.0), the step 684.2 / 679.4 / 659.3 -> 645.1 / 681.7 / 668.1 (-39.2 / +2.4 / +8.8);
the backward pass (the `loss_backward` stage, the forward recomputed inside it, as the trainer runs it -- the one stage the pair job changes) is below the two-call stage in 3 of 3 pairs, by -30.9 / -47.5 / -27.8 ms (631.9 / 633.8 / 622.4 against 601.0 / 586.3 / 594.6), and the GEMM-backward row timed alone by -54.6 / -31.0 / +16.0; the step's own cell (-39.2 / +2.4 / +8.8) is within this host's run-to-run noise, as NDT-REDUCE's was -- the one barrier fewer per linear and the narrow projections' tasks shared are worth about 5% of the backward at four threads here.

The step's whole-buffer passes run on the pool since NDT-SCAN (2026-10-13). Timed in place --
a clock read after every statement of the step -- the finiteness scans of the weights, the
activations and the gradients, the gradients' zero fill and the attention's zero fills were
11% of the four-thread step on the caller's thread alone. A scan's verdict and a
fill's zeros do not depend on the partition, so each runs as one task per chunk, and the
attention's fills run in the tasks that own the slices; every verdict and value is the serial
pass's. Measured: 3 in-place pairs: the serial passes 60.9 / 68.4 / 68.3 -> 17.4 / 16.2 / 18.6 ms (-43.5 / -52.2 / -49.7); 6 breakdown pairs at four threads under `kblock+exp+reduce`: the step 575.4 / 634.9 / 601.2 / 653.9 / 682.9 / 621.1 -> 550.9 / 588.2 / 615.7 / 616.6 / 642.6 / 563.4 ms (-24.4 / -46.6 / +14.6 / -37.2 / -40.3 / -57.7, below in 5 of 6), the forward stage 213.3 / 223.2 / 219.4 / 233.0 / 263.7 / 201.5 -> 199.9 / 201.4 / 194.0 / 213.9 / 230.6 / 199.4 (-13.4 / -21.9 / -25.4 / -19.0 / -33.1 / -2.1); the 20-step state at one and four threads the same on both trees (4bf4e057…). The same run under UBSan found the BCIR exponential's clamp passing a
NaN to an integer conversion; the clamp is total now, every other input's result unchanged.

The update's first pass runs as vector loops since NDT-PASS (2026-10-13). Its loop -- each
element finite, the second moment not negative, the largest magnitudes the preflight's bound
reads -- had compiled to scalar code (the refusal flag's short-circuit and the floating maxima
written as selects); it is now one refusal flag without a branch and the maxima as the largest
bit patterns read as int32 (a non-negative float orders as its bits; a -0.0 second moment reads
as INT32_MIN and never wins, as the select from +0.0 never took it), with AVX2 and AVX-512
instances, all three vectorized by GCC and Clang. Every verdict and every maximum the update
reads is the scalar loop's; the gradient norm's chunk partial stays the index-order chain in
double. The C harness puts a first moment only the preflight refuses at either end of either
chunk and at the vector instances' block edges, and `tools/testing/faults/ndt-pass.json` holds
the loop one element short and the maximum kept as a minimum. Measured in place at four threads
(the report's §12): the first pass 23.2 / 20.3 / 24.6 -> 12.0 / 14.6 / 13.2 ms in three
alternated pairs.

The attention backward evaluates its probability adjoint once per row block since NDT-ATTN-1
(2026-10-13). The in-place backward needs `dp_ij = dy_i . v_j` twice -- a first pass for each
row's `dot_i = sum_j p_ij dp_ij`, a second for the score adjoint `p_ij (dp_ij - dot_i) scale` --
and computed it twice, as the reference does, over operands neither pass writes. A row block now
keeps its first 256 columns' tiles between the passes (32 KB on the task's stack) and a longer
context evaluates its further tiles in both passes through one spare slot, as every tile was
before; dq, dk and dv are the reference's bit for bit (the C harness adds a 330-token context,
past the kept columns, to its attention reference; `tools/testing/faults/ndt-attn.json` NA1-NA3).
The backward kernel alone on the 9.84M decoder's shape: 14.6-15.4 -> 12.1-12.5 ms at one thread.

The attention's masked products run as AVX-512 register blocks since NDT-ATTN-2 and NDT-ATTN-3
(2026-10-13). They ran through the GEMM kernel's v1 blocks, written for the linears: the scores
computed their 64-wide column blocks whole above the causal diagonal, the context padded its 48
channels to 64 and tested the mask per term in every block the diagonal crosses, and the
backward's dv, dq and dk the same. The forward's scores and context and the backward's dv
(`P^T dY`), dq (`dS K`) and dk (`dS^T Q`) now run, on the AVX-512 instance, as blocks of eight
rows with the v1 arithmetic exactly -- an element from the value it holds (the zero its task
filled, or the sum a KV head's earlier query heads left in dk and dv), one rounded product and
one rounded sum per term in the term order -- the scores 32 keys at a time from their diagonal
on, the context and dq over keys `j <= i`, dv and dk over terms `i >= j`, the channels 48 at a
time; rows that fill no block of eight run through the kernel itself, and the AVX2 and generic
instances keep it whole. The probabilities, the context and the three adjoints are the reference
loops' bit for bit (the C harness's attention reference gains the blocks' edges: rows of no whole
block, exactly one, one and a row; channels of one, under and over a vector; `ndt-attn.json`
NA4-NA9). The kernels alone at one thread on the 9.84M decoder's shape: the forward 11.1-11.3 ->
9.1-9.4 ms, the backward 14.6-15.4 -> 8.4-9.0 with NDT-ATTN-1; in place at four threads the
attention backward 46.8 / 39.8 / 44.5 -> 23.1 / 26.3 / 27.2 ms (the report's §12).

Timing output records shape, parameter count, provider, optimizer scope and reference
mode. Timings are observations, never CI thresholds. A CPU micro-decoder win over
eager PyTorch is not evidence against compiled PyTorch, large transformers,
accelerators or distributed training.

The plan retains every layer's forward cache and reuses backward scratch; it is not
a globally optimal liveness solver. The backward consumes the cache it recomputes each step:
after `bcir_decoder_loss_backward` each layer's probability span holds its score adjoint.
Attention storage is quadratic in context length.
Flash/blocked attention, activation recomputation, mixed precision, accelerator
backward kernels, distributed collectives and device-resident checkpoints remain
separate work. The direct loop does not issue StreamPack claims for every tensor;
it does not change D1's planned/streamed logistic training contract or assert new
MLIR verifier coverage.
