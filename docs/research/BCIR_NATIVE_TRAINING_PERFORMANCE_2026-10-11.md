# Native training kernels, and a 9.84M-parameter decoder trained on BCIR and on PyTorch (2026-10-11)

> Dated, non-normative. The native-training half of the update the [2026-10-10
> assessment](BCIR_GEMPLUS_ASN1_ML_PERFORMANCE_2026-10-10.md) recommends (it landed with the
> assessment in #814), a review of that assessment against its own evidence, and one decoder trained
> end to end on BCIR's native C trainer and on PyTorch from identical weights, data and learning
> rates. Everything below is reproducible with
> [`train_comparison.py`](measurements/2026-10-11/train_comparison.py) (§8); the raw records are
> beside it in [`measurements/2026-10-11/`](measurements/2026-10-11/). Counts live only in the
> generated [`STATUS.md`](../STATUS.md).

## 0. The answer

- **BCIR's own trainer is 7.2x faster per step, and computes exactly what it computed before.** At
  the 9.84M-parameter decoder's training shape (B=8, T=128, one thread), a step on BCIR's own
  kernels took 12.30 s on the base commit and takes 1.70 s now; with OpenBLAS behind its CBLAS
  seam, 1.32 s and 0.96 s. The training state the steps leave is bit-identical to the base
  commit's, and identical across GCC 13, clang 18 and clang 23 and across the portable, AVX2 and
  AVX-512 builds (§4). The step's GEMMs went from 11.6 s to 1.24 s (5.2 to 48.8 GFLOP/s), attention
  from 431 ms to 143 ms, AdamW from 147 ms to 65 ms (§3).
- **Trained side by side from the same weights, batches and learning rates, BCIR and PyTorch
  produce the same model.** On a language model of the repository's own documentation, all four
  backends end at 1.6319 nats per byte on the test split (one at 1.6318), 53.7% next-byte
  accuracy, and 107-109 of 1,532 held-out words completed. On a copy task, each copies all 2,000
  held-out strings exactly (§5).
- **PyTorch is faster per step.** BCIR's own kernels take 1,726 ms per step, BCIR with OpenBLAS
  963 ms, eager PyTorch 774 ms and compiled PyTorch 712 ms. Because the language-model curves
  coincide, the same ratios hold for time to any quality: 2.2x eager PyTorch's time on BCIR's
  own kernels, 1.24x with OpenBLAS. The gap is the GEMM. MKL runs a step's 60.4 GFLOP in 476 ms
  and BCIR's kernel in 1,329 ms; that kernel performs one rounded multiply and one rounded add
  per term, in a fixed order, so that it is bit-identical everywhere. Everything else in the
  step is faster on BCIR (§5.5).
- **Clang built a slower trainer, and now does not (NDT-CLANG).** `NativeDecoder.build` prefers
  clang, and clang's builds of #814's kernel trained 30-34% slower than GCC's with AVX-512. The
  cause was that clang did not align the GEMM's packed tile to a cache line, so the misalignment
  varied from run to run (§4.1). The fix is bit-identical. On AVX-512 all three compilers now
  train at 1.57-1.59 s per step. The clang 18 step is 2.0x faster than on main in an interleaved
  measurement, the clang 23 step 1.23x, and the GCC step 1.05x.
- **BCIR uses half the memory.** BCIR's processes peaked at 544 MiB, PyTorch's at 1,056-1,140 MiB.
  BCIR's arena is planned before the first step and does not grow.
- **The ratio holds at longer contexts; threads and long contexts favour PyTorch (§5.8).** At
  1,024 tokens per step, BCIR's own kernels stay at 2.0-2.1x eager PyTorch's step time from T=64 to
  T=1024, and the losses agree at every context. BCIR's attention stores every probability, so its
  arena grows from 323 to 503 MiB over that range while PyTorch's flash kernel keeps its memory
  flat; one attention layer goes from faster than PyTorch's at T=64 to 1.87x slower at T=1024.
  BCIR's kernels run one thread: on this VM's four cores, PyTorch's step is 6.1x faster than theirs.
- **The 2026-10-10 assessment is accurate as stated.** It measured forward kernels only and never
  ran BCIR's own kernels in training. Those gaps are filled here (§1). Before this update, the
  backward GEMM shapes it did not measure were 86% of a training step's GEMM time.

## 1. The 2026-10-10 assessment, reviewed

Every table in the assessment regenerates from its
[`results.json`](measurements/2026-10-10/results.json) to the printed digits, and every ratio in its
prose recomputes (14.6x, 13.7x, 10.37x, 7.92x, 72.5%, 1.68x, 43.75%, 12.0%, 1.53x, under 1%). The
harness's SHA-256 matches the committed harness, the recorded production-source hashes match the
base commit `75d50d4`, and the memory figures of its §4 (193.36, 150.10, 43.26 and 39.12 MiB)
recompute from the archived decoder record. Nothing in it is wrong as stated. What it leaves out
shapes the update below:

1. **The GEMM rows are forward shapes only** (`alpha` 1, `beta` 0, A untransposed). A training step
   also issues `dx = dy W` (`beta` 0 or 1) and `dW += dy^T x` (A transposed, `beta` 1). On the base
   kernel those ran 3.1x slower per flop than the forward shapes, and 86.2% of its
   11.6 s of GEMMs per step at B=8, T=128 were backward shapes (§3).
2. **Attention is measured forward only.** The base backward cost 2.9x the forward per
   layer (§3), and it is the backward that a training step pays most for.
3. **The production optimizer is not measured.** Its P1 row rests on an external prototype's 12%;
   the shipped update took 147 ms per step at 9.84M parameters on this host.
4. **The decoder evidence never ran BCIR's own kernels.** The archived 9.84M comparison put BCIR on
   MKL and OpenBLAS providers, against compiled PyTorch only, at B=4, T=64 for 96 updates, without
   an eager baseline, a task metric or a time to quality. On its own kernels the base trainer was
   9.3x slower per step than with OpenBLAS (§4).
5. **Its portable rows are not the binary that trains.** The harness compiles the "portable C" GEMM
   and attention with `-O3 -march=native -fno-math-errno -ffp-contract=off`. The trainer was never
   built that way: `NativeDecoder.build` used `-O3 -ffp-contract=off`, and the gates their default
   `-O2`. [`march_check.py`](measurements/2026-10-11/march_check.py) builds the base kernel all
   three ways on this host. The output is the same every way; the speed is not. On the stage's
   untransposed-B shapes the harness's build took twice as long as `NativeDecoder.build`'s (36.7
   against 18.2 ms at 256x384x384). On its transposed-B shape, the gates' `-O2` build was the slow
   one (44.5 ms against 14.4). So the rows are neither the trainer's cost nor a bound on it. This
   update's stages build each trainer exactly as it ships (§3, §4), and the runtime instruction-set
   dispatch makes `-march=native` unnecessary for the new kernels.
6. **The host matters and differs.** The assessment ran on an AMD EPYC 9V74 VM; this one runs on a
   four-core Intel Xeon VM (AVX-512). Absolute times do not carry across the two reports, so every
   before/after pair here is measured on one host in one session, and §6 re-runs the assessment's
   own kernel and attention stages before and after on this host. Its records name PyTorch,
   NumPy and SciPy versions but not the BLAS behind PyTorch (MKL here) or the kernels OpenBLAS
   selects (SkylakeX here); the summary record of this update carries both.

## 2. What changed

Three kernels of the native trainer changed, and each still produces exactly the output it had:
every result below is bit-identical to the base commit's on every input the tests feed it, finite or
not. So a training run's state is the same as before: §4 checks the full state after twenty steps
across compilers and instruction-set variants, and after the baseline stage's steps against the base
commit's trainer.

**GEMM** (`bcir_tensor_mm`). The contract is unchanged: each element starts at its beta-scaled
value (zero when `beta` is zero) and adds `(alpha*A(i,l))*B(l,j)` for `l` ascending, one rounded
multiply and one rounded add per term -- the build keeps `-ffp-contract=off`, so there is no FMA.
Only the loop nest changed. A 64xNR tile of B is packed once per (column block, depth block) and
reused by every row, and an MRxNR block of C stays in registers across the tile's depth instead of
taking a load and a store per term. The base kernel was bound by exactly that store-to-load chain,
and on its `tb=0` path also by alias checks around a 32-iteration loop. The kernel is one inline
function instantiated three times: portable C (2x32 blocks) everywhere, and on x86-64 Linux with
GCC or Clang an AVX2 (2x32) and an AVX-512F (4x64) instance chosen per call with
`__builtin_cpu_supports`. All three perform the same rounded operations in the same order.
`bcir_tensor_mm_variant` runs one variant, or refuses one the host cannot run and writes nothing;
`BCIR_TENSOR_NO_DISPATCH` and `BCIR_TENSOR_MAX_VARIANT` hold a build to fewer. The block's one
precondition -- the columns it stores are the columns it loaded -- holds at every call site, and
is stated to `clang --analyze` alone (`__clang_analyzer__`, 7fdb049). As plain C, GCC 13 does not
fold the clamp away and spills in the AVX-512 instance; stated to the analyzer alone, it leaves the
object code of `bcir_tensor.c` byte for byte as it was under GCC 13, clang 18 and clang 23. So what
§3-§5 measured is the kernel #814 shipped, and §4.1 measures this update's change to it
(NDT-CLANG).

**Attention.** The forward is two products per (batch, head) through the same kernel, with a mask
over the matrices' own indices: `Q K^T` stores only `j <= i`, and `P V` takes term `l` into row `i`
only when `l <= i`. Between them is the reference loop's softmax, row by row. A masked term is
skipped, never added as a zero, so no value outside the causal triangle -- a non-finite one
included -- reaches a result. The new `bcir_tensor_attention_backward_inplace` computes dV first
(`P^T dY` over rows `i >= j`). It then evaluates the probability adjoint `dY V^T` tile by tile,
twice, as the reference does: once for each row's `dot_i`, then to write `ds_ij = p_ij (dp_ij -
dot_i) scale` into P in place. Finally it forms dQ (`dS K`) and dK (`dS^T Q`). Heads are taken in
order for the shared GQA rows, so every accumulator receives the reference's terms in the
reference's order. It needs no workspace: the decoder reads each layer's probabilities for the last
time there, so the arena plan is unchanged (the frozen `train.memory.transient` row reads the same
999,424 / 753,664 bytes). `bcir_tensor_attention_backward` stays as the scalar reference.

**AdamW** (`bcir_decoder_update`). The preflight and commit passes keep their arithmetic -- moments
rounded to float before use, the step in double -- and gain the same AVX2/AVX-512 instances; the
build's new `-fno-math-errno` lets `sqrt` vectorize, which changes no result. The first pass,
which still checks every element and sums the gradient norm in double in index order, now also
records the largest |g|, |w|, |m1| and m2. From those maxima, `adamw_bounded` proves that no
candidate can leave float's range: each rounded operation is monotone, so the same expression
evaluated on the maxima bounds every element, with half of FLT_MAX held in reserve. When the
proof holds, the preflight pass is skipped; when it does not (an epsilon of 1e-40 with moments of
1e30, say), the preflight runs exactly as before, and an update it refuses still writes nothing.

**Build.** The unit's options are `-O3 -ffp-contract=off -fno-math-errno` in the manifest (CMake
and BCIR Make) and in `NativeDecoder.build`. Before, Python built it with `-O3 -ffp-contract=off`
and the gates at their default `-O2`.

**Witnesses.** The C harness (`runtime/c/test_decoder_train.c`) checks the GEMM bit for bit against
the ascending-k loop over every transpose pair, the blocks' and tiles' edges and every variant the
host runs, and requires the refusals; attention forward and the in-place backward against the
scalar reference loops, with finite and non-finite operands, future scores exactly +0; the update
against the reference update on both its paths, a refused update included. The `decoder_train`
section runs it as C11, C23, portable-only and AVX2-capped builds. RED:
[`ndt-perf.json`](../../tools/testing/faults/ndt-perf.json) injects nine faults (a mask off by one
in each direction, a store past the diagonal, a lost row bound, the dot product accumulated twice,
dV unmasked, the bound proving everything, a negative moment admitted, an undefined variant
accepted) and the harness catches each; `ndt-gem.json`'s three GEMM faults are re-anchored on the
new kernel and still caught. The independent PyTorch differential
(`tools/models/test_native_decoder.py`) is unchanged and passes.

## 3. Kernels, before and after

The `kernels` stage builds one C program twice: once against the four trainer sources as they were
at `75d50d4`, compiled with the flags they shipped with (`-O3 -ffp-contract=off`), and once against
these, compiled with theirs. It runs every GEMM shape one training step issues at B=8, T=128, both
attention directions and the AdamW update. The two builds alternate over three rounds, and the
medians are below. Every output is hashed, and the stage requires every hash to agree: before
against after, and across the three instruction-set variants. They agree. Compiler: GCC 13.3.

| GEMM (calls per step) | m x n x k | A transposed | before (ms) | portable | AVX2 | AVX-512 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| q, o forward (12) | 1024 x 384 x 384 | no | 23.44 | 13.57 | 8.75 | 5.17 |
| k, v forward (12) | 1024 x 192 x 384 | no | 9.40 | 5.98 | 4.31 | 2.56 |
| gate, up forward (12) | 1024 x 1024 x 384 | no | 64.10 | 38.15 | 25.51 | 16.24 |
| down forward (6) | 1024 x 384 x 1024 | no | 71.28 | 39.76 | 27.26 | 16.60 |
| head forward (1) | 1024 x 260 x 384 | no | 15.22 | 9.28 | 6.61 | 4.41 |
| q, o dx (12) | 1024 x 384 x 384 | no | 70.56 | 12.50 | 8.96 | 5.33 |
| k, v dx (12) | 1024 x 384 x 192 | no | 36.48 | 5.95 | 4.18 | 2.62 |
| gate dx (6) | 1024 x 384 x 1024 | no | 198.03 | 39.35 | 29.63 | 17.43 |
| up dx (6, `beta` 1) | 1024 x 384 x 1024 | no | 191.92 | 40.44 | 29.23 | 17.64 |
| down dx (6) | 1024 x 1024 x 384 | no | 208.37 | 39.36 | 26.87 | 16.77 |
| head dx (1) | 1024 x 384 x 260 | no | 48.84 | 8.12 | 5.70 | 3.48 |
| q, o dW (12) | 384 x 384 x 1024 | yes | 72.67 | 13.07 | 8.16 | 5.33 |
| k, v dW (12) | 192 x 384 x 1024 | yes | 35.48 | 5.60 | 4.93 | 3.19 |
| gate, up dW (12) | 1024 x 384 x 1024 | yes | 204.58 | 49.78 | 27.78 | 17.39 |
| down dW (6) | 384 x 1024 x 1024 | yes | 212.87 | 44.56 | 29.98 | 20.06 |
| head dW (1) | 260 x 384 x 1024 | yes | 51.77 | 10.51 | 6.91 | 4.58 |
| **one step's GEMMs (60.4 GFLOP)** | | | **11,611** | **2,984** | **1,988** | **1,237** |
| rate | | | 5.2 GFLOP/s | 20.2 | 30.4 | 48.8 |
| speedup | | | 1x | 3.89x | 5.84x | 9.38x |

`dx` is `dy W` (`beta` 0, or 1 where a second VJP accumulates into the same input gradient); `dW`
is `dy^T x` accumulated into the gradient (`beta` 1). Each time is per call.

The base kernel ran the forward shapes at 12.5 GFLOP/s and the backward shapes at 4.0. The forward
shapes are a third of the flops but took 1.6 s; the backward shapes took 10.0 s, 86.2% of the
step's GEMM time. The new kernel runs both alike: on AVX-512, 51.4 GFLOP/s forward and 47.6
backward, between 40.1 (`down dW`) and 59.1 (`k, v forward`) GFLOP/s across the sixteen shapes.
The portable instance alone, with no instruction-set dispatch at all, is 3.89x the base kernel.
That is the number for any host the AVX variants do not cover: AArch64, other compilers, other
operating systems.

| Per step | before | after | |
| --- | ---: | ---: | ---: |
| attention forward, per layer (B=8, 8 query / 4 KV heads, T=128, d=48) | 18.48 ms | 8.30 ms | 2.23x |
| attention backward, per layer | 53.35 ms | 15.46 ms | 3.45x |
| attention, six layers, both directions | 431 ms | 143 ms | 3.02x |
| AdamW over 9,836,928 parameters (clipping norm, preflight when needed, commit) | 147.0 ms | 65.1 ms | 2.26x |

On the base commit, an attention backward cost 2.9x its forward. Now it costs 1.9x: the
backward makes three products and evaluates the probability adjoint twice, against the forward's
two products. The AdamW row is the update as the trainer calls it at the comparison's shape. The
bounded path skipped the preflight there, as it did on every step of every run in §5. The variants
make up the rest of the gain.

The kernel has no fused multiply-add, by contract. Each term of a product is a rounded multiply and
then a rounded add, which is what makes it bit-identical to the ascending-k loop on every variant
and every compiler, but it needs two instructions where an FMA kernel needs one. The CBLAS seam is
where FMA kernels belong; §4 and §5 run the trainer both ways.

## 4. The training step, before and after

The `baseline` stage trains the 9.84M decoder on the language model's first six batches twice
with each trainer: the base commit's, built from its sources with its flags, and this one. Each
pair is timed back to back, alternating which trainer goes first, and the first step of each run is
dropped. It hashes the full training state (weights and both moments) after the steps and
requires the hashes to agree.

| One training step, B=8, T=128 | before (`75d50d4`) | after | speedup | training state |
| --- | ---: | ---: | ---: | --- |
| BCIR's own kernels | 12,297 ms | 1,702 ms | 7.23x | identical |
| OpenBLAS `sgemm` behind the CBLAS seam | 1,322 ms | 958 ms | 1.38x | identical |

On the base commit, BCIR's own kernels were 9.3x slower per step than BCIR with OpenBLAS; now they
are 1.78x slower. OpenBLAS sped up too: attention and AdamW run on BCIR's own kernels in that
configuration as well.

The `identity` stage trains 20 steps with every build: GCC 13.3, clang 23.1.3 and clang 18.1.3,
each with all three instruction-set variants, plus a repeat of the first build. All ten end in one
state, SHA-256 `6c0419e85d345904...`, with a last loss of 3.5240555245059464 nats on every build.
Their step times:

| ms per step (20 steps, median) | AVX-512 (widest) | AVX2 | portable C |
| --- | ---: | ---: | ---: |
| GCC 13.3 | 1,720 (repeat: 1,708) | 2,516 | 3,462 |
| clang 23.1.3 | 2,229 | 3,022 | 5,482 |
| clang 18.1.3 | 2,306 | 3,024 | 5,507 |

### 4.1 Clang's builds, and the fix (NDT-CLANG)

The compilers agree on every bit and disagree on time. In the table above, clang's builds are
30-34% slower than GCC's with AVX-512, 20% slower with AVX2, and 58-59% slower as portable C.
This matters beyond the comparison, because `NativeDecoder.build` compiles with `$CC`, then
`clang`, then `cc`. On a host with clang installed and `CC` unset, the trainer gets clang's slower
build.

The `kernels` stage, run with each compiler, puts the whole difference in the GEMM. Attention and
AdamW cost about the same under all three compilers. The GEMM times also spread from process to
process under clang, and hardly under GCC. Across five runs of main's kernel, one step's AVX-512
GEMMs took 1.86-2.80 s under clang 18 and 1.21-1.31 s under GCC (the table below). The kernel and
the inputs were the same each time, so the variation came from somewhere else. Three causes, each
visible in the generated code:

- **The packed tile's alignment.** GCC realigns the stack to 64 bytes for the AVX-512 instance
  (`andq $-64, %rsp` in its prologue), so the 16 KiB tile of B sits on a cache-line boundary.
  Clang keeps the ABI's 16 bytes. The tile's offset within a line was then whatever the caller's
  stack depth and the process's randomized stack address made it, and a misaligned 64-byte load
  reads two cache lines. That is the run-to-run spread, and a step that calls the GEMM from a
  different stack depth than a benchmark does can draw a different alignment.
- **The accumulator block.** The store-back loop of a ragged or masked block ran to a bound read
  from the masks. That gave clang a variable index into the accumulators. Clang 18 then kept the
  block on the stack across the depth loop, storing it back after every term.
- **The packing loop.** It chose between `B` and `B^T`, and between data and padding, for every
  element it packed.

The fix changes the code's shape and not its arithmetic. The tile is declared `_Alignas(64)`.
The store-back runs to the block width and tests each column (`if (j<end)`). The packing loop is
split by layout, so each source row is read contiguously and the padding is written once. A
fourth change was measured and dropped: computing the A operand's address once per block slowed
GCC's AVX-512 instance and clang's portable one. The kernel's outputs hash the same as before on
every shape of the kernels stage, under all three compilers and all three instances. The C
harness passes on all three compilers, as C11 and C23, as portable-only and AVX2-capped builds,
and under ASan and UBSan. `clang --analyze` (18 and 23) reports nothing.
[`ndt-clang.json`](../../tools/testing/faults/ndt-clang.json) injects three faults, and the harness
catches each:

- a ragged block's last column left unstored;
- a transposed B's last column dropped from the tile;
- an untransposed B's last depth row dropped from the tile.

`ndt-gem.json`'s NG1, and the same mutant in the oracle's
`test_the_rows_fire_on_wrong_kernels_and_plans`, are re-anchored on the new packing loop and still
caught. Every NDT-GEM and NDT-PERF fault is caught again on the new kernel.

Measured against `fb8d3a0` (main, built with its own flags) on each compiler:

| GEMMs per step (ms): main -> this PR | GCC 13 | clang 23 | clang 18 |
| --- | ---: | ---: | ---: |
| AVX-512 instance | 1,234 -> 1,161 | 1,604 -> 1,396 | 1,905 -> 1,175 |
| ... its five rounds | 1,213-1,306 -> 1,024-1,235 | 1,433-1,710 -> 1,360-1,494 | 1,862-2,796 -> 1,103-1,253 |
| AVX2 instance | 2,008 -> 1,817 | 2,311 -> 2,188 | 2,409 -> 2,129 |
| portable C | 2,965 -> 2,957 | 4,526 -> 4,410 | 5,010 -> 5,048 |
| attention per layer, forward / backward | 7.9 / 14.4 -> 6.9 / 10.5 | 7.8 / 14.3 -> 7.1 / 10.5 | 9.5 / 18.4 -> 7.2 / 10.3 |

These come from the `kernels` stage, alternating the two builds over five rounds with identical
outputs throughout. Outside the GEMM and attention nothing changed, and AdamW reads 65-69 ms
either way.

| One training step (ms): main -> this PR | GCC 13 | clang 23 | clang 18 |
| --- | ---: | ---: | ---: |
| BCIR's own kernels (`baseline` stage, interleaved) | 1,683 -> 1,602 (1.05x) | 2,187 -> 1,774 (1.23x) | 3,214 -> 1,594 (2.02x) |
| with OpenBLAS behind the seam | 991 -> 927 | 947 -> 919 | 957 -> 891 |
| this PR over 20 steps (`identity`): AVX-512 / AVX2 / portable | 1,591 / 2,304 / 3,428 | 1,569 / 2,580 / 5,566 | 1,579 / 2,516 / 5,537 |

The `baseline` and `identity` rows each require one training state across the builds they
compare, and it is the same state as before the fix: `6c0419e85d345904...` after twenty steps on
all ten builds. On AVX-512, the three compilers now train at one speed (1.57-1.59 s per step over
twenty steps). On clang 18, main's step measured 2.3 s in one process and 3.2 s in another; that
spread was the stack-alignment draw, and it is gone. The GEM+ harness's `train.step.ms`, which
builds with the host's default compiler (clang 18 here), reads 3.40 ms against a 0.78 ms floor,
from 4.58 against 1.34 on main. Both the step and the floor moved, because the floor is the
same kernel's best rate.

What NDT-CLANG did not fix is clang's portable instance on x86-64. It is still 1.5-1.7x GCC's,
with or without the fix. Only builds without dispatch run it: macOS and Windows on x86-64, or
`BCIR_TENSOR_NO_DISPATCH`. Its 2x32 block needs sixteen SSE registers for the accumulators
alone, which is every register SSE2 has. A 2x16 block for that target was tried and ran 3x
slower under GCC.

## 5. A 9.84M-parameter decoder trained on BCIR and on PyTorch

### 5.1 Setup

One model, four ways to train it, two tasks. The decoder is the 9.84M-parameter one the 2026-10-10
assessment archived: 9,836,928 parameters, six layers, width 384, eight query and four KV heads of
48, SwiGLU with an inner width of 1,024, RoPE, RMSNorm, tied embeddings, a 260-entry byte
vocabulary. Every backend starts from the same weights (N(0, 0.02) matrices, unit norms, drawn once
and stored), reads the same batches in the same order (8 windows of 128 bytes, 1,024 targets a step)
and uses the same per-step learning rates: AdamW with betas 0.9/0.95, epsilon 1e-8, decoupled decay
0.1, global clipping at 1.0, 50 warmup steps, then a cosine from 5e-4 to 5e-5 over 1,000 steps (§5.7
says why 5e-4).

| Backend | What trains | Kernels |
| --- | --- | --- |
| `bcir` | BCIR's native C trainer, nothing else | its own: the AVX-512 instance on this host, no FMA, bit-identical on every host and compiler |
| `bcir-openblas` | the same trainer, OpenBLAS `sgemm` behind its CBLAS seam | OpenBLAS 0.3.34 (SkylakeX kernels, FMA) for the linears; BCIR's own for attention, norms and AdamW |
| `torch-eager` | `HostedLlama`, `torch.optim.AdamW` defaults | PyTorch 2.13.0+cu130 on the CPU: MKL 2024.2, oneDNN 3.12, AVX-512 |
| `torch-compiled` | the same modules, forward and loss through `torch.compile(fullgraph=True)`, fused AdamW | the same libraries plus Inductor's generated code |

Everything runs on one compute thread (`OMP`/`MKL`/`OpenBLAS` threads 1, `torch.set_num_threads(1)`)
in FP32, one process per run so each peak resident set is its own, one run at a time. A step's
time is the wall time of everything that trains: building the batch in Python, the forward, the
loss, the backward, clipping and the update. Evaluations are outside the training clock.

The **language model** (`lm`) trains on the repository's own tracked Markdown at the base commit
(404 documents under `docs/` and `training/`, this experiment's measurements excluded), split by
a hash of each document's path: 368 documents for training (4.33 million bytes), 20 for
validation (107 thousand) and 16 for test (750 thousand). It is scored on 64 validation and 128
test windows of 128 bytes. The scores are cross-entropy in nats per byte (and bits per byte),
next-byte accuracy, and **word completion**. Of the 1,532 test words of four or more letters,
word completion is the share whose every letter after the first the model predicts exactly,
which is what greedy completion from the first letter would produce.

The **copy** task (`copy`) streams problems of eight random lowercase letters, `=`, the same eight
letters and a newline. The first eight letters are unpredictable, so their training targets are
a reserved don't-care byte (prompt-loss masking expressed in the targets, identical for every
backend). It is scored on 2,000 held-out strings, none of which occurs among the 111,112 the
training stream holds: cross-entropy and accuracy over the nine answer bytes, and exact match, the
share of strings whose eight letters and newline are all predicted.

### 5.2 The language model

| Backend | ms per step (median) | tokens/s | training time | validation CE | test CE (nats/byte) | test next-byte accuracy | test words completed | peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bcir` | 1,726 | 593 | 28.7 min | 1.5003 | 1.6319 | 53.65% | 107 of 1,532 (6.98%) | 544 MiB |
| `bcir-openblas` | 963 | 1,064 | 15.9 min | 1.5003 | 1.6319 | 53.67% | 109 (7.11%) | 544 MiB |
| `torch-eager` | 774 | 1,323 | 12.9 min | 1.5001 | 1.6319 | 53.67% | 109 (7.11%) | 1,140 MiB |
| `torch-compiled` | 712 | 1,439 | 11.8 min | 1.5001 | 1.6318 | 53.70% | 109 (7.11%) | 1,056 MiB |

The four trainers produce the same model, to within what the evaluation can resolve. Starting
from the same loss (5.5779 nats on the first batch), the four training losses agree with
`torch-eager`'s to 1e-6 over the first ten steps and to 3e-6 over the first hundred. Rounding
differences then accumulate, but the curves never separate: the largest per-step difference over
the 1,000 steps is 0.0082 nats (`bcir`), and over the last hundred steps the mean difference is
0.0004-0.0005 nats. The validation cross-entropies are identical to four decimals through step
300 and within 0.001 nats of each other at every one of the 21 evaluations. The final
test cross-entropies differ by at most 0.0001 nats per byte (2.354 bits per byte), and the
test words completed by two of 1,532.

Because the curves coincide, time to a quality is the step time times a step count that is the
same for every backend:

| Validation cross-entropy reached | at step | `bcir` | `bcir-openblas` | `torch-eager` | `torch-compiled` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2.0 nats per byte | 400 | 687 s | 383 s | 312 s | 287 s |
| 1.75 | 550 | 943 s | 528 s | 428 s | 394 s |
| 1.6 | 750 | 1,288 s | 718 s | 582 s | 534 s |
| 1.55 | 850 | 1,462 s | 813 s | 658 s | 604 s |

BCIR's own kernels reach each level in 2.2x eager PyTorch's time and 2.4x compiled PyTorch's; with
OpenBLAS behind the seam, 1.23x eager's and 1.34x compiled's. Compilation costs `torch-compiled`
6.9 s across its first two steps (`torch-eager` 1.9 s), which its 8% shorter steps repay after 80
steps (57 on copy). One step is 1,024 target bytes, so the 1,000 steps train on 1.02 million bytes:
windows drawn at random offsets from the 4.33-million-byte training split, about a quarter of it.

### 5.3 The copy task

| Backend | ms per step (median) | training time | first evaluation at >= 99% exact match | training time to it | final exact match (2,000 held out) | final answer CE | peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bcir` | 1,725 | 28.7 min | step 500 | 860 s | 100% | 0.0010 | 607 MiB |
| `bcir-openblas` | 954 | 15.8 min | step 250 | 233 s | 100% | 0.0007 | 612 MiB |
| `torch-eager` | 788 | 13.1 min | step 300 | 235 s | 100% | 0.0008 | 1,249 MiB |
| `torch-compiled` | 713 | 11.9 min | step 300 | 218 s | 100% | 0.0008 | 1,235 MiB |

Each model learns the task completely. After 1,000 steps, each copies all 2,000 held-out strings
exactly, every letter and the newline, and none of those strings was in its training stream. The
held-out exact match over training:

| Step | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 600 | 1,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bcir` | 0% | 0% | 0% | 2.3% | 52.4% | 91.7% | 99.1% | 100% | 100% |
| `bcir-openblas` | 0% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| `torch-eager` | 0% | 0% | 99.0% | 100% | 99.95% | 99.3% | 100% | 100% | 100% |
| `torch-compiled` | 0% | 0.15% | 100% | 99.0% | 100% | 100% | 100% | 99.95% | 100% |

All four learn the task, but not at the same step, and that is a property of the task rather than of
the frameworks. Copying is a phase transition. The loss sits on a plateau, with each letter
predicted at chance, until attention learns to fetch the letter being copied; then the loss drops,
within 50 steps for three of the runs and 150 for `bcir`. When the escape happens depends on
rounding. The losses agree with `torch-eager`'s to 2e-5 over the first hundred steps (1.4959 nats at
step 100 on all four). By step 200 they differ in the second decimal: 1.2359, 1.3200, 1.2677 and
1.2272 nats. The two `bcir` runs mark the spread's two ends. They are the same trainer on the same
data, differing only in the rounding of their GEMMs (BCIR's kernel adds every term separately,
OpenBLAS fuses multiply-adds and blocks the sum differently), and they escape 250 steps apart. One
seed per backend cannot rank the backends on this task's time to quality. The per-step times can be
ranked, and they are those of §5.2.

### 5.4 Memory

BCIR's processes peak at about half of PyTorch's: 544 MiB against 1,056-1,140 MiB for the language
model, and 607-612 MiB against 1,235-1,249 MiB for copy. The peak is the whole process's resident
set (`ru_maxrss`): Python, NumPy, the training stream, the evaluations and the framework. Every
backend evaluates in the same 8-window batches and computes the same metrics in NumPy, so the
difference is the frameworks. On one side is BCIR's trainer and its arena of 335 MiB, sized
before the first step and never grown: 150 MiB of parameters, gradients and both moments, 163 MiB
of retained activations, 22 MiB of scratch. On the other are PyTorch's libraries, its autograd
graph and its caching allocator. Before the first step, BCIR's processes held 377-398 MiB and
PyTorch's 678-702 MiB. Copy's evaluation set is 288 windows against the language model's 128,
which is why those runs peak higher.

### 5.5 Where the time goes

The `breakdown` stage times one step at the comparison's shape, part by part.

- **For BCIR**, it times the binding's stages, then each kernel family the step calls. Each family
  is timed alone, through the library's own entry points, with the calls one step makes and the
  provider the run uses.
- **For eager PyTorch**, it times the step's phases, then splits the forward and backward with
  `torch.profiler`: self CPU time per operator, summed by family.

These are medians of five rounds on one thread. BCIR is built with GCC 13 and the kernel the
comparison ran.

| One step (ms) | BCIR, own kernels | BCIR + OpenBLAS | eager PyTorch |
| --- | ---: | ---: | ---: |
| GEMMs: forward + backward (60.4 GFLOP) | **1,329** (429 + 900) | 566 (184 + 382) | **476** (`aten::mm`) |
| attention: forward + backward | 141 (51 + 90) | 141 | 60 (fused SDPA, forward + backward) |
| RMSNorm, RoPE, SiLU and the SwiGLU VJP | 120 | 111 | 174, for every other operator of the forward and backward |
| loss, residuals, embedding, finiteness checks | -13, by difference | -1 | (included above) |
| optimizer | 55 (AdamW, with the norm for clipping) | 69 | 55 (AdamW 46, clipping 9) |
| **one step** | **1,632** | **885** | **762** |

PyTorch's lead is its matrix multiply. MKL's FMA kernels run the step's 60.4 GFLOP in 476 ms
(127 GFLOP/s). BCIR's own kernel takes 1,329 ms (45 GFLOP/s), and the 853 ms between them is
nearly all of the 870 ms by which the steps differ. Attention costs BCIR another 81 ms against
PyTorch's fused kernel. Everything else together is faster on BCIR: 162 ms against 229. With
OpenBLAS behind the seam, the GEMM gap shrinks to 90 ms.

How much of the GEMM gap BCIR's kernel can close is bounded by its own contract. One rounded
multiply and one rounded add per term take two instructions where an FMA takes one, so even a
perfect kernel under the contract would issue twice the vector arithmetic of MKL's. BCIR's
kernel runs at 35% of MKL's rate; at equal efficiency it could reach at most about half. Closing
the rest means giving up bit-identity across hosts, which is what the CBLAS seam is for.

### 5.6 What this comparison is not

It is one host, one compute thread, FP32, one seed per backend and one model shape. Evaluation is
not on the training clock (each backend's evaluations are recorded in its run). BCIR's evaluation
runs the native forward through the same binding: the 64 validation windows take 4.7 s there and
1.8 s on eager PyTorch. Nothing here is evidence about multi-threaded or GPU training, larger
models, or mixed precision, and PyTorch's CPU build was used as shipped. A faster per-step time in
one direction or the other would not change the models: they are the same model.

### 5.7 Why the learning rate is a quarter of the plan's

The harness was written with a peak of 2e-3. At that rate, with 1,024 targets per update, no model
formed the attention circuit a structured task needs within the steps it was given, on any of
three tasks; the runs were stopped and their curves are in
[`explorations.json`](measurements/2026-10-11/explorations.json) (PyTorch only -- they decided the
configuration before any comparison ran):

| Task (2,000 held-out problems) | Stopped at | Held-out answer | What it had learned |
| --- | --- | --- | --- |
| addition, `aaa+bbb=` then four sum digits least significant first, every target trained | step 700 of 3,000 | exact match 0.15% | the format and the carry-out digit |
| the same with the operand digits masked out of the loss | step 700 of 3,000 | exact match 0.15%; cross-entropy 1.395 nats per answer byte | the newline and the carry-out digit (99%): 1.382 is exactly three uniform digits out of five bytes, so the three mod-10 digits were still chance |
| recall, six key/value pairs then a query | step 400 of 1,000 | exact match 9.65% | the format; the value at chance (10%) |
| copy, eight letters then the same eight | step 300 of 1,000 | exact match 0% | the format; each letter at chance (1/26) |

The addition model did route operand digits to the answer -- it got the carry-out digit right, which
needs both hundreds digits -- so routing as such was learnable; the mod-10 digit table was not, in
budget. At a peak of 5e-4 the same copy model, data and batches move from 0% exact match at step
250 to 99.0% at step 300 and 100% by step 350 (the `torch-eager` run of §5). Every run in §5 uses
`--lr-scale 0.25`: a peak of 5e-4, a floor of 5e-5, the same warmup and cosine. The 2e-3 plateau is
a property of that rate at this batch size on these tasks, measured on PyTorch; nothing here says
a framework caused it.

Recall was also run at the comparison's rate, on eager PyTorch, for its full 1,000 steps. It
learned, but not to completion. Held-out exact match climbed from chance (10%) to 32.0%, with
answer accuracy at 66.0%: the newline every time, and the value in about a third of problems. A
task that ends partly learned has no time to quality to compare, so recall was not trained on the
other backends, and its run is in `explorations.json` with the others.

### 5.8 Longer contexts, and more threads

The 2026-10-10 plan asks attention for "long-context memory/time curves" and decoder training "at
64/128/256+ context", and the matrix library for "one and multiple compute threads". The
`context` stage measures both on this update's kernels (GCC 13), at the comparison's 1,024 tokens
per call or step: the batch halves as the context doubles, so the linears do the same work at
every context and only attention's work grows. Each configuration runs in its own process, twice,
in alternating order, for 12 steps; its step time is the median of the 20 steps after each run's
first two. Every run starts from the comparison's weights at its learning rates.

| One attention layer (ms) | BCIR forward | BCIR backward | PyTorch forward | PyTorch backward | BCIR / PyTorch | BCIR's stored probabilities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T=64, batch 16 | 4.8 | 7.2 | 4.0 | 9.7 | 0.88x | 2 MiB |
| T=128, batch 8 | 7.5 | 12.2 | 5.2 | 13.2 | 1.07x | 4 MiB |
| T=256, batch 4 | 14.1 | 21.4 | 7.9 | 22.7 | 1.16x | 8 MiB |
| T=512, batch 2 | 23.9 | 32.9 | 10.6 | 32.8 | 1.31x | 16 MiB |
| T=1024, batch 1 | 44.1 | 61.6 | 13.8 | 42.6 | 1.87x | 32 MiB |

The PyTorch columns time `HostedAttention`'s whole path on the same values: the four KV heads
repeated to eight, `scaled_dot_product_attention` (on this CPU,
`_scaled_dot_product_flash_attention_for_cpu` and its backward), and autograd through both. §5.5
counts SDPA's own kernels only, which is why PyTorch's attention there is smaller. The two
frameworks' outputs and gradients agree within 1.1e-5. At a fixed number of tokens, BCIR's
attention computes and stores every probability, so its work grows with T: from T=64 to T=1024
its time per layer grows 8.9x. PyTorch's flash kernel computes the scores tile by tile and keeps
one log-sum-exp per row, and its time grows 4.1x. BCIR's kernel is the faster at T=64 and 1.87x
slower at T=1024.

| One training step (ms), 1,024 tokens | BCIR own kernels | BCIR + OpenBLAS | eager PyTorch | BCIR / PyTorch | BCIR's arena | peak resident memory, BCIR / PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T=64, batch 16 | 1,545 | 865 | 734 | 2.11x | 323 MiB | 521 / 1,052 MiB |
| T=128, batch 8 | 1,569 | 893 | 739 | 2.12x | 335 MiB | 533 / 1,042 MiB |
| T=256, batch 4 | 1,659 | 1,012 | 818 | 2.03x | 359 MiB | 557 / 1,046 MiB |
| T=512, batch 2 | 1,894 | 1,186 | 932 | 2.03x | 407 MiB | 605 / 1,044 MiB |
| T=1024, batch 1 | 2,213 | 1,549 | 1,045 | 2.12x | 503 MiB | 701 / 1,044 MiB |

At every context the training losses of both BCIR backends agree with eager PyTorch's within
7.9e-7 nats over all 12 steps, on the same batches: the same model at every context. From T=64 to
T=1024, BCIR's step grows 43% and PyTorch's 42%, so BCIR's own kernels stay at 2.0-2.1x PyTorch's
step. In milliseconds, BCIR's step grows by 667 and PyTorch's by 311, and the difference is
attention. BCIR with OpenBLAS grows by 683 ms: OpenBLAS replaces the linears, not the attention,
so its lead over BCIR's own kernels is spent as the context grows (1.18x PyTorch's step at T=64,
1.48x at T=1024). BCIR's arena grows by exactly the probabilities it keeps for the backward, six
layers of B x 8 x T x T floats (12 MiB at T=64, 192 MiB at T=1024), and its peak resident memory
grows with it, from 521 to 701 MiB. PyTorch's stays at about 1,040 MiB, still 1.5x BCIR's at
T=1024. The per-process peak is the address space's own high-water mark (`VmHWM`): `ru_maxrss`
would carry the parent's resident set across `fork` and `exec`, and the parent has PyTorch
loaded.

| Training step at T=128 (ms) | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| BCIR's own kernels | 1,569 | (single-threaded) | (single-threaded) |
| BCIR + OpenBLAS | 893 | 706 | 542 |
| eager PyTorch | 739 | 434 | 257 |

BCIR's own kernels run one thread. On all four of this VM's cores, eager PyTorch trains a step in
257 ms, 6.1x faster than BCIR's own kernels and 2.1x faster than BCIR with a four-thread OpenBLAS.
PyTorch gains 2.9x from four threads. BCIR with OpenBLAS gains 1.65x, because only its linears
are threaded: attention, the norms and AdamW stay on BCIR's single-threaded kernels. The losses
agree with PyTorch's within 5.6e-7 on two and four threads as well.

## 6. The 2026-10-10 harness on this host, before and after

The assessment's own `kernels` and `attention` stages were run unchanged on this host, from a
checkout of `75d50d4` (before) and from this tree (after). The sides alternated over three
rounds; the medians are below, with OpenBLAS and PyTorch from the after side's runs (the
libraries are the same on both sides). The harness builds its "portable C" with
`-O3 -march=native -fno-math-errno -ffp-contract=off`. On this host that build ran the base
kernel's untransposed-B shapes at half the speed of the build the trainer shipped with (§1,
item 5). So the before column is not the shipped trainer's cost, and these speedups are
larger than the training step's. §3 and §4 are the before/after to read for training.

| GEMM (the assessment's shapes) | before (ms) | after (ms) | speedup | OpenBLAS | PyTorch (MKL) | after / PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 x 64 x 48 | 0.193 | 0.020 | 9.45x | 0.014 | 0.005 | 4.01x |
| 256 x 384 x 384 | 28.388 | 2.404 | 11.81x | 0.986 | 0.807 | 2.98x |
| 256 x 1024 x 384 (B transposed) | 11.685 | 5.196 | 2.25x | 1.852 | 1.631 | 3.19x |
| 384 x 384 x 256 | 35.519 | 1.889 | 18.81x | 0.749 | 0.646 | 2.92x |

| Attention forward (B=4, 8 query / 4 KV heads, d=48) | before (ms) | after (ms) | speedup | PyTorch SDPA | after / SDPA |
| --- | ---: | ---: | ---: | ---: | ---: |
| T=64 | 2.079 | 1.584 | 1.31x | 0.581 | 2.73x |
| T=128 | 8.737 | 4.793 | 1.82x | 1.785 | 2.68x |
| T=256 | 34.987 | 12.995 | 2.69x | 4.824 | 2.69x |

Per call, the assessment's GEMMs now take 2.9-4.0x PyTorch's MKL time, down from 7-43x before.
Its attention forward takes 2.7x PyTorch's fused SDPA kernel, down from 3.8-6.2x. Each before
ratio is against the PyTorch time measured in its own run. Both kernels remain slower than the
vendor ones, and for a stated reason. MKL and OpenBLAS use FMA and their own summation order,
and SDPA computes attention in tiles without storing the probabilities. BCIR's kernels keep one
rounded multiply and one rounded add per term in a fixed order and store every probability.
That is what makes them bit-identical on every host and compiler. The largest error against
FP64 is the same before and after (1.2e-06 for the GEMMs, 4.8e-08 for attention), as it must
be: the outputs are the same.

## 7. What remains

| 2026-10-10 item | After this update | Left |
| --- | --- | --- |
| P1 C training optimizer | x86-64: AVX2 and AVX-512 passes chosen at run time, bit-identical, the atomic refusal kept, the preflight skipped only under a proof | No NEON instance (AArch64 runs the portable C its compiler vectorizes); no native-ARM timing -- CI's AArch64 job runs the bit-identity tests, not a benchmark |
| P1 Attention + VJP | Both directions through the register-blocked kernel under causal masks; the backward in place, no workspace | The probabilities are still O(B·H·T²) arena; a tiled/online-softmax forward with its VJP, and an attention provider seam, are the long-context work: at T=1024 a layer takes 1.87x PyTorch's flash path, and the arena holds 192 MiB of probabilities (§5.8) |
| P2 Matrix library portfolio | SIMD register blocks, B packed per tile, every transpose and `beta`, one contract across variants | No FMA by contract -- the CBLAS seam is where vendor FMA kernels belong; no persistent packed weights; no shape-aware choice between native and vendor kernels; one thread (on four, PyTorch's step is 6.1x faster, §5.8) |
| P2 Vector math + fusion | Not started | Now measured (§5.5): RMSNorm, RoPE, SiLU and the SwiGLU VJP take 120 ms of BCIR's 1.63 s step, 102 ms of it in the SiLU forward and the SwiGLU VJP, which call libm's `expf` once per element (70 ms built by clang). A vectorized exponential must reproduce libm's results, or the contract must say otherwise; that choice is the item |
| P2 Training buffer planning | Unchanged (NDT-MEM open) | The arena's transient headroom (25% at the bench shape) |
| (found here) The compiler | Clang's builds aligned with GCC's on AVX-512 and AVX2 (NDT-CLANG, §4.1); §3 and §5 measured GCC's build of #814's kernel | Clang's portable C on x86-64 (no dispatch: macOS and Windows on x86-64) is 1.5-1.7x GCC's |
| P1 Async graph, P1 StreamPack CRC, P2 ASN.1 arrays | Not in this update | As the assessment states them |

## 8. Reproduction and validation

Host: a four-core Intel Xeon VM at 2.1 GHz (AVX-512, one thread per core), 15 GiB, Linux 6.18; GCC
13.3, clang 18 and clang 23.1.3; Python 3.11 with PyTorch 2.13.0+cu130 (the default PyPI wheel, run
on the CPU alone; it reports MKL 2024.2, oneDNN 3.12 and AVX-512), NumPy 2.2.6 and scipy-openblas32
0.3.34 -- an experiment environment, not a BCIR dependency. One compute thread throughout. No GPU,
native ARM, PMU or energy measurement is claimed; timings are this VM's, interleaved where two
builds are compared, and never a CI threshold.

```bash
H=docs/research/measurements/2026-10-11/train_comparison.py
python $H prepare --output DIR
for task in copy lm; do for backend in torch-eager torch-compiled bcir-openblas bcir; do
  python $H train --data DIR --task $task --backend $backend --eval-every 50 --lr-scale 0.25
done; done
python $H kernels   --data DIR --rounds 3         # §3: 75d50d4 against this tree, every GEMM shape
python $H baseline  --data DIR --steps 6          # §4: 75d50d4 against this tree, identical state
python $H identity  --data DIR --steps 20 --cc gcc --cc CLANG23 --cc CLANG18
python $H breakdown --data DIR --cc gcc --cc CLANG23 --cc CLANG18   # §4.1, §5.5
python $H context   --data DIR                    # §5.8: T=64..1024, and 2 and 4 threads
python $H summarize --data DIR
# §4.1: main (fb8d3a0, built with its own flags) against this tree, once per compiler
OLD=(--base fb8d3a0 --old-flags "-O3 -ffp-contract=off -fno-math-errno")
for CC in gcc CLANG23 CLANG18; do
  python $H kernels  --data DIR --cc $CC --rounds 5 "${OLD[@]}"
  python $H baseline --data DIR --cc $CC "${OLD[@]}"
done
python tools/perf/gemplus_baseline.py --compare --group train   # train.step.ms, §4.1
python docs/research/measurements/2026-10-11/march_check.py --output DIR   # §1, item 5
# §6: the 2026-10-10 harness's own stages, from a checkout of 75d50d4 and from this tree
python docs/research/measurements/2026-10-10/run_performance_tests.py --stage kernels --output OUT
python docs/research/measurements/2026-10-10/run_performance_tests.py --stage attention --output OUT
```

Run one job at a time ([AGENTS.md](../../AGENTS.md)). `prepare`, `identity` and `kernels` need only
Python and a C compiler; `baseline` also needs `scipy-openblas32`; `train`, `breakdown`, `context`
and `summarize` need NumPy, PyTorch and `scipy-openblas32`. The records committed beside the harness
are this run's: [`prepare.json`](measurements/2026-10-11/prepare.json), the eight runs
([`runs/`](measurements/2026-10-11/runs/): every step's loss, gradient norm and wall time, every
evaluation, the final metrics, resident memory, the final state's SHA-256 and the environment),
[`kernels.json`](measurements/2026-10-11/kernels.json),
[`baseline.json`](measurements/2026-10-11/baseline.json),
[`identity.json`](measurements/2026-10-11/identity.json),
[`breakdown.json`](measurements/2026-10-11/breakdown.json),
[`summary.json`](measurements/2026-10-11/summary.json), the context and thread sweep
([`context.json`](measurements/2026-10-11/context.json)), the assessment's stages before and after
([`stages/`](measurements/2026-10-11/stages/), three rounds each), the build-flag check
([`march.json`](measurements/2026-10-11/march.json)) and the configurations §5.7 records but did not
use ([`explorations.json`](measurements/2026-10-11/explorations.json)). The top-level `identity` and
`breakdown` records are main's kernel, the one the eight runs trained with;
[`ndt-clang/`](measurements/2026-10-11/ndt-clang/) holds the same two stages on this tree's kernel
and the `kernels` and `baseline` stages per compiler against main.

**Validation.** The kernel change was checked on its final source the way this repository checks a
kernel. The native trainer's C harness passes under GCC 13, clang 18 and clang 23: as C11 and C23,
as portable-only and AVX2-capped builds, as `NativeDecoder.build`'s `-O3 -fno-math-errno` build,
and under ASan with UBSan. `tools/c/analyze_hosted.sh` reports no diagnostics under clang 18 or
clang 23. Every fault in [`ndt-clang.json`](../../tools/testing/faults/ndt-clang.json),
[`ndt-gem.json`](../../tools/testing/faults/ndt-gem.json) and
[`ndt-perf.json`](../../tools/testing/faults/ndt-perf.json) is caught by its own check (RED).
`tools/c/check_runtime.sh`, the oracle's quick tier and the PyTorch differential
(`tools/models/test_native_decoder.py`) pass. So does every leg of `tools/local/check_latest.sh` on
LLVM, Clang and MLIR 23.1.3 and Node 24; its thorough tier found one oracle test still anchored on
the old packing loop, which is re-anchored here. `tools/perf/ab_audit.py` against `fb8d3a0`, with a
fresh worktree of this commit as its after tree, passes: every exact row and every correctness
digest is unchanged, every ratio row is inside its band, and the `train.step.ms` wall row reads
1.37x faster. A first run against the working checkout flagged `verify.plan.scope.overhead`, a
timing ratio over code this update does not touch; six more runs of identical code, three in each
tree, read it from 0.49 to 1.03 on this VM. The pull request that adds this report quotes each
command's output.
