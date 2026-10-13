# NDT-MEASURE: the native trainer re-measured after the pool, the fused contract and the BCIR exponential (2026-10-12)

The [training-acceleration roadmap](BCIR_TRAINING_ACCELERATION_ROADMAP.md) §4 names five
instruments and asks that each slice be measured with them before and after. This report is the
"after" for the three slices that landed on 2026-10-12 -- NDT-THREADS (a deterministic pool),
NDT-FMA (the fused GEMM contract, v2) and NDT-EXP (a BCIR-owned exponential under SiLU, the SwiGLU
VJP and the softmax) -- on the same decoder, data, optimizer and schedule as the
[2026-10-11 report](BCIR_NATIVE_TRAINING_PERFORMANCE_2026-10-11.md): the 9,836,928-parameter
byte-level decoder (vocab 260, width 384, 8 heads over 4 KV heads, 6 layers, FF 1024; batch 8 x 128
tokens, AdamW, 50 warmup steps and a cosine over 1,000 steps at a quarter of the plan's rates) trained
on the repository's documentation at the 10-11 report's pinned commit. §5 reads the result against
the roadmap's §2 budget, row by row, and names what the next slices are made of.

**Host.** A four-vCPU Intel Xeon VM (AVX-512, no `avx512_bf16`, no AMX, a shared host), 15 GiB,
Linux 6.18, GCC 13.3, `-O3 -ffp-contract=off -fno-math-errno`, one process at a time, tree at
`38afac1`. It is the same class of container as the 10-11 report's but not the same machine: this
one reports 2.80 GHz where that one reported 2.10 GHz, and it is the slower of the two on the one
measurement both took the same way (the v1 one-thread run, §4: 1,951.9 ms a step here, 1,726
there). So **every 10-11 number below is a reference across hosts, and every comparison this report
makes is a same-session pair**, measured one after the other on this machine. **What this host cannot
do:** PyTorch is not installable here (`download.pytorch.org` is refused by the proxy), so the PyTorch
rows are the 10-11 report's, and the interleaved BCIR-against-PyTorch comparison at equal thread
counts that the roadmap's protocol asks for is the next run, on a host with both. Every time below is
an indicative wall measurement (TMSAO-3 at best), never a certificate; the exact rows -- the digests,
the mismatch counts, the ulp bound -- are the claims.

## 0. The answer

- **Nothing moved a bit it was not declared to move.** The v1 state at 1, 2, 3 and 4 threads is the
  10-11 serial state, on every instruction-set variant; the 1,000-step v1 run on three threads ends
  in the 10-11 one-thread run's state, byte for byte (`aed5ffbf…`). Each v2 contract is its own
  state, identical at every thread count and on both vector widths.
- **Three threads make the step 1.6-2.0x faster than one (1.6-1.7 in the kernel breakdown, 2.0
  over the 1,000-step v1 run) and four threads 1.9-2.0x; the two v2 contracts take another
  1.2-1.3x.** The 1,000-step language-model run goes from 989 ms a step under v1 on three
  threads to 819 under `fused+exp` (758 on four); the held-out model is
  the same to four decimals (validation 1.5003, test 1.6319 nats per byte; the 10-11 PyTorch run's
  1.5001 / 1.6319).
- **The expectation table was wrong where it assumed the arithmetic was the kernel.** The fused
  contract gains x1.24 on the step's GEMMs, not the 2x of its instruction count, and
  four threads give x2.03 on the step, not 3.4x. The residual is named in §5: the
  GEMM's packing and register block, the 64-row task partition behind one barrier per GEMM, the
  reductions the pool leaves serial -- and one defect of NDT-EXP's own, found here: the v2 softmax
  calls the exponential element by element inside its row loop, so the attention forward costs a
  quarter more under `exp` than under libm (§3) -- fixed the same day as NDT-ATTN's first item: the
  row through the vector instance, the same values, 79 to 55 ms (§3).
- **OpenBLAS is now a BCIR contract, not a trust.** Reading the GEMM gap found that its `sgemm`
  computes a K-blocked fused chain, bit for bit, on every one of the trainer's shapes. Declared as
  the blocked fused contract (v3, NDT-KBLOCK) with a differential admission, it trains the same
  9.84M state on BCIR's own kernel and on OpenBLAS, at one and at four threads -- and the admitted
  provider trains at 616 ms a step on four threads once its workers sleep between calls (§7).
- **BCIR's own kernel is at the provider's rate (NDT-BLOCK, §8).** The gap §7 left to OpenBLAS
  was the schedule, not the arithmetic: with the v3 instances rewritten as intrinsic register
  blocks (8 x 48 on AVX-512) over a tile packed by vector, the kernel reads 99 GFLOP/s
  work-weighted at one thread against OpenBLAS's 89 in the same process and 219 on the pool at
  four against its 219, bit-identical to the chain on every instance; the step's GEMMs
  569 ms at one thread (952 before), and the 1,000-step run 516 ms a step on four
  threads (753), the same state `d7fc618c…`.
- **The reductions are on the pool (NDT-REDUCE, §9).** The loss's rows, the gamma VJP's row sums
  and the embedding scatter run as tasks in the serial order (no bit moves, under any contract);
  the update's first pass is one task per fixed chunk, and the clipping norm has a declared
  fixed-chunk contract, the `reduce` bit, emulated exactly by the oracle. In three same-session
  pairs at four threads the update reads 40.7 / 49.7 / 47.5 ms under `kblock+exp` against
  31.8 / 47.2 / 39.6 under `kblock+exp+reduce` (-9.0 / -2.6 / -7.9; 53 in §8), the gamma VJP row
  11.0 / 14.3 / 12.4 against 6.7 / 8.8 / 11.8; the step is unresolved on this day's host
  (its unchanged GEMM rows drifted 20% within the session), and the 1,000-step
  `kblock+exp+reduce` run's 625.8 ms a step on four threads (516 the day before) is that
  host's state, the same held-out model, its own state.

## 1. Identity: the state is the state, at every thread count and under each contract

| Run | Result |
|---|---|
| the `identity` stage, v1, `--threads 1,2,3,4` (widest, AVX2 and portable builds, and a repeat) | 16 rows, one state: `6c0419e85d345904…`, the 10-11 report's serial digest ([identity-threads.json](measurements/2026-10-12/identity-threads.json)) |
| `fused` (avx2, widest builds, threads 1, 2, 3, 4) | 12 rows, one state: `43e864317981a2f5…` | 
| `exp` (avx2, widest builds, threads 1, 2, 3, 4) | 12 rows, one state: `aaa9998c065d86d3…` | 
| `fused+exp` (avx2, widest builds, threads 1, 2, 3, 4) | 12 rows, one state: `b18d89fa9e9abd43…` | 
| the three v2 states against each other and against v1's | four distinct states |
| the 1,000-step v1 run at 3 threads ([lm-bcir-t3.json](measurements/2026-10-12/runs/lm-bcir-t3.json)) | state `aed5ffbfd54d746b…` = the 10-11 one-thread run's state ([10-11 lm-bcir.json](measurements/2026-10-11/runs/lm-bcir.json)) = this session's one-thread run's ([lm-bcir.json](measurements/2026-10-12/runs/lm-bcir.json)): 1,000 steps on three threads, the serial state byte for byte |
| the 1,000-step `fused+exp` runs at 3 and at 4 threads | one state of their own (`eb43c96d…`, the same at three and at four threads) |

The contract rows are [identity-contracts.json](measurements/2026-10-12/identity-contracts.json):
36 rows (three contracts, the widest and AVX2 builds and a repeat, four thread counts), three states.
The pool changed no arithmetic; each v2 contract is one declared change and its own state. The v2
rows omit the portable build on purpose: its fused instance is libm's `fmaf` per term, a correctness
path for hosts without a hardware FMA that takes hours for 20 steps here, not a speed path -- the C
harness holds it to the AVX instances bit for bit (`gemm_fused_contract`), which is the claim that
matters for it.

## 2. Kernels

**GEMM** ([gemm-rates-threads.json](measurements/2026-10-12/gemm-rates-threads.json); work-weighted
over the step's 13 GEMM shapes, best of 5 calls per cell, one process; OpenBLAS is
`scipy_openblas32` 0.3.34, the comparison's `bcir-openblas` backend, at its own thread count):

| GFLOP/s | 1 thread | 2 threads | 4 threads |
|---|---:|---:|---:|
| BCIR v1 (one rounded multiply and one rounded add per term) | 43.5 | 64.8 | 77.9 |
| BCIR fused v2 (one `fmaf` per term) | 55.6 | 81.2 | 111.5 |
| OpenBLAS `sgemm` | 87.1 | 144.9 | 240.7 |

A second one-thread run of the same shapes ([gemm-fused-rates.json](measurements/2026-10-12/gemm-fused-rates.json),
best of 7) read v1 46.9 and fused 66.9 GFLOP/s (x1.43; every shape between x1.33 and x1.51). The fused
instance's gain over v1 is 1.3-1.5x at one thread, short of the arithmetic's 2x because the kernel
is not only its multiply-adds: the packing of the B tile, the loads of A and the register block's
stores are the same under both contracts. Against OpenBLAS the kernel is at 0.64x on one thread and
0.46x on four: the register block and the packing are the one-thread gap, and the pool's scaling is
the rest (2.0x on four threads for the fused kernel, 2.8x for OpenBLAS) -- the GEMM's tasks are 64
rows by one column block, so a 1024 x 384 x 384 product has 16 x 6 of them and four threads share 96
tasks of unequal cost behind one barrier.

**The exponential** (`bcir_tensor_exp_sweep`, every binary32 in [-104, 89] at stride 2^8):
8,749,570 inputs, 0 where two instruction-set instances differ, 1 ulp at most from `exp` in double
rounded to binary32, identical from the gcc and clang builds (`train.exp.mismatch` 0,
`train.exp.ulp.max` 1, both exact rows in the GEM+ `train` group).

## 3. The step, by kernel

The `bcir-breakdown` stage (new in this slice): one step of the comparison's decoder through the
binding, then every kernel family the step calls, timed alone through the entry points the trainer
itself calls under the run's pool and contracts -- the pooled kernels, `bcir_tensor_mm_fused_pool`
under `fused`, the v2 SiLU, SwiGLU VJP and softmax under `exp` -- with the calls one step makes
(paired rounds: each round times the whole step through the binding and then each kernel family
alone, so "the rest" -- the step less its kernels: the loss, the embedding scatter, the residual adds
and gate multiplies, the gradient-norm pass, the binding -- is a difference taken under one drift of
the shared host rather than between medians a minute apart; the median of 5 timed rounds after one
warm-up). The records are [breakdown-bcir-{contract}-t{threads}.json](measurements/2026-10-12/).

**By contract, one thread** (the 10-11 column is that report's GCC breakdown on its host, for the
shape of the step, not for the absolute times; its kernels were timed one family after another, so
its rest row is not comparable):

| One step (ms) | 10-11 host, v1 | v1 | `fused` | `exp` | `fused+exp` |
|---|---:|---:|---:|---:|---:|
| GEMMs, forward | 428.8 | 494.3 | 390.7 | 505.1 | 367.0 |
| GEMMs, backward (dx and dW) | 899.9 | 921.4 | 750.7 | 985.5 | 714.3 |
| attention, forward | 50.7 | 62.2 | 67.4 | 79.0 | 84.7 |
| attention, backward | 89.8 | 88.8 | 95.4 | 81.7 | 82.9 |
| SiLU | 48.1 | 57.3 | 64.1 | 9.5 | 7.4 |
| SwiGLU VJP | 53.7 | 62.6 | 66.3 | 13.3 | 10.1 |
| RMSNorm | 4.3 | 8.3 | 9.2 | 8.9 | 8.6 |
| RMSNorm VJP | 5.8 | 9.2 | 9.7 | 10.4 | 9.1 |
| RoPE | 3.9 | 5.0 | 7.2 | 5.2 | 5.1 |
| RoPE, inverse | 3.9 | 4.7 | 5.8 | 6.8 | 4.9 |
| AdamW update (clip, moments, commit) | 55.4 | 76.5 | 81.0 | 89.9 | 78.3 |
| the rest (loss, embedding scatter, residuals, binding) | -12.7 | 69.2 | -6.9 | 116.5 | 134.1 |
| **step** | **1,631.9** | **1,859.9** | **1,541.1** | **1,912.3** | **1,506.9** |

**By thread count:**

| One step (ms) | v1, 1 thread | v1, 3 | v1, 4 | `fused+exp`, 1 | `fused+exp`, 3 | `fused+exp`, 4 |
|---|---:|---:|---:|---:|---:|---:|
| GEMMs, forward | 494.3 | 296.0 | 220.5 | 367.0 | 199.4 | 157.0 |
| GEMMs, backward (dx and dW) | 921.4 | 552.3 | 431.3 | 714.3 | 398.3 | 322.2 |
| attention, forward | 62.2 | 37.4 | 29.6 | 84.7 | 41.9 | 35.8 |
| attention, backward | 88.8 | 47.2 | 37.5 | 82.9 | 39.8 | 35.2 |
| SiLU | 57.3 | 26.5 | 22.8 | 7.4 | 3.6 | 3.3 |
| SwiGLU VJP | 62.6 | 32.9 | 26.7 | 10.1 | 4.8 | 4.2 |
| RMSNorm | 8.3 | 5.2 | 4.6 | 8.6 | 4.8 | 4.0 |
| RMSNorm VJP | 9.2 | 11.3 | 9.6 | 9.1 | 10.8 | 10.3 |
| RoPE | 5.0 | 4.5 | 3.9 | 5.1 | 4.1 | 3.6 |
| RoPE, inverse | 4.7 | 3.5 | 3.3 | 4.9 | 3.7 | 3.3 |
| AdamW update (clip, moments, commit) | 76.5 | 84.1 | 71.5 | 78.3 | 64.2 | 59.4 |
| the rest (loss, embedding scatter, residuals, binding) | 69.2 | 47.7 | 97.3 | 134.1 | 105.2 | 102.8 |
| **step** | **1,859.9** | **1,149.2** | **959.2** | **1,506.9** | **881.5** | **741.7** |

What the two tables say, same-session pairs only:

- **The GEMMs** are 76% of the v1 step and 72% of the `fused+exp` step at one
  thread. The fused contract takes them from 1,415.7 to 1,141.5 ms (x1.24;
  the forward x1.27, the backward x1.23), the kernel-level 1.3-1.5x of §2
  seen through the step's shapes.
- **The `exp` contract** takes SiLU and the SwiGLU VJP from 119.9 to 22.8 ms
  (x5.3): the exponential was most of those two kernels, and the vector instance over a
  whole span is what the contract bought. **It makes the attention forward slower**, 62.2 to
  79.0 ms at one thread (the backward, which reads the stored probabilities, is
  unchanged): the v2 softmax's row loop calls the scalar exponential element by element with the
  row sum in the same loop (`bcir_tensor.c`, `att_task`), which GCC does not vectorize. The fix is the
  row through the vector instance after the max-subtraction, then the sum in the same index order --
  the same values, so the `exp` state digest stays -- and it landed the same day as NDT-ATTN's first
  item ([attn0/](measurements/2026-10-12/attn0/), the same stages re-run after it): the attention
  forward 79.0 to 55.2 ms under `exp` at one thread, 84.7 to 58.1 under `fused+exp` (41.9 to 31.7 on
  three threads), against v1's 62.7 in the same run; the `exp` and `fused+exp` digests unchanged
  (`aaa9998c…`, `b18d89fa…`, at one and three threads, the widest build and a repeat). The tables
  above are the measurement before it, as the roadmap's budget was read against them.
- **Threads.** Three take the `fused+exp` step from 1,506.9 to 881.5 ms
  (x1.71) and four to 741.7 (x2.03). The GEMMs scale by
  x1.81 on three and x2.26 on four, the attention by x2.05 and
  x2.36, the AdamW update by x1.22 and x1.32 (it streams 157-197 MB
  a pass through one memory system, and its clipping norm is serial), and "the rest" not at all: it
  is the serial reductions the pool leaves alone by design, between -7 and 134 ms across the
  twelve cells with about ±50 ms of spread between them (a negative value is the step measuring
  faster than its kernels timed alone, inside that spread), which is this method's floor.

## 4. The 1,000-step language-model run

| Run (1,000 steps, batch 8 x 128) | host | ms/step | tokens/s | train s | val CE | test CE | test acc | answer acc | peak RSS MB | state |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch 2.13 eager, 1 thread | 10-11 | 774.3 | 1,323 | 774 | 1.5001 | 1.6319 | 0.5367 | 0.6197 | 1,140 | `a7cd43d5` |
| PyTorch 2.13 compiled, 1 thread | 10-11 | 711.8 | 1,439 | 711 | 1.5001 | 1.6318 | 0.5370 | 0.6196 | 1,056 | `99445b0f` |
| BCIR v1, 1 thread | 10-11 | 1,726.4 | 593 | 1,720 | 1.5003 | 1.6319 | 0.5365 | 0.6192 | 544 | `aed5ffbf` |
| BCIR v1 with OpenBLAS's GEMM, 1 thread | 10-11 | 962.6 | 1,064 | 953 | 1.5003 | 1.6319 | 0.5367 | 0.6194 | 544 | `ff6b793d` |
| BCIR v1, 1 thread | 10-12 | 1,951.9 | 525 | 1,975 | 1.5003 | 1.6319 | 0.5365 | 0.6192 | 547 | `aed5ffbf` |
| BCIR v1, 3 threads | 10-12 | 989.3 | 1,035 | 997 | 1.5003 | 1.6319 | 0.5365 | 0.6192 | 547 | `aed5ffbf` |
| BCIR `fused+exp`, 3 threads | 10-12 | 818.8 | 1,251 | 827 | 1.5003 | 1.6319 | 0.5368 | 0.6198 | 547 | `eb43c96d` |
| BCIR `fused+exp`, 4 threads | 10-12 | 758.4 | 1,350 | 763 | 1.5003 | 1.6319 | 0.5368 | 0.6198 | 547 | `eb43c96d` |

The PyTorch rows and the two 10-11 BCIR rows are the 10-11 report's (PyTorch 2.13 with MKL 2024.2,
one thread, that host); the 10-12 rows are this session's, one after another. `train s` is the
training clock over the 1,000 steps (evaluations outside it); the state is the SHA-256 of the
weights and both moments after the last step, its first eight hex digits. The two v1 rows of this
session end in the 10-11 run's state, and the two `fused+exp` rows in one state of their own.

**The model is the same model.** The validation and test cross-entropies agree to four decimals
across every row, and the two 3-thread runs' validation trajectories agree at every hundredth step
to within 0.0002 nats -- the v2 contracts changed roundings, not what the model learns:

| step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1, 3 threads: validation CE | 2.8464 | 2.4177 | 2.1221 | 1.9311 | 1.7917 | 1.6878 | 1.6212 | 1.5579 | 1.5182 | 1.5003 |
| v1, 3 threads: seconds | 98 | 198 | 299 | 398 | 497 | 600 | 701 | 802 | 900 | 997 |
| `fused+exp`, 3 threads: validation CE | 2.8464 | 2.4177 | 2.1221 | 1.9310 | 1.7918 | 1.6878 | 1.6211 | 1.5577 | 1.5183 | 1.5003 |
| `fused+exp`, 3 threads: seconds | 82 | 164 | 245 | 326 | 407 | 488 | 571 | 661 | 745 | 827 |

## 5. Reading, against the roadmap's budget

The roadmap's §2 composed the levers at their ceilings into an expectation for one step:
v1 at one thread 1,632 ms; `fused+exp` at one thread ~875; the same at four threads ~270; eager
PyTorch at one thread 762 and at four ~257. Measured on this host, same-session pairs:

| One step (ms) | v1, 1 thread | `fused+exp`, 1 thread | `fused+exp`, 3 threads | `fused+exp`, 4 threads |
|---|---:|---:|---:|---:|
| expected (§2's rows scaled by 1,860/1,632, this host's v1 step over the 10-11 host's) | 1,859.9 | ~997 | -- | ~308 |
| measured, the `bcir-breakdown` step | 1,859.9 | 1,506.9 | 881.5 | 741.7 |
| measured, the 1,000-step run's median step | 1,951.9 | -- | 818.8 | 758.4 |

Row by row:

- **GEMMs.** The budget assumed the fused contract halves the GEMM's time ("at the kernel's present
  efficiency"), because v1 spends two instructions per term and the fused instance one. It gains
  x1.24 on the step's GEMMs: the multiply-adds were never the whole kernel. The rest is
  the packing of each B tile, the A loads and the C stores of the register block, which the fused
  contract does not touch -- NDT-FMA's stated open item (a wider register block, per-panel packing of
  the transposed operand) is where the next 1.3x of the one-thread GEMM is, and OpenBLAS's 87
  GFLOP/s on the same shapes says the block exists (the ladder's NDT-BLOCK).
- **Threads.** The budget took four threads as 3.6x on the GEMMs. The step's GEMMs scale by
  x1.81 on three threads and x2.26 on four, the kernel bench by 2.0x on
  four. The partition is the reason: 64-row by one-column-block tasks, run in task order with one
  barrier per GEMM, so the step's narrow products (the K/V projections at 1024 x 192 x 384 have
  16 x 3 tasks) leave cores idle at every barrier, and every GEMM pays a barrier. NDT-SCHED's two items
  are exactly this: the plan choosing the split per call, and the two VJPs of a linear (`dx`, `dW`:
  disjoint writes) run concurrently instead of one after the other. The 1,000-step runs scale
  better than the breakdown's single steps (v1: 1,952 to 989 ms, x1.97 on three threads, against
  the breakdown's x1.62); the two runs were an hour apart on a shared host, where the breakdown's
  cells were minutes apart, so the breakdown's ratios are the ones this report reads.
- **Attention, SiLU, the VJP, the norms.** The `exp` contract did what the budget said for the
  elementwise kernels (SiLU and the VJP from 119.9 to 22.8 ms) and the opposite for
  the attention forward (§3: the scalar row loop); the attention scales on the pool (one task per
  (batch, head): x2.36 on four threads). At this context (128 tokens) attention is a tenth
  of the step; NDT-ATTN is for longer contexts, and the row-loop fix, its first item, landed with
  this report (§3).
- **AdamW and the clipping norm.** The commit scales on the pool; the clipping norm and the
  finiteness check are the serial pass they were, and the update reads 76.5, 84.1
  and 71.5 ms at one, three and four threads under v1 -- the budget's "~30 ms at four
  threads" needs NDT-NORM to be true -- measured since ([update-passes.json](measurements/2026-10-12/update-passes.json)):
  41 of the update's 53 four-thread ms are its first pass, 16 of them the gradient's index-order
  chain (serial by the contract) and 25 the weight and moment streams' maxima and finiteness, which
  are associative and can go on the pool; the commit is 12 ms pooled (33 serial). About 34 ms is
  the honest expectation, not 30.
- **The rest.** The loss, the embedding scatter and the gamma VJP's row sum are index-order
  reductions the pool leaves serial by design; at 105.2 ms on three threads they are
  12% of the step and will be a larger share of every faster step. A deterministic
  tree reduction with a fixed shape (a third declared contract, like the two above: the ladder's
  NDT-REDUCE) is the only way to thread them.

**Against PyTorch.** On four threads the `fused+exp` run trains at 758.4 ms a step and
1,350 tokens a second; the 10-11 eager PyTorch run trained at 774 ms and 1,323 tokens a
second on one thread, on a host this report's one-thread v1 run says is about 13% faster.
That is BCIR on four threads at about the speed of eager PyTorch on one, on a different host -- not
the roadmap's "ahead at four", and the comparison at equal thread counts on one host is the
measurement that decides it; it needs a host with PyTorch. What this session can say is where BCIR's
remaining time is, by kernel and by thread, and that every one of those residuals has a named slice.

## 6. Reproduce

```bash
H=docs/research/measurements/2026-10-11/train_comparison.py
python $H prepare --output DIR
python $H identity --data DIR --steps 20 --cc gcc --threads 1,2,3,4                       # v1
python $H identity --data DIR --steps 20 --cc gcc --threads 1,2,3,4 \
    --contract fused,exp,fused+exp --variants widest,avx2
for c in v1 fused exp fused+exp; do for n in 1 3; do
  python $H bcir-breakdown --data DIR --cc gcc --rounds 5 --threads $n --contract $c; done; done
for c in v1 fused+exp; do python $H bcir-breakdown --data DIR --cc gcc --rounds 5 --threads 4 --contract $c; done
python $H train --data DIR --task lm --backend bcir --threads 1 --contract v1        --eval-every 100 --lr-scale 0.25
python $H train --data DIR --task lm --backend bcir --threads 3 --contract v1        --eval-every 100 --lr-scale 0.25
python $H train --data DIR --task lm --backend bcir --threads 3 --contract fused+exp --eval-every 100 --lr-scale 0.25
python $H train --data DIR --task lm --backend bcir --threads 4 --contract fused+exp --eval-every 100 --lr-scale 0.25
# §7: the blocked contract on BCIR's kernel and on OpenBLAS admitted (scipy_openblas32 installed)
python $H identity --data DIR --steps 20 --cc gcc --threads 1,3 --contract kblock,kblock+exp --variants widest
for n in 1 4; do python $H bcir-breakdown --data DIR --cc gcc --rounds 5 --threads $n --contract kblock+exp; done
for b in bcir bcir-openblas; do for n in 1 4; do
  python $H train --data DIR --task lm --backend $b --threads $n --contract kblock+exp --eval-every 100 --lr-scale 0.25; done; done
# §9: the reductions on the pool, and the reduce contract (the breakdowns alternated, three times each)
for i in 1 2 3; do for c in kblock+exp kblock+exp+reduce; do
  python $H bcir-breakdown --data DIR --cc gcc --rounds 5 --threads 4 --contract $c; done; done
for c in kblock+exp kblock+exp+reduce; do python $H bcir-breakdown --data DIR --cc gcc --rounds 5 --threads 1 --contract $c; done
python $H identity --data DIR --steps 20 --cc gcc --threads 1,3 --contract reduce,kblock+exp+reduce --variants widest
for n in 4 1; do python $H train --data DIR --task lm --backend bcir --threads $n --contract kblock+exp+reduce --eval-every 100 --lr-scale 0.25; done
```

The GEMM-rate record is the output of a scratch driver over the trainer's own entry points
(`bcir_tensor_mm_pool`, `bcir_tensor_mm_fused_pool`, `scipy_cblas_sgemm`) on the step program's
shapes (`native_program.training_step_program(...).gemm_shapes()`); its arguments are in the
record. The exact rows are the GEM+ `train` group's
(`python tools/perf/gemplus_baseline.py --compare --group train`).

## 7. The blocked contract, and OpenBLAS as an admitted provider (NDT-KBLOCK)

The one surprise in §2's GEMM table was that OpenBLAS's `sgemm` was bit-identical to BCIR's fused
contract on every shape whose depth fits one K block, and a K-blocked form of it on the rest: its
level-3 driver splits the depth by one rule (Q 448, unroll 16 -- whole 448-blocks while two remain,
then half the remainder rounded up to a multiple of 16, then the remainder), its kernel accumulates
each block from zero in registers, and it adds the blocks to the beta-scaled c in order. That
arithmetic, declared as BCIR's **blocked fused contract (v3)** and implemented by BCIR's own kernel
on every host (the partial kept in registers across a whole block), makes OpenBLAS an admissible
provider: `use_cblas` under v3 runs a differential over the trainer's own GEMM shapes at beta 0 and
1 and admits the provider only where every output is the contract's, bit for bit; a provider that
differs is refused and the state keeps BCIR's kernels. Measured in the same session, on this host:

| Exact rows | Result |
|---|---|
| `train.mm.kblock.mismatch` (the kernel against its exact rational emulation, 40 cases; the split rule on both rails) | 0 |
| `train.mm.provider.mismatch` (OpenBLAS's `sgemm` against the kernel over the 15 step shapes at beta 0 and 1; the row is `train.mm.provider.differences` since PR817-P0, with `train.mm.provider.unsound` the graded one -- PR #817's AMD host reads 14 of 30 and refuses the provider) | 0 of 30 |
| the `identity` stage, `kblock` and `kblock+exp`, 1 and 3 threads, the widest build and a repeat ([kblock/identity-kblock.json](measurements/2026-10-12/kblock/identity-kblock.json)) | one state per contract: `5b92ff75…`, `f19b0d7d…`, apart from the other four |
| the 1,000-step `kblock+exp` run on BCIR's kernel and on OpenBLAS admitted, at 1 and at 4 threads | one state, `d7fc618c9eeb8b6f…`, on all four ([kblock/runs/](measurements/2026-10-12/kblock/runs/)): BCIR's own kernel at one and at four threads, OpenBLAS admitted at one and at four |
| the small products OpenBLAS computes on its small-matrix path (the threads bench's shapes) | refused by the differential (4 of 30 cases differ): the admission is per shape set |

**The 1,000-step language-model run under `kblock+exp`** (same data, optimizer and schedule as §4):

| Run (1,000 steps, batch 8 x 128, `kblock+exp`) | ms/step | tokens/s | train s | val CE | test CE | test acc | answer acc | peak RSS MB | state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BCIR's own kernel, 1 thread | 1,424.6 | 719 | 1,447 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| BCIR's own kernel, 4 threads | 753.0 | 1,360 | 756 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| OpenBLAS admitted, 1 thread (OpenBLAS at 1) | 1,065.5 | 961 | 1,078 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| OpenBLAS admitted, 4 threads (OpenBLAS at 4, the pool at 4; its workers spinning between calls, OpenBLAS's default) | 655.0 | 1,563 | 662 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| OpenBLAS admitted, 4 threads, its workers sleeping between calls (`OPENBLAS_THREAD_TIMEOUT=4`) | 615.9 | 1,663 | 619 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| §4: `fused+exp` on BCIR's kernel, 4 threads | 758.4 | 1,350 | 763 | 1.5003 | 1.6319 | 0.5368 | 0.6198 | 547 | `eb43c96d` |
| §4, the 10-11 host: PyTorch 2.13 eager, 1 thread | 774.3 | 1,323 | 774 | 1.5001 | 1.6319 | 0.5367 | 0.6197 | 1,140 | `a7cd43d5` |

Three readings:

- **The contract is the state.** Four runs through two GEMM implementations -- BCIR's own blocked
  kernel and OpenBLAS's `sgemm` -- each at one and at four threads, end in the same 9.84M-parameter
  state after 1,000 steps, byte for byte. The provider is admitted into a declared contract, not
  trusted; where it is not the contract (the small products its small-matrix path computes) it is
  refused, and BCIR's kernel runs.
- **The provider's rate is the step's -- once its workers stop spinning.** OpenBLAS admitted takes
  the one-thread step from 1,425 to 1,066 ms (x1.34) and the four-thread step from 753 to 655
  (x1.15) -- and the step by kernel under the provider (below) says why only x1.15: OpenBLAS's
  worker threads spin for 2^28 cycles after each call before they sleep, so between two GEMMs
  three of the four cores are burning while the pool runs the attention, the update and the
  norms (the attention 57 to 114 ms, the rest 51 to 146). With `OPENBLAS_THREAD_TIMEOUT=4` set
  before the library loads (the harness's `_openblas` now sets it) the workers sleep at once: the
  step by kernel reads 527 ms against 632 spinning, and the 1,000-step run's median 615.9 ms
  against 655.0 (x1.22 over BCIR's kernel's 753; the same state). Against the
  10-11 report's eager PyTorch run -- 774 ms a step on one thread, on a host about 13% faster --
  BCIR with the admitted provider trains the same model at 615.9 ms on four threads: faster
  than eager PyTorch's one-thread step, and about 2.4 times PyTorch's four-thread step there
  (257 ms, the 10-11 context stage), both measured on different hosts. The equal-thread comparison
  on one host stays the measurement that decides the roadmap's "ahead at four".
- **BCIR's own v3 kernel is faster than its v2 kernel.** At one thread the step's GEMMs go from
  1,081 ms (`fused+exp`, §3) to 952 under `kblock+exp`: the partial stays in registers across a
  whole K block of 448 instead of a 64-deep chunk, so c is read and written once per block. At
  four threads the two are equal within the host's spread (742 against 742 ms a step). The gap
  that remains to the provider (952 ms against OpenBLAS's rate on the same shapes) is NDT-BLOCK's
  to close, and under v3 the provider's rate is the ceiling the kernel is measured against.

The step by kernel under `kblock+exp` on BCIR's own kernel ([kblock/breakdown-bcir-kblock+exp-t{1,4}.json](measurements/2026-10-12/kblock/)):

| One step (ms), `kblock+exp`, BCIR's kernel | 1 thread | 4 threads |
|---|---:|---:|
| GEMMs, forward | 285.7 | 210.8 |
| GEMMs, backward (dx and dW) | 666.7 | 320.7 |
| attention, forward and backward | 163.8 | 57.0 |
| SiLU and the SwiGLU VJP | 23.1 | 7.2 |
| AdamW update | 75.8 | 74.4 |
| the rest | 92.7 | 50.6 |
| **step** | **1,342.3** | **741.8** |

and with OpenBLAS admitted ([kblock/breakdown-bcir-openblas-kblock+exp-t{1,4}.json](measurements/2026-10-12/kblock/)
and [-t4-spinning.json](measurements/2026-10-12/kblock/breakdown-bcir-openblas-kblock+exp-t4-spinning.json); the
runs are [kblock/runs/](measurements/2026-10-12/kblock/runs/), the spinning one `-t4-spinning.json`):

| One step (ms), `kblock+exp`, OpenBLAS admitted | 1 thread | 4 threads, workers spinning | 4 threads, workers sleeping |
|---|---:|---:|---:|
| GEMMs, forward and backward | 718.3 | 260.0 | 299.2 |
| attention, forward and backward | 137.3 | 114.2 | 61.1 |
| SiLU and the SwiGLU VJP | 20.7 | 9.0 | 7.5 |
| norms and RoPE | 28.2 | 20.0 | 20.5 |
| AdamW update | 81.1 | 81.9 | 66.9 |
| the rest | 53.5 | 146.4 | 70.7 |
| **step** | **1,039.6** | **631.9** | **526.5** |

## 8. The kernel at the provider's rate (NDT-BLOCK)

§7 left BCIR's own v3 kernel at 952 ms of GEMMs a step against the admitted provider's 718 at one
thread, and the kernel-rate record at 0.7x OpenBLAS. The gap was the schedule, not the arithmetic
-- the contract fixes every element's chain, so only the order of loads and stores could change --
and three things in it: the 4 x 64 register block read its 112 KB tile from L2 once per four rows
(272 bytes per 16 fused multiply-adds, at the L2's limit), the compiler gathered the block's A
terms through the stack, and the tile was packed scalar (16 us for a transposed tile) while a
transposed A's lines arrived one per depth with no prefetch. A scratch prototype measured the
candidates against the tree's kernel bit for bit on 42 cases: 12 x 32 at 92 GFLOP/s, 8 x 48 at
108, 6 x 64 and 4 x 96 between, and the row-panel-outer nest that repacks the tile per row panel
worse at every panel height (60-77), because the packing is the cost. What landed: the AVX-512 and
AVX2 instances of v3 as intrinsic micro-kernels (8 x 48 and 6 x 16 accumulators, every A term one
broadcast from memory, the tile three or two vector loads per depth -- eleven loads per 24 fused
multiply-adds), the tile packed by vector (masked row loads; 16 x 16 register transposes of B^T),
the transposed A prefetched eight depths ahead, the pool's tasks 128 rows by one column block of
the instance's width; the generic instance keeps the plain-C block. The C harness holds every
instance to the K-blocked `fmaf` chains on 19 shapes serially and on a pool, and
`train.mm.kblock.mismatch` now runs per instance.

**The kernel-rate record** ([block/gemm-rates-block.json](measurements/2026-10-12/block/gemm-rates-block.json);
the step's 15 GEMM shapes, best of 7 calls per cell, one process, GCC 13.3, OpenBLAS 0.3.34
SkylakeX with `OPENBLAS_THREAD_TIMEOUT=4`; every result at beta 1 identical to the old serial
kernel's, 0 of 75 cells differ):

| GEMM (ta, tb, m, n, k) | v3 kernel before (4 x 64, plain C) | v3 kernel after (8 x 48, intrinsics) | OpenBLAS `sgemm` |
|---|---:|---:|---:|
| (0, 1, 1024, 384, 384) | 71.5 | 119.3 | 79.0 |
| (0, 1, 1024, 192, 384) | 62.1 | 82.0 | 86.5 |
| (0, 1, 1024, 1024, 384) | 74.1 | 106.5 | 92.6 |
| (0, 1, 1024, 384, 1024) | 80.0 | 122.1 | 93.2 |
| (0, 1, 1024, 260, 384) | 64.8 | 101.9 | 79.4 |
| (0, 0, 1024, 384, 260) | 56.2 | 75.6 | 63.1 |
| (1, 0, 260, 384, 1024) | 84.4 | 116.0 | 88.8 |
| (0, 0, 1024, 1024, 384) | 53.9 | 94.7 | 94.3 |
| (1, 0, 384, 1024, 1024) | 78.9 | 106.3 | 93.8 |
| (0, 0, 1024, 384, 1024) | 79.5 | 119.4 | 91.2 |
| (1, 0, 1024, 384, 1024) | 31.2 | 73.6 | 89.6 |
| (0, 0, 1024, 384, 384) | 80.3 | 113.9 | 88.9 |
| (1, 0, 384, 384, 1024) | 76.2 | 95.2 | 86.7 |
| (0, 0, 1024, 384, 192) | 47.1 | 67.4 | 91.1 |
| (1, 0, 192, 384, 1024) | 63.4 | 91.9 | 70.1 |
| **work-weighted** | **61.2** | **99.4** | **88.8** |

| GEMM (ta, tb, m, n, k) | v3 kernel before, the pool at 4 | v3 kernel after, the pool at 4 | OpenBLAS at 4, workers sleeping |
|---|---:|---:|---:|
| (0, 1, 1024, 384, 384) | 105.1 | 188.6 | 185.3 |
| (0, 1, 1024, 192, 384) | 92.9 | 164.5 | 155.0 |
| (0, 1, 1024, 1024, 384) | 100.8 | 215.3 | 251.2 |
| (0, 1, 1024, 384, 1024) | 124.7 | 257.0 | 257.2 |
| (0, 1, 1024, 260, 384) | 84.1 | 203.3 | 183.6 |
| (0, 0, 1024, 384, 260) | 131.9 | 168.8 | 167.2 |
| (1, 0, 260, 384, 1024) | 147.6 | 194.7 | 161.9 |
| (0, 0, 1024, 1024, 384) | 151.4 | 208.5 | 204.5 |
| (1, 0, 384, 1024, 1024) | 192.6 | 246.8 | 255.6 |
| (0, 0, 1024, 384, 1024) | 199.9 | 297.8 | 262.1 |
| (1, 0, 1024, 384, 1024) | 142.7 | 239.1 | 270.9 |
| (0, 0, 1024, 384, 384) | 145.5 | 180.6 | 179.1 |
| (1, 0, 384, 384, 1024) | 174.1 | 208.0 | 181.2 |
| (0, 0, 1024, 384, 192) | 106.9 | 114.0 | 114.0 |
| (1, 0, 192, 384, 1024) | 110.4 | 194.7 | 212.4 |
| **work-weighted** | **135.9** | **218.9** | **219.1** |

Reading: BCIR's own kernel is now at or past the provider's rate on this host -- 1.12x OpenBLAS
at one thread and equal on four, 1.62x its own previous rate -- on 13 of the 15 shapes. The two
it still loses: `dW` with 1,024 rows and depth 1,024 (73.6 against 89.6, from 31.2), whose
transposed-A lines are 4 KB apart and so alias one cache set at every level, which the prefetch
halves but packing the panel would remove -- the serial path has no buffer for it under the
unit's no-allocation law, so it stays the open item of the ladder's row -- and the depth-192
product (67.4 against 91.1), one short K block whose per-call costs (the beta pass over c, the
tile) the chain does not cover.

**The step and the run.** Under `kblock+exp` on BCIR's kernel ([block/breakdown-bcir-kblock+exp-t{1,4}.json](measurements/2026-10-12/block/)):

| One step (ms), `kblock+exp`, BCIR's kernel | 1 thread, §7 | 1 thread, now | 4 threads, §7 | 4 threads, now |
|---|---:|---:|---:|---:|
| GEMMs, forward | 285.7 | 187.3 | 210.8 | 93.9 |
| GEMMs, backward (dx and dW) | 666.7 | 381.8 | 320.7 | 196.3 |
| attention, forward and backward | 163.8 | 128.2 | 57.0 | 54.9 |
| SiLU and the SwiGLU VJP | 23.1 | 17.5 | 7.2 | 7.5 |
| norms and RoPE | -- | 27.8 | -- | 20.7 |
| AdamW update | 75.8 | 73.3 | 74.4 | 53.1 |
| the rest | -- | 198.4 | -- | 123.7 |
| **step** | **1,342.3** | **1,014.3** | **741.8** | **550.1** |

(§7's table had no norms-and-RoPE row, so its "rest" holds them; here they are their own row. The rest grows with the faster kernel because the parts are timed on warm operands in paired rounds while the step reads its 39 MB of weights once per pass, and that traffic lands in the unattributed row; the step is the number.)

The 1,000-step language-model run (same data, optimizer and schedule as §4; [block/runs/](measurements/2026-10-12/block/runs/)):

| Run (1,000 steps, batch 8 x 128, `kblock+exp`) | ms/step | tokens/s | train s | val CE | test CE | test acc | answer acc | peak RSS MB | state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BCIR's kernel, 1 thread, §7 (the 4 x 64 block) | 1,424.6 | 719 | 1,447 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| BCIR's kernel, 1 thread, now | 973.7 | 1,052 | 983 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| OpenBLAS admitted, 1 thread, §7 | 1,065.5 | 961 | 1,078 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| BCIR's kernel, 4 threads, §7 | 753.0 | 1,360 | 756 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| BCIR's kernel, 4 threads, now | 516.3 | 1,983 | 518 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| OpenBLAS admitted, 4 threads, its workers sleeping, §7 | 615.9 | 1,663 | 619 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| eager PyTorch, 1 thread, the 10-11 host | 774.3 | -- | -- | 1.5001 | 1.6319 | -- | -- | -- | -- |

Reading:

- **The state is the state.** Both runs end in `d7fc618c…`, the state of §7's four runs: the
  kernel's new schedule is the contract's arithmetic, so the same 9.84M model to the last bit,
  and the same held-out numbers to four decimals.
- **BCIR's own kernel now trains faster than the admitted provider did.** 973.7 ms a step on
  one thread (1,424.6 before, x1.46; OpenBLAS admitted 1,065.5) and 516.3 on four (753.0 before,
  x1.46; OpenBLAS admitted with its workers sleeping 615.9) -- 1,983 tokens a second. Against
  the 10-11 host's eager PyTorch on one thread (774.3) the four-thread BCIR step is 1.5x
  faster and the one-thread step 0.8x, both on a host about 13% slower than that one; the
  equal-thread comparison on one host stays the measurement that decides "ahead at four"
  (PyTorch's four-thread step there was about 257 ms).
- **Where the step is now.** By kernel at four threads: GEMMs 290 ms (531 before), attention
  55, the update 53, norms and RoPE 21, SiLU and the VJP 8, the rest 124 -- the GEMMs are half
  the step for the first time, and the residuals are the ones the ladder names: the serial
  reductions and the update's first pass (NDT-REDUCE, NDT-NORM -- landed the next day, §9), the
  partition and the two VJPs run concurrently (NDT-SCHED), and the transposed-A packing this
  slice left open.

| Exact rows | Result |
|---|---|
| `train.mm.kblock.mismatch`, now over every instance the host runs (generic, AVX2, AVX-512; 40 cases each) | 0 |
| the C harness's blocked-contract section: 19 shapes x 4 transposes x 3 instances, serially and on a pool of three | every run bit-identical to the chain |
| the `identity` stage, `kblock` and `kblock+exp`, 1 and 3 threads, the widest, AVX2 and portable builds ([block/identity-block.json](measurements/2026-10-12/block/identity-block.json)) | 9 rows, two states, §7's: `5b92ff75…` under `kblock` and `f19b0d7d…` under `kblock+exp` -- the widest (AVX-512) and AVX2 builds at 1 and 3 threads under both contracts, and the portable build's `kblock` row at one thread; its other three rows were stopped (the portable build's v3 kernel calls `fmaf` per term and takes 66 s a step), an honest gap, not a pass |
| the 1,000-step `kblock+exp` run on BCIR's kernel, 1 and 4 threads | one state, `d7fc618c9eeb8b6f…`, on both -- §7's state |
| `ndt-kblock.json`, NK1-NK10 | 10 injected, 10 caught |

## 9. The reductions on the pool (NDT-REDUCE)

§8 left the four-thread step at 550 ms with the GEMMs at 290 and named the residuals: the update's
53 ms, 41 of them a serial first pass (the clipping norm's index-order chain and the three state
streams' maxima and finiteness), the gamma VJP's row sum, the loss's rows and the embedding scatter
-- index-order reductions the pool left serial by the law of NDT-THREADS, which admits only
partitions in which each output element, and each reduction, is produced by one task in the serial
order. Three of the four admit such a partition after all, and only the norm needed a contract.

- **The loss's rows** are tasks of sixteen rows: a row's softmax, its adjoint and its term
  (log-sum-exp plus the maximum less the target's logit, in double, as the serial loop computed it)
  are its own; the terms are kept as bytes at the scratch arena's start and added in row order by
  the caller, so the mean is the serial chain, bit for bit.
- **The gamma VJP** (`bcir_tensor_rms_backward_pool`) runs in two pooled phases: rows for `dx` and
  each row's `1/rms`, kept in a row of scratch the trainer lends (the query adjoint's region, free
  until the layer's query VJP), then columns of sixteen for the gamma adjoint, each column's sum
  over the rows in row order from the value it holds -- the serial kernel's operations, element
  for element.
- **The embedding scatter** is tasks over column ranges: a task adds every row's columns to its
  token's row in row order, so a token the batch repeats accumulates as the serial loop did.
- **The update's first pass and preflight** are one task per chunk of `bcir_decoder_reduce_chunk`
  elements -- 1,024 doubled until the chunks number at most 1,024 (16,384 for this decoder's
  9,836,928 parameters, 601 tasks), their results on the caller's stack -- whose verdicts and
  maxima are exact whatever the partition. **The clipping norm** is the contract's: under every
  contract that was, the one index-order chain in double over every element, serial, as before;
  under the new `reduce` bit (`NativeDecoder(contract="kblock+exp+reduce")`, any contract
  `+reduce`), each chunk's squares summed in index order in double from zero and the chunk sums
  added in chunk order from zero. The shape is a function of the parameter count alone, so the
  norm is the same at every thread count; it is not the chain's norm, so a state trained under it
  is its own, and the checkpoint names it. The oracle emulates both norms operation for operation
  (`grad_norm_squared_reference`), `bcir_decoder_gradient_norm` reports the square the update
  takes its root of (the root rounds a one-ulp difference of the sum away half the time; the
  square never), and `train.norm.reduce.mismatch` holds every square and root the trainer reports
  to that emulation.

The reduce bit changes the norm's last bits and nothing else -- a part in 10^12 at most on the
harness's and the threads bench's steps -- and the float rounding of every update absorbs that for
a while: the harness's twelve contracts end in six states after three steps, each reduce state its
chain's, with the norms apart at 15 of 18 steps. On the 9.84M decoder a rounding boundary is
crossed within the first twenty steps (the identity rows below), and the state is the reduce
contract's own from there. The contract's witness is the norm.

Two of the sweep's eleven faults were first reported uncaught, and both were the witnesses'
findings, not the trainer's: the wide fixture's token stream was periodic, so every repeat of a
token was the same row bit for bit and a reversed accumulation summed the same values (the
streams now repeat a token without repeating a row), and the norm's root hides a one-ulp change
of the chunked sum half the time (hence the square's entry, and a three-chunk fixture, since two
chunks commute).

The step by kernel, GCC 13.3, medians of 5 paired rounds; the four-thread cells are medians over
3 and 3 breakdowns of each contract alternated minutes apart, so the pair sees
one host, with every sample in [reduce/](measurements/2026-10-13/reduce/). §8's record of the day
before is beside them: on this day the host ran the unchanged kernels (the GEMMs, the attention,
SiLU) 25-45% slower and drifted 20% on the GEMM rows within the session's hour, so
the same-session pairs are the comparison, §8's column the earlier record, and the step's own
cell says less than the attributable rows.

| ms | `kblock+exp`, §8, 4 threads | `kblock+exp`, 4 threads | `kblock+exp+reduce`, 4 threads | `kblock+exp`, §8, 1 thread | `kblock+exp`, 1 thread | `kblock+exp+reduce`, 1 thread |
|---|---:|---:|---:|---:|---:|---:|
| GEMMs (forward + backward) | 290.2 | 370.9 | 367.4 | 569.1 | 836.9 | 865.0 |
| attention (forward + backward) | 54.9 | 69.8 | 71.5 | 128.2 | 175.9 | 172.0 |
| RMSNorm forward + gamma VJP | 13.1 | 17.9 | 13.3 | 18.0 | 26.3 | 21.6 |
| RoPE, SiLU, the SwiGLU VJP | 15.0 | 19.5 | 20.4 | 27.3 | 40.5 | 48.0 |
| AdamW update | 53.1 | 47.5 | 39.6 | 73.3 | 104.2 | 106.8 |
| the rest | 123.2 | 94.0 | 127.4 | 198.1 | 202.8 | 157.0 |
| **step** | **550.1** | **619.8** | **659.6** | **1014.3** | **1387.1** | **1370.9** |

The 1,000-step language-model run (same data, optimizer and schedule as §4;
[reduce/runs/](measurements/2026-10-13/reduce/runs/)):

| Run (1,000 steps, batch 8 x 128) | ms/step | tokens/s | train s | val CE | test CE | test acc | answer acc | peak RSS MB | state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `kblock+exp`, BCIR's kernel, 4 threads, §8 | 516.3 | 1,983 | 518 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| `kblock+exp+reduce`, 4 threads | 625.8 | 1,636 | 627 | 1.5003 | 1.6319 | 0.5368 | 0.6196 | 546 | `5137e013` |
| `kblock+exp`, BCIR's kernel, 1 thread, §8 | 973.7 | 1,052 | 983 | 1.5001 | 1.6319 | 0.5375 | 0.6200 | 547 | `d7fc618c` |
| `kblock+exp+reduce`, 1 thread | 1419.6 | 721 | 1,437 | 1.5003 | 1.6319 | 0.5368 | 0.6196 | 546 | `5137e013` |

Reading:

- **Nothing moved that was not declared to move.** The `kblock+exp` state is §7's `d7fc618c…` on
  the pooled reductions (the identity rows of the thread-identity test and the harness), and the
  `reduce` run's state is its own, not §7's `d7fc618c…` (the norm's last bits crossed a float rounding boundary within the first steps, as the 20-step identity rows already show, and the state is the reduce contract's from there), and the one-thread run ends in the same state; its held-out model is the same to four decimals.
- **The update.** in every pair the reduce contract's update is below its chain's: 40.7 / 49.7 / 47.5 ms under `kblock+exp` (the first pass pooled, its chain serial -- NDT-NORM's content) against 31.8 / 47.2 / 39.6 under `kblock+exp+reduce`, -9.0 / -2.6 / -7.9; §8's record read 53.1 the day before on a faster host. At one thread 104.2 against 106.8: the chain per element is the same work, the pass is one loop fewer.
- **The rest.** the gamma VJP row, timed alone, 11.0 / 14.3 / 12.4 against 6.7 / 8.8 / 11.8 ms (-4.3 / -5.5 / -0.6; 9.2 in §8 -- the pooled kernel on a slower host); the rest row 94.0 / 50.6 / 141.7 against 57.6 / 141.1 / 127.4 (-36.4 / +90.5 / -14.3) is the step less its parts, each a separate median, so it carries every row's noise and says nothing on this day.
- **The step.** not resolved on this day's host: the step read 558.0 / 619.8 / 667.8 ms under `kblock+exp` and 552.1 / 659.6 / 677.4 under `kblock+exp+reduce` (-5.8 / +39.8 / +9.6), while the unchanged GEMM rows drifted 20% between the quietest and the busiest breakdown of the session (329.0 / 396.4 / 370.9 under `kblock+exp` alone) -- more than the slice's whole effect on the step, which by its rows is the serial work it removed: about 10-15 ms of the update and 5 of the gamma VJPs at four threads. The 1,000-step run's 625.8 ms a step (516.3 the day before, under `kblock+exp`) is the host's state on this day, not the slice's; a re-measurement on a quiet host is owed and the exact rows are the slice's gate.

| Exact rows | Result |
|---|---|
| `train.norm.reduce.mismatch` (the norm's square and root under v1 and under `reduce` against the oracle's emulation over three steps of the threads bench; the chunk rule on both rails at 8 counts) | 0 |
| `train.state.threads.mismatch`, now under v1 and under `reduce` | 0 of 6 |
| the C harness: `reduction_references` (the pooled loss and scatter against serial references written apart from the pool, 32 and 40 rows, no pool and a pool of two), `pool_identity` (twelve contracts, the states and the losses and norms the steps report, at one to four threads), the gamma VJP against the serial kernel over three row and three column chunks, `adamw_reference` under both norm contracts on a three-chunk fixture, serially and on a pool of three, the square against the reference's sum | every run bit-identical; the reduce norms apart from the chain's at 15 of 18 and 10 of 10 steps |
| the `identity` stage, `reduce` and `kblock+exp+reduce`, 1 and 3 threads and a repeat, the widest build ([reduce/identity-reduce.json](measurements/2026-10-13/reduce/identity-reduce.json)) | 8 rows (two contracts, 1 and 3 threads, and a repeat), one state per contract: `4bf4e057…` under `kblock+exp+reduce`, `f5fede12…` under `reduce` -- each identical at every thread count and apart from its chain's 20-step state (`6c0419e8…` under v1, `f19b0d7d…` under `kblock+exp`) |
| `ndt-reduce.json`, NR1-NR11 | 11 injected, 11 caught |

## 10. The two VJPs of a linear as one job (NDT-SCHED-0)

§5 named the partition behind one barrier per GEMM as the reason four threads gave x2.0 on the
step and not more: the step's narrow products (the K/V projections at 1,024 x 192 x 384 have
8 x 4 tasks of 128 rows by 48 columns on the AVX-512 instance) leave threads idle at every
barrier, and the backward pays two barriers per linear. NDT-SCHED's first item removes one of
them: a linear's two VJPs -- `dx = dy W` (beta-scaled) and `dW += dy^T x` -- read one adjoint and
write apart, so `bcir_tensor_mm_pair_pool` runs the first GEMM's tasks and then the second's in
one pool run, each with the partition its own entry would have used (the thread that runs task t
runs the same work it would have run for that GEMM alone), so every output element keeps its
depth order and the results are the two entries' bit for bit. The trainer's `linear_backward`
dispatches the pair for BCIR's kernels under each GEMM contract; a provider's GEMM keeps its two
calls, threaded as the provider threads it. The C harness holds every shape pair of its GEMM
section under the three contracts, in both orders, to the two entries, and the twelve contracts'
three-step states at one to four threads to the serial run through the trainer's own pairs;
`ndt-threads.json` NT9 (a pair split at the first GEMM's column count) and NT10 (the pair's serial
path skips the first GEMM) are the injected faults, and NF3 / NK4 (a contract's VJPs routed to the
wrong kernel) now hit the one predicate (`gemm_contract`) the forward's linears and the pair read.

Measured in 3 same-session pairs at four threads under `kblock+exp+reduce` (the pair
job against the two calls, alternated minutes apart; [sched0/](measurements/2026-10-13/sched0/)):

| four threads, ms | two calls | the pair job | pair difference |
|---|---|---|---|
| the backward pass (`loss_backward`, the forward recomputed inside; the stage the pair changes) | 631.9 / 633.8 / 622.4 | 601.0 / 586.3 / 594.6 | -30.9 / -47.5 / -27.8 |
| GEMM backward (the 43 linears' VJPs), timed alone on synthetic operands | 292.7 / 285.9 / 266.6 | 238.0 / 254.8 / 282.6 | -54.6 / -31.0 / +16.0 |
| step | 684.2 / 679.4 / 659.3 | 645.1 / 681.7 / 668.1 | -39.2 / +2.4 / +8.8 |

Reading: the backward pass (the `loss_backward` stage, the forward recomputed inside it, as the trainer runs it -- the one stage the pair job changes) is below the two-call stage in 3 of 3 pairs, by -30.9 / -47.5 / -27.8 ms (631.9 / 633.8 / 622.4 against 601.0 / 586.3 / 594.6), and the GEMM-backward row timed alone by -54.6 / -31.0 / +16.0; the step's own cell (-39.2 / +2.4 / +8.8) is within this host's run-to-run noise, as NDT-REDUCE's was -- the one barrier fewer per linear and the narrow projections' tasks shared are worth about 5% of the backward at four threads here
