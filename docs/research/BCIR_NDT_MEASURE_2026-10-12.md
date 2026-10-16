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

## 11. Where the rest of the step went (NDT-SCAN)

The breakdown times each kernel family alone on synthetic operands and reports the rest of the
step as a difference -- about 150 ms of the four-thread step after NDT-SCHED-0. Timed in place
instead ([scan/attribution.py](measurements/2026-10-13/scan/attribution.py): a copy of the two C
units with a clock read after every statement of the forward, the backward and the update, run
on the comparison's decoder), the rest was mostly six passes over whole buffers on the caller's
thread while the other three waited:

| four threads, one in-place run before NDT-SCAN | ms |
|---|---|
| the activations' finiteness scan (about 170 MB) | 27.3 |
| the gradients' zero fill | 8.5 |
| the attention forward's zero fill of the probabilities and the context | 8.0 |
| the weights' finiteness scan | 6.7 |
| the gradients' finiteness scan | 6.2 |
| the attention backward's zero fill of dq, dk, dv | 4.2 |
| the step | 571.4 |

A scan's verdict is "some element is not finite" and a fill writes zeros, so neither depends on
the partition: NDT-SCAN runs each as one task per chunk and moves the attention's fills into the
tasks that own the slices. Measured in alternated pairs ([scan/](measurements/2026-10-13/scan/)):

| four threads, ms | before (the serial passes) | NDT-SCAN | pair difference |
|---|---|---|---|
| the six passes, in place | 60.9 / 68.4 / 68.3 | 17.4 / 16.2 / 18.6 | -43.5 / -52.2 / -49.7 |
| the step, in place | 571.4 / 710.7 / 602.8 | 591.0 / 549.5 / 568.8 | +19.6 / -161.2 / -34.0 |
| the step, the breakdown harness | 575.4 / 634.9 / 601.2 / 653.9 / 682.9 / 621.1 | 550.9 / 588.2 / 615.7 / 616.6 / 642.6 / 563.4 | -24.4 / -46.6 / +14.6 / -37.2 / -40.3 / -57.7 |
| the forward stage, the breakdown harness | 213.3 / 223.2 / 219.4 / 233.0 / 263.7 / 201.5 | 199.9 / 201.4 / 194.0 / 213.9 / 230.6 / 199.4 | -13.4 / -21.9 / -25.4 / -19.0 / -33.1 / -2.1 |

The 20-step state is the same on both trees at one and four threads
(4bf4e057…). The same in-place record caught a regression of the slice's own: the first
NaN clamp (`x==x ? x : 0`, a third select) cost the SiLU and the SwiGLU VJP their vector code, 22-24
and 26-30 ms a step more, which the step's cell had hidden behind the scans' gain; the clamp as a
maximum and then a minimum in the vector instructions' comparison order keeps it (both rows back to
their old cost). What the rest is now: the GEMMs' in-place time exceeds their synthetic
time (the backward's 43 pairs most), the update's two pooled passes are memory-bound, and the
attention is a tenth of the step at this context -- the next slices' attribution.

## 12. The update's first pass, the attention's products and the pool (NDT-PASS, NDT-ATTN-1..3, NDT-POOL)

§11's in-place record named the next rows: the update's first pass (about 20 ms at four threads:
four 39 MB streams read by a loop GCC had compiled to scalar code -- "control flow in loop"), the
attention (about 40 and 27 ms backward and forward: the GEMM kernel's v1 blocks computing 64-wide
column blocks whole above the causal diagonal, the context's 48 channels padded to 64 with a mask
test per term, and the backward's probability-adjoint tiles evaluated twice), and the pool behind
every barrier. Each slice was drafted, held to the C harness's references bit for bit, and
measured before it landed, in one session on this host
([slices/](measurements/2026-10-13/slices/): five builds -- the tree before, then each slice --
alternated through the in-place instrument of §11, three rounds, four threads, `kblock+exp+reduce`).

### 12.1 The first pass as vector loops (NDT-PASS)

The loop's verdict is now one flag without a branch and its maxima the largest bit patterns read
as int32: a non-negative float orders as its bits, and the one negative value the pass admits, a
-0.0 second moment, reads as INT32_MIN and never wins, as `x > max ? x : max` from +0.0 never took
it -- so on every chunk the pass admits, the maxima are the select loop's bit for bit (a chunk it
refuses has none the update reads). GCC and Clang vectorize the generic, AVX2 and AVX-512 instances.

| four threads, in place, ms | before | NDT-PASS | pair difference |
|---|---|---|---|
| the update's first pass | 23.2 / 20.3 / 24.6 | 12.0 / 14.6 / 13.2 | -11.2 / -5.7 / -11.4 |

### 12.2 The probability adjoint evaluated once (NDT-ATTN-1)

The in-place backward evaluates `dp = dY V^T` per row block in two passes, a first for the rows'
`dot_i` and a second for the score adjoints; both evaluate the same product over the same operands.
A row block now keeps its first 256 columns' tiles between the passes and evaluates only the
tiles past them again (through one spare slot). At T=128 that removes the second evaluation entirely: 0.49M of the two passes' 0.98M dp multiply-adds a head.

| attention kernels alone, 9.84M shape, one thread, best of 21, ms | before | NDT-ATTN-1 |
|---|---|---|
| the backward, three rounds | 14.71 / 14.55 / 15.36 | 12.07 / 12.28 / 12.48 |

### 12.3 The masked products as register blocks (NDT-ATTN-2, NDT-ATTN-3)

The attention's five masked products ran through the GEMM kernel's v1 blocks (4 x 64, written for
the linears): the scores computed their 64-wide column blocks whole above the causal diagonal
(590K multiply-adds a head at T=128 for the 396K the triangle needs), the context padded its 48
channels to 64 and tested the mask per term in every block the diagonal crosses, and the
backward's three products the same. On the AVX-512 instance they now run as blocks of eight rows
with the v1 arithmetic exactly -- the scores 32 keys at a time from their diagonal on, the context
and dq over keys `j <= i`, dv and dk over terms `i >= j`, the channels 48 at a time (three vectors,
masked at the edge) -- and a ragged tail of rows through the kernel itself. The probabilities, the
context and the adjoints are the reference loops' bit for bit; the AVX2 and generic instances keep
the kernel.

| attention kernels alone, 9.84M shape, best of 21, ms | before | NDT-ATTN-1 | the blocks alone | NDT-ATTN-1..3 |
|---|---|---|---|---|
| the forward, one thread | 11.12 / 11.22 / 11.31 | 10.98 / 11.06 / 11.23 | 9.13 / 9.22 / 9.18 | 9.09 / 9.28 / 9.42 |
| the backward, one thread | 14.71 / 14.55 / 15.36 | 12.07 / 12.28 / 12.48 | 11.01 / 10.88 / 11.11 | 8.36 / 8.95 / 8.96 |
| the forward, four threads | 3.36 / 3.72 / 3.89 | 3.79 / 3.56 / 3.67 | 2.71 / 3.61 / 4.16 | 3.61 / 3.12 / 3.71 |
| the backward, four threads | 4.87 / 5.07 / 5.35 | 4.72 / 4.16 / 4.49 | 3.77 / 4.15 / 6.44 | 3.17 / 3.54 / 3.84 |

In place at four threads, alternated in the same three rounds (the step's other rows drifted with
the host, its GEMM rows 20% apart between rounds; the attention rows are the slices' own):

| four threads, in place, ms | before | NDT-ATTN-1 | + NDT-ATTN-2 | + NDT-ATTN-3 |
|---|---|---|---|---|
| the attention forward | 26.6 / 32.1 / 27.6 | 30.8 / 33.4 / 28.5 | 20.1 / 24.8 / 21.0 | 24.1 / 21.6 / 22.8 |
| the attention backward | 46.8 / 39.8 / 44.5 | 36.8 / 39.3 / 43.7 | 34.5 / 42.2 / 38.6 | 23.1 / 26.3 / 27.2 |

The one-thread kernel rows are the slices' clean reading; at four threads each head is one task, so
the kernels' time there is a few tasks' latency over three free cores and the pool's own costs
(§12.4).

### 12.4 The pool's claims and its wait (NDT-POOL)

The four vCPUs are four cores with their own FMA units
([pool/fma-topology.json](measurements/2026-10-13/pool/fma-topology.json): twelve FMA chains in registers
run 141-148 GFLOP/s pinned to one, 284-300 to any pair, 579-584 to all four), yet the GEMMs gained
about 2x from four threads (99 GFLOP/s at one, about 200 at four: a third of the four cores' peak). The
pool's own latency was the first suspect: a worker woke on a condition variable, took the mutex to
read the job and again to report, and task t belonged to thread t % threads, so a thread that woke
late held its share back. On this VM a futex wake costs tens of microseconds. NDT-POOL: every thread
claims the next task from the job's counter (GCC's and Clang's atomics; MSVC keeps the fixed
assignment), the caller publishes a job with the generation's release store, a worker reports with
one atomic decrement and only the last reporter takes the lock (to signal the caller), and a
worker waits 64 pauses, then 512 yields, then sleeps. Which thread runs a task never reached a
result, so every state is the one it was. The budget was chosen in three stages
([pool/pool-latency.json](measurements/2026-10-13/pool/pool-latency.json)): a long pure spin is fast on
a quiet host and slow on a busy one (2,048 pauses: 63-83 us an empty job with two CPUs busy, 8,192
pauses: 281-298), and yields carry a waiting thread through on both.

| four threads, us per job (best of 5 rounds) | the pool before | NDT-POOL |
|---|---|---|
| an empty job of four tasks, back to back | 77.4 / 85.3 | 3.6 / 3.3 |
| the same after a 200 us serial gap | 121.0 / 93.4 | 22.2 / 27.1 |
| sixteen 25 us tasks (100 us of work a thread) | 195.0 / 190.8 | 113.5 / 117.2 |
| thirteen 25 us tasks (100 us) | 165.9 / 161.4 | 107.2 / 111.5 |
| sixty-four 25 us tasks (400 us) | 528.3 / 513.1 | 430.7 / 441.9 |
| sixteen 25 us tasks, two CPUs busy | 253.2 | 224.7 |
| sixty-four 25 us tasks, two CPUs busy | 934.8 | 873.9 |
| sixteen 25 us tasks, three CPUs busy | 414.9 | 433.8 |
| sixty-four 25 us tasks, three CPUs busy | 1513.5 | 1696.9 |

With three of the four CPUs held by other processes -- four pool threads on one free core -- the
largest job is 12% slower than with the old pool's sleeping workers; everywhere else it is faster.
The step's GEMM shapes at four threads, alternated call by call, every result the serial kernel's:

| work-weighted GFLOP/s, 15 shapes, 15 calls each | before | NDT-POOL |
|---|---|---|
| median, three runs | 179.6 / 174.2 / 196.1 | 199.7 / 210.1 / 222.0 |
| best, three runs | 232.4 / 227.5 / 235.1 | 240.3 / 248.3 / 257.0 |
| the claims alone (one protocol, fixed assignment -> claims), median / best | 205.8 / 242.6 | 232.0 / 270.3 |

The step, in place and through the breakdown harness, the trees alternated in each round
([pool/](measurements/2026-10-13/pool/)):

| four threads, ms | before NDT-PASS..ATTN-3 | before NDT-POOL | NDT-POOL |
|---|---|---|---|
| the step in place (median of 5 steps) | 486.6 / 545.6 / 556.8 | 535.8 / 482.1 / 554.4 | 431.3 / 422.1 / 412.6 |
| the breakdown's step | | 423.5 / 538.9 / 481.1 / 509.0 | 454.2 / 376.6 / 411.2 / 414.6 |
| the breakdown's forward stage | | 176.5 / 190.0 / 175.6 / 169.6 | 148.1 / 141.0 / 137.3 / 147.1 |
| the breakdown's backward stage | | 487.9 / 532.8 / 455.3 / 465.3 | 455.1 / 390.3 / 377.0 / 365.4 |

The forward and backward stages are below in 4 of 4 pairs, the step in 3 of 4; in place, every
small pooled row halves (the residual adds 7.6 / 6.8 / 10.4 -> 3.1 / 3.7 / 2.9 ms, the RoPE and
RMSNorm rows alike) -- they were mostly the pool's latency. The 20-step state with all four slices
at one and four threads is `4bf4e057…`, §11's. What the step is now, in place at four threads:
the GEMMs about three quarters of it (the backward's 43 pairs 210-237 ms, the forward 84-93), the
attention 38-40, the update's two passes 22-23, the RMSNorm VJP 13-15, everything else under 9 each.

## 13. The state on 2 MB pages (NDT-PAGES)

§12.4 left the GEMMs at about 200-240 GFLOP/s at four threads, 37-41% of the four cores' 580. One
cost no kernel change could reach was the page size under the operands. The trainer's state -- the
weights, gradients, both moments, activations and scratch, 335 MiB on the 9.84M decoder -- was six
ctypes arrays, and glibc serves an allocation that large from an anonymous mapping it never
advises; under this host's THP mode, `madvise` (most distributions' default), not one of its pages
was huge (`/proc/self/smaps`: AnonHugePages 0 kB under the state). A GEMM walking a transposed
operand's rows -- the weight gradients' `X^T dY` -- touches a new 4 KB page every few rows, and the
second-level TLB's reach is a few megabytes. NDT-PAGES places the six buffers on one private
anonymous mapping advised `MADV_HUGEPAGE` before any page is touched, the first buffer on a 2 MB
boundary and each next 64 bytes past the last one's end (`bcir/hosted/models/native.py`,
`_state_buffers`); where Python has no `MADV_HUGEPAGE` (Windows, macOS), or with `huge_pages=False`,
the buffers are ctypes arrays as before. No kernel's arithmetic depends on an address, so every
state is the one it was: the 20-step state at one and four threads is `4bf4e057…`, §11's and
§12's ([pages/identity-after.json](measurements/2026-10-14/pages/identity-after.json)).

The step, two trainers in one process -- one per placement, the comparison's weights and first
batch -- stepped in alternation, each step after a 30 ms rest, both states identical after every
pair ([pages/step-*.json](measurements/2026-10-14/pages/)); 334-336 MiB of huge pages under the
2 MB arm's state, none under the 4 KB arm's:

| the step, one process, ms | 4 KB pages, median | 2 MB pages, median | paired difference, median | pairs faster on 2 MB |
|---|---|---|---|---|
| four threads, 20 pairs | 403.9 | 349.2 | 51.6 | 19 of 20 |
| four threads, 20 pairs, again | 405.5 | 341.4 | 62.2 | 19 of 20 |
| two threads, 12 pairs | 668.8 | 627.6 | 66.7 | 11 of 12 |
| one thread, 8 pairs | 1101.8 | 1009.9 | 102.9 | 7 of 8 |

The step's GEMMs alone, operands on each placement, alternated call by call, nine calls each,
every result the serial kernel's ([pages/gemm-pages-*.json](measurements/2026-10-14/pages/)): every
one of the 15 shapes is faster on 2 MB pages, the transposed-A weight gradients most.

| GFLOP/s, median of 9, two runs | 4 KB, one thread | 2 MB, one thread | 4 KB, four threads | 2 MB, four threads |
|---|---|---|---|---|
| `X^T dY`, 384 x 384 x 1024 (q and o, 12 calls) | 59.1 / 56.1 | 89.0 / 88.3 | 186.9 / 170.0 | 276.3 / 241.2 |
| `X^T dY`, 1024 x 384 x 1024 (gate and up, 12) | 63.6 / 62.2 | 72.6 / 76.8 | 178.3 / 183.5 | 215.1 / 242.6 |
| `X^T dY`, 384 x 1024 x 1024 (down, 6) | 43.6 / 46.1 | 60.6 / 60.1 | 151.5 / 167.8 | 187.6 / 195.8 |
| `dY W`, 1024 x 384 x 1024 (gate and up, 12) | 82.8 / 79.0 | 97.5 / 95.1 | 236.2 / 255.6 | 292.9 / 301.3 |
| `X W^T`, 1024 x 1024 x 384 (gate and up, 12) | 84.9 / 87.0 | 96.2 / 95.1 | 223.7 / 262.1 | 254.4 / 292.7 |
| the step's 129 calls at their medians, ms | 847.8 / 832.4 | 711.7 / 704.5 | 286.3 / 273.4 | 242.3 / 228.7 |

In place at four threads, the tree before (69954e5) and with NDT-PAGES alternated in three rounds
([pages/attribution-t4-*.json](measurements/2026-10-14/pages/)):

| four threads, in place, ms | before | NDT-PAGES |
|---|---|---|
| the step (median of 5 steps) | 407.8 / 381.0 / 388.0 | 348.0 / 340.0 / 330.6 |
| the GEMMs, backward | 211.7 / 189.4 / 202.4 | 176.7 / 172.3 / 167.4 |
| the GEMMs, forward | 81.2 / 72.5 / 79.1 | 72.3 / 71.3 / 69.3 |
| the RMSNorm VJP | 14.7 / 12.6 / 14.9 | 11.0 / 8.9 / 9.4 |
| the attention, backward and forward | 20.2 / 18.1 / 19.7 and 17.3 / 15.3 / 16.8 | 17.5 / 18.0 / 17.0 and 14.6 / 14.1 / 14.7 |
| the update's commit | 12.0 / 12.0 / 11.8 | 11.0 / 10.5 / 10.8 |

The GEMMs now run at about 215-300 GFLOP/s on the step's large shapes at four threads -- 37-52% of
the four cores' peak, from 26-45% -- so the TLB was part of §12.4's gap, not all of it. The step in
place at four threads is now about 340 ms: the GEMMs about 245 of it, the attention about 32, the
update's two passes about 19, the RMSNorm VJP 9-11, everything else under 7 each. Reproduce:

```bash
CC=gcc python3 -c "from bcir.hosted.models.native import NativeDecoder as N; print(N.build('lib'))"
python3 docs/research/measurements/2026-10-14/pages/pages_ab.py --library lib/bcir_decoder_train.so \
    --data DATA --threads 4 --pairs 20 --out step-t4.json
python3 docs/research/measurements/2026-10-14/pages/gemm_pages.py lib/bcir_decoder_train.so gemm.json 9
```

## 14. The per-row chains as vector lanes (NDT-LANES)

The container restarted after §13, on a faster instance of the same CPU model (Xeon @ 2.80 GHz,
AVX-512, four vCPUs, 1 MiB of L2 a core, 33 MiB of L3); §13's absolute times compare only within
§13. On this host the step in place at four threads (a0ea747) was 236 ms: the GEMMs 166.5 of it
(backward 114.9, forward 51.6), the attention 23.8 (backward 13.2, forward 10.6), the update's two
passes 14.7, the RMSNorm VJP 6.1 and RMSNorm 2.6, the activations' finiteness scan 5.4, everything
else under 3 each.

Each RMSNorm row computes its sum of squares -- and the VJP its dot, the sum of `dy g x` -- as one
chain of rounded adds over its 384 columns in order, and each softmax row its running maximum and
its sum over its columns `j <= i`. At four cycles an add a row's chain is about 1,500 cycles
whatever the vector width, and every kernel ran its rows one at a time. NDT-LANES runs a whole chunk
of 16 rows as the 16 lanes of one AVX-512 vector: 16 columns of the 16 rows are transposed in
registers (the GEMM pack's 16 x 16 transpose) and lane r adds row r's next 16 terms in column
order, the same rounded operations in the same order, so every row's value is the scalar loop's;
the last `width % 16` columns continue each row's chain one row at a time. The softmax's maximum
takes VMAXPS, whose operand rule -- `x > m ? x : m`, the second operand on equality or NaN -- is
the scalar comparison exactly, and a lane takes a column only where its row has it (the diagonal
block's mask). Under the exp contract the exponentials run in the same transposed blocks whole: a
row's pass had sent its `i + 1` elements through the vector instance with the remainder one element
at a time, a quarter of the forward at one thread in a scratch cycle-counter profile; the
diagonal block's upper lanes are stored as the zeros they were. The attention backward's dot -- the
sum of `p_ij dp_ij` over a tile -- runs the same way, and the gamma adjoint's column tasks keep their
16 sums in registers over the 1,024 rows and store them once (they had loaded and stored them every
row). Every state is the one it was: the 20-step state at one and four threads, widest, AVX2 and
portable builds, is `4bf4e057…`
([lanes/identity.json](measurements/2026-10-16/lanes/identity.json));
the C harness holds RMSNorm at 40 x 37 and 48 x 389 and the attention at its reference's shapes
to the scalar references, and `tools/testing/faults/ndt-lanes.json` NL1-NL11 each turn it red.

In place at four threads, the tree before and with NDT-LANES alternated in five rounds
([lanes/attribution-t4-*.json](measurements/2026-10-16/lanes/)):

| four threads, in place, ms a step | before (a0ea747) | NDT-LANES |
|---|---|---|
| the attention, forward | 10.59 / 11.18 / 11.09 / 11.51 / 11.10 | 8.70 / 8.48 / 8.49 / 8.70 / 8.56 |
| the RMSNorm VJP | 5.65 / 7.16 / 6.34 / 6.35 / 6.11 | 4.39 / 4.19 / 4.48 / 4.51 / 4.83 |
| RMSNorm | 2.54 / 2.57 / 2.62 / 2.65 / 2.87 | 1.57 / 1.39 / 1.41 / 1.45 / 1.47 |
| the attention, backward | 13.35 / 14.54 / 13.96 / 14.12 / 14.19 | 13.89 / 13.18 / 13.51 / 14.17 / 13.88 |
| the GEMMs (code unchanged) | 170.0 / 175.7 / 171.8 / 178.6 / 175.9 | 169.0 / 169.3 / 175.7 / 182.2 / 173.4 |
| the step (median of eight) | 243.6 / 248.2 / 245.1 / 254.4 / 248.1 | 237.3 / 237.6 / 244.5 / 254.1 / 240.6 |

The three kernels are faster in every round, together 20.1 -> 14.5 ms of the step; the GEMMs' own
spread between rounds (169-182 ms on unchanged code) is this host's, and the step's median moved
248.1 -> 240.6 ms. The attention backward barely moved: its dot was a small part of it. What remains
of the attention is its products -- the forward's scores and context, the backward's
probability-adjoint tiles (through the general GEMM kernel, an operand packed per tile) and its dv,
dq and dk -- whose v1 arithmetic, a rounded product and a rounded sum per term, runs at half the
FMA rate by construction. Reproduce:

```bash
python3 docs/research/measurements/2026-10-13/scan/attribution.py --src runtime/c --data DATA \
    --threads 4 --steps 8 --out after.json  # and --src BEFORE/runtime/c for the other arm
python3 docs/research/measurements/2026-10-11/train_comparison.py identity --data DATA --steps 20 \
    --threads 1,4 --contract kblock+exp+reduce --variants widest,avx2,portable
```

## 15. RoPE's frequencies once a range (NDT-ROPE)

The step in place after NDT-LANES spent 4.6 ms of its four-thread step in RoPE (2.0 forward, 2.5
backward), more than the RMSNorm VJP. Its kernel computed `angle = pos * pow(base, -2j/dim)` in
double and the angle's cosine and sine rounded to float for every pair `j` and position -- 3,072
`pow`s a call on the decoder's 128 x 24 pairs where 24 values differ -- and then turned that pair in
every head of every sequence at stride `dim`. A position-range task now computes the frequencies
once per range of 64 pairs, each position's angles once (the same double product, the same `cos` and
`sin` rounded to float), and turns a head's pairs contiguously: every element takes the same
rotation. The C harness keeps the kernel as it was as a per-pair reference and holds the serial
kernel and the pool to it at 2 x 40 x 3 x 8 and 1 x 33 x 2 x 260 (three ranges of pairs), forward
and inverse -- the pool test before compared the kernel with itself -- and
`tools/testing/faults/ndt-rope.json` NO1-NO5 each turn it red; the 20-step state at one and four
threads is `4bf4e057…` ([rope/identity.json](measurements/2026-10-16/rope/identity.json)).

| four threads, in place, ms a step | NDT-LANES (28ed912) | NDT-ROPE |
|---|---|---|
| RoPE, forward | 1.98 / 2.05 / 2.05 / 2.55 | 0.82 / 0.95 / 0.96 / 0.97 |
| RoPE, backward | 2.74 / 2.32 / 2.28 / 2.92 | 0.98 / 1.16 / 1.12 / 1.13 |
| the step (median of eight) | 234.0 / 236.9 / 238.9 / 265.6 | 219.2 / 231.8 / 230.7 / 238.1 |

([rope/attribution-t4-*.json](measurements/2026-10-16/rope/), the arms alternated round by round.)

## 16. The probability adjoint from a transposed panel (NDT-ATTN-5)

The in-place backward (§12.2) evaluates each row block's tiles of `dp = dY V^T`, 32 rows by up to 64
columns, through the general GEMM kernel (the v1 contract), which packs the transposed values for
every tile it is given: per head, six tiles each re-packing up to 64 keys' 48 channels, and in a
scratch cycle-counter profile at one thread the tiles were a third of the backward. A backward task
already owns one KV head and runs its query heads in order (§12.3's partition), so it now
transposes that head's values once into its own stack -- `d x t` floats, 6,144 on the decoder, up
to 16,384 (64 KB); a longer context takes the GEMM kernel as before -- and every tile of its query
heads is AVX-512 register blocks of eight rows by 32 columns reading the panel, only the blocks that
hold some `j <= i` (the dot and the score adjoint never read the others). The arithmetic is the v1
kernel's for these operands -- each element from the zero the tile holds, one rounded product and
one rounded sum per channel in order -- so every tile, and every adjoint, is the one it was; the C
harness's attention reference holds dq, dk and dv to the reference loops at its shapes, and
`tools/testing/faults/ndt-attn.json` NA10-NA14 each turn it red; the 20-step state is `4bf4e057…`
([dp/identity.json](measurements/2026-10-16/dp/identity.json)).

| four threads, in place, ms a step | NDT-ROPE (c65f291) | NDT-ATTN-5 |
|---|---|---|
| the attention, backward | 13.02 / 15.88 / 13.24 / 14.03 | 12.53 / 12.74 / 12.20 / 12.14 |
| the attention, forward (code unchanged) | 8.14 / 8.77 / 8.37 / 8.71 | 9.22 / 9.58 / 8.65 / 8.78 |
| the GEMMs (code unchanged) | 170.2 / 181.7 / 165.6 / 177.3 | 189.7 / 194.8 / 175.6 / 170.9 |

([dp/attribution-t4-*.json](measurements/2026-10-16/dp/), alternated round by round.) The backward
is faster in every round; the rounds' unchanged rows moved by up to 15% (the GEMMs 166-195 ms) as
other work on the host came and went, so this record's step medians say nothing either way.

## 17. The four-thread GEMMs, and BCIR against PyTorch at four threads (NDT-GEMM-4T)

§13 put the step's GEMMs at 37-52% of "the four cores' peak", a peak computed from the nominal
clock. On this instance it is measured instead: twelve independent AVX-512 FMA chains a thread
([gemm4t/fma_peak.c](measurements/2026-10-16/gemm4t/fma_peak.c)) read 162-165 GFLOP/s on one thread
and 144-163 each with four running, 574-653 together, across three runs (the reading moves with
the host's other work). Every GEMM call of the step at 742234a,
timed in place with the pool's idle inside it
([gemm4t/gemm-calls-t4.json](measurements/2026-10-16/gemm4t/gemm-calls-t4.json)):

| four threads, in place | calls a step | ms a step | GFLOP/s | idle, ms a step |
|---|---|---|---|---|
| backward gate, up (dY W and dY^T X) | 12 | 50.86 | 380 | 0.77 |
| backward down (dY W and dY^T X) | 6 | 26.16 | 369 | 0.38 |
| forward gate, up (x W^T, 1024 x 384) | 12 | 24.18 | 400 | 0.32 |
| backward q, o (dY W and dY^T X) | 12 | 19.81 | 366 | 0.56 |
| backward k, v (dY W and dY^T X) | 12 | 13.82 | 262 | 0.42 |
| forward down (x W^T, 384 x 1024) | 6 | 11.11 | 435 | 0.34 |
| forward q, o (x W^T, 384 x 384) | 12 | 10.02 | 362 | 0.35 |
| forward k, v (x W^T, 192 x 384) | 12 | 4.78 | 379 | 0.24 |
| backward head (dY E and dY^T X) | 1 | 1.38 | 297 | 0.03 |
| forward head (x E^T, 260 x 384) | 1 | 0.71 | 288 | 0.02 |

The GEMMs are 162.8 ms of the step, 371 GFLOP/s overall -- 57-65% of the four-thread probe -- and
the pool is idle inside them 3.4 ms (2%), so no reordering of the calls or merging of their jobs
can buy more than that. Alone, on warm operands, the same shapes reach 290-437 GFLOP/s at four
threads, the pool busy 86-97% of their wall ([gemm4t/gemm-tasks-t4.json](measurements/2026-10-16/gemm4t/gemm-tasks-t4.json));
a task on the large shapes takes 4-8% longer at four threads than alone, on the narrow weight
gradients (16 and 24 tasks of the whole depth) 50-60% longer and on the k and v input adjoints
(k = 192) 20% -- the step's slowest calls. Cold operands -- each call on the next of six sets, past the L3 -- cost 10%: the
step's GEMMs 155.8 ms warm against 170.8 cold
([gemm4t/gemm-cold-t4.json](measurements/2026-10-16/gemm4t/gemm-cold-t4.json)). The transposed-A
weight gradients, §13's worst shapes on the previous instance (52-84 GFLOP/s at one thread against
94-114 for the rest), are 2-19% behind the same products with A transposed beforehand at one thread
here and ahead of them on four of five shapes at four, with the transpose itself costing more than
the difference ([gemm4t/transposed-a.jsonl](measurements/2026-10-16/gemm4t/transposed-a.jsonl)): a
packed A panel (NDT-TA) would buy nothing on this host. Scratch probes on the previous instance --
row tasks of 64 and 256 rows, prefetch distances 16-32, the K block in L1-sized depth slices with
the accumulators spilled exactly (NDT-KSUB) -- moved nothing beyond noise and were not kept. The
GEMMs' remaining distance from the probe is the kernel's own: the contract fixes each K block's
depth (448 at these shapes), so the 8 x 48 block streams its 86 KB tile from L2.

The 1,000-step language-model run at four threads, BCIR under `kblock+exp+reduce` against PyTorch
2.13 at four intra-op threads, eager and `torch.compile` -- the harness sets PyTorch's thread count
from `--threads` since this record and stores it -- on one host in one session, before NDT-LANES,
NDT-ROPE and NDT-ATTN-5 (a0ea747) and with them (742234a) ([lm/](measurements/2026-10-16/lm/)):

| four threads, 1,000 steps | median step, ms | tokens a second | validation / test cross-entropy | peak RSS, MB |
|---|---|---|---|---|
| BCIR, a0ea747 (two runs) | 247.2 / 249.4 | 4,142 / 4,106 | 1.5003 / 1.6319 | 528 |
| BCIR, 742234a (two runs) | 234.9 / 235.1 | 4,359 / 4,356 | 1.5003 / 1.6319 | 529 |
| PyTorch eager (three runs) | 276.7 / 287.1 / 283.8 | 3,700 / 3,566 / 3,608 | 1.5003 / 1.6319 | 1,123-1,128 |
| PyTorch compiled (two runs) | 274.0 / 259.6 | 3,737 / 3,945 | 1.4999-1.5003 / 1.6318-1.6321 | 1,084-1,208 |

All four BCIR runs end in the same state, `5137e013…`, under two library builds: the three slices
moved no bit over 1,000 steps. In the last round, run back to back, BCIR trains 21% more tokens a
second than eager PyTorch and 10% more than compiled, to the same held-out cross-entropy in under
half the memory. Reproduce:

```bash
python3 docs/research/measurements/2026-10-16/gemm4t/gemm_calls.py --src runtime/c --data DATA \
    --threads 4 --steps 8 --out gemm-calls.json
python3 docs/research/measurements/2026-10-11/train_comparison.py train --data DATA --task lm \
    --backend bcir --threads 4 --contract kblock+exp+reduce --eval-every 100 --lr-scale 0.25
python3 docs/research/measurements/2026-10-11/train_comparison.py train --data DATA --task lm \
    --backend torch-eager --threads 4 --eval-every 100 --lr-scale 0.25   # and torch-compiled
```

## 18. The beta scaling in the first K block (NDT-BETA)

§17 closed on a guess: that the kernel's distance from the probe is its tile streaming from L2, the
depth the contract fixes being 448. A probe of the AVX-512 kernel over the pool's own partition --
128 rows by 48 columns, its tasks in order on one thread or on a pool of four, every variant's C
compared with the kernel's bit for bit ([beta/gemm_probe.c](measurements/2026-10-16/beta/gemm_probe.c),
built against the unit at ad77c84) -- says otherwise. The inner loop alone over one packed tile
sustains 162-171 GFLOP/s at every depth from 64 (a 12 KB tile, in L1) to 448 (84 KB, in L2), with
A's rows 448 or 1,024 floats apart
([probe-ceiling.jsonl](measurements/2026-10-16/beta/probe-ceiling.jsonl)): the FMA probe's rate, so
neither the depth nor the tile's cache level costs anything. Timed by the cycle counter inside the
tasks at one thread, the rest of a task is the tile's pack, 5.6-12.6% of its cycles, and the pass
that scales its elements of C by beta before its K blocks, 1.6-8.9% -- the larger shares where the
depth is short, 192 or 384 ([probe-split.jsonl](measurements/2026-10-16/beta/probe-split.jsonl)).
Fourteen variants that keep the arithmetic, interleaved round by round in one process
([probe-t1.jsonl, probe-t4.jsonl](measurements/2026-10-16/beta/)), found nothing else that holds:
each K block in L1-sized depth slices with the accumulators spilled exactly (NDT-KSUB's form, five
slice shapes) 1.6-6.6% slower than the kernel over the step's shapes at one thread and -0.4 to
+4.9% at four; the packed tile prefetched 4 or 8 depths ahead, or its source 8, 16 or 32 rows ahead
inside the pack, -1.2 to +5.0% at one thread and -5.9 to -0.1% at four; row tasks of 64 rows +8.1%
and +5.8%, of 512 -3.4% and +1.0%, of 256 -2.0% and -6.7% -- and 256 rows with the pack's 16-row
prefetch -1.3% and -1.6%, so a four-thread difference of that size is within what the next variant
moves. The two candidates, 256-row tasks and the pack's prefetch, went to alternated A/Bs in place
(§19: neither kept).

The pass is the loss with a structural fix. Under the blocked contract each element's K block runs
from zero and is then added as `c + alpha*P` to C already scaled, and the first K block's epilogue
touches C anyway: NDT-BETA drops the pass (the pool's tasks had scaled their elements of C before
their K blocks, the serial path the whole of C) and the blocked kernels scale C in their first K
block. Under beta 0 that block adds its terms to +0.0 without reading C (its lanes masked out of the
load), under any other beta it scales the loaded C by the rounded product the pass computed, and a
later block adds to C as stored -- the same rounded operations in the same order, so every element
is the one it was; the generic instance does the same in plain C. A call of depth zero (no K block
runs) and the v1 and fused contracts keep the pass. On AVX-512 the block's 24 accumulators, three
tile rows and a broadcast hold 28 of the 32 vector registers through the depth loop, so beta and
the two lane masks are formed after it: the first form, with beta live across the loop and the
epilogue branching per block, spilled accumulators every depth and ran 2-4x slower. The C harness's
blocked-contract test now holds the serial entry, both pool partitions and every instance the host
runs to `beta * c` at a depth of zero under beta 0, 0.5, 1 and -2, and to +0.0 for an A of zeros at
alpha -1 under beta 0 at depths 1, 449 and 897 (a zero product times a negative alpha is -0.0; the
pass's zero plus that term is +0.0, and so must the fold's be);
`tools/testing/faults/ndt-kblock.json` NK3b and NK11-NK17 each turn it red, and NK10 and
`ndt-threads.json` NT3 follow the lines the slice rewrote. The GEM+ row `train.mm.kblock.mismatch`
had to widen with it: a later K block's epilogue is now its own code, and the row's cases deeper
than one block all multiplied by an alpha the binary32 product scales exactly (0.5 or 1), so a fused
rounding there (NK3, `test_the_rows_fire_on_wrong_kernels_and_plans`) went unseen until the
897-deep case took alpha 1.1 -- the cases' count stays 40, and the row's witness test gained the
fold's own wrong copies (c read under beta 0, c scaled at every K block). The 20-step state at one and four
threads, widest, AVX2 and portable builds, is `4bf4e057…`
([beta/identity.json](measurements/2026-10-16/beta/identity.json)).

The step's twelve GEMM shapes through the public entries, at the betas the trainer passes, the two
builds alternated in twelve rounds
([beta/gemm_entries.c](measurements/2026-10-16/beta/gemm_entries.c),
[entries-t1.jsonl, entries-t4.jsonl](measurements/2026-10-16/beta/); every C's digest equal across
the builds and the rounds):

| the step's GEMM shapes alone, ms a step | ad77c84 | NDT-BETA |
|---|---|---|
| four threads, median of the rounds (10 of 12 rounds faster) | 198.6 | 175.3 |
| four threads, each shape's minimum | 162.1 | 144.8 |
| one thread, median of the rounds (7 of 12 faster) | 551.3 | 546.7 |
| one thread, each shape's minimum | 466.7 | 464.7 |

At four threads the gain is where the pass was large against the products, by the minima: the
depth-384 forward products 18-22%, the input adjoints of q and o 21% and of k and v (depth 192,
accumulating) 37%, the weight gradients 9-13%. At one thread the slice is neutral over the step.
One shape is slower by every estimator at both counts, the gate and up projections' input adjoint
(depth 1,024, beta 0, B untransposed): 4% at one thread and 3-12% at four -- its first K block now
meets C's lines where the pass had brought them in, a guess the next probe (C prefetched ahead of
the block's epilogue) can test. In place at four threads, alternated round by round with the tree
before ([beta/attribution-t4-*.json](measurements/2026-10-16/beta/)):

| four threads, in place, ms a step | before (ad77c84) | NDT-BETA |
|---|---|---|
| the GEMMs, backward | 138.1 / 137.5 / 137.4 / 131.9 | 117.9 / 118.5 / 118.0 / 125.2 |
| the GEMMs, forward | 64.1 / 62.2 / 62.1 / 64.4 | 63.1 / 59.5 / 59.7 / 63.6 |
| the step (median of eight) | 268.1 / 266.5 / 277.4 / 267.4 | 258.2 / 250.3 / 247.5 / 268.5 |

The GEMMs are faster in every round, 199.6 -> 179.5 ms a step (medians of the rounds), the step in
three of four, 267.8 -> 254.2. This session's host is slower than §17's -- the same code's GEMMs
199.6 ms a step here against 162.8 there -- so the comparison is the alternated pair, never a
number against §17's. Reproduce:

```bash
git show ad77c84:runtime/c/bcir_tensor.c > U/bcir_tensor.c && git show ad77c84:runtime/c/bcir_tensor.h > U/bcir_tensor.h
cc -O3 -ffp-contract=off -fno-math-errno -pthread -I U -DBCIR_TENSOR_C='"U/bcir_tensor.c"' \
    docs/research/measurements/2026-10-16/beta/gemm_probe.c -o probe -lm
./probe ceiling; ./probe 9; ./probe pool 9; ./probe split
cc -O3 -ffp-contract=off -fno-math-errno -pthread -I U docs/research/measurements/2026-10-16/beta/gemm_entries.c \
    U/bcir_tensor.c -o before -lm   # and -I runtime/c ... runtime/c/bcir_tensor.c -o after
for r in $(seq 12); do ./before 9 4; ./after 9 4; done   # REPS THREADS; and at THREADS 1
python3 docs/research/measurements/2026-10-13/scan/attribution.py --src runtime/c --data DATA \
    --threads 4 --steps 8 --out after.json  # and --src BEFORE/runtime/c, alternated
```

## 19. The linears that read one input as one job (NDT-SCHED-1)

After NDT-BETA the step runs its 43 linears as 43 GEMM jobs forward and 43 pair jobs backward, each
behind its own barrier. Five of a layer's seven read one input -- the attention's q, k and v the
first norm's output, the MLP's gate and up the second's -- and their input adjoints add into one
dx, so the backward ran their pair jobs one after another, the narrow k and v (192 columns, 32
tasks) leaving threads idle at each job's tail. `bcir_tensor_mm_group_pool` (NDT-SCHED-1) runs such
calls as one job: a chain of up to four calls into one C -- the same c, m and n, so one task index
names one tile in all of them, and a tile's task runs each chained call's task for that tile in
turn, which scales the tile by that call's beta and adds its terms, the order the calls would run
whole -- and up to four independent calls, each with its own partition (the pair job's second GEMM,
generalized). The forward's q, k, v and gate, up are independent calls; the backward's input
adjoints are the chain (the first at the call's beta, the rest adding, as the separate calls
added) and the weight adjoints the independent calls -- six barriers fewer a layer. A provider's
GEMM, one thread or no pool, more than four calls of a kind, or a chain whose calls do not share c,
m and n run the calls one after another as before; every element receives the same operations in
the same order, so the results are the separate calls' bit for bit. The C harness's pool test holds
-- at each of its shapes, under the three GEMM contracts and at every thread count -- a chain of
three (the shape at its beta, at half the depth adding, at a quarter's scale) beside two
independent products, the independent products alone, the chain alone and both serial fallbacks (a
chain whose calls differ in n, five calls) to the pool entries run one after another;
`tools/testing/faults/ndt-threads.json` NT13-NT18 each turn it red; the 20-step state at one and
four threads, widest, AVX2 and portable builds, is `4bf4e057…`
([sched1/identity.json](measurements/2026-10-16/sched1/identity.json)).

In place at four threads, alternated round by round with NDT-BETA's tree and the tree before it
(§18's rounds; [sched1/attribution-t4-sched1-*.json](measurements/2026-10-16/sched1/)):

| four threads, in place, ms a step | NDT-BETA | NDT-SCHED-1 |
|---|---|---|
| the GEMMs, backward | 117.9 / 118.5 / 118.0 / 125.2 | 113.2 / 115.7 / 110.7 / 108.8 |
| the GEMMs, forward | 63.1 / 59.5 / 59.7 / 63.6 | 59.3 / 59.4 / 59.4 / 54.1 |
| the step (median of eight) | 258.2 / 250.3 / 247.5 / 268.5 | 242.0 / 245.2 / 240.4 / 230.3 |

Every row is faster in every round: the GEMMs 179.5 -> 171.3 ms a step (medians of the rounds),
the step 254.2 -> 241.2 -- with NDT-BETA, 199.6 -> 171.3 and 267.8 -> 241.2 against ad77c84. The
attribution's categories count the grouped calls (`linears(`, `linears_backward(`) with the GEMMs
since this slice.

Four more bit-identical candidates on top of this slice were measured in place and not kept: the
GEMM's row tasks 256 rows high (each B tile packed half as often; §18's one-process probe had read
it 7% faster at four threads), the pack's rows of B prefetched 16 depths ahead into L2, the first K
block's rows of C prefetched into L2 before its depth loop (for the shape NDT-BETA slowed), and the
activations' finiteness scanned layer by layer while each layer's 26 MB are still in the last-level
cache. In two A/Bs against the tree, the arms' order rotated round by round (five rounds of six
arms, then eight of four), the GEMMs' medians were 196.1, 179.4 and 176.3 ms a step (256 rows, the
pack's prefetch, C's prefetch) against the tree's 189.8, and then 169.8 and 182.9 (the pack's
prefetch, C's prefetch) against 171.9 -- C's prefetch 7% ahead in one run and 6% behind in the next
-- and the layer-by-layer scan's own row 7.16 against 7.22 and 6.79 against 6.56 ms. All of it is
within what an arm with the tree's own GEMM code moves between runs here (the scan arm's GEMMs read
177.3 and 169.6 against the tree's 189.8 and 171.9).

## 20. BCIR against PyTorch at four threads on 78cbd29

The 1,000-step language-model run at four threads, BCIR under `kblock+exp+reduce` with NDT-BETA and
NDT-SCHED-1 (78cbd29) against PyTorch 2.13 eager and `torch.compile` at four intra-op threads, in
two rounds in one session -- BCIR, eager, compiled, BCIR, then eager, BCIR, compiled
([lm/r4-\*, r5-\*](measurements/2026-10-16/lm/)):

| four threads, 1,000 steps | median step, ms | tokens a second | validation / test cross-entropy | peak RSS, MB |
|---|---|---|---|---|
| BCIR, 78cbd29 (three runs) | 241.8 / 236.7 / 256.6 | 4,236 / 4,327 / 3,991 | 1.5003 / 1.6319 | 529 |
| PyTorch eager (two runs) | 362.6 / 369.7 | 2,824 / 2,770 | 1.5003 / 1.6319 | 1,134-1,142 |
| PyTorch compiled (two runs) | 321.6 / 310.3 | 3,184 / 3,300 | 1.4998-1.5003 / 1.6319-1.6320 | 1,060-1,141 |

All three BCIR runs end in `5137e013…`, the state of §17's runs: the two slices moved no bit over
1,000 steps. Round by round BCIR trains 44-53% more tokens a second than eager PyTorch and 21-36%
more than compiled, to the same held-out cross-entropy in under half the memory. The gap is wider
than §17's (21% and 10%) for two reasons this record cannot separate from each other: the two
slices, which took 10% from the step in place in alternation with the tree before them (§18-§19),
and this session's host, on which PyTorch's arms ran a quarter slower than in §17's (eager
276.7-287.1 there, compiled 259.6-274.0) and BCIR's, with the two slices, 1-9% slower than
742234a's there. On §17's host the alternated pairs' 10% would put BCIR near 4,800 tokens a
second -- a projection, not a measurement.

## 21. The step's passes merged, and the attention's fills (NDT-FUSE, NDT-ATTN-6)

After §19 a third of the four-thread step is outside the GEMMs -- 68.2 of 228.9 ms a step in place,
in this section's first arm -- and five of its passes were there only because a neighbouring pass
stopped short: the SwiGLU forward's SiLU and its product with the up projection, two passes over each
layer's 8 x 128 x 1,024 hidden activations; the residual's two adds a layer after the RMSNorm VJPs
had stored their dx in scratch; the gradients' zero fill, 9.84M floats written before the backward's
first writer; the backward's closing finiteness scan of the same 9.84M, which the update's first pass
reads again; and in the attention, the fills that set the probabilities, the context and the
backward's adjoints to zero under kernels that then add their first terms to the zeros.

NDT-FUSE merges the first four, each the same operations in the same order:

- `bcir_tensor_silu_mul_pool` (libm's SiLU, the v1 contract) and `bcir_tensor_silu_mul_v2_pool` (the
  BCIR exponential) compute y = silu(x)·u in one pass -- s = x·σ(x) rounded, then s·u rounded, the
  two roundings the two passes made. A provider's SiLU keeps the two passes.
- `bcir_tensor_rms_backward_into_pool` takes `BCIR_TENSOR_RMSB_ADD`, which adds each element of the
  VJP, rounded as its store rounded it, to dx in place -- the residual's adjoint the separate add
  read -- and `BCIR_TENSOR_RMSB_FIRST`, which starts the gamma adjoint's column sums from +0.0 instead
  of reading dg.
- A fresh backward (accumulate 0) on BCIR's GEMM lets each gradient's first writer start it: the
  weight adjoints run at beta 0, which since §18 adds their terms to +0.0 without reading C, and the
  gamma adjoints under `BCIR_TENSOR_RMSB_FIRST`. A sum begun at +0.0 is never -0.0 under
  round-to-nearest, so the fill's +0.0 added to it had changed nothing. An untied embedding is still
  filled (its first writer, the scatter, adds to what it holds), and a provider's GEMM keeps the fill
  and beta 1.
- Inside one run call the last micro-batch's closing gradient scan is the update's first pass, which
  reads every gradient anyway; a non-finite gradient is still reported as `BCIR_DT_NUMERIC` ahead of
  the update's own `BCIR_DT_INVALID`, the order the separate scans gave, and a run's later
  micro-batches skip the opening scans their predecessor's closing ones answered. The
  language-model run calls `run` one step at a time with one micro-batch, so of the two only the
  deferred closing scan reaches it; nothing is carried from one call, or one step, to the next.

NDT-ATTN-6 takes the fills out of the AVX-512 attention's way: the scores', the context's and dq's
register blocks start from +0.0 in registers on a head's first write, and dk's and dv's on a KV
head's first query head (the group's later heads add to what is stored, as the reference does); the
probabilities' fill covers only the future columns past a row's blocks and the rows the v1 kernel
takes (`att_v1_rows`: on the AVX-512 instance the rows past the last multiple of eight, elsewhere
every row), the scores' last channel
block stores its future lanes as +0.0, as the fill had left them, and the scores' key panel is packed
by 16 x 16 transposes in registers (`kb512_transpose16`, the GEMM pack's) instead of a scalar gather.

Witnesses: the C harness's `fused_passes` holds the one-pass product on both contracts -- NaN,
infinities, ±0.0, a subnormal and overflowing operands, in place on either input -- to the two
passes; the VJP's four modes over 40 x 37 and 48 x 389 (whole 16-row chunks and a part one) to the
separate VJP and add, a column of -0.0 terms starting at +0.0; a NaN-filled gradient under
accumulate 0 to the zero-filled one, on a tied and an untied fixture under four contracts; a run's
first micro-batch refusing a NaN weight before the forward writes the arena; and a crafted state
whose activations are finite and whose gradient overflows, refused by the backward alone, by a run,
by a run on an invalid beta power (the backward's verdict first), by a two-micro-batch run at the
first micro-batch's closing scan (the second never run), while a finite step on that power reports
the update's refusal. `tools/testing/faults/ndt-fuse.json` NU1-NU14 and NZ1-NZ7 each turn the
harness red (`ndt-exp.json` NE5, `ndt-lanes.json` NL4 and `ndt-scan.json` NS6-NS7 re-anchored on the
code they guard, NS6-NS7 on the fills that remain). An
old-against-new differential ([fuse/statedump.c](measurements/2026-10-16/fuse/statedump.c)) writes
every byte the API can observe -- return codes, events, losses, norms, and the weights, moments and
gradients after every call -- over runs, single steps, the public backward over a gradient of NaN,
-0.0 and stale values, accumulated runs and the refusals, for three small specs under the twelve
contracts at one, two and four threads, and hashes the same schedule for the 9.84M spec at four:
8b4f23d and the two slices write the same 376,368,768 bytes and the same digest. The 20-step state at
one and four threads, widest, AVX2 and portable builds, is `4bf4e057…`
([fuse/identity.json](measurements/2026-10-16/fuse/identity.json)).

In place at four threads, the three arms in turn -- 8b4f23d, with NDT-FUSE, with both -- their order
rotated round by round over eight rounds ([fuse/attribution-t4-\*](measurements/2026-10-16/fuse/);
the attribution's SiLU row counts the one-pass product, and it instruments the backward's and the
update's bodies, which the public entries now wrap):

| four threads, in place, ms a step (medians of eight rounds) | 8b4f23d | NDT-FUSE | both |
|---|---|---|---|
| the gradients' zero fill / closing scan | 1.37 / 1.55 | 0 / 0 | 0 / 0 |
| SiLU + the gate product | 2.29 + 1.24 | 2.68 | 2.55 |
| the residual adds / the RMSNorm VJP | 2.64 / 4.72 | 1.32 / 5.58 | 1.17 / 5.26 |
| the attention, forward / backward | 9.04 / 12.32 | 9.01 / 12.59 | 7.59 / 11.61 |
| everything but the GEMMs | 68.2 | 64.5 | 61.3 |
| the GEMMs (their code unchanged) | 163.2 | 163.8 | 163.4 |
| the step (median of eight steps) | 228.9 | 231.6 | 221.8 |

Everything but the GEMMs is lower in seven of the eight rounds with NDT-FUSE and in all eight with
both: the merged passes took 3.7 ms a step from it and the attention's fills 3.2 more. The NDT-FUSE
arm's step reads 2.7 ms slower than 8b4f23d's -- inside the step's spread between rounds (216-246 ms
for that arm; the GEMMs alone moved 156-175 ms between rounds of the first) -- and with both slices
the step is 7.1 ms, 3.1%, faster. The residual adds left are the forward's two a layer; the RMSNorm
VJP is slower by the adds it took, and the one-pass product costs a little more than SiLU's pass
alone did and less than the two together.

## 22. BCIR against PyTorch at four threads on 64bfd62

The 1,000-step language-model run at four threads on NDT-FUSE and NDT-ATTN-6 (64bfd62), BCIR under
`kblock+exp+reduce` against PyTorch 2.13 eager and `torch.compile` at four intra-op threads, one round
in §21's session -- BCIR, eager, compiled, BCIR ([lm/r6-\*](measurements/2026-10-16/lm/); the runs
started from 8b4f23d with the slices' sources in the working tree, whose SHA-256s are the commit's
blobs, [r6-64bfd62-sources.sha256](measurements/2026-10-16/lm/r6-64bfd62-sources.sha256)):

| four threads, 1,000 steps | median step, ms | tokens a second | validation / test cross-entropy | peak RSS, MB |
|---|---|---|---|---|
| BCIR, 64bfd62 (two runs) | 227.3 / 226.0 | 4,504 / 4,531 | 1.5003 / 1.6319 | 529 |
| PyTorch eager | 353.2 | 2,900 | 1.5003 / 1.6319 | 1,123 |
| PyTorch compiled | 296.9 | 3,449 | 1.5001 / 1.6320 | 1,119 |

Both BCIR runs end in `5137e013…`, the state of §17's and §20's runs: no bit moved over 1,000 steps.
BCIR trains 55-56% more tokens a second than eager PyTorch and 31% more than compiled, to the same
held-out cross-entropy in under half the memory. Against §20's session BCIR is 4.0-6.5% faster
(236.7-241.8 ms there, leaving out its 256.6) and PyTorch 3-8% faster (eager 362.6 / 369.7, compiled
310.3 / 321.6 there), so this session's host is a little quicker than §20's; the slices' own share
is §21's alternated in-place reading, the step 3.1% faster.
