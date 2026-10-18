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

## 23. The tiles packed once, measured; the attention's diagonal terms in registers (NDT-PREPACK, NDT-ATTN-7)

§18's probe put a one-thread task's tile pack at 5.6-12.6% of its cycles, and a pooled blocked call
packs each 48-wide column block's tiles once per 128-row task -- eight times over the step's
1,024-row products. NDT-PREPACK's prototype ([prepack/prototype.py](measurements/2026-10-16/prepack/prototype.py))
gives the pool a workspace and runs a pack phase before the compute phase: one task per call and
column block packs every K block of it with the variant's own pack, the calls of one job that read
one B share one copy, and the compute tasks read their tiles there (a call whose tiles do not fit
packs per task, as before). The tiles are the same floats either way, so every product is too -- each
call's C hashes the same from both trees. It is slower. The step's twelve GEMM shapes alone at four
threads ([beta/gemm_entries.c](measurements/2026-10-16/beta/gemm_entries.c), the two trees in turn over eight rounds,
[prepack/gemm-entries-t4.jsonl](measurements/2026-10-16/prepack/gemm-entries-t4.jsonl)) take 167.4 -> 184.1 ms of calls
a step weighted by their counts (+10.0%), eleven of the twelve shapes slower; in place at four
threads, eight rounds rotated ([prepack/attribution-t4-\*](measurements/2026-10-16/prepack/)), the GEMMs 173.0 -> 178.4
ms a step and the step 235.4 -> 242.0, the prototype ahead in two rounds of eight. A task's own pack
reads B's lines, which every core holds shared, into its own L1 and L2; the pack phase's tiles are
lines one core has just written, which the other cores then read from that core's cache, and the
phase adds a pool run and its barrier to every call -- so sharing the tiles saves only the copy's
instructions, the part §18's probe showed the inner loop does not wait on. Not kept.

With the GEMMs at the kernel's own rate, the attention was the step's largest block of arithmetic
outside them: 19.2 ms a step in §21's last arm for about 1.8 GFLOP, a quarter of the GEMMs' rate.
Cycle counters around its phases at the language model's shapes on four threads (a scratch build, so
its shares, not its times) put 62% of its task time in four products: the forward's context (14%),
the backward's dv (18%), dq (16%) and dk (14%) -- the scores 10%, the dp tiles 9%, the softmax's
exponentials and sums 6%. The context's kernel (`att512_context`, which dq runs too) and the
transposed products' (`att512_transposed`: dv and dk) took a block's diagonal terms -- the keys, or
terms, only some of its eight rows take -- in a loop bounded by the term (`for (r=dj;r<8;r++)`,
`for (r=0;r<=di;r++)`), and so indexed the block's register arrays by a run-time row. At the
trainer's -O3 GCC 13.3 kept the arrays in memory: `att512_context` stored 21 of its 24 accumulators
to the stack on every step of its main loop, and `att512_transposed` moved them through the stack
around its diagonal terms; Clang 18.1 and 23.1 spilled the context's as well.

NDT-ATTN-7 runs the diagonal terms over all eight rows and masks a row that does not take a term out
of its sum: `_mm512_mask_add_ps`, merge-masked, so the row's accumulator stays the one it held, NaN or
not, and the skipped row's product -- of the probability or adjoint past its diagonal inside the
block, the +0.0 the scores stored there -- is discarded. Every register array is then indexed by a
constant, and at -O3 neither kernel has a stack store or load under GCC 13.3, Clang 18.1 or Clang
23.1 (at the harness's -O2 GCC spills neither kernel before or after, and Clang keeps the context's
main loop rolled before and after -- Clang 18 now the transposed kernel's masked loop too).

Every element is the one it was. The C harness's attention reference, finite and not, holds under
GCC and Clang, C11 and C2x, -O2 and -O3; the old-against-new differential of every byte the API can
observe ([fuse/statedump.c](measurements/2026-10-16/fuse/statedump.c)) writes the same 376,368,768 bytes (sha256
`16b0a686…`) and the same 9.84M digest (`41018dd3e5a5e469`) from 64bfd62's sources and this slice's;
and a differential over twelve shapes and four input sets ([attn7/nan_payloads.c](measurements/2026-10-16/attn7/nan_payloads.c),
[attn7/nan-payloads.json](measurements/2026-10-16/attn7/nan-payloads.json)) leaves every finite set's bytes unchanged
and, with each NaN read as one, every set's values the same under both compilers on both trees. What
can move is which of two NaN payloads survives a sum that meets both: it differs between the trees
on the sets that carry two payloads, as it already differed between GCC's and Clang's builds of the
old kernels and between every build and the scalar reference (whose differences, on all four builds,
are NaN for NaN). The contract is value for value and NaN for NaN, and no NaN reaches the trainer's
state: the activation scan and the gradient checks refuse the step first. `ndt-attn.json` NA5 and
NA8 (a diagonal term reaching every row) are re-anchored onto the masks and NA15-NA16 (a skipped
row's sum zero-masked where the add must merge) are new -- 16 injected, 16 caught. The 20-step state
at one and four threads, widest and AVX2-capped builds, is `4bf4e057…`
([attn7/identity.json](measurements/2026-10-16/attn7/identity.json); the portable build compiles none of the AVX-512
kernels).

Measured in place at four threads, 64bfd62's sources and this slice's in turn, the order rotated,
eight rounds ([attn7/attribution-t4-\*](measurements/2026-10-16/attn7/)):

| ms a step, medians of eight rounds | 64bfd62 | NDT-ATTN-7 | rounds faster |
|---|---|---|---|
| the attention forward | 7.63 | 6.65 | 8 of 8 |
| the attention backward | 12.03 | 10.06 | 8 of 8 |
| everything but the GEMMs | 62.2 | 57.8 | 8 of 8 |
| the GEMMs (their code unchanged) | 169.1 | 154.4 | -- |
| the step (median of eight steps) | 232.3 | 213.3 | -- |

The attention is 3.0 ms a step (15%) faster in every round. The GEMMs, whose code did not change,
moved 142-195 ms between this session's rounds, so the step's own difference is mostly the host's.
Alone ([attn7/attention-t1-t4.jsonl](measurements/2026-10-16/attn7/attention-t1-t4.jsonl): the kernel at the language
model's shapes, [attention_time.c](measurements/2026-10-16/attn7/attention_time.c), the two trees in turn over eight
rounds) the forward takes 4.15 -> 3.84 ms a layer and the backward 5.83 -> 5.27 at one thread, every
round; at four threads 1.18 -> 1.16 and 1.78 -> 1.72, inside the host's noise (three and four rounds
of eight) -- in place, among the step's other passes, the stores the loops no longer make are worth
more than alone.

## 24. BCIR against PyTorch at four threads on 14b5941

The 1,000-step language-model run at four threads on NDT-ATTN-7 (14b5941), BCIR under
`kblock+exp+reduce` against PyTorch 2.13 eager and `torch.compile` at four intra-op threads, one round
in §23's session -- BCIR, eager, compiled, BCIR -- every run from the committed tree
([lm/r7-14b5941-\*](measurements/2026-10-16/lm/)):

| four threads, 1,000 steps | median step, ms | tokens a second | validation / test cross-entropy | peak RSS, MB |
|---|---|---|---|---|
| BCIR, 14b5941 (two runs) | 211.0 / 225.6 | 4,853 / 4,538 | 1.5003 / 1.6319 | 529 |
| PyTorch eager | 304.5 | 3,362 | 1.5003 / 1.6319 | 1,137 |
| PyTorch compiled | 310.5 | 3,298 | 1.4998 / 1.6320 | 1,137 |

Both BCIR runs end in `5137e013…`, the state of §17's, §20's and §22's runs. BCIR trains 35-44% more
tokens a second than eager PyTorch and 38-47% more than compiled, to the same held-out cross-entropy
in under half the memory; its 211.0 ms is the fastest 1,000-step run recorded here. The session moved
PyTorch more than the slice moved BCIR: against §22's, eager's median step is 14% shorter (353.2 ms
there) and compiled's 5% longer (296.9), and the two BCIR runs 14.7 ms apart are the host's spread inside one session -- the
slice's own share is §23's alternated in-place reading, the attention 3.0 ms a step faster.

## 25. The activations' finiteness from their writers (NDT-SCAN-W)

After NDT-ATTN-7 the largest pass outside the GEMMs and the attention was the forward's verdict on
its activations: every element of the arena scanned for a value that is not finite after the forward
had written them all, 171 MB for the 9.84M decoder read back from memory -- 6.0 ms of the four-thread
step, at the host's memory bandwidth (the attribution below). NDT-GEMM-CAND had scanned each layer's
26 MB when the layer was done, hoping to find them in the last-level cache, and found 6.6-6.8 ms: the
shared L3 does not keep a layer's activations between their writes and the scan.

NDT-SCAN-W takes the verdict where the values are still in the core's cache: in the tasks that wrote
them. `bcir_tensor_rms_scan_pool`, `bcir_tensor_silu_mul_scan_pool` (and its `_v2`) and
`bcir_tensor_attention_scan_pool` (and its `_v2`) compute their plain entries' values, bit for bit,
and set a verdict when one of their regions holds an element that is not finite after the call; each
task reads back the part it wrote into its own byte of the caller's stack, and the caller ORs them (a
job of more tasks than the bytes' bound is scanned after it on the pool -- the same verdict). A
region's verdict may come from a region its value reaches, and each is a property of the kernel
alone: an RMSNorm answers for its input rows by its output (an infinite or NaN element makes its
row's 1/rms 0 or NaN and that element of the output NaN), the SwiGLU product for its gate and up
projections by its product (a non-finite gate makes the SiLU infinite or NaN -- `-inf` times the
sigmoid's 0 is NaN -- and a non-finite factor makes the product so), and the attention for its
queries, values and probabilities by its context (a query that is not finite makes every score of its
row infinite or NaN, so the row's probabilities NaN, and a NaN probability or a value that is not
finite enters a context sum -- every key's own row takes its term). The keys it reads itself, from the
first query head of each group: a key whose scores are all `-inf` weighs 0 in every row, so the
softmax can hide it, and no later activation would show it. The trainer's forward calls the three for
every region of the arena -- each layer's two norms (the residual stream, and the attention's sum), the
attention, the SwiGLU product, the final norm -- and scans only the logits after the head; a
provider's SiLU, which answers for nothing, has its three regions scanned after it. Every element is
in exactly one call's regions, each call after its regions' last write, so the verdict is the whole
scan's, and the forward runs to the end whatever an earlier layer held, as before.

Witnesses. The C harness's `writer_verdicts` holds each scan entry to its plain entry bit for bit and
its verdict to a scan of every region as the call left it -- NaN and both infinities at the ends of the
entries' task chunks in every input region (560 cases on no pool and a pool of three), outputs that
overflow from finite inputs, a verdict already set left set, and the key the softmax hides (`+inf` in
one channel, every query of its group negative there: probabilities and context finite, the verdict
the keys'). Through the trainer, with every other activation finite: that key behind RoPE (layer 0's
queries and one key on the slowest rotation pair, so both of its channels are `+inf` and every score
of it `-inf`), refused by the forward and the backward with the gradient untouched; logits that
overflow alone (an untied head at `FLT_MAX`), refused; a provider's SiLU that answers 0 for an
infinite gate, refused by the scans after it -- and each state without its overflow admitted. The
harness's Python test builds C23 once more with the bytes' bound at two tasks, so every entry takes
its after-the-job scan. `tools/testing/faults/ndt-scanw.json` SW1-SW17 -- each scan one element or
row short, the wrong region read, the keys never read or read for the first batch only, the context
one row short, the bytes folded from the first, a set verdict cleared, each fallback dropped, the
vector scans' tails skipped, the forward dropping the attention's verdict, the logits unscanned, the
provider's SiLU trusted -- 17 injected, 17 caught. An old-against-new differential of the verdict
([scanw/verdicts.c](measurements/2026-10-17/scanw/verdicts.c)): three specs (the harness's tied and
untied fixtures and a wider one, four query heads over two KV heads, two sequences of nine), the
twelve contracts and pools of one, two and four threads, ten crafted states each -- the finite
control, the hidden key, an overflowing head, final gamma, up, o and v projection, scores past
`FLT_MAX`, and the hiding provider SiLU, finite and over an overflowing gate -- through the forward,
the backward and a one-step run, every observable byte hashed: the same verdict table and digest
(`f18e42c9d62a47de`) from 4fcaa5e's sources (14b5941's C) and this slice's, the hidden key and the
hiding provider refused on every contract and pool. The differential of every byte the API can observe ([fuse/statedump.c](measurements/2026-10-16/fuse/statedump.c)) writes the same 376,368,768 bytes and the same 9.84M digest (`41018dd3e5a5e469`) from both trees -- the values it has written since 8b4f23d -- though its schedule never reaches an activation the scan refuses, which is why the verdict differential above exists. The 20-step state at one and four threads, widest and AVX2-capped builds and a repeat, is `4bf4e057…` ([scanw/identity.json](measurements/2026-10-17/scanw/identity.json)); the portable build's scans are the C harness's (its portable build runs `writer_verdicts` with the rest).

Measured in place at four threads, 4fcaa5e's sources and this slice's in turn, the order rotated,
eight rounds ([scanw/attribution-t4-\*](measurements/2026-10-17/scanw/)):

| ms a step, medians of eight rounds | 4fcaa5e | NDT-SCAN-W | rounds lower |
|---|---|---|---|
| the activations' scan | 5.95 | 0.05 (the logits) | 8 of 8 |
| the attention forward / SiLU / the RMSNorm forward | 6.36 / 2.39 / 1.47 | 6.54 / 2.51 / 1.64 | -- |
| everything but the GEMMs | 56.0 | 50.3 | 8 of 8 |
| the GEMMs (their code unchanged) | 153.8 | 151.5 | -- |
| the step (median of eight steps) | 206.6 | 204.5 | 6 of 8 |

Everything but the GEMMs is 5.7 ms a step (10%) lower in every round: the scan's 5.9 ms less the
writers' reads, 0.5 ms among the three kernels. The writers' reads cost what a pass over L1 and L2
costs ([scanw/scan_overhead.c](measurements/2026-10-17/scanw/scan_overhead.c), each entry against its
plain entry at the language model's shapes on four threads, three runs: the SwiGLU product +0.08 to
+0.31 ms a step, the RMSNorm +0.002 to +0.024, the attention within the noise) once the attention's
reads are one call a region:
a first form that read each of its strided 48-float rows through the 16-lane scan's own call spent
0.6 ms a step in the attention ([scanw/first-\*](measurements/2026-10-17/scanw/)). The GEMMs moved
135-164 ms between this session's rounds, so the step's own median says little -- the first form's A/B
had them 5 ms the other way.

## 26. BCIR against PyTorch at four threads on cf435d4

The 1,000-step language-model run at four threads on NDT-SCAN-W (cf435d4), BCIR under
`kblock+exp+reduce` against PyTorch 2.13 eager and `torch.compile` at four intra-op threads, one round
in §25's session -- BCIR, eager, compiled, BCIR -- every run from the committed tree
([lm/r8-cf435d4-\*](measurements/2026-10-17/lm/)):

| four threads, 1,000 steps | median step, ms | tokens a second | validation / test cross-entropy | peak RSS, MB |
|---|---|---|---|---|
| BCIR, cf435d4 (two runs) | 204.7 / 207.4 | 5,002 / 4,936 | 1.5003 / 1.6319 | 529 / 528 |
| PyTorch eager | 293.6 | 3,487 | 1.5003 / 1.6319 | 1,124 |
| PyTorch compiled | 267.0 | 3,835 | 1.5001 / 1.6321 | 1,116 |

Both BCIR runs end in `5137e013…`, the state of every BCIR run since §17. BCIR trains 42-43% more
tokens a second than eager PyTorch and 29-30% more than compiled, to the same held-out cross-entropy in
under half the memory; its 204.7 ms is the fastest 1,000-step run recorded and the first past 5,000
tokens a second. PyTorch moved too: against §24's session eager's median step is 4% shorter (304.5 ms
there) and compiled's 14% (310.5), so the slice's own share is §25's alternated in-place reading.

## 27. The AdamW update's commit and first pass (NDT-UPDATE)

After NDT-SCAN-W the largest pass outside the GEMMs and the attention was the AdamW update: 15.3 ms of
the four-thread step in §25's session (the commit 8.7, the first pass 6.6), two passes over the four
39 MB streams of the 9.84M decoder's gradient, weights and moments. §11 had read the update's passes
as memory-bound. A probe of the trainer's own functions at the decoder's size on a pool of four
([update/update_probe.c](measurements/2026-10-17/update/update_probe.c), the trainer's unit compiled in,
medians of 15 rounds) says neither was:

| ms a call on a pool of four | the commit | the first pass |
|---|---|---|
| before the slice | 8.40 | 6.84 (one chunk a task) |
| NDT-UPDATE | 6.15 | 3.53 (two chunks a task) |
| bounds, not exact | 5.46 with the two shared divisions as products, 4.79 with no division or root, 4.17 the streams alone | 3.72 the flag and maxima without the chain |
| measured, not kept | 8.47 with streaming stores | 8.02 eight chains from gathers |

The commit was bound by its divider: three double divisions and a root an element, against 4.17 ms for
the same streams with trivial arithmetic, and stores that bypass the cache made it slower. Two of the
three divisions are by the bias corrections 1 - β1^t and 1 - β2^t, which every element of a call shares.
The first pass was bound by its norm's chain: under the reduce contract each chunk's squares are added
in index order in double, 16,384 dependent additions a chunk at this size, and the chain ran after the
chunk's vector loop, so a task took the span's time and the chain's latency in turn.

NDT-UPDATE keeps every value. The commit's two shared divisions go through `div_shared`: y = RN(1/d)
once a call, q0 = x·y (within 1.5 ulp of x/d), a first FMA correction that leaves one of x/d's two
floating-point neighbours, and a second that is Markstein's theorem (Handbook of Floating-Point
Arithmetic, 2nd ed., Theorem 4.10): for a faithful q and a y within half an ulp of 1/d, r = x - q·d is
exact and RN(q + r·y) is RN(x/d). The quotient keeps x's sign, which the corrections' sums lose for a
zero x. The dividends are floats and the divisors lie in [2^-53, 1], so every operand stays far from
under- and overflow. The AVX-512 commit takes it, and so does a new AVX2+FMA commit, the commit's
variant bit 8, set where the fused GEMM's variants carry AVX2, which they do only with FMA. The portable
commit, the FMA-less AVX2 commit and the preflight keep the true divisions, and each element's own
division by √v̂ + ε and its root stay. In the first pass, under the reduce contract on a vector variant,
a task is two chunks. `pair_avx512` and `pair_avx2` take both chunks' flag and maxima a vector at a
time (pass_span's flag element for element, and the int32 maxima of the same bit patterns), and add
each chunk's squares to its own chain in index order between the vector loads. The two chains then run
under the four streams' reads, each under the other's latency. A pair whose second chunk is short, an
odd last chunk, the chain contract and the portable build take the per-chunk loop. Every chunk's
partial is its own chain's, so the norm is the contract's.

Witnesses. The C harness's `update_pairs` runs two decoders (vocabulary 16, width 32, four heads over
two KV heads, two layers):
- feed-forward 64 and untied: 19,616 parameters, twenty 1,024-element chunks, nine whole pairs and a
  pair whose second chunk is short;
- feed-forward 48 and tied: 16,032 parameters, sixteen chunks, the last pair short.

Every buffer is NaN past the parameters. Against `ref_update` (true divisions, each chunk's chain by
itself) the harness checks the state, the verdict and the norm bit for bit, under both norm contracts,
with no pool and a pool of three, at scales 1 and 1/3, with clipping on and off, and the square sum
through `bcir_decoder_gradient_norm` as well. It then plants a refusal at the first and last element of
each of the first sixteen chunks: a NaN gradient, both infinities, ±1e20 whose square overflows the
second moment with clipping off, a NaN weight, an infinite first moment, a negative second moment. All
512 are refused with the state untouched, and 64 -0.0 second moments are admitted as the reference
admits them. One last case puts a -0.0 gradient over a -0.0 first moment and weight, which the
reference makes +0.0: the first division keeps its zero's sign.

Under `BCIR_DT_DIVISION_PROBE`, defined by the harness's own builds and absent from the library, the
harness also holds the commit's shared division, dispatched as the commit dispatches it, to the
division over 268,444,464 dividends. There are sixteen divisors: the bias corrections of 0.9, 0.95 and
0.999 at steps 1, 2, 10 and 1,000, computed as the update computes them, and 1, 2^-53, the largest
double below 1 and an all-ones significand. The dividends are every float significand with both signs,
the zeros, the subnormals and the extremes. The harness's Python test now also builds C23 capped at
AVX2, so it runs the AVX2 pair and the AVX2+FMA commit.

The pre-push chain found one witness the slice had weakened. On a vector host the clipping-norm test's
threads bench (133 chunks) now reaches the per-chunk loop only in its short last chunk, and a reordered
sum there rounds away in the total. So `ndt-reduce.json` NR11 (that loop's chain summed from the
chunk's last element) was caught by the C harness but not by its own check. The test now also runs the
differential over one chunk of 728 parameters, whose norm is that loop's chain alone, and NR11 is caught
by its own check again (11 of 11).

`tools/testing/faults/ndt-update.json` UP1-UP14 injects fourteen defects and the harness catches all
fourteen: either pair's second chain adding the first chunk's squares, the chains adding each vector's
squares from its last, either flag forgetting the gradient, the gradient maximum keeping its sign bit,
a negative second moment admitted, a short pair taking the paired loop, either pair reading its second
chunk a vector early, the shared division without its corrections, losing a zero's sign or taking a
float reciprocal, and the first moment divided by the second's correction.

A record checks the trainer's own `div_shared` against the division
([update/division_check.c](measurements/2026-10-17/update/division_check.c)) over 24,245 divisors (14,837
distinct): the update's bias corrections as it computes them for ten betas over 2,000 steps, the edges
of [2^-53, 1] and 4,000 random divisors. Each one takes 16,782,765 dividends: every significand at one
exponent with both signs, a sweep of 277 exponents, the zeros and the extremes. A float dividend's
quotient scales by its power of two while nothing under- or overflows, so that covers every float. The
record found 0 mismatches. The form with one correction alone also missed none of its 2.0·10^11, but
no theorem covers it, so the trainer keeps the second.

The statedump differential writes the same 376,368,795 bytes and the same 9.84M digest
(`41018dd3e5a5e469`) from aaa2e6c's sources and this slice's. The 20-step state at one and four threads,
in the widest and AVX2-capped builds and a repeat, is `4bf4e057…`
([update/identity.json](measurements/2026-10-17/update/identity.json)).

Measured in place at four threads, aaa2e6c's sources (cf435d4's C) and this slice's in turn, the order
rotated, eight rounds ([update/attribution-t4-\*](measurements/2026-10-17/update/)):

| ms a step, medians of eight rounds | aaa2e6c | NDT-UPDATE | rounds lower |
|---|---|---|---|
| the update's commit | 8.73 | 6.84 | 8 of 8 |
| the update's first pass | 6.62 | 3.15 | 8 of 8 |
| the update | 15.33 | 10.06 | 8 of 8 |
| everything but the GEMMs | 50.78 | 46.22 | 8 of 8 |
| the GEMMs (their code unchanged) | 153.5 | 155.8 | -- |
| the step (median of eight steps) | 203.0 | 200.6 | 4 of 8 |

The update is 5.3 ms a step (34%) lower in every round, and everything but the GEMMs 4.6 ms (9%) lower.
The rows the slice leaves alone moved 0.7 ms the other way. The GEMMs moved 145-211 ms between rounds,
so the step's own median says little. The first pass is now at its streams' bound: 3.53 ms in pairs,
against 3.72 for the flag and maxima with no chain at all. The commit's 6.15 ms is 2.0 ms above the
streams' 4.17. 1.4 ms of that is each element's own division and root, which no other element shares,
and 0.7 ms is the corrections' FMAs over plain products.

## 28. BCIR against PyTorch at four threads on 2afa3e3

The 1,000-step language-model run at four threads on NDT-UPDATE (2afa3e3): BCIR under
`kblock+exp+reduce` against PyTorch 2.13 eager and `torch.compile` at four intra-op threads. It is one
round in §27's session (BCIR, eager, compiled, BCIR), every run from the committed tree
([lm/r9-2afa3e3-\*](measurements/2026-10-17/lm/)):

| four threads, 1,000 steps | median step, ms | tokens a second | validation / test cross-entropy | peak RSS, MB |
|---|---|---|---|---|
| BCIR, 2afa3e3 (two runs) | 203.7 / 203.4 | 5,026 / 5,035 | 1.5003 / 1.6319 | 529 / 530 |
| PyTorch eager | 321.4 | 3,187 | 1.5003 / 1.6319 | 1,123 |
| PyTorch compiled | 278.2 | 3,681 | 1.5001 / 1.6321 | 1,088 |

Both BCIR runs end in `5137e013…`, the state of every BCIR run since §17. BCIR trains 58% more tokens a
second than eager PyTorch and 37% more than compiled, to the same held-out cross-entropy in under half
the memory. Its 203.4 and 203.7 ms are the fastest 1,000-step runs recorded. Most of the wider gap is
PyTorch's: against §26's session, eager's median step is 9% longer (293.6 ms there) and compiled's 4%
(267.0), while BCIR's is 0.5-2% shorter (204.7 / 207.4). The container restarted between the two
sessions, so its instance may differ. The slice's own share is §27's alternated in-place reading.

## 29. The packed tiles kept between a thread's tasks, measured (NDT-PACK)

An external review of the trainer -- the 2026-10-13 training-architecture blueprint, written against
#818's head b47f860 -- asks second in its program to reuse packed panels where it is legal, from a
workspace the caller lends, keyed by the source value's version rather than its address. In the
pooled blocked GEMM the reuse is a thread's: a task packs its column block's tiles of B and runs its
128 rows, and a thread whose next task is the same column block's next row set could keep them. Three arms
([pack/variants.py](measurements/2026-10-17/pack/variants.py), exact-match edits to d68e3d3's
`bcir_tensor.c`) keep the partition -- the same tiles of C, each one task's, its K blocks in order --
so every product is the same floats, and only which tasks run together and what a thread keeps
change: `colmajor` numbers a job's tasks column block by column block (the kept order is row-major),
`cache` adds a thread-local copy of one column block's tiles, every K block up to K = 1,024, keyed by
a per-job sequence number -- a weight updated between steps is never served from an earlier pack --
and `grp` numbers groups of four column blocks row set by row set, so the tasks that start together
read distinct columns and a thread's next task is most often its last one's.

| GEMM ms a step at four threads | base | colmajor | cache | grp |
|---|---|---|---|---|
| shapes alone, warm, run 1 (12 rounds) | 151.55 | 141.34 (10 of 12) | 138.66 (11 of 12) | -- |
| shapes alone, warm, run 2 (12 rounds) | 134.59 | 131.56 (6 of 12) | -- | 132.20 (5 of 12) |
| shapes alone, cold, against base | -- | 168.3 -> 167.3 | 168.9 -> 165.1 | 174.2 -> 180.1 |
| in place, medians of 8 rounds | 143.7 | 142.2 (4 of 8) | 141.5 (6 of 8) | -- |

The parentheses count the rounds an arm was below base; every C hashes as base's
([pack/](measurements/2026-10-17/pack/): [gemm_ab.py](measurements/2026-10-17/pack/gemm_ab.py) over
[beta/gemm_entries.c](measurements/2026-10-16/beta/gemm_entries.c), the cold pairs through
[gemm4t/gemm_cold.py](measurements/2026-10-16/gemm4t/gemm_cold.py) on six operand sets that no core's L2
holds, the in-place rounds through the attribution instrument, rotated). These ran on the instance the
container moved to at 10:15: a 2.10 GHz Xeon with 2 MB of L2 a core and a 260 MB L3 (and AVX512-BF16 and
AMX), not §27's 2.80 GHz one with 1 MB and 33 MB -- so the cold pairs' operands, up to 72 MB, are cold in
L2 and not in the L3. The first warm run's 6.7% did not repeat: base itself was 17 ms faster in the
second, and colmajor 2.3% ahead of it in half the rounds. Cold in L2, colmajor gains 0.6% and the cache
2.2%, and grouping the columns costs 3.4%; in place both kept arms are inside base's own rounds
(132.9-165.4 ms). Whichever order, a column panel of B is packed from outside the packer's L2 the first
time a core needs it in a job; the order changes only how often a later pack finds it in the packer's
own L2, and the cache saves only the pack's copy instructions, which the inner loop did not wait on in
§18's probe. Not kept: a workspace lent to the pool and a job key in every GEMM call for about 2%,
inside every reading's spread. Nor does the row-set height a task covers, which the larger L2 might have
moved: in place, 128, 256 and 512 rows a task give 143.0, 147.0 and 141.9 ms of GEMMs a step (the larger
two below 128 in 4 and 5 rounds of 8; 128's own rounds 136.2-183.1).

## 30. BCIR against PyTorch, five seeds paired (LM-PAIRED)

The same external review (§29) names the comparison it would accept as a public claim: against the
fastest valid matched PyTorch configuration, with a paired confidence interval, at least three seeds
and held-out quality inside a margin fixed beforehand. The comparison harness gained `prepare --seed`
(aeb7d15: another replicate's initial weights and batch schedules, the corpus, its splits and the
held-out problems unchanged; the default seed's files are the 10-11 record's byte for byte) and a
memory split in its train records. Five seeds -- the record's 20261011 and 1001, 2002, 3003, 4004 --
each ran the 1,000-step language-model run at four threads on BCIR (`kblock+exp+reduce`) and on
`torch.compile` with fused AdamW (PyTorch 2.13, four intra-op threads), one pair a seed in one
session, the pair's order alternating seed by seed ([paired/](measurements/2026-10-17/paired/),
[run.sh](measurements/2026-10-17/paired/run.sh)). The analysis was fixed before the runs
([paired.py](measurements/2026-10-17/paired/paired.py)): the ratio of the two median warm steps per
seed, their geometric mean with a 95% Student-t interval over the seeds' log ratios, and the margin --
the 95% interval of BCIR's test cross-entropy minus PyTorch's inside ±0.01 nats.

| seed | median step, BCIR / PyTorch, ms | tokens a second | ratio | test cross-entropy, BCIR / PyTorch | peak RSS, MB |
|---|---|---|---|---|---|
| 20261011 | 185.2 / 237.4 | 5,529 / 4,313 | 1.282 | 1.6319 / 1.6320 | 528 / 1,055 |
| 1001 | 177.1 / 226.1 | 5,781 / 4,529 | 1.277 | 1.6350 / 1.6350 | 528 / 1,119 |
| 2002 | 177.1 / 222.9 | 5,781 / 4,595 | 1.258 | 1.6401 / 1.6401 | 529 / 1,121 |
| 3003 | 176.5 / 218.4 | 5,801 / 4,689 | 1.237 | 1.6429 / 1.6429 | 529 / 1,056 |
| 4004 | 177.8 / 223.5 | 5,759 / 4,581 | 1.257 | 1.6324 / 1.6324 | 529 / 1,061 |

BCIR trains 1.262 times PyTorch's tokens a second (95% interval 1.240-1.284) and is faster on all five
seeds: its median step is 20.8% below PyTorch's, and at the interval's bound 19.4%, so the review's bar
-- at least 5% below the fastest matched configuration measured, the paired interval supporting it -- is
met on this instance. The held-out model is the same: BCIR's test cross-entropy minus PyTorch's is
-0.00001 nats on average (95% interval -0.00006 to +0.00003, inside the margin), validation +0.00011.
Both sides hold the same 157,390,848 bytes of persistent state (weights, gradients and both moments);
BCIR's planned transient arena is 194,019,328 bytes and its peak RSS 528-529 MB, against PyTorch's
1,055-1,121 MB, whose transient activations have no peak counter on CPU. The default seed's BCIR run
ends in `5137e013…`, the state of every BCIR 1,000-step run since §17; `torch.compile`'s compile and
first two steps (5.7-31.6 s) are outside its median. Every run is FP32.

The session ran on the instance of §29, where both sides are faster than in §28's: BCIR's median step
9-13% shorter and PyTorch's 15-22%, so the margin is narrower than §28's single round (1.37), and this
table, not that round, is the comparison of record. Compiled was not always PyTorch's faster arm (§24's
session had eager 2% ahead), so a check was added after the runs, and is labelled so in `paired.py`:
eager PyTorch on seeds 20261011 and 1001 in the same session, 235.5 and 254.6 ms a step -- 0.8% faster
and 12.6% slower than that seed's compiled run, whose 20261011 arm was the session's first PyTorch run.
Against the faster arm wherever both ran, BCIR's ratio is 1.260 (95% interval 1.241-1.279).

## 31. The GEMMs on bfloat16 operands (NDT-BF16)

The instance the container ran on from 10:15 (family 6 model 207) has the AMX tile unit and
AVX512-BF16, which the 2026-10-12 host had not; the roadmap's NDT-BF16 row asked for a BF16 slice with
an explicit accumulation policy, recorded against its host gate. BCIR's contracts are semantics first
and instances second, so the slice began by pinning what the tile unit computes, bit for bit
([bf16/](measurements/2026-10-17/bf16/), [semantics.c](measurements/2026-10-17/bf16/semantics.c)):
`TDPBF16PS` against twelve candidate models on four operand distributions (51,200 outputs each) and at
every depth of one to sixteen pair rows (13,056 outputs each). One model matches every output and no
other matches all of any distribution: per instruction and output, two lanes from +0.0 -- the even
terms' fused products accumulated in one, the odd terms' in the other, each step rounded once and
flushed when tiny -- then `c = daz(c) + (E + O)`. MXCSR is ignored (0 of 13,056 outputs move under
round-toward-zero with FTZ and DAZ), DAZ applies to both operands and to C, and "tiny" is IEEE
tininess after rounding with the exponent unbounded ([tininess.c](measurements/2026-10-17/bf16/tininess.c)):
a lane value in [2^-126 - 2^-150, 2^-126 - 2^-151) is flushed although it rounds to 2^-126 on the
subnormal grid -- on 1,024,000 lanes built at that boundary the model and a replay as one `vfmadd` per
lane step under MXCSR's FTZ and DAZ equal the tile unit on all, and flushing after rounding to the
grid is wrong on 511,998. `VDPBF16PS` is another arithmetic (the odd term first, one chain from the
accumulator: 51,200 of 51,200 on all four distributions, the even-first order 45,175-49,322), so it is
no instance of the tile unit's contract. `VCVTNEPS2BF16` rounds to nearest-even after DAZ (0 of 2^32
inputs differ from that rule; 16,711,678 subnormals do without the DAZ), a NaN to a quiet NaN, MXCSR
ignored; `VCVTNE2PS2BF16`, which packs two vectors at once, is the same function on every input
([cvt2.c](measurements/2026-10-17/bf16/cvt2.c)).

That model is the contract (`bcir_tensor.h`): `a = bf16(RN(alpha·A))`, `b = bf16(B)`, the depth in
32-term blocks padded with +0.0 terms, `c0 = beta == 0 ? +0.0 : RN(beta·C)`, and per block and output
the two lanes and the flushed sums above; which NaN a NaN result is lies outside it. Three instances
compute it: portable C (the lane in double, the tininess rule written out), AVX-512 (the lanes as
vector FMAs under MXCSR's FTZ and DAZ, set around its kernel only) and AMX (2 x 2 tiles of 16 x 16
outputs), the last offered only where the CPU has AMX-TILE, AMX-BF16 and AVX512-BW/BF16, XCR0 enables
the tile state, Linux grants the process the tile permission, and a differential against the portable
instance on the contract's conversion and arithmetic edges agrees bit for bit -- asked once a process,
kept in one atomic word. The operands are packed first into workspace the caller lends (A as 16 x 32
tiles, B as 16-pair x 16-column tiles of its interleaved depth pairs, one layout for every instance),
then the pool computes regions of 32 rows by a column chunk whose B panel is at most 1 MB; a shorter
workspace runs the call in row and column chunks to the same bits, and below one 32-row strip of A and
one 32-column strip of B the entry writes nothing. A call with no rows or no columns writes nothing too,
which clang's analyzer (18 and 23, the C analysis gates) found was not so before the push: the grouped
entries give every call they take its partition, and such a call divided its region count by zero
there. The region width is counted in whole 32-column blocks now -- the same width as before on all
126,000,000 non-empty shapes compared ([cols_check.c](measurements/2026-10-17/bf16/cols_check.c)), so no
call that computes anything is partitioned differently -- and the C harness holds every entry to the
empty shapes (`tools/testing/faults/ndt-bf16.json` NB16 and NB17 put the two divisions back). An independent reference of the contract
([check.c](measurements/2026-10-17/bf16/check.c): the binary32 operands read directly, the lanes in
`__float128`, tininess by an explicit 24-bit rounding) finds 0 differences in 963 cases (every
instance, entry, chunking, group and crafted edge) under GCC 13, Clang 18 and Clang 23, 718 with the
unit capped below AMX and 473 in the portable build ([check.txt](measurements/2026-10-17/bf16/check.txt));
the oracle's emulation in exact rationals is the GEM+ row `train.mm.bf16.mismatch` (0 of 28).

In the trainer it is `BCIR_DT_CONTRACT_BF16` (state ABI 4: the state lends `workspace` and
`workspace_bytes`, at least `bcir_decoder_workspace_bytes` for its plan -- the step's largest GEMM
group, 11,534,592 bytes for the 9.84M decoder); one GEMM contract a state (with neither `fused` nor
`kblock`), and in Python `bf16`, `bf16+exp`, `bf16+reduce` and `bf16+exp+reduce`, the workspace
counted in the memory budget and no CBLAS provider admitted (an `sgemm` computes in binary32). Only
the GEMMs change; the attention, the norms, the activations, the gradients, the moments and the
update stay binary32, so the state is the bf16 contract's own.

| at four threads | FP32 `kblock` | bf16 | ratio |
|---|---|---|---|
| the step's 13 GEMM shapes alone, sum of medians | 16.536 ms (320-453 GFLOP/s) | 4.359 ms (943-1,839 GFLOP/s) | 3.79 |
| the same at one thread | 50.324 ms (107-140) | 10.429 ms (398-780) | 4.83 |
| the 9.84M decoder's step, median of 24 | 188.1 ms (5,445 tokens/s) | 79.8 ms (12,832 tokens/s) | 2.36 |
| the same at one thread, median of 8 | 623.7 ms (1,642) | 256.2 ms (3,997) | 2.43 |

([gemm-t4.txt](measurements/2026-10-17/bf16/gemm-t4.txt), [gemm-t1.txt](measurements/2026-10-17/bf16/gemm-t1.txt),
[step-t4.txt](measurements/2026-10-17/bf16/step-t4.txt), [step-t1.txt](measurements/2026-10-17/bf16/step-t1.txt):
the contracts alternated round by round, each step timed alone, the same seeded weights and batch.)
The tile instance is what is fast. The AVX-512 instance alone, at one thread, is 0.56 times the FP32
kernel ([gemm-v4.txt](measurements/2026-10-17/bf16/gemm-v4.txt): 50-73 GFLOP/s against 92-143) --
each pair of terms costs it a broadcast, a shift and a mask beside its two FMAs -- so on a host without
AMX the bf16 contract buys no speed, and its use there is a state identical to an AMX host's. The
state is the contract's everywhere: six steps of the comparison decoder end in one digest,
`e91ae701…`, through the tile instance, the AVX-512 instance and the portable one (the unit capped at
AVX-512, at AVX2 and built portable), at one and at four threads ([identity.json](measurements/2026-10-17/bf16/identity.json)).
(The portable instance is the reference, not a speed path: 80 s a step at one thread.) Timed in
place ([attribution-t4-{bf16,kblock}-{1,2,3}.json](measurements/2026-10-17/bf16/), §11's instrument,
three rounds each, alternated), the four-thread step's GEMMs go from 150.8 to 41.8 ms; the attention
(15.4 ms against 14.6), the update (9.5 against 9.3) and the rest are FP32's, so the GEMMs are now half
the step and the attention a fifth -- its products, which no contract has yet fused, are the next
slice (ATTN-FMA). Whether the model trained under the contract is as good as FP32's, and how it stands
against PyTorch's own bfloat16, is §32.

## 32. BCIR's bfloat16 against PyTorch's (LM-BF16)

§31's contract changes only the GEMMs, so the question it leaves is the model: whether a decoder
trained under it is as good as FP32's, and how BCIR's bf16 stands against PyTorch's own. The
comparison harness gained `torch-eager-bf16` and `torch-compiled-bf16`: PyTorch's CPU mixed
precision, `torch.autocast("cpu", torch.bfloat16)` around the forward and the loss, the weights,
gradients, moments and fused AdamW in FP32. What each side puts in bfloat16 differs, and is stated
rather than equalised: BCIR's contract rounds only the GEMMs' operands, and the attention, the norms,
the activations, the gradients, the moments and the update stay binary32; autocast runs the linears
and the attention (SDPA) in bfloat16 and keeps the residual stream, the norms, the loss and the
optimizer in FP32 ([bf16/lm/](measurements/2026-10-17/bf16/lm/)). LM-PAIRED's five seeds each ran
the 1,000-step language model at four threads under four arms in one session, the arms' order rotated
seed by seed -- BCIR `bf16+exp+reduce`, compiled PyTorch under autocast, BCIR `kblock+exp+reduce`,
compiled PyTorch in FP32 -- with the analysis fixed before the runs ([paired_bf16.py](measurements/2026-10-17/bf16/lm/paired_bf16.py)):
per pair of arms the ratio of the median warm steps per seed and a 95% Student-t interval on the log
ratios, and a ±0.01-nat margin on the test cross-entropy difference for the claim and for each
framework's bf16 against its FP32.

| seed | BCIR bf16, ms (tokens/s) | PyTorch bf16 | BCIR FP32 | PyTorch FP32 | test cross-entropy, BCIR bf16 / PyTorch bf16 / FP32 (both) |
|---|---|---|---|---|---|
| 20261011 | 71.4 (14,339) | 152.3 (6,722) | 167.6 (6,110) | 208.3 (4,915) | 1.6369 / 1.6314 / 1.6319 |
| 1001 | 70.0 (14,626) | 155.0 (6,607) | 172.7 (5,930) | 218.9 (4,678) | 1.6348 / 1.6348 / 1.6350 |
| 2002 | 69.8 (14,666) | 141.5 (7,235) | 169.2 (6,052) | 202.3 (5,061) | 1.6404 / 1.6405 / 1.6401 |
| 3003 | 75.0 (13,660) | 152.5 (6,715) | 172.4 (5,940) | 216.2 (4,737) | 1.6435 / 1.6451 / 1.6429 |
| 4004 | 76.2 (13,445) | 153.3 (6,681) | 170.6 (6,001) | 206.6 (4,955) | 1.6326 / 1.6318 / 1.6324 |

([summary.json](measurements/2026-10-17/bf16/lm/summary.json); the two FP32 arms' test cross-entropies
agree to the fourth decimal on every seed, as in LM-PAIRED.) BCIR under the bf16 contract trains at
**2.083 times** the tokens a second of compiled PyTorch under autocast (95% interval 1.979-2.192),
faster on five of five seeds, and its test cross-entropy is 0.0009 nats above PyTorch's (interval
-0.0025 to +0.0043): inside the margin. Against its own FP32 run the contract buys 2.354 times
(2.243-2.470) at +0.0012 nats (-0.0015 to +0.0039), inside the margin too, so the quality parity the
roadmap's NDT-BF16 row asked for holds over these seeds; autocast buys PyTorch 1.395 times
(1.352-1.439) at +0.0003. Against compiled PyTorch in FP32 BCIR's bf16 runs 2.905 times (2.729-3.092),
and BCIR's FP32 against it 1.234 times (1.197-1.272), LM-PAIRED's comparison again in this session.
The check fixed beside the analysis, eager PyTorch under autocast on two seeds, ran slower than
compiled (183.3 and 182.3 ms against 155.0 and 152.3), so the claim's PyTorch arm is its faster bf16
arm. Peak RSS is 540 MB for BCIR's bf16 runs against 1,141-1,153 MB for PyTorch's; the persistent
state is the same 157,390,848 bytes on every arm, and BCIR's transient arena under bf16 is
205,553,920 bytes (FP32's 194,019,328 and the 11,534,592-byte packing workspace). BCIR FP32's
seed-20261011 run ends in `5137e013…`, the state of every 1,000-step BCIR run since §17. The bf16
step's medians here (69.8-76.2 ms) are those of 1,000 consecutive steps under one contract; §31's
79.8 ms is the median of 24 steps alternated with the FP32 contract's in one process. Of that step
the attention is a fifth, ATTN-FMA's slice.

## 33. The attention's products fused (ATTN-FMA)

§31 left the attention a fifth of the bf16 step and its products v1's arithmetic: the scores q_i . k_j,
the context's sums of p_ij v_j, and the backward's probability adjoints dy_i . v_j, its dot sum of
p_ij dp_ij and the dq, dk and dv sums each took a rounded product and a rounded sum a term -- in the
AVX-512 register blocks a multiply and an add, two instructions for the FMA units where one does both.
The fused attention contract (`bcir_tensor.h`'s `BCIR_TENSOR_ATTENTION_FUSED`, through
`bcir_tensor_attention_contract_scan_pool` and `bcir_tensor_attention_backward_contract_pool`, whose
`BCIR_TENSOR_ATTENTION_EXP` is the v2 softmax's exponential) makes every one of those multiply-adds one
IEEE fusedMultiplyAdd (C99 `fmaf`: the product unrounded, one rounding) over the same terms in the same
order from the same start; the scaling, the softmax and the score adjoint p_ij (dp_ij - dot_i) times
the scale stay v1's, and without the bit each entry is its v1 entry bit for bit. Three instances
compute it, chosen as the fused GEMM's are: portable `fmaf` (the GEMM kernel's fused rows, under the
causal masks), AVX2+FMA (the fused GEMM's AVX2 kernel for every row) and AVX-512 -- the v1 register
blocks written once as always-inline bodies and instantiated twice, the multiply-add a constant of each
copy (`_mm512_fmadd_ps`, and for a block's diagonal terms `_mm512_mask3_fmadd_ps`, which leaves a lane
its mask excludes exactly as it was, NaN or not), the rows past the last whole eight and the scalar
tails through the fused kernel and `fmaf`. In the trainer it is `BCIR_DT_CONTRACT_ATTN` (bit 32 of the
state's contract, with any GEMM contract or none: no ABI change), and in Python `attn` and
`<contract>+attn`. FP32 throughout: only the products' roundings change.

The C harness holds the entries to `fmaf` reference loops written beside the v1 ones: 192 cases (the
attention reference's sixteen shapes, finite and with an infinite value and a NaN adjoint, under libm's
exponential and BCIR's, serially and on pools of two and four), every probability, context element and
adjoint bit-identical to the loops, 180 of them apart from v1's; the runtime gate's builds of the same
harness, portable and capped at AVX2, hold those instances to the same loops. Under the bit the
trainer's state is its own and one at every thread count (`attn`, `kblock+exp+attn` and
`bf16+exp+reduce+attn` in the thread-state test), and a probe in the tensor unit (the harness's builds
only) records the contract every attention job of a run computed with. That probe is the RED sweep's:
its first version recorded what the trainer meant to pass, a statement beside each call, and a wrong
contract at the backward's call (AF14) passed it, so the record moved into the jobs themselves.
`tools/testing/faults/ndt-attnfma.json` holds 17 injected faults (each block instantiating v1's
arithmetic, the remainders and tails taking v1's terms, the entries ignoring the bit, the trainer
refusing it, ignoring it or passing the wrong contract at either call, the Python names leaving it out),
and eight entries of five older tables anchored in the restructured blocks fire again re-anchored.

The attention alone at the language model's shapes (batch 8, eight query heads over four KV heads, 48
channels, BCIR's exponential), the two contracts alternated round by round in one process, as the
trainer calls them -- back to back on a busy pool, the operands on huge pages
([attnfma/](measurements/2026-10-18/attnfma/), [attention.c](measurements/2026-10-18/attnfma/attention.c)):

| one layer, median | v1 products | fused | ratio |
|---|---|---|---|
| forward, 128 tokens, four threads | 0.776 ms | 0.604 ms | 1.29 |
| backward | 1.104 | 0.763 | 1.45 |
| forward, 128 tokens, one thread | 2.797 | 2.148 | 1.30 |
| backward | 3.792 | 2.492 | 1.52 |
| forward, 512 tokens, four threads | 12.057 | 9.857 | 1.22 |
| backward | 23.395 | 17.093 | 1.37 |

The backward gains most: its products are most of its work, where the forward's softmax is unchanged.
(A first version of the instrument hashed each call's outputs on the calling thread between the
forward and the backward; the pool's threads slept through it and the backward measured 3.6 ms a layer
at four threads, against the 1.1-1.4 ms it costs in place, so the records are the second version's.)
In the step, in place at four threads (§11's instrument, five rounds each, alternated), the attention
forward goes from 6.18 to 5.18 ms a step and the backward from 8.35 to 6.09, every fused round's
attention below every v1 round's, the GEMMs the same 41.2 / 41.3 ms. The step alone, alternated round
by round ([step.py](measurements/2026-10-18/attnfma/step.py)):

| the 9.84M decoder's step, median of 24 | without the bit | with it | ratio |
|---|---|---|---|
| `bf16+exp+reduce`, four threads | 80.0 ms (12,805 tokens/s) | 77.6 ms (13,199) | 1.031 |
| `kblock+exp+reduce`, four threads | 176.8 ms (5,792) | 169.4 ms (6,046) | 1.044 |
| `bf16+exp+reduce`, one thread, median of 8 | 246.2 ms (4,160) | 235.8 ms (4,343) | 1.044 |

The state is the contract's on every instance: six steps of the comparison decoder end in one digest
per contract through the tile, AVX-512, AVX2 and portable builds at one and four threads --
`08da09e8…` under `bf16+exp+reduce+attn`, `562b9bd8…` under `kblock+exp+reduce+attn`
([identity-bf16.json](measurements/2026-10-18/attnfma/identity-bf16.json),
[identity-kblock.json](measurements/2026-10-18/attnfma/identity-kblock.json)) -- and the contracts
without the bit are unchanged on this commit's sources: `kblock+exp+reduce`'s 20-step state is still
`4bf4e057…`, `bf16+exp+reduce`'s six-step state still `e91ae701…`. The attention is now about 15% of
the bf16 step in place (11.3 of 73.5 ms; 14.5 of 80.6, 18%, without the bit); its products no longer
dominate it, so the
next attention slice is its schedule (the tiled online-softmax attention of the roadmap's NDT-ATTN row),
not its arithmetic.

## 34. The attention fused, against PyTorch's bfloat16 (LM-ATTN)

§33's bit changes the attention's roundings, so it leaves the question §32 answered for the bf16
contract: whether the model trained under it is as good, and what it buys over a whole run. LM-PAIRED's
five seeds each ran the 1,000-step language model at four threads under five arms in one session, their
order rotated seed by seed -- BCIR `bf16+exp+reduce+attn` and `bf16+exp+reduce`, BCIR
`kblock+exp+reduce+attn` and `kblock+exp+reduce`, and compiled PyTorch under autocast (§32's faster
PyTorch bf16 arm) -- with §32's analysis fixed before the runs, the margin graded on every pair
([attnfma/lm/](measurements/2026-10-18/attnfma/lm/), [paired_attn.py](measurements/2026-10-18/attnfma/lm/paired_attn.py)).

| seed | BCIR bf16+attn, ms (tokens/s) | PyTorch bf16 | BCIR bf16 | BCIR FP32+attn | BCIR FP32 |
|---|---|---|---|---|---|
| 20261011 | 77.8 (13,157) | 155.3 (6,592) | 82.0 (12,490) | 177.6 (5,766) | 175.7 (5,830) |
| 1001 | 68.5 (14,951) | 131.2 (7,802) | 71.8 (14,258) | 165.3 (6,194) | 170.3 (6,012) |
| 2002 | 70.8 (14,460) | 147.3 (6,952) | 72.7 (14,078) | 165.7 (6,181) | 170.6 (6,004) |
| 3003 | 68.3 (14,999) | 142.9 (7,164) | 72.0 (14,214) | 163.6 (6,259) | 169.2 (6,052) |
| 4004 | 69.7 (14,698) | 138.9 (7,374) | 68.2 (15,009) | 162.2 (6,314) | 169.7 (6,035) |

([summary.json](measurements/2026-10-18/attnfma/lm/summary.json).) BCIR's fastest contract trains at
**2.015 times** the tokens a second of compiled PyTorch under autocast (95% interval 1.926-2.107),
faster on five of five seeds, its test cross-entropy 0.00016 nats below PyTorch's on average (interval
-0.00161 to +0.00129): inside the margin; against BCIR's own FP32 run it is 2.412 times (2.297-2.532) at
+0.00081 nats (-0.00141 to +0.00303), inside it too. The attention bit buys 1.032 times under bf16
(0.993-1.073, faster on four of five) and 1.026 times in FP32 (0.999-1.053, four of five) -- §33's 3% of
the step, which five seeds of 1,000 steps on a shared host do not resolve at 95% -- at -0.00036 nats
(-0.00104 to +0.00032) under bf16 and +0.00000 in FP32, where the fused products move the final test loss
by less than 10^-5 nats. Without the bit BCIR's bf16 runs 1.952 times PyTorch's in this session
(1.842-2.067; 2.083 in §32's), PyTorch at 131-155 ms a step here against 141-155 there and BCIR at 68-82
against 70-76; and both of BCIR's arms without the bit end in §32's states on all five seeds, so the bit
is the one arithmetic this session changed. Peak RSS is 540 MB for BCIR's bf16 arms and 529-530 for its
FP32 arms against 1,132-1,148 MB for PyTorch's; the persistent state is the same 157,390,848 bytes on
every arm.

## 35. The bf16 GEMMs' packs and kernel (BF16-ATTR, BF16-KERN, BF16-PANELS-1)

After §34 the GEMMs are the larger half of the bf16 step again, so BF16-ATTR took the bf16 entry
apart at the 9.84M decoder's 13 shapes ([phases.c](measurements/2026-10-18/panels/phases.c)): each call's A pack, B pack and
compute as pool jobs of their own, alternated with the whole entry, then the tile unit's rate on
resident tiles. Before this commit, at four threads, the packs take 1.10 ms of the 13 shapes (A 0.60,
B 0.50) beside 3.35 ms of compute, which runs at 1,909 GF/s against the run's own peak probe of 7,644
summed over the four threads -- a quarter; at one thread 2.97 ms of packs beside 7.82 of compute, 818
GF/s against a peak of 1,898 ([phases-base-t4.txt](measurements/2026-10-18/panels/phases-base-t4.txt), [-t1](measurements/2026-10-18/panels/phases-base-t1.txt)).
Against oneDNN's bf16 matmul through PyTorch on the same shapes, BCIR's entry already led: oneDNN takes
7.38 ms at four threads from binary32 operands (the conversions per call, which BCIR's entry includes)
and 5.22 from operands already bfloat16, against BCIR's 3.58 ms with this commit
([onednn-t4.txt](measurements/2026-10-18/panels/onednn-t4.txt), [phases-t4.txt](measurements/2026-10-18/panels/phases-t4.txt)); at one thread 13.55 / 7.89
against 8.99. The tile unit's rate on this shared host moved during the session -- the peak probe
read 2,005 GF/s on one thread in the morning and 960-2,093 in these records, each record carrying its
own -- so only measurements within one process compare trees here.

The kernel was the first lever. [kern.c](measurements/2026-10-18/panels/kern.c) times candidates for the AMX block on the base's
packs against the base's kernel, rotated in one process, each held to it bit for bit or the run fails
([kern-t4.txt](measurements/2026-10-18/panels/kern-t4.txt), [kern-t1.txt](measurements/2026-10-18/panels/kern-t1.txt)):

| candidate (the 13 shapes' compute, ms) | four threads | one thread |
|---|---|---|
| the kernel before this commit | 4.067 | 8.090 |
| the next block's C prefetched for writing (`PREFETCHW`) | 3.858 (x1.054) | 7.919 (x1.022) |
| that, and the B tiles loaded as streamed (`TILELOADDT1`) -- **taken** | **3.663 (x1.110)** | **7.236 (x1.118)** |
| the prefetch, and the A tiles streamed instead | 3.769 (x1.079) | 8.027 (x1.008) |
| streamed B and the prefetch, the loads software-pipelined | 3.856 (x1.055) | 7.941 (x1.019) |
| streamed B, the depth in chunks of 16 tiles kept over the columns | 3.743 (x1.087) | 7.119 (x1.136) |
| columns outer over 256-row regions, the B block kept | 5.404 (x0.753) | 8.241 (x0.982) |

A region's blocks share one A strip and read each B block once, so B streamed leaves L1 to the strip,
and the stores that end a block find their lines owned. The chunked depth leads at one thread by 1.6%
and trails at four by 2.2%; the kernel takes streamed B and the prefetch (BF16-KERN). Every tile
operation is the one before, in the same order: where a tile loads from and what is prefetched are
not part of the arithmetic.

The packs were the second. Under the bf16 contract the SwiGLU block's three tensors -- the hidden
activation and the gate's and the up projection's adjoints -- are read by bf16 GEMMs alone, so their
producers now write them as those GEMMs' panels: each element the conversion of its binary32 value,
which is all a pack computes for it, stored where the pack puts it for a whole call. The hidden
activation is written as A (the down projection's input) and as B (its weight adjoint's), each adjoint
as A (its input adjoint's) and as A^T (its weight adjoint's), the two panels filling the binary32 slot
exactly. A writer computes a row segment's values by the binary32 kernels' own functions and stores
their conversions as the A panel; a 32-row block done, it derives the second panel from the first --
pairs of rows interleaved as B's tile rows, or 16 x 16 words transposed as A^T's -- so the values are
the binary32 entries' bit for bit and the conversion is the pack's. The grouped entry
(`bcir_tensor_mm_bf16_group_panels_pool`) reads them in place of its packs; a call with a panel runs
whole in the one job, its rows, columns and depth multiples of 32 and, where A is a panel, alpha 1.
The trainer takes the panels under the bf16 contract with no provider when every GEMM they feed is
whole 32-tiles -- the rows, the model's width and the hidden width multiples of 32 -- and the slots are
64-byte aligned; otherwise binary32 and the packs, as before (BF16-PANELS-1). One layer's block at the
decoder's shapes, the two arms alternated, the outputs identical, goes from 4.641 to 4.415 ms at four
threads (x1.051) and from 15.42 to 12.57 at one (x1.227) ([block-t4.txt](measurements/2026-10-18/panels/block-t4.txt),
[block-t1.txt](measurements/2026-10-18/panels/block-t1.txt)).

The step was measured with the three trees in one process -- the base (d5bc457), this commit built
without its panels (`BCIR_DT_NO_PANELS`) and this commit -- one step each in turn, the order rotated
every round, so the arms run a step apart and the host's drift reaches them alike; the three compute
one state, and every round's three losses are equal ([steppair.py](measurements/2026-10-18/panels/steppair.py)):

| the 9.84M decoder's step, `bf16+exp+reduce+attn` | four threads, 150 rounds | one thread, 40 rounds |
|---|---|---|
| the base | 99.57 ms | 329.9 ms |
| the new kernel | 96.36 (x1.033, 95% 1.022-1.053; faster in 100 rounds) | 320.4 (x1.035, 1.012-1.060; 29) |
| the kernel and the panels | 92.73 (x1.025 over the kernel, 1.013-1.042; 96) | 300.3 (x1.058, 1.014-1.130; 29) |
| the two together | x1.063 (1.049-1.076; 116) | x1.085 (1.032-1.117; 32) |

([steppair-t4.txt](measurements/2026-10-18/panels/steppair-t4.txt), [steppair-t1.txt](measurements/2026-10-18/panels/steppair-t1.txt).) In place, statement by
statement over ten rotated rounds ([ab/](measurements/2026-10-18/panels/ab/)), the kernel takes the GEMMs from 45.9 to 41.0 ms of the
four-thread step (the forward x1.17) and the panels the backward's GEMMs x1.088 further, while the
writers take 3.3 ms more than the binary32 ones (the SiLU product 2.81 -> 3.99 ms, the VJP 3.41 -> 5.51):
deriving the second panel, a third of the VJP writer's time, is the writers' next lever. Those runs,
each arm 17 s from the next, carry the host's drift: the forward's GEMMs, which the panels change
only by skipping the down projection's A pack, read x0.920 against the kernel alone (the block's
alternated pairs x1.003), and the step's medians x0.979, faster in 3 of 10 rounds; the step pairs, a
step apart, are the step's measurement.

The state is the same either way. The C harness holds the writers to the binary32 entries' values
converted in place -- shapes with and without the edge values, a row longer than a writer's
1,024-column segment, a NaN in a row's second segment, an overflowing product, values built within an
ulp or two of a bfloat16 midpoint so that a slip of one ulp converts differently, both SiLUs, one thread
and four -- and the grouped entry reading panels to the same calls packed (each kind of call the trainer
makes, a chained group, a panel that belongs to its own call alone, every refusal); the trainer to one
state with the block's panels, without them and mixed, under three bf16 contracts at one thread and
four, with the tensor unit's probe recording the values each writer's job computed; the Python witness
holds a library without panels to the same state; and six steps of the comparison decoder, every SwiGLU
block of it on the panels, end in `08da09e8…` under `bf16+exp+reduce+attn` through the tile, AVX-512,
AVX2 and portable builds at one and four threads, `e91ae701…` under `bf16+exp+reduce`, and `4bf4e057…`
under `kblock+exp+reduce` at 20 steps -- the states before this commit. `tools/testing/faults/ndt-panels.json`
holds 34 faults. Its first complete sweep caught 31: a reassociated product and a v1 flag passed under the
exponential bit move binary32 values by an ulp, which the conversion hides unless the value sits near a
midpoint, so the midpoint cases and the tensor unit's probe joined, and all 34 fire. Two defects were found
in review before the first green run, each now a case of the harness and an entry of the table: the
trainer's predicate checked the rows and the hidden width but not the model's width, so a model whose
width was off the 32 grid would have written panels for GEMMs the grouped entry then refuses, their
outputs unwritten (a second model of width 48 in the trainer's witness now writes none); and the A pack
converts alpha times A while a panel is A alone, so an A panel is admitted at alpha 1 only.

A NaN is held to be a NaN, not a pattern. The native AArch64 C runtime job failed the harness's first
version at its writer comparison -- the SwiGLU VJP's four panels against the binary32 entry's values
converted, over cases that include NaN inputs -- which held NaN elements bit for bit. Which NaN an
operation on two NaNs propagates is the compiled code's (the operand order its instructions take), and
for a NaN gate the sigmoid's NaN has the opposite sign (it is computed from -x), so two compiled copies
of one kernel can hold different NaNs -- the writers call the entries' kernels a 1,024-column segment at
a time, and an element one call computes in a vector body another can compute in a remainder loop or
an inlined copy; the bf16 contract already leaves which one to the code. The comparison now holds every element bit for bit and a quiet NaN where the reference holds one
(`same_panel`, the rule `same_bits_or_nan` holds the GEMMs to), and reports the first difference with
its case. The trainer's finite values agreed there: on the same commit the AArch64 oracle job's Python
witness held the states with the panels and without them equal, its probe showing all three panel
paths taken. On this host, on the writers' generic path ([nan/](measurements/2026-10-18/panels/nan/)), a
writer whose NaN elements differ from the reference's only in their sign fails the first version's
comparison and passes this one; and PN34, the writer storing a NaN element as a signaling NaN, passes a
comparison that takes any NaN and fails this one, which names it: the SiLU product as A, 64 x 96, v1,
one thread, element 318 `0xff81` where the reference holds `0xffc0`. With PN34 the table holds 35
faults.

## 36. The panels over a whole run, against PyTorch's bfloat16 (LM-PANELS)

§35's two changes keep every state, so a whole run measures only what they buy. LM-PAIRED's five seeds
each ran the 1,000-step language model at four threads under three arms in one session, their order
rotated seed by seed -- BCIR `bf16+exp+reduce+attn` on 5804ad5 (BF16-KERN and BF16-PANELS-1) and on
d5bc457 (§34's fastest arm, before them), and compiled PyTorch under autocast -- with §32's analysis
fixed before the runs, the margin graded on every pair ([panels/lm/](measurements/2026-10-18/panels/lm/),
[paired_panels.py](measurements/2026-10-18/panels/lm/paired_panels.py)).

| seed | BCIR with the panels, ms (tokens/s) | PyTorch bf16 | BCIR before them |
|---|---|---|---|
| 20261011 | 68.6 (14,925) | 170.5 (6,007) | 72.2 (14,180) |
| 1001 | 69.9 (14,654) | 175.0 (5,853) | 75.5 (13,559) |
| 2002 | 69.1 (14,826) | 147.1 (6,963) | 75.8 (13,500) |
| 3003 | 69.4 (14,751) | 148.2 (6,907) | 73.9 (13,864) |
| 4004 | 69.0 (14,843) | 150.9 (6,787) | 75.4 (13,578) |

([summary.json](measurements/2026-10-18/panels/lm/summary.json).) With the panels BCIR trains at
**2.282 times** the tokens a second of compiled PyTorch under autocast (95% interval 2.061-2.526),
faster on five of five seeds, its test cross-entropy 0.00054 nats above PyTorch's on average (interval
-0.00073 to +0.00181): inside the margin. Against the same contract before §35 it is 1.078 times
(1.054-1.102, faster on five of five), where §35's step pairs read x1.063 (1.049-1.076) at four threads;
and the two BCIR arms end in one state on every seed, §34's, so their test losses agree to the last bit.
The session ran slower than §34's: the arm before the panels, §34's fastest code, took 74.6 ms a step
(geometric mean) against 70.9 there, PyTorch 157.9 against 142.9, so §34's pair reads 2.117 here against
2.015 there. A session moves the ratio to PyTorch by about 5% by itself; what §35 buys is the in-session
1.078. Peak RSS is 534 MB with the panels -- they sit in the binary32 slots, so the arena (362,944,768
bytes) and the workspace (11,534,592) are unchanged -- against 1,141-1,146 MB for PyTorch; the
persistent state is the same 157,390,848 bytes on every arm.
