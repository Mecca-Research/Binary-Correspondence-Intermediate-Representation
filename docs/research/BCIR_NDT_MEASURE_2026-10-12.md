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
  quarter more under `exp` than under libm (§3), the same values a row-wide call would produce
  faster.

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
  the same values, so the `exp` state digest stays -- and it is the first item of the next attention
  slice (§5).
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
  of the step; NDT-ATTN is for longer contexts, and the row-loop fix is its first item.
- **AdamW and the clipping norm.** The commit scales on the pool; the clipping norm and the
  finiteness check are the serial pass they were, and the update reads 76.5, 84.1
  and 71.5 ms at one, three and four threads under v1 -- the budget's "~30 ms at four
  threads" needs NDT-NORM (the norm's partial sums gathered as the backward writes each gradient,
  the index-order sum kept) to be true.
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
```

The GEMM-rate record is the output of a scratch driver over the trainer's own entry points
(`bcir_tensor_mm_pool`, `bcir_tensor_mm_fused_pool`, `scipy_cblas_sgemm`) on the step program's
shapes (`native_program.training_step_program(...).gemm_shapes()`); its arguments are in the
record. The exact rows are the GEM+ `train` group's
(`python tools/perf/gemplus_baseline.py --compare --group train`).
