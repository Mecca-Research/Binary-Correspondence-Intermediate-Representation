# BCIR vs Clang: a fair comparative performance analysis

Reproduce: `python3 -m bcir.clang_compare` (harness in `bcir/clang_compare.py`;
gated guards in `bcir/tests/test_clang_compare.py`). Numbers below were measured on
an Intel Xeon @ 2.80 GHz, Clang 18, 4 cores.

## The fair frame

BCIR is a **planning / cost-governance layer that delegates instruction selection
and register allocation to LLVM** — it emits C, Clang compiles it. So it cannot, by
construction, *beat Clang at codegen*: on a kernel Clang already compiles optimally,
the ceiling is a **tie**. The honest question is therefore two-sided:

1. Where BCIR can't beat Clang (simple, already-optimal kernels), does it **MATCH** —
   i.e. does its emitted code regress nothing?
2. Where BCIR makes a **semantic choice Clang's backend cannot** (the access pattern
   / reduction order / stride-vs-gather — decisions that need program intent Clang
   doesn't have), does it **WIN**, and by how much on silicon?

To isolate BCIR's *planning* contribution from codegen, every comparison **holds the
compiler constant** (one Clang, one flag set) and pits BCIR's planned realization
against the naive/default one a developer would write. Methodology: each variant in
its **own binary**, self-timing best-of-R, run **alternated** for N trials, **median**
reported (cache-resident kernels are noise-sensitive; this cancels drift and the
warm-cache order bias a single-binary A/B suffers).

## Results

| Data-flow process | BCIR plan | Naive/default | Clang | Speedup | Verdict |
|---|---|---|---|---|---|
| Elementwise, streaming (N=4M) | idiomatic loop | idiomatic loop | `-O3` | **0.98×** | **MATCH** |
| Elementwise, L1-resident (N=4K) | idiomatic loop | idiomatic loop | `-O3` | **1.00×** | **MATCH** |
| Gather avoidance (shuffled index) | contiguous `A[i]` | gather `A[idx[i]]` | `-O2` | **6.0×** | **WIN** |
| Reduction order (permutation) | blocked sum | gather-reduce | `-O2` | **14.1×** | **WIN** |
| Strided access | direct `X[i*k]` | gather `X[idx[i]]` | `-O2` | **1.33×** | **WIN** |

### Plan / compile overhead (the cost side)

| Metric | Value |
|---|---|
| BCIR plan + emit (warm) | **0.31 ms** — ~1% of the compile that happens anyway |
| Clang compile of the kernel | 33.9 ms (dominates) |
| BCIR cold first-use (full process) | ~71 ms (≈58 ms one-time Python import + plan + emit) |

## Where we WIN

Only — and exactly — where BCIR exploits **program intent that Clang's backend does
not have**:

- **Gather avoidance (~6×).** The naive code reads `A[idx[i]]` because the source was
  written that way; BCIR knows the index is a permutation/identity and the contiguous
  layout is *equivalent*, so it plans `A[i]`. Clang must honor the gather it was
  given — it cannot change your data's access pattern. This is the realized
  `gather_penalty` (random DRAM misses) avoided.
- **Reduction order (~14×).** Integer addition is associative, so a permuted
  gather-reduce and a blocked sequential reduce give the *bit-identical* result —
  BCIR picks the cache-friendly order; Clang can't reorder memory accesses it can't
  prove equivalent.
- **Strided vs gather (~1.3×).** BCIR knows the access is `X[i*k]` and emits direct
  strided addressing instead of an indexed gather.
- **Budget feasibility (a correctness win, not a speed one).** Under a thermal/power
  cap, BCIR emits the *feasible* vec8 where a naive max-width vec16 would violate the
  budget — a constraint Clang has no notion of.

## Where we MATCH

On **dense, regular kernels with no access-pattern choice** (plain elementwise),
BCIR ties Clang at `-O3` (0.98×–1.00×). This is the intended ceiling: BCIR emits the
idiomatic loop Clang auto-vectorizes, so there is nothing to win — and, importantly,
**nothing lost** (the earlier hand-blocked-loop regression was removed in the
width-aware codegen work). BCIR's planned width is a *floor* the compiler meets.

## Where we LOSE (honest)

- **Codegen: BCIR never beats Clang — by design.** It *is* Clang's backend. On pure
  compute, MATCH is the best possible outcome; there is no scenario where BCIR's
  emitted C out-runs what Clang would do with the same source.
- **Cold-start import (~58 ms, one-time).** A single one-shot kernel pays the Python
  import before the (unavoidable) compile. Amortized across a session it vanishes,
  and the *per-kernel* planning is negligible (0.31 ms, 1% of a compile) — but for a
  literal one-and-done invocation, Clang-alone has lower end-to-end latency.
- **No win without intent.** If a workload has no avoidable gather, no reorderable
  reduction, no stride knowledge, and no budget constraint, BCIR has nothing to add
  over Clang — it MATCHES. The wins are not free; they require the planner to *know*
  something the source obscures.
- **Python is not the runtime.** The oracle plans/emits ahead of time; it is not a
  hot-path JIT. Its decisions must be frozen/compiled to be fast — which is the
  intended design, but means BCIR-the-Python-oracle is a compile-time tool, not a
  runtime competitor.

## Bottom line

BCIR meets its design contract precisely: it **ties Clang where codegen is the whole
game** (simple dense kernels, ±2%) and **beats it multiplicatively where planning-level
program intent matters** (irregular memory: 1.3×–14×). It loses only on the axes a
planning layer is *expected* to: it cannot out-codegen the backend it delegates to,
and it carries a one-time startup cost. The value is real and bounded: **a
cost-governed access-pattern/feasibility planner on top of LLVM, not a faster LLVM.**
