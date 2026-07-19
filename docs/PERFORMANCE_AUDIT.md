# BCIR bounded performance and TMSAO audit

> Source-backed snapshot: 2026-07-18, package version `0.2.0`.
>
> “Theoretical Maximum System Architecture Optimization” (TMSAO) is an
> aspirational search target, not a portable benchmark result. A theoretical maximum
> depends on a concrete processor, memory hierarchy, compiler, frequency/power state,
> workload, and proof that all relevant candidates were considered. BCIR uses the term
> here for a measurement-and-certificate discipline; this document does not claim that a
> Python run attained a hardware maximum.

## 1. Gate and evidence contract

The dependency-free audit is available through either entry point:

```bash
python -m bcir.performance_audit --repeats 3 \
  --output build/performance/tmsao-report.json
python tools/perf/run_tmsao_audit.py --repeats 3
```

The timestamp-free `bcir.tmsao_audit.v1` report covers these bounded organs:

| Group | Exercised path |
|---|---|
| Graph | deep reverse-insertion phase DAG, verification, deterministic execution |
| GEM | greedy waves, explicit async dependencies, duration-aware EFT placement |
| K_BCIR/StreamPack | tiled graph optimization, pipelined hydration, plan/pack/provenance laws |
| Memory | odd-size/alignment/lifetime static address placement and independent alias verification |
| Telemetry | fixed ring publication, wraparound, overwrite/loss accounting, drain |
| Quantization | group-32 Q8 and packed Q4 code/exponent paths |
| Unsupervised/classical ML | K-means, KNN, StandardScaler, and irregular embedding gathers |
| Linear algebra | tiled GEMM, OLS, PCA, and independent residuals |
| Sequence ML | Transformer, LSTM, and GRU references |
| Training/search | closed-set autodiff with Adam and finite-portfolio hardware PUCT |

Correctness, finite output, complete claim/resource census, repeated-result identity,
and report structure are strict gates. `definition_sha256`, per-case result hashes, and
`correctness_sha256` exclude timings. `min_ns`, `median_ns`, and `max_ns` are informative:
shared CI never fails because a noisy runner crossed an absolute latency threshold. Existing
controlled-box performance budgets remain available through `tools/perf/check_budgets.py`
with `BCIR_BAREMETAL=1`.

Optional PyTorch training, train-to-C parity, and hardware-policy learning remain separate
one-thread gates. The dependency-free audit neither imports PyTorch nor starts a GPU job.

## 2. Defects and bottlenecks found

The sweep used deep DAGs, independent graphs, dense conflicts, reverse textual order,
irregular lifetimes, ring wraparound, repeated/tied values, odd tensor extents, non-finite
values, and fractional/Boolean identifiers. It found and fixed the following:

1. **Deep legal phase graphs exhausted Python recursion.** Verifier, executor, realizer,
   scheduler, and StreamPack each carried a recursive or duplicate traversal. One shared
   iterative traversal now preserves the historical dependency/insertion order and handles
   deep DAGs and cycles without relying on the interpreter recursion limit.
2. **GEM scheduling did quadratic work on independent claims.** Wave, async-token,
   overlap, and EFT paths repeatedly compared every claim pair and allocated temporary
   sets. Per-resource reader/writer summaries now derive the same hazards and exact greedy
   waves in work proportional to resource touches plus dependency edges. EFT dispatch uses
   an indegree/successor graph and priority heap while preserving the old
   `(-duration, claim_id)` tie-break.
3. **Resource liveness followed textual phase order.** A valid DAG whose phases were stored
   out of order could alias a long-lived tensor with a logically middle tensor. Allocator
   profiling, pooling, and static memory now use executable topological positions. A
   deterministic regression demonstrates the formerly unsafe layout.
4. **Static memory repeatedly rescanned every prior allocation.** The planner now expires
   lifetimes through a heap and allocates from a coalescing address-ordered free list. It
   preserves the prior lowest-address aligned first-fit layout exactly; randomized tests
   compare every offset with the former algorithm.
5. **K-means and KNN copied feature rows in inner loops.** Flat offset-based squared distance
   removes those temporary slices. KNN uses bounded `nsmallest` when `k` is small; K-means
   stops on exact centroid convergence without changing its final result.
6. **StreamPack hydration accepted incomplete plans.** A partial result could silently emit
   a truncated executable pack. Hydration now rejects unknown, duplicate, phase-mismatched,
   out-of-topological-order, or missing claim steps before publication.
7. **Odd ML inputs could be silently coerced or become NaN-driven choices.** K-means,
   embeddings, KNN, scalers, and folds now reject fractional/Boolean structural values,
   non-finite observations, and fractional class labels where an exact integer is required.
8. **Nested C gates rediscovered a stale unversioned compiler.** On hosts that expose an
   older `clang` alongside a newer versioned installation, the runtime gate now selects a
   C23-capable default and passes that exact compiler into its sanitizer and StreamPack
   semantic sub-gates. Explicit caller-selected compilers remain authoritative.

Each item has deterministic differential or refusal coverage in
`bcir/tests/test_tmsao.py` and `bcir/tests/test_performance_audit.py`. Timing alone is not
the regression oracle.

## 3. Local before/after evidence

The following representative observations were captured on the same bounded WSL x86-64
development session with CPython 3.10. They compare the pre-change `origin/main` worktree
with this change. They are diagnostic evidence, not portable speed claims or CI floors.

| Fixture | Before | After | Observed ratio |
|---|---:|---:|---:|
| K-means, 1024×16, k=8, four iterations | 113.68 ms | 80.58 ms | 1.41× |
| GEM waves, 1000 independent claims | 381.70 ms | 1.96 ms | 194.7× |
| Async dependencies, 1000 independent claims | 385.64 ms | 1.80 ms | 214.2× |
| EFT, 1000 independent claims | 1098.89 ms | 9.08 ms | 121.0× |
| GEM waves, 1000 shared-resource claims | 488.01 ms | 3.03 ms | 161.1× |
| Async dependencies, 1000 shared-resource claims | 353.78 ms | 28.81 ms | 12.3× |
| EFT, 1000 shared-resource claims | 7185.33 ms | 244.67 ms | 29.4× |
| Static plan, 2000 overlapping lifetimes | 2083.52 ms | 141.11 ms | 14.8× |
| Static plan, 2000 sequential lifetimes | 241.24 ms | 190.75 ms | 1.26× |

The dense shared-resource cases still have quadratic-size dependency output: representing
every required predecessor edge is inherently more expensive than the independent case.
The improvement removes avoidable search and allocation overhead; it does not pretend the
output itself is subquadratic.

### 3.1 Native AI data-plane follow-up

The 2026-07-19 Python/native audit then measured the post-#642 AI surfaces on the same
bounded WSL x86 system. It moved stable numeric work—not legality or planning—into a portable
no-heap C ABI. One CPython 3.10.12/Clang 22 observation recorded Q8 and Q4 conversion at
4.00×/2.97× the Python rate including marshaling, exact 4,096×128 Q15 top-k at 113.70×, and the
dimension-64 Q8 prefill/decode measurement kernels at 158.71×/146.36×. These are diagnostic
ratios, not shared-CI floors.

The standalone C decoder now uses the same Q8 projection kernels, while Python remains the
float/Q8 differential oracle. The complete module placement, raw observations, safety contract,
and explicit non-ports are in
[`BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md`](machine-learning/BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md).

## 4. What remains hardware- and workload-gated

- Native PMU/cache/register-pressure counters, energy, thermal throttling, and stable
  frequency control require a controlled physical rig. WSL/shared-runner timings cannot
  certify them.
- Direct ARM, Raspberry Pi, and accelerator results remain hardware-gated. GitHub’s native
  ARM job proves portability, not Raspberry Pi performance.
- Vendor BLAS, GPU kernels, memory-controller behavior, real StreamPack device execution,
  and large-model layer streaming require target-specific measured candidates.
- The Python oracle prioritizes semantic transparency. Production throughput comes from
  verified C/LLVM/vendor realizations. The model-assessment CLI now defaults to its bounded
  native C measurement twin; the Python microbenchmark remains useful for algorithmic
  complexity and orchestration overhead, not peak FLOP/s.
- Full TMSAO evidence for a target requires calibrated counters, exhaustive or bounded-
  exhaustive candidate comparison, confidence intervals, thermal/energy context, exact
  output/parity certificates, and reproducible provenance. No current artifact closes that
  entire target-specific proof.

These boundaries align with
[`HARDWARE_VALIDATION.md`](kernel/HARDWARE_VALIDATION.md) and the model scheduling program in
[`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md).
