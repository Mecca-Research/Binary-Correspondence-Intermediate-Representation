# BCIR Python-to-native AI boundary audit

> Source-backed snapshot: 2026-07-19, package version `0.2.0`, after PRs
> #642–#646. This is a placement and implementation audit, not a claim that all
> Python should be replaced or that one local CPU result is portable.

## 1. Verdict

BCIR needs two deliberately different implementations:

- **Python remains the independent semantic oracle and control plane.** It owns strict
  artifact schemas, R1–R25 legality, K_BCIR decisions, claim/StreamPack construction,
  provenance, bounded search, model ingestion, training orchestration, and readable
  reference math.
- **Portable C owns stable repeated numeric work.** It now owns Q8/Q4 group
  quantization, Q8 projection loops used by standalone Llama inference, exact Q15
  nearest-pattern selection, a group-32 Q4×Q8 primitive, and bounded model
  measurement. These functions have a C ABI and are directly callable from C++.
- **C++ is reserved for a demonstrated ownership/orchestration need.** No new C++
  layer is justified for these kernels: it would add an ABI and lifetime surface but
  would not make their loops faster. A future request scheduler, paged-cache owner, or
  asynchronous device runtime may justify RAII and containers after its lifecycle is
  specified and measured.

This is prototype-then-port, not source replacement. Every native result remains pinned
to the Python oracle, and requesting native execution fails explicitly if its library or
compiler is unavailable. There is no silent fallback.

## 2. Scope and method

The audit read the production Python tree, the C/C++ runtime rails, tests, and the five
AI/control-plane landings from #642 through #646. That interval changed 65 Python files
(including tests) by +12,655/−260 lines while adding one 80-line C source and one
26-line header. The imbalance was useful evidence to investigate; it was not itself a
reason to port code.

Candidates were screened in this order:

1. Is the operation on inference, artifact-conversion, retrieval, or measurement hot
   paths rather than a one-time compiler/control path?
2. Is its input/output contract closed, bounded, and stable enough for a C ABI?
3. Can C preserve exact ordering, tie-breaking, failure behavior, and the Python wire
   representation?
4. Does a bounded measurement show material interpreter overhead?
5. Can the implementation remain portable and ownership-simple without weakening the
   oracle, legality, or provenance boundary?

Static loop counts were used only to locate candidates. Dynamic bounded fixtures and
semantic review decided placement; a large Python function is not automatically a C
candidate.

## 3. Native surface implemented by this audit

[`bcir_ai_kernels.h`](../../runtime/c/bcir_ai_kernels.h) is the versioned shared
C/C++ ABI. Explicitly loaded libraries must report ABI version 1 before Python
configures any data-plane entry point.
[`bcir_ai_kernels.c`](../../runtime/c/bcir_ai_kernels.c) allocates no memory and exposes:

| Operation | Contract and reason for native placement |
|---|---|
| Q8/Q4 group quantization | Finite `double` input, symmetric `[-127,127]` / `[-7,7]` codes, power-of-two exponents, low-nibble-first Q4, odd padding zero; byte-identical to the Python oracle |
| Q8 input-major matvec | Cache-friendly decoder projection preserving increasing-input accumulation for each output; immutable weights are validated by the loader once |
| Q8 output-major row dots | Embedding/LM-head projection with deterministic lowest-token argmax retained by the decoder |
| Exact Q15 top-k | Unsigned-64 squared distance, canonical pattern-index tie-break, optional hard-fact eligibility mask; bounded dimensions make overflow impossible |

[`bcir_q4_kernel.c`](../../runtime/c/bcir_q4_kernel.c) additionally applies separate
little-endian group-32 Q4 and Q8 exponents after exact integer accumulation. It is the
compute primitive for the landed BCIRQ4T slice, not a claim of whole-decoder Q4.

[`bcir_ai_microbench.c`](../../runtime/c/bcir_ai_microbench.c) is a fixed-storage hosted
measurement twin. Dimensions and repetitions are hard bounded; it emits strict,
timestamp-free JSON. `bcir-model-assess --microbench` now uses this native engine by
default. `--microbench-engine python-oracle` retains the old diagnostic reference.

The explicit Python bridge is [`native_ai.py`](../../bcir/kbcir/native_ai.py):

```python
kernels = NativeAIKernels.build(build_dir, cc="clang")
kernels = NativeAIKernels.load(prebuilt_library)
q8 = kernels.quantize(values, group_size=32, bits=8)
matches = NativeOptimizationIndex(kernels, memory).query(query, top_k=8)
```

`write_q8_decoder(..., native_kernels=kernels)` and
`PackedQ4Tensor.from_values_native(..., native_kernels=kernels)` are opt-in export
paths. Omitting the argument keeps the dependency-free Python oracle. Normal `bcir`
imports do not load `ctypes`, search for a compiler, or import this cold module.

The standalone C Llama runtime now routes every attention/MLP matrix-vector projection
and the tied/untied vocabulary head through the native Q8 kernels. It continues to use
the fully validating BCIRQ8 loader, GQA/KV cache, RoPE, RMSNorm, SwiGLU, deterministic
loop order, and lowest-ID argmax. A loaded model is immutable until destruction; it is
not rescanned on every request.

## 4. Safety, determinism, and portability contract

- Kernel pointers are borrowed, input/output spans must not overlap, capacities are
  explicit, and the kernels allocate no heap memory.
- Quantization validates all finite input and capacities before either output changes.
- Checked Q8 entry points reject forbidden `-128`, malformed exponent inventories, and
  invalid geometry before output changes. Arithmetic overflow clears output and fails.
- Q15 inputs are bounded to 65,536 patterns × 4,096 coordinates and top-k 1,024. The
  maximum distance is below `2^64`; no approximate ANN or learned rank can bypass hard
  facts.
- Q4×Q8 group accumulation is exact in `int64_t`, rejects `-8`/`-128`, validates the
  `[-300,300]` exponent bridge, and publishes its scalar result only on success.
- C builds use strict C11 warnings and disabled contraction where deterministic floating
  order matters. Logical `-lm` is removed only for Windows host invocation.
- The bounded benchmark uses fixed arrays, not a hidden allocator. Native build and run
  subprocesses have explicit timeouts.
- Direct C malformed-input tests run under ASan/UBSan/LSan; Python/C byte and arithmetic
  parity, tied/untied standalone decoding, import quarantine, static analyzers, native
  Windows, and native ARM CI remain required.

The audit also fixed a pre-existing scale-selection edge in both rails. The Python oracle
now derives the minimal exponent from the float's exact integer ratio, avoiding finite-
subnormal division underflow, `log2` boundary variance, and conversion overflow for wide
`_BitInt` code ranges. Q4/Q8 C derives the same mathematical exponent with `frexp` and
integer guards, without a transcendental dependency.

## 5. Bounded local evidence

One single-threaded diagnostic run used CPython 3.10.12 and Clang 22.1.8 under WSL on an
AMD Ryzen 5 2600. Times include Python-to-C value marshaling for quantization and Q15.
They are observations, not CI thresholds or cross-host claims.

| Fixture | Python oracle | Native C path | Observed ratio |
|---|---:|---:|---:|
| Q8 group-32 quantize, 262,144 values | 256.645 ms | 64.132 ms | 4.00× |
| Q4 group-32 quantize/pack, 262,144 values | 334.426 ms | 112.559 ms | 2.97× |
| Exact Q15 top-8, 4,096 × 128 | 105.131 ms | 0.925 ms | 113.70× |
| Q8 prefill reference, dimension 64 | 59.930 ms | 0.378 ms | 158.71× |
| Q8 decode reference, dimension 64 | 0.849 ms | 0.006 ms | 146.36× |

The benchmark ratios justify moving the measurement and repeated data path. They do not
show that scalar portable C is peak hardware code; target BLAS, LLVM vectorization,
SIMD kernels, and accelerator backends still require target-specific comparison.

## 6. Complete placement register

| Python area | Decision | Reason / promotion gate |
|---|---|---|
| `frontends/models/decode.py`, `quantized.py` | Keep oracle; C decoder is production twin | Independent float/Q8 parity is more valuable than deleting duplicate reference math |
| `kbcir/quantize.py`, `lowbit.py`, `weights_io.py` | Native conversion added; Python retained | Repeated numeric loops are stable; schemas, CRCs, canonical order, drift/NLL admission stay Python authorities |
| `kbcir/optimization_memory.py` | Native exact Q15 index added; Python filters and oracle retained | Large exact scans benefit immediately; hard facts and artifact semantics must not move into fuzzy/native retrieval |
| `frontends/models/model_microbench.py` | Native is default measured engine; Python remains diagnostic | Interpreter timing is not useful target-kernel evidence |
| `frontends/models/inventory.py`, `assessment.py`, `execution_plan.py` | Keep Python | Header parsing, exact accounting, finite candidate enumeration, claims, and StreamPack are one-time control work; native port has no measured payoff |
| `kbcir/ham.py`, `static_memory.py`, context-shard artifacts | Keep Python | Compiler/simulator planning and proof generation are bounded and provenance-heavy, not per-token loops |
| `kbcir/hardware_rl.py` and hosted policy model | Keep Python/PyTorch | PUCT uses dynamic evaluators and typed plan objects; PyTorch already executes tensor work in C++/ATen. Freeze a callback/trace ABI and measure before a native search engine |
| Hosted pretrain/SFT/RM/DPO/PPO/reasoning | Keep Python orchestration | Device kernels already come from PyTorch; BCIR owns safe state, objectives, and artifacts, not a duplicate autograd runtime |
| AD, losses, optimizers, planned/streamed training | Keep graph oracle; use existing emitted C kernels | Graph transformation readability and closure proofs dominate; compile fixed realized kernels rather than porting the compiler itself |
| K-means/KNN/PCA/OLS/recurrent/transformer references | Keep reference now | #645 removed avoidable algorithmic overhead. Port only a frozen inference workload with a target and a differential contract; use vetted numerical providers where appropriate |
| Tokenizers, corpus cleaning, provider adapters | Keep Python now | Text/provenance/policy work is not the standalone token-ID inference bottleneck; profile production ingestion before adding a native parser |
| Paged KV and serving scheduler | C++ later, gated | Requires request ownership, cancellation, generation-safe cache reuse, backpressure, deadlines, and measured concurrency before an ABI can be frozen |
| Whole-model Q4 | C after format/quality closure | First freeze decoder wire layout, activation/outlier policy, drift/NLL, tied/untied parity, and target kernels; the current grouped primitive is not enough |
| GPU Q8/Q4 and custom attention | Vendor/LLVM/CUDA rail later | Establish SDPA parity and measurements first; portable scalar C is not a GPU backend |
| R1–R25, K_BCIR legality, telemetry governance, provenance | Do not port for speed | These are deterministic authorities and are not allowed in the per-token L0 hot path |

## 7. Next native milestones

1. Keep this scalar C ABI as the cross-target truth and add SIMD implementations only
   behind runtime feature dispatch plus scalar/ISA differential tests on each target.
2. Finish the whole-model Q4 contract and quality gates before wiring Q4 into the decoder.
3. Profile the actual standalone decoder by operator and memory tier; address embedding,
   KV movement, and vocabulary projection based on counters rather than intuition.
4. Define a request-owned paged-KV/serving lifecycle, then choose C++ only if its
   ownership/concurrency machinery is a net reduction in risk.
5. Keep hardware-RL/HAM planners off L0. Native search or ANN adapters require a stable
   trace ABI, exact hard-filter parity, bounded cancellation, and measured compiler-time
   pressure.
6. Build GPU/vendor kernels only through the target schedule, R17, provenance, and
   real-hardware promotion gates. No learned model or native kernel changes legality.

This register is the decision record for future Python-to-native proposals. A new port
must name a measured bottleneck, stable boundary, independent parity oracle, ownership
model, target, and rollback path.
