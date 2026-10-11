# GEMplus, ASN.1 and C machine learning performance assessment — 2026-10-10

The next update should reuse GEMplus's compact dependency frontier in the async
token path, reduce the cost of producing/checksumming StreamPack bytes, specialize
ASN.1 fixed-width array decoding, and improve the native training kernels. The
current C BLAS seam already performs close to PyTorch on the tested dense shapes;
the portable matrix fallback and attention kernels have considerably more work left.

This is a measured assessment and a proposed update plan. This PR changes research
documentation and its reproducible measurement harness; it does not change the
execution core, codecs, model architecture, runtime manifest or training semantics.

## Source and measurement scope

The latest merged `main` fetched for this session was
`75d50d4f6e280e35a25032024edffd92f25c52f2` (PR #813). No later merged GEMplus/ASN.1
revision was available. These are fresh measurements of that revision, not a
before/after claim against a newer training release. Earlier audit “missing” items
must be checked against today's tree: GC pauses, slotted hot value records, native
hydration, the oracle inventory, expected-cost objectives, SDF/timed regions,
roofline/pebble bounds and joint scheduling/memory search have already landed.
See the [current roadmap](BCIR_GEMPLUS_ROADMAP.md) and
[oracle inventory](../BCIR_ORACLE_INVENTORY.md).

Host: x86-64 AMD EPYC 9V74 virtual machine, eight CPU cores of quota, 8 GiB memory,
GCC 13.3, Python 3.12.14; CPU compute comparisons use one thread. No GPU, native
ARM, PMU, energy or bare-metal measurement is claimed. PyTorch 2.14.1+cpu,
NumPy 2.3.5 and SciPy 1.17.0 are experiment dependencies, not new BCIR dependencies.
Timings are indicative on this VM; raw samples, scopes, source hashes and output
checks are in [results.json](measurements/2026-10-10/results.json).

Fixtures and input construction are outside timers. Python result allocation is
inside; cyclic collection between samples is outside, and production GC decorators
remain active. Native planner/hydrate timers reuse admitted input views and
caller-owned scratch/output; initial record decode and size queries are outside.
Consequently native hydrate is compared with **plain Python hydrate + encode**,
not with Python's pipelined scheduler. These are warm-working-set measurements.
Python stages use five timed samples after warmup; each native GEM row has three
independent harness invocations, each with nine rounds of five calls. Codec, CRC,
frontier-control and tensor details are recorded separately below.

## 1. GEMplus and StreamPack scaling

The tiled-matmul fixture scales from 64 to 32,768 claims. These calls plan and
serialize graph descriptors; they do not execute or train those matrix products.
Every Python encode/decode round trip preserves bytes. At all four sizes the
native hydrate's entire plain StreamPack equals Python's, including its checksum.

Median milliseconds per call:

| Claims | Python plan | C plan | Python hydrate + encode | C hydrate | Python encode | Python decode |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.431 | 0.031 | 0.352 | 0.035 | 0.210 | 0.350 |
| 512 | 2.987 | 0.264 | 2.586 | 0.286 | 1.447 | 2.619 |
| 4096 | 25.064 | 2.241 | 23.378 | 2.462 | 11.498 | 20.879 |
| 32768 | 259.806 | 17.760 | 273.119 | 19.946 | 115.628 | 177.670 |

At 32,768 claims the C planner is **14.6×** faster than the Python
planner, and C plain hydration/serialization is **13.7×** faster than Python's combined
path under the stated timing scopes. This includes avoiding Python object work;
it does not isolate an algorithmic advantage over another compiler.

Eight times as many claims from 4,096 to 32,768 raises Python planning time by
10.37× and native planning by 7.92×. That is near linear at this range,
with visible constant-factor/working-set effects. It is not evidence about
millions of claims, distributed execution or multi-trillion-parameter models.

Peak additional Python allocations, measured separately with `tracemalloc`
(not RSS, not inside the timing samples):

| Claims | Pipelined wire bytes | Plan peak MiB | Decode peak MiB |
| --- | --- | --- | --- |
| 64 | 11181 | 0.080 | 0.088 |
| 512 | 88577 | 0.645 | 0.670 |
| 4096 | 712521 | 4.221 | 5.296 |
| 32768 | 5762585 | 32.263 | 42.312 |

**Recommended update:** use native views/columnar arrays on production paths that
only need validated wire records, rather than rebuilding every record as a Python
object. Retain the Python oracle for conformance and diagnostics. Measure the
remaining `Candidate`/step/decoded-record allocation costs before adding more
slots or caches; the original GC/slot work is already present. Keep cache keys
bound to schema, target and generation, with explicit invalidation.

### Incremental chain: valid edits, identical complete outputs

Each sample changes one claim's count **downward within its resource bounds**.
The original audit count-increase witness deliberately produces R7 overruns;
this run instead uses accepted application edits. All 28 updated packs are
byte-identical to a full rebuild, all verifier diagnostics are empty on both
paths, and no verifier rebuild is needed.

| Claims | Delta ms | Full chain ms | Full / delta | Updated wire bytes |
| --- | --- | --- | --- | --- |
| 64 | 0.204 | 1.466 | 7.193 | 11181 |
| 512 | 0.347 | 10.534 | 30.352 | 88577 |
| 4096 | 1.473 | 93.180 | 63.252 | 712521 |
| 32768 | 11.847 | 1052.664 | 88.852 | 5762585 |

Incremental recomputation is effective, but a one-claim edit still grows more
expensive with the whole graph. `delta_pack.PackState._splice` copies fresh record
lists, joins the full body, computes its CRC and appends the trailer into another
bytes object. The public contiguous `Link.data` contract necessarily writes a
complete changed artifact; dependency-cone-local planning alone cannot make that
output cost constant.

**Recommended update:** port the delta chain/record splice to C after measuring
each component; write an admitted contiguous artifact into one allocation and
avoid the body-plus-trailer copy. Consider an internal persistent chunk view with
lazy materialization at an ABI boundary, and composed IEEE CRCs for unchanged
chunks. That is a new ownership/materialization contract requiring parity and
generation tests, not an in-place mutation of a previously published immutable link.

### Async edges still have an avoidable growth path

Current waves, tokens and EFT schedulers on the mixed shared-resource fixture:

| Claims | Async await edges | Waves ms | Tokens ms | EFT ms |
| --- | --- | --- | --- | --- |
| 512 | 189 | 0.608 | 0.741 | 2.037 |
| 1024 | 810 | 1.141 | 1.501 | 4.223 |
| 2048 | 3304 | 1.980 | 2.990 | 8.547 |
| 4096 | 13344 | 3.790 | 6.530 | 18.578 |
| 8192 | 53616 | 7.630 | 16.727 | 41.687 |

`async_plan` still calls the full `hazard_predecessors`, while scheduling already
has `_frontier_predecessors`. A separate control compares every node's entire
transitive predecessor set with integer bitsets; all five fixtures have identical
reachability after frontier reduction:

| Claims | Full edges | Frontier edges | Full hazards ms | Frontier ms |
| --- | --- | --- | --- | --- |
| 512 | 189 | 189 | 0.409 | 0.402 |
| 1024 | 810 | 810 | 0.886 | 0.838 |
| 2048 | 3304 | 2776 | 2.872 | 2.464 |
| 4096 | 13344 | 6768 | 4.413 | 3.589 |
| 8192 | 53616 | 14752 | 12.309 | 7.330 |

At 8,192 claims, the frontier removes **72.5%** of edges and takes
1.68× less time to construct. This is a verified opportunity on the fence-free
fixture, not a production patch. Publish a canonical compact hazard API shared by
tokens and scheduling; decide whether explicit-edge identity or partial-order
identity is the token contract, then update the verifier and MLIR projection
together. Require full RAW/WAR/WAW, fences, cross-phase/event, sparse-tail,
adversarial and differential corpora before switching it on.

### Native serialization: checksum work remains expensive

The production freestanding IEEE CRC32 is table-driven, one dependent table
lookup per byte. An isolated comparison checks exactly the same input/output
against native zlib, without dropping any checksum validation:

| Bytes | BCIR CRC ms | zlib CRC ms | BCIR / zlib |
| --- | --- | --- | --- |
| 16384 | 0.036 | 0.004 | 8.261 |
| 131072 | 0.285 | 0.033 | 8.697 |
| 1048576 | 2.351 | 0.260 | 9.049 |
| 8388608 | 18.855 | 2.122 | 8.886 |

The native hydrate is also hundreds of times above its separately measured
write-only floor. That floor excludes validation, layout and CRC work, so it is
not a promised attainable runtime. The CRC experiment shows substantial room in
one required primitive; it is **not** an attribution profile of the hydrate.

**Recommended update:** benchmark portable slicing-by-8/16 IEEE CRC and optional
hardware implementations, preserving polynomial, init/final convention and wire
bytes. Keep a portable fallback. Profile native hydration's validated counting
and writing passes separately; use the same target/compiler/record layout for
the profile and end-to-end test. Do not remove validation to approach `memset`.

## 2. Latest generated ASN.1 C codecs

The fixed-size schema is `SEQUENCE (SIZE(N)) OF INTEGER (0..259)`. All paths
validate that range and materialize the same uint32 token array; generated COER
and UPER additionally build a uint16 arena array. Fixed schema metadata is
implicit, not transmitted as a self-describing envelope. A custom grouped
nine-bit reader consumes **exactly the generated UPER bytes**. Native encoders
match independent Python oracle bytes. All 32 encode/decode parity checks and
80 selected truncation/trailing-byte checks pass. These are focused performance
witnesses, not a replacement for malformed-input or sanitizer gates.

Median native decode + uint32 materialization, nanoseconds (11 interleaved rounds):

| Tokens | Raw16 bytes / ns | COER bytes / ns | UPER bytes / ns | Same-byte packed9 ns |
| --- | --- | --- | --- | --- |
| 64 | 128 / 43.7 | 130 / 43.2 | 72 / 65.3 | 35.2 |
| 512 | 1024 / 241.4 | 1027 / 316.8 | 576 / 460.8 | 296.3 |
| 4096 | 8192 / 1920.1 | 8195 / 2664.8 | 4608 / 3732.2 | 2319.6 |
| 32768 | 65536 / 16344.8 | 65539 / 20939.0 | 36864 / 29314.7 | 18958.2 |

UPER saves 43.75% of bytes against uint16 raw storage at every tested size.
COER has only the fixed count-prefix overhead here. UPER is slower to decode than
raw16; compression is not free. The grouped reader demonstrates an optimization
opportunity in the generator/consumer boundary without changing the representation.
Its hand-written encoder is slower than generated UPER (raw samples include both
directions); therefore the decoder result does not license replacing the encoder.
The reader only handles these multiple-of-eight fixed-size arrays.

**Recommended update:** emit grouped fixed-width integer decode when the schema
proves its size, bit width and range, with a generic fallback for tails, different
bit offsets, extensions and variable sizes. Add decode-into-typed-buffer or typed
visitor APIs to eliminate the uint16 intermediate when the consumer needs uint32.
Make capacity, alignment, aliasing and error-side-effect contracts explicit.
Use schema-derived allocation bounds, then hoist length checks only when those
bounds prove every access safe. Preserve truncation, padding and range refusals.

### “Faster than pure binary” is workload-specific

The repository's ITS reproduction was also rerun: 40 native encode/decode checks,
three invocations of 24 interleaved rounds × 3,000 operations. Tuned schemas retain
the logical content of the reconstructed 2015 secured-message protocol.

| Envelope | Binary bytes | Tuned COER bytes | COER total / binary | Tuned UPER bytes | UPER total / binary |
| --- | --- | --- | --- | --- | --- |
| 0 | 96 | 93 | 0.741 | 86 | 1.454 |
| 1 | 222 | 216 | 0.649 | 203 | 1.324 |
| 2 | 233 | 226 | 0.649 | 213 | 1.352 |
| 3 | 230 | 224 | 0.654 | 212 | 1.343 |

Tuned COER wins encode-plus-decode time on these four envelopes; tuned UPER uses
fewer bytes but takes longer. The binary control is BCIR's reconstruction, not an
independently benchmarked deployed implementation, and the same generator emits
both controls. The fixed token test above reaches a different latency ranking.
Neither result establishes that ASN.1 generally beats a good custom binary codec.
Keep separate profiles for latency, storage and link bandwidth; price **encoding +
transport + decoding + consumer conversion**, with measured target-specific costs.

## 3. Machine learning library: kernel-level comparison

These new tests use identical deterministic FP32 matrices, dimensions, transpose,
alpha=1/beta=0, preallocated outputs and one compute thread. BCIR's portable C
kernel and its existing C OpenBLAS adapter are compared with `torch.mm`; every
result also passes an independent FP64 check (atol 2e-5, rtol 1e-4). Eleven rounds
rotate backend order. The C ctypes and PyTorch entry overhead are included, so
small-matrix timings are not pure inner-kernel throughput. Dense sizes are useful
for the current decoder's linear layers, but are not full training benchmarks.

| M × N × K; B transposed | Portable C ms | C OpenBLAS ms | PyTorch ms |
| --- | --- | --- | --- |
| 64 × 64 × 48; False | 0.082 | 0.011 | 0.007 |
| 256 × 384 × 384; False | 14.623 | 0.787 | 0.778 |
| 256 × 1024 × 384; True | 24.636 | 2.141 | 2.156 |
| 384 × 384 × 256; False | 14.834 | 0.793 | 0.819 |

On the three dense shapes the optimized C provider is close to PyTorch; the
portable fallback is much slower. **Keep the execution core in C.** C can call
optimized native libraries and implement SIMD or tiled kernels; C++ is useful
selectively at framework/accelerator integration seams, not a requirement for
these numerical improvements. The existing benchmark decoder already used BLAS,
so the fallback/provider gap cannot be claimed as an additional full-training gain.

### Causal grouped-query attention

Four batches, eight query/four KV heads, head width 48, no dropout, FP32, one CPU
thread; C forward output matches PyTorch CPU SDPA. Both prepare input outside the
timer. C uses its native B-T-head-channel layout and preallocated full probability
matrix; Torch uses contiguous B-head-T-channel layout with KV replication prepared
outside the timer and allocates its result inside. This tests practical kernel
implementations with equal outputs, not identical temporary-memory contracts.

| Context | C forward ms | Torch SDPA ms | C / Torch | C probability MiB | Max absolute error |
| --- | --- | --- | --- | --- | --- |
| 64 | 2.246 | 0.481 | 4.665 | 0.500 | 3.4e-08 |
| 128 | 8.738 | 1.714 | 5.096 | 2.000 | 3.7e-08 |
| 256 | 35.676 | 6.104 | 5.845 | 8.000 | 4.8e-08 |

The current attention path materializes O(B·H·T²) probabilities, and its runtime
grows approximately quadratically across the tested contexts. **Prioritize a
tiled/online-softmax attention forward with a matching VJP**, retaining the exact
causal mask and correct summation into shared GQA KV gradients. Extend the provider
seam for attention as well as GEMM/SiLU; keep the scalar implementation as an oracle.
The forward difference alone does not predict a full-decoder speedup or prove
backward accuracy. Require finite-difference/independent autodiff checks, extreme
logits, odd tails, non-power-of-two heads, and full-training quality measurements.

### Earlier 9.84M decoder evidence, not a new rerun

Because production model/kernel sources did not advance beyond the previous
experiment, this session adds scaling and kernel controls rather than relabeling
an unchanged full run as a new update. The earlier run's full raw record is
archived in [prior-decoder-comparison.json](measurements/2026-10-10/prior-decoder-comparison.json).
Its complete reproduction bundle was delivered with that earlier report; the
harness in this PR reproduces the **new** data-structure/kernel measurements only.

That experiment used the same 9,836,928-parameter six-layer/384-width decoder,
96 clipped AdamW updates, B=4/T=64, identical initialization/data/rates, one CPU
thread, three timing repetitions; PyTorch used fullgraph compilation/fused AdamW.

| Earlier backend | Median ms/update |
| --- | --- |
| bcir_stock_mkl_sleef | 345.564 |
| bcir_stock_openblas | 361.381 |
| bcir_vector_cached | 303.983 |
| bcir_vector_coer | 304.944 |
| bcir_vector_raw16 | 305.584 |
| bcir_vector_uper | 302.593 |
| pytorch_compiled_fused | 198.305 |

The external vector AdamW prototype reduced native MKL/SLEEF update time by 12.0%,
and all native MKL/encoding variants' complete weights, gradients and optimizer
moments were bit-identical. It was an AVX-512 experiment, not a production update
or ARM result. PyTorch still led the best comparable cached-vector path by about
1.53×. The three encoded-input medians differ by less than 1% from cached vectors;
they do not establish a training speedup from ASN.1. Five alternative approximately 10M-parameter
architectures failed the predefined validation-quality screen; the original was
retained. The [archived screen](measurements/2026-10-10/prior-architecture-screen.json)
records its selection rule and candidate scores. Do not select a faster, less capable model as a backend win.

## 4. Recommended next update, in priority order

| Priority | Component and concrete change | Acceptance evidence |
| --- | --- | --- |
| P1 | **Async graph representation:** shared compact hazard/frontier API; align tokens, verifier and MLIR with a declared partial-order contract. | Exact reachability and execution equivalence across RAW/WAR/WAW, fences, phases/events and sparse tails; edge count and allocation scaling through ≥32K claims. |
| P1 | **C training optimizer:** port the proven vector candidate passes with portable, AVX2/AVX-512 and ARM variants selected at runtime; keep numerical/atomic-preflight contracts. | Bitwise weights and both moments versus scalar C where operation order is unchanged; every invalid update leaves state unchanged; baseline and optimized full decoder quality/time on x86 and native ARM. |
| P1 | **Attention + VJP:** tiled causal GQA attention, an optional provider ABI, measured tensor layouts and forward/backward scheduling. | Independent gradients and numerical-tolerance policy; long-context memory/time curves; matched decoder training at 64/128/256+ context. |
| P1 | **StreamPack output/CRC:** faster IEEE CRC, one final output allocation, profile native size/count/write phases, then native delta splice. | Byte-for-byte complete artifact and refusal parity; immutable prior links/generation safety; timed decode/admission + update + materialization, not only cached native calls. |
| P2 | **ASN.1 array consumers:** grouped bounded integer decode and decode-into/visitor APIs, with generic tail/extension fallbacks. | Oracle octets and values, every truncated prefix, padding and out-of-range failures, capacity/alias tests, ASan/UBSan, all formats the API claims; encode and decode graded separately. |
| P2 | **Matrix library portfolio:** shape/transpose/layout-aware BLAS or native dispatch; SIMD register blocks, persistent packed weights where reuse amortizes packing, improved tiny/irregular fallback kernels. | All four transpose cases, alpha/beta/accumulation, dimensions around tile boundaries, NaN/Inf policy; same floating-point contract and full cost including packing; one and multiple compute threads. |
| P2 | **Vector math + fusion:** optimize RMS forward/VJP, RoPE, SwiGLU forward/VJP and fused cross-entropy; avoid duplicate exponentials and intermediate passes after profiling. | Independent derivative checks and extreme-input stability; declare tolerance changes for vector exp/reductions; measure allocations, bandwidth and full-training quality, not FLOP count alone. |
| P2 | **Training buffer planning:** make live-range plans drive actual buffer reuse and choose recomputation by measured cost; preserve retained activations needed by backward. | No overlapping live tensors, canary/alias and autodiff parity tests, actual peak resident/device memory, execution trace matches admitted plan. |
| P2 | **Mixed precision and larger workloads:** BF16/FP16 computation with an explicit accumulation/master-weight policy, checkpoint/restart and optimizer-state format; then accelerator/distributed integrations. | Held-out loss/accuracy and gradient stability across seeds, overflow/restart equivalence, 10M→larger parameter/context/batch scaling, measured hardware-specific cost models. |
| P3 | **Operator breadth:** add operations driven by the next concrete model (embedding/scatter, convolution or sparse/segment reductions) through a common tensor/gradient ABI. | Operator inventory, forward + VJP + integration tests, bounded work/memory contracts and performance versus an optimized reference; importing wrappers alone does not count as a kernel. |

The earlier 9.84M plan reserved 193.36 MiB total, including 150.1 MiB persistent
parameter/gradient/moment storage and 43.26 MiB transient space. Its current
concurrent-live lower bound is 39.12 MiB: roughly 4.13 MiB (9.55% of transient,
2.14% of total) of measured layout headroom under those semantics, not an unlimited
memory saving. Tiled attention or recomputation changes those semantics and must
be evaluated with a new bound. Avoid copying an arena estimate into a claim of
measured peak memory or accelerator capacity.

GEMplus can price these portfolios and value-based searches. The present native
decoder's forward/backward call order is still implemented in C; `native_program`
describes those calls and buffers, but does not make an arbitrary optimized GEM
schedule execute itself. Wire admitted plans to real dispatch/allocation and hold
that path to an execution trace before attributing training gains to scheduling.

There is no C23 language ceiling demonstrated here. The limiting work is kernel
algorithms, layouts, numerical contracts, calibrated dispatch and integration.
Use the best compatible native algorithms behind the C ABI, and C++ where it
actually simplifies an external framework/accelerator seam.

## 5. Reproduction and validation

The [measurement harness](measurements/2026-10-10/run_performance_tests.py) generates
codecs and compiles the existing native sources into a separate output directory.
Run against the tested production revision (or this docs branch, whose production
files are identical), with Python's assertions enabled and GCC available:

```bash
for stage in gem asn1 kernels crc frontier attention its; do
  python docs/research/measurements/2026-10-10/run_performance_tests.py \
    --stage "$stage" --output "/tmp/bcir-performance/$stage"
done
```

Only `kernels` and `attention` require the experiment's NumPy/SciPy/CPU-PyTorch
environment. The script currently locates SciPy's Linux OpenBLAS shared object;
it is not a portable accelerator or Windows benchmark. Run heavy stages and
repository gates serially as [AGENTS.md](../../AGENTS.md) requires.

Local validation passed: the complete quick oracle at two workers (**4,538
passed, 0 failed**), **22 focused regressions** (collector, native training program,
native hydrate and generated codec parity/refusals), docs governance, the import
quarantine/oracle inventory, pinned ruff format/lint, and `git diff --check`.
Exact commands/verdicts are in [verification.json](measurements/2026-10-10/verification.json)
and the PR verification section.
No timing threshold is added to CI.

Local latest-toolchain check: **unavailable**. `latest_toolchain.py status`
found LLVM/Clang/MLIR 23 absent and Node 24.19.0 behind 24.21.0;
`check_latest.sh` stopped at environment discovery. This is not a toolchain pass.
The repository CI matrix owns available remote toolchain/OS/native ARM validation;
GPU, PMU/energy, native ARM performance and hardware-driver results remain
explicitly unmeasured. Existing CI does not time this new research harness.
