# Native C decoder training

BCIR now has an opt-in, complete FP32 Llama/SwiGLU decoder trainer. The portable
execution core is C11, also tested as C23. Python prepares token batches, initializes
or imports weights, and saves checkpoints; the multi-step forward/backward/AdamW loop
executes in C. PyTorch remains an independent optional reference.

## Implemented contract

| Component | Native implementation |
|---|---|
| Tensor linear algebra | Register-blocked row-major GEMM: a KCxNR tile of B packed once per (column, depth) block and reused by every row, an MRxNR block of C kept in registers across the tile's depth; bit-identical to the ascending-k loop on every input. Portable C everywhere, plus AVX2 and AVX-512 variants on x86-64 Linux (GCC or Clang) chosen per call from the CPU, all bit-identical (`bcir_tensor_mm_variant` runs one). Transpose variants, linear forward, input and accumulated weight VJPs; optional explicit LP64 CBLAS adapter |
| Vector math | Stable sigmoid/SiLU, SwiGLU VJP, RMSNorm and gamma VJP, half-split RoPE and its transpose; standard libm without fast-math |
| Attention | Causal softmax attention as two masked products per (batch, head) through the GEMM kernel; a backward that rewrites the layer's probabilities in place with their score adjoint and feeds the GEMM kernel (no workspace beyond them); Q/K/V backward, GQA group-gradient accumulation, output projection backward. Both are bit-identical to the scalar reference loops, which `bcir_tensor_attention_backward` keeps |
| Decoder | Embedding scatter-add backward, both residual paths, both per-layer norms, gate/up/down projections, final norm, tied or untied vocabulary head, mean token cross-entropy |
| Optimizer | AdamW moments, constant-time bias powers, decoupled decay, global L2 clipping, mean micro-batch accumulation, per-update learning rates. The atomic preflight is skipped only when the magnitudes the first pass records prove it would admit every element; the passes carry the same AVX2/AVX-512 variants and stay bit-identical |
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
provider and thread configuration. Supply the same optimizer options and subsequent
learning-rate schedule on resume; these are caller inputs rather than checkpoint
fields. Checkpoint hashes detect corruption and do not authenticate an artifact.

An explicit provider is selected with
`trainer.use_cblas(library_path, symbol="cblas_sgemm")`. Configure its thread count
before measuring. Missing providers fail visibly. There is no automatic PyTorch
fallback. The C ABI accepts native GEMM and SiLU function pointers: C++ belongs in a
framework/vendor/SYCL adapter when that integration requires it. The core contains
no C++ objects, accelerator discovery or framework runtime.

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
| `train.step.ms` | wall, floor in the same run | one native step of the bench decoder against G26's roofline compute term: the counted work at the best rate the same kernel reaches on any of the step's own GEMM shapes | 6.9 ms against a 2.8 ms floor measured beside it (the parent: 10.0 ms against the same 2.8 ms) |

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
