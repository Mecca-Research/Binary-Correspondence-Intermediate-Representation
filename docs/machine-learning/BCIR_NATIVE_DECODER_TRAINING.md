# Native C decoder training

BCIR now has an opt-in, complete FP32 Llama/SwiGLU decoder trainer. The portable
execution core is C11, also tested as C23. Python prepares token batches, initializes
or imports weights, and saves checkpoints; the multi-step forward/backward/AdamW loop
executes in C. PyTorch remains an independent optional reference.

## Implemented contract

| Component | Native implementation |
|---|---|
| Tensor linear algebra | Blocked row-major GEMM, transpose variants, linear forward, input and accumulated weight VJPs; optional explicit LP64 CBLAS adapter |
| Vector math | Stable sigmoid/SiLU, SwiGLU VJP, RMSNorm and gamma VJP, half-split RoPE and its transpose; standard libm without fast-math |
| Attention | Causal softmax attention, Q/K/V backward, GQA group-gradient accumulation, output projection backward |
| Decoder | Embedding scatter-add backward, both residual paths, both per-layer norms, gate/up/down projections, final norm, tied or untied vocabulary head, mean token cross-entropy |
| Optimizer | AdamW moments, constant-time bias powers, decoupled decay, global L2 clipping, mean micro-batch accumulation, per-update learning rates |
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

## Evidence and scope

`tools/models/test_native_decoder.py` compares native logits, every named gradient,
and multi-step clipped AdamW against the independent hosted PyTorch model. It covers
two-layer tied/GQA, untied/MHA, and context-one cases. The math SDPA reference is used
for gradient trajectories: fused SDPA can leave rounding noise in theoretically zero
context-one Q/K gradients, which Adam's small epsilon amplifies. The timed reference
uses PyTorch's default eager path and `torch.optim.AdamW` without foreach/fused updates.

`runtime/c/test_decoder_train.c` checks finite differences for every parameter,
gradient accumulation, invalid IDs/capacities/aliasing, atomic numerical update
rejection and falling loss in the complete native loop. It runs under C11 and
C23 via `tools/c/sections/decoder_train.sh`; the Python oracle also exercises it.
The hosted CI cells require the tensor differential, including exact native resume
and training-to-Q8-to-C-inference parity.

Timing output records shape, parameter count, provider, optimizer scope and reference
mode. Timings are observations, never CI thresholds. A CPU micro-decoder win over
eager PyTorch is not evidence against compiled PyTorch, large transformers,
accelerators or distributed training.

The plan retains every layer's forward cache and reuses backward scratch; it is not
a globally optimal liveness solver. Attention storage is quadratic in context length.
Flash/blocked attention, activation recomputation, mixed precision, accelerator
backward kernels, distributed collectives and device-resident checkpoints remain
separate work. The direct loop does not issue StreamPack claims for every tensor;
it does not change D1's planned/streamed logistic training contract or assert new
MLIR verifier coverage.
