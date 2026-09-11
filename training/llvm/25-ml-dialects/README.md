# 25 — The ML dialects

MLIR's largest dialects are the machine-learning ones, and they are the reason the
operation count in
[`../24-mlir-infrastructure/05-the-dialect-landscape.md`](../24-mlir-infrastructure/05-the-dialect-landscape.md)
is two orders of magnitude above LangRef's. `linalg` alone has more operations than LLVM
has instructions.

This subject does **not** tour them. It teaches the two things that make the rest
readable: the ladder an operation descends, and what a quantized type actually is. Both
are taught against a real `mlir-opt`, and every claim below is a registered fixture rather
than a description.

## Chapters

1. [`01-the-abstraction-ladder.md`](01-the-abstraction-ladder.md) —
   `tosa.matmul` to `linalg.batch_matmul` to `linalg.generic` to an `scf.for` nest, one
   pass per rung. What each descent makes explicit, what it destroys, and the joint where
   a pass reports success and changes nothing.
2. [`02-quantization-and-bcir.md`](02-quantization-and-bcir.md) —
   the `quant` dialect's three operations, the two decisions its specification
   deliberately leaves open, what the stock pipeline silently chooses for each, and how
   BCIR's own quantizer decides them differently. Includes the round trip that measures
   exactly zero error.
3. [`03-sparsity-is-a-property-of-the-type.md`](03-sparsity-is-a-property-of-the-type.md) —
   the same `linalg.generic` as chapter 01, with one attribute on one type, becoming a
   completely different loop nest: CSR iteration derived from `#sparse_tensor.encoding`,
   bounds read out of a positions array, and the indexed load that is what sparse costs.
4. [`04-host-and-device.md`](04-host-and-device.md) —
   `gpu.launch` is a region that pretends host and device share a scope.
   `--gpu-kernel-outlining` charges for it: capture becomes an argument list, the loop
   IDs become `gpu.thread_id` reads of hardware, and the boundary becomes structural —
   on any host, with no device and no vendor toolkit.

## What this subject is not

It is not a `linalg` operation reference, and it does not attempt coverage. The corpus's
position on dialect coverage is stated and measured in
[`../24-mlir-infrastructure/05-the-dialect-landscape.md`](../24-mlir-infrastructure/05-the-dialect-landscape.md):
naming more operations is not the goal, and every dialect carries a written disposition in
[`../reference/mlir-dialect-dispositions.json`](../reference/mlir-dialect-dispositions.json)
saying which answer applies to it — and as of slice 1.9 none of the 48 is `planned`.

`gpu` is here rather than in [`../19-hardware-aware/`](../19-hardware-aware/) for a
practical reason worth stating: that subject is about what a backend does with silicon,
and this chapter is about an execution model — the host/device split is a fact about the
IR before any hardware is named. When `training/hardware/` opens it owns the silicon; this
owns the boundary.

## Where this connects

- **Down**: [`../18-mlir-lowering-to-llvm/`](../18-mlir-lowering-to-llvm/) — the conversion
  framework these pipelines are built from, destination-passing style and bufferization in
  `10-`, and shape computation in `11-`.
- **Sideways**: [`../23-version-movement/03-conversions-and-shifts.md`](../23-version-movement/03-conversions-and-shifts.md)
  — `fptosi` is poison out of range, which is what chapter 02's missing clamp runs into.
- **Up**: [`../22-bcir-approach/`](../22-bcir-approach/) — legality before cost, and the
  twelve cost axes. Chapter 01 ends at the level where a plan is still legible enough to
  price.
