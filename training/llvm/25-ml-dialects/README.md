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

## What this subject is not

It is not a `linalg` operation reference, and it does not attempt coverage. The corpus's
position on dialect coverage is stated and measured in
[`../24-mlir-infrastructure/05-the-dialect-landscape.md`](../24-mlir-infrastructure/05-the-dialect-landscape.md):
naming more operations is not the goal, and every dialect carries a written disposition in
[`../reference/mlir-dialect-dispositions.json`](../reference/mlir-dialect-dispositions.json)
saying which answer applies to it. `sparse_tensor` and `gpu` are `planned` there, with the
slice that will close them, rather than half-covered here.

## Where this connects

- **Down**: [`../18-mlir-lowering-to-llvm/`](../18-mlir-lowering-to-llvm/) — the conversion
  framework these pipelines are built from, destination-passing style and bufferization in
  `10-`, and shape computation in `11-`.
- **Sideways**: [`../23-version-movement/03-conversions-and-shifts.md`](../23-version-movement/03-conversions-and-shifts.md)
  — `fptosi` is poison out of range, which is what chapter 02's missing clamp runs into.
- **Up**: [`../22-bcir-approach/`](../22-bcir-approach/) — legality before cost, and the
  twelve cost axes. Chapter 01 ends at the level where a plan is still legible enough to
  price.
