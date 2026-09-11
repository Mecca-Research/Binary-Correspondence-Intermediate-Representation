# 04 — Host and device, and where the boundary becomes real

`gpu.launch` lets you write device code inside a host function, reading host values that
happen to be in scope. That is a convenient notation and a false picture of the machine:
nothing is in scope across that boundary. This chapter is about the pass that charges for
it, and it runs on any host — no device, no driver, no vendor toolkit.

## The lie, written down

[`examples/gpu-launch-outlining.mlir`](examples/gpu-launch-outlining.mlir) is a saxpy. The
kernel body reads `%a`, `%x` and `%y`, which are arguments of the enclosing `func.func`,
and `%tx`, which is a block argument of the launch region:

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
           threads(%tx, %ty, %tz) in (%bxs = %c256, %bys = %c1, %bzs = %c1) {
  %xv = memref.load %x[%tx] : memref<256xf32>
  ...
  gpu.terminator
}
```

Read as ordinary MLIR, that is a region closing over its environment. Read as a program, it
is two address spaces on two processors pretending to be one scope.

## `--gpu-kernel-outlining` charges for it

```mlir
module attributes {gpu.container_module} {
  func.func @saxpy(%arg0: f32, %arg1: memref<256xf32>, %arg2: memref<256xf32>) {
    gpu.launch_func @saxpy_kernel::@saxpy_kernel
      blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)
      args(%arg1 : memref<256xf32>, %arg2 : memref<256xf32>, %arg0 : f32)
    return
  }
  gpu.module @saxpy_kernel {
    gpu.func @saxpy_kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>, %arg2: f32)
        kernel attributes {known_block_size = array<i32: 256, 1, 1>, ...} {
      %thread_id_x = gpu.thread_id x
      ...
      gpu.return
    }
  }
}
```

Four things happened, and each is a fact about GPUs rather than about this pass.

**Capture became an argument list.** The three values the region read are now three kernel
parameters, passed explicitly at the launch. There is no other way for them to arrive.
Every capture in a `gpu.launch` is a value that will be copied to the device or a pointer
that must already be valid there — so a kernel that "just reads" a host variable is a
kernel with one more argument, and the convenience of the region form is exactly what hides
that from you.

**The order is the compiler's, not yours.** The source reads `%a` first; the kernel takes
the two memrefs and then the scalar. Nothing depends on the textual order of a capture.

**The IDs stopped being values and became reads.** On the host side `%tx` was a block
argument. In the kernel it is `gpu.thread_id x` — an operation, executed on the device,
that reads a hardware register. Those two things look alike in the region form and are not
alike at all: one is dataflow, the other is a per-thread read that is the *only* reason the
256 threads do different work.

**The launch geometry became an attribute.** `known_block_size` and `known_grid_size` are
propagated from the constants at the launch site, so the kernel carries what the compiler
was able to prove about how it will be invoked. Launch it with a different shape later and
that promise is a lie nobody checked.

## One dialect, then two vendors

Everything above is target-independent. `gpu.module` is where the split stops being a
question about your program and starts being a question about someone's hardware:
`--convert-gpu-to-nvvm`, `--convert-gpu-to-rocdl` and `--convert-gpu-to-spirv` each take it
somewhere different, and `--gpu-module-to-binary` finishes the job with a toolkit installed.

That staging is the value of the dialect. The host/device boundary, the argument marshalling
and the ID reads are decided once, portably, and are already correct before any vendor
appears — which is the same layering
[`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md)
describes one level down, where a portable operation survives longer than the intrinsic it
eventually becomes. This corpus stops at the portable step: it is the part that is true on
every machine, and the only part this one can check.

## Pitfall: `gpu.launch_func` contains `gpu.launch`

The fixture's gate forbids **`gpu.launch blocks`**, not `gpu.launch`. Forbidding the short
form would fire on the very operation the pass is supposed to produce. It is the same trap
as the provenance attribute in
[`03-sparsity-is-a-property-of-the-type.md`](03-sparsity-is-a-property-of-the-type.md), and
the same rule: a substring is not a match.

## Pitfalls checklist

- Do not read a `gpu.launch` region as sharing scope with its function. It shares *syntax*;
  every value crossing that line is marshalled.
- Do not count on capture order. The outliner chooses the signature.
- Do not treat `gpu.thread_id` as a loop index. It is a read that returns something
  different in every thread, and it is the whole mechanism.
- Do not launch a kernel with a geometry that contradicts its `known_block_size`. The
  attribute records what the compiler assumed, and nothing re-checks it later.
- Do not reach for `--convert-gpu-to-nvvm` to find out whether outlining worked. Outlining
  is target-independent and observable on its own; mixing the two makes a portable failure
  look like a missing toolkit.

## Checks

[`examples/gpu-launch-outlining.mlir`](examples/gpu-launch-outlining.mlir) is a Tier 3
fixture in [`../autograder/mlir-examples.json`](../autograder/mlir-examples.json). The
source must contain `gpu.launch blocks` and `gpu.terminator`; the output must contain
`gpu.container_module`, `gpu.module`, `gpu.func`, `gpu.launch_func`, `gpu.thread_id` and
`gpu.return`, and must contain neither `gpu.launch blocks` nor `gpu.terminator`.

Requiring `gpu.thread_id` in the output is deliberate and is the check most likely to be
left out: a pass could move the body into a `gpu.module` and leave the region's block
arguments dangling, which would produce every other required string and no working kernel.
