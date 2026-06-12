# Remote JITLink and heterogeneous deployment sketch

This design keeps executor negotiation, artifact production, publication, and
invocation separate.

```text
TargetDescriptor = negotiate(executor):
  id, triple, data-layout, endianness
  pointer-widths-by-address-space
  cpu/features or Wasm/RISC-V extensions
  runtime-ABI-version, telemetry-version
  artifact-kinds = {native-object, wasm-module, fpga-package, riscv-object}

plan = selectDeployment(BCIRGraph, TargetDescriptor)

switch plan.kind:
  native-object:
    LLVM IR -> target object -> remote ObjectLinkingLayer/JITLink
    publish executor address as an opaque ExecutorAddr

  wasm-module:
    LLVM/MLIR sandbox lowering -> Wasm module
    upload module; publish typed export handle
    pass linear-memory offsets, never host pointers

  fpga-package:
    accelerator lowering -> bitstream + manifest
    native host shim -> JITLink
    shim calls runtime queue/buffer/device-handle ABI

  riscv-object:
    LLVM IR with negotiated ISA features -> RISC-V object
    remote JITLink allocates/fixes up on the RISC-V executor
    publish opaque remote entry handle
```

A deployment state machine should look like:

```text
accepted -> lowered -> compiled -> uploaded -> linked/loaded
         -> validated -> published -> active -> draining -> removed
                         \-> failed (diagnostic + cleanup)
```

Required invariants:

- The module data layout comes from `TargetDescriptor`, not the host process.
- Executor addresses are serialized handles and are never dereferenced locally.
- Each protocol request carries graph ID, generation, target descriptor ID, and
  an idempotency key.
- "linked/loaded" is not "validated": BCIR semantics, conversion legality,
  runtime ABI, and target capability checks gate publication.
- Resource removal tears down remote allocations, debug/unwind registrations,
  device artifacts, and telemetry address maps for the same generation.
