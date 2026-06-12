# Dragon Egg Operation Lowering

Dragon Egg operations are BCIR operations that intentionally cross the boundary
from portable claim semantics into target/runtime-owned behavior. In LLVM IR,
model that boundary as a narrow wrapper: explicit arguments in, explicit status
or value out, and no hidden dependence on compiler-local state.

## BCIR-level meaning

- The operation may be target-specific, runtime-specific, or too large to inline
  as portable LLVM IR.
- The claim still owns resource IDs, hazard-domain rules, and diagnostics.
- The runtime call owns implementation details such as target dispatch, library
  selection, or opaque accelerator queues.

## Likely LLVM IR representation

- Decode claim fields and resource bases before the runtime call when those
  fields are part of the public ABI.
- Use a declared function for the runtime operation and a small defined wrapper
  for the BCIR-facing name.
- If a backend owns the operation directly, use a declared custom intrinsic
  shape only after documenting its TableGen signature, operand immediates, and
  fallback ABI.
- Pass all values, pointers, flags, and sizes explicitly.
- Return a status code or value instead of relying on implicit exception-like
  control flow.

## Custom intrinsic versus runtime call

Dragon Egg operations often start as runtime calls because calls are portable,
linkable, and easy for ORC JIT layers to interpose. A custom LLVM intrinsic is a
better fit only when the backend must see a first-class operation during
instruction selection, register-bank selection, or machine scheduling.

Use this decision rule:

- **Runtime call:** the operation can be implemented by a library, accelerator
  queue, or late JIT symbol without changing target instruction selection.
- **Custom intrinsic:** the operation needs target legalization, register-class
  constraints, immediate operands, or pseudo-instruction selection before normal
  call lowering would erase the operation shape.

When adding a custom intrinsic, keep a fallback lowering path. The same BCIR GEM
operation can lower to `@llvm.bcir.gem.mixed.stride.v4f32` for a BCIR-aware
backend and to `@bcir.runtime.gem.v4f32` for a generic ORC/JIT environment. The
examples in [`../12-backend-jit/examples/custom-bcir-intrinsic-jit.ll`](../12-backend-jit/examples/custom-bcir-intrinsic-jit.ll)
and [`examples/hardware-aware-gem-lowering.ll`](examples/hardware-aware-gem-lowering.ll)
show both boundary shapes.

## Example source and lowered IR

- Source prompt: [`examples/bcir-operation.prompt.md`](examples/bcir-operation.prompt.md)
- Checked wrapper output: [`examples/bcir-op-runtime-wrapper.ll`](examples/bcir-op-runtime-wrapper.ll)
- Hardware-aware GEM intrinsic sketch: [`examples/hardware-aware-gem-lowering.ll`](examples/hardware-aware-gem-lowering.ll)

## Verifier commands

From the repository root:

```bash
llvm-as llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null
```

## Verifier risks

- Declaration/call signature drift is a verifier or linker failure.
- `immarg` restrictions apply to LLVM intrinsics, but custom runtime calls still
  need stable ABI documentation for flag positions.
- Pointer attributes such as `nocapture` must be compatible with the runtime
  function's actual behavior.

## Optimization risks

- Marking a runtime call too pure can let optimizers remove required effects.
- Marking every wrapper as opaque can prevent useful scalar cleanup around the
  boundary.
- Wrapper names should remain stable enough for profile, trace, and diagnostic
  correlation.

## Hardware-aware continuation

For the Dragon Egg operation taxonomy, target-extension lowering, and the runtime/intrinsic/MIR decision boundary, continue with [`../19-hardware-aware/README.md`](../19-hardware-aware/README.md).
