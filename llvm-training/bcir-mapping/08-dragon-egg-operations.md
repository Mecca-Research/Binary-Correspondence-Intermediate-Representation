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
- Pass all values, pointers, flags, and sizes explicitly.
- Return a status code or value instead of relying on implicit exception-like
  control flow.

## Example source and lowered IR

- Source prompt: [`examples/bcir-operation.prompt.md`](examples/bcir-operation.prompt.md)
- Checked wrapper output: [`examples/bcir-op-runtime-wrapper.ll`](examples/bcir-op-runtime-wrapper.ll)

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
