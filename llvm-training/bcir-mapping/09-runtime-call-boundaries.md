# Runtime Call Boundaries

Runtime call boundaries are the safest way to lower BCIR operations whose
implementation belongs outside a standalone LLVM IR example. The IR around the
boundary should make memory, values, flags, and diagnostics visible so the call
site is reviewable and verifiable.

## BCIR-level meaning

- A runtime boundary separates claim decoding from runtime-owned execution.
- The caller owns resource lookup, argument preparation, and result/status use.
- The callee owns target-specific execution, opaque state, or library calls.
- The ABI must stay stable across generated modules, JIT modules, and runtime
  object files.

## Boundary checklist

- Declare the runtime function once with the exact return and argument types.
- Keep wrapper definitions small and named by operation.
- Load inputs and resource bases before the call; store results after the call.
- Preserve status returns instead of dropping them unless the ABI explicitly says
  the operation cannot fail.
- Attach diagnostic metadata to the call or nearby address calculation when the
  runtime needs source correlation.

## Example source and lowered IR

- Operation source prompt: [`examples/bcir-operation.prompt.md`](examples/bcir-operation.prompt.md)
- Checked runtime wrapper: [`examples/bcir-op-runtime-wrapper.ll`](examples/bcir-op-runtime-wrapper.ll)
- Resource lookup source: [`examples/claim-resource-lookup.bcir.txt`](examples/claim-resource-lookup.bcir.txt)
- Checked resource lookup: [`examples/claim-resource-lookup.ll`](examples/claim-resource-lookup.ll)

## Verifier commands

From the repository root:

```bash
llvm-as llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/claim-resource-lookup.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/claim-resource-lookup.ll -o /dev/null
```

## Verifier risks

- A call's operand list must exactly match the declared function type.
- Duplicate runtime helper definitions collide when examples are linked.
- Address-space mismatches must be handled with explicit `addrspacecast` only
  when the target ABI permits it.
- Attribute mismatches between declarations and definitions can become undefined
  behavior even when assembly succeeds.

## Optimization risks

- Call boundaries constrain optimization; keep only necessary operations behind
  the boundary.
- Missing memory-effect modeling can either block optimization or allow unsound
  motion around the call.
- JIT symbol resolution depends on exact symbol spelling and linkage.
