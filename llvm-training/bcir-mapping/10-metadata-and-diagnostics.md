# Metadata and Diagnostics

BCIR lowering should preserve enough metadata to explain where a lowered LLVM IR
operation came from without making metadata part of program semantics. Use
metadata for diagnostics, hints, and provenance; use ordinary values, calls, and
memory for behavior that must affect execution.

## BCIR-level meaning

- Claims, graph positions, rules, and source prompts often need to be reported
  after lowering or optimization.
- HAM hints and scheduling profiles are advisory unless the runtime ABI says
  otherwise.
- Diagnostic metadata should survive assembly and verification while remaining
  safe for optimizers to ignore.

## Likely LLVM IR representation

- Attach custom metadata such as `!bcir.diag` or `!bcir.ham` to the GEP, load,
  store, or call most closely associated with the source concept.
- Use `llvm.prefetch` for advisory memory prefetches and keep its `rw`, locality,
  and cache-type arguments as immediate constants.
- Use named metadata or module flags only for module-wide facts.
- Keep semantic values out of metadata; if execution depends on a field, lower it
  as an operand, memory field, or function argument.

## Example source and lowered IR

- HAM hint source prompt: [`examples/ham-hint.prompt.md`](examples/ham-hint.prompt.md)
- Checked prefetch output: [`examples/ham-hint-prefetch.ll`](examples/ham-hint-prefetch.ll)
- Diagnostic source prompt: [`examples/diagnostic-metadata.prompt.md`](examples/diagnostic-metadata.prompt.md)
- Checked diagnostic output: [`examples/diagnostic-metadata-preservation.ll`](examples/diagnostic-metadata-preservation.ll)

## Verifier commands

From the repository root:

```bash
llvm-as llvm-training/bcir-mapping/examples/ham-hint-prefetch.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/ham-hint-prefetch.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/diagnostic-metadata-preservation.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/diagnostic-metadata-preservation.ll -o /dev/null
```

## Verifier risks

- Intrinsic declarations must match the LLVM intrinsic name and type exactly.
- `llvm.prefetch` arguments marked `immarg` must be constants at the call site.
- Metadata attachment syntax is comma-separated after the instruction operands.
- Debug info is stricter than custom metadata; use complete `DI*` graphs when
  emitting real `!dbg` locations.

## Optimization risks

- Unknown custom metadata may be dropped by transforms; do not rely on it for
  semantics.
- Copied debug locations can become stale after graph rewrites.
- Excessive metadata can bloat IR and slow optimization or serialization.
- Advisory prefetches can be ignored, moved, or target-lowered differently from
  the original HAM hint.
