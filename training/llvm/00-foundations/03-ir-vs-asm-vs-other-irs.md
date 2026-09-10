# LLVM IR vs Assembly and Other IRs

## TL;DR

LLVM IR is **higher than assembly** (no registers, no machine layout,
typed values) and **higher than most other IRs** (SSA from the start,
stronger types). It's the right level for cross-target optimization;
it is the wrong level for cycle-accurate scheduling or final
codegen — that's `llc`'s job below.

## vs Assembly

The same C function shown three ways:

```c
int add(int a, int b) { return a + b; }
```

x86-64 assembly (System V ABI):
```asm
add:
  leal (%rdi, %rsi), %eax
  ret
```

LLVM IR:
```llvm
define i32 @add(i32 %a, i32 %b) {
  %r = add i32 %a, %b
  ret i32 %r
}
```

| | Assembly | LLVM IR |
|---|---|---|
| Has physical registers? | Yes (`%rdi`, `%rax`) | No (`%a`, `%b` are SSA values) |
| Has stack frame setup? | Yes (implicit, ABI-driven) | No (`alloca` is abstract) |
| Architecture-specific? | Always | Almost never (target triple is a hint) |
| Verifiable by tool? | Mostly no | Yes (`opt -passes=verify`) |
| Optimizable by `opt`? | No | Yes |
| Calls follow target ABI? | Yes (literally) | Yes (declared, ABI applied at codegen) |

Rule of thumb: if your question involves a register name or a stack
offset, you're below LLVM IR. If it involves an SSA name or a typed
operand, you're at IR.

## vs other IRs

Quick comparison of LLVM IR with the IRs you're most likely to
encounter:

### GIMPLE (GCC)

- Tree-based; LLVM IR is graph-based.
- Not in SSA by default — GCC converts to and from SSA for specific
  passes (called *SSA mode*).
- More tightly coupled to C semantics than LLVM IR.
- Use it if: you're working in the GCC tree.
- Use LLVM IR if: you want a portable optimization target.

### Java bytecode (JVM)

- **Stack-based** — operations push/pop an evaluation stack. LLVM IR
  is register/value-based.
- Type system tied to the JVM (object references, primitive types,
  array types).
- Designed for portability + sandboxing, not for cross-target
  optimization.
- Use it if: targeting the JVM.

### CIL (.NET Common Intermediate Language)

- Stack-based like Java bytecode, but richer type system (value
  types, generics).
- Tied to the .NET runtime.
- Use it if: targeting .NET.

### SPIR-V

- Binary IR for GPUs (Vulkan, OpenCL).
- SSA-based, much closer in spirit to LLVM IR than to JVM/CIL.
- Has GPU-specific concepts (execution scopes, decorations, capability
  declarations).
- Use it if: shipping a GPU compute kernel.
- Use LLVM IR if: doing CPU-side host code, or using LLVM's SPIR-V
  backend.

### MLIR

- Built **on top of** LLVM IR. Not a competitor — a framework that
  lets you define *your own* dialect that eventually lowers to LLVM
  IR.
- The right answer when you have a domain-specific IR (e.g., BCIR)
  and want to plug into LLVM's optimization pipeline.

## Summary table

| IR | SSA? | Stack vs Register | Typed? | Portable? | Notes |
|---|---|---|---|---|---|
| LLVM IR | yes | register/value | strongly | yes | This repo |
| Assembly | n/a | register + memory | no | no | Per-arch |
| GIMPLE | optional | register-like | yes | yes | GCC internal |
| Java bytecode | no | stack | yes (JVM types) | yes (JVM) | Designed for sandbox |
| CIL | no | stack | yes (.NET types) | yes (.NET) | Designed for sandbox |
| SPIR-V | yes | register | yes | yes (Vulkan/OpenCL) | GPU-focused |
| MLIR dialects | typically yes | typically register | strongly | yes | Multi-level, extensible |

## When to choose LLVM IR

✅ Multiple source languages, single backend infrastructure
✅ Multiple target CPUs/GPUs, single optimization pass library
✅ Need a stable, well-tooled IR with a verifier and JIT
✅ Want compatibility with LLVM passes (`opt`)

## When NOT to choose LLVM IR

❌ Need cycle-accurate scheduling — go below LLVM IR into `llc`
   target-specific code
❌ Need a domain-specific representation (tensors, dataflow, dialects
   with custom verifiers) — start with MLIR; lower to LLVM IR at the
   end of the pipeline
❌ Need a sandbox + portable bytecode runtime — JVM/CIL/Wasm are
   better-suited than raw LLVM IR

## See also

- [`01-what-is-llvm-ir.md`](01-what-is-llvm-ir.md)
- [`../01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) — practical IR structure
