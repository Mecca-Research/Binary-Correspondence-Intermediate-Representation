# Advanced IR 02 — Target-Specific Intrinsics

## TL;DR

Target-specific intrinsics expose instructions or architectural concepts
that are not available through portable LLVM IR. Their names usually
start with a target namespace such as:

- `llvm.x86.*`
- `llvm.aarch64.*`
- `llvm.arm.*`
- `llvm.riscv.*`
- `llvm.amdgcn.*`
- `llvm.nvvm.*`

They are useful for exact instruction selection, but they make the IR
less portable and often require explicit target features.

## Naming conventions

A target intrinsic name generally has this shape:

```text
llvm.<target>.<family>.<operation>[.<overload suffixes>]
```

Examples you may see in compiler-generated IR:

```llvm
declare <4 x float> @llvm.x86.sse.sqrt.ps(<4 x float>)
declare <2 x i64> @llvm.x86.pclmulqdq(<2 x i64>, <2 x i64>, i8 immarg)
declare i64 @llvm.aarch64.neon.uaddlv.i64.v4i16(<4 x i16>)
```

The exact spelling is not a public assembly mnemonic. It is the name
chosen by LLVM's intrinsic TableGen files.

## Why use one?

Use a target-specific intrinsic when all of these are true:

1. The generic IR operation cannot express the instruction or required
   side effect.
2. Inline assembly would be worse for optimization, register allocation,
   or verification.
3. The module is intentionally tied to a target or has a fallback path.
4. You have checked the required target features.

For many operations, generic IR is better:

- Use `add`, `mul`, `shufflevector`, `fadd`, and vector operations
  instead of directly naming a SIMD instruction when possible.
- Use generic intrinsics such as `llvm.ctpop`, `llvm.fshl`,
  `llvm.memcpy`, or `llvm.vector.reduce.*` when they express the intent.
- Let the backend choose the best instruction for the target CPU.

## Target-feature requirements

A target-specific intrinsic may require a CPU extension such as SSE4.2,
AVX2, AVX-512, AES, PCLMUL, NEON, SVE, AMX, or a GPU subtarget feature.
LLVM may accept the IR syntactically but fail during instruction
selection if the target lacks the feature.

Feature requirements can come from:

- the module target triple and datalayout,
- function attributes such as `target-features` and `target-cpu`,
- command-line options passed to `llc` or `clang`, and
- backend-specific rules in the target's intrinsic definitions.

Example function attribute pattern:

```llvm
define <2 x i64> @carryless_mul(<2 x i64> %a, <2 x i64> %b)
    #0 {
entry:
  %r = call <2 x i64> @llvm.x86.pclmulqdq(<2 x i64> %a,
                                           <2 x i64> %b,
                                           i8 0)
  ret <2 x i64> %r
}

attributes #0 = { "target-features"="+pclmul" }
```

The immediate control byte is `immarg`, so it must be a literal.

## Portability caveats for BCIR

BCIR IR may be assembled, verified, linked, optimized, and lowered in
more than one environment. A target intrinsic can break any of those
assumptions:

- It may only exist for a particular LLVM target backend.
- It may require a target feature not present on the user's CPU.
- It may have backend-specific semantics that are not meaningful in a
  generic verifier or analyzer.
- It may block cross-target reuse of runtime modules.

Recommended pattern:

1. Keep the portable implementation in generic IR.
2. Put target-specialized code in a separate function or module.
3. Gate dispatch outside the specialized function, or select the module
   only when the target is known.
4. Record required `target-features` near the function.
5. Add an assembly/codegen test with the intended `llc -mtriple` and
   feature flags.

## How to locate canonical target signatures

Use the target TableGen files in the LLVM source tree:

| Target | Typical file |
|---|---|
| Generic | `llvm/include/llvm/IR/Intrinsics.td` |
| X86 | `llvm/include/llvm/IR/IntrinsicsX86.td` |
| AArch64 | `llvm/include/llvm/IR/IntrinsicsAArch64.td` |
| ARM | `llvm/include/llvm/IR/IntrinsicsARM.td` |
| RISC-V | `llvm/include/llvm/IR/IntrinsicsRISCV.td` |
| AMDGPU | `llvm/include/llvm/IR/IntrinsicsAMDGPU.td` |
| NVPTX/NVVM | `llvm/include/llvm/IR/IntrinsicsNVVM.td` |

Also check the backend tests under `llvm/test/CodeGen/<Target>/`.
Those tests show which triples and feature flags are expected.

## Pitfalls

- **Using `llvm.x86.*` without x86 codegen.** The IR may parse, but it
  is not portable LLVM IR.
- **Missing target features.** The backend may reject the intrinsic or
  emit a slower/fallback sequence only when such a lowering exists.
- **Wrong immediate argument.** Many target intrinsics have control
  operands marked `immarg`; pass constants, not SSA values.
- **Assuming names match assembly mnemonics.** Intrinsic spellings are
  LLVM API names and can include family names, overloaded suffixes, or
  historical target naming.
- **Skipping a generic alternative.** If generic IR expresses the same
  semantics, prefer it unless you need exact instruction selection.
