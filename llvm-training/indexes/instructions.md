# Index: By instruction (most common)

| Instruction | Read |
|---|---|
| `add`, `sub`, `mul`, `sdiv`, `udiv`, `srem`, `urem` | [`01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md), [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg` | [`01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md), [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `and`, `or`, `xor`, `shl`, `lshr`, `ashr` | [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `alloca` | [`04-memory/01-alloca.md`](../04-memory/01-alloca.md) |
| `load`, `store` | [`04-memory/02-load-store.md`](../04-memory/02-load-store.md), [`02-types/05-opaque-pointer-migration-patterns.md`](../02-types/05-opaque-pointer-migration-patterns.md) |
| `getelementptr` (GEP) | [`02-types/02-composite-types.md`](../02-types/02-composite-types.md), [`02-types/05-opaque-pointer-migration-patterns.md`](../02-types/05-opaque-pointer-migration-patterns.md), [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `br`, `switch`, `indirectbr`, `ret`, `unreachable` | `05-control-flow/` (all four files) |
| `phi` | [`00-foundations/02-ssa.md`](../00-foundations/02-ssa.md) |
| `icmp`, `fcmp` | [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `select` | [`reference/instruction-quickref.md`](../reference/instruction-quickref.md), [`13-advanced-ir/05-poison-undef-freeze.md#bcir-safe-speculation-with-freeze`](../13-advanced-ir/05-poison-undef-freeze.md#bcir-safe-speculation-with-freeze) |
| `freeze` | [`13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md), [`13-advanced-ir/05-poison-undef-freeze.md#bcir-safe-speculation-with-freeze`](../13-advanced-ir/05-poison-undef-freeze.md#bcir-safe-speculation-with-freeze), [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `call`, `invoke`, `callbr` | [`reference/instruction-quickref.md`](../reference/instruction-quickref.md), [`01-syntax/04-inline-asm.md`](../01-syntax/04-inline-asm.md), [`16-exception-handling/02-itanium-landingpad.md`](../16-exception-handling/02-itanium-landingpad.md), [`bcir-mapping/09-runtime-call-boundaries.md`](../bcir-mapping/09-runtime-call-boundaries.md) |
| `atomicrmw`, `cmpxchg`, `fence` | [`11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md), [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `extractvalue`, `insertvalue`, `extractelement`, `insertelement`, `shufflevector` | [`reference/instruction-quickref.md`](../reference/instruction-quickref.md), [`09-vectorization/06-recognizing-vector-ir.md`](../09-vectorization/06-recognizing-vector-ir.md) |
| `trunc`, `zext`, `sext`, `fptrunc`, `fpext`, `fptoui`, `fptosi`, `uitofp`, `sitofp` | [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `bitcast`, `addrspacecast`, `inttoptr`, `ptrtoint` | [`02-types/06-opaque-pointer-migration-diagnostics.md`](../02-types/06-opaque-pointer-migration-diagnostics.md), [`04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md), [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `landingpad`, `resume` | [`16-exception-handling/02-itanium-landingpad.md`](../16-exception-handling/02-itanium-landingpad.md), [`16-exception-handling/04-cleanups-and-resume.md`](../16-exception-handling/04-cleanups-and-resume.md), [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `catchswitch`, `catchpad`, `cleanuppad`, `catchret`, `cleanupret` | [`16-exception-handling/03-wineh-funclets.md`](../16-exception-handling/03-wineh-funclets.md), [`16-exception-handling/04-cleanups-and-resume.md`](../16-exception-handling/04-cleanups-and-resume.md), [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| intrinsic calls (`llvm.prefetch`, `llvm.memcpy.*`, `llvm.bcir.*`) | [`reference/intrinsics-quickref.md`](../reference/intrinsics-quickref.md), [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md), [`12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md) |
