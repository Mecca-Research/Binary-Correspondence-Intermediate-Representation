# Quickref: Advanced IR

Use this sheet when an LLVM module already verifies but contains advanced
contracts that can silently change optimization, ABI lowering, or backend
selection.

## First lookup path

| If you see | Check first | Why |
| --- | --- | --- |
| `declare @llvm.*` or intrinsic-heavy calls | [`../reference/intrinsics-quickref.md`](../reference/intrinsics-quickref.md), then [`../13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md) | Intrinsics encode exact type, overload, attribute, and `immarg` contracts. |
| `llvm.x86.*`, target-feature-dependent names, or custom `llvm.bcir.*` names | [`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md), [`../12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md) | Target intrinsics and custom backend intrinsics are not portable IR by spelling alone. |
| `token`, `metadata`, target-extension types, or scalable vectors | [`../13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) | These types often imply EH, GC/statepoint, debug, scalable-vector, or backend constraints. |
| Function/call-site attributes such as `noundef`, `noalias`, `readonly`, `memory(read)`, `sret`, or `byval` | [`../13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) | Attributes are optimizer-visible promises; copying unproven attributes miscompiles code. |
| `undef`, poison-producing flags, `freeze`, or verifier-valid unsafe speculation | [`../13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md) | Valid IR can still be undefined when poison reaches control flow, memory, or calls. |
| Fast-math flags (`fast`, `nnan`, `ninf`, `nsz`, `arcp`, `contract`, `afn`, `reassoc`) | [`../13-advanced-ir/06-fast-math-flags.md`](../13-advanced-ir/06-fast-math-flags.md) | These flags relax IEEE behavior and can change vectorization/reassociation legality. |

## Review checklist

1. Reconstruct the source-level contract before preserving an intrinsic,
   attribute, no-wrap flag, or fast-math flag.
2. Confirm every intrinsic declaration exactly matches the called overloaded
   name and any `immarg` operands are constants.
3. Treat metadata and debug info as diagnostics unless a LangRef-defined
   metadata contract explicitly affects optimization.
4. Insert `freeze` before a possibly poison value controls branching, selecting
   externally visible values, memory addresses, or calls.
5. For BCIR lowering, keep domain facts in structured records or metadata until
   the LLVM operation is plain, verifiable, and ABI-explicit.

## Verification commands

```bash
./llvm-training/tools/verify-examples.sh
./llvm-training/tools/verify-exercises.sh
./llvm-training/tools/verify-invalid-fixtures.sh
```

Run the BCIR- or MLIR-specific verifiers as well when advanced IR is produced
from those frontends:

```bash
./llvm-training/tools/verify-bcir-mapping.sh
./llvm-training/tools/verify-mlir-examples.sh
```
