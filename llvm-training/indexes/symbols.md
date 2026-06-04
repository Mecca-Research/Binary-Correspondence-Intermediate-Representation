# Index: By symbol

| Symbol | Means | See |
|---|---|---|
| `%foo`, `%42` | Local (function-scope) identifier | [`01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) |
| `@foo`, `@42` | Global identifier | [`01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) |
| `!N`, `!"str"`, `!{...}` | Metadata | [`01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md), [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md) |
| `#N` | Attribute group ID | [`reference/glossary.md`](../reference/glossary.md) |
| `$foo` | Comdat name | [`reference/glossary.md`](../reference/glossary.md) |
| `i1`, `i8`, `i32`, `i64`, `iN` | Integer of N bits | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) |
| `half`, `bfloat` | 16-bit floating-point formats with different semantics | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md), [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| `token` | Opaque control value for EH/coroutine/statepoint-like IR | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md), [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| `metadata` | Metadata operand type for debug/analysis intrinsics | [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md), [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| `x86_amx` | X86 AMX target extension tile type | [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| `<vscale x N x T>` | Scalable vector type | [`09-vectorization/03-vector-predication.md`](../09-vectorization/03-vector-predication.md), [`09-vectorization/06-recognizing-vector-ir.md`](../09-vectorization/06-recognizing-vector-ir.md), [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| `undef` | Arbitrary value whose separate uses may choose different bit patterns | [`13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md) |
| `poison` | Deferred undefined-behavior value produced by violated IR promises | [`13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md) |
| `freeze` | Instruction that turns `undef` or poison into one stable arbitrary value | [`13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md), [`13-advanced-ir/examples/poison-undef-freeze.ll`](../13-advanced-ir/examples/poison-undef-freeze.ll) |
| `fast` | Fast-math shorthand for aggressive relaxed floating-point semantics | [`13-advanced-ir/06-fast-math-flags.md`](../13-advanced-ir/06-fast-math-flags.md), [`13-advanced-ir/examples/fast-math-flags.ll`](../13-advanced-ir/examples/fast-math-flags.ll) |
| `ptr` | Generic pointer (opaque) | [`02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md), [`02-types/04-opaque-pointer-migration.md`](../02-types/04-opaque-pointer-migration.md), [`02-types/05-opaque-pointer-migration-patterns.md`](../02-types/05-opaque-pointer-migration-patterns.md) |
| `;` | Comment to end of line | [`01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md) |
| `c"..."` | C-style char array constant | [`03-constants/03-strings.md`](../03-constants/03-strings.md) |
| `binary_id`, `input_class`, `trace_id` | Dynamic trace schema fields | [`15-binary-analysis/02-dynamic-traces-and-counters.md`](../15-binary-analysis/02-dynamic-traces-and-counters.md) |
| `opcode_hash`, `cyclomatic_complexity` | Interpretable BCSA feature fields | [`15-binary-analysis/03-interpretable-bcsa-features.md`](../15-binary-analysis/03-interpretable-bcsa-features.md) |
