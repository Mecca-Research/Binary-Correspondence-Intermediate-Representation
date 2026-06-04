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
| `<vscale x N x T>` | Scalable vector type | [`09-vectorization/README.md`](../09-vectorization/README.md), [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| `ptr` | Generic pointer (opaque) | [`02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md), [`02-types/04-opaque-pointer-migration.md`](../02-types/04-opaque-pointer-migration.md) |
| `;` | Comment to end of line | [`01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md) |
| `c"..."` | C-style char array constant | [`03-constants/03-strings.md`](../03-constants/03-strings.md) |
