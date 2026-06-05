# Syntax: Modules, Functions, and Instructions

## Key takeaways

- Read LLVM IR top down: module metadata and target info first, then globals, function signatures, basic blocks, and terminators.
- Every instruction result has an explicit type, and opaque pointers move pointee types onto memory, call, and GEP operations.
- Basic blocks must end with exactly one terminator; labels and predecessor lists are structural, not decorative.
- Comments are ignored by the parser, while metadata attachments can affect debug info, optimization, and diagnostics.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Module anatomy, declarations, definitions, functions, and basic blocks | [`01-modules-functions-blocks.md`](01-modules-functions-blocks.md) |
| Instruction spelling, operands, result names, and terminator format | [`02-instruction-format.md`](02-instruction-format.md) |
| Comments, metadata attachments, module flags, and target metadata | [`03-comments-metadata.md`](03-comments-metadata.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
