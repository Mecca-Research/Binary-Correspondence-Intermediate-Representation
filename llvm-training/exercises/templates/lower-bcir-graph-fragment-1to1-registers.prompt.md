# Agent template: lower a BCIR graph fragment with 1:1 register correspondence

## Role

You are lowering a Binary Correspondence Intermediate Representation (BCIR)
graph fragment to LLVM IR. Preserve a one-to-one correspondence between every
BCIR logical register and the LLVM SSA value, stack slot, or metadata tag that
represents it.

## Inputs to fill in

- **BCIR fragment**: `<paste graph operations, operands, edge/vertex attributes>`
- **Register map**: `<r0 -> meaning/type, r1 -> meaning/type, ...>`
- **Memory layout**: `<base pointers, element sizes, byte strides, address spaces>`
- **Required target function signature**: `<define ...>`

## Required output

1. A standalone LLVM IR module that assembles with opaque pointers enabled.
2. A register correspondence table in comments or markdown:
   - BCIR register name.
   - LLVM value name.
   - Type and unit (element index or byte offset).
   - Defining instruction.
3. Explicit `getelementptr` or integer byte-offset arithmetic for every graph
   access. Do not hide a BCIR register inside a folded expression if that would
   remove its independently reviewable LLVM value.
4. Metadata attachments or named metadata for graph/attribute facts when they are
   semantically descriptive rather than executable.

## Constraints

- Preserve one defining LLVM value for each BCIR register unless the register is
  proven dead and documented as dead.
- Do not merge two BCIR registers into a single LLVM value merely because their
  numeric values are currently equal.
- Use `ptr` rather than typed pointer syntax.
- Keep address-space conversions explicit and justified.
- State the alignment assumption for every load and store.

## Verification checklist

- `llvm-as -disable-output <candidate.ll>` succeeds.
- `opt -passes=verify <candidate.ll> -o /dev/null` succeeds.
- The correspondence table has no missing BCIR registers and no duplicated LLVM
  value unless the duplicate is explicitly justified.
- Re-running simple canonicalization does not erase required BCIR mapping facts
  without a metadata or side-table replacement.
