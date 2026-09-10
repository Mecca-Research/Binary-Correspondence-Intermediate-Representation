# Agent template: review mixed-stride BCIR lowering

## Role

You are reviewing a lowering that mixes element strides, row strides, byte
strides, and graph-edge strides. Identify correctness risks before the IR is
accepted as a BCIR-preserving lowering.

## Inputs to fill in

- **BCIR indexing formula**: `<for example base + row * row_stride_bytes + col * elem_size>`
- **Lowered LLVM IR**: `<paste candidate function>`
- **Declared layout facts**: `<element type, ABI size, packed/struct layout, address space>`
- **Expected register mapping**: `<BCIR register -> LLVM value>`

## Review tasks

1. Classify every stride as an element count, byte count, or structure-field
   index.
2. Check whether each `getelementptr` source element type matches the intended
   layout assumption under opaque pointers.
3. Verify that integer offset arithmetic cannot overflow in a way that changes
   BCIR semantics, or document the needed `nuw`/`nsw` avoidance.
4. Confirm that alignment annotations are no stronger than the source layout
   guarantees.
5. Confirm that address spaces are preserved and that any `addrspacecast` is
   legal for the target memory model.
6. Check that optimization cannot merge away distinct BCIR registers needed for
   one-to-one correspondence.

## Required output

- A verdict: `accept`, `accept with conditions`, or `reject`.
- A table of each access path with base, offset register, stride units, GEP source
  element type, alignment, and address space.
- A list of required fixes or invariants to add to the prompt, IR, verifier, or
  metadata.

## Verification checklist

- Candidate LLVM IR assembles and verifies if it is meant to be executable.
- Semantic review explicitly covers mixed byte-vs-element stride conversions.
- Opaque-pointer assumptions are not inferred from the pointer type alone.
