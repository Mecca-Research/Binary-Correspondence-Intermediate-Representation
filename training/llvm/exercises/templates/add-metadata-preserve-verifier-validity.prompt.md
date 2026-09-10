# Agent template: add BCIR metadata while preserving verifier validity

## Role

You are augmenting existing LLVM IR with BCIR metadata. Your job is to add graph,
register, provenance, or lowering metadata without changing executable semantics
and without creating IR that fails LLVM verification.

## Inputs to fill in

- **Original IR**: `<paste module or function>`
- **Metadata requirements**: `<graph ID, register IDs, edge/vertex attributes, transform IDs>`
- **Instructions that need attachments**: `<value names or operation descriptions>`
- **Metadata that must survive later passes**: `<facts needed after optimization>`

## Required output

1. Updated LLVM IR with metadata attached only to legal attachment sites.
2. Named metadata records for module-level catalogs when useful.
3. A short explanation of why each metadata node is descriptive, required for
   BCIR reconstruction, or safe to drop.
4. A preservation plan for metadata that an optimizer might otherwise drop.

## Constraints

- Do not attach metadata kinds that have target-specific or LLVM-defined meaning
  unless their required schema is satisfied.
- Prefer custom metadata names for BCIR facts, for example `!bcir.reg`,
  `!bcir.graph`, `!bcir.edge`, or `!bcir.transform`.
- Keep debug metadata internally consistent if you edit `!dbg` locations.
- If metadata loss would break BCIR reconstruction, add a named metadata catalog
  or side-table entry rather than relying only on an instruction attachment.

## Verification checklist

- `llvm-as -disable-output <candidate.ll>` succeeds.
- `opt -passes=verify <candidate.ll> -o /dev/null` succeeds.
- Every custom metadata node has a consumer-facing meaning documented in comments
  or the answer text.
- A transform that deletes an instruction does not delete the only copy of a
  required BCIR mapping fact.
