# Agent template: preserve metadata through an LLVM pass

## Role

You are modifying or reviewing a pass that replaces, folds, clones, outlines, or
deletes LLVM instructions. Preserve required debug, optimization, aliasing, and
BCIR correspondence metadata, or reify equivalent evidence in a documented side
table.

## Inputs to fill in

- **Input IR**: `<module with required metadata>`
- **Pass/pipeline**: `<exact pass command or implementation>`
- **Required metadata policy**: `<kind -> transfer, merge, side-table, or drop>`
- **BCIR mapping contract**: `<stable IDs and 1:1 requirements>`
- **Expected output shape**: `<replacement, fold, clone, outline, deletion>`

## Preservation procedure

1. Inventory instruction attachments, debug records, named metadata, operand
   bundles, and stable BCIR IDs before the pass.
2. For every rewritten instruction, decide whether each item transfers, merges,
   moves to named metadata, or is deliberately dropped with a proof.
3. Reject stale debug locations that claim a new operation is the removed source
   operation; preservation must remain truthful.
4. Run the exact pass and compare the semantic inventory, not just verifier
   success or textual instruction count.
5. Add focused checks for required metadata and BCIR correspondence after the
   pass and after any subsequent canonicalization that can erase values.

## Required output

- A metadata preservation policy table.
- Before/after IR excerpts.
- Explanation for every dropped or rewritten metadata record.
- A stable regression check for attachments and named side tables.
- A BCIR mapping verdict: `preserved`, `reified`, or `drifted`.

## Verification checklist

- Input and output intended as valid IR pass `llvm-as` and
  `opt -passes=verify`.
- Required metadata is checked after the actual transform, not only a verifier
  round trip.
- Deleted instructions leave equivalent side-table evidence when BCIR
  reconstruction still requires their identities.
- Debug information remains accurate rather than merely present.
- Tests fail if the pass silently drops required metadata while producing
  otherwise verifier-valid IR.
