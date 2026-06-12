# Exercise 038: Design a custom pass for BCIR invariants

## Task family

This is an **advanced custom-pass review** exercise. Design the checks for an
LLVM pass that validates BCIR lowering invariants after graph, register-binding,
and hierarchical-memory operations have been translated to ordinary LLVM IR plus
metadata. Use the formal [BCIR normal-form contract](../bcir-mapping/11-normal-forms-and-verification.md) as the specification under review.

## Scenario

A project has lowered BCIR operations into LLVM IR. The pass should not
reconstruct the full source IR, but it must reject lowered modules that are
unsafe for later optimization or code generation.

## Required review points

Write a review answer that covers:

- where the pass should run in a pipeline and whether it is an analysis pass, a
  verifier-style pass, or a transforming pass;
- how to validate BCIR named metadata catalogs and instruction attachments;
- graph lowering invariants for vertex, edge, and attribute accesses;
- register-binding invariants for logical-to-physical register tables;
- HAM hint invariants for prefetch lowering and non-semantic metadata;
- explicit byte-stride, opaque-pointer, address-space, runtime-wrapper, and safe
  poison/undef/freeze invariants;
- diagnostics that identify the offending function, instruction, stable claim and
  register IDs, and metadata node without making the IR harder to optimize;
- interaction with MLIR conversion diagnostics and deterministic New PM fenceposts.

## Expected observation

A strong answer separates semantic checks from optimization, rejects malformed
metadata or unsafe lowering, and explains which facts should remain metadata
rather than executable behavior.

## Verification command

Markdown review exercise; no LLVM assembler command is required. The checked-in
reference answer is:

```sh
cat llvm-training/exercises/038-custom-pass-bcir-invariants.solution.md
```
