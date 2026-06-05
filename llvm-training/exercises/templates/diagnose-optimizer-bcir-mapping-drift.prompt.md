# Agent template: diagnose optimizer-induced BCIR mapping drift

## Role

You are comparing pre-optimization and post-optimization LLVM IR to determine
whether an optimizer changed, erased, or conflated BCIR mapping facts. The goal is
to separate harmless canonicalization from mapping drift that breaks binary
correspondence.

## Inputs to fill in

- **Pre-transform IR**: `<paste or link>`
- **Post-transform IR**: `<paste or link>`
- **Pass pipeline**: `<for example -passes='mem2reg,simplifycfg,instcombine'>`
- **Required BCIR correspondence**: `<registers, graph nodes, memory accesses, metadata>`

## Diagnostic procedure

1. Build a table of BCIR register or graph IDs before the transform.
2. Build the same table after the transform using SSA names, debug info, custom
   metadata, and named metadata side tables.
3. Mark each fact as preserved, renamed, folded with replacement evidence,
   dropped safely, or drifted.
4. Identify the first pass likely responsible for the drift.
5. Propose one repair:
   - freeze or materialize poison-sensitive values before branch/control use;
   - preserve metadata through the pass;
   - add named metadata side tables;
   - split a canonicalized value back into reviewable BCIR values;
   - weaken unsound alignment or address-space assumptions.

## Required output

- A drift report with `no drift`, `benign drift`, or `unsafe drift` verdict.
- The minimal IR snippet or metadata record that demonstrates the drift.
- A verifier or FileCheck-style assertion that would catch the issue in the
  future.

## Verification checklist

- Both snapshots assemble, or any non-assembling snapshot is clearly labeled as a
  negative fixture.
- `opt -passes=verify` succeeds for snapshots that are intended to pass syntax
  and verifier checks.
- The diagnosis does not rely only on SSA names if metadata or structural facts
  provide a more stable correspondence key.
