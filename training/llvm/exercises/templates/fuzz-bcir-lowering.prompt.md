# Agent template: fuzz BCIR lowering

## Role

You are designing a reproducible, invariant-aware fuzzer for a BCIR-to-LLVM
lowering or LLVM pass pipeline. Generate verifier-valid seeds where possible and
use BCIR semantic oracles in addition to LLVM tools.

## Inputs to fill in

- **BCIR generator/model**: `<operations, types, register IDs, revision>`
- **Lowering entry point**: `<command or API>`
- **Pass pipeline**: `<exact ordered pipeline>`
- **Targets**: `<portable baseline plus gated target triples/features>`
- **Oracle set**: `<LLVM verify, BCIR mapping, execution, metadata, ABI, effects>`
- **Budget and seed range**: `<iterations, timeout, seeds>`

## Corpus dimensions

Cover poison/freeze, metadata replacement, address spaces, operand bundles,
debug provenance, 1:1 BCIR mapping, target intrinsics, ABI attributes,
`memory(...)` effects, and varargs ABI behavior. Mutate declaration, definition,
and call sites together when testing cross-boundary contracts.

## Reproducer contract

Every failure must retain generator revision, random seed, LLVM version, target
triple/features, data layout, pipeline, expected fixture class, stable BCIR IDs,
required metadata kinds, and first failing stage. Reducers must not delete these
facts or semantic operand bundles merely to make the failure disappear.

## Fuzz procedure

1. Generate BCIR plus an explicit invariant ledger.
2. Lower and run `llvm-as`/`opt -passes=verify` as the structural oracle.
3. Run the BCIR correspondence and semantic oracles.
4. Bisect pass prefixes when post-lowering behavior diverges.
5. Minimize while pinning reproducer metadata and required semantic evidence.
6. Promote stable failures into the appropriate adversarial class; keep
   target-specific fixtures outside portable smoke tests.

## Required output

- Generator and oracle design.
- Machine-readable reproducer header schema.
- Failure classification and first failing stage.
- Minimized IR plus its BCIR invariant ledger.
- Regression-test proposal, including target gates where needed.

## Verification checklist

- The campaign can be replayed from a recorded seed and generator revision.
- LLVM acceptance is not the only oracle.
- Portable and target-specific corpora are separated.
- Reproducer reduction preserves metadata, bundles, ABI facts, and BCIR IDs.
- Timeouts, crashes, invalid IR, semantic drift, and metadata loss have distinct
  classifications rather than one undifferentiated failure bucket.
