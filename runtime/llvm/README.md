# BCIR LLVM-First Seed

`runtime/llvm/*.ll` is the semantic source of truth for this phase.

## Field semantics
`%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }`

The first `i64` is conceptually **control**:
- bits 0..7: opcode
- bits 8..15: lane
- bits 16..27: phase
- bits 28..39: epoch
- bits 40..47: flags
- bits 48..63: stride_code

Full stride encoding goes in `imm[0]` when `stride_code` is insufficient.

## Core modules
- `bcir_master_reference_v2.ll` (self-contained first gate)
- `bcir_claim_schema.ll`
- `bcir_claim_accessors.ll`
- `bcir_registry_schema.ll`
- `bcir_ops.ll`
- `bcir_phase_epoch.ll`
- `bcir_claim_verify.ll`
- `bcir_gem_seed.ll`
- `bcir_worklist.ll`
- `bcir_phase_worklist.ll`
- `bcir_kbcost.ll`
- `bcir_schedule_schema.ll`
- `bcir_schedule_accessors.ll`
- `bcir_prefetch_profiles.ll`
- `bcir_lane_classifier.ll`
- `bcir_batch_executor.ll`
- `bcir_batch_verify.ll`
- `bcir_stream_pack.ll`
- `bcir_examples_phase3.ll`
- `bcir_examples.ll`
- `bcir_examples_worklist.ll`

Legacy runtime-hook seed moved to:
- `legacy/bcir_master_reference_rt_hooks.ll`

## Validation

The runtime validation scripts are fail-fast and print the tool paths, first
version line, repository root, build directory, and concrete validation triple
before running LLVM commands. The concrete triple is intentionally
`x86_64-unknown-linux-gnu`: newer LLVM `opt` builds can reject interface-only
modules that use `unknown-unknown-unknown` with an empty data layout because the
verifier cannot infer target layout facts.

Run:

```bash
runtime/llvm/validate_llvm_seed.sh
runtime/llvm/validate_phase2.sh
runtime/llvm/validate_phase3.sh
runtime/llvm/validate_phase4.sh
```

For environment skew triage without creating build outputs, run:

```bash
runtime/llvm/diagnose_validate_env.sh
```

The diagnostic script reuses `validate_common.sh`, prints the repository root,
build directory, `BCIR_VALIDATE_TRIPLE`, `PATH`, OS/kernel summary, LLVM tool
locations and first `--version` lines, the checked-in `tools/bcir-as/bcir-as`
location/version banner, and read-only prerequisite checks for the seed and
phase-4 validation inputs.

`validate_phase4.sh` additionally requires the checked-in `tools/bcir-as/bcir-as`
executable because it generates `build/vector_add.generated.ll` before linking
the phase-4 runtime module.

If LLVM tools are unavailable, validation is pending (do not claim success). If a
script fails, copy the `[validate]` command line immediately before the error; it
is the reproducible local command CI attempted to run.
