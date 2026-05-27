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
Run:

```bash
runtime/llvm/validate_llvm_seed.sh
runtime/llvm/validate_phase2.sh
runtime/llvm/validate_phase3.sh
```

If LLVM tools are unavailable, validation is pending (do not claim success).
