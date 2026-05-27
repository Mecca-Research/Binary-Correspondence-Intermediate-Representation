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
- `bcir_ops.ll`
- `bcir_phase_epoch.ll`
- `bcir_gem_seed.ll`
- `bcir_worklist.ll`
- `bcir_kbcost.ll`
- `bcir_examples.ll`

Additional phase-2 seeds:
- `bcir_registry_schema.ll`
- `bcir_claim_verify.ll`
- `bcir_phase_worklist.ll`

Legacy runtime-hook seed moved to:
- `legacy/bcir_master_reference_rt_hooks.ll`
# BCIR LLVM-First Seed (Version 3.1)

This directory is the BCIR semantic source of truth for the LLVM-first seed.

## Rules
- BCIR operations are represented with legal LLVM IR function calls in the `@bcir.op.*` namespace.
- Do not move semantic ownership to C++ for this phase.
- `runtime/llvm/bcir_master_reference_v2.ll` is the first validation gate and is self-contained.

## Core files
1. `bcir_claim_schema.ll`
2. `bcir_claim_accessors.ll`
3. `bcir_ops.ll`
4. `bcir_phase_epoch.ll`
5. `bcir_gem_seed.ll`
6. `bcir_worklist.ll`
7. `bcir_kbcost.ll`
8. `bcir_examples.ll`
9. `bcir_master_reference_v2.ll`

## Validation
Run:

```bash
runtime/llvm/validate_llvm_seed.sh
```

If LLVM tools are unavailable, validation is pending (do not claim success).
The script checks that the master module assembles, verifies, disassembles, and contains required BCIR features.
