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

The script checks that the master module assembles, verifies, disassembles, and contains required BCIR features.
