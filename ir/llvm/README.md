# BCIR LLVM backend

Legal LLVM IR emission for BCIR. This section owns:

- `src/llvm_emit.cpp` — epoch/registry/hazard verifiers, schedule builder, and a
  textual LLVM IR emitter.
- `src/lowering.cpp` — macro expansion, MAP-surface lowering, ROP stream <->
  graph reconstruction, and the ROP→LLVM dispatch tables (with x86/ARM/GPU/WASM
  extension hooks).
- `src/bcir_llvm_ir.cpp` — the LLVM **ABI substrate** module (legal LLVM IR only)
  and the staged build-task list.

## Removed: the hand-written `.ll` seed

The previous `runtime/llvm/*.ll` seed (declarations-only master reference plus
~30 hand-written modules and `validate_*.sh` scripts) has been **removed**. It
was an early, basic substrate that duplicated the lane/opcode/claim definitions
already owned by `../core/include/bcir/bcir_ir.hpp`, and it carried schema drift
(`BcirClaimV2` metadata vs. the C++ `BcirClaimV1`).

The forward path replaces it with two complementary producers:

1. this C++ emitter (textual today, LLVM-API later), and
2. the compiled `../mlir/` conversion to the LLVM dialect.

## Non-regression rules

- Emit standard LLVM constructs only — never invented pseudo-instructions.
- Atomics lower to `atomicrmw` / `cmpxchg`; barriers to `fence` or ABI hooks.
- Never rewrite atomics into load/op/store pseudo-atomic sequences.
