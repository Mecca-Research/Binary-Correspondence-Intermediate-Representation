# Quick Reference Cheat Sheets

One-page reminders for agents that already know the main lessons and need fast,
low-context lookup before editing LLVM IR or BCIR-facing lowering code.

## Available sheets

| Topic | Use when |
| --- | --- |
| [`opaque-pointers.md`](opaque-pointers.md) | Migrating typed-pointer IR/API assumptions to LLVM opaque pointers. |
| [`bcir-lowering.md`](bcir-lowering.md) | Lowering BCIR graph/resource/runtime concepts into verifiable LLVM IR. |
| [`vectorization.md`](vectorization.md) | Reviewing loop/SLP vectorization legality, metadata, and diagnostics. |
| [`metadata.md`](metadata.md) | Preserving debug, profile, loop, TBAA, and BCIR diagnostic metadata. |
| [`new-pass-manager.md`](new-pass-manager.md) | Building and debugging `opt -passes=...` pipelines. |
| [`advanced-ir.md`](advanced-ir.md) | Reviewing intrinsics, attributes, poison/freeze, special types, and fast-math contracts. |
| [`mlir-bridge.md`](mlir-bridge.md) | Navigating MLIR dialect, LLVM-dialect, and BCIR graph-lowering review paths. |

Each sheet links back to chapter material for deeper context.
