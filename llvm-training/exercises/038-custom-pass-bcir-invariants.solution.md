# Solution 038: Custom pass for BCIR invariants

The pass should be a verifier-style module pass that runs after BCIR-to-LLVM
lowering and before aggressive optimization. It should not rewrite IR. Its job is
to report precise errors when lowered IR and BCIR metadata disagree, while still
leaving ordinary LLVM optimizations free to transform proven-safe code.

The pass should first validate named metadata catalogs such as graph schemas,
register-binding tables, and HAM domains. Each catalog entry should have a stable
shape, expected string tags, integer widths that match the ABI contract, and no
references to missing functions or globals. Instruction attachments should refer
to catalog entries by a clear identifier or by a metadata node whose operands can
be checked locally.

For graph lowering, the pass should check that vertex and edge accesses use
consistent element sizes, index widths, and bounds assumptions. It can verify
that edge source, edge destination, and attribute metadata identify the same graph
catalog entry, but it should not require a specific `getelementptr` spelling once
normalization passes have run. Diagnostics should point to the load, store, or
call that carries the inconsistent attachment.

For register bindings, the pass should ensure logical register identifiers map to
one physical slot in the active binding table, that table element types match the
lowered load/store width, and that calls crossing the runtime ABI do not rely on
unmaterialized logical registers. Missing, duplicate, or type-incompatible
bindings should be hard errors.

For HAM hints, the executable operation is the lowered intrinsic or runtime call,
such as `llvm.prefetch`. The original BCIR HAM domain is non-semantic metadata.
The pass should verify valid locality/cache operands and well-formed domain
metadata, but it should not make optimization legality depend on preserving the
hint.

Error messages should include the function name, instruction opcode or printed
instruction, metadata kind, and failing invariant. The pass should be deterministic
and side-effect free so it can be placed in CI alongside `llvm-as` and
`opt -passes=verify`.
