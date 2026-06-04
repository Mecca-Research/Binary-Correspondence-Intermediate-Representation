# Index: By intrinsic / special type

| Name | Means | See |
|---|---|---|
| `llvm.memcpy.*` | Non-overlapping memory copy intrinsic | [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md), [`reference/intrinsics.md`](../reference/intrinsics.md) |
| `llvm.memmove.*` | Overlap-safe memory copy intrinsic | [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md), [`reference/intrinsics.md`](../reference/intrinsics.md) |
| `llvm.memset.*` | Byte-fill memory intrinsic | [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md), [`reference/intrinsics.md`](../reference/intrinsics.md) |
| `llvm.uadd.with.overflow.*` | Unsigned checked addition; returns `{T, i1}` | [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md), [`13-advanced-ir/examples/overflow-intrinsic.ll`](../13-advanced-ir/examples/overflow-intrinsic.ll) |
| `llvm.sadd.with.overflow.*` | Signed checked addition; returns `{T, i1}` | [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md), [`13-advanced-ir/examples/overflow-intrinsic.ll`](../13-advanced-ir/examples/overflow-intrinsic.ll) |
| `llvm.lifetime.start.*`, `llvm.lifetime.end.*` | Lifetime markers for optimizer-visible object liveness | [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md), [`13-advanced-ir/examples/memcpy.ll`](../13-advanced-ir/examples/memcpy.ll) |
| `llvm.prefetch` | Target-dependent cache prefetch hint | [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md), [`08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md) |
| `llvm.x86.*` | X86-specific intrinsic namespace | [`13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md) |
| `llvm.aarch64.*`, `llvm.arm.*`, `llvm.riscv.*`, `llvm.amdgcn.*`, `llvm.nvvm.*` | Other target-specific intrinsic namespaces | [`13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md) |
| `token` | Opaque control value used by advanced intrinsic families | [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md), [`13-advanced-ir/examples/token-outline.ll`](../13-advanced-ir/examples/token-outline.ll) |
| `metadata` | Special operand type for debug/analysis intrinsics | [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md), [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md) |
| `half`, `bfloat`, `x86_amx`, `<vscale x N x T>` | Special scalar/target/vector types with portability constraints | [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
