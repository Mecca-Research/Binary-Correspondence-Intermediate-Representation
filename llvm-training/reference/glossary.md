# Glossary

| Term | Meaning |
|---|---|
| **`@name`** | Global identifier — module-scope; survives across functions and modules |
| **`%name`** | Local identifier — function-scope; an SSA value or basic-block label |
| **`!N`, `!name`** | Metadata identifier |
| **`#N`** | Attribute group identifier (e.g., `#0`) |
| **`$name`** | Comdat identifier (Windows linking) |
| **`label`** | Type of basic-block references; never written as a value type by user code |
| **`token`** | Opaque control type; used by exception-handling and coroutine intrinsics |
| **`metadata`** | Type of metadata operands when passed to intrinsics |
| **Basic block (BB)** | Straight-line sequence of instructions ending in one terminator |
| **Terminator** | Instruction that ends a basic block: `ret`, `br`, `switch`, `indirectbr`, `invoke`, `callbr`, `resume`, `catchswitch`, `catchret`, `cleanupret`, `unreachable` |
| **Phi node** | SSA construct merging incoming values from predecessor blocks |
| **SSA** | Static Single Assignment — every value defined exactly once |
| **Module** | Top-level container; the unit produced by `llvm-as` from one `.ll` file |
| **Function** | Callable entity; contains basic blocks; named with `@` |
| **Global** | Module-scope variable or constant |
| **Constant** | Either: a `constant` global (immutable) or an inline literal in operand position |
| **Constant expression** | Compile-time computation in operand position, parenthesized: `getelementptr (...)`, `bitcast (...)`, etc. |
| **`alloca`** | Stack allocation; lives until function returns |
| **Opaque pointer** | `ptr` — pointer with no pointee type baked in; LLVM ≥ 15 default |
| **Address space** | Numeric tag on pointers identifying which memory region (default 0; others target-defined) |
| **`addrspacecast`** | Convert pointer between address spaces (not a bitcast) |
| **GEP** | `getelementptr` — pointer arithmetic that respects type layout |
| **`inbounds`** | GEP modifier asserting the result stays within the allocated object (out-of-bounds is UB) |
| **`align N`** | Alignment guarantee (or requirement) of N bytes |
| **`volatile`** | Modifier on load/store/atomic that preserves observable access behavior; useful for MMIO, but not a thread-safety mechanism by itself |
| **Atomic operation** | Memory operation with atomicity and an explicit ordering, such as `load atomic`, `store atomic`, `cmpxchg`, `atomicrmw`, or `fence` |
| **Atomic ordering** | Constraint on atomic synchronization: not atomic < `unordered` < `monotonic` < `acquire`/`release` < `acq_rel` < `seq_cst` |
| **`unordered`** | Weak atomic load/store ordering with atomic access but no synchronization edge |
| **`monotonic`** | Relaxed atomic ordering: coherent for the addressed atomic location, but no ordering for other memory |
| **`acquire`** | Atomic ordering that lets a thread consume data published by a matching release operation |
| **`release`** | Atomic ordering that publishes prior memory operations before the release operation |
| **`acq_rel`** | Acquire plus release ordering, valid for read-modify-write operations and fences |
| **`seq_cst`** | Sequentially consistent ordering; strongest common ordering, participating in one global SC order |
| **`cmpxchg`** | Atomic compare-and-exchange; returns `{ old_value, success_bit }` and has separate success and failure orderings |
| **`atomicrmw`** | Atomic read-modify-write operation such as `add`, `xchg`, or `or`; returns the old value |
| **`fence`** | Atomic ordering operation that does not name a memory address |
| **`syncscope("X")`** | Restricts atomic ordering to scope `X` (e.g., `singlethread`) |
| **Linkage** | Controls cross-module visibility/merging: `private`, `internal`, `weak`, `linkonce`, `external`, etc. |
| **Visibility** | `default`, `hidden`, `protected` — controls dynamic linker visibility |
| **`dso_local`** | Symbol known to be in the same DSO — enables faster code |
| **TLS** | Thread-local storage; `thread_local` keyword on globals |
| **Comdat** | Group of globals deduplicated at link time |
| **Fast-math flag** | `nnan`, `ninf`, `nsz`, `arcp`, `contract`, `afn`, `reassoc`, `fast` |
| **`nsw`, `nuw`** | "No signed/unsigned wrap" flag on integer ops; overflow ⇒ poison |
| **`exact`** | Divide/shift flag asserting no remainder bits; otherwise ⇒ poison |
| **Poison** | A value that, when used, may cause UB; safer than `undef` |
| **`undef`** | An any-value; the optimizer may pick freely |
| **`freeze`** | Convert poison/undef to a fixed but arbitrary concrete value |
| **Intrinsic** | Built-in function with name `llvm.<...>`; lowered specially by the backend |
| **Overloaded intrinsic** | Intrinsic whose selected type is encoded in the symbol name, such as `llvm.memcpy.p0.p0.i64` or `llvm.uadd.with.overflow.i64` |
| **Target-specific intrinsic** | Intrinsic in a backend namespace such as `llvm.x86.*`; often requires a target triple and CPU feature |
| **`llvm.memcpy`** | Generic intrinsic for non-overlapping byte copies; use `llvm.memmove` when ranges may overlap |
| **`llvm.memmove`** | Generic intrinsic for overlap-safe byte copies |
| **`llvm.memset`** | Generic intrinsic for filling a memory range with one byte value |
| **Overflow intrinsic** | Intrinsic such as `llvm.uadd.with.overflow.*` or `llvm.sadd.with.overflow.*` returning `{result, overflow_flag}` |
| **Lifetime intrinsic** | `llvm.lifetime.start.*` / `llvm.lifetime.end.*`; optimizer hint for object liveness, not allocation or deallocation |
| **`llvm.prefetch`** | Target-dependent cache prefetch hint; its integer policy operands are `immarg` |
| **`token`** | Opaque control value used by EH, coroutine, convergence, and statepoint-like IR; not a normal load/store-friendly value |
| **`metadata` type** | Special parameter type used by intrinsics such as `llvm.dbg.value` to carry metadata or value-as-metadata operands |
| **`half`** | IEEE 754 binary16 floating-point type |
| **`bfloat`** | 16-bit bfloat floating-point type with float32-like exponent range |
| **`x86_amx`** | X86 target extension type representing AMX tile values for target-specific intrinsics |
| **Scalable vector** | Vector written `<vscale x N x T>` whose lane count is a runtime-dependent multiple of `N` |
| **`immarg`** | Parameter attribute requiring a compile-time constant argument |
| **`opt`** | The LLVM optimization driver — runs passes on IR |
| **`llc`** | The LLVM static compiler — IR to assembly/object |
| **`lli`** | The LLVM interpreter / JIT — runs IR directly |
| **`llvm-as`** | Text IR (`.ll`) to bitcode (`.bc`) |
| **`llvm-dis`** | Bitcode (`.bc`) to text IR (`.ll`) |
| **`llvm-link`** | Merge multiple IR/bitcode files |
| **DSO** | Dynamic Shared Object (shared library) |
| **ODR** | One Definition Rule (C++ term; relevant to `linkonce_odr` etc.) |
| **TBAA** | Type-Based Alias Analysis; conveyed via `!tbaa` metadata |
| **Metadata attachment** | Side information appended to an instruction or function, written like `, !dbg !12` or `, !prof !3` |
| **Named metadata** | Module-scope metadata list such as `!llvm.dbg.cu = !{!0}` |
| **`distinct` metadata** | Metadata node whose identity must be preserved instead of being uniqued with identical nodes |
| **`!dbg`** | Debug-location attachment pointing to source-level debug-info metadata |
| **`DIFile`** | Debug-info node naming a source file and directory |
| **`DICompileUnit`** | Debug-info root for one source compilation; listed in `!llvm.dbg.cu` |
| **`DISubprogram`** | Debug-info node describing a source-level function and scope |
| **`DILocation`** | Debug-info node mapping IR to source line, column, and scope |
| **`DILocalVariable`** | Debug-info node describing a source variable |
| **`!prof`** | Profiling metadata attachment, commonly used for branch weights |
| **Branch weights** | `!prof` metadata tuple such as `!{!"branch_weights", i32 90, i32 10}` that guides hot/cold optimization decisions |
| **`!llvm.loop`** | Loop metadata attachment, usually on the latch terminator, carrying loop transformation hints |
| **`!range`** | Metadata describing allowed value intervals, commonly `[lo, hi)` for integer facts |
| **`!nonnull`** | Metadata asserting a loaded pointer is not null |
| **LangRef** | The LLVM Language Reference Manual: https://llvm.org/docs/LangRef.html — the canonical truth for IR syntax and semantics |
| **Target triple** | `<arch>-<vendor>-<sys>-<env>` string identifying the target platform |
| **Datalayout** | String describing target endianness, type alignments, and pointer sizes per address space |

## Identifier prefix table

| Prefix | Meaning | Scope |
|---|---|---|
| `@` | Global identifier (function, global variable, alias) | Module |
| `%` | Local identifier (SSA value or basic-block label) | Function |
| `!` | Metadata (string, tuple, node, named list, or attachment kind) | Module / instruction |
| `#` | Attribute group ID | Module |
| `$` | Comdat name | Module |

## See also

- [`../INDEX.md`](../INDEX.md) — top-level topic map
- [`../13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md) — common intrinsic signatures and pitfalls
- [`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md) — target intrinsic namespaces and feature requirements
- [`../13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) — special types and tokens
- `10-grammar/llvm-ir.tm` — formal grammar
- LLVM Atomic Instructions and Concurrency Guide — https://llvm.org/docs/Atomics.html
- LangRef atomic ordering — https://llvm.org/docs/LangRef.html#atomic-memory-ordering-constraints
- LangRef atomic instructions — https://llvm.org/docs/LangRef.html#memory-access-and-addressing-operations
- LangRef metadata — https://llvm.org/docs/LangRef.html#metadata
- Source Level Debugging with LLVM — https://llvm.org/docs/SourceLevelDebugging.html
