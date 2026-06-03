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
| **`volatile`** | Modifier on load/store/atomic: don't optimize or reorder; for MMIO |
| **Atomic ordering** | `unordered` < `monotonic` < `acquire`/`release` < `acq_rel` < `seq_cst` |
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
| **LangRef** | The LLVM Language Reference Manual: https://llvm.org/docs/LangRef.html — the canonical truth for IR syntax and semantics |
| **Target triple** | `<arch>-<vendor>-<sys>-<env>` string identifying the target platform |
| **Datalayout** | String describing target endianness, type alignments, and pointer sizes per address space |

## Identifier prefix table

| Prefix | Meaning | Scope |
|---|---|---|
| `@` | Global identifier (function, global variable, alias) | Module |
| `%` | Local identifier (SSA value or basic-block label) | Function |
| `!` | Metadata (string, tuple, node, or named list) | Module |
| `#` | Attribute group ID | Module |
| `$` | Comdat name | Module |

## See also

- [`../INDEX.md`](../INDEX.md) — top-level topic map
- `10-grammar/llvm-ir.tm` — formal grammar
- LangRef — https://llvm.org/docs/LangRef.html
