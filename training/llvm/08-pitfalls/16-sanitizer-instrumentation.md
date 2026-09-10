# Pitfall 16 — Sanitizer Instrumentation Is Not Dead Code

## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| Training-only exemplar; no affected BCIR `.ll` file recorded | Training-only / preventive | `opt -passes=verify <sanitized-bcir-output>.ll -o /dev/null` plus an execution test under the intended sanitizer runtime | Preserve sanitizer-inserted checks, shadow-memory traffic, redzone checks, and `!nosanitize` metadata unless a sanitizer-aware pass proves they are obsolete. | [`04-memory/02-load-store.md`](../04-memory/02-load-store.md); [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md); [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |

## The suspicious-looking IR

Sanitizer output often looks repetitive or redundant to a general IR cleanup
agent:

```llvm
%addr = ptrtoint ptr %p to i64
%shadow_idx = lshr i64 %addr, 3
%shadow_addr = add i64 %shadow_idx, %shadow_base
%shadow_ptr = inttoptr i64 %shadow_addr to ptr
%shadow = load i8, ptr %shadow_ptr, align 1, !nosanitize !0
%poisoned = icmp ne i8 %shadow, 0
br i1 %poisoned, label %slowpath, label %ok
```

A naive pass may ask: "The program already loads from `%p`; why is there an
extra load from a derived address?" That extra load is the sanitizer check. It
consults a runtime-maintained shadow memory byte that describes whether the real
program address is currently valid.

## Common sanitizer IR patterns

### Instrumentation calls

Sanitizers add calls to runtime helper functions. Exact names and signatures vary
by sanitizer, target, optimization level, and Clang/LLVM version, but common
shapes include:

- `@__asan_report_loadN` / `@__asan_report_storeN` on AddressSanitizer failure
  paths;
- `@__asan_loadN` / `@__asan_storeN` callback-style checks in some modes;
- `@__ubsan_handle_*` calls for UndefinedBehaviorSanitizer diagnostics;
- `@__tsan_read*`, `@__tsan_write*`, function-entry, and function-exit hooks for
  ThreadSanitizer;
- `@__msan_*` helpers for MemorySanitizer shadow/origin propagation.

These calls may appear in blocks that are rarely executed, branch only from a
check, and end in `unreachable` or a trap. That is intentional: the slow path
reports the bug once the fast path has proven a memory access is unsafe.

### Shadow-memory loads and stores

AddressSanitizer and MemorySanitizer maintain shadow state separate from the
program's ordinary memory. ASan commonly derives a shadow address from a real
address by shifting the real address and adding a shadow base. MSan propagates
shadow values alongside normal values.

To an optimizer that does not model sanitizer state, these operations can look
like unrelated memory traffic:

```llvm
%real = ptrtoint ptr %p to i64
%shadow_index = lshr i64 %real, 3
%shadow_addr = add i64 %shadow_index, %shadow_base
%shadow_ptr = inttoptr i64 %shadow_addr to ptr
%shadow = load i8, ptr %shadow_ptr, align 1, !nosanitize !0
```

Do not remove, fold, or reorder this traffic across the program access unless
your transform is sanitizer-aware and preserves the sanitizer's memory model.
Shadow bytes are updated by allocator hooks, stack poisoning/unpoisoning, global
registration, lifetime intrinsics, and sanitizer runtime calls that may not look
connected to the checked load or store in a local basic-block view.

### Redzone checks

Sanitizers often surround objects with poisoned redzones. A check against shadow
memory answers questions such as:

- is this address inside a poisoned heap/stack/global redzone?
- does a multi-byte access cross from an addressable region into a redzone?
- has a stack slot been poisoned after scope exit or function return?

A fast ASan check can branch when the shadow byte is nonzero, then a slow path
computes the access offset within the 8-byte granule to determine whether the
specific access overlaps a poisoned byte. That slow-path arithmetic may look
redundant because it recomputes facts derived from the same pointer, but it is
checking partial-granule legality rather than the original program value.

### `!nosanitize` metadata

Instrumentation itself is usually marked so sanitizer passes do not recursively
instrument their own checks:

```llvm
%shadow = load i8, ptr %shadow_ptr, align 1, !nosanitize !0
```

`!nosanitize` means "do not instrument this instruction with sanitizer checks."
It does **not** mean "this instruction is unimportant." Dropping the metadata can
cause a later sanitizer pass to instrument shadow-memory accesses, producing
recursion, huge IR growth, false reports, or runtime crashes. Moving metadata to
the wrong instruction can also accidentally suppress checks on real program
accesses.

## Why agents should not delete it

Sanitizer instrumentation is a semantic contract with the sanitizer runtime, not
ordinary dead code. It may look redundant because:

1. the checked program load/store still appears after the check;
2. shadow-memory addresses are derived with integer arithmetic instead of normal
   source-level pointers;
3. slow paths are cold and often end in reporting calls plus `unreachable`;
4. `!nosanitize` can be mistaken for a hint that an instruction is safe to drop;
5. runtime state is updated outside the local function, so local dataflow does
   not show who writes the shadow bytes.

Only delete sanitizer code when one of these is true:

- you are intentionally producing an unsanitized artifact and remove the whole
  sanitizer instrumentation/runtime interface consistently;
- a sanitizer-aware LLVM pass proves a specific check redundant under the same
  rules used by the sanitizer implementation;
- the code is unreachable after normal CFG simplification and the removal does
  not alter sanitizer-visible side effects;
- you regenerate the module from an unsanitized source/configuration instead of
  hand-editing partially sanitized IR.

Never delete just the report block, just the shadow load, or just the
`!nosanitize` metadata because it "does not affect the returned value." It
affects whether memory bugs are detected and whether later sanitizer passes stay
well behaved.

## Minimal verifier-safe example

See [`examples/sanitizer-instrumentation.ll`](examples/sanitizer-instrumentation.ll)
for a small ASan-like load check. The example is intentionally runtime-agnostic:
it declares the report helper and shadow-base global but does not require the
actual ASan runtime to link.

Important features to notice:

- shadow-memory arithmetic is derived from the program pointer;
- shadow loads carry `!nosanitize` metadata;
- a nonzero shadow byte branches to a slow redzone/partial-granule check;
- the report call is on the failure path and the real program load remains in
  the success block.

## BCIR-relevant note

BCIR pipelines may ingest compiler-generated IR, lifted binary stubs, or mixed
modules where only some functions were built with sanitizers. Treat sanitizer
artifacts as provenance-bearing instrumentation. If a correspondence pass strips
or normalizes memory operations, teach it to recognize sanitizer helper calls,
shadow-memory accesses, redzone poisoning, and `!nosanitize` attachments before
it decides what is semantically ignorable.

For test fixtures, prefer one of two clean modes:

1. keep sanitized IR intact and link/run with the matching sanitizer runtime; or
2. regenerate unsanitized IR from the source configuration.

Avoid a half-sanitized file where checks were removed but sanitizer globals,
constructor hooks, attributes, or metadata remain.

## See also

- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — load/store syntax and memory-access review
- [`../06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md) — instruction metadata attachments
- [`../07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) — why generic cleanup passes need semantic preconditions
